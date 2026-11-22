"""Lag-1 overlap: driver-side pure state helpers, rules O5-O8 and the alarm -> lag-0 rule (no threads, no I/O).

A run switches ``policy.rollout_lag`` to 1 at a block boundary; **any overlap alarm switches it back to 0 at the
next block boundary**; the switch is recorded in the run state and checked on resume.

The training driver (not part of this package) keeps one JSON-serialisable dict, ``state['pipeline']``, built by
:func:`new_state`, and owns everything with threads, coordinators, servers and containers. This module only reads
and writes that dict:

* rounds: :func:`record_start` (a background round started), :func:`record_accept` (learned),
  :func:`record_discard` (aborted / dropped, reason ``stop`` | ``budget`` | ``resume`` | ``rollback`` | ``error``),
  :func:`on_resume` (an in-flight round found on resume is discarded), :func:`protect_paths` (files the pruner must
  keep while a round is in flight);
* rules: :func:`o5_check` (assertion), :func:`o6_alarm`, :func:`o7_alarm`, :func:`gpu_fit` (O8);
* the lag switch: :func:`observe_update` records each learned update's lag and alarms (the learner's ``is/alarms``
  plus O6 / O7) and arms a revert; :func:`revert_action` tells the driver at the next block start to apply
  ``policy.rollout_lag = 0`` through a recorded config change; :func:`record_lag_switch` writes the switch;
  :func:`check_lag_on_resume` refuses a run whose config lag disagrees with the last recorded switch.

A lag-0 run never builds this state (``state.get('pipeline')`` is None), so nothing here touches the default path.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

PIPELINE_SCHEMA = "ppo-v7-pipeline.1"
MAX_ROUNDS = 200              # round records kept
WINDOW = 10                   # O6 / O7: the last WINDOW learned updates of the current lag-1 stretch (example value)
O6_MIN_LAG1_SHARE = 0.25      # O6: lag-1 updates below this share of the window -> alarm (example value)
O7_MAX_DISCARDS = 2           # O7: more discarded rounds than this in the window -> alarm (example value)
GPU_LIMIT_GB = 16.0           # O8: GPU memory budget in GB (example value)
DISCARD_REASONS = ("stop", "budget", "resume", "rollback", "error", "monitor_stop", "join_timeout")
REVERT_ACTION = "set_rollout_lag"


MAX_LAG = 2                   # policy.rollout_lag = 2 lets a second round run ahead (state key 'inflight2')


class PipelineError(AssertionError):
    """O5: a lag outside {0, 1, 2}, or a round started while as many as allowed are in flight (the driver maps it
    to its own block error)."""


# =========================================================================== state
def new_state() -> Dict[str, Any]:
    return {"schema": PIPELINE_SCHEMA, "inflight": None, "rounds": [], "updates": [], "lag_switches": [],
            "alarms": [], "revert": None,
            "counts": {"fills": 0, "lag1": 0, "discards": 0, "unplanned_fills": 0, "started": 0}}


def _check(st: Mapping[str, Any]) -> None:
    if not isinstance(st, Mapping) or st.get("schema") != PIPELINE_SCHEMA:
        raise ValueError(f"not a {PIPELINE_SCHEMA} pipeline state")


def _push(lst: List[Any], rec: Any, keep: int = MAX_ROUNDS) -> None:
    lst.append(rec)
    del lst[:-keep]


def record_start(st: MutableMapping[str, Any], *, update: int, rollout_id: str, acting_update: int,
                 acting_digest: str, acting_checkpoint: Optional[str] = None, acting_sha256: Optional[str] = None,
                 config_digest: Optional[str] = None, games: int = 0, pins: Optional[Mapping[str, str]] = None,
                 pin_paths: Sequence[str] = (), extra: Optional[Mapping[str, Any]] = None,
                 max_inflight: int = 1) -> Dict[str, Any]:
    """A background round for ``update`` started with the acting weights of ``acting_update``. O5: refuses while
    another round is in flight. ``max_inflight = 2`` (a ``rollout_lag = 2`` run): a second round may start
    while one is in flight; it is kept as ``st['inflight2']`` (always the round after ``st['inflight']``) and
    moves up when the older one is accepted or discarded."""
    _check(st)
    first = st.get("inflight")
    if first is not None and int(max_inflight) >= 2 and st.get("inflight2") is None:
        if int(update) != int(first["update"]) + 1:
            raise PipelineError(f"O5: round {update} started ahead of round {first['update']} in flight "
                                f"(only the next round may)")
        slot = "inflight2"
    else:
        o5_check(None, first if st.get("inflight2") is None else st.get("inflight2"), starting=True)
        slot = "inflight"
    rec = {"update": int(update), "rollout_id": str(rollout_id), "acting_update": int(acting_update),
           "acting_digest": str(acting_digest), "acting_checkpoint": acting_checkpoint,
           "acting_sha256": acting_sha256, "config_digest": config_digest, "games": int(games),
           "pins": dict(pins or {}), "pin_paths": [str(p) for p in pin_paths], "lag": None,
           "learner_digest": None, "discarded": None}
    if extra:
        rec.update(copy.deepcopy(dict(extra)))
    st[slot] = rec
    st["counts"]["started"] += 1
    return rec


def inflight_all(st: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The rounds in flight, oldest first (at most two; the second only in a ``rollout_lag = 2`` run)."""
    if not st:
        return []
    return [r for r in (st.get("inflight"), st.get("inflight2")) if r is not None]


def _promote(st: MutableMapping[str, Any]) -> None:
    """The older round left the slot: a round started ahead (``inflight2``) is now the round in flight."""
    st["inflight"] = st.get("inflight2")
    if "inflight2" in st:
        st["inflight2"] = None


def record_accept(st: MutableMapping[str, Any], *, update: int, lag: int, learner_digest: str,
                  max_lag: int = 1) -> Dict[str, Any]:
    """The in-flight round of ``update`` was loaded and learned at ``lag`` (at most ``max_lag``: the run's
    ``policy.rollout_lag``; 1 unless the driver of a lag-2 run says 2)."""
    _check(st)
    o5_check(lag, None, max_lag=max_lag)
    rec = st.get("inflight")
    if rec is None or int(rec["update"]) != int(update):
        raise PipelineError(f"accept of update {update}: in flight is {None if rec is None else rec['update']}")
    rec = dict(rec, lag=int(lag), learner_digest=str(learner_digest))
    _push(st["rounds"], rec)
    _promote(st)
    return rec


def undo_accept(st: MutableMapping[str, Any], *, update: int, reason: str) -> Optional[Dict[str, Any]]:
    """Mark the accept record of ``update`` as undone (the learner took no step on it and the driver re-collects the
    update, e.g. the O3 pre-step stop). Returns the marked record, or None."""
    _check(st)
    for rec in reversed(st.get("rounds") or []):
        if int(rec.get("update", -1)) == int(update) and rec.get("lag") is not None and not rec.get("discarded"):
            rec["undone"] = str(reason)
            return rec
    return None


def record_discard(st: MutableMapping[str, Any], *, reason: str, games: Optional[int] = None,
                   update: Optional[int] = None, unplanned_fill: bool = False,
                   operator: bool = False) -> Optional[Dict[str, Any]]:
    """Discard the in-flight round (or record a discard of ``update`` when none is in flight). ``unplanned_fill``
    (a budget cut that forces a fill) is counted separately. ``operator``: the round was aborted
    because a stop was requested (the run's STOP file, a queue stop, SIGTERM), not because something failed; the
    record says so and :func:`o7_alarm` does not count it."""
    _check(st)
    if reason not in DISCARD_REASONS:
        raise ValueError(f"discard reason {reason!r} not in {DISCARD_REASONS}")
    second = st.get("inflight2")
    if second is not None and (update is None or int(update) == int(second["update"])):
        # with two rounds in flight the round started ahead goes first (no update named), or by its number
        rec = dict(second)
        rec["discarded"] = {"reason": reason, "games": int(rec.get("games", 0) if games is None else games)}
        if operator:
            rec["discarded"]["operator"] = True
        _push(st["rounds"], rec)
        st["inflight2"] = None
        st["counts"]["discards"] += 1
        if unplanned_fill or reason == "budget":
            st["counts"]["unplanned_fills"] += 1
        return rec
    rec = st.get("inflight")
    if rec is None and update is None:
        return None
    if rec is not None and second is not None and update is not None and int(update) != int(rec["update"]):
        rec = None                                    # a third round's record: neither round in flight is touched
    keep_slot = rec is None
    rec = dict(rec) if rec is not None else {"update": int(update), "rollout_id": None, "games": 0}
    rec["discarded"] = {"reason": reason, "games": int(rec.get("games", 0) if games is None else games)}
    if operator:
        rec["discarded"]["operator"] = True
    _push(st["rounds"], rec)
    if not (keep_slot and second is not None):
        _promote(st)
    st["counts"]["discards"] += 1
    if unplanned_fill or reason == "budget":
        st["counts"]["unplanned_fills"] += 1
    return rec


def on_resume(st: MutableMapping[str, Any]) -> Optional[Dict[str, Any]]:
    """An in-flight round found on resume is discarded (reason ``resume``); the next cycle is a fill.
    (One round; a ``rollout_lag = 2`` run uses :func:`on_resume_all`.)"""
    _check(st)
    return record_discard(st, reason="resume") if inflight_all(st) else None


def on_resume_all(st: MutableMapping[str, Any]) -> List[Dict[str, Any]]:
    """Every in-flight round found on resume is discarded (reason ``resume``), the round started ahead
    first; returns the discarded records."""
    _check(st)
    out: List[Dict[str, Any]] = []
    while inflight_all(st):
        rec = record_discard(st, reason="resume")
        if rec is None:                                   # pragma: no cover - record_discard always takes one
            break
        out.append(rec)
    return out


def protect_paths(st: Optional[Mapping[str, Any]]) -> Set[str]:
    """Files the pruner must keep while a round is in flight: its acting checkpoint and its pinned pool snapshots
    for every round in flight. Empty for None (a lag-0 run)."""
    out: Set[str] = set()
    for rec in inflight_all(st):
        out |= {str(p) for p in rec.get("pin_paths") or ()}
        if rec.get("acting_checkpoint"):
            out.add(str(rec["acting_checkpoint"]))
    return out


def counts(st: Mapping[str, Any]) -> Dict[str, int]:
    _check(st)
    return dict(st["counts"])


# =========================================================================== rules O5-O8
def o5_check(lag: Optional[int], inflight: Optional[Mapping[str, Any]], *, starting: bool = False,
             max_lag: int = MAX_LAG) -> None:
    """O5: a lag outside 0 .. ``max_lag`` (never above :data:`MAX_LAG`), or a round started while one is in
    flight -> :class:`PipelineError`."""
    top = min(int(max_lag), MAX_LAG)
    if lag is not None and int(lag) not in range(top + 1):
        raise PipelineError(f"O5: update learned at lag {lag} (only 0 to {top})")
    if starting and inflight is not None:
        raise PipelineError(f"O5: a round started while update {inflight.get('update')} is in flight")


def _stretch(st: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Learned updates since the last switch to lag 1 (the current lag-1 stretch), oldest first."""
    since = None
    for sw in st.get("lag_switches") or ():
        if int(sw["new"]) >= 1 and int(sw.get("old", 0)) < 1:        # 0 -> 1 or 2 (1 <-> 2 keeps the stretch)
            since = int(sw["update"])
    ups = list(st.get("updates") or ())
    return [u for u in ups if since is None or int(u["update"]) >= since]


def o6_alarm(st: Mapping[str, Any]) -> Optional[str]:
    """O6: a lag-1 share below ``O6_MIN_LAG1_SHARE`` among the last ``WINDOW`` learned updates of the current lag-1 stretch;
    evaluated only once the stretch holds ``WINDOW`` updates (a block start, a gate and a resume are fills)."""
    win = _stretch(st)[-WINDOW:]
    if len(win) < WINDOW:
        return None
    n1 = sum(1 for u in win if int(u["lag"]) >= 1)
    if n1 < O6_MIN_LAG1_SHARE * WINDOW:
        return f"O6: {n1}/{WINDOW} lag-1 updates in the last {WINDOW} (< {O6_MIN_LAG1_SHARE:.0%})"
    return None


def o7_alarm(st: Mapping[str, Any]) -> Optional[str]:
    """O7: more than ``O7_MAX_DISCARDS`` discarded rounds of an update in the last ``WINDOW`` learned updates of the
    stretch.

    A round discarded by a requested stop (``discarded['operator']``, :func:`record_discard`) is not counted: the rule
    is for failures, and operator stops (e.g. for a code switch) must not send the run back to lag 0."""
    win = _stretch(st)[-WINDOW:]
    if not win:
        return None
    lo, hi = int(win[0]["update"]), int(win[-1]["update"])
    n = sum(1 for r in st.get("rounds") or ()
            if r.get("discarded") and not r["discarded"].get("operator") and lo <= int(r["update"]) <= hi + 1)
    if n > O7_MAX_DISCARDS:
        return f"O7: {n} discarded rounds in updates {lo}..{hi + 1} (> {O7_MAX_DISCARDS})"
    return None


def gpu_fit(learner_gb: float, drift_gb: float, servers_gb: float, limit: float = GPU_LIMIT_GB
            ) -> Tuple[bool, float]:
    """O8: max(learner peak, drift peak) + servers' peak must not exceed ``limit`` GB -> (ok, total)."""
    vals = [float(learner_gb), float(drift_gb), float(servers_gb)]
    if not all(v == v and v >= 0.0 for v in vals):
        raise ValueError(f"O8 needs finite non-negative peaks, got {vals}")
    total = max(vals[0], vals[1]) + vals[2]
    return total <= float(limit), total


# =========================================================================== the lag switch
def observe_update(st: MutableMapping[str, Any], *, update: int, lag: int, alarms: Iterable[str] = (),
                   cfg_lag: int = 1) -> List[str]:
    """Record one learned update of a run whose config lag is ``cfg_lag``: its lag, the learner's overlap alarms
    (``is/alarms``) and the driver rules O6 / O7. Returns every alarm of this update. Any alarm while the config
    lag is 1 arms the revert (back to lag 0 at the next block boundary); it stays armed until
    :func:`record_lag_switch` writes the switch."""
    _check(st)
    o5_check(lag, None, max_lag=max(1, int(cfg_lag)))
    _push(st["updates"], {"update": int(update), "lag": int(lag)})
    st["counts"]["lag1" if int(lag) >= 1 else "fills"] += 1
    if int(lag) == 2:
        st["counts"]["lag2"] = int(st["counts"].get("lag2", 0)) + 1
    out = [str(a) for a in alarms]
    if int(cfg_lag) >= 1:
        for rule in (o6_alarm, o7_alarm):
            a = rule(st)
            if a:
                out.append(a)
    if out:
        _push(st["alarms"], {"update": int(update), "alarms": list(out)})
        if int(cfg_lag) >= 1 and st.get("revert") is None:
            st["revert"] = {"first_update": int(update), "reasons": list(out)}
        elif st.get("revert") is not None:
            st["revert"]["reasons"] = (list(st["revert"]["reasons"]) + out)[-50:]
    return out


def arm_revert(st: MutableMapping[str, Any], *, update: int, reason: str) -> None:
    """Arm the revert for an alarm the driver detects itself (e.g. a monitor stop named ``overlap_is_truncation``
    or a throughput alarm attributed to overlap); idempotent."""
    _check(st)
    if st.get("revert") is None:
        st["revert"] = {"first_update": int(update), "reasons": [str(reason)]}
    else:
        st["revert"]["reasons"] = (list(st["revert"]["reasons"]) + [str(reason)])[-50:]


def revert_action(st: Optional[Mapping[str, Any]], cfg_lag: int) -> Optional[Dict[str, Any]]:
    """At a block start: the action the driver must apply before the first update, or None. Only when the run's
    config lag is 1 and an alarm armed the revert. The driver applies ``policy.rollout_lag = 0`` as a recorded
    config change and then calls :func:`record_lag_switch` (which clears the revert)."""
    if not st or int(cfg_lag) < 1 or not st.get("revert"):
        return None
    rv = st["revert"]
    return {"action": REVERT_ACTION, "key": "policy.rollout_lag", "value": 0,
            "first_update": int(rv["first_update"]), "reasons": list(rv["reasons"])}


def record_lag_switch(st: MutableMapping[str, Any], *, update: int, old: int, new: int, reason: str,
                      old_digest: Optional[str] = None, new_digest: Optional[str] = None) -> Dict[str, Any]:
    """Write a block-boundary switch of ``policy.rollout_lag`` applying from ``update`` (the first update learned
    under the new config). A switch to 0 clears the revert; a switch to 1 starts a new O6 / O7 stretch. Refuses an
    in-flight round (switches happen only at drained block boundaries)."""
    _check(st)
    o5_check(new, None)
    o5_check(old, None)
    if int(old) == int(new):
        raise ValueError(f"lag switch {old} -> {new} changes nothing")
    if inflight_all(st):
        raise PipelineError("a lag switch needs a drained pipeline (a round is in flight)")
    last = st["lag_switches"][-1] if st["lag_switches"] else None
    if last is not None and int(last["new"]) != int(old):
        raise ValueError(f"lag switch from {old}, but the last recorded switch set {last['new']}")
    if last is not None and int(update) < int(last["update"]):
        raise ValueError(f"lag switch at update {update} before the last one at {last['update']}")
    rec = {"update": int(update), "old": int(old), "new": int(new), "reason": str(reason),
           "old_digest": old_digest, "new_digest": new_digest}
    if int(new) == 0 and st.get("revert"):
        rec["revert"] = copy.deepcopy(st["revert"])
        st["revert"] = None
    st["lag_switches"].append(rec)
    return rec


def current_lag(st: Optional[Mapping[str, Any]], default: int = 0) -> int:
    """The lag the last recorded switch set (``default`` when none was recorded)."""
    if not st or not st.get("lag_switches"):
        return int(default)
    return int(st["lag_switches"][-1]["new"])


def check_lag_on_resume(st: Optional[Mapping[str, Any]], cfg_lag: int, *, fork_lag: int = 0) -> None:
    """Resume check: the config lag the block runs with must equal the last recorded switch (``fork_lag``, the lag
    the run started with, when none is recorded). A mismatch means the config was edited around the record, or a
    switch was applied without being recorded -> ValueError (the driver refuses the block)."""
    want = current_lag(st, fork_lag)
    if int(cfg_lag) != want:
        raise ValueError(f"policy.rollout_lag is {cfg_lag} but the run state records lag {want} "
                         "(switch only through a recorded block-boundary change)")
