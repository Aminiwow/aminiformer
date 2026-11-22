"""Unattended PPO monitors, rollback and stall ladder.

Pure standard library. A :class:`Monitor` is a stateful evaluator over the metric history of one run.
The learner / coordinator feeds it observations and executes the :class:`Action` objects it returns:

* :meth:`Monitor.check_identity`    update-0 identity report             -> log | stop
* :meth:`Monitor.observe_update`    one record per PPO update            -> any action
* :meth:`Monitor.observe_minibatch` finiteness of each minibatch         -> skip_minibatch | stop
* :meth:`Monitor.observe_system`    host samples (memory, swap)          -> reduce_slots | log
* :meth:`Monitor.observe_gate`      screen S / G1 results                -> rollback, restores, stall

Every action kind is one of :data:`ACTION_KINDS`. Learning-rate and beta changes are multiplicative
factors (``Action.value``); temporary ones are tracked separately and undone by one clean G1
(delta_hat >= 0) with an explicit restore action. :attr:`Monitor.lr` and :attr:`Monitor.beta` give the
effective values (beta clamped to [floor, cap]). Every observation is appended as one JSON
line to the dashboard file when ``dashboard_path`` is set; every value (metrics, action details,
numpy / torch scalars, NaN) is converted to plain JSON first, so an alarm can never crash the run. The
state round-trips through :meth:`Monitor.state_dict` / :meth:`Monitor.from_state` so a chained block
continues the same history. A stop is final: :func:`write_stop_marker` leaves
``PPO_STOPPED`` in the run directory and the block chainer never queues or starts another block after it.

``observe_update`` reads the key names of ``amini.ppo.losses.update_metrics`` directly
(``clip_frac``, ``play_clip_frac``, flat ``entropy/<head>``) and ``losses.early_stop``'s tuple of tripped
limits as ``early_stopped``; head names are canonicalised like ``losses.canonical_head``
(:data:`HEAD_ALIASES` is a copy, so this module stays torch-free; a test pins the two tables equal).

Metric coverage (no rule is silently off): :meth:`Monitor.expected_update_keys` is the set of keys the
active rules need on a given update (always-on rules, the audit / drift cadence, and the rules whose
baseline is present). Every expected key that is absent or non-finite raises one ``metrics_missing``
alarm per update (listing the keys; also kept in the dashboard record). With ``strict=True`` the monitor
raises :class:`MetricsMissingError` instead, before any state changes: tests use it, and the learner
switches it on once every metric producer is in place.
``strict`` is part of the saved state, so a chained block restored with :meth:`Monitor.from_state` keeps it
(an explicit ``strict=`` keyword wins). A flag (``early_stopped``, ``drift_alarm``) must hold a valid flag
value (bool, 0 / 1, a tuple / list); a NaN flag is missing, never True. A key and its alias resolve to the
first finite value of the two in one helper, for the coverage check and the rule alike. A partly missing IL
entropy baseline lists each head it switches off (``entropy[<head>]``) in ``baseline_missing``.

The interpretation choices are listed in :data:`MONITORS` (``condition``): the early-stop rule counts the
early-stopped updates of a sliding window (``early_stop_window`` updates, more than ``early_stop_frac`` of
them), "KL flat" means every value of the ``kl_flat_window``-update window lies within ``kl_flat_tol`` of the
window's first value, joint and play-row clip fractions are two separate streaks, there is one active
temporary lr cut per cause (clip, early stop) and a floor on the temporary lr factor, the KL-flat step and
ladder step 1 are the same single KL step, core-card streaks are per card, and the stall ladder takes one
step per non-promoting G1 from the ``stall_start_g1``-th, each >= ``stall_min_updates`` updates apart.
Every number in :class:`Thresholds` (and the ``base_lr`` / ``base_beta`` defaults of :class:`Monitor`) is an
example value; a run passes its own.

KL step at the beta floor: it never raises the lr permanently; it records a ``propose_ab`` action (an
equal-games A/B fork from the backup point, arm B at lr x ``kl_step_lr``) in :attr:`Monitor.proposals`.
After a beta KL step, a clear KL growth with a G1 that did not improve switches later KL steps off and
sends ladder step 1 straight to step 2 (:meth:`Monitor._kl_watch_check`). Every lr / beta factor is
numbered in :attr:`Monitor.scale_log` so the learner can re-derive and read back the effective values
(:meth:`Monitor.planned_since`). :meth:`Monitor._diagnostics` reads the learner's tiny-ratio shares, gate
trends, entropy versus the reference and probe flip rates; their alarm thresholds stay None (logged only)
until a calibration sets them.

Runtime thresholds: calibrated numbers can enter the RUNNING run. A recorded config change of one of the
:data:`RUNTIME_THRESHOLD_FIELDS` (the early-stop window / fraction and the diagnostic alarm thresholds) is
applied with :meth:`Monitor.apply_thresholds`, which swaps the immutable :class:`Thresholds`, rebuilds the
windows whose length changed (keeping the newest entries), refuses any other field and writes a
``thresholds`` dashboard record.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
import numbers
import os
from pathlib import Path
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import warnings

# ---------------------------------------------------------------- vocabulary
ACTION_KINDS: Tuple[str, ...] = (
    "log", "alarm", "rollback", "stop", "lr_scale", "beta_scale",
    "reduce_slots", "skip_minibatch", "enter_f1", "stall_step", "propose_ab",
)
# Severity order for :func:`most_severe` (the caller executes the most severe first). ``propose_ab`` changes
# nothing in the run: it records an equal-games A/B fork proposal for a person.
_SEVERITY = {k: i for i, k in enumerate(
    ("stop", "rollback", "enter_f1", "stall_step", "lr_scale", "beta_scale",
     "reduce_slots", "skip_minibatch", "propose_ab", "alarm", "log"))}

STOP_MARKER = "PPO_STOPPED"          # run-dir file; the block chainer refuses to chain or start after it
ROW_TYPES = ("wait", "play", "ability")
# Canonical head names and aliases: a copy of amini.ppo.losses.HEADS / HEAD_ALIASES (losses needs torch).
IDENTITY_HEADS: Tuple[str, ...] = ("family", "slot", "cells", "ability")
HEAD_ALIASES: Mapping[str, str] = {
    "family": "family", "gate": "family", "kl_family": "family", "ent_family": "family",
    "slot": "slot", "kl_slot": "slot", "ent_slot": "slot",
    "cells": "cells", "cell": "cells", "kl_cell": "cells", "kl_cells": "cells", "ent_cell": "cells",
    "ability": "ability", "kl_ability": "ability", "ent_ability": "ability",
}
TEMP_LR_FLOOR = 0.4                  # default of Thresholds.temp_lr_floor (example value)
STALL_GAMES_PER_UPDATE = 2000        # default of Thresholds.stall_games_per_update (example value)

# Stall ladder. Steps 5 and 6 (new opponents, a restart) need a person; the monitor only signals them.
# Step 2's game count is filled from Thresholds.stall_games_per_update when the step fires.
STALL_LADDER: Tuple[Mapping[str, Any], ...] = (
    {"step": 1, "name": "kl_step", "do": "beta x kl_step_beta (down to the floor); at the floor an equal-games A/B "
                                         "fork proposal (arm B lr x kl_step_lr), never an automatic lr change"},
    {"step": 2, "name": "games_per_update", "do": "more games per update (Thresholds.stall_games_per_update)",
     "games_per_update": STALL_GAMES_PER_UPDATE},
    {"step": 3, "name": "f1", "do": "F1 critic remedy: a longer GAE lambda, a short tower head, more critic epochs"},
    {"step": 4, "name": "frozen_encoder_control", "do": "control arm with the single-step encoder frozen"},
    {"step": 5, "name": "p7_opponents", "do": "harder opponent decks, an exploiter from the latest IL model"},
    {"step": 6, "name": "restart_latest_il", "do": "restart from the latest IL model"},
)


def canonical_head(name: str) -> str:
    """Canonical head name (as ``losses.canonical_head``); unknown names pass through."""
    return HEAD_ALIASES.get(name, name)


class MetricsMissingError(ValueError):
    """Strict mode: an update record lacks an expected metric (or carries a non-finite one)."""

    def __init__(self, update: int, keys: Sequence[str]):
        self.update = update
        self.keys = list(keys)
        super().__init__(f"update {update}: expected metrics absent or non-finite: {', '.join(self.keys)}")


# Keys every update needs (rules without a baseline): numbers, and flags whose value may be False / ().
ALWAYS_NUMERIC_KEYS: Tuple[str, ...] = (
    "refusals_code19", "drop_rate", "clip_frac", "play_clip_frac", "kl_ref_total", "ev_overall", "ev_last_third")
FLAG_KEYS: Tuple[str, ...] = ("early_stopped", "drift_alarm")   # valid flag value (see _flag) is enough
KEY_ALIASES: Mapping[str, str] = {"clip_frac_play": "play_clip_frac"}


@dataclass(frozen=True)
class MonitorSpec:
    """One monitor rule: metric, condition, window, action and (``plan``) the rule group it belongs to."""
    name: str
    metric: str
    condition: str
    window: str
    action: str
    plan: str


MONITORS: Tuple[MonitorSpec, ...] = (
    MonitorSpec("update0_identity", "identity report",
                "max |dlogp| per row type >= tol (fp32 1e-3; bf16 per row type), |r-1| >= 1e-3, "
                "KL(theta0||ref) >= 1e-5 for any of the 4 heads (a missing head fails), or the bitwise "
                "replay fails; absolute values are compared",
                "update 0", "stop", "update 0"),
    MonitorSpec("actor_learner_mismatch", "mu_dlogp_max/<row type>, learner_invalid_rows (learner guard)",
                ">= update-0 tol for any row type; > 0 invalid rows", "every update", "stop",
                "learner guard"),
    MonitorSpec("approx_kl_early_stop", "early_stopped (bool or losses.early_stop tuple; truthy = stopped)",
                "more than 75% of the last 8 updates early-stopped (7 of 8)",
                "8 updates (window cleared after firing)",
                "lr_scale 0.6 temporary (one active cut for this cause)", "per update"),
    MonitorSpec("clip_fraction", "clip_frac, play_clip_frac (alias clip_frac_play)",
                "joint clip fraction > 0.4, or play-row clip fraction > 0.4 (separate streaks)",
                "4 consecutive updates", "lr_scale 0.6 temporary (one active cut for this cause)",
                "per update"),
    MonitorSpec("kl_flat_stall", "kl_ref_total",
                "every value of the window within +-20% of its first value AND >= 4 consecutive G1 "
                "without promotion; once per promotion-free stretch (it is also ladder step 1's KL step)",
                "8 updates", "beta_scale 0.6 (floor 0.005); at the floor propose_ab (equal-games A/B fork from "
                "the backup point, arm B lr x2; no automatic lr change)", "per update"),
    MonitorSpec("kl_step_skip", "kl_ref_total after a beta KL step, the next G1 delta_hat",
                "mean KL after the step >= 1.5x the window mean before it AND the next G1 not promoted and "
                "moved < 1 pp from the last G1 before the step", "the stretch up to the next G1",
                "log; later KL steps are skipped and ladder step 1 goes straight to step 2",
                "stall ladder"),
    MonitorSpec("tiny_ratio", "tiny_ratio_share/play, tiny_ratio_share/wait (learner)",
                "share of rows with log(pi_theta / pi_prox) below the learner's tiny-ratio cut above "
                "tiny_share_max (None: logged only)",
                "1 update", "alarm", "diagnostics"),
    MonitorSpec("gate_trend", "gate_trend/act_rate[_behind]/<kind|all>, gate_trend/adv_gap[_behind]/<kind|all>",
                "two-sided: |relative act-rate drift| or |advantage-gap drift| over the window (OLS slope x "
                "(window - 1)) above trend_act_rel_max / trend_adv_gap_max (None: logged only)", "8 updates",
                "alarm", "diagnostics"),
    MonitorSpec("entropy_vs_reference", "entropy_vs_ref/<head>, cell_entropy_vs_ref/<card>, entropy_vs_il/<head>, "
                "cell_entropy_vs_il/<card> (learner)",
                "< 0.4x the KL reference on the same rows (or the IL model on the fixed probe rows), or above "
                "entropy_ref_max_ratio (None: no upper bound)", "1 update", "alarm", "diagnostics"),
    MonitorSpec("argmax_flip", "probe_flip/slot, probe_flip/cell (learner, fixed probe rows)",
                "flip rate above flip_max (None: logged only)", "1 update", "alarm", "diagnostics"),
    MonitorSpec("effective_values", "learner read-back of the optimiser lr and the KL term's beta",
                "differs from the planned value (config x monitor factors x action factors)", "every update",
                "stop", "learner guard"),
    MonitorSpec("kl_grad_ratio", "kl_grad_ratio/total (learner, first minibatch)",
                "beta > 0, theta != reference and ratio <= 1e-12", "every update", "stop", "learner guard"),
    MonitorSpec("entropy", "entropy/<head> (or entropy[head])", "< 0.4x IL", "1 update", "alarm", "per update"),
    MonitorSpec("plays_per_min", "plays_per_min, abilities_per_min", "outside [0.8, 1.25]x IL", "1 update",
                "alarm", "per update"),
    MonitorSpec("card_rate", "card_rate[card]", "outside [0.25, 4]x IL", "1 update", "alarm", "per update"),
    MonitorSpec("core_card_collapse", "card_rate[core card]", "one core card < 0.2x IL (per-card streak)",
                "4 consecutive updates",
                "rollback (+ lr 0.6 temporary) and flag; third occurrence -> stop", "per update"),
    MonitorSpec("mask_illegal_cell", "refusals_code19", "> 0", "1 update",
                "stop (an alarm below code19_stop_per_update / code19_stop_consecutive; the defaults 1, 1 "
                "stop at once)", "per update"),
    MonitorSpec("mask_legal_refused", "masked_legal_refused_rate", "> 3x IL (IL 0: any rate > 0)", "1 update",
                "stop", "per update"),
    MonitorSpec("mask_recompute", "mask_recompute_mismatch", "> 0 (audit every 10 updates)", "1 audit", "stop",
                "per update"),
    MonitorSpec("forced_fraction", "forced_fraction", "|change| > 20 pp vs IL", "1 update", "alarm",
                "per update"),
    MonitorSpec("drop_rate", "drop_rate", "> 1% alarm; > 5% stop", "1 update", "alarm | stop", "per update"),
    MonitorSpec("opponent_passive", "opp_il_nonwait_per_game", "< 0.4x the reference", "1 update", "alarm",
                "per update"),
    MonitorSpec("critic_ev", "ev_overall, ev_last_third",
                "overall < 0.1, or last third < 0.2, or overall < warm-up end - 0.2",
                "4 consecutive updates", "enter_f1 (once)", "per update"),
    MonitorSpec("drift", "drift_alarm", "drift evaluator flagged (every 10 updates)", "1 check", "alarm",
                "per update"),
    MonitorSpec("throughput", "games_per_hour", "< 0.5x the reference", "1 update", "alarm", "system"),
    MonitorSpec("free_memory", "mem_free_gb", "< 4 GB", "1 sample", "reduce_slots (pause new games)",
                "system"),
    MonitorSpec("swap_growth", "swap_used_gb (t_sec = wall clock, time.time())", "growth > 1 GB within 60 min",
                "60 min", "reduce_slots 4", "system"),
    MonitorSpec("nonfinite", "minibatch finite", "NaN/inf", "1 minibatch; 5 consecutive -> stop",
                "skip_minibatch | stop", "system"),
    MonitorSpec("baseline_missing", "Baselines", "a reference the rules need is missing or non-positive",
                "first update of the monitor", "alarm (rules listed; also in every dashboard record)",
                "coverage (no rule silently off)"),
    MonitorSpec("metrics_missing", "Monitor.expected_update_keys(update)",
                "an expected key (active rules, audit / drift cadence, present baselines) is absent or non-finite",
                "every update", "alarm (keys listed; also in the dashboard record); strict=True raises",
                "coverage (no rule silently off)"),
    MonitorSpec("gate_rollback", "screen / G1 delta_hat", "<= -5 pp (an invalid gate is ignored: it is replayed)",
                "3 consecutive valid gates",
                "rollback + lr 0.6 temporary; 2 rollbacks within 40 updates -> beta x3 temporary", "gates"),
    MonitorSpec("clean_g1_restore", "G1 delta_hat", ">= 0", "1 G1", "lr_scale / beta_scale restoring the temporary factors", "gates"),
    MonitorSpec("stall_ladder", "G1 without promotion",
                ">= 5 consecutive; one step per non-promoting G1, each step >= 10 updates", "G1 count",
                "stall_step (+ kl step at 1 unless kl_flat_stall already took it or KL steps are skipped, then "
                "step 2 at once; enter_f1 at 3)", "stall ladder"),
    MonitorSpec("stall_final_stop", "G1 without promotion", ">= 8 consecutive", "G1 count", "stop (for a review)",
                "stall ladder"),
)


@dataclass(frozen=True)
class Action:
    """One automatic action. ``value``: factor for lr_scale / beta_scale, slots for reduce_slots,
    step number for stall_step."""
    kind: str
    monitor: str
    reason: str
    update: Optional[int] = None
    value: Optional[float] = None
    temporary: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind {self.kind!r}")

    def to_json(self) -> Dict[str, Any]:
        """Plain JSON (detail values converted by :func:`_jsonable`)."""
        d = asdict(self)
        d["detail"] = dict(self.detail)
        return _jsonable(d)


def most_severe(actions: Sequence[Action]) -> Optional[Action]:
    """The action to execute first (stop > rollback > enter_f1 > ... > log)."""
    return min(actions, key=lambda a: _SEVERITY[a.kind]) if actions else None


class _FrozenMap(dict):
    """Read-only, hashable dict for the mapping fields of the frozen dataclasses below."""
    def _ro(self, *a: Any, **k: Any) -> None:
        raise TypeError("read-only mapping (build a new Thresholds / Baselines instead)")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _ro  # type: ignore[assignment]

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items(), key=lambda kv: str(kv[0]))))

    def __reduce__(self) -> Any:
        return (self.__class__, (dict(self),))


# Thresholds fields a recorded mid-run config change may alter (the run configuration maps its runtime keys
# onto them): the early-stop window / fraction and the diagnostic alarm thresholds a calibration sets.
# Every other field is fixed for the run.
RUNTIME_THRESHOLD_FIELDS: Tuple[str, ...] = (
    "early_stop_window", "early_stop_frac", "tiny_share_max", "trend_act_rel_max", "trend_adv_gap_max",
    "entropy_ref_max_ratio", "flip_max")


def _check_runtime_thresholds(th: "Thresholds") -> None:
    """ValueError unless the :data:`RUNTIME_THRESHOLD_FIELDS` of ``th`` are usable
    (``Monitor.apply_thresholds`` checks before it swaps anything): an int window >= 1, a fraction in [0, 1] and
    each alarm None or a finite number > 0. The run configuration checks the same keys more tightly."""
    w = th.early_stop_window
    if isinstance(w, bool) or not isinstance(w, int) or w < 1:
        raise ValueError(f"early_stop_window must be an int >= 1, got {w!r}")
    f = th.early_stop_frac
    if isinstance(f, bool) or not isinstance(f, numbers.Real) or not 0.0 <= float(f) <= 1.0:
        raise ValueError(f"early_stop_frac must be in [0, 1], got {f!r}")
    for name in ("tiny_share_max", "trend_act_rel_max", "trend_adv_gap_max", "entropy_ref_max_ratio", "flip_max"):
        v = getattr(th, name)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError(f"{name} must be None or a finite number > 0, got {v!r}")


# ---------------------------------------------------------------- thresholds and baselines
@dataclass(frozen=True)
class Thresholds:
    """Every number the monitors use (immutable, hashable). The defaults are example values for tests and
    demos, not tuned settings; a run passes its own (:meth:`from_mapping`)."""
    identity_logp_tol: Mapping[str, float] = field(
        default_factory=lambda: {"wait": 1e-3, "play": 1e-3, "ability": 1e-3})
    identity_ratio_tol: float = 1e-3
    identity_kl_tol: float = 1e-5
    early_stop_window: int = 8
    early_stop_frac: float = 0.75
    clip_max: float = 0.4
    clip_window: int = 4
    temp_lr_factor: float = 0.6
    temp_lr_floor: float = TEMP_LR_FLOOR
    kl_flat_window: int = 8
    kl_flat_tol: float = 0.2
    kl_flat_g1: int = 4
    entropy_min_ratio: float = 0.4
    plays_ratio: Tuple[float, float] = (0.8, 1.25)
    card_ratio: Tuple[float, float] = (0.25, 4.0)
    core_ratio: float = 0.2
    core_window: int = 4
    core_rollbacks_before_stop: int = 2
    masked_refusal_ratio: float = 3.0
    mask_audit_every: int = 10           # mask_recompute_mismatch expected when update % mask_audit_every == 0
    drift_every: int = 10                # drift_alarm expected when update % drift_every == 0
    forced_alarm_abs: float = 0.2
    drop_alarm: float = 0.01
    drop_stop: float = 0.05
    opp_nonwait_ratio: float = 0.4
    ev_overall_min: float = 0.1
    ev_last_min: float = 0.2
    ev_drop: float = 0.2
    ev_window: int = 4
    throughput_ratio: float = 0.5
    mem_min_gb: float = 4.0
    swap_growth_gb: float = 1.0
    swap_window_s: float = 3600.0
    reduce_slots_n: int = 4
    swap_cooldown_s: float = 0.0        # no second swap cut within this many seconds (0 = off)
    swap_mem_guard_gb: Optional[float] = None   # cut only while free memory is below this (None = off)
    code19_stop_per_update: int = 1     # code-19 refusals in one update that stop the run (1 = stop at the first)
    code19_stop_consecutive: int = 1    # ... or this many updates in a row with any (1 = stop at the first)
    nonfinite_stop: int = 5
    rollback_delta: float = -0.05
    rollback_consecutive: int = 3
    rollback_lr_factor: float = 0.6
    double_rollback_updates: int = 40
    double_rollback_beta: float = 3.0
    beta_floor: float = 0.005
    beta_cap: float = 0.1
    kl_step_beta: float = 0.6
    kl_step_lr: float = 2.0
    stall_start_g1: int = 5
    stall_min_updates: int = 10
    stall_games_per_update: int = STALL_GAMES_PER_UPDATE
    final_stop_g1: int = 8
    release_after_g1: int = 5
    # KL-step skip and learner diagnostics; None = the alarm threshold is not set yet (logged only)
    kl_skip_growth: float = 1.5
    kl_skip_g1_move: float = 0.01
    kl_grad_ratio_min: float = 1e-12
    tiny_share_max: Optional[float] = None
    trend_window: int = 8
    trend_act_rel_max: Optional[float] = None
    trend_adv_gap_max: Optional[float] = None
    entropy_ref_max_ratio: Optional[float] = None
    flip_max: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity_logp_tol", _FrozenMap(self.identity_logp_tol))
        object.__setattr__(self, "plays_ratio", tuple(self.plays_ratio))
        object.__setattr__(self, "card_ratio", tuple(self.card_ratio))

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "Thresholds":
        """Build from a config mapping (unknown keys rejected, tuples restored)."""
        names = set(cls.__dataclass_fields__)
        bad = sorted(set(m) - names)
        if bad:
            raise KeyError(f"unknown monitor thresholds: {bad}")
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in m.items()}
        return cls(**kw)


@dataclass(frozen=True)
class Baselines:
    """Reference values the ratios are taken against (the IL model playing our seat at the run's
    rollout temperature against the training opponent mix; reference measurements of the IL opponents'
    activity and of throughput; the value warm-up). A missing or non-positive reference disables that ratio; the
    monitor raises one ``baseline_missing`` alarm listing the disabled rules (:meth:`disabled_rules`),
    and keeps the list in every dashboard record. ``il_masked_legal_refused_rate`` = 0 is valid (any
    positive rate then stops)."""
    il_entropy: Mapping[str, float] = field(default_factory=dict)
    il_plays_per_min: Optional[float] = None
    il_abilities_per_min: Optional[float] = None
    il_card_rate: Mapping[str, float] = field(default_factory=dict)
    core_cards: Tuple[str, ...] = ()
    il_masked_legal_refused_rate: Optional[float] = None
    il_forced_fraction: Optional[float] = None
    p3_opp_nonwait_per_game: Optional[float] = None
    m2_games_per_hour: Optional[float] = None
    ev_warmup_end: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "il_entropy",
                           _FrozenMap({canonical_head(k): v for k, v in dict(self.il_entropy).items()}))
        object.__setattr__(self, "il_card_rate", _FrozenMap(self.il_card_rate))
        object.__setattr__(self, "core_cards", tuple(self.core_cards))

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "Baselines":
        return cls(**dict(m))

    def disabled_rules(self) -> List[str]:
        """Rules switched off (or partly off) by a missing / non-positive reference."""
        def pos(x: Any) -> bool:
            v = _num(x)
            return v is not None and v > 0.0
        out = []
        ent_off = [h for h in IDENTITY_HEADS if not pos(self.il_entropy.get(h))]
        if len(ent_off) == len(IDENTITY_HEADS):
            out.append("entropy")
        else:
            out += [f"entropy[{h}]" for h in ent_off]          # per head, like the core cards
        if not pos(self.il_plays_per_min):
            out.append("plays_per_min")
        if not pos(self.il_abilities_per_min):
            out.append("abilities_per_min")
        if not any(pos(v) for v in self.il_card_rate.values()):
            out.append("card_rate")
        out += [f"core_card_collapse[{c}]" for c in self.core_cards if not pos(self.il_card_rate.get(c))]
        v = _num(self.il_masked_legal_refused_rate)
        if v is None or v < 0.0:
            out.append("mask_legal_refused")
        if _num(self.il_forced_fraction) is None:
            out.append("forced_fraction")
        if not pos(self.p3_opp_nonwait_per_game):
            out.append("opponent_passive")
        if not pos(self.m2_games_per_hour):
            out.append("throughput")
        if _num(self.ev_warmup_end) is None:
            out.append("critic_ev[warm-up drop]")
        return out


def _num(x: Any) -> Optional[float]:
    """A finite float, or None for a missing / non-finite metric (the monitor then skips it)."""
    if x is None or isinstance(x, (bool, str, bytes)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


_EPS = 1e-9   # float slack: a value exactly on a threshold never fires (0.3 / 0.1 is 2.9999999999999996)


def _ratio(x: Any, ref: Any) -> Optional[float]:
    v, r = _num(x), _num(ref)
    if v is None or r is None or r <= 0.0:
        return None
    return v / r


def _flag(x: Any) -> Optional[bool]:
    """A flag's truth value, or None when ``x`` is not a valid flag (absent, None, a float such as NaN, a
    string, ...). Valid: bool, numpy / torch 0-d bool or integer 0 / 1, int 0 / 1, and a tuple / list /
    mapping (``losses.early_stop``'s tuple of tripped limits, the drift evaluator's list of reasons; empty =
    False). A non-finite or otherwise odd value is reported missing instead of counting as True."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (tuple, list, Mapping)):
        return len(x) > 0
    if isinstance(x, (str, bytes, float)) or x is None:
        return None
    if isinstance(x, int):
        return bool(x) if x in (0, 1) else None
    item = getattr(x, "item", None)
    if callable(item) and getattr(x, "shape", None) == ():
        try:
            v = item()
        except Exception:      # noqa: BLE001 - anything odd is "not a flag"
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, int) and v in (0, 1):
            return bool(v)
    return None


def _metric(rec: Mapping[str, Any], key: str) -> Optional[float]:
    """The first finite value among ``key`` and its :data:`KEY_ALIASES` (canonical key first), or None. One
    helper for both the coverage check and the rules, so a non-finite canonical key never hides behind (or
    shadows) a finite alias."""
    v = _num(rec.get(key))
    if v is not None:
        return v
    for a, c in KEY_ALIASES.items():
        if c == key:
            v = _num(rec.get(a))
            if v is not None:
                return v
    return None


def _is_true(x: Any) -> bool:
    """``True`` or a numpy / torch scalar holding True (never a truthy string or list)."""
    if x is True:
        return True
    item = getattr(x, "item", None)
    if callable(item) and getattr(x, "shape", None) == ():
        try:
            return item() is True
        except Exception:      # noqa: BLE001 - anything odd is "not ok"
            return False
    return False


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _opt_int(u: Any) -> Optional[int]:
    return None if u is None else int(u)


# ---------------------------------------------------------------- evaluator
class Monitor:
    """Stateful evaluator of the monitor rules over one run's metric history."""

    def __init__(self, baselines: Optional[Baselines] = None, thresholds: Optional[Thresholds] = None,
                 base_lr: float = 5e-5, base_beta: float = 0.02,
                 dashboard_path: Optional[os.PathLike] = None,
                 clock: Callable[[], str] = _utc, temp_lr_floor: Optional[float] = None,
                 strict: bool = False) -> None:
        self.base = baselines if baselines is not None else Baselines()
        self.th = thresholds if thresholds is not None else Thresholds()
        self.base_lr = float(base_lr)
        self.base_beta = float(base_beta)
        self.dashboard_path = Path(dashboard_path) if dashboard_path else None
        self.clock = clock
        self.temp_lr_floor = float(self.th.temp_lr_floor if temp_lr_floor is None else temp_lr_floor)
        self.strict = bool(strict)
        self.disabled_rules = self.base.disabled_rules()
        self.last_missing: List[str] = []
        # learning-rate / beta factors: permanent (stall steps) and temporary
        self.lr_perm = 1.0
        self.lr_temp = 1.0
        self.beta_perm = 1.0
        self.beta_temp = 1.0
        self.temp_lr_causes: List[str] = []       # causes with an active temporary lr cut (clip, early stop)
        # per-update windows
        self.early: Deque[bool] = deque(maxlen=self.th.early_stop_window)
        self.clip_streak = 0                      # joint clip fraction
        self.clip_play_streak = 0                 # play-row clip fraction (a separate streak)
        self.kl_hist: Deque[float] = deque(maxlen=self.th.kl_flat_window)
        self.kl_step_done = False
        self.kl_step_update: Optional[int] = None      # update of this stretch's beta step or A/B proposal
        self.kl_step_skipped_at: Optional[int] = None  # update whose KL step was skipped (nothing applied)
        self.core_streaks: Dict[str, int] = {}
        self.core_events = 0
        self.ev_streak = 0
        self.in_f1 = False
        self.baseline_alarm_done = False
        # gates
        self.gate_bad_streak = 0
        self.rollback_updates: List[int] = []
        self.g1_since_promo = 0
        self.had_promotion = False
        self.stall_step = 0
        self.last_stall_update: Optional[int] = None
        self.release_candidate = False
        # system
        self.nonfinite_streak = 0
        self.paused_new_games = False
        self.swap_hist: List[Tuple[float, float]] = []
        self.swap_cut_t: Optional[float] = None          # wall time of the last swap_growth cut (cool-down)
        self.code19_streak = 0                           # updates in a row with a code-19 refusal
        # outcome
        self.stopped = False
        self.stop_reason: Optional[str] = None
        self.last_update: Optional[int] = None
        self.flags: List[str] = []
        self.dashboard_errors = 0
        # A/B proposals, KL-step skip, lr / beta change log, gate trends
        self.proposals: List[Dict[str, Any]] = []
        self.kl_steps_skipped = False
        self.kl_watch: Optional[Dict[str, Any]] = None
        self.kl_watch_last: Optional[Dict[str, Any]] = None
        self.last_g1_delta: Optional[float] = None
        self.scale_seq = 0
        self.scale_log: List[Dict[str, Any]] = []
        self.trend_hist: Dict[str, List[float]] = {}
        self.last_trends: Dict[str, float] = {}

    # ------------------------------------------------------------ effective values
    @property
    def lr_scale(self) -> float:
        return self.lr_perm * self.lr_temp

    @property
    def lr(self) -> float:
        return self.base_lr * self.lr_scale

    @property
    def beta(self) -> float:
        b = self.base_beta * self.beta_perm * self.beta_temp
        return min(max(b, self.th.beta_floor), self.th.beta_cap)

    # ------------------------------------------------------------ helpers
    def _stop(self, out: List[Action], monitor: str, reason: str, update: Optional[int],
              **detail: Any) -> None:
        out.append(Action("stop", monitor, reason, update, detail=detail))
        if not self.stopped:
            self.stopped = True
            self.stop_reason = f"{monitor}: {reason}"

    def _temp_lr(self, out: List[Action], monitor: str, reason: str, update: Optional[int],
                 factor: float, cause: Optional[str] = None) -> None:
        """Temporary lr factor. A ``cause`` (clip, early stop) cuts the lr at most once until the
        next clean G1; the temporary factor never drops below ``temp_lr_floor``."""
        if cause is not None and cause in self.temp_lr_causes:
            out.append(Action("log", monitor, f"{reason}: lr already cut for this cause until a clean G1",
                              update, detail={"lr": self.lr}))
            return
        new = max(self.lr_temp * factor, self.temp_lr_floor)
        if new >= self.lr_temp:
            out.append(Action("log", monitor, f"{reason}: temporary lr at its floor "
                              f"{self.temp_lr_floor:g}", update, detail={"lr": self.lr}))
            return
        eff = new / self.lr_temp
        self.lr_temp = new
        if cause is not None:
            self.temp_lr_causes.append(cause)
        out.append(Action("lr_scale", monitor, reason, update, value=eff, temporary=True,
                          detail={"lr": self.lr}))

    def _temp_beta(self, out: List[Action], monitor: str, reason: str, update: Optional[int],
                   factor: float) -> None:
        before = self.beta
        self.beta_temp *= factor
        # keep the stored factor on the cap so the later restore is exact
        if self.base_beta * self.beta_perm * self.beta_temp > self.th.beta_cap:
            self.beta_temp = self.th.beta_cap / (self.base_beta * self.beta_perm)
        eff = self.beta / before if before > 0 else factor
        out.append(Action("beta_scale", monitor, reason, update, value=eff, temporary=True,
                          detail={"beta": self.beta}))

    def _kl_step(self, out: List[Action], monitor: str, update: Optional[int]) -> bool:
        """KL step: beta x ``kl_step_beta`` down to the floor (permanent). One KL step per promotion-free
        stretch (kl_flat_stall and ladder step 1 share it). Returns False when no beta step was taken:

        * KL steps skipped (:meth:`_kl_watch_check`): a log, nothing changes;
        * beta already at the floor: no automatic lr change. A ``propose_ab`` action records an equal-games
          A/B fork from the backup point (the barrier checkpoint of this update): arm A keeps the lr, arm B
          runs at lr x ``kl_step_lr``; a person starts it.
        A beta step opens the KL watch (KL to the reference until the next G1)."""
        self.kl_step_done = True
        if self.kl_steps_skipped:                     # recorded as skipped, not as applied
            self.kl_step_skipped_at = None if update is None else int(update)
            out.append(Action("log", "kl_step_skip", f"{monitor}: KL step skipped (KL grew after an earlier step "
                              "while G1 did not improve)", update))
            return False
        if self.base_beta * self.beta_perm > self.th.beta_floor + 1e-15:   # permanent part above the floor
            before = self.beta
            self.beta_perm *= self.th.kl_step_beta
            if self.base_beta * self.beta_perm < self.th.beta_floor:
                self.beta_perm = self.th.beta_floor / self.base_beta
            self.kl_step_update = update
            out.append(Action("beta_scale", monitor, f"KL step: beta x{self.th.kl_step_beta:g} (floor)", update,
                              value=self.beta / before, detail={"beta": self.beta}))
            kl_before = (sum(self.kl_hist) / len(self.kl_hist)) if self.kl_hist else None
            self.kl_watch = {"update": None if update is None else int(update), "kl_before": kl_before,
                             "g1_before": self.last_g1_delta, "kl_after": []}
            return True
        self.kl_step_update = update
        lr_b = self.lr * self.th.kl_step_lr
        prop = {"kind": "lr_ab_fork", "update": None if update is None else int(update), "monitor": monitor,
                "fork_from": "the barrier checkpoint of this update (backup point)", "equal_games": True,
                "min_updates_per_arm": int(self.th.stall_min_updates),
                "arms": {"A": {"lr": self.lr, "lr_factor": 1.0}, "B": {"lr": lr_b, "lr_factor": self.th.kl_step_lr}},
                "beta": self.beta, "status": "proposed (needs a person; no automatic change at night)"}
        self.proposals.append(prop)
        self.flags.append(f"ab_proposal@{update}")
        out.append(Action("propose_ab", monitor, "KL step: beta at the floor; propose an equal-games A/B fork "
                          f"(arm B lr x{self.th.kl_step_lr:g}) instead of a permanent lr change", update,
                          value=self.th.kl_step_lr, detail=prop))
        return False

    def _kl_watch_check(self, out: List[Action], update: int, delta_hat: float, promoted: bool) -> None:
        """At the first G1 after a beta KL step: when the mean ``kl_ref_total`` since the step is at least
        ``kl_skip_growth`` x the window mean before it and that G1 neither promoted nor rose by at least
        ``kl_skip_g1_move`` above the last G1 before the step, KL steps are skipped from now on (KL is not the
        bottleneck; the ladder goes to its next step). The rule is one-sided on purpose: a G1 that fell after the
        step also means the step did not help. Without KL values on both sides nothing is
        decided."""
        w, self.kl_watch = self.kl_watch, None
        after, before = [float(x) for x in w.get("kl_after") or []], w.get("kl_before")
        if promoted or not after or before is None or not float(before) > 0.0:
            return
        growth = (sum(after) / len(after)) / float(before)
        g0 = w.get("g1_before")
        move = None if g0 is None else delta_hat - float(g0)
        flat = move is None or move < self.th.kl_skip_g1_move - _EPS
        detail = {"kl_before": before, "kl_after_mean": sum(after) / len(after), "growth": growth,
                  "g1_before": g0, "g1_after": delta_hat, "g1_move": move, "step_update": w.get("update")}
        detail["skip"] = bool(growth >= self.th.kl_skip_growth - _EPS and flat)
        self.kl_watch_last = detail                     # kept in the state either way (dashboard record)
        if detail["skip"]:
            self.kl_steps_skipped = True
            self.flags.append(f"kl_steps_skipped@{update}")
            out.append(Action("log", "kl_step_skip", f"KL grew {growth:.2f}x after the KL step and G1 did not improve: "
                              "later KL steps are skipped", update, detail=detail))

    def _reset_windows(self) -> None:
        self.early.clear()
        self.clip_streak = 0
        self.clip_play_streak = 0
        self.kl_hist.clear()
        self.core_streaks = {}
        self.ev_streak = 0
        self.trend_hist = {}                      # the champion's trend starts afresh
        self.last_trends = {}

    def _rollback(self, out: List[Action], monitor: str, reason: str, update: Optional[int]) -> None:
        """Rollback: back to the champion with a temporary lr cut; a second rollback within
        ``double_rollback_updates`` updates also raises beta (temporary)."""
        update = _opt_int(update)
        out.append(Action("rollback", monitor, reason, update, detail={"target": "champion"}))
        self._temp_lr(out, monitor, f"rollback: lr x{self.th.rollback_lr_factor:g} until a clean G1", update,
                      self.th.rollback_lr_factor)
        u = update if update is not None else (self.last_update or 0)
        if self.rollback_updates and u - self.rollback_updates[-1] <= self.th.double_rollback_updates:
            self._temp_beta(out, monitor, f"second rollback within {self.th.double_rollback_updates} updates: "
                            f"beta x{self.th.double_rollback_beta:g} until a clean G1", update,
                            self.th.double_rollback_beta)
        self.rollback_updates.append(int(u))
        self.gate_bad_streak = 0
        self._reset_windows()
        self.kl_watch = None                      # the KL after a rollback is the champion's: no KL-watch reading

    def _record(self, kind: str, update: Optional[int], metrics: Mapping[str, Any],
                actions: Sequence[Action]) -> None:
        if self.dashboard_path is None:
            return
        rec = _jsonable({
            "ts": self.clock(), "kind": kind, "update": update,
            "metrics": metrics,
            "actions": [a.to_json() for a in actions],
            "lr": self.lr, "lr_scale": self.lr_scale, "lr_temp": self.lr_temp, "beta": self.beta,
            "beta_temp": self.beta_temp, "g1_since_promo": self.g1_since_promo, "stall_step": self.stall_step,
            "in_f1": self.in_f1, "stopped": self.stopped, "stop_reason": self.stop_reason,
            "flags": list(self.flags), "disabled_rules": list(self.disabled_rules),
            "metrics_missing": list(self.last_missing) if kind == "update" else [],
            "kl_steps_skipped": self.kl_steps_skipped, "proposals": len(self.proposals),
            "trends": dict(self.last_trends) if kind == "update" else {},
        })
        try:
            line = json.dumps(rec, sort_keys=True, allow_nan=False) + "\n"
            self.dashboard_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dashboard_path, "a", encoding="utf-8") as f:
                f.write(line)
        except (OSError, TypeError, ValueError) as e:
            # the actions are already decided; a dashboard failure must not lose them
            self.dashboard_errors += 1
            warnings.warn(f"PPO monitor: dashboard write failed ({e!r}); actions still returned", RuntimeWarning)

    def _finish(self, kind: str, update: Optional[int], metrics: Mapping[str, Any],
                out: List[Action]) -> List[Action]:
        self._log_scales(out)
        self._record(kind, update, metrics, out)
        return out

    SCALE_LOG_MAX = 256

    def _log_scales(self, out: Sequence[Action]) -> None:
        """Every lr_scale / beta_scale action, numbered (``scale_seq``) with the lr and beta after it, so
        the learner can re-derive the planned values from the last ones it read back
        (:meth:`planned_since`); the newest :attr:`SCALE_LOG_MAX` are kept."""
        for a in out:
            if a.kind in ("lr_scale", "beta_scale"):
                self.scale_seq += 1
                self.scale_log.append({"seq": self.scale_seq, "update": a.update, "kind": a.kind,
                                       "monitor": a.monitor, "value": a.value, "lr": self.lr, "beta": self.beta})
        if len(self.scale_log) > self.SCALE_LOG_MAX:
            self.scale_log = self.scale_log[-self.SCALE_LOG_MAX:]

    def planned_since(self, seq: int, lr: float, beta: float) -> Optional[Tuple[float, float, List[Dict[str, Any]]]]:
        """``(lr, beta, entries)`` planned from values read at ``scale_seq == seq`` times every logged lr / beta
        factor since, or None when entries were dropped from the log (cannot be re-derived)."""
        new = [e for e in self.scale_log if int(e["seq"]) > int(seq)]
        if self.scale_seq > int(seq) and (not new or int(new[0]["seq"]) != int(seq) + 1):
            return None
        for e in new:
            v = float(e["value"])
            if e["kind"] == "lr_scale":
                lr *= v
            else:
                beta *= v
        return lr, beta, new

    # ------------------------------------------------------------ update-0 identity
    def check_identity(self, report: Mapping[str, Any]) -> List[Action]:
        """Update-0 identity. ``report`` keys: ``logp_max_abs_diff`` {row type: max |dlogp|} (each row
        type separately, never averaged), ``ratio_max_abs_dev`` (training-mode |r - 1|; a scalar or
        {row type: value}), ``kl_ref_per_head`` {head: KL(theta0 || ref)} for every head of
        :data:`IDENTITY_HEADS` (aliases such as ``gate`` / ``cell`` accepted), ``replay_bitwise_ok``
        (bitwise replay; True or a numpy True), optional ``backbone`` ("fp32" | "bf16"). Absolute values
        are compared. A missing entry fails (nothing unchecked passes)."""
        th, out, fails = self.th, [], []
        logp = report.get("logp_max_abs_diff") or {}
        for rt in ROW_TYPES:
            v = _num(logp.get(rt))
            tol = th.identity_logp_tol.get(rt, 1e-3)
            if v is None or not abs(v) < tol:
                fails.append(f"logp[{rt}]={logp.get(rt)!r} >= {tol:g}")
        rdev = report.get("ratio_max_abs_dev")
        rvals = ([(f"[{rt}]", rdev.get(rt)) for rt in ROW_TYPES] if isinstance(rdev, Mapping)
                 else [("", rdev)])
        for tag, x in rvals:
            r = _num(x)
            if r is None or not abs(r) < th.identity_ratio_tol:
                fails.append(f"|r-1|{tag}={x!r} >= {th.identity_ratio_tol:g}")
        kl_raw = report.get("kl_ref_per_head") or {}
        kl: Dict[str, Any] = {}
        for h, v in kl_raw.items():
            kl[canonical_head(str(h))] = v
        for h in IDENTITY_HEADS:
            if h not in kl:
                fails.append(f"KL[{h}] missing")
        for h, v in kl.items():
            fv = _num(v)
            if fv is None or not abs(fv) < th.identity_kl_tol:
                fails.append(f"KL[{h}]={v!r} >= {th.identity_kl_tol:g}")
        if not _is_true(report.get("replay_bitwise_ok")):
            fails.append("bitwise replay not ok")
        if fails:
            hint = ("bf16 failed: retry with the fp32 backbone"
                    if report.get("backbone") == "bf16" else "")
            self._stop(out, "update0_identity", "; ".join(fails), 0, hint=hint)
        else:
            out.append(Action("log", "update0_identity", "update-0 identity passed", 0))
        return self._finish("identity", 0, report, out)

    # ------------------------------------------------------------ per-update rules
    @staticmethod
    def _entropy(rec: Mapping[str, Any]) -> Dict[str, Any]:
        """Per-head entropy from a nested ``entropy`` mapping and/or flat ``entropy/<head>`` keys
        (``losses.update_metrics``), canonical head names; the flat keys win."""
        ent: Dict[str, Any] = {}
        nested = rec.get("entropy")
        if isinstance(nested, Mapping):
            for h, v in nested.items():
                ent[canonical_head(str(h))] = v
        for k, v in rec.items():
            if isinstance(k, str) and k.startswith("entropy/"):
                ent[canonical_head(k[len("entropy/"):])] = v
        return ent

    def expected_update_keys(self, update: int) -> List[str]:
        """Metric keys the active rules need on ``update``:

        * always: :data:`ALWAYS_NUMERIC_KEYS` and the flag ``early_stopped`` (bool or the
          ``losses.early_stop`` tuple; False / () are valid values);
        * on audit updates (``update % mask_audit_every == 0``): ``mask_recompute_mismatch``;
        * on drift updates (``update % drift_every == 0``): the flag ``drift_alarm``;
        * with a present baseline: ``masked_legal_refused_rate`` (IL rate >= 0), ``entropy/<head>``
          (positive IL entropy; a nested ``entropy[head]`` also counts), ``plays_per_min``,
          ``abilities_per_min``, ``card_rate[<card>]`` (every card with a positive IL rate, core cards
          included), ``forced_fraction``, ``opp_il_nonwait_per_game``, ``games_per_hour``.

        ``ev_overall`` / ``ev_last_third`` come from ``gae.explained_variance_by_phase`` (against the
        terminal outcome); ``ev_by_opponent`` and ``drop_by_reason`` are details, not expected."""
        th, b, u = self.th, self.base, int(update)
        keys = list(ALWAYS_NUMERIC_KEYS) + ["early_stopped"]
        if th.mask_audit_every > 0 and u % th.mask_audit_every == 0:
            keys.append("mask_recompute_mismatch")
        if th.drift_every > 0 and u % th.drift_every == 0:
            keys.append("drift_alarm")
        v = _num(b.il_masked_legal_refused_rate)
        if v is not None and v >= 0.0:
            keys.append("masked_legal_refused_rate")
        keys += [f"entropy/{h}" for h, r in sorted(b.il_entropy.items()) if (_num(r) or 0.0) > 0.0]
        for key, ref in (("plays_per_min", b.il_plays_per_min), ("abilities_per_min", b.il_abilities_per_min)):
            if (_num(ref) or 0.0) > 0.0:
                keys.append(key)
        keys += [f"card_rate[{c}]" for c, r in sorted(b.il_card_rate.items(), key=lambda kv: str(kv[0]))
                 if (_num(r) or 0.0) > 0.0]
        if _num(b.il_forced_fraction) is not None:
            keys.append("forced_fraction")
        if (_num(b.p3_opp_nonwait_per_game) or 0.0) > 0.0:
            keys.append("opp_il_nonwait_per_game")
        if (_num(b.m2_games_per_hour) or 0.0) > 0.0:
            keys.append("games_per_hour")
        # diagnostic rules become expected once their thresholds are set (None: logged only, never expected)
        if th.tiny_share_max is not None:
            keys += ["tiny_ratio_share/play", "tiny_ratio_share/wait"]
        if th.trend_act_rel_max is not None:
            keys.append("gate_trend/act_rate/all")
        if th.trend_adv_gap_max is not None:
            keys.append("gate_trend/adv_gap/all")
        if th.flip_max is not None:
            keys += ["probe_flip/slot", "probe_flip/cell"]
        return keys

    def missing_update_keys(self, rec: Mapping[str, Any]) -> List[str]:
        """The keys of :meth:`expected_update_keys` that ``rec`` lacks or holds as a non-finite number
        (flags: absent or not a valid flag value, see :func:`_flag`; a NaN flag is missing, never True)."""
        u = int(rec["update"])
        ent = self._entropy(rec)
        rates = rec.get("card_rate")
        rates = rates if isinstance(rates, Mapping) else {}
        out = []
        for k in self.expected_update_keys(u):
            if k in FLAG_KEYS:
                ok = _flag(rec.get(k)) is not None
            elif k.startswith("entropy/"):
                ok = _num(ent.get(k[len("entropy/"):])) is not None
            elif k.startswith("card_rate["):
                ok = _num(rates.get(k[len("card_rate["):-1])) is not None
            else:
                ok = _metric(rec, k) is not None
            if not ok:
                out.append(k)
        return out

    def stop(self, monitor: str, reason: str, update: Optional[int], **detail: Any) -> List[Action]:
        """A stop decided outside the update record (e.g. the learner's lag-0 guard), recorded in the state
        and the dashboard like any rule's stop."""
        out: List[Action] = []
        self._stop(out, monitor, reason, update, **detail)
        return self._finish("guard", update, dict(detail), out)

    def observe_update(self, rec: Mapping[str, Any]) -> List[Action]:
        """One PPO update. ``rec["update"]`` is required; the other keys are those of
        :meth:`expected_update_keys` (see also :data:`MONITORS`; ``losses.update_metrics`` output can be
        passed as is, with ``early_stopped`` = any epoch's ``losses.early_stop`` tuple). An expected key
        that is absent or non-finite skips its rule for this update and is listed in one
        ``metrics_missing`` alarm; with ``strict=True`` :class:`MetricsMissingError` is raised before any
        state changes. Returns the actions in evaluation order."""
        th, b, out = self.th, self.base, []
        u = int(rec["update"])
        if self.stopped:
            self.last_update = u
            self.last_missing = []
            return self._finish("update", u, rec, out)
        missing = self.missing_update_keys(rec)
        if missing and self.strict:
            raise MetricsMissingError(u, missing)
        self.last_update = u
        self.last_missing = missing
        if self.disabled_rules and not self.baseline_alarm_done:
            self.baseline_alarm_done = True
            out.append(Action("alarm", "baseline_missing", "missing baselines switch off: "
                              + ", ".join(self.disabled_rules), u, detail={"rules": list(self.disabled_rules)}))
        if missing:
            out.append(Action("alarm", "metrics_missing", "expected metrics absent or non-finite (rules skipped "
                              "this update): " + ", ".join(missing), u, detail={"keys": list(missing)}))

        # -- stops: mask audit and drop rate
        c19 = _num(rec.get("refusals_code19"))
        per, cons = int(th.code19_stop_per_update), int(th.code19_stop_consecutive)
        if per <= 1 and cons <= 1:                # the defaults: stop at the first refusal, no streak kept
            if c19 is not None and c19 > 0:
                self._stop(out, "mask_illegal_cell", f"{int(c19)} illegal-cell refusals (code 19)", u)
        elif c19 is not None and c19 > 0:
            self.code19_streak += 1
            if c19 >= per or self.code19_streak >= cons:
                self._stop(out, "mask_illegal_cell", f"{int(c19)} illegal-cell refusals (code 19)", u)
            else:                                 # below both stop lines it is reported, the update is learned
                out.append(Action("alarm", "mask_illegal_cell",
                                  f"{int(c19)} illegal-cell refusals (code 19), update {self.code19_streak} in a row "
                                  f"(stop at {per} in one update or {cons} updates in a row)", u,
                                  value=float(c19), detail={"streak": int(self.code19_streak)}))
        elif c19 is not None:
            self.code19_streak = 0
        mlr, mref = _num(rec.get("masked_legal_refused_rate")), _num(b.il_masked_legal_refused_rate)
        if mlr is not None and mref is not None and mref >= 0.0 and mlr > th.masked_refusal_ratio * mref + _EPS:
            how = f"{mlr / mref:.2f}x IL" if mref > 0 else f"{mlr:.4g} with IL 0"
            self._stop(out, "mask_legal_refused", f"masked-legal-but-refused {how} > {th.masked_refusal_ratio:g}x", u)
        mm = _num(rec.get("mask_recompute_mismatch"))
        if mm is not None and mm > 0:
            self._stop(out, "mask_recompute", f"{int(mm)} recomputed-mask mismatches", u)
        drop = _num(rec.get("drop_rate"))
        if drop is not None:
            if drop > th.drop_stop + _EPS:
                self._stop(out, "drop_rate", f"drop rate {drop:.4f} > {th.drop_stop}", u,
                           by_reason=rec.get("drop_by_reason"))
            elif drop > th.drop_alarm + _EPS:
                out.append(Action("alarm", "drop_rate", f"drop rate {drop:.4f} > {th.drop_alarm}", u,
                                  detail={"by_reason": rec.get("drop_by_reason")}))

        # -- core high-cost card collapse (per card) -> rollback (second time: stop)
        rolled = False
        rates = rec.get("card_rate") or {}
        for c in b.core_cards:
            r = _ratio(rates.get(c), b.il_card_rate.get(c))
            if r is None:
                continue                                  # not observed this update: streak unchanged
            self.core_streaks[c] = self.core_streaks.get(c, 0) + 1 if r < th.core_ratio - _EPS else 0
        hit = [c for c in b.core_cards if self.core_streaks.get(c, 0) >= th.core_window]
        if hit and not self.stopped:
            self.core_events += 1
            why = ", ".join(f"{c} {_ratio(rates.get(c), b.il_card_rate.get(c)):.3f}x IL" for c in hit)
            if self.core_events > th.core_rollbacks_before_stop:
                self._stop(out, "core_card_collapse", f"repeated core-card collapse ({why}); wait for review", u)
            else:
                self.flags.append(f"core_card_collapse@{u}")
                self._rollback(out, "core_card_collapse", f"core card < {th.core_ratio:g}x IL for "
                               f"{th.core_window} updates ({why})", u)
                rolled = True
            self.core_streaks = {}

        if not rolled and not self.stopped:
            # -- approx-KL early stops: more than half of the window
            es = _flag(rec.get("early_stopped"))
            if es is not None:
                self.early.append(es)                    # bool, numpy bool, 0/1 or losses.early_stop tuple
                if (len(self.early) == th.early_stop_window
                        and sum(self.early) > th.early_stop_frac * th.early_stop_window):
                    n = sum(self.early)
                    self._temp_lr(out, "approx_kl_early_stop",
                                  f"{n}/{th.early_stop_window} updates early-stopped", u, th.temp_lr_factor,
                                  cause="approx_kl_early_stop")
                    self.early.clear()
            # -- clip fraction: joint and play rows, separate streaks
            cj = _num(rec.get("clip_frac"))
            cp = _metric(rec, "play_clip_frac")
            fired = []
            if cj is not None:
                self.clip_streak = self.clip_streak + 1 if cj > th.clip_max + _EPS else 0
                if self.clip_streak >= th.clip_window:
                    fired.append("joint")
            if cp is not None:
                self.clip_play_streak = self.clip_play_streak + 1 if cp > th.clip_max + _EPS else 0
                if self.clip_play_streak >= th.clip_window:
                    fired.append("play-row")
            if fired:
                self._temp_lr(out, "clip_fraction", f"{' and '.join(fired)} clip fraction > {th.clip_max} for "
                              f"{th.clip_window} updates", u, th.temp_lr_factor, cause="clip_fraction")
                self.clip_streak = 0
                self.clip_play_streak = 0
            # -- KL flat + no promotion -> one KL step
            kl = _num(rec.get("kl_ref_total"))
            if kl is not None:
                if self.kl_watch is not None and (self.kl_watch["update"] is None or u > self.kl_watch["update"]):
                    self.kl_watch["kl_after"].append(kl)          # KL since the last beta KL step
                self.kl_hist.append(kl)
                if (len(self.kl_hist) == th.kl_flat_window and not self.kl_step_done
                        and self.g1_since_promo >= th.kl_flat_g1 and self._kl_flat()):
                    self._kl_step(out, "kl_flat_stall", u)

        # -- alarms only
        self._alarms(out, rec, u)
        self._diagnostics(out, rec, u)

        # -- critic EV -> F1
        evo, evl = _num(rec.get("ev_overall")), _num(rec.get("ev_last_third"))
        if evo is not None or evl is not None:
            bad = ((evo is not None and evo < th.ev_overall_min)
                   or (evl is not None and evl < th.ev_last_min)
                   or (evo is not None and _num(b.ev_warmup_end) is not None
                       and evo < float(b.ev_warmup_end) - th.ev_drop - _EPS))
            self.ev_streak = self.ev_streak + 1 if bad else 0
            if self.ev_streak >= th.ev_window and not self.in_f1 and not self.stopped:
                self.in_f1 = True
                out.append(Action("enter_f1", "critic_ev", f"critic EV low for {th.ev_window} updates "
                                  f"(overall {evo}, last third {evl})", u,
                                  detail={"by_opponent": rec.get("ev_by_opponent")}))
        return self._finish("update", u, rec, out)

    def _kl_flat(self) -> bool:
        x0 = self.kl_hist[0]
        tol = self.th.kl_flat_tol * abs(x0)
        return all(abs(x - x0) <= tol + _EPS * max(abs(x0), 1e-12) for x in self.kl_hist)

    def _alarms(self, out: List[Action], rec: Mapping[str, Any], u: int) -> None:
        th, b = self.th, self.base
        for head, v in self._entropy(rec).items():
            r = _ratio(v, b.il_entropy.get(head))
            if r is not None and r < th.entropy_min_ratio - _EPS:
                out.append(Action("alarm", "entropy", f"entropy[{head}] {r:.3f}x IL < {th.entropy_min_ratio:g}x", u))
        lo, hi = th.plays_ratio
        for key, ref in (("plays_per_min", b.il_plays_per_min), ("abilities_per_min", b.il_abilities_per_min)):
            r = _ratio(rec.get(key), ref)
            if r is not None and not lo - _EPS <= r <= hi + _EPS:
                out.append(Action("alarm", "plays_per_min", f"{key} {r:.3f}x IL outside [{lo}, {hi}]", u))
        lo, hi = th.card_ratio
        for card, v in (rec.get("card_rate") or {}).items():
            r = _ratio(v, b.il_card_rate.get(card))
            if r is not None and not lo - _EPS <= r <= hi + _EPS:
                out.append(Action("alarm", "card_rate", f"card {card} {r:.3f}x IL outside [{lo}, {hi}]", u))
        ff, fref = _num(rec.get("forced_fraction")), _num(b.il_forced_fraction)
        if ff is not None and fref is not None and abs(ff - fref) > th.forced_alarm_abs + _EPS:
            out.append(Action("alarm", "forced_fraction", f"forced-row fraction {ff:.3f} vs IL {fref:.3f}", u))
        r = _ratio(rec.get("opp_il_nonwait_per_game"), b.p3_opp_nonwait_per_game)
        if r is not None and r < th.opp_nonwait_ratio - _EPS:
            out.append(Action("alarm", "opponent_passive", f"IL opponent non-wait actions {r:.3f}x the reference "
                              f"< {th.opp_nonwait_ratio:g}x", u))
        if _flag(rec.get("drift_alarm")):
            out.append(Action("alarm", "drift", "human-consistency drift flagged", u,
                              detail={"drift": rec.get("drift_alarm")}))
        r = _ratio(rec.get("games_per_hour"), b.m2_games_per_hour)
        if r is not None and r < th.throughput_ratio - _EPS:
            out.append(Action("alarm", "throughput", f"throughput {r:.3f}x the reference < {th.throughput_ratio:g}x",
                              u))

    TREND_PREFIXES: Tuple[str, ...] = ("gate_trend/act_rate/", "gate_trend/act_rate_behind/",
                                       "gate_trend/adv_gap/", "gate_trend/adv_gap_behind/")
    # per-head / per-card entropy ratios (learner), versus the KL reference and versus the IL model (probe rows)
    ENTROPY_RATIO_PREFIXES: Tuple[str, ...] = ("entropy_vs_ref/", "cell_entropy_vs_ref/", "entropy_vs_il/",
                                               "cell_entropy_vs_il/")

    @staticmethod
    def window_drift(values: Sequence[float]) -> float:
        """OLS slope of ``values`` against 0..n-1 times (n - 1): the fitted change across the window."""
        n = len(values)
        if n < 2:
            return float("nan")
        xm, ym = (n - 1) / 2.0, sum(values) / n
        sxx = sum((i - xm) ** 2 for i in range(n))
        return sum((i - xm) * (v - ym) for i, v in enumerate(values)) / sxx * (n - 1)

    def _diagnostics(self, out: List[Action], rec: Mapping[str, Any], u: int) -> None:
        """Learner diagnostics (alarm only, never a stop or a selection criterion):

        * ``tiny_ratio_share/<play|wait>`` above ``tiny_share_max``;
        * two-sided trend over the last ``trend_window`` updates of every
          ``gate_trend/act_rate[_behind]/<group>`` (relative drift = window drift / window mean) and
          ``gate_trend/adv_gap[_behind]/<group>`` (absolute drift); the drifts are always kept in
          :attr:`last_trends` (dashboard ``trends``), the alarms need their thresholds;
        * ``entropy_vs_ref/<head>`` and ``cell_entropy_vs_ref/<card>`` (versus the KL reference) and
          ``entropy_vs_il/<head>`` / ``cell_entropy_vs_il/<card>`` (versus the IL model on the fixed probe rows,
          still IL after a promotion) below ``entropy_min_ratio`` or above ``entropy_ref_max_ratio``;
        * ``probe_flip/<slot|cell>`` above ``flip_max``."""
        th = self.th
        if th.tiny_share_max is not None:
            for rt in ("play", "wait"):
                v = _num(rec.get(f"tiny_ratio_share/{rt}"))
                if v is not None and v > th.tiny_share_max + _EPS:
                    out.append(Action("alarm", "tiny_ratio", f"{rt} rows with log ratio < cut: share {v:.4g} > "
                                      f"{th.tiny_share_max:g}", u))
        trends: Dict[str, float] = {}
        for k in sorted(str(x) for x in rec):
            if not k.startswith(self.TREND_PREFIXES) or "_se/" in k or "_games/" in k:
                continue
            v = _num(rec.get(k))
            if v is None:
                continue
            h = (self.trend_hist.get(k) or []) + [v]
            self.trend_hist[k] = h[-int(th.trend_window):]
            if len(self.trend_hist[k]) < int(th.trend_window):
                continue
            d = self.window_drift(self.trend_hist[k])
            rate = k.startswith("gate_trend/act_rate")
            if rate:
                mean = sum(self.trend_hist[k]) / len(self.trend_hist[k])
                d = d / mean if mean > 0 else float("nan")
            if not math.isfinite(d):
                continue
            trends[k] = d
            lim = th.trend_act_rel_max if rate else th.trend_adv_gap_max
            if lim is not None and abs(d) > lim + _EPS:
                way = "up" if d > 0 else "down"
                out.append(Action("alarm", "gate_trend", f"{k} drifted {way} {d:+.4g} over {th.trend_window} updates "
                                  f"(|.| > {lim:g})", u, detail={"window": list(self.trend_hist[k])}))
        self.last_trends = trends
        for k in sorted(str(x) for x in rec):
            if not k.startswith(self.ENTROPY_RATIO_PREFIXES):
                continue
            r = _num(rec.get(k))
            if r is None:
                continue
            base = "IL" if "_vs_il/" in k else "the reference"
            if r < th.entropy_min_ratio - _EPS:
                out.append(Action("alarm", "entropy_vs_reference", f"{k} {r:.3f}x {base} < "
                                  f"{th.entropy_min_ratio:g}x", u))
            elif th.entropy_ref_max_ratio is not None and r > th.entropy_ref_max_ratio + _EPS:
                out.append(Action("alarm", "entropy_vs_reference", f"{k} {r:.3f}x {base} > "
                                  f"{th.entropy_ref_max_ratio:g}x", u))
        if th.flip_max is not None:
            for h in ("slot", "cell"):
                v = _num(rec.get(f"probe_flip/{h}"))
                if v is not None and v > th.flip_max + _EPS:
                    out.append(Action("alarm", "argmax_flip", f"{h} argmax flip rate {v:.4g} > {th.flip_max:g}", u))

    # ------------------------------------------------------------ system
    def observe_minibatch(self, update: int, finite: bool) -> List[Action]:
        """NaN/inf in a minibatch's loss or gradients -> skip it; ``nonfinite_stop`` in a row -> stop."""
        update = int(update)
        out: List[Action] = []
        if self.stopped:
            return out
        if finite:
            self.nonfinite_streak = 0
            return out
        self.nonfinite_streak += 1
        out.append(Action("skip_minibatch", "nonfinite", "NaN/inf in minibatch", update,
                          detail={"consecutive": self.nonfinite_streak}))
        if self.nonfinite_streak >= self.th.nonfinite_stop:
            self._stop(out, "nonfinite", f"{self.nonfinite_streak} consecutive non-finite minibatches", update)
        return self._finish("minibatch", update, {"finite": finite}, out)

    def observe_system(self, t_sec: float, mem_free_gb: Optional[float] = None,
                       swap_used_gb: Optional[float] = None, update: Optional[int] = None) -> List[Action]:
        """Host sample. ``t_sec`` is wall-clock seconds (``time.time()``), so the swap window carries
        across chained blocks; a sample dated after ``t_sec`` means the clock went back (e.g. a
        restarted per-block clock), and the whole swap history is then dropped.
        Free memory below ``mem_min_gb`` pauses new games (resumed with a log once it is back); swap growth
        above ``swap_growth_gb`` within ``swap_window_s`` removes ``reduce_slots_n`` slots (the swap baseline then
        restarts). With ``swap_cooldown_s`` > 0 a cut is followed by a quiet period: the samples keep the
        baseline at the newest value but no further cut is made (one growth episode could otherwise cut the
        slots several times before any worker process has had a chance to exit)."""
        th, out = self.th, []
        update = _opt_int(update)
        if self.stopped:
            return out
        mem = _num(mem_free_gb)
        if mem is not None:
            if mem < th.mem_min_gb - _EPS:
                out.append(Action("reduce_slots", "free_memory", f"free memory {mem:.2f} GB < {th.mem_min_gb} GB",
                                  update, value=0.0, detail={"pause_new_games": True}))
                self.paused_new_games = True
            elif self.paused_new_games:
                out.append(Action("log", "free_memory", f"free memory {mem:.2f} GB: resume opening games",
                                  update, detail={"pause_new_games": False}))
                self.paused_new_games = False
        sw = _num(swap_used_gb)
        if sw is not None:
            t = float(t_sec)
            if any(ts > t for ts, _ in self.swap_hist):
                self.swap_hist = []                       # the clock went back: the old samples are not comparable
            self.swap_hist = [(ts, v) for ts, v in self.swap_hist if ts >= t - th.swap_window_s]
            cut_t = self.swap_cut_t
            if cut_t is not None and cut_t > t:
                cut_t = self.swap_cut_t = None            # the clock went back
            if cut_t is not None and th.swap_cooldown_s > 0 and t - cut_t < th.swap_cooldown_s:
                self.swap_hist = [(t, sw)]                # cooling down: growth is measured from after the cut
                base = None
            else:
                base = min((v for _, v in self.swap_hist), default=None)
                self.swap_hist.append((t, sw))
            # swap in use is system-wide and a high-water mark (another process can push pages out); with the
            # guard set, growth alone is not acted on unless free memory is short as well. A sample without a
            # memory reading does not cut.
            guard = th.swap_mem_guard_gb
            short = guard is None or (mem is not None and mem < float(guard))
            if base is not None and sw - base > th.swap_growth_gb + _EPS and not short:
                self.swap_hist = [(t, sw)]                # measured again from here
                base = None
            if base is not None and sw - base > th.swap_growth_gb + _EPS:
                self.swap_cut_t = t
                out.append(Action("reduce_slots", "swap_growth",
                                  f"swap grew {sw - base:.2f} GB within {th.swap_window_s / 60:.0f} min", update,
                                  value=float(th.reduce_slots_n)))
                self.swap_hist = [(t, sw)]
        if not out:
            return out
        return self._finish("system", update, {"t_sec": t_sec, "mem_free_gb": mem_free_gb,
                                               "swap_used_gb": swap_used_gb}, out)

    # ------------------------------------------------------------ gates, rollback and stall ladder
    def observe_gate(self, kind: str, update: int, delta_hat: float, promoted: bool = False,
                     valid: bool = True) -> List[Action]:
        """A screen S (``kind="screen"``) or a G1 (``kind="g1"``) result. ``delta_hat`` is a fraction
        (-0.05 = -5 pp); ``promoted`` is the G1 + G1c outcome; invalid gates (incomplete) are
        logged and ignored: they are replayed, and the replay is the next gate of the consecutive run.
        (``stats.rollback_triggered`` skips None the same way.)"""
        if kind not in ("screen", "g1"):
            raise ValueError(f"gate kind must be 'screen' or 'g1', got {kind!r}")
        update = int(update)
        promoted, valid = bool(promoted), bool(valid)
        th, out = self.th, []
        metrics = {"gate": kind, "delta_hat": delta_hat, "promoted": promoted, "valid": valid}
        self.last_update = update
        if self.stopped:
            return self._finish("gate", update, metrics, out)
        if not valid:
            out.append(Action("log", "gate_rollback", f"{kind} invalid (incomplete): ignored, replay it",
                              update))
            return self._finish("gate", update, metrics, out)
        d = float(delta_hat)
        if not math.isfinite(d):
            raise ValueError("delta_hat must be finite")
        if promoted and kind != "g1":
            raise ValueError("only a G1 promotes")

        # rollback trigger: consecutive gates at <= rollback_delta
        if d <= th.rollback_delta + _EPS:
            self.gate_bad_streak += 1
        else:
            self.gate_bad_streak = 0
        if self.gate_bad_streak >= th.rollback_consecutive:
            self._rollback(out, "gate_rollback", f"{th.rollback_consecutive} consecutive gates "
                           f"delta_hat <= {th.rollback_delta * 100:.0f} pp (last {d * 100:.2f} pp)", update)

        if kind == "g1":
            if self.kl_watch is not None and (self.kl_watch["update"] is None or update > self.kl_watch["update"]):
                self._kl_watch_check(out, update, d, promoted)      # KL watch, before this G1's ladder step
            # clean G1 restores every temporary change
            if d >= 0.0 and (self.lr_temp != 1.0 or self.beta_temp != 1.0 or self.temp_lr_causes):
                if self.lr_temp != 1.0:
                    f = 1.0 / self.lr_temp
                    self.lr_temp = 1.0
                    out.append(Action("lr_scale", "clean_g1_restore", "clean G1: temporary lr restored", update,
                                      value=f, detail={"restore": True, "lr": self.lr}))
                if self.beta_temp != 1.0:
                    before = self.beta
                    self.beta_temp = 1.0
                    out.append(Action("beta_scale", "clean_g1_restore", "clean G1: temporary beta restored",
                                      update, value=self.beta / before, detail={"restore": True, "beta": self.beta}))
                self.temp_lr_causes = []
            if promoted:
                self.g1_since_promo = 0
                self.had_promotion = True
                self.stall_step = 0
                self.last_stall_update = None
                self.kl_step_done = False
                self.kl_step_update = None
                self.kl_step_skipped_at = None
                self.release_candidate = False
                self.gate_bad_streak = 0
                out.append(Action("log", "stall_ladder", "promotion: ladder and stall counters reset", update))
            else:
                self.g1_since_promo += 1
                self.last_g1_delta = d                    # before _stall: a ladder KL step reads it as "before"
                self._stall(out, update)
            self.last_g1_delta = d
        return self._finish("gate", update, metrics, out)

    def _stall(self, out: List[Action], update: int) -> None:
        th, n = self.th, self.g1_since_promo
        if self.had_promotion and n >= th.release_after_g1 and not self.release_candidate:
            self.release_candidate = True
            self.flags.append(f"release_candidate@{update}")
            out.append(Action("log", "stall_ladder", f"{n} G1 without a new promotion after a promotion: "
                              "the last champion is the release candidate", update,
                              detail={"release_candidate": True}))
        if n >= th.final_stop_g1:
            self._stop(out, "stall_final_stop", f"{n} consecutive G1 without promotion: stop for a review",
                       update, stall_step=self.stall_step)
            return
        if n < th.stall_start_g1 or self.stall_step >= len(STALL_LADDER):
            return
        if self.last_stall_update is not None and update - self.last_stall_update < th.stall_min_updates:
            return
        self.stall_step += 1
        self.last_stall_update = update
        step = dict(STALL_LADDER[self.stall_step - 1])
        if step["name"] == "kl_step" and self.kl_steps_skipped:
            # KL steps skipped: KL is not the bottleneck; step 1 is recorded as skipped (or, when this stretch's KL step
            # was already taken and judged useless, as done) and step 2 follows at once
            step["skipped"] = "KL steps skipped (KL grew after a KL step while G1 did not improve)"
            if self.kl_step_done:
                self._kl_step_record(step)
            out.append(Action("stall_step", "stall_ladder", f"{n} G1 without promotion: step 1 (kl_step) skipped",
                              update, value=1.0, detail=step))
            self.stall_step += 1
            step = dict(STALL_LADDER[self.stall_step - 1])
        if step["name"] == "games_per_update":
            step["games_per_update"] = int(th.stall_games_per_update)
            step["do"] = f"{int(th.stall_games_per_update):,} games per update"
        if step["name"] == "kl_step" and self.kl_step_done:
            self._kl_step_record(step)
        out.append(Action("stall_step", "stall_ladder", f"{n} G1 without promotion: step {self.stall_step} "
                          f"({step['name']})", update, value=float(self.stall_step), detail=step))
        if step["name"] == "kl_step":
            if not self.kl_step_done:                 # the KL-flat rule and ladder step 1 are one KL step
                self._kl_step(out, "stall_ladder", update)
        elif step["name"] == "f1" and not self.in_f1:
            self.in_f1 = True
            out.append(Action("enter_f1", "stall_ladder", "stall ladder step 3", update))

    def _kl_step_record(self, step: Dict[str, Any]) -> None:
        """The ladder record of this stretch's earlier KL step: ``kl_step_already_applied_at`` for a beta step or an
        A/B proposal, ``kl_step_skipped_at`` when the KL-step skip rule skipped it (nothing was applied)."""
        if self.kl_step_update is not None:
            step["kl_step_already_applied_at"] = self.kl_step_update
        elif self.kl_step_skipped_at is not None:
            step["kl_step_skipped_at"] = self.kl_step_skipped_at

    # ------------------------------------------------------------ runtime threshold change
    def apply_thresholds(self, new: "Thresholds", update: Optional[int] = None,
                         reason: str = "") -> Dict[str, List[Any]]:
        """Swap in ``new`` thresholds mid-run (e.g. a calibration feeding the next phase of the same run).
        Only :data:`RUNTIME_THRESHOLD_FIELDS` may differ from the current ones (ValueError otherwise,
        nothing changed). A changed ``early_stop_window`` rebuilds the early-stop window with the new length,
        keeping its newest entries; the alarm rules read ``self.th`` on their next observation, and a rule
        whose threshold leaves None becomes expected by :meth:`expected_update_keys` from then on. Writes one
        ``thresholds`` dashboard record; returns ``{field: [old, new]}`` (empty: nothing changed). The caller
        passes ``Thresholds.from_mapping(...)`` built from the changed run configuration; a resumed run
        rebuilds the monitor from the new configuration's thresholds, so the change survives a restore."""
        if not isinstance(new, Thresholds):
            raise TypeError(f"apply_thresholds needs a Thresholds, got {type(new).__name__}")
        old_d, new_d = asdict(self.th), asdict(new)
        changed = {k: [_jsonable(old_d[k]), _jsonable(new_d[k])] for k in old_d if old_d[k] != new_d[k]}
        bad = sorted(k for k in changed if k not in RUNTIME_THRESHOLD_FIELDS)
        if bad:
            raise ValueError(f"monitor thresholds {bad} may not change mid-run (allowed: "
                             f"{list(RUNTIME_THRESHOLD_FIELDS)}); start a new run instead")
        if not changed:
            return {}
        _check_runtime_thresholds(new)          # everything is checked and built before anything is swapped
        early = self.early
        if early.maxlen != new.early_stop_window:
            early = deque(list(early)[-new.early_stop_window:], maxlen=new.early_stop_window)
        self.th, self.early = new, early
        self._record("thresholds", update, {"changed": changed, "reason": str(reason)}, [])
        return changed

    # ------------------------------------------------------------ persistence
    _STATE_KEYS = ("lr_perm", "lr_temp", "beta_perm", "beta_temp", "temp_lr_causes", "clip_streak",
                   "clip_play_streak", "kl_step_done", "kl_step_update", "kl_step_skipped_at", "core_streaks",
                   "core_events", "ev_streak", "in_f1", "baseline_alarm_done", "gate_bad_streak", "rollback_updates",
                   "g1_since_promo", "had_promotion", "stall_step", "last_stall_update", "release_candidate",
                   "nonfinite_streak", "paused_new_games", "swap_hist", "swap_cut_t", "code19_streak", "stopped",
                   "stop_reason",
                   "last_update", "flags", "strict", "proposals", "kl_steps_skipped", "kl_watch", "kl_watch_last",
                   "last_g1_delta", "scale_seq", "scale_log", "trend_hist", "last_trends")

    def state_dict(self) -> Dict[str, Any]:
        """JSON-serialisable state, so the next chained block continues the same history."""
        d = {k: getattr(self, k) for k in self._STATE_KEYS}
        if not d.get("code19_streak"):
            d.pop("code19_streak", None)              # written only while a streak runs (same reason as swap_cut_t)
        if d.get("swap_cut_t") is None:
            d.pop("swap_cut_t", None)                 # written only after a cut: the state of a run without one
            #                                           keeps the shorter key set
        d["swap_hist"] = [list(x) for x in self.swap_hist]
        d["rollback_updates"] = list(self.rollback_updates)
        d["flags"] = list(self.flags)
        d["temp_lr_causes"] = list(self.temp_lr_causes)
        d["core_streaks"] = dict(self.core_streaks)
        d["early"] = list(self.early)
        d["kl_hist"] = list(self.kl_hist)
        d["base_lr"], d["base_beta"] = self.base_lr, self.base_beta
        return _jsonable(d)

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        """Restore :meth:`state_dict` (keys absent from an older state keep their fresh defaults)."""
        for k in self._STATE_KEYS:
            if k in d:
                setattr(self, k, d[k])
        self.swap_hist = [(float(a), float(b)) for a, b in d.get("swap_hist", [])]
        self.swap_cut_t = None if d.get("swap_cut_t") is None else float(d["swap_cut_t"])
        self.code19_streak = int(d.get("code19_streak") or 0)
        self.rollback_updates = [int(x) for x in d.get("rollback_updates", [])]
        self.flags = list(d.get("flags", []))
        self.temp_lr_causes = list(d.get("temp_lr_causes", []))
        self.core_streaks = {str(k): int(v) for k, v in dict(d.get("core_streaks", {})).items()}
        self.early = deque((bool(x) for x in d.get("early", [])), maxlen=self.th.early_stop_window)
        self.kl_hist = deque((float(x) for x in d.get("kl_hist", [])), maxlen=self.th.kl_flat_window)
        self.proposals = [dict(x) for x in d.get("proposals", [])]
        self.kl_watch = None if d.get("kl_watch") is None else dict(d["kl_watch"])
        self.kl_watch_last = None if d.get("kl_watch_last") is None else dict(d["kl_watch_last"])
        self.scale_seq = int(d.get("scale_seq", 0))
        self.scale_log = [dict(x) for x in d.get("scale_log", [])]
        self.trend_hist = {str(k): [float(x) for x in v] for k, v in dict(d.get("trend_hist", {})).items()}
        self.last_trends = {str(k): float(v) for k, v in dict(d.get("last_trends", {})).items()}

    @classmethod
    def from_state(cls, d: Mapping[str, Any], baselines: Optional[Baselines] = None,
                   thresholds: Optional[Thresholds] = None, **kw: Any) -> "Monitor":
        kw.setdefault("base_lr", d.get("base_lr", 5e-5))
        kw.setdefault("base_beta", d.get("base_beta", 0.02))
        kw.setdefault("strict", bool(d.get("strict", False)))   # strict survives a save / restore
        m = cls(baselines, thresholds, **kw)
        d = {k: v for k, v in d.items() if k != "strict"}    # an explicit strict= keyword wins
        m.load_state_dict(d)
        return m

    def save(self, path: os.PathLike) -> None:
        """Atomic JSON write of :meth:`state_dict`."""
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(self.state_dict(), sort_keys=True, indent=1, allow_nan=False), encoding="utf-8")
        os.replace(tmp, p)


def write_stop_marker(run_dir: os.PathLike, reason: str, update: Optional[int] = None,
                      actions: Iterable[Action] = ()) -> Path:
    """Leave the stop marker the block chainer checks. Only a person removes it."""
    p = Path(run_dir) / STOP_MARKER
    p.parent.mkdir(parents=True, exist_ok=True)
    body = _jsonable({"ts": _utc(), "reason": reason, "update": update, "actions": [a.to_json() for a in actions]})
    p.write_text(json.dumps(body, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return p


def read_dashboard(path: os.PathLike) -> List[Dict[str, Any]]:
    """All JSON-lines dashboard records."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _jsonable(x: Any) -> Any:
    """Plain JSON: numpy / torch scalars and arrays become Python values (``tolist``), non-finite
    floats become strings, unknown objects their ``str``; a NaN or numpy count never breaks the
    dashboard, the stop marker or :meth:`Monitor.save`."""
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, deque)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return [_jsonable(v) for v in sorted(x, key=str)]
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        try:
            y = tolist()
        except Exception:      # noqa: BLE001 - fall through to str
            y = x
        if y is not x and not callable(getattr(y, "tolist", None)):
            return _jsonable(y)
    if isinstance(x, numbers.Integral):
        return int(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return v if math.isfinite(v) else str(v)
