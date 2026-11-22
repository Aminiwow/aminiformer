"""Rollout shards for PPO v7: the format the rollout coordinator writes and the learner reads.

What the format covers: committed rows with their HT and seat weights (only the tokens of committed rows are
stored; the shards of the last ``KEEP_UPDATES`` updates are kept), θ_k samples at T = 1, the terminal reward
(truncated games are dropped and counted, never bootstrapped), Φ logged on every decision tick, the trained
seat roles, and the per-opponent settings row logged with every game.

Layout: ``<root>/u{update:06d}/w{worker:03d}.npz``, one file per worker per update, written once
(temporary file, fsync, then ``os.link`` (no clobber) or ``os.replace`` when ``overwrite``). No pickle:
every array is numeric or bool, the header is UTF-8 JSON stored as a ``uint8`` array, and the reader opens
with ``allow_pickle=False``.

What one shard holds
--------------------
* ``header``: schema version and digest, update, worker, the acting identity (``policy_digest`` = sha256 of
  the served checkpoint's ``model_ema``, ``rollout_id`` = one id per attempt of an update) and code / config
  digests, the kept games' :class:`GameMeta`, the dropped games' :class:`GameMeta` (with ``drop_reason`` and
  Φ at the drop, for the drop-rate monitor) plus their per-seat counters (``dropped_seats``), one metadata
  dict per kept seat, the update counters (summed and ``by_role``) and the token layout.
* ``row.<field>`` (``ROW_FIELDS``): one entry per **committed** row of every kept trained seat, seats
  concatenated in append order. ``seat_offsets [n_seats + 1]`` delimits them.
* ``seat.<field>`` (``SEAT_FIELDS``): per kept seat (game index, side, role, weight, outcome, temperature,
  hazard_dt code, counters).
* ``tok.*``: the actor's own per-row tokens (``SeatAdapter.make_row`` output), see below.

Why tokens (and not ROW / ENT / OBJ + draws)
---------------------------------------------
The format trades disk for CPU: the actor stores the tokens of committed rows and the learner recomputes
everything else. Re-tokenising from ROW / ENT / OBJ is bitwise equal only if the per-row link permutation, pi
and the per-EVT ``delta`` actually drawn are stored too, it re-runs the tokenizer on the learner and it depends
on adapter internals; the stored tokens are exactly the model input the actor's ``peek`` saw (``slot_legal``,
``ability_ready``, ``cell_mask_id`` included), which is also what the ``mask_recompute_mismatch`` audit compares
against. Group arrays are stored flat with a per-row width
(``tok.w.<GROUP>``) so the ragged ENTITY / OBJECT groups cost only their own width (a multi-row token dict
is trimmed per row to its last used token when split); :meth:`SeatGame.tokens` pads one seat-game back to
``tokens_from_rows`` form (``[T, W_max, ...]`` with the tokenizer's fill values), ready for
``TK.collate([{'tokens': t, 'labels': {}}])``.

Row conventions
---------------
* ``family`` is the sampled family as ``sample_actions`` returns it (a blocked play already folded to wait;
  the writer requires ``family == executed_family``); ``slot`` / ``cell`` / ``card`` are -1 unless the
  executed family is play (``card`` may stay -1 = unknown), ``ability`` is -1 unless it is ability (the
  stored form ``logprob.executed_logp_terms`` accepts). The ability index is the adapter's ``pointer``.
* ``mu`` is ``logprob.executed_logp`` of that action on the peek output at the sampling temperature, float64
  (a float32 ``mu`` is refused); ``mu_terms`` = (family, slot, cell, ability). ``valid`` is
  ``executed_logp_terms(...)['valid']``.
* ``stratum`` is ``gae.committed_stratum(family, threat)`` (never the actor row's stratum, which is written
  before the family is sampled); the writer checks it.
* ``phi`` is the tower potential Φ at the row tick from that seat's own view (``gae.tower_potential``).
* ``client_slot_ok`` (schema .3): the client slot-mask bits the actor's mask used (all True when the seat had
  no client mask). ``legal_slots`` are the actor's masked logits, so ``legal_slots`` = tokens' ``slot_legal``
  AND ``client_slot_ok``; the learner applies the same bits to its recompute (pi_theta, pi_prox, the KL
  reference: one support with mu).

Views and roles
---------------
``GameMeta.result``, ``phi_end``, ``phi_drop``, ``phi_kickoff`` and the Φ log are from **our_owner's** view.
``seat.outcome`` and :attr:`SeatGame.phi_end` (``UpdateData.seat_phi_end()``) are the per-seat views
(negated on the side != our_owner); use those for GAE and ``gae.short_head_gae(phi_end=...)``. The writer
requires role ``'ours'`` on our_owner's side, ``'opponent'`` on the other side of a ``self_seat`` game, and
``'mirror'`` seats exactly in ``mirror`` games (:func:`check_seat_roles`).

Kept and dropped games
----------------------
A game is dropped (rows not stored, metadata and counters kept) when the caller says so (``drop_reason``),
when it did not end regularly (``end`` != ``gae.END_NATIVE``), when ``gae.validate_episode`` rejects a trained
seat (no committed row, ticks not strictly increasing, no outcome), when a trained seat was halted (its
executed action was forced to wait: ``seat_stall``) or when a row is not ``valid`` (a writer or legality bug:
``invalid``). Each update therefore has exactly the games the coordinator played, kept plus dropped
(``load_update(..., expect_games=N)``). The coordinator starts an attempt of update N
with :func:`begin_update` (refuses, or clears, a non-empty ``u{N}``) so shards of two attempts never mix;
``load_update`` also requires one ``rollout_id`` and one ``policy_digest`` across the update's shards.
"""
from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import warnings
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from amini.nn import schema_v1 as S
from amini.ppo import gae as G

SCHEMA_VERSION = "ppo-v7-shard.3"              # .3: row.client_slot_ok (the client slot-mask bits)
KEEP_UPDATES = 2                                # keep the shards of the last KEEP_UPDATES updates
T_MIN = 1.0e-3                                  # == logprob.T_MIN (kept torch-free here; a test checks it)
MU_POS_TOL = 1.0e-12                            # a log-prob is <= 0
MU_SUM_TOL = 1.0e-9                             # mu == sum(mu_terms) on valid rows

UPDATE_RE = re.compile(r"^u(\d{6})$")
SHARD_RE = re.compile(r"^w(\d{3})\.npz$")

ROLES: Tuple[str, ...] = ("ours", "opponent", "mirror")     # trained seat roles (gae.SEAT_WEIGHT keys)
SELF_SEAT_KIND, MIRROR_KIND = "self_seat", "mirror"         # opponent kinds (config.OPPONENT_KINDS)
HAZARD_DT_MODES: Tuple[str, ...] = ("block", "interval")    # == config.HAZARD_DT_MODES
COUNTER_KEYS: Tuple[str, ...] = ("queries", "committed", "forced", "skipped")
RESERVED_COUNTERS: Tuple[str, ...] = ("games", "games_kept", "games_dropped", "rows_kept", "drop_by_reason",
                                      "by_role")
OUTCOMES = {"win": 1.0, "loss": -1.0, "draw": 0.0}

# name -> (dtype, trailing shape); one entry per committed row
ROW_FIELDS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "tick": ("int64", ()),
    "query_index": ("int32", ()),          # index of the query in the seat's query sequence (skipped included)
    "dt": ("float64", ()),                 # hazard dt in ticks used by sample_actions
    "family": ("int8", ()),                # sampled family (blocked play folded to wait) == executed_family
    "slot": ("int8", ()),                  # -1 unless the executed family is play
    "cell": ("int16", ()),                 # -1 unless play; cell = gy * 18 + gx (self frame)
    "ability": ("int16", ()),              # -1 unless ability (the adapter's ``pointer``)
    "card": ("int32", ()),                 # card id played (hand at the slot); -1 unless play or unknown
    "play_blocked": ("bool", ()),
    "mu": ("float64", ()),                 # actor executed log-prob (LOG_ZERO when not valid)
    "mu_terms": ("float64", (4,)),         # family, slot, cell, ability
    "valid": ("bool", ()),
    "executed_family": ("int8", ()),
    "forced": ("bool", ()),                # no legal slot and no ready ability (policy weight 0)
    "legal_slots": ("bool", (4,)),         # slot legality as sample_actions saw it (masked logits)
    "ability_avail": ("bool", ()),
    "n_ready": ("int16", ()),
    "possible": ("bool", (3,)),            # executable folded families (wait, play, ability)
    "stratum": ("int8", ()),               # gae.committed_stratum(family, threat)
    "threat": ("bool", ()),
    "phi": ("float64", ()),                # tower potential at the row tick (this seat's view)
    "halted": ("bool", ()),                # the seat was halted (executed action forced to wait)
    "client_slot_ok": ("bool", (4,)),      # client slot-mask bits applied to the slot logits (all True = none)
}
ROW_LIMITS: Dict[str, Tuple[int, int]] = {
    "family": (0, 2), "executed_family": (0, 2), "slot": (-1, 3), "cell": (-1, S.N_CELLS - 1),
    "ability": (-1, S.ABILITY_POINTER - 1), "stratum": (0, 2), "n_ready": (0, S.ABILITY_POINTER),
    "query_index": (0, 2 ** 31 - 1), "tick": (0, 2 ** 31 - 1), "card": (-1, 2 ** 31 - 1),
}

# name -> (dtype, trailing shape); one entry per kept trained seat
SEAT_FIELDS: Dict[str, Tuple[str, Tuple[int, ...]]] = {
    "game": ("int32", ()),                 # index into the shard's kept games
    "side": ("int8", ()),
    "role": ("int8", ()),                  # index into ROLES
    "weight": ("float64", ()),             # seat weight
    "outcome": ("float64", ()),            # terminal reward from this seat's view (+1 / -1 / 0)
    "temperature": ("float64", (3,)),      # sampling T of slot, cell, ability (gate always 1)
    "hazard_dt": ("int8", ()),             # index into HAZARD_DT_MODES
    "counters": ("int64", (len(COUNTER_KEYS),)),
}

GAME_FIELDS: Tuple[str, ...] = (
    "game_id", "update", "replica", "case_id", "level", "our_owner", "our_deck", "opp_kind", "opp_settings",
    "seeds", "tower_troops", "hazard_dt", "code_digest", "config_digest", "end", "result", "end_tick",
    "phi_end", "drop_reason", "phi_drop", "extra", "policy_digest", "rollout_id", "phi_kickoff",
    "phi_problems", "wall_s", "plan", "opp_stats")

GROUP_KEYS: Tuple[str, ...] = ("feat", "cat", "lnk", "valid")
GROUP_DTYPES: Dict[str, str] = {"feat": "float32", "cat": "int64", "lnk": "int64", "valid": "bool"}
ROW_TOKEN_REQUIRED: Tuple[str, ...] = ("tick", "tq", "pos", "slot_legal", "ability_ready", "cell_mask_id")
ROW_TOKEN_FILL: Dict[str, int] = {"ent_uid": -1, "obj_uid": -1}      # tokenizer / collate fill (else 0)
RAGGED_ROW_GROUP: Dict[str, str] = {"ent_uid": "ENTITY", "obj_uid": "OBJECT"}   # row key -> its ragged group


class SchemaError(ValueError):
    """A shard does not match this module's schema (version, digest, keys, dtypes, shapes or semantics)."""


# =========================================================================== schema digest
def _group_fill(key: str) -> Any:
    return {"feat": 0.0, "cat": 0, "lnk": S.LINK_NONE, "valid": False}[key]


def group_tails() -> Dict[str, Dict[str, Tuple[int, ...]]]:
    """Trailing shape of every token group array after ``[rows, width]`` (from ``S.GROUPS``)."""
    return {name: {"feat": (g.n_features,), "cat": (len(g.cats),), "lnk": (len(g.roles),), "valid": ()}
            for name, g in S.GROUPS.items()}


def schema_spec() -> Dict[str, Any]:
    """Canonical description of the shard schema; its sha256 is :data:`SCHEMA_DIGEST`."""
    return {
        "version": SCHEMA_VERSION,
        "row": [[k, d, list(t)] for k, (d, t) in ROW_FIELDS.items()],
        "seat": [[k, d, list(t)] for k, (d, t) in SEAT_FIELDS.items()],
        "game": list(GAME_FIELDS),
        "roles": list(ROLES), "hazard_dt": list(HAZARD_DT_MODES), "counters": list(COUNTER_KEYS),
        "groups": {n: {"cap": g.cap, "ragged": bool(g.ragged), "n_features": g.n_features,
                       "n_cats": len(g.cats), "n_roles": len(g.roles)} for n, g in S.GROUPS.items()},
        "group_dtypes": GROUP_DTYPES, "link_none": int(S.LINK_NONE),
        "row_token_required": list(ROW_TOKEN_REQUIRED), "row_token_fill": ROW_TOKEN_FILL,
        "strata": [int(S.STRATUM_ACTION), int(S.STRATUM_BASE), int(S.STRATUM_THREAT)],
        "families": [int(S.FAMILY_WAIT), int(S.FAMILY_PLAY), int(S.FAMILY_ABILITY)],
    }


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


SCHEMA_DIGEST = hashlib.sha256(_canon(schema_spec()).encode()).hexdigest()


# =========================================================================== metadata
def _jsonable(x: Any) -> Any:
    """Plain JSON value of ``x`` (dataclasses, tuples, numpy scalars / arrays); NaN / inf rejected."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not np.isfinite(v):
            raise ValueError(f"metadata value must be finite, got {x!r}")
        return v
    if x is None or isinstance(x, str):
        return x
    raise TypeError(f"metadata value of type {type(x).__name__} is not JSON-serialisable")


def _outcome(result: Any) -> Optional[float]:
    if result is None:
        return None
    if isinstance(result, (bool, np.bool_)):
        raise ValueError(f"result must be +1 / -1 / 0 or 'win' / 'loss' / 'draw', not a bool ({result!r})")
    if isinstance(result, str):
        if result not in OUTCOMES:
            raise ValueError(f"unknown result {result!r}")
        return OUTCOMES[result]
    return G.terminal_reward(result)


def _seat_view(x: float, side: int, our_owner: int) -> float:
    """``x`` (our_owner's view) from seat ``side``'s view; never -0.0."""
    return (float(x) if int(side) == int(our_owner) else -float(x)) + 0.0


def _count(name: str, v: Any) -> int:
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or int(v) < 0:
        raise ValueError(f"counter {name} must be an int >= 0, got {v!r}")
    return int(v)


@dataclasses.dataclass
class GameMeta:
    """One game's record. ``result`` is the terminal reward from OUR seat's view
    (``our_owner``; +1 / -1 / 0, or 'win' / 'loss' / 'draw'), None when the game did not finish regularly.
    ``seeds`` holds the per-role seeds (``cases.game_seeds`` plus the seats'); ``opp_settings`` the opponent's
    settings row (``cases.OpponentSpec`` or ``cfg.opponent_settings()[kind]``; a snapshot carries its own
    checkpoint digest there); ``hazard_dt`` the resolved mode per seat (e.g. {'ours': 'block', 'opponent':
    'block'}). Φ values are from our_owner's view: ``phi_end`` the real end-of-game Φ (short head; per seat
    use :attr:`SeatGame.phi_end`), ``phi_drop`` Φ when a dropped game stopped, ``phi_kickoff`` Φ at the first
    decision tick and ``phi_problems`` ``gae.potential_log_problems`` of the game's Φ log (both filled by
    ``append_game(phi_log=...)``). ``plan`` is ``dataclasses.asdict(cases.GamePlan)``; ``opp_stats`` frozen
    opponent counters (e.g. ``{'nonwait': 37}`` for ``opp_il_nonwait_per_game``); ``wall_s`` the game's wall
    time. ``drop_reason``, ``policy_digest`` and ``rollout_id`` are set by :meth:`ShardWriter.append_game`."""
    game_id: str
    update: int
    replica: int
    case_id: str
    level: int
    our_owner: int
    our_deck: Sequence[Any]
    opp_kind: str
    opp_settings: Mapping[str, Any]
    seeds: Mapping[str, int]
    tower_troops: Mapping[str, Any]
    hazard_dt: Mapping[str, Optional[str]]
    code_digest: str
    config_digest: str
    end: str
    result: Optional[Union[str, float]]
    end_tick: int
    phi_end: Optional[float] = None
    drop_reason: Optional[str] = None
    phi_drop: Optional[float] = None
    extra: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    policy_digest: Optional[str] = None
    rollout_id: Optional[str] = None
    phi_kickoff: Optional[float] = None
    phi_problems: Sequence[str] = dataclasses.field(default_factory=list)
    wall_s: Optional[float] = None
    plan: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    opp_stats: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {f: _jsonable(getattr(self, f)) for f in GAME_FIELDS}
        d["result"] = _outcome(self.result)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GameMeta":
        missing = [f for f in GAME_FIELDS if f not in d]
        extra = [k for k in d if k not in GAME_FIELDS]
        if missing or extra:
            raise SchemaError(f"game record keys: missing {missing}, unexpected {extra}")
        return cls(**{f: d[f] for f in GAME_FIELDS})

    def check(self) -> None:
        """Type / range checks (ValueError)."""
        for f in ("game_id", "case_id", "opp_kind", "code_digest", "config_digest", "end"):
            if not isinstance(getattr(self, f), str):
                raise ValueError(f"GameMeta.{f} must be a string")
        for f in ("policy_digest", "rollout_id"):
            v = getattr(self, f)
            if v is not None and (not isinstance(v, str) or not v):
                raise ValueError(f"GameMeta.{f} must be None or a non-empty string")
        for f in ("update", "replica", "level", "our_owner", "end_tick"):
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < 0:
                raise ValueError(f"GameMeta.{f} must be an int >= 0, got {v!r}")
        if int(self.our_owner) not in (0, 1):
            raise ValueError("GameMeta.our_owner must be 0 or 1")
        if self.end != G.END_NATIVE and self.end not in G.DROP_REASONS:
            raise ValueError(f"GameMeta.end must be {G.END_NATIVE!r} or one of {G.DROP_REASONS}, got {self.end!r}")
        if self.drop_reason is not None and self.drop_reason not in G.DROP_REASONS:
            raise ValueError(f"unknown drop reason {self.drop_reason!r}")
        for mode in dict(self.hazard_dt).values():
            if mode is not None and mode not in HAZARD_DT_MODES:
                raise ValueError(f"hazard_dt mode {mode!r} not in {HAZARD_DT_MODES}")
        for f in ("phi_end", "phi_drop", "phi_kickoff"):
            v = getattr(self, f)
            if v is not None and (isinstance(v, bool) or not (np.isfinite(float(v)) and -1.0 <= float(v) <= 1.0)):
                raise ValueError(f"GameMeta.{f} must be in [-1, 1], got {v!r}")
        if self.wall_s is not None and (isinstance(self.wall_s, bool) or not np.isfinite(float(self.wall_s))
                                        or float(self.wall_s) < 0):
            raise ValueError(f"GameMeta.wall_s must be a finite number >= 0, got {self.wall_s!r}")
        if isinstance(self.phi_problems, str) or not all(isinstance(x, str) for x in self.phi_problems):
            raise ValueError("GameMeta.phi_problems must be a list of strings")
        for f in ("plan", "opp_stats", "extra"):
            v = getattr(self, f)
            if not isinstance(v, Mapping) and not (dataclasses.is_dataclass(v) and not isinstance(v, type)):
                raise ValueError(f"GameMeta.{f} must be a mapping")
        _outcome(self.result)
        self.to_dict()                                                    # JSON-serialisable


@dataclasses.dataclass
class SeatRecord:
    """One trained seat of a game (frozen opponents produce no record).

    ``rows``: ``ROW_FIELDS`` arrays over the seat's committed rows (:func:`actor_rows` builds them from the
    captured peek outputs). ``tokens``: the per-row token dicts of those rows (``SeatAdapter.rows[i]['tokens']``,
    leading dimension 1 each) or one ``tokens_from_rows`` dict with leading dimension = rows (trimmed per row).
    ``counters``: at least ``COUNTER_KEYS`` (queries reached incl. skipped, committed == len(rows), forced
    among the peeked queries, skipped = not peeked); extra integer keys are summed into the update counters
    too (overall and by role). ``temperature``: the slot / cell / ability sampling T in [T_MIN, 1]."""
    side: int
    role: str
    weight: float
    rows: Mapping[str, Any]
    tokens: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]]
    counters: Mapping[str, int]
    model: str = "main"
    hazard_dt: str = "block"
    temperature: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    profile: str = "engine"


def check_seat_roles(our_owner: int, opp_kind: str, sides: Sequence[int], roles: Sequence[str]) -> None:
    """ValueError unless the trained seats fit the game: 'ours' on our_owner's side, 'opponent'
    on the other side and only in a ``self_seat`` game, 'mirror' seats exactly in a ``mirror`` game; every
    non-mirror game trains 'ours' (frozen IL / snapshot opponents train 'ours' only). The outcome sign is
    derived from side vs our_owner, so a wrong owner would otherwise invert every reward silently."""
    for side, role in zip(sides, roles):
        if role not in ROLES:
            raise ValueError(f"unknown seat role {role!r}")
        if role == "ours" and int(side) != int(our_owner):
            raise ValueError(f"seat 'ours' at side {side} but our_owner is {our_owner}")
        if role == "opponent" and int(side) == int(our_owner):
            raise ValueError(f"seat 'opponent' at our_owner's side {side}")
    rs = set(roles)
    if opp_kind == MIRROR_KIND:
        if rs != {"mirror"}:
            raise ValueError(f"a {MIRROR_KIND!r} game trains mirror seats only, got {sorted(rs)}")
        return
    if "mirror" in rs:
        raise ValueError(f"mirror seats in a {opp_kind!r} game")
    if "opponent" in rs and opp_kind != SELF_SEAT_KIND:
        raise ValueError(f"an 'opponent' seat in a {opp_kind!r} game (only {SELF_SEAT_KIND!r} trains it)")
    if "ours" not in rs:
        raise ValueError(f"a {opp_kind!r} game must train the 'ours' seat")


# =========================================================================== actor-side helper
def stack_peek_outputs(outs: Sequence[Mapping[str, Any]],
                       keys: Sequence[str] = ("family", "slot", "ability", "ability_avail", "cells", "value")
                       ) -> Dict[str, Any]:
    """Concatenate per-query peek outputs (``[1, ...]`` each) over rows (torch.cat per key)."""
    import torch
    return {k: torch.cat([torch.as_tensor(o[k]) for o in outs]) for k in keys if k in outs[0]}


def actor_rows(out: Mapping[str, Any], dt: Any, action: Mapping[str, Any], *, threat: Any, ticks: Any,
               phi: Any, halted: Any = None, query_index: Any = None, card: Any = None,
               client_ok: Any = None) -> Dict[str, np.ndarray]:
    """``ROW_FIELDS`` arrays of committed rows from the actor's peek outputs.

    ``out``: stacked peek outputs of the committed rows **at the sampling temperature** (after
    ``logprob.apply_temperature`` when T != 1); ``dt``: the hazard dt per row; ``action``: the
    ``sample_actions`` dict (or its stored form; the adapter decision's ``pointer`` is ``ability``).
    Unused heads are stored as -1. ``threat`` / ``ticks`` / ``phi`` / ``halted`` / ``query_index`` / ``card``
    (card id in the hand at the played slot; None = unknown) per row; ``client_ok`` [n, 4]: the client
    slot-mask bits already applied to ``out`` (None = no client mask, all True)."""
    import torch
    from amini.ppo import logprob as LP

    n = int(torch.as_tensor(out["family"]).shape[0])
    dt_t = torch.as_tensor(np.asarray(dt, dtype=np.float64)).reshape(n)
    terms = LP.executed_logp_terms(out, dt_t, action)
    m = LP.legal_masks(out)
    poss = LP.family_possible(m["ability_avail"], m["legal_any"], dt_t, torch.as_tensor(out["family"]))
    fam = torch.as_tensor(action["family"]).long().reshape(n).cpu().numpy()
    ex = terms["executed_family"].cpu().numpy()

    def col(key: str) -> np.ndarray:
        return torch.as_tensor(action[key]).long().reshape(n).cpu().numpy()

    blocked = (torch.as_tensor(action["play_blocked"]).bool().reshape(n).cpu().numpy()
               if action.get("play_blocked") is not None else np.zeros(n, bool))
    th = np.asarray(threat, dtype=bool).reshape(n)
    cards = np.full(n, -1, np.int64) if card is None else np.asarray(card, dtype=np.int64).reshape(n)
    rows = {
        "tick": np.asarray(ticks, dtype=np.int64).reshape(n),
        "query_index": (np.arange(n) if query_index is None else np.asarray(query_index)).reshape(n),
        "dt": dt_t.cpu().numpy(),
        "family": fam,
        "slot": np.where(ex == S.FAMILY_PLAY, col("slot"), -1),
        "cell": np.where(ex == S.FAMILY_PLAY, col("cell"), -1),
        "ability": np.where(ex == S.FAMILY_ABILITY, col("ability"), -1),
        "card": np.where(ex == S.FAMILY_PLAY, cards, -1),
        "play_blocked": blocked,
        "mu": terms["total"].detach().double().cpu().numpy(),
        "mu_terms": torch.stack([terms[k] for k in ("family", "slot", "cell", "ability")], -1
                                ).detach().double().cpu().numpy(),
        "valid": terms["valid"].cpu().numpy(),
        "executed_family": ex,
        "forced": m["forced"].cpu().numpy(),
        "legal_slots": m["slot_legal"].cpu().numpy(),
        "ability_avail": m["ability_avail"].cpu().numpy(),
        "n_ready": m["n_ready"].cpu().numpy(),
        "possible": poss.cpu().numpy(),
        "stratum": G.committed_stratum(fam, th),
        "threat": th,
        "phi": np.asarray(phi, dtype=np.float64).reshape(n),
        "halted": np.zeros(n, bool) if halted is None else np.asarray(halted, dtype=bool).reshape(n),
        "client_slot_ok": (np.ones((n, S.HAND_SLOTS), bool) if client_ok is None
                           else np.asarray(client_ok, dtype=bool).reshape(n, S.HAND_SLOTS)),
    }
    return {k: _cast(k, v, ROW_FIELDS[k]) for k, v in rows.items()}


# =========================================================================== casting / validation
def _cast(name: str, v: Any, spec: Tuple[str, Tuple[int, ...]]) -> np.ndarray:
    """``v`` as the schema dtype; raises when the cast loses information."""
    dtype, _ = spec
    a = np.asarray(v)
    if dtype == "bool":
        if a.dtype != np.bool_:
            if a.size and not np.all(np.isin(a, (0, 1))):
                raise ValueError(f"{name}: bool field holds values other than 0 / 1")
        return a.astype(np.bool_)
    out = a.astype(dtype)
    if a.size and np.dtype(dtype).kind in "iu":
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            raise ValueError(f"{name}: non-finite value in an integer field")
        if not np.array_equal(out.astype(np.float64) if a.dtype.kind == "f" else out, a):
            raise ValueError(f"{name}: values do not fit {dtype}")
    return out


def _check_rows(rows: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Validate and cast one seat's (or a shard's) committed rows (ValueError). Action invariants that a
    legality bug can break (mu sign / sum, forced => wait) are checked on ``valid`` rows only, so such a
    game is dropped as 'invalid' rather than refused."""
    missing = [k for k in ROW_FIELDS if k not in rows]
    extra = [k for k in rows if k not in ROW_FIELDS]
    if missing or extra:
        raise ValueError(f"seat rows: missing {missing}, unexpected {extra}")
    for k in ("mu", "mu_terms"):
        if np.asarray(rows[k]).dtype != np.float64:
            raise ValueError(f"seat rows: {k} must be float64, got {np.asarray(rows[k]).dtype} "
                             "(the lost bits break the fp64 actor / learner identity)")
    out = {k: _cast(k, rows[k], ROW_FIELDS[k]) for k in ROW_FIELDS}
    n = out["tick"].shape[0] if out["tick"].ndim == 1 else -1
    for k, (_, tail) in ROW_FIELDS.items():
        if out[k].shape != (n,) + tail:
            raise ValueError(f"seat rows: {k} has shape {out[k].shape}, expected {(n,) + tail}")
    for k, (lo, hi) in ROW_LIMITS.items():
        if n and (out[k].min() < lo or out[k].max() > hi):
            raise ValueError(f"seat rows: {k} outside [{lo}, {hi}]")
    if not n:
        return out
    if not np.all(np.isfinite(out["dt"])) or np.any(out["dt"] < 0):
        raise ValueError("seat rows: dt must be finite and >= 0")
    if not np.all(np.isfinite(out["phi"])) or np.any(np.abs(out["phi"]) > 1.0):
        raise ValueError("seat rows: phi must be in [-1, 1]")
    if not np.array_equal(out["family"], out["executed_family"]):
        raise ValueError("seat rows: family != executed_family (store the folded family sample_actions "
                         "returns; an unfolded blocked play gets the wrong stratum and HT weight)")
    if not np.array_equal(out["stratum"], G.committed_stratum(out["family"], out["threat"])):
        raise ValueError("seat rows: stratum != gae.committed_stratum(family, threat) "
                         "(never use the actor row's stratum)")
    legal_any = out["legal_slots"].any(-1)
    if np.any(out["legal_slots"] & ~out["client_slot_ok"]):
        raise ValueError("seat rows: a slot the client mask refused is legal (client_slot_ok must be the bits the "
                         "actor applied to the slot logits)")
    if not np.array_equal(out["forced"], ~legal_any & ~out["ability_avail"]):
        raise ValueError("seat rows: forced != no legal slot and no ready ability")
    pos = out["dt"] > 0
    poss = np.stack([np.ones(n, bool), legal_any & pos, out["ability_avail"] & pos], -1)
    if not np.array_equal(out["possible"], poss):
        raise ValueError("seat rows: possible inconsistent with legal_slots / ability_avail / dt")
    v = out["valid"]
    play, abil = out["executed_family"] == S.FAMILY_PLAY, out["executed_family"] == S.FAMILY_ABILITY
    if np.any(~play & ((out["slot"] >= 0) | (out["cell"] >= 0) | (out["card"] >= 0))) or \
            np.any(~abil & (out["ability"] >= 0)) or \
            np.any(v & play & ((out["slot"] < 0) | (out["cell"] < 0))) or np.any(v & abil & (out["ability"] < 0)):
        raise ValueError("seat rows: slot / cell / card are set on play rows only (slot / cell on every valid "
                         "play row), ability on ability rows only (on every valid one)")
    if np.any(np.isnan(out["mu"])) or not np.all(np.isfinite(out["mu"][v])):
        raise ValueError("seat rows: mu must be finite on valid rows")
    if np.any(out["mu"][v] > MU_POS_TOL):
        raise ValueError("seat rows: mu (a log-prob) must be <= 0 on valid rows")
    if v.any() and not np.allclose(out["mu"][v], out["mu_terms"][v].sum(-1), rtol=0.0, atol=MU_SUM_TOL):
        raise ValueError("seat rows: mu != sum(mu_terms) on valid rows")
    if np.any(out["forced"][v] & (out["family"][v] != S.FAMILY_WAIT)):
        raise ValueError("seat rows: a forced valid row must be a wait")
    return out


def _used_width(grp: Mapping[str, np.ndarray], tied: Sequence[Tuple[np.ndarray, int]]) -> int:
    """Last used column + 1 of one row of a ragged group (``[W, ...]`` arrays): valid, a non-fill feat / cat /
    lnk (bitwise: -0.0 counts as used), or a non-fill tied row key (``ent_uid`` / ``obj_uid``)."""
    w = int(np.asarray(grp["valid"]).shape[0])
    if w == 0:
        return 0
    used = np.asarray(grp["valid"]).astype(bool).reshape(w).copy()
    feat = np.asarray(grp["feat"]).reshape(w, -1)
    used |= ((feat != 0) | np.signbit(feat)).any(-1)
    used |= (np.asarray(grp["cat"]).reshape(w, -1) != 0).any(-1)
    used |= (np.asarray(grp["lnk"]).reshape(w, -1) != S.LINK_NONE).any(-1)
    for a, fill in tied:
        used |= (a.reshape(w, -1) != fill).any(-1)
    idx = np.nonzero(used)[0]
    return int(idx.max()) + 1 if idx.size else 0


def _token_rows(tokens: Union[Sequence[Mapping[str, Any]], Mapping[str, Any]], n: int
                ) -> List[Mapping[str, Any]]:
    """A list of ``n`` token dicts with leading dimension 1. A multi-row dict is split, and each row's ragged
    groups (with their tied row keys) are trimmed to the row's last used token, so a row does not store the
    episode's maximum width (the reader pads back with the same fill values)."""
    if not isinstance(tokens, Mapping):
        toks = list(tokens)
        if len(toks) != n:
            raise ValueError(f"{len(toks)} token rows, seat has {n} committed rows")
        return toks
    t = int(np.asarray(tokens["row"]["tick"]).shape[0])
    if t != n:
        raise ValueError(f"tokens have {t} rows, seat has {n} committed rows")
    groups_all = {g: {k: np.asarray(v) for k, v in grp.items()} for g, grp in tokens["groups"].items()}
    row_all = {k: np.asarray(v) for k, v in tokens["row"].items()}
    out = []
    for i in range(n):
        groups, widths = {}, {}
        for g, grp in groups_all.items():
            sl = {k: v[i] for k, v in grp.items()}
            spec = S.GROUPS.get(g)
            if spec is not None and spec.ragged and set(sl) == set(GROUP_KEYS) and all(
                    v.ndim >= 1 and v.shape[0] == sl["valid"].shape[0] for v in sl.values()):
                tied = [(row_all[k][i], ROW_TOKEN_FILL.get(k, 0)) for k, gg in RAGGED_ROW_GROUP.items()
                        if gg == g and k in row_all and row_all[k].ndim == 2
                        and row_all[k].shape[1] == sl["valid"].shape[0]]
                w = _used_width(sl, tied)
                sl = {k: v[:w] for k, v in sl.items()}
                widths[g] = w
            groups[g] = {k: v[None] for k, v in sl.items()}
        row = {}
        for k, v in row_all.items():
            a = v[i:i + 1]
            g = RAGGED_ROW_GROUP.get(k)
            if g in widths and a.ndim == 2 and a.shape[1] == groups_all[g]["valid"].shape[1]:
                a = a[:, :widths[g]]
            row[k] = a
        out.append({"groups": groups, "row": row})
    return out


class _TokenAcc:
    """Accumulates flat token arrays and per-row widths over the shard's kept rows."""

    def __init__(self) -> None:
        self.groups: Dict[str, Dict[str, List[np.ndarray]]] = {g: {k: [] for k in GROUP_KEYS} for g in S.GROUPS}
        self.gw: Dict[str, List[np.ndarray]] = {g: [] for g in S.GROUPS}
        self.row: Dict[str, List[np.ndarray]] = {}
        self.rw: Dict[str, List[np.ndarray]] = {}
        self.row_spec: Optional[Dict[str, Tuple[str, Tuple[int, ...], bool]]] = None  # key -> (dtype, tail, ragged)
        self.tails = group_tails()

    def prepare(self, toks: List[Mapping[str, Any]], ticks: np.ndarray) -> Dict[str, Any]:
        """Validate one seat's token rows and return the flat chunks (nothing is accumulated yet)."""
        n = len(toks)
        chunk: Dict[str, Any] = {"g": {}, "gw": {}, "r": {}, "rw": {}}
        for name, spec in S.GROUPS.items():
            widths = np.zeros(n, np.int32)
            parts: Dict[str, List[np.ndarray]] = {k: [] for k in GROUP_KEYS}
            for i, tok in enumerate(toks):
                grp = tok["groups"].get(name)
                if grp is None or set(grp) != set(GROUP_KEYS):
                    raise ValueError(f"token row {i}: group {name} missing or keys != {GROUP_KEYS}")
                w = None
                for k in GROUP_KEYS:
                    a = np.asarray(grp[k])
                    if a.dtype != np.dtype(GROUP_DTYPES[k]):
                        raise ValueError(f"token {name}.{k} dtype {a.dtype} != {GROUP_DTYPES[k]}")
                    if a.ndim < 2 or a.shape[0] != 1 or a.shape[2:] != self.tails[name][k]:
                        raise ValueError(f"token row {i}: {name}.{k} shape {a.shape} is not "
                                         f"[1, W, {self.tails[name][k]}]")
                    if w is None:
                        w = a.shape[1]
                    elif a.shape[1] != w:
                        raise ValueError(f"token row {i}: {name} keys disagree on the width")
                    parts[k].append(a[0])
                if (not spec.ragged and w != spec.cap) or w > spec.cap:
                    raise ValueError(f"token row {i}: group {name} width {w} (cap {spec.cap}, "
                                     f"{'ragged' if spec.ragged else 'fixed'})")
                widths[i] = w
            chunk["g"][name] = {k: np.concatenate(parts[k], 0) for k in GROUP_KEYS}
            chunk["gw"][name] = widths
        keys = set(toks[0]["row"])
        miss = [k for k in ROW_TOKEN_REQUIRED if k not in keys]
        if miss:
            raise ValueError(f"token rows lack {miss}")
        spec_r: Dict[str, Tuple[str, Tuple[int, ...], bool]] = {}
        for key in sorted(keys):
            arrs = []
            for i, tok in enumerate(toks):
                if set(tok["row"]) != keys:
                    raise ValueError(f"token row {i}: row keys differ from row 0")
                a = np.asarray(tok["row"][key])
                if a.ndim < 1 or a.shape[0] != 1:
                    raise ValueError(f"token row {i}: row.{key} shape {a.shape} has no leading row axis of 1")
                arrs.append(a[0])
            ref = arrs[0]
            tail = ref.shape[1:] if ref.ndim >= 1 else ()
            for a in arrs:
                if a.dtype != ref.dtype or a.ndim != ref.ndim or (a.ndim >= 1 and a.shape[1:] != tail):
                    raise ValueError(f"token row.{key}: dtype / trailing shape differs between rows")
            if ref.dtype.kind not in "biuf":
                raise ValueError(f"token row.{key}: dtype {ref.dtype} is not numeric")
            if ref.ndim == 0:
                chunk["r"][key] = np.stack(arrs)
                spec_r[key] = (ref.dtype.str, (), False)
            else:
                chunk["r"][key] = np.concatenate(arrs, 0)
                chunk["rw"][key] = np.array([a.shape[0] for a in arrs], np.int32)
                spec_r[key] = (ref.dtype.str, tuple(int(x) for x in tail), True)
        if not np.array_equal(chunk["r"]["tick"].astype(np.int64), ticks):
            raise ValueError("token row.tick differs from the rows' tick")
        if self.row_spec is not None and spec_r != self.row_spec:
            raise ValueError("token row keys / dtypes / shapes differ from the shard's first seat")
        chunk["spec"] = spec_r
        return chunk

    def add(self, chunk: Dict[str, Any]) -> None:
        if self.row_spec is None:
            self.row_spec = chunk["spec"]
            self.row = {k: [] for k in chunk["r"]}
            self.rw = {k: [] for k in chunk["rw"]}
        for name in S.GROUPS:
            for k in GROUP_KEYS:
                self.groups[name][k].append(chunk["g"][name][k])
            self.gw[name].append(chunk["gw"][name])
        for k, v in chunk["r"].items():
            self.row[k].append(v)
        for k, v in chunk["rw"].items():
            self.rw[k].append(v)

    def arrays(self) -> Dict[str, np.ndarray]:
        if self.row_spec is None:
            return {}
        out: Dict[str, np.ndarray] = {}
        for name in S.GROUPS:
            for k in GROUP_KEYS:
                out[f"tok.g.{name}.{k}"] = np.concatenate(self.groups[name][k], 0)
            out[f"tok.w.{name}"] = np.concatenate(self.gw[name])
        for k, v in self.row.items():
            out[f"tok.r.{k}"] = np.concatenate(v, 0)
        for k, v in self.rw.items():
            out[f"tok.rw.{k}"] = np.concatenate(v)
        return out


# =========================================================================== paths, retention
def update_dir(root: Union[str, Path], update: int) -> Path:
    if isinstance(update, bool) or not 0 <= int(update) <= 999999:
        raise ValueError(f"update must be in [0, 999999], got {update!r}")
    return Path(root) / f"u{int(update):06d}"


def shard_path(root: Union[str, Path], update: int, worker: int) -> Path:
    if isinstance(worker, bool) or not 0 <= int(worker) <= 999:
        raise ValueError(f"worker must be in [0, 999], got {worker!r}")
    return update_dir(root, update) / f"w{int(worker):03d}.npz"


def list_updates(root: Union[str, Path]) -> List[int]:
    """Update numbers with a ``u{NNNNNN}`` directory under ``root``, ascending."""
    p = Path(root)
    if not p.is_dir():
        return []
    return sorted(int(m.group(1)) for d in p.iterdir() if d.is_dir() and (m := UPDATE_RE.match(d.name)))


def shard_paths(root: Union[str, Path], update: int) -> List[Path]:
    """Finished shards of one update (``w{NNN}.npz``), by worker; temporary files are ignored."""
    d = update_dir(root, update)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and SHARD_RE.match(p.name))


def begin_update(root: Union[str, Path], update: int, *, clear: bool = False) -> Path:
    """Start an attempt of ``update``: create ``u{N}`` (fsynced into ``root``) and return it.

    A directory that already holds shards (finished or temporary) raises FileExistsError, or with ``clear``
    has them removed, so a re-run never mixes with an earlier attempt's shards. Anything else in it raises."""
    d = update_dir(root, update)
    if d.is_dir():
        entries = list(d.iterdir())
        foreign = [p.name for p in entries if not (p.is_file() and (SHARD_RE.match(p.name) or _is_tmp(p.name)))]
        if foreign:
            raise FileExistsError(f"{d} holds {foreign[:3]}; refusing to reuse it")
        if entries and not clear:
            raise FileExistsError(f"{d} already holds {len(entries)} shard file(s) of an earlier attempt")
        for p in entries:
            p.unlink()
        _fsync_dir(d)
        return d
    d.mkdir(parents=True, exist_ok=True)
    _fsync_dir(d.parent)
    return d


def prune_updates(root: Union[str, Path], keep: int = KEEP_UPDATES) -> List[int]:
    """Delete all but the newest ``keep`` update directories and return the removed numbers.

    Call it at the Monitor.save / checkpoint barrier, after the learner finished update N (never while an
    update is being read). Only ``u{NNNNNN}`` directories holding nothing but shard files (finished or
    temporary) are removed; a directory with anything else is left alone with a warning."""
    if isinstance(keep, bool) or int(keep) < 1:
        raise ValueError(f"keep must be >= 1, got {keep!r}")
    removed = []
    for u in list_updates(root)[:-int(keep)]:
        d = update_dir(root, u)
        foreign = [p.name for p in d.iterdir()
                   if not (p.is_file() and (SHARD_RE.match(p.name) or _is_tmp(p.name)))]
        if foreign:
            warnings.warn(f"prune_updates: {d} holds {foreign[:3]}; not removed")
            continue
        shutil.rmtree(d)
        removed.append(u)
    return removed


def _is_tmp(name: str) -> bool:
    return name.startswith(".w") and ".npz.tmp-" in name


def _identity(name: str, v: Any) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty string")
    return v


# =========================================================================== writer
class ShardWriter:
    """Writes one worker's shard of one update. Use as a context manager or call
    :meth:`close` (atomic: temporary file, fsync, then a no-clobber ``os.link``, or ``os.replace`` when
    ``overwrite``). An exception inside ``with`` or :meth:`abort` leaves no shard behind, and a later
    :meth:`close` raises. A failed :meth:`close` (e.g. disk full) keeps the data and may be retried.

    ``policy_digest``: sha256 of the served checkpoint's ``model_ema`` (the weights that acted);
    ``rollout_id``: one id per attempt of the update (the same for every worker); both are stored in the
    header and every game record and checked by :func:`load_update`. ``seat_weights``: optional role ->
    weight table (``cfg.weights``) every seat weight must equal."""

    def __init__(self, root: Union[str, Path], update: int, worker: int, *, code_digest: str,
                 config_digest: str, policy_digest: str, rollout_id: str,
                 seat_weights: Optional[Mapping[str, float]] = None, compress: bool = False,
                 overwrite: bool = False) -> None:
        self.path = shard_path(root, update, worker)
        self.update, self.worker = int(update), int(worker)
        self.code_digest = _identity("code_digest", code_digest)
        self.config_digest = _identity("config_digest", config_digest)
        self.policy_digest = _identity("policy_digest", policy_digest)
        self.rollout_id = _identity("rollout_id", rollout_id)
        self.seat_weights = None if seat_weights is None else {str(k): float(v) for k, v in seat_weights.items()}
        self.compress, self.overwrite = bool(compress), bool(overwrite)
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"shard {self.path} exists")
        self.games: List[Dict[str, Any]] = []
        self.dropped: List[Dict[str, Any]] = []
        self.dropped_seats: List[Dict[str, Any]] = []
        self.seats_meta: List[Dict[str, Any]] = []
        self._rows: Dict[str, List[np.ndarray]] = {k: [] for k in ROW_FIELDS}
        self._seat: Dict[str, List[Any]] = {k: [] for k in SEAT_FIELDS}
        self._lengths: List[int] = []
        self._tok = _TokenAcc()
        self.counters: Dict[str, Any] = {k: 0 for k in COUNTER_KEYS}
        self.counters.update(games=0, games_kept=0, games_dropped=0, rows_kept=0, drop_by_reason={},
                             by_role={})
        self._written = False
        self._aborted = False
        self._tmp: Optional[Path] = None

    @property
    def closed(self) -> bool:
        """True once the shard was published or the writer aborted (no more games)."""
        return self._written or self._aborted

    @property
    def written(self) -> bool:
        return self._written

    # ----------------------------------------------------------------- appending
    def append_game(self, meta: GameMeta, seats: Sequence[SeatRecord], *,
                    phi_log: Optional[Sequence[float]] = None) -> Optional[str]:
        """Add one finished game with its trained seats; returns its drop reason (None = kept).

        ``phi_log``: Φ on every decision tick from kick-off (our_owner's view); when given, the game
        records ``phi_kickoff`` and ``phi_problems`` (``gae.potential_log_problems`` against ``phi_end``).
        Raises ValueError for writer bugs (bad metadata, seat roles that do not fit our_owner / opp_kind,
        stratum not from ``committed_stratum``, counters inconsistent with the rows, malformed tokens); a
        game that is merely unusable is dropped instead."""
        if self.closed:
            raise RuntimeError("shard writer is closed")
        meta = dataclasses.replace(meta)
        meta.check()
        if int(meta.update) != self.update:
            raise ValueError(f"game update {meta.update} != shard update {self.update}")
        if meta.code_digest != self.code_digest or meta.config_digest != self.config_digest:
            raise ValueError("game code / config digest differs from the shard's")
        for f in ("policy_digest", "rollout_id"):
            v = getattr(meta, f)
            if v is not None and v != getattr(self, f):
                raise ValueError(f"game {f} {v!r} differs from the shard's {getattr(self, f)!r}")
            setattr(meta, f, getattr(self, f))
        if not seats:
            raise ValueError("a game needs at least one trained seat")
        sides = [int(s.side) for s in seats]
        if len(set(sides)) != len(sides) or not set(sides) <= {0, 1}:
            raise ValueError(f"seat sides must be distinct and in (0, 1), got {sides}")
        check_seat_roles(int(meta.our_owner), meta.opp_kind, sides, [s.role for s in seats])
        if phi_log is not None:
            p = np.asarray(phi_log, dtype=np.float64).reshape(-1)
            terminal = 0.0 if meta.phi_end is None else float(meta.phi_end)
            meta.phi_problems = G.potential_log_problems(p, terminal)
            meta.phi_kickoff = float(p[0]) if p.size and np.isfinite(p[0]) and abs(p[0]) <= 1.0 else None
        res = _outcome(meta.result)
        prepared = []
        reason = meta.drop_reason
        if reason is None and meta.end != G.END_NATIVE:
            reason = meta.end
        for s in seats:
            w = float(s.weight)
            if not (np.isfinite(w) and w > 0):
                raise ValueError(f"seat weight must be finite and > 0, got {s.weight!r}")
            if self.seat_weights is not None and w != self.seat_weights.get(s.role):
                raise ValueError(f"seat weight {w} != the table's {self.seat_weights.get(s.role)} for {s.role!r}")
            if s.hazard_dt not in HAZARD_DT_MODES:
                raise ValueError(f"seat hazard_dt {s.hazard_dt!r} not in {HAZARD_DT_MODES}")
            temp = np.asarray(s.temperature, dtype=np.float64)
            if temp.shape != (3,) or not np.all(np.isfinite(temp) & (temp >= T_MIN) & (temp <= 1)):
                raise ValueError(f"seat temperature must be 3 values in [{T_MIN}, 1] (a trained seat samples), "
                                 f"got {s.temperature!r}")
            if s.profile != "engine":
                raise ValueError(f"seat profile must be 'engine', got {s.profile!r}")
            rows = _check_rows(s.rows)
            n = rows["tick"].shape[0]
            cnt = {str(k): _count(str(k), v) for k, v in dict(s.counters).items()}
            missing = [k for k in COUNTER_KEYS if k not in cnt]
            if missing:
                raise ValueError(f"seat counters need {COUNTER_KEYS}, missing {missing}")
            bad = [k for k in cnt if k in RESERVED_COUNTERS]
            if bad:
                raise ValueError(f"seat counters {bad} are maintained by the writer")
            if cnt["committed"] != n:
                raise ValueError(f"counters['committed'] = {cnt['committed']} but the seat has {n} rows")
            if cnt["skipped"] + cnt["committed"] > cnt["queries"] or \
                    cnt["forced"] > cnt["queries"] - cnt["skipped"]:
                raise ValueError(f"inconsistent seat counters {cnt}")
            outcome = None if res is None else _seat_view(res, int(s.side), int(meta.our_owner))
            if reason is None:
                reason = G.validate_episode(meta.end, rows["tick"] if n else [], outcome)
            if reason is None and bool(rows["halted"].any()):
                reason = "seat_stall"
            if reason is None and not bool(rows["valid"].all()):
                reason = "invalid"
            prepared.append((s, rows, cnt, outcome, w, temp))
        if reason is None and int(meta.end_tick) < max(int(r["tick"][-1]) for _, r, *_ in prepared):
            reason = "invalid"
        chunks = []
        if reason is None:
            for s, rows, *_ in prepared:
                chunks.append(self._tok.prepare(_token_rows(s.tokens, rows["tick"].shape[0]), rows["tick"]))
            if any(c["spec"] != chunks[0]["spec"] for c in chunks):
                raise ValueError("token row keys / dtypes / shapes differ between the seats of a game")
        meta.drop_reason = reason
        by_role = self.counters["by_role"]
        for s, rows, cnt, *_ in prepared:
            rc = by_role.setdefault(s.role, {})
            for k, v in cnt.items():
                self.counters[k] = self.counters.get(k, 0) + v
                rc[k] = rc.get(k, 0) + v
        self.counters["games"] += 1
        if reason is not None:
            self.counters["games_dropped"] += 1
            dbr = self.counters["drop_by_reason"]
            dbr[reason] = dbr.get(reason, 0) + 1
            for s, rows, cnt, outcome, w, temp in prepared:
                self.dropped_seats.append({"dropped": len(self.dropped), "side": int(s.side), "role": s.role,
                                           "weight": w, "model": str(s.model), "counters": cnt})
            self.dropped.append(meta.to_dict())
            return reason
        gi = len(self.games)
        self.games.append(meta.to_dict())
        self.counters["games_kept"] += 1
        for (s, rows, cnt, outcome, w, temp), chunk in zip(prepared, chunks):
            self._tok.add(chunk)
            for k in ROW_FIELDS:
                self._rows[k].append(rows[k])
            n = rows["tick"].shape[0]
            self._lengths.append(n)
            self.counters["rows_kept"] += n
            vals = {"game": gi, "side": int(s.side), "role": ROLES.index(s.role), "weight": w,
                    "outcome": float(outcome), "temperature": temp,
                    "hazard_dt": HAZARD_DT_MODES.index(s.hazard_dt),
                    "counters": [cnt[k] for k in COUNTER_KEYS]}
            for k in SEAT_FIELDS:
                self._seat[k].append(vals[k])
            self.seats_meta.append({"game": gi, "side": int(s.side), "role": s.role, "weight": w,
                                    "model": str(s.model), "hazard_dt": s.hazard_dt,
                                    "temperature": temp.tolist(), "profile": s.profile, "counters": cnt})
        return None

    def add_counters(self, **counts: int) -> None:
        """Add worker-level integer counters (e.g. ``refusals_code19``) to the update counters."""
        for k, v in counts.items():
            if k in RESERVED_COUNTERS or k in COUNTER_KEYS:
                raise ValueError(f"counter {k!r} is maintained by append_game")
            self.counters[k] = self.counters.get(k, 0) + _count(k, v)

    # ----------------------------------------------------------------- output
    def arrays(self) -> Dict[str, np.ndarray]:
        """Every array of the shard (header excluded)."""
        out: Dict[str, np.ndarray] = {}
        for k, (dt, tail) in ROW_FIELDS.items():
            out[f"row.{k}"] = (np.concatenate(self._rows[k], 0) if self._rows[k]
                               else np.zeros((0,) + tail, dt)).astype(dt, copy=False)
        for k, (dt, tail) in SEAT_FIELDS.items():
            out[f"seat.{k}"] = np.asarray(self._seat[k], dtype=dt).reshape((len(self._lengths),) + tail)
        out["seat_offsets"] = np.concatenate([[0], np.cumsum(self._lengths, dtype=np.int64)]).astype(np.int64)
        out.update(self._tok.arrays())
        return out

    def header(self, arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        return {"schema": SCHEMA_VERSION, "schema_digest": SCHEMA_DIGEST, "update": self.update,
                "worker": self.worker, "created_unix": time.time(), "code_digest": self.code_digest,
                "config_digest": self.config_digest, "policy_digest": self.policy_digest,
                "rollout_id": self.rollout_id, "n_games": len(self.games), "n_seats": len(self._lengths),
                "n_rows": int(arrays["seat_offsets"][-1]), "games": self.games, "dropped": self.dropped,
                "dropped_seats": self.dropped_seats, "seats": self.seats_meta, "counters": self.counters,
                "token_row_spec": ({k: [d, list(t), r] for k, (d, t, r) in self._tok.row_spec.items()}
                                   if self._tok.row_spec else {})}

    def close(self) -> Path:
        """Write the shard atomically and return its path (idempotent once written). Raises RuntimeError
        after :meth:`abort`; a failure leaves no file behind and keeps the data for a retry."""
        if self._written:
            return self.path
        if self._aborted:
            raise RuntimeError(f"shard writer for {self.path} was aborted; nothing was written")
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(f"shard {self.path} exists")
        arrays = self.arrays()
        head = self.header(arrays)
        arrays["header"] = np.frombuffer(_canon(_jsonable(head)).encode("utf-8"), dtype=np.uint8)
        if not self.path.parent.is_dir():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _fsync_dir(self.path.parent.parent)                   # the new u{N} entry in the root
        self._tmp = self.path.parent / f".{self.path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            with open(self._tmp, "wb") as fh:
                (np.savez_compressed if self.compress else np.savez)(fh, **arrays)
                fh.flush()
                os.fsync(fh.fileno())
            if self.overwrite:
                os.replace(self._tmp, self.path)
            else:
                _link_no_clobber(self._tmp, self.path)
        except BaseException:
            self._discard_tmp()
            raise
        self._written = True
        self._discard_tmp()
        _fsync_dir(self.path.parent)
        return self.path

    def _discard_tmp(self) -> None:
        if self._tmp is not None:
            try:
                self._tmp.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:                                  # never mask the original error
                warnings.warn(f"could not remove {self._tmp}: {e}")
            self._tmp = None

    def abort(self) -> None:
        """Discard the shard: remove a temporary file, write nothing; a later :meth:`close` raises. No-op
        once the shard was written."""
        if self._written:
            return
        self._discard_tmp()
        self._aborted = True

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def _link_no_clobber(src: Path, dst: Path) -> None:
    """Publish ``src`` at ``dst`` unless ``dst`` exists (atomic, race-free via ``os.link``); falls back to
    check-then-replace on a filesystem without hard links."""
    try:
        os.link(src, dst)
        return
    except FileExistsError:
        raise FileExistsError(f"shard {dst} exists (written concurrently)") from None
    except OSError as e:
        if e.errno not in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK, errno.ENOSYS):
            raise
    if dst.exists():
        raise FileExistsError(f"shard {dst} exists")
    os.replace(src, dst)


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


# =========================================================================== reader
@dataclasses.dataclass
class SeatGame:
    """One kept trained seat of one game: metadata, its committed rows and (lazily) its tokens."""
    game: GameMeta
    seat: Dict[str, Any]
    rows: Dict[str, np.ndarray]
    side: int
    role: str
    weight: float
    outcome: float
    _shard: "ShardReader"
    _lo: int
    _hi: int

    @property
    def n(self) -> int:
        return self._hi - self._lo

    @property
    def temperature(self) -> Tuple[float, float, float]:
        """The seat's sampling T (slot, cell, ability) for ``logprob.apply_temperature`` on the learner."""
        t = self.seat["temperature"]
        return float(t[0]), float(t[1]), float(t[2])

    @property
    def phi_end(self) -> Optional[float]:
        """End-of-game Φ from this seat's view (``GameMeta.phi_end`` is our_owner's), for
        ``gae.short_head_gae(phi=self.rows['phi'], phi_end=...)``; None when the game has none."""
        pe = self.game.phi_end
        return None if pe is None else _seat_view(float(pe), self.side, int(self.game.our_owner))

    def tokens(self) -> Dict[str, Any]:
        """This seat-game's tokens in ``tokens_from_rows`` form (``[T, W_max, ...]``, tokenizer fill)."""
        return self._shard.tokens(self._lo, self._hi)

    def example(self) -> Dict[str, Any]:
        """``{'tokens': ..., 'labels': {}}`` for ``TK.collate`` (no IL labels, no targets)."""
        return {"tokens": self.tokens(), "labels": {}}

    def action(self) -> Dict[str, np.ndarray]:
        """The stored action in the form ``logprob.executed_logp_terms`` takes."""
        return {k: self.rows[k] for k in ("family", "slot", "cell", "ability", "play_blocked")}


def _header(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    h = arrays.get("header")
    if h is None or h.dtype != np.uint8 or h.ndim != 1:
        raise SchemaError("shard has no uint8 header")
    try:
        head = json.loads(h.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SchemaError(f"shard header is not JSON: {e}") from None
    if not isinstance(head, dict):
        raise SchemaError("shard header is not a JSON object")
    return head


def validate_arrays(arrays: Mapping[str, np.ndarray], head: Mapping[str, Any]) -> None:
    """Schema checks of one shard's arrays against its header (SchemaError)."""
    if head.get("schema") != SCHEMA_VERSION:
        raise SchemaError(f"shard schema {head.get('schema')!r} != {SCHEMA_VERSION!r}")
    if head.get("schema_digest") != SCHEMA_DIGEST:
        raise SchemaError(f"shard schema digest {head.get('schema_digest')!r} != {SCHEMA_DIGEST!r}")
    n_rows, n_seats = int(head["n_rows"]), int(head["n_seats"])
    expect: Dict[str, Tuple[np.dtype, Tuple[int, ...]]] = {"header": (np.dtype(np.uint8), (-1,))}
    for k, (d, t) in ROW_FIELDS.items():
        expect[f"row.{k}"] = (np.dtype(d), (n_rows,) + t)
    for k, (d, t) in SEAT_FIELDS.items():
        expect[f"seat.{k}"] = (np.dtype(d), (n_seats,) + t)
    expect["seat_offsets"] = (np.dtype(np.int64), (n_seats + 1,))
    spec_r = {k: (np.dtype(d), tuple(t), bool(r))
              for k, (d, t, r) in dict(head.get("token_row_spec") or {}).items()}
    if n_rows:
        if not spec_r or any(k not in spec_r for k in ROW_TOKEN_REQUIRED):
            raise SchemaError("shard with rows lacks the token row spec")
        tails = group_tails()
        for name in S.GROUPS:
            w = arrays.get(f"tok.w.{name}")
            if w is None or w.dtype != np.int32 or w.shape != (n_rows,):
                raise SchemaError(f"tok.w.{name} missing or not int32 [{n_rows}]")
            total = int(w.sum())
            for k in GROUP_KEYS:
                expect[f"tok.g.{name}.{k}"] = (np.dtype(GROUP_DTYPES[k]), (total,) + tails[name][k])
            expect[f"tok.w.{name}"] = (np.dtype(np.int32), (n_rows,))
            spec = S.GROUPS[name]
            if np.any(w < 0) or np.any(w > spec.cap) or (not spec.ragged and np.any(w != spec.cap)):
                raise SchemaError(f"tok.w.{name}: bad widths for a {'ragged' if spec.ragged else 'fixed'} group")
        for k, (d, t, ragged) in spec_r.items():
            if ragged:
                rw = arrays.get(f"tok.rw.{k}")
                if rw is None or rw.dtype != np.int32 or rw.shape != (n_rows,) or np.any(rw < 0):
                    raise SchemaError(f"tok.rw.{k} not int32 [{n_rows}]")
                expect[f"tok.rw.{k}"] = (np.dtype(np.int32), (n_rows,))
                expect[f"tok.r.{k}"] = (d, (int(rw.sum()),) + t)
            elif t:
                raise SchemaError(f"token row spec {k}: a non-ragged key has no trailing shape")
            else:
                expect[f"tok.r.{k}"] = (d, (n_rows,))
    missing = [k for k in expect if k not in arrays]
    extra = [k for k in arrays if k not in expect]
    if missing or extra:
        raise SchemaError(f"shard keys: missing {missing[:5]}, unexpected {extra[:5]}")
    for k, (d, shape) in expect.items():
        a = arrays[k]
        if a.dtype != d:
            raise SchemaError(f"{k}: dtype {a.dtype} != {d}")
        if shape != (-1,) and a.shape != shape:
            raise SchemaError(f"{k}: shape {a.shape} != {shape}")
    off = arrays["seat_offsets"]
    if off[0] != 0 or off[-1] != n_rows or np.any(np.diff(off) < 1):
        raise SchemaError("seat_offsets must start at 0, end at n_rows and give every seat >= 1 row")
    if len(head.get("games", [])) != int(head["n_games"]) or len(head.get("seats", [])) != n_seats:
        raise SchemaError("header game / seat lists disagree with n_games / n_seats")
    g = arrays["seat.game"]
    if n_seats and (g.min() < 0 or g.max() >= int(head["n_games"])):
        raise SchemaError("seat.game out of range")
    r = arrays["seat.role"]
    if n_seats and (r.min() < 0 or r.max() >= len(ROLES)):
        raise SchemaError("seat.role out of range")
    hz = arrays["seat.hazard_dt"]
    if n_seats and (hz.min() < 0 or hz.max() >= len(HAZARD_DT_MODES)):
        raise SchemaError("seat.hazard_dt out of range")


def _validate_semantics(rd: "ShardReader") -> None:
    """The writer's invariants re-checked on read (SchemaError): identity, game records, rows, seat arrays vs
    the header seat list, per-seat counters / ticks / outcomes and the seat roles of every kept game."""
    h, path = rd.header, rd.path
    m = SHARD_RE.match(path.name)
    if m and int(m.group(1)) != int(h["worker"]):
        raise SchemaError(f"{path}: header worker {h['worker']} != the file name's")
    for f in ("code_digest", "config_digest", "policy_digest", "rollout_id"):
        if not isinstance(h.get(f), str) or not h[f]:
            raise SchemaError(f"{path}: header {f} missing")
    for kept, games in ((True, rd.games), (False, rd.dropped)):
        for g in games:
            try:
                g.check()
            except (ValueError, TypeError) as e:
                raise SchemaError(f"{path}: game {g.game_id!r}: {e}") from None
            if int(g.update) != int(h["update"]):
                raise SchemaError(f"{path}: game {g.game_id!r} update {g.update} != header {h['update']}")
            for f in ("code_digest", "config_digest", "policy_digest", "rollout_id"):
                if getattr(g, f) != h[f]:
                    raise SchemaError(f"{path}: game {g.game_id!r} {f} differs from the header")
            if kept and (g.drop_reason is not None or g.end != G.END_NATIVE or g.result is None):
                raise SchemaError(f"{path}: kept game {g.game_id!r} is not a regular finish with a result")
            if not kept and g.drop_reason is None:
                raise SchemaError(f"{path}: dropped game {g.game_id!r} has no drop reason")
    for ds in h.get("dropped_seats", []):
        if not 0 <= int(ds["dropped"]) < len(rd.dropped) or ds["role"] not in ROLES:
            raise SchemaError(f"{path}: dropped seat record {ds} out of range")
    try:
        rows = _check_rows(rd.rows)
    except ValueError as e:
        raise SchemaError(f"{path}: {e}") from None
    if rd.n_rows and (not rows["valid"].all() or rows["halted"].any()):
        raise SchemaError(f"{path}: a kept row is invalid or halted")
    if rd.n_rows and not np.array_equal(rd.arrays["tok.r.tick"].astype(np.int64), rows["tick"]):
        raise SchemaError(f"{path}: token row tick differs from row.tick")
    off = rd.offsets
    lengths = np.diff(off)
    seat = rd.seat
    ci = COUNTER_KEYS.index("committed")
    if not np.array_equal(seat["counters"][:, ci], lengths):
        raise SchemaError(f"{path}: seat counters['committed'] != the seats' row counts")
    if rd.n_rows > 1:
        step = np.diff(rows["tick"]) > 0
        step[off[1:-1] - 1] = True                                # seat boundaries
        if not step.all():
            raise SchemaError(f"{path}: ticks not strictly increasing within a seat")
    t = seat["temperature"]
    if rd.n_seats and not (np.all(np.isfinite(t)) and np.all((t >= T_MIN) & (t <= 1))):
        raise SchemaError(f"{path}: seat temperature outside [{T_MIN}, 1]")
    if rd.n_seats and not np.all(np.isfinite(seat["weight"]) & (seat["weight"] > 0)):
        raise SchemaError(f"{path}: seat weight must be > 0")
    by_game: Dict[int, List[int]] = {}
    for i, sm in enumerate(h["seats"]):
        gi, side, role = int(seat["game"][i]), int(seat["side"][i]), ROLES[int(seat["role"][i])]
        try:
            same = (int(sm["game"]) == gi and int(sm["side"]) == side and sm["role"] == role
                    and float(sm["weight"]) == float(seat["weight"][i])
                    and sm["hazard_dt"] == HAZARD_DT_MODES[int(seat["hazard_dt"][i])]
                    and [float(x) for x in sm["temperature"]] == t[i].tolist()
                    and [int(sm["counters"][k]) for k in COUNTER_KEYS] == seat["counters"][i].tolist())
        except (KeyError, TypeError, ValueError):
            same = False
        if not same:
            raise SchemaError(f"{path}: header seat {i} disagrees with the seat arrays")
        g = rd.games[gi]
        if float(seat["outcome"][i]) != _seat_view(_outcome(g.result), side, int(g.our_owner)):
            raise SchemaError(f"{path}: seat {i} outcome disagrees with game {g.game_id!r} result / our_owner")
        if int(rows["tick"][off[i + 1] - 1]) > int(g.end_tick):
            raise SchemaError(f"{path}: seat {i} has rows after game {g.game_id!r} end_tick")
        by_game.setdefault(gi, []).append(i)
    if sorted(by_game) != list(range(len(rd.games))):
        raise SchemaError(f"{path}: a kept game has no seat")
    for gi, idx in by_game.items():
        g = rd.games[gi]
        sides = [int(seat["side"][i]) for i in idx]
        if len(set(sides)) != len(sides):
            raise SchemaError(f"{path}: game {g.game_id!r} has two seats on one side")
        try:
            check_seat_roles(int(g.our_owner), g.opp_kind, sides, [ROLES[int(seat["role"][i])] for i in idx])
        except ValueError as e:
            raise SchemaError(f"{path}: game {g.game_id!r}: {e}") from None


class ShardReader:
    """Reads and validates one shard (``allow_pickle=False``): keys, dtypes and shapes, then the writer's
    semantic invariants. A corrupt or truncated file raises SchemaError (its cause is the zip error)."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        try:
            with open(self.path, "rb") as fh, np.load(fh, allow_pickle=False) as z:
                self.arrays: Dict[str, np.ndarray] = {k: z[k] for k in z.files}
        except (ValueError, zipfile.BadZipFile, EOFError, zlib.error) as e:   # pickle, corrupt / truncated zip
            raise SchemaError(f"{self.path}: unreadable shard ({type(e).__name__}: {e})") from e
        self.header = _header(self.arrays)
        try:
            validate_arrays(self.arrays, self.header)
            self.games = [GameMeta.from_dict(d) for d in self.header["games"]]
            self.dropped = [GameMeta.from_dict(d) for d in self.header["dropped"]]
            self.dropped_seats: List[Dict[str, Any]] = list(self.header["dropped_seats"])
            self.counters: Dict[str, Any] = self.header["counters"]
            self.offsets = self.arrays["seat_offsets"]
            self.rows = {k: self.arrays[f"row.{k}"] for k in ROW_FIELDS}
            self.seat = {k: self.arrays[f"seat.{k}"] for k in SEAT_FIELDS}
            _validate_semantics(self)
        except (KeyError, TypeError, AttributeError, IndexError) as e:
            raise SchemaError(f"{self.path}: malformed header ({e!r})") from None
        self._goff = {name: np.concatenate([[0], np.cumsum(self.arrays[f"tok.w.{name}"], dtype=np.int64)])
                      for name in S.GROUPS} if self.n_rows else {}
        self._roff = {k[7:]: np.concatenate([[0], np.cumsum(v, dtype=np.int64)])
                      for k, v in self.arrays.items() if k.startswith("tok.rw.")}

    @property
    def n_rows(self) -> int:
        return int(self.header["n_rows"])

    @property
    def n_seats(self) -> int:
        return int(self.header["n_seats"])

    def seat_game(self, i: int) -> SeatGame:
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        gi = int(self.seat["game"][i])
        return SeatGame(game=self.games[gi], seat=dict(self.header["seats"][i]),
                        rows={k: v[lo:hi] for k, v in self.rows.items()}, side=int(self.seat["side"][i]),
                        role=ROLES[int(self.seat["role"][i])], weight=float(self.seat["weight"][i]),
                        outcome=float(self.seat["outcome"][i]), _shard=self, _lo=lo, _hi=hi)

    def seat_games(self) -> Iterator[SeatGame]:
        for i in range(self.n_seats):
            yield self.seat_game(i)

    def tokens(self, lo: int, hi: int) -> Dict[str, Any]:
        """Tokens of rows ``[lo, hi)`` padded to their own maximum width (tokenizer fill values)."""
        if not 0 <= lo < hi <= self.n_rows:
            raise IndexError(f"rows [{lo}, {hi}) outside [0, {self.n_rows})")
        groups = {}
        for name in S.GROUPS:
            w = self.arrays[f"tok.w.{name}"][lo:hi]
            off = self._goff[name]
            groups[name] = {k: _unflatten(self.arrays[f"tok.g.{name}.{k}"], off, w, lo, hi, _group_fill(k))
                            for k in GROUP_KEYS}
        row = {}
        for key in self.header["token_row_spec"]:
            a = self.arrays[f"tok.r.{key}"]
            if key in self._roff:
                row[key] = _unflatten(a, self._roff[key], self.arrays[f"tok.rw.{key}"][lo:hi], lo, hi,
                                      ROW_TOKEN_FILL.get(key, 0))
            else:
                row[key] = a[lo:hi].copy()
        return {"groups": groups, "row": row, "profile": "engine"}


def _unflatten(flat: np.ndarray, off: np.ndarray, w: np.ndarray, lo: int, hi: int, fill: Any) -> np.ndarray:
    n = hi - lo
    wmax = int(w.max()) if n else 0
    out = np.full((n, wmax) + flat.shape[1:], fill, dtype=flat.dtype)
    a, b = int(off[lo]), int(off[hi])
    if b > a:
        r = np.repeat(np.arange(n), w)
        pos = np.arange(b - a) - np.repeat(off[lo:hi] - a, w)
        out[r, pos] = flat[a:b]
    return out


def _sum_counters(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, Mapping):
            _sum_counters(dst.setdefault(k, {}), v)
        else:
            dst[k] = dst.get(k, 0) + v


@dataclasses.dataclass
class UpdateData:
    """Every shard of one update: concatenated rows / seats plus the readers (tokens are read per seat-game).

    ``seat['game']`` indexes ``games`` (all workers' kept games, worker order); ``seat_offsets`` indexes the
    concatenated ``rows``; ``lengths`` feeds ``gae.gae_ragged``. ``counters['by_role']['ours']`` holds the
    our-seat counters the monitor's forced fraction / plays per minute use; ``dropped_seats[i]['dropped']``
    indexes ``dropped``."""
    update: int
    readers: List[ShardReader]
    games: List[GameMeta]
    dropped: List[GameMeta]
    dropped_seats: List[Dict[str, Any]]
    counters: Dict[str, Any]
    rows: Dict[str, np.ndarray]
    seat: Dict[str, np.ndarray]
    seat_offsets: np.ndarray
    code_digest: str
    config_digest: str
    policy_digest: str
    rollout_id: str

    @property
    def n_rows(self) -> int:
        return int(self.seat_offsets[-1])

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.seat_offsets)

    def seat_games(self) -> Iterator[SeatGame]:
        for r in self.readers:
            yield from r.seat_games()

    def seat_phi_end(self) -> np.ndarray:
        """``[n_seats]`` end-of-game Φ from each seat's own view (NaN when the game has none): the
        ``phi_end`` of ``gae.short_head_gae`` per seat-game."""
        out = np.full(self.seat["game"].shape[0], np.nan)
        for i, (gi, side) in enumerate(zip(self.seat["game"], self.seat["side"])):
            g = self.games[int(gi)]
            if g.phi_end is not None:
                out[i] = _seat_view(float(g.phi_end), int(side), int(g.our_owner))
        return out

    def drop_counter(self, alarm_rate: float = G.DROP_ALARM_RATE, stop_rate: float = G.DROP_STOP_RATE
                     ) -> G.DropCounter:
        """``gae.DropCounter`` over this update's games (kept and dropped, Φ at the drop)."""
        dc = G.DropCounter(alarm_rate=alarm_rate, stop_rate=stop_rate)
        for _ in self.games:
            dc.add(None)
        for g in self.dropped:
            dc.add(g.drop_reason, g.phi_drop)
        return dc


def load_update(root: Union[str, Path], update: int, *, expect_workers: Optional[int] = None,
                expect_games: Optional[int] = None, expect_policy_digest: Optional[str] = None,
                expect_rollout_id: Optional[str] = None, expect_code_digest: Optional[str] = None,
                expect_config_digest: Optional[str] = None) -> UpdateData:
    """Read every shard of ``update`` and concatenate them (SchemaError on any inconsistency).

    ``expect_workers``: exactly that many shards (w000..); ``expect_games``: kept + dropped games == N
    (every update uses exactly the N games it started). All shards must share one code / config
    digest, one ``policy_digest`` and one ``rollout_id`` (no stale server, no earlier attempt mixed in); the
    ``expect_*`` digests / id, when given, must match (the learner passes the sha256 of θ_k's model_ema)."""
    paths = shard_paths(root, update)
    if not paths:
        raise FileNotFoundError(f"no shards for update {update} under {root}")
    if expect_workers is not None and [p.name for p in paths] != [f"w{i:03d}.npz" for i in range(expect_workers)]:
        raise SchemaError(f"update {update}: shards {[p.name for p in paths]} != {expect_workers} workers")
    readers = [ShardReader(p) for p in paths]
    h0 = readers[0].header
    ident = ("code_digest", "config_digest", "policy_digest", "rollout_id")
    expect = dict(zip(ident, (expect_code_digest, expect_config_digest, expect_policy_digest, expect_rollout_id)))
    for f, v in expect.items():
        if v is not None and h0[f] != v:
            raise SchemaError(f"update {update}: {f} {h0[f]!r} != expected {v!r} ({readers[0].path})")
    ref = next((r for r in readers if r.n_rows), None)
    for r in readers:
        h = r.header
        if int(h["update"]) != int(update):
            raise SchemaError(f"{r.path}: header update {h['update']} != {update}")
        for f in ident:
            if h[f] != h0[f]:
                raise SchemaError(f"{r.path}: {f} {h[f]!r} differs from {readers[0].path} ({h0[f]!r})"
                                  + ("; shards of two attempts are mixed" if f == "rollout_id" else ""))
        if r.n_rows and ref is not None and h["token_row_spec"] != ref.header["token_row_spec"]:
            raise SchemaError(f"{r.path}: token row spec differs from {ref.path}")
    games: List[GameMeta] = []
    dropped: List[GameMeta] = []
    dropped_seats: List[Dict[str, Any]] = []
    counters: Dict[str, Any] = {}
    rows = {k: [] for k in ROW_FIELDS}
    seat = {k: [] for k in SEAT_FIELDS}
    offs = [np.zeros(1, np.int64)]
    base = 0
    for r in readers:
        for k in ROW_FIELDS:
            rows[k].append(r.rows[k])
        for k in SEAT_FIELDS:
            v = r.seat[k]
            seat[k].append(v + len(games) if k == "game" else v)
        offs.append(r.offsets[1:] + base)
        base += r.n_rows
        dropped_seats += [dict(ds, dropped=int(ds["dropped"]) + len(dropped)) for ds in r.dropped_seats]
        games += r.games
        dropped += r.dropped
        _sum_counters(counters, r.counters)
    ids = [g.game_id for g in games + dropped]
    if len(set(ids)) != len(ids):
        raise SchemaError(f"update {update}: duplicate game ids")
    if expect_games is not None and len(ids) != int(expect_games):
        raise SchemaError(f"update {update}: {len(ids)} games (kept + dropped) != expected {expect_games}")
    return UpdateData(update=int(update), readers=readers, games=games, dropped=dropped,
                      dropped_seats=dropped_seats, counters=counters,
                      rows={k: np.concatenate(v, 0) for k, v in rows.items()},
                      seat={k: np.concatenate(v, 0).astype(SEAT_FIELDS[k][0]) for k, v in seat.items()},
                      seat_offsets=np.concatenate(offs), code_digest=h0["code_digest"],
                      config_digest=h0["config_digest"], policy_digest=h0["policy_digest"],
                      rollout_id=h0["rollout_id"])


def mask_recompute_mismatch(sg: SeatGame) -> int:
    """Rows whose stored legality (from the actor's masked logits) differs from the stored tokens'
    ``slot_legal`` (AND the stored client slot-mask bits ``client_slot_ok``) / ``ability_ready`` (the input of
    the learner's periodic mask audit)."""
    tok = sg.tokens()["row"]
    slot = np.asarray(tok["slot_legal"]).astype(bool) & sg.rows["client_slot_ok"]
    ready = np.asarray(tok["ability_ready"]).astype(bool)
    bad = np.any(slot != sg.rows["legal_slots"], axis=-1) | (ready.any(-1) != sg.rows["ability_avail"])
    return int(bad.sum())


__all__ = ["SCHEMA_VERSION", "SCHEMA_DIGEST", "KEEP_UPDATES", "T_MIN", "ROLES", "SELF_SEAT_KIND", "MIRROR_KIND",
           "HAZARD_DT_MODES", "COUNTER_KEYS", "ROW_FIELDS", "SEAT_FIELDS", "GAME_FIELDS", "SchemaError",
           "GameMeta", "SeatRecord", "SeatGame", "ShardWriter", "ShardReader", "UpdateData", "load_update",
           "validate_arrays", "check_seat_roles", "actor_rows", "stack_peek_outputs", "update_dir", "shard_path",
           "list_updates", "shard_paths", "begin_update", "prune_updates", "schema_spec", "group_tails",
           "mask_recompute_mismatch"]
