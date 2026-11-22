"""Model input schema ``aminiformer-v1``.

This module is the single definition of

* the numpy structured dtypes of the training shard tables (``SHARD_SCHEMA``),
  used by the tokenizer, the trainer and the synthetic shard generator;
* the token groups, their float feature columns, categorical columns and link
  roles;
* the frozen encodings: the coordinate basis ``C(u)``, the time maps
  ``Tl``/``Tg``, absolute-time quantisation ``tq`` and the self-frame helpers.

Nothing here reads data.  The tokenizer fills the arrays
described by :data:`GROUPS`; ``aminiformer.AminiFormer`` embeds them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_ID = "aminiformer-v1"
SHARD_SCHEMA = "amini-shards-v7"

# ----------------------------------------------------------------- constants
TILE = 1000
ARENA_W, ARENA_H = 18000, 32000
GRID_X, GRID_Y = 18, 32
N_CELLS = GRID_X * GRID_Y                     # 576, cell = gy * 18 + gx (self frame)
ELIXIR_ONE = 10000                            # fixed point
ELIXIR_MAX = 100000
HAND_SLOTS = 4
DECK_SLOTS = 8
N_TOWERS = 6
ENT_CAP, OBJ_CAP = 128, 256
EVENT_CAP, PENDING_CAP, OPP_PENDING_CAP = 16, 4, 4
ABILITY_POINTER = 64                          # pointer action covers the first 64 ENT
LINK_CODES = 512
N_SELF_MAX = N_TOWERS + ENT_CAP + OBJ_CAP + EVENT_CAP + PENDING_CAP + OPP_PENDING_CAP  # 414
assert N_SELF_MAX < LINK_CODES

# sampling constants (example values), used by the hazard conversion and the calibration check:
# U_TICKS is the gate's hazard time unit in ticks (also the length of one decision block), KAPPA the
# per-stratum rate of committed-wait rows (base / threat).
U_TICKS = 20
KAPPA = {"base": 0.2, "threat": 0.5}

# temporal window (example values)
WINDOW_TICKS = 1000
WINDOW_ROWS = 64
TQ_STEP = 5                                   # tq = 5 * floor(tick / 5); RoPE p = tq / 5

# action families (R.kind)
FAMILY_WAIT, FAMILY_PLAY, FAMILY_ABILITY = 0, 1, 2
N_FAMILIES = 3
STRATUM_ACTION, STRATUM_BASE, STRATUM_THREAT = 0, 1, 2

# vocab defaults.  Real sizes come from the frozen vocab files; the model
# tables are sized by config and must be >= the vocab sizes.
CARD_VOCAB_V7 = 244                           # 0 pad, 1 unk, 2..181 cards, 182..243 forms
OBJECT_VOCAB_V7 = 767                         # 590 base objects + 177 catalog objects
N_FORMS = 3                                   # 0 base, 1 evolution, 2 hero
N_POPULATIONS = 8                             # index 0 = NULL (a learned embedding)
STATE_CAP = 32
FAMILY_CAP = 8

# frames: self slot k reads world slot SELF_SLOT[side][k]
SELF_SLOT = {0: (0, 1, 2, 3, 4, 5), 1: (3, 5, 4, 0, 2, 1)}
# self-frame tower slot constants (king, left princess, right princess; self then enemy)
TOWER_SLOT_XY = ((9000, 3000), (3500, 6500), (14500, 6500),
                 (9000, 29000), (3500, 25500), (14500, 25500))
TOWER_SLOT_IS_SELF = (1, 1, 1, 0, 0, 0)

INT16_MIN = -32768
INT16_MAX = 32767

# ----------------------------------------------------------------- dtypes
GAME_V7 = np.dtype([
    ("replay_id", "S12"), ("split", "u1"), ("method", "u1"), ("caster_resolution", "u1"),
    ("mode_recorded", "<u2"), ("mode_native", "<u2"), ("population", "u1"), ("seed", "<i4"),
    ("n_equivalent_plans", "u1"), ("six_hp_exact", "?"), ("outcome_exact", "?"),
    ("tower_hp_matches", "u1"), ("hp_l1_rel", "<f4"), ("hp_total_rel", "<f4"),
    ("rejected", "<u2"), ("not_reached", "<u2"), ("forced_deals", "<u2"),
    ("first_forced_deal", "<i2", (2,)),
    ("source_end", "<i2"), ("native_stop", "<i2"), ("real_end", "<i2"),
    ("post_end_rows_dropped", "<u2"),
    ("deck_card", "<i2", (2, 8)), ("deck_level", "i1", (2, 8)), ("deck_cost", "i1", (2, 8)),
    ("deck_el", "u1", (2, 8)),
    ("tower_obj", "<i2", (6,)), ("tower_uid", "<i4", (6,)), ("tower_max_hp", "<i4", (6,)),
    ("result", "i1", (2,)), ("final_crowns", "i1", (2,)),
    ("row_lo", "<i8", (2,)), ("row_hi", "<i8", (2,)),
    ("evt_lo", "<i8"), ("evt_hi", "<i8"), ("life_lo", "<i8"), ("life_hi", "<i8"),
    ("thp_lo", "<i8"), ("thp_hi", "<i8"), ("bur_lo", "<i8"), ("bur_hi", "<i8"),
    ("plan_sha256", "S32"), ("case_digest", "S32"), ("export_revision", "<u2"),
    # causal rule: bit o set = side o had a forced deal in this game
    # (first_forced_deal[o] >= 0), whether or not a form mismatch was later observed.  Training-only
    # switch read by the tokenizer to zero side o's form columns from the forced deal on
    # (FORM_MISMATCH_COLUMNS / OPP_FORM_MISMATCH_COLUMNS); never a policy input itself.
    ("form_mismatch", "u1"),
])

EVT_V7 = np.dtype([
    ("game", "<u4"), ("idx", "<i2"), ("owner", "u1"), ("kind", "u1"), ("tap", "<i2"),
    ("D", "<i2"), ("status", "i1"), ("deck", "i1"), ("card", "<i2"), ("card_vis", "<i2"),
    ("el_vis", "u1"), ("caster_obj", "<i2"), ("caster_uid", "<i4"), ("raw_x", "<i4"),
    ("raw_y", "<i4"), ("gx", "i1"), ("gy", "i1"), ("cost", "i1"), ("cost_paid", "i1"),
    ("pd_flags", "u1"), ("pd_src", "u1"), ("forced_deal", "u1"),
])

ROW_V7 = np.dtype([
    ("game", "<u4"), ("side", "u1"), ("seq", "u1"), ("tick", "<i2"), ("kind", "i1"),
    ("stratum", "u1"), ("weight", "<f4"), ("threat", "u1"), ("elixir", "<i4"),
    ("opp_elixir", "<i4"), ("crowns", "i1", (2,)), ("multiplier", "u1"),
    ("hand_deck", "i1", (4,)), ("hand_cost", "i1", (4,)), ("hand_cost_eng", "i1", (4,)),
    ("hand_flags", "u1", (4,)), ("hand_evo", "u1", (4,)), ("next_deck", "i1"),
    ("queue", "i1", (4,)), ("tower_hp", "<i4", (6,)), ("tower_flags", "u1", (6,)),
    ("lab_slot", "i1"), ("lab_deck", "i1"), ("lab_gx", "i1"), ("lab_gy", "i1"),
    ("lab_cost", "i1"), ("lab_ability_uid", "<i4"), ("lab_event", "<i2"),
    ("lab_status", "i1"), ("n_ent", "u1"), ("n_obj", "<u2"),
])

ENT_V7 = np.dtype([
    ("uid", "<i4"), ("obj", "<i2"), ("is_self", "u1"), ("x2", "<i2"), ("y2", "<i2"),
    ("vx", "<i2"), ("vy", "<i2"), ("z", "<i2"), ("vz", "<i2"), ("hp", "<i4"),
    ("max_hp", "<i4"), ("level", "i1"), ("state", "i1"), ("deploy_left", "<i2"),
    ("head_x", "<i2"), ("head_y", "<i2"), ("ab", "u1"), ("shot_target_uid", "<i4"),
    ("born_dt", "<i2"), ("spawn_event", "<i2"),
])

# OBJ.z / OBJ.vz are <i4: some projectile heights exceed the int16 range (OBJ.vz is widened with z
# because it is a difference of two heights).
OBJ_V7 = np.dtype([
    ("uid", "<i4"), ("obj", "<i2"), ("is_self", "u1"), ("family", "u1"),
    ("x2", "<i2"), ("y2", "<i2"), ("vx", "<i2"), ("vy", "<i2"), ("z", "<i4"), ("vz", "<i4"),
    ("tx2", "<i2"), ("ty2", "<i2"), ("target_uid", "<i4"), ("source_uid", "<i4"),
    ("area_left", "<i2"), ("countdown", "<i2"), ("eta", "<i2"), ("born_dt", "<i2"),
    ("spawn_event", "<i2"), ("flags", "u1"),
])

# LIFE.died (labels only): last_seen + 1 for an object removed before the end; -1 alive at the end of the
# recording; LIFE_DIED_EXCLUDED (-2) for an object without a true remaining life (removed at or after the
# battle-over tick, or alive at the tick-6001 arena clear): no resolution label.
# LIFE.gone (presence only): the first tick the object is absent (last_seen + 1) whenever it was
# removed, the -2 objects included; -1 alive at the end.  aux_threat's "alive at t" reads gone, its
# resolution window reads died (the tokenizer's target builder).
LIFE_DIED_ALIVE_AT_END = -1
LIFE_DIED_EXCLUDED = -2
LIFE_V7 = np.dtype([
    ("game", "<u4"), ("uid", "<i4"), ("obj_first", "<i2"), ("obj_last", "<i2"),
    ("owner", "u1"), ("family", "u1"), ("born", "<i2"), ("died", "<i2"), ("gone", "<i2"),
    ("bx", "<i2"), ("by", "<i2"), ("btx", "<i2"), ("bty", "<i2"),
    ("lx", "<i2"), ("ly", "<i2"), ("ltx", "<i2"), ("lty", "<i2"),
    ("src_birth", "<i4"), ("tgt_last", "<i4"), ("play_birth", "<i2"), ("spawn_event", "<i2"),
])

TOWERHP_V7 = np.dtype([("game", "<u4"), ("tick", "<i2"), ("hp", "<i4", (6,))])

BURROW_V7 = np.dtype([
    ("game", "<u4"), ("uid", "<i4"), ("tick", "<i2"), ("old_state", "i1"),
    ("new_state", "i1"), ("x", "<i2"), ("y", "<i2"),
])

PRIV_V7 = np.dtype([
    ("opp_hand", "i1", (4,)), ("opp_queue", "i1", (4,)),
    ("opp_el", "u1", (4,)), ("opp_evo", "u1", (4,)),
])

SHARD_TABLES: Dict[str, np.dtype] = {
    "games": GAME_V7, "events": EVT_V7, "rows": ROW_V7, "entity": ENT_V7,
    "object": OBJ_V7, "life": LIFE_V7, "towerhp": TOWERHP_V7, "burrow": BURROW_V7,
    "priv": PRIV_V7,
}

# own form columns zeroed for side o when GAME.form_mismatch bit o is set (bit o set iff side o
# had a forced deal, first_forced_deal[o] >= 0), causally from that forced deal on:
# HAND on rows with t > first_forced_deal[o] (the forced play's tap is already past), own PENDING /
# EVENT_RECENT entries of events with tap >= first_forced_deal[o]; the whole episode only for a
# flag without a recorded forced deal (first_forced_deal[o] == -1).  None of
# them has a separate mask column in this schema.  Own events of that side keep their card
# only as the base card.  The same switch applies to the *opponent's* synthetic events
# (EVT.pd_src == 2) when the opponent's bit is set: OPP_FORM_MISMATCH_COLUMNS, base card shown,
# DECK_OPP evolved_seen not raised by them.  Other opponent events (pd_src 0/1) stay untouched.
FORM_MISMATCH_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "HAND": ("evolved_if_played", "hero", "evo_counter"),       # hand_flags bit3/bit4, hand_evo
    "PENDING": ("evolved",),                                   # own EVT.el_vis bit0
    "EVENT_RECENT": ("evolved", "hero"),                       # own EVT.el_vis bits; own card_vis
}                                                              # is shown as its base card
OPP_FORM_MISMATCH_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "EVENT_RECENT": ("evolved", "hero"),                       # opponent pd_src == 2 events only
    "OPP_PENDING": ("evolved", "hero"),
    "DECK_OPP": ("evolved_seen",),
}


def form_mismatch_side(game: np.void, side: int) -> bool:
    """True if GAME.form_mismatch marks ``side``'s own form columns untrustworthy."""

    return bool((int(game["form_mismatch"]) >> int(side)) & 1)


def form_mismatch_from(game: np.void, side: int) -> Optional[int]:
    """None if ``side`` is not flagged, else the tick the form mask starts from:
    ``first_forced_deal[side]`` (HAND rows with t > it, events with tap >= it).  The shard writer sets
    bit o exactly when side o had a forced deal, so a flag normally has ``first_forced_deal >= 0``;
    a flag without one returns -1 (whole episode)."""

    if not form_mismatch_side(game, side):
        return None
    ffd = int(game["first_forced_deal"][int(side)])
    return ffd if ffd >= 0 else -1


# hand_flags bits
HF_AFFORDABLE, HF_LEGAL, HF_KNOWN, HF_EVOLVED, HF_HERO = 1, 2, 4, 8, 16
# ENT ab bits.  AB_READY is the unit's raw ability-ready state (an input feature).
# AB_LEGAL (bit 3) is the policy legality of the ability press (ready, no pending press of the same unit,
# enough elixir for the ability).  The tokenizer's ability mask reads AB_LEGAL only; it is never an input feature.
AB_BOUND, AB_READY, AB_SPAWN_ROOT, AB_LEGAL = 1, 2, 4, 8
# OBJ flags
OF_TARGET_VALID, OF_TARGET_UNCHANGED, OF_HOMING, OF_SPAWN_ROOT = 1, 2, 4, 8
# tower_flags
TF_PRESENT, TF_DESTROYED, TF_ACTIVATED = 1, 2, 4

# ----------------------------------------------------------------- play landing rule
# A card play tapped at t lands at t + PLAY_DISPATCH. The game environment refuses every landing once the
# battle is over. The battle is over at the landing, whatever happens in between, when the snapshot at t
# already has a king down (3 crowns), and also when the landing falls after the end of regulation
# (t + 20 > 3600) while the crowns at t are unequal: regulation ends with that lead unless the trailing side
# takes a tower in (t, 3600], and every landing after the end of overtime (t + 20 > 6000).
# Inputs: ROW.tick and ROW.crowns (derived from the tower HP fields). Applied by the
# tokenizer to the stored HF_LEGAL bit (stored shards keep their bit).
PLAY_DISPATCH = 20
PLAY_OVERTIME_TICK = 3600
PLAY_BATTLE_END_TICK = 6000
PLAY_LANDING_RULE = 1                          # revision of the tokenizer-time play landing rule


def play_landing_decided(tick: np.ndarray, crowns: np.ndarray) -> np.ndarray:
    """``[...]`` bool: a play tapped at ``tick`` lands after the battle is decided (``crowns``: ``[..., 2]``,
    either frame). ``max(crowns) >= 3``, or ``tick + PLAY_DISPATCH > PLAY_OVERTIME_TICK`` with unequal crowns, or
    ``tick + PLAY_DISPATCH > PLAY_BATTLE_END_TICK``."""

    t = np.asarray(tick).astype(np.int64)
    c = np.asarray(crowns).astype(np.int64)
    unequal = c[..., 0] != c[..., 1]
    landing = t + PLAY_DISPATCH
    return ((np.maximum(c[..., 0], c[..., 1]) >= 3) | ((landing > PLAY_OVERTIME_TICK) & unequal)
            | (landing > PLAY_BATTLE_END_TICK))


def hand_flags_at_landing(tick: np.ndarray, crowns: np.ndarray, hand_flags: np.ndarray) -> np.ndarray:
    """``[T, 4]`` int64 ROW.hand_flags with HF_LEGAL cleared on rows where :func:`play_landing_decided`."""

    hf = np.asarray(hand_flags).astype(np.int64)
    dec = play_landing_decided(tick, crowns)
    return np.where(dec[..., None], hf & ~HF_LEGAL, hf)

# ----------------------------------------------------------------- input rules (tokenizer-time)
# Named sets of tokenizer-time rules that travel with a model (AminiFormerConfig.input_rules, the tokenizer
# tables' input_rules).  "" = no extra rule.  "v6":
#   slot_rule "tap": a hand slot is legal only if :func:`tap_affordable` (slack_ticks = tap_slack_ticks) holds.
#     The stored HF_LEGAL bit is kept and ANDed, so both the HAND "legal" input and the slot mask follow the rule.
#   deploy_open_tick: no play before this decision tick.
# Inputs: ROW.tick, ROW.elixir, ROW.hand_cost, ROW.hand_deck and the own pending EVT costs.
# "v6_ability_tap": "v6" plus
#   ability_rule "tap": an ability press is legal only if :func:`ability_tap_ok` holds.  The stored AB_LEGAL bit
#     is kept and ANDed; an unknown ability cost is refused.  No ability before deploy_open_tick either.
# tap_slack_ticks and deploy_open_tick are example values.
INPUT_RULES: Dict[str, Dict[str, object]] = {
    "": {"slot_rule": "landing", "tap_slack_ticks": 0, "deploy_open_tick": 0, "ability_rule": "landing"},
    "v6": {"slot_rule": "tap", "tap_slack_ticks": 5, "deploy_open_tick": 100, "ability_rule": "landing"},
    "v6_ability_tap": {"slot_rule": "tap", "tap_slack_ticks": 5, "deploy_open_tick": 100, "ability_rule": "tap"},
}
MIN_PLAY_COST_KEYS = frozenset({"spiritempress", "mergemaiden", "mergemaidenmounted"})   # played at MIN_PLAY_COST
MIN_PLAY_COST = 3
ELIXIR_RATES = ((2400, 178), (4800, 357), (1 << 40, 537))     # fixed-point income per tick, per game phase


def input_rules(name: str) -> Dict[str, object]:
    if name not in INPUT_RULES:
        raise ValueError(f"unknown input_rules {name!r} (known: {sorted(INPUT_RULES)})")
    return INPUT_RULES[name]


def elixir_generated(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Fixed-point income of the steps t0 -> t1 (vectorised)."""

    def total(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t).astype(np.int64)
        out, lo = np.zeros_like(t), 0
        for hi, r in ELIXIR_RATES:
            out = out + r * np.clip(t - lo, 0, hi - lo)
            lo = hi
        return out

    return np.maximum(total(t1) - total(t0), 0)


def tap_affordable(tick: np.ndarray, elixir: np.ndarray, hand_cost: np.ndarray, pending_cost: np.ndarray,
                   min_cost3: np.ndarray, slack_ticks: int = 0) -> np.ndarray:
    """``[T, 4]`` bool: the tap-time rule per hand slot (slot_rule "tap" of INPUT_RULES), in fixed point
    ``min(ELIXIR_MAX, elixir + income(t, t + slack_ticks)) - pending_cost * ELIXIR_ONE >= cost * ELIXIR_ONE``
    (cost capped at MIN_PLAY_COST where ``min_cost3``).

    ``elixir`` [T] fixed point, ``hand_cost`` [T, 4] whole elixir (< 0 unknown: never affordable),
    ``pending_cost`` [T] whole elixir of the own pending commands, ``min_cost3`` [T, 4] bool (Spirit Empress)."""

    t = np.asarray(tick).astype(np.int64)
    hc = np.asarray(hand_cost).astype(np.int64)
    cost = np.where(np.asarray(min_cost3).astype(bool), np.minimum(hc, MIN_PLAY_COST), hc)
    have = np.minimum(ELIXIR_MAX, np.asarray(elixir).astype(np.int64) + elixir_generated(t, t + int(slack_ticks)))
    have = have - np.asarray(pending_cost).astype(np.int64) * ELIXIR_ONE
    return (hc >= 0) & (have[..., None] >= cost * ELIXIR_ONE)


def ability_tap_ok(elixir: np.ndarray, cost: np.ndarray, pending_cost: np.ndarray) -> np.ndarray:
    """Bool: the tap-time rule for an ability press (ability_rule "tap" of INPUT_RULES), in fixed point
    ``elixir - pending_cost * ELIXIR_ONE >= cost * ELIXIR_ONE``.  ``elixir`` fixed point, ``cost`` whole
    elixir of the unit's ability (< 0 unknown: refused), ``pending_cost`` whole elixir of the own pending
    commands; broadcast together."""

    c = np.asarray(cost).astype(np.int64)
    have = np.asarray(elixir).astype(np.int64) - np.asarray(pending_cost).astype(np.int64) * ELIXIR_ONE
    return (c >= 0) & (have >= c * ELIXIR_ONE)


# ----------------------------------------------------------------- encodings
COORD_KX = tuple(range(-1, 5))                # 13 dims for x
COORD_KY = tuple(range(-1, 6))                # 15 dims for y
COORD_X_DIM = 1 + 2 * len(COORD_KX)
COORD_Y_DIM = 1 + 2 * len(COORD_KY)
COORD_DIM = COORD_X_DIM + COORD_Y_DIM         # 28


def coord_basis(u: np.ndarray, ks: Sequence[int]) -> np.ndarray:
    """``C(u) = [u, sin(pi 2^k u), cos(pi 2^k u)]_k`` (float32, not clipped)."""

    u = np.asarray(u, dtype=np.float64)
    parts = [u[..., None]]
    for k in ks:
        a = math.pi * (2.0 ** k) * u
        parts.append(np.sin(a)[..., None])
        parts.append(np.cos(a)[..., None])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def coord_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``[C(x/18000), C(y/32000)]`` -> ``[..., 28]`` (positions in game units)."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.concatenate([coord_basis(x / ARENA_W, COORD_KX),
                           coord_basis(y / ARENA_H, COORD_KY)], axis=-1)


def cell_centres() -> Tuple[np.ndarray, np.ndarray]:
    """Self-frame tile centres of the 576 cells in cell order (``c = gy*18 + gx``)."""

    c = np.arange(N_CELLS)
    gx, gy = c % GRID_X, c // GRID_X
    return (gx + 0.5) * TILE, (gy + 0.5) * TILE


def cell_query_coords() -> np.ndarray:
    """``[576, 28]`` coordinate basis of the cell centres (same basis as the tokens)."""

    x, y = cell_centres()
    return coord_xy(x, y)


def tile_centre(v: np.ndarray) -> np.ndarray:
    """Quantise game coordinates to the centre of their tile (visual profile)."""

    return (np.floor(np.asarray(v, dtype=np.float64) / TILE) + 0.5) * TILE


def t_lin(dt: np.ndarray, s: float) -> np.ndarray:
    """``Tl(dt, S) = min(max(dt, 0), S) / S``."""

    return (np.clip(np.asarray(dt, dtype=np.float64), 0.0, s) / s).astype(np.float32)


def t_log(dt: np.ndarray, s: float) -> np.ndarray:
    """``Tg(dt, S) = log1p(max(dt, 0)) / log1p(S)``."""

    return (np.log1p(np.maximum(np.asarray(dt, dtype=np.float64), 0.0))
            / math.log1p(s)).astype(np.float32)


def lin_log(x: np.ndarray, s: float) -> np.ndarray:
    """``x / s`` inside [-1, 1], ``sign(x) * (1 + ln(|x| / s))`` outside (continuous, slope 1 at the knee, no clip).

    Used for the ENT / OBJ ``z`` and ``vz`` columns, whose values have a long tail."""

    y = np.asarray(x, dtype=np.float64) / float(s)
    a = np.abs(y)
    return np.where(a <= 1.0, y, np.sign(y) * (1.0 + np.log(np.maximum(a, 1.0)))).astype(np.float32)


def quantise_tick(tick: np.ndarray) -> np.ndarray:
    """``tq = 5 * floor(tick / 5)`` (int64)."""

    t = np.asarray(tick, dtype=np.int64)
    return TQ_STEP * np.floor_divide(t, TQ_STEP)


def rope_positions(tq: np.ndarray) -> np.ndarray:
    """RoPE positions ``p = tq / 5`` (float32)."""

    return (np.asarray(tq, dtype=np.float64) / TQ_STEP).astype(np.float32)


def to_view(x: np.ndarray, y: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
    """World <-> owner frame (an involution; point reflection for owner 1)."""

    if int(side) == 0:
        return x, y
    return ARENA_W - x, ARENA_H - y


def multiplier_of_tick(tick: np.ndarray) -> np.ndarray:
    t = np.asarray(tick)
    return np.where(t < 2400, 1, np.where(t < 4800, 2, 3))


# ----------------------------------------------------------------- token groups
LINK_ROLES = ("self", "target", "source", "shot_target", "spawn", "caster")
LINK_NONE, LINK_ABSENT = -1, -2

# categorical embedding tables: name -> (default size, padding_idx or None)
# Values are emitted by the tokenizer already shifted so that 0 means "none"
# where a padding index exists.
CAT_TABLES: Dict[str, Tuple[int, object]] = {
    "card": (256, 0),          # card id (0 pad); size >= CARD_VOCAB_V7
    "base": (256, 0),          # base card id of the card
    "form": (N_FORMS + 1, 0),  # form + 1
    "obj": (1024, 0),          # object id (0 pad, 1 unk); size >= OBJECT_VOCAB_V7
    "family": (FAMILY_CAP + 1, 0),
    "state": (STATE_CAP + 1, 0),
    "handslot": (HAND_SLOTS + 1, 0),
    "dslot": (DECK_SLOTS + 1, 0),   # pi[deck] + 1
    "tslot": (N_TOWERS + 1, 0),
    "pop": (N_POPULATIONS, None),  # 0 = NULL, learned
}


@dataclass(frozen=True)
class GroupSpec:
    name: str
    cap: int                                  # maximum tokens (fixed groups: exact)
    features: Tuple[str, ...]                 # float columns (coordinate blocks expanded)
    cats: Tuple[Tuple[str, str], ...]         # (column, table)
    roles: Tuple[str, ...]                    # link roles carried by the tokens
    ragged: bool = False                      # padded per batch (ENTITY, OBJECT)
    engine_only: bool = False                 # whole group E (dropped in the visual profile)
    always_valid: bool = False
    e_features: Tuple[str, ...] = ()          # columns zeroed in the visual profile
    e_roles: Tuple[str, ...] = ()             # link roles removed in the visual profile

    @property
    def n_features(self) -> int:
        return len(self.features)

    def column(self, name: str) -> int:
        return self.features.index(name)

    def columns(self, prefix: str) -> List[int]:
        return [i for i, f in enumerate(self.features) if f == prefix or f.startswith(prefix + "[")]


def _coord(prefix: str) -> Tuple[str, ...]:
    return tuple(f"{prefix}x[{i}]" for i in range(COORD_X_DIM)) + \
        tuple(f"{prefix}y[{i}]" for i in range(COORD_Y_DIM))


def _onehot(name: str, n: int) -> Tuple[str, ...]:
    return tuple(f"{name}[{i}]" for i in range(n))


GLOBAL = GroupSpec(
    "GLOBAL", 1,
    ("tq", "tq_left") + _onehot("mult", 3) + ("overtime", "e_self", "e_self_after_pending",
                                               "e_opp", "crowns_self", "crowns_opp",
                                               "play_count_self", "play_count_opp",
                                               "hand_known", "opp_cycle_known"),
    (("pop", "pop"),), (), always_valid=True, e_features=("e_opp",))

TOWER = GroupSpec(
    "TOWER", N_TOWERS,
    ("is_self",) + _coord("c") + ("hp_frac", "max_hp_log", "present", "destroyed", "activated"),
    (("tslot", "tslot"), ("obj", "obj")), ("self",), always_valid=True)

HAND = GroupSpec(
    "HAND", HAND_SLOTS,
    ("cost", "level", "is_spell", "affordable", "legal", "known", "evolved_if_played",
     "hero", "evo_counter", "pending_slot", "valid"),
    (("handslot", "handslot"), ("dslot", "dslot"), ("card", "card"), ("base", "base"),
     ("form", "form")), (), always_valid=True)

NEXT = GroupSpec(
    "NEXT", 1, ("cost", "level", "is_spell", "valid"),
    (("dslot", "dslot"), ("card", "card"), ("base", "base"), ("form", "form")), ())

DECK_SELF = GroupSpec(
    "DECK_SELF", DECK_SLOTS,
    ("cost", "level", "is_spell") + _onehot("in_hand", 5) + _onehot("queue_pos", 5)
    + ("is_next", "plays"),
    (("dslot", "dslot"), ("card", "card"), ("base", "base"), ("form", "form")), ())

DECK_OPP = GroupSpec(
    "DECK_OPP", DECK_SLOTS,
    ("revealed", "plays", "since_last", "evolved_seen") + _onehot("derived_in_hand", 3)
    + _onehot("derived_queue_pos", 5) + ("derived_is_next",),
    (("card", "card"), ("base", "base"), ("form", "form")), ())

EVENT_RECENT = GroupSpec(
    "EVENT_RECENT", EVENT_CAP,
    ("is_self", "is_ability", "age_lin", "age_log") + _coord("c")
    + ("evolved", "hero", "status_accepted", "status_rejected", "cost"),
    (("card", "card"), ("base", "base"), ("form", "form"), ("caster_obj", "obj")),
    ("self", "caster"), e_features=_coord("c"))   # sub-tile E: tile centres in visual

PENDING = GroupSpec(
    "PENDING", PENDING_CAP,
    ("due_in", "is_ability") + _coord("c") + ("cost", "evolved"),
    (("card", "card"), ("base", "base"), ("form", "form"), ("caster_obj", "obj"),
     ("dslot", "dslot")),
    ("self", "caster"))

OPP_PENDING = GroupSpec(
    "OPP_PENDING", OPP_PENDING_CAP,
    ("due_in", "since_tap", "is_ability") + _coord("c") + ("cost", "evolved", "hero"),
    (("card", "card"), ("base", "base"), ("form", "form"), ("caster_obj", "obj")),
    ("self", "caster"), engine_only=True)

ENTITY = GroupSpec(
    "ENTITY", ENT_CAP,
    ("is_self",) + _coord("c") + ("z", "vx", "vy", "vz", "move_speed", "hp_frac", "max_hp_log",
                                  "level", "deploy_left", "head_x", "head_y", "ability_capable",
                                  "ability_ready", "hp_known", "is_visual_only", "born",
                                  "shot_target_valid", "spawn_root"),
    (("obj", "obj"), ("family", "family"), ("state", "state")),
    ("self", "shot_target", "spawn"), ragged=True,
    e_features=_coord("c") + ("z", "deploy_left", "shot_target_valid", "spawn_root"),
    e_roles=("shot_target", "spawn"))

OBJECT = GroupSpec(
    "OBJECT", OBJ_CAP,
    ("is_self",) + _onehot("family", 4) + _coord("c") + ("z", "vx", "vy", "vz") + _coord("t")
    + ("rel_tx", "rel_ty", "target_valid", "target_unchanged", "homing",
       "eta_lin", "eta_log", "eta_valid", "area_lin", "area_log", "area_valid",
       "cd_lin", "cd_log", "cd_valid", "radius", "radius_known", "born", "spawn_root"),
    (("obj", "obj"), ("family", "family")),
    ("self", "target", "source", "spawn"), ragged=True,
    e_features=("z",) + _coord("t") + ("rel_tx", "rel_ty", "target_unchanged", "eta_lin",
                                       "eta_log", "eta_valid", "area_lin", "area_log",
                                       "area_valid", "cd_lin", "cd_log", "cd_valid",
                                       "spawn_root"),
    e_roles=("target", "source", "spawn"))

GROUP_ORDER: Tuple[str, ...] = ("GLOBAL", "TOWER", "HAND", "NEXT", "DECK_SELF", "DECK_OPP",
                                "EVENT_RECENT", "PENDING", "OPP_PENDING", "ENTITY", "OBJECT")
GROUPS: Dict[str, GroupSpec] = {g.name: g for g in (GLOBAL, TOWER, HAND, NEXT, DECK_SELF,
                                                     DECK_OPP, EVENT_RECENT, PENDING,
                                                     OPP_PENDING, ENTITY, OBJECT)}
# order in which self-link codes are handed out inside a row
SELF_LINK_ORDER: Tuple[str, ...] = ("TOWER", "ENTITY", "OBJECT", "EVENT_RECENT", "PENDING",
                                    "OPP_PENDING")
# sub-tile coordinate columns that the visual profile quantises to tile centres
# instead of zeroing (position is on screen at tile resolution)
VISUAL_TILE_QUANTISED = {"EVENT_RECENT": "c", "ENTITY": "c"}

# privileged critic input (PRIV, never a policy input)
PRIV_CARD_COLUMNS = 8                         # opp hand 4 + opp queue 4 (card ids)
PRIV_SCALAR_COLUMNS = 9                       # opp_el bits (4) + opp_evo/3 (4) + priv_valid
PRIV_DIM = PRIV_CARD_COLUMNS + PRIV_SCALAR_COLUMNS


# ---------------------------------------------------------------- placement-cell legality
# Legal play tiles per card class as a function of public, causal state only (the card's rule class and
# which princess towers are destroyed at t), in the SELF frame (cell = gy * 18 + gx; own half gy 0..14,
# river 15..16, enemy half 17..31). The troop grid (TROOP_GRID_ROWS, rows 0..14) holds the tiles where a
# troop is placed exactly at the clicked tile.
#   CELL_OWN  (troops, buildings, own-side spells such as Heal): troop grid rows 0..14; a fallen own
#             princess frees its 3x3 footprint; a fallen enemy princess opens its lane (cols 0..8 / 9..17):
#             the bridge tile column (gx 3 / 14) on rows 15..16 and the pocket rows 17..20 (the reflected
#             troop grid, i.e. without the two river-bank corners of row 17).
#   CELL_HALF (spells deployed like troops that may also cover buildings, own side only: The Log, Barbarian
#             Barrel, Royal Delivery): every tile of rows 0..14 (towers and corners included); an open lane
#             adds its whole river (rows 15..16, all lane columns) and all its pocket columns.
#   CELL_BOTH (troops / buildings that may be deployed on the enemy side: Miner, Goblin Drill): the troop
#             grid on both halves, both bridge columns always, every fallen princess footprint.
#   CELL_ANY  (spells that may target the enemy side): all 576 tiles.
# A mask id is ``class_bits * 16 + tower_bits``; class bits may be OR-ed (the bank is the union).
# Mirror is resolved per row to the class of the card it would copy (by the tokenizer).
CELL_OWN, CELL_HALF, CELL_BOTH, CELL_ANY = 1, 2, 4, 8
CELL_MIRROR = 16                       # card-table marker only, never inside a mask id
CELL_TOWER_STATES = 16                 # bit0 own-left, bit1 own-right, bit2 enemy-left, bit3 enemy-right down
N_CELL_MASKS = 16 * CELL_TOWER_STATES  # 256
CELL_TOWER_SLOTS = (1, 2, 4, 5)        # self tower slots behind the four tower bits (TOWER_SLOT_XY order)
CELL_RULES_VERSION = "cell-rules-v7.r1"
TROOP_GRID_ROWS = (                    # troop grid rows 0..14 (self frame)
    "......######......",
    "#######....#######",
    "#######....#######",
    "#######....#######",
    "#######....#######",
    "##...########...##",
    "##...########...##",
    "##...########...##",
    "##################",
    "##################",
    "##################",
    "##################",
    "##################",
    "##################",
    ".################.",
)
CELL_PRINCESS_TILES = ((3, 6), (14, 6), (3, 25), (14, 25))   # (gx, gy) centres, CELL_TOWER_SLOTS order
CELL_BRIDGE_COLS = (3, 14)             # left / right lane bridge tile column (rows 15..16)
CELL_POCKET_ROWS = (17, 21)            # [lo, hi) rows opened in a lane by its fallen enemy princess
# vocab base keys whose class differs from the default (spell -> ANY, else OWN)
# (the vocab carries both "log" / "thelog" and "barblog" / "barbarianbarrel")
CELL_CLASS_BY_KEY = {"log": CELL_HALF, "thelog": CELL_HALF, "barblog": CELL_HALF, "barbarianbarrel": CELL_HALF,
                     "royaldelivery": CELL_HALF, "miner": CELL_BOTH, "goblindrill": CELL_BOTH,
                     "darkelixirbottle": CELL_BOTH, "globallightning": CELL_OWN, "mirror": CELL_MIRROR}


def card_cell_class(base_key: Optional[str], is_spell: bool) -> int:
    """Placement class of a card (by its BASE vocab key; forms share their base's rule)."""

    if base_key in CELL_CLASS_BY_KEY:
        return CELL_CLASS_BY_KEY[str(base_key)]
    return CELL_ANY if is_spell else CELL_OWN


def cell_tower_bits(tower_flags: np.ndarray) -> np.ndarray:
    """``[..., 6]`` self-slot ROW.tower_flags -> ``[...]`` int64 tower bits (princess destroyed)."""

    tf = np.asarray(tower_flags).astype(np.int64)
    bits = np.zeros(tf.shape[:-1], np.int64)
    for b, k in enumerate(CELL_TOWER_SLOTS):
        bits |= ((tf[..., k] & TF_DESTROYED) != 0).astype(np.int64) << b
    return bits


def _troop_grid() -> np.ndarray:
    g = np.zeros((GRID_Y, GRID_X), bool)
    g[:15] = np.array([[c == "#" for c in r] for r in TROOP_GRID_ROWS], bool)
    return g


def cell_grid(class_bits: int, tower_bits: int) -> np.ndarray:
    """``[32, 18]`` legal tiles (self frame) of the union of ``class_bits`` under ``tower_bits``."""

    troop = _troop_grid()
    reflected = troop[::-1, ::-1]
    out = np.zeros((GRID_Y, GRID_X), bool)
    lanes = (slice(0, GRID_X // 2), slice(GRID_X // 2, GRID_X))
    p_lo, p_hi = CELL_POCKET_ROWS

    def footprint(g: np.ndarray, k: int) -> None:
        cx, cy = CELL_PRINCESS_TILES[k]
        g[cy - 1:cy + 2, cx - 1:cx + 2] = True

    if class_bits & CELL_ANY:
        out[:] = True
    for cls in (CELL_OWN, CELL_HALF):
        if not class_bits & cls:
            continue
        g = troop.copy() if cls == CELL_OWN else np.zeros_like(troop)
        if cls == CELL_HALF:
            g[:15] = True
        for k in (0, 1):
            if tower_bits >> k & 1:
                footprint(g, k)
        for lane in (0, 1):
            if tower_bits >> (2 + lane) & 1:
                if cls == CELL_OWN:
                    g[15:17, CELL_BRIDGE_COLS[lane]] = True
                else:                                   # may be placed on the river: the lane's whole river
                    g[15:17, lanes[lane]] = True
                pocket = reflected[p_lo:p_hi, lanes[lane]] if cls == CELL_OWN else True
                g[p_lo:p_hi, lanes[lane]] |= pocket
        out |= g
    if class_bits & CELL_BOTH:
        g = troop | reflected
        g[15:17, list(CELL_BRIDGE_COLS)] = True
        for k in range(4):
            if tower_bits >> k & 1:
                footprint(g, k)
        out |= g
    return out


_CELL_BANK: Optional[np.ndarray] = None


def cell_mask_bank() -> np.ndarray:
    """``[256, 576]`` bool, row ``class_bits * 16 + tower_bits`` (read-only, built once)."""

    global _CELL_BANK
    if _CELL_BANK is None:
        bank = np.zeros((N_CELL_MASKS, N_CELLS), bool)
        for cb in range(16):
            for tb in range(CELL_TOWER_STATES):
                bank[cb * CELL_TOWER_STATES + tb] = cell_grid(cb, tb).reshape(-1)
        bank.setflags(write=False)
        _CELL_BANK = bank
    return _CELL_BANK


def group_cap(name: str) -> int:
    return GROUPS[name].cap


def describe() -> Dict[str, object]:
    """JSON summary used in checkpoints and reports."""

    return {
        "schema": SCHEMA_ID,
        "groups": {n: {"cap": g.cap, "features": g.n_features, "cats": [c for c, _ in g.cats],
                       "roles": list(g.roles), "ragged": g.ragged,
                       "engine_only": g.engine_only} for n, g in GROUPS.items()},
        "link_codes": LINK_CODES, "n_self_max": N_SELF_MAX,
        "window_ticks": WINDOW_TICKS, "window_rows": WINDOW_ROWS, "tq_step": TQ_STEP,
        "coord_dim": COORD_DIM,
    }


__all__ = [n for n in dir() if n.isupper() or n in (
    "GroupSpec", "coord_basis", "coord_xy", "cell_centres", "cell_query_coords", "tile_centre",
    "t_lin", "t_log", "quantise_tick", "rope_positions", "to_view", "multiplier_of_tick",
    "group_cap", "describe", "form_mismatch_side", "form_mismatch_from", "card_cell_class",
    "input_rules", "elixir_generated", "tap_affordable", "ability_tap_ok",
    "cell_tower_bits", "cell_grid", "cell_mask_bank")]
