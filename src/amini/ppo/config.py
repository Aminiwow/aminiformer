"""PPO run configuration for aminiformer v7.

One tree of frozen dataclasses, ``PPOConfig``. A run file (YAML) may spell out every field; missing keys take
the defaults. Loading:

* ``load_config(path, overrides)``: the text is linted first (a repeated key in one mapping, or a bare
  scalar that PyYAML and the il_v7 subset read differently, e.g. ``NULL``, is a ConfigError), then read by
  ``amini.train.il_v7.load_yaml`` (it does not need PyYAML, so the file stays inside that parser's subset,
  see ``YAML_SUBSET``), then ``a.b=value`` overrides, then strict building: unknown keys are refused at every
  level, types are checked, ranges and cross-field rules are checked (``PPOConfig.validate``).
* ``missing(stage)`` / ``require_ready(stage)``: the open choices and measured values that must be filled in
  before rollouts (``"rollout"``) or before policy updates (``"policy"``).
  Pass ``population_input=model_population_input(model_config)`` of the base checkpoint (or a mapping per
  model of ``opponents.checkpoint``): ``levels.population.11`` is required only while a model that plays
  reads the population token (a model config may set ``population_input: false``).
* ``describe()``: a JSON-ready record for run directories (config, digest, derived values, readiness,
  invariants); ``digest()``: sha256 of the canonical JSON of the resolved config. Both re-validate, so a
  config whose dicts were mutated after construction is refused (attributes are frozen).

Shared names with the sibling modules (one source of truth, passed unchanged):

* opponent kinds are the case builder's (``self_seat``, ``mirror``, ``il_v4_t1``, ``il_v4_tstar``,
  ``il_v3_t1``, ``snapshot``), so ``cfg.effective_mix(phase)`` feeds its mix check and update planner;
* level keys are the ints 11 / 16, so ``cfg.levels.mix`` feeds the planner's level mix; the king level
  ``kt`` of a game is its uniform level (``king_level``), never a free choice;
* ``user.tower_troop`` is the game gid that the case builder writes into a game's input;
* ``monitor_thresholds()`` is exactly the field set and units of ``amini.ppo.monitor.Thresholds``;
* ``opponent_settings()`` rows are exactly the case builder's ``OpponentSpec`` fields (checkpoint,
  temperature, gate_bias [2, 3], hazard_dt, code tree), all from config fields, so the digest covers them;
* ``chain_args()`` is the block chain script's ``--timeout / --kill-after`` budget (``chain`` section);
* ``checkpoint_pins()`` / ``opponent_pins()``: the pinned sha256 of the frozen IL checkpoint files
  (``opponents.checkpoint_sha256``; the learner verifies them on load and in the shard metadata);
* ``config_change(old, new, update=, reason=)`` / ``with_changes(cfg, {key: value})``: the only way a run's
  config changes mid-run (``RUNTIME_CHANGE_KEYS``: beta, lr, lambda_tick, games per update, critic epochs and
  the calibrated early-stop / alarm thresholds); ``Learner.from_checkpoint`` accepts a new digest only through
  that record;
* ``policy.micro_rows`` / ``policy.max_micro_tokens``: the learner's micro-batch caps; ``drift``: the drift
  evaluator's thresholds (``amini.ppo.drift``).

Checkpoint keys and their roles (``MODEL_ROLES``): ``il_v4`` = the BASE IL model (theta's init, the KL reference
until the first promotion, the training IL opponent at T = 1 and at T*_IL), ``il_v2_heldout`` = the HELD-OUT IL
gate opponent (never trained against, no gate bias), ``il_v3`` / ``il_v3_t1`` = an extra IL slot (no share by
default).

Client timing: the env parameters of the seats that stand for a live client (``rollout.client_rules_seats``):
``rollout.landing_delay`` (decision-to-stamp delay pmfs), ``rollout.stamp_rule`` (a stamp on a blocked
tick % 10 phase moves to a later phase), ``rollout.arrival`` (batched opponent-command arrival, j drawn once
per game; the seat reads its OWN command's real stamp by the same rule with the same j,
:meth:`PPOConfig.own_sighting_record`), ``rollout.min_tap_tick``, ``sim2real.delta_lag`` (always 0) and
``rollout.client_mask`` (an optional client slot mask, sampling only). These form :meth:`PPOConfig.env_record` /
``env_digest()``; the model compatibility digest must not contain them (:func:`model_tables_digest`).
``rollout.client_mask`` is recorded with the env parameters but must ship with the policy
(``describe()['invariants']['client_mask_deploy']``). The early-stop and alarm thresholds that are calibrated
during a run are runtime changes (:data:`RUNTIME_CHANGE_KEYS`, :data:`RUNTIME_THRESHOLD_KEYS`,
``monitor.Monitor.apply_thresholds``; an operator change of a running run is :func:`runtime_change`).

Defaults: the numeric defaults (learning rates, coefficients, clip values, KL targets, batch sizes, opponent
mix, gate sizes and thresholds, monitor thresholds, client timing, budgets) are example values that pass
validation, not tuned settings; a run file sets its own. Open choices are switches in ``UserConfig``.
Temperatures use 0.0 for greedy; ``sampling_mode(t)`` translates that for ``logprob.apply_temperature``,
which only takes T > 0.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, get_args, get_origin, get_type_hints

from amini.ppo import gae as G

CONFIG_SCHEMA = "ppo-v7-config.v3"
READINESS_STAGES = ("rollout", "policy")
PHASES = ("pilot", "main")
CODE_ROOT = Path(__file__).resolve().parents[3]           # src/amini/ppo/config.py -> code root
OBJECT_VOCAB = "data/object_vocab_v7.json"                # delta_lag override names are its 'ids' keys

# hazard interval modes of the env adapter (copied, not imported)
HAZARD_DT_MODES = ("block", "interval")
# the vocabulary's POPULATIONS (copied: the config must not depend on vocab internals)
POPULATIONS = ("NULL", "ranked", "ladder", "challenge_eventdeck", "tournament", "friendly", "other")
POPULATION_L11_CHOICES = ("NULL", "tournament", "challenge_eventdeck")   # level-11 population choices
LEVELS = (16, 11)                                          # only all-11 or all-16 games

# opponent kinds, named as the case builder names them. The first two are theta_k seats (training data), the
# rest are frozen.
OPPONENT_KINDS = ("self_seat", "mirror", "il_v4_t1", "il_v4_tstar", "il_v3_t1", "snapshot")
IL_KINDS = ("il_v4_t1", "il_v4_tstar", "il_v3_t1")
SELFPLAY_KINDS = ("self_seat", "mirror", "snapshot")
FROZEN_TEMPERATURE_KINDS = ("il_v4_t1", "il_v4_tstar", "il_v3_t1", "snapshot", "il_v2_heldout")
CHECKPOINT_KINDS = ("il_v4", "il_v3", "il_v2_heldout")
# per-opponent settings: per frozen model gate_bias (all three), hazard_dt and code tree (the models that are not
# theta_k's own lineage; il_v4 and snapshots use rollout.hazard_dt and this run's code tree)
GATE_BIAS_MODELS = ("il_v4", "il_v3", "il_v2_heldout")
OLD_TREE_MODELS = ("il_v3", "il_v2_heldout")
MODEL_OF_KIND = {"il_v4_t1": "il_v4", "il_v4_tstar": "il_v4", "il_v3_t1": "il_v3", "snapshot": "snapshot",
                 "il_v2_heldout": "il_v2_heldout"}
# the role each checkpoint key plays
MODEL_ROLES: Dict[str, str] = {
    "il_v4": "base IL model: theta init, KL reference until the first promotion, training IL opponent "
             "at T = 1 (il_v4_t1) and at T*_IL (il_v4_tstar)",
    "il_v3": "extra IL slot: unused while il_v3_t1 has no share in the mixes",
    "il_v2_heldout": "held-out IL gate opponent: never trained against, no gate bias, T = 1"}
RUN_CODE_TREE = "run"                                     # this run's frozen code copy
N_FAMILIES = 3                                            # gate families (wait, play, ability): gate_bias rows
# the case builder's tower-troop gids and card keys (copied: tools/ is not importable from src/)
TOWER_TROOP_GIDS: Dict[int, str] = {159000000: "tower-princess", 159000001: "cannoneer",
                                    159000002: "dagger-duchess", 159000004: "royal-chef"}
PRINCESS_TOWER_GID = 159000000                            # the tower-princess gid
# rollout.landing_delay: decision-to-stamp delay pmfs in ticks, optionally smoothed with a uniform
# U{0..uniform_max}. The lead D - tap = DISPATCH_LEAD must stay above every delay (g <= MAX_LANDING_DELAY).
ALT_PLAY_PMF: Dict[int, float] = {0: 1.0, 1: 1.0, 2: 1.0}   # another example play pmf (not a default)
ABILITY_PMF: Dict[int, float] = {1: 1.0, 2: 1.0}               # example ability pmf: U{1..2}
MAX_LANDING_DELAY = 19
LANDING_KINDS = ("play", "ability")
# ---- client timing defaults (example values): rollout.landing_delay / stamp_rule / min_tap_tick / arrival
G12A_PLAY_BASE_PMF: Dict[int, float] = {0: 1.0, 1: 2.0, 2: 1.0}   # play delay g before the stamp rule
G12A_PLAY_BASE_N, G12A_PLAY_PUSHED_N = 4, 0           # total weight of that pmf; samples moved by the stamp rule
# no uniform smoothing by default: uniform_max / uniform_kinds stay as the inactive switch (with uniform_mix 0
# they do not change any pmf)
G12A_UNIFORM_MIX, G12A_UNIFORM_MAX = 0.0, 2
# stamp rule: a click whose stamp would land on a blocked tick % 10 phase is stamped on the next tick with
# tick % 10 == the push phase
G12A_STAMP_BLOCKED_MOD10: Tuple[int, ...] = (5,)
G12A_STAMP_PUSH_TO_MOD10 = 6
G12A_MIN_TAP_TICK = 100                              # earliest accepted decision tick of a client seat
# batched opponent-command arrival: first_seen = period * (stamp // period + 1) + j, j drawn once per game
ARRIVAL_MODELS = ("batched", "table")                # 'table': legacy per-command draw (sim2real.delta_arr_table)
ARRIVAL_PERIOD = 5
G12A_SUMMARY = "example"                             # name of the source the j pmf came from (recorded only)
G12A_SUMMARY_SHA256: Optional[str] = None            # sha256 of that source file (None = no file)
# per-battle histograms of the arrival offset j (example data; j_pmf_from_battles averages them)
G12A_J_BY_BATTLE: Dict[str, Dict[int, int]] = {"example/b1": {1: 2, 2: 2}, "example/b2": {1: 1, 2: 3}}
CLIENT_SEAT_ROLES = ("train", "opponent", "gate")    # seats that stand for a live client (client timing applies)
# What each role of rollout.client_rules_seats covers. The env and gate tasks apply the rules symmetrically.
CLIENT_SEAT_ROLE_DOC: Dict[str, str] = {
    "train": "every theta_k seat of a training game (self_seat and mirror: both seats)",
    "opponent": "every frozen seat of a training game (base IL at T = 1 / T*_IL, snapshots, extra IL slot)",
    "gate": "BOTH seats of every scored gate game (candidate and gate opponent: champion, held-out IL, base IL) "
            "and both seats of every canary game (gates.canaries), so a gate never compares a seat held to the "
            "client rules with one that is not"}
# Own-command sighting. Our own PENDING stays visible to the model from the decision tick t (registered with
# D = t + DISPATCH_LEAD, as the tokenizer shows it for tap < t <= D); the sighting only decides WHEN the seat reads
# the command's real stamp and corrects D to stamp + DISPATCH_LEAD. Rule by rollout.arrival.model:
#   'batched' -> OWN_SIGHTING_BATCHED: first_seen = own_first_seen(stamp, j) = batched_first_seen(stamp, j, period),
#                with the SAME per-game j the observing seat draws for opponent arrivals;
#   'table'   -> OWN_SIGHTING_R0_RANGE: the legacy draw U{OWN_SIGHTING_R0} after the stamp.
# The env requires stamped g + sighting < DISPATCH_LEAD (the client sees its own command's stamp before it would
# judge it rejected at D): PPOConfig.own_sighting_delay_range, checked in validation.
OWN_SIGHTING_BATCHED = "batched_same_j"
OWN_SIGHTING_R0_RANGE = "r0_range"
OWN_SIGHTING_MODEL_OF_ARRIVAL: Dict[str, str] = {"batched": OWN_SIGHTING_BATCHED, "table": OWN_SIGHTING_R0_RANGE}
# own-command sighting range of the legacy 'table' model only (the rollout's default, copied: tools/ is not
# importable)
OWN_SIGHTING_R0: Tuple[int, int] = (2, 8)
G12A_OWN_ARRIVAL_RANGE: Tuple[int, int] = (2, 7)     # own sighting delay range of the default timing
MAX_ENV_SLOTS_CEILING = 64       # hard ceiling of rollout.max_env_slots = concurrent games (rollout's MAX_ENV_SLOTS)
# why sim2real.delta_lag must be 0 (the validator's message; the episode settings refuse a nonzero lag too)
DELTA_LAG_G12A = ("the env models no display lag: a nonzero lag shifts born ticks and stalls seats in the "
                  "self-check, so delta_lag is 0 everywhere")
DISPATCH_LEAD = 20
ENV_SCHEMA = "ppo-v7-env.1"
# tokenizer tables (il_v7._tables_digest_map logical names) that are ENV parameters, not model ones (the arrival
# table used at load time never makes a checkpoint incompatible); see model_tables_digest / env_record
ENV_TABLE_KEYS = ("arrival",)
G2_MODES = ("pooled_since_restart", "latest")        # G2 trend: pooled since the last restart, or the latest G2
CANARY_KINDS = ("all_wait", "random_legal")               # gates.canaries: diagnostic only
GENERAL_OPPONENTS = ("self", "mix")                       # opponents.general_opponents
PANEL_SOURCES = ("deck", "general")                       # gates.panel_source
# A fixed frozen model named in the config: "NAME|/absolute/path.pt|<sha256 of the file>". Used by
# ``opponents.snapshot_anchors`` (always in the snapshot pool) and ``gates.g2_anchor`` (the record-only G2's
# opponent instead of the IL base).
ANCHOR_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
ANCHOR_RESERVED = ("snap-u", "champ-u", "champion-u", "cand-u")


def parse_anchor(text: Any) -> Dict[str, str]:
    """``{'name', 'path', 'sha256'}`` of one anchor entry, or ValueError saying what is wrong with it."""
    parts = [x.strip() for x in str(text).split("|")]
    if len(parts) != 3:
        raise ValueError(f"{text!r}: an anchor is NAME|PATH|SHA256")
    name, path, sha = parts
    if not ANCHOR_NAME_RE.match(name) or name.startswith(ANCHOR_RESERVED):
        raise ValueError(f"{text!r}: name {name!r} (lower case letters, digits and '-', not {ANCHOR_RESERVED})")
    if not path.startswith("/") or not path.endswith(".pt"):
        raise ValueError(f"{text!r}: path must be an absolute .pt file")
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError(f"{text!r}: sha256 must be 64 hex digits (the pin of the file)")
    return {"name": name, "path": path, "sha256": sha}
# core cards: whenever the deck holds one of these, monitor.core_cards must list it (game-specific: empty here)
CORE_CARD_CANDIDATES: Tuple[str, ...] = ()
# Deck slot limits (the case builder's, copied: tools/ is not importable from src/). Card keys: an evolution slot
# is ``<card>-ev1``, a hero slot ``<card>-hero``.
DECK_SLOT_SUFFIXES = {"-ev1": "evolution", "-hero": "hero"}
MAX_DECK_EVOLUTIONS, MAX_DECK_HEROES, MAX_DECK_SPECIAL = 2, 2, 3
ROW_TYPES = ("wait", "play", "ability")                   # per-row-type bf16 parity (monitor.ROW_TYPES)
DRIFT_HEADS = ("gate", "slot", "cell", "ability")         # il_v7.evaluate()['terms'] heads
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

KL_REFERENCE_CHOICES = ("champion", "il")                 # KL reference after the first promotion
H1_READING_CHOICES = ("crn_then_seatswap", "crn_only")    # how the promotion gate is read
OPPONENT_DESIGN_CHOICES = ("selfplay_il30", "il_only")    # self-play with an IL share floor, or IL opponents only
SHAPING_CHOICES = ("none", "short_head")                  # reward shaping
OPP_SCORE_KEYS = ("opp_score_alarm", "opp_score_kind", "opp_score_window", "opp_score_min_games")
GATE_FLOWS = ("crn", "h2h")                               # gates.flow ('h2h' = the head-to-head decides)
TRAIN_SCOPE_CHOICES = ("full", "freeze_step_encoder")     # the second is a control arm
BACKBONE_DTYPES = ("bf16", "fp32", "fp16")                # learner backbone; fp16: actors and learner alike
ENV_PATHS = ("engine",)                                   # actors use the engine path
GREEDY = 0.0                                              # temperature value meaning argmax
T_MIN = 1.0e-3                                            # == logprob.T_MIN: smallest non-greedy temperature

YAML_SUBSET = """\
amini.train.il_v7.load_yaml (it does not need PyYAML) understands only:
  * nested mappings by indentation; a key with nothing after ':' opens a mapping, so null must be
    written as 'null' (never an empty value); a key appears once per mapping (load_config refuses repeats);
  * scalars: null / true / false / int / float / quoted or bare strings; floats keep a dot and a
    signed exponent (3.0e-5, not 3e-5) so that PyYAML, if present, reads the same number;
  * bare NULL / Null / None / TRUE / FALSE / yes / no / on / off are refused by load_config (YAML 1.1 and
    the subset read them differently): write the population "NULL" quoted;
  * flow lists of scalars '[a, b]' split on ',' (no nested lists, no commas inside items);
  * whole-line comments and inline comments after ' #' (so no ' #' inside values).
  No block lists ('- x'), no flow mappings except '{}', no anchors, no multi-line strings."""

# bare scalars that PyYAML (YAML 1.1) and il_v7._scalar resolve differently
_AMBIGUOUS_SCALARS = frozenset("NULL Null None TRUE FALSE yes Yes YES no No NO on On ON off Off OFF".split())


class ConfigError(ValueError):
    """A PPO config that is malformed, out of range, or not ready for the requested stage."""


def model_population_input(model_config: Mapping[str, Any]) -> bool:
    """Whether a checkpoint's model reads the GLOBAL population token (``AminiFormerConfig.population_input``;
    absent = True; a model config may set it false). Feeds ``PPOConfig.missing(population_input=...)``."""

    v = model_config.get("population_input", True)
    if not isinstance(v, bool):
        raise ConfigError(f"model_config population_input must be true or false, got {v!r}")
    return v


def sampling_mode(t: float) -> Tuple[bool, Optional[float]]:
    """``(greedy, temperature)`` for a config temperature.

    ``0.0`` (``GREEDY``) -> ``(True, None)``: argmax over the slot / cells / ability heads, gate still at T = 1;
    ``0 < t <= 1`` -> ``(False, t)``: the value for ``logprob.apply_temperature`` (which refuses T = 0).
    """

    if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(float(t)):
        raise ConfigError(f"temperature must be a finite number, got {t!r}")
    t = float(t)
    if t == GREEDY:
        return True, None
    if not T_MIN <= t <= 1.0:
        raise ConfigError(f"temperature must be 0.0 (greedy) or in [{T_MIN}, 1], got {t!r}")
    return False, t


def j_pmf_from_battles(j_by_battle: Mapping[str, Mapping[Any, int]]) -> Dict[int, float]:
    """Per-game pmf of the batched-arrival offset j from per-battle histograms ``{battle: {j: count}}``.

    j is treated as fixed within a battle, so the commands of one battle are not independent samples: each
    battle's histogram is normalised and the battles are averaged with equal weight (battles in sorted order,
    probabilities summed per j, divided by the battle count). Drawing j once per game from this pmf keeps the
    battle-weighted share of each j while j stays fixed within a game (a command-pooled histogram would
    over-weight long battles)."""

    if not j_by_battle:
        raise ConfigError("j_by_battle: no battles")
    acc: Dict[int, float] = {}
    for b in sorted(j_by_battle):
        h = {int(k): int(v) for k, v in j_by_battle[b].items()}
        n = sum(h.values())
        if n <= 0 or any(v < 0 for v in h.values()):
            raise ConfigError(f"j_by_battle[{b!r}]: counts must be >= 0 with a positive total, got {h}")
        for j in sorted(h):
            acc[j] = acc.get(j, 0.0) + h[j] / n
    return {j: acc[j] / len(j_by_battle) for j in sorted(acc) if acc[j] > 0.0}


G12A_J_PMF: Dict[int, float] = j_pmf_from_battles(G12A_J_BY_BATTLE)


def apply_stamp_rule(tick: int, blocked_mod10: Sequence[int] = G12A_STAMP_BLOCKED_MOD10,
                     push_to_mod10: int = G12A_STAMP_PUSH_TO_MOD10) -> int:
    """The stamp the client writes for a click that would be stamped on ``tick`` (the stamp rule): a tick with
    ``tick % 10`` in ``blocked_mod10`` moves to the next tick ``t > tick`` with ``t % 10 == push_to_mod10``;
    any other tick is unchanged. Pure; used for every seat that stands for a live client."""

    if isinstance(tick, bool) or not isinstance(tick, int):
        raise ConfigError(f"stamp tick must be an int, got {tick!r}")
    if tick % 10 not in tuple(blocked_mod10):
        return tick
    return tick + ((push_to_mod10 - tick) % 10 or 10)


def batched_first_seen(stamp: int, j: int, period: int = ARRIVAL_PERIOD) -> int:
    """First tick on which a batched opponent command is visible: the first batch boundary strictly after the
    stamp plus j, ``period * (stamp // period + 1) + j``, i.e. delta_arr = period - stamp % period + j. Off a
    batch boundary this equals ``period * ceil(stamp / period) + j``; a stamp on a boundary
    (``stamp % period == 0``) is read in the next batch. The command stays queued through
    D = stamp + DISPATCH_LEAD and is removed at D + 1."""

    if any(isinstance(x, bool) or not isinstance(x, int) for x in (stamp, j, period)) or period < 1:
        raise ConfigError(f"batched_first_seen: ints with period >= 1 required, got {(stamp, j, period)!r}")
    return period * (stamp // period + 1) + j


def own_first_seen(stamp: int, j: int, period: int = ARRIVAL_PERIOD) -> int:
    """First tick on which a client seat reads the real stamp of its OWN command: the batched rule of the
    opponent arrivals, :func:`batched_first_seen`, with the SAME per-game j the seat draws for the opponent's
    commands (a stamp on a batch boundary is read in the next batch, as for opponent commands). Only the moment D
    is corrected to stamp + DISPATCH_LEAD depends on it: the own PENDING is visible from the decision tick either
    way."""

    return batched_first_seen(stamp, j, period)


def model_tables_digest(tables: Optional[Mapping[str, str]]) -> Optional[str]:
    """The tokenizer-table part of the MODEL compatibility digest: sha256 of
    ``json.dumps(map, sort_keys=True)`` over ``il_v7._tables_digest_map(rec['tables'])`` WITHOUT the env tables
    (:data:`ENV_TABLE_KEYS`: the arrival table belongs to :meth:`PPOConfig.env_record`). None stays None.
    ``learner.Learner.from_il`` must use this instead of hashing the whole map, so an arrival table that differs
    from the one the base model was trained with never declares the checkpoint incompatible."""

    if tables is None:
        return None
    tab = {str(k): v for k, v in dict(tables).items() if str(k) not in ENV_TABLE_KEYS}
    return hashlib.sha256(json.dumps(tab, sort_keys=True).encode()).hexdigest()


def env_tables(tables: Optional[Mapping[str, str]]) -> Optional[Dict[str, str]]:
    """The env tables (:data:`ENV_TABLE_KEYS`) of a ``_tables_digest_map`` map, for :meth:`PPOConfig.env_record`."""

    if tables is None:
        return None
    return {str(k): v for k, v in sorted(dict(tables).items()) if str(k) in ENV_TABLE_KEYS}


# ============================================================================ sections
@dataclass(frozen=True)
class UserConfig:
    """Open choices. None = not decided yet; ``missing()`` reports them.

    There is no king-level field: only all-11 or all-16 games are played, so a game's king level
    ``kt`` is its level (``PPOConfig.king_level``); the deployment level is chosen through ``levels.mix``.
    """

    deck: Optional[List[str]] = None            # the 8 card keys incl. evolution / hero slots
    tower_troop: Optional[int] = None           # our side's tower troop, game gid (TOWER_TROOP_GIDS)
    kl_reference_after_promotion: str = "champion"   # KL reference after the first promotion: 'champion' or 'il'
    h1_reading: str = "crn_then_seatswap"       # G1 CRN then G1c seat-swap, or 'crn_only'
    bc_enabled: bool = False                    # human BC term off by default
    opponent_design: str = "selfplay_il30"      # or 'il_only' (theta_k vs IL only)
    d6_subset_alarm_only: bool = True           # the human NLL subset of the drift check only alarms


@dataclass(frozen=True)
class RolloutLandingDelayConfig:
    """Decision-to-stamp delay g before the stamp rule: per kind,
    g ~ (1 - m) * normalise(<kind>_pmf) + m * U{0..uniform_max} with m = uniform_mix for the kinds in
    ``uniform_kinds`` and 0 otherwise (``PPOConfig.landing_delay_pmf``); ``rollout.stamp_rule`` then moves the
    stamp (``PPOConfig.stamped_delay_pmf``).

    The default pmfs are example values; ``uniform_mix`` 0 switches the smoothing off (``uniform_max`` /
    ``uniform_kinds`` are then inactive). A null pmf falls back to the legacy inclusive range
    ``sim2real.landing_delay.<kind>``: with both pmfs null and ``uniform_mix: 0.0`` the legacy uniform pmfs are
    reproduced exactly. The strict own-sighting bound checks a null pmf too: a legacy range must keep its
    largest stamped delay plus the largest own sighting below DISPATCH_LEAD, as the env requires."""

    play_pmf: Optional[Dict[int, float]] = field(default_factory=lambda: dict(G12A_PLAY_BASE_PMF))
    ability_pmf: Optional[Dict[int, float]] = field(default_factory=lambda: dict(ABILITY_PMF))
    uniform_mix: float = G12A_UNIFORM_MIX
    uniform_max: int = G12A_UNIFORM_MAX
    uniform_kinds: List[str] = field(default_factory=lambda: ["play"])


@dataclass(frozen=True)
class StampRuleConfig:
    """Stamp rule: a click whose stamp would land on tick % 10 in ``blocked_mod10`` is stamped on the next tick
    with tick % 10 == ``push_to_mod10``. Applies to every seat in ``rollout.client_rules_seats``; plays and
    abilities."""

    enabled: bool = True
    blocked_mod10: List[int] = field(default_factory=lambda: list(G12A_STAMP_BLOCKED_MOD10))
    push_to_mod10: int = G12A_STAMP_PUSH_TO_MOD10


@dataclass(frozen=True)
class ArrivalConfig:
    """Opponent-command arrival on the observing seat.

    ``batched``: first_seen = :func:`batched_first_seen` (``period * (stamp // period + 1) + j``), with j drawn
    ONCE PER GAME (per observing seat) from ``j_pmf`` (e.g. :func:`j_pmf_from_battles` of per-battle
    histograms: battle-normalised histograms averaged with equal weight); visible through D = stamp +
    DISPATCH_LEAD, removed at D + 1. ``table``: the legacy per-command draw, i.e. the tokenizer's ``arrival``
    table loaded at run time (file ``sim2real.delta_arr_table``). The env task implements both in the PPO seat,
    never in the env adapter, which is a model compat file (``learner.COMPAT_FILES``) shared with the IL
    adapter. ``source`` / ``source_sha256`` name the file the pmf came from (recorded, never read at run
    time; a null sha256 = no file)."""

    model: str = "batched"
    period: int = ARRIVAL_PERIOD
    j_pmf: Dict[int, float] = field(default_factory=lambda: dict(G12A_J_PMF))
    source: str = G12A_SUMMARY
    source_sha256: Optional[str] = G12A_SUMMARY_SHA256


@dataclass(frozen=True)
class ServerConfig:
    """Inference servers of every actor: the local actors', the remote actors' and the gates'. The defaults
    (off / null) keep each server's own defaults. Anything else goes into ``EpisodeSettings.server`` (with the
    actors' autocast, :meth:`PPOConfig.actor_amp`), is recorded with the games and reaches the remote actors in
    the lease."""

    lean: bool = False                          # fewer device ops per peek (unified embedding lookup, lean temporal
    #                                             step, stacked row store): fp32 outputs move by float rounding
    flush_rows: Optional[int] = None            # theta / gate servers: a flush goes out at this many waiting rows
    max_wait_ms: Optional[float] = None         # theta / gate servers: null = the server's default
    opponent_flush_rows: Optional[int] = None   # frozen-opponent servers (few rows per call: let them fill)
    opponent_max_wait_ms: Optional[float] = None


@dataclass(frozen=True)
class RemoteActorsConfig:
    """Remote actor exchange: off by default. ``root`` is the shared exchange directory (an absolute path),
    required when enabled."""

    enabled: bool = False
    root: Optional[str] = None
    lease_s: int = 3600                         # a remote actor's work lease
    max_actors: int = 4
    tail_speed_margin: Optional[float] = None   # null = off; m >= 1: at the tail a remote chunk expected to
    #                                             need more than m x the local chunk time (both measured in this
    #                                             update) is taken back and replayed locally (never counted twice)


@dataclass(frozen=True)
class RolloutConfig:
    """Rollout / actor settings."""

    games_per_update: int = 1000                # exactly N games per update, all used
    replicas_per_group: int = 1                 # 1 = independent draws; > 1 replays each draw (leave-one-out A/B)
    temperature: float = 1.0                    # theta_k slot / cells / ability at T = 1 (fixed)
    gate_temperature: float = 1.0               # the gate is always sampled at T = 1 (fixed)
    hazard_dt: Optional[str] = None             # the base model's hazard_dt mode, passed explicitly; required
    max_env_slots: int = 8                      # concurrent games (<= MAX_ENV_SLOTS_CEILING)
    games_per_worker: int = 1                   # independent games owned by one rollout process
    env_path: str = "engine"
    emu_spot_check_games: int = 10              # parity spot-check games after each env freeze
    landing_delay: RolloutLandingDelayConfig = field(default_factory=RolloutLandingDelayConfig)
    min_tap_tick: Optional[int] = G12A_MIN_TAP_TICK   # a client seat may not act before this DECISION tick
    #                                             (plays and abilities masked to wait); null = off
    skip_forced_uncommitted: bool = False       # bitwise-equivalent skipping of forced uncommitted queries; true
    #                                             only after a parity check on the frozen env
    remote: RemoteActorsConfig = field(default_factory=RemoteActorsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)                # inference-server settings
    stamp_rule: StampRuleConfig = field(default_factory=StampRuleConfig)      # stamp rule
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)             # batched opponent arrival
    client_mask: bool = True                    # an optional client slot mask: a sampling-only slot mask (actor and
    #                                             the learner's recompute), HAND input features unchanged
    client_rules_seats: List[str] = field(default_factory=lambda: list(CLIENT_SEAT_ROLES))   # seats that stand for
    #                                             a live client: stamp rule, landing delay, min_tap_tick, client
    #                                             mask, and batched arrival of the commands they observe; the
    #                                             roles are CLIENT_SEAT_ROLE_DOC ('gate' = both seats of every
    #                                             scored gate game and of every canary game)


@dataclass(frozen=True)
class PolicyConfig:
    """Policy optimisation."""

    clip_eps: float = 0.25                      # joint ratio
    dual_clip: float = 2.5                      # arXiv 1912.09729
    lr: float = 5.0e-5                          # constant
    lr_schedule: str = "constant"
    betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_clip_raised: float = 2.0               # one-time raise ...
    grad_clip_raise_after_updates: int = 10     # ... after this many consecutive updates ...
    grad_clip_raise_clipped_frac: float = 0.8   # ... with more than this fraction of steps clipped
    minibatch_seat_games: int = 50              # seat-games per minibatch
    epochs: int = 3
    dropout: float = 0.0                        # stays on-policy
    train_scope: str = "full"                   # full fine-tune, uniform LR
    aux_heads: bool = False                     # aux_opp_hand / aux_opp_next off
    advantage_norm: str = "per_update"          # once per update, unified weights
    is_ratio_cap_lag1: float = 1.5              # policy rho-bar, only for lag-1 overlap
    gae_trace_cap_lag1: float = 1.0             # V-trace c-bar of the GAE trace at lag 1
    rollout_lag: int = 0                        # same-update data, no IS weight
    backbone_dtype: str = "fp32"                # learner backbone (bf16 needs the per-row-type tolerances
    #                                             monitor.parity.dlogp_tol_bf16)
    micro_rows: int = 256                       # rows per learner micro-batch
    max_micro_tokens: int = 65536               # padded step-encoder tokens per micro-batch (0 = off); a
    #                                             seat-game is never split


@dataclass(frozen=True)
class LearnerCompileConfig:
    """Optional PPO learner compilation; the empty mode is the frozen default."""

    compile_mode: str = ""                      # off | default | max-autotune-no-cudagraphs
    compile_dynamic: bool = False


@dataclass(frozen=True)
class KLConfig:
    """KL to the reference policy."""

    beta: float = 0.02                          # normalised by the surrogate denominator
    beta_min: float = 0.005
    beta_max: float = 0.1
    normalisation: str = "surrogate_denominator"


@dataclass(frozen=True)
class EarlyStopConfig:
    """Per-epoch early stops (calibrated in the pilot phase)."""

    joint_k3: float = 0.02
    play_k3: float = 0.1
    gate_kl: float = 0.01                       # per non-forced row
    lr_halve_if_stopped_frac: float = 0.75      # a larger share of the updates stopped early -> LR / 2 (temporary)
    lr_halve_window: int = 8                    # ... counted over the last this-many updates (monitor's reading)


@dataclass(frozen=True)
class EntropyConfig:
    """Entropy bonuses."""

    coef: Dict[str, float] = field(default_factory=lambda: {"gate": 0.0, "slot": 0.0, "cells": 0.0, "ability": 0.0})
    cell_bonus: float = 5.0e-3                  # cells bonus switched on for a card ...
    cell_trigger_frac_of_il: float = 0.25       # ... whose min cell entropy falls below this fraction of IL's


@dataclass(frozen=True)
class GAEConfig:
    """Per-tick GAE."""

    gamma: float = 0.9999
    lambda_tick: float = 0.995
    lambda_tick_f1: float = 0.997               # the longer lambda of the stall remedy (a runtime change)


@dataclass(frozen=True)
class CriticWarmupConfig:
    """Critic-only warm-up."""

    games_per_batch: int = 256
    max_updates: int = 20
    fixed_set_games: int = 1000                 # fixed recorded set for the per-phase EV
    stop_trend_updates: int = 4                 # stop when EV has not trended up over this many updates
    early_stop_min_updates: Optional[int] = None    # null = the stop rule only; N = also stop at the
    #                                             first update >= N whose fixed-set EV meets both pass lines
    overlap: bool = False                       # play the next batch while the critic learns this one (the
    #                                             policy is frozen in warm-up, so every batch is the same games)
    plateau_gain: Optional[float] = None        # null = off; G = also stop, with both pass lines met, when the
    #                                             best overall fixed-set EV of the last plateau_window updates is less
    #                                             than G above the best before them (the warm-up has stopped paying)
    plateau_window: int = 2
    pass_ev_overall: float = 0.1
    pass_ev_last_third: float = 0.2


@dataclass(frozen=True)
class CriticConfig:
    """Separate value network."""

    loss: str = "mse"                           # tanh output vs lambda-returns; no clip / Huber / PopArt
    lr: float = 3.0e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1.0e-8
    grad_clip: float = 2.0
    epochs: int = 3
    value_priv: bool = False                    # privileged-input value head (off)
    warmup: CriticWarmupConfig = field(default_factory=CriticWarmupConfig)


@dataclass(frozen=True)
class RewardConfig:
    """Rewards."""

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    shaping: str = "none"                       # terminal only, or 'short_head'
    short_head_coef: float = 0.2                # separate short-horizon head (A = A_win + A_short)
    short_head_gamma: float = 0.99              # per decision ...
    short_head_decision_ticks: int = 20         # ... of this many ticks (gae.SHORT_DECISION_TICKS); a smaller unit
    #                                             gives a shorter horizon for the same per-decision gamma


@dataclass(frozen=True)
class BCConfig:
    """Optional human BC term. Off unless user.bc_enabled."""

    weight: float = 0.0
    start_weight: float = 0.1                   # the value to start from when BC is turned on
    min_rows_per_update: int = 10000


@dataclass(frozen=True)
class WeightsConfig:
    """Row weights: HT x seat x [non-forced]."""

    seat_ours: float = 1.0
    seat_theta_opp: float = 0.25                # theta_k on the opponent seat (kind self_seat)
    seat_mirror: float = 1.0
    ht_action: float = 1.0
    ht_wait: Dict[str, float] = field(default_factory=lambda: dict(G.WEIGHT))   # 1 / KAPPA per wait stratum
    forced: float = 0.0


@dataclass(frozen=True)
class OpponentsConfig:
    """Opponent mix and frozen-opponent settings."""

    pilot_updates: int = 10                     # updates 1..pilot_updates use ``pilot``, then ``main``
    # the training IL opponents are the base model (key il_v4) at T = 1 and at T*_IL; the extra IL slot il_v3_t1
    # has no share by default
    pilot: Dict[str, float] = field(default_factory=lambda: {
        "self_seat": 0.4, "mirror": 0.1, "il_v4_t1": 0.3, "il_v4_tstar": 0.2, "il_v3_t1": 0.0,
        "snapshot": 0.0})
    main: Dict[str, float] = field(default_factory=lambda: {
        "self_seat": 0.4, "mirror": 0.1, "il_v4_t1": 0.15, "il_v4_tstar": 0.1, "il_v3_t1": 0.0,
        "snapshot": 0.25})
    il_floor: float = 0.25                      # IL share floor of every phase
    # This share of every update's games are GENERAL games -- real openings in which neither side plays
    # user.deck, theta on both seats (kind self_seat, so both seats are learned). The rest of the update follows
    # ``pilot`` / ``main`` (their shares are shares of the rest). 0 = off; the openings come from the inputs'
    # ``general_cases``.
    general_share: float = 0.0
    # Who sits on the other seat of a GENERAL game. "self": theta on both seats. "mix": the phase's mix
    # (``pilot`` / ``main``) with the mirror share added to self_seat (a general opening has no mirror form): theta
    # on both seats, a pool snapshot, or the IL base, the frozen models on the opening's own other deck; the IL
    # floor holds among the general games too. ``general_share`` may be 1.0 (no game with user.deck at all).
    general_opponents: str = "self"
    snapshot_every: int = 5
    snapshot_pool_recent: int = 4               # + the current champion, uniform
    # League anchors: fixed frozen models that are ALWAYS in the snapshot pool beside the recent snapshots and the
    # champion ("NAME|PATH|SHA256" each, :func:`parse_anchor`), so that the pool is not only the champion's own
    # lineage. Empty = no anchors.
    snapshot_anchors: List[str] = field(default_factory=list)
    # Matchup games: this share of every update's games has user.deck against ONE fixed deck, ``matchup_deck``
    # (8 card keys). The opening is synthetic: a real opening of a human who plays ``matchup_deck`` (the inputs'
    # ``matchup_cases``) with the other owner replaced by user.deck. The other seat is a frozen model and is not
    # learned: one of ``matchup_models`` ("NAME|PATH|SHA256" each, :func:`parse_anchor`; one is drawn per update
    # with ``matchup_weights``, uniform when empty) or, for ``matchup_il_share`` of the matchup games, the IL base.
    # These models are in the update's opponent table but never in the snapshot pool of the other games (unless
    # they are also league anchors). 0 = off.
    matchup_share: float = 0.0
    matchup_deck: List[str] = field(default_factory=list)
    matchup_models: List[str] = field(default_factory=list)
    matchup_weights: List[float] = field(default_factory=list)
    matchup_il_share: float = 0.0
    # A SECOND matchup: the same five fields for another fixed deck, with its own share, models, weights and IL
    # share. Its templates are the inputs' ``matchup2_cases`` / ``matchup2_panel``; its games take the replicas
    # just after the first matchup's. It needs the first matchup (``matchup_deck``) and another deck; a model in
    # both matchups is one file under one name. 0 = off.
    matchup2_share: float = 0.0
    matchup2_deck: List[str] = field(default_factory=list)
    matchup2_models: List[str] = field(default_factory=list)
    matchup2_weights: List[float] = field(default_factory=list)
    matchup2_il_share: float = 0.0
    temperature: Dict[str, Optional[float]] = field(default_factory=lambda: {   # snapshot: the temperature the
        "il_v4_t1": 1.0, "il_v4_tstar": None, "il_v3_t1": 1.0, "snapshot": 1.0, "il_v2_heldout": 1.0})  # pool plays at
    # Key roles (MODEL_ROLES): il_v4 = the BASE IL model (theta init, KL reference until the first promotion,
    # training IL opponent); il_v3 = the extra IL slot (unused at share 0); il_v2_heldout = the HELD-OUT IL gate
    # opponent (never trained against, no gate bias)
    checkpoint: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "il_v4": None, "il_v3": None, "il_v2_heldout": None})
    # sha256 of each checkpoint FILE: a pinned copy, verified on load (learner.verify_pin) and recorded per game
    # in the shard metadata (GameMeta.opp_settings['checkpoint_sha256'])
    checkpoint_sha256: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "il_v4": None, "il_v3": None, "il_v2_heldout": None})
    # per-opponent settings. gate_bias = aminiformer cfg.gate_bias as two rows over (wait, play, ability):
    # 'base' and 'threat'; all zeros = no bias (the raw model, cfg.gate_bias None); set it here to play a frozen
    # model with the gate bias it was deployed with.
    gate_bias: Dict[str, Dict[str, List[float]]] = field(default_factory=lambda: {
        m: {"base": [0.0, 0.0, 0.0], "threat": [0.0, 0.0, 0.0]} for m in GATE_BIAS_MODELS})
    # hazard_dt mode and code tree of the models outside theta_k's lineage (an older tokenizer may have other
    # legality rules and dt modes); required before the model plays
    hazard_dt: Dict[str, Optional[str]] = field(default_factory=lambda: {m: None for m in OLD_TREE_MODELS})
    code_tree: Dict[str, Optional[str]] = field(default_factory=lambda: {m: None for m in OLD_TREE_MODELS})


@dataclass(frozen=True)
class LandingDelayConfig:
    """LEGACY decision-to-stamp delay g in ticks, inclusive uniform ranges. Read only for a kind whose
    ``rollout.landing_delay.<kind>_pmf`` is null."""

    play: Tuple[int, int] = (1, 4)
    ability: Tuple[int, int] = (1, 3)


@dataclass(frozen=True)
class DeltaLagConfig:
    """Client object-appearance lag in ticks. The env models no display lag: ``DeltaLagConfig()`` is (0, {})
    and validation refuses any nonzero value (a lag shifts born ticks, the seat's self-check then stalls it); an
    override of 0 for a real object name is allowed."""

    default_ticks: int = 0                      # every object family
    overrides: Dict[str, int] = field(default_factory=dict)   # keys: data/object_vocab_v7.json 'ids'


@dataclass(frozen=True)
class Sim2RealConfig:
    """Training-only sim-to-real perturbations; always off in the client package."""

    enabled: bool = True
    landing_delay: LandingDelayConfig = field(default_factory=LandingDelayConfig)
    delta_lag: DeltaLagConfig = field(default_factory=DeltaLagConfig)
    delta_arr_table: str = "data/opp_cmd_arrival_v7.json"   # legacy arrival table, relative to the code root; read
    #                                             only by rollout.arrival.model 'table' (legacy)
    threat_from_filtered_state: bool = True


@dataclass(frozen=True)
class LevelsConfig:
    """Level mix and population; keys are the uniform game levels."""

    mix: Dict[int, float] = field(default_factory=lambda: {16: 0.8, 11: 0.2})
    population: Dict[int, Optional[str]] = field(default_factory=lambda: {16: "ranked", 11: None})


@dataclass(frozen=True)
class ScreenGateConfig:
    at_update_in_cycle: int = 5
    games: int = 256


@dataclass(frozen=True)
class G1Config:
    every_updates: int = 10
    games_per_side: int = 500
    alpha_one_sided: float = 0.1
    heldout_min_delta_pp: float = -2.0


@dataclass(frozen=True)
class G1cConfig:
    pairs: int = 500


@dataclass(frozen=True)
class G2Config:
    pairs: int = 250
    record_only: bool = True


@dataclass(frozen=True)
class ReleaseGateConfig:
    pairs: int = 500
    mirror_games: int = 200
    transitivity_pairs: int = 100
    live_games_min: int = 10


@dataclass(frozen=True)
class GatesConfig:
    """Evaluation and promotion."""

    screen: ScreenGateConfig = field(default_factory=ScreenGateConfig)
    g1: G1Config = field(default_factory=G1Config)
    g1c: G1cConfig = field(default_factory=G1cConfig)
    g2: G2Config = field(default_factory=G2Config)
    release: ReleaseGateConfig = field(default_factory=ReleaseGateConfig)
    crn_panel: Dict[str, float] = field(default_factory=lambda: {"il_v2_heldout": 0.4, "il_v4_tstar": 0.3, "il_v4_t1": 0.3})
    min_completion: float = 0.95
    bootstrap_resamples: int = 5000
    aa_null_pairs: int = 500
    il_temperature_sweep: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.0])
    deploy_temperature_sweep: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.3, 0.0])
    sweep_games: int = 500
    sprt: bool = False
    sprt_h1_pp: float = 3.0
    # Early ACCEPT of a G1. null = off (every G1 plays all its pairs; gates.sprt alone only writes the SPRT into the
    # report). a in (0, 0.05]: after every chunk of pairs the G1 stops and passes when (1) the fixed-sample rule
    # already passes on the pairs played (lower bound, held-out sub-panel, failure guard, schedule) AND (2) the
    # truncated SPRT in schedule order has decided H1 at level a. Otherwise it goes on to the full size and is
    # judged by the fixed-sample rule: a G1 that would pass still passes, none is failed early. G1c, G2 and the
    # screen are not touched.
    sprt_stop_alpha: Optional[float] = None
    # What decides a promotion. 'crn' = the screen and G1 on the fixed outside panel, then G1c. 'h2h' = every
    # ``g1.every_updates`` updates the candidate plays the champion with swapped seats (the G1c block: both models
    # hold our deck and the opening's other deck once per pair, same seeds); it is promoted when the lower bound of
    # its score is above 0.5 and its result with OUR deck is not clearly below the champion's own
    # (``h2h_guard_pairs``). No screen, no panel G1; the monitor's gate rules read the score as
    # delta_hat = score - 0.5. 'h2h' suits a candidate and champion that both beat the outside panel almost always.
    flow: str = "crn"
    h2h_stop_alpha: Optional[float] = None      # 'h2h': stop the head-to-head as soon as a sequential test at this
    #                                             level decides (better: passes if the fixed-sample rule passes on
    #                                             the pairs played; not better: fails); null = play every pair
    h2h_h1_pp: float = 3.0                      # that test's H1: a score of 0.5 + this
    h2h_guard_pairs: int = 200                  # champion against itself, the same seat-swap form, once per champion:
    #                                             the baseline of "our deck against the champion holding the other
    #                                             deck"; 0 = no guard
    h2h_guard_alpha: float = 0.1                # the guard fails when the candidate's rate is below the baseline's at
    #                                             this one-sided level (it catches a clear drop, not a small one)
    # The gate games' worker layout. null = as the rollout (rollout.games_per_worker lanes per process, the
    # block's --workers processes). Set both to fall back to one-game gate processes while the rollout keeps its
    # lanes, e.g. games_per_worker 1 and workers 8.
    games_per_worker: Optional[int] = None
    workers: Optional[int] = None
    canaries: List[str] = field(default_factory=lambda: list(CANARY_KINDS))   # diagnostic canary
    canary_games: int = 32                      # players per gate; never promote, never block
    canary_alarm_lb: float = 0.2                # a canary whose Wilson lower bound beats this flags the gate
    #                                             (passed to the gate runner; a runtime-change key)
    panel_level: Optional[int] = 16             # the gate panel plays only this level (null = every
    #                                             level with a share in levels.mix)
    # The record-only G2. ``g2_anchor`` = "NAME|PATH|SHA256": the candidate plays this fixed model instead of the IL
    # base (a fixed strong model gives a progress line that does not depend on the moving champion).
    # ``g2_every_g1`` = N > 0: the G2 is also played at every N-th G1 without a promotion (0 = only after a
    # promotion).
    g2_anchor: str = ""
    g2_every_g1: int = 0
    # The record-only matchup measurement (with ``opponents.matchup_share``): at every G1,
    # ``matchup_games`` > 0 synthetic VAL openings of the matchup (the inputs' ``matchup_panel``), each played once
    # by the candidate and once by the champion with user.deck against ``matchup_anchor`` ("NAME|PATH|SHA256") on
    # ``opponents.matchup_deck``. It never promotes and never blocks; 0 = off.
    matchup_games: int = 0
    matchup_anchor: str = ""
    # the second matchup's record-only measurement (``opponents.matchup2_deck``, the inputs' ``matchup2_panel``),
    # exactly as above in its own blocks (aa-mu2-u<update>); 0 = off
    matchup2_games: int = 0
    matchup2_anchor: str = ""
    panel_source: str = "deck"                # "deck" = the inputs' ``panel`` (val openings of user.deck);
    #                                             "general" = the inputs' ``general_panel`` (val openings in which
    #                                             neither side plays user.deck): every gate of the run is played on
    #                                             general openings
    g2_mode: str = "pooled_since_restart"       # the G2 trend reading pools every G2 since the last restart (the
    #                                             latest G2 is reported too); 'latest' = the latest G2 only
    pilot_min_delta_pp: float = -5.0            # pilot exit: the G1 at the end of the pilot not clearly worse
    pilot_joint_kl_median_max: float = 0.02     # pilot exit: median joint approx-KL over the pilot updates
    p6_entry_g2_min_score: float = 0.6          # next phase: cumulative G2 score >= this with CI > 0.5
    restart_swap_min_score: float = 0.6         # restart: v_n beats the IL reference (>= this or CI > 0.5)
    rollback_delta_pp: float = -5.0             # consecutive screens / G1s at or below this (pp) ...
    rollback_consecutive: int = 3
    rollback_lr_mult: float = 0.6               # ... -> back to the champion, LR x this (temporary)
    rollback_beta_double_window: int = 40       # two rollbacks within this many updates ...
    rollback_beta_mult: float = 3.0             # ... -> beta x this (temporary)
    stall_g1_without_promotion: int = 5         # stall ladder start
    stall_step_min_updates: int = 10            # each ladder step lasts >= this many updates
    stall_games_per_update: int = 2000          # ladder: games per update from its second step on
    stall_stop_g1: int = 8
    release_after_g1_without_new: int = 5


@dataclass(frozen=True)
class ParityMonitorConfig:
    """Update-0 consistency."""

    max_abs_dlogp_fp32: float = 1.0e-3         # fp32 learner vs the actor's mu, every committed row
    dlogp_tol_bf16: Dict[str, Optional[float]] = field(default_factory=lambda: {   # per row type, measured;
        "wait": None, "play": None, "ability": None})                          # required with a bf16 backbone
    # The same for an fp16 backbone. There the actors run fp16 too (PPOConfig.actor_amp), so this is the
    # measured spread of one fp16 forward against another (other batch shape, other card), not fp16 against fp32
    dlogp_tol_fp16: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "wait": None, "play": None, "ability": None})
    ratio_tol: float = 1.0e-3
    kl_theta0_ref_max: float = 1.0e-5
    replay_games: int = 4
    games: int = 10


@dataclass(frozen=True)
class MonitorConfig:
    """Monitors and their automatic actions; ``PPOConfig.monitor_thresholds`` maps them."""

    parity: ParityMonitorConfig = field(default_factory=ParityMonitorConfig)
    clip_frac_max: float = 0.4
    clip_frac_consecutive: int = 4
    temp_lr_mult: float = 0.6                   # temporary LR factor for clip / early-stop alarms
    temp_lr_floor: float = 0.4                  # lowest temporary lr factor (no self-made stall)
    kl_stall_window: int = 8
    kl_stall_band: float = 0.2
    kl_stall_g1_without_promotion: int = 4
    kl_stall_beta_mult: float = 0.6             # KL stall: beta x this (to the floor) ...
    lr_stall_mult: float = 2.0                  # ... else an equal-games A/B fork proposal with arm B at LR x this
    #                                             (never an automatic permanent LR change)
    kl_step_skip_growth: float = 1.5            # after a beta KL step, mean KL to the reference grew by
    kl_step_skip_g1_move_pp: float = 1.0        # >= that factor and the next G1 moved < this (pp) -> skip later
    #                                             KL steps
    kl_grad_ratio_min: float = 1.0e-12          # beta > 0 and theta != reference -> the KL / surrogate
    #                                             gradient-norm ratio must exceed this, else stop
    tiny_log_ratio: float = -10.0               # log(pi_theta / pi_prox) below this ...
    tiny_ratio_share_max: Optional[float] = None   # ... share of play / wait rows; alarm threshold (null = log only)
    gate_trend_window: int = 8                  # updates in the two-sided act-rate trend window
    gate_trend_act_rel_max: Optional[float] = None   # |relative act-rate drift over the window| (null = log only)
    gate_trend_adv_gap_max: Optional[float] = None   # |play-minus-wait advantage gap drift| (null = log only)
    entropy_max_frac_of_ref: Optional[float] = None  # upper entropy alarm vs the reference (null = log only)
    flip_rate_max: Optional[float] = None       # slot / cell argmax flip rate alarm (null = log only)
    probe_rows: int = 512                       # rows of the fixed argmax probe set (0 = off)
    plays_per_min_band: Tuple[float, float] = (0.8, 1.25)
    entropy_min_frac_of_il: float = 0.4
    card_rate_band: Tuple[float, float] = (0.25, 4.0)
    core_cards: List[str] = field(default_factory=lambda: ["example-card"])   # placeholder: the deck's core cards
    core_card_min_frac_of_il: float = 0.2
    core_card_consecutive: int = 4
    core_rollbacks_before_stop: int = 2         # rollbacks for a core-card alarm before it stops the run
    pilot_core_card_min_frac: float = 0.25
    illegal_cell_code: int = 19                 # refusals with this code: see code19_stop_*
    legal_refused_max_ratio_vs_il: float = 3.0
    forced_frac_alarm_abs: float = 0.2          # the forced-row share only alarms
    mask_recheck_every: int = 10
    drop_rate_warn: float = 0.01
    drop_rate_stop: float = 0.05
    opp_nonwait_min_frac: float = 0.4
    value_ev_overall_min: float = 0.1
    value_ev_last_third_min: float = 0.2
    value_ev_consecutive: int = 4
    value_ev_drop_from_warmup: float = 0.2
    drift_every: int = 10
    throughput_min_frac: float = 0.5
    min_free_mem_gb: float = 4.0
    swap_growth_gb: float = 1.0                 # swap growth above this ...
    swap_window_s: float = 3600.0               # ... within this window -> fewer slots
    swap_cooldown_s: float = 0.0                # after a swap cut, no further cut for this long (0 = off)
    swap_mem_guard_gb: Optional[float] = None   # null = off; g: swap growth cuts slots only while
    #                                             MemAvailable < g GB (system swap moves for other services too)
    # The illegal-cell rule. An update with code-19 refusals stops the run when it has at least
    # code19_stop_per_update of them, or when it is the code19_stop_consecutive-th such update in a row; below both
    # lines it is an alarm. The defaults (1, 1): one refusal stops.
    code19_stop_per_update: int = 1
    code19_stop_consecutive: int = 1
    # Our score in the TRAINING games against one frozen opponent kind (win 1, draw 0.5, loss 0), pooled over the
    # last opp_score_window updates. With at least opp_score_min_games games there and a score below
    # opp_score_alarm: an alarm in the update's 'opp_score' event and a state flag, no action. null = off (nothing
    # read or written).
    opp_score_alarm: Optional[float] = None
    opp_score_kind: str = "il_v4_tstar"
    opp_score_window: int = 10
    opp_score_min_games: int = 100
    host_memory_events: bool = False            # write the block process's host memory (RSS / swap
    #                                             before and after the release) as a 'host_memory' event
    #                                             after every update; the release itself always runs
    swap_slot_reduction: int = 4
    nan_skip_consecutive_stop: int = 5


@dataclass(frozen=True)
class DriftConfig:
    """Human-consistency drift (``amini.ppo.drift``): every ``monitor.drift_every`` updates,
    ``il_v7.evaluate()`` on the fixed IL selection subset; alarm only, never a stop or a selection criterion.

    The thresholds are judged values, not noise multiples (fixed data + fixed weights make the evaluation
    deterministic). Each is compared with the base IL reference on the same subset: per-head NLL increase (nats
    per weighted row), gate AUC drop, and the ratio of the predicted / observed action counts on wait rows (the
    plays-per-minute calibration of ``evaluate``) of the policy over IL's, per stratum."""

    subset_games: int = 1000                    # the fixed selection subset (smallest sha256)
    exclude_friendly: bool = True               # friendly games left out of the subset
    nll_increase_max: Dict[str, float] = field(default_factory=lambda: {
        "gate": 0.1, "slot": 0.2, "cell": 0.2, "ability": 0.2})
    gate_auc_drop_max: float = 0.03
    calibration_ratio_band: Tuple[float, float] = (0.8, 1.25)   # same band as monitor.plays_per_min_band


@dataclass(frozen=True)
class CasesConfig:
    """Opening references: populations left out of the training opening refs and of the human deck table (the
    block runner's ``inputs --exclude-population``)."""

    exclude_population: List[str] = field(default_factory=lambda: ["friendly"])


@dataclass(frozen=True)
class ChainConfig:
    """Block chaining: the chain script's ``--timeout / --kill-after``.

    Each block must end on its own budget: save at a barrier and exit 0 when less than one update cycle
    remains before ``block_timeout_s``; a block killed by the timeout exits 124 and the chain stops."""

    block_timeout_s: int = 14400                # 4 h block
    kill_after_s: int = 600                     # SIGKILL this long after the TERM


@dataclass(frozen=True)
class StorageConfig:
    """Shards."""

    rows: str = "committed_tokens"
    keep_updates: int = 2


@dataclass(frozen=True)
class PPOConfig:
    """The whole PPO run configuration. Validated on construction; attributes frozen."""

    name: str = "ppo_v7_main"
    user: UserConfig = field(default_factory=UserConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    learner: LearnerCompileConfig = field(default_factory=LearnerCompileConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    gae: GAEConfig = field(default_factory=GAEConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    sim2real: Sim2RealConfig = field(default_factory=Sim2RealConfig)
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    cases: CasesConfig = field(default_factory=CasesConfig)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Range and cross-field checks; raises ConfigError naming the field."""

        _validate(self)

    # ------------------------------------------------------------------ readiness
    def missing(self, stage: str = "rollout", *,
                population_input: Union[bool, Mapping[str, bool]] = True) -> List[str]:
        """Fields that must be set before ``stage`` ('rollout' or 'policy'); empty = ready.

        ``population_input``: whether each model reads the GLOBAL population token (``model_population_input``
        of its checkpoint), keyed like ``opponents.checkpoint``; a bool is the base model's (``il_v4``: theta,
        both il_v4 kinds and the PPO snapshots). Models not given count as reading it. The level populations are
        required only while a model that plays at that stage reads it: il_v4 always, il_v3 when ``il_v3_t1`` has
        a share, il_v2_heldout for 'policy' (gates). A run where none does may stamp any code (e.g. NULL)."""

        if stage not in READINESS_STAGES:
            raise ConfigError(f"stage must be one of {READINESS_STAGES}, not {stage!r}")
        u, o = self.user, self.opponents
        out: List[str] = []
        if u.deck is None:
            out.append("user.deck: the 8 card keys incl. evolution / hero slots")
        if u.tower_troop is None:
            out.append("user.tower_troop: our side's tower troop, the game gid written into the game input")
        if self.rollout.hazard_dt is None:
            out.append("rollout.hazard_dt: the base IL model's hazard_dt mode, block | interval")
        if self.rollout.remote.enabled and self.rollout.remote.root is None:
            out.append("rollout.remote.root: the shared exchange directory of the remote actors")
        if o.checkpoint["il_v4"] is None:
            out.append("opponents.checkpoint.il_v4: the BASE IL checkpoint (policy init, reference, opponent)")
        out += self._sha_missing("il_v4")
        used = {k for ph in PHASES for k, v in self.effective_mix(ph).items() if v > 0}
        if "il_v3_t1" in used:
            if o.checkpoint["il_v3"] is None:
                out.append("opponents.checkpoint.il_v3: extra IL slot opponent (no share by default)")
            out += self._sha_missing("il_v3")
            out += self._old_tree_missing("il_v3")
        if o.temperature["il_v4_tstar"] is None and ("il_v4_tstar" in used or stage == "policy"):
            out.append("opponents.temperature.il_v4_tstar: T*_IL from the temperature sweep, also T_gate")
        if o.temperature["snapshot"] is None and "snapshot" in used:
            out.append("opponents.temperature.snapshot: current estimate of T_dep for PPO snapshots")
        reads_pop = self._population_read(stage, used, population_input)
        for lvl in LEVELS:
            if reads_pop and self.levels.mix[lvl] > 0 and self.levels.population[lvl] is None:
                out.append(f"levels.population.{lvl}: NULL / tournament / challenge_eventdeck")
        if stage == "policy":
            if o.checkpoint["il_v2_heldout"] is None:
                out.append("opponents.checkpoint.il_v2_heldout: HELD-OUT IL gate opponent")
            out += self._sha_missing("il_v2_heldout")
            out += self._old_tree_missing("il_v2_heldout")
            if self.policy.backbone_dtype == "bf16":
                for rt in ROW_TYPES:
                    if self.monitor.parity.dlogp_tol_bf16[rt] is None:
                        out.append(f"monitor.parity.dlogp_tol_bf16.{rt}: measured bf16 update-0 tolerance, "
                                   "or policy.backbone_dtype=fp32")
            if self.policy.backbone_dtype == "fp16":
                for rt in ROW_TYPES:
                    if self.monitor.parity.dlogp_tol_fp16[rt] is None:
                        out.append(f"monitor.parity.dlogp_tol_fp16.{rt}: measured fp16 update-0 tolerance, "
                                   "or policy.backbone_dtype=fp32")
        return out

    def _population_read(self, stage: str, used: Any, population_input: Union[bool, Mapping[str, bool]]) -> bool:
        """True while a model playing at ``stage`` reads the population token (see :meth:`missing`)."""
        pi = {"il_v4": population_input} if isinstance(population_input, bool) else dict(population_input)
        bad = sorted(k for k, v in pi.items() if k not in self.opponents.checkpoint or not isinstance(v, bool))
        if bad:
            raise ConfigError(f"population_input: keys must be {sorted(self.opponents.checkpoint)} with bool "
                              f"values, got {bad}")
        models = ["il_v4"] + (["il_v3"] if "il_v3_t1" in used else [])
        models += ["il_v2_heldout"] if stage == "policy" else []
        return any(pi.get(m, True) for m in models)

    def _sha_missing(self, model: str) -> List[str]:
        if self.opponents.checkpoint_sha256[model] is None:
            return [f"opponents.checkpoint_sha256.{model}: sha256 of the pinned {model} checkpoint file "
                    "(learner.verify_pin)"]
        return []

    def _old_tree_missing(self, model: str) -> List[str]:
        o, out = self.opponents, []
        if o.hazard_dt[model] is None:
            out.append(f"opponents.hazard_dt.{model}: the hazard_dt mode {model} was trained with, block | interval")
        if o.code_tree[model] is None:
            out.append(f"opponents.code_tree.{model}: the code tree {model} runs on (an older tokenizer may have "
                       "other legality rules)")
        return out

    def require_ready(self, stage: str = "rollout", *,
                      population_input: Union[bool, Mapping[str, bool]] = True) -> None:
        """Raise ConfigError listing every field still missing for ``stage`` (``missing`` arguments)."""

        miss = self.missing(stage, population_input=population_input)
        if miss:
            raise ConfigError(f"config not ready for {stage}:\n  " + "\n  ".join(miss))

    # ------------------------------------------------------------------ derived values
    def effective_mix(self, phase: str) -> Dict[str, float]:
        """The opponent mix of ``phase`` (the case builder's kind names); 'il_only' keeps the IL rows renormalised."""

        if phase not in PHASES:
            raise ConfigError(f"phase must be one of {PHASES}, not {phase!r}")
        mix = dict(getattr(self.opponents, phase))
        if self.user.opponent_design == "il_only":
            tot = sum(mix[k] for k in IL_KINDS)
            mix = {k: (mix[k] / tot if k in IL_KINDS else 0.0) for k in OPPONENT_KINDS}
        return mix

    def general_mix(self, phase: str) -> Dict[str, float]:
        """The opponent mix of the GENERAL games of ``phase`` (``opponents.general_opponents``): 'self' = theta on
        both seats; 'mix' = :meth:`effective_mix` with the mirror share added to self_seat (a general opening has
        two different decks, so there is no mirror form of it)."""

        if self.opponents.general_opponents == "self":
            return {k: (1.0 if k == "self_seat" else 0.0) for k in OPPONENT_KINDS}
        mix = self.effective_mix(phase)
        mix["self_seat"] = mix["self_seat"] + mix["mirror"]
        mix["mirror"] = 0.0
        return mix

    def phase_for_update(self, update: int) -> str:
        """'pilot' for updates 1..pilot_updates, else 'main'."""

        if update < 1:
            raise ConfigError(f"updates are numbered from 1, not {update}")
        return "pilot" if update <= self.opponents.pilot_updates else "main"

    def trained_seats_per_game(self, phase: str) -> float:
        """Expected trained seats per game: ours + theta_k on the opponent seat + the mirror's other side."""

        mix = self.effective_mix(phase)
        return 1.0 + mix["self_seat"] + mix["mirror"]

    def king_level(self, level: int) -> int:
        """Our king level ``kt`` in a game of uniform ``level``: the level itself (the case builder writes it)."""

        if isinstance(level, bool) or level not in LEVELS:
            raise ConfigError(f"level must be one of {LEVELS}, not {level!r}")
        if self.levels.mix[level] <= 0:
            raise ConfigError(f"level {level} has no share in levels.mix")
        return int(level)

    def landing_delay_pmf(self, kind: str) -> Dict[int, float]:
        """The decision-to-stamp delay distribution of ``kind`` ('play' | 'ability') BEFORE the stamp rule,
        ``{g: p}`` over ticks with p > 0 in increasing g, summing to 1:
        ``(1 - m) * normalise(base) + m * U{0..uniform_max}``, m = ``uniform_mix`` if ``kind`` is in
        ``uniform_kinds`` else 0, where ``base`` is ``rollout.landing_delay.<kind>_pmf`` or, when that is null,
        the legacy inclusive range ``sim2real.landing_delay.<kind>``. Legacy behaviour (uniform over the ranges):
        both pmfs null, uniform_mix 0."""

        if kind not in LANDING_KINDS:
            raise ConfigError(f"landing delay kind must be one of {LANDING_KINDS}, not {kind!r}")
        ld = self.rollout.landing_delay
        pmf = getattr(ld, f"{kind}_pmf")
        if pmf is None:
            lo, hi = getattr(self.sim2real.landing_delay, kind)
            pmf = {g: 1.0 for g in range(int(lo), int(hi) + 1)}
        tot = float(sum(pmf.values()))
        mix = float(ld.uniform_mix) if kind in ld.uniform_kinds else 0.0
        out: Dict[int, float] = {}
        for g, w in pmf.items():
            out[int(g)] = out.get(int(g), 0.0) + (1.0 - mix) * float(w) / tot
        if mix > 0.0:
            for g in range(int(ld.uniform_max) + 1):
                out[g] = out.get(g, 0.0) + mix / (int(ld.uniform_max) + 1)
        return {g: p for g, p in sorted(out.items()) if p > 0.0}

    def stamp_tick(self, tick: int) -> int:
        """The stamp of a click that would be stamped on ``tick``: :func:`apply_stamp_rule` with
        ``rollout.stamp_rule`` (unchanged when the rule is disabled)."""

        sr = self.rollout.stamp_rule
        if not sr.enabled:
            if isinstance(tick, bool) or not isinstance(tick, int):
                raise ConfigError(f"stamp tick must be an int, got {tick!r}")
            return tick
        return apply_stamp_rule(tick, tuple(sr.blocked_mod10), int(sr.push_to_mod10))

    def stamped_delay_pmf(self, kind: str, decision_tick: int) -> Dict[int, float]:
        """Effective decision-to-stamp delay ``{stamp - decision: p}`` of a ``kind`` decision taken on
        ``decision_tick``: g ~ :meth:`landing_delay_pmf`, then the stamp rule. Depends on the tick only through
        ``decision_tick % 10``."""

        if isinstance(decision_tick, bool) or not isinstance(decision_tick, int) or decision_tick < 0:
            raise ConfigError(f"decision_tick must be an int >= 0, got {decision_tick!r}")
        out: Dict[int, float] = {}
        for g, p in self.landing_delay_pmf(kind).items():
            e = self.stamp_tick(decision_tick + g) - decision_tick
            out[e] = out.get(e, 0.0) + p
        return dict(sorted(out.items()))

    def max_stamped_delay(self, kind: str) -> int:
        """Largest effective decision-to-stamp delay of ``kind`` over the ten decision phases (the stamp rule
        included): what the env must bound with the own-command sighting, ``max_stamped_delay +
        own_sighting_delay_range()[1] < DISPATCH_LEAD``."""

        return max(max(self.stamped_delay_pmf(kind, 10 + ph)) for ph in range(10))

    def max_stamp_push(self) -> int:
        """Largest number of ticks the stamp rule adds to a stamp (0 when disabled)."""

        sr = self.rollout.stamp_rule
        if not sr.enabled or not sr.blocked_mod10:
            return 0
        return max((int(sr.push_to_mod10) - int(r)) % 10 or 10 for r in sr.blocked_mod10)

    def arrival_j_pmf(self) -> Dict[int, float]:
        """``rollout.arrival.j_pmf`` normalised (``{j: p}``, p > 0, increasing j): the per-game draw of j."""

        ar = self.rollout.arrival
        tot = float(sum(ar.j_pmf.values()))
        return {int(j): float(w) / tot for j, w in sorted(ar.j_pmf.items()) if w > 0.0}

    def arrival_delay_pmf(self, stamp_tick: int) -> Dict[int, float]:
        """``{delta_arr: p}`` of an opponent command stamped on ``stamp_tick`` under the batched model, with j
        unknown (marginal over the per-game pmf): delta_arr = :func:`batched_first_seen` - stamp (period -
        stamp % period + j). Within one game j is fixed, so a game draws j once (:meth:`arrival_j_pmf`) and
        every command follows from its stamp.
        The 'table' model has no closed form here (the legacy tokenizer table): ConfigError."""

        ar = self.rollout.arrival
        if ar.model != "batched":
            raise ConfigError(f"rollout.arrival.model {ar.model!r}: the arrival is the table's per-command draw")
        if isinstance(stamp_tick, bool) or not isinstance(stamp_tick, int) or stamp_tick < 0:
            raise ConfigError(f"stamp_tick must be an int >= 0, got {stamp_tick!r}")
        out: Dict[int, float] = {}
        for j, p in self.arrival_j_pmf().items():
            d = batched_first_seen(stamp_tick, j, int(ar.period)) - stamp_tick
            out[d] = out.get(d, 0.0) + p
        return dict(sorted(out.items()))

    def own_sighting_model(self) -> str:
        """How a client seat reads its own command's real stamp (:data:`OWN_SIGHTING_MODEL_OF_ARRIVAL`):
        :data:`OWN_SIGHTING_BATCHED` with ``rollout.arrival.model`` 'batched' (the default), the legacy
        :data:`OWN_SIGHTING_R0_RANGE` with 'table'."""

        return OWN_SIGHTING_MODEL_OF_ARRIVAL[self.rollout.arrival.model]

    def own_sighting_stamp_phases(self) -> Tuple[int, ...]:
        """The stamp phases (tick % 10) a client seat's own command can have: every phase the stamp rule does not
        block (all ten when ``rollout.stamp_rule`` is disabled)."""

        sr = self.rollout.stamp_rule
        blocked = set(int(x) for x in sr.blocked_mod10) if sr.enabled else set()
        return tuple(ph for ph in range(10) if ph not in blocked)

    def own_sighting_delay_pmf(self, stamp_tick: int) -> Dict[int, float]:
        """``{first_seen - stamp: p}`` of an own command stamped on ``stamp_tick`` with the game's j unknown:
        'batched' = :meth:`arrival_delay_pmf` (the same j pmf and rule as the opponent arrivals; within a game the
        own and opponent arrivals share the one j), 'table' = uniform over :data:`OWN_SIGHTING_R0`."""

        if self.own_sighting_model() == OWN_SIGHTING_R0_RANGE:
            if isinstance(stamp_tick, bool) or not isinstance(stamp_tick, int) or stamp_tick < 0:
                raise ConfigError(f"stamp_tick must be an int >= 0, got {stamp_tick!r}")
            lo, hi = OWN_SIGHTING_R0
            return {d: 1.0 / (hi - lo + 1) for d in range(lo, hi + 1)}
        return self.arrival_delay_pmf(stamp_tick)

    def own_sighting_delay_range(self) -> Tuple[int, int]:
        """Smallest and largest own-command sighting delay (first_seen - stamp) over the stamp phases the stamp rule
        allows (:meth:`own_sighting_stamp_phases`) and the j with positive weight: what the strict lead bound uses
        (``max_stamped_delay(kind) + hi < DISPATCH_LEAD``); 'table': :data:`OWN_SIGHTING_R0`."""

        if self.own_sighting_model() == OWN_SIGHTING_R0_RANGE:
            return tuple(OWN_SIGHTING_R0)   # type: ignore[return-value]
        period = int(self.rollout.arrival.period)
        phases = set(self.own_sighting_stamp_phases())
        js = [j for j, p in self.arrival_j_pmf().items() if p > 0.0]
        ds = [own_first_seen(st, j, period) - st for st in range(10 * period) if st % 10 in phases for j in js]
        return min(ds), max(ds)

    def own_sighting_record(self) -> Dict[str, Any]:
        """The own-command sighting rule as recorded in :meth:`env_record` (``own_sighting``), JSON-ready."""

        lo, hi = self.own_sighting_delay_range()
        if self.own_sighting_model() == OWN_SIGHTING_R0_RANGE:
            return {"model": OWN_SIGHTING_R0_RANGE, "rule": "first_seen = stamp + U{lo..hi}, drawn per command",
                    "range": [int(OWN_SIGHTING_R0[0]), int(OWN_SIGHTING_R0[1])], "delay_range": [lo, hi],
                    "pending_visible_from": "decision"}
        return {"model": OWN_SIGHTING_BATCHED,
                "rule": "first_seen = period * (stamp // period + 1) + j",
                "j": "rollout.arrival j of the observing seat (one draw per game, shared with opponent arrivals)",
                "period": int(self.rollout.arrival.period), "stamp_phases": list(self.own_sighting_stamp_phases()),
                "delay_range": [lo, hi], "pending_visible_from": "decision"}

    def env_record(self, tables: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        """The ENV parameters of a run (never part of the model compatibility digest):
        the env path, sim-to-real on/off, the landing-delay config and its resolved pmfs (before the stamp rule),
        the stamp rule, the arrival model with its normalised j pmf (and the legacy table path), the own-command
        sighting rule (:meth:`own_sighting_record`: batched with the arrival's per-game j; the legacy range
        only with the 'table' arrival), delta_lag, min_tap_tick, the client mask, the
        client-rule seats and ``threat_from_filtered_state``. ``tables``: ``il_v7._tables_digest_map`` of the
        tokenizer tables loaded at run time; only its env part (:func:`env_tables`, the arrival table) is
        recorded. JSON-ready.

        ``client_mask`` is recorded here because it is part of the environment's client rules, but unlike the
        timing parameters it changes the policy's log pi (a sampling-only slot mask the learner recomputes): a
        model trained with it must be deployed with it (``describe()['invariants']['client_mask_deploy']``)."""

        r, s2 = self.rollout, self.sim2real
        d = self.to_dict()
        return {
            "schema": ENV_SCHEMA,
            "env_path": r.env_path,
            "sim2real_enabled": bool(s2.enabled),
            "landing_delay": d["rollout"]["landing_delay"],
            "legacy_landing_delay": d["sim2real"]["landing_delay"],
            "landing_delay_pmf": {k: {str(g): p for g, p in self.landing_delay_pmf(k).items()}
                                  for k in LANDING_KINDS},
            "stamp_rule": d["rollout"]["stamp_rule"],
            "arrival": d["rollout"]["arrival"],
            "arrival_j_pmf": {str(j): p for j, p in self.arrival_j_pmf().items()},
            "delta_arr_table": s2.delta_arr_table,
            "own_sighting": self.own_sighting_record(),
            "delta_lag": d["sim2real"]["delta_lag"],
            "min_tap_tick": r.min_tap_tick,
            "client_mask": bool(r.client_mask),
            "client_rules_seats": list(r.client_rules_seats),
            "threat_from_filtered_state": bool(s2.threat_from_filtered_state),
            "tables": env_tables(tables),
        }

    def env_digest(self, tables: Optional[Mapping[str, str]] = None) -> str:
        """sha256 of the canonical JSON of :meth:`env_record` (re-validated first): the CLIENT-TIMING env
        identity. A different value is a different environment, never a model incompatibility.

        Pass the run-time tables (``tables=il_v7._tables_digest_map(tables.source)``) and record that one value,
        under the key ``client_env_digest`` (env freeze, run records, shard metadata), so it never shares a name
        with ``tools/ppo_v7/gate.py::env_digest`` (schema ppo-v7-gate-env.1, stored as ``env_digest`` by
        the block runner and the gate plans). The gate digest hashes the whole rollout / sim2real / levels
        sections plus the compat digest and the env runner's identity, so it already moves whenever this one
        moves (the tables excepted, which reach it through the compat digest's code files); ``describe()``
        shows the table-less value as ``client_env_digest_without_tables`` for reference only."""

        self.validate()
        blob = json.dumps(self.env_record(tables), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
        return hashlib.sha256(blob.encode("ascii")).hexdigest()

    def opponent_settings(self) -> Dict[str, Dict[str, Any]]:
        """The per-opponent settings table, one entry per frozen opponent kind, with exactly the fields
        of the case builder's ``OpponentSpec`` (``OpponentSpec(**row)``): checkpoint (None = not set yet; a
        snapshot's is picked per update from the pool), temperature (None = T*_IL not measured), gate_bias
        ([2, 3] base / threat rows, always explicit: all zeros = no bias; never None, which an older harness
        reads as 'the checkpoint's own bias'), hazard_dt (block | interval | None = not set) and
        code_tree ('run' = this run's frozen copy). Every value comes from config fields, so the digest
        covers the table."""

        o, out = self.opponents, {}
        for kind in FROZEN_TEMPERATURE_KINDS:
            model = MODEL_OF_KIND[kind]
            if model in OLD_TREE_MODELS:
                hz, tree = o.hazard_dt[model], o.code_tree[model]
            else:
                hz, tree = self.rollout.hazard_dt, RUN_CODE_TREE
            gb = [[0.0] * N_FAMILIES, [0.0] * N_FAMILIES]       # explicit zeros = no bias, never None
            if model in GATE_BIAS_MODELS:
                gb = [[float(x) for x in o.gate_bias[model]["base"]], [float(x) for x in o.gate_bias[model]["threat"]]]
            out[kind] = {"name": kind, "checkpoint": o.checkpoint.get(model), "temperature": o.temperature[kind],
                         "gate_bias": gb, "hazard_dt": hz, "code_tree": tree}
        return out

    def checkpoint_pins(self) -> Dict[str, Dict[str, Optional[str]]]:
        """``{model: {'path', 'sha256'}}`` of the pinned frozen checkpoints."""

        o = self.opponents
        return {m: {"path": o.checkpoint[m], "sha256": o.checkpoint_sha256[m]} for m in CHECKPOINT_KINDS}

    def opponent_pins(self) -> Dict[str, Optional[str]]:
        """The pinned file sha256 per frozen opponent KIND (snapshots: None, they are pinned per update by the
        pool). The rollout writes it as ``GameMeta.opp_settings['checkpoint_sha256']``;
        ``learner.verify_shard_pins`` compares."""

        out: Dict[str, Optional[str]] = {}
        for kind in FROZEN_TEMPERATURE_KINDS:
            model = MODEL_OF_KIND[kind]
            out[kind] = self.opponents.checkpoint_sha256.get(model) if model in CHECKPOINT_KINDS else None
        return out

    def chain_args(self) -> List[str]:
        """The chain script's arguments for the block budget (``chain`` section)."""

        return ["--timeout", str(self.chain.block_timeout_s), "--kill-after", str(self.chain.kill_after_s)]

    def actor_amp(self) -> Optional[str]:
        """The autocast of every inference server of this run (theta, snapshots, IL opponents, gate servers):
        'fp16' with an fp16 backbone (actors and learner use the same half precision), else None. A bf16
        backbone stays a learner-only setting."""
        return "fp16" if self.policy.backbone_dtype == "fp16" else None

    SERVER_KEYS = ("flush_rows", "max_wait_ms", "opponent_flush_rows", "opponent_max_wait_ms")

    def server_settings(self) -> Optional[Dict[str, Any]]:
        """``EpisodeSettings.server``: ``rollout.server`` plus the actors' autocast, or None when every value is the
        default (fp32 actors, default servers), so that such records and digests do not change."""
        sv = self.rollout.server
        d: Dict[str, Any] = {"lean": bool(sv.lean), "amp": self.actor_amp()}
        d.update({k: getattr(sv, k) for k in self.SERVER_KEYS})
        return d if (d["lean"] or d["amp"] or any(d[k] is not None for k in self.SERVER_KEYS)) else None

    def monitor_thresholds(self) -> Dict[str, Any]:
        """Exactly the fields and units of ``amini.ppo.monitor.Thresholds`` (for ``from_mapping``).

        Rollback is converted from pp to a fraction; the update-0 log-prob tolerance is per row type, from
        ``max_abs_dlogp_fp32`` for an fp32 backbone or the measured ``dlogp_tol_bf16`` for bf16.
        """

        m, g, pm = self.monitor, self.gates, self.monitor.parity
        if self.policy.backbone_dtype == "fp32":
            tol = {rt: pm.max_abs_dlogp_fp32 for rt in ROW_TYPES}
        elif self.policy.backbone_dtype == "fp16":
            unset = [rt for rt in ROW_TYPES if pm.dlogp_tol_fp16[rt] is None]
            if unset:
                raise ConfigError(f"monitor.parity.dlogp_tol_fp16: {unset} not measured yet for the fp16 backbone")
            tol = {rt: float(pm.dlogp_tol_fp16[rt]) for rt in ROW_TYPES}
        else:
            unset = [rt for rt in ROW_TYPES if pm.dlogp_tol_bf16[rt] is None]
            if unset:
                raise ConfigError(f"monitor.parity.dlogp_tol_bf16: {unset} not measured yet for the bf16 backbone")
            tol = {rt: float(pm.dlogp_tol_bf16[rt]) for rt in ROW_TYPES}
        return {
            "identity_logp_tol": tol,
            "identity_ratio_tol": pm.ratio_tol,
            "identity_kl_tol": pm.kl_theta0_ref_max,
            "early_stop_window": self.early_stop.lr_halve_window,
            "early_stop_frac": self.early_stop.lr_halve_if_stopped_frac,
            "clip_max": m.clip_frac_max,
            "clip_window": m.clip_frac_consecutive,
            "temp_lr_factor": m.temp_lr_mult,
            "temp_lr_floor": m.temp_lr_floor,
            "kl_flat_window": m.kl_stall_window,
            "kl_flat_tol": m.kl_stall_band,
            "kl_flat_g1": m.kl_stall_g1_without_promotion,
            "entropy_min_ratio": m.entropy_min_frac_of_il,
            "plays_ratio": tuple(m.plays_per_min_band),
            "card_ratio": tuple(m.card_rate_band),
            "core_ratio": m.core_card_min_frac_of_il,
            "core_window": m.core_card_consecutive,
            "core_rollbacks_before_stop": m.core_rollbacks_before_stop,
            "masked_refusal_ratio": m.legal_refused_max_ratio_vs_il,
            "mask_audit_every": m.mask_recheck_every,
            "drift_every": m.drift_every,
            "forced_alarm_abs": m.forced_frac_alarm_abs,
            "drop_alarm": m.drop_rate_warn,
            "drop_stop": m.drop_rate_stop,
            "opp_nonwait_ratio": m.opp_nonwait_min_frac,
            "ev_overall_min": m.value_ev_overall_min,
            "ev_last_min": m.value_ev_last_third_min,
            "ev_drop": m.value_ev_drop_from_warmup,
            "ev_window": m.value_ev_consecutive,
            "throughput_ratio": m.throughput_min_frac,
            "mem_min_gb": m.min_free_mem_gb,
            "swap_growth_gb": m.swap_growth_gb,
            "swap_window_s": m.swap_window_s,
            "swap_cooldown_s": m.swap_cooldown_s,
            "swap_mem_guard_gb": m.swap_mem_guard_gb,
            "code19_stop_per_update": m.code19_stop_per_update,
            "code19_stop_consecutive": m.code19_stop_consecutive,
            "reduce_slots_n": m.swap_slot_reduction,
            "nonfinite_stop": m.nan_skip_consecutive_stop,
            "rollback_delta": g.rollback_delta_pp / 100.0,
            "rollback_consecutive": g.rollback_consecutive,
            "rollback_lr_factor": g.rollback_lr_mult,
            "double_rollback_updates": g.rollback_beta_double_window,
            "double_rollback_beta": g.rollback_beta_mult,
            "beta_floor": self.kl.beta_min,
            "beta_cap": self.kl.beta_max,
            "kl_step_beta": m.kl_stall_beta_mult,
            "kl_step_lr": m.lr_stall_mult,
            "kl_skip_growth": m.kl_step_skip_growth,
            "kl_skip_g1_move": m.kl_step_skip_g1_move_pp / 100.0,
            "kl_grad_ratio_min": m.kl_grad_ratio_min,
            "tiny_share_max": m.tiny_ratio_share_max,
            "trend_window": m.gate_trend_window,
            "trend_act_rel_max": m.gate_trend_act_rel_max,
            "trend_adv_gap_max": m.gate_trend_adv_gap_max,
            "entropy_ref_max_ratio": m.entropy_max_frac_of_ref,
            "flip_max": m.flip_rate_max,
            "stall_start_g1": g.stall_g1_without_promotion,
            "stall_min_updates": g.stall_step_min_updates,
            "stall_games_per_update": g.stall_games_per_update,
            "final_stop_g1": g.stall_stop_g1,
            "release_after_g1": g.release_after_g1_without_new,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Plain JSON-ready dict (tuples become lists, int keys become strings, -0.0 becomes 0.0)."""
        out = _canon(json.loads(json.dumps(asdict(self))))
        # These switches are opt-in additions.  Omitting their default values keeps
        # old run/config digests and serialized config records byte-for-byte stable.
        if out.get("rollout", {}).get("games_per_worker") == 1:
            out["rollout"].pop("games_per_worker", None)
        if not out.get("monitor", {}).get("swap_cooldown_s"):
            out.get("monitor", {}).pop("swap_cooldown_s", None)
        if out.get("monitor", {}).get("swap_mem_guard_gb") is None:
            out.get("monitor", {}).pop("swap_mem_guard_gb", None)
        if out.get("monitor", {}).get("host_memory_events") is False:
            out.get("monitor", {}).pop("host_memory_events", None)
        for key in ("code19_stop_per_update", "code19_stop_consecutive"):
            if out.get("monitor", {}).get(key) == 1:
                out["monitor"].pop(key, None)
        if out.get("gates", {}).get("sprt_stop_alpha") is None:
            out.get("gates", {}).pop("sprt_stop_alpha", None)
        if out.get("monitor", {}).get("opp_score_alarm") is None:        # off: omitted
            for key in OPP_SCORE_KEYS:
                out.get("monitor", {}).pop(key, None)
        gh = out.get("gates")                                             # the 'crn' flow: h2h keys omitted
        if gh is not None and gh.get("flow") == "crn":
            for key in ("flow", "h2h_stop_alpha", "h2h_h1_pp", "h2h_guard_pairs", "h2h_guard_alpha"):
                gh.pop(key, None)
        sv = out.get("rollout", {}).get("server")                         # all defaults: omitted
        if sv is not None and not sv.get("lean") and all(sv.get(k) is None for k in self.SERVER_KEYS):
            out["rollout"].pop("server", None)
        par = out.get("monitor", {}).get("parity")                       # unset fp16 tolerances: omitted
        if par is not None and all(v is None for v in (par.get("dlogp_tol_fp16") or {}).values()):
            par.pop("dlogp_tol_fp16", None)
        gt = out.get("gates")
        if gt is not None:
            for key in ("games_per_worker", "workers"):
                if gt.get(key) is None:
                    gt.pop(key, None)
        rem = out.get("rollout", {}).get("remote")
        if rem is not None and rem.get("tail_speed_margin") is None:
            rem.pop("tail_speed_margin", None)
        ops = out.get("opponents")
        if ops is not None and not ops.get("general_share"):      # off: omitted
            ops.pop("general_share", None)
        if ops is not None and ops.get("general_opponents") == "self":    # the default: omitted
            ops.pop("general_opponents", None)
        gp = out.get("gates")
        if gp is not None and gp.get("panel_source") == "deck":           # the default: omitted
            gp.pop("panel_source", None)
        if ops is not None and not ops.get("snapshot_anchors"):          # no anchors: omitted
            ops.pop("snapshot_anchors", None)
        if gp is not None and not gp.get("g2_anchor"):
            gp.pop("g2_anchor", None)
        if gp is not None and not gp.get("g2_every_g1"):
            gp.pop("g2_every_g1", None)
        if ops is not None:                                  # no matchup games: omitted
            for key in ("matchup_share", "matchup_deck", "matchup_models", "matchup_weights", "matchup_il_share",
                        "matchup2_share", "matchup2_deck", "matchup2_models", "matchup2_weights",
                        "matchup2_il_share"):
                if not ops.get(key):
                    ops.pop(key, None)
        if gp is not None:
            for key in ("matchup_games", "matchup_anchor", "matchup2_games", "matchup2_anchor"):
                if not gp.get(key):
                    gp.pop(key, None)
        wu = out.get("critic", {}).get("warmup")
        if wu is not None and wu.get("early_stop_min_updates") is None:
            wu.pop("early_stop_min_updates", None)
        if wu is not None and wu.get("overlap") is False:
            wu.pop("overlap", None)
        if wu is not None and wu.get("plateau_gain") is None:      # off: omitted
            wu.pop("plateau_gain", None)
            wu.pop("plateau_window", None)
        learner = out.get("learner")
        if learner and learner.get("compile_mode") == "":
            out.pop("learner", None)
        return out

    def digest(self) -> str:
        """sha256 of the canonical JSON of the (re-validated) resolved config (sorted keys, no spaces)."""

        self.validate()
        blob = json.dumps({"schema": CONFIG_SCHEMA, "config": self.to_dict()}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(blob.encode("ascii")).hexdigest()

    def describe(self) -> Dict[str, Any]:
        """Run-record dump: schema, digest, the resolved config, derived values, readiness and invariants."""

        try:
            thresholds: Optional[Dict[str, Any]] = json.loads(json.dumps(self.monitor_thresholds()))
        except ConfigError:
            thresholds = None                   # bf16 tolerances not measured yet (reported by missing)
        return {
            "schema": CONFIG_SCHEMA,
            "digest": self.digest(),            # validates first
            "config": self.to_dict(),
            "derived": {
                "mix": {ph: self.effective_mix(ph) for ph in PHASES},
                "il_share": {ph: sum(self.effective_mix(ph)[k] for k in IL_KINDS) for ph in PHASES},
                "trained_seats_per_game": {ph: self.trained_seats_per_game(ph) for ph in PHASES},
                "king_level_by_level": {str(lv): lv for lv in LEVELS if self.levels.mix[lv] > 0},
                "tower_troop_key": TOWER_TROOP_GIDS.get(self.user.tower_troop) if self.user.tower_troop else None,
                "lambda_per_block": self.gae.lambda_tick ** 10,
                "lambda_scale_ticks": (1.0 / (1.0 - self.gae.lambda_tick)) if self.gae.lambda_tick < 1 else None,
                "kl_reference_before_first_promotion": "il",
                "kl_reference_after_first_promotion": self.user.kl_reference_after_promotion,
                "g1c_enabled": self.user.h1_reading == "crn_then_seatswap",
                "bc_weight": self.bc.weight if self.user.bc_enabled else 0.0,
                "monitor_thresholds": thresholds,
                "opponent_settings": self.opponent_settings(),
                "chain_args": self.chain_args(),
                "landing_delay_pmf": {k: {str(g): p for g, p in self.landing_delay_pmf(k).items()}
                                      for k in LANDING_KINDS},
                "client_env_digest_without_tables": self.env_digest(),
                "max_stamp_push": self.max_stamp_push(),
                "model_roles": dict(MODEL_ROLES),
            },
            "sampling": {"KAPPA": {s: str(k) for s, k in G.KAPPA.items()}, "WEIGHT": dict(G.WEIGHT)},
            "readiness": {st: self.missing(st) for st in READINESS_STAGES},
            "invariants": {                     # code-level rules the learner / actor must honour
                "gate_bias_train": None, "gate_bias_reference": None, "gate_bias_deploy": 0.0,
                "gate_temperature": 1.0, "rollout_temperature": 1.0, "play_blocked": "fold_into_wait",
                "kl_entropy": "folded_families", "ema_for_acting": False, "deploy_weights_key": "model_ema",
                "sim2real_in_client_package": False, "truncated_games": "dropped_no_bootstrap",
                "runtime_lr_beta": "learner_state_not_config",
                # a model trained with the client slot mask ships with it (the mask is in the actor's and the
                # learner's log pi)
                "client_mask_deploy": "required" if self.rollout.client_mask else "off"},
        }


# ============================================================================ building
def _canon(obj: Any) -> Any:
    """-0.0 -> 0.0 everywhere (so equal configs have one digest)."""

    if isinstance(obj, float):
        return obj + 0.0
    if isinstance(obj, list):
        return [_canon(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items()}
    return obj


def _coerce_key(tp: Any, k: Any, path: str) -> Any:
    if tp is int:
        if isinstance(k, bool):
            raise ConfigError(f"{path}: key {k!r} is not an integer")
        if isinstance(k, int):
            return k
        if isinstance(k, str) and k.strip().isdigit():
            return int(k.strip())
        raise ConfigError(f"{path}: key {k!r} is not an integer")
    return str(k)


def _coerce(tp: Any, v: Any, path: str) -> Any:
    """Check / convert one YAML value against a type hint."""

    if tp is str and path == "learner.compile_mode" and v is None:
        return ""                              # an override 'learner.compile_mode=' means off
    origin = get_origin(tp)
    if origin is Union:
        args = get_args(tp)
        if v is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed")
        inner = [a for a in args if a is not type(None)]
        return _coerce(inner[0], v, path)
    if v is None:
        raise ConfigError(f"{path}: null is not allowed")
    if is_dataclass(tp):
        if not isinstance(v, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(v).__name__}")
        return _build(tp, v, path)
    if tp is bool:
        if not isinstance(v, bool):
            raise ConfigError(f"{path}: expected true/false, got {v!r}")
        return v
    if tp is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{path}: expected an integer, got {v!r}")
        return v
    if tp is float:
        if isinstance(v, str):                  # PyYAML 1.1 reads '1e-5' as a string
            try:
                v = float(v)
            except ValueError:
                raise ConfigError(f"{path}: expected a number, got {v!r}") from None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {v!r}")
        try:
            f = float(v)
        except OverflowError:
            raise ConfigError(f"{path}: number too large, got an integer of {len(str(abs(v)))} digits") from None
        if not math.isfinite(f):
            raise ConfigError(f"{path}: must be finite, got {v!r}")
        return f + 0.0                          # -0.0 -> 0.0
    if tp is str:
        # The learner compile switch uses the empty string as its explicit off value;
        # all other string fields retain the strict non-empty contract.
        if isinstance(v, str) and v == "" and path == "learner.compile_mode":
            return v
        if not isinstance(v, str) or v == "":
            raise ConfigError(f"{path}: expected a non-empty string, got {v!r}")
        return v
    if origin in (list, List):
        if not isinstance(v, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {v!r}")
        (arg,) = get_args(tp)
        return [_coerce(arg, x, f"{path}[{i}]") for i, x in enumerate(v)]
    if origin in (tuple, Tuple):
        args = get_args(tp)
        if not isinstance(v, (list, tuple)) or len(v) != len(args):
            raise ConfigError(f"{path}: expected a list of {len(args)} values, got {v!r}")
        return tuple(_coerce(a, x, f"{path}[{i}]") for i, (a, x) in enumerate(zip(args, v)))
    if origin in (dict, Dict):
        if not isinstance(v, dict):
            raise ConfigError(f"{path}: expected a mapping, got {v!r}")
        key_t, val_t = get_args(tp)
        out: Dict[Any, Any] = {}
        for k, x in v.items():
            kk = _coerce_key(key_t, k, path)
            if kk in out:
                raise ConfigError(f"{path}: key {kk!r} given twice")
            out[kk] = _coerce(val_t, x, f"{path}.{k}")
        return out
    raise ConfigError(f"{path}: unsupported type {tp!r}")     # pragma: no cover


def _build(cls: Any, data: Dict[str, Any], path: str) -> Any:
    """Build dataclass ``cls`` from ``data``: unknown keys refused, missing keys take the defaults."""

    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls) if f.init]
    unknown = sorted(set(map(str, data)) - set(names))
    if unknown:
        raise ConfigError(f"{path or '<root>'}: unknown key(s) {unknown}; allowed: {names}")
    kw = {k: _coerce(hints[k], data[k], f"{path}.{k}" if path else k) for k in names if k in data}
    return cls(**kw)


def from_dict(data: Dict[str, Any]) -> PPOConfig:
    """Strictly build and validate a PPOConfig from a (YAML) mapping."""

    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    try:
        return _build(PPOConfig, data, "")
    except ConfigError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:     # pragma: no cover - defensive
        raise ConfigError(str(exc)) from exc


def lint_yaml_text(text: str) -> None:
    """Refuse what load_yaml would read silently wrong: a key repeated in one mapping (last-wins in both
    dialects) and bare scalars that PyYAML and the il_v7 subset resolve differently (``_AMBIGUOUS_SCALARS``)."""

    stack: List[Tuple[int, set, str]] = [(-1, set(), "")]
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, rest = line.strip().partition(":")
        key = key.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        _, seen, parent = stack[-1]
        path = f"{parent}.{key}" if parent else key
        if key in seen:
            raise ConfigError(f"line {no}: key {path!r} appears twice in one mapping (the earlier one would be dropped)")
        seen.add(key)
        v = rest.strip()
        if v == "":
            stack.append((indent, set(), path))
            continue
        items = v[1:-1].split(",") if v.startswith("[") and v.endswith("]") else [v]
        for it in items:
            if it.strip() in _AMBIGUOUS_SCALARS:
                raise ConfigError(f"line {no}: {path}: bare {it.strip()!r} reads differently under PyYAML and "
                                  "the il_v7 subset; quote it (or write null / true / false)")


def apply_overrides(data: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``a.b.c=value`` overrides (value parsed by the same YAML subset) to a nested mapping copy."""

    from amini.train.il_v7 import load_yaml
    out = json.loads(json.dumps(data))
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override {item!r}: expected key.path=value")
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = node[p] = {}
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {item!r}: {p!r} is not a mapping")
            node = nxt
        text = f"v: {raw.strip() if raw.strip() else 'null'}"
        lint_yaml_text(text)
        node[parts[-1]] = load_yaml(text)["v"]
    return out


def load_config(path: Union[str, Path], overrides: Sequence[str] = ()) -> PPOConfig:
    """Lint and read a YAML run file (il_v7.load_yaml subset), apply overrides, build and validate."""

    from amini.train.il_v7 import load_yaml
    text = Path(path).read_text(encoding="utf-8")
    lint_yaml_text(text)
    data = load_yaml(text)
    if overrides:
        data = apply_overrides(data, overrides)
    return from_dict(data)


# ============================================================================ recorded mid-run changes
# Keys a recorded config change may modify in the middle of a run: the pilot retry tweaks (policy.lr / kl.beta /
# gae.lambda_tick), the stall ladder's steps (games per update; a longer lambda_tick, more critic epochs), and the
# calibration after the pilot: the early-stop limits, the alarm thresholds that stay null until calibrated, and
# the canary alarm line, so the main phase continues the same run with the calibrated numbers. Everything else
# (other thresholds, mix, env, weights, reward) changes the experiment and needs a new run.
RUNTIME_CHANGE_KEYS: Tuple[str, ...] = (
    "kl.beta", "policy.lr", "gae.lambda_tick", "rollout.games_per_update", "critic.epochs",
    "early_stop.joint_k3", "early_stop.play_k3", "early_stop.gate_kl", "early_stop.lr_halve_if_stopped_frac",
    "early_stop.lr_halve_window",
    "monitor.tiny_ratio_share_max", "monitor.gate_trend_act_rel_max", "monitor.gate_trend_adv_gap_max",
    "monitor.entropy_max_frac_of_ref", "monitor.flip_rate_max",
    "gates.canary_alarm_lb")
# The runtime keys that feed ``monitor.Thresholds`` (field names of ``monitor_thresholds()``): a change of one of
# them is applied to the running monitor with ``Monitor.apply_thresholds(Thresholds.from_mapping(
# new_cfg.monitor_thresholds()))``, which refuses any other field (``monitor.RUNTIME_THRESHOLD_FIELDS``).
# The other runtime keys are read from the live config: early_stop.joint_k3 / play_k3 / gate_kl by the
# learner each update, gates.canary_alarm_lb by the gate runner at each gate.
RUNTIME_THRESHOLD_KEYS: Dict[str, str] = {
    "early_stop.lr_halve_if_stopped_frac": "early_stop_frac", "early_stop.lr_halve_window": "early_stop_window",
    "monitor.tiny_ratio_share_max": "tiny_share_max", "monitor.gate_trend_act_rel_max": "trend_act_rel_max",
    "monitor.gate_trend_adv_gap_max": "trend_adv_gap_max", "monitor.entropy_max_frac_of_ref": "entropy_ref_max_ratio",
    "monitor.flip_rate_max": "flip_max"}
CONFIG_CHANGE_SCHEMA = "ppo-v7-config-change.1"
# ``policy.rollout_lag`` is switchable mid-run, but ONLY at a block boundary and only when a caller asks for it
# explicitly (``allow_lag=True``: the driver's pending change, the block runner's ``change`` command and a fork).
# It is kept out of RUNTIME_CHANGE_KEYS on purpose: that list (and every BASE caller of with_changes /
# config_change) never includes it, so a ladder / pilot-retry change can never touch the lag and a default
# record has no new field. A record that changes the lag
# carries ``"lag_switch": [old, new]``; ``check_config_change`` re-derives it, so a record without the field can
# never smuggle a lag change and a lag record cannot be replayed onto another pair of configs.
LAG_SWITCH_KEY = "policy.rollout_lag"
LAG_SWITCH_KEYS: Tuple[str, ...] = (LAG_SWITCH_KEY,)


# The learner compile switches are part of a run's BASE config (in the config digest): they are never a mid-run
# change (no ladder / operator change can touch them, and the whitelist above does not list them).
# ``with_changes(..., allow_compile=True)`` only builds such a config from another one (an eager-vs-compiled
# comparison); a run never records such a change.
COMPILE_SWITCH_KEYS: Tuple[str, ...] = ("learner.compile_mode", "learner.compile_dynamic")


def _change_keys(allow_lag: bool, allow_compile: bool = False) -> Tuple[str, ...]:
    keys = RUNTIME_CHANGE_KEYS + LAG_SWITCH_KEYS if allow_lag else RUNTIME_CHANGE_KEYS
    return keys + COMPILE_SWITCH_KEYS if allow_compile else keys


def _leaves(d: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(d, dict):
        out: Dict[str, Any] = {}
        for k, v in d.items():
            out.update(_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: d}


def config_diff(old: PPOConfig, new: PPOConfig) -> Dict[str, Tuple[Any, Any]]:
    """``{dotted.key: (old, new)}`` of every leaf that differs (canonical JSON values)."""

    a, b = _leaves(old.to_dict()), _leaves(new.to_dict())
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def with_changes(cfg: PPOConfig, changes: Mapping[str, Any], *, allow_lag: bool = False,
                 allow_compile: bool = False) -> PPOConfig:
    """A copy of ``cfg`` with dotted keys set (e.g. the changes of a pilot retry), validated like a loaded
    file. Only :data:`RUNTIME_CHANGE_KEYS` may be named (plus :data:`LAG_SWITCH_KEYS` with ``allow_lag``: a
    block-boundary lag switch)."""

    allowed = _change_keys(allow_lag, allow_compile)
    bad = sorted(k for k in changes if k not in allowed)
    if bad:
        raise ConfigError(f"{bad} may not change mid-run (allowed: {list(allowed)})")
    d = cfg.to_dict()
    for key, v in changes.items():
        node = d
        parts = key.split(".")
        if parts[0] == "learner" and "learner" not in node:
            node["learner"] = {"compile_mode": "", "compile_dynamic": False}
        if parts[0] == "rollout" and "games_per_worker" not in node.get("rollout", {}) and parts[-1] == "games_per_worker":
            node.setdefault("rollout", {})["games_per_worker"] = 1
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = v
    return from_dict(d)


def config_change(old: PPOConfig, new: PPOConfig, *, update: int, reason: str,
                  allow_lag: bool = False, allow_compile: bool = False) -> Dict[str, Any]:
    """The record of one mid-run config change: old and new digests, the changed keys
    with both values, the update from which the new config applies, and the reason (ladder step, F1, pilot
    retry). Refuses an empty change or a key outside :data:`RUNTIME_CHANGE_KEYS`. The learner writes the
    records into its checkpoint and ``Learner.from_checkpoint`` accepts a new digest only through one.
    ``allow_lag``: :data:`LAG_SWITCH_KEYS` may change too; a record that does change the lag gets
    ``"lag_switch": [old_lag, new_lag]`` (absent otherwise, so every other record is as before).
    ``allow_compile`` (an eager checkpoint loaded with the compile switch on, on a copy, for a comparison):
    :data:`COMPILE_SWITCH_KEYS` may change and the record carries ``"compile_switch": true``. No run tool passes
    it, so a run can never record such a change."""

    if isinstance(update, bool) or not isinstance(update, int) or update < 1:
        raise ConfigError(f"config change: update must be an int >= 1, got {update!r}")
    if not isinstance(reason, str) or not reason.strip():
        raise ConfigError("config change: a non-empty reason is required")
    diff = config_diff(old, new)
    if not diff:
        raise ConfigError("config change: the two configs are identical")
    allowed = _change_keys(allow_lag, allow_compile)
    bad = sorted(k for k in diff if k not in allowed)
    if bad:
        raise ConfigError(f"config change: {bad} may not change mid-run (allowed: {list(allowed)}); "
                          "start a new run instead")
    rec = {"schema": CONFIG_CHANGE_SCHEMA, "update": int(update), "reason": reason.strip(),
           "old_digest": old.digest(), "new_digest": new.digest(),
           "changes": {k: [o, n] for k, (o, n) in diff.items()}}
    if LAG_SWITCH_KEY in diff:
        rec["lag_switch"] = list(diff[LAG_SWITCH_KEY])
    if any(k in diff for k in COMPILE_SWITCH_KEYS):
        rec["compile_switch"] = True
    return rec


def parse_runtime_changes(assignments: Sequence[str], *, allow_lag: bool = False) -> Dict[str, Any]:
    """``{key: value}`` of operator ``key=value`` strings (values read by the same YAML subset as run-file
    overrides, so ``null`` turns an alarm back off). Only :data:`RUNTIME_CHANGE_KEYS` (plus
    :data:`LAG_SWITCH_KEYS` with ``allow_lag``); a key given twice is refused."""

    allowed = _change_keys(allow_lag)
    out: Dict[str, Any] = {}
    for item in assignments:
        key = item.partition("=")[0].strip()
        if key in out:
            raise ConfigError(f"runtime change {key!r} given twice")
        if key not in allowed:
            raise ConfigError(f"{key!r} may not change mid-run (allowed: {list(allowed)})")
        node: Any = apply_overrides({}, [item])
        for part in key.split("."):
            node = node[part]
        out[key] = node
    if not out:
        raise ConfigError("runtime change: no key=value given")
    return out


def runtime_change(cfg: PPOConfig, assignments: Union[Sequence[str], Mapping[str, Any]], *, update: int,
                   reason: str, allow_lag: bool = False) -> Tuple[PPOConfig, Dict[str, Any]]:
    """An operator's mid-run change of a running run (e.g. the calibration after the pilot):
    ``(new_cfg, record)`` with ``new_cfg = with_changes(cfg, changes)`` and ``record =
    config_change(cfg, new_cfg, update=, reason=)``. ``assignments``: ``key=value`` strings
    (:func:`parse_runtime_changes`) or a mapping. ``cfg`` is the run's CURRENT config (base + the state's
    ``config_overrides``), ``update`` the run's next update. The driver then either
    stores the pending change for the next block, which applies it at its first barrier with
    ``Learner.change_config(new_cfg, reason=, update=)`` (that also calls ``Monitor.apply_thresholds`` for
    :data:`RUNTIME_THRESHOLD_KEYS`), or resumes with ``Learner.from_checkpoint(ck, new_cfg,
    config_change=record)``; either way ``state['config_overrides']`` gains the changes and a ``config_change``
    event is written. The run's base config is never edited mid-run. ``allow_lag``: the change
    may also switch :data:`LAG_SWITCH_KEY` (a block-boundary lag switch; the record carries ``lag_switch``)."""

    changes = (dict(assignments) if isinstance(assignments, Mapping)
               else parse_runtime_changes(list(assignments), allow_lag=allow_lag))
    new = with_changes(cfg, changes, allow_lag=allow_lag)
    return new, config_change(cfg, new, update=update, reason=reason, allow_lag=allow_lag)


def check_config_change(record: Mapping[str, Any], old: PPOConfig, new: PPOConfig) -> None:
    """Raise ConfigError unless ``record`` is exactly :func:`config_change` of (``old``, ``new``) with its own
    update and reason (digests, keys and values all re-derived)."""

    if not isinstance(record, Mapping) or record.get("schema") != CONFIG_CHANGE_SCHEMA:
        raise ConfigError(f"not a {CONFIG_CHANGE_SCHEMA} record: {record!r}")
    # a lag switch is accepted only from a record that says so ("lag_switch", re-derived below)
    want = config_change(old, new, update=record.get("update"), reason=record.get("reason"),
                         allow_lag="lag_switch" in record, allow_compile="compile_switch" in record)
    got = json.loads(json.dumps(dict(record)))
    if got != json.loads(json.dumps(want)):
        diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        raise ConfigError(f"config change record does not match the two configs (fields {diff})")


@functools.lru_cache(maxsize=1)
def _object_names() -> frozenset:
    p = CODE_ROOT / OBJECT_VOCAB
    try:
        return frozenset(json.loads(p.read_text(encoding="utf-8"))["ids"])
    except (OSError, ValueError, KeyError) as exc:
        raise ConfigError(f"cannot read the object vocab {p} to check delta_lag override names: {exc}") from exc


# ============================================================================ validation
def _err(path: str, msg: str) -> ConfigError:
    return ConfigError(f"{path}: {msg}")


def _in(path: str, x: float, lo: float, hi: float, lo_open: bool = False, hi_open: bool = False) -> None:
    ok_lo = x > lo if lo_open else x >= lo
    ok_hi = x < hi if hi_open else x <= hi
    if not (ok_lo and ok_hi):
        raise _err(path, f"{x!r} outside {'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}")


def _temp(path: str, t: float) -> None:
    """0.0 (greedy) or a temperature ``logprob.apply_temperature`` accepts (``[T_MIN, 1]``)."""
    _in(path, t, 0.0, 1.0)
    if 0.0 < t < T_MIN:
        raise _err(path, f"{t!r} is neither 0.0 (greedy) nor >= {T_MIN} (logprob.T_MIN)")


def _choice(path: str, x: Any, choices: Sequence[Any]) -> None:
    if x not in choices:
        raise _err(path, f"{x!r} not one of {tuple(choices)}")


def _fixed(path: str, x: Any, want: Any, why: str) -> None:
    if x != want:
        raise _err(path, f"must be {want!r} ({why}), got {x!r}")


def _keys(path: str, d: Dict[Any, Any], want: Sequence[Any]) -> None:
    if set(d) != set(want):
        raise _err(path, f"keys must be exactly {sorted(want)}, got {sorted(d)}")


def _mix(path: str, d: Dict[Any, float], want: Sequence[Any]) -> None:
    _keys(path, d, want)
    for k, v in d.items():
        _in(f"{path}.{k}", v, 0.0, 1.0)
    if abs(sum(d.values()) - 1.0) > 1e-9:
        raise _err(path, f"shares must sum to 1, got {sum(d.values())!r}")


def deck_base_key(key: str) -> str:
    """The card a deck key plays (``card-ev1`` -> ``card``)."""
    for suf in DECK_SLOT_SUFFIXES:
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


def _check_deck_slots(deck: Sequence[str]) -> None:
    """At most MAX_DECK_EVOLUTIONS evolution slots, MAX_DECK_HEROES hero slots and MAX_DECK_SPECIAL of both,
    and one form per card (``card`` and ``card-ev1`` together cannot be fielded); the synthetic case builder
    refuses such a deck too, this catches it at config load."""
    evo = sum(1 for k in deck if k.endswith("-ev1"))
    hero = sum(1 for k in deck if k.endswith("-hero"))
    if evo > MAX_DECK_EVOLUTIONS or hero > MAX_DECK_HEROES or evo + hero > MAX_DECK_SPECIAL:
        raise _err("user.deck", f"{evo} evolutions and {hero} heroes; a real deck holds at most "
                                f"{MAX_DECK_EVOLUTIONS} evolutions, {MAX_DECK_HEROES} heroes and "
                                f"{MAX_DECK_SPECIAL} of both")
    base = [deck_base_key(k) for k in deck]
    dup = sorted({b for b in base if base.count(b) > 1})
    if dup:
        raise _err("user.deck", f"two forms of one card: {dup}")


def _validate(c: PPOConfig) -> None:
    if not c.name:
        raise _err("name", "must be non-empty")
    # --- user switches
    u = c.user
    if u.deck is not None:
        if len(u.deck) != 8 or len(set(u.deck)) != 8 or any(not x for x in u.deck):
            raise _err("user.deck", f"must be 8 distinct card keys, got {u.deck!r}")
        _check_deck_slots(u.deck)
    if u.tower_troop is not None:
        _choice("user.tower_troop", u.tower_troop, tuple(TOWER_TROOP_GIDS))
    _choice("user.kl_reference_after_promotion", u.kl_reference_after_promotion, KL_REFERENCE_CHOICES)
    _choice("user.h1_reading", u.h1_reading, H1_READING_CHOICES)
    _choice("user.opponent_design", u.opponent_design, OPPONENT_DESIGN_CHOICES)
    # --- rollout
    r = c.rollout
    _in("rollout.games_per_update", r.games_per_update, 1, 1 << 16)
    _in("rollout.replicas_per_group", r.replicas_per_group, 1, 64)
    if r.games_per_update % r.replicas_per_group:
        raise _err("rollout.replicas_per_group", "must divide games_per_update")
    _fixed("rollout.temperature", r.temperature, 1.0, "theta_k is sampled at T = 1, never greedy")
    _fixed("rollout.gate_temperature", r.gate_temperature, 1.0, "the gate is sampled at T = 1")
    if r.hazard_dt is not None:
        _choice("rollout.hazard_dt", r.hazard_dt, HAZARD_DT_MODES)
    # the hard ceiling of concurrent games (the rollout's MAX_ENV_SLOTS)
    _in("rollout.max_env_slots", r.max_env_slots, 1, MAX_ENV_SLOTS_CEILING)
    if isinstance(r.games_per_worker, bool) or not isinstance(r.games_per_worker, int):
        raise _err("rollout.games_per_worker", "must be an integer")
    _in("rollout.games_per_worker", r.games_per_worker, 1, 8)
    _choice("rollout.env_path", r.env_path, ENV_PATHS)
    _in("rollout.emu_spot_check_games", r.emu_spot_check_games, 0, 10000)
    ld = r.landing_delay
    for kind in LANDING_KINDS:
        pmf = getattr(ld, f"{kind}_pmf")
        if pmf is None:
            continue
        if not pmf:
            raise _err(f"rollout.landing_delay.{kind}_pmf", "must list at least one delay (or be null: legacy range)")
        for g, w in pmf.items():
            _in(f"rollout.landing_delay.{kind}_pmf.{g}", g, 0, MAX_LANDING_DELAY)   # lead D - tap = 20 stays above g
            _in(f"rollout.landing_delay.{kind}_pmf.{g}", w, 0.0, 1e9)
        if not sum(pmf.values()) > 0.0:
            raise _err(f"rollout.landing_delay.{kind}_pmf", "weights must sum to a positive number")
    _in("rollout.landing_delay.uniform_mix", ld.uniform_mix, 0.0, 1.0)
    _in("rollout.landing_delay.uniform_max", ld.uniform_max, 0, MAX_LANDING_DELAY)
    if len(set(ld.uniform_kinds)) != len(ld.uniform_kinds):
        raise _err("rollout.landing_delay.uniform_kinds", f"must be distinct, got {ld.uniform_kinds!r}")
    for i, kd in enumerate(ld.uniform_kinds):
        _choice(f"rollout.landing_delay.uniform_kinds[{i}]", kd, LANDING_KINDS)
    if r.min_tap_tick is not None:
        _in("rollout.min_tap_tick", r.min_tap_tick, 0, 1000)
    sr = r.stamp_rule
    if len(set(sr.blocked_mod10)) != len(sr.blocked_mod10):
        raise _err("rollout.stamp_rule.blocked_mod10", f"must be distinct, got {sr.blocked_mod10!r}")
    for i, x in enumerate(sr.blocked_mod10):
        _in(f"rollout.stamp_rule.blocked_mod10[{i}]", x, 0, 9)
    _in("rollout.stamp_rule.push_to_mod10", sr.push_to_mod10, 0, 9)
    if sr.push_to_mod10 in sr.blocked_mod10:
        raise _err("rollout.stamp_rule.push_to_mod10", f"{sr.push_to_mod10} is itself blocked {sr.blocked_mod10}")
    ar = r.arrival
    _choice("rollout.arrival.model", ar.model, ARRIVAL_MODELS)
    _in("rollout.arrival.period", ar.period, 1, MAX_LANDING_DELAY)
    if not ar.j_pmf:
        raise _err("rollout.arrival.j_pmf", "must list at least one offset j")
    for j, w in ar.j_pmf.items():
        _in(f"rollout.arrival.j_pmf.{j}", j, 1, ar.period)     # arrives within the batch after the stamp
        _in(f"rollout.arrival.j_pmf.{j}", w, 0.0, 1e9)
    if not sum(ar.j_pmf.values()) > 0.0:
        raise _err("rollout.arrival.j_pmf", "weights must sum to a positive number")
    j_hi = max(j for j, w in ar.j_pmf.items() if w > 0.0)
    if ar.period + j_hi > MAX_LANDING_DELAY:                  # visible before D = stamp + 20 (phase 0 included)
        raise _err("rollout.arrival", f"largest delta_arr {ar.period + j_hi} (period + j, a stamp on a batch "
                                      f"boundary) must stay <= {MAX_LANDING_DELAY}: a command is visible before "
                                      "it executes at D")
    if ar.source_sha256 is not None and not _SHA256_RE.fullmatch(ar.source_sha256):
        raise _err("rollout.arrival.source_sha256", f"must be 64 lowercase hex characters or null, got "
                                                    f"{ar.source_sha256!r}")
    if not r.client_rules_seats:
        raise _err("rollout.client_rules_seats", "must name at least one seat role (train, opponent, gate)")
    if len(set(r.client_rules_seats)) != len(r.client_rules_seats):
        raise _err("rollout.client_rules_seats", f"must be distinct, got {r.client_rules_seats!r}")
    for i, role in enumerate(r.client_rules_seats):
        _choice(f"rollout.client_rules_seats[{i}]", role, CLIENT_SEAT_ROLES)
    rm = r.remote
    if rm.root is not None and not rm.root.startswith("/"):
        raise _err("rollout.remote.root", f"must be an absolute path or null, got {rm.root!r}")
    _in("rollout.remote.lease_s", rm.lease_s, 60, 86400)
    _in("rollout.remote.max_actors", rm.max_actors, 1, 64)
    if rm.tail_speed_margin is not None:
        _in("rollout.remote.tail_speed_margin", rm.tail_speed_margin, 1.0, 100.0)
    sv = r.server
    for key in ("flush_rows", "opponent_flush_rows"):
        if getattr(sv, key) is not None:
            _in(f"rollout.server.{key}", getattr(sv, key), 1, 255)
    for key in ("max_wait_ms", "opponent_max_wait_ms"):
        if getattr(sv, key) is not None:
            _in(f"rollout.server.{key}", getattr(sv, key), 0.0, 1000.0)
    # --- policy
    p = c.policy
    _in("policy.clip_eps", p.clip_eps, 0.0, 1.0, lo_open=True, hi_open=True)
    _in("policy.dual_clip", p.dual_clip, 1.0 + p.clip_eps, 100.0, lo_open=True)   # losses.clipped_objective: c > 1 + eps
    _in("policy.lr", p.lr, 0.0, 1e-3, lo_open=True)
    _choice("policy.lr_schedule", p.lr_schedule, ("constant",))
    for i, b in enumerate(p.betas):
        _in(f"policy.betas[{i}]", b, 0.0, 1.0, hi_open=True)
    _in("policy.adam_eps", p.adam_eps, 0.0, 1e-2, lo_open=True)
    _in("policy.weight_decay", p.weight_decay, 0.0, 0.1)
    _in("policy.grad_clip", p.grad_clip, 0.0, 100.0, lo_open=True)
    _in("policy.grad_clip_raised", p.grad_clip_raised, p.grad_clip, 100.0)
    _in("policy.grad_clip_raise_after_updates", p.grad_clip_raise_after_updates, 1, 1000)
    _in("policy.grad_clip_raise_clipped_frac", p.grad_clip_raise_clipped_frac, 0.0, 1.0, lo_open=True)
    _in("policy.minibatch_seat_games", p.minibatch_seat_games, 1, 1 << 16)
    _in("policy.epochs", p.epochs, 1, 32)
    _fixed("policy.dropout", p.dropout, 0.0, "dropout breaks on-policy ratios")
    _choice("policy.train_scope", p.train_scope, TRAIN_SCOPE_CHOICES)
    _choice("policy.advantage_norm", p.advantage_norm, ("per_update",))
    _in("policy.is_ratio_cap_lag1", p.is_ratio_cap_lag1, 1.0, 100.0)
    _in("policy.gae_trace_cap_lag1", p.gae_trace_cap_lag1, 0.0, p.is_ratio_cap_lag1, lo_open=True)
    _choice("policy.rollout_lag", p.rollout_lag, (0, 1, 2))      # 2: a second round may run ahead
    _choice("policy.backbone_dtype", p.backbone_dtype, BACKBONE_DTYPES)
    _in("policy.micro_rows", p.micro_rows, 1, 1 << 20)
    _in("policy.max_micro_tokens", p.max_micro_tokens, 0, 1 << 32)
    lc = c.learner
    if lc.compile_mode not in ("", "default", "max-autotune-no-cudagraphs"):
        raise _err("learner.compile_mode", "must be empty, default, or max-autotune-no-cudagraphs")
    if not isinstance(lc.compile_dynamic, bool):
        raise _err("learner.compile_dynamic", "must be true or false")
    # --- KL
    k = c.kl
    _in("kl.beta_min", k.beta_min, 0.0, 1.0, lo_open=True)
    _in("kl.beta_max", k.beta_max, k.beta_min, 1.0)
    _in("kl.beta", k.beta, k.beta_min, k.beta_max)
    _choice("kl.normalisation", k.normalisation, ("surrogate_denominator",))
    # --- early stops
    e = c.early_stop
    for name in ("joint_k3", "play_k3", "gate_kl"):
        _in(f"early_stop.{name}", getattr(e, name), 0.0, 10.0, lo_open=True)
    _in("early_stop.lr_halve_if_stopped_frac", e.lr_halve_if_stopped_frac, 0.0, 1.0)
    _in("early_stop.lr_halve_window", e.lr_halve_window, 1, 1000)
    # --- entropy
    _keys("entropy.coef", c.entropy.coef, ("gate", "slot", "cells", "ability"))
    for name, v in c.entropy.coef.items():
        _in(f"entropy.coef.{name}", v, 0.0, 0.1)
    _fixed("entropy.coef.gate", c.entropy.coef["gate"], 0.0, "no gate entropy bonus")
    _in("entropy.cell_bonus", c.entropy.cell_bonus, 0.0, 0.1)
    _in("entropy.cell_trigger_frac_of_il", c.entropy.cell_trigger_frac_of_il, 0.0, 1.0)
    # --- GAE
    _in("gae.gamma", c.gae.gamma, 0.0, 1.0, lo_open=True)
    _in("gae.lambda_tick", c.gae.lambda_tick, 0.0, 1.0, lo_open=True)
    _in("gae.lambda_tick_f1", c.gae.lambda_tick_f1, 0.0, 1.0, lo_open=True)
    # --- critic
    cr = c.critic
    _choice("critic.loss", cr.loss, ("mse",))
    _in("critic.lr", cr.lr, 0.0, 1e-2, lo_open=True)
    for i, b in enumerate(cr.betas):
        _in(f"critic.betas[{i}]", b, 0.0, 1.0, hi_open=True)
    _in("critic.adam_eps", cr.adam_eps, 0.0, 1e-2, lo_open=True)
    _in("critic.grad_clip", cr.grad_clip, 0.0, 100.0, lo_open=True)
    _in("critic.epochs", cr.epochs, 1, 64)
    w = cr.warmup
    _in("critic.warmup.games_per_batch", w.games_per_batch, 1, 1 << 16)
    _in("critic.warmup.max_updates", w.max_updates, 1, 1000)
    _in("critic.warmup.fixed_set_games", w.fixed_set_games, 1, 1 << 20)
    _in("critic.warmup.stop_trend_updates", w.stop_trend_updates, 1, w.max_updates)
    if not isinstance(w.overlap, bool):
        raise _err("critic.warmup.overlap", "must be true or false")
    if w.early_stop_min_updates is not None:
        if isinstance(w.early_stop_min_updates, bool) or not isinstance(w.early_stop_min_updates, int):
            raise _err("critic.warmup.early_stop_min_updates", "must be null or an integer")
        _in("critic.warmup.early_stop_min_updates", w.early_stop_min_updates, 1, w.max_updates)
    if w.plateau_gain is not None:
        if isinstance(w.plateau_gain, bool) or not isinstance(w.plateau_gain, (int, float)):
            raise _err("critic.warmup.plateau_gain", "must be null or a number")
        _in("critic.warmup.plateau_gain", w.plateau_gain, 0.0, 1.0, lo_open=True)
    if isinstance(w.plateau_window, bool) or not isinstance(w.plateau_window, int):
        raise _err("critic.warmup.plateau_window", "must be an integer")
    _in("critic.warmup.plateau_window", w.plateau_window, 1, w.max_updates)
    _in("critic.warmup.pass_ev_overall", w.pass_ev_overall, -1.0, 1.0)
    _in("critic.warmup.pass_ev_last_third", w.pass_ev_last_third, -1.0, 1.0)
    # --- reward
    rw = c.reward
    _fixed("reward.win", rw.win, 1.0, "terminal +1")
    _fixed("reward.loss", rw.loss, -1.0, "terminal -1")
    _fixed("reward.draw", rw.draw, 0.0, "draw 0")
    _choice("reward.shaping", rw.shaping, SHAPING_CHOICES)
    _in("reward.short_head_coef", rw.short_head_coef, 0.0, 1.0)
    _in("reward.short_head_gamma", rw.short_head_gamma, 0.0, 1.0, lo_open=True)
    _in("reward.short_head_decision_ticks", rw.short_head_decision_ticks, 1, 1000)
    # --- BC
    _in("bc.weight", c.bc.weight, 0.0, 1.0)
    _in("bc.start_weight", c.bc.start_weight, 0.0, 1.0)
    _in("bc.min_rows_per_update", c.bc.min_rows_per_update, 0, 1 << 24)
    if u.bc_enabled and c.bc.weight <= 0:
        raise _err("bc.weight", f"user.bc_enabled is true but the weight is 0 (start at bc.start_weight "
                                f"{c.bc.start_weight})")
    if not u.bc_enabled and c.bc.weight != 0:
        raise _err("bc.weight", "must be 0 while user.bc_enabled is false")
    # --- weights
    wt = c.weights
    _fixed("weights.seat_ours", wt.seat_ours, 1.0, "our seat weighs 1")
    _in("weights.seat_theta_opp", wt.seat_theta_opp, 0.0, 1.0)
    _in("weights.seat_mirror", wt.seat_mirror, 0.0, 1.0)
    _fixed("weights.ht_action", wt.ht_action, 1.0, "action rows weigh 1")
    _fixed("weights.forced", wt.forced, 0.0, "forced rows weigh 0")
    _keys("weights.ht_wait", wt.ht_wait, tuple(G.WEIGHT))
    for s, v in wt.ht_wait.items():
        if abs(v - G.WEIGHT[s]) > 1e-12:
            raise _err(f"weights.ht_wait.{s}", f"must equal 1/KAPPA = {G.WEIGHT[s]!r} (gae.WEIGHT), got {v!r}")
    # --- opponents
    o = c.opponents
    _in("opponents.pilot_updates", o.pilot_updates, 0, 1000)
    _mix("opponents.pilot", o.pilot, OPPONENT_KINDS)
    _mix("opponents.main", o.main, OPPONENT_KINDS)
    _fixed("opponents.pilot.snapshot", o.pilot["snapshot"], 0.0, "no snapshots in the pilot")
    _in("opponents.il_floor", o.il_floor, 0.0, 1.0)
    if isinstance(o.general_share, bool) or not isinstance(o.general_share, (int, float)):
        raise _err("opponents.general_share", "must be a number")
    _in("opponents.general_share", o.general_share, 0.0, 1.0)
    _choice("opponents.general_opponents", o.general_opponents, GENERAL_OPPONENTS)
    if o.general_opponents != "self" and not o.general_share:
        raise _err("opponents.general_opponents", "needs opponents.general_share > 0 (there are no general games)")
    if u.opponent_design == "selfplay_il30":
        for ph in PHASES:
            il = sum(getattr(o, ph)[kk] for kk in IL_KINDS)
            if il < o.il_floor - 1e-9:
                raise _err(f"opponents.{ph}", f"IL share {il:.3f} below il_floor {o.il_floor}")
    else:
        for ph in PHASES:
            if sum(getattr(o, ph)[kk] for kk in IL_KINDS) <= 0:
                raise _err(f"opponents.{ph}", "il_only needs a positive IL share to renormalise")
    _in("opponents.snapshot_every", o.snapshot_every, 1, 1000)
    _in("opponents.snapshot_pool_recent", o.snapshot_pool_recent, 1, 64)
    if not isinstance(o.snapshot_anchors, (list, tuple)):
        raise _err("opponents.snapshot_anchors", "must be a list of NAME|PATH|SHA256")
    seen_a: Dict[str, str] = {}
    for item in o.snapshot_anchors:
        try:
            an = parse_anchor(item)
        except ValueError as e:
            raise _err("opponents.snapshot_anchors", str(e)) from e
        if an["name"] in seen_a or an["sha256"] in seen_a.values():
            raise _err("opponents.snapshot_anchors", f"{an['name']}: a name or a file named twice")
        seen_a[an["name"]] = an["sha256"]
    if len(o.snapshot_anchors) > 16:
        raise _err("opponents.snapshot_anchors", "at most 16 anchors")
    # --- matchup games (the second matchup "matchup2": the same rules for each)
    seen_m: Dict[str, str] = {}                      # over both matchups: one name, one file
    for mp in ("matchup", "matchup2"):
        for key in (f"{mp}_share", f"{mp}_il_share"):
            v = getattr(o, key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise _err(f"opponents.{key}", "must be a number")
            _in(f"opponents.{key}", v, 0.0, 1.0)
        for key in (f"{mp}_deck", f"{mp}_models", f"{mp}_weights"):
            if not isinstance(getattr(o, key), (list, tuple)):
                raise _err(f"opponents.{key}", "must be a list")
        raw = getattr(o, f"{mp}_deck")
        deck, mods, ws = list(raw), list(getattr(o, f"{mp}_models")), list(getattr(o, f"{mp}_weights"))
        share, il = float(getattr(o, f"{mp}_share")), float(getattr(o, f"{mp}_il_share"))
        if deck:
            if len(deck) != 8 or len(set(deck)) != 8 or any(not isinstance(x, str) or not x for x in deck):
                raise _err(f"opponents.{mp}_deck", f"must be 8 distinct card keys, got {raw!r}")
            try:
                _check_deck_slots(deck)
            except ConfigError as e:
                raise _err(f"opponents.{mp}_deck", str(e).split(": ", 1)[-1]) from e
            if u.deck is not None and sorted(deck) == sorted(u.deck):
                raise _err(f"opponents.{mp}_deck",
                           "is user.deck itself (a mirror game is the 'mirror' opponent kind)")
        here: Dict[str, str] = {}
        for item in mods:
            try:
                an = parse_anchor(item)
            except ValueError as e:
                raise _err(f"opponents.{mp}_models", str(e)) from e
            if an["name"] in here or an["sha256"] in here.values():
                raise _err(f"opponents.{mp}_models", f"{an['name']}: a name or a file named twice")
            # the same model may also be a league anchor, but then under the same name (one row of the opponent table)
            if seen_a.get(an["name"], an["sha256"]) != an["sha256"] or \
                    any(s == an["sha256"] and n != an["name"] for n, s in seen_a.items()):
                raise _err(f"opponents.{mp}_models", f"{an['name']}: a league anchor has this name or this file "
                                                     f"under another name / file")
            # and in both matchups, then under the same name too
            if seen_m.get(an["name"], an["sha256"]) != an["sha256"] or \
                    any(s == an["sha256"] and n != an["name"] for n, s in seen_m.items()):
                raise _err(f"opponents.{mp}_models", f"{an['name']}: the other matchup has this name or this file "
                                                     f"under another name / file")
            here[an["name"]] = an["sha256"]
            seen_m[an["name"]] = an["sha256"]
        if len(mods) > 8:
            raise _err(f"opponents.{mp}_models", "at most 8 models")
        if ws:
            if len(ws) != len(mods):
                raise _err(f"opponents.{mp}_weights", f"{len(ws)} weights for {len(mods)} models (empty = uniform)")
            if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not w > 0 for w in ws):
                raise _err(f"opponents.{mp}_weights", "every weight must be a positive number")
        if share:
            if not deck:
                raise _err(f"opponents.{mp}_share", f"needs opponents.{mp}_deck (the deck on the other seat)")
            if not mods and il < 1.0:
                raise _err(f"opponents.{mp}_share", f"needs opponents.{mp}_models (or {mp}_il_share = 1)")
        if il and not share:
            raise _err(f"opponents.{mp}_il_share", f"needs opponents.{mp}_share > 0 (there are no matchup games)")
    if float(o.general_share) + float(o.matchup_share) > 1.0 + 1e-9:
        raise _err("opponents.matchup_share", f"general_share {o.general_share} + matchup_share {o.matchup_share} > 1")
    if float(o.general_share) + float(o.matchup_share) + float(o.matchup2_share) > 1.0 + 1e-9:
        raise _err("opponents.matchup2_share", f"general_share {o.general_share} + matchup_share {o.matchup_share} "
                                               f"+ matchup2_share {o.matchup2_share} > 1")
    if o.matchup2_deck:
        if not o.matchup_deck:
            raise _err("opponents.matchup2_deck", "needs opponents.matchup_deck (the first matchup comes first)")
        if sorted(o.matchup2_deck) == sorted(o.matchup_deck):
            raise _err("opponents.matchup2_deck", "is opponents.matchup_deck (one matchup per deck)")
    _keys("opponents.temperature", o.temperature, FROZEN_TEMPERATURE_KINDS)
    for kk, t in o.temperature.items():
        if t is not None:
            _temp(f"opponents.temperature.{kk}", t)
    _fixed("opponents.temperature.il_v4_t1", o.temperature["il_v4_t1"], 1.0, "the human-like IL opponent at T = 1")
    _fixed("opponents.temperature.il_v3_t1", o.temperature["il_v3_t1"], 1.0, "the extra IL slot at T = 1")
    _fixed("opponents.temperature.il_v2_heldout", o.temperature["il_v2_heldout"], 1.0,
           "the held-out gate opponent (no gate bias) plays at T = 1")
    _keys("opponents.checkpoint", o.checkpoint, CHECKPOINT_KINDS)
    for kk, ck in o.checkpoint.items():
        if ck is not None and not ck.strip():
            raise _err(f"opponents.checkpoint.{kk}", "must be a path or null")
    _keys("opponents.checkpoint_sha256", o.checkpoint_sha256, CHECKPOINT_KINDS)
    for kk, sha in o.checkpoint_sha256.items():
        if sha is not None and not _SHA256_RE.fullmatch(sha):
            raise _err(f"opponents.checkpoint_sha256.{kk}", f"must be 64 lowercase hex characters or null, got {sha!r}")
    _keys("opponents.gate_bias", o.gate_bias, GATE_BIAS_MODELS)
    for mdl, gb in o.gate_bias.items():
        _keys(f"opponents.gate_bias.{mdl}", gb, ("base", "threat"))
        for row, vals in gb.items():
            if len(vals) != N_FAMILIES:
                raise _err(f"opponents.gate_bias.{mdl}.{row}", f"needs {N_FAMILIES} values (wait, play, ability), "
                                                               f"got {vals!r}")
            for i, x in enumerate(vals):
                _in(f"opponents.gate_bias.{mdl}.{row}[{i}]", x, -10.0, 10.0)
    _keys("opponents.hazard_dt", o.hazard_dt, OLD_TREE_MODELS)
    for mdl, hz in o.hazard_dt.items():
        if hz is not None:
            _choice(f"opponents.hazard_dt.{mdl}", hz, HAZARD_DT_MODES)
    _keys("opponents.code_tree", o.code_tree, OLD_TREE_MODELS)
    # --- sim-to-real
    s2 = c.sim2real
    for kind in ("play", "ability"):
        lo, hi = getattr(s2.landing_delay, kind)
        _in(f"sim2real.landing_delay.{kind}[0]", lo, 0, 19)
        _in(f"sim2real.landing_delay.{kind}[1]", hi, lo, 19)   # lead D - tap = 20 must stay above g
    if r.stamp_rule.enabled and r.stamp_rule.blocked_mod10:  # the lead D - tap = 20 stays above the pushed g
        push = c.max_stamp_push()
        for kind in LANDING_KINDS:                               # (after the legacy ranges are checked)
            g_hi = max(c.landing_delay_pmf(kind))
            if g_hi + push > MAX_LANDING_DELAY:
                raise _err("rollout.stamp_rule", f"{kind} delay {g_hi} + stamp push {push} > {MAX_LANDING_DELAY} "
                                                 "(the lead D - tap = 20 must stay above it)")
    s_hi = c.own_sighting_delay_range()[1]                       # the strict own-sighting bound, stamp rule
    for kind in LANDING_KINDS:                                   # on or off, for every kind: a null pmf is its
        e_hi = c.max_stamped_delay(kind)                         # legacy range
        if e_hi + s_hi >= DISPATCH_LEAD:
            raise _err(f"rollout.landing_delay.{kind}_pmf",
                       f"stamped {kind} delay up to {e_hi} + own sighting {s_hi} ({c.own_sighting_model()}) >= "
                       f"{DISPATCH_LEAD}: the client would judge its command rejected before it sees it")
    _in("sim2real.delta_lag.default_ticks", s2.delta_lag.default_ticks, 0, 20)
    if s2.delta_lag.overrides:
        unknown = sorted(set(s2.delta_lag.overrides) - _object_names())
        if unknown:
            raise _err("sim2real.delta_lag.overrides", f"{unknown} are not object names of {OBJECT_VOCAB} 'ids'")
    for kk, v in s2.delta_lag.overrides.items():
        _in(f"sim2real.delta_lag.overrides.{kk}", v, 0, 20)
    # the display lag is 0 everywhere, never a parameter
    if s2.delta_lag.default_ticks != 0:
        raise _err("sim2real.delta_lag.default_ticks", f"must be 0, got {s2.delta_lag.default_ticks}: {DELTA_LAG_G12A}")
    for kk, v in sorted(s2.delta_lag.overrides.items()):
        if v != 0:
            raise _err(f"sim2real.delta_lag.overrides.{kk}", f"must be 0, got {v}: {DELTA_LAG_G12A}")
    if not s2.delta_arr_table.endswith(".json"):
        raise _err("sim2real.delta_arr_table", "must be a .json arrival table")
    # --- levels
    lv = c.levels
    _mix("levels.mix", lv.mix, LEVELS)
    _keys("levels.population", lv.population, LEVELS)
    if lv.population[16] is None and lv.mix[16] > 0:
        raise _err("levels.population.16", "must be set while level 16 has a share (e.g. ranked)")
    for kk, pop in lv.population.items():
        if pop is not None:
            _choice(f"levels.population.{kk}", pop, POPULATIONS if kk == 16 else POPULATION_L11_CHOICES)
    # --- gates
    g = c.gates
    _in("gates.g1.every_updates", g.g1.every_updates, 1, 1000)
    if g.flow == "crn":                                   # the h2h flow has no screen: its place is unused
        _in("gates.screen.at_update_in_cycle", g.screen.at_update_in_cycle, 1, g.g1.every_updates)
    _in("gates.screen.games", g.screen.games, 1, 1 << 20)
    _in("gates.g1.games_per_side", g.g1.games_per_side, 1, 1 << 20)
    _in("gates.g1.alpha_one_sided", g.g1.alpha_one_sided, 0.0, 0.5, lo_open=True)
    _in("gates.g1.heldout_min_delta_pp", g.g1.heldout_min_delta_pp, -100.0, 100.0)
    _in("gates.g1c.pairs", g.g1c.pairs, 1, 1 << 20)
    _in("gates.g2.pairs", g.g2.pairs, 1, 1 << 20)
    _fixed("gates.g2.record_only", g.g2.record_only, True, "the trend gate never blocks")
    _in("gates.release.pairs", g.release.pairs, 1, 1 << 20)
    _in("gates.release.mirror_games", g.release.mirror_games, 0, 1 << 20)
    _in("gates.release.transitivity_pairs", g.release.transitivity_pairs, 0, 1 << 20)
    _in("gates.release.live_games_min", g.release.live_games_min, 1, 1000)
    _mix("gates.crn_panel", g.crn_panel, ("il_v2_heldout", "il_v4_tstar", "il_v4_t1"))
    _in("gates.min_completion", g.min_completion, 0.5, 1.0)
    _in("gates.bootstrap_resamples", g.bootstrap_resamples, 100, 1 << 20)
    _in("gates.aa_null_pairs", g.aa_null_pairs, 1, 1 << 20)
    for name in ("il_temperature_sweep", "deploy_temperature_sweep"):
        ts = getattr(g, name)
        if not ts or len(set(ts)) != len(ts):
            raise _err(f"gates.{name}", "must be a non-empty list of distinct temperatures")
        for i, t in enumerate(ts):
            _temp(f"gates.{name}[{i}]", t)
    _in("gates.sweep_games", g.sweep_games, 1, 1 << 20)
    if len(set(g.canaries)) != len(g.canaries):
        raise _err("gates.canaries", f"must be distinct, got {g.canaries!r}")
    for i, cn in enumerate(g.canaries):
        _choice(f"gates.canaries[{i}]", cn, CANARY_KINDS)
    _in("gates.canary_games", g.canary_games, 1, 1 << 20)
    _in("gates.canary_alarm_lb", g.canary_alarm_lb, 0.0, 1.0, lo_open=True, hi_open=True)
    if g.panel_level is not None:
        _choice("gates.panel_level", g.panel_level, LEVELS)
        if c.levels.mix[g.panel_level] <= 0:
            raise _err("gates.panel_level", f"level {g.panel_level} has no share in levels.mix")
    _choice("gates.panel_source", g.panel_source, PANEL_SOURCES)
    if g.g2_anchor:
        try:
            parse_anchor(g.g2_anchor)
        except ValueError as e:
            raise _err("gates.g2_anchor", str(e)) from e
    if isinstance(g.g2_every_g1, bool) or not isinstance(g.g2_every_g1, int):
        raise _err("gates.g2_every_g1", "must be an integer")
    _in("gates.g2_every_g1", g.g2_every_g1, 0, 1000)
    for mp in ("matchup", "matchup2"):                # the first and the second matchup
        games, anchor = getattr(g, f"{mp}_games"), getattr(g, f"{mp}_anchor")
        if isinstance(games, bool) or not isinstance(games, int):
            raise _err(f"gates.{mp}_games", "must be an integer")
        _in(f"gates.{mp}_games", games, 0, 4096)
        if anchor:
            try:
                parse_anchor(anchor)
            except ValueError as e:
                raise _err(f"gates.{mp}_anchor", str(e)) from e
        if games:
            if not anchor:
                raise _err(f"gates.{mp}_games", f"needs gates.{mp}_anchor (the fixed model on the matchup deck)")
            if not getattr(c.opponents, f"{mp}_deck"):
                raise _err(f"gates.{mp}_games", f"needs opponents.{mp}_deck (the deck of the matchup)")
    if g.panel_source == "general" and not c.opponents.general_share:
        raise _err("gates.panel_source", "'general' needs opponents.general_share > 0 (general games)")
    _choice("gates.g2_mode", g.g2_mode, G2_MODES)
    _in("gates.sprt_h1_pp", g.sprt_h1_pp, 0.0, 100.0, lo_open=True)
    _choice("gates.flow", g.flow, GATE_FLOWS)
    if g.h2h_stop_alpha is not None:
        if isinstance(g.h2h_stop_alpha, bool):
            raise _err("gates.h2h_stop_alpha", "must be null or a number")
        _in("gates.h2h_stop_alpha", g.h2h_stop_alpha, 0.0, 0.05, lo_open=True)
    _in("gates.h2h_h1_pp", g.h2h_h1_pp, 0.0, 50.0, lo_open=True)
    _in("gates.h2h_guard_pairs", g.h2h_guard_pairs, 0, 1 << 20)
    _in("gates.h2h_guard_alpha", g.h2h_guard_alpha, 0.0, 0.5, lo_open=True)
    if g.flow == "crn" and (g.h2h_stop_alpha is not None or g.h2h_h1_pp != 3.0 or g.h2h_guard_pairs != 200
                            or g.h2h_guard_alpha != 0.1):
        raise _err("gates.flow", "the gates.h2h_* keys need gates.flow=h2h")
    if g.sprt_stop_alpha is not None:
        if isinstance(g.sprt_stop_alpha, bool):
            raise _err("gates.sprt_stop_alpha", "must be null or a number")
        _in("gates.sprt_stop_alpha", g.sprt_stop_alpha, 0.0, 0.05, lo_open=True)
    for name, hi in (("games_per_worker", 8), ("workers", 32)):
        v = getattr(g, name)
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, int):
                raise _err(f"gates.{name}", "must be null or an integer")
            _in(f"gates.{name}", v, 1, hi)
    _in("gates.pilot_min_delta_pp", g.pilot_min_delta_pp, -100.0, 0.0)
    _in("gates.p6_entry_g2_min_score", g.p6_entry_g2_min_score, 0.5, 1.0)
    _in("gates.restart_swap_min_score", g.restart_swap_min_score, 0.5, 1.0)
    _in("gates.rollback_delta_pp", g.rollback_delta_pp, -100.0, 0.0)
    _in("gates.rollback_consecutive", g.rollback_consecutive, 1, 100)
    _in("gates.rollback_lr_mult", g.rollback_lr_mult, 0.0, 1.0, lo_open=True)
    _in("gates.rollback_beta_double_window", g.rollback_beta_double_window, 1, 1000)
    _in("gates.rollback_beta_mult", g.rollback_beta_mult, 1.0, 10.0)
    _in("gates.stall_g1_without_promotion", g.stall_g1_without_promotion, 1, 100)
    _in("gates.stall_step_min_updates", g.stall_step_min_updates, 1, 1000)
    _in("gates.stall_games_per_update", g.stall_games_per_update, c.rollout.games_per_update, 1 << 16)
    _in("gates.pilot_joint_kl_median_max", g.pilot_joint_kl_median_max, 0.0, 1.0, lo_open=True)
    _in("gates.stall_stop_g1", g.stall_stop_g1, g.stall_g1_without_promotion, 1000)
    _in("gates.release_after_g1_without_new", g.release_after_g1_without_new, 1, 100)
    # --- monitor
    m = c.monitor
    pm = m.parity
    for name in ("max_abs_dlogp_fp32", "ratio_tol", "kl_theta0_ref_max"):
        _in(f"monitor.parity.{name}", getattr(pm, name), 0.0, 1.0, lo_open=True)
    _keys("monitor.parity.dlogp_tol_bf16", pm.dlogp_tol_bf16, ROW_TYPES)
    for rt, tol in pm.dlogp_tol_bf16.items():
        if tol is not None:
            _in(f"monitor.parity.dlogp_tol_bf16.{rt}", tol, 0.0, 1.0, lo_open=True)
    _keys("monitor.parity.dlogp_tol_fp16", pm.dlogp_tol_fp16, ROW_TYPES)
    for rt, tol in pm.dlogp_tol_fp16.items():
        if tol is not None:
            _in(f"monitor.parity.dlogp_tol_fp16.{rt}", tol, 0.0, 1.0, lo_open=True)
    _in("monitor.parity.replay_games", pm.replay_games, 0, 10000)
    _in("monitor.parity.games", pm.games, 1, 10000)
    _in("monitor.clip_frac_max", m.clip_frac_max, 0.0, 1.0, lo_open=True)
    for name in ("clip_frac_consecutive", "kl_stall_window", "kl_stall_g1_without_promotion",
                 "core_card_consecutive", "core_rollbacks_before_stop", "mask_recheck_every",
                 "value_ev_consecutive", "drift_every", "swap_slot_reduction", "nan_skip_consecutive_stop"):
        _in(f"monitor.{name}", getattr(m, name), 1, 1000)
    _in("monitor.kl_stall_band", m.kl_stall_band, 0.0, 1.0, lo_open=True)
    _in("monitor.temp_lr_mult", m.temp_lr_mult, 0.0, 1.0, lo_open=True)
    _in("monitor.temp_lr_floor", m.temp_lr_floor, 0.0, 1.0, lo_open=True)
    _in("monitor.kl_stall_beta_mult", m.kl_stall_beta_mult, 0.0, 1.0, lo_open=True)
    _in("monitor.lr_stall_mult", m.lr_stall_mult, 1.0, 10.0, lo_open=True)
    _in("monitor.kl_step_skip_growth", m.kl_step_skip_growth, 1.0, 100.0, lo_open=True)
    _in("monitor.kl_step_skip_g1_move_pp", m.kl_step_skip_g1_move_pp, 0.0, 100.0)
    _in("monitor.kl_grad_ratio_min", m.kl_grad_ratio_min, 0.0, 1.0)
    _in("monitor.tiny_log_ratio", m.tiny_log_ratio, -1000.0, 0.0, hi_open=True)
    _in("monitor.gate_trend_window", m.gate_trend_window, 2, 1000)
    _in("monitor.probe_rows", m.probe_rows, 0, 1 << 20)
    for name, hi in (("tiny_ratio_share_max", 1.0), ("gate_trend_act_rel_max", 100.0),
                     ("gate_trend_adv_gap_max", 100.0), ("flip_rate_max", 1.0)):
        if getattr(m, name) is not None:
            _in(f"monitor.{name}", getattr(m, name), 0.0, hi, lo_open=True)
    if m.flip_rate_max is not None and m.probe_rows == 0:   # no probe, no flip metric
        raise ConfigError("monitor.flip_rate_max needs the argmax probe (monitor.probe_rows > 0): without it no "
                          "probe_flip/* metric exists and the alarm could never fire")
    if m.entropy_max_frac_of_ref is not None:
        _in("monitor.entropy_max_frac_of_ref", m.entropy_max_frac_of_ref, 1.0, 100.0, lo_open=True)
    for name in ("plays_per_min_band", "card_rate_band"):
        lo, hi = getattr(m, name)
        _in(f"monitor.{name}[0]", lo, 0.0, 1.0, lo_open=True)
        _in(f"monitor.{name}[1]", hi, 1.0, 100.0)
    for name in ("entropy_min_frac_of_il", "core_card_min_frac_of_il", "pilot_core_card_min_frac",
                 "opp_nonwait_min_frac", "throughput_min_frac"):
        _in(f"monitor.{name}", getattr(m, name), 0.0, 1.0, lo_open=True)
    if not m.core_cards or any(not x for x in m.core_cards):
        raise _err("monitor.core_cards", "must list at least one card key")
    if u.deck is not None:
        extra = sorted(set(m.core_cards) - set(u.deck))
        if extra:
            raise _err("monitor.core_cards", f"{extra} not in user.deck (core cards of our deck)")
        absent = [x for x in CORE_CARD_CANDIDATES if x in u.deck and x not in m.core_cards]
        if absent:
            raise _err("monitor.core_cards", f"{absent} in user.deck must be monitored "
                                             "(CORE_CARD_CANDIDATES)")
    _in("monitor.illegal_cell_code", m.illegal_cell_code, 0, 1 << 16)
    _in("monitor.legal_refused_max_ratio_vs_il", m.legal_refused_max_ratio_vs_il, 1.0, 100.0)
    _in("monitor.forced_frac_alarm_abs", m.forced_frac_alarm_abs, 0.0, 1.0, lo_open=True)
    _in("monitor.drop_rate_warn", m.drop_rate_warn, 0.0, 1.0)
    _in("monitor.drop_rate_stop", m.drop_rate_stop, m.drop_rate_warn, 1.0)
    _in("monitor.value_ev_overall_min", m.value_ev_overall_min, -1.0, 1.0)
    _in("monitor.value_ev_last_third_min", m.value_ev_last_third_min, -1.0, 1.0)
    _in("monitor.value_ev_drop_from_warmup", m.value_ev_drop_from_warmup, 0.0, 2.0, lo_open=True)
    _in("monitor.min_free_mem_gb", m.min_free_mem_gb, 0.0, 64.0)
    _in("monitor.swap_growth_gb", m.swap_growth_gb, 0.0, 64.0, lo_open=True)
    _in("monitor.swap_window_s", m.swap_window_s, 1.0, 86400.0)
    _in("monitor.swap_cooldown_s", m.swap_cooldown_s, 0.0, 86400.0)
    if not isinstance(m.host_memory_events, bool):
        raise _err("monitor.host_memory_events", "must be true or false")
    for key in ("code19_stop_per_update", "code19_stop_consecutive"):
        v = getattr(m, key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise _err(f"monitor.{key}", "must be an integer")
        _in(f"monitor.{key}", v, 1, 1000000)
    if m.swap_mem_guard_gb is not None:
        _in("monitor.swap_mem_guard_gb", m.swap_mem_guard_gb, 0.0, 1024.0, lo_open=True)
    if m.opp_score_alarm is not None:
        if isinstance(m.opp_score_alarm, bool):
            raise _err("monitor.opp_score_alarm", "must be null or a number")
        _in("monitor.opp_score_alarm", m.opp_score_alarm, 0.0, 1.0, lo_open=True, hi_open=True)
    _choice("monitor.opp_score_kind", m.opp_score_kind, OPPONENT_KINDS)
    for key in ("opp_score_window", "opp_score_min_games"):
        v = getattr(m, key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise _err(f"monitor.{key}", "must be an integer")
        _in(f"monitor.{key}", v, 1, 1000000)
    if m.opp_score_alarm is None and (m.opp_score_kind, m.opp_score_window, m.opp_score_min_games) != (
            "il_v4_tstar", 10, 100):
        raise _err("monitor.opp_score_alarm", "the monitor.opp_score_* keys need monitor.opp_score_alarm")
    # --- drift
    dr = c.drift
    _in("drift.subset_games", dr.subset_games, 1, 1 << 20)
    _keys("drift.nll_increase_max", dr.nll_increase_max, DRIFT_HEADS)
    for kk, v in dr.nll_increase_max.items():
        _in(f"drift.nll_increase_max.{kk}", v, 0.0, 10.0, lo_open=True)
    _in("drift.gate_auc_drop_max", dr.gate_auc_drop_max, 0.0, 1.0, lo_open=True)
    lo, hi = dr.calibration_ratio_band
    _in("drift.calibration_ratio_band[0]", lo, 0.0, 1.0, lo_open=True)
    _in("drift.calibration_ratio_band[1]", hi, 1.0, 100.0)
    # --- cases
    ex = c.cases.exclude_population
    if len(set(ex)) != len(ex):
        raise _err("cases.exclude_population", f"must be distinct, got {ex!r}")
    for i, pop in enumerate(ex):
        _choice(f"cases.exclude_population[{i}]", pop, POPULATIONS)
    # --- chain
    _in("chain.block_timeout_s", c.chain.block_timeout_s, 600, 86400)
    _in("chain.kill_after_s", c.chain.kill_after_s, 60, 7200)
    # --- storage
    _choice("storage.rows", c.storage.rows, ("committed_tokens",))
    _in("storage.keep_updates", c.storage.keep_updates, 1, 100)
