#!/usr/bin/env python3
"""PPO gate runner: schedules, runs, stores and scores the evaluation gates that decide promotion.

The statistics are ``amini.ppo.stats`` (crn_gate / screen_s / seat_swap_gate / wilson / sprt_g1); this module only
builds the games, runs them through a pluggable runner and turns the stored outcomes into stats records. It uses
``ppo_v7.cases`` (case references, opponent settings rows and the stable seed hash), which is not part of this
repository; without it the module imports, and any call that needs it raises a RuntimeError.

* **CRN gates**: ``screen``, ``g1`` and ``aa`` (an A/A null test). A *unit* is one triple (val-panel opening, seed
  block, frozen panel opponent); each unit is played twice, arm ``candidate`` and arm ``champion``, both on our
  seat (the case's ``our_owner``) with the SAME environment seed, opponent seeds and (``policy_crn``) our policy
  seeds. The panel opponent of each triple is drawn with exact counts from the panel mix (``gates.crn_panel``);
  the ``holdout_kinds`` triples form the G1 held-out sub-panel. The A/A test uses independent policy seeds per arm.
* **Seat-swap gates**: ``g1c`` (pairs vs the champion, fresh seeds), ``g2`` (pairs vs the IL reference, fixed
  openings and seeds, record only), ``release`` and ``swap`` (any other head-to-head, record only). A unit is one
  pair on one opening: arm ``a0`` puts model A on owner 0, arm ``a1`` on owner 1; both games share the environment
  seed and the per-OWNER seeds (the models swap, the seats keep their random numbers). The opening is the stats
  ``cluster_id``.
* **Seed namespace**: every gate seed is ``base ^ sha256('ppo-v7-gate', block, unit, role)``, while training uses
  ``cases.replica_seed`` = ``base ^ sha256('ppo-v7', update, replica, role)``; the namespaces never meet.
  ``block_name`` gives each G1 / G1c (update, attempt) its own block (the 'crn_only' confirmation G1 its own
  ``-confirm`` block), the screen one block per (champion, attempt) (its stored champion games are reused) and
  G2 one fixed block per attempt. ``SeedLedger`` refuses a fresh-seed block (g1, g1c, release) that was already
  claimed by another candidate, and any block whose schedule changed. ``check_disjoint`` compares a gate plan
  with the run's training plans (shared 63-bit seeds or shared (case, rndSeed)) and ``check_panel`` its openings
  with the training openings.
* **Runner interface**: ``GameRunner.run(specs) -> outcomes``. A ``GameSpec`` carries everything one game
  needs (case, level, the two players by owner with their OpponentSpec settings and weight digest, the seeds
  by owner and role, the environment rndSeed); a ``GameOutcome`` returns how it ended, owner 0's score, length,
  overtime and per-owner diagnostics. The real runner plays the episode loop with frozen models in the game
  environment; the tests use a fake.
* **Resumable chunks**: the schedule of each arm is cut into fixed chunks; a chunk is written atomically (temp,
  fsync, os.replace, dir fsync) once all its games returned, so a SIGTERM / crash loses at most the running
  chunk. ``run_gate`` skips stored chunks, checks a ``stop`` predicate (``StopFlag``: SIGTERM, STOP marker files)
  and a deadline between chunks, and interleaves the arms chunk by chunk so an interrupted gate holds complete
  pairs. Every chunk header pins the block digest, the arm digest (players) and the game keys; a stored chunk
  that does not match the plan is an error, never silently reused.
* **Environment pin**: every plan carries a required ``env_digest`` (see ``env_digest()``: the rollout /
  sim2real / levels config sections, learner.compat_digest, the environment build sha, runner commit) inside its
  block digest, so stored games from another environment are never reused (the screen's champion games, a gate
  resumed in a later block); a changed environment takes a new attempt block.
* **Completeness**: a unit with a missing or failed game is dropped whole; completion is over the SCHEDULED
  units (a lost chunk counts as missing), below ``min_completion`` the gate is invalid (``replay_required``); a
  replay is a new attempt block. Too few openings is refused when planning.
* **Failure guard**: a dropped unit whose failure correlates with the scored model's losses biases the estimate.
  ``GameOutcome.fault_owner`` names the seat that faulted; the report flags (and for g1 / g1c / release blocks) a
  scored-model failure rate above ``failure_rate_max`` or failure counts that favour the scored model beyond a
  binomial bound. ``GateSettings.fault_policy='loss'`` scores a known fault as a loss of the faulting seat
  instead of dropping it.
* **Diagnostics**: per arm, the opponent seat's non-wait actions per game (the "opponent turns passive"
  check), plays / abilities per minute of both seats, per-card play rates, refusals by environment code,
  length and overtime fraction, failures by reason and faulting side, Wilson intervals of the scored side's
  score, the passive-opponent games (<= ``PASSIVE_NONWAIT_MAX`` non-wait commands) and their score, and per
  panel-opponent sub-panel deltas. ``GateReport.pilot_args`` / ``g2_p6_args`` feed the phase rules'
  ``pilot_pass`` / ``p6_entry``; ``monitor_args`` feeds Monitor.observe_gate.
* **Canaries** (config ``gates.canaries`` / ``canary_games``; diagnostic only): ``canary_plan`` puts an
  'all_wait' or 'random_legal' policy (``canary_player``, played in-process by the runner) on our seat of the
  first units of a screen / G1 / G1c / G2, against the same other seat and with the same seeds;
  ``evaluate_canary`` reports its score next to the gate and flags a canary that wins too often. A canary block
  is never a gate: ``evaluate`` refuses it, it never promotes and never reaches the monitor.
* **Win-condition breakdown** (diagnostic only): ``evaluate(..., win_conditions=)`` adds delta_hat by the
  opponent deck's win condition (``WIN_CONDITIONS``) with the null caveat of the worst of k groups.
* **Promotion**: ``promotion_decision`` needs G1 and G1c ('crn_then_seatswap') or G1 and a second G1 on its own
  ``-confirm`` block ('crn_only': two CRN gates); ``plan_gate`` builds any gate from the configured sizes, panel
  mix and held-out kinds, and ``evaluate`` refuses to pass a blocking gate that is not as configured
  (``check_schedule``).

Pure Python / numpy; no model, no game environment, no GPU. CLI (``status`` / ``evaluate``) works from the
stored ``plan.json`` and chunk files:

    python3 -B tools/ppo_v7/gate.py status --root GATE_ROOT --block g1-u000020-a0
    python3 -B tools/ppo_v7/gate.py evaluate --root GATE_ROOT --block g1-u000020-a0 [--n-boot 10000]
        [--no-schedule-check]
    python3 -B tools/ppo_v7/gate.py aa --root RUN/aa --block aa-x [--inputs RUN/inputs.json] [--write]

* **Gate-style A/A**: ``evaluate_aa`` scores a ``crn_plan('aa', ...)`` block (the same weights and settings on
  both arms, the same triples, only our policy random numbers independent): delta_hat with its two-sided 95 %
  cluster-bootstrap interval (pass: it covers 0), rho and the G1 sizing it implies (``aa_sizing``), the exact
  sign-flip null (``aa_null``) as the bootstrap calibration reference and the empirical null table of the
  per-win-condition breakdown. Record only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple,
                    Union)

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / 'src', ROOT / 'tools'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                                                    # noqa: E402

from amini.ppo import stats as S                               # noqa: E402


class _MissingCases:
    """Stands in for ``ppo_v7.cases`` (case references, opponent settings rows, the stable seed hash), which is
    not part of this repository: the module imports, and the first use of the case module raises."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__'):
            raise AttributeError(name)
        raise RuntimeError(f'ppo_v7.cases.{name}: the case module (ppo_v7.cases) is not part of this repository')


try:
    from ppo_v7 import cases as C                                     # noqa: E402
except ImportError as _exc:
    if getattr(_exc, 'name', None) not in (None, 'ppo_v7', 'ppo_v7.cases'):
        raise                                                         # the module exists; one of ITS imports failed
    C = _MissingCases()

# ------------------------------------------------------------------ constants
PLAN_SCHEMA = 'ppo-v7-gate-plan.2'      # .2: env_digest in the block digest
CHUNK_SCHEMA = 'ppo-v7-gate-chunk.1'
REPORT_SCHEMA = 'ppo-v7-gate-report.1'
LEDGER_SCHEMA = 'ppo-v7-gate-ledger.1'
GATE_NAMESPACE = 'ppo-v7-gate'           # cases.replica_seed hashes 'ppo-v7': disjoint namespaces
CRN_KINDS = ('screen', 'g1', 'aa')
SWAP_KINDS = ('g1c', 'g2', 'release', 'swap')
CANARY_KIND = 'canary'                   # a diagnostic single-arm plan next to a gate, never promoting
GATE_KINDS = CRN_KINDS + SWAP_KINDS + (CANARY_KIND,)
CANARY_ARMS = ('canary',)
CANARY_POLICIES = ('all_wait', 'random_legal')   # == amini.ppo.config.CANARY_KINDS (gates.canaries)
CANARY_CHECKPOINT_PREFIX = 'canary:'     # the canary Player's pseudo checkpoint: no file, no server
CANARY_MODEL_PREFIX = 'canary-'          # its model name in the runner's routing (in-process policy)
CANARY_ALARM_LB = 0.3                    # a canary whose Wilson lower bound beats this flags the gate (diagnostic;
#                                          example value): the CLI / test default; a caller passes the config's
#                                          gates.canary_alarm_lb
G2_MODES = ('pooled_since_restart', 'latest')   # == amini.ppo.config.G2_MODES (gates.g2_mode)
CANARY_PARENT_KINDS = ('screen', 'g1', 'g1c', 'g2')   # gates that carry canaries (A/A, release, swap do not)
WIN_CONDITION_UNKNOWN = 'unknown'        # an opening without a real opponent deck (synthetic) or no table entry
WIN_CONDITION_OTHER = 'other'
# the opponent deck's win condition, first match in this order (heavy beatdown before cycle / bait), card keys
# without the -ev1 / -hero suffix
WIN_CONDITIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ('golem', ('golem', 'elixir-golem')),
    ('lava_hound', ('lava-hound',)),
    ('giant', ('giant', 'goblin-giant', 'electro-giant', 'rune-giant')),
    ('royal_giant', ('royal-giant',)),
    ('balloon', ('balloon',)),
    ('hog', ('hog-rider', 'royal-hogs', 'ram-rider', 'battle-ram')),
    ('siege', ('x-bow', 'mortar')),
    ('graveyard_miner', ('graveyard', 'miner', 'goblin-drill')),
    ('bait', ('goblin-barrel', 'skeleton-barrel', 'wall-breakers')),
    ('bridge_spam', ('pekka', 'mega-knight', 'three-musketeers', 'sparky', 'goblinstein', 'goblin-demolisher',
                     'goblin-machine', 'boss-bandit', 'suspicious-bush')),
)
FRESH_KINDS = ('g1', 'g1c', 'release')   # a new seed block per gate (and per candidate)
BLOCKING_KINDS = ('g1', 'g1c', 'release')  # G2, screen, A/A and swap only record / alarm
CRN_ARMS = ('candidate', 'champion')
SWAP_ARMS = ('a0', 'a1')                 # model A on owner 0 / owner 1
PANEL_KINDS = ('il_v2_heldout', 'il_v4_tstar', 'il_v4_t1')   # config gates.crn_panel keys
DEFAULT_PANEL_MIX = {'il_v2_heldout': 0.5, 'il_v4_tstar': 0.3, 'il_v4_t1': 0.2}   # example shares
HOLDOUT_KINDS = ('il_v2_heldout',)
END_NATIVE = 'native'                    # == gae.END_NATIVE (a normal game finish, overtime included)
FAIL_ENDS = ('crash', 'step_guard', 'seat_stall', 'timeout', 'error', 'invalid')
ENDS = (END_NATIVE,) + FAIL_ENDS
TICKS_PER_MIN = 1200                     # 20 game ticks per second
REGULATION_TICKS = 3 * TICKS_PER_MIN     # fallback overtime rule when the runner does not say
SIDE_ROLES = ('adapter', 'torch', 'clock')                  # per seat: seat adapter rng, torch sampler, decision clock
SHARED_ROLES = ('game', 'landing_delay', 'display_lag', 'arrival')   # per unit (both seats / both arms)
POLICY_ROLES = ('adapter', 'torch', 'clock')                # the "policy random numbers" of a seat
DEFAULT_CHUNK = 64
ENV_SCHEMA = 'ppo-v7-gate-env.1'
ENV_SECTIONS = ('rollout', 'sim2real', 'levels')            # PPOConfig sections that change how a game plays
PASSIVE_NONWAIT_MAX = 3                  # an opponent with <= this many non-wait actions in a game is passive
FAULT_POLICIES = ('drop', 'loss')        # 'drop' drops a failed unit; 'loss' = a known fault loses for that seat


class GateError(ValueError):
    """A plan, store or runner inconsistency (never silently repaired)."""


class BlockReuseError(GateError):
    """A fresh-seed block was claimed again by another candidate, or a block's schedule changed."""


# ------------------------------------------------------------------ seeds
def _check_block(block: str) -> str:
    if not isinstance(block, str) or not block or '|' in block or '/' in block or block.startswith('.'):
        raise GateError(f'bad block name {block!r}')
    return block


def gate_seed(base_seed: int, block: str, unit: int, role: str) -> int:
    """Seed of one (block, unit, role) in the gate namespace: ``base ^ sha256('ppo-v7-gate', block, unit,
    role)``, 63 bits, non-negative. Disjoint from ``cases.replica_seed`` (namespace 'ppo-v7')."""
    if unit < 0:
        raise GateError('unit is non-negative')
    return (int(base_seed) & C.SEED_MASK) ^ C.stable_hash(GATE_NAMESPACE, _check_block(block), int(unit), str(role))


def gate_rnd_seed(base_seed: int, block: str, unit: int) -> int:
    """The environment ``rndSeed`` of one unit, in [1, 2**31 - 1] (same range as the training games' seeds)."""
    return 1 + gate_seed(base_seed, block, unit, 'rnd_seed') % C.RND_SEED_MAX


def block_name(kind: str, *, update: Optional[int] = None, attempt: int = 0,
               champion_digest: Optional[str] = None, tag: Optional[str] = None, confirm: bool = False) -> str:
    """The seed block of one gate.

    g1 / g1c: ``g1-u000020-a0`` (a fresh block per update and attempt; a replay is attempt + 1).
    g1 with ``confirm``: ``g1-u000020-a0-confirm``, the second CRN gate of the 'crn_only' promotion rule.
    screen: ``screen-<champion digest[:16]>-a<attempt>`` (one block per champion: its stored games are reused;
    a changed panel setting or a changed environment takes attempt + 1).
    g2: ``g2-fixed`` (fixed openings and seeds; attempt n >= 1 gives ``g2-fixed-a<n>`` after an environment change).
    release: ``release-a0``. aa: ``aa-a0``; aa with ``tag`` (a tagged gate-style A/A):
    ``aa-<tag>``, a replay ``aa-<tag>-a<attempt>`` (attempt >= 1). swap: ``swap-<tag>-a<attempt>``."""
    if kind not in GATE_KINDS:
        raise GateError(f'unknown gate kind {kind!r}')
    if kind == CANARY_KIND:
        raise GateError('a canary block is named after its gate (canary_plan)')
    if attempt < 0:
        raise GateError('attempt is non-negative')
    if confirm and kind != 'g1':
        raise GateError('only a G1 has a confirmation block')
    if kind in ('g1', 'g1c'):
        if update is None or update < 1:
            raise GateError(f'{kind} needs the update number (>= 1)')
        return f'{kind}-u{int(update):06d}-a{int(attempt)}' + ('-confirm' if confirm else '')
    if kind == 'screen':
        if not champion_digest or len(champion_digest) < 16:
            raise GateError('screen needs the champion digest (>= 16 chars)')
        return f'screen-{champion_digest[:16]}-a{int(attempt)}'
    if kind == 'g2':
        return 'g2-fixed' if attempt == 0 else f'g2-fixed-a{int(attempt)}'
    if kind == 'swap':
        if not tag:
            raise GateError('swap needs a tag')
        return _check_block(f'swap-{tag}-a{int(attempt)}')
    if kind == 'aa' and tag:                 # untagged 'aa' keeps 'aa-a<attempt>' (bit for bit as before)
        last = str(tag).rsplit('-', 1)[-1]
        if last[:1] == 'a' and last[1:].isdigit():
            raise GateError(f'aa tag {tag!r} ends like an attempt suffix (-a<n>): its blocks would collide')
        return _check_block(f'aa-{tag}' if attempt == 0 else f'aa-{tag}-a{int(attempt)}')
    return f'{kind}-a{int(attempt)}'


# ------------------------------------------------------------------ players, units, game specs
def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode('utf-8')).hexdigest()


def file_digest(path: Union[str, Path]) -> str:
    """sha256 of a checkpoint file's bytes. A recorded extra only: it is NOT the weight identity (a learner
    checkpoint also holds the critic, optimiser and monitor state, so a re-saved champion would get a new
    digest). ``Player.digest`` is ``learner.state_dict_digest(ck['model_ema'])`` (the shards' policy_digest)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def env_digest(cfg: Any = None, *, compat_digest: str = '', env_sha: str = '', runner_commit: str = '',
               extra: Optional[Mapping[str, Any]] = None) -> str:
    """The environment a gate is played in: sha256 over the ``ENV_SECTIONS`` of a
    ``PPOConfig`` (duck-typed ``to_dict()``), ``learner.compat_digest``, the game-environment build sha, the runner
    commit and any ``extra`` JSON. Goes into every plan's block digest; at least one part must be given."""
    sec = None
    if cfg is not None:
        full = cfg.to_dict()
        sec = {k: full[k] for k in ENV_SECTIONS if k in full}
    d = {'schema': ENV_SCHEMA, 'config': sec, 'compat_digest': str(compat_digest), 'env_sha': str(env_sha),
         'runner_commit': str(runner_commit), 'extra': dict(extra or {})}
    if sec is None and not (compat_digest or env_sha or runner_commit or extra):
        raise GateError('env_digest needs the config or at least one of compat / env / runner / extra')
    return _sha(d)


_SETTING_KEYS_IGNORED = ('name', 'checkpoint')


def settings_diff(x: 'Player', y: 'Player') -> List[str]:
    """Fields of the opponent settings row (temperature, gate_bias, hazard_dt, code_tree) where two players
    differ; name and checkpoint are the model, not its settings."""
    dx, dy = x.to_dict()['spec'], y.to_dict()['spec']
    return sorted(k for k in set(dx) | set(dy) if k not in _SETTING_KEYS_IGNORED and dx.get(k) != dy.get(k))


@dataclass(frozen=True)
class Player:
    """One model as it plays a gate game: name, weight identity and its opponent settings row.

    ``digest`` is the weight identity: ``learner.state_dict_digest(ck['model_ema'])`` for a PPO checkpoint (the
    same function as the shards' policy_digest), and the same function of the served state dict for an IL
    checkpoint. Not ``file_digest`` (see there)."""
    name: str
    digest: str
    spec: C.OpponentSpec

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.digest, str) or not self.digest:
            raise GateError(f'player needs a name and a digest: {self.name!r} {self.digest!r}')
        if not isinstance(self.spec, C.OpponentSpec):
            raise GateError(f'{self.name}: spec must be a cases.OpponentSpec')
        if self.spec.checkpoint is None:
            raise GateError(f'{self.name}: checkpoint not set')
        if self.spec.temperature is None:
            raise GateError(f'{self.name}: temperature not set (T*_IL must be set first)')

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self.spec)
        d['gate_bias'] = [list(r) for r in self.spec.gate_bias]
        return {'name': self.name, 'digest': self.digest, 'spec': d}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'Player':
        return cls(name=d['name'], digest=d['digest'], spec=C.OpponentSpec(**d['spec']))


@dataclass(frozen=True)
class Unit:
    """One CRN triple or one seat-swap pair. ``seeds`` are keyed by side label ('our'/'opp' for CRN,
    'o0'/'o1' for seat-swap) and role, plus the shared roles."""
    unit: int
    case_id: str
    level: int
    our_owner: int
    rnd_seed: int
    seeds: Tuple[Tuple[str, int], ...]
    opp_kind: Optional[str] = None       # CRN: the panel opponent kind; None for seat-swap
    holdout: bool = False

    def seed_map(self) -> Dict[str, int]:
        return dict(self.seeds)

    def to_dict(self) -> Dict[str, Any]:
        return {'unit': self.unit, 'case_id': self.case_id, 'level': self.level, 'our_owner': self.our_owner,
                'rnd_seed': self.rnd_seed, 'seeds': {k: v for k, v in self.seeds}, 'opp_kind': self.opp_kind,
                'holdout': self.holdout}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'Unit':
        return cls(unit=int(d['unit']), case_id=str(d['case_id']), level=int(d['level']),
                   our_owner=int(d['our_owner']), rnd_seed=int(d['rnd_seed']),
                   seeds=tuple(sorted((str(k), int(v)) for k, v in d['seeds'].items())),
                   opp_kind=d.get('opp_kind'), holdout=bool(d.get('holdout', False)))


@dataclass(frozen=True)
class GameSpec:
    """Everything one gate game needs. ``players[o]`` sits on owner ``o``; ``scored_owner`` is the seat whose
    result the gate scores (CRN: our seat; seat-swap: model A's seat). ``seeds`` are keyed
    ``o{owner}_{adapter|torch|clock}`` plus the shared roles; ``rnd_seed`` is the case's environment rndSeed.
    ``env_digest`` is the plan's environment pin: the runner refuses a spec whose digest is not its own."""
    key: str
    kind: str
    block: str
    unit: int
    arm: str
    case_id: str
    level: int
    our_owner: int
    players: Tuple[Player, Player]
    scored_owner: int
    seeds: Mapping[str, int]
    rnd_seed: int
    opp_kind: Optional[str] = None
    holdout: bool = False
    env_digest: str = ''


def game_key(block: str, arm: str, unit: int) -> str:
    return f'{block}/{arm}/{int(unit):06d}'


# ------------------------------------------------------------------ outcomes
@dataclass(frozen=True)
class SeatStats:
    """Per-owner diagnostics of one game. ``non_wait`` = the seat's non-wait commands, plays +
    abilities, accepted or not: the same definition as the rollout's 'opp_nonwait' game statistic, so gate
    diagnostics can serve as Monitor baselines. ``refusals`` by the environment's refusal code; ``cards`` = plays
    per card key."""
    queries: int = 0
    non_wait: int = 0
    plays: int = 0
    abilities: int = 0
    refusals: Mapping[str, int] = field(default_factory=dict)
    cards: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {'queries': int(self.queries), 'non_wait': int(self.non_wait), 'plays': int(self.plays),
                'abilities': int(self.abilities), 'refusals': {str(k): int(v) for k, v in sorted(self.refusals.items())},
                'cards': {str(k): int(v) for k, v in sorted(self.cards.items())}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'SeatStats':
        return cls(queries=int(d.get('queries', 0)), non_wait=int(d.get('non_wait', 0)), plays=int(d.get('plays', 0)),
                   abilities=int(d.get('abilities', 0)), refusals=dict(d.get('refusals', {})),
                   cards=dict(d.get('cards', {})))


@dataclass(frozen=True)
class GameOutcome:
    """How one game ended. ``score0`` = owner 0's score in {0, 0.5, 1} (crowns rule of the evaluation
    harness), set iff ``end == 'native'`` (a normal finish). ``overtime`` None = derived from length.
    ``fault_owner`` (failed games only): the owner whose seat caused the failure (seat_stall, step_guard by
    that seat, an invalid command), None if unknown or the environment (crash): feeds the failure guard."""
    key: str
    end: str
    score0: Optional[float] = None
    length: int = 0
    overtime: Optional[bool] = None
    seats: Tuple[SeatStats, SeatStats] = (SeatStats(), SeatStats())
    wall_s: float = 0.0
    error: Optional[str] = None
    fault_owner: Optional[int] = None

    def check(self) -> None:
        if self.end not in ENDS:
            raise GateError(f'{self.key}: end {self.end!r} not in {ENDS}')
        if self.end == END_NATIVE:
            if self.score0 is None or float(self.score0) not in S.VALID_RESULTS:
                raise GateError(f'{self.key}: a normal finish needs score0 in {S.VALID_RESULTS}, got {self.score0!r}')
        elif self.score0 is not None:
            raise GateError(f'{self.key}: a failed game ({self.end}) has no score')
        if isinstance(self.length, bool) or int(self.length) < 0:
            raise GateError(f'{self.key}: bad length {self.length!r}')
        if len(self.seats) != 2 or not all(isinstance(s, SeatStats) for s in self.seats):
            raise GateError(f'{self.key}: seats must be two SeatStats')
        if self.fault_owner is not None:
            if isinstance(self.fault_owner, bool) or self.fault_owner not in (0, 1):
                raise GateError(f'{self.key}: fault_owner must be None, 0 or 1, got {self.fault_owner!r}')
            if self.end == END_NATIVE:
                raise GateError(f'{self.key}: a normal finish has no fault_owner')

    @property
    def ok(self) -> bool:
        return self.end == END_NATIVE

    @property
    def is_overtime(self) -> bool:
        return bool(self.overtime) if self.overtime is not None else int(self.length) > REGULATION_TICKS

    def score(self, owner: int) -> Optional[float]:
        """Owner's score, None if the game failed."""
        if not self.ok:
            return None
        s = float(self.score0)
        return s if owner == 0 else 1.0 - s

    def to_dict(self) -> Dict[str, Any]:
        return {'key': self.key, 'end': self.end, 'score0': None if self.score0 is None else float(self.score0),
                'length': int(self.length), 'overtime': self.overtime,
                'seats': [s.to_dict() for s in self.seats], 'wall_s': float(self.wall_s), 'error': self.error,
                'fault_owner': None if self.fault_owner is None else int(self.fault_owner)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'GameOutcome':
        o = cls(key=d['key'], end=d['end'], score0=d.get('score0'), length=int(d.get('length', 0)),
                overtime=d.get('overtime'), seats=tuple(SeatStats.from_dict(s) for s in d['seats']),
                wall_s=float(d.get('wall_s', 0.0)), error=d.get('error'),
                fault_owner=None if d.get('fault_owner') is None else int(d['fault_owner']))
        o.check()
        return o


class GameRunner(Protocol):
    """Plays a batch of gate games (the real one plays the episode loop over frozen models in the game
    environment).
    Returns one outcome per spec, any order, keyed by ``spec.key``. A game that fails returns an outcome with
    a failure ``end`` (crash, step_guard, seat_stall, timeout, error) and, when known, the faulting owner;
    raising aborts the whole chunk. The real runner checks ``spec.env_digest`` against its own environment."""

    def run(self, specs: Sequence[GameSpec]) -> Sequence[GameOutcome]:
        ...


class FunctionRunner:
    """Adapter: a per-game function ``fn(spec) -> GameOutcome`` as a GameRunner. ``catch`` turns an exception of
    one game into ``end='error'`` (off by default: a bug must not look like a flaky game)."""

    def __init__(self, fn: Callable[[GameSpec], GameOutcome], catch: bool = False):
        self.fn, self.catch = fn, bool(catch)

    def run(self, specs: Sequence[GameSpec]) -> List[GameOutcome]:
        out = []
        for s in specs:
            if self.catch:
                try:
                    out.append(self.fn(s))
                except Exception as e:                       # noqa: BLE001 (per-game isolation, opt-in)
                    out.append(GameOutcome(key=s.key, end='error', error=f'{type(e).__name__}: {e}'[:500]))
            else:
                out.append(self.fn(s))
        return out


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class GateSettings:
    """Gate sizes and rules (defaults: example values); ``from_config`` reads ``PPOConfig.gates``."""
    screen_pairs: int = S.SCREEN_PAIRS
    g1_pairs: int = S.G1_PAIRS
    g1c_pairs: int = S.G1C_PAIRS
    g2_pairs: int = 400
    release_pairs: int = S.RELEASE_PAIRS
    aa_pairs: int = 1000
    panel_mix: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PANEL_MIX))
    holdout_kinds: Tuple[str, ...] = HOLDOUT_KINDS
    alpha: float = 0.1
    holdout_min: float = S.HOLDOUT_MIN_DELTA
    min_completion: float = S.MIN_COMPLETION
    n_boot: int = S.N_BOOT
    method: str = 'bootstrap'
    min_clusters: int = S.MIN_CLUSTERS
    swap_level: float = 0.95
    sprt: bool = False
    sprt_h1: float = S.SPRT_DELTA1
    sprt_stop_alpha: Optional[float] = None          # early accept of a G1 (gates.sprt_stop_alpha; None = off)
    h2h_stop_alpha: Optional[float] = None           # two-sided early stop of a G1c (gates.h2h_stop_alpha)
    h2h_h1: float = 0.05                             # that test's H1, score 0.5 + this (gates.h2h_h1_pp / 100)
    rollback_delta: float = S.ROLLBACK_DELTA         # gates.rollback_delta_pp / 100: the screen / G1 alarm
    fault_policy: str = 'drop'                       # FAULT_POLICIES; 'drop' drops a failed unit
    failure_rate_max: float = 0.01                   # monitor.drop_rate_warn: scored-model failures per game
    failure_asym_p: float = 0.01                     # two-sided binomial p of the failure counts of the two sides
    check_schedule: bool = True                      # g1 / g1c / release: size, panel mix and held-out as configured

    def __post_init__(self) -> None:
        if self.fault_policy not in FAULT_POLICIES:
            raise GateError(f'fault_policy {self.fault_policy!r} not in {FAULT_POLICIES}')

    def size(self, kind: str) -> int:
        return {'screen': self.screen_pairs, 'g1': self.g1_pairs, 'aa': self.aa_pairs, 'g1c': self.g1c_pairs,
                'g2': self.g2_pairs, 'release': self.release_pairs}.get(kind, 0)

    @classmethod
    def from_config(cls, cfg: Any, **overrides: Any) -> 'GateSettings':
        """From a ``amini.ppo.config.PPOConfig`` (duck-typed): sizes, the panel mix, alpha, the held-out
        rule (pp -> fraction), min completion, bootstrap resamples, the SPRT switch, the rollback threshold
        (pp -> fraction) and the failure-rate limit (monitor.drop_rate_warn)."""
        g = cfg.gates
        kw = dict(screen_pairs=int(g.screen.games), g1_pairs=int(g.g1.games_per_side), g1c_pairs=int(g.g1c.pairs),
                  g2_pairs=int(g.g2.pairs), release_pairs=int(g.release.pairs), aa_pairs=int(g.aa_null_pairs),
                  panel_mix=dict(g.crn_panel), alpha=float(g.g1.alpha_one_sided),
                  holdout_min=float(g.g1.heldout_min_delta_pp) / 100.0, min_completion=float(g.min_completion),
                  n_boot=int(g.bootstrap_resamples), sprt=bool(g.sprt), sprt_h1=float(g.sprt_h1_pp) / 100.0,
                  sprt_stop_alpha=(None if getattr(g, 'sprt_stop_alpha', None) is None
                                   else float(g.sprt_stop_alpha)),
                  h2h_stop_alpha=(None if getattr(g, 'h2h_stop_alpha', None) is None or
                                  getattr(g, 'flow', 'crn') != 'h2h' else float(g.h2h_stop_alpha)),
                  h2h_h1=float(getattr(g, 'h2h_h1_pp', 5.0)) / 100.0,
                  rollback_delta=float(g.rollback_delta_pp) / 100.0,
                  failure_rate_max=float(cfg.monitor.drop_rate_warn))
        kw.update(overrides)
        return cls(**kw)


def panel_players(cfg: Any, digests: Mapping[str, str], kinds: Sequence[str] = PANEL_KINDS) -> Dict[str, Player]:
    """The CRN panel opponents from ``cfg.opponent_settings()`` (the opponent settings table) and the caller's
    weight digests per kind. Refuses rows without checkpoint or temperature."""
    rows = cfg.opponent_settings()
    out = {}
    for k in kinds:
        if k not in rows:
            raise GateError(f'no opponent settings for panel kind {k}')
        if k not in digests:
            raise GateError(f'no weight digest for panel kind {k}')
        out[k] = Player(name=k, digest=digests[k], spec=C.OpponentSpec(**rows[k]))
    return out


# ------------------------------------------------------------------ plans
def _check_panel_cases(panel: Sequence[C.CaseRef], require_val: bool,
                       min_clusters: int = S.MIN_CLUSTERS) -> List[C.CaseRef]:
    if not panel:
        raise GateError('empty panel')
    ids = [c.case_id for c in panel]
    if len(set(ids)) != len(ids):
        raise GateError('duplicate case ids in the panel')
    if len(ids) < min_clusters:          # stats would mark every gate invalid, and a replay cannot fix that
        raise GateError(f'panel has {len(ids)} openings < {min_clusters} (stats.MIN_CLUSTERS)')
    for c in panel:
        if require_val and c.split != 'val':
            raise GateError(f'panel case {c.case_id} is split {c.split!r}: gate openings are never trained (val only)')
        if c.our_owner not in (0, 1):
            raise GateError(f'case {c.case_id}: our_owner must be 0 or 1')
        C.level_key(c.level)
    return list(panel)


def _case_order(panel: Sequence[C.CaseRef], n: int, base_seed: int, block: str) -> List[C.CaseRef]:
    """Balanced openings: a seeded permutation of the panel, cycled (each opening floor/ceil(n / P) times)."""
    order = sorted(panel, key=lambda c: (C.stable_hash(GATE_NAMESPACE, block, 'case_order', c.case_id), c.case_id))
    return [order[i % len(order)] for i in range(n)]


def _assign_kinds(n: int, mix: Mapping[str, float], base_seed: int, block: str) -> List[str]:
    """Exact panel-opponent counts (largest remainder), spread in a seeded order."""
    if any(v < 0 for v in mix.values()) or abs(sum(mix.values()) - 1.0) > 1e-9:
        raise GateError(f'panel mix must be non-negative and sum to 1: {dict(mix)}')
    counts = C.allocate(n, dict(mix))
    order = sorted(range(n), key=lambda i: (gate_seed(base_seed, block, i, 'opp_order'), i))
    out: List[Optional[str]] = [None] * n
    pos = 0
    for k, c in counts.items():
        for i in order[pos:pos + c]:
            out[i] = k
        pos += c
    return out  # type: ignore[return-value]


def _unit_seeds(base_seed: int, block: str, unit: int, sides: Sequence[str],
                arm_policy: Sequence[str] = ()) -> Tuple[Tuple[str, int], ...]:
    seeds = {r: gate_seed(base_seed, block, unit, r) for r in SHARED_ROLES}
    for side in sides:
        for r in SIDE_ROLES:
            seeds[f'{side}_{r}'] = gate_seed(base_seed, block, unit, f'{side}_{r}')
    for arm in arm_policy:                    # A/A: independent policy seeds of our seat per arm
        for r in POLICY_ROLES:
            seeds[f'our_{r}@{arm}'] = gate_seed(base_seed, block, unit, f'our_{r}@{arm}')
    return tuple(sorted(seeds.items()))


@dataclass(frozen=True)
class GatePlan:
    """The schedule of one gate: kind, seed block, units, arms and players.

    ``players`` holds the arm players ('candidate'/'champion' or 'a'/'b') and, for CRN, the panel opponents
    by kind. ``policy_crn`` False (A/A) gives our seat independent policy seeds per arm. ``env_digest`` (required,
    see ``env_digest()``) pins the environment inside the block digest."""
    kind: str
    block: str
    base_seed: int
    units: Tuple[Unit, ...]
    players: Mapping[str, Player]
    policy_crn: bool = True
    env_digest: str = ''

    def __post_init__(self) -> None:
        if self.kind not in GATE_KINDS:
            raise GateError(f'unknown gate kind {self.kind!r}')
        _check_block(self.block)
        if not isinstance(self.env_digest, str) or not self.env_digest:
            raise GateError(f'{self.block}: env_digest is required (games of another '
                            f'environment are never reused)')
        if [u.unit for u in self.units] != list(range(len(self.units))):
            raise GateError('units must be numbered 0..n-1 in order')
        need = CRN_ARMS if self.crn else CANARY_ARMS if self.canary else ('a', 'b')
        for k in need:
            if k not in self.players:
                raise GateError(f'{self.kind} plan needs player {k!r}')
        if self.canary and not is_canary(self.players['canary']):
            raise GateError(f'{self.block}: the canary arm must be a canary player (canary_player)')
        if self.crn or self.canary:
            for u in self.units:
                if u.opp_kind not in self.players:
                    raise GateError(f'unit {u.unit}: no player for panel opponent {u.opp_kind!r}')

    # -- identity
    @property
    def crn(self) -> bool:
        return self.kind in CRN_KINDS

    @property
    def canary(self) -> bool:
        return self.kind == CANARY_KIND

    @property
    def arms(self) -> Tuple[str, ...]:
        return CRN_ARMS if self.crn else CANARY_ARMS if self.canary else SWAP_ARMS

    @property
    def scored_key(self) -> str:
        """The player key that names the scored model (ledger, stored plan and report file names)."""
        return 'candidate' if self.crn else 'canary' if self.canary else 'a'

    @property
    def n_expected(self) -> int:
        return len(self.units)

    def block_dict(self) -> Dict[str, Any]:
        """The arm-independent part (units, panel opponents, seed rule): equal for every candidate that is
        scored on this block (the screen reuses the champion's stored games through it)."""
        opp = {k: p.to_dict() for k, p in sorted(self.players.items())
               if k not in CRN_ARMS + CANARY_ARMS + ('a', 'b')}
        return {'schema': PLAN_SCHEMA, 'kind': self.kind, 'block': self.block, 'base_seed': int(self.base_seed),
                'policy_crn': bool(self.policy_crn), 'namespace': GATE_NAMESPACE, 'env_digest': self.env_digest,
                'units': [u.to_dict() for u in self.units], 'opponents': opp}

    def block_digest(self) -> str:
        return _sha(self.block_dict())

    def arm_players(self, arm: str) -> Dict[str, Player]:
        if arm not in self.arms:
            raise GateError(f'{self.kind} has no arm {arm!r}')
        if self.crn or self.canary:
            return {arm: self.players[arm]}
        return {'a': self.players['a'], 'b': self.players['b']}

    def arm_digest(self, arm: str) -> str:
        return _sha({'arm': arm, 'players': {k: p.to_dict() for k, p in self.arm_players(arm).items()},
                     'block_digest': self.block_digest()})

    def to_dict(self) -> Dict[str, Any]:
        d = self.block_dict()
        d['players'] = {k: p.to_dict() for k, p in sorted(self.players.items())}
        d['block_digest'] = self.block_digest()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'GatePlan':
        if d.get('schema') != PLAN_SCHEMA:
            raise GateError(f'plan schema {d.get("schema")!r} != {PLAN_SCHEMA}')
        plan = cls(kind=d['kind'], block=d['block'], base_seed=int(d['base_seed']),
                   units=tuple(Unit.from_dict(u) for u in d['units']),
                   players={k: Player.from_dict(p) for k, p in d['players'].items()},
                   policy_crn=bool(d['policy_crn']), env_digest=str(d['env_digest']))
        if d.get('block_digest') is not None and d['block_digest'] != plan.block_digest():
            raise GateError('plan.json block digest does not match its content')
        return plan

    # -- games
    def spec(self, arm: str, unit: int) -> GameSpec:
        u = self.units[unit]
        sm = u.seed_map()
        if self.canary:              # owner-keyed seeds copied from the parent gate's game (canary_plan)
            our = u.our_owner
            players = [None, None]
            players[our] = self.players['canary']
            players[1 - our] = self.players[u.opp_kind]
            return GameSpec(key=game_key(self.block, arm, u.unit), kind=self.kind, block=self.block, unit=u.unit,
                            arm=arm, case_id=u.case_id, level=u.level, our_owner=our,
                            players=(players[0], players[1]), scored_owner=our, seeds=dict(sm), rnd_seed=u.rnd_seed,
                            opp_kind=u.opp_kind, holdout=False, env_digest=self.env_digest)
        seeds = {r: sm[r] for r in SHARED_ROLES}
        if self.crn:
            our, opp = u.our_owner, 1 - u.our_owner
            for r in SIDE_ROLES:
                pol = f'our_{r}@{arm}'
                seeds[f'o{our}_{r}'] = sm[pol] if (not self.policy_crn and pol in sm) else sm[f'our_{r}']
                seeds[f'o{opp}_{r}'] = sm[f'opp_{r}']
            players = [None, None]
            players[our] = self.players[arm]
            players[opp] = self.players[u.opp_kind]
            scored = our
        else:
            for o in (0, 1):
                for r in SIDE_ROLES:
                    seeds[f'o{o}_{r}'] = sm[f'o{o}_{r}']
            a_owner = SWAP_ARMS.index(arm)
            players = [None, None]
            players[a_owner] = self.players['a']
            players[1 - a_owner] = self.players['b']
            scored = a_owner
        return GameSpec(key=game_key(self.block, arm, u.unit), kind=self.kind, block=self.block, unit=u.unit,
                        arm=arm, case_id=u.case_id, level=u.level, our_owner=u.our_owner,
                        players=(players[0], players[1]), scored_owner=scored, seeds=seeds, rnd_seed=u.rnd_seed,
                        opp_kind=u.opp_kind, holdout=u.holdout, env_digest=self.env_digest)

    def games(self, arm: str) -> List[GameSpec]:
        return [self.spec(arm, i) for i in range(len(self.units))]

    def all_seeds(self) -> List[int]:
        """Every 63-bit seed of the plan (for disjointness checks)."""
        return [v for u in self.units for _, v in u.seeds]


def crn_plan(kind: str, block: str, n: int, *, panel: Sequence[C.CaseRef], candidate: Player, champion: Player,
             opponents: Mapping[str, Player], base_seed: int, env_digest: str,
             panel_mix: Optional[Mapping[str, float]] = None, holdout_kinds: Sequence[str] = HOLDOUT_KINDS,
             require_val: bool = True, min_clusters: int = S.MIN_CLUSTERS) -> GatePlan:
    """A CRN gate (screen / g1 / aa): ``n`` triples of (val opening, seed block, panel
    opponent). Openings are balanced over the panel; panel opponents get exact counts from ``panel_mix``
    (default ``DEFAULT_PANEL_MIX``; ``plan_gate`` threads the configured one). Both arms play every triple on our
    seat with the same seeds (A/A: independent policy seeds per arm). Both arms must have the same settings row
    (temperature, gate_bias, hazard_dt, code_tree), and in G1 / screen T_gate = T*_IL (the il_v4_tstar panel
    temperature when that kind has a share)."""
    if kind not in CRN_KINDS:
        raise GateError(f'{kind} is not a CRN gate')
    if n < 1:
        raise GateError('n >= 1')
    mix = dict(DEFAULT_PANEL_MIX if panel_mix is None else panel_mix)
    for k, w in mix.items():
        if w > 0 and k not in opponents:
            raise GateError(f'panel kind {k} has a share but no player')
    diff = settings_diff(candidate, champion)
    if diff:
        raise GateError(f'{kind}: candidate and champion settings differ in {diff} (the gate would measure the '
                        f'settings, not the training)')
    if kind in ('g1', 'screen') and mix.get('il_v4_tstar', 0) > 0 \
            and candidate.spec.temperature != opponents['il_v4_tstar'].spec.temperature:
        raise GateError(f'{kind}: T_gate {candidate.spec.temperature} != T*_IL '
                        f'{opponents["il_v4_tstar"].spec.temperature} (the gate temperature is the IL one)')
    if kind != 'aa' and candidate.digest == champion.digest:
        raise GateError(f'{kind}: candidate and champion are the same weights (use kind "aa")')
    cases = _case_order(_check_panel_cases(panel, require_val, min_clusters), n, base_seed, block)
    kinds = _assign_kinds(n, mix, base_seed, block)
    policy_crn = kind != 'aa'
    units = tuple(Unit(unit=i, case_id=cases[i].case_id, level=int(cases[i].level), our_owner=int(cases[i].our_owner),
                       rnd_seed=gate_rnd_seed(base_seed, block, i),
                       seeds=_unit_seeds(base_seed, block, i, ('our', 'opp'), () if policy_crn else CRN_ARMS),
                       opp_kind=kinds[i], holdout=kinds[i] in holdout_kinds) for i in range(n))
    players = {'candidate': candidate, 'champion': champion}
    for k in sorted(set(kinds)):
        players[k] = opponents[k]
    return GatePlan(kind=kind, block=block, base_seed=int(base_seed), units=units, players=players,
                    policy_crn=policy_crn, env_digest=env_digest)


def setting_crn_plan(tag: str, n: int, *, panel: Sequence[C.CaseRef], candidate: Player, champion: Player,
                     opponents: Mapping[str, Player], base_seed: int, env_digest: str,
                     panel_mix: Optional[Mapping[str, float]] = None, attempt: int = 0, require_val: bool = True,
                     min_clusters: int = S.MIN_CLUSTERS) -> GatePlan:
    """A CRN MEASUREMENT of a setting, never a gate (e.g. an IL temperature measured through the PPO runner):
    both arms are the SAME weights (equal digest) and differ only in their settings row (e.g. temperature: the
    IL model at T vs at T = 1). The units, seeds and pairing are exactly a G1's (``policy_crn``: same
    environment, opponent and policy seeds in both arms), so the arms are CRN pairs. Kind 'aa' (record only:
    scored by ``evaluate`` with ``stats.screen_s``, never blocking), block ``aa-<tag>-a<attempt>``.
    ``panel_mix`` default {'il_v4_t1': 1.0}: the opponent is the IL model at T = 1 with no bias, not the gate
    panel, whose T* row is what is being measured."""
    if candidate.digest != champion.digest:
        raise GateError(f'setting measurement: the arms must be the same weights ({candidate.digest[:12]} != '
                        f'{champion.digest[:12]}); use a gate for different weights')
    diff = settings_diff(candidate, champion)
    if not diff:
        raise GateError('setting measurement: the arms have the same settings (use kind "aa" for an A/A test)')
    if n < 1:
        raise GateError('n >= 1')
    block = _check_block(f'aa-{tag}-a{int(attempt)}')
    mix = dict({'il_v4_t1': 1.0} if panel_mix is None else panel_mix)
    for k, w in mix.items():
        if w > 0 and k not in opponents:
            raise GateError(f'panel kind {k} has a share but no player')
    cases = _case_order(_check_panel_cases(panel, require_val, min_clusters), n, base_seed, block)
    kinds = _assign_kinds(n, mix, base_seed, block)
    units = tuple(Unit(unit=i, case_id=cases[i].case_id, level=int(cases[i].level), our_owner=int(cases[i].our_owner),
                       rnd_seed=gate_rnd_seed(base_seed, block, i),
                       seeds=_unit_seeds(base_seed, block, i, ('our', 'opp'), ()),
                       opp_kind=kinds[i], holdout=False) for i in range(n))
    players = {'candidate': candidate, 'champion': champion}
    for k in sorted(set(kinds)):
        players[k] = opponents[k]
    return GatePlan(kind='aa', block=block, base_seed=int(base_seed), units=units, players=players,
                    policy_crn=True, env_digest=env_digest)


def seat_swap_plan(kind: str, block: str, n_pairs: int, *, panel: Sequence[C.CaseRef], a: Player, b: Player,
                   base_seed: int, env_digest: str, require_val: bool = True,
                   min_clusters: int = S.MIN_CLUSTERS) -> GatePlan:
    """A seat-swapped gate (g1c / g2 / release / swap): ``n_pairs`` pairs on balanced val
    openings; arm a0 = A on owner 0, a1 = A on owner 1; both games of a pair share the environment seed and the
    per-owner seeds. G1c (candidate vs champion at T_gate, like G1) requires equal settings rows."""
    if kind not in SWAP_KINDS:
        raise GateError(f'{kind} is not a seat-swap gate')
    if n_pairs < 1:
        raise GateError('n_pairs >= 1')
    if kind == 'g1c':
        diff = settings_diff(a, b)
        if diff:
            raise GateError(f'g1c: candidate and champion settings differ in {diff}')
        if a.digest == b.digest:
            raise GateError('g1c: candidate and champion are the same weights')
    cases = _case_order(_check_panel_cases(panel, require_val, min_clusters), n_pairs, base_seed, block)
    units = tuple(Unit(unit=i, case_id=cases[i].case_id, level=int(cases[i].level), our_owner=int(cases[i].our_owner),
                       rnd_seed=gate_rnd_seed(base_seed, block, i),
                       seeds=_unit_seeds(base_seed, block, i, ('o0', 'o1'))) for i in range(n_pairs))
    return GatePlan(kind=kind, block=block, base_seed=int(base_seed), units=units, players={'a': a, 'b': b},
                    env_digest=env_digest)


def plan_gate(kind: str, settings: GateSettings, *, panel: Sequence[C.CaseRef], candidate: Player, champion: Player,
              base_seed: int, env_digest: str, opponents: Optional[Mapping[str, Player]] = None,
              update: Optional[int] = None, attempt: int = 0, tag: Optional[str] = None, confirm: bool = False,
              n: Optional[int] = None, require_val: bool = True) -> GatePlan:
    """A gate plan with the configured block name, size, panel mix and held-out kinds.

    CRN kinds play ``candidate`` / ``champion`` against ``opponents``; seat-swap kinds play A = ``candidate``
    against B = ``champion`` (for g2 / release B is the IL reference at T*). ``n`` overrides the size (required
    for 'swap')."""
    blk = block_name(kind, update=update, attempt=attempt, champion_digest=champion.digest, tag=tag, confirm=confirm)
    size = settings.size(kind) if n is None else int(n)
    if size < 1:
        raise GateError(f'{kind}: no configured size, pass n')
    if kind in CRN_KINDS:
        if opponents is None:
            raise GateError(f'{kind} needs the panel opponents')
        return crn_plan(kind, blk, size, panel=panel, candidate=candidate, champion=champion, opponents=opponents,
                        base_seed=base_seed, env_digest=env_digest, panel_mix=settings.panel_mix,
                        holdout_kinds=settings.holdout_kinds, require_val=require_val,
                        min_clusters=settings.min_clusters)
    return seat_swap_plan(kind, blk, size, panel=panel, a=candidate, b=champion, base_seed=base_seed,
                          env_digest=env_digest, require_val=require_val, min_clusters=settings.min_clusters)


# ------------------------------------------------------------------ canaries (diagnostic only)
def canary_player(kind: str, *, hazard_dt: Optional[str] = None, code_tree: Optional[str] = 'run') -> Player:
    """A canary policy as a gate player (config ``gates.canaries``): 'all_wait' never acts,
    'random_legal' acts uniformly among the legal families / slots / cells / abilities
    (played in-process by the runner). Its checkpoint is the pseudo path ``canary:<kind>`` (no file, no server) and
    its digest a fixed hash of the kind; T = 1, no gate bias."""
    if kind not in CANARY_POLICIES:
        raise GateError(f'unknown canary {kind!r}; one of {CANARY_POLICIES}')
    name = f'{CANARY_MODEL_PREFIX}{kind}'
    return Player(name=name, digest=_sha({'canary': kind, 'namespace': GATE_NAMESPACE}),
                  spec=C.OpponentSpec(name=name, checkpoint=f'{CANARY_CHECKPOINT_PREFIX}{kind}', temperature=1.0,
                                      gate_bias=None, hazard_dt=hazard_dt, code_tree=code_tree))


def canary_kind(player: Player) -> Optional[str]:
    """The canary policy of a player ('all_wait' / 'random_legal'), or None for a model."""
    ck = str(player.spec.checkpoint or '')
    if not ck.startswith(CANARY_CHECKPOINT_PREFIX):
        return None
    kind = ck[len(CANARY_CHECKPOINT_PREFIX):]
    if kind not in CANARY_POLICIES or player.digest != canary_player(kind).digest:
        raise GateError(f'{player.name}: malformed canary player ({ck}, digest {player.digest[:12]})')
    return kind


def is_canary(player: Player) -> bool:
    return canary_kind(player) is not None


def canary_plan(parent: GatePlan, kind: str, n: int, *, hazard_dt: Optional[str] = None,
                code_tree: Optional[str] = 'run') -> GatePlan:
    """The canary block of one gate: the canary policy on OUR seat of the first ``min(n, units)``
    units of ``parent`` against the same other seat as the parent's scored arm (CRN: the unit's panel opponent;
    seat-swap: model B, on arm ``a<our_owner>``), with exactly that game's opening, environment rndSeed and per-owner
    seeds. Block ``canary-<kind>-<parent block>`` (a screen / G2 block is reused across candidates, so are its
    canary games). Single arm 'canary', kind ``canary``: never blocking, never promoting, never seen by the
    monitor (:func:`evaluate_canary`)."""
    if parent.kind not in CANARY_PARENT_KINDS:
        raise GateError(f'{parent.kind} gates carry no canaries (only {CANARY_PARENT_KINDS})')
    if int(n) < 1:
        raise GateError('canary games >= 1')
    cp = canary_player(kind, hazard_dt=hazard_dt, code_tree=code_tree)
    players: Dict[str, Player] = {'canary': cp}
    units = []
    for u in parent.units[:int(n)]:
        arm = 'candidate' if parent.crn else f'a{int(u.our_owner)}'
        sp = parent.spec(arm, u.unit)
        if sp.scored_owner != u.our_owner:                                  # pragma: no cover - plan invariant
            raise GateError(f'{sp.key}: scored owner {sp.scored_owner} != our owner {u.our_owner}')
        other = sp.players[1 - sp.scored_owner]
        key = u.opp_kind if parent.crn else 'opp_b'
        if key in players and players[key] != other:                       # pragma: no cover - plan invariant
            raise GateError(f'{parent.block}: two players for {key}')
        players[key] = other
        units.append(Unit(unit=len(units), case_id=u.case_id, level=u.level, our_owner=u.our_owner,
                          rnd_seed=u.rnd_seed, seeds=tuple(sorted((str(k), int(v)) for k, v in sp.seeds.items())),
                          opp_kind=key, holdout=False))
    return GatePlan(kind=CANARY_KIND, block=_check_block(f'canary-{kind}-{parent.block}'), base_seed=parent.base_seed,
                    units=tuple(units), players=players, policy_crn=True, env_digest=parent.env_digest)


# ------------------------------------------------------------------ disjointness
def check_disjoint(plan: GatePlan, training: Iterable[Any], *, base_seed: Optional[int] = None) -> None:
    """Raise GateError if a gate plan shares a 63-bit seed or a (case_id, rndSeed) with training games.

    ``training`` are ``cases.GamePlan`` objects (seed, rnd_seed, case_id, update, replica); with
    ``base_seed`` every role seed of ``cases.game_seeds`` is compared as well."""
    gate_seeds = set(plan.all_seeds())
    gate_cases = {(u.case_id, u.rnd_seed) for u in plan.units}
    for p in training:
        tr = {int(p.seed)}
        if base_seed is not None:
            tr.update(v for k, v in C.game_seeds(base_seed, int(p.update), int(p.replica)).items() if k != 'rnd_seed')
        hit = gate_seeds & tr
        if hit:
            raise GateError(f'gate {plan.block} shares seed {sorted(hit)[0]} with training update {p.update} '
                            f'replica {p.replica}')
        if (p.case_id, int(p.rnd_seed)) in gate_cases:
            raise GateError(f'gate {plan.block} replays training game ({p.case_id}, rndSeed {p.rnd_seed})')


def check_panel(panel: Sequence[C.CaseRef], training_case_ids: Iterable[str]) -> None:
    """Raise GateError if a panel opening is also a training opening (gate openings are never trained)."""
    hit = sorted({c.case_id for c in panel} & set(training_case_ids))
    if hit:
        raise GateError(f'{len(hit)} panel openings are training openings, e.g. {hit[:3]}')


# ------------------------------------------------------------------ atomic files
def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON to ``path`` atomically: temp file in the same directory, fsync, os.replace, dir fsync."""
    path = Path(path)
    new_dir = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_dir:
        _fsync_dir(path.parent.parent)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, sort_keys=True, indent=1, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def _read_json(path: Path) -> Any:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _finite(obj: Any) -> Any:
    """NaN / inf -> None, recursively (strict JSON)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating,)):
        return _finite(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


# ------------------------------------------------------------------ seed ledger
class SeedLedger:
    """``<root>/ledger.json``: every block ever scheduled, its kind, schedule digest and the candidates scored
    on it. ``claim(plan)`` refuses (BlockReuseError) a block whose schedule changed, and a fresh-seed block
    (g1, g1c, release) claimed again by a different candidate (a new seed block per G1)."""

    def __init__(self, root: Union[str, Path]):
        self.path = Path(root) / 'ledger.json'

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {'schema': LEDGER_SCHEMA, 'blocks': {}}
        d = _read_json(self.path)
        if d.get('schema') != LEDGER_SCHEMA:
            raise GateError(f'ledger schema {d.get("schema")!r} != {LEDGER_SCHEMA}')
        return d

    def claim(self, plan: GatePlan) -> None:
        d = self.load()
        cand = plan.players[plan.scored_key].digest
        ent = d['blocks'].get(plan.block)
        if ent is None:
            d['blocks'][plan.block] = {'kind': plan.kind, 'block_digest': plan.block_digest(), 'candidates': [cand]}
        else:
            if ent['kind'] != plan.kind or ent['block_digest'] != plan.block_digest():
                raise BlockReuseError(f'block {plan.block} was scheduled with a different schedule '
                                      f'({ent["kind"]}, {ent["block_digest"][:12]}): use a new block')
            if cand in ent['candidates']:
                return
            if plan.kind in FRESH_KINDS:
                raise BlockReuseError(f'block {plan.block} ({plan.kind}) already scored candidate '
                                      f'{ent["candidates"][0][:12]}: every {plan.kind} needs a fresh seed block')
            ent['candidates'].append(cand)
        atomic_write_json(self.path, d)


# ------------------------------------------------------------------ store
def _n_chunks(n: int, chunk_size: int) -> int:
    return (n + chunk_size - 1) // chunk_size


class GateStore:
    """Chunk files of all gates under one root (gates are stored in chunks)::

        <root>/ledger.json
        <root>/<block>/plan.json                       block part (units, panel opponents)
        <root>/<block>/<arm>-<arm digest[:16]>/arm.json players, chunk size
        <root>/<block>/<arm>-<arm digest[:16]>/c00000.json ...
        <root>/<block>/report-<candidate digest[:16]>.json
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    def block_dir(self, block: str) -> Path:
        return self.root / _check_block(block)

    def arm_dir(self, plan: GatePlan, arm: str) -> Path:
        return self.block_dir(plan.block) / f'{arm}-{plan.arm_digest(arm)[:16]}'

    def chunk_path(self, plan: GatePlan, arm: str, idx: int) -> Path:
        return self.arm_dir(plan, arm) / f'c{int(idx):05d}.json'

    # -- plan and arm records
    def prepare(self, plan: GatePlan, arm: str, chunk_size: int) -> None:
        """Write plan.json / arm.json on first use; refuse a different schedule or chunk size later."""
        pp = self.block_dir(plan.block) / 'plan.json'
        if pp.exists():
            have = _read_json(pp)
            if have.get('block_digest') != plan.block_digest():
                raise BlockReuseError(f'{pp}: stored schedule {str(have.get("block_digest"))[:12]} != plan '
                                      f'{plan.block_digest()[:12]}')
        else:
            d = plan.block_dict()
            d['block_digest'] = plan.block_digest()
            atomic_write_json(pp, d)
        ap = self.arm_dir(plan, arm) / 'arm.json'
        rec = {'arm': arm, 'arm_digest': plan.arm_digest(arm), 'block_digest': plan.block_digest(),
               'chunk_size': int(chunk_size), 'players': {k: p.to_dict() for k, p in plan.arm_players(arm).items()}}
        if ap.exists():
            have = _read_json(ap)
            if have.get('arm_digest') != rec['arm_digest'] or have.get('block_digest') != rec['block_digest']:
                raise GateError(f'{ap}: stored arm does not match the plan')
            if int(have.get('chunk_size', -1)) != int(chunk_size):
                raise GateError(f'{ap}: stored chunk size {have.get("chunk_size")} != {chunk_size} '
                                f'(resume with the same chunk size)')
        else:
            atomic_write_json(ap, rec)

    def chunk_size(self, plan: GatePlan, arm: str) -> Optional[int]:
        ap = self.arm_dir(plan, arm) / 'arm.json'
        return int(_read_json(ap)['chunk_size']) if ap.exists() else None

    def save_plan(self, plan: GatePlan) -> Path:
        """Full plan (with the arm players) for the CLI: ``<block>/plan-<candidate digest[:16]>.json``."""
        cand = plan.players[plan.scored_key].digest
        p = self.block_dir(plan.block) / f'plan-{cand[:16]}.json'
        atomic_write_json(p, plan.to_dict())
        return p

    # -- chunks
    def _chunk_units(self, plan: GatePlan, idx: int, chunk_size: int) -> range:
        lo = idx * chunk_size
        return range(lo, min(lo + chunk_size, plan.n_expected))

    def load_chunk(self, plan: GatePlan, arm: str, idx: int, chunk_size: int) -> Optional[List[GameOutcome]]:
        """The stored outcomes of one chunk, None if not written. Any mismatch with the plan raises."""
        p = self.chunk_path(plan, arm, idx)
        if not p.exists():
            return None
        try:
            d = _read_json(p)
        except (OSError, ValueError) as e:
            raise GateError(f'{p}: unreadable chunk ({e})') from e
        keys = [game_key(plan.block, arm, i) for i in self._chunk_units(plan, idx, chunk_size)]
        want = {'schema': CHUNK_SCHEMA, 'block': plan.block, 'block_digest': plan.block_digest(), 'arm': arm,
                'arm_digest': plan.arm_digest(arm), 'chunk': int(idx), 'chunk_size': int(chunk_size), 'keys': keys}
        for k, v in want.items():
            if d.get(k) != v:
                raise GateError(f'{p}: chunk {k} does not match the plan')
        outs = [GameOutcome.from_dict(o) for o in d.get('outcomes', [])]
        if [o.key for o in outs] != keys:
            raise GateError(f'{p}: outcome keys do not match the chunk keys')
        return outs

    def write_chunk(self, plan: GatePlan, arm: str, idx: int, chunk_size: int, outcomes: Sequence[GameOutcome]) -> Path:
        keys = [game_key(plan.block, arm, i) for i in self._chunk_units(plan, idx, chunk_size)]
        by_key: Dict[str, GameOutcome] = {}
        for o in outcomes:
            if not isinstance(o, GameOutcome):
                raise GateError(f'runner returned {type(o).__name__}, not GameOutcome')
            o.check()
            if o.key in by_key:
                raise GateError(f'runner returned {o.key} twice')
            by_key[o.key] = o
        if set(by_key) != set(keys):
            miss, extra = sorted(set(keys) - set(by_key)), sorted(set(by_key) - set(keys))
            raise GateError(f'chunk {idx} of {plan.block}/{arm}: missing {miss[:3]}, unexpected {extra[:3]}')
        d = {'schema': CHUNK_SCHEMA, 'block': plan.block, 'block_digest': plan.block_digest(), 'arm': arm,
             'arm_digest': plan.arm_digest(arm), 'chunk': int(idx), 'chunk_size': int(chunk_size), 'keys': keys,
             'outcomes': [by_key[k].to_dict() for k in keys]}
        p = self.chunk_path(plan, arm, idx)
        atomic_write_json(p, d)
        return p

    def outcomes(self, plan: GatePlan, arm: str) -> Dict[str, GameOutcome]:
        """All stored outcomes of one arm, by game key (missing chunks are simply absent)."""
        cs = self.chunk_size(plan, arm)
        if cs is None:
            return {}
        out: Dict[str, GameOutcome] = {}
        d = self.arm_dir(plan, arm)
        valid = {f'c{i:05d}.json' for i in range(_n_chunks(plan.n_expected, cs))}
        for f in sorted(d.glob('c*.json')):
            if f.name not in valid:
                raise GateError(f'{f}: chunk outside the schedule')
        for i in range(_n_chunks(plan.n_expected, cs)):
            ch = self.load_chunk(plan, arm, i, cs)
            for o in ch or ():
                out[o.key] = o
        return out

    def status(self, plan: GatePlan, chunk_size: Optional[int] = None) -> Dict[str, Any]:
        st = {}
        for arm in plan.arms:
            cs = self.chunk_size(plan, arm) or chunk_size or DEFAULT_CHUNK
            n = _n_chunks(plan.n_expected, cs)
            done = [i for i in range(n) if self.chunk_path(plan, arm, i).exists()]
            outs = self.outcomes(plan, arm)
            st[arm] = {'chunks_total': n, 'chunks_done': len(done), 'games': len(outs),
                       'failed': sum(1 for o in outs.values() if not o.ok)}
        return st


# ------------------------------------------------------------------ stop handling
class StopFlag:
    """Stop predicate for ``run_gate``: True after a signal (``install`` catches SIGTERM by default; pass
    SIGINT as well to catch it) or when any STOP marker file exists (e.g. a queue's or a run's STOP file)."""

    def __init__(self, markers: Sequence[Union[str, Path]] = ()):
        self.markers = [Path(m) for m in markers]
        self.signalled: Optional[int] = None
        self._old: Dict[int, Any] = {}

    def install(self, signals: Sequence[int] = (signal.SIGTERM,)) -> 'StopFlag':
        for s in signals:
            self._old[s] = signal.signal(s, self._handle)
        return self

    def uninstall(self) -> None:
        for s, h in self._old.items():
            signal.signal(s, h)
        self._old.clear()

    def _handle(self, signum: int, frame: Any) -> None:            # noqa: ARG002
        self.signalled = signum

    def __call__(self) -> bool:
        return self.signalled is not None or any(m.exists() for m in self.markers)


@dataclass
class RunStatus:
    """What ``run_gate`` did. ``state``: 'complete' (every chunk stored), 'stopped' (stop predicate),
    'deadline' (not enough time left for the next chunk) or 'early' (the ``early`` predicate accepted the
    pairs played so far; the remaining chunks are not played)."""
    block: str
    state: str
    chunks_total: int
    chunks_done: int
    chunks_run: int
    games_run: int
    failed_run: int
    reused: int
    seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_gate(plan: GatePlan, runner: GameRunner, store: GateStore, *, chunk_size: int = DEFAULT_CHUNK,
             arms: Optional[Sequence[str]] = None, stop: Optional[Callable[[], bool]] = None,
             deadline: Optional[float] = None, chunk_estimate_s: float = 0.0, retry_failed: bool = False,
             ledger: bool = True, clock: Callable[[], float] = time.monotonic,
             log: Optional[Callable[[str], None]] = None,
             early: Optional[Callable[[int], bool]] = None, group: int = 1) -> RunStatus:
    """Play every missing chunk of ``plan``, arms interleaved chunk by chunk.

    ``group`` (default 1 = one runner call per chunk and arm): the missing chunks of ``group`` consecutive chunk
    indices, all arms, are played by ONE ``runner.run`` call and then written chunk by chunk exactly as with
    ``group = 1`` (same files, same keys, same chunk size). A runner that starts and closes its worker processes
    on every call pays that cost once per group instead of once per chunk and arm.
    ``stop`` / ``deadline`` are then asked once per group and ``early`` once after each group (a subset of the
    looks of ``group = 1``; a resumed gate with the same ``group`` stops where the first run did).

    ``early``: called with the number of units whose games are stored for every arm after each chunk index
    but the last (also for chunks found in the store, so a resumed gate stops where the first run did); True ends
    the gate with state 'early'.

    Stored chunks are skipped (a resumed gate plays only what is missing, and with a deterministic runner gives
    the same outcomes as an uninterrupted one). Before each chunk: ``stop()`` -> state 'stopped'; ``deadline``
    (a ``clock()`` value) closer than the expected chunk time (max of ``chunk_estimate_s`` and the slowest chunk
    so far) -> state 'deadline'. ``retry_failed`` replays stored chunks that hold failed games (the whole chunk,
    atomically replaced). A runner exception propagates; the running chunk is then not written."""
    if chunk_size < 1:
        raise GateError('chunk_size >= 1')
    arms = tuple(plan.arms if arms is None else arms)
    for a in arms:
        if a not in plan.arms:
            raise GateError(f'{plan.kind} has no arm {a!r}')
    if ledger:
        SeedLedger(store.root).claim(plan)
    for a in arms:
        store.prepare(plan, a, chunk_size)
    store.save_plan(plan)
    t0 = clock()
    n_chunks = _n_chunks(plan.n_expected, chunk_size)
    total = n_chunks * len(arms)
    run = games = failed = reused = 0
    slowest = float(chunk_estimate_s)
    state = 'complete'
    say = log or (lambda m: None)
    group = max(1, int(group))
    for lo in (range(0, n_chunks, group) if group > 1 else ()):
        hi = min(lo + group, n_chunks)
        todo: List[Tuple[int, str]] = []
        for idx in range(lo, hi):
            for arm in arms:
                have = store.load_chunk(plan, arm, idx, chunk_size)
                if have is not None and not (retry_failed and any(not o.ok for o in have)):
                    reused += 1
                else:
                    todo.append((idx, arm))
        if todo:
            if stop is not None and stop():
                state = 'stopped'
                break
            if deadline is not None and clock() + max(slowest, float(chunk_estimate_s) * len(todo)) > deadline:
                state = 'deadline'
                break
            parts = [[plan.spec(arm, i) for i in store._chunk_units(plan, idx, chunk_size)] for idx, arm in todo]
            ts = clock()
            outs = list(runner.run([sp for part in parts for sp in part]))
            if len(outs) != sum(len(part) for part in parts):
                raise GateError(f'{plan.block}: {len(outs)} outcomes for {sum(len(part) for part in parts)} games')
            k = 0
            for (idx, arm), part in zip(todo, parts):
                mine = outs[k:k + len(part)]
                k += len(part)
                store.write_chunk(plan, arm, idx, chunk_size, mine)
                run += 1
                games += len(part)
                failed += sum(1 for o in mine if not o.ok)
            slowest = max(slowest, clock() - ts)
            say(f'{plan.block} chunks {lo + 1}-{hi}/{n_chunks} ({len(todo)} played together): '
                f'{sum(len(part) for part in parts)} games, {sum(1 for o in outs if not o.ok)} failed')
        if early is not None and hi < n_chunks and early(min(hi * chunk_size, plan.n_expected)):
            state = 'early'
            break
    for idx in (range(n_chunks) if group <= 1 else ()):
        for arm in arms:
            have = store.load_chunk(plan, arm, idx, chunk_size)
            if have is not None and not (retry_failed and any(not o.ok for o in have)):
                reused += 1
                continue
            if stop is not None and stop():
                state = 'stopped'
                break
            if deadline is not None and clock() + slowest > deadline:
                state = 'deadline'
                break
            specs = [plan.spec(arm, i) for i in store._chunk_units(plan, idx, chunk_size)]
            ts = clock()
            outs = list(runner.run(specs))
            store.write_chunk(plan, arm, idx, chunk_size, outs)
            slowest = max(slowest, clock() - ts)
            run += 1
            games += len(specs)
            failed += sum(1 for o in outs if not o.ok)
            say(f'{plan.block} {arm} chunk {idx + 1}/{n_chunks}: {len(specs)} games, '
                f'{sum(1 for o in outs if not o.ok)} failed')
        if state != 'complete':
            break
        if early is not None and idx + 1 < n_chunks and early(min((idx + 1) * chunk_size, plan.n_expected)):
            state = 'early'
            break
    done = sum(1 for i in range(n_chunks) for a in arms if store.chunk_path(plan, a, i).exists())
    return RunStatus(block=plan.block, state=state if done < total else 'complete', chunks_total=total,
                     chunks_done=done, chunks_run=run, games_run=games, failed_run=failed, reused=reused,
                     seconds=float(clock() - t0))


# ------------------------------------------------------------------ diagnostics
def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(np.mean(xs)) if len(xs) else None


def _seat_summary(stats_list: Sequence[SeatStats], minutes: Sequence[float]) -> Dict[str, Any]:
    n = len(stats_list)
    tot_min = float(sum(minutes))
    refusals: Counter = Counter()
    cards: Counter = Counter()
    for s in stats_list:
        refusals.update({str(k): int(v) for k, v in s.refusals.items()})
        cards.update({str(k): int(v) for k, v in s.cards.items()})
    plays = sum(s.plays for s in stats_list)
    return {'games': n,
            'non_wait_per_game': _mean([s.non_wait for s in stats_list]),
            'plays_per_game': _mean([s.plays for s in stats_list]),
            'plays_per_min': plays / tot_min if tot_min > 0 else None,
            'abilities_per_min': sum(s.abilities for s in stats_list) / tot_min if tot_min > 0 else None,
            'queries_per_game': _mean([s.queries for s in stats_list]),
            'refusals': dict(sorted(refusals.items())),
            'refusals_per_100_plays': {k: 100.0 * v / plays for k, v in sorted(refusals.items())} if plays else {},
            'card_play_rate': {k: v / n for k, v in sorted(cards.items())} if n else {}}


def _passive(sel: Sequence[Tuple[GameSpec, GameOutcome]]) -> Dict[str, Any]:
    """Passive-opponent games: games where the other (opponent) seat made <= PASSIVE_NONWAIT_MAX non-wait
    commands, and the scored side's score in them."""
    pas = [(s, o) for s, o in sel if o.seats[1 - s.scored_owner].non_wait <= PASSIVE_NONWAIT_MAX]
    return {'games': len(pas), 'frac': len(pas) / len(sel) if sel else None,
            'score': _mean([o.score(s.scored_owner) for s, o in pas])}


def diagnostics(plan: GatePlan, outcomes: Mapping[str, Mapping[str, GameOutcome]]) -> Dict[str, Any]:
    """Per arm: games, failures by reason (and by faulting owner), length, overtime fraction, scored side and
    other side summaries (non-wait actions per game, plays / abilities per minute, refusals by code, per-card
    play rate), the passive-opponent games (other seat <= PASSIVE_NONWAIT_MAX non-wait commands) and their score;
    for CRN also the other (panel opponent) seat by opponent kind, the "opponent turns passive" check."""
    out: Dict[str, Any] = {}
    for arm in plan.arms:
        got = outcomes.get(arm, {})
        native: List[Tuple[GameSpec, GameOutcome]] = []
        fails: Counter = Counter()
        fault_side: Counter = Counter()
        for i in range(plan.n_expected):
            spec = plan.spec(arm, i)
            o = got.get(spec.key)
            if o is None:
                fails['missing'] += 1
            elif not o.ok:
                fails[o.end] += 1
                fault_side['unknown' if o.fault_owner is None else
                           'scored' if o.fault_owner == spec.scored_owner else 'other'] += 1
            else:
                native.append((spec, o))
        minutes = [max(o.length, 1) / TICKS_PER_MIN for _, o in native]
        scored = [o.seats[s.scored_owner] for s, o in native]
        other = [o.seats[1 - s.scored_owner] for s, o in native]
        scores = [o.score(s.scored_owner) for s, o in native]
        k = float(sum(scores))
        lo, hi = S.wilson(k, len(scores))
        rec = {'scheduled': plan.n_expected, 'native': len(native), 'failures': dict(sorted(fails.items())),
               'fault_side': dict(sorted(fault_side.items())),
               'length_mean': _mean([o.length for _, o in native]),
               'length_p50': float(np.median([o.length for _, o in native])) if native else None,
               'overtime_frac': _mean([float(o.is_overtime) for _, o in native]),
               'score': k / len(scores) if scores else None, 'wilson': [lo, hi],
               'wins': sum(1 for x in scores if x == 1.0), 'draws': sum(1 for x in scores if x == 0.5),
               'losses': sum(1 for x in scores if x == 0.0),
               'scored_seat': _seat_summary(scored, minutes), 'other_seat': _seat_summary(other, minutes),
               'passive_opponent': _passive(native),
               'wall_s': float(sum(o.wall_s for o in got.values()))}
        if plan.crn or plan.canary:
            by_kind: Dict[str, Any] = {}
            for kind in sorted({s.opp_kind for s, _ in native if s.opp_kind}):
                sel = [(s, o) for s, o in native if s.opp_kind == kind]
                by_kind[kind] = {'games': len(sel),
                                 'score': _mean([o.score(s.scored_owner) for s, o in sel]),
                                 'opponent_non_wait_per_game': _mean([o.seats[1 - s.scored_owner].non_wait
                                                                      for s, o in sel]),
                                 'passive_opponent': _passive(sel)}
            rec['by_opponent'] = by_kind
        out[arm] = rec
    return out


# ------------------------------------------------------------------ evaluation
@dataclass
class GateReport:
    """One scored gate. ``stats`` is the stats.CrnResult / SeatSwapResult as a dict. ``passed`` is the gate's
    own rule (G1: LB > 0 and held-out >= -1 pp; G1c / release: CI lower bound > 0.5) AND the failure guard AND
    (``check_schedule``) the configured schedule; record-only kinds never pass and have ``blocking`` False.
    ``replay_required`` = completion < min_completion (a replay on a new attempt block can fix only that)."""
    schema: str
    kind: str
    block: str
    block_digest: str
    env_digest: str
    players: Dict[str, Dict[str, str]]
    n_expected: int
    n_complete: int
    completion: float
    valid: bool
    passed: bool
    blocking: bool
    replay_required: bool
    alarm: bool
    stats: Dict[str, Any]
    rho: Optional[float] = None
    subpanel: Dict[str, Any] = field(default_factory=dict)
    sprt: Optional[Dict[str, Any]] = None
    failure_check: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    by_win_condition: Dict[str, Any] = field(default_factory=dict)   # diagnostic (win_condition_breakdown)

    def as_dict(self) -> Dict[str, Any]:
        return _finite(asdict(self))

    @property
    def delta_hat(self) -> Optional[float]:
        v = self.stats.get('delta_hat')
        return None if v is None or not math.isfinite(v) else float(v)

    def monitor_args(self, update: int, promoted: bool = False) -> Dict[str, Any]:
        """Keyword arguments of ``monitor.Monitor.observe_gate`` (screen and G1 only)."""
        if self.kind not in ('screen', 'g1'):
            raise GateError('only screen and G1 feed Monitor.observe_gate')
        return {'kind': self.kind, 'update': int(update), 'delta_hat': self.delta_hat if self.valid else float('nan'),
                'promoted': bool(promoted), 'valid': bool(self.valid)}

    def pilot_args(self) -> Dict[str, float]:
        """``g1_delta`` / ``g1_upper`` of the phase rules' ``pilot_pass`` from a G1: the upper bound
        is the one the rule's method uses (bootstrap 1 - alpha quantile, else the analytic bound); NaN when the
        G1 is invalid."""
        if self.kind != 'g1':
            raise GateError('pilot_args takes a G1 report')
        st = self.stats
        ub = st.get('ub_boot') if st.get('method') == 'bootstrap' and st.get('ub_boot') is not None \
            else st.get('ub_analytic')
        if not self.valid or self.delta_hat is None or ub is None:
            return {'g1_delta': float('nan'), 'g1_upper': float('nan')}
        return {'g1_delta': float(self.delta_hat), 'g1_upper': float(ub)}


def boot_seed(block: str) -> int:
    """Bootstrap seed of a gate: fixed per block (reproducible, independent across gates)."""
    return C.stable_hash(GATE_NAMESPACE, 'bootstrap', _check_block(block)) & 0xFFFFFFFF


def _scored(o: Optional[GameOutcome], spec: GameSpec, policy: str) -> Optional[float]:
    """The scored side's result of one game: its score in a normal finish; under fault_policy 'loss' a failed
    game with a known fault_owner is a loss of the faulting seat; otherwise None (the unit is dropped)."""
    if o is None:
        return None
    if o.ok:
        return o.score(spec.scored_owner)
    if policy == 'loss' and o.fault_owner is not None:
        return 0.0 if o.fault_owner == spec.scored_owner else 1.0
    return None


def _crn_rows(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]],
              policy: str = 'drop') -> List[S.CrnRecord]:
    recs = []
    for u in plan.units:
        res = []
        for arm in CRN_ARMS:
            spec = plan.spec(arm, u.unit)
            res.append(_scored(outs.get(arm, {}).get(spec.key), spec, policy))
        recs.append(S.CrnRecord(u.unit, u.case_id, res[0], res[1], u.holdout))
    return recs


def _swap_rows(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]],
               policy: str = 'drop') -> List[S.SeatGame]:
    rows = []
    for u in plan.units:
        for arm in SWAP_ARMS:
            spec = plan.spec(arm, u.unit)
            rows.append(S.SeatGame(u.unit, spec.scored_owner,
                                   _scored(outs.get(arm, {}).get(spec.key), spec, policy), u.case_id))
    return rows


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value of k successes in n trials at p = 1/2."""
    if n <= 0:
        return 1.0
    m = min(int(k), int(n) - int(k))
    tail = sum(math.comb(n, i) for i in range(m + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def failure_check(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]],
                  settings: GateSettings) -> Tuple[Dict[str, Any], List[str], bool]:
    """Failure guard (an engineering guard rail). A dropped unit whose failure correlates
    with a model's losses biases the estimate in that model's favour.

    The model under test is 'x' (CRN: our seat in the candidate arm; seat-swap: model A) and the comparison
    'y' (CRN: our seat in the champion arm; seat-swap: model B). A failed game is charged to the seat in
    ``fault_owner``. An unknown fault is charged to the arm's model in CRN (the arms are otherwise identical,
    so environment faults hit both alike); in seat-swap it cannot be split between A and B, so it counts in
    A's rate (conservative) but not in the A-vs-B count comparison. Faults that ``fault_policy`` 'loss' already
    scores are not counted. Flags: a side's rate over the games it played > ``failure_rate_max``; the counts of
    the two sides differ with a two-sided binomial p < ``failure_asym_p``. Returns (record, reasons, blocks):
    only flags in x's favour block."""
    cnt = {'x': 0, 'y': 0, 'panel': 0, 'unknown': 0}
    scored_known = settings.fault_policy == 'loss'
    for arm in plan.arms:
        for i in range(plan.n_expected):
            spec = plan.spec(arm, i)
            o = outs.get(arm, {}).get(spec.key)
            if o is None or o.ok:
                continue
            if o.fault_owner is not None and scored_known:
                continue                                    # scored as a loss of the faulting seat
            own = o.fault_owner
            if plan.crn:
                if own is not None and own != spec.scored_owner:
                    cnt['panel'] += 1                       # the panel opponent: the same model in both arms
                else:
                    cnt['x' if arm == 'candidate' else 'y'] += 1
            elif own is None:
                cnt['unknown'] += 1
            else:
                cnt['x' if own == spec.scored_owner else 'y'] += 1
    n_games = plan.n_expected * (1 if plan.crn else 2)      # games the side could fault in
    charged = {'x': cnt['x'] + cnt['unknown'], 'y': cnt['y']}
    rate = {k: charged[k] / n_games if n_games else 0.0 for k in ('x', 'y')}
    p = binom_two_sided(cnt['x'], cnt['x'] + cnt['y'])
    reasons: List[str] = []
    blocks = False
    names = ('candidate', 'champion') if plan.crn else ('A', 'B')
    for side, nm in zip(('x', 'y'), names):
        if rate[side] > settings.failure_rate_max + 1e-12:
            reasons.append(f'failures: {charged[side]} {nm} failures ({rate[side]:.4f} > {settings.failure_rate_max}) '
                           f'can bias the estimate')
            blocks = blocks or side == 'x'
    if p < settings.failure_asym_p:
        favour = names[0] if cnt['x'] > cnt['y'] else names[1]
        reasons.append(f'failures: asymmetric failure counts {names[0]} {cnt["x"]} vs {names[1]} {cnt["y"]} '
                       f'(binomial p {p:.2g} < {settings.failure_asym_p}) favour {favour}')
        blocks = blocks or cnt['x'] > cnt['y']
    rec = {'counts': cnt, 'rate': rate, 'asym_p': p, 'policy': settings.fault_policy, 'blocks': blocks}
    return rec, reasons, blocks


def schedule_check(plan: GatePlan, settings: GateSettings) -> List[str]:
    """The plan has the configured size and, for CRN, the configured panel mix
    (exact counts) and held-out kinds. Returns the mismatches (empty = as configured)."""
    out: List[str] = []
    size = settings.size(plan.kind)
    if size and plan.n_expected != size:
        out.append(f'schedule: {plan.n_expected} units != configured {plan.kind} size {size}')
    if plan.crn:
        mix = {k: v for k, v in settings.panel_mix.items() if v > 0}
        want = {k: c for k, c in C.allocate(plan.n_expected, dict(mix)).items() if c}
        have = dict(Counter(u.opp_kind for u in plan.units))
        if have != want:
            out.append(f'schedule: panel counts {dict(sorted(have.items()))} != configured {dict(sorted(want.items()))}')
        bad = [u.unit for u in plan.units if u.holdout != (u.opp_kind in settings.holdout_kinds)]
        if bad:
            out.append(f'schedule: {len(bad)} units with a held-out flag other than gates holdout kinds')
    return out


def _unit_deltas(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]],
                 policy: str = 'drop') -> Dict[int, float]:
    """Per complete unit, the scored quantity of the gate: CRN ``candidate - champion``; seat-swap
    ``mean(A's two games) - 0.5`` (both games of the pair needed)."""
    out: Dict[int, float] = {}
    for u in plan.units:
        res = [_scored(outs.get(arm, {}).get(plan.spec(arm, u.unit).key), plan.spec(arm, u.unit), policy)
               for arm in plan.arms]
        if any(r is None for r in res):
            continue
        out[u.unit] = float(res[0] - res[1]) if plan.crn else float(np.mean(res)) - 0.5
    return out


def expected_min_z(k: int) -> float:
    """E[min of k independent standard normals] (Blom's approximation; -1.42 at k = 8, 0 at k = 1)."""
    if k <= 1:
        return 0.0
    from statistics import NormalDist
    return -NormalDist().inv_cdf((k - 0.375) / (k + 0.25))


def win_condition_of(card_keys: Iterable[str]) -> str:
    """Win-condition label of an opponent deck: the first :data:`WIN_CONDITIONS` group holding one of its
    cards (suffixes -ev1 / -hero dropped), else 'other'."""
    base = set()
    for k in card_keys:
        k = str(k)
        for suf in ('-ev1', '-hero'):
            if k.endswith(suf):
                k = k[:-len(suf)]
        base.add(k)
    for label, cards in WIN_CONDITIONS:
        if base.intersection(cards):
            return label
    return WIN_CONDITION_OTHER


def panel_win_conditions(panel: Sequence[C.CaseRef], cards: Any) -> Dict[str, str]:
    """{case_id: win-condition label} of a gate panel from each opening's real opponent deck
    (``cases.CardKeys.deck_keys``); 'unknown' without one (a synthetic case)."""
    return {r.case_id: (WIN_CONDITION_UNKNOWN if r.opp_deck is None else win_condition_of(cards.deck_keys(r.opp_deck)))
            for r in panel}


def inputs_win_conditions(inputs: Mapping[str, Any], cards: Any) -> Dict[str, str]:
    """:func:`panel_win_conditions` of a plan inputs document's ``panel`` rows."""
    refs = [C.CaseRef(str(r['case_id']), int(r['level']), int(r['our_owner']),
                      None if r.get('opp_deck') is None else tuple(int(x) for x in r['opp_deck']), r.get('split', 'val'))
            for r in inputs.get('panel') or []]
    return panel_win_conditions(refs, cards)


def win_condition_breakdown(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]],
                            labels: Mapping[str, str], policy: str = 'drop') -> Dict[str, Any]:
    """delta_hat by the opponent deck's win condition (DIAGNOSTIC ONLY). Per group: complete units,
    mean, standard error (units are clustered by opening only through the label, so this SE is optimistic) and
    the gap to the overall mean in SE. Caveat recorded with it: with k groups the worst group sits about
    ``expected_min_z(k)`` SE below the overall under the null (k = 8: -1.42 SE), never a gate rule. The empirical
    null is an A/A block on the same panel scored with the same labels (``evaluate_aa``, or ``gate.py evaluate
    --inputs`` on an ``aa`` block); without one, the analytic ``null_expected_min_z`` is the reference."""
    d = _unit_deltas(plan, outs, policy)
    units = {u.unit: u for u in plan.units}
    groups: Dict[str, List[float]] = {}
    for i, x in d.items():
        groups.setdefault(str(labels.get(units[i].case_id, WIN_CONDITION_UNKNOWN)), []).append(x)
    allx = list(d.values())
    overall = _mean(allx)
    rows: Dict[str, Any] = {}
    for g in sorted(groups):
        xs = np.asarray(groups[g], dtype=np.float64)
        se = float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None
        mean = float(xs.mean())
        rows[g] = {'units': int(len(xs)), 'delta_hat': mean, 'se': se,
                   'z_vs_overall': None if not se or overall is None else (mean - overall) / se}
    k = len(rows)
    return {'quantity': 'candidate - champion' if plan.crn else 'score(A) - 0.5', 'overall_delta_hat': overall,
            'units': len(allx), 'groups': rows, 'k': k, 'null_expected_min_z': expected_min_z(k),
            'blocking': False,
            'caveat': f'diagnostic only: with {k} groups the worst one is expected about '
                      f'{abs(expected_min_z(k)):.2f} SE below the overall under the null (analytic); the empirical '
                      'A/A reference is an aa block on the same panel (evaluate_aa)'}


def evaluate(plan: GatePlan, store: Optional[GateStore], settings: GateSettings = GateSettings(), *,
             outcomes: Optional[Mapping[str, Mapping[str, GameOutcome]]] = None,
             win_conditions: Optional[Mapping[str, str]] = None) -> GateReport:
    """Score a gate from its stored chunks (or given ``outcomes`` {arm: {key: outcome}}).

    CRN: stats.crn_gate (g1), stats.screen_s (screen, aa) over (unit, opening, candidate, champion, held-out);
    seat-swap: stats.seat_swap_gate with the opening as cluster. Completion is over the scheduled units. A
    blocking gate passes only if the stats rule, the failure guard and (``check_schedule``) the schedule hold.
    ``win_conditions`` ({case_id: label}, :func:`panel_win_conditions`) adds the diagnostic
    ``by_win_condition`` table; it never changes ``passed``. Canary blocks go to
    :func:`evaluate_canary`."""
    if plan.canary:
        raise GateError(f'{plan.block}: a canary block is scored by evaluate_canary, never as a gate')
    if outcomes is None and store is None:
        raise GateError('evaluate needs a store or the outcomes')
    outs = outcomes if outcomes is not None else {a: store.outcomes(plan, a) for a in plan.arms}
    for arm, d in outs.items():
        if arm not in plan.arms:
            raise GateError(f'outcomes for unknown arm {arm!r}')
        want = {game_key(plan.block, arm, i) for i in range(plan.n_expected)}
        extra = set(d) - want
        if extra:
            raise GateError(f'outcomes outside the schedule: {sorted(extra)[:3]}')
    seed = boot_seed(plan.block)
    pol = settings.fault_policy
    kw = dict(method=settings.method, n_boot=settings.n_boot, seed=seed, min_completion=settings.min_completion,
              min_clusters=settings.min_clusters)
    rho = None
    subpanel: Dict[str, Any] = {}
    sprt = None
    if plan.crn:
        recs = _crn_rows(plan, outs, pol)
        if plan.kind == 'g1':
            res = S.crn_gate(recs, n_expected=plan.n_expected, alpha=settings.alpha, holdout_min=settings.holdout_min,
                             **kw)
        else:
            res = S.screen_s(recs, n_expected=plan.n_expected, alpha=settings.alpha,
                             alarm_delta=settings.rollback_delta, **kw)
        comp = [r for r in recs if r.candidate is not None and r.champion is not None]
        if len(comp) > 2:
            c = np.array([r.candidate for r in comp])
            h = np.array([r.champion for r in comp])
            if c.std() > 0 and h.std() > 0:
                rho = float(np.corrcoef(c, h)[0, 1])
        units = {u.unit: u for u in plan.units}
        for kind in sorted({u.opp_kind for u in plan.units}):
            d = [r.candidate - r.champion for r in comp if units[r.triple_id].opp_kind == kind]
            subpanel[kind] = {'pairs': len(d), 'delta_hat': _mean(d),
                              'holdout': kind in {u.opp_kind for u in plan.units if u.holdout}}
        if settings.sprt and plan.kind == 'g1':
            diffs = [r.candidate - r.champion for r in comp]       # schedule (unit) order
            sprt = S.sprt_g1(diffs, delta1=settings.sprt_h1, alpha=settings.alpha, cap=plan.n_expected).as_dict()
        n_complete = res.n_pairs
        passed = bool(res.passed) if plan.kind == 'g1' else False
        alarm = bool(res.alarm) if plan.kind == 'screen' else False      # rollback input: screens and G1s
        if plan.kind == 'g1':
            alarm = bool(res.valid and res.n_pairs and res.delta_hat <= settings.rollback_delta)
    else:
        res = S.seat_swap_gate(_swap_rows(plan, outs, pol), n_expected=plan.n_expected, level=settings.swap_level,
                               cluster=True, **kw)
        n_complete = res.n_pairs
        passed = bool(res.passed) if plan.kind in BLOCKING_KINDS else False
        alarm = False
    reasons = list(res.reasons)
    fc, fc_reasons, fc_blocks = failure_check(plan, outs, settings)
    reasons += fc_reasons
    if fc_blocks:
        passed = False
    if settings.check_schedule and plan.kind in BLOCKING_KINDS:
        sc = schedule_check(plan, settings)
        reasons += sc
        if sc:
            passed = False
    players = {k: {'name': p.name, 'digest': p.digest} for k, p in sorted(plan.players.items())}
    return GateReport(schema=REPORT_SCHEMA, kind=plan.kind, block=plan.block, block_digest=plan.block_digest(),
                      env_digest=plan.env_digest, players=players, n_expected=plan.n_expected,
                      n_complete=int(n_complete), completion=float(res.completion), valid=bool(res.valid),
                      passed=passed, blocking=plan.kind in BLOCKING_KINDS,
                      replay_required=bool(res.completion < settings.min_completion), alarm=alarm,
                      stats=_finite(res.as_dict()), rho=rho, subpanel=subpanel, sprt=sprt, failure_check=_finite(fc),
                      diagnostics=_finite(diagnostics(plan, outs)), reasons=reasons,
                      by_win_condition={} if win_conditions is None else
                      _finite(win_condition_breakdown(plan, outs, win_conditions, pol)))


def truncated_plan(plan: GatePlan, n: int) -> GatePlan:
    """``plan`` with its first ``n`` units only (the same block name and seeds, so the game keys are the same)."""
    if not 0 < int(n) <= plan.n_expected:
        raise GateError(f'{plan.block}: cannot truncate {plan.n_expected} units to {n}')
    return replace(plan, units=tuple(plan.units[:int(n)]))


def prefix_settings(plan: GatePlan, settings: GateSettings) -> Optional[GateSettings]:
    """The settings a PREFIX of ``plan`` is scored with (early stops). The configured-schedule check belongs to the
    whole plan: a prefix has fewer units than the configured size by construction, and the panel counts of a
    prefix of the shuffled schedule are not the exact configured ones. So the whole plan is checked here (None when
    it is not as configured: no early stop, the full gate reports it) and the prefix is scored without the check
    (scored WITH the check, a prefix would never pass and the early accept would never fire)."""
    if settings.check_schedule and plan.kind in BLOCKING_KINDS and schedule_check(plan, settings):
        return None
    return replace(settings, check_schedule=False)


def early_accept(plan: GatePlan, store: GateStore, settings: GateSettings, n: int, *,
                 win_conditions: Optional[Mapping[str, str]] = None) -> Optional[GateReport]:
    """The report of a G1 over its first ``n`` units when it may stop there (``settings.sprt_stop_alpha``),
    else None. All of: a CRN g1 with fewer than all units played; every game of both arms of these ``n`` units is
    stored; the fixed-sample rule passes on them (:func:`evaluate` of the truncated schedule: valid, lower bound,
    held-out sub-panel, failure guard, schedule check); the truncated SPRT on the pair differences in schedule
    order has decided H1 at level ``sprt_stop_alpha``. The report's ``sprt`` record carries the stop
    (``early_stop``, ``played``, ``scheduled``, ``alpha``). Nothing is ever FAILED early: without an accept the
    gate goes on and the full fixed-sample rule decides, exactly as without this option."""
    a = settings.sprt_stop_alpha
    if a is None or plan.kind != 'g1' or not plan.crn or plan.canary or not 0 < int(n) < plan.n_expected:
        return None
    n = int(n)
    ps = prefix_settings(plan, settings)
    if ps is None:
        return None
    sub = truncated_plan(plan, n)
    outs: Dict[str, Dict[str, GameOutcome]] = {}
    for arm in plan.arms:
        full = store.outcomes(plan, arm)
        keys = [game_key(plan.block, arm, i) for i in range(n)]
        if any(k not in full for k in keys):
            return None
        outs[arm] = {k: full[k] for k in keys}
    rep = evaluate(sub, None, ps, outcomes=outs, win_conditions=win_conditions)
    if not (rep.valid and rep.passed) or rep.replay_required:
        return None
    recs = _crn_rows(sub, outs, settings.fault_policy)
    diffs = [r.candidate - r.champion for r in recs if r.candidate is not None and r.champion is not None]
    sp = S.sprt_g1(diffs, delta1=settings.sprt_h1, alpha=float(a), cap=plan.n_expected)
    if sp.decision != 'H1':
        return None
    return replace(rep, sprt=dict(sp.as_dict(), early_stop=True, played=n, scheduled=plan.n_expected,
                                  alpha=float(a)))


# ------------------------------------------------------------------ the head-to-head as the deciding gate
def swap_pair_scores(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]], policy: str = 'drop'
                     ) -> List[float]:
    """Model A's pair scores of a seat-swap block in schedule (unit) order, complete pairs only."""
    pairs, _, _ = S.seat_swap_collect(_swap_rows(plan, outs, policy))
    return [float(pairs[u.unit]) for u in plan.units if u.unit in pairs]


def early_swap(plan: GatePlan, store: GateStore, settings: GateSettings, n: int, *,
               win_conditions: Optional[Mapping[str, str]] = None) -> Optional[GateReport]:
    """The report of a G1c over its first ``n`` units when it may stop there (``settings.h2h_stop_alpha``),
    else None. Both arms of these ``n`` units must be stored and the truncated block valid. A truncated SPRT on
    the pair scores in schedule order (H0 score 0.5, H1 0.5 + ``h2h_h1``, level ``h2h_stop_alpha`` on both sides)
    decides: 'H1' stops and PASSES only if the fixed-sample rule also passes on these pairs (lower bound above 0.5,
    failure guard); 'H0' stops and FAILS (the candidate is not better by the H1 margin: playing on could at most
    find a smaller edge than the gate is sized for). The report's ``sprt`` record carries the stop."""
    a = settings.h2h_stop_alpha
    if a is None or plan.kind != 'g1c' or plan.crn or plan.canary or not 0 < int(n) < plan.n_expected:
        return None
    n = int(n)
    ps = prefix_settings(plan, settings)
    if ps is None:
        return None
    sub = truncated_plan(plan, n)
    outs: Dict[str, Dict[str, GameOutcome]] = {}
    for arm in plan.arms:
        full = store.outcomes(plan, arm)
        keys = [game_key(plan.block, arm, i) for i in range(n)]
        if any(k not in full for k in keys):
            return None
        outs[arm] = {k: full[k] for k in keys}
    rep = evaluate(sub, None, ps, outcomes=outs, win_conditions=win_conditions)
    if not rep.valid or rep.replay_required:
        return None
    scores = swap_pair_scores(sub, outs, settings.fault_policy)
    sp = S.sprt_g1([x - 0.5 for x in scores], delta1=settings.h2h_h1, alpha=float(a), beta=float(a),
                   cap=plan.n_expected)
    rec = dict(sp.as_dict(), early_stop=True, played=n, scheduled=plan.n_expected, alpha=float(a),
               h1=float(settings.h2h_h1))
    if sp.decision == 'H1' and rep.passed:
        return replace(rep, sprt=rec)
    if sp.decision == 'H0':
        return replace(rep, passed=False, sprt=rec,
                       reasons=list(rep.reasons) + [f'stopped at {n} of {plan.n_expected} pairs: not better by '
                                                    f'{settings.h2h_h1 * 100:.0f} pp (sequential test, level {a})'])
    return None


def h2h_monitor_delta(rep: GateReport) -> float:
    """What the monitor's gate rules see of a head-to-head (``Monitor.observe_gate`` as a G1): ``score - 0.5``.
    A block stopped short as a FAIL has few pairs and an estimate biased low (it stopped because it was low), and
    two results at or below the rollback threshold in a row roll the run back to the champion: such a stop counts
    only with what its interval supports, the upper end of the normal interval (kept below zero, it is still a
    failed gate). NaN for an invalid block."""
    score = rep.stats.get('score')
    if not rep.valid or score is None:
        return float('nan')
    d = float(score) - 0.5
    stopped_short = bool((rep.sprt or {}).get('early_stop')) and not rep.passed
    hi = rep.stats.get('hi_normal')
    if stopped_short and d < 0.0 and hi is not None:
        return max(d, min(float(hi) - 0.5, -1e-4))
    return d


def swap_side_results(plan: GatePlan, outs: Mapping[str, Mapping[str, GameOutcome]], policy: str = 'drop'
                      ) -> Dict[str, List[float]]:
    """Model A's scored results of a seat-swap block by the deck it held: 'ours' = A on the case's ``our_owner``
    (our deck against B holding the opening's other deck), 'other' = A on the other deck. Games not stored,
    dropped or failed (under the fault policy) are left out."""
    out: Dict[str, List[float]] = {'ours': [], 'other': []}
    for u in plan.units:
        for arm in SWAP_ARMS:
            spec = plan.spec(arm, u.unit)
            r = _scored(outs.get(arm, {}).get(spec.key), spec, policy)
            if r is not None:
                out['ours' if int(spec.scored_owner) == int(u.our_owner) else 'other'].append(float(r))
    return out


def our_deck_guard(main: Mapping[str, Sequence[float]], base: Mapping[str, Sequence[float]], *,
                   alpha: float = 0.05) -> Dict[str, Any]:
    """Guard of a head-to-head promotion. ``main`` = :func:`swap_side_results` of candidate (A) against
    champion (B); ``base`` = the same of champion against champion. Both 'ours' lists are results of a model
    holding OUR deck against the champion holding the other deck, so their means are comparable: the guard fails
    when the candidate's is below the champion's at the one-sided level ``alpha`` (unpaired z test; other openings
    and seeds on the two sides). With a few hundred games a side it catches a drop of several points, not less.
    A side without games fails the guard."""
    a, b = [float(x) for x in main.get('ours', ())], [float(x) for x in base.get('ours', ())]
    out: Dict[str, Any] = {'n_candidate': len(a), 'n_champion': len(b), 'alpha': float(alpha),
                           'candidate_ours': _mean(a), 'champion_ours': _mean(b),
                           'candidate_other': _mean([float(x) for x in main.get('other', ())]),
                           'champion_other': _mean([float(x) for x in base.get('other', ())])}
    if len(a) < 2 or len(b) < 2:
        return dict(out, ok=False, reason='no games on one side')
    va = max(float(np.var(a, ddof=1)), 1e-4)
    vb = max(float(np.var(b, ddof=1)), 1e-4)
    se = math.sqrt(va / len(a) + vb / len(b))
    diff = out['candidate_ours'] - out['champion_ours']
    z = diff / se
    limit = -S.z_quantile(1.0 - float(alpha))
    return dict(out, diff=diff, se=se, z=z, z_limit=limit, ok=bool(z >= limit))


def evaluate_canary(plan: GatePlan, store: Optional[GateStore], *,
                    outcomes: Optional[Mapping[str, Mapping[str, GameOutcome]]] = None,
                    alarm_lb: float = CANARY_ALARM_LB) -> Dict[str, Any]:
    """Score a canary block (DIAGNOSTIC ONLY, never promoting, never seen by the monitor): the canary's
    score on our seat with its Wilson interval, by panel opponent, the per-arm diagnostics (plays per minute,
    refusals ...) and ``alarm`` = the Wilson lower bound is above ``alarm_lb`` (an all-wait or random policy
    that wins that often means the harness or the reward reads results wrongly)."""
    if not plan.canary:
        raise GateError(f'{plan.block} is not a canary block')
    if outcomes is None and store is None:
        raise GateError('evaluate_canary needs a store or the outcomes')
    outs = outcomes if outcomes is not None else {a: store.outcomes(plan, a) for a in plan.arms}
    dg = diagnostics(plan, outs)['canary']
    lo, hi = dg['wilson']
    kind = canary_kind(plan.players['canary'])
    complete = dg['native'] / plan.n_expected if plan.n_expected else 0.0
    return _finite({'schema': REPORT_SCHEMA, 'kind': CANARY_KIND, 'canary': kind, 'block': plan.block,
                    'block_digest': plan.block_digest(), 'env_digest': plan.env_digest,
                    'n_expected': plan.n_expected, 'native': dg['native'], 'completion': complete,
                    'score': dg['score'], 'wilson': [lo, hi], 'wins': dg['wins'], 'draws': dg['draws'],
                    'losses': dg['losses'], 'by_opponent': dg.get('by_opponent', {}),
                    'alarm': bool(dg['native'] > 0 and lo is not None and lo > float(alarm_lb)),
                    'alarm_lb': float(alarm_lb), 'blocking': False, 'promoting': False, 'diagnostics': dg})


# ------------------------------------------------------------------ gate-style A/A
AA_SCHEMA = 'ppo-v7-aa-report.1'
AA_ALPHA = 0.025                          # per side: the A/A delta_hat's two-sided 95 % cluster-bootstrap interval
AA_DELTAS = (0.03, 0.04, 0.05, 0.08)      # the true effects of the power table (stats.g1_power_table)
AA_POWER = 0.80                           # reported: pairs a G1 would need for this pass rate (judgement value)
AA_PLAN_RHO = (0.1, 0.5)                  # the rho range a G1 size is planned for (example values)
AA_CALIBRATION_MIN = 0.9                  # se_boot / exact null SE below this: the bootstrap is too narrow (judgement)
AA_NULL_QUANTILES = (0.01, 0.025, 0.05, 0.5)
_AA_NULL_CHUNK = 1000                     # sign-flip draws per chunk (fixed: a seed always gives the same draws)


def aa_null_seed(block: str) -> int:
    """Seed of an A/A block's sign-flip null (fixed per block, independent of its bootstrap seed)."""
    return C.stable_hash(GATE_NAMESPACE, 'aa-null', _check_block(block)) & 0xFFFFFFFF


def aa_null(plan: GatePlan, deltas: Mapping[int, float], labels: Mapping[str, str], n_draws: int,
            seed: int) -> Dict[str, Any]:
    """The empirical null of an A/A block by per-unit sign flips.

    In a gate-style A/A both arms are the same weights with the same settings, and only our policy random numbers
    differ between them (independent draws per arm). The two games of a unit are therefore exchangeable, so the
    unit difference ``candidate - champion`` is symmetric around 0, given every triple, and independent across
    units. Flipping the sign of each unit difference at random gives the exact null distribution of every
    statistic of the block (no normal approximation, no clustering assumption). Returned:

    * ``se_delta_hat``: the null SE of delta_hat, the reference the cluster bootstrap SE is compared with;
    * ``worst_z_quantiles``: quantiles of the worst win-condition group's z (``win_condition_breakdown``:
      (group mean - overall) / group SE) under the null, the empirical counterpart of ``expected_min_z(k)``;
    * per group: quantiles of the gap (group mean - overall) under the null, i.e. how far a group of that size
      moves by chance.
    """
    units = {u.unit: u for u in plan.units}
    keys = sorted(deltas)
    x = np.asarray([float(deltas[i]) for i in keys], dtype=np.float64)
    n = int(x.size)
    lab = [str(labels.get(units[i].case_id, WIN_CONDITION_UNKNOWN)) for i in keys]
    groups = sorted(set(lab))
    gidx = {g: np.asarray([j for j, v in enumerate(lab) if v == g], dtype=np.int64) for g in groups}
    zgroups = [g for g in groups if gidx[g].size > 1]
    over = np.empty(n_draws)
    worst = np.full(n_draws, np.nan)
    gaps = {g: np.empty(n_draws) for g in groups}
    rng = np.random.default_rng(int(seed))
    for s in range(0, n_draws, _AA_NULL_CHUNK):
        m = min(_AA_NULL_CHUNK, n_draws - s)
        y = (rng.integers(0, 2, size=(m, n)) * 2 - 1) * x[None, :]
        o = y.mean(1)
        over[s:s + m] = o
        zs = []
        for g in groups:
            yg = y[:, gidx[g]]
            mg = yg.mean(1)
            gaps[g][s:s + m] = mg - o
            if g in zgroups:
                se = yg.std(1, ddof=1) / math.sqrt(yg.shape[1])
                with np.errstate(divide='ignore', invalid='ignore'):
                    zs.append(np.where(se > 0, (mg - o) / np.where(se > 0, se, 1.0), 0.0))
        if zs:
            worst[s:s + m] = np.min(np.stack(zs), axis=0)
    qs = AA_NULL_QUANTILES
    q = lambda a, p: float(np.quantile(a, p)) if a.size else None                     # noqa: E731
    return {'method': 'per-unit sign flip (exact under the A/A null: the two games of a unit are exchangeable)',
            'draws': int(n_draws), 'seed': int(seed), 'units': n,
            'se_delta_hat': float(over.std(ddof=1)) if n_draws > 1 else None,
            'delta_hat_quantiles': {str(p): q(over, p) for p in qs + (0.975, 0.99)},
            'k': len(zgroups), 'analytic_expected_min_z': expected_min_z(len(zgroups)),
            'worst_z_quantiles': {str(p): (q(worst, p) if zgroups else None) for p in qs},
            'groups': {g: {'units': int(gidx[g].size), 'gap_q025': q(gaps[g], 0.025), 'gap_q975': q(gaps[g], 0.975),
                           'abs_gap_q95': q(np.abs(gaps[g]), 0.95)} for g in groups}}


def aa_sizing(rho: Optional[float], se_cluster: Optional[float], n_pairs: int, *, g1_pairs: int = S.G1_PAIRS,
              g1_alpha: float = 0.1, deltas: Sequence[float] = AA_DELTAS, power: float = AA_POWER) -> Dict[str, Any]:
    """Gate sizing from an A/A block (the power table). Two readings, both reported:

    * ``measured``: the A/A's cluster-robust SE of one pair difference (``se_cluster * sqrt(n_pairs)``, which keeps
      the draws and the opening clustering) projected to ``g1_pairs`` pairs: the delta_hat a G1 needs, its pass rate
      at each true delta and the pairs needed for ``power``;
    * ``from_rho``: the plan's analytic form SE = sqrt(0.5 (1 - rho) / n) (``stats.g1_power_table``; it assumes
      win/loss outcomes of variance 1/4), with ``rho_in_plan_range`` against :data:`AA_PLAN_RHO`.
    """
    out: Dict[str, Any] = {'g1_pairs': int(g1_pairs), 'g1_alpha_one_sided': float(g1_alpha), 'power': float(power),
                           'rho': rho, 'plan_rho_range': list(AA_PLAN_RHO),
                           'rho_in_plan_range': None if rho is None else bool(AA_PLAN_RHO[0] <= rho <= AA_PLAN_RHO[1])}
    za, zb = S.z_quantile(1.0 - g1_alpha), S.z_quantile(power)
    if se_cluster is not None and math.isfinite(se_cluster) and se_cluster > 0 and n_pairs > 1:
        sd = float(se_cluster) * math.sqrt(n_pairs)
        se = sd / math.sqrt(g1_pairs)
        out['measured'] = {'sd_pair': sd, 'se_g1': se, 'required_delta': S.required_delta(se, g1_alpha),
                           'pass_prob': {str(x): S.pass_prob(x, se, g1_alpha) for x in deltas},
                           'pairs_for_power': {str(x): int(math.ceil(((za + zb) * sd / x) ** 2)) for x in deltas}}
    else:
        out['measured'] = None
    if rho is not None and math.isfinite(rho) and -1.0 < rho < 1.0:
        row = S.g1_power_table(rhos=(float(rho),), n=g1_pairs, deltas=tuple(deltas), alpha=g1_alpha)[0]
        row['pass'] = {str(k): v for k, v in row['pass'].items()}
        row['pairs_for_power'] = {str(x): int(math.ceil(((za + zb) * S.crn_sd(rho) / x) ** 2)) for x in deltas}
        out['from_rho'] = row
    else:
        out['from_rho'] = None
    return out


def evaluate_aa(plan: GatePlan, store: Optional[GateStore], settings: GateSettings = GateSettings(), *,
                outcomes: Optional[Mapping[str, Mapping[str, GameOutcome]]] = None,
                win_conditions: Optional[Mapping[str, str]] = None, g1_pairs: int = S.G1_PAIRS,
                g1_alpha: float = 0.1, n_null: Optional[int] = None, deltas: Sequence[float] = AA_DELTAS,
                power: float = AA_POWER) -> Dict[str, Any]:
    """Score a gate-style A/A block (an A/A null test: it passes when delta_hat's interval covers 0).

    The block must be a ``crn_plan('aa', ...)`` plan: both arms the same weights and settings row, independent
    policy seeds per arm (``policy_crn`` False), the same (opening, seed block, panel opponent) triples. Returns:

    * ``gate``: :func:`evaluate` with ``alpha`` = :data:`AA_ALPHA` (so ``lb_boot`` / ``ub_boot`` are the two-sided 95 %
      cluster-bootstrap interval) and the ``by_win_condition`` table when ``win_conditions`` is given;
    * ``delta_hat``, ``ci`` and ``passed`` = the gate is valid (completion, openings) AND the interval covers 0 AND
      ``failure_ok`` (no :func:`failure_check` flag on EITHER arm: in an A/A both arms are the same model, so a failure
      rate above ``failure_rate_max`` on either arm or asymmetric failure counts in either direction means the
      harness, not the policy, is under test; G1 blocks only on flags in the candidate's favour);
    * ``rho`` (the arms' outcome correlation over complete units) and ``sizing`` (:func:`aa_sizing`);
    * ``null`` (:func:`aa_null`, ``n_null`` draws, default ``settings.n_boot``) and ``calibration``: the cluster
      bootstrap SE over the exact null SE (``ok`` when >= :data:`AA_CALIBRATION_MIN`; a smaller ratio means the
      gate's bootstrap interval is too narrow);
    * ``win_condition_null``: the A/A's own group table with the null quantiles of each group and of the worst z,
      the empirical reference of the G1 win-condition breakdown.

    Record only: nothing here promotes, alarms or reaches the monitor."""
    if plan.kind != 'aa' or plan.policy_crn:
        raise GateError(f'{plan.block}: evaluate_aa takes a gate-style A/A block (crn_plan kind "aa", independent '
                        'policy seeds); a setting measurement (setting_crn_plan) is scored by evaluate')
    a, b = plan.players['candidate'], plan.players['champion']
    if a.digest != b.digest or settings_diff(a, b):
        raise GateError(f'{plan.block}: the A/A arms must be the same weights with the same settings')
    gs = replace(settings, alpha=AA_ALPHA, check_schedule=False)
    outs = outcomes if outcomes is not None else {arm: store.outcomes(plan, arm) for arm in plan.arms}
    rep = evaluate(plan, None, gs, outcomes=outs, win_conditions=win_conditions)
    st = rep.stats
    pick = lambda x, y: st.get(x) if st.get(x) is not None else st.get(y)                # noqa: E731
    lo, hi = pick('lb_boot', 'lb_analytic'), pick('ub_boot', 'ub_analytic')
    covers = bool(lo is not None and hi is not None and lo <= 0.0 <= hi)
    d = _unit_deltas(plan, outs, gs.fault_policy)
    nn = int(gs.n_boot if n_null is None else n_null)
    null = aa_null(plan, d, dict(win_conditions or {}), nn, aa_null_seed(plan.block)) if d and nn > 1 else None
    se_null = None if null is None else null['se_delta_hat']
    se_boot = st.get('se_boot')
    ratio = (float(se_boot) / se_null) if se_boot is not None and se_null else None
    reasons = list(rep.reasons)                    # already holds the failure_check reasons (evaluate)
    if rep.valid and not covers:
        reasons.append(f'the A/A interval [{lo:.4f}, {hi:.4f}] does not cover 0')
    _fc, fc_reasons, _fc_blocks = failure_check(plan, outs, gs)
    failure_ok = not fc_reasons
    wc = rep.by_win_condition or {}
    zs = [v.get('z_vs_overall') for v in (wc.get('groups') or {}).values() if v.get('z_vs_overall') is not None]
    wcn = None
    if wc and null is not None:
        wcn = {'observed': wc, 'observed_worst_z': min(zs) if zs else None,
               'null_worst_z_quantiles': null['worst_z_quantiles'], 'analytic_expected_min_z':
               null['analytic_expected_min_z'], 'null_groups': null['groups'],
               'use': 'reference for the G1 / screen by_win_condition table (diagnostic only): a G1 group is '
                      'unusual when its z is below the null worst-z quantile of the same k'}
    return _finite({
        'schema': AA_SCHEMA, 'kind': 'aa', 'block': plan.block, 'block_digest': plan.block_digest(),
        'env_digest': plan.env_digest, 'player': a.to_dict(), 'n_expected': rep.n_expected,
        'n_complete': rep.n_complete, 'completion': rep.completion, 'valid': rep.valid,
        'replay_required': rep.replay_required, 'delta_hat': rep.delta_hat, 'ci': [lo, hi],
        'ci_level_two_sided': 1.0 - 2.0 * AA_ALPHA, 'ci_method': 'cluster bootstrap' if st.get('lb_boot') is not None
        else 'analytic (cluster-robust)', 'covers_zero': covers, 'failure_ok': failure_ok,
        'failure_flags': list(fc_reasons), 'passed': bool(rep.valid and covers and failure_ok),
        'rule': 'pass: the gate is valid (completion >= min_completion, enough openings), the two-sided 95 % '
                'interval of delta_hat covers 0 and the failure guard raises no flag on either arm',
        'rho': rep.rho, 'se': {'iid': st.get('se_iid'), 'cluster': st.get('se_cluster'), 'boot': se_boot,
                               'null_exact': se_null},
        'calibration': {'boot_over_null': ratio, 'min': AA_CALIBRATION_MIN,
                        'ok': None if ratio is None else bool(ratio >= AA_CALIBRATION_MIN)},
        'sizing': aa_sizing(rep.rho, st.get('se_cluster'), int(rep.n_complete), g1_pairs=g1_pairs,
                            g1_alpha=g1_alpha, deltas=deltas, power=power),
        'null': null, 'win_condition_null': wcn, 'reasons': reasons, 'gate': rep.as_dict(),
        'blocking': False, 'promoting': False})


def write_aa_report(report: Mapping[str, Any], store: GateStore) -> Path:
    p = store.block_dir(str(report['block'])) / 'report-aa.json'
    atomic_write_json(p, dict(report))
    return p


def write_canary_report(report: Mapping[str, Any], store: GateStore) -> Path:
    p = store.block_dir(str(report['block'])) / 'report-canary.json'
    atomic_write_json(p, dict(report))
    return p


def write_report(report: GateReport, store: GateStore) -> Path:
    cand = report.players.get('candidate', report.players.get('a', {})).get('digest', 'unknown')
    p = store.block_dir(report.block) / f'report-{cand[:16]}.json'
    atomic_write_json(p, report.as_dict())
    return p


def promotion_decision(g1: GateReport, second: Optional[GateReport], h1_reading: str = 'crn_then_seatswap') -> bool:
    """Promote iff G1 passed and the second gate passed.

    'crn_then_seatswap' (the default): ``second`` is the G1c of candidate (a) vs champion (b).
    'crn_only' (promotion on two CRN gates): ``second`` is a second G1 of the same candidate and
    champion on its own fresh block (``block_name('g1', ..., confirm=True)``); one passing G1 alone never promotes.
    Both reports must come from the same environment (env_digest)."""
    if g1.kind != 'g1':
        raise GateError('first report must be a G1')
    if h1_reading not in ('crn_then_seatswap', 'crn_only'):
        raise GateError(f'unknown h1_reading {h1_reading!r}')
    if not (g1.valid and g1.passed):
        return False
    if h1_reading == 'crn_only':
        if second is None or second.kind != 'g1':
            raise GateError('crn_only needs a second G1 report (the confirmation CRN gate)')
        if second.block == g1.block or second.block_digest == g1.block_digest:
            raise GateError('the confirmation G1 must be played on its own fresh seed block')
        if second.players['candidate']['digest'] != g1.players['candidate']['digest'] \
                or second.players['champion']['digest'] != g1.players['champion']['digest']:
            raise GateError('the confirmation G1 must be the same candidate vs the same champion')
    else:
        if second is None or second.kind != 'g1c':
            raise GateError('crn_then_seatswap needs the G1c report')
        if second.players['a']['digest'] != g1.players['candidate']['digest'] \
                or second.players['b']['digest'] != g1.players['champion']['digest']:
            raise GateError('G1c must be candidate (a) vs champion (b) of the same G1')
    if second.env_digest != g1.env_digest:
        raise GateError('the two promotion gates were played in different environments')
    return bool(second.valid and second.passed)


def g2_trend(reports: Sequence[GateReport]) -> Dict[str, Any]:
    """The G2 trend (record only): score and CI per promotion, the least-squares slope per
    promotion over the valid ones, and ``latest`` (the last valid G2: score, CI low)."""
    rows = []
    for r in reports:
        if r.kind != 'g2':
            raise GateError('g2_trend takes G2 reports')
        hi = r.stats.get('hi_boot')
        rows.append({'block': r.block, 'candidate': r.players['a']['digest'], 'valid': r.valid,
                     'score': r.stats.get('score'), 'lo': r.stats.get('lo'),
                     'hi': hi if hi is not None else r.stats.get('hi_normal')})
    ys = [(i, x['score']) for i, x in enumerate(rows) if x['valid'] and x['score'] is not None]
    slope = None
    if len(ys) >= 2:
        xi = np.array([i for i, _ in ys], dtype=np.float64)
        yi = np.array([y for _, y in ys], dtype=np.float64)
        slope = float(np.polyfit(xi, yi, 1)[0])
    latest = rows[ys[-1][0]] if ys else None
    return {'points': rows, 'slope_per_promotion': slope, 'latest': latest, 'blocking': False}


def g2_pool(items: Sequence[Tuple[GatePlan, Mapping[str, Mapping[str, GameOutcome]]]],
            settings: GateSettings = GateSettings()) -> Dict[str, Any]:
    """The cumulative G2 trend (G2 pooled since the last restart): every complete seat-swapped pair of the given
    G2 plays ((plan, outcomes) of one environment, normally every valid G2 since the last restart) pooled into
    ONE seat-swap
    statistic. All G2s use the fixed block, so the same openings repeat across them: the pairs are clustered by
    opening over the whole pool (``stats.seat_swap_arrays`` with ``cluster_ids`` = the case id: the CI covers
    the correlation of repeated openings and seeds). Completion is over the summed schedule; faulted pairs are
    dropped whole (``settings.fault_policy``), as in each G2. The bootstrap seed is fixed by the
    pooled blocks and candidates. Returns score, lo / hi (two-sided ``settings.swap_level``), se, n_pairs,
    n_expected, n_clusters, valid, reports and the members (block, candidate)."""
    if not items:
        raise GateError('g2_pool needs at least one G2')
    envs = {plan.env_digest for plan, _ in items}
    if len(envs) != 1:
        raise GateError(f'g2_pool: G2s from {len(envs)} environments cannot be pooled (a changed environment '
                        'is a restart)')
    scores: List[float] = []
    clusters: List[Any] = []
    n_exp = 0
    members = []
    for plan, outs in items:
        if plan.kind != 'g2':
            raise GateError(f'g2_pool takes G2 plans, got {plan.kind!r} ({plan.block})')
        pairs, cl, _ = S.seat_swap_collect(_swap_rows(plan, outs, settings.fault_policy))
        for k in sorted(pairs, key=str):
            scores.append(float(pairs[k]))
            clusters.append(str(cl[k]))
        n_exp += int(plan.n_expected)
        members.append({'block': plan.block, 'candidate': plan.players['a'].digest, 'n_pairs': len(pairs),
                        'n_expected': int(plan.n_expected)})
    seed = C.stable_hash(GATE_NAMESPACE, 'g2-pool', *[f'{m["block"]}:{m["candidate"]}' for m in members]) & 0xFFFFFFFF
    res = S.seat_swap_arrays(np.array(scores, dtype=np.float64), n_expected=n_exp, cluster_ids=clusters,
                             method=settings.method, level=settings.swap_level, n_boot=settings.n_boot, seed=seed,
                             min_completion=settings.min_completion, min_clusters=settings.min_clusters)
    hi = res.hi_boot if res.hi_boot is not None else res.hi_normal
    return _finite({'mode': 'pooled_since_restart', 'reports': len(items), 'env_digest': next(iter(envs)),
                    'score': res.score, 'lo': res.lo, 'hi': hi, 'se': res.se, 'n_pairs': res.n_pairs,
                    'n_expected': res.n_expected, 'n_clusters': res.n_clusters, 'completion': res.completion,
                    'valid': bool(res.valid), 'reasons': list(res.reasons), 'members': members,
                    'blocking': False})


def g2_pool_from_store(store: GateStore, records: Sequence[Mapping[str, Any]],
                       settings: GateSettings = GateSettings()) -> Dict[str, Any]:
    """:func:`g2_pool` of stored G2s: each record names its ``block`` and ``candidate`` digest (the block
    driver's ``state['g2']`` rows); the plan is read from ``<block>/plan-<candidate[:16]>.json`` and the outcomes
    from the chunks."""
    items = []
    for r in records:
        path = store.block_dir(str(r['block'])) / f'plan-{str(r["candidate"])[:16]}.json'
        plan = GatePlan.from_dict(_read_json(path))
        if plan.players['a'].digest != str(r['candidate']):
            raise GateError(f'{path}: candidate {plan.players["a"].digest[:12]} != {str(r["candidate"])[:12]}')
        items.append((plan, {a: store.outcomes(plan, a) for a in plan.arms}))
    return g2_pool(items, settings)


def g2_p6_args(reports: Sequence[GateReport], *, mode: str = 'latest',
               pooled: Optional[Mapping[str, Any]] = None) -> Dict[str, Optional[float]]:
    """``g2_score`` / ``g2_ci_low`` of the phase rules' ``p6_entry``. ``mode`` (config
    ``gates.g2_mode``): 'pooled_since_restart' reads ``pooled`` (:func:`g2_pool` over the valid G2s
    since the last restart; None or an invalid pool gives None), 'latest' the latest valid G2 of ``reports``
    (the current champion vs the IL reference at T* on the fixed block). None when there is none.
    A caller can report the latest G2 next to the pool."""
    if mode not in G2_MODES:
        raise GateError(f'g2 mode {mode!r} not in {G2_MODES}')
    if mode == 'pooled_since_restart':
        if not pooled or not pooled.get('valid') or pooled.get('score') is None:
            return {'g2_score': None, 'g2_ci_low': None}
        return {'g2_score': float(pooled['score']), 'g2_ci_low': pooled.get('lo')}
    lt = g2_trend(reports)['latest']
    if lt is None:
        return {'g2_score': None, 'g2_ci_low': None}
    return {'g2_score': lt['score'], 'g2_ci_low': lt['lo']}


# ------------------------------------------------------------------ CLI
def _load_cli_plan(store: GateStore, block: str, candidate: Optional[str]) -> GatePlan:
    d = store.block_dir(block)
    plans = sorted(d.glob('plan-*.json'))
    if candidate:
        plans = [p for p in plans if p.name.startswith(f'plan-{candidate[:16]}')]
    if len(plans) != 1:
        raise SystemExit(f'{d}: expected one plan-*.json (got {len(plans)}); pass --candidate DIGEST')
    return GatePlan.from_dict(_read_json(plans[0]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('status', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--root', required=True)
        p.add_argument('--block', required=True)
        p.add_argument('--candidate', default=None, help='candidate digest prefix (several plans in one block)')
        if name == 'evaluate':
            p.add_argument('--n-boot', type=int, default=S.N_BOOT)
            p.add_argument('--write', action='store_true', help='write report-<candidate>.json into the block')
            p.add_argument('--no-schedule-check', action='store_true',
                           help='do not require the default configured sizes / panel mix (non-default config)')
            p.add_argument('--inputs', default=None,
                           help='plan inputs JSON: its panel decks label the win-condition '
                                'table (e.g. the A/A reference of an aa block)')
    p = sub.add_parser('aa', help='score a gate-style A/A block (evaluate_aa: CI, rho, sizing, null tables)')
    p.add_argument('--root', required=True)
    p.add_argument('--block', required=True)
    p.add_argument('--candidate', default=None, help='candidate digest prefix (several plans in one block)')
    p.add_argument('--n-boot', type=int, default=S.N_BOOT, help='bootstrap resamples and sign-flip draws')
    p.add_argument('--g1-pairs', type=int, default=S.G1_PAIRS)
    p.add_argument('--inputs', default=None, help='plan inputs JSON: its panel decks label the win-condition table')
    p.add_argument('--write', action='store_true', help='write report-aa.json into the block')
    a = ap.parse_args(argv)
    store = GateStore(a.root)
    plan = _load_cli_plan(store, a.block, a.candidate)
    if a.cmd == 'aa':
        wc = None
        if a.inputs:
            wc = inputs_win_conditions(json.loads(Path(a.inputs).read_text()), C.CardKeys.load())
        rep = evaluate_aa(plan, store, GateSettings(n_boot=a.n_boot), win_conditions=wc, g1_pairs=a.g1_pairs)
        if a.write:
            write_aa_report(rep, store)
        print(json.dumps(rep, indent=1, sort_keys=True))
        return
    if a.cmd == 'status':
        print(json.dumps(store.status(plan), indent=1, sort_keys=True))
        return
    wc = None
    if a.inputs:
        wc = inputs_win_conditions(json.loads(Path(a.inputs).read_text()), C.CardKeys.load())
    rep = evaluate(plan, store, GateSettings(n_boot=a.n_boot, check_schedule=not a.no_schedule_check),
                   win_conditions=wc)
    if a.write:
        write_report(rep, store)
    print(json.dumps(rep.as_dict(), indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
