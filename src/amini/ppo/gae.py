"""Per-tick GAE, row weights, terminal rewards and shaping helpers for PPO.

Covers committed rows and their HT weights, the terminal reward, truncation and the drop-rate monitor,
the tower-HP potential and an optional separate short-horizon head, per-tick γ and λ, seat weights,
leave-one-out baselines (logging) and critic diagnostics by game phase. Pure numpy, float64; no torch.

Conventions for one trained seat-episode of K committed rows k = 0..K-1 at ticks t_0 < ... < t_{K-1}:

    dt_k     = t_{k+1} - t_k                    (the last row's next state is the terminal)
    delta_k  = r_k + g_k * V_{k+1} * (1 - done_k) - V_k
    A_k      = delta_k + g_k * l_k * (1 - done_k) * A_{k+1}
    R_k      = A_k + V_k                        (lambda-return, the critic target)

with g_k = gamma_tick ** dt_k and l_k = lambda_tick ** dt_k (defaults ``GAMMA_TICK``, ``LAMBDA_TICK``).
r_k is the reward of the transition k -> k+1 and is not discounted inside that transition; with the
terminal-only reward (``episode_rewards``) it is 0 except r_{K-1} = +1 / -1 / 0. Forced rows stay
in the chain (they have values); only their policy weight is 0.

Row weights (one rule for every policy statistic, the "unified weighting"):

    ht      = 1 on action rows, 1/kappa_s on committed wait rows (stratum s = base or threat)
    seat    = SEAT_WEIGHT of the row's seat: ours, the current policy on the opponent seat, or both
              seats of a mirror game
    policy  = ht * seat * [not forced]          surrogate, advantage stats, approx-KL, clip frac, ...
    value   = ht * seat                         forced rows still train the value head

``ht`` makes the weighted sum over committed rows an unbiased estimate of the sum over every query
(the commit coin is independent of the action given the stratum).

The PPO stratum of a committed row comes from the sampled family and the threat bit through
``committed_stratum`` / ``query_ht_weight`` (one shared mapping for the rollout writer and the
learner). The actor's ROW fields ``kind``, ``stratum`` and ``weight`` must never be used for PPO
weights: the actor writes every query row as a BASE/THREAT wait row with weight 1/kappa before the
family is sampled, so an executed action would get 1/kappa instead of 1.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from amini.nn import schema_v1 as S

# Commit probability kappa of each wait stratum (the model schema's) and its HT weight 1 / kappa.
KAPPA: Dict[str, float] = {k: float(v) for k, v in S.KAPPA.items()}
WEIGHT: Dict[str, float] = {k: 1.0 / v for k, v in KAPPA.items()}

ArrayLike = Union[np.ndarray, Sequence[float]]

GAMMA_TICK: float = 0.9999       # per-tick discount (example value)
LAMBDA_TICK: float = 0.995       # per-tick GAE lambda (example value)

# Optional separate short-horizon head (tower-damage reward; example values)
SHORT_C: float = 0.2
SHORT_GAMMA_DECISION: float = 0.99
DECISION_TICKS: int = S.U_TICKS  # one decision (query) per decision block of S.U_TICKS ticks
# Short-head discount unit: SHORT_GAMMA_DECISION applies once per SHORT_DECISION_TICKS ticks, so a smaller
# unit gives a shorter horizon for the same per-decision discount (example value: one decision block).
SHORT_DECISION_TICKS: int = DECISION_TICKS

# Seat weights (example values) and HT weights per stratum
SEAT_WEIGHT: Dict[str, float] = {"ours": 1.0, "opponent": 0.25, "mirror": 1.0}
HT_WEIGHT: Dict[int, float] = {S.STRATUM_ACTION: 1.0,
                               S.STRATUM_BASE: float(WEIGHT["base"]),       # 1 / kappa_base
                               S.STRATUM_THREAT: float(WEIGHT["threat"])}   # 1 / kappa_threat
COMMIT_PROB: Dict[int, float] = {S.STRATUM_ACTION: 1.0,
                                 S.STRATUM_BASE: float(KAPPA["base"]),
                                 S.STRATUM_THREAT: float(KAPPA["threat"])}

# How a game ended. Only a regular finish of the game (END_NATIVE, overtime included) is kept.
END_NATIVE = "native"
DROP_REASONS = ("crash", "step_guard", "seat_stall", "invalid")
DROP_ALARM_RATE = 0.01           # drop rate above this: alarm (example value)
DROP_STOP_RATE = 0.05            # drop rate above this: stop (example value)

OUTCOME_REWARD: Dict[str, float] = {"win": 1.0, "loss": -1.0, "draw": 0.0}


class GaeResult(NamedTuple):
    advantages: np.ndarray       # A_k
    returns: np.ndarray          # R_k = A_k + V_k (lambda-returns, critic targets)


# --------------------------------------------------------------------------- GAE core
def tick_intervals(ticks: ArrayLike) -> np.ndarray:
    """dt_k = t_{k+1} - t_k as float64; the last entry is 0 (unused: its next state is terminal).

    Raises ValueError unless the ticks are finite integers and strictly increasing."""
    t = np.asarray(ticks, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("ticks must be a non-empty 1-D array")
    if not np.all(np.isfinite(t)) or np.any(t != np.round(t)):
        raise ValueError("ticks must be finite integers")
    dt = np.zeros_like(t)
    dt[:-1] = np.diff(t)
    if np.any(dt[:-1] <= 0):
        raise ValueError("ticks must be strictly increasing within an episode")
    return dt


def gae_scan(rewards: ArrayLike, values: ArrayLike, discounts: ArrayLike, lambdas: ArrayLike,
             dones: ArrayLike) -> GaeResult:
    """General backward GAE recursion with explicit per-step factors (1-D, or 2-D [B, K] scanned
    over the last axis): delta_k = r_k + g_k V_{k+1}(1-d_k) - V_k, A_k = delta_k + g_k l_k (1-d_k) A_{k+1}.

    ``discounts`` g_k and ``lambdas`` l_k are the factors of the step k -> k+1 (gamma**dt and
    lambda**dt for the per-tick form). The last row of every chain must be done: truncated chains are
    rejected, never bootstrapped."""
    r, v, g, lam, d = (np.asarray(x, dtype=np.float64) for x in (rewards, values, discounts, lambdas, dones))
    if not (r.shape == v.shape == g.shape == lam.shape == d.shape) or r.ndim not in (1, 2) or r.shape[-1] == 0:
        raise ValueError(f"shape mismatch or empty: {r.shape} {v.shape} {g.shape} {lam.shape} {d.shape}")
    for name, x in (("rewards", r), ("values", v), ("discounts", g), ("lambdas", lam)):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be finite")
    if not np.all((d == 0.0) | (d == 1.0)):
        raise ValueError("dones must be 0/1")
    if np.any(d[..., -1] != 1.0):
        raise ValueError("the last row must be terminal (truncated episodes are dropped)")
    nd = 1.0 - d
    adv = np.zeros_like(r)
    nxt_v = np.zeros(r.shape[:-1])
    nxt_a = np.zeros(r.shape[:-1])
    for k in range(r.shape[-1] - 1, -1, -1):
        delta = r[..., k] + g[..., k] * nxt_v * nd[..., k] - v[..., k]
        a = delta + g[..., k] * lam[..., k] * nd[..., k] * nxt_a
        adv[..., k] = a
        nxt_v, nxt_a = v[..., k], a
    return GaeResult(adv, adv + v)


def gae_episode(rewards: ArrayLike, values: ArrayLike, ticks: ArrayLike, *,
                gamma_tick: float = GAMMA_TICK, lambda_tick: float = LAMBDA_TICK) -> GaeResult:
    """Per-tick GAE over one regularly finished seat-episode of K committed rows."""
    dt = tick_intervals(ticks)
    r = np.asarray(rewards, dtype=np.float64)
    if r.shape != dt.shape:
        raise ValueError("rewards, values and ticks must have the same length")
    _check_factor("gamma_tick", gamma_tick)
    _check_factor("lambda_tick", lambda_tick)
    d = np.zeros_like(dt)
    d[-1] = 1.0
    return gae_scan(r, values, float(gamma_tick) ** dt, float(lambda_tick) ** dt, d)


def gae_padded(rewards: ArrayLike, values: ArrayLike, ticks: ArrayLike, lengths: ArrayLike, *,
               gamma_tick: float = GAMMA_TICK, lambda_tick: float = LAMBDA_TICK) -> GaeResult:
    """Batched per-tick GAE on padded [B, K] arrays; row b uses its first lengths[b] entries.

    Entries past each length are ignored on input and are 0 on output."""
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    t = np.asarray(ticks, dtype=np.float64)
    n = _int_lengths(lengths)
    if r.ndim != 2 or r.shape != v.shape or r.shape != t.shape or n.shape != (r.shape[0],):
        raise ValueError("rewards/values/ticks must be [B, K] and lengths [B]")
    if np.any(n < 1) or np.any(n > r.shape[1]):
        raise ValueError("lengths must lie in [1, K]")
    _check_factor("gamma_tick", gamma_tick)
    _check_factor("lambda_tick", lambda_tick)
    valid = np.arange(r.shape[1])[None, :] < n[:, None]
    dt = np.zeros_like(t)
    for b in range(r.shape[0]):
        dt[b, :n[b]] = tick_intervals(t[b, :n[b]])
    d = np.zeros_like(r)
    d[np.arange(r.shape[0]), n - 1] = 1.0
    d[~valid] = 1.0
    out = gae_scan(np.where(valid, r, 0.0), np.where(valid, v, 0.0),
                   float(gamma_tick) ** dt, float(lambda_tick) ** dt, d)
    return GaeResult(np.where(valid, out.advantages, 0.0), np.where(valid, out.returns, 0.0))


def gae_ragged(rewards: ArrayLike, values: ArrayLike, ticks: ArrayLike, lengths: ArrayLike, *,
               gamma_tick: float = GAMMA_TICK, lambda_tick: float = LAMBDA_TICK) -> GaeResult:
    """Batched per-tick GAE on flat arrays holding consecutive episodes of the given lengths."""
    n = _int_lengths(lengths)
    r = np.asarray(rewards, dtype=np.float64)
    if n.ndim != 1 or n.size == 0 or np.any(n < 1) or int(n.sum()) != r.shape[0]:
        raise ValueError("lengths must be positive and sum to the number of rows")
    rows = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    ep = np.repeat(np.arange(n.size), n)
    pad = lambda x: _scatter(np.asarray(x, dtype=np.float64), ep, rows, n.size, int(n.max()))
    out = gae_padded(pad(r), pad(values), pad(ticks), n, gamma_tick=gamma_tick, lambda_tick=lambda_tick)
    return GaeResult(out.advantages[ep, rows], out.returns[ep, rows])


def _int_lengths(lengths: ArrayLike) -> np.ndarray:
    """Episode lengths as int64; raises on non-integer values (2.9 must not silently become 2 and drop
    the row holding the terminal reward)."""
    x = np.asarray(lengths)
    if x.size and x.dtype.kind not in "iu":
        if x.dtype.kind != "f" or not np.all(np.isfinite(x)) or np.any(x != np.round(x)):
            raise ValueError(f"lengths must be integers, got {x}")
    return x.astype(np.int64)


def _scatter(x: np.ndarray, ep: np.ndarray, rows: np.ndarray, b: int, k: int) -> np.ndarray:
    if x.shape != ep.shape:
        raise ValueError("flat arrays must all have the same length")
    out = np.zeros((b, k))
    out[ep, rows] = x
    return out


def _check_factor(name: str, x: float) -> None:
    """Discounts must lie in (0, 1]; lambdas in [0, 1] (lambda 0 is one-step TD)."""
    lo_ok = float(x) >= 0.0 if name.startswith("lambda") else float(x) > 0.0
    if not (lo_ok and float(x) <= 1.0):
        raise ValueError(f"{name} out of range, got {x}")


# --------------------------------------------------------------------------- rewards, truncation
def terminal_reward(outcome: Union[str, int, float]) -> float:
    """+1 win, -1 loss, 0 draw from the trained seat's view. Accepts the names or the numbers."""
    if isinstance(outcome, str):
        if outcome not in OUTCOME_REWARD:
            raise ValueError(f"unknown outcome {outcome!r}")
        return OUTCOME_REWARD[outcome]
    x = float(outcome)
    if x not in (-1.0, 0.0, 1.0):
        raise ValueError(f"outcome must be -1, 0 or +1, got {outcome!r}")
    return x


def episode_rewards(n_rows: int, outcome: Union[str, int, float]) -> np.ndarray:
    """Terminal-only reward vector: zeros except the last committed row."""
    if n_rows < 1:
        raise ValueError("an episode needs at least one committed row")
    r = np.zeros(int(n_rows))
    r[-1] = terminal_reward(outcome)
    return r


def validate_episode(end: str, ticks: ArrayLike, outcome: Union[str, int, float, None]) -> Optional[str]:
    """None when the seat-episode is kept, else the drop reason (one of ``DROP_REASONS``).

    ``end`` is END_NATIVE for a regular finish (including overtime) or a truncation reason
    ('crash', 'step_guard', 'seat_stall'); those games are dropped, never bootstrapped.
    A regularly finished episode with no rows, bad ticks or a bad outcome is 'invalid'."""
    if end != END_NATIVE:
        return end if end in DROP_REASONS else "invalid"
    try:
        tick_intervals(ticks)
        terminal_reward(outcome)                                        # type: ignore[arg-type]
    except (ValueError, TypeError):
        return "invalid"
    return None


@dataclass
class DropCounter:
    """Drop-rate monitor by reason and by the tower potential at the drop.

    Rates are over games (or seat-episodes, whichever the caller counts); the cheat channel it
    guards is "losing games crash or stall and vanish from training".
    The counter is cumulative: use one per update (or call ``reset()`` at the start of each update),
    otherwise a fully crashing update is diluted by earlier clean ones. Pass the run's thresholds:
    ``DropCounter(alarm_rate=..., stop_rate=...)``."""
    alarm_rate: float = DROP_ALARM_RATE
    stop_rate: float = DROP_STOP_RATE
    kept: int = 0
    by_reason: Counter = field(default_factory=Counter)
    phi_by_reason: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.alarm_rate) <= float(self.stop_rate) <= 1.0):
            raise ValueError(f"need 0 <= alarm_rate <= stop_rate <= 1, got {self.alarm_rate}, {self.stop_rate}")

    def reset(self) -> None:
        """Clear the counts (start of an update); the thresholds are kept."""
        self.kept = 0
        self.by_reason = Counter()
        self.phi_by_reason = {}

    def add(self, reason: Optional[str], phi: Optional[float] = None) -> None:
        """Count one game: ``reason`` None means kept; ``phi`` is Φ at the drop (logged when known)."""
        if reason is None:
            self.kept += 1
            return
        if reason not in DROP_REASONS:
            raise ValueError(f"unknown drop reason {reason!r}")
        self.by_reason[reason] += 1
        if phi is not None:
            self.phi_by_reason.setdefault(reason, []).append(float(phi))

    @property
    def dropped(self) -> int:
        return int(sum(self.by_reason.values()))

    @property
    def total(self) -> int:
        return self.kept + self.dropped

    def rate(self, reason: Optional[str] = None) -> float:
        """Dropped share of all games (for one reason when given); 0 when nothing was counted.
        Raises on an unknown reason (a typo must not read as a clean 0)."""
        if reason is not None and reason not in DROP_REASONS:
            raise ValueError(f"unknown drop reason {reason!r}")
        n = self.by_reason[reason] if reason is not None else self.dropped
        return n / self.total if self.total else 0.0

    def level(self) -> str:
        """'ok', 'alarm' (> alarm_rate) or 'stop' (> stop_rate) on the total drop rate."""
        x = self.rate()
        return "stop" if x > self.stop_rate else "alarm" if x > self.alarm_rate else "ok"

    def summary(self) -> dict:
        return {"total": self.total, "kept": self.kept, "dropped": self.dropped, "rate": self.rate(),
                "level": self.level(),
                "by_reason": {r: {"n": self.by_reason[r], "rate": self.rate(r),
                                  "phi_mean": (float(np.mean(self.phi_by_reason[r]))
                                               if self.phi_by_reason.get(r) else None)}
                              for r in DROP_REASONS if self.by_reason[r]}}


# --------------------------------------------------------------------------- weights
def committed_stratum(family: ArrayLike, threat: ArrayLike) -> np.ndarray:
    """PPO stratum of a query from its sampled family and threat bit: STRATUM_ACTION when
    family != FAMILY_WAIT (every executed action commits, kappa 1), else STRATUM_THREAT / STRATUM_BASE.

    The single mapping the rollout writer and the learner share; never read the stratum from the
    actor's row (it is written as a wait row before the family is sampled)."""
    f = np.asarray(family)
    th = np.asarray(threat).astype(bool)
    if f.shape != th.shape:
        raise ValueError("family and threat must have the same shape")
    if f.size and not np.all(np.isin(f, (S.FAMILY_WAIT, S.FAMILY_PLAY, S.FAMILY_ABILITY))):
        raise ValueError(f"unknown family in {np.unique(f)}")
    return np.where(f != S.FAMILY_WAIT, S.STRATUM_ACTION,
                    np.where(th, S.STRATUM_THREAT, S.STRATUM_BASE)).astype(np.int64)


def query_ht_weight(family: ArrayLike, threat: ArrayLike) -> np.ndarray:
    """HT weight of a committed query from (family, threat): 1 action, 1/kappa_base base wait,
    1/kappa_threat threat wait."""
    return ht_weight(committed_stratum(family, threat))


def ht_weight(stratum: ArrayLike) -> np.ndarray:
    """Horvitz–Thompson weight per committed row from its stratum (S.STRATUM_*): 1 action,
    1/kappa_base wait, 1/kappa_threat wait (``WEIGHT``)."""
    s = np.asarray(stratum)
    out = np.full(s.shape, np.nan)
    for k, w in HT_WEIGHT.items():
        out[s == k] = w
    if np.any(np.isnan(out)):
        raise ValueError(f"unknown stratum in {np.unique(s)}")
    return out


def seat_weight(role: Union[str, float], table: Optional[Mapping[str, float]] = None) -> float:
    """Seat weight from ``SEAT_WEIGHT``: 'ours', 'opponent' (current policy on the opponent seat),
    'mirror'. Frozen opponents produce no rows, so they have no weight.

    ``role`` may also be the weight itself (e.g. a per-game opponent-seat weight, or a run
    configuration's seat weight); ``table`` overrides SEAT_WEIGHT, e.g.
    {'ours': ..., 'opponent': ..., 'mirror': ...} from a run configuration."""
    if not isinstance(role, str):
        x = float(role)
        if not (np.isfinite(x) and x > 0.0):
            raise ValueError(f"seat weight must be finite and > 0, got {role!r}")
        return x
    t = SEAT_WEIGHT if table is None else table
    if role not in t:
        raise ValueError(f"unknown or untrained seat role {role!r}")
    return float(t[role])


class RowWeights(NamedTuple):
    ht: np.ndarray       # HT weight alone
    policy: np.ndarray   # the unified policy weight: ht * seat * [not forced]
    value: np.ndarray    # ht * seat (forced rows included)


def row_weights(stratum: ArrayLike, forced: ArrayLike, seat_role: Union[str, float], *,
                family: Optional[ArrayLike] = None,
                seat_table: Optional[Mapping[str, float]] = None) -> RowWeights:
    """Weights of one seat-episode's committed rows (module docstring).

    ``stratum`` is the PPO stratum from ``committed_stratum``. Pass ``family`` (the sampled family per
    row) to cross-check it: a row with family != WAIT must carry STRATUM_ACTION and a wait row must
    not, otherwise ValueError (guards against the actor's recorded BASE/THREAT stratum)."""
    ht = ht_weight(stratum)
    f = np.asarray(forced, dtype=bool)
    if f.shape != ht.shape:
        raise ValueError("stratum and forced must have the same shape")
    if family is not None:
        fam = np.asarray(family)
        st = np.asarray(stratum)
        if fam.shape != st.shape:
            raise ValueError("family and stratum must have the same shape")
        if np.any((fam != S.FAMILY_WAIT) != (st == S.STRATUM_ACTION)):
            raise ValueError("stratum disagrees with family: action rows need STRATUM_ACTION, wait rows "
                             "BASE/THREAT (use committed_stratum, not the actor row's stratum)")
    sw = seat_weight(seat_role, seat_table)
    return RowWeights(ht, ht * sw * (~f), ht * sw)


# --------------------------------------------------------------------------- advantage normalisation
def advantage_stats(advantages: ArrayLike, weights: ArrayLike) -> tuple:
    """Weighted mean and (population) std of the advantages over a whole update batch, with the
    unified policy weight. Compute once per update."""
    a = np.asarray(advantages, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.shape != w.shape or np.any(w < 0) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(w)):
        raise ValueError("advantages and weights must be finite, same shape, weights >= 0")
    sw = float(w.sum())
    if sw <= 0.0:
        raise ValueError("advantage normalisation needs positive total weight")
    mean = float((w * a).sum() / sw)
    std = float(np.sqrt((w * (a - mean) ** 2).sum() / sw))
    return mean, std


def normalize_advantages(advantages: ArrayLike, weights: ArrayLike, *, eps: float = 1e-8,
                         stats: Optional[tuple] = None) -> tuple:
    """(A - mean) / (std + eps) with weighted stats (computed here unless ``stats`` is given).

    Returns (normalised, mean, std). Zero-weight rows are transformed too but carry no loss weight."""
    a = np.asarray(advantages, dtype=np.float64)
    mean, std = stats if stats is not None else advantage_stats(a, weights)
    return (a - mean) / (std + eps), mean, std


# --------------------------------------------------------------------------- tower potential, shaping
def tower_potential(own_hp: ArrayLike, own_max: ArrayLike, opp_hp: ArrayLike, opp_max: ArrayLike, *,
                    clip: bool = False) -> np.ndarray:
    """Φ = (Σ own tower hp/max − Σ opp tower hp/max) / 3 over the last axis (3 towers), in [-1, 1].
    A destroyed tower has hp 0. Logged on every decision tick; Φ_terminal = 0 by convention
    for potential shaping (the short-horizon head uses the real end-of-game Φ instead).

    hp > max raises ValueError (it hides a level mismatch, e.g. a tower read against the max-hp table
    of another level); ``clip=True`` clips hp/max to 1 instead."""
    oh, om, ph, pm = (np.asarray(x, dtype=np.float64) for x in (own_hp, own_max, opp_hp, opp_max))
    if oh.shape[-1:] != (3,) or ph.shape[-1:] != (3,):
        raise ValueError("tower arrays must have 3 towers on the last axis")
    if np.any(om <= 0) or np.any(pm <= 0) or np.any(oh < 0) or np.any(ph < 0):
        raise ValueError("tower hp must be >= 0 and max hp > 0")
    if not all(np.all(np.isfinite(x)) for x in (oh, om, ph, pm)):
        raise ValueError("tower hp must be finite")
    if not clip and (np.any(oh > om) or np.any(ph > pm)):
        raise ValueError("tower hp above max hp (level mismatch?); pass clip=True to clip")
    own = np.minimum(oh / om, 1.0).sum(-1)
    opp = np.minimum(ph / pm, 1.0).sum(-1)
    return (own - opp) / 3.0


def shaping_terms(phi: ArrayLike, c: float, *, phi_end: float = 0.0,
                  discounts: Optional[ArrayLike] = None) -> np.ndarray:
    """c * (g_k Φ_{k+1} − Φ_k) per committed row, with Φ_K = ``phi_end`` (0 for potential shaping).

    ``discounts`` g_k = gamma_tick ** dt_k (default None: factor 1, i.e. gamma 1)."""
    p = np.asarray(phi, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("phi must be a finite non-empty 1-D array")
    nxt = np.append(p[1:], float(phi_end))
    if discounts is not None:
        g = np.asarray(discounts, dtype=np.float64)
        if g.shape != p.shape or not np.all(np.isfinite(g)):
            raise ValueError("discounts must be finite and match phi")
        nxt = g * nxt
    return float(c) * (nxt - p)


def shaped_rewards(rewards: ArrayLike, phi: ArrayLike, c: float, *, phi_end: float = 0.0,
                   ticks: Optional[ArrayLike] = None, gamma_tick: float = GAMMA_TICK) -> np.ndarray:
    """r'_k = r_k + c (γ^dt_k Φ_{k+1} − Φ_k). With Φ_end = 0, a residual critic V = U − cΦ and the
    same ``gamma_tick`` as the GAE, this is an exact no-op in every δ, A and target;
    kept for logging. ``ticks`` are required when gamma_tick != 1 (at γ = 1 the factor is 1)."""
    r = np.asarray(rewards, dtype=np.float64)
    _check_factor("gamma_tick", gamma_tick)
    g = None
    if float(gamma_tick) != 1.0:
        if ticks is None:
            raise ValueError("ticks are required for gamma_tick != 1")
        g = float(gamma_tick) ** tick_intervals(ticks)
    s = shaping_terms(phi, c, phi_end=phi_end, discounts=g)
    if r.shape != s.shape:
        raise ValueError("rewards and phi must have the same length")
    return r + s


def telescoping_error(phi: ArrayLike, c: float, *, phi_end: float = 0.0) -> float:
    """|Σ_k c(Φ_{k+1} − Φ_k) − c(Φ_end − Φ_0)| (γ = 1). An arithmetic identity of its input: it
    guards the shaping code path (row order, the Φ_K end value), not the Φ data; for the data use
    ``potential_log_problems``."""
    p = np.asarray(phi, dtype=np.float64)
    return float(abs(shaping_terms(p, c, phi_end=phi_end).sum() - float(c) * (float(phi_end) - p[0])))


def potential_log_problems(phi_log: ArrayLike, phi_terminal: float, *, phi_kickoff: float = 0.0,
                           tol: float = 1e-9) -> List[str]:
    """Data check of one game's Φ log. ``phi_log`` is Φ on every decision tick from kick-off;
    ``phi_terminal`` is ``tower_potential`` of the end-of-game tower state. Returns the problems found
    (empty when clean): non-finite or empty log, |Φ| > 1, Φ at kick-off != ``phi_kickoff`` (0: all
    towers full), |Φ_terminal| > 1."""
    p = np.asarray(phi_log, dtype=np.float64)
    out: List[str] = []
    if p.ndim != 1 or p.size == 0:
        return ["empty or non 1-D phi log"]
    if not np.all(np.isfinite(p)):
        return ["non-finite phi"]
    if np.any(np.abs(p) > 1.0 + tol):
        out.append(f"|phi| > 1 at {int(np.argmax(np.abs(p) > 1.0 + tol))}")
    if abs(p[0] - float(phi_kickoff)) > tol:
        out.append(f"phi at kick-off {p[0]!r} != {phi_kickoff!r}")
    pt = float(phi_terminal)
    if not np.isfinite(pt) or abs(pt) > 1.0 + tol:
        out.append(f"terminal phi {pt!r} outside [-1, 1]")
    return out


def short_head_gae(phi: ArrayLike, values_short: ArrayLike, ticks: ArrayLike, *, phi_end: float,
                   c: float = SHORT_C, gamma_decision: float = SHORT_GAMMA_DECISION,
                   decision_ticks: float = SHORT_DECISION_TICKS,
                   lambda_tick: float = LAMBDA_TICK) -> GaeResult:
    """Separate short-horizon head: reward c (Φ_{k+1} − Φ_k) with Φ_K = ``phi_end``, discount
    ``gamma_decision`` per decision, i.e. gamma_decision ** (dt / decision_ticks) between committed
    rows, and TD(λ) with λ_tick ** dt.
    Returns A_short and its TD(λ) targets. This changes the objective (it is not potential-invariant),
    so it is an option to compare against, not a default. ``decision_ticks`` defaults to
    :data:`SHORT_DECISION_TICKS`: ``gamma_decision`` applies once per ``decision_ticks`` ticks.

    The discount is per tick, not per committed row: committed rows are a thinned sample of the
    decisions (commit coins, threat stratum, policy), so a per-row discount would depend on them.

    ``phi_end`` is required on purpose: the short head rewards tower damage, i.e. the real Φ at the end
    of the game; Φ_end = 0 would add a final −cΦ_{K−1} that undoes the damage reward."""
    dt = tick_intervals(ticks)
    _check_factor("gamma_decision", gamma_decision)
    _check_factor("lambda_tick", lambda_tick)
    r = shaping_terms(phi, c, phi_end=phi_end)
    if r.shape != dt.shape:
        raise ValueError("phi, values_short and ticks must have the same length")
    d = np.zeros_like(dt)
    d[-1] = 1.0
    if not (np.isfinite(float(decision_ticks)) and float(decision_ticks) > 0.0):
        raise ValueError(f"decision_ticks must be > 0, got {decision_ticks}")
    g = float(gamma_decision) ** (dt / float(decision_ticks))
    return gae_scan(r, values_short, g, float(lambda_tick) ** dt, d)


def combined_advantages(a_win: ArrayLike, a_short: ArrayLike) -> np.ndarray:
    """A = A_win + A_short; normalise the sum afterwards."""
    a, b = np.asarray(a_win, dtype=np.float64), np.asarray(a_short, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("advantage arrays must have the same shape")
    return a + b


# --------------------------------------------------------------------------- leave-one-out (logging only)
def loo_advantages(returns: ArrayLike, groups: ArrayLike) -> np.ndarray:
    """A_i = R_i − mean_{j ≠ i, same group} R_j per episode (logging only).

    Groups of one have no baseline and give NaN. Replicas must use different seeds, otherwise
    identical games make every A_i = 0."""
    r = np.asarray(returns, dtype=np.float64)
    g = np.asarray(groups)
    if r.ndim != 1 or r.shape != g.shape or not np.all(np.isfinite(r)):
        raise ValueError("returns and groups must be finite 1-D arrays of the same length")
    _, inv = np.unique(g, return_inverse=True)
    inv = inv.reshape(-1)
    n = np.bincount(inv).astype(np.float64)
    tot = np.bincount(inv, weights=r)
    others = n[inv] - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        base = np.where(others > 0, (tot[inv] - r) / np.maximum(others, 1.0), np.nan)
    return r - base


# --------------------------------------------------------------------------- critic EV by game phase
PHASE_NAMES = ("first_third", "middle_third", "last_third")


def game_phase(ticks: ArrayLike, end_ticks: ArrayLike) -> np.ndarray:
    """Phase index of each row: 0, 1, 2 for x = tick / end_tick in [0, 1/3), [1/3, 2/3), [2/3, 1].

    ``end_ticks`` is the final tick of the row's own game (per row). The boundaries are exact
    rationals: a row is in the middle third iff 3 * tick >= end_tick (no float rounding at 1/3), and in
    the last third iff 3 * tick >= 2 * end_tick; the final tick itself (x = 1) is in the last third."""
    t = np.asarray(ticks, dtype=np.float64)
    e = np.asarray(end_ticks, dtype=np.float64)
    if t.shape != e.shape or t.ndim != 1:
        raise ValueError("ticks and end_ticks must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e))):
        raise ValueError("ticks and end_ticks must be finite")
    if np.any(e <= 0):
        raise ValueError("end_ticks must be > 0")
    if np.any(t < 0) or np.any(t > e):
        raise ValueError("every tick must lie in [0, end_tick] of its game")
    return (3.0 * t >= e).astype(np.int64) + (3.0 * t >= 2.0 * e).astype(np.int64)


def _weighted_ev(y: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """1 - Var_w(y - v) / Var_w(y) (weighted, both variances about their weighted means); NaN with no
    weight or a constant outcome."""
    sw = float(w.sum())
    if sw <= 0.0:
        return float("nan")
    my = float((w * y).sum()) / sw
    var_y = float((w * (y - my) ** 2).sum()) / sw
    if var_y <= 0.0:
        return float("nan")
    r = y - v
    mr = float((w * r).sum()) / sw
    var_r = float((w * (r - mr) ** 2).sum()) / sw
    return 1.0 - var_r / var_y


def explained_variance_by_phase(values: ArrayLike, outcomes: ArrayLike, ticks: ArrayLike, end_ticks: ArrayLike,
                                weights: Optional[ArrayLike] = None,
                                groups: Optional[Sequence] = None) -> Dict[str, object]:
    """Critic explained variance against the TERMINAL OUTCOME, overall, per game third and per group
    (for a critic warm-up pass rule and the critic EV monitor; monitor keys ``ev_overall`` /
    ``ev_last_third`` / ``ev_by_opponent``).

    Rows are committed rows of trained seats. ``values[i]`` is V(s_i); ``outcomes[i]`` is the final
    result of row i's game from row i's seat (+1 / -1 / 0, :func:`terminal_reward`), NOT a lambda-return
    (lambda-returns contain the critic's own predictions and would inflate EV). ``ticks[i]`` is the
    row's tick and ``end_ticks[i]`` the final tick of the same game.

    Phase definition: x = tick / end_tick of that game; first third
    x in [0, 1/3), middle third [1/3, 2/3), last third [2/3, 1] (see :func:`game_phase`). Phases are
    relative to each game's own length, so a short and a long game both contribute to every third.

    EV = 1 - Var_w(outcome - V) / Var_w(outcome), weighted variances about the weighted means (like
    sklearn's explained_variance_score); NaN where the subset has no weight or a constant outcome.
    ``weights`` default to 1 (pass the unified value weights ``RowWeights.value`` to estimate the
    per-query EV); ``groups`` (e.g. the opponent kind per row) give ``ev_by_group``.

    Returns ``ev_overall``, ``ev_by_phase`` (list of 3), ``ev_first_third``, ``ev_middle_third``,
    ``ev_last_third``, ``ev_by_group`` {str(group): ev}, ``n_rows``, ``n_by_phase`` (list of 3),
    ``n_by_group`` and ``weight_by_phase`` (list of 3)."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if v.ndim != 1 or v.shape != y.shape:
        raise ValueError("values and outcomes must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(y))):
        raise ValueError("values and outcomes must be finite")
    ph = game_phase(ticks, end_ticks)
    if ph.shape != v.shape:
        raise ValueError("ticks must have one entry per row")
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != v.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, >= 0 and one per row")
    out: Dict[str, object] = {"ev_overall": _weighted_ev(y, v, w), "n_rows": int(v.size)}
    ev_ph, n_ph, w_ph = [], [], []
    for k in range(3):
        m = ph == k
        ev_ph.append(_weighted_ev(y[m], v[m], w[m]))
        n_ph.append(int(m.sum()))
        w_ph.append(float(w[m].sum()))
    out["ev_by_phase"] = ev_ph
    for name, ev in zip(PHASE_NAMES, ev_ph):
        out[f"ev_{name}"] = ev
    out["n_by_phase"] = n_ph
    out["weight_by_phase"] = w_ph
    by_g: Dict[str, float] = {}
    n_g: Dict[str, int] = {}
    if groups is not None:
        g = np.asarray(groups)
        if g.shape != v.shape:
            raise ValueError("groups must have one entry per row")
        gs = np.array([str(x) for x in g.tolist()], dtype=object)
        for lab in sorted(set(gs.tolist())):
            m = gs == lab
            by_g[lab] = _weighted_ev(y[m], v[m], w[m])
            n_g[lab] = int(m.sum())
    out["ev_by_group"] = by_g
    out["n_by_group"] = n_g
    return out


def value_calibration(values: ArrayLike, outcomes: ArrayLike, ticks: ArrayLike, end_ticks: ArrayLike,
                      weights: Optional[ArrayLike] = None, bins: int = 10) -> Dict[str, object]:
    """Per-phase calibration of the critic against the TERMINAL OUTCOME.

    Same inputs, phases and default weights as :func:`explained_variance_by_phase`. Rows are binned by V into
    ``bins`` equal-width bins over [-1, 1] (tanh values; the edges include both ends). For ``overall`` and each
    third: ``bins`` = [{lo, hi, n, weight, mean_v, mean_outcome}] (empty bins left out), ``mse`` (weighted
    mean of (outcome - V)^2), ``bias`` (weighted mean of outcome - V), ``slope`` / ``intercept`` of the
    weighted least-squares fit outcome ~ V (slope 1, intercept 0 = calibrated; NaN for a constant V or no
    weight) and ``ece`` (weight-averaged |mean_outcome - mean_v| over the bins)."""
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if v.ndim != 1 or v.shape != y.shape:
        raise ValueError("values and outcomes must be 1-D arrays of the same length")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(y))):
        raise ValueError("values and outcomes must be finite")
    if int(bins) < 1:
        raise ValueError("bins must be >= 1")
    ph = game_phase(ticks, end_ticks)
    if ph.shape != v.shape:
        raise ValueError("ticks must have one entry per row")
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != v.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, >= 0 and one per row")
    edges = np.linspace(-1.0, 1.0, int(bins) + 1)
    idx = np.clip(np.searchsorted(edges, np.clip(v, -1.0, 1.0), side="right") - 1, 0, int(bins) - 1)

    def one(m: np.ndarray) -> Dict[str, object]:
        vw, yw, ww, iw = v[m], y[m], w[m], idx[m]
        sw = float(ww.sum())
        rec: Dict[str, object] = {"n": int(m.sum()), "weight": sw}
        if sw <= 0.0:
            rec.update(mse=float("nan"), bias=float("nan"), slope=float("nan"), intercept=float("nan"),
                       ece=float("nan"), bins=[])
            return rec
        r = yw - vw
        rec["mse"] = float((ww * r * r).sum()) / sw
        rec["bias"] = float((ww * r).sum()) / sw
        mv, my = float((ww * vw).sum()) / sw, float((ww * yw).sum()) / sw
        var_v = float((ww * (vw - mv) ** 2).sum()) / sw
        if var_v > 0.0:
            slope = float((ww * (vw - mv) * (yw - my)).sum()) / sw / var_v
            rec["slope"], rec["intercept"] = slope, my - slope * mv
        else:
            rec["slope"] = rec["intercept"] = float("nan")
        rows, ece = [], 0.0
        for b in range(int(bins)):
            k = iw == b
            wb = float(ww[k].sum())
            if not k.any() or wb <= 0.0:
                continue
            bv, by = float((ww[k] * vw[k]).sum()) / wb, float((ww[k] * yw[k]).sum()) / wb
            rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(k.sum()), "weight": wb,
                         "mean_v": bv, "mean_outcome": by})
            ece += wb * abs(by - bv)
        rec["ece"] = ece / sw
        rec["bins"] = rows
        return rec

    out: Dict[str, object] = {"overall": one(np.ones(v.shape, dtype=bool))}
    for k, name in enumerate(PHASE_NAMES):
        out[name] = one(ph == k)
    return out
