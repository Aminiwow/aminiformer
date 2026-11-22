"""Gate statistics for PPO on aminiformer.

Pure numpy / standard library. All effects are fractions (0.05 = 5 pp), all results are scores in
{0, 0.5, 1} from the scored side's point of view. The pair counts, the thresholds and the G1 alpha
below are example values; a run passes its own.

* CRN promotion gate G1: candidate and champion each play the same (case, seed, frozen
  opponent) triple. delta_hat = mean(candidate - champion) over complete triples; SE analytic
  (cluster-robust over case_id) and by cluster bootstrap over case_id (N_BOOT resamples, seeded).
  Pass iff the one-sided (1 - alpha) lower bound > 0 AND the held-out-opponent sub-panel
  delta_hat >= HOLDOUT_MIN_DELTA.
* Screen S: same statistic on SCREEN_PAIRS pairs, alarm and rollback input only, never promotes.
* Seat-swapped head-to-head G1c / release H1: per case two games, A on seat 0 and on seat 1;
  pair score = mean of A's two results; pass iff the two-sided 95% CI excludes 0.5 from above
  (lower bound > 0.5, a one-sided 2.5% rule).
* Pair completeness: a pair with a missing game is dropped as a whole; completion < MIN_COMPLETION
  makes the gate invalid (replay it), whatever the statistic says. The record-level gates default
  ``n_expected`` to the scheduled size (G1_PAIRS, SCREEN_PAIRS; seat-swap has no default since G1c,
  H1 and G2 differ), so blocks lost without any record still count as missing.
* Guard rails: clusters are numbered in sorted order (str, like the evaluation harness's cluster
  bootstrap), so a seeded bootstrap does not depend on record order; a gate with fewer than
  MIN_CLUSTERS clusters (cases, or openings for a clustered seat-swap) is invalid.
  Seat-swapped pairs may carry an opening ``cluster_id``: when a panel reuses openings across
  pairs, the CI then resamples / clusters over openings instead of pairs.
* Rollback trigger: ROLLBACK_CONSECUTIVE consecutive delta_hat (screens or G1s) <= ROLLBACK_DELTA.
* Wilson interval, power / false-promotion helpers, an optional truncated SPRT for G1 (H0 0,
  H1 +SPRT_DELTA1, cap G1_PAIRS pairs) and a Beta posterior for snapshot win rates (PFSP).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- constants (example values)
G1_PAIRS = 2000             # G1: triples per arm (example value)
G1C_PAIRS = 2000            # G1c: seat-swapped pairs (example value)
SCREEN_PAIRS = 800          # screen S (example value)
RELEASE_PAIRS = 1500        # release H1 (example value)
N_BOOT = 10_000             # cluster bootstrap resamples over cases
MIN_COMPLETION = 0.95       # gate completeness (example value)
HOLDOUT_MIN_DELTA = -0.02   # G1 held-out sub-panel delta_hat >= -2 pp (example value)
ROLLBACK_DELTA = -0.05      # rollback: consecutive delta_hat <= -5 pp ... (example value)
ROLLBACK_CONSECUTIVE = 3    # ... this many in a row (example value)
SEATSWAP_SD = 0.3           # SD of the seat-swapped pair score assumed by the power helpers (example value)
SPRT_DELTA1 = 0.05          # SPRT H1 (example value)
MIN_CLUSTERS = 30           # guard rail: fewer cases / openings -> gate invalid
VALID_RESULTS = (0.0, 0.5, 1.0)

_N = NormalDist()
_BOOT_CHUNK = 1000          # resamples per chunk; fixed so a seed always gives the same draws


def z_quantile(p: float) -> float:
    """Standard normal quantile."""
    return _N.inv_cdf(p)


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


# ---------------------------------------------------------------- records and validation
class CrnRecord(NamedTuple):
    """One CRN triple. A missing game is None (the triple is then dropped as a whole)."""
    triple_id: Any
    case_id: Any
    candidate: Optional[float]
    champion: Optional[float]
    holdout: bool = False     # triple's opponent is in the held-out IL sub-panel


class SeatGame(NamedTuple):
    """One seat-swapped game: A's result with A on ``seat`` (0 or 1). None = missing.

    ``case_id`` identifies the PAIR (two games, one per seat). ``cluster_id`` is the opening the
    pair was played on, when openings repeat across pairs (e.g. a G1c panel with fewer openings than pairs);
    None means the pair is its own cluster."""
    case_id: Any
    seat: int
    a_result: Optional[float]
    cluster_id: Any = None


def _is_missing(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and math.isnan(float(x)))


def _check_result(x: float) -> float:
    x = float(x)
    if x not in VALID_RESULTS:
        raise ValueError(f'game result must be one of {VALID_RESULTS}, got {x!r}')
    return x


def completion(n_complete: int, n_expected: int, min_rate: float = MIN_COMPLETION) -> Tuple[float, bool]:
    """(completion rate, valid) for a gate of ``n_expected`` pairs (below ``min_rate`` -> invalid, replay)."""
    if n_expected <= 0:
        return 0.0, False
    if n_complete > n_expected:
        raise ValueError(f'{n_complete} complete pairs > {n_expected} expected')
    rate = n_complete / n_expected
    return rate, rate >= min_rate


# ---------------------------------------------------------------- cluster statistics
def _cluster_key(c: Any) -> Tuple[str, str, str]:
    # str first: the evaluation harness's order (its cluster bootstrap); the type and
    # repr only break ties between distinct ids with the same str (e.g. 1 and '1').
    return str(c), type(c).__name__, repr(c)


def _cluster_index(cluster_ids: Sequence[Any]) -> Tuple[np.ndarray, int]:
    """(cluster index per value, number of clusters), clusters numbered in sorted order so a
    seeded bootstrap is independent of the order of the records."""
    uniq = sorted(dict.fromkeys(cluster_ids), key=_cluster_key)
    keys = {c: i for i, c in enumerate(uniq)}
    inv = np.fromiter((keys[c] for c in cluster_ids), dtype=np.int64, count=len(cluster_ids))
    return inv, len(keys)


def _check_array(x: np.ndarray, name: str, allowed: Optional[Sequence[float]] = VALID_RESULTS) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f'{name} has NaN/inf: drop incomplete pairs before the array API')
    if allowed is not None and not np.all(np.isin(x, allowed)):
        raise ValueError(f'{name} values must be in {tuple(allowed)}')


def cluster_se(values: np.ndarray, cluster_inv: np.ndarray, n_clusters: int) -> float:
    """Analytic cluster-robust SE of the mean (CR1 sandwich). With one value per cluster this equals
    the iid SE sd(ddof=1)/sqrt(n)."""
    n = values.size
    if n < 2 or n_clusters < 2:
        return float('nan')
    resid = np.bincount(cluster_inv, weights=values - values.mean(), minlength=n_clusters)
    return float(math.sqrt(n_clusters / (n_clusters - 1) * float(np.sum(resid ** 2))) / n)


def cluster_bootstrap_means(values: np.ndarray, cluster_inv: np.ndarray, n_clusters: int,
                            n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the mean over values, resampling CLUSTERS with replacement
    (each resample's mean = sum of drawn cluster sums / sum of drawn cluster sizes, as in the
    harness's cluster_bootstrap). Reproducible for a given seed."""
    if n_clusters < 1 or n_boot < 1:
        return np.empty(0)
    sums = np.bincount(cluster_inv, weights=values, minlength=n_clusters)
    cnts = np.bincount(cluster_inv, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for s in range(0, n_boot, _BOOT_CHUNK):
        m = min(_BOOT_CHUNK, n_boot - s)
        idx = rng.integers(0, n_clusters, size=(m, n_clusters))
        out[s:s + m] = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    return out


# ---------------------------------------------------------------- CRN gate G1 and screen S
@dataclass
class CrnResult:
    """G1 or screen statistic. ``lb`` is the one-sided (1 - alpha) lower bound used by the rule."""
    kind: str
    n_expected: int
    n_pairs: int
    n_cases: int
    completion: float
    valid: bool
    delta_hat: float
    se_iid: float
    se_cluster: float
    se_boot: Optional[float]
    lb_analytic: float
    ub_analytic: float
    lb_boot: Optional[float]
    ub_boot: Optional[float]
    method: str
    lb: float
    holdout_pairs: int
    holdout_delta: Optional[float]
    passed: bool
    alarm: bool
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _one_sided(alpha: float) -> float:
    return z_quantile(1.0 - alpha)


def crn_arrays(case_ids: Sequence[Any], candidate: np.ndarray, champion: np.ndarray,
               holdout: Optional[np.ndarray] = None, *, n_expected: Optional[int] = None,
               kind: str = 'G1', method: str = 'bootstrap', alpha: float = 0.1,
               n_boot: int = N_BOOT, seed: int = 0, min_completion: float = MIN_COMPLETION,
               holdout_min: float = HOLDOUT_MIN_DELTA, require_holdout: bool = True,
               alarm_delta: float = ROLLBACK_DELTA, min_clusters: int = MIN_CLUSTERS) -> CrnResult:
    """CRN statistic on complete triples given as arrays.

    ``method`` 'bootstrap' (default; the ``alpha`` quantile of the case-cluster bootstrap) or
    'analytic' (delta_hat - z_(1-alpha) * cluster-robust SE) picks the bound the rule uses; both are
    reported when bootstrapping. ``kind`` 'G1' applies the promotion rule; 'S' (screen) never
    passes and only sets ``alarm`` (delta_hat <= ``alarm_delta``, the rollback input).

    The arrays hold complete triples only. ``n_expected`` None means they are the whole schedule
    (no completeness check); gates fed from records use crn_gate / screen_s, whose default is the
    scheduled size. Fewer than ``min_clusters`` distinct cases makes the gate invalid: the
    bootstrap degenerates (one case -> every resample = delta_hat) and CR1 is anti-conservative."""
    if method not in ('bootstrap', 'analytic'):
        raise ValueError(f'unknown method {method!r}')
    if kind not in ('G1', 'S'):
        raise ValueError(f'unknown kind {kind!r}')
    cand = np.asarray(candidate, dtype=np.float64)
    champ = np.asarray(champion, dtype=np.float64)
    if cand.shape != champ.shape or cand.ndim != 1 or len(case_ids) != cand.size:
        raise ValueError('case_ids, candidate and champion must be 1-D and equally long')
    _check_array(cand, 'candidate')
    _check_array(champ, 'champion')
    hold = np.zeros(cand.size, dtype=bool) if holdout is None else np.asarray(holdout, dtype=bool)
    if hold.shape != cand.shape:
        raise ValueError('holdout must be as long as candidate')
    n = int(cand.size)
    n_exp = n if n_expected is None else int(n_expected)
    rate, valid = completion(n, n_exp, min_completion)
    reasons: List[str] = []
    if not valid:
        reasons.append(f'incomplete: {n}/{n_exp} pairs ({rate:.4f} < {min_completion})')
    d = cand - champ
    inv, g = _cluster_index(case_ids)
    if g < min_clusters:
        valid = False
        reasons.append(f'too few cases: {g} < {min_clusters}')
    delta = float(d.mean()) if n else float('nan')
    se_iid = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float('nan')
    se_cl = cluster_se(d, inv, g)
    z = _one_sided(alpha)
    lb_a, ub_a = delta - z * se_cl, delta + z * se_cl
    se_b = lb_b = ub_b = None
    if method == 'bootstrap' and n:
        boot = cluster_bootstrap_means(d, inv, g, n_boot, seed)
        se_b = float(boot.std(ddof=1))
        lb_b, ub_b = (float(x) for x in np.quantile(boot, [alpha, 1.0 - alpha]))
    lb = lb_b if method == 'bootstrap' and lb_b is not None else lb_a
    h_n = int(hold.sum())
    h_delta = float(d[hold].mean()) if h_n else None
    passed = False
    if kind == 'G1':
        ok = valid
        if not (lb > 0.0):
            reasons.append(f'lower bound {lb:.4f} <= 0')
            ok = False
        if h_delta is None:
            if require_holdout:
                reasons.append('no held-out sub-panel pairs')
                ok = False
        elif h_delta < holdout_min:
            reasons.append(f'held-out delta {h_delta:.4f} < {holdout_min}')
            ok = False
        passed = bool(ok)
    alarm = bool(valid and n and delta <= alarm_delta)
    return CrnResult(kind=kind, n_expected=n_exp, n_pairs=n, n_cases=g, completion=rate, valid=valid,
                     delta_hat=delta, se_iid=se_iid, se_cluster=se_cl, se_boot=se_b, lb_analytic=lb_a,
                     ub_analytic=ub_a, lb_boot=lb_b, ub_boot=ub_b, method=method, lb=float(lb),
                     holdout_pairs=h_n, holdout_delta=h_delta, passed=passed, alarm=alarm, reasons=reasons)


def _crn_complete(records: Iterable[Sequence[Any]]) -> Tuple[List[Any], List[float], List[float], List[bool], int]:
    seen = set()
    cases: List[Any] = []
    cand: List[float] = []
    champ: List[float] = []
    hold: List[bool] = []
    for r in records:
        rec = CrnRecord(*r)
        if rec.triple_id in seen:
            raise ValueError(f'duplicate triple_id {rec.triple_id!r}')
        seen.add(rec.triple_id)
        if _is_missing(rec.candidate) or _is_missing(rec.champion):
            continue                              # a pair with a missing game is dropped whole
        cases.append(rec.case_id)
        cand.append(_check_result(rec.candidate))
        champ.append(_check_result(rec.champion))
        hold.append(bool(rec.holdout))
    return cases, cand, champ, hold, len(seen)


def crn_gate(records: Iterable[Sequence[Any]], *, n_expected: int = G1_PAIRS,
             method: str = 'bootstrap', n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.1,
             min_completion: float = MIN_COMPLETION, holdout_min: float = HOLDOUT_MIN_DELTA,
             require_holdout: bool = True, min_clusters: int = MIN_CLUSTERS) -> CrnResult:
    """G1 promotion gate on records (triple_id, case_id, candidate, champion[, holdout]).

    ``n_expected`` is the number of triples SCHEDULED (default G1_PAIRS), not the number seen:
    blocks lost to a crash or SIGTERM leave no record and must still count as missing."""
    cases, cand, champ, hold, n_seen = _crn_complete(records)
    if n_seen > n_expected:
        raise ValueError(f'{n_seen} triples seen > n_expected {n_expected}')
    return crn_arrays(cases, np.array(cand), np.array(champ), np.array(hold, dtype=bool),
                      n_expected=n_expected, kind='G1', method=method, alpha=alpha, n_boot=n_boot,
                      seed=seed, min_completion=min_completion, holdout_min=holdout_min,
                      require_holdout=require_holdout, min_clusters=min_clusters)


def screen_s(records: Iterable[Sequence[Any]], *, n_expected: int = SCREEN_PAIRS,
             method: str = 'bootstrap', n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.1,
             min_completion: float = MIN_COMPLETION, alarm_delta: float = ROLLBACK_DELTA,
             min_clusters: int = MIN_CLUSTERS) -> CrnResult:
    """Screen S: CRN statistic on SCREEN_PAIRS triples (the scheduled count, default) against
    the champion's stored games. Never passes; ``alarm`` = valid and delta_hat <= ``alarm_delta`` (feeds
    ``rollback_triggered``)."""
    cases, cand, champ, hold, n_seen = _crn_complete(records)
    if n_seen > n_expected:
        raise ValueError(f'{n_seen} triples seen > n_expected {n_expected}')
    return crn_arrays(cases, np.array(cand), np.array(champ), np.array(hold, dtype=bool),
                      n_expected=n_expected, kind='S', method=method, alpha=alpha, n_boot=n_boot,
                      seed=seed, min_completion=min_completion, alarm_delta=alarm_delta,
                      min_clusters=min_clusters)


def rollback_triggered(deltas: Sequence[Optional[float]], threshold: float = ROLLBACK_DELTA,
                       consecutive: int = ROLLBACK_CONSECUTIVE) -> bool:
    """The last ``consecutive`` delta_hat (screens and G1s in time order) are all <= ``threshold``.
    An invalid gate (None) is skipped: it is replayed, and the replay is the next gate of the run
    (same rule as ``monitor.Monitor.observe_gate``)."""
    deltas = [x for x in deltas if x is not None]
    if len(deltas) < consecutive:
        return False
    tail = deltas[-consecutive:]
    return all(x is not None and x <= threshold for x in tail)


# ---------------------------------------------------------------- seat-swapped head-to-head
@dataclass
class SeatSwapResult:
    """G1c / H1 statistic over complete seat-swapped pairs."""
    n_expected: int
    n_pairs: int
    completion: float
    valid: bool
    score: float
    sd: float
    se: float
    lo_normal: float
    hi_normal: float
    lo_boot: Optional[float]
    hi_boot: Optional[float]
    method: str
    lo: float
    passed: bool
    n_clusters: int = 0           # resampling units: pairs, or openings when cluster_ids are given
    clustered: bool = False
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def seat_swap_collect(games: Iterable[Sequence[Any]]) -> Tuple[Dict[Any, float], Dict[Any, Any], int]:
    """(pair scores {case_id: (A on seat 0 + A on seat 1) / 2} over complete pairs, the cluster of
    each complete pair (its games' cluster_id, else the case_id itself), number of distinct pairs
    seen). A pair with a missing game is dropped whole. The two games of a pair must not
    name different cluster_ids."""
    by_case: Dict[Any, Dict[int, Optional[float]]] = {}
    cluster: Dict[Any, Any] = {}
    for r in games:
        g = SeatGame(*r)
        if g.seat not in (0, 1):
            raise ValueError(f'seat must be 0 or 1, got {g.seat!r}')
        slot = by_case.setdefault(g.case_id, {})
        if g.seat in slot:
            raise ValueError(f'duplicate game for case {g.case_id!r} seat {g.seat}')
        slot[g.seat] = None if _is_missing(g.a_result) else _check_result(g.a_result)
        if g.cluster_id is not None:
            prev = cluster.setdefault(g.case_id, g.cluster_id)
            if prev != g.cluster_id:
                raise ValueError(f'pair {g.case_id!r}: cluster_id {prev!r} vs {g.cluster_id!r}')
    pairs = {c: (s[0] + s[1]) / 2.0 for c, s in by_case.items()
             if s.get(0) is not None and s.get(1) is not None}
    return pairs, {c: cluster.get(c, c) for c in pairs}, len(by_case)


def seat_swap_pairs(games: Iterable[Sequence[Any]]) -> Tuple[Dict[Any, float], int]:
    """Pair scores {case_id: (A on seat 0 + A on seat 1) / 2} over complete cases, and the number of
    distinct cases seen. A case with a missing game is dropped whole."""
    pairs, _, n_seen = seat_swap_collect(games)
    return pairs, n_seen


def seat_swap_arrays(pair_scores: np.ndarray, *, n_expected: Optional[int] = None,
                     cluster_ids: Optional[Sequence[Any]] = None, method: str = 'bootstrap',
                     level: float = 0.95, n_boot: int = N_BOOT, seed: int = 0,
                     min_completion: float = MIN_COMPLETION, null: float = 0.5,
                     min_clusters: int = MIN_CLUSTERS) -> SeatSwapResult:
    """Seat-swapped statistic from complete pair scores. Pass iff the two-sided ``level`` CI lower
    bound > 0.5 (bootstrap percentile by default; 'normal' uses mean - z * SE).

    Without ``cluster_ids`` each pair is a resampling unit (SE = sd/sqrt(n)). With ``cluster_ids``
    (the opening of each pair, when openings repeat) the bootstrap resamples openings and the SE is
    the cluster-robust CR1 over openings. ``n_expected`` None means the array is the whole schedule;
    fewer than ``min_clusters`` units makes the gate invalid."""
    if method not in ('bootstrap', 'normal'):
        raise ValueError(f'unknown method {method!r}')
    s = np.asarray(pair_scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError('pair_scores must be 1-D')
    _check_array(s, 'pair_scores', allowed=None)
    if s.size and not (np.all(s >= 0.0) and np.all(s <= 1.0)):
        raise ValueError('pair scores must be in [0, 1]')
    n = int(s.size)
    n_exp = n if n_expected is None else int(n_expected)
    rate, valid = completion(n, n_exp, min_completion)
    reasons: List[str] = []
    if not valid:
        reasons.append(f'incomplete: {n}/{n_exp} pairs ({rate:.4f} < {min_completion})')
    if cluster_ids is None:
        inv, g = np.arange(n), n
    else:
        if len(cluster_ids) != n:
            raise ValueError('cluster_ids must be as long as pair_scores')
        inv, g = _cluster_index(cluster_ids)
    if g < min_clusters:
        valid = False
        reasons.append(f'too few {"openings" if cluster_ids is not None else "pairs"}: {g} < {min_clusters}')
    score = float(s.mean()) if n else float('nan')
    sd = float(s.std(ddof=1)) if n > 1 else float('nan')
    if cluster_ids is None:
        se = sd / math.sqrt(n) if n > 1 else float('nan')
    else:
        se = cluster_se(s, inv, g)
    z = z_quantile(0.5 + level / 2.0)
    lo_n, hi_n = score - z * se, score + z * se
    lo_b = hi_b = None
    if method == 'bootstrap' and n:
        boot = cluster_bootstrap_means(s, inv, g, n_boot, seed)
        q = (1.0 - level) / 2.0
        lo_b, hi_b = (float(x) for x in np.quantile(boot, [q, 1.0 - q]))
    lo = lo_b if method == 'bootstrap' and lo_b is not None else lo_n
    ok = valid
    if not (lo > null):
        reasons.append(f'CI lower bound {lo:.4f} <= {null}')
        ok = False
    return SeatSwapResult(n_expected=n_exp, n_pairs=n, completion=rate, valid=valid, score=score, sd=sd, se=se,
                          lo_normal=lo_n, hi_normal=hi_n, lo_boot=lo_b, hi_boot=hi_b, method=method,
                          lo=float(lo), passed=bool(ok), n_clusters=g, clustered=cluster_ids is not None,
                          reasons=reasons)


def seat_swap_gate(games: Iterable[Sequence[Any]], *, n_expected: int,
                   method: str = 'bootstrap', level: float = 0.95, n_boot: int = N_BOOT, seed: int = 0,
                   min_completion: float = MIN_COMPLETION, cluster: Optional[bool] = None,
                   min_clusters: int = MIN_CLUSTERS) -> SeatSwapResult:
    """G1c / release H1 / G2 on games (case_id, seat, a_result[, cluster_id]): pass iff the
    CI lower bound > 0.5.

    ``n_expected`` (required) is the number of pairs SCHEDULED: G1C_PAIRS, RELEASE_PAIRS or the G2
    size. ``cluster`` None clusters by opening iff any game carries a cluster_id; with no cluster_id
    the result equals the harness's cluster_bootstrap over pairs bit for bit."""
    pairs, clusters, n_seen = seat_swap_collect(games)
    if n_seen > n_expected:
        raise ValueError(f'{n_seen} pairs seen > n_expected {n_expected}')
    keys = sorted(pairs, key=_cluster_key)
    use = any(clusters[k] != k for k in keys) if cluster is None else bool(cluster)
    return seat_swap_arrays(np.array([pairs[k] for k in keys], dtype=np.float64), n_expected=n_expected,
                            cluster_ids=[clusters[k] for k in keys] if use else None, method=method,
                            level=level, n_boot=n_boot, seed=seed, min_completion=min_completion,
                            min_clusters=min_clusters)


# ---------------------------------------------------------------- Wilson
def wilson(k: float, n: int, level: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for k wins in n games (e.g. 13/20 -> [0.43, 0.82])."""
    if n <= 0:
        return None, None
    z = z_quantile(0.5 + level / 2.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- power and false promotion
def crn_sd(rho: float) -> float:
    """SD of a CRN pair difference with win/loss-like outcomes of variance 1/4: sqrt(0.5 (1 - rho))."""
    return math.sqrt(0.5 * (1.0 - rho))


def crn_se(rho: float, n: int) -> float:
    """SE(rho, n) = sqrt(0.5 (1 - rho) / n) of the CRN delta_hat over n pairs."""
    return math.sqrt(0.5 * (1.0 - rho) / n)


def required_delta(se: float, alpha: float = 0.1) -> float:
    """delta_hat needed for the one-sided (1 - alpha) lower bound to clear 0."""
    return _one_sided(alpha) * se


def pass_prob(delta: float, se: float, alpha: float = 0.1) -> float:
    """P(LB > 0) for a true effect ``delta`` (normal approximation)."""
    return norm_cdf(delta / se - _one_sided(alpha))


def prob_below(threshold: float, se: float, delta: float = 0.0) -> float:
    """P(delta_hat <= threshold) for a true ``delta`` (e.g. a screen at or below the rollback threshold)."""
    return norm_cdf((threshold - delta) / se)


def consecutive_rollback_prob(se_first: float, se_second: float, threshold: float = ROLLBACK_DELTA,
                              delta: float = 0.0) -> float:
    """P(two independent consecutive delta_hat <= threshold), e.g. screen then G1."""
    return prob_below(threshold, se_first, delta) * prob_below(threshold, se_second, delta)


def g1_power_table(rhos: Sequence[float] = (0.14, 0.30, 0.45), n: int = G1_PAIRS,
                   deltas: Sequence[float] = (0.03, 0.04, 0.05, 0.08), alpha: float = 0.1) -> List[Dict[str, Any]]:
    """Power table: per rho, SE, the required delta_hat and the pass rate at each true delta."""
    rows = []
    for rho in rhos:
        se = crn_se(rho, n)
        rows.append({'rho': rho, 'sd': crn_sd(rho), 'se': se, 'required_delta': required_delta(se, alpha),
                     'pass': {d: pass_prob(d, se, alpha) for d in deltas}})
    return rows


def seat_swap_se(n: int, sd: float = SEATSWAP_SD) -> float:
    return sd / math.sqrt(n)


def seat_swap_threshold(n: int, sd: float = SEATSWAP_SD, level: float = 0.95) -> float:
    """Pair score needed for the two-sided CI lower bound to clear 0.5."""
    return 0.5 + z_quantile(0.5 + level / 2.0) * seat_swap_se(n, sd)


def seat_swap_pass_prob(true_score: float, n: int, sd: float = SEATSWAP_SD, level: float = 0.95) -> float:
    """P(CI lower bound > 0.5) at a true pair score."""
    return norm_cdf((true_score - seat_swap_threshold(n, sd, level)) / seat_swap_se(n, sd))


def seat_swap_game_ratio(rho_crn: float, sd: float = SEATSWAP_SD) -> float:
    """Games seat-swap needs per CRN game for equal power on an effect delta (seat-swap moves the pair
    score by delta/2): (2 sd)^2 / (0.5 (1 - rho))."""
    return (2.0 * sd) ** 2 / (0.5 * (1.0 - rho_crn))


def false_promotion_rate(alpha_g1: float = 0.1, alpha_g1c: float = 0.025) -> float:
    """At equal strength G1 then G1c on fresh games: alpha_g1 * alpha_g1c."""
    return alpha_g1 * alpha_g1c


# ---------------------------------------------------------------- optional SPRT for G1
@dataclass
class SprtResult:
    decision: str            # 'H1' (promote side), 'H0', 'cap' (truncated, not passed), 'continue'
    n: int
    llr: float
    lower: float
    upper: float
    sigma: float
    passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sprt_bounds(alpha: float = 0.1, beta: float = 0.20) -> Tuple[float, float]:
    """Wald bounds (log B, log A) = (log(beta / (1 - alpha)), log((1 - beta) / alpha))."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def sprt_g1(diffs: Sequence[float], *, delta1: float = SPRT_DELTA1, alpha: float = 0.1, beta: float = 0.20,
            cap: int = G1_PAIRS, sigma: Optional[float] = None, rho: Optional[float] = None,
            min_pairs: int = 64) -> SprtResult:
    """Truncated SPRT on CRN pair differences in play order, H0 delta = 0 vs H1 delta = delta1,
    normal approximation: LLR_n = (delta1 / sigma^2) (S_n - n delta1 / 2).

    sigma: given, else crn_sd(rho) if rho is given, else the running sample SD (ddof 1, floored
    at 0.05). No decision before ``min_pairs``. Reaching ``cap`` without a crossing returns 'cap'
    and does not pass (conservative: the false-pass rate stays below the Wald bound). Fewer than
    ``cap`` pairs and no crossing returns 'continue'. The held-out sub-panel check of G1 still
    applies on top of an 'H1' decision."""
    d = np.asarray(diffs, dtype=np.float64)[:cap]
    lo, hi = sprt_bounds(alpha, beta)
    n_all = d.size
    if n_all == 0:
        return SprtResult('continue', 0, 0.0, lo, hi, float('nan'), False)
    k = np.arange(1, n_all + 1, dtype=np.float64)
    s = np.cumsum(d)
    if sigma is None and rho is not None:
        sigma = crn_sd(rho)
    if sigma is not None:
        var = np.full(n_all, float(sigma) ** 2)
    else:
        ss = np.cumsum(d * d)
        with np.errstate(invalid='ignore', divide='ignore'):
            var = (ss - s * s / k) / np.maximum(k - 1.0, 1.0)
        var = np.maximum(var, 0.05 ** 2)
    llr = delta1 / var * (s - k * delta1 / 2.0)
    active = k >= min_pairs
    cross = np.flatnonzero(active & ((llr >= hi) | (llr <= lo)))
    if cross.size:
        i = int(cross[0])
        dec = 'H1' if llr[i] >= hi else 'H0'
        return SprtResult(dec, i + 1, float(llr[i]), lo, hi, float(math.sqrt(var[i])), dec == 'H1')
    dec = 'cap' if n_all >= cap else 'continue'
    return SprtResult(dec, int(n_all), float(llr[-1]), lo, hi, float(math.sqrt(var[-1])), False)


# ---------------------------------------------------------------- Beta posterior (PFSP win rates)
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularised incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) (the Beta(a, b) CDF)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


@dataclass(frozen=True)
class BetaPosterior:
    """Beta(a, b) posterior of a snapshot's win rate (draws count half a win and half a loss)."""
    a: float = 1.0
    b: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.a) and math.isfinite(self.b) and self.a > 0.0 and self.b > 0.0):
            raise ValueError(f'Beta parameters must be finite and > 0, got ({self.a}, {self.b})')

    @classmethod
    def from_results(cls, results: Iterable[float], prior: Tuple[float, float] = (1.0, 1.0)) -> 'BetaPosterior':
        a, b = prior
        for r in results:
            r = _check_result(r)
            a += r
            b += 1.0 - r
        return cls(a, b)

    def update(self, result: float, weight: float = 1.0) -> 'BetaPosterior':
        r = _check_result(result)
        if not (math.isfinite(weight) and weight >= 0.0):
            raise ValueError(f'weight must be finite and >= 0, got {weight}')
        return BetaPosterior(self.a + weight * r, self.b + weight * (1.0 - r))

    def decay(self, gamma: float, prior: Tuple[float, float] = (1.0, 1.0)) -> 'BetaPosterior':
        """Shrink the evidence toward the prior by ``gamma`` in [0, 1] (for a moving opponent)."""
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f'gamma must be in [0, 1], got {gamma}')
        a0, b0 = prior
        return BetaPosterior(a0 + gamma * (self.a - a0), b0 + gamma * (self.b - b0))

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def var(self) -> float:
        s = self.a + self.b
        return self.a * self.b / (s * s * (s + 1.0))

    def cdf(self, x: float) -> float:
        return betainc(self.a, self.b, x)

    def quantile(self, q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError('q must be in [0, 1]')
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self.cdf(mid) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Equal-tailed credible interval."""
        t = (1.0 - level) / 2.0
        return self.quantile(t), self.quantile(1.0 - t)

    def sample(self, rng: np.random.Generator, size: Optional[int] = None) -> Any:
        """Thompson draw(s) of the win rate."""
        return rng.beta(self.a, self.b, size)


__all__ = ['G1_PAIRS', 'G1C_PAIRS', 'SCREEN_PAIRS', 'RELEASE_PAIRS', 'N_BOOT', 'MIN_COMPLETION',
           'HOLDOUT_MIN_DELTA', 'ROLLBACK_DELTA', 'ROLLBACK_CONSECUTIVE', 'SEATSWAP_SD', 'SPRT_DELTA1',
           'MIN_CLUSTERS', 'CrnRecord', 'SeatGame', 'CrnResult', 'SeatSwapResult', 'SprtResult',
           'BetaPosterior', 'z_quantile', 'norm_cdf', 'completion', 'cluster_se', 'cluster_bootstrap_means',
           'crn_arrays', 'crn_gate', 'screen_s', 'rollback_triggered', 'seat_swap_collect',
           'seat_swap_pairs', 'seat_swap_arrays',
           'seat_swap_gate', 'wilson', 'crn_sd', 'crn_se', 'required_delta', 'pass_prob', 'prob_below',
           'consecutive_rollback_prob', 'g1_power_table', 'seat_swap_se', 'seat_swap_threshold',
           'seat_swap_pass_prob', 'seat_swap_game_ratio', 'false_promotion_rate', 'sprt_bounds', 'sprt_g1',
           'betainc']
