"""Lag-1 overlap collection (``policy.rollout_lag = 1``): the learner-side math and the learner monitors.

``policy.rollout_lag`` may change at block boundaries; any overlap alarm sends the run back to lag 0 at the next
block boundary (``pipeline.revert_action``).

Notation (one update u, learner at θ_{u-1}):

* μ        the stored behaviour log-prob ``mu`` of the executed joint action (the actor wrote it; at lag 1 the
           round was played by θ_{u-2}, the learner's own weights at the start of update u-1);
* π_prox   the learner's recompute with θ_{u-1} at update start (``logp_prox``; the decoupled-PPO proximal
           policy, the ratio r = π_θ / π_prox starts at 1 every update);
* w        = exp(clamp(log π_prox − μ, ±20)), exactly 1 on forced rows (no choice) and on rows the learner
           cannot score (invalid rows, which stop the update anyway);
* ρ        = min(ρ̄, w)  (ρ̄ = ``policy.is_ratio_cap_lag1``): detached weight on the per-row clipped objective;
* c        = min(c̄, w)  (c̄ = ``policy.gae_trace_cap_lag1``): the V-trace trace and δ weight of the critic target.

V-trace style per-tick GAE over one seat-episode (:func:`vtrace_scan`, gae_scan's operation order with c inserted,
so c ≡ 1 reproduces ``gae.gae_scan`` bit for bit):

    δ_k = r_k + g_k V_{k+1} (1 − d_k) − V_k
    u_k = c_k δ_k + (((g_k l_k) c_k) (1 − d_k)) u_{k+1}           R_k = V_k + u_k   (critic target)
    Â_k = δ_k + ((g_k l_k) (1 − d_k)) u_{k+1}                                      (policy advantage)

What the correction leaves out: uncommitted waits (not stored) and the co-trained opponent seat get no
correction; ``is/wait_dlogp_committed_*`` indicates the size of that gap.

Rules (only in ``rollout_lag = 1`` runs; :func:`check_learner`):

* O1  data version not 0 or 1 -> :class:`LagError` before any work (:func:`lag_of`);
* O2  the parent forward's ``mu_dlogp_max/<type>`` -> the existing ``actor_learner_mismatch`` stop (learner);
* O3  truncation share over {overall, play, ability} (a type counts with ≥ ``MIN_TYPE_ROWS`` live rows):
      a scope above its alarm line (``TRUNC_ALARM``; the ability type ``TRUNC_ALARM_BY_TYPE``) -> alarm; the
      largest share > ``TRUNC_STOP`` -> pre-step stop ``overlap_is_truncation`` (no optimiser step);
* O4  normalised effective sample min over {overall, play, ability} (same count rule) < ``ESS_ALARM`` alarm.

O5-O8 are driver rules (``amini.ppo.pipeline``). Nothing in this module is imported by a lag-0 path.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

OVERLAP_SCHEMA = "ppo-v7-overlap.1"
TRUNC_ALARM = 0.1             # O3 alarm line, per row type (example value)
# Per-type alarm lines that replace TRUNC_ALARM for that type, e.g. for a row type with few rows whose share moves
# in large steps (example value). The stop line below is the same for every scope.
TRUNC_ALARM_BY_TYPE: Dict[str, float] = {"ability": 0.2}
TRUNC_STOP = 0.3              # O3 pre-step stop (example value)
ESS_ALARM = 0.5               # O4 alarm (example value)
MIN_TYPE_ROWS = 100           # a row type enters O3 / O4 only with at least this many live rows (example value)
IS_LOG_CLAMP = 20.0           # |log π_prox − μ| bound before exp (losses.LOG_RATIO_CLAMP)
TRACE_WINDOW = 25             # is/trace_prod_25: rows per window
STOP_NAME = "overlap_is_truncation"
ARM_SPECIFIC_STOPS: Tuple[str, ...] = (STOP_NAME,)   # stop reasons that belong to the overlap path
ROW_TYPES: Tuple[str, ...] = ("wait", "play", "ability")   # family codes 0 / 1 / 2 (schema FAMILY_*)
RULE_TYPES: Tuple[str, ...] = ("play", "ability")           # the per-type scopes of O3 / O4 besides overall
HEAD_NAMES: Tuple[str, ...] = ("family", "slot", "cell", "ability")     # order of the stored mu_terms
IS_PREFIX = "is/"


class LagError(ValueError):
    """O1: the data's policy version is neither the learner's (lag 0) nor its recorded parent (lag 1)."""


# =========================================================================== version
def lag_of(data_digest: str, own: str, prev_start: Optional[str], prev_end: Optional[str]) -> int:
    """0 when the data was played by the learner's own weights; 1 only when it was played by the
    weights the learner held at the start of its previous update (``prev_start``) AND the learner still holds the
    weights that update ended with (``prev_end``), so a rollback, reload or weight swap in between makes lag 1
    impossible; anything else raises :class:`LagError` (O1)."""
    d = str(data_digest)
    if d == str(own):
        return 0
    if prev_start is not None and prev_end is not None and d == str(prev_start) and str(own) == str(prev_end):
        return 1
    why = ("no previous update recorded (the first update of a block, a resume or a rollback is a fill)"
           if prev_start is None else
           f"previous update {str(prev_start)[:12]} -> {str(prev_end)[:12]}" +
           (" but the learner's weights changed since" if d == str(prev_start) else ""))
    raise LagError(f"O1: shards played by policy {d[:12]}; the learner holds {str(own)[:12]} ({why}); only lag 0 "
                   "or lag 1 data may be learned")


def lag_of_chain(data_digest: str, own: str, chain: Sequence[Tuple[Optional[str], Optional[str]]]) -> int:
    """For ``policy.rollout_lag = 2``: :func:`lag_of` over the learner's last updates. ``chain[i]`` is the
    (start digest, end digest) of the update i + 1 steps back, newest first. 0 when the data was played by the
    learner's own weights; k when it was played by the weights the learner held at the start of the update k steps
    back AND the updates since then form an unbroken chain up to the weights it holds now (``chain[0]`` ends at
    ``own`` and every older entry ends where the next one started), so a rollback, a reload or a weight swap
    anywhere in between makes that lag impossible. Anything else raises :class:`LagError` (O1). With one entry
    this is :func:`lag_of`."""
    d = str(data_digest)
    if d == str(own):
        return 0
    end = str(own)
    for k, (start, stop) in enumerate(chain, 1):
        if start is None or stop is None or str(stop) != end:
            break
        if d == str(start):
            return k
        end = str(start)
    if not chain or chain[0][0] is None:
        why = "no previous update recorded (the first update of a block, a resume or a rollback is a fill)"
    else:
        why = "previous updates " + ", ".join(f"{str(a)[:12]} -> {str(b)[:12]}" for a, b in chain)
    raise LagError(f"O1: shards played by policy {d[:12]}; the learner holds {str(own)[:12]} ({why}); only data of "
                   f"lag 0 to {max(1, len(chain))} may be learned")


# =========================================================================== V-trace GAE
class VtraceResult(NamedTuple):
    advantages: np.ndarray      # Â_k (policy advantage)
    targets: np.ndarray         # R_k = V_k + u_k (critic targets)
    u: np.ndarray               # u_k (the corrected λ-return minus V)


def vtrace_scan(rewards: Any, values: Any, discounts: Any, lambdas: Any, traces: Any, dones: Any) -> VtraceResult:
    """Backward V-trace / GAE recursion (module docstring) on 1-D or 2-D [B, K] arrays, with gae_scan's input
    checks and operation order; ``traces`` c_k must be finite and ≥ 0. With c ≡ 1 the result equals
    ``gae.gae_scan`` bit for bit (advantages, and targets = advantages + values)."""
    r, v, g, lam, c, d = (np.asarray(x, dtype=np.float64) for x in (rewards, values, discounts, lambdas, traces,
                                                                   dones))
    if not (r.shape == v.shape == g.shape == lam.shape == c.shape == d.shape) or r.ndim not in (1, 2) \
            or r.shape[-1] == 0:
        raise ValueError(f"shape mismatch or empty: {r.shape} {v.shape} {g.shape} {lam.shape} {c.shape} {d.shape}")
    for name, x in (("rewards", r), ("values", v), ("discounts", g), ("lambdas", lam), ("traces", c)):
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be finite")
    if np.any(c < 0.0):
        raise ValueError("traces must be >= 0")
    if not np.all((d == 0.0) | (d == 1.0)):
        raise ValueError("dones must be 0/1")
    if np.any(d[..., -1] != 1.0):
        raise ValueError("the last row must be terminal (truncated episodes are dropped)")
    nd = 1.0 - d
    adv = np.zeros_like(r)
    uu = np.zeros_like(r)
    nxt_v = np.zeros(r.shape[:-1])
    nxt_u = np.zeros(r.shape[:-1])
    for k in range(r.shape[-1] - 1, -1, -1):
        delta = r[..., k] + g[..., k] * nxt_v * nd[..., k] - v[..., k]
        a = delta + g[..., k] * lam[..., k] * nd[..., k] * nxt_u
        u = c[..., k] * delta + g[..., k] * lam[..., k] * c[..., k] * nd[..., k] * nxt_u
        adv[..., k] = a
        uu[..., k] = u
        nxt_v, nxt_u = v[..., k], u
    return VtraceResult(adv, uu + v, uu)


def gae_vtrace_ragged(rewards: Any, values: Any, ticks: Any, lengths: Any, traces: Any, *,
                      gamma_tick: float, lambda_tick: float) -> VtraceResult:
    """:func:`vtrace_scan` over flat arrays of consecutive episodes, built exactly as ``gae.gae_ragged`` /
    ``gae.gae_padded`` build the per-tick factors (g = γ_tick^Δt, l = λ_tick^Δt, last row terminal)."""
    from amini.ppo import gae as G

    n = G._int_lengths(lengths)
    r = np.asarray(rewards, dtype=np.float64)
    if n.ndim != 1 or n.size == 0 or np.any(n < 1) or int(n.sum()) != r.shape[0]:
        raise ValueError("lengths must be positive and sum to the number of rows")
    rows = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    ep = np.repeat(np.arange(n.size), n)
    k = int(n.max())
    pad = lambda x: G._scatter(np.asarray(x, dtype=np.float64), ep, rows, n.size, k)   # noqa: E731
    rp, vp, tp = pad(r), pad(values), pad(ticks)
    cp = np.ones((n.size, k))
    cp[ep, rows] = np.asarray(traces, dtype=np.float64)
    G._check_factor("gamma_tick", gamma_tick)
    G._check_factor("lambda_tick", lambda_tick)
    valid = np.arange(k)[None, :] < n[:, None]
    dt = np.zeros_like(tp)
    for b in range(n.size):
        dt[b, :n[b]] = G.tick_intervals(tp[b, :n[b]])
    d = np.zeros_like(rp)
    d[np.arange(n.size), n - 1] = 1.0
    d[~valid] = 1.0
    out = vtrace_scan(np.where(valid, rp, 0.0), np.where(valid, vp, 0.0), float(gamma_tick) ** dt,
                      float(lambda_tick) ** dt, cp, d)
    adv = np.where(valid, out.advantages, 0.0)
    tgt = np.where(valid, out.targets, 0.0)
    uu = np.where(valid, out.u, 0.0)
    return VtraceResult(adv[ep, rows], tgt[ep, rows], uu[ep, rows])


# =========================================================================== IS weights
def is_weights(logp_prox: Any, mu: Any, forced: Any, *, rho_bar: float, c_bar: float, clamp: float = IS_LOG_CLAMP,
               neutral: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (w, ρ, c) as float64 arrays (module docstring). ``neutral``: extra rows forced to w = 1 (rows the
    learner cannot score, e.g. invalid rows; they carry policy weight 0 and stop the update anyway). A
    non-finite log-ratio on any other row raises ValueError."""
    lp = np.asarray(logp_prox, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    f = np.asarray(forced, dtype=bool)
    if not (lp.shape == m.shape == f.shape) or lp.ndim != 1:
        raise ValueError(f"shape mismatch: {lp.shape} {m.shape} {f.shape}")
    if not (float(rho_bar) >= 1.0 and 0.0 < float(c_bar) <= float(rho_bar)):
        raise ValueError(f"need rho_bar >= 1 and 0 < c_bar <= rho_bar, got {rho_bar}, {c_bar}")
    one = f.copy()
    if neutral is not None:
        one |= np.asarray(neutral, dtype=bool)
    x = np.where(one, 0.0, lp - m)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite log(pi_prox / mu) on a scored row")
    w = np.exp(np.clip(x, -float(clamp), float(clamp)))
    w[one] = 1.0
    return w, np.minimum(float(rho_bar), w), np.minimum(float(c_bar), w)


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    s = float(w.sum())
    return float((w * x).sum() / s) if s > 0 else float("nan")


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q)) if x.size else float("nan")


def _ess_norm(w: np.ndarray, rho: np.ndarray) -> float:
    """(Σ Wρ)² / Σ W²ρ² over (Σ W)² / Σ W²; 1 = no loss of effective sample."""
    if not float(w.sum()) > 0:
        return float("nan")
    wr = w * rho
    den = float((wr * wr).sum())
    base = float(w.sum()) ** 2 / float((w * w).sum())
    return (float(wr.sum()) ** 2 / den) / base if den > 0 else float("nan")


def k3_np(x: np.ndarray) -> np.ndarray:
    """k3 estimator (e^x − 1) − x ≥ 0 of a log-ratio array (losses.k3 in numpy)."""
    return np.clip(np.expm1(x) - x, 0.0, None)


def trace_prod(c: np.ndarray, lengths: Sequence[int], window: int = TRACE_WINDOW) -> float:
    """Mean over every window of ``window`` consecutive rows inside one episode of the product of c (NaN when no
    episode is that long)."""
    lc = np.log(np.maximum(np.asarray(c, dtype=np.float64), 1e-300))
    vals: List[np.ndarray] = []
    off = 0
    for n in lengths:
        n = int(n)
        if n >= window:
            cs = np.concatenate([[0.0], np.cumsum(lc[off:off + n])])
            vals.append(np.exp(cs[window:] - cs[:-window]))
        off += n
    return float(np.concatenate(vals).mean()) if vals else float("nan")


def is_metrics(*, w: np.ndarray, rho: np.ndarray, c: np.ndarray, weight: np.ndarray, value_weight: np.ndarray,
               row_type: np.ndarray, valid: np.ndarray, logp_prox: np.ndarray, mu: np.ndarray, dt: np.ndarray,
               lengths: Sequence[int], lambda_tick: float, rho_bar: float, c_bar: float,
               adv_vt: np.ndarray, adv_plain: np.ndarray, tgt_vt: np.ndarray, tgt_plain: np.ndarray,
               head_prox: Optional[np.ndarray] = None, head_mu: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Every ``is/`` number of one update (plain floats / ints). ``weight`` W = the unified policy
    weight (HT × seat × [not forced], 0 on invalid rows); live rows are W > 0. ``dt`` = tick interval to the next
    row (0 on an episode's last row). ``head_prox`` / ``head_mu`` [n, 4]: per-head executed log-prob terms
    (family, slot, cell, ability) of π_prox and of the actor (``mu_terms``).

    ``is/first_step_kl`` (logged every update) is the W-weighted k3 of log(π_prox / μ): the KL(μ ‖ π_prox)
    estimate the first optimiser step of the update starts from (the behaviour policy vs the learner's
    starting policy; ≈ 0 at lag 0), with ``/<type>`` per row type."""
    W = np.asarray(weight, dtype=np.float64)
    Wv = np.asarray(value_weight, dtype=np.float64)
    rt = np.asarray(row_type)
    ok = np.asarray(valid, dtype=bool)
    live = W > 0
    x = np.where(ok, np.asarray(logp_prox, np.float64) - np.asarray(mu, np.float64), 0.0)
    xc = np.clip(x, -IS_LOG_CLAMP, IS_LOG_CLAMP)
    out: Dict[str, Any] = {}
    wl = w[live]
    out["is/ratio_mean"] = _wmean(w, W)
    out["is/ratio_p01"], out["is/ratio_p50"], out["is/ratio_p99"] = _q(wl, .01), _q(wl, .5), _q(wl, .99)
    out["is/ratio_min"] = float(wl.min()) if wl.size else float("nan")
    out["is/ratio_max"] = float(wl.max()) if wl.size else float("nan")
    trunc = (w > float(rho_bar)).astype(np.float64)
    cut = (w < float(c_bar)).astype(np.float64)
    out["is/trunc_share"] = _wmean(trunc, W)
    out["is/trunc_n"] = int(live.sum())
    k3 = k3_np(xc)
    out["is/first_step_kl"] = _wmean(k3, W)
    out["is/ess_norm"] = _ess_norm(W, rho)
    for code, name in enumerate(ROW_TYPES):
        sel = rt == code
        Wt = np.where(sel, W, 0.0)
        out[f"is/trunc_share/{name}"] = _wmean(trunc, Wt)
        out[f"is/trunc_n/{name}"] = int((live & sel).sum())
        out[f"is/ess_norm/{name}"] = _ess_norm(W[sel], rho[sel])
        out[f"is/first_step_kl/{name}"] = _wmean(k3, Wt)
        d = np.abs(x[sel & ok])
        out[f"is/prox_dlogp_max/{name}"] = float(d.max()) if d.size else 0.0
    out["is/trace_cut_share"] = _wmean(cut, W)
    logc = np.log(np.maximum(c, 1e-300))
    out["is/log_c_mean"] = _wmean(logc, W)
    swdt = float((W * dt).sum())
    out["is/lambda_tick_eff"] = (float(lambda_tick) * math.exp(float((W * logc).sum()) / swdt)
                                 if swdt > 0 else float("nan"))
    out["is/rho_mean"] = _wmean(rho, W)
    out["is/c_mean"] = _wmean(c, W)
    out["is/trace_prod_25"] = trace_prod(c, lengths)
    out["is/w_max"] = float((W * rho).max()) if W.size else float("nan")
    wr = W * rho
    out["is/joint_k3_isw_start"] = _wmean(k3, wr)       # ρ-weighted first-step KL (diagnostic)
    if head_prox is not None and head_mu is not None:
        hp, hm = np.asarray(head_prox, np.float64), np.asarray(head_mu, np.float64)
        use = {"family": live, "slot": live & (rt == 1), "cell": live & (rt == 1), "ability": live & (rt == 2)}
        for j, h in enumerate(HEAD_NAMES):
            s = use[h] & ok
            hx = np.clip(hp[s, j] - hm[s, j], -IS_LOG_CLAMP, IS_LOG_CLAMP)
            out[f"is/head_ratio_p99/{h}"] = _q(np.exp(hx), .99)
    sd = math.sqrt(max(_wmean((adv_plain - _wmean(adv_plain, W)) ** 2, W), 0.0)) if float(W.sum()) > 0 else 0.0
    out["is/adv_shift"] = _wmean(adv_vt - adv_plain, W) / sd if sd > 0 else float("nan")
    sdt = math.sqrt(max(_wmean((tgt_plain - _wmean(tgt_plain, Wv)) ** 2, Wv), 0.0)) if float(Wv.sum()) > 0 else 0.0
    out["is/target_shift"] = _wmean(tgt_vt - tgt_plain, Wv) / sdt if sdt > 0 else float("nan")
    wd = np.abs(x[live & (rt == 0)])
    out["is/wait_dlogp_committed_mean"] = float(wd.mean()) if wd.size else 0.0
    out["is/wait_dlogp_committed_p99"] = _q(wd, .99) if wd.size else 0.0
    return out


# =========================================================================== rules O3 / O4
def _scoped(metrics: Mapping[str, Any], key: str) -> List[Tuple[str, float]]:
    """(scope, value) for overall and every RULE_TYPES type with at least MIN_TYPE_ROWS live rows; non-finite
    values are reported as NaN so the caller can flag them."""
    out = [("overall", float(metrics.get(key, float("nan"))))]
    for t in RULE_TYPES:
        if int(metrics.get(f"is/trunc_n/{t}", 0) or 0) >= MIN_TYPE_ROWS:
            out.append((t, float(metrics.get(f"{key}/{t}", float("nan")))))
    return out


def trunc_alarm_of(scope: str) -> float:
    """The O3 alarm line of a scope ('overall' or a row type)."""
    return float(TRUNC_ALARM_BY_TYPE.get(scope, TRUNC_ALARM))


def check_learner(metrics: Mapping[str, Any]) -> Tuple[List[str], Optional[str]]:
    """O3 / O4 on one update's ``is/`` numbers -> (alarms, pre-step stop reason or None). A missing or non-finite
    O3 / O4 number is itself an alarm (the rule cannot be evaluated)."""
    alarms: List[str] = []
    stop: Optional[str] = None
    tr = _scoped(metrics, "is/trunc_share")
    bad = [s for s, v in tr if not math.isfinite(v)]
    if bad:
        alarms.append(f"O3: truncation share not finite ({', '.join(bad)})")
    fin = [(s, v) for s, v in tr if math.isfinite(v)]
    if fin:
        s, v = max(fin, key=lambda t: t[1])
        if v > TRUNC_STOP:
            stop = f"O3: IS truncation share {v:.4f} ({s}) > {TRUNC_STOP:g} (w > rho_bar on too many rows)"
            alarms.append(stop)
        else:                                    # each scope against its own line; the one furthest over is named
            over = [(s, v) for s, v in fin if v > trunc_alarm_of(s)]
            if over:
                s, v = max(over, key=lambda t: t[1] / trunc_alarm_of(t[0]))
                alarms.append(f"O3: IS truncation share {v:.4f} ({s}) > {trunc_alarm_of(s):g}")
    es = _scoped(metrics, "is/ess_norm")
    bad = [s for s, v in es if not math.isfinite(v)]
    if bad:
        alarms.append(f"O4: effective sample not finite ({', '.join(bad)})")
    fin = [(s, v) for s, v in es if math.isfinite(v)]
    if fin:
        s, v = min(fin, key=lambda t: t[1])
        if v < ESS_ALARM:
            alarms.append(f"O4: normalised effective sample {v:.4f} ({s}) < {ESS_ALARM:g}")
    return alarms, stop
