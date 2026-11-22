"""PPO losses and update metrics for aminiformer.

Pure torch functions over per-row tensors; no model code. The caller (the learner) supplies
per-row joint log-probabilities of the executed action (``executed_logp`` in ``ppo.logprob``),
per-row per-head KL / entropy values, advantages, value outputs and targets, and the **unified
policy weight** of every row:

    w = HT weight (1 on action rows, 1/kappa on committed waits) x seat weight x [non-forced row]

Forced rows (and padding) carry w = 0 and are excluded from every term and metric. Exclusion is done
by sanitising the inputs of w == 0 rows before any arithmetic (``torch.where`` on the inputs, not on
the output), so a NaN / inf / ``LOG_ZERO`` value on a dead row changes neither the loss, nor the
gradient, nor the early-stop metrics (0 * NaN would otherwise be NaN, forward and backward). A
non-finite value on a LIVE row still propagates: that NaN is the learner's skip signal.

Head names (``HEADS``): ``family`` (the gate), ``slot``, ``cells``, ``ability``. ``canonical_head``
also accepts the ``ppo.logprob`` keys (``kl_*`` of ``head_kl``, ``ent_*`` of ``head_entropy``) and
the run-configuration keys (``gate``, ``cells``), so the learner can pass either dict as is.

Denominator convention (one rule for every policy term)
-------------------------------------------------------
Every policy-side term is ``sum_rows w * x / D``. ``D`` is the surrogate denominator, the sum of
the unified weights over the rows the optimiser step covers.

* ``denom=None`` (local D): D is the sum of ``weight`` over the rows passed in. Use this when one
  call sees the whole optimiser step (one minibatch, no micro-batching). The term is then the
  weighted mean, as in standard PPO.
* ``denom=D`` (global D): the caller passes the step-wide sum (for example the minibatch sum when
  the minibatch is split into micro-batches, as a micro-batching trainer does with its step-wide
  denominators). Each micro-batch call then returns its *share* ``sum_mb w*x / D``; the shares add
  up to the step's weighted mean, so the accumulated gradient equals the full-step gradient no
  matter how the rows are split. A micro-batch result alone is NOT a mean (it is scaled by
  ``D_mb / D``). Do not pass an epoch- or update-wide D for a per-minibatch step: that shrinks each
  step's gradient by the number of minibatches (an implicit learning-rate change).
  A global D must cover the call's rows: if D < sum(w) of the call (beyond rounding,
  ``DENOM_RTOL``), the term returns NaN (no host sync) so the learner's NaN guard catches a wrong D
  (D = 0 with live rows, a D from the wrong rows) instead of silently zeroing or inflating the step.

The KL anchor and the optional BC term use the SAME D as the surrogate, never their own row
counts: per-head "own-row" means would make the anchors of heads that live on few rows (slot,
cells) much stronger than the same beta on the joint KL. Own-row means are still reported, for
monitoring only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

# --------------------------------------------------------------------------- constants
CLIP_EPS = 0.25                # PPO clip on the joint ratio (example value)
DUAL_CLIP = 2.5                # dual-clip c for A < 0 (arXiv 1912.09729; example value)
KL_BETA = 0.02                 # anchor KL coefficient (example value; a run configuration may schedule it)
BC_COEF = 0.0                  # BC is off by default
ENTROPY_COEF = 0.0             # entropy bonus off by default
LOG_RATIO_CLAMP = 20.0         # |log r| bound before exp (see clipped_surrogate)
DENOM_RTOL = 1e-3              # a global D below sum(w) * (1 - DENOM_RTOL) is an error (NaN)
TINY_LOG_RATIO = -10.0         # example value: log(pi_theta / pi_prox) below this is a "tiny" ratio

# Early stop per epoch (example values). Strictly greater than trips.
EARLY_STOP = {"joint_k3": 0.02, "play_k3": 0.1, "gate_kl": 0.01}

# Canonical head names and the aliases used by ppo.logprob (kl_*, ent_*) and the run configuration (gate, cells).
HEADS = ("family", "slot", "cells", "ability")
HEAD_ALIASES = {
    "family": "family", "gate": "family", "kl_family": "family", "ent_family": "family",
    "slot": "slot", "kl_slot": "slot", "ent_slot": "slot",
    "cells": "cells", "cell": "cells", "kl_cell": "cells", "kl_cells": "cells", "ent_cell": "cells",
    "ability": "ability", "kl_ability": "ability", "ent_ability": "ability",
}

# Update-0 identity tolerances (example values).
IDENTITY_RATIO_TOL = 1e-3      # |r - 1| per row type, fp32 learner vs itself
IDENTITY_KL_TOL = 1e-5         # KL_h(theta_0 || ref) for every head

# Row types for the per-type identity check (wait / play / ability never pooled).
ROW_WAIT, ROW_PLAY, ROW_ABILITY = 0, 1, 2
ROW_TYPE_NAMES = {ROW_WAIT: "wait", ROW_PLAY: "play", ROW_ABILITY: "ability"}

Tensor = torch.Tensor
HeadRows = Tuple[Tensor, Tensor]          # (per-row value, per-row unified weight) on the head's own rows


# --------------------------------------------------------------------------- helpers
def _up(x: Tensor) -> Tensor:
    """At least fp32; fp64 stays fp64 (the log-prob math of ``ppo.logprob`` is fp64)."""
    return x if x.dtype == torch.float64 else x.float()


def _check_same_shape(**ts: Tensor) -> None:
    shapes = {k: tuple(v.shape) for k, v in ts.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"per-row tensors must share one shape, got {shapes}")


def canonical_head(name: str) -> str:
    """Canonical head name (``HEADS``) for a logprob / config alias; unknown names pass through."""
    return HEAD_ALIASES.get(name, name)


def _canonical_keys(d: Mapping[str, object], what: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        c = canonical_head(k)
        if c in out:
            raise ValueError(f"{what}: two keys map to head {c!r} ({list(d)})")
        out[c] = v
    return out


def _coef_for(coef: Union[float, Mapping[str, float]], head: str, what: str) -> float:
    if not isinstance(coef, Mapping):
        return float(coef)
    cm = _canonical_keys(coef, what)
    if head not in cm:
        raise KeyError(f"{what}: no coefficient for head {head!r} (have {sorted(cm)})")
    return float(cm[head])


def _live(weight: Tensor) -> Tensor:
    """Rows that count: w != 0 (a NaN weight stays live, so it propagates)."""
    return weight != 0


def _dead_to(x: Tensor, live: Tensor, fill: Union[float, Tensor] = 0.0) -> Tensor:
    """x on live rows, ``fill`` on dead rows (forward AND backward never touch the dead value)."""
    f = fill if torch.is_tensor(fill) else torch.full_like(x, fill)
    return torch.where(live, x, f)


def surrogate_denominator(weight: Tensor) -> Tensor:
    """D = sum of unified policy weights over the rows of one optimiser step."""
    return _up(weight).sum()


def _resolve_denom(weight: Tensor, denom: Optional[Union[Tensor, float]]) -> Tuple[Tensor, bool]:
    """(D, check): D as a 0-dim tensor; ``check`` is True for a caller-supplied (global) D."""
    w = _up(weight)
    if denom is None:
        return w.sum(), False
    d = torch.as_tensor(denom.detach() if torch.is_tensor(denom) else denom, dtype=w.dtype, device=w.device)
    if d.dim() != 0:
        raise ValueError("denom must be a scalar")
    return d, True


def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    """num / den, 0 when den == 0 (an all-forced step contributes nothing, never NaN)."""
    return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(num))


def _normalise(num: Tensor, w_sum: Tensor, d: Tensor, check: bool) -> Tensor:
    """num / D; NaN (no host sync) when a global D is smaller than the call's weight sum."""
    out = _safe_div(num, d)
    if check:
        bad = d < w_sum.to(d.dtype) * (1.0 - DENOM_RTOL)
        out = torch.where(bad, torch.full_like(out, float("nan")), out)
    return out


def weighted_mean(x: Tensor, weight: Tensor) -> Tensor:
    """sum(w * x) / sum(w) over rows with w != 0 (dead-row values ignored, even NaN); 0 when
    sum(w) == 0."""
    x = _up(x)
    w = _up(weight).to(x.dtype)
    x = _dead_to(x, _live(w))
    return _safe_div((w * x).sum(), w.sum())


# --------------------------------------------------------------------------- surrogate
def clipped_objective(ratio: Tensor, advantage: Tensor, *, eps: float = CLIP_EPS,
                      dual_clip: Optional[float] = DUAL_CLIP) -> Tensor:
    """Per-row PPO objective (to maximise): min(r A, clip(r, 1-eps, 1+eps) A), and for A < 0
    additionally max(that, c A) (dual-clip, arXiv 1912.09729). ``dual_clip=None`` disables it."""
    if dual_clip is not None and not dual_clip > 1.0 + eps:
        raise ValueError(f"dual_clip c must exceed 1 + eps, got c={dual_clip}, eps={eps}")
    a = advantage.to(ratio.dtype)
    obj = torch.min(ratio * a, torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * a)
    if dual_clip is not None:
        obj = torch.where(a < 0, torch.max(obj, dual_clip * a), obj)
    return obj


def clipped_surrogate(logp_new: Tensor, logp_prox: Tensor, advantage: Tensor, weight: Tensor, *,
                      denom: Optional[Union[Tensor, float]] = None, eps: float = CLIP_EPS,
                      dual_clip: Optional[float] = DUAL_CLIP, is_weight: Optional[Tensor] = None) -> Tensor:
    """L_PPO = -sum_rows w * clipped_objective(r, A) / D with r = exp(logp_new - logp_prox) on the
    JOINT executed-action log-prob. ``logp_prox`` is treated as a constant (detached).
    ``denom``: see the module docstring (None = local weighted mean, else the caller's step-wide D).
    At r = 1 and local D this is minus the weighted mean advantage. Rows with w = 0 are sanitised
    (log r = 0, A = 0) before any arithmetic, so non-finite values there are ignored.

    The log-ratio is clamped before ``exp``: without it an fp32 ratio overflows past log r = 88 and
    the clipped branch back-propagates 0 * inf = NaN. Lower bound -LOG_RATIO_CLAMP (20): rows below
    are clipped (A < 0) or have objective r A changed by less than 2.1e-9 |A| (A > 0, gradient set
    to 0). Upper bound: +LOG_RATIO_CLAMP when the dual clip is on and c < e^20 (every row above is
    clipped: r > 1 + eps with A > 0, r > c with A < 0); otherwise log(finfo.max) - 1 of the dtype,
    so an unclipped A < 0 row keeps its exact objective r A and gradient up to that bound.

    ``is_weight`` (lag-1 overlap only, ``ppo.overlap``): the detached truncated IS weight
    rho = min(rho_bar, pi_prox / mu) per row multiplies the per-row clipped objective (dual-clip floor included)
    before the w-weighted sum; D is unchanged (importance-sampled mean, not self-normalised). None (default) runs
    exactly the statements above it and nothing else."""
    _check_same_shape(logp_new=logp_new, logp_prox=logp_prox, advantage=advantage, weight=weight)
    lp_new = _up(logp_new)
    w = _up(weight).detach().to(lp_new.dtype)
    live = _live(w)
    log_ratio = _dead_to(lp_new - _up(logp_prox).detach().to(lp_new.dtype), live)
    if dual_clip is not None and math.log(dual_clip) < LOG_RATIO_CLAMP:
        hi = LOG_RATIO_CLAMP
    else:
        hi = math.log(torch.finfo(lp_new.dtype).max) - 1.0
    ratio = torch.exp(log_ratio.clamp(-LOG_RATIO_CLAMP, hi))
    a = _dead_to(_up(advantage).detach().to(ratio.dtype), live)
    obj = clipped_objective(ratio, a, eps=eps, dual_clip=dual_clip)
    if is_weight is not None:
        _check_same_shape(is_weight=is_weight, weight=weight)
        obj = obj * _dead_to(_up(is_weight).detach().to(obj.dtype), live)
    d, check = _resolve_denom(w, denom)
    return -_normalise((w * obj).sum(), w.sum(), d.to(obj.dtype), check)


# --------------------------------------------------------------------------- KL anchor
@dataclass
class KLPenalty:
    """beta * sum_h sum_rows w KL_h / D, plus per-head pieces for logging. All values are tensors
    (no host sync inside the learner's micro-batch loop); call ``log()`` once per step to get floats."""
    loss: Tensor                                   # total penalty (with beta), differentiable
    terms: Dict[str, Tensor]                       # per head: beta_h * sum w KL_h / D (differentiable)
    own_row_mean: Dict[str, Tensor]                # per head: sum w KL_h / sum w on its own rows (detached, monitor)
    own_weight_sum: Dict[str, Tensor]              # per head: sum w on its own rows (detached, for aggregation)
    denom: Tensor                                  # the D used (detached)
    beta: Dict[str, float] = field(default_factory=dict)   # per head: the beta this penalty used (for read-back)

    def log(self, prefix: str = "kl") -> Dict[str, float]:
        """Floats for logging: ``<prefix>/<head>`` own-row mean, ``<prefix>_term/<head>`` penalty term,
        ``<prefix>_weight_sum/<head>``."""
        out: Dict[str, float] = {}
        for h in self.terms:
            out[f"{prefix}/{h}"] = float(self.own_row_mean[h])
            out[f"{prefix}_term/{h}"] = float(self.terms[h].detach())
            out[f"{prefix}_weight_sum/{h}"] = float(self.own_weight_sum[h])
        return out


def _head_sums(head_rows: Mapping[str, HeadRows], denom: Union[Tensor, float],
               coef: Union[float, Mapping[str, float]], what: str):
    """Shared by kl_penalty / entropy_bonus: per canonical head (coef_h * sum w x / D, own-row mean,
    own weight sum), with dead-row sanitising and the global-D check."""
    if not head_rows:
        raise ValueError(f"{what} needs at least one head")
    rows = _canonical_keys(head_rows, what)
    out = {}
    for h, (x, w) in rows.items():
        _check_same_shape(x=x, w=w)
        b = _coef_for(coef, h, what)
        x_u = _up(x)
        w_u = _up(w).detach().to(x_u.dtype)
        x_u = _dead_to(x_u, _live(w_u))
        d, _ = _resolve_denom(w_u, denom)
        w_sum = w_u.sum()
        term = b * _normalise((w_u * x_u).sum(), w_sum, d, True)
        out[h] = (term, _safe_div((w_u * x_u.detach()).sum(), w_sum), w_sum, d)
    return out


def kl_penalty(head_kl: Mapping[str, HeadRows], denom: Union[Tensor, float], *,
               beta: Union[float, Mapping[str, float]] = KL_BETA) -> KLPenalty:
    """KL anchor to the reference policy normalised by the surrogate denominator.

    ``head_kl[h] = (kl_rows, w_rows)``: the per-row reverse KL of head
    h on its own rows (e.g. slot on play rows with >= 2 legal slots, cells on the executed play's
    card, family on every non-forced row, folded) and the unified weight of those rows. Each head
    is summed over its own rows and divided by the SAME ``denom`` D as the surrogate (pass the
    surrogate's D; for one step that is ``surrogate_denominator(weight)`` over all rows). A head's
    rows are a subset of the step's rows, so D < sum(w_h) is an error and gives NaN.
    Head names may be canonical (``HEADS``) or ``ppo.logprob`` / run-configuration aliases; ``terms``
    is keyed by the canonical name. ``beta`` is one float for every head or a per-head mapping
    (aliases allowed; a missing head raises KeyError)."""
    parts = _head_sums(head_kl, denom, beta, "kl_penalty")
    terms = {h: p[0] for h, p in parts.items()}
    total: Optional[Tensor] = None
    for t in terms.values():
        total = t if total is None else total + t.to(total.dtype)
    d = next(iter(parts.values()))[3]
    used = {h: _coef_for(beta, h, "kl_penalty") for h in parts}
    return KLPenalty(loss=total, terms=terms, own_row_mean={h: p[1] for h, p in parts.items()},
                     own_weight_sum={h: p[2] for h, p in parts.items()}, denom=d.detach(), beta=used)


def entropy_bonus(head_ent: Mapping[str, HeadRows], denom: Union[Tensor, float], *,
                  coef: Union[float, Mapping[str, float]] = ENTROPY_COEF) -> Tensor:
    """Entropy-bonus LOSS term (add it to the loss): -sum_h coef_h * sum_rows w H_h / D, with the
    surrogate's D like the KL anchor. ``head_ent[h] = (ent_rows, w_rows)`` on the head's own
    rows. Default: 0 on every head. A bonus meant for some rows only (for example the cell head on
    cards whose cell entropy collapsed) is applied by passing only those rows (or zeroing the
    others' weights). All-zero ``coef`` returns an exact zero with no graph."""
    if not isinstance(coef, Mapping) and float(coef) == 0.0:
        if not head_ent:
            raise ValueError("entropy_bonus needs at least one head")
        x0 = _up(next(iter(head_ent.values()))[0])
        return torch.zeros((), dtype=x0.dtype, device=x0.device)
    parts = _head_sums(head_ent, denom, coef, "entropy_bonus")
    total: Optional[Tensor] = None
    for term, *_ in parts.values():
        total = term if total is None else total + term.to(total.dtype)
    return -total


def grad_norm(loss: Tensor, params: Sequence[Tensor]) -> float:
    """L2 norm of d loss / d params (unused params count as zero); keeps the graph."""
    ps = [p for p in params if p.requires_grad]
    if not ps or not loss.requires_grad:
        return 0.0
    gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    sq = sum(float(g.detach().double().pow(2).sum()) for g in gs if g is not None)
    return math.sqrt(sq)


def kl_grad_ratio(kl_terms: Mapping[str, Tensor], surrogate: Tensor,
                  params: Sequence[Tensor]) -> Dict[str, float]:
    """Per head: ||grad(beta * KL_h term)|| / ||grad(surrogate)|| over ``params``. ``kl_terms`` is
    ``KLPenalty.terms``. Ratio is ``inf`` when the surrogate gradient is zero and the KL one is not,
    ``0`` when both are zero.
    Costs one extra backward per head: call it on one minibatch per update, not every step."""
    g_s = grad_norm(surrogate, params)
    out: Dict[str, float] = {}
    for h, t in kl_terms.items():
        g_h = grad_norm(t, params)
        out[h] = g_h / g_s if g_s > 0 else (math.inf if g_h > 0 else 0.0)
    return out


# --------------------------------------------------------------------------- value
def value_loss(value: Tensor, target: Tensor, weight: Tensor, *,
               denom: Optional[Union[Tensor, float]] = None) -> Tensor:
    """sum w (v - G)^2 / D: plain MSE between the critic's tanh output ``value`` and the lambda-return
    target ``target`` (computed with the pre-update critic; detached here). No value clipping, no
    Huber, no PopArt (arXiv 2006.05990). ``weight`` is the value weight (the caller
    decides; forced rows may train the critic). ``denom`` follows the module-docstring convention
    over the value weights (None = local weighted mean; a global D below the call's value-weight
    sum gives NaN). Rows with value weight 0 are sanitised (value and target ignored, even NaN)."""
    _check_same_shape(value=value, target=target, weight=weight)
    v = _up(value)
    w = _up(weight).detach().to(v.dtype)
    live = _live(w)
    v = _dead_to(v, live)
    g = _dead_to(_up(target).detach().to(v.dtype), live)
    err = (v - g).pow(2)
    d, check = _resolve_denom(w, denom)
    return _normalise((w * err).sum(), w.sum(), d.to(v.dtype), check)


# --------------------------------------------------------------------------- BC (optional)
def bc_loss(logp_human: Tensor, weight: Tensor, denom: Union[Tensor, float], *,
            coef: float = BC_COEF) -> Tensor:
    """c_bc * (-sum w log pi_theta(a_human) / D) on human-game rows. Default ``coef`` 0 returns an
    exact zero with no graph. ``denom`` is the PPO surrogate's D of the same step (BC normalised
    by the PPO denominator, not by its own row count), so c_bc is in the units of the surrogate.
    ``weight`` is the BC rows' own weight (e.g. their imitation-learning row weight), so D is
    NOT checked against it (the BC rows are not the PPO rows); D = 0 returns 0. Rows with weight 0
    are sanitised (their log-prob is ignored, even -inf / NaN)."""
    _check_same_shape(logp_human=logp_human, weight=weight)
    lp = _up(logp_human)
    if coef == 0.0:
        return torch.zeros((), dtype=lp.dtype, device=lp.device)
    w = _up(weight).detach().to(lp.dtype)
    lp = _dead_to(lp, _live(w))
    d, _ = _resolve_denom(w, denom)
    return coef * _safe_div(-(w * lp).sum(), d.to(lp.dtype))


# --------------------------------------------------------------------------- metrics
def k3(log_ratio: Tensor) -> Tensor:
    """Per-row k3 KL estimator (r - 1) - log r = expm1(x) - x, x = log r (joschu.net/blog/kl-approx).
    Non-negative and exactly 0 at r = 1; ``clamp_min(0)`` removes rounding below 0. log r = +-inf
    gives +inf (expm1(inf) - inf would be NaN); NaN stays NaN."""
    x = _up(log_ratio).detach()
    val = (torch.expm1(x) - x).clamp_min(0.0)
    return torch.where(torch.isinf(x), torch.full_like(val, float("inf")), val)


def _q(x: Tensor, q: float) -> float:
    return float(torch.quantile(x.double(), q)) if x.numel() else float("nan")


def update_metrics(logp_new: Tensor, logp_prox: Tensor, weight: Tensor, play: Tensor, *,
                   advantage: Optional[Tensor] = None, gate_kl: Optional[Tensor] = None,
                   entropy: Optional[Mapping[str, HeadRows]] = None, eps: float = CLIP_EPS,
                   dual_clip: Optional[float] = DUAL_CLIP) -> Dict[str, float]:
    """Monitoring numbers for one minibatch / epoch, all under the unified weight:

    * ``joint_k3`` / ``play_k3``: weighted mean k3 of the joint ratio on all rows / on play rows;
    * ``clip_frac`` / ``play_clip_frac``: weighted share of rows with |r - 1| > eps;
    * ``dual_clip_frac`` (needs ``advantage``): weighted share of rows where the dual clip binds
      (A < 0 and r > c);
    * ``ratio_mean`` (weighted), ``ratio_min`` / ``ratio_max`` / ``ratio_p01`` / ``ratio_p99``
      (unweighted, over rows with w > 0);
    * ``gate_kl``: weighted mean of the per-row (folded) family KL(theta || prox) over non-forced
      rows (w > 0) -- the caller passes the per-row KL, e.g. from ``ppo.logprob``;
    * ``entropy/<head>``: weighted mean entropy of each head on its own rows (``entropy[h] =
      (ent_rows, w_rows)``, head names canonicalised), and ``entropy_weight_sum/<head>``. A head
      with no live rows reports NaN (not 0.0, which an entropy alarm would read as a collapsed
      head); aggregate over minibatches with ``entropy_weight_sum`` > 0.

    Rows with w = 0 are excluded from every number, even when their log-probs are NaN / inf /
    ``LOG_ZERO`` (their log r is set to 0 before ``exp``). ``play_k3`` / ``play_clip_frac`` read 0
    when the minibatch has no play rows (``play_weight_sum`` = 0), so they never trip early stop.
    Weighted means combine across minibatches through ``weight_sum`` / ``play_weight_sum``.

    ``play`` is a bool mask of play rows (executed card plays). Values are Python floats."""
    _check_same_shape(logp_new=logp_new, logp_prox=logp_prox, weight=weight, play=play)
    with torch.no_grad():
        lp_new = _up(logp_new).detach()
        w = _up(weight).detach().to(lp_new.dtype)
        live = _live(w)
        lr = _dead_to(lp_new - _up(logp_prox).detach().to(lp_new.dtype), live)
        r = torch.exp(lr)
        pm = play.bool()
        wp = torch.where(pm, w, torch.zeros_like(w))
        kk = k3(lr)
        clipped = ((r - 1.0).abs() > eps).to(r.dtype)
        rl = r[w > 0]
        out: Dict[str, float] = {
            "joint_k3": float(weighted_mean(kk, w)),
            "play_k3": float(weighted_mean(kk, wp)),
            "clip_frac": float(weighted_mean(clipped, w)),
            "play_clip_frac": float(weighted_mean(clipped, wp)),
            "ratio_mean": float(weighted_mean(r, w)),
            "ratio_min": float(rl.min()) if rl.numel() else float("nan"),
            "ratio_max": float(rl.max()) if rl.numel() else float("nan"),
            "ratio_p01": _q(rl, 0.01),
            "ratio_p99": _q(rl, 0.99),
            "weight_sum": float(w.sum()),
            "play_weight_sum": float(wp.sum()),
        }
        if advantage is not None:
            _check_same_shape(advantage=advantage, weight=weight)
            if dual_clip is None:
                out["dual_clip_frac"] = 0.0
            else:
                a = _dead_to(_up(advantage).detach().to(r.dtype), live)
                hit = ((a < 0) & (r > dual_clip)).to(r.dtype)
                out["dual_clip_frac"] = float(weighted_mean(hit, w))
        if gate_kl is not None:
            _check_same_shape(gate_kl=gate_kl, weight=weight)
            out["gate_kl"] = float(weighted_mean(_up(gate_kl).detach().to(r.dtype), w))
        for h, (ent, we) in _canonical_keys(entropy or {}, "update_metrics entropy").items():
            _check_same_shape(ent=ent, w=we)
            we_u = _up(we).detach()
            ws = float(we_u.sum())
            out[f"entropy/{h}"] = float(weighted_mean(_up(ent).detach(), we_u)) if ws > 0 else float("nan")
            out[f"entropy_weight_sum/{h}"] = ws
    return out


def tiny_ratio_shares(logp_new: Tensor, logp_prox: Tensor, weight: Tensor, row_type: Tensor, *,
                      cut: float = TINY_LOG_RATIO) -> Dict[str, float]:
    """Diagnostic for a few collapsed rows dominating the pooled approx-KL: per row type (``wait``,
    ``play``, ``ability`` of ``ROW_TYPE_NAMES``), the share of live rows (w > 0) whose log ratio
    log(pi_theta / pi_prox) is below ``cut`` (default ``TINY_LOG_RATIO``): ``tiny_ratio_share/<type>`` (row count share,
    NaN without live rows of that type), ``tiny_ratio_wshare/<type>`` (unified-weight share) and
    ``tiny_ratio_rows/<type>``. Diagnostic only; dead rows never count, even with a NaN log-prob."""
    _check_same_shape(logp_new=logp_new, logp_prox=logp_prox, weight=weight, row_type=row_type)
    out: Dict[str, float] = {}
    with torch.no_grad():
        lp = _up(logp_new).detach()
        w = _up(weight).detach().to(lp.dtype)
        live = _live(w)
        lr = _dead_to(lp - _up(logp_prox).detach().to(lp.dtype), live)
        tiny = live & (lr < float(cut))
        rt = torch.as_tensor(row_type, device=lp.device).long()
        for code, name in ROW_TYPE_NAMES.items():
            sel = live & (rt == code)
            n = int(sel.sum())
            k = int((tiny & sel).sum())
            ws = float(w[sel].sum())
            out[f"tiny_ratio_share/{name}"] = k / n if n else float("nan")
            out[f"tiny_ratio_wshare/{name}"] = float(w[tiny & sel].sum()) / ws if ws > 0 else float("nan")
            out[f"tiny_ratio_rows/{name}"] = float(n)
    return out


def early_stop(metrics: Mapping[str, float],
               thresholds: Optional[Mapping[str, Optional[float]]] = None, *,
               strict: bool = True) -> Tuple[str, ...]:
    """Names of the early-stop limits that tripped this epoch: joint k3, play-row k3 and gate KL
    per non-forced row, each above its ``EARLY_STOP`` limit. ``thresholds`` is MERGED into
    ``EARLY_STOP`` (calibrating one limit keeps the others); a ``None`` value removes that
    limit. A non-finite metric trips (NaN must stop the epoch, never pass). A limit whose metric is
    missing raises KeyError (``strict``, default: a learner that forgets ``gate_kl`` must not run
    without the gate limit, since the ratio clip does not bound the gate); ``strict=False`` skips it.
    Empty tuple = keep going."""
    th = {**EARLY_STOP, **(thresholds or {})}
    tripped = []
    for k, lim in th.items():
        if lim is None:
            continue
        if k not in metrics:
            if strict:
                raise KeyError(f"early_stop: metric {k!r} missing (pass it, set thresholds={{{k!r}: None}} "
                               f"to drop the limit, or strict=False)")
            continue
        v = float(metrics[k])
        if not math.isfinite(v) or v > lim:
            tripped.append(k)
    return tuple(tripped)


# --------------------------------------------------------------------------- update-0 identity
@dataclass
class IdentityReport:
    """Update-0 identity numbers. ``ok`` iff every row type's
    max |r - 1| < ``ratio_tol`` and every head's max KL(theta_0 || ref) < ``kl_tol``."""
    max_abs_ratio_dev: Dict[str, float]
    rows: Dict[str, int]
    max_ref_kl: Dict[str, float]
    ratio_tol: float = IDENTITY_RATIO_TOL
    kl_tol: float = IDENTITY_KL_TOL
    failures: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.failures


def identity_check(logp_new: Tensor, logp_prox: Tensor, row_type: Tensor,
                   ref_kl: Optional[Mapping[str, Union[Tensor, float]]] = None, *,
                   mask: Optional[Tensor] = None, ratio_tol: float = IDENTITY_RATIO_TOL,
                   kl_tol: float = IDENTITY_KL_TOL) -> IdentityReport:
    """Update-0 identity: max |r - 1| per row type (ROW_WAIT / ROW_PLAY / ROW_ABILITY, never
    pooled) at the first training-mode step, and KL_h(theta_0 || ref) per head (max over rows when
    a tensor is given). ``mask`` restricts the ratio check (default: all rows). Row types with no
    rows report 0 deviation. Non-finite values fail."""
    _check_same_shape(logp_new=logp_new, logp_prox=logp_prox, row_type=row_type)
    with torch.no_grad():
        dev = (torch.exp(_up(logp_new).detach().double() - _up(logp_prox).detach().double()) - 1.0).abs()
        keep = torch.ones_like(dev, dtype=torch.bool) if mask is None else mask.bool()
        rt = row_type.long()
        unknown = set(int(x) for x in torch.unique(rt[keep]).tolist()) - set(ROW_TYPE_NAMES)
        if unknown:
            raise ValueError(f"unknown row types {sorted(unknown)} (expected {sorted(ROW_TYPE_NAMES)})")
        mx: Dict[str, float] = {}
        cnt: Dict[str, int] = {}
        fails = []
        for code, name in ROW_TYPE_NAMES.items():
            sel = keep & (rt == code)
            d = dev[sel]
            cnt[name] = int(sel.sum())
            mx[name] = float(d.max()) if d.numel() else 0.0
            if not math.isfinite(mx[name]) or (d.numel() and not bool(torch.isfinite(d).all())) \
                    or not mx[name] < ratio_tol:
                fails.append(f"ratio/{name}")
        kls: Dict[str, float] = {}
        for h, v in (ref_kl or {}).items():
            t = torch.as_tensor(v).detach().double()
            kls[h] = float(t.max()) if t.numel() else 0.0
            if not math.isfinite(kls[h]) or (t.numel() and not bool(torch.isfinite(t).all())) \
                    or not kls[h] < kl_tol:
                fails.append(f"kl/{h}")
    return IdentityReport(max_abs_ratio_dev=mx, rows=cnt, max_ref_kl=kls, ratio_tol=ratio_tol,
                          kl_tol=kl_tol, failures=tuple(fails))


def assert_identity(*args, **kwargs) -> IdentityReport:
    """``identity_check`` that raises ``AssertionError`` naming each failure (a failed check stops the run)."""
    rep = identity_check(*args, **kwargs)
    if not rep.ok:
        raise AssertionError(
            f"PPO update-0 identity failed: {', '.join(rep.failures)}; "
            f"max|r-1| by row type {rep.max_abs_ratio_dev} (tol {rep.ratio_tol}), "
            f"max KL(theta0||ref) {rep.max_ref_kl} (tol {rep.kl_tol})")
    return rep


__all__ = [
    "CLIP_EPS", "DUAL_CLIP", "KL_BETA", "BC_COEF", "ENTROPY_COEF", "LOG_RATIO_CLAMP", "DENOM_RTOL",
    "EARLY_STOP", "IDENTITY_RATIO_TOL", "IDENTITY_KL_TOL", "ROW_WAIT", "ROW_PLAY", "ROW_ABILITY",
    "ROW_TYPE_NAMES", "HEADS", "HEAD_ALIASES", "canonical_head",
    "surrogate_denominator", "weighted_mean", "clipped_objective", "clipped_surrogate",
    "KLPenalty", "kl_penalty", "entropy_bonus", "grad_norm", "kl_grad_ratio", "value_loss", "bc_loss", "k3",
    "update_metrics", "early_stop", "IdentityReport", "identity_check", "assert_identity",
    "TINY_LOG_RATIO", "tiny_ratio_shares",
]
