"""Log-probabilities, entropies, KL and temperature for the factorised policy.

The policy acts in three levels (``aminiformer.sample_actions``): a family drawn from the
competing-hazard map of the gate logits (``hazard_probs``), then slot -> cell for a play, or the
ability pointer.  A play drawn on a row with no legal slot is executed as a wait (``play_blocked``),
so every quantity here uses the FOLDED family distribution::

    wait_exec = P_wait + P_play * [no legal slot]
    play_exec = P_play * [some legal slot]
    ability   = P_ability

and the log-probability of the EXECUTED action (one function shared by actor and learner)::

    wait          log(P_wait + P_play * [no legal slot])
    play (s, c)   log P_play + log softmax_legal(slot)[s] + log softmax(masked cells[s])[c]
    ability k     log P_ability + log softmax_ready(ability)[k]

Everything is computed in float64 and stays differentiable (the learner calls the same functions on its
forward outputs).  The hazard terms use the closed form of ``hazard_probs`` in log space:
``log P_wait = -rate`` and ``log P_act = log(-expm1(-rate))``, with the series
``log rate - rate/2 + rate^2/24`` below ``SERIES_RATE`` (values and gradients stay finite even when the
rate underflows).  An impossible event gets the finite log-probability ``LOG_ZERO``
(``exp(LOG_ZERO) == 0`` exactly).  ``executed_logp_terms`` flags actions ``out`` cannot execute
(``valid``); their ``total`` is the constant ``LOG_ZERO`` with no gradient.

Cells come in three forms: peek ``[n, 4, 576]``; learner row-aligned ``[n, 576]`` (cells of the executed
slot); or the raw ``forward(cell_rows=..., play_slot=...)`` output ``cells [n_play, 576]`` plus
``cell_play_index``, which is scattered to ``[n, 576]`` here (``scatter_cells``).

Legality is read from the logits exactly as ``sample_actions`` reads it: an entry is legal iff its logit
is ``> MASK_FILL / 2``; the family uses ``out['ability_avail']`` as ``hazard_probs`` does.  Gate logits
are taken as given: whatever ``gate_bias`` the caller's model applied stays (the PPO policy and the KL
reference use ``None``).  Nothing here normalises over rows: KL and entropy are returned per row
with their row masks, and the surrogate-denominator normalisation lives in ``losses``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Union

import torch

from amini.nn import schema_v1 as S
from amini.nn.aminiformer import MASK_FILL

LOG_ZERO = -1.0e30          # log-probability of an impossible event (exp(LOG_ZERO) == 0.0 in float64)
LEGAL_THRESHOLD = MASK_FILL / 2   # the legality test of sample_actions: logit > MASK_FILL / 2
SERIES_RATE = 1.0e-4        # below this rate log(1 - exp(-rate)) uses its series (error ~ rate^4 / 2880)
T_MIN = 1.0e-3              # smallest temperature apply_temperature accepts

WAIT, PLAY, ABILITY = S.FAMILY_WAIT, S.FAMILY_PLAY, S.FAMILY_ABILITY

Out = Mapping[str, torch.Tensor]
Action = Mapping[str, Any]


# =========================================================================== masks
def legal(logits: torch.Tensor) -> torch.Tensor:
    """Legal entries of masked logits, exactly as ``sample_actions`` decides (``> MASK_FILL / 2``)."""

    return logits.detach() > LEGAL_THRESHOLD


def legal_masks(out: Out) -> Dict[str, torch.Tensor]:
    """Row legality of one ``peek`` / forward output.

    ``slot_legal [n, 4]``, ``legal_any [n]`` (some legal slot), ``n_slots [n]``,
    ``ability_ready [n, 64]``, ``ability_avail [n]`` (as the family sees it), ``n_ready [n]``,
    ``forced [n]`` (see ``forced_mask``)."""

    slot_legal = legal(out["slot"])
    ready = legal(out["ability"])
    avail = torch.as_tensor(out["ability_avail"], device=slot_legal.device).bool().reshape(-1)
    legal_any = slot_legal.any(-1)
    return {"slot_legal": slot_legal, "legal_any": legal_any, "n_slots": slot_legal.sum(-1),
            "ability_ready": ready, "ability_avail": avail, "n_ready": ready.sum(-1),
            "forced": ~legal_any & ~avail}


def forced_mask(out: Out) -> torch.Tensor:
    """Rows where wait is the only executable action (no legal slot and no ready ability).

    Such rows contribute nothing to the policy loss, KL or entropy; their weight in every policy
    statistic is 0 (they still train the value head)."""

    return legal_masks(out)["forced"]


# =========================================================================== family (gate)
def _dt64(dt: Union[torch.Tensor, float], like: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(dt, dtype=torch.float64, device=like.device)
    return t.expand(like.shape[0]) if t.dim() == 0 else t.reshape(-1)


def family_log_probs(family_logits: torch.Tensor, ability_avail: torch.Tensor, legal_any: torch.Tensor,
                     dt: Union[torch.Tensor, float], u_ticks: float = S.U_TICKS) -> torch.Tensor:
    """Folded family log-probabilities ``[n, 3]`` (wait_exec, play_exec, ability), float64.

    Same hazard as ``hazard_probs``: ``odds_f = exp(l_f - l_wait)`` (ability odds 0 when not available),
    ``rate = (odds_play + odds_ability) * max(dt, 0) / U``, ``P_wait = exp(-rate)``,
    ``P_act = 1 - exp(-rate)`` split by the odds; then the blocked play is folded into wait.
    Impossible entries are ``LOG_ZERO``.  The gate is never temperature-scaled."""

    lf = family_logits.double()
    avail = ability_avail.to(lf.device).bool().reshape(-1)
    la_ok = legal_any.to(lf.device).bool().reshape(-1)
    d_p = lf[:, PLAY] - lf[:, WAIT]
    d_a = torch.where(avail, lf[:, ABILITY] - lf[:, WAIT], torch.full_like(d_p, LOG_ZERO))
    log_tot = torch.logsumexp(torch.stack([d_p, d_a], -1), -1)          # log(odds_p + odds_a)
    dt64 = _dt64(dt, lf)
    act_ok = dt64 > 0                                                   # play / ability possible at all
    rate = torch.exp(log_tot) * dt64.clamp(min=0) / float(u_ticks)       # exactly hazard_probs' rate
    log_none = -rate
    # log P_act = log(1 - exp(-rate)); the series branch keeps value and gradient finite when the rate is
    # tiny or underflows (the backward of log(x) is 1/x).  Unused branches get safe inputs (no inf * 0).
    small = rate < SERIES_RATE
    one, zero = torch.ones_like(rate), torch.zeros_like(rate)
    log_act_big = torch.log(-torch.expm1(-torch.where(small, one, rate)))
    r_s = torch.where(small, rate, zero)
    log_rate = log_tot + torch.log(torch.where(act_ok, dt64, one) / float(u_ticks))
    log_act_small = log_rate - r_s / 2 + r_s * r_s / 24
    log_act = torch.where(act_ok, torch.where(small, log_act_small, log_act_big), torch.full_like(rate, LOG_ZERO))
    log_play = log_act + d_p - log_tot
    log_ab = torch.where(avail & act_ok, log_act + d_a - log_tot, torch.full_like(rate, LOG_ZERO))
    log_wait_exec = torch.where(la_ok, log_none, torch.logaddexp(log_none, log_play))
    log_play_exec = torch.where(la_ok & act_ok, log_play, torch.full_like(rate, LOG_ZERO))
    return torch.stack([log_wait_exec, log_play_exec, log_ab], -1)


def family_possible(ability_avail: torch.Tensor, legal_any: torch.Tensor,
                    dt: Union[torch.Tensor, float], family_logits: torch.Tensor) -> torch.Tensor:
    """``[n, 3]`` bool: which folded families can be executed (wait always; play / ability need dt > 0)."""

    pos = _dt64(dt, family_logits) > 0
    avail = ability_avail.to(pos.device).bool().reshape(-1)
    return torch.stack([torch.ones_like(pos), legal_any.to(pos.device).bool().reshape(-1) & pos,
                        avail & pos], -1)


# =========================================================================== cells forms
def scatter_cells(cells: Optional[torch.Tensor], cell_play_index: torch.Tensor, n: int, *,
                  n_cells: int = S.N_CELLS, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Row-aligned ``[n, C]`` learner cells from ``forward(play_slot=...)``'s ``cells [n_play, C]`` and
    ``cell_play_index [n_play]`` (row of each play); rows without a play are all ``MASK_FILL``.
    Differentiable in ``cells``."""

    idx = torch.as_tensor(cell_play_index).long().reshape(-1)
    if cells is None:
        if idx.numel():
            raise ValueError("cell_play_index names play rows but out['cells'] is missing")
        return torch.full((n, n_cells), MASK_FILL, dtype=dtype or torch.float32, device=idx.device)
    if cells.dim() != 2 or cells.shape[0] != idx.numel():
        raise ValueError(f"cells {tuple(cells.shape)} do not match cell_play_index ({idx.numel()} rows)")
    full = torch.full((n, cells.shape[-1]), MASK_FILL, dtype=cells.dtype, device=cells.device)
    return full.index_put((idx.to(cells.device),), cells)


def row_cells(out: Out) -> Optional[torch.Tensor]:
    """Row-aligned cells of ``out``: peek ``[n, 4, C]`` or learner ``[n, C]`` (raw forward output with
    ``cell_play_index`` is scattered); ``None`` when ``out`` carries no cells."""

    n = out["family"].shape[0]
    cells = out.get("cells")
    idx = out.get("cell_play_index")
    n_idx = torch.as_tensor(idx).numel() if idx is not None else -1
    if idx is not None and (cells is None or (cells.dim() == 2 and cells.shape[0] == n_idx)):
        dtype = cells.dtype if cells is not None else out["slot"].dtype
        return scatter_cells(cells, idx, n, dtype=dtype)     # all-play batches: idx = arange(n), identity
    if cells is None:
        return None
    if cells.dim() not in (2, 3) or cells.shape[0] != n:
        raise ValueError(f"cells {tuple(cells.shape)} are not row-aligned with {n} rows "
                         "(raw forward cells need out['cell_play_index'])")
    return cells


# =========================================================================== per-head distributions
def _log_softmax64(logits: torch.Tensor) -> torch.Tensor:
    """log softmax over the last axis in float64: masked entries (MASK_FILL) get ~-1e9 (p = 0 exactly);
    a fully masked row is uniform, exactly the distribution ``sample_actions`` draws from."""

    return torch.log_softmax(logits.double(), -1)


def head_log_probs(out: Out, dt: Union[torch.Tensor, float], *, u_ticks: float = S.U_TICKS
                   ) -> Dict[str, torch.Tensor]:
    """All per-head log-probabilities of one output dict, float64.

    ``family [n, 3]`` (folded), ``slot [n, 4]``, ``cells [n, 4, 576]`` (peek) or ``[n, 576]`` (cells of
    the executed slot, learner form; see ``row_cells``), ``ability [n, 64]``, plus the masks of
    ``legal_masks`` and ``family_possible [n, 3]``."""

    m = legal_masks(out)
    fam = out["family"]
    lp = {"family": family_log_probs(fam, m["ability_avail"], m["legal_any"], dt, u_ticks),
          "family_possible": family_possible(m["ability_avail"], m["legal_any"], dt, fam),
          "slot": _log_softmax64(out["slot"]), "ability": _log_softmax64(out["ability"])}
    cells = row_cells(out)
    if cells is not None:
        lp["cells"] = _log_softmax64(cells)
        lp["cell_legal"] = legal(cells)
    lp.update(m)
    return lp


# =========================================================================== executed action
def _idx(action: Action, key: str, n: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(action[key], device=device).long().reshape(n)


def executed_family(out: Out, action: Action) -> torch.Tensor:
    """Executed family ``[n]``: a drawn play with no legal slot (``play_blocked``) is a wait."""

    legal_any = legal(out["slot"]).any(-1)
    n = legal_any.shape[0]
    fam = _idx(action, "family", n, legal_any.device)
    blocked = (fam == PLAY) & ~legal_any
    if "play_blocked" in action and action["play_blocked"] is not None:
        blocked = blocked | torch.as_tensor(action["play_blocked"], device=fam.device).bool().reshape(n)
    return torch.where(blocked, torch.full_like(fam, WAIT), fam)


def _executed_cells(cells: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """``[n, C]`` cells of the executed slot from peek ``[n, 4, C]`` or learner ``[n, C]`` cells."""

    if cells.dim() == 3:
        return cells[torch.arange(cells.shape[0], device=cells.device), slot]
    return cells


def executed_logp_terms(out: Out, dt: Union[torch.Tensor, float], action: Action, *,
                        u_ticks: float = S.U_TICKS, lp: Optional[Dict[str, torch.Tensor]] = None
                        ) -> Dict[str, torch.Tensor]:
    """Per-term log pi of the executed action, float64 ``[n]`` each.

    ``action`` is the dict ``sample_actions`` returns (``family`` already folded, ``slot``, ``cell``,
    ``ability``, ``play_blocked``) or its stored form (-1 on unused heads); only the terms of the executed
    branch enter (the slot / cell drawn on a wait row are discarded).  Keys: ``family``, ``slot``,
    ``cell``, ``ability``, ``total``, ``executed_family``, ``valid``.

    ``valid [n]`` is False when ``out`` cannot execute the action (family impossible, slot / cell
    illegal, pointer not ready, index out of range); ``sample_actions`` never produces such rows, so it
    only flags writer or legality bugs.  On invalid rows ``total`` is the constant ``LOG_ZERO`` (no
    gradient, ratio exactly 1): ``losses`` must assert ``valid`` or zero-weight and count those rows.
    Without ``lp`` only the executed slot's cells are log-softmaxed (actor cost)."""

    fam = executed_family(out, action)
    if lp is None:
        m = legal_masks(out)
        f_lp = family_log_probs(out["family"], m["ability_avail"], m["legal_any"], dt, u_ticks)
        poss = family_possible(m["ability_avail"], m["legal_any"], dt, out["family"])
        s_lp, a_lp = _log_softmax64(out["slot"]), _log_softmax64(out["ability"])
        cells = row_cells(out)
    else:
        m, f_lp, poss, s_lp, a_lp = lp, lp["family"], lp["family_possible"], lp["slot"], lp["ability"]
        cells = None
    n = fam.shape[0]
    dev = f_lp.device
    fam = fam.to(dev)
    slot_raw, ab_raw = _idx(action, "slot", n, dev), _idx(action, "ability", n, dev)
    n_slot, n_ab = s_lp.shape[-1], a_lp.shape[-1]
    slot, ab = slot_raw.clamp(0, n_slot - 1), ab_raw.clamp(0, n_ab - 1)
    is_play, is_ab = fam == PLAY, fam == ABILITY
    zero = torch.zeros(n, dtype=torch.float64, device=dev)
    fam_ok = (fam >= 0) & (fam <= 2)
    t_fam = f_lp.gather(-1, fam.clamp(0, 2).unsqueeze(-1)).squeeze(-1)
    t_slot = torch.where(is_play, s_lp.gather(-1, slot.unsqueeze(-1)).squeeze(-1), zero)
    t_ab = torch.where(is_ab, a_lp.gather(-1, ab.unsqueeze(-1)).squeeze(-1), zero)
    has_cells = cells is not None if lp is None else "cells" in lp
    if bool(is_play.any()) and not has_cells:
        raise KeyError("executed play rows need out['cells']")
    play_ok = (slot_raw >= 0) & (slot_raw < n_slot) & m["slot_legal"].gather(-1, slot.unsqueeze(-1)).squeeze(-1)
    if has_cells:
        if lp is None:                       # gather the executed slot first, then one log-softmax
            raw = _executed_cells(cells, slot)
            cl, c_ok = _log_softmax64(raw), legal(raw)
        else:
            cl, c_ok = _executed_cells(lp["cells"], slot), _executed_cells(lp["cell_legal"], slot)
        cell_raw = _idx(action, "cell", n, dev)
        cell = cell_raw.clamp(0, cl.shape[-1] - 1)
        t_cell = torch.where(is_play, cl.gather(-1, cell.unsqueeze(-1)).squeeze(-1), zero)
        play_ok = (play_ok & (cell_raw >= 0) & (cell_raw < cl.shape[-1])
                   & c_ok.gather(-1, cell.unsqueeze(-1)).squeeze(-1))
    else:
        t_cell = zero
    ab_ok = (ab_raw >= 0) & (ab_raw < n_ab) & m["ability_ready"].gather(-1, ab.unsqueeze(-1)).squeeze(-1)
    valid = (fam_ok & poss.gather(-1, fam.clamp(0, 2).unsqueeze(-1)).squeeze(-1)
             & (~is_play | play_ok) & (~is_ab | ab_ok))
    total = torch.where(valid, t_fam + t_slot + t_cell + t_ab, torch.full_like(zero, LOG_ZERO))
    return {"family": t_fam, "slot": t_slot, "cell": t_cell, "ability": t_ab,
            "total": total, "executed_family": fam, "valid": valid}


def executed_logp(out: Out, dt: Union[torch.Tensor, float], action: Action, *,
                  u_ticks: float = S.U_TICKS) -> torch.Tensor:
    """log pi of the EXECUTED action per row, float64 ``[n]`` (shared by actor and learner).

    Invalid actions give ``LOG_ZERO`` without gradient; use ``executed_logp_terms(...)['valid']``."""

    return executed_logp_terms(out, dt, action, u_ticks=u_ticks)["total"]


# =========================================================================== entropy / KL
def _cat_entropy(logp: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    p = torch.exp(logp)
    return -torch.where(ok, p * logp, torch.zeros_like(logp)).sum(-1)


def _cat_kl(logp: torch.Tensor, logq: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """Exact categorical KL(p || q) over the entries ``ok`` (legal under p)."""

    p = torch.exp(logp)
    return torch.where(ok, p * (logp - logq), torch.zeros_like(logp)).sum(-1)


def _row_masks(lp: Dict[str, torch.Tensor], fam: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per-head row masks (sampled-branch rows of the executed action)."""

    is_play, is_ab = fam == PLAY, fam == ABILITY
    return {"mask_family": ~lp["forced"],
            "mask_slot": is_play & (lp["n_slots"] >= 2),
            "mask_cell": is_play,
            "mask_ability": is_ab & (lp["n_ready"] >= 2)}


def head_entropy(out: Out, dt: Union[torch.Tensor, float], action: Action, *,
                 u_ticks: float = S.U_TICKS) -> Dict[str, torch.Tensor]:
    """Per-head entropies over LEGAL choices, float64 ``[n]``, with the row masks.

    ``ent_family`` (folded 3-way, mask: non-forced rows), ``ent_slot`` (mask: executed play with >= 2
    legal slots), ``ent_cell`` (cells of the executed slot, mask: executed play), ``ent_ability``
    (mask: executed ability with >= 2 ready).  Values are computed on every row; only masked rows are
    meaningful.  ``ent_joint`` (chain rule ``H_fam + P_play (H_slot + sum_s pi_s H_cell(s)) + P_ab H_ab``)
    is added when the cells of all four slots are present."""

    lp = head_log_probs(out, dt, u_ticks=u_ticks)
    fam = executed_family(out, action).to(lp["family"].device)
    n = fam.shape[0]
    slot = _idx(action, "slot", n, fam.device).clamp(0, lp["slot"].shape[-1] - 1)
    res = {"ent_family": _cat_entropy(lp["family"], lp["family_possible"]),
           "ent_slot": _cat_entropy(lp["slot"], lp["slot_legal"]),
           "ent_ability": _cat_entropy(lp["ability"], lp["ability_ready"])}
    if "cells" in lp:
        res["ent_cell"] = _cat_entropy(_executed_cells(lp["cells"], slot),
                                       _executed_cells(lp["cell_legal"], slot))
        if lp["cells"].dim() == 3:
            pf = torch.exp(lp["family"])
            h_cells = _cat_entropy(lp["cells"], lp["cell_legal"])                     # [n, 4]
            ps = torch.where(lp["slot_legal"], torch.exp(lp["slot"]), torch.zeros_like(lp["slot"]))
            res["ent_joint"] = (res["ent_family"]
                                + pf[:, PLAY] * (res["ent_slot"] + (ps * h_cells).sum(-1))
                                + pf[:, ABILITY] * res["ent_ability"])
    res.update(_row_masks(lp, fam))
    return res


def head_kl(out: Out, ref_out: Out, dt: Union[torch.Tensor, float], action: Action, *,
            u_ticks: float = S.U_TICKS) -> Dict[str, torch.Tensor]:
    """Exact per-head categorical KL(pi_theta || pi_ref) per row, float64 ``[n]``, with row masks.

    ``kl_family``: folded 3-way gate distribution (mask ``mask_family``: non-forced rows);
    ``kl_slot``: legal slots (mask: executed play with >= 2 legal slots);
    ``kl_cell``: cells of the EXECUTED slot (mask: executed play);
    ``kl_ability``: ready abilities (mask: executed ability with >= 2 ready).
    The masked per-head sums divided by the surrogate denominator (``losses``) estimate the joint
    per-row KL (sampled-branch estimator).  ``kl_joint`` is the exact joint KL per row
    (``KL_fam + P_play (KL_slot + sum_s pi_s KL_cell(s)) + P_ab KL_ab``), added when both outputs carry
    the cells of all four slots.  Legality is theta's (both models see the same row masks).
    ``ref_out`` comes from the frozen reference (``gate_bias`` None) on the same rows and dt."""

    lp = head_log_probs(out, dt, u_ticks=u_ticks)
    lq = head_log_probs(ref_out, dt, u_ticks=u_ticks)
    fam = executed_family(out, action).to(lp["family"].device)
    n = fam.shape[0]
    slot = _idx(action, "slot", n, fam.device).clamp(0, lp["slot"].shape[-1] - 1)
    res = {"kl_family": _cat_kl(lp["family"], lq["family"], lp["family_possible"]),
           "kl_slot": _cat_kl(lp["slot"], lq["slot"], lp["slot_legal"]),
           "kl_ability": _cat_kl(lp["ability"], lq["ability"], lp["ability_ready"])}
    if "cells" in lp and "cells" in lq:
        res["kl_cell"] = _cat_kl(_executed_cells(lp["cells"], slot), _executed_cells(lq["cells"], slot),
                                 _executed_cells(lp["cell_legal"], slot))
        if lp["cells"].dim() == 3 and lq["cells"].dim() == 3:
            pf = torch.exp(lp["family"])
            kl_cells = _cat_kl(lp["cells"], lq["cells"], lp["cell_legal"])          # [n, 4]
            ps = torch.where(lp["slot_legal"], torch.exp(lp["slot"]), torch.zeros_like(lp["slot"]))
            res["kl_joint"] = (res["kl_family"]
                               + pf[:, PLAY] * (res["kl_slot"] + (ps * kl_cells).sum(-1))
                               + pf[:, ABILITY] * res["kl_ability"])
    res.update(_row_masks(lp, fam))
    return res


# =========================================================================== temperature
def _check_t(name: str, t: float) -> float:
    t = float(t)
    if not (math.isfinite(t) and t >= T_MIN):
        raise ValueError(f"{name} must be a finite temperature >= {T_MIN}, got {t}")
    return t


def apply_temperature(out: Out, t_slot: float = 1.0, t_cell: float = 1.0, t_ability: float = 1.0
                      ) -> Dict[str, torch.Tensor]:
    """Tempered copy of a ``peek`` / forward output: LEGAL slot, cell and ability logits / T.

    Masked entries keep their exact value (MASK_FILL).  T must be finite and ``>= T_MIN`` (greedy acting
    stays ``sample_actions(greedy=True)``); a tempered legal logit that would fall to ``<= MASK_FILL / 2``
    or overflow raises ``ValueError``, so the legality test of ``sample_actions`` (``> MASK_FILL / 2``)
    never changes.  The family / gate logits are never touched.  T == 1 returns the input tensors
    themselves (bitwise identical).  Differentiable; the learner must evaluate log pi at the same T the
    actor sampled at."""

    res = dict(out)
    for key, name, t in (("slot", "t_slot", t_slot), ("cells", "t_cell", t_cell),
                         ("ability", "t_ability", t_ability)):
        t = _check_t(name, t)
        if t == 1.0 or key not in out:
            continue
        x = out[key]
        ok = legal(x)
        y = torch.where(ok, x / t, x)
        yd = y.detach()
        if not (torch.equal(yd > LEGAL_THRESHOLD, ok) and bool(torch.isfinite(yd[ok]).all())):
            raise ValueError(f"{name}={t} pushes legal {key} logits out of range (legality would change)")
        res[key] = y
    return res


__all__ = ["LOG_ZERO", "LEGAL_THRESHOLD", "SERIES_RATE", "T_MIN", "scatter_cells", "row_cells", "legal",
           "legal_masks", "forced_mask", "family_log_probs", "family_possible", "head_log_probs",
           "executed_family", "executed_logp_terms", "executed_logp", "head_entropy", "head_kl",
           "apply_temperature"]
