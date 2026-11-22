"""Tests for amini.ppo.logprob.

Real outputs come from ``tiny_model`` + ``example`` through ``peek`` (cell mask on); synthetic dicts
cover the edge cases (blocked play with and without a ready ability, forced row, single legal slot,
few legal cells).  The sampler under test is the unmodified ``aminiformer.sample_actions``.
Tests on real outputs are skipped without the private dataset (``common.example``).
"""

from __future__ import annotations

import math
import sys
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tests/v7 (common)
from common import S, example, tiny_model, to_batch  # noqa: E402

from amini.nn.aminiformer import MASK_FILL, hazard_probs, sample_actions  # noqa: E402
from amini.ppo import logprob as LP  # noqa: E402

torch.set_num_threads(4)

TEMPS = (1.0, 0.5, 0.25, 2.0, 5.0)
N_DRAWS = 100_000
NB = 4                                  # probability-mass bins of the cell distribution per slot
K = 1 + S.HAND_SLOTS * NB + S.ABILITY_POINTER
HEADS = ("family", "slot", "ability", "ability_avail", "cells")


# =========================================================================== fixtures
@lru_cache(maxsize=4)
def peek_outputs(seed: int = 0, ex: int = 2):
    """``peek`` over every row of example ``ex`` (commit all), concatenated to ``n`` rows."""

    model = tiny_model(seed, cell_mask=True)
    e, _ = example(ex)
    b = to_batch([e])
    r = b["row"]
    n = int(r["row_valid"][0].sum())
    state = model.initial_state(1, dtype=torch.float64)

    def sl(v, t):
        if isinstance(v, dict):
            return {k: sl(x, t) for k, x in v.items()}
        return v[0:1, t]

    outs = []
    for t in range(n):
        tok = {"groups": sl(b["groups"], t),
               "row": {k: r[k][0:1, t] for k in ("slot_legal", "ability_ready", "cell_mask_id")}}
        outs.append(model.peek(tok, r["pos"][0:1, t], state))
        model.commit(state, torch.tensor([True]))
    return {k: torch.cat([o[k] for o in outs]) for k in HEADS}


def synth_out():
    """Edge-case rows (float64, model conventions: masked ability family logit when not available).

    0 blocked play, 2 ready abilities   1 forced (no slot, no ability)   2 single legal slot, 6 legal cells
    3 three legal slots, 1 ready        4 four legal slots, 3 ready, half the cells masked"""

    g = torch.Generator().manual_seed(11)
    n = 5
    fam = torch.tensor([[0.0, 1.0, 0.5], [0.0, 2.0, MASK_FILL], [0.0, 0.3, MASK_FILL],
                        [0.0, 0.2, -0.4], [0.0, -0.5, 0.7]], dtype=torch.float64)
    avail = torch.tensor([True, False, False, True, True])
    slot_legal = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1]]).bool()
    ready = torch.zeros(n, S.ABILITY_POINTER, dtype=torch.bool)
    ready[0, [3, 10]] = True
    ready[3, 0] = True
    ready[4, [1, 5, 60]] = True
    slot = (1.5 * torch.randn(n, 4, generator=g, dtype=torch.float64)).masked_fill(~slot_legal, MASK_FILL)
    ab = torch.randn(n, S.ABILITY_POINTER, generator=g, dtype=torch.float64).masked_fill(~ready, MASK_FILL)
    cells = 2.0 * torch.randn(n, 4, S.N_CELLS, generator=g, dtype=torch.float64)
    cmask = torch.ones(n, 4, S.N_CELLS, dtype=torch.bool)
    cmask[2, 2] = False
    cmask[2, 2, [0, 17, 100, 300, 301, 575]] = True
    cmask[4] = torch.rand(4, S.N_CELLS, generator=g) < 0.5
    cells = cells.masked_fill(~cmask, MASK_FILL)
    return {"family": fam, "slot": slot, "ability": ab, "ability_avail": avail, "cells": cells}


def cat_rows(a, b):
    return {k: torch.cat([a[k], b[k]]) for k in HEADS}


def pick_rows(out):
    """Real rows covering the cases: >=2 slots & >=2 ready, blocked + ability, single slot, forced."""

    m = LP.legal_masks(out)
    ns, nr = m["n_slots"], m["n_ready"]
    conds = [(ns >= 2) & (nr >= 2), (ns == 0) & (nr >= 1), (ns == 1) & (nr == 0), (ns == 0) & (nr == 0),
             (ns >= 3) & (nr == 0)]
    rows = []
    for c in conds:
        idx = c.nonzero(as_tuple=True)[0]
        assert idx.numel(), "fixture lacks a row case"
        rows.append(int(idx[0]))
    return rows


def sub(out, idx):
    idx = torch.as_tensor(idx, dtype=torch.long)
    return {k: out[k][idx] for k in HEADS}


def unit_rate_dt(out):
    """dt per row with hazard rate 1 (P_wait = e^-1): every branch is frequent in the empirical test."""

    lf = out["family"].double()
    odds = torch.exp(lf[:, 1] - lf[:, 0]) + torch.where(out["ability_avail"].bool(),
                                                        torch.exp(lf[:, 2] - lf[:, 0]),
                                                        torch.zeros(lf.shape[0], dtype=torch.float64))
    return S.U_TICKS / odds


def enumerate_actions(out, i):
    """All executable actions of row ``i`` (wait; plays on legal slots x every cell; ready abilities)."""

    m = LP.legal_masks(out)
    fam, slot, cell, ab = [0], [0], [0], [0]
    if bool(m["legal_any"][i]):
        for s in m["slot_legal"][i].nonzero(as_tuple=True)[0].tolist():
            fam += [1] * S.N_CELLS
            slot += [s] * S.N_CELLS
            cell += list(range(S.N_CELLS))
            ab += [0] * S.N_CELLS
    if bool(m["ability_avail"][i]):
        for k in m["ability_ready"][i].nonzero(as_tuple=True)[0].tolist():
            fam.append(2)
            slot.append(0)
            cell.append(0)
            ab.append(k)
    a = len(fam)
    rep = {k: out[k][i:i + 1].expand(a, *out[k].shape[1:]) for k in HEADS}
    act = {"family": torch.tensor(fam), "slot": torch.tensor(slot), "cell": torch.tensor(cell),
           "ability": torch.tensor(ab)}
    return rep, act


def folded_family_probs(out, dt):
    """Independent folded family probabilities from ``hazard_probs`` (probability space)."""

    p = hazard_probs(out["family"], out["ability_avail"], torch.as_tensor(dt, dtype=torch.float64)
                     .expand(out["family"].shape[0]))
    la = (out["slot"] > MASK_FILL / 2).any(-1).double()
    return torch.stack([p[:, 0] + p[:, 1] * (1 - la), p[:, 1] * la, p[:, 2]], -1)


def np_softmax_legal(x, ok):
    x = np.asarray(x, np.float64)[ok]
    e = np.exp(x - x.max())
    return e / e.sum()


def np_kl(p, q):
    nz = p > 0
    return float(np.sum(p[nz] * (np.log(p[nz]) - np.log(q[nz]))))


def np_ent(p):
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))


# =========================================================================== family
class TestFamily(unittest.TestCase):
    def test_matches_hazard_probs_folded(self):
        out = cat_rows(peek_outputs(), synth_out())
        for dt in (0.0, -3.0, 1.0, 10.0, 37.5, 1e4):
            m = LP.legal_masks(out)
            lp = LP.family_log_probs(out["family"], m["ability_avail"], m["legal_any"], dt)
            self.assertEqual(lp.dtype, torch.float64)
            ref = folded_family_probs(out, dt)
            self.assertTrue(torch.allclose(lp.exp(), ref, rtol=1e-12, atol=1e-15), (dt, (lp.exp() - ref).abs().max()))
            self.assertTrue(torch.allclose(lp.exp().sum(-1), torch.ones(lp.shape[0], dtype=torch.float64),
                                           atol=1e-14))
            poss = LP.family_possible(m["ability_avail"], m["legal_any"], dt, out["family"])
            self.assertTrue(bool((lp[~poss] == LP.LOG_ZERO).all()))
            self.assertTrue(bool(torch.isfinite(lp).all()))

    def test_small_rate_is_stable(self):
        # rate ~ 1e-20: naive log(1 - exp(-rate)) is -inf, the closed form keeps log(rate * share)
        fam = torch.tensor([[0.0, -10.0, -11.0]], dtype=torch.float64)
        avail, la = torch.tensor([True]), torch.tensor([True])
        odds = math.exp(-10.0) + math.exp(-11.0)
        dt = 1e-20 * S.U_TICKS / odds
        lp = LP.family_log_probs(fam, avail, la, torch.tensor([dt], dtype=torch.float64))[0]
        self.assertEqual(1.0 - math.exp(-1e-20), 0.0)                       # the naive form underflows
        self.assertAlmostEqual(float(lp[1]), math.log(1e-20 * math.exp(-10.0) / odds), places=9)
        self.assertAlmostEqual(float(lp[2]), math.log(1e-20 * math.exp(-11.0) / odds), places=9)
        self.assertAlmostEqual(float(lp[0]), -1e-20, places=30)

    def test_folding_and_forced(self):
        out = synth_out()
        m = LP.legal_masks(out)
        self.assertEqual(m["forced"].tolist(), [False, True, False, False, False])
        self.assertTrue(torch.equal(LP.forced_mask(out), m["forced"]))
        lp = LP.family_log_probs(out["family"], m["ability_avail"], m["legal_any"], 10.0)
        p = hazard_probs(out["family"], out["ability_avail"], torch.full((5,), 10.0, dtype=torch.float64))
        # blocked row with an ability: wait_exec = P_none + P_play, play impossible, ability unchanged
        self.assertAlmostEqual(float(lp[0, 0].exp()), float(p[0, 0] + p[0, 1]), places=14)
        self.assertEqual(float(lp[0, 1]), LP.LOG_ZERO)
        self.assertAlmostEqual(float(lp[0, 2].exp()), float(p[0, 2]), places=14)
        # forced row: wait with probability 1
        self.assertAlmostEqual(float(lp[1, 0]), 0.0, places=14)
        self.assertEqual(lp[1, 1:].tolist(), [LP.LOG_ZERO, LP.LOG_ZERO])


# =========================================================================== executed action
class TestExecutedLogp(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        real = peek_outputs()
        rows = pick_rows(real) + list(range(0, real["family"].shape[0], 17))
        out = cat_rows(sub(real, rows), synth_out())
        worst = 0.0
        for t in TEMPS:
            ot = LP.apply_temperature(out, t, t, t)
            for dt in (0.0, 3.0, 10.0, 1000.0):
                for i in range(out["family"].shape[0]):
                    rep, act = enumerate_actions(ot, i)
                    s = float(LP.executed_logp(rep, dt, act).exp().sum())
                    worst = max(worst, abs(s - 1.0))
        self.assertLess(worst, 1e-12)
        print(f"sum over executable actions: max |sum - 1| = {worst:.1e}")

    def test_matches_sample_actions_frequencies(self):
        """exp(executed_logp) vs 1e5 sample_actions draws per row and T, grouped by executed action
        (family, slot, cell mass-bin, ability); blocked plays checked against P_play."""

        real = peek_outputs()
        out = cat_rows(sub(real, pick_rows(real)), synth_out())
        r = out["family"].shape[0]
        dt = unit_rate_dt(out)
        gen = torch.Generator().manual_seed(2026)
        zs, zero_viol, blocked_rows = [], 0, 0
        for t in TEMPS:
            ot = LP.apply_temperature(out, t, t, t)
            # cell mass bins per (row, slot) from the tempered cell distribution
            pc = torch.softmax(ot["cells"].double(), -1)
            order = pc.argsort(-1, descending=True)
            ps = pc.gather(-1, order)
            before = ps.cumsum(-1) - ps
            bins = (before * NB).floor().long().clamp(0, NB - 1)
            bin_map = torch.empty_like(bins).scatter_(-1, order, bins)
            # expected group probabilities from the enumeration
            exp_p = torch.zeros(r, K, dtype=torch.float64)
            for i in range(r):
                rep, act = enumerate_actions(ot, i)
                p = LP.executed_logp(rep, dt[i], act).exp()
                key = torch.where(act["family"] == 0, 0,
                                  torch.where(act["family"] == 1, 1 + act["slot"] * NB
                                              + bin_map[i, act["slot"], act["cell"]],
                                              1 + S.HAND_SLOTS * NB + act["ability"]))
                exp_p[i].scatter_add_(0, key, p)
            # draws
            o32 = {"family": ot["family"], "ability_avail": ot["ability_avail"], "slot": ot["slot"].float(),
                   "ability": ot["ability"].float(), "cells": ot["cells"].float()}
            m = 1000
            counts = torch.zeros(r * K, dtype=torch.long)
            blocked = torch.zeros(r, dtype=torch.long)
            for c in range(N_DRAWS // m):
                idx = torch.arange(r).repeat_interleave(m)
                a = sample_actions({k: v[idx] for k, v in o32.items()}, dt[idx], generator=gen)
                if c == 0:   # every drawn action is executable under executed_logp
                    lp = LP.executed_logp({k: v[idx] for k, v in ot.items()}, dt[idx], a)
                    self.assertTrue(bool((lp > LP.LOG_ZERO / 2).all()))
                    self.assertTrue(bool(torch.isfinite(lp).all()))
                f = a["family"]
                key = torch.where(f == 0, 0, torch.where(f == 1, 1 + a["slot"] * NB
                                                         + bin_map[idx, a["slot"], a["cell"]],
                                                         1 + S.HAND_SLOTS * NB + a["ability"]))
                counts += torch.bincount(idx * K + key, minlength=r * K)
                blocked += torch.bincount(idx[a["play_blocked"]], minlength=r)
            counts = counts.view(r, K).double()
            n = float(N_DRAWS)
            for i in range(r):
                for k in range(K):
                    p = float(exp_p[i, k])
                    if p <= 0.0 or p >= 1.0 - 1e-15:
                        zero_viol += int(abs(counts[i, k] - n * p) > 0.5)
                        continue
                    zs.append((float(counts[i, k]) - n * p) / math.sqrt(n * p * (1 - p)))
            p_play = hazard_probs(ot["family"], ot["ability_avail"], dt)[:, 1]
            la = LP.legal_masks(ot)["legal_any"]
            for i in range(r):
                if not bool(la[i]):
                    blocked_rows += 1
                    pp = float(p_play[i])
                    zs.append((float(blocked[i]) - n * pp) / math.sqrt(n * pp * (1 - pp)))
                else:
                    self.assertEqual(int(blocked[i]), 0)
        zs = np.abs(np.asarray(zs))
        g = len(zs)
        # 3-sigma per group; with g groups about 0.27% exceed 3 by chance, so allow the binomial
        # 99.9% quantile of exceedances, and none beyond 5 sigma
        q, acc, k_allow = 0.0027, 0.0, 0
        for k in range(g + 1):
            acc += math.comb(g, k) * q ** k * (1 - q) ** (g - k)
            if acc >= 0.999:
                k_allow = k
                break
        n3 = int((zs > 3).sum())
        print(f"empirical check: {g} groups ({blocked_rows} blocked-row checks), max |z| = {zs.max():.2f}, "
              f"|z| > 3: {n3} (allowed {k_allow}), impossible-group violations: {zero_viol}")
        self.assertEqual(zero_viol, 0)
        self.assertGreater(blocked_rows, 0)
        self.assertLess(zs.max(), 5.0)
        self.assertLessEqual(n3, k_allow)

    def test_blocked_play_is_wait(self):
        out = synth_out()
        dt = torch.full((5,), 10.0, dtype=torch.float64)
        wait = {"family": torch.zeros(5, dtype=torch.long), "slot": torch.zeros(5, dtype=torch.long),
                "cell": torch.zeros(5, dtype=torch.long), "ability": torch.zeros(5, dtype=torch.long)}
        drawn = dict(wait, family=torch.ones(5, dtype=torch.long))                 # unfolded draw
        lw = LP.executed_logp(out, dt, wait)
        lb = LP.executed_logp(out, dt, drawn)
        self.assertTrue(torch.equal(lw[:2], lb[:2]))                                # rows 0, 1 blocked
        flagged = dict(drawn, play_blocked=torch.tensor([True, True, False, False, False]))
        self.assertTrue(torch.equal(LP.executed_family(out, flagged)[:2], torch.zeros(2, dtype=torch.long)))
        p = folded_family_probs(out, 10.0)
        self.assertTrue(torch.allclose(lw.exp(), p[:, 0], rtol=1e-13))
        # forced row: wait has probability exactly 1 (log 0)
        self.assertAlmostEqual(float(lw[1]), 0.0, places=15)

    def test_terms_and_learner_cells(self):
        real = peek_outputs()
        n = real["family"].shape[0]
        dt = torch.full((n,), 10.0, dtype=torch.float64)
        a = sample_actions(real, unit_rate_dt(real), generator=torch.Generator().manual_seed(3))
        terms = LP.executed_logp_terms(real, dt, a)
        fam = terms["executed_family"]
        self.assertTrue(torch.equal(fam, a["family"]))
        for k, f in (("slot", 1), ("cell", 1), ("ability", 2)):
            self.assertTrue(bool((terms[k][fam != f] == 0).all()), k)
        self.assertTrue(bool((fam == 1).any()) and bool((fam == 2).any()) and bool((fam == 0).any()))
        # learner form: cells of the executed slot only, [n, 576]
        learner = dict(real, cells=real["cells"][torch.arange(n), a["slot"]])
        self.assertTrue(torch.equal(LP.executed_logp(learner, dt, a), terms["total"]))
        # single legal slot -> slot term 0; manual check of a play row
        lp_slot = torch.log_softmax(real["slot"].double(), -1)
        i = int((fam == 1).nonzero()[0])
        s, c = int(a["slot"][i]), int(a["cell"][i])
        manual = (float(LP.family_log_probs(real["family"][i:i + 1], real["ability_avail"][i:i + 1],
                                            torch.tensor([True]), dt[i:i + 1])[0, 1])
                  + float(lp_slot[i, s]) + float(torch.log_softmax(real["cells"][i, s].double(), -1)[c]))
        self.assertAlmostEqual(float(terms["total"][i]), manual, places=12)
        one = (LP.legal_masks(real)["n_slots"] == 1) & (fam == 1)
        if bool(one.any()):
            self.assertTrue(bool((terms["slot"][one].abs() < 1e-12).all()))

    def test_gradients_finite(self):
        out = synth_out()
        leaves = {k: out[k].clone().requires_grad_(True) for k in ("family", "slot", "ability", "cells")}
        o = dict(out, **leaves)
        dt = torch.tensor([10.0, 10.0, 0.0, 1e-30, 1e5], dtype=torch.float64)
        acts = [{"family": torch.tensor(f), "slot": torch.tensor([0, 0, 2, 1, 3]),
                 "cell": torch.tensor([0, 0, 17, 5, 9]), "ability": torch.tensor([3, 0, 0, 0, 60])}
                for f in ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 0, 1, 2, 2])]
        ref = {k: (v.detach() + 0.1) if v.dtype == torch.float64 else v for k, v in out.items()}
        total = 0.0
        for a in acts:
            total = total + LP.executed_logp(o, dt, a).sum()
            kl = LP.head_kl(o, ref, dt, a)
            en = LP.head_entropy(o, dt, a)
            for d in (kl, en):
                for k, v in d.items():
                    if not k.startswith("mask"):
                        total = total + v.sum()
        total.backward()
        for k, v in leaves.items():
            self.assertTrue(bool(torch.isfinite(v.grad).all()), k)


# =========================================================================== temperature
class TestTemperature(unittest.TestCase):
    def test_identity_at_one(self):
        out = peek_outputs()
        res = LP.apply_temperature(out, 1.0, 1.0, 1.0)
        for k in HEADS:
            self.assertIs(res[k], out[k])
            self.assertTrue(torch.equal(res[k].view(torch.int64) if res[k].dtype == torch.float64 else res[k],
                                        out[k].view(torch.int64) if out[k].dtype == torch.float64 else out[k]))
        o32 = {k: (v.float() if v.is_floating_point() else v) for k, v in out.items()}
        r32 = LP.apply_temperature(o32)
        self.assertTrue(all(r32[k] is o32[k] for k in HEADS))

    def test_masked_entries_and_legality(self):
        out = cat_rows(peek_outputs(), synth_out())
        for t in TEMPS[1:] + (0.1, 0.05, 10.0):
            res = LP.apply_temperature(out, t, t, t)
            self.assertIs(res["family"], out["family"])                       # gate untouched
            self.assertIs(res["ability_avail"], out["ability_avail"])
            for k in ("slot", "cells", "ability"):
                ok = out[k] > MASK_FILL / 2
                self.assertTrue(torch.equal(res[k] > MASK_FILL / 2, ok), (k, t))
                self.assertTrue(bool((res[k][~ok] == MASK_FILL).all()), (k, t))
                self.assertTrue(torch.equal(res[k][ok], out[k][ok] / t), (k, t))
            # sample_actions sees the same legality: blocked rows unchanged
            dt = unit_rate_dt(out)
            a0 = sample_actions(out, dt, generator=torch.Generator().manual_seed(5))
            a1 = sample_actions(res, dt, generator=torch.Generator().manual_seed(5))
            self.assertTrue(torch.equal(a0["play_blocked"], a1["play_blocked"]))
            self.assertTrue(torch.equal(a0["family"], a1["family"]))
        # per-head temperatures are independent
        res = LP.apply_temperature(out, 0.5, 1.0, 2.0)
        self.assertIs(res["cells"], out["cells"])
        self.assertFalse(torch.equal(res["slot"], out["slot"]))

    def test_invalid_temperature(self):
        out = synth_out()
        for bad in (0.0, -1.0, float("nan"), float("inf"), 1e-9, 0.5 * LP.T_MIN):
            with self.assertRaises(ValueError):
                LP.apply_temperature(out, t_cell=bad)


# =========================================================================== KL and entropy
def random_actions(out, seed):
    return sample_actions(out, unit_rate_dt(out), generator=torch.Generator().manual_seed(seed))


class TestKLEntropy(unittest.TestCase):
    def test_kl_zero_for_identical_models(self):
        out = cat_rows(peek_outputs(0), synth_out())
        a = random_actions(out, 1)
        same = {k: v.clone() for k, v in out.items()}
        for dt in (10.0, unit_rate_dt(out)):
            kl = LP.head_kl(out, same, dt, a)
            for k in ("kl_family", "kl_slot", "kl_cell", "kl_ability", "kl_joint"):
                self.assertEqual(float(kl[k].abs().max()), 0.0, k)

    def test_kl_matches_brute_force(self):
        th = peek_outputs(0)
        ref = peek_outputs(1)
        n = th["family"].shape[0]
        rows = sorted(set(pick_rows(th) + list(range(0, n, 11))))
        th, ref = sub(th, rows), sub(ref, rows)
        a = random_actions(th, 2)
        cases = [(th, ref, "seed0 || seed1")]
        cases.append((LP.apply_temperature(th, 0.5, 0.25, 2.0), th, "tempered || untempered"))
        worst = 0.0
        for p_out, q_out, name in cases:
            for dt in (10.0, 250.0):
                kl = LP.head_kl(p_out, q_out, dt, a)
                fp, fq = folded_family_probs(p_out, dt).numpy(), folded_family_probs(q_out, dt).numpy()
                for j in range(len(rows)):
                    # joint: sum over every executable action of pi (log pi - log ref)
                    rep_p, act = enumerate_actions(p_out, j)
                    rep_q = {k: q_out[k][j:j + 1].expand(*rep_p[k].shape) for k in HEADS}
                    lp = LP.executed_logp(rep_p, dt, act)
                    lq = LP.executed_logp(rep_q, dt, act)
                    brute = float((lp.exp() * (lp - lq)).sum())
                    exact = float(kl["kl_joint"][j])
                    worst = max(worst, abs(brute - exact) / max(1.0, abs(brute)))
                    # per head, independent numpy
                    self.assertAlmostEqual(float(kl["kl_family"][j]), np_kl(fp[j], fq[j]), places=10)
                    ok = (p_out["slot"][j] > MASK_FILL / 2).numpy()
                    if ok.any():
                        self.assertAlmostEqual(float(kl["kl_slot"][j]), np_kl(
                            np_softmax_legal(p_out["slot"][j], ok), np_softmax_legal(q_out["slot"][j], ok)),
                            places=10)
                    s = int(a["slot"][j])
                    okc = (p_out["cells"][j, s] > MASK_FILL / 2).numpy()
                    self.assertAlmostEqual(float(kl["kl_cell"][j]), np_kl(
                        np_softmax_legal(p_out["cells"][j, s], okc), np_softmax_legal(q_out["cells"][j, s], okc))
                        if okc.any() else 0.0, places=10)
                    oka = (p_out["ability"][j] > MASK_FILL / 2).numpy()
                    if oka.any():
                        self.assertAlmostEqual(float(kl["kl_ability"][j]), np_kl(
                            np_softmax_legal(p_out["ability"][j], oka),
                            np_softmax_legal(q_out["ability"][j], oka)), places=10)
                    else:
                        self.assertEqual(float(kl["kl_ability"][j]), 0.0)
        self.assertLess(worst, 1e-10)
        print(f"joint KL exact vs brute force: max rel err {worst:.1e}")

    def test_entropy_matches_brute_force(self):
        out = cat_rows(sub(peek_outputs(0), pick_rows(peek_outputs(0))), synth_out())
        a = random_actions(out, 4)
        for t in (1.0, 0.25, 5.0):
            ot = LP.apply_temperature(out, t, t, t)
            dt = unit_rate_dt(ot)
            en = LP.head_entropy(ot, dt, a)
            fp = folded_family_probs(ot, dt).numpy()
            for j in range(ot["family"].shape[0]):
                rep, act = enumerate_actions(ot, j)
                lp = LP.executed_logp(rep, dt[j], act)
                p = lp.exp()
                brute = float(-(torch.where(p > 0, p * lp, torch.zeros_like(p))).sum())
                self.assertAlmostEqual(float(en["ent_joint"][j]), brute, places=9)
                self.assertAlmostEqual(float(en["ent_family"][j]), np_ent(fp[j]), places=10)
                ok = (ot["slot"][j] > MASK_FILL / 2).numpy()
                self.assertAlmostEqual(float(en["ent_slot"][j]),
                                       np_ent(np_softmax_legal(ot["slot"][j], ok)) if ok.any() else 0.0, places=10)
                s = int(a["slot"][j])
                okc = (ot["cells"][j, s] > MASK_FILL / 2).numpy()
                self.assertAlmostEqual(float(en["ent_cell"][j]),
                                       np_ent(np_softmax_legal(ot["cells"][j, s], okc)) if okc.any() else 0.0,
                                       places=10)
                oka = (ot["ability"][j] > MASK_FILL / 2).numpy()
                self.assertAlmostEqual(float(en["ent_ability"][j]),
                                       np_ent(np_softmax_legal(ot["ability"][j], oka)) if oka.any() else 0.0,
                                       places=10)

    def test_masks(self):
        out = synth_out()
        dt = torch.full((5,), 10.0, dtype=torch.float64)
        # rows: 0 blocked+2 ready, 1 forced, 2 single slot, 3 three slots+1 ready, 4 four slots+3 ready
        a = {"family": torch.tensor([2, 0, 1, 1, 2]), "slot": torch.tensor([0, 0, 2, 1, 0]),
             "cell": torch.tensor([0, 0, 17, 4, 0]), "ability": torch.tensor([10, 0, 0, 0, 5])}
        ref = {k: (v + 0.3) if v.dtype == torch.float64 else v for k, v in out.items()}
        for d in (LP.head_kl(out, ref, dt, a), LP.head_entropy(out, dt, a)):
            self.assertEqual(d["mask_family"].tolist(), [True, False, True, True, True])
            self.assertEqual(d["mask_slot"].tolist(), [False, False, False, True, False])
            self.assertEqual(d["mask_cell"].tolist(), [False, False, True, True, False])
            self.assertEqual(d["mask_ability"].tolist(), [True, False, False, False, True])
        en = LP.head_entropy(out, dt, a)
        self.assertEqual(float(en["ent_slot"][2]), 0.0)          # single legal slot: zero entropy
        self.assertEqual(float(en["ent_slot"][0]), 0.0)          # no legal slot: 0, not log 4
        self.assertEqual(float(en["ent_ability"][2]), 0.0)       # no ready unit: 0, not log 64
        self.assertAlmostEqual(float(en["ent_family"][1]), 0.0, places=14)   # forced row
        ok6 = (out["cells"][2, 2] > MASK_FILL / 2).numpy()
        self.assertEqual(int(ok6.sum()), 6)
        self.assertAlmostEqual(float(en["ent_cell"][2]), np_ent(np_softmax_legal(out["cells"][2, 2], ok6)),
                               places=12)
        # a blocked row (0) is non-forced; its family entropy is over {wait_exec, ability}
        p = folded_family_probs(out, 10.0)[0].numpy()
        self.assertEqual(p[1], 0.0)
        self.assertAlmostEqual(float(en["ent_family"][0]), np_ent(p), places=12)
        # real rows: the masks follow the legality counts
        real = peek_outputs()
        ar = random_actions(real, 9)
        kl = LP.head_kl(real, peek_outputs(1), 10.0, ar)
        m = LP.legal_masks(real)
        self.assertTrue(torch.equal(kl["mask_family"], ~m["forced"]))
        self.assertTrue(torch.equal(kl["mask_slot"], (ar["family"] == 1) & (m["n_slots"] >= 2)))
        self.assertTrue(torch.equal(kl["mask_cell"], ar["family"] == 1))
        self.assertTrue(torch.equal(kl["mask_ability"], (ar["family"] == 2) & (m["n_ready"] >= 2)))


if __name__ == "__main__":
    unittest.main()
