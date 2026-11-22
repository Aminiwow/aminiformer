"""Tests for src/amini/ppo/losses.py.

* surrogate: r = 1 gives minus the weighted mean advantage; clip and dual-clip cases with
  positive / negative A inside / outside the clip, gradient exactly zero where clipped;
* KL anchor normalised by the surrogate denominator D (hand computation), own-row means,
  KL / surrogate gradient-norm ratio;
* value MSE (no clip, no Huber), optional BC on the PPO denominator;
* k3 non-negative and 0 at r = 1; update metrics; early-stop thresholds; update-0 identity;
* global-D vs local-D behaviour (micro-batch shares add up to the step mean);
* a tiny real model (tests/v7/common.py) through the hazard gate: identity and finite grads
  (skipped without the private dataset).

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_losses.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "tests" / "v7"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from amini.ppo import losses as L  # noqa: E402

D64 = torch.float64


def t(*xs) -> torch.Tensor:
    return torch.tensor(xs, dtype=D64)


def surrogate_grad(log_ratio: float, adv: float, w: float = 1.0, **kw):
    """(loss, d loss / d logp_new) for one row."""
    lp = torch.tensor([log_ratio], dtype=D64, requires_grad=True)
    loss = L.clipped_surrogate(lp, torch.zeros(1, dtype=D64), t(adv), t(w), **kw)
    loss.backward()
    return float(loss), float(lp.grad[0])


class SurrogateTest(unittest.TestCase):

    def test_ratio_one_is_minus_weighted_mean_advantage(self):
        g = torch.Generator().manual_seed(0)
        lp = torch.randn(50, generator=g, dtype=D64)
        adv = torch.randn(50, generator=g, dtype=D64)
        w = torch.rand(50, generator=g, dtype=D64) * 6.0
        w[::7] = 0.0                                          # forced rows
        loss = L.clipped_surrogate(lp, lp.clone(), adv, w)
        expect = -(w * adv).sum() / w.sum()
        self.assertAlmostEqual(float(loss), float(expect), places=12)

    def test_positive_advantage(self):
        # inside the clip: objective r A, gradient -w r A / D
        loss, grad = surrogate_grad(math.log(1.1), 2.0)
        self.assertAlmostEqual(loss, -1.1 * 2.0, places=12)
        self.assertAlmostEqual(grad, -1.1 * 2.0, places=12)
        # above 1 + eps: clipped to (1 + eps) A, zero gradient
        loss, grad = surrogate_grad(math.log(1.5), 2.0)
        self.assertAlmostEqual(loss, -1.25 * 2.0, places=12)
        self.assertEqual(grad, 0.0)
        # below 1 - eps with A > 0: min picks r A (pessimistic), gradient flows
        loss, grad = surrogate_grad(math.log(0.5), 2.0)
        self.assertAlmostEqual(loss, -0.5 * 2.0, places=12)
        self.assertAlmostEqual(grad, -0.5 * 2.0, places=12)

    def test_negative_advantage(self):
        # below 1 - eps: clipped to (1 - eps) A, zero gradient
        loss, grad = surrogate_grad(math.log(0.5), -2.0)
        self.assertAlmostEqual(loss, 0.75 * 2.0, places=12)
        self.assertEqual(grad, 0.0)
        # inside the clip
        loss, grad = surrogate_grad(math.log(0.9), -2.0)
        self.assertAlmostEqual(loss, 0.9 * 2.0, places=12)
        self.assertAlmostEqual(grad, 0.9 * 2.0, places=12)
        # 1 + eps < r < c: standard PPO does not clip; objective r A, gradient flows
        loss, grad = surrogate_grad(math.log(2.0), -2.0)
        self.assertAlmostEqual(loss, 2.0 * 2.0, places=12)
        self.assertAlmostEqual(grad, 2.0 * 2.0, places=12)

    def test_dual_clip(self):
        # r > c with A < 0: the loss is bounded at -c A, zero gradient
        loss, grad = surrogate_grad(math.log(5.0), -2.0)
        self.assertAlmostEqual(loss, 2.5 * 2.0, places=12)
        self.assertEqual(grad, 0.0)
        # without the dual clip the loss grows with r
        loss, grad = surrogate_grad(math.log(5.0), -2.0, dual_clip=None)
        self.assertAlmostEqual(loss, 5.0 * 2.0, places=12)
        self.assertAlmostEqual(grad, 5.0 * 2.0, places=12)
        # the dual clip never touches A >= 0 rows
        loss, _ = surrogate_grad(math.log(5.0), 2.0)
        self.assertAlmostEqual(loss, -1.25 * 2.0, places=12)
        with self.assertRaises(ValueError):
            L.clipped_objective(t(1.0), t(-1.0), eps=0.2, dual_clip=1.1)

    def test_objective_vectorised_matches_cases(self):
        r = t(0.5, 0.9, 1.1, 1.5, 2.0, 5.0, 0.5, 0.9, 1.1, 1.5, 2.0, 5.0)
        a = t(*([1.0] * 6 + [-1.0] * 6))
        got = L.clipped_objective(r, a)
        want = t(0.5, 0.9, 1.1, 1.25, 1.25, 1.25, -0.75, -0.9, -1.1, -1.5, -2.0, -2.5)
        self.assertTrue(torch.allclose(got, want, atol=1e-12), got)

    def test_weight_and_prox_detached(self):
        lp = torch.zeros(3, dtype=D64, requires_grad=True)
        prox = torch.zeros(3, dtype=D64, requires_grad=True)
        w = torch.ones(3, dtype=D64, requires_grad=True)
        L.clipped_surrogate(lp, prox, t(1.0, -1.0, 2.0), w).backward()
        self.assertIsNone(prox.grad)
        self.assertIsNone(w.grad)
        self.assertTrue(torch.allclose(lp.grad, t(-1.0, 1.0, -2.0) / 3.0))

    def test_all_forced_rows_give_zero_not_nan(self):
        lp = torch.zeros(4, dtype=D64, requires_grad=True)
        loss = L.clipped_surrogate(lp, torch.zeros(4, dtype=D64), t(1, 2, 3, 4), torch.zeros(4, dtype=D64))
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(lp.grad).all())
        self.assertEqual(float(L.value_loss(t(0.1), t(1.0), t(0.0))), 0.0)

    def test_autograd_finite_at_extremes(self):
        g = torch.Generator().manual_seed(1)
        lp = (torch.randn(200, generator=g) * 30).requires_grad_(True)   # fp32, ratios up to e^±90
        adv = torch.randn(200, generator=g) * 100
        w = torch.rand(200, generator=g) * 7
        loss = L.clipped_surrogate(lp, torch.zeros(200), adv, w)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(lp.grad).all())
        # direct fp32 overflow rows (the random sample above rarely hits the 0 * inf branch)
        x = torch.tensor([100.0, 100.0, -100.0, -100.0], requires_grad=True)
        L.clipped_surrogate(x, torch.zeros(4), torch.tensor([1.0, -1.0, 1.0, -1.0]), torch.ones(4)).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_upper_clamp_without_dual_clip_is_exact(self):
        # no dual clip: an A < 0 row with log r = 25 is unclipped; objective and gradient are r A
        loss, grad = surrogate_grad(25.0, -1.0, dual_clip=None)
        self.assertAlmostEqual(loss / math.exp(25.0), 1.0, places=12)
        self.assertAlmostEqual(grad / math.exp(25.0), 1.0, places=12)
        loss, grad = surrogate_grad(25.0, -1.0)                       # dual clip on: bounded at c |A|
        self.assertAlmostEqual(loss, 2.5, places=12)
        self.assertEqual(grad, 0.0)

    def test_zero_weight_rows_ignored_even_if_nonfinite(self):
        lp = t(-1.0, float("nan"), -2.0).requires_grad_(True)
        prox = t(-1.1, -1.0, float("-inf"))
        adv = t(1.0, float("nan"), float("inf"))
        w = t(1.0, 0.0, 0.0)
        loss = L.clipped_surrogate(lp, prox, adv, w)
        loss.backward()
        self.assertAlmostEqual(float(loss), -math.exp(0.1) * 1.0, places=12)
        self.assertTrue(torch.equal(lp.grad[1:], torch.zeros(2, dtype=D64)))
        # a NaN on a LIVE row still propagates (the learner's skip signal)
        self.assertTrue(math.isnan(float(L.clipped_surrogate(t(float("nan")), t(0.0), t(1.0), t(1.0)))))

    def test_global_denominator_checked(self):
        x = t(0.0, 0.0).requires_grad_(True)
        self.assertTrue(math.isnan(float(L.clipped_surrogate(x, t(0.0, 0.0), t(1.0, 1.0), t(1.0, 1.0),
                                                             denom=0.0))))
        self.assertTrue(math.isnan(float(L.clipped_surrogate(x, t(0.0, 0.0), t(1.0, 1.0), t(1.0, 1.0),
                                                             denom=1.0))))
        # within DENOM_RTOL of sum(w) (rounding of a step-wide sum) is accepted
        self.assertAlmostEqual(float(L.clipped_surrogate(x, t(0.0, 0.0), t(1.0, 1.0), t(1.0, 1.0),
                                                         denom=2.0 * (1 - 1e-6))), -1.0, places=5)
        # an all-forced micro-batch with D = 0 is fine
        self.assertEqual(float(L.clipped_surrogate(x, t(0.0, 0.0), t(1.0, 1.0), t(0.0, 0.0), denom=0.0)), 0.0)


class DenominatorTest(unittest.TestCase):
    """Global D: micro-batch shares add up to the full-step weighted mean (step-wide
    denominators); local D per micro-batch averages per micro-batch instead."""

    def setUp(self):
        g = torch.Generator().manual_seed(2)
        n = 30
        self.lp = (torch.randn(n, generator=g, dtype=D64) * 0.3)
        self.prox = torch.zeros(n, dtype=D64)
        self.adv = torch.randn(n, generator=g, dtype=D64)
        self.w = torch.rand(n, generator=g, dtype=D64) * 6.0
        self.w[:5] = 0.0
        self.parts = [slice(0, 4), slice(4, 21), slice(21, 30)]   # deliberately uneven

    def test_global_denominator_shares_sum_to_full_step(self):
        lp_full = self.lp.clone().requires_grad_(True)
        full = L.clipped_surrogate(lp_full, self.prox, self.adv, self.w)
        full.backward()
        D = L.surrogate_denominator(self.w)
        lp_mb = self.lp.clone().requires_grad_(True)
        total = 0.0
        for s in self.parts:
            part = L.clipped_surrogate(lp_mb[s], self.prox[s], self.adv[s], self.w[s], denom=D)
            part.backward()
            total += float(part)
        self.assertAlmostEqual(total, float(full), places=12)
        self.assertTrue(torch.allclose(lp_mb.grad, lp_full.grad, atol=1e-14))

    def test_local_denominator_is_per_call_mean(self):
        full = float(L.clipped_surrogate(self.lp, self.prox, self.adv, self.w))
        local = [float(L.clipped_surrogate(self.lp[s], self.prox[s], self.adv[s], self.w[s]))
                 for s in self.parts]
        # each local value is that micro-batch's weighted mean ...
        # (the first micro-batch is all forced rows: its local value is 0, not NaN)
        self.assertEqual(local[0], 0.0)
        for s, v in zip(self.parts[1:], local[1:]):
            obj = L.clipped_objective(torch.exp(self.lp[s]), self.adv[s])
            self.assertAlmostEqual(v, float(-(self.w[s] * obj).sum() / self.w[s].sum()), places=12)
        # ... so their sum is not the step mean (documented: use a global D when micro-batching)
        self.assertGreater(abs(sum(local) - full), 1e-3)
        # weighting the local means by their weight share recovers it
        ws = [float(self.w[s].sum()) for s in self.parts]
        self.assertAlmostEqual(sum(v * x for v, x in zip(local, ws)) / sum(ws), full, places=12)

    def test_value_and_kl_follow_same_convention(self):
        v = torch.tanh(self.lp)
        G = self.adv.clamp(-1, 1)
        D = self.w.sum()
        full = float(L.value_loss(v, G, self.w))
        parts = sum(float(L.value_loss(v[s], G[s], self.w[s], denom=D)) for s in self.parts)
        self.assertAlmostEqual(parts, full, places=12)
        kl = self.lp.abs()
        full_kl = float(L.kl_penalty({"family": (kl, self.w)}, D).loss)
        parts_kl = sum(float(L.kl_penalty({"family": (kl[s], self.w[s])}, D).loss) for s in self.parts)
        self.assertAlmostEqual(parts_kl, full_kl, places=14)
        with self.assertRaises(ValueError):
            L.clipped_surrogate(self.lp, self.prox, self.adv, self.w, denom=torch.ones(2))


class KLTest(unittest.TestCase):

    def setUp(self):
        # 6 rows: 0-3 waits (HT weight 6), 4-5 plays (weight 1); row 3 forced (weight 0)
        self.w = t(6.0, 6.0, 6.0, 0.0, 1.0, 1.0)
        self.D = float(self.w.sum())
        self.fam_kl = t(0.001, 0.002, 0.0005, 0.3, 0.05, 0.04)
        self.slot_kl = t(0.2)              # only row 4 has >= 2 legal slots
        self.slot_w = t(1.0)
        self.cell_kl = t(0.5, 0.7)         # both play rows, executed card
        self.cell_w = t(1.0, 1.0)

    def heads(self):
        return {"family": (self.fam_kl, self.w), "slot": (self.slot_kl, self.slot_w),
                "cells": (self.cell_kl, self.cell_w)}

    def test_normalised_by_surrogate_denominator(self):
        beta = 0.01
        pen = L.kl_penalty(self.heads(), self.w.sum(), beta=beta)
        fam = sum(w * k for w, k in zip([6.0, 6.0, 6.0, 0.0, 1.0, 1.0],
                                         [0.001, 0.002, 0.0005, 0.3, 0.05, 0.04]))
        hand = beta * (fam + 1.0 * 0.2 + 0.5 + 0.7) / self.D
        self.assertAlmostEqual(float(pen.loss), hand, places=14)
        self.assertAlmostEqual(float(pen.terms["slot"]), beta * 0.2 / self.D, places=14)
        self.assertAlmostEqual(float(pen.terms["cells"]), beta * 1.2 / self.D, places=14)
        # own-row means (monitor only) are NOT what the loss uses
        self.assertAlmostEqual(float(pen.own_row_mean["slot"]), 0.2, places=14)
        self.assertAlmostEqual(float(pen.own_row_mean["cells"]), 0.6, places=14)
        self.assertAlmostEqual(float(pen.own_row_mean["family"]), fam / self.D, places=14)
        self.assertAlmostEqual(float(pen.denom), self.D, places=12)
        # the own-row form would be D / N_play times stronger on cells
        self.assertGreater(float(pen.own_row_mean["cells"]) * beta / float(pen.terms["cells"]), 9.0)
        # monitor values stay tensors (no host sync per micro-batch); log() converts once
        self.assertTrue(all(torch.is_tensor(v) for v in pen.own_row_mean.values()))
        logged = pen.log()
        self.assertAlmostEqual(logged["kl/cells"], 0.6, places=14)
        self.assertAlmostEqual(logged["kl_term/slot"], beta * 0.2 / self.D, places=14)
        self.assertEqual(logged["kl_weight_sum/cells"], 2.0)

    def test_per_head_beta(self):
        pen = L.kl_penalty(self.heads(), self.D, beta={"family": 0.01, "slot": 0.02, "cells": 0.0})
        self.assertAlmostEqual(float(pen.terms["slot"]), 0.02 * 0.2 / self.D, places=14)
        self.assertEqual(float(pen.terms["cells"]), 0.0)
        with self.assertRaises(KeyError):
            L.kl_penalty(self.heads(), self.D, beta={"family": 0.01})
        with self.assertRaises(ValueError):
            L.kl_penalty({}, self.D)

    def test_head_aliases(self):
        # ppo.logprob keys (kl_*) and run-configuration keys (gate, cells) map to one canonical set
        logprob_keys = {"kl_family": (self.fam_kl, self.w), "kl_slot": (self.slot_kl, self.slot_w),
                        "kl_cell": (self.cell_kl, self.cell_w)}
        config_beta = {"gate": 0.01, "slot": 0.02, "cells": 0.03, "ability": 0.04}
        a = L.kl_penalty(logprob_keys, self.D, beta=config_beta)
        b = L.kl_penalty(self.heads(), self.D, beta={"family": 0.01, "slot": 0.02, "cells": 0.03})
        self.assertEqual(set(a.terms), {"family", "slot", "cells"})
        self.assertAlmostEqual(float(a.loss), float(b.loss), places=15)
        self.assertEqual(L.canonical_head("kl_ability"), "ability")
        self.assertEqual(L.canonical_head("kl_joint"), "kl_joint")          # not a head: passes through
        self.assertEqual(L.HEADS, ("family", "slot", "cells", "ability"))
        with self.assertRaises(ValueError):                                  # two keys, one head
            L.kl_penalty({"gate": (self.fam_kl, self.w), "family": (self.fam_kl, self.w)}, self.D)

    def test_global_denominator_too_small_is_nan(self):
        # a head's rows are a subset of the step's rows: D < sum(w_h) is a caller error
        pen = L.kl_penalty(self.heads(), 0.5 * self.D)
        self.assertTrue(math.isnan(float(pen.terms["family"])))
        self.assertFalse(math.isnan(float(pen.terms["slot"])))               # 1.0 <= D / 2
        self.assertTrue(math.isnan(float(L.kl_penalty(self.heads(), 0.0).loss)))

    def test_grad_norm_ratio(self):
        x = torch.tensor([0.3, -0.2], dtype=D64, requires_grad=True)
        n = 4
        lp = x[0] * t(1.0, 1.0, 1.0, 1.0)                 # surrogate depends on x0 only
        w = torch.ones(n, dtype=D64)
        adv = t(1.0, 1.0, 1.0, 1.0)
        surr = L.clipped_surrogate(lp, lp.detach(), adv, w)   # d/dx0 = -1 at r = 1
        kl_fam = (x[1] ** 2) * torch.ones(n, dtype=D64)       # d/dx1 = 2 x1 per row
        kl_slot = (x[0] * x[1]) * torch.ones(1, dtype=D64)
        pen = L.kl_penalty({"family": (kl_fam, w), "slot": (kl_slot, torch.ones(1, dtype=D64))},
                           w.sum(), beta=0.5)
        ratio = L.kl_grad_ratio(pen.terms, surr, [x])
        self.assertAlmostEqual(ratio["family"], 0.5 * 2 * 0.2 * n / n / 1.0, places=12)
        self.assertAlmostEqual(ratio["slot"], 0.5 * math.hypot(-0.2, 0.3) / n, places=12)
        # graphs kept: the real backward still works afterwards
        (surr + pen.loss).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(L.kl_grad_ratio({"z": torch.zeros((), dtype=D64)},
                                         torch.zeros((), dtype=D64), [x]), {"z": 0.0})


class ValueBCTest(unittest.TestCase):

    def test_value_mse_no_clip(self):
        v = t(0.9, -0.5, 0.0).requires_grad_(True)
        G = t(-1.0, -0.5, 1.0)
        w = t(1.0, 2.0, 0.5)
        loss = L.value_loss(v, G, w)
        hand = (1.0 * 1.9 ** 2 + 0.0 + 0.5 * 1.0) / 3.5
        self.assertAlmostEqual(float(loss), hand, places=14)       # quadratic in a 1.9 error: no Huber
        loss.backward()
        self.assertTrue(torch.allclose(v.grad, 2 * w * (v.detach() - G) / 3.5))

    def test_value_target_detached(self):
        v = t(0.2).requires_grad_(True)
        G = t(0.5).requires_grad_(True)
        L.value_loss(v, G, t(1.0)).backward()
        self.assertIsNone(G.grad)

    def test_value_global_denominator_checked(self):
        v, G, w = t(0.1, 0.2), t(0.0, 0.0), t(1.0, 1.0)
        self.assertAlmostEqual(float(L.value_loss(v, G, w, denom=4.0)), 0.05 / 4.0, places=15)
        self.assertTrue(math.isnan(float(L.value_loss(v, G, w, denom=1.0))))
        self.assertTrue(math.isnan(float(L.value_loss(v, G, w, denom=0.0))))
        self.assertEqual(float(L.value_loss(v, G, torch.zeros(2, dtype=D64), denom=0.0)), 0.0)

    def test_entropy_bonus_hand(self):
        ent = {"slot": (t(1.0, 0.5), t(1.0, 6.0)), "kl_cell": (t(2.0, 3.0, 9.0), t(1.0, 1.0, 0.0))}
        D = 10.0
        coef = {"gate": 0.0, "slot": 0.002, "cells": 1e-3, "ability": 0.0}
        x = ent["slot"][0].clone().requires_grad_(True)
        loss = L.entropy_bonus({"slot": (x, ent["slot"][1]), "kl_cell": ent["kl_cell"]}, D, coef=coef)
        hand = -(0.002 * (1.0 + 6.0 * 0.5) + 1e-3 * (2.0 + 3.0)) / D
        self.assertAlmostEqual(float(loss), hand, places=15)
        loss.backward()
        self.assertTrue(torch.allclose(x.grad, -0.002 * t(1.0, 6.0) / D))   # maximises entropy
        off = L.entropy_bonus(ent, D)                                          # default coef 0
        self.assertEqual(float(off), 0.0)
        self.assertFalse(off.requires_grad)
        with self.assertRaises(KeyError):
            L.entropy_bonus(ent, D, coef={"slot": 0.002})
        # same D convention as the KL: micro-batch shares add up
        e, w = t(0.3, 0.1, 0.7, 0.2), t(1.0, 2.0, 0.0, 6.0)
        full = float(L.entropy_bonus({"cells": (e, w)}, w.sum(), coef=1e-3))
        parts = sum(float(L.entropy_bonus({"cells": (e[s], w[s])}, w.sum(), coef=1e-3))
                    for s in (slice(0, 1), slice(1, 4)))
        self.assertAlmostEqual(parts, full, places=15)

    def test_bc_default_off_and_on(self):
        lp = t(-0.5, -2.0, -1.0).requires_grad_(True)
        w = t(1.0, 1.0, 6.0)
        off = L.bc_loss(lp, w, 100.0)
        self.assertEqual(float(off), 0.0)
        self.assertFalse(off.requires_grad)
        on = L.bc_loss(lp, w, 100.0, coef=0.05)                # D = PPO surrogate denominator
        self.assertAlmostEqual(float(on), 0.05 * (0.5 + 2.0 + 6.0) / 100.0, places=14)
        on.backward()
        self.assertTrue(torch.allclose(lp.grad, -0.05 * w / 100.0))


class MetricsTest(unittest.TestCase):

    def test_k3_nonnegative_and_zero_at_one(self):
        x = torch.linspace(-5, 5, 2001, dtype=D64)
        k = L.k3(x)
        self.assertTrue(bool((k >= 0).all()))
        self.assertEqual(float(L.k3(torch.zeros(3, dtype=D64)).abs().max()), 0.0)
        r = torch.exp(x)
        self.assertTrue(torch.allclose(k, r - 1 - x, atol=1e-12))
        small = L.k3(torch.tensor([1e-6], dtype=D64))
        self.assertAlmostEqual(float(small), 0.5e-12, delta=1e-17)    # ~ x^2 / 2 near r = 1
        self.assertTrue(bool((L.k3(torch.randn(1000) * 1e-4) >= 0).all()))  # fp32 rounding
        inf = float("inf")
        self.assertEqual(L.k3(t(inf, -inf)).tolist(), [inf, inf])           # infinite ratio: +inf, not NaN
        self.assertTrue(math.isnan(float(L.k3(t(float("nan"))))))

    def test_update_metrics_hand(self):
        lr = t(0.0, 0.1, -0.3, 1.5, 0.25)
        prox = t(-1.0, -2.0, -1.0, -3.0, -0.5)
        new = prox + lr
        w = t(6.0, 6.0, 1.0, 1.0, 0.0)
        play = torch.tensor([False, False, True, True, True])
        adv = t(1.0, -1.0, 1.0, -1.0, -1.0)
        gate = t(0.001, 0.003, 0.0, 0.01, 99.0)
        ent = {"slot": (t(1.0, 0.5), t(1.0, 1.0)), "family": (t(0.2, 0.3, 0.4, 0.5, 9.0), w)}
        m = L.update_metrics(new, prox, w, play, advantage=adv, gate_kl=gate, entropy=ent)
        r = torch.exp(lr)
        k = r - 1 - lr
        W = float(w.sum())
        self.assertAlmostEqual(m["joint_k3"], float((w * k).sum()) / W, places=12)
        self.assertAlmostEqual(m["play_k3"], (float(k[2]) + float(k[3])) / 2.0, places=12)
        self.assertAlmostEqual(m["clip_frac"], 2.0 / W, places=12)        # rows 2 and 3 clipped
        self.assertAlmostEqual(m["play_clip_frac"], 1.0, places=12)
        self.assertAlmostEqual(m["dual_clip_frac"], 1.0 / W, places=12)   # row 3: r = e^1.5 > c, A < 0
        self.assertAlmostEqual(m["gate_kl"], (6.0 * 0.004 + 0.01) / W, places=12)
        self.assertAlmostEqual(m["entropy/slot"], 0.75, places=12)
        self.assertAlmostEqual(m["entropy/family"], (6.0 * 0.5 + 0.4 + 0.5) / W, places=12)
        self.assertAlmostEqual(m["ratio_max"], math.exp(1.5), places=12)  # row 4 (w = 0) excluded
        self.assertAlmostEqual(m["ratio_min"], math.exp(-0.3), places=12)
        self.assertAlmostEqual(m["ratio_mean"], float((w * r).sum()) / W, places=12)
        for key in ("ratio_p01", "ratio_p99"):
            self.assertTrue(math.isfinite(m[key]))

    def test_metrics_at_identity(self):
        lp = torch.randn(40, dtype=D64)
        play = torch.arange(40) % 5 == 0
        m = L.update_metrics(lp, lp.clone(), torch.ones(40, dtype=D64), play,
                             gate_kl=torch.zeros(40, dtype=D64))
        self.assertEqual(m["joint_k3"], 0.0)
        self.assertEqual(m["play_k3"], 0.0)
        self.assertEqual(m["clip_frac"], 0.0)
        self.assertEqual(L.early_stop(m), ())

    def test_early_stop_thresholds(self):
        ok = {"joint_k3": 0.02, "play_k3": 0.1, "gate_kl": 0.01}        # equal: not tripped
        self.assertEqual(L.early_stop(ok), ())
        self.assertEqual(L.early_stop({**ok, "joint_k3": 0.0201}), ("joint_k3",))
        self.assertEqual(L.early_stop({**ok, "play_k3": 0.101}), ("play_k3",))
        self.assertEqual(L.early_stop({**ok, "gate_kl": 0.0101}), ("gate_kl",))
        self.assertEqual(L.early_stop({"joint_k3": 1.0, "play_k3": 1.0, "gate_kl": 1.0}),
                         ("joint_k3", "play_k3", "gate_kl"))
        self.assertEqual(L.early_stop({**ok, "joint_k3": float("nan")}), ("joint_k3",))
        # a missing metric raises by default (forgetting gate_kl must not drop the gate limit) ...
        with self.assertRaises(KeyError):
            L.early_stop({"joint_k3": 0.01, "play_k3": 0.0})
        # ... is skipped with strict=False, or dropped explicitly with a None threshold
        self.assertEqual(L.early_stop({"joint_k3": 0.01}, strict=False), ())
        self.assertEqual(L.early_stop({"joint_k3": 0.01, "play_k3": 0.0}, {"gate_kl": None}), ())
        # a partial override is merged: the other limits stay at their defaults
        self.assertEqual(L.early_stop({**ok, "joint_k3": 0.01}, {"joint_k3": 0.005}), ("joint_k3",))
        self.assertEqual(L.early_stop({"joint_k3": 1.0, "play_k3": 1.0, "gate_kl": 0.0025}, {"gate_kl": 0.003}),
                         ("joint_k3", "play_k3"))
        self.assertEqual(L.EARLY_STOP, {"joint_k3": 0.02, "play_k3": 0.1, "gate_kl": 0.01})

    def test_metrics_ignore_dead_rows_and_empty_heads(self):
        new, prox = t(-1.0, -2.0, -1.0), t(-1.05, -2.05, -1.0e30)    # row 2: LOG_ZERO, forced
        w = t(1.0, 2.0, 0.0)
        play = torch.tensor([False, True, True])
        m = L.update_metrics(new, prox, w, play, gate_kl=t(0.0, 0.001, float("nan")),
                             entropy={"kl_ability": (torch.zeros(0, dtype=D64), torch.zeros(0, dtype=D64)),
                                      "slot": (t(0.4), t(2.0))})
        clean = L.update_metrics(new[:2], prox[:2], w[:2], play[:2], gate_kl=t(0.0, 0.001))
        for k in ("joint_k3", "play_k3", "clip_frac", "ratio_mean", "gate_kl", "ratio_max"):
            self.assertAlmostEqual(m[k], clean[k], places=12, msg=k)
        self.assertEqual(L.early_stop(m), ())
        self.assertTrue(math.isnan(m["entropy/ability"]))               # no rows: NaN, not "collapsed"
        self.assertEqual(m["entropy_weight_sum/ability"], 0.0)
        self.assertAlmostEqual(m["entropy/slot"], 0.4, places=12)
        self.assertEqual(m["entropy_weight_sum/slot"], 2.0)


class IdentityTest(unittest.TestCase):

    def setUp(self):
        self.rt = torch.tensor([0, 0, 0, 1, 1, 2])
        self.lp = t(-0.01, -0.02, -0.07, -3.0, -4.0, -2.5)

    def test_pass(self):
        new = self.lp + t(5e-4, -5e-4, 0, 9e-4, 0, -9e-4)
        rep = L.assert_identity(new, self.lp, self.rt, {"family": torch.zeros(6), "slot": 3e-6})
        self.assertTrue(rep.ok)
        self.assertEqual(rep.rows, {"wait": 3, "play": 2, "ability": 1})
        self.assertLess(rep.max_abs_ratio_dev["play"], 1e-3)

    def test_fail_per_row_type(self):
        new = self.lp + t(0, 0, 0, 2e-3, 0, 0)            # one bad play row, waits clean
        rep = L.identity_check(new, self.lp, self.rt)
        self.assertEqual(rep.failures, ("ratio/play",))
        with self.assertRaises(AssertionError):
            L.assert_identity(new, self.lp, self.rt)
        # masking that row out passes
        mask = torch.tensor([True, True, True, False, True, True])
        self.assertTrue(L.identity_check(new, self.lp, self.rt, mask=mask).ok)

    def test_fail_kl_and_nan(self):
        rep = L.identity_check(self.lp, self.lp, self.rt, {"cells": t(0.0, 2e-5), "slot": 0.0})
        self.assertEqual(rep.failures, ("kl/cells",))
        bad = self.lp.clone()
        bad[0] = float("nan")
        self.assertIn("ratio/wait", L.identity_check(bad, self.lp, self.rt).failures)
        with self.assertRaises(ValueError):
            L.identity_check(self.lp, self.lp, torch.tensor([0, 0, 0, 1, 1, 7]))

    def test_empty_row_type(self):
        rep = L.identity_check(self.lp[:3], self.lp[:3], self.rt[:3])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.rows["ability"], 0)


class TinyModelTest(unittest.TestCase):
    """Real model outputs: the gate's executed log-prob from the tiny AminiFormer (common.py). The
    identity check here compares one forward with its detached copy (|r - 1| = 0 by construction);
    the two-forward dropout case is test_ppo_losses_adv.TinyModelIdentityTest. Skipped without the
    private dataset (``common.example``)."""

    @classmethod
    def setUpClass(cls):
        import common as C
        from amini.nn.aminiformer import hazard_probs
        cls.C, cls.hazard_probs = C, staticmethod(hazard_probs)
        ex, _ = C.example(0)
        cls.batch = C.to_batch([ex])

    def gate_logp(self, model):
        out = model(self.batch, with_aux=False)
        rv = self.batch["row"]["row_valid"].bool()
        avail = self.batch["row"]["ability_ready"][rv].bool().any(-1)
        dt = torch.full((out["family"].shape[0],), 10.0, dtype=D64)
        p = self.hazard_probs(out["family"], avail, dt)
        return out, torch.log(p.clamp_min(1e-300)), p

    def test_identity_surrogate_and_grads(self):
        model = self.C.tiny_model(seed=0).train()
        out, logp_all, p = self.gate_logp(model)
        n = logp_all.shape[0]
        fam = torch.arange(n) % 3
        fam = torch.where(p.gather(1, fam[:, None]).squeeze(1) > 0, fam, torch.zeros_like(fam))
        logp = logp_all.gather(1, fam[:, None]).squeeze(1)
        prox = logp.detach().clone()
        g = torch.Generator().manual_seed(3)
        adv = torch.randn(n, generator=g, dtype=D64)
        w = torch.where(fam == 0, torch.full((n,), 6.0, dtype=D64), torch.ones(n, dtype=D64))
        rt = fam.clone()                                   # 0 wait, 1 play, 2 ability
        rep = L.assert_identity(logp, prox, rt)
        self.assertEqual(rep.max_abs_ratio_dev["wait"], 0.0)
        surr = L.clipped_surrogate(logp, prox, adv, w)
        self.assertAlmostEqual(float(surr), float(-(w * adv).sum() / w.sum()), places=10)
        q = p.detach()
        fam_kl = (p * (torch.log(p.clamp_min(1e-300)) - torch.log(q.clamp_min(1e-300)))).sum(-1)
        pen = L.kl_penalty({"family": (fam_kl, w)}, w.sum())
        self.assertLess(abs(float(pen.loss)), 1e-12)
        vl = L.value_loss(out["value"], adv.clamp(-1, 1), w)
        params = [x for x in model.parameters() if x.requires_grad]
        ratio = L.kl_grad_ratio(pen.terms, surr, params)
        self.assertTrue(math.isfinite(ratio["family"]))
        self.assertLess(ratio["family"], 1e-6)              # KL at its minimum: ~zero gradient
        (surr + pen.loss + vl).backward()
        grads = [x.grad for x in params if x.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(bool(torch.isfinite(gr).all()) for gr in grads))
        m = L.update_metrics(logp, prox, w, fam == 1, advantage=adv, gate_kl=fam_kl.detach())
        self.assertEqual(L.early_stop(m), ())


if __name__ == "__main__":
    unittest.main()
