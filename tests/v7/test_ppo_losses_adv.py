"""Adversarial tests for src/amini/ppo/losses.py.

Three kinds of test:

* independent checks (must pass): a CleanRL-style reference surrogate (values and gradients),
  ``gradcheck`` away from the kinks, a direct fp32 overflow case that fails if ``LOG_RATIO_CLAMP`` is
  removed, fp32 vs fp64, scale / permutation / duplication invariance of the
  unified weight, k3 as the exact KL(prox || new) under behaviour weights, epoch aggregation of
  ``update_metrics`` through ``weight_sum``, micro-batch shares with heads absent from some micro-batches,
  metric / gradient consistency of the clip and dual-clip counters, and a real tiny AminiFormer whose
  joint log-prob (built here from ``sample_actions`` + ``hazard_probs``, not from ``ppo.logprob``) must
  make the update-0 identity check fail when dropout is on and pass when it is off (skipped without
  the private dataset);
* ``test_fixed_*``: each asserts a property the module docstring promises and an earlier version
  violated (they were ``test_BUG_*`` expected failures until the fix);
* ``test_DOC_*``: pin behaviour the learner relies on, so a change is deliberate (missing gate_kl
  raises, empty entropy heads read NaN with their weight sum, a global D below the call's weight
  sum gives NaN).

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_losses_adv.py -v
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
NAN, INF = float("nan"), float("inf")


def t(*xs) -> torch.Tensor:
    return torch.tensor(xs, dtype=D64)


def rows(n: int, seed: int, spread: float = 1.0):
    """Random per-row tensors: logp_new, logp_prox, advantage, unified weight (some forced), play mask."""
    g = torch.Generator().manual_seed(seed)
    prox = -torch.rand(n, generator=g, dtype=D64) * 12.0
    new = prox + (torch.rand(n, generator=g, dtype=D64) * 2 - 1) * spread
    adv = torch.randn(n, generator=g, dtype=D64)
    w = torch.where(torch.rand(n, generator=g, dtype=D64) < 0.7, torch.full((n,), 6.0, dtype=D64),
                    torch.ones(n, dtype=D64))
    w = w * torch.where(torch.rand(n, generator=g, dtype=D64) < 0.2, torch.tensor(0.25, dtype=D64),
                        torch.tensor(1.0, dtype=D64))                                # opponent seat
    w[torch.rand(n, generator=g, dtype=D64) < 0.15] = 0.0                            # forced rows
    play = torch.rand(n, generator=g) < 0.3
    return new, prox, adv, w, play


def reference_surrogate(new, prox, adv, w, eps=0.25, c=2.5):
    """CleanRL / dual-clip PPO written independently: max(-A r, -A clip(r)), then for A < 0 min(., -c A)."""
    r = torch.exp(new - prox)
    pg = torch.max(-adv * r, -adv * torch.clamp(r, 1 - eps, 1 + eps))
    pg = torch.where(adv < 0, torch.min(pg, -c * adv), pg)
    return (w * pg).sum() / w.sum()


# =========================================================================== independent checks
class ReferenceTest(unittest.TestCase):

    def test_reference_surrogate_values_and_grads(self):
        new, prox, adv, w, _ = rows(5000, 11, spread=4.0)      # ratios e^-4 .. e^4: every branch
        a = new.clone().requires_grad_(True)
        b = new.clone().requires_grad_(True)
        got = L.clipped_surrogate(a, prox, adv, w)
        want = reference_surrogate(b, prox, adv, w)
        got.backward()
        want.backward()
        self.assertAlmostEqual(float(got), float(want), places=12)
        self.assertTrue(torch.allclose(a.grad, b.grad, atol=1e-15, rtol=1e-12))
        r = torch.exp(new - prox)
        for name, sel in (("dual", (adv < 0) & (r > 2.5)), ("upper", (adv > 0) & (r > 1.25)),
                          ("lower", (adv < 0) & (r < 0.75)), ("mid_neg", (adv < 0) & (r > 1.25) & (r < 2.5))):
            self.assertGreater(int((sel & (w > 0)).sum()), 50, name)   # the sample covers each branch

    def test_gradcheck_away_from_kinks(self):
        g = torch.Generator().manual_seed(12)
        lr = (torch.rand(400, generator=g, dtype=D64) * 2.4 - 1.2)
        kinks = torch.tensor([math.log(0.75), math.log(1.25), math.log(2.5)], dtype=D64)
        lr = lr[(lr[:, None] - kinks).abs().min(-1).values > 1e-3][:200]
        prox = -torch.rand(lr.shape[0], generator=g, dtype=D64) * 5
        adv = torch.randn(lr.shape[0], generator=g, dtype=D64)
        w = torch.rand(lr.shape[0], generator=g, dtype=D64) * 6.0
        new = (prox + lr).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: L.clipped_surrogate(x, prox, adv, w), (new,)))
        kl = torch.rand(lr.shape[0], generator=g, dtype=D64).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda k: L.kl_penalty({"family": (k, w)}, w.sum() * 1.3).loss, (kl,)))
        v = (torch.rand(lr.shape[0], generator=g, dtype=D64) * 2 - 1).requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(lambda x: L.value_loss(x, adv.clamp(-1, 1), w), (v,)))

    def test_fp32_and_mixed_precision_match_fp64(self):
        new, prox, adv, w, play = rows(4000, 13, spread=1.5)
        ref = float(L.clipped_surrogate(new, prox, adv, w))
        f32 = L.clipped_surrogate(new.float(), prox.float(), adv.float(), w.float())
        mixed = L.clipped_surrogate(new.float(), prox, adv, w)
        self.assertEqual(f32.dtype, torch.float32)
        self.assertLess(abs(float(f32) - ref), 1e-5 * max(1.0, abs(ref)))
        self.assertLess(abs(float(mixed) - ref), 1e-5 * max(1.0, abs(ref)))
        m64 = L.update_metrics(new, prox, w, play, advantage=adv)
        m32 = L.update_metrics(new.float(), prox.float(), w.float(), play, advantage=adv.float())
        for k in ("joint_k3", "play_k3", "ratio_mean"):
            self.assertLess(abs(m64[k] - m32[k]), 1e-5 * max(1.0, abs(m64[k])), k)
        # bf16 inputs are accepted and upcast (the caller must not feed bf16 log-probs: ~1e-3 error)
        self.assertEqual(L.clipped_surrogate(new.bfloat16(), prox.bfloat16(), adv, w).dtype, torch.float32)

    def test_clamp_guards_fp32_overflow(self):
        # test_ppo_losses.test_autograd_finite_at_extremes still passes with the clamp disabled (one row
        # past |log r| = 88, not on the NaN branch). This case does not: fp32, log r = 100, A > 0.
        def grads():
            x = torch.tensor([100.0, 100.0, -100.0, -100.0], requires_grad=True)
            L.clipped_surrogate(x, torch.zeros(4), torch.tensor([1.0, -1.0, 1.0, -1.0]), torch.ones(4)).backward()
            return x.grad
        self.assertTrue(bool(torch.isfinite(grads()).all()))
        saved = L.LOG_RATIO_CLAMP
        try:
            L.LOG_RATIO_CLAMP = float("inf")                 # the test has teeth: no clamp -> NaN
            self.assertFalse(bool(torch.isfinite(grads()).all()))
        finally:
            L.LOG_RATIO_CLAMP = saved

    def test_k3_mean_is_exact_kl_prox_to_new_under_behaviour_weights(self):
        # Rows = outcomes of one categorical, weight q(x): sum_x q k3(log p/q) = sum p - 1 + KL(q||p).
        g = torch.Generator().manual_seed(14)
        lq = torch.log_softmax(torch.randn(7, generator=g, dtype=D64), -1)
        lp = torch.log_softmax(lq + torch.randn(7, generator=g, dtype=D64) * 0.3, -1)
        m = L.update_metrics(lp, lq, lq.exp(), torch.ones(7, dtype=torch.bool))
        kl_qp = float((lq.exp() * (lq - lp)).sum())
        self.assertAlmostEqual(m["joint_k3"], kl_qp, places=12)
        self.assertAlmostEqual(m["play_k3"], kl_qp, places=12)
        self.assertAlmostEqual(m["ratio_mean"], 1.0, places=12)          # E_q[p/q] = 1


class WeightSemanticsTest(unittest.TestCase):
    """The unified weight is a Horvitz-Thompson weight: scaling all weights does nothing (local D),
    row order does nothing, and a row of weight 2w is the same as two copies of weight w."""

    def setUp(self):
        self.new, self.prox, self.adv, self.w, self.play = rows(300, 21, spread=1.0)
        g = torch.Generator().manual_seed(22)
        self.kl = torch.rand(300, generator=g, dtype=D64) * 0.1
        self.v = torch.rand(300, generator=g, dtype=D64) * 2 - 1
        self.gate = torch.rand(300, generator=g, dtype=D64) * 0.01

    def all_terms(self, new, prox, adv, w, play, kl, v, gate):
        m = L.update_metrics(new, prox, w, play, advantage=adv, gate_kl=gate,
                             entropy={"family": (kl, w)})
        keys = ("joint_k3", "play_k3", "clip_frac", "play_clip_frac", "dual_clip_frac", "ratio_mean",
                "gate_kl", "entropy/family")
        return ([float(L.clipped_surrogate(new, prox, adv, w)),
                 float(L.kl_penalty({"family": (kl, w)}, L.surrogate_denominator(w)).loss),
                 float(L.value_loss(v, adv.clamp(-1, 1), w))] + [m[k] for k in keys])

    def assertTermsEqual(self, a, b):
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)

    def test_scale_invariance(self):
        base = self.all_terms(self.new, self.prox, self.adv, self.w, self.play, self.kl, self.v, self.gate)
        scaled = self.all_terms(self.new, self.prox, self.adv, self.w * 3.7, self.play, self.kl, self.v,
                                self.gate)
        self.assertTermsEqual(base, scaled)

    def test_permutation_invariance(self):
        perm = torch.randperm(300, generator=torch.Generator().manual_seed(23))
        base = self.all_terms(self.new, self.prox, self.adv, self.w, self.play, self.kl, self.v, self.gate)
        shuf = self.all_terms(*(x[perm] for x in (self.new, self.prox, self.adv, self.w, self.play,
                                                   self.kl, self.v, self.gate)))
        self.assertTermsEqual(base, shuf)

    def test_duplicate_row_equals_double_weight(self):
        k = int((self.w > 0).nonzero()[3])
        w2 = self.w.clone()
        w2[k] *= 2
        doubled = self.all_terms(self.new, self.prox, self.adv, w2, self.play, self.kl, self.v, self.gate)
        dup = self.all_terms(*(torch.cat([x, x[k:k + 1]]) for x in (self.new, self.prox, self.adv, self.w,
                                                                     self.play, self.kl, self.v,
                                                                     self.gate)))
        self.assertTermsEqual(doubled, dup)


class AggregationTest(unittest.TestCase):

    def test_epoch_aggregation_through_weight_sums(self):
        new, prox, adv, w, play = rows(600, 31, spread=0.8)
        gate = torch.rand(600, generator=torch.Generator().manual_seed(32), dtype=D64) * 0.01
        full = L.update_metrics(new, prox, w, play, advantage=adv, gate_kl=gate)
        parts = [slice(0, 7), slice(7, 250), slice(250, 251), slice(251, 600)]
        ms = [L.update_metrics(new[s], prox[s], w[s], play[s], advantage=adv[s], gate_kl=gate[s])
              for s in parts]
        for key, wk in (("joint_k3", "weight_sum"), ("clip_frac", "weight_sum"),
                        ("dual_clip_frac", "weight_sum"), ("ratio_mean", "weight_sum"),
                        ("gate_kl", "weight_sum"), ("play_k3", "play_weight_sum"),
                        ("play_clip_frac", "play_weight_sum")):
            agg = sum(m[key] * m[wk] for m in ms) / sum(m[wk] for m in ms)
            self.assertAlmostEqual(agg, full[key], places=12, msg=key)

    def test_global_denominator_with_heads_absent_from_micro_batches(self):
        new, prox, adv, w, play = rows(120, 33, spread=0.6)
        g = torch.Generator().manual_seed(34)
        fam_kl = torch.rand(120, generator=g, dtype=D64) * 0.01
        ab_rows = torch.zeros(120, dtype=torch.bool)
        ab_rows[[50, 51, 52]] = True                          # the ability head lives in micro-batch 2 only
        ab_kl = torch.rand(120, generator=g, dtype=D64)
        hum = -torch.rand(120, generator=g, dtype=D64) * 4
        hw = (torch.rand(120, generator=g) < 0.1).to(D64)

        def total(x, s, D):
            sel = ab_rows[s]
            pen = L.kl_penalty({"family": (fam_kl[s] * x[s].exp(), w[s]),
                                "ability": (ab_kl[s][sel] * x[s][sel].exp(), w[s][sel])}, D,
                               beta={"family": 0.01, "ability": 0.02})
            return (L.clipped_surrogate(x[s], prox[s], adv[s], w[s], denom=D) + pen.loss
                    + L.bc_loss(x[s] + hum[s], hw[s], D, coef=0.05))

        D = L.surrogate_denominator(w)
        xf = new.clone().requires_grad_(True)
        full = total(xf, slice(0, 120), D)
        full.backward()
        xm = new.clone().requires_grad_(True)
        acc = 0.0
        for s in (slice(0, 40), slice(40, 80), slice(80, 120)):
            part = total(xm, s, D)
            part.backward()
            acc += float(part)
        self.assertAlmostEqual(acc, float(full), places=12)
        self.assertTrue(torch.allclose(xm.grad, xf.grad, atol=1e-15))


class ConsistencyTest(unittest.TestCase):

    def test_clip_counters_match_zero_gradient_rows(self):
        new, prox, adv, w, play = rows(3000, 41, spread=3.0)
        w = torch.where(w > 0, w, torch.ones_like(w))         # every row live, so the gradient is per row
        x = new.clone().requires_grad_(True)
        L.clipped_surrogate(x, prox, adv, w).backward()
        r = torch.exp(new - prox)
        zero = x.grad == 0
        dual = (adv < 0) & (r > 2.5)
        self.assertTrue(bool(zero[dual].all()))                # the dual clip kills the gradient ...
        m = L.update_metrics(new, prox, w, play, advantage=adv)
        self.assertAlmostEqual(m["dual_clip_frac"], float((w * dual).sum() / w.sum()), places=12)
        clipped = (r - 1).abs() > 0.25
        self.assertTrue(bool(clipped[zero].all()))             # ... and every zero-gradient row is counted
        self.assertGreaterEqual(m["clip_frac"], float((w * zero).sum() / w.sum()))


class TinyModelIdentityTest(unittest.TestCase):
    """test_ppo_losses.TinyModelTest compares a forward with a detached copy of itself (|r - 1| = 0 by
    construction). Here two separate train-mode forwards are compared, with the joint executed log-prob
    built independently from sample_actions + hazard_probs: dropout 0.1 must fail the identity check,
    dropout 0 must pass it (update-0 identity). Skipped without the private dataset (``common.example``)."""

    @classmethod
    def setUpClass(cls):
        import common as C
        from amini.nn.aminiformer import MASK_FILL, hazard_probs, sample_actions
        cls.C, cls.MASK_FILL = C, MASK_FILL
        cls.hazard_probs, cls.sample_actions = staticmethod(hazard_probs), staticmethod(sample_actions)
        ex, _ = C.example(0)
        cls.batch = C.to_batch([ex])

    def outputs(self, model):
        rv = self.batch["row"]["row_valid"].bool()
        out = model(self.batch, cell_rows=rv, all_slots=True, with_aux=False)
        return {"family": out["family"], "slot": out["slot"], "ability": out["ability"],
                "ability_avail": out["ability_avail"], "cells": out["cells_all"]}

    def joint_logp(self, out, act, dt):
        p = self.hazard_probs(out["family"], out["ability_avail"], dt)
        legal_any = out["slot"].max(-1).values > self.MASK_FILL / 2
        n = p.shape[0]
        ar = torch.arange(n)
        fam = act["family"]
        lwait = torch.log(p[:, 0] + p[:, 1] * (~legal_any).to(p.dtype))
        lslot = torch.log_softmax(out["slot"].double(), -1)[ar, act["slot"]]
        lcell = torch.log_softmax(out["cells"].double()[ar, act["slot"]], -1)[ar, act["cell"]]
        lab = torch.log_softmax(out["ability"].double(), -1)[ar, act["ability"]]
        lp_play = torch.log(p[:, 1].clamp_min(1e-300)) + lslot + lcell
        lp_ab = torch.log(p[:, 2].clamp_min(1e-300)) + lab
        return torch.where(fam == 0, lwait, torch.where(fam == 1, lp_play, lp_ab)), fam

    def run_pair(self, dropout):
        model = self.C.tiny_model(seed=0, dropout=dropout).train()
        with torch.no_grad():
            o1 = self.outputs(model)
        n = o1["family"].shape[0]
        dt = torch.full((n,), 10.0, dtype=D64)
        act = self.sample_actions(o1, dt, generator=torch.Generator().manual_seed(5))
        # force a mix of row types: every third row plays / uses an ability when that is possible
        fam = act["family"].clone()
        legal_any = o1["slot"].max(-1).values > self.MASK_FILL / 2
        fam = torch.where((torch.arange(n) % 3 == 1) & legal_any, torch.ones_like(fam), fam)
        fam = torch.where((torch.arange(n) % 3 == 2) & o1["ability_avail"].bool(), torch.full_like(fam, 2), fam)
        act = dict(act, family=fam)
        prox, rt = self.joint_logp(o1, act, dt)
        o2 = self.outputs(model)
        new, _ = self.joint_logp(o2, act, dt)
        return new, prox, rt

    def test_identity_fails_with_dropout_and_passes_without(self):
        new, prox, rt = self.run_pair(0.0)
        self.assertGreater(int((rt == 1).sum()), 0)
        rep = L.assert_identity(new, prox.detach(), rt)
        self.assertTrue(rep.ok)
        w = torch.where(rt == 0, torch.full_like(new, 6.0), torch.ones_like(new))
        adv = torch.randn(new.shape[0], generator=torch.Generator().manual_seed(6), dtype=D64)
        self.assertAlmostEqual(float(L.clipped_surrogate(new, prox, adv, w)),
                               float(-(w * adv).sum() / w.sum()), places=10)
        new, prox, rt = self.run_pair(0.1)
        rep = L.identity_check(new.detach(), prox, rt)
        self.assertFalse(rep.ok, rep)
        self.assertIn("ratio/play", rep.failures)
        m = L.update_metrics(new.detach(), prox, torch.ones_like(new), rt == 1)
        self.assertGreater(m["joint_k3"], 0.0)


# =========================================================================== fixed issues
class ZeroWeightRowsTest(unittest.TestCase):
    """Forced rows (w = 0) must be excluded even when their values are non-finite: 0 * NaN = 0 * inf =
    NaN, so the module sanitises dead-row inputs before any arithmetic (forward and backward)."""

    def setUp(self):
        self.new, self.prox = t(-1.0, -2.0), t(-1.1, -2.2)
        self.adv, self.w = t(1.0, -1.0), t(1.0, 2.0)
        self.play = torch.tensor([False, True])

    def with_row(self, base, extra):
        return torch.cat([base, extra.reshape(1).to(base.dtype)])

    def test_fixed_surrogate_ignores_nonfinite_zero_weight_row(self):
        clean = float(L.clipped_surrogate(self.new, self.prox, self.adv, self.w))
        for new_x, prox_x, adv_x in ((NAN, -1.0, 1.0), (-INF, -INF, 1.0), (-1.0, NAN, 1.0), (-1.0, -1.0, NAN)):
            x = self.with_row(self.new, t(new_x)).requires_grad_(True)
            loss = L.clipped_surrogate(x, self.with_row(self.prox, t(prox_x)), self.with_row(self.adv, t(adv_x)),
                                       self.with_row(self.w, t(0.0)))
            loss.backward()
            self.assertAlmostEqual(float(loss), clean, places=12, msg=(new_x, prox_x, adv_x))
            self.assertTrue(bool(torch.isfinite(x.grad).all()), (new_x, prox_x, adv_x, x.grad))

    def test_fixed_metrics_ignore_nonfinite_zero_weight_row(self):
        clean = L.update_metrics(self.new, self.prox, self.w, self.play, gate_kl=t(0.0, 0.0))
        for new_x, prox_x in ((-1.0, -INF), (-1.0, -1.0e30), (NAN, -1.0)):   # -1e30 = ppo.logprob LOG_ZERO
            m = L.update_metrics(self.with_row(self.new, t(new_x)), self.with_row(self.prox, t(prox_x)),
                                 self.with_row(self.w, t(0.0)), torch.tensor([False, True, False]),
                                 gate_kl=t(0.0, 0.0, NAN))
            for k in ("joint_k3", "play_k3", "clip_frac", "ratio_mean", "gate_kl"):
                self.assertAlmostEqual(m[k], clean[k], places=12, msg=(k, new_x, prox_x))
            # compare with the verdict on the clean rows alone (they may trip a limit by themselves)
            self.assertEqual(L.early_stop(m), L.early_stop(clean))

    def test_fixed_kl_and_value_ignore_nonfinite_zero_weight_row(self):
        kl = L.kl_penalty({"family": (t(0.1, NAN), t(1.0, 0.0))}, 1.0)
        self.assertAlmostEqual(float(kl.loss), 0.02 * 0.1, places=14)
        v = t(0.1, 0.2).requires_grad_(True)
        vl = L.value_loss(v, t(1.0, NAN), t(1.0, 0.0))
        vl.backward()
        self.assertAlmostEqual(float(vl), 0.81, places=12)
        self.assertTrue(bool(torch.isfinite(v.grad).all()), v.grad)

    def test_fixed_k3_of_infinite_ratio_is_nonnegative(self):
        # docstring: 'Non-negative'; expm1(inf) - inf = NaN (an infinite ratio should read +inf)
        self.assertEqual(float(L.k3(t(INF))), INF)


class ThresholdAndClampTest(unittest.TestCase):

    def test_fixed_partial_threshold_override_keeps_other_limits(self):
        # Calibrating one limit must not switch the others off.
        m = {"joint_k3": 1.0, "play_k3": 1.0, "gate_kl": 0.0001}
        self.assertEqual(L.early_stop(m, {"gate_kl": 0.003}), ("joint_k3", "play_k3"))

    def test_fixed_log_ratio_clamp_without_dual_clip(self):
        # clipped_surrogate docstring: 'Every clamped row is already clipped (... r > c with A < 0 ...)'.
        # With dual_clip=None a row with A < 0 and log r = 25 is unclipped: objective r A, gradient r A.
        x = t(25.0).requires_grad_(True)
        loss = L.clipped_surrogate(x, t(0.0), t(-1.0), t(1.0), dual_clip=None)
        loss.backward()
        self.assertAlmostEqual(float(loss) / math.exp(25.0), 1.0, places=9)
        self.assertAlmostEqual(float(x.grad[0]) / math.exp(25.0), 1.0, places=9)


# =========================================================================== pinned risky behaviour
class PinnedBehaviourTest(unittest.TestCase):

    def test_DOC_missing_gate_kl_raises(self):
        # update_metrics(gate_kl=None) omits the key; early_stop then raises (strict default) instead of
        # silently dropping the gate limit (the ratio clip cannot bound the gate).
        m = L.update_metrics(t(0.0, 0.0), t(0.0, 0.0), t(1.0, 1.0), torch.tensor([False, True]))
        self.assertNotIn("gate_kl", m)
        with self.assertRaises(KeyError):
            L.early_stop(m)
        self.assertEqual(L.early_stop(m, strict=False), ())

    def test_DOC_empty_entropy_head_reads_nan_with_zero_weight(self):
        # A head with no rows in the minibatch reports NaN (not 0.0, which reads as collapsed) and its
        # weight sum, so epoch aggregation can skip it.
        empty = torch.zeros(0, dtype=D64)
        m = L.update_metrics(t(0.0), t(0.0), t(1.0), torch.tensor([False]), entropy={"ability": (empty, empty)})
        self.assertTrue(math.isnan(m["entropy/ability"]))
        self.assertEqual(m["entropy_weight_sum/ability"], 0.0)

    def test_DOC_global_denominator_is_checked(self):
        # denom = 0 with live rows, or denom < sum(w), is a caller error: the loss reads NaN so the
        # learner's NaN guard skips the step (no host sync inside the loss).
        x = t(0.0, 0.0).requires_grad_(True)
        loss = L.clipped_surrogate(x, t(0.0, 0.0), t(1.0, 1.0), t(1.0, 1.0), denom=0.0)
        loss.backward()
        self.assertTrue(math.isnan(float(loss)))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertTrue(math.isnan(float(L.clipped_surrogate(t(0.0, 0.0), t(0.0, 0.0), t(1.0, 1.0), t(1.0, 1.0),
                                                             denom=0.5))))


if __name__ == "__main__":
    unittest.main()
