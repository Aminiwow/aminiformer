"""Adversarial tests for amini.ppo.logprob.

What test_ppo_logprob.py does not cover:
* the LEARNER path: ``AminiFormer.forward(cell_rows=..., play_slot=...)`` against the actor's ``peek``
  through the one shared ``executed_logp`` (fp64 and fp32, T = 1 and T = 0.5), plus gradients;
* exact per-cell / per-pointer frequencies of ``sample_actions`` (test_ppo_logprob.py bins cells by mass);
* unbiasedness of the sampled-branch KL / entropy estimator (executed-branch row masks) by enumeration;
* autograd ``gradcheck`` of every log-prob term, empty batches, stored-action conventions, dt forms;
* five defects of an earlier version, now fixed: raw forward cells + ``cell_play_index`` scatter, the
  temperature floor / legality guard, the ``valid`` flag for impossible actions, finite gradients at
  subnormal rates, and the gather-first actor path.

Tests on real model outputs are skipped without the private dataset (``common.example``).
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

HEADS = ("family", "slot", "ability", "ability_avail", "cells")


# =========================================================================== fixtures
def _slice_t(v, t):
    if isinstance(v, dict):
        return {k: _slice_t(x, t) for k, x in v.items()}
    return v[0:1, t]


@lru_cache(maxsize=4)
def actor_and_batch(seed: int = 0, ex: int = 2, double: bool = True):
    """Actor side: ``peek`` + commit over every row of one example (all rows committed, as the learner's
    committed-row forward assumes).  Returns (model, batch, n, actor_out)."""

    model = tiny_model(seed, double=double, cell_mask=True)
    e, _ = example(ex)
    b = to_batch([e], double=double)
    r = b["row"]
    n = int(r["row_valid"][0].sum())
    state = model.initial_state(1, dtype=torch.float64 if double else torch.float32)
    outs = []
    for t in range(n):
        tok = {"groups": _slice_t(b["groups"], t),
               "row": {k: r[k][0:1, t] for k in ("slot_legal", "ability_ready", "cell_mask_id")}}
        outs.append(model.peek(tok, r["pos"][0:1, t], state))
        model.commit(state, torch.tensor([True]))
    return model, b, n, {k: torch.cat([o[k] for o in outs]) for k in HEADS}


def unit_rate_dt(out):
    lf = out["family"].double()
    odds = torch.exp(lf[:, 1] - lf[:, 0]) + torch.where(out["ability_avail"].bool(),
                                                        torch.exp(lf[:, 2] - lf[:, 0]),
                                                        torch.zeros(lf.shape[0], dtype=torch.float64))
    return S.U_TICKS / odds


def learner_out(model, b, n, action, *, grad: bool = False):
    """Learner side: one standard forward over the committed rows with teacher-forced play slots, cells of
    the executed slot scattered back to ``[n, 576]`` (what the learner builds from ``cell_play_index``)."""

    rv = b["row"]["row_valid"].bool()
    play = action["family"] == S.FAMILY_PLAY
    cell_rows = torch.zeros_like(rv)
    play_slot = torch.full(rv.shape, -1, dtype=torch.long)
    cell_rows[0, :n] = play
    play_slot[0, :n] = torch.where(play, action["slot"], torch.full_like(action["slot"], -1))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        raw = model(b, cell_rows=cell_rows, play_slot=play_slot, with_aux=False)
        cells = torch.full((n, S.N_CELLS), MASK_FILL, dtype=raw["family"].dtype)
        if raw["cell_play_index"].numel():
            cells = cells.index_put((raw["cell_play_index"],), raw["cells"])
    out = {k: raw[k] for k in ("family", "slot", "ability", "ability_avail")}
    out["cells"] = cells
    return out, raw


def synth(n_cells: int = S.N_CELLS, dtype=torch.float64, seed: int = 7):
    """Edge rows with few legal choices so per-action frequencies are testable exactly.

    0 single legal slot (2), 6 legal cells, no ability      1 blocked play, 2 ready abilities
    2 two legal slots (0, 3) x 5 legal cells, 3 ready         3 forced (no slot, no ability)"""

    g = torch.Generator().manual_seed(seed)
    n = 4
    fam = torch.tensor([[0.0, 0.4, MASK_FILL], [0.0, 0.9, 0.2], [0.0, -0.3, 0.5], [0.0, 1.0, MASK_FILL]],
                       dtype=torch.float64)
    avail = torch.tensor([False, True, True, False])
    slot_legal = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]]).bool()
    ready = torch.zeros(n, S.ABILITY_POINTER, dtype=torch.bool)
    ready[1, [4, 9]] = True
    ready[2, [0, 2, 63]] = True
    slot = (1.2 * torch.randn(n, 4, generator=g, dtype=torch.float64)).masked_fill(~slot_legal, MASK_FILL)
    ab = torch.randn(n, S.ABILITY_POINTER, generator=g, dtype=torch.float64).masked_fill(~ready, MASK_FILL)
    cells = 1.5 * torch.randn(n, 4, n_cells, generator=g, dtype=torch.float64)
    cm = torch.zeros(n, 4, n_cells, dtype=torch.bool)
    cm[0, 2, [1, 3, 5, 8, 11, n_cells - 1]] = True
    cm[2, 0, [0, 2, 4, 6, 7]] = True
    cm[2, 3, [1, 9, 10, 12, 13]] = True
    cm[1] = True
    cells = cells.masked_fill(~cm, MASK_FILL)
    out = {"family": fam, "slot": slot, "ability": ab, "ability_avail": avail, "cells": cells}
    return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in out.items()}


def enum_exact(out, dt):
    """{(row, family, slot, cell, ability): p} over every executable action, from executed_logp."""

    m = LP.legal_masks(out)
    res = {}
    n = out["family"].shape[0]
    for i in range(n):
        acts = [(0, 0, 0, 0)]
        for s in m["slot_legal"][i].nonzero(as_tuple=True)[0].tolist():
            for c in (out["cells"][i, s] > MASK_FILL / 2).nonzero(as_tuple=True)[0].tolist():
                acts.append((1, s, c, 0))
        if bool(m["ability_avail"][i]):
            for k in m["ability_ready"][i].nonzero(as_tuple=True)[0].tolist():
                acts.append((2, 0, 0, k))
        a = len(acts)
        rep = {k: out[k][i:i + 1].expand(a, *out[k].shape[1:]) for k in HEADS}
        t = torch.tensor(acts)
        act = {"family": t[:, 0], "slot": t[:, 1], "cell": t[:, 2], "ability": t[:, 3]}
        p = LP.executed_logp(rep, dt[i], act).exp()
        for j, key in enumerate(acts):
            res[(i,) + key] = float(p[j])
    return res


# =========================================================================== learner path
class TestLearnerPath(unittest.TestCase):
    """One shared function: actor (peek) and learner (forward) give the same log pi."""

    def _check(self, double: bool, tol: float):
        model, b, n, actor = actor_and_batch(0, 2, double)
        dt = unit_rate_dt(actor)
        a = sample_actions(actor, dt, generator=torch.Generator().manual_seed(21))
        fam = a["family"]
        self.assertTrue(bool((fam == 1).any()) and bool((fam == 2).any()) and bool((fam == 0).any()))
        learner, raw = learner_out(model, b, n, a)
        worst = 0.0
        for t in (1.0, 0.5):
            la = LP.executed_logp(LP.apply_temperature(actor, t, t, t), dt, a)
            ll = LP.executed_logp(LP.apply_temperature(learner, t, t, t), dt, a)
            worst = max(worst, float((la - ll).abs().max()))
            # update-0 identity: every head's KL(learner || actor) vanishes (mixed peek / learner forms)
            kl = LP.head_kl(LP.apply_temperature(learner, t, t, t), LP.apply_temperature(actor, t, t, t), dt, a)
            for k in ("kl_family", "kl_slot", "kl_cell", "kl_ability"):
                self.assertLess(float(kl[k].abs().max()), max(tol, 1e-12) * 10, (k, t))
        self.assertLess(worst, tol)
        print(f"actor peek vs learner forward ({'fp64' if double else 'fp32'}): max |dlogp| {worst:.2e} "
              f"over {n} rows ({int((fam == 1).sum())} plays, {int((fam == 2).sum())} abilities)")

    def test_forward_equals_peek_fp64(self):
        self._check(True, 1e-9)

    def test_forward_equals_peek_fp32(self):
        # fp32 actor and learner (no autocast): well inside the update-0 identity tolerance
        self._check(False, 1e-4)

    def test_raw_forward_cells_are_scattered(self):
        """forward(play_slot=...) returns cells only for the play rows ([n_play, 576] + cell_play_index).
        Passed straight in, they are scattered to row-aligned [n, 576] (bitwise the manual scatter); without
        the index they are rejected with ValueError, never misaligned."""

        model, b, n, actor = actor_and_batch(0, 2, True)
        dt = unit_rate_dt(actor)
        a = sample_actions(actor, dt, generator=torch.Generator().manual_seed(21))
        learner, raw = learner_out(model, b, n, a)
        self.assertLess(raw["cells"].shape[0], n)
        self.assertTrue(torch.equal(LP.row_cells(raw), learner["cells"]))
        self.assertTrue(torch.equal(LP.executed_logp(raw, dt, a), LP.executed_logp(learner, dt, a)))
        kl = LP.head_kl(raw, actor, dt, a)
        for k in ("kl_family", "kl_slot", "kl_cell", "kl_ability"):
            self.assertLess(float(kl[k].abs().max()), 1e-9, k)
        with self.assertRaises(ValueError):
            LP.executed_logp({k: v for k, v in raw.items() if k != "cell_play_index"}, dt, a)
        # no play rows at all: empty index, no cells -> every row MASK_FILL
        empty = dict(raw, cell_play_index=raw["cell_play_index"][:0])
        del empty["cells"]
        rc = LP.row_cells(empty)
        self.assertEqual(tuple(rc.shape), (n, S.N_CELLS))
        self.assertTrue(bool((rc == MASK_FILL).all()))

    def test_learner_gradients(self):
        model, b, n, actor = actor_and_batch(0, 2, True)
        dt = unit_rate_dt(actor)
        a = sample_actions(actor, dt, generator=torch.Generator().manual_seed(22))
        model.zero_grad(set_to_none=True)
        learner, _ = learner_out(model, b, n, a, grad=True)
        ent = LP.head_entropy(learner, dt, a)
        loss = LP.executed_logp(learner, dt, a).sum() + (ent["ent_cell"] * ent["mask_cell"]).sum()
        loss.backward()
        grads = {nm: p.grad for nm, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads.values()))
        m = LP.legal_masks(learner)
        need = {"gate.": True, "cells.": bool((a["family"] == 1).any()),
                "slot_head.": bool(((a["family"] == 1) & (m["n_slots"] >= 2)).any()),
                "ability_head.": bool(((a["family"] == 2) & (m["n_ready"] >= 2)).any())}
        self.assertTrue(need["cells."])
        for pre, on in need.items():
            if on:
                self.assertTrue(any(nm.startswith(pre) and float(g.abs().sum()) > 0
                                    for nm, g in grads.items()), pre)
        model.zero_grad(set_to_none=True)


# =========================================================================== exact frequencies
class TestExactFrequencies(unittest.TestCase):
    def test_per_cell_and_pointer_frequencies(self):
        """Every executable (family, slot, cell, ability) of few-choice rows vs 1e5 sample_actions draws,
        fp32 sampler input (the actor's dtype), T in {1, 0.25, 2}; blocked plays fold into wait."""

        n_draw, chunk = 100_000, 10_000
        gen = torch.Generator().manual_seed(77)
        zs, viol, groups = [], 0, 0
        for t in (1.0, 0.25, 2.0):
            out = LP.apply_temperature(synth(dtype=torch.float32), t, t, t)
            n = out["family"].shape[0]
            dt = torch.tensor([10.0, 7.0, 13.0, 10.0], dtype=torch.float64)
            exact = enum_exact(out, dt)
            self.assertAlmostEqual(sum(exact.values()), float(n), places=12)
            counts = {}
            for _ in range(n_draw // chunk):
                idx = torch.arange(n).repeat_interleave(chunk)
                a = sample_actions({k: v[idx] for k, v in out.items()}, dt[idx], generator=gen)
                f = a["family"]
                key = torch.stack([idx, f, torch.where(f == 1, a["slot"], 0), torch.where(f == 1, a["cell"], 0),
                                   torch.where(f == 2, a["ability"], 0)], -1)
                u, c = torch.unique(key, dim=0, return_counts=True)
                for kk, cc in zip(u.tolist(), c.tolist()):
                    counts[tuple(kk)] = counts.get(tuple(kk), 0) + cc
            for kk, cc in counts.items():
                if kk not in exact:
                    viol += cc                       # drawn but logprob says impossible
            rare = {}                                # per row: expected < 10 counts pooled (Poisson regime)
            for kk, p in exact.items():
                cc = counts.get(kk, 0)
                if p <= 0.0 or p >= 1.0 - 1e-15:
                    viol += int(abs(cc - n_draw * p) > 0.5)
                    continue
                if n_draw * p < 10:
                    e0, c0 = rare.get(kk[0], (0.0, 0))
                    rare[kk[0]] = (e0 + p, c0 + cc)
                    continue
                groups += 1
                zs.append((cc - n_draw * p) / math.sqrt(n_draw * p * (1 - p)))
            for p, cc in rare.values():
                e = n_draw * p
                if e >= 10:
                    groups += 1
                    zs.append((cc - e) / math.sqrt(e * (1 - p)))
                else:
                    self.assertLessEqual(cc, e + 5 * math.sqrt(e) + 5)
        zs = np.abs(np.asarray(zs))
        print(f"exact per-action groups: {groups}, max |z| {zs.max():.2f}, |z| > 3: {int((zs > 3).sum())}")
        self.assertEqual(viol, 0)
        self.assertLess(zs.max(), 4.6)               # ~100 groups: P(any > 4.6) ~ 4e-4
        self.assertLessEqual(int((zs > 3).sum()), 3)


# =========================================================================== estimator / row masks
class TestSampledBranchEstimator(unittest.TestCase):
    """Slot / ability rows are EXECUTED-branch rows.  Then
    E_{a ~ pi}[kl_family + m_slot kl_slot + m_cell kl_cell + m_ab kl_ab] equals the exact joint KL per row
    (the per-row quantity ``losses`` divides by the surrogate's denominator).  Same for entropy."""

    def test_estimator_is_unbiased_for_joint(self):
        _, _, n, th = actor_and_batch(0, 2, True)
        _, _, _, ref = actor_and_batch(1, 2, True)
        dt = torch.full((n,), 10.0, dtype=torch.float64)
        lp = LP.head_log_probs(th, dt)
        pf = lp["family"].exp()
        ps = torch.where(lp["slot_legal"], lp["slot"].exp(), torch.zeros_like(lp["slot"]))
        pa = torch.where(lp["ability_ready"], lp["ability"].exp(), torch.zeros_like(lp["ability"]))
        z = torch.zeros(n, dtype=torch.long)

        def est(action):
            kl = LP.head_kl(th, ref, dt, action)
            en = LP.head_entropy(th, dt, action)
            e_kl = (kl["kl_family"] + kl["mask_slot"] * kl["kl_slot"] + kl["mask_cell"] * kl["kl_cell"]
                    + kl["mask_ability"] * kl["kl_ability"])
            e_en = (en["ent_family"] + en["mask_slot"] * en["ent_slot"] + en["mask_cell"] * en["ent_cell"]
                    + en["mask_ability"] * en["ent_ability"])
            return e_kl, e_en, kl, en

        k0, h0, kl0, en0 = est({"family": z, "slot": z, "cell": z, "ability": z})        # executed wait
        kl_joint, ent_joint = kl0["kl_joint"], en0["ent_joint"]
        e_kl, e_en = pf[:, 0] * k0, pf[:, 0] * h0
        for s in range(4):                                                                # play on slot s
            k1, h1, _, _ = est({"family": torch.ones(n, dtype=torch.long), "slot": torch.full((n,), s),
                                "cell": z, "ability": z})
            e_kl = e_kl + pf[:, 1] * ps[:, s] * k1
            e_en = e_en + pf[:, 1] * ps[:, s] * h1
        k2, h2, _, _ = est({"family": torch.full((n,), 2), "slot": z, "cell": z, "ability": z})
        e_kl = e_kl + pf[:, 2] * pa.sum(-1) * k2
        e_en = e_en + pf[:, 2] * pa.sum(-1) * h2
        rel_kl = float(((e_kl - kl_joint).abs() / kl_joint.abs().clamp(min=1e-3)).max())
        rel_en = float(((e_en - ent_joint).abs() / ent_joint.abs().clamp(min=1e-3)).max())
        m = LP.legal_masks(th)
        # an all-rows mask ("every row with >= 2 legal slots / ready units") vs the joint's share
        lit = float(((m["n_slots"] >= 2) * kl0["kl_slot"] + (m["n_ready"] >= 2) * kl0["kl_ability"]).sum())
        share = float((pf[:, 1] * kl0["kl_slot"] + pf[:, 2] * kl0["kl_ability"]).sum())
        print(f"sampled-branch estimator vs exact joint: KL rel {rel_kl:.1e}, entropy rel {rel_en:.1e}; "
              f"literal all-rows slot+ability KL {lit:.3e} vs joint share {share:.3e} (x{lit / share:.1f})")
        self.assertLess(rel_kl, 1e-10)
        self.assertLess(rel_en, 1e-10)
        self.assertGreater(lit, 2.0 * share)

    def test_folded_family_ignores_blocked_play_odds(self):
        out = synth()
        dt = torch.full((4,), 10.0, dtype=torch.float64)
        ref = {k: v.clone() for k, v in out.items()}
        ref["family"][3, 1] += 3.0                  # forced row: play odds only (no behavioural effect)
        a = {"family": torch.zeros(4, dtype=torch.long), "slot": torch.zeros(4, dtype=torch.long),
             "cell": torch.zeros(4, dtype=torch.long), "ability": torch.zeros(4, dtype=torch.long)}
        kl = LP.head_kl(out, ref, dt, a)
        self.assertAlmostEqual(float(kl["kl_family"][3]), 0.0, places=14)
        ref["family"][1, 1] += 3.0                  # blocked row with abilities: wait/ability split moves
        kl = LP.head_kl(out, ref, dt, a)
        p = hazard_probs(out["family"], out["ability_avail"], dt)[1]
        q = hazard_probs(ref["family"], ref["ability_avail"], dt)[1]
        fp = np.array([float(p[0] + p[1]), float(p[2])])
        fq = np.array([float(q[0] + q[1]), float(q[2])])
        self.assertAlmostEqual(float(kl["kl_family"][1]), float((fp * np.log(fp / fq)).sum()), places=13)


# =========================================================================== numerics / API edges
class TestEdges(unittest.TestCase):
    def test_gradcheck_all_terms(self):
        """Analytic vs finite-difference gradients of executed_logp (every family, blocked, forced, small
        rate) and of the per-head KL / entropy; 16 cells keep the Jacobian small."""

        out = synth(n_cells=16)
        dt = torch.tensor([10.0, 7.0, 1e-9, 10.0], dtype=torch.float64)
        acts = [{"family": torch.tensor(f), "slot": torch.tensor([2, 0, 3, 0]), "cell": torch.tensor([3, 0, 9, 0]),
                 "ability": torch.tensor([0, 9, 63, 0])}
                for f in ([0, 0, 0, 0], [1, 1, 1, 1], [0, 2, 2, 0])]
        g = torch.Generator().manual_seed(3)
        ref = dict(out, family=out["family"] + torch.tensor([0.0, 0.3, -0.2], dtype=torch.float64),
                   **{k: out[k] + 0.4 * torch.randn(out[k].shape, generator=g, dtype=torch.float64)
                      for k in ("slot", "ability", "cells")})
        keys = ("family", "slot", "ability", "cells")

        for a in acts:
            def f(fam, slot, ab, cells, a=a):
                o = dict(out, family=fam, slot=slot, ability=ab, cells=cells)
                kl = LP.head_kl(o, ref, dt, a)
                en = LP.head_entropy(o, dt, a)
                return (LP.executed_logp(o, dt, a), kl["kl_family"], kl["kl_slot"], kl["kl_cell"],
                        kl["kl_ability"], en["ent_family"], en["ent_cell"])
            inp = tuple(out[k].clone().requires_grad_(True) for k in keys)
            self.assertTrue(torch.autograd.gradcheck(f, inp, eps=1e-6, atol=1e-6, rtol=1e-5))

    def test_empty_batch(self):
        out = {"family": torch.zeros(0, 3), "slot": torch.zeros(0, 4), "ability": torch.zeros(0, 64),
               "ability_avail": torch.zeros(0, dtype=torch.bool), "cells": torch.zeros(0, 4, S.N_CELLS)}
        a = {k: torch.zeros(0, dtype=torch.long) for k in ("family", "slot", "cell", "ability")}
        a["play_blocked"] = torch.zeros(0, dtype=torch.bool)
        self.assertEqual(LP.executed_logp(out, 10.0, a).shape, (0,))
        self.assertEqual(LP.head_kl(out, out, 10.0, a)["kl_joint"].shape, (0,))
        self.assertEqual(LP.head_entropy(out, torch.zeros(0), a)["ent_joint"].shape, (0,))
        self.assertEqual(LP.forced_mask(out).shape, (0,))
        self.assertEqual(LP.apply_temperature(out, 0.5, 0.5, 0.5)["cells"].shape, (0, 4, S.N_CELLS))

    def test_stored_action_conventions_and_dt_forms(self):
        """Shard-style actions (numpy int8 / int16, -1 sentinels on unused heads, lists) and dt as int64,
        float32, [n, 1], numpy or python give bitwise the same log pi."""

        _, _, n, out = actor_and_batch(0, 2, True)
        dt = torch.full((n,), 10.0, dtype=torch.float64)
        a = sample_actions(out, dt, generator=torch.Generator().manual_seed(5))
        base = LP.executed_logp(out, dt, a)
        f = a["family"]
        st = {"family": f.numpy().astype(np.int8),
              "slot": torch.where(f == 1, a["slot"], -1).numpy().astype(np.int8),
              "cell": torch.where(f == 1, a["cell"], -1).numpy().astype(np.int16),
              "ability": torch.where(f == 2, a["ability"], -1).numpy().astype(np.int8),
              "play_blocked": a["play_blocked"].numpy()}
        self.assertTrue(torch.equal(LP.executed_logp(out, dt, st), base))
        lists = {k: v.tolist() for k, v in st.items()}
        self.assertTrue(torch.equal(LP.executed_logp(out, dt, lists), base))
        for d in (10, 10.0, torch.full((n,), 10, dtype=torch.int64), torch.full((n, 1), 10.0, dtype=torch.float32),
                  np.full(n, 10, np.int32)):
            self.assertTrue(torch.equal(LP.executed_logp(out, d, a), base), type(d))

    def test_temperature_keeps_fp32_and_masks_on_real_rows(self):
        _, _, _, out = actor_and_batch(0, 2, False)
        self.assertEqual(out["cells"].dtype, torch.float32)
        for t in (0.05, 0.25, 4.0):
            r = LP.apply_temperature(out, t, t, t)
            for k in ("slot", "ability", "cells"):
                self.assertEqual(r[k].dtype, torch.float32)
                ok = out[k] > MASK_FILL / 2
                self.assertTrue(torch.equal(r[k] > MASK_FILL / 2, ok))
                self.assertTrue(bool((r[k][~ok] == MASK_FILL).all()))


# =========================================================================== fixed defects (were pinned)
class TestFixedDefects(unittest.TestCase):
    def test_tiny_temperature_must_not_change_legality(self):
        """A legal logit of -1 at T = 1e-9 would become -1e9 (illegal); fp32 +50 at T = 1e-38 is +inf.
        T < T_MIN is rejected; a T >= T_MIN that would still cross MASK_FILL / 2 is rejected by the guard."""

        out = synth()
        out["slot"][0, 2] = -1.0                       # the only legal slot of row 0
        for t in (1e-9, 1e-38, 0.5 * LP.T_MIN):
            with self.assertRaises(ValueError):
                LP.apply_temperature(out, t, 1.0, 1.0)
        o32 = synth(dtype=torch.float32)
        with self.assertRaises(ValueError):
            LP.apply_temperature(o32, 1.0, 1.0, 1e-38)
        far = {k: v.clone() for k, v in out.items()}
        far["slot"][0, 2] = -1.0e6                     # legal, but -1e6 / 1e-3 = -1e9 would be masked
        with self.assertRaises(ValueError):
            LP.apply_temperature(far, LP.T_MIN, 1.0, 1.0)
        r = LP.apply_temperature(out, LP.T_MIN, LP.T_MIN, LP.T_MIN)
        for k in ("slot", "ability", "cells"):
            self.assertTrue(torch.equal(LP.legal(r[k]), LP.legal(out[k])), k)

    def test_impossible_action_is_flagged(self):
        """An action ``out`` cannot execute (ability family on a row with no ready unit / a pointer that is not
        ready) is flagged ``valid = False``, gets exactly LOG_ZERO and sends NO gradient; possible rows are
        valid and differ between models."""

        out = synth()
        other = dict(out, ability=out["ability"] + torch.linspace(0, 3, 64, dtype=torch.float64))
        a = {"family": torch.full((4,), 2), "slot": torch.zeros(4, dtype=torch.long),
             "cell": torch.zeros(4, dtype=torch.long), "ability": torch.tensor([5, 9, 63, 5])}
        dt = torch.full((4,), 10.0, dtype=torch.float64)
        terms = LP.executed_logp_terms(out, dt, a)
        self.assertEqual(terms["valid"].tolist(), [False, True, True, False])
        lp1, lp2 = LP.executed_logp(out, dt, a), LP.executed_logp(other, dt, a)
        self.assertEqual(float(lp1[0]), LP.LOG_ZERO)
        self.assertNotEqual(float(lp1[1] - lp2[1]), 0.0)
        ab = out["ability"].clone().requires_grad_(True)
        LP.executed_logp(dict(out, ability=ab), dt, a).sum().backward()
        self.assertEqual(float(ab.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(ab.grad[3].abs().sum()), 0.0)
        self.assertGreater(float(ab.grad[1].abs().sum()), 0.0)
        # illegal slot / illegal cell / out-of-range index / dt = 0 play are all invalid
        p = {"family": torch.tensor([1, 1, 1, 1]), "slot": torch.tensor([1, 0, 0, 2]),
             "cell": torch.tensor([1, 0, 1, 700]), "ability": torch.full((4,), -1)}
        v = LP.executed_logp_terms(out, dt, p)["valid"]
        # row 0: slot 1 illegal; row 1: blocked -> executed wait (valid); row 2: cell 1 illegal on slot 0;
        # row 3: forced -> wait (valid)
        self.assertEqual(v.tolist(), [False, True, False, True])
        ok = {"family": torch.tensor([1, 0, 1, 0]), "slot": torch.tensor([2, -1, 3, -1]),
              "cell": torch.tensor([3, -1, 9, -1]), "ability": torch.full((4,), -1)}
        self.assertTrue(bool(LP.executed_logp_terms(out, dt, ok)["valid"].all()))
        self.assertFalse(bool(LP.executed_logp_terms(out, torch.zeros(4), ok)["valid"][0]))
        bad = dict(ok, cell=torch.tensor([700, -1, 9, -1]))
        self.assertFalse(bool(LP.executed_logp_terms(out, dt, bad)["valid"][0]))

    def test_sampled_actions_are_valid(self):
        _, _, n, out = actor_and_batch(0, 2, True)
        dt = unit_rate_dt(out)
        for t in (1.0, 0.25, 2.0):
            o = LP.apply_temperature(out, t, t, t)
            for seed in range(3):
                a = sample_actions(o, dt, generator=torch.Generator().manual_seed(seed))
                self.assertTrue(bool(LP.executed_logp_terms(o, dt, a)["valid"].all()), (t, seed))
        o = synth()
        idx = torch.arange(4).repeat_interleave(500)
        rep = {k: v[idx] for k, v in o.items()}
        a = sample_actions(rep, torch.full((2000,), 10.0, dtype=torch.float64),
                           generator=torch.Generator().manual_seed(9))
        self.assertTrue(bool(LP.executed_logp_terms(rep, 10.0, a)["valid"].all()))

    def test_subnormal_rate_gradient_finite(self):
        """Play - wait gaps of -712 / -720 make the rate subnormal (old backward: +-inf); -760 underflows the
        rate to 0.  Values follow log P_play = gap + log(dt / U) and the gradient is [-1, 1, 0]."""

        dt = 10.0
        for gap in (-30.0, -700.0, -712.0, -720.0, -760.0):
            fam = torch.tensor([[0.0, gap, MASK_FILL]], dtype=torch.float64, requires_grad=True)
            lp = LP.family_log_probs(fam, torch.tensor([False]), torch.tensor([True]), dt)
            self.assertAlmostEqual(float(lp[0, 1]), gap + math.log(dt / S.U_TICKS), places=9, msg=gap)
            lp[0, 1].backward()
            self.assertTrue(bool(torch.isfinite(fam.grad).all()), gap)
            self.assertTrue(torch.allclose(fam.grad, torch.tensor([[-1.0, 1.0, 0.0]], dtype=torch.float64),
                                           atol=1e-9), (gap, fam.grad))

    def test_series_branch_matches_closed_form(self):
        """Either side of SERIES_RATE the log P_act value equals log(-expm1(-rate)) (python float64)."""

        for r in (1e-12, 1e-6, 0.99 * LP.SERIES_RATE, LP.SERIES_RATE, 1.01 * LP.SERIES_RATE, 1e-2, 3.0):
            dt = r * S.U_TICKS                           # odds 1 (gap 0), no ability: rate = r
            fam = torch.tensor([[0.0, 0.0, MASK_FILL]], dtype=torch.float64)
            lp = LP.family_log_probs(fam, torch.tensor([False]), torch.tensor([True]), dt)
            want = math.log(-math.expm1(-r))
            self.assertLess(abs(float(lp[0, 1]) - want), 1e-14 * max(1.0, abs(want)), r)

    def test_actor_fast_path_matches_full(self):
        """The actor path gathers the executed slot before the log-softmax; same terms as the full path."""

        _, _, n, out = actor_and_batch(0, 2, False)
        dt = unit_rate_dt(out)
        a = sample_actions(out, dt, generator=torch.Generator().manual_seed(4))
        fast = LP.executed_logp_terms(out, dt, a)
        full = LP.executed_logp_terms(out, dt, a, lp=LP.head_log_probs(out, dt))
        for k in ("family", "slot", "cell", "ability", "total"):
            self.assertLess(float((fast[k] - full[k]).abs().max()), 1e-12, k)
        self.assertTrue(torch.equal(fast["valid"], full["valid"]))
        import time
        big = {k: v.repeat(3, *([1] * (v.dim() - 1))) for k, v in out.items()}
        ab = {k: v.repeat(3) for k, v in a.items() if k != "probs"}
        dtb = dt.repeat(3)
        t0 = time.perf_counter()
        for _ in range(20):
            LP.executed_logp(big, dtb, ab)
        t1 = time.perf_counter()
        for _ in range(20):
            sample_actions(big, dtb)
        t2 = time.perf_counter()
        print(f"actor cost ({big['family'].shape[0]} rows fp32): executed_logp {(t1 - t0) / 20 * 1e3:.2f} ms, "
              f"sample_actions {(t2 - t1) / 20 * 1e3:.2f} ms")


if __name__ == "__main__":
    unittest.main()
