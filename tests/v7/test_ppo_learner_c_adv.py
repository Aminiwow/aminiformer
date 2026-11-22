"""Adversarial checks of the learner's critic warm-up, micro-batches, policy_params rule, drift wiring, config
changes and pins.

What is attacked (src/amini/ppo/learner.py, config.py, drift.py):

* critic warm-up: no policy / reference forward, no optimiser or monitor change; the warm-up critic step equals the
  PPO update's critic step on the same data; the pass rule must hold on every path into a policy step;
  the critic optimiser carried by ``from_il(critic_from=)``; the fixed set's recording policy;
* micro-batches: il_v7's ``_split`` rule (and the documented oversized-seat-game difference), the inclusive
  token / row caps, ``micro_cost`` = the padding ``TK.collate`` really produces, and a whole ``Learner.update``
  (KL anchor active) that gives the same weights, critic and metrics for one micro-batch, one seat-game per
  micro-batch, a token-cap split and a reversed micro-batch order;
* the policy_params rule on every model-building path (``set_reference`` included), with a AminiFormerConfig
  subclass that has the ``policy_params`` field;
* drift: judge edges, the alarm through ``Learner.update`` (alarm, never a stop), the gate_bias precondition;
* config changes: every config leaf that can change is either on ``RUNTIME_CHANGE_KEYS`` or refused, the diff
  is one-to-one with the leaves, the whitelisted keys leave the monitor thresholds alone and take effect;
* pins: an unquoted sha256 in YAML / overrides, the pin checked before unpickling, a resume with no recorded
  source sha256.

The defects this review found were pinned as ``expectedFailure`` BUG pins; all of them are fixed and the tests
now run as plain regression tests (docstrings still say "BUG" to name what they caught). ``BUG`` stays available
for a new pin (``PPO_ADV_SHOW_BUGS=1`` runs pins as plain tests). CPU only, deterministic, synthetic shards
(test_ppo_learner fixture, tests/v7/common.py; skipped without the private dataset), no game environment, no GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import tiny_model  # noqa: E402
from test_ppo_learner import (BASE_OVERRIDES, MAIN_YAML, RID, STEP_OVERRIDES, UPDATE, Fixture,  # noqa: E402
                              config, write_update)
from test_ppo_learner_c import WARMUP_OVERRIDES, cfg_with, snap  # noqa: E402
import test_ppo_drift as TD  # noqa: E402

from amini.nn.aminiformer import AminiFormerConfig  # noqa: E402
from amini.ppo import buffer as B  # noqa: E402
from amini.ppo import config as C  # noqa: E402
from amini.ppo import drift as DR  # noqa: E402
from amini.ppo import gae as G  # noqa: E402
from amini.ppo import learner as LR  # noqa: E402
from amini.ppo import monitor as M  # noqa: E402
from amini.train import il_v7 as IL  # noqa: E402

torch.set_num_threads(2)

BUG = (lambda f: f) if os.environ.get("PPO_ADV_SHOW_BUGS") else unittest.expectedFailure
T1 = (1.0, 1.0, 1.0)
WU1 = STEP_OVERRIDES + ["critic.warmup.max_updates=1", "critic.warmup.stop_trend_updates=1"]


def il_split(sizes, toks, micro_rows, max_tokens):
    """il_v7.EpisodeBatches._split on chunks of the given row counts / token maxima (no dataset needed)."""
    eb = object.__new__(IL.EpisodeBatches)
    eb.items = [((0, i, 0, 0, n), 0, 0, n) for i, n in enumerate(sizes)]
    eb.tok = list(toks)
    eb.cfg = NS(micro_rows=micro_rows, max_batch_tokens=max_tokens)
    return eb._split(list(range(len(sizes))))


def close_states(tc, a, b, rtol, atol, what):
    tc.assertEqual(set(a), set(b), what)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=rtol, atol=atol, msg=f"{what}: {k}")


# =========================================================================== critic warm-up
class TestWarmupAdv(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        wcfg = config(WARMUP_OVERRIDES)
        cls.wroot = cls.root / "adv-warmup"
        for u in (1, 2):
            write_update(cls.wroot, wcfg, policy_digest=cls.digest, update=u)
        cls.froot = cls.root / "adv-fixed"
        write_update(cls.froot, wcfg, policy_digest=cls.digest, update=1)
        cls.offroot = cls.root / "adv-off-policy"                   # same rows, stamped with another policy
        write_update(cls.offroot, wcfg, policy_digest="f" * 64, update=1)
        cls.wu1root = cls.root / "adv-wu1"                          # shards under the WU1 config digest
        write_update(cls.wu1root, config(WU1), policy_digest=cls.digest, update=1)

    def fixed(self, lr, root=None):
        return LR.FixedSetEV(lr, [B.load_update(root or self.froot, 1)], expect_games=4)

    def test_warmup_never_runs_the_policy_or_reference_and_leaves_optimiser_and_monitor(self):
        lr = LR.Learner.from_il(config(WARMUP_OVERRIDES), self.src)
        fs = self.fixed(lr)
        mon = json.dumps(lr.monitor.state_dict(), sort_keys=True, default=str)
        ref = snap(lr.reference)
        boom = AssertionError("forward of a model the warm-up must not run")
        with mock.patch.object(lr.policy, "forward", side_effect=boom), \
                mock.patch.object(lr.reference, "forward", side_effect=boom):
            lr.warmup_update(B.load_update(self.wroot, 1), fs)
        self.assertEqual(lr.opt.state_dict()["state"], {})              # policy optimiser never stepped
        self.assertEqual(json.dumps(lr.monitor.state_dict(), sort_keys=True, default=str), mon)
        self.assertTrue(all(torch.equal(v, snap(lr.reference)[k]) for k, v in ref.items()))
        self.assertTrue(all(p.grad is None for p in lr.critic.step_encoder.parameters()))
        self.assertEqual((lr.steps_total, lr.next_update, lr.policy_digest()), (0, 1, self.digest))

    def test_warmup_refuses_off_policy_shards(self):
        lr = LR.Learner.from_il(config(WARMUP_OVERRIDES), self.src)
        with self.assertRaisesRegex(ValueError, "lag 0"):
            lr.warmup_update(B.load_update(self.offroot, 1), self.fixed(lr))
        self.assertEqual(lr.warmup["done"], 0)

    def test_warmup_critic_step_equals_the_ppo_update_critic_step(self):
        """prepare_critic + train_critic is exactly the critic half of Learner.update (same λ-returns, value
        weights, minibatch order): the warm-up trains the critic PPO will use, nothing else."""
        cfg = config(STEP_OVERRIDES)
        a, b = LR.Learner.from_il(cfg, self.src), LR.Learner.from_il(cfg, self.src)
        a.warmup_update(self.data(a, self.shards_step), LR.FixedSetEV(a, [B.load_update(self.shards_step, 1)],
                                                                       expect_games=4))
        res = b.update(self.data(b, self.shards_step), replay_bitwise_ok=True)
        self.assertGreater(res.steps, 0)
        close_states(self, snap(a.critic), snap(b.critic), 0.0, 1e-12, "critic")
        self.assertAlmostEqual(a.last_metrics["critic_mse_after"], res.metrics["critic_mse_after"], delta=1e-12)

    def test_fixed_set_values_do_not_depend_on_chunk_or_micro_order(self):
        lr = LR.Learner.from_il(config(WARMUP_OVERRIDES), self.src)
        sgs = list(B.load_update(self.froot, 1).seat_games())
        v64 = lr.critic_values(sgs, chunk=64)
        v1 = lr.critic_values(sgs, chunk=1)
        rev = lambda items, rows, tok=0: [[it] for it in reversed(list(items))]   # noqa: E731
        with mock.patch.object(LR, "split_micro", side_effect=rev):
            vr = lr.critic_values(sgs, chunk=64)
        self.assertEqual(v64.shape, (sum(sg.n for sg in sgs),))
        np.testing.assert_allclose(v1, v64, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vr, v64, rtol=0, atol=1e-12)

    def test_resumed_warmup_refuses_another_fixed_set(self):
        lr = LR.Learner.from_il(config(WARMUP_OVERRIDES), self.src)
        lr.warmup_update(B.load_update(self.wroot, 1), self.fixed(lr))
        p = self.root / "adv-wu-resume.pt"
        lr.save(p)
        lr2 = LR.Learner.from_checkpoint(p, lr.cfg)
        other = LR.FixedSetEV(lr2, [B.load_update(self.wroot, 2)], expect_games=4)    # same games, update 2
        self.assertNotEqual(other.digest, self.fixed(lr2).digest)
        with self.assertRaisesRegex(ValueError, "fixed EV set changed"):
            lr2.warmup_update(B.load_update(self.wroot, 2), other)
        lr2.warmup_update(B.load_update(self.wroot, 2), self.fixed(lr2, self.wroot))   # identical recording: ok

    def _failed_warmup(self):
        cfg = config(WU1)
        lr = LR.Learner.from_il(cfg, self.src)
        res = lr.warmup_update(B.load_update(self.wu1root, 1), self.fixed(lr, self.wu1root))
        self.assertTrue(res.status.stop)
        lr.warmup["status"]["passed"] = False          # the tiny critic's EV means nothing: pin the outcome
        lr.warmup["status"]["reason"] = "reached 1 warm-up updates"
        return cfg, lr

    def _expect_refused(self, lr, data):
        try:
            res = lr.update(data, replay_bitwise_ok=True)
        except (RuntimeError, ValueError) as e:
            self.assertRegex(str(e), "warm-?up")
            return
        self.fail(f"{res.steps} policy steps after a failed critic warm-up (the critic remedy comes before any "
                  "policy step)")

    def test_failed_warmup_then_resume_cannot_step_the_policy(self):
        """BUG: the pass rule is enforced only by ``from_il(critic_from=)``. A block that resumes the
        warm-up checkpoint with ``from_checkpoint`` (same config) and calls ``update`` trains the policy on a
        critic whose warm-up failed."""
        cfg, lr = self._failed_warmup()
        p = self.root / "adv-wu-failed.pt"
        lr.save(p)
        lr2 = LR.Learner.from_checkpoint(p, cfg)
        self.assertFalse(lr2.warmup["status"]["passed"])
        self._expect_refused(lr2, B.load_update(self.wu1root, 1))

    def test_failed_warmup_same_learner_cannot_step_the_policy(self):
        """BUG (same cause): the learner that just failed its warm-up accepts ``update`` at once."""
        _, lr = self._failed_warmup()
        self._expect_refused(lr, B.load_update(self.wu1root, 1))

    def _passed_warmup_ckpt(self, name):
        lr = LR.Learner.from_il(config(WARMUP_OVERRIDES), self.src)
        fs = self.fixed(lr)
        for u in (1, 2):
            lr.warmup_update(B.load_update(self.wroot, u), fs)
        lr.warmup["status"].update(passed=True, ev_warmup_end=0.2)
        p = self.root / name
        lr.save(p)
        return lr, p

    def test_critic_from_uses_the_ppo_configs_critic_optimiser_settings(self):
        """BUG: ``critic_opt.load_state_dict`` restores the warm-up's param-group hyper-parameters, so a
        PPO config whose ``critic.lr`` / betas / eps differ from the warm-up config is silently ignored
        (``from_il(critic_from=)`` explicitly allows another config digest)."""
        _, p = self._passed_warmup_ckpt("adv-wu-lr.pt")
        lr = LR.Learner.from_il(config(BASE_OVERRIDES + ["critic.lr=5.0e-5"]), self.src, critic_from=p)
        self.assertEqual([g["lr"] for g in lr.critic_opt.param_groups], [5e-5])

    def test_fixed_set_must_be_recorded_by_the_learners_policy(self):
        """BUG: FixedSetEV takes shards stamped with any policy digest; the fixed set is defined as recorded
        with theta = the base IL model (the warm-up learner's acting weights), and the warm-up batches are
        already held to that (prepare_critic)."""
        lr = LR.Learner.from_il(config(WARMUP_OVERRIDES), self.src)
        with self.assertRaisesRegex(ValueError, "polic"):
            self.fixed(lr, self.offroot)


# =========================================================================== micro-batches
class TestMicroAdv(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cfg = config(STEP_OVERRIDES)

        def load(lr):
            return lr.load(cls.shards_step, UPDATE, expect_games=4, expect_workers=2, expect_rollout_id=RID,
                           expect_config_digest=lr.config_digest)

        probe = LR.Learner.from_il(cfg, cls.src)
        prep = probe.prepare(load(probe))
        cls.tok_cap = min(IL.micro_cost(it.n, *it.tok) for it in prep.items)
        rev = lambda items, rows, tok=0: [[it] for it in reversed(list(items))]   # noqa: E731
        # a reference away from theta (as after a promotion), so the KL anchor has a gradient in every run;
        # identity=False because KL(theta_0 || ref) > 0 by construction (the micro path is what is tested)
        ref = copy.deepcopy(probe.policy)
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for prm in ref.parameters():
                prm.add_(0.05 * torch.randn(prm.shape, generator=gen, dtype=prm.dtype))
        cls.runs = {}
        for name, kw in (("one", {}), ("rows1", {"max_micro_rows": 1}), ("tok", {"max_micro_tokens": cls.tok_cap}),
                         ("rev", {})):
            lr = LR.Learner.from_il(cfg, cls.src, **kw)
            lr.set_reference(ref)
            data = load(lr)
            if name == "rev":
                with mock.patch.object(LR, "split_micro", side_effect=rev):
                    res = lr.update(data, identity=False)
            else:
                res = lr.update(data, identity=False)
            cls.runs[name] = (lr, res)

    def test_split_micro_vs_il_split(self):
        """Seat-games within the row cap: exactly il_v7's _split. With oversized seat-games il_v7's start count
        ceil(rows / cap) over-counts (one 700-row seat-game is one bin, not 2.3), so it often falls through to
        one bin per item; the learner's start count is a true lower bound, so it never needs more micro-batches,
        every multi-item bin keeps both caps and an oversized seat-game is alone (the documented difference)."""
        fewer = 0
        for seed in range(60):
            rng = np.random.default_rng(1000 + seed)
            n = int(rng.integers(1, 30))
            big = seed % 2 == 1
            sizes = [int(x) for x in rng.integers(1, 700 if big else 301, n)]
            toks = [(int(a), int(b)) for a, b in rng.integers(0, 120, (n, 2))]
            cap = int(rng.choice([0, 20000, 60000]))
            items = [NS(n=s, temperature=T1, tok=t, i=i) for i, (s, t) in enumerate(zip(sizes, toks))]
            ours = [[it.i for it in m] for m in LR.split_micro(items, 300, cap)]
            theirs = il_split(sizes, toks, 300, cap)
            with self.subTest(seed=seed, oversized=big):
                if not big:
                    self.assertEqual(ours, theirs)
                    continue
                self.assertLessEqual(len(ours), len(theirs))
                fewer += len(ours) < len(theirs)
                self.assertEqual(sorted(i for m in ours for i in m), list(range(n)))
                for m in ours:
                    self.assertEqual(m, sorted(m))
                    if len(m) > 1:
                        rows = sum(sizes[i] for i in m)
                        self.assertLessEqual(rows, 300)
                        if cap:
                            self.assertLessEqual(IL.micro_cost(rows, max(toks[i][0] for i in m),
                                                               max(toks[i][1] for i in m)), cap)
                for i, s in enumerate(sizes):
                    if s > 300:
                        self.assertIn([i], ours)
        self.assertGreater(fewer, 0)

    def test_caps_are_inclusive_like_il(self):
        a, b = NS(n=100, temperature=T1, tok=(10, 20)), NS(n=50, temperature=T1, tok=(30, 5))
        cost = IL.micro_cost(150, 30, 20)
        self.assertEqual(len(LR.split_micro([a, b], 1024, cost)), 1)            # == cap fits
        self.assertEqual(len(LR.split_micro([a, b], 1024, cost - 1)), 2)        # one token over splits
        self.assertEqual(len(LR.split_micro([a, b], 150, 0)), 1)                # == row cap fits
        self.assertEqual(len(LR.split_micro([a, b], 149, 0)), 2)
        self.assertEqual(il_split([100, 50], [(10, 20), (30, 5)], 1024, cost), [[0, 1]])
        self.assertEqual(il_split([100, 50], [(10, 20), (30, 5)], 1024, cost - 1), [[0], [1]])
        self.assertEqual(LR.split_micro([a], 1, 1), [[a]])                      # one seat-game always runs
        mic = LR.split_micro([a, b], 1, 1)
        self.assertEqual(sorted(map(id, (it for m in mic for it in m))), sorted(map(id, (a, b))))

    def test_micro_cost_is_the_padding_collate_produces(self):
        lr = self.runs["one"][0]
        items = [lr._item(i, sg) for i, sg in enumerate(B.load_update(self.shards_step, UPDATE).seat_games())]
        for grp in [[it] for it in items] + [items[:2], items[2:5], items]:
            b = lr._batch(grp)
            width = 1 + sum(int(g["valid"].shape[-1]) for g in b["groups"].values())
            real = int(b["row"]["row_valid"].sum()) * width
            want = IL.micro_cost(sum(it.n for it in grp), max(it.tok[0] for it in grp), max(it.tok[1] for it in grp))
            with self.subTest(n=len(grp)):
                self.assertEqual(real, want)

    def test_the_variants_really_split(self):
        one = self.runs["one"][1].metrics
        self.assertEqual(one["micro_over_token_cap"], 0)
        for name in ("rows1", "tok"):
            m = self.runs[name][1].metrics
            with self.subTest(name=name):
                self.assertEqual(m["micro_batches"], m["seat_games"])
                self.assertGreater(m["micro_batches"], one["micro_batches"])
        self.assertEqual(self.runs["tok"][1].metrics["micro_token_cap"], self.tok_cap)
        self.assertEqual(self.runs["rows1"][1].metrics["micro_rows"], 1)

    def test_full_update_is_split_invariant(self):
        """Global denominators: one micro-batch per minibatch, one seat-game per micro-batch (row cap 1 or a
        token cap) and a reversed micro-batch order give the same optimiser steps, policy, critic and record."""
        base_lr, base = self.runs["one"]
        skip_prefix = ("micro_", "peak_mem", "policy_digest", "kl_grad_ratio/")
        for name in ("rows1", "tok", "rev"):
            lr, res = self.runs[name]
            with self.subTest(name=name):
                self.assertEqual((res.steps, res.early_stopped, res.skipped_minibatches),
                                 (base.steps, base.early_stopped, base.skipped_minibatches))
                self.assertGreater(res.steps, 0)
                close_states(self, snap(lr.policy), snap(base_lr.policy), 1e-7, 1e-10, f"{name} policy")
                close_states(self, snap(lr.critic), snap(base_lr.critic), 1e-7, 1e-10, f"{name} critic")
                for k, v in base.metrics.items():
                    if k.startswith(skip_prefix) or isinstance(v, bool) or not isinstance(v, (int, float)):
                        continue
                    w = res.metrics[k]
                    if not np.isfinite(v):
                        self.assertFalse(np.isfinite(w), k)
                        continue
                    self.assertAlmostEqual(w, v, delta=1e-8 * max(1.0, abs(v)), msg=k)

    def test_kl_grad_ratio_is_split_invariant(self):
        """BUG (diagnostic only): ``kl_grad_ratio/<head>`` is taken on the FIRST MICRO-BATCH of the first
        minibatch (``mi == 0``), so the logged value depends on the micro-batch caps; the losses module defines
        it on one minibatch."""
        base = self.runs["one"][1].metrics
        other = self.runs["rows1"][1].metrics
        keys = sorted(k for k in base if k.startswith("kl_grad_ratio/"))
        self.assertTrue(keys)
        for k in keys:
            self.assertAlmostEqual(other[k], base[k], delta=1e-8 * max(1.0, abs(base[k])), msg=k)


# =========================================================================== policy_params rule
@dataclasses.dataclass
class _PPConfig(AminiFormerConfig):
    """AminiFormerConfig plus a ``policy_params`` field (used when AminiFormerConfig has no such field)."""
    policy_params: Optional[Dict[str, Any]] = None


def _with_pp(model, pp):
    """A copy of ``model`` whose config carries ``policy_params=pp``: ``dataclasses.replace`` when
    AminiFormerConfig has the field, else the subclass."""
    m = copy.deepcopy(model)
    if "policy_params" in {f.name for f in dataclasses.fields(AminiFormerConfig)}:
        m.cfg = dataclasses.replace(m.cfg, policy_params=pp)
        return m
    fields = {f.name: getattr(m.cfg, f.name) for f in dataclasses.fields(AminiFormerConfig)}
    m.cfg = _PPConfig(**fields, policy_params=pp)
    return m


class TestPolicyParamsAdv(Fixture):

    def test_neutral_spellings_are_neutral(self):
        mc = self.learner().policy.cfg.to_dict()
        for pp in ({"greedy": {"slot": False, "cell": False}}, {"temperature": {"cell": 1}},
                   {"temperature": np.float64(1.0)}, {"temperature": 1, "greedy": False, "gate_stream": "shared"},
                   {"temperature": None, "greedy": None}):
            with self.subTest(pp=pp):
                self.assertTrue(LR.is_neutral_policy_params(pp))
                # absent (AminiFormerConfig without the field) or None (with the field: neutral = None)
                self.assertIsNone(dataclasses.asdict(LR.ppo_config(dict(mc, policy_params=pp))).get("policy_params"))

    def test_learner_refuses_models_whose_config_class_has_the_field(self):
        lr = self.learner()
        bad = _with_pp(lr.policy, {"temperature": {"slot": 0.5}})
        with self.assertRaisesRegex(LR.PolicyParamsError, "policy"):
            LR.Learner(lr.cfg, bad, lr.reference, lr.critic)
        LR.Learner(lr.cfg, _with_pp(lr.policy, None), lr.reference, lr.critic)    # neutral: fine

    def test_resume_refuses_a_non_neutral_reference_config(self):
        lr = self.learner()
        p = self.root / "adv-ref-pp.pt"
        lr.save(p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ck["ppo"]["reference"]["model_config"] = dict(ck["ppo"]["reference"]["model_config"],
                                                      policy_params={"greedy": True})
        torch.save(ck, p)
        with self.assertRaises(LR.PolicyParamsError):
            LR.Learner.from_checkpoint(p, lr.cfg)

    def test_a_non_neutral_reference_never_survives_a_resume(self):
        """Whatever set_reference does, the next block refuses such a reference (model_from_state)."""
        lr = self.learner()
        champ = _with_pp(lr.policy, {"temperature": 0.5})
        try:
            lr.set_reference(champ)
        except LR.PolicyParamsError:
            return                                             # the fixed behaviour (BUG pin below)
        self.assertFalse(LR.is_neutral_policy_params(lr.reference.cfg.policy_params))
        p = self.root / "adv-ref-pp2.pt"
        lr.save(p)
        with self.assertRaises(LR.PolicyParamsError):
            LR.Learner.from_checkpoint(p, lr.cfg)

    def test_set_reference_refuses_non_neutral_policy_params(self):
        """BUG: ``set_reference`` (the promotion path) skips ``check_model_policy_params``; with the model
        config's field a tempered champion becomes the KL reference, is saved, and the chain only fails at the
        next block's ``from_checkpoint``."""
        lr = self.learner()
        with self.assertRaises(LR.PolicyParamsError):
            lr.set_reference(_with_pp(lr.policy, {"temperature": {"slot": 0.5}}))


# =========================================================================== drift
class TestDriftAdv(unittest.TestCase):

    def test_judge_edges(self):
        cfg = TD.ppo_cfg(nll_increase_max={"gate": 0.25, "slot": 0.5, "cell": 0.5, "ability": 0.5},
                         gate_auc_drop_max=0.125, calibration_ratio_band=[0.5, 2.0])
        ref = TD.report(terms={"gate": 0.5}, auc=0.75)
        self.assertEqual(DR.judge(cfg, ref, TD.report(terms={"gate": 0.75}, auc=0.625))[0], ())   # == limits
        reasons, _ = DR.judge(cfg, ref, TD.report(terms={"gate": 0.75 + 1e-9}, auc=0.625 - 1e-9))
        self.assertEqual(len(reasons), 2)
        reasons, d = DR.judge(cfg, ref, TD.report(base=0.0))           # policy predicts no plays at all
        self.assertEqual((len(reasons), d["calibration_rel/base"]), (1, 0.0))
        self.assertEqual(DR.judge(cfg, TD.report(base=-1.0), TD.report(base=7.0))[0], ())   # IL ratio <= 0
        no_cal = {k: v for k, v in TD.report().items() if k != "calibration"}
        self.assertEqual(len(DR.judge(cfg, TD.report(), no_cal)[0]), 2)  # both strata vanished: alarm
        self.assertEqual(DR.judge(cfg, no_cal, TD.report())[0], ())       # IL had none: not judged

    def test_cadence_is_the_monitors(self):
        cfg = TD.ppo_cfg()
        d = cfg.to_dict()
        d["policy"]["backbone_dtype"] = "fp32"
        mon = M.Monitor(thresholds=M.Thresholds.from_mapping(C.from_dict(d).monitor_thresholds()))
        for u in range(1, 26):
            with self.subTest(u=u):
                self.assertEqual(DR.drift_due(cfg, u), "drift_alarm" in mon.expected_update_keys(u))

    def test_evaluate_leaves_weights_grads_and_mode(self):
        ev = DR.DriftEvaluator(TD.ppo_cfg(), TD.train_cfg(), TD.source(), TD.tables(), items=TD.train_items())
        m = TD.model(0)
        m.eval()
        before = snap(m)
        a = ev.evaluate(m)
        b = ev.evaluate(m)
        self.assertFalse(m.training)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertTrue(all(torch.equal(v, snap(m)[k]) for k, v in before.items()))
        self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_reference_with_gate_bias_is_refused(self):
        """BUG: the drift reference must be the base IL model with gate_bias None (module and set_reference
        docstrings); ``il_v7.load_policy`` keeps a checkpoint's calibration bias, and the evaluator accepted it,
        so the calibration ratio and gate NLL would compare a biased IL with the unbiased policy."""
        ev = DR.DriftEvaluator(TD.ppo_cfg(), TD.train_cfg(), TD.source(), TD.tables(), items=TD.train_items())
        biased = tiny_model(0, double=False, gate_bias=[[0.0, -0.05, 0.0], [0.0, -0.1, 0.0]])   # example values
        with self.assertRaises(ValueError):
            ev.set_reference(biased)


class TestDriftThroughUpdate(Fixture):

    def test_drift_alarm_reaches_the_monitor_as_an_alarm_only(self):
        lr = self.learner(STEP_OVERRIDES)
        res = DR.DriftResult(update=1, reasons=("gate NLL +0.2000 over IL (limit 0.1)",),
                             deltas={"nll_delta/gate": 0.2, "gate_auc_drop": None}, report={"selection_nll": 1.0},
                             subset_digest="d", n_games=1)
        out = lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True, extra_metrics=res.metrics())
        self.assertIn(("alarm", "drift"), [(a.kind, a.monitor) for a in out.actions])
        self.assertFalse(out.stopped)
        self.assertGreater(out.steps, 0)
        self.assertEqual(out.metrics["drift_alarm"], list(res.reasons))
        self.assertTrue(np.isnan(out.metrics["drift/gate_auc_drop"]))


# =========================================================================== recorded config changes
def _leaf_paths(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaf_paths(v, prefix + (k,))
        else:
            yield prefix + (k,), v


def _set(d, path, v):
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = v


def _mutations(v):
    """Candidate new values for a leaf, never equal to the old one."""
    if v is None:                     # a null threshold a calibration fills in (runtime keys)
        out = [0.5, 0.25, 1.5]
    elif isinstance(v, bool):
        out = [not v]
    elif isinstance(v, int):
        out = [v + 1, v - 1, v * 2]
    elif isinstance(v, float):
        out = [v * 1.5, v * 0.5, v + 1e-4, v - 1e-4, 0.5, v + 1.0] if v else [0.5, 0.25, 1.0]
    elif isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        out = [[m] + list(v[1:]) for m in _mutations(v[0])] + [list(v[:-1]) + [m] for m in _mutations(v[-1])]
    else:
        out = []
    return [m for m in out if m != v]


class TestConfigChangeAdv(Fixture):

    def test_every_changeable_leaf_is_whitelisted_or_refused(self):
        """Mutate each numeric / bool / numeric-list leaf of the run config (first value that validates): the
        diff names exactly that leaf, the digest changes, and ``config_change`` accepts it iff the leaf is in
        ``RUNTIME_CHANGE_KEYS``."""
        old = config(STEP_OVERRIDES)
        base = old.to_dict()
        tried, accepted = 0, set()
        for path, v in _leaf_paths(base):
            dotted = ".".join(str(p) for p in path)
            new = None
            for nv in _mutations(v):
                d = copy.deepcopy(base)
                _set(d, path, nv)
                try:
                    new = C.from_dict(d)
                    break
                except C.ConfigError:
                    continue
            if new is None:
                continue
            tried += 1
            with self.subTest(leaf=dotted):
                self.assertNotEqual(new.digest(), old.digest())
                self.assertEqual(list(C.config_diff(old, new)), [dotted])
                if dotted in C.RUNTIME_CHANGE_KEYS:
                    rec = C.config_change(old, new, update=3, reason="adv")
                    C.check_config_change(rec, old, new)
                    accepted.add(dotted)
                else:
                    with self.assertRaisesRegex(C.ConfigError, "may not change mid-run"):
                        C.config_change(old, new, update=3, reason="adv")
                    with self.assertRaises(C.ConfigError):
                        C.with_changes(old, {dotted: nv})
        self.assertGreater(tried, 100)
        self.assertEqual(accepted, set(C.RUNTIME_CHANGE_KEYS))

    def test_a_valid_record_cannot_carry_a_second_change(self):
        lr = self.learner(STEP_OVERRIDES)
        p = self.root / "adv-chg.pt"
        lr.save(p)
        old = lr.cfg
        new = C.with_changes(old, {"kl.beta": 0.04})
        rec = C.config_change(old, new, update=1, reason="pilot retry")
        smuggled = C.from_dict(dict(new.to_dict(), early_stop=dict(new.to_dict()["early_stop"], joint_k3=9.0)))
        with self.assertRaises(C.ConfigError):
            LR.Learner.from_checkpoint(p, smuggled, config_change=rec)
        forged = dict(rec, changes={"kl.beta": [old.kl.beta, 0.04]}, new_digest=smuggled.digest())
        with self.assertRaises(C.ConfigError):
            LR.Learner.from_checkpoint(p, smuggled, config_change=forged)
        for bad in ({"kl": {"beta": 0.02}}, {"kl.beta.x": 1}, {"policy.lr ": 1e-4}):
            with self.subTest(bad=bad):
                with self.assertRaises(C.ConfigError):
                    C.with_changes(old, bad)
        for kw in ({"update": 0, "reason": "x"}, {"update": True, "reason": "x"}, {"update": 2, "reason": "  "}):
            with self.subTest(kw=kw):
                with self.assertRaises(C.ConfigError):
                    C.config_change(old, new, **kw)

    def test_whitelisted_keys_leave_the_monitor_thresholds_alone(self):
        old = config(STEP_OVERRIDES)
        th = M.Thresholds.from_mapping(old.monitor_thresholds())
        for key, v in (("kl.beta", 0.04), ("policy.lr", 5e-4), ("gae.lambda_tick", 0.997),
                       ("rollout.games_per_update", 2000), ("critic.epochs", 6)):
            with self.subTest(key=key):
                self.assertEqual(M.Thresholds.from_mapping(C.with_changes(old, {key: v}).monitor_thresholds()), th)

    def test_change_takes_effect_in_the_next_update(self):
        lr = self.learner(STEP_OVERRIDES)
        new = C.with_changes(lr.cfg, {"critic.epochs": 6, "gae.lambda_tick": 0.997})
        lr.change_config(new, reason="critic remedy (stall ladder)")
        data = lr.load(self.shards_step, UPDATE, expect_games=4)       # recorded before the change
        prep = lr.prepare(data)
        for it in prep.items:
            r = G.gae_episode(G.episode_rewards(it.n, it.sg.outcome), it.value_old, it.sg.rows["tick"],
                              gamma_tick=new.gae.gamma, lambda_tick=0.997)
            np.testing.assert_allclose(it.ret.cpu().numpy(), r.returns, rtol=0, atol=1e-12)
        res = lr.update(data, replay_bitwise_ok=True)
        n_mb = -(-res.metrics["seat_games"] // new.policy.minibatch_seat_games)
        self.assertEqual(res.metrics["critic_steps"], 6 * n_mb)


# =========================================================================== checkpoint pins
class TestPinsAdv(Fixture):

    def test_unquoted_sha256_in_yaml_and_overrides(self):
        sha = hashlib.sha256(b"example pin").hexdigest()
        self.assertEqual(config(BASE_OVERRIDES + [f"opponents.checkpoint_sha256.il_v4={sha}"])
                         .opponents.checkpoint_sha256["il_v4"], sha)
        lines = MAIN_YAML.read_text(encoding="utf-8").splitlines()
        i = next(k for k, ln in enumerate(lines) if ln.split(" #")[0].strip() == "checkpoint_sha256:")
        j = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith("il_v4:"))
        lines[j] = lines[j].replace("null", sha)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as d:
            p = Path(d) / "pinned.yaml"
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(C.load_config(p, BASE_OVERRIDES).opponents.checkpoint_sha256["il_v4"], sha)
        for bad in ("1" * 64, sha.upper(), sha[:-1]):               # read as an int / wrong case / short: loud
            with self.subTest(bad=bad[:8]):
                with self.assertRaises(C.ConfigError):
                    config(BASE_OVERRIDES + [f"opponents.checkpoint_sha256.il_v4={bad}"])

    def test_config_change_cannot_touch_the_pins(self):
        old = config(STEP_OVERRIDES)
        d = old.to_dict()
        d["opponents"]["checkpoint_sha256"]["il_v4"] = "a" * 64
        with self.assertRaisesRegex(C.ConfigError, "checkpoint_sha256"):
            C.config_change(old, C.from_dict(d), update=2, reason="x")

    def test_pin_is_checked_before_the_checkpoint_is_unpickled(self):
        """BUG: ``from_il`` unpickles the file (``torch.load(weights_only=False)``) before comparing its
        sha256 with the pin; hashing first is free and keeps a wrong file from running any pickle code."""
        with mock.patch.object(LR.torch, "load", side_effect=AssertionError("unpickled before the pin check")):
            with self.assertRaises(LR.PinError):
                LR.Learner.from_il(cfg_with(il_v4="0" * 64), self.src)

    def test_resume_with_a_pin_needs_the_recorded_source_sha(self):
        """BUG: ``from_checkpoint`` skips the il_v4 pin when the checkpoint has no recorded source sha256
        (``ppo.source`` missing or without ``sha256``), so a set pin is not enforced on that resume."""
        sha = IL.file_sha256(str(self.src))
        cfg = cfg_with(il_v4=sha)
        lr = LR.Learner.from_il(cfg, self.src)
        p = self.root / "adv-pin-nosrc.pt"
        lr.save(p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ck["ppo"]["source"] = {k: v for k, v in ck["ppo"]["source"].items() if k != "sha256"}
        torch.save(ck, p)
        with self.assertRaises(LR.PinError):
            LR.Learner.from_checkpoint(p, cfg)


if __name__ == "__main__":
    unittest.main()
