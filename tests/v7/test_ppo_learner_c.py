"""Learner tests (src/amini/ppo/learner.py): critic warm-up, micro-batches, policy_params, config changes, pins.

* critic warm-up: ``Learner.warmup_update`` (critic only), ``FixedSetEV`` -> ``phase_rules.critic_warmup_status``,
  the warm-up checkpoint taken into the PPO run by ``Learner.from_il(critic_from=)`` (compat digest, weights,
  frozen step encoder, pass), warm-up resume across blocks;
* micro-batches: ``policy.micro_rows`` / ``policy.max_micro_tokens``, il_v7's split rule, seat-games never split,
  the same gradient for any split, caps / counts / peak memory in the update record;
* the policy_params rule: OpponentSpec temperature is the only tempering, with synthetic checkpoint
  configs (with or without the key in AminiFormerConfig);
* recorded mid-run config changes and checkpoint pins.

Fixture: test_ppo_learner's il_v7-format source checkpoint and the synthetic four-game shards (skipped without
the private dataset). CPU only, deterministic, no game environment, no GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import S  # noqa: E402
from test_ppo_learner import (BASE_OVERRIDES, CODE, RID, STEP_OVERRIDES, UPDATE, Fixture, actor_model,  # noqa: E402
                              config, meta, seat, write_source, write_update)

from amini.ppo import buffer as B  # noqa: E402
from amini.ppo import config as C  # noqa: E402
from amini.ppo import gae as G  # noqa: E402
from amini.ppo import learner as LR  # noqa: E402
from amini.ppo import monitor as M  # noqa: E402
from amini.train import il_v7 as IL  # noqa: E402

try:
    from amini.ppo import phase_rules as PR  # noqa: E402
except ImportError:                                        # not part of this repository
    PR = None

torch.set_num_threads(2)

WARMUP_OVERRIDES = BASE_OVERRIDES + ["critic.warmup.max_updates=2", "critic.warmup.stop_trend_updates=1"]


def snap(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def cfg_with(over=BASE_OVERRIDES, **sha):
    """config(over) with opponents.checkpoint_sha256 entries set (built from the dict: a sha is no YAML scalar)."""
    d = config(over).to_dict()
    d["opponents"]["checkpoint_sha256"].update(sha)
    return C.from_dict(d)


# =========================================================================== critic warm-up
class TestWarmup(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cfg = config(WARMUP_OVERRIDES)
        cls.wroot = cls.root / "warmup"
        for u in (1, 2, 3):
            write_update(cls.wroot, cfg, policy_digest=cls.digest, update=u)
        cls.froot = cls.root / "fixed"
        write_update(cls.froot, cfg, actor_seed=0, policy_digest=cls.digest, update=1)

    def wl(self, over=WARMUP_OVERRIDES, **kw):
        return LR.Learner.from_il(config(over), self.src, **kw)

    def wdata(self, lr, u, root=None):
        return lr.load(root or self.wroot, u, expect_games=4, expect_config_digest=lr.config_digest)

    def fixed(self, lr):
        return LR.FixedSetEV(lr, [B.load_update(self.froot, 1)], expect_games=4)

    def test_warmup_trains_the_critic_only(self):
        lr = self.wl()
        pol, ref, se = snap(lr.policy), snap(lr.reference), snap(lr.critic.step_encoder)
        tv = snap(lr.critic.temporal)
        res = lr.warmup_update(self.wdata(lr, 1), self.fixed(lr))
        self.assertTrue(same(pol, snap(lr.policy)))                  # no policy step at all
        self.assertTrue(same(ref, snap(lr.reference)))
        self.assertTrue(same(se, snap(lr.critic.step_encoder)))      # frozen part untouched
        self.assertFalse(same(tv, snap(lr.critic.temporal)))         # trainable part moved
        self.assertTrue(all(p.grad is None for p in lr.policy.parameters()))
        self.assertEqual((lr.steps_total, lr.next_update, lr.warmup["done"]), (0, 1, 1))
        m = res.metrics
        self.assertGreater(m["critic_steps"], 0)
        for k in ("critic_loss_first", "critic_mse_before", "critic_mse_after", "fixed/ev_overall",
                  "fixed/ev_last_third", "ev_overall"):
            self.assertTrue(np.isfinite(m[k]), k)
        self.assertEqual((m["seat_games"], m["rows"] > 0, m["micro_rows"]), (6, True, lr.cfg.policy.micro_rows))
        self.assertNotIn("joint_k3", m)                              # no policy numbers, no KL
        self.assertFalse(res.status.stop)
        self.assertEqual(res.status.reason, "continue")
        self.assertGreater(m["peak_mem_bytes"], 0)

    def test_no_identity_or_mu_requirement(self):
        """A shard whose stored mu is wrong would stop a PPO update (actor guard); the warm-up ignores mu."""
        lr = self.wl()
        data = self.wdata(lr, 1)
        for r in data.readers:
            r.arrays["mu"] = r.arrays["mu"] + 1.0 if "mu" in r.arrays else None
        with mock.patch.object(LR.Learner, "prepare", side_effect=AssertionError("policy pre-pass used")):
            lr.warmup_update(data, self.fixed(lr))

    def test_fixed_set_ev_is_gae_explained_variance_by_phase(self):
        lr = self.wl()
        fs = self.fixed(lr)
        ev = fs()
        data = B.load_update(self.froot, 1)
        vals, sgs = [], list(data.seat_games())
        for sg in sgs:                                              # one seat-game per forward, no micro split
            b = LR._to_torch(IL_collate([sg.example()]), lr.device, torch.float64)
            with torch.no_grad():
                vals.append(lr.critic(b).double().numpy())
        lengths = np.array([sg.n for sg in sgs])
        w = np.concatenate([G.row_weights(sg.rows["stratum"], sg.rows["forced"], sg.weight,
                                          family=sg.rows["family"]).value for sg in sgs])
        want = G.explained_variance_by_phase(np.concatenate(vals), np.repeat([sg.outcome for sg in sgs], lengths),
                                             np.concatenate([sg.rows["tick"] for sg in sgs]),
                                             np.repeat([sg.game.end_tick for sg in sgs], lengths), weights=w,
                                             groups=np.repeat([sg.game.opp_kind for sg in sgs], lengths))
        for k in ("ev_overall", "ev_first_third", "ev_middle_third", "ev_last_third"):
            self.assertAlmostEqual(ev[k], want[k], places=10, msg=k)
        self.assertEqual(ev["n_games"], 4)
        self.assertEqual(ev["fixed_set_digest"], fs.digest)
        self.assertTrue(lr.critic.training)                         # mode restored
        with self.assertRaisesRegex(ValueError, f"expected {lr.cfg.critic.warmup.fixed_set_games}"):   # the default
            LR.FixedSetEV(lr, [data])

    def test_stop_rule_and_history(self):
        lr = self.wl()
        fs = self.fixed(lr)
        r1 = lr.warmup_update(self.wdata(lr, 1), fs)
        r2 = lr.warmup_update(self.wdata(lr, 2), fs)
        self.assertTrue(r2.status.stop)
        self.assertIn("reached 2", r2.status.reason)
        want = PR.critic_warmup_status(lr.cfg, lr.warmup["history"])
        self.assertEqual(r2.status, want)
        self.assertEqual(len(lr.warmup["history"]), 2)
        self.assertEqual(lr.warmup["history"][1]["ev_overall"], r2.ev_fixed["ev_overall"])
        self.assertEqual(r2.status.ev_warmup_end, r2.ev_fixed["ev_overall"])
        self.assertNotEqual(r1.ev_fixed["ev_overall"], r2.ev_fixed["ev_overall"])
        with self.assertRaisesRegex(RuntimeError, "already stopped"):
            lr.warmup_update(self.wdata(lr, 3), fs)

    def test_guards(self):
        lr = self.wl()
        fs = self.fixed(lr)
        with self.assertRaisesRegex(ValueError, "expected 1"):
            lr.warmup_update(self.wdata(lr, 2), fs)
        lr.warmup_update(self.wdata(lr, 1), fs)
        other = LR.FixedSetEV(lr, [B.load_update(self.wroot, 3)], expect_games=4)
        self.assertNotEqual(other.digest, fs.digest)               # same games / rows, another update
        with self.assertRaisesRegex(ValueError, "fixed EV set changed"):
            lr.warmup_update(self.wdata(lr, 2), other)
        with self.assertRaisesRegex(ValueError, "another learner"):
            lr.warmup_update(self.wdata(lr, 2), self.fixed(self.wl()))
        lr.steps_total = 1
        with self.assertRaisesRegex(RuntimeError, "before any policy update"):
            lr.warmup_update(self.wdata(lr, 2), fs)

    def test_warmup_resumes_in_the_next_block(self):
        over = WARMUP_OVERRIDES + ["critic.warmup.max_updates=3"]
        root = self.root / "warmup3"                                # shards under this config's digest
        for u in (1, 2):
            write_update(root, config(over), policy_digest=self.digest, update=u)
        lr = self.wl(over)
        fs = self.fixed(lr)
        lr.warmup_update(self.wdata(lr, 1, root), fs)
        p = self.root / "wu-resume.pt"
        lr.save(p)
        lr2 = LR.Learner.from_checkpoint(p, lr.cfg)
        self.assertEqual(lr2.warmup["done"], 1)
        self.assertEqual(lr2.warmup["history"], lr.warmup["history"])
        self.assertTrue(same(snap(lr2.critic), snap(lr.critic)))
        res = lr2.warmup_update(self.wdata(lr2, 2, root), self.fixed(lr2))
        self.assertEqual(res.update, 2)
        self.assertEqual(len(lr2.warmup["history"]), 2)

    def _warm_ckpt(self, name, passed=True):
        lr = self.wl()
        fs = self.fixed(lr)
        lr.warmup_update(self.wdata(lr, 1), fs)
        lr.warmup_update(self.wdata(lr, 2), fs)
        st = lr.warmup["status"]
        st["passed"] = passed            # the tiny critic's EV is not meaningful: pin the pass flag for the test
        st["ev_warmup_end"] = 0.2 if passed else st["ev_warmup_end"]
        p = self.root / name
        lr.save(p)
        return lr, p

    def test_from_il_takes_the_warmed_critic(self):
        wl, p = self._warm_ckpt("wu-pass.pt")
        lr = LR.Learner.from_il(config(STEP_OVERRIDES), self.src, critic_from=p)    # another config digest is fine
        self.assertTrue(same(snap(lr.critic), snap(wl.critic)))
        self.assertTrue(same(snap(lr.policy), snap(LR.Learner.from_il(config(), self.src).policy)))
        self.assertEqual(lr.critic_opt.state_dict()["state"].keys(), wl.critic_opt.state_dict()["state"].keys())
        self.assertGreater(len(lr.critic_opt.state_dict()["state"]), 0)
        rec = lr.critic_warmup
        self.assertEqual((rec["updates"], rec["passed"], rec["ev_warmup_end"]), (2, True, 0.2))
        self.assertEqual(rec["sha256"], IL.file_sha256(str(p)))
        self.assertEqual(rec["compat_digest"], lr.compat_digest)
        self.assertEqual(lr.source["critic_warmup"], rec)
        lr.save(self.root / "ppo-after-warmup.pt")                  # the record travels with the run
        lr2 = LR.Learner.from_checkpoint(self.root / "ppo-after-warmup.pt", config(STEP_OVERRIDES))
        self.assertEqual(lr2.critic_warmup, rec)
        # the PPO update then trains on from the warmed critic
        res = lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True)
        self.assertGreater(res.metrics["critic_steps"], 0)

    def test_from_il_refusals(self):
        _, fail = self._warm_ckpt("wu-fail.pt", passed=False)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            LR.Learner.from_il(config(), self.src, critic_from=fail)
        LR.Learner.from_il(config(), self.src, critic_from=fail, require_warmup_pass=False)
        _, ok = self._warm_ckpt("wu-ok.pt")
        ck = torch.load(ok, map_location="cpu", weights_only=False)
        bad = copy.deepcopy(ck)
        bad["ppo"]["compat_digest"] = "0" * 64
        bad["ppo"]["compat"] = dict(bad["ppo"]["compat"], hazard_dt="interval")
        torch.save(bad, self.root / "wu-compat.pt")
        with self.assertRaisesRegex(ValueError, r"compatibility digest changed.*hazard_dt"):
            LR.Learner.from_il(config(), self.src, critic_from=self.root / "wu-compat.pt")
        LR.Learner.from_il(config(), self.src, critic_from=self.root / "wu-compat.pt", allow_compat_change=True)
        bad = copy.deepcopy(ck)
        bad["ppo"]["steps_total"] = 3
        torch.save(bad, self.root / "wu-steps.pt")
        with self.assertRaisesRegex(ValueError, "not a warm-up checkpoint"):
            LR.Learner.from_il(config(), self.src, critic_from=self.root / "wu-steps.pt")
        bad = copy.deepcopy(ck)
        key = next(k for k in bad["ppo"]["critic"] if k.startswith("step_encoder."))
        bad["ppo"]["critic"][key] = bad["ppo"]["critic"][key] + 1.0
        torch.save(bad, self.root / "wu-se.pt")
        with self.assertRaisesRegex(ValueError, "frozen step encoder differs"):
            LR.Learner.from_il(config(), self.src, critic_from=self.root / "wu-se.pt")
        other = self.root / "il_other.pt"                           # another IL checkpoint (other weights)
        write_source(other, actor_model(1))
        with self.assertRaisesRegex(ValueError, "acting weights"):
            LR.Learner.from_il(config(), other, critic_from=ok)
        fresh = self.root / "no-warmup.pt"
        LR.Learner.from_il(config(), self.src).save(fresh)
        with self.assertRaisesRegex(ValueError, "no warm-up update"):
            LR.Learner.from_il(config(), self.src, critic_from=fresh)

    def test_update_gate_after_a_warmup_on_this_learner(self):
        """The warm-up gate on every path into a policy step: an unfinished or failed warm-up refuses update(); a
        passed one, or the explicit escape hatch, lets it step."""
        cfg = config(STEP_OVERRIDES)                     # max_updates > 1: one warm-up update leaves it unfinished
        lr = LR.Learner.from_il(cfg, self.src)
        lr.warmup_update(B.load_update(self.shards_step, 1),
                         LR.FixedSetEV(lr, [B.load_update(self.shards_step, 1)], expect_games=4))
        self.assertFalse(lr.warmup["status"]["stop"])
        before = snap(lr.policy)
        with self.assertRaisesRegex(RuntimeError, "warm-up has not finished"):
            lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True)
        lr.warmup["status"].update(stop=True, passed=False, reason="reached the warm-up update limit")
        with self.assertRaisesRegex(RuntimeError, "warm-up did not pass"):
            lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True)
        self.assertTrue(same(before, snap(lr.policy)))
        self.assertEqual((lr.steps_total, lr.next_update), (0, 1))
        p = self.root / "wu-gate.pt"
        lr.save(p)
        hatch = LR.Learner.from_checkpoint(p, cfg, allow_failed_warmup=True)      # experiments only
        self.assertGreater(hatch.update(self.data(hatch, self.shards_step), replay_bitwise_ok=True).steps, 0)
        lr.warmup["status"].update(passed=True)
        self.assertGreater(lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True).steps, 0)

    def test_require_warmup(self):
        with self.assertRaisesRegex(ValueError, "require_warmup"):
            LR.Learner.from_il(config(), self.src, require_warmup=True)
        _, ok = self._warm_ckpt("wu-req.pt")
        lr = LR.Learner.from_il(config(), self.src, require_warmup=True, critic_from=ok)
        self.assertTrue(lr.critic_warmup["passed"])

    def test_critic_optimiser_follows_the_config(self):
        """The warm-up's Adam moments are kept, its lr / betas / eps are not."""
        wl, p = self._warm_ckpt("wu-hp.pt")
        lr = LR.Learner.from_il(config(BASE_OVERRIDES + ["critic.lr=5.0e-5", "critic.adam_eps=1.0e-7"]), self.src,
                                critic_from=p)
        g = lr.critic_opt.param_groups[0]
        self.assertEqual((g["lr"], g["eps"], g["betas"]), (5e-5, 1e-7, tuple(lr.cfg.critic.betas)))
        st, wst = lr.critic_opt.state_dict()["state"], wl.critic_opt.state_dict()["state"]
        self.assertTrue(all(torch.equal(st[k]["exp_avg"], wst[k]["exp_avg"]) for k in wst))


def IL_collate(examples):
    from amini.nn import tokenizer_v7 as TK
    return TK.collate(examples)


# =========================================================================== micro-batches
class TestMicroBatches(Fixture):

    @staticmethod
    def il_split(sizes, toks, micro_rows, max_tokens):
        """il_v7.EpisodeBatches._split on chunks of the given row counts / token maxima."""
        eb = object.__new__(IL.EpisodeBatches)
        eb.items = [((0, i, 0, 0, n), 0, 0, n) for i, n in enumerate(sizes)]
        eb.tok = list(toks)
        eb.cfg = NS(micro_rows=micro_rows, max_batch_tokens=max_tokens)
        return eb._split(list(range(len(sizes))))

    def test_same_rule_as_il_v7_split(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(1, 40))
            sizes = [int(x) for x in rng.integers(1, 300, n)]
            toks = [(int(a), int(b)) for a, b in rng.integers(0, 120, (n, 2))]
            cap = int(rng.choice([0, 30000, 60000]))
            items = [NS(n=s, temperature=(1.0, 1.0, 1.0), tok=t, i=i) for i, (s, t) in enumerate(zip(sizes, toks))]
            ours = [[it.i for it in m] for m in LR.split_micro(items, 300, cap)]
            with self.subTest(seed=seed):
                self.assertEqual(ours, self.il_split(sizes, toks, 300, cap))

    def test_whole_seat_games_one_temperature_caps_kept(self):
        rng = np.random.default_rng(5)
        temps = [(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]
        items = [NS(n=int(rng.integers(1, 700)), temperature=temps[i % 2], tok=tuple(int(x) for x in rng.integers(0, 64, 2)),
                    i=i) for i in range(60)]
        cap = 40000
        mic = LR.split_micro(items, 512, cap)
        flat = sorted(it.i for m in mic for it in m)
        self.assertEqual(flat, list(range(60)))                      # every seat-game exactly once, never split
        for m in mic:
            self.assertEqual(len({it.temperature for it in m}), 1)
            if len(m) > 1:
                rows = sum(it.n for it in m)
                self.assertLessEqual(rows, 512)
                self.assertLessEqual(IL.micro_cost(rows, max(it.tok[0] for it in m), max(it.tok[1] for it in m)), cap)
            self.assertEqual([it.i for it in m], sorted(it.i for it in m))   # order kept inside a micro-batch
        big = [it for it in items if it.n > 512]
        self.assertTrue(big)
        for it in big:
            self.assertIn([it], mic)                                  # oversized: alone

    def test_oversized_seat_game_does_not_inflate_the_bin_count(self):
        its = [NS(n=n, temperature=t) for n, t in ((200, (1.0, 1.0, 1.0)), (150, (0.5, 0.5, 0.5)),
                                                    (90, (1.0, 1.0, 1.0)), (400, (1.0, 1.0, 1.0)),
                                                    (100, (0.5, 0.5, 0.5)))]
        self.assertEqual([[it.n for it in m] for m in LR.split_micro(its, 300)], [[200, 90], [400], [150, 100]])

    def test_config_sets_the_caps(self):
        lr = self.learner()
        self.assertEqual((lr.max_micro_rows, lr.max_micro_tokens), (256, 65536))          # example defaults
        lr = self.learner(BASE_OVERRIDES + ["policy.micro_rows=300", "policy.max_micro_tokens=0"])
        self.assertEqual((lr.max_micro_rows, lr.max_micro_tokens), (300, 0))
        self.assertEqual(self.learner(max_micro_rows=7).max_micro_rows, 7)       # timing-run override
        with self.assertRaises(C.ConfigError):
            config(BASE_OVERRIDES + ["policy.micro_rows=0"])
        with self.assertRaises(ValueError):
            self.learner(max_micro_tokens=-1)

    def test_seat_items_carry_token_maxima(self):
        lr = self.learner()
        prep = lr.prepare(self.data(lr))
        for it in prep.items:
            g = it.example["tokens"]["groups"]
            self.assertEqual(it.tok, (int(g["ENTITY"]["valid"].sum(-1).max()), int(g["OBJECT"]["valid"].sum(-1).max())))
        self.assertTrue(any(t > 0 for it in prep.items for t in it.tok))

    def test_token_cap_split_gives_the_same_gradient(self):
        """Global denominators: a token cap that forces one seat-game per micro-batch changes no gradient."""
        lr1 = self.learner(STEP_OVERRIDES)
        prep1 = lr1.prepare(self.data(lr1, self.shards_step))
        cost = min(IL.micro_cost(it.n, *it.tok) for it in prep1.items[:4])
        lr2 = self.learner(STEP_OVERRIDES, max_micro_tokens=cost)
        prep2 = lr2.prepare(self.data(lr2, self.shards_step))
        self.assertEqual(len(lr1._micro(prep1.items[:4])), 1)
        self.assertEqual(len(lr2._micro(prep2.items[:4])), 4)
        grads = []
        for lr, prep in ((lr1, prep1), (lr2, prep2)):
            st = lr.accumulate_gradients(prep.items[:4], beta=0.01, coef={"gate": 0.0, "slot": 1e-3, "cells": 1e-3,
                                                                          "ability": 1e-3})
            grads.append({n: p.grad.clone() for n, p in lr.policy.named_parameters() if p.grad is not None})
        for n in grads[0]:
            torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-9, atol=1e-13, msg=n)
        torch.testing.assert_close(torch.from_numpy(np.concatenate([it.value_old for it in prep1.items])),
                                   torch.from_numpy(np.concatenate([it.value_old for it in prep2.items])),
                                   rtol=0, atol=1e-12)

    def test_update_record_has_caps_counts_and_peak_memory(self):
        lr = self.learner(STEP_OVERRIDES + ["policy.micro_rows=400", "policy.max_micro_tokens=0"])
        data = lr.load(self.shards_step, UPDATE, expect_games=4, expect_workers=2, expect_rollout_id=RID,
                       expect_config_digest=config(STEP_OVERRIDES).digest())   # recorded under the base caps
        res = lr.update(data, replay_bitwise_ok=True)
        m = res.metrics
        self.assertEqual((m["micro_rows"], m["micro_token_cap"]), (400, 0))
        self.assertGreaterEqual(m["micro_batches"], 2)
        self.assertGreater(m["micro_tokens_max"], 0)
        self.assertEqual(m["micro_over_token_cap"], 0)
        self.assertEqual(m["peak_mem_source"], "process_peak_rss")
        self.assertGreater(m["peak_mem_bytes"], 1 << 20)
        for k in ("micro_rows", "peak_mem_bytes"):                 # the learner owns them
            with self.assertRaisesRegex(ValueError, "learner-produced"):
                self.learner(STEP_OVERRIDES)._check_extra({k: 1})
        p = self.root / "micro.pt"
        lr.save(p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.assertEqual(ck["ppo"]["micro"], {"micro_rows": 400, "micro_token_cap": 0})
        self.assertEqual(ck["metrics"]["micro_rows"], 400)

    def test_peak_memory_source(self):
        b, src = LR.peak_memory("cpu")
        self.assertEqual(src, "process_peak_rss")
        self.assertGreater(b, 0)


# =========================================================================== policy_params rule
NONDEFAULT = {"temperature": {"slot": 0.7, "cell": 0.5}}


class TestPolicyParams(Fixture):

    def test_canonical_and_neutral(self):
        for pp in (None, {}, {"temperature": 1.0}, {"temperature": {"slot": 1.0}}, {"greedy": False},
                   {"gate_stream": "shared"}, copy.deepcopy(LR.NEUTRAL_POLICY_PARAMS)):
            with self.subTest(pp=pp):
                self.assertTrue(LR.is_neutral_policy_params(pp))
        for pp in (NONDEFAULT, {"greedy": True}, {"gate_stream": "separate"}, {"temperature": 2.0}):
            with self.subTest(pp=pp):
                self.assertFalse(LR.is_neutral_policy_params(pp))
        self.assertEqual(LR.canonical_policy_params({"temperature": 0.5, "greedy": {"cell": True}}),
                         {"temperature": {"slot": 0.5, "cell": 0.5, "ability": 0.5},
                          "greedy": {"slot": False, "cell": True, "ability": False}, "gate_stream": "shared"})
        for bad in ({"tempo": 1.0}, {"temperature": [1.0, 1.0, 1.0]}, {"temperature": {"gate": 0.5}},
                    {"temperature": 0.0}, {"temperature": 101.0}, {"temperature": True}, {"greedy": 1},
                    {"gate_stream": "own"}, "x"):
            with self.subTest(bad=bad):
                with self.assertRaises(LR.PolicyParamsError):
                    LR.canonical_policy_params(bad)

    def test_opponent_temperature_conversion(self):
        self.assertEqual(LR.policy_params_temperature(None), 1.0)
        self.assertEqual(LR.policy_params_temperature({"temperature": 0.5}), 0.5)
        self.assertEqual(LR.policy_params_temperature({"temperature": {"slot": 0.25, "cell": 0.25, "ability": 0.25}}),
                         0.25)
        self.assertEqual(LR.policy_params_temperature({"greedy": True, "temperature": 0.3}), 0.0)   # greedy wins
        for bad in (NONDEFAULT, {"greedy": {"slot": True}}, {"temperature": 2.0}, {"gate_stream": "separate"},
                    {"temperature": 5e-4}):
            with self.subTest(bad=bad):
                with self.assertRaises(LR.PolicyParamsError):
                    LR.policy_params_temperature(bad)

    def test_theta_reference_critic_refuse_a_non_default_checkpoint(self):
        p = self.root / "il_pp.pt"
        write_source(p, actor_model(0), policy_params=NONDEFAULT)
        with self.assertRaisesRegex(LR.PolicyParamsError, "only tempering"):
            LR.Learner.from_il(config(), p)
        with self.assertRaises(LR.PolicyParamsError):
            LR.load_source_policy(p)
        with self.assertRaises(LR.PolicyParamsError):
            LR.ppo_config(dict(actor_model(0).cfg.to_dict(), policy_params={"greedy": True}))

    def test_neutral_or_absent_key_is_accepted_and_dropped(self):
        for i, pp in enumerate((None, {"temperature": 1.0, "gate_stream": "shared"})):
            p = self.root / f"il_neutral{i}.pt"
            write_source(p, actor_model(0), policy_params=pp)
            lr = LR.Learner.from_il(config(), p)
            self.assertEqual(lr.policy_digest(), self.digest)
            self.assertIsNone(getattr(lr.policy.cfg, "policy_params", None))
            out = self.root / f"ppo_neutral{i}.pt"
            lr.save(out)
            ck = torch.load(out, map_location="cpu", weights_only=False)
            self.assertNotIn("policy_params", ck["model_config"])
            self.assertNotIn("policy_params", ck["ppo"]["reference"]["model_config"])
            IL.load_policy(str(out))                                 # the servers' loader still reads it

    def test_resume_refuses_a_non_default_saved_config(self):
        lr = self.learner()
        out = self.root / "ppo_pp.pt"
        lr.save(out)
        ck = torch.load(out, map_location="cpu", weights_only=False)
        ck["model_config"] = dict(ck["model_config"], policy_params={"temperature": 0.5})
        torch.save(ck, out)
        with self.assertRaises(LR.PolicyParamsError):
            LR.Learner.from_checkpoint(out, lr.cfg)

    def test_built_models_with_policy_params_are_refused(self):
        lr = self.learner()
        for name in ("policy", "reference", "critic"):
            pol, ref, crit = (copy.deepcopy(lr.policy), copy.deepcopy(lr.reference), copy.deepcopy(lr.critic))
            target = {"policy": pol, "reference": ref, "critic": crit}[name]
            target.cfg = copy.copy(target.cfg)
            object.__setattr__(target.cfg, "policy_params", {"temperature": {"slot": 0.5}})   # the model's field
            with self.subTest(name=name):
                with self.assertRaisesRegex(LR.PolicyParamsError, name):
                    LR.Learner(lr.cfg, pol, ref, crit)

    def test_config_class_with_the_field(self):
        """When AminiFormerConfig has policy_params, the PPO config sets it to None explicitly."""
        @dataclasses.dataclass
        class Cfg:
            d_model: int = 8
            policy_params: object = "unset"

            @classmethod
            def from_dict(cls, d):
                return cls(**d)

        with mock.patch.object(LR, "AminiFormerConfig", Cfg):
            self.assertIsNone(LR._aminiformer_config({"d_model": 4}).policy_params)
        # AminiFormerConfig with the field: set to None (neutral); without the field: absent. Either way no
        # non-neutral policy_params reaches the PPO config.
        d = dataclasses.asdict(LR._aminiformer_config(dict(actor_model(0).cfg.to_dict())))
        self.assertIsNone(d.get("policy_params"))

    def test_frozen_opponent_conversion(self):
        mc = dict(actor_model(0).cfg.to_dict(), policy_params={"temperature": 0.5})
        out = LR.opponent_model_config(mc, 0.5)
        self.assertNotIn("policy_params", out)
        with self.assertRaisesRegex(LR.PolicyParamsError, "set opponents.temperature to 0.5"):
            LR.opponent_model_config(mc, 1.0)
        with self.assertRaises(LR.PolicyParamsError):
            LR.opponent_model_config(mc, None)
        self.assertEqual(LR.opponent_model_config(actor_model(0).cfg.to_dict(), 0.25), actor_model(0).cfg.to_dict())
        with self.assertRaises(LR.PolicyParamsError):
            LR.opponent_model_config(dict(mc, policy_params=NONDEFAULT), 0.5)
        src = self.root / "opp_pp.pt"
        write_source(src, actor_model(0), policy_params={"greedy": True})
        dst = self.root / "opp_pp_converted.pt"
        rec = LR.convert_policy_params(src, dst)
        self.assertEqual(rec["temperature"], 0.0)
        self.assertEqual(rec["sha256"], IL.file_sha256(str(dst)))
        ck = torch.load(dst, map_location="cpu", weights_only=False)
        self.assertNotIn("policy_params", ck["model_config"])
        self.assertEqual(ck["history_policy_params"][-1]["converted_to_opponent_temperature"], 0.0)
        self.assertEqual(LR.load_source_policy(dst)[1]["sha256"], rec["sha256"])
        self.assertEqual(LR.state_dict_digest(ck["model_ema"]), self.digest)      # weights untouched
        with self.assertRaises(ValueError):
            LR.convert_policy_params(src, src)
        write_source(self.root / "opp_bad.pt", actor_model(0), policy_params=NONDEFAULT)
        with self.assertRaises(LR.PolicyParamsError):
            LR.convert_policy_params(self.root / "opp_bad.pt", self.root / "opp_bad2.pt")
        self.assertFalse((self.root / "opp_bad2.pt").exists())


# =========================================================================== recorded config changes
class TestConfigChange(Fixture):

    def test_change_inside_a_block(self):
        lr = self.learner(STEP_OVERRIDES)
        old = lr.cfg
        new = C.with_changes(old, {"kl.beta": 0.04, "policy.lr": 5e-4})
        rec = lr.change_config(new, reason="pilot retry: kl flat")
        self.assertEqual((rec["old_digest"], rec["new_digest"], rec["update"]), (old.digest(), new.digest(), 1))
        self.assertEqual(rec["changes"], {"kl.beta": [old.kl.beta, 0.04], "policy.lr": [0.001, 5e-4]})
        self.assertEqual((lr.monitor.base_beta, lr.monitor.base_lr, lr.config_digest), (0.04, 5e-4, new.digest()))
        self.assertEqual((lr.monitor.beta, lr.monitor.lr), (0.04, 5e-4))
        p = self.root / "chg-in.pt"
        lr.save(p)
        lr2 = LR.Learner.from_checkpoint(p, new)
        self.assertEqual(lr2.config_changes, [rec])
        self.assertEqual((lr2.monitor.base_beta, lr2.monitor.base_lr), (0.04, 5e-4))
        with self.assertRaisesRegex(ValueError, "config digest"):
            LR.Learner.from_checkpoint(p, old)
        with self.assertRaisesRegex(C.ConfigError, "may not change mid-run"):
            lr.change_config(C.from_dict(dict(new.to_dict(), policy=dict(new.to_dict()["policy"],
                                                                         epochs=new.policy.epochs + 1))),
                             reason="x")
        with self.assertRaisesRegex(ValueError, "next update"):
            lr.change_config(C.with_changes(new, {"kl.beta": 0.03}), reason="x", update=5)

    def test_change_between_blocks_needs_the_record(self):
        lr = self.learner(STEP_OVERRIDES)
        lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True)
        p = self.root / "chg-between.pt"
        lr.save(p)
        old = lr.cfg
        new = C.with_changes(old, {"gae.lambda_tick": 0.997, "rollout.games_per_update": 2000, "critic.epochs": 6})
        with self.assertRaisesRegex(ValueError, "needs config_change"):
            LR.Learner.from_checkpoint(p, new)
        rec = C.config_change(old, new, update=2, reason="critic remedy (stall ladder)")
        lr2 = LR.Learner.from_checkpoint(p, new, config_change=rec)
        self.assertEqual(lr2.config_changes[-1], json.loads(json.dumps(rec)))
        self.assertEqual((lr2.cfg.gae.lambda_tick, lr2.cfg.critic.epochs, lr2.next_update), (0.997, 6, 2))
        self.assertEqual(lr2.policy_digest(), lr.policy_digest())
        self.assertEqual(lr2.monitor.state_dict()["early"], lr.monitor.state_dict()["early"])   # history kept
        lr2.save(self.root / "chg-between-2.pt")
        LR.Learner.from_checkpoint(self.root / "chg-between-2.pt", new)           # recorded: plain resume
        tampered = dict(rec, new_digest="0" * 64)
        with self.assertRaisesRegex(C.ConfigError, "does not match"):
            LR.Learner.from_checkpoint(p, new, config_change=tampered)
        with self.assertRaisesRegex(ValueError, "continues with update 2"):
            LR.Learner.from_checkpoint(p, new, config_change=C.config_change(old, new, update=3, reason="x"))
        with self.assertRaisesRegex(ValueError, "already equals"):
            LR.Learner.from_checkpoint(p, old, config_change=rec)
        # early_stop.* are runtime keys; monitor.clip_frac_max may not change
        bad = C.from_dict(dict(old.to_dict(), monitor=dict(old.to_dict()["monitor"], clip_frac_max=0.5)))
        with self.assertRaisesRegex(C.ConfigError, "may not change"):
            LR.Learner.from_checkpoint(p, bad, config_change={"schema": C.CONFIG_CHANGE_SCHEMA, "update": 2,
                                                              "reason": "x"})

    def test_stored_config_rebuilds_its_digest(self):
        lr = self.learner()
        p = self.root / "chg-cfg.pt"
        lr.save(p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.assertEqual(C.from_dict(ck["ppo"]["config"]).digest(), lr.config_digest)
        self.assertEqual(ck["ppo"]["config_changes"], [])


# =========================================================================== checkpoint pins
class TestPins(Fixture):

    def test_from_il_checks_the_pin(self):
        sha = IL.file_sha256(str(self.src))
        lr = LR.Learner.from_il(cfg_with(il_v4=sha), self.src, require_pin=True)
        self.assertEqual(lr.source["sha256"], sha)
        with self.assertRaisesRegex(LR.PinError, "checkpoint_sha256.il_v4"):
            LR.Learner.from_il(cfg_with(il_v4="0" * 64), self.src)
        with self.assertRaisesRegex(LR.PinError, "not set"):
            LR.Learner.from_il(config(), self.src, require_pin=True)
        p = self.root / "pinned.pt"
        lr.save(p)
        LR.Learner.from_checkpoint(p, lr.cfg)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ck["ppo"]["source"]["sha256"] = "1" * 64
        torch.save(ck, self.root / "pinned-bad.pt")
        with self.assertRaisesRegex(LR.PinError, "not the pinned base IL checkpoint"):
            LR.Learner.from_checkpoint(self.root / "pinned-bad.pt", lr.cfg)

    def test_verify_pin_and_config_pins(self):
        sha = IL.file_sha256(str(self.src))
        self.assertEqual(LR.verify_pin(self.src, sha), sha)
        self.assertEqual(LR.verify_pin(self.src, None), sha)
        with self.assertRaises(LR.PinError):
            LR.verify_pin(self.src, "f" * 64)
        d = config().to_dict()
        d["opponents"]["checkpoint"]["il_v4"] = str(self.src)
        d["opponents"]["checkpoint_sha256"]["il_v4"] = sha
        cfg = C.from_dict(d)
        self.assertEqual(LR.verify_pinned_checkpoints(cfg), {"il_v4": sha})
        d["opponents"]["checkpoint_sha256"]["il_v4"] = "e" * 64
        with self.assertRaisesRegex(LR.PinError, "opponents.checkpoint.il_v4"):
            LR.verify_pinned_checkpoints(C.from_dict(d))
        self.assertEqual(LR.verify_pinned_checkpoints(C.from_dict(d), models=("il_v3",)), {})

    def _write(self, root, opp_settings):
        cfg = config()
        cd = cfg.digest()
        table = {"ours": cfg.weights.seat_ours, "opponent": cfg.weights.seat_theta_opp,
                 "mirror": cfg.weights.seat_mirror}
        with B.ShardWriter(root, UPDATE, 0, code_digest=CODE, config_digest=cd, policy_digest=self.digest,
                           rollout_id=RID, seat_weights=table) as sw:
            for i, (kind, st) in enumerate(opp_settings):
                m = meta(f"p{i}g{i}", [seat(0, i, 0, "ours")], kind=kind, result="win", config_digest=cd)
                m.opp_settings = st
                self.assertIsNone(sw.append_game(m, [seat(0, i, 0, "ours")]))

    def test_shard_pins(self):
        root = Path(tempfile.mkdtemp(dir=self.root))
        v4, v3, snap_sha = "a" * 64, "b" * 64, "c" * 64
        self._write(root, [("il_v4_t1", {"name": "il_v4_t1", "checkpoint_sha256": v4}),
                           ("il_v3_t1", {"name": "il_v3_t1", "checkpoint_sha256": v3}),
                           ("snapshot", {"name": "snapshot", "server": "snap-u10", "checkpoint_sha256": snap_sha})])
        data = B.load_update(root, UPDATE)
        cfg = cfg_with(il_v4=v4, il_v3=v3)
        self.assertEqual(LR.verify_shard_pins(data, cfg, snapshot_sha256={"snap-u10": snap_sha}, require=True),
                         {"verified": 3, "unrecorded": 0, "unpinned": 0})
        self.assertEqual(LR.verify_shard_pins(data, config()), {"verified": 0, "unrecorded": 0, "unpinned": 3})
        with self.assertRaisesRegex(LR.PinError, "no pin"):
            LR.verify_shard_pins(data, config(), require=True)
        with self.assertRaisesRegex(LR.PinError, "il_v3_t1"):
            LR.verify_shard_pins(data, cfg_with(il_v4=v4, il_v3="d" * 64))
        with self.assertRaisesRegex(LR.PinError, "snapshot"):
            LR.verify_shard_pins(data, cfg, snapshot_sha256={"snap-u10": "d" * 64})
        # the learner's load runs the check
        lr = LR.Learner.from_il(cfg_with(il_v4=IL.file_sha256(str(self.src)), il_v3=v3), self.src)
        with self.assertRaisesRegex(LR.PinError, "il_v4_t1"):
            lr.load(root, UPDATE)
        root2 = Path(tempfile.mkdtemp(dir=self.root))
        self._write(root2, [("il_v4_t1", {"name": "il_v4_t1"})])
        lr.load(root2, UPDATE)
        self.assertEqual(lr.last_pin_check, {"verified": 0, "unrecorded": 1, "unpinned": 0})
        with self.assertRaisesRegex(LR.PinError, "no checkpoint_sha256"):
            lr.load(root2, UPDATE, require_pins=True)

    def test_opponent_pins_by_kind(self):
        cfg = cfg_with(il_v4="a" * 64, il_v2_heldout="c" * 64)
        self.assertEqual(cfg.opponent_pins(), {"il_v4_t1": "a" * 64, "il_v4_tstar": "a" * 64, "il_v3_t1": None,
                                               "snapshot": None, "il_v2_heldout": "c" * 64})
        self.assertEqual(cfg.checkpoint_pins()["il_v4"], {"path": None, "sha256": "a" * 64})


if __name__ == "__main__":
    unittest.main()
