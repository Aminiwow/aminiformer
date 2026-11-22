"""Tests for the PPO v7 learner (src/amini/ppo/learner.py).

A synthetic update is written through ``ppo.buffer`` exactly as the rollout actor writes it: per-row tokens, the
tiny fp64 actor model's ``peek`` + ``commit`` row by row, ``sample_actions``, ``buffer.actor_rows``, two worker
shards with four games (ours vs a frozen IL opponent, theta on both seats of a self_seat game, a mirror game,
ours on side 1). The learner starts from an il_v7-format checkpoint of the same weights whose model_config
carries dropout 0.1 and a gate bias (both must be dropped). CPU only, deterministic seeds, no game environment,
no GPU.

The seat-games are tokenised examples of the test shards (private dataset format, shard tokenizer): without
them every test on the :class:`Fixture` is skipped. ``config()`` reads the run file when it exists, else the
``PPOConfig`` defaults (the run file spells out the defaults).
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # tests/v7 (common, buffer fixture helpers)
import common  # noqa: E402
from common import ROOT, S, example, tiny_model  # noqa: E402
from test_ppo_buffer import _peek_input, _row_tokens, _unit_rate_dt  # noqa: E402

TK = getattr(common, "TK", None)                           # the shard tokenizer (not part of this repository)
SKIP_REASON = getattr(common, "SKIP_REASON", "needs the private dataset")

from amini.nn.aminiformer import sample_actions  # noqa: E402
from amini.ppo import buffer as B  # noqa: E402
from amini.ppo import config as C  # noqa: E402
from amini.ppo import learner as LR  # noqa: E402
from amini.ppo import losses as L  # noqa: E402
from amini.ppo import gae as G  # noqa: E402
from amini.ppo import monitor as M  # noqa: E402
from amini.train import il_v7 as IL  # noqa: E402

torch.set_num_threads(2)

MAIN_YAML = ROOT / "configs" / "ppo_v7_main.yaml"
UPDATE = 1
RID = "rollout-u1-a"
CODE = "code-" + "a" * 16
BASE_OVERRIDES = ["policy.backbone_dtype=fp32", "policy.minibatch_seat_games=2"]
STEP_OVERRIDES = BASE_OVERRIDES + ["policy.lr=0.001", "early_stop.joint_k3=10.0", "early_stop.play_k3=10.0",
                                   "early_stop.gate_kl=10.0"]
SOURCE_GATE_BIAS = [[0.0, -0.05, 0.0], [0.0, -0.1, 0.0]]       # example values


# --------------------------------------------------------------------------- fixture: actor seat-games
@lru_cache(maxsize=None)
def actor_model(seed):
    return tiny_model(seed, double=True, cell_mask=True)


@lru_cache(maxsize=None)
def actor_seat(seed, i):
    """(rows, per-row tokens) of synthetic episode i played by actor_model(seed) (as test_ppo_buffer)."""
    e, _ = example(i)
    tok = e["tokens"]
    ticks = np.asarray(tok["row"]["tick"], np.int64)
    idx = np.nonzero(np.r_[True, ticks[1:] > ticks[:-1]])[0]
    rows_tok = [_row_tokens(tok, int(t)) for t in idx]
    m = actor_model(seed)
    state = m.initial_state(1, dtype=torch.float64)
    outs = []
    for rt in rows_tok:
        tt, pos = _peek_input(rt)
        outs.append(m.peek(tt, pos, state, ticks=torch.from_numpy(rt["row"]["tick"])))
        m.commit(state, torch.tensor([True]))
    actor = B.stack_peek_outputs(outs)
    rng = np.random.default_rng(300 + i)
    n = len(idx)
    dt = (_unit_rate_dt(actor) * torch.from_numpy(rng.uniform(0.5, 2.0, n))).numpy()
    act = sample_actions(actor, torch.from_numpy(dt), generator=torch.Generator().manual_seed(40 + i))
    cards = np.where(np.asarray(act["family"]) == S.FAMILY_PLAY, 26000000 + np.asarray(act["slot"]), -1)
    rows = B.actor_rows(actor, dt, act, threat=rng.random(n) < 0.3, ticks=ticks[idx],
                        phi=rng.uniform(-0.6, 0.6, n), query_index=np.arange(n) * 2, card=cards)
    return rows, rows_tok


def seat(seed, i, side, role):
    rows, rows_tok = actor_seat(seed, i)
    n = len(rows["tick"])
    counters = {"queries": n + 2, "committed": n, "forced": int(rows["forced"].sum()), "skipped": 2}
    return B.SeatRecord(side=side, role=role, weight=G.seat_weight(role), rows={k: v.copy() for k, v in rows.items()},
                        tokens=rows_tok, counters=counters)


def meta(gid, seats, *, kind, result, our_owner=0, update=UPDATE, config_digest):
    end_tick = max(int(s.rows["tick"][-1]) for s in seats) + 7
    return B.GameMeta(game_id=gid, update=update, replica=int(gid[-1]), case_id=f"case-{gid}", level=16,
                      our_owner=our_owner, our_deck=(26000000, 26000001), opp_kind=kind,
                      opp_settings={"name": kind}, seeds={"game": 11}, tower_troops={"ours": 159000000},
                      hazard_dt={"ours": "block"}, code_digest=CODE, config_digest=config_digest, end="native",
                      result=result, end_tick=end_tick, phi_end=0.2)


def write_update(root, cfg, *, actor_seed=0, policy_digest, update=UPDATE):
    """Two worker shards, four games, six trained seat-games."""
    cd = cfg.digest()
    games = [
        [("g0", [seat(actor_seed, 0, 0, "ours")], "il_v4_t1", "win", 0),
         ("g1", [seat(actor_seed, 1, 0, "ours"), seat(actor_seed, 2, 1, "opponent")], "self_seat", -1, 0)],
        [("g2", [seat(actor_seed, 3, 0, "mirror"), seat(actor_seed, 4, 1, "mirror")], "mirror", "win", 0),
         ("g3", [seat(actor_seed, 5, 1, "ours")], "il_v3_t1", "loss", 1)],
    ]
    table = {"ours": cfg.weights.seat_ours, "opponent": cfg.weights.seat_theta_opp, "mirror": cfg.weights.seat_mirror}
    for w, gl in enumerate(games):
        with B.ShardWriter(root, update, w, code_digest=CODE, config_digest=cd, policy_digest=policy_digest,
                           rollout_id=RID, seat_weights=table) as sw:
            for gid, seats, kind, res, owner in gl:
                reason = sw.append_game(meta(f"w{w}{gid}", seats, kind=kind, result=res, our_owner=owner,
                                             update=update, config_digest=cd), seats)
                assert reason is None, reason


def write_source(path, model, **cfg_over):
    """An il_v7-format checkpoint (model_config + model_ema / model_raw) of ``model``."""
    mc = dict(model.cfg.to_dict(), **cfg_over)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"format": LR.CKPT_FORMAT, "schema": S.SCHEMA_ID, "trainer": IL.TRAINER_VERSION,
                "model_config": mc, "model_ema": sd, "model_raw": sd, "step": 5, "generation": "g-test",
                "tables": None}, path)


def config(over=BASE_OVERRIDES):
    if MAIN_YAML.is_file():
        return C.load_config(MAIN_YAML, list(over))
    return C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), list(over)))


class Fixture(unittest.TestCase):
    """Source checkpoint (seed-0 weights, dropout 0.1 + gate bias in its config) and the update-1 shards."""

    @classmethod
    def setUpClass(cls):
        if TK is None:                                      # tokenised test shards and the learner's batches
            raise unittest.SkipTest(SKIP_REASON)
        cls.tmp = tempfile.TemporaryDirectory(prefix="ppo-learner-", dir=os.environ.get("TMPDIR") or None)
        cls.root = Path(cls.tmp.name)
        cls.src = cls.root / "il_source.pt"
        write_source(cls.src, actor_model(0), dropout=0.1, gate_bias=SOURCE_GATE_BIAS)
        cls.digest = LR.state_dict_digest(actor_model(0).state_dict())
        cls.shards = cls.root / "shards"
        write_update(cls.shards, config(), policy_digest=cls.digest)
        cls.shards_step = cls.root / "shards-step"                 # same rows, config digest of STEP_OVERRIDES
        write_update(cls.shards_step, config(STEP_OVERRIDES), policy_digest=cls.digest)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def learner(self, over=BASE_OVERRIDES, **kw):
        return LR.Learner.from_il(config(over), self.src, **kw)

    def data(self, lr, root=None):
        return lr.load(root or self.shards, UPDATE, expect_games=4, expect_workers=2, expect_rollout_id=RID,
                       expect_config_digest=lr.config_digest)


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def changed(a, b, prefix=""):
    return sorted(k for k in a if k.startswith(prefix) and not torch.equal(a[k], b[k]))


# --------------------------------------------------------------------------- construction
class TestBuild(Fixture):

    def test_policy_and_reference_drop_dropout_and_gate_bias(self):
        lr = self.learner()
        self.assertEqual(lr.policy.cfg.dropout, 0.0)
        self.assertIsNone(lr.policy.cfg.gate_bias)
        self.assertIsNone(lr.reference.cfg.gate_bias)
        self.assertEqual(lr.reference.cfg.dropout, 0.0)
        self.assertEqual(lr.source["source_gate_bias"], SOURCE_GATE_BIAS)
        self.assertEqual(lr.source["source_dropout"], 0.1)
        self.assertEqual(lr.source["sha256"], IL.file_sha256(str(self.src)))
        self.assertEqual(next(lr.policy.parameters()).dtype, torch.float64)   # dtype of the stored weights
        self.assertEqual(lr.policy_digest(), self.digest)                    # the actor's weights exactly
        self.assertTrue(lr.policy.training)
        self.assertFalse(lr.reference.training)

    def test_frozen_modules_are_requires_grad_false(self):
        lr = self.learner()
        self.assertTrue(all(not p.requires_grad for p in lr.reference.parameters()))
        self.assertTrue(all(not p.requires_grad for p in lr.critic.step_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in lr.critic.temporal.parameters()))
        self.assertTrue(all(p.requires_grad for p in lr.critic.value_head.parameters()))
        unused = LR.policy_unused_prefixes()
        for name, p in lr.policy.named_parameters():
            self.assertEqual(p.requires_grad, not name.startswith(unused), name)
        opt_ids = {id(p) for g in lr.opt.param_groups for p in g["params"]}
        self.assertEqual(opt_ids, {id(p) for p in lr.policy.parameters() if p.requires_grad})
        copt_ids = {id(p) for g in lr.critic_opt.param_groups for p in g["params"]}
        self.assertEqual(copt_ids, {id(p) for m in (lr.critic.temporal, lr.critic.value_head) for p in m.parameters()})
        ref_ids = {id(p) for p in lr.reference.parameters()} | {id(p) for p in lr.critic.step_encoder.parameters()}
        self.assertFalse(opt_ids & copt_ids)
        self.assertFalse((opt_ids | copt_ids) & ref_ids)
        self.assertEqual({g["weight_decay"] for g in lr.opt.param_groups} | {lr.opt.defaults["weight_decay"]}, {0.0})
        self.assertEqual(lr.opt.defaults["betas"], tuple(lr.cfg.policy.betas))          # the config's (example values)
        self.assertEqual(lr.opt.defaults["eps"], lr.cfg.policy.adam_eps)
        self.assertEqual((lr.critic_opt.defaults["lr"], lr.critic.cfg.dropout), (lr.cfg.critic.lr, 0.0))

    def test_freeze_step_encoder_scope(self):
        lr = self.learner(BASE_OVERRIDES + ["policy.train_scope=freeze_step_encoder"])
        se = {id(p) for p in lr.policy.step_encoder.parameters()}
        self.assertTrue(all(not p.requires_grad for p in lr.policy.step_encoder.parameters()))
        opt_ids = {id(p) for g in lr.opt.param_groups for p in g["params"]}
        self.assertFalse(opt_ids & se)
        self.assertTrue(any(p.requires_grad for p in lr.policy.temporal.parameters()))

    def test_critic_starts_as_the_il_value_head(self):
        lr = self.learner()
        data = self.data(lr)
        sg = next(data.seat_games())
        b = LR._to_torch(TK.collate([sg.example()]), lr.device, torch.float64)
        with torch.no_grad():
            v_il = lr.policy(b, with_aux=False)["value"]
            v_c = lr.critic(b)
        self.assertEqual(v_c.shape, (sg.n,))
        torch.testing.assert_close(v_c, v_il, rtol=0, atol=1e-12)

    def test_compat_digest(self):
        rec = LR.compat_record(hazard_dt="block")
        self.assertEqual(set(rec["files"]), set(LR.COMPAT_FILES))
        self.assertEqual(LR.compat_digest(rec), LR.compat_digest(LR.compat_record(hazard_dt="block")))
        self.assertNotEqual(LR.compat_digest(rec), LR.compat_digest(LR.compat_record(hazard_dt="interval")))
        self.assertNotEqual(LR.compat_digest(rec), LR.compat_digest(LR.compat_record(hazard_dt="block",
                                                                                   tables_digest="x")))

    def test_micro_batches_group_by_temperature_and_rows(self):
        from types import SimpleNamespace as NS
        lr = self.learner(max_micro_rows=300)
        its = [NS(n=n, temperature=t) for n, t in ((200, (1.0, 1.0, 1.0)), (150, (0.5, 0.5, 0.5)),
                                                    (90, (1.0, 1.0, 1.0)), (400, (1.0, 1.0, 1.0)),
                                                    (100, (0.5, 0.5, 0.5)))]
        mic = lr._micro(its)
        self.assertEqual([[it.n for it in m] for m in mic], [[200, 90], [400], [150, 100]])
        self.assertTrue(all(len({it.temperature for it in m}) == 1 for m in mic))

    def test_set_reference_is_frozen_without_gate_bias(self):
        lr = self.learner()
        champ = copy.deepcopy(lr.policy)
        champ.cfg = champ.cfg.__class__.from_dict(dict(champ.cfg.to_dict(), gate_bias=SOURCE_GATE_BIAS))
        lr.set_reference(champ)
        self.assertIsNone(lr.reference.cfg.gate_bias)
        self.assertTrue(all(not p.requires_grad for p in lr.reference.parameters()))
        self.assertEqual([p.requires_grad for p in champ.parameters()],       # the champion itself is untouched
                         [p.requires_grad for p in lr.policy.parameters()])
        self.assertEqual(champ.cfg.gate_bias, SOURCE_GATE_BIAS)
        self.assertFalse(lr.reference.training)

    def test_unsupported_settings_are_refused(self):
        with self.assertRaises(NotImplementedError):
            self.learner(BASE_OVERRIDES + ["reward.shaping=short_head"])
        with self.assertRaises(C.ConfigError):                   # bf16 tolerances not set
            LR.Learner.from_il(config(["policy.minibatch_seat_games=2", "policy.backbone_dtype=bf16"]), self.src)


# --------------------------------------------------------------------------- pre-pass: GAE, weights, audit
class TestPrepare(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tc = cls("test_gae_matches_per_episode")
        cls.lr = tc.learner()
        cls.prep = cls.lr.prepare(tc.data(cls.lr), identity=True)

    def test_cached_critic_z_owns_its_memory(self):
        """``z_step`` is a view of the step encoder's token-level output, so a cached slice of it would keep that
        whole tensor alive. The cached piece must be a copy that owns exactly its own bytes."""
        for it in self.prep.items:
            z = it.critic_z
            self.assertEqual(z.untyped_storage().nbytes(), z.numel() * z.element_size())

    def test_gae_matches_per_episode(self):
        cfg = self.lr.cfg
        adv = np.concatenate([it.adv.cpu().numpy() for it in self.prep.items])
        raw = []
        for it in self.prep.items:
            r = G.gae_episode(G.episode_rewards(it.n, it.sg.outcome), it.value_old, it.sg.rows["tick"],
                              gamma_tick=cfg.gae.gamma, lambda_tick=cfg.gae.lambda_tick)
            raw.append(r.advantages)
            np.testing.assert_allclose(it.ret.cpu().numpy(), r.returns, rtol=0, atol=1e-12)
            self.assertEqual(float(it.ret[-1]), it.sg.outcome)                   # terminal: R = r
        raw = np.concatenate(raw)
        w = np.concatenate([it.w_policy.cpu().numpy() for it in self.prep.items])
        np.testing.assert_allclose(adv, (raw - self.prep.adv_mean) / (self.prep.adv_std + 1e-8), atol=1e-12)
        self.assertAlmostEqual(float((w * adv).sum() / w.sum()), 0.0, delta=1e-12)
        self.assertAlmostEqual(float(np.sqrt((w * adv ** 2).sum() / w.sum())), 1.0, delta=1e-6)

    def test_unified_weights(self):
        roles = []
        for it in self.prep.items:
            r = it.sg.rows
            wp, wv = it.w_policy.cpu().numpy(), it.w_value.cpu().numpy()
            wt = self.lr.cfg.weights                                         # the config's seat weights
            sw = {"ours": wt.seat_ours, "opponent": wt.seat_theta_opp, "mirror": wt.seat_mirror}[it.sg.role]
            ht = G.ht_weight(G.committed_stratum(r["family"], r["threat"]))
            np.testing.assert_allclose(wv, ht * sw, rtol=0, atol=0)
            np.testing.assert_allclose(wp, ht * sw * ~r["forced"], rtol=0, atol=0)
            roles.append(it.sg.role)
        self.assertEqual(sorted(roles), ["mirror", "mirror", "opponent", "ours", "ours", "ours"])
        self.assertTrue(any(it.sg.rows["forced"].any() for it in self.prep.items))

    def test_audit_and_identity_numbers(self):
        m = self.prep.metrics
        self.assertEqual(m["mask_recompute_mismatch"], 0)
        self.assertEqual(m["learner_invalid_rows"], 0)
        self.assertEqual((m["games_kept"], m["games_dropped"], m["seat_games"]), (4, 0, 6))
        self.assertEqual(m["drop_rate"], 0.0)
        for k in ("ev_overall", "ev_last_third", "critic_mse_before"):
            self.assertTrue(np.isfinite(m[k]), k)
        self.assertEqual(set(m["ev_by_opponent"]), {"il_v4_t1", "self_seat", "mirror", "il_v3_t1"})
        self.assertTrue(m["min_cell_entropy"])
        self.assertTrue(all(np.isfinite(v) and v >= 0 for v in m["min_cell_entropy"].values()))
        self.assertLess(max(self.prep.identity["logp_max_abs_diff"].values()), 1e-9)   # fp64: forward == peek
        self.assertEqual(max(self.prep.identity["kl_ref_per_head"].values()), 0.0)
        for it in self.prep.items:
            torch.testing.assert_close(it.logp_prox, it.mu, rtol=0, atol=1e-9)

    def test_microbatch_split_gives_the_same_gradient(self):
        """Global surrogate denominator: one micro-batch or one per seat-game give the same gradient."""
        lr1 = self.learner(STEP_OVERRIDES)
        lr2 = self.learner(STEP_OVERRIDES, max_micro_rows=1)
        grads = []
        for lr in (lr1, lr2):
            prep = lr.prepare(self.data(lr, self.shards_step))
            st = lr.accumulate_gradients(prep.items[:4], beta=0.01, coef={"gate": 0.0, "slot": 1e-3,
                                                                          "cells": 1e-3, "ability": 1e-3})
            self.assertTrue(st.finite)
            grads.append({n: p.grad.clone() for n, p in lr.policy.named_parameters() if p.grad is not None})
        self.assertEqual(len(lr2._micro(prep.items[:4])), 4)
        self.assertEqual(set(grads[0]), set(grads[1]))
        for n in grads[0]:
            torch.testing.assert_close(grads[0][n], grads[1][n], rtol=1e-9, atol=1e-13, msg=n)


# --------------------------------------------------------------------------- one full update
class TestUpdate(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tc = cls("test_identity_at_update_zero")
        cls.lr = tc.learner(STEP_OVERRIDES)
        cls.before = {"policy": snapshot(cls.lr.policy), "ref": snapshot(cls.lr.reference),
                      "critic": snapshot(cls.lr.critic)}
        cls.data_ = tc.data(cls.lr, cls.shards_step)
        cls.res = cls.lr.update(cls.data_, replay_bitwise_ok=True)

    def test_identity_at_update_zero(self):
        rep = self.res.identity
        self.assertIsNotNone(rep)
        self.assertEqual(rep["failures"], [])
        self.assertLess(max(rep["ratio_max_abs_dev"].values()), 1e-12)        # training-mode r == 1 (fp64)
        self.assertLess(max(rep["logp_max_abs_diff"].values()), 1e-9)
        self.assertEqual(max(rep["kl_ref_per_head"].values()), 0.0)
        self.assertEqual([a["kind"] for a in rep["monitor_actions"]], ["log"])
        self.assertTrue(all(rep["rows"][k] > 0 for k in ("wait", "play")))

    def test_one_update_changes_the_policy(self):
        r = self.res
        self.assertGreater(r.steps, 0)
        self.assertEqual(r.skipped_minibatches, 0)
        self.assertEqual(r.early_stopped, ())
        self.assertEqual(r.steps, self.lr.cfg.policy.epochs * 3)             # policy.epochs x 3 minibatches of 2
        after = snapshot(self.lr.policy)
        self.assertTrue(changed(self.before["policy"], after, "step_encoder."))
        self.assertTrue(changed(self.before["policy"], after, "temporal."))
        self.assertTrue(changed(self.before["policy"], after, "gate."))
        self.assertNotEqual(r.metrics["policy_digest_after"], r.metrics["policy_digest_before"])
        with self.assertRaises(ValueError):                                  # the shards are now off-policy
            self.lr.prepare(self.data_)

    def test_kl_to_reference_positive_and_metrics_finite(self):
        m = self.res.metrics
        for k in ("clip_frac", "play_clip_frac", "joint_k3", "play_k3", "gate_kl", "kl_ref_total", "ev_overall",
                  "ev_last_third", "ratio_mean", "critic_loss_first", "critic_loss_last", "grad_norm_max"):
            self.assertTrue(np.isfinite(m[k]), (k, m[k]))
        self.assertGreater(m["kl_ref_total"], 0.0)
        self.assertGreater(m["joint_k3"], 0.0)
        self.assertGreater(m["gate_kl"], 0.0)
        for h in L.HEADS:
            self.assertIn(f"entropy/{h}", m)
            self.assertIn(f"kl_grad_ratio/{h}", m)
        # KL(theta_1 || ref) of the updated policy on fresh shards of the same rows
        lr2 = copy.deepcopy(self.lr)
        lr2.next_update = UPDATE
        d = lr2.policy_digest()
        root = Path(self.tmp.name) / "shards-post"
        write_update(root, lr2.cfg, policy_digest=d)
        prep = lr2.prepare(lr2.load(root, UPDATE))
        kl = lr2.kl_to_reference(prep)
        self.assertTrue(all(np.isfinite(v) for v in kl.values()))
        self.assertGreater(kl["total"], 0.0)
        self.assertGreater(kl["family"], 0.0)

    def test_frozen_parts_unchanged_trained_parts_changed(self):
        self.assertEqual(changed(self.before["ref"], snapshot(self.lr.reference)), [])
        crit = snapshot(self.lr.critic)
        self.assertEqual(changed(self.before["critic"], crit, "step_encoder."), [])
        self.assertTrue(changed(self.before["critic"], crit, "temporal."))
        self.assertTrue(changed(self.before["critic"], crit, "value_head."))
        pol = snapshot(self.lr.policy)
        for pre in LR.policy_unused_prefixes():
            self.assertEqual(changed(self.before["policy"], pol, pre), [], pre)
        self.assertEqual(self.res.metrics["critic_steps"], self.lr.cfg.critic.epochs * 3)
        self.assertLess(self.res.metrics["critic_mse_after"], self.res.metrics["critic_mse_before"])

    def test_monitor_record_and_state(self):
        m = self.res.metrics
        for k in LR.LEARNER_KEYS:
            self.assertIn(k, m)
        self.assertEqual(m["update"], UPDATE)
        self.assertEqual(self.lr.next_update, UPDATE + 1)
        self.assertEqual(self.lr.monitor.last_update, UPDATE)
        self.assertFalse(self.res.stopped)
        self.assertNotIn("clip_frac", self.lr.monitor.last_missing)
        self.assertNotIn("kl_ref_total", self.lr.monitor.last_missing)
        self.assertNotIn("ev_overall", self.lr.monitor.last_missing)
        self.assertIn("refusals_code19", self.lr.monitor.last_missing)       # the rollout's key, not the learner's
        self.assertEqual(self.lr.clipped_frac_history, [m["grad_clipped_frac"]])
        self.assertEqual(self.lr.steps_total, self.res.steps)


# --------------------------------------------------------------------------- checkpoints
class TestCheckpoint(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tc = cls("test_round_trip_through_il_load_policy")
        cls.lr = tc.learner(STEP_OVERRIDES)
        cls.lr.update(tc.data(cls.lr, cls.shards_step), replay_bitwise_ok=True)
        cls.path = cls.lr.save(cls.root / "ppo_u000001.pt", monitor_path=cls.root / "monitor.json")

    def test_round_trip_through_il_load_policy(self):
        m = IL.load_policy(str(self.path))
        self.assertEqual(m.cfg.dropout, 0.0)
        self.assertIsNone(m.cfg.gate_bias)
        sd, ref = m.state_dict(), self.lr.policy.state_dict()
        self.assertEqual(set(sd), set(ref))
        for k in sd:
            self.assertTrue(torch.equal(sd[k], ref[k].to(sd[k].dtype)), k)
        ck = torch.load(str(self.path), map_location="cpu", weights_only=False)
        self.assertEqual(LR.state_dict_digest(ck["model_ema"]), self.lr.policy_digest())
        self.assertEqual(ck["ppo"]["policy_digest"], self.lr.policy_digest())
        self.assertTrue((self.root / "monitor.json").is_file())
        self.assertFalse((self.root / "ppo_u000001.pt.tmp").exists())

    def test_resume_restores_everything(self):
        r = LR.Learner.from_checkpoint(self.path, self.lr.cfg)
        self.assertEqual((r.next_update, r.steps_total), (self.lr.next_update, self.lr.steps_total))
        self.assertEqual(r.policy_digest(), self.lr.policy_digest())
        self.assertEqual(r.reference_digest(), self.lr.reference_digest())
        self.assertEqual(changed(snapshot(r.critic), snapshot(self.lr.critic)), [])
        self.assertEqual(r.monitor.state_dict(), self.lr.monitor.state_dict())
        self.assertEqual(r.clipped_frac_history, self.lr.clipped_frac_history)
        a, b = r.opt.state_dict(), self.lr.opt.state_dict()
        self.assertEqual(set(a["state"]), set(b["state"]))
        for k in a["state"]:
            self.assertTrue(torch.equal(a["state"][k]["exp_avg"], b["state"][k]["exp_avg"]))
        ca, cb = r.critic_opt.state_dict(), self.lr.critic_opt.state_dict()
        for k in ca["state"]:
            self.assertTrue(torch.equal(ca["state"][k]["exp_avg_sq"], cb["state"][k]["exp_avg_sq"]))
        self.assertTrue(all(not p.requires_grad for p in r.reference.parameters()))
        self.assertIsNone(r.reference.cfg.gate_bias)

    def test_compat_and_config_changes_are_refused(self):
        ck = torch.load(str(self.path), map_location="cpu", weights_only=False)
        ck["ppo"]["compat_digest"] = "0" * 64
        bad = self.root / "bad_compat.pt"
        torch.save(ck, bad)
        with self.assertRaisesRegex(ValueError, "compatibility"):
            LR.Learner.from_checkpoint(bad, self.lr.cfg)
        LR.Learner.from_checkpoint(bad, self.lr.cfg, allow_compat_change=True)
        with self.assertRaisesRegex(ValueError, "config digest"):
            LR.Learner.from_checkpoint(self.path, config(STEP_OVERRIDES + [
                f"policy.epochs={self.lr.cfg.policy.epochs + 1}"]))
        with self.assertRaises(ValueError):                                 # an IL checkpoint is not a PPO one
            LR.Learner.from_checkpoint(self.src, self.lr.cfg)


# --------------------------------------------------------------------------- guards
class TestGuards(Fixture):

    def test_identity_failure_raises_before_any_step(self):
        """Shards played by other weights but stamped with this policy's digest: update 0 must stop."""
        root = self.root / "shards-wrong-actor"
        write_update(root, config(), actor_seed=1, policy_digest=self.digest)
        lr = self.learner()
        before = snapshot(lr.policy)
        with self.assertRaises(LR.IdentityError):
            lr.update(self.data(lr, root), replay_bitwise_ok=True)
        self.assertEqual(changed(before, snapshot(lr.policy)), [])
        self.assertTrue(lr.monitor.stopped)
        self.assertEqual(lr.steps_total, 0)
        with self.assertRaises(RuntimeError):                               # a stopped run never continues
            lr.update(self.data(lr), replay_bitwise_ok=True)

    def test_identity_needs_the_replay_flag(self):
        lr = self.learner()
        with self.assertRaises(LR.IdentityError):
            lr.update(self.data(lr), replay_bitwise_ok=None)

    def test_off_policy_shards_are_refused(self):
        root = self.root / "shards-other-digest"
        write_update(root, config(), policy_digest="f" * 64)
        lr = self.learner()
        with self.assertRaises(B.SchemaError):
            self.data(lr, root)
        with self.assertRaises(ValueError):
            lr.prepare(B.load_update(root, UPDATE))

    def test_wrong_update_index_is_refused(self):
        lr = self.learner()
        lr.next_update = 5
        with self.assertRaisesRegex(ValueError, "expects update 5"):
            lr.update(self.data(lr), replay_bitwise_ok=True)

    def test_nonfinite_minibatches_are_skipped_then_stop(self):
        lr = self.learner()
        before = snapshot(lr.policy)
        real = L.clipped_surrogate
        with mock.patch.object(L, "clipped_surrogate", side_effect=lambda *a, **k: real(*a, **k) * float("nan")):
            res = lr.update(self.data(lr), replay_bitwise_ok=True)
        self.assertEqual(res.steps, 0)
        self.assertEqual(res.skipped_minibatches, lr.cfg.monitor.nan_skip_consecutive_stop)   # nonfinite_stop
        self.assertTrue(res.stopped)
        self.assertEqual(changed(before, snapshot(lr.policy)), [])
        self.assertIn("non-finite", lr.monitor.stop_reason)

    def test_strict_monitor_refuses_before_stepping(self):
        cfg = config()
        mon = M.Monitor(thresholds=M.Thresholds.from_mapping(cfg.monitor_thresholds()), strict=True)
        lr = LR.Learner.from_il(cfg, self.src, monitor=mon)
        before = snapshot(lr.policy)
        with self.assertRaises(M.MetricsMissingError) as cm:
            lr.update(self.data(lr), replay_bitwise_ok=True)
        self.assertIn("refusals_code19", str(cm.exception))
        self.assertEqual(changed(before, snapshot(lr.policy)), [])
        self.assertEqual(lr.next_update, UPDATE)

    def test_fp32_learner_passes_the_fp32_identity(self):
        """A float32 learner on the fp64 actor's mu (the deployment-like case): identity within 1e-4."""
        cfg = config(STEP_OVERRIDES)
        lr = LR.Learner.from_il(cfg, self.src, dtype=torch.float32)
        self.assertEqual(next(lr.policy.parameters()).dtype, torch.float32)
        d = lr.policy_digest()
        root = self.root / "shards-fp32"
        write_update(root, cfg, policy_digest=d)
        res = lr.update(self.data(lr, root), replay_bitwise_ok=True)
        rep = res.identity
        self.assertEqual(rep["failures"], [])
        self.assertLess(max(rep["logp_max_abs_diff"].values()), 1e-4)
        self.assertLess(max(rep["ratio_max_abs_dev"].values()), 1e-4)
        self.assertGreater(res.steps, 0)
        print(f"fp32 identity: max|dlogp| {rep['logp_max_abs_diff']}, max|r-1| {rep['ratio_max_abs_dev']}")

    def test_bf16_autocast_path_runs(self):
        """bf16 backbone (CPU autocast here): finite numbers; the thresholds themselves are set per run."""
        over = STEP_OVERRIDES[1:] + ["policy.backbone_dtype=bf16", "monitor.parity.dlogp_tol_bf16.wait=0.5",
                                     "monitor.parity.dlogp_tol_bf16.play=0.5",
                                     "monitor.parity.dlogp_tol_bf16.ability=0.5"]
        cfg = config(over)
        lr = LR.Learner.from_il(cfg, self.src, dtype=torch.float32)
        self.assertEqual(lr.precision, "bf16")
        root = self.root / "shards-bf16"
        write_update(root, cfg, policy_digest=lr.policy_digest())
        prep = lr.prepare(self.data(lr, root), identity=True)
        st = lr.accumulate_gradients(prep.items[:2], beta=0.01)
        self.assertTrue(st.finite)
        d = prep.identity["logp_max_abs_diff"]
        self.assertTrue(all(np.isfinite(v) for v in d.values()))
        print(f"bf16 (CPU autocast, tiny model) max|dlogp| by row type: {d}")


if __name__ == "__main__":
    unittest.main()
