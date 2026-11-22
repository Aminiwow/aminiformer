"""Adversarial tests for the PPO v7 learner.

Covered: gate_bias None, π_prox recomputed (lag 0, unified weights, global D), frozen means requires_grad False,
the separate critic, the sampling temperature, the update-0 identity, early stop, the strict monitor, model_ema
and the compat digest.

``test_BUG_*`` pin defects found in review (they were ``expectedFailure`` until the fixes; they must pass now).
The other tests check conventions the main tests leave open and must pass.
Fixture: tests/v7/test_ppo_learner.py's, tiny fp64 model, CPU only, deterministic seeds, no game environment,
no GPU (skipped without the private dataset).
"""

from __future__ import annotations

import copy
import dataclasses
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import S, example  # noqa: E402
from test_ppo_buffer import _peek_input, _row_tokens, _unit_rate_dt  # noqa: E402
from test_ppo_learner import (CODE, RID, STEP_OVERRIDES, UPDATE, Fixture, actor_model, changed,  # noqa: E402
                              config, meta, snapshot, write_source, write_update)

from amini.nn.aminiformer import sample_actions  # noqa: E402
from amini.ppo import buffer as B  # noqa: E402
from amini.ppo import gae as G  # noqa: E402
from amini.ppo import learner as LR  # noqa: E402
from amini.ppo import logprob as LP  # noqa: E402
from amini.ppo import monitor as M  # noqa: E402

torch.set_num_threads(2)

BIG_GATE_BIAS = [[0.0, -1.0, 0.0], [0.0, -1.5, 0.0]]


# --------------------------------------------------------------------------- helpers
def custom_seat(model, i, *, temperature=(1.0, 1.0, 1.0), threat_to_peek=False):
    """(rows, per-row tokens) of synthetic episode i acted by ``model`` at ``temperature``; with
    ``threat_to_peek`` the row's threat goes into ``peek`` (so a cfg.gate_bias is applied, as the server does)."""
    e, _ = example(i)
    tok = e["tokens"]
    ticks = np.asarray(tok["row"]["tick"], np.int64)
    idx = np.nonzero(np.r_[True, ticks[1:] > ticks[:-1]])[0]
    rows_tok = [_row_tokens(tok, int(t)) for t in idx]
    n = len(idx)
    rng = np.random.default_rng(700 + i)
    threat = rng.random(n) < 0.3
    state = model.initial_state(1, dtype=torch.float64)
    outs = []
    for k, rt in enumerate(rows_tok):
        tt, pos = _peek_input(rt)
        if threat_to_peek:
            tt["row"]["threat"] = torch.tensor([bool(threat[k])])
        outs.append(model.peek(tt, pos, state, ticks=torch.from_numpy(rt["row"]["tick"])))
        model.commit(state, torch.tensor([True]))
    actor = LP.apply_temperature(B.stack_peek_outputs(outs), *temperature)
    dt = (_unit_rate_dt(actor) * torch.from_numpy(rng.uniform(0.5, 2.0, n))).numpy()
    act = sample_actions(actor, torch.from_numpy(dt), generator=torch.Generator().manual_seed(90 + i))
    cards = np.where(np.asarray(act["family"]) == S.FAMILY_PLAY, 26000000 + np.asarray(act["slot"]), -1)
    rows = B.actor_rows(actor, dt, act, threat=threat, ticks=ticks[idx], phi=rng.uniform(-0.6, 0.6, n),
                        query_index=np.arange(n) * 2, card=cards)
    return rows, rows_tok


def record(rows, rows_tok, side, role, temperature=(1.0, 1.0, 1.0)):
    n = len(rows["tick"])
    counters = {"queries": n + 2, "committed": n, "forced": int(rows["forced"].sum()), "skipped": 2}
    return B.SeatRecord(side=side, role=role, weight=G.seat_weight(role), rows={k: v.copy() for k, v in rows.items()},
                        tokens=rows_tok, counters=counters, temperature=tuple(temperature))


def write_games(root, cfg, games, *, policy_digest, update=UPDATE):
    """``games``: per worker, a list of (gid, [SeatRecord], opp_kind, result, our_owner)."""
    cd = cfg.digest()
    table = {"ours": cfg.weights.seat_ours, "opponent": cfg.weights.seat_theta_opp, "mirror": cfg.weights.seat_mirror}
    for w, gl in enumerate(games):
        with B.ShardWriter(root, update, w, code_digest=CODE, config_digest=cd, policy_digest=policy_digest,
                           rollout_id=RID, seat_weights=table) as sw:
            for gid, seats, kind, res, owner in gl:
                reason = sw.append_game(meta(f"w{w}{gid}", seats, kind=kind, result=res, our_owner=owner,
                                             update=update, config_digest=cd), seats)
                assert reason is None, reason


def write_fresh(root, lr, update):
    """Update ``update`` acted by the learner's CURRENT weights (genuinely on-policy): 4 games, 2 workers."""
    m = copy.deepcopy(lr.policy).eval()
    seats = [record(*custom_seat(m, i), 1 if i == 3 else 0, "ours") for i in range(4)]
    write_games(root, lr.cfg, [[("g0", [seats[0]], "il_v4_t1", "win", 0), ("g1", [seats[1]], "il_v3_t1", "loss", 0)],
                               [("g2", [seats[2]], "il_v4_t1", "loss", 0), ("g3", [seats[3]], "il_v3_t1", "win", 1)]],
                policy_digest=lr.policy_digest(), update=update)


def load(lr, root, update=UPDATE, games=4, workers=2):
    return lr.load(root, update, expect_games=games, expect_workers=workers, expect_rollout_id=RID,
                   expect_config_digest=lr.config_digest)


def mu_gap(prep):
    """max |logp_prox - mu| over every row of a prepared update (what the identity measures at update 0)."""
    return max(float((it.logp_prox - it.mu).abs().max()) for it in prep.items)


# --------------------------------------------------------------------------- lag 0 after update 0
class TestActorLearnerAgreementAfterUpdate0(Fixture):
    """After the first step the learner never compares the actor's mu with its own pi_prox (pi_prox is
    recomputed; the identity runs only while steps_total == 0). A stale server (not restarted with theta_k)
    or a served gate bias, stamped with the right digest by the coordinator, is then trained on as if
    on-policy."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tc = cls("test_DOC_stale_server_rows_differ_beyond_the_identity_tolerance")
        cls.lr = tc.learner(STEP_OVERRIDES)
        cls.res1 = cls.lr.update(tc.data(cls.lr, cls.shards_step), replay_bitwise_ok=True)
        cls.u2 = cls.root / "shards-u2-stale"                 # theta_0's rows, stamped with theta_1's digest
        write_update(cls.u2, config(STEP_OVERRIDES), policy_digest=cls.lr.policy_digest(), update=UPDATE + 1)
        cls.u2_fresh = cls.root / "shards-u2-fresh"           # rows acted by theta_1 itself
        write_fresh(cls.u2_fresh, cls.lr, UPDATE + 1)

    def test_on_policy_rows_after_a_step_still_meet_the_identity_tolerance(self):
        """So a per-update mu-vs-pi_prox guard at the update-0 tolerance would not false-alarm on correct data."""
        lr = copy.deepcopy(self.lr)
        prep = lr.prepare(load(lr, self.u2_fresh, UPDATE + 1), identity=True)
        self.assertLess(mu_gap(prep), 1e-9)
        self.assertEqual(prep.metrics["learner_invalid_rows"], 0)

    def test_stale_server_update_stops_the_monitor_with_no_step(self):
        lr = copy.deepcopy(self.lr)
        before = snapshot(lr.policy)
        with self.assertRaises(LR.IdentityError) as cm:
            lr.update(load(lr, self.u2, UPDATE + 1))
        self.assertIn("logp/", str(cm.exception))
        self.assertTrue(lr.monitor.stopped)
        self.assertIn("actor_learner_mismatch", lr.monitor.stop_reason)
        self.assertEqual(changed(before, snapshot(lr.policy)), [])
        self.assertEqual(lr.steps_total, self.lr.steps_total)

    def test_on_policy_update_after_a_step_trains_and_reports_the_gap(self):
        lr = copy.deepcopy(self.lr)
        res = lr.update(load(lr, self.u2_fresh, UPDATE + 1))
        self.assertGreater(res.steps, 0)
        self.assertIsNone(res.identity)
        for rt in ("wait", "play", "ability"):
            self.assertLess(res.metrics[f"mu_dlogp_max/{rt}"], 1e-9, rt)

    def test_DOC_stale_server_rows_differ_beyond_the_identity_tolerance(self):
        lr = copy.deepcopy(self.lr)
        prep = lr.prepare(load(lr, self.u2, UPDATE + 1))
        gap = mu_gap(prep)
        print(f"stale-server update 2: max |logp_prox - mu| = {gap:.3e}")
        self.assertGreater(gap, 1e-3)

    def test_BUG_mismatched_mu_must_not_be_trained_on_silently(self):
        """Desired: refuse (IdentityError / ValueError) or stop with no step when mu and pi_prox disagree beyond
        the per-row-type tolerance on any update, not only update 0."""
        lr = copy.deepcopy(self.lr)
        try:
            res = lr.update(load(lr, self.u2, UPDATE + 1))
        except (LR.IdentityError, ValueError):
            return
        self.assertTrue(res.stopped and res.steps == 0, (res.steps, res.stopped))


# --------------------------------------------------------------------------- strict monitor
class TestStrictMonitorAfterSteps(Fixture):

    def test_BUG_strict_nan_metric_raises_only_after_the_steps(self):
        """All games won: ev_overall / ev_last_third are NaN (constant outcome; known in prepare, before any
        step). The preflight checks key presence only, so the strict monitor raises inside observe_update after
        the optimiser steps and critic epochs: weights changed, next_update not advanced."""
        cfg = config(STEP_OVERRIDES)
        m = actor_model(0)
        seats = [record(*custom_seat(m, i), 0, "ours") for i in range(3)] + [record(*custom_seat(m, 3), 1, "ours")]
        root = self.root / "shards-all-win"
        write_games(root, cfg, [[("g0", [seats[0]], "il_v4_t1", "win", 0), ("g1", [seats[1]], "il_v3_t1", "win", 0)],
                                [("g2", [seats[2]], "il_v4_t1", "win", 0), ("g3", [seats[3]], "il_v3_t1", "win", 1)]],
                    policy_digest=self.digest)
        mon = M.Monitor(thresholds=M.Thresholds.from_mapping(cfg.monitor_thresholds()), base_lr=cfg.policy.lr,
                        base_beta=cfg.kl.beta, strict=True)
        lr = LR.Learner.from_il(cfg, self.src, monitor=mon)
        data = load(lr, root)
        self.assertTrue(np.isnan(lr.prepare(data).metrics["ev_overall"]))
        before = snapshot(lr.policy)
        try:
            lr.update(data, replay_bitwise_ok=True, extra_metrics={"refusals_code19": 0})
        except M.MetricsMissingError as e:
            self.assertIn("ev_overall", str(e))
            self.assertEqual(changed(before, snapshot(lr.policy)), [], "stepped before the strict refusal")
            self.assertEqual(lr.next_update, UPDATE)


# --------------------------------------------------------------------------- audit and overrides
class TestAuditStops(Fixture):

    def test_mask_audit_failure_stops_the_monitor(self):
        lr = self.learner()
        with mock.patch.object(B, "mask_recompute_mismatch", return_value=1):
            res = lr.update(self.data(lr), replay_bitwise_ok=True)
        self.assertGreaterEqual(res.metrics["mask_recompute_mismatch"], 6)
        self.assertTrue(res.stopped)
        self.assertIn("mismatch", lr.monitor.stop_reason)
        self.assertEqual(res.steps, 0)                    # known in prepare: stopped before any step
        self.assertEqual(lr.steps_total, 0)
        self.assertTrue(res.metrics["prestep_stop"])

    def test_BUG_extra_metrics_can_hide_a_failed_mask_audit(self):
        """extra_metrics override the learner's own values, including the audit it computed (a stop)."""
        lr = self.learner()
        with mock.patch.object(B, "mask_recompute_mismatch", return_value=1):
            try:
                res = lr.update(self.data(lr), replay_bitwise_ok=True, extra_metrics={"mask_recompute_mismatch": 0})
            except ValueError:
                return
        self.assertTrue(res.stopped)


# --------------------------------------------------------------------------- one update: invariants
class TestOneUpdateInvariants(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tc = cls("test_frozen_parameters_have_no_grad_and_no_adam_state")
        cls.lr = tc.learner(STEP_OVERRIDES)
        cls.data_ = tc.data(cls.lr, cls.shards_step)
        cls.res = cls.lr.update(cls.data_, replay_bitwise_ok=True)

    def test_frozen_parameters_have_no_grad_and_no_adam_state(self):
        lr = self.lr
        self.assertGreater(self.res.steps, 0)
        n_frozen = n_state = 0
        for name, p in lr.policy.named_parameters():
            if p.requires_grad:
                if p in lr.opt.state:
                    n_state += 1
                    self.assertLessEqual(int(lr.opt.state[p]["step"]), self.res.steps, name)
            else:
                n_frozen += 1
                self.assertIsNone(p.grad, name)
                self.assertNotIn(p, lr.opt.state, name)
        self.assertGreater(n_frozen, 0)
        self.assertGreater(n_state, 0)
        self.assertTrue(all(p.grad is None for p in lr.reference.parameters()))
        for p in lr.critic.step_encoder.parameters():
            self.assertIsNone(p.grad)
            self.assertNotIn(p, lr.critic_opt.state)
        self.assertFalse(lr.critic.step_encoder.training)

    def test_policy_critic_reference_share_no_storage(self):
        ptr = lambda mod: {p.data_ptr() for p in mod.parameters()}   # noqa: E731
        pol, ref, crit = ptr(self.lr.policy), ptr(self.lr.reference), ptr(self.lr.critic)
        self.assertFalse(pol & ref)
        self.assertFalse(pol & crit)
        self.assertFalse(ref & crit)

    def test_BUG_forced_fraction_uses_the_rollout_definition(self):
        """The rollout's definition: (forced + skipped) / queries. The learner used forced / queries, and the
        strict preflight then counted the key as produced."""
        ours = self.data_.counters["by_role"]["ours"]
        want = (ours["forced"] + ours["skipped"]) / ours["queries"]
        self.assertAlmostEqual(self.res.metrics["forced_fraction"], want, delta=1e-12)


class TestLossAndCritic(Fixture):

    def test_loss_at_theta0_is_minus_weighted_mean_advantage_over_the_minibatch_D(self):
        lr = self.learner(STEP_OVERRIDES)
        prep = lr.prepare(self.data(lr, self.shards_step))
        mb = prep.items[:3]
        st = lr.accumulate_gradients(mb, beta=0.01)
        D = sum(float(it.w_policy.sum()) for it in mb)
        self.assertAlmostEqual(st.denom, D, delta=1e-9)
        self.assertLess(D, sum(it.wp_sum for it in prep.items))          # minibatch-wide, not update-wide
        want = -sum(float((it.w_policy * it.adv).sum()) for it in mb) / D
        self.assertAlmostEqual(st.loss, want, delta=1e-10)               # r = 1, KL(theta0 || ref) = 0, entropy 0
        self.assertLess(abs(st.kl_sum), 1e-12)

    def test_critic_training_touches_only_the_critic(self):
        lr = self.learner(STEP_OVERRIDES)
        prep = lr.prepare(self.data(lr, self.shards_step))
        pol, ref, crit = snapshot(lr.policy), snapshot(lr.reference), snapshot(lr.critic)
        out = lr.train_critic(prep)
        self.assertEqual(out["critic_steps"], lr.cfg.critic.epochs * 3)
        self.assertEqual(changed(pol, snapshot(lr.policy)), [])
        self.assertTrue(all(p.grad is None for p in lr.policy.parameters()))
        self.assertEqual(changed(ref, snapshot(lr.reference)), [])
        c2 = snapshot(lr.critic)
        self.assertEqual(changed(crit, c2, "step_encoder."), [])
        self.assertTrue(changed(crit, c2, "temporal."))
        self.assertTrue(changed(crit, c2, "value_head."))
        self.assertLess(out["critic_mse_after"], prep.metrics["critic_mse_before"])

    def test_freeze_step_encoder_arm_keeps_the_step_encoder_bitwise(self):
        over = STEP_OVERRIDES + ["policy.train_scope=freeze_step_encoder"]
        root = self.root / "shards-fse"
        write_update(root, config(over), policy_digest=self.digest)
        lr = self.learner(over)
        before = snapshot(lr.policy)
        res = lr.update(load(lr, root), replay_bitwise_ok=True)
        self.assertGreater(res.steps, 0)
        after = snapshot(lr.policy)
        self.assertEqual(changed(before, after, "step_encoder."), [])
        self.assertTrue(changed(before, after, "temporal."))
        for p in lr.policy.step_encoder.parameters():
            self.assertFalse(p.requires_grad)
            self.assertNotIn(p, lr.opt.state)


# --------------------------------------------------------------------------- early stop order
class TestEarlyStopOrder(Fixture):

    def test_BUG_early_stop_takes_the_step_that_tripped(self):
        """Spinning Up checks the minibatch's approx-KL BEFORE its optimiser step and skips that step. The learner
        steps first and checks after, so a trip on minibatch j still applies minibatch j's step. With a tiny
        joint_k3 the first minibatch (r = 1) passes and the second trips: 1 step expected, 2 taken."""
        over = STEP_OVERRIDES + ["early_stop.joint_k3=1e-12"]
        root = self.root / "shards-es"
        write_update(root, config(over), policy_digest=self.digest)
        lr = self.learner(over)
        res = lr.update(load(lr, root), replay_bitwise_ok=True)
        self.assertIn("joint_k3", res.early_stopped)
        self.assertEqual(res.steps, 1)


# --------------------------------------------------------------------------- construction and checkpoints
class TestConstructionAndCheckpoint(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tc = cls("test_resume_continues_bitwise")
        cls.lr = tc.learner(STEP_OVERRIDES)
        cls.lr.update(tc.data(cls.lr, cls.shards_step), replay_bitwise_ok=True)
        cls.path = cls.lr.save(cls.root / "ppo_u1.pt")
        cls.u2 = cls.root / "shards-u2"                       # acted by theta_1 (on-policy for update 2)
        write_fresh(cls.u2, cls.lr, UPDATE + 1)

    def test_BUG_passed_monitor_must_match_the_config(self):
        """A monitor built without base_lr / base_beta silently replaces cfg.policy.lr (1e-3 here) by the monitor's
        default base_lr while the config digest still says 1e-3."""
        cfg = config(STEP_OVERRIDES)
        mon = M.Monitor(thresholds=M.Thresholds.from_mapping(cfg.monitor_thresholds()))
        try:
            lr = LR.Learner.from_il(cfg, self.src, monitor=mon)
        except ValueError:
            return
        self.assertEqual((lr.monitor.lr, lr.monitor.beta), (cfg.policy.lr, cfg.kl.beta))

    def test_from_il_reads_model_ema_not_model_raw(self):
        src = self.root / "ema_vs_raw.pt"
        write_source(src, actor_model(0))
        ck = torch.load(str(src), map_location="cpu", weights_only=False)
        ck["model_raw"] = {k: v.detach().clone() for k, v in actor_model(1).state_dict().items()}
        torch.save(ck, src)
        lr = LR.Learner.from_il(config(), src)
        self.assertEqual(lr.policy_digest(), self.digest)

    def test_BUG_from_checkpoint_checks_the_acting_weights(self):
        """The servers act with model_ema; from_checkpoint restores the policy from model_raw and never compares
        it with model_ema or ppo.policy_digest."""
        ck = torch.load(str(self.path), map_location="cpu", weights_only=False)
        raw = {k: v.clone() for k, v in ck["model_raw"].items()}
        k0 = next(k for k, v in raw.items() if k.startswith("temporal.") and v.is_floating_point())
        raw[k0] = raw[k0] + 1e-3
        ck["model_raw"] = raw
        bad = self.root / "raw_tampered.pt"
        torch.save(ck, bad)
        try:
            r = LR.Learner.from_checkpoint(bad, self.lr.cfg)
        except ValueError:
            return
        self.assertEqual(r.policy_digest(), ck["ppo"]["policy_digest"])

    def test_resume_continues_bitwise(self):
        """An uninterrupted learner and one restored from the barrier checkpoint give the same update 2."""
        a = copy.deepcopy(self.lr)
        b = LR.Learner.from_checkpoint(self.path, self.lr.cfg)
        ra = a.update(load(a, self.u2, UPDATE + 1))
        rb = b.update(load(b, self.u2, UPDATE + 1))
        self.assertGreater(ra.steps, 0)
        self.assertEqual(ra.steps, rb.steps)
        self.assertEqual(a.policy_digest(), b.policy_digest())
        self.assertEqual(changed(snapshot(a.critic), snapshot(b.critic)), [])
        self.assertEqual(a.monitor.state_dict(), b.monitor.state_dict())
        self.assertEqual(ra.metrics["kl_ref_total"], rb.metrics["kl_ref_total"])


# --------------------------------------------------------------------------- conventions end to end
class TestConventions(Fixture):

    def test_identity_catches_a_gate_biased_actor_the_digest_cannot_see(self):
        """A server that kept the checkpoint's gate bias acts with the same weights (same digest), so only the
        update-0 identity can catch it."""
        biased = copy.deepcopy(actor_model(0))
        biased.cfg = dataclasses.replace(biased.cfg, gate_bias=BIG_GATE_BIAS)
        self.assertEqual(LR.state_dict_digest(biased.state_dict()), self.digest)
        seats = [record(*custom_seat(biased, i, threat_to_peek=True), 0, "ours") for i in range(2)]
        root = self.root / "shards-biased"
        write_games(root, config(), [[("g0", [seats[0]], "il_v4_t1", "win", 0),
                                      ("g1", [seats[1]], "il_v3_t1", "loss", 0)]], policy_digest=self.digest)
        lr = self.learner()
        before = snapshot(lr.policy)
        with self.assertRaises(LR.IdentityError) as cm:
            lr.update(load(lr, root, games=2, workers=1), replay_bitwise_ok=True)
        self.assertIn("logp/", str(cm.exception))
        self.assertEqual(changed(before, snapshot(lr.policy)), [])

    def test_pi_prox_uses_the_actor_temperature(self):
        """A seat sampled at T = 0.5 must be scored at T = 0.5 (and T = 1 would be visibly off)."""
        t = (0.5, 0.5, 0.5)
        m = actor_model(0)
        seats = [record(*custom_seat(m, i, temperature=t), 0, "ours", temperature=t) for i in range(2)]
        root = self.root / "shards-t05"
        write_games(root, config(), [[("g0", [seats[0]], "il_v4_t1", "win", 0),
                                      ("g1", [seats[1]], "il_v3_t1", "loss", 0)]], policy_digest=self.digest)
        lr = self.learner()
        prep = lr.prepare(load(lr, root, games=2, workers=1), identity=True)
        self.assertEqual({it.temperature for it in prep.items}, {t})
        self.assertLess(max(prep.identity["logp_max_abs_diff"].values()), 1e-9)
        it = prep.items[0]
        b = lr._batch([it])
        with torch.no_grad():
            out = lr._forward(lr.policy, b, lr._cell_args(b, [it]))
        lp1 = LP.executed_logp(out, it.dt, it.action)
        self.assertTrue(bool(it.play.any()))
        self.assertGreater(float((lp1 - it.mu)[it.play].abs().max()), 1e-3)


# --------------------------------------------------------------------------- regression tests of the review fixes
class TestFixLearnerGuards(Fixture):
    """Regression tests of the learner's guards added after review."""

    def test_learner_invalid_rows_stop_before_any_step(self):
        real = LP.executed_logp_terms

        def one_invalid(*a, **k):
            t = real(*a, **k)
            t["valid"] = t["valid"].clone()
            t["valid"][0] = False
            return t

        lr = self.learner()
        before = snapshot(lr.policy)
        with mock.patch.object(LP, "executed_logp_terms", side_effect=one_invalid):
            with self.assertRaises(LR.IdentityError) as cm:
                lr.update(self.data(lr), replay_bitwise_ok=True)
        self.assertIn("learner_invalid_rows", str(cm.exception))
        self.assertTrue(lr.monitor.stopped)
        self.assertEqual(changed(before, snapshot(lr.policy)), [])

    def test_extra_may_not_set_learner_keys(self):
        lr = self.learner()
        for bad in ({"clip_frac": 0.0}, {"entropy/family": 1.0}, {"ev_overall": 0.5}, {"mu_dlogp_max/wait": 0.0},
                    {"learner_invalid_rows": 0}, {"drop_rate": 0.5}):
            with self.assertRaises(ValueError, msg=str(bad)):
                lr.update(self.data(lr), replay_bitwise_ok=True, extra_metrics=bad)
        self.assertEqual(lr.steps_total, 0)
        self.assertFalse(lr.monitor.stopped)
        res = lr.update(self.data(lr), replay_bitwise_ok=True,                # agreeing drop rate + rollout keys pass
                        extra_metrics={"drop_rate": 0.0, "refusals_code19": 0, "forced_fraction": 0.25})
        self.assertEqual(res.metrics["forced_fraction"], 0.25)                # the rollout's value beats the fallback
        self.assertEqual(res.metrics["refusals_code19"], 0)
        self.assertFalse(res.stopped)

    def test_strict_post_step_missing_raises_after_bookkeeping(self):
        cfg = config(STEP_OVERRIDES)
        mon = M.Monitor(thresholds=M.Thresholds.from_mapping(cfg.monitor_thresholds()), base_lr=cfg.policy.lr,
                        base_beta=cfg.kl.beta, strict=True)
        lr = LR.Learner.from_il(cfg, self.src, monitor=mon)
        real = LR.L.update_metrics

        def nan_clip(*a, **k):
            out = real(*a, **k)
            out["clip_frac"] = float("nan")
            return out

        with mock.patch.object(LR.L, "update_metrics", side_effect=nan_clip):
            with self.assertRaises(M.MetricsMissingError) as cm:
                lr.update(self.data(lr, self.shards_step), replay_bitwise_ok=True,
                          extra_metrics={"refusals_code19": 0})
        self.assertIn("clip_frac", cm.exception.keys)
        res = cm.exception.result
        self.assertGreater(res.steps, 0)
        self.assertEqual(lr.next_update, UPDATE + 1)                          # bookkeeping done: can save
        self.assertEqual(lr.monitor.last_update, UPDATE)
        self.assertIn("clip_frac", lr.monitor.last_missing)
        self.assertTrue(lr.monitor.strict)

    def test_load_checks_the_config_digest_when_the_chain_passes_it(self):
        lr = self.learner(STEP_OVERRIDES)                     # self.shards carry the BASE_OVERRIDES digest
        with self.assertRaises(B.SchemaError):
            lr.load(self.shards, UPDATE, expect_config_digest=lr.config_digest)
        lr.load(self.shards_step, UPDATE, expect_config_digest=lr.config_digest)


class TestFixLearnerCheckpoint(Fixture):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cfg = config(STEP_OVERRIDES)
        cls.dash = cls.root / "dash.jsonl"
        cls.base = M.Baselines(il_plays_per_min=2.5, il_card_rate={"example-card": 0.4}, il_forced_fraction=0.3)
        mon = M.Monitor(cls.base, M.Thresholds.from_mapping(cfg.monitor_thresholds()), base_lr=cfg.policy.lr,
                        base_beta=cfg.kl.beta, dashboard_path=cls.dash)
        cls.lr = LR.Learner.from_il(cfg, cls.src, monitor=mon)
        cls.path = cls.lr.save(cls.root / "ppo_u0.pt")

    def test_baselines_dashboard_and_kw_survive_a_restore(self):
        r = LR.Learner.from_checkpoint(self.path, self.lr.cfg, seed=7, first_update=3)
        self.assertEqual(r.monitor.base, self.base)
        self.assertEqual(r.monitor.disabled_rules, self.lr.monitor.disabled_rules)
        self.assertEqual(r.monitor.dashboard_path, self.dash)
        self.assertEqual((r.seed, r.next_update), (7, 3))
        other = M.Baselines()
        self.assertEqual(LR.Learner.from_checkpoint(self.path, self.lr.cfg, baselines=other).monitor.base, other)

    def test_tampered_reference_or_ema_digest_is_refused(self):
        ck = torch.load(str(self.path), map_location="cpu", weights_only=False)
        for what in ("reference", "ema"):
            c = copy.deepcopy(ck)
            sd = c["ppo"]["reference"]["state"] if what == "reference" else c["model_ema"]
            k0 = next(k for k, v in sd.items() if k.startswith("temporal.") and v.is_floating_point())
            sd[k0] = sd[k0] + 1e-3
            if what == "ema":
                c["model_raw"] = sd
            bad = self.root / f"bad_{what}.pt"
            torch.save(c, bad)
            with self.assertRaises(ValueError, msg=what):
                LR.Learner.from_checkpoint(bad, self.lr.cfg)

    def test_monitor_with_other_thresholds_is_refused(self):
        cfg = self.lr.cfg
        th = dataclasses.replace(M.Thresholds.from_mapping(cfg.monitor_thresholds()), drop_stop=0.5)
        with self.assertRaises(ValueError):
            LR.Learner.from_il(cfg, self.src, monitor=M.Monitor(thresholds=th, base_lr=cfg.policy.lr,
                                                                 base_beta=cfg.kl.beta))


if __name__ == "__main__":
    unittest.main()
