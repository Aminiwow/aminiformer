"""Checks of the optional PPO learner compiler (no game environment; the smoke tests need the private dataset)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent            # as the other test_ppo_* files: run by file path
for _p in (_HERE, _HERE.parents[1] / 'tools', _HERE.parents[1] / 'src'):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from test_ppo_learner import BASE_OVERRIDES, Fixture, config
from amini.ppo import config as C
from amini.ppo import learner as LR


class TestCompileConfig(unittest.TestCase):
    def test_defaults_are_omitted_from_digest(self):
        base = config()
        explicit = config(BASE_OVERRIDES + ["learner.compile_mode=", "learner.compile_dynamic=false",
                                            "rollout.games_per_worker=1"])
        self.assertEqual(base.digest(), explicit.digest())
        self.assertNotIn("learner", base.to_dict())
        self.assertNotIn("games_per_worker", base.to_dict()["rollout"])

    def test_enabled_config_is_recorded_and_validated(self):
        cfg = config(BASE_OVERRIDES + ["learner.compile_mode=default", "learner.compile_dynamic=true",
                                       "rollout.games_per_worker=4"])
        self.assertEqual(cfg.learner.compile_mode, "default")
        self.assertTrue(cfg.learner.compile_dynamic)
        self.assertEqual(cfg.rollout.games_per_worker, 4)
        self.assertEqual(cfg.to_dict()["learner"]["compile_mode"], "default")
        with self.assertRaises(C.ConfigError):
            config(BASE_OVERRIDES + ["rollout.games_per_worker=9"])

    def test_compile_is_a_base_setting_never_a_runtime_change(self):
        base = config()
        self.assertFalse(set(C.COMPILE_SWITCH_KEYS) & set(C.RUNTIME_CHANGE_KEYS))
        with self.assertRaisesRegex(C.ConfigError, "may not change mid-run"):
            C.with_changes(base, {"learner.compile_mode": "default"})
        with self.assertRaisesRegex(C.ConfigError, "may not change mid-run"):
            C.with_changes(base, {"learner.compile_mode": "default"}, allow_lag=True)
        on = C.with_changes(base, {"learner.compile_mode": "default", "learner.compile_dynamic": True},
                            allow_compile=True)                          # a comparison config
        self.assertEqual((on.learner.compile_mode, on.learner.compile_dynamic), ("default", True))
        with self.assertRaises(C.ConfigError):
            C.config_change(base, on, update=3, reason="x")              # never recorded as a mid-run change
        with self.assertRaises(C.ConfigError):
            C.config_change(base, on, update=3, reason="x", allow_lag=True)
        rec = C.config_change(base, on, update=3, reason="compile comparison", allow_compile=True)
        self.assertIs(rec["compile_switch"], True)                        # a comparison copy only, and marked
        C.check_config_change(rec, base, on)
        plain = {k: v for k, v in rec.items() if k != "compile_switch"}
        with self.assertRaises(C.ConfigError):                           # an unmarked record cannot smuggle it
            C.check_config_change(plain, base, on)


class TestCompileSmoke(Fixture):
    def test_smoke_failure_falls_back_before_prepare(self):
        cfg = config(BASE_OVERRIDES + ["learner.compile_mode=default"])
        lr = LR.Learner.from_il(cfg, self.src)
        data = lr.load(self.shards, 1, expect_games=4, expect_workers=2, expect_rollout_id="rollout-u1-a")
        with mock.patch("torch.compile", side_effect=RuntimeError("compiler unavailable")):
            prep = lr.prepare(data)
        self.assertEqual(lr.compile_status["smoke"], "fallback")
        self.assertFalse(lr.compile_status["active"])
        self.assertIn("compiler unavailable", lr.compile_status["error"])

    def test_aot_eager_smoke_has_stable_graph_budget(self):
        cfg = config(BASE_OVERRIDES + ["learner.compile_mode=default", "learner.compile_dynamic=true"])
        lr = LR.Learner.from_il(cfg, self.src)
        data = lr.load(self.shards, 1, expect_games=4, expect_workers=2, expect_rollout_id="rollout-u1-a")
        old = os.environ.get("PPO_LEARNER_COMPILE_BACKEND")
        os.environ["PPO_LEARNER_COMPILE_BACKEND"] = "aot_eager"
        try:
            prep = lr.prepare(data)
        finally:
            if old is None:
                os.environ.pop("PPO_LEARNER_COMPILE_BACKEND", None)
            else:
                os.environ["PPO_LEARNER_COMPILE_BACKEND"] = old
        self.assertEqual(lr.compile_status["smoke"], "pass")
        self.assertTrue(lr.compile_status["active"])
        self.assertLessEqual(lr.compile_status["graphs"], LR.COMPILE_MAX_GRAPHS)
        self.assertIn("learner_compile/active", prep.metrics)


class TestCompiledCopies(Fixture):
    """A copy of a compiled module must not run on the source's weights."""

    def _compiled_learner(self):
        cfg = config(BASE_OVERRIDES + ["learner.compile_mode=default", "learner.compile_dynamic=true"])
        lr = LR.Learner.from_il(cfg, self.src)
        data = lr.load(self.shards, 1, expect_games=4, expect_workers=2, expect_rollout_id="rollout-u1-a")
        with mock.patch.dict(os.environ, {"PPO_LEARNER_COMPILE_BACKEND": "aot_eager"}):
            lr.prepare(data)
        self.assertTrue(lr.compile_status["active"], lr.compile_status)
        self.assertIn("forward", lr.policy.temporal.__dict__)            # the compiled override is installed
        it = lr._item(0, next(iter(data.seat_games())))
        b = lr._batch([it])
        return lr, it, b, lr._cell_args(b, [it])

    @staticmethod
    def _perturb(policy):
        import torch
        with torch.no_grad():
            for p in policy.temporal.parameters():
                p.add_(torch.randn_like(p) * 0.5)

    def test_module_copy_drops_instance_forward_only(self):
        import torch
        lin = torch.nn.Linear(2, 2)
        plain = LR.module_copy(lin)
        self.assertNotIn("forward", plain.__dict__)
        self.assertEqual(sorted(plain.state_dict()), sorted(lin.state_dict()))
        lin.forward = lambda x: x                                           # an instance-level override
        self.assertNotIn("forward", LR.module_copy(lin).__dict__)
        self.assertIn("forward", lin.__dict__)                              # the source keeps its own

    def test_reference_swapped_to_the_compiled_policy_keeps_its_own_weights(self):
        import torch
        lr, it, b, ca = self._compiled_learner()
        with mock.patch.dict(os.environ, {"PPO_LEARNER_COMPILE_BACKEND": "aot_eager"}):
            lr.set_reference(lr.policy)                                     # promotion: reference = champion
        fwd = lr.reference.temporal.__dict__.get("forward")
        self.assertIsNot(fwd, lr.policy.temporal.__dict__["forward"])
        with torch.no_grad():
            before = {k: v.clone() for k, v in lr._forward(lr.reference, b, ca, [it]).items() if torch.is_tensor(v)}
            pol_before = lr._forward(lr.policy, b, ca, [it])["slot"].clone()
            self._perturb(lr.policy)
            after = lr._forward(lr.reference, b, ca, [it])
            pol_after = lr._forward(lr.policy, b, ca, [it])["slot"]
        self.assertFalse(torch.equal(pol_before, pol_after))                # the perturbation is visible at all
        for k, v in before.items():
            self.assertTrue(torch.equal(v, after[k]), k)

    def test_lag1_parent_copy_keeps_its_own_weights(self):
        import torch
        lr, it, b, ca = self._compiled_learner()
        parent_state = {k: v.detach().clone() for k, v in lr.policy.state_dict().items()}
        ov = lr._overlap_ctx(1, parent_state, None, None)
        self.assertNotIn("forward", ov.parent.temporal.__dict__)
        with torch.no_grad():
            before = lr._forward(ov.parent, b, ca, [it])["slot"].clone()
            self._perturb(lr.policy)
            after = lr._forward(ov.parent, b, ca, [it])["slot"]
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
