"""PPO config tests (src/amini/ppo/config.py, configs/ppo_v7_main.yaml): the learner micro-batch caps, the
drift thresholds, the checkpoint pins and the recorded mid-run config change (``RUNTIME_CHANGE_KEYS``,
``with_changes``, ``config_change``, ``check_config_change``).

The run file is not published: the tests that read it are skipped.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_config_c.py
"""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amini.ppo import config as C  # noqa: E402

YAML_PATH = ROOT / "configs" / "ppo_v7_main.yaml"
YAML_SKIP = "needs the run configuration file configs/ppo_v7_main.yaml (not published)"
needs_yaml = unittest.skipUnless(YAML_PATH.is_file(), YAML_SKIP)
SHA = "0123456789abcdef" * 4


def _with(**sections) -> C.PPOConfig:
    d = C.PPOConfig().to_dict()
    for sec, vals in sections.items():
        node = d
        parts = sec.split("__")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = vals
    return C.from_dict(d)


class TestNewFields(unittest.TestCase):

    @needs_yaml
    def test_yaml_spells_the_new_fields_with_the_defaults(self):
        c = C.load_config(YAML_PATH)
        self.assertEqual((c.policy.micro_rows, c.policy.max_micro_tokens), (256, 65536))
        self.assertEqual(c.drift, C.DriftConfig())
        self.assertEqual(c.opponents.checkpoint_sha256, {"il_v4": None, "il_v3": None, "il_v2_heldout": None})
        self.assertEqual(c.digest(), C.PPOConfig().digest())

    @needs_yaml
    def test_micro_caps_ranges(self):
        self.assertEqual(C.load_config(YAML_PATH, ["policy.max_micro_tokens=0"]).policy.max_micro_tokens, 0)
        for bad in ("policy.micro_rows=0", "policy.max_micro_tokens=-1", "policy.micro_rows=1.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(C.ConfigError):
                    C.load_config(YAML_PATH, [bad])

    @needs_yaml
    def test_checkpoint_sha256_pins(self):
        c = C.load_config(YAML_PATH, [f'opponents.checkpoint_sha256.il_v4="{SHA}"'])
        self.assertEqual(c.opponents.checkpoint_sha256["il_v4"], SHA)
        self.assertEqual(c.checkpoint_pins()["il_v4"], {"path": None, "sha256": SHA})
        self.assertEqual(c.opponent_pins()["il_v4_t1"], SHA)
        self.assertEqual(c.opponent_pins()["il_v4_tstar"], SHA)
        self.assertIsNone(c.opponent_pins()["snapshot"])
        self.assertNotEqual(c.digest(), C.PPOConfig().digest())       # the pin is part of the run identity
        for bad in (SHA.upper(), SHA[:-1], SHA + "0", "g" * 64, "", 1234):
            with self.subTest(bad=bad):
                with self.assertRaises(C.ConfigError):
                    _with(opponents__checkpoint_sha256={"il_v4": bad, "il_v3": None, "il_v2_heldout": None})
        with self.assertRaises(C.ConfigError):
            _with(opponents__checkpoint_sha256={"il_v4": None, "il_v3": None})
        miss = [m.split(":")[0] for m in C.PPOConfig().missing("rollout")]
        self.assertIn("opponents.checkpoint_sha256.il_v4", miss)


@needs_yaml
class TestConfigChange(unittest.TestCase):

    def setUp(self):
        self.old = C.load_config(YAML_PATH)

    def test_whitelist(self):
        self.assertEqual(C.RUNTIME_CHANGE_KEYS[:5], ("kl.beta", "policy.lr", "gae.lambda_tick",
                                                     "rollout.games_per_update", "critic.epochs"))
        # the calibration keys: early-stop limits, alarm thresholds, the canary alarm line
        self.assertEqual(set(C.RUNTIME_CHANGE_KEYS[5:]), {
            "early_stop.joint_k3", "early_stop.play_k3", "early_stop.gate_kl", "early_stop.lr_halve_if_stopped_frac",
            "early_stop.lr_halve_window", "monitor.tiny_ratio_share_max", "monitor.gate_trend_act_rel_max",
            "monitor.gate_trend_adv_gap_max", "monitor.entropy_max_frac_of_ref", "monitor.flip_rate_max",
            "gates.canary_alarm_lb"})
        new = C.with_changes(self.old, {"kl.beta": 0.005, "policy.lr": 2e-4, "gae.lambda_tick": 0.999,
                                        "rollout.games_per_update": 2000, "critic.epochs": 6})
        self.assertEqual((new.kl.beta, new.policy.lr, new.gae.lambda_tick, new.rollout.games_per_update,
                          new.critic.epochs), (0.005, 2e-4, 0.999, 2000, 6))
        self.assertEqual(self.old.kl.beta, 0.02)                       # the old config is untouched
        with self.assertRaisesRegex(C.ConfigError, "may not change mid-run"):
            C.with_changes(self.old, {"early_stop.joint_k3": 1.0, "monitor.clip_frac_max": 0.5})
        with self.assertRaises(C.ConfigError):                         # validated like a loaded file
            C.with_changes(self.old, {"kl.beta": -1.0})

    def test_record_carries_both_digests_and_values(self):
        new = C.with_changes(self.old, {"rollout.games_per_update": 2000})
        rec = C.config_change(self.old, new, update=41, reason="stall ladder step 2")
        self.assertEqual(rec, {"schema": C.CONFIG_CHANGE_SCHEMA, "update": 41, "reason": "stall ladder step 2",
                               "old_digest": self.old.digest(), "new_digest": new.digest(),
                               "changes": {"rollout.games_per_update": [1000, 2000]}})
        json.dumps(rec, allow_nan=False)                               # plain JSON for the checkpoint
        C.check_config_change(rec, self.old, new)
        C.check_config_change(json.loads(json.dumps(rec)), self.old, new)

    def test_refusals(self):
        new = C.with_changes(self.old, {"kl.beta": 0.04})
        for kw in ({"update": 0, "reason": "x"}, {"update": True, "reason": "x"}, {"update": 2.0, "reason": "x"},
                   {"update": 2, "reason": " "}):
            with self.subTest(kw=kw):
                with self.assertRaises(C.ConfigError):
                    C.config_change(self.old, new, **kw)
        with self.assertRaisesRegex(C.ConfigError, "identical"):
            C.config_change(self.old, self.old, update=2, reason="x")
        other = C.load_config(YAML_PATH, ["monitor.drop_rate_stop=0.2"])
        with self.assertRaisesRegex(C.ConfigError, "monitor.drop_rate_stop.*may not change"):
            C.config_change(self.old, other, update=2, reason="x")
        rec = C.config_change(self.old, new, update=2, reason="x")
        for field, val in (("new_digest", "0" * 64), ("old_digest", "0" * 64), ("changes", {"kl.beta": [0.02, 0.05]}),
                           ("extra", 1)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(C.ConfigError, "does not match"):
                    C.check_config_change(dict(rec, **{field: val}), self.old, new)
        with self.assertRaisesRegex(C.ConfigError, "not a"):
            C.check_config_change(dict(rec, schema="x"), self.old, new)
        newer = C.with_changes(new, {"kl.beta": 0.05})
        with self.assertRaisesRegex(C.ConfigError, "does not match"):
            C.check_config_change(rec, self.old, newer)

    def test_diff_lists_every_changed_leaf(self):
        a = C.PPOConfig()
        b = _with(drift__gate_auc_drop_max=0.05, policy__micro_rows=512)
        self.assertEqual(C.config_diff(a, b), {"drift.gate_auc_drop_max": (0.03, 0.05),
                                               "policy.micro_rows": (256, 512)})
        self.assertEqual(C.config_diff(a, copy.deepcopy(a)), {})


if __name__ == "__main__":
    unittest.main()
