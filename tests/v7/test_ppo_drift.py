"""Tests for the PPO drift evaluator (src/amini/ppo/drift.py).

* the fixed selection subset is il_v7's val subset rule (val chunks, friendly drop, smallest-sha256 cap or the
  pinned games file) and its digest ignores item order;
* ``DriftEvaluator.evaluate`` is exactly ``il_v7.evaluate`` on the subset (the IL trainer's own entry);
* ``judge`` applies the judged thresholds of ``cfg.drift`` (per-head NLL increase, gate AUC drop, plays-per-
  minute calibration band), skips what has no rows, alarms on a value that turned non-finite;
* ``DriftResult.metrics`` gives the monitor flag ``drift_alarm`` (a reason list, empty = no alarm) on the
  ``monitor.drift_every`` cadence; the IL reference survives a chained block (``state_dict``).

Tiny fp32 models on the synthetic shards of tests/v7/common.py (private dataset format: the subset and evaluation
tests are skipped without them; the judging and cadence tests run). CPU only, deterministic, no game environment,
no GPU.
"""

from __future__ import annotations

import copy
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
import common  # noqa: E402
from common import ROOT, shard_root, source, tables, tiny_model  # noqa: E402

from amini.ppo import config as C  # noqa: E402
from amini.ppo import drift as DR  # noqa: E402
from amini.ppo import learner as LR  # noqa: E402
from amini.ppo import monitor as M  # noqa: E402
from amini.train import il_v7 as IL  # noqa: E402

torch.set_num_threads(2)

SMOKE = str(ROOT / "configs" / "aminiformer_v1_smoke.yaml")
MAIN_YAML = ROOT / "configs" / "ppo_v7_main.yaml"
SKIP_REASON = getattr(common, "SKIP_REASON", "needs the private dataset")


def train_cfg():
    if not Path(SMOKE).is_file():                     # the IL smoke config of the test shards
        raise unittest.SkipTest(SKIP_REASON)
    return IL.TrainConfig.from_file(SMOKE, {"num_workers": 0})


def ppo_cfg(**drift):
    d = C.PPOConfig().to_dict()
    d["drift"].update(drift)
    return C.from_dict(d)


def model(seed):
    return tiny_model(seed, double=False)


def train_items():
    """Split-0 items of the synthetic generation (five games; the val split has one)."""
    tc = train_cfg()
    return IL.chunk_episodes(source(), 0, tc.max_seq_rows, tc.burn_in_rows)


def report(terms=None, auc=0.8, base=1.0, threat=1.0):
    t = {"gate": 0.5, "slot": 1.0, "cell": 2.0, "ability": 0.3}
    t.update(terms or {})
    return {"selection_nll": sum(t.values()), "terms": t, "gate_auc": auc,
            "calibration": {"base": {"ratio": base}, "threat": {"ratio": threat}}}


# =========================================================================== the fixed subset
class TestSubset(unittest.TestCase):

    def test_selection_items_follow_il_v7_val_rule(self):
        src, tc = source(), train_cfg()
        val = IL.chunk_episodes(src, 1, tc.max_seq_rows, tc.burn_in_rows)
        self.assertTrue(val)
        items, rec = DR.selection_items(src, tc, max_games=1000)
        want = IL.cap_val_items(src, IL.drop_games(val, IL.friendly_games(src))[0], 1000)
        self.assertEqual(items, want)
        self.assertEqual((rec["max_games"], rec["exclude_friendly"]), (1000, True))
        with mock.patch.object(IL, "friendly_games", return_value={(it[0][0], it[0][1]) for it in val}):
            items, rec = DR.selection_items(src, tc, max_games=1000)
            self.assertEqual(items, [])
            self.assertEqual(rec["friendly_dropped"]["games"], 1)
            with self.assertRaisesRegex(ValueError, "empty"):
                DR.DriftEvaluator(ppo_cfg(), tc, src, tables())
            items, _ = DR.selection_items(src, tc, max_games=1000, exclude_friendly=False)
            self.assertEqual(items, val)

    def test_cap_keeps_the_smallest_sha256_games(self):
        src, tc = source(), train_cfg()
        with mock.patch.object(IL, "chunk_episodes", return_value=train_items()):
            items, _ = DR.selection_items(src, tc, max_games=2, exclude_friendly=False)
        self.assertEqual(items, IL.cap_val_items(src, train_items(), 2))
        self.assertEqual(len({(it[0][0], it[0][1]) for it in items}), 2)

    def test_pinned_games_file(self):
        src, tc = source(), train_cfg()
        val = IL.chunk_episodes(src, 1, tc.max_seq_rows, tc.burn_in_rows)
        tag = IL._tag_text(src, val[0][0][0], val[0][0][1])
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as d:
            f = Path(d) / "val_games.txt"
            f.write_text(f"# pinned\n{tag}\n", encoding="ascii")
            items, rec = DR.selection_items(src, tc, max_games=1000, games_file=str(f))
            self.assertEqual(items, val)
            self.assertEqual(rec["games_file_sha256"], IL.file_sha256(str(f)))
            other = IL._tag_text(src, *next((it[0][0], it[0][1]) for it in train_items()))
            f.write_text(other + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "not val games"):
                DR.selection_items(src, tc, max_games=1000, games_file=str(f))

    def test_digest_ignores_order_and_sees_membership(self):
        src, items = source(), train_items()
        a = DR.subset_digest(src, items)
        self.assertEqual(a, DR.subset_digest(src, list(reversed(items))))
        self.assertNotEqual(a, DR.subset_digest(src, items[1:]))
        ev = DR.DriftEvaluator(ppo_cfg(), train_cfg(), src, tables(), items=items)
        self.assertEqual((ev.digest, ev.n_games), (a, len({(it[0][0], it[0][1]) for it in items})))
        self.assertEqual(ev.selection, {"given": True})


# =========================================================================== evaluation
class TestEvaluate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tc = train_cfg()
        cls.items = train_items()

    def evaluator(self, cfg=None, **kw):
        return DR.DriftEvaluator(cfg or ppo_cfg(), self.tc, source(), tables(), items=self.items, **kw)

    def test_evaluate_is_il_v7_evaluate(self):
        m = model(0)
        m.train()
        rep = self.evaluator().evaluate(m)
        self.assertTrue(m.training)                                   # mode restored
        ds = IL.EpisodeBatches(source(), self.items, tables(), self.tc, train=False)
        want = IL.evaluate(m, IL.make_loader(ds, 0), torch.device("cpu"), torch.float32, self.tc.loss, 0,
                           opp_hand_reduce=self.tc.aux_opp_hand_reduce)
        self.assertEqual(rep["selection_nll"], want["selection_nll"])
        self.assertEqual(rep["terms"], want["terms"])
        self.assertEqual(json.dumps(rep["calibration"], sort_keys=True), json.dumps(want["calibration"], sort_keys=True))
        self.assertEqual(rep["gate_auc"], want["gate_auc"])
        for h in DR.HEADS:
            self.assertTrue(np.isfinite(rep["terms"][h]), h)

    def test_same_policy_raises_no_alarm(self):
        ev = self.evaluator()
        ref = ev.set_reference(model(0))
        res = ev.check(copy.deepcopy(model(0)), update=5)
        self.assertFalse(res.alarm)
        self.assertEqual(res.reasons, ())
        for h in DR.HEADS:
            self.assertEqual(res.deltas[f"nll_delta/{h}"], 0.0, h)
        self.assertEqual(res.deltas["gate_auc_drop"], 0.0)
        for s in DR.STRATA:
            if res.deltas[f"calibration_rel/{s}"] is not None:
                self.assertEqual(res.deltas[f"calibration_rel/{s}"], 1.0)
        m = res.metrics()
        self.assertEqual(m["drift_alarm"], [])
        self.assertIs(M._flag(m["drift_alarm"]), False)               # a valid flag, not 'missing'
        self.assertEqual(m["drift/selection_nll"], ref["selection_nll"])
        self.assertEqual((res.update, res.subset_digest, res.n_games), (5, ev.digest, ev.n_games))

    def test_another_policy_alarms_on_the_judged_thresholds(self):
        ev = self.evaluator()
        reps = {s: ev.evaluate(model(s)) for s in (0, 1)}
        self.assertNotEqual(reps[0]["terms"]["gate"], reps[1]["terms"]["gate"])
        good, bad = sorted((0, 1), key=lambda s: reps[s]["terms"]["gate"])   # reference = the lower gate NLL
        tight = self.evaluator(ppo_cfg(nll_increase_max={"gate": 1e-9, "slot": 10.0, "cell": 10.0, "ability": 10.0},
                                       gate_auc_drop_max=1.0, calibration_ratio_band=[1e-6, 100.0]))
        tight.set_reference(model(good))
        res = tight.check(model(bad), update=10)
        self.assertTrue(res.alarm)
        self.assertEqual(len(res.reasons), 1)
        self.assertIn("gate NLL", res.reasons[0])
        self.assertAlmostEqual(res.deltas["nll_delta/gate"], reps[bad]["terms"]["gate"] - reps[good]["terms"]["gate"],
                               places=12)
        self.assertEqual((res.reasons, res.deltas), DR.judge(tight.cfg, tight.reference, res.report))
        self.assertIs(M._flag(res.metrics()["drift_alarm"]), True)
        loose = self.evaluator(ppo_cfg(nll_increase_max={"gate": 10.0, "slot": 10.0, "cell": 10.0, "ability": 10.0},
                                       gate_auc_drop_max=1.0, calibration_ratio_band=[1e-6, 100.0]))
        loose.set_reference(model(good))
        self.assertFalse(loose.check(model(bad), update=10).alarm)

    def test_reference_state_survives_a_chained_block(self):
        ev = self.evaluator()
        with self.assertRaisesRegex(RuntimeError, "set_reference"):
            ev.check(model(0), update=5)
        ev.set_reference(model(0))
        st = ev.state_dict()
        self.assertEqual((st["schema"], st["subset_digest"]), (DR.DRIFT_SCHEMA, ev.digest))
        ev2 = self.evaluator(reference=st)
        self.assertEqual(ev2.reference, ev.reference)
        self.assertEqual(ev2.check(model(1), update=5).deltas, ev.check(model(1), update=5).deltas)
        other = DR.DriftEvaluator(ppo_cfg(), self.tc, source(), tables(), items=self.items[1:])
        with self.assertRaisesRegex(ValueError, "another subset"):
            other.load_state_dict(st)
        with self.assertRaisesRegex(ValueError, "schema"):
            ev.load_state_dict(dict(st, schema="x"))

    def test_models_with_a_gate_bias_are_refused(self):
        """Both sides act without a gate calibration bias (module docstring)."""
        ev = self.evaluator()
        biased = tiny_model(0, double=False, gate_bias=[[0.0, -0.05, 0.0], [0.0, -0.1, 0.0]])   # example values
        with self.assertRaisesRegex(ValueError, "gate_bias"):
            ev.set_reference(biased)
        self.assertIsNone(ev.reference)
        ev.set_reference(model(0))
        with self.assertRaisesRegex(ValueError, "gate_bias"):
            ev.check(biased, update=5)

    def test_open_reads_the_generation(self):
        ev = DR.DriftEvaluator.open(ppo_cfg(), self.tc, [shard_root()])
        val = IL.chunk_episodes(ev.src, 1, self.tc.max_seq_rows, self.tc.burn_in_rows)
        self.assertEqual(ev.n_games, len({(it[0][0], it[0][1]) for it in val}))
        self.assertEqual(ev.selection["max_games"], ppo_cfg().drift.subset_games)
        rep = ev.evaluate(model(0))
        self.assertTrue(np.isfinite(rep["selection_nll"]))


# =========================================================================== judging and cadence
class TestJudge(unittest.TestCase):

    def test_thresholds_are_the_config_values(self):
        cfg = ppo_cfg()
        ref = report()
        lim = cfg.drift.nll_increase_max
        ok = report({h: ref["terms"][h] + 0.5 * lim[h] for h in DR.HEADS}, auc=0.8 - 0.5 * cfg.drift.gate_auc_drop_max,
                    base=1.2, threat=0.8)
        self.assertEqual(DR.judge(cfg, ref, ok)[0], ())
        bad = report({h: ref["terms"][h] + 2 * lim[h] for h in DR.HEADS}, auc=0.8 - 2 * cfg.drift.gate_auc_drop_max,
                     base=2.0, threat=0.5)
        reasons, deltas = DR.judge(cfg, ref, bad)
        self.assertEqual(len(reasons), 4 + 1 + 2)
        for h in DR.HEADS:
            self.assertAlmostEqual(deltas[f"nll_delta/{h}"], 2 * lim[h], places=12)
        self.assertAlmostEqual(deltas["gate_auc_drop"], 2 * cfg.drift.gate_auc_drop_max, places=12)
        self.assertEqual((deltas["calibration_rel/base"], deltas["calibration_rel/threat"]), (2.0, 0.5))
        # a lower NLL / higher AUC is never an alarm (drift means moving AWAY from the human data)
        self.assertEqual(DR.judge(cfg, ref, report({h: 0.0 for h in DR.HEADS}, auc=1.0))[0], ())
        # the band edges are inside
        lo, hi = cfg.drift.calibration_ratio_band
        self.assertEqual(DR.judge(cfg, ref, report(base=lo, threat=hi))[0], ())
        wide = ppo_cfg(calibration_ratio_band=[0.25, 4.0])
        self.assertEqual(DR.judge(wide, ref, report(base=2.0, threat=0.5))[0], ())

    def test_missing_rows_and_non_finite_values(self):
        cfg = ppo_cfg()
        nan = float("nan")
        ref = report({"ability": nan}, threat=nan)
        reasons, deltas = DR.judge(cfg, ref, report({"ability": nan}, threat=nan))
        self.assertEqual(reasons, ())                                  # no rows in both: not judged
        self.assertIsNone(deltas["nll_delta/ability"])
        self.assertIsNone(deltas["calibration_rel/threat"])
        reasons, _ = DR.judge(cfg, report(), report({"cell": nan}, auc=nan, base=float("inf")))
        self.assertEqual(len(reasons), 3)
        self.assertTrue(any("cell NLL not finite" in r for r in reasons))
        self.assertTrue(any("gate AUC not finite" in r for r in reasons))
        self.assertTrue(any("base calibration not finite" in r for r in reasons))
        self.assertEqual(DR.judge(cfg, report(base=0.0), report(base=5.0))[0], ())   # IL ratio 0: not judged
        m = DR.DriftResult(update=5, reasons=(), deltas={"nll_delta/ability": None}, report=report(),
                           subset_digest="d", n_games=1).metrics()
        self.assertTrue(np.isnan(m["drift/nll_delta/ability"]))

    def test_config_validation(self):
        for bad in ({"nll_increase_max": {"gate": 0.0, "slot": 0.1, "cell": 0.1, "ability": 0.1}},
                    {"nll_increase_max": {"gate": 0.1, "slot": 0.1, "cell": 0.1}},
                    {"gate_auc_drop_max": 0.0}, {"calibration_ratio_band": [1.2, 1.5]},
                    {"calibration_ratio_band": [0.5, 0.9]}, {"subset_games": 0}):
            with self.subTest(bad=bad):
                with self.assertRaises(C.ConfigError):
                    ppo_cfg(**bad)

    @unittest.skipUnless(MAIN_YAML.is_file(), "needs the run configuration file")
    def test_run_file_drift_section_is_the_default(self):
        self.assertEqual(C.load_config(MAIN_YAML).drift, C.DriftConfig())

    def test_cadence_and_monitor_keys(self):
        cfg = ppo_cfg()
        every = cfg.monitor.drift_every
        self.assertEqual([u for u in range(1, 21) if DR.drift_due(cfg, u)], list(range(every, 21, every)))
        self.assertFalse(DR.drift_due(NS(monitor=NS(drift_every=0)), 5))
        d = cfg.to_dict()
        d["policy"]["backbone_dtype"] = "fp32"                        # bf16 tolerances are set per run
        th = M.Thresholds.from_mapping(C.from_dict(d).monitor_thresholds())
        self.assertEqual(th.drift_every, every)
        m = DR.DriftResult(update=every, reasons=("x",), deltas={"gate_auc_drop": 0.1}, report=report(),
                           subset_digest="d", n_games=1).metrics()
        self.assertEqual(m["drift_alarm"], ["x"])
        for k in m:                                                    # never a learner-owned key
            self.assertNotIn(k, LR.OWNED_KEYS)
            self.assertFalse(k.startswith(LR.OWNED_PREFIXES), k)


if __name__ == "__main__":
    unittest.main()
