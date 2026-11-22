"""Metric coverage of amini.ppo.monitor ("no rule silently off").

* ``expected_update_keys`` follows the active rules, the audit / drift cadence and the present baselines;
* a record lacking an expected key (or carrying NaN) raises one ``metrics_missing`` alarm per update, listing
  every such key, and the dashboard record keeps the list;
* ``strict=True`` raises MetricsMissingError before any state changes; a full record passes strictly;
* the former code-only constants (temp lr floor, ladder step 2's games) come from Thresholds.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_monitor_coverage.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from amini.ppo import monitor as M  # noqa: E402

GS = "card-a"


def _base(**kw) -> M.Baselines:
    d = dict(il_entropy={"cell": 2.0, "gate": 0.5, "slot": 1.0, "ability": 0.7},
             il_plays_per_min=10.0, il_abilities_per_min=2.0,
             il_card_rate={GS: 0.10, "card-c": 0.20}, core_cards=(GS,), il_masked_legal_refused_rate=0.01,
             il_forced_fraction=0.30, p3_opp_nonwait_per_game=40.0, m2_games_per_hour=5000.0, ev_warmup_end=0.40)
    d.update(kw)
    return M.Baselines(**d)


def full_record(update: int) -> dict:
    """Every key the producers deliver, at healthy values (no rule fires)."""
    rec = {"update": update, "refusals_code19": 0, "drop_rate": 0.001, "clip_frac": 0.1, "play_clip_frac": 0.12,
           "kl_ref_total": 0.01 * (1 + update), "ev_overall": 0.30, "ev_last_third": 0.50, "early_stopped": (),
           "masked_legal_refused_rate": 0.01, "entropy/family": 0.5, "entropy/cells": 2.0, "entropy/slot": 1.0,
           "entropy/ability": 0.7, "plays_per_min": 10.0, "abilities_per_min": 2.0, "card_rate": {GS: 0.10, "card-c": 0.20}, "forced_fraction": 0.31,
           "opp_il_nonwait_per_game": 40.0, "games_per_hour": 5000.0, "ev_by_opponent": {"il_v4_t1": 0.3}}
    if update % 10 == 0:                     # the example audit / drift cadence
        rec["mask_recompute_mismatch"] = 0
        rec["drift_alarm"] = False
    return rec


class TestExpectedKeys(unittest.TestCase):
    def test_keys_follow_rules_cadence_and_baselines(self):
        m = M.Monitor(_base())
        keys = m.expected_update_keys(1)
        for k in M.ALWAYS_NUMERIC_KEYS + ("early_stopped", "masked_legal_refused_rate", "entropy/family",
                                          "entropy/cells", "plays_per_min", "abilities_per_min", f"card_rate[{GS}]",
                                          "card_rate[card-c]", "forced_fraction", "opp_il_nonwait_per_game",
                                          "games_per_hour"):
            self.assertIn(k, keys)
        self.assertNotIn("mask_recompute_mismatch", keys)
        self.assertNotIn("drift_alarm", keys)
        k10 = m.expected_update_keys(10)
        self.assertEqual(set(k10) - set(keys), {"mask_recompute_mismatch", "drift_alarm"})
        bare = M.Monitor().expected_update_keys(10)                    # no baselines: only baseline-free rules
        self.assertEqual(set(bare), set(M.ALWAYS_NUMERIC_KEYS) | {"early_stopped", "mask_recompute_mismatch",
                                                                   "drift_alarm"})
        th = M.Thresholds(mask_audit_every=2, drift_every=3)
        self.assertIn("mask_recompute_mismatch", M.Monitor(thresholds=th).expected_update_keys(4))
        self.assertNotIn("drift_alarm", M.Monitor(thresholds=th).expected_update_keys(4))
        zero_il = M.Monitor(_base(il_masked_legal_refused_rate=0.0)).expected_update_keys(1)
        self.assertIn("masked_legal_refused_rate", zero_il)             # IL 0 is a valid reference

    def test_full_record_has_nothing_missing(self):
        m = M.Monitor(_base(), strict=True)
        for u in range(1, 11):
            self.assertEqual(m.missing_update_keys(full_record(u)), [], u)
            acts = m.observe_update(full_record(u))
            self.assertEqual([a.kind for a in acts], [], [a.to_json() for a in acts])

    def test_aliases_flags_and_nested_entropy(self):
        m = M.Monitor(_base())
        rec = full_record(10)
        rec["clip_frac_play"] = rec.pop("play_clip_frac")               # alias
        rec["entropy"] = {"gate": 0.5, "cell": 2.0, "slot": 1.0, "ability": 0.7}   # nested form, config head names
        del rec["entropy/family"], rec["entropy/cells"], rec["entropy/slot"], rec["entropy/ability"]
        rec["early_stopped"] = False                                     # False / () are values, not "missing"
        self.assertEqual(m.missing_update_keys(rec), [])
        rec["drift_alarm"] = None
        self.assertEqual(m.missing_update_keys(rec), ["drift_alarm"])


class TestMissingAlarm(unittest.TestCase):
    def test_one_alarm_per_update_listing_every_key(self):
        m = M.Monitor(_base())
        rec = full_record(10)
        del rec["drop_rate"], rec["mask_recompute_mismatch"]
        rec["ev_overall"] = float("nan")
        rec["card_rate"] = {"card-c": 0.2}                               # core card missing
        acts = m.observe_update(rec)
        miss = [a for a in acts if a.monitor == "metrics_missing"]
        self.assertEqual(len(miss), 1)
        self.assertEqual(miss[0].kind, "alarm")
        self.assertEqual(miss[0].detail["keys"], ["drop_rate", "ev_overall", "mask_recompute_mismatch",
                                                  f"card_rate[{GS}]"])
        acts = m.observe_update(full_record(11))
        self.assertEqual([a.monitor for a in acts], [])                  # complete again: no alarm
        rec12 = full_record(12)
        del rec12["games_per_hour"]
        self.assertEqual([a.monitor for a in m.observe_update(rec12)], ["metrics_missing"])  # every update

    def test_losses_only_record_raises_the_alarm(self):
        """A learner passing only losses.update_metrics output turns rules off."""
        m = M.Monitor(_base())
        rec = {"update": 1, "clip_frac": 0.1, "play_clip_frac": 0.1, "joint_k3": 0.01, "play_k3": 0.02,
               "gate_kl": 0.001, "entropy/family": 0.5, "entropy/cells": 2.0, "early_stopped": ()}
        acts = m.observe_update(rec)
        keys = [a for a in acts if a.monitor == "metrics_missing"][0].detail["keys"]
        for k in ("refusals_code19", "drop_rate", "ev_overall", "ev_last_third", "masked_legal_refused_rate",
                  f"card_rate[{GS}]", "plays_per_min", "opp_il_nonwait_per_game", "games_per_hour", "kl_ref_total"):
            self.assertIn(k, keys)

    def test_dashboard_keeps_the_list(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dash.jsonl"
            m = M.Monitor(_base(), dashboard_path=p)
            rec = full_record(1)
            del rec["drop_rate"]
            m.observe_update(rec)
            m.observe_update(full_record(2))
            m.observe_gate("screen", 10, 0.01)
            recs = M.read_dashboard(p)
            self.assertEqual([r["metrics_missing"] for r in recs], [["drop_rate"], [], []])
            json.dumps(recs, allow_nan=False)

    def test_stopped_monitor_does_not_alarm(self):
        m = M.Monitor(_base())
        m.observe_update(dict(full_record(1), refusals_code19=1))
        self.assertTrue(m.stopped)
        self.assertEqual(m.observe_update({"update": 2}), [])


class TestStrict(unittest.TestCase):
    def test_raises_before_any_state_change(self):
        m = M.Monitor(_base(), strict=True)
        m.observe_update(full_record(1))
        before = json.dumps(m.state_dict(), sort_keys=True)
        rec = full_record(2)
        rec["clip_frac"] = 0.9                                           # would start a clip streak
        del rec["ev_last_third"]
        with self.assertRaises(M.MetricsMissingError) as cm:
            m.observe_update(rec)
        self.assertEqual(cm.exception.keys, ["ev_last_third"])
        self.assertEqual(cm.exception.update, 2)
        self.assertIsInstance(cm.exception, ValueError)
        self.assertEqual(json.dumps(m.state_dict(), sort_keys=True), before)
        self.assertEqual(m.clip_streak, 0)

    def test_strict_survives_save_and_restore(self):
        m = M.Monitor(_base(), strict=True)
        m.observe_update(full_record(1))
        r = M.Monitor.from_state(m.state_dict(), _base(), strict=True)
        self.assertTrue(r.strict)
        with self.assertRaises(M.MetricsMissingError):
            r.observe_update({"update": 2})


class TestFormerCodeConstants(unittest.TestCase):
    def test_temp_lr_floor_from_thresholds(self):
        m = M.Monitor(_base(), thresholds=M.Thresholds(temp_lr_floor=0.5))
        self.assertEqual(m.temp_lr_floor, 0.5)
        for u in range(1, 13):
            m.observe_update({"update": u, "clip_frac": 0.9, "early_stopped": True})
        m.observe_gate("screen", 10, -0.05)
        m.observe_gate("screen", 20, -0.05)
        m.observe_gate("screen", 30, -0.05)                              # rollback: another lr cut request
        self.assertGreaterEqual(m.lr_temp, 0.5 - 1e-12)
        self.assertEqual(M.Monitor(temp_lr_floor=0.1).temp_lr_floor, 0.1)  # explicit argument still wins
        self.assertEqual(M.Monitor().temp_lr_floor, M.TEMP_LR_FLOOR)

    def test_ladder_step2_games_from_thresholds(self):
        m = M.Monitor(_base(), thresholds=M.Thresholds(stall_games_per_update=4096))
        steps = []
        for u in range(20, 200, 20):
            steps += [a for a in m.observe_gate("g1", u, -0.01) if a.kind == "stall_step"]
        step2 = [a for a in steps if a.detail.get("name") == "games_per_update"]
        self.assertEqual(len(step2), 1)
        self.assertEqual(step2[0].detail["games_per_update"], 4096)
        self.assertIn("4,096", step2[0].detail["do"])
        self.assertEqual(M.STALL_LADDER[1]["games_per_update"], M.STALL_GAMES_PER_UPDATE)


if __name__ == "__main__":
    unittest.main()
