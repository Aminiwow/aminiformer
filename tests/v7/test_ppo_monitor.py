"""Tests for src/amini/ppo/monitor.py.

* every monitor fires just past its (example) threshold and stays quiet on it (boundaries);
* temporary lr / beta changes are undone by one clean G1 (delta_hat >= 0) and only by a G1;
* rollback trigger (consecutive gates at or below the rollback threshold; core-card collapse; a repeated
  collapse stops), a double rollback within the window -> temporary beta increase;
* stall ladder timing (start, minimum spacing between steps, final stop);
* state round trip, JSON-lines dashboard, stop marker.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_monitor.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import types  # noqa: E402

from amini.ppo import monitor as _monitor  # noqa: E402


class _RuleMonitor(_monitor.Monitor):
    """The real Monitor without the per-update ``metrics_missing`` alarm in ``observe_update``'s result.
    These rule tests feed partial records on purpose; the alarm, the expected-key set and strict mode are
    covered in tests/v7/test_ppo_monitor_coverage.py."""

    def observe_update(self, rec):
        return [a for a in super().observe_update(rec) if a.monitor != "metrics_missing"]


M = types.SimpleNamespace(**vars(_monitor))
M.Monitor = _RuleMonitor

GS = "card-a"


def _base(**kw) -> M.Baselines:
    d = dict(il_entropy={"cell": 2.0, "gate": 0.5, "slot": 1.0, "ability": 0.7},
             il_plays_per_min=10.0, il_abilities_per_min=2.0,
             il_card_rate={GS: 0.10, "card-c": 0.20}, core_cards=(GS,), il_masked_legal_refused_rate=0.01,
             il_forced_fraction=0.30, p3_opp_nonwait_per_game=40.0, m2_games_per_hour=5000.0, ev_warmup_end=0.40)
    d.update(kw)
    return M.Baselines(**d)


def _mon(**kw) -> M.Monitor:
    return M.Monitor(_base(), **kw)


def _kinds(acts, monitor=None):
    return [a.kind for a in acts if monitor is None or a.monitor == monitor]


def _good_identity(**kw):
    r = {"logp_max_abs_diff": {"wait": 5e-6, "play": 1.3e-5, "ability": 8.6e-6}, "ratio_max_abs_dev": 1e-6,
         "kl_ref_per_head": {h: 0.0 for h in M.IDENTITY_HEADS}, "replay_bitwise_ok": True}
    r.update(kw)
    return r


class TestTableAndAction(unittest.TestCase):
    def test_table(self):
        names = [m.name for m in M.MONITORS]
        self.assertEqual(len(names), len(set(names)))
        for m in M.MONITORS:
            self.assertTrue(any(k in m.action for k in M.ACTION_KINDS), m.name)
            self.assertTrue(m.plan, m.name)                     # every rule names its rule group
        for need in ("update0_identity", "approx_kl_early_stop", "clip_fraction", "kl_flat_stall",
                     "core_card_collapse", "mask_illegal_cell", "drop_rate", "critic_ev", "swap_growth",
                     "nonfinite", "gate_rollback", "stall_ladder", "stall_final_stop"):
            self.assertIn(need, names)

    def test_action_kind_checked(self):
        with self.assertRaises(ValueError):
            M.Action("reboot", "x", "y")
        a = M.Action("alarm", "x", "y", 3, detail={"k": 1})
        self.assertEqual(a.to_json()["detail"], {"k": 1})
        self.assertEqual(M.most_severe([M.Action("alarm", "a", ""), M.Action("stop", "b", ""),
                                        M.Action("rollback", "c", "")]).kind, "stop")
        self.assertIsNone(M.most_severe([]))


class TestIdentity(unittest.TestCase):
    def test_pass(self):
        m = _mon()
        self.assertEqual(_kinds(m.check_identity(_good_identity())), ["log"])
        self.assertFalse(m.stopped)

    def test_logp_boundary_per_row_type(self):
        for v, stop in ((0.99e-3, False), (1e-3, True), (2e-3, True)):
            m = _mon()
            r = _good_identity()
            r["logp_max_abs_diff"]["play"] = v
            m.check_identity(r)
            self.assertEqual(m.stopped, stop, v)
        # bf16 tolerances per row type: wait may be looser without hiding a play-row problem
        th = M.Thresholds(identity_logp_tol={"wait": 5e-3, "play": 1e-3, "ability": 1e-3})
        m = M.Monitor(_base(), th)
        m.check_identity(_good_identity(logp_max_abs_diff={"wait": 4e-3, "play": 1e-5, "ability": 1e-5}))
        self.assertFalse(m.stopped)
        m = M.Monitor(_base(), th)
        acts = m.check_identity(_good_identity(backbone="bf16",
                                               logp_max_abs_diff={"wait": 4e-3, "play": 1.5e-3, "ability": 1e-5}))
        self.assertTrue(m.stopped)
        self.assertIn("fp32", acts[0].detail["hint"])

    def test_other_identity_failures(self):
        for kw in ({"ratio_max_abs_dev": 1e-3}, {"kl_ref_per_head": {"gate": 0.0, "cell": 1e-5}},
                   {"replay_bitwise_ok": False}, {"kl_ref_per_head": {}},
                   {"logp_max_abs_diff": {"wait": 1e-6, "play": 1e-6}}, {"ratio_max_abs_dev": float("nan")}):
            m = _mon()
            self.assertEqual(_kinds(m.check_identity(_good_identity(**kw))), ["stop"], kw)
        m = _mon()
        m.check_identity(_good_identity(ratio_max_abs_dev=0.99e-3,
                                        kl_ref_per_head={h: 0.99e-5 for h in M.IDENTITY_HEADS}))
        self.assertFalse(m.stopped)
        # aliases (gate / cell, as ppo.logprob and the run configuration name them) and a per-row-type ratio report
        m = _mon()
        m.check_identity(_good_identity(ratio_max_abs_dev={"wait": 1e-6, "play": -2e-6, "ability": 0.0},
                                        kl_ref_per_head={"gate": 0.0, "slot": 0.0, "cell": 0.0, "ability": 0.0}))
        self.assertFalse(m.stopped)
        # a missing head, a signed / negative value, a per-row-type ratio over tolerance all stop
        for kw in ({"kl_ref_per_head": {"family": 0.0, "slot": 0.0, "cells": 0.0}},
                   {"kl_ref_per_head": {h: -1e-3 for h in M.IDENTITY_HEADS}},
                   {"logp_max_abs_diff": {"wait": -5e-3, "play": 0.0, "ability": 0.0}},
                   {"ratio_max_abs_dev": -1.0},
                   {"ratio_max_abs_dev": {"wait": 0.0, "play": 2e-3, "ability": 0.0}},
                   {"replay_bitwise_ok": 1}, {"replay_bitwise_ok": "True"}):
            m = _mon()
            self.assertEqual(_kinds(m.check_identity(_good_identity(**kw))), ["stop"], kw)


class TestLrMonitors(unittest.TestCase):
    def _early(self, pattern):
        m = _mon()
        acts = []
        for i, es in enumerate(pattern):
            acts += m.observe_update({"update": i + 1, "early_stopped": es})
        return m, acts

    def test_early_stop_fraction(self):
        m, acts = self._early([True] * 6 + [False] * 2)              # exactly 75%: quiet
        self.assertEqual(_kinds(acts), [])
        self.assertEqual(m.lr_scale, 1.0)
        m, acts = self._early([True] * 7 + [False] * 1)
        self.assertEqual(_kinds(acts), ["lr_scale"])
        self.assertTrue(acts[0].temporary)
        self.assertAlmostEqual(m.lr, 3e-5)
        # not before the window is full, and the window is cleared after firing
        m, acts = self._early([True] * 7)
        self.assertEqual(_kinds(acts), [])
        m, acts = self._early([True] * 15)
        self.assertEqual(_kinds(acts), ["lr_scale"])
        # one active cut per cause until a clean G1 (no compounding into a self-made stall)
        m, acts = self._early([True] * 16)
        self.assertEqual(_kinds(acts), ["lr_scale", "log"])
        self.assertAlmostEqual(m.lr_scale, 0.6)
        m.observe_gate("g1", 25, 0.0)                                          # clean G1 restores
        acts = []
        for u in range(26, 36):
            acts += m.observe_update({"update": u, "early_stopped": True})
        self.assertEqual(_kinds(acts), ["lr_scale"])
        # losses.early_stop's tuple of tripped limits (empty = not stopped), numpy bools, 0/1
        m, acts = self._early([("joint_k3",)] * 7 + [()] * 1)
        self.assertEqual(_kinds(acts), ["lr_scale"])
        m, acts = self._early([("gate_kl",)] * 6 + [()] * 2)
        self.assertEqual(_kinds(acts), [])
        m, acts = self._early([1] * 7 + [0] * 1)
        self.assertEqual(_kinds(acts), ["lr_scale"])

    def test_clip_fraction(self):
        m = _mon()
        acts = []
        for u in range(1, 5):
            acts += m.observe_update({"update": u, "clip_frac": 0.4, "clip_frac_play": 0.4})
        self.assertEqual(acts, [])
        # joint and play-row clip fractions are two separate streaks (alternating highs never add up)
        m = _mon()
        acts = []
        for u, (j, p) in enumerate([(0.41, 0.1), (0.1, 0.5), (0.2, 0.41), (0.41, 0.1)], 1):
            acts += m.observe_update({"update": u, "clip_frac": j, "clip_frac_play": p})
        self.assertEqual(acts, [])
        for key, other in (("clip_frac", "play_clip_frac"), ("play_clip_frac", "clip_frac"),
                           ("clip_frac_play", "clip_frac")):
            m = _mon()
            acts = []
            for u in range(1, 5):
                acts += m.observe_update({"update": u, key: 0.41, other: 0.1})
            self.assertEqual(_kinds(acts), ["lr_scale"], key)
            self.assertEqual(acts[0].update, 4)
            self.assertTrue(acts[0].temporary)
        # a second trigger while the clip cut is active only logs; the floor caps all causes
        m = _mon()
        acts = []
        for u in range(1, 31):
            acts += m.observe_update({"update": u, "clip_frac": 0.5})
        self.assertEqual(_kinds(acts), ["lr_scale"] + ["log"] * 6)
        self.assertAlmostEqual(m.lr_scale, 0.6)
        m = _mon()
        acts = []
        for u, c in enumerate([0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5], 1):
            acts += m.observe_update({"update": u, "clip_frac": c})
        self.assertEqual(acts, [])

    def _kl_run(self, values, g1s=4, base_beta=0.02):
        m = M.Monitor(_base(), base_beta=base_beta)
        for i in range(g1s):
            m.observe_gate("g1", 20 * (i + 1), 0.0)
        acts = []
        for i, v in enumerate(values):
            acts += m.observe_update({"update": 100 + i, "kl_ref_total": v})
        return m, acts

    def test_kl_flat(self):
        flat = [0.10, 0.12, 0.08, 0.11, 0.10, 0.09, 0.12, 0.10]
        m, acts = self._kl_run(flat)
        self.assertEqual(_kinds(acts), ["beta_scale"])
        self.assertFalse(acts[0].temporary)
        self.assertAlmostEqual(m.beta, 0.012)
        m, acts = self._kl_run(flat + flat)              # one step per promotion-free stretch
        self.assertEqual(_kinds(acts), ["beta_scale"])
        m, acts = self._kl_run(flat[:-1] + [0.1211])     # 21.1% off the first value
        self.assertEqual(acts, [])
        m, acts = self._kl_run(flat, g1s=3)              # one G1 without promotion too few
        self.assertEqual(acts, [])
        m, acts = self._kl_run(flat[:7])                 # window not full
        self.assertEqual(acts, [])

    def test_kl_step_floor_then_lr(self):
        m, acts = self._kl_run([0.1] * 8, base_beta=0.008)
        self.assertAlmostEqual(m.beta, 0.005)            # 0.0048 clamped to the floor
        m, acts = self._kl_run([0.1] * 8, base_beta=0.005)
        # at the floor there is no automatic permanent lr change; an equal-games A/B fork from the backup
        # point is proposed instead (arm B at lr x kl_step_lr) and the lr stays
        self.assertEqual(_kinds(acts), ["propose_ab"])
        self.assertAlmostEqual(acts[0].value, 2.0)
        self.assertAlmostEqual(m.lr, 5e-5)
        self.assertAlmostEqual(acts[0].detail["arms"]["B"]["lr"], 1e-4)
        self.assertEqual(len(m.proposals), 1)
        self.assertFalse(acts[0].temporary)


class TestAlarms(unittest.TestCase):
    def _one(self, **rec):
        m = _mon()
        rec.setdefault("update", 1)
        return m.observe_update(rec)

    def test_alarm_boundaries(self):
        cases = [
            ("entropy", {"entropy": {"cell": 0.8}}, {"entropy": {"cell": 0.78}}),
            ("plays_per_min", {"plays_per_min": 8.0}, {"plays_per_min": 7.9}),
            ("plays_per_min", {"plays_per_min": 12.5}, {"plays_per_min": 12.6}),
            ("plays_per_min", {"abilities_per_min": 2.5}, {"abilities_per_min": 2.52}),
            ("card_rate", {"card_rate": {"card-c": 0.05}}, {"card_rate": {"card-c": 0.049}}),
            ("card_rate", {"card_rate": {"card-c": 0.80}}, {"card_rate": {"card-c": 0.801}}),
            ("forced_fraction", {"forced_fraction": 0.50}, {"forced_fraction": 0.501}),
            ("opponent_passive", {"opp_il_nonwait_per_game": 16.0}, {"opp_il_nonwait_per_game": 15.9}),
            ("throughput", {"games_per_hour": 2500.0}, {"games_per_hour": 2499.0}),
            ("drift", {"drift_alarm": False}, {"drift_alarm": ["slot NLL +0.05"]}),
            ("drop_rate", {"drop_rate": 0.01}, {"drop_rate": 0.0101}),
        ]
        for name, quiet, loud in cases:
            self.assertEqual(self._one(**quiet), [], (name, quiet))
            acts = self._one(**loud)
            self.assertEqual([(a.kind, a.monitor) for a in acts], [("alarm", name)], (name, loud))

    def test_missing_or_nan_metrics_are_skipped(self):
        self.assertEqual(self._one(entropy={"unknown_head": 0.0}, clip_frac=float("nan"),
                                   games_per_hour=None, card_rate={"unknown": 0.0}), [])
        m = M.Monitor(M.Baselines(core_cards=(GS,)))       # no references -> no ratios, one alarm says so
        acts = m.observe_update({"update": 1, "plays_per_min": 100.0, "card_rate": {GS: 0.0}})
        self.assertEqual([(a.kind, a.monitor) for a in acts], [("alarm", "baseline_missing")])
        self.assertIn(f"core_card_collapse[{GS}]", acts[0].detail["rules"])
        self.assertIn("mask_legal_refused", acts[0].detail["rules"])
        self.assertEqual(m.observe_update({"update": 2, "plays_per_min": 100.0}), [])   # once only


class TestStops(unittest.TestCase):
    def test_mask_audit(self):
        for rec, stop in (({"refusals_code19": 0}, False), ({"refusals_code19": 1}, True),
                          ({"masked_legal_refused_rate": 0.03}, False), ({"masked_legal_refused_rate": 0.0301}, True),
                          ({"mask_recompute_mismatch": 0}, False), ({"mask_recompute_mismatch": 1}, True)):
            m = _mon()
            acts = m.observe_update(dict(rec, update=5))
            self.assertEqual(m.stopped, stop, rec)
            self.assertEqual(_kinds(acts), ["stop"] if stop else [], rec)

    def test_drop_rate(self):
        m = _mon()
        self.assertEqual(_kinds(m.observe_update({"update": 1, "drop_rate": 0.05})), ["alarm"])
        self.assertFalse(m.stopped)
        acts = m.observe_update({"update": 2, "drop_rate": 0.0501, "drop_by_reason": {"crash": 3}})
        self.assertEqual(_kinds(acts), ["stop"])
        self.assertEqual(acts[0].detail["by_reason"], {"crash": 3})
        self.assertTrue(m.stopped)
        self.assertIn("drop_rate", m.stop_reason)
        # a stopped monitor does nothing more
        self.assertEqual(m.observe_update({"update": 3, "refusals_code19": 9}), [])
        self.assertEqual(m.observe_gate("g1", 20, -0.1), [])

    def test_nonfinite(self):
        m = _mon()
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(m.observe_minibatch(1, True), [])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertEqual(_kinds(m.observe_minibatch(1, False)), ["skip_minibatch"])
        self.assertFalse(m.stopped)
        self.assertEqual(_kinds(m.observe_minibatch(2, False)), ["skip_minibatch", "stop"])
        self.assertTrue(m.stopped)


class TestSystem(unittest.TestCase):
    def test_free_memory(self):
        m = _mon()
        self.assertEqual(m.observe_system(0, mem_free_gb=4.0), [])
        acts = m.observe_system(10, mem_free_gb=3.99)
        self.assertEqual(_kinds(acts), ["reduce_slots"])
        self.assertTrue(acts[0].detail["pause_new_games"])
        acts = m.observe_system(20, mem_free_gb=5.0)
        self.assertEqual(_kinds(acts), ["log"])
        self.assertFalse(acts[0].detail["pause_new_games"])
        self.assertEqual(m.observe_system(30, mem_free_gb=5.0), [])

    def test_swap_growth(self):
        m = _mon()
        self.assertEqual(m.observe_system(0, swap_used_gb=1.0), [])
        self.assertEqual(m.observe_system(1800, swap_used_gb=1.6), [])
        self.assertEqual(m.observe_system(3600, swap_used_gb=2.0), [])          # exactly +1 GB
        acts = m.observe_system(3601, swap_used_gb=2.61)                         # +1.01 over the 1800 s sample
        self.assertEqual(_kinds(acts), ["reduce_slots"])
        self.assertEqual(acts[0].value, 4.0)
        self.assertEqual(m.observe_system(3700, swap_used_gb=3.4), [])           # baseline restarted
        m = _mon()
        m.observe_system(0, swap_used_gb=1.0)
        self.assertEqual(m.observe_system(3601, swap_used_gb=2.2), [])           # slow growth over > 60 min
        self.assertEqual(_kinds(m.observe_system(3610, swap_used_gb=3.4)), ["reduce_slots"])


class TestCoreCardAndEv(unittest.TestCase):
    def test_core_card_rollback_then_stop(self):
        m = _mon()
        acts = []
        for u in range(1, 5):
            acts += m.observe_update({"update": u, "card_rate": {GS: 0.02}})       # exactly 0.2x
        self.assertEqual(_kinds(acts, "core_card_collapse"), [])
        acts = []
        for u, r in zip(range(5, 12), [0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01]):   # streak broken
            acts += m.observe_update({"update": u, "card_rate": {GS: r}})
        self.assertEqual(_kinds(acts, "core_card_collapse"), [])
        acts = m.observe_update({"update": 12, "card_rate": {GS: 0.01}})
        self.assertEqual(_kinds(acts, "core_card_collapse"), ["rollback", "lr_scale"])
        self.assertTrue(acts[[a.kind for a in acts].index("lr_scale")].temporary)
        self.assertEqual(m.flags, ["core_card_collapse@12"])
        self.assertAlmostEqual(m.lr_scale, 0.6)
        self.assertFalse(m.stopped)
        acts = []
        for u in range(13, 17):                                                   # second collapse: rollback
            acts += m.observe_update({"update": u, "card_rate": {GS: 0.01}})
        self.assertEqual(_kinds(acts, "core_card_collapse"), ["rollback", "lr_scale", "beta_scale"])
        self.assertFalse(m.stopped)
        acts = []
        for u in range(17, 21):
            acts += m.observe_update({"update": u, "card_rate": {GS: 0.01}})
        self.assertIn("stop", _kinds(acts, "core_card_collapse"))
        self.assertNotIn("rollback", _kinds(acts))
        self.assertTrue(m.stopped)

    def test_rollback_resets_lr_windows(self):
        m = _mon()
        for u in (1, 2, 3):
            m.observe_update({"update": u, "clip_frac": 0.5, "card_rate": {GS: 0.01}})
        acts = m.observe_update({"update": 4, "clip_frac": 0.5, "card_rate": {GS: 0.01}})
        self.assertEqual(_kinds(acts), ["rollback", "lr_scale", "alarm"])   # no extra clip cut
        self.assertAlmostEqual(m.lr_scale, 0.6)

    def test_ev_to_f1(self):
        for warm, recs, fire in (
            (None, [{"ev_overall": 0.1, "ev_last_third": 0.2}] * 4, False),
            (None, [{"ev_overall": 0.099, "ev_last_third": 0.5}] * 4, True),
            (None, [{"ev_overall": 0.4, "ev_last_third": 0.199}] * 4, True),
            (0.40, [{"ev_overall": 0.20}] * 4, False),                       # exactly warm-up end - 0.2
            (0.40, [{"ev_overall": 0.199}] * 4, True),
            (None, [{"ev_overall": 0.05}] * 3 + [{"ev_overall": 0.5}] + [{"ev_overall": 0.05}] * 3, False),
            (None, [{"ev_overall": 0.05}] * 3, False),
        ):
            m = M.Monitor(_base(ev_warmup_end=warm))
            acts = []
            for u, r in enumerate(recs, 1):
                acts += m.observe_update(dict(r, update=u))
            acts = [a for a in acts if a.monitor != "baseline_missing"]       # warm-up end None is listed
            self.assertEqual(_kinds(acts), ["enter_f1"] if fire else [], (warm, recs))
        m = _mon()
        acts = []
        for u in range(1, 10):
            acts += m.observe_update({"update": u, "ev_overall": 0.0})
        self.assertEqual(_kinds(acts), ["enter_f1"])                           # entered once
        self.assertTrue(m.in_f1)


class TestGates(unittest.TestCase):
    def test_clean_g1_restores_temporaries(self):
        m = _mon()
        for u in range(1, 5):
            m.observe_update({"update": u, "clip_frac": 0.5})
        self.assertAlmostEqual(m.lr_scale, 0.6)
        self.assertEqual(m.observe_gate("screen", 10, 0.05), [])               # a screen never restores
        self.assertEqual(_kinds(m.observe_gate("g1", 20, -0.001)), [])         # not clean
        self.assertAlmostEqual(m.lr_scale, 0.6)
        acts = m.observe_gate("g1", 40, 0.0)                                    # clean: delta_hat >= 0
        self.assertEqual(_kinds(acts, "clean_g1_restore"), ["lr_scale"])
        self.assertAlmostEqual(acts[0].value, 1 / 0.6)
        self.assertTrue(acts[0].detail["restore"])
        self.assertAlmostEqual(m.lr, 5e-5)

    def test_permanent_changes_survive_clean_g1(self):
        m = _mon()
        for u in (20, 40, 60, 80):
            m.observe_gate("g1", u, -0.01)
        for u in range(81, 89):
            m.observe_update({"update": u, "kl_ref_total": 0.1, "clip_frac": 0.5})
        self.assertAlmostEqual(m.beta, 0.012)
        self.assertLess(m.lr_scale, 1.0)
        m.observe_gate("g1", 100, 0.04, promoted=True)
        self.assertAlmostEqual(m.beta, 0.012)                                   # KL step is permanent
        self.assertAlmostEqual(m.lr_scale, 1.0)

    def test_rollback_trigger(self):
        m = _mon()
        self.assertEqual(m.observe_gate("screen", 10, -0.05), [])
        self.assertEqual(m.observe_gate("screen", 15, -0.05), [])
        acts = m.observe_gate("g1", 20, -0.05)
        self.assertEqual(_kinds(acts, "gate_rollback"), ["rollback", "lr_scale"])
        self.assertEqual(acts[0].detail["target"], "champion")
        self.assertAlmostEqual(m.lr_scale, 0.6)
        for seq in ([-0.05, -0.05, -0.0499], [-0.06, -0.06, 0.0, -0.06], [-0.0499, -0.0499, -0.0499]):
            m = _mon()
            acts = []
            for i, d in enumerate(seq):
                acts += m.observe_gate("screen", 10 * (i + 1), d)
            self.assertNotIn("rollback", _kinds(acts), seq)
        m = _mon()                                                              # invalid gate ignored
        m.observe_gate("screen", 10, -0.1)
        m.observe_gate("screen", 15, -0.1)
        self.assertEqual(_kinds(m.observe_gate("g1", 20, -0.1, valid=False)), ["log"])
        self.assertIn("rollback", _kinds(m.observe_gate("screen", 30, -0.1)))

    def test_double_rollback_beta(self):
        def run(gap):
            m = _mon()
            m.observe_gate("screen", 10, -0.06)
            m.observe_gate("screen", 15, -0.06)
            m.observe_gate("g1", 20, -0.06)                                     # rollback 1 at 20
            m.observe_gate("screen", 20 + gap - 2, -0.06)
            m.observe_gate("screen", 20 + gap - 1, -0.06)
            acts = m.observe_gate("screen", 20 + gap, -0.06)                    # rollback 2
            return m, acts
        m, acts = run(40)
        self.assertEqual(_kinds(acts), ["rollback", "lr_scale", "beta_scale"])
        self.assertTrue(acts[2].temporary)
        self.assertAlmostEqual(m.beta, 0.06)
        self.assertAlmostEqual(m.lr_scale, 0.4)
        acts = m.observe_gate("g1", 80, 0.02)
        self.assertEqual(sorted(_kinds(acts, "clean_g1_restore")), ["beta_scale", "lr_scale"])
        self.assertAlmostEqual(m.beta, 0.02)
        self.assertAlmostEqual(m.lr_scale, 1.0)
        m, acts = run(41)
        self.assertEqual(_kinds(acts), ["rollback", "lr_scale"])
        self.assertAlmostEqual(m.beta, 0.02)

    def test_beta_cap(self):
        m = M.Monitor(_base(), base_beta=0.08)
        for u in (10, 20, 30, 40, 50, 60):
            m.observe_gate("screen", u, -0.06)
        self.assertAlmostEqual(m.beta, 0.1)
        m.observe_gate("g1", 80, 0.0)
        self.assertAlmostEqual(m.beta, 0.08)

    def test_stall_ladder_regular_cadence(self):
        m = _mon()
        steps, stop_at, f1 = {}, None, None
        for u in range(20, 181, 20):
            acts = m.observe_gate("g1", u, -0.01)
            for a in acts:
                if a.kind == "stall_step":
                    steps[u] = int(a.value)
                if a.kind == "stop":
                    stop_at = u
                if a.kind == "enter_f1":
                    f1 = u
            if u == 100:
                self.assertIn("beta_scale", _kinds(acts))                       # step 1 = KL step
        self.assertEqual(steps, {100: 1, 120: 2, 140: 3})
        self.assertEqual(f1, 140)
        self.assertEqual(stop_at, 160)
        self.assertIn("stall_final_stop", m.stop_reason)

    def test_stall_ladder_min_updates(self):
        m = _mon()
        for u in (20, 40, 60, 80):
            self.assertEqual(_kinds(m.observe_gate("g1", u, -0.01)), [])
        self.assertEqual(_kinds(m.observe_gate("g1", 100, -0.01))[0], "stall_step")
        self.assertEqual(_kinds(m.observe_gate("g1", 109, -0.01)), [])           # 9 updates after step 1
        acts = m.observe_gate("g1", 110, -0.01)                                  # 10 after step 1
        self.assertEqual((acts[0].kind, acts[0].value), ("stall_step", 2.0))
        self.assertEqual(_kinds(m.observe_gate("g1", 130, -0.01)), ["stop"])    # 8th G1 without promotion

    def test_promotion_resets_and_release_candidate(self):
        m = _mon()
        for u in (20, 40, 60, 80, 100):
            m.observe_gate("g1", u, -0.01)
        self.assertEqual(m.stall_step, 1)
        acts = m.observe_gate("g1", 120, 0.05, promoted=True)
        self.assertEqual((m.g1_since_promo, m.stall_step), (0, 0))
        self.assertEqual(_kinds(acts), ["log"])
        for u in (140, 160, 180, 200):
            self.assertEqual(m.observe_gate("g1", u, 0.0), [])
        acts = m.observe_gate("g1", 220, 0.0)
        self.assertEqual(_kinds(acts), ["log", "stall_step", "beta_scale"])
        self.assertTrue(acts[0].detail["release_candidate"])
        self.assertTrue(m.release_candidate)
        with self.assertRaises(ValueError):
            m.observe_gate("screen", 230, 0.1, promoted=True)
        with self.assertRaises(ValueError):
            m.observe_gate("g2", 230, 0.1)


class TestPersistenceAndDashboard(unittest.TestCase):
    def test_state_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            m = _mon()
            for u in (1, 2, 3):
                m.observe_update({"update": u, "clip_frac": 0.5, "kl_ref_total": 0.1, "early_stopped": True})
            m.observe_system(0, swap_used_gb=1.0)
            m.observe_gate("screen", 2, -0.05)
            m.observe_gate("screen", 3, -0.05)
            m.save(Path(td) / "monitor_state.json")
            d = json.loads((Path(td) / "monitor_state.json").read_text())
            m2 = M.Monitor.from_state(d, _base())
            self.assertEqual(m2.state_dict(), m.state_dict())
            acts = m2.observe_update({"update": 4, "clip_frac": 0.5})
            self.assertEqual(_kinds(acts), ["lr_scale"])
            self.assertIn("rollback", _kinds(m2.observe_gate("g1", 20, -0.05)))
            self.assertEqual(_kinds(m2.observe_system(100, swap_used_gb=2.1)), ["reduce_slots"])

    def test_dashboard_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dash" / "ppo.jsonl"
            m = M.Monitor(_base(), dashboard_path=path, clock=lambda: "1970-01-01T00:00:00Z")
            m.check_identity(_good_identity())
            m.observe_update({"update": 1, "drop_rate": 0.02, "clip_frac": float("nan")})
            m.observe_minibatch(1, False)
            m.observe_gate("g1", 20, 0.01)
            recs = M.read_dashboard(path)
            self.assertEqual([r["kind"] for r in recs], ["identity", "update", "minibatch", "gate"])
            self.assertEqual(recs[1]["actions"][0]["kind"], "alarm")
            self.assertEqual(recs[1]["metrics"]["clip_frac"], "nan")
            for r in recs:
                for k in ("ts", "lr", "beta", "stopped", "flags", "g1_since_promo"):
                    self.assertIn(k, r)

    def test_stop_marker(self):
        with tempfile.TemporaryDirectory() as td:
            p = M.write_stop_marker(Path(td) / "run", "drop rate", 7, [M.Action("stop", "drop_rate", "x", 7)])
            self.assertEqual(p.name, M.STOP_MARKER)
            self.assertEqual(json.loads(p.read_text())["update"], 7)

    def test_thresholds_from_mapping(self):
        th = M.Thresholds.from_mapping({"clip_max": 0.25, "plays_ratio": [0.7, 1.6]})
        self.assertEqual(th.plays_ratio, (0.7, 1.6))
        with self.assertRaises(KeyError):
            M.Thresholds.from_mapping({"nope": 1})
        b = M.Baselines.from_mapping({"core_cards": [GS, "card-b"]})
        self.assertEqual(b.core_cards, (GS, "card-b"))


if __name__ == "__main__":
    unittest.main()
