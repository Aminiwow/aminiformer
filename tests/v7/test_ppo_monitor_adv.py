"""Adversarial tests for src/amini/ppo/monitor.py.

Three kinds of test:

* independent checks (must pass): an executor that applies every ``lr_scale`` / ``beta_scale`` action
  value reproduces ``Monitor.lr`` / ``Monitor.beta`` over random event streams; a save / restore in the
  middle of a random stream gives the same actions and state as the uninterrupted monitor; ladder steps
  keep their minimum spacing and the final stop lands on the ``final_stop_g1``-th non-promoting G1 for
  random G1 spacings; a stop is absorbing;
* ``test_BUG_*``: regression tests, each for a defect found in review (behaviour the module docstring
  or the sibling losses module requires);
* ``test_DOC_*`` (pass): pin behaviour the learner relies on, so a change is deliberate (baseline alarm,
  one KL step, temporary-lr cap, signed identity values, separate clip streaks,
  invalid gates).

The numbers follow the example defaults of ``monitor.Thresholds``.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_monitor_adv.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
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

try:
    import numpy as np
except ImportError:  # pragma: no cover - the pinned runtime has numpy
    np = None
try:
    import torch
    from amini.ppo import losses as L
except ImportError:  # pragma: no cover - torch is optional
    torch = None
    L = None

GS = "card-a"
NAN = float("nan")


def _base(**kw) -> M.Baselines:
    d = dict(il_entropy={"cell": 2.0, "gate": 0.5, "slot": 1.0, "ability": 0.7},
             il_plays_per_min=10.0, il_abilities_per_min=2.0,
             il_card_rate={GS: 0.10, "card-b": 0.10, "card-c": 0.20}, core_cards=(GS,),
             il_masked_legal_refused_rate=0.01, il_forced_fraction=0.30, p3_opp_nonwait_per_game=40.0,
             m2_games_per_hour=5000.0, ev_warmup_end=0.40)
    d.update(kw)
    return M.Baselines(**d)


def _good_identity(**kw):
    r = {"logp_max_abs_diff": {"wait": 5e-6, "play": 1.3e-5, "ability": 8.6e-6}, "ratio_max_abs_dev": 1e-6,
         "kl_ref_per_head": {h: 0.0 for h in M.IDENTITY_HEADS}, "replay_bitwise_ok": True}
    r.update(kw)
    return r


def _kinds(acts, monitor=None):
    return [a.kind for a in acts if monitor is None or a.monitor == monitor]


# ------------------------------------------------------------------ random event streams
def _random_events(seed: int, n_updates: int = 160):
    """A plausible stream: one update record per update, a screen at 10 mod 20, a G1 at 0 mod 20."""
    rng = random.Random(seed)
    ev = []
    for u in range(1, n_updates + 1):
        rec = {"update": u}
        if rng.random() < 0.8:
            rec["early_stopped"] = rng.random() < 0.6
        if rng.random() < 0.8:
            rec["clip_frac"] = rng.choice([0.1, 0.35, 0.5])
            rec["clip_frac_play"] = rng.choice([0.1, 0.2, 0.5])
        rec["kl_ref_total"] = 0.10 * (1.0 + rng.uniform(-0.08, 0.08))
        if rng.random() < 0.5:
            rec["card_rate"] = {GS: rng.choice([0.01, 0.05, 0.1])}
        if rng.random() < 0.3:
            rec["ev_overall"] = rng.choice([0.05, 0.3])
        ev.append(("update", rec))
        if u % 20 == 10:
            ev.append(("gate", ("screen", u, rng.choice([-0.05, -0.02, 0.0, 0.03]), False)))
        if u % 20 == 0:
            d = rng.choice([-0.05, -0.01, 0.0, 0.02, 0.04])
            ev.append(("gate", ("g1", u, d, d >= 0.04 and rng.random() < 0.5)))
        if rng.random() < 0.2:
            ev.append(("system", (u * 600.0, rng.choice([3.0, 6.0]), 1.0 + 0.02 * u)))
        if rng.random() < 0.1:
            ev.append(("mb", (u, rng.random() < 0.7)))
    return ev


def _apply(m: M.Monitor, e):
    kind, x = e
    if kind == "update":
        return m.observe_update(x)
    if kind == "gate":
        k, u, d, promo = x
        return m.observe_gate(k, u, d, promoted=promo)
    if kind == "system":
        t, mem, sw = x
        return m.observe_system(t, mem_free_gb=mem, swap_used_gb=sw)
    u, fin = x
    return m.observe_minibatch(u, fin)


# ------------------------------------------------------------------ independent checks
class TestInvariants(unittest.TestCase):
    def test_action_values_reproduce_lr_and_beta(self):
        """An executor that multiplies the lr / beta by each action's value tracks Monitor.lr / beta."""
        for seed in range(25):
            for base_beta in (0.02, 0.08, 0.005):
                m = M.Monitor(_base(), base_beta=base_beta)
                lr, beta = m.base_lr, base_beta
                for e in _random_events(seed):
                    for a in _apply(m, e):
                        if a.kind == "lr_scale":
                            lr *= a.value
                        elif a.kind == "beta_scale":
                            beta *= a.value
                    self.assertAlmostEqual(lr / m.lr, 1.0, places=12, msg=(seed, base_beta, e))
                    self.assertAlmostEqual(beta / m.beta, 1.0, places=12, msg=(seed, base_beta, e))
                    self.assertGreaterEqual(m.beta, 0.005 - 1e-15)
                    self.assertLessEqual(m.beta, 0.1 + 1e-15)
                    if m.stopped:
                        break

    def test_restore_mid_stream_is_seamless(self):
        """save() at a random point + from_state() continues with exactly the same actions and state."""
        for seed in range(25):
            ev = _random_events(seed)
            cut = random.Random(1000 + seed).randrange(1, len(ev) - 1)
            a = M.Monitor(_base())
            for e in ev[:cut]:
                _apply(a, e)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "monitor_state.json"
                a.save(p)
                b = M.Monitor.from_state(json.loads(p.read_text()), _base())
            for e in ev[cut:]:
                ra = [x.to_json() for x in _apply(a, e)]
                rb = [x.to_json() for x in _apply(b, e)]
                self.assertEqual(ra, rb, (seed, cut, e))
            self.assertEqual(a.state_dict(), b.state_dict(), seed)

    def test_ladder_spacing_and_final_stop(self):
        """Random G1 spacing: steps >= 10 updates apart, stop exactly at the 8th non-promoting G1."""
        for seed in range(50):
            rng = random.Random(seed)
            m = M.Monitor(_base())
            u, n, step_updates, stop_n = 0, 0, [], None
            while not m.stopped:
                u += rng.randint(5, 35)
                n += 1
                for a in m.observe_gate("g1", u, -0.01):
                    if a.kind == "stall_step":
                        step_updates.append(u)
                    if a.kind == "stop":
                        stop_n = n
            self.assertEqual(stop_n, 8, seed)
            self.assertTrue(all(b - a >= 10 for a, b in zip(step_updates, step_updates[1:])), (seed, step_updates))
            self.assertLessEqual(len(step_updates), 3, seed)

    def test_stop_is_absorbing(self):
        m = M.Monitor(_base())
        m.observe_update({"update": 1, "refusals_code19": 1})
        self.assertTrue(m.stopped)
        self.assertEqual(m.observe_update({"update": 2, "drop_rate": 0.5, "clip_frac": 0.9}), [])
        self.assertEqual(m.observe_gate("g1", 20, -0.5), [])
        self.assertEqual(m.observe_system(0, mem_free_gb=0.1, swap_used_gb=50.0), [])
        self.assertEqual(m.observe_minibatch(3, False), [])
        self.assertIn("mask_illegal_cell", m.stop_reason)

    def test_rollback_on_float_noisy_threshold(self):
        """delta_hat computed from win rates (0.45 - 0.50 = -0.04999999999999999) counts as <= -5 pp."""
        m = M.Monitor(_base())
        m.observe_gate("screen", 10, 0.45 - 0.50)
        m.observe_gate("screen", 15, 0.40 - 0.45)
        self.assertIn("rollback", _kinds(m.observe_gate("g1", 20, 0.50 - 0.55)))


# ------------------------------------------------------------------ monitor defects
class TestMonitorBugs(unittest.TestCase):
    def test_BUG_identity_passes_with_missing_heads(self):
        """Update-0 identity: KL(theta0 || ref) < tol for EVERY head. check_identity only loops over the heads
        that are present (IDENTITY_HEADS is never used), so a report with one head passes."""
        m = M.Monitor(_base())
        acts = m.check_identity(_good_identity(kl_ref_per_head={"gate": 0.0}))
        self.assertEqual(_kinds(acts), ["stop"])

    def test_BUG_early_stopped_truthy_non_bool_ignored(self):
        """isinstance(es, bool) drops np.bool_, 0/1 and losses.early_stop's tuple: the lr cut never fires."""
        results = []
        values = [1, ("joint_k3",)] + ([np.bool_(True)] if np is not None else [])
        for v in values:
            m = M.Monitor(_base())
            acts = []
            for u in range(1, 11):
                acts += m.observe_update({"update": u, "early_stopped": v})
            results.append(_kinds(acts))
        self.assertEqual(results, [["lr_scale"]] * len(values))

    @unittest.skipIf(np is None, "numpy missing")
    def test_BUG_numpy_bool_replay_ok_stops_run(self):
        """replay_bitwise_ok is np.bool_(True) when computed as (a == b).all(): `is not True` stops."""
        m = M.Monitor(_base())
        self.assertEqual(_kinds(m.check_identity(_good_identity(replay_bitwise_ok=np.bool_(True)))), ["log"])

    def test_BUG_nan_in_action_detail_crashes_dashboard(self):
        """ev_by_opponent with NaN (an opponent type with too few rows) goes into Action.detail, which is
        not passed through _jsonable; json.dumps(allow_nan=False) raises AFTER in_f1 was set, so the
        enter_f1 action is lost for good."""
        with tempfile.TemporaryDirectory() as td:
            m = M.Monitor(_base(), dashboard_path=Path(td) / "d.jsonl")
            acts = []
            try:
                for u in range(1, 5):
                    acts += m.observe_update({"update": u, "ev_overall": 0.05, "ev_by_opponent": {"il": NAN}})
            finally:
                in_f1 = m.in_f1
            self.assertTrue(in_f1)
            self.assertIn("enter_f1", _kinds(acts))

    @unittest.skipIf(np is None, "numpy missing")
    def test_BUG_numpy_counts_in_drop_alarm_crash_dashboard(self):
        """An alarm-only condition (drop 2%) with numpy drop counts raises TypeError in _record."""
        with tempfile.TemporaryDirectory() as td:
            m = M.Monitor(_base(), dashboard_path=Path(td) / "d.jsonl")
            acts = m.observe_update({"update": 1, "drop_rate": 0.02, "drop_by_reason": {"crash": np.int64(3)}})
            self.assertEqual(_kinds(acts), ["alarm"])

    @unittest.skipIf(np is None, "numpy missing")
    def test_BUG_numpy_update_index_breaks_save(self):
        """observe_gate / observe_minibatch / observe_system keep `update` as given; an np.int64 lands in
        rollback_updates and last_update, and save() (the block hand-off) raises TypeError."""
        m = M.Monitor(_base())
        m.observe_gate("screen", np.int64(10), -0.05)
        m.observe_gate("screen", np.int64(15), -0.05)
        self.assertIn("rollback", _kinds(m.observe_gate("screen", np.int64(20), -0.05)))
        with tempfile.TemporaryDirectory() as td:
            m.save(Path(td) / "monitor_state.json")

    def test_BUG_zero_il_reference_turns_off_masked_refusal_stop(self):
        """Masked-legal-but-refused > masked_refusal_ratio x IL -> stop. With an IL rate of 0 every positive
        rate is above that, but _ratio returns None for a non-positive reference and the stop rule is silently off."""
        m = M.Monitor(_base(il_masked_legal_refused_rate=0.0))
        m.observe_update({"update": 1, "masked_legal_refused_rate": 0.05})
        self.assertTrue(m.stopped)

    def test_BUG_core_card_streak_mixes_cards(self):
        """'core card < core_ratio x IL for core_window consecutive updates' is per card. With two core
        cards a single streak counter rolls back when the two cards take turns being low."""
        m = M.Monitor(_base(core_cards=(GS, "card-b")))
        acts = []
        for u, (g, f) in enumerate([(0.01, 0.10), (0.10, 0.01), (0.01, 0.10), (0.10, 0.01)], 1):
            acts += m.observe_update({"update": u, "card_rate": {GS: g, "card-b": f}})
        self.assertNotIn("rollback", _kinds(acts))

    def test_BUG_swap_window_keeps_samples_from_the_future(self):
        """observe_system prunes only ts < t - swap_window_s; samples with ts > t (a restored state whose clock
        restarted, e.g. seconds since block start) stay as the baseline and give a false reduce_slots."""
        m = M.Monitor(_base())
        for t in range(0, 3601, 600):
            m.observe_system(t, swap_used_gb=1.0)
        m2 = M.Monitor.from_state(json.loads(json.dumps(m.state_dict())), _base())
        # next block, clock restarted; swap grew 1.2 GB over hours (slow, not a growth within the window)
        self.assertEqual(m2.observe_system(0, swap_used_gb=2.2), [])

    def test_BUG_default_thresholds_are_shared_and_mutable(self):
        """Monitor()'s default Thresholds() instance is created once at def time; its dict field is shared
        by every default Monitor (and makes the frozen dataclass unhashable)."""
        m1 = M.Monitor()
        with self.assertRaises(TypeError):
            m1.th.identity_logp_tol["play"] = 1.0                  # read-only mapping field
        with self.assertRaises(TypeError):
            m1.base.il_card_rate[GS] = 1.0
        self.assertIsNot(m1.th, M.Monitor().th)
        self.assertEqual(M.Monitor().th.identity_logp_tol["play"], 1e-3)
        hash(M.Thresholds())
        hash(_base())
        self.assertEqual(M.Thresholds.from_mapping({"identity_logp_tol": {"wait": 1e-3, "play": 1e-3,
                                                                          "ability": 1e-3}}), M.Thresholds())


@unittest.skipIf(torch is None, "needs torch (amini.ppo.losses)")
class TestSiblingModuleContract(unittest.TestCase):
    """The learner will feed losses.update_metrics / losses.early_stop output to observe_update.
    Missing keys are skipped silently, so a name mismatch switches a monitor off."""

    def _metrics(self):
        n = 8
        logp_prox = torch.zeros(n, dtype=torch.float64)
        logp_new = torch.tensor([1.0] * 4 + [0.0] * 4, dtype=torch.float64)    # play rows: r = e, clipped
        weight = torch.tensor([1.0] * 4 + [9.0] * 4, dtype=torch.float64)      # joint clip share 0.1
        play = torch.tensor([True] * 4 + [False] * 4)
        ent = {"cell": (torch.full((4,), 0.1, dtype=torch.float64), torch.ones(4, dtype=torch.float64))}
        return L.update_metrics(logp_new, logp_prox, weight, play, entropy=ent)

    def test_metrics_shape(self):
        mt = self._metrics()
        self.assertAlmostEqual(mt["clip_frac"], 0.1)
        self.assertAlmostEqual(mt["play_clip_frac"], 1.0)
        self.assertAlmostEqual(mt["entropy/cells"], 0.1)          # losses canonicalises cell -> cells
        self.assertNotIn("clip_frac_play", mt)
        self.assertNotIn("entropy", mt)
        # fixed: the monitor's head vocabulary is a copy of losses' (monitor stays torch-free)
        self.assertEqual(L.HEADS, ("family", "slot", "cells", "ability"))
        self.assertEqual(tuple(M.IDENTITY_HEADS), tuple(L.HEADS))
        self.assertEqual(dict(M.HEAD_ALIASES), dict(L.HEAD_ALIASES))

    def test_real_identity_heads_pass(self):
        """An identity report keyed by losses.HEADS (or the logprob kl_* aliases) passes."""
        for keys in (L.HEADS, ("kl_family", "kl_slot", "kl_cell", "kl_ability")):
            m = M.Monitor(_base())
            self.assertEqual(_kinds(m.check_identity(_good_identity(kl_ref_per_head={h: 0.0 for h in keys}))),
                             ["log"], keys)

    def test_BUG_play_row_clip_from_update_metrics_is_ignored(self):
        """Play-row clip fraction > clip_max for clip_window updates -> lr cut. losses names it play_clip_frac,
        monitor reads clip_frac_play."""
        m = M.Monitor(_base())
        acts = []
        for u in range(1, 5):
            acts += m.observe_update(dict(self._metrics(), update=u))
        self.assertEqual(_kinds(acts, "clip_fraction"), ["lr_scale"])

    def test_BUG_entropy_from_update_metrics_is_ignored(self):
        """Entropy alarm (< entropy_min_ratio x IL): losses emits flat 'entropy/<head>' keys, monitor reads a
        nested 'entropy' mapping."""
        m = M.Monitor(_base(il_entropy={"cells": 2.0, "cell": 2.0}))
        acts = m.observe_update(dict(self._metrics(), update=1))
        self.assertEqual(_kinds(acts, "entropy"), ["alarm"])

    def test_BUG_early_stop_tuple_is_ignored(self):
        """losses.early_stop returns the tuple of tripped limits; observe_update drops anything not bool."""
        tripped = L.early_stop({"joint_k3": 0.05, "play_k3": 0.0, "gate_kl": 0.0})
        self.assertEqual(tripped, ("joint_k3",))
        m = M.Monitor(_base())
        acts = []
        for u in range(1, 11):
            acts += m.observe_update({"update": u, "early_stopped": tripped})
        self.assertEqual(_kinds(acts), ["lr_scale"])


class TestStatsContract(unittest.TestCase):
    def test_DOC_invalid_gate_is_ignored_by_monitor(self):
        """An invalid gate (incomplete) is replayed, so the replay is the next gate of the run: the monitor
        ignores the invalid one, and bad gates with an invalid one between them still roll back
        (stats.rollback_triggered skips None the same way; this test does not call stats)."""
        m = M.Monitor(_base())
        m.observe_gate("screen", 10, -0.10)
        m.observe_gate("screen", 15, -0.10)
        self.assertEqual(_kinds(m.observe_gate("g1", 20, -0.10, valid=False)), ["log"])
        self.assertIn("rollback", _kinds(m.observe_gate("screen", 30, -0.10)))


# ------------------------------------------------------------------ pinned risky behaviour
class TestMonitorDoc(unittest.TestCase):
    def test_DOC_default_baselines_say_which_rules_are_off(self):
        """Baselines without reference rates (only a core card configured): the ratio rules cannot run,
        but one baseline_missing alarm lists them and every dashboard record carries the list."""
        with tempfile.TemporaryDirectory() as td:
            m = M.Monitor(M.Baselines(core_cards=(GS,)), dashboard_path=Path(td) / "d.jsonl")
            acts = []
            for u in range(1, 7):
                acts += m.observe_update({"update": u, "card_rate": {GS: 0.0}, "masked_legal_refused_rate": 0.9,
                                          "plays_per_min": 100.0, "games_per_hour": 1.0})
            self.assertEqual([(a.kind, a.monitor, a.update) for a in acts], [("alarm", "baseline_missing", 1)])
            rules = acts[0].detail["rules"]
            for r in (f"core_card_collapse[{GS}]", "mask_legal_refused", "plays_per_min", "throughput"):
                self.assertIn(r, rules)
            self.assertFalse(m.stopped)
            recs = M.read_dashboard(Path(td) / "d.jsonl")
            self.assertEqual(len(recs), 6)
            self.assertTrue(all(r["disabled_rules"] == rules for r in recs))
            # the alarm is part of the state: a restored monitor does not repeat it
            m2 = M.Monitor.from_state(m.state_dict())
            self.assertEqual(m2.observe_update({"update": 7}), [])

    def test_DOC_kl_flat_step_and_ladder_step_1_share_one_kl_step(self):
        """KL flat (after kl_flat_g1 non-promoting G1) lowers beta at update 88; ladder step 1 at update 100 is
        the same KL step (one step per stretch), so it is logged as a stall step without a second beta change."""
        m = M.Monitor(_base())
        for u in (20, 40, 60, 80):
            m.observe_gate("g1", u, -0.01)
        steps = []
        for u in range(81, 89):
            for a in m.observe_update({"update": u, "kl_ref_total": 0.1}):
                steps.append((u, a.kind, round(m.beta, 6)))
        g = m.observe_gate("g1", 100, -0.01)
        for a in g:
            if a.kind == "beta_scale":
                steps.append((100, a.kind, round(m.beta, 6)))
        self.assertEqual(steps, [(88, "beta_scale", 0.012)])
        self.assertEqual(_kinds(g), ["stall_step"])
        self.assertEqual(g[0].detail["kl_step_already_applied_at"], 88)
        self.assertAlmostEqual(m.beta, 0.012)

    def test_DOC_temporary_lr_capped(self):
        """Clip above clip_max every update: one active cut per cause (temp_lr_factor after 30 updates, no G1
        between); all temporary causes together stop at the floor TEMP_LR_FLOOR."""
        m = M.Monitor(_base())
        for u in range(1, 31):
            m.observe_update({"update": u, "clip_frac": 0.5})
        self.assertAlmostEqual(m.lr_scale, 0.6)
        for u in range(31, 41):
            m.observe_update({"update": u, "early_stopped": True})
        self.assertAlmostEqual(m.lr_scale, 0.4)
        m.observe_gate("screen", 41, -0.05)
        m.observe_gate("screen", 42, -0.05)
        acts = m.observe_gate("screen", 43, -0.05)                 # rollback at the floor: lr unchanged
        self.assertEqual(_kinds(acts), ["rollback", "log"])
        self.assertAlmostEqual(m.lr_scale, M.TEMP_LR_FLOOR)
        self.assertEqual(_kinds(m.observe_gate("g1", 60, 0.0)), ["lr_scale"])
        self.assertAlmostEqual(m.lr_scale, 1.0)

    def test_DOC_signed_or_negative_identity_values_stop(self):
        """A signed max (logp_new - mu).max() or a negative k1 KL estimate is compared by |value| (fixed)."""
        for kw in ({"logp_max_abs_diff": {"wait": -5e-3, "play": 0.0, "ability": 0.0}},
                   {"ratio_max_abs_dev": -1.0},
                   {"kl_ref_per_head": {h: -1e-3 for h in M.IDENTITY_HEADS}}):
            m = M.Monitor(_base())
            self.assertEqual(_kinds(m.check_identity(_good_identity(**kw))), ["stop"], kw)

    def test_DOC_clip_streaks_are_separate(self):
        """Joint and play-row clip fractions are separate streaks: joint, play-row, joint, play-row above
        clip_max is no clip_window-update streak."""
        m = M.Monitor(_base())
        acts = []
        for u, (j, p) in enumerate([(0.5, 0.0), (0.0, 0.5), (0.5, 0.0), (0.0, 0.5)], 1):
            acts += m.observe_update({"update": u, "clip_frac": j, "play_clip_frac": p})
        self.assertEqual(acts, [])


if __name__ == "__main__":
    unittest.main()
