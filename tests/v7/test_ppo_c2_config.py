"""Client-timing config tests: the default landing-delay pmfs, the own-command sighting rule and zero display lag.

* the default play / ability pmfs with no uniform mix (the inactive smoothing switch leaves every pmf unchanged);
* own sighting: the batched rule with the arrival's per-game j (distribution per stamp phase, the phases the
  stamp rule leaves), the strict lead bound (exhaustive and in validation), env_record;
* delta_lag: DeltaLagConfig() is (0, {}); any nonzero default or override is refused.

The checks against the private client-timing measurements, the run file and the object vocabulary are not
published: the tests that need them are skipped. CPU only, deterministic; unittest only.
Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_c2_config.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from amini.ppo import config as C  # noqa: E402

YAML_PATH = ROOT / "configs" / "ppo_v7_main.yaml"
YAML_SKIP = "needs the run configuration file configs/ppo_v7_main.yaml (not published)"
HAVE_VOCAB = (ROOT / C.OBJECT_VOCAB).is_file()             # delta_lag override names are checked against it
VOCAB_SKIP = "needs the private dataset"
MEASURED_SKIP = "compares the defaults with private client-timing measurements (not published)"
DISPATCH = 20


def _cfg(*over: str) -> C.PPOConfig:
    return C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), list(over)))


# ============================================================================ landing-delay defaults
class TestLandingDelayDefaults(unittest.TestCase):
    @unittest.skip(MEASURED_SKIP)
    def test_base_pmf_is_the_unpushed_g(self):
        """The default play pmf is the part of a measured decision-to-stamp histogram that the stamp rule did not
        push, re-derived from the per-decision-phase table."""

    def test_defaults_no_mix(self):
        c = C.PPOConfig()
        ld = c.rollout.landing_delay
        self.assertEqual((C.G12A_UNIFORM_MIX, ld.uniform_mix), (0.0, 0.0))
        self.assertEqual(ld.play_pmf, {0: 1.0, 1: 2.0, 2: 1.0})
        self.assertEqual(ld.ability_pmf, {1: 1.0, 2: 1.0})                     # U{1..2}
        self.assertEqual(c.landing_delay_pmf("play"), {0: 1 / 4, 1: 2 / 4, 2: 1 / 4})
        self.assertEqual(c.landing_delay_pmf("ability"), {1: 0.5, 2: 0.5})
        for um in ("rollout.landing_delay.uniform_max=6", "rollout.landing_delay.uniform_kinds=[play, ability]"):
            self.assertEqual(_cfg(um).landing_delay_pmf("play"), c.landing_delay_pmf("play"), um)   # inactive
        if not YAML_PATH.is_file():
            self.skipTest(YAML_SKIP)
        self.assertEqual(C.load_config(YAML_PATH), c)

    @unittest.skip(MEASURED_SKIP)
    def test_stamped_marginal_and_pushed_tail(self):
        """The stamped delay marginal of the defaults over the measured decision phases is close (total variation)
        to the measured decision-to-stamp histogram, and every measured (phase, delay) pair is possible."""

    @unittest.skipUnless(YAML_PATH.is_file(), YAML_SKIP)
    def test_files_spell_the_corrections(self):
        y = YAML_PATH.read_text(encoding="utf-8")
        self.assertIn("uniform_mix: 0.0", y)
        self.assertEqual(C.load_config(YAML_PATH).rollout.landing_delay, C.PPOConfig().rollout.landing_delay)


# ============================================================================ own-command sighting
class TestOwnSighting(unittest.TestCase):
    def setUp(self):
        self.c = C.PPOConfig()

    def test_rule_is_the_batched_arrival_with_the_same_j(self):
        for st in range(90, 400):
            for j in (1, 2, 3):
                self.assertEqual(C.own_first_seen(st, j), C.batched_first_seen(st, j))
                if st % 5:
                    self.assertEqual(C.own_first_seen(st, j), 5 * -(-st // 5) + j)   # the ceil form
        self.assertEqual(C.own_first_seen(100, 1), 106)                       # a batch boundary: the next batch
        self.assertEqual(self.c.own_sighting_model(), C.OWN_SIGHTING_BATCHED)
        rec = self.c.own_sighting_record()
        self.assertIn("rollout.arrival j", rec["j"])
        self.assertEqual(rec["period"], self.c.rollout.arrival.period)

    def test_distribution_over_stamp_phases(self):
        phases = (0, 1, 2, 3, 4, 6, 7, 8, 9)                                  # every phase but the blocked 5
        self.assertEqual(self.c.own_sighting_stamp_phases(), phases)
        support = set()
        for ph in phases:
            for st in (1000 + ph, 2370 + ph):
                pmf = self.c.own_sighting_delay_pmf(st)
                with self.subTest(stamp=st):
                    self.assertEqual(pmf, self.c.arrival_delay_pmf(st))      # one j pmf for own and opponent
                    self.assertEqual(pmf, {5 - ph % 5 + j: p for j, p in self.c.arrival_j_pmf().items()})
                    self.assertAlmostEqual(sum(pmf.values()), 1.0, places=12)
            support |= set(self.c.own_sighting_delay_pmf(1000 + ph))
        self.assertEqual(support, set(range(2, 8)))
        self.assertEqual((min(support), max(support)), C.G12A_OWN_ARRIVAL_RANGE)
        self.assertEqual(self.c.own_sighting_delay_range(), C.G12A_OWN_ARRIVAL_RANGE)

    def test_variants(self):
        off = _cfg("rollout.stamp_rule.enabled=false")
        self.assertEqual(off.own_sighting_stamp_phases(), tuple(range(10)))
        self.assertEqual(off.own_sighting_delay_range(), (2, 7))             # phase 4 / 9: 1 + j; 0 / 5: 5 + j
        both = _cfg("rollout.stamp_rule.blocked_mod10=[0, 5]")
        self.assertEqual(both.own_sighting_delay_range(), (2, 6))            # no stamp on a batch boundary
        j4 = _cfg("rollout.arrival.j_pmf.4=0.1")
        self.assertEqual(j4.own_sighting_delay_range(), (2, 9))
        tab = _cfg("rollout.arrival.model=table")
        self.assertEqual(tab.own_sighting_model(), C.OWN_SIGHTING_R0_RANGE)
        self.assertEqual(tab.own_sighting_delay_range(), C.OWN_SIGHTING_R0)
        self.assertEqual(tab.own_sighting_delay_pmf(1001), {d: 1 / 7 for d in range(2, 9)})
        self.assertEqual(tab.env_record()["own_sighting"]["model"], C.OWN_SIGHTING_R0_RANGE)
        self.assertEqual(tab.env_record()["own_sighting"]["range"], [2, 8])
        self.assertEqual(C.OWN_SIGHTING_R0, (2, 8))
        with self.assertRaises(C.ConfigError):
            tab.own_sighting_delay_pmf(-1)

    def test_strict_bound_exhaustive(self):
        """Every (decision phase, kind, g, j): the stamp is read before the registered D = t + 20, and the sum bound
        the validator uses (largest stamped delay + largest own sighting) is never below the exact largest lead."""
        worst = 0
        for ph in range(10):
            t = 1000 + ph
            for kind in C.LANDING_KINDS:
                for g in self.c.landing_delay_pmf(kind):
                    st = self.c.stamp_tick(t + g)
                    for j in self.c.arrival_j_pmf():
                        lead = C.own_first_seen(st, j) - t
                        self.assertLess(lead, DISPATCH, (ph, kind, g, j))
                        worst = max(worst, lead)
        hi = self.c.own_sighting_delay_range()[1]
        bound = max(self.c.max_stamped_delay(k) for k in C.LANDING_KINDS) + hi
        self.assertEqual((worst, bound), (9, 10))                             # the bound is not tight here

    def test_strict_bound_in_validation(self):
        self.assertEqual(_cfg("rollout.landing_delay.ability_pmf.11=1.0").max_stamped_delay("ability"), 12)  # 19
        with self.assertRaisesRegex(C.ConfigError, r"stamped ability delay up to 13 \+ own sighting 7 "
                                                   r"\(batched_same_j\) >= 20"):
            _cfg("rollout.landing_delay.ability_pmf.12=1.0")
        with self.assertRaisesRegex(C.ConfigError, r"stamped play delay up to 13 \+ own sighting 7"):
            _cfg("rollout.landing_delay.play_pmf.12=1.0")
        with self.assertRaisesRegex(C.ConfigError, r"own sighting 8 \(r0_range\)"):
            _cfg("rollout.landing_delay.ability_pmf.11=1.0", "rollout.arrival.model=table")
        with self.assertRaisesRegex(C.ConfigError, r"own sighting 17 \(batched_same_j\)"):
            _cfg("rollout.arrival.period=12", "rollout.arrival.j_pmf.5=0.1")      # j 5: 12 + 5 = 17, + 3 = 20
        _cfg("rollout.arrival.period=12", "rollout.arrival.j_pmf.4=0.1")           # 3 + 16 = 19
        with self.assertRaisesRegex(C.ConfigError, r"own sighting 7 \(batched_same_j\)"):
            _cfg("rollout.stamp_rule.enabled=false", "rollout.landing_delay.play_pmf.13=1.0")   # 13 + 7
        _cfg("rollout.stamp_rule.enabled=false", "rollout.landing_delay.play_pmf.12=1.0")      # 12 + 7 = 19
        _cfg("rollout.stamp_rule.enabled=false", "rollout.arrival.model=table",
             "rollout.landing_delay.play_pmf=null", "rollout.landing_delay.ability_pmf=null")   # legacy: 4 + 8 = 12
        # a longer period puts an allowed stamp further from the next batch (period + j): 15 + 2 + 3 = 20
        with self.assertRaisesRegex(C.ConfigError, r"stamped play delay up to 3 \+ own sighting 17"):
            _cfg("rollout.arrival.period=15")
        self.assertEqual(_cfg("rollout.arrival.period=9").own_sighting_delay_range(), (2, 11))

    def test_env_record(self):
        rec = self.c.env_record()["own_sighting"]
        self.assertEqual(rec, {"model": C.OWN_SIGHTING_BATCHED, "rule": "first_seen = period * (stamp // period + 1) + j",
                               "j": "rollout.arrival j of the observing seat (one draw per game, shared with "
                                    "opponent arrivals)", "period": 5, "stamp_phases": [0, 1, 2, 3, 4, 6, 7, 8, 9],
                               "delay_range": [2, 7], "pending_visible_from": "decision"})
        json.dumps(rec, allow_nan=False)
        self.assertNotEqual(_cfg("rollout.arrival.model=table").env_digest(), self.c.env_digest())


# ============================================================================ zero display lag
class TestDeltaLagZero(unittest.TestCase):
    def test_default_is_zero(self):
        dl = C.DeltaLagConfig()
        self.assertEqual((dl.default_ticks, dl.overrides), (0, {}))
        self.assertEqual(C.PPOConfig().sim2real.delta_lag, dl)
        if not YAML_PATH.is_file():
            self.skipTest(YAML_SKIP)
        self.assertEqual(C.load_config(YAML_PATH).sim2real.delta_lag, dl)

    def test_nonzero_is_refused(self):
        for over, rx in ((["sim2real.delta_lag.default_ticks=1"], r"sim2real.delta_lag.default_ticks: must be 0, got 1"),
                         (["sim2real.delta_lag.default_ticks=3"], r"default_ticks: must be 0, got 3: the env models no"),
                         (["sim2real.delta_lag.overrides.Fireball=1"],
                          r"sim2real.delta_lag.overrides.Fireball: must be 0, got 1: the env models no"),
                         (["sim2real.enabled=false", "sim2real.delta_lag.default_ticks=2"], r"must be 0"),
                         (["sim2real.delta_lag.overrides.Fireball=0", "sim2real.delta_lag.overrides.Rocket=4"],
                          r"overrides.Rocket: must be 0")):
            with self.subTest(over=over):
                if any("overrides" in o for o in over) and not HAVE_VOCAB:   # override names need the vocabulary
                    self.skipTest(VOCAB_SKIP)
                with self.assertRaisesRegex(C.ConfigError, rx):
                    _cfg(*over)
                if YAML_PATH.is_file():
                    with self.assertRaisesRegex(C.ConfigError, rx):
                        C.load_config(YAML_PATH, over)
        d = C.PPOConfig().to_dict()
        d["sim2real"]["delta_lag"]["default_ticks"] = 3
        with self.assertRaisesRegex(C.ConfigError, "self-check"):
            C.from_dict(d)
        with self.assertRaisesRegex(C.ConfigError, r"outside \[0, 20\]"):
            _cfg("sim2real.delta_lag.default_ticks=-1")                       # the range message comes first
        if not HAVE_VOCAB:
            self.skipTest(VOCAB_SKIP)
        with self.assertRaisesRegex(C.ConfigError, "not object names"):
            _cfg("sim2real.delta_lag.overrides.Firebal=1")                   # so does the name check

    @unittest.skipUnless(HAVE_VOCAB, VOCAB_SKIP)
    def test_zero_override_is_allowed(self):
        c = _cfg("sim2real.delta_lag.overrides.Fireball=0")
        self.assertEqual(dict(c.sim2real.delta_lag.overrides), {"Fireball": 0})
        self.assertNotEqual(c.env_digest(), C.PPOConfig().env_digest())


if __name__ == "__main__":
    unittest.main()
