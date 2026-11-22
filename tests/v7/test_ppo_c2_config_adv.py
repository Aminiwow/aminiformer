"""Adversarial verification of the client-timing config (own-command sighting, stamp rule, zero display lag).

Checks, independently of the builder's own tests (tests/v7/test_ppo_c2_config.py):

* digests: the default config / client env digests, and that two changed timing values move them only through
  the changed keys (the config diff; the env record's landing-delay entries);
* the own-command sighting bound: on a grid of timing configs, the config's acceptance equals an independent
  first-principles bound (largest stamped delay + largest own sighting < 20), the rollout's ``LandingDelay`` (the
  env side) accepts exactly the same configs, the bound is sound (exhaustive worst lead < 20 when accepted), and
  config / timing helpers agree;
* the own sighting is the ceil form off a batch boundary (the next batch on one) on every stamp the default
  stamp rule can produce;
* zero display lag: every construction path refuses a nonzero sim2real.delta_lag (YAML file, override, from_dict,
  direct dataclass, dataclasses.replace, runtime change, post-construction mutation caught by digest / env_digest);
* a run.set file restates exactly the env defaults (env digest equal).

The env side (the rollout and timing modules), the run files and the object vocabulary are not published: the
checks that need them are skipped. CPU only, deterministic; unittest only.
Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_c2_config_adv.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from amini.ppo import config as C  # noqa: E402
try:                                                    # the env-side timing helpers
    from amini.ppo import timing as TM  # noqa: E402
except ImportError:                                     # not published: reported as a skip
    TM = None

try:                                                    # the env side; heavy imports
    from ppo_v7 import rollout as R  # noqa: E402
except Exception as _exc:                               # pragma: no cover - reported as a skip
    R = None
    _R_ERR = repr(_exc)

YAML_PATH = ROOT / "configs" / "ppo_v7_main.yaml"
RUN_SET = ROOT / "tools" / "ppo_v7" / "run_set.example"
YAML_SKIP = "needs the run configuration file configs/ppo_v7_main.yaml (not published)"
R_SKIP = "needs an unpublished module (ppo_v7.rollout)"
HAVE_VOCAB = (ROOT / C.OBJECT_VOCAB).is_file()             # delta_lag override names are checked against it
VOCAB_SKIP = "needs the private dataset"
NEW_CFG = "4c99e65a76e02f1b6bdd24f2c4ecce5923c68b8fd3003a4ef4db99fb9f2f776e"
NEW_ENV = "cf474c2bf0c27dcfd2240383f8136b6e50c256ce434696f56304eeaa5a27556c"
NEW_ENV_WITH_TABLES = "08625e7c78003025379e33783152275fba2d436566fdc68d27afccec6ccc7862"   # client_env_digest
NEW_ARRIVAL_TABLE = "413127b3d529f8c775f7b8a98847ac61fa161921137f6e854ebf83c4089bc11c"    # data/opp_cmd_arrival_v7
DISPATCH = 20


def _cfg(*over: str) -> C.PPOConfig:
    return C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), list(over)))


def _env_sha(rec: dict) -> str:
    blob = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


# ---------------------------------------------------------------- independent first-principles timing
def _stamp(x: int, enabled: bool, blocked: Sequence[int], push_to: int) -> int:
    if enabled and x % 10 in blocked:
        return x + ((push_to - x) % 10 or 10)
    return x


def _first_seen(st: int, j: int, period: int) -> int:        # the batched rule (floor + 1 form)
    return period * (st // period + 1) + j


def _indep(gs: Iterable[int], enabled: bool, blocked: Sequence[int], push_to: int, period: int,
           js: Sequence[int], table: bool) -> Tuple[int, int, int]:
    """(largest stamped delay, largest own sighting, exhaustive worst decision-to-sighting lead)."""
    gs = sorted(set(gs))
    e_hi = max(_stamp(10 + ph + g, enabled, blocked, push_to) - (10 + ph) for g in gs for ph in range(10))
    if table:
        s_hi = 8
        worst = e_hi + 8
    else:
        phases = [p for p in range(10) if not (enabled and p in blocked)]
        s_hi = max(_first_seen(st, j, period) - st for st in range(10 * period) if st % 10 in phases for j in js)
        worst = 0
        for t in range(1000, 1000 + 10 * period):
            for g in gs:
                st = _stamp(t + g, enabled, blocked, push_to)
                for j in js:
                    worst = max(worst, _first_seen(st, j, period) - t)
    return e_hi, s_hi, worst


# ============================================================================ digests
class TestDigestsAdv(unittest.TestCase):
    def test_new_digests(self):
        c = C.PPOConfig()
        self.assertEqual(c.digest(), NEW_CFG)
        self.assertEqual(c.env_digest(), NEW_ENV)
        self.assertEqual((C.CONFIG_SCHEMA, C.ENV_SCHEMA), ("ppo-v7-config.v3", "ppo-v7-env.1"))
        self.assertEqual(c.describe()["derived"]["client_env_digest_without_tables"], NEW_ENV)
        if not YAML_PATH.is_file():
            self.skipTest(YAML_SKIP)
        y = C.load_config(YAML_PATH)
        self.assertEqual((y.digest(), y.env_digest()), (NEW_CFG, NEW_ENV))

    def test_config_digest_moved_only_by_the_two_decided_values(self):
        old = _cfg("rollout.landing_delay.play_pmf.0=2.0", "rollout.landing_delay.uniform_mix=0.2")
        self.assertNotEqual(old.digest(), NEW_CFG)              # the two values move the digest ...
        self.assertEqual(set(C.config_diff(C.PPOConfig(), old)),   # ... and nothing else differs
                         {"rollout.landing_delay.play_pmf.0", "rollout.landing_delay.uniform_mix"})

    def test_env_digest_moved_only_by_the_pmf_and_the_own_sighting_record(self):
        old = _cfg("rollout.landing_delay.play_pmf.0=2.0", "rollout.landing_delay.uniform_mix=0.2")
        rec = old.env_record()
        self.assertEqual(rec["own_sighting"]["model"], C.OWN_SIGHTING_BATCHED)
        self.assertNotEqual(_env_sha(rec), NEW_ENV)
        new = C.PPOConfig().env_record()
        self.assertEqual(_env_sha(new), NEW_ENV)
        moved = sorted(k for k in new if new[k] != old.env_record()[k])
        self.assertEqual(moved, ["landing_delay", "landing_delay_pmf"])   # own_sighting is the same rule for both

    def test_runtime_env_digest_with_the_arrival_table(self):
        """The client_env_digest that real runs record (rollout.runtime_env_tables: the arrival table)."""
        if R is None:
            self.skipTest(R_SKIP)
        c = C.PPOConfig()
        tables = R.runtime_env_tables(c)
        self.assertEqual(tables, {"arrival": NEW_ARRIVAL_TABLE})
        self.assertEqual(c.env_digest(tables), NEW_ENV_WITH_TABLES)
        self.assertEqual(C.load_config(YAML_PATH).env_digest(tables), NEW_ENV_WITH_TABLES)

    def test_with_tables_moves_only_by_the_table(self):
        c = C.PPOConfig()
        t1, t2 = {"arrival": "a" * 64, "card_vocab": "b" * 64}, {"arrival": "c" * 64, "card_vocab": "b" * 64}
        self.assertNotEqual(c.env_digest(t1), c.env_digest(t2))
        self.assertEqual(C.model_tables_digest(t1), C.model_tables_digest(dict(t1, arrival="c" * 64)))


# ============================================================================ landing-delay pmf
class TestLandingDelayPmfAdv(unittest.TestCase):
    def test_exact_floats_no_mix(self):
        c = C.PPOConfig()
        play = c.landing_delay_pmf("play")
        self.assertEqual(play, {0: 1.0 / 4.0, 1: 2.0 / 4.0, 2: 1.0 / 4.0})   # bitwise
        self.assertEqual(c.landing_delay_pmf("ability"), {1: 0.5, 2: 0.5})
        self.assertEqual(C.G12A_PLAY_BASE_N, int(sum(C.G12A_PLAY_BASE_PMF.values())))
        # mix 0: zero-weight uniform keys are never added, so neither the support nor max_stamped_delay grows
        for over in ("rollout.landing_delay.uniform_max=7", "rollout.landing_delay.uniform_kinds=[play, ability]"):
            cc = _cfg(over)
            self.assertEqual(cc.landing_delay_pmf("play"), play, over)
            self.assertEqual((cc.max_stamped_delay("play"), cc.max_stamped_delay("ability")), (3, 3), over)
        if R is not None:                                        # the env's form of the pmfs
            ld, _ = R.sim2real_of(c)
            self.assertEqual(ld["play"], tuple((g, p) for g, p in play.items()))
            self.assertEqual(ld["ability"], (1, 2))              # the legacy randint stream for U{1..2}


# ============================================================================ own sighting
class TestOwnSightingAdv(unittest.TestCase):
    def test_decision_ceil_form_on_every_default_stamp(self):
        """Own first_seen = period * ceil(stamp / period) + j off a batch boundary, and the next batch on one (the
        committed floor + 1 form), on every stamp a decision can produce with the default stamp rule."""
        c = C.PPOConfig()
        per = c.rollout.arrival.period
        lo, hi = c.own_sighting_delay_range()
        n = 0
        for t in range(90, 600):
            for kind in C.LANDING_KINDS:
                for g in c.landing_delay_pmf(kind):
                    st = c.stamp_tick(t + g)
                    self.assertNotIn(st % 10, tuple(c.rollout.stamp_rule.blocked_mod10))
                    for j in c.arrival_j_pmf():
                        fs = C.own_first_seen(st, j)
                        if st % per:
                            self.assertEqual(fs, per * -(-st // per) + j, (t, kind, g, j))
                        else:
                            self.assertEqual(fs, st + per + j, (t, kind, g, j))
                        if TM is not None:
                            self.assertEqual(fs, TM.own_first_seen(st, j))
                        self.assertTrue(lo <= fs - st <= hi)
                        self.assertLess(fs - t, DISPATCH)
                        n += 1
        self.assertGreater(n, 5000)
        # on a batch boundary the two forms differ: next batch, not + j
        self.assertEqual((C.own_first_seen(100, 1), 5 * -(-100 // 5) + 1), (106, 101))

    def test_record_is_the_rule_and_same_j(self):
        c = C.PPOConfig()
        rec = c.env_record()["own_sighting"]
        self.assertEqual(rec, c.own_sighting_record())
        self.assertEqual(rec["pending_visible_from"], "decision")
        self.assertEqual((rec["model"], rec["delay_range"], rec["stamp_phases"]),
                         (C.OWN_SIGHTING_BATCHED, [2, 7], [0, 1, 2, 3, 4, 6, 7, 8, 9]))
        self.assertNotIn("range", rec)                          # not the legacy range
        for st in range(91, 400):
            if st % 10 in (5,):
                continue
            self.assertEqual(c.own_sighting_delay_pmf(st), c.arrival_delay_pmf(st))
        tab = _cfg("rollout.arrival.model=table").env_record()["own_sighting"]
        self.assertEqual((tab["model"], tab["range"]), (C.OWN_SIGHTING_R0_RANGE, [2, 8]))
        self.assertEqual(C.OWN_SIGHTING_MODEL_OF_ARRIVAL, {"batched": C.OWN_SIGHTING_BATCHED,
                                                           "table": C.OWN_SIGHTING_R0_RANGE})
        self.assertEqual(set(C.OWN_SIGHTING_MODEL_OF_ARRIVAL), set(C.ARRIVAL_MODELS))

    def test_bound_grid_config_env_and_first_principles_agree(self):
        """A grid of timing configs: config acceptance == independent bound < 20 == rollout LandingDelay
        acceptance; accepted configs have an exhaustive worst lead < 20; config and timing helpers agree."""
        stamp_rules = [(True, (5,), 6), (False, (5,), 6), (True, (4,), 5), (True, (0,), 1), (True, (0,), 2),
                       (True, (3, 4, 5), 6)]
        periods = [5, 9, 10, 11]
        j_extra = [None, 4, 5]
        play_extra = [None, 4, 8, 11]
        ab_extra = [None, 6, 10]
        models = ["batched", "table"]
        checked = refused = env_checked = 0
        for (en, bl, pu), per, jx, px, ax, mdl in itertools.product(stamp_rules, periods, j_extra, play_extra,
                                                                       ab_extra, models):
            over = [f"rollout.stamp_rule.enabled={'true' if en else 'false'}",
                    f"rollout.stamp_rule.blocked_mod10=[{', '.join(map(str, bl))}]",
                    f"rollout.stamp_rule.push_to_mod10={pu}", f"rollout.arrival.period={per}",
                    f"rollout.arrival.model={mdl}"]
            js = [1, 2]
            play = dict(C.G12A_PLAY_BASE_PMF)
            ab = dict(C.ABILITY_PMF)
            if jx is not None:
                over.append(f"rollout.arrival.j_pmf.{jx}=0.1")
                js.append(jx)
            if px is not None:
                over.append(f"rollout.landing_delay.play_pmf.{px}=1.0")
                play[px] = 1.0
            if ax is not None:
                over.append(f"rollout.landing_delay.ability_pmf.{ax}=1.0")
                ab[ax] = 1.0
            e_hi, s_hi, worst = _indep(list(play) + list(ab), en, bl, pu, per, js, mdl == "table")
            want_ok = e_hi + s_hi < DISPATCH
            with self.subTest(over=over):
                try:
                    cfg = _cfg(*over)
                    ok, msg = True, ""
                except C.ConfigError as exc:
                    ok, msg = False, str(exc)
                self.assertEqual(ok, want_ok, (e_hi, s_hi, msg))
                if not ok:
                    refused += 1
                    self.assertIn("own sighting", msg)
                else:
                    self.assertLess(worst, DISPATCH)              # sound: the stamp is read before D = t + 20
                    self.assertLessEqual(worst, e_hi + s_hi)
                    self.assertEqual(max(cfg.max_stamped_delay(k) for k in C.LANDING_KINDS), e_hi)
                    self.assertEqual(cfg.own_sighting_delay_range()[1], s_hi)
                    if mdl == "batched" and TM is not None:
                        self.assertEqual(cfg.own_sighting_delay_range(),
                                         TM.own_sighting_delays(TM.BatchedArrival.from_config(cfg),
                                                                TM.StampRule.from_config(cfg)))
                    self.assertEqual(cfg.env_record()["own_sighting"]["delay_range"][1], s_hi)
                checked += 1
                if R is None:
                    continue
                # the env side: LandingDelay with the same pmfs, stamp rule and own arrival
                rule = TM.StampRule(tuple(bl), pu, en)
                arr = None if mdl == "table" else TM.BatchedArrival(j_pmf=tuple((j, 1.0) for j in js), period=per)
                try:
                    R.LandingDelay(play=dict(play), ability=dict(ab), seed=0, stamp_rule=rule, own_arrival=arr)
                    env_ok = True
                except ValueError:
                    env_ok = False
                self.assertEqual(env_ok, want_ok, "config and env bounds disagree")
                env_checked += 1
        self.assertGreater(checked, 1000)
        self.assertGreater(refused, 100)
        if R is not None:
            self.assertEqual(env_checked, checked)

    def test_legacy_null_pmfs_with_the_stamp_rule(self):
        """Legacy null pmfs are bounded like any pmf: with the default stamp rule and batched arrival a legacy
        range U{1..12} is stamped up to 13, + own sighting 7 = 20, which the env refuses; the config refuses it
        too, and accepts it with the stamp rule off (12 + 7 = 19)."""
        with self.assertRaisesRegex(C.ConfigError, r"stamped play delay up to 13 \+ own sighting 7"):
            _cfg("rollout.landing_delay.play_pmf=null", "rollout.landing_delay.ability_pmf=null",
                 "rollout.landing_delay.uniform_mix=0.0", "sim2real.landing_delay.play=[1, 12]")
        off = _cfg("rollout.landing_delay.play_pmf=null", "rollout.landing_delay.ability_pmf=null",
                   "rollout.landing_delay.uniform_mix=0.0", "sim2real.landing_delay.play=[1, 12]",
                   "rollout.stamp_rule.enabled=false")
        self.assertEqual(off.max_stamped_delay("play"), 12)
        on = C.PPOConfig()                                          # the default stamp rule and batched arrival
        self.assertEqual(max(on.stamp_tick(10 + ph + g) - (10 + ph) for ph in range(10) for g in range(1, 13)), 13)
        self.assertEqual(on.own_sighting_delay_range()[1], 7)
        if R is None:
            self.skipTest(R_SKIP)
        with self.assertRaises(ValueError):
            R.LandingDelay(play=(1, 12), ability=(1, 3), seed=0, stamp_rule=TM.StampRule.from_config(on),
                           own_arrival=TM.BatchedArrival.from_config(on))

    def test_null_pmf_and_mix_grid_config_env_and_first_principles_agree(self):
        """Null (legacy-range) pmfs and a uniform mix are bounded like any pmf. Grid over play/ability null or
        not, mix 0 / 0.1 on [play] / [play, ability], uniform_max 3 / 12, stamp rule on / off, batched / table:
        config acceptance == independent bound < 20 == rollout LandingDelay acceptance."""
        checked = refused = 0
        for pn, an, mix, kinds, umax, en, mdl in itertools.product(
                (False, True), (False, True), (0.0, 0.1), (("play",), ("play", "ability")), (3, 12), (True, False),
                ("batched", "table")):
            over = [f"rollout.stamp_rule.enabled={'true' if en else 'false'}", f"rollout.arrival.model={mdl}",
                    f"rollout.landing_delay.uniform_mix={mix}", f"rollout.landing_delay.uniform_max={umax}",
                    f"rollout.landing_delay.uniform_kinds=[{', '.join(kinds)}]"]
            sup = {"play": set(C.G12A_PLAY_BASE_PMF), "ability": set(C.ABILITY_PMF)}
            if pn:
                over.append("rollout.landing_delay.play_pmf=null")
                sup["play"] = set(range(1, 5))
            if an:
                over.append("rollout.landing_delay.ability_pmf=null")
                sup["ability"] = set(range(1, 4))
            if mix > 0.0:
                for k in kinds:
                    sup[k] |= set(range(umax + 1))
            e_hi, s_hi, worst = _indep(sup["play"] | sup["ability"], en, (5,), 6, 5, [1, 2], mdl == "table")
            want_ok = e_hi + s_hi < DISPATCH
            with self.subTest(over=over):
                try:
                    cfg = _cfg(*over)
                    ok, msg = True, ""
                except C.ConfigError as exc:
                    ok, msg = False, str(exc)
                self.assertEqual(ok, want_ok, (e_hi, s_hi, msg))
                if ok:
                    self.assertLess(worst, DISPATCH)
                    self.assertEqual({k: set(cfg.landing_delay_pmf(k)) for k in C.LANDING_KINDS}, sup)
                    self.assertEqual(max(cfg.max_stamped_delay(k) for k in C.LANDING_KINDS), e_hi)
                else:
                    refused += 1
                    self.assertIn("own sighting", msg)
                checked += 1
                if R is None:
                    continue
                rule = TM.StampRule((5,), 6, en)
                arr = None if mdl == "table" else TM.BatchedArrival(j_pmf=((1, 1.0), (2, 1.0)), period=5)
                try:
                    R.LandingDelay(play={g: 1.0 for g in sup["play"]}, ability={g: 1.0 for g in sup["ability"]},
                                   seed=0, stamp_rule=rule, own_arrival=arr)
                    env_ok = True
                except ValueError:
                    env_ok = False
                self.assertEqual(env_ok, want_ok, "config and env bounds disagree")
        self.assertEqual(checked, 128)
        self.assertGreater(refused, 10)

    def test_legacy_reproduction_still_accepted(self):
        """Stamp rule off + table arrival + legacy ranges: 4 + 8 = 12 < 20 (the legacy env)."""
        cfg = _cfg("rollout.landing_delay.play_pmf=null", "rollout.landing_delay.ability_pmf=null",
                   "rollout.landing_delay.uniform_mix=0.0", "rollout.stamp_rule.enabled=false",
                   "rollout.arrival.model=table")
        self.assertEqual((cfg.max_stamped_delay("play"), cfg.own_sighting_delay_range()), (4, (2, 8)))
        self.assertEqual(cfg.landing_delay_pmf("play"), {g: 0.25 for g in range(1, 5)})

    def test_constants_agree_with_rollout(self):
        if R is None:
            self.skipTest(R_SKIP)
        self.assertEqual(tuple(R.DEFAULT_OWN_SIGHTING), C.OWN_SIGHTING_R0)
        self.assertEqual(R.DISPATCH, C.DISPATCH_LEAD)
        for st in range(0, 300):
            for j in range(1, 6):
                for per in (5, 9, 10, 11):
                    self.assertEqual(TM.own_first_seen(st, j, per), C.own_first_seen(st, j, per))


# ============================================================================ zero display lag
class TestDeltaLagAdv(unittest.TestCase):
    def test_default(self):
        self.assertEqual(dataclasses.astuple(C.DeltaLagConfig()), (0, {}))
        self.assertEqual(C.Sim2RealConfig().delta_lag, C.DeltaLagConfig())
        if R is not None:
            _, dl = R.sim2real_of(C.PPOConfig())
            R.check_zero_delta_lag(dl, "verify")               # the env accepts the config's block

    def test_every_construction_path_refuses(self):
        rx = r"must be 0, got 2: the env models no display lag.*self-check"
        with self.assertRaisesRegex(C.ConfigError, rx):        # direct dataclass construction
            C.PPOConfig(sim2real=C.Sim2RealConfig(delta_lag=C.DeltaLagConfig(default_ticks=2)))
        with self.subTest("override name"):
            if not HAVE_VOCAB:                                 # override names need the vocabulary
                self.skipTest(VOCAB_SKIP)
            with self.assertRaisesRegex(C.ConfigError, rx):
                C.PPOConfig(sim2real=C.Sim2RealConfig(delta_lag=C.DeltaLagConfig(overrides={"Rocket": 2})))
        base = C.PPOConfig()
        with self.assertRaisesRegex(C.ConfigError, rx):        # dataclasses.replace re-runs __post_init__
            dataclasses.replace(base, sim2real=dataclasses.replace(
                base.sim2real, delta_lag=C.DeltaLagConfig(default_ticks=2)))
        with self.assertRaisesRegex(C.ConfigError, rx):        # sim2real off does not excuse it
            C.PPOConfig(sim2real=C.Sim2RealConfig(enabled=False, delta_lag=C.DeltaLagConfig(default_ticks=2)))
        with self.assertRaises(C.ConfigError):                 # mid-run change: not a runtime key either way
            C.runtime_change(base, ["sim2real.delta_lag.default_ticks=2"], update=3, reason="x")
        with self.assertRaises(C.ConfigError):
            C.with_changes(base, {"sim2real.delta_lag.default_ticks": 2})
        with self.assertRaises(C.ConfigError):                 # a bool is not an int (no True == 1 sneak-in)
            _cfg("sim2real.delta_lag.overrides.Rocket=true")
        with self.assertRaises(C.ConfigError):
            _cfg("sim2real.delta_lag.default_ticks=0.5")
        if not YAML_PATH.is_file():
            self.skipTest(YAML_SKIP)
        tmp = tempfile.mkdtemp(prefix="c2-adv-", dir=os.environ.get("TMPDIR"))
        try:                                                    # the YAML file itself (not an override)
            p = Path(tmp) / "lag.yaml"
            text = YAML_PATH.read_text(encoding="utf-8")
            self.assertEqual(text.count("    default_ticks: 0\n"), 1)
            p.write_text(text.replace("    default_ticks: 0\n", "    default_ticks: 2\n"), encoding="utf-8")
            with self.assertRaisesRegex(C.ConfigError, rx):
                C.load_config(p)
        finally:
            for f in Path(tmp).iterdir():
                f.unlink()
            os.rmdir(tmp)

    @unittest.skipUnless(HAVE_VOCAB, VOCAB_SKIP)
    def test_post_construction_mutation_is_caught_by_the_digests(self):
        c = C.PPOConfig()
        c.sim2real.delta_lag.overrides["Rocket"] = 3            # frozen dataclass, mutable dict: a bypass attempt
        try:
            for fn in (c.digest, c.env_digest, c.describe):
                with self.assertRaisesRegex(C.ConfigError, "overrides.Rocket: must be 0"):
                    fn()
            if R is not None:
                with self.assertRaises(ValueError):
                    R.EpisodeSettings.from_config(c, base_seed=0, population_codes={16: 0}, opponents={})
        finally:
            c.sim2real.delta_lag.overrides.clear()
        self.assertEqual(c.digest(), NEW_CFG)

    def test_range_and_name_messages_come_first(self):
        with self.assertRaisesRegex(C.ConfigError, r"outside \[0, 20\]"):
            _cfg("sim2real.delta_lag.default_ticks=21")
        if not HAVE_VOCAB:
            self.skipTest(VOCAB_SKIP)
        with self.assertRaisesRegex(C.ConfigError, "not object names"):
            _cfg("sim2real.delta_lag.overrides.NoSuchObject=0")    # a zero for an unknown name is refused too
        c = _cfg("sim2real.delta_lag.overrides.Rocket=0", "sim2real.delta_lag.default_ticks=0")
        self.assertEqual(dict(c.sim2real.delta_lag.overrides), {"Rocket": 0})


# ============================================================================ files
class TestFilesAdv(unittest.TestCase):
    @unittest.skipUnless(YAML_PATH.is_file() and RUN_SET.is_file(), "needs the run configuration files (not published)")
    def test_run_set_restates_exactly_the_env_defaults(self):
        rs = RUN_SET.read_text(encoding="utf-8").splitlines()
        live = [ln.strip() for ln in rs if ln.strip() and not ln.lstrip().startswith("#") and "<" not in ln]
        env = [ln for ln in live if ln.startswith(("rollout.landing_delay", "rollout.stamp_rule", "rollout.arrival",
                                                   "rollout.min_tap_tick", "rollout.client", "sim2real."))]
        self.assertGreaterEqual(len(env), 18)
        self.assertEqual(C.load_config(YAML_PATH, env).env_digest(), NEW_ENV)
        self.assertNotIn("sim2real.delta_lag.default_ticks=3", live)

    @unittest.skipUnless(YAML_PATH.is_file(), YAML_SKIP)
    def test_yaml_timing_block_equals_the_defaults(self):
        y = C.load_config(YAML_PATH)
        self.assertEqual(y.rollout.landing_delay, C.RolloutLandingDelayConfig())
        self.assertEqual(y.sim2real.delta_lag, C.DeltaLagConfig())
        self.assertEqual(y.env_record(), C.PPOConfig().env_record())


if __name__ == "__main__":
    unittest.main()
