"""Adversarial verification of src/amini/ppo/config.py and the run file configs/ppo_v7_main.yaml.

The builder's suite checks that the YAML equals the dataclass defaults, which is tautological for the
numbers themselves (a wrong number in both places passes).  This file adds:

* every default number pinned independently (opponent rows, gates, monitors, sim-to-real, critic warm-up,
  derived rows per update, HT weights = 1/KAPPA as exact fractions);
* generic walks over the whole tree: an unknown key at every mapping level, null at every leaf (only the
  declared open items may be null), an ``a.b=value`` override of every leaf with its own value (digest
  unchanged), no aliasing of the input mapping, strict JSON in describe();
* the PyYAML branch of ``il_v7.load_yaml`` (skipped where PyYAML is absent);
* regression tests for defects found in review (each was an ``expectedFailure`` before its fix):
    - rollout.temperature accepted 0.0 (greedy) and 0.5 although theta_k is trained at T = 1;
    - opponents.temperature.snapshot / il_v2_heldout could be null without readiness reporting it;
    - a single user.king_level could not serve a mixed level mix (the case builder asserts kt == level):
      kt is now the game's level (``PPOConfig.king_level``), and level keys are 16 / 11 only;
    - user.tower_troop was free text, but the case builder needs the game's tower-troop gid (int);
    - effective_mix() keys (theta_opp_seat, snapshots) were refused by the case builder (self_seat, snapshot);
    - MonitorConfig could not build monitor.Thresholds (other names; rollback in pp vs fraction): now
      ``PPOConfig.monitor_thresholds()``;
    - a deck holding a core-card candidate that monitor.core_cards does not list was accepted;
    - '-0.0' gave an equal config with another digest;
    - a repeated section in the run file silently dropped the earlier block (``load_config`` lints);
    - a huge integer raised OverflowError instead of ConfigError;
    - describe() / digest() of a config mutated after construction did not re-validate;
    - under PyYAML, a bare 'NULL' population reads as null (``load_config`` refuses bare NULL; quote it).

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_config_adv.py -v
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from amini.ppo import config as C  # noqa: E402
from amini.ppo import gae as SMP  # noqa: E402  (KAPPA and WEIGHT = 1 / KAPPA)

YAML_PATH = ROOT / "configs" / "ppo_v7_main.yaml"
YAML_SKIP = "needs the run configuration file configs/ppo_v7_main.yaml (not published)"
HAVE_VOCAB = (ROOT / C.OBJECT_VOCAB).is_file()             # delta_lag override names are checked against it
VOCAB_SKIP = "needs the private dataset"
HAVE_CASES = importlib.util.find_spec("ppo_v7.cases") is not None
CASES_SKIP = "needs an unpublished module (ppo_v7.cases)"
needs_yaml = unittest.skipUnless(YAML_PATH.is_file(), YAML_SKIP)
needs_cases = unittest.skipUnless(HAVE_CASES, CASES_SKIP)

try:                                     # PyYAML is optional
    import yaml  # type: ignore
except ImportError:                      # pragma: no cover - depends on the interpreter
    yaml = None

# placeholder card keys; "example-card" is the default core card, "core-b" a core-card candidate in DECK_B
DECK_A = ["card-a", "card-b-ev1", "example-card", "card-d", "card-e-ev1", "card-f", "card-g-hero", "card-h"]
DECK_B = ["card-i", "card-j", "card-b-ev1", "card-k-hero", "core-b", "example-card", "card-l", "card-m-ev1"]
TOWER_PRINCESS_GID = 159000000           # the tower-princess gid (config.TOWER_TROOP_GIDS)

# leaves that may legitimately be null (open items and measurements)
OPTIONAL_LEAVES = {
    "user.deck", "user.tower_troop", "rollout.hazard_dt",
    "opponents.temperature.il_v4_tstar", "opponents.temperature.snapshot",
    "opponents.checkpoint.il_v4", "opponents.checkpoint.il_v3", "opponents.checkpoint.il_v2_heldout",
    "opponents.checkpoint_sha256.il_v4", "opponents.checkpoint_sha256.il_v3",
    "opponents.checkpoint_sha256.il_v2_heldout",
    "levels.population.11",
    "opponents.hazard_dt.il_v3", "opponents.hazard_dt.il_v2_heldout",
    "opponents.code_tree.il_v3", "opponents.code_tree.il_v2_heldout",
    "monitor.parity.dlogp_tol_bf16.wait", "monitor.parity.dlogp_tol_bf16.play",
    "monitor.parity.dlogp_tol_bf16.ability",
}
# leaves whose null is a setting, not an open item: never reported by readiness
NULL_BY_DESIGN = {
    "rollout.min_tap_tick",                                      # null = no earliest tap
    "rollout.remote.root",                                       # required only with rollout.remote.enabled
    "monitor.tiny_ratio_share_max", "monitor.gate_trend_act_rel_max", "monitor.gate_trend_adv_gap_max",
    "monitor.entropy_max_frac_of_ref", "monitor.flip_rate_max",  # logged only until calibrated
    "rollout.arrival.source_sha256",                             # an unpinned arrival source
    "gates.panel_level",                                         # null = every level with a share
}


def _load_yaml():
    from amini.train.il_v7 import load_yaml   # imports torch through amini.train
    return load_yaml


def _ready(**user: Any) -> Dict[str, Any]:
    """Default config as a mapping with every rollout / policy prerequisite filled in."""

    d = C.PPOConfig().to_dict()
    d["user"].update({"deck": list(DECK_A), "tower_troop": TOWER_PRINCESS_GID})
    d["user"].update(user)
    d["rollout"]["hazard_dt"] = "block"
    d["opponents"]["checkpoint"].update({"il_v4": "/x/v4.pt", "il_v3": "/x/v3.pt", "il_v2_heldout": "/x/v2.pt"})
    d["opponents"]["checkpoint_sha256"].update({"il_v4": "a" * 64, "il_v3": "b" * 64, "il_v2_heldout": "c" * 64})
    d["opponents"]["hazard_dt"].update({"il_v3": "interval", "il_v2_heldout": "block"})
    d["opponents"]["code_tree"].update({"il_v3": "v3-frozen", "il_v2_heldout": "v2-frozen"})
    d["opponents"]["temperature"]["il_v4_tstar"] = 0.5
    d["levels"]["population"]["11"] = "challenge_eventdeck"
    d["monitor"]["parity"]["dlogp_tol_bf16"] = {"wait": 5e-4, "play": 2e-4, "ability": 2e-4}
    return d


def _leaves(d: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            yield from _leaves(v, p)
        else:
            yield p, v


def _mappings(d: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Dict[str, Any]]]:
    yield prefix, d
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _mappings(v, f"{prefix}.{k}" if prefix else k)


def _get(d: Dict[str, Any], path: str) -> Any:
    for p in path.split("."):
        d = d[p]
    return d


def _set(d: Dict[str, Any], path: str, v: Any) -> None:
    parts = path.split(".")
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = v


def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_yaml_scalar(x) for x in v) + "]"
    return repr(v) if isinstance(v, float) else str(v)


# ============================================================================ default numbers
class TestPlanNumbersIndependently(unittest.TestCase):
    """Numbers the builder's suite only checks as YAML == defaults (tautological for the values)."""

    def setUp(self):
        self.c = C.PPOConfig()

    def test_opponent_rows(self):                         # the opponent mix, row by row
        o = self.c.opponents
        # the IL share is the base model's (il_v4); the extra IL slot has none
        self.assertEqual(o.pilot, {"self_seat": 0.4, "mirror": 0.1, "il_v4_t1": 0.3, "il_v4_tstar": 0.2,
                                   "il_v3_t1": 0.0, "snapshot": 0.0})
        self.assertEqual(o.main, {"self_seat": 0.4, "mirror": 0.1, "il_v4_t1": 0.15, "il_v4_tstar": 0.1,
                                  "il_v3_t1": 0.0, "snapshot": 0.25})
        self.assertEqual(o.temperature["snapshot"], 1.0)          # the snapshot pool's temperature
        self.assertEqual((o.pilot_updates, o.il_floor, o.snapshot_every, o.snapshot_pool_recent), (10, 0.25, 5, 4))
        self.assertEqual((o.temperature["il_v4_t1"], o.temperature["il_v3_t1"], o.temperature["il_v2_heldout"]),
                         (1.0, 1.0, 1.0))

    def test_gate_numbers(self):                          # gates, rollback and stall rules
        g = self.c.gates
        self.assertEqual((g.screen.at_update_in_cycle, g.screen.games), (5, 256))
        self.assertEqual((g.g1.every_updates, g.g1.games_per_side, g.g1.alpha_one_sided, g.g1.heldout_min_delta_pp),
                         (10, 500, 0.1, -2.0))
        self.assertEqual((g.g1c.pairs, g.g2.pairs, g.g2.record_only), (500, 250, True))
        self.assertEqual((g.release.pairs, g.release.mirror_games, g.release.transitivity_pairs,
                          g.release.live_games_min), (500, 200, 100, 10))
        self.assertEqual(g.crn_panel, {"il_v2_heldout": 0.4, "il_v4_tstar": 0.3, "il_v4_t1": 0.3})
        self.assertEqual((g.min_completion, g.bootstrap_resamples, g.aa_null_pairs, g.sweep_games),
                         (0.95, 5000, 500, 500))
        self.assertEqual(g.il_temperature_sweep, [1.0, 0.5, 0.0])                      # 0.0 = greedy
        self.assertEqual(g.deploy_temperature_sweep, [1.0, 0.7, 0.3, 0.0])
        self.assertEqual((g.sprt, g.sprt_h1_pp, g.pilot_min_delta_pp), (False, 3.0, -5.0))
        self.assertEqual((g.rollback_delta_pp, g.rollback_consecutive, g.rollback_beta_double_window),
                         (-5.0, 3, 40))
        self.assertEqual((g.stall_g1_without_promotion, g.stall_stop_g1, g.release_after_g1_without_new), (5, 8, 5))

    def test_monitor_numbers(self):                       # the monitor table
        m = self.c.monitor
        self.assertEqual((m.parity.max_abs_dlogp_fp32, m.parity.ratio_tol, m.parity.kl_theta0_ref_max,
                          m.parity.replay_games, m.parity.games), (1e-3, 1e-3, 1e-5, 4, 10))
        self.assertEqual((m.clip_frac_max, m.clip_frac_consecutive), (0.4, 4))
        self.assertEqual((m.kl_stall_window, m.kl_stall_band, m.kl_stall_g1_without_promotion, m.lr_stall_mult),
                         (8, 0.2, 4, 2.0))
        self.assertEqual((m.plays_per_min_band, m.entropy_min_frac_of_il, m.card_rate_band),
                         ((0.8, 1.25), 0.4, (0.25, 4.0)))
        self.assertEqual((m.core_cards, m.core_card_min_frac_of_il, m.core_card_consecutive,
                          m.pilot_core_card_min_frac), (["example-card"], 0.2, 4, 0.25))
        self.assertEqual((m.illegal_cell_code, m.legal_refused_max_ratio_vs_il, m.mask_recheck_every), (19, 3.0, 10))
        self.assertEqual((m.drop_rate_warn, m.drop_rate_stop, m.opp_nonwait_min_frac), (0.01, 0.05, 0.4))
        self.assertEqual((m.value_ev_overall_min, m.value_ev_last_third_min, m.value_ev_consecutive,
                          m.value_ev_drop_from_warmup), (0.1, 0.2, 4, 0.2))
        self.assertEqual((m.drift_every, m.throughput_min_frac, m.min_free_mem_gb, m.swap_growth_gb, m.swap_window_s,
                          m.swap_slot_reduction, m.nan_skip_consecutive_stop), (10, 0.5, 4.0, 1.0, 3600.0, 4, 5))
        self.assertEqual((m.temp_lr_mult, m.kl_stall_beta_mult, m.core_rollbacks_before_stop), (0.6, 0.6, 2))
        g = self.c.gates                                  # rollback / stall factors
        self.assertEqual((g.rollback_lr_mult, g.rollback_beta_mult, g.stall_step_min_updates), (0.6, 3.0, 10))
        self.assertEqual((g.p6_entry_g2_min_score, g.restart_swap_min_score), (0.6, 0.6))

    def test_sim2real_numbers(self):                      # sim-to-real settings
        s = self.c.sim2real
        self.assertTrue(s.enabled)
        self.assertEqual((s.landing_delay.play, s.landing_delay.ability), ((1, 4), (1, 3)))
        self.assertEqual(s.delta_lag.default_ticks, 0)            # no display lag, every object family
        self.assertEqual(s.delta_lag.overrides, {})
        self.assertTrue(s.threat_from_filtered_state)
        if not (ROOT / s.delta_arr_table).is_file():
            self.skipTest(VOCAB_SKIP)                     # the legacy arrival table is private data
        tab = json.loads((ROOT / s.delta_arr_table).read_text())
        self.assertGreater(tab["n"], 0)                   # the legacy arrival table's sample count

    def test_critic_and_appendix_a(self):
        c = self.c
        w = c.critic.warmup
        self.assertEqual((w.games_per_batch, w.max_updates, w.fixed_set_games, w.stop_trend_updates,
                          w.pass_ev_overall, w.pass_ev_last_third), (256, 20, 1000, 4, 0.1, 0.2))
        self.assertEqual(w.games_per_batch * w.max_updates, 5120)                      # at most this many games
        self.assertEqual((c.critic.loss, c.critic.lr, c.critic.grad_clip, c.critic.epochs), ("mse", 3e-4, 2.0, 3))
        self.assertEqual((c.policy.grad_clip_raise_after_updates, c.policy.grad_clip_raise_clipped_frac), (10, 0.8))
        self.assertEqual((c.policy.lr_schedule, c.policy.advantage_norm, c.policy.rollout_lag), ("constant",
                                                                                               "per_update", 0))
        self.assertEqual((c.entropy.cell_bonus, c.entropy.cell_trigger_frac_of_il), (5e-3, 0.25))
        self.assertEqual(c.early_stop.lr_halve_if_stopped_frac, 0.75)
        self.assertEqual((c.reward.win, c.reward.loss, c.reward.draw), (1.0, -1.0, 0.0))
        self.assertEqual((c.bc.start_weight, c.bc.min_rows_per_update), (0.1, 10000))
        self.assertEqual((c.storage.rows, c.storage.keep_updates), ("committed_tokens", 2))
        self.assertEqual((c.user.kl_reference_after_promotion, c.user.h1_reading, c.user.bc_enabled,
                          c.user.opponent_design, c.user.d6_subset_alarm_only),
                         ("champion", "crn_then_seatswap", False, "selfplay_il30", True))

    def test_ht_wait_is_exact_inverse_kappa(self):
        for s, k in SMP.KAPPA.items():
            self.assertEqual(Fraction(self.c.weights.ht_wait[s]).limit_denominator(1000), 1 / k)
            self.assertEqual(self.c.weights.ht_wait[s], float(1 / k))

    def test_derived_rows_per_update(self):               # trained seat-games per update and minibatches
        seats = self.c.trained_seats_per_game("main")
        self.assertAlmostEqual(seats, 1.5, places=12)
        self.assertAlmostEqual(self.c.rollout.games_per_update * seats, 1500.0, places=9)
        self.assertEqual(round(self.c.rollout.games_per_update * seats / self.c.policy.minibatch_seat_games), 30)
        self.assertAlmostEqual(self.c.describe()["derived"]["lambda_per_block"], 0.951, places=3)


# ============================================================================ generic robustness
class TestTreeWalks(unittest.TestCase):
    def test_unknown_key_at_every_mapping_level(self):
        base = C.PPOConfig().to_dict()
        for path, _ in _mappings(base):
            with self.subTest(path=path or "<root>"):
                d = copy.deepcopy(base)
                (_get(d, path) if path else d)["zz_unknown"] = 1
                with self.assertRaises(C.ConfigError):
                    C.from_dict(d)

    def test_null_only_at_declared_open_items(self):
        base = _ready()
        # fp32 and no il_v3 share are the defaults; switch both on so that the bf16 tolerances and the extra-IL
        # settings are open items again and their readiness reporting stays covered
        base["policy"]["backbone_dtype"] = "bf16"
        base["opponents"]["main"].update(il_v3_t1=0.05, il_v4_t1=0.1)
        accepted = set()
        for path, _ in _leaves(base):
            d = copy.deepcopy(base)
            _set(d, path, None)
            try:
                C.from_dict(d)
            except C.ConfigError:
                continue
            accepted.add(path)
        self.assertLessEqual(accepted, OPTIONAL_LEAVES | NULL_BY_DESIGN)
        # every null-able leaf is reported by readiness when null on an otherwise ready config
        for path in sorted(accepted - NULL_BY_DESIGN):
            d = copy.deepcopy(base)
            _set(d, path, None)
            with self.subTest(path=path):
                self.assertIn(path, [m.split(":")[0] for m in C.from_dict(d).missing("policy")])
        self.assertNotIn("opponents.temperature.il_v2_heldout", accepted)       # fixed at T = 1

    def test_override_every_leaf_with_its_own_value(self):
        base = _ready()
        want = C.from_dict(base).digest()
        for path, v in _leaves(base):
            if isinstance(v, dict):                       # empty mappings: none in the tree today
                continue
            with self.subTest(path=path):
                got = C.from_dict(C.apply_overrides(base, [f"{path}={_yaml_scalar(v)}"])).digest()
                self.assertEqual(got, want)

    def test_from_dict_does_not_alias_input(self):
        d = _ready()
        c = C.from_dict(d)
        d["user"]["deck"][0] = "zz"
        d["opponents"]["pilot"]["mirror"] = 0.9
        d["weights"]["ht_wait"]["base"] = 0.0
        self.assertEqual(c.user.deck[0], DECK_A[0])
        self.assertEqual(c.opponents.pilot["mirror"], 0.1)
        self.assertEqual(c.weights.ht_wait["base"], SMP.WEIGHT["base"])

    def test_describe_is_strict_json_and_stable(self):
        c = C.from_dict(_ready())
        rec = c.describe()
        blob = json.dumps(rec, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(blob), rec)
        self.assertEqual(rec["readiness"], {"rollout": [], "policy": []})
        self.assertEqual(rec["digest"], C.from_dict(rec["config"]).digest())

    def test_non_finite_strings_rejected(self):
        for s in ("nan", "inf", "-inf", "1e400", "Infinity"):
            with self.subTest(s=s):
                with self.assertRaisesRegex(C.ConfigError, r"policy.lr"):
                    C.from_dict({"policy": {"lr": s}})

    @needs_yaml
    def test_overrides_on_yaml_file(self):
        load_yaml = _load_yaml()
        data = load_yaml(YAML_PATH.read_text(encoding="utf-8"))
        out = C.apply_overrides(data, ["opponents.temperature.il_v4_tstar=0.0", "user.deck=[" + ", ".join(DECK_B) + "]",
                                       "monitor.core_cards=[example-card, core-b]"])
        c = C.from_dict(out)
        self.assertEqual(c.opponents.temperature["il_v4_tstar"], 0.0)
        self.assertEqual(c.user.deck, DECK_B)
        self.assertEqual(data["user"]["deck"], None)      # the input is not modified


# ============================================================================ YAML dialects
@needs_yaml
class TestYamlDialects(unittest.TestCase):
    def test_subset_reads_null_population_as_string(self):
        load_yaml = _load_yaml()
        text = YAML_PATH.read_text(encoding="utf-8").replace("    11: null ", "    11: NULL ", 1)
        self.assertIn("11: NULL", text)
        try:
            import yaml as _y  # noqa: F401
            self.skipTest("PyYAML present: load_yaml uses it")
        except ImportError:
            pass
        self.assertEqual(C.from_dict(load_yaml(text)).levels.population[11], "NULL")

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_pyyaml_reads_main_file_as_defaults(self):
        raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
        self.assertEqual(C.from_dict(raw).digest(), C.PPOConfig().digest())

    def test_bare_null_population_refused_by_load_config(self):
        # YAML 1.1 reads NULL / Null as null; the il_v7 subset reads them as strings: load_config refuses
        # the bare form in either dialect, the quoted form is the same string in both.
        text = YAML_PATH.read_text(encoding="utf-8")
        self.assertIn('"NULL" (quoted)', text)
        with self.assertRaisesRegex(C.ConfigError, r"bare 'NULL'"):
            C.lint_yaml_text(text.replace("    11: null ", "    11: NULL ", 1))
        C.lint_yaml_text(text.replace("    11: null ", '    11: "NULL" ', 1))

    @unittest.skipUnless(yaml is not None, "PyYAML not installed")
    def test_pyyaml_reads_quoted_null_population_like_the_subset(self):
        text = YAML_PATH.read_text(encoding="utf-8").replace("    11: null ", '    11: "NULL" ', 1)
        self.assertEqual(C.from_dict(yaml.safe_load(text)).levels.population[11], "NULL")


# ============================================================================ defects (expected failures)
class TestDefects(unittest.TestCase):
    """Regression tests for the defects of the review (each was an expectedFailure before the fix)."""

    def test_rollout_temperature_fixed_at_one(self):
        # theta_k slot / cells / ability sampled at T = 1, never greedy
        for t in (0.0, 0.5):
            d = C.PPOConfig().to_dict()
            d["rollout"]["temperature"] = t
            with self.assertRaises(C.ConfigError):
                C.from_dict(d)

    def test_frozen_opponent_temperatures_reported_when_null(self):
        d = _ready()
        d["opponents"]["temperature"]["snapshot"] = None
        c = C.from_dict(d)                                # main has a snapshot share
        for st in C.READINESS_STAGES:
            self.assertIn("opponents.temperature.snapshot", [m.split(":")[0] for m in c.missing(st)])
        d = _ready()
        d["opponents"]["temperature"]["il_v2_heldout"] = None
        with self.assertRaisesRegex(C.ConfigError, r"il_v2_heldout: must be 1.0"):
            C.from_dict(d)                                # the G1 held-out panel plays at T = 1

    @needs_cases
    def test_cases_writer_needs_king_level_equal_to_case_level(self):
        """The case builder's modify_input refuses kt != the case's level; the config's kt is that level."""
        from ppo_v7 import cases as K
        c = C.from_dict(_ready())
        offsets = {i: 1 for i in range(1, 9)}
        offsets[TOWER_PRINCESS_GID] = 1
        side = {"sp": [{"d": i, "l": 10} for i in range(1, 9)], "sc": [{"d": TOWER_PRINCESS_GID, "l": 10}]}
        inp11 = {"battle": {"hbd": [{"kt": 11}, {"kt": 11}], "deck0": copy.deepcopy(side),
                            "deck1": copy.deepcopy(side)}, "rndSeed": 7}
        self.assertEqual(K.input_level(inp11, offsets), 11)
        K.modify_input(inp11, our_owner=0, tower_troop=c.user.tower_troop, king_level=c.king_level(11), rnd_seed=1,
                       offsets=offsets)
        with self.assertRaises(K.LevelError):
            K.modify_input(inp11, our_owner=0, tower_troop=c.user.tower_troop, king_level=c.king_level(16),
                           rnd_seed=1, offsets=offsets)

    def test_king_level_follows_level_mix(self):
        # only all-11 or all-16 games; every level with a share gets kt = its level
        c = C.from_dict(_ready())
        for lv, share in c.levels.mix.items():
            if share > 0:
                self.assertEqual(c.king_level(lv), lv)
        d = _ready()
        d["user"]["king_level"] = 16                      # no single account-wide kt any more
        with self.assertRaisesRegex(C.ConfigError, r"unknown key"):
            C.from_dict(d)

    def test_levels_only_11_or_16(self):
        d = _ready()
        d["levels"]["mix"] = {"16": 0.8, "13": 0.2}
        with self.assertRaises(C.ConfigError):
            C.from_dict(d)
        with self.assertRaises(C.ConfigError):
            C.from_dict(_ready()).king_level(13)

    @needs_cases
    def test_tower_troop_takes_native_gid(self):
        # the case builder's modify_input(tower_troop: int) takes the game gid (159000000 = tower-princess)
        from ppo_v7 import cases as K
        c = C.from_dict(_ready(tower_troop=TOWER_PRINCESS_GID))
        self.assertEqual(c.user.tower_troop, TOWER_PRINCESS_GID)
        self.assertIn(c.user.tower_troop, K.TOWER_TROOPS)
        self.assertEqual(C.TOWER_TROOP_GIDS, {g: key for g, (key, _) in K.TOWER_TROOPS.items()})
        with self.assertRaises(C.ConfigError):
            C.from_dict(_ready(tower_troop=159000003))    # not a known tower troop

    @needs_cases
    def test_opponent_tables_agree_with_cases(self):
        """The config's and the case builder's opponent / level tables use the same names and agree."""
        from ppo_v7 import cases as K
        c = C.PPOConfig()
        for ph in C.PHASES:
            self.assertEqual(c.effective_mix(ph), K.OPPONENT_MIX[ph])
        self.assertEqual(K.PILOT_UPDATES, c.opponents.pilot_updates)
        self.assertEqual(K.IL_FLOOR, c.opponents.il_floor)
        self.assertEqual(K.DEFAULT_LEVEL_MIX, c.levels.mix)

    @needs_cases
    def test_effective_mix_feeds_cases_planner(self):
        from ppo_v7 import cases as K
        c = C.PPOConfig()
        for ph in C.PHASES:
            K.check_mix(c.effective_mix(ph), ph)

    def test_monitor_rollback_units(self):
        """gates.rollback_delta_pp is in pp; monitor_thresholds() hands monitor.Thresholds a fraction."""
        from amini.ppo import monitor as M
        th = M.Thresholds()
        c = C.from_dict(_ready())
        self.assertAlmostEqual(c.gates.rollback_delta_pp, 100 * th.rollback_delta)
        self.assertAlmostEqual(c.monitor_thresholds()["rollback_delta"], th.rollback_delta)
        self.assertEqual(set(c.monitor_thresholds()["identity_logp_tol"]), set(th.identity_logp_tol))

    def test_monitor_config_builds_monitor_thresholds(self):
        from amini.ppo import monitor as M
        fp32 = C.from_dict(C.apply_overrides(_ready(), ["policy.backbone_dtype=fp32"]))
        self.assertEqual(M.Thresholds.from_mapping(fp32.monitor_thresholds()), M.Thresholds())
        bf16 = M.Thresholds.from_mapping(C.from_dict(C.apply_overrides(
            _ready(), ["policy.backbone_dtype=bf16"])).monitor_thresholds())
        self.assertEqual(dict(bf16.identity_logp_tol), {"wait": 5e-4, "play": 2e-4, "ability": 2e-4})
        self.assertEqual(M.Thresholds.from_mapping(C.PPOConfig().monitor_thresholds()), M.Thresholds())   # fp32
        with self.assertRaises(C.ConfigError):            # bf16 tolerances not measured yet
            C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), ["policy.backbone_dtype=bf16"])).monitor_thresholds()

    def test_greedy_zero_translated_for_apply_temperature(self):
        """logprob refuses T = 0; config.sampling_mode turns the 0.0 = greedy convention into a flag."""
        import torch
        from amini.ppo import logprob as LP
        c = C.from_dict(C.apply_overrides(_ready(), ["opponents.temperature.il_v4_tstar=0.0"]))
        t = c.opponents.temperature["il_v4_tstar"]
        with self.assertRaises(ValueError):
            LP.apply_temperature({"slot": torch.zeros(1, 4)}, t_slot=t)
        self.assertEqual(C.sampling_mode(t), (True, None))
        greedy, t2 = C.sampling_mode(0.5)
        self.assertFalse(greedy)
        LP.apply_temperature({"slot": torch.zeros(1, 4)}, t_slot=t2)

    def test_deck_b_needs_its_core_card(self):
        # a core-card candidate in the deck must be monitored
        with mock.patch.object(C, "CORE_CARD_CANDIDATES", ("example-card", "core-b")):
            with self.assertRaises(C.ConfigError):
                C.from_dict(_ready(deck=list(DECK_B)))
            d = _ready(deck=list(DECK_B))
            d["monitor"]["core_cards"] = ["example-card", "core-b"]
            C.from_dict(d)

    @unittest.skipUnless(HAVE_VOCAB, VOCAB_SKIP)
    def test_delta_lag_names_are_object_vocab_keys(self):
        ids = json.loads((ROOT / "data" / "object_vocab_v7.json").read_text())["ids"]
        self.assertLessEqual(set(C.PPOConfig().sim2real.delta_lag.overrides), set(ids))

    @unittest.skipUnless(HAVE_VOCAB, VOCAB_SKIP)
    def test_delta_lag_override_typo_rejected(self):
        d = C.PPOConfig().to_dict()
        d["sim2real"]["delta_lag"]["overrides"]["Fireball"] = 0       # a real name is accepted ...
        C.from_dict(d)
        d["sim2real"]["delta_lag"]["overrides"]["Firebal"] = d["sim2real"]["delta_lag"]["overrides"].pop("Fireball")
        with self.assertRaises(C.ConfigError):
            C.from_dict(d)

    def test_negative_zero_same_digest(self):
        d = C.PPOConfig().to_dict()
        d["policy"]["weight_decay"] = -0.0
        c = C.from_dict(d)
        self.assertEqual(c, C.PPOConfig())
        self.assertEqual(c.digest(), C.PPOConfig().digest())

    @needs_yaml
    def test_repeated_section_rejected(self):
        import tempfile
        text = YAML_PATH.read_text(encoding="utf-8").replace("  lr: 5.0e-5 ", "  lr: 1.0e-5 ", 1)
        text += "\npolicy:\n  epochs: 3\n"                # e.g. an appended 'override' block
        with self.assertRaisesRegex(C.ConfigError, r"'policy' appears twice"):
            C.lint_yaml_text(text)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run.yaml"
            p.write_text(text, encoding="utf-8")
            with self.assertRaises(C.ConfigError):
                C.load_config(p)

    def test_huge_int_is_a_config_error(self):
        with self.assertRaises(C.ConfigError):
            C.from_dict({"policy": {"lr": 10 ** 400}})

    def test_mutated_config_is_not_described(self):
        c = C.PPOConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.kl.beta = 0.5                               # type: ignore[misc]
        c.levels.mix[11] = 0.5                            # dict contents stay mutable ...
        with self.assertRaises(C.ConfigError):
            c.describe()                                  # ... but the run record re-validates
        with self.assertRaises(C.ConfigError):
            c.digest()


if __name__ == "__main__":
    unittest.main()
