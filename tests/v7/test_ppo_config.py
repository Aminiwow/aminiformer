"""PPO config tests: src/amini/ppo/config.py and the run file configs/ppo_v7_main.yaml.

* the run file loads with il_v7.load_yaml, spells out every field and equals the dataclass defaults;
* unknown keys are refused at every level; types, ranges and cross-field rules are checked;
* the readiness check reports the open choices and measurements per stage;
* the digest is stable (key order, fresh interpreter with another PYTHONHASHSEED) and sensitive;
* overrides, describe() and the derived values (opponent mix, trained seats, phase);
* the shared schemas: opponent kinds / level keys / tower troop gids of the case builder and the
  field set and units of amini.ppo.monitor.Thresholds; the YAML lint (repeated keys, NULL).

The run file, the object vocabulary and the case builder are not published: the tests that need them are
skipped.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_config.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import dataclasses
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from amini.ppo import config as C  # noqa: E402
from amini.train.il_v7 import load_yaml  # noqa: E402
from amini.ppo import gae as SMP  # noqa: E402  (KAPPA and WEIGHT = 1 / KAPPA)

YAML_PATH = ROOT / "configs" / "ppo_v7_main.yaml"
YAML_SKIP = "needs the run configuration file configs/ppo_v7_main.yaml (not published)"
HAVE_VOCAB = (ROOT / C.OBJECT_VOCAB).is_file()             # delta_lag override names are checked against it
VOCAB_SKIP = "needs the private dataset"
HAVE_CASES = importlib.util.find_spec("ppo_v7.cases") is not None
CASES_SKIP = "needs an unpublished module (ppo_v7.cases)"
needs_yaml = unittest.skipUnless(YAML_PATH.is_file(), YAML_SKIP)
needs_cases = unittest.skipUnless(HAVE_CASES, CASES_SKIP)
# placeholder card keys (two evolution slots and one hero slot); "example-card" is the default core card
DECK = ["card-a", "card-b-ev1", "example-card", "card-d", "card-e-ev1", "card-f", "card-g-hero", "card-h"]


def _str_keys(x):
    if isinstance(x, dict):
        return {str(k): _str_keys(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_str_keys(v) for v in x]
    return x


def _raw() -> dict:
    return load_yaml(YAML_PATH.read_text(encoding="utf-8"))


def _ready(**user) -> dict:
    """The default config as a dict with every rollout / policy prerequisite filled in."""

    d = C.PPOConfig().to_dict()
    d["user"].update({"deck": list(DECK), "tower_troop": 159000000}, **user)
    d["rollout"]["hazard_dt"] = "block"
    d["opponents"]["checkpoint"].update({"il_v4": "/x/v4.pt", "il_v3": "/x/v3.pt", "il_v2_heldout": "/x/v2.pt"})
    d["opponents"]["checkpoint_sha256"].update({"il_v4": "a" * 64, "il_v3": "b" * 64, "il_v2_heldout": "c" * 64})
    d["opponents"]["hazard_dt"].update({"il_v3": "interval", "il_v2_heldout": "block"})
    d["opponents"]["code_tree"].update({"il_v3": "v3-frozen", "il_v2_heldout": "v2-frozen"})
    d["opponents"]["temperature"]["il_v4_tstar"] = 0.5
    d["levels"]["population"]["11"] = "challenge_eventdeck"
    d["monitor"]["parity"]["dlogp_tol_bf16"] = {"wait": 5e-4, "play": 2e-4, "ability": 2e-4}
    return d


def _keys(d: dict, prefix: str = "") -> set:
    out = set()
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        out.add(p)
        if isinstance(v, dict):
            out |= _keys(v, p)
    return out


class TestYamlFile(unittest.TestCase):
    @needs_yaml
    def test_yaml_equals_defaults(self):
        raw = _raw()
        # PyYAML (used by load_yaml when installed) reads numeric mapping keys as ints; the built-in
        # subset parser keeps them as strings. Compare with string keys so both give the same answer.
        self.assertEqual(_str_keys(raw), C.PPOConfig().to_dict())
        self.assertEqual(C.from_dict(raw), C.PPOConfig())

    @needs_yaml
    def test_yaml_spells_out_every_field(self):
        self.assertEqual(_keys(_raw()), _keys(C.PPOConfig().to_dict()))

    @needs_yaml
    def test_load_config_equals_defaults(self):
        cfg = C.load_config(YAML_PATH)
        self.assertEqual(cfg, C.PPOConfig())
        self.assertEqual(cfg.digest(), C.PPOConfig().digest())

    @needs_yaml
    def test_yaml_stays_in_load_yaml_subset(self):
        for i, line in enumerate(YAML_PATH.read_text(encoding="utf-8").splitlines(), 1):
            body = line.split(" #")[0] if not line.lstrip().startswith("#") else ""
            if not body.strip():
                continue
            self.assertFalse(body.lstrip().startswith("- "), f"line {i}: block list")
            self.assertNotIn("{", body.replace("{}", ""), f"line {i}: flow mapping")
            self.assertNotIn("&", body, f"line {i}: anchor")
            body = re.sub(r'"[^"]*"', '""', body)      # quoted strings (e.g. a sha256) are not numbers
            self.assertIsNone(re.search(r"(?<![\w.])\d+[eE][-+]?\d", body), f"line {i}: float without a dot")
            self.assertIsNone(re.search(r"\d[eE]\d", body), f"line {i}: exponent without a sign")

    def test_plan_numbers(self):
        c = C.PPOConfig()
        self.assertEqual(c.rollout.games_per_update, 1000)
        self.assertEqual((c.policy.clip_eps, c.policy.dual_clip, c.policy.lr), (0.25, 2.5, 5e-5))
        self.assertEqual((c.policy.betas, c.policy.adam_eps, c.policy.weight_decay), ((0.9, 0.999), 1e-8, 0.0))
        self.assertEqual((c.policy.grad_clip, c.policy.grad_clip_raised), (1.0, 2.0))
        self.assertEqual((c.policy.minibatch_seat_games, c.policy.epochs), (50, 3))
        self.assertEqual((c.kl.beta, c.kl.beta_min, c.kl.beta_max), (0.02, 0.005, 0.1))
        self.assertEqual((c.early_stop.joint_k3, c.early_stop.play_k3, c.early_stop.gate_kl), (0.02, 0.1, 0.01))
        self.assertEqual(set(c.entropy.coef.values()), {0.0})
        self.assertEqual(c.entropy.cell_bonus, 5e-3)
        self.assertEqual((c.gae.gamma, c.gae.lambda_tick), (0.9999, 0.995))
        self.assertEqual((c.critic.lr, c.critic.grad_clip, c.critic.epochs), (3e-4, 2.0, 3))
        self.assertEqual((c.critic.warmup.games_per_batch, c.critic.warmup.max_updates,
                          c.critic.warmup.fixed_set_games), (256, 20, 1000))
        self.assertEqual((c.reward.shaping, c.reward.short_head_coef, c.reward.short_head_gamma), ("none", 0.2, 0.99))
        self.assertEqual((c.bc.weight, c.policy.aux_heads, c.policy.dropout), (0.0, False, 0.0))
        self.assertEqual((c.weights.seat_ours, c.weights.seat_theta_opp, c.weights.seat_mirror), (1.0, 0.25, 1.0))
        self.assertEqual(c.weights.ht_wait, SMP.WEIGHT)
        self.assertEqual((c.sim2real.landing_delay.play, c.sim2real.landing_delay.ability), ((1, 4), (1, 3)))
        self.assertEqual(c.storage.keep_updates, 2)
        self.assertIsNone(c.rollout.hazard_dt)            # the base model's mode, an open item
        self.assertEqual((c.rollout.temperature, c.rollout.gate_temperature), (1.0, 1.0))
        self.assertEqual((c.policy.is_ratio_cap_lag1, c.policy.gae_trace_cap_lag1), (1.5, 1.0))
        self.assertEqual(c.levels.mix, {16: 0.8, 11: 0.2})
        self.assertTrue(0.0 < c.levels.mix[11] < 1.0)
        self.assertEqual((c.gates.p6_entry_g2_min_score, c.gates.restart_swap_min_score), (0.6, 0.6))
        self.assertEqual((c.gates.rollback_lr_mult, c.gates.rollback_beta_mult, c.gates.stall_step_min_updates),
                         (0.6, 3.0, 10))
        self.assertEqual((c.monitor.temp_lr_mult, c.monitor.kl_stall_beta_mult, c.monitor.lr_stall_mult),
                         (0.6, 0.6, 2.0))
        # numbers the sibling modules read from the config
        self.assertEqual(c.reward.short_head_decision_ticks, 20)           # one decision block per short-head step
        self.assertEqual(c.monitor.temp_lr_floor, 0.4)
        self.assertEqual(c.gates.stall_games_per_update, 2000)             # the stall ladder's games per update
        self.assertEqual(c.gates.pilot_joint_kl_median_max, 0.02)          # pilot exit
        self.assertEqual((c.chain.block_timeout_s, c.chain.kill_after_s), (14400, 600))   # 4 h block

    @unittest.skipUnless((ROOT / C.PPOConfig().sim2real.delta_arr_table).is_file(), VOCAB_SKIP)
    def test_delta_arr_table_exists(self):
        self.assertTrue((ROOT / C.PPOConfig().sim2real.delta_arr_table).is_file())


class TestStrictBuild(unittest.TestCase):
    def _bad(self, mutate, pattern: str):
        d = C.PPOConfig().to_dict()
        mutate(d)
        with self.assertRaisesRegex(C.ConfigError, pattern):
            C.from_dict(d)

    def test_unknown_keys_rejected(self):
        self._bad(lambda d: d.update(learning_rate=1e-5), r"<root>: unknown key\(s\) \['learning_rate'\]")
        self._bad(lambda d: d["policy"].update(clip=0.2), r"policy: unknown key\(s\) \['clip'\]")
        self._bad(lambda d: d["critic"]["warmup"].update(games=1), r"critic.warmup: unknown key")
        self._bad(lambda d: d["gates"]["g1"].update(alpha=0.1), r"gates.g1: unknown key")
        self._bad(lambda d: d["opponents"]["pilot"].update(il_v5_t1=0.0), r"opponents.pilot: keys must be exactly")
        self._bad(lambda d: d["entropy"]["coef"].update(value=0.0), r"entropy.coef: keys must be exactly")
        self._bad(lambda d: d["opponents"]["temperature"].pop("il_v3_t1"), r"opponents.temperature: keys")

    @needs_yaml
    def test_unknown_key_in_yaml_text(self):
        text = YAML_PATH.read_text(encoding="utf-8").replace("  epochs: 3\n", "  epochs: 3\n  epoch: 4\n", 1)
        with self.assertRaisesRegex(C.ConfigError, r"policy: unknown key\(s\) \['epoch'\]"):
            C.from_dict(load_yaml(text))

    def test_types_rejected(self):
        self._bad(lambda d: d["policy"].__setitem__("epochs", 2.5), r"policy.epochs: expected an integer")
        self._bad(lambda d: d["policy"].__setitem__("epochs", True), r"policy.epochs: expected an integer")
        self._bad(lambda d: d["policy"].__setitem__("lr", "fast"), r"policy.lr: expected a number")
        self._bad(lambda d: d["policy"].__setitem__("lr", None), r"policy.lr: null is not allowed")
        self._bad(lambda d: d["policy"].__setitem__("aux_heads", 0), r"policy.aux_heads: expected true/false")
        self._bad(lambda d: d["policy"].__setitem__("betas", [0.9]), r"policy.betas: expected a list of 2")
        self._bad(lambda d: d.__setitem__("kl", 0.01), r"kl: expected a mapping")
        self._bad(lambda d: d["policy"].__setitem__("lr", float("nan")), r"policy.lr: must be finite")

    def test_numeric_strings_and_ints_coerced(self):
        d = C.PPOConfig().to_dict()
        d["policy"]["lr"] = "5e-5"          # what PyYAML 1.1 returns for '5e-5'
        d["policy"]["grad_clip"] = 1        # an int where a float is expected
        self.assertEqual(C.from_dict(d), C.PPOConfig())

    def test_out_of_range_rejected(self):
        cases = [
            (lambda d: d["policy"].__setitem__("lr", -1e-5), r"policy.lr: .* outside"),
            (lambda d: d["policy"].__setitem__("clip_eps", 1.5), r"policy.clip_eps: .* outside"),
            (lambda d: d["policy"].__setitem__("dual_clip", 1.0), r"policy.dual_clip: .* outside"),
            (lambda d: d["policy"].__setitem__("grad_clip_raised", 0.4), r"policy.grad_clip_raised"),
            (lambda d: d["policy"].__setitem__("dropout", 0.1), r"policy.dropout: must be 0.0"),
            (lambda d: d["kl"].__setitem__("beta", 0.001), r"kl.beta: .* outside"),
            (lambda d: d["kl"].__setitem__("beta", 0.2), r"kl.beta: .* outside"),
            (lambda d: d["entropy"]["coef"].__setitem__("gate", 1e-3), r"entropy.coef.gate: must be 0.0"),
            (lambda d: d["gae"].__setitem__("lambda_tick", 1.2), r"gae.lambda_tick"),
            (lambda d: d["gae"].__setitem__("gamma", 0.0), r"gae.gamma"),
            (lambda d: d["rollout"].__setitem__("gate_temperature", 0.5), r"rollout.gate_temperature: must be 1.0"),
            (lambda d: d["rollout"].__setitem__("temperature", 1.5), r"rollout.temperature: must be 1.0"),
            (lambda d: d["rollout"].__setitem__("temperature", 0.0), r"rollout.temperature: must be 1.0"),
            (lambda d: d["rollout"].__setitem__("temperature", 0.5), r"rollout.temperature: must be 1.0"),
            (lambda d: d["rollout"].__setitem__("hazard_dt", "auto"), r"rollout.hazard_dt: 'auto' not one of"),
            (lambda d: d["rollout"].__setitem__("max_env_slots", 65), r"rollout.max_env_slots"),   # ceiling 64
            (lambda d: d["rollout"].__setitem__("replicas_per_group", 3), r"must divide games_per_update"),
            (lambda d: d["opponents"]["main"].__setitem__("snapshot", 0.4), r"opponents.main: shares must sum to 1"),
            (lambda d: d["opponents"]["pilot"].update(snapshot=0.1, self_seat=0.3),
             r"opponents.pilot.snapshot: must be 0.0"),
            (lambda d: d["opponents"]["main"].update(il_v4_t1=0.1, snapshot=0.3), r"IL share 0.200 below il_floor"),
            (lambda d: d["opponents"]["temperature"].__setitem__("il_v4_t1", 0.5), r"temperature.il_v4_t1: must be"),
            (lambda d: d["opponents"]["temperature"].__setitem__("il_v2_heldout", None),
             r"temperature.il_v2_heldout: must be 1.0"),
            (lambda d: d["user"].__setitem__("deck", ["a"] * 8), r"user.deck: must be 8 distinct"),
            (lambda d: d["user"].__setitem__("deck", ["a", "b"]), r"user.deck: must be 8 distinct"),
            (lambda d: d["user"].__setitem__("king_level", 16), r"user: unknown key\(s\) \['king_level'\]"),
            (lambda d: d["user"].__setitem__("tower_troop", 26000034), r"user.tower_troop: 26000034 not one of"),
            (lambda d: d["user"].__setitem__("tower_troop", "tower-princess"), r"user.tower_troop: expected an integer"),
            (lambda d: d["user"].__setitem__("kl_reference_after_promotion", "v2"), r"kl_reference_after_promotion"),
            (lambda d: d["user"].__setitem__("h1_reading", "seatswap"), r"user.h1_reading"),
            (lambda d: d["user"].__setitem__("opponent_design", "pfsp"), r"user.opponent_design"),
            (lambda d: d["user"].__setitem__("bc_enabled", True), r"bc.weight: user.bc_enabled is true"),
            (lambda d: d["bc"].__setitem__("weight", 0.05), r"bc.weight: must be 0 while"),
            (lambda d: d["weights"]["ht_wait"].__setitem__("base", 4.5), r"weights.ht_wait.base: must equal 1/KAPPA"),
            (lambda d: d["weights"].__setitem__("forced", 0.1), r"weights.forced: must be 0.0"),
            (lambda d: d["sim2real"]["landing_delay"].__setitem__("play", [4, 1]), r"landing_delay.play\[1\]"),
            (lambda d: d["sim2real"]["landing_delay"].__setitem__("ability", [1, 25]), r"landing_delay.ability\[1\]"),
            (lambda d: d["sim2real"]["delta_lag"]["overrides"].__setitem__("Rocket", -1), r"overrides.Rocket"),
            (lambda d: d["sim2real"]["delta_lag"]["overrides"].__setitem__("Firebal", 4),
             r"overrides: \['Firebal'\] are not object names"),
            (lambda d: d["levels"]["mix"].update({"16": 0.9}), r"levels.mix: shares must sum to 1"),
            (lambda d: d["levels"]["mix"].update({"13": 0.0}), r"levels.mix: keys must be exactly"),
            (lambda d: d["levels"]["mix"].update({"l16": 0.0}), r"levels.mix: key 'l16' is not an integer"),
            (lambda d: d["levels"]["population"].__setitem__("11", "ladder"), r"levels.population.11: 'ladder'"),
            (lambda d: d["monitor"]["parity"]["dlogp_tol_bf16"].__setitem__("play", 0.0),
             r"dlogp_tol_bf16.play: .* outside"),
            (lambda d: d["reward"].__setitem__("win", 2.0), r"reward.win: must be 1.0"),
            (lambda d: d["reward"].__setitem__("shaping", "tower_hp"), r"reward.shaping"),
            (lambda d: d["gates"]["g2"].__setitem__("record_only", False), r"gates.g2.record_only"),
            (lambda d: d["gates"]["screen"].__setitem__("at_update_in_cycle", 21), r"gates.screen.at_update_in_cycle"),
            (lambda d: d["gates"]["crn_panel"].__setitem__("il_v4_t1", 0.5), r"gates.crn_panel: shares must sum"),
            (lambda d: d["gates"].__setitem__("il_temperature_sweep", [1.0, 1.0]), r"distinct temperatures"),
            (lambda d: d["monitor"].__setitem__("drop_rate_stop", 0.001), r"monitor.drop_rate_stop"),
            (lambda d: d["monitor"].__setitem__("plays_per_min_band", [1.2, 1.5]), r"plays_per_min_band\[0\]"),
            (lambda d: d["storage"].__setitem__("keep_updates", 0), r"storage.keep_updates"),
        ]
        for mutate, pattern in cases:
            with self.subTest(pattern=pattern):
                if "overrides" in pattern and not HAVE_VOCAB:   # delta_lag override names need the vocabulary
                    self.skipTest(VOCAB_SKIP)
                self._bad(mutate, pattern)

    def test_core_cards_must_be_in_deck(self):
        d = _ready()
        d["monitor"]["core_cards"] = ["example-card", "card-x"]
        with self.assertRaisesRegex(C.ConfigError, r"monitor.core_cards: \['card-x'\] not in user.deck"):
            C.from_dict(d)

    def test_core_card_candidate_in_deck_is_monitored(self):  # a core-card candidate in the deck is monitored
        deck_b = ["card-i", "card-j", "card-b-ev1", "card-k-hero", "core-b", "example-card", "card-l",
                  "card-m-ev1"]
        with mock.patch.object(C, "CORE_CARD_CANDIDATES", ("example-card", "core-b")):
            with self.assertRaisesRegex(C.ConfigError, r"monitor.core_cards: \['core-b'\] in user.deck must be"):
                C.from_dict(_ready(deck=deck_b))
            d = _ready(deck=deck_b)
            d["monitor"]["core_cards"] = ["example-card", "core-b"]
            self.assertEqual(C.from_dict(d).monitor.core_cards, ["example-card", "core-b"])

    def test_huge_int_and_negative_zero(self):
        with self.assertRaisesRegex(C.ConfigError, r"policy.lr: number too large"):
            C.from_dict({"policy": {"lr": 10 ** 400}})
        d = C.PPOConfig().to_dict()
        d["policy"]["weight_decay"] = -0.0
        self.assertEqual(C.from_dict(d).digest(), C.PPOConfig().digest())
        self.assertEqual(C.PPOConfig(policy=C.PolicyConfig(weight_decay=-0.0)).digest(), C.PPOConfig().digest())

    def test_frozen_and_revalidated(self):
        c = C.PPOConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.kl.beta = 0.5                                   # type: ignore[misc]
        c.opponents.main["mirror"] = 0.5                      # dicts stay mutable: digest / describe re-validate
        with self.assertRaisesRegex(C.ConfigError, r"opponents.main: shares must sum to 1"):
            c.digest()
        with self.assertRaisesRegex(C.ConfigError, r"opponents.main"):
            c.describe()

    def test_direct_construction_validates(self):
        with self.assertRaisesRegex(C.ConfigError, r"kl.beta"):
            C.PPOConfig(kl=C.KLConfig(beta=0.5))

    def test_bc_switch_on(self):
        d = C.PPOConfig().to_dict()
        d["user"]["bc_enabled"] = True
        d["bc"]["weight"] = 0.05
        c = C.from_dict(d)
        self.assertEqual(c.describe()["derived"]["bc_weight"], 0.05)


class TestReadiness(unittest.TestCase):
    def test_defaults_report_open_items(self):
        c = C.PPOConfig()
        miss = [m.split(":")[0] for m in c.missing("rollout")]
        # no il_v3 share and an fp32 backbone by default, so neither the extra-IL pins nor the bf16 tolerances are
        # open items
        self.assertEqual(miss, ["user.deck", "user.tower_troop", "rollout.hazard_dt", "opponents.checkpoint.il_v4",
                                "opponents.checkpoint_sha256.il_v4", "opponents.temperature.il_v4_tstar",
                                "levels.population.11"])
        pol = [m.split(":")[0] for m in c.missing("policy")]
        self.assertEqual(pol, miss + ["opponents.checkpoint.il_v2_heldout", "opponents.checkpoint_sha256.il_v2_heldout",
                                      "opponents.hazard_dt.il_v2_heldout", "opponents.code_tree.il_v2_heldout"])
        bf16 = C.from_dict(C.apply_overrides(c.to_dict(), ["policy.backbone_dtype=bf16"]))
        self.assertEqual([m.split(":")[0] for m in bf16.missing("policy")][-3:],
                         ["monitor.parity.dlogp_tol_bf16.wait", "monitor.parity.dlogp_tol_bf16.play",
                          "monitor.parity.dlogp_tol_bf16.ability"])
        old_il = C.from_dict(C.apply_overrides(c.to_dict(), ["opponents.main.il_v3_t1=0.05",
                                                             "opponents.main.il_v4_t1=0.1"]))
        self.assertEqual([m.split(":")[0] for m in old_il.missing("rollout")][5:9],
                         ["opponents.checkpoint.il_v3", "opponents.checkpoint_sha256.il_v3",
                          "opponents.hazard_dt.il_v3", "opponents.code_tree.il_v3"])   # only while il_v3 has a share
        with self.assertRaisesRegex(C.ConfigError, r"not ready for rollout:\n  user.deck"):
            c.require_ready("rollout")

    def test_filled_config_is_ready(self):
        c = C.from_dict(_ready())
        self.assertEqual(c.missing("rollout"), [])
        self.assertEqual(c.missing("policy"), [])
        c.require_ready("rollout")
        c.require_ready("policy")

    def test_policy_stage_needs_heldout(self):
        d = _ready()
        d["opponents"]["checkpoint"]["il_v2_heldout"] = None
        c = C.from_dict(d)
        self.assertEqual(c.missing("rollout"), [])
        self.assertEqual([m.split(":")[0] for m in c.missing("policy")], ["opponents.checkpoint.il_v2_heldout"])

    def test_bf16_tolerances_only_for_bf16(self):
        d = _ready()
        d["monitor"]["parity"]["dlogp_tol_bf16"] = {"wait": None, "play": None, "ability": None}
        d["policy"]["backbone_dtype"] = "bf16"                     # the default is fp32
        self.assertEqual(C.from_dict(d).missing("rollout"), [])
        self.assertEqual(len(C.from_dict(d).missing("policy")), 3)
        d["policy"]["backbone_dtype"] = "fp32"
        self.assertEqual(C.from_dict(d).missing("policy"), [])

    def test_snapshot_temperature_reported_when_used(self):
        d = _ready()
        d["opponents"]["temperature"]["snapshot"] = None
        self.assertEqual([m.split(":")[0] for m in C.from_dict(d).missing("rollout")],
                         ["opponents.temperature.snapshot"])
        d["user"]["opponent_design"] = "il_only"              # no snapshot games: not needed
        self.assertEqual(C.from_dict(d).missing("rollout"), [])

    def test_conditional_items(self):
        d = _ready()
        d["levels"]["population"]["11"] = None
        d["levels"]["mix"] = {"16": 1.0, "11": 0.0}              # level 16 only: no level-11 population needed
        d["opponents"]["checkpoint"]["il_v3"] = None
        d["opponents"]["hazard_dt"]["il_v3"] = None                 # settings of an unused model: not needed
        d["opponents"]["code_tree"]["il_v3"] = None
        self.assertEqual((d["opponents"]["pilot"]["il_v3_t1"], d["opponents"]["main"]["il_v3_t1"]), (0.0, 0.0))
        # no il_v3 share (the default): checkpoint not needed
        self.assertEqual(C.from_dict(d).missing("rollout"), [])
        d["opponents"]["temperature"]["il_v4_tstar"] = None
        d["opponents"]["pilot"].update(il_v4_tstar=0.0, il_v4_t1=0.5)
        d["opponents"]["main"].update(il_v4_tstar=0.0, il_v4_t1=0.25)
        c = C.from_dict(d)
        self.assertEqual(c.missing("rollout"), [])                # T*_IL unused in rollouts ...
        self.assertEqual([m.split(":")[0] for m in c.missing("policy")],
                         ["opponents.temperature.il_v4_tstar"])     # ... but still T_gate for G1

    def test_bad_stage(self):
        with self.assertRaises(C.ConfigError):
            C.PPOConfig().missing("eval")


class TestDigest(unittest.TestCase):
    def test_stable_and_order_independent(self):
        a = C.PPOConfig()
        self.assertEqual(a.digest(), C.PPOConfig().digest())
        self.assertRegex(a.digest(), r"^[0-9a-f]{64}$")
        d = a.to_dict()
        rev = {k: (dict(reversed(list(v.items()))) if isinstance(v, dict) else v) for k, v in reversed(list(d.items()))}
        self.assertEqual(C.from_dict(rev).digest(), a.digest())
        if not YAML_PATH.is_file():
            self.skipTest(YAML_SKIP)
        self.assertEqual(C.load_config(YAML_PATH).digest(), a.digest())

    @needs_yaml
    def test_fresh_interpreter(self):
        code = ("import sys; sys.path.insert(0, %r); from amini.ppo import config as C; "
                "print(C.load_config(%r).digest())" % (str(ROOT / "src"), str(YAML_PATH)))
        env = dict(os.environ, PYTHONHASHSEED="12345")
        out = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), C.PPOConfig().digest())

    def test_sensitive(self):
        base = C.PPOConfig().digest()
        seen = {base}
        for item in ("policy.lr=1.0e-5", "gae.lambda_tick=0.999", "rollout.hazard_dt=interval",
                     "user.kl_reference_after_promotion=il", "sim2real.delta_lag.overrides.Rocket=0",
                     "levels.mix.16=0.9", ):
            with self.subTest(item=item):
                if "delta_lag.overrides" in item and not HAVE_VOCAB:      # override names need the vocabulary
                    self.skipTest(VOCAB_SKIP)
                if item.startswith("levels.mix.16"):
                    items = [item, "levels.mix.11=0.1"]
                else:
                    items = [item]
                dg = C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), items)).digest()
                self.assertNotIn(dg, seen, item)
                seen.add(dg)


class TestOverridesAndDescribe(unittest.TestCase):
    @needs_yaml
    def test_overrides(self):
        c = C.load_config(YAML_PATH, ["policy.lr=1.0e-5", "user.deck=[a, b, c, d, e, f, g, example-card]",
                                      "user.tower_troop=159000001", "opponents.temperature.il_v4_tstar=0.25",
                                      "user.h1_reading=crn_only"])
        self.assertEqual(c.policy.lr, 1e-5)
        self.assertEqual(c.user.deck, ["a", "b", "c", "d", "e", "f", "g", "example-card"])
        self.assertEqual(c.user.tower_troop, 159000001)
        self.assertEqual(c.describe()["derived"]["tower_troop_key"], "cannoneer")
        self.assertEqual(c.opponents.temperature["il_v4_tstar"], 0.25)
        self.assertFalse(c.describe()["derived"]["g1c_enabled"])
        with self.assertRaisesRegex(C.ConfigError, r"policy: unknown key"):
            C.load_config(YAML_PATH, ["policy.learning_rate=1.0e-5"])
        with self.assertRaisesRegex(C.ConfigError, r"expected key.path=value"):
            C.load_config(YAML_PATH, ["policy.lr"])
        with self.assertRaisesRegex(C.ConfigError, r"policy.lr: null is not allowed"):
            C.load_config(YAML_PATH, ["policy.lr="])
        with self.assertRaisesRegex(C.ConfigError, r"bare 'NULL'"):
            C.load_config(YAML_PATH, ["levels.population.11=NULL"])
        c = C.load_config(YAML_PATH, ['levels.population.11="NULL"'])
        self.assertEqual(c.levels.population[11], "NULL")

    def test_describe(self):
        c = C.PPOConfig()
        rec = c.describe()
        json.dumps(rec, allow_nan=False)
        self.assertEqual(rec["schema"], C.CONFIG_SCHEMA)
        self.assertEqual(rec["digest"], c.digest())
        self.assertEqual(rec["config"], c.to_dict())
        self.assertEqual(rec["readiness"]["rollout"], c.missing("rollout"))
        der = rec["derived"]
        self.assertAlmostEqual(der["il_share"]["pilot"], 0.5)
        self.assertAlmostEqual(der["il_share"]["main"], 0.25)
        self.assertAlmostEqual(der["trained_seats_per_game"]["main"], 1.5)
        self.assertAlmostEqual(der["trained_seats_per_game"]["pilot"], 1.5)
        self.assertAlmostEqual(der["lambda_per_block"], 0.995 ** 10)
        self.assertAlmostEqual(der["lambda_scale_ticks"], 200.0)
        self.assertTrue(der["g1c_enabled"])
        self.assertEqual(rec["sampling"]["KAPPA"], {s: str(k) for s, k in SMP.KAPPA.items()})
        self.assertIsNone(rec["invariants"]["gate_bias_train"])
        self.assertEqual(rec["invariants"]["gate_bias_deploy"], 0.0)
        self.assertEqual(der["monitor_thresholds"], json.loads(json.dumps(c.monitor_thresholds())))   # fp32
        bf16 = C.from_dict(C.apply_overrides(c.to_dict(), ["policy.backbone_dtype=bf16"]))
        self.assertIsNone(bf16.describe()["derived"]["monitor_thresholds"])   # bf16 tolerances not measured
        self.assertEqual(der["king_level_by_level"], {"16": 16, "11": 11})
        self.assertEqual(der["client_env_digest_without_tables"], c.env_digest())
        ready = C.from_dict(C.apply_overrides(_ready(), ["policy.backbone_dtype=bf16"])).describe()
        self.assertEqual(ready["derived"]["monitor_thresholds"]["identity_logp_tol"],
                         {"wait": 5e-4, "play": 2e-4, "ability": 2e-4})
        self.assertEqual(ready["derived"]["tower_troop_key"], "tower-princess")

    def test_il_only_mix(self):
        d = C.PPOConfig().to_dict()
        d["user"]["opponent_design"] = "il_only"
        c = C.from_dict(d)
        for ph in C.PHASES:
            mix = c.effective_mix(ph)
            self.assertAlmostEqual(sum(mix.values()), 1.0)
            for k in C.SELFPLAY_KINDS:
                self.assertEqual(mix[k], 0.0)
            self.assertAlmostEqual(c.trained_seats_per_game(ph), 1.0)
        self.assertAlmostEqual(c.effective_mix("pilot")["il_v4_t1"], 0.3 / 0.5)

    def test_phase_for_update(self):
        c = C.PPOConfig()
        self.assertEqual([c.phase_for_update(u) for u in (1, 10, 11, 200)], ["pilot", "pilot", "main", "main"])
        with self.assertRaises(C.ConfigError):
            c.phase_for_update(0)

    def test_to_dict_is_a_copy(self):
        c = C.PPOConfig()
        d = c.to_dict()
        d["opponents"]["pilot"]["mirror"] = 0.9
        self.assertEqual(c.opponents.pilot["mirror"], 0.1)
        self.assertEqual(copy.deepcopy(c), c)


@needs_yaml
class TestYamlLint(unittest.TestCase):
    def _load_text(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run.yaml"
            p.write_text(text, encoding="utf-8")
            return C.load_config(p)

    def test_repeated_section_and_key_rejected(self):
        text = YAML_PATH.read_text(encoding="utf-8")
        with self.assertRaisesRegex(C.ConfigError, r"key 'policy' appears twice"):
            self._load_text(text + "\npolicy:\n  epochs: 3\n")
        with self.assertRaisesRegex(C.ConfigError, r"key 'policy.epochs' appears twice"):
            self._load_text(text.replace("  epochs: 3\n", "  epochs: 3\n  epochs: 4\n", 1))
        self.assertEqual(self._load_text(text), C.PPOConfig())

    def test_ambiguous_scalars_rejected(self):
        text = YAML_PATH.read_text(encoding="utf-8")
        for bad in ("NULL", "None", "yes", "off", "[NULL, ranked]"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(C.ConfigError, r"reads differently under PyYAML"):
                    C.lint_yaml_text(text.replace("    16: ranked", f"    16: {bad}", 1))
        c = self._load_text(text.replace("    11: null ", '    11: "NULL" ', 1))
        self.assertEqual(c.levels.population[11], "NULL")


class TestSharedSchemas(unittest.TestCase):
    """The names passed between modules unchanged."""

    @needs_cases
    def test_opponent_and_level_tables_match_cases(self):
        from ppo_v7 import cases as K
        c = C.PPOConfig()
        for ph in C.PHASES:
            self.assertEqual(c.effective_mix(ph), K.OPPONENT_MIX[ph])
            K.check_mix(c.effective_mix(ph), ph)
        self.assertEqual(set(C.OPPONENT_KINDS), set(K.OPPONENT_MIX["main"]))
        self.assertEqual(set(C.IL_KINDS), set(K.IL_KINDS))
        self.assertEqual((c.opponents.pilot_updates, c.opponents.il_floor), (K.PILOT_UPDATES, K.IL_FLOOR))
        self.assertEqual(c.levels.mix, K.DEFAULT_LEVEL_MIX)
        self.assertEqual(set(C.LEVELS), set(K.LEVELS))
        self.assertEqual(C.TOWER_TROOP_GIDS, {g: key for g, (key, _) in K.TOWER_TROOPS.items()})
        il_only = C.from_dict(dict(C.PPOConfig().to_dict(), user={"opponent_design": "il_only"}))
        for ph in C.PHASES:
            K.check_mix(il_only.effective_mix(ph), ph)

    def test_king_level_is_the_game_level(self):
        c = C.PPOConfig()
        self.assertEqual([c.king_level(lv) for lv in (16, 11)], [16, 11])
        for bad in (13, True, "16"):
            with self.assertRaises(C.ConfigError):
                c.king_level(bad)
        only16 = C.from_dict(C.apply_overrides(c.to_dict(), ["levels.mix.16=1.0", "levels.mix.11=0.0"]))
        with self.assertRaisesRegex(C.ConfigError, r"level 11 has no share"):
            only16.king_level(11)

    def test_monitor_thresholds_match_monitor(self):
        from amini.ppo import monitor as M
        fp32 = C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), ["policy.backbone_dtype=fp32"]))
        m = fp32.monitor_thresholds()
        self.assertEqual(set(m), set(M.Thresholds.__dataclass_fields__))     # every monitor number has a source
        self.assertEqual(M.Thresholds.from_mapping(m), M.Thresholds())        # same values and units
        self.assertAlmostEqual(m["rollback_delta"], -0.05)                    # pp -> fraction
        c = C.from_dict(C.apply_overrides(_ready(), ["gates.rollback_delta_pp=-2.0", "policy.backbone_dtype=bf16"]))
        th = M.Thresholds.from_mapping(c.monitor_thresholds())
        self.assertAlmostEqual(th.rollback_delta, -0.02)
        self.assertEqual(dict(th.identity_logp_tol), {"wait": 5e-4, "play": 2e-4, "ability": 2e-4})
        self.assertEqual(C.PPOConfig().monitor_thresholds(), m)               # fp32 is the default
        with self.assertRaisesRegex(C.ConfigError, r"dlogp_tol_bf16"):        # bf16, not measured
            C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), ["policy.backbone_dtype=bf16"])).monitor_thresholds()
        self.assertEqual(C.ROW_TYPES, M.ROW_TYPES)

    def test_monitor_thresholds_carry_former_code_constants(self):
        from amini.ppo import monitor as M
        fp32 = C.from_dict(C.apply_overrides(C.PPOConfig().to_dict(), ["policy.backbone_dtype=fp32"]))
        m = fp32.monitor_thresholds()
        self.assertEqual((m["temp_lr_floor"], m["stall_games_per_update"], m["mask_audit_every"], m["drift_every"]),
                         (M.TEMP_LR_FLOOR, M.STALL_GAMES_PER_UPDATE, 10, 10))
        c = C.from_dict(C.apply_overrides(fp32.to_dict(), ["monitor.temp_lr_floor=0.5",
                                                           "gates.stall_games_per_update=4096"]))
        th = M.Thresholds.from_mapping(c.monitor_thresholds())
        self.assertEqual((th.temp_lr_floor, th.stall_games_per_update), (0.5, 4096))
        self.assertEqual(M.Monitor(thresholds=th).temp_lr_floor, 0.5)
        with self.assertRaisesRegex(C.ConfigError, r"gates.stall_games_per_update"):
            C.from_dict(C.apply_overrides(fp32.to_dict(), ["gates.stall_games_per_update=512"]))

    @needs_cases
    def test_opponent_settings_table(self):
        from ppo_v7 import cases as K
        c = C.from_dict(_ready())
        tab = c.opponent_settings()
        self.assertEqual(set(tab), set(C.FROZEN_TEMPERATURE_KINDS))
        for kind, row in tab.items():
            spec = K.OpponentSpec(**row)                                       # exactly OpponentSpec's fields
            self.assertEqual(spec.name, kind)
        self.assertEqual(tab["il_v4_tstar"]["checkpoint"], "/x/v4.pt")
        self.assertEqual((tab["il_v4_t1"]["hazard_dt"], tab["snapshot"]["hazard_dt"]), ("block", "block"))
        self.assertEqual((tab["il_v3_t1"]["hazard_dt"], tab["il_v3_t1"]["code_tree"]), ("interval", "v3-frozen"))
        self.assertEqual(tab["il_v4_t1"]["code_tree"], C.RUN_CODE_TREE)
        self.assertEqual(tab["il_v2_heldout"]["gate_bias"], [[0.0] * 3, [0.0] * 3])   # explicit zeros = no bias
        d = _ready()
        d["opponents"]["gate_bias"]["il_v2_heldout"] = {"base": [0.0, -0.1, 0.0], "threat": [0.0, -0.2, 0.0]}
        c2 = C.from_dict(d)
        self.assertEqual(c2.opponent_settings()["il_v2_heldout"]["gate_bias"], [[0.0, -0.1, 0.0], [0.0, -0.2, 0.0]])
        K.OpponentSpec(**c2.opponent_settings()["il_v2_heldout"])
        self.assertNotEqual(c2.digest(), c.digest())                           # the settings table is in the digest
        self.assertEqual(c2.describe()["derived"]["opponent_settings"], c2.opponent_settings())
        for bad, pat in (({"base": [0.0, 0.0], "threat": [0.0, 0.0, 0.0]}, r"needs 3 values"),
                         ({"base": [0.0, 0.0, 0.0]}, r"keys must be exactly"),
                         ({"base": [0.0, 0.0, 0.0], "threat": [0.0, 99.0, 0.0]}, r"outside")):
            d["opponents"]["gate_bias"]["il_v2_heldout"] = bad
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(C.ConfigError, pat):
                    C.from_dict(d)
        d = _ready()
        d["opponents"]["hazard_dt"]["il_v3"] = "auto"
        with self.assertRaisesRegex(C.ConfigError, r"opponents.hazard_dt.il_v3: 'auto' not one of"):
            C.from_dict(d)

    def test_chain_args_match_chain_sh_defaults(self):
        c = C.PPOConfig()
        self.assertEqual(c.chain_args(), ["--timeout", "14400", "--kill-after", "600"])
        for bad in ("chain.block_timeout_s=60", "chain.kill_after_s=10"):
            with self.assertRaises(C.ConfigError):
                C.from_dict(C.apply_overrides(c.to_dict(), [bad]))
        chain = ROOT / "tools" / "ppo_v7" / "chain.sh"
        if not chain.is_file():
            self.skipTest("needs the block chain script tools/ppo_v7/chain.sh (not published)")
        text = chain.read_text(encoding="utf-8")
        self.assertRegex(text, r"\bTMO=%d\b" % c.chain.block_timeout_s)
        self.assertRegex(text, r"\bKILL_AFTER=%d\b" % c.chain.kill_after_s)

    def test_short_head_decision_ticks_matches_gae(self):
        from amini.nn import schema_v1 as S
        from amini.ppo import gae as G
        c = C.PPOConfig()
        self.assertEqual(c.reward.short_head_decision_ticks, G.SHORT_DECISION_TICKS)
        self.assertEqual((c.reward.short_head_coef, c.reward.short_head_gamma), (G.SHORT_C, G.SHORT_GAMMA_DECISION))
        self.assertEqual(G.DECISION_TICKS, S.U_TICKS)                          # one query per decision block
        with self.assertRaises(C.ConfigError):
            C.from_dict(C.apply_overrides(c.to_dict(), ["reward.short_head_decision_ticks=0"]))

    def test_sampling_mode(self):
        self.assertEqual(C.sampling_mode(C.GREEDY), (True, None))
        self.assertEqual(C.sampling_mode(0.25), (False, 0.25))
        self.assertEqual(C.sampling_mode(1), (False, 1.0))
        for bad in (-0.1, 1.5, float("nan"), True, "1"):
            with self.subTest(bad=bad):
                with self.assertRaises(C.ConfigError):
                    C.sampling_mode(bad)


if __name__ == "__main__":
    unittest.main()
