"""Tests of tools/ppo_v7/gate.py: CRN triples and seat-swap pairs, the gate seed namespace, resumable chunk
files, pair completeness and the pass / fail rules, all with a fake game runner whose true effect is known. No
game environment, no worker, no model, no GPU. They need ``ppo_v7.cases``, which is not part of this
repository: without it the module is skipped.

    PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_gate.py
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import shutil
import signal
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT / 'tests' / 'v7', ROOT / 'tools'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from ppo_v7 import cases as C                                     # noqa: E402
except ImportError:                                                   # pragma: no cover
    raise unittest.SkipTest('needs ppo_v7.cases, which is not part of this repository')
from ppo_v7 import gate as G                                          # noqa: E402
from amini.ppo import stats as S                               # noqa: E402

BASE = 314159                      # base seed of the test plans (determinism only)
N_BOOT = 1000
ENV = 'e' * 64                     # the plans' environment pin (gate.env_digest of the test environment)


# ------------------------------------------------------------------ fixtures
def panel(n: int = 64, split: str = 'val') -> List[C.CaseRef]:
    return [C.CaseRef(case_id=f'VAL{i:04d}', level=16 if i % 8 else 11, our_owner=i % 2, split=split)
            for i in range(n)]


def player(name: str, digest: Optional[str] = None, t: float = 1.0) -> G.Player:
    return G.Player(name=name, digest=digest or (name * 8 + '0' * 64)[:64],
                    spec=C.OpponentSpec(name=name, checkpoint=f'/ckpt/{name}.pt', temperature=t, hazard_dt='block',
                                        code_tree='run'))


def opponents() -> Dict[str, G.Player]:
    return {k: player(k) for k in G.PANEL_KINDS}


def u01(*parts) -> float:
    return C.stable_hash('fake', *[str(p) for p in parts]) / float(C.SEED_MASK + 1)


class FakeRunner:
    """Deterministic fake game runner with known win probabilities.

    CRN: the scored model P beats panel opponent O with p = 0.5 + strength[P] - strength[O] (+ per-(P, kind)
    offset). Seat-swap: A beats B on owner o with p = 0.5 + strength[A] - strength[B] + adv * (1 if o == 0 else -1).
    A fraction ``indep`` of games uses a uniform from the game key, the unit's environment seed and the model
    (independent across arms, and across gates that differ only in base_seed); the rest use the unit's shared
    environment seed (common random numbers), so the two arms are correlated. Score:
    win if u < p - draw/2, draw if u < p + draw/2 (expected score = p). Diagnostics are deterministic.
    ``fail`` = keys returned as crashes; ``boom_after`` raises after that many games (a killed coordinator)."""

    def __init__(self, strength: Dict[str, float], *, offset: Optional[Dict[tuple, float]] = None, adv: float = 0.0,
                 indep: float = 0.6, draw: float = 0.06, fail: Sequence[str] = (), boom_after: Optional[int] = None):
        self.strength, self.offset, self.adv = strength, offset or {}, adv
        self.indep, self.draw, self.fail = indep, draw, set(fail)
        self.boom_after = boom_after
        self.calls: List[str] = []

    def p_win(self, spec: G.GameSpec) -> float:
        me = spec.players[spec.scored_owner]
        other = spec.players[1 - spec.scored_owner]
        p = 0.5 + self.strength.get(me.name, 0.0) - self.strength.get(other.name, 0.0)
        p += self.offset.get((me.name, spec.opp_kind), 0.0)
        if spec.kind in G.SWAP_KINDS:
            p += self.adv * (1 if spec.scored_owner == 0 else -1)
        return min(max(p, 0.0), 1.0)

    def game(self, spec: G.GameSpec) -> G.GameOutcome:
        if self.boom_after is not None and len(self.calls) >= self.boom_after:
            raise KeyboardInterrupt('coordinator killed')
        self.calls.append(spec.key)
        if spec.key in self.fail:
            return G.GameOutcome(key=spec.key, end='crash', error='fake crash')
        me = spec.players[spec.scored_owner]
        if u01(spec.key, spec.rnd_seed, 'mix') < self.indep:
            u = u01(spec.key, spec.rnd_seed, me.digest, 'indep')
        else:
            u = u01(spec.rnd_seed, spec.seeds[f'o{spec.scored_owner}_torch'], 'common')
        p = self.p_win(spec)
        s = 1.0 if u < p - self.draw / 2 else 0.5 if u < p + self.draw / 2 else 0.0
        score0 = s if spec.scored_owner == 0 else 1.0 - s
        length = 3000 + int(u01(spec.key, 'len') * 2400)
        seats = []
        for o in (0, 1):
            pl = spec.players[o]
            plays = 10 + int(u01(spec.key, o, 'plays') * 20)
            ab = int(u01(spec.key, o, 'ab') * 3)
            ref = {'19': 1} if u01(spec.key, o, 'ref') < 0.1 else {}
            seats.append(G.SeatStats(queries=300, non_wait=plays + ab + sum(ref.values()), plays=plays, abilities=ab,
                                     refusals=ref, cards={'hog-rider': plays // 3, f'card-{pl.name}': plays - plays // 3}))
        return G.GameOutcome(key=spec.key, end='native', score0=score0, length=length, seats=tuple(seats))

    def run(self, specs):
        return [self.game(s) for s in specs]


def settings(**kw) -> G.GateSettings:
    """Test settings: small gates are scored without the configured-schedule check (tested on its own)."""
    kw.setdefault('check_schedule', False)
    return G.GateSettings(n_boot=N_BOOT, **kw)


def g1_plan(update: int = 20, n: int = 1024, base: int = BASE, cand_t: float = 1.0, attempt: int = 0,
            pan: Optional[List[C.CaseRef]] = None) -> G.GatePlan:
    return G.crn_plan('g1', G.block_name('g1', update=update, attempt=attempt), n, panel=pan or panel(),
                      candidate=player('cand', t=cand_t), champion=player('champ'), opponents=opponents(),
                      base_seed=base, env_digest=ENV)


def swap_plan(kind: str = 'g1c', n: int = 1024, base: int = BASE, update: int = 20) -> G.GatePlan:
    blk = G.block_name(kind, update=update) if kind in ('g1', 'g1c') else G.block_name(kind)
    return G.seat_swap_plan(kind, blk, n, panel=panel(), a=player('cand'), b=player('champ'), base_seed=base, env_digest=ENV)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='ppo-gate-'))
        self.store = G.GateStore(self.tmp / 'gates')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_eval(self, plan, runner, store=None, **kw):
        store = store or self.store
        st = G.run_gate(plan, runner, store, chunk_size=kw.pop('chunk_size', 64), **kw)
        return st, G.evaluate(plan, store, settings())


# ------------------------------------------------------------------ seeds
class TestSeeds(unittest.TestCase):
    def test_stable_value_and_range(self):
        a = G.gate_seed(12345, 'g1-u000020-a0', 7, 'game')
        self.assertEqual(a, G.gate_seed(12345, 'g1-u000020-a0', 7, 'game'))
        self.assertTrue(0 <= a < 2 ** 63)
        r = G.gate_rnd_seed(12345, 'g1-u000020-a0', 7)
        self.assertTrue(1 <= r <= C.RND_SEED_MAX)
        self.assertNotEqual(a, G.gate_seed(12345, 'g1-u000020-a0', 8, 'game'))
        self.assertNotEqual(a, G.gate_seed(12345, 'g1-u000040-a0', 7, 'game'))
        self.assertNotEqual(a, G.gate_seed(12346, 'g1-u000020-a0', 7, 'game'))

    def test_namespace_disjoint_from_training(self):
        # Every training seed of 40 updates x 1,024 replicas (all roles, same base) vs every seed of the gates
        # of the same updates: no 63-bit value in common.
        train = set()
        for u in range(1, 41):
            for i in range(1024):
                train.update(v for k, v in C.game_seeds(BASE, u, i).items() if k != 'rnd_seed')
        gates = set()
        for u in (20, 40):
            for p in (g1_plan(update=u), swap_plan('g1c', update=u)):
                gates.update(p.all_seeds())
        gates.update(G.crn_plan('screen', G.block_name('screen', champion_digest='c' * 64), 512, panel=panel(),
                                candidate=player('cand'), champion=player('champ'), opponents=opponents(),
                                base_seed=BASE, env_digest=ENV).all_seeds())
        self.assertGreater(len(train), 600_000)
        self.assertGreater(len(gates), 40_000)
        self.assertEqual(train & gates, set())
        # identical (update, replica, role) arguments still differ: the namespace string separates them
        self.assertNotEqual(G.gate_seed(BASE, 'u', 3, 'game'), C.replica_seed(BASE, 3, 3, 'game'))

    def test_fresh_block_per_g1(self):
        a, b = g1_plan(update=20), g1_plan(update=40)
        r = g1_plan(update=20, attempt=1)
        self.assertEqual(set(a.all_seeds()) & set(b.all_seeds()), set())
        self.assertEqual(set(a.all_seeds()) & set(r.all_seeds()), set())
        self.assertEqual({(u.case_id, u.rnd_seed) for u in a.units} & {(u.case_id, u.rnd_seed) for u in b.units}, set())
        self.assertEqual(a.block, 'g1-u000020-a0')
        self.assertEqual(r.block, 'g1-u000020-a1')

    def test_block_names(self):
        self.assertEqual(G.block_name('g1c', update=7, attempt=2), 'g1c-u000007-a2')
        self.assertEqual(G.block_name('screen', champion_digest='ab' * 32), 'screen-' + 'ab' * 8 + '-a0')
        self.assertEqual(G.block_name('screen', champion_digest='ab' * 32, attempt=1), 'screen-' + 'ab' * 8 + '-a1')
        self.assertEqual(G.block_name('g2'), 'g2-fixed')
        self.assertEqual(G.block_name('g2', attempt=2), 'g2-fixed-a2')
        self.assertEqual(G.block_name('g1', update=20, confirm=True), 'g1-u000020-a0-confirm')
        with self.assertRaises(G.GateError):
            G.block_name('g1c', update=20, confirm=True)
        self.assertEqual(G.block_name('release', attempt=1), 'release-a1')
        for bad in (dict(kind='g1'), dict(kind='screen'), dict(kind='swap'), dict(kind='nope')):
            with self.assertRaises(G.GateError):
                G.block_name(**bad)
        with self.assertRaises(G.GateError):
            G.gate_seed(1, 'a|b', 0, 'game')

    def test_check_disjoint_with_training_plans(self):
        cases = {16: [C.CaseRef(f'TR{i:03d}', 16, i % 2, split='train') for i in range(40)],
                 11: [C.CaseRef(f'TS{i:03d}', 11, i % 2, split='train') for i in range(10)]}
        opps = {k: C.OpponentSpec(name=k, checkpoint='/x', temperature=1.0) for k in C.IL_KINDS}
        deck = list(range(8))
        train = []
        for u in (19, 20):
            train += C.plan_update(u, 256, base_seed=BASE, our_deck=deck, cases_by_level=cases, opponents=opps,
                                   opp_decks_by_level={16: [(tuple(range(10, 18)), 1.0)], 11: [(tuple(range(10, 18)), 1.0)]},
                                   mirror_fallback='mark')
        plan = g1_plan()
        G.check_disjoint(plan, train, base_seed=BASE)          # no exception
        G.check_panel(panel(), [c.case_id for t in cases.values() for c in t])
        # a training game that replays a gate (case, rndSeed) or a gate seed is refused
        u0 = plan.units[0]
        fake = C.GamePlan(update=1, replica=0, phase='pilot', case_id=u0.case_id, level=u0.level, our_owner=0,
                          seed=1, rnd_seed=u0.rnd_seed, opp_kind='il_v4_t1', opp_model='x', opp_deck=None,
                          opp_deck_source='case', self_weight=1.0, opp_seat_weight=0.0)
        with self.assertRaises(G.GateError):
            G.check_disjoint(plan, [fake])
        fake2 = C.GamePlan(**{**fake.__dict__, 'case_id': 'OTHER', 'seed': u0.seed_map()['game']})
        with self.assertRaises(G.GateError):
            G.check_disjoint(plan, [fake2])
        with self.assertRaises(G.GateError):
            G.check_panel(panel(), ['VAL0003'])


# ------------------------------------------------------------------ plans
class TestPlans(unittest.TestCase):
    def test_crn_panel_counts_and_holdout(self):
        plan = g1_plan()
        kinds = Counter(u.opp_kind for u in plan.units)
        self.assertEqual(G.DEFAULT_PANEL_MIX, {'il_v2_heldout': 0.5, 'il_v4_tstar': 0.3, 'il_v4_t1': 0.2})
        self.assertEqual(kinds, {'il_v2_heldout': 512, 'il_v4_tstar': 307, 'il_v4_t1': 205})   # largest remainder
        self.assertTrue(all(u.holdout == (u.opp_kind == 'il_v2_heldout') for u in plan.units))
        per_case = Counter(u.case_id for u in plan.units)
        self.assertEqual(len(per_case), 64)
        self.assertEqual(set(per_case.values()), {16})        # balanced openings
        self.assertEqual(plan.n_expected, 1024)
        # config mix: exact counts follow it
        p2 = G.crn_plan('g1', 'g1-u000020-a0', 100, panel=panel(), candidate=player('cand'), champion=player('champ'),
                        opponents=opponents(), base_seed=BASE, env_digest=ENV,
                        panel_mix={'il_v2_heldout': 0.6, 'il_v4_tstar': 0.2, 'il_v4_t1': 0.2})
        self.assertEqual(Counter(u.opp_kind for u in p2.units), {'il_v2_heldout': 60, 'il_v4_tstar': 20, 'il_v4_t1': 20})

    def test_crn_arms_share_everything_but_the_model(self):
        plan = g1_plan(n=64)
        for i in range(64):
            c, h = plan.spec('candidate', i), plan.spec('champion', i)
            self.assertEqual(c.seeds, h.seeds)
            self.assertEqual((c.rnd_seed, c.case_id, c.opp_kind), (h.rnd_seed, h.case_id, h.opp_kind))
            self.assertEqual(c.scored_owner, c.our_owner)
            self.assertEqual(c.players[c.our_owner].name, 'cand')
            self.assertEqual(h.players[h.our_owner].name, 'champ')
            self.assertEqual(c.players[1 - c.our_owner], h.players[1 - h.our_owner])
            self.assertEqual(c.players[1 - c.our_owner].name, c.opp_kind)
            self.assertEqual(set(c.seeds), {f'o{o}_{r}' for o in (0, 1) for r in G.SIDE_ROLES} | set(G.SHARED_ROLES))

    def test_aa_independent_policy_seeds(self):
        champ = player('champ')
        plan = G.crn_plan('aa', G.block_name('aa'), 64, panel=panel(), candidate=champ, champion=champ,
                          opponents=opponents(), base_seed=BASE, env_digest=ENV)
        self.assertFalse(plan.policy_crn)
        for i in range(64):
            c, h = plan.spec('candidate', i), plan.spec('champion', i)
            our, opp = c.our_owner, 1 - c.our_owner
            self.assertEqual(c.rnd_seed, h.rnd_seed)
            for r in G.SIDE_ROLES:
                self.assertNotEqual(c.seeds[f'o{our}_{r}'], h.seeds[f'o{our}_{r}'])
                self.assertEqual(c.seeds[f'o{opp}_{r}'], h.seeds[f'o{opp}_{r}'])
        with self.assertRaises(G.GateError):          # same weights in a G1 is an A/A test
            G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=champ, champion=champ,
                       opponents=opponents(), base_seed=BASE, env_digest=ENV)

    def test_seat_swap_pairs(self):
        plan = swap_plan(n=128)
        for i in range(128):
            a0, a1 = plan.spec('a0', i), plan.spec('a1', i)
            self.assertEqual((a0.scored_owner, a1.scored_owner), (0, 1))
            self.assertEqual(a0.players[0].name, 'cand')
            self.assertEqual(a1.players[1].name, 'cand')
            self.assertEqual(a0.players[1].name, 'champ')
            self.assertEqual(a0.seeds, a1.seeds)                 # seats keep their seeds, models swap
            self.assertEqual(a0.rnd_seed, a1.rnd_seed)
            self.assertNotEqual(a0.seeds['o0_torch'], a0.seeds['o1_torch'])

    def test_plan_guards(self):
        with self.assertRaises(G.GateError):                     # training openings are refused
            G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(split='train'), candidate=player('cand'),
                       champion=player('champ'), opponents=opponents(), base_seed=BASE, env_digest=ENV)
        with self.assertRaises(G.GateError):                     # T_gate must match in G1
            g1_plan(cand_t=0.5)
        with self.assertRaises(G.GateError):                     # panel kind without a player
            opp = opponents()
            del opp['il_v2_heldout']
            G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand'), champion=player('champ'),
                       opponents=opp, base_seed=BASE, env_digest=ENV)
        with self.assertRaises(G.GateError):                     # T* not measured yet
            G.Player('x', 'd' * 64, C.OpponentSpec(name='x', checkpoint='/x', temperature=None))
        with self.assertRaises(G.GateError):
            G.Player('x', 'd' * 64, C.OpponentSpec(name='x', checkpoint=None, temperature=1.0))
        with self.assertRaises(G.GateError):
            G.seat_swap_plan('g1', 'x', 4, panel=panel(), a=player('a'), b=player('b'), base_seed=1, env_digest=ENV)

    def test_plan_round_trip(self):
        for plan in (g1_plan(n=100), swap_plan(n=50)):
            back = G.GatePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
            self.assertEqual(back.block_digest(), plan.block_digest())
            for arm in plan.arms:
                self.assertEqual(back.arm_digest(arm), plan.arm_digest(arm))
                self.assertEqual(back.spec(arm, 7), plan.spec(arm, 7))

    def test_settings_from_config(self):
        from amini.ppo import config as CFG
        cfg = CFG.PPOConfig()
        st = G.GateSettings.from_config(cfg)
        g = cfg.gates
        self.assertEqual((st.g1_pairs, st.screen_pairs, st.g1c_pairs, st.g2_pairs, st.release_pairs, st.aa_pairs),
                         (g.g1.games_per_side, g.screen.games, g.g1c.pairs, g.g2.pairs, g.release.pairs,
                          g.aa_null_pairs))
        self.assertAlmostEqual(st.holdout_min, g.g1.heldout_min_delta_pp / 100.0)
        self.assertEqual(dict(st.panel_mix), dict(g.crn_panel))
        self.assertEqual(st.n_boot, g.bootstrap_resamples)
        self.assertEqual(st.min_completion, g.min_completion)
        # panel players from the settings table need checkpoints and T*: the default config has neither
        with self.assertRaises(G.GateError):
            G.panel_players(cfg, {k: 'd' * 64 for k in G.PANEL_KINDS})


# ------------------------------------------------------------------ pass / fail with known truth
class TestCrnRules(TmpCase):
    def test_true_gain_passes(self):
        # true delta = +10 pp (strength 0.10 vs 0.00 against every panel opponent)
        for base in (BASE, BASE + 1, BASE + 2):
            store = G.GateStore(self.tmp / f'g{base}')
            st, rep = self.run_eval(g1_plan(base=base), FakeRunner({'cand': 0.10}), store=store)
            self.assertEqual(st.state, 'complete')
            self.assertTrue(rep.valid and rep.passed and rep.blocking, rep.reasons)
            self.assertEqual(rep.n_complete, 1024)
            se = rep.stats['se_cluster']
            self.assertLess(abs(rep.delta_hat - 0.10), 4 * se)
            self.assertGreater(rep.rho, 0.05)                    # common random numbers correlate the arms
            #   (each arm uses the common uniform in 40 % of games, so both do in ~16 %: rho ~ 0.1-0.15)
            self.assertFalse(rep.replay_required)

    def test_no_gain_rarely_passes(self):
        # true delta = 0: the one-sided alpha rule; 20 independent gates of 256 triples, expect ~alpha x 20 passes
        # (the fake's uniforms include the environment seed, so the gates are independent; the adv calibration test
        # is stronger)
        passes = 0
        for k in range(20):
            plan = g1_plan(n=256, base=BASE + 100 + k)
            store = G.GateStore(self.tmp / f'n{k}')
            _, rep = self.run_eval(plan, FakeRunner({}), store=store)
            self.assertTrue(rep.valid)
            passes += rep.passed
        self.assertLessEqual(passes, 6)

    def test_loss_fails_and_alarms(self):
        _, rep = self.run_eval(g1_plan(), FakeRunner({'cand': -0.12}))
        self.assertFalse(rep.passed)
        self.assertTrue(rep.alarm)                               # delta_hat <= rollback_delta: rollback input
        self.assertTrue(any('lower bound' in r for r in rep.reasons))
        args = rep.monitor_args(update=20)
        self.assertEqual(args['kind'], 'g1')
        self.assertTrue(args['valid'])
        self.assertLess(args['delta_hat'], -0.03)

    def test_holdout_subpanel_blocks(self):
        # +20 pp vs the other panel opponents, -6 pp vs held-out IL: overall +7 pp (LB > 0) but the held-out
        # sub-panel is below holdout_min
        off = {('cand', 'il_v4_tstar'): 0.20, ('cand', 'il_v4_t1'): 0.20, ('cand', 'il_v2_heldout'): -0.06}
        _, rep = self.run_eval(g1_plan(), FakeRunner({}, offset=off))
        self.assertGreater(rep.stats['lb'], 0.0)
        self.assertFalse(rep.passed)
        self.assertTrue(any('held-out' in r for r in rep.reasons), rep.reasons)
        self.assertLess(rep.subpanel['il_v2_heldout']['delta_hat'], -0.01)
        self.assertTrue(rep.subpanel['il_v2_heldout']['holdout'])
        self.assertEqual(rep.subpanel['il_v2_heldout']['pairs'] + rep.subpanel['il_v4_t1']['pairs']
                         + rep.subpanel['il_v4_tstar']['pairs'], 1024)

    def test_aa_null_records_rho(self):
        champ = player('champ')
        plan = G.crn_plan('aa', G.block_name('aa'), 512, panel=panel(), candidate=champ, champion=champ,
                          opponents=opponents(), base_seed=BASE, env_digest=ENV)
        _, rep = self.run_eval(plan, FakeRunner({}))
        self.assertTrue(rep.valid)
        self.assertFalse(rep.passed or rep.blocking or rep.alarm)
        self.assertIsNotNone(rep.rho)
        self.assertLess(abs(rep.delta_hat), 4 * rep.stats['se_cluster'])
        with self.assertRaises(G.GateError):
            rep.monitor_args(update=1)

    def test_matches_stats_directly(self):
        plan = g1_plan(n=256)
        st, rep = self.run_eval(plan, FakeRunner({'cand': 0.05}))
        outs = {a: self.store.outcomes(plan, a) for a in plan.arms}
        recs = []
        for u in plan.units:
            c = outs['candidate'][G.game_key(plan.block, 'candidate', u.unit)].score(u.our_owner)
            h = outs['champion'][G.game_key(plan.block, 'champion', u.unit)].score(u.our_owner)
            recs.append((u.unit, u.case_id, c, h, u.holdout))
        ref = S.crn_gate(recs, n_expected=256, n_boot=N_BOOT, seed=G.boot_seed(plan.block))
        self.assertEqual(rep.stats['delta_hat'], ref.delta_hat)
        self.assertEqual(rep.stats['lb'], ref.lb)
        self.assertEqual(rep.passed, ref.passed)

    def test_screen_reuses_champion_games(self):
        champ = player('champ')
        blk = G.block_name('screen', champion_digest=champ.digest)
        mk = lambda cand: G.crn_plan('screen', blk, 512, panel=panel(), candidate=cand, champion=champ,
                                     opponents=opponents(), base_seed=BASE, env_digest=ENV)
        r1 = FakeRunner({'cand1': 0.0})
        st1, rep1 = self.run_eval(mk(player('cand1')), r1)
        self.assertEqual(st1.games_run, 1024)
        self.assertFalse(rep1.passed)                            # a screen never passes
        self.assertFalse(rep1.blocking)
        r2 = FakeRunner({'cand2': -0.1})
        st2, rep2 = self.run_eval(mk(player('cand2')), r2)
        self.assertEqual(st2.games_run, 512)                     # only the new candidate's games
        self.assertTrue(all('/candidate/' in k for k in r2.calls))
        self.assertTrue(rep2.alarm)
        self.assertEqual(rep2.monitor_args(update=30)['kind'], 'screen')


class TestSeatSwapRules(TmpCase):
    def test_g1c_pass_and_fail(self):
        _, good = self.run_eval(swap_plan(), FakeRunner({'cand': 0.08}, adv=0.05))
        self.assertTrue(good.valid and good.passed, good.reasons)
        self.assertTrue(good.stats['clustered'])                 # opening = cluster
        self.assertEqual(good.stats['n_clusters'], 64)
        store = G.GateStore(self.tmp / 'even')
        _, even = self.run_eval(swap_plan(base=BASE + 7), FakeRunner({}, adv=0.05), store=store)
        self.assertFalse(even.passed)
        self.assertLess(abs(even.stats['score'] - 0.5), 0.04)

    def test_g2_record_only_and_trend(self):
        _, rep = self.run_eval(swap_plan('g2', n=500), FakeRunner({'cand': 0.15}))
        self.assertFalse(rep.blocking)
        self.assertFalse(rep.passed)                             # never blocks, never passes
        self.assertGreater(rep.stats['lo'], 0.5)
        # G2 is fixed: the same openings and seeds for the next champion
        p2 = G.seat_swap_plan('g2', 'g2-fixed', 500, panel=panel(), a=player('cand2'), b=player('champ'),
                              base_seed=BASE, env_digest=ENV)
        self.assertEqual(p2.block_digest(), swap_plan('g2', n=500).block_digest())
        _, rep2 = self.run_eval(p2, FakeRunner({'cand2': 0.20}))
        tr = G.g2_trend([rep, rep2])
        self.assertFalse(tr['blocking'])
        self.assertEqual(len(tr['points']), 2)
        self.assertGreater(tr['slope_per_promotion'], 0.0)

    def test_promotion_decision(self):
        _, g1 = self.run_eval(g1_plan(), FakeRunner({'cand': 0.10}))
        _, g1c = self.run_eval(swap_plan(), FakeRunner({'cand': 0.10}))
        self.assertTrue(G.promotion_decision(g1, g1c))
        # crn_only (two CRN gates): one G1 never promotes; a second G1 on the confirmation block does
        with self.assertRaises(G.GateError):
            G.promotion_decision(g1, None, 'crn_only')
        with self.assertRaises(G.GateError):                     # G1c is not the confirmation CRN gate
            G.promotion_decision(g1, g1c, 'crn_only')
        with self.assertRaises(G.GateError):                     # the same block twice is not a second gate
            G.promotion_decision(g1, g1, 'crn_only')
        conf_plan = G.crn_plan('g1', G.block_name('g1', update=20, confirm=True), 1024, panel=panel(),
                               candidate=player('cand'), champion=player('champ'), opponents=opponents(),
                               base_seed=BASE, env_digest=ENV)
        _, conf = self.run_eval(conf_plan, FakeRunner({'cand': 0.10}))
        self.assertTrue(conf.passed)
        self.assertTrue(G.promotion_decision(g1, conf, 'crn_only'))
        cs = G.GateStore(self.tmp / 'conf-null')
        _, conf_null = self.run_eval(conf_plan, FakeRunner({}), store=cs)
        self.assertFalse(conf_null.passed)
        self.assertFalse(G.promotion_decision(g1, conf_null, 'crn_only'))
        store = G.GateStore(self.tmp / 'weak')
        _, g1c_weak = self.run_eval(swap_plan(base=BASE + 3), FakeRunner({}), store=store)
        self.assertFalse(G.promotion_decision(g1, g1c_weak))
        with self.assertRaises(G.GateError):
            G.promotion_decision(g1, None)


# ------------------------------------------------------------------ resume and chunks
class TestResume(TmpCase):
    def test_resume_after_kill_is_identical(self):
        plan = g1_plan(n=512)
        ref_store = G.GateStore(self.tmp / 'ref')
        _, ref = self.run_eval(plan, FakeRunner({'cand': 0.04}), store=ref_store, chunk_size=50)
        killed = FakeRunner({'cand': 0.04}, boom_after=333)
        with self.assertRaises(KeyboardInterrupt):
            G.run_gate(plan, killed, self.store, chunk_size=50)
        st = self.store.status(plan)
        done = sum(v['chunks_done'] for v in st.values())
        self.assertEqual(done, 333 // 50)                        # only whole chunks were written
        # (a kill inside the atomic write itself: adv test_kill_during_chunk_write_leaves_nothing_partial)
        partial = G.evaluate(plan, self.store, settings())
        self.assertFalse(partial.valid)
        self.assertTrue(partial.replay_required or partial.completion < settings().min_completion)
        resumed = FakeRunner({'cand': 0.04})
        st2 = G.run_gate(plan, resumed, self.store, chunk_size=50)
        self.assertEqual(st2.state, 'complete')
        self.assertEqual(st2.reused, done)
        self.assertEqual(len(resumed.calls), 1024 - done * 50)
        self.assertEqual(len(set(resumed.calls) & set(killed.calls[:done * 50])), 0)
        rep = G.evaluate(plan, self.store, settings())
        self.assertEqual(json.dumps(rep.as_dict(), sort_keys=True), json.dumps(ref.as_dict(), sort_keys=True))
        # chunk files are byte-identical to the uninterrupted run
        for arm in plan.arms:
            for f in sorted(ref_store.arm_dir(plan, arm).glob('c*.json')):
                g = self.store.arm_dir(plan, arm) / f.name
                self.assertEqual(f.read_bytes(), g.read_bytes())

    def test_stop_and_deadline(self):
        plan = g1_plan(n=256)
        calls = {'n': 0}

        def stop():
            calls['n'] += 1
            return calls['n'] > 3
        st = G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32, stop=stop)
        self.assertEqual((st.state, st.chunks_run), ('stopped', 3))
        # deadline: a fake clock where each chunk takes 10 s; 25 s left -> 2 more chunks then 'deadline'
        t = {'now': 0.0}

        class SlowRunner(FakeRunner):
            def run(self, specs):
                t['now'] += 10.0
                return super().run(specs)
        st2 = G.run_gate(plan, SlowRunner({}), self.store, chunk_size=32, deadline=25.0, clock=lambda: t['now'])
        self.assertEqual(st2.state, 'deadline')
        self.assertEqual(st2.chunks_run, 2)
        st3 = G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)
        self.assertEqual(st3.state, 'complete')
        self.assertEqual(st3.chunks_done, 16)

    def test_stop_flag_marker_and_signal(self):
        marker = self.tmp / 'STOP'
        flag = G.StopFlag([marker])
        self.assertFalse(flag())
        marker.write_text('')
        self.assertTrue(flag())
        marker.unlink()
        flag.install([signal.SIGUSR1])
        try:
            os.kill(os.getpid(), signal.SIGUSR1)
            self.assertTrue(flag())
            self.assertEqual(flag.signalled, signal.SIGUSR1)
        finally:
            flag.uninstall()

    def test_store_refuses_mismatches(self):
        plan = g1_plan(n=128)
        G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)
        with self.assertRaises(G.GateError):                     # resume with another chunk size
            G.run_gate(plan, FakeRunner({}), self.store, chunk_size=16, ledger=False)
        # a changed schedule under the same block name
        other = g1_plan(n=128, pan=panel(70))
        with self.assertRaises(G.BlockReuseError):
            G.run_gate(other, FakeRunner({}), self.store, chunk_size=32)
        # a tampered chunk (wrong outcome key) is refused, not silently used
        p = self.store.chunk_path(plan, 'candidate', 1)
        d = json.loads(p.read_text())
        d['outcomes'][0]['key'] = d['outcomes'][1]['key']
        p.write_text(json.dumps(d))
        with self.assertRaises(G.GateError):
            G.evaluate(plan, self.store, settings())
        # a stray chunk outside the schedule is refused
        d['outcomes'][0]['key'] = d['keys'][0]
        p.write_text(json.dumps(d))
        (self.store.arm_dir(plan, 'candidate') / 'c00099.json').write_text('{}')
        with self.assertRaises(G.GateError):
            G.evaluate(plan, self.store, settings())

    def test_runner_contract(self):
        plan = g1_plan(n=64)

        class Short(FakeRunner):
            def run(self, specs):
                return super().run(specs)[:-1]

        class Dup(FakeRunner):
            def run(self, specs):
                out = super().run(specs)
                return out + out[:1]
        for bad in (Short({}), Dup({})):
            store = G.GateStore(self.tmp / type(bad).__name__)
            with self.assertRaises(G.GateError):
                G.run_gate(plan, bad, store, chunk_size=32)
            self.assertFalse(store.chunk_path(plan, 'candidate', 0).exists())
        with self.assertRaises(G.GateError):                     # a normal finish without a score
            G.GameOutcome(key='k', end='native').check()
        with self.assertRaises(G.GateError):                     # a failed game with a score
            G.GameOutcome(key='k', end='crash', score0=1.0).check()
        fr = G.FunctionRunner(lambda s: 1 / 0, catch=True)
        out = fr.run([plan.spec('candidate', 0)])
        self.assertEqual(out[0].end, 'error')
        with self.assertRaises(ZeroDivisionError):
            G.FunctionRunner(lambda s: 1 / 0).run([plan.spec('candidate', 0)])


class TestLedger(TmpCase):
    def test_fresh_block_refused_for_new_candidate(self):
        plan = g1_plan(n=64)
        G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)
        G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)          # same candidate: resume ok
        other = G.crn_plan('g1', plan.block, 64, panel=panel(), candidate=player('cand2'), champion=player('champ'),
                           opponents=opponents(), base_seed=BASE, env_digest=ENV)
        with self.assertRaises(G.BlockReuseError):
            G.run_gate(other, FakeRunner({}), self.store, chunk_size=32)
        # a fresh attempt block works
        nxt = G.crn_plan('g1', G.block_name('g1', update=20, attempt=1), 64, panel=panel(), candidate=player('cand2'),
                         champion=player('champ'), opponents=opponents(), base_seed=BASE, env_digest=ENV)
        self.assertEqual(G.run_gate(nxt, FakeRunner({}), self.store, chunk_size=32).state, 'complete')
        led = json.loads((self.store.root / 'ledger.json').read_text())
        self.assertEqual(set(led['blocks']), {'g1-u000020-a0', 'g1-u000020-a1'})


# ------------------------------------------------------------------ completeness
class TestCompleteness(TmpCase):
    def test_pair_dropped_whole(self):
        plan = g1_plan(n=1024)
        fail = [G.game_key(plan.block, 'candidate', 5), G.game_key(plan.block, 'champion', 9)]
        st, rep = self.run_eval(plan, FakeRunner({'cand': 0.1}, fail=fail))
        self.assertEqual(rep.n_complete, 1022)
        self.assertAlmostEqual(rep.completion, 1022 / 1024)
        self.assertTrue(rep.valid)
        self.assertEqual(rep.diagnostics['candidate']['failures'], {'crash': 1})
        self.assertEqual(rep.diagnostics['champion']['native'], 1023)
        # unit 5's champion game and unit 9's candidate game exist but are not scored
        ok = FakeRunner({'cand': 0.1})
        recs = []
        for u in plan.units:
            if u.unit in (5, 9):
                continue
            c = ok.game(plan.spec('candidate', u.unit)).score(u.our_owner)
            h = ok.game(plan.spec('champion', u.unit)).score(u.our_owner)
            recs.append((u.unit, u.case_id, c, h, u.holdout))
        ref = S.crn_gate(recs, n_expected=1024, n_boot=N_BOOT, seed=G.boot_seed(plan.block))
        self.assertEqual(rep.stats['delta_hat'], ref.delta_hat)

    def test_low_completion_invalid_even_with_big_gain(self):
        plan = g1_plan(n=1024)
        step = max(1, int(1.0 / (1.0 - settings().min_completion)) // 2)   # about twice the tolerated failed share
        fail = [G.game_key(plan.block, 'candidate', i) for i in range(0, 1024, step)]
        st, rep = self.run_eval(plan, FakeRunner({'cand': 0.2}, fail=fail))
        self.assertFalse(rep.valid)
        self.assertFalse(rep.passed)
        self.assertTrue(rep.replay_required)
        self.assertTrue(any('incomplete' in r for r in rep.reasons))
        self.assertFalse(rep.monitor_args(update=20)['valid'])
        # retry_failed replays only the chunks holding failures; a working runner then completes the gate
        r = FakeRunner({'cand': 0.2})
        st2 = G.run_gate(plan, r, self.store, chunk_size=64, retry_failed=True)
        self.assertEqual(st2.chunks_run, 16)                     # every candidate chunk had a failure
        self.assertTrue(all('/candidate/' in k for k in r.calls))
        rep2 = G.evaluate(plan, self.store, settings())
        self.assertTrue(rep2.valid and rep2.passed)

    def test_lost_chunk_counts_as_missing(self):
        plan = swap_plan(n=1024)
        G.run_gate(plan, FakeRunner({'cand': 0.1}), self.store, chunk_size=16)
        self.store.chunk_path(plan, 'a1', 3).unlink()             # 16 pairs lost without any record
        rep = G.evaluate(plan, self.store, settings())
        self.assertEqual(rep.n_complete, 1008)
        self.assertEqual(rep.n_expected, 1024)
        self.assertTrue(rep.valid)                               # 1008 / 1024 = 98.4 % >= min_completion
        self.assertAlmostEqual(rep.completion, 1008 / 1024)

    def test_lost_chunk_completion_value(self):
        plan = swap_plan(n=1024)
        G.run_gate(plan, FakeRunner({'cand': 0.1}), self.store, chunk_size=16)
        lost = int((1.0 - settings().min_completion) * 1024 / 16) + 1          # chunks: just below min_completion
        for i in range(3, 3 + lost):
            self.store.chunk_path(plan, 'a1', i).unlink()
        rep = G.evaluate(plan, self.store, settings())
        self.assertEqual(rep.n_complete, 1024 - 16 * lost)
        self.assertLess(rep.completion, settings().min_completion)
        self.assertFalse(rep.valid)
        self.assertEqual(rep.diagnostics['a1']['failures'], {'missing': 16 * lost})


# ------------------------------------------------------------------ diagnostics and report
class TestDiagnostics(TmpCase):
    def test_per_game_diagnostics(self):
        plan = g1_plan(n=128)
        runner = FakeRunner({'cand': 0.05})
        _, rep = self.run_eval(plan, runner)
        outs = {a: self.store.outcomes(plan, a) for a in plan.arms}
        for arm in plan.arms:
            d = rep.diagnostics[arm]
            games = [(plan.spec(arm, i), outs[arm][plan.spec(arm, i).key]) for i in range(128)]
            opp_nw = np.mean([o.seats[1 - s.scored_owner].non_wait for s, o in games])
            self.assertAlmostEqual(d['other_seat']['non_wait_per_game'], opp_nw)
            minutes = sum(o.length / G.TICKS_PER_MIN for _, o in games)
            plays = sum(o.seats[s.scored_owner].plays for s, o in games)
            self.assertAlmostEqual(d['scored_seat']['plays_per_min'], plays / minutes)
            ref = sum(o.seats[s.scored_owner].refusals.get('19', 0) for s, o in games)
            self.assertEqual(d['scored_seat']['refusals'].get('19', 0), ref)
            ot = np.mean([o.length > G.REGULATION_TICKS for _, o in games])
            self.assertAlmostEqual(d['overtime_frac'], ot)
            self.assertAlmostEqual(d['scored_seat']['card_play_rate']['hog-rider'],
                                   np.mean([o.seats[s.scored_owner].cards['hog-rider'] for s, o in games]))
            k = sum(o.score(s.scored_owner) for s, o in games)
            self.assertEqual(d['wilson'], list(S.wilson(k, 128)))
            self.assertEqual(set(d['by_opponent']), set(G.PANEL_KINDS))
            self.assertEqual(sum(v['games'] for v in d['by_opponent'].values()), 128)
        p = G.write_report(rep, self.store)
        back = json.loads(p.read_text())
        self.assertEqual(back['schema'], G.REPORT_SCHEMA)
        self.assertEqual(back['block'], plan.block)

    def test_cli_status_and_evaluate(self):
        plan = g1_plan(n=64)
        G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            G.main(['status', '--root', str(self.store.root), '--block', plan.block])
        st = json.loads(buf.getvalue())
        self.assertEqual(st['candidate']['chunks_done'], 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            G.main(['evaluate', '--root', str(self.store.root), '--block', plan.block, '--n-boot', str(N_BOOT)])
        rep = json.loads(buf.getvalue())
        self.assertEqual(rep['kind'], 'g1')
        self.assertEqual(rep['n_complete'], 64)


# ------------------------------------------------------------------ environment pin, settings rows, failure guard
def outs_of(plan: G.GatePlan, runner) -> dict:
    return {arm: {s.key: o for s, o in zip(plan.games(arm), runner.run(plan.games(arm)))} for arm in plan.arms}


class FaultRunner(FakeRunner):
    """FakeRunner whose games listed in ``faults`` {key: fault_owner} end in a seat stall of that owner."""

    def __init__(self, *a, faults: Dict[str, Optional[int]], **kw):
        super().__init__(*a, **kw)
        self.faults = faults

    def game(self, spec):
        if spec.key in self.faults:
            return G.GameOutcome(key=spec.key, end='seat_stall', fault_owner=self.faults[spec.key])
        return super().game(spec)


class TestEnvironmentPin(TmpCase):
    def test_env_digest_required_and_in_block_digest(self):
        with self.assertRaises(TypeError):                       # a required keyword
            G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand'), champion=player('champ'),
                       opponents=opponents(), base_seed=BASE)
        with self.assertRaises(G.GateError):
            G.seat_swap_plan('g1c', 'g1c-u000001-a0', 64, panel=panel(), a=player('cand'), b=player('champ'),
                             base_seed=BASE, env_digest='')
        a = g1_plan(n=64)
        self.assertEqual(a.block_dict()['env_digest'], ENV)
        self.assertTrue(all(sp.env_digest == ENV for sp in a.games('candidate')))
        b = G.crn_plan('g1', a.block, 64, panel=panel(), candidate=player('cand'), champion=player('champ'),
                       opponents=opponents(), base_seed=BASE, env_digest='f' * 64)
        self.assertNotEqual(a.block_digest(), b.block_digest())
        self.assertEqual(a.all_seeds(), b.all_seeds())           # same seeds, other environment
        self.assertEqual(G.GatePlan.from_dict(json.loads(json.dumps(a.to_dict()))).env_digest, ENV)

    def test_screen_games_never_reused_across_environments(self):
        champ = player('champ')

        def mk(cand, env, attempt=0):
            return G.crn_plan('screen', G.block_name('screen', champion_digest=champ.digest, attempt=attempt), 64,
                              panel=panel(), candidate=player(cand), champion=champ, opponents=opponents(),
                              base_seed=BASE, env_digest=env)
        G.run_gate(mk('cand1', ENV), FakeRunner({}), self.store, chunk_size=32)
        with self.assertRaises(G.BlockReuseError):               # ledger: the block's schedule changed
            G.run_gate(mk('cand2', 'f' * 64), FakeRunner({}), self.store, chunk_size=32)
        with self.assertRaises(G.BlockReuseError):               # store, even without the ledger
            G.run_gate(mk('cand2', 'f' * 64), FakeRunner({}), self.store, chunk_size=32, ledger=False)
        r = FakeRunner({})
        st = G.run_gate(mk('cand2', 'f' * 64, attempt=1), r, self.store, chunk_size=32)
        self.assertEqual((st.state, len(r.calls)), ('complete', 128))     # champion games replayed in the new env

    def test_env_digest_helper(self):
        from amini.ppo import config as CFG
        cfg = CFG.PPOConfig()
        d = G.env_digest(cfg, compat_digest='c1', env_sha='z1', runner_commit='r1')
        self.assertEqual(d, G.env_digest(cfg, compat_digest='c1', env_sha='z1', runner_commit='r1'))
        self.assertEqual(len(d), 64)
        self.assertNotEqual(d, G.env_digest(cfg, compat_digest='c1', env_sha='z2', runner_commit='r1'))
        self.assertNotEqual(d, G.env_digest(cfg, compat_digest='c2', env_sha='z1', runner_commit='r1'))
        self.assertNotEqual(d, G.env_digest(None, compat_digest='c1', env_sha='z1', runner_commit='r1'))
        with self.assertRaises(G.GateError):
            G.env_digest()


class TestArmSettings(unittest.TestCase):
    def test_arm_settings_must_match(self):
        cand = player('cand')
        for field_, val in (('gate_bias', ((0.0, -0.05, 0.0), (0.0, -0.1, 0.0))), ('hazard_dt', 'interval'),
                            ('code_tree', '/code/v2'), ('temperature', 0.5)):
            champ = G.Player('champ', 'c' * 64, dataclasses.replace(player('champ').spec, **{field_: val}))
            self.assertEqual(G.settings_diff(cand, champ), [field_])
            for kind in ('g1', 'screen'):
                with self.assertRaises(G.GateError):
                    G.crn_plan(kind, 'x-a0', 64, panel=panel(), candidate=cand, champion=champ, opponents=opponents(),
                               base_seed=BASE, env_digest=ENV)
            with self.assertRaises(G.GateError):
                G.seat_swap_plan('g1c', 'g1c-u000001-a0', 64, panel=panel(), a=cand, b=champ, base_seed=BASE,
                                 env_digest=ENV)
        # G2 / release / swap: A vs the IL reference at T* legitimately differ in settings
        il = G.Player('il_v4_tstar', 'i' * 64, dataclasses.replace(player('il').spec, temperature=0.5,
                                                                  hazard_dt='interval'))
        G.seat_swap_plan('g2', 'g2-fixed', 64, panel=panel(), a=cand, b=il, base_seed=BASE, env_digest=ENV)
        G.seat_swap_plan('release', 'release-a0', 64, panel=panel(), a=cand, b=il, base_seed=BASE, env_digest=ENV)

    def test_t_gate_is_t_star_il(self):
        opp = opponents()
        opp['il_v4_tstar'] = G.Player('il_v4_tstar', opp['il_v4_tstar'].digest,
                                      dataclasses.replace(opp['il_v4_tstar'].spec, temperature=0.5))
        with self.assertRaises(G.GateError):
            G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand'), champion=player('champ'),
                       opponents=opp, base_seed=BASE, env_digest=ENV)
        G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand', t=0.5),
                   champion=player('champ', t=0.5), opponents=opp, base_seed=BASE, env_digest=ENV)
        # no T* opponent in the mix: nothing to tie T_gate to
        G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand'), champion=player('champ'),
                   opponents=opp, base_seed=BASE, env_digest=ENV, panel_mix={'il_v2_heldout': 0.5, 'il_v4_t1': 0.5})

    def test_small_panel_refused_and_replay_only_for_incompleteness(self):
        with self.assertRaises(G.GateError):
            g1_plan(n=64, pan=panel(20))
        with self.assertRaises(G.GateError):
            G.seat_swap_plan('g1c', 'g1c-u000001-a0', 64, panel=panel(29), a=player('cand'), b=player('champ'),
                             base_seed=BASE, env_digest=ENV)
        plan = g1_plan(n=20)                                     # 20 openings scheduled: invalid, but complete
        rep = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FakeRunner({})))
        self.assertFalse(rep.valid)
        self.assertEqual(rep.completion, 1.0)
        self.assertFalse(rep.replay_required)                    # a replay cannot add openings
        self.assertTrue(any('too few cases' in r for r in rep.reasons))


class TestFailureGuard(unittest.TestCase):
    def test_outcome_fault_owner_contract(self):
        with self.assertRaises(G.GateError):
            G.GameOutcome(key='k', end='native', score0=1.0, fault_owner=0).check()
        with self.assertRaises(G.GateError):
            G.GameOutcome(key='k', end='seat_stall', fault_owner=2).check()
        o = G.GameOutcome(key='k', end='seat_stall', fault_owner=1)
        self.assertEqual(G.GameOutcome.from_dict(json.loads(json.dumps(o.to_dict()))), o)
        self.assertIsNone(G.GameOutcome.from_dict({'key': 'k', 'end': 'crash', 'seats': [{}, {}]}).fault_owner)

    def test_candidate_faults_block_g1(self):
        plan = g1_plan(n=1024)
        # true +10 pp, but the candidate (our seat) stalls in more units than failure_rate_max allows
        step = 1024 // (int(settings().failure_rate_max * 1024) + 4)
        faults = {plan.spec('candidate', i).key: plan.units[i].our_owner for i in range(0, 1024, step)}
        self.assertGreater(len(faults) / 1024, settings().failure_rate_max)
        rep = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FaultRunner({'cand': 0.10}, faults=faults)))
        self.assertTrue(rep.valid and rep.stats['lb'] > 0)
        self.assertEqual(rep.failure_check['counts']['x'], len(faults))
        self.assertTrue(rep.failure_check['blocks'])
        self.assertFalse(rep.passed)
        self.assertTrue(any(r.startswith('failures:') for r in rep.reasons), rep.reasons)
        # 'loss' policy: the stalls are scored as candidate losses, nothing is dropped and nothing blocks
        loss = G.evaluate(plan, None, settings(fault_policy='loss'),
                          outcomes=outs_of(plan, FaultRunner({'cand': 0.10}, faults=faults)))
        self.assertEqual(loss.n_complete, 1024)
        self.assertFalse(loss.failure_check['blocks'])
        self.assertTrue(loss.passed)
        self.assertLess(loss.delta_hat, rep.delta_hat)
        with self.assertRaises(G.GateError):
            G.GateSettings(fault_policy='ignore')

    def test_panel_and_champion_faults_do_not_block(self):
        plan = g1_plan(n=1024)
        opp = {plan.spec(arm, i).key: 1 - plan.units[i].our_owner for i in range(0, 1024, 100) for arm in plan.arms}
        rep = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FaultRunner({'cand': 0.10}, faults=opp)))
        self.assertEqual(rep.failure_check['counts'], {'x': 0, 'y': 0, 'panel': 22, 'unknown': 0})
        self.assertTrue(rep.passed, rep.reasons)
        champ = {plan.spec('champion', i).key: plan.units[i].our_owner for i in range(0, 1024, 50)}
        rep2 = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FaultRunner({'cand': 0.10}, faults=champ)))
        self.assertTrue(any('champion failures' in r for r in rep2.reasons), rep2.reasons)
        self.assertFalse(rep2.failure_check['blocks'])           # biased against the candidate: recorded only
        self.assertTrue(rep2.passed)

    def test_asymmetry_below_the_rate_limit(self):
        # 5 candidate-only faults vs 0 in the champion arm: rate 0.49 % is within the limit, p = 0.0625 does not flag;
        # with a looser asymmetry level it does and blocks
        plan = g1_plan(n=1024)
        faults = {plan.spec('candidate', i).key: None for i in (3, 200, 400, 600, 800)}
        outs = outs_of(plan, FaultRunner({'cand': 0.10}, faults=faults))
        rep = G.evaluate(plan, None, settings(), outcomes=outs)
        self.assertFalse(rep.failure_check['blocks'])
        self.assertAlmostEqual(rep.failure_check['asym_p'], 0.0625)
        loose = G.evaluate(plan, None, settings(failure_asym_p=0.1), outcomes=outs)
        self.assertTrue(loose.failure_check['blocks'])
        self.assertFalse(loose.passed)

    def test_seat_swap_environment_faults_are_not_asymmetric(self):
        plan = swap_plan('g1c', n=1024)
        env = {plan.spec(arm, i).key: None for i in range(0, 1024, 256) for arm in plan.arms}   # 8 unknown
        rep = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FaultRunner({'cand': 0.10}, faults=env)))
        self.assertEqual(rep.failure_check['counts']['unknown'], 8)
        self.assertEqual(rep.failure_check['asym_p'], 1.0)
        self.assertTrue(rep.passed, rep.reasons)
        # A (the candidate) stalls on its own seat in more games than failure_rate_max allows: blocks
        step = 1024 // (int(settings().failure_rate_max * 1024) + 4)
        a = {plan.spec(arm, i).key: plan.spec(arm, i).scored_owner for i in range(0, 1024, step) for arm in plan.arms}
        self.assertGreater(len(a) / 2048, settings().failure_rate_max)
        rep2 = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FaultRunner({'cand': 0.10}, faults=a)))
        self.assertEqual(rep2.failure_check['counts']['x'], len(a))
        self.assertFalse(rep2.passed)

    def test_binomial(self):
        self.assertEqual(G.binom_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(G.binom_two_sided(5, 5), 0.0625)
        self.assertAlmostEqual(G.binom_two_sided(0, 5), 0.0625)
        self.assertEqual(G.binom_two_sided(3, 6), 1.0)


class TestScheduleAndIntegration(TmpCase):
    def test_schedule_check_and_plan_gate(self):
        st = G.GateSettings(n_boot=N_BOOT)
        small = g1_plan(n=256)
        rep = G.evaluate(small, None, st, outcomes=outs_of(small, FakeRunner({'cand': 0.15})))
        self.assertTrue(rep.stats['passed'])
        self.assertFalse(rep.passed)                             # a 256-triple G1 is not the configured G1
        self.assertTrue(any(f'configured g1 size {st.g1_pairs}' in r for r in rep.reasons), rep.reasons)
        mix = G.GateSettings(n_boot=N_BOOT, panel_mix={'il_v2_heldout': 0.6, 'il_v4_tstar': 0.2, 'il_v4_t1': 0.2})
        plan = G.plan_gate('g1', mix, panel=panel(), candidate=player('cand'), champion=player('champ'),
                           opponents=opponents(), base_seed=BASE, env_digest=ENV, update=20)
        self.assertEqual(plan.block, 'g1-u000020-a0')
        self.assertEqual(Counter(u.opp_kind for u in plan.units), C.allocate(mix.g1_pairs, dict(mix.panel_mix)))
        self.assertEqual(G.schedule_check(plan, mix), [])
        self.assertTrue(G.schedule_check(plan, st))              # default mix: the plan is not as configured
        self.assertTrue(G.schedule_check(g1_plan(), mix))
        g1c = G.plan_gate('g1c', st, panel=panel(), candidate=player('cand'), champion=player('champ'),
                          base_seed=BASE, env_digest=ENV, update=20)
        self.assertEqual((g1c.block, g1c.n_expected, g1c.players['a'].name), ('g1c-u000020-a0', st.g1c_pairs, 'cand'))
        scr = G.plan_gate('screen', st, panel=panel(), candidate=player('cand'), champion=player('champ'),
                          opponents=opponents(), base_seed=BASE, env_digest=ENV)
        self.assertEqual((scr.block, scr.n_expected), ('screen-' + player('champ').digest[:16] + '-a0', st.screen_pairs))
        conf = G.plan_gate('g1', st, panel=panel(), candidate=player('cand'), champion=player('champ'),
                           opponents=opponents(), base_seed=BASE, env_digest=ENV, update=20, confirm=True)
        self.assertEqual(conf.block, 'g1-u000020-a0-confirm')
        self.assertEqual(set(conf.all_seeds()) & set(plan.all_seeds()), set())
        with self.assertRaises(G.GateError):
            G.plan_gate('swap', st, panel=panel(), candidate=player('cand'), champion=player('champ'),
                        base_seed=BASE, env_digest=ENV, tag='x')
        # settings from the config read the rollback threshold and the failure limit
        from amini.ppo import config as CFG
        cfg = CFG.PPOConfig()
        cs = G.GateSettings.from_config(cfg)
        self.assertAlmostEqual(cs.rollback_delta, cfg.gates.rollback_delta_pp / 100.0)
        self.assertAlmostEqual(cs.failure_rate_max, cfg.monitor.drop_rate_warn)

    def test_rollback_threshold_from_settings(self):
        plan = g1_plan(n=1024)
        outs = outs_of(plan, FakeRunner({'cand': -0.02}))
        rep = G.evaluate(plan, None, settings(), outcomes=outs)
        self.assertGreater(rep.delta_hat, settings().rollback_delta)
        self.assertFalse(rep.alarm)
        self.assertTrue(G.evaluate(plan, None, settings(rollback_delta=rep.delta_hat + 1e-9), outcomes=outs).alarm)
        self.assertFalse(G.evaluate(plan, None, settings(rollback_delta=rep.delta_hat - 1e-9), outcomes=outs).alarm)

    def test_pilot_and_p6_args(self):
        from amini.ppo import config as CFG
        try:
            from amini.ppo import phase_rules as PR
        except ImportError:
            self.skipTest('needs amini.ppo.phase_rules, which is not part of this repository')
        plan = g1_plan(n=1024)
        rep = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, FakeRunner({'cand': -0.02})))
        pa = rep.pilot_args()
        self.assertEqual(pa['g1_delta'], rep.delta_hat)
        self.assertEqual(pa['g1_upper'], rep.stats['ub_boot'])
        self.assertGreater(pa['g1_upper'], 0.0)
        ana = G.evaluate(plan, None, settings(method='analytic'), outcomes=outs_of(plan, FakeRunner({'cand': -0.02})))
        self.assertEqual(ana.pilot_args()['g1_upper'], ana.stats['ub_analytic'])
        cfg = CFG.PPOConfig()
        res = PR.pilot_pass(cfg, stop_rule_fired=False, joint_kl_per_update=[0.01] * 20, plays_per_min_ratio=1.0,
                            core_card_ratio={c: 1.0 for c in cfg.monitor.core_cards}, ev_overall_end=0.3, ev_warmup_end=0.3, **pa)
        self.assertTrue(res.passed, res)
        with self.assertRaises(G.GateError):
            G.evaluate(swap_plan(n=64), None, settings(), outcomes=outs_of(swap_plan(n=64), FakeRunner({}))).pilot_args()
        g2a = swap_plan('g2', n=500)
        r1 = G.evaluate(g2a, None, settings(), outcomes=outs_of(g2a, FakeRunner({'cand': 0.15})))
        self.assertEqual(G.g2_p6_args([]), {'g2_score': None, 'g2_ci_low': None})
        args = G.g2_p6_args([r1])
        self.assertEqual((args['g2_score'], args['g2_ci_low']), (r1.stats['score'], r1.stats['lo']))
        self.assertTrue(PR.p6_entry(cfg, had_promotion=False, plateau_after_promotion=False, **args)[0])

    def test_passive_opponent_diagnostics(self):
        plan = g1_plan(n=128)

        class Passive(FakeRunner):
            def game(self, spec):
                o = super().game(spec)
                if spec.unit % 4:
                    return o
                seats = list(o.seats)
                seats[1 - spec.scored_owner] = G.SeatStats(queries=300, non_wait=2, plays=2)
                return dataclasses.replace(o, seats=tuple(seats))
        rep = G.evaluate(plan, None, settings(), outcomes=outs_of(plan, Passive({})))
        for arm in plan.arms:
            pas = rep.diagnostics[arm]['passive_opponent']
            self.assertEqual(pas['games'], 32)
            self.assertAlmostEqual(pas['frac'], 0.25)
            self.assertIsNotNone(pas['score'])
            self.assertEqual(sum(v['passive_opponent']['games'] for v in rep.diagnostics[arm]['by_opponent'].values()),
                             32)


if __name__ == '__main__':
    unittest.main()
