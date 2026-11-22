"""Adversarial tests of tools/ppo_v7/gate.py.

Fake runners only: no game environment, worker, model or GPU. They need ``ppo_v7.cases``, which is not part of
this repository: without it the module is skipped. Two kinds of tests:

* plain tests check claims of the module (or cases the basic tests do not cover): seat advantage cancels in
  seat-swap pairs, null calibration with opening-level heterogeneity, kill during the atomic chunk write, stray
  temp files, corrupt or foreign chunks, arm order, the ledger's fresh kinds, empty gates, SPRT record-only,
  Monitor integration, config wiring, and the size of the bias when failures correlate with the candidate's
  losses;
* ``test_fixed_*`` tests pin issues found in a review of the module that are fixed now.

    PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_gate_adv.py
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT / 'tests' / 'v7', ROOT / 'tools'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from ppo_v7 import cases as C                                     # noqa: E402
except ImportError:                                                   # pragma: no cover
    raise unittest.SkipTest('needs ppo_v7.cases, which is not part of this repository')
from ppo_v7 import gate as G                                          # noqa: E402
from amini.ppo import config as CFG                            # noqa: E402
from amini.ppo import monitor as M                             # noqa: E402
from amini.ppo import stats as S                               # noqa: E402
from test_ppo_gate import (BASE, ENV, N_BOOT, FakeRunner, g1_plan, opponents, panel, player,   # noqa: E402
                           swap_plan, u01)

ADV = BASE + 9000


def st(**kw) -> G.GateSettings:
    kw.setdefault('check_schedule', False)            # small gates; the schedule check has its own tests
    return G.GateSettings(n_boot=kw.pop('n_boot', N_BOOT), **kw)


def outcomes_of(plan: G.GatePlan, runner) -> dict:
    """Play a plan in memory (no store): {arm: {key: outcome}}."""
    return {arm: {s.key: o for s, o in zip(plan.games(arm), runner.run(plan.games(arm)))} for arm in plan.arms}


class HeteroRunner:
    """CRN fake with opening-level heterogeneity: champion p = 0.5, candidate p = 0.5 + eff[case]. A share
    ``indep`` of games uses an independent uniform, the rest the unit's shared environment seed and our torch seed
    (CRN). No draws. Unlike the basic FakeRunner, every uniform includes the unit's environment seed, so gates
    that differ only in base_seed are independent (the basic fake's 'indep' uniforms hash only the game key,
    which is the same for every base under one block name)."""

    def __init__(self, eff, indep: float = 0.5):
        self.eff, self.indep = eff, indep

    def game(self, spec: G.GameSpec) -> G.GameOutcome:
        so = spec.scored_owner
        me = spec.players[so]
        p = 0.5 + (self.eff.get(spec.case_id, 0.0) if me.name == 'cand' else 0.0)
        if u01(spec.key, spec.rnd_seed, 'mix') < self.indep:
            u = u01(spec.key, spec.rnd_seed, me.digest, 'i')
        else:
            u = u01(spec.rnd_seed, spec.seeds[f'o{so}_torch'], 'c')
        s = 1.0 if u < p else 0.0
        return G.GameOutcome(key=spec.key, end='native', score0=s if so == 0 else 1.0 - s, length=3600)

    def run(self, specs):
        return [self.game(s) for s in specs]


class LoserStall(FakeRunner):
    """FakeRunner whose candidate stalls (end 'seat_stall') in a share ``q`` of the games it would have lost."""

    def __init__(self, *a, q: float, **kw):
        super().__init__(*a, **kw)
        self.q = q

    def game(self, spec):
        o = super().game(spec)
        if spec.arm == 'candidate' and o.score(spec.scored_owner) == 0.0 and u01(spec.key, 'stall') < self.q:
            return G.GameOutcome(key=spec.key, end='seat_stall', error='fake stall')
        return o


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='ppo-gate-adv-'))
        self.store = G.GateStore(self.tmp / 'gates')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------ statistics with known truth
class TestKnownTruth(unittest.TestCase):
    def test_seat_advantage_cancels_in_pairs(self):
        # seat advantage 20 pp for owner 0, no strength difference: each arm is far from 0.5 but the pair is not
        plan = swap_plan('g1c', n=1024, base=ADV)
        rep = G.evaluate(plan, None, st(), outcomes=outcomes_of(plan, FakeRunner({}, adv=0.20)))
        self.assertTrue(rep.valid)
        self.assertLess(abs(rep.stats['score'] - 0.5), 4 * rep.stats['se'])
        self.assertGreater(rep.diagnostics['a0']['score'], 0.62)   # A on owner 0
        self.assertLess(rep.diagnostics['a1']['score'], 0.38)      # A on owner 1
        # +6 pp with the same seat advantage: the pair score moves by the strength difference only
        rep2 = G.evaluate(plan, None, st(), outcomes=outcomes_of(plan, FakeRunner({'cand': 0.06}, adv=0.20)))
        self.assertLess(abs(rep2.stats['score'] - 0.56), 4 * rep2.stats['se'])

    def test_crn_null_calibrated_with_opening_heterogeneity(self):
        # 120 independent null G1s of 256 triples over 64 openings. Per gate, the candidate's effect on each opening
        # is drawn uniform with SD 25 pp around a population mean of 0 (the panel mean is not 0): the opening cluster
        # bootstrap must hold its 5 % level. The same gates must pass a loose 30 % rule at about 30 %, so the check is
        # not vacuous.
        pan = panel(64)
        p05 = p30 = 0
        for k in range(120):
            eff = {c.case_id: 0.25 * math.sqrt(3) * (2 * u01('eff', k, c.case_id) - 1) for c in pan}
            plan = g1_plan(n=256, base=ADV + k, pan=pan)
            outs = outcomes_of(plan, HeteroRunner(eff))
            rep = G.evaluate(plan, None, st(n_boot=500, alpha=0.05), outcomes=outs)
            self.assertTrue(rep.valid)
            self.assertEqual(rep.stats['n_cases'], 64)
            p05 += rep.stats['lb'] > 0
            p30 += G.evaluate(plan, None, st(n_boot=500, alpha=0.30), outcomes=outs).stats['lb'] > 0
        self.assertLessEqual(p05, 14)          # Bin(120, 0.05): mean 6, P(>= 15) ~ 0.1 %
        self.assertGreaterEqual(p30, 20)       # Bin(120, 0.30): mean 36, P(< 20) ~ 0.01 %

    def test_candidate_correlated_failures_bias_delta(self):
        # True delta 0. The candidate stalls in ~4 % of the games it would lose: ~1.8 % of units are dropped,
        # completion stays above min_completion, the gate is valid, and delta_hat moves up by roughly half the
        # dropped share.
        plan = g1_plan(n=1024, base=ADV + 500)
        ref = G.evaluate(plan, None, st(), outcomes=outcomes_of(plan, FakeRunner({}, indep=1.0)))
        bad = G.evaluate(plan, None, st(), outcomes=outcomes_of(plan, LoserStall({}, indep=1.0, q=0.04)))
        self.assertTrue(ref.valid and bad.valid)
        dropped = 1024 - bad.n_complete
        self.assertGreater(dropped, 10)
        self.assertGreaterEqual(bad.completion, st().min_completion)
        self.assertEqual(bad.diagnostics['candidate']['failures'], {'seat_stall': dropped})
        self.assertEqual(bad.diagnostics['champion']['failures'], {})
        self.assertGreater(bad.delta_hat - ref.delta_hat, 0.004)

    def test_sprt_is_record_only(self):
        plan = g1_plan(n=256, base=ADV + 600)
        outs = outcomes_of(plan, FakeRunner({'cand': 0.10}))
        off = G.evaluate(plan, None, st(), outcomes=outs)
        on = G.evaluate(plan, None, st(sprt=True), outcomes=outs)
        self.assertIsNone(off.sprt)
        self.assertIn(on.sprt['decision'], ('H1', 'H0', 'cap', 'continue'))
        self.assertLessEqual(on.sprt['n'], 256)
        self.assertEqual((on.passed, on.stats), (off.passed, off.stats))     # the pass rule is unchanged
        swap = swap_plan('g1c', n=128, base=ADV + 601)
        self.assertIsNone(G.evaluate(swap, None, st(sprt=True), outcomes=outcomes_of(swap, FakeRunner({}))).sprt)


# ------------------------------------------------------------------ plans and seeds
class TestPlansAdv(unittest.TestCase):
    def test_balanced_openings_when_panel_does_not_divide(self):
        plan = g1_plan(n=1024, pan=panel(300))
        per = Counter(u.case_id for u in plan.units)
        self.assertEqual(len(per), 300)
        self.assertEqual(set(per.values()), {3, 4})
        for n, want in ((1000, (500, 300, 200)), (512, (256, 154, 102)), (7, (4, 2, 1)), (1023, (511, 307, 205))):
            p = g1_plan(n=n)
            got = Counter(u.opp_kind for u in p.units)
            self.assertEqual((got['il_v2_heldout'], got['il_v4_tstar'], got['il_v4_t1']), want)

    def test_crn_spec_maps_our_seat_for_both_owners(self):
        plan = g1_plan(n=128)
        owners = set()
        for u in plan.units:
            c = plan.spec('candidate', u.unit)
            sm, our, opp = u.seed_map(), u.our_owner, 1 - u.our_owner
            owners.add(our)
            self.assertIs(c.players[our], plan.players['candidate'])
            self.assertIs(c.players[opp], plan.players[u.opp_kind])
            for r in G.SIDE_ROLES:
                self.assertEqual(c.seeds[f'o{our}_{r}'], sm[f'our_{r}'])
                self.assertEqual(c.seeds[f'o{opp}_{r}'], sm[f'opp_{r}'])
                self.assertNotEqual(sm[f'our_{r}'], sm[f'opp_{r}'])
        self.assertEqual(owners, {0, 1})

    def test_blocks_of_one_update_share_no_seed(self):
        champ = player('champ')
        plans = [g1_plan(update=20), swap_plan('g1c', update=20),
                 G.crn_plan('screen', G.block_name('screen', champion_digest=champ.digest), 512, panel=panel(),
                            candidate=player('cand'), champion=champ, opponents=opponents(), base_seed=BASE, env_digest=ENV),
                 G.crn_plan('aa', G.block_name('aa'), 256, panel=panel(), candidate=champ, champion=champ,
                            opponents=opponents(), base_seed=BASE, env_digest=ENV),
                 swap_plan('g2', n=500), swap_plan('release', n=1000)]
        seen = {}
        for p in plans:
            s = set(p.all_seeds())
            for blk, other in seen.items():
                self.assertEqual(s & other, set(), (p.block, blk))
            seen[p.block] = s
        self.assertEqual(len({G.boot_seed(p.block) for p in plans}), len(plans))

    def test_panel_players_from_a_filled_config(self):
        cfg = CFG.PPOConfig()
        o = cfg.opponents
        cfg = dataclasses.replace(
            cfg, rollout=dataclasses.replace(cfg.rollout, hazard_dt='block'),
            opponents=dataclasses.replace(
                o, checkpoint={**o.checkpoint, 'il_v4': '/ck/v4.pt', 'il_v2_heldout': '/ck/v2.pt'},
                temperature={**o.temperature, 'il_v4_tstar': 0.5},
                hazard_dt={**o.hazard_dt, 'il_v2_heldout': 'interval'},
                code_tree={**o.code_tree, 'il_v2_heldout': '/code/v2'},
                gate_bias={**o.gate_bias, 'il_v2_heldout': {'base': [0.0, -0.05, 0.0], 'threat': [0.0, -0.1, 0.0]}}))
        pl = G.panel_players(cfg, {k: k[-2:] * 32 for k in G.PANEL_KINDS})
        self.assertEqual(pl['il_v4_tstar'].spec.temperature, 0.5)
        self.assertEqual(pl['il_v4_t1'].spec.temperature, 1.0)
        self.assertEqual(pl['il_v4_t1'].spec.harness_gate_bias(), 'none')       # never None
        self.assertEqual(pl['il_v2_heldout'].spec.harness_gate_bias(), [[0.0, -0.05, 0.0], [0.0, -0.1, 0.0]])
        self.assertEqual((pl['il_v2_heldout'].spec.hazard_dt, pl['il_v2_heldout'].spec.code_tree), ('interval', '/code/v2'))
        self.assertEqual((pl['il_v4_t1'].spec.hazard_dt, pl['il_v4_t1'].spec.code_tree), ('block', 'run'))
        # the plan keeps the settings row: round trip through JSON gives the same harness value (no None leak)
        with self.assertRaises(G.GateError):          # T_gate 1.0 != T*_IL 0.5
            G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand'), champion=player('champ'),
                       opponents=pl, base_seed=BASE, env_digest=ENV)
        plan = G.crn_plan('g1', 'g1-u000001-a0', 64, panel=panel(), candidate=player('cand', t=0.5),
                          champion=player('champ', t=0.5), opponents=pl, base_seed=BASE, env_digest=ENV)
        back = G.GatePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual(back.players['il_v2_heldout'].spec.harness_gate_bias(), [[0.0, -0.05, 0.0], [0.0, -0.1, 0.0]])

    def test_settings_from_a_changed_config(self):
        cfg = CFG.PPOConfig()
        g = cfg.gates
        cfg = dataclasses.replace(cfg, gates=dataclasses.replace(
            g, g1=dataclasses.replace(g.g1, alpha_one_sided=0.1, heldout_min_delta_pp=-2.0, games_per_side=2048),
            crn_panel={'il_v2_heldout': 0.6, 'il_v4_tstar': 0.2, 'il_v4_t1': 0.2}, sprt=True, sprt_h1_pp=5.0,
            min_completion=0.99, bootstrap_resamples=2000))
        s = G.GateSettings.from_config(cfg)
        self.assertEqual((s.alpha, s.g1_pairs, s.sprt, s.min_completion, s.n_boot), (0.1, 2048, True, 0.99, 2000))
        self.assertAlmostEqual(s.holdout_min, -0.02)
        self.assertAlmostEqual(s.sprt_h1, 0.05)
        self.assertEqual(dict(s.panel_mix), {'il_v2_heldout': 0.6, 'il_v4_tstar': 0.2, 'il_v4_t1': 0.2})


# ------------------------------------------------------------------ store, resume, ledger
class TestStoreAdv(Tmp):
    def test_kill_during_chunk_write_leaves_nothing_partial(self):
        plan = g1_plan(n=256)
        ref_store = G.GateStore(self.tmp / 'ref')
        G.run_gate(plan, FakeRunner({'cand': 0.05}), ref_store, chunk_size=32)
        ref = G.evaluate(plan, ref_store, st())
        real_replace = os.replace
        state = {'chunks': 0}

        def replace(src, dst):
            if Path(dst).name.startswith('c0') and Path(dst).suffix == '.json':
                state['chunks'] += 1
                if state['chunks'] == 5:
                    raise KeyboardInterrupt('killed inside atomic_write_json')
            return real_replace(src, dst)
        with mock.patch.object(os, 'replace', side_effect=replace):
            with self.assertRaises(KeyboardInterrupt):
                G.run_gate(plan, FakeRunner({'cand': 0.05}), self.store, chunk_size=32)
        for arm in plan.arms:
            d = self.store.arm_dir(plan, arm)
            self.assertEqual([f.name for f in d.iterdir() if f.name.endswith('.tmp')], [])
        self.assertEqual(sum(v['chunks_done'] for v in self.store.status(plan).values()), 4)
        r = FakeRunner({'cand': 0.05})
        self.assertEqual(G.run_gate(plan, r, self.store, chunk_size=32).state, 'complete')
        self.assertEqual(len(r.calls), 512 - 4 * 32)
        self.assertEqual(json.dumps(G.evaluate(plan, self.store, st()).as_dict(), sort_keys=True),
                         json.dumps(ref.as_dict(), sort_keys=True))

    def test_stray_temp_file_from_a_hard_kill_is_ignored(self):
        plan = g1_plan(n=128)
        G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)
        d = self.store.arm_dir(plan, 'candidate')
        (d / '.c00004.json.x1y2.tmp').write_text('{"half": ')
        rep = G.evaluate(plan, self.store, st())
        self.assertEqual(rep.n_complete, 128)
        self.assertEqual(G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32).chunks_run, 0)

    def test_truncated_chunk_raises_instead_of_being_rerun(self):
        plan = g1_plan(n=128)
        G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)
        p = self.store.chunk_path(plan, 'champion', 2)
        p.write_text(p.read_text()[:200])
        with self.assertRaises(G.GateError):
            G.evaluate(plan, self.store, st())
        with self.assertRaises(G.GateError):
            G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32)

    def test_chunk_from_another_block_is_refused(self):
        a, b = g1_plan(update=20, n=128), g1_plan(update=40, n=128)
        G.run_gate(a, FakeRunner({}), self.store, chunk_size=32)
        G.run_gate(b, FakeRunner({}), self.store, chunk_size=32)
        shutil.copy(self.store.chunk_path(a, 'candidate', 1), self.store.chunk_path(b, 'candidate', 1))
        with self.assertRaises(G.GateError):
            G.evaluate(b, self.store, st())

    def test_runner_answering_with_the_other_arm_writes_nothing(self):
        plan = g1_plan(n=64)

        class Swapped(FakeRunner):
            def run(self, specs):
                return super().run([plan.spec('champion' if s.arm == 'candidate' else 'candidate', s.unit)
                                    for s in specs])
        with self.assertRaises(G.GateError):
            G.run_gate(plan, Swapped({}), self.store, chunk_size=32)
        self.assertFalse(self.store.chunk_path(plan, 'candidate', 0).exists())

    def test_arm_order_does_not_matter(self):
        plan = g1_plan(n=256)
        ref_store = G.GateStore(self.tmp / 'ref')
        G.run_gate(plan, FakeRunner({'cand': 0.03}), ref_store, chunk_size=64)
        G.run_gate(plan, FakeRunner({'cand': 0.03}), self.store, chunk_size=64, arms=('champion',))
        part = G.evaluate(plan, self.store, st())
        self.assertEqual(part.n_complete, 0)
        self.assertFalse(part.valid)
        G.run_gate(plan, FakeRunner({'cand': 0.03}), self.store, chunk_size=64, arms=('candidate',))
        self.assertEqual(json.dumps(G.evaluate(plan, self.store, st()).as_dict(), sort_keys=True),
                         json.dumps(G.evaluate(plan, ref_store, st()).as_dict(), sort_keys=True))

    def test_stop_between_arms_scores_complete_pairs_only(self):
        plan = swap_plan('g1c', n=256)
        n = {'k': 0}

        def stop():
            n['k'] += 1
            return n['k'] > 5                       # a0 c0, a1 c0, a0 c1, a1 c1, a0 c2: a1 c2 missing
        s = G.run_gate(plan, FakeRunner({}), self.store, chunk_size=32, stop=stop)
        self.assertEqual((s.state, s.chunks_run, s.chunks_done), ('stopped', 5, 5))
        rep = G.evaluate(plan, self.store, st())
        self.assertEqual(rep.n_complete, 64)
        self.assertEqual(rep.diagnostics['a0']['native'], 96)
        self.assertEqual(rep.diagnostics['a1']['failures'], {'missing': 192})

    def test_retry_failed_reruns_the_whole_chunk(self):
        # documents the cost and the semantics: one failed game re-plays its whole chunk (good games included)
        plan = g1_plan(n=128)
        G.run_gate(plan, FakeRunner({}, fail=[G.game_key(plan.block, 'champion', 70)]), self.store, chunk_size=32)
        r = FakeRunner({})
        s = G.run_gate(plan, r, self.store, chunk_size=32, retry_failed=True)
        self.assertEqual((s.chunks_run, len(r.calls)), (1, 32))
        self.assertTrue(all('/champion/' in k for k in r.calls))
        self.assertEqual(G.evaluate(plan, self.store, st()).n_complete, 128)

    def test_ledger_fresh_kinds(self):
        def swap(kind, blk, a):
            return G.seat_swap_plan(kind, blk, 64, panel=panel(), a=player(a), b=player('champ'), base_seed=BASE, env_digest=ENV)
        for kind, blk in (('g1c', G.block_name('g1c', update=20)), ('release', G.block_name('release'))):
            G.run_gate(swap(kind, blk, 'cand'), FakeRunner({}), self.store, chunk_size=64)
            with self.assertRaises(G.BlockReuseError):
                G.run_gate(swap(kind, blk, 'cand2'), FakeRunner({}), self.store, chunk_size=64)
        for kind, blk in (('g2', G.block_name('g2')), ('swap', G.block_name('swap', tag='t'))):
            G.run_gate(swap(kind, blk, 'cand'), FakeRunner({}), self.store, chunk_size=64)
            G.run_gate(swap(kind, blk, 'cand2'), FakeRunner({}), self.store, chunk_size=64)   # record-only: reused
        led = json.loads((self.store.root / 'ledger.json').read_text())
        self.assertEqual(len(led['blocks']['g2-fixed']['candidates']), 2)
        self.assertEqual(len(led['blocks']['g1c-u000020-a0']['candidates']), 1)

    def test_screen_block_after_panel_settings_change(self):
        # a changed panel setting (here the held-out opponent's code tree) refuses the old screen block loudly, and
        # the next attempt gives the same champion a new screen block
        champ = player('champ')

        def mk(cand, code, attempt):
            opp = opponents()
            opp['il_v2_heldout'] = G.Player('il_v2_heldout', opp['il_v2_heldout'].digest,
                                            dataclasses.replace(opp['il_v2_heldout'].spec, code_tree=code))
            return G.crn_plan('screen', G.block_name('screen', champion_digest=champ.digest, attempt=attempt), 64,
                              panel=panel(), candidate=player(cand), champion=champ, opponents=opp, base_seed=BASE,
                              env_digest=ENV)
        G.run_gate(mk('cand', 'run', 0), FakeRunner({}), self.store, chunk_size=64)
        with self.assertRaises(G.BlockReuseError):
            G.run_gate(mk('cand2', '/code/v2', 0), FakeRunner({}), self.store, chunk_size=64)
        self.assertEqual(G.run_gate(mk('cand2', '/code/v2', 1), FakeRunner({}), self.store, chunk_size=64).state,
                         'complete')


# ------------------------------------------------------------------ reports and integration
class TestReportsAdv(Tmp):
    def test_empty_gate_is_invalid_strict_json(self):
        for plan in (g1_plan(n=64), swap_plan('g1c', n=64)):
            rep = G.evaluate(plan, self.store, st())
            self.assertEqual((rep.valid, rep.passed, rep.alarm, rep.replay_required, rep.n_complete),
                             (False, False, False, True, 0))
            json.dumps(rep.as_dict(), allow_nan=False)
            for arm in plan.arms:
                self.assertEqual(rep.diagnostics[arm]['failures'], {'missing': 64})
                self.assertEqual(rep.diagnostics[arm]['wilson'], [None, None])
        args = G.evaluate(g1_plan(n=64), self.store, st()).monitor_args(update=20)
        self.assertFalse(args['valid'])
        self.assertTrue(math.isnan(args['delta_hat']))

    def test_monitor_rollback_from_gate_reports(self):
        champ = player('champ')
        scr = G.crn_plan('screen', G.block_name('screen', champion_digest=champ.digest), 256, panel=panel(),
                         candidate=player('cand'), champion=champ, opponents=opponents(), base_seed=ADV, env_digest=ENV)
        r_scr = G.evaluate(scr, None, st(), outcomes=outcomes_of(scr, FakeRunner({'cand': -0.25})))
        g1 = g1_plan(update=20, n=256, base=ADV)
        r_g1 = G.evaluate(g1, None, st(), outcomes=outcomes_of(g1, FakeRunner({'cand': -0.25})))
        g1_bad = g1_plan(update=20, n=256, base=ADV, attempt=1)
        part = outcomes_of(g1_bad, FakeRunner({'cand': -0.25}))
        part['champion'] = {k: v for i, (k, v) in enumerate(part['champion'].items()) if i < 128}
        r_bad = G.evaluate(g1_bad, None, st(), outcomes=part)
        self.assertTrue(r_scr.valid and r_g1.valid and not r_bad.valid)
        self.assertTrue(r_scr.alarm and r_g1.alarm)
        mon = M.Monitor()
        for i in range(mon.th.rollback_consecutive - 1):   # alarming gates below the consecutive count: no rollback
            self.assertFalse(any(a.kind == 'rollback' for a in mon.observe_gate(**r_scr.monitor_args(update=10 + i))))
        self.assertFalse(any(a.kind == 'rollback' for a in mon.observe_gate(**r_bad.monitor_args(update=20))))
        acts = mon.observe_gate(**r_g1.monitor_args(update=20))
        self.assertTrue(any(a.kind == 'rollback' for a in acts), acts)

    def test_promotion_identity_check_catches_swapped_roles(self):
        g1 = g1_plan(n=256, base=ADV)
        r_g1 = G.evaluate(g1, None, st(), outcomes=outcomes_of(g1, FakeRunner({'cand': 0.15})))
        rev = G.seat_swap_plan('g1c', G.block_name('g1c', update=20), 128, panel=panel(), a=player('champ'),
                               b=player('cand'), base_seed=ADV, env_digest=ENV)
        r_rev = G.evaluate(rev, None, st(), outcomes=outcomes_of(rev, FakeRunner({'champ': 0.15})))
        self.assertTrue(r_g1.passed and r_rev.passed)
        with self.assertRaises(G.GateError):
            G.promotion_decision(r_g1, r_rev)


# ------------------------------------------------------------------ pinned review findings (fixed)
class TestFindings(unittest.TestCase):
    def test_fixed_crn_only_promotes_on_a_single_g1(self):
        # the 'crn_only' rule means "promotion uses two CRN gates" (same game count as G1 + G1c);
        # promotion_decision(..., 'crn_only') never promotes on ONE passing G1.
        g1 = g1_plan(n=256, base=ADV)
        rep = G.evaluate(g1, None, st(), outcomes=outcomes_of(g1, FakeRunner({'cand': 0.15})))
        self.assertTrue(rep.passed)
        with self.assertRaises(G.GateError):        # wants the second CRN confirmation report
            G.promotion_decision(rep, None, 'crn_only')

    def test_fixed_candidate_correlated_failures_not_flagged(self):
        # a valid gate whose dropped units all come from candidate stalls in lost games is flagged and, being a G1,
        # cannot pass (true +10 pp here, so only the guard stops it).
        plan = g1_plan(n=1024, base=ADV + 500)
        rep = G.evaluate(plan, None, st(), outcomes=outcomes_of(plan, LoserStall({}, indep=1.0, q=0.04)))
        self.assertTrue(any('fail' in r.lower() or 'stall' in r.lower() for r in rep.reasons), rep.reasons)
        good = G.evaluate(plan, None, st(), outcomes=outcomes_of(plan, LoserStall({'cand': 0.10}, indep=1.0, q=0.04)))
        self.assertTrue(good.valid and good.stats['lb'] > 0)
        self.assertTrue(good.failure_check['blocks'])
        self.assertFalse(good.passed)

    def test_fixed_plan_does_not_pin_the_environment(self):
        # the environment / runner (sim-to-real switches, environment build, compat digest) enters the block digest,
        # so the screen never reuses champion games and a resumed gate never keeps chunks across an environment change.
        champ = player('champ')
        plan = G.crn_plan('screen', G.block_name('screen', champion_digest=champ.digest), 64, panel=panel(),
                          candidate=player('cand'), champion=champ, opponents=opponents(), base_seed=BASE, env_digest=ENV)
        keys = set(plan.block_dict())
        self.assertTrue(keys & {'env', 'env_digest', 'runner', 'runner_digest', 'compat_digest'}, sorted(keys))
        other = G.crn_plan('screen', plan.block, 64, panel=panel(), candidate=player('cand'), champion=champ,
                           opponents=opponents(), base_seed=BASE, env_digest='f' * 64)
        self.assertNotEqual(other.block_digest(), plan.block_digest())

    def test_fixed_g1_arms_with_different_settings_accepted(self):
        # G1 / screen compare the whole settings row: a champion carrying a deployed gate bias (or another hazard_dt /
        # code tree) against an unbiased candidate is refused, else G1 would measure the settings.
        cand = player('cand')
        champ = G.Player('champ', 'c' * 64, C.OpponentSpec(name='champ', checkpoint='/ck/champ.pt', temperature=1.0,
                                                            gate_bias=((0.0, -0.05, 0.0), (0.0, -0.1, 0.0)),
                                                            hazard_dt='interval', code_tree='run'))
        with self.assertRaises(G.GateError):
            G.crn_plan('g1', 'g1-u000020-a0', 64, panel=panel(), candidate=cand, champion=champ,
                       opponents=opponents(), base_seed=BASE, env_digest=ENV)

    def test_fixed_g1c_temperatures_not_checked(self):
        # G1c is "candidate vs champion" at T_gate like G1; different temperatures are refused.
        with self.assertRaises(G.GateError):
            G.seat_swap_plan('g1c', 'g1c-u000020-a0', 64, panel=panel(), a=player('cand', t=1.0),
                             b=player('champ', t=0.5), base_seed=BASE, env_digest=ENV)

    def test_fixed_screen_block_has_no_attempt(self):
        # a screen block has an attempt number; see TestStoreAdv.test_screen_block_after_panel_settings_change.
        d = 'ab' * 32
        self.assertNotEqual(G.block_name('screen', champion_digest=d, attempt=1), G.block_name('screen', champion_digest=d))


if __name__ == '__main__':
    unittest.main()
