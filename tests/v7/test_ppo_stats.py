"""Tests for src/amini/ppo/stats.py.

* sizing numbers at the module's example sizes: power table, seat-swap SE / threshold / power, screen and
  rollback probabilities, Wilson intervals;
* null simulations (2,000 gates each, fixed seeds): G1 false pass ~alpha, seat-swap rule ~2.5%,
  and the cluster bootstrap calibrated on smaller gates;
* completeness (a pair with a missing game dropped whole, below MIN_COMPLETION invalid), held-out
  sub-panel rule, rollback trigger, bootstrap reproducibility, SPRT and Beta posterior.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_stats.py -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from amini.ppo import stats as S


def _outcomes(rng: np.random.Generator, p: np.ndarray, draw: float = 0.08) -> np.ndarray:
    """Results in {0, 0.5, 1}: win with prob p (1 - draw), draw with prob draw."""
    u = rng.random(p.shape)
    pw = p * (1.0 - draw)
    return np.where(u < pw, 1.0, np.where(u < pw + draw, 0.5, 0.0))


def _crn_sim(rng: np.random.Generator, n_cases: int, per_case: int, delta: float = 0.0):
    """CRN triples: a case strength, a triple-level shift shared by both arms (the common random
    numbers), independent game noise. Candidate win prob shifted by ``delta``."""
    pc = rng.beta(2.0, 2.0, n_cases)
    p = np.clip(np.repeat(pc, per_case) + rng.normal(0.0, 0.15, n_cases * per_case), 0.02, 0.98)
    champ = _outcomes(rng, p)
    cand = _outcomes(rng, np.clip(p + delta, 0.0, 1.0))
    cases = np.repeat(np.arange(n_cases), per_case)
    hold = np.tile(np.arange(per_case) % 2 == 0, n_cases)       # half of each case's opponents held out
    return cases, cand, champ, hold


def _seat_swap_sim(rng: np.random.Generator, n_pairs: int, edge: float = 0.0) -> np.ndarray:
    """Pair scores of A vs B: per case a seat-0 advantage, A's edge ``edge`` on both seats."""
    p0 = np.clip(0.5 + rng.normal(0.0, 0.19, n_pairs), 0.02, 0.98)
    g1 = _outcomes(rng, np.clip(p0 + edge, 0.0, 1.0))            # A on seat 0
    g2 = 1.0 - _outcomes(rng, np.clip(p0 - edge, 0.0, 1.0))      # A on seat 1 (seat 0 = B)
    return (g1 + g2) / 2.0


class PlanNumbers(unittest.TestCase):
    """Power and sizing helpers at the module's example sizes."""

    def test_crn_sd_and_se(self):
        self.assertEqual([round(S.crn_sd(r), 3) for r in (0.14, 0.30, 0.45)], [0.656, 0.592, 0.524])
        self.assertEqual([round(100 * S.crn_se(r, S.G1_PAIRS), 2) for r in (0.14, 0.30, 0.45)], [1.47, 1.32, 1.17])

    def test_power_table(self):
        rows = S.g1_power_table()
        self.assertEqual([round(100 * r['required_delta'], 1) for r in rows], [1.9, 1.7, 1.5])
        got = [[round(100 * r['pass'][d]) for d in (0.03, 0.04, 0.05, 0.08)] for r in rows]
        self.assertEqual(got, [[78, 93, 98, 100], [84, 96, 99, 100], [90, 98, 100, 100]])
        self.assertAlmostEqual(S.pass_prob(0.0, S.crn_se(0.3, S.G1_PAIRS)), 0.1, places=12)

    def test_false_promotion(self):
        self.assertAlmostEqual(S.false_promotion_rate(), 0.0025)         # alpha_g1 * alpha_g1c

    def test_seat_swap_numbers(self):
        self.assertEqual(round(100 * S.seat_swap_se(100), 2), 3.00)
        self.assertEqual(round(100 * S.seat_swap_se(S.RELEASE_PAIRS), 2), 0.77)
        self.assertAlmostEqual(100 * S.seat_swap_se(S.G1C_PAIRS), 0.67, places=2)
        self.assertEqual(round(S.seat_swap_threshold(S.G1C_PAIRS), 3), 0.513)
        self.assertEqual(round(100 * S.seat_swap_pass_prob(0.515, S.G1C_PAIRS)), 61)
        self.assertEqual(round(100 * S.seat_swap_pass_prob(0.52, S.G1C_PAIRS)), 85)
        self.assertEqual([round(S.seat_swap_game_ratio(r), 2) for r in (0.14, 0.30, 0.45)], [0.84, 1.03, 1.31])
        self.assertAlmostEqual(S.seat_swap_pass_prob(0.5, S.G1C_PAIRS), 0.025, places=12)

    def test_screen_rollback_and_pilot(self):
        def phi(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        se_s = [S.crn_se(r, S.SCREEN_PAIRS) for r in (0.14, 0.30)]
        self.assertEqual([round(100 * x, 1) for x in se_s], [2.3, 2.1])
        p = [S.prob_below(S.ROLLBACK_DELTA, x) for x in se_s]
        self.assertTrue(all(0.005 <= x <= 0.02 for x in p), p)           # one screen at or below the threshold
        se_g = S.crn_se(0.30, S.G1_PAIRS)
        both = S.consecutive_rollback_prob(se_s[1], se_g)
        self.assertAlmostEqual(both / (phi(S.ROLLBACK_DELTA / se_s[1]) * phi(S.ROLLBACK_DELTA / se_g)), 1.0, places=9)
        pilot = [S.prob_below(-0.03, S.crn_se(r, S.G1_PAIRS)) for r in (0.30, 0.14)]
        self.assertEqual([round(100 * x, 1) for x in pilot], [1.2, 2.0])

    def test_wilson(self):
        lo, hi = S.wilson(13, 20)
        self.assertEqual((round(lo, 2), round(hi, 2)), (0.43, 0.82))
        self.assertEqual((round(lo, 3), round(hi, 3)), (0.433, 0.819))
        lo, hi = S.wilson(12, 20)
        self.assertEqual((round(lo, 2), round(hi, 2)), (0.39, 0.78))
        self.assertEqual(S.wilson(0, 0), (None, None))
        self.assertEqual(S.wilson(20, 20)[1], 1.0)
        self.assertAlmostEqual(S.wilson(0, 20)[0], 0.0, places=12)


class NullSimulations(unittest.TestCase):
    """2,000 gates under equal strength."""

    def test_g1_false_pass_analytic(self):
        rng = np.random.default_rng(17)
        lb_only = full = 0
        se = []
        for _ in range(2000):
            cases, cand, champ, hold = _crn_sim(rng, 256, 4)
            r = S.crn_arrays(cases, cand, champ, hold, method='analytic')
            lb_only += r.lb > 0
            full += r.passed
            se.append(r.se_cluster)
            # the rule is exactly: valid AND LB > 0 AND held-out delta_hat >= -1 pp
            self.assertEqual(r.passed, bool(r.valid and r.lb > 0 and r.holdout_delta >= S.HOLDOUT_MIN_DELTA))
        lb_rate, full_rate = lb_only / 2000, full / 2000
        self.assertTrue(0.08 <= lb_rate <= 0.12, lb_rate)                 # the default alpha is 0.1
        self.assertTrue(0.06 <= full_rate, full_rate)
        self.assertTrue(0.015 < float(np.mean(se)) < 0.025)            # SE of a 1,024-triple gate
        # case-level effect heterogeneity (the null where clustering matters) is in
        # test_ppo_stats_adv.AdvNull.test_g1_null_with_case_heterogeneity

    def test_seat_swap_false_pass_normal(self):
        rng = np.random.default_rng(7)
        n = 0
        sds = []
        for _ in range(2000):
            r = S.seat_swap_arrays(_seat_swap_sim(rng, 1024), method='normal')
            n += r.passed
            sds.append(r.sd)
        rate = n / 2000
        self.assertTrue(0.015 <= rate <= 0.037, rate)
        self.assertTrue(0.25 < float(np.mean(sds)) < 0.40)            # a realistic pair-score SD

    def test_bootstrap_calibration(self):
        rng = np.random.default_rng(11)
        g1 = ss = 0
        gates = 1000
        for i in range(gates):
            cases, cand, champ, hold = _crn_sim(rng, 64, 4)
            r = S.crn_arrays(cases, cand, champ, hold, method='bootstrap', n_boot=1000, seed=i)
            g1 += r.lb > 0
            self.assertAlmostEqual(r.se_boot, r.se_cluster, delta=0.25 * r.se_cluster)
            s = S.seat_swap_arrays(_seat_swap_sim(rng, 256), method='bootstrap', n_boot=1000, seed=i)
            ss += s.passed
        self.assertTrue(0.06 <= g1 / gates <= 0.15, g1 / gates)
        self.assertTrue(0.01 <= ss / gates <= 0.045, ss / gates)

    def test_g1_power_positive_effect(self):
        rng = np.random.default_rng(3)
        passes = sum(S.crn_arrays(*_crn_sim(rng, 256, 4, delta=0.08), method='analytic').passed
                     for _ in range(200))
        self.assertGreater(passes / 200, 0.9)


class Completeness(unittest.TestCase):

    def _records(self, n: int, delta_wins: int = 0, holdout_every: int = 2):
        """Triples where the candidate wins ``delta_wins`` extra games over a tie baseline."""
        recs = []
        for i in range(n):
            champ = float(i % 2)
            cand = 1.0 if i < delta_wins else champ
            recs.append(S.CrnRecord(f't{i}', f'c{i // 4}', cand, champ, i % holdout_every == 0))
        return recs

    def test_crn_missing_game_drops_pair(self):
        recs = self._records(S.G1_PAIRS, delta_wins=400)
        lost = [S.CrnRecord(r.triple_id, r.case_id, None if k % 2 else r.candidate,
                            r.champion if k % 2 else None, r.holdout)
                for k, r in enumerate(recs[:20])]
        r = S.crn_gate(lost + recs[20:], method='analytic')
        self.assertEqual(r.n_pairs, S.G1_PAIRS - 20)
        self.assertEqual(r.n_expected, S.G1_PAIRS)
        self.assertTrue(r.valid)
        self.assertAlmostEqual(r.completion, (S.G1_PAIRS - 20) / S.G1_PAIRS)
        kept = recs[20:]
        self.assertAlmostEqual(r.delta_hat, np.mean([x.candidate - x.champion for x in kept]))
        self.assertTrue(r.passed, r.reasons)

    def test_crn_below_min_completion_invalid(self):
        recs = self._records(S.G1_PAIRS, delta_wins=800)
        lost = [S.CrnRecord(x.triple_id, x.case_id, None, x.champion, x.holdout) for x in recs[:101]]
        r = S.crn_gate(lost + recs[101:], method='analytic')
        self.assertEqual(r.n_pairs, 1899)                               # 1,899 / 2,000 < 95%
        self.assertFalse(r.valid)
        self.assertGreater(r.lb, 0)
        self.assertFalse(r.passed)
        self.assertTrue(any('incomplete' in x for x in r.reasons))
        # triples lost without any record are counted through n_expected
        r2 = S.crn_gate(recs[:1899], n_expected=S.G1_PAIRS, method='analytic')
        self.assertFalse(r2.valid)
        self.assertFalse(S.crn_gate(recs, n_expected=S.G1_PAIRS, method='analytic').reasons)

    def test_screen_never_passes_and_alarms(self):
        recs = [S.CrnRecord(i, i // 4, 0.0 if i < 64 else 1.0, 1.0) for i in range(S.SCREEN_PAIRS)]
        r = S.screen_s(recs, method='analytic')
        self.assertFalse(r.passed)
        self.assertAlmostEqual(r.delta_hat, -64 / S.SCREEN_PAIRS)
        self.assertTrue(r.alarm)
        good = [S.CrnRecord(i, i // 4, 1.0, 0.0 if i < 64 else 1.0) for i in range(S.SCREEN_PAIRS)]
        self.assertFalse(S.screen_s(good, method='analytic').alarm)
        self.assertFalse(S.screen_s(good, method='analytic').passed)

    def test_seat_swap_missing_game_drops_case(self):
        games = []
        for c in range(1000):
            games += [S.SeatGame(c, 0, 1.0), S.SeatGame(c, 1, 0.5 if c % 3 else 0.0)]
        games[1] = S.SeatGame(0, 1, None)           # case 0 loses its seat-1 game
        del games[2:4]                              # case 1 loses both games: needs n_expected
        pairs, n_seen = S.seat_swap_pairs(games)
        self.assertNotIn(0, pairs)
        self.assertNotIn(1, pairs)
        self.assertEqual((len(pairs), n_seen), (998, 999))
        r = S.seat_swap_gate(games, n_expected=1000, method='normal')
        self.assertEqual(r.n_pairs, 998)
        self.assertTrue(r.valid)
        self.assertAlmostEqual(r.score, np.mean(list(pairs.values())))
        self.assertTrue(r.passed)
        r = S.seat_swap_gate(games[: 2 * 940], n_expected=1000, method='normal')
        self.assertFalse(r.valid)
        self.assertFalse(r.passed)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            S.crn_gate([(1, 1, 0.7, 0.0)])
        with self.assertRaises(ValueError):
            S.crn_gate([(1, 1, 1.0, 0.0), (1, 2, 1.0, 0.0)])
        with self.assertRaises(ValueError):
            S.seat_swap_pairs([(1, 2, 1.0)])
        with self.assertRaises(ValueError):
            S.seat_swap_pairs([(1, 0, 1.0), (1, 0, 0.0)])
        self.assertEqual(S.completion(0, 0), (0.0, False))
        with self.assertRaises(ValueError):
            S.completion(5, 4)


class Rules(unittest.TestCase):

    def test_holdout_subpanel(self):
        # overall clearly positive, but the held-out sub-panel is at -3.9 pp (20 losses in 512)
        recs = []
        for i in range(1024):
            hold = i % 2 == 0
            if hold:
                cand, champ = (0.0, 1.0) if i < 40 else (1.0, 1.0)
            else:
                cand, champ = (1.0, 0.0) if i < 600 else (0.0, 0.0)
            recs.append(S.CrnRecord(i, i // 4, cand, champ, hold))
        r = S.crn_gate(recs, n_expected=len(recs), method='analytic')
        self.assertGreater(r.lb, 0)
        self.assertAlmostEqual(r.holdout_delta, -20 / 512)
        self.assertFalse(r.passed)
        self.assertTrue(any('held-out' in x for x in r.reasons))
        no_hold = [S.CrnRecord(x.triple_id, x.case_id, x.candidate, x.champion, False) for x in recs]
        self.assertFalse(S.crn_gate(no_hold, n_expected=len(recs), method='analytic').passed)
        self.assertTrue(S.crn_gate(no_hold, n_expected=len(recs), method='analytic', require_holdout=False).passed)
        # -2 pp exactly passes (>= HOLDOUT_MIN_DELTA)
        rows = [S.CrnRecord(i, i, 1.0, 1.0, True) for i in range(100)]
        rows[0] = S.CrnRecord(0, 0, 0.0, 1.0, True)
        rows[1] = S.CrnRecord(1, 1, 0.0, 1.0, True)
        rows += [S.CrnRecord(100 + i, 100 + i, 1.0, 0.0, False) for i in range(100)]
        r = S.crn_gate(rows, n_expected=len(rows), method='analytic')
        self.assertAlmostEqual(r.holdout_delta, -0.02)
        self.assertTrue(r.passed, r.reasons)

    def test_rollback_trigger(self):
        self.assertTrue(S.rollback_triggered([0.02, -0.06, -0.05, -0.07]))
        self.assertFalse(S.rollback_triggered([-0.06, -0.06, -0.049]))
        self.assertFalse(S.rollback_triggered([-0.06, -0.06, None]))
        self.assertFalse(S.rollback_triggered([-0.06, -0.06]))
        self.assertFalse(S.rollback_triggered([-0.06, -0.06, -0.06, 0.0]))

    def test_cluster_se_reduces_to_iid(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=300)
        se = S.cluster_se(d, np.arange(300), 300)
        self.assertAlmostEqual(se, d.std(ddof=1) / math.sqrt(300), places=12)
        # clustered with a strong case effect: cluster SE > iid SE
        inv = np.repeat(np.arange(50), 6)
        d2 = np.repeat(rng.normal(size=50), 6) + 0.1 * rng.normal(size=300)
        self.assertGreater(S.cluster_se(d2, inv, 50), 1.5 * d2.std(ddof=1) / math.sqrt(300))

    def test_bootstrap_reproducible(self):
        rng = np.random.default_rng(5)
        cases, cand, champ, hold = _crn_sim(rng, 256, 4, delta=0.03)
        a = S.crn_arrays(cases, cand, champ, hold, seed=42)
        b = S.crn_arrays(cases, cand, champ, hold, seed=42)
        c = S.crn_arrays(cases, cand, champ, hold, seed=43)
        self.assertEqual(a.as_dict(), b.as_dict())
        self.assertNotEqual(a.se_boot, c.se_boot)             # another seed draws other resamples
        self.assertEqual(a.method, 'bootstrap')
        self.assertEqual(a.lb, a.lb_boot)
        self.assertAlmostEqual(a.lb_boot, a.lb_analytic, delta=0.006)
        x = S.cluster_bootstrap_means(cand - champ, cases, 256, n_boot=2500, seed=1)
        y = S.cluster_bootstrap_means(cand - champ, cases, 256, n_boot=2500, seed=1)
        self.assertEqual(x.shape, (2500,))
        self.assertTrue(np.array_equal(x, y))
        scores = _seat_swap_sim(np.random.default_rng(1), 300)
        s1 = S.seat_swap_arrays(scores, seed=9)
        self.assertEqual(s1.as_dict(), S.seat_swap_arrays(scores.copy(), seed=9).as_dict())
        self.assertNotEqual(s1.lo_boot, S.seat_swap_arrays(scores, seed=10).lo_boot)
        self.assertAlmostEqual(s1.lo_boot, s1.lo_normal, delta=0.01)
        # the record API is independent of record order (clusters numbered in sorted order)
        recs = [S.CrnRecord(i, int(cases[i]), float(cand[i]), float(champ[i]), bool(hold[i]))
                for i in range(cases.size)]
        perm = np.random.default_rng(0).permutation(len(recs))
        g0 = S.crn_gate(recs, seed=42, n_boot=2000)
        g1 = S.crn_gate([recs[i] for i in perm], seed=42, n_boot=2000)
        self.assertEqual(g0.as_dict(), g1.as_dict())

    def test_seat_swap_positive(self):
        rng = np.random.default_rng(8)
        r = S.seat_swap_arrays(_seat_swap_sim(rng, 1024, edge=0.08), n_boot=2000)
        self.assertTrue(r.passed, r.reasons)
        self.assertGreater(r.lo, 0.5)
        self.assertAlmostEqual(r.lo_boot, r.lo_normal, delta=0.004)


class Sprt(unittest.TestCase):

    def test_bounds_and_llr(self):
        lo, hi = S.sprt_bounds(0.05, 0.20)
        self.assertAlmostEqual(lo, math.log(0.2 / 0.95))
        self.assertAlmostEqual(hi, math.log(0.8 / 0.05))
        d = [0.5, -0.5, 1.0, 0.0]
        r = S.sprt_g1(d, sigma=0.6, min_pairs=1)
        self.assertEqual(r.decision, 'continue')
        self.assertAlmostEqual(r.llr, S.SPRT_DELTA1 / 0.36 * (1.0 - 4 * S.SPRT_DELTA1 / 2))
        self.assertEqual(S.sprt_g1([], rho=0.3).decision, 'continue')
        self.assertEqual(S.sprt_g1([1.0] * 200, rho=0.3).decision, 'H1')
        self.assertEqual(S.sprt_g1([1.0] * 200, rho=0.3).n, 64)            # no decision before min_pairs
        self.assertEqual(S.sprt_g1([-1.0] * 200, rho=0.3).decision, 'H0')

    def test_null_and_power(self):
        rng = np.random.default_rng(99)
        false = cap = 0
        n_null = []
        for _ in range(1000):
            cases, cand, champ, _ = _crn_sim(rng, S.G1_PAIRS // 4, 4)
            r = S.sprt_g1(cand - champ)
            false += r.passed
            cap += r.decision == 'cap'
            n_null.append(r.n)
            self.assertIn(r.decision, ('H0', 'H1', 'cap'))
        self.assertLessEqual(false / 1000, 0.125)                       # Wald bound alpha / (1 - beta)
        wins = 0
        n_alt = []
        for _ in range(300):
            cases, cand, champ, _ = _crn_sim(rng, S.G1_PAIRS // 4, 4, delta=0.06)
            r = S.sprt_g1(cand - champ)
            wins += r.passed
            n_alt.append(r.n)
        self.assertGreater(wins / 300, 0.6)
        self.assertLess(float(np.mean(n_null)), 900)
        self.assertLess(float(np.mean(n_alt)), 900)


class Beta(unittest.TestCase):

    def test_betainc(self):
        for x in (0.1, 0.37, 0.8):
            self.assertAlmostEqual(S.betainc(1.0, 1.0, x), x, places=12)
            self.assertAlmostEqual(S.betainc(2.0, 1.0, x), x * x, places=12)
        self.assertAlmostEqual(S.betainc(2.0, 3.0, 0.4), 0.5248, places=12)
        self.assertAlmostEqual(S.betainc(30.5, 12.5, 0.7) + S.betainc(12.5, 30.5, 0.3), 1.0, places=12)

    def test_posterior(self):
        post = S.BetaPosterior.from_results([1.0, 1.0, 0.5, 0.0, 1.0])
        self.assertEqual((post.a, post.b), (4.5, 2.5))
        self.assertAlmostEqual(post.mean, 4.5 / 7)
        self.assertAlmostEqual(S.BetaPosterior(2.0, 1.0).quantile(0.25), 0.5, places=9)
        lo, hi = post.interval(0.9)
        self.assertAlmostEqual(post.cdf(lo), 0.05, places=9)
        self.assertAlmostEqual(post.cdf(hi), 0.95, places=9)
        draws = post.sample(np.random.default_rng(0), 200_000)
        self.assertAlmostEqual(float(draws.mean()), post.mean, places=2)
        self.assertAlmostEqual(float(draws.var()), post.var, places=3)
        self.assertEqual(post.update(0.0, weight=2.0), S.BetaPosterior(4.5, 4.5))
        self.assertEqual(post.decay(0.0), S.BetaPosterior(1.0, 1.0))
        self.assertEqual(post.decay(1.0), post)
        with self.assertRaises(ValueError):
            post.update(0.3)


if __name__ == '__main__':
    unittest.main()
