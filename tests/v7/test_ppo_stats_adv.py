"""Adversarial tests for src/amini/ppo/stats.py.

Tests marked FIXED are regression tests for defects found in review; the others pin down properties
the first tests did not exercise:

* completeness must not be bypassed by the n_expected default; degenerate gates must not pass;
* the cluster bootstrap must not depend on the order of the records (a G1 decision can flip);
* the bootstrap must equal the evaluation harness's cluster bootstrap formula;
* calibration under case-level effect heterogeneity (the only null where clustering matters);
* G1c with openings reused across pairs (a panel with fewer openings than pairs);
* float32 NaN as a missing game, exact rule boundaries, betainc against the binomial identity.

Sizes and thresholds follow the module's example defaults.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_stats_adv.py -v
"""
from __future__ import annotations

import json
import math
import random
import unittest

import numpy as np

from amini.ppo import stats as S


def _outcomes(rng: np.random.Generator, p: np.ndarray, draw: float = 0.08) -> np.ndarray:
    u = rng.random(p.shape)
    pw = p * (1.0 - draw)
    return np.where(u < pw, 1.0, np.where(u < pw + draw, 0.5, 0.0))


def _het_crn(rng: np.random.Generator, n_cases: int, per_case: int, tau: float, delta: float = 0.0):
    """CRN triples whose candidate-champion effect varies by case (N(0, tau) in win probability,
    mean 0 over the panel): the null under which case clustering actually matters."""
    pc = rng.beta(2.0, 2.0, n_cases)
    eff = np.repeat(rng.normal(0.0, tau, n_cases), per_case)
    p = np.clip(np.repeat(pc, per_case) + rng.normal(0.0, 0.15, n_cases * per_case), 0.02, 0.98)
    champ = _outcomes(rng, p)
    cand = _outcomes(rng, np.clip(p + eff + delta, 0.0, 1.0))
    cases = np.repeat(np.arange(n_cases), per_case)
    hold = np.tile(np.arange(per_case) % 2 == 0, n_cases)
    return cases, cand, champ, hold


def _harness_bootstrap(values_by_case, n_boot: int = 10000, seed: int = 0, q=(2.5, 97.5)):
    """Copy of the evaluation harness's cluster bootstrap (not importable here)."""
    keys = sorted(values_by_case, key=str)
    sums = np.array([float(np.sum(values_by_case[k])) for k in keys])
    cnts = np.array([len(values_by_case[k]) for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    bs = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1e-12)
    return [float(x) for x in np.percentile(bs, list(q))]


def _flip_dataset():
    """A 1,024-triple gate (256 cases x 4, dataset seed 159) for the record-order check."""
    rng = np.random.default_rng(159)
    p = np.repeat(rng.beta(2, 2, 256), 4)
    champ = (rng.random(1024) < p).astype(float)
    cand = (rng.random(1024) < np.clip(p + 0.035, 0, 1)).astype(float)
    cases = np.arange(1024) // 4
    hold = np.arange(1024) % 2 == 0
    return cases, cand, champ, hold


class AdvCompleteness(unittest.TestCase):
    """'completion < MIN_COMPLETION -> the gate is invalid and replayed'."""

    def _favourable(self, n: int):
        return [S.CrnRecord(i, i // 4, 1.0 if i % 5 == 0 else float(i % 2), float(i % 2), i % 2 == 0)
                for i in range(n)]

    def test_default_n_expected_does_not_bypass_completeness(self):
        """FIXED. A G1 resumed after SIGTERM with only 600 of the scheduled triples played (the lost
        blocks wrote no records) must not be a valid, passing gate when called without n_expected."""
        recs = self._favourable(S.G1_PAIRS)[:600]
        try:
            r = S.crn_gate(recs, method='analytic')
        except TypeError:
            return                                   # n_expected made required: acceptable fix
        self.assertFalse(r.valid, f'600/{S.G1_PAIRS} triples judged valid (completion {r.completion}), '
                                  f'passed={r.passed}')

    def test_tiny_gate_does_not_pass(self):
        """FIXED. One held-out triple won by the candidate is not a G1 pass."""
        try:
            r = S.crn_gate([S.CrnRecord(0, 0, 1.0, 0.0, True)])
        except TypeError:
            return
        self.assertFalse(r.passed, r.as_dict())


class AdvClusters(unittest.TestCase):

    def test_single_cluster_bootstrap_not_degenerate(self):
        """FIXED. All triples carry one case_id (e.g. a caller bug). The bootstrap has one cluster, so
        every resample equals delta_hat and LB = delta_hat > 0 'passes', while the analytic bound is NaN
        and fails. A gate with < 2 clusters must not pass by either method."""
        recs = [S.CrnRecord(i, 'same', 1.0 if i < 3 else float(i % 2), float(i % 2), i % 2 == 0)
                for i in range(S.G1_PAIRS)]
        ra = S.crn_gate(recs, method='analytic')
        self.assertFalse(ra.passed)
        rb = S.crn_gate(recs, method='bootstrap')
        self.assertFalse(rb.passed, f'delta {rb.delta_hat}, lb {rb.lb}, se_boot {rb.se_boot}')

    def test_one_pair_seat_swap_bootstrap(self):
        """FIXED. One seat-swapped pair scoring 1.0: 'normal' fails (SE undefined) but the
        bootstrap gives lo = 1.0 and passes."""
        self.assertFalse(S.seat_swap_arrays(np.array([1.0]), method='normal').passed)
        self.assertFalse(S.seat_swap_arrays(np.array([1.0]), method='bootstrap').passed)

    def test_crn_bootstrap_independent_of_record_order(self):
        """FIXED. _cluster_index numbered cases by first appearance, so the same records in a different
        order (games.jsonl is written in completion order by parallel workers) gave a different bootstrap
        LB with the same seed, and a G1 decision near the bound could flip."""
        cases, cand, champ, hold = _flip_dataset()
        r0 = S.crn_arrays(cases, cand, champ, hold, seed=0)
        perm = np.random.default_rng(1001).permutation(1024)
        r1 = S.crn_arrays(cases[perm], cand[perm], champ[perm], hold[perm], seed=0)
        self.assertEqual(r0.delta_hat, r1.delta_hat)
        self.assertEqual((r0.lb_boot, r0.passed), (r1.lb_boot, r1.passed))

    def test_crn_record_shuffles_same_lb(self):
        """FIXED. crn_gate on shuffled records: identical data, identical seed -> identical LB."""
        rng = np.random.default_rng(5)
        p = np.repeat(rng.beta(2, 2, 256), 4)
        champ = (rng.random(1024) < p).astype(float)
        cand = (rng.random(1024) < p + 0.03).astype(float)
        recs = [S.CrnRecord(i, i // 4, cand[i], champ[i], i % 2 == 0) for i in range(1024)]
        rnd = random.Random(1)
        lbs = set()
        for _ in range(5):
            rr = list(recs)
            rnd.shuffle(rr)
            lbs.add(S.crn_gate(rr, n_expected=1024, seed=0, n_boot=2000).lb_boot)
        self.assertEqual(len(lbs), 1, sorted(lbs))

    def test_crn_bootstrap_matches_harness_formula(self):
        """FIXED (same root cause as the order tests). The docstring says the estimator is the harness's
        cluster bootstrap; with the harness's sorted-by-str case order and the same seed the alpha and
        1 - alpha percentiles (default alpha 0.1) should be identical."""
        rng = np.random.default_rng(3)
        d = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], 1024)
        cases = np.arange(1024) // 4
        cand = np.where(d > 0, d, 0.0)
        champ = cand - d
        byc = {}
        for c, x in zip(cases, d):
            byc.setdefault(int(c), []).append(float(x))
        r = S.crn_arrays(cases, cand, champ, None, require_holdout=False, seed=4)
        self.assertEqual([r.lb_boot, r.ub_boot], _harness_bootstrap(byc, seed=4, q=(10, 90)))

    def test_seat_swap_bootstrap_matches_harness_formula(self):
        """Passes: seat_swap_gate sorts cases like the harness, and the 1,000-row chunks consume the
        generator exactly like one (10,000, G) draw, so the CI equals the harness's to the bit."""
        rng = np.random.default_rng(3)
        games, by = [], {}
        for c in range(300):
            a0, a1 = (float(x) for x in rng.choice([0.0, 0.5, 1.0], 2))
            games += [S.SeatGame(c, 0, a0), S.SeatGame(c, 1, a1)]
            by[c] = [a0, a1]
        rnd = random.Random(2)
        rnd.shuffle(games)
        r = S.seat_swap_gate(games, n_expected=300, seed=4)
        self.assertEqual([r.lo_boot, r.hi_boot], _harness_bootstrap(by, seed=4))
        v = rng.random(500)
        m = S.cluster_bootstrap_means(v, np.arange(500), 500, n_boot=10000, seed=9)
        idx = np.random.default_rng(9).integers(0, 500, size=(10000, 500))
        self.assertTrue(np.array_equal(m, v[idx].sum(1) / 500.0))


class AdvNull(unittest.TestCase):

    def test_g1_null_with_case_heterogeneity(self):
        """Passes. Case-level effect heterogeneity (tau 0.15, 128 cases x 8): the CR1 bound stays near
        alpha while an iid bound is anti-conservative, so the clustering is load-bearing; a null without
        such heterogeneity cannot tell cluster from iid."""
        rng = np.random.default_rng(123)
        n, a, iid = 2000, 0, 0
        z = S.z_quantile(0.9)                           # the default alpha 0.1
        for _ in range(n):
            c, x, y, h = _het_crn(rng, 128, 8, 0.15)
            r = S.crn_arrays(c, x, y, h, method='analytic')
            a += r.lb_analytic > 0
            iid += (r.delta_hat - z * r.se_iid) > 0
        self.assertTrue(0.07 <= a / n <= 0.13, a / n)
        self.assertGreater(iid / n, 0.11, iid / n)
        rng = np.random.default_rng(124)
        b = 0
        for i in range(400):
            c, x, y, h = _het_crn(rng, 128, 8, 0.15)
            b += S.crn_arrays(c, x, y, h, method='bootstrap', n_boot=1000, seed=i).lb_boot > 0
        self.assertLessEqual(b / 400, 0.13, b / 400)

    def test_g1_null_tight(self):
        """Passes. 10,000 analytic gates on the plain null: the rule is calibrated at alpha."""
        rng = np.random.default_rng(777)
        n, a = 10000, 0
        for _ in range(n):
            pc = rng.beta(2.0, 2.0, 256)
            p = np.clip(np.repeat(pc, 4) + rng.normal(0.0, 0.15, 1024), 0.02, 0.98)
            champ, cand = _outcomes(rng, p), _outcomes(rng, p)
            a += S.crn_arrays(np.repeat(np.arange(256), 4), cand, champ, None, method='analytic',
                              require_holdout=False).lb > 0
        self.assertTrue(0.09 <= a / n <= 0.11, a / n)

    def test_seat_swap_repeated_openings(self):
        """Documents a G1c risk (passes). Pairs over fewer openings than pairs, with opening-level edge
        heterogeneity (tau 0.15). seat_swap_pairs refuses a repeated (opening, seat), so the caller must key
        by pair, and the per-pair rule's false-pass rate then rises above the nominal 2.5%; clustering by
        opening (seat_swap_arrays(cluster_ids=...)) brings it back below."""
        with self.assertRaises(ValueError):
            S.seat_swap_pairs([S.SeatGame('op7', 0, 1.0), S.SeatGame('op7', 1, 0.0),
                               S.SeatGame('op7', 0, 0.5), S.SeatGame('op7', 1, 1.0)])
        rng = np.random.default_rng(31)
        n, per_pair, clustered = 4000, 0, 0
        for _ in range(n):
            op = np.sort(rng.integers(0, 400, 1024))
            seat_adv = rng.normal(0.0, 0.19, 400)[op]
            edge = rng.normal(0.0, 0.15, 400)[op]
            p0 = np.clip(0.5 + seat_adv + rng.normal(0.0, 0.05, 1024), 0.02, 0.98)
            s = (_outcomes(rng, np.clip(p0 + edge, 0, 1)) + 1.0 - _outcomes(rng, np.clip(p0 - edge, 0, 1))) / 2
            per_pair += S.seat_swap_arrays(s, method='normal').passed
            clustered += S.seat_swap_arrays(s, cluster_ids=op, method='normal').passed
        self.assertGreater(per_pair / n, 0.037, per_pair / n)
        self.assertLess(clustered / n, 0.033, clustered / n)

    def test_sprt_null_rate(self):
        """Passes. 4,000 null SPRTs of G1_PAIRS pairs: false pass below the Wald bound alpha / (1 - beta),
        and a modest mean sample size."""
        rng = np.random.default_rng(55)
        n, f, ns = 4000, 0, []
        for _ in range(n):
            pc = rng.beta(2.0, 2.0, S.G1_PAIRS // 4)
            p = np.clip(np.repeat(pc, 4) + rng.normal(0.0, 0.15, S.G1_PAIRS), 0.02, 0.98)
            r = S.sprt_g1(_outcomes(rng, p) - _outcomes(rng, p))
            f += r.passed
            ns.append(r.n)
        self.assertLess(f / n, 0.125, f / n)
        self.assertLess(float(np.mean(ns)), 700)


class AdvInputs(unittest.TestCase):

    def test_float32_nan_is_missing(self):
        """FIXED. _is_missing only recognised Python/np.float64 NaN; a float32 NaN (results read from a
        float32 column) raised ValueError instead of dropping the pair."""
        recs = [S.CrnRecord(0, 0, np.float32('nan'), 1.0, True), S.CrnRecord(1, 1, 1.0, 0.0, True)]
        r = S.crn_gate(recs, n_expected=2, method='analytic', min_completion=0.5)
        self.assertEqual(r.n_pairs, 1)
        pairs, seen = S.seat_swap_pairs([S.SeatGame(0, 0, np.float32('nan')), S.SeatGame(0, 1, 1.0)])
        self.assertEqual((pairs, seen), ({}, 1))

    def test_exact_rule_boundaries(self):
        """Passes. delta_hat exactly at ROLLBACK_DELTA (-5 pp) alarms (<=), held-out exactly at
        HOLDOUT_MIN_DELTA (-2 pp) from 300 pairs passes."""
        recs = [S.CrnRecord(i, i, 0.0 if i < 5 else 1.0, 1.0) for i in range(100)]
        r = S.screen_s(recs, n_expected=100, method='analytic')
        self.assertEqual(r.delta_hat, -0.05)
        self.assertTrue(r.alarm)
        self.assertTrue(S.rollback_triggered([r.delta_hat, -0.05, -0.05]))
        rows = [S.CrnRecord(i, i, 0.0 if i < 6 else 1.0, 1.0, True) for i in range(300)]
        rows += [S.CrnRecord(300 + i, 300 + i, 1.0, 0.0, False) for i in range(300)]
        g = S.crn_gate(rows, n_expected=600, method='analytic')
        self.assertEqual(g.holdout_delta, -0.02)
        self.assertTrue(g.passed, g.reasons)

    def test_empty_gate_is_invalid_and_serialisable(self):
        """Passes. Nothing played: invalid, not passed, no crash; as_dict is JSON (NaN allowed)."""
        r = S.crn_gate([], n_expected=S.G1_PAIRS)
        self.assertFalse(r.valid)
        self.assertFalse(r.passed)
        json.dumps(r.as_dict())
        s = S.seat_swap_gate([], n_expected=S.G1C_PAIRS)
        self.assertFalse(s.passed)
        json.dumps(s.as_dict())

    def test_betainc_binomial_identity(self):
        """Passes. I_x(k, n-k+1) = P(Bin(n, x) >= k), up to n = 10,000 (worst error ~2e-13)."""
        def tail(n: int, k: int, x: float) -> float:
            lp = [math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * math.log(x)
                  + (n - j) * math.log1p(-x) for j in range(k, n + 1)]
            m = max(lp)
            return math.exp(m) * sum(math.exp(v - m) for v in lp)
        for n, k, x in [(20, 13, 0.5), (1000, 520, 0.5), (1000, 600, 0.58), (5000, 2600, 0.51),
                        (3000, 1, 1e-4), (3000, 2999, 0.999), (400, 200, 0.45), (10000, 5100, 0.5)]:
            self.assertAlmostEqual(S.betainc(k, n - k + 1, x), tail(n, k, x), delta=1e-11)

    def test_beta_posterior_rejects_negative_evidence(self):
        """FIXED. update(weight=-5) makes b = -3: mean -2.0 and quantile() raises a math
        domain error later, far from the cause. Invalid weights (and gamma outside [0, 1]) should
        raise at the call."""
        with self.assertRaises(ValueError):
            S.BetaPosterior(2.0, 2.0).update(0.0, weight=-5.0)


class AdvFixes(unittest.TestCase):
    """Guard rails and opening clusters for seat-swap."""

    def test_seat_swap_gate_requires_n_expected(self):
        """G1c, H1 and G2 sizes differ, so there is no safe default."""
        with self.assertRaises(TypeError):
            S.seat_swap_gate([S.SeatGame(0, 0, 1.0), S.SeatGame(0, 1, 1.0)])   # type: ignore[call-arg]

    def test_more_seen_than_expected_raises(self):
        recs = [S.CrnRecord(i, i, 1.0, 0.0, True) for i in range(40)]
        with self.assertRaises(ValueError):
            S.crn_gate(recs, n_expected=39)
        with self.assertRaises(ValueError):
            S.screen_s(recs, n_expected=39)
        games = [S.SeatGame(c, s, 1.0) for c in range(40) for s in (0, 1)]
        with self.assertRaises(ValueError):
            S.seat_swap_gate(games, n_expected=39)

    def test_few_cases_invalid(self):
        """20 cases x 50 triples, all won by the candidate: a huge delta, but too few clusters for any
        bound to mean something; the default MIN_CLUSTERS guard makes the gate invalid."""
        recs = [S.CrnRecord(i, i % 20, 1.0, 0.0, i % 2 == 0) for i in range(1000)]
        r = S.crn_gate(recs, n_expected=1000)
        self.assertFalse(r.valid)
        self.assertFalse(r.passed)
        self.assertTrue(any('too few cases' in x for x in r.reasons), r.reasons)
        self.assertTrue(S.crn_gate(recs, n_expected=1000, min_clusters=2).passed)
        recs = [S.CrnRecord(i, i % 40, 1.0, 0.0, i % 2 == 0) for i in range(1000)]
        self.assertTrue(S.crn_gate(recs, n_expected=1000).passed)

    def test_array_api_rejects_bad_results(self):
        with self.assertRaises(ValueError):
            S.crn_arrays(np.arange(3), np.array([1.0, np.nan, 0.0]), np.zeros(3))
        with self.assertRaises(ValueError):
            S.crn_arrays(np.arange(3), np.array([1.0, 0.7, 0.0]), np.zeros(3))
        with self.assertRaises(ValueError):
            S.seat_swap_arrays(np.array([0.5, np.nan]))
        with self.assertRaises(ValueError):
            S.seat_swap_arrays(np.array([0.5, 1.5]))

    def test_seat_swap_gate_clusters_by_opening(self):
        """Pairs carry the opening as cluster_id: the gate resamples openings, equal to the harness's
        cluster_bootstrap over {opening: [pair scores]} to the bit, and independent of game order."""
        rng = np.random.default_rng(12)
        games, by_open = [], {}
        for pair in range(600):
            op = f'op{int(rng.integers(0, 150))}'
            a0, a1 = (float(x) for x in rng.choice([0.0, 0.5, 1.0], 2, p=[0.4, 0.1, 0.5]))
            games += [S.SeatGame(pair, 0, a0, op), S.SeatGame(pair, 1, a1, op)]
            by_open.setdefault(op, []).append((a0 + a1) / 2.0)
        r = S.seat_swap_gate(games, n_expected=600, seed=3)
        self.assertTrue(r.clustered)
        self.assertEqual(r.n_clusters, len(by_open))
        self.assertEqual([r.lo_boot, r.hi_boot], _harness_bootstrap(by_open, seed=3))
        self.assertAlmostEqual(r.score, float(np.mean([x for v in by_open.values() for x in v])), places=12)
        random.Random(4).shuffle(games)
        self.assertEqual(S.seat_swap_gate(games, n_expected=600, seed=3).as_dict(), r.as_dict())
        flat = S.seat_swap_gate(games, n_expected=600, seed=3, cluster=False)
        self.assertFalse(flat.clustered)
        self.assertEqual(flat.n_clusters, 600)
        with self.assertRaises(ValueError):
            S.seat_swap_pairs([S.SeatGame(0, 0, 1.0, 'a'), S.SeatGame(0, 1, 1.0, 'b')])

    def test_seat_swap_clustered_null_rate(self):
        """1,024 pairs over 300 openings with opening-level edge heterogeneity (tau 0.15): the
        clustered bootstrap keeps the false-pass rate near the nominal 2.5% (400 gates, 1,000 draws)."""
        rng = np.random.default_rng(41)
        n, f = 400, 0
        for i in range(n):
            op = rng.integers(0, 300, 1024)
            edge = rng.normal(0.0, 0.15, 300)[op]
            p0 = np.clip(0.5 + rng.normal(0.0, 0.19, 300)[op], 0.02, 0.98)
            s = (_outcomes(rng, np.clip(p0 + edge, 0, 1)) + 1.0 - _outcomes(rng, np.clip(p0 - edge, 0, 1))) / 2
            f += S.seat_swap_arrays(s, cluster_ids=op, n_boot=1000, seed=i).passed
        self.assertLessEqual(f / n, 0.05, f / n)

    def test_beta_posterior_rejects_invalid(self):
        with self.assertRaises(ValueError):
            S.BetaPosterior(0.0, 1.0)
        with self.assertRaises(ValueError):
            S.BetaPosterior(1.0, float('nan'))
        with self.assertRaises(ValueError):
            S.BetaPosterior(2.0, 2.0).decay(1.5)
        with self.assertRaises(ValueError):
            S.BetaPosterior(2.0, 2.0).decay(-0.1)
        with self.assertRaises(ValueError):
            S.BetaPosterior.from_results([], prior=(0.0, 1.0))
        self.assertEqual(S.BetaPosterior(2.0, 2.0).update(1.0, weight=0.0), S.BetaPosterior(2.0, 2.0))


if __name__ == '__main__':
    unittest.main()
