"""Adversarial verification of src/amini/ppo/buffer.py (the shard format the rollout writes and the learner
reads).

What the builder's suite (tests/v7/test_ppo_buffer.py) does not cover, and is checked here:

* zero-width ENTITY / OBJECT rows (``tokens_from_rows`` gives width 0 for a row with no entity / object, which
  ``SeatAdapter.make_row`` hands to ``peek`` unpadded) round trip, and the learner forward on the shard tokens
  (collate pads them to an all-invalid width 8) still equals the actor's mu;
* one batched learner forward over several seat-games (``TK.collate`` of many examples, rows in row-major order)
  equals the concatenated stored mu, i.e. the shard row order is the forward's valid-row order;
* a seat sampled at T != 1: the stored per-seat temperature reproduces mu on the learner side;
* fp32 peek outputs (what a remote policy server returns): mu is computed and stored in float64, bitwise;
* tokens stay aligned with rows when kept and dropped games interleave, after a rejected (ValueError) game, and
  across workers including a middle worker with only dropped games;
* the mirror outcome sign with a decisive result (the builder's mirror game is a draw, so -0 == 0 hid it);
* the shard carries every learner input (GAE, HT x seat weights, EV by phase and opponent);
* the writer copies its inputs (a rollout that reuses buffers after append_game cannot change the shard);
* bytes per row, and the size cost of the multi-row token dict.

``test_BUG_*`` tests pinned defects found in review as ``expectedFailure``; all are fixed and now run as plain
regression tests. ``test_DOC_*`` tests pin documented facts of the format; :class:`TestFixes` covers the
remaining review items. CPU only, deterministic seeds, no game environment. The episodes come from the private
dataset: the tests that need them are skipped.

Run: PYTHONPATH=src python -B -m unittest tests/v7/test_ppo_buffer_adv.py -v
"""

from __future__ import annotations

import copy
import inspect
import os
import sys
import tempfile
import unittest
import zipfile
from functools import lru_cache
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # tests/v7 (common, the builder's fixture)
import common  # noqa: E402
from common import S, example  # noqa: E402
import test_ppo_buffer as TB  # noqa: E402  (module import only: its TestCase classes are not collected here)

TK = getattr(common, "TK", None)                         # the tokenizer (not published)

from amini.nn.aminiformer import sample_actions  # noqa: E402
from amini.ppo import buffer as B  # noqa: E402
from amini.ppo import gae as G  # noqa: E402
from amini.ppo import logprob as LP  # noqa: E402

torch.set_num_threads(2)

UPDATE, CODE, CONF, POL, RID = TB.UPDATE, TB.CODE, TB.CONF, TB.POL, TB.RID
TOL = 1e-9                                                  # fp64 actor-vs-learner identity (builder's bound)
IDENTITY_PARAMS = ("policy_digest", "rollout_id", "run_id", "attempt")


# --------------------------------------------------------------------------- actor / learner helpers
def _cut(rt, name, uid_key):
    """Row token dict ``rt`` with group ``name`` (and its uid row key) cut to width 0."""
    rt = copy.deepcopy(rt)
    rt["groups"][name] = {k: np.ascontiguousarray(v[:, :0]) for k, v in rt["groups"][name].items()}
    rt["row"][uid_key] = np.ascontiguousarray(rt["row"][uid_key][:, :0])
    return rt


def run_actor(rows_tok, ticks, seed, temp=None, fp32=False):
    """The builder's actor loop on our own per-row tokens: peek + commit row by row, optional fp32 outputs
    (as a remote policy server returns them) and temperature (logprob.apply_temperature), sample_actions,
    actor_rows."""
    m = TB.model()
    state = m.initial_state(1, dtype=torch.float64)
    outs = []
    for rt in rows_tok:
        tt, pos = TB._peek_input(rt)
        outs.append(m.peek(tt, pos, state, ticks=torch.from_numpy(rt["row"]["tick"])))
        m.commit(state, torch.tensor([True]))
    actor = B.stack_peek_outputs(outs)
    if fp32:
        actor = {k: (v.float() if v.is_floating_point() else v) for k, v in actor.items()}
    if temp is not None:
        actor = LP.apply_temperature(actor, *temp)
    rng = np.random.default_rng(500 + seed)
    n = len(rows_tok)
    dt = (TB._unit_rate_dt({k: v.double() if v.is_floating_point() else v for k, v in actor.items()})
          * torch.from_numpy(rng.uniform(0.5, 2.0, n))).numpy()
    act = sample_actions(actor, torch.from_numpy(dt), generator=torch.Generator().manual_seed(900 + seed))
    rows = B.actor_rows(actor, dt, act, threat=rng.random(n) < 0.3, ticks=np.asarray(ticks, np.int64),
                        phi=rng.uniform(-0.6, 0.6, n), query_index=np.arange(n) * 2)
    return rows, actor, act


@lru_cache(maxsize=None)
def zero_width_seat(i, all_obj=False):
    """Episode i with OBJECT cut to width 0 on every 3rd row (every row when ``all_obj``) and ENTITY cut to
    width 0 on every 5th row (ability_ready cleared there, as a row without entities has no pointer target)."""
    _, rows_tok, _, _ = TB.actor_seat(i)
    out = []
    for j, rt in enumerate(rows_tok):
        if all_obj or j % 3 == 0:
            rt = _cut(rt, "OBJECT", "obj_uid")
        if j % 5 == 1:
            rt = _cut(rt, "ENTITY", "ent_uid")
            rt["row"]["ability_ready"] = np.zeros_like(rt["row"]["ability_ready"])
        out.append(rt)
    ticks = [int(rt["row"]["tick"][0]) for rt in out]
    rows, _, _ = run_actor(out, ticks, seed=i + (100 if all_obj else 0))
    return rows, out


@lru_cache(maxsize=None)
def tempered_seat(i, temp=(0.5, 0.7, 0.4)):
    _, rows_tok, _, _ = TB.actor_seat(i)
    ticks = [int(rt["row"]["tick"][0]) for rt in rows_tok]
    rows, _, _ = run_actor(rows_tok, ticks, seed=200 + i, temp=temp)
    return rows, rows_tok


def make_seat(rows, rows_tok, side, role="ours", **kw):
    n = len(rows["tick"])
    counters = {"queries": 2 * n, "committed": n, "forced": int(rows["forced"].sum()), "skipped": 0}
    return B.SeatRecord(side=side, role=role, weight=G.seat_weight(role), rows={k: v.copy() for k, v in rows.items()},
                        tokens=rows_tok, counters=counters, **kw)


def learner_terms(sgs, temp=None):
    """One train-mode forward of the tiny model over ``TK.collate`` of all seat-games ``sgs``, then
    ``executed_logp_terms`` on the raw output (rows in row-major valid order) with the stored actions."""
    m = copy.deepcopy(TB.model()).train()                  # dropout 0: train == eval
    b = TB.to_torch64(TK.collate([sg.example() for sg in sgs]))
    nb, tm = b["row"]["row_valid"].shape
    cell_rows = torch.zeros((nb, tm), dtype=torch.bool)
    play_slot = torch.full((nb, tm), -1, dtype=torch.long)
    acts = []
    for j, sg in enumerate(sgs):
        a = {k: torch.from_numpy(v.astype(np.int64) if v.dtype != np.bool_ else v) for k, v in sg.action().items()}
        play = a["family"] == S.FAMILY_PLAY
        cell_rows[j, :sg.n] = play
        play_slot[j, :sg.n] = torch.where(play, a["slot"], torch.full_like(a["slot"], -1))
        acts.append(a)
    with torch.no_grad():
        raw = m(b, cell_rows=cell_rows, play_slot=play_slot, with_aux=False)
    if temp is not None:
        raw = LP.apply_temperature(raw, *temp)
    act = {k: torch.cat([a[k] for a in acts]) for k in acts[0]}
    dt = torch.from_numpy(np.concatenate([sg.rows["dt"] for sg in sgs]))
    return LP.executed_logp_terms(raw, dt, act), raw


def _writer(root, worker=0, **kw):
    return TB.writer(root, worker, **kw)


class TmpDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ppo-buffer-adv-", dir=os.environ.get("TMPDIR") or None)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


# --------------------------------------------------------------------------- tokens and the learner forward
class TestTokensAndForward(TmpDir):

    @unittest.skipIf(TK is None, common.SKIP_REASON)
    def test_zero_width_ragged_rows_round_trip_and_match_mu(self):
        """make_row gives ENTITY / OBJECT width 0 on a row with no entity / object; peek sees width 0, the
        learner sees an all-invalid width 8 after collate. Both must give the same log pi."""
        rows_a, tok_a = zero_width_seat(0)
        rows_b, tok_b = zero_width_seat(1, all_obj=True)
        widths = [rt["groups"]["OBJECT"]["valid"].shape[1] for rt in tok_a]
        self.assertIn(0, widths)
        self.assertTrue(all(rt["groups"]["OBJECT"]["valid"].shape[1] == 0 for rt in tok_b))
        self.assertIn(0, [rt["groups"]["ENTITY"]["valid"].shape[1] for rt in tok_a])
        sa, sb = make_seat(rows_a, tok_a, 0, "ours"), make_seat(rows_b, tok_b, 1, "opponent")
        with _writer(self.root) as w:
            self.assertIsNone(w.append_game(TB.meta("w0-0", [sa, sb], result="win"), [sa, sb]))
        r = B.ShardReader(w.path)
        self.assertEqual(int(r.arrays["tok.w.OBJECT"][r.offsets[1]:].sum()), 0)
        sgs = list(r.seat_games())
        tok = sgs[1].tokens()
        self.assertEqual(tok["groups"]["OBJECT"]["valid"].shape[:2], (sgs[1].n, 0))
        self.assertEqual(tok["row"]["obj_uid"].shape, (sgs[1].n, 0))
        for sg, rts in zip(sgs, (tok_a, tok_b)):
            t = sg.tokens()
            for i, rt in enumerate(rts):
                for name, grp in rt["groups"].items():
                    for k, v in grp.items():
                        np.testing.assert_array_equal(t["groups"][name][k][i:i + 1, :v.shape[1]], v)
            terms, _ = learner_terms([sg])
            self.assertTrue(bool(terms["valid"].all()))
            self.assertLess(float((terms["total"] - torch.from_numpy(sg.rows["mu"])).abs().max()), TOL)
            self.assertEqual(B.mask_recompute_mismatch(sg), 0)

    @unittest.skipIf(TK is None, common.SKIP_REASON)
    def test_batched_forward_over_many_seat_games_matches_mu(self):
        """The learner batches seat-games; collate pads rows and ragged widths. The forward's valid rows in
        row-major order must line up with UpdateData.rows (the concatenated shard rows)."""
        TB.write_shard(self.root, 0)
        TB.write_shard(self.root, 1)
        u = B.load_update(self.root, UPDATE, expect_workers=2, expect_games=16)
        sgs = list(u.seat_games())
        self.assertGreater(len({sg.n for sg in sgs}), 1)
        terms, raw = learner_terms(sgs)
        self.assertEqual(int(raw["family"].shape[0]), u.n_rows)
        self.assertTrue(bool(terms["valid"].all()))
        self.assertLess(float((terms["total"] - torch.from_numpy(u.rows["mu"])).abs().max()), TOL)
        np.testing.assert_array_equal(LP.forced_mask(raw).numpy(), u.rows["forced"])

    @unittest.skipIf(TK is None, common.SKIP_REASON)
    def test_tempered_seat_reproduces_mu_from_the_stored_temperature(self):
        temp = (0.5, 0.7, 0.4)
        rows, rts = tempered_seat(2, temp)
        s = make_seat(rows, rts, 0, temperature=temp)
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-0", [s]), [s])
        sg = B.ShardReader(w.path).seat_game(0)
        t = tuple(float(x) for x in sg.seat["temperature"])
        self.assertEqual(t, temp)
        np.testing.assert_array_equal(B.ShardReader(w.path).seat["temperature"][0], temp)
        terms, _ = learner_terms([sg], temp=t)
        self.assertLess(float((terms["total"] - torch.from_numpy(sg.rows["mu"])).abs().max()), TOL)
        untempered, _ = learner_terms([sg])                  # T = 1 on the learner would be wrong
        self.assertGreater(float((untempered["total"] - torch.from_numpy(sg.rows["mu"])).abs().max()), 1e-3)

    def test_fp32_peek_outputs_store_float64_mu_bitwise(self):
        _, rows_tok, _, _ = TB.actor_seat(3)
        ticks = [int(rt["row"]["tick"][0]) for rt in rows_tok]
        rows, actor32, act = run_actor(rows_tok, ticks, seed=3, fp32=True)
        self.assertEqual(actor32["slot"].dtype, torch.float32)
        self.assertEqual(rows["mu"].dtype, np.float64)
        ref = LP.executed_logp(actor32, torch.from_numpy(rows["dt"]), act).numpy()
        np.testing.assert_array_equal(rows["mu"][rows["valid"]], ref[rows["valid"]])
        self.assertFalse(np.array_equal(rows["mu"], rows["mu"].astype(np.float32).astype(np.float64)))
        s = make_seat(rows, rows_tok, 0)
        with _writer(self.root, compress=True) as w:
            w.append_game(TB.meta("w0-0", [s]), [s])
        got = B.ShardReader(w.path).rows["mu"]
        np.testing.assert_array_equal(got.view(np.uint64), rows["mu"].view(np.uint64))


# --------------------------------------------------------------------------- row / game alignment
class TestAlignment(TmpDir):

    def _check_alignment(self, reader_or_update, expect):
        """Every seat-game's rows equal the input seat's rows and its tokens equal that seat's per-row tokens."""
        sgs = list(reader_or_update.seat_games())
        self.assertEqual([sg.game.game_id for sg in sgs], [gid for gid, _ in expect])
        for sg, (gid, s) in zip(sgs, expect):
            for k in B.ROW_FIELDS:
                np.testing.assert_array_equal(sg.rows[k], np.asarray(s.rows[k]).astype(B.ROW_FIELDS[k][0]), k)
            t = sg.tokens()
            np.testing.assert_array_equal(t["row"]["tick"].astype(np.int64), sg.rows["tick"])
            for i, rt in enumerate(s.tokens):
                for name in ("ENTITY", "OBJECT", "HAND"):
                    v = rt["groups"][name]["feat"]
                    np.testing.assert_array_equal(t["groups"][name]["feat"][i:i + 1, :v.shape[1]], v)
                np.testing.assert_array_equal(t["row"]["ent_uid"][i:i + 1, :rt["row"]["ent_uid"].shape[1]],
                                              rt["row"]["ent_uid"])

    def test_interleaved_kept_and_dropped_games_keep_tokens_aligned(self):
        kept, dropped = TB.kept_games(), TB.dropped_games()
        order = [kept[0], dropped[0][:2], kept[1], dropped[2][:2], dropped[4][:2], kept[2], dropped[1][:2]]
        with _writer(self.root) as w:
            for m, s in order:
                w.append_game(m, s)
        r = B.ShardReader(w.path)
        expect = [(m.game_id, s) for m, ss in kept for s in ss]
        self._check_alignment(r, expect)
        self.assertEqual(r.counters["games"], 7)

    def test_rejected_game_leaves_no_partial_rows_or_tokens(self):
        good, bad = TB.seat(3, 0, "mirror"), TB.seat(4, 1, "mirror")
        bad.tokens = [copy.deepcopy(t) for t in bad.tokens]
        bad.tokens[-1]["groups"]["HAND"]["cat"] = bad.tokens[-1]["groups"]["HAND"]["cat"].astype(np.int32)
        a = [TB.seat(0, 0)]
        c = [TB.seat(1, 0), TB.seat(2, 1, "opponent")]
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-0", a), a)
            with self.assertRaises(ValueError):
                w.append_game(TB.meta("w0-1", [good, bad]), [good, bad])
            w.append_game(TB.meta("w0-2", c, result=-1), c)
        r = B.ShardReader(w.path)
        self.assertEqual((len(r.games), r.n_seats, r.counters["games"]), (2, 3, 2))
        self._check_alignment(r, [("w0-0", a[0]), ("w0-2", c[0]), ("w0-2", c[1])])

    def test_load_update_with_a_middle_worker_of_only_dropped_games(self):
        TB.write_shard(self.root, 0)
        with _writer(self.root, 1) as w:
            for m, s, _ in TB.dropped_games(1):
                w.append_game(m, s)
        TB.write_shard(self.root, 2)
        u = B.load_update(self.root, UPDATE, expect_workers=3, expect_games=8 + 5 + 8)
        self.assertEqual(len(u.games), 6)
        self.assertEqual(len(u.seat_offsets), u.seat["game"].shape[0] + 1)
        expect = [(m.game_id, s) for wk in (0, 2) for m, ss in TB.kept_games(wk) for s in ss]
        self._check_alignment(u, expect)
        for i, sg in enumerate(u.seat_games()):
            self.assertEqual(u.games[int(u.seat["game"][i])].game_id, sg.game.game_id)
            lo, hi = int(u.seat_offsets[i]), int(u.seat_offsets[i + 1])
            np.testing.assert_array_equal(u.rows["tick"][lo:hi], sg.rows["tick"])
        self.assertEqual(u.drop_counter().total, 21)

    def test_writer_copies_its_inputs(self):
        s = TB.seat(0, 0)
        s.tokens = [copy.deepcopy(t) for t in s.tokens]
        mu0 = s.rows["mu"].copy()
        feat0 = s.tokens[2]["groups"]["ENTITY"]["feat"].copy()
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-0", [s]), [s])
            s.rows["mu"][:] = 0.0                              # a rollout reusing its buffers
            s.tokens[2]["groups"]["ENTITY"]["feat"][:] = 7.0
        sg = B.ShardReader(w.path).seat_game(0)
        np.testing.assert_array_equal(sg.rows["mu"], mu0)
        np.testing.assert_array_equal(sg.tokens()["groups"]["ENTITY"]["feat"][2:3, :feat0.shape[1]], feat0)


# --------------------------------------------------------------------------- outcomes and learner inputs
class TestOutcomesAndLearnerInputs(TmpDir):

    def test_mirror_outcome_sign_with_a_decisive_result(self):
        c = [TB.seat(3, 0, "mirror"), TB.seat(4, 1, "mirror")]
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-2", c, result="win", our_owner=1), c)
        r = B.ShardReader(w.path)
        self.assertEqual([(sg.side, sg.outcome) for sg in r.seat_games()], [(0, -1.0), (1, 1.0)])

    def test_shard_carries_every_learner_input(self):
        """GAE (ticks, lengths, terminal reward per seat), HT x seat weights, EV by phase / opponent: all from
        UpdateData alone."""
        TB.write_shard(self.root, 0)
        TB.write_shard(self.root, 1)
        u = B.load_update(self.root, UPDATE)
        lengths = u.lengths
        last = np.cumsum(lengths) - 1
        rewards = np.zeros(u.n_rows)
        rewards[last] = u.seat["outcome"]
        values = np.zeros(u.n_rows)
        res = G.gae_ragged(rewards, values, u.rows["tick"], lengths)
        np.testing.assert_allclose(res.advantages[last], u.seat["outcome"])
        pol, val = [], []
        for i, sg in enumerate(u.seat_games()):
            rw = G.row_weights(sg.rows["stratum"], sg.rows["forced"], float(u.seat["weight"][i]),
                               family=sg.rows["family"])
            pol.append(rw.policy)
            val.append(rw.value)
        pol, val = np.concatenate(pol), np.concatenate(val)
        self.assertEqual(pol.shape, (u.n_rows,))
        self.assertTrue(np.all(pol[u.rows["forced"]] == 0) and np.all(val > 0))
        seat_of_row = np.repeat(np.arange(len(lengths)), lengths)
        game_of_row = u.seat["game"][seat_of_row]
        end_ticks = np.array([u.games[g].end_tick for g in game_of_row], np.int64)
        self.assertTrue(np.all(end_ticks >= u.rows["tick"]))
        outcome_row = u.seat["outcome"][seat_of_row]
        groups = np.array([u.games[g].opp_kind for g in game_of_row])
        ev = G.explained_variance_by_phase(np.zeros(u.n_rows), outcome_row, u.rows["tick"], end_ticks,
                                           weights=val, groups=groups)
        self.assertIn("ev_by_group", ev)


# --------------------------------------------------------------------------- documented facts of the format
class TestDocFacts(TmpDir):

    def test_DOC_bytes_per_row(self):
        """Token bytes per row = fixed groups + per-ENTITY / per-OBJECT token cost (int64 cat / lnk included)."""
        path, _ = TB.write_shard(self.root, drops=False)
        r = B.ShardReader(path)
        n = r.n_rows
        per = {name: g.n_features * 4 + len(g.cats) * 8 + len(g.roles) * 8 + 1 for name, g in S.GROUPS.items()}
        fixed = sum(g.cap * per[name] for name, g in S.GROUPS.items() if not g.ragged)
        per_ent, per_obj = per["ENTITY"] + 8, per["OBJECT"] + 8          # + ent_uid / obj_uid
        w_ent, w_obj = r.arrays["tok.w.ENTITY"].mean(), r.arrays["tok.w.OBJECT"].mean()
        tok = sum(v.nbytes for k, v in r.arrays.items() if k.startswith("tok."))
        other_row = (tok - n * (fixed + w_ent * per_ent + w_obj * per_obj)) / n
        self.assertGreater(other_row, 0)
        self.assertLess(other_row, 200)                                    # scalars, slot_legal, widths ...
        file_per_row = path.stat().st_size / n
        print(f"\n[bytes/row] file {file_per_row:.0f}, tokens {tok / n:.0f} (fixed groups {fixed}, "
              f"ENTITY {per_ent}/token x {w_ent:.1f}, OBJECT {per_obj}/token x {w_obj:.1f}, other {other_row:.0f}), "
              f"rows {sum(v.nbytes for k, v in r.arrays.items() if k.startswith('row.')) / n:.0f}")
        self.assertLess(file_per_row, 1.5 * 11_000)

    def test_DOC_multi_row_token_dict_is_trimmed_per_row(self):
        """Fixed: a tokens_from_rows dict is split and each row trimmed to its last used token, so it costs
        about what the adapter's per-row dicts cost."""
        rows, rows_tok, sub, _ = TB.actor_seat(0)
        sizes = {}
        for name, tokens in (("per_row", rows_tok), ("multi_row", sub)):
            s = TB.seat(0, 0)
            s.tokens = tokens
            with _writer(self.root / name) as w:
                w.append_game(TB.meta("w0-0", [s]), [s])
            sizes[name] = w.path.stat().st_size
        print(f"\n[multi-row dict] {sizes['multi_row'] / sizes['per_row']:.2f}x the per-row size")
        self.assertLess(sizes["multi_row"], 1.02 * sizes["per_row"])

    def test_DOC_unfolded_family_row_is_valid_with_the_action_stratum(self):
        """A caller that passes the unfolded family (play) with play_blocked gets a valid row whose executed
        family is wait but whose stratum is STRATUM_ACTION (HT weight 1 instead of 1 / kappa). actor_rows still
        builds it; the writer now refuses it (test_BUG_unfolded_family_is_rejected)."""
        rows, _, _, actor = TB.actor_seat(0)
        act = {k: torch.from_numpy(rows[k].astype(np.int64) if rows[k].dtype != np.bool_ else rows[k].copy())
               for k in ("family", "slot", "cell", "ability", "play_blocked")}
        r = int(np.nonzero((rows["family"] == S.FAMILY_WAIT) & rows["legal_slots"].any(-1))[0][0])
        act["family"][r] = S.FAMILY_PLAY
        act["play_blocked"][r] = True
        new = B.actor_rows(actor, rows["dt"], act, threat=rows["threat"], ticks=rows["tick"], phi=rows["phi"],
                           query_index=rows["query_index"])
        self.assertTrue(bool(new["valid"][r]))
        self.assertEqual(int(new["executed_family"][r]), S.FAMILY_WAIT)
        self.assertEqual(int(new["stratum"][r]), S.STRATUM_ACTION)
        self.assertEqual(float(new["mu"][r]), float(rows["mu"][r]))
        self.assertEqual(float(G.ht_weight(new["stratum"][r:r + 1])[0]), 1.0)
        self.assertGreater(float(G.ht_weight(rows["stratum"][r:r + 1])[0]), 1.0)

    def test_DOC_stale_shard_of_an_earlier_attempt_is_rejected(self):
        """Fixed: a re-run of the same update number that rewrites only some workers (new
        rollout_id) no longer loads; begin_update refuses the non-empty u{N} unless told to clear it."""
        TB.write_shard(self.root, 0)
        TB.write_shard(self.root, 1)                          # attempt A
        a = [TB.seat(0, 0)]
        with _writer(self.root, 0, overwrite=True, rollout_id="rollout-u7-b") as w:   # attempt B: worker 0 only
            w.append_game(TB.meta("B-0", a), a)
        with self.assertRaisesRegex(B.SchemaError, "rollout_id.*two attempts"):
            B.load_update(self.root, UPDATE, expect_workers=2, expect_games=1 + 8)
        with self.assertRaises(FileExistsError):
            B.begin_update(self.root, UPDATE)
        d = B.begin_update(self.root, UPDATE, clear=True)
        self.assertEqual(list(d.iterdir()), [])

    def test_DOC_corrupt_shards_raise_schema_error_caused_by_badzipfile(self):
        path, _ = TB.write_shard(self.root, drops=False)
        data = bytearray(path.read_bytes())
        with zipfile.ZipFile(path) as z:
            info = z.getinfo("row.mu.npy")
        off = info.header_offset
        start = off + 30 + int.from_bytes(data[off + 26:off + 28], "little") + \
            int.from_bytes(data[off + 28:off + 30], "little")
        major = data[start + 6]
        hl = int.from_bytes(data[start + 8:start + (10 if major == 1 else 12)], "little")
        data[start + (10 if major == 1 else 12) + hl + 8] ^= 0x01            # one bit of row.mu[1]
        bad = self.root / "crc.npz"
        bad.write_bytes(bytes(data))
        with self.assertRaises(B.SchemaError) as cm:                    # fixed: SchemaError
            B.ShardReader(bad)
        self.assertIsInstance(cm.exception.__cause__, zipfile.BadZipFile)   # the CRC is still checked
        trunc = self.root / "trunc.npz"
        trunc.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
        with self.assertRaises(B.SchemaError) as cm:
            B.ShardReader(trunc)
        self.assertIsInstance(cm.exception.__cause__, zipfile.BadZipFile)


# --------------------------------------------------------------------------- defects (expectedFailure pins)
class TestDefects(TmpDir):

    def test_BUG_close_after_a_failed_write_reports_a_missing_shard(self):
        """close() failed (disk full), abort() marked the writer closed, and a retry of close() returns the
        shard path although nothing was written. A retrying coordinator would count the worker as done."""
        w = _writer(self.root)
        w.append_game(*TB.kept_games()[0])
        with mock.patch.object(B.np, "savez", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                w.close()
        try:
            p = w.close()
        except Exception:                                      # refusing to close again is a fine fix
            return
        self.assertTrue(p.exists(), f"close() returned {p} but no shard was written")

    def test_BUG_close_after_abort_reports_a_missing_shard(self):
        w = _writer(self.root)
        w.append_game(*TB.kept_games()[0])
        w.abort()
        try:
            p = w.close()
        except Exception:
            return
        self.assertTrue(p.exists(), f"close() after abort() returned {p} but no shard was written")

    def test_BUG_seat_role_must_match_our_owner(self):
        """outcome is derived from side vs our_owner; a seat 'ours' that is not on our_owner's side (or an
        'opponent' seat on it) means our_owner or the seat sides are wrong, and every outcome flips silently."""
        s = TB.seat(0, 1, "ours")
        with _writer(self.root) as w, self.assertRaises(ValueError):
            w.append_game(TB.meta("w0-0", [s], our_owner=0), [s])

    def test_BUG_unfolded_family_is_rejected(self):
        """family != executed_family on a valid row: the stored family is not the folded sample, so the stratum
        (and the HT weight) is wrong; sample_actions never produces it."""
        rows, rows_tok, _, actor = TB.actor_seat(0)
        act = {k: torch.from_numpy(rows[k].astype(np.int64) if rows[k].dtype != np.bool_ else rows[k].copy())
               for k in ("family", "slot", "cell", "ability", "play_blocked")}
        r = int(np.nonzero((rows["family"] == S.FAMILY_WAIT) & rows["legal_slots"].any(-1))[0][0])
        act["family"][r] = S.FAMILY_PLAY
        act["play_blocked"][r] = True
        new = B.actor_rows(actor, rows["dt"], act, threat=rows["threat"], ticks=rows["tick"], phi=rows["phi"],
                           query_index=rows["query_index"])
        s = make_seat(new, rows_tok, 0)
        with _writer(self.root) as w, self.assertRaises(ValueError):
            w.append_game(TB.meta("w0-0", [s]), [s])

    def test_BUG_float32_mu_is_rejected(self):
        """mu given as float32 is upcast silently; the lost bits cannot be recovered (fp64 identity 1e-9)."""
        s = TB.seat(0, 0)
        s.rows["mu"] = s.rows["mu"].astype(np.float32)
        with _writer(self.root) as w, self.assertRaises(ValueError):
            w.append_game(TB.meta("w0-0", [s]), [s])

    def test_BUG_greedy_temperature_is_rejected_for_a_trained_seat(self):
        """T = 0 means greedy (config.sampling_mode); a greedy seat has no sampling log-prob, and the learner's
        apply_temperature refuses T < T_MIN only later."""
        s = TB.seat(0, 0, temperature=(0.0, 0.0, 0.0))
        with _writer(self.root) as w, self.assertRaises(ValueError):
            w.append_game(TB.meta("w0-0", [s]), [s])

    def test_BUG_phi_end_has_no_seat_view(self):
        """Rows store Φ from each seat's own view but GameMeta.phi_end is one number with no stated view;
        short_head_gae(phi=rows, phi_end=...) on the opponent / second mirror seat needs -phi_end."""
        b = [TB.seat(1, 0, "ours"), TB.seat(2, 1, "opponent")]
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-1", b, result=-1, phi_end=0.25), b)
        sg = list(B.ShardReader(w.path).seat_games())[1]
        view = getattr(sg, "phi_end", None)
        if view is None:
            view = sg.seat.get("phi_end")
        self.assertEqual(view, -0.25)

    def test_BUG_no_acting_policy_identity(self):
        """Nothing in a shard says which weights acted (code / config digests do not), so a rollout served by
        a stale server or an earlier attempt's shard is not detectable."""
        params = inspect.signature(B.ShardWriter).parameters
        self.assertTrue(any(p in params for p in IDENTITY_PARAMS) or "policy_digest" in B.GAME_FIELDS)

    def test_BUG_load_update_rejects_shards_of_different_attempts(self):
        params = inspect.signature(B.ShardWriter).parameters
        ident = next((p for p in IDENTITY_PARAMS if p in params), None)
        kw_a = {ident: "attempt-a"} if ident else {}
        kw_b = {ident: "attempt-b"} if ident else {}
        for wk in (0, 1):
            with _writer(self.root, wk, **kw_a) as w:
                for m, s in TB.kept_games(wk):
                    w.append_game(m, s)
        a = [TB.seat(0, 0)]
        with _writer(self.root, 0, overwrite=True, **kw_b) as w:
            w.append_game(TB.meta("B-0", a), a)
        with self.assertRaises(B.SchemaError):
            B.load_update(self.root, UPDATE, expect_workers=2)

    def test_BUG_token_row_spec_unchecked_when_worker0_has_no_rows(self):
        """load_update compares each shard's token row spec with readers[0] only, and skips the comparison
        when readers[0] has no rows: w001 and w002 with different row dtypes are then never compared."""
        with _writer(self.root, 0) as w:
            for m, s, _ in TB.dropped_games(0):
                w.append_game(m, s)
        TB.write_shard(self.root, 1, drops=False)
        s = TB.seat(0, 0)
        s.tokens = [copy.deepcopy(t) for t in s.tokens]
        for t in s.tokens:
            t["row"]["seq"] = t["row"]["seq"].astype(np.int32)
        with _writer(self.root, 2) as w:
            w.append_game(TB.meta("w2-0", [s]), [s])
        with self.assertRaises(B.SchemaError):
            B.load_update(self.root, UPDATE, expect_workers=3)

    def test_BUG_corrupt_shard_raises_schema_error(self):
        """A truncated shard raises zipfile.BadZipFile, not SchemaError (callers that quarantine on
        SchemaError crash instead)."""
        path, _ = TB.write_shard(self.root, drops=False)
        trunc = self.root / "trunc.npz"
        trunc.write_bytes(path.read_bytes()[: path.stat().st_size // 2])
        with self.assertRaises(B.SchemaError):
            B.ShardReader(trunc)



# --------------------------------------------------------------------------- the other review items (fixed)
def _act_of(rows):
    return {k: torch.from_numpy(rows[k].astype(np.int64) if rows[k].dtype != np.bool_ else rows[k].copy())
            for k in ("family", "slot", "cell", "ability", "play_blocked")}


class TestFixes(TmpDir):

    def test_t_min_matches_logprob(self):
        self.assertEqual(B.T_MIN, LP.T_MIN)

    def test_identity_is_required_and_stamped(self):
        with self.assertRaises(TypeError):
            B.ShardWriter(self.root, UPDATE, 0, code_digest=CODE, config_digest=CONF)
        for bad in ({"policy_digest": ""}, {"rollout_id": ""}):
            with self.assertRaises(ValueError):
                _writer(self.root, **bad)
        s = [TB.seat(0, 0)]
        with _writer(self.root) as w:
            with self.assertRaisesRegex(ValueError, "policy_digest"):
                w.append_game(TB.meta("w0-0", s, policy_digest="pol-other"), s)
            self.assertIsNone(w.append_game(TB.meta("w0-1", s, policy_digest=POL), s))
        g = B.ShardReader(w.path).games[0]
        self.assertEqual((g.policy_digest, g.rollout_id), (POL, RID))

    def test_load_update_identity_expectations(self):
        TB.write_shard(self.root, 0)
        TB.write_shard(self.root, 1)
        u = B.load_update(self.root, UPDATE, expect_policy_digest=POL, expect_rollout_id=RID,
                          expect_code_digest=CODE, expect_config_digest=CONF)
        self.assertEqual((u.policy_digest, u.rollout_id), (POL, RID))
        for kw in ({"expect_policy_digest": "pol-stale"}, {"expect_rollout_id": "rollout-u7-z"},
                   {"expect_code_digest": "code-x"}, {"expect_config_digest": "conf-x"}):
            with self.assertRaises(B.SchemaError):
                B.load_update(self.root, UPDATE, **kw)

    def test_seat_roles_must_fit_our_owner_and_opp_kind(self):
        ok = [(0, "il_v4_t1", [0], ["ours"]), (1, "self_seat", [1, 0], ["ours", "opponent"]),
              (0, "self_seat", [0], ["ours"]), (1, "mirror", [0, 1], ["mirror", "mirror"])]
        for args in ok:
            B.check_seat_roles(*args)
        bad = [(0, "self_seat", [0, 1], ["opponent", "ours"]),     # opponent on our_owner's side
               (0, "il_v4_t1", [0, 1], ["ours", "opponent"]),       # theta on a frozen opponent's seat
               (0, "self_seat", [0, 1], ["ours", "mirror"]),
               (0, "mirror", [0, 1], ["ours", "mirror"]),
               (0, "self_seat", [1], ["opponent"])]                 # a non-mirror game always trains ours
        for args in bad:
            with self.assertRaises(ValueError, msg=str(args)):
                B.check_seat_roles(*args)
        b = [TB.seat(1, 0, "ours"), TB.seat(2, 1, "opponent")]
        with _writer(self.root) as w, self.assertRaisesRegex(ValueError, "opponent"):
            w.append_game(TB.meta("w0-0", b, opp_kind="snapshot"), b)

    def test_seat_weights_table(self):
        table = {"ours": 1.0, "opponent": 0.25, "mirror": 1.0}            # == gae.SEAT_WEIGHT
        b = [TB.seat(1, 0, "ours"), TB.seat(2, 1, "opponent")]
        with _writer(self.root, seat_weights=table) as w:
            self.assertIsNone(w.append_game(TB.meta("w0-0", b), b))
            s = [TB.seat(0, 0, "ours", weight=0.7)]
            with self.assertRaisesRegex(ValueError, "table"):
                w.append_game(TB.meta("w0-1", s), s)

    def test_phi_end_per_seat_view_and_positive_zero_draw(self):
        TB.write_shard(self.root, 0, drops=False)
        u = B.load_update(self.root, UPDATE)
        got = u.seat_phi_end()
        # w0-0 ours@0 (owner 0); w0-1 ours@0, opponent@1 (owner 0); w0-2 mirror@0, mirror@1 (owner 1)
        np.testing.assert_array_equal(got, [0.25, 0.25, -0.25, -0.25, 0.25])
        self.assertEqual([sg.phi_end for sg in u.seat_games()], got.tolist())
        o = u.seat["outcome"]
        self.assertEqual(int((o == 0).sum()), 2)
        self.assertFalse(bool(np.signbit(o[o == 0]).any()))                    # a draw is +0.0 on both seats
        sg = list(u.seat_games())[2]
        res = G.short_head_gae(sg.rows["phi"], np.zeros(sg.n), sg.rows["tick"], phi_end=sg.phi_end)
        self.assertEqual(res.advantages.shape, (sg.n,))

    def test_phi_log_records_kickoff_and_problems(self):
        s = [TB.seat(0, 0)]
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-0", s), s, phi_log=[0.0, 0.1, -0.2, 0.25])
            s2 = [TB.seat(1, 0)]
            w.append_game(TB.meta("w0-1", s2), s2, phi_log=[0.3, 0.1])
        g0, g1 = B.ShardReader(w.path).games
        self.assertEqual((g0.phi_kickoff, g0.phi_problems), (0.0, []))
        self.assertEqual(g1.phi_kickoff, 0.3)
        self.assertTrue(any("kick-off" in x for x in g1.phi_problems))

    def test_counters_by_role_and_dropped_seat_counters(self):
        TB.write_shard(self.root, 0)
        TB.write_shard(self.root, 1)
        u = B.load_update(self.root, UPDATE)
        seats = [s for wk in (0, 1) for _, ss in TB.kept_games(wk) for s in ss] + \
            [s for wk in (0, 1) for _, ss, _ in TB.dropped_games(wk) for s in ss]
        for role in B.ROLES:
            for k in B.COUNTER_KEYS + ("plays",):
                self.assertEqual(u.counters["by_role"][role][k], sum(s.counters[k] for s in seats if s.role == role),
                                 (role, k))
        n_dropped_seats = sum(len(ss) for wk in (0, 1) for _, ss, _ in TB.dropped_games(wk))
        self.assertEqual(len(u.dropped_seats), n_dropped_seats)
        for ds in u.dropped_seats:                                   # re-indexed into the update's dropped list
            g = u.dropped[ds["dropped"]]
            self.assertIn(ds["role"], B.ROLES)
            self.assertTrue(g.game_id.startswith(("w0-", "w1-")))
        self.assertEqual(u.dropped[u.dropped_seats[-1]["dropped"]].game_id, "w1-14")

    def test_card_field(self):
        rows, rows_tok, _, actor = TB.actor_seat(0)
        n = len(rows["tick"])
        cards = 26000000 + np.arange(n)
        new = B.actor_rows(actor, rows["dt"], _act_of(rows), threat=rows["threat"], ticks=rows["tick"],
                           phi=rows["phi"], query_index=rows["query_index"], card=cards)
        play = new["executed_family"] == S.FAMILY_PLAY
        self.assertTrue(bool(play.any()))
        np.testing.assert_array_equal(new["card"], np.where(play, cards, -1))
        np.testing.assert_array_equal(rows["card"], -1)               # unknown by default
        s = make_seat(new, rows_tok, 0)
        with _writer(self.root) as w:
            w.append_game(TB.meta("w0-0", [s]), [s])
            bad = make_seat(new, rows_tok, 0)
            bad.rows["card"][int(np.nonzero(~play)[0][0])] = 5
            with self.assertRaisesRegex(ValueError, "card"):
                w.append_game(TB.meta("w0-1", [bad]), [bad])
        np.testing.assert_array_equal(B.ShardReader(w.path).rows["card"], new["card"])

    def test_row_invariants(self):
        base = TB.seat(0, 0)
        r = int(np.nonzero(base.rows["valid"] & (base.rows["family"] == S.FAMILY_PLAY))[0][0])
        cases = []
        s = TB.seat(0, 0); s.rows["mu"][r] = 0.5; cases.append((s, "<= 0"))                       # noqa: E702
        s = TB.seat(0, 0); s.rows["mu"][r] -= 1e-3; cases.append((s, "sum"))                      # noqa: E702
        s = TB.seat(0, 0); s.rows["forced"][r] = ~s.rows["forced"][r]; cases.append((s, "forced"))  # noqa: E702
        s = TB.seat(0, 0); s.rows["possible"][r, 0] = False; cases.append((s, "possible"))        # noqa: E702
        s = TB.seat(0, 0); s.rows["slot"][r] = -1; cases.append((s, "slot"))                      # noqa: E702
        with _writer(self.root) as w:
            for i, (s, msg) in enumerate(cases):
                with self.assertRaisesRegex(ValueError, msg):
                    w.append_game(TB.meta(f"w0-{i}", [s]), [s])
            self.assertEqual(w.counters["games"], 0)

    def test_type_gaps(self):
        s = [TB.seat(0, 0)]
        with _writer(self.root) as w:
            with self.assertRaisesRegex(ValueError, "bool"):
                w.append_game(TB.meta("w0-0", s, result=True), s)
            f = TB.seat(0, 0)
            f.counters = dict(f.counters, plays=2.5)
            with self.assertRaisesRegex(ValueError, "plays"):
                w.append_game(TB.meta("w0-1", [f]), [f])
            f = TB.seat(0, 0)
            f.counters = dict(f.counters, by_role=1)
            with self.assertRaisesRegex(ValueError, "maintained"):
                w.append_game(TB.meta("w0-2", [f]), [f])
            with self.assertRaises(ValueError):
                w.add_counters(refusals=1.5)

    def test_reader_semantic_validation(self):
        def tamper(fn, msg, name="w000.npz"):
            root = self.root / f"t{tamper.i}"
            tamper.i += 1
            path, _ = TB.write_shard(root, drops=False)
            TB.rewrite(path, fn)
            if name != path.name:
                path = path.rename(path.with_name(name))
            with self.assertRaisesRegex(B.SchemaError, msg):
                B.ShardReader(path)
        tamper.i = 0

        def flip_outcome(a, h):
            o = a["seat.outcome"].copy()
            o[1] = -o[1]
            return dict(a, **{"seat.outcome": o}), h

        def bad_stratum(a, h):
            st = a["row.stratum"].copy()
            st[0] = (st[0] + 1) % 3
            return dict(a, **{"row.stratum": st}), h

        def bad_committed(a, h):
            c = a["seat.counters"].copy()
            c[0, B.COUNTER_KEYS.index("committed")] += 1
            return dict(a, **{"seat.counters": c}), h

        def bad_seat_meta(a, h):
            seats = [dict(x) for x in h["seats"]]
            seats[2]["weight"] = 1.0
            return a, dict(h, seats=seats)

        def bad_roles(a, h):
            games = [dict(g) for g in h["games"]]
            games[1]["our_owner"] = 1                               # ours@0 / opponent@1 no longer fit
            return a, dict(h, games=games)

        def bad_ticks(a, h):
            t = a["row.tick"].copy()
            t[1] = t[0]
            return dict(a, **{"row.tick": t}), h

        tamper(flip_outcome, "outcome")
        tamper(bad_stratum, "stratum")
        tamper(bad_committed, "committed")
        tamper(bad_seat_meta, "header seat 2")
        tamper(bad_roles, "outcome|ours|opponent")
        tamper(bad_ticks, "tick")
        tamper(lambda a, h: (a, h), "worker", name="w005.npz")

    def test_no_clobber_between_concurrent_writers(self):
        a, b = _writer(self.root), _writer(self.root)               # e.g. an orphan of a killed attempt
        for w in (a, b):
            w.append_game(*TB.kept_games()[0])
        a.close()
        with self.assertRaises(FileExistsError):
            b.close()
        c = _writer(self.root, 1)
        c.append_game(*TB.kept_games()[0])
        _writer(self.root, 1).close()                                # the other process publishes first ...
        with mock.patch.object(type(c.path), "exists", return_value=False):   # ... after c's existence check
            with self.assertRaisesRegex(FileExistsError, "concurrently"):
                c.close()
        self.assertEqual(sorted(p.name for p in B.update_dir(self.root, UPDATE).iterdir()),
                         ["w000.npz", "w001.npz"])
        self.assertEqual(B.ShardReader(a.path).n_seats, 1)
        self.assertEqual(B.ShardReader(c.path).n_seats, 0)          # the first publisher's shard is intact

    def test_abort_after_close_keeps_the_shard_and_root_is_fsynced(self):
        calls = []
        real = B._fsync_dir
        with mock.patch.object(B, "_fsync_dir", side_effect=lambda d: (calls.append(Path(d)), real(d))):
            with _writer(self.root) as w:
                w.append_game(*TB.kept_games()[0])
        self.assertIn(self.root, calls)                               # the new u{N} entry
        self.assertIn(B.update_dir(self.root, UPDATE), calls)
        w.abort()
        self.assertTrue(w.path.exists() and w.closed and w.written)


if __name__ == "__main__":
    unittest.main()
