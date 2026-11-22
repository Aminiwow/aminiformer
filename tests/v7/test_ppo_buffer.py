"""Tests for the PPO v7 rollout shards (src/amini/ppo/buffer.py).

Synthetic seat-games are built the way the rollout actor builds them: per-row tokens (one committed row per
dict, ragged groups cut to the row's own width as ``SeatAdapter.make_row`` produces them), the tiny fp64
model's ``peek`` + ``commit`` row by row, ``sample_actions``, then ``buffer.actor_rows`` (which calls
``logprob.executed_logp_terms``). CPU only, deterministic seeds, no game environment.

The episodes come from the private dataset (tokenizer, shards) and the game records use the unpublished case
builder: the tests that need them are skipped.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import unittest
import warnings
from functools import lru_cache
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # tests/v7 (common)
import common  # noqa: E402
from common import ROOT, S, example, tiny_model  # noqa: E402

TK = getattr(common, "TK", None)                         # the tokenizer (not published)
sys.path.insert(0, str(ROOT / "tools"))
try:
    from ppo_v7 import cases as CS  # noqa: E402
except ImportError:                                        # not published
    CS = None

from amini.nn.aminiformer import sample_actions  # noqa: E402
from amini.ppo import buffer as B  # noqa: E402
from amini.ppo import gae as G  # noqa: E402
from amini.ppo import logprob as LP  # noqa: E402

torch.set_num_threads(2)

UPDATE = 7
CODE = "code-" + "a" * 16
CONF = "conf-" + "b" * 16
POL = "pol-" + "c" * 16                                    # sha256 of the served model_ema (stand-in)
RID = "rollout-u7-a"                                       # one id per attempt of an update
RAGGED_ROW_KEYS = {"ent_uid": "ENTITY", "obj_uid": "OBJECT"}


# --------------------------------------------------------------------------- fixture: actor-side seat-games
def _fill(key):
    return {"feat": 0.0, "cat": 0, "lnk": S.LINK_NONE, "valid": False}[key]


def _row_tokens(tok, t):
    """Row ``t`` of a multi-row token dict as the adapter's one-row dict, ragged groups cut to the last
    non-fill slot (at least 1)."""
    groups, widths = {}, {}
    for name, grp in tok["groups"].items():
        sl = {k: v[t:t + 1] for k, v in grp.items()}
        if S.GROUPS[name].ragged:
            used = sl["valid"][0].copy()
            for k in ("feat", "cat", "lnk"):
                a = sl[k][0]
                used |= (a != _fill(k)).reshape(a.shape[0], -1).any(-1)
            w = max(1, int(np.nonzero(used)[0].max()) + 1 if used.any() else 1)
            sl = {k: v[:, :w].copy() for k, v in sl.items()}
            widths[name] = w
        groups[name] = {k: np.ascontiguousarray(v) for k, v in sl.items()}
    row = {}
    for k, v in tok["row"].items():
        a = np.asarray(v)[t:t + 1]
        if k in RAGGED_ROW_KEYS:
            w = widths[RAGGED_ROW_KEYS[k]]
            assert np.all(a[:, w:] == -1), f"{k} has entries past the group width"
            a = a[:, :w]
        row[k] = np.ascontiguousarray(a)
    return {"groups": groups, "row": row, "profile": tok.get("profile", "engine")}


def _peek_input(rt):
    conv = (lambda a: torch.from_numpy(a).double() if a.dtype.kind == "f" else torch.from_numpy(a))
    return ({"groups": {n: {k: conv(v) for k, v in g.items()} for n, g in rt["groups"].items()},
             "row": {k: conv(rt["row"][k]) for k in ("slot_legal", "ability_ready", "cell_mask_id")}},
            torch.from_numpy(rt["row"]["pos"]).double())


def _unit_rate_dt(out):
    lf = out["family"].double()
    odds = torch.exp(lf[:, 1] - lf[:, 0]) + torch.where(out["ability_avail"].bool(),
                                                        torch.exp(lf[:, 2] - lf[:, 0]),
                                                        torch.zeros(lf.shape[0], dtype=torch.float64))
    return S.U_TICKS / odds


@lru_cache(maxsize=1)
def model():
    return tiny_model(0, double=True, cell_mask=True)


@lru_cache(maxsize=None)
def actor_seat(i):
    """(rows, per-row tokens, multi-row tokens of the kept rows, actor outputs) of synthetic episode i."""
    e, _ = example(i)
    tok = e["tokens"]
    ticks = np.asarray(tok["row"]["tick"], np.int64)
    idx = np.nonzero(np.r_[True, ticks[1:] > ticks[:-1]])[0]     # one query per tick (strictly increasing)
    rows_tok = [_row_tokens(tok, int(t)) for t in idx]
    m = model()
    state = m.initial_state(1, dtype=torch.float64)
    outs = []
    for rt in rows_tok:
        tt, pos = _peek_input(rt)
        outs.append(m.peek(tt, pos, state, ticks=torch.from_numpy(rt["row"]["tick"])))
        m.commit(state, torch.tensor([True]))
    actor = B.stack_peek_outputs(outs)
    rng = np.random.default_rng(100 + i)
    n = len(idx)
    dt = (_unit_rate_dt(actor) * torch.from_numpy(rng.uniform(0.5, 2.0, n))).numpy()
    act = sample_actions(actor, torch.from_numpy(dt), generator=torch.Generator().manual_seed(7 + i))
    rows = B.actor_rows(actor, dt, act, threat=rng.random(n) < 0.3, ticks=ticks[idx],
                        phi=rng.uniform(-0.6, 0.6, n), query_index=np.arange(n) * 3 + 1)
    sub = {"groups": {g: {k: v[idx] for k, v in grp.items()} for g, grp in tok["groups"].items()},
           "row": {k: np.asarray(v)[idx] for k, v in tok["row"].items()}}
    return rows, rows_tok, sub, actor


def seat(i, side, role="ours", weight=None, **kw):
    rows, rows_tok, _, _ = actor_seat(i)
    n = len(rows["tick"])
    counters = {"queries": 3 * n + 2, "committed": n, "forced": int(rows["forced"].sum()) + 1, "skipped": 2,
                "plays": int((rows["family"] == S.FAMILY_PLAY).sum())}
    w = G.seat_weight(role) if weight is None else weight
    return B.SeatRecord(side=side, role=role, weight=w, rows={k: v.copy() for k, v in rows.items()},
                        tokens=rows_tok, counters=counters, **kw)


def kind_of(seats):
    """The opponent kind the seat roles imply (check_seat_roles): mirror, self_seat or a frozen IL kind."""
    roles = {s.role for s in seats}
    return "mirror" if "mirror" in roles else "self_seat" if "opponent" in roles else "il_v4_t1"


def writer(root, worker=0, *, update=UPDATE, **kw):
    for k, v in (("code_digest", CODE), ("config_digest", CONF), ("policy_digest", POL), ("rollout_id", RID)):
        kw.setdefault(k, v)
    return B.ShardWriter(root, update, worker, **kw)


def meta(gid, seats, *, result="win", end="native", our_owner=0, **kw):
    if CS is None:
        raise unittest.SkipTest("needs an unpublished module (ppo_v7.cases)")
    end_tick = max(int(s.rows["tick"][-1]) for s in seats) + 11 if seats else 100
    spec = CS.OpponentSpec(name="il_v4", checkpoint="/x/v4.pt", temperature=1.0, hazard_dt="block",
                           code_tree="run")
    d = dict(game_id=gid, update=UPDATE, replica=int(gid.split("-")[-1]), case_id=f"case-{gid}", level=16,
             our_owner=our_owner, our_deck=(26000000, 26000001, 26000002), opp_kind=kind_of(seats), opp_settings=spec,
             seeds=CS.game_seeds(11, UPDATE, int(gid.split("-")[-1])),
             tower_troops={"ours": 159000000, "opp": 159000001}, hazard_dt={"ours": "block", "opp": "block"},
             code_digest=CODE, config_digest=CONF, end=end, result=result, end_tick=end_tick, phi_end=0.25,
             extra={"opp_deck_source": "case", "needs_synthetic": False})
    d.update(kw)
    return B.GameMeta(**d)


def kept_games(worker=0):
    """Three kept games: ours vs frozen IL, theta on the opponent seat, mirror."""
    a = [seat(0, 0, "ours")]
    b = [seat(1, 0, "ours"), seat(2, 1, "opponent", model="main", temperature=(1.0, 1.0, 1.0))]
    c = [seat(3, 0, "mirror"), seat(4, 1, "mirror")]
    return [(meta(f"w{worker}-0", a, result="win"), a),
            (meta(f"w{worker}-1", b, result=-1), b),
            (meta(f"w{worker}-2", c, result="draw", our_owner=1), c)]


def dropped_games(worker=0):
    """(meta, seats, expected reason) of every drop path."""
    out = []
    s = [seat(5, 0)]
    out.append((meta(f"w{worker}-10", s, end="crash", result=None, phi_drop=-0.3), s, "crash"))
    s = [seat(0, 0)]
    out.append((meta(f"w{worker}-11", s, drop_reason="step_guard", phi_drop=0.1), s, "step_guard"))
    s = [seat(1, 0)]
    s[0].rows["halted"][-3:] = True
    out.append((meta(f"w{worker}-12", s), s, "seat_stall"))
    s = [seat(2, 0)]
    out.append((meta(f"w{worker}-13", s, result=None), s, "invalid"))           # regular end, no outcome
    s = [seat(3, 0), seat(4, 1, "opponent")]
    s[1].rows["valid"][5] = False
    out.append((meta(f"w{worker}-14", s), s, "invalid"))                       # a non-executable row
    return out


def write_shard(root, worker=0, *, drops=True, compress=False, update=UPDATE):
    with writer(root, worker, update=update, compress=compress) as w:
        reasons = [w.append_game(m, s) for m, s in kept_games(worker)]
        if drops:
            reasons += [w.append_game(m, s) for m, s, _ in dropped_games(worker)]
        w.add_counters(refusals_code19=worker + 1)
    return w.path, reasons


def rewrite(path, fn):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    head = json.loads(arrays["header"].tobytes().decode())
    arrays, head = fn(arrays, head)
    arrays["header"] = np.frombuffer(json.dumps(head).encode(), np.uint8)
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)


def to_torch64(batch):
    def conv(v):
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(v))
            return t.double() if t.is_floating_point() else t
        return v
    return conv(batch)


class TmpDir(unittest.TestCase):
    def setUp(self):
        base = os.environ.get("TMPDIR") or None
        self.tmp = tempfile.TemporaryDirectory(prefix="ppo-buffer-", dir=base)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


# --------------------------------------------------------------------------- round trip
class TestRoundTrip(TmpDir):

    def test_fixture_covers_every_family_and_both_strata(self):
        fam = np.concatenate([actor_seat(i)[0]["family"] for i in range(5)])
        st = np.concatenate([actor_seat(i)[0]["stratum"] for i in range(5)])
        for f in (S.FAMILY_WAIT, S.FAMILY_PLAY, S.FAMILY_ABILITY):
            self.assertTrue(bool((fam == f).any()), f"fixture lacks family {f}")
        self.assertEqual(set(np.unique(st)), {S.STRATUM_ACTION, S.STRATUM_BASE, S.STRATUM_THREAT})
        widths = {rt["groups"]["ENTITY"]["valid"].shape[1] for i in range(5) for rt in actor_seat(i)[1]}
        self.assertGreater(len(widths), 1, "ragged widths must vary between rows")

    def test_rows_meta_and_counters_bitwise(self):
        path, reasons = write_shard(self.root)
        self.assertEqual(reasons, [None] * 3 + [r for _, _, r in dropped_games()])
        r = B.ShardReader(path)
        self.assertEqual((r.header["schema"], r.header["schema_digest"]), (B.SCHEMA_VERSION, B.SCHEMA_DIGEST))
        self.assertEqual((len(r.games), len(r.dropped), r.n_seats), (3, 5, 5))
        expect = [s for _, ss in kept_games() for s in ss]
        outcome = [1.0, -1.0, 1.0, 0.0, 0.0]
        for sg, s, o in zip(r.seat_games(), expect, outcome):
            for k, (dt, tail) in B.ROW_FIELDS.items():
                self.assertEqual(sg.rows[k].dtype, np.dtype(dt), k)
                self.assertEqual(sg.rows[k].shape, (len(s.rows["tick"]),) + tail, k)
                np.testing.assert_array_equal(sg.rows[k], np.asarray(s.rows[k]).astype(dt), err_msg=k)
            self.assertEqual((sg.side, sg.role, sg.weight, sg.outcome), (s.side, s.role, s.weight, o))
            self.assertEqual(sg.seat["counters"], dict(s.counters))
            self.assertEqual(sg.seat["model"], s.model)
        for g, (m, _) in zip(r.games, kept_games()):                # the writer stamps the acting identity
            self.assertEqual(g.to_dict(), dataclasses.replace(m, policy_digest=POL, rollout_id=RID).to_dict())
        self.assertEqual((r.header["policy_digest"], r.header["rollout_id"]), (POL, RID))
        self.assertEqual(r.games[0].opp_settings["gate_bias"], [[0.0] * 3, [0.0] * 3])
        self.assertEqual(r.games[1].result, -1.0)
        c = r.counters
        all_seats = [s for _, ss in kept_games() for s in ss] + [s for _, ss, _ in dropped_games() for s in ss]
        for k in B.COUNTER_KEYS + ("plays",):
            self.assertEqual(c[k], sum(s.counters[k] for s in all_seats), k)
        self.assertEqual((c["games"], c["games_kept"], c["games_dropped"]), (8, 3, 5))
        self.assertEqual(c["rows_kept"], r.n_rows)
        self.assertEqual(c["drop_by_reason"], {"crash": 1, "step_guard": 1, "seat_stall": 1, "invalid": 2})
        self.assertEqual(c["refusals_code19"], 1)

    def test_tokens_rebuild_bitwise(self):
        path, _ = write_shard(self.root, drops=False)
        r = B.ShardReader(path)
        for sg, i in zip(r.seat_games(), (0, 1, 2, 3, 4)):
            _, rows_tok, sub, _ = actor_seat(i)
            tok = sg.tokens()
            self.assertEqual(tok["profile"], "engine")
            self.assertEqual(set(tok["row"]), set(rows_tok[0]["row"]))
            for t, rt in enumerate(rows_tok):                    # every row equals the actor's own input
                for name, grp in rt["groups"].items():
                    for k, v in grp.items():
                        w = v.shape[1]
                        got = tok["groups"][name][k][t:t + 1]
                        self.assertEqual(got.dtype, v.dtype)
                        np.testing.assert_array_equal(got[:, :w], v)
                        self.assertTrue(np.all(got[:, w:] == _fill(k)))
                for k, v in rt["row"].items():
                    got = tok["row"][k][t:t + 1]
                    self.assertEqual(got.dtype, v.dtype)
                    np.testing.assert_array_equal(got[:, :v.shape[1]] if v.ndim > 1 else got, v)
            for name, grp in sub["groups"].items():              # = tokens_from_rows of the episode, cropped
                for k, v in grp.items():
                    got = tok["groups"][name][k]
                    w = got.shape[1]
                    np.testing.assert_array_equal(got, v[:, :w])
                    self.assertTrue(np.all(v[:, w:] == _fill(k)))

    @unittest.skipIf(TK is None, common.SKIP_REASON)
    def test_learner_forward_from_shard_matches_mu(self):
        path, _ = write_shard(self.root, drops=False)
        r = B.ShardReader(path)
        m = copy.deepcopy(model()).train()                       # dropout 0: train mode == eval mode
        worst = 0.0
        for sg in r.seat_games():
            b = to_torch64(TK.collate([sg.example()]))
            n = sg.n
            self.assertEqual(int(b["row"]["row_valid"].sum()), n)
            a = {k: torch.from_numpy(v.astype(np.int64) if v.dtype != np.bool_ else v)
                 for k, v in sg.action().items()}
            play = a["family"] == S.FAMILY_PLAY
            cell_rows = torch.zeros_like(b["row"]["row_valid"])
            play_slot = torch.full(cell_rows.shape, -1, dtype=torch.long)
            cell_rows[0, :n] = play
            play_slot[0, :n] = torch.where(play, a["slot"], torch.full_like(a["slot"], -1))
            with torch.no_grad():
                raw = m(b, cell_rows=cell_rows, play_slot=play_slot, with_aux=False)
            terms = LP.executed_logp_terms(raw, torch.from_numpy(sg.rows["dt"]), a)
            self.assertTrue(bool(terms["valid"].all()))
            worst = max(worst, float((terms["total"] - torch.from_numpy(sg.rows["mu"])).abs().max()))
            np.testing.assert_array_equal(LP.forced_mask(raw).numpy(), sg.rows["forced"])
            self.assertEqual(B.mask_recompute_mismatch(sg), 0)
            rw = G.row_weights(sg.rows["stratum"], sg.rows["forced"], sg.weight, family=sg.rows["family"])
            self.assertTrue(np.all(rw.policy[sg.rows["forced"]] == 0))
        self.assertLess(worst, 1e-9)

    def test_compressed_round_trip(self):
        p1, _ = write_shard(self.root / "a")
        p2, _ = write_shard(self.root / "b", compress=True)
        r1, r2 = B.ShardReader(p1), B.ShardReader(p2)
        self.assertEqual(set(r1.arrays), set(r2.arrays))
        for k in r1.arrays:
            if k != "header":
                np.testing.assert_array_equal(r1.arrays[k], r2.arrays[k], err_msg=k)
        self.assertLess(p2.stat().st_size, p1.stat().st_size)

    def test_load_update_concatenates_workers(self):
        p0, _ = write_shard(self.root, 0)
        p1, _ = write_shard(self.root, 1)
        u = B.load_update(self.root, UPDATE, expect_workers=2, expect_games=16)
        r0, r1 = B.ShardReader(p0), B.ShardReader(p1)
        self.assertEqual(u.n_rows, r0.n_rows + r1.n_rows)
        self.assertEqual(len(u.games), 6)
        np.testing.assert_array_equal(u.lengths, np.r_[np.diff(r0.offsets), np.diff(r1.offsets)])
        for k in B.ROW_FIELDS:
            np.testing.assert_array_equal(u.rows[k], np.concatenate([r0.rows[k], r1.rows[k]]))
            self.assertEqual(u.rows[k].dtype, np.dtype(B.ROW_FIELDS[k][0]))
        sgs = list(u.seat_games())
        self.assertEqual(len(sgs), 10)
        for i, sg in enumerate(sgs):                              # global seat -> global game, global rows
            self.assertEqual(u.games[int(u.seat["game"][i])].game_id, sg.game.game_id)
            lo, hi = u.seat_offsets[i], u.seat_offsets[i + 1]
            np.testing.assert_array_equal(u.rows["mu"][lo:hi], sg.rows["mu"])
        self.assertEqual(u.counters["refusals_code19"], 1 + 2)
        self.assertEqual(u.counters["drop_by_reason"]["invalid"], 4)
        dc = u.drop_counter(alarm_rate=0.005, stop_rate=0.02)
        self.assertEqual((dc.kept, dc.dropped, dc.level()), (6, 10, "stop"))
        self.assertEqual(dc.phi_by_reason["crash"], [-0.3, -0.3])
        res = G.gae_ragged(np.zeros(u.n_rows), np.zeros(u.n_rows), u.rows["tick"], u.lengths)
        self.assertEqual(res.advantages.shape, (u.n_rows,))


# --------------------------------------------------------------------------- dropped games
class TestDropped(TmpDir):

    def test_every_drop_path_is_excluded_and_counted(self):
        path, _ = write_shard(self.root)
        r = B.ShardReader(path)
        kept_ids = {sg.game.game_id for sg in r.seat_games()}
        self.assertEqual(kept_ids, {"w0-0", "w0-1", "w0-2"})
        self.assertEqual([g.drop_reason for g in r.dropped], [x for _, _, x in dropped_games()])
        self.assertEqual([g.phi_drop for g in r.dropped][:2], [-0.3, 0.1])
        self.assertTrue(all(g.drop_reason is None for g in r.games))
        n_kept = sum(len(s.rows["tick"]) for _, ss in kept_games() for s in ss)
        self.assertEqual(r.n_rows, n_kept)

    def test_shard_of_only_dropped_games(self):
        with writer(self.root, 3) as w:
            for m, s, _ in dropped_games():
                w.append_game(m, s)
        r = B.ShardReader(w.path)
        self.assertEqual((r.n_rows, r.n_seats, len(r.dropped)), (0, 0, 5))
        self.assertFalse(any(k.startswith("tok.") for k in r.arrays))
        self.assertEqual(list(r.seat_games()), [])
        u = B.load_update(self.root, UPDATE, expect_games=5)
        self.assertEqual(u.n_rows, 0)

    def test_short_end_tick_and_empty_seat_are_invalid(self):
        s = [seat(0, 0)]
        m = meta("w0-20", s)
        m.end_tick = int(s[0].rows["tick"][-1]) - 1
        empty = seat(1, 0)
        empty.rows = {k: v[:0] for k, v in empty.rows.items()}
        empty.tokens = []
        empty.counters = dict(empty.counters, committed=0)
        with writer(self.root, 0) as w:
            self.assertEqual(w.append_game(m, s), "invalid")
            self.assertEqual(w.append_game(meta("w0-21", [seat(1, 0)]), [empty]), "invalid")


# --------------------------------------------------------------------------- atomic write
class TestAtomicWrite(TmpDir):

    def _writer(self, worker=0, **kw):
        return writer(self.root, worker, **kw)

    def _files(self):
        d = B.update_dir(self.root, UPDATE)
        return sorted(p.name for p in d.iterdir()) if d.is_dir() else []

    def test_nothing_visible_until_close(self):
        w = self._writer()
        for m, s in kept_games():
            w.append_game(m, s)
        self.assertEqual(self._files(), [])
        self.assertEqual(w.close(), w.path)
        self.assertEqual(self._files(), ["w000.npz"])
        self.assertEqual(w.close(), w.path)                       # idempotent
        with self.assertRaises(RuntimeError):
            w.append_game(*kept_games()[0])

    def test_exception_inside_with_leaves_nothing(self):
        with self.assertRaises(KeyError):
            with self._writer() as w:
                m, s = kept_games()[0]
                w.append_game(m, s)
                raise KeyError("coordinator died")
        self.assertEqual(self._files(), [])

    def test_failed_write_removes_the_temporary_file(self):
        w = self._writer()
        w.append_game(*kept_games()[0])
        with mock.patch.object(B.np, "savez", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                w.close()
        self.assertEqual(self._files(), [])
        with mock.patch.object(B.os, "link", side_effect=OSError("link failed")):       # no-clobber publish
            w2 = self._writer(1)
            w2.append_game(*kept_games()[0])
            with self.assertRaises(OSError):
                w2.close()
        with mock.patch.object(B.os, "replace", side_effect=OSError("rename failed")):    # overwrite publish
            w3 = self._writer(2, overwrite=True)
            w3.append_game(*kept_games()[0])
            with self.assertRaises(OSError):
                w3.close()
        self.assertEqual(self._files(), [])

    def test_no_overwrite_and_stray_temporaries_ignored(self):
        path, _ = write_shard(self.root, drops=False)
        with self.assertRaises(FileExistsError):
            self._writer()
        before = path.read_bytes()
        stray = path.parent / ".w001.npz.tmp-123-deadbeef"
        stray.write_bytes(b"partial")
        self.assertEqual(B.shard_paths(self.root, UPDATE), [path])
        B.load_update(self.root, UPDATE, expect_workers=1, expect_games=3)
        with self._writer(overwrite=True) as w:
            w.append_game(*kept_games()[0])
        self.assertNotEqual(path.read_bytes(), before)
        self.assertEqual(B.ShardReader(path).n_seats, 1)


# --------------------------------------------------------------------------- schema
class TestSchema(TmpDir):

    def setUp(self):
        super().setUp()
        self.path, _ = write_shard(self.root, drops=False)

    def _bad(self, fn, msg):
        rewrite(self.path, fn)
        with self.assertRaisesRegex(B.SchemaError, msg):
            B.ShardReader(self.path)

    def test_digest_is_the_spec_hash(self):
        spec = json.dumps(B.schema_spec(), sort_keys=True, separators=(",", ":"))
        self.assertEqual(B.SCHEMA_DIGEST, hashlib.sha256(spec.encode()).hexdigest())

    def test_version_mismatch(self):
        self._bad(lambda a, h: (a, dict(h, schema="ppo-v7-shard.0")), "schema")

    def test_digest_mismatch(self):
        self._bad(lambda a, h: (a, dict(h, schema_digest="0" * 64)), "digest")

    def test_dtype_mismatch(self):
        self._bad(lambda a, h: (dict(a, **{"row.mu": a["row.mu"].astype(np.float32)}), h), "row.mu: dtype")

    def test_token_dtype_mismatch(self):
        k = "tok.g.ENTITY.feat"
        self._bad(lambda a, h: (dict(a, **{k: a[k].astype(np.float64)}), h), "ENTITY.feat: dtype")

    def test_missing_and_extra_keys(self):
        self._bad(lambda a, h: ({k: v for k, v in a.items() if k != "row.phi"}, h), "missing")
        self.path, _ = write_shard(self.root / "x", drops=False)
        self._bad(lambda a, h: (dict(a, **{"row.extra": np.zeros(3)}), h), "unexpected")

    def test_shape_mismatch(self):
        self._bad(lambda a, h: (dict(a, **{"row.tick": a["row.tick"][:-1]}), h), "row.tick: shape")

    def test_token_width_mismatch(self):
        def fn(a, h):
            w = a["tok.w.ENTITY"].copy()
            w[0] += 1
            return dict(a, **{"tok.w.ENTITY": w}), h
        self._bad(fn, "ENTITY")

    def test_fixed_group_width_rejected(self):
        name = next(n for n, g in S.GROUPS.items() if not g.ragged)

        def fn(a, h):
            w = a[f"tok.w.{name}"].copy()
            w[:] = S.GROUPS[name].cap + 1
            return dict(a, **{f"tok.w.{name}": w}), h
        self._bad(fn, name)

    def test_ragged_flag_tamper_rejected(self):
        def fn(a, h):
            spec = dict(h["token_row_spec"])
            spec["ent_uid"] = [spec["ent_uid"][0], spec["ent_uid"][1], False]
            return a, dict(h, token_row_spec=spec)
        self._bad(fn, "ent_uid")

    def test_object_array_rejected(self):
        with np.load(self.path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["row.tick"] = np.array([{"x": 1}], dtype=object)
        with open(self.path, "wb") as fh:
            np.savez(fh, **arrays)
        with self.assertRaises(B.SchemaError):
            B.ShardReader(self.path)

    def test_update_level_mismatches(self):
        write_shard(self.root, 1)
        rewrite(B.shard_path(self.root, UPDATE, 1), lambda a, h: (a, dict(h, config_digest="other")))
        with self.assertRaisesRegex(B.SchemaError, "digest"):
            B.load_update(self.root, UPDATE)
        rewrite(B.shard_path(self.root, UPDATE, 1), lambda a, h: (a, dict(h, config_digest=CONF, update=8)))
        with self.assertRaisesRegex(B.SchemaError, "update"):
            B.load_update(self.root, UPDATE)
        B.shard_path(self.root, UPDATE, 1).unlink()
        with self.assertRaisesRegex(B.SchemaError, "expected 4"):
            B.load_update(self.root, UPDATE, expect_games=4)
        with self.assertRaisesRegex(B.SchemaError, "workers"):
            B.load_update(self.root, UPDATE, expect_workers=2)
        with self.assertRaises(FileNotFoundError):
            B.load_update(self.root, UPDATE + 1)

    def test_duplicate_game_ids_rejected(self):
        with writer(self.root, 1) as w:
            w.append_game(*kept_games(0)[0])                     # same id as worker 0's first game
        with self.assertRaisesRegex(B.SchemaError, "duplicate"):
            B.load_update(self.root, UPDATE)


# --------------------------------------------------------------------------- writer validation
class TestWriterValidation(TmpDir):

    def setUp(self):
        super().setUp()
        self.w = writer(self.root, 0)

    def tearDown(self):
        self.w.abort()
        super().tearDown()

    def test_actor_row_stratum_rejected(self):
        s = seat(0, 0)
        s.rows["stratum"] = np.where(s.rows["threat"], S.STRATUM_THREAT, S.STRATUM_BASE)   # query rows as waits
        with self.assertRaisesRegex(ValueError, "committed_stratum"):
            self.w.append_game(meta("w0-0", [s]), [s])

    def test_counter_and_metadata_errors(self):
        s = seat(0, 0)
        s.counters = dict(s.counters, committed=1)
        with self.assertRaisesRegex(ValueError, "committed"):
            self.w.append_game(meta("w0-0", [s]), [s])
        s = seat(0, 0)
        with self.assertRaisesRegex(ValueError, "update"):
            self.w.append_game(meta("w0-0", [s], update=UPDATE + 1), [s])
        with self.assertRaisesRegex(ValueError, "digest"):
            self.w.append_game(meta("w0-0", [s], config_digest="x"), [s])
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.w.append_game(meta("w0-0", [s]), [s, seat(1, 0)])
        with self.assertRaisesRegex(ValueError, "role"):
            self.w.append_game(meta("w0-0", [s]), [seat(0, 0, "frozen", weight=1.0)])
        with self.assertRaisesRegex(ValueError, "end"):
            self.w.append_game(meta("w0-0", [s], end="timeout"), [s])
        with self.assertRaisesRegex(ValueError, "hazard_dt"):
            self.w.append_game(meta("w0-0", [s], hazard_dt={"ours": "fixed"}), [s])
        with self.assertRaises(ValueError):
            self.w.append_game(meta("w0-0", [s], phi_end=float("nan")), [s])
        self.assertEqual(self.w.counters["games"], 0)             # nothing was counted

    def test_row_value_errors(self):
        s = seat(0, 0)
        s.rows["cell"] = s.rows["cell"].astype(np.int64)
        s.rows["cell"][0] = 600
        with self.assertRaisesRegex(ValueError, "cell"):
            self.w.append_game(meta("w0-0", [s]), [s])
        s = seat(0, 0)
        s.rows["phi"][0] = 1.5
        with self.assertRaisesRegex(ValueError, "phi"):
            self.w.append_game(meta("w0-0", [s]), [s])
        s = seat(0, 0)
        del s.rows["mu"]
        with self.assertRaisesRegex(ValueError, "missing"):
            self.w.append_game(meta("w0-0", [s]), [s])

    def test_token_errors(self):
        s = seat(0, 0)
        s.tokens = list(s.tokens)[:-1]
        with self.assertRaisesRegex(ValueError, "token rows"):
            self.w.append_game(meta("w0-0", [s]), [s])
        s = seat(0, 0)
        s.tokens = [dict(t) for t in s.tokens]
        bad = copy.deepcopy(s.tokens[3])
        bad["groups"]["ENTITY"]["feat"] = bad["groups"]["ENTITY"]["feat"].astype(np.float64)
        s.tokens[3] = bad
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.w.append_game(meta("w0-0", [s]), [s])
        s = seat(0, 0)
        s.tokens = [copy.deepcopy(t) for t in s.tokens]
        s.tokens[0]["row"]["tick"] = s.tokens[0]["row"]["tick"] + 1
        with self.assertRaisesRegex(ValueError, "tick"):
            self.w.append_game(meta("w0-0", [s]), [s])

    def test_seats_of_a_game_must_share_the_token_layout(self):
        s0, s1 = seat(3, 0, "mirror"), seat(4, 1, "mirror")
        s1.tokens = [copy.deepcopy(t) for t in s1.tokens]
        for t in s1.tokens:
            t["row"]["seq"] = t["row"]["seq"].astype(np.int32)
        with self.assertRaisesRegex(ValueError, "differ between the seats"):
            self.w.append_game(meta("w0-0", [s0, s1]), [s0, s1])
        self.assertEqual(self.w.counters["games"], 0)
        self.assertIsNone(self.w.append_game(meta("w0-1", [s0]), [s0]))
        with self.assertRaisesRegex(ValueError, "first seat"):
            self.w.append_game(meta("w0-2", [s1]), [s1])

    def test_multi_row_token_dict_accepted(self):
        rows, _, sub, _ = actor_seat(0)
        s = seat(0, 0)
        s.tokens = sub                                            # tokens_from_rows form, episode width
        self.assertIsNone(self.w.append_game(meta("w0-0", [s]), [s]))
        path = self.w.close()
        tok = B.ShardReader(path).seat_game(0).tokens()                # trimmed per row, same content
        for name, grp in sub["groups"].items():
            for k, v in grp.items():
                w = tok["groups"][name][k].shape[1]
                np.testing.assert_array_equal(tok["groups"][name][k], v[:, :w])
                self.assertTrue(np.all(v[:, w:] == _fill(k)))
        for k, v in sub["row"].items():
            w = tok["row"][k].shape[1] if v.ndim > 1 else None
            np.testing.assert_array_equal(tok["row"][k], v[:, :w] if w is not None else v)

    def test_add_counters_guards(self):
        with self.assertRaises(ValueError):
            self.w.add_counters(queries=3)
        with self.assertRaises(ValueError):
            self.w.add_counters(refusals_code19=-1)
        self.w.add_counters(refusals_code19=2)
        self.w.add_counters(refusals_code19=1)
        self.assertEqual(self.w.counters["refusals_code19"], 3)


# --------------------------------------------------------------------------- retention
class TestRetention(TmpDir):

    def test_keep_last_two_updates(self):
        for u in range(1, 6):
            d = B.update_dir(self.root, u)
            d.mkdir(parents=True)
            (d / "w000.npz").write_bytes(b"x")
            (d / ".w001.npz.tmp-1-abcd").write_bytes(b"y")
        (self.root / "notes").mkdir()
        self.assertEqual(B.list_updates(self.root), [1, 2, 3, 4, 5])
        self.assertEqual(B.prune_updates(self.root), [1, 2, 3])
        self.assertEqual(B.list_updates(self.root), [4, 5])
        self.assertTrue((self.root / "notes").is_dir())
        self.assertEqual(B.prune_updates(self.root, keep=2), [])

    def test_foreign_content_is_kept(self):
        for u in (1, 2, 3):
            B.update_dir(self.root, u).mkdir(parents=True)
        (B.update_dir(self.root, 1) / "checkpoint.pt").write_bytes(b"z")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(B.prune_updates(self.root, keep=1), [2])
        self.assertTrue(any("not removed" in str(c.message) for c in caught))
        self.assertEqual(B.list_updates(self.root), [1, 3])
        with self.assertRaises(ValueError):
            B.prune_updates(self.root, keep=0)

    def test_paths(self):
        self.assertEqual(B.shard_path("/r", 12, 3), Path("/r/u000012/w003.npz"))
        for bad in (-1, 10 ** 6, True):
            with self.assertRaises(ValueError):
                B.update_dir("/r", bad)
        with self.assertRaises(ValueError):
            B.shard_path("/r", 1, 1000)


if __name__ == "__main__":
    unittest.main()
