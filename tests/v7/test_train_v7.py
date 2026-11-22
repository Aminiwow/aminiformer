"""Tests of the imitation-learning trainer (``amini.train.il_v7``): training lock, runtime guard, losses, run
control and the end-to-end CPU smoke run.

Most tests need the private dataset (synthetic shards in the private format), the shard tokenizer
(``amini.nn.tokenizer_v7``), the vocabulary module (``amini.v7``) or the training configs; none of them is part of
this repository, so those tests are skipped and the configuration / pure-function tests run.
"""

from __future__ import annotations

import copy
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import signal
import time
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # tests/v7 (common, make_synthetic_shards)
from common import ROOT, TINY_MODEL, example, shard_root, source, tables, tiny_model, to_batch

from amini.train import il_v7


def _have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAVE_TOKENIZER = _have("amini.nn.tokenizer_v7")
HAVE_VOCAB = _have("amini.v7.vocab")
NEEDS_TOKENIZER = "needs the shard tokenizer, which is not part of this repository"
NEEDS_VOCAB = "needs the vocabulary module (amini.v7), which is not part of this repository"
NEEDS_CONFIGS = "needs the training configs, which are not part of this repository"
NEEDS_DATA = "needs the private dataset"


def _config(name: str) -> str:
    """Path of a training config of the private run set; the calling test is skipped when it is missing."""
    p = ROOT / "configs" / name
    if not p.is_file():
        raise unittest.SkipTest(NEEDS_CONFIGS)
    return str(p)


def _tmpd(case: unittest.TestCase, **kw) -> Path:
    """A fresh temp directory removed when the test ends (TMPDIR stays clean)."""
    d = Path(tempfile.mkdtemp(**kw))
    case.addCleanup(shutil.rmtree, d, True)
    return d


class TestTrainingLock(unittest.TestCase):
    def test_lock_shared_with_holder_record(self):
        gen = _tmpd(self).resolve() / "generations" / "g1"        # resolved: the lock path is a resolved path
        gen.mkdir(parents=True)
        (gen / "MANIFEST.json").write_text(json.dumps({"generation": "g1", "shards": []}))
        with il_v7.TrainingLock([str(gen)], "g1") as lk:
            self.assertEqual(lk.lock_path, gen.parent / ".training.lock")
            holders = list((gen.parent / ".training.holders").glob("*.json"))
            self.assertEqual(len(holders), 1)
            self.assertEqual(json.loads(holders[0].read_text())["generation"], "g1")
            with open(lk.lock_path, "a+") as fh:          # a publisher's LOCK_EX|LOCK_NB must fail
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with open(lk.lock_path, "a+") as fh:          # another trainer (LOCK_SH) may share it
                fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
        self.assertEqual(list((gen.parent / ".training.holders").glob("*.json")), [])
        with open(gen.parent / ".training.lock", "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)


class TestRuntimeGuard(unittest.TestCase):
    def test_refuses_fake_314_before_loading_a_shard(self):
        # imports happen under the real interpreter; the guard then sees a 3.14 version_info
        code = ("import sys; import amini.train.il_v7 as m; "
                "a = ['--config', sys.argv[1], '--data', sys.argv[2], '--out', sys.argv[3]]; "
                "sys.version_info = (3, 14, 0, 'final', 0); sys.exit(m.main(a))")
        tmp = _tmpd(self)
        out = tmp / "run"
        data = tmp / "generations" / "g"          # without the guard: lock file + FileNotFoundError
        data.mkdir(parents=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        p = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT / "configs/aminiformer_v1_smoke.yaml"),
                            str(data), str(out)], capture_output=True, text=True, env=env, timeout=300)
        self.assertNotEqual(p.returncode, 0)
        msg = p.stderr + p.stdout
        self.assertIn("refusing", msg)
        self.assertIn("3.14", msg)
        self.assertFalse(out.exists(), "nothing may be written before the guard")
        self.assertEqual(sorted(x.name for x in tmp.rglob("*")), ["g", "generations"])
        self.assertNotIn("FileNotFoundError", msg)

    def test_accepts_this_interpreter(self):
        il_v7.runtime_check()


class TestEndToEnd(unittest.TestCase):
    def test_cpu_smoke_train(self):
        cfg = _smoke_cfg(log_every=2)
        out = _tmpd(self) / "run"
        logs = []
        res = il_v7.train(cfg, [shard_root()], str(out), log=logs.append)
        self.assertEqual(res["steps"], 6)
        best = json.loads((out / "best.json").read_text())
        self.assertTrue(1 <= len(best) <= 3)
        self.assertTrue(Path(best[0][2]).exists())
        evals = [json.loads(l) for l in logs if '"event": "eval"' in l]
        self.assertTrue(evals)
        e = evals[-1]
        for k in ("selection_nll", "gate_nll_by_stratum", "calibration", "gate_auc", "diag"):
            self.assertIn(k, e)
        model = il_v7.load_policy(best[0][2])
        self.assertEqual(model.cfg.d_model, cfg.model["d_model"])


def _seal_write(path, doc):
    """Write ``doc`` with its recorded canonical sha256 (the table's self digest)."""
    from amini.v7 import vocab as V
    Path(path).write_text(json.dumps(V.seal(dict(doc, revision=doc.get("revision", "r0")))))


def _write_table_files(d: Path):
    """Real-layout table files built from the synthetic tables (for from_files tests)."""
    if not HAVE_VOCAB:
        raise unittest.SkipTest(NEEDS_VOCAB)
    from amini.v7 import vocab as V
    t = tables()
    cv = {"schema": V.CARD_SCHEMA, "entries": [
        {"id": i, "key": f"c{i}", "base_id": int(t.card_base[i]), "form": int(t.card_form[i]),
         "is_spell": bool(t.card_is_spell[i])} for i in range(t.card_vocab_size)]}
    names = {f"obj{i}": i for i in range(t.obj_vocab_size)}
    kinds = {0: "troop", 2: "troop", 3: "area", 4: "projectile", 5: "troop"}
    cat = {"schema": "cr-native-object-catalog.v1", "objects": {
        n: {"kind": kinds[int(t.obj_family[i])], "radius": None, "visual_only": False}
        for n, i in names.items()}}
    files = {"card_vocab": d / "card_vocab_v7.json", "object_vocab": d / "object_vocab_v7.json",
             "object_catalog": d / "native_object_catalog.json", "arrival": d / "opp_cmd_arrival_v7.json"}
    _seal_write(files["card_vocab"], cv)
    _seal_write(files["object_vocab"], {"schema": V.OBJECT_SCHEMA, "ids": names})
    files["object_catalog"].write_text(json.dumps(cat))         # the catalog records no self digest
    _seal_write(files["arrival"], {"schema": V.ARRIVAL_SCHEMA, "counts": {str(k): 3 for k in range(1, 11)}})
    return {k: str(v) for k, v in files.items()}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@unittest.skipUnless(HAVE_TOKENIZER, NEEDS_TOKENIZER)
class TestTables(unittest.TestCase):
    def cfg(self, **over):
        return _smoke_cfg(**over)

    def test_missing_tables_refused(self):
        with self.assertRaises(ValueError) as cm:
            il_v7.build_tables(self.cfg(tables={}))
        self.assertIn("card_vocab", str(cm.exception))
        with self.assertRaises(ValueError):
            il_v7.build_tables(self.cfg(tables={"card_vocab": "x.json"}))
        for prof in (("main", "large") if (ROOT / "configs").is_dir() else ()):   # shipped real profiles
            c = il_v7.TrainConfig.from_file(_config(f"aminiformer_v1_{prof}.yaml"))   # never fall back to synthetic
            self.assertFalse(c.tables.get("synthetic"))
            for k in il_v7.REQUIRED_TABLES:
                self.assertTrue(c.tables.get(k), (prof, k))

    def test_synthetic_opt_in_only_on_synthetic_shards(self):
        t = il_v7.build_tables(self.cfg())
        self.assertEqual(t.source["kind"], "synthetic")
        rec = il_v7.verify_tables(t, source())          # synthetic shards: accepted
        self.assertEqual(rec["kind"], "synthetic")
        real = types.SimpleNamespace(synthetic=False, table_digests=lambda: {})
        with self.assertRaises(ValueError):
            il_v7.verify_tables(t, real)
        pinned = types.SimpleNamespace(synthetic=True, table_digests=lambda: {"card_vocab": "0" * 64})
        with self.assertRaises(ValueError):
            il_v7.verify_tables(t, pinned)

    def test_file_tables_must_match_manifest_pins(self):
        d = _tmpd(self)
        files = _write_table_files(d)
        t = il_v7.build_tables(self.cfg(tables=files))
        self.assertEqual(set(t.source["files"]), set(files))
        pins = {Path(p).name: _sha(p) for p in files.values()}      # pinned by file name
        ok = types.SimpleNamespace(synthetic=False, table_digests=lambda: pins)
        rec = il_v7.verify_tables(t, ok)
        self.assertEqual(rec["files"]["card_vocab"]["sha256"], _sha(files["card_vocab"]))
        bad = dict(pins, **{"card_vocab_v7.json": "f" * 64})
        with self.assertRaises(ValueError) as cm:
            il_v7.verify_tables(t, types.SimpleNamespace(synthetic=False, table_digests=lambda: bad))
        self.assertIn("card_vocab", str(cm.exception))
        part = {k: v for k, v in pins.items() if k != "opp_cmd_arrival_v7.json"}
        with self.assertRaises(ValueError) as cm:
            il_v7.verify_tables(t, types.SimpleNamespace(synthetic=False, table_digests=lambda: part))
        self.assertIn("not pinned", str(cm.exception))

    def test_manifest_digest_collection(self):
        root = _tmpd(self) / "data"
        shutil.copytree(shard_root(), root)
        sd = next((root / "shards").iterdir())
        m = json.loads((sd / "MANIFEST.json").read_text())
        m["vocab_digests"] = {"card_vocab_v7.json": "a" * 64}
        m["table_digests"] = {"opp_cmd_arrival_v7.json": {"sha256": "b" * 64}}
        (sd / "MANIFEST.json").write_text(json.dumps(m))
        src, _ = il_v7.ShardSource.open([str(root / "generations" / "synthetic")])
        pins = src.table_digests()
        self.assertEqual(pins["card_vocab"], "a" * 64)
        self.assertEqual(pins["opp_cmd_arrival"], "b" * 64)
        gm = root / "generations" / "synthetic" / "MANIFEST.json"
        g = json.loads(gm.read_text())
        g["vocab_digests"] = {"card_vocab": "c" * 64}                 # conflicting pin
        gm.write_text(json.dumps(g))
        src, _ = il_v7.ShardSource.open([str(root / "generations" / "synthetic")])
        with self.assertRaises(ValueError):
            src.table_digests()

    def test_real_catalog_layout(self):
        """The real object catalog file (kind / visual_only per object)."""
        cat_path = ROOT / "data" / "native_object_catalog.json"
        if not cat_path.is_file():
            self.skipTest(NEEDS_DATA)
        cat = json.loads(cat_path.read_text())["objects"]
        d = _tmpd(self)
        files = _write_table_files(d)
        names = sorted(cat)
        from amini.v7 import vocab as V
        _seal_write(files["object_vocab"], {"schema": V.OBJECT_SCHEMA,
                                            "ids": {n: i + 2 for i, n in enumerate(names)}})
        from amini.nn.tokenizer_v7 import TokenizerTables
        t = TokenizerTables.from_files(files["card_vocab"], files["object_vocab"], files["arrival"],
                                       str(cat_path))
        self.assertEqual(t.source["catalog_matched"], len(cat))
        want = {"area": 3, "projectile": 4, "troop": 5, "building": 5}
        for i, n in enumerate(names):
            self.assertEqual(int(t.obj_family[i + 2]), want[cat[n]["kind"]], n)
            self.assertEqual(bool(t.obj_has_hp[i + 2]), cat[n]["kind"] in ("troop", "building"), n)
            self.assertEqual(float(t.obj_visual_only[i + 2]), float(cat[n]["visual_only"]), n)


@unittest.skipUnless(HAVE_TOKENIZER and HAVE_VOCAB and (ROOT / "data").is_dir(),
                     "needs the private vocabulary tables and the shard tokenizer")
class TestTableSelfDigests(unittest.TestCase):
    """from_files verifies each table's recorded sha256 (vocab.load_verified); the MANIFEST
    byte pin (verify_tables) stays an independent second check."""

    NAMES = {"card_vocab": "card_vocab_v7.json", "object_vocab": "object_vocab_v7.json",
             "arrival": "opp_cmd_arrival_v7.json", "object_catalog": "native_object_catalog.json",
             "visual_alias": "object_alias_visual_v7.json"}

    def copy_real(self):
        d = _tmpd(self, prefix="wpg-tables-")
        out = {}
        for k, n in self.NAMES.items():
            shutil.copy(ROOT / "data" / n, d / n)
            out[k] = str(d / n)
        return out

    def load(self, f):
        from amini.nn.tokenizer_v7 import TokenizerTables
        return TokenizerTables.from_files(f["card_vocab"], f["object_vocab"], f["arrival"],
                                          f["object_catalog"], f["visual_alias"])

    def test_real_tables_verify(self):
        f = self.copy_real()
        t = self.load(f)
        recs = t.source["files"]
        for k in ("card_vocab", "object_vocab", "arrival", "visual_alias"):
            doc = json.loads(Path(f[k]).read_text())
            self.assertEqual(recs[k]["self_sha256"], doc["sha256"], k)
            self.assertEqual(recs[k]["sha256"], _sha(f[k]), k)          # byte digest for the pins
        self.assertNotIn("self_sha256", recs["object_catalog"])         # records none: plain load

    def test_tampered_copy_refused(self):
        from amini.v7 import vocab as V

        def tamper(doc, k):
            if k == "card_vocab":
                doc["entries"][5]["is_spell"] = not doc["entries"][5]["is_spell"]
            elif k == "object_vocab":
                a, b = list(doc["ids"])[2:4]
                doc["ids"][a], doc["ids"][b] = doc["ids"][b], doc["ids"][a]
            elif k == "arrival":
                key = sorted(doc["pmf"])[0]
                doc["pmf"][key] = float(doc["pmf"][key]) + 0.01
            else:
                key = sorted(doc["alias"])[0]
                doc["alias"][key] = key
            return doc

        for k in ("card_vocab", "object_vocab", "arrival", "visual_alias"):
            f = self.copy_real()
            doc = json.loads(Path(f[k]).read_text())
            Path(f[k]).write_text(json.dumps(tamper(doc, k)))          # recorded sha256 kept
            with self.assertRaises(V.VocabError) as cm:
                self.load(f)
            self.assertIn("recorded sha256", str(cm.exception), k)
            f = self.copy_real()
            doc = json.loads(Path(f[k]).read_text())
            doc.pop("sha256")
            Path(f[k]).write_text(json.dumps(doc))                     # self digest missing
            with self.assertRaises(V.VocabError):
                self.load(f)
            f = self.copy_real()
            doc = json.loads(Path(f[k]).read_text())
            doc["schema"] = "something-else"
            Path(f[k]).write_text(json.dumps(V.seal(doc)))             # wrong schema, sealed
            with self.assertRaises(V.VocabError):
                self.load(f)
        f = self.copy_real()                                          # a catalog that records a
        cat = json.loads(Path(f["object_catalog"]).read_text())       # digest must match it
        cat.update(revision="r0", sha256="0" * 64)
        Path(f["object_catalog"]).write_text(json.dumps(cat))
        with self.assertRaises(V.VocabError):
            self.load(f)
        cat.pop("sha256")
        Path(f["object_catalog"]).write_text(json.dumps(V.seal(cat)))
        self.assertIn("self_sha256", self.load(f).source["files"]["object_catalog"])

    def test_manifest_byte_pin_still_checked(self):
        """A re-serialised table keeps a valid self digest but changes the bytes: from_files
        accepts it, verify_tables refuses it against the MANIFEST pin of the original."""
        f = self.copy_real()
        pins = {self.NAMES[k]: _sha(p) for k, p in f.items()}
        doc = json.loads(Path(f["card_vocab"]).read_text())
        Path(f["card_vocab"]).write_text(json.dumps(doc, indent=3))
        t = self.load(f)
        with self.assertRaises(ValueError) as cm:
            il_v7.verify_tables(t, types.SimpleNamespace(synthetic=False, table_digests=lambda: pins))
        self.assertIn("card_vocab", str(cm.exception))
        pins["card_vocab_v7.json"] = _sha(f["card_vocab"])
        rec = il_v7.verify_tables(t, types.SimpleNamespace(synthetic=False, table_digests=lambda: pins))
        self.assertEqual(rec["kind"], "files")


class TestLockOrder(unittest.TestCase):
    def test_lock_and_holder_before_any_memmap(self):
        root = _tmpd(self) / "data"
        shutil.copytree(shard_root(), root)
        gen = root / "generations" / "synthetic"
        seen = []
        orig = il_v7.ShardSource.__init__

        def check():
            # record only the first data open (the storage path and the fallback reader
            # may both be on the call stack); the holder record and LOCK_SH must exist
            if seen:
                return
            holders = list((gen.parent / ".training.holders").glob("*.json"))
            seen.append([json.loads(h.read_text())["generation"] for h in holders])
            with open(gen.parent / ".training.lock", "a+") as fh:     # LOCK_EX must already fail
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen.append("LOCK_EX acquired")
                    fcntl.flock(fh, fcntl.LOCK_UN)
                except BlockingIOError:
                    pass

        def spy(obj, *a, **k):
            check()
            orig(obj, *a, **k)

        patches = [mock.patch.object(il_v7.ShardSource, "__init__", spy)]
        try:                                    # the optional storage module, when installed, is the generation path
            from amini.v7 import storage as storage_mod
        except ImportError:
            storage_mod = None
        if storage_mod is not None and hasattr(storage_mod, "open_generation"):
            orig_open = storage_mod.open_generation

            def spy_open(p, *a, **k):
                check()
                return orig_open(p, *a, **k)
            patches.append(mock.patch.object(storage_mod, "open_generation", spy_open))

        cfg = _smoke_cfg(max_steps=1, eval_every=0, num_workers=0, batch_rows=300, eval_max_batches=1)
        with contextlib.ExitStack() as stack:
            for pt in patches:
                stack.enter_context(pt)
            il_v7.train(cfg, [str(gen)], str(root.parent / "run"), log=lambda *_: None)
        self.assertEqual(seen, [["synthetic"]])


class TestGenerationAdapter(unittest.TestCase):
    def with_storage(self, open_generation):
        import amini.v7 as pkg
        mod = types.ModuleType("amini.v7.storage")
        mod.open_generation = open_generation
        return mock.patch.dict(sys.modules, {"amini.v7.storage": mod}), \
            mock.patch.object(pkg, "storage", mod, create=True)

    def test_compatible_generation_matches_fallback(self):
        ref = source()
        gen_dir = str(Path(shard_root()) / "generations" / "synthetic")

        class Gen:
            shards = [dict(t) for t in ref.shards]
            episodes = list(ref.episodes)
            ent_off = ref.ent_off
            obj_off = ref.obj_off

        p1, p2 = self.with_storage(lambda path: Gen())
        with p1, p2:
            src, gid = il_v7.ShardSource.open([gen_dir])
        self.assertIsInstance(src, il_v7.GenerationAdapter)
        self.assertEqual(gid, "synthetic")
        self.assertEqual(src.episodes, ref.episodes)
        self.assertTrue(src.synthetic)
        for ep in ref.episodes[:4]:
            a, b = src.episode(ep), ref.episode(ep)
            self.assertEqual(a.tag, b.tag)
            for k in ("rows", "ent", "obj", "evt", "life", "priv"):
                self.assertTrue(np.array_equal(getattr(a, k), getattr(b, k)), k)
            self.assertEqual(src.split_of(ep), ref.split_of(ep))
            self.assertEqual(src.token_max(ep, ep[3], ep[4]), ref.token_max(ep, ep[3], ep[4]))

    def test_form_mismatch_manifest_copy_must_agree(self):
        ref = source()
        gen_dir = str(Path(shard_root()) / "generations" / "synthetic")
        fm = np.asarray(ref.shards[0]["games"]["form_mismatch"]).astype(np.uint8)
        self.assertTrue(np.any(fm != 0))

        def gen(copy_fm):
            return types.SimpleNamespace(shards=[dict(t) for t in ref.shards], episodes=list(ref.episodes),
                                         ent_off=ref.ent_off, obj_off=ref.obj_off, form_mismatch=copy_fm)

        for ok in ([fm.copy()], [np.zeros_like(fm)]):         # equal, or absent (zeros) in the MANIFEST
            p1, p2 = self.with_storage(lambda path, ok=ok: gen(ok))
            with p1, p2:
                src, _ = il_v7.ShardSource.open([gen_dir])
            self.assertIsInstance(src, il_v7.GenerationAdapter)
        bad = fm.copy()
        zeros = np.nonzero(fm == 0)[0]
        if len(zeros):
            bad[zeros[0]] = 1
        else:
            bad[0] = 3 if fm[0] != 3 else 1
        p1, p2 = self.with_storage(lambda path: gen([bad]))
        with p1, p2:
            with self.assertRaises(RuntimeError) as cm:
                il_v7.ShardSource.open([gen_dir])
        self.assertIn("form_mismatch", str(cm.exception))

        def gen_m(copy_fm, manifests):
            g = gen(copy_fm)
            g.manifests = manifests
            return g

        # a numpy array (not a list) of per-shard copies is accepted when equal
        p1, p2 = self.with_storage(lambda path: gen(np.asarray([fm.copy()])))
        with p1, p2:
            src, _ = il_v7.ShardSource.open([gen_dir])
        self.assertIsInstance(src, il_v7.GenerationAdapter)
        # refused: wrong length; wrong shard count; all-zero copy from a MANIFEST that has the key
        bad_cases = [
            gen([fm[:-1].copy()]),
            gen([fm.copy(), fm.copy()]),
            gen_m([np.zeros_like(fm)], [{"generation": "g"}, {"form_mismatch": [0] * len(fm)}]),
            gen_m([np.zeros_like(fm)], [{"form_mismatch": [0] * len(fm)}]),
        ]
        for g in bad_cases:
            p1, p2 = self.with_storage(lambda path, g=g: g)
            with p1, p2:
                with self.assertRaises(RuntimeError) as cm:
                    il_v7.ShardSource.open([gen_dir])
            self.assertIn("form_mismatch", str(cm.exception))
        # all-zero copy tolerated when the shard MANIFEST provably lacks the key
        p1, p2 = self.with_storage(lambda path: gen_m([np.zeros_like(fm)], [{"generation": "g"}, {}]))
        with p1, p2:
            src, _ = il_v7.ShardSource.open([gen_dir])
        self.assertIsInstance(src, il_v7.GenerationAdapter)

    def test_incompatible_generation_refused(self):
        gen_dir = str(Path(shard_root()) / "generations" / "synthetic")
        p1, p2 = self.with_storage(lambda path: types.SimpleNamespace(tables={}))
        with p1, p2:
            with self.assertRaises(RuntimeError) as cm:
                il_v7.ShardSource.open([gen_dir])
        self.assertIn("episodes", str(cm.exception))
        ref = source()
        bad_off = types.SimpleNamespace(shards=[dict(t) for t in ref.shards], episodes=ref.episodes,
                                        ent_off=[x[:-1] for x in ref.ent_off])
        p1, p2 = self.with_storage(lambda path: bad_off)
        with p1, p2:
            with self.assertRaises(ValueError):
                il_v7.ShardSource.open([gen_dir])


class TestMicroBatching(unittest.TestCase):
    def test_step_split_gives_exact_full_step_gradient(self):
        import torch
        exs = [example(i)[0] for i in range(3)]
        for e in exs:
            e["labels"]["loss_mask"] = np.ones(len(e["labels"]["family"]), bool)
        model = tiny_model()
        full = to_batch(exs)
        cr, ps = il_v7.cell_and_slot(full)
        model.zero_grad()
        lf = il_v7.compute_losses(model(full, cell_rows=cr, play_slot=ps), full, il_v7.DEFAULT_LOSS)
        lf["total"].backward()
        g_full = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        micro = [to_batch(exs[:2]), to_batch(exs[2:])]
        dens = {}
        for b in micro:
            for k, v in il_v7.loss_denominators(b).items():
                dens[k] = dens[k] + v if k in dens else v
        model.zero_grad()
        total = 0.0
        terms = {}
        for b in micro:
            cr, ps = il_v7.cell_and_slot(b)
            r = il_v7.compute_losses(model(b, cell_rows=cr, play_slot=ps), b, il_v7.DEFAULT_LOSS, dens=dens)
            r["total"].backward()
            total += float(r["total"])
            for k, v in r["terms"].items():
                terms[k] = terms.get(k, 0.0) + float(v)
        self.assertAlmostEqual(total, float(lf["total"]), places=10)
        for k, v in lf["terms"].items():
            self.assertAlmostEqual(terms[k], float(v), places=10, msg=k)
        self.assertEqual(set(g_full), {n for n, p in model.named_parameters() if p.grad is not None})
        for n, g in g_full.items():
            self.assertTrue(torch.allclose(g, dict(model.named_parameters())[n].grad, atol=1e-10, rtol=1e-8), n)

    def test_token_budget_and_row_caps(self):
        src = source()
        items = il_v7.chunk_episodes(src, 0, 512, 192)
        budget = il_v7.micro_cost(120, 24, 24)
        cfg = _smoke_cfg(batch_rows=500, micro_batch_rows=200, max_batch_tokens=budget)
        ds = il_v7.EpisodeBatches(src, items, tables(), cfg, train=True)
        seen = []
        n_split = 0
        for step in ds.batches:
            rows_step = sum(items[i][3] - items[i][1] for m in step for i in m)
            self.assertTrue(rows_step <= 500 or sum(len(m) for m in step) == 1)
            n_split += len(step) > 1
            for m in step:
                seen += m
                rows = sum(items[i][3] - items[i][1] for i in m)
                e = max(ds.tok[i][0] for i in m)
                o = max(ds.tok[i][1] for i in m)
                if len(m) > 1:
                    self.assertLessEqual(rows, 200)
                    self.assertLessEqual(il_v7.micro_cost(rows, e, o), budget)
        self.assertEqual(sorted(seen), list(range(len(items))))
        self.assertGreater(n_split, 0)
        got = ds[0]
        self.assertEqual(len(got), len(ds.batches[0]))
        for nb, m in zip(got, ds.batches[0]):
            self.assertEqual(int(nb["row"]["row_valid"].sum()), sum(items[i][3] - items[i][1] for i in m))
            g = nb["groups"]
            pad = il_v7.fixed_tokens() + g["ENTITY"]["valid"].shape[2] + g["OBJECT"]["valid"].shape[2]
            self.assertEqual(il_v7.micro_cost(int(nb["row"]["row_valid"].sum()),
                                              max(ds.tok[i][0] for i in m), max(ds.tok[i][1] for i in m)),
                             int(nb["row"]["row_valid"].sum()) * pad)

    def test_shipped_profiles_use_micro_batches(self):
        for prof in ("main", "large"):
            c = il_v7.TrainConfig.from_file(_config(f"aminiformer_v1_{prof}.yaml"))
            self.assertLess(c.micro_rows, c.batch_rows)
            self.assertGreater(c.max_batch_tokens, 0)
            self.assertEqual(c.cuda_alloc_conf, "expandable_segments:True")


class TestAllocatorAndConstants(unittest.TestCase):
    def test_alloc_conf_setdefault(self):
        cfg = il_v7.TrainConfig()                                 # default cuda_alloc_conf
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
        with mock.patch.dict(os.environ, env, clear=True):
            rec = il_v7.cuda_alloc_setup(cfg)
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual(rec["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        with mock.patch.dict(os.environ, dict(env, PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"), clear=True):
            il_v7.cuda_alloc_setup(cfg)
            self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:128")

    def test_sampling_constants_single_source(self):
        """The model's hazard unit is the schema's U_TICKS; KAPPA has one rate per committed-wait stratum."""
        from amini.nn import schema_v1 as S
        self.assertEqual(set(S.KAPPA), {"base", "threat"})
        self.assertTrue(all(0.0 < float(k) <= 1.0 for k in S.KAPPA.values()))
        import inspect
        from amini.nn.aminiformer import hazard_probs, sample_actions
        self.assertEqual(inspect.signature(hazard_probs).parameters["u_ticks"].default, S.U_TICKS)
        self.assertEqual(inspect.signature(sample_actions).parameters["u_ticks"].default, S.U_TICKS)

    def test_aux_opp_hand_scale(self):
        import torch
        exs = [example(i)[0] for i in range(2)]
        batch = to_batch(exs)
        model = tiny_model()
        cr, ps = il_v7.cell_and_slot(batch)
        with torch.no_grad():
            out = model(batch, cell_rows=cr, play_slot=ps)
            r = il_v7.compute_losses(out, batch, il_v7.DEFAULT_LOSS, opp_hand_reduce="class_mean")
            d = il_v7.compute_losses(out, batch, il_v7.DEFAULT_LOSS)          # default: sum_hand
        v = float(r["terms"]["aux_opp_hand"])
        # per-class mean BCE at the 4-of-V prior: about 0.08
        self.assertLess(v, 0.5)
        self.assertGreater(v, 0.0)
        # sum over V / 4 at the prior: V/4 x the class mean, far below the no-prior scale V/4 x ln 2
        vv = out["aux_opp_hand"].shape[-1]
        dv = float(d["terms"]["aux_opp_hand"])
        self.assertAlmostEqual(dv, v * vv / 4, places=3)
        self.assertLess(dv, 0.5 * vv / 4 * math.log(2))


SMOKE = str(ROOT / "configs/aminiformer_v1_smoke.yaml")
# stand-in for the smoke config when the training configs are not available: a tiny model on synthetic tables
SMOKE_EXAMPLE = {"name": "smoke", "model": dict(TINY_MODEL), "tables": {"synthetic": True}}


def _smoke_cfg(**over):
    base = {"max_steps": 6, "eval_every": 3, "num_workers": 0, "batch_rows": 400, "log_every": 1,
            "eval_max_batches": 2}
    base.update(over)
    if Path(SMOKE).is_file():
        return il_v7.TrainConfig.from_file(SMOKE, base)
    return il_v7.TrainConfig(**{**SMOKE_EXAMPLE, **base})


def _ck(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


class TestRunControl(unittest.TestCase):
    def test_total_steps_from_epochs_and_cosine(self):
        c = _smoke_cfg(max_epochs=1.5, warmup_steps=2)
        self.assertEqual(il_v7.total_train_steps(c, 10), 15)
        self.assertEqual(il_v7.total_train_steps(_smoke_cfg(), 10), 6)          # max_steps
        self.assertEqual(il_v7.total_train_steps(_smoke_cfg(max_epochs=1.0), 7), 7)
        self.assertAlmostEqual(il_v7.lr_at(15, c, 15), c.lr * c.min_lr_frac)
        self.assertAlmostEqual(il_v7.lr_at(2, c, 15), c.lr)
        self.assertGreater(il_v7.lr_at(8, c, 15), il_v7.lr_at(8, c, 10))
        with self.assertRaises(ValueError):
            _smoke_cfg(max_epochs=-1)

    def test_cap_val_items_deterministic(self):
        src = source()
        items = il_v7.chunk_episodes(src, 0, 512, 192) + il_v7.chunk_episodes(src, 1, 512, 192)
        games = {(it[0][0], it[0][1]) for it in items}
        self.assertGreater(len(games), 2)
        self.assertEqual(il_v7.cap_val_items(src, items, 0), items)
        a = il_v7.cap_val_items(src, items, 2)
        self.assertEqual(a, il_v7.cap_val_items(src, list(items), 2))
        self.assertEqual(len({(it[0][0], it[0][1]) for it in a}), 2)
        keep = {(it[0][0], it[0][1]) for it in a}
        self.assertEqual(a, [it for it in items if (it[0][0], it[0][1]) in keep])   # order, all sides/chunks
        h = sorted(games, key=lambda g: hashlib.sha256(bytes(src.shards[g[0]]["games"]["replay_id"][g[1]])).digest())
        self.assertEqual(keep, set(h[:2]))
        self.assertEqual(il_v7.cap_val_items(src, items, 10 ** 6), items)

    def test_pinned_val_items(self):
        import tempfile
        src = source()
        items = il_v7.chunk_episodes(src, 0, 512, 192) + il_v7.chunk_episodes(src, 1, 512, 192)
        a = il_v7.cap_val_items(src, items, 2)
        keep = sorted({(it[0][0], it[0][1]) for it in a})
        tags = [il_v7._tag_text(src, *g) for g in keep]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "val.txt"
            f.write_text("# pinned\n" + "\n".join(tags) + "\n\n")
            b, digest = il_v7.pinned_val_items(src, items, str(f), 2000)
            self.assertEqual(b, a)                                     # same games, same order
            self.assertEqual(digest, hashlib.sha256(f.read_bytes()).hexdigest())
            f.write_text("\n".join(tags[:1]))
            b1, _ = il_v7.pinned_val_items(src, list(reversed(items)), str(f))
            self.assertEqual({(it[0][0], it[0][1]) for it in b1}, {keep[0]})
            f.write_text("\n".join(tags + ["NOT-A-GAME"]))
            with self.assertRaises(ValueError):
                il_v7.pinned_val_items(src, items, str(f))             # unknown game refused
            f.write_text("\n".join(tags))
            with self.assertRaises(ValueError):
                il_v7.pinned_val_items(src, items, str(f), 1)          # longer than the cap

    def test_grad_stats(self):
        g = il_v7._grad_stats([0.5, 2.0, 3.0, float("inf"), 4.0], 1.0)
        self.assertEqual(g["n"], 5)
        self.assertEqual(g["nonfinite"], 1)
        self.assertAlmostEqual(g["median"], 2.5)
        self.assertAlmostEqual(g["clipped_frac"], 0.75)
        self.assertEqual(il_v7._grad_stats([float("nan")], 1.0), {"n": 1, "nonfinite": 1})

    def test_early_stop_result_and_best(self):
        cfg = _smoke_cfg(max_steps=20, eval_every=1, early_stop_patience=2, early_stop_min_steps=2,
                         early_stop_min_delta=1e6)          # nothing after the first eval counts
        out = _tmpd(self) / "run"
        logs = []
        res = il_v7.train(cfg, [shard_root()], str(out), log=logs.append)
        # eval 1 sets the best; evals 2 and 3 do not improve; patience 2 -> stop at step 3
        self.assertEqual(res["stop_reason"], "early_stop")
        self.assertEqual(res["steps"], 3)
        r = json.loads((out / "result.json").read_text())
        self.assertEqual(r["stop_reason"], "early_stop")
        self.assertEqual(r["early_stop"]["best_step"], 1)
        self.assertEqual(r["early_stop"]["bad_evals"], 2)
        self.assertTrue(Path(r["best_ckpt"]).exists())
        self.assertEqual(r["best_step"], min(r["best"])[1])
        self.assertEqual(os.stat(out / "best.pt").st_ino, os.stat(r["best_ckpt"]).st_ino)
        self.assertEqual(os.stat(out / "last.pt").st_ino, os.stat(out / "ckpt-0000003.pt").st_ino)
        self.assertTrue(any('"event": "early_stop"' in l for l in logs))
        for k in ("train_seconds", "eval_seconds", "rows", "rows_per_s"):
            self.assertIn(k, r["timing"])
        for k in ("median", "p90", "clipped_frac"):
            self.assertIn(k, r["grad_norm"])
        recs = [json.loads(l) for l in logs if l.startswith('{"step"')]
        for k in ("gn_median", "gn_p90", "clipped_frac", "rows_s"):
            self.assertIn(k, recs[-1])

    def test_rows_s_excludes_eval_time(self):
        cfg = _smoke_cfg(max_steps=4, eval_every=2, log_every=2)
        real = il_v7.evaluate

        def slow(*a, **k):
            time.sleep(1.0)
            return real(*a, **k)
        out = _tmpd(self) / "run"
        logs = []
        t0 = time.time()
        with mock.patch.object(il_v7, "evaluate", slow):
            res = il_v7.train(cfg, [shard_root()], str(out), log=logs.append)
        wall = time.time() - t0
        tm = res["timing"]
        self.assertGreaterEqual(tm["eval_seconds"], 2.0)
        self.assertLess(tm["train_seconds"] + tm["eval_seconds"], wall + 1e-6)
        evs = [json.loads(l) for l in logs if '"event": "eval"' in l]
        self.assertTrue(all(e["eval_seconds"] >= 1.0 for e in evs))
        recs = [json.loads(l) for l in logs if l.startswith('{"step"')]
        # the step-4 interval contains the step-2 eval (1 s sleep) only through its wall time
        rows4 = recs[-1]["rows_s"]
        self.assertGreater(rows4, 0)
        self.assertGreater(tm["rows_per_s"], tm["rows"] / wall)

    def test_signal_stop_and_exact_resume(self):
        import torch
        cfg = _smoke_cfg(max_steps=7, eval_every=3)
        a = _tmpd(self) / "a"
        ra = il_v7.train(cfg, [shard_root()], str(a), log=lambda m: None)
        self.assertEqual(ra["stop_reason"], "total_steps")
        b = _tmpd(self) / "b"

        def killer(m):
            if m.startswith('{"step": 4,'):
                os.kill(os.getpid(), signal.SIGTERM)
        old = signal.getsignal(signal.SIGTERM)
        rb = il_v7.train(cfg, [shard_root()], str(b), log=killer)
        self.assertIs(signal.getsignal(signal.SIGTERM), old)              # handler restored
        self.assertEqual(rb["stop_reason"], "signal")
        self.assertEqual(rb["steps"], 4)                                  # the step finished
        self.assertEqual(json.loads((b / "result.json").read_text())["stop_reason"], "signal")
        ck = _ck(b / "last.pt")
        self.assertEqual(ck["step"], 4)
        self.assertEqual(ck["resume"]["step"], 4)
        with self.assertRaises(ValueError):                               # the trajectory would change
            il_v7.train(_smoke_cfg(max_steps=7, eval_every=3, lr=0.002), [shard_root()], str(b),
                        log=lambda m: None, resume=True)
        rc = il_v7.train(_smoke_cfg(max_steps=7, eval_every=3, log_every=2), [shard_root()], str(b),
                         log=lambda m: None, resume=True)                 # log_every is resume-free
        self.assertEqual(rc["stop_reason"], "total_steps")
        self.assertEqual(rc["steps"], 7)
        self.assertEqual(rc["resumes"], [4])
        fa, fb = _ck(a / "ckpt-0000007.pt"), _ck(b / "ckpt-0000007.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))
        self.assertEqual(fa["metrics"]["selection_nll"], fb["metrics"]["selection_nll"])
        la = [h["loss"] for h in ra["history"] if "loss" in h and h["step"] in (5, 6, 7)]
        lb = [h["loss"] for h in rc["history"] if "loss" in h and h["step"] in (6,)]
        self.assertEqual(la[1], lb[0])
        with self.assertRaises(FileNotFoundError):
            il_v7.train(cfg, [shard_root()], str(_tmpd(self) / "none"), log=lambda m: None, resume=True)

    def test_sigterm_subprocess_exit0_then_resume(self):
        out = _tmpd(self) / "run"
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        cmd = [sys.executable, "-B", "-m", "amini.train.il_v7", "--config", SMOKE,
               "--data", shard_root(), "--out", str(out), "--set", "max_steps=400", "eval_every=0",
               "num_workers=0", "batch_rows=400", "log_every=1"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                             cwd=str(ROOT))
        lines = []
        try:
            for line in p.stdout:
                lines.append(line)
                if line.startswith('{"step": 2,'):
                    p.send_signal(signal.SIGTERM)
                    break
            rest, _ = p.communicate(timeout=300)
        finally:
            if p.poll() is None:
                p.kill()
        lines += rest.splitlines()
        self.assertEqual(p.returncode, 0, "".join(lines[-20:]))
        r = json.loads((out / "result.json").read_text())
        self.assertEqual(r["stop_reason"], "signal")
        self.assertEqual(r["signal"], int(signal.SIGTERM))
        self.assertLess(r["steps"], 400)
        self.assertTrue(any('"event": "signal_stop"' in l for l in lines))
        done = json.loads([l for l in lines if l.startswith('{"done"')][-1])
        self.assertEqual(done["stop_reason"], "signal")
        n = r["steps"]
        cmd2 = cmd + ["--resume"]
        p2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                              cwd=str(ROOT))
        try:
            for line in p2.stdout:
                if line.startswith('{"step": %d,' % (n + 1)):
                    p2.send_signal(signal.SIGINT)                         # SIGINT works the same way
                    break
            p2.communicate(timeout=300)
        finally:
            if p2.poll() is None:
                p2.kill()
        self.assertEqual(p2.returncode, 0)
        r2 = json.loads((out / "result.json").read_text())
        self.assertEqual(r2["stop_reason"], "signal")
        self.assertEqual(r2["resumes"], [n])
        self.assertGreater(r2["steps"], n)


def _logs_json(logs, prefix):
    return [json.loads(l) for l in logs if l.startswith(prefix)]


class TestWarmStart(unittest.TestCase):
    """Warm start: init_from / init_weights / eval_at_start / lr_mult."""

    @classmethod
    def setUpClass(cls):
        cls._dir = Path(tempfile.mkdtemp(prefix="a0-warm-"))
        src = cls._dir / "src"
        il_v7.train(_smoke_cfg(max_steps=6, eval_every=6), [shard_root()], str(src), log=lambda m: None)
        cls.init = str(src / "ckpt-0000006.pt")
        cls.init_sha = hashlib.sha256(Path(cls.init).read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._dir, True)

    def _warm(self, out, log=None, resume=False, **over):
        base = {"init_from": self.init, "eval_at_start": True, "max_steps": 3, "eval_every": 0,
                "warmup_steps": 2}
        base.update(over)
        logs = [] if log is None else None
        res = il_v7.train(_smoke_cfg(**base), [shard_root()], str(out),
                          log=logs.append if log is None else log, resume=resume)
        return res, logs

    def test_exact_weights_ema_default_and_model_option(self):
        import torch
        src = _ck(self.init)
        self.assertFalse(all(torch.equal(v, src["model_raw"][k]) for k, v in src["model_ema"].items()
                             if v.is_floating_point()))                  # EMA != raw in the source
        for weights, key in (("ema", "model_ema"), ("model", "model_raw")):
            out = _tmpd(self) / weights
            over = {} if weights == "ema" else {"init_weights": "model"}
            res, logs = self._warm(out, **over)
            c0 = _ck(out / "ckpt-0000000.pt")                            # step-0 eval checkpoint
            for k, v in src[key].items():
                self.assertTrue(torch.equal(c0["model_raw"][k], v.to(c0["model_raw"][k].dtype)), (weights, k))
                self.assertTrue(torch.equal(c0["model_ema"][k], v.float() if v.is_floating_point() else v),
                                (weights, k))
            self.assertEqual(c0["optimizer"]["state"], {})               # fresh optimizer
            self.assertEqual(c0["step"], 0)
            # init record: start event, run.json, every checkpoint
            start = _logs_json(logs, '{"event": "start"')[0]
            run = json.loads((out / "run.json").read_text())
            for rec in (start["init"], run["info"]["init"], c0["init"], _ck(out / "ckpt-0000003.pt")["init"]):
                self.assertEqual(rec["sha256"], self.init_sha)
                self.assertEqual(rec["path"], str(Path(self.init).resolve()))
                self.assertEqual(rec["weights"], weights)
                self.assertEqual(rec["source_step"], 6)
            self.assertEqual(run["config"]["init_from"], self.init)
            # LR schedule from step 0 (warmup, then cosine over this run's 3 steps)
            cfg = _smoke_cfg(max_steps=3, warmup_steps=2)
            recs = _logs_json(logs, '{"step"')
            self.assertEqual([r["step"] for r in recs], [1, 2, 3])
            for r in recs:
                self.assertAlmostEqual(r["lr"], il_v7.lr_at(r["step"], cfg, 3), places=15)
            self.assertEqual(res["steps"], 3)

    def test_eval_at_start_equals_standalone_evaluation(self):
        import torch
        out = _tmpd(self) / "w"
        _, logs = self._warm(out, max_steps=1)
        ev = _logs_json(logs, '{"event": "eval"')
        self.assertEqual([e["step"] for e in ev], [0, 1])
        e0 = ev[0]
        cfg = _smoke_cfg(init_from=self.init, eval_at_start=True, max_steps=1)
        src, _ = il_v7.ShardSource.open([shard_root()])
        tabs = il_v7.build_tables(cfg)
        val_items = il_v7.cap_val_items(src, il_v7.chunk_episodes(src, 1, cfg.max_seq_rows, cfg.burn_in_rows),
                                        cfg.val_max_games)
        val_ds = il_v7.EpisodeBatches(src, val_items, tabs, cfg, train=False)
        model = il_v7.load_policy(self.init)
        rep = il_v7.evaluate(model, il_v7.make_loader(val_ds, 0), torch.device("cpu"), torch.float32, cfg.loss,
                             cfg.eval_max_batches, opp_hand_reduce=cfg.aux_opp_hand_reduce)
        self.assertEqual(e0["selection_nll"], rep["selection_nll"])
        self.assertEqual(e0["terms"], rep["terms"])
        self.assertEqual(e0["gate_nll_by_stratum"], rep["gate_nll_by_stratum"])
        self.assertEqual(e0["batches"], rep["batches"])
        # the step-0 eval is a normal eval: history, best.json, early-stopping reference
        best = json.loads((out / "best.json").read_text())
        self.assertIn(0, [b[1] for b in best])
        # without eval_at_start there is no step-0 eval
        _, logs2 = self._warm(_tmpd(self) / "n", max_steps=1, eval_at_start=False)
        self.assertEqual([e["step"] for e in _logs_json(logs2, '{"event": "eval"')], [1])

    def test_exact_resume_of_warm_started_run(self):
        import torch
        init = _tmpd(self) / "init.pt"
        shutil.copyfile(self.init, init)
        over = {"init_from": str(init), "max_steps": 7, "eval_every": 3, "log_every": 1}
        a = _tmpd(self) / "a"
        ra, _ = self._warm(a, **over)
        b = _tmpd(self) / "b"

        def killer(m):
            if m.startswith('{"step": 4,'):
                os.kill(os.getpid(), signal.SIGTERM)
        rb, _ = self._warm(b, log=killer, **over)
        self.assertEqual((rb["stop_reason"], rb["steps"]), ("signal", 4))
        init.unlink()                                       # resume wins over init_from: never read again
        rc, _ = self._warm(b, resume=True, **over)
        self.assertEqual((rc["stop_reason"], rc["steps"], rc["resumes"]), ("total_steps", 7, [4]))
        fa, fb = _ck(a / "ckpt-0000007.pt"), _ck(b / "ckpt-0000007.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))
        self.assertEqual(fa["metrics"]["selection_nll"], fb["metrics"]["selection_nll"])
        self.assertEqual(fb["init"]["sha256"], self.init_sha)               # carried through the resume
        la = {h["step"]: h["loss"] for h in ra["history"] if "loss" in h}
        lc = {h["step"]: h["loss"] for h in rc["history"] if "loss" in h}
        for s in (5, 6, 7):
            self.assertEqual(la[s], lc[s])

    def test_lr_mult_group_lrs(self):
        import torch
        mult = {"gate.": 0.0, "ability_head.": 2.5, "aux_": 0.5, "cells.aux_threat.": 3.0}
        cfg_kw = {"lr_mult": mult, "max_steps": 6, "warmup_steps": 2, "eval_every": 0, "eval_at_start": True}
        seen = []
        real_step = torch.optim.AdamW.step

        def spy(opt, *a, **k):
            seen.append([(g.get("lr_mult"), g["lr"], g["weight_decay"]) for g in opt.param_groups])
            return real_step(opt, *a, **k)
        out = _tmpd(self) / "m"
        with mock.patch.object(torch.optim.AdamW, "step", spy):
            res, logs = self._warm(out, **cfg_kw)
        cfg = _smoke_cfg(**cfg_kw)
        self.assertEqual(len(seen), 6)
        for step, groups in enumerate(seen):
            base = il_v7.lr_at(step, cfg, 6)
            self.assertEqual({g[0] for g in groups}, {0.0, 0.5, 1.0, 2.5, 3.0})
            for m, lr, _ in groups:
                self.assertEqual(lr, base * m)
        self.assertGreater(il_v7.lr_at(5, cfg, 6), 0.0)
        start = _logs_json(logs, '{"event": "start"')[0]
        self.assertEqual(sum(g["params"] for g in start["lr_mult_groups"]),
                         len(list(il_v7.load_policy(self.init).parameters())))
        # multiplier 0 freezes the gate head (AdamW with lr 0 also skips decay); other heads move
        c0, c6 = _ck(out / "ckpt-0000000.pt")["model_raw"], _ck(out / "ckpt-0000006.pt")["model_raw"]
        for k in c0:
            if k.startswith("gate."):
                self.assertTrue(torch.equal(c0[k], c6[k]), k)
        for pre in ("ability_head.", "slot_head.", "cells.aux_threat.", "aux_opp_next.", "temporal."):
            self.assertTrue(any(not torch.equal(c0[k], c6[k]) for k in c0 if k.startswith(pre)), pre)
        # unknown prefix refused before training
        with self.assertRaises(ValueError):
            self._warm(_tmpd(self) / "x", lr_mult={"no_such_head.": 2.0})
        with self.assertRaises(ValueError):
            _smoke_cfg(lr_mult={"gate.": -1.0})

    def test_set_overrides_loss_and_lr_mult(self):
        cfg_path = _config("aminiformer_v3_warm.yaml")
        base = il_v7.load_yaml(Path(cfg_path).read_text())
        ov = il_v7.parse_set_overrides(base, ["loss.ability=2.0", "lr_mult.ability_head.=3",
                                              "lr_mult.cells.aux_threat.=0.5", "lr=1e-4",
                                              "init_weights=model", "eval_at_start=false", "model.dropout=0.2"])
        c = il_v7.TrainConfig.from_file(cfg_path, ov)
        ref = il_v7.TrainConfig.from_file(cfg_path)
        self.assertEqual(c.loss["ability"], 2.0)
        self.assertEqual(c.loss["gate"], ref.loss["gate"])                      # other terms kept
        self.assertEqual(c.loss["value"], ref.loss["value"])
        self.assertEqual(c.lr_mult, {"ability_head.": 3.0, "cells.aux_threat.": 0.5})
        self.assertEqual((c.lr, c.init_weights, c.eval_at_start), (1e-4, "model", False))
        self.assertEqual(c.model["dropout"], 0.2)
        self.assertEqual(c.model["d_model"], ref.model["d_model"])
        # the warm-start config: the base config's model and recipe + the warm-start keys
        v3 = il_v7.TrainConfig.from_file(cfg_path)
        v2_path = _config("aminiformer_v1_v2.yaml")
        v2 = il_v7.TrainConfig.from_file(v2_path)
        self.assertEqual(v3.model, v2.model)
        self.assertEqual((v3.init_weights, v3.eval_at_start, v3.lr_mult), ("ema", True, {}))
        self.assertTrue(v3.init_from)
        for k in ("batch_rows", "micro_batch_rows", "max_batch_tokens", "ema_decay", "grad_clip", "weight_decay",
                  "val_games_file", "loss", "aux_opp_hand_reduce", "seed", "tables"):
            self.assertEqual(getattr(v3, k), getattr(v2, k), k)
        # the CLI: --init-from / --init-weights / --eval-at-start / --set reach the config
        got = {}

        def fake_train(cfg, data, out, **kw):
            got["cfg"] = cfg
            return {"steps": 0, "stop_reason": "x"}
        with mock.patch.object(il_v7, "train", fake_train), open(os.devnull, "w") as dn, \
                contextlib.redirect_stdout(dn):
            il_v7.main(["--config", v2_path, "--data", "x", "--out", "y",
                        "--init-from", "/tmp/i.pt", "--init-weights", "model", "--eval-at-start",
                        "--set", "lr_mult.aux_=0.5", "loss.cell=1.5"])
        c = got["cfg"]
        self.assertEqual((c.init_from, c.init_weights, c.eval_at_start), ("/tmp/i.pt", "model", True))
        self.assertEqual((c.lr_mult, c.loss["cell"], c.loss["slot"]), ({"aux_": 0.5}, 1.5, v2.loss["slot"]))
        with mock.patch.object(il_v7, "train", fake_train), open(os.devnull, "w") as dn, \
                contextlib.redirect_stdout(dn):
            il_v7.main(["--config", v2_path, "--data", "x", "--out", "y"])
        c = got["cfg"]
        self.assertEqual((c.init_from, c.init_weights, c.eval_at_start, c.lr_mult), ("", "ema", False, {}))

    def test_resume_config_diff_tolerates_pre_v3_checkpoints(self):
        from dataclasses import asdict
        cfg = asdict(_smoke_cfg())
        old = {k: v for k, v in cfg.items() if k not in ("init_from", "init_weights", "eval_at_start", "lr_mult")}
        self.assertEqual(il_v7._config_diff(cfg, old), [])                # absent keys = their defaults
        self.assertEqual(il_v7._config_diff(asdict(_smoke_cfg(lr_mult={"gate.": 2.0})), old), ["lr_mult"])
        self.assertEqual(il_v7._config_diff(asdict(_smoke_cfg(eval_at_start=True)), cfg), ["eval_at_start"])

    def test_refusals(self):
        import torch
        from amini.nn.aminiformer import AminiFormer, AminiFormerConfig
        cfg = _smoke_cfg(init_from=self.init)
        src = _ck(self.init)
        rec = src["tables"]
        self.assertEqual(rec["kind"], "synthetic")
        d = _tmpd(self)

        def variant(name, fn):
            ck = copy.deepcopy(src)
            fn(ck)
            p = d / name
            torch.save(ck, p)
            return str(p)

        def try_load(path, **over):
            c = _smoke_cfg(init_from=path, **over)
            return il_v7.warm_start(AminiFormer(AminiFormerConfig.from_dict(c.model)), c, rec)

        self.assertEqual(try_load(self.init)["sha256"], self.init_sha)
        files = {"kind": "files", "files": {"card_vocab": {"path": "a.json", "sha256": "a" * 64}}}
        with self.assertRaisesRegex(ValueError, "tokenizer-table"):
            try_load(variant("t1.pt", lambda ck: ck.__setitem__("tables", files)))
        with self.assertRaisesRegex(ValueError, "tokenizer-table"):
            try_load(variant("t2.pt", lambda ck: ck.__setitem__("tables", None)))
        other = {"kind": "files", "files": {"card_vocab": {"path": "a.json", "sha256": "b" * 64}}}
        c = _smoke_cfg(init_from=variant("t3.pt", lambda ck: ck.__setitem__("tables", files)))
        with self.assertRaisesRegex(ValueError, "tokenizer-table"):            # file pins differ
            il_v7.warm_start(AminiFormer(AminiFormerConfig.from_dict(c.model)), c, other)
        self.assertEqual(il_v7._tables_digest_map(files), {"card_vocab": "a" * 64})
        with self.assertRaisesRegex(ValueError, "shape/function"):            # d_lnk 16 -> 8
            try_load(self.init, model=dict(cfg.model, d_lnk=8))
        k0 = next(iter(src["model_ema"]))
        with self.assertRaisesRegex(ValueError, "missing"):
            try_load(variant("m.pt", lambda ck: ck["model_ema"].pop(k0)))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            try_load(variant("u.pt", lambda ck: ck["model_ema"].__setitem__("bogus.weight", torch.zeros(1))))
        kw = [k for k, v in src["model_ema"].items() if v.ndim == 2][0]
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            try_load(variant("s.pt", lambda ck: ck["model_ema"].__setitem__(kw, ck["model_ema"][kw][:, :-1])))
        with self.assertRaisesRegex(ValueError, "model_raw"):
            try_load(variant("r.pt", lambda ck: ck.pop("model_raw")), init_weights="model")
        with self.assertRaises(FileNotFoundError):
            try_load(str(d / "none.pt"))
        with self.assertRaises(ValueError):
            _smoke_cfg(init_weights="raw")
        # tolerated difference (dropout) is recorded
        r = try_load(self.init, model=dict(cfg.model, dropout=0.0))
        self.assertEqual(r["model_config_differences"],
                         {"dropout": {"run": 0.0, "checkpoint": src["model_config"]["dropout"]}})
        # through train(): refused before any step, no checkpoint written
        out = d / "run"
        with self.assertRaisesRegex(ValueError, "tokenizer-table"):
            il_v7.train(_smoke_cfg(init_from=variant("tab4.pt", lambda ck: ck.__setitem__("tables", files))),
                        [shard_root()], str(out), log=lambda m: None)
        self.assertFalse(list(out.glob("ckpt-*.pt")) if out.exists() else [])


def _reference_il_v7():
    """The trainer before the warm-start keys (commit 6473d06), as a separate module: from the file named
    by IL_V7_REF_FILE, else from git.  None when neither is available."""

    import importlib.util
    src = None
    f = os.environ.get("IL_V7_REF_FILE")
    if f and Path(f).is_file():
        src = Path(f).read_text()
    else:
        try:
            src = subprocess.run(["git", "-C", str(ROOT), "show", "6473d06:src/amini/train/il_v7.py"],
                                 capture_output=True, text=True, check=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    d = Path(tempfile.mkdtemp(prefix="a0-ref-"))
    p = d / "il_v7_head_6473d06.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("il_v7_head_6473d06", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                       # dataclasses look the module up while defining
    try:
        spec.loader.exec_module(mod)
    finally:
        shutil.rmtree(d, True)
    return mod


class TestDefaultsBitIdentical(unittest.TestCase):
    def test_defaults_bit_identical_to_head(self):
        """Without the warm-start keys a run equals the earlier trainer (6473d06): 32 steps, same seed, identical
        per-step losses and terms, eval metrics and final raw / EMA parameters."""
        import torch
        ref = _reference_il_v7()
        if ref is None:
            self.skipTest("needs a copy of the earlier trainer (set IL_V7_REF_FILE)")
        self.assertFalse(hasattr(ref.TrainConfig, "__dataclass_fields__") and
                         "lr_mult" in ref.TrainConfig.__dataclass_fields__)
        over = {"max_steps": 32, "eval_every": 16, "num_workers": 0, "batch_rows": 400, "log_every": 1,
                "eval_max_batches": 2, "warmup_steps": 8}
        runs = {}
        for name, mod in (("head", ref), ("new", il_v7)):
            out = _tmpd(self) / name
            cfg = mod.TrainConfig.from_file(SMOKE, dict(over))
            res = mod.train(cfg, [shard_root()], str(out), log=lambda m: None)
            runs[name] = (res, _ck(out / "ckpt-0000032.pt"))
        (ra, ca), (rb, cb) = runs["head"], runs["new"]
        ha = [h for h in ra["history"] if "loss" in h]
        hb = [h for h in rb["history"] if "loss" in h]
        self.assertEqual(len(ha), 32)
        drop = ("rows_s",)
        for x, y in zip(ha, hb):
            self.assertEqual({k: v for k, v in x.items() if k not in drop},
                             {k: v for k, v in y.items() if k not in drop})
        ea = [h["eval"]["selection_nll"] for h in ra["history"] if "eval" in h]
        eb = [h["eval"]["selection_nll"] for h in rb["history"] if "eval" in h]
        self.assertEqual(ea, eb)
        self.assertEqual(len(ea), 2)
        for key in ("model_raw", "model_ema"):
            self.assertEqual(set(ca[key]), set(cb[key]))
            for k, v in ca[key].items():
                self.assertTrue(torch.equal(v, cb[key][k]), (key, k))
        self.assertEqual(len(ca["optimizer"]["param_groups"]), 2)
        self.assertEqual(len(cb["optimizer"]["param_groups"]), 2)
        self.assertNotIn("lr_mult", cb["optimizer"]["param_groups"][0])
        self.assertIsNone(cb["init"])


class TestSpeedSwitches(unittest.TestCase):
    """Speed switches: pack_tokens / compile_mode / compile_dynamic / pack_multiple."""

    def test_config_validation(self):
        c = _smoke_cfg()
        self.assertEqual((c.pack_tokens, c.compile_mode, c.compile_dynamic, c.pack_multiple), (False, "", False, 0))
        for bad in ({"compile_mode": "default"},                                   # needs pack_tokens
                    {"pack_tokens": True, "compile_mode": "default"},              # static needs a ladder
                    {"pack_tokens": True, "compile_mode": "reduce-overhead", "compile_dynamic": True},
                    {"pack_multiple": 64},                                         # ladder needs packing
                    {"pack_tokens": True, "pack_multiple": -8}):
            with self.assertRaises(ValueError, msg=bad):
                _smoke_cfg(**bad)
        with self.assertRaises(ValueError):                                        # stage-S ladder missing
            _smoke_cfg(pack_tokens=True, compile_mode="default", pack_multiple=4096)
        with self.assertRaises(ValueError):
            _smoke_cfg(cell_row_multiple=-1)
        ok = _smoke_cfg(pack_tokens=True, compile_mode="default", pack_multiple=4096, cell_row_multiple=64)
        self.assertEqual((ok.pack_multiple, ok.cell_row_multiple), (4096, 64))
        _smoke_cfg(pack_tokens=True, compile_mode="max-autotune-no-cudagraphs", compile_dynamic=True)

    def test_apply_speed_settings(self):
        model = tiny_model(double=False)
        il_v7.apply_speed_settings(model, _smoke_cfg(pack_tokens=True, pack_multiple=64, cell_row_multiple=8))
        se, cd = model.step_encoder, model.cells
        self.assertEqual((se.packed, se.pad_multiple, se.compiled, cd.pad_rows, cd.compiled), (True, 64, None, 8, None))
        il_v7.apply_speed_settings(model, _smoke_cfg())
        self.assertEqual((se.packed, se.pad_multiple, se.compiled, cd.pad_rows, cd.compiled), (False, 0, None, 0, None))

    def test_packed_run_matches_default(self):
        """Same seed, no dropout: a packed run (with a token ladder) follows the default run's trajectory
        (per-step losses, eval NLL, final weights) to float rounding; the batches are identical."""
        import torch
        over = {"max_steps": 8, "eval_every": 8, "warmup_steps": 2}
        runs = {}
        for name, extra in (("pad", {}), ("pack", {"pack_tokens": True, "pack_multiple": 32,
                                                     "cell_row_multiple": 16})):
            o = dict(over)
            o.update(extra)
            cfg = _smoke_cfg(**o)
            cfg.model = dict(cfg.model, dropout=0.0)
            out = _tmpd(self) / name
            logs = []
            res = il_v7.train(cfg, [shard_root()], str(out), log=logs.append)
            runs[name] = (res, _ck(out / "ckpt-0000008.pt"), _logs_json(logs, '{"event": "start"'))
        (ra, ca, sa), (rb, cb, sb) = runs["pad"], runs["pack"]
        ha = [h for h in ra["history"] if "loss" in h]
        hb = [h for h in rb["history"] if "loss" in h]
        self.assertEqual(len(ha), 8)
        for x, y in zip(ha, hb):
            self.assertEqual(x["micro"], y["micro"])
            self.assertAlmostEqual(x["loss"], y["loss"], delta=1e-4 * max(1.0, abs(x["loss"])))
        ea = [h["eval"]["selection_nll"] for h in ra["history"] if "eval" in h]
        eb = [h["eval"]["selection_nll"] for h in rb["history"] if "eval" in h]
        self.assertEqual(len(ea), 1)
        self.assertAlmostEqual(ea[0], eb[0], delta=1e-4)
        # scalar biases added to every logit of one softmax have an analytically zero gradient: AdamW turns
        # their float-noise gradients into lr-sized random walks in BOTH runs, so they are not compared
        shift_invariant = ("slot_head.v.bias", "ability_head.v.bias", "cells.logit.bias")
        for k, v in ca["model_raw"].items():
            if k in shift_invariant:
                continue
            self.assertLess(float((v - cb["model_raw"][k]).abs().max()), 1e-3, k)
        self.assertTrue(cb["train_config"]["pack_tokens"])
        self.assertNotIn("packed", cb["model_config"])           # runtime switch, not a model field
        self.assertEqual(set(ca["model_ema"]), set(cb["model_ema"]))

    def test_packed_resume_exact(self):
        import torch
        cfg = _smoke_cfg(max_steps=6, eval_every=3, pack_tokens=True)
        a = _tmpd(self) / "a"
        il_v7.train(cfg, [shard_root()], str(a), log=lambda m: None)
        b = _tmpd(self) / "b"

        def killer(m):
            if m.startswith('{"step": 3,'):
                os.kill(os.getpid(), signal.SIGTERM)
        rb = il_v7.train(cfg, [shard_root()], str(b), log=killer)
        self.assertEqual(rb["stop_reason"], "signal")
        c = _tmpd(self) / "c"                                # speed keys are resume-free: resume unpacked
        shutil.copytree(b, c)
        rc = il_v7.train(_smoke_cfg(max_steps=6, eval_every=3), [shard_root()], str(c),
                         log=lambda m: None, resume=True)
        self.assertEqual((rc["stop_reason"], rc["steps"]), ("total_steps", 6))
        with self.assertRaises(ValueError):                  # an objective key still is not
            il_v7.train(_smoke_cfg(max_steps=6, eval_every=3, pack_tokens=True, lr=0.002), [shard_root()],
                        str(b), log=lambda m: None, resume=True)
        il_v7.train(cfg, [shard_root()], str(b), log=lambda m: None, resume=True)
        fa, fb = _ck(a / "ckpt-0000006.pt"), _ck(b / "ckpt-0000006.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))

    def test_v3_fast_config_is_v2_plus_speed_keys(self):
        from dataclasses import asdict
        v2 = asdict(il_v7.TrainConfig.from_file(_config("aminiformer_v1_v2.yaml")))
        fast = asdict(il_v7.TrainConfig.from_file(_config("aminiformer_v3_fast.yaml")))
        speed = {"pack_tokens", "compile_mode", "compile_dynamic", "pack_multiple", "cell_row_multiple"}
        diff = {k for k in v2 if v2[k] != fast[k]}
        self.assertEqual(diff - speed, {"name"})
        self.assertTrue(fast["pack_tokens"])
        self.assertEqual(fast["init_from"], "")                     # warm-start keys are commented out
        self.assertTrue(speed <= set(il_v7.RESUME_FREE_KEYS))

    def test_load_policy_packed(self):
        import torch
        out = _tmpd(self) / "r"
        il_v7.train(_smoke_cfg(max_steps=3, eval_every=3), [shard_root()], str(out), log=lambda m: None)
        ck = str(out / "ckpt-0000003.pt")
        a = il_v7.load_policy(ck)
        b = il_v7.load_policy(ck, packed=True)
        self.assertFalse(a.step_encoder.packed)
        self.assertTrue(b.step_encoder.packed)
        cfg = _smoke_cfg(max_steps=3)
        src = source()
        items = il_v7.chunk_episodes(src, 1, cfg.max_seq_rows, cfg.burn_in_rows)
        ds = il_v7.EpisodeBatches(src, items, tables(), cfg, train=False)
        dev = torch.device("cpu")
        ra = il_v7.evaluate(a, il_v7.make_loader(ds, 0), dev, torch.float32, cfg.loss)
        rb = il_v7.evaluate(b, il_v7.make_loader(ds, 0), dev, torch.float32, cfg.loss)
        self.assertAlmostEqual(ra["selection_nll"], rb["selection_nll"], delta=1e-5)
        for k in ra["terms"]:
            self.assertAlmostEqual(ra["terms"][k], rb["terms"][k], delta=1e-5, msg=k)


# ---------------------------------------------------------------- exclude_friendly / population_input
V4_REF_COMMIT = "ecda7ea"            # the trainer before these keys (reference for the bit-identity test)


def _ref_source(env: str, path: str) -> "str | None":
    f = os.environ.get(env)
    if f and Path(f).is_file():
        return Path(f).read_text()
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", f"{V4_REF_COMMIT}:{path}"],
                              capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _load_module(name: str, text: str):
    import importlib.util
    d = Path(tempfile.mkdtemp(prefix="v4-ref-"))
    p = d / f"{name.rsplit('.', 1)[-1]}.py"
    p.write_text(text)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        shutil.rmtree(d, True)
    return mod


def _reference_v4_head():
    """(il_v7, aminiformer) of V4_REF_COMMIT as separate modules (IL_V7_V4REF_FILE /
    AMINIFORMER_REF_FILE, else git), or None."""

    a = _ref_source("IL_V7_V4REF_FILE", "src/amini/train/il_v7.py")
    b = _ref_source("AMINIFORMER_REF_FILE", "src/amini/nn/aminiformer.py")
    if a is None or b is None:
        return None
    return _load_module("il_v7_ref_v4", a), _load_module("amini.nn._aminiformer_ref_v4t", b)


# synthetic populations of the 6 games (5 = friendly) and splits (1 = val): friendly = games 0, 2, 4
_V4_POP = [5, 1, 5, 4, 5, 3]
_V4_SPLIT = [0, 1, 1, 0, 0, 0]
FRIENDLY_ID = 5


def _friendly_root(case) -> str:
    """A copy of the synthetic shards whose games have the populations / splits above."""

    root = _tmpd(case) / "fr"
    shutil.copytree(shard_root(), root)
    for gp in root.glob("shards/*/games.npy"):
        g = np.load(gp, allow_pickle=False)
        assert len(g) == len(_V4_POP)
        g["population"] = _V4_POP
        g["split"] = _V4_SPLIT
        np.save(gp, g)
    return str(root)


def _start(logs):
    return [json.loads(l) for l in logs if l.startswith('{"event": "start"')][0]


class TestExcludeFriendly(unittest.TestCase):
    @unittest.skipUnless(HAVE_VOCAB, NEEDS_VOCAB)
    def test_friendly_ids_from_vocab(self):
        from amini.v7 import vocab as V
        self.assertEqual(il_v7.friendly_population_ids(), frozenset({FRIENDLY_ID}))
        self.assertEqual(V.POPULATIONS[FRIENDLY_ID], "friendly")
        for asset in ("Friendly", "Showdown_Friendly"):
            self.assertEqual(V.POPULATIONS.index(V.population_name(asset)), FRIENDLY_ID)
        for asset in ("Ladder", "Ranked1v1_X", "Tournament", "Challenge_X", "Something", None):
            self.assertNotEqual(V.POPULATIONS.index(V.population_name(asset)), FRIENDLY_ID)

    def test_friendly_games_and_drop(self):
        root = _friendly_root(self)
        src, _ = il_v7.ShardSource.open([root])
        fr = il_v7.friendly_games(src)
        self.assertEqual(fr, {(0, 0), (0, 2), (0, 4)})
        items = il_v7.chunk_episodes(src, 0, 512, 192)
        kept, d = il_v7.drop_games(items, fr)
        self.assertEqual(kept, [it for it in items if it[0][1] not in (0, 2, 4)])       # order kept
        self.assertEqual(d["games"], 2)                                   # train friendly: games 0, 4
        self.assertEqual(d["items"], len(items) - len(kept))
        self.assertEqual(d["rows"], sum(it[3] - it[2] for it in items if it[0][1] in (0, 4)))
        self.assertEqual(il_v7.drop_games(items, set()), (items, {"games": 0, "items": 0, "rows": 0}))
        g = src.shards[0]["games"]
        bad = np.array(g)
        bad["population"][1] = 7
        src.shards[0] = dict(src.shards[0], games=bad)
        with self.assertRaises(ValueError):
            il_v7.friendly_games(src)

    def test_train_drops_friendly_train_and_val(self):
        root = _friendly_root(self)
        src, _ = il_v7.ShardSource.open([root])
        tags = {gi: il_v7._tag_text(src, 0, gi) for gi in range(6)}
        val_file = _tmpd(self) / "val.txt"
        val_file.write_text(f"# pinned\n{tags[1]}\n{tags[2]}\n")
        all_train = il_v7.chunk_episodes(src, 0, 512, 192)
        runs = {}
        for name, over in (("off", {}), ("on", {"exclude_friendly": True}),
                           ("on_pin", {"exclude_friendly": True, "val_games_file": str(val_file)}),
                           ("off_pin", {"val_games_file": str(val_file)})):
            logs = []
            il_v7.train(_smoke_cfg(max_steps=1, eval_every=0, **over), [root], str(_tmpd(self) / name),
                        log=logs.append)
            runs[name] = _start(logs)
        off, on, pin, offp = runs["off"], runs["on"], runs["on_pin"], runs["off_pin"]
        self.assertNotIn("exclude_friendly", off)                         # off: start event unchanged
        self.assertNotIn("exclude_friendly", offp)
        self.assertEqual(off["val_games"], 2)
        self.assertEqual(offp["val_games"], 2)
        rec = on["exclude_friendly"]
        self.assertEqual(rec["population"], "friendly")
        self.assertEqual(rec["population_ids"], [FRIENDLY_ID])
        self.assertEqual(rec["games_in_source"], 3)
        self.assertEqual(rec["train"]["games"], 2)
        self.assertEqual(rec["val"]["games"], 1)
        self.assertEqual(on["val_games"], 1)
        self.assertEqual(on["train_items"], off["train_items"] - rec["train"]["items"])
        self.assertEqual(on["train_rows"], off["train_rows"] - rec["train"]["rows"])
        self.assertEqual(on["train_items"], len([it for it in all_train if it[0][1] not in (0, 4)]))
        self.assertNotIn("val_games_file", rec)
        self.assertEqual(pin["exclude_friendly"]["val_games_file"], {"listed": 2, "dropped": 1, "kept": 1})
        self.assertEqual(pin["val_games"], 1)
        self.assertEqual(pin["val_games_file_sha256"], hashlib.sha256(val_file.read_bytes()).hexdigest())
        # a listed game that is not a val game of the generation is still refused (friendly or not)
        val_file.write_text(f"{tags[1]}\n{tags[0]}\n")
        with self.assertRaises(ValueError):
            il_v7.train(_smoke_cfg(max_steps=1, eval_every=0, exclude_friendly=True,
                                   val_games_file=str(val_file)), [root], str(_tmpd(self) / "bad"),
                        log=lambda m: None)

    @unittest.skipUnless(HAVE_VOCAB, NEEDS_VOCAB)
    def test_real_shard_count(self):
        """On a real shard: friendly_games == the games whose recorded mode asset maps to friendly
        (independently through the frozen game-mode vocab)."""

        from amini.v7 import vocab as V
        d = os.environ.get("V7_REAL_SHARD", "")
        if not d or not Path(d, "games.npy").is_file():
            self.skipTest("set V7_REAL_SHARD to a real v7 shard directory")
        src = il_v7.ShardSource([d])
        modes = V.GameModeVocab.from_file(ROOT / "data/game_mode_vocab_v7.json")
        g = src.shards[0]["games"]
        mr = np.asarray(g["mode_recorded"], dtype=np.int64)
        if int(mr.max()) >= len(modes.modes):
            self.skipTest("shard uses mode ids newer than data/game_mode_vocab_v7.json")
        want = {(0, gi) for gi in range(len(g))
                if V.population_name(modes.modes[int(mr[gi])]["asset"]) == "friendly"}
        got = il_v7.friendly_games(src)
        self.assertEqual(got, want)
        self.assertGreater(len(got), 0)
        pop = np.asarray(g["population"], dtype=np.int64)
        for gi in range(len(g)):                           # GAME.population agrees with the mode's population
            name = modes.modes[int(mr[gi])]["population"]
            self.assertEqual(V.POPULATIONS[pop[gi]], name, gi)
        print(f"real shard {Path(d).name}: {len(g)} games, {len(got)} friendly")


class TestV4Keys(unittest.TestCase):
    def test_defaults_bit_identical_to_v4_head(self):
        """Without the exclude_friendly / population keys (and with them at their defaults) a run equals the earlier trainer + model
        (ecda7ea): 32 steps, identical per-step losses / terms, eval metrics, raw / EMA parameters and
        model_config."""
        import torch
        ref = _reference_v4_head()
        if ref is None:
            self.skipTest("needs a copy of the earlier trainer and model (set IL_V7_V4REF_FILE and AMINIFORMER_REF_FILE)")
        ref_il, ref_model = ref
        self.assertNotIn("exclude_friendly", ref_il.TrainConfig.__dataclass_fields__)
        over = {"max_steps": 32, "eval_every": 16, "num_workers": 0, "batch_rows": 400, "log_every": 1,
                "eval_max_batches": 2, "warmup_steps": 8}
        runs = {}
        for name, mod, extra in (("head", ref_il, {}), ("new", il_v7, {}),
                                 ("new_explicit", il_v7, {"exclude_friendly": False})):
            out = _tmpd(self) / name
            cfg = mod.TrainConfig.from_file(SMOKE, dict(over, **extra))
            ctx = mock.patch.dict(sys.modules, {"amini.nn.aminiformer": ref_model}) \
                if name == "head" else contextlib.nullcontext()
            with ctx:
                res = mod.train(cfg, [shard_root()], str(out), log=lambda m: None)
            runs[name] = (res, _ck(out / "ckpt-0000032.pt"))
        ra, ca = runs["head"]
        for other in ("new", "new_explicit"):
            rb, cb = runs[other]
            ha = [h for h in ra["history"] if "loss" in h]
            hb = [h for h in rb["history"] if "loss" in h]
            self.assertEqual(len(ha), 32)
            for x, y in zip(ha, hb):
                self.assertEqual({k: v for k, v in x.items() if k != "rows_s"},
                                 {k: v for k, v in y.items() if k != "rows_s"})
            ea = [h["eval"]["selection_nll"] for h in ra["history"] if "eval" in h]
            eb = [h["eval"]["selection_nll"] for h in rb["history"] if "eval" in h]
            self.assertEqual(ea, eb)
            self.assertEqual(len(ea), 2)
            self.assertEqual(ca["model_config"], cb["model_config"])
            self.assertNotIn("population_input", cb["model_config"])
            for key in ("model_raw", "model_ema"):
                self.assertEqual(set(ca[key]), set(cb[key]))
                for k, v in ca[key].items():
                    self.assertTrue(torch.equal(v, cb[key][k]), (other, key, k))

    def test_population_off_checkpoint_round_trip_and_exact_resume(self):
        import torch
        root = _friendly_root(self)
        over = {"max_steps": 7, "eval_every": 3, "exclude_friendly": True,
                "model": dict(_smoke_cfg().model, population_input=False)}
        a = _tmpd(self) / "a"
        ra = il_v7.train(_smoke_cfg(**over), [root], str(a), log=lambda m: None)
        self.assertEqual(ra["stop_reason"], "total_steps")
        ck = _ck(a / "ckpt-0000007.pt")
        self.assertIs(ck["model_config"]["population_input"], False)
        self.assertIs(ck["train_config"]["exclude_friendly"], True)
        pol = il_v7.load_policy(str(a / "ckpt-0000007.pt"))
        self.assertFalse(pol.cfg.population_input)
        self.assertFalse(pol.step_encoder.embed.population_input)
        b = _tmpd(self) / "b"

        def killer(m):
            if m.startswith('{"step": 4,'):
                os.kill(os.getpid(), signal.SIGTERM)
        rb = il_v7.train(_smoke_cfg(**over), [root], str(b), log=killer)
        self.assertEqual((rb["stop_reason"], rb["steps"]), ("signal", 4))
        with self.assertRaisesRegex(ValueError, "exclude_friendly"):     # the data would change
            il_v7.train(_smoke_cfg(**dict(over, exclude_friendly=False)), [root], str(b),
                        log=lambda m: None, resume=True)
        with self.assertRaisesRegex(ValueError, "model"):               # the function would change
            il_v7.train(_smoke_cfg(**dict(over, model=_smoke_cfg().model)), [root], str(b),
                        log=lambda m: None, resume=True)
        rc = il_v7.train(_smoke_cfg(**over), [root], str(b), log=lambda m: None, resume=True)
        self.assertEqual((rc["stop_reason"], rc["steps"], rc["resumes"]), ("total_steps", 7, [4]))
        fa, fb = _ck(a / "ckpt-0000007.pt"), _ck(b / "ckpt-0000007.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))
        self.assertEqual(fa["metrics"]["selection_nll"], fb["metrics"]["selection_nll"])
        la = {h["step"]: h["loss"] for h in ra["history"] if "loss" in h}
        lc = {h["step"]: h["loss"] for h in rc["history"] if "loss" in h}
        for s in (5, 6, 7):
            self.assertEqual(la[s], lc[s])
        # pop_null_p is irrelevant with the population input off: the same run with another value is identical
        d = _tmpd(self) / "d"
        rd = il_v7.train(_smoke_cfg(**dict(over, pop_null_p=0.9)), [root], str(d), log=lambda m: None)
        fd = _ck(d / "ckpt-0000007.pt")
        for k, v in fa["model_ema"].items():
            self.assertTrue(torch.equal(v, fd["model_ema"][k]), k)
        self.assertEqual([h["loss"] for h in ra["history"] if "loss" in h],
                         [h["loss"] for h in rd["history"] if "loss" in h])

    def test_warm_start_into_population_off(self):
        """A warm start from a population-input checkpoint into population_input false is allowed
        (same shapes; the difference is recorded) and loads the weights exactly."""
        import torch
        src = _tmpd(self) / "src"
        il_v7.train(_smoke_cfg(max_steps=3, eval_every=3), [shard_root()], str(src), log=lambda m: None)
        init = str(src / "ckpt-0000003.pt")
        logs = []
        out = _tmpd(self) / "w"
        il_v7.train(_smoke_cfg(init_from=init, eval_at_start=True, max_steps=1, eval_every=0,
                               model=dict(_smoke_cfg().model, population_input=False)),
                    [shard_root()], str(out), log=logs.append)
        st = _start(logs)
        self.assertEqual(st["init"]["model_config_differences"],
                         {"population_input": {"run": False, "checkpoint": None}})
        c0, s0 = _ck(out / "ckpt-0000000.pt"), _ck(init)
        for k, v in s0["model_ema"].items():
            self.assertTrue(torch.equal(v, c0["model_ema"][k]), k)

# ------------------------------------------------------------------ training-loop CPU overhead
S5_REF_COMMIT = "4f0070c"            # the trainer before these keys (reference for the bit-identity test)
S5_KEYS = ("defer_stats", "host_index", "ema_foreach")


def _reference_s5_head():
    """(il_v7, aminiformer) of S5_REF_COMMIT as separate modules (IL_V7_S5REF_FILE / AMINIFORMER_S5REF_FILE,
    else git), or None."""

    out = []
    for env, path in (("IL_V7_S5REF_FILE", "src/amini/train/il_v7.py"),
                      ("AMINIFORMER_S5REF_FILE", "src/amini/nn/aminiformer.py")):
        f = os.environ.get(env)
        if f and Path(f).is_file():
            out.append(Path(f).read_text())
            continue
        try:
            out.append(subprocess.run(["git", "-C", str(ROOT), "show", f"{S5_REF_COMMIT}:{path}"],
                                      capture_output=True, text=True, check=True, timeout=30).stdout)
        except (OSError, subprocess.SubprocessError):
            return None
    return _load_module("il_v7_ref_s5", out[0]), _load_module("amini.nn._aminiformer_ref_s5", out[1])


def _s5_run(case, name, mod=None, model_mod=None, **over):
    """A 32-step smoke run (2 micro-batches per step, the smoke config's dropout, eval at 16 and 32)."""
    mod = mod or il_v7
    base = {"max_steps": 32, "eval_every": 16, "num_workers": 0, "batch_rows": 400, "micro_batch_rows": 200,
            "log_every": 1, "eval_max_batches": 2, "warmup_steps": 8}
    base.update(over)
    root = shard_root()                       # the private dataset first: the caller is skipped without it
    cfg = mod.TrainConfig.from_file(SMOKE, base) if mod is not il_v7 else _smoke_cfg(**base)
    out = _tmpd(case) / name
    # torch.manual_seed imports torch._dynamo lazily: import it first, or patch.dict would drop it again on
    # exit and the next import would re-register its op library (RuntimeError)
    import torch._dynamo  # noqa: F401
    ctx = mock.patch.dict(sys.modules, {"amini.nn.aminiformer": model_mod}) \
        if model_mod is not None else contextlib.nullcontext()
    with ctx:
        res = mod.train(cfg, [root], str(out), log=lambda m: None)
    return res, _ck(out / f"ckpt-{base['max_steps']:07d}.pt")


class TestV5Speed(unittest.TestCase):
    """Training-loop speed keys: defer_stats / host_index / ema_foreach (same arithmetic) and adamw_impl (kernel only)."""

    def assert_runs_equal(self, a, b, tag):
        import torch
        (ra, ca), (rb, cb) = a, b
        ha = [h for h in ra["history"] if "loss" in h]
        hb = [h for h in rb["history"] if "loss" in h]
        self.assertGreater(len(ha), 0, tag)
        self.assertEqual(len(ha), len(hb), tag)
        for x, y in zip(ha, hb):
            self.assertEqual({k: v for k, v in x.items() if k != "rows_s"},
                             {k: v for k, v in y.items() if k != "rows_s"}, tag)
        ea = [h["eval"] for h in ra["history"] if "eval" in h]
        eb = [h["eval"] for h in rb["history"] if "eval" in h]
        self.assertEqual(len(ea), 2, tag)
        for x, y in zip(ea, eb):                      # json: NaN entries (empty calibration bins) compare equal
            self.assertEqual(json.dumps({k: v for k, v in x.items() if k != "eval_seconds"}, sort_keys=True),
                             json.dumps({k: v for k, v in y.items() if k != "eval_seconds"}, sort_keys=True), tag)
        self.assertEqual(ra["grad_norm"], rb["grad_norm"], tag)
        self.assertEqual(ca["model_config"], cb["model_config"], tag)
        for key in ("model_raw", "model_ema"):
            self.assertEqual(set(ca[key]), set(cb[key]), tag)
            for k, v in ca[key].items():
                self.assertTrue(torch.equal(v, cb[key][k]), (tag, key, k))
        for (ka, va), (kb, vb) in zip(sorted(ca["optimizer"]["state"].items()),
                                      sorted(cb["optimizer"]["state"].items())):
            for f in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(va[f], vb[f]), (tag, ka, f))

    def test_config_keys(self):
        c = _smoke_cfg()
        self.assertEqual((c.defer_stats, c.host_index, c.ema_foreach, c.adamw_impl), (False, False, False, "auto"))
        self.assertTrue(set(S5_KEYS) | {"adamw_impl"} <= set(il_v7.RESUME_FREE_KEYS))
        for bad in ({"adamw_impl": "apex"}, {"defer_stats": 2}, {"host_index": "yes"}):
            with self.assertRaises(ValueError, msg=bad):
                _smoke_cfg(**bad)
        ok = _smoke_cfg(defer_stats=True, host_index=True, ema_foreach=True, adamw_impl="foreach")
        self.assertEqual((ok.defer_stats, ok.host_index, ok.ema_foreach, ok.adamw_impl), (True, True, True, "foreach"))

    def test_defaults_bit_identical_to_v5_head(self):
        """Without the speed keys (and with them explicitly at their defaults) a run equals the earlier trainer +
        model (4f0070c): 32 steps x 2 micro-batches, identical per-step losses / terms / grad norms, eval
        reports, raw / EMA parameters and AdamW moments."""
        ref = _reference_s5_head()
        if ref is None:
            self.skipTest("needs a copy of the earlier trainer and model (set IL_V7_S5REF_FILE and AMINIFORMER_S5REF_FILE)")
        ref_il, ref_model = ref
        self.assertNotIn("defer_stats", ref_il.TrainConfig.__dataclass_fields__)
        head = _s5_run(self, "head", ref_il, ref_model)
        self.assert_runs_equal(head, _s5_run(self, "new"), "new")
        self.assert_runs_equal(head, _s5_run(self, "explicit", defer_stats=False, host_index=False,
                                             ema_foreach=False, adamw_impl="auto"), "explicit")
        hp = _s5_run(self, "head_pack", ref_il, ref_model, pack_tokens=True, log_every=5)
        self.assert_runs_equal(hp, _s5_run(self, "new_pack", pack_tokens=True, log_every=5), "new_pack")

    def test_speed_keys_bit_identical_to_default(self):
        """defer_stats, host_index, ema_foreach (each alone and together) do not change the math: on CPU the
        run equals the default run bit for bit (padded and packed step encoder; log_every 1 and 5)."""
        base = _s5_run(self, "base")
        for keys in (S5_KEYS, ("defer_stats",), ("host_index",), ("ema_foreach",)):
            tag = "+".join(keys)
            self.assert_runs_equal(base, _s5_run(self, tag, **{k: True for k in keys}), tag)
        for log_every in (5, 7):
            pack = _s5_run(self, f"pack{log_every}", pack_tokens=True, log_every=log_every)
            self.assert_runs_equal(pack, _s5_run(self, f"pack{log_every}_all", pack_tokens=True,
                                                 log_every=log_every, **{k: True for k in S5_KEYS}),
                                   f"pack_all_log{log_every}")

    def test_speed_keys_resume_free(self):
        """A keys-on run stopped at step 3 and resumed with the keys off ends exactly like the keys-on run."""
        import torch
        on = {k: True for k in S5_KEYS}
        cfg = _smoke_cfg(max_steps=6, eval_every=3, **on)
        a = _tmpd(self) / "a"
        il_v7.train(cfg, [shard_root()], str(a), log=lambda m: None)
        b = _tmpd(self) / "b"

        def killer(m):
            if m.startswith('{"step": 3,'):
                os.kill(os.getpid(), signal.SIGTERM)
        rb = il_v7.train(cfg, [shard_root()], str(b), log=killer)
        self.assertEqual(rb["stop_reason"], "signal")
        rc = il_v7.train(_smoke_cfg(max_steps=6, eval_every=3), [shard_root()], str(b), log=lambda m: None,
                         resume=True)
        self.assertEqual((rc["stop_reason"], rc["steps"]), ("total_steps", 6))
        fa, fb = _ck(a / "ckpt-0000006.pt"), _ck(b / "ckpt-0000006.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))

    def _nan_at(self, call_no):
        """compute_losses wrapper: the total of the call_no-th training call (1-based) becomes NaN."""
        orig = il_v7.compute_losses
        n = {"calls": 0}

        def wrapped(out, batch, loss_w, dens=None, **kw):
            res = orig(out, batch, loss_w, dens=dens, **kw)
            if dens is not None:
                n["calls"] += 1
                if n["calls"] == call_no:
                    res["total"] = res["total"] * float("nan")
            return res
        return mock.patch.object(il_v7, "compute_losses", wrapped)

    def test_deferred_nonfinite_loss_raises_before_any_save(self):
        """Without defer_stats the run raises at the bad micro-batch; defer_stats raises at the next flush (log step or eval step),
        naming the same step, before anything is logged or saved after it."""
        for defer, over, detected in ((False, {"log_every": 10, "eval_every": 3}, None),
                                      (True, {"log_every": 10, "eval_every": 3}, 3),
                                      (True, {"log_every": 2, "eval_every": 0}, 2)):
            out = _tmpd(self) / f"nan{int(defer)}{over['log_every']}"
            logs = []
            cfg = _smoke_cfg(max_steps=8, micro_batch_rows=0, defer_stats=defer, **over)
            with self._nan_at(2), self.assertRaisesRegex(FloatingPointError, "non-finite loss at step 1") as cm:
                il_v7.train(cfg, [shard_root()], str(out), log=logs.append)
            if detected:
                self.assertIn(f"detected at step {detected}", str(cm.exception))
            self.assertEqual(list(out.glob("*.pt")), [], over)
            steps = [json.loads(l)["step"] for l in logs if l.startswith('{"step"')]
            self.assertEqual(steps, [1], (defer, over))          # step 1 logged (finite); nothing after

    def test_ema_foreach_bit_identical(self):
        import torch
        model = tiny_model(double=False)
        model.register_buffer("int_buf", torch.arange(5))      # a non-floating entry (copied, not averaged)
        ea, eb = il_v7.EMA(model, 0.99), il_v7.EMA(model, 0.99, foreach=True)
        g = torch.Generator().manual_seed(1)
        for i in range(40):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn(p.shape, generator=g) * 0.01)
                model.int_buf.add_(1)
            ea.update(model)
            eb.update(model)
        self.assertEqual(ea.steps, eb.steps)
        for k, v in ea.shadow.items():
            self.assertTrue(torch.equal(v, eb.shadow[k]), k)
        self.assertTrue(torch.equal(eb.shadow["int_buf"], torch.arange(5) + 40))

    def test_adamw_impl_kernels(self):
        """'auto' is fused on CUDA only.  On CPU: 'foreach' and 'fused' match 'auto' after one
        step to fp32 rounding (measured below); 'fused' is skipped where this torch has no CPU kernel."""
        import torch
        dev = torch.device("cpu")
        self.assertFalse(il_v7.make_optimizer(tiny_model(double=False), _smoke_cfg(), dev).defaults["fused"])
        g = torch.Generator().manual_seed(2)
        base = tiny_model(double=False)
        grads = {n: torch.randn(p.shape, generator=g) for n, p in base.named_parameters()}

        def stepped(impl, steps):
            m = copy.deepcopy(base)
            opt = il_v7.make_optimizer(m, _smoke_cfg(adamw_impl=impl), dev)
            for _ in range(steps):
                for n, p in m.named_parameters():
                    p.grad = grads[n].clone()
                opt.step()
            return dict(m.named_parameters())

        impls = ["foreach"]
        try:
            stepped("fused", 1)
            impls.append("fused")
        except RuntimeError as e:                              # torch < 2.4: no fused CPU AdamW
            print(f"adamw fused on CPU unavailable: {e}")
        for steps, tol in ((1, 2e-7), (30, 2e-6)):
            ref = stepped("auto", steps)
            for impl in impls:
                got = stepped(impl, steps)
                worst = max(float((ref[n] - got[n]).abs().max()) for n in ref)
                print(f"adamw {impl} vs auto after {steps} step(s): max |dparam| = {worst:.3e}")
                self.assertLess(worst, tol, (impl, steps))
                moved = max(float((ref[n] - dict(base.named_parameters())[n]).abs().max()) for n in ref)
                self.assertGreater(moved, 100 * tol)           # the step itself is far above the tolerance

    def test_v5_fast_config_is_v4_plus_speed_keys(self):
        from dataclasses import asdict
        v4 = asdict(il_v7.TrainConfig.from_file(_config("aminiformer_v4.yaml")))
        v5 = asdict(il_v7.TrainConfig.from_file(_config("aminiformer_v5_fast.yaml")))
        self.assertEqual({k for k in v4 if v4[k] != v5[k]}, {"name", *S5_KEYS})
        self.assertTrue(all(v5[k] for k in S5_KEYS))
        self.assertEqual(v5["adamw_impl"], "auto")               # = fused on CUDA



# ------------------------------------------------------------------ optional label weights
DW5_REF_COMMIT = "e6720cc"           # the trainer before these keys (reference for the bit-identity test)
DW5_KEYS = ("skill_weights_file", "skill_weights_sha256", "afk_tail_ticks", "eval_skill_weights_file",
            "eval_afk_tail_ticks")


def _reference_dw5_head():
    """il_v7 of DW5_REF_COMMIT as a separate module (IL_V7_DW5REF_FILE, else git), or None.  The model file is
    unchanged by these keys, so the reference trainer runs with the current aminiformer."""
    f = os.environ.get("IL_V7_DW5REF_FILE")
    if f and Path(f).is_file():
        return _load_module("il_v7_ref_dw5", Path(f).read_text())
    try:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{DW5_REF_COMMIT}:src/amini/train/il_v7.py"],
                              capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return _load_module("il_v7_ref_dw5", text)


def _skill_file(case, sides, name="sw.json"):
    p = _tmpd(case) / name
    p.write_text(json.dumps({"schema": "il-v7-skill-weights", "version": 1,
                             "provenance": {"generation": "synthetic"}, "sides": sides}))
    return str(p)


def _brute_tail(src, ep, ticks):
    """Reference AFK-tail start of one episode, one row at a time."""
    si, gi, side, lo, hi = ep
    g = src.shards[si]["games"][gi]
    re = int(g["real_end"])
    if hi <= lo or re <= 0:
        return hi
    rows = np.asarray(src.shards[si]["rows"][lo:hi])
    acts = [k for k in range(hi - lo) if int(rows["kind"][k]) in (1, 2)]
    if not acts:
        return lo if re >= ticks else hi
    k = acts[-1]
    return lo + k + 1 if re - int(rows["tick"][k]) >= ticks else hi


def _strip_weighted(res):
    out = copy.deepcopy(res)
    for h in out["history"]:
        if "eval" in h:
            h["eval"].pop("weighted", None)
    return out


def _synthetic_tags():
    src = source()
    return [il_v7._tag_text(src, 0, gi) for gi in range(len(src.shards[0]["games"]))]


class TestV5DataWeights(unittest.TestCase):
    """Optional label weights: skill_weights_file (per-side loss multiplier), afk_tail_ticks (AFK-tail zero weight),
    and the eval_* secondary weighted metric.  Defaults reproduce the unweighted trainer bit for bit."""

    def test_config_keys(self):
        c = _smoke_cfg()
        self.assertEqual([getattr(c, k) for k in DW5_KEYS], ["", "", 0, "", 0])
        self.assertFalse(set(DW5_KEYS) & set(il_v7.RESUME_FREE_KEYS))       # they change the objective
        for bad in ({"afk_tail_ticks": -1}, {"afk_tail_ticks": 1.5}, {"eval_afk_tail_ticks": True},
                    {"skill_weights_sha256": "ab"}, {"skill_weights_sha256": "a" * 64},     # pin needs a file
                    {"skill_weights_file": "x.json", "skill_weights_sha256": "g" * 64}):
            with self.assertRaises(ValueError, msg=bad):
                _smoke_cfg(**bad)
        ok = _smoke_cfg(skill_weights_file="x.json", skill_weights_sha256="AB" * 32, afk_tail_ticks=500)
        self.assertEqual((ok.skill_weights_sha256, ok.afk_tail_ticks), ("ab" * 32, 500))
        self.assertEqual(il_v7.parse_set_overrides({}, ["skill_weights_file=", "afk_tail_ticks=0"]),
                         {"skill_weights_file": None, "afk_tail_ticks": 0})
        self.assertEqual(_smoke_cfg(skill_weights_file=None).skill_weights_file, "")

    def test_afk_tail_starts_hand_cases(self):
        """Known last plays: tail after the last action row iff real_end - its tick >= min_ticks; a side without
        an action: the whole episode iff real_end >= min_ticks; empty side / unknown real_end: none."""
        # rows 0-4: game 0 side 0 (ticks 0, 10, 20, 30, 40; play at row 1, ability at row 2);
        # rows 5-7: game 0 side 1 (ticks 0, 300, 600; play at row 6); rows 8-9: game 1 side 0 (no action);
        # game 1 side 1 empty; rows 10-11: game 2 side 0 (play at row 11), real_end 0 (unknown); side 1 empty
        kind = np.array([0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1], np.int8)
        tick = np.array([0, 10, 20, 30, 40, 0, 300, 600, 0, 10, 0, 5], np.int16)
        lo = np.array([[0, 5], [8, 10], [10, 12]])
        hi = np.array([[5, 8], [10, 10], [12, 12]])
        for real_end, want in ((600, [[3, 8], [8, 10], [12, 12]]),      # 600-20 >= 500; 600-300 < 500
                               (800, [[3, 7], [8, 10], [12, 12]]),      # 800-300 >= 500 -> after row 6
                               (519, [[5, 8], [8, 10], [12, 12]]),      # 519-20 < 500: none; no action: all
                               (499, [[5, 8], [10, 10], [12, 12]])):    # no-action side: 499 < 500 -> none
            got = il_v7.afk_tail_starts(kind, tick, lo, hi, np.array([real_end, real_end, 0]), 500)
            self.assertEqual(got.tolist(), want, real_end)

    def test_afk_tail_starts_match_bruteforce(self):
        src = source()
        for ticks in (1, 50, 200, 500, 10 ** 6):
            w = il_v7.RowWeighting(src, afk_tail_ticks=ticks)
            for ep in src.episodes:
                self.assertEqual(w.tail_start(ep), _brute_tail(src, ep, ticks), (ticks, ep))
        w = il_v7.RowWeighting(src, afk_tail_ticks=1)
        self.assertGreater(sum(w.tail_start(e) < e[4] for e in src.episodes), 0)   # the test exercises tails

    def test_apply_train_and_eval_labels(self):
        src = source()
        tags = _synthetic_tags()
        table = {t: [0.5 + 0.1 * gi, 0.25] for gi, t in enumerate(tags) if gi}       # game 0 missing
        w = il_v7.RowWeighting(src, _skill_file(self, table), afk_tail_ticks=1)
        self.assertEqual(w.side_weight(src.episodes[0]), 1.0)
        for ep in src.episodes:
            lo, hi = ep[3], ep[4]
            n = hi - lo
            lm = np.ones(n, bool)
            lm[:3] = False                                     # a burn-in prefix stays False
            lab = {"loss_mask": lm.copy()}
            w.apply(lab, ep, lo, hi, eval_mode=False)
            want_w = 1.0 if ep[1] == 0 else (0.5 + 0.1 * ep[1] if ep[2] == 0 else 0.25)
            self.assertEqual(lab["loss_scale"].dtype, np.float32)
            self.assertTrue(np.all(lab["loss_scale"] == np.float32(want_w)), ep)
            tail = _brute_tail(src, ep, 1) - lo
            want_lm = lm.copy()
            want_lm[tail:] = False
            self.assertEqual(lab["loss_mask"].tolist(), want_lm.tolist(), ep)
            ev = {"loss_mask": lm.copy()}
            w.apply(ev, ep, lo, hi, eval_mode=True)
            self.assertEqual(ev["loss_mask"].tolist(), lm.tolist())          # eval: the metric's mask untouched
            self.assertNotIn("loss_scale", ev)
            want_sc = np.full(n, want_w, np.float32)
            want_sc[tail:] = 0
            self.assertEqual(ev["eval_scale"].tolist(), want_sc.tolist())
            mid = (lo + hi) // 2                               # a chunk [mid, hi): the tail is cut at its start
            lab = {"loss_mask": np.ones(hi - mid, bool)}
            w.apply(lab, ep, mid, hi, eval_mode=False)
            self.assertEqual(lab["loss_mask"].tolist(), [mid + i < tail + lo for i in range(hi - mid)], ep)
        only_afk = il_v7.RowWeighting(src, afk_tail_ticks=1)
        lab = {"loss_mask": np.ones(5, bool)}
        only_afk.apply(lab, src.episodes[0], src.episodes[0][3], src.episodes[0][3] + 5, eval_mode=False)
        self.assertNotIn("loss_scale", lab)
        self.assertIsNone(il_v7.RowWeighting.build(src))
        with self.assertRaises(ValueError):
            il_v7.RowWeighting(src, _skill_file(self, table), sha256_pin="0" * 64)
        bad = _tmpd(self) / "bad.json"
        for doc in ({"schema": "other", "sides": {}}, {"schema": "il-v7-skill-weights", "sides": {"X": [1.0]}},
                    {"schema": "il-v7-skill-weights", "sides": {"X": [1.0, float("nan")]}},
                    {"schema": "il-v7-skill-weights", "sides": {"X": [1.0, -0.1]}}):
            bad.write_text(json.dumps(doc))
            with self.assertRaises(ValueError, msg=doc):
                il_v7.load_skill_weights(str(bad))

    @unittest.skipUnless(HAVE_TOKENIZER, NEEDS_TOKENIZER)
    def test_loss_weights_scaled_per_side(self):
        """Every loss term's per-element weight is the unweighted one times the row's side weight (exact)."""
        import torch
        from amini.nn import tokenizer_v7 as TK
        src = source()
        table = {t: [0.5 + 0.1 * gi, 0.25] for gi, t in enumerate(_synthetic_tags())}
        cfg = _smoke_cfg(micro_batch_rows=0)
        items = il_v7.chunk_episodes(src, 0, cfg.max_seq_rows, cfg.burn_in_rows)
        rw = il_v7.RowWeighting(src, _skill_file(self, table))
        ds = il_v7.EpisodeBatches(src, items, tables(), cfg, train=True, weighting=rw)
        ds0 = il_v7.EpisodeBatches(src, items, tables(), cfg, train=True)
        seen = set()
        for b in range(len(ds)):
            for c, c0 in zip(ds[b], ds0[b]):
                batch, b0 = TK.to_torch(c), TK.to_torch(c0)
                self.assertEqual(set(b0["labels"]) | {"loss_scale"}, set(batch["labels"]))
                W, W0 = il_v7.loss_weights(batch), il_v7.loss_weights(b0)
                rv = batch["row"]["row_valid"]
                s = batch["labels"]["loss_scale"][rv]
                ridx, pidx = il_v7.loss_indices(batch)
                self.assertEqual(set(W), set(W0))
                for k in W:
                    f = s[pidx] if k == "cell" else s[ridx] if k == "aux_threat" else s
                    if W0[k].dim() == 2:
                        f = f.unsqueeze(-1)
                    self.assertTrue(torch.equal(W[k], W0[k] * f), k)
                    seen.add(k)
                seen.update(float(x) for x in s.unique())
        self.assertTrue({"gate", "slot", "cell", "ability", "value", "aux_threat", 0.25} <= seen, seen)

    def _run(self, name, mod=None, **over):
        return _s5_run(self, name, mod, None, **over)

    def test_defaults_bit_identical_to_head(self):
        """Without the label-weight keys (and with them explicitly at their defaults) a run equals the earlier trainer
        (e6720cc) bit for bit: 32 steps x 2 micro-batches (losses / terms / grad norms, eval reports, raw / EMA
        weights, AdamW moments); also with the packed encoder and the speed keys."""
        ref = _reference_dw5_head()
        if ref is None:
            self.skipTest("needs a copy of the earlier trainer (set IL_V7_DW5REF_FILE)")
        self.assertNotIn("afk_tail_ticks", ref.TrainConfig.__dataclass_fields__)
        head = self._run("head", ref)
        TestV5Speed.assert_runs_equal(self, head, self._run("new"), "new")
        TestV5Speed.assert_runs_equal(self, head, self._run("explicit", skill_weights_file="", skill_weights_sha256="",
                                                            afk_tail_ticks=0, eval_skill_weights_file="",
                                                            eval_afk_tail_ticks=0), "explicit")
        hp = self._run("head_pack", ref, pack_tokens=True, host_index=True, defer_stats=True, log_every=5)
        TestV5Speed.assert_runs_equal(self, hp, self._run("new_pack", pack_tokens=True, host_index=True,
                                                          defer_stats=True, log_every=5), "new_pack")

    def test_neutral_weighting_bit_identical(self):
        """The code path itself is exact: an all-1.0 table and an AFK threshold no game reaches give the default
        run bit for bit (also with host_index); the eval_* keys add report["weighted"] and change nothing else,
        and with neutral weights the weighted metric equals the selection metric."""
        tags = _synthetic_tags()
        ones = _skill_file(self, {t: [1.0, 1.0] for t in tags[:-1]})
        base = self._run("base")
        TestV5Speed.assert_runs_equal(self, base, self._run("neutral", skill_weights_file=ones,
                                                            afk_tail_ticks=10 ** 6), "neutral")
        hx = self._run("base_hx", host_index=True)
        TestV5Speed.assert_runs_equal(self, hx, self._run("neutral_hx", host_index=True, skill_weights_file=ones,
                                                          afk_tail_ticks=10 ** 6), "neutral_hx")
        ev_res, ev_ck = self._run("eval_only", eval_skill_weights_file=ones, eval_afk_tail_ticks=10 ** 6)
        evals = [h["eval"] for h in ev_res["history"] if "eval" in h]
        self.assertEqual(len(evals), 2)
        for e in evals:
            self.assertEqual(e["weighted"]["selection_nll"], e["selection_nll"])
            self.assertEqual(e["weighted"]["terms"], e["terms"])
        TestV5Speed.assert_runs_equal(self, base, (_strip_weighted(ev_res), ev_ck), "eval_only")
        self.assertNotIn("weighted", [h["eval"] for h in base[0]["history"] if "eval" in h][0])

    def test_weighting_changes_training_and_is_recorded(self):
        src = source()
        tags = _synthetic_tags()
        sw = _skill_file(self, {t: [0.5, 0.7] for t in tags[1:]})          # game 0 missing -> 1.0
        logs = []
        cfg = _smoke_cfg(max_steps=2, eval_every=2, skill_weights_file=sw, afk_tail_ticks=50,
                         eval_skill_weights_file=sw, eval_afk_tail_ticks=50)
        res = il_v7.train(cfg, [shard_root()], str(_tmpd(self) / "w"), log=logs.append)
        st = _start(logs)
        rec, erec = st["row_weighting"], st["eval_row_weighting"]
        tr = il_v7.chunk_episodes(src, 0, cfg.max_seq_rows, cfg.burn_in_rows)
        va = il_v7.chunk_episodes(src, 1, cfg.max_seq_rows, cfg.burn_in_rows)
        sk = rec["skill_weights"]
        self.assertEqual(sk["sha256"], hashlib.sha256(Path(sw).read_bytes()).hexdigest())
        self.assertEqual(sk["games"], len({it[0][1] for it in tr}))
        self.assertEqual(sk["games_missing"], int(any(it[0][1] == 0 for it in tr)))
        rows = sum(it[3] - it[2] for it in tr)
        self.assertEqual(sk["loss_rows"], rows)
        f32 = lambda x: float(np.float32(x))                          # the trainer multiplies by float32 weights
        want = sum((it[3] - it[2]) * (1.0 if it[0][1] == 0 else (0.5 if it[0][2] == 0 else f32(0.7))) for it in tr)
        self.assertAlmostEqual(sk["mean_row_weight"], want / rows, places=12)
        af = rec["afk_tail"]
        self.assertEqual(af["ticks"], 50)
        self.assertEqual(af["loss_rows_zeroed"],
                         sum(max(0, it[3] - max(it[2], _brute_tail(src, it[0], 50))) for it in tr))
        self.assertEqual(af["tail_episodes"], sum(1 for e in {it[0] for it in tr} if _brute_tail(src, e, 50) < e[4]))
        self.assertGreater(af["loss_rows_zeroed"], 0)
        self.assertEqual(erec["afk_tail"]["loss_rows"], sum(it[3] - it[2] for it in va))
        ev = [h["eval"] for h in res["history"] if "eval" in h][0]
        self.assertNotEqual(ev["weighted"]["selection_nll"], ev["selection_nll"])
        logs0 = []
        il_v7.train(_smoke_cfg(max_steps=2, eval_every=2), [shard_root()], str(_tmpd(self) / "d"), log=logs0.append)
        self.assertNotIn("row_weighting", _start(logs0))
        self.assertNotIn("eval_row_weighting", _start(logs0))
        lw = [json.loads(l) for l in logs if l.startswith('{"step"')]
        l0 = [json.loads(l) for l in logs0 if l.startswith('{"step"')]
        # the objective changed.  Step 1 ties exactly: its two chunks are side 0 of games 3 and 4 (both 0.5, no
        # AFK tail at 50 ticks) and every term is normalised by its own weight sum, so a uniform weight cancels;
        # step 2 mixes 0.5 / 0.7 rows and an AFK tail.
        self.assertNotEqual([x["loss"] for x in lw], [x["loss"] for x in l0])

    def test_resume_exact_and_refusals(self):
        import torch
        sw = _skill_file(self, {t: [0.5, 0.7] for t in _synthetic_tags()})
        sha = hashlib.sha256(Path(sw).read_bytes()).hexdigest()
        on = dict(skill_weights_file=sw, skill_weights_sha256=sha, afk_tail_ticks=50)
        cfg = _smoke_cfg(max_steps=6, eval_every=3, **on)
        a = _tmpd(self) / "a"
        il_v7.train(cfg, [shard_root()], str(a), log=lambda m: None)
        b = _tmpd(self) / "b"

        def killer(m):
            if m.startswith('{"step": 3,'):
                os.kill(os.getpid(), signal.SIGTERM)
        self.assertEqual(il_v7.train(cfg, [shard_root()], str(b), log=killer)["stop_reason"], "signal")
        for over in ({"afk_tail_ticks": 0}, {"skill_weights_file": "", "skill_weights_sha256": ""}):
            with self.assertRaisesRegex(ValueError, "config differs"):
                il_v7.train(_smoke_cfg(max_steps=6, eval_every=3, **{**on, **over}), [shard_root()], str(b),
                            log=lambda m: None, resume=True)
        rc = il_v7.train(cfg, [shard_root()], str(b), log=lambda m: None, resume=True)
        self.assertEqual((rc["stop_reason"], rc["steps"]), ("total_steps", 6))
        fa, fb = _ck(a / "ckpt-0000006.pt"), _ck(b / "ckpt-0000006.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))
        Path(sw).write_text(Path(sw).read_text().replace("0.7", "0.8"))      # the pinned file changed
        with self.assertRaisesRegex(ValueError, "sha256"):
            il_v7.train(cfg, [shard_root()], str(_tmpd(self) / "c"), log=lambda m: None)

    def test_real_shard_counts(self):
        """On a real shard (V7_REAL_SHARD): the vectorised AFK tails equal the row-by-row reference, with
        counts; with V7_SKILL_WEIGHTS also the table coverage of the shard."""
        d = os.environ.get("V7_REAL_SHARD", "")
        if not d or not Path(d, "games.npy").is_file():
            self.skipTest("set V7_REAL_SHARD to a real v7 shard directory")
        src = il_v7.ShardSource([d])
        w = il_v7.RowWeighting(src, os.environ.get("V7_SKILL_WEIGHTS", ""), afk_tail_ticks=500)
        tails = 0
        for ep in src.episodes:
            t = w.tail_start(ep)
            self.assertEqual(t, _brute_tail(src, ep, 500), ep)
            tails += t < ep[4]
        items = il_v7.chunk_episodes(src, 0, 512, 192)
        rec = w.record(items)
        print(f"real shard {Path(d).name}: {len(src.episodes)} episodes, {tails} AFK tails (500 ticks); "
              f"train record {json.dumps(rec, default=str)}")
        self.assertGreater(tails, 0)


# ------------------------------------------------------------------ stage-S subset and runtime keys
SP_REF_COMMIT = "5842082"            # the trainer before these keys (reference for the bit-identity test)
SP_RUNTIME_KEYS = ("eval_every_early", "eval_late_frac", "ckpt_every", "loader_prefetch", "stage_p_compiled")


def _reference_sp_head():
    """(il_v7, aminiformer, tokenizer_v7) of SP_REF_COMMIT as separate modules (IL_V7_SPREF_FILE /
    AMINIFORMER_SPREF_FILE / TOKENIZER_SPREF_FILE, else git), or None."""

    out = []
    for env, path in (("IL_V7_SPREF_FILE", "src/amini/train/il_v7.py"),
                      ("AMINIFORMER_SPREF_FILE", "src/amini/nn/aminiformer.py"),
                      ("TOKENIZER_SPREF_FILE", "src/amini/nn/tokenizer_v7.py")):
        f = os.environ.get(env)
        if f and Path(f).is_file():
            out.append(Path(f).read_text())
            continue
        try:
            out.append(subprocess.run(["git", "-C", str(ROOT), "show", f"{SP_REF_COMMIT}:{path}"],
                                      capture_output=True, text=True, check=True, timeout=30).stdout)
        except (OSError, subprocess.SubprocessError):
            return None
    return (_load_module("il_v7_ref_sp", out[0]), _load_module("amini.nn._aminiformer_ref_sp", out[1]),
            _load_module("amini.nn._tokenizer_v7_ref_sp", out[2]))


def _sp_rows_expected(src, item, base_frac, threat_frac):
    """Stage-S rows of a chunk computed from the WHOLE episode (rows lo..hi): chunking-independent."""
    from amini.nn import tokenizer_v7 as TK
    ep, lo, _, hi = item                     # shard row numbers; the episode starts at ep[3]
    full = src.episode(ep)
    sel = TK.stage_s_rows(full.tag, full.rows, base_frac, threat_frac)
    assert len(sel) == ep[4] - ep[3] and ep[3] <= lo < hi <= ep[4]
    return sel[lo - ep[3]:hi - ep[3]]


class _SPCrash(Exception):
    pass


class TestV5SP(unittest.TestCase):
    """stage_s_threat_frac (objective), eval_every_early / eval_late_frac, ckpt_every, loader_prefetch,
    stage_p_compiled (runtime).  Defaults = the earlier trainer (5842082) bit for bit."""

    assert_runs_equal = TestV5Speed.assert_runs_equal

    def test_config_keys(self):
        c = _smoke_cfg()
        self.assertEqual((c.stage_s_threat_frac, c.eval_every_early, c.eval_late_frac, c.ckpt_every,
                          c.loader_prefetch, c.stage_p_compiled), (1.0, 0, 0.8, 0, 4, False))
        self.assertTrue(set(SP_RUNTIME_KEYS) <= set(il_v7.RESUME_FREE_KEYS))
        self.assertNotIn("stage_s_threat_frac", il_v7.RESUME_FREE_KEYS)       # an objective key
        self.assertNotIn("stage_s_base_frac", il_v7.RESUME_FREE_KEYS)
        for bad in ({"stage_s_threat_frac": 1.5}, {"stage_s_threat_frac": -0.1}, {"stage_s_threat_frac": True},
                    {"stage_s_threat_frac": "half"}, {"eval_late_frac": 1.2}, {"eval_every_early": -1},
                    {"eval_every_early": 1.5}, {"eval_every_early": True}, {"ckpt_every": -2},
                    {"ckpt_every": "100"}, {"loader_prefetch": 0}, {"loader_prefetch": 2.0},
                    {"stage_p_compiled": 2}, {"stage_p_compiled": "yes"},
                    {"stage_p_compiled": True},                                   # needs model.cell_fp32 logit
                    {"stage_p_compiled": True, "model": dict(c.model, cell_fp32="auto")}):
            with self.assertRaises(ValueError, msg=bad):
                _smoke_cfg(**bad)
        ok = _smoke_cfg(stage_s_threat_frac=0, eval_every_early=4000, eval_late_frac=1, ckpt_every=500,
                        loader_prefetch=2, stage_p_compiled=True, model=dict(c.model, cell_fp32="logit"))
        self.assertEqual((ok.stage_s_threat_frac, ok.eval_late_frac, ok.stage_p_compiled), (0.0, 1.0, True))
        self.assertIsInstance(ok.stage_s_threat_frac, float)
        model = tiny_model(double=False)
        il_v7.apply_speed_settings(model, ok)
        self.assertTrue(model.cells.p_regions)
        il_v7.apply_speed_settings(model, c)
        self.assertFalse(model.cells.p_regions)
        # an objective key: resuming with another stage_s_threat_frac is refused
        out = _tmpd(self) / "r"
        il_v7.train(_smoke_cfg(max_steps=2, eval_every=1), [shard_root()], str(out), log=lambda m: None)
        with self.assertRaisesRegex(ValueError, "stage_s_threat_frac"):
            il_v7.train(_smoke_cfg(max_steps=2, eval_every=1, stage_s_threat_frac=0.5), [shard_root()], str(out),
                        log=lambda m: None, resume=True)

    def test_defaults_bit_identical_to_head(self):
        """Without the SP keys, with them at their defaults, and with the runtime keys set (ckpt_every,
        loader_prefetch: they do not change the math), a run equals the earlier trainer + model (5842082) bit for
        bit; with the v5 speed keys on both sides as well."""
        ref = _reference_sp_head()
        if ref is None:
            self.skipTest("needs a copy of the earlier trainer and model (set IL_V7_SPREF_FILE and AMINIFORMER_SPREF_FILE)")
        ref_il, ref_model, _ = ref
        self.assertNotIn("stage_s_threat_frac", ref_il.TrainConfig.__dataclass_fields__)
        head = _s5_run(self, "head", ref_il, ref_model)
        self.assert_runs_equal(head, _s5_run(self, "new"), "new")
        self.assert_runs_equal(head, _s5_run(self, "explicit", stage_s_threat_frac=1.0, eval_every_early=0,
                                             eval_late_frac=0.9, ckpt_every=0, loader_prefetch=4,
                                             stage_p_compiled=False), "explicit")
        self.assert_runs_equal(head, _s5_run(self, "runtime", ckpt_every=5, loader_prefetch=2), "runtime")
        on = {k: True for k in S5_KEYS}
        h5 = _s5_run(self, "head_s5", ref_il, ref_model, pack_tokens=True, log_every=5, **on)
        self.assert_runs_equal(h5, _s5_run(self, "new_s5", pack_tokens=True, log_every=5, **on), "new_s5")

    def test_tokenizer_default_rule_is_head(self):
        """stage_s_rows with threat_frac 1 (the default) equals the earlier stage_s_rows for every base_frac; with
        the default the examples equal the earlier make_example output."""
        ref = _reference_sp_head()
        if ref is None:
            self.skipTest("needs a copy of the earlier tokenizer")
        from amini.nn import tokenizer_v7 as TK
        ref_tk = ref[2]
        src = source()
        for ep_key in src.episodes:
            ep = src.episode(ep_key)
            for bf in (0.0, 0.3, 1.0):
                a = TK.stage_s_rows(ep.tag, ep.rows, bf)
                self.assertTrue(np.array_equal(a, ref_tk.stage_s_rows(ep.tag, ep.rows, bf)), (ep_key, bf))
                self.assertTrue(np.array_equal(a, TK.stage_s_rows(ep.tag, ep.rows, bf, 1.0)), (ep_key, bf))
        ep = src.episode(src.episodes[0])
        dr = TK.draw_parity(ep.tag, int(ep.rows["side"][0]), ep.rows, ep.evt, tables())
        a = TK.make_example(ep, dr, tables(), stage_s_base_frac=0.5)
        b = ref_tk.make_example(ep, dr, tables(), stage_s_base_frac=0.5)
        for sec in ("labels", "targets"):
            self.assertEqual(set(a[sec]), set(b[sec]))
            for k in a[sec]:
                x, y = np.asarray(a[sec][k]), np.asarray(b[sec][k])
                self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), (sec, k))
                self.assertEqual(x.tobytes(), y.tobytes(), (sec, k))

    @unittest.skipUnless(HAVE_TOKENIZER, NEEDS_TOKENIZER)
    def test_threat_frac_rows(self):
        """0 -> play rows (+ the base_frac subset); 1 -> HEAD; 0.5 -> a deterministic hash subset of about half
        the threat rows, nested in the frac, independent of base_frac; play rows always kept."""
        from amini.nn import tokenizer_v7 as TK
        from amini.nn import schema_v1 as S
        src = source()
        n_thr, n_half = 0, 0
        for ep_key in src.episodes:
            ep = src.episode(ep_key)
            r = ep.rows
            play = r["kind"].astype(np.int64) == S.FAMILY_PLAY
            thr = (r["stratum"] == S.STRATUM_THREAT) & ~play
            base = (r["stratum"] == S.STRATUM_BASE) & ~play
            self.assertTrue(np.array_equal(TK.stage_s_rows(ep.tag, r, 0.0, 0.0), play), ep_key)
            self.assertTrue(np.array_equal(TK.stage_s_rows(ep.tag, r, 1.0, 0.0), play | base), ep_key)
            self.assertTrue(np.array_equal(TK.stage_s_rows(ep.tag, r, 0.0, 1.0),
                                           play | (r["stratum"] == S.STRATUM_THREAT)), ep_key)
            half = TK.stage_s_rows(ep.tag, r, 0.0, 0.5)
            self.assertTrue(np.array_equal(half, TK.stage_s_rows(ep.tag, r, 0.0, 0.5)))   # deterministic
            self.assertTrue(np.array_equal(half & ~thr, play))                          # play kept, no base
            lo, hi = TK.stage_s_rows(ep.tag, r, 0.0, 0.3), TK.stage_s_rows(ep.tag, r, 0.0, 0.7)
            self.assertFalse((lo & ~half).any() or (half & ~hi).any())                  # nested
            # base_frac does not move the threat subset (own salt): with base_frac 0.5 the threat part is equal
            both = TK.stage_s_rows(ep.tag, r, 0.5, 0.5)
            self.assertTrue(np.array_equal(both & thr, half & thr))
            self.assertTrue(np.array_equal(both & base, TK.stage_s_rows(ep.tag, r, 0.5, 0.0) & base))
            n_thr += int(thr.sum())
            n_half += int((half & thr).sum())
        print(f"stage_s_threat_frac 0.5: {n_half} of {n_thr} threat rows ({n_half / n_thr:.3f})")
        self.assertGreater(n_thr, 200)
        self.assertLess(abs(n_half / n_thr - 0.5), 0.08)

    def test_threat_frac_batches_stable_across_epochs_and_workers(self):
        """The loader's stage-S rows (targets.cell_rows, train and val) are the per-row hash rule on the whole
        episode, in every epoch and with 0 or 2 loader workers."""
        src = source()
        for frac in (0.0, 0.5):
            cfg = _smoke_cfg(stage_s_threat_frac=frac, stage_s_base_frac=0.0, micro_batch_rows=200,
                             max_seq_rows=64, burn_in_rows=16)
            seen = chunked = 0
            for split, train in ((0, True), (1, False)):
                items = il_v7.chunk_episodes(src, split, cfg.max_seq_rows, cfg.burn_in_rows)
                ds = il_v7.EpisodeBatches(src, items, tables(), cfg, train=train)
                for epoch, workers in ((0, 0), (1, 0), (1, 2)):
                    ds.set_epoch(epoch)
                    for b, step in enumerate(il_v7.make_loader(ds, workers, prefetch=2)):
                        step = step if isinstance(step, (list, tuple)) else [step]
                        self.assertEqual(len(step), len(ds.batches[b]))
                        for mb, micro in zip(step, ds.batches[b]):
                            cr = np.asarray(mb["targets"]["cell_rows"])
                            for j, i in enumerate(micro):
                                want = _sp_rows_expected(src, items[i], 0.0, frac)
                                self.assertEqual(len(want), items[i][3] - items[i][1])
                                self.assertTrue(np.array_equal(cr[j, :len(want)], want), (frac, epoch, workers, i))
                                self.assertFalse(cr[j, len(want):].any())
                                seen += 1
                                chunked += items[i][1] > items[i][0][3]
            self.assertGreater(seen, 20)
            self.assertGreater(chunked, 0)                 # chunks that start inside an episode are covered

    @unittest.skipUnless(HAVE_TOKENIZER, NEEDS_TOKENIZER)
    def test_host_index_with_threat_frac(self):
        """host_index gives the indices of the same (fewer) stage-S rows: equal to loss_indices of the plain
        batch, and a short training run with host_index equals the run without it bit for bit."""
        import torch
        from amini.nn import tokenizer_v7 as TK
        src = source()
        for frac in (0.0, 0.5):
            cfg = _smoke_cfg(stage_s_threat_frac=frac, stage_s_base_frac=0.0, host_index=True,
                             micro_batch_rows=200)
            items = il_v7.chunk_episodes(src, 0, cfg.max_seq_rows, cfg.burn_in_rows)
            ds = il_v7.EpisodeBatches(src, items, tables(), cfg, train=True)
            n = 0
            for b in range(len(ds)):
                for mb in ds[b]:
                    hx = mb["host_index"]
                    plain = TK.to_torch({k: v for k, v in mb.items() if k != "host_index"}, "cpu")
                    ridx, pidx = il_v7.loss_indices(plain)
                    self.assertTrue(torch.equal(hx["ridx"], ridx) and torch.equal(hx["pidx"], pidx), frac)
                    rv = plain["row"]["row_valid"].bool()
                    self.assertEqual(int(ridx.numel()), int(plain["targets"]["cell_rows"][rv].sum()))
                    n += int(ridx.numel())
            self.assertGreater(n, 0)
        half = {"stage_s_threat_frac": 0.5, "stage_s_base_frac": 0.0}
        self.assert_runs_equal(_s5_run(self, "thr_off", **half),
                               _s5_run(self, "thr_hx", host_index=True, **half), "threat_frac+host_index")

    def test_threat_frac_changes_only_stage_s(self):
        """A run with threat_frac 0 (play rows only) trains: fewer stage-S rows, the slot / family terms of the
        first step are unchanged (same batch), aux_threat differs."""
        m0 = dict(_smoke_cfg().model, dropout=0.0)                    # no dropout: same forward outside stage S
        a = il_v7.train(_smoke_cfg(max_steps=2, eval_every=2, stage_s_base_frac=0.0, model=m0), [shard_root()],
                        str(_tmpd(self) / "a"), log=lambda m: None)
        b = il_v7.train(_smoke_cfg(max_steps=2, eval_every=2, stage_s_base_frac=0.0, stage_s_threat_frac=0.0,
                                   model=m0), [shard_root()], str(_tmpd(self) / "b"), log=lambda m: None)
        ha, hb = a["history"][0], b["history"][0]
        self.assertEqual((ha["step"], hb["step"]), (1, 1))
        for k in ("gate", "slot", "ability", "value"):
            self.assertEqual(ha[k], hb[k], k)
        self.assertNotEqual(ha["aux_threat"], hb["aux_threat"])

    def test_eval_schedule(self):
        """Which steps evaluate: every eval_every by default; early / late cadence; the final step always once."""
        c = _smoke_cfg(eval_every=1500)
        for total in (5000, 6000, 115000):
            want = [s for s in range(1500, total + 1, 1500)]
            if want[-1] != total:
                want.append(total)
            self.assertEqual(c.eval_steps(total), want, total)
            for s in range(1, total + 1, 7):                              # in-loop rule: multiples of eval_every
                self.assertEqual(c.evaluates_at(s, total), s % 1500 == 0)
        e = _smoke_cfg(eval_every=1500, eval_every_early=3000, eval_late_frac=0.8)
        self.assertEqual(e.eval_steps(5000), [3000, 4500, 5000])
        long = e.eval_steps(115000)
        self.assertEqual(long, list(range(3000, 92000, 3000)) + list(range(93000, 115000, 1500)) + [115000])
        self.assertEqual((len(long), long[29], long[30]), (46, 90000, 93000))
        self.assertEqual(_smoke_cfg(eval_every=1500, eval_every_early=3000, eval_late_frac=0.0).eval_steps(115000),
                         c.eval_steps(115000))
        self.assertEqual(_smoke_cfg(eval_every=1500, eval_every_early=3000, eval_late_frac=1.0).eval_steps(10000),
                         [3000, 6000, 9000, 10000])
        self.assertEqual(_smoke_cfg(eval_every=0, eval_every_early=3).eval_steps(10), [3, 6, 10])   # 9: late
        self.assertEqual(_smoke_cfg(eval_every=0).eval_steps(10), [10])
        # in a real run: evals (history, ckpt files) at exactly eval_steps(total)
        for over, want in (({"max_steps": 10, "eval_every": 2, "eval_every_early": 4, "eval_late_frac": 0.5},
                            [4, 6, 8, 10]),
                           ({"max_steps": 9, "eval_every": 4, "eval_every_early": 3, "eval_late_frac": 0.5},
                            [3, 8, 9])):
            cfg = _smoke_cfg(keep_best=100, **over)
            self.assertEqual(cfg.eval_steps(over["max_steps"]), want)
            out = _tmpd(self) / "sched"
            res = il_v7.train(cfg, [shard_root()], str(out), log=lambda m: None)
            self.assertEqual([h["eval"]["step"] for h in res["history"] if "eval" in h], want, over)
            self.assertEqual(sorted(p.name for p in out.glob("ckpt-*.pt")), [f"ckpt-{s:07d}.pt" for s in want])
            self.assertEqual(os.stat(out / "last.pt").st_ino, os.stat(out / f"ckpt-{want[-1]:07d}.pt").st_ino)

    def test_ckpt_every_resume_reproduces_uninterrupted_run(self):
        """A run that dies after a resume-only checkpoint (step 6) and is resumed from last.pt ends exactly
        like the uninterrupted run (raw / EMA weights, AdamW moments, per-step losses, ranking)."""
        import torch
        cfg = _smoke_cfg(max_steps=8, eval_every=4, ckpt_every=3)
        a = _tmpd(self) / "a"
        ra = il_v7.train(cfg, [shard_root()], str(a), log=lambda m: None)
        b = _tmpd(self) / "b"

        def crash(m):
            if m.startswith('{"step": 7,'):
                raise _SPCrash("simulated crash after step 7 (no signal checkpoint)")
        with self.assertRaises(_SPCrash):
            il_v7.train(cfg, [shard_root()], str(b), log=crash)
        self.assertEqual(sorted(p.name for p in b.glob("resume-*.pt")), ["resume-0000003.pt", "resume-0000006.pt"])
        self.assertEqual(os.stat(b / "last.pt").st_ino, os.stat(b / "resume-0000006.pt").st_ino)
        r6 = _ck(b / "resume-0000006.pt")
        self.assertEqual((r6["step"], r6["metrics"], r6["resume"]["step"]), (6, {"resume_only": True, "step": 6}, 6))
        self.assertEqual([x[1] for x in json.loads((b / "best.json").read_text())], [4])
        rb = il_v7.train(_smoke_cfg(max_steps=8, eval_every=4), [shard_root()], str(b), log=lambda m: None,
                         resume=True)                                   # ckpt_every is resume-free
        self.assertEqual((rb["stop_reason"], rb["steps"], rb["resumes"]), ("total_steps", 8, [6]))
        fa, fb = _ck(a / "ckpt-0000008.pt"), _ck(b / "ckpt-0000008.pt")
        for key in ("model_ema", "model_raw"):
            for k, v in fa[key].items():
                self.assertTrue(torch.equal(v, fb[key][k]), (key, k))
        for (ka, va), (kb, vb) in zip(sorted(fa["optimizer"]["state"].items()),
                                      sorted(fb["optimizer"]["state"].items())):
            for f in ("exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(va[f], vb[f]), (ka, f))
        la = [h for h in ra["history"] if "loss" in h]
        lb = [h for h in rb["history"] if "loss" in h]
        self.assertEqual([h["step"] for h in lb], list(range(1, 9)))
        for x, y in zip(la, lb):
            self.assertEqual(x["loss"], y["loss"], x["step"])
        self.assertEqual([x[1] for x in ra["best"]], [x[1] for x in rb["best"]])
        self.assertEqual(json.loads((a / "best.json").read_text()), [[x[0], x[1], x[2].replace(str(b), str(a))]
                                                                     for x in json.loads((b / "best.json").read_text())])

    def test_ckpt_every_pruning_and_ranking(self):
        """Resume-only checkpoints: only on non-eval steps, the newest 2 kept, never in best.json / best.pt;
        ranked ckpt-*.pt files follow keep_best as before."""
        logs = []
        out = _tmpd(self) / "p"
        res = il_v7.train(_smoke_cfg(max_steps=8, eval_every=4, ckpt_every=1, keep_best=1), [shard_root()],
                          str(out), log=logs.append)
        ev = [json.loads(m) for m in logs if m.startswith('{"event": "resume_checkpoint"')]
        self.assertEqual([e["step"] for e in ev], [1, 2, 3, 5, 6, 7])
        self.assertEqual([e["pruned"] for e in ev], [[], [], ["resume-0000001.pt"], ["resume-0000002.pt"],
                                                     ["resume-0000003.pt"], ["resume-0000005.pt"]])
        self.assertEqual(sorted(p.name for p in out.glob("resume-*.pt")), ["resume-0000006.pt", "resume-0000007.pt"])
        best = json.loads((out / "best.json").read_text())
        self.assertEqual(len(best), 1)
        self.assertIn(best[0][1], (4, 8))
        self.assertEqual(sorted(p.name for p in out.glob("ckpt-*.pt")), [f"ckpt-{best[0][1]:07d}.pt"])
        self.assertEqual(os.stat(out / "best.pt").st_ino, os.stat(out / f"ckpt-{best[0][1]:07d}.pt").st_ino)
        self.assertEqual([x[1] for x in res["best"]], [best[0][1]])
        self.assertTrue(all("resume-" not in x[2] for x in res["best"]))
        # no resume-only checkpoint without ckpt_every, or in a benchmark run
        out2, bm = _tmpd(self) / "q", _tmpd(self) / "bm"
        il_v7.train(_smoke_cfg(max_steps=4, eval_every=2), [shard_root()], str(out2), log=lambda m: None)
        il_v7.train(_smoke_cfg(max_steps=4, eval_every=2, ckpt_every=1), [shard_root()], str(bm),
                    log=lambda m: None, benchmark_steps=3)
        self.assertEqual(list(out2.glob("resume-*.pt")), [])
        self.assertTrue(bm.is_dir())
        self.assertEqual(list(bm.glob("resume-*.pt")), [])

    def test_loader_prefetch(self):
        cfg = _smoke_cfg()
        src = source()
        ds = il_v7.EpisodeBatches(src, il_v7.chunk_episodes(src, 0, cfg.max_seq_rows, cfg.burn_in_rows),
                                  tables(), cfg, train=True)
        self.assertEqual(il_v7.make_loader(ds, 2).prefetch_factor, 4)            # default
        self.assertEqual(il_v7.make_loader(ds, 2, prefetch=8).prefetch_factor, 8)
        self.assertIsNone(il_v7.make_loader(ds, 0, prefetch=8).prefetch_factor)

if __name__ == "__main__":
    unittest.main()
