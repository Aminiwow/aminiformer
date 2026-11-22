"""Human-consistency drift evaluator for the PPO run: the producer of the monitor's ``drift_alarm``.

Every ``monitor.drift_every`` updates the current policy is evaluated with ``il_v7.evaluate()`` (the IL
trainer's own evaluation entry: per-head NLL, gate AUC, predicted / observed action calibration) on the FIXED
IL selection subset: the ``drift.subset_games`` val games with the smallest sha256(replay_id)
(``il_v7.cap_val_items``) or the pinned ``val_games_file`` of the IL run, friendly games left out
(``drift.exclude_friendly``). The same subset with the base IL weights gives the reference, evaluated once and
kept in :meth:`DriftEvaluator.state_dict` (both models with ``gate_bias`` None, as the PPO policy acts).

PPO does not select on human NLL, so the result is an ALARM only (``Monitor`` rule ``drift``; never a stop).
The thresholds are judged values from ``cfg.drift`` (fixed data and weights make the evaluation deterministic,
so they are not noise multiples):

* per-head NLL (``gate`` / ``slot`` / ``cell`` / ``ability``, nats per weighted row) above IL by more than
  ``drift.nll_increase_max[head]``;
* gate AUC (action vs wait) below IL by more than ``drift.gate_auc_drop_max``;
* plays-per-minute calibration: per stratum (base, threat) the ratio (predicted / observed actions on wait
  rows) of the policy over IL's outside ``drift.calibration_ratio_band``.

A head / stratum with no rows on the subset (NaN in both) is not judged; a value that turns non-finite while
IL's is finite is an alarm. :meth:`DriftResult.metrics` gives the monitor keys: ``drift_alarm`` = the list of
reasons (empty = no alarm; ``monitor._flag`` reads it) plus ``drift/<delta>`` details.

Held-out games of our own deck, when they exist, are passed as ``items`` by the caller (a second evaluator).

Run wiring (the run driver's ``--drift-data ...`` options): :class:`DriftSpec` holds the CLI values,
:class:`DriftRun` sets the check up once per block (the base IL checkpoint, sha256-checked against the run state,
evaluated once per run with ``gate_bias`` None and kept in ``RUN/drift/reference.json``; the tables digest and the
generation checked against the learner and the IL checkpoint) and runs it on drift updates
(``RUN/drift/uNNNNNN.json`` per check). :func:`side_items` selects our deck's held-out sides (a second evaluator,
reasons prefixed ``deck: ``, metrics ``drift/deck/...``). Off unless the run asks for it; alarm only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from amini.train import il_v7 as IL

DRIFT_SCHEMA = "ppo-v7-drift.1"
HEADS: Tuple[str, ...] = ("gate", "slot", "cell", "ability")      # il_v7.evaluate()['terms'] policy heads
STRATA: Tuple[str, ...] = ("base", "threat")                       # il_v7.evaluate()['calibration'] strata


def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ----------------------------------------------------------------------------- the fixed subset
def selection_items(src: Any, train_cfg: Any, *, max_games: int, exclude_friendly: bool = True,
                    games_file: Optional[str] = None) -> Tuple[List[Any], Dict[str, Any]]:
    """The fixed selection subset as ``il_v7.train`` builds its val subset: val chunks
    (``chunk_episodes(split=1)``), then the pinned ``games_file`` (friendly games dropped after it) or the
    friendly drop followed by ``cap_val_items(max_games)`` (smallest sha256 replay ids). Returns
    ``(items, record)``."""
    items = IL.chunk_episodes(src, 1, train_cfg.max_seq_rows, train_cfg.burn_in_rows)
    rec: Dict[str, Any] = {"max_games": int(max_games), "exclude_friendly": bool(exclude_friendly)}
    fr = IL.friendly_games(src) if exclude_friendly else set()
    if games_file:
        items, rec["games_file_sha256"] = IL.pinned_val_items(src, items, games_file, max_games)
        if fr:
            items, rec["friendly_dropped"] = IL.drop_games(items, fr)
    else:
        if fr:
            items, rec["friendly_dropped"] = IL.drop_games(items, fr)
        items = IL.cap_val_items(src, items, max_games)
    return items, rec


def subset_digest(src: Any, items: Sequence[Any]) -> str:
    """sha256 of the subset's (game tag, side, chunk bounds relative to the episode start), sorted: equal
    subsets agree whatever the shard layout."""
    keys = sorted({json.dumps([IL._tag_text(src, it[0][0], it[0][1]), int(it[0][2]), int(it[1]) - int(it[0][3]),
                               int(it[2]) - int(it[0][3]), int(it[3]) - int(it[0][3])]) for it in items})
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- judging
def judge(cfg: Any, reference: Mapping[str, Any], report: Mapping[str, Any]
          ) -> Tuple[Tuple[str, ...], Dict[str, Optional[float]]]:
    """``(reasons, deltas)`` of one evaluation against the IL reference with the ``cfg.drift`` thresholds
    (module docstring). ``deltas``: ``nll_delta/<head>`` (policy - IL), ``gate_auc_drop`` (IL - policy) and
    ``calibration_rel/<stratum>`` (policy ratio / IL ratio); None where not judged."""
    dc = cfg.drift
    reasons: List[str] = []
    deltas: Dict[str, Optional[float]] = {}
    for h in HEADS:
        ref, got = _num(reference["terms"].get(h)), _num(report["terms"].get(h))
        lim = float(dc.nll_increase_max[h])
        if ref is None:
            deltas[f"nll_delta/{h}"] = None
            continue
        if got is None:
            deltas[f"nll_delta/{h}"] = None
            reasons.append(f"{h} NLL not finite (IL {ref:.4f})")
            continue
        d = got - ref
        deltas[f"nll_delta/{h}"] = d
        if d > lim:
            reasons.append(f"{h} NLL +{d:.4f} over IL (limit {lim})")
    ref, got = _num(reference.get("gate_auc")), _num(report.get("gate_auc"))
    deltas["gate_auc_drop"] = None
    if ref is not None:
        if got is None:
            reasons.append(f"gate AUC not finite (IL {ref:.4f})")
        else:
            deltas["gate_auc_drop"] = ref - got
            if ref - got > float(dc.gate_auc_drop_max):
                reasons.append(f"gate AUC {got:.4f} is {ref - got:.4f} below IL (limit {dc.gate_auc_drop_max})")
    lo, hi = (float(x) for x in dc.calibration_ratio_band)
    for s in STRATA:
        r0 = _num(((reference.get("calibration") or {}).get(s) or {}).get("ratio"))
        r = _num(((report.get("calibration") or {}).get(s) or {}).get("ratio"))
        deltas[f"calibration_rel/{s}"] = None
        if r0 is None or r0 <= 0.0:
            continue
        if r is None:
            reasons.append(f"{s} calibration not finite (IL ratio {r0:.4f})")
            continue
        q = r / r0
        deltas[f"calibration_rel/{s}"] = q
        if not lo <= q <= hi:
            reasons.append(f"{s} plays calibration {q:.3f}x IL outside [{lo}, {hi}]")
    return tuple(reasons), deltas


@dataclass(frozen=True)
class DriftResult:
    """One drift check: ``alarm`` iff ``reasons`` is non-empty; ``report`` is ``il_v7.evaluate``'s."""
    update: int
    reasons: Tuple[str, ...]
    deltas: Dict[str, Optional[float]]
    report: Dict[str, Any]
    subset_digest: str
    n_games: int
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def alarm(self) -> bool:
        return bool(self.reasons)

    def metrics(self) -> Dict[str, Any]:
        """Monitor keys: ``drift_alarm`` (list of reasons, empty = False) and ``drift/<delta>`` (NaN = not
        judged), for ``Learner.update(extra_metrics=...)`` on a drift update."""
        out: Dict[str, Any] = {"drift_alarm": list(self.reasons)}
        out.update({f"drift/{k}": (float("nan") if v is None else float(v)) for k, v in self.deltas.items()})
        out["drift/selection_nll"] = float(self.report.get("selection_nll", float("nan")))
        return out


def drift_due(cfg: Any, update: int) -> bool:
    """True on the updates where the monitor expects ``drift_alarm`` (``update % monitor.drift_every == 0``)."""
    every = int(cfg.monitor.drift_every)
    return every > 0 and int(update) % every == 0


# ----------------------------------------------------------------------------- evaluator
class DriftEvaluator:
    """Drift on a fixed subset: :meth:`set_reference` once with the base IL policy, then :meth:`check` with
    the current policy on every drift update. ``train_cfg`` is the IL run's ``il_v7.TrainConfig`` (window,
    burn-in, loss weights, micro-batch caps, tables); ``items`` replaces the default subset
    (:func:`selection_items` with ``cfg.drift.subset_games``)."""

    def __init__(self, cfg: Any, train_cfg: Any, src: Any, tables: Any, *, items: Optional[Sequence[Any]] = None,
                 games_file: Optional[str] = None, device: Any = "cpu", amp_dtype: torch.dtype = torch.float32,
                 reference: Optional[Mapping[str, Any]] = None) -> None:
        self.cfg, self.train_cfg, self.src, self.tables = cfg, train_cfg, src, tables
        if items is None:
            items, rec = selection_items(src, train_cfg, max_games=int(cfg.drift.subset_games),
                                         exclude_friendly=bool(cfg.drift.exclude_friendly), games_file=games_file)
        else:
            items, rec = list(items), {"given": True}
        if not items:
            raise ValueError("drift: the selection subset is empty")
        self.items = list(items)
        self.selection = rec
        self.n_games = len({(it[0][0], it[0][1]) for it in self.items})
        self.digest = subset_digest(src, self.items)
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype
        self.reference: Optional[Dict[str, Any]] = None
        if reference is not None:
            self.load_state_dict(reference)

    @classmethod
    def open(cls, cfg: Any, train_cfg: Any, data: Sequence[str], **kw: Any) -> "DriftEvaluator":
        """Open the IL generation (``ShardSource.open``), build and verify its tables, then construct."""
        src, _ = IL.ShardSource.open(list(data))
        tables = build_tables_at_root(train_cfg)
        IL.verify_tables(tables, src)
        return cls(cfg, train_cfg, src, tables, **kw)

    def evaluate(self, model: torch.nn.Module) -> Dict[str, Any]:
        """``il_v7.evaluate`` of ``model`` on the subset (whole subset, parity draws); the model's training
        mode is restored afterwards."""
        ds = IL.EpisodeBatches(self.src, self.items, self.tables, self.train_cfg, train=False)
        was = model.training
        try:
            return IL.evaluate(model, IL.make_loader(ds, 0), self.device, self.amp_dtype, self.train_cfg.loss, 0,
                               opp_hand_reduce=self.train_cfg.aux_opp_hand_reduce)
        finally:
            model.train(was)

    @staticmethod
    def _require_unbiased(model: torch.nn.Module, what: str) -> None:
        """Both sides of the comparison act without a gate calibration bias, as the PPO policy does (module
        docstring): ``il_v7.load_policy`` keeps a checkpoint's bias, so strip it first (``learner.make_reference``
        or a :func:`learner.ppo_config` model)."""
        gb = getattr(getattr(model, "cfg", None), "gate_bias", None)
        if gb is not None:
            raise ValueError(f"drift: the {what} has gate_bias {gb!r}; evaluate it with gate_bias None "
                             "(learner.make_reference strips it)")

    def set_reference(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Evaluate the IL reference (the base IL model with gate_bias None, refused otherwise) once; kept in
        :meth:`state_dict`."""
        self._require_unbiased(model, "IL reference")
        self.reference = self.evaluate(model)
        return self.reference

    def due(self, update: int) -> bool:
        return drift_due(self.cfg, update)

    def check(self, model: torch.nn.Module, update: int) -> DriftResult:
        """Evaluate ``model`` and judge it against the reference (:func:`judge`)."""
        if self.reference is None:
            raise RuntimeError("drift: set_reference(base IL model) first")
        self._require_unbiased(model, "policy")
        rep = self.evaluate(model)
        reasons, deltas = judge(self.cfg, self.reference, rep)
        return DriftResult(update=int(update), reasons=reasons, deltas=deltas, report=rep,
                           subset_digest=self.digest, n_games=self.n_games)

    def state_dict(self) -> Dict[str, Any]:
        """The IL reference report with the subset identity (saved with the run, so a chained block does not
        re-evaluate the base IL model)."""
        return {"schema": DRIFT_SCHEMA, "subset_digest": self.digest, "n_games": self.n_games,
                "reference": self.reference}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        if d.get("schema") != DRIFT_SCHEMA or d.get("subset_digest") != self.digest:
            raise ValueError(f"drift state is for another subset ({str(d.get('subset_digest'))[:12]} != "
                             f"{self.digest[:12]}) or schema")
        self.reference = None if d.get("reference") is None else dict(d["reference"])


# ----------------------------------------------------------------------------- run wiring
DRIFT_RUN_SCHEMA = "ppo-v7-drift-run.1"        # RUN/drift/reference.json
DRIFT_CHECK_SCHEMA = "ppo-v7-drift-check.1"    # RUN/drift/uNNNNNN.json
DRIFT_AMP: Dict[str, torch.dtype] = {"bf16": torch.bfloat16, "fp32": torch.float32}
REFERENCE_NAME = "reference.json"
RELOAD_KEYS: Tuple[str, ...] = ("il_sha256", "amp", "train_config_sha256", "games_file_sha256", "deck_sides_sha256",
                                 "generation")
OFFLINE_PREFIX = "offline-"                   # RUN/drift/offline-uNNNNNN.json: an after-the-fact check of a checkpoint


class DriftSetupError(ValueError):
    """The drift check cannot be set up as asked: the block refuses before anything starts."""


def _file_sha(path: Optional[str]) -> Optional[str]:
    return None if not path else hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, sort_keys=True, indent=1, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_sides(path: str) -> List[Tuple[str, int]]:
    """``replay_id<TAB or space>side`` lines (``#`` comments, blank lines skipped) of our deck's held-out sides."""
    out: List[Tuple[str, int]] = []
    for n, ln in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 2 or parts[1] not in ("0", "1"):
            raise DriftSetupError(f"{path}:{n}: expected 'replay_id side' (side 0 or 1), got {ln!r}")
        out.append((parts[0], int(parts[1])))
    return out


def side_items(src: Any, train_cfg: Any, sides: Iterable[Tuple[str, int]], *, exclude_friendly: bool = True
               ) -> Tuple[List[Any], Dict[str, Any]]:
    """Our deck's held-out sides: the VAL chunks (``chunk_episodes(split=1)``;
    a train chunk is never returned) whose (replay id, side) is listed, friendly games dropped even when listed.
    Returns ``(items, record)``; ``missing_sides`` counts listed sides with no val chunk left."""
    want = {(str(r), int(s)) for r, s in sides}
    items = IL.chunk_episodes(src, 1, train_cfg.max_seq_rows, train_cfg.burn_in_rows)
    tags: Dict[Tuple[int, int], str] = {}
    keep = []
    for it in items:
        g = (it[0][0], it[0][1])
        if g not in tags:
            tags[g] = IL._tag_text(src, *g)
        if (tags[g], int(it[0][2])) in want:
            keep.append(it)
    rec: Dict[str, Any] = {"listed": len(want), "exclude_friendly": bool(exclude_friendly)}
    if exclude_friendly and keep:
        fr = IL.friendly_games(src)
        if fr:
            keep, rec["friendly_dropped"] = IL.drop_games(keep, fr)
    found = {(tags[(it[0][0], it[0][1])], int(it[0][2])) for it in keep}
    rec["found_sides"], rec["missing_sides"] = len(found), len(want - found)
    return keep, rec


def deck_sides(src: Any, deck_ids: Sequence[int], *, split: int = 1, exclude_friendly: bool = True
               ) -> Tuple[List[Tuple[str, int]], Dict[str, Any]]:
    """Every (replay id, side) of ``split`` (1 = val) in ``src`` whose exact deck is ``deck_ids`` (the sorted 8
    ``deck_card`` ids of that side in the shards' ``games`` table), friendly games left out
    (``il_v7.friendly_games``), superseded games never seen (``src.episodes`` skips them). The side index is the
    ``games`` row's own: ``deck_card[side]`` and the IL episode ``(shard, game, side, ...)`` index the same
    player, so no join with another generation is needed.
    Returns ``(sorted unique sides, record)``; the input of :func:`side_items`."""
    want = tuple(sorted(int(x) for x in deck_ids))
    if len(want) != 8 or len(set(want)) != 8:
        raise DriftSetupError(f"drift-deck: an exact deck is 8 distinct card ids, got {list(deck_ids)}")
    fr = IL.friendly_games(src) if exclude_friendly else set()
    out = set()
    friendly = set()
    games = set()
    for ep in src.episodes:
        si, gi, side = int(ep[0]), int(ep[1]), int(ep[2])
        if src.split_of(ep) != int(split):
            continue
        g = src.shards[si]["games"][gi]
        if tuple(sorted(int(x) for x in g["deck_card"][side])) != want:
            continue
        if (si, gi) in fr:
            friendly.add((si, gi, side))
            continue
        out.add((IL._tag_text(src, si, gi), side))
        games.add((si, gi))
    rec = {"deck_ids": list(want), "split": int(split), "exclude_friendly": bool(exclude_friendly),
           "sides": len(out), "games": len(games), "friendly_sides_dropped": len(friendly)}
    return sorted(out), rec


@dataclass(frozen=True)
class DriftSpec:
    """The run driver's drift options (CLI values, never config keys)."""
    data: Tuple[str, ...]
    train_config: Optional[str] = None
    games_file: Optional[str] = None
    deck_sides: Optional[str] = None
    amp: str = "bf16"

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", tuple(str(x) for x in (self.data or ())))
        if not self.data:
            raise DriftSetupError("drift: --drift-data names no IL generation")
        if self.amp not in DRIFT_AMP:
            raise DriftSetupError(f"drift: amp {self.amp!r} not in {sorted(DRIFT_AMP)}")

    def to_dict(self) -> Dict[str, Any]:
        return {"data": list(self.data), "train_config": self.train_config, "games_file": self.games_file,
                "deck_sides": self.deck_sides, "amp": self.amp}

    def missing(self) -> List[str]:
        """Paths that do not exist (the run preflight lists them before anything starts)."""
        out = [f"--drift-data {p}" for p in self.data if not Path(p).exists()]
        for opt, p in (("--drift-train-config", self.train_config), ("--drift-games-file", self.games_file),
                       ("--drift-deck-sides", self.deck_sides)):
            if p and not Path(p).is_file():
                out.append(f"{opt} {p}")
        return out


CODE_ROOT = Path(__file__).resolve().parents[3]


def build_tables_at_root(train_cfg: Any, root: Optional[Path] = None) -> Any:
    """``il_v7.build_tables`` with every relative table path read from the code root instead of the working
    directory. An IL train config names its tables relative to the code root; a block started from the run
    directory would not find them. The table digests are content hashes, so the result is the same tables."""
    root = Path(root) if root is not None else CODE_ROOT
    t = dict(getattr(train_cfg, "tables", None) or {})
    fixed = {k: (str(root / v) if isinstance(v, str) and v and not Path(v).is_absolute() else v)
             for k, v in t.items()}
    return IL.build_tables(dataclasses.replace(train_cfg, tables=fixed) if fixed != t else train_cfg)


def _cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class DriftRun:
    """The drift check of one PPO run (the run driver's ``--drift-data``): :meth:`setup` once per block,
    :meth:`check` on every drift update. Files under ``root`` (``RUN/drift``): ``reference.json``
    (:data:`DRIFT_RUN_SCHEMA`, written once per run) and one ``uNNNNNN.json`` per check (:data:`DRIFT_CHECK_SCHEMA`).

    The reference is the BASE IL checkpoint of the run (``state['il_checkpoint']``, its sha256 checked against
    ``state['il_sha256']``), loaded with ``learner.load_source_policy`` and made unbiased with
    ``learner.make_reference``; it is never ``learner.reference`` (which follows the champion after a promotion)."""

    def __init__(self, cfg: Any, spec: DriftSpec, root: Path):
        self.cfg, self.spec, self.root = cfg, spec, Path(root)
        self.main: Optional[DriftEvaluator] = None
        self.deck: Optional[DriftEvaluator] = None
        self.record: Dict[str, Any] = {}

    @property
    def reference_path(self) -> Path:
        return self.root / REFERENCE_NAME

    def check_path(self, update: int, *, offline: bool = False) -> Path:
        return self.root / f"{OFFLINE_PREFIX if offline else ''}u{int(update):06d}.json"

    def _train_config(self, ck: Mapping[str, Any]) -> Tuple[Any, str]:
        if self.spec.train_config:
            return IL.TrainConfig.from_file(self.spec.train_config), str(self.spec.train_config)
        d = ck.get("train_config")
        if not isinstance(d, dict):
            raise DriftSetupError("drift: the base IL checkpoint has no train_config; pass --drift-train-config")
        names = {f for f in IL.TrainConfig.__dataclass_fields__}
        unknown = set(d) - names
        if unknown:
            raise DriftSetupError(f"drift: the IL checkpoint's train_config has unknown keys {sorted(unknown)}; "
                                  "pass --drift-train-config")
        return IL.TrainConfig(**d), "il_checkpoint"

    def setup(self, *, il_checkpoint: str, il_sha256: Optional[str], tables_digest: Optional[str], device: Any,
              load_model: Optional[Callable[[str, Any], torch.nn.Module]] = None) -> Dict[str, Any]:
        """Open the subsets, check the tables and the generation, load or evaluate the reference. Returns the
        record for ``state['drift']``. Any mismatch raises :class:`DriftSetupError` naming the field."""
        from amini.ppo import config as CF
        from amini.ppo import learner as LR
        t_open = time.monotonic()
        path = Path(str(il_checkpoint))
        if not path.is_file():
            raise DriftSetupError(f"drift: base IL checkpoint {path} not found")
        sha = IL.file_sha256(str(path))
        if il_sha256 and sha != il_sha256:
            raise DriftSetupError(f"drift: il_sha256: {path} is {sha[:12]}, the run state says {str(il_sha256)[:12]}")
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        tc, tc_src = self._train_config(ck)
        tc_sha = hashlib.sha256(json.dumps(asdict(tc), sort_keys=True, default=str).encode()).hexdigest()
        src, gen = IL.ShardSource.open(list(self.spec.data))
        tables = build_tables_at_root(tc)
        trec = IL.verify_tables(tables, src)
        td = CF.model_tables_digest(IL._tables_digest_map(trec))
        flags: List[str] = []
        if tables_digest is None:
            flags.append("drift_tables_unchecked")      # the learner's compat names no tables digest
        elif td != tables_digest:
            raise DriftSetupError(f"drift: tables: the generation's tables digest {str(td)[:12]} != the learner's "
                                  f"{str(tables_digest)[:12]} (the policy would be scored on another tokenisation)")
        games_file = self.spec.games_file or (getattr(tc, "val_games_file", "") or None)
        il_gen = ck.get("generation")
        if not (il_gen and gen):
            flags.append("drift_generation_unchecked")  # the IL checkpoint or the data names no generation
        elif str(gen) != str(il_gen) and not games_file:
            raise DriftSetupError(f"drift: generation: --drift-data is {gen}, the base IL trained on {il_gen}; "
                                  "without a games file the subset would not be the IL selection subset")
        amp = DRIFT_AMP[self.spec.amp]
        try:
            self.main = DriftEvaluator(self.cfg, tc, src, tables, games_file=games_file, device=device,
                                       amp_dtype=amp)
        except ValueError as e:
            raise DriftSetupError(f"drift: main subset: {e}") from e
        deck_rec: Optional[Dict[str, Any]] = None
        if self.spec.deck_sides:
            items, deck_rec = side_items(src, tc, read_sides(self.spec.deck_sides),
                                         exclude_friendly=bool(self.cfg.drift.exclude_friendly))
            if items:
                self.deck = DriftEvaluator(self.cfg, tc, src, tables, items=items, device=device, amp_dtype=amp)
            else:
                flags.append("drift_deck_empty")
        else:
            flags.append("drift_deck_absent")
        expect = {"il_sha256": sha, "amp": self.spec.amp, "train_config_sha256": tc_sha,
                  "games_file_sha256": _file_sha(games_file), "deck_sides_sha256": _file_sha(self.spec.deck_sides),
                  "generation": gen}
        open_s = time.monotonic() - t_open
        ref_s = None
        if self.reference_path.exists():
            d = json.loads(self.reference_path.read_text())
            if d.get("schema") != DRIFT_RUN_SCHEMA:
                raise DriftSetupError(f"drift: {self.reference_path} schema {d.get('schema')!r} != {DRIFT_RUN_SCHEMA}")
            for k in RELOAD_KEYS:
                if d.get(k) != expect[k]:
                    raise DriftSetupError(f"drift: {k}: {self.reference_path} has {str(d.get(k))[:16]}, this block "
                                          f"has {str(expect[k])[:16]} (the reference is fixed for the run)")
            try:
                self.main.load_state_dict(d.get("main") or {})
                if self.deck is not None:
                    if not d.get("deck"):
                        raise ValueError("the reference has no deck subset")
                    self.deck.load_state_dict(d["deck"])
                elif d.get("deck"):
                    raise ValueError("the reference has a deck subset this block does not")
            except ValueError as e:
                raise DriftSetupError(f"drift: subset: {e}") from e
            evaluated = False
        else:
            t0 = time.monotonic()
            if load_model is not None:
                model = load_model(str(path), device)
            else:
                model = LR.load_source_policy(str(path), device=device)[0]
            ref = LR.make_reference(model)
            self.main.set_reference(ref)
            if self.deck is not None:
                self.deck.set_reference(ref)
            del model, ref
            _cuda_empty_cache()
            ref_s = time.monotonic() - t0
            _atomic_json(self.reference_path, dict(
                expect, schema=DRIFT_RUN_SCHEMA, main=self.main.state_dict(),
                deck=None if self.deck is None else self.deck.state_dict(), il_checkpoint=str(path),
                data=list(self.spec.data), il_generation=il_gen, train_config_source=tc_src,
                games_file=games_file, deck_sides=self.spec.deck_sides, selection=self.main.selection,
                deck_selection=deck_rec, n_games=self.main.n_games,
                deck_n_games=None if self.deck is None else self.deck.n_games, reference_eval_s=ref_s,
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            evaluated = True
        self.record = {"reference": str(self.reference_path), "spec": self.spec.to_dict(),
                       "subset_digest": self.main.digest, "n_games": self.main.n_games,
                       "deck_subset_digest": None if self.deck is None else self.deck.digest,
                       "deck_n_games": 0 if self.deck is None else self.deck.n_games, "flags": flags,
                       "generation": gen, "games_file": games_file, "reference_evaluated": evaluated,
                       "reference_eval_s": ref_s, "open_s": open_s}
        return dict(self.record)

    def check(self, model: torch.nn.Module, update: int, *, policy_digest: Optional[str] = None,
              offline: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """One check: the main subset (and the deck subset), merged into the monitor metrics
        (``drift_alarm`` = every reason, the deck's prefixed ``deck: ``; ``drift/...`` and ``drift/deck/...``); the
        full record is written to ``uNNNNNN.json`` (``offline``: ``offline-uNNNNNN.json`` with that mapping under
        ``offline``, for an after-the-fact check of a checkpoint). Returns ``(metrics, record)``."""
        if self.main is None:
            raise RuntimeError("drift: setup() first")
        t0 = time.monotonic()
        cuda = torch.cuda.is_available() and self.main.device.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.main.device)
        r = self.main.check(model, update)
        metrics = r.metrics()
        reasons = list(r.reasons)
        rec: Dict[str, Any] = {"schema": DRIFT_CHECK_SCHEMA, "update": int(update), "policy_digest": policy_digest,
                               "main": {"reasons": list(r.reasons), "deltas": r.deltas, "report": r.report,
                                        "subset_digest": r.subset_digest, "n_games": r.n_games}}
        if self.deck is not None:
            rd = self.deck.check(model, update)
            reasons += [f"deck: {x}" for x in rd.reasons]
            md = rd.metrics()
            metrics.update({f"drift/deck/{k[len('drift/'):]}": v for k, v in md.items() if k.startswith("drift/")})
            rec["deck"] = {"reasons": list(rd.reasons), "deltas": rd.deltas, "report": rd.report,
                           "subset_digest": rd.subset_digest, "n_games": rd.n_games}
        metrics["drift_alarm"] = reasons
        rec["reasons"] = reasons
        rec["seconds"] = time.monotonic() - t0
        rec["peak_gb"] = (torch.cuda.max_memory_allocated(self.main.device) / 2 ** 30) if cuda else None
        if offline is not None:
            rec["offline"] = dict(offline)
        _atomic_json(self.check_path(update, offline=offline is not None), rec)
        return metrics, rec


__all__ = ["DRIFT_SCHEMA", "HEADS", "STRATA", "selection_items", "subset_digest", "judge", "DriftResult",
           "drift_due", "DriftEvaluator", "DRIFT_RUN_SCHEMA", "DRIFT_CHECK_SCHEMA", "DRIFT_AMP", "DriftSetupError",
           "read_sides", "side_items", "deck_sides", "DriftSpec", "DriftRun", "OFFLINE_PREFIX"]
