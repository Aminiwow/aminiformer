"""Imitation-learning trainer for AminiFormer on ``amini-shards-v7`` training shards.

Recipe (defaults in ``TrainConfig``; the coefficients are example values):

1. ``group_drop_p = 0`` for every group (no drop code path exists);
2. loss: gate CE weighted by the row's sampling weight ``R.weight`` (1 for action rows, the inverse sampling
   rate of its stratum for committed-wait rows) and normalised by its sum over all rows (family mask applied);
   slot / cell / ability-pointer CE on action rows with their label masks; every aux head with a small
   weight, each normalised over its own valid targets; value and value_priv as weighted MSE;
3. stage S on all play rows, all threat-stratum rows (``stage_s_threat_frac``: a hash subset) and
   (``stage_s_base_frac``) base rows;
4. attention / FFN dropout, AdamW (no weight decay on norms, biases, embeddings), weight EMA used for
   evaluation and the shipped checkpoint;
5. checkpoint selection: lowest weighted policy NLL on the val split, on EMA weights
   (gate + slot + cell + ability; aux and value excluded), reported per stratum;
6. calibration check (predicted vs observed actions per stratum, flagged when they differ by more than 5 %);
7. gate AUC / accuracy weighted like the gate loss; top-1 diagnostics only.

The trainer holds ``LOCK_SH`` on the training lock for the whole run and records
the generation it uses; the lock and the holder record are taken before any table
is memmapped.  It refuses an unsupported Python before touching data.

Memory: an optimizer step of ``batch_rows`` rows is split into
micro-batches of at most ``micro_batch_rows`` rows and ``max_batch_tokens`` padded
step-encoder tokens (gradient accumulation; every loss term is normalised by its
step-wide denominator, so the objective equals the unsplit step), and
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is set by default.

Run control: the run lasts ``max_epochs`` epochs of the train split (cosine horizon) or
``max_steps``; it stops early after ``early_stop_patience`` evals (EMA val selection NLL,
deterministic val subset of ``val_max_games`` games) without improvement; SIGINT/SIGTERM
finish the current step, save a resumable ``last.pt`` (``--resume``) and exit 0.  Every
stop writes ``result.json`` (``stop_reason``, best checkpoint, train rows/s without eval
time, eval seconds, pre-clip gradient-norm median / p90 / clipped fraction).

Tokenizer tables come from the frozen table files named in the config and must equal
the digests the MANIFESTs pin; stand-in synthetic tables need ``tables: {synthetic:
true}`` and are refused on non-synthetic shards.  The tokenizer (``amini.nn.tokenizer_v7``) and the
shard storage / vocabulary modules (``amini.v7``) are not part of this repository; the trainer imports
them when it needs them.

    python -m amini.train.il_v7 --config CONFIG.yaml \
        --data <generation dir | shard dir ...> --out <run dir>

Warm start and per-head learning rates (all default to no-ops; see ``TrainConfig``): ``init_from`` /
``--init-from`` (warm start from a checkpoint, strict load, fresh optimizer / schedule / step / data order),
``init_weights`` (ema | model), ``eval_at_start`` / ``--eval-at-start`` (step-0 eval of the selection subset),
``lr_mult`` (``{parameter-name prefix: multiplier}`` param groups; prefixes in ``AminiFormer.HEAD_PREFIXES``;
``--set lr_mult.ability_head.=2``).
"""

from __future__ import annotations

import sys


# ------------------------------------------------------------------------- runtime guard
def runtime_check() -> None:
    """Refuse an unsupported interpreter before anything else is imported or loaded.

    Uses ``amini.v7.runtime_guard.check()`` when that optional module is installed, else an
    inline check: CPython >= 3.11 and < 3.14 with numpy < 2 (the pinned runtime is
    Python 3.11 with numpy 1.26).
    """

    try:
        from amini.v7 import runtime_guard  # type: ignore
    except ImportError:
        runtime_guard = None
    if runtime_guard is not None:
        try:
            runtime_guard.check()
        except Exception as exc:                       # RuntimeGuardError -> clean non-zero exit
            raise SystemExit(f"il_v7: refusing this interpreter: {exc}")
        return
    v = tuple(sys.version_info[:2])
    if v >= (3, 14) or v < (3, 11):
        raise SystemExit(f"il_v7: refusing Python {v[0]}.{v[1]}; use Python 3.11 to 3.13")
    try:
        import numpy
        if int(numpy.__version__.split(".")[0]) >= 2:
            raise SystemExit(f"il_v7: refusing numpy {numpy.__version__}; the trainer needs numpy < 2")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"il_v7: numpy missing ({exc})")


import argparse  # noqa: E402
import collections  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass, field, fields  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

TRAINER_VERSION = "aminiformer-v1-il.1"


def _optional_module(name: str, what: str):
    """Import ``name``, a module that is not part of this repository (the shard tokenizer, the vocabulary
    tables): a clear RuntimeError when it is not installed."""

    import importlib
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not name.startswith(exc.name):
            raise                                      # the module exists; one of ITS imports failed
        raise RuntimeError(f"{what} ({name}) is not part of this repository") from exc


# ------------------------------------------------------------------------- tiny YAML
def _scalar(v: str) -> Any:
    v = v.strip()
    if v == "" or v in ("null", "~", "None"):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v == "{}":
        return {}
    if (v[0] == v[-1]) and v[0] in "'\"":
        return v[1:-1]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [] if not inner else [_scalar(x) for x in inner.split(",")]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_yaml(text: str) -> Dict[str, Any]:
    """Minimal YAML subset: nested mappings by indentation, scalars, flow lists, comments.
    (PyYAML is used instead when importable; the pinned runtime does not ship it.)"""

    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, rest = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            child: Dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(rest)
    return root


# ------------------------------------------------------------------------- config
# loss coefficients (example values; the four policy terms add up to the selection metric)
DEFAULT_LOSS = {"gate": 1.0, "slot": 1.0, "cell": 1.0, "ability": 1.0,
                "aux_land": 0.2, "aux_emerge": 0.2, "aux_threat": 0.2, "aux_tower_dmg": 0.2,
                "aux_opp_next": 0.2, "aux_opp_hand": 0.2, "value": 0.5, "value_priv": 0.5}

# aux_opp_hand reduction over the card vocab: "sum_hand" = sum of the per-class BCE over V
# divided by HAND_SLOTS (the default); "class_mean" = mean over V.
AUX_OPP_HAND_REDUCE = ("class_mean", "sum_hand")
DEFAULT_OPP_HAND_REDUCE = "sum_hand"

# keys that may differ between a checkpoint and the config that resumes it (they do not
# change the optimisation trajectory: data order, LR schedule, loss and model are fixed)
RESUME_FREE_KEYS = frozenset({
    "num_workers", "log_every", "eval_every", "eval_max_batches", "keep_best",
    "early_stop_patience", "early_stop_min_steps", "early_stop_min_delta",
    "cuda_alloc_conf", "name",
    # speed switches: how the same function is computed (float rounding / dropout draws only), so a run
    # may be resumed with them changed (e.g. without compilation if a compiler problem appears)
    "pack_tokens", "compile_mode", "compile_dynamic", "pack_multiple", "cell_row_multiple",
    # training-loop speed switches: the same arithmetic (defer_stats, host_index, ema_foreach: bit-identical on
    # CPU) or the optimizer kernel only (adamw_impl: fp32 rounding)
    "defer_stats", "host_index", "ema_foreach", "adamw_impl",
    # runtime keys: when to evaluate / save and the loader queue depth (stage_s_threat_frac is NOT here:
    # it changes the objective)
    "eval_every_early", "eval_late_frac", "ckpt_every", "loader_prefetch", "stage_p_compiled"})
ADAMW_IMPLS = ("auto", "fused", "foreach")
COMPILE_MODES = ("", "default", "max-autotune-no-cudagraphs")
RESUME_FILE = "last.pt"                # resumable state (signal stop / every eval / every ckpt_every)
INIT_WEIGHTS = ("ema", "model")        # which weights of an init_from checkpoint initialise the run
# model-config fields that change parameter shapes or what the same weights compute: a warm start
# refuses a checkpoint that differs in any of them (dropout, cell_mask, cell_fp32, window_*,
# checkpoint_*, gate_bias may differ; the differences are recorded in the init record)
WARM_START_MODEL_KEYS = ("d_model", "n_heads", "ffn_mult", "n_step_layers", "n_temporal_layers", "d_lnk",
                         "card_vocab", "obj_vocab", "n_pop", "cell_dim", "cell_heads", "cell_layers",
                         "cell_ffn_mult", "rope_base", "priv_card_dim")
# the population name whose games ``exclude_friendly`` drops.  GAME.population holds the population id =
# the index in vocab.POPULATIONS (GameModeVocab refuses any other id order).
FRIENDLY_POPULATION = "friendly"


@dataclass
class TrainConfig:
    # optimisation defaults are example values; a run sets its own in the config file
    name: str = "main"
    model: Dict[str, Any] = field(default_factory=dict)
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 200000
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 0.5
    ema_decay: float = 0.99
    batch_rows: int = 1024             # rows per optimizer step (the effective batch)
    micro_batch_rows: int = 0          # rows per forward/backward micro-batch (0 = batch_rows)
    max_batch_tokens: int = 0          # cap on valid rows x padded tokens per micro-batch (0 = off)
    max_seq_rows: int = 256            # longest chunk of an episode (rows)
    burn_in_rows: int = 64             # context-only rows in front of a later chunk
    loss: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS))
    # label options passed through to the tokenizer's ``make_example`` (the loss does not use them)
    fid_exact: float = 1.0
    fid_inexact: float = 1.0
    stage_s_base_frac: float = 1.0
    pop_null_p: float = 0.1            # probability of a NULL population input in training (example value)
    group_drop_p: float = 0.0
    input_profile: str = "engine"
    eval_every: int = 1000
    eval_max_batches: int = 0          # 0 = the whole val split
    log_every: int = 100
    keep_best: int = 3
    num_workers: int = 4
    seed: int = 0
    force_fp16: bool = False
    # aux_opp_hand: per-row reduction over the card vocab (AUX_OPP_HAND_REDUCE) and whether the
    # head's output bias starts at the 4-of-V prior (False: bias zeroed at init, no RNG draw).
    aux_opp_hand_reduce: str = DEFAULT_OPP_HAND_REDUCE
    aux_opp_hand_prior_bias: bool = True
    # early stopping on the EMA val selection NLL (0 = off): stop after ``patience`` evals at or
    # after ``min_steps`` without an improvement larger than ``min_delta``.
    early_stop_patience: int = 0
    early_stop_min_steps: int = 0
    early_stop_min_delta: float = 0.0
    # full data: deterministic val subset of at most this many games (0 = whole val split);
    # max_epochs > 0 sets the total steps (and cosine horizon) to ceil(max_epochs x steps of
    # one train epoch), replacing max_steps.
    val_max_games: int = 0
    max_epochs: float = 0.0
    # optional pinned val selection subset: a text file of replay ids (one per line) that must all be
    # val games of the generation; replaces the smallest-sha256 rule (then val_max_games is only a cap
    # check).  Keeps the selection subset fixed across generations (disjoint from an eval holdout).
    val_games_file: str = ""
    # drop every game whose population is friendly (vocab.population_name(recorded_asset) == "friendly")
    # from the train AND the val items, including games listed in val_games_file (dropped and counted, not
    # refused).  False = every game.
    exclude_friendly: bool = False
    # a file of replay ids (one per line, ``#`` comments) whose games leave the train AND the val items,
    # including games listed in val_games_file (dropped and counted, not refused).  Ids the generation does
    # not hold are ignored (one list may serve several generations).  "" = off (start event unchanged).
    exclude_games_file: str = ""
    cuda_alloc_conf: str = "expandable_segments:True"   # PYTORCH_CUDA_ALLOC_CONF default (setdefault)
    # frozen tokenizer tables: card_vocab, object_vocab, object_catalog, arrival (+ visual_alias for
    # the visual profile).  ``{synthetic: true}`` is the only way to get stand-in tables and is
    # for smoke runs / tests on synthetic shards; real shards pin the table digests.
    tables: Dict[str, Any] = field(default_factory=dict)
    # ---- warm start and per-head learning rates.  Defaults are no-ops (bit-identical runs).
    # init_from: checkpoint whose weights initialise the model ("" = random init).  Loaded strictly
    # (missing / unexpected / shape-mismatched keys refused; model-config fields that change shapes or
    # the function of the weights, WARM_START_MODEL_KEYS, must equal the run's); the EMA starts from the
    # same weights; the optimizer, the LR schedule (warmup + cosine over THIS run's steps), the step
    # counter and the data order start fresh.  The checkpoint's tokenizer-table digests must equal the
    # run's.  --resume of a warm-started run ignores init_from (last.pt holds everything).
    init_from: str = ""
    init_weights: str = "ema"          # "ema" (the shipped weights) | "model" (raw weights)
    # evaluate the selection subset before the first step (logged and checkpointed as a step-0 eval;
    # it also seeds best.pt and early stopping, so a warm start never ships worse than its init)
    eval_at_start: bool = False
    # per-head LR multipliers {parameter-name prefix: multiplier} as optimizer param groups; longest
    # matching prefix wins; prefixes in AminiFormer.HEAD_PREFIXES.  Empty = the two plain groups.
    lr_mult: Dict[str, float] = field(default_factory=dict)
    # ---- speed switches.  Defaults off = the plain computation exactly.
    # pack_tokens: the step encoder embeds / normalises / projects / runs the FFN on the valid tokens
    # only (StepEncoder.packed); attention keeps the padded layout.  Same function (float rounding),
    # same batches, same data order, same step semantics; training, eval and the step-0 eval all use it.
    pack_tokens: bool = False
    # compile_mode: "" (off) | "default" | "max-autotune-no-cudagraphs": torch.compile the token-wise
    # regions (LayerNorm + projections, output projection + FFN + residuals) of the packed step encoder
    # and of the cell decoder's stage S (needs pack_tokens).  compile_dynamic false = static shapes: the
    # packed token count is padded up to a multiple of pack_multiple and the stage-S rows up to a multiple
    # of cell_row_multiple (a small fixed set of shapes, no recompiles after warm-up); true = dynamic
    # shapes (the multiples may be 0).  Padding rows are inert fillers (never read back).
    compile_mode: str = ""
    compile_dynamic: bool = False
    pack_multiple: int = 0
    cell_row_multiple: int = 0
    # ---- training-loop CPU overhead.  Defaults = the plain computation bit for bit; every key is
    # a resume-free runtime switch (not a model field: checkpoints, warm start, inference and deployment are
    # unchanged).
    # defer_stats: no GPU->CPU sync per micro-batch in the step loop: the loss / term values of a logged step
    #   are read with one transfer at log time (same floats, same Python summation order), the non-finite-loss
    #   check and the model-vs-label index check become device flags read at the next flush (every logged
    #   step, before every eval / checkpoint, on a stop signal; a non-finite loss still raises
    #   FloatingPointError naming its step, after at most log_every steps and before anything is saved),
    #   and the row count comes from the host copy of the batch.  Same math.
    defer_stats: bool = False
    # host_index: the loader workers also build the integer indices the forward and the losses otherwise
    #   derive on the device with syncs (valid rows: bool-mask indexing -> index_select on a host-built
    #   flat index; stage-S / stage-P rows: nonzero; the step encoder's attention length (.item()) and the
    #   packed slot / token indices (nonzero)).  Same ops on the same labels / masks, so the same ints;
    #   bool-mask gather and index_select pick the same elements and every row gets exactly one gradient
    #   contribution in both, so values and gradients are the same.  Train batches only (eval unchanged).
    host_index: bool = False
    # ema_foreach: the EMA update as torch._foreach_mul_ / _foreach_add_ over all floating tensors (2 fused
    #   multi-tensor launches instead of 2 per tensor); the same s = s*d + (1-d)*v per element.
    ema_foreach: bool = False
    # adamw_impl: "auto" = fused=True on CUDA, PyTorch's default elsewhere; "fused" = fused kernel on
    #   any device where torch has one (CPU since torch 2.4); "foreach" = the multi-tensor foreach
    #   implementation (the pre-fused baseline, for benchmarks).  Fused and foreach evaluate the same AdamW
    #   update with different kernels (fused: one pass per element in fp32 registers, the bias corrections
    #   and the decoupled weight decay folded in; foreach: separate multi-tensor ops with fp32 rounding of
    #   every intermediate), so they agree to fp32 rounding (a few ulp per step), not bit for bit.
    adamw_impl: str = "auto"
    # CUDA graphs: not used, on purpose.  The forward / backward is shape-dynamic per
    #   micro-batch (rows, ragged groups, the packed token count M, the stage-S / stage-P row counts all vary
    #   and the static-compile ladders only bound them), so it cannot be captured without padding every
    #   micro-batch to a worst-case shape (wasted FLOPs).  The fixed-shape remainder (clip, AdamW, EMA) is a
    #   handful of multi-tensor launches with fused AdamW + ema_foreach, so capturing it would save little
    #   while needing capturable AdamW, a device-tensor learning rate, static .grad buffers
    #   (zero_grad(set_to_none=False)) and a second checkpoint format path.
    # ---- optional LABEL weights, training batches only (never a model input, never in the selection metric,
    # never at deployment).  Defaults off = uniform row weights.  :class:`RowWeighting`.
    # skill_weights_file: optional per-side row weights (JSON, schema il-v7-skill-weights: ``sides``
    #   {replay tag: [w0, w1]}); every loss term of a training row is multiplied by the weight of the row's
    #   acting side (labels ``loss_scale`` x loss_mask in :func:`loss_weights`); a game absent from the file
    #   gets 1.0 (counted in the start event).  skill_weights_sha256: optional pin of the file's sha256
    #   ("" = not checked; with a pin a changed file is refused, also on --resume).
    skill_weights_file: str = ""
    skill_weights_sha256: str = ""
    # afk_tail_ticks (0 = off): rows of a side after that side's last play / ability row of the game get loss
    #   weight 0 (loss_mask False: kept as temporal context, like burn-in rows) when real_end - tick of that
    #   last action >= afk_tail_ticks (a side without any action: every row, when real_end >= afk_tail_ticks).
    #   Removes idle tails at the end of a game.  The game's own recorded rows only.
    afk_tail_ticks: int = 0
    # eval_skill_weights_file / eval_afk_tail_ticks: the same weighting applied to a SECONDARY eval metric
    #   (report["weighted"]: selection NLL and terms with those weights); the selection metric, best.pt and
    #   early stopping stay unweighted.  Lets runs that compare settings report the same weighted metric.
    eval_skill_weights_file: str = ""
    eval_afk_tail_ticks: int = 0
    # ---- stage-S subset and run-time keys.  Defaults = the plain computation bit for bit.
    # stage_s_threat_frac (objective key, like stage_s_base_frac; NOT resume-free): stage S runs on a
    #   deterministic hash subset (tag, side, tick; salt v7cellthr) of the threat-stratum rows too
    #   (tokenizer_v7.stage_s_rows).  Play rows are always kept: base_frac 0 + threat_frac 0 = play rows only;
    #   aux_threat stays normalised over the rows evaluated.  Train and val use the same rule.
    stage_s_threat_frac: float = 1.0
    # eval_every_early (0 = off) / eval_late_frac (runtime, resume-free): with eval_every_early > 0 the steps
    #   before ceil(eval_late_frac x total steps) evaluate every eval_every_early steps, the steps from there on
    #   every eval_every (multiples of the step number in both); the final step is always evaluated and saved.
    #   Every eval is the full selection eval and enters the keep_best ranking.
    eval_every_early: int = 0
    eval_late_frac: float = 0.8        # example value
    # ckpt_every (0 = off; runtime, resume-free): every ckpt_every steps that are not eval steps, a resume-only
    #   checkpoint resume-NNNNNNN.pt (the same payload as an eval checkpoint: raw + EMA weights, optimizer,
    #   scaler, loader position, RNG state; metrics {"resume_only": true}) and last.pt relinked to it.  Never
    #   ranked (keep_best / best.json / best.pt untouched); the resume-only files are pruned to the newest 2.
    ckpt_every: int = 0
    # loader_prefetch (runtime, resume-free): DataLoader prefetch_factor of the train and eval loaders (each item
    #   is one optimizer step of micro-batches).  Unused with num_workers 0.
    loader_prefetch: int = 4
    # stage_p_compiled (runtime, resume-free; needs model.cell_fp32 "logit"): with compile_mode set, stage P's
    #   cross block runs through the compiled token-wise regions as stage S does (CellDecoder.p_regions; same
    #   function to float rounding; rows padded to cell_row_multiple under static compile).  Without compile_mode
    #   it does nothing (so a compile-off retry still runs).  False = the plain Block.forward.
    stage_p_compiled: bool = False

    def __post_init__(self) -> None:
        if self.group_drop_p != 0.0:
            raise ValueError("group_drop_p must be 0 (no group dropout)")
        if self.input_profile not in ("engine", "visual"):
            raise ValueError("input_profile must be engine or visual")
        if self.aux_opp_hand_reduce not in AUX_OPP_HAND_REDUCE:
            raise ValueError(f"aux_opp_hand_reduce must be one of {AUX_OPP_HAND_REDUCE}")
        self.aux_opp_hand_prior_bias = bool(self.aux_opp_hand_prior_bias)
        if self.early_stop_patience < 0 or self.early_stop_min_steps < 0 or self.val_max_games < 0 \
                or self.max_epochs < 0:
            raise ValueError("early_stop_* / val_max_games / max_epochs must be >= 0")
        self.betas = tuple(self.betas)  # type: ignore
        loss = dict(DEFAULT_LOSS)
        loss.update(self.loss or {})
        unknown = set(loss) - set(DEFAULT_LOSS)
        if unknown:
            raise ValueError(f"unknown loss terms {sorted(unknown)}")
        self.loss = loss
        if self.micro_batch_rows < 0 or self.max_batch_tokens < 0:
            raise ValueError("micro_batch_rows / max_batch_tokens must be >= 0")
        self.tables = dict(self.tables or {})
        if self.init_weights not in INIT_WEIGHTS:
            raise ValueError(f"init_weights must be one of {INIT_WEIGHTS}")
        self.init_from = str(self.init_from or "")
        self.eval_at_start = bool(self.eval_at_start)
        self.exclude_friendly = bool(self.exclude_friendly)
        self.exclude_games_file = str(self.exclude_games_file or "")
        lm = {}
        for k, v in dict(self.lr_mult or {}).items():
            v = float(v)
            if not str(k) or not math.isfinite(v) or v < 0:
                raise ValueError(f"lr_mult: bad entry {k!r}: {v} (non-empty prefix, finite multiplier >= 0)")
            lm[str(k)] = v
        self.lr_mult = lm
        self.pack_tokens = bool(self.pack_tokens)
        self.compile_dynamic = bool(self.compile_dynamic)
        self.compile_mode = str(self.compile_mode or "")
        self.pack_multiple = int(self.pack_multiple)
        self.cell_row_multiple = int(self.cell_row_multiple)
        if self.cell_row_multiple < 0:
            raise ValueError("cell_row_multiple must be >= 0")
        if self.compile_mode not in COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
        if self.pack_multiple < 0 or (self.pack_multiple and not self.pack_tokens):
            raise ValueError("pack_multiple must be >= 0 and needs pack_tokens")
        if self.compile_mode and not self.pack_tokens:
            raise ValueError("compile_mode needs pack_tokens (it compiles the packed step-encoder regions)")
        if self.compile_mode and not self.compile_dynamic and not (self.pack_multiple and self.cell_row_multiple):
            raise ValueError("static compile (compile_dynamic false) needs pack_multiple > 0 and "
                             "cell_row_multiple > 0 (fixed shapes)")
        for k in ("defer_stats", "host_index", "ema_foreach"):
            v = getattr(self, k)
            if not isinstance(v, (bool, int)) or v not in (0, 1):
                raise ValueError(f"{k} must be true or false")
            setattr(self, k, bool(v))
        self.adamw_impl = str(self.adamw_impl or "auto")
        if self.adamw_impl not in ADAMW_IMPLS:
            raise ValueError(f"adamw_impl must be one of {ADAMW_IMPLS}")
        self.skill_weights_file = str(self.skill_weights_file or "")
        self.eval_skill_weights_file = str(self.eval_skill_weights_file or "")
        self.skill_weights_sha256 = str(self.skill_weights_sha256 or "").lower()
        if self.skill_weights_sha256 and (len(self.skill_weights_sha256) != 64 or not self.skill_weights_file
                                          or set(self.skill_weights_sha256) - set("0123456789abcdef")):
            raise ValueError("skill_weights_sha256 must be a 64-hex sha256 and needs skill_weights_file")
        for k in ("afk_tail_ticks", "eval_afk_tail_ticks"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{k} must be an integer >= 0 (0 = off)")
        for k in ("eval_every_early", "ckpt_every"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{k} must be an integer >= 0 (0 = off)")
        for k in ("stage_s_threat_frac", "eval_late_frac"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{k} must be a number in [0, 1]")
            setattr(self, k, float(v))
        v = self.loader_prefetch
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError("loader_prefetch must be an integer >= 1")
        if not isinstance(self.stage_p_compiled, (bool, int)) or self.stage_p_compiled not in (0, 1):
            raise ValueError("stage_p_compiled must be true or false")
        self.stage_p_compiled = bool(self.stage_p_compiled)
        if self.stage_p_compiled and (self.model or {}).get("cell_fp32") != "logit":
            raise ValueError("stage_p_compiled needs model.cell_fp32: logit")

    def evaluates_at(self, step: int, total_steps: int) -> bool:
        """Eval schedule: is ``step`` (>= 1) an eval step of a run of ``total_steps`` steps (the in-loop
        rule; the final step is evaluated after the loop when this is False)?  eval_every_early 0 = every
        ``eval_every`` steps."""
        k = self.eval_every
        if self.eval_every_early and step < math.ceil(self.eval_late_frac * total_steps - 1e-9):
            k = self.eval_every_early
        return bool(k) and step % k == 0

    def eval_steps(self, total_steps: int) -> List[int]:
        """Every step a full run of ``total_steps`` steps evaluates (in-loop evals, then the final step once;
        no step-0 eval, no early stop, a non-empty val split)."""
        out = [s for s in range(1, int(total_steps) + 1) if self.evaluates_at(s, total_steps)]
        if not out or out[-1] != total_steps:
            out.append(int(total_steps))
        return out

    @property
    def micro_rows(self) -> int:
        return min(self.micro_batch_rows or self.batch_rows, self.batch_rows)

    @classmethod
    def from_file(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> "TrainConfig":
        d = load_yaml(Path(path).read_text())
        d.update(overrides or {})
        # relative table paths: as given (cwd) or else relative to the repo root of the config
        repo = Path(path).resolve().parent.parent
        tabs = dict(d.get("tables") or {})
        for k, v in tabs.items():
            if isinstance(v, str) and not Path(v).is_absolute() and not Path(v).exists() \
                    and (repo / v).exists():
                tabs[k] = str(repo / v)
        if tabs:
            d["tables"] = tabs
        for k in ("skill_weights_file", "eval_skill_weights_file"):      # same rule as the tables
            v = d.get(k)
            if isinstance(v, str) and v and not Path(v).is_absolute() and not Path(v).exists() \
                    and (repo / v).exists():
                d[k] = str(repo / v)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown train config keys: {sorted(unknown)}")
        return cls(**d)


# ------------------------------------------------------------------------- storage
Episode = Tuple[int, int, int, int, int]


@dataclass
class ResolvedData:
    """What ``--data`` names, resolved WITHOUT opening any table (so the trainer can
    take the training lock before it memmaps the generation)."""

    paths: List[str]
    generation: str
    generation_dir: Optional[str]          # generations/<id> when a generation was named
    shard_dirs: List[str]
    superseded: set


def resolve_data(paths: Sequence[str]) -> ResolvedData:
    """Resolve a generation directory (``MANIFEST.json`` listing shards) or shard dirs.
    Reads only the generation MANIFEST; never memmaps a table."""

    dirs: List[str] = []
    superseded: set = set()
    gen_dir: Optional[str] = None
    gen_id = "shards:" + hashlib.sha256("|".join(sorted(map(str, paths))).encode()).hexdigest()[:12]
    for p in paths:
        pp = Path(p)
        man = pp / "MANIFEST.json"
        if (pp / "rows.npy").exists():
            dirs.append(str(pp))
        elif man.exists():
            m = json.loads(man.read_text())
            shard_root = pp.parent.parent / "shards"
            for sid in m.get("shards", []):
                sid = sid["id"] if isinstance(sid, dict) else sid
                dirs.append(str(pp / sid) if (pp / sid / "rows.npy").exists() else str(shard_root / sid))
            superseded |= {s.encode() for s in m.get("superseded", []) if isinstance(s, str)}
            gen_id = str(m.get("generation", pp.name))
            gen_dir = str(pp)
        elif (pp / "shards").is_dir():
            dirs += [str(x) for x in sorted((pp / "shards").iterdir()) if (x / "rows.npy").exists()]
        else:
            raise FileNotFoundError(f"no shard or generation at {p}")
    if gen_dir is not None and len(paths) != 1:
        raise ValueError("a generation directory must be the only --data path")
    return ResolvedData(list(map(str, paths)), gen_id, gen_dir, dirs, superseded)


def _read_manifest(path: Path) -> Optional[Dict[str, Any]]:
    f = path / "MANIFEST.json"
    return json.loads(f.read_text()) if f.exists() else None


class ShardSource:
    """Read-only view of one or more v7 shards (memmapped .npy tables).

    The reader for shard directories and generation MANIFESTs: memmap every
    table, ``ent_off``/``obj_off`` as int64 cumsums at load, and an episode index of
    ``(shard, game, side, row_lo, row_hi)`` skipping superseded games.  The optional storage module's
    ``amini.v7.storage.open_generation`` (not part of this repository) is used through
    :class:`GenerationAdapter` when it is installed (see :meth:`open`).
    """

    TABLES = ("games", "events", "rows", "entity", "object", "life", "towerhp", "burrow", "priv")

    def __init__(self, shard_dirs: Sequence[str], superseded: Optional[set] = None) -> None:
        _raise_nofile(len(shard_dirs))
        self.dirs = [Path(d) for d in shard_dirs]
        shards = []
        for d in self.dirs:
            shards.append({name: np.load(d / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                           for name in self.TABLES})
        self._setup(shards, None, None, None, superseded or set())
        self.manifests: List[Dict[str, Any]] = [m for m in (_read_manifest(d) for d in self.dirs) if m]

    # shared by the adapter ------------------------------------------------------
    def _setup(self, shards: List[Dict[str, np.ndarray]], ent_off: Optional[List[np.ndarray]],
               obj_off: Optional[List[np.ndarray]], episodes: Optional[Sequence[Sequence[int]]],
               superseded: set) -> None:
        from amini.nn import schema_v1 as S
        self.shards = shards
        for si, t in enumerate(shards):
            for name in self.TABLES:
                if t[name].dtype != S.SHARD_TABLES[name]:
                    raise ValueError(f"shard {si}: {name} dtype {t[name].dtype} != schema {S.SHARD_TABLES[name]}")
            if len(t["priv"]) != len(t["rows"]):
                raise ValueError(f"shard {si}: priv is not row-aligned")
        self.ent_off = [np.asarray(x, dtype=np.int64) for x in ent_off] if ent_off is not None else \
            [np.concatenate([[0], np.cumsum(t["rows"]["n_ent"], dtype=np.int64)]) for t in shards]
        self.obj_off = [np.asarray(x, dtype=np.int64) for x in obj_off] if obj_off is not None else \
            [np.concatenate([[0], np.cumsum(t["rows"]["n_obj"], dtype=np.int64)]) for t in shards]
        for si, t in enumerate(shards):
            if len(self.ent_off[si]) != len(t["rows"]) + 1 or len(self.obj_off[si]) != len(t["rows"]) + 1:
                raise ValueError(f"shard {si}: ent_off/obj_off length != rows + 1")
            if int(self.ent_off[si][-1]) != len(t["entity"]) or int(self.obj_off[si][-1]) != len(t["object"]):
                raise ValueError(f"shard {si}: ent_off/obj_off do not cover entity/object")
        self._n_ent: Dict[int, np.ndarray] = {}
        self._n_obj: Dict[int, np.ndarray] = {}
        if episodes is not None:
            self.episodes: List[Episode] = [tuple(int(x) for x in e) for e in episodes]  # type: ignore
            for e in self.episodes:
                if len(e) != 5:
                    raise ValueError("episode index entries must be (shard, game, side, row_lo, row_hi)")
            return
        self.episodes = []
        for si, t in enumerate(shards):
            g = t["games"]
            for gi in range(len(g)):
                if bytes(g["replay_id"][gi]) in superseded:
                    continue
                for side in (0, 1):
                    lo, hi = int(g["row_lo"][gi][side]), int(g["row_hi"][gi][side])
                    if hi > lo:
                        self.episodes.append((si, gi, side, lo, hi))

    @classmethod
    def open(cls, paths: Sequence[str], resolved: Optional[ResolvedData] = None
             ) -> Tuple["ShardSource", str]:
        """Open what ``resolve_data(paths)`` names.  Returns the source and the generation id.

        A generation directory goes through ``amini.v7.storage.open_generation``
        when that module is installed, wrapped by :class:`GenerationAdapter` (which refuses an
        object without the loader interface instead of training on it).
        """

        rd = resolved if resolved is not None else resolve_data(paths)
        try:
            from amini.v7 import storage  # type: ignore
        except ImportError:
            storage = None
        if storage is not None and rd.generation_dir is not None:
            if not hasattr(storage, "open_generation"):
                raise RuntimeError("amini.v7.storage exists but has no open_generation(path)")
            gen = storage.open_generation(rd.generation_dir)
            return GenerationAdapter.wrap(gen, rd), rd.generation
        src = cls(rd.shard_dirs, rd.superseded)
        if rd.generation_dir is not None:
            gm = _read_manifest(Path(rd.generation_dir))
            if gm:
                src.manifests.insert(0, gm)
        return src, rd.generation

    def split_of(self, ep: Episode) -> int:
        si, gi = ep[0], ep[1]
        return int(self.shards[si]["games"]["split"][gi])

    def token_max(self, ep: Episode, lo: int, hi: int) -> Tuple[int, int]:
        """Largest ENTITY / OBJECT count over rows ``[lo, hi)`` (micro-batch token budget)."""

        si = ep[0]
        if si not in self._n_ent:
            r = self.shards[si]["rows"]
            self._n_ent[si] = np.asarray(r["n_ent"], dtype=np.int64)
            self._n_obj[si] = np.asarray(r["n_obj"], dtype=np.int64)
        if hi <= lo:
            return 0, 0
        return int(self._n_ent[si][lo:hi].max()), int(self._n_obj[si][lo:hi].max())

    def table_digests(self) -> Dict[str, str]:
        """Vocab / table digests pinned by the generation and shard MANIFESTs,
        keyed by every alias of each table name (logical name, file name, stem).
        Conflicting pins raise."""

        out: Dict[str, str] = {}
        for m in self.manifests:
            for sec in ("vocab_digests", "table_digests", "vocab", "tables"):
                d = m.get(sec)
                if not isinstance(d, dict):
                    continue
                for k, v in d.items():
                    sha = v.get("sha256") if isinstance(v, dict) else v
                    if not isinstance(sha, str) or len(sha) != 64:
                        continue
                    for alias in _table_aliases(str(k)):
                        if out.get(alias, sha) != sha:
                            raise ValueError(f"MANIFESTs pin two digests for {k}: {out[alias][:12]} vs {sha[:12]}")
                        out[alias] = sha
        return out

    @property
    def synthetic(self) -> bool:
        """True only if every shard MANIFEST says ``synthetic: true`` (test shards)."""

        shard_m = [m for m in self.manifests if "shard_id" in m or "files" in m]
        return bool(shard_m) and len(shard_m) >= len(self.shards) and all(m.get("synthetic") for m in shard_m)

    def episode(self, ep: Episode, lo: Optional[int] = None, hi: Optional[int] = None):
        EpisodeArrays = _optional_module("amini.nn.tokenizer_v7", "the shard tokenizer").EpisodeArrays
        si, gi, side, row_lo, row_hi = ep
        lo = row_lo if lo is None else lo
        hi = row_hi if hi is None else hi
        t = self.shards[si]
        g = t["games"][gi]
        eo, oo = self.ent_off[si], self.obj_off[si]
        return EpisodeArrays(
            tag=bytes(g["replay_id"]), game=np.array(g, dtype=g.dtype)[()],
            rows=np.asarray(t["rows"][lo:hi]), ent=np.asarray(t["entity"][eo[lo]:eo[hi]]),
            obj=np.asarray(t["object"][oo[lo]:oo[hi]]),
            evt=np.asarray(t["events"][int(g["evt_lo"]):int(g["evt_hi"])]),
            life=np.asarray(t["life"][int(g["life_lo"]):int(g["life_hi"])]),
            towerhp=np.asarray(t["towerhp"][int(g["thp_lo"]):int(g["thp_hi"])]),
            burrow=np.asarray(t["burrow"][int(g["bur_lo"]):int(g["bur_hi"])]),
            priv=np.asarray(t["priv"][lo:hi]))


def _raise_nofile(n_shards: int) -> None:
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = hard if hard != resource.RLIM_INFINITY else 65536
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, 65536), hard))
        soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if 8 * n_shards + 256 > soft:
            raise RuntimeError(f"too many shards ({n_shards}) for RLIMIT_NOFILE {soft}")
    except (ImportError, ValueError):
        pass


def _table_aliases(name: str) -> List[str]:
    base = Path(name).name
    stem = base[:-5] if base.endswith(".json") else base
    out = [base, stem]
    if stem.endswith("_v7"):
        out.append(stem[:-3])
    return list(dict.fromkeys(out))


class GenerationAdapter(ShardSource):
    """Wraps the ``Generation`` returned by the storage module's ``storage.open_generation``.

    Required interface (the loader contract, nothing more): ``shards`` (per shard a
    mapping or object with the nine memmapped tables), ``episodes`` (the
    ``(shard, game, side, row_lo, row_hi)`` index, superseded games already skipped);
    optional ``ent_off`` / ``obj_off`` (recomputed and checked otherwise) and
    ``manifests`` / ``shard_dirs`` (otherwise the MANIFEST files on disk are read).
    Anything else is refused with the missing names, never guessed.
    """

    REQUIRED = ("shards", "episodes")

    @classmethod
    def wrap(cls, gen: Any, rd: ResolvedData) -> "GenerationAdapter":
        missing = [a for a in cls.REQUIRED if not hasattr(gen, a)]
        if missing:
            raise RuntimeError(
                f"amini.v7.storage.open_generation() returned {type(gen).__name__} without "
                f"{missing}; il_v7 needs the loader interface (per-shard memmapped tables and the "
                f"(shard, game, side, row_lo, row_hi) episode index). Update GenerationAdapter.")
        self = cls.__new__(cls)
        self.gen = gen
        shards = []
        for si, t in enumerate(gen.shards):
            tab = {}
            for name in cls.TABLES:
                try:
                    tab[name] = t[name] if hasattr(t, "__getitem__") and not isinstance(t, np.ndarray) \
                        else getattr(t, name)
                except (KeyError, AttributeError, TypeError, IndexError):
                    raise RuntimeError(f"open_generation(): shard {si} has no table {name!r}")
            shards.append(tab)
        _raise_nofile(len(shards))
        dirs = [Path(d) for d in getattr(gen, "shard_dirs", None) or rd.shard_dirs]
        self.dirs = dirs
        self._setup(shards, getattr(gen, "ent_off", None), getattr(gen, "obj_off", None),
                    list(gen.episodes), set())
        # GAME.form_mismatch is the field the tokenizer reads; the storage module's MANIFEST copy
        # (Generation.form_mismatch) must equal it.  storage fills zeros when a shard MANIFEST
        # lacks the key; only then (key provably absent, or not knowable) is an all-zero copy
        # tolerated.  A length mismatch is corruption.
        _check_form_mismatch(gen, shards)
        mans = getattr(gen, "manifests", None)
        if mans is None:
            mans = [m for m in (_read_manifest(d) for d in dirs) if m]
            if rd.generation_dir is not None:
                gm = _read_manifest(Path(rd.generation_dir))
                if gm:
                    mans = [gm] + mans
        self.manifests = list(mans)
        return self


def _check_form_mismatch(gen: Any, shards: Sequence[Dict[str, np.ndarray]]) -> None:
    fm = getattr(gen, "form_mismatch", None)
    fm = [] if fm is None else list(fm)
    if fm and len(fm) != len(shards):
        raise RuntimeError(f"open_generation(): form_mismatch covers {len(fm)} shards, generation has {len(shards)}")
    mans = getattr(gen, "manifests", None)
    mans = [] if mans is None else list(mans)
    off = len(mans) - len(shards)                        # storage puts the generation MANIFEST first
    for si, (tab, want) in enumerate(zip(shards, fm)):
        want = np.asarray(want, dtype=np.int64).reshape(-1)
        have = np.asarray(tab["games"]["form_mismatch"], dtype=np.int64)
        if len(want) != len(have):
            raise RuntimeError(f"shard {si}: MANIFEST form_mismatch has {len(want)} games, "
                               f"games has {len(have)}")
        if np.array_equal(want, have):
            continue
        man = mans[si + off] if off in (0, 1) and isinstance(mans[si + off], dict) else None
        absent = man is None or "form_mismatch" not in man
        if absent and not np.any(want != 0):
            continue
        raise RuntimeError(f"shard {si}: MANIFEST form_mismatch disagrees with games.form_mismatch")


def chunk_episodes(src: ShardSource, split: int, max_rows: int, burn_in: int
                   ) -> List[Tuple[Episode, int, int, int]]:
    """Split episodes longer than ``max_rows`` into chunks with ``burn_in`` context rows.

    Returns ``(episode, lo, loss_lo, hi)``: rows ``[lo, loss_lo)`` are temporal context
    only (no loss).  Whole episodes (the common case) train exactly as deployment runs them.
    """

    out = []
    for ep in src.episodes:
        if src.split_of(ep) != split:
            continue
        lo, hi = ep[3], ep[4]
        if hi - lo <= max_rows:
            out.append((ep, lo, lo, hi))
            continue
        if max_rows <= burn_in:
            raise ValueError("max_seq_rows must exceed burn_in_rows")
        start = lo
        while start < hi:
            a = max(lo, start - burn_in)
            b = min(hi, a + max_rows)
            out.append((ep, a, start, b))
            start = b
    return out


def _round8(n: int) -> int:
    return max(8, -(-int(n) // 8) * 8)


def fixed_tokens() -> int:
    """Tokens per row that do not depend on the data: the fixed-cap groups plus [STEP]."""

    from amini.nn import schema_v1 as S
    return 1 + sum(g.cap for g in S.GROUPS.values() if not g.ragged)


def micro_cost(rows: int, max_ent: int, max_obj: int) -> int:
    """Padded step-encoder tokens of a micro-batch: valid rows x (fixed + round8(ENT) +
    round8(OBJ)), exactly the padding ``tokenizer_v7.collate`` produces."""

    return int(rows) * (fixed_tokens() + _round8(max_ent) + _round8(max_obj))


class EpisodeBatches:
    """Map-style dataset: item i = one optimizer step = a list of collated micro-batches.

    Episodes (chunks) are packed into steps of at most ``batch_rows`` rows (the effective
    batch).  Each step is split greedily into micro-batches of at most
    ``micro_batch_rows`` rows whose padded token count (:func:`micro_cost`) stays at or
    under ``max_batch_tokens``, so one unusually crowded episode cannot push the batch
    maximum padding (and peak memory) past the GPU.  The loss is normalised over the
    whole step (``train``), so the split does not change the objective.
    """

    def __init__(self, src: ShardSource, items, tables, cfg: TrainConfig, *, train: bool,
                 epoch: int = 0, parity: bool = False, weighting: Optional["RowWeighting"] = None) -> None:
        self.src, self.items, self.tables, self.cfg = src, list(items), tables, cfg
        self.train, self.parity = train, parity
        # optional label weights (train: loss_mask / loss_scale; eval: the secondary eval_scale); None = off
        self.weighting = weighting
        self.tok = [src.token_max(it[0], it[1], it[3]) for it in self.items]
        self.set_epoch(epoch)

    def _fits(self, micro: List[int]) -> bool:
        if len(micro) <= 1:
            return True                      # a single chunk always runs (max_seq_rows bounds it)
        rows = sum(self.items[i][3] - self.items[i][1] for i in micro)
        if rows > self.cfg.micro_rows:
            return False
        cap = self.cfg.max_batch_tokens
        return not cap or micro_cost(rows, max(self.tok[i][0] for i in micro),
                                     max(self.tok[i][1] for i in micro)) <= cap

    def _split(self, step: List[int]) -> List[List[int]]:
        """Fewest micro-batches that satisfy both caps, balanced by rows (LPT: longest
        chunk first into the lightest micro-batch); chunk order inside a micro-batch
        follows the step order.  Deterministic."""

        n_rows = {i: self.items[i][3] - self.items[i][1] for i in step}
        pos = {i: k for k, i in enumerate(step)}
        by_size = sorted(step, key=lambda i: (-n_rows[i], pos[i]))
        m = max(1, -(-sum(n_rows.values()) // self.cfg.micro_rows))
        while True:
            bins: List[List[int]] = [[] for _ in range(m)]
            load = [0] * m
            for i in by_size:
                j = min(range(m), key=lambda k: (load[k], k))
                bins[j].append(i)
                load[j] += n_rows[i]
            bins = [sorted(b, key=pos.__getitem__) for b in bins if b]
            if m >= len(step) or all(self._fits(b) for b in bins):
                return sorted(bins, key=lambda b: pos[b[0]])
            m += 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        order = np.arange(len(self.items))
        if self.train:
            np.random.default_rng((self.cfg.seed, epoch)).shuffle(order)
        steps, cur, rows = [], [], 0
        for i in order:
            n = self.items[i][3] - self.items[i][1]
            if cur and rows + n > self.cfg.batch_rows:
                steps.append(cur)
                cur, rows = [], 0
            cur.append(int(i))
            rows += n
        if cur:
            steps.append(cur)
        self.batches = [self._split(s) for s in steps]

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, b: int) -> List[Dict[str, Any]]:
        TK = _optional_module("amini.nn.tokenizer_v7", "the shard tokenizer")
        rng = np.random.default_rng((self.cfg.seed, self.epoch, b, 1 if self.train else 2))
        out = []
        for micro in self.batches[b]:
            examples = []
            for i in micro:
                ep, lo, loss_lo, hi = self.items[i]
                arr = self.src.episode(ep, lo, hi)
                if self.train and not self.parity:
                    draws = TK.draw_train(rng, len(arr.rows), arr.evt, self.tables, self.cfg.pop_null_p)
                else:
                    draws = TK.draw_parity(arr.tag, int(arr.rows["side"][0]), arr.rows, arr.evt, self.tables)
                ex = TK.make_example(arr, draws, self.tables, profile=self.cfg.input_profile,
                                     stage_s_base_frac=self.cfg.stage_s_base_frac,
                                     fid_exact=self.cfg.fid_exact, fid_inexact=self.cfg.fid_inexact,
                                     stage_s_threat_frac=self.cfg.stage_s_threat_frac)
                lm = np.ones(hi - lo, bool)
                lm[:loss_lo - lo] = False
                ex["labels"]["loss_mask"] = lm
                if self.weighting is not None:
                    self.weighting.apply(ex["labels"], ep, lo, hi, eval_mode=not self.train)
                if (self.cfg.model or {}).get("cell_mask") and "cell_legal" in ex["labels"]:
                    # a label cell outside the slot's placement mask has no cell term (the slot term stays)
                    lab = ex["labels"]
                    lab["cell"] = np.where(lab["cell_legal"], lab["cell"], -1)
                examples.append(ex)
            c = TK.collate(examples)
            if self.train and self.cfg.host_index:        # train batches only (eval is unchanged)
                c["host_index"] = host_index(c)
            out.append(c)
        return out

# ------------------------------------------------------------------------- losses
def _row_sel(batch: Dict[str, Any]):
    """The valid-row selector of ``batch``: its [B, T] bool ``row_valid``, or with the
    ``host_index`` key the loader-built int64 flat index of the same rows (no GPU->CPU sync)."""
    hx = batch.get("host_index")
    return batch["row"]["row_valid"] if hx is None else hx["rv"]


def _rows(x, sel):
    from amini.nn.aminiformer import take_rows
    return take_rows(x, sel)


def _flat(batch: Dict[str, Any], section: str, key: str):
    return _rows(batch[section][key], _row_sel(batch))


def _up(x):
    """At least fp32 (fp64 stays fp64, so the double-precision tests are exact)."""
    import torch
    return x if x.dtype == torch.float64 else x.float()


def loss_indices(batch: Dict[str, Any]):
    """Stage-S and stage-P row indices (over valid rows) exactly as the model derives
    them from :func:`cell_and_slot`; labels only, so they are known before the forward."""

    import torch
    hx = batch.get("host_index")
    if hx is not None:                     # host_index: the same indices, built on the host
        return hx["ridx"], hx["pidx"]
    rv = batch["row"]["row_valid"].bool()
    cell_rows, play_slot = cell_and_slot(batch)
    ridx = cell_rows[rv].bool().nonzero(as_tuple=True)[0]
    ps = play_slot[rv][ridx].long()
    pidx = ridx[(ps >= 0).nonzero(as_tuple=True)[0]]
    return ridx, pidx


def loss_weights(batch: Dict[str, Any], ridx=None, pidx=None) -> Dict[str, Any]:
    """Per-element weights of every loss term.  They depend on labels and
    targets only (never on model outputs), so a step split into micro-batches can be
    normalised by the step-wide sums before any backward pass.

    Every row weighs the same, except: the gate and value terms carry the row's sampling weight
    ``row_weight`` (1 for action rows, the inverse sampling rate of its stratum for committed-wait rows),
    ``loss_mask`` zeroes context-only rows, and the optional per-row multiplier ``loss_scale``
    (:class:`RowWeighting`; absent = 1) scales every term of its row."""

    import torch
    lab = batch["labels"]
    tg = batch.get("targets") or {}
    rv = _row_sel(batch)
    pok = _up(_rows(lab["policy_ok"], rv))
    lm = _up(_rows(lab["loss_mask"], rv)) if "loss_mask" in lab else torch.ones_like(pok)
    if "loss_scale" in lab:                # optional row multiplier (RowWeighting): every term below carries lm
        lm = lm * _up(_rows(lab["loss_scale"], rv))
    ok = pok * lm
    if ridx is None or pidx is None:
        ridx, pidx = loss_indices(batch)
    w: Dict[str, Any] = {}
    w["gate"] = _up(_rows(lab["row_weight"], rv)) * ok
    m_slot = _up((_rows(lab["slot"], rv) >= 0)) * ok
    w["slot"] = m_slot
    w["cell"] = (m_slot * _up(_rows(lab["cell"], rv) >= 0))[pidx]
    w["ability"] = _up((_rows(lab["ability"], rv) >= 0)) * ok
    if "value" in tg:
        w["value"] = _up(_rows(lab["row_weight"], rv)) * lm
        w["value_priv"] = w["value"]
    if "aux_opp_next" in tg:
        w["aux_opp_next"] = _up((_rows(tg["aux_opp_next"], rv) >= 0)) * lm
    if "aux_opp_hand" in tg:
        w["aux_opp_hand"] = _up(_rows(tg["aux_opp_hand_mask"], rv)) * lm
    if "aux_tower_dmg" in tg:
        w["aux_tower_dmg"] = lm
    for key, mkey in (("aux_land", "aux_land_mask"), ("aux_emerge", "aux_emerge_mask")):
        if key in tg:
            w[key] = _up(_rows(tg[mkey], rv)) * lm.unsqueeze(-1)
    if "aux_threat" in tg:
        w["aux_threat"] = lm[ridx]
    return w


def loss_denominators(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.sum() for k, v in loss_weights(batch).items()}


def compute_losses(out: Dict[str, Any], batch: Dict[str, Any], loss_w: Dict[str, float],
                   dens: Optional[Dict[str, Any]] = None,
                   opp_hand_reduce: str = DEFAULT_OPP_HAND_REDUCE,
                   deferred_checks: Optional[List[Any]] = None) -> Dict[str, Any]:
    """The training loss.  Returns the total, every term, and the selection sums.

    ``dens``: step-wide denominators (:func:`loss_denominators` summed over the step's
    micro-batches).  Each term is then this micro-batch's share ``num / D_step`` and the
    micro-batch totals add up to exactly the full-step loss.  Without ``dens`` every term
    is normalised by its own micro-batch sum (eval, single micro-batch).

    ``deferred_checks`` (``defer_stats``): a list that receives the model-vs-label index check as
    device bool tensors (True = mismatch) instead of syncing with ``torch.equal``; the trainer reads
    them at its next flush.  A shape mismatch still raises at once (shapes are host-side).
    """

    import torch
    import torch.nn.functional as F

    lab = batch["labels"]
    tg = batch.get("targets") or {}
    rv = _row_sel(batch)
    ridx, pidx = loss_indices(batch)
    if dens is not None:          # step-normalised training: the model must use the same rows
        for key, ref in (("cell_row_index", ridx), ("cell_play_index", pidx)):
            if key not in out or out[key] is ref:      # host_index: the model returns the given tensors
                continue
            if deferred_checks is not None and out[key].shape == ref.shape:
                deferred_checks.append((out[key] != ref).any())
            elif deferred_checks is not None or not torch.equal(out[key], ref):
                raise RuntimeError(f"model {key} differs from loss_indices(batch)")
    ridx = out.get("cell_row_index", ridx)
    pidx = out.get("cell_play_index", pidx)
    W = loss_weights(batch, ridx, pidx)
    eps = 1e-8
    nums: Dict[str, Any] = {}
    zero = _up(out["family"]).sum() * 0.0

    fam = _rows(lab["family"], rv).long()
    fam_safe = torch.where(W["gate"] > 0, fam, torch.zeros_like(fam))
    nums["gate"] = (W["gate"] * F.cross_entropy(_up(out["family"]), fam_safe, reduction="none")).sum()
    slot = _rows(lab["slot"], rv).long()
    nums["slot"] = (W["slot"] * F.cross_entropy(_up(out["slot"]), slot.clamp_min(0), reduction="none")).sum()
    if "cells" in out and pidx.numel():
        cell = _rows(lab["cell"], rv).long()[pidx]
        nums["cell"] = (W["cell"] * F.cross_entropy(_up(out["cells"]), cell.clamp_min(0),
                                                    reduction="none")).sum()
    else:
        nums["cell"] = zero
    ab = _rows(lab["ability"], rv).long()
    nums["ability"] = (W["ability"] * F.cross_entropy(_up(out["ability"]), ab.clamp_min(0),
                                                      reduction="none")).sum()

    # value / privileged critic (MSE weighted by the row's sampling weight)
    if "value" in W:
        y = _up(_rows(tg["value"], rv))
        nums["value"] = (W["value"] * (_up(out["value"]) - y) ** 2).sum()
        if "value_priv" in out:
            nums["value_priv"] = (W["value_priv"] * (_up(out["value_priv"]) - y) ** 2).sum()

    # auxiliary heads: each normalised over its own valid targets
    if "aux_opp_next" in out and "aux_opp_next" in W:
        y = _rows(tg["aux_opp_next"], rv).long()
        nums["aux_opp_next"] = (W["aux_opp_next"] * F.cross_entropy(
            _up(out["aux_opp_next"]), y.clamp_min(0), reduction="none")).sum()
    if "aux_opp_hand" in out and "aux_opp_hand" in W:
        # multi-label BCE over the card vocab, averaged over classes so the per-row
        # loss is on the scale of the other aux heads (the head's bias starts at the
        # 4-of-V prior); normalised over valid rows like every "per valid row" aux term.
        idx = _rows(tg["aux_opp_hand"], rv).long()
        logits = _up(out["aux_opp_hand"])
        v = logits.shape[-1]
        target = torch.zeros(idx.shape[0], v + 1, device=idx.device, dtype=logits.dtype)
        target.scatter_(1, torch.where(idx >= 0, idx, torch.full_like(idx, v)), 1.0)
        target = target[:, :v]
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if opp_hand_reduce == "class_mean":
            per_row = bce.mean(-1)
        elif opp_hand_reduce == "sum_hand":          # sum over V / the 4 hand slots
            from amini.nn import schema_v1 as S
            per_row = bce.sum(-1) / S.HAND_SLOTS
        else:
            raise ValueError(f"unknown opp_hand_reduce {opp_hand_reduce!r}")
        nums["aux_opp_hand"] = (W["aux_opp_hand"] * per_row).sum()
    if "aux_tower_dmg" in out and "aux_tower_dmg" in W:
        y = _up(_rows(tg["aux_tower_dmg"], rv))
        bce = F.binary_cross_entropy_with_logits(_up(out["aux_tower_dmg"]), y,
                                                 reduction="none").mean((-1, -2))
        nums["aux_tower_dmg"] = (W["aux_tower_dmg"] * bce).sum()
    for key in ("aux_land", "aux_emerge"):
        if key in out and key in W:
            g = out[key].shape[1]
            y = _up(_rows(tg[key], rv))[:, :g]
            if W[key].shape[1] != g:
                raise RuntimeError(f"{key}: target width {W[key].shape[1]} != output width {g}")
            err = F.smooth_l1_loss(_up(out[key]), y, reduction="none").mean(-1)
            nums[key] = (W[key] * err).sum()
    if "aux_threat" in out and "aux_threat" in W:
        ridx = out["cell_row_index"]
        y = _up(_rows(tg["aux_threat"], rv)[ridx])
        bce = F.binary_cross_entropy_with_logits(_up(out["aux_threat"]), y,
                                                 reduction="none").mean((-1, -2))
        nums["aux_threat"] = (W["aux_threat"] * bce).sum()

    terms: Dict[str, Any] = {}
    sums: Dict[str, Any] = {}
    for k, num in nums.items():
        den_local = W[k].sum()
        sums[k] = (num, den_local)
        den = dens[k] if dens is not None else den_local
        terms[k] = num / den.clamp_min(eps)
    for k in ("gate", "slot", "cell", "ability"):
        sums.setdefault(k, (zero, zero))
    total = sum(loss_w[k] * v for k, v in terms.items() if loss_w.get(k, 0.0) != 0.0)
    return {"total": total, "terms": terms, "sums": sums}


def selection_value(sums: Dict[str, Tuple[float, float]]) -> float:
    """Weighted policy NLL: gate (sampling-weighted over all rows) + slot + cell + ability
    (over action rows).  Aux and value terms are excluded by construction."""

    total = 0.0
    for k in ("gate", "slot", "cell", "ability"):
        num, den = sums.get(k, (0.0, 0.0))
        total += float(num) / max(float(den), 1e-8)
    return total


# ------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of every floating parameter and buffer."""

    def __init__(self, model, decay: float, foreach: bool = False) -> None:
        import torch
        self.decay = decay
        self.foreach = bool(foreach)              # ema_foreach (runtime only, not saved)
        self.shadow = {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
                       for k, v in model.state_dict().items()}
        self.steps = 0

    def update(self, model) -> None:
        import torch
        self.steps += 1
        d = self.decay
        with torch.no_grad():
            if self.foreach:
                # the same s = s*d + (1-d)*v per floating tensor as the loop below, as two multi-tensor ops
                fs, fv = [], []
                for k, v in model.state_dict().items():
                    s = self.shadow[k]
                    if v.is_floating_point():
                        fs.append(s)
                        fv.append(v.detach().float())
                    else:
                        s.copy_(v)
                if fs:
                    torch._foreach_mul_(fs, d)
                    torch._foreach_add_(fs, fv, alpha=1.0 - d)
                return
            for k, v in model.state_dict().items():
                s = self.shadow[k]
                if v.is_floating_point():
                    s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    s.copy_(v)

    def copy_to(self, model) -> None:
        cur = model.state_dict()
        sd = {k: v.to(cur[k].dtype) for k, v in self.shadow.items()}
        model.load_state_dict(sd)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "steps": self.steps, "shadow": self.shadow}


# ------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, model, ema: EMA, cfg: TrainConfig, step: int,
                    metrics: Dict[str, Any], optimizer=None, scaler=None,
                    generation: str = "", tables_record: Optional[Dict[str, Any]] = None,
                    resume: Optional[Dict[str, Any]] = None, init: Optional[Dict[str, Any]] = None) -> None:
    """``resume``: trainer state (step, data position, RNG, early stopping, history) that
    :func:`train` needs to continue the run exactly (``--resume``).  ``init``: the warm-start record
    (:func:`warm_start`; None for a run from random weights)."""

    import torch
    from amini.nn import schema_v1 as S
    payload = {
        "format": "aminiformer-v1-ckpt.1", "schema": S.SCHEMA_ID, "trainer": TRAINER_VERSION,
        "model_config": model.cfg.to_dict(), "train_config": asdict(cfg), "step": step,
        "metrics": metrics, "generation": generation,
        "model_ema": {k: v.clone() for k, v in ema.shadow.items()},
        "model_raw": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "schema_describe": S.describe(),
        "tables": tables_record,
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "resume": resume,
        "init": init,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_policy(path: str, device: Any = "cpu", packed: bool = False):
    """Load the SHIPPED (EMA) weights of a checkpoint into a fresh model (eval mode).
    ``packed``: run the step encoder on valid tokens only (speed switch; same outputs)."""

    import torch
    from amini.nn.aminiformer import AminiFormer, AminiFormerConfig
    ck = torch.load(path, map_location=device, weights_only=False)
    model = AminiFormer(AminiFormerConfig.from_dict(ck["model_config"]))
    model.load_state_dict({k: v for k, v in ck["model_ema"].items()})
    model.step_encoder.packed = bool(packed)
    return model.to(device).eval()


def apply_speed_settings(model, cfg: "TrainConfig") -> None:
    """Runtime speed switches on a model (defaults: no change).  Packing / compilation change only how
    the step encoder computes, never what: same outputs to float rounding, same parameters."""

    se, cd = model.step_encoder, model.cells
    se.packed = bool(cfg.pack_tokens)
    se.pad_multiple = int(cfg.pack_multiple) if cfg.pack_tokens else 0
    cd.pad_rows = int(cfg.cell_row_multiple)
    se.compiled = cd.compiled = None
    if cfg.compile_mode:
        from amini.nn.aminiformer import compile_regions
        se.compiled = cd.compiled = compile_regions(cfg.compile_mode, True if cfg.compile_dynamic else False)
    if hasattr(cd, "p_regions"):                       # stage_p_compiled (acts only with compile_mode)
        cd.p_regions = bool(cfg.stage_p_compiled)


def compile_counters() -> Dict[str, int]:
    """Dynamo graph / frame counters (recompiles after warm-up must stay 0)."""

    try:
        from torch._dynamo.utils import counters
        return {"unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                "calls_captured": int(counters["stats"].get("calls_captured", 0))}
    except Exception:                                  # pragma: no cover - torch without dynamo
        return {}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _tables_digest_map(rec: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """``{logical table: sha256}`` of a tables record (``verify_tables``); ``{"synthetic": ...}`` for
    stand-in tables; None when the record is missing or unreadable."""

    if not isinstance(rec, dict):
        return None
    if rec.get("kind") == "synthetic":
        return {"synthetic": "synthetic"}
    files = rec.get("files")
    if not isinstance(files, dict):
        return None
    out = {}
    for k, v in files.items():
        sha = v.get("sha256") if isinstance(v, dict) else None
        if not isinstance(sha, str):
            return None
        out[str(k)] = sha
    return out


def warm_start(model, cfg: TrainConfig, tables_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Load ``cfg.init_from`` (``cfg.init_weights``: the EMA or the raw weights) into ``model`` strictly.

    Refuses (``ValueError``, nothing loaded): a checkpoint without the chosen weights; a model config
    that differs in any of ``WARM_START_MODEL_KEYS``; tokenizer-table digests that differ from the run's
    (``tables_record``, checked like the data shards); missing, unexpected or shape-mismatched state-dict
    keys.  No RNG is drawn.  Returns the init record (path, sha256, weights, source step / generation /
    trainer, the tolerated model-config differences) stored in the start event, run.json and every
    checkpoint."""

    import torch
    from amini.nn import schema_v1 as S
    from amini.nn.aminiformer import AminiFormerConfig
    path = cfg.init_from
    if not Path(path).is_file():
        raise FileNotFoundError(f"init_from: {path} is not a file")
    sha = file_sha256(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    key = {"ema": "model_ema", "model": "model_raw"}[cfg.init_weights]
    if not isinstance(ck, dict) or not isinstance(ck.get(key), dict) or "model_config" not in ck:
        raise ValueError(f"init_from {path}: not an il_v7 checkpoint with {key!r} and model_config")
    want = model.cfg.to_dict()
    have = AminiFormerConfig.from_dict(dict(ck["model_config"])).to_dict()
    bad_cfg = [k for k in WARM_START_MODEL_KEYS if want.get(k) != have.get(k)]
    if bad_cfg:
        raise ValueError("init_from: model config differs from the checkpoint in shape/function keys "
                         + ", ".join(f"{k}: run {want.get(k)!r} vs checkpoint {have.get(k)!r}" for k in bad_cfg))
    ck_tab, run_tab = _tables_digest_map(ck.get("tables")), _tables_digest_map(tables_record)
    if ck_tab is None or run_tab is None or ck_tab != run_tab:
        diff = sorted(k for k in set(ck_tab or {}) | set(run_tab or {})
                      if (ck_tab or {}).get(k) != (run_tab or {}).get(k))
        raise ValueError(f"init_from: tokenizer-table digests of the checkpoint differ from the run's "
                         f"tables ({diff if ck_tab is not None else 'checkpoint records no tables'}); "
                         f"the weights were trained on other vocab/table ids")
    sd = ck[key]
    cur = model.state_dict()
    missing = sorted(set(cur) - set(sd))
    unexpected = sorted(set(sd) - set(cur))
    shape = sorted(k for k in set(cur) & set(sd) if tuple(cur[k].shape) != tuple(sd[k].shape))
    if missing or unexpected or shape:
        raise ValueError(f"init_from {path} ({key}): state dict does not match the model: "
                         f"missing {missing[:8]}{'...' if len(missing) > 8 else ''} ({len(missing)}), "
                         f"unexpected {unexpected[:8]}{'...' if len(unexpected) > 8 else ''} ({len(unexpected)}), "
                         f"shape mismatch {shape[:8]} ({len(shape)})")
    model.load_state_dict({k: v.to(cur[k].dtype) for k, v in sd.items()}, strict=True)
    tolerated = {k: {"run": want.get(k), "checkpoint": have.get(k)}
                 for k in sorted(set(want) | set(have)) if want.get(k) != have.get(k)}
    return {"path": str(Path(path).resolve()), "sha256": sha, "weights": cfg.init_weights,
            "source_step": ck.get("step"), "source_generation": ck.get("generation"),
            "source_trainer": ck.get("trainer"), "source_schema": ck.get("schema"),
            "schema": S.SCHEMA_ID, "source_selection_nll": (ck.get("metrics") or {}).get("selection_nll"),
            "model_config_differences": tolerated}


# ------------------------------------------------------------------------- training lock
class TrainingLock:
    """Training-lock contract: ``LOCK_SH`` on ``<generations>/.training.lock`` for
    the whole run and a holder record naming the generation in use."""

    def __init__(self, data_paths: Sequence[str], generation: str) -> None:
        p = Path(data_paths[0]).resolve()
        if (p / "MANIFEST.json").exists() and not (p / "rows.npy").exists():
            lock_dir = p.parent                      # generations/<id> -> generations/
        elif (p / "rows.npy").exists():
            lock_dir = p.parent                      # shards/<sid> -> shards/
        else:
            lock_dir = p
        self.lock_path = lock_dir / ".training.lock"
        self.holder_dir = lock_dir / ".training.holders"
        self.generation = generation
        self.fh = None
        self.holder: Optional[Path] = None

    def __enter__(self) -> "TrainingLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.lock_path, "a+")
        fcntl.flock(self.fh, fcntl.LOCK_SH)
        self.holder_dir.mkdir(exist_ok=True)
        self.holder = self.holder_dir / f"{socket.gethostname()}-{os.getpid()}.json"
        self.holder.write_text(json.dumps({"generation": self.generation, "pid": os.getpid(),
                                           "host": socket.gethostname(), "started": time.time()}))
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.holder is not None:
            with contextlib.suppress(FileNotFoundError):
                self.holder.unlink()
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


# ------------------------------------------------------------------------- device
def device_setup(force_fp16: bool = False):
    import torch
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        cap = torch.cuda.get_device_capability()
        if cap >= (8, 0) and not force_fp16:
            return dev, torch.bfloat16, False
        return dev, torch.float16, True
    return torch.device("cpu"), torch.float32, False


REQUIRED_TABLES = ("card_vocab", "object_vocab", "object_catalog", "arrival")


def build_tables(cfg: TrainConfig):
    """Tokenizer tables from the frozen table files named in ``cfg.tables``.

    Stand-in synthetic tables (random catalog features, provisional arrival table) are
    opt-in only (``tables: {synthetic: true}``): a config without the four file paths
    is refused instead of silently tokenising real shards with random features.  Needs the
    shard tokenizer (``amini.nn.tokenizer_v7``, not part of this repository).
    """

    TK = _optional_module("amini.nn.tokenizer_v7", "the shard tokenizer")
    TokenizerTables, with_input_rules = TK.TokenizerTables, TK.with_input_rules
    t = dict(cfg.tables or {})
    rules = str((cfg.model or {}).get("input_rules", "") or "")   # the model's tokenizer-time rule set
    synthetic = bool(t.pop("synthetic", False))
    known = set(REQUIRED_TABLES) | {"visual_alias", "ability_costs"}
    unknown = set(t) - known
    if unknown:
        raise ValueError(f"unknown tables keys {sorted(unknown)}")
    if synthetic:      # explicit opt-in; verify_tables still refuses it on non-synthetic shards
        return with_input_rules(TokenizerTables.synthetic(), rules)
    missing = [k for k in REQUIRED_TABLES if not t.get(k)]
    if cfg.input_profile == "visual" and not t.get("visual_alias"):
        missing.append("visual_alias")
    from amini.nn import schema_v1 as _S
    if _S.input_rules(rules)["ability_rule"] == "tap" and not t.get("ability_costs"):
        missing.append("ability_costs")            # the 'tap' ability rule reads the ability cost table
    if missing:
        raise ValueError(
            f"config tables: {missing} not set. Real training needs the frozen table files "
            f"(card vocab, object vocab, object catalog, arrival table). "
            f"'tables: {{synthetic: true}}' is for smoke runs on synthetic shards only.")
    return with_input_rules(TokenizerTables.from_files(t["card_vocab"], t["object_vocab"], t["arrival"],
                                                       t["object_catalog"], t.get("visual_alias"),
                                                       ability_costs=t.get("ability_costs")), rules)


def verify_tables(tables, src: "ShardSource") -> Dict[str, Any]:
    """Refuse tables that are not the ones the data pins (every MANIFEST pins the
    vocab and table digests).  Returns the provenance record stored in run.json and in
    every checkpoint (the adapter must load the same tables)."""

    pins = src.table_digests()
    kind = tables.source.get("kind")
    if kind == "synthetic":
        if not src.synthetic or pins:
            raise ValueError("synthetic tokenizer tables on non-synthetic shards (MANIFESTs without "
                             "'synthetic': true, or pinning vocab digests); set the table files")
        return {"kind": "synthetic", "source": dict(tables.source)}
    files = tables.source.get("files", {})
    bad, unpinned = [], []
    for logical, rec in files.items():
        sha = rec["sha256"]
        want = None
        for alias in [logical] + _table_aliases(rec["path"]):
            if alias in pins:
                want = pins[alias]
                break
        if want is None and logical == "arrival" and getattr(tables, "arrival_rule", "table") == "batched":
            # a 'batched' arrival table is applied at tokenization (self-sealed, verified by from_files), not a
            # property of the stored data, so a generation that does not pin it can be tokenized with it;
            # recorded as unpinned in the provenance.
            unpinned.append(logical)
        elif want is None:
            bad.append(f"{logical} ({Path(rec['path']).name}): not pinned by any MANIFEST")
        elif want != sha:
            bad.append(f"{logical} ({Path(rec['path']).name}): sha256 {sha[:12]} != MANIFEST {want[:12]}")
    if bad:
        raise ValueError("tokenizer tables do not match the data: " + "; ".join(bad))
    if unpinned:
        return {"kind": "files", "files": files, "unpinned": unpinned}
    return {"kind": "files", "files": files}


def cell_and_slot(batch: Dict[str, Any]):
    import torch
    lab = batch["labels"]
    tg = batch.get("targets") or {}
    ok = lab["policy_ok"].bool()
    if "loss_mask" in lab:
        ok = ok & lab["loss_mask"].bool()
    play_slot = torch.where(ok & (lab["slot"] >= 0), lab["slot"], torch.full_like(lab["slot"], -1))
    cell_rows = tg.get("cell_rows")
    if cell_rows is None:
        cell_rows = lab["family"] == 1
    cell_rows = cell_rows.bool() | (play_slot >= 0)
    return cell_rows, play_slot


# ------------------------------------------------------------------------- evaluation
def evaluate(model, loader, device, amp_dtype, loss_w: Dict[str, float], max_batches: int = 0,
             opp_hand_reduce: str = DEFAULT_OPP_HAND_REDUCE) -> Dict[str, Any]:
    """Selection metric on EMA weights, per-stratum NLL, calibration, gate AUC/accuracy."""

    import torch
    import torch.nn.functional as F
    from amini.nn import schema_v1 as S
    TK = _optional_module("amini.nn.tokenizer_v7", "the shard tokenizer")

    model.eval()
    acc: Dict[str, List[float]] = {k: [0.0, 0.0] for k in ("gate", "slot", "cell", "ability")}
    strata = {s: {"gate": [0.0, 0.0]} for s in ("action", "base", "threat")}
    calib = {s: {"pred": 0.0, "obs": 0.0} for s in ("base", "threat")}
    diag = {"slot_top1": [0, 0], "cell_top1": [0, 0], "ability_top1": [0, 0], "joint_top1": [0, 0]}
    auc_scores, auc_labels, auc_w = [], [], []
    gate_acc = [0.0, 0.0]
    aux_terms: Dict[str, List[float]] = {}
    n_batches = 0
    acc_w: Dict[str, List[float]] = {k: [0.0, 0.0] for k in acc}
    n_weighted = 0
    def micro_batches():
        for item in loader:
            for nb in (item if isinstance(item, (list, tuple)) else [item]):
                yield nb

    with torch.no_grad():
        for nb in micro_batches():
            batch = TK.to_torch(nb, device)
            cell_rows, play_slot = cell_and_slot(batch)
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype != torch.float32 \
                else contextlib.nullcontext()
            with ctx:
                out = model(batch, cell_rows=cell_rows, play_slot=play_slot)
            res = compute_losses(out, batch, loss_w, opp_hand_reduce=opp_hand_reduce)
            for k in acc:
                acc[k][0] += float(res["sums"][k][0])
                acc[k][1] += float(res["sums"][k][1])
            if "eval_scale" in batch["labels"]:      # secondary metric (eval_skill_weights / eval_afk)
                lab_w = {k: v for k, v in batch["labels"].items() if k != "eval_scale"}
                lab_w["loss_scale"] = batch["labels"]["eval_scale"]
                res_w = compute_losses(out, {**batch, "labels": lab_w}, loss_w, opp_hand_reduce=opp_hand_reduce)
                for k in acc_w:
                    acc_w[k][0] += float(res_w["sums"][k][0])
                    acc_w[k][1] += float(res_w["sums"][k][1])
                n_weighted += 1
            for k, v in res["terms"].items():
                aux_terms.setdefault(k, []).append(float(v))
            lab = batch["labels"]
            rv = batch["row"]["row_valid"]
            fam = lab["family"][rv]
            lm = lab.get("loss_mask", torch.ones_like(lab["policy_ok"]).bool())[rv].float()
            okr = lab["policy_ok"][rv].float() * lm
            w = lab["row_weight"][rv].float() * okr
            ce = F.cross_entropy(out["family"].float(), torch.where(okr > 0, fam, torch.zeros_like(fam)),
                                 reduction="none")
            stratum = lab["stratum"][rv]
            for si, sn in ((0, "action"), (1, "base"), (2, "threat")):
                m = (stratum == si).float() * w
                strata[sn]["gate"][0] += float((m * ce).sum())
                strata[sn]["gate"][1] += float(m.sum())
            # calibration: predicted actions = sum over wait rows of w * (odds_p + odds_a)
            lf = out["family"].double()
            odds = torch.exp(lf[:, 1] - lf[:, 0]) + torch.where(
                out["ability_avail"], torch.exp(lf[:, 2] - lf[:, 0]), torch.zeros_like(lf[:, 0]))
            threat = lab["threat"][rv]
            is_wait = fam == 0
            for tv, sn in ((0, "base"), (1, "threat")):
                mw = is_wait & (threat == tv) & (okr > 0)
                calib[sn]["pred"] += float((w.double()[mw] * odds[mw]).sum())
                ma = (~is_wait) & (threat == tv) & (okr > 0)
                calib[sn]["obs"] += float(lm[ma].sum())
            p_act = 1.0 - F.softmax(out["family"].float(), -1)[:, 0]
            auc_scores.append(p_act.cpu().numpy())
            auc_labels.append((fam != 0).cpu().numpy())
            auc_w.append(w.cpu().numpy())
            pred = out["family"].argmax(-1)
            gate_acc[0] += float((w * (pred == fam).float()).sum())
            gate_acc[1] += float(w.sum())
            slot = lab["slot"][rv]
            ms = slot >= 0
            diag["slot_top1"][0] += int((out["slot"].argmax(-1)[ms] == slot[ms]).sum())
            diag["slot_top1"][1] += int(ms.sum())
            if "cells" in out and out["cell_play_index"].numel():
                ci = out["cell_play_index"]
                cl = lab["cell"][rv][ci]
                hit = out["cells"].argmax(-1) == cl
                diag["cell_top1"][0] += int(hit.sum())
                diag["cell_top1"][1] += int(ci.numel())
                sh = out["slot"].argmax(-1)[ci] == slot[ci]
                diag["joint_top1"][0] += int((hit & sh).sum())
                diag["joint_top1"][1] += int(ci.numel())
            ab = lab["ability"][rv]
            ma = ab >= 0
            diag["ability_top1"][0] += int((out["ability"].argmax(-1)[ma] == ab[ma]).sum())
            diag["ability_top1"][1] += int(ma.sum())
            n_batches += 1
            if max_batches and n_batches >= max_batches:
                break
    sel = selection_value({k: (v[0], v[1]) for k, v in acc.items()})
    s = np.concatenate(auc_scores) if auc_scores else np.zeros(0)
    y = np.concatenate(auc_labels) if auc_labels else np.zeros(0, bool)
    ww = np.concatenate(auc_w) if auc_w else np.zeros(0)
    report = {
        "selection_nll": sel,
        "terms": {k: v[0] / max(v[1], 1e-8) for k, v in acc.items()},
        "gate_nll_by_stratum": {k: v["gate"][0] / max(v["gate"][1], 1e-8) for k, v in strata.items()},
        "calibration": {k: {"pred": v["pred"], "obs": v["obs"],
                            "ratio": v["pred"] / v["obs"] if v["obs"] > 0 else float("nan"),
                            "flag": bool(v["obs"] > 0 and abs(v["pred"] / v["obs"] - 1.0) > 0.05)}
                        for k, v in calib.items()},
        "gate_auc": weighted_auc(s, y, ww),
        "gate_acc": gate_acc[0] / max(gate_acc[1], 1e-8),
        "diag": {k: v[0] / max(v[1], 1) for k, v in diag.items()},
        "aux_terms": {k: float(np.mean(v)) for k, v in aux_terms.items()},
        "batches": n_batches,
    }
    if n_weighted:                         # only with eval weighting (the default report has no "weighted")
        report["weighted"] = {"selection_nll": selection_value({k: (v[0], v[1]) for k, v in acc_w.items()}),
                              "terms": {k: v[0] / max(v[1], 1e-8) for k, v in acc_w.items()},
                              "micro_batches": n_weighted}
    return report


# ------------------------------------------------------------------------- gate calibration
def gate_bias_from_calibration(calibration: Dict[str, Dict[str, float]],
                               current: Optional[Sequence[Sequence[float]]] = None) -> List[List[float]]:
    """Per-stratum gate-logit bias that makes :func:`evaluate`'s calibration ratio exactly 1.

    ``calibration[s] = {pred, obs}`` (``evaluate(...)["calibration"]``, measured with the model's current
    ``cfg.gate_bias`` = ``current``): pred = sum over wait rows of w * (odds_play + odds_ability), so adding
    ``delta`` to the play and ability logits scales pred by ``exp(delta)``; ``delta_s = log(obs_s / pred_s)``.
    A stratum without predictions or observations keeps its current bias."""

    out = [list(map(float, r)) for r in (current if current is not None else [[0.0] * 3, [0.0] * 3])]
    for si, sn in ((0, "base"), (1, "threat")):
        c = calibration.get(sn) or {}
        pred, obs = float(c.get("pred", 0.0)), float(c.get("obs", 0.0))
        if pred > 0 and obs > 0 and math.isfinite(pred) and math.isfinite(obs):
            d = math.log(obs / pred)
            out[si][1] += d
            out[si][2] += d
    return out


def fit_gate_bias(family_logits: np.ndarray, family: np.ndarray, threat: np.ndarray, weight: np.ndarray,
                  obs_weight: np.ndarray, ability_avail: np.ndarray) -> List[List[float]]:
    """Array form of the fit on given val rows: gate logits ``[N, 3]`` (with any bias already in them),
    family labels, threat flag, the gate row weight (row_weight x policy_ok x loss_mask) and the
    observed-action weight (loss_mask, action rows); returns the ADDITIONAL [2, 3] bias."""

    lf = np.asarray(family_logits, np.float64)
    fam = np.asarray(family).astype(np.int64)
    thr = np.asarray(threat).astype(np.int64)
    w = np.asarray(weight, np.float64)
    wo = np.asarray(obs_weight, np.float64)
    odds = np.exp(lf[:, 1] - lf[:, 0]) + np.where(np.asarray(ability_avail, bool), np.exp(lf[:, 2] - lf[:, 0]), 0.0)
    cal = {}
    for tv, sn in ((0, "base"), (1, "threat")):
        mw = (fam == 0) & (thr == tv) & (w > 0)
        ma = (fam != 0) & (thr == tv) & (w > 0)
        cal[sn] = {"pred": float((w[mw] * odds[mw]).sum()), "obs": float(wo[ma].sum())}
    return gate_bias_from_calibration(cal)


def fit_gate_bias_on_loader(model, loader, device, amp_dtype, loss_w: Dict[str, float],
                            max_batches: int = 0) -> Dict[str, Any]:
    """Fit ``cfg.gate_bias`` on the rows of ``loader`` (e.g. a holdout) and return
    ``{"gate_bias", "before", "after"}`` (calibration reports with the old and the new bias). The model's
    ``cfg.gate_bias`` is set to the fitted value."""

    before = evaluate(model, loader, device, amp_dtype, loss_w, max_batches=max_batches)
    bias = gate_bias_from_calibration(before["calibration"], model.cfg.gate_bias)
    model.cfg.gate_bias = bias
    after = evaluate(model, loader, device, amp_dtype, loss_w, max_batches=max_batches)
    return {"gate_bias": bias, "before": before["calibration"], "after": after["calibration"],
            "gate_nll_before": before["gate_nll_by_stratum"], "gate_nll_after": after["gate_nll_by_stratum"]}


def set_gate_bias(ckpt_in: str, bias: Optional[Sequence[Sequence[float]]], ckpt_out: str) -> Dict[str, Any]:
    """Write ``model_config.gate_bias`` (None clears it) into a copy of a checkpoint; weights untouched."""

    import torch
    from amini.nn.aminiformer import AminiFormerConfig
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    mc["gate_bias"] = None if bias is None else [list(map(float, r)) for r in bias]
    AminiFormerConfig.from_dict(mc)                      # validates the shape
    ck["model_config"] = mc
    ck.setdefault("history_gate_bias", []).append(mc["gate_bias"])
    tmp = Path(ckpt_out).with_suffix(".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, ckpt_out)
    return mc


def weighted_auc(score: np.ndarray, label: np.ndarray, w: np.ndarray) -> float:
    if len(score) == 0 or label.all() or (~label).all():
        return float("nan")
    o = np.argsort(score, kind="stable")
    s, y, ww = score[o], label[o], w[o]
    wn = np.where(~y, ww, 0.0)
    wp = np.where(y, ww, 0.0)
    cum_neg = np.cumsum(wn) - wn            # negatives strictly below (ties ignored: continuous scores)
    return float((wp * (cum_neg + 0.5 * wn)).sum() / max(wp.sum() * wn.sum(), 1e-12))


# ------------------------------------------------------------------------- training
def lr_at(step: int, cfg: TrainConfig, total_steps: Optional[int] = None) -> float:
    """Linear warmup, then cosine to ``min_lr_frac`` at ``total_steps`` (default max_steps)."""

    total = int(total_steps) if total_steps else cfg.max_steps
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def _loader_worker_init(_worker_id: int) -> None:
    """DataLoader workers ignore SIGINT: a terminal Ctrl-C reaches the whole process group,
    and a worker that exits on it would abort the step the main process is finishing."""

    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def make_loader(ds, num_workers: int, start: int = 0, prefetch: int = 4):
    """Steps ``start ..`` of ``ds`` in order.  The loader's own generator seeds the workers,
    so creating a loader never draws from the global torch RNG (dropout): a run resumed
    mid-epoch draws the same dropout masks as the uninterrupted run.  ``prefetch``: items queued
    per worker (``loader_prefetch``)."""

    import torch
    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=num_workers,
                                       sampler=range(int(start), len(ds)),
                                       generator=torch.Generator().manual_seed(0),
                                       worker_init_fn=_loader_worker_init if num_workers else None,
                                       pin_memory=torch.cuda.is_available(),
                                       persistent_workers=False,
                                       prefetch_factor=int(prefetch) if num_workers else None)


def cap_val_items(src: "ShardSource", items, max_games: int):
    """Deterministic val subset: the ``max_games`` val games with the smallest
    sha256(replay_id) (both sides and every chunk of a chosen game; item order kept).
    Independent of shard layout and of the training seed."""

    if not max_games:
        return list(items)
    def key(si: int, gi: int) -> bytes:
        try:
            tag = bytes(src.shards[si]["games"]["replay_id"][gi])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            tag = f"{si}:{gi}".encode()
        return hashlib.sha256(tag).digest()
    games = sorted({(it[0][0], it[0][1]) for it in items}, key=lambda g: (key(*g), g))
    keep = set(games[:int(max_games)])
    return [it for it in items if (it[0][0], it[0][1]) in keep]


def _tag_text(src: "ShardSource", si: int, gi: int) -> str:
    return bytes(src.shards[si]["games"]["replay_id"][gi]).rstrip(b"\0").decode("ascii", "replace").strip()


def pinned_val_items(src: "ShardSource", items, games_file: str, max_games: int = 0):
    """Val items of exactly the games listed in ``games_file`` (replay ids, one per line; ``#`` comments).
    Raises if a listed game is not in ``items`` (not a val game of this generation) or if the list is
    longer than ``max_games`` (when > 0).  Item order kept.  Returns (items, sha256 of the file)."""

    raw = Path(games_file).read_bytes()
    want = {ln.split("#")[0].strip() for ln in raw.decode("ascii").splitlines()} - {""}
    if max_games and len(want) > int(max_games):
        raise ValueError(f"val_games_file lists {len(want)} games > val_max_games {max_games}")
    tag = {}
    for it in items:
        g = (it[0][0], it[0][1])
        if g not in tag:
            tag[g] = _tag_text(src, *g)
    found = {t for t in tag.values() if t in want}
    if found != want:
        miss = sorted(want - found)
        raise ValueError(f"val_games_file: {len(miss)} listed games are not val games of this "
                         f"generation (first {miss[:3]})")
    keep = {g for g, t in tag.items() if t in want}
    return [it for it in items if (it[0][0], it[0][1]) in keep], hashlib.sha256(raw).hexdigest()


def friendly_population_ids() -> FrozenSet[int]:
    """Population ids (GAME.population) of the friendly population: the indices of
    ``FRIENDLY_POPULATION`` in ``vocab.POPULATIONS`` (the id order GameModeVocab enforces).  Needs the
    vocabulary module ``amini.v7.vocab`` (not part of this repository)."""

    V = _optional_module("amini.v7.vocab", "the game-mode vocabulary")
    ids = frozenset(i for i, name in enumerate(V.POPULATIONS) if name == FRIENDLY_POPULATION)
    if not ids or FRIENDLY_POPULATION not in {r["population"] for r in V.POPULATION_RULES}:
        raise RuntimeError(f"vocab has no {FRIENDLY_POPULATION!r} population")
    return ids


def friendly_games(src: "ShardSource") -> set:
    """``{(shard, game)}`` of every friendly game of ``src`` (GAME.population in
    :func:`friendly_population_ids`).  A population id outside ``vocab.POPULATIONS`` raises (it cannot
    be classified)."""

    V = _optional_module("amini.v7.vocab", "the game-mode vocabulary")
    ids = np.asarray(sorted(friendly_population_ids()), dtype=np.int64)
    out = set()
    for si, t in enumerate(src.shards):
        pop = np.asarray(t["games"]["population"], dtype=np.int64)
        if pop.size and (int(pop.min()) < 0 or int(pop.max()) >= len(V.POPULATIONS)):
            raise ValueError(f"shard {si}: GAME.population outside 0..{len(V.POPULATIONS) - 1}")
        out.update((si, int(gi)) for gi in np.flatnonzero(np.isin(pop, ids)))
    return out


def listed_games(src: "ShardSource", games_file: str) -> Tuple[set, int, str]:
    """``{(shard, game)}`` of the games of ``src`` whose replay id is listed in ``games_file`` (one id per
    line; ``#`` comments), the number of ids listed and the file's sha256.  Listed ids that ``src`` does not
    hold are not an error (an ``exclude_games_file`` list may cover more games than one generation)."""

    raw = Path(games_file).read_bytes()
    want = {ln.split("#")[0].strip() for ln in raw.decode("ascii").splitlines()} - {""}
    out = set()
    for si, t in enumerate(src.shards):
        for gi in range(len(t["games"]["replay_id"])):
            if _tag_text(src, si, gi) in want:
                out.add((si, gi))
    return out, len(want), hashlib.sha256(raw).hexdigest()


def drop_games(items, games: set) -> Tuple[List[Any], Dict[str, int]]:
    """``items`` without the chunks of ``games`` (order kept) and the counts removed
    (games, items = chunks, rows = loss rows)."""

    kept, gone, n_items, n_rows = [], set(), 0, 0
    for it in items:
        g = (it[0][0], it[0][1])
        if g in games:
            gone.add(g)
            n_items += 1
            n_rows += int(it[3] - it[2])
        else:
            kept.append(it)
    return kept, {"games": len(gone), "items": n_items, "rows": n_rows}


# ------------------------------------------------------------------------- optional label weights
SKILL_WEIGHTS_SCHEMA = "il-v7-skill-weights"


def load_skill_weights(path: str, sha256_pin: str = "") -> Tuple[Dict[bytes, Tuple[float, float]], Dict[str, Any]]:
    """``({replay tag: (w side 0, w side 1)}, record)`` of a per-side row-weights file (JSON:
    ``{"schema": SKILL_WEIGHTS_SCHEMA, "sides": {replay tag: [w0, w1]}, "provenance": {...}}``).  Refuses
    another schema, a sha256 that differs from ``sha256_pin`` and any weight that is not finite and >= 0."""

    p = Path(path)
    sha = file_sha256(str(p))
    if sha256_pin and sha != sha256_pin:
        raise ValueError(f"skill_weights_file {p}: sha256 {sha[:12]} != pinned skill_weights_sha256 {sha256_pin[:12]}")
    doc = json.loads(p.read_text())
    if doc.get("schema") != SKILL_WEIGHTS_SCHEMA:
        raise ValueError(f"skill_weights_file {p}: schema {doc.get('schema')!r} != {SKILL_WEIGHTS_SCHEMA!r}")
    out: Dict[bytes, Tuple[float, float]] = {}
    for rid, w in doc["sides"].items():
        if len(w) != 2:
            raise ValueError(f"skill_weights_file {p}: {rid}: want 2 side weights, got {w!r}")
        a, b = float(w[0]), float(w[1])
        if not (math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0):
            raise ValueError(f"skill_weights_file {p}: {rid}: weights must be finite and >= 0: {w!r}")
        out[str(rid).encode()] = (a, b)
    prov = doc.get("provenance") or {}
    return out, {"file": str(p.resolve()), "sha256": sha, "games_in_file": len(out),
                 "generation_of_file": prov.get("generation")}


def afk_tail_starts(kind: np.ndarray, tick: np.ndarray, row_lo: np.ndarray, row_hi: np.ndarray,
                    real_end: np.ndarray, min_ticks: int) -> np.ndarray:
    """Per (game, side) the absolute row index from which the side's rows are an idle (AFK) tail (``row_hi`` = none):
    the row after the side's last play / ability row (ROW.kind 1 / 2) when ``real_end - tick of that row >=
    min_ticks``; a side without any action row: ``row_lo`` when ``real_end >= min_ticks``.  ``row_lo`` /
    ``row_hi``: [G, 2]; ``real_end``: [G] (<= 0 = unknown: no tail).  Recorded rows of the game only."""

    kind = np.asarray(kind)
    tick = np.asarray(tick, dtype=np.int64)
    lo = np.asarray(row_lo, dtype=np.int64)
    hi = np.asarray(row_hi, dtype=np.int64)
    re = np.asarray(real_end, dtype=np.int64)[:, None]
    n = len(kind)
    act = (kind == 1) | (kind == 2)
    pm = np.maximum.accumulate(np.where(act, np.arange(n, dtype=np.int64), -1)) if n else np.zeros(0, np.int64)
    nonempty = hi > lo
    last = np.where(nonempty, pm[np.clip(hi - 1, 0, max(n - 1, 0))] if n else -1, -1)
    has = nonempty & (last >= lo)
    ref = np.where(has, tick[np.clip(last, 0, max(n - 1, 0))] if n else 0, 0)
    tail = nonempty & (re > 0) & (re - ref >= int(min_ticks))
    return np.where(tail, np.where(has, last + 1, lo), hi)


class RowWeighting:
    """Optional per-row LABEL weights of one EpisodeBatches: per-side weights from a file (``skill_file``,
    :func:`load_skill_weights`) and the AFK-tail zero weight (``afk_tail_ticks``).  Training (:meth:`apply`,
    eval_mode False): AFK-tail rows get
    loss_mask False (context only, as burn-in rows) and every row the side's weight as labels ``loss_scale``
    (a multiplier on every loss term, :func:`loss_weights`).  Eval (eval_mode True): only the separate labels
    ``eval_scale`` = weight x (0 on the tail), read by :func:`evaluate` for its secondary ``weighted``
    report; the selection metric is untouched.  Built once in the trainer process; plain numpy arrays."""

    def __init__(self, src: "ShardSource", skill_file: str = "", afk_tail_ticks: int = 0,
                 sha256_pin: str = "") -> None:
        self.afk_tail_ticks = int(afk_tail_ticks)
        self.skill: Optional[Dict[str, Any]] = None
        self.side_w: Optional[List[np.ndarray]] = None
        self.found: Optional[List[np.ndarray]] = None
        self.tail_lo: Optional[List[np.ndarray]] = None
        if skill_file:
            table, self.skill = load_skill_weights(skill_file, sha256_pin)
            self.side_w, self.found = [], []
            for t in src.shards:
                rid = t["games"]["replay_id"]
                w = np.ones((len(rid), 2), np.float32)
                f = np.zeros(len(rid), bool)
                for gi in range(len(rid)):
                    v = table.get(bytes(rid[gi]))
                    if v is not None:
                        w[gi] = v
                        f[gi] = True
                self.side_w.append(w)
                self.found.append(f)
        if self.afk_tail_ticks:
            self.tail_lo = []
            for t in src.shards:
                r, g = t["rows"], t["games"]
                self.tail_lo.append(afk_tail_starts(np.asarray(r["kind"]), np.asarray(r["tick"]),
                                                    np.asarray(g["row_lo"]), np.asarray(g["row_hi"]),
                                                    np.asarray(g["real_end"]), self.afk_tail_ticks))

    @classmethod
    def build(cls, src: "ShardSource", skill_file: str = "", afk_tail_ticks: int = 0,
              sha256_pin: str = "") -> Optional["RowWeighting"]:
        return cls(src, skill_file, afk_tail_ticks, sha256_pin) if (skill_file or afk_tail_ticks) else None

    def side_weight(self, ep: Episode) -> float:
        return 1.0 if self.side_w is None else float(self.side_w[ep[0]][ep[1], ep[2]])

    def tail_start(self, ep: Episode) -> int:
        """Absolute row index where the episode's AFK tail starts (= the episode's row_hi: no tail)."""
        return int(ep[4]) if self.tail_lo is None else int(self.tail_lo[ep[0]][ep[1], ep[2]])

    def apply(self, labels: Dict[str, np.ndarray], ep: Episode, lo: int, hi: int, eval_mode: bool) -> None:
        n = hi - lo
        k = min(max(self.tail_start(ep) - lo, 0), n)          # rows [k, n) of this chunk are AFK tail
        if eval_mode:
            sc = np.full(n, self.side_weight(ep), np.float32)
            sc[k:] = 0.0
            labels["eval_scale"] = sc
            return
        if k < n:
            lm = labels["loss_mask"].copy()
            lm[k:] = False
            labels["loss_mask"] = lm
        if self.side_w is not None:
            labels["loss_scale"] = np.full(n, self.side_weight(ep), np.float32)

    def record(self, items) -> Dict[str, Any]:
        """Start-event counts over ``items`` (chunks; loss rows = [loss_lo, hi))."""
        rec: Dict[str, Any] = {}
        games = {(it[0][0], it[0][1]) for it in items}
        if self.skill is not None:
            miss = {g for g in games if not self.found[g[0]][g[1]]}
            rows_w: Dict[str, int] = collections.Counter()
            tot = wsum = 0.0
            for it in items:
                w = self.side_weight(it[0])
                n = int(it[3] - it[2])
                rows_w[repr(round(w, 6))] += n
                tot += n
                wsum += w * n
            rec["skill_weights"] = {**self.skill, "games": len(games), "games_missing": len(miss),
                                    "games_missing_weight": 1.0, "loss_rows": int(tot),
                                    "mean_row_weight": wsum / tot if tot else None,
                                    "loss_rows_by_weight": dict(sorted(rows_w.items()))}
        if self.tail_lo is not None:
            eps = {it[0] for it in items}
            tail_eps = sum(1 for e in eps if self.tail_start(e) < e[4])
            no_act = sum(1 for e in eps if self.tail_start(e) == e[3] and e[4] > e[3])
            zeroed = sum(max(0, int(it[3]) - max(int(it[2]), self.tail_start(it[0]))) for it in items)
            rows = sum(int(it[3] - it[2]) for it in items)
            rec["afk_tail"] = {"ticks": self.afk_tail_ticks, "episodes": len(eps), "tail_episodes": tail_eps,
                               "no_action_episodes": no_act, "loss_rows": rows, "loss_rows_zeroed": zeroed,
                               "zeroed_frac": zeroed / rows if rows else None}
        return rec


def _grad_stats(gns: Sequence[float], clip: float) -> Dict[str, Any]:
    """Pre-clip gradient-norm summary: median, p90, fraction above ``clip`` (finite norms),
    and the count of non-finite norms (fp16 overflow steps the scaler skips)."""

    a = np.asarray(gns, dtype=np.float64)
    fin = a[np.isfinite(a)]
    if not len(fin):
        return {"n": int(len(a)), "nonfinite": int(len(a))}
    return {"n": int(len(a)), "median": float(np.median(fin)), "p90": float(np.percentile(fin, 90)),
            "clipped_frac": float((fin > clip).mean()) if clip and clip > 0 else 0.0,
            "nonfinite": int(len(a) - len(fin))}


def cuda_alloc_setup(cfg: TrainConfig) -> Dict[str, Any]:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` (setdefault: an explicit environment value wins)
    before the CUDA caching allocator initialises.  ``expandable_segments`` avoids
    fragmentation out-of-memory errors near the GPU's memory limit.  Returns the record for
    run.json (``initialised_before``: CUDA was already initialised, so the value may not
    have taken effect)."""

    import torch
    before = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    if cfg.cuda_alloc_conf:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cfg.cuda_alloc_conf)
    return {"PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "initialised_before": before}


def _expandable_active() -> Optional[bool]:
    import torch
    if not torch.cuda.is_available():
        return None
    try:
        segs = torch.cuda.memory_snapshot()
        flags = [s.get("is_expandable") for s in segs if "is_expandable" in s]
        return bool(flags) and all(flags)
    except Exception:          # pragma: no cover - snapshot format differs
        return None


def make_optimizer(model, cfg: TrainConfig, device):
    """AdamW over the model's param groups; ``cfg.adamw_impl`` picks the kernel ("auto" = fused on CUDA)."""

    import torch
    groups = model.param_groups(cfg.weight_decay, cfg.lr_mult or None)
    if cfg.adamw_impl == "fused":
        kw: Dict[str, Any] = {"fused": True}
    elif cfg.adamw_impl == "foreach":
        kw = {"foreach": True, "fused": False}
    else:
        kw = {"fused": device.type == "cuda"}
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, **kw)


def host_index(batch: Dict[str, Any]) -> Dict[str, Any]:
    """``host_index``: on the host (loader worker), the integer indices of a collated micro-batch that the
    model forward and the losses otherwise derive on the device: ``rv`` (flat row-major index of the valid
    rows), ``ridx`` / ``pidx`` (:func:`loss_indices`), ``pidx_local`` (stage-P rows within ``ridx``, as the
    model derives them) and ``step`` (:meth:`StepEncoder.host_index`).  The arrays go through the same
    float64 -> float32 conversion as :func:`tokenizer_v7.to_torch` and the same torch ops as on the device."""

    import torch
    from amini.nn.aminiformer import StepEncoder
    TK = _optional_module("amini.nn.tokenizer_v7", "the shard tokenizer")
    lab, tg = batch["labels"], batch.get("targets") or {}
    sub = TK.to_torch({"row": {"row_valid": batch["row"]["row_valid"]},
                       "labels": {k: lab[k] for k in ("policy_ok", "loss_mask", "slot", "family") if k in lab},
                       "targets": {k: tg[k] for k in ("cell_rows",) if k in tg},
                       "groups": {n: {"valid": g["valid"]} for n, g in batch["groups"].items()}}, "cpu")
    rv = sub["row"]["row_valid"].bool()
    ridx, pidx = loss_indices(sub)
    cell_rows, play_slot = cell_and_slot(sub)
    ps = play_slot[rv][ridx].long()
    pidx_local = (ps >= 0).nonzero(as_tuple=True)[0]
    return {"rv": rv.reshape(-1).nonzero(as_tuple=True)[0], "ridx": ridx, "pidx": ridx[pidx_local],
            "pidx_local": pidx_local,
            "step": StepEncoder.host_index({n: g["valid"][rv] for n, g in sub["groups"].items()})}


def total_train_steps(cfg: TrainConfig, steps_per_epoch: int) -> int:
    """Optimizer steps of the run (= the cosine horizon): ``ceil(max_epochs x steps of one
    train epoch)`` when ``max_epochs > 0``, else ``max_steps``."""

    if cfg.max_epochs and cfg.max_epochs > 0:
        return max(1, int(math.ceil(cfg.max_epochs * steps_per_epoch - 1e-9)))
    return int(cfg.max_steps)


def _config_diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Keys whose values differ (``RESUME_FREE_KEYS`` and table paths ignored; the table
    contents are compared through their digests)."""

    skip = set(RESUME_FREE_KEYS) | {"tables"}
    keys = (set(a) | set(b)) - skip
    norm = lambda v: json.loads(json.dumps(v, default=str))          # tuples == lists
    # a key one side does not record (a checkpoint written before the key existed) has its default
    dflt = asdict(TrainConfig())
    get = lambda d, k: d[k] if k in d else dflt.get(k)
    return sorted(k for k in keys if norm(get(a, k)) != norm(get(b, k)))


class _StopFlag:
    """SIGINT / SIGTERM set a flag; the trainer finishes the current optimizer step, saves a
    resumable ``last.pt``, writes ``result.json`` and returns (exit status 0).  A second
    SIGINT raises ``KeyboardInterrupt`` (the operator insists)."""

    def __init__(self) -> None:
        self.signum: Optional[int] = None
        self._old: Dict[int, Any] = {}

    def _handle(self, signum, _frame) -> None:
        import signal
        if self.signum is not None and signum == signal.SIGINT:
            raise KeyboardInterrupt
        self.signum = int(signum)

    def __enter__(self) -> "_StopFlag":
        import signal
        import threading
        if threading.current_thread() is threading.main_thread():
            for s in (signal.SIGINT, signal.SIGTERM):
                self._old[s] = signal.signal(s, self._handle)
        return self

    def __exit__(self, *exc: Any) -> None:
        import signal
        for s, h in self._old.items():
            signal.signal(s, h)


def _link_replace(src: Path, dst: Path) -> None:
    """``dst`` becomes a hard link to ``src`` (atomic replace; same directory)."""

    tmp = dst.with_name(dst.name + ".lnk")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    os.link(src, tmp)
    os.replace(tmp, dst)


def train(cfg: TrainConfig, data: Sequence[str], out_dir: str, *, log=print,
          benchmark_steps: int = 0, resume: bool = False) -> Dict[str, Any]:
    """Run (or with ``resume`` continue from ``<out>/last.pt``) one training run.

    Stops at the total step count (:func:`total_train_steps`), on early stopping, or on
    SIGINT/SIGTERM (after the current step; ``last.pt`` is resumable).  Writes
    ``result.json`` in every case; ``stop_reason`` says which.
    """

    import torch
    from amini.nn.aminiformer import AminiFormer, AminiFormerConfig, count_parameters
    TK = _optional_module("amini.nn.tokenizer_v7", "the shard tokenizer")

    alloc = cuda_alloc_setup(cfg)
    tables = build_tables(cfg)                         # config errors before any lock or data access
    rd = resolve_data(data)                            # generation id only; no table is opened yet
    out = Path(out_dir)
    with TrainingLock(data, rd.generation) as lock, _StopFlag() as stop:
        src, gen_id = ShardSource.open(data, rd)
        tables_record = verify_tables(tables, src)
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        device, amp_dtype, use_scaler = device_setup(cfg.force_fp16)
        mcfg = AminiFormerConfig.from_dict(cfg.model)
        model = AminiFormer(mcfg).to(device)
        apply_speed_settings(model, cfg)                # runtime switches: not weights, not saved
        if not cfg.aux_opp_hand_prior_bias:            # bias zeroed instead of the 4-of-V prior (no RNG draw)
            with torch.no_grad():
                model.aux_opp_hand[-1].bias.zero_()
        n_params = count_parameters(model)
        init_record: Optional[Dict[str, Any]] = None
        if cfg.init_from and not resume:               # resume wins: last.pt holds the whole state
            init_record = warm_start(model, cfg, tables_record)
        opt = make_optimizer(model, cfg, device)
        scaler = torch.amp.GradScaler("cuda") if use_scaler else None
        ema = EMA(model, cfg.ema_decay, foreach=cfg.ema_foreach)
        train_items = chunk_episodes(src, 0, cfg.max_seq_rows, cfg.burn_in_rows)
        val_all = val_listed = chunk_episodes(src, 1, cfg.max_seq_rows, cfg.burn_in_rows)
        friendly_record: Optional[Dict[str, Any]] = None
        fr: set = set()
        if cfg.exclude_friendly:
            # friendly games leave train and val before any selection (the smallest-sha256 val subset
            # is drawn from the non-friendly val games; a pinned list loses its friendly games below)
            fr = friendly_games(src)
            train_items, d_train = drop_games(train_items, fr)
            val_all, d_val = drop_games(val_all, fr)
            friendly_record = {"population": FRIENDLY_POPULATION,
                               "population_ids": sorted(friendly_population_ids()),
                               "games_in_source": len(fr), "train": d_train, "val": d_val}
        excl_record: Optional[Dict[str, Any]] = None
        ex: set = set()
        if cfg.exclude_games_file:
            # listed games leave train and val before any selection (as the friendly games above)
            ex, ex_listed, ex_sha = listed_games(src, cfg.exclude_games_file)
            train_items, d_train = drop_games(train_items, ex)
            val_all, d_val = drop_games(val_all, ex)
            excl_record = {"file": cfg.exclude_games_file, "sha256": ex_sha, "listed": ex_listed,
                           "games_in_source": len(ex), "train": d_train, "val": d_val}
        val_pin = None
        if cfg.val_games_file:
            # every listed game must be a val game of the generation (friendly or not); with
            # exclude_friendly the listed friendly games are then dropped and counted
            val_items, val_pin = pinned_val_items(src, val_listed, cfg.val_games_file, cfg.val_max_games)
            if friendly_record is not None:
                n_listed = len({(it[0][0], it[0][1]) for it in val_items})
                val_items, d_pin = drop_games(val_items, fr)
                friendly_record["val_games_file"] = {"listed": n_listed, "dropped": d_pin["games"],
                                                     "kept": n_listed - d_pin["games"]}
            if excl_record is not None:
                n_listed = len({(it[0][0], it[0][1]) for it in val_items})
                val_items, d_pin = drop_games(val_items, ex)
                excl_record["val_games_file"] = {"listed": n_listed, "dropped": d_pin["games"],
                                                 "kept": n_listed - d_pin["games"]}
        else:
            val_items = cap_val_items(src, val_all, cfg.val_max_games)
        # optional label weights (None = off); the eval weighting only feeds the secondary report["weighted"]
        train_w = RowWeighting.build(src, cfg.skill_weights_file, cfg.afk_tail_ticks, cfg.skill_weights_sha256)
        if (cfg.eval_skill_weights_file, cfg.eval_afk_tail_ticks) == (cfg.skill_weights_file, cfg.afk_tail_ticks):
            eval_w = train_w
        else:
            eval_w = RowWeighting.build(src, cfg.eval_skill_weights_file, cfg.eval_afk_tail_ticks)
        train_ds = EpisodeBatches(src, train_items, tables, cfg, train=True, weighting=train_w)
        val_ds = EpisodeBatches(src, val_items, tables, cfg, train=False, weighting=eval_w)
        steps_per_epoch = len(train_ds)
        total_steps = total_train_steps(cfg, steps_per_epoch)     # cosine horizon
        run_steps = int(benchmark_steps) if benchmark_steps else total_steps
        n_micro = [len(b) for b in train_ds.batches]
        # max_batch_tokens guard: a single chunk always runs, even above the cap; count them
        over = 0
        if cfg.max_batch_tokens:
            for stp in train_ds.batches:
                for m in stp:
                    rows = sum(train_items[i][3] - train_items[i][1] for i in m)
                    if micro_cost(rows, max(train_ds.tok[i][0] for i in m),
                                  max(train_ds.tok[i][1] for i in m)) > cfg.max_batch_tokens:
                        over += 1
        info = {"params": n_params, "device": str(device), "amp": str(amp_dtype),
                "scaler": use_scaler, "train_items": len(train_items), "val_items": len(val_items),
                "val_items_all": len(val_all),
                "val_games": len({(it[0][0], it[0][1]) for it in val_items}),
                "train_rows": int(sum(it[3] - it[2] for it in train_items)),
                "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "run_steps": run_steps,
                "generation": gen_id, "lock": str(lock.lock_path), "cuda_alloc": alloc,
                "batch_rows": cfg.batch_rows, "micro_batch_rows": cfg.micro_rows,
                "max_batch_tokens": cfg.max_batch_tokens, "over_token_cap_micro": over,
                "micro_per_step": {"mean": float(np.mean(n_micro)) if n_micro else 0.0,
                                   "max": int(max(n_micro, default=0))},
                "tables": tables_record,
                "val_games_file": cfg.val_games_file or None, "val_games_file_sha256": val_pin}
        if friendly_record is not None:                 # logged only when on (the default start event is unchanged)
            info["exclude_friendly"] = friendly_record
        if excl_record is not None:                     # logged only when on
            info["exclude_games"] = excl_record
        if train_w is not None:                         # logged only when on
            info["row_weighting"] = train_w.record(train_items)
        if eval_w is not None:
            info["eval_row_weighting"] = eval_w.record(val_items)
        if cfg.lr_mult:
            info["lr_mult_groups"] = [{"lr_mult": g["lr_mult"], "weight_decay": g["weight_decay"],
                                       "params": len(g["params"]),
                                       "numel": int(sum(p.numel() for p in g["params"]))}
                                      for g in opt.param_groups]
        st: Dict[str, Any] = {"step": 0, "epoch": 0, "pos": 0, "best": [], "history": [],
                              "es_best": None, "es_best_step": None, "es_bad": 0,
                              "train_s": 0.0, "eval_s": 0.0, "rows": 0, "gn": [], "resumes": []}
        last_path = out / RESUME_FILE
        if resume:
            if not last_path.exists():
                raise FileNotFoundError(f"--resume: {last_path} does not exist")
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
            if not ck.get("resume"):
                raise ValueError(f"{last_path} holds no resume state")
            diff = _config_diff(asdict(cfg), ck["train_config"])
            if diff:
                raise ValueError(f"--resume: config differs from the checkpoint in {diff}")
            if ck.get("generation") != gen_id:
                raise ValueError(f"--resume: generation {gen_id} != checkpoint {ck.get('generation')}")
            if json.dumps(ck.get("tables"), sort_keys=True, default=str) != \
                    json.dumps(tables_record, sort_keys=True, default=str):
                raise ValueError("--resume: table digests differ from the checkpoint")
            model.load_state_dict(ck["model_raw"])
            ema.shadow = {k: v.to(device) for k, v in ck["model_ema"].items()}
            opt.load_state_dict(ck["optimizer"])
            if scaler is not None and ck.get("scaler") is not None:
                scaler.load_state_dict(ck["scaler"])
            r = ck["resume"]
            if int(r["total_steps"]) != total_steps:
                raise ValueError(f"--resume: total steps {total_steps} != checkpoint {r['total_steps']}")
            st.update({k: r[k] for k in st if k in r})
            ema.steps = int(r.get("ema_steps", st["step"]))
            torch.set_rng_state(r["rng_torch"])
            if device.type == "cuda" and r.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(r["rng_cuda"])
            np.random.set_state(r["rng_numpy"])
            st["resumes"] = list(st["resumes"]) + [int(st["step"])]
            init_record = ck.get("init")
            del ck
        info["resumed_from"] = st["resumes"][-1] if st["resumes"] else None
        info["init"] = init_record
        log(json.dumps({"event": "start", **info}, default=str))
        (out / ("run.json" if not resume else f"run-resume-{st['step']:07d}.json")).write_text(
            json.dumps({"config": asdict(cfg), "info": info}, indent=1, default=str))

        def resume_state() -> Dict[str, Any]:
            return {**{k: st[k] for k in st}, "total_steps": total_steps, "ema_steps": ema.steps,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "rng_numpy": np.random.get_state()}

        if cfg.eval_at_start and not resume and not benchmark_steps and st["step"] == 0 and len(val_ds):
            # selection subset before the first step: a normal step-0 eval (log, history, ckpt-0000000.pt,
            # best.pt, early-stopping reference); it reads only the EMA (= the init weights here)
            _eval_and_save(model, ema, val_ds, cfg, device, amp_dtype, 0, out, st, opt, scaler, gen_id,
                           log, tables_record, resume_state, init_record)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_bench0 = None
        rows_bench = 0
        stop_reason: Optional[str] = None
        t_last = time.time()
        eval_in_iv = 0.0                               # eval seconds inside the current log interval
        rows_since = 0
        gn_buf: List[Any] = []
        defer = cfg.defer_stats
        pend_bad: List[Tuple[int, Any]] = []           # defer_stats: (step, device flag: non-finite loss)
        pend_chk: List[Any] = []                       # defer_stats: device flags of the index checks

        def flush_checks() -> None:
            """defer_stats: read the pending device flags with one transfer; raise as the per-step checks would."""
            if not pend_bad and not pend_chk:
                return
            flags = torch.stack([f for _, f in pend_bad] + pend_chk).cpu().tolist()
            bad = [stp for (stp, _), f in zip(pend_bad, flags) if f]
            chk = any(flags[len(pend_bad):])
            pend_bad.clear()
            pend_chk.clear()
            if bad:
                raise FloatingPointError(f"non-finite loss at step {bad[0]} (defer_stats: detected at "
                                         f"step {st['step']})")
            if chk:
                raise RuntimeError("model cell_row_index / cell_play_index differs from loss_indices(batch)")

        while st["step"] < run_steps and stop_reason is None:
            train_ds.set_epoch(st["epoch"])
            loader = make_loader(train_ds, cfg.num_workers, start=st["pos"], prefetch=cfg.loader_prefetch)
            for micro in loader:
                model.train()
                step = st["step"]
                micro = micro if isinstance(micro, (list, tuple)) else [micro]
                batches = [TK.to_torch(nb, device) for nb in micro]
                # step-wide denominators (labels only): the micro-batch losses add up to the
                # exact full-step loss, however the step was split
                dens: Dict[str, Any] = {}
                for b in batches:
                    for k, v in loss_denominators(b).items():
                        dens[k] = dens[k] + v if k in dens else v
                lr_now = lr_at(step, cfg, total_steps)
                for gp in opt.param_groups:
                    gp["lr"] = lr_now * gp["lr_mult"] if "lr_mult" in gp else lr_now
                opt.zero_grad(set_to_none=True)
                loss_total = 0.0
                terms_total: Dict[str, float] = {}
                n_rows = 0
                # defer_stats: is this step logged (the log condition below)?  Only then are its values read
                will_log = (step + 1) % cfg.log_every == 0 or step + 1 == 1 or step + 1 >= run_steps
                kept: List[Tuple[Any, List[Tuple[str, Any]]]] = []
                for b, nb in zip(batches, micro):
                    cell_rows, play_slot = cell_and_slot(b)
                    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) \
                        if amp_dtype != torch.float32 else contextlib.nullcontext()
                    with ctx:
                        o = model(b, cell_rows=cell_rows, play_slot=play_slot)
                    res = compute_losses(o, b, cfg.loss, dens=dens,
                                         opp_hand_reduce=cfg.aux_opp_hand_reduce,
                                         deferred_checks=pend_chk if defer else None)
                    loss = res["total"]
                    if defer:
                        pend_bad.append((step, ~torch.isfinite(loss.detach())))
                    elif not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {step}: "
                                                 f"{ {k: float(v) for k, v in res['terms'].items()} }")
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if defer:
                        if will_log:
                            kept.append((loss.detach(), [(k, v.detach()) for k, v in res["terms"].items()]))
                        n_rows += int(nb["row"]["row_valid"].sum())      # host copy: no device sync
                    else:
                        loss_total += float(loss)
                        for k, v in res["terms"].items():
                            terms_total[k] = terms_total.get(k, 0.0) + float(v)
                        n_rows += int(b["row"]["row_valid"].sum())
                    del o, res, loss
                if defer and kept:
                    # one transfer; float64 holds every fp32 value exactly, summed in the per-step order
                    vals = iter(torch.stack([t.double() for lt, tt in kept for t in [lt] + [v for _, v in tt]])
                                .cpu().tolist())
                    for _, tt in kept:
                        loss_total += next(vals)
                        for k, _ in tt:
                            terms_total[k] = terms_total.get(k, 0.0) + next(vals)
                    kept = []
                if scaler is not None:
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                ema.update(model)
                gn_buf.append(gn.detach())                 # pre-clip norm; read at log time (no sync)
                rows_since += n_rows
                if benchmark_steps:
                    if step == min(10, benchmark_steps // 4):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t_bench0 = time.time()
                        rows_bench = 0
                    elif t_bench0 is not None:
                        rows_bench += n_rows
                st["step"] = step = step + 1
                st["pos"] += 1
                # eval schedule (eval_every_early / eval_late_frac; the eval_every rule by default) and
                # resume-only checkpoints (ckpt_every) on the steps without an eval checkpoint
                ev_now = not benchmark_steps and cfg.evaluates_at(step, total_steps)
                ck_now = bool(not benchmark_steps and cfg.ckpt_every and step % cfg.ckpt_every == 0
                              and not (ev_now and len(val_ds)) and not (step >= run_steps and len(val_ds)))
                if defer and (will_log or stop.signum is not None or ev_now or ck_now):
                    flush_checks()                     # before any log record, eval or checkpoint
                if step % cfg.log_every == 0 or step == 1 or step >= run_steps:
                    gns = torch.stack(gn_buf).float().cpu().numpy().tolist()
                    gn_buf = []
                    st["gn"].extend(gns)
                    now = time.time()
                    dt = max(now - t_last - eval_in_iv, 1e-9)
                    st["train_s"] += dt
                    st["rows"] += rows_since
                    gs = _grad_stats(gns, cfg.grad_clip)
                    rec = {"step": step, "epoch": st["epoch"], "loss": loss_total,
                           "lr": lr_at(step, cfg, total_steps), "gn": gns[-1],
                           "gn_median": gs.get("median"), "gn_p90": gs.get("p90"),
                           "clipped_frac": gs.get("clipped_frac"), "gn_nonfinite": gs["nonfinite"],
                           "rows_s": rows_since / dt, "micro": len(batches), **terms_total}
                    if scaler is not None:
                        rec["scale"] = float(scaler.get_scale())
                    if cfg.compile_mode:
                        rec["compile"] = compile_counters()
                    if device.type == "cuda":
                        rec["peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
                    st["history"].append(rec)
                    log(json.dumps(rec))
                    t_last, rows_since, eval_in_iv = now, 0, 0.0
                if ev_now and len(val_ds):
                    rep, es = _eval_and_save(model, ema, val_ds, cfg, device, amp_dtype, step, out,
                                             st, opt, scaler, gen_id, log, tables_record, resume_state,
                                             init_record)
                    eval_in_iv += rep["eval_seconds"]
                    if es:
                        stop_reason = "early_stop"
                        log(json.dumps({"event": "early_stop", "step": step, "best_step": st["es_best_step"],
                                        "best_selection_nll": st["es_best"], "bad_evals": st["es_bad"]}))
                if ck_now and stop.signum is None and stop_reason is None:
                    _save_resume_only(model, ema, cfg, step, out, opt, scaler, gen_id, log, tables_record,
                                      resume_state, init_record)
                if stop.signum is not None and stop_reason is None:
                    stop_reason = "signal"
                if stop_reason is not None or step >= run_steps:
                    break
            del loader
            if st["pos"] >= len(train_ds):
                st["epoch"] += 1
                st["pos"] = 0
        if stop_reason is None:
            stop_reason = "benchmark" if benchmark_steps else "total_steps"
        flush_checks()
        if gn_buf:                                     # steps after the last log record
            st["gn"].extend(torch.stack(gn_buf).float().cpu().numpy().tolist())
            gn_buf = []
            st["train_s"] += max(time.time() - t_last - eval_in_iv, 0.0)
            st["rows"] += rows_since
        info["cuda_expandable_active"] = _expandable_active()
        step = st["step"]
        if stop_reason == "signal":
            # replaces the directory entry only (last.pt may be a hard link to an eval checkpoint)
            save_checkpoint(last_path, model, ema, cfg, step, {"stop": "signal"}, opt, scaler,
                            gen_id, tables_record, resume_state(), init_record)
            log(json.dumps({"event": "signal_stop", "signal": stop.signum, "step": step,
                            "checkpoint": str(last_path)}))
        elif not benchmark_steps and len(val_ds) and not cfg.evaluates_at(step, total_steps):
            _eval_and_save(model, ema, val_ds, cfg, device, amp_dtype, step, out, st, opt, scaler,
                           gen_id, log, tables_record, resume_state, init_record)
        result: Dict[str, Any] = {"info": info, "steps": step, "stop_reason": stop_reason,
                                  "signal": stop.signum, "epochs_done": st["epoch"] + st["pos"] / max(1, len(train_ds)),
                                  "history": st["history"], "resumes": st["resumes"]}
        result["timing"] = {"train_seconds": st["train_s"], "eval_seconds": st["eval_s"],
                            "rows": st["rows"], "rows_per_s": st["rows"] / max(st["train_s"], 1e-9)}
        result["grad_norm"] = _grad_stats(st["gn"], cfg.grad_clip)
        if benchmark_steps and t_bench0 is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            el = time.time() - t_bench0
            result["benchmark"] = {"rows_per_s": rows_bench / max(el, 1e-9), "seconds": el,
                                   "rows": rows_bench,
                                   "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                                   if device.type == "cuda" else None),
                                   "peak_reserved_mb": (torch.cuda.max_memory_reserved() / 2 ** 20
                                                        if device.type == "cuda" else None)}
        result["best"] = st["best"]
        if st["best"]:
            result["best_step"], result["best_selection_nll"] = st["best"][0][1], st["best"][0][0]
            result["best_ckpt"] = st["best"][0][2]
        result["early_stop"] = {"patience": cfg.early_stop_patience, "min_steps": cfg.early_stop_min_steps,
                                "min_delta": cfg.early_stop_min_delta, "best": st["es_best"],
                                "best_step": st["es_best_step"], "bad_evals": st["es_bad"]}
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1, default=str))
        os.replace(tmp, out / "result.json")
        return result


def _early_stop(cfg: TrainConfig, st: Dict[str, Any], rep: Dict[str, Any], step: int) -> bool:
    """Track the best EMA val selection NLL; True once ``early_stop_patience`` evals at or
    after ``early_stop_min_steps`` brought no improvement larger than ``min_delta``."""

    sel = float(rep["selection_nll"])
    if st["es_best"] is None or sel < float(st["es_best"]) - cfg.early_stop_min_delta:
        st["es_best"], st["es_best_step"], st["es_bad"] = sel, step, 0
        return False
    if step >= cfg.early_stop_min_steps:
        st["es_bad"] += 1
    return bool(cfg.early_stop_patience and st["es_bad"] >= cfg.early_stop_patience)


def _eval_and_save(model, ema, val_ds, cfg, device, amp_dtype, step, out, st, opt, scaler,
                   gen_id, log, tables_record=None, resume_state=None, init_record=None):
    """EMA eval on the (capped) val split, early-stop update, checkpoint with resume state;
    ``last.pt`` and ``best.pt`` are hard links to the newest and the best checkpoint.
    Returns ``(report, early_stop_now)``."""

    t0 = time.time()
    eval_model = copy.deepcopy(model)
    ema.copy_to(eval_model)
    rep = evaluate(eval_model, make_loader(val_ds, min(cfg.num_workers, 4), prefetch=cfg.loader_prefetch),
                   device, amp_dtype, cfg.loss, cfg.eval_max_batches, opp_hand_reduce=cfg.aux_opp_hand_reduce)
    del eval_model
    rep["step"] = step
    rep["eval_seconds"] = time.time() - t0
    st["eval_s"] += rep["eval_seconds"]
    log(json.dumps({"event": "eval", **rep}, default=str))
    st["history"].append({"eval": rep})
    path = out / f"ckpt-{step:07d}.pt"
    es = _early_stop(cfg, st, rep, step)              # before the save: a resume continues identically
    best = sorted([b for b in st["best"] if b[2] != str(path)] + [[rep["selection_nll"], step, str(path)]])
    keep, drop = best[:cfg.keep_best], best[cfg.keep_best:]
    st["best"] = keep
    save_checkpoint(path, model, ema, cfg, step, rep, opt, scaler, gen_id, tables_record,
                    resume_state() if resume_state else None, init_record)
    for _, _, p in drop:
        with contextlib.suppress(FileNotFoundError):
            Path(p).unlink()
    if Path(path).exists():
        _link_replace(path, out / RESUME_FILE)
    if keep and Path(keep[0][2]).exists():
        _link_replace(Path(keep[0][2]), out / "best.pt")
    (out / "best.json").write_text(json.dumps(keep, indent=1))
    return rep, es


RESUME_ONLY_KEEP = 2                   # ckpt_every: newest resume-only checkpoints kept


def _save_resume_only(model, ema, cfg, step, out, opt, scaler, gen_id, log, tables_record=None,
                      resume_state=None, init_record=None) -> Path:
    """``ckpt_every``: a resume-only checkpoint ``resume-NNNNNNN.pt`` (the eval checkpoint's payload and
    resume state, no eval) and ``last.pt`` relinked to it.  Not ranked: ``st["best"]``, ``best.json`` and
    ``best.pt`` are untouched; the resume-only files beyond the newest ``RESUME_ONLY_KEEP`` are removed (ranked
    ``ckpt-*.pt`` files are never touched)."""

    t0 = time.time()
    path = out / f"resume-{step:07d}.pt"
    save_checkpoint(path, model, ema, cfg, step, {"resume_only": True, "step": step}, opt, scaler, gen_id,
                    tables_record, resume_state() if resume_state else None, init_record)
    _link_replace(path, out / RESUME_FILE)
    old = sorted(p for p in out.glob("resume-*.pt") if p.name[7:-3].isdigit())[:-RESUME_ONLY_KEEP]
    for p in old:
        with contextlib.suppress(FileNotFoundError):
            p.unlink()
    log(json.dumps({"event": "resume_checkpoint", "step": step, "checkpoint": str(path),
                    "seconds": time.time() - t0, "pruned": [p.name for p in old]}))
    return path


# ------------------------------------------------------------------------- CLI
SET_SECTIONS = ("model.", "tables.", "loss.", "lr_mult.")


def parse_set_overrides(base: Dict[str, Any], items: Sequence[str]) -> Dict[str, Any]:
    """``--set`` items ``key=value`` -> config overrides.  ``<section>.<sub>=v`` for the mapping
    sections (``SET_SECTIONS``) updates one entry and keeps the section's other entries from ``base``
    (``lr_mult.cells.aux_threat.=0.5``: the sub key is everything after the first dot)."""

    ov: Dict[str, Any] = {}
    for item in items:
        k, _, v = item.partition("=")
        if k.startswith(SET_SECTIONS):
            sec, _, sub = k.partition(".")
            m = dict(ov.get(sec, base.get(sec, {}) or {}))
            m[sub] = _scalar(v)
            ov[sec] = m
        else:
            ov[k] = _scalar(v)
    return ov


def main(argv: Optional[List[str]] = None) -> int:
    runtime_check()                                    # before parsing data paths or loading shards
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--batch-rows", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--eval-every", type=int)
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps and report rows/s + peak memory")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/last.pt (same config except the resume-free keys)")
    ap.add_argument("--init-from", help="warm start: checkpoint whose weights initialise the run (init_from)")
    ap.add_argument("--init-weights", choices=INIT_WEIGHTS, help="ema (default) | model (init_weights)")
    ap.add_argument("--eval-at-start", action="store_true", default=None,
                    help="evaluate the selection subset before the first step (eval_at_start)")
    ap.add_argument("--set", nargs="*", action="append", default=[],
                    help="config overrides key=value (model.key=value, tables.key=value, loss.term=w, "
                         "lr_mult.<prefix>=m); repeatable")
    a = ap.parse_args(argv)
    base = load_yaml(Path(a.config).read_text())
    ov = parse_set_overrides(base, [x for group in a.set for x in group])
    for k in ("max_steps", "batch_rows", "num_workers", "eval_every", "init_from", "init_weights",
              "eval_at_start"):
        v = getattr(a, k)
        if v is not None:
            ov[k] = v
    cfg = TrainConfig.from_file(a.config, ov)
    if cfg.cuda_alloc_conf:                            # before torch touches CUDA
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cfg.cuda_alloc_conf)
    res = train(cfg, a.data, a.out, benchmark_steps=a.benchmark, resume=a.resume,
                log=lambda m: print(m, flush=True))
    print(json.dumps({"done": True, "steps": res["steps"], "stop_reason": res["stop_reason"],
                      "benchmark": res.get("benchmark"), "best": res.get("best")}, default=str),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
