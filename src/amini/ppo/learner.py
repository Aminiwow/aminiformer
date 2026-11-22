"""PPO learner for aminiformer v7: policy, frozen reference, separate critic, one update per call.

What one update does (:meth:`Learner.update`)
---------------------------------------------
1. Read the update's shards (:func:`amini.ppo.buffer.load_update`) with ``expect_policy_digest`` = the
   sha256 of the current acting weights, so a stale server or a second attempt cannot slip in (lag 0).
2. Pre-pass, no gradient (:meth:`Learner.prepare`): for every kept seat-game, the policy's own forward over its
   committed rows gives ``logp_prox`` (π_prox recomputed at update start) and the θ_k distribution used by
   the gate-KL stop; the frozen reference's forward gives the KL anchor targets; the critic's frozen step
   encoder runs once (its ``z_step`` is cached for the critic epochs) and the pre-update critic gives V.
   The learner forward is the standard full-sequence forward over the seat-game's committed rows only: the
   shard holds committed rows only, and the committed-row forward equals the actor's ``peek``. Seat-games
   are never chunked (a burn-in window does not give the actor's context).
3. Per-tick GAE on the pre-update values (``gae.gae_ragged`` with the config's ``gae.gamma`` and
   ``gae.lambda_tick``, terminal reward only), the unified weights (``gae.row_weights`` with ``family=``;
   HT × seat × [not forced]) and one advantage normalisation over the whole update.
4. Guards before the first optimiser step. Every update: the actor's μ vs the learner's π_prox per row type
   (``mu_dlogp_max/<row type>``) within the update-0 tolerance and no executed action illegal for the learner
   (the shard digest covers the weights only, so a stale server or a served gate bias is caught here); the
   strict monitor's pre-pass keys present and finite; the monitor stops already decided by the pre-pass
   numbers (mask audit, refusals, drop rate) taken with no step. Update 0: the full identity — learner
   log π vs μ per row type, the training-mode |r − 1| of the first minibatch per row type, KL_h(θ_0 ‖ ref) per
   head (``losses.identity_check`` + ``Monitor.check_identity``). A failure raises :class:`IdentityError` with
   no step taken.
5. ``policy.epochs`` epochs over minibatches of ``policy.minibatch_seat_games`` whole seat-games:
   clipped surrogate on the joint executed-action log π (``losses.clipped_surrogate``) + β·KL anchor
   (``losses.kl_penalty``) + entropy bonus (0 by default) with the minibatch-wide surrogate denominator D across
   micro-batches; NaN/inf minibatches are skipped (``Monitor.observe_minibatch``); gradients clipped at the
   current clip (``phase_rules.grad_clip_raise`` raises it once); early stop on joint k3 / play k3 / gate KL
   over the epoch so far, measured before each step (``losses.early_stop``; the minibatch that trips is not
   stepped, as in Spinning Up), ends the update's epochs.
6. ``critic.epochs`` epochs of the critic alone (MSE of tanh V vs the λ-returns, its own Adam optimiser and
   gradient clip from ``critic.*``).
7. The monitor record (``losses.update_metrics`` keys, ``kl_ref_total``, per-phase EV against the terminal
   outcome, ``mask_recompute_mismatch``, drop rate, plus the caller's rollout keys, which may not overwrite a
   learner key) goes to ``Monitor.observe_update``.

Modules
-------
* Policy: the IL checkpoint's ``model_ema`` with ``dropout = 0`` and ``gate_bias = None`` (the PPO policy
  never carries the IL calibration bias). Full fine-tune by default; the value / value_priv / privileged and
  aux heads are not used by the policy loss and get ``requires_grad = False``; ``train_scope =
  freeze_step_encoder`` (the control arm) freezes the embeddings + step encoder with ``requires_grad = False``
  (never ``lr_mult = 0``, which still computes gradients). Optimiser: ``torch.optim.AdamW`` over
  ``model.param_groups`` (weight decay 0 = Adam).
* Reference: a frozen copy (eval mode, every parameter ``requires_grad = False``, ``gate_bias`` explicitly None).
  Until the first promotion it is the IL model itself; :meth:`Learner.set_reference` swaps it.
* Critic (:class:`Critic`): the IL model's embeddings + step encoder (frozen copy), a trainable copy of its
  temporal encoder and value head; separate Adam.

Checkpoints (:meth:`Learner.save`)
-----------------------------------
The il_v7 checkpoint layout, so ``il_v7.load_policy`` and an inference server load the acting weights: they are
saved under ``model_ema`` (and ``model_raw``), with ``model_config`` = the policy's config (dropout 0, gate_bias
None). The ``ppo`` section holds everything a chained block needs to continue: next update index, optimiser,
critic and its optimiser, reference weights, monitor state, grad-clip state, policy / reference digests, the
config digest and the PPO compatibility digest, which :meth:`Learner.from_checkpoint` compares.

Critic warm-up
--------------
:meth:`Learner.warmup_update` = :meth:`Learner.prepare_critic` (critic forward, GAE, value weights; no policy or
reference forward, no μ check, no identity) + :meth:`Learner.train_critic`, then :class:`FixedSetEV` (the
per-phase EV of the current critic on a fixed recorded set of ``critic.warmup.fixed_set_games`` games,
``gae.explained_variance_by_phase``) feeds ``phase_rules.critic_warmup_status``. :meth:`Learner.save` keeps the
warm-up state; ``Learner.from_il(..., critic_from=warm-up checkpoint)`` takes the warmed critic into the PPO run
after checking the compat digest, the acting weights / frozen step encoder and the pass.

Micro-batches: ``policy.micro_rows`` and ``policy.max_micro_tokens`` (the IL trainer's caps), split by il_v7's
rule without ever splitting a seat-game (:func:`split_micro`); the losses use the minibatch-wide denominators.
Both caps, the micro-batch counts and the peak memory (cuda max allocated, else peak RSS) are in every update
record.

Rules: the OpponentSpec temperature is the only tempering (``policy_params`` neutral or refused,
:func:`split_policy_params`); checkpoint files are pinned by sha256 (``opponents.checkpoint_sha256``,
:func:`verify_pin`, :func:`verify_shard_pins`); a mid-run config change goes through ``config.config_change``
(:meth:`Learner.change_config`, ``from_checkpoint(config_change=)``).

Diagnostics: every update reads back the lr of every param group and the beta every KL term used against the
plan (config x monitor factors, re-derived across update and gate actions from ``Monitor.scale_log``;
:class:`EffectiveValueError`), stops when the KL anchor's gradient vanishes while theta != reference, and logs
tiny-ratio shares, gate trends by opponent kind (with the game-clustered play-minus-wait advantage gap),
entropies versus the reference per head and per card, and argmax flip rates on a fixed probe set (saved with the
checkpoint). None of it changes a step (a test pins this).

Precision: the backbone runs in the dtype of the loaded weights (fp32 for a real checkpoint) unless
``policy.backbone_dtype = bf16``, which runs the policy / reference / critic forwards under bf16 autocast and
needs the per-row-type tolerances ``monitor.parity.dlogp_tol_bf16``: ``cfg.monitor_thresholds()`` refuses
bf16 until they are set. All log-probabilities, KL and losses are float64 (``ppo.logprob``).

Optional modules: the shard tokenizer (``amini.nn.tokenizer_v7``: collates the stored tokens), the client
action-mask rules (``amini.ppo.client_mask``) and the training phase rules (``amini.ppo.phase_rules``) are not
part of this repository. The learner imports without them; the code paths that call them raise a RuntimeError.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import heapq
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from amini.nn import schema_v1 as S
from amini.nn.aminiformer import AminiFormer, AminiFormerConfig, StepEncoder, take_rows
from amini.ppo import buffer as B
from amini.ppo import config as C
from amini.ppo import gae as G
from amini.ppo import logprob as LP
from amini.ppo import losses as L
from amini.ppo import monitor as M
from amini.train import il_v7 as IL

# Optional modules that are not part of this repository (module docstring): None when not installed; the code
# paths that call them raise a RuntimeError (:func:`_require`).
try:
    from amini.nn import tokenizer_v7 as TK
except ImportError:
    TK = None
try:
    from amini.ppo import client_mask as CM
except ImportError:
    CM = None
try:
    from amini.ppo import phase_rules as PR
except ImportError:
    PR = None

NO_TOKENIZER = "the shard tokenizer is not part of this repository"
NO_CLIENT_MASK = "client action-mask rules are not part of this repository"
NO_PHASE_RULES = "training phase rules are not part of this repository"


def _require(module: Any, message: str) -> Any:
    """``module`` (one of the optional imports above), or RuntimeError(``message``) when it is not installed."""
    if module is None:
        raise RuntimeError(message)
    return module


ROOT = Path(__file__).resolve().parents[3]
COMPILE_MAX_GRAPHS = 64
COMPILE_HALF_GRAD_RTOL = 0.05     # smoke test, per-tensor relative gradient error under an fp16 / bf16 backbone
LEARNER_VERSION = "ppo-v7-learner.1"
CKPT_FORMAT = "aminiformer-v1-ckpt.1"        # the il_v7 format (load_policy reads model_ema)
COMPAT_SCHEMA = "ppo-v7-compat.1"
# PPO compatibility digest: the files of this repository whose code changes what a stored row / token / action
# means (the shard tokenizer is not part of this repository)
COMPAT_FILES: Tuple[str, ...] = ("src/amini/nn/schema_v1.py",)
MAX_MICRO_ROWS = 256                             # default of policy.micro_rows (example value)
KL_MASK = {"kl_family": "mask_family", "kl_slot": "mask_slot", "kl_cell": "mask_cell",
           "kl_ability": "mask_ability"}
ENT_MASK = {"ent_family": "mask_family", "ent_slot": "mask_slot", "ent_cell": "mask_cell",
            "ent_ability": "mask_ability"}
OUT_ROW_KEYS = ("family", "slot", "ability", "ability_avail")      # row-aligned outputs logprob reads
# per-row arrays the argmax probe keeps besides slot / play: the inputs of the IL entropy baseline
PROBE_ROW_KEYS: Tuple[str, ...] = ("family", "cell", "ability", "play_blocked", "dt", "card")
# Keys of Monitor.expected_update_keys the learner produces itself; the rest come from the rollout and the run
# driver
LEARNER_KEYS: Tuple[str, ...] = ("clip_frac", "play_clip_frac", "kl_ref_total", "ev_overall", "ev_last_third",
                                 "early_stopped", "mask_recompute_mismatch", "drop_rate")
# Of those, the ones only known after the optimiser steps (the strict preflight uses placeholders for them)
POST_STEP_KEYS: Tuple[str, ...] = ("clip_frac", "play_clip_frac", "kl_ref_total")
# Diagnostic keys the steps produce (tiny ratios, argmax probe flips); expected by a strict monitor once their
# thresholds are set, so the preflight treats them as post-step keys too
POST_STEP_TRACK_B_KEYS: Tuple[str, ...] = ("tiny_ratio_share/play", "tiny_ratio_share/wait", "probe_flip/slot",
                                           "probe_flip/cell")
# Keys the rollout also produces: ``drop_rate`` must agree with the learner's; the learner's ``drop_by_reason`` is
# kept
SHARED_KEYS: Tuple[str, ...] = ("drop_rate", "drop_by_reason")
# Names the learner writes into the monitor record; ``extra_metrics`` may not set them
OWNED_KEYS: Tuple[str, ...] = tuple(k for k in LEARNER_KEYS if k not in SHARED_KEYS) + (
    "joint_k3", "play_k3", "gate_kl", "ratio_mean", "ratio_min", "ratio_max", "ratio_p01", "ratio_p99",
    "weight_sum", "play_weight_sum", "dual_clip_frac", "entropy", "update", "steps", "skipped_minibatches",
    "grad_clip", "grad_clipped_frac", "grad_norm_max", "lr", "beta", "backbone", "epochs_done",
    "policy_digest_before", "policy_digest_after", "learner_invalid_rows", "extra_ignored", "prestep_stop",
    "micro_rows", "micro_token_cap", "micro_batches", "micro_rows_max", "micro_tokens_max", "micro_over_token_cap",
    "peak_mem_bytes", "peak_mem_source", "kl_grad_check", "cell_entropy_by_card")
OWNED_PREFIXES: Tuple[str, ...] = ("entropy/", "kl_grad_ratio/", "last_epoch/", "mu_dlogp_max/", "critic_",
                                   # diagnostics (effective values, tiny ratios, gate trends, entropies, probe)
                                   "effective/", "tiny_ratio_", "gate_trend/", "entropy_theta/", "entropy_ref/",
                                   "entropy_vs_ref/", "cell_entropy_vs_ref/", "entropy_vs_il/",
                                   "cell_entropy_vs_il/", "probe_",
                                   # lag-1 overlap (ppo.overlap; written only when policy.rollout_lag = 1)
                                   "is/")
# Monitor stops whose inputs are all known after the pre-pass: evaluated before the first optimiser step
PRESTEP_STOP_MONITORS: Tuple[str, ...] = ("mask_illegal_cell", "mask_legal_refused", "mask_recompute", "drop_rate")


class IdentityError(AssertionError):
    """The update-0 identity failed (the run stops), or on a later update the actor's μ disagrees with the
    learner's π_prox beyond the update-0 tolerance (π_prox replaces μ only while actor == learner); no
    optimiser step was taken and the monitor is stopped."""


class EffectiveValueError(IdentityError):
    """Effective-value check: at the start of an update the lr / beta the monitor plans cannot be re-derived from
    the last read-back values and the logged factors, or an optimiser param group does not hold the planned lr. No
    step was taken and the monitor is stopped (an :class:`IdentityError`, so a caller's stop path catches it)."""


EFFECTIVE_RTOL = 1e-9                             # read-back vs planned lr / beta (products of logged factors)


def _close(a: float, b: float, rtol: float = EFFECTIVE_RTOL) -> bool:
    a, b = float(a), float(b)
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= rtol * max(abs(a), abs(b)) + 1e-300


# =========================================================================== digests
def state_dict_digest(sd: Mapping[str, torch.Tensor]) -> str:
    """sha256 over the sorted keys, dtypes, shapes and bytes of a state dict. ``policy_digest`` of a shard is
    this function of the served checkpoint's ``model_ema`` (the coordinator must use the same function)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8") + b"\0" + str(t.dtype).encode() + b"\0" + repr(tuple(t.shape)).encode() + b"\0")
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _file_sha(path: Path) -> str:
    return IL.file_sha256(str(path))


def compat_record(*, hazard_dt: Optional[str], tables_digest: Optional[str] = None,
                  root: Union[str, Path] = ROOT) -> Dict[str, Any]:
    """The PPO compatibility record: sha256 of the :data:`COMPAT_FILES` sources, ``S.describe()``,
    ``PLAY_LANDING_RULE``, the sampling constants (``S.U_TICKS``, ``S.KAPPA``), the resolved hazard_dt mode and the
    tokenizer-table digest. ``warm_start`` checks none of the code-side items."""
    root = Path(root)
    files = {}
    for rel in COMPAT_FILES:
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"compat_record: {p} is missing")
        files[rel] = _file_sha(p)
    sampling = {"U_TICKS": int(S.U_TICKS), "KAPPA": {k: float(v) for k, v in sorted(S.KAPPA.items())}}
    return {"schema": COMPAT_SCHEMA, "files": files, "schema_describe": S.describe(),
            "play_landing_rule": int(S.PLAY_LANDING_RULE), "sampling": sampling,
            "hazard_dt": hazard_dt, "tables_digest": tables_digest}


def compat_digest(rec: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of a :func:`compat_record`."""
    blob = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


# =========================================================================== policy_params rule
# The IL model config may carry ``AminiFormerConfig.policy_params`` (per-head temperature / greedy, gate_stream; read
# by its ``sample_actions(params=model)``). PPO pins ONE rule: the OpponentSpec temperature (``cases.OpponentSpec``,
# ``opponents.temperature``, applied by ``logprob.apply_temperature`` / ``RecordingPolicy``) is the only
# tempering. theta, the reference and the critic are built with policy_params at its neutral default (the key
# is dropped, or set to None where the config class has it); a checkpoint whose policy_params is not neutral is
# refused, unless it is a frozen opponent whose settings are converted to that OpponentSpec temperature
# (:func:`policy_params_temperature`, :func:`convert_policy_params`, :func:`opponent_model_config`). Works
# whether or not the key exists in the checkpoint config or in AminiFormerConfig.
POLICY_PARAM_HEADS: Tuple[str, ...] = ("slot", "cell", "ability")
GATE_STREAMS: Tuple[str, ...] = ("shared", "separate")
NEUTRAL_POLICY_PARAMS: Dict[str, Any] = {"temperature": {h: 1.0 for h in POLICY_PARAM_HEADS},
                                         "greedy": {h: False for h in POLICY_PARAM_HEADS}, "gate_stream": "shared"}
MAX_POLICY_TEMPERATURE = 100.0                   # aminiformer.MAX_TEMPERATURE


class PolicyParamsError(ValueError):
    """A checkpoint's ``policy_params`` is not neutral (or not convertible): in PPO the OpponentSpec temperature
    is the only tempering."""


def canonical_policy_params(pp: Any) -> Dict[str, Any]:
    """Canonical dict of a ``policy_params`` value (as the model's ``PolicyParams.from_dict(...).to_dict()``:
    None = defaults; ``temperature`` / ``greedy`` one value or a dict over a subset of the heads; unknown keys,
    lists, temperatures outside (0, 100] and non-bool greedy flags are refused)."""
    if pp is None:
        return copy.deepcopy(NEUTRAL_POLICY_PARAMS)
    if not isinstance(pp, Mapping):
        raise PolicyParamsError(f"policy_params must be a mapping or None, not {type(pp).__name__}")
    unknown = sorted(set(pp) - {"temperature", "greedy", "gate_stream"})
    if unknown:
        raise PolicyParamsError(f"unknown policy_params keys {unknown}")

    def per_head(v: Any, default: Any, name: str) -> Dict[str, Any]:
        if v is None:
            return {h: default for h in POLICY_PARAM_HEADS}
        if isinstance(v, Mapping):
            bad = sorted(set(v) - set(POLICY_PARAM_HEADS))
            if bad:
                raise PolicyParamsError(f"unknown policy_params {name} heads {bad}")
            return {h: v.get(h, default) for h in POLICY_PARAM_HEADS}
        if isinstance(v, (list, tuple)):
            raise PolicyParamsError(f"policy_params {name}: one value or a mapping over {POLICY_PARAM_HEADS}")
        return {h: v for h in POLICY_PARAM_HEADS}

    temp = per_head(pp.get("temperature"), 1.0, "temperature")
    for h, t in temp.items():
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating)) \
                or not (math.isfinite(float(t)) and 0.0 < float(t) <= MAX_POLICY_TEMPERATURE):
            raise PolicyParamsError(f"policy_params temperature.{h} must be in (0, {MAX_POLICY_TEMPERATURE}], got {t!r}")
        temp[h] = float(t)
    greedy = per_head(pp.get("greedy"), False, "greedy")
    for h, g in greedy.items():
        if not isinstance(g, (bool, np.bool_)):
            raise PolicyParamsError(f"policy_params greedy.{h} must be true or false, got {g!r}")
        greedy[h] = bool(g)
    gs = pp.get("gate_stream", "shared")
    if gs not in GATE_STREAMS:
        raise PolicyParamsError(f"policy_params gate_stream must be one of {GATE_STREAMS}, got {gs!r}")
    return {"temperature": temp, "greedy": greedy, "gate_stream": gs}


def is_neutral_policy_params(pp: Any) -> bool:
    """True for an absent / None / all-default ``policy_params`` (temperature 1, no greedy, shared stream)."""
    return canonical_policy_params(pp) == NEUTRAL_POLICY_PARAMS


def policy_params_temperature(pp: Any) -> float:
    """The ``cases.OpponentSpec`` temperature equivalent to a ``policy_params`` value: all three heads greedy ->
    0.0 (greedy, the gate still sampled at T = 1); no greedy head and one common temperature in
    [logprob.T_MIN, 1] -> that temperature; neutral -> 1.0. Per-head greedy or temperatures, T > 1 and
    ``gate_stream = separate`` have no OpponentSpec form and raise :class:`PolicyParamsError`."""
    c = canonical_policy_params(pp)
    if c["gate_stream"] != "shared":
        raise PolicyParamsError("policy_params gate_stream 'separate' has no OpponentSpec equivalent")
    g = [c["greedy"][h] for h in POLICY_PARAM_HEADS]
    t = [c["temperature"][h] for h in POLICY_PARAM_HEADS]
    if all(g):
        return 0.0
    if any(g):
        raise PolicyParamsError(f"policy_params greedy on some heads only ({c['greedy']}) has no OpponentSpec form")
    if len(set(t)) != 1:
        raise PolicyParamsError(f"policy_params per-head temperatures {c['temperature']} have no OpponentSpec form")
    if not LP.T_MIN <= t[0] <= 1.0:
        raise PolicyParamsError(f"policy_params temperature {t[0]} outside the OpponentSpec range [{LP.T_MIN}, 1]")
    return float(t[0])


def split_policy_params(model_config: Mapping[str, Any], *, convert: bool = False
                        ) -> Tuple[Dict[str, Any], Optional[float]]:
    """``(model_config without policy_params, converted temperature or None)``. A non-neutral value raises
    :class:`PolicyParamsError` unless ``convert`` (frozen opponents only), which returns its
    :func:`policy_params_temperature`."""
    d = dict(model_config)
    pp = d.pop("policy_params", None)
    if is_neutral_policy_params(pp):
        return d, None
    if not convert:
        raise PolicyParamsError(
            f"checkpoint model_config has non-default policy_params {canonical_policy_params(pp)}: in PPO the "
            "OpponentSpec temperature is the only tempering, so theta, the reference and the critic need the "
            "neutral default. For a frozen opponent convert it (learner.convert_policy_params writes a copy "
            "without it and returns the temperature for opponents.temperature); for theta use the IL weights "
            "without policy_params")
    return d, policy_params_temperature(pp)


def _aminiformer_config(d: Mapping[str, Any]) -> AminiFormerConfig:
    """AminiFormerConfig from a dict whose policy_params was removed: set to None where the class knows the field,
    absent otherwise."""
    d = dict(d)
    if "policy_params" in {f.name for f in dataclasses.fields(AminiFormerConfig)}:
        d["policy_params"] = None
    return AminiFormerConfig.from_dict(d)


def check_model_policy_params(model: nn.Module, what: str) -> None:
    """Refuse a built model whose ``cfg.policy_params`` is not neutral."""
    pp = getattr(getattr(model, "cfg", None), "policy_params", None)
    if not is_neutral_policy_params(pp):
        raise PolicyParamsError(f"the {what} carries non-default policy_params {canonical_policy_params(pp)}; "
                                "PPO tempers only through the OpponentSpec temperature")


def opponent_model_config(model_config: Mapping[str, Any], temperature: Optional[float]) -> Dict[str, Any]:
    """The model config a frozen opponent's server loads (policy_params removed, so its tempering cannot stack
    with the seat's): a non-neutral policy_params is accepted only when its :func:`policy_params_temperature`
    equals the opponent's OpponentSpec ``temperature`` (the conversion recorded in the run config)."""
    d, t = split_policy_params(model_config, convert=True)
    if t is not None and (temperature is None or abs(float(temperature) - t) > 1e-12):
        raise PolicyParamsError(f"opponent checkpoint policy_params is temperature {t}, but its OpponentSpec "
                                f"temperature is {temperature!r}: set opponents.temperature to {t} or convert it")
    return d


def convert_policy_params(ckpt_in: Union[str, Path], ckpt_out: Union[str, Path]) -> Dict[str, Any]:
    """Write a copy of a checkpoint without ``model_config.policy_params`` (weights untouched) and return
    ``{'temperature', 'policy_params', 'sha256', 'path'}``: the OpponentSpec temperature to put into
    ``opponents.temperature`` and the new file's sha256 for ``opponents.checkpoint_sha256``. Refuses to
    overwrite its input and a policy_params with no OpponentSpec form."""
    if os.path.realpath(str(ckpt_in)) == os.path.realpath(str(ckpt_out)):
        raise ValueError("convert_policy_params writes a copy: ckpt_out must differ from ckpt_in")
    ck = torch.load(str(ckpt_in), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_config" not in ck:
        raise ValueError(f"{ckpt_in}: not a checkpoint with model_config")
    pp = dict(ck["model_config"]).get("policy_params")
    t = policy_params_temperature(pp)
    ck["model_config"] = split_policy_params(ck["model_config"], convert=True)[0]
    ck.setdefault("history_policy_params", []).append({"converted_to_opponent_temperature": t,
                                                      "policy_params": canonical_policy_params(pp)})
    out = Path(ckpt_out)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(ck, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    return {"temperature": t, "policy_params": canonical_policy_params(pp), "sha256": IL.file_sha256(str(out)),
            "path": str(out.resolve())}


# =========================================================================== checkpoint pins
class PinError(ValueError):
    """A checkpoint file or a shard's recorded opponent checkpoint does not match its pinned sha256."""


def verify_pin(path: Union[str, Path], sha256: Optional[str], *, what: str = "checkpoint") -> str:
    """sha256 of ``path``; raises :class:`PinError` when ``sha256`` (the config pin) is given and differs."""
    got = IL.file_sha256(str(path))
    if sha256 is not None and got != sha256:
        raise PinError(f"{what} {path}: sha256 {got[:12]} != pinned {str(sha256)[:12]} "
                       "(opponents.checkpoint_sha256; was the file rewritten after pinning? pin a copy)")
    return got


def verify_pinned_checkpoints(cfg: Any, models: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """:func:`verify_pin` on every configured checkpoint (``cfg.checkpoint_pins()``) of ``models`` (default all
    with a path). A path without a pin is reported, not refused (``cfg.missing`` lists the pin)."""
    out: Dict[str, str] = {}
    for model, pin in cfg.checkpoint_pins().items():
        if (models is not None and model not in models) or pin["path"] is None:
            continue
        out[model] = verify_pin(pin["path"], pin["sha256"], what=f"opponents.checkpoint.{model}")
    return out


def verify_shard_pins(data: B.UpdateData, cfg: Any, *, snapshot_sha256: Optional[Mapping[str, str]] = None,
                      require: bool = False) -> Dict[str, int]:
    """Check the frozen-opponent checkpoint digests the rollout records per game in ``GameMeta.opp_settings
    ['checkpoint_sha256']`` (kept and dropped games) against the config pins (``cfg.opponent_pins()``) and,
    for snapshots, ``snapshot_sha256`` keyed by the snapshot's server name (``opp_settings['server']``).
    A recorded digest that differs raises :class:`PinError`; with ``require`` a frozen-opponent game without a
    recorded digest, or with no pin to compare, raises too. Returns counts (verified / unrecorded / unpinned)."""
    pins = cfg.opponent_pins()
    snaps = dict(snapshot_sha256 or {})
    counts = {"verified": 0, "unrecorded": 0, "unpinned": 0}
    for g in list(data.games) + list(data.dropped):
        kind = str(g.opp_kind)
        if kind in ("self_seat", "mirror"):
            continue
        st = dict(g.opp_settings or {})
        got = st.get("checkpoint_sha256")
        want = snaps.get(str(st.get("server") or st.get("name"))) if kind == "snapshot" else pins.get(kind)
        if got is None:
            if require:
                raise PinError(f"game {g.game_id} ({kind}): no checkpoint_sha256 in opp_settings")
            counts["unrecorded"] += 1
            continue
        if want is None:
            if require:
                raise PinError(f"game {g.game_id} ({kind}): recorded {str(got)[:12]} but no pin to compare")
            counts["unpinned"] += 1
            continue
        if got != want:
            raise PinError(f"game {g.game_id} ({kind}): opponent checkpoint {str(got)[:12]} != pinned {str(want)[:12]}")
        counts["verified"] += 1
    return counts


# =========================================================================== micro-batches and memory
def item_tokens(example: Mapping[str, Any]) -> Tuple[int, int]:
    """Largest ENTITY / OBJECT count over a seat-game's rows (``il_v7.micro_cost`` inputs)."""
    g = example["tokens"]["groups"]
    return tuple(int(np.asarray(g[name]["valid"]).sum(-1).max(initial=0)) for name in ("ENTITY", "OBJECT"))  # type: ignore[return-value]


def _micro_fits(b: Sequence[Any], micro_rows: int, max_tokens: int) -> bool:
    if len(b) <= 1:
        return True                                   # one seat-game always runs (never split)
    rows = sum(int(it.n) for it in b)
    if rows > micro_rows:
        return False
    if not max_tokens:
        return True
    toks = [getattr(it, "tok", (0, 0)) for it in b]
    return IL.micro_cost(rows, max(t[0] for t in toks), max(t[1] for t in toks)) <= max_tokens


def split_micro(items: Sequence[Any], micro_rows: int, max_tokens: int = 0) -> List[List[Any]]:
    """Micro-batches of whole seat-games (``.n`` rows, ``.temperature``, optional ``.tok``), one sampling
    temperature each, by ``il_v7.EpisodeBatches._split``'s rule within each temperature: the fewest
    micro-batches that keep at most ``micro_rows`` rows and ``il_v7.micro_cost`` <= ``max_tokens`` (0 = off),
    balanced by rows (longest seat-game first into the lightest bin), seat-game order kept inside a bin. A
    seat-game above a cap is its own micro-batch (and counts as one whole bin in the starting bin count, the one
    difference from il_v7, whose chunks never exceed the row cap). The losses use the minibatch-wide denominators, so the split
    never changes the objective. Deterministic."""
    groups: Dict[Tuple[float, float, float], List[Any]] = {}
    for it in items:
        groups.setdefault(tuple(it.temperature), []).append(it)
    out: List[List[Any]] = []
    for grp in groups.values():
        n = [int(it.n) for it in grp]
        by_size = sorted(range(len(grp)), key=lambda i: (-n[i], i))
        big = [x for x in n if x > micro_rows]          # an oversized seat-game needs its own bin
        m = max(1, len(big) + -(-(sum(n) - sum(big)) // int(micro_rows)))
        while True:
            heap = [(0, k) for k in range(m)]
            bins: List[List[int]] = [[] for _ in range(m)]
            for i in by_size:
                load, k = heapq.heappop(heap)
                bins[k].append(i)
                heapq.heappush(heap, (load + n[i], k))
            got = sorted((sorted(b) for b in bins if b), key=lambda b: b[0])
            if m >= len(grp) or all(_micro_fits([grp[i] for i in b], micro_rows, max_tokens) for b in got):
                out.extend([grp[i] for i in b] for b in got)
                break
            m += 1
    return out


def peak_memory(device: Any) -> Tuple[Optional[int], str]:
    """Peak memory in bytes and its source: ``torch.cuda.max_memory_allocated`` on a cuda device (reset at
    the start of each update), else the process peak RSS (``resource.getrusage``; never reset)."""
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(dev)), "cuda_max_memory_allocated"
    try:
        import resource
        r = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):       # pragma: no cover - platform without resource
        return None, "unavailable"
    return (r if sys.platform == "darwin" else r * 1024), "process_peak_rss"


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


# =========================================================================== model construction
def ppo_config(model_config: Mapping[str, Any]) -> AminiFormerConfig:
    """The PPO model config of an IL checkpoint: dropout 0 (on-policy), gate_bias None and policy_params
    neutral (refused otherwise, :func:`split_policy_params`)."""
    d, _ = split_policy_params(model_config)
    d["dropout"] = 0.0
    d["gate_bias"] = None
    return _aminiformer_config(d)


def model_from_state(model_config: Mapping[str, Any], state: Mapping[str, torch.Tensor], *,
                     device: Any = "cpu", dtype: Optional[torch.dtype] = None) -> AminiFormer:
    """A model with :func:`ppo_config` and ``state`` loaded strictly. ``dtype`` defaults to the dtype of the
    state's floating tensors (an fp64 test checkpoint stays fp64; a real one is fp32)."""
    model = AminiFormer(ppo_config(model_config))
    if dtype is None:
        dtype = next((v.dtype for v in state.values() if torch.is_tensor(v) and v.is_floating_point()),
                     torch.float32)
    model = model.to(dtype)
    model.load_state_dict(dict(state), strict=True)
    return model.to(device)


def load_source_policy(path: Union[str, Path], *, device: Any = "cpu", dtype: Optional[torch.dtype] = None
                       ) -> Tuple[AminiFormer, Dict[str, Any]]:
    """The IL checkpoint's shipped weights (``model_ema``, exactly what ``il_v7.load_policy`` reads) as a PPO
    model (:func:`ppo_config`), plus a source record (path, sha256, step, generation, the dropped gate_bias)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or not isinstance(ck.get("model_ema"), dict) or "model_config" not in ck:
        raise ValueError(f"{path}: not an il_v7 checkpoint with model_ema and model_config")
    model = model_from_state(ck["model_config"], ck["model_ema"], device=device, dtype=dtype)
    rec = {"path": str(Path(path).resolve()), "sha256": IL.file_sha256(str(path)), "step": ck.get("step"),
           "generation": ck.get("generation"), "trainer": ck.get("trainer"), "schema": ck.get("schema"),
           "source_gate_bias": dict(ck["model_config"]).get("gate_bias"),
           "source_dropout": dict(ck["model_config"]).get("dropout"), "tables": ck.get("tables"),
           "source_policy_params": dict(ck["model_config"]).get("policy_params")}
    return model, rec


def freeze(module: nn.Module) -> nn.Module:
    """requires_grad False on every parameter and eval mode (frozen means no gradient at all)."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()


def module_copy(module: nn.Module) -> nn.Module:
    """``copy.deepcopy(module)`` without instance-level ``forward`` overrides.

    The optional learner compile installs ``temporal.forward = torch.compile(temporal.forward)`` in the instance
    ``__dict__``; ``deepcopy`` treats that function as atomic, so the copy's ``forward`` would stay bound to the
    SOURCE module and run with the source's weights. Dropping the override gives the copy its class ``forward``
    back; nothing is removed when no module was compiled."""
    out = copy.deepcopy(module)
    for sub in out.modules():
        sub.__dict__.pop("forward", None)
    return out


def make_reference(policy: AminiFormer) -> AminiFormer:
    """Frozen KL reference: a deep copy (:func:`module_copy`) with ``gate_bias`` explicitly None, eval mode."""
    ref = module_copy(policy)
    ref.cfg = dataclasses.replace(ref.cfg, gate_bias=None)
    return freeze(ref)


def policy_unused_prefixes() -> Tuple[str, ...]:
    """Parameter prefixes the policy loss never reaches (value, privileged critic, aux heads)."""
    hp = AminiFormer.HEAD_PREFIXES
    return tuple(hp["value"]) + tuple(hp["aux"])


def set_policy_trainable(policy: AminiFormer, scope: str = "full") -> None:
    """``full`` trains every module the policy uses; ``freeze_step_encoder`` (control arm) also
    freezes the embeddings + step encoder. Unused heads are frozen in both (requires_grad False)."""
    if scope not in ("full", "freeze_step_encoder"):
        raise ValueError(f"train_scope must be full | freeze_step_encoder, not {scope!r}")
    unused = policy_unused_prefixes()
    for name, p in policy.named_parameters():
        frozen = name.startswith(unused) or (scope == "freeze_step_encoder" and name.startswith("step_encoder."))
        p.requires_grad_(not frozen)


class Critic(nn.Module):
    """Separate value network: frozen copy of the IL embeddings + step encoder, trainable copies of the
    temporal encoder and the value head. Output = tanh value per valid row, as ``AminiFormer.value_heads``."""

    def __init__(self, source: AminiFormer) -> None:
        super().__init__()
        self.cfg = source.cfg
        self.step_encoder = freeze(copy.deepcopy(source.step_encoder))
        self.temporal = copy.deepcopy(source.temporal)
        self.value_head = copy.deepcopy(source.value_head)
        for mod in (self.temporal, self.value_head):
            for p in mod.parameters():
                p.requires_grad_(True)
        # Kept outside Module._modules so checkpoint/state_dict keys stay identical
        # when learner compilation is enabled.
        self.__dict__["_ppo_compiled_temporal"] = None

    def train(self, mode: bool = True) -> "Critic":
        super().train(mode)
        self.step_encoder.eval()                      # frozen part never switches mode
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def set_compiled_temporal(self, fn: Any) -> None:
        """Install a compiled temporal callable without registering a second module."""
        self.__dict__["_ppo_compiled_temporal"] = fn

    @torch.no_grad()
    def step_features(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """``z_step [N, d]`` of the valid rows (the frozen part; computed once per update and cached). With the
        batch's host index (:func:`host_index_of`) the rows and the token layout are taken as given, as in
        ``AminiFormer.forward`` (no ``x[bool_mask]`` / ``nonzero`` / ``.item()`` on the device)."""
        hx = batch_host_index(batch)
        if hx is not None:
            return self.step_encoder(AminiFormer.flatten_groups(batch["groups"], hx["rv"]), hx["step"]).z_step.detach()
        rv = batch["row"]["row_valid"].bool()
        return self.step_encoder(AminiFormer.flatten_groups(batch["groups"], rv)).z_step.detach()

    def values_from_steps(self, z_step: torch.Tensor, row_valid: torch.Tensor, pos: torch.Tensor,
                          tq: torch.Tensor, sel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tanh V over the valid rows from cached ``z_step`` (same arithmetic as ``AminiFormer.forward``).
        ``sel``: the int64 flat row-major index of the valid rows (the host index's ``rv``); the same rows in the
        same order as ``row_valid``, written and read with ``index_copy`` / ``index_select`` (no device sync)."""
        rv = row_valid.bool()
        b, t = rv.shape
        if sel is None:
            zs = z_step.new_zeros(b, t, z_step.shape[-1])
            zs[rv] = z_step
        else:
            zs = z_step.new_zeros(b * t, z_step.shape[-1]).index_copy(0, sel, z_step).view(b, t, z_step.shape[-1])
        p = pos.to(torch.float64 if zs.dtype == torch.float64 else torch.float32)
        temporal = self.__dict__.get("_ppo_compiled_temporal")
        if temporal is None:
            temporal = self.temporal
        zt = temporal(zs, p, tq.long(), rv)
        z = zt[rv] if sel is None else take_rows(zt, sel)
        v = self.value_head(z)
        if v.dtype in (torch.float16, torch.bfloat16):
            v = v.float()
        return torch.tanh(v.squeeze(-1))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        row = batch["row"]
        return self.values_from_steps(self.step_features(batch), row["row_valid"], row["pos"], row["tq"],
                                      sel=batch_row_sel(batch))


# =========================================================================== per-seat data
@dataclass
class SeatItem:
    """One kept trained seat-game of the update, with everything the epochs need on the learner device."""
    index: int
    sg: B.SeatGame
    held: Optional[Dict[str, Any]]                # the padded tokens when they are kept (see :attr:`example`)
    n: int
    temperature: Tuple[float, float, float]
    action: Dict[str, torch.Tensor]
    dt: torch.Tensor
    mu: torch.Tensor
    row_type: torch.Tensor
    play: torch.Tensor
    card: np.ndarray
    # filled by Learner.prepare
    logp_prox: Optional[torch.Tensor] = None
    prox_out: Optional[Dict[str, torch.Tensor]] = None
    ref_out: Optional[Dict[str, torch.Tensor]] = None
    critic_z: Optional[torch.Tensor] = None
    value_old: Optional[np.ndarray] = None
    valid: Optional[torch.Tensor] = None
    w_policy: Optional[torch.Tensor] = None
    w_value: Optional[torch.Tensor] = None
    adv: Optional[torch.Tensor] = None
    ret: Optional[torch.Tensor] = None
    wp_sum: float = 0.0
    wv_sum: float = 0.0
    tok: Tuple[int, int] = (0, 0)                 # largest ENTITY / OBJECT count (micro-batch token cap)
    client_ok: Optional[np.ndarray] = None        # [n, 4] client slot-mask bits (None: the rows' / all True)
    # lag-1 overlap only (None otherwise): rho = min(rho_bar, pi_prox / mu) per row (the surrogate's IS weight,
    # set at lag 1 only), and the per-row numbers the is/ metrics read (effective mu, per-head prox / mu terms)
    is_w: Optional[torch.Tensor] = None
    ov_mu: Optional[np.ndarray] = None
    ov_head_prox: Optional[np.ndarray] = None
    ov_head_mu: Optional[np.ndarray] = None
    # host memory: the two row arrays the critic epochs read (``_row_batch``), so that they never need tokens
    row_pos: Optional[np.ndarray] = None
    row_tq: Optional[np.ndarray] = None
    # host index: the executed play rows and their slot (-1: no play) as host arrays, i.e. what
    # ``Learner._cell_args`` puts on the device as ``cell_rows`` / ``play_slot`` (None: an item without actions)
    play_host: Optional[np.ndarray] = None
    slot_host: Optional[np.ndarray] = None

    @property
    def example(self) -> Dict[str, Any]:
        """``{'tokens', 'labels'}`` for ``TK.collate``: the kept copy, or (``Learner.hold_examples`` False)
        rebuilt from the shard's flat arrays each time it is read -- the same arrays bit for bit
        (``SeatGame.example`` is a pure function of the shard). The padded copy of a seat-game is larger than the
        flat arrays that are held anyway, and rebuilding it is cheap."""
        return self.held if self.held is not None else self.sg.example()


def _client_ok(it: SeatItem) -> np.ndarray:
    """The client slot-mask bits of an item's rows: its own copy, else the shard rows' ``client_slot_ok``, else
    all True (a probe recorded before the mask existed)."""
    if it.client_ok is not None:
        return np.asarray(it.client_ok, dtype=bool)
    if it.sg is not None:
        return np.asarray(it.sg.rows["client_slot_ok"], dtype=bool)
    return np.ones((int(it.n), S.HAND_SLOTS), bool)


@dataclass
class OverlapCtx:
    """Lag-1 overlap context of one update (built only in a ``policy.rollout_lag = 1`` run or by the bit-identity
    test seam, :meth:`Learner._update_impl`): the lag of the data, the learner's own digest at update start, the
    parent (θ_{u-2}, the acting weights of a lag-1 round) and its digest, the test override ``mu_override =
    'prox'`` (μ := π_prox, so w = ρ = c = 1 exactly) and the caps ρ̄ / c̄ (defaults: the config defaults of
    ``policy.is_ratio_cap_lag1`` / ``policy.gae_trace_cap_lag1``, example values)."""
    lag: int
    own_digest: str
    parent: Optional[AminiFormer] = None
    parent_digest: Optional[str] = None
    mu_override: Optional[str] = None
    rho_bar: float = 1.5
    c_bar: float = 1.0


@dataclass
class PreparedUpdate:
    """Result of :meth:`Learner.prepare`: per-seat items plus the update-level numbers."""
    data: B.UpdateData
    items: List[SeatItem]
    adv_mean: float
    adv_std: float
    metrics: Dict[str, Any]
    identity: Dict[str, Any]


@dataclass
class MinibatchStats:
    """One minibatch's accumulated gradients (not yet stepped) and its detached per-row numbers."""
    denom: float
    loss: float
    finite: bool
    rows: Dict[str, Any]
    kl_sum: float
    kl_grad_ratio: Dict[str, float] = field(default_factory=dict)
    beta_used: Dict[str, float] = field(default_factory=dict)     # effective values: the beta every KL head was given


@dataclass
class UpdateResult:
    """What :meth:`Learner.update` returns. ``metrics`` is the record passed to ``Monitor.observe_update``."""
    update: int
    metrics: Dict[str, Any]
    actions: List[M.Action]
    identity: Optional[Dict[str, Any]]
    steps: int
    skipped_minibatches: int
    early_stopped: Tuple[str, ...]
    stopped: bool


# =========================================================================== batch helpers
def _to_torch(batch: Any, device: torch.device, fdtype: torch.dtype) -> Any:
    """Collated numpy batch -> torch on ``device``; floating arrays in the model dtype (fp64 tests stay fp64)."""
    if isinstance(batch, dict):
        return {k: _to_torch(v, device, fdtype) for k, v in batch.items()}
    if isinstance(batch, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(batch))
        if t.is_floating_point():
            t = t.to(fdtype)
        return t.to(device)
    return batch


HOST_INDEX_KEY, HOST_ROWS_KEY = "host_index", "host_index_rows"


def host_index_of(c: Mapping[str, Any], items: Sequence[SeatItem]) -> Dict[str, Any]:
    """The model's ``host_index`` of a collated (numpy) micro-batch, built on the host.

    ``AminiFormer.forward`` otherwise derives these integers on the device, and each derivation waits for the GPU
    (``x[bool_mask]``, ``nonzero``, ``.item()``): ``rv`` (flat row-major index of the valid rows), ``step``
    (``StepEncoder.host_index``: the packed token layout), and, when every item carries its executed plays
    (``SeatItem.play_host``), ``ridx`` / ``pidx_local`` / ``pidx`` (the stage-S / stage-P rows of
    ``Learner._cell_args``'s ``cell_rows`` / ``play_slot``, by the model's own rule). The IL trainer builds the
    same record in its loader (``il_v7.host_index``, from its labels). Same ops on the same booleans and ints as on
    the device, so the same indices; the forward then returns the same numbers (checked once per learner,
    ``Learner._host_index_decide``)."""
    rv = torch.from_numpy(np.ascontiguousarray(c["row"]["row_valid"])).bool()
    hx: Dict[str, Any] = {
        "rv": rv.reshape(-1).nonzero(as_tuple=True)[0],
        "step": StepEncoder.host_index({n: torch.from_numpy(np.ascontiguousarray(g["valid"]))[rv]
                                        for n, g in c["groups"].items()})}
    if all(getattr(it, "play_host", None) is not None for it in items):
        bsz, tm = rv.shape
        cell_rows = torch.zeros(bsz, tm, dtype=torch.bool)
        play_slot = torch.full((bsz, tm), -1, dtype=torch.long)
        for bi, it in enumerate(items):
            cell_rows[bi, :it.n] = torch.from_numpy(it.play_host)
            play_slot[bi, :it.n] = torch.from_numpy(it.slot_host)
        ridx = cell_rows[rv].nonzero(as_tuple=True)[0]
        pidx_local = (play_slot[rv][ridx] >= 0).nonzero(as_tuple=True)[0]
        hx.update(ridx=ridx, pidx=ridx[pidx_local], pidx_local=pidx_local)
    return hx


def _hx_to(o: Any, device: torch.device) -> Any:
    if isinstance(o, dict):
        return {k: _hx_to(v, device) for k, v in o.items()}
    return o.to(device) if isinstance(o, torch.Tensor) else o


def batch_host_index(batch: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The batch's host index for the row / token layout: the full one (the policy forward reads it) or the rows-only
    one of a batch whose items carry no actions (the critic's fixed set)."""
    hx = batch.get(HOST_INDEX_KEY)
    return hx if hx is not None else batch.get(HOST_ROWS_KEY)


def batch_row_sel(batch: Mapping[str, Any]) -> Optional[torch.Tensor]:
    hx = batch_host_index(batch)
    return None if hx is None else hx["rv"]


def _cat(items: Sequence[SeatItem], attr: str) -> torch.Tensor:
    return torch.cat([getattr(it, attr) for it in items])


def _cat_action(items: Sequence[SeatItem]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([it.action[k] for it in items]) for k in items[0].action}


def _split_out(out: Mapping[str, torch.Tensor], lengths: Sequence[int]) -> List[Dict[str, torch.Tensor]]:
    """Per-seat pieces (detached) of a forward output over consecutive seat-games: the row-aligned keys and the
    play-row cells with seat-local ``cell_play_index``."""
    idx = out.get("cell_play_index")
    cells = out.get("cells")
    pieces, off = [], 0
    for n in lengths:
        p = {k: out[k][off:off + n].detach() for k in OUT_ROW_KEYS}
        if idx is not None and cells is not None:
            sel = (idx >= off) & (idx < off + n)
            p["cells"] = cells[sel].detach()
            p["cell_play_index"] = (idx[sel] - off).detach()
        pieces.append(p)
        off += n
    return pieces


def _join_out(pieces: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`_split_out` for any sequence of seat pieces (row offsets re-applied)."""
    out = {k: torch.cat([p[k] for p in pieces]) for k in OUT_ROW_KEYS}
    cells, idx, off = [], [], 0
    for p in pieces:
        if "cells" in p and p["cells"].shape[0]:
            cells.append(p["cells"])
            idx.append(p["cell_play_index"] + off)
        off += int(p["family"].shape[0])
    if cells:
        out["cells"] = torch.cat(cells)
        out["cell_play_index"] = torch.cat(idx)
    return out


def _head_rows(per_row: Mapping[str, torch.Tensor], w: torch.Tensor, names: Mapping[str, str]
               ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """{head: (per-row value, unified weight on the head's own rows)} (the heads' row masks)."""
    return {k: (per_row[k], w * per_row[m].to(w.dtype)) for k, m in names.items() if k in per_row}


def _add_grads(acc: Dict[str, List[Optional[torch.Tensor]]], key: str, loss: torch.Tensor,
               params: Sequence[torch.Tensor]) -> None:
    """Add d ``loss`` / d ``params`` (graph kept) into ``acc[key]`` (one tensor per parameter, None = zero)."""
    cur = acc.setdefault(key, [None] * len(params))
    if not loss.requires_grad:
        return
    for i, g in enumerate(torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)):
        if g is not None:
            cur[i] = g.detach().clone() if cur[i] is None else cur[i].add_(g.detach())


def _grad_ratios(acc: Mapping[str, Sequence[Optional[torch.Tensor]]], surr_key: str) -> Dict[str, float]:
    """``losses.kl_grad_ratio`` on accumulated gradients: per KL head ||g_h|| / ||g_surrogate|| (``inf`` when
    only the surrogate's is zero, 0 when both are), plus ``total``: the norm of the whole KL anchor's gradient
    (the per-head gradients summed) over the surrogate's, the number the KL-gradient guard reads."""
    def norm(gs: Sequence[Optional[torch.Tensor]]) -> float:
        return math.sqrt(sum(float(g.double().pow(2).sum()) for g in gs if g is not None))

    def ratio(g_h: float, g_s: float) -> float:
        return g_h / g_s if g_s > 0 else (math.inf if g_h > 0 else 0.0)
    g_s = norm(acc.get(surr_key, ()))
    out: Dict[str, float] = {}
    total: List[Optional[torch.Tensor]] = []
    for h, gs in acc.items():
        if h == surr_key:
            continue
        out[h] = ratio(norm(gs), g_s)
        if not total:
            total = [None] * len(gs)
        for i, g in enumerate(gs):
            if g is not None:
                total[i] = g.double().clone() if total[i] is None else total[i] + g.double()
    if len(out):
        out["total"] = ratio(norm(total), g_s)
    return out


def _finite(x: Any) -> Optional[float]:
    """``float(x)`` when ``x`` is a finite number (not a bool or string), else None (as the monitor reads it)."""
    if x is None or isinstance(x, (bool, str, bytes)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _baselines_dict(b: M.Baselines) -> Dict[str, Any]:
    """Plain-dict form of the monitor baselines (``Baselines.from_mapping`` restores it; saved so a chained
    block keeps its behaviour / core-card rules)."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(b):
        v = getattr(b, f.name)
        out[f.name] = dict(v) if isinstance(v, Mapping) else list(v) if isinstance(v, tuple) else v
    return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.tolist()
    return x


def ev_by_phase(sgs: Sequence[B.SeatGame], values: np.ndarray, weights: Optional[np.ndarray] = None
                ) -> Dict[str, Any]:
    """``gae.explained_variance_by_phase`` of the seat-games' rows (concatenated in order): EV against each
    row's terminal outcome from its seat, phases by tick / that game's end tick, groups = opponent kind,
    ``weights`` default the unified value weights (HT x seat, forced rows included)."""
    lengths = np.array([sg.n for sg in sgs], np.int64)
    ticks = np.concatenate([sg.rows["tick"] for sg in sgs]).astype(np.int64)
    if weights is None:
        weights = np.concatenate([G.row_weights(sg.rows["stratum"], sg.rows["forced"], sg.weight,
                                                family=sg.rows["family"]).value for sg in sgs])
    outcomes = np.repeat([sg.outcome for sg in sgs], lengths)
    end_ticks = np.repeat([int(sg.game.end_tick) for sg in sgs], lengths)
    groups = np.repeat([str(sg.game.opp_kind) for sg in sgs], lengths)
    return G.explained_variance_by_phase(np.asarray(values, np.float64), outcomes, ticks, end_ticks,
                                         weights=weights, groups=groups)


def calibration_by_phase(sgs: Sequence[B.SeatGame], values: np.ndarray, weights: Optional[np.ndarray] = None,
                         bins: int = 10) -> Dict[str, Any]:
    """``gae.value_calibration`` of the seat-games' rows: the same rows, outcomes, phases and default weights as
    :func:`ev_by_phase`."""
    lengths = np.array([sg.n for sg in sgs], np.int64)
    ticks = np.concatenate([sg.rows["tick"] for sg in sgs]).astype(np.int64)
    if weights is None:
        weights = np.concatenate([G.row_weights(sg.rows["stratum"], sg.rows["forced"], sg.weight,
                                                family=sg.rows["family"]).value for sg in sgs])
    outcomes = np.repeat([sg.outcome for sg in sgs], lengths)
    end_ticks = np.repeat([int(sg.game.end_tick) for sg in sgs], lengths)
    return G.value_calibration(np.asarray(values, np.float64), outcomes, ticks, end_ticks, weights=weights, bins=bins)


TREND_ALL = "all"
_ND = "__ndarray__"


def pack_arrays(x: Any) -> Any:
    """Numpy arrays inside nested dicts / lists as ``{__ndarray__: uint8 tensor, dtype, shape}`` (byte-exact), so a
    checkpoint holding the probe set stays loadable with ``torch.load(weights_only=True)``. Object arrays raise."""
    if isinstance(x, Mapping):
        return {k: pack_arrays(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [pack_arrays(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.dtype.hasobject:
            raise TypeError("pack_arrays: object arrays cannot be stored")
        a = np.ascontiguousarray(x)
        return {_ND: torch.from_numpy(np.frombuffer(a.tobytes(), np.uint8).copy()), "dtype": a.dtype.str,
                "shape": list(a.shape)}
    if isinstance(x, np.generic):
        return x.item()
    return x


def unpack_arrays(x: Any) -> Any:
    """Inverse of :func:`pack_arrays`."""
    if isinstance(x, Mapping):
        if _ND in x:
            return np.frombuffer(x[_ND].numpy().tobytes(), np.dtype(x["dtype"])).reshape(x["shape"]).copy()
        return {k: unpack_arrays(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unpack_arrays(v) for v in x]
    return x


def gate_trend_metrics(seats: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Gate-trend diagnostics (a policy trained with PPO can drift toward passivity): gate statistics of one
    update from the learner's own rows, by opponent kind (``self_seat`` and ``mirror`` stay separate kinds) and
    over ``all``, for every row (no suffix) and for the 'tower HP behind' stratum (``_behind``: Φ < 0, i.e. our
    summed tower HP fraction below the opponent's at the row; ``gae.tower_potential``).

    Each seat is a mapping with ``game`` (game id), ``kind`` (opponent kind), ``family`` (executed family per
    row), ``live`` (bool: unified policy weight > 0, i.e. valid and not forced), ``ht`` (HT weight), ``phi`` and
    ``adv`` (the normalised advantages the loss uses). Per group:

    * ``gate_trend/act_rate[_behind]/<g>``: HT-weighted share of actionable (live) decisions that act (play or
      ability), a per-query estimate;
    * ``gate_trend/adv_gap[_behind]/<g>``: mean over games of the same-game difference "HT-weighted mean
      advantage of the play rows minus that of the wait rows" (seats of one game averaged first; games without
      both row kinds left out), with its game-clustered standard error ``..._se/<g>`` (NaN below 2 games) and
      the number of games ``..._games/<g>``.
    Diagnostic only (the alarm thresholds are config values); NaN where a group has no rows."""
    kinds = sorted({str(s["kind"]) for s in seats})
    out: Dict[str, Any] = {}
    for suffix in ("", "_behind"):
        num: Dict[str, float] = {}
        den: Dict[str, float] = {}
        gaps: Dict[str, Dict[str, List[float]]] = {}
        for s in seats:
            fam = np.asarray(s["family"])
            live = np.asarray(s["live"], bool)
            if suffix:
                live = live & (np.asarray(s["phi"], np.float64) < 0.0)
            ht = np.asarray(s["ht"], np.float64)
            adv = np.asarray(s["adv"], np.float64)
            act = fam != S.FAMILY_WAIT
            wp = ht * (live & (fam == S.FAMILY_PLAY))
            ww = ht * (live & (fam == S.FAMILY_WAIT))
            gap = None
            if wp.sum() > 0 and ww.sum() > 0:
                gap = float((wp * adv).sum() / wp.sum() - (ww * adv).sum() / ww.sum())
            for g in (str(s["kind"]), TREND_ALL):
                num[g] = num.get(g, 0.0) + float((ht * (live & act)).sum())
                den[g] = den.get(g, 0.0) + float((ht * live).sum())
                if gap is not None:
                    gaps.setdefault(g, {}).setdefault(str(s["game"]), []).append(gap)
        for g in kinds + [TREND_ALL]:
            d = den.get(g, 0.0)
            out[f"gate_trend/act_rate{suffix}/{g}"] = num.get(g, 0.0) / d if d > 0 else float("nan")
            per_game = np.array([float(np.mean(v)) for v in (gaps.get(g) or {}).values()], np.float64)
            n = int(per_game.size)
            out[f"gate_trend/adv_gap{suffix}/{g}"] = float(per_game.mean()) if n else float("nan")
            out[f"gate_trend/adv_gap{suffix}_se/{g}"] = (float(per_game.std(ddof=1) / math.sqrt(n)) if n >= 2
                                                         else float("nan"))
            out[f"gate_trend/adv_gap{suffix}_games/{g}"] = n
    return out


def argmax_flips(prev: Optional[Mapping[str, np.ndarray]], cur: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Argmax probe: per head ('slot', 'cell') the share of probe rows whose argmax changed between two evaluations
    (rows eligible in both: >= 2 legal choices), ``probe_flip/<h>`` (NaN without a previous evaluation or rows)
    and ``probe_rows/<h>``."""
    out: Dict[str, float] = {}
    for h in ("slot", "cell"):
        m = np.asarray(cur[f"{h}_rows"], bool)
        out[f"probe_rows/{h}"] = float(m.sum())
        if prev is None or np.asarray(prev[f"{h}_rows"]).shape != m.shape:
            out[f"probe_flip/{h}"] = float("nan")
            continue
        both = m & np.asarray(prev[f"{h}_rows"], bool)
        n = int(both.sum())
        diff = np.asarray(prev[h])[both] != np.asarray(cur[h])[both]
        out[f"probe_flip/{h}"] = float(diff.sum()) / n if n else float("nan")
    return out


def entropy_vs_baseline(cur: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray], cards: np.ndarray
                        ) -> Dict[str, Any]:
    """Entropy versus the IL baseline: per head, the summed per-row entropy of the current policy over
    that of the IL baseline on the probe rows where both are finite (``entropy_vs_il/<head>``, with the means
    ``probe_entropy/<head>`` and ``probe_entropy_il/<head>``), and per played card (``cards >= 0``) the same for
    the cells of the executed slot (``cell_entropy_vs_il/<card>``). NaN where the baseline sum is not positive."""
    out: Dict[str, Any] = {}
    cards = np.asarray(cards, np.int64)
    for h in sorted(set(cur) & set(base)):
        x, y = np.asarray(cur[h], np.float64), np.asarray(base[h], np.float64)
        if x.shape != y.shape:
            continue
        sel = np.isfinite(x) & np.isfinite(y)
        n = int(sel.sum())
        sx, sy = float(x[sel].sum()), float(y[sel].sum())
        out[f"probe_entropy/{h}"] = sx / n if n else float("nan")
        out[f"probe_entropy_il/{h}"] = sy / n if n else float("nan")
        out[f"entropy_vs_il/{h}"] = sx / sy if n and sy > 0 else float("nan")
        if h == "cells" and cards.shape == x.shape:
            for c in sorted({int(v) for v in cards[sel & (cards >= 0)]}):
                m = sel & (cards == c)
                sy_c = float(y[m].sum())
                out[f"cell_entropy_vs_il/{c}"] = float(x[m].sum()) / sy_c if sy_c > 0 else float("nan")
    return out


def _rows_sha(sg: B.SeatGame) -> str:
    """sha256 of a seat-game's row ticks, families and stored μ (the fixed-set identity)."""
    h = hashlib.sha256()
    for k, dt in (("tick", np.int64), ("family", np.int64), ("mu", np.float64)):
        h.update(k.encode() + b"\0" + np.ascontiguousarray(np.asarray(sg.rows[k], dt)).tobytes())
    return h.hexdigest()


@dataclass
class WarmupResult:
    """What :meth:`Learner.warmup_update` returns (critic warm-up)."""
    update: int
    metrics: Dict[str, Any]
    ev_fixed: Dict[str, Any]
    status: PR.WarmupStatus


class FixedSetEV:
    """The per-phase EV of the CURRENT critic on a fixed recorded set of ``critic.warmup.fixed_set_games``
    games, the input of ``phase_rules.critic_warmup_status``. The set is a list of ``buffer.UpdateData`` recorded
    once by the learner's acting weights (the base IL model); its digest (per update: update, rollout id, policy
    digest; per kept trained seat-game: game id, side, rows, outcome, end tick and a sha256 of its row ticks /
    families / mu) is checked across warm-up updates and blocks. Tokens are read lazily, ``chunk`` seat-games at
    a time (no token cache for the whole set)."""

    def __init__(self, learner: "Learner", data: Sequence[B.UpdateData], *, expect_games: Optional[int] = -1,
                 chunk: int = 64) -> None:
        self.learner = learner
        self.data = list(data)
        if not self.data:
            raise ValueError("fixed set: no recorded data")
        want_digest = learner.policy_digest()
        bad = sorted({str(d.policy_digest) for d in self.data if str(d.policy_digest) != want_digest})
        if bad:                                         # the set is recorded by the learner's acting weights
            raise ValueError(f"fixed set: recorded by policy {[b[:12] for b in bad]}, not the learner's acting "
                             f"weights {want_digest[:12]} (the set is recorded with theta = the base IL model)")
        n_games = sum(len(d.games) for d in self.data)
        want = learner.cfg.critic.warmup.fixed_set_games if expect_games == -1 else expect_games
        if want is not None and n_games != int(want):
            raise ValueError(f"fixed set: {n_games} kept games, expected {want} (critic.warmup.fixed_set_games)")
        self.n_games = n_games
        self.chunk = int(chunk)
        ident = [[int(d.update), str(d.rollout_id), str(d.policy_digest), str(sg.game.game_id), int(sg.side),
                  int(sg.n), float(sg.outcome), int(sg.game.end_tick), _rows_sha(sg)]
                 for d in self.data for sg in d.seat_games()]
        self.digest = hashlib.sha256(json.dumps(sorted(ident), separators=(",", ":")).encode()).hexdigest()

    def __call__(self) -> Dict[str, Any]:
        sgs = [sg for d in self.data for sg in d.seat_games()]
        values = self.learner.critic_values(sgs, chunk=self.chunk)
        ev = ev_by_phase(sgs, values)
        ev["n_games"] = self.n_games
        ev["fixed_set_digest"] = self.digest
        return ev


# =========================================================================== learner
class Learner:
    """One PPO update per :meth:`update` call (module docstring). Build with :meth:`from_il` (first block) or
    :meth:`from_checkpoint` (a chained block); :meth:`save` writes the barrier checkpoint."""

    SCALER_FORCE = False             # tests: gradient scaling on without an fp16 backbone (it must change nothing)


    # Host memory: False = an update's padded tokens are not kept per seat-game for the whole update but rebuilt
    # from the shard's flat arrays for each micro-batch (``SeatItem.example``); no value changes (a bitwise
    # comparison with True runs with this default). True = every seat-game's padded tokens are kept for the update.
    hold_examples = False
    # Host index: every micro-batch carries the model's host index (:func:`host_index_of`). Off = the plain
    # batches and forward. Not a config key: the numbers are the same (checked once per learner by
    # :meth:`_host_index_decide`, which turns it off again if they are not).
    host_index = False

    def __init__(self, cfg: Any, policy: AminiFormer, reference: AminiFormer, critic: Critic, *,
                 device: Any = "cpu", monitor: Optional[M.Monitor] = None, seed: int = 0,
                 first_update: int = 1, max_micro_rows: Optional[int] = None,
                 max_micro_tokens: Optional[int] = None, tables_digest: Optional[str] = None,
                 source: Optional[Mapping[str, Any]] = None, allow_failed_warmup: bool = False) -> None:
        for what, bad in (("bc.weight", cfg.bc.weight != 0.0), ("reward.shaping", cfg.reward.shaping != "none"),
                          ("policy.aux_heads", bool(cfg.policy.aux_heads)),
                          ("policy.advantage_norm", cfg.policy.advantage_norm != "per_update"),
                          ("critic.value_priv", bool(cfg.critic.value_priv)),
                          ("policy.lr_schedule", cfg.policy.lr_schedule != "constant")):
            if bad:
                raise NotImplementedError(f"learner: {what} is not supported (only the default is implemented)")
        self.cfg = cfg
        self.device = torch.device(device)
        self.precision = cfg.policy.backbone_dtype
        self.thresholds = M.Thresholds.from_mapping(cfg.monitor_thresholds())   # bf16 refused until tolerances set
        if self.precision == "bf16" and self.device.type not in ("cuda", "cpu"):
            raise ValueError(f"bf16 autocast needs a cuda or cpu device, not {self.device}")
        if self.precision == "fp16" and self.device.type != "cuda":
            raise ValueError(f"fp16 autocast needs a cuda device, not {self.device}")
        if float(policy.cfg.dropout) != 0.0 or policy.cfg.gate_bias is not None:
            raise ValueError("the PPO policy needs dropout 0 and gate_bias None (build it with ppo_config)")
        if reference.cfg.gate_bias is not None:
            raise ValueError("the KL reference needs gate_bias None")
        for what, mod in (("policy", policy), ("reference", reference), ("critic", critic)):
            check_model_policy_params(mod, what)
        self.policy = policy.to(self.device).train()
        self.reference = freeze(reference.to(self.device))
        self.critic = critic.to(self.device).train()
        set_policy_trainable(self.policy, cfg.policy.train_scope)
        p = cfg.policy
        self.opt = torch.optim.AdamW(self.policy.param_groups(p.weight_decay), lr=p.lr, betas=tuple(p.betas),
                                     eps=p.adam_eps, weight_decay=p.weight_decay)
        c = cfg.critic
        self.critic_opt = torch.optim.Adam(self.critic.trainable_parameters(), lr=c.lr, betas=tuple(c.betas),
                                           eps=c.adam_eps)
        # Gradient scaling for an fp16 backbone (a gradient that passes an fp16 activation is fp16, and a
        # small one would round to 0). Disabled scalers pass everything through unchanged, so fp32 and bf16 run
        # the code they always ran. Order per minibatch: scale(loss).backward() per micro-batch, unscale_ once,
        # then the finite check, the clip and the step all see true gradients (accumulate_gradients, _opt_step).
        amp_scale = self.precision == "fp16" or bool(self.SCALER_FORCE)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=amp_scale)
        self.critic_scaler = torch.amp.GradScaler(self.device.type, enabled=amp_scale)
        self._scaler_pending = False                 # unscale_ done, update() not yet (one per optimiser step)
        if monitor is not None:                       # the config sets lr / beta / limits
            got = (float(monitor.base_lr), float(monitor.base_beta))
            if got != (float(p.lr), float(cfg.kl.beta)) or monitor.th != self.thresholds:
                raise ValueError(f"monitor (base_lr, base_beta) {got} or thresholds differ from the config "
                                 f"({p.lr}, {cfg.kl.beta}); build it from cfg")
        self.monitor = monitor if monitor is not None else M.Monitor(
            thresholds=self.thresholds, base_lr=p.lr, base_beta=cfg.kl.beta)
        self.seed = int(seed)
        self.next_update = int(first_update)
        self.steps_total = 0
        # micro-batch caps: the config's unless a test or a timing run overrides them
        self.max_micro_rows = int(p.micro_rows if max_micro_rows is None else max_micro_rows)
        self.max_micro_tokens = int(p.max_micro_tokens if max_micro_tokens is None else max_micro_tokens)
        if self.max_micro_rows < 1 or self.max_micro_tokens < 0:
            raise ValueError("max_micro_rows must be >= 1 and max_micro_tokens >= 0")
        self.grad_clip = float(p.grad_clip)
        self.grad_clip_raised = False
        self.clipped_frac_history: List[float] = []
        self.compat = compat_record(hazard_dt=cfg.rollout.hazard_dt, tables_digest=tables_digest)
        self.compat_digest = compat_digest(self.compat)
        self.config_digest = cfg.digest()
        self.source = dict(source or {})
        self.last_metrics: Dict[str, Any] = {}
        self._fdtype = next(self.policy.parameters()).dtype
        self.config_changes: List[Dict[str, Any]] = []          # C.config_change records (oldest first)
        self.warmup: Dict[str, Any] = {"done": 0, "history": [], "fixed_set_digest": None, "status": None}
        self.critic_warmup: Optional[Dict[str, Any]] = None     # the warm-up checkpoint the critic came from
        self.last_pin_check: Dict[str, int] = {}
        # Escape hatch for experiments on the critic remedy only (not saved; every block passes it again): let
        # update() step the policy after a warm-up on this learner that failed or did not finish
        self.allow_failed_warmup = bool(allow_failed_warmup)
        # diagnostics: the lr / beta read back at the end of the last update (effective-value check) and the fixed
        # argmax probe; both saved in the checkpoint
        self.effective_last: Dict[str, Any] = self._effective_mark()
        self._eff_monitor = self.monitor                 # the monitor whose factor log the mark refers to
        self.probe: Optional[Dict[str, Any]] = None
        self.probe_prev: Optional[Dict[str, np.ndarray]] = None
        # policy digest at which probe_prev was taken: a weight swap since (rollback_to, a stop after some steps)
        # re-bases it before the next update
        self.probe_prev_digest: Optional[str] = None
        self._ref_digest: Optional[Tuple[Any, str]] = None      # (reference module, its digest)
        # lag-1 overlap, in memory only, never saved (a block always ends drained, so the first update of a block,
        # a resume or a fork is a fill): (start weights on CPU, start digest, end digest) of the previous update of
        # a policy.rollout_lag = 1 run, and the reusable parent module (θ_{u-2})
        self._lag1_prev: Optional[Tuple[Dict[str, torch.Tensor], str, str]] = None
        # policy.rollout_lag = 2: the same record of the update before that one (theta_{u-3} at its start)
        self._lag2_prev: Optional[Tuple[Dict[str, torch.Tensor], str, str]] = None
        self._parent_mod: Optional[AminiFormer] = None
        # Optional learner compilation. Compiled callables live outside
        # Module._modules so state_dicts and optimizer parameter identities stay stable.
        lc = getattr(cfg, "learner", None)
        self.compile_mode = str(getattr(lc, "compile_mode", "") or "")
        self.compile_dynamic = bool(getattr(lc, "compile_dynamic", False))
        self.compile_requested = bool(self.compile_mode)
        self.compile_status: Dict[str, Any] = {
            "requested": self.compile_requested, "mode": self.compile_mode,
            "dynamic": self.compile_dynamic, "active": False,
            "smoke": "pending" if self.compile_requested else "off",
            "backend": None, "error": None, "graphs": 0, "recompiles": 0,
        }
        self.__dict__["_compiled_policy"] = None
        self.__dict__["_compiled_reference"] = None
        self.__dict__["_compile_decided"] = not self.compile_requested
        self.__dict__["_compile_graph_base"] = 0
        self.__dict__["_compile_backups"] = []
        self.__dict__["_compile_methods"] = []

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_il(cls, cfg: Any, path: Union[str, Path], *, device: Any = "cpu", dtype: Optional[torch.dtype] = None,
                critic_from: Optional[Union[str, Path]] = None, require_warmup_pass: bool = True,
                allow_compat_change: bool = False, require_pin: bool = False, require_warmup: bool = False,
                **kw: Any) -> "Learner":
        """First block: policy, reference and critic from one IL checkpoint.

        The file's sha256 must equal ``opponents.checkpoint_sha256.il_v4`` (the base IL model's pin) when that pin
        is set (``require_pin``: also when it is not). ``critic_from``: a critic warm-up checkpoint (:meth:`save` of
        a learner that ran only :meth:`warmup_update`) whose trained critic and critic optimiser replace the fresh
        IL copy: it must hold the same acting weights (no policy step), the same frozen step encoder, the same PPO
        compatibility digest (unless ``allow_compat_change``) and a passed warm-up (unless ``require_warmup_pass``
        is False; a failed warm-up needs the critic remedy first). The warm-up record (path, sha256,
        ``ev_warmup_end`` for ``Baselines``) is :attr:`critic_warmup` and ``source['critic_warmup']``.
        ``require_warmup`` (a production PPO run): refuse to start without ``critic_from``.
        The pin is compared before the file is unpickled (and again on the bytes that were loaded)."""
        if require_warmup and critic_from is None:
            raise ValueError("require_warmup: the PPO run starts from a passed critic warm-up (critic_from=)")
        pin = cfg.opponents.checkpoint_sha256.get("il_v4")
        if pin is None and require_pin:
            raise PinError("opponents.checkpoint_sha256.il_v4 is not set (pin the base IL checkpoint before the run)")
        if pin is not None:                           # no pickle code from a wrong file
            got = IL.file_sha256(str(path))
            if got != pin:
                raise PinError(f"{path}: sha256 {got[:12]} != opponents.checkpoint_sha256.il_v4 {pin[:12]}")
        policy, rec = load_source_policy(path, device=device, dtype=dtype)
        if pin is not None and rec["sha256"] != pin:
            raise PinError(f"{path}: sha256 {rec['sha256'][:12]} != opponents.checkpoint_sha256.il_v4 {pin[:12]}")
        # environment tables (config.ENV_TABLE_KEYS, recorded by config.env_record) are never part of the model
        # compatibility digest (config.model_tables_digest drops them)
        kw.setdefault("tables_digest", C.model_tables_digest(IL._tables_digest_map(rec.get("tables"))))
        rec.pop("tables", None)
        lr = cls(cfg, policy, make_reference(policy), Critic(policy), device=device, source=rec, **kw)
        if critic_from is not None:
            lr._load_warmup_critic(critic_from, require_pass=require_warmup_pass,
                                   allow_compat_change=allow_compat_change)
        return lr

    def _load_warmup_critic(self, path: Union[str, Path], *, require_pass: bool, allow_compat_change: bool) -> None:
        """Take the critic (and its optimiser) of a warm-up checkpoint (:meth:`from_il` ``critic_from``)."""
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        ppo = ck.get("ppo") if isinstance(ck, dict) else None
        if not isinstance(ppo, dict) or ppo.get("version") != LEARNER_VERSION:
            raise ValueError(f"{path}: not a {LEARNER_VERSION} checkpoint")
        wu = ppo.get("warmup") or {}
        if int(ppo.get("steps_total", -1)) != 0 or int(ppo.get("next_update", 0)) != 1:
            raise ValueError(f"{path}: not a warm-up checkpoint (policy steps {ppo.get('steps_total')}, "
                             f"next update {ppo.get('next_update')})")
        if int(wu.get("done", 0)) < 1:
            raise ValueError(f"{path}: no warm-up update was run")
        if ppo["policy_digest"] != self.policy_digest():
            raise ValueError(f"{path}: warm-up acting weights {str(ppo['policy_digest'])[:12]} != this IL checkpoint "
                             f"{self.policy_digest()[:12]} (the critic's frozen step encoder must be the same)")
        if ppo["compat_digest"] != self.compat_digest and not allow_compat_change:
            diff = sorted(k for k in set(self.compat) | set(ppo["compat"])
                          if self.compat.get(k) != ppo["compat"].get(k))
            raise ValueError(f"{path}: PPO compatibility digest changed since the warm-up ({diff})")
        status = wu.get("status") or {}
        if require_pass and status.get("passed") is not True:
            raise ValueError(f"{path}: the critic warm-up did not pass ({status.get('reason')}, "
                             f"{status.get('failures')}); the critic remedy comes before any policy step")
        state = ppo["critic"]
        se = {k[len("step_encoder."):]: v for k, v in state.items() if k.startswith("step_encoder.")}
        own = self.critic.step_encoder.state_dict()
        if set(se) != set(own) or any(not torch.equal(se[k].to(own[k].device, own[k].dtype), own[k]) for k in own):
            raise ValueError(f"{path}: the warm-up critic's frozen step encoder differs from this IL checkpoint's")
        self.critic.load_state_dict(state, strict=True)
        self.critic_opt.load_state_dict(ppo["critic_optimizer"])     # Adam moments of the warm-up
        self._apply_critic_hparams()                                  # the PPO config's lr / betas / eps
        self.critic_warmup = {"path": str(Path(path).resolve()), "sha256": IL.file_sha256(str(path)),
                              "updates": int(wu["done"]), "fixed_set_digest": wu.get("fixed_set_digest"),
                              "ev_warmup_end": status.get("ev_warmup_end"), "passed": status.get("passed"),
                              "reason": status.get("reason"), "config_digest": ppo.get("config_digest"),
                              "compat_digest": ppo["compat_digest"]}
        self.source["critic_warmup"] = dict(self.critic_warmup)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], cfg: Any, *, device: Any = "cpu",
                        monitor: Optional[M.Monitor] = None, baselines: Optional[M.Baselines] = None,
                        allow_compat_change: bool = False, config_change: Optional[Mapping[str, Any]] = None,
                        **kw: Any) -> "Learner":
        """Continue from a :meth:`save` checkpoint. Refuses a different PPO compatibility digest (unless
        ``allow_compat_change``) or config digest, and acting weights (``model_ema``, what the servers load) that
        differ from ``model_raw`` or from the stored policy digest, or a reference that differs from its digest.
        The monitor is restored from the checkpoint unless given, with the stored baselines and dashboard path
        unless ``baselines`` is given.

        A different config digest is accepted only through ``config_change``, a ``config.config_change(old,
        new, update=, reason=)`` record whose old digest is the checkpoint's (its stored config is rebuilt),
        whose new digest is ``cfg``'s and whose keys are all in ``config.RUNTIME_CHANGE_KEYS``; it is appended to
        :attr:`config_changes`. A restored monitor takes the new config's base lr / beta. The init checkpoint's
        sha256 must equal a set ``opponents.checkpoint_sha256.il_v4``."""
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        ppo = ck.get("ppo") if isinstance(ck, dict) else None
        if not isinstance(ppo, dict) or ppo.get("version") != LEARNER_VERSION:
            raise ValueError(f"{path}: not a {LEARNER_VERSION} checkpoint")
        pending: Optional[Dict[str, Any]] = None
        if cfg.digest() != ppo["config_digest"]:
            if config_change is None:
                raise ValueError(f"config digest {cfg.digest()} != checkpoint {ppo['config_digest']} (a mid-run "
                                 "change needs config_change=config.config_change(old, new, ...))")
            if not isinstance(ppo.get("config"), Mapping):
                raise ValueError(f"{path}: the checkpoint has no stored config to check a change against")
            old = C.from_dict(json.loads(json.dumps(ppo["config"])))
            if old.digest() != ppo["config_digest"]:
                raise ValueError(f"{path}: the stored config does not rebuild to the stored digest")
            C.check_config_change(config_change, old, cfg)
            if int(config_change["update"]) != int(ppo["next_update"]):
                raise ValueError(f"config change applies from update {config_change['update']}, the checkpoint "
                                 f"continues with update {ppo['next_update']}")
            pending = json.loads(json.dumps(dict(config_change)))
        elif config_change is not None:
            raise ValueError("config_change given but the config digest already equals the checkpoint's")
        pin = cfg.opponents.checkpoint_sha256.get("il_v4")
        src_sha = (ppo.get("source") or {}).get("sha256")
        if pin is not None and src_sha != pin:        # a missing record is a mismatch
            raise PinError(f"{path}: the run started from {str(src_sha)[:12] if src_sha else 'an unrecorded file'}"
                           f", not the pinned base IL checkpoint {pin[:12]}")
        ema = ck["model_ema"]
        d_ema = state_dict_digest(ema)
        if d_ema != ppo["policy_digest"]:
            raise ValueError(f"{path}: model_ema digest {d_ema[:12]} != stored policy digest "
                             f"{str(ppo['policy_digest'])[:12]}")
        raw = ck.get("model_raw")
        if raw is not None and raw is not ema and state_dict_digest(raw) != d_ema:
            raise ValueError(f"{path}: model_raw differs from model_ema (the PPO learner saves them equal)")
        policy = model_from_state(ck["model_config"], ema, device=device)
        ref = model_from_state(ppo["reference"]["model_config"], ppo["reference"]["state"], device=device)
        d_ref = state_dict_digest(ref.state_dict())
        if d_ref != ppo["reference"].get("digest"):
            raise ValueError(f"{path}: reference digest {d_ref[:12]} != stored {str(ppo['reference'].get('digest'))[:12]}")
        critic = Critic(model_from_state(ck["model_config"], ema, device=device))
        critic.load_state_dict(ppo["critic"], strict=True)
        restored = monitor is None
        if monitor is None:
            if baselines is None and ppo.get("monitor_baselines") is not None:
                baselines = M.Baselines.from_mapping(ppo["monitor_baselines"])
            mkw: Dict[str, Any] = {}
            if ppo.get("monitor_dashboard"):
                mkw["dashboard_path"] = ppo["monitor_dashboard"]
            if pending is not None:                   # the change sets the base values the monitor scales
                mkw.update(base_lr=float(cfg.policy.lr), base_beta=float(cfg.kl.beta))
            monitor = M.Monitor.from_state(ppo["monitor"], baselines,
                                           M.Thresholds.from_mapping(cfg.monitor_thresholds()), **mkw)
        kw.setdefault("tables_digest", ppo["compat"].get("tables_digest"))
        kw.setdefault("seed", ppo["seed"])
        kw.setdefault("first_update", ppo["next_update"])
        kw.setdefault("source", ppo.get("source"))
        lr = cls(cfg, policy, ref, critic, device=device, monitor=monitor, **kw)
        if lr.policy_digest() != d_ema:
            raise ValueError(f"{path}: rebuilt policy digest {lr.policy_digest()[:12]} != {d_ema[:12]}")
        if lr.compat_digest != ppo["compat_digest"] and not allow_compat_change:
            diff = sorted(k for k in set(lr.compat) | set(ppo["compat"])
                          if lr.compat.get(k) != ppo["compat"].get(k))
            raise ValueError(f"PPO compatibility digest changed ({diff}); re-check before continuing")
        lr.opt.load_state_dict(ck["optimizer"])
        lr.critic_opt.load_state_dict(ppo["critic_optimizer"])
        if lr.scaler.is_enabled() and ck.get("scaler"):              # absent in a checkpoint of an fp32 run
            lr.scaler.load_state_dict(ck["scaler"])
        if lr.critic_scaler.is_enabled() and ppo.get("critic_scaler"):
            lr.critic_scaler.load_state_dict(ppo["critic_scaler"])
        lr._apply_critic_hparams()
        lr.steps_total = int(ppo["steps_total"])
        lr.grad_clip = float(ppo["grad_clip"])
        lr.grad_clip_raised = bool(ppo["grad_clip_raised"])
        lr.clipped_frac_history = [float(x) for x in ppo["clipped_frac_history"]]
        lr.last_metrics = dict(ck.get("metrics") or {})
        lr.config_changes = [dict(r) for r in ppo.get("config_changes") or []]
        if pending is not None:
            lr.config_changes.append(pending)
        if ppo.get("warmup") is not None:
            lr.warmup = copy.deepcopy(ppo["warmup"])
        lr.critic_warmup = copy.deepcopy(ppo.get("critic_warmup"))
        if ppo.get("effective_last") is not None and pending is None and restored:
            lr.effective_last = dict(ppo["effective_last"])      # the chain of effective-value checks continues
        if ppo.get("probe") is not None:
            lr.probe = unpack_arrays(ppo["probe"])
            lr.probe_prev = unpack_arrays(ppo.get("probe_prev"))
            lr.probe_prev_digest = ppo.get("probe_prev_digest")
        return lr

    def adopt(self, other: "Learner") -> None:
        """Become ``other`` in place (remote learner): every attribute of a learner rebuilt from the
        checkpoint a remote update returned replaces this object's, so the driver keeps the object it holds.
        ``other`` must be the same kind of learner on the same config and compatibility digests; it is the
        caller's to discard afterwards (both objects then share the modules)."""
        if type(other) is not type(self):
            raise ValueError(f"cannot adopt a {type(other).__name__}")
        if other.config_digest != self.config_digest or other.compat_digest != self.compat_digest:
            raise ValueError("cannot adopt a learner of another config or compatibility digest")
        if other.device != self.device:
            raise ValueError(f"cannot adopt a learner on {other.device} (this one is on {self.device})")
        state = dict(other.__dict__)
        self.__dict__.clear()
        self.__dict__.update(state)

    def change_config(self, new_cfg: Any, *, reason: str, update: Optional[int] = None) -> Dict[str, Any]:
        """Apply a recorded mid-run config change at a barrier (stall ladder, critic remedy, pilot retry):
        ``config.config_change`` refuses anything outside ``RUNTIME_CHANGE_KEYS``. From the next
        update the learner uses ``new_cfg`` (lambda_tick, critic epochs; lr / beta through the monitor's base
        values, its temporary / permanent factors kept); the record (old and new digests) goes into
        :attr:`config_changes` and every later checkpoint. Shards of later updates carry the new digest."""
        u = self.next_update if update is None else int(update)
        if u != self.next_update:
            raise ValueError(f"a config change applies from the next update {self.next_update}, not {u}")
        new_lc = getattr(new_cfg, "learner", None)
        new_mode = str(getattr(new_lc, "compile_mode", "") or "")
        new_dynamic = bool(getattr(new_lc, "compile_dynamic", False))
        # getattr: the config-change tests call this method on a stand-in object without the compile fields
        if (new_mode, new_dynamic) != (getattr(self, "compile_mode", ""), getattr(self, "compile_dynamic", False)):
            if self.__dict__.get("_compile_decided", True):
                raise ValueError("learner compile settings are fixed at block start; start a new block")
            self.compile_mode, self.compile_dynamic = new_mode, new_dynamic
            self.compile_requested = bool(new_mode)
            self.compile_status.update({"requested": self.compile_requested, "mode": new_mode,
                                        "dynamic": new_dynamic, "smoke": "pending" if new_mode else "off"})
        rec = C.config_change(self.cfg, new_cfg, update=u, reason=reason)
        th = M.Thresholds.from_mapping(new_cfg.monitor_thresholds())
        if th != self.thresholds:      # calibrated thresholds (config.RUNTIME_THRESHOLD_KEYS), atomic swap
            self.monitor.apply_thresholds(th, update=u, reason=reason)
            self.thresholds = th
        self.cfg = new_cfg
        self.config_digest = new_cfg.digest()
        self.monitor.base_lr = float(new_cfg.policy.lr)
        self.monitor.base_beta = float(new_cfg.kl.beta)
        self.config_changes.append(rec)
        self.effective_last = self._effective_mark()    # the recorded change re-bases the planned lr / beta
        if int(new_cfg.policy.rollout_lag) < 1:         # a switch back to lag 0 forgets the parent
            self._lag1_prev = None
        if int(new_cfg.policy.rollout_lag) < 2:         # ... and one to lag 0 or 1 the grandparent
            self._lag2_prev = None
        return rec

    # ------------------------------------------------------------------ effective lr / beta
    def _effective_mark(self) -> Dict[str, Any]:
        """The monitor's current lr / beta and its factor-log position (the base of the next effective-value
        check)."""
        return {"lr": float(self.monitor.lr), "beta": float(self.monitor.beta), "seq": int(self.monitor.scale_seq)}

    def _set_lr(self, lr: float) -> List[float]:
        """Write ``lr`` (x each group's ``lr_mult``) into every policy param group and read it back, divided by
        the group's ``lr_mult``: what the next ``opt.step`` uses."""
        for g in self.opt.param_groups:
            g["lr"] = lr * float(g.get("lr_mult", 1.0))
        return self._read_lr()

    def _read_lr(self) -> List[float]:
        return [float(g["lr"]) / float(g.get("lr_mult", 1.0)) for g in self.opt.param_groups]

    def _effective_start(self, u: int) -> Dict[str, Any]:
        """Effective-value check at the start of update ``u``, before any step: the planned lr / beta (config base
        x the monitor's factors) must equal the values read back at the end of the last update times every lr /
        beta factor the monitor logged since (update actions, gate actions, rollbacks: ``Monitor.planned_since``);
        the lr the policy param groups hold is recorded (``effective/lr_groups_found_*``, informational), then the
        planned lr is written into every group and read back, and the critic groups must hold ``critic.lr``.
        A failure stops the monitor and raises :class:`EffectiveValueError`."""
        lr, beta = float(self.monitor.lr), float(self.monitor.beta)
        rec: Dict[str, Any] = {"effective/lr_planned": lr, "effective/beta_planned": beta,
                               "effective/lr_config": float(self.cfg.policy.lr),
                               "effective/beta_config": float(self.cfg.kl.beta)}
        fails: List[str] = []
        last = self.effective_last
        got = None
        if self._eff_monitor is not self.monitor or int(last["seq"]) > int(self.monitor.scale_seq):
            rec["effective/chain"] = "rebased (the monitor was replaced since the last read-back)"
        else:
            got = self.monitor.planned_since(int(last["seq"]), float(last["lr"]), float(last["beta"]))
            if got is None:
                rec["effective/chain"] = "unverifiable (factor log trimmed)"
        if got is not None:
            elr, ebeta, entries = got
            rec["effective/chain"] = "ok"
            rec["effective/chain_factors"] = len(entries)
            if not _close(elr, lr):
                fails.append(f"planned lr {lr!r} != last read-back {last['lr']!r} x logged factors = {elr!r}")
            if not _close(ebeta, beta):
                fails.append(f"planned beta {beta!r} != last read-back {last['beta']!r} x logged factors = {ebeta!r}")
        found = self._read_lr()                         # informational: a rollback's fresh AdamW holds the config lr
        rec["effective/lr_groups_found_min"], rec["effective/lr_groups_found_max"] = min(found), max(found)
        groups = self._set_lr(lr)
        rec["effective/lr_groups_min"], rec["effective/lr_groups_max"] = min(groups), max(groups)
        if not all(_close(x, lr) for x in groups):
            fails.append(f"policy param groups hold lr {sorted(set(groups))}, planned {lr!r}")
        crit = [float(g["lr"]) for g in self.critic_opt.param_groups]
        rec["effective/critic_lr"] = crit[0] if len(set(crit)) == 1 else crit
        if not all(_close(x, self.cfg.critic.lr) for x in crit):
            fails.append(f"critic param groups hold lr {sorted(set(crit))}, config {self.cfg.critic.lr!r}")
        if fails:
            rec["effective/failures"] = fails
            self._monitor_stop("effective_values", "; ".join(fails), u, rec)
            raise EffectiveValueError(f"update {u}: effective lr / beta differ from the plan: {'; '.join(fails)}")
        return rec

    def _effective_end(self, u: int, rec: Dict[str, Any], lr: float, beta: float,
                       actions: Sequence[M.Action]) -> List[M.Action]:
        """Effective-value check after the update's monitor actions: the monitor's new lr / beta must equal the
        planned values times the factors of this update's ``lr_scale`` / ``beta_scale`` actions (temporary lr /
        beta factors, KL step, restores); the new lr is written into the param groups and read back. Records
        ``effective/next_*``; a mismatch stops the monitor (returned actions). The read-back becomes the base of
        the next check."""
        elr, ebeta = float(lr), float(beta)
        for a in actions:
            if a.kind == "lr_scale":
                elr *= float(a.value)
            elif a.kind == "beta_scale":
                ebeta *= float(a.value)
        nlr, nbeta = float(self.monitor.lr), float(self.monitor.beta)
        groups = self._set_lr(nlr)
        rec.update({"effective/next_lr_planned": nlr, "effective/next_beta_planned": nbeta,
                    "effective/next_lr_groups_min": min(groups), "effective/next_lr_groups_max": max(groups)})
        fails = []
        if not _close(elr, nlr):
            fails.append(f"lr after the actions {nlr!r} != {lr!r} x action factors = {elr!r}")
        if not _close(ebeta, nbeta):
            fails.append(f"beta after the actions {nbeta!r} != {beta!r} x action factors = {ebeta!r}")
        if not all(_close(x, nlr) for x in groups):
            fails.append(f"policy param groups hold lr {sorted(set(groups))} after the actions, planned {nlr!r}")
        self.effective_last = {"lr": nlr, "beta": nbeta, "seq": int(self.monitor.scale_seq)}
        self._eff_monitor = self.monitor
        if not fails or self.monitor.stopped:
            return []
        rec["effective/failures"] = fails
        return self._monitor_stop("effective_values", "; ".join(fails), u, {"failures": fails})

    # ------------------------------------------------------------------ fixed argmax probe
    def _probe_items(self) -> List[SeatItem]:
        pr = self.probe or {}
        dev = self.device
        full = all(k in pr for k in PROBE_ROW_KEYS)       # probes made before the IL baseline hold slot / play only
        out = []
        for i, (ex, n, play, slot, temp, tok) in enumerate(zip(pr["examples"], pr["n"], pr["play"], pr["slot"],
                                                                 pr["temperature"], pr["tok"])):
            action = {"slot": torch.from_numpy(np.asarray(slot, np.int64)).to(dev)}
            dt = torch.empty(0)
            if full:
                for k in ("family", "cell", "ability"):
                    action[k] = torch.from_numpy(np.asarray(pr[k][i], np.int64)).to(dev)
                action["play_blocked"] = torch.from_numpy(np.asarray(pr["play_blocked"][i], bool)).to(dev)
                dt = torch.from_numpy(np.asarray(pr["dt"][i], np.float64)).to(dev)
            ok = pr["client_slot_ok"][i] if pr.get("client_slot_ok") is not None else None
            out.append(SeatItem(index=i, sg=None, held=ex, n=int(n), temperature=tuple(temp),  # type: ignore[arg-type]
                                action=action, dt=dt, mu=torch.empty(0), row_type=torch.empty(0),
                                play=torch.from_numpy(np.asarray(play, bool)).to(dev), card=np.empty(0),
                                tok=tuple(tok), client_ok=None if ok is None else np.asarray(ok, bool)))
        return out

    def _make_probe(self, items: Sequence[SeatItem], update: int) -> None:
        """The fixed probe set, chosen once (the first update this run's learner sees, unless restored from a
        checkpoint): whole kept seat-games in (game id, side) order until ``monitor.probe_rows`` rows.
        Besides the tokens it keeps each row's executed action, dt and card (:data:`PROBE_ROW_KEYS`), the inputs
        of ``logprob.head_entropy`` for the IL entropy baseline (:meth:`_probe_eval`)."""
        cap = int(self.cfg.monitor.probe_rows)
        if cap <= 0 or self.probe is not None:
            return
        chosen, rows = [], 0
        for it in sorted(items, key=lambda x: (str(x.sg.game.game_id), int(x.sg.side))):
            if rows >= cap:
                break
            chosen.append(it)
            rows += it.n
        self.probe = {"update": int(update), "rows": rows,
                      "ids": [[str(it.sg.game.game_id), int(it.sg.side)] for it in chosen],
                      "examples": [copy.deepcopy(it.example) for it in chosen], "n": [int(it.n) for it in chosen],
                      "play": [it.play.cpu().numpy().astype(bool) for it in chosen],
                      "slot": [it.action["slot"].cpu().numpy().astype(np.int64) for it in chosen],
                      "temperature": [list(it.temperature) for it in chosen], "tok": [list(it.tok) for it in chosen],
                      "family": [it.action["family"].cpu().numpy().astype(np.int64) for it in chosen],
                      "cell": [it.action["cell"].cpu().numpy().astype(np.int64) for it in chosen],
                      "ability": [it.action["ability"].cpu().numpy().astype(np.int64) for it in chosen],
                      "play_blocked": [it.action["play_blocked"].cpu().numpy().astype(bool) for it in chosen],
                      "dt": [it.dt.cpu().numpy().astype(np.float64) for it in chosen],
                      "card": [np.asarray(it.card, np.int64) for it in chosen],
                      "client_slot_ok": [_client_ok(it).copy() for it in chosen],   # the actor's support
                      "il_entropy": None, "il_digest": None}

    @torch.no_grad()
    def probe_argmax(self) -> Optional[Dict[str, np.ndarray]]:
        """Argmax of the slot head (rows with >= 2 legal slots) and of the cells of the executed slot (play rows
        with >= 2 legal cells) of the CURRENT policy on the probe rows (untempered, training-mode forward with
        dropout 0, no state change); -1 where a row is not eligible. None without a probe."""
        return self._probe_eval(entropy=False)[0]

    @torch.no_grad()
    def _probe_eval(self, *, entropy: bool) -> Tuple[Optional[Dict[str, np.ndarray]],
                                                     Optional[Dict[str, np.ndarray]]]:
        """One forward of the CURRENT policy on the probe rows: :meth:`probe_argmax`'s argmax arrays and, with
        ``entropy`` (and a probe that holds :data:`PROBE_ROW_KEYS`), the per-row entropies of each head
        (``logprob.head_entropy`` of the untempered output, NaN outside the head's row mask or non-finite),
        keyed by the canonical head names. No gradient, no RNG draw, no module-state change."""
        if not self.probe:
            return None, None
        items = self._probe_items()
        entropy = bool(entropy) and all(k in self.probe for k in PROBE_ROW_KEYS)
        res: Dict[int, Tuple[np.ndarray, ...]] = {}
        ents: Dict[int, Dict[str, np.ndarray]] = {}
        for micro in self._micro(items):
            b = self._batch(micro)
            out = self._forward(self.policy, b, self._cell_args(b, micro), micro)
            sl = out["slot"].detach()
            sleg = LP.legal(sl)
            s_rows = sleg.sum(-1) >= 2
            s_arg = torch.where(sleg, sl.double(), torch.full_like(sl, -math.inf, dtype=torch.float64)).argmax(-1)
            cells = LP.row_cells(out)
            if cells is None:
                c_rows = torch.zeros_like(s_rows)
                c_arg = torch.full_like(s_arg, -1)
            else:
                cells = cells.detach()
                cleg = LP.legal(cells)
                play = torch.cat([it.play for it in micro])
                c_rows = play & (cleg.sum(-1) >= 2)
                c_arg = torch.where(cleg, cells.double(),
                                    torch.full_like(cells, -math.inf, dtype=torch.float64)).argmax(-1)
            ent_rows: Dict[str, np.ndarray] = {}
            if entropy:
                ent = LP.head_entropy(out, _cat(micro, "dt"), _cat_action(micro))
                for k, mk in ENT_MASK.items():
                    if k in ent:
                        x = ent[k].detach().double()
                        x = torch.where(ent[mk].bool() & torch.isfinite(x), x, torch.full_like(x, math.nan))
                        ent_rows[L.canonical_head(k)] = x.cpu().numpy()
            off = 0
            for it in micro:
                sl_ = slice(off, off + it.n)
                res[it.index] = tuple(x[sl_].cpu().numpy() for x in (s_arg, s_rows, c_arg, c_rows))
                ents[it.index] = {h: v[sl_] for h, v in ent_rows.items()}
                off += it.n
        parts = [res[i] for i in range(len(items))]
        s_rows = np.concatenate([p[1] for p in parts]).astype(bool)
        c_rows = np.concatenate([p[3] for p in parts]).astype(bool)
        arg = {"slot": np.where(s_rows, np.concatenate([p[0] for p in parts]), -1).astype(np.int64),
               "slot_rows": s_rows,
               "cell": np.where(c_rows, np.concatenate([p[2] for p in parts]), -1).astype(np.int64),
               "cell_rows": c_rows}
        if not entropy:
            return arg, None
        heads = sorted(ents[0]) if ents else []
        return arg, {h: np.concatenate([ents[i][h] for i in range(len(items))]).astype(np.float64) for h in heads}

    def set_reference(self, model: AminiFormer) -> None:
        """Swap the KL reference (the champion after a promotion); stored frozen, gate_bias None.
        A champion with non-neutral ``policy_params`` is refused (the OpponentSpec temperature is the only
        tempering)."""
        check_model_policy_params(model, "reference")
        self.reference = make_reference(model).to(self.device)
        self._ref_digest = None
        if self.compile_requested and self.compile_status.get("active"):
            se, cd = self.reference.step_encoder, self.reference.cells
            se.packed = True
            se.pad_multiple = 0 if self.compile_dynamic else 4096
            se.compiled = cd.compiled = self.policy.step_encoder.compiled
            cd.pad_rows = 0 if self.compile_dynamic else 64
            cd.p_regions = True
            temporal = self.reference.temporal
            kw = dict(self.__dict__["_compile_kwargs"])
            self.__dict__["_compile_methods"].append((temporal, temporal.forward))
            temporal.forward = torch.compile(temporal.forward, **kw)

    def _apply_critic_hparams(self) -> None:
        """The critic optimiser's lr / betas / eps from the current config (a loaded optimiser state keeps only
        its moments)."""
        c = self.cfg.critic
        for g in self.critic_opt.param_groups:
            g["lr"] = float(c.lr)
            g["betas"] = tuple(float(b) for b in c.betas)
            g["eps"] = float(c.adam_eps)

    def _warmup_gate(self) -> None:
        """No policy step on a critic whose warm-up on this learner (or the checkpoint it resumed) failed or did
        not finish; the critic remedy comes first (``allow_failed_warmup`` for experiments on the remedy only)."""
        if self.steps_total != 0 or self.allow_failed_warmup or int(self.warmup.get("done", 0)) < 1:
            return
        st = self.warmup.get("status") or {}
        if not (st.get("stop") and st.get("passed") is True):
            state = "did not pass" if st.get("stop") else "has not finished"
            raise RuntimeError(f"the critic warm-up {state} ({st.get('reason')}, {st.get('failures')}): no policy "
                               "step before a passed warm-up (the critic remedy first, or "
                               "allow_failed_warmup=True for an experiment on the remedy)")

    # ------------------------------------------------------------------ identity of the acting weights
    def policy_digest(self) -> str:
        """sha256 of the acting weights (what :meth:`save` writes under ``model_ema``)."""
        return state_dict_digest(self.policy.state_dict())

    def reference_digest(self) -> str:
        return state_dict_digest(self.reference.state_dict())

    def _reference_digest_cached(self) -> str:
        """:meth:`reference_digest`, hashed once per reference module (the reference is frozen; :meth:`set_reference`
        and any reassignment of :attr:`reference` give a new module, so a new digest)."""
        c = self._ref_digest
        if c is None or c[0] is not self.reference:
            c = self._ref_digest = (self.reference, self.reference_digest())
        return c[1]

    # ------------------------------------------------------------------ data
    def load(self, root: Union[str, Path], update: int, *, expect_games: Optional[int] = None,
             expect_workers: Optional[int] = None, expect_rollout_id: Optional[str] = None,
             expect_config_digest: Optional[str] = None, expect_code_digest: Optional[str] = None,
             snapshot_sha256: Optional[Mapping[str, str]] = None, require_pins: bool = False,
             expect_policy_digest: Optional[str] = None) -> B.UpdateData:
        """``buffer.load_update`` with the acting weights' digest required (lag 0). The chain passes
        ``expect_config_digest = self.config_digest`` (the rollout coordinator also checks it). It is not the
        default because a measurement run may load shards recorded under the base config into a learner built with
        a ``policy.backbone_dtype`` override.
        The frozen-opponent checkpoint digests recorded per game are checked against the config pins
        (:func:`verify_shard_pins`; ``require_pins`` makes a missing record an error).

        ``expect_policy_digest`` (lag-1 overlap; None = the BASE behaviour above): the acting digest the round
        was played with; only the learner's own digest or :meth:`lag1_digest` (the parent of a lag-1 round) is
        accepted, anything else raises ``overlap.LagError`` (O1) before the shards are read."""
        if expect_policy_digest is None:
            data = B.load_update(root, update, expect_workers=expect_workers, expect_games=expect_games,
                                 expect_policy_digest=self.policy_digest(), expect_rollout_id=expect_rollout_id,
                                 expect_config_digest=expect_config_digest, expect_code_digest=expect_code_digest)
        else:
            from amini.ppo import overlap as OV
            own, par = self.policy_digest(), self.lag1_digest()
            ok = {own} | ({par} if par is not None else set()) | set(self.lag_digests())
            if str(expect_policy_digest) not in ok:
                raise OV.LagError(f"O1: round played by {str(expect_policy_digest)[:12]}; the learner accepts its own "
                                  f"{own[:12]} or its lag-1 parent {str(par)[:12] if par else 'none'}"
                                  + (f" (or {[d[:12] for d in self.lag_digests()[1:]]})" if self._lag2_prev else ""))
            data = B.load_update(root, update, expect_workers=expect_workers, expect_games=expect_games,
                                 expect_policy_digest=str(expect_policy_digest), expect_rollout_id=expect_rollout_id,
                                 expect_config_digest=expect_config_digest, expect_code_digest=expect_code_digest)
        self.last_pin_check = verify_shard_pins(data, self.cfg, snapshot_sha256=snapshot_sha256, require=require_pins)
        return data

    def _amp(self) -> contextlib.AbstractContextManager:
        if self.precision == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        if self.precision == "fp16":
            return torch.autocast(device_type=self.device.type, dtype=torch.float16)
        return contextlib.nullcontext()

    # -------------------------------------------------------------- optional compile
    def _compile_metrics(self) -> Dict[str, Any]:
        """Run-record fields for an explicitly enabled compiler (off adds no keys)."""
        if not self.compile_requested:
            return {}
        out = dict(self.compile_status)
        out["compile_mode"] = out.pop("mode", self.compile_mode)
        out["compile_dynamic"] = bool(out.pop("dynamic", self.compile_dynamic))
        return {f"learner_compile/{k}": v for k, v in out.items()}

    def _compile_counter(self) -> int:
        from torch._dynamo.utils import counters
        return int(counters["stats"].get("unique_graphs", 0))

    def _compile_smoke_pass(self, it: SeatItem, b: Mapping[str, Any], ca: Mapping[str, Any]
                            ) -> Tuple[Dict[str, torch.Tensor], Tuple[Any, ...]]:
        """No optimizer step and no .grad mutation; exercise policy + critic backward."""
        with torch.enable_grad():
            out = self._forward(self.policy, b, ca, [it])
            loss = out["z"].square().mean() + out["family"].square().mean()
            params = [p for p in self.policy.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            z = self.critic.step_features(b)
            rb = self._row_batch([it])
            with self._amp():
                values = self.critic.values_from_steps(z, rb["row_valid"], rb["pos"], rb["tq"])
            cg = torch.autograd.grad(values.square().mean(), self.critic.trainable_parameters(),
                                     allow_unused=True)
        result = {"policy/" + k: v.detach() for k, v in out.items() if torch.is_tensor(v)}
        result["critic"] = values.detach()
        with torch.no_grad():
            ref = self._forward(self.reference, b, ca, [it])
        result.update({"reference/" + k: v.detach() for k, v in ref.items() if torch.is_tensor(v)})
        return result, tuple(None if g is None else g.detach() for g in grads + cg)

    def _compile_configure(self, regions: Mapping[str, Any]) -> None:
        """Turn on the same packed token/cell regions used by IL, for all three models."""
        for model in (self.policy, self.reference):
            se, cd = model.step_encoder, model.cells
            self.__dict__["_compile_backups"].append((se, cd, se.packed, se.pad_multiple,
                                                       se.compiled, cd.compiled, cd.pad_rows, cd.p_regions))
            se.packed = True
            se.pad_multiple = 0 if self.compile_dynamic else 4096
            se.compiled = regions
            cd.compiled = regions
            cd.pad_rows = 0 if self.compile_dynamic else 64
            cd.p_regions = True
        se = self.critic.step_encoder
        self.__dict__["_compile_backups"].append((se, None, se.packed, se.pad_multiple,
                                                   se.compiled, None, None, None))
        se.packed = True
        se.pad_multiple = 0 if self.compile_dynamic else 4096
        se.compiled = regions

    def _compile_restore(self) -> None:
        for se, cd, packed, pad_multiple, compiled, cd_compiled, cd_pad_rows, p_regions in \
                self.__dict__.get("_compile_backups", []):
            se.packed, se.pad_multiple, se.compiled = packed, pad_multiple, compiled
            if cd is not None:
                cd.compiled, cd.pad_rows, cd.p_regions = cd_compiled, cd_pad_rows, p_regions
        self.__dict__["_compile_backups"] = []
        for module, fn in self.__dict__.get("_compile_methods", []):
            module.forward = fn
        self.__dict__["_compile_methods"] = []
        self.__dict__["_compiled_policy"] = None
        self.__dict__["_compiled_reference"] = None
        self.critic.set_compiled_temporal(None)

    def _compile_decide(self, data: B.UpdateData) -> None:
        """Compile and smoke-test once per learner block, before any optimizer step."""
        if self.__dict__.get("_compile_decided", True):
            return
        self.__dict__["_compile_decided"] = True
        start = time.perf_counter()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            it = self._item(0, next(iter(data.seat_games())))
            b = self._batch([it])
            ca = self._cell_args(b, [it])
            with torch.random.fork_rng(devices=devices):
                baseline, baseline_grad = self._compile_smoke_pass(it, b, ca)
                from amini.nn.aminiformer import compile_regions
                # torch >= 2.6: a compiled backward "donates" its buffers and then refuses
                # backward(retain_graph=True), which the learner's gradient guards use ("This backward function
                # was compiled with non-empty donated buffers which requires create_graph=False and
                # retain_graph=False ... or set torch._functorch.config.donated_buffer=False"). Older torch has no
                # such switch.
                try:
                    import torch._functorch.config as _fconf
                    if hasattr(_fconf, "donated_buffer"):
                        _fconf.donated_buffer = False
                except Exception:                                    # noqa: BLE001 (no switch: nothing to turn off)
                    pass
                backend = os.environ.get("PPO_LEARNER_COMPILE_BACKEND", "").strip() or None
                self.__dict__["_compile_graph_base"] = self._compile_counter()
                regions = compile_regions(self.compile_mode, self.compile_dynamic, backend=backend)
                self._compile_configure(regions)
                kw: Dict[str, Any] = {"dynamic": self.compile_dynamic}
                if backend:
                    kw["backend"] = backend
                elif self.compile_mode != "default":
                    kw["mode"] = self.compile_mode
                self.__dict__["_compile_kwargs"] = dict(kw)
                # Full-policy tracing hits dynamic boolean scatters and token-layout
                # .item() graph breaks. Compile the temporal tensor boundary instead;
                # packed step/cell regions above cover their forward and backward.
                for model in (self.policy, self.reference):
                    temporal = model.temporal
                    self.__dict__["_compile_methods"].append((temporal, temporal.forward))
                    temporal.forward = torch.compile(temporal.forward, **kw)
                self.critic.set_compiled_temporal(torch.compile(self.critic.temporal, **kw))
                got, got_grad = self._compile_smoke_pass(it, b, ca)
                # Under a half-precision backbone the compiled graph rounds differently from the eager one (an
                # output can differ by one half-precision step), so the fp32 tolerance would always send the run
                # back to eager. Outputs: a half-precision tolerance. Gradients: the relative error of each whole
                # tensor (single entries near zero differ by more than any elementwise bound in half precision).
                half = str(getattr(self.cfg.policy, "backbone_dtype", "fp32")) in ("fp16", "bf16")
                tol = {"atol": 1e-9, "rtol": 1e-9} if self._fdtype == torch.float64 else \
                    ({"atol": 2e-2, "rtol": 2e-2} if half else {"atol": 1e-5, "rtol": 1e-4})
                for key in baseline:
                    torch.testing.assert_close(got[key], baseline[key], **tol, msg=lambda m: f"{key}: {m}")
                for i, (a, b_) in enumerate(zip(baseline_grad, got_grad)):
                    if a is None or b_ is None:
                        nonempty = a if a is not None else b_
                        if nonempty is not None and bool(torch.count_nonzero(nonempty)):
                            raise RuntimeError(f"smoke gradient {i}: nonzero / None mismatch")
                        continue
                    if half and self._fdtype != torch.float64:
                        num, den = float((b_.double() - a.double()).norm()), float(a.double().norm())
                        if not (num <= COMPILE_HALF_GRAD_RTOL * den + 1e-12):
                            raise RuntimeError(f"smoke gradient {i}: relative error {num / max(den, 1e-30):.3g} > "
                                               f"{COMPILE_HALF_GRAD_RTOL} (half-precision backbone)")
                        continue
                    torch.testing.assert_close(b_, a, **tol)
            graphs = self._compile_counter() - self.__dict__["_compile_graph_base"]
            if graphs > COMPILE_MAX_GRAPHS:
                raise RuntimeError(f"torch.compile produced {graphs} graphs during smoke (limit {COMPILE_MAX_GRAPHS})")
            self.__dict__["_compile_smoke_graphs"] = graphs
            self.compile_status.update({"active": True, "smoke": "pass", "backend": backend or "inductor",
                                        "graphs": int(graphs), "recompiles": 0, "error": None})
        except Exception as exc:
            self._compile_restore()
            self.compile_status.update({"active": False, "smoke": "fallback", "backend": None,
                                        "error": f"{type(exc).__name__}: {exc}"[:2000]})
        self.compile_status["warmup_seconds"] = time.perf_counter() - start

    def _compile_check_budget(self) -> None:
        if not self.compile_requested or not self.compile_status.get("active"):
            return
        graphs = self._compile_counter() - self.__dict__.get("_compile_graph_base", 0)
        if graphs > COMPILE_MAX_GRAPHS:
            raise RuntimeError(f"torch.compile graph budget exceeded: {graphs} > {COMPILE_MAX_GRAPHS}")
        self.compile_status["graphs"] = int(graphs)
        self.compile_status["recompiles"] = max(0, int(graphs) - self.__dict__.get("_compile_smoke_graphs", 0))

    def _item(self, i: int, sg: B.SeatGame) -> SeatItem:
        dev = self.device
        r = sg.rows
        act = {k: torch.from_numpy(r[k].astype(np.int64)).to(dev) for k in ("family", "slot", "cell", "ability")}
        act["play_blocked"] = torch.from_numpy(r["play_blocked"].astype(bool)).to(dev)
        fam = act["family"]
        ex = sg.example()
        row = ex["tokens"]["row"]
        play_h = r["family"].astype(np.int64) == S.FAMILY_PLAY
        return SeatItem(index=i, sg=sg, held=ex if self.hold_examples else None, n=sg.n,
                        temperature=sg.temperature, action=act,
                        dt=torch.from_numpy(r["dt"].astype(np.float64)).to(dev),
                        mu=torch.from_numpy(r["mu"].astype(np.float64)).to(dev), row_type=fam,
                        play=fam == S.FAMILY_PLAY, card=r["card"].astype(np.int64), tok=item_tokens(ex),
                        row_pos=np.array(row["pos"], copy=True), row_tq=np.array(row["tq"], copy=True),
                        play_host=play_h, slot_host=np.where(play_h, r["slot"].astype(np.int64), -1))

    def _micro(self, items: Sequence[SeatItem]) -> List[List[SeatItem]]:
        """Micro-batches of whole seat-games (:func:`split_micro` with ``policy.micro_rows`` /
        ``policy.max_micro_tokens``, il_v7's split rule; one sampling temperature each)."""
        return split_micro(items, self.max_micro_rows, self.max_micro_tokens)

    def micro_stats(self, items: Sequence[SeatItem]) -> Dict[str, Any]:
        """The micro-batch settings and what they give on ``items`` (run record): caps,
        number of micro-batches, largest rows / padded tokens of one micro-batch, and the micro-batches above
        the token cap (single seat-games, which always run)."""
        mic = self._micro(items)
        cost = [IL.micro_cost(sum(it.n for it in m), max(getattr(it, "tok", (0, 0))[0] for it in m),
                              max(getattr(it, "tok", (0, 0))[1] for it in m)) for m in mic]
        cap = self.max_micro_tokens
        return {"micro_rows": self.max_micro_rows, "micro_token_cap": cap, "micro_batches": len(mic),
                "micro_rows_max": max((sum(it.n for it in m) for m in mic), default=0),
                "micro_tokens_max": max(cost, default=0),
                "micro_over_token_cap": int(sum(1 for c in cost if cap and c > cap))}

    def _batch(self, items: Sequence[SeatItem], host_index: Optional[bool] = None) -> Dict[str, Any]:
        c = _require(TK, NO_TOKENIZER).collate([it.example for it in items])
        b = _to_torch(c, self.device, self._fdtype)
        if self.host_index if host_index is None else host_index:
            hx = host_index_of(c, items)
            b[HOST_INDEX_KEY if "ridx" in hx else HOST_ROWS_KEY] = _hx_to(hx, self.device)
        return b

    def _host_index_decide(self, items: Sequence[SeatItem]) -> None:
        """Once per learner, in the first prepare pass (the pass's own module modes, no gradient): one micro-batch
        through the plain path and through the host-index path, policy forward and critic values, each from the
        same random state. The outputs must be EQUAL; if they are not, or anything raises, the learner goes on
        without the host index and says so in ``host_index_status`` (``learner_host_index/*`` in the metrics)."""
        if not self.host_index or self.__dict__.get("_host_index_decided"):
            return
        self.__dict__["_host_index_decided"] = True
        status: Dict[str, Any] = {"active": False, "check": "not run"}
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            micro = self._micro(items)[0]
            outs = []
            with torch.no_grad(), torch.random.fork_rng(devices=devices):
                for hx in (False, True):
                    b = self._batch(micro, host_index=hx)
                    torch.manual_seed(0)
                    o = self._forward(self.policy, b, self._cell_args(b, micro), micro)
                    with self._amp():
                        v = self.critic.values_from_steps(self.critic.step_features(b), b["row"]["row_valid"],
                                                          b["row"]["pos"], b["row"]["tq"],
                                                          sel=batch_row_sel(b) if hx else None)
                    outs.append((o, v))
            (o0, v0), (o1, v1) = outs
            bad = [k for k in OUT_ROW_KEYS + ("cells", "cell_play_index")
                   if (k in o0) != (k in o1) or (k in o0 and not torch.equal(o0[k], o1[k]))]
            if not torch.equal(v0, v1):
                bad.append("critic")
            status = {"active": not bad, "check": "equal" if not bad else "differs: " + ", ".join(bad)}
        except Exception as e:                                       # noqa: BLE001 (then the plain path)
            status = {"active": False, "check": f"error: {type(e).__name__}: {e}"[:300]}
        self.__dict__["host_index_status"] = status
        if not status["active"]:
            self.host_index = False
            print(f"learner: host index off ({status['check']})", file=sys.stderr, flush=True)

    def _host_index_metrics(self) -> Dict[str, Any]:
        st = self.__dict__.get("host_index_status")
        return {} if not st else {"learner_host_index/active": bool(st["active"]),
                                  "learner_host_index/check": str(st["check"])}

    def _cell_args(self, b: Mapping[str, Any], items: Sequence[SeatItem]) -> Dict[str, torch.Tensor]:
        """Teacher-forced stage P on the executed play rows (cells of the played slot only)."""
        rv = b["row"]["row_valid"]
        cell_rows = torch.zeros_like(rv, dtype=torch.bool)
        play_slot = torch.full(rv.shape, -1, dtype=torch.long, device=rv.device)
        for bi, it in enumerate(items):
            cell_rows[bi, :it.n] = it.play
            play_slot[bi, :it.n] = torch.where(it.play, it.action["slot"], torch.full_like(it.action["slot"], -1))
        return {"cell_rows": cell_rows, "play_slot": play_slot}

    def _forward(self, model: AminiFormer, b: Mapping[str, Any], ca: Mapping[str, torch.Tensor],
                 micro: Optional[Sequence[SeatItem]] = None) -> Dict[str, torch.Tensor]:
        """Forward on the stored tokens; with ``micro`` the rows' stored client slot-mask bits
        (``buffer.ROW_FIELDS['client_slot_ok']``) are applied to the slot logits exactly as the actor applied them
        (``client_mask.apply_slot_mask``, not part of this repository), so pi_theta, pi_prox, the KL reference and
        mu share one support. Rows without a client mask carry all-True bits: the output is then returned
        unchanged."""
        with self._amp():
            compiled = self.__dict__.get("_compiled_policy") if model is self.policy else \
                (self.__dict__.get("_compiled_reference") if model is self.reference else None)
            fn = compiled if compiled is not None else model
            out = fn(b, cell_rows=ca["cell_rows"], play_slot=ca["play_slot"], with_aux=False)
        self._compile_check_budget()
        if micro is None:
            return out
        ok = np.concatenate([_client_ok(it) for it in micro])
        return _require(CM, NO_CLIENT_MASK).apply_slot_mask(out, torch.from_numpy(ok).to(out["slot"].device))

    def _row_batch(self, items: Sequence[SeatItem]) -> Dict[str, torch.Tensor]:
        """row_valid / pos / tq only (the critic epochs read cached z_step, not tokens)."""
        tm = max(it.n for it in items)
        rv = torch.zeros(len(items), tm, dtype=torch.bool)
        pos = torch.zeros(len(items), tm, dtype=self._fdtype)
        tq = torch.zeros(len(items), tm, dtype=torch.long)
        for bi, it in enumerate(items):
            if it.row_pos is not None:                    # kept per item, the tokens are not rebuilt for this
                row = {"pos": it.row_pos, "tq": it.row_tq}
            else:
                row = it.example["tokens"]["row"]
            rv[bi, :it.n] = True
            pos[bi, :it.n] = torch.from_numpy(np.asarray(row["pos"])).to(self._fdtype)
            tq[bi, :it.n] = torch.from_numpy(np.asarray(row["tq"]).astype(np.int64))
        out = {"row_valid": rv.to(self.device), "pos": pos.to(self.device), "tq": tq.to(self.device)}
        if self.host_index:
            out["sel"] = rv.reshape(-1).nonzero(as_tuple=True)[0].to(self.device)
        return out

    def _check_input_rules(self, items: Sequence[SeatItem]) -> None:
        """Every seat-game was tokenized with the policy's own rule set (the model config's ``input_rules``;
        recorded per owner in ``GameMeta.extra['input_rules']`` by the rollout, absent = the default rule set). An
        actor or inference server that dropped the model's rule set would otherwise train the policy on another
        rule set's masks without any error."""
        TKN = _require(TK, NO_TOKENIZER)
        want = TKN.input_rules_of(self.policy)
        for it in items:
            rec = (it.sg.game.extra or {}).get("input_rules") or {}
            got = str(rec.get(str(it.sg.side), "") or "")
            if got != want:
                raise ValueError(f"game {it.sg.game.game_id} seat {it.sg.side}: rows tokenized with input_rules "
                                 f"{got!r}, the policy carries {want!r}")

    def _check_seat_weights(self, items: Sequence[SeatItem]) -> None:
        seat_table = {"ours": self.cfg.weights.seat_ours, "opponent": self.cfg.weights.seat_theta_opp,
                      "mirror": self.cfg.weights.seat_mirror}
        for it in items:
            if abs(it.sg.weight - seat_table[it.sg.role]) > 1e-12:
                raise ValueError(f"seat weight {it.sg.weight} of role {it.sg.role} != config {seat_table[it.sg.role]}")

    # ------------------------------------------------------------------ pre-pass
    @torch.no_grad()
    def prepare(self, data: B.UpdateData, *, identity: bool = False, lambda_tick: Optional[float] = None,
                overlap: Optional[OverlapCtx] = None) -> PreparedUpdate:
        """No-gradient pass at update start (module docstring step 2-3): π_prox, θ_k distribution, reference
        outputs, critic z_step and V; GAE, unified weights, one advantage normalisation; the mask audit
        (``mask_recompute_mismatch``), EV against the terminal outcome and the identity numbers.

        ``overlap`` (lag-1 overlap, None = BASE): in a lag-1 run every update also writes the ``is/`` metrics; at
        lag 1 the shards must be the parent's, the parent's forward gives ``mu_dlogp_max/*`` (the behaviour guard, O2),
        ``SeatItem.is_w`` = ρ and the win GAE is the V-trace form with traces c (``overlap.gae_vtrace_ragged``)."""
        self._compile_decide(data)
        ov = overlap
        if ov is not None and ov.lag >= 1:
            from amini.ppo import overlap as OV
            if ov.parent is None or data.policy_digest != ov.parent_digest:
                raise OV.LagError(f"update {data.update}: lag-1 shards played by {data.policy_digest[:12]}, the parent "
                                  f"is {str(ov.parent_digest)[:12]}")
            if identity:
                raise OV.LagError(f"update {data.update}: a lag-1 update can never be the identity update")
        elif data.policy_digest != self.policy_digest():
            raise ValueError(f"update {data.update}: shards were played by policy {data.policy_digest[:12]}, "
                             f"the learner holds {self.policy_digest()[:12]} (lag 0 only)")
        items = [self._item(i, sg) for i, sg in enumerate(data.seat_games())]
        if not items:
            raise ValueError(f"update {data.update}: no kept seat-games")
        self._check_seat_weights(items)
        self._check_input_rules(items)
        mismatch = sum(B.mask_recompute_mismatch(it.sg) for it in items)
        invalid_rows = 0
        dlogp = {name: 0.0 for name in L.ROW_TYPE_NAMES.values()}
        ref_kl0 = {h: 0.0 for h in L.HEADS}
        min_cell: Dict[str, float] = {}
        # per-head and per-card cell entropy of theta_k vs the KL reference on the same rows (sums, rows)
        ent_sum: Dict[str, List[float]] = {L.canonical_head(k): [0.0, 0.0, 0.0] for k in ENT_MASK}
        card_ent: Dict[str, List[float]] = {}
        self.policy.train()
        self.critic.train()
        self._host_index_decide(items)                   # once per learner, in this pass's own modes
        micros = self._micro(items)
        # ``o2_guard_every`` K > 1: the O2 guard's forward of the acting weights runs on one micro-batch in K (which
        # ones rotates with the update number) instead of all of them. It is a check, not an input of the update: ρ
        # and the traces use the μ the actor recorded; it is one of the three policy forwards of this pass. K = 1
        # (the default) checks every micro-batch.
        o2_every = max(1, int(getattr(self, "o2_guard_every", 1) or 1))
        o2_off = int(data.update) % o2_every if len(micros) >= o2_every else 0
        o2_rows = 0
        for mi, micro in enumerate(micros):
            b = self._batch(micro)
            ca = self._cell_args(b, micro)
            out = self._forward(self.policy, b, ca, micro)
            ref = self._forward(self.reference, b, ca, micro)
            outT = LP.apply_temperature(out, *micro[0].temperature)
            act, dt = _cat_action(micro), _cat(micro, "dt")
            terms = LP.executed_logp_terms(outT, dt, act)
            o2 = ov is not None and ov.lag >= 1 and (o2_every == 1 or mi % o2_every == o2_off)
            if o2:                                      # O2: the acting weights (θ_{u-2}) recomputed vs the actor
                par = self._forward(ov.parent, b, ca, micro)
                pterms = LP.executed_logp_terms(LP.apply_temperature(par, *micro[0].temperature), dt, act)
                o2_rows += int(sum(it.n for it in micro))
            lens = [it.n for it in micro]
            lm = LP.legal_masks(out)
            stored_slots = torch.from_numpy(np.concatenate([it.sg.rows["legal_slots"] for it in micro])).to(self.device)
            stored_forced = torch.from_numpy(np.concatenate([it.sg.rows["forced"] for it in micro])).to(self.device)
            stored_avail = torch.from_numpy(np.concatenate([it.sg.rows["ability_avail"] for it in micro])).to(self.device)
            bad = ((lm["slot_legal"] != stored_slots).any(-1) | (lm["forced"] != stored_forced)
                   | (lm["ability_avail"] != stored_avail))
            mismatch += int(bad.sum())
            ent = LP.head_entropy(out, dt, act)                # min cell entropy per card at theta_k
            if "ent_cell" in ent:
                cards = np.concatenate([it.card for it in micro])
                hc = ent["ent_cell"].double().cpu().numpy()
                for i in np.nonzero(ent["mask_cell"].cpu().numpy() & (cards >= 0))[0]:
                    key = str(int(cards[i]))
                    min_cell[key] = min(min_cell.get(key, math.inf), float(hc[i]))
            self._entropy_vs_ref(ent, LP.head_entropy(ref, dt, act), terms["valid"],
                                 np.concatenate([it.card for it in micro]), ent_sum, card_ent)
            rt = terms["executed_family"]           # every update: actor μ vs learner π_prox (lag 0)
            if ov is None:
                d = (terms["total"] - _cat(micro, "mu")).abs()
            else:
                mu_eff = terms["total"].detach() if ov.mu_override == "prox" else _cat(micro, "mu")
                d = None if (ov.lag >= 1 and not o2) else ((pterms["total"] if ov.lag >= 1 else terms["total"])
                                                           - mu_eff).abs()
            for code, name in L.ROW_TYPE_NAMES.items() if d is not None else ():
                sel = rt == code
                if bool(sel.any()):
                    x = float(d[sel].max())
                    dlogp[name] = x if not math.isfinite(x) else max(dlogp[name], x)
            if identity:
                kl = LP.head_kl(out, ref, dt, act)
                for k, m in KL_MASK.items():
                    if k in kl and bool(kl[m].any()):
                        h = L.canonical_head(k)
                        ref_kl0[h] = max(ref_kl0[h], float(kl[k][kl[m]].max()))
            z = self.critic.step_features(b)
            with self._amp():
                v = self.critic.values_from_steps(z, b["row"]["row_valid"], b["row"]["pos"], b["row"]["tq"],
                                                  sel=batch_row_sel(b))
            off = 0
            for it, po, rf in zip(micro, _split_out(out, lens), _split_out(ref, lens)):
                sl = slice(off, off + it.n)
                it.logp_prox = terms["total"][sl].detach()
                it.valid = terms["valid"][sl].detach()
                it.prox_out = {k: po[k] for k in OUT_ROW_KEYS}
                it.ref_out = rf
                it.critic_z = z[sl].clone()      # a copy: a slice of z_step keeps the token-level tensor alive
                it.value_old = v[sl].double().cpu().numpy()
                if o2:                                  # a row invalid under the learner or the parent counts once
                    invalid_rows += int((~(it.valid & pterms["valid"][sl])).sum())
                else:
                    invalid_rows += int((~it.valid).sum())
                if ov is not None:
                    it.ov_mu = mu_eff[sl].double().cpu().numpy()
                    it.ov_head_prox = torch.stack([terms[k][sl] for k in ("family", "slot", "cell", "ability")],
                                                  -1).double().cpu().numpy()
                    it.ov_head_mu = (it.ov_head_prox.copy() if ov.mu_override == "prox"
                                     else np.asarray(it.sg.rows["mu_terms"], np.float64))
                off += it.n
        # ---- GAE (pre-update critic), unified weights, one normalisation
        lengths = np.array([it.n for it in items], np.int64)
        values = np.concatenate([it.value_old for it in items])
        rewards = np.concatenate([G.episode_rewards(it.n, it.sg.outcome) for it in items])
        ticks = np.concatenate([it.sg.rows["tick"] for it in items]).astype(np.int64)
        lam = self.cfg.gae.lambda_tick if lambda_tick is None else float(lambda_tick)
        res = G.gae_ragged(rewards, values, ticks, lengths, gamma_tick=self.cfg.gae.gamma, lambda_tick=lam)
        ovg: Optional[Dict[str, Any]] = None
        if ov is not None:
            res, ovg = self._overlap_gae(ov, items, res, rewards, values, ticks, lengths, lam)
        w_pol, w_val = [], []
        for it in items:
            r = it.sg.rows
            rw = G.row_weights(r["stratum"], r["forced"], it.sg.weight, family=r["family"])
            wp = np.where(it.valid.cpu().numpy(), rw.policy, 0.0)     # invalid rows weight 0, counted
            w_pol.append(wp)
            w_val.append(rw.value)
        wp_all, wv_all = np.concatenate(w_pol), np.concatenate(w_val)
        adv, a_mean, a_std = G.normalize_advantages(res.advantages, wp_all)
        off = 0
        for it, wp, wv in zip(items, w_pol, w_val):
            sl = slice(off, off + it.n)
            it.w_policy = torch.from_numpy(wp).to(self.device)
            it.w_value = torch.from_numpy(wv).to(self.device)
            it.wp_sum, it.wv_sum = float(wp.sum()), float(wv.sum())
            it.adv = torch.from_numpy(adv[sl].copy()).to(self.device)
            it.ret = torch.from_numpy(res.returns[sl].copy()).to(self.device)
            off += it.n
        # ---- gate trend inputs from these rows (act rate, same-game play - wait advantage gap)
        trend = gate_trend_metrics([
            {"game": it.sg.game.game_id, "kind": it.sg.game.opp_kind, "family": it.sg.rows["family"],
             "live": wp > 0, "phi": it.sg.rows["phi"], "adv": adv[o:o + it.n],
             "ht": G.row_weights(it.sg.rows["stratum"], it.sg.rows["forced"], it.sg.weight,
                                 family=it.sg.rows["family"]).ht}
            for it, wp, o in zip(items, w_pol, np.concatenate([[0], np.cumsum(lengths)[:-1]]))])
        # ---- per-phase EV against the terminal outcome, per opponent kind
        ev = ev_by_phase([it.sg for it in items], values, wv_all)
        v_err = float((wv_all * (values - res.returns) ** 2).sum() / wv_all.sum())
        dc = data.drop_counter(self.cfg.monitor.drop_rate_warn, self.cfg.monitor.drop_rate_stop)
        metrics: Dict[str, Any] = {
            "ev_overall": ev["ev_overall"], "ev_last_third": ev["ev_last_third"],
            "ev_first_third": ev["ev_first_third"], "ev_middle_third": ev["ev_middle_third"],
            "ev_by_opponent": ev["ev_by_group"], "critic_mse_before": v_err,
            "mask_recompute_mismatch": int(mismatch), "learner_invalid_rows": int(invalid_rows),
            "drop_rate": dc.rate(), "drop_by_reason": dc.summary()["by_reason"],
            "adv_mean": a_mean, "adv_std": a_std, "rows": int(lengths.sum()), "seat_games": len(items),
            "games_kept": len(data.games), "games_dropped": len(data.dropped),
            "policy_weight_sum": float(wp_all.sum()), "min_cell_entropy": min_cell,
        }
        metrics.update({f"mu_dlogp_max/{k}": v for k, v in dlogp.items()})
        if ov is not None and ov.lag >= 1 and o2_every > 1:     # how much of the round the O2 guard looked at
            metrics["is/o2_guard_every"] = int(o2_every)
            metrics["is/o2_guard_rows_share"] = float(o2_rows) / float(max(1, sum(it.n for it in items)))
        metrics.update(self.micro_stats(items))
        metrics.update(trend)
        for h, (s_t, s_r, n) in ent_sum.items():
            metrics[f"entropy_theta/{h}"] = s_t / n if n else float("nan")
            metrics[f"entropy_ref/{h}"] = s_r / n if n else float("nan")
            metrics[f"entropy_vs_ref/{h}"] = s_t / s_r if n and s_r > 0 else float("nan")
        metrics["cell_entropy_by_card"] = {c: {"theta": s_t / n, "ref": s_r / n, "rows": int(n)}
                                           for c, (s_t, s_r, n) in sorted(card_ent.items())}
        for c, (s_t, s_r, n) in sorted(card_ent.items()):
            metrics[f"cell_entropy_vs_ref/{c}"] = s_t / s_r if s_r > 0 else float("nan")
        if ovg is not None:
            metrics.update(self._overlap_metrics(ov, ovg, items, wp_all, wv_all, lengths, lam, data.policy_digest))
        metrics.update(self._compile_metrics())
        metrics.update(self._host_index_metrics())
        return PreparedUpdate(data=data, items=items, adv_mean=a_mean, adv_std=a_std, metrics=metrics,
                              identity={"logp_max_abs_diff": dlogp, "kl_ref_per_head": ref_kl0})

    # ------------------------------------------------------------------ lag-1 overlap (ppo.overlap)
    def _overlap_gae(self, ov: OverlapCtx, items: Sequence[SeatItem], res: G.GaeResult, rewards: np.ndarray,
                     values: np.ndarray, ticks: np.ndarray, lengths: np.ndarray, lam: float
                     ) -> Tuple[G.GaeResult, Dict[str, Any]]:
        """Per-row (w, ρ, c) from π_prox and the (effective) μ; at lag 1 the V-trace win GAE replaces the plain one
        (``res``, kept for the shift diagnostics) and ``SeatItem.is_w`` = ρ. At lag 0 nothing the update uses
        changes: the weights only feed the ``is/`` metrics."""
        from amini.ppo import overlap as OV
        prox = np.concatenate([it.logp_prox.double().cpu().numpy() for it in items])
        mu = np.concatenate([it.ov_mu for it in items])
        forced = np.concatenate([np.asarray(it.sg.rows["forced"], bool) for it in items])
        valid = np.concatenate([it.valid.cpu().numpy() for it in items])
        w, rho, c = OV.is_weights(prox, mu, forced, rho_bar=ov.rho_bar, c_bar=ov.c_bar, neutral=~valid)
        out = {"w": w, "rho": rho, "c": c, "prox": prox, "mu": mu, "valid": valid, "plain": res}
        if ov.lag < 1:
            return res, out
        vt = OV.gae_vtrace_ragged(rewards, values, ticks, lengths, c, gamma_tick=self.cfg.gae.gamma,
                                  lambda_tick=lam)
        off = 0
        for it in items:
            it.is_w = torch.from_numpy(rho[off:off + it.n].copy()).to(self.device)
            off += it.n
        out["vt"] = vt
        return G.GaeResult(vt.advantages, vt.targets), out

    def _overlap_metrics(self, ov: OverlapCtx, g: Mapping[str, Any], items: Sequence[SeatItem], wp_all: np.ndarray,
                         wv_all: np.ndarray, lengths: np.ndarray, lam: float, behaviour: str) -> Dict[str, Any]:
        """The ``is/`` record of one update of a lag-1 run (``overlap.is_metrics``) plus ``is/lag`` and
        ``is/behaviour_digest``."""
        from amini.ppo import overlap as OV
        dt = np.concatenate([G.tick_intervals(it.sg.rows["tick"]) for it in items])
        rt = np.concatenate([np.asarray(it.sg.rows["family"], np.int64) for it in items])
        plain: G.GaeResult = g["plain"]
        if ov.lag >= 1:
            adv_vt, tgt_vt = g["vt"].advantages, g["vt"].targets
        else:
            adv_vt, tgt_vt = plain.advantages, plain.returns
        m = OV.is_metrics(w=g["w"], rho=g["rho"], c=g["c"], weight=wp_all, value_weight=wv_all, row_type=rt,
                          valid=g["valid"], logp_prox=g["prox"], mu=g["mu"], dt=dt, lengths=[int(n) for n in lengths],
                          lambda_tick=lam, rho_bar=ov.rho_bar, c_bar=ov.c_bar, adv_vt=adv_vt,
                          adv_plain=plain.advantages, tgt_vt=tgt_vt, tgt_plain=plain.returns,
                          head_prox=np.concatenate([it.ov_head_prox for it in items]),
                          head_mu=np.concatenate([it.ov_head_mu for it in items]))
        m["is/lag"] = int(ov.lag)
        m["is/behaviour_digest"] = str(behaviour)
        m["is/rho_bar"], m["is/c_bar"] = float(ov.rho_bar), float(ov.c_bar)
        if ov.mu_override is not None:
            m["is/mu_override"] = ov.mu_override
        return m

    @staticmethod
    def _entropy_vs_ref(ent: Mapping[str, torch.Tensor], ent_ref: Mapping[str, torch.Tensor], valid: torch.Tensor,
                        cards: np.ndarray, ent_sum: Dict[str, List[float]], card_ent: Dict[str, List[float]]) -> None:
        """Add theta_k's and the reference's per-row entropies on each head's own valid rows (row masks of
        ``logprob.head_entropy``; rows where either is non-finite are left out) and, for the cells head, per
        played card (``card >= 0``)."""
        v = valid.detach().bool()
        for k, mk in ENT_MASK.items():
            if k not in ent or k not in ent_ref:
                continue
            x, y = ent[k].detach().double(), ent_ref[k].detach().double()
            sel = ent[mk].bool() & v & torch.isfinite(x) & torch.isfinite(y)
            acc = ent_sum[L.canonical_head(k)]
            acc[0] += float(x[sel].sum())
            acc[1] += float(y[sel].sum())
            acc[2] += float(sel.sum())
            if k == "ent_cell":
                idx = np.nonzero(sel.cpu().numpy() & (cards >= 0))[0]
                xs, ys = x.cpu().numpy(), y.cpu().numpy()
                for i in idx:
                    a = card_ent.setdefault(str(int(cards[i])), [0.0, 0.0, 0.0])
                    a[0] += float(xs[i])
                    a[1] += float(ys[i])
                    a[2] += 1.0

    # ------------------------------------------------------------------ critic only (warm-up)
    @torch.no_grad()
    def critic_values(self, sgs: Sequence[B.SeatGame], *, chunk: int = 64) -> np.ndarray:
        """tanh V of the current critic on the seat-games' rows (concatenated in order), in micro-batches,
        reading tokens ``chunk`` seat-games at a time."""
        self.critic.eval()
        out: List[np.ndarray] = []
        try:
            for s0 in range(0, len(sgs), int(chunk)):
                lite = []
                for i, sg in enumerate(sgs[s0:s0 + int(chunk)]):
                    ex = sg.example()
                    lite.append(SeatItem(index=i, sg=sg, held=ex, n=sg.n, temperature=sg.temperature, action={},
                                         dt=torch.empty(0), mu=torch.empty(0), row_type=torch.empty(0),
                                         play=torch.empty(0), card=np.empty(0), tok=item_tokens(ex)))
                vals: Dict[int, np.ndarray] = {}
                for micro in self._micro(lite):
                    b = self._batch(micro)
                    z = self.critic.step_features(b)
                    with self._amp():
                        v = self.critic.values_from_steps(z, b["row"]["row_valid"], b["row"]["pos"], b["row"]["tq"],
                                                          sel=batch_row_sel(b))
                    v = v.double().cpu().numpy()
                    off = 0
                    for it in micro:
                        vals[it.index] = v[off:off + it.n]
                        off += it.n
                out.extend(vals[i] for i in range(len(lite)))
        finally:
            self.critic.train()
        return np.concatenate(out) if out else np.zeros(0)

    @torch.no_grad()
    def prepare_critic(self, data: B.UpdateData, *, lambda_tick: Optional[float] = None) -> PreparedUpdate:
        """Critic-only pre-pass (warm-up): no policy or reference forward, no μ check, no identity.
        Per seat-game: the frozen step encoder's z_step (cached for the critic epochs) and the pre-update V;
        per-tick GAE λ-returns, the unified value weights (HT × seat, forced rows included) and the batch's
        per-phase EV against the terminal outcome. Policy weights / advantages are left at zero."""
        self._compile_decide(data)
        if data.policy_digest != self.policy_digest():
            raise ValueError(f"update {data.update}: shards were played by policy {data.policy_digest[:12]}, "
                             f"the learner holds {self.policy_digest()[:12]} (lag 0 only)")
        items = [self._item(i, sg) for i, sg in enumerate(data.seat_games())]
        if not items:
            raise ValueError(f"update {data.update}: no kept seat-games")
        self._check_seat_weights(items)
        self._check_input_rules(items)
        self.critic.train()
        for micro in self._micro(items):
            b = self._batch(micro)
            z = self.critic.step_features(b)
            with self._amp():
                v = self.critic.values_from_steps(z, b["row"]["row_valid"], b["row"]["pos"], b["row"]["tq"],
                                                  sel=batch_row_sel(b))
            off = 0
            for it in micro:
                sl = slice(off, off + it.n)
                it.critic_z = z[sl].clone()      # a copy: a slice of z_step keeps the token-level tensor alive
                it.value_old = v[sl].double().cpu().numpy()
                it.valid = torch.ones(it.n, dtype=torch.bool, device=self.device)
                off += it.n
        lengths = np.array([it.n for it in items], np.int64)
        values = np.concatenate([it.value_old for it in items])
        rewards = np.concatenate([G.episode_rewards(it.n, it.sg.outcome) for it in items])
        ticks = np.concatenate([it.sg.rows["tick"] for it in items]).astype(np.int64)
        lam = self.cfg.gae.lambda_tick if lambda_tick is None else float(lambda_tick)
        res = G.gae_ragged(rewards, values, ticks, lengths, gamma_tick=self.cfg.gae.gamma, lambda_tick=lam)
        w_val = [G.row_weights(it.sg.rows["stratum"], it.sg.rows["forced"], it.sg.weight,
                               family=it.sg.rows["family"]).value for it in items]
        off = 0
        for it, wv in zip(items, w_val):
            sl = slice(off, off + it.n)
            it.w_value = torch.from_numpy(wv).to(self.device)
            it.wv_sum = float(wv.sum())
            it.w_policy = torch.zeros(it.n, dtype=torch.float64, device=self.device)
            it.adv = torch.zeros(it.n, dtype=torch.float64, device=self.device)
            it.ret = torch.from_numpy(res.returns[sl].copy()).to(self.device)
            off += it.n
        wv_all = np.concatenate(w_val)
        ev = ev_by_phase([it.sg for it in items], values, wv_all)
        metrics: Dict[str, Any] = {
            "ev_overall": ev["ev_overall"], "ev_last_third": ev["ev_last_third"],
            "ev_first_third": ev["ev_first_third"], "ev_middle_third": ev["ev_middle_third"],
            "ev_by_opponent": ev["ev_by_group"],
            "critic_mse_before": float((wv_all * (values - res.returns) ** 2).sum() / wv_all.sum()),
            "rows": int(lengths.sum()), "seat_games": len(items), "games_kept": len(data.games),
            "games_dropped": len(data.dropped), "lambda_tick": lam}
        metrics.update(self.micro_stats(items))
        metrics.update(self._compile_metrics())
        metrics.update(self._host_index_metrics())
        return PreparedUpdate(data=data, items=items, adv_mean=0.0, adv_std=0.0, metrics=metrics, identity={})

    def warmup_update(self, data: B.UpdateData, fixed_set: FixedSetEV, *,
                      lambda_tick: Optional[float] = None) -> WarmupResult:
        """One critic warm-up update: :meth:`prepare_critic` + :meth:`train_critic` on a fresh
        batch (no policy step, no KL, no identity requirement; ``policy.lr`` is irrelevant), then the fixed-set
        EV (:class:`FixedSetEV`) and ``phase_rules.critic_warmup_status`` over the whole history. Warm-up
        updates are numbered 1, 2, ... in their own shard root and run before any PPO update; one after the
        stop raises. The state (count, EV history, fixed-set digest, status) is saved by :meth:`save`, so a
        chained block continues it (:meth:`from_checkpoint`) and :meth:`from_il` ``critic_from`` takes the
        result into the PPO run."""
        pr = _require(PR, NO_PHASE_RULES)                  # the warm-up status comes from the phase rules
        if self.steps_total != 0 or self.next_update != 1:
            raise RuntimeError("the critic warm-up runs before any policy update")
        st = self.warmup.get("status") or {}
        if st.get("stop"):
            raise RuntimeError(f"the critic warm-up already stopped: {st.get('reason')}")
        u = int(data.update)
        if u != int(self.warmup["done"]) + 1:
            raise ValueError(f"warm-up update {u} given, expected {int(self.warmup['done']) + 1}")
        if self.warmup["fixed_set_digest"] not in (None, fixed_set.digest):
            raise ValueError("the fixed EV set changed during the warm-up (one fixed recorded set)")
        if fixed_set.learner is not self:
            raise ValueError("fixed_set was built for another learner")
        _reset_peak(self.device)
        prep = self.prepare_critic(data, lambda_tick=lambda_tick)
        crit = self.train_critic(prep)
        ev = fixed_set()
        self.warmup["done"] = u
        self.warmup["fixed_set_digest"] = fixed_set.digest
        self.warmup["history"].append({k: _finite(ev[k]) for k in
                                       ("ev_overall", "ev_first_third", "ev_middle_third", "ev_last_third")})
        status = pr.critic_warmup_status(self.cfg, self.warmup["history"])
        early = getattr(self.cfg.critic.warmup, "early_stop_min_updates", None)
        if not status.stop and early is not None:          # option, off by default
            status = pr.critic_warmup_early_status(
                self.cfg, self.warmup["history"], min_updates=int(early),
                why=f"critic.warmup.early_stop_min_updates = {int(early)}") or status
        if not status.stop:                                # option, off by default: the warm-up stopped paying
            status = pr.critic_warmup_plateau_status(self.cfg, self.warmup["history"]) or status
        self.warmup["status"] = _jsonable(dataclasses.asdict(status))
        rec: Dict[str, Any] = dict(prep.metrics)
        rec.update(crit)
        rec.update({f"fixed/{k}": ev[k] for k in ("ev_overall", "ev_first_third", "ev_middle_third", "ev_last_third")})
        rec["fixed/ev_by_opponent"] = ev["ev_by_group"]
        rec.update({"warmup_update": u, "fixed_set_digest": fixed_set.digest, "warmup_stop": status.stop,
                    "warmup_passed": status.passed, "warmup_reason": status.reason})
        rec["peak_mem_bytes"], rec["peak_mem_source"] = peak_memory(self.device)
        self.last_metrics = rec
        return WarmupResult(update=u, metrics=rec, ev_fixed=ev, status=status)

    def warmup_early_stop(self, why: str) -> Optional[Any]:
        """End a running warm-up now if the pass rule already holds (``phase_rules.critic_warmup_early_status``;
        called by the run driver on an operator's request). Returns the stop status, or None when nothing changed
        (no update yet, already stopped, or a pass line not met: the warm-up then simply continues)."""
        st = self.warmup.get("status") or {}
        if st.get("stop") or not self.warmup.get("history"):
            return None
        status = _require(PR, NO_PHASE_RULES).critic_warmup_early_status(self.cfg, self.warmup["history"],
                                                                          why=str(why))
        if status is not None:
            self.warmup["status"] = _jsonable(dataclasses.asdict(status))
        return status

    # ------------------------------------------------------------------ one minibatch
    def _scaler_flush(self) -> None:
        """Close a minibatch whose gradients were unscaled but not stepped (skipped, early stop, monitor stop):
        GradScaler wants one ``update`` per ``unscale_``; it also lowers the scale after non-finite gradients."""
        if self._scaler_pending:
            self.scaler.update()
            self._scaler_pending = False

    def _opt_step(self) -> None:
        """The policy optimiser step of the minibatch in ``.grad`` (already unscaled and clipped)."""
        if self.scaler.is_enabled():
            self.scaler.step(self.opt)                   # no step if unscale_ saw a non-finite gradient
            self.scaler.update()
            self._scaler_pending = False
        else:
            self.opt.step()

    def accumulate_gradients(self, items: Sequence[SeatItem], *, beta: float,
                             coef: Union[float, Mapping[str, float]] = 0.0, cell_cards: Sequence[str] = (),
                             kl_grad_ratio: bool = False) -> Optional[MinibatchStats]:
        """Forward + backward of one minibatch of whole seat-games over its micro-batches, every policy term
        divided by the minibatch-wide surrogate denominator D, so the accumulated gradient
        equals the full-minibatch gradient however the rows are split. Gradients are left in ``.grad`` (zeroed
        first); no optimiser step. Returns None when D = 0 (an all-forced minibatch).

        The ratio uses the actor's sampling temperature (``apply_temperature``); the KL anchor and the entropy
        are taken on the untempered policy (trained seats normally sample at T = 1). ``coef`` /
        ``cell_cards``: entropy coefficients (config names) and the cards whose cells bonus is on
        (``phase_rules.cell_entropy_bonus``); other cards' cell rows get weight 0 in the bonus.

        ``kl_grad_ratio`` (diagnostic, one minibatch per update): the per-head KL and surrogate gradients
        are summed over ALL micro-batches of this minibatch before the norms are taken, so the ratio is the
        whole minibatch's whatever the caps. Cost on that minibatch only: one extra
        backward per head per micro-batch and (1 + KL heads) gradient copies of the trainable parameters."""
        D = float(sum(it.wp_sum for it in items))
        if not D > 0.0:
            return None
        self._scaler_flush()                             # a minibatch that was unscaled but not stepped
        self.opt.zero_grad(set_to_none=True)
        params = [p for p in self.policy.parameters() if p.requires_grad]
        p = self.cfg.policy
        coef_all_zero = (not isinstance(coef, Mapping) and float(coef) == 0.0) or (
            isinstance(coef, Mapping) and all(float(v) == 0.0 for v in coef.values()))
        cards = {str(c) for c in cell_cards}
        rows: Dict[str, List[Any]] = {k: [] for k in ("logp", "prox", "w", "play", "adv", "gate_kl", "row_type")}
        ent_rows: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {k: [] for k in ENT_MASK}
        loss_sum, kl_sum = 0.0, 0.0
        beta_used: Dict[str, float] = {}
        grads: Optional[Dict[str, List[Optional[torch.Tensor]]]] = {} if kl_grad_ratio else None
        surr_key = "\0surrogate"
        Dt = torch.tensor(D, dtype=torch.float64, device=self.device)
        for micro in self._micro(items):
            b = self._batch(micro)
            ca = self._cell_args(b, micro)
            out = self._forward(self.policy, b, ca, micro)
            act, dt = _cat_action(micro), _cat(micro, "dt")
            w, A, prox = _cat(micro, "w_policy"), _cat(micro, "adv"), _cat(micro, "logp_prox")
            logp = LP.executed_logp(LP.apply_temperature(out, *micro[0].temperature), dt, act)
            if micro[0].is_w is None:
                surr = L.clipped_surrogate(logp, prox, A, w, denom=Dt, eps=p.clip_eps, dual_clip=p.dual_clip)
            else:                                           # lag 1: rho on the per-row clipped objective
                if any(it.is_w is None for it in micro):
                    raise ValueError("lag-1 minibatch with rows lacking an IS weight")
                rho = _cat(micro, "is_w")
                surr = L.clipped_surrogate(logp, prox, A, w, denom=Dt, eps=p.clip_eps, dual_clip=p.dual_clip,
                                           is_weight=rho)
                rows.setdefault("rho", []).append(rho)
            kl = LP.head_kl(out, _join_out([it.ref_out for it in micro]), dt, act)
            pen = L.kl_penalty(_head_rows(kl, w, KL_MASK), Dt, beta=beta)
            for h, b_ in pen.beta.items():                  # effective values: what the KL term actually used
                beta_used[h] = b_ if beta_used.get(h, b_) == b_ else math.nan
            ent = LP.head_entropy(out, dt, act)
            er = _head_rows(ent, w, ENT_MASK)
            loss = surr + pen.loss
            if not coef_all_zero:
                bonus_rows = dict(er)
                if cards and "ent_cell" in bonus_rows:
                    keep = torch.from_numpy(np.isin(np.concatenate([it.card for it in micro]).astype(str),
                                                    sorted(cards))).to(self.device)
                    x, wc = bonus_rows["ent_cell"]
                    bonus_rows["ent_cell"] = (x, torch.where(keep, wc, torch.zeros_like(wc)))
                loss = loss + L.entropy_bonus(bonus_rows, Dt, coef=coef)
            if grads is not None:
                _add_grads(grads, surr_key, surr, params)
                for h, t in pen.terms.items():
                    _add_grads(grads, h, t, params)
            self.scaler.scale(loss).backward()
            loss_sum += float(loss.detach())
            kl_sum += float(sum(float(pen.own_row_mean[h] * pen.own_weight_sum[h]) for h in pen.terms))
            with torch.no_grad():
                gk = LP.head_kl(out, _join_out([it.prox_out for it in micro]), dt, act)["kl_family"]
            rows["logp"].append(logp.detach())
            rows["prox"].append(prox)
            rows["w"].append(w)
            rows["play"].append(_cat(micro, "play"))
            rows["adv"].append(A)
            rows["gate_kl"].append(gk.detach())
            rows["row_type"].append(_cat(micro, "row_type"))
            for k in ENT_MASK:
                if k in er:
                    ent_rows[k].append((er[k][0].detach(), er[k][1].detach()))
                else:                                            # no play rows in this micro-batch
                    z = torch.zeros_like(w)
                    ent_rows[k].append((z, z))
        cat = {k: torch.cat(v) for k, v in rows.items()}
        cat["entropy"] = {k: (torch.cat([x for x, _ in v]), torch.cat([y for _, y in v])) for k, v in ent_rows.items()}
        if self.scaler.is_enabled():                     # true gradients from here on (finite check, clip, step)
            self.scaler.unscale_(self.opt)
            self._scaler_pending = True
        finite = math.isfinite(loss_sum) and all(bool(torch.isfinite(q.grad).all())
                                                 for q in params if q.grad is not None)
        ratios = _grad_ratios(grads, surr_key) if grads is not None else {}
        return MinibatchStats(denom=D, loss=loss_sum, finite=finite, rows=cat, kl_sum=kl_sum, kl_grad_ratio=ratios,
                              beta_used=beta_used)

    # ------------------------------------------------------------------ the update
    def _identity(self, prep: PreparedUpdate, first: MinibatchStats, replay_bitwise_ok: Optional[bool]
                  ) -> Dict[str, Any]:
        """Update-0 identity: raises IdentityError (and records the stop) on failure."""
        th = self.thresholds
        r = first.rows
        live = r["w"] > 0
        rep = L.identity_check(r["logp"], r["prox"], r["row_type"], prep.identity["kl_ref_per_head"],
                               mask=live, ratio_tol=th.identity_ratio_tol, kl_tol=th.identity_kl_tol)
        fails = list(rep.failures)
        for name, v in prep.identity["logp_max_abs_diff"].items():
            tol = th.identity_logp_tol.get(name, 1e-3)
            if not (math.isfinite(v) and v < tol):
                fails.append(f"logp/{name}")
        report = {"logp_max_abs_diff": dict(prep.identity["logp_max_abs_diff"]),
                  "ratio_max_abs_dev": rep.max_abs_ratio_dev, "kl_ref_per_head": rep.max_ref_kl,
                  "replay_bitwise_ok": replay_bitwise_ok, "backbone": self.precision, "rows": rep.rows,
                  "failures": fails}
        acts = self.monitor.check_identity(report)
        report["monitor_actions"] = [a.to_json() for a in acts]
        if fails or self.monitor.stopped:
            raise IdentityError(f"PPO update-0 identity failed: {fails or self.monitor.stop_reason}; {report}")
        return report

    def _check_extra(self, extra: Mapping[str, Any], prep_keys: Iterable[str] = ()) -> None:
        """``extra_metrics`` may not overwrite a key the learner produces (passing ``mask_recompute_mismatch = 0``
        would hide a failed audit). Raised before any step."""
        owned = set(OWNED_KEYS) | (set(prep_keys) - set(SHARED_KEYS))
        bad = sorted(k for k in extra if k in owned or str(k).startswith(OWNED_PREFIXES))
        if bad:
            raise ValueError(f"extra_metrics may not set learner-produced keys: {bad}")

    def _merge_extra(self, rec: Dict[str, Any], extra: Mapping[str, Any], data: B.UpdateData) -> None:
        """Pre-step record: the caller's keys (``extra``: rollout and run-driver metrics) next to the learner's.
        ``drop_rate`` must agree with the learner's (same games); ``refusals_code19`` from the shard counters and
        from the rollout are merged by max (the stop direction); ``forced_fraction`` is the rollout's key and falls
        back to the rollout's formula ((forced + skipped) / queries of the 'ours' seats) on the kept seats."""
        x = dict(extra)
        if "drop_rate" in x:
            a, b = _finite(x["drop_rate"]), _finite(rec.get("drop_rate"))
            if a is None or b is None or abs(a - b) > 1e-12:
                raise ValueError(f"extra drop_rate {x['drop_rate']!r} != the learner's {rec.get('drop_rate')!r}")
            x.pop("drop_rate")
        ignored = [k for k in ("drop_by_reason",) if k in x]
        for k in ignored:
            x.pop(k)
        c = data.counters
        if "refusals_code19" in c:
            v = c["refusals_code19"]
            if "refusals_code19" in x and _finite(x["refusals_code19"]) is not None:
                v = max(float(v), float(x["refusals_code19"]))
            x["refusals_code19"] = v
        if "forced_fraction" not in x:
            ours = (c.get("by_role") or {}).get("ours") or {}
            if ours.get("queries"):
                x["forced_fraction"] = (float(ours.get("forced", 0)) + float(ours.get("skipped", 0))) \
                    / float(ours["queries"])
        rec.update(x)
        if ignored:
            rec["extra_ignored"] = ignored

    @staticmethod
    def _post_step_placeholders(value: float) -> Dict[str, Any]:
        """The keys only the steps produce (finite placeholders for the strict preflight, NaN for a pre-step
        stop record)."""
        d: Dict[str, Any] = {k: value for k in POST_STEP_KEYS + POST_STEP_TRACK_B_KEYS}
        d.update({f"entropy/{h}": value for h in L.HEADS})
        d["early_stopped"] = ()
        return d

    def _preflight(self, update: int, rec: Mapping[str, Any], late_keys: Sequence[str] = ()) -> None:
        """Strict monitor: refuse before any step when an expected key is absent or non-finite among those known
        after the pre-pass (the learner's ``ev_*`` / ``drop_rate`` / ``mask_recompute_mismatch`` and the caller's
        keys). Post-step keys are checked after the steps (:meth:`update`)."""
        if not self.monitor.strict:
            return
        probe = dict(rec)
        probe.update(self._post_step_placeholders(0.0))
        probe.update({k: False for k in late_keys if k not in probe})   # flags that arrive with late_metrics
        probe["update"] = update
        missing = self.monitor.missing_update_keys(probe)
        if missing:
            raise M.MetricsMissingError(update, missing)

    def _monitor_stop(self, name: str, reason: str, update: int, detail: Mapping[str, Any]) -> List[M.Action]:
        """Record a learner-detected stop in the monitor (state + dashboard) with ``Monitor.stop``."""
        return self.monitor.stop(name, reason, update, **_jsonable(dict(detail)))

    def _actor_guard(self, u: int, prep: PreparedUpdate, identity: bool) -> None:
        """Lag-0 guard on every update: the actor's μ must equal the learner's π_prox per
        row type within the update-0 tolerance, and no executed action may be illegal for the learner. The
        shard digest covers the weights only, so a stale server or a served gate bias is caught here. On the
        identity update the μ comparison is left to the full update-0 identity."""
        th = self.thresholds
        fails: List[str] = []
        d = {k[len("mu_dlogp_max/"):]: v for k, v in prep.metrics.items() if k.startswith("mu_dlogp_max/")}
        if not identity:
            for name, v in d.items():
                tol = th.identity_logp_tol.get(name, 1e-3)
                if not (math.isfinite(v) and v < tol):
                    fails.append(f"logp/{name}={v:.3g} >= {tol:g}")
        n_bad = int(prep.metrics.get("learner_invalid_rows", 0))
        if n_bad > 0:
            fails.append(f"learner_invalid_rows={n_bad}")
        if fails:
            detail = {"mu_dlogp_max": d, "learner_invalid_rows": n_bad, "failures": fails,
                      "policy_digest": prep.data.policy_digest}
            self._monitor_stop("actor_learner_mismatch", "; ".join(fails), u, detail)
            raise IdentityError(f"update {u}: actor and learner disagree ({'; '.join(fails)}); the shards are not "
                                f"theta_k's on-policy data (stale server or served gate bias?)")

    def _prestep_stop(self, u: int, rec: Mapping[str, Any]) -> bool:
        """True when a monitor stop is already decided by the pre-pass numbers (mask audit, code-19 refusals,
        masked-legal refusals, drop rate). Evaluated on a throw-away copy of the monitor
        (no dashboard), so the real monitor sees each update once."""
        probe = copy.deepcopy(self.monitor)
        probe.dashboard_path = None
        probe.strict = False
        r = dict(rec)
        r.update(self._post_step_placeholders(float("nan")))
        r["update"] = u
        acts = probe.observe_update(r)
        return any(a.kind == "stop" and a.monitor in PRESTEP_STOP_MONITORS for a in acts)

    def _observe(self, rec: Mapping[str, Any]) -> List[M.Action]:
        """``Monitor.observe_update`` with strict mode off (the caller decides what a missing key means)."""
        strict = self.monitor.strict
        self.monitor.strict = False
        try:
            return self.monitor.observe_update(rec)
        finally:
            self.monitor.strict = strict

    def update(self, data: B.UpdateData, *, replay_bitwise_ok: Optional[bool] = None,
               extra_metrics: Optional[Mapping[str, Any]] = None, cell_bonus: Optional[PR.CellBonus] = None,
               identity: Optional[bool] = None, lambda_tick: Optional[float] = None,
               actor_guard: bool = True, late_metrics: Optional[Callable[[], Mapping[str, Any]]] = None,
               late_keys: Sequence[str] = ()) -> UpdateResult:
        """One PPO update on ``data`` (module docstring). ``data.update`` must be :attr:`next_update`.
        ``identity`` defaults to True before the first optimiser step of the run (update 0);
        ``replay_bitwise_ok`` is the rollout's bitwise replay result for that check. ``extra_metrics`` carries the
        caller's monitor keys (refusals, behaviour rates, forced fraction, games per hour, drift flag); a
        learner-produced key there raises ValueError. ``cell_bonus``: ``phase_rules.cell_entropy_bonus`` result
        (coefficients + cards); default the config's ``entropy.coef``. ``actor_guard`` (default on; a training run
        never turns it off): the per-update μ vs π_prox / illegal-action guard, off only for timing runs whose
        learner precision differs from the recording server's.

        Before any step: IdentityError (update-0 identity, or μ ≠ π_prox / illegal executed actions on any
        update; monitor stopped), MetricsMissingError (strict monitor, a pre-pass key absent or non-finite), or
        a monitor stop decided by the pre-pass numbers (returned with ``steps = 0``, ``stopped = True``).
        After the steps, a strict monitor missing a post-step key raises MetricsMissingError with the finished
        :class:`UpdateResult` as ``.result`` (bookkeeping done, so the barrier can save). Returns the monitor's
        actions (the caller executes stop / rollback).

        Guards and diagnostics (none changes a step): effective values (:meth:`_effective_start`,
        raising :class:`EffectiveValueError` before any work; the lr every step reads from the param groups and
        the beta every KL term used, ``effective/*``; :meth:`_effective_end` after the actions) and the KL
        gradient guard on the first minibatch (beta > 0 and theta != reference: ``kl_grad_ratio/total`` must
        exceed ``monitor.kl_grad_ratio_min``, else the monitor stops before that minibatch's step);
        ``tiny_ratio_*``; ``gate_trend/*`` and ``entropy_*`` / ``cell_entropy_vs_ref/*``
        (:meth:`prepare`); ``probe_flip/*`` on the fixed probe rows (:meth:`probe_argmax`, re-based after a
        weight swap, :meth:`_probe_start`) and, from the same probe forward after the steps, the entropy versus
        the IL baseline (``entropy_vs_il/*``, ``cell_entropy_vs_il/*``, :func:`entropy_vs_baseline`).

        Lag-1 overlap (``policy.rollout_lag = 1``, a block-boundary runtime switch): the
        lag of ``data`` is decided here (:meth:`lag_of`; ``overlap.LagError`` = O1 before any work); every update
        of such a run snapshots its start weights in memory (the parent of the next round) and writes the ``is/``
        metrics (``is/first_step_kl`` among them); lag-1 data is learned with ρ on the surrogate and the V-trace
        GAE; O3 can stop before the first step (``overlap_is_truncation``). With ``rollout_lag = 0`` this calls
        :meth:`_update_impl` with ``lag = 0`` and no overlap context: the BASE update, statement for statement."""
        kw = dict(replay_bitwise_ok=replay_bitwise_ok, extra_metrics=extra_metrics, cell_bonus=cell_bonus,
                  identity=identity, lambda_tick=lambda_tick, actor_guard=actor_guard)
        if late_metrics is not None:                 # (off: the keywords are not passed at all)
            kw.update(late_metrics=late_metrics, late_keys=tuple(late_keys))
        cfg_lag = int(self.cfg.policy.rollout_lag)
        if cfg_lag < 1:
            return self._update_impl(data, lag=0, parent_state=None, mu_override=None, **kw)
        from amini.ppo import overlap as OV
        own = self.policy_digest()
        prev = self._lag1_prev
        # rollout_lag = 2: the update before the previous one, kept only while it leads to it
        prev2 = self._lag2_prev if (cfg_lag >= 2 and prev is not None) else None
        if prev2 is None:
            lag = OV.lag_of(data.policy_digest, own, prev[1] if prev else None, prev[2] if prev else None)
        else:
            lag = OV.lag_of_chain(data.policy_digest, own, [(prev[1], prev[2]), (prev2[1], prev2[2])])
        parent_state = None if lag == 0 else (prev[0] if lag == 1 else prev2[0])
        snap = {k: v.detach().to("cpu", copy=True) for k, v in self.policy.state_dict().items()}
        keep2 = prev if (cfg_lag >= 2 and prev is not None and prev[2] == own) else None
        self._lag1_prev = None                           # an update that raises leaves no parent: next is a fill
        self._lag2_prev = None
        try:
            res = self._update_impl(data, lag=lag, parent_state=parent_state, mu_override=None, own_digest=own,
                                    **kw)
        except M.MetricsMissingError as e:               # bookkeeping done (the barrier saves): keep the chain
            r = getattr(e, "result", None)
            if r is not None:
                self._lag1_prev = (snap, own, str(r.metrics["policy_digest_after"]))
                self._lag2_prev = keep2
            raise
        self._lag1_prev = (snap, own, str(res.metrics["policy_digest_after"]))
        self._lag2_prev = keep2
        return res

    # ------------------------------------------------------------------ lag-1 overlap: version helpers
    def lag1_digest(self) -> Optional[str]:
        """The acting digest a lag-1 round must carry (the weights at the start of the previous update), or None
        when no lag-1 round can be learned now (lag-0 run, first update of a block, after a rollback / reload /
        weight swap, or a previous update that raised)."""
        prev = self._lag1_prev
        if prev is None or self.policy_digest() != prev[2]:
            return None
        return prev[1]

    def lag_digests(self) -> List[str]:
        """Every acting digest a round may carry besides the learner's own, nearest first: the lag-1 parent
        (:meth:`lag1_digest`) and, in a ``rollout_lag = 2`` run whose last two updates form an unbroken chain, the
        weights at the start of the update before the previous one. Empty when no lagged round can be learned."""
        prev, own = self._lag1_prev, self.policy_digest()
        if prev is None or own != prev[2]:
            return []
        out = [prev[1]]
        prev2 = self._lag2_prev
        if int(self.cfg.policy.rollout_lag) >= 2 and prev2 is not None and prev2[2] == prev[1]:
            out.append(prev2[1])
        return out

    def lag_of(self, data_digest: str) -> int:
        """0 / 1 (/ 2 in a ``rollout_lag = 2`` run) for shards played by ``data_digest`` (``overlap.lag_of`` /
        ``overlap.lag_of_chain``; raises ``overlap.LagError``, O1)."""
        from amini.ppo import overlap as OV
        prev = self._lag1_prev
        prev2 = self._lag2_prev if (int(self.cfg.policy.rollout_lag) >= 2 and prev is not None) else None
        if prev2 is None:
            return OV.lag_of(str(data_digest), self.policy_digest(), prev[1] if prev else None,
                             prev[2] if prev else None)
        return OV.lag_of_chain(str(data_digest), self.policy_digest(), [(prev[1], prev[2]), (prev2[1], prev2[2])])

    def lag1_state(self) -> Optional[Tuple[Dict[str, torch.Tensor], str, str]]:
        """The in-memory lag-1 record (start weights on the CPU, start digest, end digest of the previous update)
        or None; a remote learner sends it with a request (no checkpoint holds it)."""
        return self._lag1_prev

    def set_lag1_state(self, snap: Mapping[str, torch.Tensor], own: str, end: str) -> None:
        """Set the lag-1 record (:meth:`lag1_state`): on the remote worker from the request, and on the driver's
        learner after it adopted a remote result (its own start weights, as a local update keeps them). The lag-2
        record does not travel: a remote update leaves the next round learnable at lag 0 or 1 only."""
        self._lag1_prev = ({k: v.detach().to("cpu") for k, v in snap.items()}, str(own), str(end))
        self._lag2_prev = None

    def drop_parent(self) -> None:
        """Forget the lag-1 parent (the driver calls it after ``rollback_to`` or any weight swap; the digest check
        in :meth:`lag_of` catches a forgotten call too)."""
        self._lag1_prev = None
        self._lag2_prev = None

    def _overlap_ctx(self, lag: int, parent_state: Optional[Mapping[str, torch.Tensor]], mu_override: Optional[str],
                     own: Optional[str]) -> OverlapCtx:
        from amini.ppo import overlap as OV
        if int(lag) not in (0, 1, 2):
            raise OV.LagError(f"O5: lag {lag} (only 0, 1 or 2)")
        if mu_override not in (None, "prox"):
            raise ValueError(f"mu_override must be None or 'prox', got {mu_override!r}")
        own = self.policy_digest() if own is None else str(own)
        parent, pd = None, None
        if int(lag) >= 1:                                # the parent = the weights that PLAYED the round (O2)
            if parent_state is None:
                raise OV.LagError("a lagged update needs the weights that played its round")
            if self._parent_mod is None:                 # built lazily, once: a deep copy draws no RNG
                self._parent_mod = module_copy(self.policy)     # never the policy's compiled forward
                for q in self._parent_mod.parameters():
                    q.requires_grad_(False)
                    q.grad = None
            self._parent_mod.load_state_dict(dict(parent_state), strict=True)
            self._parent_mod.train(self.policy.training)
            parent, pd = self._parent_mod, state_dict_digest(self._parent_mod.state_dict())
        p = self.cfg.policy
        return OverlapCtx(lag=int(lag), own_digest=own, parent=parent, parent_digest=pd, mu_override=mu_override,
                          rho_bar=float(p.is_ratio_cap_lag1), c_bar=float(p.gae_trace_cap_lag1))

    @staticmethod
    def _k3_isw(rows: Sequence[Mapping[str, Any]]) -> float:
        """is/joint_k3_isw: Σ Wρ k3(log π_θ/π_prox) / Σ Wρ over the stepped minibatches (ρ = 1 at lag 0)."""
        lr_ = torch.cat([r["logp"] for r in rows]).double() - torch.cat([r["prox"] for r in rows]).double()
        w = torch.cat([r["w"] for r in rows]).double()
        if all("rho" in r for r in rows):
            w = w * torch.cat([r["rho"] for r in rows]).double()
        live = w > 0
        x = torch.where(live, lr_, torch.zeros_like(lr_))
        return float(L.weighted_mean(L.k3(x), w))

    def _stop_metrics(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        """The epoch's numbers that the early-stop limits are compared with (``losses.early_stop``).

        On rows that carry importance weights (an update at lag 1 or 2) ``joint_k3`` is the importance-weighted
        one (:meth:`_k3_isw`, Σ Wρ k3 / Σ Wρ). The plain ``joint_k3`` weighs every row by W alone, also the rows
        the importance weight removes from the loss: a row to which π_prox gives almost no behaviour probability
        (ρ at the clamp) can drift while it stays at about zero probability, and its huge ratio π_θ / π_prox would
        trip the early stop although the row carries no weight in the loss. ``play_k3`` and ``gate_kl`` are
        unchanged; the record keeps the plain ``joint_k3`` and ``is/joint_k3_isw`` both. Without ``rho`` on the
        rows (lag 0) nothing changes."""
        m = self._metrics(rows)
        if rows and all("rho" in r for r in rows):
            m = dict(m)
            m["joint_k3"] = self._k3_isw(rows)
        return m

    def _update_impl(self, data: B.UpdateData, *, lag: int, parent_state: Optional[Mapping[str, torch.Tensor]] = None,
                     mu_override: Optional[str] = None, replay_bitwise_ok: Optional[bool] = None,
                     extra_metrics: Optional[Mapping[str, Any]] = None, cell_bonus: Optional[PR.CellBonus] = None,
                     identity: Optional[bool] = None, lambda_tick: Optional[float] = None,
                     actor_guard: bool = True, own_digest: Optional[str] = None,
                     late_metrics: Optional[Callable[[], Mapping[str, Any]]] = None,
                     late_keys: Sequence[str] = ()) -> UpdateResult:
        """The one update path (:meth:`update` docstring). ``lag = 0`` with no ``parent_state`` / ``mu_override``
        in a ``rollout_lag = 0`` run is the BASE update; otherwise an :class:`OverlapCtx` is built. Test seam:
        ``lag = 1``, ``parent_state`` = the learner's own state, ``mu_override = 'prox'`` gives
        w = ρ = c = 1 exactly and must equal the lag-0 update bit for bit."""
        u = int(data.update)
        if u != self.next_update:
            raise ValueError(f"update {u} given, the learner expects update {self.next_update}")
        if self.monitor.stopped:
            raise RuntimeError(f"monitor stopped: {self.monitor.stop_reason}")
        _require(PR, NO_PHASE_RULES)                     # every update ends with phase_rules.grad_clip_raise
        self._warmup_gate()
        extra = dict(extra_metrics or {})
        self._check_extra(extra)
        _reset_peak(self.device)
        eff = self._effective_start(u)                   # effective values, before any work (EffectiveValueError)
        ov: Optional[OverlapCtx] = None
        if lag != 0 or parent_state is not None or mu_override is not None or int(self.cfg.policy.rollout_lag) >= 1:
            ov = self._overlap_ctx(lag, parent_state, mu_override, own_digest)
        # prepare() refuses shards whose digest is not the learner's acting weights, so the shard digest stands in
        # for a second hash of the policy here; at lag 1 the learner's own digest
        beh = str(data.policy_digest) if ov is None or ov.lag == 0 else ov.own_digest
        theta_is_ref = beh == self._reference_digest_cached()
        identity = (self.steps_total == 0) if identity is None else bool(identity)
        _t_prep0 = time.perf_counter()
        prep = self.prepare(data, identity=identity, lambda_tick=lambda_tick, overlap=ov)
        _t_prep1 = time.perf_counter()
        self._check_extra(extra, prep.metrics)
        if actor_guard:
            self._actor_guard(u, prep, identity)
        base: Dict[str, Any] = dict(prep.metrics)
        self._merge_extra(base, extra, data)
        self._preflight(u, base, late_keys)
        cfg, p = self.cfg, self.cfg.policy
        beta, lr = self.monitor.beta, self.monitor.lr
        if ov is not None:                               # O3 / O4; alarms go to the driver
            from amini.ppo import overlap as OV
            ov_alarms, ov_stop = OV.check_learner(base)
            base["is/alarms"] = list(ov_alarms)
            base["is/prestep_stop"] = ov_stop is not None
            if ov_stop is not None:                      # no optimiser step on far off-policy data
                detail = {k: v for k, v in base.items() if str(k).startswith(("is/trunc", "is/ess", "is/lag"))}
                # the driver drops this update without a barrier and resumes from barrier u-1 at lag 0: the run
                # does not stop (the dashboard record says so)
                detail["lag_fallback"] = ("no optimiser step; the block ends without a barrier for this update and "
                                          "the next block re-collects it at lag 0 (the run continues)")
                stop_acts = self._monitor_stop(OV.STOP_NAME, ov_stop, u, detail)
                rec = dict(base)
                rec.update(self._post_step_placeholders(float("nan")))
                rec.update({"update": u, "steps": 0, "skipped_minibatches": 0, "grad_clip": self.grad_clip,
                            "grad_clipped_frac": float("nan"), "grad_norm_max": float("nan"), "lr": lr,
                            "beta": beta, "backbone": self.precision, "epochs_done": 0, "policy_digest_before": beh,
                            "prestep_stop": True})
                rec.update(eff)
                actions = list(stop_acts) + list(self._observe(rec))
                actions = actions + self._effective_end(u, rec, lr, beta, actions)
                return self._finish_update(u, rec, actions, None, 0, 0, (), float("nan"))
        if self._prestep_stop(u, base):
            rec = dict(base)
            rec.update(self._post_step_placeholders(float("nan")))
            rec.update({"update": u, "steps": 0, "skipped_minibatches": 0, "grad_clip": self.grad_clip,
                        "grad_clipped_frac": float("nan"), "grad_norm_max": float("nan"), "lr": lr, "beta": beta,
                        "backbone": self.precision, "epochs_done": 0, "policy_digest_before": beh,
                        "prestep_stop": True})
            rec.update(eff)
            actions = self._observe(rec)
            actions = list(actions) + self._effective_end(u, rec, lr, beta, actions)
            return self._finish_update(u, rec, actions, None, 0, 0, (), float("nan"))
        self._make_probe(prep.items, u)                  # argmax probe: chosen once, then fixed
        probe_rec = self._probe_start(beh, theta_is_ref)
        coef: Union[float, Mapping[str, float]] = dict(cfg.entropy.coef)
        cards: Sequence[str] = ()
        if cell_bonus is not None:
            coef, cards = dict(cell_bonus.coef), tuple(cell_bonus.cards)
        thresholds = {"joint_k3": cfg.early_stop.joint_k3, "play_k3": cfg.early_stop.play_k3,
                      "gate_kl": cfg.early_stop.gate_kl}
        items = prep.items
        mbs = int(p.minibatch_seat_games)
        steps = skipped = clipped = 0
        early: Tuple[str, ...] = ()
        id_report: Optional[Dict[str, Any]] = None
        all_rows: List[Dict[str, Any]] = []
        epoch_metrics: List[Dict[str, float]] = []
        kl_num = kl_den = 0.0
        ratios: Dict[str, float] = {}
        grad_norms: List[float] = []
        lr_read: List[float] = []                        # effective values: lr each step read from the param groups
        beta_read: List[float] = []                      # effective values: beta each KL head used
        kl_check = "not measured"
        guard_actions: List[M.Action] = []
        last_ep_rows: List[Dict[str, Any]] = []
        self.policy.train()
        stop = False
        for epoch in range(int(p.epochs)):
            order = np.random.default_rng([self.seed, u, 1, epoch]).permutation(len(items))
            ep_rows: List[Dict[str, Any]] = []
            for s in range(0, len(items), mbs):
                mb = [items[i] for i in order[s:s + mbs]]
                st = self.accumulate_gradients(mb, beta=beta, coef=coef, cell_cards=cards,
                                               kl_grad_ratio=(steps == 0 and skipped == 0))
                if st is None:
                    continue
                if identity and id_report is None:
                    try:
                        id_report = self._identity(prep, st, replay_bitwise_ok)
                    except IdentityError:
                        self.opt.zero_grad(set_to_none=True)
                        raise
                self.monitor.observe_minibatch(u, st.finite)
                if not st.finite:
                    self.opt.zero_grad(set_to_none=True)
                    skipped += 1
                    if self.monitor.stopped:
                        stop = True
                        break
                    continue
                beta_read.extend(st.beta_used.values())
                if st.kl_grad_ratio:
                    ratios = st.kl_grad_ratio
                    kl_check = self._kl_grad_check(beta, ratios, theta_is_ref)
                    if kl_check.startswith("fail"):          # no step on a dead KL anchor
                        self.opt.zero_grad(set_to_none=True)
                        guard_actions = self._monitor_stop("kl_grad_ratio", kl_check, u,
                                                           {"kl_grad_ratio": dict(ratios), "beta": beta})
                        stop = True
                        break
                # early stop BEFORE the step (as in Spinning Up): the minibatch's KL is measured at the current θ
                # and a minibatch that trips the limit is not stepped
                kl_num += st.kl_sum
                kl_den += st.denom
                ep_rows.append(st.rows)
                all_rows.append(st.rows)
                tripped = L.early_stop(self._stop_metrics(ep_rows), thresholds)
                if tripped:
                    self.opt.zero_grad(set_to_none=True)
                    early = tripped
                    break
                params = [q for q in self.policy.parameters() if q.requires_grad]
                gn = float(torch.nn.utils.clip_grad_norm_(params, self.grad_clip))
                grad_norms.append(gn)
                clipped += int(gn > self.grad_clip)
                lr_read.extend(self._set_lr(lr))         # what this step uses, read back
                self._opt_step()
                steps += 1
                self.steps_total += 1
            if ep_rows:
                epoch_metrics.append(self._metrics(ep_rows))
                last_ep_rows = ep_rows
            if early or stop:
                break
        self._scaler_flush()
        self.opt.zero_grad(set_to_none=True)
        _t_pol1 = time.perf_counter()
        critic = self.train_critic(prep) if not stop else {}
        if os.environ.get("PPO_LEARNER_TIMING"):        # diagnostics only: one line on stderr, nothing recorded
            print(json.dumps({"learner_timing": {"update": int(u), "prepare_s": round(_t_prep1 - _t_prep0, 2),
                                                 "policy_s": round(_t_pol1 - _t_prep1, 2),
                                                 "critic_s": round(time.perf_counter() - _t_pol1, 2),
                                                 "steps": int(steps), "seat_games": len(items)}}),
                  file=sys.stderr, flush=True)
        # ---- argmax probe flips after the steps
        flips: Dict[str, Any] = dict(probe_rec)
        if self.probe is not None and not stop:
            il = self.probe.get("il_entropy")
            cur, ent = self._probe_eval(entropy=il is not None)
            flips.update(argmax_flips(self.probe_prev, cur))
            if il is not None and ent is not None:       # entropy vs the IL baseline on the fixed rows
                flips.update(entropy_vs_baseline(ent, il, np.concatenate(self.probe["card"])))
            self.probe_prev = cur
            self.probe_prev_digest = None                # set from policy_digest_after below
        if late_metrics is not None:
            # Metrics of a check that ran BESIDE this update (the driver's drift check on a frozen copy of
            # the pre-update policy). They are extra metrics like any other, only later: same ownership check,
            # same place in the record. ``late_keys`` named the flags among them for the pre-step check.
            late = dict(late_metrics())
            self._check_extra(late, prep.metrics)
            base.update(late)
        # ---- monitor record: the learner's values win over any residual extra key
        rec: Dict[str, Any] = dict(base)
        if all_rows:
            rec.update(self._metrics(all_rows))
        else:
            rec.update({"clip_frac": float("nan"), "play_clip_frac": float("nan")})
        if ov is not None:
            rec["is/joint_k3_isw"] = self._k3_isw(all_rows) if all_rows else float("nan")
        if epoch_metrics:
            last = epoch_metrics[-1]
            rec.update({f"last_epoch/{k}": v for k, v in last.items()})
        rec["kl_ref_total"] = kl_num / kl_den if kl_den > 0 else float("nan")
        rec.update({f"kl_grad_ratio/{h}": v for h, v in ratios.items()})
        rec["early_stopped"] = early
        rec.update(critic)
        frac = clipped / steps if steps else float("nan")
        rec.update({"update": u, "steps": steps, "skipped_minibatches": skipped, "grad_clip": self.grad_clip,
                    "grad_clipped_frac": frac, "grad_norm_max": max(grad_norms) if grad_norms else float("nan"),
                    "lr": lr, "beta": beta, "backbone": self.precision, "epochs_done": len(epoch_metrics),
                    "policy_digest_before": beh})
        # ---- diagnostics record: effective values, KL gradient guard, tiny ratios, probe flips
        rec.update(eff)
        step_fails = self._effective_steps(rec, lr, beta, lr_read, beta_read)
        rec["kl_grad_check"] = kl_check
        if all_rows:
            rec.update(self._tiny(all_rows))
        if last_ep_rows:
            rec.update({f"last_epoch/{k}": v for k, v in self._tiny(last_ep_rows).items()})
        rec.update(flips)
        missing: List[str] = []
        if stop:
            actions: List[M.Action] = list(guard_actions)
        else:
            if self.monitor.strict and not self.monitor.stopped:
                missing = self.monitor.missing_update_keys(rec)
            actions = self._observe(rec) if missing else self.monitor.observe_update(rec)
        actions = list(actions) + self._effective_end(u, rec, lr, beta, actions)
        if step_fails and not self.monitor.stopped:
            actions += self._monitor_stop("effective_values", "; ".join(step_fails), u, {"failures": step_fails})
        res = self._finish_update(u, rec, actions, id_report, steps, skipped, early, frac)
        if self.probe is not None and not stop:
            self.probe_prev_digest = str(rec["policy_digest_after"])     # the weights probe_prev was taken at
        if missing:
            err = M.MetricsMissingError(u, missing)
            err.result = res                                  # type: ignore[attr-defined]
            raise err
        return res

    def _finish_update(self, u: int, rec: Dict[str, Any], actions: Sequence[M.Action],
                       id_report: Optional[Dict[str, Any]], steps: int, skipped: int, early: Tuple[str, ...],
                       frac: float) -> UpdateResult:
        """Bookkeeping shared by every path that observed (or skipped) the update: grad-clip history and the
        one-time raise (``phase_rules.grad_clip_raise``), next update index, last metrics."""
        self.clipped_frac_history.append(frac)
        raised = _require(PR, NO_PHASE_RULES).grad_clip_raise(self.cfg, self.clipped_frac_history,
                                                              self.grad_clip_raised)
        if raised is not None:
            self.grad_clip, self.grad_clip_raised = float(raised), True
        rec["policy_digest_after"] = self.policy_digest()
        rec["peak_mem_bytes"], rec["peak_mem_source"] = peak_memory(self.device)
        self.next_update = u + 1
        self.last_metrics = rec
        return UpdateResult(update=u, metrics=rec, actions=list(actions), identity=id_report, steps=steps,
                            skipped_minibatches=skipped, early_stopped=early, stopped=self.monitor.stopped)

    def _probe_start(self, digest: str, theta_is_ref: bool) -> Dict[str, Any]:
        """Argmax probe before the steps: take the probe argmax of theta_k when there is none yet, or again when
        the weights changed since it was taken (``probe_prev_digest`` != the acting digest: a rollback, a stop
        after some steps, a checkpoint without the digest), so the flip rate never counts a weight swap
        (``probe_rebased``). On the probe's first evaluation before any optimiser step (theta = reference = the
        base IL model) the same forward stores the IL entropy baseline (``probe['il_entropy']``, kept for the
        whole run, also after promotions)."""
        if self.probe is None:
            return {}
        rebased = self.probe_prev is not None and self.probe_prev_digest != str(digest)
        if self.probe_prev is not None and not rebased:
            return {"probe_rebased": False, "probe_il_baseline": self.probe.get("il_entropy") is not None}
        want_il = (self.probe.get("il_entropy") is None and self.steps_total == 0 and theta_is_ref
                   and all(k in self.probe for k in PROBE_ROW_KEYS))
        self.probe_prev, ent = self._probe_eval(entropy=want_il)
        self.probe_prev_digest = str(digest)
        if want_il and ent is not None:
            self.probe["il_entropy"], self.probe["il_digest"] = ent, str(digest)
        return {"probe_rebased": rebased, "probe_il_baseline": self.probe.get("il_entropy") is not None}

    def _kl_grad_check(self, beta: float, ratios: Mapping[str, float], theta_is_ref: bool) -> str:
        """KL-gradient guard: with beta > 0 and theta != the reference (digests differ at the update start; on the
        first update theta IS the reference, so the KL and its gradient are 0 by construction) the whole KL
        anchor's gradient norm over the surrogate's (``kl_grad_ratio/total``, first minibatch) must exceed
        ``monitor.kl_grad_ratio_min``. Returns 'pass', 'skipped: ...' or 'fail: ...'."""
        if not beta > 0.0:
            return "skipped: beta is 0"
        if theta_is_ref:
            return "skipped: theta == reference (KL gradient 0 by construction)"
        r = ratios.get("total")
        lim = float(self.thresholds.kl_grad_ratio_min)
        if r is None or not float(r) > lim:
            return f"fail: KL / surrogate gradient-norm ratio {r!r} <= {lim:g} with beta {beta:g} (KL anchor dead)"
        return "pass"

    def _effective_steps(self, rec: Dict[str, Any], lr: float, beta: float, lr_read: Sequence[float],
                         beta_read: Sequence[float]) -> List[str]:
        """Read-back during the steps: the lr every optimiser step took from the param groups and the beta
        every KL head used, against the planned values (records ``effective/lr_steps_*``,
        ``effective/beta_used_*``); returns the failures."""
        fails = []
        if lr_read:
            rec["effective/lr_steps_min"], rec["effective/lr_steps_max"] = min(lr_read), max(lr_read)
            if not all(_close(x, lr) for x in lr_read):
                fails.append(f"steps used lr {sorted(set(lr_read))}, planned {lr!r}")
        if beta_read:
            rec["effective/beta_used_min"], rec["effective/beta_used_max"] = min(beta_read), max(beta_read)
            if not all(_close(x, beta) for x in beta_read):
                fails.append(f"KL terms used beta {sorted(set(beta_read))}, planned {beta!r}")
        if fails:
            rec["effective/step_failures"] = fails
        return fails

    def _tiny(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        """Tiny ratios: ``losses.tiny_ratio_shares`` over the rows of several minibatches (each row as measured
        before its minibatch's step), with ``monitor.tiny_log_ratio``."""
        cat = {k: torch.cat([r[k] for r in rows]) for k in ("logp", "prox", "w", "row_type")}
        return L.tiny_ratio_shares(cat["logp"], cat["prox"], cat["w"], cat["row_type"],
                                   cut=float(self.cfg.monitor.tiny_log_ratio))

    def _metrics(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        """``losses.update_metrics`` over the concatenated rows of several minibatches (unified weights)."""
        cat = {k: torch.cat([r[k] for r in rows]) for k in ("logp", "prox", "w", "play", "adv", "gate_kl")}
        ent = {k: (torch.cat([r["entropy"][k][0] for r in rows]), torch.cat([r["entropy"][k][1] for r in rows]))
               for k in ENT_MASK}
        p = self.cfg.policy
        return L.update_metrics(cat["logp"], cat["prox"], cat["w"], cat["play"], advantage=cat["adv"],
                                gate_kl=cat["gate_kl"], entropy=ent, eps=p.clip_eps, dual_clip=p.dual_clip)

    # ------------------------------------------------------------------ critic
    def train_critic(self, prep: PreparedUpdate) -> Dict[str, float]:
        """``critic.epochs`` epochs of the separate critic on the update's λ-returns (pre-update V):
        weighted MSE with the value weights (HT × seat, forced rows included), minibatch-wide denominator,
        own Adam and clip. Non-finite minibatches are skipped and counted."""
        c, p = self.cfg.critic, self.cfg.policy
        items = prep.items
        params = self.critic.trainable_parameters()
        self._apply_critic_hparams()
        self.critic.train()
        losses, skipped, steps = [], 0, 0
        for epoch in range(int(c.epochs)):
            order = np.random.default_rng([self.seed, prep.data.update, 2, epoch]).permutation(len(items))
            for s in range(0, len(items), int(p.minibatch_seat_games)):
                mb = [items[i] for i in order[s:s + int(p.minibatch_seat_games)]]
                Dv = float(sum(it.wv_sum for it in mb))
                if not Dv > 0:
                    continue
                Dt = torch.tensor(Dv, dtype=torch.float64, device=self.device)
                self.critic_opt.zero_grad(set_to_none=True)
                tot = 0.0
                for micro in self._micro(mb):
                    rb = self._row_batch(micro)
                    with self._amp():
                        v = self.critic.values_from_steps(_cat(micro, "critic_z"), rb["row_valid"], rb["pos"], rb["tq"],
                                                          sel=rb.get("sel"))
                    loss = L.value_loss(v, _cat(micro, "ret"), _cat(micro, "w_value"), denom=Dt)
                    self.critic_scaler.scale(loss).backward()
                    tot += float(loss.detach())
                scaled = self.critic_scaler.is_enabled()
                if scaled:
                    self.critic_scaler.unscale_(self.critic_opt)
                ok = math.isfinite(tot) and all(bool(torch.isfinite(q.grad).all()) for q in params if q.grad is not None)
                if not ok:
                    self.critic_opt.zero_grad(set_to_none=True)
                    if scaled:
                        self.critic_scaler.update()
                    skipped += 1
                    continue
                torch.nn.utils.clip_grad_norm_(params, float(c.grad_clip))
                if scaled:
                    self.critic_scaler.step(self.critic_opt)
                    self.critic_scaler.update()
                else:
                    self.critic_opt.step()
                steps += 1
                losses.append(tot)
        self.critic_opt.zero_grad(set_to_none=True)
        num = den = 0.0
        with torch.no_grad():                         # weighted MSE on the whole update after training
            for micro in self._micro(items):
                rb = self._row_batch(micro)
                with self._amp():
                    v = self.critic.values_from_steps(_cat(micro, "critic_z"), rb["row_valid"], rb["pos"], rb["tq"],
                                                      sel=rb.get("sel"))
                wv = _cat(micro, "w_value")
                num += float((wv * (v.double() - _cat(micro, "ret")) ** 2).sum())
                den += float(wv.sum())
        return {"critic_loss_first": losses[0] if losses else float("nan"),
                "critic_loss_last": losses[-1] if losses else float("nan"),
                "critic_mse_after": num / den if den > 0 else float("nan"),
                "critic_steps": steps, "critic_skipped": skipped}

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def kl_to_reference(self, prep: PreparedUpdate) -> Dict[str, float]:
        """Per-head KL(π_θ ‖ ref) of the CURRENT policy on the prepared rows, summed over each head's rows with
        the unified weight and divided by the update's D (the surrogate normalisation); ``total`` is their sum."""
        num = {h: 0.0 for h in L.HEADS}
        D = float(sum(it.wp_sum for it in prep.items))
        for micro in self._micro(prep.items):
            b = self._batch(micro)
            out = self._forward(self.policy, b, self._cell_args(b, micro), micro)
            kl = LP.head_kl(out, _join_out([it.ref_out for it in micro]), _cat(micro, "dt"), _cat_action(micro))
            for k, (x, w) in _head_rows(kl, _cat(micro, "w_policy"), KL_MASK).items():
                num[L.canonical_head(k)] += float((w * x).sum())
        res = {h: v / D for h, v in num.items()}
        res["total"] = sum(res.values())
        return res

    # ------------------------------------------------------------------ checkpoint
    def save(self, path: Union[str, Path], *, monitor_path: Optional[Union[str, Path]] = None) -> Path:
        """Barrier checkpoint (il_v7 layout; acting weights under ``model_ema``; the monitor state is written at
        the same barrier, inside the checkpoint and to ``monitor_path`` when given).
        Atomic: temporary file, fsync, ``os.replace``."""
        path = Path(path)
        sd = {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()}
        payload = {
            "format": CKPT_FORMAT, "schema": S.SCHEMA_ID, "trainer": LEARNER_VERSION,
            "model_config": self.policy.cfg.to_dict(), "train_config": None, "step": self.steps_total,
            "metrics": _jsonable(self.last_metrics), "generation": f"ppo-u{self.next_update - 1:06d}",
            "model_ema": sd, "model_raw": sd, "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "schema_describe": S.describe(), "tables": None, "resume": None, "init": self.source or None,
            "ppo": {
                "version": LEARNER_VERSION, "next_update": self.next_update, "steps_total": self.steps_total,
                "seed": self.seed, "policy_digest": state_dict_digest(sd),
                "reference": {"model_config": self.reference.cfg.to_dict(),
                              "state": {k: v.detach().cpu().clone() for k, v in self.reference.state_dict().items()},
                              "digest": self.reference_digest()},
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
                "critic_optimizer": self.critic_opt.state_dict(),
                "monitor": self.monitor.state_dict(), "monitor_baselines": _baselines_dict(self.monitor.base),
                "monitor_dashboard": None if self.monitor.dashboard_path is None else str(self.monitor.dashboard_path),
                "config_digest": self.config_digest,
                "compat": self.compat, "compat_digest": self.compat_digest, "grad_clip": self.grad_clip,
                "grad_clip_raised": self.grad_clip_raised, "clipped_frac_history": list(self.clipped_frac_history),
                "precision": self.precision, "source": self.source,
                "config": self.cfg.to_dict(), "config_changes": list(self.config_changes),
                "warmup": copy.deepcopy(self.warmup), "critic_warmup": copy.deepcopy(self.critic_warmup),
                "micro": {"micro_rows": self.max_micro_rows, "micro_token_cap": self.max_micro_tokens},
                "effective_last": dict(self.effective_last), "probe": pack_arrays(self.probe),
                "probe_prev": pack_arrays(self.probe_prev), "probe_prev_digest": self.probe_prev_digest,
            },
        }
        if self.compile_requested:
            payload["ppo"]["compile"] = dict(self.compile_status)
        if self.critic_scaler.is_enabled():              # fp16 only: an fp32 / bf16 checkpoint keeps its old keys
            payload["ppo"]["critic_scaler"] = self.critic_scaler.state_dict()
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        if monitor_path is not None:
            self.monitor.save(monitor_path)
        return path


__all__ = ["LEARNER_VERSION", "CKPT_FORMAT", "COMPAT_SCHEMA", "COMPAT_FILES", "MAX_MICRO_ROWS", "LEARNER_KEYS",
           "POST_STEP_KEYS", "SHARED_KEYS", "OWNED_KEYS", "OWNED_PREFIXES", "PRESTEP_STOP_MONITORS",
           "IdentityError", "state_dict_digest", "compat_record", "compat_digest", "ppo_config",
           "model_from_state", "load_source_policy", "freeze", "make_reference", "policy_unused_prefixes",
           "set_policy_trainable", "Critic", "SeatItem", "PreparedUpdate", "MinibatchStats", "UpdateResult",
           "Learner", "PolicyParamsError", "NEUTRAL_POLICY_PARAMS", "canonical_policy_params",
           "is_neutral_policy_params", "policy_params_temperature", "split_policy_params",
           "check_model_policy_params", "opponent_model_config", "convert_policy_params", "PinError", "verify_pin",
           "verify_pinned_checkpoints", "verify_shard_pins", "item_tokens", "split_micro", "peak_memory",
           "ev_by_phase", "calibration_by_phase", "WarmupResult", "FixedSetEV", "EffectiveValueError",
           "EFFECTIVE_RTOL", "gate_trend_metrics", "argmax_flips", "TREND_ALL", "POST_STEP_TRACK_B_KEYS",
           "PROBE_ROW_KEYS", "entropy_vs_baseline"]
