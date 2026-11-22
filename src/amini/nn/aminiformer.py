"""AminiFormer (``aminiformer-v1``): a pure-transformer Clash Royale policy.

Structure::

    typed token embeddings (per-group column projection + type + categorical tables
                            + per-row link codes through Lnk[512, d_lnk] and role projections)
      -> set step encoder: pre-LN self-attention over the VALID tokens of the step
         (packed: invalid tokens are removed, no slot positions) plus a learned [STEP]
      -> z_step = [STEP] output
      -> causal temporal encoder over z_step, RoPE at p = tq / 5, band mask
         (row j visible to row i iff j <= i, i - j < K and tq_j > tq_i - W, with K = window_rows
         and W = window_ticks)
      -> heads: gate (3 families), slot pointer over HAND outputs, ability pointer over
         ENTITY outputs (first 64), two-stage cell decoder, value, privileged critic, aux.

Two-stage cell decoder:
  stage S: 576 cell queries ``E_cell[c] + W_pos [C(x_c), C(y_c)]`` plus ``W_z z_t``,
           ``L_s`` x (cross-attention over the step's token outputs + MLP); aux_threat reads it;
  stage P: ``h_ci = S_c + MLP-residual(S_c + W_h HANDout_i)``, one more cross-attention layer,
           linear logit (LayerNorm + logit in fp32).  Training teacher-forces the label slot;
           ``peek`` evaluates all four slots.

PRIV / LIFE / TOWERHP / BURROW never reach the policy branch: they only enter
through ``targets`` (aux losses) and ``priv_*`` (the privileged critic head, which
reads ``z_t`` but feeds nothing back into any policy logit).

Incremental API::

    state = policy.initial_state(n, window_ticks=cfg.window_ticks, window_rows=cfg.window_rows)
    out   = policy.peek(tokens, positions, state)    # no write
    policy.commit(state, rows_mask)                  # write the pending (k, v)
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from . import schema_v1 as S

MASK_FILL = -1.0e9


# =========================================================================== config
@dataclass
class AminiFormerConfig:
    # model sizes, dropout and the temporal window below are example values
    d_model: int = 128
    n_heads: int = 4
    ffn_mult: int = 2
    n_step_layers: int = 2
    n_temporal_layers: int = 2
    dropout: float = 0.2                 # attention and residual dropout
    d_lnk: int = 16
    card_vocab: int = 256                # >= 244
    obj_vocab: int = 1024                # >= 767
    n_pop: int = S.N_POPULATIONS
    cell_dim: int = 64
    cell_heads: int = 2
    cell_layers: int = 1                 # L_s
    cell_ffn_mult: int = 4
    cell_fp32: str = "auto"              # "logit" | "stage_p" | "auto" (stage_p under bf16)
    window_ticks: int = S.WINDOW_TICKS
    window_rows: int = S.WINDOW_ROWS
    rope_base: float = 10000.0
    priv_card_dim: int = 8
    group_drop_p: float = 0.0            # must stay 0; no drop path exists
    checkpoint_cells: bool = False       # activation checkpointing for the cell decoder
    checkpoint_step: bool = False        # activation checkpointing for the step-encoder blocks
    # placement legality: mask the cell logits with S.cell_mask_bank()[row.cell_mask_id] (training
    # loss = softmax over the legal cells; peek / act identical). False = no placement mask.
    cell_mask: bool = False
    # gate calibration: additive gate-logit bias per stratum, [[wait, play, ability] base,
    # [...] threat] (stratum = the row's public threat flag). Applied wherever gate probabilities are
    # produced for acting or evaluation (peek, eval-mode forward), never in the training loss. None = 0.
    gate_bias: Optional[List[List[float]]] = None
    # False = the model never sees the population.  The GLOBAL group's "pop" category is replaced by the
    # NULL id (0) inside TokenEmbedder, so training, evaluation and acting (forward and peek) give
    # bit-identical outputs for any population value, and training's pop_null_p has no effect.  Saved in
    # model_config only when False (to_dict), so configs written without this key are unchanged; True =
    # the population is an input.
    population_input: bool = True
    # inference-time policy settings (MODEL parameters, like hyperparameters; they never enter the
    # training loss).  ``sample_actions`` reads them through ``PolicyParams.of(model)`` so a checkpoint's
    # settings travel with it (load_policy -> model config) and no controller option exists.  Canonical form
    # (``PolicyParams.to_dict``): {"temperature": {"slot", "cell", "ability"} (> 0, default 1.0),
    # "greedy": {"slot", "cell", "ability"} (default false), "gate_stream": "shared" | "separate"}.
    # All defaults -> stored as None and left out of to_dict, so default configs / checkpoints keep the
    # dict without this key; an older loader refuses a non-default value (unknown key) instead of ignoring it.
    policy_params: Optional[Dict[str, Any]] = None
    # the temporal encoder additionally hides every EARLIER row of the same decision block
    # (tq // S.U_TICKS == tick // S.U_TICKS, since tq = 5 * floor(tick / 5)), so a row cannot
    # read "the previous row of this block waited, so this one acts": training rows at a tap and at the
    # block's planned wait share a block, while acting queries once per block and never has
    # such a row.  The diagonal stays visible.  False (default) = no extra mask; saved in model_config only
    # when True (to_dict), so configs written without this key are unchanged and an older loader refuses it.
    same_block_mask: bool = False
    # the tokenizer-time rule set this model was trained with (S.INPUT_RULES: "" = none,
    # "v6" = tap-time slot rule + deploy gate).  Not used by the network itself: every consumer builds its
    # tokenizer tables with these rules (cfg.input_rules).  Saved only when set.
    input_rules: str = ""

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer (RoPE)")
        if self.cell_dim % self.cell_heads:
            raise ValueError("cell_dim must be divisible by cell_heads")
        if self.group_drop_p != 0.0:
            raise ValueError("group_drop_p must be 0 (no group dropout)")
        if self.card_vocab < S.CARD_VOCAB_V7 or self.obj_vocab < S.OBJECT_VOCAB_V7:
            raise ValueError("card/object tables smaller than the v7 vocabularies")
        if self.cell_fp32 not in ("logit", "stage_p", "auto"):
            raise ValueError("cell_fp32 must be logit | stage_p | auto")
        if self.gate_bias is not None:
            gb = np.asarray(self.gate_bias, dtype=np.float64)
            if gb.shape != (2, S.N_FAMILIES) or not np.isfinite(gb).all():
                raise ValueError(f"gate_bias must be a finite [2, {S.N_FAMILIES}] list (base, threat)")
            self.gate_bias = gb.tolist()
        if not isinstance(self.population_input, bool):
            raise ValueError("population_input must be true or false")
        if not isinstance(self.same_block_mask, bool):
            raise ValueError("same_block_mask must be true or false")
        S.input_rules(self.input_rules)
        if self.policy_params is not None:
            pp = PolicyParams.from_dict(self.policy_params)
            self.policy_params = None if pp.is_default() else pp.to_dict()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AminiFormerConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["population_input"]:          # default: key left out (older loaders accept the dict)
            del d["population_input"]
        if d["policy_params"] is None:     # default: key left out
            del d["policy_params"]
        if not d["same_block_mask"]:       # default: key left out
            del d["same_block_mask"]
        if not d["input_rules"]:
            del d["input_rules"]
        return d


POLICY_HEADS = ("slot", "cell", "ability")
GATE_STREAMS = ("shared", "separate")
MAX_TEMPERATURE = 100.0


@dataclass(frozen=True)
class PolicyParams:
    """Inference-time policy settings of one model (``AminiFormerConfig.policy_params``), per conditional head.

    * ``temperature`` (slot, cell, ability): the sampled heads draw from ``softmax(logit / T)`` over the
      legal entries (masked entries stay at MASK_FILL).  1.0 = the trained distribution (default).
    * ``greedy`` (slot, cell, ability): masked argmax for that head (its temperature is then irrelevant).
      The gate family is NEVER argmaxed: it is always drawn from the competing-hazard probabilities.
    * ``gate_stream``: "shared" (default) draws the gate uniform from the head generator, so the
      number of head draws per query (0 to 3: one per sampled head) shifts every later gate draw;
      "separate" draws it from its own generator (``gate_generator_for``), so arms that differ only in
      head settings (greedy vs sampled, temperatures) see exactly the same gate uniforms.
    """

    temperature: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    greedy: Tuple[bool, bool, bool] = (False, False, False)
    gate_stream: str = "shared"

    def __post_init__(self) -> None:
        for x in self.temperature:
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
                raise ValueError(f"policy_params temperature must be a number, not {x!r}")
        t = tuple(float(x) for x in self.temperature)
        g = tuple(self.greedy)
        if len(t) != 3 or len(g) != 3:
            raise ValueError("policy_params: one temperature and one greedy flag per head (slot, cell, ability)")
        for x in t:
            if not (math.isfinite(x) and 0.0 < x <= MAX_TEMPERATURE):
                raise ValueError(f"policy_params temperature must be in (0, {MAX_TEMPERATURE}], not {x!r} "
                                 "(use greedy for the argmax)")
        for x in g:
            if not isinstance(x, (bool, np.bool_)):
                raise ValueError(f"policy_params greedy flags must be true or false, not {x!r}")
        if self.gate_stream not in GATE_STREAMS:
            raise ValueError(f"policy_params gate_stream must be one of {GATE_STREAMS}, not {self.gate_stream!r}")
        object.__setattr__(self, "temperature", t)
        object.__setattr__(self, "greedy", tuple(bool(x) for x in g))

    # -- (de)serialisation
    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PolicyParams":
        """Canonical or partial dict: ``temperature`` / ``greedy`` may be one value for all heads or a dict
        over a subset of POLICY_HEADS (missing heads keep the default)."""

        if d is None:
            return cls()
        if not isinstance(d, dict):
            raise ValueError(f"policy_params must be a dict, not {type(d).__name__}")
        unknown = set(d) - {"temperature", "greedy", "gate_stream"}
        if unknown:
            raise ValueError(f"unknown policy_params keys: {sorted(unknown)}")

        def per_head(v: Any, default: Any, name: str) -> tuple:
            if v is None:
                return (default,) * 3
            if isinstance(v, dict):
                bad = set(v) - set(POLICY_HEADS)
                if bad:
                    raise ValueError(f"unknown policy_params {name} heads: {sorted(bad)} (have {POLICY_HEADS})")
                return tuple(v.get(h, default) for h in POLICY_HEADS)
            if isinstance(v, (list, tuple)):
                raise ValueError(f"policy_params {name}: a value or a dict over {POLICY_HEADS}, not a list")
            return (v,) * 3

        return cls(temperature=per_head(d.get("temperature"), 1.0, "temperature"),
                   greedy=per_head(d.get("greedy"), False, "greedy"),
                   gate_stream=d.get("gate_stream", "shared"))

    def to_dict(self) -> Dict[str, Any]:
        return {"temperature": dict(zip(POLICY_HEADS, self.temperature)),
                "greedy": dict(zip(POLICY_HEADS, self.greedy)),
                "gate_stream": self.gate_stream}

    def is_default(self) -> bool:
        return self == PolicyParams()

    @classmethod
    def of(cls, src: Any = None) -> "PolicyParams":
        """The settings of ``src``: None (defaults), a PolicyParams, a dict, a AminiFormerConfig, a model with
        ``cfg`` (AminiFormer), or a policy object exposing ``policy_params`` (e.g. the inference server's
        RemotePolicy); any other policy object (RandomPolicy, test stubs) has the defaults."""

        if src is None:
            return cls()
        if isinstance(src, PolicyParams):
            return src
        if isinstance(src, dict):
            return cls.from_dict(src)
        if isinstance(src, AminiFormerConfig):
            return cls.from_dict(src.policy_params)
        cfg = getattr(src, "cfg", None)
        if isinstance(cfg, AminiFormerConfig):
            return cls.from_dict(cfg.policy_params)
        if hasattr(src, "policy_params"):
            return cls.of(src.policy_params)
        return cls()

    def with_greedy(self, greedy: Optional[bool]) -> "PolicyParams":
        """An explicit experiment override of all three greedy flags (None = the model's own flags)."""

        if greedy is None:
            return self
        return PolicyParams(self.temperature, (bool(greedy),) * 3, self.gate_stream)


_GATE_SALT = 0x6A09E667F3BCC909          # sqrt(2) fraction bits (fixed; any change re-seeds every gate stream)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def gate_generator_for(generator: "torch.Generator") -> "torch.Generator":
    """The gate stream paired with a head generator: a fresh generator seeded from
    ``generator.initial_seed()`` (splitmix64 with a fixed salt).  It never draws from ``generator`` and
    depends only on its seed, so creating it changes nothing else and is the same in every arm."""

    seed = _splitmix64(int(generator.initial_seed()) ^ _GATE_SALT) & ((1 << 63) - 1)
    return torch.Generator(device=generator.device).manual_seed(seed)


def fp32_region(t: torch.Tensor) -> contextlib.AbstractContextManager:
    if t.device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=t.device.type, enabled=False)
    return contextlib.nullcontext()


def _hp(x: torch.Tensor) -> torch.Tensor:
    """At least fp32 (keeps fp64 in double-precision tests)."""

    return x if x.dtype == torch.float64 else x.float()


def _autocast_dtype(t: torch.Tensor) -> Optional[torch.dtype]:
    dev = t.device.type
    if dev not in ("cuda", "cpu"):
        return None
    try:
        on = torch.is_autocast_enabled(dev)
        return torch.get_autocast_dtype(dev) if on else None
    except TypeError:                                    # torch < 2.4
        if dev == "cuda":
            return torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        return torch.get_autocast_cpu_dtype() if torch.is_autocast_cpu_enabled() else None


# =========================================================================== blocks
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float) -> None:
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        self.register_buffer("inv_freq", inv.float(), persistent=False)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``positions [..., T]`` -> cos/sin ``[..., 1, T, hd/2]``."""

        dt = torch.float64 if positions.dtype == torch.float64 else torch.float32
        ang = positions.to(dt).unsqueeze(-1) * self.inv_freq.to(dt)
        return torch.cos(ang).unsqueeze(-3), torch.sin(ang).unsqueeze(-3)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    xf = x if x.dtype == torch.float64 else x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos.to(xf.dtype), sin.to(xf.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


def lean_step_inputs(cos: torch.Tensor, sin: torch.Tensor, ring_mask: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What every temporal layer of one ``peek`` shares: ``c2 = [cos | cos]``, ``s2 = [-sin | sin]`` for
    :func:`lean_rotary`, and the key mask ``[N, 1, 1, M + 1]`` (ring rows, then the new row)."""

    ones = torch.ones_like(ring_mask[:, :1])
    mask = torch.cat([ring_mask, ones], dim=1)[:, None, None, :]
    return torch.cat([cos, cos], dim=-1), torch.cat([-sin, sin], dim=-1), mask


def lean_rotary(x: torch.Tensor, c2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """:func:`apply_rotary` in four device ops instead of seven. With ``x = [x1 | x2]`` it returns
    ``[x1 | x2] * [c | c] + [x2 | x1] * [-s | s]``: the first half is ``x1*c + x2*(-s)`` and the second
    ``x2*c + x1*s``, which are the IEEE values of ``x1*c - x2*s`` and ``x1*s + x2*c`` (negation and the order of
    one addition are exact), so the result equals :func:`apply_rotary` bit for bit."""

    half = x.shape[-1] // 2
    xf = x if x.dtype == torch.float64 else x.float()
    sw = torch.cat([xf[..., half:], xf[..., :half]], dim=-1)
    return (xf * c2.to(xf.dtype) + sw * s2.to(xf.dtype)).to(x.dtype)


class MHA(nn.Module):
    """Multi-head attention (self or cross) over SDPA with a boolean key mask."""

    def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.h = n_heads
        self.hd = d_q // n_heads
        self.dropout = dropout
        self.q = nn.Linear(d_q, d_q)
        self.kv = nn.Linear(d_kv, 2 * d_q)
        self.o = nn.Linear(d_q, d_q)

    def heads(self, x: torch.Tensor, w: nn.Linear, n: int) -> torch.Tensor:
        b, t, _ = x.shape
        return w(x).view(b, t, n, self.h, self.hd).permute(2, 0, 3, 1, 4)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               mask: Optional[torch.Tensor]) -> torch.Tensor:
        p = self.dropout if self.training else 0.0
        if mask is not None and q.dtype != k.dtype:
            k, v = k.to(q.dtype), v.to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        b, h, t, hd = y.shape
        return self.o(y.transpose(1, 2).reshape(b, t, h * hd))

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor], mask: Optional[torch.Tensor],
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        q = self.heads(x, self.q, 1)[0]
        kv = self.heads(x if mem is None else mem, self.kv, 2)
        k, v = kv[0], kv[1]
        if rope is not None:
            q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        return self.attend(q, k, v, mask)


class Block(nn.Module):
    """Pre-LN transformer block; ``cross=True`` attends to a separate memory."""

    def __init__(self, d: int, n_heads: int, ffn_mult: int, dropout: float,
                 d_mem: Optional[int] = None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, d_mem or d, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        # dropout on attention probabilities and on both residual branches (attention / FFN
        # output); none inside the FFN hidden layer (it would store a 4d-wide mask per token)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                mem: Optional[torch.Tensor] = None,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, mem, mask, rope))
        return x + self.drop(self.ffn(self.ln2(x)))

    def step(self, x: torch.Tensor, k_ring: torch.Tensor, v_ring: torch.Tensor,
             ring_mask: torch.Tensor, rope: Tuple[torch.Tensor, torch.Tensor],
             lean: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One new row ``x [N,1,d]`` against the ring; returns ``(y, k_new, v_new)``.

        ``lean`` = :func:`lean_step_inputs` of this call: the rotary factors and the key mask built once for
        all layers, and q / k rotated in one pass. Bit for bit the default path (see :func:`lean_rotary`)."""

        h = self.ln1(x)
        a = self.attn
        q = a.heads(h, a.q, 1)[0]
        kv = a.heads(h, a.kv, 2)
        k, v = kv[0], kv[1]
        if lean is None:
            q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        else:
            qk = lean_rotary(torch.stack([q, k]), lean[0], lean[1])
            q, k = qk[0], qk[1]
        keys = torch.cat([k_ring.to(k.dtype), k], dim=2)
        vals = torch.cat([v_ring.to(v.dtype), v], dim=2)
        if lean is None:
            ones = torch.ones_like(ring_mask[:, :1])
            mask = torch.cat([ring_mask, ones], dim=1)[:, None, None, :]
        else:
            mask = lean[2]
        x = x + self.drop(a.attend(q, keys, vals, mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x, k, v


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))


# =========================================================================== embeddings
class TokenEmbedder(nn.Module):
    def __init__(self, cfg: AminiFormerConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.proj = nn.ModuleDict({n: nn.Linear(S.GROUPS[n].n_features, d) for n in S.GROUP_ORDER})
        self.type_emb = nn.Parameter(torch.randn(len(S.GROUP_ORDER), d) * 0.02)
        sizes = {"card": cfg.card_vocab, "base": cfg.card_vocab, "obj": cfg.obj_vocab,
                 "pop": cfg.n_pop}
        self.tables = nn.ModuleDict()
        for name, (size, pad) in S.CAT_TABLES.items():
            emb = nn.Embedding(sizes.get(name, size), d, padding_idx=pad)
            nn.init.normal_(emb.weight, std=0.02)
            if pad is not None:
                with torch.no_grad():
                    emb.weight[pad].zero_()
            self.tables["t_" + name] = emb
        self.lnk = nn.Embedding(S.LINK_CODES, cfg.d_lnk)
        nn.init.normal_(self.lnk.weight, std=1.0)
        self.role_proj = nn.ModuleDict({r: nn.Linear(cfg.d_lnk, d, bias=False) for r in S.LINK_ROLES})
        self.absent = nn.Parameter(torch.randn(len(S.LINK_ROLES), d) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.population_input = bool(cfg.population_input)

    def forward(self, name: str, grp: Dict[str, torch.Tensor]) -> torch.Tensor:
        spec = S.GROUPS[name]
        gi = S.GROUP_ORDER.index(name)
        x = self.proj[name](grp["feat"]) + self.type_emb[gi]
        cat = grp["cat"]
        for j, (_, table) in enumerate(spec.cats):
            c = cat[..., j]
            if table == "pop" and not self.population_input:
                c = torch.zeros_like(c)                 # population NULL for every input
            x = x + self.tables["t_" + table](c)
        lnk = grp["lnk"]
        for r, role in enumerate(spec.roles):
            code = lnk[..., r]
            e = self.lnk(code.clamp(min=0)) * (code >= 0).unsqueeze(-1).to(self.lnk.weight.dtype)
            x = x + self.role_proj[role](e)
            x = x + self.absent[S.LINK_ROLES.index(role)] * (code == S.LINK_ABSENT).unsqueeze(-1).to(x.dtype)
        return self.ln(x)

    # ------------------------------------------------------------- unified lookup (fewer device ops)
    # ``forward`` adds, per token, the group's type row, one row per categorical column and, per link role, either
    # ``role_proj(lnk[code])`` (code >= 0), the role's ``absent`` row (code == LINK_ABSENT) or nothing. All of these
    # are rows of ONE table, so every group can be embedded with one index build, one lookup and one sum instead
    # of a Python loop that issues about five small device ops per column. Same function; the per-token terms are
    # summed in another order, so the result differs from ``forward`` by float rounding only. Runtime switch
    # (``StepEncoder.unified``), no weight.
    UNI_CATS = max(len(g.cats) for g in S.GROUPS.values())
    UNI_ROLES = max(len(g.roles) for g in S.GROUPS.values())

    def clear_unified(self) -> None:
        """Drop the cached table (after the weights changed in place)."""
        self.__dict__.pop("_uni", None)

    def _unified_tables(self) -> Dict[str, Any]:
        w0 = self.lnk.weight
        cache = self.__dict__.get("_uni")
        if cache is not None and cache["W"].device == w0.device and cache["W"].dtype == w0.dtype:
            return cache
        d = self.type_emb.shape[1]
        rows = [w0.new_zeros(1, d), self.type_emb]              # row 0: "nothing"; rows 1..: group types
        off = 1 + self.type_emb.shape[0]
        t_off: Dict[str, int] = {}
        for name in S.CAT_TABLES:
            emb = self.tables["t_" + name]
            w = emb.weight
            if emb.padding_idx is not None and torch.is_grad_enabled():
                # nn.Embedding gives its pad row no gradient: same values (x*1 + 0 and 0 + x are exact), the pad
                # row's gradient cut
                keep = torch.ones(w.shape[0], 1, dtype=w.dtype, device=w.device)
                keep[emb.padding_idx] = 0
                w = w * keep + (w * (1 - keep)).detach()
            t_off[name] = off
            rows.append(w)
            off += w.shape[0]
        r_off: Dict[str, int] = {}
        for ri, role in enumerate(S.LINK_ROLES):
            r_off[role] = off
            rows.append(w0 @ self.role_proj[role].weight.t())    # = role_proj[role](lnk.weight), [LINK_CODES, d]
            rows.append(self.absent[ri:ri + 1])                  # row LINK_CODES of the role: its absent vector
            off += w0.shape[0] + 1
        cache = {"W": torch.cat(rows, 0), "t_off": t_off, "r_off": r_off, "layouts": {}}
        if not self.training and not torch.is_grad_enabled():   # weights are fixed: keep (see clear_unified)
            self.__dict__["_uni"] = cache
        return cache

    def _unified_layout(self, cache: Dict[str, Any], names: Tuple[str, ...], sizes: Tuple[int, ...]
                        ) -> Dict[str, torch.Tensor]:
        key = (names, sizes)
        lay = cache["layouts"].get(key)
        if lay is not None:
            return lay
        C, R, lc = self.UNI_CATS, self.UNI_ROLES, self.lnk.weight.shape[0]
        typ, coff, cmul, roff, rabs = [], [], [], [], []
        for name, g in zip(names, sizes):
            spec = S.GROUPS[name]
            co = [cache["t_off"][t] for _, t in spec.cats]
            cm = [0 if (t == "pop" and not self.population_input) else 1 for _, t in spec.cats]
            ro = [cache["r_off"][r] for r in spec.roles]
            pad_c, pad_r = [0] * (C - len(co)), [0] * (R - len(ro))
            typ += [[1 + S.GROUP_ORDER.index(name)]] * g
            coff += [co + pad_c] * g
            cmul += [cm + pad_c] * g
            roff += [ro + pad_r] * g
            rabs += [[x + lc for x in ro] + pad_r] * g
        dev = cache["W"].device
        mk = lambda rows, width: torch.tensor(rows, dtype=torch.long, device=dev).reshape(len(rows), width)
        lay = {"type": mk(typ, 1).unsqueeze(0), "coff": mk(coff, C), "cmul": mk(cmul, C), "roff": mk(roff, R),
               "rabs": mk(rabs, R), "zero": torch.zeros((), dtype=torch.long, device=dev),
               "none": torch.full((1, 1, R), S.LINK_NONE, dtype=torch.long, device=dev)}
        cache["layouts"][key] = lay
        return lay

    def unified(self, names: Sequence[str], groups: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """``forward(name, groups[name])`` of every group in ``names``, concatenated on the token axis:
        ``[n, sum G, d]`` in ``names`` order (the groups must have tokens)."""

        cache = self._unified_tables()
        names = tuple(names)
        lay = self._unified_layout(cache, names, tuple(int(groups[nm]["feat"].shape[1]) for nm in names))
        C, R = self.UNI_CATS, self.UNI_ROLES
        feats, cats, lnks = [], [], []
        for nm in names:
            g = groups[nm]
            feats.append(self.proj[nm](g["feat"]))
            c = g["cat"]
            cats.append(c if c.shape[-1] == C else F.pad(c, (0, C - c.shape[-1])))
            l = g["lnk"]
            if l.shape[-1] == 0:
                l = lay["none"].expand(c.shape[0], c.shape[1], R)
            elif l.shape[-1] < R:
                l = F.pad(l, (0, R - l.shape[-1]), value=S.LINK_NONE)
            lnks.append(l)
        cat, code = torch.cat(cats, dim=1), torch.cat(lnks, dim=1)
        ic = cat * lay["cmul"] + lay["coff"]
        il = torch.where(code >= 0, code + lay["roff"], torch.where(code == S.LINK_ABSENT, lay["rabs"], lay["zero"]))
        idx = torch.cat([lay["type"].expand(cat.shape[0], -1, -1), ic, il], dim=-1)
        add = F.embedding(idx, cache["W"]).sum(dim=2)
        return self.ln(torch.cat(feats, dim=1) + add)


def take_rows(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """``x[sel]`` for a [B, T] bool row mask ``sel`` (the plain path, a GPU->CPU sync on CUDA), or for the
    int64 flat row-major index of its True rows (``host_index``: ``index_select`` on ``x`` viewed as
    [B*T, ...], no sync).  Same rows, same order, same values (and the same gradients: every selected
    row receives exactly one gradient contribution in both)."""

    if sel.dtype != torch.long:                 # mask indexing, whatever the mask dtype
        return x[sel]
    return x.reshape((x.shape[0] * x.shape[1],) + tuple(x.shape[2:])).index_select(0, sel)


# =========================================================================== encoders
@dataclass
class StepOut:
    z_step: torch.Tensor            # [N, d]
    mem: torch.Tensor               # [N, L, d] packed token outputs (index 0 = [STEP])
    mem_mask: torch.Tensor          # [N, L] bool
    hand: torch.Tensor              # [N, 4, d]
    ent: torch.Tensor               # [N, Ge, d]
    ent_valid: torch.Tensor         # [N, Ge]
    obj: torch.Tensor               # [N, Go, d]
    obj_valid: torch.Tensor         # [N, Go]
    tower: torch.Tensor             # [N, 6, d]


class StepEncoder(nn.Module):
    def __init__(self, cfg: AminiFormerConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed = TokenEmbedder(cfg)
        self.step_token = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
                                    for _ in range(cfg.n_step_layers))
        self.ln_out = nn.LayerNorm(d)
        # runtime switch, not a weight or a config field (checkpoints and acting unchanged):
        # embed, LayerNorm, project and FFN only the VALID tokens (packed [M, d]); only the attention
        # itself runs on the padded [N, L] layout.  Every valid token computes the same function as on
        # the padded path (pad keys are masked there), so outputs agree to float rounding.
        self.packed = False
        # runtime switch (not a weight or a config field): embed every group with TokenEmbedder.unified (one
        # lookup for all additive terms) on the padded path; outputs agree with the default to float rounding
        self.unified = False
        # compile support (runtime only): compiled token-wise regions ({"pre", "post"} from
        # compile_regions) and the packed-token ladder (M rounded up to a multiple; 0 = off)
        self.compiled: Optional[Dict[str, Any]] = None
        self.pad_multiple = 0

    @staticmethod
    def _token_layout(valid_of: Dict[str, torch.Tensor], n: int):
        """Token validity layout shared by :meth:`forward` and :meth:`host_index` (same ops, same ints):
        group names with tokens, their [N, G] valid masks, offsets, the [N, 1 + sum G] valid mask and
        the stable valid-first order."""

        names, valids, offs = [], [], {}
        off = 1
        for name in S.GROUP_ORDER:
            gv = valid_of[name]
            gsize = gv.shape[1]
            offs[name] = (off, gsize)
            off += gsize
            if gsize == 0:
                continue
            v = gv.bool()
            if S.GROUPS[name].always_valid:
                v = torch.ones_like(v)
            names.append(name)
            valids.append(v)
        dev = valids[0].device
        valid = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=dev)] + valids, dim=1)
        order = torch.sort((~valid).to(torch.int8), dim=1, stable=True).indices
        return names, valids, offs, valid, order

    @staticmethod
    def host_index(valid_of: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """``host_index``: the integer layout :meth:`forward` would otherwise derive on the device
        with GPU->CPU syncs (``length`` via ``.item()``, the packed slot / token indices via ``nonzero``),
        computed from the same [N, G] valid masks on the host (loader worker).  Same ops on the same
        booleans, so the same ints; :meth:`forward` / :meth:`_forward_packed` then take them as given."""

        n = next(iter(valid_of.values())).shape[0]
        names, valids, offs, valid, order = StepEncoder._token_layout(valid_of, n)
        total = valid.shape[1]
        length = min(total, max(8, -(-int(valid.sum(dim=1).max().item()) // 8) * 8))
        mc = valid.gather(1, order[:, :length])
        return {"length": length,
                "pslots": mc.reshape(-1).nonzero(as_tuple=True)[0],
                "fi": {name: v.reshape(-1).nonzero(as_tuple=True)[0] for name, v in zip(names, valids)}}

    def forward(self, groups: Dict[str, Dict[str, torch.Tensor]],
                pre: Optional[Dict[str, Any]] = None) -> StepOut:
        """``pre``: :meth:`host_index` of these groups (None = derive it here)."""

        n = groups["GLOBAL"]["feat"].shape[0]
        d = self.cfg.d_model
        names, valids, offs, valid, order = self._token_layout({k: g["valid"] for k, g in groups.items()}, n)
        dev = valid.device
        total = valid.shape[1]
        if pre is None:
            length = min(total, max(8, -(-int(valid.sum(dim=1).max().item()) // 8) * 8))
        else:
            length = int(pre["length"])
        idx = order[:, :length]
        mc = valid.gather(1, idx)
        mask = mc[:, None, None, :]
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(total, device=dev).expand(n, -1))
        if self.packed:
            y = self._forward_packed(groups, names, valids, offs, inv, mc, mask, n, length, pre)
        else:
            if self.unified:
                embs = [self.embed.unified(names, groups)]
            else:
                embs = [self.embed(name, groups[name]) for name in names]
            step = self.step_token.to(embs[0].dtype).expand(n, 1, d)
            x = torch.cat([step] + embs, dim=1)
            xc = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
            for blk in self.blocks:
                if self.cfg.checkpoint_step and self.training and torch.is_grad_enabled():
                    xc = torch.utils.checkpoint.checkpoint(blk, xc, mask, use_reentrant=False)
                else:
                    xc = blk(xc, mask)
            y = self.ln_out(xc)

        def take(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            o, gsz = offs[name]
            pos = inv[:, o:o + gsz]
            ok = valid[:, o:o + gsz]                    # valid tokens all sit below `length`
            pc = pos.clamp(max=length - 1)
            out = y.gather(1, pc.unsqueeze(-1).expand(-1, -1, d)) * ok.unsqueeze(-1).to(y.dtype)
            return out, ok

        hand, _ = take("HAND")
        ent, ent_ok = take("ENTITY")
        obj, obj_ok = take("OBJECT")
        tower, _ = take("TOWER")
        return StepOut(z_step=y[:, 0], mem=y, mem_mask=mc, hand=hand, ent=ent, ent_valid=ent_ok,
                       obj=obj, obj_valid=obj_ok, tower=tower)

    # ------------------------------------------------------------- packed path
    def _forward_packed(self, groups, names, valids, offs, inv, mc, mask, n: int, length: int,
                        pre: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Packed step encoder: returns ``y [n, length, d]`` in the padded slot layout (valid slots
        hold the encoder outputs, pad slots are 0; consumers mask them)."""

        d = self.cfg.d_model
        dev = mc.device
        slot = torch.arange(n, device=dev).unsqueeze(1) * length + inv     # token -> padded slot
        pslots = mc.reshape(-1).nonzero(as_tuple=True)[0] if pre is None else pre["pslots"]  # valid slots
        m = pslots.numel()
        pk = torch.full((n * length,), -1, dtype=torch.long, device=dev)
        pk[pslots] = torch.arange(m, device=dev)                           # padded slot -> packed index
        parts, dest = [], []
        for name, v in zip(names, valids):
            o, gsz = offs[name]
            fi = v.reshape(-1).nonzero(as_tuple=True)[0] if pre is None else pre["fi"][name]  # row-major
            sub = {k: t.reshape((n * gsz,) + t.shape[2:]).index_select(0, fi)
                   for k, t in groups[name].items()}
            parts.append(self.embed(name, sub))
            dest.append(pk.index_select(0, slot[:, o:o + gsz].reshape(-1).index_select(0, fi)))
        step = self.step_token.to(parts[0].dtype).expand(n, d)
        e = torch.cat([step] + parts, dim=0)
        dst = torch.cat([pk[slot[:, 0]]] + dest)
        perm = torch.empty_like(dst)
        perm[dst] = torch.arange(m, device=dev)
        xp = e[perm]                                                        # [M, d] in slot order
        pm = int(self.pad_multiple or 0)
        if pm > 0:                       # fixed shapes for compiled regions: M rounded up to a ladder
            xp = F.pad(xp, (0, 0, 0, -(-m // pm) * pm - m))
        for blk in self.blocks:
            if self.cfg.checkpoint_step and self.training and torch.is_grad_enabled():
                xp = torch.utils.checkpoint.checkpoint(_block_packed, blk, xp, pslots, mask, n, length,
                                                       self.compiled, use_reentrant=False)
            else:
                xp = _block_packed(blk, xp, pslots, mask, n, length, self.compiled)
        y = self.ln_out(xp[:m])
        return y.new_zeros(n * length, d).index_copy(0, pslots, y).view(n, length, d)


def _sa_pre(x: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
            eps: float) -> torch.Tensor:
    """LayerNorm + the fused q|k|v projection (token-wise; a compile region)."""

    return F.linear(F.layer_norm(x, (x.shape[-1],), ln_w, ln_b, eps), w, b)


def _sa_post(x: torch.Tensor, y: torch.Tensor, wo: torch.Tensor, bo: torch.Tensor, ln_w: torch.Tensor,
             ln_b: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor,
             eps: float, p: float, training: bool) -> torch.Tensor:
    """Output projection + residual, LayerNorm + FFN + residual (token-wise; a compile region).
    Same ops as :meth:`Block.forward` after the attention (dropout on both residual branches)."""

    x = x + F.dropout(F.linear(y, wo, bo), p, training)
    h = F.gelu(F.linear(F.layer_norm(x, (x.shape[-1],), ln_w, ln_b, eps), w1, b1))
    return x + F.dropout(F.linear(h, w2, b2), p, training)


def _block_packed(blk: "Block", xp: torch.Tensor, pslots: torch.Tensor, mask: torch.Tensor,
                  n: int, length: int, compiled: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """:meth:`Block.forward` (self-attention, no RoPE) on packed valid tokens ``xp [M(+pad), d]``: the
    LayerNorms, projections and FFN run on the packed tokens; q/k/v of the M valid tokens are scattered
    into the padded ``[n, length]`` layout (pad slots 0, masked as keys) for SDPA, and the valid outputs
    gathered back.  q and k|v come from one GEMM over the concatenated weights (same columns as
    ``MHA.heads``).  Rows past M (``StepEncoder.pad_multiple``) are inert fillers: never read back."""

    a = blk.attn
    d = xp.shape[-1]
    m = pslots.numel()
    pre = (compiled or {}).get("pre", _sa_pre)
    post = (compiled or {}).get("post", _sa_post)
    qkv = pre(xp, blk.ln1.weight, blk.ln1.bias, torch.cat([a.q.weight, a.kv.weight], 0),
              torch.cat([a.q.bias, a.kv.bias], 0), blk.ln1.eps)
    if qkv.shape[0] != m:
        qkv = qkv[:m]
    buf = qkv.new_zeros(n * length, 3 * d).index_copy(0, pslots, qkv)
    buf = buf.view(n, length, 3, a.h, a.hd).permute(2, 0, 3, 1, 4)
    p = a.dropout if a.training else 0.0
    y = F.scaled_dot_product_attention(buf[0], buf[1], buf[2], attn_mask=mask, dropout_p=p)
    y = y.transpose(1, 2).reshape(n * length, d).index_select(0, pslots)
    if y.shape[0] != xp.shape[0]:
        y = F.pad(y, (0, 0, 0, xp.shape[0] - y.shape[0]))
    f1, f2 = blk.ffn[0], blk.ffn[2]
    return post(xp, y, a.o.weight, a.o.bias, blk.ln2.weight, blk.ln2.bias, f1.weight, f1.bias,
                f2.weight, f2.bias, blk.ln2.eps, blk.drop.p, blk.drop.training)


def _cross_block_flat(blk: "Block", xf: torch.Tensor, m: torch.Tensor, mask: torch.Tensor, r: int, c: int,
                      compiled: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """:meth:`Block.forward` with memory (cross-attention, no RoPE) for ``r x c`` query tokens held flat
    in ``xf [r*c(+pad), dq]``: LayerNorm + q projection and the output projection + FFN run as the
    token-wise regions; k/v come from the memory ``m [r, L, d_mem]`` (as ``MHA.heads``) and SDPA runs on
    ``[r, h, c, hd]``.  Rows past r*c are inert fillers (never read back)."""

    a = blk.attn
    dq = xf.shape[-1]
    t = r * c
    pre = (compiled or {}).get("pre", _sa_pre)
    post = (compiled or {}).get("post", _sa_post)
    q = pre(xf, blk.ln1.weight, blk.ln1.bias, a.q.weight, a.q.bias, blk.ln1.eps)
    q = q[:t].view(r, c, a.h, a.hd).transpose(1, 2)
    kv = a.heads(m, a.kv, 2)
    k, v = kv[0], kv[1]
    if q.dtype != k.dtype:
        k, v = k.to(q.dtype), v.to(q.dtype)
    p = a.dropout if a.training else 0.0
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
    y = y.transpose(1, 2).reshape(t, dq)
    if y.shape[0] != xf.shape[0]:
        y = F.pad(y, (0, 0, 0, xf.shape[0] - y.shape[0]))
    f1, f2 = blk.ffn[0], blk.ffn[2]
    return post(xf, y, a.o.weight, a.o.bias, blk.ln2.weight, blk.ln2.bias, f1.weight, f1.bias,
                f2.weight, f2.bias, blk.ln2.eps, blk.drop.p, blk.drop.training)


def compile_regions(mode: str = "default", dynamic: Optional[bool] = False,
                    backend: Optional[str] = None) -> Dict[str, Any]:
    """torch.compile'd token-wise regions of the packed step encoder (pure tensor functions: no module
    identity in the guards, so deep copies (the eval model) and EMA weight swaps reuse the graphs)."""

    import os
    import torch._dynamo
    import torch._inductor.config
    torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
    if "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:
        # inductor's default is a pool of one compile worker per CPU (many processes and a lot of RAM);
        # the regions are small, so compile them in-process
        torch._inductor.config.compile_threads = 1
    kw: Dict[str, Any] = {"dynamic": dynamic}
    if backend:
        kw["backend"] = backend          # tests: "eager" / "aot_eager" (no C++ toolchain needed)
    elif mode and mode != "default":
        kw["mode"] = mode
    return {"pre": torch.compile(_sa_pre, **kw), "post": torch.compile(_sa_post, **kw)}


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: AminiFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.dropout)
                                    for _ in range(cfg.n_temporal_layers))
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_base)

    def band_mask(self, tq: torch.Tensor, row_valid: torch.Tensor) -> torch.Tensor:
        """``[B,1,T,T]``: key j visible to query i iff j <= i, i - j < K,
        tq_j > tq_i - W and row j valid (the diagonal is always visible); with ``same_block_mask`` also
        tq_j // U != tq_i // U for j < i, U = S.U_TICKS (no earlier row of the query's own decision block)."""

        t = tq.shape[1]
        ar = torch.arange(t, device=tq.device)
        diff = ar[:, None] - ar[None, :]
        band = (diff >= 0) & (diff < self.cfg.window_rows)
        recent = tq[:, None, :] > (tq[:, :, None] - self.cfg.window_ticks)
        allowed = band[None] & recent & row_valid[:, None, :]
        if self.cfg.same_block_mask:
            blk = torch.div(tq, S.U_TICKS, rounding_mode="floor")      # block = tick // U_TICKS
            allowed = allowed & (blk[:, None, :] != blk[:, :, None])
        allowed = allowed | torch.eye(t, dtype=torch.bool, device=tq.device)[None]
        return allowed[:, None]

    def forward(self, z: torch.Tensor, pos: torch.Tensor, tq: torch.Tensor,
                row_valid: torch.Tensor) -> torch.Tensor:
        mask = self.band_mask(tq, row_valid)
        rope = self.rope(pos)
        x = z
        for blk in self.blocks:
            x = blk(x, mask, rope=rope)
        return self.ln_out(x)


# =========================================================================== heads
class PointerHead(nn.Module):
    """Additive pointer: ``logit_i = v . GELU(A z + B h_i)`` (permutation-equivariant)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.v(F.gelu(self.a(z).unsqueeze(-2) + self.b(h))).squeeze(-1)


class CellDecoder(nn.Module):
    def __init__(self, cfg: AminiFormerConfig) -> None:
        super().__init__()
        d, dc = cfg.d_model, cfg.cell_dim
        self.cfg = cfg
        self.cell_emb = nn.Parameter(torch.randn(S.N_CELLS, dc) * 0.02)
        self.register_buffer("cell_coord", torch.from_numpy(S.cell_query_coords()), persistent=False)
        self.w_pos = nn.Linear(S.COORD_DIM, dc)
        self.w_z = nn.Linear(d, dc)
        self.mem_ln = nn.LayerNorm(d)
        self.stage_s = nn.ModuleList(Block(dc, cfg.cell_heads, cfg.cell_ffn_mult, cfg.dropout, d_mem=d)
                                     for _ in range(cfg.cell_layers))
        self.s_ln = nn.LayerNorm(dc)
        self.aux_threat = nn.Linear(dc, 2)
        self.w_h = nn.Linear(d, dc)
        self.p_ln = nn.LayerNorm(dc)
        self.p_mlp = _mlp(dc, cfg.cell_ffn_mult * dc, dc)
        self.p_cross = Block(dc, cfg.cell_heads, cfg.cell_ffn_mult, cfg.dropout, d_mem=d)
        self.out_ln = nn.LayerNorm(dc)
        self.logit = nn.Linear(dc, 1)
        # runtime switches (not weights, not config): stage S through the token-wise regions
        # (_sa_pre / _sa_post, compiled when ``compiled`` is set) with the R*576 query tokens padded up
        # to a multiple of ``pad_rows`` rows (fixed shapes; 0 = off).  Same function as Block.forward.
        self.compiled: Optional[Dict[str, Any]] = None
        self.pad_rows = 0
        # runtime switch (trainer key stage_p_compiled): stage P's cross block through the same token-wise
        # regions as stage S (_cross_block_flat) when ``compiled`` is set; off (or compiled None) = Block.forward
        self.p_regions = False

    def queries(self) -> torch.Tensor:
        return self.cell_emb + self.w_pos(self.cell_coord.to(self.cell_emb.dtype))

    def run_s(self, z: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        """Stage S: ``[R, 576, dc]`` conditioned on ``z_t`` (added), not on a card."""

        x = self.queries().unsqueeze(0) + self.w_z(z).unsqueeze(1)
        m = self.mem_ln(mem)
        mask = mem_mask[:, None, None, :]
        if self.compiled is not None or self.pad_rows:
            return self._run_s_regions(x, m, mask)
        for blk in self.stage_s:
            if self.cfg.checkpoint_cells and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, mask, m, use_reentrant=False)
            else:
                x = blk(x, mask, mem=m)
        return self.s_ln(x)

    def _run_s_regions(self, x: torch.Tensor, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Stage S with the token-wise work on flat ``[R*576(+pad), dc]`` query tokens."""

        r, c, dc = x.shape
        t = r * c
        xf = x.reshape(t, dc)
        pr = int(self.pad_rows or 0)
        if pr > 0:
            xf = F.pad(xf, (0, 0, 0, (-(-r // pr) * pr - r) * c))
        for blk in self.stage_s:
            if self.cfg.checkpoint_cells and self.training and torch.is_grad_enabled():
                xf = torch.utils.checkpoint.checkpoint(_cross_block_flat, blk, xf, m, mask, r, c,
                                                       self.compiled, use_reentrant=False)
            else:
                xf = _cross_block_flat(blk, xf, m, mask, r, c, self.compiled)
        return self.s_ln(xf[:t].view(r, c, dc))

    def threat_logits(self, s: torch.Tensor) -> torch.Tensor:
        # aux regression head: matmul under autocast, logits (and the BCE) in fp32
        return _hp(self.aux_threat(s))

    def _p(self, s: torch.Tensor, hand: torch.Tensor, mem: torch.Tensor,
           mem_mask: torch.Tensor) -> torch.Tensor:
        r, p = hand.shape[0], hand.shape[1]
        h = s.unsqueeze(1) + self.w_h(hand).unsqueeze(2)                  # [R, P, 576, dc]
        h = h + self.p_mlp(self.p_ln(h))
        h = h.reshape(r * p, S.N_CELLS, -1)
        m = self.mem_ln(mem).repeat_interleave(p, dim=0)
        mask = mem_mask.repeat_interleave(p, dim=0)[:, None, None, :]
        if self.p_regions and self.compiled is not None:
            rp, t = r * p, r * p * S.N_CELLS
            hf = h.reshape(t, -1)
            pr = int(self.pad_rows or 0)                 # static compile: rows padded as in stage S (inert)
            if pr > 0:
                hf = F.pad(hf, (0, 0, 0, (-(-rp // pr) * pr - rp) * S.N_CELLS))
            h = _cross_block_flat(self.p_cross, hf, m, mask, rp, S.N_CELLS, self.compiled)[:t]
        else:
            h = self.p_cross(h, mask, mem=m)
        return h.view(r, p, S.N_CELLS, -1)

    def run_p(self, s: torch.Tensor, hand: torch.Tensor, mem: torch.Tensor,
              mem_mask: torch.Tensor) -> torch.Tensor:
        """Stage P for ``hand [R, P, d]`` (P = 1 teacher-forced or 4): fp32 ``[R, P, 576]``."""

        mode = self.cfg.cell_fp32
        if mode == "auto":
            mode = "stage_p" if _autocast_dtype(s) == torch.bfloat16 else "logit"
        if mode == "stage_p":
            with fp32_region(s):
                h = self._p(_hp(s), _hp(hand), _hp(mem), mem_mask)
        elif self.cfg.checkpoint_cells and self.training and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(self._p, s, hand, mem, mem_mask, use_reentrant=False)
        else:
            h = self._p(s, hand, mem, mem_mask)
        with fp32_region(h):
            return self.logit(self.out_ln(_hp(h))).squeeze(-1)


# =========================================================================== state
@dataclass
class PolicyState:
    """Per-env ring of committed temporal (k, v) rows."""

    k: List[torch.Tensor]                 # per temporal layer [N, H, M, hd]
    v: List[torch.Tensor]
    ring_tq: torch.Tensor                 # [N, M] long
    ring_valid: torch.Tensor              # [N, M] bool
    write_ptr: torch.Tensor               # [N] long
    last_tick: torch.Tensor               # [N] long (-1 = nothing committed)
    window_ticks: int
    window_rows: int
    pending: Optional[Dict[str, Any]] = None

    @property
    def n(self) -> int:
        return int(self.write_ptr.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.ring_valid.shape[1])

    def reset(self, rows: Optional[Sequence[int]] = None) -> "PolicyState":
        sel = slice(None) if rows is None else torch.as_tensor(rows, dtype=torch.long,
                                                               device=self.write_ptr.device)
        self.ring_valid[sel] = False
        self.write_ptr[sel] = 0
        self.last_tick[sel] = -1
        self.pending = None
        return self


# =========================================================================== model
class AminiFormer(nn.Module):
    """Pure-transformer policy/value network for ``aminiformer-v1``."""

    # runtime switch (class default; not a weight or a config field): ``peek`` builds the rotary factors and
    # the key mask once for all temporal layers (lean_step_inputs). Bit for bit the default path.
    lean_peek = False

    def load_state_dict(self, *args: Any, **kwargs: Any):
        out = super().load_state_dict(*args, **kwargs)
        self.step_encoder.embed.clear_unified()         # the cached unified table belongs to the old weights
        return out

    def __init__(self, cfg: AminiFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.step_encoder = StepEncoder(cfg)
        self.temporal = TemporalEncoder(cfg)
        self.gate = _mlp(d, d, S.N_FAMILIES)
        self.slot_head = PointerHead(d)
        self.ability_head = PointerHead(d)
        self.cells = CellDecoder(cfg)
        self.value_head = _mlp(d, d, 1)
        # privileged critic: its own card table (used by the critic only), reads z_t + PRIV, outputs value_priv only
        self.priv_card = nn.Embedding(cfg.card_vocab, cfg.priv_card_dim, padding_idx=0)
        self.priv_proj = nn.Linear(S.PRIV_CARD_COLUMNS * cfg.priv_card_dim + S.PRIV_SCALAR_COLUMNS, d)
        self.value_priv_head = _mlp(2 * d, d, 1)
        self.aux_opp_next = _mlp(d, d, cfg.card_vocab)
        self.aux_opp_hand = _mlp(d, d, cfg.card_vocab)
        with torch.no_grad():   # multi-label prior: 4 hand cards out of the vocab
            self.aux_opp_hand[-1].bias.fill_(math.log(S.HAND_SLOTS / max(1, cfg.card_vocab - S.HAND_SLOTS)))
        self.aux_tower = _mlp(2 * d, d, 2)
        self.aux_land = _mlp(d, d, 3)
        self.aux_emerge = _mlp(d, d, 3)
        # placement mask bank (constant, not in the state dict, so checkpoints without it load unchanged)
        self.register_buffer("cell_bank", torch.from_numpy(S.cell_mask_bank().copy()), persistent=False)

    def gate_bias_tensor(self, like: torch.Tensor) -> Optional[torch.Tensor]:
        if self.cfg.gate_bias is None:
            return None
        return torch.tensor(self.cfg.gate_bias, dtype=like.dtype, device=like.device)

    # ------------------------------------------------------------- parameter groups
    # Parameter-name prefixes of the heads (per-head LR multipliers, ``lr_mult`` in the train config).
    # A prefix is matched with ``str.startswith`` on ``named_parameters()`` names; when several prefixes
    # match one parameter the LONGEST wins, so ``cells.aux_threat.`` (the threat aux head that lives inside
    # the cell decoder) can be set apart from ``cells.``.
    HEAD_PREFIXES = {
        "gate": ("gate.",),                         # family head wait / play / ability (MLP d -> 3)
        "slot": ("slot_head.",),                    # pointer over the 4 hand slots
        "ability": ("ability_head.",),              # pointer over ready units (ENTITY tokens)
        "cell": ("cells.",),                        # cell decoder: stage S + stage P + logit
                                                    # (includes cells.aux_threat.; see aux)
        "aux": ("aux_opp_next.", "aux_opp_hand.", "aux_tower.", "aux_land.", "aux_emerge.",
                "cells.aux_threat."),
        "value": ("value_head.", "value_priv_head.", "priv_card.", "priv_proj."),
        "trunk": ("step_encoder.", "temporal."),    # step encoder + temporal encoder (shared)
    }

    def param_groups(self, weight_decay: float, lr_mult: Optional[Dict[str, float]] = None
                     ) -> List[Dict[str, Any]]:
        """AdamW groups: no decay on norms, biases, embeddings and learned tokens/queries.

        ``lr_mult`` ({parameter-name prefix: multiplier}, empty/None = off): each decay class is split
        further by the multiplier of the longest matching prefix (1.0 when none matches); every group
        carries ``lr_mult`` and the trainer sets ``lr = schedule(step) x lr_mult``.  Without ``lr_mult``
        there are exactly the two decay groups (no ``lr_mult`` key).  A prefix that matches no
        trainable parameter, or a negative / non-finite multiplier, raises ``ValueError``."""

        no_decay_ids = set()
        for mod in self.modules():
            if isinstance(mod, (nn.LayerNorm, nn.Embedding)):
                for p in mod.parameters(recurse=False):
                    no_decay_ids.add(id(p))
        mult = {str(k): float(v) for k, v in (lr_mult or {}).items()}
        for k, v in mult.items():
            if not k or not math.isfinite(v) or v < 0:
                raise ValueError(f"lr_mult: bad entry {k!r}: {v} (non-empty prefix, finite multiplier >= 0)")
        named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        unused = [k for k in mult if not any(n.startswith(k) for n, _ in named)]
        if unused:
            raise ValueError(f"lr_mult: prefixes {unused} match no parameter "
                             f"(heads: {self.HEAD_PREFIXES})")

        def mult_of(name: str) -> float:
            hits = [k for k in mult if name.startswith(k)]
            return mult[max(hits, key=len)] if hits else 1.0

        decay, no_decay = [], []
        for name, p in named:
            if p.ndim < 2 or id(p) in no_decay_ids or name.endswith((
                    "type_emb", "absent", "step_token", "cell_emb")):
                no_decay.append((name, p))
            else:
                decay.append((name, p))
        if not mult:
            return [{"params": [p for _, p in decay], "weight_decay": weight_decay},
                    {"params": [p for _, p in no_decay], "weight_decay": 0.0}]
        groups: List[Dict[str, Any]] = []
        for members, wd in ((decay, weight_decay), (no_decay, 0.0)):
            by: Dict[float, List[Tuple[str, nn.Parameter]]] = {}
            for name, p in members:
                by.setdefault(mult_of(name), []).append((name, p))
            for m in sorted(by):
                groups.append({"params": [p for _, p in by[m]], "weight_decay": wd, "lr_mult": m})
        return groups

    # ------------------------------------------------------------- shared pieces
    @staticmethod
    def flatten_groups(groups: Dict[str, Dict[str, torch.Tensor]], row_valid: torch.Tensor
                       ) -> Dict[str, Dict[str, torch.Tensor]]:
        """``row_valid``: a [B, T] bool mask, or (``host_index``) the int64 flat row-major index of its
        True entries; both select the same rows in the same order (the index needs no GPU->CPU sync)."""
        return {n: {k: take_rows(v, row_valid) for k, v in g.items()} for n, g in groups.items()}

    def policy_heads(self, z: torch.Tensor, st: StepOut, slot_legal: torch.Tensor,
                     ability_ready: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Masked fp32 family / slot / ability logits for ``N`` rows."""

        fam = _hp(self.gate(z))
        ready = ability_ready.bool()
        avail = ready.any(dim=-1)
        fam = fam.clone()
        fam[:, S.FAMILY_ABILITY] = fam[:, S.FAMILY_ABILITY].masked_fill(~avail, MASK_FILL)
        slot = _hp(self.slot_head(z, st.hand)).masked_fill(~slot_legal.bool(), MASK_FILL)
        ge = min(st.ent.shape[1], S.ABILITY_POINTER)
        ab = torch.full((z.shape[0], S.ABILITY_POINTER), MASK_FILL, dtype=torch.float64 if z.dtype == torch.float64 else torch.float32, device=z.device)
        if ge > 0:
            ab[:, :ge] = _hp(self.ability_head(z, st.ent[:, :ge]))
        ab = ab.masked_fill(~ready, MASK_FILL)
        return {"family": fam, "slot": slot, "ability": ab, "ability_avail": avail}

    def value_heads(self, z: torch.Tensor, priv_cards: Optional[torch.Tensor],
                    priv_scalars: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {"value": torch.tanh(_hp(self.value_head(z)).squeeze(-1))}
        if priv_cards is not None and priv_scalars is not None:
            pe = self.priv_card(priv_cards.long()).flatten(-2)
            pv = self.priv_proj(torch.cat([pe, priv_scalars.to(pe.dtype)], dim=-1))
            vp = self.value_priv_head(torch.cat([z, pv.to(z.dtype)], -1))
            out["value_priv"] = torch.tanh(_hp(vp).squeeze(-1))
        return out

    # ------------------------------------------------------------- training forward
    def forward(self, batch: Dict[str, Any], *, cell_rows: Optional[torch.Tensor] = None,
                play_slot: Optional[torch.Tensor] = None, with_aux: bool = True,
                all_slots: bool = False) -> Dict[str, torch.Tensor]:
        """Full-sequence forward over ``[B, T]`` rows.  Outputs are over the ``N`` valid
        rows in row-major order (``batch['row']['row_valid']``).

        ``cell_rows [B,T]``: rows on which stage S runs; ``play_slot [B,T]``: the
        teacher-forced slot (-1: no stage P).  ``all_slots`` evaluates stage P for all
        four slots on the cell rows (used by tests and diagnostics).
        """

        row = batch["row"]
        rv = row["row_valid"].bool()
        # batch["host_index"] (the trainer's host_index key, built by the loader workers) carries the
        # integer row / cell / token indices this forward would otherwise derive with GPU->CPU syncs
        hx = batch.get("host_index")
        sel = rv if hx is None else hx["rv"]
        flat = self.flatten_groups(batch["groups"], sel)
        st = self.step_encoder(flat, None if hx is None else hx["step"])
        b, t = rv.shape
        d = self.cfg.d_model
        if hx is None:
            zs = st.z_step.new_zeros(b, t, d)
            zs[rv] = st.z_step
        else:
            zs = st.z_step.new_zeros(b * t, d).index_copy(0, sel, st.z_step).view(b, t, d)
        pos = row["pos"].to(torch.float64 if zs.dtype == torch.float64 else torch.float32)
        zt_full = self.temporal(zs, pos, row["tq"].long(), rv)
        z = take_rows(zt_full, sel)
        out = self.policy_heads(z, st, take_rows(row["slot_legal"], sel), take_rows(row["ability_ready"], sel))
        gb = self.gate_bias_tensor(out["family"])
        lab = batch.get("labels") or {}
        if gb is not None and not self.training and "threat" in lab:
            out["family"] = apply_gate_bias(out["family"], take_rows(lab["threat"], sel), gb)
        mids = None
        if self.cfg.cell_mask:
            if "cell_mask_id" not in row:
                raise KeyError("cfg.cell_mask needs batch['row']['cell_mask_id'] (from the tokenizer)")
            mids = take_rows(row["cell_mask_id"], sel).long()
        targets = batch.get("targets") or {}
        pc = take_rows(targets["priv_cards"], sel) if "priv_cards" in targets else None
        ps = take_rows(targets["priv_scalars"], sel) if "priv_scalars" in targets else None
        out.update(self.value_heads(z, pc, ps))
        out["z"] = z
        if with_aux:
            out["aux_opp_next"] = _hp(self.aux_opp_next(z))
            out["aux_opp_hand"] = _hp(self.aux_opp_hand(z))
            zt6 = z.unsqueeze(1).expand(-1, S.N_TOWERS, -1)
            out["aux_tower_dmg"] = _hp(self.aux_tower(torch.cat([st.tower, zt6], -1)))
            out["aux_land"] = _hp(self.aux_land(st.obj))
            out["aux_emerge"] = _hp(self.aux_emerge(st.ent))
        if cell_rows is not None:
            if hx is None:
                cr = cell_rows[rv].bool()
                ridx = cr.nonzero(as_tuple=True)[0]
            else:                                       # = cell_rows[rv].nonzero() of the host batch
                ridx = hx["ridx"]
            out["cell_row_index"] = ridx
            if ridx.numel():
                s = self.cells.run_s(z[ridx], st.mem[ridx], st.mem_mask[ridx])
                if with_aux:
                    out["aux_threat"] = self.cells.threat_logits(s)
                if all_slots:
                    out["cells_all"] = self.cells.run_p(s, st.hand[ridx], st.mem[ridx],
                                                        st.mem_mask[ridx])
                    if mids is not None:
                        out["cells_all"] = apply_cell_mask(out["cells_all"], mids[ridx], self.cell_bank)
                if play_slot is not None:
                    ps = take_rows(play_slot, sel)[ridx].long()
                    if hx is None:
                        pidx = (ps >= 0).nonzero(as_tuple=True)[0]
                        out["cell_play_index"] = ridx[pidx]
                    else:                               # host: (ps >= 0).nonzero() and ridx[pidx]
                        pidx = hx["pidx_local"]
                        out["cell_play_index"] = hx["pidx"]
                    if pidx.numel():
                        hsel = st.hand[ridx[pidx], ps[pidx]].unsqueeze(1)
                        out["cells"] = self.cells.run_p(s[pidx], hsel, st.mem[ridx[pidx]],
                                                        st.mem_mask[ridx[pidx]]).squeeze(1)
                        if mids is not None:
                            out["cells"] = apply_cell_mask(out["cells"], mids[ridx[pidx], ps[pidx]],
                                                           self.cell_bank)
        return out

    # ------------------------------------------------------------- incremental API
    def initial_state(self, n: int, window_ticks: int = S.WINDOW_TICKS,
                      window_rows: int = S.WINDOW_ROWS, device: Any = None,
                      dtype: Optional[torch.dtype] = None) -> PolicyState:
        if window_ticks != self.cfg.window_ticks or window_rows != self.cfg.window_rows:
            raise ValueError("deployment window must equal the trained window")
        p = next(self.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        m = max(window_rows - 1, 0)
        cfg = self.cfg
        k = [torch.zeros(n, cfg.n_heads, m, cfg.head_dim, device=device, dtype=dtype)
             for _ in range(cfg.n_temporal_layers)]
        v = [torch.zeros_like(x) for x in k]
        return PolicyState(k=k, v=v, ring_tq=torch.zeros(n, m, dtype=torch.long, device=device),
                           ring_valid=torch.zeros(n, m, dtype=torch.bool, device=device),
                           write_ptr=torch.zeros(n, dtype=torch.long, device=device),
                           last_tick=torch.full((n,), -1, dtype=torch.long, device=device),
                           window_ticks=window_ticks, window_rows=window_rows)

    @torch.no_grad()
    def peek(self, tokens: Dict[str, Any], positions: torch.Tensor, state: PolicyState,
             ticks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """One query row per env: family/slot/ability/value logits and cell logits
        ``[n, 4, 576]`` for all four hand slots.  Stores the per-layer (k, v) of this
        row in ``state.pending``; nothing is written to the ring.

        ``tokens`` = ``{"groups": {...}, "row": {"slot_legal", "ability_ready"}}`` with
        leading dimension n.  ``positions`` = tq / 5 (required; no default).  Optional row keys:
        ``cell_mask_id [n, 4]`` (required when ``cfg.cell_mask``; the cell logits are masked
        exactly as in training) and ``threat [n]`` (the query's stratum: ``cfg.gate_bias`` is added to
        the family logits when both are present).
        """

        if positions is None:
            raise ValueError("positions (tq / 5) are required")
        n = state.n
        positions = positions.reshape(n)
        tq = torch.round(positions.double() * S.TQ_STEP).long()
        tick = tq if ticks is None else ticks.reshape(n).long()
        if bool((tick < state.last_tick).any()):
            raise AssertionError("query tick precedes the last committed tick")
        st = self.step_encoder(tokens["groups"])
        x = st.z_step.view(n, 1, -1)
        pos = positions.to(torch.float64 if x.dtype == torch.float64 else torch.float32)
        cos, sin = self.temporal.rope(pos.view(n, 1))
        ring_mask = state.ring_valid & (state.ring_tq > (tq.view(n, 1) - state.window_ticks))
        lean = lean_step_inputs(cos, sin, ring_mask) if self.lean_peek else None
        new_k, new_v = [], []
        for li, blk in enumerate(self.temporal.blocks):
            x, k, v = blk.step(x, state.k[li], state.v[li], ring_mask, (cos, sin), lean)
            new_k.append(k)
            new_v.append(v)
        z = self.temporal.ln_out(x).view(n, -1)
        row = tokens["row"]
        out = self.policy_heads(z, st, row["slot_legal"], row["ability_ready"])
        gb = self.gate_bias_tensor(out["family"])
        if gb is not None and "threat" in row:
            out["family"] = apply_gate_bias(out["family"], row["threat"].reshape(n), gb)
        out.update(self.value_heads(z, None, None))
        s = self.cells.run_s(z, st.mem, st.mem_mask)
        out["cells"] = self.cells.run_p(s, st.hand, st.mem, st.mem_mask)    # [n, 4, 576]
        if self.cfg.cell_mask:
            if "cell_mask_id" not in row:
                raise KeyError("cfg.cell_mask needs tokens['row']['cell_mask_id']")
            out["cells"] = apply_cell_mask(out["cells"], row["cell_mask_id"].reshape(n, -1).long(),
                                           self.cell_bank)
        out["z"] = z
        state.pending = {"k": new_k, "v": new_v, "tq": tq, "tick": tick}
        return out

    @torch.no_grad()
    def commit(self, state: PolicyState, rows_mask: torch.Tensor) -> None:
        """Write the pending row into each selected env's ring (per-row write pointer)."""

        if state.pending is None:
            raise RuntimeError("commit() without a preceding peek()")
        mask = torch.as_tensor(rows_mask, device=state.write_ptr.device).bool().reshape(state.n)
        p = state.pending
        if bool((p["tick"][mask] < state.last_tick[mask]).any()):
            raise AssertionError("commit tick is not monotone")
        m = state.capacity
        if m > 0 and bool(mask.any()):
            rows = mask.nonzero(as_tuple=True)[0]
            ptr = state.write_ptr[rows]
            for li in range(len(state.k)):
                state.k[li][rows, :, ptr] = p["k"][li][rows, :, 0].to(state.k[li].dtype)
                state.v[li][rows, :, ptr] = p["v"][li][rows, :, 0].to(state.v[li].dtype)
            state.ring_tq[rows, ptr] = p["tq"][rows]
            state.ring_valid[rows, ptr] = True
            state.write_ptr[rows] = (ptr + 1) % m
        state.last_tick = torch.where(mask, p["tick"], state.last_tick)
        state.pending = None


# =========================================================================== masks / calibration
def apply_cell_mask(cells: torch.Tensor, mask_id: torch.Tensor, bank: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
    """Cell logits ``[..., 576]`` with the illegal cells of ``bank[mask_id]`` set to MASK_FILL (fp32).

    The ONE masking function of training (forward), deployment (peek) and the env adapter's override;
    ``mask_id`` has the leading shape of ``cells`` (the tokenizer's per-slot cell mask ids)."""

    if bank is None:
        bank = torch.from_numpy(S.cell_mask_bank().copy())
    bank = bank.to(cells.device)
    legal = bank[mask_id.to(cells.device).long()]
    return _hp(cells).masked_fill(~legal, MASK_FILL)


def apply_gate_bias(family: torch.Tensor, threat: torch.Tensor, bias: Any) -> torch.Tensor:
    """``family [N, 3] + bias[threat]`` (bias ``[2, 3]``: base, threat). The ONE gate-bias function."""

    b = torch.as_tensor(bias, dtype=family.dtype, device=family.device)
    return family + b[threat.to(family.device).long().clamp(0, 1)]


# =========================================================================== acting
def hazard_probs(family_logits: torch.Tensor, ability_avail: torch.Tensor, dt: torch.Tensor,
                 u_ticks: float = S.U_TICKS) -> torch.Tensor:
    """Competing-hazard conversion, fp64: ``[n, 3]`` = P(none), P(play), P(ability)."""

    lf = family_logits.double()
    odds_p = torch.exp(lf[:, S.FAMILY_PLAY] - lf[:, S.FAMILY_WAIT])
    odds_a = torch.exp(lf[:, S.FAMILY_ABILITY] - lf[:, S.FAMILY_WAIT])
    odds_a = torch.where(ability_avail.bool(), odds_a, torch.zeros_like(odds_a))
    tot = odds_p + odds_a
    rate = tot * dt.double().clamp(min=0) / float(u_ticks)
    p_none = torch.exp(-rate)
    p_act = -torch.expm1(-rate)
    safe = torch.where(tot > 0, tot, torch.ones_like(tot))
    p_play = p_act * odds_p / safe
    p_ab = p_act * odds_a / safe
    return torch.stack([p_none, p_play, p_ab], dim=-1)


def _pick(logits: torch.Tensor, greedy: bool, temperature: float,
          generator: Optional[torch.Generator]) -> torch.Tensor:
    """One conditional head: masked argmax, or a draw from ``softmax(logit / T)`` over the legal entries
    (T = 1 is the plain softmax draw exactly; masked entries keep MASK_FILL for any T)."""

    if greedy:
        return logits.argmax(-1)
    if temperature != 1.0:
        logits = torch.where(logits > MASK_FILL / 2, logits / temperature, logits)
    return torch.multinomial(F.softmax(logits, -1), 1, generator=generator).squeeze(-1)


@torch.no_grad()
def sample_actions(out: Dict[str, torch.Tensor], dt: torch.Tensor, *, greedy: Optional[bool] = None,
                   generator: Optional[torch.Generator] = None,
                   u_ticks: float = S.U_TICKS, params: Any = None,
                   gate_generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """Sample one action per row from ``peek`` outputs.

    ``params``: the model's policy settings (``PolicyParams.of``: the model, its config, a dict, a
    PolicyParams; None = defaults).  ``greedy``: None = the model's own per-head flags; True /
    False = an explicit EXPERIMENT override of all three flags (the temperatures stay the model's).
    Deployment passes the model and no ``greedy``.

    The family is ALWAYS sampled from the hazard probabilities, never argmaxed.  Its uniform is the first
    draw of ``generator`` per call (``gate_stream`` "shared", the default), or a draw of ``gate_generator``
    ("separate"; required then, see ``gate_generator_for``).  With the shared stream every sampled head
    then draws from the same generator (slot, ability, cell order), so two arms with different head
    settings share only the first call's gate draw; with the separate stream they share every gate draw.
    A play with no legal slot becomes a wait (``play_blocked``).
    """

    pp = PolicyParams.of(params).with_greedy(greedy)
    probs = hazard_probs(out["family"], out["ability_avail"], dt, u_ticks)
    n = probs.shape[0]
    if pp.gate_stream == "separate":
        if gate_generator is None:
            raise ValueError("gate_stream 'separate' needs gate_generator (gate_generator_for(generator))")
        u = torch.rand(n, generator=gate_generator, dtype=torch.float64, device=probs.device)
    else:
        u = torch.rand(n, generator=generator, dtype=torch.float64, device=probs.device)
    cdf = probs.cumsum(-1)
    family = (u.unsqueeze(-1) > cdf[:, :2]).sum(-1)
    (t_slot, t_cell, t_ab), (g_slot, g_cell, g_ab) = pp.temperature, pp.greedy
    slot_logits = out["slot"].float()
    legal_any = slot_logits.max(-1).values > MASK_FILL / 2
    slot = _pick(slot_logits, g_slot, t_slot, generator)
    ab = _pick(out["ability"].float(), g_ab, t_ab, generator)
    cells = out["cells"].float()[torch.arange(n, device=probs.device), slot]
    cell = _pick(cells, g_cell, t_cell, generator)
    blocked = (family == S.FAMILY_PLAY) & ~legal_any
    family = torch.where(blocked, torch.zeros_like(family), family)
    return {"family": family, "slot": slot, "cell": cell, "ability": ab, "probs": probs,
            "play_blocked": blocked}


def set_policy_params(ckpt_in: str, params: Optional[Dict[str, Any]], ckpt_out: str) -> Dict[str, Any]:
    """Write ``model_config.policy_params`` (``params`` replaces the old settings; None / all defaults clear
    them) into a COPY of a trainer checkpoint, for A/B variants; weights, EMA and everything else untouched.
    Returns the new model config.  Refuses to overwrite its own input."""

    import os
    from pathlib import Path
    if os.path.realpath(ckpt_in) == os.path.realpath(ckpt_out):
        raise ValueError("set_policy_params writes a copy: ckpt_out must differ from ckpt_in")
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    mc["policy_params"] = None if params is None else PolicyParams.from_dict(params).to_dict()
    mc = AminiFormerConfig.from_dict(mc).to_dict()       # validates; all defaults -> key absent
    ck["model_config"] = mc
    ck.setdefault("history_policy_params", []).append(mc.get("policy_params"))
    tmp = Path(str(ckpt_out) + ".tmp")
    torch.save(ck, tmp)
    os.replace(tmp, ckpt_out)
    return mc


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m amini.nn.aminiformer set-policy-params --in CK --out CK2 --params JSON|none``
    (``show-policy-params CK`` prints a checkpoint's settings)."""

    import argparse
    import json
    ap = argparse.ArgumentParser(prog="aminiformer", description=_main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("set-policy-params", help="write a checkpoint copy with the given policy settings")
    p.add_argument("--in", dest="ckpt_in", required=True)
    p.add_argument("--out", dest="ckpt_out", required=True)
    p.add_argument("--params", required=True,
                   help='JSON, e.g. \'{"temperature": {"slot": 0.7, "cell": 0.5}}\' or \'{"greedy": {"slot": true, '
                        '"cell": true}, "gate_stream": "separate"}\'; "none" = defaults')
    q = sub.add_parser("show-policy-params")
    q.add_argument("ckpt")
    a = ap.parse_args(argv)
    if a.cmd == "show-policy-params":
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        pp = PolicyParams.of(AminiFormerConfig.from_dict(ck["model_config"]))
        print(json.dumps({"policy_params": pp.to_dict(), "default": pp.is_default()}))
        return 0
    params = None if a.params.strip().lower() in ("none", "null", "") else json.loads(a.params)
    mc = set_policy_params(a.ckpt_in, params, a.ckpt_out)
    print(json.dumps({"out": a.ckpt_out, "policy_params": PolicyParams.of(mc.get("policy_params")).to_dict()}))
    return 0


__all__ = ["AminiFormer", "AminiFormerConfig", "PolicyState", "hazard_probs", "sample_actions",
           "count_parameters", "MASK_FILL", "apply_cell_mask", "apply_gate_bias", "PolicyParams",
           "gate_generator_for", "set_policy_params", "POLICY_HEADS", "GATE_STREAMS"]


if __name__ == "__main__":
    raise SystemExit(_main())
