"""Runtime device / dtype / attention-backend dispatch for CUDA GPUs, Apple MPS and CPU.

One config runs on every supported device; only the values returned by
:func:`resolve` differ:

=====================  ======  ========  ==================  ====================
device                 cc      dtype     scaler              SDPA backends
=====================  ======  ========  ==================  ====================
CUDA GPU, cc >= 8      (8, x)  bf16      no                  flash > mem_eff > math
CUDA GPU, cc 7.x       (7, x)  fp16      ``GradScaler``      mem_eff > math
Apple MPS              --      fp32      no                  math
CPU                    --      fp32      no                  math
=====================  ======  ========  ==================  ====================

Hard rules enforced here:

* the bf16 dtype is only ever produced inside the ``cc >= 8`` branch -- sm_75
  raises on bf16, it does not degrade;
* ``force_fp16=True`` pins fp16 + scaler on every CUDA card (cross-card A/B);
* TF32 is off unless explicitly requested and the card supports it;
* log-probs / ratios / KL / entropy and the placement logits are computed in
  fp32 by the callers regardless of the autocast dtype.
"""

from __future__ import annotations

import contextlib
import os
import resource
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (env var must be set before the MPS backend initialises)

_BACKEND_ORDER: Dict[str, Tuple[str, ...]] = {
    "flash": ("flash", "mem_efficient", "math"),
    "mem_efficient": ("mem_efficient", "math"),
    "math": ("math",),
}


@dataclass(frozen=True)
class DevicePolicy:
    """Everything the trainer needs to know about the current machine."""

    device: str
    dtype: torch.dtype
    use_scaler: bool
    attn_backend: str
    tf32: bool
    batch_seqs: int
    grad_accum: int
    capability: Optional[Tuple[int, int]] = None
    device_name: str = ""

    @property
    def amp_enabled(self) -> bool:
        return self.dtype != torch.float32

    @property
    def sdpa_backends(self) -> Tuple[str, ...]:
        return _BACKEND_ORDER[self.attn_backend]

    def describe(self) -> Dict[str, Any]:
        """JSON-serialisable summary (printed by preflight / selftest, stored in provenance)."""

        return {
            "device": self.device,
            "device_name": self.device_name,
            "capability": list(self.capability) if self.capability else None,
            "dtype": str(self.dtype).replace("torch.", ""),
            "use_scaler": self.use_scaler,
            "attn_backend": self.attn_backend,
            "sdpa_backends": list(self.sdpa_backends),
            "tf32": self.tf32,
            "batch_seqs": self.batch_seqs,
            "grad_accum": self.grad_accum,
            "torch": torch.__version__,
        }

    def summary_tuple(self) -> Tuple[str, str, str, int]:
        """``(device, dtype, attn_backend, batch_seqs)``."""

        return (self.device, str(self.dtype).replace("torch.", ""), self.attn_backend,
                self.batch_seqs)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve(device: str = "auto", *, force_fp16: bool = False,
            allow_tf32: bool = False) -> DevicePolicy:
    """Pick dtype / attention backend / default batch for ``device``.

    ``device`` is ``auto`` (cuda -> mps -> cpu), ``cuda``, ``cuda:N``, ``mps`` or
    ``cpu``.  Asking for an unavailable device raises ``RuntimeError``.
    """

    requested = device
    if device == "auto":
        device = _auto_device()
    kind = device.split(":", 1)[0]
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"device {requested!r} requested but CUDA is unavailable")
        index = torch.device(device).index or 0
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        if major >= 8 and not force_fp16:
            # compute capability >= 8 (Ampere and newer): the only bf16 branch.
            return DevicePolicy(device, torch.bfloat16, False, "flash", bool(allow_tf32),
                                32, 1, (major, minor), name)
        if major >= 8:
            return DevicePolicy(device, torch.float16, True, "flash", bool(allow_tf32),
                                32, 1, (major, minor), name)
        if major == 7:
            # compute capability 7.x (Volta / Turing): no bf16, no FlashAttention-2, no TF32.
            return DevicePolicy(device, torch.float16, True, "mem_efficient", False,
                                8, 4, (major, minor), name)
        return DevicePolicy(device, torch.float16, True, "mem_efficient", False,
                            4, 8, (major, minor), name)
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(f"device {requested!r} requested but MPS is unavailable")
        return DevicePolicy("mps", torch.float32, False, "math", False, 4, 1, None,
                            "Apple MPS")
    if kind == "cpu":
        return DevicePolicy("cpu", torch.float32, False, "math", False, 2, 1, None, "cpu")
    raise ValueError(f"unknown device {requested!r}")


def apply_backend_flags(policy: DevicePolicy) -> None:
    """Set TF32 / SDPA enable flags explicitly (never rely on library defaults)."""

    if policy.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = policy.tf32
        torch.backends.cudnn.allow_tf32 = policy.tf32
        allowed = set(policy.sdpa_backends)
        torch.backends.cuda.enable_flash_sdp("flash" in allowed)
        torch.backends.cuda.enable_mem_efficient_sdp("mem_efficient" in allowed)
        torch.backends.cuda.enable_math_sdp("math" in allowed)


def autocast(policy: DevicePolicy) -> contextlib.AbstractContextManager:
    """Autocast context for the forward pass (a no-op in fp32)."""

    if not policy.amp_enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=policy.device.split(":", 1)[0], dtype=policy.dtype)


@contextlib.contextmanager
def sdpa_context(policy: DevicePolicy) -> Iterator[None]:
    """Restrict ``scaled_dot_product_attention`` to the policy's backends."""

    if not policy.device.startswith("cuda"):
        yield
        return
    from torch.nn.attention import SDPBackend, sdpa_kernel

    table = {"flash": SDPBackend.FLASH_ATTENTION,
             "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
             "math": SDPBackend.MATH}
    with sdpa_kernel([table[name] for name in policy.sdpa_backends]):
        yield


def make_grad_scaler(policy: DevicePolicy) -> Optional["torch.amp.GradScaler"]:
    """A ``GradScaler`` for fp16, ``None`` otherwise (bf16/fp32 need none)."""

    if not policy.use_scaler:
        return None
    return torch.amp.GradScaler("cuda")


def synchronize(policy: DevicePolicy) -> None:
    if policy.device.startswith("cuda"):
        torch.cuda.synchronize()
    elif policy.device == "mps":
        torch.mps.synchronize()


def reset_peak_memory(policy: DevicePolicy) -> None:
    if policy.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def memory_stats(policy: DevicePolicy) -> Dict[str, Optional[float]]:
    """Peak/current memory in MiB for the active device (best available counter)."""

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss_kb / (1024.0 * 1024.0) if sys.platform == "darwin" else rss_kb / 1024.0
    out: Dict[str, Optional[float]] = {"host_peak_rss_mib": round(rss_mib, 1),
                                       "device_peak_mib": None, "device_current_mib": None}
    if policy.device.startswith("cuda"):
        out["device_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        out["device_current_mib"] = round(torch.cuda.memory_allocated() / 2 ** 20, 1)
    elif policy.device == "mps":
        out["device_current_mib"] = round(torch.mps.current_allocated_memory() / 2 ** 20, 1)
        out["device_peak_mib"] = round(torch.mps.driver_allocated_memory() / 2 ** 20, 1)
    return out


def list_backends() -> List[str]:
    """Names accepted by ``DevicePolicy.attn_backend``."""

    return list(_BACKEND_ORDER)


__all__ = [
    "DevicePolicy", "apply_backend_flags", "autocast", "list_backends", "make_grad_scaler",
    "memory_stats", "reset_peak_memory", "resolve", "sdpa_context", "synchronize",
]
