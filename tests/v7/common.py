"""Shared fixtures for the v7 tests (tiny models).

The tokenised examples of the full suite are built from shards in the private dataset format with
the private tokenizer, neither of which is published: the shard and example helpers below raise
``unittest.SkipTest``, so the tests that need them are reported as skipped.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "tests" / "v7"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from amini.nn import schema_v1 as S  # noqa: E402,F401

SKIP_REASON = "needs the private dataset"

TINY_MODEL = dict(d_model=32, n_heads=4, ffn_mult=2, n_step_layers=2, n_temporal_layers=2,
                  dropout=0.0, d_lnk=8, cell_dim=16, cell_heads=2, cell_layers=1, cell_ffn_mult=2)


def shard_root() -> str:
    """Directory of the test shards (private dataset format): not available, the caller is skipped."""
    raise unittest.SkipTest(SKIP_REASON)


def source():
    """Shard reader over ``shard_root()``: not available, the caller is skipped."""
    raise unittest.SkipTest(SKIP_REASON)


def tables():
    """Tokenizer tables: not available, the caller is skipped."""
    raise unittest.SkipTest(SKIP_REASON)


def episode(i: int = 0):
    """One episode of the test shards: not available, the caller is skipped."""
    raise unittest.SkipTest(SKIP_REASON)


def example(*args, **kwargs):
    """A tokenised example (and its draws) of the test shards: not available, the caller is skipped."""
    raise unittest.SkipTest(SKIP_REASON)


def tiny_model(seed: int = 0, double: bool = True, **over):
    import torch
    from amini.nn.aminiformer import AminiFormer, AminiFormerConfig
    torch.manual_seed(seed)
    cfg = dict(TINY_MODEL)
    cfg.update(over)
    m = AminiFormer(AminiFormerConfig.from_dict(cfg))
    if double:
        m = m.double()
    return m.eval()


def to_batch(examples, double: bool = True):
    """Collate tokenised examples with the private tokenizer: not available, the caller is skipped."""
    raise unittest.SkipTest(SKIP_REASON)
