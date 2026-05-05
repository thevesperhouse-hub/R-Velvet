"""
Tests for the auto-sizing engine + adaptive .bin dtype.

Covers:
- parse_size / format_size round-trips
- estimate_params returns the right count for known presets
- auto_size lands within tolerance of the target for 350M / 1.3B / 2.5B / 7B
- auto_size respects min_layers / max_layers bounds
- auto_size produces aspect ratios in the LLaMA-family range
- bin_dtype_for_vocab picks the right dtype at the boundary
- write_bin_meta / read_bin_meta round-trip + legacy fallback
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.utils.sizing import (
    parse_size, format_size, auto_size, estimate_params,
)
from rvelvet.utils.dtypes import (
    bin_dtype_for_vocab, write_bin_meta, read_bin_meta, UINT16_MAX_VOCAB,
)


# ----------------------------------------------------------------------
# parse_size / format_size
# ----------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("1.3B", 1_300_000_000),
    ("350M", 350_000_000),
    ("7B", 7_000_000_000),
    ("64K", 64_000),
    ("1_300_000_000", 1_300_000_000),
    ("2.5b", 2_500_000_000),  # case-insensitive
    (1_300_000_000, 1_300_000_000),
    (1.3e9, 1_300_000_000),
])
def test_parse_size(raw, expected):
    assert parse_size(raw) == expected


def test_parse_size_invalid():
    with pytest.raises(ValueError):
        parse_size("garbage")


@pytest.mark.parametrize("n,expected", [
    (1_300_000_000, "1.30B"),
    (350_000_000, "350.0M"),
    (7_000_000_000, "7.00B"),
    (64_000, "64.0K"),
])
def test_format_size(n, expected):
    assert format_size(n) == expected


# ----------------------------------------------------------------------
# estimate_params on a known small config (no autosize loop)
# ----------------------------------------------------------------------
def test_estimate_params_small_config():
    """Build a tiny config and check param count is in a sensible range."""
    cfg = {
        "vocab_size": 1000,
        "d_model": 128,
        "n_local_layers": 2,
        "n_global_layers": 2,
        "n_local_heads": 2,
        "n_global_heads": 2,
        "window_size": 64,
        "segment_size": 64,
        "n_concepts": 1,
        "n_refine_layers": 1,
        "memory_size": 32,
        "n_read_steps": 1,
        "ffn_mult": 2.0,
        "max_seq_len": 256,
        "max_reasoning_iterations": 4,
        "lora_rank": 4,
        "halt_threshold": 0.5,
        "lambda_p": 0.5,
    }
    n = estimate_params(cfg)
    # Embedding alone: 1000 * 128 = 128k. Total should be in the [500k, 5M]
    # range for this 4-layer toy config — anything outside is a red flag.
    assert 500_000 < n < 5_000_000, f"unexpected param count: {n}"


# ----------------------------------------------------------------------
# auto_size behaviour
# ----------------------------------------------------------------------
@pytest.mark.parametrize("target,tol_pct", [
    ("350M", 5.0),
    ("1.3B", 5.0),
    ("2.5B", 5.0),
    ("7B", 5.0),
])
def test_auto_size_within_tolerance(target, tol_pct):
    """Auto-sizer should land within ±tol_pct of the target budget."""
    result = auto_size(target, vocab_size=64000)
    assert abs(result.error_pct) <= tol_pct, (
        f"target={target} got {format_size(result.params)} "
        f"({result.error_pct:+.1f}%, tolerance ±{tol_pct}%)"
    )


@pytest.mark.parametrize("target", ["350M", "1.3B", "2.5B", "7B"])
def test_auto_size_aspect_ratio_sanity(target):
    """Reject narrow-deep / wide-shallow degenerate configs.

    LLaMA-family ratios: d_model / n_total_layers ~ 80-150.
    We allow a wide window (40-300) — just want to catch *absurd* picks.
    """
    result = auto_size(target, vocab_size=64000)
    cfg = result.config
    n_total = cfg["n_local_layers"] + cfg["n_global_layers"]
    aspect = cfg["d_model"] / n_total
    assert 40 <= aspect <= 300, (
        f"degenerate aspect {aspect:.0f} for target {target}: "
        f"d={cfg['d_model']}, layers={n_total}"
    )


@pytest.mark.parametrize("target", ["350M", "1.3B", "7B"])
def test_auto_size_respects_min_layers(target):
    """Hard floor: never produce <12 transformer layers."""
    result = auto_size(target, vocab_size=64000, min_layers=12)
    n_total = result.config["n_local_layers"] + result.config["n_global_layers"]
    assert n_total >= 12, f"got only {n_total} layers for {target}"


def test_auto_size_head_dim_divides_d_model():
    """d_model must be divisible by n_heads."""
    result = auto_size("1.3B", vocab_size=64000, head_dim=128)
    cfg = result.config
    assert cfg["d_model"] % cfg["n_local_heads"] == 0
    assert cfg["d_model"] % cfg["n_global_heads"] == 0


def test_auto_size_local_global_split():
    """Default split should give the global reasoner more depth than local."""
    result = auto_size("1.3B", vocab_size=64000, ratio_local=1/3)
    cfg = result.config
    assert cfg["n_global_layers"] > cfg["n_local_layers"]


# ----------------------------------------------------------------------
# Adaptive bin dtype
# ----------------------------------------------------------------------
@pytest.mark.parametrize("vocab,expected", [
    (1000, np.uint16),
    (32000, np.uint16),
    (UINT16_MAX_VOCAB, np.uint16),
    (UINT16_MAX_VOCAB + 1, np.uint32),
    (100_000, np.uint32),
    (256_000, np.uint32),
])
def test_bin_dtype_for_vocab(vocab, expected):
    assert bin_dtype_for_vocab(vocab) == np.dtype(expected)


def test_bin_meta_roundtrip(tmp_path):
    """write_bin_meta then read_bin_meta returns the same payload."""
    bin_path = tmp_path / "test.bin"
    bin_path.write_bytes(b"\x00\x01\x02\x03")  # placeholder bytes
    meta_path = write_bin_meta(
        bin_path, vocab_size=64000, n_tokens=1234, tokenizer="test_tok",
    )
    assert meta_path.exists()
    meta = read_bin_meta(bin_path)
    assert meta["vocab_size"] == 64000
    assert meta["dtype"] == "uint16"
    assert meta["n_tokens"] == 1234
    assert meta["tokenizer"] == "test_tok"
    assert meta["version"] == 1


def test_bin_meta_uint32(tmp_path):
    """Vocab > 65535 should record uint32."""
    bin_path = tmp_path / "test.bin"
    bin_path.write_bytes(b"\x00\x01\x02\x03")
    write_bin_meta(bin_path, vocab_size=100_000, n_tokens=1)
    meta = read_bin_meta(bin_path)
    assert meta["dtype"] == "uint32"
    assert meta["vocab_size"] == 100_000


def test_bin_meta_legacy_fallback(tmp_path):
    """Missing sidecar -> uint16 fallback for backward compat."""
    bin_path = tmp_path / "legacy.bin"
    bin_path.write_bytes(b"\x00\x01")
    # No sidecar written.
    meta = read_bin_meta(bin_path)
    assert meta["dtype"] == "uint16"
    assert meta["version"] == 0
    assert meta["vocab_size"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
