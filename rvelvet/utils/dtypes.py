"""Adaptive dtype helpers for tokenized .bin files.

For vocab ≤ 65535 we use uint16 (2 bytes/token) — half the disk size and IO.
For larger vocabs we fall back to uint32 (4 bytes/token).

The chosen dtype is recorded in a sidecar JSON next to the .bin file so the
TextDataset can recover it without ambiguity. Older .bin files without a
sidecar default to uint16 for backward compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# uint16 holds 0..65535 — that's `vocab_size` ≤ 65536 in practice
# (token id = vocab_size - 1 must fit). Using ≤ 65535 keeps a margin.
UINT16_MAX_VOCAB = 65535


def bin_dtype_for_vocab(vocab_size: int) -> np.dtype:
    """Return the smallest unsigned-int dtype that can hold all token ids."""
    if vocab_size <= UINT16_MAX_VOCAB:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)


def _meta_path(bin_path) -> Path:
    """Sidecar location: foo.bin -> foo.bin.meta.json."""
    p = Path(bin_path)
    return p.with_suffix(p.suffix + ".meta.json")


def write_bin_meta(bin_path, *, vocab_size: int, n_tokens: int,
                   tokenizer: Optional[str] = None,
                   extra: Optional[dict] = None) -> Path:
    """Write a sidecar JSON describing how to read the .bin file."""
    meta = {
        "version": 1,
        "vocab_size": int(vocab_size),
        "dtype": bin_dtype_for_vocab(vocab_size).name,
        "n_tokens": int(n_tokens),
        "tokenizer": tokenizer,
    }
    if extra:
        meta.update(extra)

    path = _meta_path(bin_path)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def read_bin_meta(bin_path) -> dict:
    """Read sidecar metadata for a .bin. Falls back to uint16 if missing."""
    path = _meta_path(bin_path)
    if not path.exists():
        # Legacy .bin without sidecar — assume uint16 (the original format).
        return {"version": 0, "dtype": "uint16", "vocab_size": None}
    return json.loads(path.read_text(encoding="utf-8"))
