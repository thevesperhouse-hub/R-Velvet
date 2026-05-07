"""Streaming multi-source text dataset for large-scale training.

Loads N HuggingFace datasets in streaming mode, interleaves them with given
probabilities, tokenizes on-the-fly, packs across documents into fixed-length
sequences, and yields (input_ids, targets) pairs.

Used when the training corpus is too big to fit a pre-tokenized .bin file
(typical for >50B tokens). For small/debug runs, prefer TextDataset.

Sharding across DataLoader workers is handled via worker_info: each worker
gets a distinct seed and skips examples not in its shard, so workers don't
duplicate data.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


class StreamingTextDataset(IterableDataset):
    """Multi-source streaming dataset with on-the-fly tokenization + packing.

    Args:
        sources: list of dicts, each:
            {
                "path": "HuggingFaceFW/fineweb-2",   # HF dataset id
                "name": "fra_Latn",                  # subset (optional)
                "split": "train",
                "weight": 0.6,                       # sampling probability
                "text_field": "text",                # which field to tokenize
            }
            Weights are normalized internally; absolute scale doesn't matter.

        tokenizer: HF AutoTokenizer-compatible (must have `encode` returning ids)
        seq_len: output sequence length (input_ids has seq_len, targets shifted)
        shuffle_buffer: shuffle buffer size for the interleaved stream
        seed: master seed; per-worker seeds are derived from it
        eos_token_id: inserted between documents (default: tokenizer.eos_token_id
                      or a fallback constant if absent)
        max_doc_tokens: hard cap on tokens taken from any single document, to
                        avoid one giant document monopolizing the batch.
                        None = unlimited.
        stopping_strategy: "all_exhausted" loops the smallest source until the
                           biggest is exhausted (= one epoch over the union).
                           "first_exhausted" stops at the smallest. We default
                           to "all_exhausted" so weights are respected.
    """

    def __init__(
        self,
        sources: List[dict],
        tokenizer,
        seq_len: int = 2048,
        shuffle_buffer: int = 10_000,
        seed: int = 0,
        eos_token_id: Optional[int] = None,
        max_doc_tokens: Optional[int] = 65536,
        stopping_strategy: str = "all_exhausted",
    ):
        super().__init__()
        if not sources:
            raise ValueError("StreamingTextDataset requires at least one source.")

        self.sources = sources
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.max_doc_tokens = max_doc_tokens
        self.stopping_strategy = stopping_strategy

        # Resolve EOS once. Fall back to a sentinel if the tokenizer has none —
        # without a doc separator, packed sequences would silently bleed across
        # unrelated documents and harm long-range learning.
        if eos_token_id is None:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                eos = getattr(tokenizer, "sep_token_id", None)
            if eos is None:
                eos = 0  # last-resort; tokenizer ought to have one
        self.eos_token_id = int(eos_token_id if eos_token_id is not None else eos)

        # Normalize weights up-front so the user can pass any positive scale.
        weights = np.array([float(s.get("weight", 1.0)) for s in sources], dtype=np.float64)
        if (weights <= 0).any():
            raise ValueError("All source weights must be > 0.")
        self._probs = (weights / weights.sum()).tolist()

    # ------------------------------------------------------------------
    # Stream construction
    # ------------------------------------------------------------------
    def _build_stream(self, worker_seed: int):
        """Build the interleaved + shuffled HF stream for this worker."""
        from datasets import load_dataset, interleave_datasets

        streams = []
        for src in self.sources:
            ds = load_dataset(
                src["path"],
                name=src.get("name"),
                split=src.get("split", "train"),
                streaming=True,
                data_files=src.get("data_files"),
            )
            streams.append(ds)

        if len(streams) == 1:
            interleaved = streams[0]
        else:
            interleaved = interleave_datasets(
                streams,
                probabilities=self._probs,
                seed=worker_seed,
                stopping_strategy=self.stopping_strategy,
            )

        # Shuffle buffer to break up locality after interleave; without this,
        # successive samples often come from the same source / shard.
        if self.shuffle_buffer > 0:
            interleaved = interleaved.shuffle(
                seed=worker_seed,
                buffer_size=self.shuffle_buffer,
            )
        return interleaved

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading.
            worker_id, num_workers = 0, 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Distinct seed per worker so they don't sample identical streams.
        worker_seed = self.seed + worker_id * 100003

        stream = self._build_stream(worker_seed)

        # Map source list → text field selectors keyed by "path" so we can pick
        # the right field for each example. (Interleaved stream loses provenance
        # so we read both common field names with a fallback.)
        primary_fields = list({s.get("text_field", "text") for s in self.sources})

        # Buffer of tokenized data from which we cut seq_len+1 chunks.
        token_buffer: List[int] = []
        chunk_len = self.seq_len + 1
        rng = random.Random(worker_seed)

        for i, example in enumerate(stream):
            # Worker sharding: each worker keeps every num_workers-th example.
            # interleave_datasets+shuffle doesn't shard automatically.
            if num_workers > 1 and (i % num_workers) != worker_id:
                continue

            text = None
            for field in primary_fields:
                if field in example:
                    text = example[field]
                    if text:
                        break
            if not text:
                continue

            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            if self.max_doc_tokens is not None and len(ids) > self.max_doc_tokens:
                # Random crop instead of head: avoids systematically losing the
                # tail of long documents.
                start = rng.randrange(0, len(ids) - self.max_doc_tokens + 1)
                ids = ids[start:start + self.max_doc_tokens]

            token_buffer.extend(ids)
            token_buffer.append(self.eos_token_id)

            # Drain the buffer in chunks of (seq_len + 1).
            while len(token_buffer) >= chunk_len:
                chunk = token_buffer[:chunk_len]
                token_buffer = token_buffer[chunk_len:]
                arr = np.asarray(chunk, dtype=np.int64)
                yield {
                    "input_ids": torch.from_numpy(arr[:-1]),
                    "targets": torch.from_numpy(arr[1:]),
                }

        # Tail flush: pad the last incomplete chunk so workers always emit at
        # least one final sample even if the stream ended mid-document.
        if len(token_buffer) >= 2:
            if len(token_buffer) < chunk_len:
                pad_id = self.eos_token_id
                token_buffer.extend([pad_id] * (chunk_len - len(token_buffer)))
            arr = np.asarray(token_buffer[:chunk_len], dtype=np.int64)
            yield {
                "input_ids": torch.from_numpy(arr[:-1]),
                "targets": torch.from_numpy(arr[1:]),
            }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, data_cfg, tokenizer, seed: int = 0) -> "StreamingTextDataset":
        """Build a StreamingTextDataset from a data YAML section.

        Expected fields in data_cfg:
            sources: [ {path, name, split, weight, text_field}, ... ]
            seq_len: int
            shuffle_buffer: int (optional, default 10_000)
            max_doc_tokens: int or null (optional)
            stopping_strategy: "all_exhausted" | "first_exhausted" (optional)
        """
        return cls(
            sources=list(data_cfg.sources),
            tokenizer=tokenizer,
            seq_len=int(data_cfg.seq_len),
            shuffle_buffer=int(getattr(data_cfg, "shuffle_buffer", 10_000)),
            seed=seed,
            eos_token_id=getattr(data_cfg, "eos_token_id", None),
            max_doc_tokens=getattr(data_cfg, "max_doc_tokens", 65536),
            stopping_strategy=getattr(data_cfg, "stopping_strategy", "all_exhausted"),
        )
