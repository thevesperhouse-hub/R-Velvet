"""
Tokenize raw text into a binary .bin file for training.

Picks the smallest dtype that can hold all token ids:
    vocab ≤ 65535 → uint16 (2 bytes/token)
    vocab > 65535 → uint32 (4 bytes/token)

A sidecar `<output>.meta.json` records the dtype + vocab so TextDataset can
read the file unambiguously. Older .bin files without sidecar are still
readable as uint16.

Usage:
    # With custom French tokenizer (any vocab size)
    python scripts/tokenize_data.py --input data/corpus_fr.txt --output data/train.bin --tokenizer data/velvet_tok

    # With HuggingFace tokenizer
    python scripts/tokenize_data.py --input data/corpus_fr.txt --output data/train.bin --tokenizer gpt2
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Make the rvelvet package importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rvelvet.utils.dtypes import bin_dtype_for_vocab, write_bin_meta


def load_tokenizer(name_or_path: str):
    """Load tokenizer from HuggingFace name or local directory."""
    from transformers import AutoTokenizer

    path = Path(name_or_path)
    if path.is_dir():
        print(f"Loading local tokenizer: {name_or_path}")
        return AutoTokenizer.from_pretrained(str(path))
    else:
        print(f"Loading HuggingFace tokenizer: {name_or_path}")
        return AutoTokenizer.from_pretrained(name_or_path)


def main():
    parser = argparse.ArgumentParser(description="Tokenize raw text → .bin")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output .bin file")
    parser.add_argument("--tokenizer", type=str, default="data/velvet_tok_100k_unigram", help="Tokenizer name or local path")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Lines per chunk")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    dtype = bin_dtype_for_vocab(vocab_size)
    print(f"  vocab_size={vocab_size:,} → dtype={dtype.name} "
          f"({dtype.itemsize} bytes/token)")

    print(f"Counting lines in {args.input}...")
    n_lines = 0
    with open(args.input, 'r', encoding='utf-8') as f:
        for _ in f:
            n_lines += 1
    print(f"  {n_lines:,} lines")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0

    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(args.output, 'wb') as f_out:

        chunk_lines = []
        for line in tqdm(f_in, total=n_lines, desc="Tokenizing"):
            chunk_lines.append(line)

            if len(chunk_lines) >= args.chunk_size:
                text = "".join(chunk_lines)
                ids = tokenizer.encode(text)
                arr = np.array(ids, dtype=dtype)
                arr.tofile(f_out)
                total_tokens += len(ids)
                chunk_lines = []

        if chunk_lines:
            text = "".join(chunk_lines)
            ids = tokenizer.encode(text)
            arr = np.array(ids, dtype=dtype)
            arr.tofile(f_out)
            total_tokens += len(ids)

    meta_path = write_bin_meta(
        args.output, vocab_size=vocab_size, n_tokens=total_tokens,
        tokenizer=str(args.tokenizer),
    )
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nDone: {total_tokens:,} tokens → {args.output} ({size_mb:.1f} MB)")
    print(f"  Sidecar: {meta_path}")


if __name__ == "__main__":
    main()
