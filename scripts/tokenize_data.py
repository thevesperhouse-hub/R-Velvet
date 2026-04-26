"""
Tokenize raw text into a binary .bin file (uint16) for training.

Supports both HuggingFace model names (e.g. "gpt2") and local tokenizer
directories (e.g. "data/tokenizer_fr" produced by train_tokenizer.py).

Usage:
    # With custom French tokenizer
    python scripts/tokenize_data.py --input data/corpus_fr.txt --output data/train.bin --tokenizer data/tokenizer_fr

    # With HuggingFace tokenizer
    python scripts/tokenize_data.py --input data/corpus_fr.txt --output data/train.bin --tokenizer gpt2

    # Validation set
    python scripts/tokenize_data.py --input data/val_fr.txt --output data/val.bin --tokenizer data/tokenizer_fr
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


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
    parser = argparse.ArgumentParser(description="Tokenize raw text → .bin (uint16)")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output .bin file")
    parser.add_argument("--tokenizer", type=str, default="data/velvet_tok", help="Tokenizer name or local path")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Lines per chunk")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"  vocab_size={vocab_size}")

    if vocab_size > 65535:
        raise ValueError(
            f"Vocab size {vocab_size} exceeds uint16 max (65535). "
            "Use a smaller tokenizer or switch to uint32."
        )

    # Count lines (without loading all into memory)
    print(f"Counting lines in {args.input}...")
    n_lines = 0
    with open(args.input, 'r', encoding='utf-8') as f:
        for _ in f:
            n_lines += 1
    print(f"  {n_lines:,} lines")

    # Tokenize in streaming chunks, write incrementally
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
                arr = np.array(ids, dtype=np.uint16)
                arr.tofile(f_out)
                total_tokens += len(ids)
                chunk_lines = []

        # Remaining
        if chunk_lines:
            text = "".join(chunk_lines)
            ids = tokenizer.encode(text)
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f_out)
            total_tokens += len(ids)

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"\nDone: {total_tokens:,} tokens → {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
