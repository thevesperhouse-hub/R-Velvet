"""
Tokenize dataset into a binary .bin file for training.

Supports:
  - Plain text files (.txt)
  - JSONL files (.jsonl)
  - Parquet directories (*.parquet)

Auto-selects uint16 (vocab < 65536) or uint32.

Usage:
    # From parquet shards (Vesper-FR)
    python scripts/tokenize_data.py \
        --input data/vesper_edu_fr_parquet \
        --output data/train.bin \
        --tokenizer data/velvet_tok_100k_unigram

    # From JSONL
    python scripts/tokenize_data.py \
        --input data/vesper_edu_fr_dedup.jsonl \
        --output data/train.bin \
        --tokenizer data/velvet_tok_100k_unigram \
        --text-field text

    # From plain text
    python scripts/tokenize_data.py \
        --input data/corpus.txt \
        --output data/train.bin \
        --tokenizer data/velvet_tok_100k_unigram
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer
    path = Path(name_or_path)
    if path.is_dir():
        print(f"Loading local tokenizer: {name_or_path}")
        return AutoTokenizer.from_pretrained(str(path))
    else:
        print(f"Loading HuggingFace tokenizer: {name_or_path}")
        return AutoTokenizer.from_pretrained(name_or_path)


def iter_texts_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_texts_jsonl(path, text_field):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get(text_field, "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def iter_texts_parquet(path, text_field):
    import pyarrow.parquet as pq
    parquet_dir = Path(path)
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet files in {parquet_dir}")
    print(f"  Found {len(shards)} parquet shards")
    for shard in shards:
        table = pq.read_table(shard, columns=[text_field])
        col = table.column(text_field)
        for val in col:
            text = val.as_py()
            if text:
                yield text


def count_items(path, text_field):
    """Quick count for progress bar."""
    p = Path(path)
    if p.is_dir():
        import pyarrow.parquet as pq
        total = 0
        for shard in sorted(p.glob("*.parquet")):
            total += pq.ParquetFile(shard).metadata.num_rows
        return total
    elif p.suffix == ".jsonl":
        n = 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
        return n
    else:
        n = 0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
        return n


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset → .bin")
    parser.add_argument("--input", required=True,
                        help="Input: .txt file, .jsonl file, or parquet directory")
    parser.add_argument("--output", required=True, help="Output .bin file")
    parser.add_argument("--tokenizer", default="data/velvet_tok_100k_unigram",
                        help="Tokenizer name or local path")
    parser.add_argument("--text-field", default="text",
                        help="Text field name (for JSONL/parquet)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Docs to tokenize at once")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    print(f"  vocab_size={vocab_size}, dtype={dtype.__name__}")

    # Detect input type
    input_path = Path(args.input)
    if input_path.is_dir():
        print(f"Input: parquet directory")
        text_iter = iter_texts_parquet(args.input, args.text_field)
    elif input_path.suffix == ".jsonl":
        print(f"Input: JSONL file")
        text_iter = iter_texts_jsonl(args.input, args.text_field)
    else:
        print(f"Input: text file")
        text_iter = iter_texts_txt(args.input)

    total = count_items(args.input, args.text_field)
    print(f"  Total docs: {total:,}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0
    eos_id = tokenizer.eos_token_id or 0

    with open(args.output, "wb") as f_out:
        batch = []
        pbar = tqdm(text_iter, total=total, unit="docs", desc="Tokenizing")
        for text in pbar:
            batch.append(text)

            if len(batch) >= args.batch_size:
                # Batch tokenize
                encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
                all_ids = []
                for ids in encoded:
                    all_ids.extend(ids)
                    all_ids.append(eos_id)
                arr = np.array(all_ids, dtype=dtype)
                arr.tofile(f_out)
                total_tokens += len(all_ids)
                pbar.set_postfix(tokens=f"{total_tokens/1e9:.2f}B")
                batch = []

        # Flush remaining
        if batch:
            encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
            all_ids = []
            for ids in encoded:
                all_ids.extend(ids)
                all_ids.append(eos_id)
            arr = np.array(all_ids, dtype=dtype)
            arr.tofile(f_out)
            total_tokens += len(all_ids)

        pbar.close()

    # Write companion .dtype file for TextDataset auto-detection
    dtype_path = Path(args.output).with_suffix(".dtype")
    dtype_path.write_text(dtype.__name__)

    size_gb = Path(args.output).stat().st_size / 1e9
    print(f"\nDone: {total_tokens:,} tokens ({total_tokens/1e9:.1f}B)")
    print(f"Output: {args.output} ({size_gb:.1f} GB)")
    print(f"Dtype: {dtype.__name__} ({dtype().itemsize} bytes/token)")
    print(f"Dtype file: {dtype_path}")


if __name__ == "__main__":
    main()
