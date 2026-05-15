"""Convert JSONL dataset to Parquet format.

Streams in chunks to avoid loading everything in memory.

Usage:
    python scripts/jsonl_to_parquet.py --input data/vesper_edu_fr_dedup.jsonl \
        --output data/vesper_edu_fr.parquet

    # With row group size control
    python scripts/jsonl_to_parquet.py --input data/vesper_edu_fr_dedup.jsonl \
        --output data/vesper_edu_fr.parquet --chunk-size 500000
"""

import argparse
import json
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to Parquet")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Rows per chunk (default 500k)")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    print(f"Converting {args.input} -> {args.output}")
    print(f"Chunk size: {args.chunk_size:,}")

    t0 = time.time()
    writer = None
    n_total = 0
    chunk = []

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        pbar = tqdm(f, unit="docs", desc="Converting")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            chunk.append(record)
            n_total += 1

            if len(chunk) >= args.chunk_size:
                table = pa.Table.from_pylist(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(args.output, table.schema,
                                              compression="zstd")
                writer.write_table(table)
                pbar.set_postfix(written=f"{n_total:,}")
                chunk = []

        # Write remaining
        if chunk:
            table = pa.Table.from_pylist(chunk)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema,
                                          compression="zstd")
            writer.write_table(table)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    import os
    size_mb = os.path.getsize(args.output) / 1e6

    print(f"\nDone: {n_total:,} rows in {elapsed:.0f}s")
    print(f"Output: {args.output} ({size_mb:,.0f} MB)")


if __name__ == "__main__":
    main()
