"""Convert JSONL dataset to sharded Parquet files.

Outputs multiple parquet files (~500MB each) for HF compatibility.

Usage:
    python scripts/jsonl_to_parquet.py --input data/vesper_edu_fr_dedup.jsonl \
        --output-dir data/vesper_edu_fr_parquet

    # Custom shard size
    python scripts/jsonl_to_parquet.py --input data/vesper_edu_fr_dedup.jsonl \
        --output-dir data/vesper_edu_fr_parquet --rows-per-shard 500000
"""

import argparse
import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to sharded Parquet")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for parquet shards")
    parser.add_argument("--rows-per-shard", type=int, default=500_000,
                        help="Rows per shard file (default 500k, ~500MB-1GB each)")
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Converting {args.input} -> {args.output_dir}/")
    print(f"Rows per shard: {args.rows_per_shard:,}")

    t0 = time.time()
    n_total = 0
    n_shards = 0
    chunk = []
    schema = None

    def write_shard(records, shard_idx, schema):
        table = pa.Table.from_pylist(records)
        if schema is None:
            schema = table.schema
        n_total_shards = "XXXXX"  # placeholder, renamed at the end
        shard_path = os.path.join(
            args.output_dir,
            f"train-{shard_idx:05d}-of-{n_total_shards}.parquet"
        )
        # Write to temp name first
        tmp_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, tmp_path, compression="zstd")
        size_mb = os.path.getsize(tmp_path) / 1e6
        print(f"  Shard {shard_idx:03d}: {len(records):,} rows ({size_mb:.0f} MB)")
        return schema

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        pbar = tqdm(f, unit="docs", desc="Reading")
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

            if len(chunk) >= args.rows_per_shard:
                schema = write_shard(chunk, n_shards, schema)
                n_shards += 1
                pbar.set_postfix(shards=n_shards, rows=f"{n_total:,}")
                chunk = []

        # Write remaining
        if chunk:
            schema = write_shard(chunk, n_shards, schema)
            n_shards += 1

        pbar.close()

    # Rename shards to final names: train-00000-of-00040.parquet
    total_str = f"{n_shards:05d}"
    total_size = 0
    for i in range(n_shards):
        tmp_path = os.path.join(args.output_dir, f"shard_{i:05d}.parquet")
        final_name = f"train-{i:05d}-of-{total_str}.parquet"
        final_path = os.path.join(args.output_dir, final_name)
        os.rename(tmp_path, final_path)
        total_size += os.path.getsize(final_path)

    elapsed = time.time() - t0

    print(f"\nDone: {n_total:,} rows -> {n_shards} shards in {elapsed:.0f}s")
    print(f"Total size: {total_size / 1e9:.1f} GB")
    print(f"Output: {args.output_dir}/")

    # List shards
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".parquet"):
            size = os.path.getsize(os.path.join(args.output_dir, f)) / 1e6
            print(f"  {f} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
