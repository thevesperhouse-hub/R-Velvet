"""Peek at JSONL files without loading everything in memory.

Usage:
    python scripts/peek.py data/vesper_edu_fr.jsonl
    python scripts/peek.py data/vesper_edu_fr.jsonl --head 10
    python scripts/peek.py data/vesper_edu_fr.jsonl --tail 10
    python scripts/peek.py data/vesper_edu_fr.jsonl --count
    python scripts/peek.py data/vesper_edu_fr.jsonl --sample 5
"""

import argparse
import collections
import json
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def show_record(r, idx=None):
    prefix = f"[{idx}] " if idx is not None else ""
    scores = " ".join(f"{k}={v:.2f}" for k, v in sorted(r.items()) if k.startswith("score_"))
    url = r.get("url", "")
    print(f"{prefix}{scores}")
    if url:
        print(f"  url: {url}")
    print(f"  {r.get('text', '')[:300]}")
    print()


def count_lines(path):
    """Count lines without loading file into memory."""
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
    return n


def read_head(path, n):
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def read_tail(path, n):
    """Read last N lines using a deque (constant memory)."""
    buf = collections.deque(maxlen=n)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                buf.append(line)
    records = []
    for line in buf:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def read_sample(path, n, total=None):
    """Reservoir sampling: pick N random records in one pass."""
    if total is None:
        total = count_lines(path)
    if total <= n:
        return read_head(path, total)
    # Pick N random line indices
    indices = set(sorted(random.sample(range(total), n)))
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i in indices:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if i > max(indices):
                break
    return records


def main():
    parser = argparse.ArgumentParser(description="Peek at JSONL files (no full load)")
    parser.add_argument("file", help="JSONL file to inspect")
    parser.add_argument("--head", type=int, default=0, help="Show first N records")
    parser.add_argument("--tail", type=int, default=0, help="Show last N records")
    parser.add_argument("--sample", type=int, default=0, help="Show N random records")
    parser.add_argument("--count", action="store_true", help="Count total records")
    args = parser.parse_args()

    # Default: show first 5
    if not args.head and not args.tail and not args.sample and not args.count:
        args.head = 5

    size_mb = os.path.getsize(args.file) / 1e6
    print(f"File: {args.file} ({size_mb:,.0f} MB)")

    if args.count:
        print("Counting...")
        n = count_lines(args.file)
        print(f"Total records: {n:,}")
        return

    if args.head:
        records = read_head(args.file, args.head)
        print(f"\n=== First {len(records)} ===")
        for i, r in enumerate(records):
            show_record(r, i)

    if args.tail:
        records = read_tail(args.file, args.tail)
        print(f"\n=== Last {len(records)} ===")
        for i, r in enumerate(records):
            show_record(r, i)

    if args.sample:
        print("Sampling (counting lines first)...")
        records = read_sample(args.file, args.sample)
        print(f"\n=== Random {len(records)} ===")
        for i, r in enumerate(records):
            show_record(r, i)


if __name__ == "__main__":
    main()
