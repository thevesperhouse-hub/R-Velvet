"""Deduplicate a JSONL dataset.

Three levels of dedup (all enabled by default):
  1. Exact:  identical text (MD5 hash)
  2. URL:    same URL = same content
  3. Fuzzy:  near-duplicates via normalized prefix hash (catches same
             article with different headers/footers)

Streams line by line — only hashes are kept in memory (~500MB for 20M docs).

Usage:
    python scripts/dedup_dataset.py --input data/vesper_edu_fr.jsonl \
        --output data/vesper_edu_fr_dedup.jsonl

    # Exact only (fastest)
    python scripts/dedup_dataset.py --input data/vesper_edu_fr.jsonl \
        --output data/vesper_edu_fr_dedup.jsonl --no-url --no-fuzzy
"""

import argparse
import hashlib
import json
import os
import sys
import time

from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def normalize(text, n_chars=1000):
    """Normalize text for fuzzy matching: lowercase, collapse whitespace, take first N chars."""
    t = " ".join(text.lower().split())
    return t[:n_chars]


def hash_text(text):
    """MD5 hash of full text."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).digest()


def hash_prefix(text, n_chars=1000):
    """MD5 hash of normalized prefix — catches near-duplicates."""
    return hashlib.md5(normalize(text, n_chars).encode("utf-8", errors="replace")).digest()


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSONL dataset")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL (deduped)")
    parser.add_argument("--no-url", action="store_true", help="Disable URL dedup")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy/prefix dedup")
    parser.add_argument("--fuzzy-chars", type=int, default=1000,
                        help="Number of chars for fuzzy prefix (default 1000)")
    args = parser.parse_args()

    seen_text = set()
    seen_url = set()
    seen_prefix = set()

    n_total = 0
    n_kept = 0
    n_exact = 0
    n_url = 0
    n_fuzzy = 0
    n_parse_err = 0

    t0 = time.time()

    # Count total lines for tqdm (fast: just count newlines via file size estimate)
    file_size = os.path.getsize(args.input)

    with open(args.input, "r", encoding="utf-8", errors="replace") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:

        pbar = tqdm(f_in, unit="docs", desc="Dedup",
                    total=None)  # unknown total, shows rate + count

        for line in pbar:
            n_total += 1
            n_removed = n_exact + n_url + n_fuzzy
            if n_total % 10_000 == 0:
                dup_pct = n_removed / max(n_total, 1) * 100
                pbar.set_postfix(
                    kept=f"{n_kept:,}",
                    dupes=f"{n_removed:,} ({dup_pct:.1f}%)",
                    exact=n_exact, url=n_url, fuzzy=n_fuzzy
                )

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_parse_err += 1
                continue

            text = record.get("text", "")
            if not text:
                continue

            # Check 1: exact text hash
            h = hash_text(text)
            if h in seen_text:
                n_exact += 1
                continue
            seen_text.add(h)

            # Check 2: URL dedup
            if not args.no_url:
                url = record.get("url", "")
                if url:
                    if url in seen_url:
                        n_url += 1
                        continue
                    seen_url.add(url)

            # Check 3: fuzzy prefix dedup
            if not args.no_fuzzy:
                ph = hash_prefix(text, args.fuzzy_chars)
                if ph in seen_prefix:
                    n_fuzzy += 1
                    continue
                seen_prefix.add(ph)

            f_out.write(line + "\n")
            n_kept += 1

        pbar.close()

    elapsed = time.time() - t0
    n_removed = n_total - n_kept - n_parse_err
    dup_pct = n_removed / max(n_total, 1) * 100

    print(f"\n{'='*60}")
    print(f"Deduplication complete")
    print(f"{'='*60}")
    print(f"  Input:         {n_total:,} docs")
    print(f"  Kept:          {n_kept:,} docs")
    print(f"  Removed:       {n_removed:,} ({dup_pct:.2f}%)")
    print(f"    Exact dupes: {n_exact:,}")
    print(f"    URL dupes:   {n_url:,}")
    print(f"    Fuzzy dupes: {n_fuzzy:,}")
    print(f"    Parse errs:  {n_parse_err:,}")
    print(f"  Time:          {elapsed/60:.1f}min ({n_total/elapsed:,.0f} lines/s)")
    print(f"  Output:        {args.output}")

    mem_mb = (len(seen_text) * 16 + len(seen_url) * 50 + len(seen_prefix) * 16) / 1e6
    print(f"  Memory (hashes): ~{mem_mb:,.0f} MB")


if __name__ == "__main__":
    main()
