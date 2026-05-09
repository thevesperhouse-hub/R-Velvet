"""Inspect annotations: show recent entries, stats, and examples per score.

Usage:
    python scripts/inspect_annotations.py --file data/annotations_fr.jsonl
    python scripts/inspect_annotations.py --file data/annotations_fr.jsonl --last 10
    python scripts/inspect_annotations.py --file data/annotations_fr.jsonl --show-score 0
    python scripts/inspect_annotations.py --file data/annotations_fr.jsonl --show-score 10
"""

import argparse
import json
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

DIMS = ["coherence", "pedagogy", "linguistic", "depth", "factuality", "code_quality"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="annotations JSONL")
    parser.add_argument("--last", type=int, default=5, help="Show last N entries")
    parser.add_argument("--show-score", type=int, default=None,
                        help="Show examples with this total score")
    parser.add_argument("--show-dim", type=str, default=None,
                        help="Show examples where this dimension = 0 (worst)")
    args = parser.parse_args()

    records = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print("No annotations found.")
        return

    # Stats
    print(f"Total annotations: {len(records):,}")
    print()

    # Dimension averages
    print("Dimension averages (0-5):")
    for dim in DIMS:
        vals = [r[dim] for r in records if dim in r]
        if vals:
            avg = sum(vals) / len(vals)
            bar = "#" * int(avg * 8)
            print(f"  {dim:15s}: {avg:.3f}/5.0  {bar}  (n={len(vals):,})")

    # Total distribution
    print()
    print("Total score distribution (0-25):")
    totals = [r["total"] for r in records if "total" in r]
    counts = Counter(totals)
    for s in range(26):
        c = counts.get(s, 0)
        pct = c / len(totals) * 100 if totals else 0
        bar = "#" * int(pct)
        print(f"  {s:2d}: {c:>5,} ({pct:4.1f}%) {bar}")

    # Last N
    print()
    print(f"=== Last {args.last} annotations ===")
    for r in records[-args.last:]:
        dims_str = " ".join(
            f"{d[:4]}={r[d]}" for d in DIMS if d in r
        )
        print(f"[total={r.get('total','?')}] {dims_str}")
        print(f"  {r['text'][:250]}")
        print()

    # Show specific score
    if args.show_score is not None:
        target = args.show_score
        examples = [r for r in records if r.get("total") == target]
        print(f"=== Examples with total={target} ({len(examples)} found) ===")
        for r in examples[:5]:
            dims_str = " ".join(f"{d[:4]}={r[d]}" for d in DIMS if d in r)
            print(f"  [{dims_str}]")
            print(f"  {r['text'][:300]}")
            print()

    # Show worst on a dimension
    if args.show_dim:
        dim = args.show_dim
        examples = [r for r in records if r.get(dim) == 0]
        print(f"=== Examples with {dim}=0 ({len(examples)} found) ===")
        for r in examples[:5]:
            dims_str = " ".join(f"{d[:4]}={r[d]}" for d in DIMS if d in r)
            print(f"  [{dims_str}]")
            print(f"  {r['text'][:300]}")
            print()


if __name__ == "__main__":
    main()
