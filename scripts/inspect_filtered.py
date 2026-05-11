"""Inspect filtered dataset: show samples and score distributions.

Usage:
    python scripts/inspect_filtered.py --file data/test_filter.jsonl
    python scripts/inspect_filtered.py --file data/test_filter.jsonl --last 10
    python scripts/inspect_filtered.py --file data/vesper_edu_fr.jsonl --worst 5
"""

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--last", type=int, default=5, help="Show last N entries")
    parser.add_argument("--best", type=int, default=0, help="Show N highest-scored entries")
    parser.add_argument("--worst", type=int, default=0, help="Show N lowest-scored entries")
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
        print("No records found.")
        return

    print(f"Total filtered docs: {len(records):,}")

    # Score stats
    score_keys = [k for k in records[0] if k.startswith("score_")]
    if score_keys:
        print(f"\nConfidence scores (from fasttext):")
        for key in sorted(score_keys):
            vals = [r[key] for r in records if key in r]
            if vals:
                avg = sum(vals) / len(vals)
                mn = min(vals)
                mx = max(vals)
                print(f"  {key:20s}: avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}")

    # Text length stats
    lengths = [len(r.get("text", "")) for r in records]
    print(f"\nText lengths:")
    print(f"  avg={sum(lengths)/len(lengths):.0f}  min={min(lengths)}  max={max(lengths)}")

    def show_record(r, idx=None):
        prefix = f"[{idx}] " if idx is not None else ""
        scores = " ".join(f"{k}={v:.2f}" for k, v in r.items() if k.startswith("score_"))
        print(f"{prefix}{scores}")
        print(f"  {r['text'][:300]}")
        print()

    # Last N
    print(f"\n=== Last {args.last} ===")
    for r in records[-args.last:]:
        show_record(r)

    # Best N (by sum of scores)
    if args.best > 0:
        scored = [(sum(v for k, v in r.items() if k.startswith("score_")), r) for r in records]
        scored.sort(key=lambda x: -x[0])
        print(f"\n=== Best {args.best} ===")
        for score, r in scored[:args.best]:
            show_record(r)

    # Worst N
    if args.worst > 0:
        scored = [(sum(v for k, v in r.items() if k.startswith("score_")), r) for r in records]
        scored.sort(key=lambda x: x[0])
        print(f"\n=== Worst {args.worst} ===")
        for score, r in scored[:args.worst]:
            show_record(r)


if __name__ == "__main__":
    main()
