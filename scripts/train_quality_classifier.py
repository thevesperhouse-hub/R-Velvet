"""Train fasttext classifiers on multi-dimensional LLM annotations.

Pipeline step 2/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs on 5 axes
    2. train_quality_classifier.py — train one fasttext per axis (this script)
    3. filter_dataset.py     — apply classifiers to full corpus

Trains one binary classifier per dimension (score >= 1 = positive).
Also trains a "global" classifier on the total score.

Usage:
    python scripts/train_quality_classifier.py \
        --annotations data/annotations_fr.jsonl \
        --output data/quality_classifier
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


DIMENSIONS = ["coherence", "pedagogy", "linguistic", "depth", "factuality", "code_quality"]


def load_annotations(path: str) -> list:
    """Load JSONL annotations."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "text" in rec and "total" in rec:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def prepare_text(text: str, max_words: int = 500) -> str:
    """Clean text for fasttext."""
    clean = " ".join(text.split())
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words])
    return clean


def train_dimension_classifier(
    train_records, val_records, dim, threshold, output_dir, args
):
    """Train a binary classifier for one dimension."""
    import fasttext

    train_path = output_dir / f"train_{dim}.txt"
    val_path = output_dir / f"val_{dim}.txt"

    def write_data(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                score = rec.get(dim, -1)
                if score < 0:
                    continue
                label = "high" if score >= threshold else "low"
                f.write(f"__label__{label} {prepare_text(rec['text'])}\n")

    write_data(train_records, train_path)
    write_data(val_records, val_path)

    # Check if we have data for this dimension
    n_train = sum(1 for r in train_records if r.get(dim, -1) >= 0)
    if n_train < 100:
        print(f"  {dim}: skipped (only {n_train} samples)")
        return None

    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epochs,
        lr=args.lr,
        wordNgrams=args.wordNgrams,
        dim=args.dim,
        loss="softmax",
        thread=os.cpu_count() or 4,
        verbose=0,
    )

    # Evaluate
    n_val = sum(1 for r in val_records if r.get(dim, -1) >= 0)
    if n_val > 0:
        n, precision, recall = model.test(str(val_path))
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    else:
        precision = recall = f1 = 0.0

    model_path = output_dir / f"classifier_{dim}.bin"
    model.save_model(str(model_path))

    # Cleanup temp files
    train_path.unlink(missing_ok=True)
    val_path.unlink(missing_ok=True)

    return {"precision": precision, "recall": recall, "f1": f1, "n_train": n_train}


def main():
    parser = argparse.ArgumentParser(
        description="Train per-dimension quality classifiers on LLM annotations")
    parser.add_argument("--annotations", type=str, required=True,
                        help="JSONL file from annotate_quality.py")
    parser.add_argument("--output", type=str, default="data/quality_classifier",
                        help="Output directory for classifiers")
    parser.add_argument("--dim-threshold", type=int, default=3,
                        help="Score >= this is 'high' for each dimension (0-5 scale)")
    parser.add_argument("--total-threshold", type=int, default=17,
                        help="Total score >= this is 'high' for global classifier (0-25 scale)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--wordNgrams", type=int, default=2)
    parser.add_argument("--dim", type=int, default=100,
                        help="Word vector dimension")
    args = parser.parse_args()

    import fasttext

    print(f"Loading annotations from {args.annotations}...")
    records = load_annotations(args.annotations)
    print(f"  Loaded {len(records):,} annotations")

    # Show dimension stats
    print(f"\n  Dimension distributions (0-5):")
    for dim in DIMENSIONS:
        counts = Counter(r.get(dim, -1) for r in records)
        if counts.get(-1, 0) == len(records):
            continue
        parts = " ".join(f"{s}:{counts.get(s, 0):,}" for s in range(6))
        print(f"    {dim:15s}: {parts}")

    total_counts = Counter(r["total"] for r in records)
    print(f"\n  Total score distribution (0-25):")
    for s in sorted(total_counts):
        bar = "█" * (total_counts[s] // max(len(records) // 200, 1))
        print(f"    {s:2d}: {total_counts[s]:>6,} {bar}")

    # Split
    import random
    random.seed(42)
    random.shuffle(records)

    val_size = int(len(records) * args.val_ratio)
    train_records = records[val_size:]
    val_records = records[:val_size]
    print(f"\n  Train: {len(train_records):,} | Val: {len(val_records):,}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train per-dimension classifiers
    print(f"\nTraining per-dimension classifiers (threshold={args.dim_threshold})...")
    results = {}
    for dim in DIMENSIONS:
        t0 = time.time()
        res = train_dimension_classifier(
            train_records, val_records, dim, args.dim_threshold, output_dir, args
        )
        if res:
            elapsed = time.time() - t0
            results[dim] = res
            print(f"  {dim:15s}: P={res['precision']:.3f} R={res['recall']:.3f} "
                  f"F1={res['f1']:.3f} (n={res['n_train']:,}, {elapsed:.1f}s)")

    # Train global classifier on total score
    print(f"\nTraining global classifier (total >= {args.total_threshold})...")
    t0 = time.time()

    train_path = output_dir / "train_global.txt"
    val_path = output_dir / "val_global.txt"

    for recs, path in [(train_records, train_path), (val_records, val_path)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                label = "high" if rec["total"] >= args.total_threshold else "low"
                f.write(f"__label__{label} {prepare_text(rec['text'])}\n")

    global_model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epochs,
        lr=args.lr,
        wordNgrams=args.wordNgrams,
        dim=args.dim,
        loss="softmax",
        thread=os.cpu_count() or 4,
        verbose=0,
    )

    n, precision, recall = global_model.test(str(val_path))
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    elapsed = time.time() - t0
    print(f"  global:          P={precision:.3f} R={recall:.3f} F1={f1:.3f} ({elapsed:.1f}s)")

    global_model.save_model(str(output_dir / "classifier_global.bin"))
    train_path.unlink(missing_ok=True)
    val_path.unlink(missing_ok=True)

    # Save config
    config = {
        "dimensions": list(results.keys()),
        "dim_threshold": args.dim_threshold,
        "total_threshold": args.total_threshold,
        "results": {dim: {k: float(v) for k, v in res.items()} for dim, res in results.items()},
        "global_results": {"precision": float(precision), "recall": float(recall), "f1": float(f1)},
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    total_size = sum(
        (output_dir / f"classifier_{dim}.bin").stat().st_size
        for dim in results
    ) + (output_dir / "classifier_global.bin").stat().st_size

    print(f"\nSaved to: {output_dir}")
    print(f"  Classifiers: {len(results)} dimensions + 1 global")
    print(f"  Total size: {total_size / 1e6:.1f} MB")
    print(f"  Config: {config_path}")

    print(f"\nUsage for filtering:")
    print(f"  python scripts/filter_dataset.py --classifier {output_dir} --mode global")
    print(f"  python scripts/filter_dataset.py --classifier {output_dir} --mode dimensions")


if __name__ == "__main__":
    main()
