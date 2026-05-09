"""Train a fasttext classifier on LLM-annotated quality scores.

Pipeline step 2/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs
    2. train_quality_classifier.py — train fasttext on annotations (this script)
    3. filter_dataset.py     — apply classifier to full corpus

Usage:
    python scripts/train_quality_classifier.py \
        --annotations data/annotations_fr.jsonl \
        --output data/quality_classifier \
        --min-score 3
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


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
                if "text" in rec and "score" in rec:
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def text_to_fasttext_line(text: str, label: str) -> str:
    """Convert text + label to fasttext format."""
    # fasttext format: __label__X text on one line
    # Clean text: replace newlines, collapse whitespace
    clean = " ".join(text.split())
    # Truncate to ~500 words for training efficiency
    words = clean.split()
    if len(words) > 500:
        clean = " ".join(words[:500])
    return f"__label__{label} {clean}"


def main():
    parser = argparse.ArgumentParser(
        description="Train fasttext quality classifier on LLM annotations")
    parser.add_argument("--annotations", type=str, required=True,
                        help="JSONL file from annotate_quality.py")
    parser.add_argument("--output", type=str, default="data/quality_classifier",
                        help="Output directory for the classifier")
    parser.add_argument("--min-score", type=int, default=3,
                        help="Minimum score to consider 'high quality' (binary split)")
    parser.add_argument("--mode", type=str, default="binary",
                        choices=["binary", "multiclass"],
                        help="binary: high/low quality. multiclass: predict exact score 0-5")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate")
    parser.add_argument("--wordNgrams", type=int, default=2,
                        help="Max word n-gram length")
    parser.add_argument("--dim", type=int, default=100,
                        help="Word vector dimension")
    args = parser.parse_args()

    import fasttext

    print(f"Loading annotations from {args.annotations}...")
    records = load_annotations(args.annotations)
    print(f"  Loaded {len(records):,} annotations")

    # Show score distribution
    from collections import Counter
    score_counts = Counter(r["score"] for r in records)
    print(f"  Score distribution:")
    for s in sorted(score_counts):
        print(f"    {s}: {score_counts[s]:,}")

    # Prepare fasttext training data
    import random
    random.seed(42)
    random.shuffle(records)

    val_size = int(len(records) * args.val_ratio)
    train_records = records[val_size:]
    val_records = records[:val_size]

    print(f"\n  Train: {len(train_records):,} | Val: {len(val_records):,}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write fasttext format files
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"

    def write_ft_file(records, path):
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                score = rec["score"]
                if args.mode == "binary":
                    label = "high" if score >= args.min_score else "low"
                else:
                    label = str(score)
                f.write(text_to_fasttext_line(rec["text"], label) + "\n")

    write_ft_file(train_records, train_path)
    write_ft_file(val_records, val_path)

    print(f"\nTraining fasttext classifier...")
    print(f"  Mode: {args.mode} (threshold={args.min_score})")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Word n-grams: {args.wordNgrams}")
    print(f"  Dimensions: {args.dim}")

    t0 = time.time()
    model = fasttext.train_supervised(
        input=str(train_path),
        epoch=args.epochs,
        lr=args.lr,
        wordNgrams=args.wordNgrams,
        dim=args.dim,
        loss="softmax",
        thread=os.cpu_count() or 4,
        verbose=2,
    )
    elapsed = time.time() - t0
    print(f"  Trained in {elapsed:.1f}s")

    # Evaluate on validation set
    print(f"\nValidation results:")
    n_val, precision, recall = model.test(str(val_path))
    print(f"  Samples: {n_val:,}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {2 * precision * recall / max(precision + recall, 1e-8):.4f}")

    # Per-label metrics
    if args.mode == "binary":
        labels = ["high", "low"]
    else:
        labels = [str(i) for i in range(6)]

    print(f"\n  Per-label:")
    for label in labels:
        n, p, r = model.test(str(val_path), k=1, threshold=0.0)
        # Count per-label manually
        correct = 0
        total_pred = 0
        total_true = 0
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                true_label = line.split()[0].replace("__label__", "")
                text = " ".join(line.split()[1:])
                pred = model.predict(text)[0][0].replace("__label__", "")
                if true_label == label:
                    total_true += 1
                if pred == label:
                    total_pred += 1
                if pred == label and true_label == label:
                    correct += 1
        p = correct / max(total_pred, 1)
        r = correct / max(total_true, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        print(f"    {label:>5}: P={p:.3f} R={r:.3f} F1={f1:.3f} (n={total_true:,})")

    # Save model
    model_path = output_dir / "quality_classifier.bin"
    model.save_model(str(model_path))
    print(f"\nModel saved: {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1e6:.1f} MB")

    # Save config for filter_dataset.py
    config = {
        "mode": args.mode,
        "min_score": args.min_score,
        "model_path": str(model_path),
        "precision": float(precision),
        "recall": float(recall),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
