"""Filter FineWeb-2 FR using the trained quality classifier.

Pipeline step 3/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs
    2. train_quality_classifier.py — train fasttext on annotations
    3. filter_dataset.py     — apply classifier to full corpus (this script)

Usage:
    # Filter and save locally as JSONL
    python scripts/filter_dataset.py \
        --classifier data/quality_classifier \
        --output data/vesper_edu_fr.jsonl \
        --threshold 0.65

    # Filter and push directly to HuggingFace
    python scripts/filter_dataset.py \
        --classifier data/quality_classifier \
        --output data/vesper_edu_fr.jsonl \
        --push-to-hub AkiraXan/Vesper-Edu-FR \
        --threshold 0.65
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Filter FineWeb-2 FR with quality classifier")
    parser.add_argument("--classifier", type=str, required=True,
                        help="Path to classifier directory (from train_quality_classifier.py)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for filtered dataset")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Minimum confidence for 'high' label to keep a doc")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max docs to process from source (None = all)")
    parser.add_argument("--hf-source", type=str,
                        default="HuggingFaceFW/fineweb-2")
    parser.add_argument("--hf-name", type=str, default="fra_Latn")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF repo to push filtered dataset (e.g. AkiraXan/Vesper-Edu-FR)")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="Write/push in chunks of this size")
    args = parser.parse_args()

    import fasttext
    from datasets import load_dataset

    # Load classifier
    classifier_dir = Path(args.classifier)
    config_path = classifier_dir / "config.json"
    model_path = classifier_dir / "quality_classifier.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Classifier not found: {model_path}")

    with open(config_path) as f:
        config = json.load(f)

    model = fasttext.load_model(str(model_path))
    mode = config["mode"]

    print(f"Loaded classifier: {model_path}")
    print(f"  Mode: {mode}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Source: {args.hf_source} ({args.hf_name})")

    # Stream source dataset
    ds = load_dataset(args.hf_source, name=args.hf_name,
                      split=args.hf_split, streaming=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(output_path, "w", encoding="utf-8")

    n_total = 0
    n_kept = 0
    n_rejected = 0
    t0 = time.time()
    score_dist = [0] * 6
    chunk_records = []

    def classify_text(text: str):
        """Classify a single text. Returns (label, confidence, predicted_score)."""
        clean = " ".join(text.split())
        words = clean.split()
        if len(words) > 500:
            clean = " ".join(words[:500])

        labels, probs = model.predict(clean, k=-1)  # all labels + probs

        if mode == "binary":
            # Find "high" label probability
            for label, prob in zip(labels, probs):
                if label == "__label__high":
                    return "high", float(prob)
            return "low", 0.0
        else:
            # Multiclass: return top label and confidence
            top_label = labels[0].replace("__label__", "")
            top_prob = float(probs[0])
            return top_label, top_prob

    print(f"\nFiltering...")

    try:
        for example in ds:
            text = example.get(args.text_field) or ""
            if not text or len(text) < 100:
                n_total += 1
                n_rejected += 1
                continue

            label, confidence = classify_text(text)

            if mode == "binary":
                keep = (label == "high" and confidence >= args.threshold)
            else:
                score = int(label) if label.isdigit() else 0
                keep = (score >= config.get("min_score", 3) and
                        confidence >= args.threshold)

            n_total += 1

            if keep:
                record = {
                    "text": text,
                    "quality_score": confidence,
                    "url": example.get("url", ""),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_kept += 1
                chunk_records.append(record)
            else:
                n_rejected += 1

            if n_total % 50_000 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                keep_pct = n_kept / max(n_total, 1) * 100
                print(f"  {n_total:,} processed | "
                      f"{n_kept:,} kept ({keep_pct:.1f}%) | "
                      f"{rate:.0f} docs/s | "
                      f"{elapsed/3600:.1f}h elapsed")

            # Push chunks to HF
            if args.push_to_hub and len(chunk_records) >= args.chunk_size:
                _push_chunk(chunk_records, args.push_to_hub, n_kept)
                chunk_records = []

            if args.max_docs and n_total >= args.max_docs:
                break

    except KeyboardInterrupt:
        print(f"\nInterrupted at {n_total:,} docs")
    finally:
        f_out.close()

    # Final push
    if args.push_to_hub and chunk_records:
        _push_chunk(chunk_records, args.push_to_hub, n_kept)

    elapsed = time.time() - t0
    keep_pct = n_kept / max(n_total, 1) * 100

    print(f"\n{'='*60}")
    print(f"Filtering complete")
    print(f"  Total processed: {n_total:,}")
    print(f"  Kept:            {n_kept:,} ({keep_pct:.1f}%)")
    print(f"  Rejected:        {n_rejected:,}")
    print(f"  Time:            {elapsed/3600:.1f}h")
    print(f"  Rate:            {n_total/elapsed:.0f} docs/s")
    print(f"  Output:          {output_path}")
    if args.push_to_hub:
        print(f"  Hub:             https://huggingface.co/datasets/{args.push_to_hub}")


def _push_chunk(records: list, repo_id: str, total_kept: int):
    """Push a chunk of records to HuggingFace as a dataset."""
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

        ds = Dataset.from_list(records)
        ds.push_to_hub(
            repo_id,
            split="train",
            commit_message=f"Add chunk ({total_kept:,} docs total)",
        )
        print(f"  Pushed {len(records):,} docs to {repo_id}")
    except Exception as e:
        print(f"  Push failed: {e} — data saved locally")


if __name__ == "__main__":
    main()
