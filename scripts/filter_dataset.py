"""Filter FineWeb-2 FR using trained quality classifiers.

Pipeline step 3/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs
    2. train_quality_classifier.py — train fasttext on annotations
    3. filter_dataset.py     — apply classifiers to full corpus (this script)

Two modes:
    --mode global:     use only the global (total score) classifier
    --mode dimensions: require ALL dimension classifiers to pass

Usage:
    python scripts/filter_dataset.py \
        --classifier data/quality_classifier \
        --output data/vesper_edu_fr.jsonl \
        --mode dimensions \
        --threshold 0.65 \
        --push-to-hub AkiraXan/Vesper-Edu-FR
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


def prepare_text(text: str, max_words: int = 500) -> str:
    """Clean text for fasttext classification."""
    clean = " ".join(text.split())
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words])
    return clean


def is_code(text: str) -> bool:
    """Heuristic: detect if text is primarily code."""
    indicators = [
        "def ", "class ", "import ", "function ", "return ",
        "if (", "for (", "while (", "const ", "let ", "var ",
        "#include", "public static", "func ", "fn ",
    ]
    lines = text[:2000].split("\n")
    code_lines = sum(1 for l in lines if any(ind in l for ind in indicators))
    return code_lines > len(lines) * 0.3


def main():
    parser = argparse.ArgumentParser(
        description="Filter FineWeb-2 FR with quality classifiers")
    parser.add_argument("--classifier", type=str, required=True,
                        help="Path to classifier directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file for filtered dataset")
    parser.add_argument("--mode", type=str, default="dimensions",
                        choices=["global", "dimensions"],
                        help="global: single classifier. dimensions: all axes must pass")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Min confidence for 'high' label")
    parser.add_argument("--min-dims-pass", type=int, default=None,
                        help="In dimensions mode, min number of dimensions that must pass. "
                             "Default: all of them")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Max docs to process (None = all)")
    parser.add_argument("--hf-source", type=str,
                        default="HuggingFaceFW/fineweb-2")
    parser.add_argument("--hf-name", type=str, default="fra_Latn")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF dataset repo to push (e.g. AkiraXan/Vesper-Edu-FR)")
    parser.add_argument("--chunk-size", type=int, default=100_000,
                        help="Push in chunks of this size")
    args = parser.parse_args()

    import fasttext
    from datasets import load_dataset

    classifier_dir = Path(args.classifier)
    config_path = classifier_dir / "config.json"

    with open(config_path) as f:
        config = json.load(f)

    # Load classifiers
    # code_quality only applies to code, not to French text — load it
    # separately so it doesn't block text documents from passing.
    TEXT_DIMS = ["coherence", "pedagogy", "linguistic", "depth", "factuality"]
    CODE_DIMS = ["coherence", "code_quality", "pedagogy", "depth", "factuality"]

    models = {}
    if args.mode == "global":
        global_path = classifier_dir / "classifier_global.bin"
        models["global"] = fasttext.load_model(str(global_path))
        print(f"Loaded global classifier")
    else:
        dims = config["dimensions"]
        for dim in dims:
            model_path = classifier_dir / f"classifier_{dim}.bin"
            if model_path.exists():
                models[dim] = fasttext.load_model(str(model_path))
        print(f"Loaded {len(models)} dimension classifiers: {list(models.keys())}")

    min_dims = args.min_dims_pass or len(models)
    print(f"Mode: {args.mode}")
    print(f"Threshold: {args.threshold}")
    if args.mode == "dimensions":
        print(f"Min dimensions to pass: {min_dims}/{len(models)}")
    print(f"Source: {args.hf_source} ({args.hf_name})")

    ds = load_dataset(args.hf_source, name=args.hf_name,
                      split=args.hf_split, streaming=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(output_path, "w", encoding="utf-8")

    n_total = 0
    n_kept = 0
    n_rejected = 0
    t0 = time.time()
    chunk_records = []

    # Track per-dimension pass rates
    dim_pass_counts = {dim: 0 for dim in models}

    def classify(text: str) -> tuple:
        """Classify text, return (predictions dict, relevant_dims list)."""
        clean = prepare_text(text)
        # Pick relevant dimensions based on text type
        relevant = CODE_DIMS if is_code(text) else TEXT_DIMS
        result = {}
        for dim in relevant:
            if dim not in models:
                continue
            model = models[dim]
            labels, probs = model.predict(clean, k=-1)
            for label, prob in zip(labels, probs):
                if label == "__label__high":
                    result[dim] = ("high", float(prob))
                    break
            else:
                result[dim] = ("low", 0.0)
        return result, relevant

    print(f"\nFiltering...")

    try:
        for example in ds:
            text = example.get(args.text_field) or ""
            if not text or len(text) < 100:
                n_total += 1
                n_rejected += 1
                continue

            n_total += 1
            predictions, relevant = classify(text)

            if args.mode == "global":
                label, conf = predictions.get("global", ("low", 0.0))
                keep = (label == "high" and conf >= args.threshold)
            else:
                # Count how many relevant dimensions pass
                dims_passed = 0
                n_relevant = len([d for d in relevant if d in models])
                required = args.min_dims_pass or n_relevant
                for dim, (label, conf) in predictions.items():
                    passes = (label == "high" and conf >= args.threshold)
                    if passes:
                        dims_passed += 1
                        dim_pass_counts[dim] = dim_pass_counts.get(dim, 0) + 1
                keep = (dims_passed >= required)

            if keep:
                record = {
                    "text": text,
                    "url": example.get("url", ""),
                }
                # Add per-dimension confidences
                for dim, (label, conf) in predictions.items():
                    record[f"score_{dim}"] = round(conf, 4)

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_kept += 1
                chunk_records.append(record)
            else:
                n_rejected += 1

            if n_total % 50_000 == 0:
                elapsed = time.time() - t0
                rate = n_total / elapsed
                keep_pct = n_kept / max(n_total, 1) * 100

                dim_parts = []
                if args.mode == "dimensions":
                    for dim in models:
                        dpct = dim_pass_counts[dim] / max(n_total, 1) * 100
                        dim_parts.append(f"{dim[:4]}={dpct:.0f}%")

                print(f"  {n_total:,} processed | "
                      f"{n_kept:,} kept ({keep_pct:.1f}%) | "
                      f"{rate:.0f} docs/s | "
                      f"{' '.join(dim_parts)}")

            if args.push_to_hub and len(chunk_records) >= args.chunk_size:
                _push_chunk(chunk_records, args.push_to_hub, n_kept)
                chunk_records = []

            if args.max_docs and n_total >= args.max_docs:
                break

    except KeyboardInterrupt:
        print(f"\nInterrupted at {n_total:,} docs")
    finally:
        f_out.close()

    if args.push_to_hub and chunk_records:
        _push_chunk(chunk_records, args.push_to_hub, n_kept)

    elapsed = time.time() - t0
    keep_pct = n_kept / max(n_total, 1) * 100

    print(f"\n{'='*60}")
    print(f"Filtering complete")
    print(f"  Total processed:  {n_total:,}")
    print(f"  Kept:             {n_kept:,} ({keep_pct:.1f}%)")
    print(f"  Rejected:         {n_rejected:,}")
    print(f"  Time:             {elapsed/3600:.1f}h")
    print(f"  Rate:             {n_total/elapsed:.0f} docs/s")
    print(f"  Output:           {output_path}")

    if args.mode == "dimensions":
        print(f"\n  Per-dimension pass rates:")
        for dim in models:
            dpct = dim_pass_counts[dim] / max(n_total, 1) * 100
            bar = "█" * int(dpct / 2)
            print(f"    {dim:15s}: {dpct:5.1f}% {bar}")

    if args.push_to_hub:
        print(f"  Hub: https://huggingface.co/datasets/{args.push_to_hub}")


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
