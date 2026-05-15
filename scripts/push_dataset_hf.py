"""Push Vesper-Edu-FR dataset to HuggingFace Hub (private).

Usage:
    python scripts/push_dataset_hf.py \
        --parquet data/vesper_edu_fr.parquet \
        --repo AkiraXan/Vesper-Edu-FR
"""

import argparse
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

README_TEMPLATE = """---
language:
- fr
license: odc-by
size_categories:
- 10M<n<100M
task_categories:
- text-generation
tags:
- pretrain
- french
- education
- quality-filtered
- fineweb
dataset_info:
  features:
  - name: text
    dtype: string
  - name: url
    dtype: string
  - name: score_coherence
    dtype: float64
  - name: score_depth
    dtype: float64
  - name: score_factuality
    dtype: float64
  - name: score_linguistic
    dtype: float64
  - name: score_pedagogy
    dtype: float64
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/*.parquet"
---

# Vesper-Edu-FR

**High-quality French pretraining dataset filtered from FineWeb-2.**

{n_docs} documents selected for educational value, linguistic quality, and factual reliability.

## Overview

Vesper-Edu-FR is a curated subset of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (French partition), filtered using multi-dimensional quality classifiers. The goal is to provide a high-quality French corpus for LLM pretraining, similar to what [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) achieves for English.

## Filtering pipeline

The dataset was built in 3 steps:

### Step 1 — LLM annotation (~650k documents)
A sample of FineWeb-2 FR was scored by **Qwen2.5-32B-Instruct** on 5 independent quality dimensions (0-5 scale each):

| Dimension | Description |
|---|---|
| **coherence** | Logical structure, reasoning quality |
| **pedagogy** | Educational value, clarity of explanations |
| **linguistic** | Grammar, syntax, proper French |
| **depth** | Substance, examples, developed arguments |
| **factuality** | Reliable, verifiable information |

For code documents, `linguistic` is replaced by `code_quality` (clean, documented, idiomatic code).

The scoring prompt was calibrated to be strict: most web text scores 1-3, a 5 is rare and reserved for excellence.

### Step 2 — Fasttext classifier training
One binary fasttext classifier was trained per dimension on the LLM annotations (score >= 3 → "high", < 3 → "low"). Validation F1 scores:

| Classifier | F1 |
|---|---|
| coherence | ~0.88 |
| pedagogy | ~0.85 |
| linguistic | ~0.87 |
| depth | ~0.86 |
| factuality | ~0.88 |

### Step 3 — Full corpus filtering
The fasttext classifiers were applied to the full FineWeb-2 FR corpus (~186M documents processed). A document is kept only if **all relevant dimensions** pass with confidence >= 0.65.

Text documents must pass: coherence, pedagogy, linguistic, depth, factuality.
Code documents must pass: coherence, code_quality, pedagogy, depth, factuality.

### Step 4 — Deduplication
The filtered output was deduplicated using three methods:
- **Exact dedup**: MD5 hash of full text
- **URL dedup**: same URL → same content
- **Fuzzy dedup**: MD5 hash of first 1000 normalized characters

## Dataset fields

| Field | Type | Description |
|---|---|---|
| `text` | string | Document text |
| `url` | string | Source URL |
| `score_coherence` | float | Fasttext confidence for "high" coherence |
| `score_pedagogy` | float | Fasttext confidence for "high" pedagogy |
| `score_linguistic` | float | Fasttext confidence for "high" linguistic quality |
| `score_depth` | float | Fasttext confidence for "high" depth |
| `score_factuality` | float | Fasttext confidence for "high" factuality |

> **Note**: The `score_*` fields are fasttext classifier confidences (0-1), not the original LLM scores (0-5).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0]["text"][:500])
```

## Source

- **Base dataset**: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`fra_Latn` partition)
- **Annotation model**: Qwen2.5-32B-Instruct-AWQ
- **Built with**: [R-Velvet](https://github.com/thevesperhouse-hub/R-Velvet)

## License

This dataset inherits the [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) license from FineWeb-2.
"""


def main():
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("--parquet", required=True, help="Parquet file to push")
    parser.add_argument("--repo", required=True, help="HF repo (e.g. AkiraXan/Vesper-Edu-FR)")
    parser.add_argument("--public", action="store_true", help="Make repo public (default: private)")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    import pyarrow.parquet as pq

    api = HfApi()

    # Get row count
    pf = pq.ParquetFile(args.parquet)
    n_docs = pf.metadata.num_rows
    size_mb = os.path.getsize(args.parquet) / 1e6

    print(f"Dataset: {args.parquet}")
    print(f"  Rows: {n_docs:,}")
    print(f"  Size: {size_mb:,.0f} MB")
    print(f"  Repo: {args.repo}")
    print(f"  Private: {not args.public}")

    # Create repo
    api.create_repo(
        args.repo,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    print(f"Repo created/exists: {args.repo}")

    # Generate README
    readme_content = README_TEMPLATE.format(
        n_docs=f"**~{n_docs/1e6:.1f}M**",
        repo_id=args.repo,
    )

    # Write README to temp file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False,
                                      encoding="utf-8") as f:
        f.write(readme_content)
        readme_path = f.name

    try:
        print("Uploading README.md...")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
    finally:
        os.unlink(readme_path)

    # Upload parquet
    print(f"Uploading parquet ({size_mb:,.0f} MB)...")
    api.upload_file(
        path_or_fileobj=args.parquet,
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"Add dataset ({n_docs:,} rows)",
    )

    print(f"\nDone!")
    print(f"  https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
