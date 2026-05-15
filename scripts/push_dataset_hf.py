"""Push Vesper-Edu-FR dataset to HuggingFace Hub (private).

Usage:
    python scripts/push_dataset_hf.py \
        --parquet-dir data/vesper_edu_fr_parquet \
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

# Vesper-FR

**Quality-filtered French pretraining corpus derived from FineWeb-2.**

{n_docs} documents selected for coherence, linguistic quality, depth, and factual reliability from ~186M processed documents.

## Overview

Vesper-FR is a curated subset of [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (French partition, `fra_Latn`). Documents are filtered using multi-dimensional quality classifiers trained via LLM-assisted annotation, then deduplicated.

This is not an "educational" dataset — it is a general-purpose, quality-filtered French web corpus suitable for LLM pretraining.

## Filtering pipeline

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
One binary fasttext classifier was trained per dimension on the LLM annotations (score >= 3 = "high", < 3 = "low").

| Classifier | F1 | Precision | Recall | Train samples |
|---|---|---|---|---|
| coherence | 0.831 | 0.831 | 0.831 | 588k |
| pedagogy | 0.870 | 0.870 | 0.870 | 588k |
| linguistic | 0.924 | 0.924 | 0.924 | 581k |
| depth | 0.831 | 0.831 | 0.831 | 588k |
| factuality | 0.866 | 0.866 | 0.866 | 588k |
| code_quality | 1.000 | 1.000 | 1.000 | 7k |
| **global** | **0.848** | **0.848** | **0.848** | 588k |

### Step 3 — Corpus filtering
The classifiers were applied to ~186M documents from FineWeb-2 FR (streamed). A document is kept only if **all relevant dimensions** pass with confidence >= 0.65.

- Text documents must pass: coherence, pedagogy, linguistic, depth, factuality
- Code documents must pass: coherence, code_quality, pedagogy, depth, factuality

Retention rate: ~10-11%.

### Step 4 — Deduplication
The filtered output was deduplicated using three methods:
- **Exact**: MD5 hash of full text
- **URL**: same URL = same content
- **Fuzzy**: MD5 hash of first 1000 normalized characters (catches near-duplicates with different headers/footers)

## Fields

| Field | Type | Description |
|---|---|---|
| `text` | string | Document text |
| `url` | string | Source URL |
| `score_coherence` | float | Fasttext confidence for "high" coherence |
| `score_pedagogy` | float | Fasttext confidence for "high" pedagogy |
| `score_linguistic` | float | Fasttext confidence for "high" linguistic quality |
| `score_depth` | float | Fasttext confidence for "high" depth |
| `score_factuality` | float | Fasttext confidence for "high" factuality |

> The `score_*` fields are fasttext classifier confidences (0-1), not the original LLM scores (0-5). They are metadata and can be ignored for training.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0]["text"][:500])
```

## Source & License

- **Base dataset**: [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`fra_Latn`)
- **Annotation model**: Qwen2.5-32B-Instruct-AWQ
- **Built with**: [R-Velvet](https://github.com/thevesperhouse-hub/R-Velvet)
- **License**: [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/) (inherited from FineWeb-2)
"""


def main():
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace Hub")
    parser.add_argument("--parquet-dir", required=True,
                        help="Directory containing parquet shards")
    parser.add_argument("--repo", required=True,
                        help="HF repo (e.g. AkiraXan/Vesper-Edu-FR)")
    parser.add_argument("--public", action="store_true",
                        help="Make repo public (default: private)")
    parser.add_argument("--delete-old", action="store_true",
                        help="Delete old single parquet file from repo first")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    from pathlib import Path
    import shutil

    api = HfApi()

    # Count rows across all shards
    parquet_dir = Path(args.parquet_dir)
    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        print(f"No parquet files found in {parquet_dir}")
        sys.exit(1)

    n_docs = 0
    total_size = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        n_docs += pf.metadata.num_rows
        total_size += shard.stat().st_size

    print(f"Dataset: {parquet_dir}")
    print(f"  Shards: {len(shards)}")
    print(f"  Total rows: {n_docs:,}")
    print(f"  Total size: {total_size / 1e9:.1f} GB")
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

    # Delete old single parquet if requested
    if args.delete_old:
        try:
            api.delete_file(
                path_in_repo="data/train-00000-of-00001.parquet",
                repo_id=args.repo,
                repo_type="dataset",
                commit_message="Remove old single parquet file",
            )
            print("Deleted old single parquet file")
        except Exception:
            print("No old single parquet to delete")

    # Build staging folder
    staging = Path(tempfile.mkdtemp(prefix="vesper_hf_"))
    data_dir = staging / "data"
    data_dir.mkdir()

    # Generate README
    readme_content = README_TEMPLATE.format(
        n_docs=f"**~{n_docs/1e6:.1f}M**",
        repo_id=args.repo,
    )
    (staging / "README.md").write_text(readme_content, encoding="utf-8")

    # Symlink shards into data/
    for shard in shards:
        dest = data_dir / shard.name
        src = shard.resolve()
        try:
            os.symlink(src, dest)
        except OSError:
            print(f"Symlink failed for {shard.name}, copying...")
            shutil.copy2(src, dest)

    print(f"\nStaging folder: {staging}")
    print(f"  {len(shards)} shards in data/")
    print(f"Uploading with upload_large_folder...")

    api.upload_large_folder(
        folder_path=str(staging),
        repo_id=args.repo,
        repo_type="dataset",
    )

    # Cleanup staging
    shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone!")
    print(f"  https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
