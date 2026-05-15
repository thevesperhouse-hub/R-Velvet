"""Push only the README to an existing HF dataset repo.

Usage:
    python scripts/push_readme_hf.py --repo AkiraXan/Vesper-FR \
        --parquet-dir data/vesper_edu_fr_parquet
"""

import argparse
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Import README template from push_dataset_hf
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from push_dataset_hf import README_TEMPLATE


def main():
    parser = argparse.ArgumentParser(description="Push README to HF dataset repo")
    parser.add_argument("--repo", required=True, help="HF repo id")
    parser.add_argument("--parquet-dir", required=True,
                        help="Parquet dir (to count rows for README)")
    args = parser.parse_args()

    from huggingface_hub import HfApi
    import pyarrow.parquet as pq
    from pathlib import Path

    parquet_dir = Path(args.parquet_dir)
    shards = sorted(parquet_dir.glob("*.parquet"))
    n_docs = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)

    readme = README_TEMPLATE.format(
        n_docs=f"**~{n_docs/1e6:.1f}M**",
        repo_id=args.repo,
    )

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
    tmp.write(readme)
    tmp.close()

    api = HfApi()
    api.upload_file(
        path_or_fileobj=tmp.name,
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    os.unlink(tmp.name)
    print(f"README pushed to {args.repo}")


if __name__ == "__main__":
    main()
