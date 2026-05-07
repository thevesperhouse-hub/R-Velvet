"""Push a local tokenizer to HuggingFace Hub.

Usage:
    python scripts/push_tokenizer_hf.py --path data/velvet_tok_100k_unigram --repo AkiraXan/velvet-tok-100k-unigram
"""
import argparse
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=True, help="Local tokenizer folder")
parser.add_argument("--repo", required=True, help="HF repo id (user/name)")
args = parser.parse_args()

api = HfApi()
api.create_repo(args.repo, exist_ok=True)
api.upload_folder(folder_path=args.path, repo_id=args.repo, repo_type="model")
print(f"Pushed {args.path} -> https://huggingface.co/{args.repo}")
