"""Evaluate an existing tokenizer's chars/token on a held-out FR corpus.

No retraining — just streams N docs and measures real compression.

Usage:
    python scripts/eval_tokenizer.py \
        --tokenizer data/velvet_tok_64k \
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \
        --n-docs 5000
"""

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Path to saved tokenizer dir")
    ap.add_argument("--hf-source", default=None,
                    help="HF spec '<repo>:<name>:<split>'")
    ap.add_argument("--input", default=None, help="Local text file")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--n-docs", type=int, default=5000)
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    print(f"Loaded tokenizer: {args.tokenizer} (vocab={tok.vocab_size:,})")

    if args.hf_source:
        from datasets import load_dataset
        parts = args.hf_source.split(":")
        if len(parts) == 2:
            repo, split = parts
            name = None
        else:
            repo, name, split = parts
            name = name or None
        ds = load_dataset(repo, name=name, split=split, streaming=True)
        def gen():
            for ex in ds:
                t = ex.get(args.text_field) or ""
                if t:
                    yield t
    elif args.input:
        def gen():
            with open(args.input, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line
    else:
        raise SystemExit("Provide --hf-source or --input.")

    total_chars = 0
    total_tokens = 0
    n = 0
    for doc in gen():
        ids = tok.encode(doc, add_special_tokens=False)
        total_chars += len(doc)
        total_tokens += len(ids)
        n += 1
        if n >= args.n_docs:
            break
        if n % 500 == 0:
            print(f"  ...{n:,} docs, running ratio = "
                  f"{total_chars / max(1, total_tokens):.3f}")

    print()
    print(f"Docs:           {n:,}")
    print(f"Total chars:    {total_chars:,}")
    print(f"Total tokens:   {total_tokens:,}")
    print(f"Chars/token:    {total_chars / max(1, total_tokens):.3f}")
    print()
    print("Reference (FR):")
    print("  >= 4.0   OK")
    print("  >= 4.5   excellent")
    print("  >= 5.0   exceptional (matches hand-tuned FR-only tokenizers)")


if __name__ == "__main__":
    main()
