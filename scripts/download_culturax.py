"""
Download French text from CulturaX via HuggingFace streaming.

Streams the dataset so you never need to download the whole thing.
Stops after reaching the target token count (estimated from char count).

Usage:
    # Download ~1B tokens of French text
    python scripts/download_culturax.py --output data/corpus_fr.txt --target_tokens 1_000_000_000

    # Smaller sample for testing
    python scripts/download_culturax.py --output data/corpus_fr.txt --target_tokens 10_000_000

    # Also extract a validation set
    python scripts/download_culturax.py --output data/corpus_fr.txt --val_output data/val_fr.txt --target_tokens 1_000_000_000 --val_ratio 0.005

Prerequisites:
    pip install datasets
    huggingface-cli login  (CulturaX requires accepting terms on HuggingFace)
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download French text from CulturaX")
    parser.add_argument("--output", type=str, default="data/corpus_fr.txt", help="Output text file")
    parser.add_argument("--val_output", type=str, default=None, help="Validation set output (optional)")
    parser.add_argument("--target_tokens", type=int, default=1_000_000_000, help="Target token count (approx)")
    parser.add_argument("--val_ratio", type=float, default=0.005, help="Fraction of data for validation")
    parser.add_argument("--chars_per_token", type=float, default=4.5, help="Estimated chars per token for French")
    parser.add_argument("--min_length", type=int, default=200, help="Min chars per document (filter short junk)")
    args = parser.parse_args()

    from datasets import load_dataset

    target_chars = int(args.target_tokens * args.chars_per_token)
    val_chars = int(target_chars * args.val_ratio) if args.val_output else 0
    train_chars = target_chars - val_chars

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.val_output:
        Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Target: ~{args.target_tokens:,} tokens ({target_chars:,} chars)")
    print(f"  Train: ~{train_chars:,} chars")
    if val_chars:
        print(f"  Val:   ~{val_chars:,} chars")
    print(f"Min doc length: {args.min_length} chars")
    print(f"Streaming CulturaX (fr)...\n")

    ds = load_dataset("uonlp/CulturaX", "fr", split="train", streaming=True)

    total_chars = 0
    n_docs = 0
    n_skipped = 0
    t0 = time.time()
    phase = "train"  # start writing train, switch to val

    f_train = open(args.output, 'w', encoding='utf-8')
    f_val = open(args.val_output, 'w', encoding='utf-8') if args.val_output else None

    try:
        for example in ds:
            text = example['text']

            # Filter short documents
            if len(text) < args.min_length:
                n_skipped += 1
                continue

            # Clean: ensure single newline at end
            text = text.strip() + "\n\n"
            n_chars = len(text)

            # Decide where to write
            if phase == "train":
                f_train.write(text)
                if total_chars >= train_chars and f_val is not None:
                    phase = "val"
                    print(f"\n  Switching to validation set...")
            else:
                f_val.write(text)

            total_chars += n_chars
            n_docs += 1

            # Progress
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = int(total_chars / args.chars_per_token)
                pct = min(100, total_chars / target_chars * 100)
                speed = total_chars / elapsed / 1_000_000
                print(
                    f"  {n_docs:>10,} docs | "
                    f"~{est_tokens:>13,} tokens | "
                    f"{pct:5.1f}% | "
                    f"{speed:.1f} Mchars/s | "
                    f"skipped {n_skipped:,}"
                )

            # Stop when target reached
            if total_chars >= target_chars:
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        f_train.close()
        if f_val:
            f_val.close()

    elapsed = time.time() - t0
    est_tokens = int(total_chars / args.chars_per_token)
    train_size = Path(args.output).stat().st_size / (1024 ** 3)

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Documents: {n_docs:,} (skipped {n_skipped:,})")
    print(f"  Total chars: {total_chars:,}")
    print(f"  Estimated tokens: ~{est_tokens:,}")
    print(f"  Train file: {args.output} ({train_size:.2f} GB)")
    if args.val_output and Path(args.val_output).exists():
        val_size = Path(args.val_output).stat().st_size / (1024 ** 3)
        print(f"  Val file:   {args.val_output} ({val_size:.2f} GB)")


if __name__ == "__main__":
    main()
