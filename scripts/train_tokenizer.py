"""
Train a BPE tokenizer on French text corpus.

Produces a HuggingFace-compatible tokenizer that can be loaded with
AutoTokenizer.from_pretrained("data/tokenizer_fr").

Usage:
    # Train on the full corpus
    python scripts/train_tokenizer.py --input data/corpus_fr.txt --output data/tokenizer_fr

    # Quick test on a small sample
    python scripts/train_tokenizer.py --input data/corpus_fr.txt --output data/tokenizer_fr --max_lines 1000000

    # Custom vocab size
    python scripts/train_tokenizer.py --input data/corpus_fr.txt --output data/tokenizer_fr --vocab_size 32000
"""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on French text")
    parser.add_argument("--input", type=str, required=True, help="Input text file (corpus_fr.txt)")
    parser.add_argument("--output", type=str, default="data/tokenizer_fr", help="Output tokenizer directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--max_lines", type=int, default=None, help="Max lines to train on (None = all)")
    parser.add_argument("--min_frequency", type=int, default=2, help="Min token frequency to keep")
    args = parser.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    print(f"Training BPE tokenizer")
    print(f"  Input:      {args.input}")
    print(f"  Vocab size: {args.vocab_size:,}")
    print(f"  Max lines:  {args.max_lines or 'all'}")

    # --- Build tokenizer ---
    tokenizer = Tokenizer(models.BPE())

    # Normalizer: unicode NFC + lowercase accents preserved
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
    ])

    # Pre-tokenizer: split on whitespace + punctuation (like GPT-2 style)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # --- Trainer ---
    special_tokens = ["<|endoftext|>", "<|padding|>", "<|unknown|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # --- Prepare input ---
    # If max_lines set, create a temporary subset
    input_files = [args.input]

    if args.max_lines:
        print(f"\n  Extracting first {args.max_lines:,} lines...")
        subset_path = args.input + ".subset"
        with open(args.input, 'r', encoding='utf-8') as f_in, \
             open(subset_path, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= args.max_lines:
                    break
                f_out.write(line)
        input_files = [subset_path]
        print(f"  Subset saved: {subset_path}")

    # --- Train ---
    print(f"\nTraining...")
    t0 = time.time()
    tokenizer.train(input_files, trainer)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")

    # --- Post-processing: add end-of-text token ---
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    # --- Save as HuggingFace-compatible ---
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wrap in PreTrainedTokenizerFast for full HF compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        unk_token="<|unknown|>",
    )
    hf_tokenizer.save_pretrained(str(output_dir))

    # --- Verify ---
    print(f"\nSaved to: {output_dir}")
    print(f"  Files: {[f.name for f in output_dir.iterdir()]}")

    # Quick test
    test_sentences = [
        "Bonjour, comment allez-vous aujourd'hui ?",
        "Le développement de l'intelligence artificielle progresse rapidement.",
        "R-Velvet compresse 1M de tokens en 500 concepts.",
    ]
    print(f"\nTest encoding:")
    total_tokens = 0
    total_chars = 0
    for sent in test_sentences:
        ids = hf_tokenizer.encode(sent)
        decoded = hf_tokenizer.decode(ids)
        tokens = hf_tokenizer.convert_ids_to_tokens(ids)
        print(f"  \"{sent}\"")
        print(f"    → {len(ids)} tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"    → decoded: \"{decoded}\"")
        total_tokens += len(ids)
        total_chars += len(sent)

    ratio = total_chars / total_tokens
    print(f"\n  Avg chars/token: {ratio:.2f} (GPT-2 on French ≈ 3.0, good French BPE ≈ 4.5+)")

    if args.vocab_size > 65535:
        print(f"\n  WARNING: vocab_size {args.vocab_size} > 65535. tokenize_data.py uses uint16!")
        print(f"  Either reduce vocab_size or switch to uint32.")

    # Cleanup subset if created
    if args.max_lines:
        Path(args.input + ".subset").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
