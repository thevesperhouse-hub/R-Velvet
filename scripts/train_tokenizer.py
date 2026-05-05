"""
Train a BPE tokenizer for R-Velvet (FR-tuned by default).

Defaults to 64k vocab — sweet spot for 1.3B-2.5B FR models. Adaptive dtype
in tokenize_data.py handles >65k automatically (uint32).

Pre-tokenizer is byte-level + a Split layer that breaks French contractions
(l', d', qu', etc.) and digit runs into individual digits. This gives the BPE
merger room to learn meaningful sub-words on FR text rather than fusing
contractions + numbers into rare merges.

Sources:
    --input <file>                       single local text file
    --hf-source <repo>:<name>:<split>    streaming HF dataset (e.g.
                                          HuggingFaceFW/fineweb-2:fra_Latn:train)

Usage:
    # Local file, default 64k FR
    python scripts/train_tokenizer.py --input data/corpus_fr.txt --output data/velvet_tok_64k

    # Stream from FineWeb-2 FR (no full download needed)
    python scripts/train_tokenizer.py \
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \
        --max-examples 5_000_000 \
        --output data/velvet_tok_64k

    # Custom vocab size
    python scripts/train_tokenizer.py --input corpus.txt --output data/tok_100k --vocab-size 100000
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional

# ByteLevel BPE produces non-ASCII display tokens (Ġ U+0120 etc.).
# Reconfigure stdout to UTF-8 on Windows where cp1252 is the default.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def _iter_local_file(path: str, max_examples: Optional[int]) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line
            n += 1
            if max_examples and n >= max_examples:
                return


def _iter_hf_stream(spec: str, max_examples: Optional[int],
                    text_field: str = "text") -> Iterator[str]:
    """spec format: '<repo>:<name>:<split>' or '<repo>::<split>' if no config."""
    parts = spec.split(":")
    if len(parts) == 2:
        repo, split = parts
        name = None
    elif len(parts) == 3:
        repo, name, split = parts
        if name == "":
            name = None
    else:
        raise ValueError(
            f"Bad --hf-source spec {spec!r}. "
            "Expected '<repo>:<name>:<split>' or '<repo>::<split>'."
        )

    from datasets import load_dataset
    ds = load_dataset(repo, name=name, split=split, streaming=True)
    print(f"  HF streaming: {repo} (config={name}, split={split})")
    n = 0
    for example in ds:
        txt = example.get(text_field) or ""
        if not txt:
            continue
        yield txt + "\n"
        n += 1
        if max_examples and n >= max_examples:
            return


def _build_corpus_iterator(args) -> Iterable[str]:
    if args.input:
        return _iter_local_file(args.input, args.max_examples)
    if args.hf_source:
        return _iter_hf_stream(args.hf_source, args.max_examples,
                               text_field=args.text_field)
    raise SystemExit("Provide either --input or --hf-source.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer (FR-tuned) for R-Velvet")
    parser.add_argument("--input", type=str, default=None,
                        help="Local text file (one document or one line per doc)")
    parser.add_argument("--hf-source", type=str, default=None,
                        help="HF streaming spec '<repo>:<name>:<split>'")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Column name for HF datasets (default 'text')")
    parser.add_argument("--output", type=str, default="data/velvet_tok_64k",
                        help="Output tokenizer directory")
    parser.add_argument("--vocab-size", type=int, default=64000,
                        help="Vocabulary size. 64k is the FR sweet spot. "
                             ">65535 stays valid (adaptive dtype handles it).")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap on lines/examples to train on (None = all)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Min token frequency to keep")
    parser.add_argument("--no-fr-pretokenizer", action="store_true",
                        help="Disable the FR-specific Split layer (contractions, digits)")
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        parser.error("Provide either --input or --hf-source.")

    # Local imports — keep startup snappy when --help is the goal.
    from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, normalizers, decoders
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    print("Training BPE tokenizer")
    src = args.input or args.hf_source
    print(f"  Source:       {src}")
    print(f"  Vocab size:   {args.vocab_size:,}")
    print(f"  Max examples: {args.max_examples or 'all'}")
    print(f"  FR pre-tok:   {'off' if args.no_fr_pretokenizer else 'on'}")

    tokenizer = Tokenizer(models.BPE())

    # NFC keeps composed forms (œ, é) — important for FR vocabulary efficiency.
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
    ])

    # FR-aware pre-tokenizer (single regex Split + byte-only ByteLevel).
    #
    # The previous design (separate Split layers for apostrophes and digits,
    # followed by default ByteLevel) was broken: ByteLevel's internal GPT-2
    # regex re-split `d'` -> [d, '] and `aujourd'` -> [aujourd, '], which
    # undid the apostrophe isolation and starved BPE of the contractions.
    # Result was ~3.0 chars/token (GPT-2 level) instead of the 4.5+ target.
    #
    # New design: one comprehensive regex inspired by the GPT-2 / LLaMA
    # pretokenize patterns, then ByteLevel with use_regex=False so it only
    # byte-encodes without re-splitting.
    #
    # The regex captures, in priority order:
    #   ` ?\p{L}+'?`       optional leading space + letters + optional trailing apostrophe
    #                      -> keeps `aujourd'`, `l'`, `d'`, `Qu'` as single units
    #                      -> attaches the space to the next word (GPT-2 convention)
    #   ` ?\p{N}{1,3}`     1-3 digit groups (LLaMA-style); avoids splitting numbers
    #                      character-by-character which wastes ~50% of tokens on numbers.
    #   ` ?[^\s\p{L}\p{N}]+`  punctuation runs, optionally with leading space
    #   `\s+`              remaining whitespace runs (indents, double spaces)
    if not args.no_fr_pretokenizer:
        FR_PRETOKENIZE_REGEX = (
            r" ?\p{L}+'?"
            r"| ?\p{N}{1,3}"
            r"| ?[^\s\p{L}\p{N}]+"
            r"|\s+"
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(
                pattern=Regex(FR_PRETOKENIZE_REGEX),
                behavior="isolated",
            ),
            # use_regex=False disables ByteLevel's GPT-2 splitter so it doesn't
            # undo the Split above. add_prefix_space=False because our regex
            # already attaches leading spaces to the right token.
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    else:
        # Plain GPT-2-style ByteLevel BPE (no FR-specific handling).
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<|endoftext|>", "<|padding|>", "<|unknown|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print("\nTraining...")
    t0 = time.time()
    tokenizer.train_from_iterator(_build_corpus_iterator(args), trainer)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")

    eos_id = tokenizer.token_to_id("<|endoftext|>")
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        unk_token="<|unknown|>",
    )
    hf_tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved to: {output_dir}")
    print(f"  Files: {[f.name for f in output_dir.iterdir()]}")

    # Validation: encoding sanity + chars/token ratio on FR sentences.
    test_sentences = [
        "Bonjour, comment allez-vous aujourd'hui ?",
        "Le développement de l'intelligence artificielle progresse rapidement.",
        "R-Velvet compresse 1M de tokens en 500 concepts.",
        "Les œufs et le bœuf coûtent 12,50 € au marché.",
        "Qu'est-ce que tu fais demain à 14h30 ?",
    ]
    print("\nTest encoding:")
    total_tokens = 0
    total_chars = 0
    for sent in test_sentences:
        ids = hf_tokenizer.encode(sent)
        tokens = hf_tokenizer.convert_ids_to_tokens(ids)
        print(f"  \"{sent}\"")
        print(f"    -> {len(ids)} tokens: {tokens[:12]}{'...' if len(tokens) > 12 else ''}")
        total_tokens += len(ids)
        total_chars += len(sent)

    ratio = total_chars / max(1, total_tokens)
    print(f"\n  Avg chars/token: {ratio:.2f} "
          f"(target: >= 4.0 for FR; GPT-2 ~3.0, well-tuned FR BPE ~4.5+)")


if __name__ == "__main__":
    main()
