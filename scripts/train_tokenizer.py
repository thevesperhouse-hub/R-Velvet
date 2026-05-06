"""
Train a tokenizer for R-Velvet (FR-tuned by default).

Two algorithms supported:
  --algo unigram  (default)  SentencePiece-style Unigram LM. Optimizes log-
                             likelihood globally rather than greedy bottom-up
                             merging. Used by Mistral, Gemma, T5, BLOOM. Gives
                             ~5-10% better compression than BPE on FR at the
                             same vocab size.
  --algo bpe                 Byte-level BPE with FR-aware Split layer.

Default vocab is 80k — sweet spot for FR-only models targeting 5+ chars/token
(exceptional tier). Adaptive dtype in tokenize_data.py handles >65k (uint32).

Sources (multiple supported, with optional weight for mixing):
    --input <file>                                 single local text file
    --hf-source <repo>:<name>:<split>[@<weight>]   streaming HF dataset
                                                   pass multiple times to mix

Usage:
    # Local file, default unigram 80k FR
    python scripts/train_tokenizer.py --input data/corpus_fr.txt --output data/velvet_tok_80k

    # Stream from FineWeb-2 FR (no full download needed)
    python scripts/train_tokenizer.py \
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \
        --max-examples 5_000_000 \
        --output data/velvet_tok_80k

    # Combo gagnant: FineWeb (70%) + Wikipedia FR (30%), Unigram 80k
    python scripts/train_tokenizer.py \
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train@0.7 \
        --hf-source wikimedia/wikipedia:20231101.fr:train@0.3 \
        --vocab-size 80000 \
        --max-examples 8_000_000 \
        --output data/velvet_tok_80k_unigram

    # Old-school BPE for comparison / baseline
    python scripts/train_tokenizer.py --algo bpe --input corpus.txt --output data/tok_bpe
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

# Unigram / ByteLevel can produce non-ASCII display tokens (▁ U+2581, Ġ U+0120).
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


def _parse_hf_spec(spec: str) -> Tuple[str, Optional[str], str, float]:
    """spec format:
        '<repo>:<name>:<split>'           weight defaults to 1.0
        '<repo>::<split>'                 no config
        '<repo>:<name>:<split>@<weight>'  explicit mixing weight
    """
    weight = 1.0
    if "@" in spec:
        spec, w = spec.rsplit("@", 1)
        weight = float(w)

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
            "Expected '<repo>:<name>:<split>[@<weight>]' or '<repo>::<split>[@<weight>]'."
        )
    return repo, name, split, weight


def _iter_hf_streams(specs: List[str], max_examples: Optional[int],
                     text_field: str = "text") -> Iterator[str]:
    """Stream from one or more HF datasets, interleaved by weight if multiple."""
    from datasets import load_dataset, interleave_datasets

    parsed = [_parse_hf_spec(s) for s in specs]

    datasets_list = []
    weights = []
    for repo, name, split, weight in parsed:
        ds = load_dataset(repo, name=name, split=split, streaming=True)
        print(f"  HF streaming: {repo} (config={name}, split={split}, weight={weight})")
        datasets_list.append(ds)
        weights.append(weight)

    if len(datasets_list) == 1:
        ds = datasets_list[0]
    else:
        # Normalize weights into probabilities. all_exhausted keeps sampling
        # from each stream until every one runs dry — gives true mixing rather
        # than stopping at the smallest source.
        total = sum(weights)
        probs = [w / total for w in weights]
        print(f"  Interleaving {len(datasets_list)} sources with probs={probs}")
        ds = interleave_datasets(
            datasets_list,
            probabilities=probs,
            stopping_strategy="all_exhausted",
        )

    n = 0
    for example in ds:
        txt = example.get(text_field) or ""
        if not txt:
            continue
        yield txt + "\n"
        n += 1
        if max_examples and n >= max_examples:
            return


def _build_corpus_iterator(args, max_examples: Optional[int] = None) -> Iterable[str]:
    """`max_examples` overrides args.max_examples if given (used for eval split)."""
    cap = max_examples if max_examples is not None else args.max_examples
    if args.input:
        return _iter_local_file(args.input, cap)
    if args.hf_source:
        return _iter_hf_streams(args.hf_source, cap, text_field=args.text_field)
    raise SystemExit("Provide either --input or --hf-source.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a tokenizer (FR-tuned) for R-Velvet")
    parser.add_argument("--algo", choices=["unigram", "bpe"], default="unigram",
                        help="Tokenization algorithm. unigram is recommended for FR "
                             "(matches Mistral/Gemma/T5). bpe kept for baseline.")
    parser.add_argument("--input", type=str, default=None,
                        help="Local text file (one document or one line per doc)")
    parser.add_argument("--hf-source", type=str, action="append", default=None,
                        help="HF streaming spec '<repo>:<name>:<split>[@<weight>]'. "
                             "Pass multiple times to mix sources by weight.")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Column name for HF datasets (default 'text')")
    parser.add_argument("--output", type=str, default="data/velvet_tok_80k",
                        help="Output tokenizer directory")
    parser.add_argument("--vocab-size", type=int, default=80000,
                        help="Vocabulary size. 80k targets exceptional FR "
                             "compression (>= 5.0 chars/token). Adaptive dtype "
                             "handles >65535 automatically.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Cap on lines/examples to train on (None = all)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Min token frequency to keep (BPE only)")
    parser.add_argument("--no-fr-pretokenizer", action="store_true",
                        help="(BPE only) Disable the FR-specific Split layer")
    parser.add_argument("--n-sub-iterations", type=int, default=2,
                        help="(Unigram only) EM sub-iterations. Default 2, "
                             "try 4 for slightly better vocab convergence.")
    parser.add_argument("--shrinking-factor", type=float, default=0.75,
                        help="(Unigram only) Fraction of vocab kept per "
                             "iteration. Default 0.75, try 0.7 for more "
                             "aggressive pruning.")
    parser.add_argument("--max-piece-length", type=int, default=16,
                        help="(Unigram only) Max piece length. Default 16. "
                             "Lower reduces suffix-array memory at compression "
                             "cost; do not change unless trainer hangs.")
    parser.add_argument("--eval-docs", type=int, default=2000,
                        help="After training, stream N held-out docs and report "
                             "real chars/token. Set 0 to skip. Default 2000.")
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        parser.error("Provide either --input or --hf-source.")

    # Local imports — keep startup snappy when --help is the goal.
    from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, normalizers, decoders
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    print(f"Training {args.algo.upper()} tokenizer")
    src = args.input or args.hf_source
    print(f"  Source:       {src}")
    print(f"  Vocab size:   {args.vocab_size:,}")
    print(f"  Max examples: {args.max_examples or 'all'}")

    special_tokens = ["<|endoftext|>", "<|padding|>", "<|unknown|>"]

    # NFC keeps composed forms (œ, é) — important for FR vocabulary efficiency.
    common_normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
    ])

    if args.algo == "unigram":
        # SentencePiece-style: Metaspace replaces whitespace with ▁ marker
        # and operates on raw Unicode chars (no byte-level encoding). For FR,
        # this means `é` is a single token char rather than 2 bytes — saves
        # roughly 5-10% on accent-heavy text.
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.normalizer = common_normalizer
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁",
            prepend_scheme="always",
        )
        tokenizer.decoder = decoders.Metaspace(
            replacement="▁",
            prepend_scheme="always",
        )
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            unk_token="<|unknown|>",
            show_progress=True,
            n_sub_iterations=args.n_sub_iterations,
            shrinking_factor=args.shrinking_factor,
            max_piece_length=args.max_piece_length,
        )
    else:
        # FR-aware byte-level BPE.
        #
        # The pre-tokenizer combines a comprehensive regex Split with a
        # ByteLevel(use_regex=False). Without use_regex=False the GPT-2
        # internal regex re-splits `d'`, `l'`, `aujourd'` into pieces and
        # starves BPE of contractions — that path scored ~3.0 chars/token.
        tokenizer = Tokenizer(models.BPE())
        tokenizer.normalizer = common_normalizer

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
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ])
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

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

    # EOS-excluded ratio (each test had <|endoftext|> appended).
    total_tokens_no_eos = total_tokens - len(test_sentences)
    ratio = total_chars / max(1, total_tokens_no_eos)
    print(f"\n  Avg chars/token (synthetic, EOS-excluded): {ratio:.2f}")
    print(f"    Note: these 5 stress-test sentences are biased toward "
          f"single-char tokens\n    (punctuation, English loanwords, time/currency "
          f"formats). Real FR prose\n    typically scores +0.5 to +1.0 above this.")

    # Honest evaluation: stream a held-out chunk of real text.
    if args.eval_docs > 0:
        print(f"\nEvaluating on {args.eval_docs:,} held-out docs from the source...")
        try:
            eval_iter = _build_corpus_iterator(
                args,
                max_examples=(args.max_examples or 0) + args.eval_docs,
            )
            eval_chars = 0
            eval_tokens = 0
            n_eval = 0
            for i, doc in enumerate(eval_iter):
                if args.input and i < (args.max_examples or 0):
                    continue
                ids = hf_tokenizer.encode(doc, add_special_tokens=False)
                eval_chars += len(doc)
                eval_tokens += len(ids)
                n_eval += 1
                if n_eval >= args.eval_docs:
                    break
            if n_eval > 0:
                real_ratio = eval_chars / max(1, eval_tokens)
                print(f"  Docs evaluated:  {n_eval:,}")
                print(f"  Total chars:     {eval_chars:,}")
                print(f"  Total tokens:    {eval_tokens:,}")
                print(f"  Real chars/token: {real_ratio:.3f}  "
                      f"(>= 4.0 OK, >= 4.5 excellent, >= 5.0 exceptional)")
            else:
                print("  No held-out docs available for evaluation.")
        except Exception as e:
            print(f"  Eval skipped: {e}")


if __name__ == "__main__":
    main()
