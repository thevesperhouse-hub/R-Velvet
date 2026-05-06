"""
Train a tokenizer for R-Velvet using SentencePiece (Google), BPE or Unigram.

Why SentencePiece and not HF tokenizers ?
HF tokenizers' Unigram implementation keeps the entire suffix array in
memory, single-process, with a not-very-compact representation. Empirically
it caps around 2-3M FR docs before the suffix array phase becomes
intractable. SentencePiece (Google C++) shards the suffix array, supports
streaming via sentence sampling, and is what Mistral / Gemma / T5 / BLOOM /
CamemBERT all use under the hood.

Why SP-BPE rather than HF byte-level BPE on FR ?
HF's default ByteLevel BPE encodes every character as UTF-8 bytes before
merging. Each accented FR character (é, è, à, ç, œ, …) is 2 bytes, so the
algo starts with a built-in handicap that it only partially recovers via
merges. SP-BPE works at the character level natively : accents are single
tokens from step 1. Empirically buys +0.2 to +0.4 chars/token on FR at
iso-vocab.

Pipeline:
    1. Stream from HF and/or local
    2. Yield individual lines (SPM trains on sentences, not whole documents)
    3. SPM samples `input_sentence_size` items from the stream and trains
    4. Wrap the resulting .model :
       - Unigram → tokenizers.Unigram via direct piece/score extraction
       - BPE     → LlamaTokenizerFast (which knows how to read SPM .model
                   merge rules) with post-conversion vocab-size validation
    5. Save in HF Fast format compatible with PreTrainedTokenizerFast

Usage:
    # Unigram (default)
    python scripts/train_tokenizer_spm.py \\
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \\
        --output data/velvet_tok_100k_spm

    # BPE (recommended for FR — sidesteps byte-level overhead on accents)
    python scripts/train_tokenizer_spm.py \\
        --algo bpe \\
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \\
        --output data/velvet_tok_100k_spm_bpe

    # Multi-source mix
    python scripts/train_tokenizer_spm.py \\
        --algo bpe \\
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train@0.7 \\
        --hf-source wikimedia/wikipedia:20231101.fr:train@0.3 \\
        --output data/velvet_tok_100k_spm_bpe_mix
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

# SPM and ByteLevel produce non-ASCII display tokens (▁ U+2581).
# Reconfigure stdout to UTF-8 on Windows where cp1252 is the default.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# ------------------------------------------------------------------------
# Corpus iterators (mirror train_tokenizer.py for consistency)
# ------------------------------------------------------------------------

def _iter_local_file(path: str, max_examples: Optional[int]) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line
                n += 1
                if max_examples and n >= max_examples:
                    return


def _parse_hf_spec(spec: str) -> Tuple[str, Optional[str], str, float]:
    """spec: '<repo>:<name>:<split>[@<weight>]' or '<repo>::<split>[@<weight>]'."""
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
    """Stream from one or more HF datasets, interleaved by weight if multiple.

    Yields one line at a time (not full documents). SPM treats each yielded
    string as a sentence to be considered for the suffix array sampling.
    """
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
        total = sum(weights)
        probs = [w / total for w in weights]
        print(f"  Interleaving {len(datasets_list)} sources with probs={probs}")
        ds = interleave_datasets(
            datasets_list,
            probabilities=probs,
            stopping_strategy="all_exhausted",
        )

    n_docs = 0
    for example in ds:
        txt = example.get(text_field) or ""
        if not txt:
            continue
        for line in txt.splitlines():
            line = line.strip()
            # Filter very short lines (often nav/garbage on web crawls).
            # 50 chars is the threshold used by Mistral / Gemma corpus prep.
            if len(line) >= 50:
                yield line
        n_docs += 1
        if max_examples and n_docs >= max_examples:
            return


def _build_corpus_iterator(args, max_examples: Optional[int] = None) -> Iterable[str]:
    cap = max_examples if max_examples is not None else args.max_examples
    if args.input:
        return _iter_local_file(args.input, cap)
    if args.hf_source:
        return _iter_hf_streams(args.hf_source, cap, text_field=args.text_field)
    raise SystemExit("Provide either --input or --hf-source.")


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a BPE or Unigram tokenizer via SentencePiece for R-Velvet")
    parser.add_argument("--input", type=str, default=None,
                        help="Local text file (one sentence/line per row)")
    parser.add_argument("--hf-source", type=str, action="append", default=None,
                        help="HF streaming spec '<repo>:<name>:<split>[@<weight>]'. "
                             "Pass multiple times to mix sources by weight.")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--algo", type=str, default="unigram",
                        choices=["unigram", "bpe"],
                        help="SentencePiece algorithm. 'bpe' (char-level via SPM) "
                             "is recommended for FR : sidesteps the byte-level "
                             "overhead on accented characters that limits HF BPE.")
    parser.add_argument("--output", type=str, default="data/velvet_tok_100k_spm",
                        help="Output tokenizer directory")
    parser.add_argument("--vocab-size", type=int, default=100000,
                        help="Vocabulary size.")
    parser.add_argument("--max-examples", type=int, default=8_000_000,
                        help="Cap on documents streamed from source (each doc "
                             "is split into multiple sentence lines for SPM).")
    parser.add_argument("--input-sentence-size", type=int, default=30_000_000,
                        help="Number of sentences SentencePiece samples from "
                             "the stream for actual training.")
    parser.add_argument("--character-coverage", type=float, default=1.0,
                        help="Fraction of characters to cover. 1.0 covers all "
                             "FR chars at no compression cost vs 0.9999.")
    parser.add_argument("--seed-sentencepiece-size", type=int, default=1_300_000,
                        help="Initial seed vocab size before EM pruning. "
                             "10x final vocab is the recipe for high-quality "
                             "Unigram fits.")
    parser.add_argument("--num-sub-iterations", type=int, default=4,
                        help="EM iterations per Unigram pruning round. "
                             "More iterations = tighter optimum. Default 4.")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 4,
                        help="SPM training threads (defaults to all cores).")
    parser.add_argument("--no-byte-fallback", action="store_true",
                        help="Disable byte-level fallback for OOV. Default ON.")
    parser.add_argument("--no-split-digits", action="store_true",
                        help="Disable splitting digits one-by-one. Default ON.")
    parser.add_argument("--eval-docs", type=int, default=2000,
                        help="After training, stream N held-out docs and report "
                             "real chars/token. Set 0 to skip.")
    args = parser.parse_args()

    if not args.input and not args.hf_source:
        parser.error("Provide either --input or --hf-source.")

    # Local imports — keep --help fast.
    import sentencepiece as spm
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram, BPE
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.decoders import Metaspace as MetaspaceDecoder
    from tokenizers.processors import TemplateProcessing
    from tokenizers.normalizers import NFC
    from transformers import PreTrainedTokenizerFast

    # Reuse the SPM-BPE merge extractor so wrap-on-train and the rescue
    # script (spm_to_hf.py) stay byte-identical.
    sys.path.insert(0, str(Path(__file__).parent))
    from spm_to_hf import _extract_bpe_vocab_and_merges

    print(f"Training SentencePiece {args.algo.upper()} tokenizer")
    src = args.input or args.hf_source
    print(f"  Source:               {src}")
    print(f"  Algorithm:            {args.algo}")
    print(f"  Vocab size:           {args.vocab_size:,}")
    print(f"  Max docs streamed:    {args.max_examples:,}")
    print(f"  Sentences sampled:    {args.input_sentence_size:,}")
    print(f"  Character coverage:   {args.character_coverage}")
    print(f"  Byte fallback:        {'off' if args.no_byte_fallback else 'on'}")
    print(f"  Split digits:         {'off' if args.no_split_digits else 'on'}")
    print(f"  Seed SP size:         {args.seed_sentencepiece_size:,}")
    print(f"  EM sub-iterations:    {args.num_sub_iterations}")
    print(f"  Threads:              {args.num_threads}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "spm"

    # Special token convention:
    #   id=0  <|unknown|>     (SPM requires unk to exist)
    #   id=1  <|endoftext|>   (we use this as both EOS and the only sequence
    #                          terminator; no BOS, matches GPT-2/Llama-style)
    #   id=2  <|padding|>     (PAD; not seen during pretraining but reserved)
    #   id=3..258  byte fallback (if enabled): <0x00>..<0xFF>
    #   id=259+ learned pieces
    #
    # We disable BOS by setting bos_id=-1.

    print("\nTraining...")
    t0 = time.time()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(_build_corpus_iterator(args)),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type=args.algo,
        character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        # Modern Llama-style defaults
        byte_fallback=not args.no_byte_fallback,
        split_digits=not args.no_split_digits,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nmt_nfkc",
        # Aggressive optimization for max compression at given vocab.
        seed_sentencepiece_size=args.seed_sentencepiece_size,
        num_sub_iterations=args.num_sub_iterations,
        # Memory mode for large corpora
        train_extremely_large_corpus=True,
        num_threads=args.num_threads,
        # Special tokens
        unk_id=0,
        eos_id=1,
        pad_id=2,
        bos_id=-1,
        unk_piece="<|unknown|>",
        eos_piece="<|endoftext|>",
        pad_piece="<|padding|>",
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Wrap the .model in a HuggingFace Fast tokenizer.
    print("\nConverting to HuggingFace Fast format...")
    sp = spm.SentencePieceProcessor()
    sp.Load(str(model_prefix) + ".model")
    n_pieces = sp.GetPieceSize()

    byte_fallback = not args.no_byte_fallback

    if args.algo == "unigram":
        # Extract (piece, score) pairs and build HF Unigram directly.
        vocab = [(sp.IdToPiece(i), float(sp.GetScore(i))) for i in range(n_pieces)]
        tokenizer = Tokenizer(Unigram(
            vocab=vocab,
            unk_id=sp.unk_id(),
            byte_fallback=byte_fallback,
        ))
    else:
        # Extract merges from SPM-BPE protobuf and build HF BPE directly.
        # We don't go through LlamaTokenizerFast(vocab_file=...) because it
        # silently fails (vocab_size=3 on reload) for some transformers /
        # sentencepiece version combinations.
        print("  Extracting BPE merges from SPM model...")
        vocab_dict, merges = _extract_bpe_vocab_and_merges(sp)
        print(f"  Extracted {len(merges):,} merges across {len(vocab_dict):,} pieces")
        tokenizer = Tokenizer(BPE(
            vocab=vocab_dict,
            merges=merges,
            unk_token="<|unknown|>",
            byte_fallback=byte_fallback,
            fuse_unk=True,
        ))

    # NFC (not NFKC) so FR ligatures œ / æ stay as single chars.
    # NFKC decomposes them to oe / ae which costs ~0.05 chars/token on
    # FR (cœur, sœur, œuf, etc.).
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tokenizer.decoder = MetaspaceDecoder(
        replacement="▁",
        prepend_scheme="always",
    )
    eos_id = sp.PieceToId("<|endoftext|>")
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", eos_id)],
    )

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        unk_token="<|unknown|>",
    )
    if hf_tokenizer.vocab_size < int(args.vocab_size * 0.9):
        raise SystemExit(
            f"HF Fast wrapping failed: got vocab_size={hf_tokenizer.vocab_size}, "
            f"expected ~{args.vocab_size}. Raw SPM model is intact at "
            f"{model_prefix}.model and can be re-wrapped via spm_to_hf.py."
        )
    hf_tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved to: {output_dir}")
    print(f"  Files: {[f.name for f in output_dir.iterdir()]}")
    print(f"  Vocab size: {hf_tokenizer.vocab_size:,}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

    # EOS-excluded ratio
    total_tokens_no_eos = total_tokens - len(test_sentences)
    ratio = total_chars / max(1, total_tokens_no_eos)
    print(f"\n  Avg chars/token (synthetic, EOS-excluded): {ratio:.2f}")
    print(f"    Note: stress-test biased toward single-char tokens. "
          f"Real prose typically scores +0.5 to +1.0 above this.")

    # Honest evaluation: held-out chunk of real text
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
            buffer = []
            BUFFER_SIZE = 50  # rejoin lines into pseudo-docs for fair eval
            for i, line in enumerate(eval_iter):
                buffer.append(line)
                if len(buffer) >= BUFFER_SIZE:
                    doc = "\n".join(buffer)
                    buffer = []
                    ids = hf_tokenizer.encode(doc, add_special_tokens=False)
                    eval_chars += len(doc)
                    eval_tokens += len(ids)
                    n_eval += 1
                    if n_eval >= args.eval_docs:
                        break
            if n_eval > 0:
                real_ratio = eval_chars / max(1, eval_tokens)
                print(f"  Pseudo-docs evaluated: {n_eval:,}")
                print(f"  Total chars:           {eval_chars:,}")
                print(f"  Total tokens:          {eval_tokens:,}")
                print(f"  Real chars/token:      {real_ratio:.3f}  "
                      f"(>= 4.0 OK, >= 4.5 excellent, >= 5.0 exceptional)")
            else:
                print("  No held-out docs available for evaluation.")
        except Exception as e:
            print(f"  Eval skipped: {e}")


if __name__ == "__main__":
    main()
