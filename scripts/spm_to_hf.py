"""
Convert a SentencePiece .model file to a HuggingFace Fast tokenizer.

Use this to fix-up an SPM training output when LlamaTokenizerFast wrapping
failed silently (typically results in a tokenizer.json that only contains
the special tokens, with vocab_size=3 on reload).

This script bypasses LlamaTokenizerFast entirely and builds the Fast
tokenizer directly from the SPM model via the tokenizers library, which
handles SPM Unigram + byte_fallback natively.

Usage:
    python scripts/spm_to_hf.py \\
        --spm-model data/velvet_tok_100k_spm/spm.model \\
        --output data/velvet_tok_100k_spm
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-model", required=True,
                        help="Path to spm.model file")
    parser.add_argument("--output", required=True,
                        help="Output directory for the HF Fast tokenizer")
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--pad-token", default="<|padding|>")
    parser.add_argument("--unk-token", default="<|unknown|>")
    args = parser.parse_args()

    import sentencepiece as spm
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.decoders import Metaspace as MetaspaceDecoder
    from tokenizers.processors import TemplateProcessing
    from tokenizers.normalizers import NFKC
    from transformers import PreTrainedTokenizerFast

    # Load the SPM model and pull out (piece, score) for every id, in order.
    print(f"Loading SPM model: {args.spm_model}")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    n = sp.GetPieceSize()
    print(f"  Pieces: {n:,}")
    print(f"  unk_id: {sp.unk_id()}, eos_id: {sp.eos_id()}, pad_id: {sp.pad_id()}")

    vocab = [(sp.IdToPiece(i), float(sp.GetScore(i))) for i in range(n)]

    # Build the Fast tokenizer.
    # byte_fallback=True is critical: it tells the Unigram model to encode
    # OOV characters as their UTF-8 bytes via the <0xXX> pieces that SPM
    # added during training.
    tokenizer = Tokenizer(Unigram(
        vocab=vocab,
        unk_id=sp.unk_id(),
        byte_fallback=True,
    ))

    # SPM was trained with normalization_rule_name="nmt_nfkc". The closest
    # HF normalizer is NFKC; nmt-specific tweaks (e.g. control char strip)
    # don't materially affect FR text. Add NFKC to mirror SPM's normalizer.
    tokenizer.normalizer = NFKC()

    # SPM uses ▁ to mark the start of a "word" (whitespace boundary). The
    # Metaspace pre-tokenizer / decoder pair is the HF equivalent.
    tokenizer.pre_tokenizer = Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tokenizer.decoder = MetaspaceDecoder(
        replacement="▁",
        prepend_scheme="always",
    )

    # Auto-append EOS on encode() so downstream training code doesn't have
    # to special-case it.
    eos_id = sp.PieceToId(args.eos_token)
    if eos_id < 0 or eos_id == sp.unk_id():
        raise SystemExit(
            f"EOS token {args.eos_token!r} not found in SPM vocab. "
            f"Either retrain with eos_piece={args.eos_token!r} or pass --eos-token."
        )
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[(args.eos_token, eos_id)],
    )

    # Wrap as a HuggingFace Fast tokenizer and persist.
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=args.eos_token,
        pad_token=args.pad_token,
        unk_token=args.unk_token,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(str(output_dir))

    print(f"\nSaved HF Fast tokenizer to: {output_dir}")
    print(f"  Vocab size: {hf_tokenizer.vocab_size:,}")
    print(f"  Files: {[f.name for f in output_dir.iterdir()]}")

    # Sanity check
    test = "Bonjour, comment allez-vous aujourd'hui ?"
    ids = hf_tokenizer.encode(test)
    tokens = hf_tokenizer.convert_ids_to_tokens(ids)
    print(f"\nSanity check:")
    print(f"  Text:   {test}")
    print(f"  Tokens: {tokens}")
    print(f"  Ratio:  {len(test) / max(1, len(ids) - 1):.2f} chars/token")


if __name__ == "__main__":
    main()
