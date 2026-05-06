"""
Convert a SentencePiece .model file to a HuggingFace Fast tokenizer.

Auto-detects Unigram vs BPE from the SPM model and builds the matching
HF tokenizer model directly via the `tokenizers` library, bypassing
LlamaTokenizerFast(vocab_file=...) which silently fails to convert the
vocab in some transformers / sentencepiece version combinations
(symptom : tokenizer.json contains only the 3 special tokens, vocab_size=3
on reload).

Usage:
    # Unigram or BPE, auto-detected
    python scripts/spm_to_hf.py \\
        --spm-model data/velvet_tok_100k_spm_bpe/spm.model \\
        --output data/velvet_tok_100k_spm_bpe
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def _detect_model_type(model_path: str) -> str:
    """Read the SPM protobuf header to detect Unigram vs BPE."""
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    proto = sp_pb2.ModelProto()
    with open(model_path, "rb") as f:
        proto.ParseFromString(f.read())
    # 1=UNIGRAM 2=BPE 3=WORD 4=CHAR
    return {1: "unigram", 2: "bpe", 3: "word", 4: "char"}.get(
        proto.trainer_spec.model_type, "unknown"
    )


def _extract_bpe_vocab_and_merges(sp):
    """Extract (vocab, merges) from an SPM-BPE model.

    SPM-BPE doesn't store merges explicitly — they're implicit in the
    piece IDs (lower id = earlier merge). For each multi-char piece P,
    we enumerate every (left, right) split where both halves exist in
    the vocab. The resulting merges are sorted by piece-id ascending so
    the BPE encoder applies them in the same order they were learned.

    This mirrors transformers.convert_slow_tokenizer.SentencePieceExtractor
    but is reproduced inline so we don't depend on a private API that
    drifts across transformers versions.
    """
    n = sp.GetPieceSize()
    vocab = {sp.IdToPiece(i): i for i in range(n)}

    merges = []
    for i in range(n):
        # Skip specials, byte-fallback pieces, and base-character pieces.
        if sp.IsControl(i) or sp.IsByte(i) or sp.IsUnknown(i):
            continue
        piece = sp.IdToPiece(i)
        if len(piece) <= 1:
            continue

        local = []
        for split in range(1, len(piece)):
            left, right = piece[:split], piece[split:]
            if left in vocab and right in vocab:
                local.append((left, right, vocab[left], vocab[right]))
        # Within a piece, prefer splits where both halves came earlier.
        local.sort(key=lambda t: (t[2], t[3]))
        for left, right, _, _ in local:
            merges.append((left, right, i))

    # Globally, sort by piece-id (= chronological merge order).
    merges.sort(key=lambda t: t[2])
    merges = [(left, right) for left, right, _ in merges]
    return vocab, merges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-model", required=True,
                        help="Path to spm.model file")
    parser.add_argument("--output", required=True,
                        help="Output directory for the HF Fast tokenizer")
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--pad-token", default="<|padding|>")
    parser.add_argument("--unk-token", default="<|unknown|>")
    parser.add_argument("--no-byte-fallback", action="store_true",
                        help="Disable byte-level fallback. Default ON, must "
                             "match what was used during SPM training.")
    parser.add_argument("--algo", choices=["auto", "unigram", "bpe"],
                        default="auto",
                        help="Override the auto-detected SPM model type.")
    args = parser.parse_args()

    import sentencepiece as spm
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram, BPE
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers.decoders import Metaspace as MetaspaceDecoder
    from tokenizers.processors import TemplateProcessing
    from tokenizers.normalizers import NFC
    from transformers import PreTrainedTokenizerFast

    print(f"Loading SPM model: {args.spm_model}")
    detected = _detect_model_type(args.spm_model)
    algo = args.algo if args.algo != "auto" else detected
    print(f"  Detected type: {detected}")
    print(f"  Using algo:    {algo}")
    if algo not in ("unigram", "bpe"):
        raise SystemExit(f"Unsupported SPM model type: {algo}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    n = sp.GetPieceSize()
    print(f"  Pieces: {n:,}")
    print(f"  unk_id: {sp.unk_id()}, eos_id: {sp.eos_id()}, pad_id: {sp.pad_id()}")

    byte_fallback = not args.no_byte_fallback

    if algo == "unigram":
        # Build HF Unigram from (piece, score) pairs.
        vocab = [(sp.IdToPiece(i), float(sp.GetScore(i))) for i in range(n)]
        tokenizer = Tokenizer(Unigram(
            vocab=vocab,
            unk_id=sp.unk_id(),
            byte_fallback=byte_fallback,
        ))
    else:
        print("  Extracting BPE merges from SPM model (this can take a minute)...")
        vocab_dict, merges = _extract_bpe_vocab_and_merges(sp)
        print(f"  Extracted {len(merges):,} merges across {len(vocab_dict):,} pieces")
        tokenizer = Tokenizer(BPE(
            vocab=vocab_dict,
            merges=merges,
            unk_token=args.unk_token,
            byte_fallback=byte_fallback,
            fuse_unk=True,
        ))

    # NFC (not NFKC) so FR ligatures œ / æ stay as single chars.
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tokenizer.decoder = MetaspaceDecoder(
        replacement="▁",
        prepend_scheme="always",
    )

    eos_id = sp.PieceToId(args.eos_token)
    if eos_id < 0 or eos_id == sp.unk_id():
        raise SystemExit(
            f"EOS token {args.eos_token!r} not found in SPM vocab. "
            f"Either retrain with eos_piece={args.eos_token!r} or pass --eos-token."
        )
    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {args.eos_token}",
        special_tokens=[(args.eos_token, eos_id)],
    )

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

    # Sanity check on a few FR sentences.
    print("\nSanity check:")
    for test in [
        "Bonjour, comment allez-vous aujourd'hui ?",
        "Les œufs et le bœuf coûtent 12,50 € au marché.",
        "Le développement de l'intelligence artificielle progresse rapidement.",
    ]:
        ids = hf_tokenizer.encode(test)
        tokens = hf_tokenizer.convert_ids_to_tokens(ids)
        ratio = len(test) / max(1, len(ids) - 1)
        print(f"  \"{test}\"")
        print(f"    -> {len(ids)} tokens, ratio {ratio:.2f}: {tokens[:14]}"
              f"{'...' if len(tokens) > 14 else ''}")


if __name__ == "__main__":
    main()
