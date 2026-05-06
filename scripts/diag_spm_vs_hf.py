"""
Diagnostic: compare raw SentencePiece encoding vs the HF Fast wrapper
produced by spm_to_hf.py / train_tokenizer_spm.py.

If the two paths produce different token counts on the same text, the
HF wrapping is suboptimal (= my SPM-BPE merge reconstruction is wrong).
If they produce the same token counts, the SPM model itself is the
bottleneck (= the algo / training was suboptimal, not the wrapping).

Usage:
    python scripts/diag_spm_vs_hf.py \\
        --spm-model data/velvet_tok_100k_spm_bpe/spm.model \\
        --hf-tokenizer data/velvet_tok_100k_spm_bpe \\
        --hf-source HuggingFaceFW/fineweb-2:fra_Latn:train \\
        --n-docs 200
"""

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm-model", required=True)
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--hf-source", default=None,
                    help="HF spec '<repo>:<name>:<split>' for held-out eval")
    ap.add_argument("--input", default=None,
                    help="Local text file (one doc per line)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--show-diffs", type=int, default=3,
                    help="Show first N docs where SPM and HF disagree")
    args = ap.parse_args()

    import sentencepiece as spm
    from transformers import PreTrainedTokenizerFast

    print("Loading...")
    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    hf = PreTrainedTokenizerFast.from_pretrained(args.hf_tokenizer)
    print(f"  SPM pieces:  {sp.GetPieceSize():,}")
    print(f"  HF vocab:    {hf.vocab_size:,}")
    if sp.GetPieceSize() != hf.vocab_size:
        print(f"  !! Vocab size mismatch — wrapping is definitely broken.")

    # Build doc iterator
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
        # Fallback: a few representative FR sentences.
        def gen():
            for _ in range(1):
                yield (
                    "Le développement de l'intelligence artificielle progresse rapidement. "
                    "Les œufs et le bœuf coûtent 12,50 € au marché. "
                    "Qu'est-ce que tu fais demain à 14h30 ? "
                    "Bonjour, comment allez-vous aujourd'hui ?"
                )

    total_chars = 0
    spm_tokens_total = 0
    hf_tokens_total = 0
    n_match = 0
    n_diff = 0
    diffs_shown = 0
    n = 0

    print(f"\nEncoding {args.n_docs} docs through both paths...\n")
    for doc in gen():
        n += 1
        chars = len(doc)
        spm_ids = sp.EncodeAsIds(doc)
        hf_ids = hf.encode(doc, add_special_tokens=False)

        total_chars += chars
        spm_tokens_total += len(spm_ids)
        hf_tokens_total += len(hf_ids)

        if spm_ids == hf_ids:
            n_match += 1
        else:
            n_diff += 1
            if diffs_shown < args.show_diffs:
                # Find first divergence
                for i, (a, b) in enumerate(zip(spm_ids, hf_ids)):
                    if a != b:
                        first_diff = i
                        break
                else:
                    first_diff = min(len(spm_ids), len(hf_ids))

                preview = doc[:120].replace("\n", " ")
                print(f"  Diff #{diffs_shown + 1} (doc {n}):")
                print(f"    text: {preview!r}")
                print(f"    SPM:  {len(spm_ids)} tokens")
                print(f"    HF:   {len(hf_ids)} tokens")
                print(f"    First divergence at position {first_diff}:")
                ctx_lo = max(0, first_diff - 2)
                ctx_hi = min(min(len(spm_ids), len(hf_ids)), first_diff + 5)
                spm_ctx = [sp.IdToPiece(t) for t in spm_ids[ctx_lo:ctx_hi]]
                hf_ctx = hf.convert_ids_to_tokens(hf_ids[ctx_lo:ctx_hi])
                print(f"      SPM: {spm_ctx}")
                print(f"      HF:  {hf_ctx}")
                print()
                diffs_shown += 1

        if n >= args.n_docs:
            break

    print("=" * 60)
    print(f"Docs:                {n:,}")
    print(f"Total chars:         {total_chars:,}")
    print(f"SPM raw  total tok:  {spm_tokens_total:,}  ({total_chars/max(1,spm_tokens_total):.3f} c/t)")
    print(f"HF wrap  total tok:  {hf_tokens_total:,}  ({total_chars/max(1,hf_tokens_total):.3f} c/t)")
    print(f"Identical encoding:  {n_match}/{n} ({100*n_match/max(1,n):.1f}%)")
    print(f"Different encoding:  {n_diff}/{n}")
    print()

    if hf_tokens_total == spm_tokens_total:
        print("==> Wrapping is CORRECT. The SPM model itself is the bottleneck.")
        print("    The 4.4 chars/token reflects what SPM-BPE actually produces ;")
        print("    no improvement to expect from rewrapping.")
    elif hf_tokens_total > spm_tokens_total:
        delta = hf_tokens_total - spm_tokens_total
        pct = 100 * delta / spm_tokens_total
        spm_ratio = total_chars / max(1, spm_tokens_total)
        hf_ratio = total_chars / max(1, hf_tokens_total)
        print(f"==> Wrapping is BROKEN. HF emits +{delta:,} tokens ({pct:.2f}%).")
        print(f"    SPM raw  : {spm_ratio:.3f} c/t")
        print(f"    HF wrap  : {hf_ratio:.3f} c/t")
        print(f"    Lost     : {spm_ratio - hf_ratio:.3f} c/t to the wrapping.")
        print(f"    Fix the merge extraction in spm_to_hf.py to recover this gap.")
    else:
        print("==> HF emits FEWER tokens than SPM raw — unusual. Check the")
        print("    HF normalizer / pre-tokenizer for unintended dropouts.")


if __name__ == "__main__":
    main()
