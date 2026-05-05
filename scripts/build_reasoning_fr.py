"""Build a French reasoning corpus by translating GSM8K and MATH to French.

Pipeline:
    1. Pull GSM8K (8.5K train) and MATH (7.5K train) from HuggingFace.
    2. Translate the natural-language parts (question + non-formula prose)
       via NLLB-200-distilled-600M (English -> French).
    3. Preserve LaTeX expressions and numerical answers as-is.
    4. Format each example with explicit reasoning markers:
            Q: <translated question>
            R:
            <translated reasoning>
            #### <answer>
       so the iterative reasoner sees clear question / chain / answer
       boundaries.
    5. Write JSONL to data/reasoning_fr/{gsm8k_fr,math_fr}.jsonl

Usage:
    # Default: use NLLB-200-distilled-600M, all 16K examples, batch 16
    python scripts/build_reasoning_fr.py

    # Quick test (100 examples, no GPU)
    python scripts/build_reasoning_fr.py --limit 100 --device cpu

    # Bigger / better translator
    python scripts/build_reasoning_fr.py --translator facebook/nllb-200-3.3B
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

# Make `rvelvet` importable when running from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------
# LaTeX-aware splitting
# ----------------------------------------------------------------------
# We don't translate inside math expressions. Tokens we keep verbatim:
#   $...$ inline math
#   $$...$$ display math
#   \\[...\\] / \\(...\\) display/inline math
#   \\boxed{...} answer wrappers
#   numbers (12, 3.14, 5/8)
#
# Strategy: replace each protected span with a placeholder __MATH_i__,
# translate the rest, then restore.

_MATH_PATTERNS = [
    (re.compile(r"\$\$.+?\$\$", re.DOTALL), "$$"),
    (re.compile(r"\$[^$\n]+\$"), "$"),
    (re.compile(r"\\\[.+?\\\]", re.DOTALL), "bracket"),
    (re.compile(r"\\\(.+?\\\)", re.DOTALL), "paren"),
    (re.compile(r"\\boxed\{[^{}]*\}"), "boxed"),
]


def _protect_math(text: str):
    placeholders = []

    def _stash(m):
        placeholders.append(m.group(0))
        return f" __MATH_{len(placeholders) - 1}__ "

    for pat, _ in _MATH_PATTERNS:
        text = pat.sub(_stash, text)
    return text, placeholders


def _restore_math(text: str, placeholders: List[str]) -> str:
    for i, raw in enumerate(placeholders):
        text = text.replace(f"__MATH_{i}__", raw)
        # Translators sometimes mangle the placeholder — handle simple variants.
        text = text.replace(f"__ MATH _{i}__", raw)
        text = text.replace(f"__MATH{i}__", raw)
    return text


# ----------------------------------------------------------------------
# Translator
# ----------------------------------------------------------------------
class NLLBTranslator:
    """Thin wrapper around NLLB-200 for batch English -> French translation."""

    def __init__(self, model_id: str = "facebook/nllb-200-distilled-600M",
                 device: str = "cuda", batch_size: int = 16, max_length: int = 512):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

        print(f"  Loading translator: {model_id} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()

        # Locate the FR target token in a way that works across NLLB tokenizer
        # variants (the API moved between transformers versions).
        fr_id = self.tokenizer.convert_tokens_to_ids("fra_Latn")
        if fr_id is None or fr_id == self.tokenizer.unk_token_id:
            # Newer transformers expose lang_code_to_id only on Slow tokenizers.
            fr_id = self.tokenizer.lang_code_to_id["fra_Latn"]
        self.fr_id = fr_id

    def translate(self, texts: Iterable[str]) -> List[str]:
        torch = self.torch
        out: List[str] = []
        batch: List[str] = []

        def _flush():
            if not batch:
                return
            enc = self.tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    forced_bos_token_id=self.fr_id,
                    max_length=self.max_length,
                    num_beams=2,
                    early_stopping=True,
                )
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            out.extend(decoded)
            batch.clear()

        for t in texts:
            batch.append(t)
            if len(batch) >= self.batch_size:
                _flush()
        _flush()
        return out


def translate_with_protected_math(translator: NLLBTranslator, texts: List[str]) -> List[str]:
    """Translate a batch of texts while keeping LaTeX/math spans verbatim."""
    protected = [_protect_math(t) for t in texts]
    cleaned = [p[0] for p in protected]
    translated = translator.translate(cleaned)
    return [_restore_math(tr, p[1]) for tr, p in zip(translated, protected)]


# ----------------------------------------------------------------------
# GSM8K
# ----------------------------------------------------------------------
def build_gsm8k_fr(translator: NLLBTranslator, out_path: Path, limit: int = None):
    from datasets import load_dataset

    print("\n[GSM8K] loading...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  {len(ds)} examples")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        questions = [ex["question"] for ex in ds]
        # GSM8K answers contain reasoning + "#### <number>"
        raw_answers = [ex["answer"] for ex in ds]

        # Split the canonical "#### <num>" suffix off so we don't translate it.
        bodies, finals = [], []
        for a in raw_answers:
            if "####" in a:
                body, final = a.rsplit("####", 1)
                bodies.append(body.strip())
                finals.append(final.strip())
            else:
                bodies.append(a)
                finals.append("")

        print("  Translating questions...")
        q_fr = translate_with_protected_math(translator, questions)
        print("  Translating chains-of-thought...")
        b_fr = translate_with_protected_math(translator, bodies)

        for q, body_fr, final in zip(q_fr, b_fr, finals):
            text = f"Q : {q}\nR :\n{body_fr}"
            if final:
                text += f"\n#### {final}"
            f.write(json.dumps({"text": text, "source": "gsm8k_fr"},
                               ensure_ascii=False) + "\n")
    print(f"  → {out_path}")


# ----------------------------------------------------------------------
# MATH
# ----------------------------------------------------------------------
def build_math_fr(translator: NLLBTranslator, out_path: Path, limit: int = None):
    from datasets import load_dataset

    print("\n[MATH] loading...")
    # The hendrycks/competition_math dataset isn't always available on HF;
    # try a couple of mirrors before giving up.
    ds = None
    for repo in ("hendrycks/competition_math", "lighteval/MATH"):
        try:
            ds = load_dataset(repo, split="train")
            print(f"  Loaded from {repo}")
            break
        except Exception as e:
            print(f"  {repo} failed ({e}); trying next mirror.")
    if ds is None:
        print("  Could not load MATH dataset; skipping.")
        return

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"  {len(ds)} examples")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        problems = [ex.get("problem") or ex.get("question", "") for ex in ds]
        solutions = [ex.get("solution") or ex.get("answer", "") for ex in ds]

        print("  Translating problems...")
        p_fr = translate_with_protected_math(translator, problems)
        print("  Translating solutions...")
        s_fr = translate_with_protected_math(translator, solutions)

        for q, s in zip(p_fr, s_fr):
            text = f"Q : {q}\nR :\n{s}"
            f.write(json.dumps({"text": text, "source": "math_fr"},
                               ensure_ascii=False) + "\n")
    print(f"  → {out_path}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/reasoning_fr")
    parser.add_argument("--translator", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu / cuda:0 (default: auto)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap each dataset to N examples (debug).")
    parser.add_argument("--skip_gsm8k", action="store_true")
    parser.add_argument("--skip_math", action="store_true")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    translator = NLLBTranslator(
        model_id=args.translator,
        device=args.device,
        batch_size=args.batch_size,
    )

    if not args.skip_gsm8k:
        build_gsm8k_fr(translator, out_dir / "gsm8k_fr.jsonl", limit=args.limit)
    if not args.skip_math:
        build_math_fr(translator, out_dir / "math_fr.jsonl", limit=args.limit)

    print("\nDone. Pipe these JSONL files via configs/data/reasoning_fr.yaml")


if __name__ == "__main__":
    main()
