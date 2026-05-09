"""Annotate FineWeb-2 FR documents with educational quality scores using vLLM.

Pipeline step 1/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs (this script)
    2. train_quality_classifier.py — train fasttext on annotations
    3. filter_dataset.py     — apply classifier to full corpus

Usage:
    # Start vLLM server first:
    #   python -m vllm.entrypoints.openai.api_server \
    #       --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    #       --quantization awq --max-model-len 2048 --gpu-memory-utilization 0.90

    # Then run annotation:
    python scripts/annotate_quality.py \
        --output data/annotations_fr.jsonl \
        --n-docs 500000 \
        --batch-size 64 \
        --api-url http://localhost:8000/v1
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


SCORING_PROMPT = """Tu es un expert en évaluation de la qualité de textes francophones pour l'entraînement de modèles de langage.

Évalue l'extrait suivant selon ces critères, en attribuant 1 point par critère rempli :

1. **Pertinence et cohérence** : Le texte traite d'un sujet identifiable de manière cohérente (pas du spam, navigation, liste de liens, ou texte incompréhensible).
2. **Valeur informative** : Le texte apporte des connaissances, explications ou raisonnements utiles (pas juste des opinions vagues, de la pub ou du contenu vide).
3. **Qualité linguistique** : Le texte est bien écrit en français correct, avec une grammaire et une syntaxe propres.
4. **Profondeur** : Le texte va au-delà du superficiel — il développe ses idées, fournit des exemples, ou structure un raisonnement.
5. **Valeur éducative** : Le texte pourrait être utilisé dans un contexte éducatif (école, université, formation, vulgarisation scientifique).

Extrait :
---
{text}
---

Réponds UNIQUEMENT avec un JSON : {{"score": N}} où N est entre 0 et 5."""


def build_messages(text: str, max_chars: int = 1500) -> list:
    """Build chat messages for the scoring prompt."""
    # Truncate long documents to fit in context
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return [
        {"role": "user", "content": SCORING_PROMPT.format(text=text)},
    ]


def parse_score(response_text: str) -> int:
    """Extract score from LLM response. Returns -1 if unparseable."""
    # Try JSON parse first
    try:
        data = json.loads(response_text.strip())
        score = int(data.get("score", -1))
        if 0 <= score <= 5:
            return score
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: look for "score": N or just a digit
    match = re.search(r'"score"\s*:\s*(\d)', response_text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 5:
            return score

    match = re.search(r'\b([0-5])\b', response_text)
    if match:
        return int(match.group(1))

    return -1


def score_batch(client, texts: list, model: str) -> list:
    """Score a batch of texts using the vLLM OpenAI-compatible API."""
    results = []

    def score_one(text):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(text),
                max_tokens=30,
                temperature=0.0,
            )
            reply = response.choices[0].message.content
            return parse_score(reply)
        except Exception as e:
            return -1

    with ThreadPoolExecutor(max_workers=min(len(texts), 32)) as pool:
        futures = {pool.submit(score_one, t): i for i, t in enumerate(texts)}
        results = [None] * len(texts)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Annotate FineWeb-2 FR with educational quality scores")
    parser.add_argument("--output", type=str, default="data/annotations_fr.jsonl",
                        help="Output JSONL file for annotations")
    parser.add_argument("--n-docs", type=int, default=500_000,
                        help="Number of documents to annotate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for concurrent API calls")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--model", type=str,
                        default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
                        help="Model name served by vLLM")
    parser.add_argument("--hf-source", type=str,
                        default="HuggingFaceFW/fineweb-2",
                        help="HuggingFace dataset to annotate")
    parser.add_argument("--hf-name", type=str, default="fra_Latn",
                        help="Dataset config/subset name")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    from openai import OpenAI
    from datasets import load_dataset

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count existing annotations
    n_done = 0
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            n_done = sum(1 for _ in f)
        print(f"Resuming from {n_done:,} existing annotations")

    print(f"Annotating {args.n_docs:,} docs from {args.hf_source} ({args.hf_name})")
    print(f"Model: {args.model}")
    print(f"API: {args.api_url}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")

    ds = load_dataset(args.hf_source, name=args.hf_name,
                      split=args.hf_split, streaming=True)

    mode = "a" if args.resume else "w"
    f_out = open(output_path, mode, encoding="utf-8")

    batch_texts = []
    batch_ids = []
    n_annotated = 0
    n_skipped = 0
    n_total = 0
    t0 = time.time()
    score_dist = [0] * 6  # counts for scores 0-5

    try:
        for example in ds:
            n_total += 1

            # Skip already-annotated docs when resuming
            if n_total <= n_done:
                continue

            text = example.get(args.text_field) or ""
            if not text or len(text) < 100:
                continue

            batch_texts.append(text)
            batch_ids.append(n_total)

            if len(batch_texts) >= args.batch_size:
                scores = score_batch(client, batch_texts, args.model)

                for text, doc_id, score in zip(batch_texts, batch_ids, scores):
                    if score < 0:
                        n_skipped += 1
                        continue
                    record = {
                        "text": text[:5000],  # cap stored text
                        "score": score,
                        "doc_id": doc_id,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_annotated += 1
                    score_dist[score] += 1

                f_out.flush()
                batch_texts = []
                batch_ids = []

                if n_annotated % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = n_annotated / elapsed
                    eta = (args.n_docs - n_annotated) / max(rate, 0.01)
                    dist_str = " ".join(f"{i}:{score_dist[i]}" for i in range(6))
                    print(f"  {n_annotated:,}/{args.n_docs:,} | "
                          f"{rate:.1f} docs/s | "
                          f"ETA {eta/3600:.1f}h | "
                          f"skip={n_skipped} | "
                          f"dist=[{dist_str}]")

            if n_annotated >= args.n_docs:
                break

    except KeyboardInterrupt:
        print(f"\nInterrupted at {n_annotated:,} annotations")
    finally:
        # Flush remaining batch
        if batch_texts:
            scores = score_batch(client, batch_texts, args.model)
            for text, doc_id, score in zip(batch_texts, batch_ids, scores):
                if score >= 0:
                    record = {"text": text[:5000], "score": score, "doc_id": doc_id}
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_annotated += 1
                    score_dist[score] += 1
        f_out.close()

    elapsed = time.time() - t0
    print(f"\nDone: {n_annotated:,} annotations in {elapsed/3600:.1f}h")
    print(f"Skipped (unparseable): {n_skipped:,}")
    print(f"Score distribution:")
    for i in range(6):
        pct = score_dist[i] / max(n_annotated, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"  {i}: {score_dist[i]:>7,} ({pct:5.1f}%) {bar}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
