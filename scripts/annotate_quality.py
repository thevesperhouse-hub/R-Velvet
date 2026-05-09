"""Annotate FineWeb-2 FR documents with multi-dimensional quality scores using vLLM.

Pipeline step 1/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs (this script)
    2. train_quality_classifier.py — train fasttext on annotations
    3. filter_dataset.py     — apply classifier to full corpus

Each document gets scored on 6 independent axes (0-2 each):
    - coherence:  logical structure, reasoning quality
    - pedagogy:   educational value, clarity of explanations
    - linguistic: grammar, syntax, proper French
    - depth:      substance, examples, developed arguments
    - factuality: reliable, verifiable information
    - code_quality: (code only) clean, documented, idiomatic

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


DIMENSIONS = ["coherence", "pedagogy", "linguistic", "depth", "factuality"]

SCORING_PROMPT = """Tu es un expert en évaluation de la qualité de textes francophones pour l'entraînement de modèles de langage.

Évalue l'extrait suivant sur chaque axe avec une note de 0 à 2 :

**coherence** — Structure logique et raisonnement
  0 = incohérent, spam, fragments sans lien
  1 = compréhensible mais mal organisé, saute du coq à l'âne
  2 = raisonnement clair, progression logique, bien structuré

**pedagogy** — Valeur pédagogique et clarté des explications
  0 = aucune valeur éducative (pub, navigation, opinions vides)
  1 = informatif mais pas pédagogique, manque d'explications
  2 = explique clairement, pourrait servir dans un cours ou un manuel

**linguistic** — Qualité du français
  0 = illisible, bourrés de fautes, mélange de langues
  1 = correct mais maladroit, registre familier, quelques erreurs
  2 = français soigné, syntaxe maîtrisée, registre approprié

**depth** — Profondeur et substance
  0 = superficiel, une phrase, liste sans contexte
  1 = traite le sujet mais reste en surface
  2 = développe en profondeur, exemples, nuances, argumentation

**factuality** — Fiabilité factuelle
  0 = faux, trompeur, conspirationniste, désinformation
  1 = plausible mais non vérifiable, opinions présentées comme faits
  2 = fiable, sourcé ou vérifiable, contenu factuel solide

Extrait :
---
{text}
---

Réponds UNIQUEMENT avec un JSON, rien d'autre :
{{"coherence": N, "pedagogy": N, "linguistic": N, "depth": N, "factuality": N}}"""


SCORING_PROMPT_CODE = """Tu es un expert en évaluation de la qualité de code source pour l'entraînement de modèles de langage.

Évalue l'extrait suivant sur chaque axe avec une note de 0 à 2 :

**coherence** — Structure logique du code
  0 = fragments incomplets, code cassé, pas de logique visible
  1 = fonctionne probablement mais mal organisé
  2 = bien structuré, flux logique clair, bonne architecture

**code_quality** — Propreté et bonnes pratiques
  0 = illisible, noms de variables obscurs, code spaghetti
  1 = acceptable mais pas exemplaire, manque de standards
  2 = propre, idiomatique, suit les conventions du langage

**pedagogy** — Valeur éducative du code
  0 = aucune (boilerplate, config auto-générée, code trivial)
  1 = montre un pattern utile mais sans explication
  2 = code instructif, bien commenté, bon exemple à apprendre

**depth** — Complexité pertinente
  0 = trivial (hello world, imports seuls, fichiers vides)
  1 = résout un problème réel mais simple
  2 = algorithme non trivial, design pattern, logique métier intéressante

**factuality** — Fiabilité technique
  0 = bugs évidents, anti-patterns dangereux, code vulnérable
  1 = fonctionne mais avec des pratiques discutables
  2 = correct, sécurisé, gestion d'erreurs appropriée

Extrait :
---
{text}
---

Réponds UNIQUEMENT avec un JSON, rien d'autre :
{{"coherence": N, "code_quality": N, "pedagogy": N, "depth": N, "factuality": N}}"""


def is_code(text: str) -> bool:
    """Heuristic: detect if text is primarily code."""
    indicators = [
        "def ", "class ", "import ", "function ", "return ",
        "if (", "for (", "while (", "const ", "let ", "var ",
        "#include", "public static", "func ", "fn ",
    ]
    lines = text[:2000].split("\n")
    code_lines = sum(1 for l in lines if any(ind in l for ind in indicators))
    return code_lines > len(lines) * 0.3


def build_messages(text: str, max_chars: int = 1500) -> list:
    """Build chat messages for the scoring prompt."""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    prompt = SCORING_PROMPT_CODE if is_code(text) else SCORING_PROMPT
    return [
        {"role": "user", "content": prompt.format(text=text)},
    ]


def get_dimensions(text: str) -> list:
    """Return the dimension names for this text type."""
    if is_code(text):
        return ["coherence", "code_quality", "pedagogy", "depth", "factuality"]
    return DIMENSIONS


def parse_scores(response_text: str, text: str) -> dict:
    """Extract multi-dimensional scores from LLM response.

    Returns dict with dimension scores, or None if unparseable.
    """
    # Try JSON parse
    try:
        # Find JSON in response (might have text around it)
        match = re.search(r'\{[^}]+\}', response_text)
        if match:
            data = json.loads(match.group())
            dims = get_dimensions(text)
            scores = {}
            for dim in dims:
                val = data.get(dim, -1)
                if isinstance(val, (int, float)) and 0 <= val <= 2:
                    scores[dim] = int(val)
                else:
                    return None
            # Compute aggregate (sum of all dims, max = 10)
            scores["total"] = sum(scores[d] for d in dims)
            return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return None


def score_batch(client, texts: list, model: str) -> list:
    """Score a batch of texts using the vLLM OpenAI-compatible API."""

    def score_one(text):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(text),
                max_tokens=80,
                temperature=0.0,
            )
            reply = response.choices[0].message.content
            return parse_scores(reply, text)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(len(texts), 32)) as pool:
        futures = {pool.submit(score_one, t): i for i, t in enumerate(texts)}
        results = [None] * len(texts)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Annotate FineWeb-2 FR with multi-dimensional quality scores")
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

    # Resume support
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
    print(f"Dimensions: coherence, pedagogy, linguistic, depth, factuality (+code_quality for code)")

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

    # Track per-dimension averages
    dim_sums = {}
    dim_counts = {}

    try:
        for example in ds:
            n_total += 1

            if n_total <= n_done:
                continue

            text = example.get(args.text_field) or ""
            if not text or len(text) < 100:
                continue

            batch_texts.append(text)
            batch_ids.append(n_total)

            if len(batch_texts) >= args.batch_size:
                results = score_batch(client, batch_texts, args.model)

                for text, doc_id, scores in zip(batch_texts, batch_ids, results):
                    if scores is None:
                        n_skipped += 1
                        continue
                    record = {
                        "text": text[:5000],
                        "doc_id": doc_id,
                        **scores,
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_annotated += 1

                    for dim, val in scores.items():
                        if dim == "total":
                            continue
                        dim_sums[dim] = dim_sums.get(dim, 0) + val
                        dim_counts[dim] = dim_counts.get(dim, 0) + 1

                f_out.flush()
                batch_texts = []
                batch_ids = []

                if n_annotated % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = n_annotated / elapsed
                    eta = (args.n_docs - n_annotated) / max(rate, 0.01)

                    avg_parts = []
                    for dim in ["coherence", "pedagogy", "linguistic", "depth", "factuality", "code_quality"]:
                        if dim in dim_counts and dim_counts[dim] > 0:
                            avg = dim_sums[dim] / dim_counts[dim]
                            avg_parts.append(f"{dim[:4]}={avg:.2f}")

                    print(f"  {n_annotated:,}/{args.n_docs:,} | "
                          f"{rate:.1f} docs/s | "
                          f"ETA {eta/3600:.1f}h | "
                          f"skip={n_skipped} | "
                          f"{' '.join(avg_parts)}")

            if n_annotated >= args.n_docs:
                break

    except KeyboardInterrupt:
        print(f"\nInterrupted at {n_annotated:,} annotations")
    finally:
        if batch_texts:
            results = score_batch(client, batch_texts, args.model)
            for text, doc_id, scores in zip(batch_texts, batch_ids, results):
                if scores is not None:
                    record = {"text": text[:5000], "doc_id": doc_id, **scores}
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_annotated += 1
        f_out.close()

    elapsed = time.time() - t0
    print(f"\nDone: {n_annotated:,} annotations in {elapsed/3600:.1f}h")
    print(f"Skipped (unparseable): {n_skipped:,}")

    print(f"\nDimension averages (0-2 scale):")
    for dim in ["coherence", "pedagogy", "linguistic", "depth", "factuality", "code_quality"]:
        if dim in dim_counts and dim_counts[dim] > 0:
            avg = dim_sums[dim] / dim_counts[dim]
            bar = "█" * int(avg * 20)
            print(f"  {dim:15s}: {avg:.3f}/2.0 {bar}")

    print(f"\nSaved: {output_path}")
    print(f"\nEach record contains: text, doc_id, coherence, pedagogy, linguistic|code_quality, depth, factuality, total")
    print(f"Total score range: 0-10 (sum of 5 dimensions)")


if __name__ == "__main__":
    main()
