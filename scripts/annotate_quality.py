"""Annotate FineWeb-2 FR documents with multi-dimensional quality scores using vLLM.

Pipeline step 1/3 for building Vesper-Edu-FR:
    1. annotate_quality.py   — LLM scores ~500k docs (this script)
    2. train_quality_classifier.py — train fasttext on annotations
    3. filter_dataset.py     — apply classifier to full corpus

Each document gets scored on 5 independent axes (0-5 each, total 0-25):
    - coherence:  logical structure, reasoning quality
    - pedagogy:   educational value, clarity of explanations
    - linguistic: grammar, syntax, proper French
    - depth:      substance, examples, developed arguments
    - factuality: reliable, verifiable information
    - code_quality: (code only, replaces linguistic) clean, documented, idiomatic

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

SCORING_PROMPT = """Tu es un évaluateur STRICT de textes francophones pour l'entraînement de modèles de langage. Tu dois être EXIGEANT : un 5 est RARE et réservé à un texte d'excellence. La plupart des textes web méritent entre 1 et 3.

Évalue l'extrait suivant sur chaque axe avec une note de 0 à 5 :

**coherence** — Structure logique et raisonnement
  0 = spam, code HTML, fragments sans sens, texte généré
  1 = compréhensible mais décousu, pas de fil conducteur
  2 = suit un sujet mais mal organisé, transitions abruptes
  3 = structure correcte, le lecteur peut suivre sans effort
  4 = bien organisé, progression logique claire, bon plan
  5 = raisonnement rigoureux, structure exemplaire (niveau article académique)

**pedagogy** — Valeur pédagogique et clarté des explications
  0 = aucune (pub, navigation, metadata, listings, opinions sans fond)
  1 = mentionne des faits mais n'explique rien
  2 = informatif mais pas pédagogique, le lecteur n'apprend pas vraiment
  3 = explique un sujet de manière compréhensible
  4 = pédagogie claire avec exemples, pourrait servir dans un cours
  5 = excellent matériel éducatif (niveau manuel scolaire ou universitaire)

**linguistic** — Qualité du français
  0 = illisible, langue étrangère, bouillie de caractères
  1 = français cassé, fautes majeures à chaque phrase
  2 = compréhensible mais maladroit, registre familier, anglicismes
  3 = français correct, quelques maladresses mineures
  4 = bien écrit, syntaxe maîtrisée, vocabulaire précis
  5 = français exemplaire, prose soignée (niveau littéraire ou journalistique)

**depth** — Profondeur et substance
  0 = vide (une phrase, titre seul, liste de liens, metadata)
  1 = superficiel, survole le sujet en quelques lignes
  2 = aborde le sujet mais reste en surface, pas de détails
  3 = traitement correct avec quelques détails ou exemples
  4 = analyse approfondie, arguments développés, nuances
  5 = traitement exhaustif, multiple angles, niveau expert

**factuality** — Fiabilité factuelle
  0 = faux, désinformation, conspirationniste, dangereux
  1 = très douteux, affirmations non vérifiables, biais forts
  2 = mélange vrai/faux, opinions présentées comme faits
  3 = globalement plausible mais sans sources
  4 = fiable, informations vérifiables, peu d'erreurs
  5 = rigoureux, sourcé ou vérifiable, niveau encyclopédique

EXEMPLES DE CALIBRATION :

Texte : "Accueil > Nos produits > Mentions légales > Contact | © 2024 Tous droits réservés"
→ {{"coherence": 0, "pedagogy": 0, "linguistic": 1, "depth": 0, "factuality": 0}}

Texte : "La polémique entourant les covers du second mini-album des Girls' Generation aura finalement obligé SM à repousser sa sortie."
→ {{"coherence": 2, "pedagogy": 0, "linguistic": 3, "depth": 1, "factuality": 2}}

Texte : "La photosynthèse est le processus par lequel les plantes convertissent le CO2 et l'eau en glucose et en oxygène, en utilisant l'énergie lumineuse captée par la chlorophylle. Ce mécanisme se déroule en deux phases : les réactions lumineuses dans les thylakoïdes, puis le cycle de Calvin dans le stroma."
→ {{"coherence": 5, "pedagogy": 5, "linguistic": 5, "depth": 4, "factuality": 5}}

Maintenant évalue cet extrait :
---
{text}
---

Réponds UNIQUEMENT avec un JSON, rien d'autre :
{{"coherence": N, "pedagogy": N, "linguistic": N, "depth": N, "factuality": N}}"""


SCORING_PROMPT_CODE = """Tu es un évaluateur STRICT de code source pour l'entraînement de modèles de langage. Tu dois être EXIGEANT : un 5 est RARE. La plupart du code sur GitHub mérite entre 1 et 3.

Évalue l'extrait suivant sur chaque axe avec une note de 0 à 5 :

**coherence** — Structure logique du code
  0 = fragments incomplets, code cassé, pas de logique visible
  1 = code qui existe mais sans structure claire
  2 = fonctionne probablement mais mal organisé
  3 = structure correcte, fonctions séparées, flux logique clair
  4 = bien architecturé, séparation des responsabilités
  5 = architecture exemplaire, patterns clairs, code maintenable

**code_quality** — Propreté et bonnes pratiques
  0 = illisible, noms de variables obscurs (a, b, x1), code spaghetti
  1 = lisible mais ne suit aucune convention
  2 = acceptable, quelques bonnes pratiques
  3 = propre, nommage clair, suit les conventions du langage
  4 = idiomatique, gestion d'erreurs, types quand approprié
  5 = exemplaire, pourrait être dans la documentation officielle du langage

**pedagogy** — Valeur éducative du code
  0 = aucune (config auto-générée, boilerplate, fichiers vides, minifié)
  1 = code trivial sans intérêt pédagogique
  2 = montre un pattern mais sans explication
  3 = code utile pour apprendre, logique claire
  4 = bien commenté, bon exemple d'un concept ou pattern
  5 = tutoriel quality, code instructif avec commentaires pédagogiques

**depth** — Complexité pertinente
  0 = trivial (hello world, imports seuls, constantes)
  1 = très simple (getter/setter, CRUD basique)
  2 = résout un problème simple
  3 = logique métier réelle, algorithme non trivial
  4 = système complexe, design patterns, optimisations
  5 = algorithme avancé, architecture système complète

**factuality** — Fiabilité technique
  0 = bugs évidents, anti-patterns dangereux, vulnérabilités
  1 = code douteux, race conditions, injections possibles
  2 = fonctionne mais pratiques discutables
  3 = correct, pas de bug évident
  4 = robuste, gestion d'erreurs, code défensif
  5 = production-ready, sécurisé, testé

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
                if isinstance(val, (int, float)) and 0 <= val <= 5:
                    scores[dim] = int(val)
                else:
                    return None
            # Compute aggregate (sum of all dims, max = 25)
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

    # Resume support — never silently overwrite existing annotations
    n_done = 0
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            n_done = sum(1 for _ in f)
        if n_done > 0 and not args.resume:
            print(f"ERROR: {output_path} already has {n_done:,} annotations.")
            print(f"Use --resume to continue, or delete the file to restart.")
            sys.exit(1)
        if args.resume:
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

    print(f"\nDimension averages (0-5 scale):")
    for dim in ["coherence", "pedagogy", "linguistic", "depth", "factuality", "code_quality"]:
        if dim in dim_counts and dim_counts[dim] > 0:
            avg = dim_sums[dim] / dim_counts[dim]
            bar = "█" * int(avg * 8)
            print(f"  {dim:15s}: {avg:.3f}/5.0 {bar}")

    print(f"\nSaved: {output_path}")
    print(f"\nEach record contains: text, doc_id, coherence, pedagogy, linguistic|code_quality, depth, factuality, total")
    print(f"Total score range: 0-25 (sum of 5 dimensions)")


if __name__ == "__main__":
    main()
