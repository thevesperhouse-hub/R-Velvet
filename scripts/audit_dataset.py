"""Audit quality of filtered dataset (Vesper-Edu-FR).

Runs multiple checks:
  1. LLM re-scoring: re-score a sample with the LLM and compare with fasttext
  2. Domain diversity: top domains, concentration
  3. Boilerplate detection: cookie banners, nav menus, CGU patterns
  4. Near-duplicate detection: simple n-gram fingerprinting
  5. Text stats: length distribution, language check

Usage:
    # Full audit with LLM re-scoring (needs vLLM running):
    python scripts/audit_dataset.py --file data/vesper_edu_fr.jsonl \
        --api-url http://localhost:8000/v1 --n 200

    # Quick audit without LLM (no vLLM needed):
    python scripts/audit_dataset.py --file data/vesper_edu_fr.jsonl --no-llm --n 500
"""

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


# --- Scoring prompts (same as annotate_quality.py) ---

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

Maintenant évalue cet extrait :
---
{text}
---

Réponds UNIQUEMENT avec un JSON, rien d'autre :
{{"coherence": N, "pedagogy": N, "linguistic": N, "depth": N, "factuality": N}}"""


# --- Helpers ---

BOILERPLATE_PATTERNS = [
    r"(?i)accept.*cookies?",
    r"(?i)politique de confidentialit",
    r"(?i)mentions? l[ée]gales?",
    r"(?i)tous droits r[ée]serv[ée]s",
    r"(?i)conditions g[ée]n[ée]rales",
    r"(?i)panier.*article",
    r"(?i)ajouter au panier",
    r"(?i)votre adresse e-?mail",
    r"(?i)inscri(vez|s)-?(vous|toi).*newsletter",
    r"(?i)navigation.*menu",
    r"(?i)accueil\s*[>|/»]",
    r"(?i)©\s*\d{4}",
    r"(?i)powered by",
    r"(?i)lire la suite\.\.\.",
    r"(?i)partager sur (facebook|twitter|linkedin)",
]


def extract_domain(url):
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def ngram_fingerprint(text, n=5):
    """Create a set of word n-grams for near-duplicate detection."""
    words = text.lower().split()[:200]
    if len(words) < n:
        return set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def count_boilerplate(text):
    """Count how many boilerplate patterns match."""
    return sum(1 for p in BOILERPLATE_PATTERNS if re.search(p, text[:2000]))


def parse_scores(response_text):
    try:
        match = re.search(r'\{[^}]+\}', response_text)
        if match:
            data = json.loads(match.group())
            scores = {}
            for dim in DIMENSIONS:
                val = data.get(dim, -1)
                if isinstance(val, (int, float)) and 0 <= val <= 5:
                    scores[dim] = int(val)
                else:
                    return None
            scores["total"] = sum(scores[d] for d in DIMENSIONS)
            return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def sample_lines(path, n, total=None):
    """Sample N random lines from a JSONL file in one pass."""
    if total is None:
        total = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                total += 1

    if total <= n:
        indices = set(range(total))
    else:
        indices = set(random.sample(range(total), n))

    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i in indices:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if indices and i > max(indices):
                break
    return records, total


def llm_rescore(client, model, records, batch_size=16):
    """Re-score records with LLM, return list of score dicts."""

    def score_one(text):
        try:
            if len(text) > 1500:
                text = text[:1500] + "..."
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": SCORING_PROMPT.format(text=text)}],
                max_tokens=80,
                temperature=0.0,
            )
            reply = response.choices[0].message.content
            return parse_scores(reply)
        except Exception:
            return None

    results = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=min(len(batch), 16)) as pool:
            futures = {pool.submit(score_one, r["text"]): j
                       for j, r in enumerate(batch)}
            batch_results = [None] * len(batch)
            for future in as_completed(futures):
                idx = futures[future]
                batch_results[idx] = future.result()
            results.extend(batch_results)
        print(f"  Re-scored {min(i + batch_size, len(records))}/{len(records)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Audit filtered dataset quality")
    parser.add_argument("--file", required=True, help="Filtered JSONL file")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of docs to sample for audit")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM re-scoring (no vLLM needed)")
    parser.add_argument("--api-url", default="http://localhost:8000/v1",
                        help="vLLM API URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ",
                        help="Model name on vLLM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    size_mb = os.path.getsize(args.file) / 1e6

    print(f"{'='*60}")
    print(f"DATASET AUDIT: {args.file}")
    print(f"File size: {size_mb:,.0f} MB")
    print(f"Sample size: {args.n}")
    print(f"{'='*60}")

    # --- Sample ---
    print(f"\nSampling {args.n} random docs...")
    records, total = sample_lines(args.file, args.n)
    print(f"Total docs in file: {total:,}")
    print(f"Sampled: {len(records)}")

    # ==========================================
    # CHECK 1: Text stats
    # ==========================================
    print(f"\n{'='*60}")
    print("CHECK 1: Text statistics")
    print(f"{'='*60}")

    lengths = [len(r.get("text", "")) for r in records]
    lengths.sort()
    avg_len = sum(lengths) / len(lengths)
    median_len = lengths[len(lengths) // 2]
    p10 = lengths[int(len(lengths) * 0.1)]
    p90 = lengths[int(len(lengths) * 0.9)]

    print(f"  Avg length:    {avg_len:,.0f} chars")
    print(f"  Median length: {median_len:,} chars")
    print(f"  P10-P90:       {p10:,} - {p90:,} chars")
    print(f"  Min:           {lengths[0]:,} chars")
    print(f"  Max:           {lengths[-1]:,} chars")

    # Length buckets
    buckets = {"<500": 0, "500-2k": 0, "2k-5k": 0, "5k-10k": 0, "10k+": 0}
    for l in lengths:
        if l < 500: buckets["<500"] += 1
        elif l < 2000: buckets["500-2k"] += 1
        elif l < 5000: buckets["2k-5k"] += 1
        elif l < 10000: buckets["5k-10k"] += 1
        else: buckets["10k+"] += 1

    print(f"\n  Length distribution:")
    for bucket, count in buckets.items():
        pct = count / len(records) * 100
        bar = "█" * int(pct / 2)
        print(f"    {bucket:8s}: {count:4d} ({pct:4.1f}%) {bar}")

    # ==========================================
    # CHECK 2: Domain diversity
    # ==========================================
    print(f"\n{'='*60}")
    print("CHECK 2: Domain diversity")
    print(f"{'='*60}")

    domains = [extract_domain(r.get("url", "")) for r in records]
    domain_counts = collections.Counter(domains)
    n_unique = len(domain_counts)
    top_20 = domain_counts.most_common(20)

    print(f"  Unique domains in sample: {n_unique}")
    print(f"  Top 20:")
    for domain, count in top_20:
        pct = count / len(records) * 100
        print(f"    {domain:40s}: {count:3d} ({pct:.1f}%)")

    # Concentration: top-10 domains share
    top10_share = sum(c for _, c in domain_counts.most_common(10)) / len(records) * 100
    print(f"\n  Top-10 concentration: {top10_share:.1f}%")
    if top10_share > 50:
        print(f"  ⚠ High concentration — dataset may lack diversity")
    else:
        print(f"  OK — reasonable diversity")

    # ==========================================
    # CHECK 3: Boilerplate detection
    # ==========================================
    print(f"\n{'='*60}")
    print("CHECK 3: Boilerplate / low-quality patterns")
    print(f"{'='*60}")

    boilerplate_scores = [(count_boilerplate(r["text"]), r) for r in records]
    n_clean = sum(1 for s, _ in boilerplate_scores if s == 0)
    n_mild = sum(1 for s, _ in boilerplate_scores if s == 1)
    n_bad = sum(1 for s, _ in boilerplate_scores if s >= 2)

    print(f"  Clean (0 patterns):     {n_clean} ({n_clean/len(records)*100:.1f}%)")
    print(f"  Mild (1 pattern):       {n_mild} ({n_mild/len(records)*100:.1f}%)")
    print(f"  Suspicious (2+ patterns): {n_bad} ({n_bad/len(records)*100:.1f}%)")

    if n_bad > 0:
        print(f"\n  Worst boilerplate examples:")
        boilerplate_scores.sort(key=lambda x: -x[0])
        for score, r in boilerplate_scores[:3]:
            print(f"    [{score} patterns] {r.get('url', '')}")
            print(f"    {r['text'][:200]}")
            print()

    # ==========================================
    # CHECK 4: Near-duplicate detection
    # ==========================================
    print(f"\n{'='*60}")
    print("CHECK 4: Near-duplicate detection")
    print(f"{'='*60}")

    fingerprints = [(ngram_fingerprint(r["text"]), r) for r in records]
    dup_pairs = []
    # Compare all pairs (feasible for ~200 docs)
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            fp_i, r_i = fingerprints[i]
            fp_j, r_j = fingerprints[j]
            sim = jaccard(fp_i, fp_j)
            if sim > 0.5:
                dup_pairs.append((sim, i, j))

    if dup_pairs:
        dup_pairs.sort(key=lambda x: -x[0])
        print(f"  Found {len(dup_pairs)} near-duplicate pairs (Jaccard > 0.5)")
        for sim, i, j in dup_pairs[:3]:
            print(f"\n    Similarity: {sim:.2f}")
            print(f"    A: {records[i].get('url', '')}")
            print(f"       {records[i]['text'][:150]}")
            print(f"    B: {records[j].get('url', '')}")
            print(f"       {records[j]['text'][:150]}")
    else:
        print(f"  No near-duplicates found (Jaccard > 0.5)")
        print(f"  OK — sample looks clean")

    # ==========================================
    # CHECK 5: LLM re-scoring
    # ==========================================
    if not args.no_llm:
        print(f"\n{'='*60}")
        print("CHECK 5: LLM re-scoring (comparing fasttext vs LLM)")
        print(f"{'='*60}")

        try:
            from openai import OpenAI
            client = OpenAI(base_url=args.api_url, api_key="dummy")

            # Test connection
            client.models.list()
            print(f"  Connected to {args.api_url}")
        except Exception as e:
            print(f"  Cannot connect to vLLM: {e}")
            print(f"  Skipping LLM re-scoring. Use --no-llm to skip this check.")
            args.no_llm = True

    if not args.no_llm:
        # Use a smaller sample for LLM (slower)
        llm_n = min(len(records), 100)
        llm_sample = random.sample(records, llm_n)
        print(f"  Re-scoring {llm_n} docs with {args.model}...")

        t0 = time.time()
        llm_scores = llm_rescore(client, args.model, llm_sample)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s ({llm_n/elapsed:.1f} docs/s)")

        # Compare
        valid = [(r, s) for r, s in zip(llm_sample, llm_scores) if s is not None]
        n_valid = len(valid)
        n_failed = llm_n - n_valid
        print(f"  Valid scores: {n_valid}/{llm_n} (failed to parse: {n_failed})")

        if valid:
            print(f"\n  LLM score distribution (0-5 per dimension):")
            for dim in DIMENSIONS:
                vals = [s[dim] for _, s in valid if dim in s]
                if vals:
                    avg = sum(vals) / len(vals)
                    low = sum(1 for v in vals if v <= 1)
                    mid = sum(1 for v in vals if 2 <= v <= 3)
                    high = sum(1 for v in vals if v >= 4)
                    print(f"    {dim:12s}: avg={avg:.2f}  "
                          f"low(0-1)={low}({low/len(vals)*100:.0f}%)  "
                          f"mid(2-3)={mid}({mid/len(vals)*100:.0f}%)  "
                          f"high(4-5)={high}({high/len(vals)*100:.0f}%)")

            totals = [s["total"] for _, s in valid]
            avg_total = sum(totals) / len(totals)
            print(f"\n    Total avg: {avg_total:.1f}/25")

            # Quality verdict
            pct_above_15 = sum(1 for t in totals if t >= 15) / len(totals) * 100
            pct_above_20 = sum(1 for t in totals if t >= 20) / len(totals) * 100
            pct_below_10 = sum(1 for t in totals if t < 10) / len(totals) * 100

            print(f"    Total >= 20 (excellent): {pct_above_20:.1f}%")
            print(f"    Total >= 15 (good):      {pct_above_15:.1f}%")
            print(f"    Total < 10 (poor):       {pct_below_10:.1f}%")

            if pct_below_10 > 20:
                print(f"\n  ⚠ {pct_below_10:.0f}% scored below 10/25 — fasttext may be too lenient")
            elif pct_above_15 > 60:
                print(f"\n  OK — {pct_above_15:.0f}% score 15+ on re-evaluation")
            else:
                print(f"\n  Mixed — {pct_above_15:.0f}% score 15+, review samples below")

            # Show worst docs according to LLM
            valid.sort(key=lambda x: x[1]["total"])
            print(f"\n  Worst 5 according to LLM:")
            for r, s in valid[:5]:
                scores_str = " ".join(f"{d}={s[d]}" for d in DIMENSIONS)
                print(f"    [total={s['total']}] {scores_str}")
                print(f"    url: {r.get('url', '')}")
                print(f"    {r['text'][:200]}")
                print()

            print(f"  Best 3 according to LLM:")
            for r, s in valid[-3:]:
                scores_str = " ".join(f"{d}={s[d]}" for d in DIMENSIONS)
                print(f"    [total={s['total']}] {scores_str}")
                print(f"    url: {r.get('url', '')}")
                print(f"    {r['text'][:200]}")
                print()
    else:
        print(f"\n(LLM re-scoring skipped)")

    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"  File:              {args.file}")
    print(f"  Total docs:        {total:,}")
    print(f"  Sample size:       {len(records)}")
    print(f"  Avg text length:   {avg_len:,.0f} chars")
    print(f"  Unique domains:    {n_unique} (in sample)")
    print(f"  Top-10 share:      {top10_share:.1f}%")
    print(f"  Clean (no boiler): {n_clean/len(records)*100:.1f}%")
    print(f"  Near-duplicates:   {len(dup_pairs)} pairs")
    if not args.no_llm and valid:
        print(f"  LLM avg total:     {avg_total:.1f}/25")
        print(f"  LLM good (>=15):   {pct_above_15:.1f}%")
        print(f"  LLM poor (<10):    {pct_below_10:.1f}%")


if __name__ == "__main__":
    main()
