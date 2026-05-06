"""Diagnostic: test if HF streaming for FineWeb-2 + Wikipedia works in isolation.

Runs three checks, each timed:
  1. load_dataset(FineWeb-2, streaming=True) resolves shards
  2. First 3 docs from FineWeb-2 are yielded
  3. Same for Wikipedia FR

If any step hangs >60s, that's the bug. Output tells us exactly where.

Usage:
    python scripts/diag_streaming.py
"""

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass


def stage(label, fn, max_wait=120):
    t0 = time.time()
    print(f"[{0.0:6.1f}s] START {label}", flush=True)
    try:
        result = fn()
    except Exception as e:
        print(f"[{time.time()-t0:6.1f}s] CRASH {label}: {type(e).__name__}: {e}", flush=True)
        raise
    elapsed = time.time() - t0
    if elapsed > max_wait:
        print(f"[{elapsed:6.1f}s] SLOW  {label} (>{max_wait}s)", flush=True)
    else:
        print(f"[{elapsed:6.1f}s] OK    {label}", flush=True)
    return result


def main():
    print("Testing HF streaming for FineWeb-2 fra_Latn + Wikipedia FR...", flush=True)
    print()

    from datasets import load_dataset, interleave_datasets

    # === Stage 1: FineWeb-2 metadata
    fineweb = stage(
        "load_dataset(FineWeb-2 fra_Latn streaming)",
        lambda: load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="fra_Latn",
            split="train",
            streaming=True,
        ),
    )

    # === Stage 2: First 3 docs from FineWeb-2
    def pull_docs(ds, n):
        out = []
        for i, ex in enumerate(ds):
            out.append(ex.get("text", "")[:80])
            if i + 1 >= n:
                break
        return out

    docs = stage(
        "yield first 3 docs from FineWeb-2",
        lambda: pull_docs(fineweb, 3),
    )
    for i, d in enumerate(docs):
        print(f"           doc{i}: {d!r}", flush=True)

    # === Stage 3: Wikipedia FR metadata
    wiki = stage(
        "load_dataset(Wikipedia 20231101.fr streaming)",
        lambda: load_dataset(
            "wikimedia/wikipedia",
            name="20231101.fr",
            split="train",
            streaming=True,
        ),
    )

    # === Stage 4: First 3 docs from Wikipedia
    docs = stage(
        "yield first 3 docs from Wikipedia",
        lambda: pull_docs(wiki, 3),
    )
    for i, d in enumerate(docs):
        print(f"           doc{i}: {d!r}", flush=True)

    # === Stage 5: Interleave + shuffle (the heavy combo used by the trainer)
    def build_interleave():
        fw = load_dataset("HuggingFaceFW/fineweb-2", name="fra_Latn",
                          split="train", streaming=True)
        wk = load_dataset("wikimedia/wikipedia", name="20231101.fr",
                          split="train", streaming=True)
        mix = interleave_datasets(
            [fw, wk],
            probabilities=[0.8, 0.2],
            seed=0,
            stopping_strategy="all_exhausted",
        )
        return mix.shuffle(seed=0, buffer_size=1000)

    mix = stage("build interleave + shuffle(buffer_size=1000)", build_interleave)
    docs = stage("yield first 3 docs from interleaved+shuffled", lambda: pull_docs(mix, 3))
    for i, d in enumerate(docs):
        print(f"           doc{i}: {d!r}", flush=True)

    print()
    print("All stages passed. The streaming pipeline itself is OK; the hang must")
    print("be in the multi-worker DataLoader setup or the StreamingTextDataset.")


if __name__ == "__main__":
    main()
