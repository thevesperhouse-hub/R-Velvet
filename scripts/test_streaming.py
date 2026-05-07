"""Quick test: can we stream from each source and get text?"""

from datasets import load_dataset

sources = [
    ("uonlp/CulturaX", "fr", "train", "text"),
    ("wikimedia/wikipedia", "20231101.fr", "train", "text"),
    ("bigcode/starcoderdata", None, "train", "content"),
]

for repo, name, split, field in sources:
    print(f"\n{'='*60}")
    print(f"Testing: {repo} (name={name}, split={split}, field={field})")
    try:
        ds = load_dataset(repo, name=name, split=split, streaming=True)
        for i, ex in enumerate(ds):
            keys = list(ex.keys())
            txt = ex.get(field) or ""
            print(f"  Keys: {keys}")
            print(f"  Field '{field}': {repr(txt[:200])}")
            if i >= 1:
                break
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\n{'='*60}")
print("Done.")
