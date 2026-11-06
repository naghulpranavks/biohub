import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

for i, out in enumerate(nb['cells'][7]['outputs']):
    text = "".join(out.get('text', []))
    print(f"--- OUTPUT {i} ({len(text.splitlines())} lines) ---")
    for l in text.splitlines():
        if any(k in l.lower() for k in ["completed in", "elapsed", "runtime", "validator", "merged", "shards"]):
            print(" ", l[:100])
