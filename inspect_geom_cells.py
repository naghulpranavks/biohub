import json
from pathlib import Path
import re

f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

for i, c in enumerate(nb2['cells']):
    src = "".join(c.get("source", []))
    print(f"==================================================")
    print(f"CELL {i} (Lines: {len(src.splitlines())})")
    print(f"==================================================")
    # Check for paths, weights, models, datasets
    for line in src.splitlines():
        if any(w in line for w in ["/kaggle/input", "WEIGHT", "CKPT", "MODEL", "CHECKPOINT", "PP_CANDIDATES", "def prune_weak_leaf_nodes", "def division_recoverability", "def _prefix_guard_ok"]):
            print("  ", line[:100])
