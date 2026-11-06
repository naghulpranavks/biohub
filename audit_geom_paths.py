import json
from pathlib import Path
import re

f = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

# Find all /kaggle/input paths
kaggle_paths = set(re.findall(r'[\'"](/kaggle/input/[^\'"]+)[\'"]', src))
print("Required /kaggle/input paths:")
for p in sorted(kaggle_paths):
    print(" ", p)

# Find all assert statements
asserts = [line.strip() for line in src.splitlines() if line.strip().startswith("assert ")]
print(f"\nTotal assertions found: {len(asserts)}")
for a in asserts[:10]:
    print(" ", a[:90])
