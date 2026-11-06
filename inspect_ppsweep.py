import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

src10 = "".join(nb['cells'][10]['source'])
print("="*80)
print("CELL 10 SOURCE (PPSWEEP CANDIDATES):")
print("="*80)
print(src10[:3000])
