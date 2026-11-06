import json
from pathlib import Path

f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

for idx in [7, 8, 9]:
    c = nb2['cells'][idx]
    print("="*80)
    print(f"CELL {idx}")
    print("="*80)
    print("".join(c.get("source", []))[:3000])
