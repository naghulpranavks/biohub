import json
from pathlib import Path

f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f2, "r", encoding="utf-8") as f: nb2 = json.load(f)

for idx in [6, 10, 11]:
    c = nb2['cells'][idx]
    print("="*80)
    print(f"OUTPUT OF CELL {idx}")
    print("="*80)
    for out in c.get('outputs', []):
        text = "".join(out.get('text', []))
        print(text)
