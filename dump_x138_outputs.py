import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

for idx in [4, 5, 6, 10, 11]:
    c = nb['cells'][idx]
    print("="*80)
    print(f"OUTPUT OF CELL {idx} IN x138")
    print("="*80)
    for out in c.get('outputs', []):
        t = "".join(out.get('text', []))
        print(t)
