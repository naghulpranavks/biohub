import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

print("="*80)
print("CELL 0 OF biohub-x138.ipynb")
print("="*80)
print("".join(nb['cells'][0]['source']))
