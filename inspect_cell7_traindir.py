import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

src7 = "".join(nb['cells'][7]['source'])
print("="*80)
print("CELL 7 SOURCE (TRAIN_DIR & VALIDATOR):")
print("="*80)
print(src7[:2000])
