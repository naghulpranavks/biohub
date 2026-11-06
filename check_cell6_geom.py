import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

out6 = "".join(nb['cells'][6]['outputs'][0]['text'])
print("="*80)
print("CELL 6 OUTPUT (DIAGNOSTICS & RETENTION FALLBACKS):")
print("="*80)
print(out6)
