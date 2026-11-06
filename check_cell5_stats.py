import json
from pathlib import Path

f_x138 = Path(r"D:\Downloads_2\biohub-x138.ipynb")
f_geom = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")

with open(f_x138, "r", encoding="utf-8") as fp: nb_x = json.load(fp)
with open(f_geom, "r", encoding="utf-8") as fp: nb_g = json.load(fp)

# Let's inspect Cell 5 outputs of both
print("="*60)
print("X138 CELL 5 OUTPUT SUMMARY:")
print("="*60)
for out in nb_x['cells'][5].get('outputs', []):
    for line in "".join(out.get('text', [])).splitlines():
        if any(w in line for w in ["FINAL", "readmitted", "gap filler", "low-detection", "divisions="]):
            print(" ", line[:100])

print("\n" + "="*60)
print("GEOM FUSION CELL 5 OUTPUT SUMMARY:")
print("="*60)
for out in nb_g['cells'][5].get('outputs', []):
    for line in "".join(out.get('text', [])).splitlines():
        if any(w in line for w in ["FINAL", "leaf_pruned", "safe-division", "divisions="]):
            print(" ", line[:100])
