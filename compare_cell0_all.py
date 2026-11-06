import json
import difflib
from pathlib import Path

f_final = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f_x138 = Path(r"D:\Downloads_2\biohub-x138.ipynb")
f_geom = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")

with open(f_final, "r", encoding="utf-8") as f: nb_final = json.load(f)
with open(f_x138, "r", encoding="utf-8") as f: nb_x138 = json.load(f)
with open(f_geom, "r", encoding="utf-8") as f: nb_geom = json.load(f)

# Extract code from cell 0 of each
print("="*60)
print("CELL 0 OF FINAL:")
print("".join(nb_final["cells"][0]["source"])[:1200])

print("="*60)
print("CELL 0 OF X138:")
print("".join(nb_x138["cells"][0]["source"])[:1200])

print("="*60)
print("CELL 0 OF GEOM FUSION:")
print("".join(nb_geom["cells"][0]["source"])[:1200])
