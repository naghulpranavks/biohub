import json
from pathlib import Path

f_x = Path(r"D:\Downloads_2\biohub-x138.ipynb")
f_f = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")

with open(f_x, "r", encoding="utf-8") as fp: nb_x = json.load(fp)
with open(f_f, "r", encoding="utf-8") as fp: nb_f = json.load(fp)

print("="*60)
print("X138 METADATA:")
print("="*60)
print(nb_x.get("metadata", {}))

print("="*60)
print("FINAL NOTEBOOK METADATA:")
print("="*60)
print(nb_f.get("metadata", {}))

# Check Cell 6 of both to see experiment provenance
print("="*60)
print("X138 PROVENANCE (CELL 6):")
print("="*60)
for out in nb_x["cells"][6].get("outputs", []):
    t = "".join(out.get("text", []))
    for l in t.splitlines():
        if any(k in l for k in ["source_kernel", "experiment", "attribution"]):
            print(l)

print("="*60)
print("FINAL NOTEBOOK PROVENANCE:")
print("="*60)
src_f = "".join(nb_f["cells"][0]["source"])
for l in src_f.splitlines():
    if any(k in l for k in ["source_kernel", "experiment", "attribution"]):
        print(l)
