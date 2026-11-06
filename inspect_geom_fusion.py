import json
from pathlib import Path
import re

f1 = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")

with open(f1, "r", encoding="utf-8") as f:
    nb1 = json.load(f)
with open(f2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

print(f"Final notebook cells: {len(nb1['cells'])}")
print(f"GeomFusion notebook cells: {len(nb2['cells'])}")

print("\n--- GeomFusion Cells Overview ---")
for i, c in enumerate(nb2['cells']):
    src = "".join(c.get("source", []))
    lines = [l for l in src.split("\n") if l.strip()]
    first_non_empty = lines[0] if lines else "<EMPTY>"
    print(f"Cell {i:2d} ({c.get('cell_type'):5s}, {len(lines):4d} code/txt lines): {first_non_empty[:80]}")

# Now let's compare the code between nb1 and nb2
code1 = "\n\n".join("".join(c["source"]) for c in nb1["cells"] if c.get("cell_type") == "code")
code2 = "\n\n".join("".join(c["source"]) for c in nb2["cells"] if c.get("cell_type") == "code")

# Check function definitions in both
def extract_def_names(code):
    return set(re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code))

defs1 = extract_def_names(code1)
defs2 = extract_def_names(code2)

print("\n--- Functions in GeomFusion but NOT in Final ---")
for d in sorted(defs2 - defs1):
    print("  +", d)

print("\n--- Functions in Final but NOT in GeomFusion ---")
for d in sorted(defs1 - defs2):
    print("  -", d)

# Check class definitions
def extract_class_names(code):
    return set(re.findall(r'class\s+([a-zA-Z0-9_]+)', code))

classes1 = extract_class_names(code1)
classes2 = extract_class_names(code2)

print("\n--- Classes in GeomFusion but NOT in Final ---")
for c in sorted(classes2 - classes1):
    print("  +", c)

print("\n--- Classes in Final but NOT in GeomFusion ---")
for c in sorted(classes1 - classes2):
    print("  -", c)
