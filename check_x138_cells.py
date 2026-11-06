import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    lines = [l for l in src.splitlines() if l.strip()]
    first_non_empty = lines[0] if lines else "<EMPTY>"
    print(f"Cell {i:2d} ({c.get('cell_type'):5s}, {len(lines):4d} code/txt lines): {first_non_empty[:80]}")

# Check outputs of each cell
print("\n--- Outputs of Cells ---")
for i, c in enumerate(nb['cells']):
    outputs = c.get('outputs', [])
    if outputs:
        print(f"Cell {i} has {len(outputs)} output(s)")
        for out in outputs[:2]:
            t = "".join(out.get('text', []))
            if t:
                print("  Sample:", t[:200].replace("\n", " "))
