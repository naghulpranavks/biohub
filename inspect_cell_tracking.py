import json
from pathlib import Path
import re

f = Path(r"D:\Downloads_2\biohub-cell-tracking.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

print(f"Total cells in biohub-cell-tracking.ipynb: {len(nb['cells'])}")

for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    lines = [l for l in src.splitlines() if l.strip()]
    first_line = lines[0] if lines else "<EMPTY>"
    print(f"Cell {i:2d} ({c.get('cell_type'):5s}, {len(lines):4d} non-empty lines): {first_line[:80]}")

code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
env_vars = dict(re.findall(r'os\.environ\["([^"]+)"\]\s*=\s*["\']([^"\']+)["\']', code))
print("\n--- Environment Variables in biohub-cell-tracking.ipynb ---")
for k, v in sorted(env_vars.items()):
    print(f"  {k:40s} = {v}")

# Check any outputs of cells
print("\n--- Outputs of Cells ---")
for i, c in enumerate(nb['cells']):
    outputs = c.get('outputs', [])
    if outputs:
        print(f"Cell {i} has {len(outputs)} output(s)")
        for out in outputs[:2]:
            t = "".join(out.get('text', []))
            if t:
                print("  Sample:", t[:200].replace("\n", " "))
