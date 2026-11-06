import json
from pathlib import Path
import re

f = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

print(f"Total cells in biohub-x138.ipynb: {len(nb['cells'])}")

for i, c in enumerate(nb['cells']):
    src = "".join(c.get("source", []))
    lines = [l for l in src.splitlines() if l.strip()]
    first_line = lines[0] if lines else "<EMPTY>"
    print(f"Cell {i:2d} ({c.get('cell_type'):5s}, {len(lines):4d} non-empty lines): {first_line[:80]}")

code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
env_vars = dict(re.findall(r'os\.environ\["([^"]+)"\]\s*=\s*["\']([^"\']+)["\']', code))
print("\n--- Environment Variables in biohub-x138.ipynb ---")
for k, v in sorted(env_vars.items()):
    print(f"  {k:40s} = {v}")

# Check any score or LB mentions
print("\n--- Mentions of Score / LB / Models ---")
for line in code.splitlines():
    if any(k in line.lower() for k in ["score", "0.9", "lb", "axis", "preset", "checkpoint", "weights"]):
        if len(line.strip()) > 0 and not line.strip().startswith("#"):
            print("  ", line.strip()[:100])
        elif any(k in line.lower() for k in ["0.9", "lb", "axis", "preset"]):
            print("  [#]", line.strip()[:100])
