import json
from pathlib import Path

p = Path("D:/Downloads_2/biohub-0-95.ipynb")
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print("Cells in biohub-0-95.ipynb:", len(nb["cells"]))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        first_line = [l for l in src.splitlines() if l.strip()][:1]
        print(f"Cell {i} ({c.get('cell_type')}): {first_line}")
