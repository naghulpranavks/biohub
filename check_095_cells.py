import json
from pathlib import Path

p = Path("D:/Downloads_2/biohub-0-95.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [1, 2, 3, 4, 6]:
    src = "".join(nb["cells"][i].get("source", []))
    print(f"================ CELL {i} ================")
    print(src[:1500])
