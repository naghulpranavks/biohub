import json
from pathlib import Path

p = Path("D:/Downloads_2/notebook19b565aa38.ipynb")
if p.exists():
    with open(p, "r", encoding="utf-8") as fp:
        nb = json.load(fp)
    print(f"Cells in notebook19b565aa38.ipynb: {len(nb['cells'])}")
    src0 = "".join(nb['cells'][0].get("source", []))
    print("Cell 0:\n", src0[:300])
