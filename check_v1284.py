import json
from pathlib import Path

f_x = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f_x, "r", encoding="utf-8") as fp: nb_x = json.load(fp)

src = "\n".join("".join(c["source"]) for c in nb_x["cells"] if c.get("cell_type") == "code")

for i, line in enumerate(src.splitlines()):
    if "v1284" in line:
        print(f"Line {i}: {line}")
        # print context
        for ctx in src.splitlines()[max(0, i-5):min(len(src.splitlines()), i+25)]:
            print("  ", ctx)
        break
