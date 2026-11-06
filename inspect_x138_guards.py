import json
from pathlib import Path

f_x = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f_x, "r", encoding="utf-8") as fp:
    nb_x = json.load(fp)

src = "\n".join("".join(c["source"]) for c in nb_x["cells"] if c.get("cell_type") == "code")

# Check all assertions and guards in x138
guards = []
for line in src.splitlines():
    if "assert " in line or "_guard" in line or "drift" in line:
        guards.append(line.strip())

print(f"Total guards/asserts in x138: {len(guards)}")
for g in guards[:15]:
    print(" ", g)
