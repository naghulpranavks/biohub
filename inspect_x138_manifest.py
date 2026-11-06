import json
from pathlib import Path

f_x = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f_x, "r", encoding="utf-8") as fp:
    nb_x = json.load(fp)

print("="*60)
print("X138 ENVIRONMENT VARIABLES & SETTINGS:")
print("="*60)
src0 = "".join(nb_x["cells"][0]["source"])
for line in src0.splitlines():
    if "BIOHUB_" in line:
        print(line.strip())

print("\n" + "="*60)
print("X138 CELL 11 (PIPELINE MANIFEST OUTPUT):")
print("="*60)
for out in nb_x["cells"][11].get("outputs", []):
    print("".join(out.get("text", [])))
