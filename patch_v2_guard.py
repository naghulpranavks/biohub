import json
from pathlib import Path

path = Path(r"D:\Downloads_2\biohub-final-notebook-v2.ipynb")
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

source = nb["cells"][0]["source"]
replacements = 0

for i, line in enumerate(source):
    if '_bidirectional_weight_guard, 0.15' in line:
        source[i] = line.replace('0.15', '0.40')
        replacements += 1
        print(f"Patched line {i}: {source[i].strip()}")
    if '"expected_bidirectional_weight": 0.15' in line:
        source[i] = line.replace('0.15', '0.40')
        replacements += 1
        print(f"Patched line {i}: {source[i].strip()}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print(f"Total replacements made: {replacements}")
