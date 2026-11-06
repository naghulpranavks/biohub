import json
from pathlib import Path

p_v2 = Path(r"D:\Downloads_2\biohub-final-notebook-v2.ipynb")
with open(p_v2, "r", encoding="utf-8") as f:
    nb_v2 = json.load(f)

# Check cell 0 of v2
src_v2 = "".join(nb_v2["cells"][0]["source"])
print("="*60)
print("CELL 0 OF FAILED V2:")
print("="*60)
for l in src_v2.splitlines()[:30]:
    if any(k in l for k in ["WEIGHT", "VETO", "VELOCITY", "RETENTION"]):
        print(l)
