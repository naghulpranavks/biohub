import json
from pathlib import Path

f1 = Path(r"D:\Downloads_2\biohub-x138.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-cell-tracking.ipynb")

with open(f1, "r", encoding="utf-8") as fp: nb1 = json.load(fp)
with open(f2, "r", encoding="utf-8") as fp: nb2 = json.load(fp)

out1 = "".join(nb1['cells'][6]['outputs'][0]['text'])
out2 = "".join(nb2['cells'][6]['outputs'][0]['text'])

print("Output of Cell 6 match exactly:", out1 == out2)

# Check all cell outputs
outputs_match = True
for i in range(min(len(nb1['cells']), len(nb2['cells']))):
    o1 = nb1['cells'][i].get('outputs', [])
    o2 = nb2['cells'][i].get('outputs', [])
    if len(o1) != len(o2):
        print(f"Cell {i} outputs count mismatch: {len(o1)} vs {len(o2)}")
        outputs_match = False
    else:
        for j in range(len(o1)):
            t1 = "".join(o1[j].get('text', []))
            t2 = "".join(o2[j].get('text', []))
            if t1 != t2:
                print(f"Cell {i} output {j} content mismatch!")
                outputs_match = False

import difflib

out1_4 = "".join(nb1['cells'][4]['outputs'][4].get('text', [])).splitlines(keepends=True)
out2_4 = "".join(nb2['cells'][4]['outputs'][4].get('text', [])).splitlines(keepends=True)
print("=== Unified diff of Cell 4 Output 4 ===")
for l in list(difflib.unified_diff(out1_4, out2_4, fromfile="x138", tofile="cell_tracking"))[:20]:
    print(l, end="")

out1_5 = "".join(nb1['cells'][5]['outputs'][0].get('text', [])).splitlines(keepends=True)
out2_5 = "".join(nb2['cells'][5]['outputs'][0].get('text', [])).splitlines(keepends=True)
print("\n=== Unified diff of Cell 5 Output 0 ===")
for l in list(difflib.unified_diff(out1_5, out2_5, fromfile="x138", tofile="cell_tracking"))[:20]:
    print(l, end="")
