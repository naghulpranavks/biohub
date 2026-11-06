import json
from pathlib import Path
import difflib

f1 = Path(r"D:\Downloads_2\biohub-x138.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-cell-tracking.ipynb")

with open(f1, "r", encoding="utf-8") as fp: nb1 = json.load(fp)
with open(f2, "r", encoding="utf-8") as fp: nb2 = json.load(fp)

def get_code_tokens(nb):
    lines = []
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            for l in "".join(c["source"]).splitlines():
                if l.strip():
                    lines.append(l.strip())
    return lines

lines1 = get_code_tokens(nb1)
lines2 = get_code_tokens(nb2)

print(f"Non-empty lines: x138={len(lines1)} | cell_tracking={len(lines2)}")

diff = list(difflib.unified_diff(lines1, lines2, fromfile="x138", tofile="cell_tracking"))
print(f"Differences in stripped code: {len(diff)}")
for line in diff:
    print(line)
