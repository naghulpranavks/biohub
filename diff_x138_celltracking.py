import json
from pathlib import Path
import difflib

f1 = Path(r"D:\Downloads_2\biohub-x138.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-cell-tracking.ipynb")

with open(f1, "r", encoding="utf-8") as fp: nb1 = json.load(fp)
with open(f2, "r", encoding="utf-8") as fp: nb2 = json.load(fp)

def get_code_lines(nb):
    lines = []
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            lines.extend("".join(c["source"]).splitlines(keepends=True))
    return lines

lines1 = get_code_lines(nb1)
lines2 = get_code_lines(nb2)

print(f"Lines of code: x138={len(lines1)} | cell_tracking={len(lines2)}")

diff = list(difflib.unified_diff(lines1, lines2, fromfile="x138", tofile="cell_tracking", n=2))
print(f"Diff lines count: {len(diff)}")
for line in diff[:50]:
    print(line, end="")
