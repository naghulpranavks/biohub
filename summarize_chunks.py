import json
import difflib
from pathlib import Path

f1 = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")

with open(f1, "r", encoding="utf-8") as f: nb1 = json.load(f)
with open(f2, "r", encoding="utf-8") as f: nb2 = json.load(f)

def get_code_lines(nb):
    lines = []
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            lines.extend("".join(c["source"]).splitlines(keepends=True))
    return lines

lines1 = get_code_lines(nb1)
lines2 = get_code_lines(nb2)

diff = list(difflib.unified_diff(lines1, lines2, fromfile="Final_0947", tofile="GeomFusion", n=2))
chunks = []
curr = []
for line in diff:
    if line.startswith("@@"):
        if curr: chunks.append("".join(curr))
        curr = []
    curr.append(line)
if curr: chunks.append("".join(curr))

print(f"Total chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    # Print the added/removed lines
    added = [l[1:].strip() for l in c.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in c.splitlines() if l.startswith("-") and not l.startswith("---")]
    summary_add = "; ".join(added[:2]) if added else ""
    summary_rem = "; ".join(removed[:2]) if removed else ""
    print(f"Chunk {i+1:2d}: -({len(removed)} lines) +({len(added)} lines) | Rem: {summary_rem[:50]} | Add: {summary_add[:50]}")
