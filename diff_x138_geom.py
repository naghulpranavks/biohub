import json
from pathlib import Path
import difflib

f_final = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f_geom = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
f_x138 = Path(r"D:\Downloads_2\biohub-x138.ipynb")

with open(f_final, "r", encoding="utf-8") as f: nb_final = json.load(f)
with open(f_geom, "r", encoding="utf-8") as f: nb_geom = json.load(f)
with open(f_x138, "r", encoding="utf-8") as f: nb_x138 = json.load(f)

def get_code_lines(nb):
    lines = []
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            lines.extend("".join(c["source"]).splitlines(keepends=True))
    return lines

lines_final = get_code_lines(nb_final)
lines_geom = get_code_lines(nb_geom)
lines_x138 = get_code_lines(nb_x138)

print(f"Lines of code: Final={len(lines_final)} | GeomFusion={len(lines_geom)} | x138={len(lines_x138)}")

# Compare x138 vs GeomFusion
diff_geom_x138 = list(difflib.unified_diff(lines_geom, lines_x138, fromfile="GeomFusion", tofile="x138", n=2))
chunks_gx = []
curr = []
for l in diff_geom_x138:
    if l.startswith("@@"):
        if curr: chunks_gx.append("".join(curr))
        curr = []
    curr.append(l)
if curr: chunks_gx.append("".join(curr))

print(f"\nTotal diff chunks between GeomFusion and x138: {len(chunks_gx)}")
for i, c in enumerate(chunks_gx):
    added = [l[1:].strip() for l in c.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in c.splitlines() if l.startswith("-") and not l.startswith("---")]
    summary_add = "; ".join(added[:2]) if added else ""
    summary_rem = "; ".join(removed[:2]) if removed else ""
    print(f"Chunk {i+1:2d}: -({len(removed)} lines) +({len(added)} lines) | Rem: {summary_rem[:50]} | Add: {summary_add[:50]}")
