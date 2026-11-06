import json
import difflib
from pathlib import Path

f1 = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")

with open(f1, "r", encoding="utf-8") as f:
    nb1 = json.load(f)
with open(f2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

# Concatenate all code cells with clear line numbers
def get_code_lines(nb):
    lines = []
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            lines.extend("".join(c["source"]).splitlines(keepends=True))
    return lines

lines1 = get_code_lines(nb1)
lines2 = get_code_lines(nb2)

print(f"Total code lines in Final: {len(lines1)}")
print(f"Total code lines in GeomFusion: {len(lines2)}")

diff = list(difflib.unified_diff(lines1, lines2, fromfile="Final_0947", tofile="GeomFusion", n=3))

print("\n--- Summary of Diffs ---")
diff_chunks = []
current_chunk = []
for line in diff:
    if line.startswith("@@"):
        if current_chunk:
            diff_chunks.append("".join(current_chunk))
            current_chunk = []
    current_chunk.append(line)
if current_chunk:
    diff_chunks.append("".join(current_chunk))

print(f"Total diff chunks found: {len(diff_chunks)}")
for i, chunk in enumerate(diff_chunks[15:], start=16):
    print(f"\n================ CHUNK {i} ================")
    print(chunk)
