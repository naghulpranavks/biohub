from pathlib import Path

p = Path("nb_source.py")
text = p.read_text(encoding="utf-8")
lines = [l for l in text.splitlines(keepends=True) if "from __future__ import annotations" not in l]
p.write_text("from __future__ import annotations\n" + "".join(lines), encoding="utf-8")
print("Successfully moved future annotations to top of nb_source.py")
