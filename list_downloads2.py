from pathlib import Path

p = Path("D:/Downloads_2")
if p.exists():
    for f in sorted(p.iterdir()):
        if any(f.name.lower().endswith(ext) for ext in [".ipynb", ".csv", ".txt", ".json", ".pt", ".pth", ".geff"]):
            print(f"{f.name:50s} ({f.stat().st_size:,} bytes)")
