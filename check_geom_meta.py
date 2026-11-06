import json
from pathlib import Path

f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f2, "r", encoding="utf-8") as f: nb2 = json.load(f)

print(f"Total cells: {len(nb2['cells'])}")
for i, c in enumerate(nb2['cells']):
    ct = c.get('cell_type')
    if ct == 'markdown':
        print(f"Cell {i} (markdown):")
        print("".join(c['source']))
    elif ct == 'code':
        # Check outputs if saved in ipynb
        outputs = c.get('outputs', [])
        if outputs:
            print(f"Cell {i} has {len(outputs)} outputs!")
            for out in outputs[:2]:
                text = "".join(out.get('text', []))
                if text:
                    print("Output sample:", text[:300])
