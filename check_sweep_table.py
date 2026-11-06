import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

out10 = "".join(nb['cells'][10]['outputs'][0]['text'])
idx = out10.find("SWEEP TABLE")
if idx != -1:
    print(out10[idx:idx+2500])
