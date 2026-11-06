import json
from pathlib import Path

f_final = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
with open(f_final, "r", encoding="utf-8") as fp: nb_f = json.load(fp)

src = "".join(nb_f["cells"][0]["source"])

# Find filter_output_graph definition
idx = src.find("def filter_output_graph(")
if idx != -1:
    print(src[idx:idx+2500])
