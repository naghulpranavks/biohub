import json
import re
from pathlib import Path

f_final = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f_x138 = Path(r"D:\Downloads_2\biohub-x138.ipynb")

with open(f_final, "r", encoding="utf-8") as fp: nb_f = json.load(fp)
with open(f_x138, "r", encoding="utf-8") as fp: nb_x = json.load(fp)

src_f = "".join(nb_f["cells"][0]["source"])
src_x = "\n".join("".join(c["source"]) for c in nb_x["cells"] if c.get("cell_type") == "code")

def get_env_vars(text):
    return dict(re.findall(r'os\.environ\["([^"]+)"\]\s*=\s*["\']([^"\']+)["\']', text))

env_f = get_env_vars(src_f)
env_x = get_env_vars(src_x)

all_keys = sorted(set(env_f.keys()) | set(env_x.keys()))
print("="*75)
print(f"{'VARIABLE':38s} | {'OUR 0.947':15s} | {'X138 (0.953)':15s}")
print("="*75)
for k in all_keys:
    vf = env_f.get(k, "<NONE>")
    vx = env_x.get(k, "<NONE>")
    if vf != vx:
        print(f"{k:38s} | {vf:15s} | {vx:15s}")
