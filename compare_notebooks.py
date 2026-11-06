import json
from pathlib import Path

f1 = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f2 = Path(r"D:\Downloads_2\biohub-geometric-fusion.ipynb")

with open(f1, "r", encoding="utf-8") as f:
    nb1 = json.load(f)

with open(f2, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

src1 = "".join(nb1["cells"][0]["source"])
src2 = ""
for c in nb2["cells"]:
    src2 += "".join(c["source"]) + "\n"

import re
def get_env_vars(text):
    return dict(re.findall(r'os\.environ\["([^"]+)"\]\s*=\s*["\']([^"\']+)["\']', text))

env1 = get_env_vars(src1)
env2 = get_env_vars(src2)

print("="*65)
print("ENVIRONMENT VARIABLE COMPARISON:")
print("="*65)
all_keys = sorted(set(env1.keys()) | set(env2.keys()))
for k in all_keys:
    v1 = env1.get(k, "<MISSING>")
    v2 = env2.get(k, "<MISSING>")
    if v1 != v2:
        print(f"  {k:35s}: Final={v1:15s} | GeomFusion={v2:15s}")

print("="*65)
