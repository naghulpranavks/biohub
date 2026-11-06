import json
from pathlib import Path

f = Path(r"D:\Downloads_2\biohub-x138.ipynb")
with open(f, "r", encoding="utf-8") as fp:
    nb = json.load(fp)

# Let's inspect Cell 5 around the new functions
src = "".join(nb['cells'][5]['source'])

# Look for flow predictor, gap filler, readmission functions
for func in ["_flow_predictor", "fill_gaps_from_low_detections", "readmit_discarded_detections"]:
    idx = src.find(f"def {func}")
    if idx != -1:
        print("="*80)
        print(f"FUNCTION: {func}")
        print("="*80)
        print(src[idx:idx+2500])
