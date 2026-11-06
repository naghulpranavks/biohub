import json
from pathlib import Path

f_final = Path(r"D:\Downloads_2\biohub-final-notebook.ipynb")
f_x138 = Path(r"D:\Downloads_2\biohub-x138.ipynb")

with open(f_final, "r", encoding="utf-8") as fp: nb_f = json.load(fp)
with open(f_x138, "r", encoding="utf-8") as fp: nb_x = json.load(fp)

src_f = "".join(nb_f["cells"][0]["source"])
src_x = "\n".join("".join(c["source"]) for c in nb_x["cells"] if c.get("cell_type") == "code")

# Check which features exist in x138 vs final
features = [
    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA",
    "BIOHUB_DEEPCENTER_TTA",
    "BIOHUB_MOTION_RELINK_FLOW_MODE",
    "_flow_predictor",
    "readmit_discarded_detections",
    "fill_gaps_from_low_detections",
    "prune_weak_leaf_nodes",
    "BIOHUB_ILP_APPEARANCE_WEIGHT",
    "BIOHUB_DET_THRESHOLD",
]

print("="*60)
print("FEATURE AUDIT: OUR MODEL (0.947) vs X138 (0.953)")
print("="*60)
for feat in features:
    in_f = feat in src_f
    in_x = feat in src_x
    print(f"  {feat:35s}: In Our 0.947 = {str(in_f):5s} | In x138 (0.953) = {str(in_x):5s}")
