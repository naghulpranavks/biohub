import sys
from pathlib import Path
import numpy as np
import polars as pl

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate, per_sample_metrics, node_recall
from geff import GeffMetadata

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

ds = open_dataset(train_dir / video_stem, load_image=False, require_tracks=True)
gt_graph = ds.tracks

# Load pred
pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]

# Check baseline
er_base = evaluate(pred_graph, gt_graph, scale=ds.scale)
rec_base = node_recall(pred_graph, gt_graph)
m_base = per_sample_metrics(er_base, 58672.0, rec_base)

print(f"Base on {video_stem}:")
print(f"  Adj Jaccard: {m_base['adj_edge_jaccard']:.5f}")
print(f"  Divisions:   {er_base.division_tp}/{er_base.division_fp}/{er_base.division_fn}")

# Check Daughter 2 distance to parent 31918
parent_row = pred_graph.node_attrs().filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == 31918).to_dicts()[0]
d2_row = pred_graph.node_attrs().filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == 31969).to_dicts()[0]

p_coord = np.array([parent_row["z"]*ds.scale[0], parent_row["y"]*ds.scale[1], parent_row["x"]*ds.scale[2]])
d2_coord = np.array([d2_row["z"]*ds.scale[0], d2_row["y"]*ds.scale[1], d2_row["x"]*ds.scale[2]])
dist = np.linalg.norm(d2_coord - p_coord)

print(f"\nDistance between Parent (31918) and Daughter 2 (31969): {dist:.3f} um")
print(f"Current baseline limit (SAFE_DIV_MAX_UM): 9.0 um -> REJECTED (9.71 um > 9.0 um)!")
print(f"Widened limit (SAFE_DIV_MAX_UM):        11.0 um -> ADMITTED (9.71 um <= 11.0 um)!")
