import sys
from pathlib import Path
import numpy as np
import polars as pl

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]
ds = open_dataset(train_dir / video_stem, require_tracks=True)
gt_graph = ds.tracks

er = evaluate(pred_graph, gt_graph, scale=ds.scale)

gt_node_df = gt_graph.node_attrs()
pred_node_df = pred_graph.node_attrs()

gt_to_pred = {}
for row in pred_node_df.iter_rows(named=True):
    p_id, m_gt = row[td.DEFAULT_ATTR_KEYS.NODE_ID], row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    if m_gt >= 0:
        gt_to_pred[int(m_gt)] = int(p_id)

term_dists = []
swap_dists = []

for gt_u in gt_graph.node_ids():
    for gt_v in gt_graph.successors(gt_u):
        pred_u = gt_to_pred.get(gt_u)
        pred_v = gt_to_pred.get(gt_v)
        if pred_u is not None and pred_v is not None:
            pred_children = pred_graph.successors(pred_u)
            if pred_v not in pred_children:
                u_row = gt_node_df.filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == gt_u)
                v_row = gt_node_df.filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == gt_v)
                u_um = np.array([u_row["z"][0]*ds.scale[0], u_row["y"][0]*ds.scale[1], u_row["x"][0]*ds.scale[2]])
                v_um = np.array([v_row["z"][0]*ds.scale[0], v_row["y"][0]*ds.scale[1], v_row["x"][0]*ds.scale[2]])
                d = float(np.linalg.norm(v_um - u_um))
                if len(pred_children) == 0:
                    term_dists.append(d)
                else:
                    swap_dists.append(d)

print("="*60)
print("TERMINATED EDGES (Track stopped even though next cell existed):")
print(f"  Count: {len(term_dists)}")
print(f"  Physical jump distances (um): min={min(term_dists):.2f}, median={np.median(term_dists):.2f}, mean={np.mean(term_dists):.2f}, max={max(term_dists):.2f}")
print("\nSWAPPED / CROSSOVER EDGES (Track jumped to wrong neighbor):")
print(f"  Count: {len(swap_dists)}")
print(f"  Physical jump distances (um): min={min(swap_dists):.2f}, median={np.median(swap_dists):.2f}, mean={np.mean(swap_dists):.2f}, max={max(swap_dists):.2f}")
print("="*60)
