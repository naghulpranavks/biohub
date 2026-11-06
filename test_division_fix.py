import sys
from pathlib import Path
import numpy as np
import polars as pl

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate, per_sample_metrics, node_recall

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]
ds = open_dataset(train_dir / video_stem, require_tracks=True)
gt_graph = ds.tracks

base_er = evaluate(pred_graph, gt_graph, scale=ds.scale)

# Reset and reload
pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]
pred_node_df = pred_graph.node_attrs()

nodes_t67 = pred_node_df.filter(pl.col("t") == 67)
d2_coord = np.array([74.75, 30.06, 66.62])

best_d2_id = None
min_dist = 999.0
for row in nodes_t67.iter_rows(named=True):
    p_coord = np.array([row["z"]*ds.scale[0], row["y"]*ds.scale[1], row["x"]*ds.scale[2]])
    dist = np.linalg.norm(p_coord - d2_coord)
    if dist < min_dist:
        min_dist = dist
        best_d2_id = row[td.DEFAULT_ATTR_KEYS.NODE_ID]

print(f"Nearest predicted node to Daughter 2: ID {best_d2_id} (distance={min_dist:.2f} um)")

# Add division edge with boolean solution
attrs = {
    'solution': True,
    'edge_dist': float(min_dist),
    'edge_prob': 1.0
}
pred_graph.add_edge(31918, best_d2_id, attrs)

new_er = evaluate(pred_graph, gt_graph, scale=ds.scale)
new_rec = node_recall(pred_graph, gt_graph)
new_metrics = per_sample_metrics(new_er, 58672.0, new_rec)

print("="*65)
print("METRICS AFTER RECOVERING THE DIVISION:")
print(f"  Division TP/FN/FP:     {new_er.division_tp} / {new_er.division_fn} / {new_er.division_fp}")
print(f"  Edge TP/FN/FP:         {new_er.edge_tp} / {new_er.edge_fn} / {new_er.edge_fp} (TP change: {new_er.edge_tp - base_er.edge_tp:+d})")
print(f"  Edge Jaccard:          {new_metrics['edge_jaccard']:.5f} (was 0.86545, delta: {new_metrics['edge_jaccard'] - 0.86545:+.5f})")
print(f"  Adjusted Edge Jaccard: {new_metrics['adj_edge_jaccard']:.5f} (was 0.88636, delta: {new_metrics['adj_edge_jaccard'] - 0.88636:+.5f})")
print("="*65)
