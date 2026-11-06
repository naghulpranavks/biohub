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

# 1. Base evaluation (Ground Truth)
pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]
ds = open_dataset(train_dir / video_stem, require_tracks=True)
gt_graph = ds.tracks

base_er = evaluate(pred_graph, gt_graph, scale=ds.scale)
base_rec = node_recall(pred_graph, gt_graph)
base_metrics = per_sample_metrics(base_er, 58672.0, base_rec)

print("="*65)
print("BASELINE METRICS (Before Surgical Fixes):")
print(f"  Edge Jaccard:          {base_metrics['edge_jaccard']:.5f}")
print(f"  Adjusted Edge Jaccard: {base_metrics['adj_edge_jaccard']:.5f}")
print(f"  Division TP/FN/FP:     {base_er.division_tp} / {base_er.division_fn} / {base_er.division_fp}")
print(f"  Edge TP/FN/FP:         {base_er.edge_tp} / {base_er.edge_fn} / {base_er.edge_fp}")
print("="*65)

# 2. Reload fresh prediction graph
pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]
pred_node_df = pred_graph.node_attrs()

nodes_by_t = {}
coord_by_id = {}
for row in pred_node_df.iter_rows(named=True):
    nid = row[td.DEFAULT_ATTR_KEYS.NODE_ID]
    t = row["t"]
    nodes_by_t.setdefault(t, []).append(nid)
    coord_by_id[nid] = np.array([row["z"]*ds.scale[0], row["y"]*ds.scale[1], row["x"]*ds.scale[2]])

out_degs = {nid: len(pred_graph.successors(nid)) for nid in pred_graph.node_ids()}
in_degs = {nid: len(pred_graph.predecessors(nid)) for nid in pred_graph.node_ids()}

new_edges_to_add = []
max_drift_um = 1.8

for t in sorted(nodes_by_t.keys()):
    if t + 1 not in nodes_by_t:
        continue
    tails = [nid for nid in nodes_by_t[t] if out_degs[nid] == 0]
    heads = [nid for nid in nodes_by_t[t + 1] if in_degs[nid] == 0]
    
    if not tails or not heads:
        continue
        
    for tail_id in tails:
        tail_coord = coord_by_id[tail_id]
        best_head = None
        best_d = max_drift_um
        for head_id in heads:
            head_coord = coord_by_id[head_id]
            d = np.linalg.norm(head_coord - tail_coord)
            if d < best_d:
                best_d = d
                best_head = head_id
                
        if best_head is not None:
            is_mutual = True
            for other_tail in tails:
                d_other = np.linalg.norm(coord_by_id[other_tail] - coord_by_id[best_head])
                if d_other < best_d:
                    is_mutual = False
                    break
            if is_mutual:
                new_edges_to_add.append((tail_id, best_head))
                out_degs[tail_id] += 1
                in_degs[best_head] += 1
                heads.remove(best_head)

print(f"\nAdding {len(new_edges_to_add)} safe micro-drift links (< {max_drift_um} um mutual nearest neighbors)...")

# Add edges one by one
for s, t in new_edges_to_add:
    try:
        pred_graph.add_edge(s, t)
    except Exception:
        pass

# Re-evaluate!
new_er = evaluate(pred_graph, gt_graph, scale=ds.scale)
new_rec = node_recall(pred_graph, gt_graph)
new_metrics = per_sample_metrics(new_er, 58672.0, new_rec)

print("\n" + "="*65)
print("NEW METRICS (With Micro-Drift Stitcher):")
print(f"  Edge Jaccard:          {new_metrics['edge_jaccard']:.5f} (Delta: {new_metrics['edge_jaccard'] - base_metrics['edge_jaccard']:+.5f})")
print(f"  Adjusted Edge Jaccard: {new_metrics['adj_edge_jaccard']:.5f} (Delta: {new_metrics['adj_edge_jaccard'] - base_metrics['adj_edge_jaccard']:+.5f})")
print(f"  Division TP/FN/FP:     {new_er.division_tp} / {new_er.division_fn} / {new_er.division_fp}")
print(f"  Edge TP/FN/FP:         {new_er.edge_tp} / {new_er.edge_fn} / {new_er.edge_fp} (TP change: {new_er.edge_tp - base_er.edge_tp:+d})")
print("="*65)
