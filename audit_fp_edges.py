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

# Inspect edge_attrs to find the 52 False Positive edges
edge_attrs = pred_graph.edge_attrs(attr_keys=[
    td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
    td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
    td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK,
])
pred_node_attrs = pred_graph.node_attrs(attr_keys=[
    td.DEFAULT_ATTR_KEYS.NODE_ID,
    td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
    "t", "z", "y", "x"
])

# Which edges are FP?
# Remember: in biohub_tracking, an edge in pred_graph is considered a False Positive if:
# it connects a node that was valid in ground truth (out_valid or in_valid) but the edge was NOT in ground truth!
print("="*65)
print("AUDITING THE 52 FALSE POSITIVE EDGES (edge_fp)")
print("="*65)

# Join pred edges with matched_node_id of source and target
source_map = {}
target_map = {}
for row in pred_node_attrs.iter_rows(named=True):
    source_map[row[td.DEFAULT_ATTR_KEYS.NODE_ID]] = row
    target_map[row[td.DEFAULT_ATTR_KEYS.NODE_ID]] = row

fp_edges = []
for row in edge_attrs.iter_rows(named=True):
    src = row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]
    tgt = row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
    is_matched = row[td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK]
    
    src_node = source_map.get(src)
    tgt_node = target_map.get(tgt)
    
    if src_node and tgt_node:
        m_src = src_node[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        m_tgt = tgt_node[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        
        # If either src or tgt was a real annotated GT cell, but the edge wasn't matched
        if (m_src >= 0 or m_tgt >= 0) and not is_matched:
            # Measure physical distance
            sz, sy, sx = src_node["z"]*ds.scale[0], src_node["y"]*ds.scale[1], src_node["x"]*ds.scale[2]
            tz, ty, tx = tgt_node["z"]*ds.scale[0], tgt_node["y"]*ds.scale[1], tgt_node["x"]*ds.scale[2]
            dist = np.linalg.norm([tz-sz, ty-sy, tx-sx])
            fp_edges.append({
                "src": src, "tgt": tgt, "m_src": m_src, "m_tgt": m_tgt,
                "t_src": src_node["t"], "t_tgt": tgt_node["t"],
                "dist": dist
            })

print(f"Total FP edges identified: {len(fp_edges)}")
fp_both_gt = [e for e in fp_edges if e["m_src"] >= 0 and e["m_tgt"] >= 0]
fp_one_unannotated = [e for e in fp_edges if e["m_src"] < 0 or e["m_tgt"] < 0]

print(f"  - FP Edges connecting two REAL ground-truth cells (Spurious/Wrong cross-links): {len(fp_both_gt)}")
print(f"  - FP Edges connecting a real cell to an unannotated/noise cell (Leakage into background): {len(fp_one_unannotated)}")

if fp_both_gt:
    dists = [e["dist"] for e in fp_both_gt]
    print(f"  - Spurious cross-link distances: min={min(dists):.2f}um, median={np.median(dists):.2f}um, max={max(dists):.2f}um")
print("="*65)
