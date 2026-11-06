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

# Run official evaluation matching
er = evaluate(pred_graph, gt_graph, scale=ds.scale)

print("="*65)
print("1. DEEP DIVE: THE MISSED DIVISION")
print("="*65)
gt_divs = gt_graph.dividing_nodes()
pred_divs = pred_graph.dividing_nodes()
print(f"Ground Truth dividing nodes: {gt_divs}")
print(f"Predicted dividing nodes:    {pred_divs}")

gt_node_df = gt_graph.node_attrs()

for div_node in gt_divs:
    p_row = gt_node_df.filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == div_node)
    t = p_row["t"][0]
    pz, py, px = p_row["z"][0], p_row["y"][0], p_row["x"][0]
    p_um = np.array([pz * ds.scale[0], py * ds.scale[1], px * ds.scale[2]])
    daughters = gt_graph.successors(div_node)
    print(f"\nGT Dividing Node ID {div_node}:")
    print(f"  Frame t={t}, Coords (um): z={p_um[0]:.2f}, y={p_um[1]:.2f}, x={p_um[2]:.2f}")
    print(f"  Daughters ({len(daughters)}): {daughters}")
    for d in daughters:
        d_row = gt_node_df.filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == d)
        dz, dy, dx = d_row["z"][0], d_row["y"][0], d_row["x"][0]
        d_um = np.array([dz * ds.scale[0], dy * ds.scale[1], dx * ds.scale[2]])
        dist = np.linalg.norm(d_um - p_um)
        print(f"    -> Daughter ID {d}: dist={dist:.2f} um, coords: z={d_um[0]:.2f}, y={d_um[1]:.2f}, x={d_um[2]:.2f}")
    
    # Check matching in pred_graph
    pred_node_df = pred_graph.node_attrs()
    matched_p = pred_node_df.filter(pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) == div_node)
    if len(matched_p) > 0:
        pred_p_id = int(matched_p[td.DEFAULT_ATTR_KEYS.NODE_ID][0])
        pred_succ = pred_graph.successors(pred_p_id)
        print(f"\n  MATCH FOUND IN PREDICTION! Pred Node ID {pred_p_id}:")
        print(f"    Predicted children: {pred_succ} (count={len(pred_succ)})")
        if len(pred_succ) == 1:
            child_id = pred_succ[0]
            m_child = pred_node_df.filter(pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID) == child_id)[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID][0]
            print(f"    -> Predicted single child {child_id} (matched to GT {m_child})")
            print("    ==> ROOT CAUSE: Pipeline collapsed the division into a single linear continuation track!")
        elif len(pred_succ) == 0:
            print("    ==> ROOT CAUSE: Pipeline terminated the track at the division point!")
        elif len(pred_succ) > 1:
            print("    ==> Predicted branching occurred, but daughter matching failed!")
    else:
        print("  ==> ROOT CAUSE: The dividing parent cell itself was NOT detected!")

print("\n" + "="*65)
print("2. DEEP DIVE: THE 59 MISSED EDGES (edge_fn)")
print("="*65)
pred_node_df = pred_graph.node_attrs()
gt_to_pred = {}
for row in pred_node_df.iter_rows(named=True):
    p_id, m_gt = row[td.DEFAULT_ATTR_KEYS.NODE_ID], row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    if m_gt >= 0:
        gt_to_pred[int(m_gt)] = int(p_id)

matched_gt_nodes = set(gt_to_pred.keys())
print(f"GT Nodes detected: {len(matched_gt_nodes)} / {gt_graph.num_nodes()} ({len(matched_gt_nodes)/gt_graph.num_nodes():.1%})")

fn_both_nodes_detected = 0
fn_missing_node = 0
relink_swaps = 0
track_terminations = 0

for gt_u in gt_graph.node_ids():
    for gt_v in gt_graph.successors(gt_u):
        pred_u = gt_to_pred.get(gt_u)
        pred_v = gt_to_pred.get(gt_v)
        
        if pred_u is None or pred_v is None:
            fn_missing_node += 1
            continue
            
        pred_children = pred_graph.successors(pred_u)
        if pred_v in pred_children:
            pass # Matched
        else:
            fn_both_nodes_detected += 1
            if len(pred_children) > 0:
                relink_swaps += 1
            else:
                track_terminations += 1

print(f"\nBreakdown of 59 Missed GT Edges:")
print(f"  - Missed because one of the cells was NOT detected: {fn_missing_node}")
print(f"  - Missed despite BOTH cells being detected:        {fn_both_nodes_detected}")
print(f"      * Track linked to a WRONG neighbor (Crossover): {relink_swaps}")
print(f"      * Track prematurely TERMINATED (Broken Link):   {track_terminations}")
print("="*65)
