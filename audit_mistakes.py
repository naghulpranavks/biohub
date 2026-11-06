import sys
from pathlib import Path

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
import numpy as np
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate, _evaluate
from biohub_tracking.division_metrics import evaluate_divisions

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

pred_graph = td.graph.IndexedRXGraph.from_geff(pred_path)[0]
ds = open_dataset(train_dir / video_stem, require_tracks=True)
gt_graph = ds.tracks

print("--- AUDITING THE MISSED DIVISION ---")
# Find division nodes in ground truth (out-degree > 1)
gt_edges = gt_graph.edges()
gt_parents = {}
for u, v in zip(gt_edges[0], gt_edges[1]):
    u, v = int(u), int(v)
    gt_parents.setdefault(u, []).append(v)

divisions_gt = {u: vs for u, vs in gt_parents.items() if len(vs) > 1}
print(f"Found {len(divisions_gt)} division(s) in Ground Truth:")
for parent_id, daughter_ids in divisions_gt.items():
    p_coord = gt_graph.node_coordinates(parent_id)
    print(f"  Parent ID {parent_id}: coords={p_coord}, Daughters={daughter_ids}")
    for d_id in daughter_ids:
        d_coord = gt_graph.node_coordinates(d_id)
        dist = np.linalg.norm((d_coord - p_coord) * np.array(ds.scale))
        print(f"    -> Daughter ID {d_id}: coords={d_coord}, physical distance={dist:.3f} um")

# Now check what the prediction did at these coordinates
print("\nChecking matching in prediction:")
_evaluate(pred_graph, gt_graph, "jaccard", scale=ds.scale, max_distance=7.0)

for parent_id, daughter_ids in divisions_gt.items():
    # Find matched pred node for parent
    matched_p = None
    for p_idx in range(pred_graph.num_nodes()):
        # Check if matched to parent_id
        pass

print("\n--- AUDITING THE 59 MISSED EDGES (edge_fn) ---")
# Let's inspect ground truth edges and whether their endpoints were matched
gt_node_matched = {}
# tracksdata matching puts matched IDs
pred_to_gt = {}
if "gt_id" in pred_graph.node_attributes():
    pred_gt_ids = pred_graph.node_attributes()["gt_id"]
    for p_id, g_id in enumerate(pred_gt_ids):
        if g_id >= 0:
            pred_to_gt[p_id] = int(g_id)
            gt_node_matched[int(g_id)] = p_id

print(f"GT nodes matched in prediction: {len(gt_node_matched)} / {gt_graph.num_nodes()} ({len(gt_node_matched)/gt_graph.num_nodes():.1%})")

fn_endpoints_both_detected = 0
fn_one_endpoint_missing = 0
fn_distances = []

for u, v in zip(gt_edges[0], gt_edges[1]):
    u, v = int(u), int(v)
    u_matched = u in gt_node_matched
    v_matched = v in gt_node_matched
    
    # Check if edge (u_pred, v_pred) exists in pred_graph
    edge_exists = False
    if u_matched and v_matched:
        fn_endpoints_both_detected += 1
        u_p = gt_node_matched[u]
        v_p = gt_node_matched[v]
        # Check if edge u_p -> v_p exists
        pred_targets = pred_graph.neighbors(u_p)
        if v_p in pred_targets:
            edge_exists = True
        else:
            u_coord = gt_graph.node_coordinates(u)
            v_coord = gt_graph.node_coordinates(v)
            d = np.linalg.norm((v_coord - u_coord) * np.array(ds.scale))
            fn_distances.append(d)
    else:
        fn_one_endpoint_missing += 1

print(f"Missed Edges where BOTH cells were successfully detected: {fn_endpoints_both_detected}")
print(f"Missed Edges where one cell was missing: {fn_one_endpoint_missing}")
if fn_distances:
    print(f"Missed Edges jump distances: min={min(fn_distances):.2f}um, median={np.median(fn_distances):.2f}um, max={max(fn_distances):.2f}um, p90={np.percentile(fn_distances, 90):.2f}um")
