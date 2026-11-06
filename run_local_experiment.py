import sys
from pathlib import Path
import json

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import types
m_ipython = types.ModuleType("IPython")
m_display = types.ModuleType("IPython.display")
m_display.display = lambda *a, **k: None
m_ipython.display = m_display
sys.modules["IPython"] = m_ipython
sys.modules["IPython.display"] = m_display

import os
os.environ["BIOHUB_MODEL_ARTIFACTS"] = r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"

# Import from nb_source
import nb_source as nb

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

# Load GT
gt_path = train_dir / f"{video_stem}.geff"
gt_graph = nb.graph_from_geff(gt_path)
gt_nodes_plain, gt_edges_plain = nb.graph_to_plain(gt_graph)
t_true = nb.read_estimated_true_node_count(gt_path)

# Load Pred
pred_graph = nb.graph_from_geff(pred_path)
raw_nodes_by_id = {}
for row in pred_graph.node_attrs().iter_rows(named=True):
    node_id = int(row["node_id"])
    raw_nodes_by_id[node_id] = {
        "node_id": node_id, "t": int(row["t"]),
        "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
    }
raw_edges = []
for row in pred_graph.edge_attrs().iter_rows(named=True):
    edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
    raw_edges.append({
        "source_id": int(row["source_id"]),
        "target_id": int(row["target_id"]),
        "edge_prob": edge_prob,
    })

print(f"Loaded {len(raw_nodes_by_id)} raw nodes, {len(raw_edges)} raw edges for {video_stem}")
print(f"GT has {len(gt_nodes_plain)} nodes, {len(gt_edges_plain)} edges, t_true = {t_true}")

def eval_config(config_dict, label):
    # Deepcopy raw graph
    import copy
    nodes_copy = copy.deepcopy(raw_nodes_by_id)
    edges_copy = copy.deepcopy(raw_edges)
    
    # Save & apply overrides
    saved = nb.pp_apply(config_dict)
    try:
        nodes_filtered, edges_filtered, stats = nb.filter_output_graph(nodes_copy, edges_copy, dataset=video_stem)
    finally:
        nb.pp_restore(saved)
        
    plain_nodes = nb.nodes_by_id_to_plain(nodes_filtered)
    plain_edges = [(int(e["source_id"]), int(e["target_id"])) for e in edges_filtered]
    
    row = nb.score_sample(plain_nodes, plain_edges, gt_nodes_plain, gt_edges_plain, t_true)
    proxy = row["adjusted_edge_jaccard"] + nb.VALIDATOR_DIVISION_WEIGHT * row["div_jaccard"]
    return row, proxy

# 1. Base
row_base, proxy_base = eval_config({}, "base")

print("="*75)
print(f"{'CONFIGURATION':30s} | {'ADJ JACCARD':12s} | {'DIV JAC':10s} | {'PROXY':10s} | {'TP/FP/FN'}")
print("="*75)
print(f"{'base (Current 0.947)':30s} | {row_base['adjusted_edge_jaccard']:.5f}      | {row_base['div_jaccard']:.5f}    | {proxy_base:.5f}    | {row_base['edge_tp']}/{row_base['edge_fp']}/{row_base['edge_fn']}")

candidates = {
    "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "vel025": {"MOTION_RELINK_VELOCITY_WEIGHT": 0.25},
    "divwide": {"SAFE_DIV_MAX_UM": 11.0, "SAFE_DIV_SISTER_MAX_UM": 16.0, "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0},
    "combo_t55_vel025": {"MOTION_RELINK_TIGHT_UM": 5.5, "MOTION_RELINK_VELOCITY_WEIGHT": 0.25},
    "combo_t55_divwide": {"MOTION_RELINK_TIGHT_UM": 5.5, "SAFE_DIV_MAX_UM": 11.0, "SAFE_DIV_SISTER_MAX_UM": 16.0, "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0},
}

for name, cfg in candidates.items():
    row, proxy = eval_config(cfg, name)
    delta = proxy - proxy_base
    sign = "+" if delta >= 0 else ""
    print(f"{name:30s} | {row['adjusted_edge_jaccard']:.5f}      | {row['div_jaccard']:.5f}    | {proxy:.5f} ({sign}{delta:.5f}) | {row['edge_tp']}/{row['edge_fp']}/{row['edge_fn']}")
