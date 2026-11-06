import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate as compute_metric, node_recall, per_sample_metrics
from geff import GeffMetadata

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

# Load GT
ds = open_dataset(train_dir / video_stem, load_image=False, require_tracks=True)
meta = GeffMetadata.read(train_dir / f"{video_stem}.geff")
n_total = float(meta.extra["estimated_number_of_nodes"])

# Load base prediction
pred_result = td.graph.IndexedRXGraph.from_geff(pred_path)
graph = pred_result[0] if isinstance(pred_result, tuple) else pred_result

# Extract nodes and edges
nodes_dict = {}
for r in graph.node_attrs().iter_rows(named=True):
    nodes_dict[int(r["node_id"])] = {
        "node_id": int(r["node_id"]),
        "t": int(r["t"]),
        "z": float(r["z"]),
        "y": float(r["y"]),
        "x": float(r["x"]),
    }

raw_edges = []
learned_edge_probs = {}
for r in graph.edge_attrs().iter_rows(named=True):
    s = int(r["source_id"])
    t = int(r["target_id"])
    prob = r.get("edge_prob", 0.0)
    raw_edges.append((s, t))
    learned_edge_probs[(s, t)] = prob

VOXEL_SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

def pos_um(node):
    return np.array([node["z"], node["y"], node["x"]], dtype=np.float64) * VOXEL_SCALE_UM

position_um = {nid: pos_um(n) for nid, n in nodes_dict.items()}

# Function to run relinker
def run_motion_relink(tight_um=7.0, vel_weight=0.5, use_flow=False, flow_k=12, flow_radius_um=40.0):
    ids_by_t = {}
    for nid, n in nodes_dict.items():
        ids_by_t.setdefault(int(n["t"]), []).append(nid)
    
    selected_edges = []
    predecessor_position_um = {}
    
    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
            
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        
        # Build flow if requested
        flow_fn = None
        if use_flow and len(predecessor_position_um) >= 4:
            flow_src = []
            flow_disp = []
            for s_id in source_ids:
                if s_id in predecessor_position_um:
                    flow_src.append(position_um[s_id])
                    flow_disp.append(position_um[s_id] - predecessor_position_um[s_id])
            if len(flow_src) >= 4:
                flow_tree = cKDTree(flow_src)
                def get_flow(p):
                    dist, idx = flow_tree.query(p, k=min(flow_k+1, len(flow_src)), distance_upper_bound=flow_radius_um)
                    dist = np.atleast_1d(dist)
                    idx = np.atleast_1d(idx)
                    keep = np.isfinite(dist) & (dist >= 1.5)
                    idx = idx[keep][:flow_k]
                    if idx.size == 0:
                        return None
                    return np.median(np.array(flow_disp)[idx], axis=0)
                flow_fn = get_flow

        # Pass 1: tight
        for pass_name, gate_um in [("tight", tight_um), ("relaxed", 10.0)]:
            pass_sources = [s for s in source_ids if s in unmatched_sources]
            pass_targets = [t for t in target_ids if t in unmatched_targets]
            if not pass_sources or not pass_targets:
                continue
                
            big = gate_um * 1000.0 + 1.0
            cost = np.full((len(pass_sources), len(pass_targets)), big, dtype=np.float64)
            for i, s_id in enumerate(pass_sources):
                s_pos = position_um[s_id]
                prev_pos = predecessor_position_um.get(s_id)
                
                predicted = s_pos
                if flow_fn is not None:
                    step = flow_fn(s_pos)
                    if step is not None:
                        predicted = s_pos + step
                    elif prev_pos is not None:
                        predicted = s_pos + vel_weight * (s_pos - prev_pos)
                elif prev_pos is not None:
                    predicted = s_pos + vel_weight * (s_pos - prev_pos)
                    
                for j, t_id in enumerate(pass_targets):
                    t_pos = position_um[t_id]
                    raw = float(np.linalg.norm(t_pos - s_pos))
                    if raw > gate_um:
                        continue
                    motion = float(np.linalg.norm(t_pos - predicted))
                    prob = learned_edge_probs.get((s_id, t_id), 0.0)
                    cost[i, j] = motion + 0.05 * raw - 1.0 * prob
                    
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= big:
                    continue
                s_id = pass_sources[int(r)]
                t_id = pass_targets[int(c)]
                unmatched_sources.remove(s_id)
                unmatched_targets.remove(t_id)
                selected_edges.append((s_id, t_id))
                predecessor_position_um[t_id] = position_um[s_id]
                
    return selected_edges

# Function to build td graph and evaluate
def evaluate_edges(edges_list, label):
    # Construct td graph
    node_schema = graph.node_attrs().schema
    import polars as pl
    node_df = graph.node_attrs().to_pandas()
    
    edge_df = {
        "edge_id": list(range(len(edges_list))),
        "source_id": [e[0] for e in edges_list],
        "target_id": [e[1] for e in edges_list],
        "solution": [True] * len(edges_list),
        "edge_prob": [learned_edge_probs.get(e, 0.5) for e in edges_list],
        "edge_dist": [1.0] * len(edges_list),
    }
    pred_graph_new = td.graph.IndexedRXGraph.from_data(
        node_attrs=pl.DataFrame(node_df),
        edge_attrs=pl.DataFrame(edge_df),
        node_id_col="node_id",
        edge_id_col="edge_id",
        edge_source_col="source_id",
        edge_target_col="target_id",
    )
    er = compute_metric(pred_graph_new, ds.tracks, scale=ds.scale)
    rec = node_recall(pred_graph_new, ds.tracks)
    m = per_sample_metrics(er, n_total, rec)
    return m, er

print("="*80)
print(f"{'EXPERIMENT':35s} | {'ADJ JAC':10s} | {'RAW JAC':10s} | {'TP/FP/FN':12s} | {'DELTA'}")
print("="*80)

# Evaluate baseline raw edges
m_base, er_base = evaluate_edges(raw_edges, "Raw ILP Baseline")
print(f"{'Raw ILP Baseline (Current 0.947)':35s} | {m_base['adj_edge_jaccard']:.5f}    | {m_base['edge_jaccard']:.5f}    | {er_base.edge_tp}/{er_base.edge_fp}/{er_base.edge_fn}   | base")

# Test relinker variations
experiments = [
    ("Relinker (tight=7.0, vel=0.50)", dict(tight_um=7.0, vel_weight=0.50, use_flow=False)),
    ("Relinker (tight=5.5, vel=0.50)", dict(tight_um=5.5, vel_weight=0.50, use_flow=False)),
    ("Relinker (tight=5.5, vel=0.25)", dict(tight_um=5.5, vel_weight=0.25, use_flow=False)),
    ("Relinker + Flow (tight=5.5, vel=0.25)", dict(tight_um=5.5, vel_weight=0.25, use_flow=True)),
]

for label, kwargs in experiments:
    relinked = run_motion_relink(**kwargs)
    m, er = evaluate_edges(relinked, label)
    delta = m['adj_edge_jaccard'] - m_base['adj_edge_jaccard']
    sign = "+" if delta >= 0 else ""
    print(f"{label:35s} | {m['adj_edge_jaccard']:.5f}    | {m['edge_jaccard']:.5f}    | {er.edge_tp}/{er.edge_fp}/{er.edge_fn}   | {sign}{delta:.5f}")
