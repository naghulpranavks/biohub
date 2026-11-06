import sys
from pathlib import Path
import numpy as np

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
base_graph = pred_result[0] if isinstance(pred_result, tuple) else pred_result

# Baseline score
er_base = compute_metric(base_graph, ds.tracks, scale=ds.scale)
rec_base = node_recall(base_graph, ds.tracks)
m_base = per_sample_metrics(er_base, n_total, rec_base)

print("="*75)
print(f"BASELINE METRIC FOR FIXTURE {video_stem}:")
print("="*75)
print(f"  Edge TP:           {er_base.edge_tp}")
print(f"  Edge FP:           {er_base.edge_fp}")
print(f"  Edge FN:           {er_base.edge_fn}")
print(f"  Raw Edge Jaccard:  {m_base['edge_jaccard']:.5f}")
print(f"  Adj Edge Jaccard:  {m_base['adj_edge_jaccard']:.5f}")
print(f"  Total Node Ratio:  {m_base['total_node_ratio']:.5f}")
print(f"  Divisions TP/FP/FN:{er_base.division_tp}/{er_base.division_fp}/{er_base.division_fn}")
print("="*75)
