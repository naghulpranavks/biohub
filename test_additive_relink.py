import sys
from pathlib import Path
import numpy as np

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
base_rec = node_recall(pred_graph, gt_graph)
base_metrics = per_sample_metrics(base_er, 58672.0, base_rec)

print("="*65)
print("TESTING ADDITIVE RELINKER LOGIC LOCALLY")
print(f"  Baseline Edge Jaccard:          {base_metrics['edge_jaccard']:.5f}")
print(f"  Baseline Adjusted Edge Jaccard: {base_metrics['adj_edge_jaccard']:.5f}")
print("="*65)
