import sys
from pathlib import Path

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
from biohub_tracking.io import open_dataset
from biohub_tracking.metrics import evaluate as compute_metric, node_recall, per_sample_metrics
from geff import GeffMetadata

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

pred_result = td.graph.IndexedRXGraph.from_geff(pred_path)
pred_graph = pred_result[0] if isinstance(pred_result, tuple) else pred_result
ds = open_dataset(train_dir / video_stem, load_image=False, require_tracks=True)

meta = GeffMetadata.read(train_dir / f"{video_stem}.geff")
n_total = float(meta.extra["estimated_number_of_nodes"])

er = compute_metric(pred_graph, ds.tracks, scale=ds.scale)
recall = node_recall(pred_graph, ds.tracks)
metrics = per_sample_metrics(er, n_total, recall)

print("="*65)
print(f"OFFICIAL PER-SAMPLE METRIC SUMMARY: {video_stem}")
print("="*65)
print(f"  Estimated True Nodes (N_total): {int(n_total)}")
print(f"  Predicted Nodes:               {pred_graph.num_nodes()}")
for k, v in sorted(metrics.items()):
    if isinstance(v, float):
        print(f"  {k:30s}: {v:.5f}")
    else:
        print(f"  {k:30s}: {v}")
print("="*65)
