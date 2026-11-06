import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

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

VOXEL_SCALE = np.array([1.0, 0.1625, 0.1625]) # z, y, x scale from ds.scale
scale = ds.scale

print("Voxel scale:", scale)

# Extract nodes and edges
nodes_df = graph.node_attrs().to_pandas()
edges_df = graph.edge_attrs().to_pandas()

print(f"Total graph nodes: {len(nodes_df)}, edges: {len(edges_df)}")
