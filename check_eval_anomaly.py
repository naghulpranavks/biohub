import sys
from pathlib import Path

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td
from biohub_tracking.io import open_dataset

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
train_dir = Path(r"D:\Downloads_2\biohub-cell-tracking-during-development\train")
video_stem = "44b6_12dfb391"

pred_result = td.graph.IndexedRXGraph.from_geff(pred_path)
pred_graph = pred_result[0] if isinstance(pred_result, tuple) else pred_result
ds = open_dataset(train_dir / video_stem, require_tracks=True)

print("Pred nodes:", pred_graph.num_nodes())
print("Pred edges:", pred_graph.num_edges())
print("GT tracks num_nodes:", ds.tracks.num_nodes())
print("GT tracks num_edges:", ds.tracks.num_edges())

# Let's inspect the GeffMetadata of GT
from geff import GeffMetadata
meta = GeffMetadata.read(train_dir / f"{video_stem}.geff")
print("Meta extra:", meta.extra)
