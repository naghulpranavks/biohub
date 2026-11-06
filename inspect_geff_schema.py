import sys
from pathlib import Path

repo_src = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\src")
sys.path.insert(0, str(repo_src))

import tracksdata as td

pred_path = Path(r"c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\repo\predictions\Naghul Pranav\unet_transformer\split_0\44b6_12dfb391.zarr.geff")
pred_result = td.graph.IndexedRXGraph.from_geff(pred_path)
graph = pred_result[0] if isinstance(pred_result, tuple) else pred_result

print("Node attributes schema:")
print(graph.node_attrs().schema)

print("\nEdge attributes schema:")
print(graph.edge_attrs().schema)

# Sample 3 nodes and 3 edges
print("\nSample nodes:")
for r in list(graph.node_attrs().iter_rows(named=True))[:3]:
    print(r)

print("\nSample edges:")
for r in list(graph.edge_attrs().iter_rows(named=True))[:3]:
    print(r)
