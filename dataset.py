import os
import zarr
import numpy as np
import pandas as pd

class BiohubDataset:
    def __init__(self, data_dir):
        """
        data_dir: Path to the directory containing train/ and test/ folders.
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        
    def get_train_samples(self):
        """Returns a list of sample IDs available in the training set."""
        if not os.path.exists(self.train_dir):
            return []
        samples = set()
        for f in os.listdir(self.train_dir):
            if f.endswith('.zarr') or f.endswith('.geff'):
                samples.add(f.split('.')[0])
        return sorted(list(samples))

    def load_volume(self, sample_id, split='train'):
        """
        Loads the Zarr image volume for a given sample.
        Returns a Zarr array-like object of shape (T, Z, Y, X).
        """
        split_dir = self.train_dir if split == 'train' else self.test_dir
        zarr_path = os.path.join(split_dir, f"{sample_id}.zarr")
        
        # In Zarr v3, the array is typically at the root or under '0/'
        # The competition description says: "Each contains a single array at path 0/ with shape (T, Z, Y, X)"
        try:
            root = zarr.open(zarr_path, mode='r')
            volume = root['0']
            return volume
        except Exception as e:
            print(f"Error loading Zarr volume for {sample_id}: {e}")
            return None

    def load_ground_truth(self, sample_id):
        """
        Loads the ground truth GEFF graph data.
        Returns a tuple: (nodes_df, edges_df)
        """
        geff_path = os.path.join(self.train_dir, f"{sample_id}.geff")
        try:
            root = zarr.open(geff_path, mode='r')
            
            # Load nodes
            node_ids = root['nodes/ids'][:]
            t = root['nodes/props/t/values'][:]
            z = root['nodes/props/z/values'][:]
            y = root['nodes/props/y/values'][:]
            x = root['nodes/props/x/values'][:]
            
            nodes_df = pd.DataFrame({
                'node_id': node_ids,
                't': t,
                'z': z,
                'y': y,
                'x': x
            })
            
            # Load edges
            edges_raw = root['edges/ids'][:]
            edges_df = pd.DataFrame({
                'source_id': edges_raw[:, 0],
                'target_id': edges_raw[:, 1]
            })
            
            return nodes_df, edges_df
        except Exception as e:
            print(f"Error loading GEFF ground truth for {sample_id}: {e}")
            return None, None

if __name__ == '__main__':
    print("Dataset loader initialized.")
    # Example usage:
    # dataset = BiohubDataset('./data')
    # samples = dataset.get_train_samples()
    # if samples:
    #     vol = dataset.load_volume(samples[0])
    #     nodes, edges = dataset.load_ground_truth(samples[0])
    #     print(f"Loaded volume shape: {vol.shape}")
    #     print(f"Nodes:\n{nodes.head()}")
    #     print(f"Edges:\n{edges.head()}")
