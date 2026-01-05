import numpy as np
import torch
import os

class AdjacencyLoader:
    def __init__(self, file_path, device='cuda'):
        """
        Loads DTW adjacency matrix from .npz file at runtime and moves it to the desired device.

        Args:
            file_path (str): Path to the .npz file.
            device (str): 'cuda' or 'cpu'.
        """
        self.device = device
        self.file_path = file_path
        self.adj_matrix = self.load_adj_matrix()

    def load_adj_matrix(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        print(f"Loading DTW matrix from: {self.file_path}")
        npz_data = np.load(self.file_path)
        # Handle either single array or named array
        if isinstance(npz_data, np.lib.npyio.NpzFile):
            adj = npz_data[npz_data.files[0]]  # Load the first array inside
        else:
            adj = npz_data
        
        adj_tensor = torch.tensor(adj, dtype=torch.float32).to(self.device)
        print(f"Loaded DTW adjacency matrix with shape: {adj_tensor.shape}")
        return adj_tensor

    def get(self):
        """Returns the loaded DTW adjacency matrix as a PyTorch tensor."""
        return self.adj_matrix
