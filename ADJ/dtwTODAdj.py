import os
import torch
import torch.nn as nn
import numpy as np
from dtaidistance import dtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_dtw_adjacency(data, normalize=True):
    """
    data: numpy array of shape (T, D, N)
    Returns: normalized adjacency matrix of shape (T, N, N)
    """
    time_steps, days, num_nodes = data.shape
    dtw_matrices = np.zeros((time_steps, num_nodes, num_nodes), dtype=np.float64)

    for t in tqdm(range(time_steps), desc="Computing Fast DTW adjacency"):
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                s1 = np.ascontiguousarray(data[t, :, i], dtype=np.float64)
                s2 = np.ascontiguousarray(data[t, :, j], dtype=np.float64)
                distance = dtw.distance_fast(s1, s2)
                dtw_matrices[t, i, j] = distance
                dtw_matrices[t, j, i] = distance

    if normalize:
        sigma = np.std(dtw_matrices)
        adj = np.exp(-dtw_matrices ** 2 / (2 * sigma ** 2))
    else:
        min_val = dtw_matrices.min()
        max_val = dtw_matrices.max()
        adj = 1 - (dtw_matrices - min_val) / (max_val - min_val + 1e-5)

    return adj




def save_visualizations(adj_matrices, visual_output_dir, indices):
    os.makedirs(visual_output_dir, exist_ok=True)
    for idx in indices:
        plt.figure(figsize=(6, 5))
        plt.imshow(adj_matrices[idx], cmap='hot', interpolation='nearest')
        # plt.title(f"Relational Map at Time Step {idx}")
        # plt.colorbar(label="Score")
        # plt.xlabel("Sensor Index")
        # plt.ylabel("Sensor Index")
        plt.tight_layout()
        img_path = os.path.join(visual_output_dir, f"dtw_adj_timestep_{idx}.png")
        plt.savefig(img_path)
        plt.close()
    print(f"[INFO] Saved DTW adjacency visualizations to: {visual_output_dir}")


class dtwTODAdj(nn.Module):
    def __init__(self, output_path_dir, visual_output_dir, window_size=12):
        super().__init__()
        self.output_path = os.path.join(output_path_dir, "dtw_adj_matrices.npz")
        self.visual_output_dir = visual_output_dir
        self.window_size = window_size
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        os.makedirs(self.visual_output_dir, exist_ok=True)

        if os.path.exists(self.output_path):
            print(f"[INFO] File '{self.output_path}' exists. Loading...")
            dtw_adj = np.load(self.output_path)["dtw_adj"]
            save_visualizations(dtw_adj, self.visual_output_dir, indices=[0,35,71,107,143,179,215,251,287])
            return self.dummy_param * future_data

        x = history_data.squeeze(0)
        x = x[:, :, 0]  # shape: (T, N)
        total_timesteps = x.shape[0]
        time_steps = 288
        assert total_timesteps % time_steps == 0, "Time must be divisible by time_steps"
        num_days = total_timesteps // time_steps

        reshaped = x.reshape(num_days, time_steps, -1).permute(1, 0, 2).cpu().numpy()


    
        # reshaped = x.reshape(self.num_days, time_steps, -1).permute(1, 0, 2).cpu().numpy()  # (T, D, N)
        print(f"[DEBUG] reshaped input shape: {reshaped.shape}")

        dtw_adj = compute_dtw_adjacency(reshaped, normalize=True)
        np.savez_compressed(self.output_path, dtw_adj=dtw_adj)
        print(f"[INFO] Saved normalized DTW adjacency matrices to '{self.output_path}'")

        save_visualizations(dtw_adj, self.visual_output_dir, indices=[0, 1, 2, 3, 4, 5, 6, 12, 24, 65, 78, 99])
        return self.dummy_param * future_data
