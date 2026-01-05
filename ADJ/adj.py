import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_visualizations(adj_matrices, visual_output_dir, indices=None, prefix="Relational Map"):
    os.makedirs(visual_output_dir, exist_ok=True)
    indices = indices or [0, 72, 144, 216]

    for idx in indices:
        if idx >= len(adj_matrices):
            continue
        plt.figure(figsize=(6, 5))
        plt.imshow(adj_matrices[idx], cmap='hot', interpolation='nearest')
        plt.title(f"{prefix} at Time Step {idx}")
        # plt.colorbar(label="Adj Value")
        plt.xlabel("Sensor Index")
        plt.ylabel("Sensor Index")
        plt.tight_layout()
        plt.savefig(os.path.join(visual_output_dir, f"{prefix}_timestep_{idx}.png"))
        plt.close()


class TODAdj(nn.Module):
    def __init__(self, output_path_dir, visual_output_dir, window_size=12, time_steps_per_day=288, visualize_indices=None):
        super().__init__()
        self.output_path = os.path.join(output_path_dir, "adj_matrices.npz")
        self.visual_output_dir = visual_output_dir
        self.window_size = window_size
        self.time_steps_per_day = time_steps_per_day
        self.visualize_indices = [0,35,71,107,143,179,215,251,287]
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        if os.path.exists(self.output_path):
            loaded = np.load(self.output_path)
            dtw_matrices = loaded['TODAdj']
            save_visualizations(dtw_matrices, self.visual_output_dir, self.visualize_indices)
            return future_data

        # Prepare input tensor
        x = history_data.squeeze(0)  # Shape: (T, N, 1)
        x = x[:, :, 0]  # Shape: (T, N)
        total_timesteps, num_nodes = x.shape
        assert total_timesteps % self.time_steps_per_day == 0, "Mismatch in total timesteps and time_steps_per_day"

        num_days = total_timesteps // self.time_steps_per_day

        # Reshape to (days, time_per_day, nodes) → (time_per_day, days, nodes)
        reshaped = x.reshape(num_days, self.time_steps_per_day, num_nodes).permute(1, 0, 2).detach().cpu().numpy()
        assert reshaped.shape == (self.time_steps_per_day, num_days, num_nodes), "Reshaping failed"

        # Compute TOD adjacency using outer product
        outer = np.einsum('tbn,tbm->tbnm', reshaped, reshaped)
        tod_adj = np.mean(outer, axis=1)  # Shape: (T, N, N)

        # Save result
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        np.savez_compressed(self.output_path, TODAdj=tod_adj)

        save_visualizations(tod_adj, self.visual_output_dir, self.visualize_indices)

        return future_data
