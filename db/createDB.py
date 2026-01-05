import torch.nn as nn
import numpy as np
import faiss
import os
import torch

class createDB(nn.Module):
    def __init__(self, db_path, window_size=12):
        super().__init__()
        self.db_path = db_path
        self.window_size = window_size
        self.dummy_param = nn.Parameter(torch.zeros(1))
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        x: input time series, shape [T, 170, 3]
        """
        print(history_data.shape)
        # print(-p)
        x = history_data.squeeze(0)
        T, F, C = x.shape
        W = self.window_size
        D = W * F * C
        num_windows = T - W + 1

        windows = np.lib.stride_tricks.sliding_window_view(
            x.cpu().numpy(), (W, F, C)
        ).squeeze(axis=1)
        flat_windows = windows.reshape(num_windows, D).astype(np.float32)

        index = faiss.IndexFlatL2(D)
        index.add(flat_windows)

        os.makedirs(self.db_path, exist_ok=True)
        faiss.write_index(index, os.path.join(self.db_path, "faiss.index"))
        np.save(os.path.join(self.db_path, "time_series.npy"), x.cpu().numpy())

        print(f"Saved FAISS index and time series to {self.db_path}")
        return future_data +0.0 * self.dummy_param
