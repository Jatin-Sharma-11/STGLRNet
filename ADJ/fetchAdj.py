import os
import torch
import torch.nn as nn
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os 


# def extractTODadj(history_data, adj_all, time_of_day_size=288, device='cuda'):
#     """
#     Extract time-aware adjacency matrices for each time step in each sample.

#     Args:
#         history_data (Tensor): shape [B, 12, 170, 3], with feature 1 being time_in_day (0-1 normalized).
#         adj_all (Tensor): shape [288, 170, 170], full temporal adjacency matrix set.
#         time_of_day_size (int): Number of time slots (e.g., 288 for 5-min intervals).
#         device (str): Device to move outputs to.

#     Returns:
#         Tensor: [B, 12, 170, 170], adjacency matrices matched to time-in-day steps.
#     """
#     B, T, N, _ = history_data.shape

#     # Extract time_in_day values: shape [B, T, N]
#     time_in_day_raw = history_data[..., 1]

#     # Use last node's time value in each time step (or average over nodes)
#     time_in_day_idx = (time_in_day_raw[:, :, 0] * time_of_day_size).long()  # [B, T]
#     time_in_day_idx = time_in_day_idx.clamp(0, time_of_day_size - 1)

#     # Fetch the adjacency matrices per time step
#     # adj_all: [288, 170, 170] → we index it for each (B, T)
#     adj_seq = []
#     for b in range(B):
#         batch_adj = []
#         for t in range(T):
#             idx = time_in_day_idx[b, t].item()
#             batch_adj.append(adj_all[idx])  # [170, 170]
#         adj_seq.append(torch.stack(batch_adj))  # [12, 170, 170]

#     adj_seq = torch.stack(adj_seq).to(device)  # [B, 12, 170, 170]
#     return adj_seq

def extractTODadj(history_data, adj_all, time_of_day_size=288, device='cuda'):
    """
    Extract time-aware adjacency matrices for 12 current and 12 future time steps.

    Args:
        history_data (Tensor): [B, 12, 170, 3], with feature 1 being normalized time_in_day (0-1).
        adj_all (Tensor): [288, 170, 170], full temporal adjacency matrix set.
        time_of_day_size (int): Total time slots in a day (e.g., 288 for 5-min intervals).
        device (str): Target device.

    Returns:
        Tuple[Tensor, Tensor]: 
            - [B, 12, 170, 170]: Adjacency for current time steps.
            - [B, 12, 170, 170]: Adjacency for future time steps.
    """
    B, T, N, _ = history_data.shape
    assert T == 12, "This function assumes input time steps = 12"

    # Use the last node's time for simplicity (or average if needed)
    base_time_idx = (history_data[:, 0, 0, 1] * time_of_day_size).long()  # [B]
    base_time_idx = base_time_idx.clamp(0, time_of_day_size - 1)

    # Generate time indices for t to t+11 and t+12 to t+23
    current_idx = base_time_idx.unsqueeze(1) + torch.arange(12, device=base_time_idx.device).unsqueeze(0)  # [B, 12]
    future_idx = base_time_idx.unsqueeze(1) + torch.arange(12, 24, device=base_time_idx.device).unsqueeze(0)  # [B, 12]

    # Wrap around using modulo
    current_idx = current_idx % time_of_day_size
    future_idx = future_idx % time_of_day_size

    # Gather adjacency matrices
    adj_seq_now = []
    adj_seq_next = []

    for b in range(B):
        batch_now = [adj_all[idx] for idx in current_idx[b]]
        batch_next = [adj_all[idx] for idx in future_idx[b]]
        adj_seq_now.append(torch.stack(batch_now))    # [12, 170, 170]
        adj_seq_next.append(torch.stack(batch_next))  # [12, 170, 170]

    adj_seq_now = torch.stack(adj_seq_now).to(device)    # [B, 12, 170, 170]
    adj_seq_next = torch.stack(adj_seq_next).to(device)  # [B, 12, 170, 170]

    return adj_seq_now, adj_seq_next


