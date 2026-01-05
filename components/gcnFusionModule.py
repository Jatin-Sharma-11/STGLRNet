import torch
import torch.nn as nn
import math
import inspect
class ABFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, fused_out, pivot_out):
        x = torch.cat([fused_out, pivot_out], dim=-1)  # [B, N, 2F]
        alpha = self.gate(x)  # [B, N, F]
        beta = 1 - alpha
        return alpha * fused_out + beta * pivot_out
