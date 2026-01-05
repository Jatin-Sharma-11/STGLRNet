import torch
import torch.nn as nn
import math

class GC1(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GC1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x: (B, 170, 128)
        adj: (170, 170)
        """
        #print(x.shape, "input shape")  # (B, 170, 128)
        support = torch.matmul(x, self.weight)  # (B, 170, out_features)
        #print("support.shape : ", support.shape)
        #print("adj shape : ", adj.shape)

        # Don't permute! It's already [B, N, F]
        output = torch.einsum('ij,bik->bik', adj, support)  # (B, 170, out_features)

        if self.bias is not None:
            output = output + self.bias  # (F,) will broadcast

        return output





class GC1L(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3, dropout=0.3):
        super(GC1L, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(GC1(in_features, hidden_features))
        self.norms.append(nn.LayerNorm(hidden_features))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.layers.append(GC1(hidden_features, hidden_features))
            self.norms.append(nn.LayerNorm(hidden_features))

        # Last layer (no norm needed if not followed by activation)
        self.layers.append(GC1(hidden_features, out_features))
        self.norms.append(None)  # Placeholder

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers):
            x = layer(x, adj)
            if i != self.num_layers - 1:  # Not last layer
                x = self.norms[i](x)      # LayerNorm across feature dim
                x = self.relu(x)
                x = self.dropout(x)
        return x
