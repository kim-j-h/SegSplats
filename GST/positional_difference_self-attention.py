import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn

class positional_difference_self_attention(nn.Module):
    def __init__(self, D, subtract_similarity = True):
        super(positional_difference_self_attention, self).__init__()
        self.subtract_similarity = subtract_similarity
        self.W_q = nn.Linear(D, D, bias=False)
        self.W_k = nn.Linear(D, D, bias=False)
        self.W_v = nn.Linear(D, D, bias=False)
        self.cbr = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=1, bias=False),
            nn.BatchNorm1d(D),
            nn.ReLU(inplace=True)
        )
        
    def forward (self, f_in):
        ### f_in: (B, N, D)
        Q, K, V = self.W_q(f_in), self.W_k(f_in), self.W_v(f_in)  # (B, N, D)
        if self.subtract_similarity:
            A = torch.softmax(Q - K)  # (B, N, N)
        else:
            A = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5), dim=-1)  # (B, N, N)
        f_sa = A * V  # (B, N, D)
        return f_in + self.cbr((f_in - f_sa).permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, D)
