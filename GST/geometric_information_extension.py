import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn

class geometric_information_extension(nn.Module):
    def __init__(self, cov_emb_dim = 24, rel_pos_emb_dim = 24, cov_hid_dim = 12, rel_pos_hid_dim = 12):
        super(geometric_information_extension, self).__init__()
        self.covariance_embedding = nn.Sequential(
            nn.Conv1d(6, cov_hid_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(cov_hid_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(cov_hid_dim, cov_emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(cov_emb_dim),
            nn.ReLU(inplace=True)
        )
        self.relative_position_embedding = nn.Sequential(
            nn.Conv1d(3, rel_pos_hid_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(rel_pos_hid_dim), 
            nn.ReLU(inplace=True),
            nn.Conv1d(rel_pos_hid_dim, rel_pos_emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(rel_pos_emb_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, xyz, sigma):
        ### xyz: (B, N, 3), sigma: (B, N, 6)
        batch_size, num_points, _ = xyz.size()
        centroid = torch.mean(xyz, dim=1, keepdim=True)  # (B, 1, 3)
        relative_position = xyz - centroid  # (B, N, 3)
        relative_position_emb = self.relative_position_embedding(relative_position.permute(0, 2, 1))  # (B, 24, N)
        covariance_emb = self.covariance_embedding(sigma.permute(0, 2, 1))  # (B, 24, N)
        combined_feature = torch.cat([relative_position_emb, covariance_emb], dim=1)  # (B, 48, N)
        return combined_feature.permute(0, 2, 1)
