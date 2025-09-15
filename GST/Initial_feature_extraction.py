import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn

class InitialFeatureExtractor(nn.Module):
    def __init__(self, out_channels, k = 20, xyz_embedding_dim = 12, alpha = 1000):
        super(InitialFeatureExtractor, self).__init__()
        self.k = k
        self.xyz_embedding_dim = xyz_embedding_dim # D_xyz
        self.alpha = alpha
        self.cbr = nn.Sequential(
            nn.Conv1d(xyz_embedding_dim * 6, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def trigonometric_embedding(self, xyz):
        ### xyz: (B, N, 3)
        batch_size, num_points, _ = xyz.size()
        c = torch.arange(self.xyz_embedding_dim, device=xyz.device).float()  # (D_xyz,)
        freq_factor = self.alpha ** (12(2*c - self.xyz_embedding_dim) / self.xyz_embedding_dim)  # (D_xyz,)
        freq_factor = freq_factor.view(1, 1, 1, self.xyz_embedding_dim)  # (1, 1, 1, D_xyz)
        even = torch.sin(xyz.unsqueeze(-1) * freq_factor)  # (B, N, 3, D_xyz)
        odd = torch.cos(xyz.unsqueeze(-1) * freq_factor)   # (B, N, 3, D_xyz)
        embedding = torch.stack([even, odd], dim=-1).flatten(start_dim=-2)
        embedding = embedding.view(batch_size, num_points, -1)  # (B, N, 3*D_xyz*2)
        return embedding  # (B, N, 6*D_xyz)

    def knn(self, xyz, k):
        ### xyz: (B, N, 3)
        batch_size, num_points, _ = xyz.size()
        xyz_flat = xyz_tensor.view(-1, 3) # (B*N, 3)
        batch_idx = torch.arange(batch_size, device=xyz_tensor.device).repeat_interleave(num_points)
        edge_index = knn_graph(xyz_flat, k=k, batch=batch_idx, loop=True)
        num_edges = edge_index.size(1)
        knn_indices_flat = edge_index[1].view(batch_size, num_points, k)
        return knn_indices_flat  # (B, N, k)

    def forward(self, xyz, feature):
        ### xyz: (B, N, 3), feature: (B, N, D)
        batch_size, num_points, _ = xyz.size()
        k = self.k
        idx = self.knn(xyz, k=k)  # (B, N, k)
        f_ck = feature.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, D)
        f_ck = f_ck.reshape(B, N, -1)  # (B, N, k*D)
        TriE = self.trigonometric_embedding(xyz)  # (B, N, 6*D_xyz)
        f_ck^w = (f_ck + TriE) * TriE  # (B, N, k*D)
        maxpool_f_ck = torch.max(f_ck^w, dim=2)[0]  # (B, N, k*D)
        avgpool_f_ck = torch.mean(f_ck^w, dim=2)  # (B, N, k*D)
        aggregated_feature = max_pool_f_ck + avgpool_f_ck  # (B, N, k*D)

        aggregated_feature_perm = aggregated_feature.permute(0, 2, 1)  # (B, k*D, N)
        updated_feature = self.cbr(aggregated_feature_perm)  # (B, k*D, N)
        
        return updated_feature.permute(0, 2, 1)  # (B, N, k*D)