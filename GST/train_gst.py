# Standard Library Imports
import sys
import os
import time
import copy
from argparse import ArgumentParser
from pathlib import Path
from typing import List

# Append specific path to the system
sys.path.append("../feature-splatting-inria")

# Third-party Library Imports
import numpy as np
import torch
import open3d as o3d
import roboticstoolbox as rtb
from spatialmath import SE3, SO3
import transforms3d.euler as euler

# Viser Library Imports
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf

# Custom Module Imports
from scene import Scene, skip_feat_decoder
from arguments import ModelParams, get_combined_args, PipelineParams, OptimizationParams
from gaussian_renderer import GaussianModel, render
import featsplat_editor
from grasping import grasping_utils, plan_utils
from gaussian_edit import edit_utils
from positional_difference_self_attention import positional_difference_self_attention
from geometric_information_extension import geometric_information_extension
from Initial_feature_extraction import InitialFeatureExtractor


class dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

class GaussianTransformer(torch.nn.Module):
    #### k * D = 6 * D_xyz
    def __init__(self, D = 128, k=4, xyz_embedding_dim=8, alpha=1000):
        super(GaussianTransformer, self).__init__()
        self.pdsa = positional_difference_self_attention(D=D, subtract_similarity=True)
        self.gie = geometric_information_extension(cov_emb_dim=24, rel_pos_emb_dim=24, cov_hid_dim=12, rel_pos_hid_dim=12)
        self.ife = InitialFeatureExtractor(out_channels=D, k=k, xyz_embedding_dim=xyz_embedding_dim, alpha=1000)

    def forward(self, xyz, sigma, feature):
        ### xyz: (B, N, 3), sigma: (B, N, 6), feature: (B, N, D)
        gie_feature = self.gie(xyz, sigma)  # (B, N, 48)
        feature = torch.cat([feature, gie_feature], dim=-1)  # (B, N, D+48)
        feature = self.ife(xyz, feature)  # (B, N, D)
        feature = self.pdsa(feature)  # (B, N, D)
        return feature  # (B, N, D)

def build_model(args) -> torch.nn.Module:
    # Initialize the Gaussian Transformer
    gst = GaussianTransformer(D=args.model_params.feature_dim, k=args.pipeline_params.k, xyz_embedding_dim=args.pipeline_params.xyz_embedding_dim, alpha=args.pipeline_params.alpha)
    
    # Initialize the Skip Feature Decoder
    skip_decoder = skip_feat_decoder(input_dim=args.model_params.feature_dim, part_level=args.model_params.part_level)
    
    return torch.nn.ModuleDict({
        'gst': gst,
        'skip_decoder': skip_decoder
    })

def train(gst, dataset, args):

    optimizer = torch.optim.Adam(gst.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(args.num_epochs):
        gst.train()
        total_loss = 0.0
        for batch in dataset:
            optimizer.zero_grad()
            if batch.shape == 2:
                



if __main__ == "__main__":
    parser = ArgumentParser()
    parser = get_combined_args(parser)
    args = parser.parse_args()

    model = build_model(args)
    train(model, args)

    
    