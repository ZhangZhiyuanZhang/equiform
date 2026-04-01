import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import numpy as np
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from einops import rearrange
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_multiply, matrix_to_quaternion
from canonical_policy.model.vision.canonical_utils.vec_pointnet import VecPointNet, VN_Regressor
from canonical_policy.model.vision.canonical_utils.agg_encoder import AggEncoder
from canonical_policy.model.vision.canonical_utils.utils import construct_rotation_matrix, extract_z_rotation
from lightly.loss import NTXentLoss
from pytorch3d.ops import knn_points
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import torch
import math
import os
import numpy as np

class SequentialGeometricUpdate(nn.Module):
    """
    几何一致化模块（仅输入 xyz）:
      1) 一次 KNN -> 用于法向估计 + 法向修正
      2) 法向修正: 将点沿法向压回表面
      3) FPS下采样
      4) 切向平滑: 在更新后的点云上沿切平面靠近局部 group mean
    """
    def __init__(self, k=16, num_fps=256):
        super().__init__()
        self.k = k
        self.num_fps = num_fps

    @staticmethod
    def estimate_normals_from_knn(xyz, neighbors):
        """
        基于已知邻域点集估计法向:
          xyz: [B, N, 3]
          neighbors: [B, N, K, 3]
          返回 normal: [B, N, 3]
        """
        k = neighbors.shape[2]
        X = neighbors - neighbors.mean(dim=2, keepdim=True)  # 去中心化
        C = torch.matmul(X.transpose(-1, -2), X) / (k - 1)   # 协方差矩阵
        eigvals, eigvecs = torch.linalg.eigh(C)
        normals = F.normalize(eigvecs[..., 0], dim=-1)       # 最小特征值方向
        return normals

    def forward(self, xyz):
        """
        Args:
            xyz: [B, N, 3] 原始点云
        Returns:
            xyz_updated: [B, num_fps, 3] 更新后点云
        """
        B, N, _ = xyz.shape
        device = xyz.device
        batch_idx = torch.arange(B, device=device)[:, None, None]

        # ---------- Step 0: 一次性计算 KNN ----------
        knn1 = knn_points(xyz, xyz, K=self.k, return_nn=True)
        neighbors = knn1.knn      # [B, N, K, 3]
        idx = knn1.idx            # [B, N, K]

        # ---------- Step 1: 法向估计 ----------
        normal = self.estimate_normals_from_knn(xyz, neighbors)

        # ---------- Step 2: 法向修正 ----------
        x_mean = neighbors.mean(dim=2)  # 邻域均值
        neighbor_normals = normal[batch_idx, idx, :]  # 邻域法向
        n_mean = F.normalize(neighbor_normals.mean(dim=2), dim=-1)  # 平均法向

        delta = xyz - x_mean
        n = n_mean.unsqueeze(-1)
        Pn = n @ n.transpose(-1, -2)
        delta_corr_n = Pn @ delta.unsqueeze(-1)
        xyz_normal_updated = xyz - delta_corr_n.squeeze(-1)

        # ---------- Step 3: FPS 下采样 ----------
        xyz_fps, fps_idx = sample_farthest_points(xyz_normal_updated, K=self.num_fps)

        # ---------- Step 4: 切向修正 ----------
        knn2 = knn_points(xyz_fps, xyz_normal_updated, K=self.k, return_nn=True)
        neighbors2 = knn2.knn
        idx2 = knn2.idx
        x_mean2 = neighbors2.mean(dim=2)

        neighbor_normals2 = n_mean[batch_idx, idx2, :]
        n_mean2 = F.normalize(neighbor_normals2.mean(dim=2), dim=-1)

        delta2 = xyz_fps - x_mean2
        n2 = n_mean2.unsqueeze(-1)
        Pn2 = n2 @ n2.transpose(-1, -2)
        Pt2 = torch.eye(3, device=device).view(1, 1, 3, 3) - Pn2
        delta_corr_t = Pt2 @ delta2.unsqueeze(-1)
        xyz_updated = xyz_fps - delta_corr_t.squeeze(-1)

        return xyz_updated


def pointcloud_augment(
    x: torch.Tensor,
    rot_angle_range: float = np.pi,
    jitter_std: float = 0.05,
    drop_ratio: float = 0.1,
    insert_ratio: float = 0.1,
    crop_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply SO(2) (z-axis) rotation, Gaussian jitter, dropout, insert, and optional spatial cropping to point cloud.

    Args:
        x: [B, N, 3] torch.Tensor, point cloud (z-up)
        jitter_std: std dev of Gaussian noise
        drop_ratio: ratio of points to randomly drop (zero-out)
        insert_ratio: ratio of points to randomly duplicate
        crop_ratio: ratio of spatial region to crop and set to 0 (e.g., 0.3)

    Returns:
        x_aug: [B, N, 3], augmented point cloud
        R: [B, 3, 3], applied SO(2) rotation matrix
    """
    B, N, _ = x.shape
    device = x.device

    # SO(2) rotation around z-axis
    angles = (torch.rand(B, device=device) * 2 - 1) * (rot_angle_range / 2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    R = torch.stack([
        torch.stack([cos, -sin, torch.zeros_like(cos)], dim=-1),
        torch.stack([sin,  cos, torch.zeros_like(cos)], dim=-1),
        torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)], dim=-1)
    ], dim=-2)  # [B, 3, 3]
    x_rot = torch.bmm(x, R.transpose(1, 2))  # [B, N, 3]

    # Gaussian jitter
    x_jittered = x_rot + torch.randn_like(x_rot) * jitter_std

    # Random dropout
    keep_mask = (torch.rand(B, N, device=device) > drop_ratio).unsqueeze(-1).float()
    x_dropped = x_jittered * keep_mask

    # Spatial cropping
    if crop_ratio > 0.0:
        xyz_min, xyz_max = x_dropped.amin(dim=1, keepdim=True), x_dropped.amax(dim=1, keepdim=True)
        crop_center = xyz_min + (torch.rand_like(xyz_min) - 0.5) * (xyz_max - xyz_min)
        crop_size = (xyz_max - xyz_min) * crop_ratio
        lower, upper = crop_center - crop_size / 2, crop_center + crop_size / 2
        crop_mask = ~( (x_dropped >= lower) & (x_dropped <= upper) ).all(dim=-1).unsqueeze(-1)
        crop_mask = crop_mask.float()
        x_dropped *= crop_mask

    # Random insert
    insert_n = int(insert_ratio * N)
    insert_idx = torch.randint(0, N, (B, insert_n), device=device)
    inserted_points = torch.gather(x_dropped, 1, insert_idx.unsqueeze(-1).expand(-1, -1, 3))

    # Concatenate and crop back to N
    x_aug = torch.cat([x_dropped, inserted_points], dim=1)
    idx = torch.rand(B, x_aug.shape[1], device=device).argsort(dim=1)[:, :N].unsqueeze(-1).expand(-1, -1, 3)
    x_aug = torch.gather(x_aug, 1, idx)

    return x_aug, R


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class CanonicalEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict,
                 canonical_encoder_cfg: Optional[Dict],
                 out_channel=256,
                 state_mlp_size=(64, 64),
                 state_mlp_activation_fn=nn.ReLU,
                 use_pc_color=False,
                 state_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
                 point_cloud_key='point_cloud',
                 n_obs_steps=2,
                 ):
        super().__init__()
        self.state_keys = state_keys
        self.point_cloud_key = point_cloud_key
        self.n_obs_steps = n_obs_steps

        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_size = sum([observation_space[key][0] for key in self.state_keys])

        cprint(f"[CanonicalEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CanonicalEncoder] state shape: {self.state_size}", "yellow")

        self.use_pc_color = use_pc_color
        self.obs_dim = out_channel

        self.num_points = canonical_encoder_cfg.num_points  
        self.ksize = canonical_encoder_cfg.neighbor_num
        self.rot_hidden_dim = canonical_encoder_cfg.rot_hidden_dim
        self.rot_layers = canonical_encoder_cfg.rot_layers
        self.use_contra = canonical_encoder_cfg.use_contra
        self.use_geo = canonical_encoder_cfg.use_geo

        cprint(f"[CanonicalEncoder] use_contra: {self.use_contra}", "yellow")
        cprint(f"[CanonicalEncoder] use_geo: {self.use_geo}", "yellow")

        self.xyz_extractor = AggEncoder(hidden_dim=self.obs_dim, ksize=self.ksize)
        self.equiv_extractor = VecPointNet(h_dim=self.rot_hidden_dim, c_dim=self.rot_hidden_dim, num_layers=self.rot_layers, ksize=self.ksize)
        self.rot_predictor = VN_Regressor(pc_feat_dim=self.rot_hidden_dim)
        self.update = SequentialGeometricUpdate(k=11)

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        state_dim = state_mlp_size[-1]
        self.state_mlp = nn.Sequential(*create_mlp(self.state_size, state_dim, net_arch, state_mlp_activation_fn))

        self.n_output_channels = self.obs_dim + state_dim
        cprint(f"[CanonicalEncoder] output dim: {self.n_output_channels}", "red")

        self.ntxent_loss = NTXentLoss(temperature=0.1)

    def forward(self, observations: Dict) -> torch.Tensor:

        # extract state
        state_pos = observations[self.state_keys[0]]  # [BTo, 3]
        state_quat = observations[self.state_keys[1]][:, [3, 0, 1, 2]]  # [BTo, 4]  ijkw->wijk
        state_gripper = observations[self.state_keys[2]]  # [BTo, 2]

        # extract point cloud
        points = observations[self.point_cloud_key]  # [B*To, N, 6]
        points_xyz = points[..., :3]

        # 1. Center the point cloud
        if self.use_geo:
            points_xyz_update = self.update(points_xyz)  # [BTo, N, 3]
        else:
            points_xyz_update, _ = sample_farthest_points(points_xyz, K=self.num_points)

        points_xyz_update_t0 = rearrange(points_xyz_update, '(B To) N C -> B To N C', To=self.n_obs_steps)[:, 0, :, :]  # [B, N, 3]
        points_center = points_xyz_update_t0.mean(dim=1, keepdim=True)  # [B, 1, 3]
        points_center = points_center.repeat(1, self.n_obs_steps, 1).reshape(-1, 3).unsqueeze(1)  # [BTo, 1, 3]
        
        input_pcl_update = points_xyz_update - points_center  # [BTo, G, 3]  centered point cloud

        input_pcl_update_t0 = rearrange(input_pcl_update, '(B To) N C -> B To N C', To=self.n_obs_steps)[:, 0, :, :]  # [B, N, 3]
        
        # 2. Using SO3-equivariant Network to predict rotation
        equiv_feat = self.equiv_extractor(input_pcl_update_t0)  # [B, 32, 3], [B, 64]
        v1, v2 = self.rot_predictor(equiv_feat)
        rot_mat = construct_rotation_matrix(v1, v2)  # [B, 3, 3]  SO3
        rot_mat = rot_mat.unsqueeze(1).repeat(1, self.n_obs_steps, 1, 1).reshape(-1, 3, 3)  # [BTo, 3, 3]
        est_rot = extract_z_rotation(rot_mat)  # [BTo, 3, 3]  SO2
        
        # Use quaternion to represent the rotation
        est_quat = matrix_to_quaternion(est_rot)    # [BTo, 4]  wijk
        est_quat_inv = quaternion_invert(est_quat)      # [BTo, 4]  wijk

        # 3. SE3-inverse for state
        state_pos = quaternion_apply(est_quat_inv, state_pos - points_center.squeeze(1))   # [BTo, 3]
        state_quat = quaternion_multiply(est_quat_inv, state_quat)  # [BTo, 4]  wijk

        # 4. SO3-inverse for point cloud, (R^(-1)*x.T).T = x*R
        rot_input_pcl = torch.matmul(input_pcl_update, est_rot)

        # 5. Extract features for observations
        xyz_feat = self.xyz_extractor(rot_input_pcl, rot_input_pcl)  # [BTo, obs_dim]
        state = torch.cat([state_pos, state_quat, state_gripper], dim=-1)   # [BTo, Ds]
        state_feat = self.state_mlp(state)  # [BTo, state_dim]

        final_feat = torch.cat([xyz_feat, state_feat], dim=-1)  # [BTo, self.n_output_channels]

        ret = {}
        ret['final_feat'] = final_feat
        ret['points_center'] = points_center
        ret['est_quat'] = est_quat

        if self.use_contra:
            pcl_aug, R_aug = pointcloud_augment(input_pcl_update_t0, rot_angle_range=np.pi, jitter_std=0.1, drop_ratio=0.1, insert_ratio=0.1, crop_ratio=0.1)  # [B, N, 3], [B, 3, 3]
            equiv_feat_aug = self.equiv_extractor(pcl_aug)  # [B, D, 3]
            equiv_feat_rot = torch.bmm(equiv_feat, R_aug.transpose(1, 2))  # [B, D, 3]
            inv_feat_aug = F.normalize(equiv_feat_aug.reshape(-1, 3*self.rot_hidden_dim), dim=-1)  # [B, D]
            inv_feat_rot = F.normalize(equiv_feat_rot.reshape(-1, 3*self.rot_hidden_dim), dim=-1)  # [B, D]
            ret['contrastive_equiv'] = self.ntxent_loss(inv_feat_aug, inv_feat_rot)
        return ret

    def output_shape(self):
        return self.n_output_channels