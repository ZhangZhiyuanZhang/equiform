import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from lightly.loss import NTXentLoss
from pytorch3d.ops import knn_points, sample_farthest_points
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
)
from termcolor import cprint
from typing import Dict, List, Optional, Tuple, Type

from canonical_policy.model.vision.canonical_utils.agg_encoder import AggEncoder
from canonical_policy.model.vision.canonical_utils.utils import (
    construct_rotation_matrix,
    extract_z_rotation,
)
from canonical_policy.model.vision.canonical_utils.vec_pointnet import (
    VN_Regressor,
    VecPointNet,
)


class SequentialGeometricUpdate(nn.Module):
    """
    Geometry-aware point cloud refinement using xyz only.

    Pipeline:
        1. KNN search for local neighborhoods
        2. Normal estimation from local covariance
        3. Normal-direction correction to project points back to the surface
        4. FPS downsampling
        5. Tangential smoothing in the local tangent plane
    """

    def __init__(self, k: int = 16, num_fps: int = 256):
        super().__init__()
        self.k = k
        self.num_fps = num_fps

    @staticmethod
    def estimate_normals_from_knn(neighbors: torch.Tensor) -> torch.Tensor:
        """
        Estimate normals from local neighborhoods.

        Args:
            neighbors: [B, N, K, 3]

        Returns:
            normals: [B, N, 3]
        """
        k = neighbors.shape[2]
        centered = neighbors - neighbors.mean(dim=2, keepdim=True)
        cov = torch.matmul(centered.transpose(-1, -2), centered) / max(k - 1, 1)
        _, eigvecs = torch.linalg.eigh(cov)
        normals = F.normalize(eigvecs[..., 0], dim=-1)
        return normals

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [B, N, 3]

        Returns:
            xyz_updated: [B, num_fps, 3]
        """
        batch_size, _, _ = xyz.shape
        device = xyz.device
        batch_idx = torch.arange(batch_size, device=device)[:, None, None]

        # Step 1: KNN on original points
        knn_out = knn_points(xyz, xyz, K=self.k, return_nn=True)
        neighbors = knn_out.knn                  # [B, N, K, 3]
        neighbor_idx = knn_out.idx               # [B, N, K]

        # Step 2: Estimate normals
        normals = self.estimate_normals_from_knn(neighbors)

        # Step 3: Normal-direction correction
        local_mean = neighbors.mean(dim=2)       # [B, N, 3]
        neighbor_normals = normals[batch_idx, neighbor_idx, :]   # [B, N, K, 3]
        mean_normal = F.normalize(neighbor_normals.mean(dim=2), dim=-1)  # [B, N, 3]

        delta = xyz - local_mean
        n = mean_normal.unsqueeze(-1)            # [B, N, 3, 1]
        proj_normal = n @ n.transpose(-1, -2)    # [B, N, 3, 3]
        delta_normal = proj_normal @ delta.unsqueeze(-1)
        xyz_normal_updated = xyz - delta_normal.squeeze(-1)

        # Step 4: FPS downsampling
        xyz_fps, _ = sample_farthest_points(xyz_normal_updated, K=self.num_fps)

        # Step 5: Tangential smoothing
        knn_out_fps = knn_points(xyz_fps, xyz_normal_updated, K=self.k, return_nn=True)
        neighbors_fps = knn_out_fps.knn
        neighbor_idx_fps = knn_out_fps.idx

        local_mean_fps = neighbors_fps.mean(dim=2)
        neighbor_normals_fps = mean_normal[batch_idx, neighbor_idx_fps, :]
        mean_normal_fps = F.normalize(neighbor_normals_fps.mean(dim=2), dim=-1)

        delta_fps = xyz_fps - local_mean_fps
        n_fps = mean_normal_fps.unsqueeze(-1)
        proj_normal_fps = n_fps @ n_fps.transpose(-1, -2)
        proj_tangent_fps = torch.eye(3, device=device).view(1, 1, 3, 3) - proj_normal_fps

        delta_tangent = proj_tangent_fps @ delta_fps.unsqueeze(-1)
        xyz_updated = xyz_fps - delta_tangent.squeeze(-1)

        return xyz_updated


def pointcloud_augment(
    x: torch.Tensor,
    rot_angle_range: float = np.pi,
    jitter_std: float = 0.05,
    drop_ratio: float = 0.1,
    insert_ratio: float = 0.1,
    crop_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentation to a point cloud:
        - random SO(2) rotation around z-axis
        - Gaussian jitter
        - random dropout
        - optional spatial crop
        - random point duplication

    Args:
        x: [B, N, 3]

    Returns:
        x_aug: [B, N, 3]
        R: [B, 3, 3] applied rotation matrix
    """
    batch_size, num_points, _ = x.shape
    device = x.device

    # Random SO(2) rotation around z-axis
    angles = (torch.rand(batch_size, device=device) * 2 - 1) * (rot_angle_range / 2)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    R = torch.stack(
        [
            torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)], dim=-1),
            torch.stack([sin_a,  cos_a, torch.zeros_like(cos_a)], dim=-1),
            torch.stack([torch.zeros_like(cos_a), torch.zeros_like(cos_a), torch.ones_like(cos_a)], dim=-1),
        ],
        dim=-2,
    )  # [B, 3, 3]

    x_rot = torch.bmm(x, R.transpose(1, 2))

    # Gaussian jitter
    x_aug = x_rot + torch.randn_like(x_rot) * jitter_std

    # Random dropout
    keep_mask = (torch.rand(batch_size, num_points, device=device) > drop_ratio).unsqueeze(-1).float()
    x_aug = x_aug * keep_mask

    # Spatial crop
    if crop_ratio > 0.0:
        xyz_min = x_aug.amin(dim=1, keepdim=True)
        xyz_max = x_aug.amax(dim=1, keepdim=True)

        crop_center = xyz_min + (torch.rand_like(xyz_min) - 0.5) * (xyz_max - xyz_min)
        crop_size = (xyz_max - xyz_min) * crop_ratio
        lower = crop_center - crop_size / 2
        upper = crop_center + crop_size / 2

        crop_mask = ~((x_aug >= lower) & (x_aug <= upper)).all(dim=-1).unsqueeze(-1)
        x_aug = x_aug * crop_mask.float()

    # Random point duplication
    insert_n = int(insert_ratio * num_points)
    if insert_n > 0:
        insert_idx = torch.randint(0, num_points, (batch_size, insert_n), device=device)
        inserted_points = torch.gather(
            x_aug,
            1,
            insert_idx.unsqueeze(-1).expand(-1, -1, 3),
        )
        x_aug = torch.cat([x_aug, inserted_points], dim=1)

        shuffle_idx = (
            torch.rand(batch_size, x_aug.shape[1], device=device)
            .argsort(dim=1)[:, :num_points]
            .unsqueeze(-1)
            .expand(-1, -1, 3)
        )
        x_aug = torch.gather(x_aug, 1, shuffle_idx)

    return x_aug, R


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Build an MLP as a list of modules.
    """
    layers: List[nn.Module] = []

    prev_dim = input_dim
    for hidden_dim in net_arch:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_fn())
        prev_dim = hidden_dim

    if output_dim > 0:
        layers.append(nn.Linear(prev_dim, output_dim))

    if squash_output:
        layers.append(nn.Tanh())

    return layers


class CanonicalEncoder(nn.Module):
    def __init__(
        self,
        observation_space: Dict,
        canonical_encoder_cfg: Optional[Dict],
        out_channel: int = 256,
        state_mlp_size: Tuple[int, ...] = (64, 64),
        state_mlp_activation_fn: Type[nn.Module] = nn.ReLU,
        use_pc_color: bool = False,
        state_keys: List[str] = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        point_cloud_key: str = "point_cloud",
        n_obs_steps: int = 2,
    ):
        super().__init__()

        self.state_keys = state_keys
        self.point_cloud_key = point_cloud_key
        self.n_obs_steps = n_obs_steps
        self.use_pc_color = use_pc_color
        self.obs_dim = out_channel

        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_size = sum(observation_space[key][0] for key in self.state_keys)

        cprint(f"[CanonicalEncoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[CanonicalEncoder] state dim: {self.state_size}", "yellow")

        self.num_points = canonical_encoder_cfg.num_points
        self.ksize = canonical_encoder_cfg.neighbor_num
        self.rot_hidden_dim = canonical_encoder_cfg.rot_hidden_dim
        self.rot_layers = canonical_encoder_cfg.rot_layers
        self.use_contra = canonical_encoder_cfg.use_contra
        self.use_geo = canonical_encoder_cfg.use_geo

        self.xyz_extractor = AggEncoder(hidden_dim=self.obs_dim, ksize=self.ksize)
        self.equiv_extractor = VecPointNet(
            h_dim=self.rot_hidden_dim,
            c_dim=self.rot_hidden_dim,
            num_layers=self.rot_layers,
            ksize=self.ksize,
        )
        self.rot_predictor = VN_Regressor(pc_feat_dim=self.rot_hidden_dim)
        self.geometric_update = SequentialGeometricUpdate(k=16, num_fps=self.num_points)

        if len(state_mlp_size) == 0:
            raise ValueError("state_mlp_size must not be empty")

        if len(state_mlp_size) == 1:
            hidden_dims = []
            state_out_dim = state_mlp_size[0]
        else:
            hidden_dims = list(state_mlp_size[:-1])
            state_out_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(
            *create_mlp(
                input_dim=self.state_size,
                output_dim=state_out_dim,
                net_arch=hidden_dims,
                activation_fn=state_mlp_activation_fn,
            )
        )

        self.n_output_channels = self.obs_dim + state_out_dim
        cprint(f"[CanonicalEncoder] output dim: {self.n_output_channels}", "red")

        self.ntxent_loss = NTXentLoss(temperature=0.1)

    def _preprocess_point_cloud(self, points_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample / refine the point cloud and compute the canonical center.

        Args:
            points_xyz: [B*To, N, 3]

        Returns:
            centered_points: [B*To, M, 3]
            points_center: [B*To, 1, 3]
        """
        if self.use_geo:
            points_xyz_updated = self.geometric_update(points_xyz)
        else:
            points_xyz_updated, _ = sample_farthest_points(points_xyz, K=self.num_points)

        points_xyz_t0 = rearrange(
            points_xyz_updated, "(B To) N C -> B To N C", To=self.n_obs_steps
        )[:, 0]  # [B, N, 3]

        points_center = points_xyz_t0.mean(dim=1, keepdim=True)  # [B, 1, 3]
        points_center = (
            points_center.repeat(1, self.n_obs_steps, 1)
            .reshape(-1, 3)
            .unsqueeze(1)
        )  # [B*To, 1, 3]

        centered_points = points_xyz_updated - points_center
        return centered_points, points_center

    def _predict_canonical_rotation(self, centered_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict canonical SO(2) rotation from the first observation step.

        Args:
            centered_points: [B*To, N, 3]

        Returns:
            est_rot: [B*To, 3, 3]
            est_quat: [B*To, 4]
        """
        centered_points_t0 = rearrange(
            centered_points, "(B To) N C -> B To N C", To=self.n_obs_steps
        )[:, 0]  # [B, N, 3]

        equiv_feat = self.equiv_extractor(centered_points_t0)
        v1, v2 = self.rot_predictor(equiv_feat)

        rot_mat = construct_rotation_matrix(v1, v2)  # [B, 3, 3]
        rot_mat = (
            rot_mat.unsqueeze(1)
            .repeat(1, self.n_obs_steps, 1, 1)
            .reshape(-1, 3, 3)
        )  # [B*To, 3, 3]

        est_rot = extract_z_rotation(rot_mat)
        est_quat = matrix_to_quaternion(est_rot)
        return est_rot, est_quat

    def _compute_contrastive_loss(self, centered_points: torch.Tensor) -> torch.Tensor:
        """
        Compute SO(2)-equivariant contrastive consistency loss
        using the first observation step only.
        """
        centered_points_t0 = rearrange(
            centered_points, "(B To) N C -> B To N C", To=self.n_obs_steps
        )[:, 0]  # [B, N, 3]

        pcl_aug, R_aug = pointcloud_augment(
            centered_points_t0,
            rot_angle_range=np.pi,
            jitter_std=0.1,
            drop_ratio=0.1,
            insert_ratio=0.1,
            crop_ratio=0.1,
        )

        equiv_feat = self.equiv_extractor(centered_points_t0)      # [B, D, 3]
        equiv_feat_aug = self.equiv_extractor(pcl_aug)             # [B, D, 3]
        equiv_feat_rot = torch.bmm(equiv_feat, R_aug.transpose(1, 2))

        inv_feat_aug = F.normalize(
            equiv_feat_aug.reshape(-1, 3 * self.rot_hidden_dim), dim=-1
        )
        inv_feat_rot = F.normalize(
            equiv_feat_rot.reshape(-1, 3 * self.rot_hidden_dim), dim=-1
        )

        return self.ntxent_loss(inv_feat_aug, inv_feat_rot)

    def forward(self, observations: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            observations: dict containing state and point cloud observations

        Returns:
            A dictionary containing:
                - final_feat: [B*To, D]
                - points_center: [B*To, 1, 3]
                - est_quat: [B*To, 4]
                - contrastive_equiv: optional scalar loss
        """
        state_pos = observations[self.state_keys[0]]                      # [B*To, 3]
        state_quat = observations[self.state_keys[1]][:, [3, 0, 1, 2]]   # xyzw -> wxyz
        state_gripper = observations[self.state_keys[2]]                  # [B*To, G]

        points = observations[self.point_cloud_key]                       # [B*To, N, C]
        points_xyz = points[..., :3]

        # 1. Preprocess and center point cloud
        centered_points, points_center = self._preprocess_point_cloud(points_xyz)

        # 2. Predict canonical rotation
        est_rot, est_quat = self._predict_canonical_rotation(centered_points)
        est_quat_inv = quaternion_invert(est_quat)

        # 3. Canonicalize state
        canonical_state_pos = quaternion_apply(
            est_quat_inv, state_pos - points_center.squeeze(1)
        )
        canonical_state_quat = quaternion_multiply(est_quat_inv, state_quat)

        # 4. Canonicalize point cloud
        canonical_points = torch.matmul(centered_points, est_rot)

        # 5. Extract features
        xyz_feat = self.xyz_extractor(canonical_points, canonical_points)
        state_feat_input = torch.cat(
            [canonical_state_pos, canonical_state_quat, state_gripper], dim=-1
        )
        state_feat = self.state_mlp(state_feat_input)

        final_feat = torch.cat([xyz_feat, state_feat], dim=-1)

        output = {
            "final_feat": final_feat,
            "points_center": points_center,
            "est_quat": est_quat,
        }

        if self.use_contra:
            output["contrastive_equiv"] = self._compute_contrastive_loss(centered_points)

        return output

    def output_shape(self) -> int:
        return self.n_output_channels