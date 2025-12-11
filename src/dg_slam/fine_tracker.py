import torch
import numpy as np
from dg_slam.gaussian.common import get_tensor_from_camera, get_camera_from_tensor_4x4
from dg_slam.gaussian.loss_utils import l1_loss, ssim
from dg_slam.gaussian.graphics_utils import getProjectionMatrix, focal2fov
from dg_slam.gaussian.gaussian_render import render
import torch.nn.functional as F


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=v.dtype, device=v.device)


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    omega = xi[:3]
    upsilon = xi[3:]
    theta = torch.norm(omega)
    dtype = xi.dtype
    device = xi.device
    if theta.item() < 1e-8:
        R = torch.eye(3, device=device, dtype=dtype) + _skew_symmetric(omega)
        V = torch.eye(3, device=device, dtype=dtype) + 0.5 * _skew_symmetric(omega)
    else:
        axis = omega / theta
        K = _skew_symmetric(axis)
        R = torch.eye(3, device=device, dtype=dtype) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        A = torch.sin(theta) / theta
        B = (1 - torch.cos(theta)) / (theta * theta)
        V = torch.eye(3, device=device, dtype=dtype) + A * K + B * (K @ K)
    t = V @ upsilon
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class FineTracker:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        H: int,
        W: int,
        tracking_iterations: int = 20,
        lambda_rgb: float = 0.9,
        lambda_ssim: float = 0.2,
        lambda_depth: float = 0.1,
        learning_rate: float = 0.001,
        device: str = 'cuda:0'
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.H = H
        self.W = W
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tracking_iterations = tracking_iterations
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.learning_rate = learning_rate
        self.FoVx = focal2fov(fx, W)
        self.FoVy = focal2fov(fy, H)
        self.znear = 0.01
        self.zfar = 100.0
        pm = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.projection_matrix = pm.to(device=self.device, dtype=torch.float32).contiguous()

    def pose_to_matrix(self, pose_param: torch.Tensor) -> torch.Tensor:
        q = pose_param[:4]
        q = q / (q.norm() + 1e-12)
        t = pose_param[4:7]
        pose_normed = torch.cat([q, t], dim=0)
        return get_camera_from_tensor_4x4(pose_normed)

    def matrix_to_pose(self, matrix: torch.Tensor) -> torch.Tensor:
        return get_tensor_from_camera(matrix, Tquad=True)

    def render_frame(self, gaussians, pose_matrix: torch.Tensor, active_sh_degree: int = 0) -> dict:
        xyz = gaussians.get_xyz()
        features_dc = gaussians.get_features_dc()
        features_rest = gaussians.get_features_rest()
        opacity = gaussians.get_opacity()
        scaling = gaussians.get_scaling()
        rotation = gaussians.get_rotation()
        n_gaussians = int(xyz.shape[0])
        params = {
            'xyz': xyz,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'opacity': opacity,
            'scaling': scaling,
            'rotation': rotation
        }
        for name, param in params.items():
            if int(param.shape[0]) != n_gaussians:
                raise ValueError(f"{name} batch size mismatch: {param.shape[0]} vs {n_gaussians}")
            if torch.isnan(param).any():
                raise ValueError(f"{name} contains NaN")
            if torch.isinf(param).any():
                raise ValueError(f"{name} contains Inf")
            if param.device != xyz.device:
                raise ValueError(f"{name} on wrong device: {param.device} vs {xyz.device}")
            if not param.is_contiguous():
                params[name] = param.contiguous()
        xyz = params['xyz']
        features_dc = params['features_dc']
        features_rest = params['features_rest']
        opacity = params['opacity']
        scaling = params['scaling']
        rotation = params['rotation']
        if pose_matrix.shape != (4, 4):
            raise ValueError(f"Invalid pose shape: {pose_matrix.shape}")
        if torch.isnan(pose_matrix).any() or torch.isinf(pose_matrix).any():
            raise ValueError("Pose contains NaN or Inf")
        w2c = torch.eye(4, device=pose_matrix.device, dtype=pose_matrix.dtype)
        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]
        w2c[:3, :3] = R.T
        w2c[:3, 3] = -R.T @ t
        camera_center = pose_matrix[:3, 3]
        render_pkg = render(
            xyz=xyz,
            features_dc=features_dc,
            features_rest=features_rest,
            opacity=opacity,
            scaling=scaling,
            rotation=rotation,
            active_sh_degree=active_sh_degree,
            max_sh_degree=gaussians.max_sh_degree,
            camera_center=camera_center,
            world_view_transform=w2c.transpose(0, 1),
            projection_matrix=self.projection_matrix,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            image_height=self.H,
            image_width=self.W,
            scaling_modifier=1.0
        )
        return {
            'rgb': render_pkg['render'],
            'depth': render_pkg['depth'],
            'opacity': render_pkg['acc'],
            'visibility_filter': render_pkg['visibility_filter']
        }

    def compute_tracking_loss(
        self,
        rendered_rgb: torch.Tensor,
        rendered_depth: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        motion_mask: torch.Tensor,
        lambda_rgb: float = 0.9,
        lambda_ssim: float = 0.2,
        lambda_depth: float = 0.1
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute masked photometric + depth loss for a single frame.
        Handles batch and single-frame inputs.
        """
        device = rendered_rgb.device

        # Ensure all tensors are 4D (B,C,H,W) for RGB, 3D (B,H,W) for depth/mask
        if rendered_rgb.dim() == 3:
            rendered_rgb = rendered_rgb.unsqueeze(0)
        if gt_rgb.dim() == 3:
            gt_rgb = gt_rgb.unsqueeze(0)
        if rendered_depth.dim() == 2:
            rendered_depth = rendered_depth.unsqueeze(0)
        if gt_depth.dim() == 2:
            gt_depth = gt_depth.unsqueeze(0)
        if motion_mask.dim() == 2:
            motion_mask = motion_mask.unsqueeze(0)

        motion_mask = motion_mask.bool().to(device)

        # Compute RGB loss
        rgb_loss = F.l1_loss(rendered_rgb * motion_mask.unsqueeze(1),
                             gt_rgb * motion_mask.unsqueeze(1))

        # SSIM loss placeholder (can use pytorch_ssim or kornia)
        ssim_loss = 1.0 - self.ssim_loss_fn(rendered_rgb * motion_mask.unsqueeze(1),
                                            gt_rgb * motion_mask.unsqueeze(1))

        # Depth loss (only valid pixels)
        valid_depth = (gt_depth > 0) & motion_mask
        if valid_depth.sum() > 0:
            depth_loss = F.l1_loss(rendered_depth[valid_depth], gt_depth[valid_depth])
        else:
            depth_loss = torch.tensor(0.0, device=device)

        total_loss = lambda_rgb * rgb_loss + lambda_ssim * ssim_loss + lambda_depth * depth_loss
        loss_dict = {
            'total': total_loss.item(),
            'rgb': rgb_loss.item(),
            'ssim': ssim_loss.item(),
            'depth': depth_loss.item()
        }

        return total_loss, loss_dict

    def ssim_loss_fn(self, x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        Simple SSIM approximation for RGB images.
        Expects shape (B,3,H,W), returns mean SSIM over batch.
        """
        # Mean and variance
        mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
        mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)
        sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=window_size // 2) - mu_x**2
        sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=window_size // 2) - mu_y**2
        sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size // 2) - mu_x * mu_y

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
            ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        return ssim_map.mean()

    def track_frame(
        self,
        frame: dict,
        coarse_pose: np.ndarray,
        motion_mask: np.ndarray,
        gaussians,
        verbose: bool = False
    ) -> tuple[np.ndarray, dict]:

        device = self.device

        # --- Convert frame data to tensors ---
        gt_rgb = torch.as_tensor(frame['rgb'], dtype=torch.float32, device=device)
        gt_depth = torch.as_tensor(frame['depth'], dtype=torch.float32, device=device)

        # Ensure batch and channel dims
        if gt_rgb.ndim == 3:  # HWC -> NCHW
            gt_rgb = gt_rgb.permute(2, 0, 1).unsqueeze(0) if gt_rgb.shape[2] == 3 else gt_rgb.unsqueeze(0)
        if gt_depth.ndim == 2:
            gt_depth = gt_depth.unsqueeze(0)
        motion_mask = torch.as_tensor(motion_mask, dtype=torch.bool, device=device)
        if motion_mask.ndim == 2:
            motion_mask = motion_mask.unsqueeze(0)

        # --- Initialize pose tensor ---
        pose = torch.as_tensor(coarse_pose, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam([pose], lr=self.learning_rate)

        # --- Optimization loop ---
        for iter_idx in range(self.tracking_iterations):

            optimizer.zero_grad()

            # Render scene from current pose using FineTracker's render_frame
            render_pkg = self.render_frame(gaussians, pose)
            rendered_rgb = render_pkg['rgb']
            rendered_depth = render_pkg['depth']

            # Compute tracking loss
            total_loss, loss_dict = self.compute_tracking_loss(
                rendered_rgb, rendered_depth, gt_rgb, gt_depth, motion_mask
            )

            # Backprop
            total_loss.backward()
            optimizer.step()

            if verbose and (iter_idx % 5 == 0 or iter_idx == self.tracking_iterations - 1):
                print(f" Iter {iter_idx + 1}/{self.tracking_iterations}: "
                      f"Total loss = {loss_dict['total']:.6f}")

        # After optimization, return final pose as numpy array
        refined_pose = pose.detach().cpu().numpy()
        return refined_pose, loss_dict
