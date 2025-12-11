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
        ssim_loss = 1.0 - ssim_loss_fn(rendered_rgb * motion_mask.unsqueeze(1),
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

    def track_frame(self, frame: dict, coarse_pose: np.ndarray, motion_mask: np.ndarray, gaussians, verbose: bool = False) -> tuple:
        print("TRACKING FRAME NOW")

        def _ensure_tensor(x, name, device=self.device):
            if not isinstance(x, torch.Tensor):
                try:
                    x = torch.as_tensor(x, device=device)
                except Exception as ex:
                    print(f"[BUG] {name} cannot be turned into a tensor: {type(x)} -> {ex}")
                    raise
            # detach and move to device to avoid lazy objects / graph issues
            x = x.detach()
            return x
        gt_rgb = torch.from_numpy(frame['rgb']).to(self.device).float()
        if gt_rgb.max() > 1.0:
            gt_rgb = gt_rgb / 255.0
        gt_rgb = gt_rgb.permute(2, 0, 1)
        gt_depth = torch.from_numpy(frame['depth']).to(self.device).float()
        if 'depth_scale' in frame:
            gt_depth = gt_depth / float(frame['depth_scale'])
        else:
            gt_depth = gt_depth / 5000.0
        motion_mask_tensor = torch.from_numpy(motion_mask).to(self.device)
        coarse_pose_tensor = torch.from_numpy(coarse_pose).to(self.device).float().contiguous()
        initial_pose_matrix = coarse_pose_tensor
        delta = torch.zeros(6, device=self.device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.learning_rate)
        loss_history = []
        for iter_idx in range(self.tracking_iterations):
            optimizer.zero_grad()
            exp_mat = se3_exp(delta)
            current_pose_matrix = initial_pose_matrix @ exp_mat
            render_output = self.render_frame(gaussians=gaussians, pose_matrix=current_pose_matrix, active_sh_degree=0)
            rendered_rgb = render_output['rgb']
            rendered_depth = render_output['depth']

            try:
                rendered_rgb_t = _ensure_tensor(rendered_rgb, "rendered_rgb")
                rendered_depth_t = _ensure_tensor(rendered_depth, "rendered_depth")
                gt_rgb_t = _ensure_tensor(gt_rgb, "gt_rgb")
                gt_depth_t = _ensure_tensor(gt_depth, "gt_depth")
                motion_mask_t = _ensure_tensor(motion_mask_tensor, "motion_mask")

                # make mask boolean and place on same device as rendered tensors
                motion_mask_t = motion_mask_t.to(device=rendered_rgb_t.device, dtype=torch.bool)

                # normalize dimensions to batch-first canonical shapes
                if rendered_rgb_t.ndim == 3:
                    rendered_rgb_t = rendered_rgb_t.unsqueeze(0)
                if rendered_depth_t.ndim == 2:
                    rendered_depth_t = rendered_depth_t.unsqueeze(0)
                if gt_rgb_t.ndim == 3:
                    gt_rgb_t = gt_rgb_t.unsqueeze(0)
                if gt_depth_t.ndim == 2:
                    gt_depth_t = gt_depth_t.unsqueeze(0)
                if motion_mask_t.ndim == 2:
                    motion_mask_t = motion_mask_t.unsqueeze(0)

                # final sanity print
                print("compute_tracking_loss call shapes:")
                print("  rendered_rgb:", tuple(rendered_rgb_t.shape), rendered_rgb_t.dtype, rendered_rgb_t.device)
                print("  rendered_depth:", tuple(rendered_depth_t.shape), rendered_depth_t.dtype, rendered_depth_t.device)
                print("  gt_rgb:", tuple(gt_rgb_t.shape), gt_rgb_t.dtype, gt_rgb_t.device)
                print("  gt_depth:", tuple(gt_depth_t.shape), gt_depth_t.dtype, gt_depth_t.device)
                print("  motion_mask:", tuple(motion_mask_t.shape), motion_mask_t.dtype, motion_mask_t.device)

                loss, loss_dict = self.compute_tracking_loss(
                    rendered_rgb=rendered_rgb_t,
                    rendered_depth=rendered_depth_t,
                    gt_rgb=gt_rgb_t,
                    gt_depth=gt_depth_t,
                    motion_mask=motion_mask_t
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)
                optimizer.step()
                with torch.no_grad():
                    delta[3:].clamp_(-5.0, 5.0)
                loss_history.append(loss_dict)
                if verbose and (iter_idx + 1) % 5 == 0:
                    print(
                        f"    Iter {iter_idx + 1}/{self.tracking_iterations}: Loss={loss_dict['total']:.6f} (RGB={loss_dict['rgb']:.4f}, SSIM={loss_dict['ssim']:.4f}, Depth={loss_dict['depth']:.4f})")
                with torch.no_grad():
                    final_exp = se3_exp(delta)
                    refined_pose_matrix = (initial_pose_matrix @ final_exp).cpu().numpy()
                tracking_info = {
                    'initial_pose': coarse_pose,
                    'refined_pose': refined_pose_matrix,
                    'loss_history': loss_history,
                    'final_loss': loss_history[-1] if len(loss_history) > 0 else None,
                    'pose_delta': float(torch.norm(final_exp[:3, 3]).item())
                }
                return refined_pose_matrix, tracking_info

            except Exception as e:
                # extra debug dump if something still fails
                try:
                    types_shapes = {
                        'rendered_rgb': (type(rendered_rgb), getattr(rendered_rgb, 'shape', None)),
                        'rendered_depth': (type(rendered_depth), getattr(rendered_depth, 'shape', None)),
                        'gt_rgb': (type(gt_rgb), getattr(gt_rgb, 'shape', None)),
                        'gt_depth': (type(gt_depth), getattr(gt_depth, 'shape', None)),
                        'motion_mask': (type(motion_mask_tensor), getattr(motion_mask_tensor, 'shape', None))
                    }
                    print("DEBUG: Pre-call raw types & shapes:", types_shapes)
                except Exception:
                    pass
                print("ERROR while preparing compute_tracking_loss ->", repr(e))
            raise
