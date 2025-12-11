"""
Fine-grained camera tracking using Gaussian Splatting.
Based on DG-SLAM's tracking and mapping approach.
"""
import torch
import numpy as np
from dg_slam.gaussian.common import (
    get_camera_from_tensor_4x4, get_tensor_from_camera
)
from dg_slam.gaussian.loss_utils import l1_loss, ssim
from dg_slam.gaussian.graphics_utils import getProjectionMatrix, focal2fov
from dg_slam.gaussian.gaussian_render import render


class FineTracker:
    """
    Fine-grained camera pose tracking using Gaussian Splatting.
    Implements the tracking stage from DG-SLAM Section 3.3.
    """

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
        """
        Initialize fine tracker.

        Args:
            fx, fy, cx, cy: Camera intrinsics
            H, W: Image height and width
            tracking_iterations: Number of optimization iterations
            lambda_rgb: Weight for RGB loss (λ1 in paper)
            lambda_ssim: Weight for SSIM loss (λ2 in paper)
            lambda_depth: Weight for depth loss (λ3 in paper)
            learning_rate: Learning rate for pose optimization
            device: Computing device
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.H = H
        self.W = W
        self.device = device

        # Optimization parameters
        self.tracking_iterations = tracking_iterations
        self.lambda_rgb = lambda_rgb
        self.lambda_ssim = lambda_ssim
        self.lambda_depth = lambda_depth
        self.learning_rate = learning_rate

        # Precompute projection matrix
        self.FoVx = focal2fov(fx, W)
        self.FoVy = focal2fov(fy, H)
        self.znear = 0.01
        self.zfar = 100.0

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy
        ).transpose(0, 1).to(device)

    def pose_to_matrix(self, pose_param: torch.Tensor) -> torch.Tensor:
        """
        Convert pose parameters to 4x4 transformation matrix.

        Args:
            pose_param: (7,) tensor [qw, qx, qy, qz, tx, ty, tz]

        Returns:
            4x4 transformation matrix
        """
        return get_camera_from_tensor_4x4(pose_param)

    def matrix_to_pose(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert 4x4 transformation matrix to pose parameters.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            (7,) tensor [qw, qx, qy, qz, tx, ty, tz]
        """
        return get_tensor_from_camera(matrix, Tquad=False)

    def render_frame(
        self,
        gaussians,
        pose_matrix: torch.Tensor,
        active_sh_degree: int = 0
    ) -> dict[str, torch.Tensor]:
        """Render with extensive validation."""

        print("    Validating Gaussians before render...")

        # Get all parameters
        xyz = gaussians.get_xyz()
        features_dc = gaussians.get_features_dc()
        features_rest = gaussians.get_features_rest()
        opacity = gaussians.get_opacity()
        scaling = gaussians.get_scaling()
        rotation = gaussians.get_rotation()

        n_gaussians = xyz.shape[0]
        print(f"      Rendering {n_gaussians} Gaussians")

        # Check each parameter
        params = {
            'xyz': xyz,
            'features_dc': features_dc,
            'features_rest': features_rest,
            'opacity': opacity,
            'scaling': scaling,
            'rotation': rotation
        }

        for name, param in params.items():
            # Shape check
            if param.shape[0] != n_gaussians:
                raise ValueError(f"{name} batch size mismatch: {param.shape[0]} vs {n_gaussians}")

            # NaN check
            if torch.isnan(param).any():
                n_nan = torch.isnan(param).sum().item()
                raise ValueError(f"{name} contains {n_nan} NaN values!")

            # Inf check
            if torch.isinf(param).any():
                n_inf = torch.isinf(param).sum().item()
                raise ValueError(f"{name} contains {n_inf} Inf values!")

            # Device check
            if param.device != xyz.device:
                raise ValueError(f"{name} on wrong device: {param.device} vs {xyz.device}")

            # Contiguity check
            if not param.is_contiguous():
                print(f"      WARNING: {name} not contiguous, making contiguous")
                params[name] = param.contiguous()

        # Print ranges
        print(f"      xyz: [{xyz.min():.3f}, {xyz.max():.3f}]")
        print(f"      opacity: [{opacity.min():.6f}, {opacity.max():.6f}]")
        print(f"      scaling: [{scaling.min():.6f}, {scaling.max():.6f}]")
        print(f"      rotation norm: [{torch.norm(rotation, dim=1).min():.6f}, {torch.norm(rotation, dim=1).max():.6f}]")

        # Check for degenerate Gaussians
        if (scaling < 1e-7).any():
            n_tiny = (scaling < 1e-7).sum().item()
            print(f"      WARNING: {n_tiny} scaling values < 1e-7 (may cause issues)")

        if (opacity < 0.001).any():
            n_transparent = (opacity < 0.001).sum().item()
            print(f"      INFO: {n_transparent} nearly transparent Gaussians")

        # Validate pose
        print(f"    Validating pose...")
        if pose_matrix.shape != (4, 4):
            raise ValueError(f"Invalid pose shape: {pose_matrix.shape}")

        if torch.isnan(pose_matrix).any() or torch.isinf(pose_matrix).any():
            raise ValueError("Pose contains NaN or Inf")

        # Convert c2w to w2c
        w2c = torch.inverse(pose_matrix)
        world_view_transform = w2c.transpose(0, 1)
        camera_center = pose_matrix[:3, 3]

        print(f"      Camera center: [{camera_center[0]:.3f}, {camera_center[1]:.3f}, {camera_center[2]:.3f}]")

        # Ensure everything is contiguous
        xyz = params['xyz'].contiguous()
        features_dc = params['features_dc'].contiguous()
        features_rest = params['features_rest'].contiguous()
        opacity = params['opacity'].contiguous()
        scaling = params['scaling'].contiguous()
        rotation = params['rotation'].contiguous()

        print(f"    Calling rasterizer...")

        try:
            # Render
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
                world_view_transform=world_view_transform,
                projection_matrix=self.projection_matrix,
                FoVx=self.FoVx,
                FoVy=self.FoVy,
                image_height=self.H,
                image_width=self.W,
                scaling_modifier=1.0
            )

            print(f"    ✓ Render successful")

            return {
                'rgb': render_pkg['render'],
                'depth': render_pkg['depth'],
                'opacity': render_pkg['acc'],
                'visibility_filter': render_pkg['visibility_filter']
            }

        except Exception as e:
            print(f"    ✗ Rasterizer error: {type(e).__name__}")
            print(f"    Error message: {str(e)[:500]}")

            # Print debug info
            print(f"\n    Debug info:")
            print(f"      Device: {xyz.device}")
            print(f"      Image size: {self.H}x{self.W}")
            print(f"      Projection matrix device: {self.projection_matrix.device}")
            print(f"      All parameters on same device: {all(p.device == xyz.device for p in params.values())}")

            raise

    def compute_tracking_loss(
        self,
        rendered_rgb: torch.Tensor,
        rendered_depth: torch.Tensor,
        gt_rgb: torch.Tensor,
        gt_depth: torch.Tensor,
        motion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute tracking loss using Eq. 10 from the paper.

        Args:
            rendered_rgb: (3, H, W) rendered RGB image
            rendered_depth: (H, W) rendered depth map
            gt_rgb: (3, H, W) ground truth RGB image
            gt_depth: (H, W) ground truth depth map
            motion_mask: (H, W) static region mask (True = static)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        # Apply motion mask to focus on static regions
        mask = motion_mask.float()

        # RGB L1 loss
        rgb_loss = l1_loss(rendered_rgb * mask, gt_rgb * mask)

        # SSIM loss (computed on full image for structural comparison)
        ssim_loss = 1.0 - ssim(
            rendered_rgb.unsqueeze(0),
            gt_rgb.unsqueeze(0)
        )

        # Depth loss (only where depth is valid and in static regions)
        valid_depth = (gt_depth > 0) & motion_mask
        if valid_depth.sum() > 0:
            depth_loss = l1_loss(
                rendered_depth[valid_depth],
                gt_depth[valid_depth]
            )
        else:
            depth_loss = torch.tensor(0.0, device=self.device)

        # Combined loss (Eq. 10)
        total_loss = (
            self.lambda_rgb * rgb_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_depth * depth_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'rgb': rgb_loss.item(),
            'ssim': ssim_loss.item(),
            'depth': depth_loss.item()
        }

        return total_loss, loss_dict

    def track_frame(
        self,
        frame: dict,
        coarse_pose: np.ndarray,
        motion_mask: np.ndarray,
        gaussians,
        verbose: bool = False
    ) -> tuple[np.ndarray, dict]:
        """
        Track a single frame by refining the coarse pose.

        Args:
            frame: Frame dictionary with 'rgb' and 'depth'
            coarse_pose: (4, 4) initial pose from coarse tracking
            motion_mask: (H, W) boolean mask (True = static)
            gaussians: GaussianModel instance
            verbose: Whether to print progress

        Returns:
            refined_pose: (4, 4) refined camera pose
            tracking_info: Dictionary with tracking statistics
        """
        # Convert inputs to tensors
        gt_rgb = torch.from_numpy(frame['rgb']).to(self.device).float()
        if gt_rgb.max() > 1.0:
            gt_rgb = gt_rgb / 255.0
        gt_rgb = gt_rgb.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)

        gt_depth = torch.from_numpy(frame['depth']).to(self.device).float()
        if 'depth_scale' in frame:
            gt_depth = gt_depth / frame['depth_scale']
        else:
            gt_depth = gt_depth / 5000.0  # Default TUM scale

        motion_mask_tensor = torch.from_numpy(motion_mask).to(self.device)

        # Initialize pose from coarse estimate
        coarse_pose_tensor = torch.from_numpy(coarse_pose).to(self.device).float()
        pose_param = self.matrix_to_pose(coarse_pose_tensor)
        pose_param = pose_param.detach().requires_grad_(True)

        # Setup optimizer
        optimizer = torch.optim.Adam([pose_param], lr=self.learning_rate)

        # Track loss history
        loss_history = []

        # Optimization loop
        for iter_idx in range(self.tracking_iterations):
            optimizer.zero_grad()

            # Convert pose parameters to matrix
            current_pose = self.pose_to_matrix(pose_param)

            # Render frame
            render_output = self.render_frame(
                gaussians=gaussians,
                pose_matrix=current_pose,
                active_sh_degree=0
            )

            rendered_rgb = render_output['rgb']
            rendered_depth = render_output['depth']

            # Compute loss
            loss, loss_dict = self.compute_tracking_loss(
                rendered_rgb=rendered_rgb,
                rendered_depth=rendered_depth,
                gt_rgb=gt_rgb,
                gt_depth=gt_depth,
                motion_mask=motion_mask_tensor
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_history.append(loss_dict)

            if verbose and (iter_idx + 1) % 5 == 0:
                print(f"    Iter {iter_idx + 1}/{self.tracking_iterations}: "
                      f"Loss={loss_dict['total']:.6f} "
                      f"(RGB={loss_dict['rgb']:.4f}, "
                      f"SSIM={loss_dict['ssim']:.4f}, "
                      f"Depth={loss_dict['depth']:.4f})")

        # Extract final refined pose
        with torch.no_grad():
            refined_pose_matrix = self.pose_to_matrix(pose_param)
            refined_pose = refined_pose_matrix.cpu().numpy()

        # Compile tracking info
        tracking_info = {
            'initial_pose': coarse_pose,
            'refined_pose': refined_pose,
            'loss_history': loss_history,
            'final_loss': loss_history[-1],
            'pose_delta': np.linalg.norm(refined_pose - coarse_pose)
        }

        return refined_pose, tracking_info
