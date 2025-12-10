import numpy as np
import torch
from dg_slam.gaussian.sh_utils import RGB2SH


class Gaussian:
    def __init__(self, mu, sigma, alpha, feature):
        self.mu = mu            # center (3D)
        self.sigma = sigma      # anisotropic covariance (3-vector for simplicity)
        self.alpha = alpha      # opacity
        self.feature = feature  # e.g. RGB color


class GaussianModel:
    """Scene model of 3D Gaussians for fine alignment."""

    def __init__(self, points, colors, device='cuda:0'):
        # Initialize each point as a Gaussian blob with small sigma and color
        N = len(points)
        self.gaussians = []
        for p, c in zip(points, colors):
            sigma = np.array([0.05, 0.05, 0.05])  # small initial extent
            alpha = 1.0
            gaussian = Gaussian(mu=p, sigma=sigma, alpha=alpha, feature=c)
            self.gaussians.append(gaussian)

        # Initialize PyTorch tensors for Gaussian Splatting
        self._xyz = torch.from_numpy(points).float().to(device)
        self._features_dc = torch.from_numpy(RGB2SH(colors)).float().unsqueeze(1).to(device)  # (N, 1, 3)
        self._features_rest = torch.zeros((N, 15, 3), device=device)  # Higher order SH
        self._scaling = torch.log(torch.ones((N, 3), device=device) * 0.05)  # Log scale
        self._rotation = torch.zeros((N, 4), device=device)  # Quaternion
        self._rotation[:, 0] = 1.0  # Identity rotation (w=1)
        self._opacity = torch.ones((N, 1), device=device) * 0.5  # Logit space

        self.max_sh_degree = 0  # Start with DC component only
        self.active_sh_degree = 0

    def get_xyz(self):
        return self._xyz

    def get_features_dc(self):
        return self._features_dc

    def get_features_rest(self):
        return self._features_rest

    def get_scaling(self):
        return torch.exp(self._scaling)

    def get_rotation(self):
        return self._rotation / (torch.norm(self._rotation, dim=-1, keepdim=True) + 1e-8)

    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def pts_num(self):
        return self._xyz.shape[0]

    def input_pos(self):
        return self._xyz.detach().cpu().numpy()

    def input_rgb(self):
        # Convert SH back to RGB
        from dg_slam.gaussian.sh_utils import SH2RGB
        rgb = SH2RGB(self._features_dc.squeeze(1))
        return rgb.detach().cpu().numpy()


class AdaptiveGaussianManager:
    """
    Manages adaptive Gaussian point addition and pruning.
    Implements Section 3.4 strategies from the paper.
    """

    def __init__(
        self,
        tau_alpha: float = 0.005,
        tau_s1: float = 0.4,
        tau_s2: float = 36.0,
        o_th: float = 0.5
    ):
        self.tau_alpha = tau_alpha  # Opacity threshold
        self.tau_s1 = tau_s1  # Max scale threshold
        self.tau_s2 = tau_s2  # Scale ratio threshold
        self.o_th = o_th  # Accumulated opacity threshold

    def compute_point_density_radius(
        self,
        rgb_image: np.ndarray,
        base_radius: float = 0.05
    ) -> np.ndarray:
        """
        Compute adaptive point density based on color gradient.
        High texture areas get denser points.

        Args:
            rgb_image: (H, W, 3) RGB image
            base_radius: Base radius for low-texture regions

        Returns:
            density_map: (H, W) radius values for each pixel
        """
        # Compute gradient magnitude
        gray = np.mean(rgb_image, axis=2)
        grad_x = np.abs(np.gradient(gray, axis=1))
        grad_y = np.abs(np.gradient(gray, axis=0))
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to [0, 1]
        gradient_mag = gradient_mag / (gradient_mag.max() + 1e-8)

        # Inverse relationship: high gradient -> small radius (dense points)
        density_map = base_radius * (1.0 - 0.8 * gradient_mag)

        return density_map

    def should_add_gaussian(
        self,
        accumulated_opacity: float,
        depth_residual: float,
        depth_residual_threshold: float = 0.1
    ) -> bool:
        """
        Determine if a Gaussian point should be added at this location.

        Args:
            accumulated_opacity: Opacity at this pixel
            depth_residual: Depth error at this pixel
            depth_residual_threshold: Threshold for depth error

        Returns:
            True if should add point
        """
        # Add if under-fitted or large depth error
        return (accumulated_opacity < self.o_th) or (depth_residual > depth_residual_threshold)

    def should_prune_gaussian(
        self,
        alpha: float,
        scale_vector: np.ndarray
    ) -> bool:
        """
        Determine if a Gaussian should be pruned based on Eq. 13.

        Args:
            alpha: Opacity value
            scale_vector: (3,) scale values along each axis

        Returns:
            True if should prune
        """
        max_scale = np.max(scale_vector)
        min_scale = np.min(scale_vector)

        # Better protection against division by zero
        if min_scale < 1e-6:
            scale_ratio = float('inf')
        else:
            scale_ratio = max_scale / min_scale

        # Prune based on three criteria (Eq. 13)
        return (
            alpha < self.tau_alpha or
            max_scale > self.tau_s1 or
            scale_ratio > self.tau_s2
        )

    def get_scaling(self):
        return self._scaling

    def get_rotation(self):
        return self._rotation

    def get_xyz(self):
        return self._xyz

    def get_features_dc(self):
        return self._features_dc

    def get_features_rest(self):
        return self._features_rest

    def get_opacity(self):
        return self._opacity
