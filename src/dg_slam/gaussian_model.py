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
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        pts = np.asarray(points, dtype=np.float32)
        cols = np.asarray(colors, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be (N,3), got {pts.shape}")
        if cols.ndim != 2 or cols.shape[1] != 3:
            raise ValueError(f"colors must be (N,3), got {cols.shape}")

        N = pts.shape[0]
        if N == 0:
            raise ValueError("Cannot initialize GaussianModel with 0 points")
        if cols.shape[0] != N:
            raise ValueError(f"Points ({N}) and colors ({cols.shape[0]}) must have same length")

        print(f"Initializing GaussianModel with {N} points on {self.device}")

        # Convert to torch tensors (explicit dtype, device, contiguity)
        self._xyz = torch.as_tensor(pts, dtype=self.dtype, device=self.device).contiguous()  # (N,3)

        # SH: be explicit about output shape of RGB2SH
        sh_coeffs = RGB2SH(cols)  # make sure this returns (N, sh_dim) not (N,1,3)
        sh_coeffs = np.asarray(sh_coeffs, dtype=np.float32)
        if sh_coeffs.ndim == 3 and sh_coeffs.shape[1] == 1:
            # collapse singleton middle dim if returned that way
            sh_coeffs = sh_coeffs.squeeze(1)

        self._features_dc = torch.as_tensor(sh_coeffs, dtype=self.dtype, device=self.device).contiguous()  # (N, sh_dim)

        # choose sh_rest shape according to max_sh_degree (here we start with 0)
        self.max_sh_degree = 0
        self.active_sh_degree = 0
        sh_rest_dim = 0  # set based on max_sh_degree; if >0, allocate appropriately
        if sh_rest_dim > 0:
            self._features_rest = torch.zeros((N, sh_rest_dim), dtype=self.dtype, device=self.device)
        else:
            self._features_rest = torch.zeros((N, 0), dtype=self.dtype, device=self.device)  # empty

        # scaling stored in log-space (we use log(scale) so exp(get_scaling) returns scale)
        init_scale = 0.05
        self._scaling = torch.log(torch.full((N, 3), init_scale, dtype=self.dtype, device=self.device)).contiguous()

        # rotation as quaternion (w, x, y, z), identity = (1,0,0,0)
        self._rotation = torch.zeros((N, 4), dtype=self.dtype, device=self.device).contiguous()
        self._rotation[:, 0] = 1.0

        # opacity: store *logit* values so that sigmoid(opacity) gives desired initial 0.5
        # sigmoid(0.0) == 0.5, so initialize with zeros (not 0.5)
        self._opacity = torch.zeros((N, 1), dtype=self.dtype, device=self.device).contiguous()

    def get_xyz(self):
        return self._xyz

    def get_features_dc(self):
        return self._features_dc

    def get_features_rest(self):
        return self._features_rest

    def get_scaling(self):
        return torch.exp(self._scaling)

    def get_rotation(self):
        norm = torch.norm(self._rotation, dim=-1, keepdim=True)
        return self._rotation / (norm + 1e-8)

    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def pts_num(self):
        return int(self._xyz.shape[0])

    def input_pos(self):
        return self._xyz.detach().cpu().numpy()

    def input_rgb(self):
        from dg_slam.gaussian.sh_utils import SH2RGB
        rgb = SH2RGB(self._features_dc)  # expect (N,3) output
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
