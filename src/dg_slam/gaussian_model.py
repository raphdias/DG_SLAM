import numpy as np
import torch
from dg_slam.gaussian.sh_utils import RGB2SH


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
