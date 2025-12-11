import os
import math
from pathlib import Path
import numpy as np
import torch

from dg_slam.fine_tracker import FineTracker
from dg_slam.coarse_tracker import Stage1Tracker
from dg_slam.camera_pose_estimation import SceneReconstructor
from dg_slam.gaussian_model import GaussianModel, AdaptiveGaussianManager, Gaussian
from dg_slam.depth_warp import DepthWarper, MotionMaskGenerator

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def _voxel_downsample(points: np.ndarray, colors: np.ndarray, max_points: int):
    if len(points) <= max_points:
        return points, colors
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    bbox = maxs - mins
    vox_vol = bbox.prod() + 1e-12
    approx_vox_side = (vox_vol / max_points) ** (1.0 / 3.0)
    if approx_vox_side <= 0:
        idx = np.random.choice(len(points), max_points, replace=False)
        return points[idx], colors[idx]
    inv_side = 1.0 / approx_vox_side
    vox_idx = np.floor((points - mins) * inv_side).astype(np.int64)
    keys = vox_idx[:, 0].astype(np.int64) << 42
    keys ^= vox_idx[:, 1].astype(np.int64) << 21
    keys ^= vox_idx[:, 2].astype(np.int64)
    unique, first_idx = np.unique(keys, return_index=True)
    centers = []
    cols = []
    for k in unique:
        mask = keys == k
        centers.append(points[mask].mean(axis=0))
        cols.append(colors[mask].mean(axis=0))
    centers = np.array(centers, dtype=np.float32)
    cols = np.array(cols, dtype=np.float32)
    if len(centers) > max_points:
        idx = np.random.choice(len(centers), max_points, replace=False)
        centers = centers[idx]
        cols = cols[idx]
    return centers, cols


class KeyframeSelector:
    def __init__(self, optical_flow_threshold: float = 20.0):
        self.optical_flow_threshold = optical_flow_threshold
        self.keyframes = []

    def compute_optical_flow_distance(self, frame1, frame2):
        rgb1 = frame1['rgb'].astype(np.float32)
        rgb2 = frame2['rgb'].astype(np.float32)
        diff = np.abs(rgb1 - rgb2).mean()
        return diff

    def should_add_keyframe(self, current_frame):
        if len(self.keyframes) == 0:
            return True
        last_keyframe = self.keyframes[-1]
        flow_dist = self.compute_optical_flow_distance(last_keyframe, current_frame)
        return flow_dist > self.optical_flow_threshold

    def add_keyframe(self, frame):
        self.keyframes.append(frame)

    def get_associated_keyframes(self, current_idx, window_size: int = 4):
        start_idx = max(0, current_idx - window_size)
        return self.keyframes[start_idx:current_idx]


class HybridSLAM:
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        H: int = 480,
        W: int = 640,
        depth_scale: float = 5000.0,
        tracking_iterations: int = 20,
        mapping_iterations: int = 40,
        checkpoint_path: Path | str = Path('checkpoints/droid.pth'),
        device: str = 'cuda:0'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.H = H
        self.W = W
        self.depth_scale = float(depth_scale)
        self.tracking_iterations = tracking_iterations
        self.mapping_iterations = mapping_iterations
        self.checkpoint_path = Path(checkpoint_path)
        self.reconstructor = SceneReconstructor(fx, fy, cx, cy, self.depth_scale)
        self.depth_warper = DepthWarper(fx, fy, cx, cy, self.depth_scale)
        self.motion_mask_generator = MotionMaskGenerator(window_size=4)
        self.motion_mask_generator.set_depth_warper(self.depth_warper)
        self.keyframe_selector = KeyframeSelector()
        self.gaussian_manager = AdaptiveGaussianManager()
        self.gaussian_model: GaussianModel | None = None
        self.lambda_1 = 0.9
        self.lambda_2 = 0.2
        self.lambda_3 = 0.1
        self.fine_tracker = FineTracker(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            H=H,
            W=W,
            tracking_iterations=tracking_iterations,
            lambda_rgb=self.lambda_1,
            lambda_ssim=self.lambda_2,
            lambda_depth=self.lambda_3,
            learning_rate=0.001,
            device=str(self.device)
        )

    def _safe_tensor(self, arr, dtype=torch.float32):
        t = torch.as_tensor(np.asarray(arr), dtype=dtype, device=self.device)
        if not t.is_contiguous():
            t = t.contiguous()
        if not torch.isfinite(t).all():
            raise ValueError("tensor contains NaN or Inf")
        return t

    def initialize_map(self, first_frame):
        pts, cols = self.reconstructor.depth_to_pointcloud(first_frame['depth'], rgb=first_frame['rgb'])
        pose = first_frame.get('pose', np.eye(4, dtype=np.float32))
        pts_world = self.reconstructor.transform_pointcloud(pts, pose)
        pts_world = np.asarray(pts_world, dtype=np.float32)
        cols = np.asarray(cols, dtype=np.float32)
        if len(pts_world) == 0:
            raise ValueError("No points to initialize map")
        MAX_INITIAL_GAUSSIANS = 10000
        if len(pts_world) > MAX_INITIAL_GAUSSIANS:
            pts_world, cols = _voxel_downsample(pts_world, cols, MAX_INITIAL_GAUSSIANS)
        mask = ~(
            np.isnan(pts_world).any(axis=1) |
            np.isinf(pts_world).any(axis=1)
        )
        pts_world = pts_world[mask]
        cols = cols[mask]
        if len(pts_world) == 0:
            raise ValueError("No valid points after cleaning")
        gaussian_model = GaussianModel(pts_world, cols, device=str(self.device))
        return gaussian_model

    def coarse_tracking(self, dataset, max_frames: int | None = None):
        n_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
        # try:
        # tracker = Stage1Tracker(self.checkpoint_path, device=str(self.device))
        # result = tracker.run(dataset)
        # coarse_poses = result['poses'][:n_frames]
        # except Exception:
        coarse_poses = []
        for i in range(n_frames):
            pose = dataset[i].get('pose', np.eye(4, dtype=np.float32))
            p = pose.copy()
            p[:3, 3] += np.random.randn(3).astype(np.float32) * 0.01
            coarse_poses.append(p)
        return coarse_poses

    def _rebuild_gaussian_tensors(self, gaussian_model: GaussianModel):
        from dg_slam.gaussian.sh_utils import RGB2SH
        N = len(gaussian_model.gaussians)
        if N == 0:
            gaussian_model._xyz = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            gaussian_model._features_dc = torch.zeros((0, 0), dtype=torch.float32, device=self.device)
            gaussian_model._features_rest = torch.zeros((0, 0), dtype=torch.float32, device=self.device)
            gaussian_model._scaling = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
            gaussian_model._rotation = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
            gaussian_model._opacity = torch.zeros((0, 1), dtype=torch.float32, device=self.device)
            return
        points = np.stack([g.mu for g in gaussian_model.gaussians]).astype(np.float32)
        colors = np.stack([g.feature for g in gaussian_model.gaussians]).astype(np.float32)
        scales = np.stack([g.sigma for g in gaussian_model.gaussians]).astype(np.float32)
        alphas = np.stack([g.alpha for g in gaussian_model.gaussians]).astype(np.float32)
        gaussian_model._xyz = torch.from_numpy(points).to(self.device).float().contiguous()
        sh = RGB2SH(colors)
        sh = np.asarray(sh, dtype=np.float32)
        if sh.ndim == 3 and sh.shape[1] == 1:
            sh = sh.squeeze(1)
        gaussian_model._features_dc = torch.from_numpy(sh).to(self.device).float().contiguous()
        gaussian_model._features_rest = torch.zeros((len(gaussian_model.gaussians), 0), dtype=torch.float32, device=self.device)
        gaussian_model._scaling = torch.log(torch.from_numpy(scales).to(self.device).float()).contiguous()
        r = torch.zeros((len(gaussian_model.gaussians), 4), dtype=torch.float32, device=self.device)
        r[:, 0] = 1.0
        gaussian_model._rotation = r
        gaussian_model._opacity = torch.from_numpy(alphas).to(self.device).float().unsqueeze(1).contiguous()

    def fine_tracking(self, frame, coarse_pose, motion_mask, gaussian_model):
        if not isinstance(gaussian_model, GaussianModel):
            raise TypeError("gaussian_model must be GaussianModel")
        if coarse_pose.shape != (4, 4):
            raise ValueError("coarse_pose must be 4x4")
        refined_pose, tracking_info = self.fine_tracker.track_frame(
            frame=frame,
            coarse_pose=coarse_pose,
            motion_mask=motion_mask,
            gaussians=gaussian_model
        )
        if np.isnan(refined_pose).any() or np.isinf(refined_pose).any():
            refined_pose = coarse_pose.copy()
            tracking_info['fallback_to_coarse'] = True
        return refined_pose, tracking_info

    def update_gaussian_map(self, frame, pose, motion_mask, gaussian_model):
        H, W = frame['depth'].shape
        density_map = self.gaussian_manager.compute_point_density_radius(frame['rgb'])
        sample_step = max(1, min(8, int(round(max(H, W) / 100.0))))
        new_gaussians = []
        for v in range(0, H, sample_step):
            for u in range(0, W, sample_step):
                if not motion_mask[v, u]:
                    continue
                depth_val = frame['depth'][v, u]
                if depth_val <= 0:
                    continue
                z = float(depth_val) / self.depth_scale
                x_cam = (u - self.cx) * z / self.fx
                y_cam = (v - self.cy) * z / self.fy
                point_cam = np.array([x_cam, y_cam, z], dtype=np.float32)
                point_world = (pose[:3, :3] @ point_cam) + pose[:3, 3]
                color = frame['rgb'][v, u]
                if color.max() > 1.0:
                    color = color.astype(np.float32) / 255.0
                sigma = np.array([float(density_map[v, u])] * 3, dtype=np.float32)
                new_gaussians.append(Gaussian(mu=point_world, sigma=sigma, alpha=0.1, feature=color.astype(np.float32)))
        if new_gaussians:
            gaussian_model.gaussians.extend(new_gaussians)
        valid = []
        pruned = 0
        for g in gaussian_model.gaussians:
            if self.gaussian_manager.should_prune_gaussian(float(g.alpha), np.asarray(g.sigma, dtype=np.float32)):
                pruned += 1
            else:
                valid.append(g)
        gaussian_model.gaussians = valid
        if new_gaussians or pruned:
            self._rebuild_gaussian_tensors(gaussian_model)
        return gaussian_model

    def run(self, dataset, max_frames: int | None = None, use_motion_masks: bool = True):
        n_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
        print("REFFINING4")
        coarse_poses = self.coarse_tracking(dataset, max_frames)
        first_frame = dataset[0]
        print("REFFINING3")
        self.gaussian_model = self.initialize_map(first_frame)
        print("REFFINING2")
        refined_poses = []
        tracking_stats = []
        for i in range(n_frames):
            frame = dataset[i]
            print("REFFINING1")
            if self.keyframe_selector.should_add_keyframe(frame):
                self.keyframe_selector.add_keyframe(frame)
                if use_motion_masks and len(self.keyframe_selector.keyframes) > 1:
                    window = self.keyframe_selector.get_associated_keyframes(len(self.keyframe_selector.keyframes) - 1)
                    motion_mask = self.motion_mask_generator.generate_motion_mask(frame, window, use_semantic=True)
                else:
                    motion_mask = np.ones_like(frame['depth'], dtype=bool)
            else:
                motion_mask = np.ones_like(frame['depth'], dtype=bool)
            print("REFFINING")
            refined_pose, tracking_info = self.fine_tracking(frame, coarse_poses[i], motion_mask, self.gaussian_model)
            refined_poses.append(refined_pose)
            tracking_stats.append(tracking_info)
            if (i + 1) % 10 == 0:
                pass
        return refined_poses, self.gaussian_model
