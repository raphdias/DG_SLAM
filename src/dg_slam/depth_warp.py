"""
Depth warping for TUM RGB-D dataset to identify static scene regions
"""
import numpy as np


class DepthWarper:
    """
    Warp depth between frames to identify static vs dynamic regions
    Compatible with TUM dataset and SceneReconstructor
    """

    def __init__(self, fx=535.4, fy=539.2, cx=320.1, cy=247.6, depth_scale=5000.0):
        """
        Initialize with TUM FR3 default camera intrinsics
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def generate_pixel_grid(self, H, W):
        """
        Generate pixel coordinate grid
        """
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        # Shape: (H, W, 2) where last dim is [u, v]
        return np.stack([u, v], axis=-1)

    def backproject_depth(self, depth, pose):
        """
        Backproject depth image to 3D points in world coordinates

        Args:
            depth: (H, W) depth image in raw format
            pose: (4, 4) camera pose matrix (camera-to-world)

        Returns:
            points_3d: (H, W, 3) 3D points in world coordinates
            valid_mask: (H, W) boolean mask of valid depth values
        """
        H, W = depth.shape

        # Convert depth to meters
        z = depth.astype(np.float32) / self.depth_scale

        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Backproject to camera coordinates
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        z_cam = z

        # Stack into (H, W, 3)
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world coordinates
        # Reshape to (H*W, 3) for transformation
        points_cam_flat = points_cam.reshape(-1, 3)
        points_cam_h = np.hstack([points_cam_flat, np.ones((points_cam_flat.shape[0], 1))])
        points_world_flat = (pose @ points_cam_h.T).T[:, :3]
        points_world = points_world_flat.reshape(H, W, 3)

        # Valid depth mask
        valid_mask = z > 0

        return points_world, valid_mask

    def project_points(self, points_3d, pose):
        """
        Project 3D world points to image coordinates using camera pose

        Args:
            points_3d: (H, W, 3) or (N, 3) 3D points in world coordinates
            pose: (4, 4) camera pose matrix (camera-to-world)

        Returns:
            uv: (..., 2) pixel coordinates
            depth: (...,) depth values in camera frame
        """
        original_shape = points_3d.shape
        is_2d = len(original_shape) == 3

        if is_2d:
            H, W = original_shape[:2]
            points_flat = points_3d.reshape(-1, 3)
        else:
            points_flat = points_3d

        # Transform to camera coordinates (world-to-camera)
        pose_inv = np.linalg.inv(pose)
        points_h = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_cam = (pose_inv @ points_h.T).T[:, :3]

        # Project to image plane
        x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]

        u = self.fx * x_cam / z_cam + self.cx
        v = self.fy * y_cam / z_cam + self.cy

        uv = np.stack([u, v], axis=-1)

        if is_2d:
            uv = uv.reshape(H, W, 2)
            z_cam = z_cam.reshape(H, W)

        return uv, z_cam

    def depth_warp_pixel(self, rays_o, rays_d, gt_depth, gt_color,
                         pose_init, pose_target, rgb_target, H, W,
                         color_threshold=0.6):
        """
        Warp specific pixels (given rays) from initial to target frame
        Similar to depth_warp_pixel in reference code

        Args:
            rays_o: (N, 3) ray origins in world coordinates
            rays_d: (N, 3) ray directions (normalized)
            gt_depth: (N,) depth values for each ray
            gt_color: (N, 3) color values for each ray
            pose_init: (4, 4) initial camera pose
            pose_target: (4, 4) target camera pose
            rgb_target: (H, W, 3) target frame RGB image (0-255)
            H, W: image dimensions
            color_threshold: threshold for color consistency (0-1 scale)

        Returns:
            final_mask: (N,) boolean mask where True = static pixels
        """
        N = rays_o.shape[0]

        # Compute 3D points in world coordinates
        points_3d = rays_o + rays_d * gt_depth[:, None]

        # Project to target frame
        pose_target_inv = np.linalg.inv(pose_target)
        points_h = np.hstack([points_3d, np.ones((N, 1))])
        points_cam = (pose_target_inv @ points_h.T).T[:, :3]

        # Project to image
        x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
        u = self.fx * x_cam / z_cam + self.cx
        v = self.fy * y_cam / z_cam + self.cy

        # Check visibility
        edge = 0
        visible_mask = (
            (u >= edge) & (u < W - edge) &
            (v >= edge) & (v < H - edge) &
            (z_cam > 0)  # In front of camera
        )

        # Sample colors from target frame
        rgb_target_norm = rgb_target.astype(np.float32) / 255.0

        u_clipped = np.clip(u, 0, W - 1)
        v_clipped = np.clip(v, 0, H - 1)

        u0 = np.floor(u_clipped).astype(np.int32)
        u1 = np.minimum(u0 + 1, W - 1)
        v0 = np.floor(v_clipped).astype(np.int32)
        v1 = np.minimum(v0 + 1, H - 1)

        wu = u_clipped - u0
        wv = v_clipped - v0

        # Bilinear interpolation
        sampled_color = (
            (1 - wu)[:, None] * (1 - wv)[:, None] * rgb_target_norm[v0, u0] +
            wu[:, None] * (1 - wv)[:, None] * rgb_target_norm[v0, u1] +
            (1 - wu)[:, None] * wv[:, None] * rgb_target_norm[v1, u0] +
            wu[:, None] * wv[:, None] * rgb_target_norm[v1, u1]
        )

        # Normalize gt_color if needed
        if gt_color.max() > 1.0:
            gt_color = gt_color / 255.0

        # Color residual
        color_residual = np.abs(sampled_color - gt_color)

        # Complex mask logic matching reference:
        # Static if: (residual < threshold AND sampled > 0) OR (sampled < gt AND sampled > 0)
        color_mask = (
            ((color_residual < color_threshold) & (sampled_color > 0)) |
            ((sampled_color < gt_color) & (sampled_color > 0))
        )

        # All color channels must pass
        color_mask_all = np.all(color_mask, axis=1)

        # Combine with visibility
        final_mask = visible_mask & color_mask_all

        return final_mask

    def depth_warp_to_mask(self, frame_init, frame_target, color_threshold=0.3):
        """
        Warp depth from initial frame to target frame and compute static mask
        Similar to depth_warp_to_mask in reference code (full image)

        Args:
            frame_init: dict with keys 'rgb', 'depth', 'pose'
            frame_target: dict with keys 'rgb', 'depth', 'pose'
            color_threshold: threshold for color consistency (0-1 scale)

        Returns:
            static_mask: (H, W) boolean mask where True = static region
        """
        # Extract data
        rgb_init = frame_init['rgb'].astype(np.float32) / 255.0
        depth_init = frame_init['depth']
        pose_init = frame_init['pose']

        rgb_target = frame_target['rgb'].astype(np.float32) / 255.0
        pose_target = frame_target['pose']

        H, W = depth_init.shape

        # Step 1: Backproject initial frame to 3D world points
        points_3d_world, valid_init = self.backproject_depth(depth_init, pose_init)

        # Step 2: Project to target frame
        uv_target, depth_target = self.project_points(points_3d_world, pose_target)

        # Step 3: Check which points are visible in target frame
        u_target = uv_target[..., 0]
        v_target = uv_target[..., 1]

        edge_margin = 0
        visible_mask = (
            (u_target >= edge_margin) & (u_target < W - edge_margin) &
            (v_target >= edge_margin) & (v_target < H - edge_margin) &
            (depth_target > 0) &
            valid_init
        )

        # Step 4: Sample colors from target frame at projected locations
        # Normalize UV coordinates to [-1, 1] for sampling
        u_norm = 2.0 * u_target / W - 1.0
        v_norm = 2.0 * v_target / H - 1.0

        # Bilinear interpolation (manual implementation)
        u_target_clipped = np.clip(u_target, 0, W - 1)
        v_target_clipped = np.clip(v_target, 0, H - 1)

        u0 = np.floor(u_target_clipped).astype(np.int32)
        u1 = np.minimum(u0 + 1, W - 1)
        v0 = np.floor(v_target_clipped).astype(np.int32)
        v1 = np.minimum(v0 + 1, H - 1)

        wu = u_target_clipped - u0
        wv = v_target_clipped - v0

        # Sample and interpolate colors
        color_target_sampled = (
            (1 - wu)[..., None] * (1 - wv)[..., None] * rgb_target[v0, u0] +
            wu[..., None] * (1 - wv)[..., None] * rgb_target[v0, u1] +
            (1 - wu)[..., None] * wv[..., None] * rgb_target[v1, u0] +
            wu[..., None] * wv[..., None] * rgb_target[v1, u1]
        )

        # Step 5: Compare colors
        color_residual = np.abs(color_target_sampled - rgb_init)
        color_residual_mean = np.mean(color_residual, axis=-1)

        # Step 6: Create static mask
        static_mask = visible_mask & (color_residual_mean < color_threshold)

        return static_mask

    def batch_warp_consecutive_frames(self, dataset, max_frames=None,
                                      color_threshold=0.3, stride=1):
        """
        Warp depth across consecutive frames to identify static regions

        Args:
            dataset: TUM dataset object
            max_frames: maximum number of frames to process
            color_threshold: threshold for color consistency
            stride: frame stride for comparison

        Returns:
            static_masks: list of (H, W) boolean masks
        """
        n_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
        static_masks = []

        for i in range(n_frames - stride):
            frame_init = dataset[i]
            frame_target = dataset[i + stride]

            mask = self.depth_warp_to_mask(frame_init, frame_target, color_threshold)
            static_masks.append(mask)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{n_frames - stride} frame pairs")

        return static_masks

    def generate_rays_for_pixels(self, pixel_coords, depth_values, rgb_values, pose):
        """
        Helper function to generate rays for specific pixels

        Args:
            pixel_coords: (N, 2) array of [u, v] pixel coordinates
            depth_values: (N,) depth values at those pixels
            rgb_values: (N, 3) RGB values at those pixels
            pose: (4, 4) camera pose

        Returns:
            rays_o: (N, 3) ray origins in world coordinates
            rays_d: (N, 3) ray directions in world coordinates
            gt_depth: (N,) depth values
            gt_color: (N, 3) color values
        """
        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]

        # Convert depth to meters
        z = depth_values / self.depth_scale

        # Backproject to camera coordinates
        x_cam = (u - self.cx) * z / self.fx
        y_cam = (v - self.cy) * z / self.fy
        z_cam = z

        # Ray direction in camera frame (normalized)
        ray_d_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        ray_d_cam = ray_d_cam / np.linalg.norm(ray_d_cam, axis=-1, keepdims=True)

        # Transform to world coordinates
        R = pose[:3, :3]
        t = pose[:3, 3]

        rays_d = (R @ ray_d_cam.T).T  # Rotate direction
        rays_o = np.tile(t, (len(u), 1))  # Ray origin is camera center

        return rays_o, rays_d, z, rgb_values


class MotionMaskGenerator:
    """
    Generates motion masks by fusing depth warp masks and semantic masks.
    Implements Section 3.2 from the paper.
    """

    def __init__(self, window_size: int = 4, depth_threshold: float = 0.6):
        self.window_size = window_size
        self.depth_threshold = depth_threshold
        self.depth_warper = None

    def set_depth_warper(self, warper: DepthWarper):
        """Set the depth warper instance."""
        self.depth_warper = warper

    def generate_depth_warp_mask(
        self,
        keyframe_i: dict,
        keyframe_j: dict,
        threshold: float = None
    ) -> np.ndarray:
        """
        Generate depth warp mask between two keyframes.

        Args:
            keyframe_i: Source keyframe dict
            keyframe_j: Target keyframe dict
            threshold: Depth residual threshold (uses self.depth_threshold if None)

        Returns:
            warp_mask: (H, W) boolean mask (True = static)
        """
        if threshold is None:
            threshold = self.depth_threshold

        if self.depth_warper is None:
            raise ValueError("Depth warper not set. Call set_depth_warper first.")

        # Use depth warper to compute static mask
        static_mask = self.depth_warper.depth_warp_to_mask(
            keyframe_i,
            keyframe_j,
            color_threshold=0.3
        )

        return static_mask

    def generate_motion_mask(
        self,
        current_keyframe: dict,
        keyframe_window: list[dict],
        use_semantic: bool = True
    ) -> np.ndarray:
        """
        Generate final motion mask using spatio-temporal consistency.
        Implements Eq. 6-7 from the paper.

        Args:
            current_keyframe: Current keyframe dict with 'rgb', 'depth', 'pose', 'seg_mask'
            keyframe_window: List of associated keyframes within sliding window
            use_semantic: Whether to fuse with semantic mask

        Returns:
            motion_mask: (H, W) boolean mask (True = static, False = dynamic)
        """
        H, W = current_keyframe['depth'].shape

        # Initialize with all static
        combined_warp_mask = np.ones((H, W), dtype=bool)

        # Compute depth warp masks across temporal window
        for kf in keyframe_window:
            warp_mask = self.generate_depth_warp_mask(current_keyframe, kf)
            # Intersection for spatial-temporal consistency (Eq. 7)
            combined_warp_mask = combined_warp_mask & warp_mask

        # Start with depth warp mask
        final_mask = combined_warp_mask.copy()

        # Fuse with semantic mask if available
        if use_semantic and current_keyframe.get('seg_mask') is not None:
            seg_mask = current_keyframe['seg_mask']

            # Semantic mask: True where static objects are present
            # Assume seg_mask marks dynamic objects (needs inversion)
            if seg_mask.dtype == bool:
                semantic_static_mask = ~seg_mask
            else:
                # If integer labels, assume 0 = static, >0 = dynamic classes
                semantic_static_mask = (seg_mask == 0)

            # Union operation: static if either depth warp OR semantic says static
            final_mask = combined_warp_mask | semantic_static_mask

        return final_mask
