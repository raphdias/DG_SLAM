"""
Complete Hybrid SLAM implementation following DG-SLAM architecture.
Implements coarse-to-fine camera tracking with motion mask generation.
"""
import numpy as np
from pathlib import Path
from dg_slam.fine_tracker import FineTracker
from dg_slam.coarse_tracker import Stage1Tracker
from dg_slam.camera_pose_estimation import SceneReconstructor
from dg_slam.gaussian_model import GaussianModel, AdaptiveGaussianManager, Gaussian
from dg_slam.depth_warp import DepthWarper, MotionMaskGenerator


class KeyframeSelector:
    """
    Selects keyframes based on optical flow distance.
    Implements keyframe selection strategy from Section 3.2.
    """

    def __init__(self, optical_flow_threshold: float = 20.0):
        self.optical_flow_threshold = optical_flow_threshold
        self.keyframes = []

    def compute_optical_flow_distance(
        self,
        frame1: dict,
        frame2: dict
    ) -> float:
        """
        Compute optical flow distance between two frames.
        Simplified version using pixel-wise RGB difference as proxy.

        Args:
            frame1, frame2: Frame dicts with 'rgb' key

        Returns:
            flow_distance: Average pixel movement magnitude
        """
        rgb1 = frame1['rgb'].astype(np.float32)
        rgb2 = frame2['rgb'].astype(np.float32)

        # Simple approximation: mean absolute difference
        diff = np.abs(rgb1 - rgb2).mean()

        return diff

    def should_add_keyframe(self, current_frame: dict) -> bool:
        """
        Determine if current frame should be added as keyframe.

        Args:
            current_frame: Current frame dict

        Returns:
            True if should add as keyframe
        """
        if len(self.keyframes) == 0:
            return True

        last_keyframe = self.keyframes[-1]
        flow_dist = self.compute_optical_flow_distance(last_keyframe, current_frame)

        return flow_dist > self.optical_flow_threshold

    def add_keyframe(self, frame: dict):
        """Add frame to keyframe list."""
        self.keyframes.append(frame)

    def get_associated_keyframes(
        self,
        current_idx: int,
        window_size: int = 4
    ) -> list[dict]:
        """
        Get associated keyframes within sliding window.

        Args:
            current_idx: Index of current keyframe
            window_size: Size of sliding window

        Returns:
            list of associated keyframes
        """
        start_idx = max(0, current_idx - window_size)
        end_idx = current_idx

        return self.keyframes[start_idx:end_idx]


class HybridSLAM:
    """
    Main DG-SLAM system implementing coarse-to-fine tracking.
    Follows the architecture in Figure 1 of the paper.
    """

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
        checkpoint_path=Path('checkpoints/droid.pth'),
        device: str = 'cuda:0'
    ):
        self.H = H
        self.W = W
        # Droid Weights
        self.checkpoint_path = checkpoint_path
        self.device = device

        # Core components
        self.reconstructor = SceneReconstructor(fx, fy, cx, cy, depth_scale)
        self.depth_warper = DepthWarper(fx, fy, cx, cy, depth_scale)

        # Motion mask and keyframe management
        self.motion_mask_generator = MotionMaskGenerator(window_size=4)
        self.motion_mask_generator.set_depth_warper(self.depth_warper)
        self.keyframe_selector = KeyframeSelector()

        # Gaussian management
        self.gaussian_manager = AdaptiveGaussianManager()
        self.gaussian_model = None

        # Optimization parameters
        self.tracking_iterations = tracking_iterations
        self.mapping_iterations = mapping_iterations

        # Loss weights (from paper)
        self.lambda_1 = 0.9  # RGB loss weight
        self.lambda_2 = 0.2  # SSIM loss weight
        self.lambda_3 = 0.1  # Depth loss weight

        # Initialize fine tracker
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
            device=device
        )

    def initialize_map(self, first_frame: dict) -> GaussianModel:
        """
        Initialize Gaussian map from first frame.
        Implements map initialization from Section 3.4.

        Args:
            first_frame: First frame dict with 'rgb', 'depth', 'pose'

        Returns:
            Initialized GaussianModel
        """
        print("Initializing Gaussian map from first frame...")

        # Generate point cloud from first frame
        points, colors = self.reconstructor.depth_to_pointcloud(
            first_frame['depth'],
            rgb=first_frame['rgb']
        )

        # Transform to world coordinates (first frame pose is identity or given)
        pose = first_frame.get('pose', np.eye(4))
        points_world = self.reconstructor.transform_pointcloud(points, pose)

        # Initialize Gaussian model
        gaussian_model = GaussianModel(points_world, colors)

        print(f"  Initialized {len(gaussian_model.gaussians)} Gaussians")

        return gaussian_model

    def coarse_tracking(
        self,
        dataset,
        max_frames: int | None = None
    ) -> list[np.ndarray]:
        """
        Coarse stage: Use DROID-SLAM for initial pose estimation.
        Implements coarse pose estimation from Section 3.3.

        Args:
            dataset: TUM dataset object
            max_frames: Maximum frames to process
s
        Returns:
            list of coarse pose estimates (4x4 matrices)
        """

        print("Stage 1: Coarse pose estimation via DROID-SLAM...")
        tracker = Stage1Tracker(self.checkpoint_path, device=self.device)
        result = tracker.run(dataset)
        coarse_poses = result['poses']

        if max_frames is not None:
            coarse_poses = coarse_poses[:max_frames]

        return coarse_poses

    def fine_tracking(
        self,
        frame: dict,
        coarse_pose: np.ndarray,
        motion_mask: np.ndarray,
        gaussian_model: GaussianModel
    ) -> tuple[np.ndarray, dict]:
        """
        Fine stage: Refine pose using Gaussian splatting photometric alignment.
        Implements Eq. 10 from Section 3.3.

        This method performs gradient-based optimization to refine the camera pose
        by minimizing photometric and geometric errors between rendered and observed images.

        Args:
            frame: Current frame dict
            coarse_pose: Initial pose from coarse stage
            motion_mask: Motion mask (True = static)
            gaussian_model: Current Gaussian scene model

        Returns:
            refined_pose: Refined camera pose (4x4 matrix)
            tracking_info: Dictionary containing tracking statistics
        """

        # TODO: Issue with inputs currently, remove this line later
        if coarse_pose.shape != (4, 4):
            raise ValueError(f"Expected coarse_pose shape (4, 4), got {coarse_pose.shape}")
        if 'rgb' not in frame or 'depth' not in frame:
            raise ValueError("Frame must contain 'rgb' and 'depth' keys")

        # Start from coarse pose
        refined_pose = coarse_pose.copy()

        # Use the fine tracker to refine the pose
        refined_pose, tracking_info = self.fine_tracker.track_frame(
            frame=frame,
            coarse_pose=coarse_pose,
            motion_mask=motion_mask,
            gaussians=gaussian_model
        )

        # Validate refined pose
        if np.isnan(refined_pose).any() or np.isinf(refined_pose).any():
            print("  Warning: Invalid refined pose detected, using coarse pose")
            refined_pose = coarse_pose.copy()
            tracking_info['fallback_to_coarse'] = True

        # Check tracking quality
        pose_delta = tracking_info['pose_delta']
        if pose_delta > 0.5:  # Large correction threshold
            print(f"  Warning: Large pose correction: {pose_delta:.4f}m")
            tracking_info['large_correction'] = True

        # Log final tracking loss
        final_loss = tracking_info['final_loss']['total']
        print(f"  Fine tracking complete: Final loss = {final_loss:.6f}")

        return refined_pose, tracking_info

    def update_gaussian_map(
        self,
        frame: dict,
        pose: np.ndarray,
        motion_mask: np.ndarray,
        gaussian_model: GaussianModel
    ) -> GaussianModel:
        """
        Update Gaussian map with new observations.
        Implements adaptive point addition and pruning from Section 3.4.

        Args:
            frame: Current frame dict
            pose: Estimated camera pose
            motion_mask: Motion mask (True = static)
            gaussian_model: Current Gaussian model

        Returns:
            Updated Gaussian model
        """
        # Add new Gaussians in under-fitted regions
        density_map = self.gaussian_manager.compute_point_density_radius(frame['rgb'])

        # Sample points to add (simplified)
        H, W = frame['depth'].shape
        sample_step = 10  # Sample every 10 pixels for efficiency

        new_gaussians_count = 0
        for v in range(0, H, sample_step):
            for u in range(0, W, sample_step):
                if not motion_mask[v, u]:
                    continue  # Skip dynamic regions

                # Check if we should add Gaussian here,
                # We're using placeholder weights for now, in practice
                # should be rendered and checked
                if self.gaussian_manager.should_add_gaussian(
                    accumulated_opacity=0.3,
                    depth_residual=0.05
                ):
                    # Add Gaussian at this location
                    depth_val = frame['depth'][v, u]
                    if depth_val > 0:
                        z = depth_val / self.reconstructor.depth_scale
                        x_cam = (u - self.reconstructor.cx) * z / self.reconstructor.fx
                        y_cam = (v - self.reconstructor.cy) * z / self.reconstructor.fy

                        point_cam = np.array([x_cam, y_cam, z])
                        point_world = (pose[:3, :3] @ point_cam) + pose[:3, 3]

                        color = frame['rgb'][v, u] / 255.0 if frame['rgb'][v, u].max() > 1 else frame['rgb'][v, u]

                        sigma = np.array([density_map[v, u]] * 3)
                        new_gaussian = Gaussian(
                            mu=point_world,
                            sigma=sigma,
                            alpha=0.1,  # Initial opacity
                            feature=color
                        )
                        gaussian_model.gaussians.append(new_gaussian)
                        new_gaussians_count += 1

        # Prune invalid Gaussians
        valid_gaussians = []
        pruned_count = 0
        for gaussian in gaussian_model.gaussians:
            if not self.gaussian_manager.should_prune_gaussian(
                gaussian.alpha,
                gaussian.sigma
            ):
                valid_gaussians.append(gaussian)
            else:
                pruned_count += 1

        gaussian_model.gaussians = valid_gaussians

        if new_gaussians_count > 0 or pruned_count > 0:
            print(f" Map update: +{new_gaussians_count} Gaussians, -{pruned_count} pruned")

        return gaussian_model

    def run(
        self,
        dataset,
        max_frames: int | None = None,
        use_motion_masks: bool = True
    ) -> tuple[list[np.ndarray], GaussianModel]:
        """
        Run complete DG-SLAM pipeline.

        Args:
            dataset: TUM dataset object
            max_frames: Maximum frames to process
            use_motion_masks: Whether to use motion mask generation

        Returns:
            refined_poses: list of refined camera poses
            gaussian_model: Final Gaussian scene model
        """
        n_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))

        print(f"Running DG-SLAM on {n_frames} frames...")
        print("=" * 60)

        # --- Stage 1: Coarse Pose Estimation ---
        coarse_poses = self.coarse_tracking(dataset, max_frames)

        # Update dataset poses with coarse estimates
        for i, pose in enumerate(coarse_poses):
            if i < len(dataset):
                dataset.poses[i] = pose

        # --- Stage 2: Initialize Gaussian Map ---
        first_frame = dataset[0]
        self.gaussian_model = self.initialize_map(first_frame)

        # --- Stage 3: Fine Tracking and Mapping ---
        print("\nStage 2: Fine pose refinement and mapping...")
        refined_poses = []
        tracking_stats = []

        for frame_idx in range(n_frames):
            frame = dataset[frame_idx]

            # Check if keyframe
            if self.keyframe_selector.should_add_keyframe(frame):
                self.keyframe_selector.add_keyframe(frame)

                # Generate motion mask for keyframe
                if use_motion_masks and len(self.keyframe_selector.keyframes) > 1:
                    keyframe_window = self.keyframe_selector.get_associated_keyframes(
                        len(self.keyframe_selector.keyframes) - 1
                    )
                    motion_mask = self.motion_mask_generator.generate_motion_mask(
                        frame,
                        keyframe_window,
                        use_semantic=True
                    )
                else:
                    # First keyframe or no motion masking
                    motion_mask = np.ones_like(frame['depth'], dtype=bool)
            else:
                # Use mask from last keyframe
                motion_mask = np.ones_like(frame['depth'], dtype=bool)

            # Fine tracking
            refined_pose, tracking_info = self.fine_tracking(
                frame,
                coarse_poses[frame_idx],
                motion_mask,
                self.gaussian_model
            )
            refined_poses.append(refined_pose)
            tracking_stats.append(tracking_info)

            # Update map for keyframes
            if len(self.keyframe_selector.keyframes) > 0 and \
               self.keyframe_selector.keyframes[-1] == frame:
                self.gaussian_model = self.update_gaussian_map(
                    frame,
                    refined_pose,
                    motion_mask,
                    self.gaussian_model
                )

            if (frame_idx + 1) % 10 == 0:
                avg_loss = np.mean([s['final_loss']['total'] for s in tracking_stats[-10:]])
                print(f"  Processed {frame_idx + 1}/{n_frames} frames "
                      f"(Avg loss: {avg_loss:.6f})")
                print(f"    Gaussians: {len(self.gaussian_model.gaussians)}, "
                      f"Keyframes: {len(self.keyframe_selector.keyframes)}")

        print("\n" + "=" * 60)
        print("DG-SLAM complete!")
        print("Final statistics:")
        print(f"  Total frames: {n_frames}")
        print(f"  Keyframes: {len(self.keyframe_selector.keyframes)}")
        print(f"  Gaussians: {len(self.gaussian_model.gaussians)}")

        # Tracking quality statistics
        avg_final_loss = np.mean([s['final_loss']['total'] for s in tracking_stats])
        avg_pose_delta = np.mean([s['pose_delta'] for s in tracking_stats])
        print(f"  Average tracking loss: {avg_final_loss:.6f}")
        print(f"  Average pose correction: {avg_pose_delta:.4f}m")

        # Reconstruct final point cloud
        all_points = []
        for frame in dataset[:n_frames]:
            pts, _ = self.reconstructor.depth_to_pointcloud(
                frame['depth'],
                rgb=frame['rgb']
            )
            pts_world = self.reconstructor.transform_pointcloud(
                pts,
                frame['pose']
            )
            all_points.append(pts_world)

        all_points = np.vstack(all_points)
        print(f"  Final static point cloud: {all_points.shape[0]:,} points")

        return refined_poses, self.gaussian_model
