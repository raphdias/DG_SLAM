
import numpy as np
from dg_slam.camera_pose_estimation import PoseEstimator, SceneReconstructor
from dg_slam.gaussian import GaussianModel, AdaptiveGaussianManager
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
            List of associated keyframes
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
        depth_scale: float = 5000.0,
        tracking_iterations: int = 20,
        mapping_iterations: int = 40
    ):
        # Core components
        self.pose_estimator = PoseEstimator()
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

    def run(self, dataset):
        # --- Coarse Stage: DROID-SLAM ---
        # TODO: Implement DROID-SLAM -> Coarse STAGE HERE

        # TODO: Generate initial static scene points (from first frame for simplicity)

        # TODO: Transform to world coordinates using the coarse pose
        # points_world = self.reconstructor.transform_pointcloud(points, dataset[0]['pose'])

        # TODO: Initialize Gaussian scene
        # gaussian_model = GaussianModel(points_world, colors)

        # --- Fine Stage: Photometric Alignment ---
        print("Stage 2: Fine pose refinement via Gaussian splatting photometric alignment...")
        intrinsics = np.array([[self.reconstructor.fx, 0, self.reconstructor.cx],
                               [0, self.reconstructor.fy, self.reconstructor.cy],
                               [0, 0, 1]])
        refined_poses = gaussian_model.optimize_poses(dataset, intrinsics)
        dataset.poses = refined_poses
        print("  Pose refinement complete.")

        # --- Visualization ---
        # Reconstruct full static scene with refined poses
        all_points = []
        for f in dataset:
            pts, cls = self.reconstructor.depth_to_pointcloud(f['depth'], rgb=f['rgb'])
            pts_world = self.reconstructor.transform_pointcloud(pts, f['pose'])
            all_points.append(pts_world)
        all_points = np.vstack(all_points)
        print(f"Final static point cloud: {all_points.shape[0]} points (stacked).")

        return refined_poses
