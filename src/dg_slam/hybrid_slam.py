
import numpy as np
from dg_slam.camera_pose_estimation import PoseEstimator, SceneReconstructor
from dg_slam.gaussian import GaussianModel


class HybridSLAM:
    def __init__(self, fx, fy, cx, cy, depth_scale=5000.0):
        self.pose_estimator = PoseEstimator()
        self.reconstructor = SceneReconstructor(fx, fy, cx, cy, depth_scale)

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
