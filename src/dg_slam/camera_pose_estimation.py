"""
Given a sequence of RGB-D frames {I_i, D_i} for i=1 to N
where I_i in R^3 and D_i in R,

1. Estimate the camera poses
2. Reconstruct the static 3D scene
    G = {G_i: (mu_i, sigma_i, alpha_i, h_i)}
"""
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation


class TUM:
    """TUM RGB-D dataset loader"""
    POSE_FILENAME = "groundtruth.txt"
    DEPTH_FILENAME = "depth.txt"
    RGB_FILENAME = "rgb.txt"

    def __init__(self, folder_path: Path, frame_rate=-1, max_frames=None):
        self.folder_path = folder_path
        self.rgb_images = []
        self.depth_images = []
        self.poses = []
        self.timestamps = []
        self.seg_masks = []
        self.frame_rate = frame_rate
        self.max_frames = max_frames
        self._load_folder()

    def _load_folder(self):
        """Load the TUM folder with optional frame rate subsampling"""
        image_data = self._load_file(self.folder_path / self.RGB_FILENAME)
        depth_data = self._load_file(self.folder_path / self.DEPTH_FILENAME)
        pose_data = self._load_file(self.folder_path / self.POSE_FILENAME, skiprows=1)

        pose_timestamps = pose_data[:, 0].astype(np.float64)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        rgb_timestamps = image_data[:, 0].astype(np.float64)
        rgb_files = image_data[:, 1]

        depth_timestamps = depth_data[:, 0].astype(np.float64)
        depth_files = depth_data[:, 1]

        # Associate and optionally subsample
        self._associate_data(rgb_timestamps, rgb_files,
                             depth_timestamps, depth_files,
                             pose_timestamps, pose_vecs)

        print(f"Loaded {len(self.rgb_images)} frames from {self.folder_path}")

    def _associate_data(self, rgb_timestamps, rgb_files,
                        depth_timestamps, depth_files,
                        pose_timestamps, pose_vecs, max_dt=0.08):
        """Associate RGB, depth, and pose data by matching timestamps"""
        associations = []

        # First pass: associate all frames
        for i in range(len(rgb_timestamps)):
            depth_idx = np.argmin(np.abs(depth_timestamps - rgb_timestamps[i]))
            if np.abs(depth_timestamps[depth_idx] - rgb_timestamps[i]) > max_dt:
                continue

            pose_idx = np.argmin(np.abs(pose_timestamps - rgb_timestamps[i]))
            if np.abs(pose_timestamps[pose_idx] - rgb_timestamps[i]) > max_dt:
                continue

            associations.append((i, depth_idx, pose_idx))

        # Second pass: subsample by frame rate
        if self.frame_rate > 0:
            indices = [0]
            for i in range(1, len(associations)):
                t0 = rgb_timestamps[associations[indices[-1]][0]]
                t1 = rgb_timestamps[associations[i][0]]
                if t1 - t0 > 1.0 / self.frame_rate:
                    indices.append(i)
            associations = [associations[i] for i in indices]

        # Third pass: limit max frames
        if self.max_frames is not None:
            associations = associations[:self.max_frames]

        # Load data
        for (rgb_idx, depth_idx, pose_idx) in associations:
            rgb_path = self.folder_path / rgb_files[rgb_idx]
            depth_path = self.folder_path / depth_files[depth_idx]

            if not rgb_path.exists() or not depth_path.exists():
                continue

            rgb_img = np.array(Image.open(rgb_path))
            depth_img = np.array(Image.open(depth_path))

            # Check for segmentation mask
            seg_mask_path = self.folder_path / 'seg_mask' / rgb_files[rgb_idx].split('/')[-1]
            if seg_mask_path.exists():
                seg_mask = np.array(Image.open(seg_mask_path))
            else:
                seg_mask = None

            pose = self._pose_vector_to_matrix(pose_vecs[pose_idx])

            self.rgb_images.append(rgb_img)
            self.depth_images.append(depth_img)
            self.poses.append(pose)
            self.timestamps.append(rgb_timestamps[rgb_idx])
            self.seg_masks.append(seg_mask)

    def _pose_vector_to_matrix(self, pose_vec):
        """Convert pose vector [tx, ty, tz, qx, qy, qz, qw] to 4x4 matrix"""
        translation = pose_vec[:3]
        quaternion = pose_vec[3:]  # [qx, qy, qz, qw]

        rotation = Rotation.from_quat(quaternion).as_matrix()

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = translation

        return pose_matrix

    def get_frame(self, idx):
        """Get a specific frame by index"""
        return {
            'rgb': self.rgb_images[idx],
            'depth': self.depth_images[idx],
            'pose': self.poses[idx],
            'timestamp': self.timestamps[idx],
            'seg_mask': self.seg_masks[idx]
        }

    def _load_file(self, filepath: Path, skiprows: int = 0):
        """Load text file"""
        return np.loadtxt(filepath, delimiter=' ', dtype=np.str_, skiprows=skiprows)

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        return self.get_frame(idx)


class SceneReconstructor:
    """3D scene reconstruction from RGB-D frames"""

    def __init__(self, fx=535.4, fy=539.2, cx=320.1, cy=247.6, depth_scale=5000.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale

    def depth_to_pointcloud(self, depth, rgb=None, mask=None):
        """Convert depth image to point cloud"""
        h, w = depth.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32) / self.depth_scale

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        valid = z > 0
        if mask is not None:
            valid = valid & mask

        points = np.stack([x[valid], y[valid], z[valid]], axis=-1)

        if rgb is not None:
            colors = rgb[valid] / 255.0
            return points, colors

        return points

    def transform_pointcloud(self, points, pose):
        """Transform point cloud by pose matrix"""
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        points_world = (pose @ points_h.T).T[:, :3]
        return points_world


if __name__ == "__main__":
    dataset = TUM(Path('data/TUM/rgbd_dataset_freiburg3_walking_rpy'))
    reconstructor = SceneReconstructor()

    # Process first frame as example
    if len(dataset) > 0:
        frame = dataset[0]
        print("Frame 0:")
        print(f"  RGB shape: {frame['rgb'].shape}")
        print(f"  Depth shape: {frame['depth'].shape}")
        print(f"  Pose:\n{frame['pose']}")

        # Generate point cloud
        points, colors = reconstructor.depth_to_pointcloud(
            frame['depth'], frame['rgb']
        )
        points_world = reconstructor.transform_pointcloud(points, frame['pose'])

        print(f"  Point cloud: {points_world.shape[0]} points")
