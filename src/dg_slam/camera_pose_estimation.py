"""
Given a sequence of RGB-D frames {I_i, D_i} for i=1 to N
where I_i in R^3 and D_i in R,

1. Estimate the camera poses
2. Reconstruct the static 3D scene
    G = {G_i: (mu_i, sigma_i, alpha_i, h_i)}
"""
import numpy as np
from pathlib import Path


class TUM:
    """
    For TUM, we want to load, and do something similar to HW3
    """
    POSE_FILENAME = "groundtruth.txt"
    DEPTH_FILENAME = "depth.txt"
    RGB_FILENAME = "rgb.txt"

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self._load_folder()

    def _load_folder(self):
        """
        Load the TUM folder
        """
        image_data = self._load_file(self.folder_path / self.RGB_FILENAME)
        depth_data = self._load_file(self.folder_path / self.DEPTH_FILENAME)
        pose_data = self._load_file(self.folder_path / self.POSE_FILENAME)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

    def _load_file(self, filepath: Path, skiprows: int = 0):
        """
        Load the file in the Tum
        """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.str_, skiprows=skiprows)


if __name__ == "__main__":
    TUM(Path('data/TUM/rgbd_dataset_freiburg3_walking_rpy'))
