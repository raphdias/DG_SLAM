import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path

# Import your modules
from dg_slam.camera_pose_estimation import TUM, SceneReconstructor


class FrameInspector:
    """Interactive visualization for TUM dataset frames."""

    def __init__(self, dataset_path: str):
        self.dataset = TUM(Path(dataset_path))
        self.reconstructor = SceneReconstructor()
        self.current_idx = 0
        self.points = None
        self.colors = None
        self.points_world = None

        # Process first frame
        self._process_frame(0)

    def _process_frame(self, idx: int):
        """Process a frame and generate point cloud."""
        if idx >= len(self.dataset):
            print(f"Index {idx} out of range (dataset has {len(self.dataset)} frames)")
            return

        self.current_idx = idx
        self.frame = self.dataset[idx]

        # Generate point cloud
        self.points, self.colors = self.reconstructor.depth_to_pointcloud(
            self.frame['depth'], self.frame['rgb']
        )
        self.points_world = self.reconstructor.transform_pointcloud(
            self.points, self.frame['pose']
        )

    def inspect(self, subsample: int = 10):
        """
        Open interactive inspection window.

        Args:
            subsample: Subsample factor for point cloud (for performance)
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'TUM Dataset Frame Inspector - Frame {self.current_idx}', fontsize=14)
        ax_rgb = fig.add_subplot(2, 3, 1)
        ax_depth = fig.add_subplot(2, 3, 2)
        ax_depth_hist = fig.add_subplot(2, 3, 3)
        ax_pointcloud = fig.add_subplot(2, 3, 4, projection='3d')
        ax_pose = fig.add_subplot(2, 3, 5)
        ax_info = fig.add_subplot(2, 3, 6)

        # 1. RGB Image
        ax_rgb.set_title(f'RGB Image {self.frame["rgb"].shape}')
        rgb_img = self.frame['rgb']
        if rgb_img.max() > 1:
            rgb_img = rgb_img / 255.0
        ax_rgb.imshow(rgb_img)
        ax_rgb.axis('off')

        # 2. Depth Image
        ax_depth.set_title(f'Depth Map {self.frame["depth"].shape}')
        depth_img = ax_depth.imshow(self.frame['depth'], cmap='viridis')
        plt.colorbar(depth_img, ax=ax_depth, label='Depth (m)')
        ax_depth.axis('off')

        # 3. Depth Histogram
        ax_depth_hist.set_title('Depth Distribution')
        valid_depth = self.frame['depth'][self.frame['depth'] > 0]
        ax_depth_hist.hist(valid_depth.flatten(), bins=50, color='steelblue', edgecolor='black')
        ax_depth_hist.set_xlabel('Depth (m)')
        ax_depth_hist.set_ylabel('Pixel Count')
        ax_depth_hist.axvline(valid_depth.mean(), color='red', linestyle='--',
                              label=f'Mean: {valid_depth.mean():.2f}m')
        ax_depth_hist.legend()

        # 4. Point Cloud (subsampled for performance)
        ax_pointcloud.set_title(f'Point Cloud ({self.points_world.shape[0]:,} points)')
        pts = self.points_world[::subsample]
        cols = self.colors[::subsample] / 255.0 if self.colors.max() > 1 else self.colors[::subsample]

        ax_pointcloud.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=cols, s=1, alpha=0.6
        )
        ax_pointcloud.set_xlabel('X')
        ax_pointcloud.set_ylabel('Y')
        ax_pointcloud.set_zlabel('Z')

        # Set equal aspect ratio for point cloud
        max_range = np.pmax = np.abs(pts).max()
        ax_pointcloud.set_xlim([-max_range, max_range])
        ax_pointcloud.set_ylim([-max_range, max_range])
        ax_pointcloud.set_zlim([0, max_range * 2])

        # 5. Camera Pose Visualization
        ax_pose.set_title('Camera Pose (4x4 Matrix)')
        pose = self.frame['pose']
        ax_pose.axis('off')

        # Display pose matrix as table
        cell_text = [[f'{val:.4f}' for val in row] for row in pose]
        table = ax_pose.table(
            cellText=cell_text,
            rowLabels=['R1', 'R2', 'R3', 'T'],
            colLabels=['X', 'Y', 'Z', 'W'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # 6. Frame Info
        ax_info.axis('off')
        info_text = f"""
Frame Information
─────────────────────────
Frame Index: {self.current_idx}
Dataset Size: {len(self.dataset)} frames

RGB Shape: {self.frame['rgb'].shape}
  • Width: {self.frame['rgb'].shape[1]} px
  • Height: {self.frame['rgb'].shape[0]} px
  • Channels: {self.frame['rgb'].shape[2]}
  • Dtype: {self.frame['rgb'].dtype}

Depth Shape: {self.frame['depth'].shape}
  • Width: {self.frame['depth'].shape[1]} px
  • Height: {self.frame['depth'].shape[0]} px
  • Dtype: {self.frame['depth'].dtype}
  • Min: {self.frame['depth'].min():.3f} m
  • Max: {self.frame['depth'].max():.3f} m
  • Mean: {valid_depth.mean():.3f} m

Point Cloud:
  • Total Points: {self.points_world.shape[0]:,}
  • Displayed: {len(pts):,} (1/{subsample})

Camera Position:
  • X: {pose[0, 3]:.4f}
  • Y: {pose[1, 3]:.4f}
  • Z: {pose[2, 3]:.4f}
"""
        ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

    def inspect_sequence(self, start: int = 0, end: int = None, step: int = 1):
        """
        Inspect multiple frames with navigation.

        Args:
            start: Starting frame index
            end: Ending frame index (default: all frames)
            step: Step between frames
        """
        if end is None:
            end = len(self.dataset)

        frames = list(range(start, min(end, len(self.dataset)), step))
        current = [0]  # Use list to allow modification in nested function

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2)

        def update(idx):
            self._process_frame(frames[idx])

            for ax in axes.flat:
                ax.clear()

            # RGB
            axes[0, 0].set_title(f'RGB - Frame {frames[idx]}')
            rgb = self.frame['rgb'] / 255.0 if self.frame['rgb'].max() > 1 else self.frame['rgb']
            axes[0, 0].imshow(rgb)
            axes[0, 0].axis('off')

            # Depth
            axes[0, 1].set_title('Depth Map')
            axes[0, 1].imshow(self.frame['depth'], cmap='viridis')
            axes[0, 1].axis('off')

            # Point cloud top-down view
            axes[1, 0].set_title('Point Cloud (Top-Down)')
            pts = self.points_world[::20]
            cols = self.colors[::20] / 255.0 if self.colors.max() > 1 else self.colors[::20]
            axes[1, 0].scatter(pts[:, 0], pts[:, 2], c=cols, s=0.5)
            axes[1, 0].set_xlabel('X')
            axes[1, 0].set_ylabel('Z')
            axes[1, 0].set_aspect('equal')

            # Info
            axes[1, 1].axis('off')
            info = f"Frame: {frames[idx]}\nRGB: {self.frame['rgb'].shape}\nDepth: {self.frame['depth'].shape}\nPoints: {self.points_world.shape[0]:,}"
            axes[1, 1].text(0.5, 0.5, info, ha='center', va='center', fontsize=12, fontfamily='monospace')

            fig.suptitle(f'Frame {frames[idx]} / {len(self.dataset) - 1}')
            fig.canvas.draw_idle()

        # Navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.04])
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])

        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        slider = Slider(ax_slider, 'Frame', 0, len(frames) - 1, valinit=0, valstep=1)

        def next_frame(event):
            if current[0] < len(frames) - 1:
                current[0] += 1
                slider.set_val(current[0])

        def prev_frame(event):
            if current[0] > 0:
                current[0] -= 1
                slider.set_val(current[0])

        def on_slider(val):
            current[0] = int(val)
            update(current[0])

        btn_next.on_clicked(next_frame)
        btn_prev.on_clicked(prev_frame)
        slider.on_changed(on_slider)

        update(0)
        plt.show()


def quick_inspect(dataset_path: str, frame_idx: int = 0, subsample: int = 10):
    """
    Quick inspection of a single frame.

    Args:
        dataset_path: Path to TUM dataset
        frame_idx: Frame index to inspect
        subsample: Subsample factor for point cloud
    """
    inspector = FrameInspector(dataset_path)
    inspector._process_frame(frame_idx)
    inspector.inspect(subsample=subsample)


if __name__ == '__main__':
    dataset_path = 'data/TUM/rgbd_dataset_freiburg3_walking_rpy'

    # Quick single frame inspection
    quick_inspect(dataset_path, frame_idx=0, subsample=10)

    # Full inspector with sequence navigation
    # inspector = FrameInspector(dataset_path)
    # inspector.inspect(subsample=10)  # Single frame detailed view
    # inspector.inspect_sequence(start=0, end=100, step=5)  # Navigate sequence
