#!/usr/bin/env python3
"""
Generate trajectory visualization plots for DG-SLAM evaluation.
Creates publication-ready figures comparing estimated vs ground truth trajectories.

Usage:
    python visualize_trajectory.py --mode comparison
    python visualize_trajectory.py --mode all_experiments
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('dg_slam')

import numpy as np
import torch
import lietorch
import cv2
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from dg_slam.dg_model import dg_model
from dg_slam.config import load_config
from dg_slam.gaussian.common import setup_seed
from lietorch import SE3
from evaluation.tartanair_evaluator import TartanAirEvaluator


# ============== Data Loading Functions (from your existing code) ==============

def parse_list(filepath, skiprows=0):
    data = np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    associations = []
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_depth - t))
        k = np.argmin(np.abs(tstamp_pose - t))
        if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
            associations.append((i, j, k))
    return associations

def get_tensor_from_camera(RT, Tquad=False):
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    quad = rot.as_quat()
    quad = np.roll(quad, 1)
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    tensor[3:] = tensor[3:][[1,2,3,0]]
    return tensor

def loadtum(datapath, frame_rate=-1, mask_dir='seg_mask', use_masks=True):
    if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
        pose_list = os.path.join(datapath, 'groundtruth.txt')
    elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
        pose_list = os.path.join(datapath, 'pose.txt')

    image_list = os.path.join(datapath, 'rgb.txt')
    depth_list = os.path.join(datapath, 'depth.txt')

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)
    pose_vecs = pose_data[:, 1:].astype(np.float64)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

    indicies = [0]
    for i in range(1, len(associations)):
        t0 = tstamp_image[associations[indicies[-1]][0]]
        t1 = tstamp_image[associations[i][0]]
        if t1 - t0 > 1.0 / frame_rate:
            indicies += [i]

    images, poses, depths, seg_masks = [], [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images.append(os.path.join(datapath, image_data[i, 1]))
        
        if use_masks:
            img_filename = os.path.basename(image_data[i, 1])
            mask_path = os.path.join(datapath, mask_dir, img_filename)
            seg_masks.append(mask_path if os.path.exists(mask_path) else None)
        else:
            seg_masks.append(None)
            
        depths.append(os.path.join(datapath, depth_data[j, 1]))
        c2w = torch.from_numpy(pose_vecs[k]).float()
        poses.append(c2w)

    return images, depths, poses, seg_masks

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[535.4, 539.2, 320.1, 247.6], 
                 png_depth_scale=5000.0, mask_dir='seg_mask', num_frames=100, use_masks=True):
    color_paths, depth_paths, poses, seg_mask_paths = loadtum(
        datapath, frame_rate=32, mask_dir=mask_dir, use_masks=use_masks)
    
    color_paths = color_paths[:num_frames]
    depth_paths = depth_paths[:num_frames]
    poses = poses[:num_frames]
    seg_mask_paths = seg_mask_paths[:num_frames]

    data = []
    for t in range(len(color_paths)):
        images = [cv2.cvtColor(cv2.resize(cv2.imread(color_paths[t]), (image_size[1], image_size[0])), cv2.COLOR_BGR2RGB)]
        images = torch.from_numpy(np.stack(images, 0)).permute(0, 3, 1, 2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)
        pose = poses[t]

        if use_masks and seg_mask_paths[t] is not None:
            seg_mask = cv2.imread(seg_mask_paths[t], cv2.IMREAD_GRAYSCALE)
            seg_mask = cv2.resize(seg_mask, (image_size[1], image_size[0]))
            seg_mask_data = torch.from_numpy((seg_mask > 128).astype(np.uint8))
        else:
            seg_mask_data = torch.zeros((image_size[0], image_size[1]), dtype=torch.uint8)

        depth_data = cv2.resize(cv2.imread(depth_paths[t], cv2.IMREAD_UNCHANGED), (image_size[1], image_size[0]))
        depth_data = depth_data.astype(np.float32) / png_depth_scale
        depth_data = torch.from_numpy(depth_data)

        data.append((t, images, depth_data, intrinsics, pose, seg_mask_data))
    return data


# ============== Trajectory Extraction ==============

def run_slam_get_trajectory(args, cfg, mask_dir='seg_mask', use_masks=True, num_frames=100):
    """Run DG-SLAM and return both ground truth and estimated trajectories"""
    torch.cuda.empty_cache()
    
    dg_slam = dg_model(cfg, args)
    scenedir = "data/TUM/rgbd_dataset_freiburg3_walking_xyz"
    
    intrinsics_vec = [cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']]
    data_reader = image_stream(scenedir, intrinsics_vec=intrinsics_vec, 
                                mask_dir=mask_dir, num_frames=num_frames, use_masks=use_masks)
    cfg["data"]["n_img"] = len(data_reader)

    traj_gt = []
    for (tstamp, image, depth, intrinsics, pose, seg_mask) in tqdm(data_reader, desc="Running SLAM"):
        dg_slam.track(tstamp, image, depth, pose, intrinsics, seg_mask)
        traj_gt.append(pose.cpu().numpy())
    traj_gt = np.array(traj_gt)

    N = dg_slam.video.counter.value
    traj_est_key = dg_slam.tracking_mapping.estimate_c2w_list[:N]
    traj_est = []
    for i in range(N):
        pose = get_tensor_from_camera(traj_est_key[i], Tquad=True).cpu()
        traj_est.append(pose)

    traj_est = torch.stack(traj_est, dim=0)
    dg_slam.video.poses[:N] = lietorch.cat([SE3(traj_est)], 0).inv().data
    traj_est = dg_slam.terminate_woBA(data_reader)
    
    return traj_gt, traj_est


def extract_xyz_from_traj(traj):
    """Extract XYZ positions from trajectory array"""
    if isinstance(traj, np.ndarray):
        if traj.shape[1] == 7:  # quaternion format [tx, ty, tz, qw, qx, qy, qz]
            return traj[:, :3]
        elif traj.shape[1] == 4 and traj.shape[2] == 4:  # 4x4 matrix
            return traj[:, :3, 3]
    return traj[:, :3]


def align_trajectories(traj_gt, traj_est):
    """
    Align estimated trajectory to ground truth using SE(3) alignment (Horn's method).
    This is what the ATE evaluator does internally before computing error.
    
    Returns: aligned_est trajectory
    """
    gt_xyz = extract_xyz_from_traj(traj_gt)
    est_xyz = extract_xyz_from_traj(traj_est)
    
    # Ensure same length
    min_len = min(len(gt_xyz), len(est_xyz))
    gt_xyz = gt_xyz[:min_len]
    est_xyz = est_xyz[:min_len]
    
    # Center both trajectories
    gt_centroid = np.mean(gt_xyz, axis=0)
    est_centroid = np.mean(est_xyz, axis=0)
    
    gt_centered = gt_xyz - gt_centroid
    est_centered = est_xyz - est_centroid
    
    # Compute rotation using SVD (Kabsch algorithm)
    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale (optional, for Sim(3) alignment)
    scale = np.sum(S) / np.sum(est_centered ** 2)
    
    # Apply transformation: scale * R * est + t
    est_aligned = scale * (est_xyz @ R.T) + gt_centroid - scale * (est_centroid @ R.T)
    
    return gt_xyz, est_aligned


# ============== Visualization Functions ==============

def plot_trajectory_2d(traj_gt, traj_est, title, save_path, ate_value=None):
    """Create 2D trajectory comparison plot (top-down view)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CRITICAL: Align trajectories before plotting
    gt_xyz, est_xyz = align_trajectories(traj_gt, traj_est)
    
    # XY plot (top-down)
    axes[0].plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', linewidth=2, label='Ground Truth')
    axes[0].plot(est_xyz[:, 0], est_xyz[:, 1], 'r--', linewidth=2, label='Estimated')
    axes[0].scatter(gt_xyz[0, 0], gt_xyz[0, 1], c='green', s=100, marker='o', zorder=5, label='Start')
    axes[0].scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], c='black', s=100, marker='x', zorder=5, label='End')
    axes[0].set_xlabel('X (m)', fontsize=12)
    axes[0].set_ylabel('Y (m)', fontsize=12)
    axes[0].set_title('Top-Down View (XY)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # XZ plot (side view)
    axes[1].plot(gt_xyz[:, 0], gt_xyz[:, 2], 'b-', linewidth=2, label='Ground Truth')
    axes[1].plot(est_xyz[:, 0], est_xyz[:, 2], 'r--', linewidth=2, label='Estimated')
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Z (m)', fontsize=12)
    axes[1].set_title('Side View (XZ)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    # YZ plot (front view)
    axes[2].plot(gt_xyz[:, 1], gt_xyz[:, 2], 'b-', linewidth=2, label='Ground Truth')
    axes[2].plot(est_xyz[:, 1], est_xyz[:, 2], 'r--', linewidth=2, label='Estimated')
    axes[2].set_xlabel('Y (m)', fontsize=12)
    axes[2].set_ylabel('Z (m)', fontsize=12)
    axes[2].set_title('Front View (YZ)', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')
    
    if ate_value:
        fig.suptitle(f'{title}\nATE: {ate_value*100:.2f} cm', fontsize=16, fontweight='bold')
    else:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_trajectory_3d(traj_gt, traj_est, title, save_path, ate_value=None):
    """Create 3D trajectory comparison plot"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # CRITICAL: Align trajectories before plotting
    gt_xyz, est_xyz = align_trajectories(traj_gt, traj_est)
    
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'b-', linewidth=2, label='Ground Truth')
    ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], 'r--', linewidth=2, label='Estimated')
    
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], c='green', s=100, marker='o', label='Start')
    ax.scatter(gt_xyz[-1, 0], gt_xyz[-1, 1], gt_xyz[-1, 2], c='black', s=100, marker='x', label='End')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    if ate_value:
        ax.set_title(f'{title}\nATE: {ate_value*100:.2f} cm', fontsize=14, fontweight='bold')
    else:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_error_over_time(traj_gt, traj_est, title, save_path):
    """Plot position error over time"""
    # CRITICAL: Align trajectories before computing error
    gt_xyz, est_xyz = align_trajectories(traj_gt, traj_est)
    
    errors = np.linalg.norm(gt_xyz - est_xyz, axis=1) * 100  # Convert to cm
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, 'b-', linewidth=1.5)
    ax.fill_between(range(len(errors)), errors, alpha=0.3)
    ax.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f} cm')
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Position Error (cm)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_results_bar_chart(results_dict, save_path):
    """Create bar chart comparing all experimental results"""
    labels = list(results_dict.keys())
    ate_values = [v['ate'] * 100 for v in results_dict.values()]  # Convert to cm
    std_values = [v['std'] * 100 for v in results_dict.values()]  # Convert to cm
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, ate_values, yerr=std_values, capsize=5, color='steelblue', 
                  edgecolor='black', linewidth=1.2, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, ate_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Experiment Configuration', fontsize=12)
    ax.set_ylabel('Absolute Trajectory Error (cm)', fontsize=12)
    ax.set_title('DG-SLAM Performance Across Experimental Conditions (100 Frames)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add reference line for paper's result
    ax.axhline(y=1.6, color='red', linestyle='--', linewidth=2, label="Paper's reported (1.6 cm)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def create_drift_comparison_chart(save_path):
    """Create grouped bar chart comparing 100 vs 250 frames"""
    categories = ['No Semantics', 'YOLO Masks']
    frames_100 = [2.23, 1.23]
    frames_250 = [3.40, 2.14]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars1 = ax.bar(x - width/2, frames_100, width, label='100 Frames', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, frames_250, width, label='250 Frames', color='coral', edgecolor='black')
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Absolute Trajectory Error (cm)', fontsize=12)
    ax.set_title('Error Drift: 100 vs 250 Frames', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add percentage annotations
    ax.annotate('+52%', xy=(0.18, 3.40), xytext=(0.35, 3.05),
                fontsize=10, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    ax.annotate('+74%', xy=(1.18, 2.14), xytext=(1.35, 1.75),
                fontsize=10, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    
    # Add reference line for paper's result
    ax.axhline(y=1.6, color='green', linestyle='--', linewidth=2, label="Paper's reported (1.6 cm)")
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='comparison',
                       choices=['comparison', 'bar_chart'],
                       help='Visualization mode')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    
    # DG-SLAM arguments
    parser.add_argument("--weights", default="checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384, 512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--filter_thresh", type=float, default=1.75)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--keyframe_thresh", type=float, default=2.25)
    parser.add_argument("--frontend_thresh", type=float, default=12.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=15.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument('--config', default="configs/TUM_RGBD/rgbd_dataset_freiburg3_walking_xyz.yaml")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'comparison':
        torch.multiprocessing.set_start_method('spawn')
        
        cfg = load_config(args.config, 'configs/dg_slam.yaml')
        setup_seed(cfg["setup_seed"])
        
        print("="*60)
        print("🎨 Generating Trajectory Visualizations")
        print("="*60)
        
        # Run with YOLO masks
        print("\n📍 Running DG-SLAM with YOLO masks...")
        traj_gt, traj_est = run_slam_get_trajectory(args, cfg, mask_dir='seg_mask', use_masks=True)
        
        # Generate plots
        plot_trajectory_2d(traj_gt, traj_est, 
                          'DG-SLAM with YOLO Semantic Masks',
                          f'{args.output_dir}/trajectory_2d_yolo.png',
                          ate_value=0.0123)
        
        plot_trajectory_3d(traj_gt, traj_est,
                          'DG-SLAM with YOLO Semantic Masks',
                          f'{args.output_dir}/trajectory_3d_yolo.png',
                          ate_value=0.0123)
        
        plot_error_over_time(traj_gt, traj_est,
                            'Position Error Over Time (YOLO Masks)',
                            f'{args.output_dir}/error_over_time_yolo.png')
        
    elif args.mode == 'bar_chart':
        # Create bar chart from your collected results (100 frames)
        results_100 = {
            'No Sem.': {'ate': 0.0223, 'std': 0.0097},
            'YOLO': {'ate': 0.0123, 'std': 0.0058},
            '2x Skip': {'ate': 0.0227, 'std': 0.0096},
            '3x Skip': {'ate': 0.0300, 'std': 0.0123},
            '20% Noise': {'ate': 0.0130, 'std': 0.0057},
            '40% Noise': {'ate': 0.0202, 'std': 0.0082},
            '25% Dens.': {'ate': 0.0127, 'std': 0.0058},
            '50% Dens.': {'ate': 0.0112, 'std': 0.0058},
            '75% Dens.': {'ate': 0.0130, 'std': 0.0067},
        }
        
        create_results_bar_chart(results_100, f'{args.output_dir}/results_comparison_100frames.png')
        
        # Create comparison chart for 100 vs 250 frames
        create_drift_comparison_chart(f'{args.output_dir}/drift_comparison.png')
    
    print("\n" + "="*60)
    print("🎉 Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()