#!/usr/bin/env python3
"""
Run DG-SLAM density experiments and collect results.
This runs tests for 25%, 50%, 75% dynamic content density.

Usage:
    python run_density_experiments.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('dg_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import argparse
import json
from datetime import datetime

from dg_slam.dg_model import dg_model
from dg_slam.config import load_config
from dg_slam.gaussian.common import setup_seed
from lietorch import SE3
from evaluation.tartanair_evaluator import TartanAirEvaluator


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


def loadtum(datapath, frame_rate=-1, mask_dir='seg_mask'):
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
        
        img_filename = os.path.basename(image_data[i, 1])
        mask_path = os.path.join(datapath, mask_dir, img_filename)
        if os.path.exists(mask_path):
            seg_masks.append(mask_path)
        else:
            seg_masks.append(None)
            
        depths.append(os.path.join(datapath, depth_data[j, 1]))
        c2w = torch.from_numpy(pose_vecs[k]).float()
        poses.append(c2w)

    return images, depths, poses, seg_masks


def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[535.4, 539.2, 320.1, 247.6], 
                 png_depth_scale=5000.0, mask_dir='seg_mask', num_frames=100):
    color_paths, depth_paths, poses, seg_mask_paths = loadtum(datapath, frame_rate=32, mask_dir=mask_dir)
    
    # Limit frames
    color_paths = color_paths[:num_frames]
    depth_paths = depth_paths[:num_frames]
    poses = poses[:num_frames]
    seg_mask_paths = seg_mask_paths[:num_frames]
    
    masks_found = sum(1 for m in seg_mask_paths if m is not None)
    print(f"   Frames: {len(color_paths)}, Masks found: {masks_found}")

    data = []
    for t in range(len(color_paths)):
        images = [cv2.cvtColor(cv2.resize(cv2.imread(color_paths[t]), (image_size[1], image_size[0])), cv2.COLOR_BGR2RGB)]
        images = torch.from_numpy(np.stack(images, 0)).permute(0, 3, 1, 2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)
        pose = poses[t]

        if seg_mask_paths[t] is not None:
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


def run_single_experiment(mask_dir, scene, args, cfg):
    """Run a single experiment and return results"""
    torch.cuda.empty_cache()
    
    dg_slam = dg_model(cfg, args)
    scenedir = os.path.join("data/TUM", scene)
    
    intrinsics_vec = [cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']]
    data_reader = image_stream(scenedir, intrinsics_vec=intrinsics_vec, mask_dir=mask_dir, num_frames=100)
    cfg["data"]["n_img"] = len(data_reader)

    traj = []
    for (tstamp, image, depth, intrinsics, pose, seg_mask) in tqdm(data_reader, desc=f"   Tracking"):
        dg_slam.track(tstamp, image, depth, pose, intrinsics, seg_mask)
        traj.append(pose.cpu().numpy())
    traj_ref = np.array(traj)

    evaluator = TartanAirEvaluator()
    N = dg_slam.video.counter.value
    traj_est_key = dg_slam.tracking_mapping.estimate_c2w_list[:N]
    traj_est = []
    for i in range(N):
        pose = get_tensor_from_camera(traj_est_key[i], Tquad=True).cpu()
        traj_est.append(pose)

    traj_est = torch.stack(traj_est, dim=0)
    dg_slam.video.poses[:N] = lietorch.cat([SE3(traj_est)], 0).inv().data
    
    save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])
    traj_est = dg_slam.terminate_woBA(data_reader)

    results = evaluator.evaluate_one_trajectory(
        traj_ref, traj_est, scale=True, title=f"{scene}_{mask_dir}", save_path=save_path)
    
    return results, traj_ref, traj_est


def main():
    print("="*70)
    print("🧪 DG-SLAM Dynamic Object Density Experiment Runner")
    print("="*70)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="/SSD_DISK/datasets/TUM")
    parser.add_argument("--weights", default="checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384, 512])
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--id", type=int, default=-1)
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
    parser.add_argument('--config', default="configs/TUM_RGBD/fr3_walk_xyz.yaml", type=str)
    args = parser.parse_args()
    
    torch.multiprocessing.set_start_method('spawn')
    
    scene = "rgbd_dataset_freiburg3_walking_xyz"
    
    # Density levels to test
    density_configs = [
        ("seg_mask", "Clean YOLO (baseline)"),
        ("seg_mask_density25", "25% density"),
        ("seg_mask_density50", "50% density"),
        ("seg_mask_density75", "75% density"),
    ]
    
    all_results = {}
    
    for mask_dir, description in density_configs:
        print(f"\n{'='*70}")
        print(f"🔬 Testing: {description}")
        print(f"   Mask directory: {mask_dir}")
        print('='*70)
        
        # Check if mask directory exists
        mask_path = f"data/TUM/{scene}/{mask_dir}"
        if not os.path.exists(mask_path):
            print(f"   ❌ Mask directory not found: {mask_path}")
            print(f"   Run: python generate_density_masks.py --all")
            continue
        
        args.config = f"configs/TUM_RGBD/{scene}.yaml"
        cfg = load_config(args.config, 'configs/dg_slam.yaml')
        setup_seed(cfg["setup_seed"])
        
        save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])
        os.makedirs(save_path, exist_ok=True)
        
        results, traj_ref, traj_est = run_single_experiment(mask_dir, scene, args, cfg)
        
        all_results[mask_dir] = {
            'description': description,
            'ate_score': results['ate_score'],
            'ate_std': results['ate_std'],
            'rpe_score': results['rpe_score']
        }
        
        print(f"\n   ✅ Results for {description}:")
        print(f"      ATE: {results['ate_score']*100:.2f} cm (±{results['ate_std']*100:.2f})")
    
    # Print summary table
    print("\n" + "="*70)
    print("📊 DENSITY EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Configuration':<25} {'ATE (cm)':<12} {'STD (cm)':<12} {'vs Baseline':<12}")
    print("-"*70)
    
    baseline_ate = all_results.get('seg_mask', {}).get('ate_score', 0)
    
    for mask_dir, data in all_results.items():
        ate_cm = data['ate_score'] * 100
        std_cm = data['ate_std'] * 100
        
        if baseline_ate > 0 and mask_dir != 'seg_mask':
            change = ((data['ate_score'] - baseline_ate) / baseline_ate) * 100
            change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        else:
            change_str = "—"
        
        print(f"{data['description']:<25} {ate_cm:<12.2f} {std_cm:<12.2f} {change_str:<12}")
    
    print("="*70)
    
    # Save results to JSON
    results_file = f"density_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()