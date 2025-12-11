import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('dg_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import argparse

from dg_slam.dg_model import dg_model
from dg_slam.config import load_config
from dg_slam.gaussian.common import setup_seed
from lietorch import SE3
from evaluation.tartanair_evaluator import TartanAirEvaluator

def parse_list(filepath, skiprows=0):
    """ read list data """
    data = np.loadtxt(filepath, delimiter=' ',
                        dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    """ pair images, depths, and poses """
    associations = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))
        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k = np.argmin(np.abs(tstamp_pose - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))
    return associations

def pose_matrix_from_quaternion(pvec):
    """ convert 4x4 pose matrix to (t, q) """
    from scipy.spatial.transform import Rotation
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    return pose

def get_tensor_from_camera(RT, Tquad=False):
    """ Convert transformation matrix to quaternion and translation. """
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

def loadtum(datapath, frame_rate=-1, use_seg_masks=True, mask_dir='seg_mask'):
    """ read video data in tum-rgbd format """
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
    associations = associate_frames(
        tstamp_image, tstamp_depth, tstamp_pose)

    indicies = [0]
    for i in range(1, len(associations)):
        t0 = tstamp_image[associations[indicies[-1]][0]]
        t1 = tstamp_image[associations[i][0]]
        if t1 - t0 > 1.0 / frame_rate:
            indicies += [i]

    images, poses, depths, seg_masks = [], [], [], []
    for ix in indicies:
        (i, j, k) = associations[ix]
        images += [os.path.join(datapath, image_data[i, 1])]
        
        # UPDATED: Load semantic masks if available
        if use_seg_masks:
            # Get the image filename and create corresponding mask path
            img_filename = os.path.basename(image_data[i, 1])
            mask_path = os.path.join(datapath, mask_dir, img_filename)
            if os.path.exists(mask_path):
                seg_masks += [mask_path]
                print(f"✓ Found mask for frame {ix}: {img_filename}")
            else:
                seg_masks += [None]
                print(f"⚠ No mask for frame {ix}: {img_filename}")
        else:
            seg_masks += [None]
            
        depths += [os.path.join(datapath, depth_data[j, 1])]
        c2w = pose_vecs[k]
        c2w = torch.from_numpy(c2w).float()
        poses += [c2w]

    return images, depths, poses, seg_masks

def as_intrinsics_matrix(intrinsics):
    """ Get matrix representation of intrinsics. """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def image_stream(datapath, image_size=[384, 512], intrinsics_vec=[535.4, 539.2, 320.1, 247.6], png_depth_scale = 5000.0, mask_dir='seg_mask'):
    """ image generator """
    # TUM
    # read all png images in folder
    color_paths, depth_paths, poses, seg_mask_paths = loadtum(datapath, frame_rate=32, use_seg_masks=True, mask_dir=mask_dir)
    
    # QUICK TEST: Process only first 100 frames
    test_frames = 100
    color_paths = color_paths[:test_frames]
    depth_paths = depth_paths[:test_frames] 
    poses = poses[:test_frames]
    seg_mask_paths = seg_mask_paths[:test_frames]
    print(f"QUICK TEST MODE: Processing only {test_frames} frames")
    
    # Count how many masks we have
    masks_found = sum(1 for m in seg_mask_paths if m is not None)
    print(f"Semantic masks found: {masks_found}/{test_frames}")

    data = []
    for t in range(len(color_paths)):
        images = [cv2.cvtColor(cv2.resize(cv2.imread(color_paths[t]), (image_size[1], image_size[0])) , cv2.COLOR_BGR2RGB)]
        images = torch.from_numpy(np.stack(images, 0)).permute(0,3,1,2)
        intrinsics = .8 * torch.as_tensor(intrinsics_vec)
        pose = poses[t]

        # UPDATED: Load actual semantic masks
        if seg_mask_paths[t] is not None:
            seg_mask = cv2.imread(seg_mask_paths[t], cv2.IMREAD_GRAYSCALE)
            seg_mask = cv2.resize(seg_mask, (image_size[1], image_size[0]))
            # Convert to binary (255 -> 1, 0 -> 0)
            seg_mask_data = torch.from_numpy((seg_mask > 128).astype(np.uint8))
        else:
            # No mask available - assume everything is static (zeros)
            seg_mask_data = torch.zeros((image_size[0], image_size[1]), dtype=torch.uint8)

        depth_data = cv2.resize(cv2.imread(depth_paths[t], cv2.IMREAD_UNCHANGED), (image_size[1], image_size[0]))
        depth_data = depth_data.astype(np.float32) / png_depth_scale
        depth_data = torch.from_numpy(depth_data)

        data.append((t, images, depth_data, intrinsics, pose, seg_mask_data))
    return data

if __name__ == '__main__':
    print('Start Running DG-SLAM QUICK TEST (100 frames, WITH SEMANTIC MASKS)....')
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="/SSD_DISK/datasets/TUM")
    parser.add_argument("--weights", default="checkpoints/droid.pth")
    parser.add_argument("--buffer", type=int, default=1000)
    parser.add_argument("--image_size", default=[384,512])
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

    parser.add_argument('--config', default="configs/TUM_RGBD/fr3_walk_xyz.yaml", type=str, help='Path to config file.')
    parser.add_argument('--mask_dir', type=str, default='seg_mask', 
                   help='Name of mask directory (seg_mask, seg_mask_noise20, etc.)')
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    tum_split = ["rgbd_dataset_freiburg3_walking_xyz"]

    ate_list = []
    for scene in tum_split:
        print("Performing evaluation on {}".format(scene))
        torch.cuda.empty_cache()

        args.config = "configs/TUM_RGBD/" + scene + ".yaml"
        cfg = load_config(args.config, 'configs/dg_slam.yaml')
        setup_seed(cfg["setup_seed"])

        save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        dg_slam = dg_model(cfg, args)
        
        # Update datapath to use actual TUM data location
        scenedir = os.path.join("data/TUM", scene)  # Changed from args.datapath
        print(f"Loading data from: {scenedir}")

        intrinsics_vec = [cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']]
        data_reader = image_stream(scenedir, intrinsics_vec = intrinsics_vec, mask_dir=args.mask_dir)
        cfg["data"]["n_img"] = len(data_reader)

        ## tracking and mapping
        traj = []
        for (tstamp, image, depth, intrinsics, pose, seg_mask) in tqdm(data_reader):
            dg_slam.track(tstamp, image, depth, pose, intrinsics, seg_mask)
            traj.append(pose.cpu().numpy())
        traj_ref = np.array(traj) 

        evaluator = TartanAirEvaluator()        
        N = dg_slam.video.counter.value
        traj_est_key = dg_slam.tracking_mapping.estimate_c2w_list[:N]
        traj_est= []
        for i in range(N):
            pose = get_tensor_from_camera(traj_est_key[i], Tquad = True).cpu()
            traj_est.append(pose)

        traj_est = torch.stack(traj_est, dim=0)       
        dg_slam.video.poses[:N] = lietorch.cat([SE3(traj_est)], 0).inv().data
        traj_est = dg_slam.terminate_woBA(data_reader)

        results = evaluator.evaluate_one_trajectory(
            traj_ref, traj_est, scale=True, title=scene + '_' + cfg["data"]["exp_name"], save_path = save_path)
        print(results)
        ate_list.append(results["ate_score"])
    
    print("="*60)
    print(f"FINAL ATE: {ate_list[0]*100:.2f} cm")
    print(f"Previous (no semantics): 2.23 cm")
    print(f"Paper's result: 1.6 cm")
    print("="*60)