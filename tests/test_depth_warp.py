"""I know this is not how tests are suppposed to work -- im just being lazy"""

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from dg_slam.camera_pose_estimation import TUM
    from dg_slam.depth_warp import DepthWarper
    dataset = TUM(Path('data/TUM/rgbd_dataset_freiburg3_walking_rpy'))
    warper = DepthWarper()

    # Test 1: Full image warping (depth_warp_to_mask)
    frame0 = dataset[0]
    frame1 = dataset[1]

    print("=== Test 1: Full image warping (depth_warp_to_mask) ===")
    static_mask = warper.depth_warp_to_mask(frame0, frame1, color_threshold=0.3)

    print(f"Static mask shape: {static_mask.shape}")
    print(f"Static pixels: {np.sum(static_mask)} / {static_mask.size}")
    print(f"Static ratio: {np.sum(static_mask) / static_mask.size:.2%}")

    # Test 2: Pixel-wise warping (depth_warp_pixel)
    print("\n=== Test 2: Pixel-wise warping (depth_warp_pixel) ===")
    # Sample some pixels
    H, W = frame0['depth'].shape
    sample_coords = np.array([
        [W // 2, H // 2],      # Center
        [W // 4, H // 4],      # Top-left
        [3 * W // 4, 3 * H // 4],  # Bottom-right
    ])

    sample_depths = np.array([
        frame0['depth'][H // 2, W // 2],
        frame0['depth'][H // 4, W // 4],
        frame0['depth'][3 * H // 4, 3 * W // 4],
    ])

    sample_colors = np.array([
        frame0['rgb'][H // 2, W // 2],
        frame0['rgb'][H // 4, W // 4],
        frame0['rgb'][3 * H // 4, 3 * W // 4],
    ])

    # Generate rays for these pixels
    rays_o, rays_d, gt_depth, gt_color = warper.generate_rays_for_pixels(
        sample_coords, sample_depths, sample_colors, frame0['pose']
    )

    # Warp these specific pixels
    pixel_mask = warper.depth_warp_pixel(
        rays_o, rays_d, gt_depth, gt_color,
        frame0['pose'], frame1['pose'], frame1['rgb'], H, W
    )

    print(f"Tested {len(pixel_mask)} pixels")
    print(f"Static pixels: {np.sum(pixel_mask)} / {len(pixel_mask)}")

    # Batch processing
    print("\n=== Test 3: Batch processing consecutive frames ===")
    masks = warper.batch_warp_consecutive_frames(dataset, max_frames=10, stride=1)
    print(f"Generated {len(masks)} static masks")
