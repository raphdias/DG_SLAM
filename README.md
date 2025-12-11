# DG-SLAM: Robust Dynamic Gaussian Splatting SLAM

DG-SLAM: Robust Dynamic Gaussian Splatting SLAM with Hybrid Pose Optimization (Xu et al., NeurIPS 2024)

This repository provides an implementation of **DG-SLAM**, a robust RGB-D SLAM pipeline based on Gaussian Splatting and hybrid pose optimization, along with a complete evaluation framework containing experiment scripts, logs, and configurations.

DG-SLAM introduces a two-stage tracking system (coarse + fine), motion masking via depth warping, and optional semantic segmentation support. This project also includes the full experimental setup used to evaluate the system on the TUM RGB-D dataset.

## Features
### SLAM Pipeline
- RGB-D SLAM using hybrid pose optimization
- Two-stage tracking:
  - Coarse tracking: neural-network estimation
  - Fine tracking: Gaussian splatting optimization
- Motion masking with depth warping
- Optional semantic masks (e.g., YOLO-generated)
- Supports the TUM RGB-D dataset (RPY sequences recommended)

### Evaluation Tools
- Baseline SLAM experiments
- Semantic segmentation tests
- Density variation studies
- Frame-skipping robustness stress tests
- Execution logs, timing results, and visualizations

---

## Installation

NOTE: CUDA-capable GPU is needed

Install a venv
```bash
python -m venv venv
source venv/bin/activate
```

Install the DG-SLAM package
```bash
pip install -e .
```

It's best if we all use TUM - RPY images for development for reproducible results
Run the following in terminal to download the data set to `./data`
```bash
./scripts/download_tum.sh
```

## Basic Usage
```python 
from pathlib import Path
from dg_slam.camera_pose_estimation import TUM
from dg_slam.hybrid_slam import HybridSLAM

# Load dataset
dataset = TUM(
    Path('data/TUM/rgbd_dataset_freiburg3_walking_rpy'),
    frame_rate=10,  # subsample to 10 fps
    max_frames=100
)

# Initialize SLAM system
slam = HybridSLAM(
    fx=535.4,  # focal length x
    fy=539.2,  # focal length y
    cx=320.1,  # principal point x
    cy=247.6,  # principal point y
    H=480,     # image height
    W=640,     # image width
    depth_scale=5000.0,
    device='cuda:0'
)

# Run SLAM
refined_poses, gaussian_map = slam.run(
    dataset,
    max_frames=100,
    use_motion_masks=True
)

print(f"Processed {len(refined_poses)} frames")
print(f"Reconstructed {gaussian_map.pts_num()} Gaussian points")

```

With Semantic Segmentation
```Python
slam = HybridSLAM(
    fx=535.4, fy=539.2, cx=320.1, cy=247.6,
    H=480, W=640,
    semantic_mask_dir='path/to/segmentation/masks'  # Optional
)
```

## Project Structure
### DG_SLAM Implementation
```
dg_slam/
├── src/dg_slam/
│   ├── camera_pose_estimation.py  # Dataset loading and point cloud generation
│   ├── coarse_tracker.py          # Neural network-based coarse tracking
│   ├── fine_tracker.py            # Gaussian splatting-based fine tracking
│   ├── depth_warp.py              # Depth warping and motion mask generation
│   ├── gaussian_model.py          # Gaussian scene representation
│   ├── hybrid_slam.py             # Main SLAM pipeline
│   ├── utils.py                   # Utility functions
│   └── gaussian/                  # Gaussian rendering utilities
├── data/                          # Dataset directory
├── scripts/                       # Helper scripts
└── README.md
```

### Evaluation Framework
```
experiments/
├── baseline/        # No semantics
├── density/         # Gaussian density variations
├── robustness/      # Frame skipping tests
├── semantic/        # Semantic mask generation & tests
results/
├── logs/            # Output logs
├── figures/         # Plots and visualizations
```
## NOTES

- We have camera pose estimation and depth warp 
- DROID-SLAM is not used

## Team
* Ahmad Hassan
* Raphael Dias
* Mir Munavvar Ali

## Citation
```bibtex
@inproceedings{xu2024dgslam,
  title={DG-SLAM: Robust Dynamic Gaussian Splatting SLAM with Hybrid Pose Optimization},
  author={Xu, et al.},
  booktitle={NeurIPS},
  year={2024}
}
```


## Acknowledgments

- Based on the DG-SLAM paper (NeurIPS 2024)
- Uses concepts from DROID-SLAM and Gaussian Splatting
- TUM RGB-D dataset for evaluation
