# DG-SLAM

DG-SLAM: Robust Dynamic Gaussian Splatting SLAM with Hybrid Pose Optimization (Xu et al., NeurIPS 2024)


## Installation

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


## NOTES

- We have camera pose estimation and depth warp 
- We need coarse stage approx. DROID-SLAM (? docker-container ?) 
- Fine Stage Gaussian (Gaussian only have pseudo right now)
- Verification