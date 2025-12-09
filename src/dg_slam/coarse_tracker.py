import torch
from dg_slam.utils import quaternion_to_matrix, exponential_map, matrix_to_quaternion


class DROIDNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
            torch.nn.GroupNorm(8, 64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.GroupNorm(8, 128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.GroupNorm(8, 128),
            torch.nn.ReLU(),
        )

        self.depth_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 7, stride=2, padding=3),
            torch.nn.GroupNorm(8, 64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.GroupNorm(8, 128),
            torch.nn.ReLU(),
        )

        self.gru = torch.nn.GRUCell(256, 128)
        self.pose_delta = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 6)
        )
        self.confidence = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, images, depths, num_iterations=8):
        N = images.shape[0]
        device = images.device

        img_features = self.image_encoder(images)
        depth_features = self.depth_encoder(depths.unsqueeze(1))

        img_vec = self.pool(img_features).squeeze(-1).squeeze(-1)
        depth_vec = self.pool(depth_features).squeeze(-1).squeeze(-1)

        features = torch.cat([img_vec, depth_vec], dim=1)

        poses = torch.zeros(N, 7, device=device)
        poses[:, 0] = 1.0
        hidden = torch.zeros(N, 128, device=device)

        for _ in range(num_iterations):
            hidden = self.gru(features, hidden)
            delta = self.pose_delta(hidden)
            weight = self.confidence(hidden)

            for i in range(1, N):
                T_curr = quaternion_to_matrix(poses[i])
                T_delta = exponential_map(delta[i] * weight[i] * 0.1)
                T_new = T_delta @ T_curr
                poses[i] = matrix_to_quaternion(T_new)

        return poses


class Stage1Tracker:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        print("\nInitializing Stage 1 Tracker...")

        self.model = DROIDNetwork().to(device)

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model_dict = self.model.state_dict()
        loaded_keys = 0

        for key, value in state_dict.items():
            clean_key = key.replace('module.', '').replace('net.', '')

            if clean_key in model_dict and value.shape == model_dict[clean_key].shape:
                model_dict[clean_key] = value
                loaded_keys += 1

        self.model.load_state_dict(model_dict, strict=False)
        self.model.eval()

        print(f"Loaded {loaded_keys}/{len(model_dict)} parameters")

    def run(self, dataset):
        N = dataset.rgb_images.shape[0]
        print("\nStep 1: DROID-SLAM...")
        with torch.no_grad():
            poses_quat = self.model(dataset.rgb_images, dataset.depth_images, num_iterations=8)
        poses_matrix = quaternion_to_matrix(poses_quat)

        return {
            'poses': poses_matrix,
            'num_frames': N
        }
