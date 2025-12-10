import torch


def quaternion_to_matrix(quat_pose):
    """[N,7] (qw,qx,qy,qz,tx,ty,tz) -> [N,4,4]"""
    single = quat_pose.dim() == 1
    if single:
        quat_pose = quat_pose.unsqueeze(0)

    N = quat_pose.shape[0]
    device = quat_pose.device

    q = quat_pose[:, :4] / (torch.norm(quat_pose[:, :4], dim=1, keepdim=True) + 1e-8)
    t = quat_pose[:, 4:]

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(N, 3, 3, device=device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    T = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t

    return T[0] if single else T


def matrix_to_quaternion(T):
    """[N,4,4] -> [N,7]"""
    single = T.dim() == 2
    if single:
        T = T.unsqueeze(0)

    N = T.shape[0]
    device = T.device
    quat_pose = torch.zeros(N, 7, device=device)

    for i in range(N):
        R = T[i, :3, :3]
        t = T[i, :3, 3]
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        quat_pose[i, :4] = torch.tensor([w, x, y, z], device=device)
        quat_pose[i, 4:] = t

    return quat_pose[0] if single else quat_pose


def exponential_map(xi):
    """se(3) -> SE(3)"""
    single = xi.dim() == 1
    if single:
        xi = xi.unsqueeze(0)

    N = xi.shape[0]
    device = xi.device
    omega = xi[:, :3]
    v = xi[:, 3:]
    T = torch.zeros(N, 4, 4, device=device)

    for i in range(N):
        theta = torch.norm(omega[i])

        if theta < 1e-6:
            T[i] = torch.eye(4, device=device)
            T[i, :3, 3] = v[i]
        else:
            w = omega[i] / theta
            K = torch.tensor([
                [0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]
            ], device=device)

            K2 = K @ K
            R = torch.eye(3, device=device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K2
            V = torch.eye(3, device=device) + ((1 - torch.cos(theta)) / theta) * K + \
                ((theta - torch.sin(theta)) / theta) * K2

            T[i, :3, :3] = R
            T[i, :3, 3] = V @ v[i]
            T[i, 3, 3] = 1

    return T[0] if single else T
