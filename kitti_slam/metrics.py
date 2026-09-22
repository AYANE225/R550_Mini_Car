"""轨迹评测：SE(3) 对齐后的 ATE（真值仅在此使用）。"""
from __future__ import annotations

import numpy as np


def umeyama_se3(src, dst):
    """求刚体 (R,t) 使 R@src+t ≈ dst（无尺度，激光本就是米制）。src,dst: (N,3)。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_d - R @ mu_s
    return R, t


def ate(pred_poses, gt_poses):
    """SE(3) 对齐后的绝对轨迹误差 RMSE(米)。pred/gt: (N,4,4)，同一坐标系(velo)。"""
    p = pred_poses[:, :3, 3]
    g = gt_poses[:len(p), :3, 3]
    R, t = umeyama_se3(p, g)
    aligned = (R @ p.T).T + t
    err = np.linalg.norm(aligned - g, axis=1)
    return {'ate_rmse_m': float(np.sqrt(np.mean(err ** 2))),
            'ate_mean_m': float(err.mean()), 'ate_max_m': float(err.max()),
            'aligned': aligned, 'gt_xyz': g}


def path_length(poses):
    d = np.diff(poses[:, :3, 3], axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))
