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


def kitti_rpe(pred, gt, lengths=(100, 200, 300, 400, 500, 600, 700, 800), step=10):
    """KITTI 里程计官方相对位姿误差：按 100..800m 子段平均。

    对每个起点 i、每个段长 L，取沿 GT 行进 L 米后的帧 j，比较 pred 与 gt 的相对位姿：
      err = (gt_i^{-1} gt_j)^{-1} (pred_i^{-1} pred_j)
    平移误差归一化到 %/段长、旋转误差归一化到 deg/m，再对所有 (i,L) 平均。
    pred/gt 均为 (N,4,4) 且用同一 body 约定（此处 velodyne），相对量与世界系无关。
    """
    from scipy.spatial.transform import Rotation
    n = len(pred)
    gt = gt[:n]
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1))])
    t_err, r_err = [], []
    for i in range(0, n, step):
        for L in lengths:
            j = int(np.searchsorted(dist, dist[i] + L))
            if j >= n:
                continue
            gt_rel = np.linalg.inv(gt[i]) @ gt[j]
            pr_rel = np.linalg.inv(pred[i]) @ pred[j]
            err = np.linalg.inv(gt_rel) @ pr_rel
            t_err.append(np.linalg.norm(err[:3, 3]) / L)
            r_err.append(Rotation.from_matrix(err[:3, :3]).magnitude() / L)
    if not t_err:
        return {'trans_err_pct': float('nan'), 'rot_err_deg_per_m': float('nan'), 'n_segments': 0}
    return {'trans_err_pct': float(np.mean(t_err) * 100),
            'rot_err_deg_per_m': float(np.rad2deg(np.mean(r_err))),
            'n_segments': len(t_err)}
