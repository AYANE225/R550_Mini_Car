"""纯 Python/NumPy 的 point-to-plane ICP 参考实现（与 C++ native/icp.cpp 同算法同口径）。

用于和 C++ 版对拍：验证结果一致 + 对比速度。correspondences 用 scipy cKDTree，
高斯牛顿累加向量化。故意与 icp_cpp 保持同样的线性化/收敛，作为对照基线。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    a = w / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def icp_point_to_plane(source, target, normals, init, max_dist=1.0, max_iter=30):
    """返回 (T_target_source 4x4, fitness, inlier_rmse)。"""
    tree = cKDTree(target)
    T = np.array(init, dtype=float)
    fitness = rmse = 0.0
    for _ in range(max_iter):
        R, t = T[:3, :3], T[:3, 3]
        p = source @ R.T + t
        d, idx = tree.query(p, distance_upper_bound=max_dist)
        m = np.isfinite(d)
        if m.sum() < 6:
            break
        pp, q, n = p[m], target[idx[m]], normals[idx[m]]
        r = np.einsum('ij,ij->i', n, pp - q)               # 残差
        J = np.hstack([np.cross(pp, n), n])                # (K,6): [(p×n), n]
        A = J.T @ J
        b = J.T @ r
        delta = np.linalg.solve(A, -b)
        inc = np.eye(4)
        inc[:3, :3] = _rodrigues(delta[:3])
        inc[:3, 3] = delta[3:]
        T = inc @ T
        fitness = float(m.sum()) / len(source)
        rmse = float(np.sqrt(np.mean(r ** 2)))
        if np.linalg.norm(delta) < 1e-6:
            break
    return T, fitness, rmse
