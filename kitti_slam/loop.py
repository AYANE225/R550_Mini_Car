"""回环检测：里程计位置近邻 → ICP 验证 → 相对位姿约束（无研究代码）。

用里程计轨迹的空间近邻找候选（时间上隔开足够久），再用 point-to-plane ICP 验证并给出
回环相对位姿。ICP 以里程计相对位姿为初值；fitness/rmse 阈值拒伪。返回位姿图回环边。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .registration import preprocess, icp_point_to_plane


def detect_candidates(poses, min_gap=150, radius=20.0, stride=5):
    """返回 [(i, j)]：i 与更早的 j 空间近(<radius) 且序号隔开(>min_gap)。每 stride 帧考察一次。"""
    pos = poses[:, :3, 3]
    cands = []
    last_added = -10 ** 9
    for i in range(min_gap, len(pos), stride):
        tree = cKDTree(pos[:i - min_gap])
        d, j = tree.query(pos[i])
        if d < radius and (i - last_added) > min_gap // 2:
            cands.append((int(i), int(j)))
            last_added = i
    return cands


def verify(load_frame, poses, cand, voxel=0.5, max_dist=3.0,
           min_fitness=0.85, max_rmse=0.85, init=None):
    """对候选 (i,j) 做 ICP 验证。返回 (accepted, T_j_from_i, fitness, rmse)。
    init 缺省用里程计相对位姿作初值；漂移大时可外部传入(如 Scan Context 偏航)覆盖。"""
    i, j = cand
    src = preprocess(load_frame(i), voxel=voxel)     # 帧 i
    tgt = preprocess(load_frame(j), voxel=voxel)     # 帧 j
    if init is None:
        init = np.linalg.inv(poses[j]) @ poses[i]    # 里程计给的 T_j_from_i 初值
    T, fit, rmse = icp_point_to_plane(src, tgt, init=init, max_dist=max_dist, max_iter=60)
    accepted = fit >= min_fitness and rmse <= max_rmse
    return accepted, T, fit, rmse


def find_loops(load_frame, poses, **kw):
    """检测 + 验证，返回被接受的回环边 [(i, j, T_j_from_i, fitness)]。"""
    cands = detect_candidates(poses, **{k: kw[k] for k in ('min_gap', 'radius', 'stride') if k in kw})
    loops = []
    for c in cands:
        ok, T, fit, rmse = verify(load_frame, poses, c)
        tag = 'ACCEPT' if ok else 'reject'
        print(f'  loop cand {c[0]}<->{c[1]}: fit={fit:.2f} rmse={rmse:.2f} {tag}', flush=True)
        if ok:
            loops.append((c[0], c[1], T, fit))
    return loops
