"""子图(submap)SLAM：把长轨迹切成子图，子图内累积稠密点云，子图间做位姿图 + 稠密回环。

为"更大/多楼层"场景铺路：单块稠密全局图内存/回环都吃不消，改为
  轨迹 → 若干子图(各自局部稠密点云) → 子图级位姿图(相邻边=里程计, 回环边=submap-to-submap ICP)
        → 优化子图原点位姿 → 展开回每帧 → 拼有界内存全局图。
submap-to-submap ICP 用稠密点云配准，比单帧回环鲁棒，能抓回单帧漏掉的回环。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .registration import preprocess, to_o3d, icp_point_to_plane
from . import posegraph as pg


def build_submaps(poses, load_frame, size=40, voxel=0.5, normal_radius=1.0):
    """把帧按 size 分组成子图。返回 list[dict]：origin_pose(世界), cloud(o3d,子图局部系,带法线), frames。

    子图局部系 = 该子图首帧的里程计位姿；子图点云在此局部系累积。
    """
    subs = []
    for s in range(0, len(poses), size):
        idx = list(range(s, min(s + size, len(poses))))
        origin = poses[idx[0]]
        origin_inv = np.linalg.inv(origin)
        acc = o3d.geometry.PointCloud()
        for i in idx:
            pc = preprocess(load_frame(i), voxel=voxel, normal_radius=normal_radius)
            rel = origin_inv @ poses[i]                      # 帧相对子图原点
            pts = np.asarray(pc.points)
            acc += to_o3d((pts @ rel[:3, :3].T) + rel[:3, 3])
        acc = acc.voxel_down_sample(voxel)
        acc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        subs.append({'origin': origin, 'cloud': acc, 'frames': idx,
                     'frame_poses': {i: poses[i] for i in idx}})
    return subs


def submap_loops(subs, min_gap=3, radius=25.0, min_fitness=0.7, max_rmse=0.9, max_dist=3.0):
    """子图间稠密 ICP 回环。返回位姿图回环边 [(i, j, T_j_from_i, fit)]（作用于子图原点）。"""
    origins = np.array([s['origin'][:3, 3] for s in subs])
    loops = []
    for i in range(len(subs)):
        tree = cKDTree(origins[:max(0, i - min_gap)]) if i - min_gap > 0 else None
        if tree is None:
            continue
        d, j = tree.query(origins[i])
        if d >= radius:
            continue
        init = np.linalg.inv(subs[j]['origin']) @ subs[i]['origin']   # T_j_from_i(原点)
        T, fit, rmse = icp_point_to_plane(subs[i]['cloud'], subs[j]['cloud'],
                                          init=init, max_dist=max_dist, max_iter=60)
        tag = 'ACCEPT' if (fit >= min_fitness and rmse <= max_rmse) else 'reject'
        print(f'  submap loop {i}<->{j}: fit={fit:.2f} rmse={rmse:.2f} {tag}', flush=True)
        if fit >= min_fitness and rmse <= max_rmse:
            loops.append((i, j, T, fit))
    return loops


def optimize_submaps(subs, loops):
    """子图级位姿图优化，返回修正后的子图原点位姿 (S,4,4)。"""
    origins = np.array([s['origin'] for s in subs])
    return pg.optimize(pg.build(origins, loops))


def expand_to_frames(subs, opt_origins, n_frames):
    """把子图原点的修正量传播回每帧：frame_pose = opt_origin @ (orig_origin_inv @ orig_frame)。

    这里子图内各帧相对原点的位姿沿用里程计（刚体不变），仅整体套上子图原点的修正。
    """
    poses = np.repeat(np.eye(4)[None], n_frames, axis=0)
    for s, opt in zip(subs, opt_origins):
        origin_inv = np.linalg.inv(s['origin'])
        for i in s['frames']:
            rel = origin_inv @ s['frame_poses'][i]      # 帧相对子图原点(里程计,刚体不变)
            poses[i] = opt @ rel                         # 套上优化后的子图原点
    return poses
