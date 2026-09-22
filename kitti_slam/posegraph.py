"""位姿图后端（Open3D 全局优化，Levenberg-Marquardt）。

约定（对齐 Open3D 官方 multiway registration 示例）：
- 节点 pose = T_world_node（世界=0 号帧），直接用里程计位姿。
- 边(source=s, target=t).transformation = T_{t<-s}（把 s 帧点变到 t 帧），即
  inv(pose_t) @ pose_s。里程计边 uncertain=False；回环边 uncertain=True（LM 用 line-process 抗外点）。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


def build(poses, loops, info_scale=1.0):
    """poses:(N,4,4) 里程计位姿；loops:[(i,j,T_j_from_i,fit)]。返回 Open3D PoseGraph。"""
    pg = reg.PoseGraph()
    for p in poses:
        pg.nodes.append(reg.PoseGraphNode(p.copy()))
    info = np.eye(6) * info_scale
    for s in range(len(poses) - 1):                       # 里程计边
        t = s + 1
        T_t_s = np.linalg.inv(poses[t]) @ poses[s]
        pg.edges.append(reg.PoseGraphEdge(s, t, T_t_s, info, uncertain=False))
    for (i, j, T_j_i, _fit) in loops:                     # 回环边（i 为较新帧，j 较早）
        pg.edges.append(reg.PoseGraphEdge(i, j, T_j_i, info, uncertain=True))
    return pg


def optimize(pg, max_corr=3.0):
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=max_corr, edge_prune_threshold=0.25, reference_node=0)
    reg.global_optimization(
        pg, reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(), option)
    return np.array([n.pose for n in pg.nodes])
