"""在已建 3D 地图上做 scan-to-map ICP 定位（LiDAR 先验地图定位的工业标准做法）。

每帧把实时扫描配准到**全局固定地图**（裁出当前位置附近一块作 target），初值取上一帧
位姿叠里程计增量。因为对齐的是固定地图而非局部滑窗，位姿不随时间漂移→误差有界。
（注：先试过似然域 MCL 粒子滤波，在 KITTI 这种朝向弱约束的大场景会发散，故改用 scan-to-map ICP。）
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from .registration import preprocess, to_o3d


class MapLocalizer:
    def __init__(self, map_pts, voxel=0.5, normal_radius=1.0,
                 crop_radius=60.0, icp_max_dist=2.0):
        pc = to_o3d(map_pts).voxel_down_sample(voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30))
        self.mpts = np.asarray(pc.points)
        self.mnorm = np.asarray(pc.normals)
        self.crop_radius = crop_radius
        self.icp_max_dist = icp_max_dist

    def _local_target(self, pos):
        m = np.linalg.norm(self.mpts - pos, axis=1) < self.crop_radius
        tgt = to_o3d(self.mpts[m])
        tgt.normals = o3d.utility.Vector3dVector(self.mnorm[m])
        return tgt

    def localize(self, scan_points, init):
        """把一帧扫描配准到地图。返回 (T_world_velo, fitness, rmse)。"""
        src = preprocess(scan_points, voxel=0.5)
        tgt = self._local_target(init[:3, 3])
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, self.icp_max_dist, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        return np.asarray(reg.transformation), reg.fitness, reg.inlier_rmse
