"""激光里程计：scan-to-local-map point-to-plane ICP + 匀速运动先验。

标准做法（KISS-ICP 风格的简化版）：每帧下采样后，用上一帧的相对运动做初值，配准到
由近邻若干帧累积成的局部地图上；累积位姿即里程计轨迹。无回环（回环在 loop/posegraph 里）。
"""
from __future__ import annotations

from collections import deque

import numpy as np
import open3d as o3d

from .registration import preprocess, icp_point_to_plane, to_o3d, deskew


class LidarOdometry:
    def __init__(self, voxel=0.5, local_map_frames=12, map_voxel=0.5,
                 normal_radius=1.0, icp_max_dist=2.0):
        self.voxel = voxel
        self.map_voxel = map_voxel
        self.normal_radius = normal_radius
        self.icp_max_dist = icp_max_dist
        self.frames = deque(maxlen=local_map_frames)   # 近邻帧的世界系点(Nx3)
        self.poses = []                                 # T_world_velo 累积
        self.delta = np.eye(4)                          # 上一帧相对运动（匀速先验）
        self._local = None

    def _rebuild_local_map(self):
        pts = np.vstack(self.frames)
        pc = to_o3d(pts).voxel_down_sample(self.map_voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.normal_radius, max_nn=30))
        self._local = pc

    def process(self, points):
        """输入一帧原始 velodyne Nx4/Nx3，返回该帧 T_world_velo。"""
        if not self.poses:
            src = preprocess(points, voxel=self.voxel, normal_radius=self.normal_radius)
            pose = np.eye(4)
            self.poses.append(pose)
            self.frames.append(np.asarray(src.points))
            self._rebuild_local_map()
            return pose
        # 注：试过基于方位角的运动去畸变(deskew)，但缺真实逐点时间戳、扫描起始方位与旋向
        # 约定难定，跨序列净效果不稳(seq02 略好、其余变差)，故不在里程计里启用。见 registration.deskew。
        src = preprocess(points, voxel=self.voxel, normal_radius=self.normal_radius)
        init = self.poses[-1] @ self.delta               # 匀速先验作 ICP 初值
        pose, fit, rmse = icp_point_to_plane(src, self._local, init=init,
                                             max_dist=self.icp_max_dist)
        self.delta = np.linalg.inv(self.poses[-1]) @ pose
        self.poses.append(pose)
        world_pts = (np.asarray(src.points) @ pose[:3, :3].T) + pose[:3, 3]
        self.frames.append(world_pts)
        self._rebuild_local_map()
        return pose

    def run(self, load_frame, n, progress=0):
        """load_frame(i)->points；跑 n 帧，返回 (N,4,4) 位姿。progress>0 每隔若干帧打印。"""
        for i in range(n):
            self.process(load_frame(i))
            if progress and (i % progress == 0):
                print(f'  odom frame {i}/{n}', flush=True)
        return np.array(self.poses)
