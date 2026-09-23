"""无初值全局重定位（kidnapped robot）：Scan Context 外观检索给粗位姿 → scan-to-map ICP 精化。

先验地图定位常规假设已知初值(见 localize.py / run_localize.py)；这里**去掉初值**——车像被
"绑架"到地图上未知位置，仅凭一帧激光找回 6DOF 世界位姿。做法两步、全复用现有自研件：
  ① Scan Context 外观检索(极坐标最大高度指纹，与位置/漂移无关)→ 最相似的建图关键帧 + 相对偏航；
  ② 以该关键帧位姿(按相对偏航绕竖直轴校正)作粗位姿，scan-to-map 点面 ICP 精化到米级/亚米级。
对照：没有外观检索时只能从地图某固定点(如原点)盲配，除起点附近外基本失败——正是 place
recognition 让全局重定位成为可能。纯 numpy/scipy + open3d(ICP)。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import scan_context as sc
from .localize import MapLocalizer


def _rz(a):
    """绕竖直(z)轴旋转 a 弧度的 4×4。"""
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


class GlobalRelocalizer:
    """SC 外观检索 + scan-to-map ICP 的无初值重定位器。

    db_scs/db_keys/db_poses 为先验地图的关键帧描述子/环键/世界位姿(索引一一对应)。
    """

    def __init__(self, map_pts, db_scs, db_keys, db_poses, n_ring=20, n_sector=60,
                 max_range=80.0, n_candidates=10, icp_max_dist=4.0, crop_radius=60.0,
                 voxel=0.5):
        self.loc = MapLocalizer(np.asarray(map_pts, np.float64), voxel=voxel,
                                crop_radius=crop_radius, icp_max_dist=icp_max_dist)
        self.scs = list(db_scs)
        self.poses = np.asarray(db_poses)
        self.tree = cKDTree(np.asarray(db_keys))
        self.n_ring, self.n_sector, self.max_range = n_ring, n_sector, max_range
        self.n_cand = n_candidates

    def retrieve(self, scan_pts):
        """外观检索(无初值)：返回 (db_idx, sc_dist∈[0,1], rel_yaw[rad])。"""
        q = sc.scan_context(scan_pts[:, :3], self.n_ring, self.n_sector, self.max_range)
        key = sc.ring_key(q)
        k = min(self.n_cand, len(self.scs))
        _, cand = self.tree.query(key, k=k)
        best = (0, 2.0, 0)
        for j in np.atleast_1d(cand):
            d, shift = sc.sc_distance(q, self.scs[int(j)])
            if d < best[1]:
                best = (int(j), d, shift)
        return best[0], best[1], sc.yaw_from_shift(best[2], self.n_sector)

    def coarse_pose(self, db_idx, yaw):
        """粗位姿 = 检索到的建图关键帧位姿，绕竖直轴按相对偏航校正。"""
        return self.poses[db_idx] @ _rz(-yaw)

    def relocalize(self, scan_pts):
        """无初值重定位。返回 dict(db_idx, sc_dist, coarse, T, fitness, rmse)。"""
        j, d, yaw = self.retrieve(scan_pts)
        coarse = self.coarse_pose(j, yaw)
        T, fit, rmse = self.loc.localize(scan_pts, coarse)
        return dict(db_idx=j, sc_dist=d, yaw=yaw, coarse=coarse, T=T, fitness=fit, rmse=rmse)
