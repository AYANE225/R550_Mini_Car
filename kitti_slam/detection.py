"""经典 LiDAR 物体检测：地面分割 → 欧氏聚类 → BEV 有向包围盒（无深度学习、无标注需求）。

感知基本功的几何版：RANSAC 拟合地面并剔除 → 对非地面点 DBSCAN 聚类 → 每簇用 PCA 在 BEV
求主方向拟合有向框 → 按尺寸/点数过滤出车辆等物体候选。是 PointPillars 等 DL 检测的经典对照，
也可反哺 SLAM（分出动态物体后只用静态点建图）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .registration import to_o3d


@dataclass
class Box3D:
    cx: float
    cy: float
    cz: float
    l: float          # 沿主方向长
    w: float          # 次方向宽
    h: float          # 高
    yaw: float        # BEV 朝向(rad)
    n: int            # 点数

    def corners_bev(self):
        """返回 BEV 四角 (4,2)（世界/雷达系 x-y）。"""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        hl, hw = self.l / 2, self.w / 2
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        return local @ R.T + [self.cx, self.cy]


def _oriented_box(pts):
    """对一簇点用 BEV 主成分拟合有向框。pts:(K,3)。"""
    xy = pts[:, :2]
    c = xy.mean(0)
    u, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    axis = vt[0]                              # BEV 主方向
    yaw = float(np.arctan2(axis[1], axis[0]))
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cs, -sn], [sn, cs]])
    proj = (xy - c) @ R.T                     # 旋到主轴系
    lo, hi = proj.min(0), proj.max(0)
    l, w = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    ctr = c + ((lo + hi) / 2) @ np.array([[np.cos(yaw), np.sin(yaw)],
                                          [-np.sin(yaw), np.cos(yaw)]])
    z = pts[:, 2]
    return Box3D(cx=float(ctr[0]), cy=float(ctr[1]), cz=float(z.mean()),
                 l=max(l, w), w=min(l, w), h=float(z.max() - z.min()),
                 yaw=yaw if l >= w else yaw + np.pi / 2, n=len(pts))


def is_vehicle(b):
    """车辆类粗筛：BEV 尺寸/高度落在轿车~厢车范围（class-agnostic 聚类里挑车）。"""
    return (2.0 <= b.l <= 5.5) and (1.2 <= b.w <= 2.4) and (1.0 <= b.h <= 2.2)


def detect(points, ground_dist=0.25, eps=0.6, min_points=15,
           min_range=3.0, max_range=45.0, z_lo=-1.6, z_hi=2.5,
           dim_range=(0.3, 8.0), h_range=(0.4, 4.0), vehicles_only=False,
           voxel=0.0, ransac_iters=200):
    """一帧原始 velodyne → 物体候选框列表 + 非地面点(用于可视化)。

    过滤：距离/高度带裁剪 → RANSAC 去地面 → DBSCAN 聚类 → 按 BEV 尺寸与高度筛物体。
    vehicles_only=True 时只保留车辆类尺寸的框。voxel>0 时先体素下采样(提速,适合整序列跑)。
    """
    xyz = points[:, :3]
    r = np.linalg.norm(xyz[:, :2], axis=1)
    m = (r > min_range) & (r < max_range) & (xyz[:, 2] > z_lo) & (xyz[:, 2] < z_hi)
    pc = to_o3d(xyz[m])
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    if len(pc.points) < 100:
        return [], np.empty((0, 3))
    _, inliers = pc.segment_plane(distance_threshold=ground_dist, ransac_n=3, num_iterations=ransac_iters)
    obj = pc.select_by_index(inliers, invert=True)
    pts = np.asarray(obj.points)
    if len(pts) < min_points:
        return [], pts
    labels = np.asarray(obj.cluster_dbscan(eps=eps, min_points=min_points))
    boxes = []
    for lab in range(labels.max() + 1):
        cp = pts[labels == lab]
        if len(cp) < min_points:
            continue
        b = _oriented_box(cp)
        if not (dim_range[0] <= b.w and b.l <= dim_range[1] and h_range[0] <= b.h <= h_range[1]):
            continue
        if vehicles_only and not is_vehicle(b):
            continue
        boxes.append(b)
    return boxes, pts
