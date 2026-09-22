"""建图：用优化后的位姿把各帧点云拼成 3D 体素地图，并投影出 2D 占据栅格（导航/定位用）。

坐标系 = SLAM 世界系（velodyne 帧 0：x 前、y 左、**z 上**）。地面≈z=-1.73（雷达高度）。
2D 占据：取一段高度带内的点（去地面、去高空树冠）投到 x–y 栅格。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .registration import preprocess, to_o3d


def build_point_map(poses, load_frame, stride=2, voxel=0.3, downsample_every=100):
    """拼接体素点云地图。返回 (M,3) SLAM 世界系点。stride 抽帧、voxel 下采样以控体量。"""
    acc = o3d.geometry.PointCloud()
    for i in range(0, len(poses), stride):
        pc = preprocess(load_frame(i), voxel=voxel)      # 已裁剪+下采样(带法线,这里只用点)
        pts = np.asarray(pc.points)
        T = poses[i]
        world = (pts @ T[:3, :3].T) + T[:3, 3]
        acc += to_o3d(world)
        if (i // stride) % downsample_every == 0 and len(acc.points) > 0:
            acc = acc.voxel_down_sample(voxel)
    return np.asarray(acc.voxel_down_sample(voxel).points)


@dataclass
class OccGrid2D:
    occ: np.ndarray        # bool [H, W]，True=占据
    res: float
    x0: float              # 世界坐标(栅格 col0,row0 中心)
    y0: float

    def world_to_cell(self, x, y):
        return int(round((y - self.y0) / self.res)), int(round((x - self.x0) / self.res))

    def cell_to_world(self, r, c):
        return self.x0 + c * self.res, self.y0 + r * self.res


def occupancy_2d(map_pts, res=0.3, z_min=-1.3, z_max=2.0, min_pts=2, pad=2.0):
    """把 3D 地图点按高度带投影成 2D 占据栅格。z 为高度(上为正)。"""
    band = map_pts[(map_pts[:, 2] > z_min) & (map_pts[:, 2] < z_max)]
    xy = band[:, :2]
    x0, y0 = xy[:, 0].min() - pad, xy[:, 1].min() - pad
    x1, y1 = xy[:, 0].max() + pad, xy[:, 1].max() + pad
    W = int(np.ceil((x1 - x0) / res)) + 1
    H = int(np.ceil((y1 - y0) / res)) + 1
    cols = np.round((xy[:, 0] - x0) / res).astype(int)
    rows = np.round((xy[:, 1] - y0) / res).astype(int)
    count = np.zeros((H, W), dtype=np.int32)
    np.add.at(count, (rows, cols), 1)
    return OccGrid2D(occ=count >= min_pts, res=res, x0=x0, y0=y0)
