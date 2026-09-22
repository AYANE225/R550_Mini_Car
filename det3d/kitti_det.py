"""KITTI 3D 目标检测数据读取 + 标定/坐标变换（cam→LiDAR），供 PointPillars 训练。

KITTI 3D-Object 布局：training/{velodyne,calib,label_2}, testing/{velodyne,calib}。
标注在**相机系**（loc=框底中心, ry 绕相机 y 轴）；训练在 **LiDAR 系**，需用 calib 转换：
  X_cam = R0_rect · Tr_velo_to_cam · X_velo  ⇒  X_velo = (R0·V2C)^{-1} X_cam
框：velo 中心 = 变换后的底中心，z += h/2；朝向 yaw = -ry - π/2；尺寸 (l,w,h)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT3D = Path('/media/4T/cst/DATASETS/SLAM/KITTI_3DObject')


def read_velodyne(split, idx, root=ROOT3D):
    p = root / split / 'velodyne' / f'{int(idx):06d}.bin'
    return np.fromfile(p, dtype=np.float32).reshape(-1, 4)          # x,y,z,intensity


def read_calib(split, idx, root=ROOT3D):
    d = {}
    for line in open(root / split / 'calib' / f'{int(idx):06d}.txt'):
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        d[k.strip()] = np.array([float(x) for x in v.split()], dtype=np.float64)
    R0 = np.eye(4); R0[:3, :3] = d['R0_rect'].reshape(3, 3)
    V2C = np.eye(4); V2C[:3] = d['Tr_velo_to_cam'].reshape(3, 4)
    return {'R0': R0, 'V2C': V2C, 'C2V': np.linalg.inv(R0 @ V2C), 'P2': d['P2'].reshape(3, 4)}


def _cam_to_velo(pts_cam, calib):
    h = np.c_[pts_cam, np.ones(len(pts_cam))]
    return (h @ calib['C2V'].T)[:, :3]


CLASSES = ('Car',)                                                 # 先只做 Car（最标准）


def read_labels(split, idx, calib, classes=CLASSES, root=ROOT3D):
    """返回 (M,7) LiDAR 系框 [x,y,z,l,w,h,yaw]，仅指定类别、且非 DontCare/截断过重。"""
    path = root / split / 'label_2' / f'{int(idx):06d}.txt'
    boxes = []
    for line in open(path):
        f = line.split()
        if f[0] not in classes:
            continue
        h_, w_, l_ = float(f[8]), float(f[9]), float(f[10])
        loc = np.array([[float(f[11]), float(f[12]), float(f[13])]])   # cam 底中心
        ry = float(f[14])
        c = _cam_to_velo(loc, calib)[0]
        yaw = -ry - np.pi / 2
        boxes.append([c[0], c[1], c[2] + h_ / 2, l_, w_, h_, yaw])     # z 抬到框中心
    return np.array(boxes, dtype=np.float32).reshape(-1, 7)


def load_split(name, root=ROOT3D):
    return [int(x) for x in (root / 'ImageSets' / f'{name}.txt').read_text().split()]


def box_corners_bev(box):
    """框的 BEV 四角 (4,2)。box=[x,y,z,l,w,h,yaw]。"""
    x, y, _, l, w, _, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    return corners @ R.T + [x, y]
