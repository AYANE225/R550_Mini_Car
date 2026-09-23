"""SemanticKITTI 数据：官方 learning-map、球面投影(点云↔距离图)、Dataset、类名/配色。

从零做 LiDAR 语义分割的数据侧：把一帧无序点云按激光的方位角/俯仰角投到 64×W 的
二维「距离图」(每像素 5 通道 [range,x,y,z,intensity])，标签同投；网络在图上做 2D 分割，
再把逐像素预测投回每个 3D 点算点级 mIoU。raw 语义 id(0..259)按 SemanticKITTI 官方
learning-map 归并到 19 个评测类(+ignore)。纯 numpy/torch，不依赖官方 devkit。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEQ_ROOT = Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked/dataset/sequences')
TRAIN_SEQS = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
VAL_SEQS = ['08']

# 官方 learning_map：raw semantic id → 0(ignore)+1..19。训练时再 -1 得 0..18，ignore→255。
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
    32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
    71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
    256: 5, 257: 5, 258: 4, 259: 5,
}
CLASS_NAMES = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
               'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
               'other-ground', 'building', 'fence', 'vegetation', 'trunk',
               'terrain', 'pole', 'traffic-sign']
N_CLASSES = 19
# 每类 RGB(0..1)，近 SemanticKITTI 官方配色，供点云/地图着色
CLASS_COLORS = np.array([
    [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80], [255, 0, 0],
    [30, 30, 255], [200, 40, 255], [90, 30, 150], [255, 0, 255], [255, 150, 255],
    [75, 0, 75], [75, 0, 175], [0, 200, 255], [50, 120, 255], [0, 175, 0],
    [0, 60, 135], [80, 240, 150], [150, 240, 255], [0, 0, 255],
], np.float32) / 255.0

_LUT = np.zeros(260, np.int64)              # raw id(0..259) → 0..19（0=ignore）
for k, v in LEARNING_MAP.items():
    _LUT[k] = v


def raw_to_train(raw_sem):
    """raw semantic id(uint32 低16位已取)→ 训练标签 0..18，ignore=255。"""
    m = _LUT[np.clip(raw_sem, 0, 259)]      # 0..19
    out = m.astype(np.int64) - 1            # -1..18；-1 即 ignore
    out[out < 0] = 255
    return out


def read_scan(seq, i):
    f = SEQ_ROOT / seq / 'velodyne' / f'{i:06d}.bin'
    return np.fromfile(f, np.float32).reshape(-1, 4)


def read_label(seq, i):
    f = SEQ_ROOT / seq / 'labels' / f'{i:06d}.label'
    return np.fromfile(f, np.uint32) & 0xFFFF


def frame_ids(seq):
    d = SEQ_ROOT / seq / 'velodyne'
    return sorted(int(p.stem) for p in d.glob('*.bin'))


def project(points, labels=None, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
    """球面投影：点云 (N,≥4) → 距离图 (5,H,W)[range,x,y,z,intensity] + 有效掩码。

    返回 img(5,H,W), mask(H,W)bool, px(N,)列, py(N,)行(每点所投像素，用于把预测投回点)。
    近点覆盖远点(按 range 降序赋值)。labels 给出则同时返回标签图 lab(H,W)。
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inten = points[:, 3]
    r = np.linalg.norm(points[:, :3], axis=1) + 1e-6
    fu = fov_up / 180.0 * np.pi
    fd = fov_down / 180.0 * np.pi
    fov = abs(fu) + abs(fd)
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(np.clip(z / r, -1, 1))
    px = 0.5 * (yaw / np.pi + 1.0) * W
    py = (1.0 - (pitch + abs(fd)) / fov) * H
    px = np.clip(np.floor(px).astype(np.int64), 0, W - 1)
    py = np.clip(np.floor(py).astype(np.int64), 0, H - 1)

    img = np.zeros((5, H, W), np.float32)
    mask = np.zeros((H, W), bool)
    lab = np.full((H, W), 255, np.int64) if labels is not None else None
    order = np.argsort(-r)                    # 远→近，近点最后写覆盖远点
    yy, xx = py[order], px[order]
    img[0, yy, xx] = r[order]
    img[1, yy, xx] = x[order]; img[2, yy, xx] = y[order]; img[3, yy, xx] = z[order]
    img[4, yy, xx] = inten[order]
    mask[yy, xx] = True
    if labels is not None:
        lab[yy, xx] = labels[order]
    return img, mask, px, py, lab


# 5 通道归一化(range/x/y/z/intensity)：均值/尺度按数据大致量级，稳定训练
IMG_MEAN = np.array([10.0, 0.0, 0.0, -1.0, 0.25], np.float32).reshape(5, 1, 1)
IMG_STD = np.array([12.0, 12.0, 12.0, 2.0, 0.15], np.float32).reshape(5, 1, 1)


def normalize(img, mask):
    out = (img - IMG_MEAN) / IMG_STD
    out *= mask[None]                          # 空像素置 0
    return out
