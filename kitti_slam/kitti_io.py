"""KITTI Odometry 数据读取（公开数据集，与任何研究代码无关）。

约定：
- velodyne/*.bin  : float32 Nx4 (x,y,z,反射率)，激光雷达坐标系。
- poses/SS.txt    : 每行 3x4 行主序，T_world_cam0(t)（**真值，仅用于评测**）。
- calib.txt       : P0..P3 + Tr(velo->cam0)。unpacked 版缺 Tr，用 calibration_official_v1。
- times.txt       : 每帧时间戳(秒)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 数据根：velodyne/poses/times 用 unpacked；带 Tr 的完整标定用 official。
UNPACKED = Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked/dataset')
CALIB_OFFICIAL = Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/calibration_official_v1/dataset')


def seq_dir(seq):
    return UNPACKED / 'sequences' / f'{int(seq):02d}'


def read_velodyne(seq, frame):
    p = seq_dir(seq) / 'velodyne' / f'{int(frame):06d}.bin'
    return np.fromfile(p, dtype=np.float32).reshape(-1, 4)


def num_frames(seq):
    return len(list((seq_dir(seq) / 'velodyne').glob('*.bin')))


def _parse_calib(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, vals = line.split(':', 1)
        out[key.strip()] = np.array([float(x) for x in vals.split()], dtype=np.float64)
    return out


def read_calib(seq):
    """返回 dict：Tr(4x4 velo->cam0), Tr_inv(cam0->velo), P2(3x4)。Tr 取 official 版。"""
    c_off = _parse_calib(CALIB_OFFICIAL / 'sequences' / f'{int(seq):02d}' / 'calib.txt')
    Tr = np.eye(4)
    Tr[:3] = c_off['Tr'].reshape(3, 4)
    P2 = c_off.get('P2', np.zeros(12)).reshape(3, 4)
    return {'Tr': Tr, 'Tr_inv': np.linalg.inv(Tr), 'P2': P2}


def read_gt_poses(seq):
    """真值：返回 (N,4,4) 的 T_world_cam0。仅评测用。"""
    raw = np.loadtxt(UNPACKED / 'poses' / f'{int(seq):02d}.txt', dtype=np.float64, ndmin=2)
    poses = np.repeat(np.eye(4)[None], len(raw), axis=0)
    poses[:, :3] = raw.reshape(-1, 3, 4)
    return poses


def gt_poses_velodyne(seq):
    """真值转到 velodyne 坐标系：T_world_velo = T_world_cam0 @ Tr。与激光里程计同系，便于比对。"""
    poses = read_gt_poses(seq)
    Tr = read_calib(seq)['Tr']
    return poses @ Tr


def read_times(seq):
    return np.loadtxt(seq_dir(seq) / 'times.txt', dtype=np.float64)
