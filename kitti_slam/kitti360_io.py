"""KITTI-360 (HDL-64E 城区大场景) 读取，接口对齐 kitti_io，SLAM 栈原样复用。

- velodyne 帧连续编号：data_3d_raw/<drive>/velodyne_points/data/{f:010d}.bin (Nx4 float32)
- 真值位姿稀疏(非每帧)：cam0_to_world.txt 给 T_world_cam0，转 velodyne：
      T_world_velo = T_world_cam0 @ inv(T_velo_cam0),  T_velo_cam0 来自 calib_cam_to_velo.txt
  (KITTI-360 标定方向是 cam->velo，与 KITTI odometry 的 Tr(velo->cam0) 相反，故取逆。)

里程计在连续 velodyne 帧上跑；ATE/RPE 按**绝对帧号**只在有真值的帧上对齐评测。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DATAROOT = Path(os.environ.get(
    'KITTI360_ROOT', '/media/4T/cst/DATASETS/SLAM/KITTI_360/unpacked/KITTI-360'))


def drive_name(drive) -> str:
    return f'2013_05_28_drive_{int(drive):04d}_sync'


def velo_dir(drive) -> Path:
    return DATAROOT / 'data_3d_raw' / drive_name(drive) / 'velodyne_points' / 'data'


def read_velodyne(drive, frame) -> np.ndarray:
    """一帧点云 (N,4) x/y/z/reflectance，按绝对帧号读取。"""
    p = velo_dir(drive) / f'{int(frame):010d}.bin'
    return np.fromfile(p, dtype=np.float32).reshape(-1, 4)


def num_velodyne(drive) -> int:
    """连续 velodyne 帧数(0..num-1)。"""
    return len(list(velo_dir(drive).glob('*.bin')))


def _mat4(vals) -> np.ndarray:
    """12 或 16 个数 → 4x4 齐次矩阵。"""
    v = np.asarray(vals, dtype=np.float64).ravel()
    T = np.eye(4)
    T[:3, :4] = v[:12].reshape(3, 4) if v.size == 12 else v.reshape(4, 4)[:3, :4]
    return T


def read_calib() -> dict:
    """T_velo_cam0 (calib_cam_to_velo.txt, 3x4) 及其逆。"""
    txt = (DATAROOT / 'calibration' / 'calib_cam_to_velo.txt').read_text().split()
    T_velo_cam0 = _mat4([float(x) for x in txt])
    return {'T_velo_cam0': T_velo_cam0, 'T_cam0_velo': np.linalg.inv(T_velo_cam0)}


def read_cam0_to_world(drive) -> dict:
    """{frame_idx(int) -> T_world_cam0(4x4)}，来自 cam0_to_world.txt（每行 frame + 4x4）。"""
    p = DATAROOT / 'data_poses' / drive_name(drive) / 'cam0_to_world.txt'
    out = {}
    for line in p.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        out[int(t[0])] = _mat4([float(x) for x in t[1:17]])
    return out


def gt_poses_velodyne(drive):
    """真值 velodyne 位姿。返回 (frames[int, M], poses[M,4,4])，按帧号升序。

    T_world_velo = T_world_cam0 @ inv(T_velo_cam0)。仅含既有真值又有 velodyne 的帧。
    """
    cw = read_cam0_to_world(drive)
    T_cam0_velo = read_calib()['T_cam0_velo']
    nv = num_velodyne(drive)
    frames = sorted(f for f in cw if 0 <= f < nv)
    poses = np.array([cw[f] @ T_cam0_velo for f in frames])
    return np.array(frames, dtype=int), poses


def longest_dense_window(frames):
    """稀疏真值帧号里最长的 step-1 连续段，返回 (start_frame, length)。"""
    frames = np.asarray(frames, dtype=int)
    best_s, best_len, s = frames[0], 1, frames[0]
    for a, b in zip(frames[:-1], frames[1:]):
        if b != a + 1:
            if a - s + 1 > best_len:
                best_len, best_s = a - s + 1, s
            s = b
    if frames[-1] - s + 1 > best_len:
        best_len, best_s = frames[-1] - s + 1, s
    return int(best_s), int(best_len)


def drives():
    """既有 poses 又有 velodyne 的可用 drive 编号。"""
    out = []
    for d in sorted((DATAROOT / 'data_poses').glob('2013_05_28_drive_*_sync')):
        num = int(d.name.split('_')[4])
        if velo_dir(num).exists() and any(velo_dir(num).glob('*.bin')):
            out.append(num)
    return out
