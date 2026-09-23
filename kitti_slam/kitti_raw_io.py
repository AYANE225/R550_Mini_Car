"""KITTI-raw 读取：velodyne + OXTS 真 IMU（加计/陀螺/速度/姿态）+ imu↔velo 外参。

KITTI odometry 那套只有位姿真值、没有 IMU；这里读 KITTI-**raw** 的 OXTS 惯导单元(RT3003)：
10 Hz 的三轴加计 ax/ay/az(含重力,车体系 x前-y左-z上)、三轴陀螺 wx/wy/wz、前/左/上速度、
roll/pitch/yaw 与经纬高。用它做从零 IMU 预积分(见 imu.py)并演示"LiDAR 盲区惯导补位"。
真值位姿由经纬高(墨卡托投影)+ rpy 推出 T_world_imu，首帧置原点。纯 numpy。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# raw 数据解包位置：<RAW>/<date>/<date>_drive_XXXX_sync/{velodyne_points,oxts}
RAW = Path('/media/4T/cst/DATASETS/SLAM/KITTI_raw/unpacked')


def drive_dir(date, drive):
    return RAW / date / f'{date}_drive_{drive:04d}_sync'


def read_velodyne(date, drive, i):
    """一帧激光点云 (N,4) [x,y,z,intensity]，velodyne 体系。"""
    f = drive_dir(date, drive) / 'velodyne_points' / 'data' / f'{i:010d}.bin'
    return np.fromfile(f, np.float32).reshape(-1, 4)


def n_frames(date, drive):
    return len(list((drive_dir(date, drive) / 'velodyne_points' / 'data').glob('*.bin')))


def read_timestamps(date, drive, sensor='oxts'):
    """各帧绝对时间(秒，相对首帧)。用于预积分的 dt。"""
    lines = (drive_dir(date, drive) / sensor / 'timestamps.txt').read_text().strip().splitlines()
    ts = []
    for ln in lines:
        hms = ln.split(' ')[1]
        h, m, s = hms.split(':')
        ts.append(int(h) * 3600 + int(m) * 60 + float(s))
    ts = np.array(ts)
    return ts - ts[0]


def read_oxts(date, drive):
    """读全序列 OXTS，返回 dict：
      accel (N,3) 车体系 [ax,ay,az](含重力)、gyro (N,3) [wx,wy,wz]、
      vel_body (N,3) [vf,vl,vu]、rpy (N,3)、lla (N,3)。
    """
    dd = drive_dir(date, drive) / 'oxts' / 'data'
    files = sorted(dd.glob('*.txt'))
    M = np.array([np.fromstring(f.read_text(), sep=' ') for f in files])
    return dict(lla=M[:, 0:3], rpy=M[:, 3:6], vel_body=M[:, 8:11],
                accel=M[:, 11:14], gyro=M[:, 17:20])


def _rpy_to_R(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll)，OXTS 约定。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def oxts_to_poses(lla, rpy):
    """经纬高(墨卡托)+ rpy → T_world_imu (N,4,4)，首帧置原点。"""
    lat0 = lla[0, 0]
    er = 6378137.0
    scale = np.cos(lat0 * np.pi / 180.0)
    x = scale * lla[:, 1] * np.pi / 180.0 * er
    y = scale * er * np.log(np.tan((90.0 + lla[:, 0]) * np.pi / 360.0))
    z = lla[:, 2]
    T = np.repeat(np.eye(4)[None], len(lla), axis=0)
    for i in range(len(lla)):
        T[i, :3, :3] = _rpy_to_R(*rpy[i])
        T[i, :3, 3] = [x[i], y[i], z[i]]
    T0inv = np.linalg.inv(T[0])
    return np.einsum('ij,njk->nik', T0inv, T)


def calib_imu_to_velo(date):
    """读 calib_imu_to_velo.txt，返回 (R_iv, t_iv)：v_velo = R_iv @ v_imu + t_iv。"""
    txt = (RAW / date / 'calib_imu_to_velo.txt').read_text().splitlines()
    R = t = None
    for ln in txt:
        if ln.startswith('R:'):
            R = np.fromstring(ln[2:], sep=' ').reshape(3, 3)
        elif ln.startswith('T:'):
            t = np.fromstring(ln[2:], sep=' ')
    return R, t
