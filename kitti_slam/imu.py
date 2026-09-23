"""从零 IMU 预积分（strap-down 机械编排）：陀螺→姿态、加计(去重力)→速度/位置。

不依赖任何 IMU 库。给定起始姿态/位置/速度与一段车体系加计(含重力)、陀螺、各步 dt，
按经典捷联积分推进：
  R_{k+1} = R_k · Exp(ω_k·dt)                 陀螺积分姿态
  a_w = R_k·a_body + g_world                  去重力得世界系真加速度 (g_world=[0,0,-g])
  v_{k+1} = v_k + a_w·dt                       积分速度
  p_{k+1} = p_k + v_k·dt + ½·a_w·dt²           积分位置
纯惯导会随时间平方漂移——正是要 LiDAR 来约束；但 LiDAR 盲区/退化的几秒里，IMU 能把位姿
稳稳桥过去(见 run_lio.py)。加计/陀螺若在 imu 体系，先用 imu→velo 外参旋到 velodyne 体系。
"""
from __future__ import annotations

import numpy as np


def so3_exp(w):
    """SO(3) 指数映射（Rodrigues）：旋转向量 w(3,) → 3×3 旋转矩阵。"""
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def estimate_gravity(accel_body, rpy):
    """用近水平段估计重力大小：把车体加计按姿态旋到世界系，取竖直分量中位数。"""
    from .kitti_raw_io import _rpy_to_R
    gz = []
    for a, e in zip(accel_body, rpy):
        aw = _rpy_to_R(*e) @ a
        gz.append(aw[2])
    return float(np.median(gz))


def preintegrate(R0, p0, v0, accel_body, gyro_body, dt, g):
    """从 (R0,p0,v0) 起，用车体系加计/陀螺积分一段轨迹。

    accel_body/gyro_body: (M,3) 已在与 R0 同一体系(velodyne 体系)下的加计(含重力)/角速度。
    dt: (M,) 每步时长。g: 重力大小(m/s²，世界系 -z)。
    返回 Rs (M+1,3,3), ps (M+1,3), vs (M+1,3)，含起点。
    """
    M = len(accel_body)
    Rs = np.zeros((M + 1, 3, 3)); ps = np.zeros((M + 1, 3)); vs = np.zeros((M + 1, 3))
    Rs[0], ps[0], vs[0] = R0, p0, v0
    g_world = np.array([0, 0, -g])
    for k in range(M):
        a_w = Rs[k] @ accel_body[k] + g_world
        ps[k + 1] = ps[k] + vs[k] * dt[k] + 0.5 * a_w * dt[k] ** 2
        vs[k + 1] = vs[k] + a_w * dt[k]
        Rs[k + 1] = Rs[k] @ so3_exp(gyro_body[k] * dt[k])
    return Rs, ps, vs


def to_velo_frame(accel_imu, gyro_imu, R_iv):
    """imu 体系的加计/陀螺旋到 velodyne 体系：v_velo = R_iv · v_imu。"""
    return accel_imu @ R_iv.T, gyro_imu @ R_iv.T
