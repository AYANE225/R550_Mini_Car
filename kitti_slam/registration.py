"""点云预处理与配准（基于 Open3D，标准 point-to-plane ICP；无任何研究代码）。"""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp


def deskew(points, delta):
    """运动畸变去除：旋转雷达一圈内各点按方位角推得的采集时刻，用上一帧相对运动(delta,
    匀速假设)把点补偿到"扫描起始"时刻。KITTI 10Hz 下一帧车走 ~1m，去畸变对高速段尤为关键。

    points: (N,>=3) 雷达系原始点；delta: 4x4 上一帧 body 系相对运动 T_{t-1->t}。返回 (N,3)。
    """
    xyz = points[:, :3].astype(np.float64)
    R_delta = delta[:3, :3]
    t_delta = delta[:3, 3]
    if Rotation.from_matrix(R_delta).magnitude() < 1e-4 and np.linalg.norm(t_delta) < 1e-3:
        return xyz
    az = np.arctan2(xyz[:, 1], xyz[:, 0])              # [-pi,pi]
    alpha = (np.pi - az) / (2 * np.pi)                 # [0,1]：velodyne 顺时针转,时间随方位角减小而增
    # 去畸变到"扫描中点"参考(KISS-ICP 式)：把各点从其采集时刻 α 补到 0.5 时刻，
    # 校正量对称减半、对匀速先验的反馈更稳（补到起点会引入振荡发散）。
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([np.eye(3), R_delta])))
    RhT = slerp(0.5).as_matrix().T
    R_a = slerp(alpha).as_matrix()                     # (N,3,3)
    Rrel = np.einsum('ij,njk->nik', RhT, R_a)
    return np.einsum('nij,nj->ni', Rrel, xyz) + (RhT @ t_delta) * (alpha - 0.5)[:, None]


def to_o3d(points_xyz):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points_xyz[:, :3], dtype=np.float64))
    return pc


def preprocess(points, voxel=0.5, min_range=3.0, max_range=80.0, normal_radius=1.0):
    """裁剪自车/远点 → 体素下采样 → 估计法线（point-to-plane 需要）。返回 o3d 点云。"""
    xyz = points[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    xyz = xyz[(r > min_range) & (r < max_range)]
    pc = to_o3d(xyz).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    return pc


def icp_point_to_plane(source, target, init=np.eye(4), max_dist=1.0, max_iter=30):
    """point-to-plane ICP：把 source 配到 target。返回 (T_target_source 4x4, fitness, rmse)。"""
    reg = o3d.pipelines.registration.registration_icp(
        source, target, max_dist, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    return np.asarray(reg.transformation), reg.fitness, reg.inlier_rmse
