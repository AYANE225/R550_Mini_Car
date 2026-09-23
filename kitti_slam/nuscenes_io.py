"""nuScenes(Motional/nuTonomy)LiDAR 读取——公开自动驾驶数据集，与任何研究代码无关。

用来证明本工程的 SLAM 栈不只吃 KITTI：nuScenes 用 **Velodyne HDL-32E**（32 线，KITTI 是
HDL-64E 64 线；采样更稀、旋转 20Hz），采集于波士顿/新加坡，是不同厂商配置的真实车载激光。

不依赖 nuscenes-devkit，直接解析 v1.0 的 json：
- sample_data : 每条传感器数据（点云在 samples/ 或 sweeps/ 下的 *.pcd.bin，float32 Nx5=x,y,z,intensity,ring）。
- ego_pose    : 车体在全局地图系的位姿（translation + 四元数 wxyz），nuScenes 用地图定位给出，作参考真值。
- calibrated_sensor : 传感器相对车体的外参。
- sensor / scene / sample : 元信息与场景分段。
LiDAR 全局位姿 = T_global_ego @ T_ego_lidar，与激光里程计做 ATE 对齐（Umeyama，坐标系无关）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATAROOT = Path('/media/4T/cst/DATASETS/SLAM/nuScenes_mini')


def _quat_wxyz_to_R(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),     1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),     s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ])


def _pose(translation, rotation_wxyz):
    T = np.eye(4)
    T[:3, :3] = _quat_wxyz_to_R(rotation_wxyz)
    T[:3, 3] = translation
    return T


class NuScenesLidar:
    def __init__(self, dataroot=DATAROOT, version='v1.0-mini'):
        self.root = Path(dataroot)
        meta = self.root / version

        def load(name):
            return json.loads((meta / f'{name}.json').read_text())

        self.scene = load('scene')
        self.sample_data = {r['token']: r for r in load('sample_data')}
        self.ego_pose = {r['token']: r for r in load('ego_pose')}
        self.calib = {r['token']: r for r in load('calibrated_sensor')}
        sample = load('sample')
        self.sample_scene = {r['token']: r['scene_token'] for r in sample}
        sensor = {r['token']: r for r in load('sensor')}
        # LIDAR_TOP 传感器 token；哪些 calibrated_sensor 属于该雷达
        lidar_sensor = {t for t, s in sensor.items() if s['channel'] == 'LIDAR_TOP'}
        self.lidar_calib = {t for t, c in self.calib.items() if c['sensor_token'] in lidar_sensor}

    def scenes(self):
        """返回 [(name, scene_token, nbr_samples), ...]。"""
        return [(s['name'], s['token'], s['nbr_samples']) for s in self.scene]

    def scene_lidar_frames(self, scene=0):
        """某场景的 LIDAR_TOP 数据流（关键帧+sweeps，按时间排序）→ sample_data 记录列表。"""
        if isinstance(scene, int):
            tok = self.scene[scene]['token']
        else:  # 按名字或 token
            match = [s for s in self.scene if scene in (s['name'], s['token'])]
            tok = match[0]['token']
        recs = [r for r in self.sample_data.values()
                if r['calibrated_sensor_token'] in self.lidar_calib
                and self.sample_scene.get(r['sample_token']) == tok]
        recs.sort(key=lambda r: r['timestamp'])
        return recs

    def load_points(self, rec):
        """读该帧 LiDAR 点云（雷达系，z 朝上），返回 Nx4 (x,y,z,intensity)。"""
        p = self.root / rec['filename']
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 5)
        return pts[:, :4]

    def lidar_global_pose(self, rec):
        """该帧 LiDAR 在全局系的位姿 T_global_lidar = T_global_ego @ T_ego_lidar（参考真值）。"""
        ego = self.ego_pose[rec['ego_pose_token']]
        cs = self.calib[rec['calibrated_sensor_token']]
        T_ge = _pose(ego['translation'], ego['rotation'])
        T_el = _pose(cs['translation'], cs['rotation'])
        return T_ge @ T_el
