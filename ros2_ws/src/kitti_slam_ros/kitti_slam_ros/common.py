"""公用：定位工程根目录（好让节点复用 kitti_slam 算法）+ numpy↔PointCloud2 转换。"""
import os
import sys
from pathlib import Path

import numpy as np


def project_root() -> Path:
    """kitti_slam_nav 工程根：优先环境变量 KITTI_SLAM_ROOT，否则按源码相对路径回溯。"""
    env = os.environ.get('KITTI_SLAM_ROOT')
    if env and (Path(env) / 'kitti_slam').is_dir():
        return Path(env)
    # 源码位于 <root>/ros2_ws/src/kitti_slam_ros/kitti_slam_ros/common.py
    for p in Path(__file__).resolve().parents:
        if (p / 'kitti_slam').is_dir() and (p / 'scripts').is_dir():
            return p
    raise RuntimeError('找不到工程根，请设 KITTI_SLAM_ROOT 环境变量')


def add_project_to_path():
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def spin(node):
    """标准 spin + 干净关停（避免 Ctrl-C 时 destroy_node 竞态报栈）。"""
    import rclpy
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


# ---- PointCloud2 <-> numpy（只用 xyz，float32，避免依赖 ros2_numpy）----
def xyz_to_pointcloud2(header, pts):
    """pts:(N,3) float → sensor_msgs/PointCloud2（3×float32 fields）。"""
    from sensor_msgs.msg import PointCloud2, PointField
    pts = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(pts)
    msg.fields = [PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
                  for n, o in (('x', 0), ('y', 4), ('z', 8))]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.is_dense = True
    msg.data = pts.tobytes()
    return msg


def pointcloud2_to_xyz(msg):
    """PointCloud2 → (N,3) float32。按 field offset 抽 x,y,z。"""
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}
    xyz = np.empty((n, 3), np.float32)
    for k, ax in enumerate(('x', 'y', 'z')):
        xyz[:, k] = buf[:, off[ax]:off[ax] + 4].copy().view(np.float32).ravel()
    return xyz
