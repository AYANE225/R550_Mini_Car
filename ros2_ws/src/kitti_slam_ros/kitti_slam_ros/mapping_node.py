"""mapping_node：订阅 /velodyne_points + /odom，用当前位姿把扫描转世界系累积成地图，
体素下采样限规模，定期发布 /map (PointCloud2, map 系)。在线增量建图。
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from .common import add_project_to_path, pointcloud2_to_xyz, xyz_to_pointcloud2

add_project_to_path()


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        self.declare_parameter('voxel', 0.3)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_every', 5)      # 每几帧发一次 /map
        self.declare_parameter('range_max', 45.0)       # 只累积近距点(去远处噪声)
        self.voxel = float(self.get_parameter('voxel').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.every = int(self.get_parameter('publish_every').value)
        self.rmax = float(self.get_parameter('range_max').value)

        self.pose = None
        self.map_pts = np.empty((0, 3), np.float32)
        self.k = 0
        self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.create_subscription(PointCloud2, '/velodyne_points', self.on_cloud, 10)
        self.pub = self.create_publisher(PointCloud2, '/map', 1)
        self.get_logger().info('mapping_node: (/velodyne_points, /odom) -> /map')

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose
        R = quat_to_rot(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        T = np.eye(4); T[:3, :3] = R
        T[:3, 3] = [p.position.x, p.position.y, p.position.z]
        self.pose = T

    def _voxel(self, P):
        key = np.floor(P / self.voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        return P[idx]

    def on_cloud(self, msg: PointCloud2):
        if self.pose is None:
            return
        pts = pointcloud2_to_xyz(msg)
        r = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[(r > 3.0) & (r < self.rmax)]
        world = (pts @ self.pose[:3, :3].T + self.pose[:3, 3]).astype(np.float32)
        self.map_pts = self._voxel(np.vstack([self.map_pts, world]))
        self.k += 1
        if self.k % self.every == 0:
            header = Header(); header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.map_frame
            self.pub.publish(xyz_to_pointcloud2(header, self.map_pts))
            if self.k % 50 == 0:
                self.get_logger().info(f'  map {len(self.map_pts):,} pts')


def main(args=None):
    rclpy.init(args=args)
    from .common import spin
    spin(MappingNode())


if __name__ == '__main__':
    main()
