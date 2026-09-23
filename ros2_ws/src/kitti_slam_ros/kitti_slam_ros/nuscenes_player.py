"""nuscenes_player：把 nuScenes(Motional, Velodyne HDL-32E)某场景当**虚拟 LiDAR** 回放，
发布 /velodyne_points——证明这套 ROS2 在线栈换个厂商的激光照样跑，不是 KITTI 专用。

与 cloud_player 接口一致（同 topic/frame_id），下游 odometry/mapping/detection 节点无需改动。
参数：scene(场景序号 0-9)、rate、frames、frame_id、loop。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from .common import add_project_to_path, xyz_to_pointcloud2

add_project_to_path()
from kitti_slam.nuscenes_io import NuScenesLidar          # noqa: E402


class NuScenesPlayer(Node):
    def __init__(self):
        super().__init__('nuscenes_player')
        self.declare_parameter('scene', 0)
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('frames', -1)
        self.declare_parameter('frame_id', 'velodyne')
        self.declare_parameter('loop', False)

        self.frame_id = self.get_parameter('frame_id').value
        self.loop = self.get_parameter('loop').value
        self.nu = NuScenesLidar()
        scene = self.get_parameter('scene').value
        self.recs = self.nu.scene_lidar_frames(scene)
        f = self.get_parameter('frames').value
        self.n = len(self.recs) if f < 0 else min(f, len(self.recs))
        self.name = self.nu.scenes()[scene][0]
        self.pub = self.create_publisher(PointCloud2, '/velodyne_points', 10)
        self.i = 0
        rate = float(self.get_parameter('rate').value)
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'nuscenes_player: {self.name} (HDL-32E), {self.n} frames @ {rate} Hz '
                               f'-> /velodyne_points (frame {self.frame_id})')

    def tick(self):
        if self.i >= self.n:
            if self.loop:
                self.i = 0
            else:
                self.get_logger().info('playback done'); self.timer.cancel(); return
        pts = self.nu.load_points(self.recs[self.i])[:, :3]
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        self.pub.publish(xyz_to_pointcloud2(header, pts))
        if self.i % 50 == 0:
            self.get_logger().info(f'  published frame {self.i}/{self.n} ({len(pts)} pts)')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    from .common import spin
    spin(NuScenesPlayer())


if __name__ == '__main__':
    main()
