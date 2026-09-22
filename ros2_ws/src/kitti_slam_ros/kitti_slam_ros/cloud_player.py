"""cloud_player：把数据集（KITTI velodyne）当**虚拟 LiDAR** 按帧率发布 PointCloud2。

这是数据集/仿真回放式的"传感器源"——工业里调 SLAM 的标准做法。发布 /velodyne_points，
frame_id=velodyne，按 --rate Hz 推进；到末尾可循环或停止。
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from .common import add_project_to_path, xyz_to_pointcloud2

add_project_to_path()
from kitti_slam import kitti_io as io          # noqa: E402


class CloudPlayer(Node):
    def __init__(self):
        super().__init__('cloud_player')
        self.declare_parameter('seq', 0)
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('start', 0)
        self.declare_parameter('frames', -1)          # -1=整条序列
        self.declare_parameter('frame_id', 'velodyne')
        self.declare_parameter('loop', False)

        self.seq = self.get_parameter('seq').value
        self.frame_id = self.get_parameter('frame_id').value
        self.loop = self.get_parameter('loop').value
        self.start = self.get_parameter('start').value
        total = io.num_frames(self.seq)
        f = self.get_parameter('frames').value
        self.n = total - self.start if f < 0 else min(f, total - self.start)
        self.pub = self.create_publisher(PointCloud2, '/velodyne_points', 10)
        self.i = 0
        rate = float(self.get_parameter('rate').value)
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'cloud_player: seq{self.seq:02d}, {self.n} frames @ {rate} Hz '
                               f'-> /velodyne_points (frame {self.frame_id})')

    def tick(self):
        if self.i >= self.n:
            if self.loop:
                self.i = 0
            else:
                self.get_logger().info('playback done'); self.timer.cancel(); return
        pts = io.read_velodyne(self.seq, self.start + self.i)[:, :3]
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
    spin(CloudPlayer())


if __name__ == '__main__':
    main()
