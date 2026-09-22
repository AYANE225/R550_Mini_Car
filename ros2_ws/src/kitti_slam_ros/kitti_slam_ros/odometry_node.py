"""odometry_node：订阅 /velodyne_points，跑自研 LidarOdometry（scan-to-local-map 点面 ICP
+ 匀速先验），在线输出 nav_msgs/Odometry(/odom)、TF(map→velodyne)、轨迹(/odom_path)。

复用工程里的 kitti_slam.odometry.LidarOdometry，不重写算法——ROS2 只是把它接成在线节点。
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster

from .common import add_project_to_path, pointcloud2_to_xyz

add_project_to_path()
from kitti_slam.odometry import LidarOdometry          # noqa: E402


def rot_to_quat(R):
    """3x3 旋转矩阵 → 四元数 (x,y,z,w)。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.declare_parameter('voxel', 0.5)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('sensor_frame', 'velodyne')
        self.map_frame = self.get_parameter('map_frame').value
        self.sensor_frame = self.get_parameter('sensor_frame').value
        self.odom = LidarOdometry(voxel=float(self.get_parameter('voxel').value))

        self.sub = self.create_subscription(PointCloud2, '/velodyne_points', self.on_cloud, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.pub_path = self.create_publisher(Path, '/odom_path', 10)
        self.tf = TransformBroadcaster(self)
        self.path = Path()
        self.k = 0
        self.get_logger().info('odometry_node: /velodyne_points -> /odom + TF(map->velodyne)')

    def on_cloud(self, msg: PointCloud2):
        pts = pointcloud2_to_xyz(msg)
        pose = self.odom.process(pts)                  # T_world_velo (4x4)
        stamp = msg.header.stamp
        q = rot_to_quat(pose[:3, :3])
        t = pose[:3, 3]

        tfm = TransformStamped()
        tfm.header.stamp = stamp; tfm.header.frame_id = self.map_frame
        tfm.child_frame_id = self.sensor_frame
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, t)
        (tfm.transform.rotation.x, tfm.transform.rotation.y,
         tfm.transform.rotation.z, tfm.transform.rotation.w) = map(float, q)
        self.tf.sendTransform(tfm)

        od = Odometry()
        od.header.stamp = stamp; od.header.frame_id = self.map_frame
        od.child_frame_id = self.sensor_frame
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = map(float, t)
        (od.pose.pose.orientation.x, od.pose.pose.orientation.y,
         od.pose.pose.orientation.z, od.pose.pose.orientation.w) = map(float, q)
        self.pub_odom.publish(od)

        ps = PoseStamped(); ps.header = od.header; ps.pose = od.pose.pose
        self.path.header = od.header
        self.path.poses.append(ps)
        self.pub_path.publish(self.path)
        self.k += 1
        if self.k % 50 == 0:
            self.get_logger().info(f'  odom frame {self.k}: pos=({t[0]:.1f},{t[1]:.1f},{t[2]:.1f})')


def main(args=None):
    rclpy.init(args=args)
    from .common import spin
    spin(OdometryNode())


if __name__ == '__main__':
    main()
