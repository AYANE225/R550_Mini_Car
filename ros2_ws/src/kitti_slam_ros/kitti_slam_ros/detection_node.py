"""detection_node：订阅 /velodyne_points，跑几何车辆检测（地面 RANSAC→DBSCAN→PCA 有向框，
纯 numpy/open3d，不依赖 torch），发布 visualization_msgs/MarkerArray(/detections) 供 RViz 画框。

注：DL 版 PointPillars 依赖 torch(在 vggt-slam env 训好)，跨 conda 环境不便直接进 ROS2 节点，
这里用同工程的经典几何检测器做在线检测；两者接口一致（Box3D），需要时可换。
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray

from .common import add_project_to_path, pointcloud2_to_xyz

add_project_to_path()
from kitti_slam.detection import detect          # noqa: E402


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.declare_parameter('sensor_frame', 'velodyne')
        self.declare_parameter('voxel', 0.2)
        self.frame = self.get_parameter('sensor_frame').value
        self.voxel = float(self.get_parameter('voxel').value)
        self.sub = self.create_subscription(PointCloud2, '/velodyne_points', self.on_cloud, 5)
        self.pub = self.create_publisher(MarkerArray, '/detections', 5)
        self.k = 0
        self.get_logger().info('detection_node: /velodyne_points -> /detections (MarkerArray)')

    def on_cloud(self, msg: PointCloud2):
        pts = pointcloud2_to_xyz(msg)
        boxes, _ = detect(pts, vehicles_only=True, voxel=self.voxel, ransac_iters=80, min_points=8)
        arr = MarkerArray()
        clear = Marker(); clear.header = msg.header; clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for j, b in enumerate(boxes):
            m = Marker()
            m.header = msg.header; m.ns = 'vehicles'; m.id = j
            m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = b.cx, b.cy, b.cz
            m.pose.orientation.z = float(np.sin(b.yaw / 2))
            m.pose.orientation.w = float(np.cos(b.yaw / 2))
            m.scale.x, m.scale.y, m.scale.z = max(b.l, 0.2), max(b.w, 0.2), max(b.h, 0.2)
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.82, 0.3, 0.45
            arr.markers.append(m)
        self.pub.publish(arr)
        self.k += 1
        if self.k % 50 == 0:
            self.get_logger().info(f'  frame {self.k}: {len(boxes)} vehicles')


def main(args=None):
    rclpy.init(args=args)
    from .common import spin
    spin(DetectionNode())


if __name__ == '__main__':
    main()
