"""一键起 KITTI 回放式 ROS2 SLAM+检测 demo：cloud_player + odometry + mapping + detection (+ RViz2)。

用法：
  ros2 launch kitti_slam_ros slam_demo.launch.py seq:=0 rate:=10.0 rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    seq = LaunchConfiguration('seq')
    rate = LaunchConfiguration('rate')
    frames = LaunchConfiguration('frames')
    use_rviz = LaunchConfiguration('rviz')
    rviz_cfg = os.path.join(get_package_share_directory('kitti_slam_ros'), 'rviz', 'slam_demo.rviz')

    return LaunchDescription([
        DeclareLaunchArgument('seq', default_value='0'),
        DeclareLaunchArgument('rate', default_value='10.0'),
        DeclareLaunchArgument('frames', default_value='-1'),
        DeclareLaunchArgument('rviz', default_value='true'),

        Node(package='kitti_slam_ros', executable='cloud_player', name='cloud_player',
             parameters=[{'seq': seq, 'rate': rate, 'frames': frames}], output='screen'),
        Node(package='kitti_slam_ros', executable='odometry_node', name='odometry_node',
             output='screen'),
        Node(package='kitti_slam_ros', executable='mapping_node', name='mapping_node',
             output='screen'),
        Node(package='kitti_slam_ros', executable='detection_node', name='detection_node',
             output='screen'),
        Node(package='rviz2', executable='rviz2', name='rviz2',
             arguments=['-d', rviz_cfg], condition=IfCondition(use_rviz), output='screen'),
    ])
