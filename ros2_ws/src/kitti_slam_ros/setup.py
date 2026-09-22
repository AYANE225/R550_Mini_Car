from glob import glob

from setuptools import find_packages, setup

package_name = 'kitti_slam_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AYANE225',
    maintainer_email='ayane225@users.noreply.github.com',
    description='ROS2 online LiDAR SLAM + detection (dataset/sim replay, RViz2).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'cloud_player = kitti_slam_ros.cloud_player:main',
            'odometry_node = kitti_slam_ros.odometry_node:main',
            'mapping_node = kitti_slam_ros.mapping_node:main',
            'detection_node = kitti_slam_ros.detection_node:main',
        ],
    },
)
