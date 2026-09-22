"""画 ROS2 计算图（节点/话题/消息类型）——用于 README 展示 ROS2 封装架构。

纯绘图，不依赖 ROS。用法：python scripts/plot_ros2_graph.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 8)); fig.patch.set_facecolor('#0b0b12')
    ax.set_facecolor('#0b0b12'); ax.set_xlim(0, 15.6); ax.set_ylim(0, 8); ax.axis('off')

    NODE = dict(boxstyle='round,pad=0.35', fc='#16324f', ec='#39d0e0', lw=2)
    TOPIC = dict(boxstyle='round,pad=0.28', fc='#241a30', ec='#c9a0ff', lw=1.5)

    def node(x, y, name, sub):
        ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.42), 2.3, 0.84, **NODE))
        ax.text(x, y + 0.12, name, ha='center', va='center', color='w', fontsize=11, weight='bold')
        ax.text(x, y - 0.2, sub, ha='center', va='center', color='#9ec', fontsize=8)

    def topic(x, y, name, typ):
        ax.add_patch(FancyBboxPatch((x - 1.25, y - 0.32), 2.5, 0.64, **TOPIC))
        ax.text(x, y + 0.08, name, ha='center', va='center', color='#e8d9ff', fontsize=9.5, weight='bold')
        ax.text(x, y - 0.16, typ, ha='center', va='center', color='#a48ac0', fontsize=7)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14,
                                     color='#5a6b7a', lw=1.3, shrinkA=2, shrinkB=2))

    # 传感器源
    node(1.6, 4, 'cloud_player', 'dataset→virtual LiDAR')
    topic(4.6, 4, '/velodyne_points', 'sensor_msgs/PointCloud2')
    arrow(2.75, 4, 3.35, 4)

    # 计算节点
    node(8, 6.2, 'odometry_node', 'scan-to-map ICP')
    node(8, 4, 'detection_node', 'vehicle detection')
    node(8, 1.8, 'mapping_node', 'incremental map')
    for y in (6.2, 4, 1.8):
        arrow(5.85, 4, 6.85, y)

    # 输出话题
    topic(11.6, 6.8, '/odom + /tf', 'nav_msgs/Odometry')
    topic(11.6, 5.6, '/odom_path', 'nav_msgs/Path')
    topic(11.6, 4, '/detections', 'visualization_msgs/MarkerArray')
    topic(11.6, 1.8, '/map', 'sensor_msgs/PointCloud2')
    arrow(9.15, 6.4, 10.35, 6.8)
    arrow(9.15, 6.0, 10.35, 5.6)
    arrow(9.15, 4, 10.35, 4)
    arrow(9.15, 1.8, 10.35, 1.8)
    # odom 反馈给 mapping（位姿）
    arrow(11.6, 6.46, 11.6, 6.9)
    ax.add_patch(FancyArrowPatch((10.35, 6.7), (9.15, 2.1), arrowstyle='-|>', mutation_scale=12,
                                 color='#8a5a5a', lw=1.1, ls=(0, (4, 3)),
                                 connectionstyle='arc3,rad=0.3'))
    ax.text(10.0, 3.4, 'pose', color='#c88', fontsize=7.5, rotation=72)

    # RViz
    node(13.9, 4, 'RViz2', 'live 3D view')
    for y in (6.8, 5.6, 4, 1.8):
        arrow(12.85, y, 13.15, 4.0 + (y - 4) * 0.15)

    ax.text(0.2, 7.6, 'ROS2 (Humble) online SLAM + perception graph', color='w',
            fontsize=15, weight='bold')
    ax.text(0.2, 7.15, 'dataset/sim replay  →  self-built odometry / mapping / detection nodes  →  '
            'RViz2  ·  one launch file  ·  ros2 bag record & replay', color='#8899aa', fontsize=10)
    out = ROOT / 'reports' / 'ros2_graph.png'
    fig.savefig(out, dpi=140, facecolor='#0b0b12', bbox_inches='tight')
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
