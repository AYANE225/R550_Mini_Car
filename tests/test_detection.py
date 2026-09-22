"""检测测试（有向框拟合 + 车辆筛 + 端到端聚类；需要 open3d，CI 已安装）。"""
import numpy as np

from kitti_slam.detection import _oriented_box, is_vehicle, Box3D, detect


def test_oriented_box_recovers_dims_and_yaw():
    rng = np.random.default_rng(0)
    yaw = 0.6
    l, w, h = 4.0, 1.8, 1.5
    local = np.column_stack([rng.uniform(-l/2, l/2, 800), rng.uniform(-w/2, w/2, 800),
                             rng.uniform(0, h, 800)])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    pts = local @ R.T + [10, -3, 0]
    b = _oriented_box(pts)
    assert abs(b.l - l) < 0.3 and abs(b.w - w) < 0.3 and abs(b.h - h) < 0.3
    assert abs(((b.yaw - yaw + np.pi/2) % np.pi) - np.pi/2) < 0.15   # 主轴朝向(mod π)


def test_is_vehicle_filter():
    assert is_vehicle(Box3D(0, 0, 0, l=4.2, w=1.8, h=1.5, yaw=0, n=100))
    assert not is_vehicle(Box3D(0, 0, 0, l=0.4, w=0.4, h=3.0, yaw=0, n=100))   # 杆


def test_detect_finds_a_box_on_synthetic_scene():
    rng = np.random.default_rng(1)
    ground = np.column_stack([rng.uniform(-20, 20, 6000), rng.uniform(-20, 20, 6000),
                              rng.normal(-1.72, 0.02, 6000)])                   # 地面
    car = np.column_stack([rng.uniform(8, 12, 500), rng.uniform(-1, 1, 500),
                           rng.uniform(-1.5, 0.0, 500)])                        # 一辆车
    pts = np.vstack([ground, car])
    boxes, obj = detect(pts, min_points=10)
    assert len(boxes) >= 1
    assert any(abs(b.cx - 10) < 2 and abs(b.cy) < 2 for b in boxes)             # 命中车位置
