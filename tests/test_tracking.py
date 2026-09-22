"""多目标跟踪测试（纯 numpy）。"""
import numpy as np

from kitti_slam.tracking import MOT
from kitti_slam.mcl import pose_2d, rel_motion


def test_moving_point_tracked_with_correct_speed():
    mot = MOT(gate=2.0, min_hits=2)
    v = np.array([0.8, 0.0])                      # 0.8 m/帧
    for i in range(10):
        mot.step(np.array([[i * v[0], 0.0]]), t=i * 0.1, dt=0.1)
    tr = mot.all_tracks(min_hits=2)
    assert len(tr) == 1
    assert abs(tr[0].speed - 8.0) < 2.0           # 0.8m/0.1s = 8 m/s


def test_static_point_low_net_displacement():
    mot = MOT(min_hits=2)
    for i in range(12):
        mot.step(np.array([[5.0, 5.0]]), t=i * 0.1, dt=0.1)
    h = np.array([(x, y) for _, x, y in mot.all_tracks(2)[0].history])
    assert np.linalg.norm(h[-1] - h[0]) < 0.5     # 净位移≈0 → 静态


def test_two_objects_get_two_tracks():
    mot = MOT(gate=2.0, min_hits=2)
    for i in range(8):
        mot.step(np.array([[i * 0.5, 0.0], [i * 0.5, 20.0]]), t=i * 0.1, dt=0.1)
    assert len(mot.all_tracks(min_hits=2)) == 2


def test_rel_motion_and_pose_2d():
    import math
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:2, :2] = [[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]]
    T1[0, 3] = 1.0
    x, y, yaw = pose_2d(T1)
    assert abs(x - 1.0) < 1e-9 and abs(yaw - 0.2) < 1e-9
    dx, dy, dyaw = rel_motion(T0, T1)
    assert abs(dx - 1.0) < 1e-9 and abs(dyaw - 0.2) < 1e-9
