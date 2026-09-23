"""IMU 预积分单测（合成数据，无数据依赖）。"""
import numpy as np

from kitti_slam import imu


def test_so3_exp_is_rotation():
    for w in [np.zeros(3), np.array([0, 0, np.pi / 2]), np.array([0.3, -0.2, 0.1])]:
        R = imu.so3_exp(w)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
    R = imu.so3_exp(np.array([0, 0, np.pi / 2]))
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-6)


def test_level_constant_velocity():
    """水平静止姿态、加计只含重力、陀螺为零 → 应保持匀速直线。"""
    g = 9.81
    M = 20
    dt = np.full(M, 0.1)
    accel = np.tile([0, 0, g], (M, 1))          # 车体系水平，只测到重力反力(+z)
    gyro = np.zeros((M, 3))
    v0 = np.array([10.0, 0, 0])
    Rs, ps, vs = imu.preintegrate(np.eye(3), np.zeros(3), v0, accel, gyro, dt, g)
    assert np.allclose(vs[-1], v0, atol=1e-6)     # 去重力后无净加速度
    assert np.allclose(ps[-1], [10.0 * M * 0.1, 0, 0], atol=1e-6)
    assert np.allclose(Rs[-1], np.eye(3), atol=1e-9)


def test_to_velo_frame_identity():
    a = np.random.randn(5, 3); w = np.random.randn(5, 3)
    av, wv = imu.to_velo_frame(a, w, np.eye(3))
    assert np.allclose(av, a) and np.allclose(wv, w)
