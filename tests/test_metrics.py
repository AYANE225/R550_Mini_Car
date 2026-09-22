"""评测函数测试（纯 numpy/scipy，不依赖 KITTI 数据/GPU）。"""
import numpy as np
from scipy.spatial.transform import Rotation

from kitti_slam.metrics import umeyama_se3, ate, kitti_rpe, path_length


def _random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    p = np.zeros(3)
    yaw = 0.0
    for i in range(1, n):
        yaw += rng.normal(0, 0.05)
        p = p + Rotation.from_euler('z', yaw).apply([1.0, 0, 0])   # ~1m/帧前进
        poses[i, :3, :3] = Rotation.from_euler('z', yaw).as_matrix()
        poses[i, :3, 3] = p
    return poses


def test_umeyama_recovers_rigid_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(200, 3))
    R = Rotation.from_euler('xyz', [0.3, -0.5, 1.1]).as_matrix()
    t = np.array([3.0, -2.0, 1.0])
    dst = src @ R.T + t
    R_est, t_est = umeyama_se3(src, dst)
    assert np.allclose(R_est, R, atol=1e-6)
    assert np.allclose(t_est, t, atol=1e-6)


def test_ate_zero_for_identical():
    poses = _random_poses(50)
    m = ate(poses.copy(), poses.copy())
    assert m['ate_rmse_m'] < 1e-9


def test_ate_invariant_to_global_rigid_transform():
    poses = _random_poses(60)
    R = Rotation.from_euler('z', 0.7).as_matrix()
    moved = poses.copy()
    moved[:, :3, 3] = poses[:, :3, 3] @ R.T + [10, -5, 0]        # 整体刚体搬移
    assert ate(moved, poses)['ate_rmse_m'] < 1e-6                 # 对齐后应≈0


def test_kitti_rpe_zero_for_identical():
    poses = _random_poses(300)
    m = kitti_rpe(poses, poses, lengths=(20, 40, 60), step=5)
    assert m['n_segments'] > 0
    assert m['trans_err_pct'] < 1e-6 and m['rot_err_deg_per_m'] < 1e-6


def test_kitti_rpe_detects_scale_error():
    poses = _random_poses(300)
    bad = poses.copy()
    bad[:, :3, 3] *= 1.1                                          # 10% 尺度漂移
    m = kitti_rpe(bad, poses, lengths=(20, 40, 60), step=5)
    assert m['trans_err_pct'] > 1.0


def test_path_length_straight_line():
    poses = np.repeat(np.eye(4)[None], 11, axis=0)
    poses[:, 0, 3] = np.arange(11) * 2.0                          # 每步 2m，共 20m
    assert abs(path_length(poses) - 20.0) < 1e-9
