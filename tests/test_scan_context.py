"""Scan Context 描述子单测（合成点云，不依赖 KITTI/GPU）。"""
import numpy as np

from kitti_slam.scan_context import ring_key, scan_context, sc_distance

N_RING, N_SECTOR = 20, 60


def _rot_z(pts, ang):
    c, s = np.cos(ang), np.sin(ang)
    return pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def _scene(seed=0, n=24000):
    """随机几面带高度的"墙"分布在不同方位/距离，构成有辨识度的场景（随 seed 而异）。"""
    rng = np.random.default_rng(seed)
    nw = 5
    walls = list(zip(rng.uniform(-np.pi, np.pi, nw),      # 方位
                     rng.uniform(10, 55, nw),             # 距离
                     rng.uniform(3, 14, nw)))             # 高度
    pts = []
    for ang, rad, h in walls:
        m = n // nw
        a = ang + rng.normal(0, 0.04, m)
        r = rad + rng.normal(0, 1.0, m)
        z = rng.uniform(-1.7, h, m)
        pts.append(np.stack([r * np.cos(a), r * np.sin(a), z], 1))
    return np.concatenate(pts).astype(np.float32)


def test_shape_and_nonneg():
    sc = scan_context(_scene(), N_RING, N_SECTOR)
    assert sc.shape == (N_RING, N_SECTOR)
    assert (sc >= 0).all()


def test_ring_key_yaw_invariant():
    a = _scene(0)
    ka = ring_key(scan_context(a, N_RING, N_SECTOR))
    kb = ring_key(scan_context(_rot_z(a, 1.047), N_RING, N_SECTOR))   # 转 ~pi/3
    assert np.linalg.norm(ka - kb) < 0.08 * np.linalg.norm(ka)


def test_distance_self_and_rotated_small_other_large():
    a = scan_context(_scene(0), N_RING, N_SECTOR)
    b = scan_context(_rot_z(_scene(0), 2 * np.pi * 10 / N_SECTOR), N_RING, N_SECTOR)  # 转正好 10 扇区
    other = scan_context(_scene(7), N_RING, N_SECTOR)
    d_self, _ = sc_distance(a, a)
    d_rot, shift = sc_distance(a, b)
    d_other, _ = sc_distance(a, other)
    assert d_self < 1e-4
    assert d_rot < 0.10                       # 同一场景旋转后仍很近
    assert d_other > d_rot + 0.10             # 不同场景明显更远
    assert min(abs(shift - 10), abs(shift - (N_SECTOR - 10))) <= 1   # 恢复出旋转量
