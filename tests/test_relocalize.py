"""无初值全局重定位单测：SC 外观检索 + 粗位姿合成（合成数据，不跑 ICP）。"""
import numpy as np

from kitti_slam import scan_context as sc
from kitti_slam.relocalize import GlobalRelocalizer, _rz


def test_rz_rotation():
    R = _rz(np.pi / 2)
    assert np.allclose(R @ R.T, np.eye(4), atol=1e-9)
    assert np.allclose((R[:3, :3] @ np.array([1, 0, 0])), [0, 1, 0], atol=1e-6)


def _cloud(rng, n=3000):
    return rng.uniform(-40, 40, (n, 3)).astype(np.float64)


def test_retrieve_finds_matching_keyframe():
    rng = np.random.default_rng(1)
    clouds = [_cloud(rng) for _ in range(5)]
    scs = [sc.scan_context(c) for c in clouds]
    keys = [sc.ring_key(s) for s in scs]
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    for i in range(5):
        poses[i, :3, 3] = [i * 10, 0, 0]
    rel = GlobalRelocalizer(_cloud(rng, 5000), scs, keys, poses)
    # 用第 2 个建库点云再查一次：应检索到索引 2，SC 距离近 0
    j, d, yaw = rel.retrieve(clouds[2])
    assert j == 2 and d < 0.05
    T = rel.coarse_pose(j, yaw)
    assert np.allclose(T, poses[2] @ _rz(-yaw))
