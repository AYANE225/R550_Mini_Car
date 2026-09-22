"""C++ ICP 扩展测试：与纯 NumPy 参考实现对拍（未编译则跳过）。"""
import numpy as np
import pytest

from kitti_slam.icp_py import icp_point_to_plane as py_icp

icp_cpp = pytest.importorskip("kitti_slam.icp_cpp",
                              reason="C++ 扩展未编译（bash native/build.sh）")


def _plane_scene(seed=0):
    """造一个带结构的目标点云(两面墙+地面)+已知位姿扰动的源，用于点面 ICP。"""
    rng = np.random.default_rng(seed)
    ground = np.column_stack([rng.uniform(-10, 10, 3000), rng.uniform(-10, 10, 3000),
                              np.zeros(3000)])
    wall1 = np.column_stack([rng.uniform(-10, 10, 1500), np.full(1500, 8.0),
                             rng.uniform(0, 3, 1500)])
    wall2 = np.column_stack([np.full(1500, -8.0), rng.uniform(-10, 10, 1500),
                             rng.uniform(0, 3, 1500)])
    tgt = np.vstack([ground, wall1, wall2])
    normals = np.vstack([np.tile([0, 0, 1.], (3000, 1)),
                         np.tile([0, -1., 0], (1500, 1)),
                         np.tile([1., 0, 0], (1500, 1))])
    th = 0.05
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    src = (tgt - [0.3, -0.2, 0.0]) @ R                      # 源=目标施加小位姿偏移
    return src, tgt, normals


def test_cpp_matches_numpy_reference():
    src, tgt, n = _plane_scene()
    Tc, fc, rc = icp_cpp.icp_point_to_plane(src, tgt, n, np.eye(4), 1.0, 40)
    Tp, fp, rp = py_icp(src, tgt, n, np.eye(4), 1.0, 40)
    assert np.allclose(Tc, Tp, atol=1e-3)                   # C++ 与 NumPy 位姿一致
    assert abs(fc - fp) < 0.02
    assert rc < 0.05                                        # 收敛到低残差


def test_cpp_reduces_alignment_error():
    src, tgt, n = _plane_scene(seed=3)
    Tc, fc, rc = icp_cpp.icp_point_to_plane(src, tgt, n, np.eye(4), 1.0, 40)
    # 配准后源点更贴合目标平面：残差 rmse 应很小
    assert rc < 0.05 and fc > 0.8
