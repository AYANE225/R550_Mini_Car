"""KITTI-360 读取的纯逻辑测试（不依赖 KITTI-360 数据/GPU）。

只测数据集无关的两块自研逻辑：3x4→4x4 齐次矩阵解析、稀疏真值帧的最长稠密段。
"""
import numpy as np

from kitti_slam.kitti360_io import _mat4, longest_dense_window


def test_mat4_from_3x4_appends_homogeneous_row():
    vals = [0.043, -0.088, 0.995, 0.804,
            -0.999, 0.0078, 0.0439, 0.299,
            -0.0116, -0.996, -0.0879, -0.177]
    T = _mat4(vals)
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(T[:3, :4], np.array(vals).reshape(3, 4))


def test_mat4_roundtrips_16_values():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [5, -2, 1]
    assert np.allclose(_mat4(T.ravel()), T)


def test_longest_dense_window_picks_longest_step1_run():
    # 稀疏帧：段 [1..3]、[10..14]、[100..102]；最长应为 10 起、长 5
    frames = [1, 2, 3, 10, 11, 12, 13, 14, 100, 101, 102]
    s, L = longest_dense_window(frames)
    assert (s, L) == (10, 5)


def test_longest_dense_window_single_frame():
    assert longest_dense_window([7]) == (7, 1)
