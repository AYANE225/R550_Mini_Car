"""PointPillars ONNX 部署测试：导出后 ONNXRuntime 与 PyTorch 数值对齐（合成数据；无 onnx/torch 则跳过）。"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')

from det3d import deploy as D
from det3d.pointpillars import PointPillars
from det3d.voxelize import NX, NY


def _rand_inputs(P=500, seed=0):
    rng = np.random.default_rng(seed)
    pillars = rng.standard_normal((P, 32, 9)).astype(np.float32)
    npt = rng.integers(1, 33, P)
    mask = (np.arange(32)[None] < npt[:, None]).astype(np.float32)
    coords = np.stack([rng.integers(0, NY, P), rng.integers(0, NX, P)], 1).astype(np.int64)
    return pillars, mask, coords


def test_onnx_matches_torch(tmp_path):
    torch.manual_seed(0)
    net = PointPillars().eval()
    onnx_path = tmp_path / 'pp.onnx'
    D.export_onnx(net, onnx_path)
    assert onnx_path.exists()

    inp = _rand_inputs()
    ref = D.TorchRunner(net, 'cpu')(*inp)
    got = D.OrtRunner(onnx_path, ['CPUExecutionProvider'])(*inp)
    for k in ('cls', 'box', 'dir'):
        assert ref[k].shape == got[k].shape          # 输出结构一致
    assert D.max_abs_diff(ref, got) < 1e-3           # 导出无损


def test_dynamic_pillar_count(tmp_path):
    torch.manual_seed(1)
    net = PointPillars().eval()
    onnx_path = tmp_path / 'pp.onnx'
    D.export_onnx(net, onnx_path)
    runner = D.OrtRunner(onnx_path, ['CPUExecutionProvider'])
    o1 = runner(*_rand_inputs(P=300, seed=2))         # 动态柱数：不同 P 都能跑
    o2 = runner(*_rand_inputs(P=900, seed=3))
    assert o1['box'].shape == o2['box'].shape         # 锚框数固定，与 P 无关
