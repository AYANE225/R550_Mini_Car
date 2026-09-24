"""语义分割数据/模型单测（合成数据，CPU）。"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')     # CI(无 torch)自动跳过,本地照常跑

from semseg import data as D
from semseg.model import RangeUNet


def test_learning_map():
    raw = np.array([10, 40, 0, 252, 70, 81])       # car,road,unlabeled,moving-car,veg,traf-sign
    tr = D.raw_to_train(raw)
    assert tr[0] == 0 and tr[1] == 8               # car→0, road→8
    assert tr[2] == 255                            # unlabeled→ignore
    assert tr[3] == 0                              # moving-car→car
    assert tr[4] == 14 and tr[5] == 18


def test_projection_shapes_and_backproject():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 20, (2000, 4)).astype(np.float32)
    lab = rng.integers(0, 19, 2000).astype(np.int64)
    img, mask, px, py, labimg = D.project(pts, lab, W=512)
    assert img.shape == (5, 64, 512) and mask.dtype == bool
    assert px.max() < 512 and py.max() < 64 and px.min() >= 0
    # 每点所投像素的标签图取值应落在有效标签集合内
    back = labimg[py, px]
    assert np.all(np.isin(back[back != 255], np.unique(lab)))
    norm = D.normalize(img, mask)
    assert np.all(norm[:, ~mask] == 0)             # 空像素归零


def test_model_forward():
    net = RangeUNet()
    y = net(torch.randn(2, 5, 64, 128))
    assert y.shape == (2, 19, 64, 128)
    assert sum(p.numel() for p in net.parameters()) < 2e6
