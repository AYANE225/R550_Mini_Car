"""语义分割推理封装：载权重 + 单帧点云 → 逐点类别(把距离图预测投回每个 3D 点)。"""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from .model import RangeUNet


def load_model(path, dev='cuda'):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = RangeUNet().to(dev).eval()
    net.load_state_dict(ck['model'])
    return net, ck.get('width', 1024)


@torch.no_grad()
def predict_points(net, scan, W=1024, dev='cuda'):
    """一帧点云 (N,≥4) → 逐点预测类别 (N,) ∈ 0..18。"""
    img, mask, px, py, _ = D.project(scan, None, W=W)
    x = torch.from_numpy(D.normalize(img, mask))[None].to(dev)
    with torch.cuda.amp.autocast():
        pred = net(x).argmax(1)[0].cpu().numpy()          # (H,W)
    return pred[py, px]
