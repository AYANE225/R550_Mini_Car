"""PointPillars 检测模型（纯 PyTorch，无 spconv/mmcv）：PillarVFE → Scatter → 2D 骨干 → SSD 头。

- PillarVFE：每点 9 维 → Linear(9,64)+BN+ReLU → 柱内 max → 柱特征 [P,64]
- Scatter：按 (b,iy,ix) 摊回 [B,64,NY,NX] 伪图
- Backbone2D：SECOND 风格 3 下采样块 + 反卷积上采样拼接 → [B,384,NY/2,NX/2]
- SSDHead：每位置 2 个朝向锚(单类 Car)，出 cls / box(7) / dir 预测
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .voxelize import NX, NY


class PillarVFE(nn.Module):
    def __init__(self, in_ch=9, out_ch=64):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch, bias=False)
        self.bn = nn.BatchNorm1d(out_ch, eps=1e-3, momentum=0.01)

    def forward(self, pillars, mask):
        # pillars [P, maxpts, 9], mask [P, maxpts] (1=有效点)
        x = self.linear(pillars)                       # [P,maxpts,64]
        P, M, C = x.shape
        x = self.bn(x.view(-1, C)).view(P, M, C)
        x = torch.relu(x)
        x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        return x.max(dim=1)[0]                          # [P,64]（无效点被 -inf 屏蔽）


def scatter(feats, coords, batch_size):
    # feats [P,64], coords [P,3]=(b,iy,ix) -> [B,64,NY,NX]
    C = feats.shape[1]
    canvas = feats.new_zeros((batch_size, C, NY, NX))
    b, iy, ix = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
    canvas[b, :, iy, ix] = feats
    return canvas


def _block(inc, outc, n, stride):
    layers = [nn.Conv2d(inc, outc, 3, stride=stride, padding=1, bias=False),
              nn.BatchNorm2d(outc, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True)]
    for _ in range(n):
        layers += [nn.Conv2d(outc, outc, 3, padding=1, bias=False),
                   nn.BatchNorm2d(outc, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def _up(inc, outc, stride):
    return nn.Sequential(
        nn.ConvTranspose2d(inc, outc, stride, stride=stride, bias=False),
        nn.BatchNorm2d(outc, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True))


class Backbone2D(nn.Module):
    def __init__(self, inc=64):
        super().__init__()
        self.b1 = _block(inc, 64, 3, 2)          # -> NY/2
        self.b2 = _block(64, 128, 5, 2)          # -> NY/4
        self.b3 = _block(128, 256, 5, 2)         # -> NY/8
        self.u1 = _up(64, 128, 1)
        self.u2 = _up(128, 128, 2)
        self.u3 = _up(256, 128, 4)

    def forward(self, x):
        x1 = self.b1(x); x2 = self.b2(x1); x3 = self.b3(x2)
        return torch.cat([self.u1(x1), self.u2(x2), self.u3(x3)], dim=1)   # [B,384,NY/2,NX/2]


class SSDHead(nn.Module):
    def __init__(self, inc=384, n_anchors=2, n_cls=1, box_code=7):
        super().__init__()
        self.n_anchors, self.n_cls, self.box_code = n_anchors, n_cls, box_code
        self.cls = nn.Conv2d(inc, n_anchors * n_cls, 1)
        self.box = nn.Conv2d(inc, n_anchors * box_code, 1)
        self.dir = nn.Conv2d(inc, n_anchors * 2, 1)

    def forward(self, x):
        def perm(t, c):
            B, _, H, W = t.shape
            return t.view(B, self.n_anchors, c, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, c)
        return {'cls': perm(self.cls(x), self.n_cls),
                'box': perm(self.box(x), self.box_code),
                'dir': perm(self.dir(x), 2)}


class PointPillars(nn.Module):
    def __init__(self):
        super().__init__()
        self.vfe = PillarVFE()
        self.backbone = Backbone2D()
        self.head = SSDHead()

    def forward(self, pillars, mask, coords, batch_size):
        feats = self.vfe(pillars, mask)
        canvas = scatter(feats, coords, batch_size)
        bev = self.backbone(canvas)
        return self.head(bev), bev.shape[-2:]                 # preds, (Hf,Wf)
