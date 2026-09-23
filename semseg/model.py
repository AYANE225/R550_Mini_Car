"""从零的紧凑距离图分割网（SalsaNet 风格 U-Net，~1M 参数）。

输入 5 通道距离图 (B,5,64,W)，2D 编码器-解码器带跳连，输出逐像素 19 类 logits。
下采样在高/宽两个方向(H=64 只降 3 次)，解码器上采样并 concat 跳连特征。纯 PyTorch。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inc, outc, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(inc, outc, k, s, k // 2, bias=False),
        nn.BatchNorm2d(outc), nn.LeakyReLU(0.1, inplace=True))


class Block(nn.Module):
    """两层卷积残差块。"""
    def __init__(self, inc, outc):
        super().__init__()
        self.c1 = conv_bn(inc, outc)
        self.c2 = conv_bn(outc, outc)
        self.skip = nn.Conv2d(inc, outc, 1, bias=False) if inc != outc else nn.Identity()

    def forward(self, x):
        return self.c2(self.c1(x)) + self.skip(x)


class RangeUNet(nn.Module):
    def __init__(self, in_ch=5, n_cls=19, base=32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.stem = conv_bn(in_ch, c1)
        self.e1 = Block(c1, c1)
        self.e2 = Block(c1, c2)
        self.e3 = Block(c2, c3)
        self.bott = Block(c3, c3)
        self.d3 = Block(c3 + c3, c2)
        self.d2 = Block(c2 + c2, c1)
        self.d1 = Block(c1 + c1, c1)
        self.head = nn.Conv2d(c1, n_cls, 1)
        self.drop = nn.Dropout2d(0.1)

    def forward(self, x):
        s0 = self.stem(x)                                  # 64×W
        s1 = self.e1(s0)
        p1 = F.max_pool2d(s1, 2)                            # 32×W/2
        s2 = self.e2(p1)
        p2 = F.max_pool2d(s2, 2)                            # 16×W/4
        s3 = self.e3(p2)
        p3 = F.max_pool2d(s3, 2)                            # 8×W/8
        b = self.drop(self.bott(p3))
        u3 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = self.d3(torch.cat([u3, s3], 1))
        u2 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.d2(torch.cat([u2, s2], 1))
        u1 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.d1(torch.cat([u1, s1], 1))
        return self.head(u1)
