"""SemanticKITTI torch Dataset：投影 → 归一化 → (5,H,W) 图 + (H,W) 标签。

训练增广：方位角循环平移(LiDAR 360°环形，沿宽度 roll 天然合法) + 水平翻转。
val 不增广。frames 按 stride 抽稀以便小网快训。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import data as D


class SemKitti(Dataset):
    def __init__(self, seqs, W=1024, stride=1, augment=False):
        self.W, self.augment = W, augment
        self.items = []
        for s in seqs:
            ids = D.frame_ids(s)[::stride]
            self.items += [(s, i) for i in ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        s, i = self.items[k]
        pts = D.read_scan(s, i)
        lab = D.raw_to_train(D.read_label(s, i))
        img, mask, _, _, labimg = D.project(pts, lab, W=self.W)
        img = D.normalize(img, mask)
        if self.augment:
            if np.random.rand() < 0.5:                       # 方位循环平移
                sh = np.random.randint(self.W)
                img = np.roll(img, sh, axis=2); labimg = np.roll(labimg, sh, axis=1)
            if np.random.rand() < 0.5:                       # 水平翻转
                img = img[:, :, ::-1].copy(); labimg = labimg[:, ::-1].copy()
                img[2] = -img[2]                             # y 取反
        return torch.from_numpy(img), torch.from_numpy(labimg)
