"""KITTI PointPillars 数据集 + collate（pillars/coords/targets 打包成 batch）。"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import kitti_det as kd
from .anchors import assign
from .voxelize import points_to_pillars


class KittiPillars(Dataset):
    def __init__(self, split_name, anchors, max_pts=32, max_pillars=12000, augment=False):
        self.frames = kd.load_split(split_name)
        self.anchors = anchors
        self.max_pts, self.max_pillars, self.augment = max_pts, max_pillars, augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        idx = self.frames[i]
        pts = kd.read_velodyne('training', idx)
        calib = kd.read_calib('training', idx)
        gt = kd.read_labels('training', idx, calib)
        if self.augment and len(gt):
            if np.random.rand() < 0.5:                      # 随机沿 x 轴镜像(y 翻转)
                pts = pts.copy(); pts[:, 1] *= -1
                gt = gt.copy(); gt[:, 1] *= -1; gt[:, 6] = -gt[:, 6]
        pil, coords, npt = points_to_pillars(pts, self.max_pts, self.max_pillars)
        labels, reg_t, dir_t, _ = assign(self.anchors, gt)
        return {'pillars': pil, 'coords': coords, 'npoints': npt,
                'labels': labels, 'reg_t': reg_t, 'dir_t': dir_t}


def collate(batch):
    pillars = np.concatenate([b['pillars'] for b in batch], 0)
    npoints = np.concatenate([b['npoints'] for b in batch], 0)
    coords = np.concatenate([np.c_[np.full(len(b['coords']), bi, np.int32), b['coords']]
                             for bi, b in enumerate(batch)], 0)
    labels = np.stack([b['labels'] for b in batch])
    reg_t = np.stack([b['reg_t'] for b in batch])
    dir_t = np.stack([b['dir_t'] for b in batch])
    maxpts = pillars.shape[1]
    mask = np.arange(maxpts)[None, :] < npoints[:, None]
    t = torch.from_numpy
    return {'pillars': t(pillars), 'mask': t(mask), 'coords': t(coords),
            'labels': t(labels), 'reg_t': t(reg_t), 'dir_t': t(dir_t),
            'batch_size': len(batch)}
