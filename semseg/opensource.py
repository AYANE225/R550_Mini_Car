"""集成主流开源 LiDAR 语义分割 WaffleIron(valeoai, ICCV'23)做同口径对照。

薄适配层(第二条腿：结合主流开源的能力)：把 third_party/WaffleIron 挂到 sys.path，
按官方 kitti 配置搭 Segmenter、载官方 SemanticKITTI 预训练权重(WaffleIron-48-256,
~6.1M 参数)，复用其点云预处理(体素化 0.1m / 三平面投影 / 16 邻域)对 seq08 逐帧出
「逐点 19 类」预测。类序与本项目自研 semseg 完全一致(官方 SemanticKITTI 顺序,无需 remap)，
预测按原始 .bin 点序对齐,可与自研 0.88M 距离图网做严格点级对照。

WaffleIron 是纯 PyTorch(MLP + 稠密 2D 卷积,无 spconv/torchsparse),因此可在
Blackwell sm_120(RTX 5090)上直接推理——这正是选它做集成对象的原因。

第三方代码与权重不入库(见 .gitignore)；本文件只做加载与推理编排,不改动其源码。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
WI_ROOT = ROOT / 'third_party' / 'WaffleIron'
WI_CONFIG = WI_ROOT / 'configs' / 'WaffleIron-48-256__kitti.yaml'
WI_CKPT = (ROOT / 'third_party' / 'weights' / 'pretrained_models'
           / 'WaffleIron-48-256__kitti' / 'ckpt_last.pth')
# WaffleIron 内部再拼 dataset/sequences/XX/velodyne/*.bin
KITTI_ROOT = '/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked'


def _ensure_waffleiron_on_path():
    """把 WaffleIron 根置于 sys.path[0]：本地 datasets/utils 包须盖过已装的同名包(如 HF datasets)。"""
    p = str(WI_ROOT)
    if sys.path[:1] != [p]:
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


class WaffleIronSeg:
    """官方 WaffleIron-48-256 KITTI 权重推理封装：seq08 帧号 → 逐点 0..18 预测(原始点序)。"""

    def __init__(self, ckpt=WI_CKPT, config=WI_CONFIG, rootdir=KITTI_ROOT,
                 phase='val', dev='cuda'):
        _ensure_waffleiron_on_path()
        from waffleiron import Segmenter          # noqa: E402  (需先改 sys.path)
        from datasets import SemanticKITTI, Collate

        with open(config) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.dev = dev
        self.nb_class = cfg['classif']['nb_class']

        net = Segmenter(
            input_channels=cfg['embedding']['size_input'],
            feat_channels=cfg['waffleiron']['nb_channels'],
            depth=cfg['waffleiron']['depth'],
            grid_shape=cfg['waffleiron']['grids_size'],
            nb_class=cfg['classif']['nb_class'],
            drop_path_prob=cfg['waffleiron']['drop'],
        )
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        sd = ck['net']
        try:
            net.load_state_dict(sd)
        except Exception:                          # DataParallel 存的权重带 module. 前缀
            net.load_state_dict({k[len('module.'):]: v for k, v in sd.items()})
        net.compress()                             # 官方推理前的通道压缩(见 eval_kitti.py)
        self.net = net.to(dev).eval()

        self.dataset = SemanticKITTI(
            rootdir=rootdir,
            input_feat=cfg['embedding']['input_feat'],
            voxel_size=cfg['embedding']['voxel_size'],
            num_neighbors=cfg['embedding']['neighbors'],
            dim_proj=cfg['waffleiron']['dim_proj'],
            grids_shape=cfg['waffleiron']['grids_size'],
            fov_xyz=cfg['waffleiron']['fov_xyz'],
            phase=phase,
            tta=False,
        )
        self.collate = Collate()
        # seq08 帧号(000000.bin→0) → 数据集内部索引
        self._frame2idx = {int(Path(f).stem): i
                           for i, f in enumerate(self.dataset.im_idx)}

    def has_frame(self, frame) -> bool:
        return int(frame) in self._frame2idx

    def frames(self):
        return sorted(self._frame2idx)

    @torch.inference_mode()
    def predict_frame(self, frame):
        """seq08 帧号 → 逐点预测 (N,) ∈ 0..18，按原始 .bin 点序对齐(与 D.read_scan 同序同长)。"""
        idx = self._frame2idx[int(frame)]
        batch = self.collate([self.dataset[idx]])
        feat = batch['feat'].to(self.dev, non_blocking=True)
        cell_ind = batch['cell_ind'].to(self.dev, non_blocking=True)
        occ = batch['occupied_cells'].to(self.dev, non_blocking=True)
        nb = batch['neighbors_emb'].to(self.dev, non_blocking=True)
        up = batch['upsample'][0].to(self.dev, non_blocking=True)
        with torch.autocast('cuda', enabled=True):
            out = self.net(feat, cell_ind, occ, nb)          # (1, nb_class, Nmax)
        logits = out[0, :, up].T                              # (N_orig, nb_class) 体素→原始点最近邻上采样
        return logits.argmax(1).cpu().numpy().astype(np.int64)
