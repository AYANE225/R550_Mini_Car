"""用 VGGT（前馈视觉几何大模型）对 KITTI 相机图做稠密 3D 重建并渲染（仅可视化，不融合进 SLAM）。

只调**公开 VGGT 模型 + 官方权重**（PYTHONPATH 指向公开 VGGT-Long/base_models 的 vggt 包，权重
/media/4T/cst/model/VGGT.pt），**绝不导入任何非公开的 LiDAR-VGGT-SLAM 融合工程**。
输入 KITTI seq 的 image_2 一段窗口 → 每像素世界点 + RGB → 按置信度过滤 → BEV + 3D 渲染。
用法：VGGT_SRC=... python scripts/run_vggt.py --seq 0 --start 0 --count 20
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# 公开 VGGT 代码路径（非融合工程）；可用环境变量覆盖
VGGT_SRC = os.environ.get('VGGT_SRC', '/media/4T/cst/PROJECT/Compare_SLAM/VGGT-Long/base_models')
VGGT_WEIGHTS = os.environ.get('VGGT_WEIGHTS', '/media/4T/cst/model/VGGT.pt')
sys.path.insert(0, VGGT_SRC)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def kitti_image(seq, frame):
    return (Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked/dataset/sequences')
            / f'{int(seq):02d}' / 'image_2' / f'{int(frame):06d}.png')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=20, help='送入 VGGT 的连续帧数(窗口)')
    ap.add_argument('--step', type=int, default=3, help='帧间隔(拉开基线)')
    ap.add_argument('--conf-pct', type=float, default=50.0, help='置信度分位阈值(丢低于此的点)')
    args = ap.parse_args(argv)

    frames = list(range(args.start, args.start + args.count * args.step, args.step))
    paths = [str(kitti_image(args.seq, f)) for f in frames]
    print(f'VGGT on seq{args.seq:02d} frames {frames[0]}..{frames[-1]} ({len(paths)} imgs)')
    print(f'  code={VGGT_SRC}\n  weights={VGGT_WEIGHTS}')

    device = 'cuda'
    model = VGGT()
    sd = torch.load(VGGT_WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval().to(device).requires_grad_(False)

    images = load_and_preprocess_images(paths).to(device)      # [S,3,H,W] in [0,1]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        pred = model(images)
    wp = pred['world_points'][0].float().cpu().numpy()         # [S,H,W,3]
    conf = pred['world_points_conf'][0].float().cpu().numpy()  # [S,H,W]
    imgs = pred['images'][0].float().cpu().numpy().transpose(0, 2, 3, 1)  # [S,H,W,3]
    print(f'  world_points {wp.shape}, conf[min,med,max]='
          f'{conf.min():.1f}/{np.median(conf):.1f}/{conf.max():.1f}')

    P = wp.reshape(-1, 3)
    C = imgs.reshape(-1, 3).clip(0, 1)
    K = conf.reshape(-1)
    keep = K >= np.percentile(K, args.conf_pct)
    P, C = P[keep], C[keep]
    print(f'  kept {len(P):,} points (conf>=p{args.conf_pct:.0f})')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # VGGT 世界系(首帧相机)：x 右、y 下、z 前 → 俯视取 (x,z)
    fig = plt.figure(figsize=(18, 8)); fig.patch.set_facecolor('#0b0b12')
    axb = fig.add_subplot(1, 2, 1); axb.set_facecolor('#0b0b12')
    lim = np.percentile(np.abs(P), 98, axis=0)
    axb.scatter(P[:, 0], P[:, 2], c=C, s=0.4, linewidths=0, rasterized=True)
    axb.set_xlim(-lim[0], lim[0]); axb.set_ylim(0, lim[2])
    axb.set_aspect('equal'); axb.set_title('VGGT dense reconstruction — BEV (x–z)', color='w')
    axb.set_xlabel('x [rel]'); axb.set_ylabel('z [rel]')
    ax3 = fig.add_subplot(1, 2, 2, projection='3d'); ax3.set_facecolor('#0b0b12')
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.set_pane_color((0.04, 0.04, 0.07, 1.0)); pane.label.set_color('w')
        pane.set_tick_params(colors='#888')
    sub = np.random.default_rng(0).choice(len(P), min(150000, len(P)), replace=False)
    ax3.scatter(P[sub, 0], P[sub, 2], -P[sub, 1], c=C[sub], s=0.5, linewidths=0, depthshade=False)
    ax3.set_title('VGGT dense point cloud (3D)', color='w')
    ax3.set_xlabel('x'); ax3.set_ylabel('z (forward)'); ax3.set_zlabel('up')
    ax3.set_xlim(-lim[0], lim[0]); ax3.set_ylim(0, lim[2]); ax3.set_zlim(-lim[1], lim[1])
    ax3.set_box_aspect((lim[0] * 2, lim[2], lim[1] * 2))
    ax3.view_init(elev=22, azim=-70)
    for ax in (axb,):
        ax.tick_params(colors='w'); [s.set_color('w') for s in ax.spines.values()]
    fig.suptitle(f'KITTI seq{args.seq:02d} f{frames[0]}-{frames[-1]}: VGGT feed-forward '
                 f'monocular reconstruction (render-only)', color='w')
    out = ROOT / 'reports' / f'vggt_seq{args.seq:02d}_{frames[0]:06d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor='#0b0b12')
    print('  saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
