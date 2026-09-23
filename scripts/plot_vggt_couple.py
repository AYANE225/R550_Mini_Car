"""把"里程计中断→VGGT 接回轨迹"做成一张 GitHub 展示用的对比图。

读 run_vggt_couple.py 存下的 reports/vggt_couple_seq{seq}.npz(不重跑 VGGT), 画一张聚焦
"LiDAR 里程计丢失一段"场景的单幅对比:LiDAR-only 断裂错位 vs LiDAR+VGGT 接回真值。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import plotstyle as ps
from kitti_slam.metrics import ate


def _xz(P, gt):
    d = ate(P, gt)
    return d['aligned'][:, [0, 2]], d['gt_xyz'][:, [0, 2]], d['ate_rmse_m']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--out', type=str, default='reports/vggt_dropout_seq{seq:02d}.png')
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = np.load(ROOT / 'reports' / f'vggt_couple_seq{a.seq:02d}.npz')
    gt = d['gt']
    g0, g1 = (int(x) for x in d['gap'])
    lidar, gxz, ate_l = _xz(d['LiDAR-only_(dropout)'], gt)
    fused, _, ate_f = _xz(d['LiDAR+VGGT_(dropout)'], gt)

    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    fig.patch.set_facecolor(ps.BG)

    ax.plot(gxz[:, 0], gxz[:, 1], color=ps.GT, lw=3.0, label='Ground truth', zorder=3)
    ax.plot(lidar[:, 0], lidar[:, 1], color=ps.MUTED, lw=2.2, ls='--',
            label=f'LiDAR-only  ·  ATE {ate_l:.2f} m', zorder=4)
    ax.plot(fused[:, 0], fused[:, 1], color=ps.EST, lw=2.4,
            label=f'LiDAR + VGGT  ·  ATE {ate_f:.2f} m', zorder=5)

    # 中断段高亮 + 标注
    ax.plot(gxz[g0:g1, 0], gxz[g0:g1, 1], color=ps.ACCENT, lw=6, alpha=0.55,
            solid_capstyle='round', zorder=2, label=f'LiDAR odometry lost ({g1 - g0} frames)')
    mid = (g0 + g1) // 2
    ax.annotate('LiDAR odometry\ndrops out here',
                xy=(gxz[mid, 0], gxz[mid, 1]), xytext=(gxz[mid, 0] - 42, gxz[mid, 1] - 6),
                color=ps.ACCENT, fontsize=11, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=ps.ACCENT, lw=1.6))
    ax.scatter(gxz[0, 0], gxz[0, 1], s=90, color=ps.START, edgecolor='w', zorder=6, label='start')

    ax.text(0.97, 0.16,
            'LiDAR alone loses tracking through the gap\nand the whole tail drifts off.',
            transform=ax.transAxes, color=ps.MUTED, fontsize=10.5, va='center', ha='right')
    ax.text(0.97, 0.06,
            'VGGT visual relative-pose factors bridge the gap\nand pull the trajectory back onto ground truth.',
            transform=ax.transAxes, color=ps.EST, fontsize=10.5, va='center', ha='right',
            fontweight='bold')

    ax.set_title('Sensor redundancy: when LiDAR odometry drops out, VGGT holds the trajectory\n'
                 f'seq{a.seq:02d}  ·  VGGT relative-pose factors fused into the pose graph  ·  '
                 f'ATE {ate_l:.1f} m → {ate_f:.2f} m',
                 color=ps.FG, fontsize=12.5)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_aspect('equal', 'datalim')
    ps.style_ax(ax)
    ps.style_legend(ax.legend(loc='upper left', fontsize=9.5))
    ps.savefig(fig, ROOT / a.out.format(seq=a.seq))
    print(f'  saved {a.out.format(seq=a.seq)}  (LiDAR {ate_l:.2f} -> fused {ate_f:.2f} m)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
