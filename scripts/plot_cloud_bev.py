"""俯视图（BEV）激光点云图：把 SLAM 建出的 3D 点云地图按高度着色、俯视渲染。

用法：python scripts/plot_cloud_bev.py --seq 0 [--stride 1]
依赖 run_mapping 的 reports/map_seqSS.npz（含 map_pts, Nx3, velodyne 系 z 朝上）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1, help='点抽稀步长(大图提速)')
    ap.add_argument('--size', type=float, default=0.15, help='散点大小')
    args = ap.parse_args(argv)

    d = np.load(ROOT / 'reports' / f'map_seq{args.seq:02d}.npz')
    pts = d['map_pts'][::args.stride]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # 高度着色范围裁到分位数，避免极端离群点吃掉色带
    zlo, zhi = np.percentile(z, [2, 98])
    # SLAM 行驶轨迹（同一世界系，取 opt 位姿），叠在点云上
    traj = None
    sp = ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz'
    if sp.exists():
        traj = np.load(sp)['opt'][:, :2, 3]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    span_x, span_y = x.max() - x.min(), y.max() - y.min()
    fig, ax = plt.subplots(figsize=(12, 12 * span_y / span_x))
    ax.set_facecolor('#0b0b12')
    sc = ax.scatter(x, y, c=np.clip(z, zlo, zhi), s=args.size, cmap='turbo',
                    linewidths=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01); cb.set_label('height z [m]')
    if traj is not None:
        ax.plot(traj[:, 0], traj[:, 1], '-', color='white', lw=1.6, alpha=0.9,
                label='SLAM trajectory')
        ax.plot(traj[0, 0], traj[0, 1], 'o', color='lime', ms=11, mec='k', label='start')
        ax.legend(loc='upper right', framealpha=0.3, labelcolor='white')
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'KITTI seq{args.seq:02d} — LiDAR point-cloud map (BEV, N={len(d["map_pts"]):,})')
    out = ROOT / 'reports' / f'cloud_bev_seq{args.seq:02d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor='#0b0b12')
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
