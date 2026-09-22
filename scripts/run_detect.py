"""经典 LiDAR 物体检测演示：对某帧检测 + BEV 可视化（点云 + 有向框）。

用法：python scripts/run_detect.py --seq 0 --frame 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.detection import detect


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--frame', type=int, default=0)
    args = ap.parse_args(argv)

    pts = io.read_velodyne(args.seq, args.frame)
    boxes, obj = detect(pts)
    print(f'seq{args.seq:02d} frame {args.frame}: {len(boxes)} object candidates '
          f'from {len(obj)} non-ground points')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    raw = pts[:, :3]
    r = np.linalg.norm(raw[:, :2], axis=1)
    raw = raw[(r > 3) & (r < 45)]
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor('#0b0b12')
    ax.scatter(raw[:, 0], raw[:, 1], s=0.4, c='#556', linewidths=0, rasterized=True)
    ax.scatter(obj[:, 0], obj[:, 1], s=0.6, c='#8fd', linewidths=0, rasterized=True)
    for b in boxes:
        ax.add_patch(Polygon(b.corners_bev(), closed=True, fill=False,
                             ec='#ff4d4d', lw=1.6))
        d = np.array([np.cos(b.yaw), np.sin(b.yaw)]) * b.l / 2   # 朝向短须
        ax.plot([b.cx, b.cx + d[0]], [b.cy, b.cy + d[1]], '-', color='#ffd24d', lw=1.2)
    ax.plot(0, 0, 'w^', ms=10)                                    # 自车(雷达原点)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'KITTI seq{args.seq:02d} frame {args.frame}: classical LiDAR detection '
                 f'({len(boxes)} objects)', color='w')
    ax.tick_params(colors='w'); [s.set_color('w') for s in ax.spines.values()]
    ax.xaxis.label.set_color('w'); ax.yaxis.label.set_color('w')
    out = ROOT / 'reports' / f'detect_seq{args.seq:02d}_{args.frame:06d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor='#0b0b12')
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
