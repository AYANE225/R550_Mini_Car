"""校验坐标变换：画某训练帧的 BEV 点云 + GT 车辆框（LiDAR 系）。框应贴合车点簇。

用法：python scripts/det3d_vis_gt.py --idx 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from det3d import kitti_det as kd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--idx', type=int, default=7)
    ap.add_argument('--split', default='training')
    args = ap.parse_args(argv)

    pts = kd.read_velodyne(args.split, args.idx)
    calib = kd.read_calib(args.split, args.idx)
    boxes = kd.read_labels(args.split, args.idx, calib)
    print(f'frame {args.idx:06d}: {len(pts)} points, {len(boxes)} Car boxes')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    m = (pts[:, 0] > 0) & (pts[:, 0] < 70) & (np.abs(pts[:, 1]) < 40)
    p = pts[m]
    fig, ax = plt.subplots(figsize=(12, 10)); ax.set_facecolor('#0b0b12')
    ax.scatter(p[:, 0], p[:, 1], s=0.4, c=p[:, 2], cmap='turbo', linewidths=0, rasterized=True)
    for b in boxes:
        ax.add_patch(Polygon(kd.box_corners_bev(b), closed=True, fill=False, ec='#ff3b3b', lw=2))
        d = np.array([np.cos(b[6]), np.sin(b[6])]) * b[3] / 2       # 朝向须
        ax.plot([b[0], b[0] + d[0]], [b[1], b[1] + d[1]], '-', color='#ffd24d', lw=1.5)
    ax.plot(0, 0, 'w^', ms=12)
    ax.set_aspect('equal'); ax.set_xlabel('x (forward) [m]'); ax.set_ylabel('y (left) [m]')
    ax.set_title(f'KITTI 3D-Object frame {args.idx:06d}: LiDAR BEV + GT Car boxes '
                 f'({len(boxes)} cars)', color='w')
    ax.tick_params(colors='#888'); [s.set_color('w') for s in ax.spines.values()]
    out = ROOT / 'reports' / f'det3d_gt_{args.idx:06d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor='#0b0b12')
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
