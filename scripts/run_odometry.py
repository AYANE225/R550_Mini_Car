"""里程计里程碑：在 KITTI 某序列前 N 帧跑 scan-to-map ICP，报 ATE 并画轨迹对比。

用法：python scripts/run_odometry.py --seq 0 --frames 300 [--out reports/odom_seq00.png]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.metrics import ate, path_length
from kitti_slam.odometry import LidarOdometry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--voxel', type=float, default=0.5)
    ap.add_argument('--out', default=None, help='轨迹对比 PNG 路径')
    args = ap.parse_args(argv)

    total = io.num_frames(args.seq)
    n = min(args.frames, total) if args.frames > 0 else total
    print(f'seq {args.seq:02d}: {total} frames total, running {n}')

    odom = LidarOdometry(voxel=args.voxel)
    t0 = time.perf_counter()
    poses = odom.run(lambda i: io.read_velodyne(args.seq, i), n, progress=max(1, n // 10))
    wall = time.perf_counter() - t0

    gt = io.gt_poses_velodyne(args.seq)[:n]
    m = ate(poses, gt)
    print(f'\n=== seq{args.seq:02d} first {n} frames ===')
    print(f'  path length   : {path_length(poses):.1f} m (gt {path_length(gt):.1f} m)')
    print(f'  ATE rmse      : {m["ate_rmse_m"]:.3f} m  (mean {m["ate_mean_m"]:.3f}, max {m["ate_max_m"]:.3f})')
    print(f'  wall time     : {wall:.1f}s  ({wall / n * 1000:.0f} ms/frame)')

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        g = m['gt_xyz']; a = m['aligned']       # KITTI 相机世界系水平面=x–z
        ax.plot(g[:, 0], g[:, 2], '-', color='k', lw=2, label='ground truth')
        ax.plot(a[:, 0], a[:, 2], '--', color='tab:red', lw=1.5, label='LiDAR odometry (aligned)')
        ax.plot(g[0, 0], g[0, 2], 'go', ms=9, label='start')
        ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
        ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
        ax.set_title(f'seq{args.seq:02d} [{n}f]  ATE={m["ate_rmse_m"]:.2f}m  len={path_length(poses):.0f}m')
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.out, dpi=120)
        print('  saved', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
