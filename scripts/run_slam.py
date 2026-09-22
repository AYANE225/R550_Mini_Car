"""SLAM 里程碑 M2：里程计 → 回环检测 → 位姿图优化，报优化前后 ATE 并画轨迹。

用法：python scripts/run_slam.py --seq 0 [--frames -1 全序列] [--out reports/slam_seq00.png]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam import loop as loopmod
from kitti_slam import posegraph as pg
from kitti_slam.metrics import ate, path_length
from kitti_slam.odometry import LidarOdometry


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--frames', type=int, default=-1, help='-1=全序列')
    ap.add_argument('--voxel', type=float, default=0.5)
    ap.add_argument('--radius', type=float, default=20.0)
    ap.add_argument('--min-gap', type=int, default=150)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    total = io.num_frames(args.seq)
    n = total if args.frames < 0 else min(args.frames, total)
    load = lambda i: io.read_velodyne(args.seq, i)
    gt = io.gt_poses_velodyne(args.seq)[:n]

    cache = ROOT / 'reports' / f'odom_poses_seq{args.seq:02d}_{n}_{args.voxel}.npy'
    if cache.exists():
        poses = np.load(cache)
        print(f'seq{args.seq:02d}: loaded cached odometry ({len(poses)} poses) from {cache.name}')
    else:
        print(f'seq{args.seq:02d}: odometry over {n}/{total} frames ...')
        t0 = time.perf_counter()
        odom = LidarOdometry(voxel=args.voxel)
        poses = odom.run(load, n, progress=max(1, n // 10))
        print(f'  odometry done in {time.perf_counter() - t0:.0f}s')
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, poses)

    # 约定自检：仅里程计边优化应≈不变
    p_check = pg.optimize(pg.build(poses, []))
    d = np.linalg.norm(p_check[:, :3, 3] - poses[:, :3, 3], axis=1).max()
    print(f'  [sanity] no-loop optimize max move = {d:.3f} m (应≈0)')

    print('detecting loops ...')
    loops = loopmod.find_loops(load, poses, radius=args.radius, min_gap=args.min_gap, stride=5)
    print(f'  accepted {len(loops)} loop constraints')

    opt = pg.optimize(pg.build(poses, loops))

    npz = ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz'
    np.savez(npz, odom=poses, opt=opt,
             loop_ij=np.array([[i, j] for (i, j, _T, _f) in loops], dtype=int),
             n=n, seq=args.seq)
    print(f'  saved poses -> {npz.name}')

    m0, m1 = ate(poses, gt), ate(opt, gt)
    print(f'\n=== seq{args.seq:02d} [{n}f] path {path_length(poses):.0f}m, {len(loops)} loops ===')
    print(f'  ATE  odometry -> optimized : {m0["ate_rmse_m"]:.3f} m  ->  {m1["ate_rmse_m"]:.3f} m')
    print(f'  max  odometry -> optimized : {m0["ate_max_m"]:.3f} m  ->  {m1["ate_max_m"]:.3f} m')

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        # KITTI 相机世界系：水平面是 x–z（y 为竖直方向），俯视图取 (x, z)。
        for k, (mm, title) in enumerate(((m0, f'odometry  ATE={m0["ate_rmse_m"]:.2f}m'),
                                         (m1, f'+loop closure  ATE={m1["ate_rmse_m"]:.2f}m'))):
            g, a = mm['gt_xyz'], mm['aligned']
            ax[k].plot(g[:, 0], g[:, 2], '-', color='k', lw=2, label='ground truth')
            ax[k].plot(a[:, 0], a[:, 2], '--', color='tab:red', lw=1.3, label='estimate')
            ax[k].plot(g[0, 0], g[0, 2], 'go', ms=9)
            ax[k].set_aspect('equal'); ax[k].grid(alpha=0.3); ax[k].legend()
            ax[k].set_title(title); ax[k].set_xlabel('x [m]'); ax[k].set_ylabel('z [m]')
        fig.suptitle(f'KITTI seq{args.seq:02d} [{n} frames, {len(loops)} loops]')
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.out, dpi=120)
        print('  saved', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
