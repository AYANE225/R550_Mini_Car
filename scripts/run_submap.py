"""子图 SLAM：子图级位姿图 + 稠密 submap-to-submap 回环，对长/大场景更稳更省内存。

对比单帧回环版(slam_seqSS.npz)看是否抓回更多回环、降 ATE。用缓存的里程计位姿。
用法：python scripts/run_submap.py --seq 2 [--size 40]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam import submap as sm
from kitti_slam.metrics import ate, path_length


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=2)
    ap.add_argument('--size', type=int, default=40)
    ap.add_argument('--voxel', type=float, default=0.5)
    ap.add_argument('--radius', type=float, default=25.0)
    args = ap.parse_args(argv)

    n = io.num_frames(args.seq)
    load = lambda i: io.read_velodyne(args.seq, i)
    cache = ROOT / 'reports' / f'odom_poses_seq{args.seq:02d}_{n}_{args.voxel}.npy'
    odom = np.load(cache)
    print(f'seq{args.seq:02d}: {len(odom)} poses, building submaps (size {args.size}) ...')

    t0 = time.perf_counter()
    subs = sm.build_submaps(odom, load, size=args.size, voxel=args.voxel)
    print(f'  {len(subs)} submaps in {time.perf_counter()-t0:.0f}s; submap-to-submap loops ...')
    loops = sm.submap_loops(subs, radius=args.radius)
    print(f'  accepted {len(loops)} submap loops')
    opt_origins = sm.optimize_submaps(subs, loops)
    opt = sm.expand_to_frames(subs, opt_origins, len(odom))

    gt = io.gt_poses_velodyne(args.seq)[:len(odom)]
    m0, m1 = ate(odom, gt), ate(opt, gt)
    # 单帧回环版对照
    single = None
    sp = ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz'
    if sp.exists():
        single = ate(np.load(sp)['opt'], gt)['ate_rmse_m']

    print(f'\n=== seq{args.seq:02d} [{len(odom)}f, {path_length(odom):.0f}m] submap SLAM ===')
    print(f'  ATE odometry        : {m0["ate_rmse_m"]:.2f} m')
    if single is not None:
        print(f'  ATE single-scan loop: {single:.2f} m')
    print(f'  ATE submap SLAM     : {m1["ate_rmse_m"]:.2f} m  ({len(loops)} submap loops)')

    np.savez(ROOT / 'reports' / f'submap_seq{args.seq:02d}.npz', opt=opt,
             opt_origins=opt_origins, n=len(odom))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    g = m1['gt_xyz']; a = m1['aligned']; a0 = m0['aligned']
    ax.plot(g[:, 0], g[:, 2], '-', color='k', lw=2, label='ground truth')
    ax.plot(a0[:, 0], a0[:, 2], ':', color='tab:gray', lw=1, label=f'odometry ({m0["ate_rmse_m"]:.1f}m)')
    ax.plot(a[:, 0], a[:, 2], '--', color='tab:red', lw=1.3,
            label=f'submap SLAM ({m1["ate_rmse_m"]:.1f}m)')
    ax.plot(g[0, 0], g[0, 2], 'go', ms=9)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    ax.set_title(f'KITTI seq{args.seq:02d}: submap SLAM ({len(subs)} submaps, {len(loops)} loops)')
    fig.tight_layout(); fig.savefig(ROOT / 'reports' / f'submap_seq{args.seq:02d}.png', dpi=120)
    print('  saved', f'reports/submap_seq{args.seq:02d}.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
