"""多序列验证：在多条 KITTI 序列上跑完整 SLAM，汇总 ATE(优化前/后) 表 + 轨迹拼图。

用法：python scripts/run_multi.py [--seqs 0 2 5 6 7 9]
里程计位姿按序列缓存到 reports/odom_poses_seqSS_*.npy，复跑秒级。
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


def run_seq(seq, voxel=0.5):
    n = io.num_frames(seq)
    load = lambda i: io.read_velodyne(seq, i)
    cache = ROOT / 'reports' / f'odom_poses_seq{seq:02d}_{n}_{voxel}.npy'
    if cache.exists():
        poses = np.load(cache)
        print(f'seq{seq:02d}: cached odometry ({len(poses)})')
    else:
        print(f'seq{seq:02d}: odometry {n} frames ...')
        t0 = time.perf_counter()
        poses = LidarOdometry(voxel=voxel).run(load, n, progress=max(1, n // 5))
        print(f'  done {time.perf_counter() - t0:.0f}s')
        np.save(cache, poses)
    loops = loopmod.find_loops(load, poses, radius=20.0, min_gap=150, stride=5)
    opt = pg.optimize(pg.build(poses, loops))
    gt = io.gt_poses_velodyne(seq)[:n]
    m0, m1 = ate(poses, gt), ate(opt, gt)
    np.savez(ROOT / 'reports' / f'slam_seq{seq:02d}.npz', odom=poses, opt=opt,
             loop_ij=np.array([[i, j] for (i, j, _T, _f) in loops], dtype=int), n=n, seq=seq)
    return {'seq': seq, 'n': n, 'len': path_length(poses), 'loops': len(loops),
            'ate0': m0['ate_rmse_m'], 'ate1': m1['ate_rmse_m'],
            'max0': m0['ate_max_m'], 'max1': m1['ate_max_m'], 'm1': m1}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seqs', type=int, nargs='+', default=[0, 2, 5, 6, 7, 9])
    args = ap.parse_args(argv)

    rows = [run_seq(s) for s in args.seqs]

    hdr = f"{'seq':>4} {'frames':>7} {'length':>8} {'loops':>6} {'ATE_odom':>9} {'ATE_opt':>8} {'improve':>8}"
    print('\n' + hdr); print('-' * len(hdr))
    lines = ['| seq | frames | length(m) | loops | ATE odom(m) | ATE opt(m) | improve |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        imp = 100 * (r['ate0'] - r['ate1']) / r['ate0']
        print(f"{r['seq']:>4d} {r['n']:>7d} {r['len']:>8.0f} {r['loops']:>6d} "
              f"{r['ate0']:>9.2f} {r['ate1']:>8.2f} {imp:>7.0f}%")
        lines.append(f"| {r['seq']:02d} | {r['n']} | {r['len']:.0f} | {r['loops']} | "
                     f"{r['ate0']:.2f} | {r['ate1']:.2f} | {imp:.0f}% |")
    mean0 = np.mean([r['ate0'] for r in rows]); mean1 = np.mean([r['ate1'] for r in rows])
    print('-' * len(hdr))
    print(f"{'MEAN':>4} {'':>7} {'':>8} {'':>6} {mean0:>9.2f} {mean1:>8.2f} "
          f"{100*(mean0-mean1)/mean0:>7.0f}%")
    lines.append(f"| **mean** | | | | **{mean0:.2f}** | **{mean1:.2f}** | "
                 f"**{100*(mean0-mean1)/mean0:.0f}%** |")
    (ROOT / 'reports' / 'multi_seq_table.md').write_text('\n'.join(lines) + '\n')
    print('  wrote reports/multi_seq_table.md')

    # 拼图：每序列 优化后轨迹 vs 真值（相机系俯视 x–z）
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps
    ncol = 3; nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow))
    fig.patch.set_facecolor(ps.BG)
    for ax, r in zip(np.atleast_1d(axes).ravel(), rows):
        g, a = r['m1']['gt_xyz'], r['m1']['aligned']
        ax.plot(g[:, 0], g[:, 2], '-', color=ps.GT, lw=2, label='GT')
        ax.plot(a[:, 0], a[:, 2], '--', color=ps.EST, lw=1.2, label='SLAM')
        ax.plot(g[0, 0], g[0, 2], 'o', color=ps.START, ms=8)
        ax.set_aspect('equal'); ps.style_ax(ax)
        ax.set_title(f"seq{r['seq']:02d}: ATE {r['ate1']:.2f}m ({r['loops']} loops)")
    for ax in np.atleast_1d(axes).ravel()[len(rows):]:
        ax.axis('off')
    ps.style_legend(axes.ravel()[0].legend())
    fig.suptitle('KITTI multi-sequence LiDAR SLAM (optimized trajectory vs ground truth)', color=ps.FG)
    fig.tight_layout(); ps.savefig(fig, ROOT / 'reports' / 'multi_seq.png', dpi=110)
    print('  wrote reports/multi_seq.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
