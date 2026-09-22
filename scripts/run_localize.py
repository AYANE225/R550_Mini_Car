"""M3b 定位：在已建 3D 地图上做 scan-to-map ICP 定位，只用激光+里程计增量，不用真值。

对固定地图配准→误差有界（对比原始里程计会漂）。依赖 run_mapping 的 map_seqSS.npz 与
run_slam 的 slam_seqSS.npz（里程计增量作 ICP 初值）。
用法：python scripts/run_localize.py --seq 0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.localize import MapLocalizer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--crop', type=float, default=60.0)
    args = ap.parse_args(argv)

    mp = np.load(ROOT / 'reports' / f'map_seq{args.seq:02d}.npz')
    sl = np.load(ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz')
    odom, opt = sl['odom'], sl['opt']
    n = len(odom)
    loc = MapLocalizer(mp['map_pts'].astype(np.float64), crop_radius=args.crop)

    idx = list(range(0, n, args.stride))
    est = np.repeat(np.eye(4)[None], len(idx), axis=0)
    est[0] = opt[0]                                   # 已知初始位姿（先验地图定位常规设定）
    fits = []
    t0 = time.perf_counter()
    for k in range(1, len(idx)):
        i, ip = idx[k], idx[k - 1]
        motion = np.linalg.inv(odom[ip]) @ odom[i]    # 里程计增量作初值
        init = est[k - 1] @ motion
        T, fit, rmse = loc.localize(io.read_velodyne(args.seq, i), init)
        est[k] = T
        fits.append(fit)
        if k % max(1, len(idx) // 10) == 0:
            print(f'  localize {i}/{n} fit={fit:.2f}', flush=True)
    wall = time.perf_counter() - t0

    est_xy = est[:, :2, 3]
    opt_xy = opt[idx, :2, 3]
    odom_xy = odom[idx, :2, 3]
    e_loc = np.linalg.norm(est_xy - opt_xy, axis=1)
    e_odom = np.linalg.norm(odom_xy - opt_xy, axis=1)
    print(f'\n=== seq{args.seq:02d} scan-to-map ICP localization ===')
    print(f'  localization vs SLAM-opt : RMSE {np.sqrt((e_loc**2).mean()):.3f} m, '
          f'mean {e_loc.mean():.3f}, max {e_loc.max():.3f}  (mean fit {np.mean(fits):.2f})')
    print(f'  raw odometry vs SLAM-opt : RMSE {np.sqrt((e_odom**2).mean()):.3f} m  (对照:会漂)')
    print(f'  wall {wall:.0f}s ({wall / len(idx) * 1000:.0f} ms/frame)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    occ = mp['occ']; res = float(mp['res']); x0 = float(mp['x0']); y0 = float(mp['y0'])
    ext = [x0, x0 + occ.shape[1] * res, y0, y0 + occ.shape[0] * res]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(occ, origin='lower', extent=ext, cmap='Greys', alpha=0.7)
    ax[0].plot(opt[:, 0, 3], opt[:, 1, 3], '-', color='k', lw=2, label='SLAM reference')
    ax[0].plot(est_xy[:, 0], est_xy[:, 1], '--', color='tab:red', lw=1.2, label='ICP localization')
    ax[0].plot(est_xy[0, 0], est_xy[0, 1], 'go', ms=9)
    ax[0].set_aspect('equal'); ax[0].legend(); ax[0].grid(alpha=0.2)
    ax[0].set_xlabel('x [m]'); ax[0].set_ylabel('y [m]'); ax[0].set_title('scan-to-map localization')
    ai = np.array(idx)
    ax[1].plot(ai, e_loc, color='tab:red', label='scan-to-map ICP')
    ax[1].plot(ai, e_odom, color='tab:gray', lw=1, label='raw odometry (drifts)')
    ax[1].set_xlabel('frame'); ax[1].set_ylabel('error vs SLAM-opt [m]')
    ax[1].grid(alpha=0.3); ax[1].legend()
    ax[1].set_title(f'localization RMSE {np.sqrt((e_loc**2).mean()):.2f} m (bounded)')
    fig.suptitle(f'KITTI seq{args.seq:02d}: prior-map localization (lidar + odom init, no GT)')
    fig.tight_layout(); fig.savefig(ROOT / 'reports' / f'localize_seq{args.seq:02d}.png', dpi=120)
    print('  saved', f'reports/localize_seq{args.seq:02d}.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
