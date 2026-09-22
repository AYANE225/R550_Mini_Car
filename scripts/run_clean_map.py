"""感知↔SLAM 联动：用跟踪出的动态物体，从建图里剔除动态点，得到干净静态地图。

依赖 run_track 的 reports/dynamic_seqSS.npz（动态物体逐帧世界位置）+ slam_seqSS.npz(位姿)。
逐帧把扫描转世界系，落在任一动态物体附近(半径 R)的点标为动态剔除，其余累积成静态地图。
用法：python scripts/run_clean_map.py --seq 0 --start 0 --frames 800
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.registration import preprocess


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=800)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--radius', type=float, default=3.5)
    args = ap.parse_args(argv)

    poses = np.load(ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz')['opt']
    dyn = np.load(ROOT / 'reports' / f'dynamic_seq{args.seq:02d}.npz')['dyn_pts']
    per_frame = {}
    for t, x, y in dyn:                             # t = frame*0.1
        per_frame.setdefault(int(round(t / 0.1)), []).append((x, y))

    end = min(args.start + args.frames, len(poses))
    static, removed = [], []
    t0 = time.perf_counter()
    for i in range(args.start, end, args.stride):
        pc = preprocess(io.read_velodyne(args.seq, i), voxel=0.3)
        pts = np.asarray(pc.points)
        P = poses[i]
        world = (pts @ P[:3, :3].T) + P[:3, 3]
        dyn_here = per_frame.get(i, [])
        if dyn_here:
            dp = np.array(dyn_here)
            d = np.min(np.linalg.norm(world[:, None, :2] - dp[None, :, :], axis=2), axis=1)
            mask = d < args.radius
            removed.append(world[mask]); static.append(world[~mask])
        else:
            static.append(world)
    static = np.vstack(static)
    removed = np.vstack(removed) if removed else np.empty((0, 3))
    print(f'seq{args.seq:02d} frames {args.start}-{end}: static {len(static)} pts, '
          f'removed dynamic {len(removed)} pts  {time.perf_counter()-t0:.0f}s')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = static[::3]
    fig, ax = plt.subplots(figsize=(13, 9)); ax.set_facecolor('#0b0b12')
    ax.scatter(s[:, 0], s[:, 1], s=0.3, c='#7fb', linewidths=0, rasterized=True,
               label=f'static map ({len(static):,} pts)')
    if len(removed):
        ax.scatter(removed[:, 0], removed[:, 1], s=1.2, c='#ff4d4d', linewidths=0,
                   rasterized=True, label=f'removed dynamic ({len(removed):,} pts)')
    ax.set_aspect('equal'); ax.legend(loc='upper right', labelcolor='w', framealpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'KITTI seq{args.seq:02d}: dynamic-aware mapping '
                 f'(perception removes moving objects)', color='w')
    ax.tick_params(colors='w'); [sp.set_color('w') for sp in ax.spines.values()]
    ax.xaxis.label.set_color('w'); ax.yaxis.label.set_color('w')
    out = ROOT / 'reports' / f'clean_map_seq{args.seq:02d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor='#0b0b12')
    print('  saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
