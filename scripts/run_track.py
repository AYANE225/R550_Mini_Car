"""检测 + 多目标跟踪：逐帧检车 → 用 SLAM 位姿转世界系 → 卡尔曼 MOT → 判静/动 → BEV 画航迹。

用法：python scripts/run_track.py --seq 0 --start 0 --frames 800
产出 reports/track_seqSS.png 与 reports/dynamic_seqSS.npz（动态物体世界位置，供建图剔除）。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.detection import detect
from kitti_slam.tracking import MOT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=800)
    ap.add_argument('--dyn-disp', type=float, default=3.0, help='净位移>此值(米)判动态')
    args = ap.parse_args(argv)

    poses = np.load(ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz')['opt']
    end = min(args.start + args.frames, len(poses))
    mot = MOT()
    t0 = time.perf_counter()
    for i in range(args.start, end):
        boxes, _ = detect(io.read_velodyne(args.seq, i), vehicles_only=True)
        P = poses[i]
        world = []
        for b in boxes:
            p = P[:3, :3] @ np.array([b.cx, b.cy, b.cz]) + P[:3, 3]
            world.append(p[:2])
        mot.step(np.array(world) if world else np.empty((0, 2)), t=i * 0.1, dt=0.1)
        if (i - args.start) % max(1, (end - args.start) // 8) == 0:
            print(f'  track frame {i}/{end} ({len(mot.tracks)} active)', flush=True)
    wall = time.perf_counter() - t0

    tracks = mot.all_tracks(min_hits=4)
    dyn, sta = [], []
    for tr in tracks:
        h = np.array([(x, y) for _, x, y in tr.history])
        disp = float(np.linalg.norm(h[-1] - h[0]))
        dur = tr.history[-1][0] - tr.history[0][0]
        (dyn if (disp > args.dyn_disp and dur > 0.5) else sta).append((tr, h, disp))
    print(f'\n=== seq{args.seq:02d} frames {args.start}-{end}: {len(tracks)} tracks '
          f'({len(dyn)} dynamic, {len(sta)} static)  {wall:.0f}s ===')

    # 保存动态物体逐帧世界位置(供建图剔除动态点)
    dyn_pts = np.array([[t] + [x, y] for tr, h, _ in dyn for (t, x, y) in tr.history]) \
        if dyn else np.empty((0, 3))
    np.savez(ROOT / 'reports' / f'dynamic_seq{args.seq:02d}.npz', dyn_pts=dyn_pts)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traj = poses[args.start:end, :2, 3]
    fig, ax = plt.subplots(figsize=(12, 11)); ax.set_facecolor('#0b0b12')
    ax.plot(traj[:, 0], traj[:, 1], '-', color='#888', lw=1, label='ego (SLAM)')
    for tr, h, _ in sta:
        ax.plot(h[:, 0], h[:, 1], '.', color='#3fd0d0', ms=1.5)
    for tr, h, disp in dyn:
        ax.plot(h[:, 0], h[:, 1], '-', color='#ff4d4d', lw=1.8)
        ax.plot(h[0, 0], h[0, 1], 'o', color='#ffd24d', ms=4)
    ax.plot([], [], '.', color='#3fd0d0', label=f'static objects ({len(sta)})')
    ax.plot([], [], '-', color='#ff4d4d', label=f'dynamic tracks ({len(dyn)})')
    ax.set_aspect('equal'); ax.legend(loc='upper right', labelcolor='w', framealpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'KITTI seq{args.seq:02d}: LiDAR detection + tracking (world frame)', color='w')
    ax.tick_params(colors='w'); [s.set_color('w') for s in ax.spines.values()]
    ax.xaxis.label.set_color('w'); ax.yaxis.label.set_color('w')
    out = ROOT / 'reports' / f'track_seq{args.seq:02d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=140, facecolor='#0b0b12')
    print('  saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
