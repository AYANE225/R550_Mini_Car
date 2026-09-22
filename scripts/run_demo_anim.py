"""整体 demo 动画：全景 + 跟车 双联，叠加检测框与多目标航迹(静/动)。

逐帧检测车辆→世界系卡尔曼 MOT→按净位移判静/动；左=固定全景(看边跑边建图)、右=跟车(±win,看
细节)。精算结果缓存到 pkl，改视觉时秒级复渲。
用法：python scripts/run_demo_anim.py --seq 0 --start 0 --frames 800 --stride 4
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.detection import detect
from kitti_slam.tracking import MOT


def precompute(seq, start, end, idx, poses):
    """逐帧检测+跟踪；只在动画帧(idx)存显示数据。返回 (scans,bg,boxes_w,egos,tracks)。"""
    show = set(idx)
    rng = np.random.default_rng(0)
    mot = MOT(gate=2.5, max_misses=5, min_hits=3)
    scans, bg, boxes_w, egos = {}, {}, {}, {}
    t0 = time.perf_counter()
    for i in range(start, end):
        pts = io.read_velodyne(seq, i)
        P = poses[i]
        bx, _ = detect(pts, vehicles_only=True, voxel=0.2, ransac_iters=80, min_points=8)
        ctr = [P[:3, :3] @ np.array([b.cx, b.cy, b.cz]) + P[:3, 3] for b in bx]
        mot.step(np.array([c[:2] for c in ctr]) if ctr else np.empty((0, 2)), t=i * 0.1, dt=0.1)
        if i in show:
            xyz = pts[:, :3]
            r = np.linalg.norm(xyz[:, :2], axis=1)
            world = (xyz[(r > 3) & (r < 45)] @ P[:3, :3].T) + P[:3, 3]
            scans[i] = world[rng.choice(len(world), min(3500, len(world)), replace=False)]
            bg[i] = world[rng.choice(len(world), min(600, len(world)), replace=False)]
            boxes_w[i] = [((np.c_[b.corners_bev(), np.full(4, b.cz)] @ P[:3, :3].T)
                           + P[:3, 3])[:, :2] for b in bx]
            egos[i] = P[:2, 3]
        if (i - start) % max(1, (end - start) // 8) == 0:
            print(f'  {i}/{end} ({len(bx)} veh, {len(mot.tracks)} tracks)', flush=True)
    tracks = {}
    for tr in mot.all_tracks(min_hits=3):
        h = np.array([(t, x, y) for t, x, y in tr.history])
        tracks[tr.id] = (bool(np.linalg.norm(h[-1, 1:] - h[0, 1:]) > 3.0
                              and h[-1, 0] - h[0, 0] > 0.5), h)
    print(f'  precompute {time.perf_counter()-t0:.0f}s; {len(tracks)} tracks '
          f'({sum(v[0] for v in tracks.values())} dynamic)')
    return ([scans[i] for i in idx], [bg[i] for i in idx],
            [boxes_w[i] for i in idx], [egos[i] for i in idx], tracks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=800)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--win', type=float, default=55.0)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--dpi', type=int, default=60)
    ap.add_argument('--render-step', type=int, default=2, help='渲染时再抽帧(缩小 GIF)')
    args = ap.parse_args(argv)

    poses = np.load(ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz')['opt']
    end = min(args.start + args.frames, len(poses))
    idx = list(range(args.start, end, args.stride))
    cache = ROOT / 'reports' / f'demo_cache_seq{args.seq:02d}_{args.start}_{end}_{args.stride}.pkl'
    if cache.exists():
        print(f'seq{args.seq:02d}: load cache {cache.name}')
        scans, bg, boxes_w, egos, tracks = pickle.load(open(cache, 'rb'))
    else:
        print(f'seq{args.seq:02d}: track every frame {args.start}-{end}, animate stride {args.stride}')
        scans, bg, boxes_w, egos, tracks = precompute(args.seq, args.start, end, idx, poses)
        pickle.dump((scans, bg, boxes_w, egos, tracks), open(cache, 'wb'))

    traj = poses[args.start:end, :2, 3]
    tf = [(i - args.start) for i in idx]
    gxy = np.vstack(bg)
    gxlim = (gxy[:, 0].min() - 10, gxy[:, 0].max() + 10)
    gylim = (gxy[:, 1].min() - 10, gxy[:, 1].max() + 10)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon
    fig, (axg, axc) = plt.subplots(1, 2, figsize=(16, 8))
    rk = list(range(0, len(idx), max(1, args.render_step)))     # 渲染抽帧
    print(f'  rendering GIF ({len(rk)} frames @ dpi {args.dpi}) ...')

    def style(ax, title):
        ax.set_facecolor('#0b0b12'); ax.set_aspect('equal')
        ax.set_title(title, color='w', fontsize=11)
        ax.tick_params(colors='w'); [sp.set_color('w') for sp in ax.spines.values()]

    def draw(k):
        tnow = idx[k] * 0.1
        acc = np.vstack(bg[:k + 1])
        ex, ey = egos[k]
        # 左：全景
        axg.clear(); style(axg, f'global map & trajectory  (f{idx[k]})')
        axg.scatter(acc[:, 0], acc[:, 1], s=0.4, c='#2c4', alpha=0.5, linewidths=0, rasterized=True)
        axg.plot(traj[:tf[k] + 1, 0], traj[:tf[k] + 1, 1], '-', color='#ddd', lw=1.2)
        for is_dyn, h in tracks.values():
            hh = h[h[:, 0] <= tnow + 1e-6]
            if len(hh) >= 2 and is_dyn:
                axg.plot(hh[:, 1], hh[:, 2], '-', color='#ff4d4d', lw=1.5)
        axg.plot(ex, ey, '^', color='w', ms=11)
        axg.set_xlim(*gxlim); axg.set_ylim(*gylim)
        # 右：跟车
        axc.clear(); style(axc, 'chase view — detection (yellow) + tracks (red=dyn, cyan=static)')
        axc.scatter(acc[:, 0], acc[:, 1], s=0.6, c='#2c4', alpha=0.45, linewidths=0, rasterized=True)
        s = scans[k]
        axc.scatter(s[:, 0], s[:, 1], s=1.1, c=s[:, 2], cmap='turbo', linewidths=0, rasterized=True)
        axc.plot(traj[:tf[k] + 1, 0], traj[:tf[k] + 1, 1], '-', color='#ddd', lw=1.3)
        for cw in boxes_w[k]:
            axc.add_patch(Polygon(cw, closed=True, fill=False, ec='#ffd24d', lw=1.6))
        for is_dyn, h in tracks.values():
            hh = h[h[:, 0] <= tnow + 1e-6]
            if len(hh) < 2:
                continue
            col = '#ff4d4d' if is_dyn else '#3fd0d0'
            axc.plot(hh[:, 1], hh[:, 2], '-', color=col, lw=2 if is_dyn else 1, alpha=0.9)
            axc.plot(hh[-1, 1], hh[-1, 2], 'o', color=col, ms=5)
        axc.plot(ex, ey, '^', color='w', ms=13)
        axc.set_xlim(ex - args.win, ex + args.win); axc.set_ylim(ey - args.win, ey + args.win)
        fig.suptitle(f'KITTI seq{args.seq:02d} — LiDAR SLAM + detection + tracking', color='w')

    anim = FuncAnimation(fig, lambda j: draw(rk[j]), frames=len(rk), interval=1000 / args.fps)
    out = ROOT / 'reports' / f'demo_seq{args.seq:02d}.gif'
    fig.patch.set_facecolor('#0b0b12')
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi,
              savefig_kwargs={'facecolor': '#0b0b12'})
    print('  saved', out, f'({out.stat().st_size/1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
