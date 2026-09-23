"""M4 导航：在建好的 2D 地图上做全局路径规划，给定起终点规划沿街最短路并渲染。

依赖 run_mapping 的 map_seqSS.npz 与 run_slam 的 slam_seqSS.npz（取轨迹作可行驶走廊）。
默认起点=轨迹起点，终点=离起点最远的轨迹点（考验在路网上另择近路）。
用法：python scripts/run_nav.py --seq 0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam.planning import plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--gif', action='store_true', help='额外渲染全局路径逐点铺开的动画 GIF')
    args = ap.parse_args(argv)

    mp = np.load(ROOT / 'reports' / f'map_seq{args.seq:02d}.npz')
    sl = np.load(ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz')
    occ, res = mp['occ'], float(mp['res'])
    x0, y0 = float(mp['x0']), float(mp['y0'])
    traj = sl['opt'][:, :2, 3]

    start = tuple(traj[0])
    goal = tuple(traj[np.argmax(np.linalg.norm(traj - traj[0], axis=1))])   # 离起点最远处
    print(f'seq{args.seq:02d}: plan {tuple(round(v,1) for v in start)} -> {tuple(round(v,1) for v in goal)}')
    t0 = time.perf_counter()
    path, (free, res2, x0c, y0c) = plan(occ, res, x0, y0, traj, start, goal)
    if path is None:
        print('  no path found'); return 1
    path = np.array(path)
    length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    driven = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    print(f'  planned route {length:.0f} m ({len(path)} wp) in {time.perf_counter()-t0:.1f}s')
    print(f'  vs driven trajectory length {driven:.0f} m  (规划在路网上另择路径)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps
    ext = [x0, x0 + occ.shape[1] * res, y0, y0 + occ.shape[0] * res]
    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor(ps.BG)
    ax.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.85)
    ax.plot(traj[:, 0], traj[:, 1], '-', color=ps.GT, lw=1, alpha=0.55, label='driven trajectory')
    ax.plot(path[:, 0], path[:, 1], '-', color=ps.EST, lw=2.5, label=f'planned route {length:.0f}m')
    ax.plot(*start, 'o', color=ps.START, ms=13, label='start')
    ax.plot(*goal, '*', color=ps.EST, ms=20, mec='w', mew=0.6, label='goal')
    ax.set_aspect('equal'); ps.style_legend(ax.legend(loc='upper left')); ps.style_ax(ax)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'KITTI seq{args.seq:02d}: global path planning on SLAM-built map')
    fig.tight_layout(); ps.savefig(fig, ROOT / 'reports' / f'nav_seq{args.seq:02d}.png', dpi=120)
    print('  saved', f'reports/nav_seq{args.seq:02d}.png')
    if args.gif:
        render_gif(args.seq, occ, res, x0, y0, traj, path, length, start, goal,
                   str(ROOT / 'reports' / f'nav_seq{args.seq:02d}.gif'))
    return 0


def render_gif(seq, occ, res, x0, y0, traj, path, length, start, goal, out,
               fps=14, dpi=80, target_frames=72):
    """把全局 A* 路径**逐点铺开**：占据图与驶过轨迹作底，规划路线从起点向终点一段段延伸、
    端点游标推进——直观展示 A* 在路网上另择的一条最短几何路。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from kitti_slam import plotstyle as ps

    ext = [x0, x0 + occ.shape[1] * res, y0, y0 + occ.shape[0] * res]
    pad = 25
    x_lo, x_hi = path[:, 0].min() - pad, path[:, 0].max() + pad
    y_lo, y_hi = path[:, 1].min() - pad, path[:, 1].max() + pad
    fw = 12.0; fh = max(4.0, min(9.0, fw * (y_hi - y_lo) / (x_hi - x_lo) + 0.8))
    fig, ax = plt.subplots(figsize=(fw, fh)); fig.patch.set_facecolor(ps.BG)
    ax.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.85)
    ax.plot(traj[:, 0], traj[:, 1], '-', color=ps.GT, lw=1, alpha=0.5, label='driven trajectory')
    ax.plot(*start, 'o', color=ps.START, ms=13, label='start')
    ax.plot(*goal, '*', color=ps.EST, ms=20, mec='w', mew=0.6, label='goal')
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal'); ps.style_ax(ax)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    route, = ax.plot([], [], '-', color=ps.EST, lw=2.5, label=f'planned route {length:.0f}m')
    head, = ax.plot([path[0, 0]], [path[0, 1]], 'o', color=ps.EST, ms=6, zorder=6)
    ps.style_legend(ax.legend(loc='upper left'))
    title = ax.set_title('')
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])

    ks = list(range(0, len(path), max(1, len(path) // target_frames)))
    if ks[-1] != len(path) - 1:
        ks.append(len(path) - 1)

    def update(k):
        route.set_data(path[:k + 1, 0], path[:k + 1, 1])
        head.set_data([path[k, 0]], [path[k, 1]])
        title.set_text(f'KITTI seq{seq:02d}: A* global planning  ·  '
                       f'{cum[k]:.0f} / {length:.0f} m')
        return route, head, title

    anim = FuncAnimation(fig, update, frames=ks, interval=1000 / fps)
    fig.tight_layout()
    ps.save_gif(anim, out, fps=fps, dpi=dpi, disposal=1)
    plt.close(fig)
    print('  saved', out, f'({Path(out).stat().st_size/1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
