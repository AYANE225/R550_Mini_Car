"""闭环导航：全局 A* 规一条路 → 用运动学模型 + DWA 局部规划/控制**真正把车开过去**。

复用 run_mapping 的 map_seqSS.npz（占据图）与 run_slam 的 slam_seqSS.npz（轨迹作走廊）。
可选注入若干全局图里没有的临时障碍（--obstacles），演示局部规划器反应式绕开。
输出：地图 + 全局路径 + 实际行驶轨迹(按速度着色) + 控制曲线，诚实报横向误差/余隙/是否到达。
用法：python scripts/run_local_nav.py --seq 0 [--obstacles 3]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import control as ctrl
from kitti_slam.planning import plan


def clear_tube(obst, res, x0, y0, traj, tube_r=1.6):
    """驶过的轨迹必然可行：在障碍图里沿轨迹清出半径 tube_r 的可行管道（去近场杂点）。"""
    cols = np.round((traj[:, 0] - x0) / res).astype(int)
    rows = np.round((traj[:, 1] - y0) / res).astype(int)
    ok = (rows >= 0) & (rows < obst.shape[0]) & (cols >= 0) & (cols < obst.shape[1])
    tube = np.zeros_like(obst)
    tube[rows[ok], cols[ok]] = True
    tube = binary_dilation(tube, iterations=max(1, int(round(tube_r / res))))
    obst = obst.copy(); obst[tube] = False
    return obst


def carve_obstacles(obst, res, x0, y0, path, k, radius=1.2, offset=1.5):
    """在 path 上等间隔取 k 个点，沿路径法向偏移 offset，在障碍图里填半径 radius 的圆。

    模拟全局图未知的临时障碍（挡住一侧车道）：offset>radius 使障碍紧贴路侧、逼车绕行，
    又给对侧留出可通行余量。只影响局部规划余隙，全局路径不变。
    返回 (obst_with_obstacles, centers[K,2], radius)。
    """
    obst = obst.copy()
    if k <= 0:
        return obst, np.zeros((0, 2)), radius
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    centers = []
    for f in np.linspace(0.28, 0.82, k):
        i = int(np.searchsorted(L, f * L[-1]))
        i = min(max(i, 1), len(path) - 2)
        tang = path[i + 1] - path[i - 1]
        nrm = np.array([-tang[1], tang[0]]); nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
        p = path[i] + nrm * offset
        centers.append(p)
        rr = int(round(radius / res))
        cc = int(round((p[0] - x0) / res)); rc = int(round((p[1] - y0) / res))
        r0, r1 = max(0, rc - rr), min(obst.shape[0], rc + rr + 1)
        c0, c1 = max(0, cc - rr), min(obst.shape[1], cc + rr + 1)
        ys, xs = np.ogrid[r0:r1, c0:c1]
        obst[r0:r1, c0:c1][(ys - rc) ** 2 + (xs - cc) ** 2 <= rr ** 2] = True
    return obst, np.array(centers), radius


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--obstacles', type=int, default=3, help='注入的临时障碍数(全局图未知)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    mp = np.load(ROOT / 'reports' / f'map_seq{a.seq:02d}.npz')
    sl = np.load(ROOT / 'reports' / f'slam_seq{a.seq:02d}.npz')
    occ, res = mp['occ'], float(mp['res'])
    x0, y0 = float(mp['x0']), float(mp['y0'])
    traj = sl['opt'][:, :2, 3]
    start = tuple(traj[0])
    goal = tuple(traj[np.argmax(np.linalg.norm(traj - traj[0], axis=1))])

    path, (free, res2, x0c, y0c) = plan(occ, res, x0, y0, traj, start, goal)
    if path is None:
        print('  no global path'); return 1
    path = np.array(path)
    glen = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    print(f'seq{a.seq:02d}: global A* route {glen:.0f} m ({len(path)} wp)')

    obst = clear_tube(occ.astype(bool), res, x0, y0, traj, tube_r=3.5)
    obst = clear_tube(obst, res, x0, y0, path, tube_r=3.5)
    obst, obs, orad = carve_obstacles(obst, res, x0, y0, path, a.obstacles)
    clg = ctrl.clearance_grid(~obst, res)
    t0 = time.perf_counter()
    r = ctrl.run_closed_loop(path, clg, res, x0, y0)
    m = r['metrics']
    print(f'  closed-loop drive: {"REACHED" if m["reached"] else "STUCK"}  '
          f'{m["driven_m"]:.0f} m in {m["sim_time_s"]:.0f}s sim ({time.perf_counter()-t0:.1f}s wall)')
    print(f'  cross-track err mean {m["xte_mean"]:.2f} m / max {m["xte_max"]:.2f} m  ·  '
          f'min clearance {m["clr_min"]:.2f} m  ·  mean speed {m["v_mean"]:.1f} m/s')
    if len(obs):
        print(f'  avoided {len(obs)} injected obstacles (unknown to global planner)')

    render(a.seq, occ, res, x0, y0, traj, path, glen, r, obs, orad,
           a.out or str(ROOT / 'reports' / f'local_nav_seq{a.seq:02d}.png'))
    return 0


def _clearest_avoidance(driven_xy, path, obs, win=90):
    """选“绕障动作最明显”的障碍下标：驶过轨迹在其最近点邻域内偏离全局路径最大的那个。"""
    best_j, best_sw = 0, -1.0
    for j, c in enumerate(obs):
        i = int(np.argmin(np.linalg.norm(driven_xy - c, axis=1)))
        w = driven_xy[max(0, i - win):i + win]
        if len(w) == 0:
            continue
        sw = max(np.min(np.linalg.norm(path - p, axis=1)) for p in w)
        if sw > best_sw:
            best_sw, best_j = sw, j
    return best_j


def render(seq, occ, res, x0, y0, traj, path, glen, r, obs, orad, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from kitti_slam import plotstyle as ps

    st, cmds, m = r['states'], r['cmds'], r['metrics']
    fig = plt.figure(figsize=(17, 9)); fig.patch.set_facecolor(ps.BG)
    gs = GridSpec(2, 2, width_ratios=[2.1, 1.0], figure=fig)

    ax = fig.add_subplot(gs[:, 0]); ax.set_facecolor(ps.BG)
    ext = [x0, x0 + occ.shape[1] * res, y0, y0 + occ.shape[0] * res]
    ax.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.85)
    ax.plot(path[:, 0], path[:, 1], '--', color=ps.GT, lw=1.8, label=f'global A* route {glen:.0f}m')
    spd = np.r_[cmds[:, 0], cmds[-1, 0]] if len(cmds) else st[:, 0]
    sc = ax.scatter(st[:, 0], st[:, 1], c=spd, cmap='turbo', s=6, label='driven (color=speed)')
    if len(obs):
        for p in obs:
            ax.add_patch(plt.Circle(p, orad, color='#ff3b52', alpha=0.9, zorder=5))
        ax.scatter(obs[:, 0], obs[:, 1], s=1, color='#ff3b52', label='injected obstacles (unknown to A*)')
    ax.plot(*path[0], 'o', color=ps.START, ms=13, label='start')
    ax.plot(*path[-1], '*', color=ps.EST, ms=20, mec='w', mew=0.6, label='goal')
    pad = 25
    ax.set_xlim(st[:, 0].min() - pad, st[:, 0].max() + pad)
    ax.set_ylim(st[:, 1].min() - pad, st[:, 1].max() + pad)
    ax.set_aspect('equal'); ps.style_legend(ax.legend(loc='best')); ps.style_ax(ax)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f"seq{seq:02d} closed-loop nav — {'REACHED' if m['reached'] else 'STUCK'}  ·  "
                 f"cross-track {m['xte_mean']:.2f} m (max {m['xte_max']:.2f})  ·  "
                 f"min clearance {m['clr_min']:.2f} m")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label='speed [m/s]')

    if len(obs):                                         # 放大展示一次真实尺度的反应式绕障
        j = _clearest_avoidance(st[:, :2], path, obs)
        c = obs[j]
        axin = ax.inset_axes([0.03, 0.03, 0.34, 0.34])
        axin.set_facecolor(ps.BG)
        axin.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.85)
        axin.plot(path[:, 0], path[:, 1], '--', color=ps.GT, lw=1.6)
        axin.scatter(st[:, 0], st[:, 1], c=spd, cmap='turbo', s=14,
                     norm=sc.norm, zorder=4)
        axin.add_patch(plt.Circle(c, orad, color='#ff3b52', alpha=0.9, zorder=5))
        axin.set_xlim(c[0] - 13, c[0] + 13); axin.set_ylim(c[1] - 13, c[1] + 13)
        axin.set_aspect('equal'); axin.set_xticks([]); axin.set_yticks([])
        axin.set_title('reactive avoidance (true scale)', color=ps.FG, fontsize=9)
        for s in axin.spines.values():
            s.set_edgecolor(ps.FG); s.set_alpha(0.5)
        ax.indicate_inset_zoom(axin, edgecolor=ps.FG, alpha=0.5)

    t = np.arange(len(cmds)) * 0.1
    a1 = fig.add_subplot(gs[0, 1]); a1.set_facecolor(ps.BG)
    a1.plot(t, cmds[:, 0], color=ps.EST, lw=1.6, label='speed [m/s]')
    a1b = a1.twinx()
    a1b.plot(t, np.rad2deg(cmds[:, 1]), color=ps.GT, lw=1.0, alpha=0.8, label='steer [deg]')
    a1.set_title('control: speed & steering', color=ps.FG); ps.style_ax(a1)
    a1.set_xlabel('sim time [s]'); a1.set_ylabel('speed [m/s]')
    a1b.set_ylabel('steer [deg]', color=ps.GT); a1b.tick_params(colors=ps.GT)

    a2 = fig.add_subplot(gs[1, 1]); a2.set_facecolor(ps.BG)
    a2.plot(np.arange(len(r['xte'])) * 0.1, r['xte'], color=ps.EST, lw=1.4, label='cross-track err [m]')
    a2.plot(np.arange(len(r['clr'])) * 0.1, r['clr'], color=ps.GT, lw=1.2, label='clearance [m]')
    a2.axhline(1.0, color='#ff3b52', lw=0.8, ls=':', alpha=0.7, label='safety radius')
    a2.set_title('tracking error & obstacle clearance', color=ps.FG)
    a2.set_xlabel('sim time [s]'); a2.set_ylabel('[m]'); ps.style_legend(a2.legend(loc='best')); ps.style_ax(a2)

    fig.suptitle(f'KITTI seq{seq:02d}: global plan + local DWA planning & control on SLAM-built map',
                 color=ps.FG, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); ps.savefig(fig, out, dpi=120)
    print('  saved', out)


if __name__ == '__main__':
    raise SystemExit(main())

