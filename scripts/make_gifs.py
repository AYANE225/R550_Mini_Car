"""把 README 里几张**本身就是时序过程**的静态图做成动画 GIF（从已缓存的结果直接复用，不重算）。

覆盖：
  slam      —— 里程计 vs 回环+位姿图：真值/估计双联逐帧铺开，回环边到点即连（漂移被拽回）
  localize  —— 先验图 scan-to-map ICP 定位：参考轨迹/定位轨迹逐帧铺开 + 误差曲线同步扫过
  kitti360  —— 2.4 km 大场景：真值/估计逐帧铺开（漂移主导，如实呈现）
这些都是"只增长、无移动 artist"的铺开式动画 → 用 disposal=1 让 GIF 帧间只编码增量，体积很小。
生动的"小车实时开"动画见 run_local_nav.py --gif；A* 路径铺开见 run_nav.py --gif。

用法：python scripts/make_gifs.py --which slam localize   # 或 --which all
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import plotstyle as ps


def _reveal_idx(n, target=90):
    """把 n 帧下采样到约 target 个动画关键帧（末帧必到）。"""
    ks = list(range(0, n, max(1, n // target)))
    if ks[-1] != n - 1:
        ks.append(n - 1)
    return ks


def gif_slam(seq=0, fps=14, dpi=90):
    """里程计 vs 回环+位姿图：双联(x–z 俯视)逐帧铺开真值(青)与对齐估计(橙)，回环边到点即连。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from kitti_slam import kitti_io as io
    from kitti_slam.metrics import ate

    sl = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    odom, opt, loop_ij = sl['odom'], sl['opt'], sl['loop_ij']
    gt = io.gt_poses_velodyne(seq)[:len(opt)]
    m0, m1 = ate(odom, gt), ate(opt, gt)
    n = len(gt)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7)); fig.patch.set_facecolor(ps.BG)
    arts = []
    for k, (mm, ttl) in enumerate(((m0, f'odometry  ATE={m0["ate_rmse_m"]:.2f} m'),
                                    (m1, f'+loop closure  ATE={m1["ate_rmse_m"]:.2f} m'))):
        g, a = mm['gt_xyz'], mm['aligned']
        A = ax[k]
        A.plot(g[:, 0], g[:, 2], '-', color=ps.GT, lw=2, alpha=0.25)     # 完整真值淡底（提示全貌）
        gt_ln, = A.plot([], [], '-', color=ps.GT, lw=2, label='ground truth')
        es_ln, = A.plot([], [], '--', color=ps.EST, lw=1.3, label='estimate')
        hd, = A.plot([g[0, 0]], [g[0, 2]], 'o', color=ps.EST, ms=5)
        A.plot(g[0, 0], g[0, 2], 'o', color=ps.START, ms=9)
        pad = 20
        A.set_xlim(g[:, 0].min() - pad, g[:, 0].max() + pad)
        A.set_ylim(g[:, 2].min() - pad, g[:, 2].max() + pad)
        A.set_aspect('equal'); ps.style_ax(A); ps.style_legend(A.legend(loc='best'))
        A.set_title(ttl); A.set_xlabel('x [m]'); A.set_ylabel('z [m]')
        arts.append((g, a, gt_ln, es_ln, hd))
    loops = [(int(i), int(j)) for i, j in loop_ij] if len(loop_ij) else []
    loop_pts = ax[1].scatter([], [], s=36, color=ps.ACCENT, zorder=6, marker='*')
    ttl = fig.suptitle('', color=ps.FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ks = _reveal_idx(n)

    def update(kf):
        for g, a, gt_ln, es_ln, hd in arts:
            gt_ln.set_data(g[:kf + 1, 0], g[:kf + 1, 2])
            es_ln.set_data(a[:kf + 1, 0], a[:kf + 1, 2])
            hd.set_data([a[kf, 0]], [a[kf, 2]])
        g1 = arts[1][0]
        done = [(i, j) for (i, j) in loops if j <= kf]
        if done:
            pts = np.array([[g1[j, 0], g1[j, 2]] for _, j in done])
            loop_pts.set_offsets(pts)
        ttl.set_text(f'KITTI seq{seq:02d}: odometry → loop closure + pose-graph  ·  '
                     f'frame {kf}/{n - 1}  ·  {len(done)}/{len(loops)} loops closed')
        return ()

    anim = FuncAnimation(fig, update, frames=ks, interval=1000 / fps)
    out = str(ROOT / 'reports' / f'slam_seq{seq:02d}.gif')
    ps.save_gif(anim, out, fps=fps, dpi=dpi, disposal=1)
    plt.close(fig)
    print('  saved', out, f'({Path(out).stat().st_size/1e6:.1f} MB)')


def gif_localize(seq=0, stride=1, crop=60.0, fps=14, dpi=88):
    """scan-to-map ICP 定位：占据图上参考轨迹(青)与定位轨迹(橙)逐帧铺开 + 误差曲线同步扫过。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    mp = np.load(ROOT / 'reports' / f'map_seq{seq:02d}.npz')
    sl = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    cache = ROOT / 'reports' / f'loc_est_seq{seq:02d}_{stride}_{crop}.npz'
    c = np.load(cache)
    est = c['est']
    odom, opt = sl['odom'], sl['opt']
    idx = list(range(0, len(odom), stride))
    est_xy, opt_xy, odom_xy = est[:, :2, 3], opt[idx, :2, 3], odom[idx, :2, 3]
    e_loc = np.linalg.norm(est_xy - opt_xy, axis=1)
    e_odom = np.linalg.norm(odom_xy - opt_xy, axis=1)
    rmse = float(np.sqrt((e_loc ** 2).mean()))
    occ = mp['occ']; res = float(mp['res']); x0 = float(mp['x0']); y0 = float(mp['y0'])
    ext = [x0, x0 + occ.shape[1] * res, y0, y0 + occ.shape[0] * res]
    ai = np.array(idx); n = len(idx)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7)); fig.patch.set_facecolor(ps.BG)
    a0 = ax[0]
    a0.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.9)
    ref_ln, = a0.plot([], [], '-', color=ps.GT, lw=2, label='SLAM reference')
    loc_ln, = a0.plot([], [], '--', color=ps.EST, lw=1.3, label='ICP localization')
    hd, = a0.plot([est_xy[0, 0]], [est_xy[0, 1]], 'o', color=ps.EST, ms=5)
    a0.plot(est_xy[0, 0], est_xy[0, 1], 'o', color=ps.START, ms=9)
    pad = 20
    a0.set_xlim(opt_xy[:, 0].min() - pad, opt_xy[:, 0].max() + pad)
    a0.set_ylim(opt_xy[:, 1].min() - pad, opt_xy[:, 1].max() + pad)
    a0.set_aspect('equal'); ps.style_legend(a0.legend(loc='best')); ps.style_ax(a0)
    a0.set_xlabel('x [m]'); a0.set_ylabel('y [m]'); a0.set_title('scan-to-map localization')

    a1 = ax[1]
    a1.plot(ai, e_odom, color=ps.MUTED, lw=1, alpha=0.25)                 # 全程淡底
    eo_ln, = a1.plot([], [], color=ps.MUTED, lw=1, label='raw odometry (drifts)')
    el_ln, = a1.plot([], [], color=ps.EST, lw=1.6, label='scan-to-map ICP')
    a1.set_xlim(ai[0], ai[-1]); a1.set_ylim(0, float(max(e_odom.max(), e_loc.max())) * 1.05)
    a1.set_xlabel('frame'); a1.set_ylabel('error vs SLAM-opt [m]')
    ps.style_ax(a1); ps.style_legend(a1.legend(loc='upper left'))
    a1.set_title(f'localization RMSE {rmse:.2f} m (bounded)')
    ttl = fig.suptitle('', color=ps.FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    ks = _reveal_idx(n)

    def update(kf):
        ref_ln.set_data(opt_xy[:kf + 1, 0], opt_xy[:kf + 1, 1])
        loc_ln.set_data(est_xy[:kf + 1, 0], est_xy[:kf + 1, 1])
        hd.set_data([est_xy[kf, 0]], [est_xy[kf, 1]])
        eo_ln.set_data(ai[:kf + 1], e_odom[:kf + 1])
        el_ln.set_data(ai[:kf + 1], e_loc[:kf + 1])
        ttl.set_text(f'KITTI seq{seq:02d}: prior-map localization (lidar + odom init, no GT)  ·  '
                     f'frame {ai[kf]}/{ai[-1]}  ·  ICP err {e_loc[kf]:.2f} m vs odom {e_odom[kf]:.1f} m')
        return ()

    anim = FuncAnimation(fig, update, frames=ks, interval=1000 / fps)
    out = str(ROOT / 'reports' / f'localize_seq{seq:02d}.gif')
    ps.save_gif(anim, out, fps=fps, dpi=dpi, disposal=1)
    plt.close(fig)
    print('  saved', out, f'({Path(out).stat().st_size/1e6:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--which', nargs='+', default=['all'],
                    choices=['all', 'slam', 'localize'])
    ap.add_argument('--seq', type=int, default=0)
    a = ap.parse_args(argv)
    which = ['slam', 'localize'] if 'all' in a.which else a.which
    if 'slam' in which:
        print('slam trajectory reveal ...'); gif_slam(a.seq)
    if 'localize' in which:
        print('localization reveal ...'); gif_localize(a.seq)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
