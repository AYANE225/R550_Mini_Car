"""无初值全局重定位（kidnapped robot）：Scan Context 外观检索 → scan-to-map ICP 精化。

先验地图定位常规设定已知初值(run_localize.py)；这里**去掉初值**：在已建地图上随机撒一批查询
帧，每帧仅凭一帧激光找回 6DOF 世界位姿。SC 按外观检索出最相似的建图关键帧(+相对偏航)作粗
位姿，点面 ICP 精化。对照组"无外观检索"只从地图中心盲配——除中心附近外几乎全失败，凸显
place recognition 对全局重定位的必要性。复用 sc_desc / map / slam 缓存，不重算描述子。
用法：python scripts/run_relocalize.py --seq 0 [--gif]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam import scan_context as sc
from kitti_slam.relocalize import GlobalRelocalizer


def pick_queries(N, n_q, db_stride):
    """撒 n_q 个查询帧，避开建图关键帧栅格(frame%db_stride!=0)，让查询落在关键帧之间。"""
    qs = []
    for f in np.linspace(int(N * 0.02), int(N * 0.98), n_q):
        f = int(round(f))
        if f % db_stride == 0:
            f += 2
        qs.append(min(max(f, 1), N - 1))
    return sorted(set(qs))


def run(seq, n_q=90, db_stride=5, n_ring=20, n_sector=60, max_range=80.0,
        icp_max_dist=4.0, ok_m=2.0):
    mp = np.load(ROOT / 'reports' / f'map_seq{seq:02d}.npz')
    sl = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    opt = sl['opt']
    N = len(opt)

    cache = ROOT / 'reports' / f'sc_desc_seq{seq:02d}_s{db_stride}_{n_ring}x{n_sector}.npz'
    if not cache.exists():
        raise SystemExit(f'missing {cache.name}; run scripts/run_scan_context.py --seq {seq} first')
    d = np.load(cache)
    db_scs, db_keys = d['scs'], d['keys']
    db_frames = list(range(0, N, db_stride))[:len(db_scs)]
    db_poses = opt[db_frames]
    print(f'seq{seq:02d}: prior map {len(mp["map_pts"])//1000}k pts, '
          f'SC db {len(db_scs)} keyframes (stride {db_stride})')

    rel = GlobalRelocalizer(mp['map_pts'], db_scs, db_keys, db_poses, n_ring=n_ring,
                            n_sector=n_sector, max_range=max_range, icp_max_dist=icp_max_dist)
    # 对照组：无检索，一律从地图中心(单位朝向)盲配
    center = np.eye(4)
    center[:3, 3] = mp['map_pts'][:, :3].mean(0)

    qf = pick_queries(N, n_q, db_stride)
    rows = []
    t0 = time.perf_counter()
    for k, f in enumerate(qf):
        scan = io.read_velodyne(seq, f)
        r = rel.relocalize(scan)
        gt = opt[f, :3, 3]
        err = float(np.linalg.norm(r['T'][:3, 3] - gt))
        cerr = float(np.linalg.norm(r['coarse'][:3, 3] - gt))
        retr = float(np.linalg.norm(db_poses[r['db_idx'], :3, 3] - gt))  # 检索到的关键帧离真值多远
        bT, bfit, _ = rel.loc.localize(scan, center)                    # 对照：中心盲配
        berr = float(np.linalg.norm(bT[:3, 3] - gt))
        rows.append(dict(f=f, gt=gt, T=r['T'], coarse=r['coarse'], scd=r['sc_dist'],
                         fit=r['fitness'], err=err, cerr=cerr, retr=retr, berr=berr,
                         bfit=bfit, bT=bT))
        if k % max(1, len(qf) // 8) == 0:
            print(f'  reloc {k}/{len(qf)} f={f} scd={r["sc_dist"]:.3f} '
                  f'coarse={cerr:5.1f}m -> refined={err:5.2f}m (fit {r["fitness"]:.2f})', flush=True)
    wall = time.perf_counter() - t0

    err = np.array([r['err'] for r in rows])
    cerr = np.array([r['cerr'] for r in rows])
    retr = np.array([r['retr'] for r in rows])
    berr = np.array([r['berr'] for r in rows])
    ok = err < ok_m
    res = dict(seq=seq, mp=mp, opt=opt, rows=rows, qf=qf, ok_m=ok_m, wall=wall,
               succ=float(ok.mean()), med=float(np.median(err[ok])) if ok.any() else float('nan'),
               retr_hit=float((retr < 10.0).mean()), base_succ=float((berr < ok_m).mean()),
               err=err, cerr=cerr, retr=retr, berr=berr, ok=ok)
    return res


def summary(r):
    print(f"\n=== seq{r['seq']:02d} global relocalization (no pose prior) ===")
    print(f"  queries: {len(r['qf'])}  ·  {r['wall']/len(r['qf'])*1000:.0f} ms/query")
    print(f"  SC retrieval hit(<10m): {r['retr_hit']*100:.0f}%   "
          f"coarse median {np.median(r['cerr']):.1f} m")
    print(f"  SC+ICP success(<{r['ok_m']:.0f}m): {r['succ']*100:.0f}%   "
          f"median refined err {r['med']:.2f} m")
    print(f"  baseline (no place recognition, blind ICP from map center): "
          f"success {r['base_succ']*100:.0f}%")


def _occ_ext(mp):
    occ = mp['occ']; res = float(mp['res']); x0 = float(mp['x0']); y0 = float(mp['y0'])
    return occ, [x0, x0 + occ.shape[1] * res, y0, y0 + occ.shape[0] * res]


def render(r, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps

    occ, ext = _occ_ext(r['mp'])
    opt = r['opt']
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1.5, 1.0]})
    fig.patch.set_facecolor(ps.BG)

    # 左：地图 + 每个查询「真位姿↔重定位估计」连线（成功绿 / 失败红）
    a0 = ax[0]
    a0.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.9)
    a0.plot(opt[:, 0, 3], opt[:, 1, 3], '-', color=ps.MUTED, lw=0.6, alpha=0.6)
    for row, ok in zip(r['rows'], r['ok']):
        gt, est = row['gt'], row['T'][:3, 3]
        c = ps.START if ok else '#ff3b3b'
        a0.plot([gt[0], est[0]], [gt[1], est[1]], '-', color=c, lw=1.0, alpha=0.9)
    a0.scatter([row['gt'][0] for row in r['rows']], [row['gt'][1] for row in r['rows']],
               s=14, color=ps.GT, label='query true pose', zorder=4)
    a0.scatter([row['T'][0, 3] for row in r['rows']], [row['T'][1, 3] for row in r['rows']],
               s=10, marker='x', color=ps.EST, label='relocalized estimate', zorder=5)
    a0.set_aspect('equal'); a0.set_xlabel('x [m]'); a0.set_ylabel('y [m]')
    a0.set_title(f"cold-start relocalization on prior map  ·  "
                 f"{r['succ']*100:.0f}% success (<{r['ok_m']:.0f} m)", color=ps.FG)
    ps.style_ax(a0); ps.style_legend(a0.legend(loc='upper right', fontsize=9))

    # 右：成功率-阈值曲线，SC粗检索 / SC+ICP精化 / 无检索盲配 三条
    a1 = ax[1]
    th = np.linspace(0, 12, 200)
    for arr, col, lab in [(r['cerr'], ps.ACCENT, 'SC retrieval only (coarse)'),
                          (r['err'], ps.EST, 'SC + scan-to-map ICP (refined)'),
                          (r['berr'], ps.MUTED, 'no place recognition (blind ICP)')]:
        frac = [(arr < t).mean() for t in th]
        a1.plot(th, frac, '-', color=col, lw=2.2, label=lab)
    a1.axvline(r['ok_m'], color='#556', ls=':', lw=1)
    a1.set_xlabel('position error threshold [m]'); a1.set_ylabel('fraction of queries localized')
    a1.set_ylim(0, 1.02); a1.set_xlim(0, 12)
    a1.set_title(f"refined median {r['med']:.2f} m  ·  retrieval hit {r['retr_hit']*100:.0f}%",
                 color=ps.FG)
    ps.style_ax(a1); ps.style_legend(a1.legend(loc='lower right', fontsize=9))

    fig.suptitle(f"KITTI seq{r['seq']:02d}: global relocalization = Scan Context (appearance) "
                 f"→ scan-to-map ICP, no pose prior", color=ps.FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ps.savefig(fig, out)
    print('  saved', out)


def render_gif(r, out, fps=12, dpi=68, max_frames=72):
    """冷启动重定位逐帧扫过：当前查询的激光按重定位位姿贴到地图上(青)，真位姿绿环、估计橙叉；
    已解出的查询点累积亮起(成功绿/失败红)，标题滚动 SC 距离、粗→精误差、累计成功率。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from kitti_slam import plotstyle as ps

    occ, ext = _occ_ext(r['mp'])
    rows, ok = r['rows'], r['ok']
    ks = list(range(len(rows)))
    if len(ks) > max_frames:
        ks = list(range(0, len(rows), int(np.ceil(len(rows) / max_frames))))
    scans = {}                                       # 预读并抽稀当前查询点(世界系 xy)
    for k in ks:
        s = io.read_velodyne(r['seq'], rows[k]['f'])[:, :3]
        s = s[np.linalg.norm(s[:, :2], axis=1) < 45][::12]
        scans[k] = s

    fig, ax = plt.subplots(figsize=(9.5, 8.4)); fig.patch.set_facecolor(ps.BG)
    ax.imshow(occ, origin='lower', extent=ext, cmap=ps.OCC, alpha=0.9)
    ax.plot(r['opt'][:, 0, 3], r['opt'][:, 1, 3], '-', color=ps.MUTED, lw=0.6, alpha=0.5)
    sc_sca = ax.scatter([], [], s=1.5, color=ps.GT, alpha=0.55, zorder=3)   # 当前帧激光
    succ = ax.scatter([], [], s=26, color=ps.START, edgecolor='none', zorder=4)
    fail = ax.scatter([], [], s=26, marker='x', color='#ff3b3b', zorder=4)
    cur_gt, = ax.plot([], [], 'o', mfc='none', mec=ps.START, mew=2, ms=15, zorder=6)
    cur_est, = ax.plot([], [], 'x', color=ps.EST, mew=2.5, ms=12, zorder=7)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ps.style_ax(ax)
    ttl = fig.suptitle('', color=ps.FG, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    empty = np.empty((0, 2))

    def update(kf):
        k = ks[kf]
        row = rows[k]
        T = row['T']
        w = scans[k] @ T[:3, :3].T + T[:3, 3]
        sc_sca.set_offsets(w[:, :2])
        done = ks[:kf + 1]
        sg = np.array([rows[j]['gt'][:2] for j in done if ok[j]]) if any(ok[j] for j in done) else empty
        sf = np.array([rows[j]['gt'][:2] for j in done if not ok[j]]) if any(not ok[j] for j in done) else empty
        succ.set_offsets(sg); fail.set_offsets(sf)
        cur_gt.set_data([row['gt'][0]], [row['gt'][1]])
        cur_est.set_data([T[0, 3]], [T[1, 3]])
        sr = np.mean([ok[j] for j in done]) * 100
        ttl.set_text(f"KITTI seq{r['seq']:02d} · cold-start relocalization (no pose prior)\n"
                     f"query {kf+1}/{len(ks)} · SC dist {row['scd']:.3f} · "
                     f"coarse {row['cerr']:.1f} m → refined {row['err']:.2f} m · "
                     f"success {sr:.0f}%")
        return ()

    anim = FuncAnimation(fig, update, frames=len(ks), interval=1000 / fps)
    ps.save_gif(anim, out, fps=fps, dpi=dpi, disposal=2)
    plt.close(fig)
    print('  saved', out, f'({Path(out).stat().st_size/1e6:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--n-queries', type=int, default=90)
    ap.add_argument('--db-stride', type=int, default=5)
    ap.add_argument('--icp-max-dist', type=float, default=4.0)
    ap.add_argument('--ok-m', type=float, default=2.0, help='判成功的位置误差阈值(米)')
    ap.add_argument('--gif', action='store_true')
    a = ap.parse_args(argv)

    r = run(a.seq, n_q=a.n_queries, db_stride=a.db_stride,
            icp_max_dist=a.icp_max_dist, ok_m=a.ok_m)
    summary(r)
    render(r, str(ROOT / 'reports' / f'relocalize_seq{a.seq:02d}.png'))
    np.savez(ROOT / 'reports' / f'reloc_seq{a.seq:02d}.npz',
             err=r['err'], cerr=r['cerr'], retr=r['retr'], berr=r['berr'],
             succ=r['succ'], base_succ=r['base_succ'], retr_hit=r['retr_hit'])
    if a.gif:
        render_gif(r, str(ROOT / 'reports' / f'relocalize_seq{a.seq:02d}.gif'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
