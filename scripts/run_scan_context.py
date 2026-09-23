"""Scan Context 外观级回环检测：里程计漂移下也能找回闭环（描述子级，自研）。

位置法回环(loop.py)靠里程计坐标近邻找候选——里程计一漂，重访点在里程计系里被推开就漏检。
Scan Context 按**外观**匹配(极坐标最大高度指纹 + 旋转不变环键)，与里程计位置无关。

本脚本在 seq00 上：
  ① 用 GT 定义"真闭环"(重访同一地点、帧号隔开足够久)的关键帧；
  ② 对比两种前端召回这些闭环的能力——位置法(扫描搜索半径 = 模拟漂移严重程度) vs Scan Context；
  ③ 把 Scan Context 检出的闭环经 ICP 验证喂进同一个位姿图，报 ATE。
仅用公开数据，耦合逻辑全自研。
用法: python scripts/run_scan_context.py --seq 0 --stride 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam import loop as loopmod
from kitti_slam import posegraph as pg
from kitti_slam import scan_context as sc
from kitti_slam.metrics import ate, path_length


def build_descriptors(seq, frames, n_ring, n_sector, max_range, cache):
    """每个关键帧一张 scan context + 环键；缓存到 npz（gitignore）。"""
    if cache.exists():
        d = np.load(cache)
        return d['scs'], d['keys']
    scs, keys = [], []
    for t, f in enumerate(frames):
        s = sc.scan_context(io.read_velodyne(seq, f)[:, :3], n_ring, n_sector, max_range)
        scs.append(s)
        keys.append(sc.ring_key(s))
        if t % 200 == 0:
            print(f'  scan context {t}/{len(frames)}', flush=True)
    scs, keys = np.array(scs), np.array(keys)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, scs=scs, keys=keys)
    return scs, keys


def nearest_earlier(pos, gap_k):
    """每个关键帧下标 t：在 < t-gap_k 的历史里最近的一帧下标与距离。"""
    j = np.full(len(pos), -1, int)
    d = np.full(len(pos), np.inf)
    for t in range(len(pos)):
        if t - gap_k <= 0:
            continue
        dt, jt = cKDTree(pos[:t - gap_k]).query(pos[t])
        d[t], j[t] = dt, jt
    return j, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--min-gap', type=int, default=150, help='闭环两帧最小帧号间隔')
    ap.add_argument('--n-ring', type=int, default=20)
    ap.add_argument('--n-sector', type=int, default=60)
    ap.add_argument('--max-range', type=float, default=80.0)
    ap.add_argument('--sc-thresh', type=float, default=0.14, help='Scan Context 距离阈值(收闭环)')
    ap.add_argument('--r-true', type=float, default=8.0, help='GT 判定同一地点的距离(米)')
    ap.add_argument('--radii', type=str, default='2,3,4,5,7,10,15,20')
    ap.add_argument('--n-candidates', type=int, default=10)
    ap.add_argument('--out', type=str, default='reports/scan_context_seq{seq:02d}.png')
    a = ap.parse_args(argv)
    seq = a.seq

    slam = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    odom, opt_pos = slam['odom'], slam['opt']
    N = len(odom)
    gt = io.gt_poses_velodyne(seq)[:N]
    frames = list(range(0, N, a.stride))
    kf_gt = gt[frames][:, :3, 3]                # 关键帧 GT 位置(相机世界系, 水平面 x-z)
    kf_od = odom[frames][:, :3, 3]              # 关键帧里程计位置
    gap_k = max(1, a.min_gap // a.stride)       # 关键帧下标间隔

    cache = ROOT / 'reports' / f'sc_desc_seq{seq:02d}_s{a.stride}_{a.n_ring}x{a.n_sector}.npz'
    print(f'building scan-context descriptors for {len(frames)} keyframes ...')
    t0 = time.perf_counter()
    scs, keys = build_descriptors(seq, frames, a.n_ring, a.n_sector, a.max_range, cache)
    print(f'  done in {time.perf_counter() - t0:.0f}s')

    # —— GT 真闭环：每个关键帧是否有更早的重访伙伴 ——
    jg, dg = nearest_earlier(kf_gt, gap_k)
    has_true = dg < a.r_true
    n_true = int(has_true.sum())

    # —— Scan Context 前端（外观级，与里程计无关）——
    mgr = sc.ScanContextManager(a.n_ring, a.n_sector, a.max_range, a.n_candidates)
    for t, f in enumerate(frames):
        mgr.scs.append(scs[t]); mgr.keys.append(keys[t]); mgr.frames.append(f)
    sc_match = np.full(len(frames), -1, int)
    sc_dist = np.full(len(frames), np.nan)
    sc_yaw = np.zeros(len(frames))
    for t in range(len(frames)):
        q = mgr.query(t, min_gap_idx=gap_k)
        if q is None:
            continue
        j, d, yaw = q
        if d < a.sc_thresh:
            sc_match[t], sc_dist[t], sc_yaw[t] = j, d, yaw
    sc_acc = sc_match >= 0
    sc_correct = sc_acc & (np.linalg.norm(kf_gt - kf_gt[np.where(sc_acc, sc_match, 0)], axis=1) < a.r_true)
    sc_recall = (has_true & sc_correct).sum() / max(1, n_true)
    sc_prec = sc_correct.sum() / max(1, sc_acc.sum())

    # —— 位置法前端：扫描搜索半径(半径越小 = 里程计漂移越"致命")——
    radii = [float(x) for x in a.radii.split(',')]
    jo, do = nearest_earlier(kf_od, gap_k)
    od_correct_place = np.linalg.norm(kf_gt - kf_gt[np.where(jo >= 0, jo, 0)], axis=1) < a.r_true
    pos_recall = []
    for R in radii:
        det = (do < R) & (jo >= 0)
        pos_recall.append((has_true & det & od_correct_place).sum() / max(1, n_true))

    # —— 端到端：Scan Context 检出的闭环 → ICP 验证 → 位姿图 → ATE ——
    # 关键：ICP 初值取 Scan Context 的相对偏航(绕竖直轴 Rz(-yaw)) + 零平移(重访即共位)，
    # **不用里程计相对位姿**——漂移大时里程计初值会把两帧点云推到几公里外(fit=0)，
    # 而 SC 偏航与漂移无关，正是它救回闭环的地方。
    def _rz(a):
        c, s = np.cos(a), np.sin(a)
        T = np.eye(4); T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        return T
    load = lambda x: io.read_velodyne(seq, x)
    loops, kept_fi = [], []
    order = np.argsort(np.where(sc_acc, sc_dist, np.inf))     # 先验证最可信的
    for t in order:
        if not sc_acc[t]:
            break
        fi, fj = frames[t], frames[sc_match[t]]
        # 去重：按 SC 距离升序取最可信的，跳过与已接受回环帧号太近的(避免冗余边)
        if any(abs(fi - kf) < a.min_gap // 2 for kf in kept_fi):
            continue
        ok, T, fit, rmse = loopmod.verify(load, odom, (fi, fj), init=_rz(-sc_yaw[t]))
        print(f'  SC loop {fi}<->{fj}: scd={sc_dist[t]:.3f} yaw={np.degrees(sc_yaw[t]):+5.0f} '
              f'fit={fit:.2f} rmse={rmse:.2f} {"ACCEPT" if ok else "reject"}', flush=True)
        if ok:
            loops.append((fi, fj, T, fit))
            kept_fi.append(fi)
    opt_sc = pg.optimize(pg.build(odom, loops))

    ate_od = ate(odom, gt)['ate_rmse_m']
    ate_pos = ate(opt_pos, gt)['ate_rmse_m']
    ate_sc = ate(opt_sc, gt)['ate_rmse_m']

    print(f'\n=== seq{seq:02d} Scan Context loop closure  ({len(frames)} keyframes, {path_length(odom):.0f}m) ===')
    print(f'  GT true-revisit keyframes: {n_true}')
    print(f'  Scan Context: {int(sc_acc.sum())} detections, precision {sc_prec:.2f}, recall {sc_recall:.2f}')
    print(f'  position-based recall vs search radius:')
    for R, r in zip(radii, pos_recall):
        print(f'    r={R:4.1f} m -> recall {r:.2f}')
    print(f'  end-to-end ATE:  odometry {ate_od:.2f} m  |  position-loops(r=20) {ate_pos:.2f} m  '
          f'|  Scan-Context-loops {ate_sc:.2f} m  ({len(loops)} verified loops)')

    _render(a.out.format(seq=seq), seq, gt, frames, kf_gt, sc_acc, sc_correct, sc_match,
            radii, np.array(pos_recall), sc_recall, ate_od, ate_sc, len(loops))
    np.savez(ROOT / 'reports' / f'scan_context_seq{seq:02d}.npz',
             radii=radii, pos_recall=pos_recall, sc_recall=sc_recall, n_true=n_true,
             sc_prec=sc_prec, ate_od=ate_od, ate_pos=ate_pos, ate_sc=ate_sc)
    return 0

def _render(out, seq, gt, frames, kf_gt, sc_acc, sc_correct, sc_match,
            radii, pos_recall, sc_recall, ate_od, ate_sc, n_loops):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps

    g = gt[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    fig.patch.set_facecolor(ps.BG)

    # 左：轨迹 + Scan Context 检出的闭环连线（水平面 x-z）
    ax = axes[0]
    ax.plot(g[:, 0], g[:, 2], color=ps.GT, lw=1.8, label='ground-truth trajectory')
    lbl = True
    for t in np.where(sc_acc & sc_correct)[0]:
        p, q = kf_gt[t], kf_gt[sc_match[t]]
        ax.plot([p[0], q[0]], [p[2], q[2]], color=ps.EST, lw=1.2, alpha=0.85,
                label='Scan Context loop' if lbl else None)
        lbl = False
    ax.scatter(g[0, 0], g[0, 2], s=90, color=ps.START, edgecolor='w', zorder=5, label='start')
    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('x (m)'); ax.set_ylabel('z (m)')
    ax.set_title(f'Scan Context loop detections ({n_loops} verified)  ·  ATE {ate_od:.2f} → {ate_sc:.2f} m',
                 color=ps.FG, fontsize=11.5)
    ps.style_ax(ax); ps.style_legend(ax.legend(loc='upper left', fontsize=9))

    # 右：召回率 vs 搜索半径 —— 位置法随半径收紧塌陷，Scan Context 与半径无关(平线)
    ax = axes[1]
    ax.plot(radii, pos_recall, '-o', color=ps.MUTED, lw=2, label='position search (odometry)')
    ax.axhline(sc_recall, color=ps.EST, lw=2.4, label=f'Scan Context (appearance)  recall {sc_recall:.2f}')
    ax.set_xlabel('position-search radius (m)  ·  tighter = worse odometry drift  ←')
    ax.set_ylabel('loop recall')
    ax.set_ylim(0, 1.02)
    ax.invert_xaxis()
    ax.set_title('Appearance-based recall is drift-independent', color=ps.FG, fontsize=11.5)
    ps.style_ax(ax); ps.style_legend(ax.legend(loc='lower left', fontsize=9))

    fig.suptitle(f'Scan Context: appearance-based loop closure survives odometry drift  ·  seq{seq:02d}',
                 color=ps.FG, fontsize=13)
    ps.savefig(fig, ROOT / out)
    print(f'  saved {out}')


if __name__ == '__main__':
    raise SystemExit(main())