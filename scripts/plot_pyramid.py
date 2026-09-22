"""分层金字塔图：把系统输出分成四层水平面自底向上堆叠（数据→抽象的金字塔）。

  顶  ①轨迹 (trajectory)            —— 最抽象、最紧凑
      ②动态物体 (dynamic objects, 红=动/青=静)
      ③激光点云 (raw LiDAR points, 稀疏)
  底  ④稠密建图 (dense map, 按高度着色)  —— 数据量最大

所有层共用 seq00 SLAM 世界系的同一 XY 底面，垂直错开堆叠；图例标注在右侧。
用法：python scripts/plot_pyramid.py --seq 0
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--gap', type=float, default=185.0, help='层间垂直间距(米)')
    args = ap.parse_args(argv)

    sl = np.load(ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz')
    traj = sl['opt'][:, :2, 3]
    mp = np.load(ROOT / 'reports' / f'map_seq{args.seq:02d}.npz')['map_pts']
    rng = np.random.default_rng(0)
    dense = mp[rng.choice(len(mp), min(240000, len(mp)), replace=False)]
    sparse = mp[rng.choice(len(mp), min(38000, len(mp)), replace=False)]
    tracks = {}
    for c in sorted((ROOT / 'reports').glob(f'demo_cache_seq{args.seq:02d}_*.pkl')):
        tracks = pickle.load(open(c, 'rb'))[4]; break

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 12)); fig.patch.set_facecolor('#0b0b12')
    ax = fig.add_axes([-0.02, 0.0, 0.82, 1.0], projection='3d')     # 左侧放图、右侧留白给标注
    ax.set_facecolor('#0b0b12')

    x0, x1 = mp[:, 0].min(), mp[:, 0].max()
    y0, y1 = mp[:, 1].min(), mp[:, 1].max()
    frame = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    Z = [0, args.gap, 2 * args.gap, 3 * args.gap]                   # 底→顶

    def plane(z):
        ax.plot(frame[:, 0], frame[:, 1], z, color='#2a2a38', lw=1.0)

    # ④ 底：稠密建图（按高度着色）
    ax.scatter(dense[:, 0], dense[:, 1], np.full(len(dense), Z[0]),
               c=dense[:, 2] - dense[:, 2].min(), cmap='turbo', s=0.25, linewidths=0, depthshade=False)
    plane(Z[0])
    # ③ 激光点云（稀疏、单色青）
    ax.scatter(sparse[:, 0], sparse[:, 1], np.full(len(sparse), Z[1]),
               c='#39d0e0', s=0.3, linewidths=0, alpha=0.65, depthshade=False)
    plane(Z[1])
    # ② 动态物体：动态红线突出，静态弱化成暗灰小点
    for is_dyn, h in tracks.values():
        if is_dyn:
            ax.plot(h[:, 1], h[:, 2], np.full(len(h), Z[2]), color='#ff3b3b', lw=2.6)
            ax.scatter(h[-1, 1], h[-1, 2], Z[2], c='#ffd24d', s=14, linewidths=0, depthshade=False)
        else:
            ax.scatter(h[-1, 1], h[-1, 2], Z[2], c='#4a5a66', s=4, linewidths=0,
                       alpha=0.5, depthshade=False)
    plane(Z[2])
    # ① 轨迹（顶）
    ax.plot(traj[:, 0], traj[:, 1], np.full(len(traj), Z[3]), color='#ffd24d', lw=2.4)
    plane(Z[3])

    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.04, 0.04, 0.07, 1.0))
    ax.set_box_aspect(((x1 - x0), (y1 - y0), 3.5 * args.gap))
    ax.set_zticks([]); ax.set_xlabel('x [m]', color='#aaa'); ax.set_ylabel('y [m]', color='#aaa')
    ax.tick_params(colors='#666'); ax.grid(False)
    ax.view_init(elev=20, azim=-60)

    # 右侧图例（顶→底，配色与各层一致）
    fig.text(0.80, 0.93, 'perception + SLAM\noutput pyramid', color='w',
             fontsize=15, weight='bold', va='top')
    rows = [(0.78, '#ffd24d', '① trajectory', 'SLAM ego path'),
            (0.62, '#ff3b3b', '② dynamic objects', 'tracked moving vehicles'),
            (0.46, '#39d0e0', '③ LiDAR point cloud', 'raw sparse scans'),
            (0.30, '#39ff9e', '④ dense mapping', 'accumulated dense map')]
    for y, col, name, desc in rows:
        fig.text(0.815, y, '■', color=col, fontsize=20, va='center')
        fig.text(0.845, y + 0.008, name, color='w', fontsize=14, weight='bold', va='center')
        fig.text(0.845, y - 0.022, desc, color='#9aa', fontsize=10.5, va='center')
    fig.text(0.815, 0.10, f'KITTI seq{args.seq:02d}\n{len(traj)} frames · 3.7 km\n'
             f'{sum(v[0] for v in tracks.values())} dynamic tracks',
             color='#8899aa', fontsize=10.5, va='top')

    out = ROOT / 'reports' / f'pyramid_seq{args.seq:02d}.png'
    fig.savefig(out, dpi=150, facecolor='#0b0b12')
    print('saved', out, f'({len(tracks)} tracks, dense {len(dense)}, traj {len(traj)})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
