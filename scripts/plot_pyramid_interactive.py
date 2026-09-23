"""金字塔五层「交互版」：Plotly 3D，真按钮一键在【分离五层】↔【合并重叠】间切换。

GitHub README 是被 sanitize 的 Markdown，跑不了 JS/onclick，内联做不出真按钮；
这里把五层做成 5 条 Plotly trace + updatemenus 真按钮，导出自包含 HTML：
  · 左上按钮「🔀 分离五层 / 📚 合并重叠」把五层整体拉开或叠回同一平面；
  · 右侧图例点一下单独显隐某层、双击只留一层；拖拽即可旋转俯仰、滚轮缩放。
数据来源与 plot_pyramid.py 完全一致；缺 VGGT 层则退回四层。放 docs/，
用 GitHub Pages 或 htmlpreview 打开。用法：python scripts/plot_pyramid_interactive.py --seq 0
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_layers(seq, rng):
    """读五层原始数据（点数抽稀到可嵌进 HTML 的量级）。"""
    sl = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    traj = sl['opt'][:, :2, 3]
    mp = np.load(ROOT / 'reports' / f'map_seq{seq:02d}.npz')['map_pts']
    dense = mp[rng.choice(len(mp), min(15000, len(mp)), replace=False)]
    sparse = mp[rng.choice(len(mp), min(7000, len(mp)), replace=False)]
    tracks = {}
    for c in sorted((ROOT / 'reports').glob(f'demo_cache_seq{seq:02d}_*.pkl')):
        tracks = pickle.load(open(c, 'rb'))[4]
        break
    vggt = None
    vnpz = ROOT / 'reports' / f'vggt_fused_seq{seq:02d}.npz'
    if vnpz.exists():
        vf = np.load(vnpz)
        vp, vc = vf['points'], vf['colors']
        sub = rng.choice(len(vp), min(11000, len(vp)), replace=False)
        vggt = (vp[sub], np.clip(vc[sub] ** 0.7 * 1.15, 0, 1))
    return traj, dense, sparse, tracks, vggt


def build_traces(seq, gap):
    """自底向上生成 5 条 trace（初始 z 即「分离」高度 = 层序号 × gap）。"""
    import plotly.graph_objects as go
    rng = np.random.default_rng(0)
    traj, dense, sparse, tracks, vggt = load_layers(seq, rng)
    tr = []

    def zf(n):                                # 当前层的分离高度：已入栈层数 × gap
        return np.full(n, len(tr) * gap)

    if vggt is not None:                      # ⑤ 底：VGGT×LiDAR 稠密带色重建
        vp, vc = vggt
        cols = ['rgb(%d,%d,%d)' % tuple(int(x) for x in c) for c in (vc * 255)]
        tr.append(go.Scatter3d(x=vp[:, 0], y=vp[:, 1], z=zf(len(vp)), mode='markers',
                               name='⑤ VGGT×LiDAR dense',
                               marker=dict(size=1.4, color=cols, opacity=0.9)))
    dz = dense[:, 2] - dense[:, 2].min()      # ④ LiDAR 稠密建图（按高度着色）
    tr.append(go.Scatter3d(x=dense[:, 0], y=dense[:, 1], z=zf(len(dense)), mode='markers',
                           name='④ LiDAR dense map',
                           marker=dict(size=1.5, color=dz, colorscale='Turbo', opacity=0.9)))
    tr.append(go.Scatter3d(x=sparse[:, 0], y=sparse[:, 1], z=zf(len(sparse)), mode='markers',
                           name='③ LiDAR point cloud',       # ③ 稀疏激光（单色青）
                           marker=dict(size=1.3, color='#39d0e0', opacity=0.6)))
    xs, ys = [], []                           # ② 动态物体：动态轨迹红线（NaN 分段）
    for is_dyn, h in tracks.values():
        if is_dyn:
            xs += list(h[:, 1]) + [np.nan]
            ys += list(h[:, 2]) + [np.nan]
    if xs:
        tr.append(go.Scatter3d(x=xs, y=ys, z=zf(len(xs)), mode='lines',
                               name='② dynamic objects', line=dict(color='#ff3b3b', width=4)))
    tr.append(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=zf(len(traj)), mode='lines',
                           name='① trajectory', line=dict(color='#ffd24d', width=5)))  # ① 顶：轨迹
    n_dyn = sum(v[0] for v in tracks.values())
    return tr, len(traj), n_dyn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--gap', type=float, default=185.0, help='分离态层间垂直间距(米)')
    args = ap.parse_args(argv)

    import plotly.graph_objects as go
    tr, n_traj, n_dyn = build_traces(args.seq, args.gap)
    fig = go.Figure(data=tr)

    # updatemenus 真按钮：整体改写每条 trace 的 z —— 分离(层序号×gap) / 合并(全 0)
    sep_z = [np.full(len(t.x), k * args.gap) for k, t in enumerate(tr)]
    mrg_z = [np.zeros(len(t.x)) for t in tr]
    idx = list(range(len(tr)))
    fig.update_layout(updatemenus=[dict(
        type='buttons', direction='right', x=0.01, y=0.99, xanchor='left', yanchor='top',
        pad=dict(l=4, r=4, t=4, b=4), bgcolor='#1a1a26', bordercolor='#444',
        font=dict(color='#eee', size=13), showactive=True, buttons=[
            dict(label='🔀 分离五层', method='restyle', args=[{'z': sep_z}, idx]),
            dict(label='📚 合并重叠', method='restyle', args=[{'z': mrg_z}, idx])])])

    ax = dict(backgroundcolor='#0b0b12', gridcolor='#1e1e2a', color='#889',
              showspikes=False, zerolinecolor='#1e1e2a')
    fig.update_layout(
        paper_bgcolor='#0b0b12', font_color='#cdd', showlegend=True,
        margin=dict(l=0, r=0, t=54, b=0),
        title=dict(text=f'KITTI seq{args.seq:02d} · perception + SLAM output pyramid '
                        f'&nbsp;—&nbsp; {n_traj} frames · {n_dyn} dynamic tracks '
                        '&nbsp;·&nbsp; drag to rotate, click legend to toggle a layer',
                   x=0.5, xanchor='center', font=dict(color='#fff', size=15)),
        legend=dict(bgcolor='rgba(20,20,30,0.6)', font=dict(color='#eee', size=12),
                    itemsizing='constant', x=0.99, xanchor='right', y=0.9),
        scene=dict(bgcolor='#0b0b12', aspectmode='data',
                   xaxis=dict(title='x [m]', **ax), yaxis=dict(title='y [m]', **ax),
                   zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=0.85))))

    out = ROOT / 'docs' / f'pyramid_seq{args.seq:02d}.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs='cdn', full_html=True,
                   config={'displaylogo': False, 'scrollZoom': True})
    print('saved', out, f'({out.stat().st_size / 1e6:.2f} MB, {len(tr)} layers)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
