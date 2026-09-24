"""两条腿并排展示：同一段 seq08,自研 0.88M 距离图网 vs 集成开源 WaffleIron 6.1M。

每帧点云分别过两模型得逐点类别,按真值位姿搬进世界系,落进 0.4m BEV 栅格多数投票,
生长出两张并排的语义地图;另附各类点级 IoU 分组柱(读 semseg_compare.npz)。
用法：python scripts/semseg_compare_demo.py [--start 2800 --count 350 --stride 4 --gif]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from semseg import data as D
from semseg.infer import load_model, predict_points
from semseg.opensource import WaffleIronSeg

RES = 0.4                                   # BEV 栅格分辨率(米)


def _colorize(votes):
    tot = votes.sum(2)
    arg = votes.argmax(2)
    rgb = np.zeros((*tot.shape, 3), np.float32)
    seen = tot > 0
    rgb[seen] = D.CLASS_COLORS[arg[seen]]
    return rgb


def build(weights, start, count, stride, dev):
    """一趟遍历,累积两套 BEV 语义投票;返回两模型逐帧快照(RGB)、范围、车辆路径。"""
    net, W = load_model(weights, dev)
    wi = WaffleIronSeg(dev=dev)
    poses = io.gt_poses_velodyne('08')
    fr = list(range(start, min(start + count, len(poses)), stride))
    path = poses[fr][:, [0, 2], 3]
    lo = path.min(0) - 45; hi = path.max(0) + 45
    nx = int((hi[0] - lo[0]) / RES); nz = int((hi[1] - lo[1]) / RES)
    ext = [lo[0], lo[0] + nx * RES, lo[1], lo[1] + nz * RES]
    vm = np.zeros((nz, nx, D.N_CLASSES), np.uint16)     # mine
    vw = np.zeros((nz, nx, D.N_CLASSES), np.uint16)     # waffleiron
    snaps_m, snaps_w = [], []
    t0 = time.perf_counter()
    for k, f in enumerate(fr):
        scan = D.read_scan('08', f)
        r = np.linalg.norm(scan[:, :3], axis=1)
        m = (r > 2.5) & (r < 40)
        cls_m = predict_points(net, scan[m], W, dev)
        cls_w = wi.predict_frame(f)[m]                  # 原始点序对齐,直接用同一掩码
        pw = (poses[f] @ np.c_[scan[m, :3], np.ones(m.sum())].T).T   # world (cam0 系)
        cx = ((pw[:, 0] - lo[0]) / RES).astype(int)
        cz = ((pw[:, 2] - lo[1]) / RES).astype(int)
        ok = (cx >= 0) & (cx < nx) & (cz >= 0) & (cz < nz)
        np.add.at(vm, (cz[ok], cx[ok], cls_m[ok]), 1)
        np.add.at(vw, (cz[ok], cx[ok], cls_w[ok]), 1)
        snaps_m.append(_colorize(vm)); snaps_w.append(_colorize(vw))
        if k % 20 == 0:
            print(f'  frame {k}/{len(fr)} (world f={f})', flush=True)
    print(f'  built {len(fr)} frames in {time.perf_counter()-t0:.0f}s')
    return snaps_m, snaps_w, ext, path


def _present(rgb):
    return [c for c in range(D.N_CLASSES)
            if np.any(np.all(np.isclose(rgb, D.CLASS_COLORS[c]), axis=2))]


def _legend_handles(present):
    import matplotlib.patches as mp
    return [mp.Patch(color=D.CLASS_COLORS[c], label=D.CLASS_NAMES[c]) for c in present]


def _draw_map(ax, rgb, ext, path, title, ps):
    ax.imshow(np.clip(rgb, 0, 1), origin='lower', extent=ext)
    ax.plot(path[:, 0], path[:, 1], '-', color='w', lw=1.1, alpha=0.8, label='vehicle path')
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
    ax.set_title(title, color=ps.FG, fontsize=11)
    ps.style_ax(ax)


def _draw_bars(ax, ps):
    """从 semseg_compare.npz 画分组水平 IoU 柱(mine vs WaffleIron),按 WaffleIron 升序。"""
    cmp = ROOT / 'reports' / 'semseg_compare.npz'
    if not cmp.exists():
        ax.text(0.5, 0.5, 'run semseg_compare_eval.py first', ha='center',
                va='center', color=ps.MUTED, transform=ax.transAxes)
        ps.style_ax(ax); return
    d = np.load(cmp, allow_pickle=True)
    im, iw = d['iou_mine'] * 100, d['iou_wi'] * 100
    mm, mw = float(d['miou_mine']) * 100, float(d['miou_wi']) * 100
    order = np.argsort(iw)
    y = np.arange(D.N_CLASSES); h = 0.4
    ax.barh(y + h / 2, iw[order], height=h, color=ps.GT, label=f'WaffleIron 6.1M (mIoU {mw:.1f})')
    ax.barh(y - h / 2, im[order], height=h, color=ps.EST, label=f'from-scratch 0.88M (mIoU {mm:.1f})')
    ax.set_yticks(y); ax.set_yticklabels([D.CLASS_NAMES[o] for o in order], fontsize=8)
    ax.set_xlabel('point-level IoU [%]'); ax.set_xlim(0, 100)
    ax.set_title('SemanticKITTI val (seq08) per-class IoU, same protocol', color=ps.FG, fontsize=11)
    ps.style_ax(ax)
    ps.style_legend(ax.legend(loc='lower right', fontsize=8))


def render_png(snaps_m, snaps_w, ext, path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from kitti_slam import plotstyle as ps

    fm, fw = snaps_m[-1], snaps_w[-1]
    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor(ps.BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.35, 1.0], hspace=0.22, wspace=0.12)
    a_m = fig.add_subplot(gs[0, 0]); a_w = fig.add_subplot(gs[0, 1])
    a_b = fig.add_subplot(gs[1, :])
    _draw_map(a_m, fm, ext, path, 'from-scratch range-image U-Net (0.88M params)', ps)
    _draw_map(a_w, fw, ext, path, 'open-source WaffleIron-48-256 (6.1M params)', ps)
    present = sorted(set(_present(fm)) | set(_present(fw)))
    ps.style_legend(a_w.legend(handles=_legend_handles(present), loc='upper right',
                               fontsize=7, ncol=2))
    _draw_bars(a_b, ps)
    fig.suptitle('KITTI seq08 LiDAR semantic segmentation — two legs: '
                 'self-built + integrated open-source (same eval protocol)',
                 color=ps.FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ps.savefig(fig, out)
    print('  saved', out)


def render_gif(snaps_m, snaps_w, ext, path, out, fps=10, dpi=76):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from kitti_slam import plotstyle as ps

    present = sorted(set(_present(snaps_m[-1])) | set(_present(snaps_w[-1])))
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.6)); fig.patch.set_facecolor(ps.BG)
    im0 = ax[0].imshow(np.clip(snaps_m[0], 0, 1), origin='lower', extent=ext)
    im1 = ax[1].imshow(np.clip(snaps_w[0], 0, 1), origin='lower', extent=ext)
    pl0, = ax[0].plot([], [], '-', color='w', lw=1.1, alpha=0.85)
    pl1, = ax[1].plot([], [], '-', color='w', lw=1.1, alpha=0.85)
    c0, = ax[0].plot([], [], 'o', mfc='none', mec='w', mew=2, ms=11)
    c1, = ax[1].plot([], [], 'o', mfc='none', mec='w', mew=2, ms=11)
    for a, t in ((ax[0], 'from-scratch 0.88M'), (ax[1], 'WaffleIron 6.1M')):
        a.set_aspect('equal'); a.set_xlabel('x [m]'); a.set_ylabel('z [m]')
        a.set_title(t, color=ps.FG, fontsize=11); ps.style_ax(a)
    ps.style_legend(ax[1].legend(handles=_legend_handles(present), loc='upper right',
                                 fontsize=6, ncol=2))
    ttl = fig.suptitle('', color=ps.FG, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(k):
        im0.set_data(np.clip(snaps_m[k], 0, 1)); im1.set_data(np.clip(snaps_w[k], 0, 1))
        pl0.set_data(path[:k + 1, 0], path[:k + 1, 1]); pl1.set_data(path[:k + 1, 0], path[:k + 1, 1])
        c0.set_data([path[k, 0]], [path[k, 1]]); c1.set_data([path[k, 0]], [path[k, 1]])
        ttl.set_text(f'KITTI seq08 semantic mapping · self-built vs open-source '
                     f'· frame {k+1}/{len(snaps_m)}')
        return ()

    anim = FuncAnimation(fig, update, frames=len(snaps_m), interval=1000 / fps)
    ps.save_gif(anim, out, fps=fps, dpi=dpi, disposal=1)
    plt.close(fig)
    print('  saved', out, f'({Path(out).stat().st_size/1e6:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=str(ROOT / 'reports' / 'semseg_rangeunet.pth'))
    ap.add_argument('--start', type=int, default=2800)
    ap.add_argument('--count', type=int, default=350)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--gif', action='store_true')
    a = ap.parse_args(argv)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    snaps_m, snaps_w, ext, path = build(a.weights, a.start, a.count, a.stride, dev)
    render_png(snaps_m, snaps_w, ext, path, str(ROOT / 'reports' / 'semseg_compare_map_seq08.png'))
    if a.gif:
        render_gif(snaps_m, snaps_w, ext, path, str(ROOT / 'reports' / 'semseg_compare_map_seq08.gif'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
