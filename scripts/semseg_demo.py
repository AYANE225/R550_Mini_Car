"""语义分割成果展示：把逐帧点级预测按位姿累积成 BEV 语义地图 + 生长 GIF。

对 seq08 一段城区：每帧点云过分割网得逐点类别，按真值位姿(T_world_velo)搬进世界系，
落进 0.4 m 的 BEV 栅格做多数投票，随帧生长出一张语义地图(道路/人行道/建筑/植被/车…按
SemanticKITTI 官方配色)。另出静态图：最终语义地图 + 各类点级 IoU 柱(读 semseg_eval.npz)。
用法：python scripts/semseg_demo.py [--start 2800 --count 350 --stride 4 --gif]
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

RES = 0.4                                   # BEV 栅格分辨率(米)


def build(weights, start, count, stride, dev):
    """累积 BEV 语义投票网格；返回逐帧 argmax 快照(RGB)、范围、车辆路径(x,z)。"""
    net, W = load_model(weights, dev)
    poses = io.gt_poses_velodyne('08')
    fr = list(range(start, min(start + count, len(poses)), stride))
    path = poses[fr][:, [0, 2], 3]
    lo = path.min(0) - 45; hi = path.max(0) + 45
    nx = int((hi[0] - lo[0]) / RES); nz = int((hi[1] - lo[1]) / RES)
    votes = np.zeros((nz, nx, D.N_CLASSES), np.uint16)
    ext = [lo[0], lo[0] + nx * RES, lo[1], lo[1] + nz * RES]
    snaps = []
    t0 = time.perf_counter()
    for k, f in enumerate(fr):
        scan = D.read_scan('08', f)
        r = np.linalg.norm(scan[:, :3], axis=1)
        m = (r > 2.5) & (r < 40)
        cls = predict_points(net, scan[m], W, dev)
        pw = (poses[f] @ np.c_[scan[m, :3], np.ones(m.sum())].T).T   # world (cam0 系)
        cx = ((pw[:, 0] - lo[0]) / RES).astype(int)
        cz = ((pw[:, 2] - lo[1]) / RES).astype(int)
        ok = (cx >= 0) & (cx < nx) & (cz >= 0) & (cz < nz)
        np.add.at(votes, (cz[ok], cx[ok], cls[ok]), 1)
        rgb = _colorize(votes)
        snaps.append(rgb)
        if k % 20 == 0:
            print(f'  frame {k}/{len(fr)} (world f={f})', flush=True)
    print(f'  built {len(fr)} frames in {time.perf_counter()-t0:.0f}s')
    return snaps, ext, path, fr


def _colorize(votes):
    tot = votes.sum(2)
    arg = votes.argmax(2)
    rgb = np.zeros((*tot.shape, 3), np.float32)
    seen = tot > 0
    rgb[seen] = D.CLASS_COLORS[arg[seen]]
    return rgb


def _legend_handles(present):
    import matplotlib.patches as mp
    return [mp.Patch(color=D.CLASS_COLORS[c], label=D.CLASS_NAMES[c]) for c in present]


def render_png(snaps, ext, path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps

    final = snaps[-1]
    present = [c for c in range(D.N_CLASSES)
               if np.any(np.all(np.isclose(final, D.CLASS_COLORS[c]), axis=2))]
    ev = ROOT / 'reports' / 'semseg_eval.npz'
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={'width_ratios': [1.5, 1.0]})
    fig.patch.set_facecolor(ps.BG)

    a0 = ax[0]
    a0.imshow(np.clip(final, 0, 1), origin='lower', extent=ext)
    a0.plot(path[:, 0], path[:, 1], '-', color='w', lw=1.2, alpha=0.8, label='vehicle path')
    a0.set_aspect('equal'); a0.set_xlabel('x [m]'); a0.set_ylabel('z [m]')
    a0.set_title('accumulated BEV semantic map (per-point predictions, GT poses)', color=ps.FG)
    ps.style_ax(a0)
    ps.style_legend(a0.legend(handles=_legend_handles(present), loc='upper right',
                              fontsize=7, ncol=2))

    a1 = ax[1]
    if ev.exists():
        d = np.load(ev, allow_pickle=True)
        iou, miou = d['iou'], float(d['miou'])
        order = np.argsort(iou)
        y = np.arange(D.N_CLASSES)
        a1.barh(y, iou[order] * 100, color=[D.CLASS_COLORS[o] for o in order])
        a1.set_yticks(y); a1.set_yticklabels([D.CLASS_NAMES[o] for o in order], fontsize=8)
        a1.set_xlabel('point-level IoU [%]')
        a1.set_title(f'SemanticKITTI val (seq08) · mIoU {miou*100:.1f}', color=ps.FG)
        ps.style_ax(a1)
    fig.suptitle('KITTI seq08: from-scratch range-image LiDAR semantic segmentation '
                 '(0.88M params)', color=ps.FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ps.savefig(fig, out)
    print('  saved', out)


def render_gif(snaps, ext, path, out, fps=10, dpi=78):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from kitti_slam import plotstyle as ps

    present = [c for c in range(D.N_CLASSES)
               if np.any(np.all(np.isclose(snaps[-1], D.CLASS_COLORS[c]), axis=2))]
    fig, ax = plt.subplots(figsize=(9.2, 8.2)); fig.patch.set_facecolor(ps.BG)
    im = ax.imshow(np.clip(snaps[0], 0, 1), origin='lower', extent=ext)
    pl, = ax.plot([], [], '-', color='w', lw=1.2, alpha=0.85)
    cur, = ax.plot([], [], 'o', mfc='none', mec='w', mew=2, ms=13)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]'); ps.style_ax(ax)
    ps.style_legend(ax.legend(handles=_legend_handles(present), loc='upper right',
                              fontsize=7, ncol=2))
    ttl = fig.suptitle('', color=ps.FG, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    def update(k):
        im.set_data(np.clip(snaps[k], 0, 1))
        pl.set_data(path[:k + 1, 0], path[:k + 1, 1])
        cur.set_data([path[k, 0]], [path[k, 1]])
        ttl.set_text(f'KITTI seq08 · from-scratch LiDAR semantic segmentation\n'
                     f'building BEV semantic map · frame {k+1}/{len(snaps)}')
        return ()

    anim = FuncAnimation(fig, update, frames=len(snaps), interval=1000 / fps)
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

    snaps, ext, path, fr = build(a.weights, a.start, a.count, a.stride, dev)
    render_png(snaps, ext, path, str(ROOT / 'reports' / 'semseg_map_seq08.png'))
    if a.gif:
        render_gif(snaps, ext, path, str(ROOT / 'reports' / 'semseg_map_seq08.gif'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
