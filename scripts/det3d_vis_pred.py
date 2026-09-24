"""可视化 PointPillars 预测：某帧 BEV 上画预测框(绿,带分数) + GT 框(红)。

用法：python scripts/det3d_vis_pred.py --idx 1 --split val --score 0.3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from det3d import kitti_det as kd
from det3d.anchors import generate_anchors, _corners
from det3d.infer import postprocess
from det3d.pointpillars import PointPillars
from det3d.voxelize import points_to_pillars


def load_model(dev='cuda'):
    net = PointPillars().to(dev)
    ck = torch.load(ROOT / 'reports' / 'pp_ckpt.pth', map_location=dev)
    net.load_state_dict(ck['model']); net.eval()
    print(f'loaded ckpt @ it {ck.get("it", "?")}')
    return net


def infer_frame(net, anchors, idx, dev='cuda', score=0.3, pts=None):
    if pts is None:
        pts = kd.read_velodyne('training', idx)
    pil, coords, npt = points_to_pillars(pts)
    maxpts = pil.shape[1]
    mask = torch.arange(maxpts, device=dev)[None] < torch.from_numpy(npt).to(dev)[:, None]
    co = torch.from_numpy(np.c_[np.zeros(len(coords), np.int32), coords]).to(dev)
    with torch.no_grad():
        preds, _ = net(torch.from_numpy(pil).to(dev), mask, co, 1)
    p = {k: v[0].float().cpu().numpy() for k, v in preds.items()}
    return postprocess(p, anchors, score_thr=score), pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--idx', type=int, default=None, help='帧号；不给则取 val 里第 --nth 个')
    ap.add_argument('--split', default='val')
    ap.add_argument('--nth', type=int, default=0)
    ap.add_argument('--score', type=float, default=0.3)
    args = ap.parse_args(argv)

    idx = args.idx if args.idx is not None else kd.load_split(args.split)[args.nth]
    anchors = generate_anchors(248, 216)
    net = load_model()
    (boxes, scores), pts = infer_frame(net, anchors, idx, score=args.score)
    calib = kd.read_calib('training', idx)
    gt = kd.read_labels('training', idx, calib)
    print(f'frame {idx:06d}: {len(boxes)} preds (score>={args.score}), {len(gt)} GT cars')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from kitti_slam import plotstyle as ps
    m = (pts[:, 0] > 0) & (pts[:, 0] < 70) & (np.abs(pts[:, 1]) < 40)
    p = pts[m]
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(ps.BG)
    ax.scatter(p[:, 0], p[:, 1], s=0.4, c=p[:, 2], cmap='turbo', linewidths=0, rasterized=True)
    for g in gt:
        ax.add_patch(Polygon(_corners(g), closed=True, fill=False, ec='#ff5252', lw=2))
    for b, s in zip(boxes, scores):
        ax.add_patch(Polygon(_corners(b), closed=True, fill=False, ec='#39ff9e', lw=2))
        ax.text(b[0], b[1], f'{s:.2f}', color='#39ff9e', fontsize=8)
    ax.plot([], [], color='#ff5252', label=f'GT ({len(gt)})')
    ax.plot([], [], color='#39ff9e', label=f'pred ({len(boxes)})')
    ax.plot(0, 0, 'w^', ms=12)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'PointPillars pred (green) vs GT (red) — frame {idx:06d}')
    ps.style_ax(ax)
    ps.style_legend(ax.legend(loc='upper right'))
    out = ROOT / 'reports' / f'det3d_pred_{idx:06d}.png'
    ps.savefig(fig, out)
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
