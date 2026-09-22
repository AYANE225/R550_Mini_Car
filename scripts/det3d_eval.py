"""在 KITTI val 上评 PointPillars 的 Car BEV AP（IoU 0.5 / 0.7，R40 插值）。

注：为简洁按"全部 Car"评（未按 easy/mod/hard 分层，非官方分档），指标仅供横向参考。
用法：python scripts/det3d_eval.py [--max-frames 500 --score 0.1]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from det3d import kitti_det as kd
from det3d.anchors import generate_anchors, _corners
from scripts.det3d_vis_pred import load_model, infer_frame


def bev_iou(a, b):
    pa, pb = Polygon(_corners(a)), Polygon(_corners(b))
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter + 1e-9)


def ap_r40(scores, tps, n_gt):
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    tp = np.cumsum(np.array(tps)[order])
    fp = np.cumsum(1 - np.array(tps)[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    ap = 0.0
    for r in np.linspace(0, 1, 41)[1:]:                 # R40
        p = prec[rec >= r].max() if np.any(rec >= r) else 0.0
        ap += p / 40
    return ap * 100


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-frames', type=int, default=500)
    ap.add_argument('--score', type=float, default=0.1)
    args = ap.parse_args(argv)

    anchors = generate_anchors(248, 216)
    net = load_model()
    frames = kd.load_split('val')[:args.max_frames]
    res = {0.5: [[], []], 0.7: [[], []]}                 # thr -> [scores, tps]
    n_gt = 0
    t0 = time.perf_counter()
    for k, idx in enumerate(frames):
        (boxes, scores), _ = infer_frame(net, anchors, idx, score=args.score)
        gt = kd.read_labels('training', idx, kd.read_calib('training', idx))
        n_gt += len(gt)
        order = scores.argsort()[::-1]
        for thr in (0.5, 0.7):
            matched = np.zeros(len(gt), bool)
            for i in order:
                best, bj = thr, -1
                for j in range(len(gt)):
                    if matched[j]:
                        continue
                    iou = bev_iou(boxes[i], gt[j])
                    if iou >= best:
                        best, bj = iou, j
                res[thr][0].append(scores[i])
                res[thr][1].append(1 if bj >= 0 else 0)
                if bj >= 0:
                    matched[bj] = True
        if (k + 1) % 100 == 0:
            print(f'  {k+1}/{len(frames)} frames, {n_gt} GT', flush=True)
    print(f'\n=== KITTI val Car BEV AP (R40, {len(frames)} frames, {n_gt} GT, '
          f'{time.perf_counter()-t0:.0f}s) ===')
    for thr in (0.5, 0.7):
        print(f'  AP@IoU{thr}: {ap_r40(res[thr][0], res[thr][1], n_gt):.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
