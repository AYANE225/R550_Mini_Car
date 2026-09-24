"""同口径点级 mIoU 对照：自研 0.88M 距离图 U-Net vs 集成开源 WaffleIron-48-256(6.1M)。

单趟遍历 SemanticKITTI val(seq08)：每帧同一份 GT、同一批点,分别取两模型逐点预测,
各自累计 19 类交/并集,报点级 IoU + mIoU(官方口径,忽略 unlabeled)。保证两条腿完全可比。
用法：python scripts/semseg_compare_eval.py [--stride 1 --max-frames -1]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semseg import data as D
from semseg.infer import load_model, predict_points
from semseg.opensource import WaffleIronSeg


def _accum(pr, gt, inter, union, valid):
    for c in range(D.N_CLASSES):
        pc = (pr == c) & valid
        gc = (gt == c) & valid
        inter[c] += int(np.logical_and(pc, gc).sum())
        union[c] += int(np.logical_or(pc, gc).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=str(ROOT / 'reports' / 'semseg_rangeunet.pth'))
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max-frames', type=int, default=-1)
    a = ap.parse_args(argv)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    net, W = load_model(a.weights, dev)
    wi = WaffleIronSeg(dev=dev)

    frames = D.frame_ids('08')[::a.stride]
    if a.max_frames > 0:
        frames = frames[:a.max_frames]

    im = np.zeros(D.N_CLASSES, np.int64); un = np.zeros(D.N_CLASSES, np.int64)   # mine
    iw = np.zeros(D.N_CLASSES, np.int64); uw = np.zeros(D.N_CLASSES, np.int64)   # waffleiron
    t0 = time.perf_counter()
    for k, f in enumerate(frames):
        scan = D.read_scan('08', f)
        gt = D.raw_to_train(D.read_label('08', f))
        valid = gt != 255
        _accum(predict_points(net, scan, W, dev), gt, im, un, valid)
        _accum(wi.predict_frame(f), gt, iw, uw, valid)
        if k % 200 == 0:
            print(f'  {k}/{len(frames)}', flush=True)

    iou_m = im / np.maximum(un, 1)
    iou_w = iw / np.maximum(uw, 1)
    dt = time.perf_counter() - t0
    print(f'\n=== SemanticKITTI val (seq08) · {len(frames)} frames · {dt:.0f}s ===')
    print(f'  {"class":16s} {"mine":>7s} {"WaffleIron":>11s}')
    for c in range(D.N_CLASSES):
        print(f'  {D.CLASS_NAMES[c]:16s} {iou_m[c]*100:7.1f} {iou_w[c]*100:11.1f}')
    print(f'  {"mIoU":16s} {iou_m.mean()*100:7.1f} {iou_w.mean()*100:11.1f}')
    print(f'  {"params":16s} {"0.88M":>7s} {"6.1M":>11s}')

    out = ROOT / 'reports' / 'semseg_compare.npz'
    np.savez(out, iou_mine=iou_m, miou_mine=iou_m.mean(),
             iou_wi=iou_w, miou_wi=iou_w.mean(),
             names=D.CLASS_NAMES, n_frames=len(frames))
    print('  saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
