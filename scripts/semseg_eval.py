"""点级 mIoU 评测（SemanticKITTI val = seq08）：逐点预测 vs 逐点真值标签。

把距离图分割结果投回每个 3D 点，按点累计各类交/并集，报 19 类 IoU + mIoU。这是官方口径
(点级、忽略 unlabeled)。用法：python scripts/semseg_eval.py [--stride 1 --max-frames -1]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default=str(ROOT / 'reports' / 'semseg_rangeunet.pth'))
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=-1)
    a = ap.parse_args(argv)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, W = load_model(a.weights, dev)

    ids = []
    for s in D.VAL_SEQS:
        ids += [(s, i) for i in D.frame_ids(s)[::a.stride]]
    if a.max_frames > 0:
        ids = ids[:a.max_frames]
    inter = np.zeros(D.N_CLASSES, np.int64); union = np.zeros(D.N_CLASSES, np.int64)
    t0 = time.perf_counter()
    for k, (s, i) in enumerate(ids):
        scan = D.read_scan(s, i)
        gt = D.raw_to_train(D.read_label(s, i))
        pr = predict_points(net, scan, W, dev)
        valid = gt != 255
        for c in range(D.N_CLASSES):
            pc = (pr == c) & valid; gc = (gt == c) & valid
            inter[c] += int(np.logical_and(pc, gc).sum())
            union[c] += int(np.logical_or(pc, gc).sum())
        if k % 200 == 0:
            print(f'  {k}/{len(ids)}', flush=True)
    iou = inter / np.maximum(union, 1)
    ms = (time.perf_counter() - t0) / len(ids) * 1000
    print(f'\n=== SemanticKITTI val (seq08) point-level IoU · {len(ids)} frames · {ms:.0f} ms/frame ===')
    for c in range(D.N_CLASSES):
        print(f'  {D.CLASS_NAMES[c]:16s} {iou[c]*100:5.1f}')
    print(f'  {"mIoU":16s} {np.mean(iou)*100:5.1f}')
    np.savez(ROOT / 'reports' / 'semseg_eval.npz', iou=iou, miou=np.mean(iou),
             names=D.CLASS_NAMES)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
