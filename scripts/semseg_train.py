"""从零训练距离图语义分割网（SemanticKITTI，train=00-07/09/10，val=08）。

加权交叉熵(按逆频, ignore=255) + AdamW + cosine LR + AMP。每轮在 val(seq08)上算
像素级 mIoU，存最优权重到 reports/semseg_rangeunet.pth。点级 mIoU 见 semseg_eval.py。
用法：python scripts/semseg_train.py --epochs 12 --bs 16 --train-stride 2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semseg import data as D
from semseg.dataset import SemKitti
from semseg.model import RangeUNet


def class_weights(seqs, W, n_sample=400):
    """逆频类权重：w_c = 1/log(1.02 + p_c)，在抽样帧上统计。"""
    rng = np.random.default_rng(0)
    items = [(s, i) for s in seqs for i in D.frame_ids(s)]
    hist = np.zeros(D.N_CLASSES, np.float64)
    for k in rng.choice(len(items), min(n_sample, len(items)), replace=False):
        s, i = items[k]
        lab = D.raw_to_train(D.read_label(s, i))
        lab = lab[lab != 255]
        hist += np.bincount(lab, minlength=D.N_CLASSES)
    p = hist / hist.sum()
    w = 1.0 / np.log(1.02 + p)
    return torch.tensor(w / w.mean(), dtype=torch.float32), p


@torch.no_grad()
def val_miou(net, loader, dev):
    net.eval()
    inter = np.zeros(D.N_CLASSES); union = np.zeros(D.N_CLASSES)
    for img, lab in loader:
        img = img.to(dev, non_blocking=True)
        pred = net(img).argmax(1).cpu().numpy()
        lab = lab.numpy()
        for c in range(D.N_CLASSES):
            pc, lc = pred == c, lab == c
            inter[c] += np.logical_and(pc, lc).sum()
            union[c] += np.logical_and(np.logical_or(pc, lc), lab != 255).sum()
    iou = inter / np.maximum(union, 1)
    return float(np.mean(iou)), iou


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--width', type=int, default=1024)
    ap.add_argument('--train-stride', type=int, default=2)
    ap.add_argument('--val-stride', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(argv)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    tr = SemKitti(D.TRAIN_SEQS, W=a.width, stride=a.train_stride, augment=True)
    va = SemKitti(D.VAL_SEQS, W=a.width, stride=a.val_stride, augment=False)
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    print(f'train {len(tr)} frames · val {len(va)} · device {dev}')

    w, p = class_weights(D.TRAIN_SEQS, a.width)
    print('class freq %:', ' '.join(f'{n[:4]}={q*100:.1f}' for n, q in zip(D.CLASS_NAMES, p)))
    net = RangeUNet().to(dev)
    npar = sum(x.numel() for x in net.parameters()) / 1e6
    print(f'RangeUNet {npar:.2f}M params')
    crit = nn.CrossEntropyLoss(weight=w.to(dev), ignore_index=255)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs, steps_per_epoch=len(tl))
    scaler = torch.cuda.amp.GradScaler()

    best = -1.0
    out = ROOT / 'reports' / 'semseg_rangeunet.pth'
    for ep in range(a.epochs):
        net.train(); t0 = time.perf_counter(); run = 0.0
        for it, (img, lab) in enumerate(tl):
            img, lab = img.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = crit(net(img), lab)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if it % 100 == 0:
                print(f'  ep{ep} it{it}/{len(tl)} loss {loss.item():.3f}', flush=True)
        miou, iou = val_miou(net, vl, dev)
        print(f'[ep{ep}] loss {run/len(tl):.3f} · val mIoU {miou*100:.1f} · {time.perf_counter()-t0:.0f}s',
              flush=True)
        if miou > best:
            best = miou
            torch.save({'model': net.state_dict(), 'miou': miou, 'iou': iou,
                        'width': a.width, 'names': D.CLASS_NAMES}, out)
            print(f'  saved {out.name} (best mIoU {best*100:.1f})', flush=True)
    print(f'done · best val mIoU {best*100:.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
