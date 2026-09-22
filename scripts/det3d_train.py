"""训练 PointPillars（KITTI Car，纯 PyTorch，5090）。

用法：python scripts/det3d_train.py --epochs 40 --bs 4 [--iters N 快速冒烟]
产出 reports/pp_ckpt.pth。
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from det3d.anchors import generate_anchors
from det3d.dataset import KittiPillars, collate
from det3d.loss import det_loss
from det3d.pointpillars import PointPillars


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--iters', type=int, default=0, help='>0 时只跑这么多 iter(冒烟测试)')
    ap.add_argument('--ckpt-every', type=int, default=400, help='每多少 iter 存一次(抗中断)')
    ap.add_argument('--resume', action='store_true', help='从 reports/pp_ckpt.pth 续训')
    args = ap.parse_args(argv)

    dev = 'cuda'
    anchors = generate_anchors(248, 216)
    ds = KittiPillars('train', anchors, augment=True)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
    print(f'train frames: {len(ds)}, anchors: {len(anchors)}, batches/epoch: {len(dl)}')

    net = PointPillars().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    total_iters = args.iters if args.iters else args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_iters)
    ckpt_path = ROOT / 'reports' / 'pp_ckpt.pth'
    start_it = 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ck['model'])
        if 'opt' in ck: opt.load_state_dict(ck['opt'])
        if 'sched' in ck: sched.load_state_dict(ck['sched'])
        start_it = ck.get('it', 0)
        print(f'resumed from it {start_it}')

    def save(it):
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'it': it}, ckpt_path)

    net.train(); it = start_it; t0 = time.perf_counter()
    for ep in range(args.epochs):
        for b in dl:
            preds, _ = net(b['pillars'].to(dev), b['mask'].to(dev),
                           b['coords'].to(dev), b['batch_size'])
            loss, parts = det_loss(preds, b['labels'].to(dev), b['reg_t'].to(dev), b['dir_t'].to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step(); sched.step(); it += 1
            if it % 20 == 0:
                print(f'ep{ep} it{it}/{total_iters} loss {float(loss.detach()):.3f} '
                      f'(cls {parts["cls"]:.3f} reg {parts["reg"]:.3f} dir {parts["dir"]:.3f}) '
                      f'{(it-start_it)/(time.perf_counter()-t0):.1f} it/s', flush=True)
            if it % args.ckpt_every == 0:
                save(it)
            if args.iters and it >= start_it + args.iters:
                break
        if args.iters and it >= start_it + args.iters:
            break
    save(it)
    print(f'saved {ckpt_path.name} at it {it}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
