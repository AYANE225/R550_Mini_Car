"""按 KITTI 官方相对误差指标(平移%% / 旋转 deg/m)评测各序列的 SLAM 结果 + ATE。

依赖 run_slam/run_multi 产出的 reports/slam_seqSS.npz。
用法：python scripts/eval_kitti.py [--seqs 0 2 5 6 7 9]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.metrics import ate, kitti_rpe, path_length


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seqs', type=int, nargs='+', default=[0, 2, 5, 6, 7, 9])
    args = ap.parse_args(argv)

    hdr = f"{'seq':>4} {'length':>8} {'ATE(m)':>8} {'t_err%':>8} {'r_err(deg/m)':>13}"
    print(hdr); print('-' * len(hdr))
    rows = []
    for s in args.seqs:
        npz = ROOT / 'reports' / f'slam_seq{s:02d}.npz'
        if not npz.exists():
            print(f'{s:>4}  (no slam_seq{s:02d}.npz, skip)'); continue
        opt = np.load(npz)['opt']
        gt = io.gt_poses_velodyne(s)[:len(opt)]
        m = kitti_rpe(opt, gt)
        a = ate(opt, gt)
        rows.append((s, path_length(opt), a['ate_rmse_m'], m['trans_err_pct'], m['rot_err_deg_per_m']))
        print(f"{s:>4} {rows[-1][1]:>8.0f} {rows[-1][2]:>8.2f} {rows[-1][3]:>8.2f} {rows[-1][4]:>13.4f}")
    if rows:
        arr = np.array([r[2:] for r in rows])
        print('-' * len(hdr))
        print(f"{'mean':>4} {'':>8} {arr[:,0].mean():>8.2f} {arr[:,1].mean():>8.2f} {arr[:,2].mean():>13.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
