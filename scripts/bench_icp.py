"""ICP 后端对拍：从零实现的 C++/Eigen 版 vs 纯 NumPy 版 vs Open3D，比精度与速度。

在 KITTI 相邻两帧上：各自跑 point-to-plane ICP，报耗时与相对位姿一致性。
用法：python scripts/bench_icp.py --seq 0 --frame 0 --repeat 20
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.registration import preprocess, icp_point_to_plane as o3d_icp
from kitti_slam.icp_py import icp_point_to_plane as py_icp
from kitti_slam import icp_cpp


def _timeit(fn, repeat):
    fn()                                    # warmup
    t = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return out, (time.perf_counter() - t) / repeat * 1000     # ms/次


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--max-dist', type=float, default=1.0)
    args = ap.parse_args(argv)

    src_pc = preprocess(io.read_velodyne(args.seq, args.frame + 1), voxel=0.5)
    tgt_pc = preprocess(io.read_velodyne(args.seq, args.frame), voxel=0.5)
    src = np.asarray(src_pc.points)
    tgt = np.asarray(tgt_pc.points)
    nrm = np.asarray(tgt_pc.normals)
    init = np.eye(4)
    print(f'seq{args.seq:02d} f{args.frame}->{args.frame+1}: '
          f'src {len(src)} pts, tgt {len(tgt)} pts, {args.repeat} runs each')

    (Tc, fc, rc), tc = _timeit(lambda: icp_cpp.icp_point_to_plane(
        src, tgt, nrm, init, args.max_dist, 30), args.repeat)
    (Tp, fp, rp), tp = _timeit(lambda: py_icp(src, tgt, nrm, init, args.max_dist, 30), args.repeat)
    (To, fo, ro), to = _timeit(lambda: o3d_icp(src_pc, tgt_pc, init, args.max_dist, 30), args.repeat)

    def dtrans(A, B):
        return float(np.linalg.norm(A[:3, 3] - B[:3, 3]))

    print('\n backend      time(ms)   fitness   rmse(m)   Δtrans-vs-Open3D(m)')
    print(f' C++/Eigen   {tc:8.2f}   {fc:6.3f}   {rc:.4f}    {dtrans(Tc, To):.4f}')
    print(f' NumPy       {tp:8.2f}   {fp:6.3f}   {rp:.4f}    {dtrans(Tp, To):.4f}')
    print(f' Open3D      {to:8.2f}   {fo:6.3f}   {ro:.4f}    (ref)')
    print(f'\n C++ vs NumPy speedup: {tp/tc:.1f}x    C++ vs Open3D: {to/tc:.2f}x')
    print(f' C++↔Open3D 位姿差 {dtrans(Tc, To)*100:.1f} cm（同款线性化，应很接近）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
