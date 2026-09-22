"""M3a 建图：用 SLAM 优化位姿拼 3D 体素地图 + 投影 2D 占据栅格，出 BEV 与 3D 图。

依赖 run_slam.py 产出的 reports/slam_seqSS.npz（含 opt 位姿）。
用法：python scripts/run_mapping.py --seq 0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io
from kitti_slam.mapping import build_point_map, occupancy_2d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--voxel', type=float, default=0.3)
    ap.add_argument('--res', type=float, default=0.3, help='2D 占据栅格分辨率(米)')
    args = ap.parse_args(argv)

    npz = ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz'
    data = np.load(npz)
    opt = data['opt']
    print(f'seq{args.seq:02d}: building map from {len(opt)} optimized poses ...')
    t0 = time.perf_counter()
    map_pts = build_point_map(opt, lambda i: io.read_velodyne(args.seq, i),
                              stride=args.stride, voxel=args.voxel)
    grid = occupancy_2d(map_pts, res=args.res)
    print(f'  map: {len(map_pts)} pts, occ grid {grid.occ.shape} '
          f'({grid.occ.mean():.1%} occupied), {time.perf_counter() - t0:.0f}s')

    out = ROOT / 'reports' / f'map_seq{args.seq:02d}.npz'
    np.savez(out, map_pts=map_pts.astype(np.float32), occ=grid.occ,
             res=grid.res, x0=grid.x0, y0=grid.y0)
    print('  saved', out.name)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    traj = opt[:, :3, 3]
    # BEV：2D 占据 + 轨迹（SLAM 世界系 z 朝上，水平面就是 x–y）
    fig, ax = plt.subplots(figsize=(10, 9))
    ext = [grid.x0, grid.x0 + grid.occ.shape[1] * grid.res,
           grid.y0, grid.y0 + grid.occ.shape[0] * grid.res]
    ax.imshow(grid.occ, origin='lower', extent=ext, cmap='Greys', alpha=0.85)
    ax.plot(traj[:, 0], traj[:, 1], '-', color='tab:blue', lw=1.5, label='SLAM trajectory')
    ax.plot(traj[0, 0], traj[0, 1], 'go', ms=10, label='start')
    ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.2)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(f'KITTI seq{args.seq:02d}: built 2D occupancy map + trajectory')
    fig.tight_layout(); fig.savefig(ROOT / 'reports' / f'map_seq{args.seq:02d}_bev.png', dpi=120)
    print('  saved BEV')

    # 3D 点云地图（按高度着色，抽稀）
    sub = map_pts[::5]
    fig3 = plt.figure(figsize=(11, 7))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], s=0.4, cmap='viridis')
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]'); ax3.set_zlabel('z [m]')
    ax3.set_title(f'KITTI seq{args.seq:02d}: 3D LiDAR map (N={len(map_pts)})')
    fig3.tight_layout(); fig3.savefig(ROOT / 'reports' / f'map_seq{args.seq:02d}_3d.png', dpi=120)
    print('  saved 3D')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
