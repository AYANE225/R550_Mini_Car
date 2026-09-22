"""相机-LiDAR 上色地图：把 KITTI 彩色相机(image_2)按标定投影到激光点上，建**真彩稠密地图**。

每帧：velodyne 扫描 → 用 `P2 @ Tr` 投影到 image_2 像素 → 取该像素 RGB 给点上色（仅相机
视野内、z>0 的点）→ 用 SLAM 优化位姿转世界系累积 → 体素下采样成全局真彩地图。
与 VGGT 稠密重建不同：这里是**度量精确的相机-LiDAR 硬标定融合**（无学习、无尺度歧义）。
用法：python scripts/run_color_map.py --seq 0 --stride 2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_io as io


def voxel_downsample(P, C, voxel):
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[idx], C[idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--stride', type=int, default=2, help='抽帧(相邻帧覆盖高度重叠)')
    ap.add_argument('--frames', type=int, default=-1, help='-1=全序列')
    ap.add_argument('--voxel', type=float, default=0.3)
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    import matplotlib.image as mpimg
    npz = ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz'
    if not npz.exists():
        print(f'需要先跑 run_slam.py 生成 {npz.name}'); return 1
    poses = np.load(npz)['opt']                                   # T_world_velo
    cal = io.read_calib(args.seq)
    PT = cal['P2'] @ cal['Tr']                                    # 3x4：velo→像素(齐次)
    n = len(poses) if args.frames < 0 else min(args.frames, len(poses))
    imdir = io.seq_dir(args.seq) / 'image_2'

    accP, accC = np.empty((0, 3)), np.empty((0, 3))              # 已下采样的累积地图
    bufP, bufC = [], []                                         # 待并入缓冲
    t0 = time.perf_counter()
    for k, i in enumerate(range(0, n, args.stride)):
        pts = io.read_velodyne(args.seq, i)[:, :3]
        img = mpimg.imread(str(imdir / f'{i:06d}.png'))           # H,W,3 in [0,1]
        H, W = img.shape[:2]
        Xh = np.c_[pts, np.ones(len(pts))]                        # (N,4)
        uvw = Xh @ PT.T                                           # (N,3)
        z = uvw[:, 2]
        front = z > 0.1
        u = np.full(len(pts), -1.0); v = np.full(len(pts), -1.0)
        u[front] = uvw[front, 0] / z[front]
        v[front] = uvw[front, 1] / z[front]
        keep = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if keep.sum() == 0:
            continue
        px, py = u[keep].astype(int), v[keep].astype(int)
        bufC.append(img[py, px, :3])
        bufP.append((poses[i][:3, :3] @ pts[keep].T + poses[i][:3, 3:4]).T)  # → world
        if k % 200 == 199:                                       # 定期并入并下采样(限内存/加速)
            accP = np.vstack([accP] + bufP); accC = np.vstack([accC] + bufC)
            accP, accC = voxel_downsample(accP, accC, args.voxel)
            bufP, bufC = [], []
            print(f'  frame {i}/{n}: map {len(accP):,} pts ({time.perf_counter()-t0:.0f}s)', flush=True)
    accP = np.vstack([accP] + bufP); accC = np.vstack([accC] + bufC)
    P, C = voxel_downsample(accP, accC, args.voxel)
    C = C.clip(0, 1)
    print(f'colorized LiDAR map: {len(P):,} points (voxel {args.voxel}m, {time.perf_counter()-t0:.0f}s)')
    out_npz = ROOT / 'reports' / f'color_map_seq{args.seq:02d}.npz'
    np.savez(out_npz, points=P, colors=C, traj=poses[:n, :3, 3])
    print('  saved', out_npz.name)

    _render(args, P, C, poses[:n, :3, 3])
    return 0


def _render(args, P, C, traj):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C = np.clip(C ** 0.7 * 1.25, 0, 1)                           # 提亮(gamma)：KITTI 图偏暗，仅显示用
    fig = plt.figure(figsize=(18, 8)); fig.patch.set_facecolor('#0b0b12')
    axb = fig.add_subplot(1, 2, 1); axb.set_facecolor('#0b0b12')
    axb.scatter(P[:, 0], P[:, 1], c=C, s=0.25, linewidths=0, rasterized=True)
    axb.plot(traj[:, 0], traj[:, 1], '-', color='#ffe14d', lw=1.4, label='SLAM trajectory')
    axb.set_aspect('equal'); axb.legend(labelcolor='w', loc='upper right')
    axb.set_title('Camera-colorized LiDAR map (BEV, metric)', color='w')
    axb.set_xlabel('x [m]'); axb.set_ylabel('y [m]')
    ax3 = fig.add_subplot(1, 2, 2, projection='3d'); ax3.set_facecolor('#0b0b12')
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.set_pane_color((0.04, 0.04, 0.07, 1.0)); pane.label.set_color('w')
        pane.set_tick_params(colors='#888')
    sub = np.random.default_rng(0).choice(len(P), min(300000, len(P)), replace=False)
    ax3.scatter(P[sub, 0], P[sub, 1], P[sub, 2], c=C[sub], s=0.4, linewidths=0, depthshade=False)
    ax3.set_title('Camera×LiDAR true-color dense map (3D)', color='w')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
    ax3.view_init(elev=40, azim=-60)
    axb.tick_params(colors='w'); [sp.set_color('w') for sp in axb.spines.values()]
    fig.suptitle(f'KITTI seq{args.seq:02d}: camera RGB projected onto LiDAR via '
                 f'P2·Tr calibration → true-color metric map', color='w')
    out = args.out or ROOT / 'reports' / f'color_map_seq{args.seq:02d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor='#0b0b12')
    print('  saved', out)


if __name__ == '__main__':
    raise SystemExit(main())
