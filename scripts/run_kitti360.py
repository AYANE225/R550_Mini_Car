"""在 KITTI-360 (HDL-64E 城区大场景) 上跑本工程 LiDAR SLAM：里程计 → 回环 → 位姿图。

KITTI-360 单条 drive 长达数公里、真值位姿稀疏(非每帧)。策略：里程计在**连续 velodyne 帧**
上跑(匀速先验要稠密输入)，ATE/RPE 按**绝对帧号**只在有真值的帧上对齐评测(SE(3) 对齐，坐标系无关)。
参数与 KITTI(HDL-64E) 完全一致：voxel=0.5、回环 radius=20/min_gap=150/stride=5，零重调。

用法: python scripts/run_kitti360.py --drive 0            # 缺省自动选最长稠密真值段
      python scripts/run_kitti360.py --drive 0 --start 1125 --count 3000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti360_io as io
from kitti_slam import loop as loopmod
from kitti_slam import plotstyle as ps
from kitti_slam import posegraph as pg
from kitti_slam.metrics import ate, kitti_rpe, path_length
from kitti_slam.odometry import LidarOdometry


def pick_window(drive, start, count):
    """确定跑的连续帧窗口 [start, start+count)：start<0 时自动取最长稠密真值段。"""
    fr, _ = io.gt_poses_velodyne(drive)
    ws, wl = io.longest_dense_window(fr)
    s = ws if start < 0 else start
    if count > 0:
        c = count
    elif start < 0:
        c = wl
    else:
        c = io.num_velodyne(drive) - s
    return int(s), int(min(c, io.num_velodyne(drive) - s))


def run_drive(drive, start=-1, count=-1, voxel=0.5, do_loop=True):
    start, count = pick_window(drive, start, count)
    fr, gp = io.gt_poses_velodyne(drive)
    gt_dict = {int(f): T for f, T in zip(fr, gp)}
    load = lambda i: io.read_velodyne(drive, start + i)

    cache = ROOT / 'reports' / f'k360_odom_d{drive}_{start}_{count}_{voxel}.npy'
    if cache.exists():
        poses = np.load(cache)
        print(f'drive{drive:04d}: cached odometry ({len(poses)} frames from {start})')
    else:
        print(f'drive{drive:04d}: odometry {count} frames from {start} ...', flush=True)
        t0 = time.perf_counter()
        poses = LidarOdometry(voxel=voxel).run(load, count, progress=max(1, count // 6))
        print(f'  odometry done {time.perf_counter() - t0:.0f}s')
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, poses)

    loops = loopmod.find_loops(load, poses, radius=20.0, min_gap=150, stride=5) if do_loop else []
    opt = pg.optimize(pg.build(poses, loops)) if loops else poses

    # 按绝对帧号取有真值的帧做评测
    idx = [i for i in range(count) if (start + i) in gt_dict]
    gt_m = np.array([gt_dict[start + i] for i in idx])
    m0, m1 = ate(poses[idx], gt_m), ate(opt[idx], gt_m)
    rpe = kitti_rpe(opt[idx], gt_m)
    return dict(drive=drive, start=start, count=count, n_gt=len(idx),
                poses=poses, opt=opt, loops=loops, load=load,
                m0=m0, m1=m1, rpe=rpe, len=path_length(poses))


def render(r, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor(ps.BG)

    # 轨迹：真值 vs 估计(回环后优先)，KITTI-360 世界系 z 朝上 → 俯视 (x,y)
    a0 = ax[0]; a0.set_facecolor(ps.BG)
    m = r['m1'] if r['loops'] else r['m0']
    g, a = m['gt_xyz'], m['aligned']
    a0.plot(g[:, 0], g[:, 1], '-', color=ps.GT, lw=2.4, label='KITTI-360 GT (cam0_to_world)')
    a0.plot(a[:, 0], a[:, 1], '--', color=ps.EST, lw=1.6, label='our LiDAR SLAM')
    a0.plot(g[0, 0], g[0, 1], 'o', color=ps.START, ms=11)
    a0.set_aspect('equal')
    a0.set_title(f"trajectory  ATE={m['ate_rmse_m']:.2f} m  ·  "
                 f"rel {r['rpe']['trans_err_pct']:.2f}%  ({r['n_gt']} GT frames)", color=ps.FG)
    a0.set_xlabel('x [m]', color=ps.FG); a0.set_ylabel('y [m]', color=ps.FG)
    ps.style_ax(a0); ps.style_legend(a0.legend())

    # 累积点云 BEV：估计位姿把每帧点转世界系，按高度着色
    a1 = ax[1]; a1.set_facecolor('#05050a')
    step = max(1, r['count'] // 140)
    P = []
    for i in range(0, r['count'], step):
        pts = r['load'](i)[:, :3]
        rng = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[(rng > 3) & (rng < 60)][::3]
        T = r['poses'][i]
        P.append((pts @ T[:3, :3].T) + T[:3, 3])
    P = np.vstack(P); P = P[np.argsort(P[:, 2])]
    a1.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap='turbo', s=0.2, alpha=0.5,
               vmin=np.percentile(P[:, 2], 2), vmax=np.percentile(P[:, 2], 98))
    tr = r['poses'][:, :3, 3]
    a1.plot(tr[:, 0], tr[:, 1], '-', color='w', lw=1.2)
    a1.set_aspect('equal'); a1.axis('off')
    a1.set_title(f"accumulated cloud ({len(P)//1000}k pts, odometry-stitched)", color=ps.FG)

    fig.suptitle(f"KITTI-360 drive_{r['drive']:04d}  ·  Velodyne HDL-64E  ·  "
                 f"{r['count']} frames from #{r['start']} ({r['len']:.0f} m)  "
                 f"·  our KITTI-built SLAM stack, zero retuning", color=ps.FG, fontsize=13)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ps.savefig(fig, out)
    print('  saved', out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--drive', type=int, default=0)
    ap.add_argument('--start', type=int, default=-1, help='<0 自动取最长稠密真值段起点')
    ap.add_argument('--count', type=int, default=-1, help='<0 自动取该段长度')
    ap.add_argument('--voxel', type=float, default=0.5)
    ap.add_argument('--no-loop', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)

    r = run_drive(a.drive, a.start, a.count, a.voxel, do_loop=not a.no_loop)
    print(f"\nKITTI-360 drive_{a.drive:04d}  {r['count']} frames from #{r['start']}  "
          f"({r['len']:.0f} m, {r['n_gt']} GT frames, {len(r['loops'])} loops)")
    print(f"  ATE  odom {r['m0']['ate_rmse_m']:.2f} m  ->  opt {r['m1']['ate_rmse_m']:.2f} m")
    print(f"  RPE  trans {r['rpe']['trans_err_pct']:.2f}%  rot {r['rpe']['rot_err_deg_per_m']:.4f} deg/m "
          f"({r['rpe']['n_segments']} seg)")
    out = a.out or str(ROOT / 'reports' / f'kitti360_d{a.drive:04d}.png')
    render(r, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
