"""在 nuScenes(Motional，Velodyne HDL-32E)上跑本工程的 LiDAR 里程计 → 回环 → 位姿图，
证明 SLAM 栈不是 KITTI 专用：换个厂商配置的激光（32 线 vs KITTI 64 线、20Hz）照样稳。

真值用 nuScenes 自带的全局 ego_pose（地图定位），ATE 用 SE(3) 对齐（坐标系无关）。
用法：python scripts/run_nuscenes.py [--scene 0] [--all] [--out reports/nuscenes.png]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import loop as loopmod
from kitti_slam import posegraph as pg
from kitti_slam.metrics import ate, kitti_rpe, path_length
from kitti_slam.nuscenes_io import NuScenesLidar
from kitti_slam.odometry import LidarOdometry


def run_scene(nu, scene, voxel, do_loop=True):
    recs = nu.scene_lidar_frames(scene)
    n = len(recs)
    load = lambda i: nu.load_points(recs[i])
    gt = np.array([nu.lidar_global_pose(r) for r in recs])

    cache = ROOT / 'reports' / f'nuodom_scene{scene}_{n}_{voxel}.npy'
    if cache.exists():
        poses = np.load(cache)
    else:
        odom = LidarOdometry(voxel=voxel)
        poses = odom.run(load, n, progress=max(1, n // 5))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, poses)

    loops = []
    if do_loop:
        loops = loopmod.find_loops(load, poses, radius=15.0, min_gap=80, stride=3)
    opt = pg.optimize(pg.build(poses, loops)) if loops else poses

    m0 = ate(poses, gt)
    m1 = ate(opt, gt)
    rpe = kitti_rpe(poses, gt, lengths=(20, 40, 60, 80, 100), step=5)
    return dict(recs=recs, n=n, gt=gt, poses=poses, opt=opt, loops=loops,
                m0=m0, m1=m1, rpe=rpe, load=load,
                name=nu.scenes()[scene][0] if isinstance(scene, int) else scene)


def render(nu, r, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('#0b0b12')
    # 轨迹（nuScenes 全局系 z 朝上 → 俯视取 x-y）
    a0 = ax[0]
    m = r['m1'] if r['loops'] else r['m0']
    g, a = m['gt_xyz'], m['aligned']
    a0.set_facecolor('#0b0b12')
    a0.plot(g[:, 0], g[:, 1], '-', color='#5ad1ff', lw=2.4, label='nuScenes ego (map GT)')
    a0.plot(a[:, 0], a[:, 1], '--', color='#ff7043', lw=1.6, label='our LiDAR odometry')
    a0.plot(g[0, 0], g[0, 1], 'o', color='#7CFC00', ms=11)
    a0.set_aspect('equal'); a0.grid(alpha=0.2, color='#445')
    a0.set_title(f"trajectory  ATE={m['ate_rmse_m']:.2f} m  ·  "
                 f"rel {r['rpe']['trans_err_pct']:.2f}%", color='w')
    a0.set_xlabel('x [m]', color='#aab'); a0.set_ylabel('y [m]', color='#aab')
    a0.tick_params(colors='#889'); a0.legend(facecolor='#16324f', edgecolor='#39d0e0', labelcolor='w')

    # 累积点云 BEV（用估计位姿把每帧点转世界系，按高度着色）
    a1 = ax[1]; a1.set_facecolor('#05050a')
    step = max(1, r['n'] // 120)
    P = []
    for i in range(0, r['n'], step):
        pts = r['load'](i)[:, :3]
        rng = np.linalg.norm(pts[:, :2], axis=1)
        pts = pts[(rng > 3) & (rng < 60)][::3]
        T = r['poses'][i]
        P.append((pts @ T[:3, :3].T) + T[:3, 3])
    P = np.vstack(P)
    order = np.argsort(P[:, 2])
    P = P[order]
    a1.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap='turbo', s=0.2, alpha=0.5,
               vmin=np.percentile(P[:, 2], 2), vmax=np.percentile(P[:, 2], 98))
    tr = r['poses'][:, :3, 3]
    a1.plot(tr[:, 0], tr[:, 1], '-', color='w', lw=1.4)
    a1.set_aspect('equal'); a1.axis('off')
    a1.set_title(f"accumulated cloud ({len(P)//1000}k pts, odometry-stitched)", color='w')

    fig.suptitle(f"nuScenes {r['name']}  ·  Velodyne HDL-32E  ·  {r['n']} frames @ 20 Hz  "
                 f"·  our KITTI-built SLAM stack, zero retuning",
                 color='w', fontsize=13)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print('  saved', out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', type=int, default=0)
    ap.add_argument('--voxel', type=float, default=0.4)
    ap.add_argument('--all', action='store_true', help='跑全部 10 个场景出汇总表')
    ap.add_argument('--out', default=str(ROOT / 'reports' / 'nuscenes_scene0.png'))
    args = ap.parse_args(argv)

    nu = NuScenesLidar()
    scenes = range(len(nu.scenes())) if args.all else [args.scene]

    rows = []
    for s in scenes:
        t0 = time.perf_counter()
        r = run_scene(nu, s, args.voxel)
        dt = time.perf_counter() - t0
        rows.append((r['name'], r['n'], path_length(r['poses']),
                     r['m0']['ate_rmse_m'], r['m1']['ate_rmse_m'],
                     r['rpe']['trans_err_pct'], len(r['loops'])))
        print(f"{r['name']}: {r['n']}f {path_length(r['poses']):.0f}m  "
              f"ATE odom {r['m0']['ate_rmse_m']:.2f} -> opt {r['m1']['ate_rmse_m']:.2f} m  "
              f"rel {r['rpe']['trans_err_pct']:.2f}%  {len(r['loops'])} loops  ({dt:.0f}s)")
        if s == args.scene:
            render(nu, r, args.out)

    if args.all:
        rel = np.array([x[5] for x in rows])
        ate0 = np.array([x[3] for x in rows])
        moving = np.isfinite(rel)                     # 排除近乎静止的场景(rel 无定义)
        print('\n=== nuScenes mini · 10 scenes (Velodyne HDL-32E) ===')
        print(f"  mean odometry ATE {ate0.mean():.2f} m (all 10)  ·  "
              f"mean relative trans {np.nanmean(rel):.2f}% ({int(moving.sum())} moving scenes)")
        md = ROOT / 'reports' / 'nuscenes_multi.md'
        with open(md, 'w') as f:
            f.write('| scene | frames | path (m) | ATE odom (m) | ATE opt (m) | rel trans % | loops |\n')
            f.write('| --- | --- | --- | --- | --- | --- | --- |\n')
            for nm, n, pl, a0, a1, rp, nl in rows:
                rps = f'{rp:.2f}' if np.isfinite(rp) else '— (static)'
                f.write(f'| {nm} | {n} | {pl:.0f} | {a0:.2f} | {a1:.2f} | {rps} | {nl} |\n')
            f.write(f'| **mean** | | | **{ate0.mean():.2f}** | | **{np.nanmean(rel):.2f}** | |\n')
        print('  saved', md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
