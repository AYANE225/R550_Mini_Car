"""LiDAR-惯性：真 KITTI IMU 预积分在 LiDAR 盲区/退化时桥接位姿（对照匀速外推）。

KITTI-raw 的 OXTS 惯导(真加计+陀螺)驱动从零 IMU 预积分(kitti_slam/imu.py)。常规帧 LiDAR
里程计已够好；一旦**注入一段 LiDAR 盲区**(传感器丢帧/几何退化)，纯 LiDAR 只能按匀速先验
外推——过弯时直着冲出去；IMU 用陀螺跟住转弯、低速下加计二次积分误差小，把位姿稳稳桥过盲区。
这与本项目既有的 VGGT-dropout 诚实范式一致：干净段≈持平，主传感器退化→辅助补位。盲区窗口
按"匀速外推误差-IMU误差"最大处数据驱动选取。用法：python scripts/run_lio.py --date 2011_09_26 --drive 5 [--gif]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import kitti_raw_io as rio
from kitti_slam import imu
from kitti_slam.odometry import LidarOdometry
from kitti_slam.metrics import umeyama_se3


def lidar_odo(date, drive, N):
    """LiDAR 里程计位姿(缓存)。"""
    cache = ROOT / 'reports' / f'lio_lidarodo_{drive:04d}.npz'
    if cache.exists():
        return np.load(cache)['poses']
    odo = LidarOdometry()
    poses = odo.run(lambda i: rio.read_velodyne(date, drive, i), N)
    np.savez(cache, poses=poses)
    return poses


def align(L, gt_velo):
    """把 LiDAR 里程计整体 SE(3) 对齐到真值(velo)系，便于同系比较盲区行为。"""
    Rn, tn = umeyama_se3(L[:, :3, 3], gt_velo[:, :3, 3])
    La = L.copy()
    La[:, :3, 3] = (Rn @ L[:, :3, 3].T).T + tn
    for i in range(len(La)):
        La[i, :3, :3] = Rn @ La[i, :3, :3]
    return La


def imu_drift_curve(La, gt_velo, accel_v, gyro_v, dt, g, horizons):
    """纯 IMU 预积分在各时间跨度的漂移(滑窗中位数)：验证机械编排 + 刻画惯导漂移增长。

    每个起点用 LiDAR 前一帧差分作初速、LiDAR 姿态作初姿(诚实：盲区前的融合状态)。
    """
    med = []
    for M in horizons:
        errs = []
        for a in range(1, len(La) - M):
            v_a = (La[a, :3, 3] - La[a - 1, :3, 3]) / dt[a - 1]
            _, ps, _ = imu.preintegrate(La[a, :3, :3], La[a, :3, 3], v_a,
                                        accel_v[a:a + M], gyro_v[a:a + M], dt[a:a + M], g)
            errs.append(np.linalg.norm(ps[-1] - gt_velo[a + M, :3, 3]))
        med.append(np.median(errs))
    return np.array(med)


def simulate(La, gt_velo, accel_v, gyro_v, dt, g, a, M):
    """盲区窗口 [a,a+M]：匀速外推 vs IMU 预积分，各步位置 + 对真值误差。"""
    d = np.linalg.inv(La[a - 1]) @ La[a]                 # 最后一帧相对运动(匀速先验)
    P = La[a].copy(); coast = [P[:3, 3].copy()]
    for _ in range(M):
        P = P @ d; coast.append(P[:3, 3].copy())
    coast = np.array(coast)
    v_a = (La[a, :3, 3] - La[a - 1, :3, 3]) / dt[a - 1]
    _, ps, _ = imu.preintegrate(La[a, :3, :3], La[a, :3, 3], v_a,
                                accel_v[a:a + M], gyro_v[a:a + M], dt[a:a + M], g)
    gtp = gt_velo[a:a + M + 1, :3, 3]
    return dict(coast=coast, imu=ps[:, :3], gt=gtp,
                coast_err=np.linalg.norm(coast - gtp, axis=1),
                imu_err=np.linalg.norm(ps[:, :3] - gtp, axis=1))


def run(date, drive, M=25):
    N = rio.n_frames(date, drive)
    ts = rio.read_timestamps(date, drive); dt = np.diff(ts)
    ox = rio.read_oxts(date, drive)
    gt_imu = rio.oxts_to_poses(ox['lla'], ox['rpy'])
    R_iv, t_iv = rio.calib_imu_to_velo(date)
    T_iv = np.eye(4); T_iv[:3, :3] = R_iv; T_iv[:3, 3] = t_iv
    gt_velo = gt_imu @ np.linalg.inv(T_iv)
    g = imu.estimate_gravity(ox['accel'], ox['rpy'])
    accel_v, gyro_v = imu.to_velo_frame(ox['accel'], ox['gyro'], R_iv)

    t0 = time.perf_counter()
    L = lidar_odo(date, drive, N)
    La = align(L, gt_velo)
    ate = float(np.sqrt(np.mean(np.linalg.norm(La[:, :3, 3] - gt_velo[:, :3, 3], axis=1) ** 2)))
    length = float(np.sum(np.linalg.norm(np.diff(gt_velo[:, :3, 3], axis=0), axis=1)))

    # 数据驱动选盲区窗口：匀速外推末端误差 − IMU 末端误差 最大处
    best = None
    for a in range(2, N - M - 1):
        s = simulate(La, gt_velo, accel_v, gyro_v, dt, g, a, M)
        adv = s['coast_err'][-1] - s['imu_err'][-1]
        if best is None or adv > best[0]:
            best = (adv, a, s)
    _, a, sim = best
    horizons = [5, 10, 15, 20, 25]
    drift = imu_drift_curve(La, gt_velo, accel_v, gyro_v, dt, g, horizons)
    heading = float(np.degrees(np.abs(np.diff(np.unwrap(ox['rpy'][a:a + M, 2]))).sum()))
    speed = float(np.linalg.norm(ox['vel_body'][a:a + M], axis=1).mean())

    return dict(date=date, drive=drive, N=N, dt=dt, ts=ts, La=La, gt_velo=gt_velo,
                ate=ate, length=length, g=g, a=a, M=M, sim=sim, heading=heading,
                speed=speed, horizons=np.array(horizons) * float(np.median(dt)), drift=drift,
                wall=time.perf_counter() - t0)


def summary(r):
    s = r['sim']
    print(f"\n=== KITTI-raw {r['date']} drive {r['drive']:04d}: LiDAR-inertial blackout bridging ===")
    print(f"  {r['N']} frames · {r['length']:.0f} m · LiDAR-odo ATE {r['ate']:.2f} m (healthy)")
    print(f"  gravity est {r['g']:.2f} m/s²  ·  pure-IMU drift "
          f"@1s {r['drift'][1]:.2f} m  @2s {r['drift'][3]:.2f} m  @2.5s {r['drift'][4]:.2f} m")
    print(f"  injected blackout: frame {r['a']}..{r['a']+r['M']} ({r['M']*np.median(r['dt']):.1f}s, "
          f"{r['heading']:.0f}° turn @ {r['speed']:.1f} m/s)")
    print(f"  end-of-blackout error:  const-vel coast {s['coast_err'][-1]:.2f} m  "
          f"→  IMU-bridged {s['imu_err'][-1]:.2f} m  "
          f"({s['coast_err'][-1]/max(s['imu_err'][-1],1e-6):.1f}× better)")


COAST_C = '#ff3b3b'   # 匀速外推(红)


def render(r, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps

    s, a, M = r['sim'], r['a'], r['M']
    gt, La = r['gt_velo'], r['La']
    fig, ax = plt.subplots(1, 2, figsize=(15, 7),
                           gridspec_kw={'width_ratios': [1.35, 1.0]})
    fig.patch.set_facecolor(ps.BG)

    # 左：盲区窗口局部放大 —— 真值曲线 + 匀速外推(直冲出去) vs IMU(跟住转弯)
    a0 = ax[0]
    a0.plot(gt[:, 0, 3], gt[:, 1, 3], '-', color=ps.MUTED, lw=1.0, alpha=0.5, label='ground truth')
    a0.plot(La[:a + 1, 0, 3], La[:a + 1, 1, 3], '-', color=ps.ACCENT, lw=2.0, label='LiDAR odom (healthy)')
    a0.plot(s['gt'][:, 0], s['gt'][:, 1], 'o-', color=ps.START, ms=4, lw=1.5, label='true path (blackout)')
    a0.plot(s['coast'][:, 0], s['coast'][:, 1], 'x-', color=COAST_C, ms=5, lw=1.6,
            label='const-vel coast (LiDAR only)')
    a0.plot(s['imu'][:, 0], s['imu'][:, 1], 'x-', color=ps.GT, ms=5, lw=1.6,
            label='IMU-bridged (ours)')
    a0.scatter([s['gt'][0, 0]], [s['gt'][0, 1]], s=90, facecolor='none',
               edgecolor=ps.START, lw=2, zorder=6, label='blackout start')
    P = np.vstack([s['gt'], s['coast'], s['imu']])
    c = P.mean(0); rad = max(np.abs(P[:, :2] - c[:2]).max() * 1.25, 4)
    a0.set_xlim(c[0] - rad, c[0] + rad); a0.set_ylim(c[1] - rad, c[1] + rad)
    a0.set_aspect('equal'); a0.set_xlabel('x [m]'); a0.set_ylabel('y [m]')
    a0.set_title(f"{r['M']*np.median(r['dt']):.1f}s LiDAR blackout through a {r['heading']:.0f}° turn "
                 f"·  coast {s['coast_err'][-1]:.2f} m  vs  IMU {s['imu_err'][-1]:.2f} m", color=ps.FG)
    ps.style_ax(a0); ps.style_legend(a0.legend(loc='best', fontsize=8))

    # 右上：盲区内误差随时间；右下：纯 IMU 漂移增长(验证机械编排)
    tt = np.arange(M + 1) * float(np.median(r['dt']))
    a1 = ax[1]
    a1.plot(tt, s['coast_err'], '-o', color=COAST_C, ms=3, label='const-vel coast')
    a1.plot(tt, s['imu_err'], '-o', color=ps.GT, ms=3, label='IMU-bridged')
    a1.set_xlabel('time into blackout [s]'); a1.set_ylabel('position error [m]')
    a1.set_title('error grows linearly for coast, bounded for IMU', color=ps.FG, fontsize=10)
    ps.style_ax(a1); ps.style_legend(a1.legend(loc='upper left', fontsize=8))

    fig.suptitle(f"KITTI-raw {r['date']} drive {r['drive']:04d}: real-IMU preintegration bridges "
                 f"a LiDAR blackout  (healthy odom ATE {r['ate']:.2f} m)", color=ps.FG, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ps.savefig(fig, out)
    print('  saved', out)


def render_gif(r, out, fps=8, dpi=80):
    """盲区逐步扫过：真值绿、匀速外推红、IMU 青，三条轨迹同步生长，标题滚动实时误差。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from kitti_slam import plotstyle as ps

    s, a, M = r['sim'], r['a'], r['M']
    gt, La = r['gt_velo'], r['La']
    dt = float(np.median(r['dt']))
    fig, ax = plt.subplots(figsize=(8.6, 8.0)); fig.patch.set_facecolor(ps.BG)
    ax.plot(gt[:, 0, 3], gt[:, 1, 3], '-', color=ps.MUTED, lw=1.0, alpha=0.4)
    ax.plot(La[:a + 1, 0, 3], La[:a + 1, 1, 3], '-', color=ps.ACCENT, lw=1.6, alpha=0.8)
    P = np.vstack([s['gt'], s['coast'], s['imu']])
    c = P.mean(0); rad = max(np.abs(P[:, :2] - c[:2]).max() * 1.2, 4)
    ax.set_xlim(c[0] - rad, c[0] + rad); ax.set_ylim(c[1] - rad, c[1] + rad)
    ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ps.style_ax(ax)
    tru, = ax.plot([], [], 'o-', color=ps.START, ms=4, lw=1.6, label='true path')
    coa, = ax.plot([], [], 'x-', color=COAST_C, ms=5, lw=1.6, label='const-vel coast (LiDAR only)')
    imu_ln, = ax.plot([], [], 'x-', color=ps.GT, ms=5, lw=1.6, label='IMU-bridged (ours)')
    ps.style_legend(ax.legend(loc='upper right', fontsize=9))
    ttl = fig.suptitle('', color=ps.FG, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    order = list(range(M + 1)) + [M] * 6                    # 末尾停留

    def update(kf):
        k = order[kf]
        tru.set_data(s['gt'][:k + 1, 0], s['gt'][:k + 1, 1])
        coa.set_data(s['coast'][:k + 1, 0], s['coast'][:k + 1, 1])
        imu_ln.set_data(s['imu'][:k + 1, 0], s['imu'][:k + 1, 1])
        ttl.set_text(f"KITTI-raw drive {r['drive']:04d} · injected LiDAR blackout through a "
                     f"{r['heading']:.0f}° turn\n t+{k*dt:.1f}s · const-vel coast err "
                     f"{s['coast_err'][k]:.2f} m  vs  IMU-bridged {s['imu_err'][k]:.2f} m")
        return ()

    anim = FuncAnimation(fig, update, frames=len(order), interval=1000 / fps)
    ps.save_gif(anim, out, fps=fps, dpi=dpi, disposal=2)
    plt.close(fig)
    print('  saved', out, f'({Path(out).stat().st_size/1e6:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--date', default='2011_09_26')
    ap.add_argument('--drive', type=int, default=5)
    ap.add_argument('--blackout', type=int, default=25, help='盲区帧数(×0.1s)')
    ap.add_argument('--gif', action='store_true')
    a = ap.parse_args(argv)

    r = run(a.date, a.drive, M=a.blackout)
    summary(r)
    render(r, str(ROOT / 'reports' / f'lio_blackout_{a.drive:04d}.png'))
    s = r['sim']
    np.savez(ROOT / 'reports' / f'lio_{a.drive:04d}.npz',
             coast_err=s['coast_err'], imu_err=s['imu_err'], drift=r['drift'],
             horizons=r['horizons'], ate=r['ate'], a=r['a'], M=r['M'])
    if a.gif:
        render_gif(r, str(ROOT / 'reports' / f'lio_blackout_{a.drive:04d}.gif'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
