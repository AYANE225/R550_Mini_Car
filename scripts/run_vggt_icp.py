"""VGGT 深度约束进 ICP(点级深耦合):把 VGGT 稠密点补进稀疏 LiDAR 扫描, 看配准精度能否救回。

动机(诚实): 干净 64 线 KITTI 上 LiDAR ICP 已亚分米, VGGT 点(median 0.46m 噪)只会添乱——
和位姿图那步结论一致。真正可能见价值的是 **LiDAR 被抽稀**(模拟更便宜/更稀的雷达):稀疏
LiDAR 配准退化, 用 VGGT 逐帧稠密点(按窗口尺度转 velo 系)补进去, 测帧间相对位姿精度能否恢复。

frame-to-frame point-to-plane ICP, 与 GT 相对位姿对比。三种源: 满 LiDAR / 抽稀 LiDAR /
抽稀 LiDAR + VGGT 稠密点。仅调公开 VGGT 权重。
用法: python scripts/run_vggt_icp.py --seq 0 --start 100 --count 32 --frac 0.03
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VGGT_SRC = os.environ.get('VGGT_SRC', '/media/4T/cst/PROJECT/Compare_SLAM/VGGT-Long/base_models')
VGGT_WEIGHTS = os.environ.get('VGGT_WEIGHTS', '/media/4T/cst/model/VGGT.pt')
sys.path.insert(0, VGGT_SRC)

import open3d as o3d
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from kitti_slam import kitti_io as io

reg = o3d.pipelines.registration


def kitti_image(seq, frame):
    return (Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked/dataset/sequences')
            / f'{int(seq):02d}' / 'image_2' / f'{int(frame):06d}.png')


def umeyama_scale(A, B):
    """只取尺度 s 使 B ≈ s·R·A + t（含尺度 Umeyama）。A,B:(N,3)。"""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, D, Vt = np.linalg.svd((Ac.T @ Bc) / len(A))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return float((D * np.diag(S)).sum() / ((Ac ** 2).sum() / len(A)))


def pose_err(T_est, T_gt):
    """返回 (旋转误差 deg, 平移误差 m)。"""
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return ang, float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def make_pcd(pts, voxel=0.0, normals=False):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts[:, :3]))
    if voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    if normals:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    return pc


def icp_p2l(src, tgt, init=np.eye(4), max_corr=1.5):
    """point-to-plane ICP, 返回把 src 对到 tgt 的 T（= T_tgt<-src）。tgt 需已有法向。"""
    res = reg.registration_icp(
        src, tgt, max_corr, init, reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=40))
    return res.transformation.copy()


def vggt_velo_points(world_pts, conf, extr_k, s, Tr_inv, conf_pct=50):
    """把 VGGT 世界系稠密点转到第 k 帧 velo 系(米制): world→cam_k→×尺度→velo。"""
    X = world_pts.reshape(-1, 3)
    C = conf.reshape(-1)
    keep = C >= np.percentile(C, conf_pct)
    X = X[keep]
    Xc = (extr_k[:3, :3] @ X.T + extr_k[:3, 3:4]).T * s      # 米制 cam_k 坐标
    Xv = (Tr_inv[:3, :3] @ Xc.T + Tr_inv[:3, 3:4]).T          # velo_k 系
    return Xv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=100)
    ap.add_argument('--count', type=int, default=32)
    ap.add_argument('--fracs', type=str, default='0.03,0.01,0.005,0.002',
                    help='LiDAR 抽稀保留比例扫描(模拟从稠密到极稀的雷达)')
    ap.add_argument('--conf-pct', type=float, default=50.0)
    ap.add_argument('--vggt-voxel', type=float, default=0.3, help='VGGT 点体素下采样(米)')
    ap.add_argument('--out', type=str, default='reports/vggt_icp_seq{seq:02d}.png')
    a = ap.parse_args(argv)
    seq, start, count = a.seq, a.start, a.count
    frames = list(range(start, start + count))
    rng = np.random.default_rng(0)

    slam = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    calib = io.read_calib(seq)
    Tr, Tr_inv = calib['Tr'], calib['Tr_inv']
    gt = io.gt_poses_velodyne(seq)
    cam0_world = (slam['odom'] @ Tr_inv)[:, :3, 3]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'loading VGGT ({device}) ...')
    model = VGGT()
    model.load_state_dict(torch.load(VGGT_WEIGHTS, map_location='cpu', weights_only=False), strict=False)
    model = model.to(device).eval()

    paths = [str(kitti_image(seq, f)) for f in frames]
    images = load_and_preprocess_images(paths).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        pred = model(images)
    extr, _ = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
    extr = extr[0].float().cpu().numpy()
    E = np.repeat(np.eye(4)[None], len(extr), axis=0)
    E[:, :3, :] = extr
    wp = pred['world_points'][0].float().cpu().numpy()          # [S,H,W,3]
    cf = pred['world_points_conf'][0].float().cpu().numpy()     # [S,H,W]
    cam_v = np.array([-e[:3, :3].T @ e[:3, 3] for e in E])
    s = umeyama_scale(cam_v, cam0_world[frames])
    print(f'window scale = {s:.2f}, VGGT dense pts/frame ~ {(cf[0] >= np.percentile(cf[0], a.conf_pct)).sum()}')

    lidar = [io.read_velodyne(seq, f)[:, :3] for f in frames]
    vpts = []
    for k in range(count):
        vp = vggt_velo_points(wp[k], cf[k], E[k], s, Tr_inv, a.conf_pct)
        if a.vggt_voxel > 0:                                    # 预先体素下采样 VGGT 稠密点
            vp = np.asarray(make_pcd(vp, voxel=a.vggt_voxel).points)
        vpts.append(vp)

    agg_full = []
    T_gts = []
    for k in range(count - 1):
        T_gt = np.linalg.inv(gt[frames[k + 1]]) @ gt[frames[k]]      # T_(velo_{k+1}<-velo_k)
        T_gts.append(T_gt)
        tgt = make_pcd(lidar[k + 1], normals=True)
        agg_full.append(pose_err(icp_p2l(make_pcd(lidar[k]), tgt), T_gt))
    full = np.array(agg_full).mean(0)

    fracs = [float(x) for x in a.fracs.split(',')]
    npts, sparse_e, fused_e = [], [], []
    for frac in fracs:
        sp, fu = [], []
        for k in range(count - 1):
            i_s = rng.choice(len(lidar[k]), max(120, int(len(lidar[k]) * frac)), replace=False)
            i_t = rng.choice(len(lidar[k + 1]), max(120, int(len(lidar[k + 1]) * frac)), replace=False)
            tgt_s = make_pcd(lidar[k + 1][i_t], normals=True)
            sp.append(pose_err(icp_p2l(make_pcd(lidar[k][i_s]), tgt_s), T_gts[k]))
            tgt_f = make_pcd(np.vstack([lidar[k + 1][i_t], vpts[k + 1]]), normals=True)
            src_f = make_pcd(np.vstack([lidar[k][i_s], vpts[k]]))
            fu.append(pose_err(icp_p2l(src_f, tgt_f), T_gts[k]))
        npts.append(int(len(lidar[0]) * frac))
        sparse_e.append(np.array(sp).mean(0))
        fused_e.append(np.array(fu).mean(0))
    sparse_e, fused_e = np.array(sparse_e), np.array(fused_e)

    print(f'\n=== seq{seq:02d} frame-to-frame point-to-plane ICP, {count - 1} pairs ===')
    print(f'  full LiDAR (~{len(lidar[0])} pts):  rot {full[0]:.3f} deg   trans {full[1]:.3f} m')
    print(f'  {"LiDAR pts":>10s} | {"sparse rot":>10s} {"+VGGT rot":>10s} | {"sparse trans":>12s} {"+VGGT trans":>11s}')
    for i, n in enumerate(npts):
        print(f'  {n:10d} | {sparse_e[i,0]:10.3f} {fused_e[i,0]:10.3f} | '
              f'{sparse_e[i,1]:12.3f} {fused_e[i,1]:11.3f}')

    _render(a.out.format(seq=seq), seq, npts, sparse_e, fused_e, full, len(lidar[0]))
    return 0


def _render(out, seq, npts, sparse_e, fused_e, full, n_full):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from kitti_slam import plotstyle as ps

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.patch.set_facecolor(ps.BG)
    for ax, col, ylab, ttl in [(axes[0], 0, 'rotation error (deg)', 'Frame-to-frame rotation error'),
                               (axes[1], 1, 'translation error (m)', 'Frame-to-frame translation error')]:
        ax.plot(npts, sparse_e[:, col], '-o', color=ps.MUTED, lw=2, label='sparse LiDAR only')
        ax.plot(npts, fused_e[:, col], '-o', color=ps.EST, lw=2.2, label='sparse LiDAR + VGGT points')
        ax.axhline(full[col], color=ps.GT, ls='--', lw=1.6, label=f'full LiDAR (~{n_full // 1000}k pts)')
        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('LiDAR points per scan (sparser →)')
        ax.set_ylabel(ylab)
        ax.set_title(ttl, color=ps.FG, fontsize=12)
        ps.style_ax(ax)
        ps.style_legend(ax.legend(loc='upper left', fontsize=9))
    fig.suptitle(f'VGGT depth into ICP: dense visual points rescue registration when LiDAR is starved  ·  seq{seq:02d}',
                 color=ps.FG, fontsize=13)
    ps.savefig(fig, ROOT / out)
    print(f'  saved {out}')


if __name__ == '__main__':
    raise SystemExit(main())

