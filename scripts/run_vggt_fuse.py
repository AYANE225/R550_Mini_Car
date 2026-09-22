"""VGGT 深度耦合进 LiDAR SLAM：把 VGGT 前馈稠密重建按位姿配准、定尺度，融进 SLAM 世界系。

不是"仅渲染"了——这里做真正的耦合：
  · VGGT 每个窗口输出稠密带色点云 + 相机位姿，但**单目、尺度任意、各窗口独立**；
  · LiDAR SLAM 提供**全局一致的度量位姿**（slam_seqSS.npz 的 opt 位姿，含回环/PGO）；
  · 对每个窗口，用 VGGT 相机中心 vs SLAM 相机中心做 **Sim(3)（含尺度）Umeyama 对齐**，
    把该窗口的稠密点旋到 SLAM 世界系并赋予真实米制尺度；
  · 多窗口累积 + 体素下采样 → 一张**全局一致、度量正确、稠密带色**的 VGGT 地图，
    与 LiDAR 轨迹/地图同系（叠轨迹即可验证配准）。

即：LiDAR 给尺度与全局约束，VGGT 给稠密光度几何，两者经位姿耦合。**仅调公开 VGGT 权重，
绝不导入任何非公开的 LiDAR-VGGT-SLAM 融合工程**（PYTHONPATH 指公开 VGGT-Long/base_models）。
用法：python scripts/run_vggt_fuse.py --seq 0 --end 360 --count 16 --step 2
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

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from kitti_slam import kitti_io as io


def kitti_image(seq, frame):
    return (Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked/dataset/sequences')
            / f'{int(seq):02d}' / 'image_2' / f'{int(frame):06d}.png')


def umeyama_sim3(A, B):
    """求 s,R,t 使 B ≈ s·R·A + t（含尺度的相似变换，Umeyama 1991）。A,B: (N,3)。"""
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    H = (Ac.T @ Bc) / len(A)
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    var_a = (Ac ** 2).sum() / len(A)
    s = float((D * np.diag(S)).sum() / var_a)
    t = mu_b - s * R @ mu_a
    return s, R, t


def voxel_downsample(P, C, voxel):
    """体素栅格下采样：每格取首个点。返回下采样后的 P,C。"""
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[idx], C[idx]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=360, help='覆盖到的帧（含多个窗口）')
    ap.add_argument('--count', type=int, default=16, help='每窗口送入 VGGT 的帧数')
    ap.add_argument('--step', type=int, default=2, help='窗口内帧间隔')
    ap.add_argument('--conf-pct', type=float, default=55.0)
    ap.add_argument('--voxel', type=float, default=0.3, help='累积地图体素下采样(米)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)

    npz = ROOT / 'reports' / f'slam_seq{args.seq:02d}.npz'
    if not npz.exists():
        print(f'需要先跑 run_slam.py 生成 {npz.name}'); return 1
    T_wv = np.load(npz)['opt']                                  # T_world_velo（含回环/PGO）
    Tr_inv = io.read_calib(args.seq)['Tr_inv']
    T_wc = T_wv @ Tr_inv                                        # T_world_cam0：相机在世界系
    cam_w = T_wc[:, :3, 3]                                      # SLAM 相机中心轨迹

    device = 'cuda'
    print(f'loading VGGT ({VGGT_WEIGHTS}) ...')
    model = VGGT()
    model.load_state_dict(torch.load(VGGT_WEIGHTS, map_location='cpu', weights_only=True), strict=False)
    model.eval().to(device).requires_grad_(False)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    span = args.count * args.step
    win_starts = list(range(args.start, min(args.end, len(T_wv)) - span + 1, span))
    print(f'seq{args.seq:02d}: {len(win_starts)} windows × {args.count} frames, '
          f'covering f{args.start}..{win_starts[-1] + span}')

    all_P, all_C = [], []
    for w0 in win_starts:
        frames = list(range(w0, w0 + span, args.step))
        paths = [str(kitti_image(args.seq, f)) for f in frames]
        images = load_and_preprocess_images(paths).to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            pred = model(images)
        wp = pred['world_points'][0].float().cpu().numpy()          # [S,H,W,3]
        conf = pred['world_points_conf'][0].float().cpu().numpy()   # [S,H,W]
        imgs = pred['images'][0].float().cpu().numpy().transpose(0, 2, 3, 1)
        extr, _ = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
        extr = extr[0].float().cpu().numpy()                        # [S,3,4] world->cam
        # VGGT 相机中心（VGGT 世界系）= -R^T t
        cam_v = np.array([-e[:3, :3].T @ e[:3, 3] for e in extr])

        s, R, t = umeyama_sim3(cam_v, cam_w[frames])                # VGGT→SLAM 世界系
        P = wp.reshape(-1, 3); C = imgs.reshape(-1, 3).clip(0, 1); K = conf.reshape(-1)
        keep = K >= np.percentile(K, args.conf_pct)
        P, C = P[keep], C[keep]
        Pw = (s * (P @ R.T)) + t                                    # 融进 SLAM 世界系(米制)
        Pw, C = voxel_downsample(Pw, C, args.voxel)
        all_P.append(Pw); all_C.append(C)
        fit = np.linalg.norm((s * (cam_v @ R.T) + t) - cam_w[frames], axis=1).mean()
        print(f'  win f{w0:04d}: scale={s:.2f}, cam-align RMSE={fit:.2f}m, '
              f'+{len(Pw):,} pts', flush=True)

    P = np.vstack(all_P); C = np.vstack(all_C)
    P, C = voxel_downsample(P, C, args.voxel)
    print(f'fused VGGT map: {len(P):,} points (voxel {args.voxel}m)')
    out_npz = ROOT / 'reports' / f'vggt_fused_seq{args.seq:02d}.npz'
    np.savez(out_npz, points=P, colors=C, cam_w=cam_w[:win_starts[-1] + span])
    print('  saved', out_npz.name)

    _render(args, P, C, cam_w[args.start:win_starts[-1] + span])
    return 0


def _render(args, P, C, traj):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # SLAM velo 世界系：z 竖直。俯视取 (x, y)。
    fig = plt.figure(figsize=(18, 8)); fig.patch.set_facecolor('#0b0b12')
    axb = fig.add_subplot(1, 2, 1); axb.set_facecolor('#0b0b12')
    axb.scatter(P[:, 0], P[:, 1], c=C, s=0.35, linewidths=0, rasterized=True)
    axb.plot(traj[:, 0], traj[:, 1], '-', color='#ffe14d', lw=1.6, label='SLAM trajectory')
    axb.set_aspect('equal'); axb.legend(labelcolor='w', loc='upper right')
    axb.set_title('VGGT dense map fused into SLAM world (BEV, metric)', color='w')
    axb.set_xlabel('x [m]'); axb.set_ylabel('y [m]')
    ax3 = fig.add_subplot(1, 2, 2, projection='3d'); ax3.set_facecolor('#0b0b12')
    for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        pane.set_pane_color((0.04, 0.04, 0.07, 1.0)); pane.label.set_color('w')
        pane.set_tick_params(colors='#888')
    sub = np.random.default_rng(0).choice(len(P), min(200000, len(P)), replace=False)
    ax3.scatter(P[sub, 0], P[sub, 1], P[sub, 2], c=C[sub], s=0.4, linewidths=0, depthshade=False)
    ax3.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color='#ffe14d', lw=1.4)
    ax3.set_title('VGGT×LiDAR dense reconstruction (3D, world frame)', color='w')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')
    ax3.view_init(elev=38, azim=-60)
    axb.tick_params(colors='w'); [sp.set_color('w') for sp in axb.spines.values()]
    fig.suptitle(f'KITTI seq{args.seq:02d}: VGGT feed-forward reconstruction deeply coupled '
                 f'to LiDAR SLAM (pose-anchored Sim(3) fusion)', color='w')
    out = args.out or ROOT / 'reports' / f'vggt_fused_seq{args.seq:02d}.png'
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor='#0b0b12')
    print('  saved', out)


if __name__ == '__main__':
    raise SystemExit(main())
