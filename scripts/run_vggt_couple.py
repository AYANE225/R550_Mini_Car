"""VGGT × LiDAR 深耦合(联合后端)：把 VGGT 前馈视觉相对位姿作为因子融进 LiDAR 位姿图。

不同于 run_vggt_fuse.py(把 VGGT 稠密点按位姿渲染进世界系, VGGT 不影响轨迹), 这里 VGGT
真正进入估计: 每个滑窗内 VGGT 给出逐帧相对相机运动, 用 LiDAR 定尺度后转到 velodyne 系,
作为相对位姿边加入位姿图, 与 LiDAR 里程计边一起全局优化——VGGT 因此改变最终轨迹。

价值验证(诚实): 平滑 KITTI 上纯 LiDAR 里程计已很强, 融合在干净数据上 ≈ 打平; 真正体现
耦合价值的是**里程计中断**——模拟一段 LiDAR 里程计丢失(该段无里程计边、位姿冻结), LiDAR-only
轨迹从此错位断裂, 而 VGGT 视觉相对位姿因子把这段接回正确路径。

仅调公开 VGGT 权重(VGGT_SRC / VGGT_WEIGHTS), 绝不导入任何非公开融合工程。
用法: python scripts/run_vggt_couple.py --seq 0 --start 0 --end 300 --gap 150,190
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
from kitti_slam import posegraph as pg
from kitti_slam import plotstyle as ps
from kitti_slam.metrics import ate

reg = o3d.pipelines.registration


def kitti_image(seq, frame):
    return (Path('/media/4T/cst/DATASETS/SLAM/KITTI_Odometry/unpacked/dataset/sequences')
            / f'{int(seq):02d}' / 'image_2' / f'{int(frame):06d}.png')


def umeyama_sim3(A, B):
    """s,R,t 使 B ≈ s·R·A + t（含尺度）。A,B:(N,3)。"""
    mu_a, mu_b = A.mean(0), B.mean(0)
    Ac, Bc = A - mu_a, B - mu_b
    U, D, Vt = np.linalg.svd((Ac.T @ Bc) / len(A))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = float((D * np.diag(S)).sum() / ((Ac ** 2).sum() / len(A)))
    return s, R, mu_b - s * R @ mu_a


def load_vggt(device):
    model = VGGT()
    sd = torch.load(VGGT_WEIGHTS, map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def window_edges(model, seq, frames, cam0_world, Tr, Tr_inv, weight, device):
    """一个连续滑窗前馈 VGGT，返回该窗 velo 系逐帧相对位姿边 + 相机对齐 RMSE。

    frames: 连续全局帧号列表; cam0_world: 这些帧的 SLAM 相机中心(cam0 世界系, 用于定尺度)。
    返回 [(a, b, T_velo_b<-a, weight, uncertain=True)], a/b 为全局帧号。
    """
    paths = [str(kitti_image(seq, f)) for f in frames]
    images = load_and_preprocess_images(paths).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        pred = model(images)
    extr, _ = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
    extr = extr[0].float().cpu().numpy()                     # [S,3,4] world_vggt->cam
    E = np.repeat(np.eye(4)[None], len(extr), axis=0)
    E[:, :3, :] = extr
    cam_v = np.array([-e[:3, :3].T @ e[:3, 3] for e in E])   # VGGT 相机中心(任意尺度)
    s, R, t = umeyama_sim3(cam_v, cam0_world)                # 用 LiDAR 相机轨迹定尺度
    rmse = float(np.linalg.norm((s * (cam_v @ R.T) + t) - cam0_world, axis=1).mean())
    edges = []
    for k in range(len(frames) - 1):
        Tcam = E[k + 1] @ np.linalg.inv(E[k])                # T_(cam_{k+1}<-cam_k), VGGT 单位
        Tcam = Tcam.copy()
        Tcam[:3, 3] *= s                                     # 定尺度→米制
        Tvelo = Tr_inv @ Tcam @ Tr                           # 转 velo 系: T_(velo_{k+1}<-velo_k)
        edges.append((frames[k], frames[k + 1], Tvelo, weight, True))
    return edges, s, rmse


def build_graph(node_init, edge_poses, drop, extra):
    """node_init: 各节点初值(T_world_velo); edge_poses: 算里程计相对位姿的来源(干净里程计);
    drop: 丢弃里程计边的起点集合; extra: [(a,b,T_b<-a,w,uncertain)] 局部索引附加边(VGGT)。"""
    g = reg.PoseGraph()
    for p in node_init:
        g.nodes.append(reg.PoseGraphNode(p.copy()))
    I = np.eye(6)
    for a in range(len(node_init) - 1):
        if a in drop:
            continue
        T = np.linalg.inv(edge_poses[a + 1]) @ edge_poses[a]
        g.edges.append(reg.PoseGraphEdge(a, a + 1, T, I, uncertain=False))
    for (a, b, T, w, unc) in extra:
        g.edges.append(reg.PoseGraphEdge(a, b, T, I * w, uncertain=unc))
    return g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=300)
    ap.add_argument('--count', type=int, default=24, help='每个 VGGT 滑窗的连续帧数')
    ap.add_argument('--stride', type=int, default=12, help='滑窗步长(重叠保证逐帧覆盖)')
    ap.add_argument('--gap', type=str, default='150,190', help='模拟 LiDAR 里程计中断区间 g0,g1')
    ap.add_argument('--weight', type=float, default=0.3, help='VGGT 相对位姿边信息权重')
    ap.add_argument('--out', type=str, default='reports/vggt_couple_seq{seq:02d}.png')
    a = ap.parse_args(argv)

    seq, start, end = a.seq, a.start, a.end
    N = end - start
    slam = np.load(ROOT / 'reports' / f'slam_seq{seq:02d}.npz')
    odom = slam['odom']                                   # T_world_velo (里程计)
    calib = io.read_calib(seq)
    Tr, Tr_inv = calib['Tr'], calib['Tr_inv']
    gt = io.gt_poses_velodyne(seq)
    odom_sub, gt_sub = odom[start:end], gt[start:end]
    cam0_world = (odom @ Tr_inv)[:, :3, 3]                # 各帧 SLAM 相机中心(cam0 世界系)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'loading VGGT ({device}) ...')
    model = load_vggt(device)

    starts = list(range(start, end - a.count + 1, a.stride))
    if starts[-1] != end - a.count:
        starts.append(end - a.count)
    seen, vggt_edges, rmses = set(), [], []
    for w0 in starts:
        frames = list(range(w0, w0 + a.count))
        edges, s, rmse = window_edges(model, seq, frames, cam0_world[frames],
                                      Tr, Tr_inv, a.weight, device)
        rmses.append(rmse)
        for (ga, gb, T, w, unc) in edges:                 # 转局部索引, 去重(重叠窗)
            if (ga - start, gb - start) in seen:
                continue
            seen.add((ga - start, gb - start))
            vggt_edges.append((ga - start, gb - start, T, w, unc))
        print(f'  win {w0:4d}-{w0 + a.count - 1:<4d}  scale={s:6.2f}  cam-align RMSE={rmse * 100:5.1f} cm')
    print(f'VGGT: {len(vggt_edges)} 条相对位姿边, {len(starts)} 窗, 平均对齐 RMSE {np.mean(rmses) * 100:.1f} cm')

    g0, g1 = (int(x) for x in a.gap.split(','))
    dg0, dg1 = g0 - start, g1 - start                     # 局部索引
    drop = set(range(dg0, dg1))                           # 中断段: 丢弃里程计边
    frozen = odom_sub.copy()                              # 里程计丢失 → 冻结在中断起点位姿
    delta = odom_sub[dg0][:3, 3] - odom_sub[dg1][:3, 3]   # 未测到的位移(尾段整体平移量)
    for k in range(dg0 + 1, dg1 + 1):
        frozen[k] = odom_sub[dg0].copy()                  # 中断内: 停在最后已知位姿
    for k in range(dg1 + 1, N):
        frozen[k] = odom_sub[k].copy()
        frozen[k][:3, 3] += delta                         # 中断后: 尾段整体错位(丢了这段位移)

    runs = {
        'LiDAR-only (clean)':  build_graph(odom_sub, odom_sub, set(), []),
        'LiDAR+VGGT (clean)':  build_graph(odom_sub, odom_sub, set(), vggt_edges),
        'LiDAR-only (dropout)': build_graph(frozen, odom_sub, drop, []),
        'LiDAR+VGGT (dropout)': build_graph(frozen, odom_sub, drop, vggt_edges),
    }
    poses, ates = {}, {}
    for name, graph in runs.items():
        P = pg.optimize(graph)
        poses[name] = P
        ates[name] = ate(P, gt_sub)['ate_rmse_m']
    print(f'\n=== seq{seq:02d} [{start}:{end}] VGGT×LiDAR 联合后端 (中断 {g0}-{g1}) ===')
    for name in runs:
        print(f'  {name:24s} ATE {ates[name]:6.3f} m')

    _render(a.out.format(seq=seq), seq, gt_sub, poses, ates, (dg0, dg1))
    np.savez(ROOT / 'reports' / f'vggt_couple_seq{seq:02d}.npz',
             gt=gt_sub, gap=(g0, g1), **{k.replace(' ', '_'): v for k, v in poses.items()})
    return 0


def _xz(P, gt):
    """把估计位姿 SE3 对齐到 GT 后取 (x,z) 便于同框比较（KITTI 世界系 y 为竖直）。"""
    d = ate(P, gt)
    return d['aligned'][:, [0, 2]], d['gt_xyz'][:, [0, 2]]


def _render(out, seq, gt, poses, ates, gap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(ps.BG)
    panels = [('Clean data: fusion is break-even (honest)', 'LiDAR-only (clean)', 'LiDAR+VGGT (clean)'),
              ('LiDAR odometry dropout: VGGT bridges the gap', 'LiDAR-only (dropout)', 'LiDAR+VGGT (dropout)')]
    for ax, (title, lidar_key, fuse_key) in zip(axes, panels):
        est_l, gxz = _xz(poses[lidar_key], gt)
        est_f, _ = _xz(poses[fuse_key], gt)
        ax.plot(gxz[:, 0], gxz[:, 1], color=ps.GT, lw=2.4, label='Ground truth', zorder=3)
        ax.plot(est_l[:, 0], est_l[:, 1], color=ps.MUTED, lw=1.8, ls='--',
                label=f'LiDAR-only   ATE {ates[lidar_key]:.2f} m', zorder=4)
        ax.plot(est_f[:, 0], est_f[:, 1], color=ps.EST, lw=1.8,
                label=f'LiDAR+VGGT  ATE {ates[fuse_key]:.2f} m', zorder=5)
        g0i, g1i = gap
        ax.scatter(gxz[g0i:g1i, 0], gxz[g0i:g1i, 1], s=14, color=ps.ACCENT,
                   alpha=0.55, zorder=2, label='dropout span' if 'dropout' in lidar_key else None)
        ax.scatter(gxz[0, 0], gxz[0, 1], s=70, color=ps.START, edgecolor='w',
                   zorder=6, label='start')
        ax.set_title(title, color=ps.FG, fontsize=12)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_aspect('equal', 'datalim')
        ps.style_ax(ax)
        ps.style_legend(ax.legend(loc='best', fontsize=8))
    fig.suptitle(f'VGGT x LiDAR deep coupling (joint pose-graph backend)  ·  seq{seq:02d}',
                 color=ps.FG, fontsize=14)
    ps.savefig(fig, ROOT / out)
    print(f'  saved {out}')


if __name__ == '__main__':
    raise SystemExit(main())


