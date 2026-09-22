"""锚框：生成 / 编解码 / 目标分配（BEV 旋转 IoU 用 shapely + KD 树空间预筛，纯 numpy）。

顺序与 SSDHead 输出一致：按 (H, W, anchor) 展平。单类 Car：尺寸 [3.9,1.6,1.56]，两朝向 0/90°。
编码沿用 SECOND 残差：t=(Δ中心/对角, log 尺寸比, Δyaw)。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from .voxelize import PC_RANGE

CAR = (3.9, 1.6, 1.56)          # l, w, h
Z_CENTER = -1.0
YAWS = (0.0, np.pi / 2)


def generate_anchors(Hf, Wf):
    x0, y0, _, x1, y1, _ = PC_RANGE
    sx, sy = (x1 - x0) / Wf, (y1 - y0) / Hf
    xs = x0 + (np.arange(Wf) + 0.5) * sx
    ys = y0 + (np.arange(Hf) + 0.5) * sy
    yy, xx = np.meshgrid(ys, xs, indexing='ij')          # [H,W]
    A = np.zeros((Hf, Wf, len(YAWS), 7), dtype=np.float32)
    for a, yaw in enumerate(YAWS):
        A[:, :, a, 0] = xx; A[:, :, a, 1] = yy; A[:, :, a, 2] = Z_CENTER
        A[:, :, a, 3], A[:, :, a, 4], A[:, :, a, 5] = CAR
        A[:, :, a, 6] = yaw
    return A.reshape(-1, 7)                                # [H*W*2,7]，序 (h,w,a)


def _corners(b):
    x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    loc = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
    return loc @ R.T + [x, y]


def encode(gt, anc):
    da = np.sqrt(anc[:, 3] ** 2 + anc[:, 4] ** 2)
    return np.stack([
        (gt[:, 0] - anc[:, 0]) / da, (gt[:, 1] - anc[:, 1]) / da,
        (gt[:, 2] - anc[:, 2]) / anc[:, 5],
        np.log(gt[:, 3] / anc[:, 3]), np.log(gt[:, 4] / anc[:, 4]),
        np.log(gt[:, 5] / anc[:, 5]), gt[:, 6] - anc[:, 6]], axis=1).astype(np.float32)


def decode(t, anc):
    da = np.sqrt(anc[:, 3] ** 2 + anc[:, 4] ** 2)
    return np.stack([
        t[:, 0] * da + anc[:, 0], t[:, 1] * da + anc[:, 1],
        t[:, 2] * anc[:, 5] + anc[:, 2],
        np.exp(t[:, 3]) * anc[:, 3], np.exp(t[:, 4]) * anc[:, 4],
        np.exp(t[:, 5]) * anc[:, 5], t[:, 6] + anc[:, 6]], axis=1).astype(np.float32)


def assign(anchors, gt, pos_iou=0.6, neg_iou=0.45, radius=3.0):
    """返回 labels[Na]∈{-1 忽略,0 负,1 正}、reg_t[Na,7]、dir_t[Na]、matched_gt[Na]。"""
    Na = len(anchors)
    labels = np.zeros(Na, dtype=np.int64)
    reg_t = np.zeros((Na, 7), dtype=np.float32)
    dir_t = np.zeros(Na, dtype=np.int64)
    matched = -np.ones(Na, dtype=np.int64)
    if len(gt) == 0:
        return labels, reg_t, dir_t, matched
    tree = cKDTree(anchors[:, :2])
    best_iou = np.zeros(Na, dtype=np.float32)
    gt_polys = [Polygon(_corners(g)) for g in gt]
    for gi, g in enumerate(gt):
        cand = tree.query_ball_point(g[:2], radius)          # 只算近邻锚
        if not cand:
            continue
        gp = gt_polys[gi]; ga = gp.area
        best_local, best_ai = -1.0, -1
        for ai in cand:
            ap = Polygon(_corners(anchors[ai]))
            inter = ap.intersection(gp).area
            iou = inter / (ap.area + ga - inter + 1e-9)
            if iou > best_iou[ai]:
                best_iou[ai] = iou; matched[ai] = gi
            if iou > best_local:
                best_local, best_ai = iou, ai
        if best_ai >= 0:                                     # 保证每个 GT 至少一个正锚
            best_iou[best_ai] = max(best_iou[best_ai], pos_iou)
            matched[best_ai] = gi
    labels[best_iou < neg_iou] = 0
    labels[(best_iou >= neg_iou) & (best_iou < pos_iou)] = -1
    pos = best_iou >= pos_iou
    labels[pos] = 1
    if pos.any():
        gp = gt[matched[pos]]
        reg_t[pos] = encode(gp, anchors[pos])
        dir_t[pos] = (((gp[:, 6] + np.pi) % (2 * np.pi)) >= np.pi).astype(np.int64)
    return labels, reg_t, dir_t, matched
