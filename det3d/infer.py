"""推理后处理：解码预测 → 方向消歧 → 分数阈值 → BEV 旋转 NMS。"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .anchors import decode, _corners


def _limit(v, period=2 * np.pi):
    return v - np.floor(v / period + 0.5) * period      # 回到 [-period/2, period/2)


def decode_predictions(preds, anchors, score_thr=0.3):
    """preds: 单样本 dict(cls[Na,1],box[Na,7],dir[Na,2]) 的 numpy。返回 boxes[K,7], scores[K]。"""
    scores = 1.0 / (1.0 + np.exp(-preds['cls'][:, 0]))
    keep = scores >= score_thr
    if keep.sum() == 0:
        return np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)
    boxes = decode(preds['box'][keep], anchors[keep])
    dir_label = preds['dir'][keep].argmax(1)             # 0/1
    yaw = _limit(boxes[:, 6])
    upper = ((yaw + np.pi) % (2 * np.pi)) >= np.pi       # 当前是否落在上半(与 dir_t 定义一致)
    flip = upper != (dir_label == 1)
    yaw[flip] += np.pi
    boxes[:, 6] = _limit(yaw)
    return boxes.astype(np.float32), scores[keep].astype(np.float32)


def rotated_nms(boxes, scores, iou_thr=0.1):
    if len(boxes) == 0:
        return np.zeros((0,), np.int64)
    polys = [Polygon(_corners(b)) for b in boxes]
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(boxes), bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if suppressed[j] or j == i:
                continue
            inter = polys[i].intersection(polys[j]).area
            iou = inter / (polys[i].area + polys[j].area - inter + 1e-9)
            if iou > iou_thr:
                suppressed[j] = True
    return np.array(keep, np.int64)


def postprocess(preds, anchors, score_thr=0.3, nms_thr=0.1, max_out=50):
    boxes, scores = decode_predictions(preds, anchors, score_thr)
    idx = rotated_nms(boxes, scores, nms_thr)[:max_out]
    return boxes[idx], scores[idx]
