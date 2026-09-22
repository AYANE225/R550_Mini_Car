"""检测损失：sigmoid focal(分类) + smooth-L1(框回归,角度用 sin 差) + 交叉熵(方向)。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = targets * p + (1 - targets) * (1 - p)
    w = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - pt).pow(gamma)
    return (w * ce).sum()


def _sin_diff(pred, gt):
    # 角度维(第6列)：用 sin(θp-θg) 平滑 → 把 pred/gt 的角度换成 sinθp·cosθg / cosθp·sinθg
    rp = torch.sin(pred[..., 6:7]) * torch.cos(gt[..., 6:7])
    rg = torch.cos(pred[..., 6:7]) * torch.sin(gt[..., 6:7])
    pred = torch.cat([pred[..., :6], rp], dim=-1)
    gt = torch.cat([gt[..., :6], rg], dim=-1)
    return pred, gt


def det_loss(preds, labels, reg_t, dir_t, w_reg=2.0, w_dir=0.2):
    cls, box, dir_ = preds['cls'], preds['box'], preds['dir']   # [B,Na,1],[B,Na,7],[B,Na,2]
    pos = labels == 1
    valid = labels >= 0                                          # 忽略 -1
    npos = pos.sum().clamp(min=1).float()

    cls_t = pos.unsqueeze(-1).float()
    cls_loss = sigmoid_focal(cls[valid], cls_t[valid]) / npos

    if pos.any():
        p, g = _sin_diff(box[pos], reg_t[pos])
        reg_loss = F.smooth_l1_loss(p, g, reduction='sum', beta=1.0 / 9) / npos
        dir_loss = F.cross_entropy(dir_[pos], dir_t[pos], reduction='mean')
    else:
        reg_loss = box.sum() * 0.0
        dir_loss = dir_.sum() * 0.0
    total = cls_loss + w_reg * reg_loss + w_dir * dir_loss
    return total, {'cls': float(cls_loss.detach()), 'reg': float(reg_loss.detach()),
                   'dir': float(dir_loss.detach())}
