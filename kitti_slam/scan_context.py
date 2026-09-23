"""Scan Context：外观级回环检测描述子（自研，无研究代码）。

Kim & Kim, IROS 2018。把一帧激光的水平面切成 极坐标(环×扇区) 栅格，每格存该格里点的
**最大高度**，得到一张 (n_ring, n_sector) 的"场景指纹"。要点：

- **旋转不变环键**(ring key)：每环在扇区维上取均值 → (n_ring,) 向量。车辆偏航只在扇区维
  上循环平移描述子，逐环均值不变——用它建 KD 树快速筛候选。
- **列移不变距离**：两张描述子按扇区循环平移逐列比余弦距离，取所有平移下的最小值（顺带得到
  相对偏航）。

关键价值：它**按外观匹配**，不依赖里程计位置。位置法回环靠"两次经过同一地点的里程计坐标要
足够近"——里程计一旦漂移，重访点在里程计系里被推开就漏检；Scan Context 不受漂移影响。
纯 numpy / scipy。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def scan_context(pts, n_ring=20, n_sector=60, max_range=80.0, z_offset=2.0):
    """(N,3+) velodyne 点(z 朝上) → (n_ring, n_sector) 最大高度极坐标描述子。空格为 0。"""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y)
    keep = (r > 1e-3) & (r < max_range)
    r, z = r[keep], z[keep] + z_offset
    theta = np.arctan2(y[keep], x[keep])                         # [-pi, pi)
    ring = np.minimum((r / max_range * n_ring).astype(np.intp), n_ring - 1)
    sector = np.minimum(((theta + np.pi) / (2 * np.pi) * n_sector).astype(np.intp), n_sector - 1)
    flat = np.zeros(n_ring * n_sector, np.float32)
    np.maximum.at(flat, ring * n_sector + sector, np.maximum(z, 0.0))
    return flat.reshape(n_ring, n_sector)


def ring_key(sc):
    """旋转不变环键：每环在扇区上的均值 → (n_ring,)。偏航只置换扇区，逐环均值不变。"""
    return sc.mean(axis=1)


def sc_distance(a, b):
    """列移不变余弦距离(对应偏航旋转)。返回 (最小距离∈[0,1], 最佳扇区平移)。"""
    na = np.linalg.norm(a, axis=0)
    nb = np.linalg.norm(b, axis=0)
    best_d, best_s = 2.0, 0
    for s in range(a.shape[1]):
        nbs = np.roll(nb, s)
        both = (na > 1e-6) & (nbs > 1e-6)
        if not both.any():
            continue
        bs = np.roll(b, s, axis=1)
        cos = (a[:, both] * bs[:, both]).sum(0) / (na[both] * nbs[both])
        d = 1.0 - float(cos.mean())
        if d < best_d:
            best_d, best_s = d, s
    return best_d, best_s


def yaw_from_shift(shift, n_sector):
    """扇区平移 → 相对偏航角(弧度)。"""
    return 2.0 * np.pi * shift / n_sector


class ScanContextManager:
    """在线维护每个关键帧的描述子 + 环键，查询与历史最相似的一帧。"""

    def __init__(self, n_ring=20, n_sector=60, max_range=80.0, n_candidates=10):
        self.n_ring, self.n_sector, self.max_range = n_ring, n_sector, max_range
        self.n_candidates = n_candidates
        self.scs, self.keys, self.frames = [], [], []

    def add(self, frame, pts):
        sc = scan_context(pts, self.n_ring, self.n_sector, self.max_range)
        self.scs.append(sc)
        self.keys.append(ring_key(sc))
        self.frames.append(int(frame))
        return sc

    def query(self, i, min_gap_idx=1):
        """在第 i 个描述子之前(索引间隔≥min_gap_idx)找最相似历史帧。
        返回 (best_i_index, dist, yaw)；无候选返回 None。索引为加入顺序下标(非帧号)。"""
        upto = i - min_gap_idx
        if upto <= 0:
            return None
        tree = cKDTree(np.asarray(self.keys[:upto]))
        k = min(self.n_candidates, upto)
        _, cand = tree.query(self.keys[i], k=k)
        best = (None, 2.0, 0)
        for j in np.atleast_1d(cand):
            d, s = sc_distance(self.scs[i], self.scs[int(j)])
            if d < best[1]:
                best = (int(j), d, s)
        if best[0] is None:
            return None
        return best[0], best[1], yaw_from_shift(best[2], self.n_sector)
