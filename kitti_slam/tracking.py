"""多目标跟踪（世界系 2D 恒速卡尔曼 + 最近邻关联 + 航迹管理）。

在世界系(用 SLAM 位姿把每帧检测框中心变到世界)跟踪物体：静态物体世界坐标不动、动态物体
在动——由此可判 dynamic/static，供建图剔除动态点。经典 MOT（卡尔曼预测/更新、门控关联、
命中/丢失计数出生消亡），感知基本功。
"""
from __future__ import annotations

import numpy as np


class Track:
    _next = 0

    def __init__(self, xy, t):
        self.id = Track._next; Track._next += 1
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])      # [x,y,vx,vy]
        self.P = np.diag([1., 1., 10., 10.])
        self.hits = 1
        self.misses = 0
        self.history = [(t, xy[0], xy[1])]

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q = 0.5
        Q = q * np.array([[dt**3/3, 0, dt**2/2, 0], [0, dt**3/3, 0, dt**2/2],
                          [dt**2/2, 0, dt, 0], [0, dt**2/2, 0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, xy, t):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.diag([0.5, 0.5])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(xy) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1; self.misses = 0
        self.history.append((t, self.x[0], self.x[1]))

    @property
    def speed(self):
        return float(np.hypot(self.x[2], self.x[3]))


class MOT:
    def __init__(self, gate=2.5, max_misses=5, min_hits=3):
        self.gate = gate
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.tracks = []
        self.finished = []

    def step(self, dets_xy, t, dt):
        """dets_xy: (M,2) 世界系检测中心；返回当前活跃且已确认的航迹。"""
        for tr in self.tracks:
            tr.predict(dt)
        used = set()
        # 贪心最近邻关联（门控内）
        for tr in self.tracks:
            best, bd = -1, self.gate
            for j, d in enumerate(dets_xy):
                if j in used:
                    continue
                dist = np.hypot(tr.x[0] - d[0], tr.x[1] - d[1])
                if dist < bd:
                    bd, best = dist, j
            if best >= 0:
                tr.update(dets_xy[best], t); used.add(best)
            else:
                tr.misses += 1
        for j, d in enumerate(dets_xy):          # 未匹配检测→新航迹
            if j not in used:
                self.tracks.append(Track(d, t))
        alive = []                                # 航迹消亡
        for tr in self.tracks:
            if tr.misses > self.max_misses:
                self.finished.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]

    def all_tracks(self, min_hits=3):
        return [tr for tr in self.tracks + self.finished if tr.hits >= min_hits]
