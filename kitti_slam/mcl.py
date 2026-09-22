"""蒙特卡洛定位（粒子滤波 / AMCL 风格）在已建 2D 占据图上定位。

似然域测量模型：对占据图做距离变换，扫描端点落在离障碍越近处似然越高（快且稳）。
运动模型：吃里程计帧间相对位姿（body 系 dx,dy,dyaw）加噪。真值不参与，仅事后评测。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


class LikelihoodField:
    """由占据栅格算"到最近障碍的距离场"，给扫描端点打似然。"""

    def __init__(self, occ, res, x0, y0, sigma=0.5):
        self.res, self.x0, self.y0, self.sigma = res, x0, y0, sigma
        self.H, self.W = occ.shape
        self.dist = distance_transform_edt(~occ) * res     # 每个自由格到最近障碍(米)
        self.max_dist = float(self.dist.max())

    def batch_loglik(self, wx, wy, z_rand=0.05):
        """wx,wy: (N,K) 世界系端点(N 粒子×K 点)。返回每粒子对数似然 (N,)。全向量化。"""
        cols = np.round((wx - self.x0) / self.res).astype(int)
        rows = np.round((wy - self.y0) / self.res).astype(int)
        inb = (rows >= 0) & (rows < self.H) & (cols >= 0) & (cols < self.W)
        rr = np.clip(rows, 0, self.H - 1)
        cc = np.clip(cols, 0, self.W - 1)
        d = np.where(inb, self.dist[rr, cc], self.max_dist)
        p = np.exp(-(d ** 2) / (2 * self.sigma ** 2)) + z_rand
        return np.log(p).sum(axis=1)


def pose_2d(T):
    """4x4(velo 系,z 上) → (x, y, yaw)。"""
    return T[0, 3], T[1, 3], np.arctan2(T[1, 0], T[0, 0])


def rel_motion(T_prev, T_cur):
    """帧间相对运动（前一帧 body 系）：返回 (dx, dy, dyaw)。"""
    d = np.linalg.inv(T_prev) @ T_cur
    return d[0, 3], d[1, 3], np.arctan2(d[1, 0], d[0, 0])


class MCL:
    def __init__(self, field, n=500, motion_sigma=(0.05, 0.05, 0.01), rng=None):
        self.field = field
        self.n = n
        self.ms = np.asarray(motion_sigma)
        self.rng = rng or np.random.default_rng(0)
        self.P = np.zeros((n, 3))      # 粒子 (x,y,yaw)
        self.w = np.ones(n) / n

    def init(self, pose_xy_yaw, spread=(1.0, 1.0, 0.1)):
        x, y, yaw = pose_xy_yaw
        s = np.asarray(spread)
        self.P = np.column_stack([
            self.rng.normal(x, s[0], self.n),
            self.rng.normal(y, s[1], self.n),
            self.rng.normal(yaw, s[2], self.n)])
        self.w[:] = 1.0 / self.n

    def predict(self, dxy_yaw):
        dx, dy, dyaw = dxy_yaw
        noise = self.rng.normal(0, 1, (self.n, 3)) * self.ms
        c, s = np.cos(self.P[:, 2]), np.sin(self.P[:, 2])   # 把 body 位移旋到世界
        self.P[:, 0] += (c * dx - s * dy) + noise[:, 0]
        self.P[:, 1] += (s * dx + c * dy) + noise[:, 1]
        self.P[:, 2] += dyaw + noise[:, 2]

    def update(self, scan_xy_body):
        """scan_xy_body: (K,2) 当前帧障碍点(body 系)。更新权重并按需重采样。全向量化。"""
        bx, by = scan_xy_body[:, 0], scan_xy_body[:, 1]
        c = np.cos(self.P[:, 2])[:, None]; s = np.sin(self.P[:, 2])[:, None]
        wx = self.P[:, 0:1] + c * bx[None, :] - s * by[None, :]     # (N,K)
        wy = self.P[:, 1:2] + s * bx[None, :] + c * by[None, :]
        logw = self.field.batch_loglik(wx, wy)
        logw -= logw.max()
        w = np.exp(logw) * self.w
        self.w = w / w.sum()
        if 1.0 / np.sum(self.w ** 2) < self.n / 2:      # 有效粒子过少→重采样
            self._resample()

    def _resample(self):
        pos = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(self.w), pos)
        idx = np.clip(idx, 0, self.n - 1)
        self.P = self.P[idx] + self.rng.normal(0, 1, (self.n, 3)) * (self.ms * 0.5)
        self.w[:] = 1.0 / self.n

    def estimate(self):
        x = np.average(self.P[:, 0], weights=self.w)
        y = np.average(self.P[:, 1], weights=self.w)
        yaw = np.arctan2(np.average(np.sin(self.P[:, 2]), weights=self.w),
                         np.average(np.cos(self.P[:, 2]), weights=self.w))
        return np.array([x, y, yaw])
