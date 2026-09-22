"""在建好的 2D 占据地图上做全局路径规划（栅格 A*）。

障碍(建筑/墙)按车体半径膨胀；规划限制在"可行驶走廊"内——即行驶过的轨迹附近一段，
避免 A* 抄近路穿过未观测的空白格。返回沿街道的最短连通路径（世界坐标）。
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation


def build_costmap(occ, res, x0, y0, traj_xy, inflate_m=1.0, corridor_m=10.0, coarsen=2):
    """返回 (blocked, drivable, res2, x0, y0)：粗化后的障碍膨胀图 + 可行驶走廊掩码。"""
    if coarsen > 1:                                     # 降分辨率提速：块内任一占据即占据
        H, W = occ.shape
        H2, W2 = H // coarsen, W // coarsen
        occ = occ[:H2 * coarsen, :W2 * coarsen].reshape(H2, coarsen, W2, coarsen).any(axis=(1, 3))
        res = res * coarsen
    r = max(1, int(round(inflate_m / res)))
    blocked = binary_dilation(occ, iterations=r)
    # 可行驶走廊：把轨迹点刻进栅格再膨胀 corridor_m
    drive = np.zeros_like(occ)
    cols = np.round((traj_xy[:, 0] - x0) / res).astype(int)
    rows = np.round((traj_xy[:, 1] - y0) / res).astype(int)
    ok = (rows >= 0) & (rows < occ.shape[0]) & (cols >= 0) & (cols < occ.shape[1])
    drive[rows[ok], cols[ok]] = True
    drive = binary_dilation(drive, iterations=max(1, int(round(corridor_m / res))))
    return blocked, drive, res, x0, y0


def astar(free, start_rc, goal_rc):
    """8 邻接 A*，free[r,c]=可通行。返回栅格路径 [(r,c),...] 或 None。"""
    H, W = free.shape
    if not free[start_rc] or not free[goal_rc]:
        return None
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    h = lambda c: np.hypot(c[0] - goal_rc[0], c[1] - goal_rc[1])
    openh = [(h(start_rc), 0.0, start_rc)]
    came, best = {}, {start_rc: 0.0}
    while openh:
        _, g, cur = heapq.heappop(openh)
        if cur == goal_rc:
            path = [cur]
            while cur in came:
                cur = came[cur]; path.append(cur)
            return path[::-1]
        if g > best.get(cur, 1e18):
            continue
        for dr, dc in nbrs:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < H and 0 <= nb[1] < W) or not free[nb]:
                continue
            ng = g + np.hypot(dr, dc)
            if ng < best.get(nb, 1e18):
                best[nb] = ng; came[nb] = cur
                heapq.heappush(openh, (ng + h(nb), ng, nb))
    return None


def _nearest_free(free, rc, max_r=40):
    if free[rc]:
        return rc
    r0, c0 = rc
    H, W = free.shape
    for rad in range(1, max_r):
        lo_r, hi_r = max(0, r0 - rad), min(H, r0 + rad + 1)
        lo_c, hi_c = max(0, c0 - rad), min(W, c0 + rad + 1)
        sub = free[lo_r:hi_r, lo_c:hi_c]
        if sub.any():
            rr, cc = np.argwhere(sub)[0]
            return (lo_r + rr, lo_c + cc)
    return None


def plan(occ, res, x0, y0, traj_xy, start_xy, goal_xy, **kw):
    """世界坐标起终点 → 世界坐标路径 [(x,y),...] 或 None。"""
    blocked, drive, res2, x0, y0 = build_costmap(occ, res, x0, y0, traj_xy, **kw)
    free = drive & ~blocked
    # 把行驶过的轨迹刻成可通行并膨胀 2 格：车实际走过必然可行，且相邻帧间隔>1格需加粗
    # 桥接成连通线，否则碎点无法 8 邻接走通。保证路网连通（膨胀误封时兜底）。
    cols = np.round((traj_xy[:, 0] - x0) / res2).astype(int)
    rows = np.round((traj_xy[:, 1] - y0) / res2).astype(int)
    ok = (rows >= 0) & (rows < free.shape[0]) & (cols >= 0) & (cols < free.shape[1])
    tmask = np.zeros_like(free)
    tmask[rows[ok], cols[ok]] = True
    free |= binary_dilation(tmask, iterations=2)
    to_rc = lambda p: (int(round((p[1] - y0) / res2)), int(round((p[0] - x0) / res2)))
    s, g = _nearest_free(free, to_rc(start_xy)), _nearest_free(free, to_rc(goal_xy))
    path = astar(free, s, g) if s and g else None
    if path is None:
        return None, (free, res2, x0, y0)
    world = [(x0 + c * res2, y0 + r * res2) for r, c in path]
    return world, (free, res2, x0, y0)
