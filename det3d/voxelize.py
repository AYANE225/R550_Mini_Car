"""点云 → pillars（PointPillars 前处理，纯 numpy 向量化）。

把 x-y 平面切成柱子(pillar)，每柱取至多 max_pts 个点，每点 9 维特征：
  [x, y, z, intensity, xc,yc,zc(到柱内点均值的偏移), xp,yp(到柱中心的偏移)]。
返回 pillars[P,max_pts,9]、coords[P,2]=(iy,ix)、npoints[P]。
"""
from __future__ import annotations

import numpy as np

# KITTI Car 标准范围/柱尺寸
PC_RANGE = (0.0, -39.68, -3.0, 69.12, 39.68, 1.0)     # xmin,ymin,zmin,xmax,ymax,zmax
VSIZE = (0.16, 0.16, 4.0)
NX = int(round((PC_RANGE[3] - PC_RANGE[0]) / VSIZE[0]))   # 432
NY = int(round((PC_RANGE[4] - PC_RANGE[1]) / VSIZE[1]))   # 496


def points_to_pillars(pts, max_pts=32, max_pillars=12000,
                      pc_range=PC_RANGE, vsize=VSIZE):
    x0, y0, z0, x1, y1, z1 = pc_range
    vx, vy, _ = vsize
    m = ((pts[:, 0] >= x0) & (pts[:, 0] < x1) &
         (pts[:, 1] >= y0) & (pts[:, 1] < y1) &
         (pts[:, 2] >= z0) & (pts[:, 2] < z1))
    p = pts[m]
    ix = ((p[:, 0] - x0) / vx).astype(np.int32)
    iy = ((p[:, 1] - y0) / vy).astype(np.int32)
    key = iy.astype(np.int64) * NX + ix
    order = np.argsort(key, kind='stable')
    p, key, ix, iy = p[order], key[order], ix[order], iy[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    if len(uniq) > max_pillars:
        sel = np.argsort(counts)[::-1][:max_pillars]          # 保留点最多的柱
        sel.sort()
        uniq, start, counts = uniq[sel], start[sel], counts[sel]
    P = len(uniq)
    pillars = np.zeros((P, max_pts, 9), dtype=np.float32)
    coords = np.zeros((P, 2), dtype=np.int32)
    npoints = np.minimum(counts, max_pts).astype(np.int32)
    for j in range(P):
        s, cnt = start[j], min(counts[j], max_pts)
        pp = p[s:s + cnt]                                     # 该柱的点(截断)
        mean = p[s:s + counts[j]].mean(0)                     # 柱内点均值(全量)
        cx = x0 + (ix[s] + 0.5) * vx
        cy = y0 + (iy[s] + 0.5) * vy
        f = np.empty((cnt, 9), dtype=np.float32)
        f[:, 0:4] = pp[:, 0:4]                                # x,y,z,intensity
        f[:, 4:7] = pp[:, 0:3] - mean[0:3]                    # xc,yc,zc
        f[:, 7] = pp[:, 0] - cx                               # xp
        f[:, 8] = pp[:, 1] - cy                               # yp
        pillars[j, :cnt] = f
        coords[j] = (iy[s], ix[s])
    return pillars, coords, npoints
