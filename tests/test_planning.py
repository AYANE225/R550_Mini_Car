"""全局规划测试（numpy/scipy，不依赖 open3d/数据）。"""
import numpy as np

from kitti_slam.planning import astar, plan


def test_astar_finds_path_on_open_grid():
    free = np.ones((20, 20), dtype=bool)
    path = astar(free, (0, 0), (19, 19))
    assert path is not None and path[0] == (0, 0) and path[-1] == (19, 19)


def test_astar_blocked_returns_none():
    free = np.ones((10, 10), dtype=bool)
    free[:, 5] = False                       # 一整列墙，无缺口
    assert astar(free, (0, 0), (9, 9)) is None


def test_astar_routes_around_obstacle():
    free = np.ones((11, 11), dtype=bool)
    free[0:8, 5] = False                     # 墙留下方缺口
    path = astar(free, (0, 0), (0, 10))
    assert path is not None
    assert any(r >= 8 for r, _ in path)      # 必须绕到下方缺口


def test_plan_on_synthetic_map_follows_corridor():
    # 20x20m 占据栅格(res 0.5)：中间一堵带缺口的墙；轨迹沿 y=0 走廊
    res = 0.5
    occ = np.zeros((40, 40), dtype=bool)
    occ[0:18, 20] = True                     # 竖墙(下段)，上方留口
    x0, y0 = -10.0, -10.0
    traj = np.column_stack([np.linspace(-8, 8, 50), np.zeros(50)])
    path, _ = plan(occ, res, x0, y0, traj, (-8.0, 0.0), (8.0, 0.0),
                   inflate_m=0.2, corridor_m=3.0, coarsen=1)
    assert path is not None and len(path) >= 2
    assert abs(path[-1][0] - 8.0) < 1.0
