"""闭环导航控制的纯逻辑测试（合成数据，不依赖 KITTI/GPU）。

只测数据集无关的自研逻辑：运动学自行车模型、余隙场、pure-pursuit、以及在无障碍
合成路径上闭环仿真能开到终点且横向误差小。
"""
import numpy as np

from kitti_slam import control as ctrl


def test_bicycle_step_straight_advances_x_only():
    s = ctrl.bicycle_step(np.array([0.0, 0.0, 0.0]), v=2.0, delta=0.0, dt=0.5, wheelbase=2.7)
    assert np.allclose(s, [1.0, 0.0, 0.0])


def test_bicycle_step_left_turn_increases_heading():
    s = ctrl.bicycle_step(np.array([0.0, 0.0, 0.0]), v=2.0, delta=0.3, dt=0.5, wheelbase=2.7)
    assert s[2] > 0.0                       # 正前轮转角 → 航向增大（左转）
    assert s[0] > 0.0                       # 仍向前推进


def test_clearance_grid_distance_from_obstacle():
    free = np.ones((5, 5), bool)
    free[2, 2] = False                      # 中心为障碍
    clg = ctrl.clearance_grid(free, res=0.5)
    assert clg[2, 2] == 0.0
    assert np.isclose(clg[2, 0], 1.0)       # 距最近障碍 2 格 × 0.5 m


def test_pure_pursuit_zero_when_target_straight_ahead():
    d = ctrl.pure_pursuit_delta(np.array([0.0, 0.0, 0.0]), target=np.array([10.0, 0.0]),
                                wheelbase=2.7)
    assert abs(d) < 1e-9


def test_pure_pursuit_positive_when_target_left():
    d = ctrl.pure_pursuit_delta(np.array([0.0, 0.0, 0.0]), target=np.array([10.0, 3.0]),
                                wheelbase=2.7)
    assert d > 0.0                          # 目标在左 → 左转


def _clear_grid(res=0.5, h=200, w=400):
    """无障碍走廊：仅四周边界不可行，内部余隙很大。返回 (clg, res, x0, y0)。"""
    free = np.ones((h, w), bool)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    return ctrl.clearance_grid(free, res), res, 0.0, 0.0


def test_run_closed_loop_reaches_goal_on_clear_straight_path():
    clg, res, x0, y0 = _clear_grid()
    xs = np.arange(20.0, 160.0, 1.0)
    path = np.column_stack([xs, np.full_like(xs, 50.0)])
    r = ctrl.run_closed_loop(path, clg, res, x0, y0)
    m = r['metrics']
    assert m['reached']
    assert m['xte_mean'] < 1.0              # 直路应贴合良好
    assert m['clr_min'] > 1.0               # 全程未逼近边界


def test_run_closed_loop_tracks_gentle_turn():
    clg, res, x0, y0 = _clear_grid()
    t = np.linspace(0, 1, 140)
    path = np.column_stack([20.0 + 100.0 * t, 40.0 + 20.0 * t ** 2])  # 平滑右弯
    r = ctrl.run_closed_loop(path, clg, res, x0, y0)
    assert r['metrics']['reached']
    assert r['metrics']['xte_mean'] < 1.5
