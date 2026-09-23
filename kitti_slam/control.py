"""闭环导航：运动学自行车模型 + pure-pursuit 跟踪 + DWA 局部避障（转向/速度解耦）。

全局 A*（`planning.plan`）只给一条几何路径；这里把它**真正开出来**——用车辆运动学模型前向
仿真：每步在 pure-pursuit 转向扇里短程 rollout、按占据栅格拒碰撞、评分选最优转向（沿路径推进
+ 保余隙 − 偏离路径 − 舵变化率）；纵向速度按**转弯限速 + 到障碍的刹车距离**独立平滑调速，弯道/
障碍前提前减速而非冲上去急刹。对全局图里没有的临时障碍也能反应式绕开。诚实报出横向误差 / 余隙
/ 是否到达。全离线、复用现有 SLAM 占据图，不依赖任何外部规划库。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def bicycle_step(state, v, delta, dt, wheelbase):
    """运动学自行车模型一步。state=(x,y,theta) 世界系；v[m/s]、delta[rad] 前轮转角。"""
    x, y, th = state
    return np.array([x + v * np.cos(th) * dt,
                     y + v * np.sin(th) * dt,
                     th + v / wheelbase * np.tan(delta) * dt])


def clearance_grid(free, res):
    """每格到最近不可行格的距离(米)。free[r,c]=可行驶（保车在街道走廊内、离墙有余隙）。"""
    return distance_transform_edt(free) * res


def _clear_at(clg, res, x0, y0, xy):
    """世界坐标 (x,y) 处的余隙(米)；出界记 0。"""
    c = int(round((xy[0] - x0) / res)); r = int(round((xy[1] - y0) / res))
    if 0 <= r < clg.shape[0] and 0 <= c < clg.shape[1]:
        return float(clg[r, c])
    return 0.0


def _nearest_on_path(path, p, i0, window=80):
    """path 上离 p 最近点的索引（从 i0 起窗口内搜，单调前进）。"""
    j = min(i0 + window, len(path))
    d = np.linalg.norm(path[i0:j] - p[:2], axis=1)
    return i0 + int(np.argmin(d))


def _lookahead_target(path, i_near, Ld):
    """从最近点沿路径走 Ld 米的前视点索引。"""
    acc = 0.0
    for k in range(i_near, len(path) - 1):
        acc += np.linalg.norm(path[k + 1] - path[k])
        if acc >= Ld:
            return k + 1
    return len(path) - 1


def pure_pursuit_delta(state, target, wheelbase):
    """pure-pursuit 转向角：目标点在车体系的横向偏差决定曲率。"""
    x, y, th = state
    dx, dy = target[0] - x, target[1] - y
    ld = np.hypot(dx, dy)
    if ld < 1e-6:
        return 0.0
    alpha = np.arctan2(dy, dx) - th
    return np.arctan2(2 * wheelbase * np.sin(alpha), ld)


class Vehicle:
    """车辆 / 控制限幅参数（KITTI 城区乘用车尺度）。"""
    def __init__(self, wheelbase=2.7, v_max=6.0, a_max=2.0, a_lat_max=2.0,
                 delta_max=np.deg2rad(35), r_safe=1.0, dt=0.1):
        self.wheelbase, self.v_max, self.a_max = wheelbase, v_max, a_max
        self.a_lat_max = a_lat_max                       # 横向加速度上限 → 转弯限速
        self.delta_max, self.r_safe, self.dt = delta_max, r_safe, dt


def dwa_command(state, v, path, cum, i_near, clg, res, x0, y0, veh, de_prev=0.0,
                horizon=3.0):
    """局部规划一步（转向/速度解耦）：在 pure-pursuit 转向扇里 rollout 拒碰撞选最优转向，
    再按**转弯限速 + 到障碍的刹车距离**定纵向速度。返回 (v_cmd, delta_cmd, ok)。

    转向：pp 为中心的扇形候选，每个短程 rollout（固定探测速度，看前方约 v·horizon 米），
    评分 = 沿全局路径推进 + 保余隙 − 偏离路径 − 舵变化率；选最优转向及其“无碰撞可行前进
    距离”free。速度：v_des = min(v_max, 转弯限速 sqrt(a_lat·R), 刹车限速 sqrt(2·a·free))，
    再按 a_max 限幅——车会在障碍/弯道前**提前平滑减速**，而非冲上去急刹（消除速度锯齿）。
    ok=False 表示前方已无可行进距离且已停住（真正受阻）。
    """
    Ld = 3.0 + 0.5 * v                                   # 前视距随速自适应
    target = path[_lookahead_target(path, i_near, Ld)]
    pp = pure_pursuit_delta(state, target, veh.wheelbase)
    deltas = np.clip(pp + np.linspace(-0.5, 0.5, 21), -veh.delta_max, veh.delta_max)
    n = max(1, int(horizon / veh.dt))
    vp = max(v, 3.0)                                      # 固定探测速度：前视距离与命令速度解耦
    base, best = cum[i_near], -1e18
    best_de, best_free = 0.0, 0.0
    for de in deltas:
        s = state.copy(); free = 0.0; hit = False; min_clr = 1e9
        for _ in range(n):
            s = bicycle_step(s, vp, de, veh.dt, veh.wheelbase)
            clr = _clear_at(clg, res, x0, y0, s)
            min_clr = min(min_clr, clr)
            if clr < veh.r_safe:
                hit = True; break
            free += vp * veh.dt
        k = _nearest_on_path(path, s, i_near, window=160)
        score = (cum[k] - base + 0.5 * min(min_clr, 3.0)
                 - 0.7 * np.hypot(s[0] - path[k, 0], s[1] - path[k, 1])
                 - 0.5 * abs(de - de_prev) - 0.2 * abs(de))
        if not hit:
            score += 2.0                                 # 全程无碰撞的转向优先
        if score > best:
            best, best_de = score, de
            best_free = 1e9 if not hit else free
    de = best_de
    R = veh.wheelbase / max(abs(np.tan(de)), 1e-3)       # 转弯半径
    v_curve = np.sqrt(veh.a_lat_max * R)                 # 转弯限速
    v_brake = (veh.v_max if best_free > 1e8
               else np.sqrt(2 * veh.a_max * max(best_free - veh.r_safe, 0.0)))
    v_des = min(veh.v_max, v_curve, v_brake)
    v_cmd = float(np.clip(v_des, max(0.0, v - veh.a_max * veh.dt), v + veh.a_max * veh.dt))
    ok = best_free > 0.3 or v_cmd > 0.1
    return v_cmd, de, ok


def run_closed_loop(path, clg, res, x0, y0, veh=None, start_theta=None,
                    max_steps=8000, goal_tol=2.5):
    """闭环仿真：沿全局 path 把车开到终点。返回 states/cmds/xte/clr/metrics。"""
    veh = veh or Vehicle()
    path = np.asarray(path, float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if start_theta is None:
        start_theta = np.arctan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
    state = np.array([path[0, 0], path[0, 1], start_theta])
    goal, v, de, i_near = path[-1], 0.0, 0.0, 0
    states, cmds = [state.copy()], []
    stall = 0                                            # 连续原地不前进的步数
    for _ in range(max_steps):
        i_near = _nearest_on_path(path, state, i_near, window=160)
        if np.hypot(state[0] - goal[0], state[1] - goal[1]) < goal_tol and i_near >= len(path) - 3:
            break
        i_prev = i_near
        v, de, ok = dwa_command(state, v, path, cum, i_near, clg, res, x0, y0, veh, de)
        state = bicycle_step(state, v, de, veh.dt, veh.wheelbase)
        states.append(state.copy()); cmds.append((v, de))
        i_after = _nearest_on_path(path, state, i_prev, window=160)
        stall = stall + 1 if (i_after <= i_prev and v < 0.5) else 0
        if stall > 40:                                   # 4s 原地不前进：诚实报未达
            break
    states = np.array(states); cmds = np.array(cmds) if cmds else np.zeros((0, 2))
    reached = bool(np.hypot(states[-1, 0] - goal[0], states[-1, 1] - goal[1]) < goal_tol)
    xte = np.array([np.min(np.linalg.norm(path - s[:2], axis=1)) for s in states])
    clr = np.array([_clear_at(clg, res, x0, y0, s) for s in states])
    driven = float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)))
    m = dict(reached=reached, steps=len(states), driven_m=driven,
             xte_mean=float(xte.mean()), xte_max=float(xte.max()), clr_min=float(clr.min()),
             v_mean=float(cmds[:, 0].mean()) if len(cmds) else 0.0, sim_time_s=len(cmds) * veh.dt)
    return dict(states=states, cmds=cmds, xte=xte, clr=clr, metrics=m, path=path)
