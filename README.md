# KITTI 大场景 LiDAR SLAM · 定位 · 导航

真实大场景（KITTI Odometry，室外城区，单序列数公里）上的**稳定轨迹建图 + 定位 + 导航**
完整栈。纯 LiDAR、离线、可复现。全部自研，**不依赖任何非公开研究代码，暂不融合视觉**。

> 库依赖：`numpy / scipy / open3d`（Open3D 用于 ICP 配准与位姿图优化）。数据为公开的
> KITTI Odometry（velodyne 扫描 + 标定 + 真值位姿）。真值**仅用于评测**（declare→predict→evaluate）。

## 流水线与结果（KITTI seq00，4541 帧 / 3737 m）

| 阶段 | 方法 | 结果 |
| --- | --- | --- |
| ① 里程计 | scan-to-local-map point-to-plane ICP + 匀速先验 | 前 200 帧 ATE **0.22 m** |
| ② 回环+后端 | 位置近邻→ICP 验证回环；Open3D 位姿图全局优化(LM) | 全程 ATE **4.85 m → 2.25 m**（10 回环） |
| ③ 建图 | 优化位姿拼 3D 体素地图 + 高度带投影 2D 占据栅格 | 310 万点，~600×600 m 城区图 |
| ④ 定位 | 在先验地图上 scan-to-map ICP（里程计增量作初值） | RMSE **0.05 m（有界）** vs 里程计漂 7.7 m |
| ⑤ 导航 | 建图上全局 A*（障碍膨胀 + 可行驶走廊） | 起点→最远点规划 **650 m**（实际绕行 3737 m） |

产物在 `reports/`：`slam_seq00.png`（回环前后轨迹）、`map_seq00_bev.png` / `_3d.png`（建出的
2D/3D 地图）、`localize_seq00.png`（定位误差有界 vs 里程计漂移）、`nav_seq00.png`（路网规划）。

## 运行

```bash
PY=/media/4T/cst/envs/vggt-slam/bin/python3     # 仅借其 numpy/scipy/open3d，不 import 任何研究模块
$PY scripts/run_odometry.py --seq 0 --frames 200 --out reports/odom.png   # ① 里程计里程碑
$PY scripts/run_slam.py     --seq 0 --frames -1 --out reports/slam_seq00.png  # ②(缓存里程计位姿+SLAM npz)
$PY scripts/run_mapping.py  --seq 0     # ③ 建 3D/2D 地图
$PY scripts/run_localize.py --seq 0     # ④ scan-to-map 定位
$PY scripts/run_nav.py      --seq 0     # ⑤ 全局路径规划
```

## 模块

```
kitti_slam/
  kitti_io.py      读 velodyne/标定(Tr 取 official)/真值位姿/时间戳
  registration.py  点云裁剪+体素下采样+法线 & point-to-plane ICP（Open3D）
  odometry.py      LidarOdometry：scan-to-local-map 里程计
  loop.py          回环检测（位置近邻→ICP 验证收伪）
  posegraph.py     位姿图后端（Open3D 全局优化 LM）
  mapping.py       3D 体素地图 + 2D 占据栅格投影
  localize.py      MapLocalizer：scan-to-map ICP 先验地图定位
  planning.py      栅格 A* 全局规划（障碍膨胀 + 可行驶走廊）
  metrics.py       SE(3) 对齐 ATE（真值仅在此用）
```

## 设计要点 / 踩过的坑

- **坐标系**：里程计在 velodyne 系（z 朝上，水平面 x–y）；KITTI 真值在相机系（y 竖直，水平
  面 x–z），比对/画俯视图务必分清。ATE 用 SE(3) 对齐，与坐标系无关。
- **定位为何用 ICP 而非 MCL**：先实现了似然域 MCL 粒子滤波，在 KITTI 大场景朝向弱约束、远
  点敏感，会发散到百米级；换 **scan-to-map ICP**（配准固定地图，误差不随时间漂）立刻到亚分米。
- **回环收伪**：ICP fitness≥0.85 且 rmse≤0.85 才接受，正确拒掉起点误匹配等伪回环。
- **规划连通**：把行驶轨迹刻进代价图并加粗桥接，保证路网连通（膨胀可能误封窄街）。

## 路线图

- ☐ 更大 / 多楼层数据集（Newer College / Hilti，需子图 + 位姿图架构）。
- ☐ 独立接入 VGGT 等前馈视觉模型作**视觉前端**（只调冻结权重，与 LiDAR 松耦合做消融）——
  独立实现，不复用任何非公开研究工程。
