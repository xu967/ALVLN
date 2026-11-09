# 3DGS 场地记录：Wanderland 1MlUucug（公开，纽约商业街人行道）

- 日期：2026-11-07（SCENE-A 子 agent 接入；未 commit，分支 `scene-wanderland`）
- 数据盘：`$DATA_ROOT/uav_rgb/3dgs-scenes/wanderland_1MlUucug/`（`scene.json`、`scene.ply` 3DGS、`lidar.ply` LiDAR 真值、`collision_mesh.ply` 官方碰撞网格）；原始下载 `$DATA_ROOT/uav_rgb/downloads/wanderland/1MlUucugRAnrx86cpUuQt8fUtxFMddxju/`
- 接入脚本：`tools/scene_import_wanderland.py`（下载→转换→对齐→scene.json→渲染探针→冒烟→视频，`IMPORT_STEPS` 选步）

## 来源与许可

| 项 | 值 |
| --- | --- |
| 数据集 | Wanderland（NYU AI4CE，CVPR'26），HF `ai4ce/wanderland`，无 gating，`huggingface_hub` 直接下 |
| 许可 | Apache-2.0（HF 卡片 + GitHub `LICENSE`），可再分发、可商用，需保留声明 |
| 场景 | `1MlUucugRAnrx86cpUuQt8fUtxFMddxju`，`data/v1/02_evaluation_ready/`，manifest v1.3 `quality_tier=evaluation_ready`，无质量标记 |
| 采集设备 | 手持 MetaCam（鱼眼相机 + LiDAR，LIV-SLAM 米制），563 帧 COLMAP 位姿，轨迹 ≈52 m 去回各一趟 |
| 取用文件 | `3dgs.ply` 473 MB（2.0 M 高斯，INRIA 布局，gsplat 训练）、`raw_pcd.ply` 911 MB（33.7 M 点，double xyz + uchar rgb）、`scene.usdz` 325 MB（含 ASCII `scene.usda` 碰撞网格 1.03 M 顶点 / 2.10 M 面）、`transforms.json`、`episodes.json`、`sparse/0/*`；共 1.71 GB（未取 images/fisheye tar 572 MB） |

## 坐标系与对齐（接入 §3 步骤 6）

- 源系：3DGS / LiDAR / COLMAP / episodes 同一个 COLMAP 式 **y-down** 世界系（README 写 "Y-up" 与实测不符；`transforms.json` 另是 Z-up 系，未用）。实测：图像 up 向量均值 (0.016, −0.9998, −0.007)，倾斜 1.0°；地面在世界 y≈+1.08，相机 y≈−0.2。
- 不改 PLY，旋转平移全部放 `scene.json.nav_to_world`（列 = nav 轴在世界系）：`[[0,1,0,0],[0,0,−1,1.082],[−1,0,0,0],[0,0,0,1]]`，nav x = 轨迹方向（世界 −z），nav z = up（世界 −y），`ground_z = 0`。
- 尺度：LiDAR 米制，相机离地面 p10/50/90 = 1.23 / 1.29 / 1.33 m（手持，合理）。
- **飞行高度复核（Callinstrasse P0 教训）**：沿 563 个相机位姿，nav z=2.0 离局部地面（1.5 m 内 LiDAR 点 10 分位）p10/50/90 = 1.90 / 2.00 / 2.20 m；16/563 离群点集中在 (8,−1.5) 与 (40.6,−10.7) 两处（头顶有遮挡/树冠，局部地面估计失效，不是地面真的高）。独立复核：路线旁 2 m 内低位 LiDAR 点 z p5/50/95 = −0.19 / 0.01 / 0.23 m；渲染器沿 6 条路线俯视（pitch −90°）期望深度中位数 1.96–2.16 m。结论：nav z=2 m 沿路线离地 2.0 ± 0.2 m。
- 地面平面整体倾斜 1.58°（含路面横坡），轴对齐旋转未校正；相机高度从 x 0–16 m 的 1.32 m 降到 x 48–56 m 的 1.10 m（人行道沿路下坡或持机变化，未分辨）。

## 碰撞几何

- `collision.source = "points"`（`lidar.ply`，`min_points 3`，体素 0.2 m，飞行层 1.8–2.2 m）是 scene 默认；高斯代理 v2 只作对照（见报告 `SCENE_A_20261107.md` §4）。
- 官方 `collision_mesh.ply` 由 `pcd2mesh.py --filter-vert 1.5` 从 LiDAR 建，**顶部只到 nav z≈1.64 m**（p99.9），不覆盖 2 m 飞行层；与 LiDAR 地面一致（网格−LiDAR 地面 p10/50/90 = −0.07/−0.02/−0.00 m），只用于地面交叉验证。

## 路线（nav 系，z=2.0 m，`python -m sim.3dgs.planner --scene … --json` 在 LiDAR 真值上全部 feasible / clear_ok）

| idx | name | spawn (x, y, yaw°) | waypoint (x, y) | max_steps | 航程 m | 累计转向° | 最小净空 m | 设计意图 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | r0_path_east | (4, −1.5, −15) | (16, −3.5) | 126 | 12.8 | 123 | 0.80 | 沿人行道东行，左侧店面卷帘门、右侧行道树 |
| 1 | r1_tree_row_south | (24, −5.5, −10) | (38, −8.5) | 147 | 15.2 | 131 | 0.80 | 穿第一片行道树/停车区，右侧停着的车 |
| 2 | r2_face_fence_left | (14, −3.5, −80) | (26, −5.5) | 129 | 12.8 | 160 | 1.02 | 出生正对 1.5 m 外卷帘门，左转 ~90° 再东行 |
| 3 | r3_reverse_west | (20, −4.5, 165) | (6, −2) | 145 | 15.0 | 136 | 0.85 | 反向西行（回程视角） |
| 4 | r4_face_tree_turn | (26, −4.5, 60) | (34, −7.5) | 110 | 9.6 | 261 | 0.80 | 出生斜对树干，先转向再贴树绕 |
| 5 | r5_bend_south | (44, −11, −30) | (51.5, −15.5) | 95 | 9.4 | 82 | 0.63 | 东端拐弯处，净空最小的一段 |

弃用候选：「第二片树丛 (36,−7.5)→(48,−12) 穿树干」——LiDAR 真值 2 m 层被树冠封死，A* 绕到轨迹以北 4 m 的未拍摄区。

## 渲染质量探针（沿 A* 路径每 1 m 一帧，320×240，`collect_3dgs.frame_quality`，低质阈值 lapvar<60）

| route | 帧数 | lapvar min / med | 低质帧 |
| --- | --- | --- | --- |
| 0 | 13 | 310 / 658 | 0 % |
| 1 | 15 | 314 / 862 | 0 % |
| 2 | 13 | 81 / 672 | 0 %（首帧正对卷帘门 lapvar 81，平面纹理少，不是雾） |
| 3 | 15 | 604 / 714 | 0 % |
| 4 | 10 | 491 / 820 | 0 % |
| 5 | 9 | 225 / 339 | 0 %（东端拐弯处整体偏糊，仍远高于阈值） |

已知雾区：轨迹以北 >4 m（街道/对面）和 x<0 的停车场西侧没拍到；路线都在轨迹 2 m 带内。

## 冒烟（`collect_3dgs --policy astar --dynamics cf_sitl --hold 0.5`，每路线 2 集，`tmp/agent4-scene-wanderland/import/smoke/`）

| route | seed | steps / 预算 | 到达 | 碰撞 | 最小净空 m | 低质帧 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 / 1 | 88 / 89 (126) | 2/2 | 0 | 0.73 / 0.84 | 0 | goal-stop |
| 1 | 100 / 101 | 95 / 98 (147) | 2/2 | 0 | 0.46 / 0.72 | 0 | seed 100 终点前 `backward_under_forward` 截断，已在 0.69 m 内算到达 |
| 2 | 200 / 201 | 93 / 95 (129) | 2/2 | 0 | 0.76 / 0.69 | 0 | goal-stop |
| 3 | 300 / 301 | 108 / 104 (145) | 2/2 | 0 | 1.27 / 1.29 | 0 | goal-stop |
| 4 | 400 / 401 | 69 / 78 (110) | 2/2 | 0 | 0.37 / 0.57 | 0 | 树干旁 0.37 m |
| 5 | 500 / 501 | 78 / 63 (95) | 1/2 | 0 | 0.55 / 0.56 | 0 | seed 500 在 (48.5,−13) 距终点 4 m 处 YAW_L/YAW_R 交替、FORWARD 零位移，第 77 步 `backward_under_forward` 截断 |

合计 11/12 到达、0 碰撞、0 低质帧。三视角视频 `import/smoke/route_*/ep_00000/views.webm` + 关键帧 PNG。

## 未做 / 待确认

- 没按 §4 跑 ≥20 集/路线（任务只要求 2 集冒烟）；PSNR 无（没拿 images tar，也没训练）。
- `scene.json.license` 写 "Apache-2.0 (Wanderland, ai4ce)"，`source` 写 HF 路径；进论文前需在致谢/引用里加 Wanderland（arXiv 2511.20620）。
- 地面 1.58° 倾斜与沿路 0.2 m 高差未校正；若要严格 2.0 m，可在 `scene_import_wanderland.step_align` 把 `nav_to_world` 改成拟合平面旋转。
- r5 seed 500 的停滞是 A* 教师在拐角 YAW 抖动 + cf_sitl 零位移的组合，与 Callinstrasse route 5 seed 500 同类；`planner.py`/`collect_3dgs.py` 本任务只读，没改。

## ToF 地形打标（2026-11-09，AUDIT_SIM #2，`python -m sim.3dgs.planner --scene … --json`）

口径见 `3dgs-site-garden.md` §8，这里几何是 LiDAR 点云（`source=points`，min_points 3）：`[0.3 m, z_slice 下沿 1.8 m)` 层沿 A* 路径查正下方最高占据格顶 → `tof_terrain_m`；不为 None 或路线名含 `over_` → `agl_model_unverified`（写进 manifest / eval.json，不改 ToF 输入）。

| 路线 | tof_terrain_m | agl_model_unverified | 原因 |
| --- | --- | --- | --- |
| r0_path_east | 0.5 | 是 | tof_terrain |
| r1_tree_row_south | — | 否 | |
| r2_face_fence_left | — | 否 | |
| r3_reverse_west | — | 否 | |
| r4_face_tree_turn | 0.5 | 是 | tof_terrain |
| r5_bend_south | — | 否 | |

r0/r4 脚下 0.3–0.5 m 有 LiDAR 点（路缘 / 低矮植被，0.2 m 体素所以格顶 0.5）；其余四条脚下 0.3 m 以上为空。地面 1.58° 倾斜与沿路 0.2 m 高差（上文「未做」）不在这项里，仍未校正。
