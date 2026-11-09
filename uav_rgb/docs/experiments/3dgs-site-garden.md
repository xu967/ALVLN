# 3DGS 场景接入：Mip-NeRF 360 `garden`（Inria 预训练 30k）

- 时间：2026-11-07 06:20–07:30（SCENE-B 子 agent）
- 数据盘：`$DATA_ROOT/uav_rgb/3dgs-scenes/garden/{scene.json, garden_30k_m.ply, align.json, probe_quality/}`；原始件 `$DATA_ROOT/uav_rgb/downloads/inria-3dgs-pretrained/garden/{point_cloud_30000.ply, cameras.json, cfg_args, input.ply}`
- 结论：尺度/重力/地面用 COLMAP 稀疏点 + 桌面已知高定出，nav z 与离地高度之差 ≤ 0.1 m（俯视深度实测）；6 条 8–11 m 路线全部 A* 可行、沿路 0 低质帧；冒烟（astar + cf_sitl，每路线 2 集）结果见 §6。

## 1 来源与许可

- 照片与 COLMAP：Mip-NeRF 360（Barron et al. CVPR 2022）`garden`，185 张，室外桌子+花盆，环绕采集，相机离地 p10/50/90 = 1.18 / 1.68 / 2.31 m（对齐后算出，手持高度自洽）。
- 3DGS：Inria/MPII gaussian-splatting 官方预训练包 `models.zip`（14.66 GB）里的 `garden/point_cloud/iteration_30000/point_cloud.ply`（1.45 GB，5.83 M 高斯，SH 3 阶）。包无单场景链接，但服务器支持 HTTP Range，`tmp/agent4-scene-mipnerf/remote_zip.py` 只读中央目录后按成员抽取，没下整包。
- 许可（写进 `scene.json.license`）：Mip-NeRF 360 项目页本身没有任何许可文字（2026-11-07 核对），CC-BY 4.0 是同一作者组在 SMERF 页面声明的（「Both datasets and checkpoints are released under CC-BY 4.0」）；Inria 预训练 PLY 没有单独声明许可，gaussian-splatting 软件是非商业研究许可。口径：仅研究用途，署名 Barron 2022 + Kerbl 2023。

## 2 尺度与地面（align.json）

Mip-NeRF 360 的 COLMAP 世界系是任意的：garden 的重力方向相对 −y 轴斜 29°，`bicycle_v2.json` 那种 `--up=-y` 轴对齐翻转会让地面斜着。做法：

1. 重力：对 `input.ply`（COLMAP 稀疏点）RANSAC 拟合地面平面，法线 `[-0.0103, -0.8767, -0.4810]`；与「相机 down 轴均值」「相机轨道平面法线」三者两两夹角 < 2°。
2. 尺度：以地面法线为高度轴，取轨道中心 1.5 单位内的稀疏点，高度直方图在 1.30–1.40 单位有一个 8035 点的水平面峰，这就是桌面；桌面标准高 0.72–0.76 m，取 0.75 m → `scale = 0.5550 m/COLMAP 单位`。
3. 烘焙：尺度烘进 PLY（`tmp/agent4-scene-mipnerf/ply_scale.py`：`xyz *= s`，`scale_0..2 += ln s`，球谐 `f_rest` 原样，所以 PLY 里不旋转）；旋转 + 平移写进 `nav_to_world`（刚体，`collision._rigid_parts` 不接受缩放/镜像）。nav 原点 = 相机轨道中心投到地面，`ground_z = 0`，nav +x = 世界 +x 在水平面的投影。
4. `nav_to_world` = `[[0.999947,0,-0.010304,0.029118],[-0.009033,-0.48102,-0.876663,1.545754],[-0.004956,0.87671,-0.480994,0.852108],[0,0,0,1]]`。

### 验证（nav z=2 m 是否像离地 2 m）

| 位置 (x,y) | nav z | 俯视 −89° 中心深度 m | 推出的地面高 m | 平视底行深度 m（平地期望 h/tan 23.4°） |
| --- | --- | --- | --- | --- |
| (−3.5,−2) 西侧草坪（r0 出生点） | 1 / 2 / 3 | 0.99 / 1.99 / 3.00 | 0.01 / 0.01 / 0.00 | 2.12 / 4.48 / 5.99（期望 2.31 / 4.62 / 6.93；底行前方有矮植被所以偏近） |
| (2,−2) 桌子东南 | 1 / 2 / 3 | 0.91 / 1.91 / 2.90 | 0.09 / 0.09 / 0.10 | 底行被 0.5 m 外的花盆/树干挡住，不适用 |
| (−2,3) 桌子西北 | 1 / 2 / 3 | 0.96 / 1.95 / 2.94 | 0.04 / 0.05 / 0.06 | 2.16 / 4.37 / 6.26 |

- 俯视深度在 z=2 处给出桌面 0.74 m（标定 0.75），草坪 0.1–0.2 m（草的高度）。
- 高度阶梯图（`probe_quality/ladder_*.png`，z=1/2/3 + z=2 俯视 30°）：z=1 桌面齐眼、z=2 俯视桌面与花盆、z=3 进树冠下沿，差 1 m 肉眼可辨。
- 训练相机复现（`tmp/agent4-scene-mipnerf/views_garden/train_cam_*.png`）：用 `cameras.json` 的 3 个位姿渲染，桌子居中、地平线位置与手持 1.2–2.3 m 相符。
- 占据切片 z∈[1.8,2.2]（`views_garden/occ_z2.png`）：中部只有树（nav (2.3..4.7, −1.5..2.5)）与四周墙/篱笆；桌面 0.75 m、花盆 ≤1.4 m 不在飞行高度层，所以 2 m 高度的中部障碍只有这棵树。

## 3 scene.json

- `camera` 60° HFOV / 320×240 / pitch 0；`collision` z_slice [1.8,2.2]、voxel 0.1、envelope e1_mini；`reach_radius_m` 1.0；`step_s` 0.5；顶层 `spawn/waypoint/max_steps` = routes[0]。
- `max_steps` = `planner.route_check` 步数预算（cf_sitl 每 0.5 s 步 0.15 m / 17°，×1.25）。
- `python -m sim.3dgs.planner --scene …/garden/scene.json --json` rc=0，6 条全部 feasible / clear_ok。

## 4 路线（nav 系，z=2.0 m）

| idx | name | spawn (x,y,yaw°) | waypoint | 航程 m | 转向° | 最小净空 m | max_steps | 意图 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | r0_south_lane_east | (−3.5,−2,0) | (4,−3.5) | 8.12 | 78 | 1.12 | 84 | 南侧直线向东，树在左手，终点离房屋南墙 1.5 m |
| 1 | r1_sw_ne_around_tree | (−3,−4,45) | (6,1) | 11.31 | 184 | 0.60 | 118 | 直线会撞树，A* 绕树南侧再北上 |
| 2 | r2_nw_se_past_table | (−3,3,−45) | (4,−3.5) | 9.69 | 18 | 0.71 | 93 | 西北→东南斜穿，擦树西南缘到房屋前 |
| 3 | r3_north_south_over_table | (0,4,−90) | (0,−5) | 9.00 | 0 | 1.22 | 85 | 北→南直线飞越桌面上方 |
| 4 | r4_east_return_west | (6,−1,90) | (−2,3) | 11.15 | 246 | 0.60 | 122 | 东侧朝北出生，左转绕树回桌子西北 |
| 5 | r5_west_edge_north | (−3,−5,90) | (−2,4) | 9.41 | 77 | 0.70 | 95 | 沿草坪西缘北行，桌子在右手 |

## 5 清晰度探测（`tools/scene_probe_quality.py`，每 0.5 m 一帧，阈值 = `collect_3dgs.FRAME_LAPVAR_MIN` 60，门槛 low_frac ≤ 10%）

| 路线 | 帧 | 低质帧 | lapvar min / p10 / med | hole_frac max | 结果 |
| --- | --- | --- | --- | --- | --- |
| r0_south_lane_east | 17 | 0 | 636 / 793 / 1341 | 0.25 | PASS |
| r1_sw_ne_around_tree | 22 | 0 | 1574 / 1635 / 2110 | 0.29 | PASS |
| r2_nw_se_past_table | 19 | 0 | 619 / 725 / 1626 | 0.30 | PASS |
| r3_north_south_over_table | 19 | 0 | 635 / 1006 / 1551 | 0.29 | PASS |
| r4_east_return_west | 21 | 0 | 1139 / 1342 / 2774 | 0.33 | PASS |
| r5_west_edge_north | 20 | 0 | 1299 / 1907 / 2658 | 0.28 | PASS |

- hole_frac 0.25–0.33 是天空（garden 没有天空高斯，alpha<0.05 记作空洞），不是重建空洞；bicycle 的 hole_frac < 0.01 是因为树冠盖住了天空。
- 淘汰过的候选：g0/g1 出生点 (−5,−3)/(−5,−4) 在西墙占据里（A* 无路径）；g2 终点贴东墙的帧 lapvar 300–1600 却是漂浮碎片——lapvar 量锐度不量正确性，拼图必须人看。拼图：`probe_quality/route_*.png`。

## 6 冒烟（`collect_3dgs --policy astar --dynamics cf_sitl --episodes 2 --hold 0.5 --stop-on-collision`，seed 1000+10·r）

输出 `$DATA_ROOT/uav_rgb/3dgs-smoke-20261107/garden/r<i>/ep_0000{0,1}/`（含 `trace.jsonl`、`views.webm` 三视角、`views_key_*.png`）。

| 路线 | 集 | 步数/预算 | 碰撞 | goal-stop | 终点距航点 m | 到达 | z 范围 m | 低质帧 | lapvar min | 最小净空 m | 动力学异常 | 墙钟 s | 视频 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r0_south_lane_east | 0 | 50/84 | 否 | 是 | 0.43 | 是 | 1.98–2.04 | 0 (0.0) | 633.8 | 1.17 | - | 3.052 | 1 |
| r0_south_lane_east | 1 | 61/84 | 否 | 是 | 0.44 | 是 | 1.98–2.02 | 0 (0.0) | 705.0 | 0.95 | - | 2.078 | 1 |
| r1_sw_ne_around_tree | 0 | 85/118 | 否 | 是 | 0.44 | 是 | 1.98–2.04 | 0 (0.0) | 1545.1 | 0.68 | - | 5.649 | 1 |
| r1_sw_ne_around_tree | 1 | 76/118 | 否 | 否 | 1.85 | 否 | 1.98–2.0 | 0 (0.0) | 1550.9 | 0.62 | backward_under_forward@75 | 4.283 | 1 |
| r2_nw_se_past_table | 0 | 67/93 | 否 | 是 | 0.43 | 是 | 1.98–2.03 | 0 (0.0) | 667.3 | 0.69 | - | 3.251 | 1 |
| r2_nw_se_past_table | 1 | 63/93 | 否 | 是 | 0.41 | 是 | 1.98–2.01 | 0 (0.0) | 627.4 | 0.81 | - | 2.468 | 1 |
| r3_north_south_over_table | 0 | 66/85 | 否 | 是 | 0.48 | 是 | 1.98–2.01 | 0 (0.0) | 748.2 | 1.31 | - | 3.44 | 1 |
| r3_north_south_over_table | 1 | 65/85 | 否 | 是 | 0.36 | 是 | 1.98–2.01 | 0 (0.0) | 719.2 | 1.32 | - | 2.294 | 1 |
| r4_east_return_west | 0 | 80/122 | 否 | 是 | 0.4 | 是 | 1.98–2.04 | 0 (0.0) | 1061.3 | 0.26 | - | 7.991 | 1 |
| r4_east_return_west | 1 | 89/122 | 否 | 是 | 0.49 | 是 | 1.98–2.03 | 0 (0.0) | 1253.5 | 0.17 | - | 7.266 | 1 |
| r5_west_edge_north | 0 | 70/95 | 否 | 是 | 0.49 | 是 | 1.98–2.03 | 0 (0.0) | 1486.3 | 0.65 | - | 3.994 | 1 |
| r5_west_edge_north | 1 | 60/95 | 否 | 是 | 0.4 | 是 | 1.98–2.0 | 0 (0.0) | 1241.6 | 0.65 | - | 2.074 | 1 |

- 12 集：碰撞 0，到达 11/12，goal-stop 11，低质帧 0，动力学异常 1（r1 seed 1011 第 75 步 backward_under_forward：连续 3 步 YAW_RIGHT 后首个 FORWARD 窗口机体沿新航向后退 5.5 cm，cf_sitl 的转向后漂移，与场景无关；Callinstrasse 也出现过；该集 physics_verified=False，preprocess 会拒收）。
- 飞行高度 1.97–2.04 m（cf_sitl 定高）。r4 实飞最小净空 0.17–0.26 m（planner 沿规划路径算的是 0.60）：246° 大转向时纯追踪切了树西侧的弯，离碰撞阈值不远，pilot 规模采集时要盯 r4 的碰撞率，必要时把 (6,−1) 出生点往东挪 0.5 m。
- 三视角视频 `ep_*/views.webm` + `views_key_*.png` 每集 1 个；每路线 2 集全部有。

## 7 没做什么 / 风险

- 没有下载 Mip-NeRF 360 的原始照片（360_v2.zip 12.5 GB 超预算），训练相机复现只对比了「像不像手持照片」，没有和真实照片逐像素比。
- 尺度只用桌面一个参照（0.75 m 标准高，实物可能 0.72–0.78 m → 尺度误差 ±4%）；没有第二个已知尺寸交叉验证。
- 飞行高度沿用项目口径 2.0 m（在采集高度带 1.2–2.3 m 内）。1 m 高度会把桌子/花盆变成障碍、路线更有意思，但相机轨道下沿只到 1.18 m，1 m 以下视角外推，没试。
- 占据代理 v2 只看 opacity ≥ 0.3 的椭球，花盆的细叶、遮阳伞边缘可能漏掉；清晰但有漂浮碎片的外围（东墙、北侧棚屋 >4 m 处）路线都避开了。

## 8 ToF 地形打标（2026-11-09，AUDIT_SIM #2，`python -m sim.3dgs.planner --scene … --json`）

SITL 的 ToF 是平地模型（`(z − ground_z)/cos`），飞行层以下的桌面/花盆在仿真里不存在；真机 ToF 到桌面上方会读短约 0.75 m、执行器按「离地 2 m」爬升再降回，画面高度与俯仰在这几步与仿真不同。只打标不改 ToF：`planner.route_check` 用同一份高斯在 `[0.3 m, z_slice 下沿 1.8 m)` 再切一层占据，沿 A* 路径查正下方那一列最高占据格的格顶 → `tof_terrain_m`；它不为 None 或路线名含 `over_` → `agl_model_unverified`，写进每集 manifest / eval.json。

| 路线 | tof_terrain_m | agl_model_unverified | 原因 |
| --- | --- | --- | --- |
| r0_south_lane_east | 1.1 | 是 | tof_terrain |
| r1_sw_ne_around_tree | 1.4 | 是 | tof_terrain |
| r2_nw_se_past_table | 1.3 | 是 | tof_terrain |
| r3_north_south_over_table | 1.3 | 是 | tof_terrain, route_name |
| r4_east_return_west | 1.4 | 是 | tof_terrain |
| r5_west_edge_north | 0.7 | 是 | tof_terrain |

六条全部打标：桌面 0.75 m 之外，花盆/灌木（≤1.4 m）和树的低枝都在 0.3–1.8 m 层里，garden 的中部没有一条路线脚下是空的。所以 garden 的集都不算「2 m AGL」口径，只当外观数据；要当等高飞行数据得给 SITL 加「机体正下方地面高度」输入（AUDIT_SIM #2 的完整修法，未做）。
