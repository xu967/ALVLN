# 3DGS 场景接入：Mip-NeRF 360 `bicycle`（Inria 预训练 30k）

- 时间：2026-11-07 06:20–07:30（SCENE-B 子 agent）
- 数据盘：`$DATA_ROOT/uav_rgb/3dgs-scenes/bicycle/{scene.json, bicycle_30k_m.ply, align.json, probe_quality/}`；原始件 `$DATA_ROOT/uav_rgb/downloads/inria-3dgs-pretrained/bicycle/`。原 `scene.json`（HF 7k mini `bicycle.ply`，`--up=-y` 翻转）备份为 `scene.json.bak-20261107-scene-b`；`bicycle_v2.json / bicycle_real.json / scene_flip*.json` 和对应 PLY 未动。
- 结论：长椅座高定尺度（0.9367 m/单位），nav z 与离地高度差 ≤ 0.05 m；6 条 9–12.5 m 路线 A* 可行、0 低质帧；冒烟结果见 §6。

## 1 来源与许可

- 照片与 COLMAP：Mip-NeRF 360 `bicycle`，194 张，公园长椅 + 自行车 + 树丛，环绕采集，相机离地 p10/50/90 = 0.60 / 1.41 / 2.54 m（比 garden 更贴地，有不少蹲拍）。
- 3DGS：Inria 预训练 `bicycle/point_cloud/iteration_30000/point_cloud.ply`（1.52 GB，6.13 M 高斯），同 garden 用 HTTP Range 抽取。本地已有的 `bicycle.ply`（141 MB）是 HF 7k mini，是同一场景的低迭代版本，不再用。
- 许可：同 garden（Mip-NeRF 360 照片 CC-BY 4.0 按 SMERF 页面声明、项目页无文字；Inria PLY 未声明、软件非商业研究许可；仅研究用途，双署名），写进 `scene.json.license`。

## 2 尺度与地面（align.json）

- 重力：COLMAP 稀疏点 RANSAC 地面法线 `[-0.0264, -0.9706, -0.2392]`（相对 −y 斜 14°），与相机 down 均值、轨道平面法线夹角 < 2°。
- 尺度：轨道中心 1.5 单位内的稀疏点高度直方图在 0.45–0.50 单位有 588 点的水平面峰 = 长椅座面；座高标准 0.43–0.46 m，取 0.45 m → `scale = 0.9367 m/单位`；靠背顶 ≈ 0.95 单位 → 0.89 m，与真实长椅靠背 0.85–0.95 m 自洽（第二个参照）。
- 烘焙方式同 garden：尺度进 PLY（`ply_scale.py`），旋转 + 平移进 `nav_to_world`，nav 原点 = 轨道中心投到地面，`ground_z = 0`；轨道半径 3.50 m。

### 验证

| 位置 (x,y) | nav z | 俯视 −89° 中心深度 m | 推出的地面高 m | 平视底行深度 m（平地期望 2.31 / 4.62 / 6.93） |
| --- | --- | --- | --- | --- |
| (−5,−3) r0 出生点 | 1 / 2 / 3 | 0.96 / 1.95 / 2.95 | 0.04 / 0.05 / 0.05 | 2.09 / 4.57 / 7.05 |
| (−3,3) 西北草坪 | 1 / 2 / 3 | 0.96 / 1.99 / 2.99 | 0.04 / 0.01 / 0.01 | 2.28 / 4.58 / 6.99 |
| (2,−4) 南侧小径 | 1 / 2 / 3 | 1.01 / 2.00 / 2.97 | −0.01 / 0.00 / 0.03 | 2.18 / 4.32 / 6.84 |

- 平地几何闭合：底行深度与 h/tan(VFOV/2) 的期望差 < 7%，三个高度都成立，说明 nav z 就是离地高度、地面也平。
- 高度阶梯（`probe_quality/ladder_*.png`）：z=1 长椅靠背齐眼，z=2 俯视长椅与车，z=3 树冠下沿。
- 训练相机复现 3 张（`views_bicycle/train_cam_*.png`，lapvar 2600–4600，无空洞）。
- 占据切片 z∈[1.8,2.2]（`views_bicycle/occ_z2.png`）：中部只有东侧树丛 nav (4..7, −2.5..1.5)（树干 + 低垂树冠），长椅（≤1.0 m）和车不在 2 m 层；外围 >8 m 处的树墙是环形占据。

## 3 scene.json

- 同 garden：camera 60°/320×240/pitch 0，collision z_slice [1.8,2.2] voxel 0.1 envelope e1_mini，`reach_radius_m` 1.0，`step_s` 0.5，顶层 spawn/waypoint = routes[0]，`max_steps` = route_check 预算。
- `python -m sim.3dgs.planner --scene …/bicycle/scene.json --json` rc=0，6 条 feasible / clear_ok。

## 4 路线（nav 系，z=2.0 m）

| idx | name | spawn (x,y,yaw°) | waypoint | 航程 m | 转向° | 最小净空 m | max_steps | 意图 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | r0_path_west_east | (−5,−3,0) | (3.5,−3.5) | 8.71 | 39 | 1.72 | 86 | 西→东沿南侧小径，长椅在左前，终点在树丛南侧 |
| 1 | r1_west_ese_past_bench | (−6,1,−20) | (3,−3.5) | 10.86 | 94 | 2.00 | 108 | 西侧出发朝东南偏东，从长椅南侧经过 |
| 2 | r2_east_north_turn_west | (5,−5,90) | (−2,4) | 11.90 | 120 | 0.92 | 118 | 东南角朝北出生，前方 3 m 是树丛，必须左转绕其南缘 |
| 3 | r3_north_south_over_bench | (2,5,−90) | (−1,−5) | 11.24 | 104 | 1.51 | 112 | 北→南从长椅东侧上方掠过，树丛在左 |
| 4 | r4_south_north_turn_west | (3,−5,90) | (−4,3) | 10.90 | 90 | 1.39 | 108 | 南侧朝北，左转向西北，长椅从右侧经过 |
| 5 | r5_se_nw_around_tree | (7,−4,135) | (−3,2) | 12.49 | 121 | 1.40 | 123 | 东南角朝西北，树丛在右前，绕其南侧后直指长椅 |

## 5 清晰度探测（`tools/scene_probe_quality.py`）

| 路线 | 帧 | 低质帧 | lapvar min / p10 / med | hole_frac max | 结果 |
| --- | --- | --- | --- | --- | --- |
| r0_path_west_east | 19 | 0 | 2246 / 2293 / 2619 | 0.009 | PASS |
| r1_west_ese_past_bench | 22 | 0 | 2399 / 2429 / 2657 | 0.009 | PASS |
| r2_east_north_turn_west | 23 | 0 | 996 / 1145 / 1414 | 0.001 | PASS |
| r3_north_south_over_bench | 23 | 0 | 1631 / 1960 / 2252 | 0.001 | PASS |
| r4_south_north_turn_west | 21 | 0 | 1214 / 1239 / 1436 | 0.001 | PASS |
| r5_se_nw_around_tree | 24 | 0 | 1124 / 1229 / 1646 | 0.000 | PASS |

- 淘汰的候选：`r1_sw_east_past_tree` 与 `r2_nw_se_around_tree` 都在 (7,−4) 朝东南结束，末段 5–6 帧正对轨道外 >4 m 的树丛，渲染成绿色雾墙，但 lapvar 仍 >1000（雾里有高频碎片）——按阈值是 PASS，按拼图不能要，换成现在的 r1/r2。
- 这个场景整体比 garden 锐（lapvar 中位 1400–2650 vs 1300–2800 但 min 更高），因为相机离地更低、轨道半径更大，2 m 高度仍在采集视角包络内。

## 6 冒烟（`collect_3dgs --policy astar --dynamics cf_sitl --episodes 2 --hold 0.5 --stop-on-collision`，seed 1000+10·r）

输出 `$DATA_ROOT/uav_rgb/3dgs-smoke-20261107/bicycle/r<i>/ep_0000{0,1}/`。

| 路线 | 集 | 步数/预算 | 碰撞 | goal-stop | 终点距航点 m | 到达 | z 范围 m | 低质帧 | lapvar min | 最小净空 m | 动力学异常 | 墙钟 s | 视频 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r0_path_west_east | 0 | 57/86 | 否 | 是 | 0.39 | 是 | 1.98–2.04 | 0 (0.0) | 1623.0 | 1.98 | - | 3.427 | 1 |
| r0_path_west_east | 1 | 63/86 | 否 | 是 | 0.44 | 是 | 1.98–2.02 | 0 (0.0) | 2057.6 | 1.72 | - | 2.379 | 1 |
| r1_west_ese_past_bench | 0 | 77/108 | 否 | 是 | 0.34 | 是 | 1.98–2.04 | 0 (0.0) | 2172.8 | 1.98 | - | 4.138 | 1 |
| r1_west_ese_past_bench | 1 | 82/108 | 否 | 是 | 0.43 | 是 | 1.97–2.01 | 0 (0.0) | 2100.3 | 1.97 | - | 2.992 | 1 |
| r2_east_north_turn_west | 0 | 83/118 | 否 | 是 | 0.33 | 是 | 1.98–2.04 | 0 (0.0) | 1021.9 | 0.89 | - | 7.845 | 1 |
| r2_east_north_turn_west | 1 | 90/118 | 否 | 是 | 0.3 | 是 | 1.97–2.01 | 0 (0.0) | 995.7 | 0.66 | - | 4.904 | 1 |
| r3_north_south_over_bench | 0 | 69/112 | 否 | 是 | 0.39 | 是 | 1.98–2.01 | 0 (0.0) | 1805.7 | 1.6 | - | 5.283 | 1 |
| r3_north_south_over_bench | 1 | 68/112 | 否 | 是 | 0.47 | 是 | 1.98–2.01 | 0 (0.0) | 1800.9 | 1.6 | - | 2.966 | 1 |
| r4_south_north_turn_west | 0 | 71/108 | 否 | 是 | 0.25 | 是 | 1.97–2.04 | 0 (0.0) | 1222.3 | 1.36 | - | 5.056 | 1 |
| r4_south_north_turn_west | 1 | 97/108 | 否 | 是 | 0.35 | 是 | 1.98–2.03 | 0 (0.0) | 1115.7 | 1.32 | - | 4.241 | 1 |
| r5_se_nw_around_tree | 0 | 96/123 | 否 | 是 | 0.4 | 是 | 1.98–2.04 | 0 (0.0) | 1128.3 | 1.32 | - | 4.714 | 1 |
| r5_se_nw_around_tree | 1 | 82/123 | 否 | 是 | 0.41 | 是 | 1.97–2.01 | 0 (0.0) | 1103.0 | 1.29 | - | 2.72 | 1 |

- 12 集：碰撞 0，到达 12/12，goal-stop 12，低质帧 0，动力学异常 0；飞行高度 1.97–2.04 m；实飞最小净空 0.66 m（r2 绕树丛南缘）。
- 三视角视频每集 1 个。

## 7 没做什么 / 风险

- 尺度参照是长椅座高 0.45 m（公园长椅 0.43–0.46），靠背 0.89 m 自洽；自行车轮径没量（车轮在稀疏点里不成平面峰）。尺度误差估计 ±4%。
- 相机离地 p10 只有 0.60 m，2 m 高度处在采集带上沿（p90 2.54 m），树冠下沿在 2.5–3 m，所以 z_slice [1.8,2.2] 的树丛占据主要是树干 + 低垂枝，3 m 以上没验证。
- 旧的 `scene.json.bak-20261107-scene-b`（7k mini + 轴翻转）保留着没删；`bicycle_v2.json/bicycle_real.json/scene_flip*.json` 指向的 7k PLY 也没删（共 0.7 GB），是否清理由用户定。
