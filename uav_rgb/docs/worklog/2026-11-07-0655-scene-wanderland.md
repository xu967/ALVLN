# 公开 3DGS 场景 Wanderland 接入 + 碰撞代理 v3（LiDAR/网格真值）

- 时间：2026-11-07 06:13–06:55（SCENE-A 子 agent，worktree `tmp/wt-scene-wanderland`，分支 `scene-wanderland`，**未 commit**）
- 范围：`sim/3dgs/collision.py`、`sim/3dgs/test_collision.py`（v3）；新 `tools/scene_import_wanderland.py`、`tools/test_scene_import_wanderland.py`、`tools/collision_truth_compare.py`；数据盘 `3dgs-scenes/wanderland_1MlUucug/`、`downloads/wanderland/`；文档 `docs/experiments/3dgs-site-wanderland.md`
- 结论：Wanderland（Apache-2.0，335 公开场景，HF 无 gating）场景 1MlUucug 接入完成，nav z=2 m 沿路线离地 2.0±0.2 m 经 LiDAR 地面 + 相机高度 + 渲染俯视深度三方复核；6 条路线在 LiDAR 真值上 feasible、渲染低质帧 0%、astar 冒烟 11/12 到达 0 碰撞。高斯代理 v2 在这个场景对 LiDAR 真值的 0.2 m 覆盖率 98.7%、误报 5.1%（Callinstrasse 报告的 24% 是 COLMAP 初始化手机场景的特例，不是代理方法本身的上限）；默认 opacity_min/sigma_k 不改。报告：`tmp/phase3-20261107/reports/SCENE_A_20261107.md`。

## 做了什么

- **A 调研**：HF `ai4ce/wanderland`：Apache-2.0；manifest v1.3 335 场景（v1 293 = showcase 49 / evaluation_ready 113 / training_ready 131；v2 showcase 42）；v1 每场景 `3dgs.ply`（gsplat 训练，INRIA 布局，直接可读，`splat_to_ply.py` 不用改）、`raw_pcd.ply`（LIV-SLAM 米制 LiDAR 稠密云）、`scene.usdz`（Isaac Sim 包，内含 ASCII `scene.usda` 碰撞网格 + 3DGRUT usdz）、`episodes.json`（导航 episode，相机高度）、`sparse/0`（`points3D.bin` 空）、`transforms.json`（另一个 Z-up 系）、images/fisheye tar；v2 showcase 为 `scene.ply` 网格 + `lidar/*.las`，无 episodes、Z-up。单场景 0.6–2.3 GB。下载方式：`huggingface_hub.hf_hub_download(repo_type="dataset")` 按文件取（官方 `download.py` 也是这条路）。选 1MlUucug（evaluation_ready、无质量标记、113 个 eval 场景里轨迹最平、52 m 商业街人行道：卷帘门/行道树/停车/垃圾桶），下了 8 个文件 1.71 GB（守护进程，`DONE`）。
- **B 接入** `tools/scene_import_wanderland.py`（顶部配置区；`IMPORT_STEPS` 选步；每步写 json）：
  - download：`hf_hub_download` 4 并行，校验 HF 大小；
  - convert：`scene.ply`/`lidar.ply` 硬链接原文件；从 `scene.usda` 文本解析 Collider 网格（`usda_mesh_to_arrays`，不依赖 pxr）写 `collision_mesh.ply`；
  - align：COLMAP `images.bin` 相机 up 向量均值 → 世界 up = −y（轴对齐）；nav x = 轨迹方向；`ground_z` = 相机脚下 1.5 m 内 LiDAR 高度 10 分位的轨迹中位数 → `nav_to_world`；验证相机离地 1.23/1.29/1.33 m、nav z=2 离局部地面 1.90/2.00/2.20 m、网格 vs LiDAR 地面 −0.07/−0.02/−0.00 m、网格顶 1.64 m；PIL 顶视图 `map_lidar_nav.png`；
  - scene：写 `scene.json`（schema v1，camera 60°/320×240，collision `source=points`/`geometry=lidar.ply`/`min_points=3`/voxel 0.2/z_slice 1.8–2.2/envelope e1_mini，6 条 routes，`max_steps=route_check.steps_estimate`）；任一路线不可行即报错；`map_routes_truth.png`；
  - probe：沿 A* 路径每 1 m 渲染 320×240，`collect_3dgs.frame_quality`，低质帧 ≥10% 报错；加俯视（pitch −90°）期望深度中位数作独立高度复核（1.96–2.16 m）；`probe_r*.png` 联络表；
  - smoke：`collect_3dgs --policy astar --dynamics cf_sitl --hold 0.5 --stop-on-collision` 每路线 2 集，复用 `tools/demo_3dgs_routes.episode_row` 汇总 → `smoke_report.json`；
  - video：`sim.3dgs.video` 三视角 6 集（`views.webm` + 关键帧）。
  - 路线：原 r2「第二片树丛穿树干」在 LiDAR 真值上 2 m 层被树冠封死、A* 绕进未拍摄区，改为「正对卷帘门左转」。
- **C 碰撞代理 v3** `sim/3dgs/collision.py`：`read_ply_vertices`（binary LE/BE/ascii，xyz 任意数值类型，三角面，stride）、`sample_mesh_surface`（面上按间距撒点）、`Occupancy(..., min_points, source)`（纯点云路径按体素计数过滤噪点；`proxy_version` 按 source 取 `3dgs.occ.v2` / `3dgs.occ.v3.points` / `3dgs.occ.v3.mesh`）、`from_points` / `from_mesh`、`from_scene` 读 `collision.source|geometry|min_points|geometry_stride` 并可用参数覆盖；`summary()` 加 `source/geometry/min_points`。`collect_3dgs`/`planner`/`video` 不改即切到真值（它们只调 `from_scene`）。`test_collision.py` +8 用例（点云墙、min_points 滤噪、仅纯点云可设 min_points、飞行层外点忽略、y-down `nav_to_world`、网格撒点 vs 稀疏顶点、PLY 三种格式回读/抽稀/带面拒抽稀、`from_scene` source 切换与非法值），24/24 通过。
- **C 对照** `tools/collision_truth_compare.py`：同网格参数下高斯代理 vs LiDAR 真值的 2D 飞行层投影（只统计离相机轨迹 6 m 内的格）：覆盖率@0/0.2/0.4 m、误报率@0/0.2/0.4 m、每路线 `route_check` 两边对比 + 代理路径在真值上 sweep / 真值路径在代理上 sweep；叠加图 `overlay_truth_vs_proxy.png`；`--sweep` 扫 opacity_min×sigma_k 6×4。结果与默认值结论见报告 §4。
- **测试** `tools/test_scene_import_wanderland.py`（usda 解析/四边形拒绝/网格 PLY 回读/y-down nav 轴/局部地面分位），5/5 通过。
- **文档**：`docs/experiments/3dgs-site-wanderland.md`（§4 口径：来源/许可/对齐/路线/探针/冒烟/未做）、本 worklog、`tmp/agent4-scene-wanderland/PROGRESS.md`。

## 没做什么（及原因）

- 没改 `scene.py / renderer.py / collect_3dgs.py / planner.py / video.py`（只读域）。需要主线处理的两点写在报告 §6：① `collect_3dgs` 的 `collision_proxy` 字段名在真值源下仍叫 proxy（manifest 里 `proxy_version` 已能区分）；② r5 seed 500 / r1 seed 100 的 `backward_under_forward` 截断是 A* 教师拐角 YAW 抖动 + cf_sitl 零位移，与 Callinstrasse route 5 同类。
- `splat_to_ply.py` 未扩展：`3dgs.ply` 已是 INRIA 布局（opacity logit、scale log、f_rest [3][K]），直接读。
- Callinstrasse 的代理 vs 真值对照没做：`3dgs-scenes/callinstrasse/colmap.zip` 里没有 `points3D.bin`（只有 cameras/images），没有稀疏云可用；需要主线重跑 COLMAP 点三角化或提供 `points3D`。
- 官方碰撞网格没用作飞行层真值：`pcd2mesh.py --filter-vert 1.5` 把网格裁到相机上下 1.5 m，顶 ≈1.64 m，2 m 层空。`from_mesh` 路径靠合成墙单测 + 网格地面交叉验证。
- 没改 `DEFAULT_OPACITY_MIN / DEFAULT_SIGMA_K`：sweep 显示 (0.3, 1.5) 在覆盖率/误报率拐点上（见报告），单场景证据不足以动 Callinstrasse 也在用的默认值。
- 地面 1.58° 倾斜与沿路 0.2 m 高差未校正（轴对齐旋转），nav z=2 实际 1.8–2.2 m。
- 没跑 ≥20 集/路线 pilot、没算 PSNR（没下 images tar）、没下 v2 场景。
- 没 commit / push。

## 待确认

- `tools/collision_truth_compare.py` 不在任务列出的可写文件里（任务要「对照脚本」，放 tools/ 新文件最省事）；若主线希望并进 `scene_import_wanderland.py` 或 `sim/3dgs/`，告知即可。
- scene 默认 `collision.source=points`：采集 manifest 的 `collision_proxy.proxy_version` 会变成 `3dgs.occ.v3.points`，下游 `preprocess`/数据卡若按 `3dgs.occ.v2` 字符串过滤需要确认。
- Wanderland 引用/致谢：arXiv 2511.20620，Apache-2.0 要求保留 LICENSE/NOTICE。

## 验证到哪一步

- 单测：`python -m unittest sim.3dgs.test_collision`（24 OK）、`tools.test_scene_import_wanderland`（5 OK）；`collision._selftest()` PASS。
- `planner --json` 6 路线 feasible/clear_ok；probe 6 路线低质 0%；smoke 12 集 11 到达 0 碰撞；video 6 集生成。
- `sim/3dgs/test_collect_eval.py` 15 OK（mock 渲染，高斯路径，`PROXY_VERSION` 常量不变）；`planner` selftest PASS；`planner --scene wanderland --route 0..5 --json` rc=0。
- 没跑 `tools/test_splat_to_ply.py`（没改）、`RUN_3DGS_REAL` 实渲染测试。

## 风险与遗留

- `min_points=3` 对 LiDAR 稠密云合适；COLMAP 稀疏云必须在 scene.json 里写 `min_points: 1`，否则整张图空（`from_points` 不自动判断）。
- LiDAR 真值含行人/车辆等动态物（采集时在场），2 m 层主要是树冠/招牌/卷帘门，影响小，但没有逐格核对。
- 渲染东端 (x>44) 整体偏糊（lapvar 225–460），r5 的 2 m 帧勉强；若主线要「清晰度分级」，这段应标低档。
