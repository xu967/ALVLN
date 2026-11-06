# FAST Lab 仿真/控制侧精读 → `forest_e1` 场景族（E1 MINI 随机圆柱森林）

- 时间：2026-11-07 01:15
- 范围：`sim/tools/scene_forest.py`（新）、`sim/tools/test_scene_forest.py`（新）；报告 `/root/rivermind-data/uav_rgb/tmp/phase2-20261106/reports/FASTLAB_B_sim_20261107.md`
- 结论：完成（场景族 + 自检 + 统计 + 单测 + 报告）；接入采集/审计链和教师圆柱支持属其它文件域，只写了改法

## 做了什么

- `sim/tools/scene_forest.py`：按 seed 确定性生成随机圆柱森林，三档最窄缝 0.45/0.35/0.28 m（任意两柱面间距的硬契约），半径 0.05–0.15 m，森林区 x∈[1.5,5.5]、y∈[−1.5,1.5]，高 3 m。每个场景用 2.5 cm 栅格 A*（障碍按圆 + E1 半径 0.102 + 0.025 m 精确膨胀）保证出生点→(6,0) 有路，并要求 y=0 直线上至少一根柱（否则盲直飞能过）。版本 `uavrgb.forest_e1.v1` 独立，meta 显式写 `split_salt`；每根柱同时给 `radius` 与外接方框 `size`，旧的 gid/方框教师直接能读。复用 `scene_gen` 的模板标记、外观 v1、sha256、原子写；载具 pose 自己匹配 `x3|e1_mini`。子命令：`--seed/--density/--out`、`--self-check`、`--stats-out`。所有常量在顶部配置区并带理由。
- `sim/tools/test_scene_forest.py`：4 seed × 3 档：确定性、缝宽/半径/区域契约、直线上有柱、走廊端点/净空/瓶颈、版本与 split 复算、载具 pose、SDF 柱数与半径字符串、XML 合法性（E1 模板自身注释含 `--` 过不了 expat 时只解析柱段）、`teacher._point_blocked` 兼容、12 个 gid 两两不同、legacy/v1 外观几何一致、横墙负例判无路、未知档位报错。
- 20 seed × 3 档统计与 3 张俯视 PNG：`/root/rivermind-data/uav_rgb/tmp/agent-fastlab-b/forest_stats/`，副本在 `reports/fastlab_b_forest/`。
- 精读并写进报告（file:line）：EGO-Planner-v2 `map_generator/random_forest_sensing.cpp`、`launch_gen.py`、`mockamap/maps.cpp`、`so3_quadrotor_simulator`、`local_sensing/depth_render.cu`；Fast-Drone-250 `px4ctrl` FSM / `estimateThrustModel` RLS / 超时处理；Ground-effect-controller README 标定法；OPNet 文件树与训练脚本头部。报告含给 `sim/cf_sitl/exec/` 的 7 条 FSM/悬停油门建议（S1–S7），只建议不改。
- RSA 饱和标定（临时脚本 `agent-fastlab-b/calib_rsa.py`，未入库）：4×3 m²、半径均值 0.10 m 时按三档缝宽顺序吸附到预算用尽，10 seed 均值 25.4 / 33.6 / 44.5 根；`DENSITY_COUNT` 取其 55–73%。

## 没做什么

- 没改 `sim/tools/scene_gen.py`（`--task forest_e1` 分发）、`collect.py`、`audit_dataset.py`、`teacher.py`/`teacher_kino.py`（认 `radius`）、`sim/cf_sitl/exec/`
  - 原因：不在本任务的文件域（任务明确只碰新场景族 + docs，FAST-A/SITL 文件不能动）；改法写在报告 §4/§5
- 没起 Gazebo 载入生成的 SDF
  - 原因：任务禁止；用 `xml.etree` 验了 X3 模板输出整篇合法、E1 模板输出的柱段合法
- 没提交 `sim/worlds/uav_rgb_e1.sdf`、`sim/dynamics/e1_mini.json` 之外别人的文件
  - 原因：E1 模板是 GZ agent 的未提交文件，只复制到工作树跑测试；`e1_mini.json` 已由 FAST-A 的提交带入分支
- 没做柱倾斜、高度随机、悬挂障碍等 3D 变化
  - 原因：超出范围（任务只要圆柱森林；4 动作策略在固定高度飞）；要做必须 bump `forest_e1.v2`
- 没测 `teacher_kino` 在森林上的到达率
  - 原因：FAST-A 的文件；且教师认 `radius` 前结果只反映方框近似的保守
- OPNet / e2e_opt 没克隆完整
  - 原因：直连超时（25 MB / 213 MB）；用 GitHub API 读了文件树、README、训练脚本头部；e2e_opt 只是项目网页

## 待确认

- 三档缝宽按任务原文 0.45/0.35/0.28 取；按 `e1_mini.json` 外接圆 0.204 m 是 2.2×/1.7×/1.4× 而不是 2.5×/2×/1.5×。dense 单侧余量 3.8 cm，要不要再加一档 0.40 作为训练主力档，由采集负责人定。
- 森林数据进采集链前，`audit_dataset.py` 需要森林分支（报告 R3），否则 `geometry_version != SPLIT_SALT` 直接 FAIL。
- `sim/worlds/uav_rgb_e1.sdf` 注释里的 `--` 让 expat 解析失败（Gazebo 的 tinyxml2 宽容），建议 GZ agent 改掉（报告 R5）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python sim/tools/test_scene_forest.py` | 通过（E1 模板；打一条 note 说明模板自身非严格 XML） |
| `python sim/tools/scene_forest.py --self-check`（30 seed × 3 档 = 90 场景） | 通过：同 seed 字节级一致；min_gap ≥ 档位值；走廊净空 ≥ 0.127、瓶颈 ≥ 档位值；split 复算一致；重采最多 2 次 |
| `python sim/tools/scene_forest.py --stats-out …`（20 seed × 3 档） | 通过，8 s；sparse/medium/dense 柱数 15.9/22.7/29.2，最窄缝 min 0.451/0.350/0.280，路径 6.68/6.74/6.80 m（直线 6.5），瓶颈 min 0.453/0.351/0.282 |
| CLI 生成 `--seed 7 --density dense`（E1 模板）与 `--seed 7 --density sparse --template uav_rgb.sdf`（X3） | 通过；X3 输出整篇 `xml.etree` 可解析，E1 输出 28 根 `trunk_*` 模型、载具 pose 替换正确，x3 模板给出警告 |
| `UAV_RGB_PYTHON=<主仓库 venv> bash tools/check.sh --python-only` | 通过（含 test_scene_forest、FAST-A 的 test_teacher_kino） |
| 100 seed/档 split 分布 | sparse 78/11/11、medium 80/12/8、dense 86/3/11（train/val/test），重采 ≤2 次 |
| Gazebo 载入 | 未跑（任务禁止） |
| 盲直飞 / 0.182 m 判据走廊探针（临时脚本） | 三档盲直飞 20/20 撞；0.182 m 判据走廊 20/20、20/20、18/20 |

## 风险与遗留

- 方框教师在 medium/dense 上偏保守 ≤6 cm，dense 对现有教师接近死局；需 `_boxes()` 认 `radius`（报告 R1）。
- 栅格 A* 判"有路"是几何可达，不含偏航/速度动力学；E1 0.3 m/s、一窗 0.27 m 下应当够，提速后要用 `teacher_kino` 复核。
- 场景只保证"至少一条路"；dense 档路径/直线只有 1.047，若希望更多绕行决策应加密度档或缩窄走廊带，不要加宽森林。
- `GEOMETRY_VERSION_FOREST` 下任何配置区几何常量改动必须 bump 版本（与旧柱阵同规矩）；外观常量 `TRUNK_RGB` 不进 gid。
