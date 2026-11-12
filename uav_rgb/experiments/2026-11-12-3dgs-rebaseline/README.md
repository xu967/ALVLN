# 3DGS 碰撞口径重定基线（`2026-11-12-3dgs-rebaseline`，任务卡 S1b 定稿）

## 背景与定稿碰撞口径

交叉审核 B（`review-b.md`）与第二轮审核 J（`review2-j.md`）指出旧 3DGS 碰撞判定（`collision_model = "v1"`）在 0.1 m / 0.2 m 体素下存在十字核斜向漏判、层边界跳层与只扫直线弦三项问题；S1b 第一轮的未定稿 `v2` 未出过任何正式数字，`v2` 名称与 `+coll.v2` 后缀直接沿用给本轮定稿口径：
- **连续坐标碰撞判定**（`collides`、`sweep`、`sweep_path`、规则臂 `occ_columns`）：机体视为圆柱（半径 = `body_radius + inflate`，半高 = `body_half_height`），原始占据每个体素为实心方块，圆柱与任一方块相交即碰，判定不再加高度余量；每步碰撞沿 SITL 窗口内 10 Hz `trace.jsonl` 折线逐段扫。
- **规划硬栅格与出生检查**（按路线自己的名义出生高度 `spawn.z` 取整集高度包络）：由 16 个已完成来源（1,824 集）实测极值定出 `Z_ENVELOPE_DOWN_M = 0.035 m`、`Z_ENVELOPE_UP_M = 0.075 m`、`SPAWN_DRIFT_MARGIN_M = 0.105 m`，使规划与出生检查看到的障碍层成为全集碰撞判定可能查到的层的超集。
- **规则深度带独立说明**：`sim/3dgs/rule_oracle.depth_columns` 的 `BAND_MARGIN_M = 0.10`（`±0.128 m`）是深度规则策略自身的参数，不随碰撞口径变化。

## 套件、臂与规模

- **3 个路线集 × 2 个碰撞口径 = 6 个套件**（缺省 `SPLIT_FROM_IDENTITY=true`，按几何哈希写真实 `split` 与 `diag-train-scene`）：
  - `indoor40_v1`、`indoor40_v2`：K1 / W1-11 / W1-12 室内正式 40 条 × 3 集 = 120 集/臂（`seed = 8000 + 100·路线 + k`，扰动 ±0.2 m）
  - `holdout_v1`、`holdout_v2`：K3 室内留出 32 条 × 3 集 = 96 集/臂（`seed = 5000 + 100·路线 + k`，扰动 ±0.2 m）
  - `outdoor_v1`、`outdoor_v2`：室外 4 场景 × 6 条 × 5 集 = 120 集/臂（`seed = 7000 + 100·路线 + k`，扰动 ±0.5 m）
- **每个套件 6 个主臂 + `v2` 套件 1 个诊断臂**（合计 `336 × 6 + 336 × 7 = 4,368` 集）：
  1. `teacher-tol`：占据 A* 教师，航向容差 18.9076°（`ray`，同 W1-11）
  2. `teacher-swept`：占据 A* 教师，航向容差 18.9076° + `teacher_clearance = "swept"`（同 W1-12 臂 A）
  3. `forward+route`：盲前进 + 路线方位护栏 45°（同 K1 / K3）
  4. `rule-occ+route`：规则 v1，占据列距离，`r_safe = 0.25, d_look = 1.2`（同 K1 / K3）
  5. `rule2-occ+route`：规则 v2 选中变体 `v2f`，占据列距离（同 K3 主臂）
  6. `rule2-depth+route`：规则 v2 对应 0.25 档变体 `v2d`，渲染深度列距离（同 K3 depth 臂）
  7. `rule2-depth-op030+route`（仅 `v2` 套件，B5 诊断臂）：同上但渲染深度加 `depth_opacity_min = 0.30`

## 命令

```bash
D=/root/rivermind-data/uav_rgb

# 1. 离线重判已录轨迹（第 7 条验收：对拍审核 J 的 98/120 与定稿 v2）
CUDA_VISIBLE_DEVICES="" $D/3dgs-venv/bin/python experiments/2026-11-12-3dgs-rebaseline/rescore_offline.py

# 2. 桩演练（无 GPU，验证实例锁、冻结副本、续跑、半成品记录与汇总）
DRY_RUN=1 FREEZE=tree DRIVER=$PWD/experiments/2026-11-12-3dgs-rebaseline/dry_run_driver.py \
  OUT_ROOT=$D/tmp/phase12-cards/s1b/r2/dry_rehearse WATCH_S=0.2 \
  setsid -w bash experiments/2026-11-12-3dgs-rebaseline/run.sh

# 3. 单车道真实 GPU 小冒烟（1 条路线 × 6+7 臂 = 13 集，约 1 分钟，不拿池锁）
SMOKE=1 FREEZE=tree LANES=1 SUITES="indoor40_v1 indoor40_v2" INDOOR_SCENES=sage3d_841150 \
  ROUTES_LIMIT=1 K_INDOOR40=1 OUT_ROOT=$D/tmp/phase12-cards/s1b/r2/smoke_gpu \
  setsid -w nice -n 10 bash experiments/2026-11-12-3dgs-rebaseline/run.sh

# 4. 正式实验（主会话审完合入 main 后守护进程起跑，2 条车道，排 gs3d.lock）
setsid nohup bash experiments/2026-11-12-3dgs-rebaseline/run.sh > /dev/null 2>&1 < /dev/null &
```

## 文件说明

- `run.sh`：守护进程入口（预注册原文、`SPLIT_FROM_IDENTITY=true` 缺省开启、实例锁、代码冻结、追加式起跑记录、`gs3d.lock` 排队、2 车道驱动、体积看门狗、真实退出码 `DONE`）。
- `build_config.py`：生成 6 个套件的 `config.json`（含 `collision_model`、`split_from_identity: true`、`drop_trace: false` 及 `v2` 的 B5 诊断臂）。
- `summarize.py`：走 `sim.3dgs.route_dirs` 汇总新旧口径配对（含「v2 判碰而 v1 未碰集数」「规划与判定层不一致步数」、同起点/异起点拆分）、v1 对原实验逐集对拍、`*.incomplete-*` 续跑半成品表。
- `rescore_offline.py`：用入库的定稿 `Occupancy` 对 W1-11 / W1-12 / K1 / K3 共 23 个已完成臂离线重判（含重定基线臂表之外的 K1 `forward+route@0.5`、`rule-occ+route@0.5`、`rule-occ+goal`、`rule-depth+route`，它们没参与高度包络的测量，用来核对包络外步数在样本外也为 0）。
- `measure_z_margin.py`：整集高度包络与起飞水平漂移测量脚本（16 个来源 1,824 集 → `0.035 / 0.075 / 0.105 m`）。
- `dry_run_driver.py`、`test_driver_collision_model.py`、`test_measure_z_margin.py`、`test_rebaseline_entry.py`：桩驱动与单测。
