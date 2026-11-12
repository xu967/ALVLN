# 3DGS 碰撞口径重定基线（`2026-11-12-3dgs-rebaseline`，任务卡 S1b）

## 背景与目的

交叉审核 B（`docs/reviews/2026-11-12-cross-review/review-b.md`）指出旧 3DGS 碰撞判定（`collision_model = "v1"`）在 0.1 m 体素下存在三项系统偏松：
1. **B1**：膨胀核退化为十字形，45° 对角邻格不膨胀；
2. **B2 / B4 / B7**：飞行高度压在体素层边界上且 z 向不膨胀，判定、教师规划、规则臂列距离与出生检查各用一套 z 口径；
3. **B3**：每步碰撞只扫窗口首尾直线弦，漏掉 hold 1.0 s 下的弧线横偏。

本入口在同一批路线、同一批随机种子上对全部对照臂同时运行 `collision_model = "v1"`（复现对拍原实验）与 `collision_model = "v2"`（B1 + 层带 `z ± 0.053 m` + SITL 10 Hz `trace.jsonl` 折线逐段扫），保留每集 `trace.jsonl`，逐集配对重定 3DGS 线基线。

## 套件、臂与规模

- **3 个路线集 × 2 个碰撞口径 = 6 个套件**：
  - `indoor40_v1`、`indoor40_v2`：K1 / W1-11 / W1-12 室内正式 40 条 × 3 集 = 120 集/臂（`seed = 8000 + 100·路线 + k`，扰动 ±0.2 m）
  - `holdout_v1`、`holdout_v2`：K3 室内留出 32 条 × 3 集 = 96 集/臂（`seed = 5000 + 100·路线 + k`，扰动 ±0.2 m）
  - `outdoor_v1`、`outdoor_v2`：室外 4 场景 × 6 条 × 5 集 = 120 集/臂（`seed = 7000 + 100·路线 + k`，扰动 ±0.5 m）
- **每个套件 6 个臂**（合计 `(120 + 96 + 120) × 6 × 2 = 4032` 集）：
  1. `teacher-tol`：占据 A* 教师，航向容差 18.9076°（`ray`，同 W1-11）
  2. `teacher-swept`：占据 A* 教师，航向容差 18.9076° + `teacher_clearance = "swept"`（同 W1-12 臂 A，不带 `spawn_two_layer`）
  3. `forward+route`：盲前进 + 路线方位护栏 45°（同 K1 / K3）
  4. `rule-occ+route`：规则 v1，占据列距离，`r_safe = 0.25, d_look = 1.2`（同 K1 / K3）
  5. `rule2-occ+route`：规则 v2 选中变体 `v2f`，占据列距离（同 K3 主臂）
  6. `rule2-depth+route`：规则 v2 对应 0.25 档变体 `v2d`，渲染深度列距离（同 K3 depth 臂）

## 卡 S1a（审核 D1）合并后的起跑准备

本入口在 `run.sh` 顶部配置区与 `build_config.py` 中预留了卡 S1a 新增的 `driver.py` 全局配置键 `split_from_identity`：
- 在主会话将卡 `p12-s1a` 与 `p12-s1b` 合并进 `main` 之前，`SPLIT_FROM_IDENTITY` 留空，不向 `config.json` 写入该键；
- 合并进 `main` 后、第二轮正式起跑前，将 `run.sh` 配置区的 `SPLIT_FROM_IDENTITY` 设为 `"true"` 并提交，使所有评测集按几何哈希写入真实 split 与 `diag-train-scene`。

## 命令

```bash
# 1. 桩演练（无 GPU，验证实例锁、冻结副本、续跑、半成品记录与汇总）
D=/root/rivermind-data/uav_rgb
DRY_RUN=1 FREEZE=tree DRIVER=$PWD/experiments/2026-11-12-3dgs-rebaseline/dry_run_driver.py \
  OUT_ROOT=$D/tmp/phase12-cards/s1b/dry_run WATCH_S=0.2 \
  setsid -w bash experiments/2026-11-12-3dgs-rebaseline/run.sh

# 2. 单车道真实 GPU 小冒烟（1 条路线 × 6 臂 × {v1, v2} = 12 集，约 1 分钟，不拿池锁）
SMOKE=1 FREEZE=tree LANES=1 SUITES="indoor40_v1 indoor40_v2" INDOOR_SCENES=sage3d_841150 \
  ROUTES_LIMIT=1 K_INDOOR40=1 OUT_ROOT=$D/tmp/phase12-cards/s1b/smoke \
  setsid -w nice -n 10 bash experiments/2026-11-12-3dgs-rebaseline/run.sh

# 3. 正式实验（合入 main 后守护进程起跑，2 条车道，排 gs3d.lock）
setsid nohup bash experiments/2026-11-12-3dgs-rebaseline/run.sh > /dev/null 2>&1 < /dev/null &

# 4. 单独重跑汇总或测量高度余量
$D/3dgs-venv/bin/python experiments/2026-11-12-3dgs-rebaseline/summarize.py $D/exp-3dgs-rebaseline-20261112 \
  --w111 $D/exp-teacher-tol-3dgs-20261110 --w112 $D/exp-teacher-clearance-3dgs-20261111 \
  --k1 $D/exp-rule-oracle-3dgs-20261110 --k3 $D/exp-rule-v2-3dgs-20261111
$D/3dgs-venv/bin/python experiments/2026-11-12-3dgs-rebaseline/measure_z_margin.py
```

## 文件说明

- `run.sh`：守护进程入口（预注册原文、顶部配置区、实例锁、代码冻结、追加式起跑记录、`gs3d.lock` 排队、2 车道驱动、体积看门狗、真实退出码 `DONE`）。
- `build_config.py`：生成 6 个套件的 `config.json`（含 `collision_model`、`drop_trace: false` 与 S1a 的 `split_from_identity` 预留位）。
- `summarize.py`：走 `sim.3dgs.route_dirs` 汇总新旧口径配对（拆分同出生位姿 / 异出生位姿）、v1 对原实验逐集对拍、`*.incomplete-*` 续跑半成品表。
- `measure_z_margin.py`：稳态飞行高度余量测量脚本（12 个来源 `z − spawn.z` 的 p1–p99 → `HEIGHT_MARGIN_M = 0.025 m`）。
- `dry_run_driver.py`：无卡演练桩驱动。
- `test_driver_collision_model.py`、`test_measure_z_margin.py`、`test_rebaseline_entry.py`：单测与端到端链路测试。
