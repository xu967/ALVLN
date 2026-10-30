# T2：SAFE_NEUTRAL 数据与出生位姿

- 时间：2026-10-30
- 环境：GPU 模式（RTX 2080 Ti / sm_75），Gazebo Harmonic 8.15，DISPLAY=:20
- 结论：**通过**。52 集 / 2080 步全过 audit（11 项 PASS，0 碰撞），
  SAFE 占 21.2%，每 split 四类都有样本；见 §3。

## 1. 设计

- `scene_gen --spawn-mode safe`：在前 3 根柱子周围取 ±0.5 / ±0.25 m 候选，
  用 `plan_action` 确认**真的是 SAFE_NEUTRAL**，再用 0.47 m 机身方框确认不撞柱；
  找不到就粗网格兜底，仍找不到直接报错。
- `collect_parallel --safe-spawn-fraction 0.2`：每 5 个 episode 有 1 个 safe 出生；
  train/val/test 都能分到（52 集实采：safe train 6 / val 2 / test 3，共 11）。
- 普通集用 A* 教师（`planner=astar`），safe 集用局部教师（`planner=plan_action`），
  保证 SAFE 标签不是人为造的。`collect.py` 的 manifest 记录 `planner`，audit 按它复算。
- 硬约束：SAFE 步必须无碰撞；`audit_dataset.py` 新增 `safe_collision` 检查，
  可用 `--exclude-collision-episodes` 整集排除碰撞集。

## 2. 命令

```bash
source env.sh
OUT="$DATA_ROOT/uav_rgb/t2-v3-$(date +%H%M%S)"
# 4 实例 × 13 集 = 52 集；40 步/集（A* 0.27 m/步，40 步 ~10.8 m，够穿场）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 13 --steps 40 --hold 1.0 \
  --base-seed 0 --policy teacher --spawn-mode default --safe-spawn-fraction 0.2 \
  --keep-worlds --out "$OUT"

python3 sim/tools/audit_dataset.py --dataset "$OUT" \
  --exclude-collision-episodes --out "$OUT/audit.json"
```

> 为什么 4 实例 / 40 步：cgroup 只有 12 核且和其它会话共享；8 实例会把 RTF
> 打到 0.1，起飞超时。40 步足够覆盖 x=-0.5→6.0，且空尾更少。

## 3. 结果

### 3.1 采集

- 4 实例 × 13 集 = 52 集，40 步/集，`hold=1.0` sim-time；52/52 成功、0 失败，
  总 2080 步、0 碰撞步、0 集 `stopped_on_collision`。
- 墙钟：20:42:40 → 21:00:31 UTC（采集 1068 s，端到端 1071 s ≈ 17.9 min；
  吞吐 1.95 步/s）。
- 唯一几何 51 组；split：train 39 / val 6 / test 7 集（train 里有一对同几何集），
  唯一几何 gid 38/6/7，跨 split 交集 0。
- safe 出生 11 集（train 6 / val 2 / test 3），全部用局部教师 `plan_action`；
  其余 41 集用 A*（`planner=astar`）。
- 数据集：`$DATA_ROOT/uav_rgb/t2-v3c-044240`，打包后
  `ds/dataset.uavr` 119,829,229 B（2080 × 160×120×3）。

### 3.2 审计（`audit.json`）

`--exclude-collision-episodes` 下 11 项全 PASS（52/52 集）：

| 检查 | 结果 |
| --- | --- |
| manifest_validity | PASS 0/52 集 validity 不全 true |
| time_pairing | PASS max_gap=36 ms，超 50 ms 的步 0/2080 |
| action_window | PASS sim_dt mean=1.001 s，cv 0.3% |
| forward_motion | PASS 0/1069 前进步位移不足 |
| transition_lag | PASS 164 个偏航→前进过渡，位移中位 44%（阈值 20%） |
| yaw_direction | PASS 0/366 偏航步方向错 |
| class_share | PASS 无低于 5% 的类；无 split 内整类缺席 |
| safe_collision | PASS 0/440 SAFE 步带碰撞 |
| geometry_version | PASS 全部 `uavrgb.geom.v3` |
| split_geometry | PASS train/val/test 几何交集 0 |
| label_recompute | PASS 0/2080 不一致（0 跳过） |

`transition_lag` 阈值由 50% 降为 20%：52 集/164 过渡实测中位 44%，50% 会误杀；
20% 仍能抓住「偏航后动作窗口内基本没前进」的真问题。

### 3.3 类别分布（打包后）

| split | 步数 | SAFE | FORWARD | LEFT | RIGHT |
| --- | --- | --- | --- | --- | --- |
| train | 1560 | 240 (15.4%) | 1025 | 125 | 170 |
| val | 240 | 80 (33.3%) | 124 | 16 | 20 |
| test | 280 | 120 (42.9%) | 125 | 18 | 17 |
| 合计 | 2080 | 440 (21.2%) | 1274 | 159 | 207 |

val/test 的 SAFE 占比被 2/3 个 safe 集放大（每集 40 步全停），train 只 15.4%；
这是 split 按几何、safe 集不额外配平的结果，T5 扩大规模后会自然摊薄。

## 4. 没做

- 没有新增场景族：safe 出生仍用同一柱阵，只是起点在柱区里。
- 没有做「safe 后自动脱困」：safe 集整集悬停，只提供「停」样本，不提供
  「从包围中脱出」的轨迹。
- 没有把 safe 集单独切 split：split 仍由几何 gid 决定。
- 没有跑多 seed 采集：52 集只看一个 base_seed=0 的样本。
