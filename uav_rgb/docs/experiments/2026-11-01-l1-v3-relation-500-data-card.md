# 数据卡：L1 v3 同色柱对 + 关系词 500 集

- 时间：2026-11-01 05:21–06:48（端到端 86 min）
- 路径：`$DATA_ROOT/uav_rgb/l1-v3-relation-500/`（52 MB）、决定性集
  `$DATA_ROOT/uav_rgb/l1-v3-dec/`（18 MB）
- 采集命令：

```
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 5 --episodes-per-instance 100 --steps 60 --hold 1.0 \
  --base-seed 900000 --task l1 --spawn-mode random --window-from observation \
  --out $DATA_ROOT/uav_rgb/l1-v3-relation-500
```

- 口径：teacher `uavrgb.teacher.v5`；`task_version=l1.v3`；`stop_radius=0.5`；
  几何 `uavrgb.geom.v5`；`window_from=observation`；`spawn_mode=random`
  （L1 关系任务不再支持 default 出生）。

## 规模与分布

采集 500 集 → 3 集失败（见下）→ 497 集 → 排除 3 个含碰撞整集 → **494 集 / 7104 步**。

| split | 集 | 步 | SAFE | FORWARD | YAW_LEFT | YAW_RIGHT |
| --- | --- | --- | --- | --- | --- | --- |
| train | 398 | 5705 | 20.9% | 71.6% | 3.7% | 3.8% |
| val | 47 | 702 | 20.1% | 71.7% | 3.8% | 4.4% |
| test | 49 | 697 | 21.1% | 70.3% | 4.3% | 4.3% |
| **合计** | **494** | **7104** | **20.9%** | **71.5%** | **3.7%** | **3.9%** |

- 关系词：train 200 left / 198 right，val 22/25，test 25/24。
- 目标色：red 130 / blue 120 / yellow 127 / green 117。
- episode 步数：p10=10、p50=14、p90=21、max=28（`--steps 60` 从未跑满，
  全部 `stopped_on_goal=true`）。
- 唯一几何 378 组；`split_geometry` 交集 0（test=49, train=398, val=47）。

与 v2 的差别：SAFE 49.7% → 20.9%（safe 出生集已删除，剩下的都是真 SAFE）；
FORWARD 39.3% → 71.5%；YAW 5.5/5.4% → 3.7/3.9%。

## 失败与排除

| 类别 | 集 | 原因 |
| --- | --- | --- |
| 采集失败 | 900188 / 900201 / 900204 | `collect_parallel` 记为 failed，另有同 seed 重试 1 次仍失败 |
| 排除（含碰撞） | 900034 / 900133 / 900295 | 各 1 个 collision 步；标签不可信，整集排除 |

## 验收（`audit.json`，494 集 / 7104 步）

14 项 PASS，**3 项 FAIL**：

| 检查 | 状态 | 实测 |
| --- | --- | --- |
| `time_pairing` | **FAIL** | 1/7104 步 depth–RGB gap = 100 ms（阈值 50 ms）。采集器的停更门限本身是 100 ms，这一步正好卡在线上 |
| `class_share` | **FAIL** | YAW_LEFT 3.7% / YAW_RIGHT 3.9%，低于 5% 阈值。split 内无整类缺席 |
| `safe_collision` | **FAIL** | 审计集 0/1483；但 `train/900295` step 9 有 1 个 SAFE 步带 contact。该集已排除，检查按设计仍统计排除前的数 |
| 其余 14 项 | PASS | `teacher_stop` 494/494 停在 waypoint 半径内且末动作 SAFE；`label_recompute` 0/7104 不一致；`relation_balance` left 248 / right 249、0 集未分居左右 |

## 决定性帧（文本监督强度）

`make_decisive.py --splits train`：

| 量 | v2 | **v3** |
| --- | --- | --- |
| 决定性源集 | — | 401 / 401（每集至少一帧随指令变化） |
| 决定性唯一状态 | ~13% | **3470 / 5762 = 60.2%** |
| 输出 | — | 802 集 / 6940 样本（每个状态一对，仅关系词不同） |

60.2% 意味着训练集里六成样本**必须读指令**才能拟合；v2 只有 ~13%，这是此前
FiLM 学不会颜色/关系的直接原因。

## 注意

- `make_decisive.py` 此前的输出目录编号有撞号 bug（相邻 `episode_id` 差 1，
  `9_000_000 + sid` 再 +1/+2 会互相覆盖），旧 `ds-dec` 实际只有约六成集，
  且被覆盖的恰好是**反事实那一半**。本数据卡的 60.2% 是修好之后重算的。
- 76% 的集 `|waypoint.y − spawn.y| < 0.3 m`（几乎直飞），24% 需要横移 1.56 m。
  原因是 `waypoint_for` 把 waypoint 朝走廊中心偏 0.8 m，当目标是靠近出生点的
  那根柱时，waypoint 正好落回出生横线上。这是 YAW 占比只有 3.7% 的直接来源。
- 三处 FAIL 未处理：没有重采、没有改阈值、没有为了过门禁删数据。
