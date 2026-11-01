# 删除 safe 出生模式（几何 v4）

- 时间：2026-11-01 03:30–03:55
- 范围：`sim/tools/`（scene_gen / collect / collect_parallel / audit_dataset / test_spawn）、
  `tools/`（preprocess / check.sh）、`docs/EVAL_STANDARD.md`、`docs/HANDOFF.md`
- 结论：safe 出生模式已整条删除。它是相机贴柱面（0.100 m）导致 34.5% 纯色帧的唯一来源，
  且**无法修复**——`plan_action` 判 SAFE 的位姿定义上就必然贴着柱子。

## 做了什么

依据（2026-11-01 实测，100 个 seed）：

| 量 | 结果 |
| --- | --- |
| `make_spawn(safe)` 相机到柱面距离 | min 0.100 / 中位 **0.100** / max 0.112 m |
| 全柱区 0.25 m 网格里「`plan_action`=SAFE 且视野 ≥0.8 m」的位姿 | **0/100 个 seed** |
| safe 集贡献的步数 | 4000 / 10442（38%），**100% SAFE** |
| 去掉 safe 集后 astar 的 SAFE 占比 | 18.5%（1194/6442） |

- `sim/tools/scene_gen.py`：`make_spawn` 只剩 `default`/`random`；删除 safe 分支及其
  `teacher.plan_action` 依赖。`GEOMETRY_VERSION` v3 → **v4**（`split_manifest.SPLIT_SALT`
  随之重排，旧数据与新数据不能混包）。`--spawn-mode` 选项去掉 `safe`。
- `sim/tools/collect.py`：`planner_kind` 固定 `"astar"`；`goal_stop = bool(stop_radius)`；
  `global_plan=True`；删除 `scene_spawn_mode` 变量。
- `sim/tools/collect_parallel.py`：删除 `--safe-spawn-fraction` 及其校验、`safe_stride`、
  `spawn_mode` 轮换；manifest 去掉 `safe_spawn_fraction` / `safe_episodes_requested`。
- `sim/tools/audit_dataset.py`：删除 `safe_full_length` 闸门；`teacher_waypoint` 去掉
  `spawn.mode != "safe"` 过滤。
- `tools/preprocess.py`：删除 `--exclude-safe-episodes`（safe 集已不存在，标志成了死开关）。
- `tools/check.sh`：删除对应的 `--exclude-safe-episodes` 负例；PASS 行同步。
- `sim/tools/test_spawn.py`：只测 `default`/`random`；新增一条「`make_spawn(..., "safe")`
  必须抛 `ValueError`」的回归断言，防止有人把它加回来。
- `docs/EVAL_STANDARD.md` §5 终止条件、`docs/HANDOFF.md` 几何常量表：同步为 v4 口径。

## 没做什么

- 没重采、没重训、没跑闭环。本轮只删生成路径，数据要等场景升级（同色柱对）那一轮一起采。
  - 原因：单独重采一版只是把废帧去掉，任务本身还是「看图选颜色」，不值得再花 3 小时。
- 没动 `sim/tools/traverse_rate.py` 的 safe 剔除逻辑（3 行，`test_traverse_rate.py` 仍在测）。
  - 原因：它读的是 `scene.json` 的 `spawn.mode`，对归档的 v3 数据仍然正确，删了会让旧数据
    的穿越率被稀释到 1−0.2。
- 没删 `_spawn_collides`。
  - 原因：`test_spawn.py` 仍用它断言 default/random 出生点不撞柱，是有效检查。
- 没改 `docs/worklog/` 与 `docs/experiments/` 里的历史记录（T2、T5、l1-pilot 等）。
  - 原因：那些是当时的事实记录，不改写；只在 HANDOFF 当前状态表上加了 v4 说明。
- 没提交、没推送：用户未授权。

## 待确认

- 删除 safe 集后 SAFE 只剩 18.5%，比现在更不均衡；下一轮训练必须带 `--class-weights auto`，
  否则 SAFE 类会被压掉。这条写进了风险，未实测。
- `geometry_gid` 因 SPLIT_SALT 变化全部重排，新采集的 train/val/test 与旧 500 集完全不同一批
  几何。这是有意的（保证新旧不泄漏），但如果要拿新旧数据对比指标，需要注明不可直接比。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python -m py_compile`（6 个改动文件） | PASS |
| `bash -n tools/check.sh` | PASS |
| `sim/tools/scene_gen.py --self-check` | PASS（同 seed 字节级一致，缝隙契约全过） |
| `sim/tools/test_spawn.py` | PASS（20 seeds × 2 modes；safe 模式已删除） |
| `bash tools/check.sh` | **PASS**（21/21 CTest + 全部 Python 自检，rc=0） |
| 重采 / 重训 / 闭环 | 未跑（见「没做什么」） |

## 风险与遗留

- **`GEOMETRY_VERSION` bump 后，v3 的采集目录不能再被 `preprocess.py` / `audit_dataset.py`
  打包**（会因 `geometry_version` 不符被拒）。已打好的 `ds/`、`ds-cf/`、`ds-dec/` 不受影响，
  训练与 eval 照常。知道这点就不用去重打包旧数据。
- 「纯色 SAFE 帧 → 闭环 35% 提前停车」仍然是**假设**，本轮只删了来源，没有做对照实验证实。
  如果下一轮数据（无 safe 集）训练后提前停车显著下降，这条假设才算被支持。
- 删除面涉及 7 个文件 + 2 份文档，行为变化集中在采集链路；C++ 侧、训练器、评测口径均未动。
