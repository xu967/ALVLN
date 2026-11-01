# 重采 500 集（几何 v5 / l1.v3 同色柱对）

- 时间：2026-11-01 05:00–06:55
- 范围：`sim/tools/collect_parallel.py`、`sim/tools/make_decisive.py`、
  `sim/tools/test_make_decisive.py`；产物在 `$DATA_ROOT/uav_rgb/`
- 结论：完成。494 集 / 7104 步可用，audit 14 项 PASS / 3 项 FAIL（未处理，
  理由见下）

## 做了什么

- 冒烟三批（4 / 8 / 20 集）确认新场景可采：全部 `stopped_on_goal=true`，
  episode 11–22 步内到点停车，relation left/right 平衡。
- 冒烟中发现问题并修复：两根目标柱贴画面边缘。实测 500 个 seed 里 **15 个
  （3%）最外侧柱心水平角 >30°**（最大 32.2°），那根只剩一条边。
  `landmarks.py` 加视场门禁（`h_angle` + `HALF_FOV_DEG=30`/`FOV_MARGIN_DEG=2`），
  优先选两根都 ≤28° 的柱对；复测 max 27.9°、0/500 超界。
- 正式采集：5 实例 × 100 集、60 步、hold 1.0、`--task l1 --spawn-mode random`、
  `--window-from observation`，base-seed 900000。端到端 86 min，497/500 成功。
- `sim/tools/collect_parallel.py`：`SPLIT_POLICY` 里写死的 `uavrgb.geom.v3`
  改成 `<SPLIT_SALT>` 占位（描述串，不影响 split 计算；写死过一次就再也对不上）。
- `sim/tools/make_decisive.py`：**修输出目录撞号 bug**。原来 `base = 9_000_000 + sid`，
  相邻集的 sid 只差 1，第 i 集的 `base+2` 会撞上第 i+1 集的 `base+1`；
  v3 实测 478 个目录里 324 个被覆盖，且被覆盖的恰好是**反事实那一半**。
  改为 `base = 9_000_000 + 2 * sid`。重跑后 802 个目录、无孤儿帧。
- `sim/tools/test_make_decisive.py`：加回归断言——两个相邻 `episode_id` 必须产出
  4 个互不相同的目录、每个目录 2 行标签 2 张图。
- 产出决定性集 `$DATA_ROOT/uav_rgb/l1-v3-dec`（802 集 / 6940 样本）。
- 写了数据卡 `docs/experiments/2026-11-01-l1-v3-relation-500-data-card.md`。

## 没做什么

- **没重训、没跑闭环。** 本轮只到「数据可用」为止。
  - 原因：训练要另外定 --class-weights / 文本分支配置，且 YAW 占比的处置
    还没定（见待确认）。
- **没有为了让 audit 变绿去改阈值、删步或重采个别 seed。**
  - 原因：三条 FAIL 分别是阈值边界（1 步 100 ms）、任务固有分布（YAW 3.7%）、
    1 个落在已排除集里的 SAFE+contact 步。改阈值等于把门禁改成能过，
    需要用户点头。
- 没换相机视场（仍 60°）。加了视场门禁之后 0/500 出画，扩视场反而会引入
  观测分布变化这个混杂因素。
- 没动 `waypoint_for` 的 0.8 m 偏置规则。它造成 76% 的集几乎直飞，
  改它等于改成功判定的目标点，属于任务口径变更。
- 没提交、没推送：用户未授权。

## 待确认

- **`class_share` 的 YAW 3.7% / 3.9% 怎么处置**：(a) 接受，训练时用
  `--class-weights auto`；(b) 改 `waypoint_for` 让左右两个分支都必须横移
  1.6 m（直飞占比会从 76% 掉到 0，YAW 占比大概翻倍），然后重采 86 min；
  (c) 调低 5% 阈值。我倾向 (a)：YAW 有 561 个样本，绝对量够，而且文本实验
  的关键指标是决定性帧占比（60.2%），不是 YAW 占比。
- `time_pairing` 的那 1 步（100 ms）要不要为此重采整集：我倾向不重采。
- `safe_collision` 的那 1 步落在已排除的 `train/900295` 里，实际不会进训练集；
  但门禁按设计统计排除前的数，所以一直 FAIL。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `audit_dataset.py --exclude-collision-episodes` | 494 集 / 7104 步，14 PASS / **3 FAIL**（`time_pairing`、`class_share`、`safe_collision`） |
| 其余 14 项（含 `label_recompute` 0/7104、`teacher_stop` 494/494、`relation_balance` 248/249、`geometry_version` v5、`split_geometry` 交集 0） | PASS |
| `test_make_decisive.py` | PASS（含新增撞号回归） |
| `make_decisive.py` 实跑 v3 数据 | 802 集 / 6940 样本，802 目录、6940 图、无孤儿帧 |
| 决定性帧占比 | **60.2%**（3470/5762；v2 约 13%） |
| `bash tools/check.sh` | **PASS**（21/21 CTest + 全部 Python 自检） |
| 重训 / 闭环 | 未跑 |

## 风险与遗留

- **YAW 占比从 5.5% 掉到 3.7%**，且 76% 的集几乎直飞。如果下一轮训练 YAW 类
  召回明显掉，要先怀疑是数据分布而不是文本分支。
- `make_decisive.py` 的撞号 bug 影响过所有历史 `ds-dec`（包括训练
  `run-film-dec` 的那份）。那些实验的结论要打折扣——当时训练集里六成的
  反事实对被覆盖掉了。
- 三条 FAIL 没消。按 `docs/EVAL_STANDARD.md` 的口径，这份数据还不能当作
  「通过验收」的正式数据集引用，只能当「已采集、有 3 处已知问题」使用。
