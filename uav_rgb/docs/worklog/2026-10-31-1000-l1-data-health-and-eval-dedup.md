# L1 数据体检 + 评测去重口径修复

- 时间：2026-10-31 09:45–10:05
- 范围：`sim/tools/eval_closed_loop.py`、`docs/experiments/`（只读分析 400 采集产物）
- 结论：完成。发现 3 个口径问题，修了 1 个（L1 去重），另 2 个记录待拍板。

## 做了什么

- 只读分析 `l1-full-400-0803` 中途产物（330 集）：
  - 几何重复：57 组，其中 56 组是「同几何不同目标色」的反事实对，仅 1 组纯重复；
  - 唯一 `(gid, target_color)` 315 个（330 集）；
  - `spawn.mode=safe` 66/325（20%），且 66 集 manifest 全部带 waypoint。
- 彩色柱可见性：40 个默认出生集 × 40 帧按颜色像素数统计，
  目标色 40/40 出现过、33/40 在 step 0 可见；两色同时可见 31/40（step 0）、40/40（任意帧）。
- 修 `eval_closed_loop.py`：`--task l1` 去重 key 由 `gid` 改为 `(gid, target_color)`；
  `eval.json` per_episode 增加 `target_color`。
- 新增 `docs/experiments/2026-10-31-l1-data-health.md`（体检结果）。
- `docs/experiments/2026-10-31-l1-language-task-design.md` 追加 §7 L2 预注册
  （四组评测、冻结阈值、依赖项）。
- 清理临时脚本 `/tmp/vis_check.py`。

## 没做什么

- **没有改 safe 出生集的语义**：66 集指令是装饰性的（teacher 走局部规划不跟 waypoint）。
  要干净得重采或把 instruction 置空，属任务设计变更，等拍板。
  - 原因：会让 120 pilot 的 `ds` 与 audit 口径失效，需要重跑。
- **没有加 `--text-override`**：`collect.py` 正在被 400 采集占用，改它会让同批数据
  出现两种行为。等采集结束再加。
- **没有跑 audit / preprocess / 训练**：采集未结束，且训练与 Gazebo 不能并行。
- **没有验证去重改动的闭环行为**：需要起 Gazebo，等采集结束后的 L1 eval 一起验。
- 没有提交、没有推送（未获本轮授权）。

## 待确认

- safe 出生集（20%）怎么处理：保留当 SAFE 来源 / 置空指令 / 整集丢弃。
- D-STACK（L2 全自研 C++/CUDA 还是 Python 原型）仍待定；预注册协议与栈无关。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `py_compile sim/tools/eval_closed_loop.py` | 通过 |
| `test_eval_summary.py` | PASS |
| 400 采集进程 | 未受影响（只改了 eval 侧，不在采集路径） |

## 风险与遗留

- 体检数字是采集中途抽样（330/400），最终要在采集完成后重跑一遍再写进结论。
- 几何重复 20% 意味着 400 集的唯一几何只有约 320；test 有效样本 33 集（25 唯一 gid），
  10 pp 阈值约等于 3.3 集，统计力弱。
- 颜色可见性判据是自调阈值，只能说明「有没有」，不能当定量指标。
