# 代码审核：L1 v3 目标点退化，闭环 SR 测不出语言

- 时间：2026-11-01 17:00–17:40
- 范围：只读审核 + 离线探针 + 3 集 teacher 实飞；未改仓库代码
- 结论：**任务定义坏了，不是模型坏了。** 274 个 test 场景里没有一个要求按关系词
  横移——两个候选 waypoint 都落在出生航线上。修 3 行几何可以 274/274 变成必须
  选边，teacher 实飞验证可解（3/3，16–18 步）。

## 做了什么

- 读了目标点链路：`landmarks.waypoint_for` / `assign_landmarks` → `scene_gen` →
  `teacher.plan_action_global` / `astar_path` → `eval_closed_loop.summarize_episode`。
- 用离线探针（不依赖 Gazebo）遍历 seed 0–2999 里 split=test 的 274 个场景，量：
  - 真实 waypoint 与反向 waypoint 各自离出生航线的横向距离
  - 两个候选点的间距、是否落在 0.7 m 成功半径内
  - 「找不到柱对」「柱子在 ±28° 视场外」「`waypoint_for` 符号翻转」「提案规则撞柱」的比率
- 用 500 集已有 teacher 采集量了数据侧的行为分布（|Δy|、FORWARD 占比、
  teacher 自己到 waypoint 的精度）。
- 用**临时脚本**（改 scene_gen 写出的 meta 的 waypoint，不动仓库代码）跑了 3 集
  teacher 实飞，验证新规则下 teacher 还能不能飞到：seed 113 / 172 / 180。
- 更正文档 `docs/experiments/2026-11-01-l1-v3-closed-loop.md`：早先「必须横移 10/49」
  是拿错了反向柱算的（同色柱对后两根候选都染色，`next(p for p in ps if not
  p.get("color"))` 返回第三根柱），改成 0/274。
- 清理：删掉 `/tmp/probe_geom.py`、`/tmp/probe_geom2.py`、`/tmp/feas_teacher.py`
  和 `/tmp/feas_teacher/`。

## 没做什么

- **没改任何仓库代码。** 改 `waypoint_for` 等于改任务口径（成功判定的目标点变了），
  需要拍板后再动。
- **没重采、没重训、没重跑闭环。** 本轮只做审核与可行性验证。
- **没改 `make_counterfactual.py`。** 它是 `waypoint_for` 的第二个调用方，签名
  一改就要跟着传 `scene["spawn"]`；等改动落地时一起改。
- **没改 `test_landmarks.py`。** 第 48–50 行断言「relation=left → 目标是最左的柱」，
  是现行映射的回归守卫；改规则时必须同步改成「最右的柱」，否则测试会直接红。
- **没修 `carry_resume_best`**（跨数据集续训冻结 `ckpt_best.bin`），已知未修项。
- **没跑 `bash tools/check.sh` 全量。** 本轮没改代码，跑它没有新信息。
- **没把 teacher 实飞的每帧轨迹留下来做可视化。**
- 没提交、没推送：用户未授权。

## 待确认

- 要不要按下面的补丁改任务定义 + 重采（约 86 min）+ 重训（约 66 min）。

## 验证到哪一步

### 现行规则：274 个 test 场景，0 个需要选边

| 指标 | 数值 |
| --- | --- |
| 真实 waypoint 的 \|横向偏移\| | p50 **0.09 m**（成功半径 0.7） |
| 反向 waypoint 的 \|横向偏移\| | p50 **0.08 m** |
| 两个候选点相距 ≤0.7（选错也算对） | 79/274 = 28.8% |
| **两个候选都必须横移 >0.7** | **0/274 = 0%** |
| 找不到柱对 | 0/274 |
| 选中柱对里有柱子出 ±28° 视场 | 0/274 |
| `waypoint_for` 触发符号翻转（死代码） | 0/274 |

根因：`landmarks.py:66` 的 `sign = -1 if target["y"] > 0 else +1`（朝走廊中心偏
0.8）。柱子只在 y≈0 和 y≈±1.6 三条车道，出生点在 y=±(0.72–0.88)，所以无论目标是
哪一根，`wy` 都落在 ±0.8——正好是出生点的航线上。

数据侧：500 集 teacher 采集 |Δy| 中位 **0.12 m**，FORWARD 占 **73%**，
teacher 自己 `wp_final ≤ 0.7` 的比例 **99.8%**（任务可解，只是不含选边）。

### 提案：waypoint 放在柱子「远离出生航线」的一侧

| 指标 | 现行 | 提案 |
| --- | --- | --- |
| 真实 waypoint \|横移\| p50 | 0.09 m | **1.60 m** |
| 两边都必须横移 >0.7 | 0/274 | **274/274** |
| 两候选点间距 p50 | 1.00 m | **3.20 m** |
| 找不到 waypoint / 撞柱 | 0 / 0 | 0 / 0 |

### teacher 实飞（临时脚本，只改 meta 的 waypoint，不动代码）

| seed | 出生 y | 目标 y | 新 waypoint y | 步数 | wp_final | 动作 |
| --- | --- | --- | --- | --- | --- | --- |
| 113 | -0.76 | -1.57 | -2.37 | 16 | 0.34 | 12 FWD / 1 YAW_R / 3 SAFE，末步 SAFE |
| 172 | +0.75 | +1.54 | +2.34 | 18 | 0.17 | 11 FWD / 4 YAW / 3 SAFE，末步 SAFE |
| 180 | +0.72 | +1.52 | +2.32 | 18 | 0.41 | 12 FWD / 3 YAW / 3 SAFE，末步 SAFE |

3/3 到达且按规矩停下；步数 16–18，远低于闭环评测现有的 `--steps 40`。

## 风险与遗留

- **改动会让新旧数据不可比。** 建议只 bump `task_version` `l1.v3` → `l1.v4`
  （`geometry_version` 保持 v5：柱子/出生点没变，split 不变，新旧数据在同一批
  场景上可以直接对比）。`preprocess.py:240` 会拦 `task_version` 混包。
- **`waypoint_for` 的符号翻转要删掉。** 现在它是一层静默降级：一旦触发就把
  waypoint 挪到柱子另一侧，静默产出退化场景且语义反了。改成直接抛错。
- **关系词的语义要跟着翻。** waypoint 挪到外侧后，目标柱在停止点的内侧；
  「pass the X pillar on your left」必须对应「目标 = 右边那根柱」。只是一行。
- **`eval_closed_loop.py` 的 spawn 修复仍无回归测试**（上一轮改的）。
- **`--text-mode` 的 help 文案过期**（`eval_closed_loop.py:344` 还写着
  「shuffle=换一个目标色」，实际是翻关系词）。
