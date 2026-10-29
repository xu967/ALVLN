# D0b + D1：动作窗口改仿真时间，采集记动作前位姿

- 时间：2026-10-29 23:38
- 范围：`sim/tools/collect.py`、`sim/tools/collect_parallel.py`、
  `sim/tools/eval_closed_loop.py`、`sim/tools/audit_dataset.py`、
  `sim/tools/test_audit_dataset.py`
- 结论：**完成**。动作窗口改用仿真时间后 `sim_dt` 变异系数 18.3% → 0.0%；
  `label_recompute` 从 SKIP 变成真跑，新采集 16 步 0 不一致。

## 做了什么

- `collect.py` 动作窗口改为仿真时间：窗口终点 = 动作前位姿戳 + `hold` 秒，
  轮询 `/model/x3/pose`（`POSE_TOPIC`）的最新戳，走满才发零速。墙钟只做兜底
  （`hold / MIN_WINDOW_RTF`，`MIN_WINDOW_RTF=0.05`），超时先调 `_try_zero()` 再报
  本步失败。前提是零速发得出去：两条通道都失败时只打警告，动作会留到外层
  `finally` 再补一次零速（评审 P2-9 已确认这条链成立，但不是强保证）。
- `collect.py` 的 `actions.jsonl` 增记 `pose_before` / `yaw_before`。
  注意写盘处是**字段白名单投影**，不改这里新字段不会落盘（第一次改漏了，
  跑完发现行里没有，补在 `collect.py` 的写盘处）。
- `audit_dataset.py` 实现 `label_recompute`：对 `teacher=plan_action` 的集，
  用 `pose_before` + `scene.json` 柱子真值重跑 `plan_action`，与原标签逐条比。
  阈值 `MAX_LABEL_MISMATCH=0`（该函数确定性，同输入必同输出）。
  非 `plan_action` 的集（`expert_action` / `forward` / `student`）标 SKIP 并报出
  实际 teacher 分布——拿柱子真值去复算深度专家等于换教师，比出来没意义。
- `collect_parallel.py` / `eval_closed_loop.py` 的 `--hold` 帮助文本同步改成仿真时间。
- `test_audit_dataset.py` 增第 5 组用例：`plan_action` 自洽标签 → PASS，
  改坏一条 → FAIL。

## 没做什么

- **没有存 depth**（任务板 D0b 原本写了「depth 存盘」）。
  - 原因：现在没有消费方。要做的「图像-标签对齐」检查需要把柱子真值投影到
    图像平面，得先定相机模型约定和容差，是一块独立工作；先存盘只是白占
    ~75 KB/步（2 万步 ≈ 1.5 GB）。等这条检查真写出来再加开关。
- **没有改 `Z_KP` 的注释**。它还写着「wall-clock hold 下 sim 步长会漂」，
  现在前提没了，但值本身没验证过该不该变，不动。
- **没有重跑大规模采集**。只跑了 2 集 × 8 步的验证量，阈值（
  `transition_lag` 50%）没有用大批数据重新标定。
- **没有改 `transition_lag` 阈值**。新口径下 4 个过渡步中位数 43%，低于 50%
  判 FAIL；样本太小，先记进「待确认」，不为了让闸门变绿而调阈值。

## 待确认

- `transition_lag` 50% 阈值在仿真时间窗口下是否还合适。物理上偏航接前进有
  加速滞后，位移打折是真实的；如果打折到 43% 是常态，这个检查的语义应该
  改成「不要低到 20% 以下」，而不是卡 50%。
- `MIN_WINDOW_RTF=0.05` 是拍的。实测 RTF 约 0.25；取 0.05 是为了宁可多等，
  因为误判超时会白白丢一步。要不要收紧到 0.1 得看低配机器上的表现。
- `class_share` 在 16 步的小样本上必然 FAIL（`SAFE_NEUTRAL` 没出现），
  这是样本量问题不是 bug；但闸门要不要对小数据集给「样本不足」单独的
  状态，而不是硬判 FAIL，需要定。

## 验证到哪一步

| 项 | 真实输出 |
| --- | --- |
| `python3 sim/tools/test_audit_dataset.py` | **PASS**，EXIT=0 |
| `python3 -m py_compile` 三个改动文件 | OK |
| 真实 Gazebo 采集 A（`--instances 2 --steps 8 --hold 1.0 --base-seed 0`） | 2/2 集、16/16 步，无残留进程 |
| 真实 Gazebo 采集 B（同 seed 重跑） | 2/2 集、16/16 步 |
| `audit_dataset.py --dataset d1-verify-a` | EXIT=1，9 项里 6 PASS / 2 FAIL / 1 SKIP，见下 |

两次采集的 `sim_dt`（从 `oracle.jsonl` 的 action_start/end 算）：

```
scene 0  sim_dt A: [1.0]×8   max-min=0.000   cv=0.00%
scene 0  sim_dt B: [1.0]×8   max-min=0.000   cv=0.00%
scene 1  sim_dt A: [1.0]×8   max-min=0.000   cv=0.00%
scene 1  sim_dt B: [1.0]×8   max-min=0.000   cv=0.00%
```

对照旧数据（`/tmp/ds200`，墙钟 hold）：`sim_dt mean=0.467s min=0.220 max=0.880
cv=18.3%`。

同 seed 两次采集的动作序列：scene 1 完全相同
（`[3,1,1,3,1,1,2,2]`）；scene 0 在第 4 步分叉（A 走 `YAW_LEFT`，B 走
`FORWARD_SLOW`）。Gazebo 不是逐位确定的，微小位姿差会让 `plan_action`
在阈值附近翻面——这属于物理，不属于采集器缺陷；`sim_dt` 这条验收口径
（同 seed 两次一致）是满足的。

新数据跑审计的真实结果：

```
集数=2 步数=16
  [PASS] manifest_validity  0/2 集 validity 不全 true
  [PASS] time_pairing       max_gap=36.0 ms，超 50 ms 的步 0/16
  [PASS] action_window      sim_dt mean=1.000s min=1.000 max=1.000 cv=0.0%
  [PASS] forward_motion     0/3 个稳态前进步位移不足 0.12×dt（0.0%，阈值 1%）
  [FAIL] transition_lag     偏航接前进的 4 步，位移中位数为标称的 43%（阈值 50%）
  [PASS] yaw_direction      0/9 个偏航步转向与标签相反或过小
  [FAIL] class_share        每类占比低于 5%：['SAFE_NEUTRAL']
  [SKIP] split_geometry     少于两个 split 且有 gid，无法核验
  [PASS] label_recompute    0/16 步复算标签与原标签不一致（0.0%，阈值 0%；跳过 0 步）
```

## 风险与遗留

- **`label_recompute` 只覆盖 `teacher=plan_action`**。深度反应式专家
  （`expert_action`）和盲策略的标签无法用柱子真值复算，这条闸门对它们
  仍然是盲的。要覆盖得另行定义「深度专家在什么输入下该输出什么」。
- 同 seed 动作序列不完全可复现（scene 0 第 4 步分叉）。依赖「同 seed 同轨迹」
  的下游实验不能假设位级一致。
- 旧数据（`/tmp/ds200`）仍是无 `pose_before` 的格式，`label_recompute` 对它
  永远 SKIP——**旧数据集不可用于训练**这条结论不变。
- **闭环评测会变慢**。`eval_closed_loop.py` 的 `--episode-timeout` 默认 600 s，
  以前按墙钟 hold 走 80 步只需 ~35 s；改成仿真时间后，80 步要 80 s 仿真时间，
  RTF 0.25 时约 320 s 墙钟。仍在 600 s 以内，但 RTF 再低就会撞上限被判超时。
  重跑闭环前先确认 RTF，必要时调 `--episode-timeout`。

## 更新 2026-10-30 02:30：评审后的修复

评审查出 17 条，这里记与 D0b/D1 相关的修复（其余见 D0c 的 worklog）。

- `collect.py` 动作窗口循环增加 `subs["pose"].healthy()` 判断：订阅进程死掉时
  立刻以「pose 订阅进程已退出」失败，不再白等满 20 s 再报「RTF 过低」。
- `collect.py` 现在把 `--scene-meta` 的 JSON 原样落盘成 episode 目录下的
  `scene.json`（新增 `write_scene_meta()`）。**这条是覆盖面缺口**：之前只有
  `collect_parallel.py` 会复制 `scene.json`，单跑 `collect.py --scene-meta`
  产出的集会永远 `label_recompute SKIP`，而 README 的示例正是这条路径。
- `collect.py` manifest 里 `hold_s` 的注释还写着「墙钟秒、随 RTF 漂移」，已改。
- `audit_dataset.py` 三处判定修正：
  - `scene.json` 的 `pillars` 为 `null` 时 `geometry_gid(None)` 抛 `TypeError`
    会让整个 audit 崩（连 `audit.json` 都没有）；现在 `isinstance(list)` 挡住，
    按「没有真值」处理。
  - `label_recompute` 不再要求「所有 plan_action 集都有 `pose_before`」——
    新旧格式混在一起时，一条 `all(...)` 会把新格式的集也一起降级成 SKIP。
    改成逐集/逐步统计，输出里写明跳过多少步、另有非 plan_action 的集未参与。
  - `total == 0` 从「`share=1.0` → FAIL（100% 不一致）」改成 SKIP。
    一步都没复算成不等于全错，报 FAIL 会把排查引向错误方向。
- `test_audit_dataset.py` 补 3 组用例锁住上面三条（0/0 → SKIP、混合格式仍验
  新集且 `covered_steps=12`、`pillars=null` 不崩）。

验证：`test_audit_dataset` PASS、`test_collect` 14 项、`tools/check.sh` EXIT=0。
