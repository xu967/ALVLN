# 审核：L1 v4 闭环为什么只能到 SR 0.340

- 时间：2026-11-01 21:30 起
- 范围：只审核，未改代码。产物在 `$DATA_ROOT/uav_rgb/l1-v4-cf/run-film-cf`、
  `l1-v4-closed-20261101`
- 结论：**指令学会了，停不会。** v4 把「按关系词选边」做成了可观测的（shuffle
  成功率 0.000 vs keep 0.426，McNemar p<1e-5），但把「停在哪」做成了不可观测的。
  用户 2026-11-01 拍板方案 A：去掉「必须停」的判定，改成 goal-reach。

## 做了什么

- 跑完 v4 的离线、文本探针、闭环三组（keep / shuffle / 盲前进）。
- 逐帧核算停止点的可观测性：用相机内参和柱体尺寸算「任一像素在 60° 视场内」。
- 对失败集做失败分解，并挑一集逐步对照教师动作。
- **方案 A**：`eval_closed_loop.summarize_episode` 的 L1 分支改成 goal-reach
  （无碰撞 + 末态距 waypoint ≤ `REACH_RADIUS=1.5`，不再看末动作）；
  `success_criterion` 从 `goal_stop` 改成 `goal_reach`；`test_eval_summary.py`
  的 L1 断言跟着改并补半径边界与「到位但撞柱」两条守卫。
- `docs/EVAL_STANDARD.md §5.1`：补 goal_reach 口径与 1.5 m 的几何来历；
  修正关系词定义（`left` 是**经过时**目标柱在左侧，与 v3 的「出生时刻左侧」相反）。
- 起两个只读代码审计（C++ 训练/推理核、Python 数据与评测管线）。

## 核心数据

### 闭环（48 seed，去重 47 集；SR 用 goal_reach）

| 策略 | SR | 碰撞 | false_forward |
| --- | --- | --- | --- |
| student + keep | **0.426** | 0.277 | 0.234 |
| student + shuffle | **0.000** | 0.383 | 0.362 |
| 盲前进（44 集） | 0.000 | 0.455 | 0.455 |

共同完成的 44 集上：keep SR 0.409 / 碰撞 0.295；shuffle 与盲前进都是 0.000。
keep 的 47 集结局互斥且刚好分完：成功 20、到了却撞柱 13、到了但没进 1.5 m 14。
**13 个碰撞全部发生在距 waypoint 1.5 m 以内。**

### 停止点看不见

| | SAFE_NEUTRAL 帧 | 其中视野内零柱子 |
| --- | --- | --- |
| v3 | 147 | 0（0%） |
| **v4** | 144 | **57（40%）** |

v4 的 waypoint 在 `target.y ± 0.8`，柱子只在 `y ∈ {0, ±1.6}`，于是停止点落在
±2.4 m——已经出了柱阵。同时飞行途中有 13% 的 FORWARD 帧也是空的。同一张空图在
数据里既标 FORWARD（77 帧）又标 SAFE（57 帧）。

### 逐个追踪一集（seed 900101，停早 2.49 m）

学生前 12 步与教师逐帧一致，第 12 步（距 waypoint 1.77 m，**视野里有 3 根柱子**）
突然输出 SAFE，教师要求 YAW_R。所以这不只是「看不见」，还有「看得见也分不出远近」：
巷道网格是重复的，单帧图像无法唯一确定沿 x 的位置。

## 没做什么

- 没改任何代码、没重采、没重训。
- 没做 PO/EO/AGV/MP（仍未实现，`EVAL_STANDARD §5.2`）。
- 没给 `carry_resume_best` 补测试。
- 没提交、没推送：用户未授权。

## 待确认

- 修法三选一（详见下）需要用户拍板。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `audit_dataset.py`（v4 采集） | 17/17 PASS |
| 离线 test（874 样本） | acc 0.8478 / macro_f1 0.7740 / bal_acc 0.7623 |
| 文本探针（801 决定性帧） | flip 0.9638 / acc_correct 0.8814 / follow 0.8090 |
| 闭环 keep / shuffle | 完成，47/47，SR 0.426 / 0.000 |
| 闭环 盲前进 | 完成，44/48（3 集 infra 超时），SR 0.000 / 碰撞 0.455 |
| `test_eval_summary.py` | PASS（含新增的 1.4/1.6 m 边界与「到位但撞柱」） |
| `test_eval_metrics.py` | PASS |
| 2 集端到端复跑（900160 / 900101） | `success_criterion=goal_reach`，判定与复算一致 |

## 风险与遗留

- **v3 与 v4 的 SR 不可比**：任务定义不同，v3 的 0.667 是在「停止可观测、
  指令不可观测」的任务上测的；v3 的旧口径是 goal_stop，也和本文的 goal_reach 不同。
- 若直接放宽 `stop_radius`，教师数据也得重采（教师按 0.5 m 停），不能只改评测。
- **盲前进基线的分母小 3 集**：`900032/900052/900101` infra 超时，学生那边都完成。
  已另算共同完成的 44 集，报告以那一组为准。
- **`audit_dataset.py` 认证不了图像**：17 条检查只读 `actions/oracle/scene` 的
  jsonl，从不打开 PNG；`label_recompute` 是同一个 teacher、同一个 `pose_before`
  重算，对未改动数据数学上恒等。所以 17/17 PASS 只证明**原始 499 集内部自洽**，
  既不覆盖实际训练的 902 集配对包（反事实集的 oracle/collision/validity 是从原集
  复制的，按原集口径审必然 FAIL），也发现不了本次的「空帧同图标 FORWARD/SAFE」。
- **`probe` 的 `flip` 只说明「文本会改变动作」**，不说明改对了；严格指标是
  `follow`（0.8090）。报告里两个数要一起给。
- 代码审计发现但**未修**（本轮不影响 v4，因为 v4 没有 resume）：
  `carry_resume_best` 跨数据集冻结 best（`src/trainer.cpp:40-72`）；
  resume 静默继承旧 `class_weights`（`src/class_weights.cpp:134-142`）；
  训练末尾那次 test 用的是末轮模型而不是 `ckpt_best`（`src/trainer.cpp:284-295`），
  直接读 `metrics.jsonl` 最后一条会偏低。
