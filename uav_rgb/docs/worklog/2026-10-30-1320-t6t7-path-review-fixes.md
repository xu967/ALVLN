# T6/T7 链路审核与修复（独立 review + 主会话复核）

- 时间：2026-10-30 13:20
- 范围：`tools/preprocess.py`、`tools/check.sh`、`sim/tools/eval_closed_loop.py`、
  `sim/tools/test_eval_summary.py`（新）
- 背景：T5 全量采集中（600 集），等数据期间审核采集结束后要跑的训练/评测链路。
  独立 review agent 看 Python 侧（preprocess / eval_closed_loop / baseline_priors），
  主会话看 C++ 侧（train_main / trainer / trainer_eval / dataset / class_weights /
  eval_metrics / checkpoint / infer / model / run_owner）。

## 做了什么

| # | 发现 | 处置 |
| --- | --- | --- |
| 1 | `preprocess.py` 只查「同 gid 跨 split」，不查 `split == split_for_gid(gid)`；单集包内 manifest 被改过时无对照集，泄漏门形同虚设 | `load_episode` 加硬校验（gid 非空时）；`check.sh` 新增负例：把合成集改成「合法 gver + 错误 split」必须被拒 |
| 2 | `scan()` 不去重，同一 episode 目录经两个 glob/软链传两次会静默双打包（同批样本算两遍） | 按 `os.path.realpath` 去重并 die；`check.sh` 新增负例：同一目录传两次必须被拒 |
| 3 | `eval_closed_loop` 把 `collision`（= contact **或** stuck）直接当碰撞；`EVAL_STANDARD §5.1` 明确 stuck 不是碰撞（旧数据高估 8 倍） | `summarize_episode` 改为 contact-only 判碰撞与 false_forward，stuck 单列 `stuck_rate`；无 `src` 字段的旧数据退回原口径；新增 `test_eval_summary.py`（6 组断言） |
| 4 | 显式 `--seeds` 里同几何 seed 被跳过时完全静默 | 跳过时打印原因（`[seed N] 跳过: 几何 … 已评测`） |
| 5 | `baseline_priors` 不按 `collision_flags` 过滤，与学生 `--drop-collision` 不是同一批 val/test。T5 已实测出现 contact 步（13:22 时 9 步，见下） | 默认按 flags 剔除（与训练同口径），`--keep-collision` 保留旧口径；报告记 `collision_filtered/collision_dropped`；新增 `--self-check` 并接入 `check.sh` |

`src` 字段：`collect_one` 早已算出 `contact/stuck/contact+stuck`，但 actions.jsonl
写入白名单没带它，导致评测结束后无法区分。**collect.py 的修复要等 T5 采集结束**
（采集每个 episode 都新起进程导入该文件，不能在运行中改）。

## 验证

| 项 | 结果 |
| --- | --- |
| `py_compile`（preprocess / eval_closed_loop / 新测试） | PASS |
| `test_eval_summary.py`（contact / stuck-only / contact+stuck / 旧数据 / 空集） | PASS |
| `test_eval_closed_loop_args.py` | PASS |
| 新门禁扫 T5 pilot 52 集 | PASS（52 集 / 2080 步，0 丢弃，0.35 s） |
| 负例消息实测 | 重复目录 → `重复的 episode 目录…`；split 不符 → `split=val … 应属的 split=train` |
| `baseline_priors --self-check` | PASS |
| 合成数据集注入 2 个碰撞步 | 默认剔除 2 步（train），`--keep-collision` 保留；多数类随样本集变化 |
| `bash tools/check.sh --python-only` | PASS（含新增负例与 baseline 自检） |

## 没做什么

- **没改 `collect.py` 写 `src`**：T5 采集中，改它会污染同一数据集的 schema；
  T5 结束后补一行，再跑 T7（在此之前 eval 按旧口径回退，不会把历史结果改数）。
- **没把 T5 的 collision 步分开统计**：13:22 日志已见 9 步 `collision=True(contact)`
  （同一实例 x≈1.1 m 蹭柱后悬停），说明全量数据不是 0 碰撞；T6 仍按
  `--drop-collision` 丢步，整集是否要丢等 audit 的 `--exclude-collision-episodes` 结果再定。
- **没修 S1 的集数语义**：显式 seed 列表有重复几何时只跑唯一 gid 数、rc=1；
  T7 用冻结的 20 个 test seed（已复核全部 test 且 gid 唯一），不受影响。
- **没修 N 系列**（hold 不进 meta、`int()` 截断、infer_ms 只含前向、goal_margin
  未校验、num_classes 未断言等）：不在 T6/T7 关键路径，列在下面备查。
- **没跑全量 L2（C++ 构建 + CTest）**：T5 正在抢 CPU；本轮只改 Python，里程碑时再跑。
- **没有 commit**：用户未授权。

## 备查（未修）

- `preprocess` 不记录/不校验 `hold_s`、`planner`、`teacher_version`（混包时无迹可查）。
- `preprocess` 对 `action_expert=1.7`/`true` 用 `int()` 静默截断；`step` 缺字段抛 traceback。
- `inference_ms` 只是 C++ `forward()` 耗时，不含取帧/缩放/管道。
- `baseline_priors` 的「sklearn macro」注释不严谨（固定除 4；与训练器同口径，对比本身成立）。
- `eval_closed_loop` 未校验 `goal_margin` 有限正数；`--policy forward --ckpt` 会忽略 ckpt。
- `student.py`/`collect.py` 对 0..3 之外的动作静默悬停。
- `tools/` 下没有 `test_preprocess.py` / `test_baseline_priors.py` 单测。
