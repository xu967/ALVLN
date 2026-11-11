# 第十二阶段 W1-3：学生概率与帧间隔落盘

- 时间：2026-11-12 00:29（接在 W1-4 之后，见 `2026-11-11-2358-w1-4-block-survival.md`）
- 范围：`sim/tools/collect.py`（`collect_one`、`action_row`）、`sim/tools/eval_closed_loop.py`（`summarize_episode`、`run_episode` 的一处调用，新增 2 个函数和 1 个常量）；测试 `test_collect.py`、`test_eval_summary.py`、`test_eval_frame_gap.py`（新）、`test_action_latency.py`、`test_eval_blocks.py`
- 分支：`p12-w1-3`，从 `p12-w1-4`（75e85e2）起，所以带着 W1-4 的两个提交；本卡提交 `309fcc4`（实现与测试）、`a1a4fc7`（补零样本基线 one-hot 的测试与注释）、本条文档（见 `git log`）。不合并、不推送，`progress.md` 与 `CLAUDE.md` 没动。
- 详细报告：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/w1-3.md`

## 做了什么

1. `collect_one()`：学生分支原来把 `student.act` 的第三个返回值丢进 `_probs`，现在保留为 `probs`。非 None 时转成 4 个 `float` 并保留 4 位小数；这一步放在原有的 `try` 里，概率不是数字列表时在动作派发前判本步失败（文案「学生推理失败」），不会等动作发出后、组装行时才抛异常。返回的行先组装成 `row`，只在 `probs is not None` 时加 `"probs"` 键；教师、前向、规则等策略不给 `probs` 赋值，行里没有这个键。零样本基线（`baselines/baseline_infer.py`）经 `--policy student --infer-bin` 走同一条学生分支，它没有四类概率，`probs` 是 one-hot（`[0.0, 1.0, 0.0, 0.0]` 这类），所以基线的 `actions.jsonl` 行也有 `probs`（见「待确认」第 9 条）。
2. `action_row()`：可选键清单加 `probs`，所以学生行写进 `actions.jsonl` 时多一个 `probs`。位置在 `executed_by` 等可选键之后、W1-0 的 `dispatch_lag_s` / `late_dispatch` 之前，W1-0 的「排在最后」不变。模块 docstring 的字段表加一行。
3. `eval_closed_loop.py`：新增常量 `FRAME_GAP_FLAG_FACTOR = 1.2`、函数 `max_frame_gap_s(actions)`（相邻两行 `rgb_ts` 之差的最大值，秒；缺 `rgb_ts` 的行不桥接）和 `frame_gap_reference_s(actions, hold)`（给了 hold 用 hold，没给取行里 `sim_dt` 的中位数）。`summarize_episode` 加关键字形参 `hold=None`：有可算的相邻对时写 `max_frame_gap_s`，大于 1.2 × 参照时再写 `frame_gap_flag: true`，不大于则不写这个键。`run_episode` 的调用加一行 `hold=args.hold`：gz 采集写的 `actions.jsonl` 行没有 `sim_dt`，参照只能由调用方传。
4. 测试：
   - `test_collect.py`：新增 `probs_checks()` 12 项。学生行有 4 个概率、保留 4 位小数、和为 1 ± 1e-3；`action_row` 写出 `probs`，`dispatch_lag_s` 仍是最后一个键；学生行键集合 = 默认键 + `probs`；学生给 None 时不写键、步成功；`probs` 不是数字时判「学生推理失败」；零样本基线的 one-hot 照常落盘；β=1 执行教师动作时 `probs` 仍是学生的；教师与前向行没有 `probs` 键；主循环写出的学生与教师 `actions.jsonl`。原有一项断言（学生行的键集合）改成多 `executed_by` 与 `probs`。
   - `test_eval_summary.py`：新增 `frame_gap_checks()`（2.4 s 的轨迹旗为真，1.0 s 的无旗；1.2 s 边界不打旗，1.201 s 打旗；hold=0.5 时 1.0 s 打旗；`sim_dt` 后备，hold 优先于 `sim_dt`，`sim_dt` 取中位数；没有 hold 也没有 `sim_dt` 时只写间隔；缺 `rgb_ts` 的行不桥接；航点任务；新键恰为这两个、其余值不变）和 `old_outputs_unchanged_checks()`（旧数据、航点任务、一步集、空集的键集合冻结为改前输出的 28 个键）。
   - `test_eval_frame_gap.py`（新，5 项）：走完整 `ec.main()` 到 `eval.json`。默认 hold 下 2.4 s 的集旗为真，1.0 s 的集无旗；行里没有 `rgb_ts` 的集键集合不变；顶层键不变；旧键的值与没有时间戳的同一批轨迹相同；`--hold 2.0` 时 2.4 s 不打旗，`--hold 0.5` 时 1.0 s 打旗。
   - `test_action_latency.py`（W1-0 的文件）：2 处学生行键集合断言加 `probs`。`test_eval_blocks.py`：`run_eval` 加 `extra_argv`（新文件用它传 `--hold`）。

## 没做什么

1. 没有把帧间隔汇总到 `eval.json` 顶层（被打旗的集数、全部集的最大间隔）：卡只要求 `per_episode`。代价：看一次评测有没有断帧要遍历 `per_episode`。
2. 没改 `sim/3dgs/eval_3dgs.py`：它调用 `summarize_episode` 时不传 hold，走卡里写的后备（行里的 `sim_dt`，3DGS 的行 `sim_dt` 等于 `hold_s`）。副作用见「待确认」第 3 条。
3. 没改 `docs/EVAL_STANDARD.md`：卡没列；W1-2 的卡要在它的 §5 追加主比较协议，并行分支改同一文件容易冲突（W1-0 的那一节已经在 main 里）。新键的定义写在函数 docstring、本文和报告里。
4. 没写 `probs` 的置信度分析（置信度与错误的关系、校准曲线）：卡只要求落盘。
5. 没起 Gazebo，没用 GPU，没跑真实学生闭环：真实 `student` 行里 `probs` 的数值未核实（`src/infer.cpp` 的输出格式只读过代码）。验证用假学生加真的 `collect_one` / `action_row` / 主循环，帧间隔另在已有实验的真实轨迹上离线重算。
6. 没给 `probs` 做 `--save-probs` 之类的开关：卡写的是学生策略下默认加键。
7. 没跑 C++/CUDA 构建、CTest、GPU 测试、cf_sitl：`check.sh --python-only` 本来不含这些，它的末行也写明了（见「验证到哪一步」）。

## 待确认（主会话）

1. `probs` 在行内的位置：放在可选键（`executed_by` 等）一组里，所以在 `dispatch_lag_s` / `late_dispatch` 之前。如果要求 `probs` 排在最后，需要同时改 W1-0 的 `test_action_latency.py` 里「新键排在最后」那条断言。
2. `probs` 的容错：非 None 但不是数字列表时，本步在派发前判失败；学生返回 None 时不写键、步照常。卡没写这两种情况，是我定的。
3. 3DGS 评测（`eval_3dgs.py`）共用 `summarize_episode`：它的 `per_episode` 也会多 `max_frame_gap_s`（行里有 `rgb_ts` 与 `sim_dt`），可能多 `frame_gap_flag`。卡对 3DGS 一侧没有逐位不变的要求，这是共用函数的结果；如果 3DGS 一侧要保持原样，需要在 `eval_3dgs.py` 里去掉这两个键。
4. 没有 hold 时的后备取行里 `sim_dt` 的中位数：卡写「用行里的 sim_dt」，没写多行怎么合并。
5. `frame_gap_flag` 只在为真时出现；`max_frame_gap_s` 在一步集、空集、旧数据（行里没有 `rgb_ts`）里不出现；缺 `rgb_ts` 的行不桥接。阈值是严格大于 1.2 × hold：hold=1.0 时 1.2 s 的间隔不打旗。
6. 计划文档 `PLAN_2026-11-10_phase11.md` A6 写的是「超过 1.2 s 标记」（绝对值），卡写的是 `1.2 × hold`；hold=1.0 时两者相同，按卡实现。
7. `run_episode` 多改了一行（卡只点名 `summarize_episode`）：理由见「做了什么」第 3 条。
8. 新测试文件 `test_eval_frame_gap.py` 复用 W1-4 的 `test_eval_blocks.py` 里的夹具：本分支从 `p12-w1-4` 起，合并顺序是 W1-4 先于 W1-3 时没有问题；反过来，需要把 W1-4 的测试文件一起带上。
9. 零样本基线（`exp-baselines-zeroshot` 用的 `baseline_infer.py`）经 `--policy student --infer-bin` 走学生分支，所以它们的 `actions.jsonl` 行也有 `probs`，值是 one-hot。卡的条件是「学生策略下」，按字面实现。要让基线行不带 `probs`，需要 `collect_one` 区分真学生与基线推理进程，卡没写，没做。

## 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| 改前基线：相关 32 个测试文件逐个 `nice -n 10` | 全部 rc=0 |
| 先红：旧树（`git archive 75e85e2` 全量导出，源码是旧版）加本卡终版测试文件，逐个 `nice -n 10` | `test_collect.py` rc=1（改动的那一项断言失败）；`test_eval_summary.py` rc=1（`TypeError: summarize_episode() got an unexpected keyword argument 'hold'`）；`test_eval_frame_gap.py` rc=1（5 项里 1 FAIL、2 ERROR）；`test_action_latency.py` rc=1（24 项里 2 FAIL，学生行缺 `probs`）；`test_eval_blocks.py` rc=0（33 项，锁旧行为，旧树本来就绿） |
| 先红，逐检查项：旧树上跑 `probs_checks()` 的 12 项 | 8 项失败；4 项通过（教师行与前向行没有 `probs`，学生给 None 无键，主循环教师行无 `probs`，都是旧行为） |
| 后绿：新树（提交 `a1a4fc7`）33 个相关测试文件逐个 `nice -n 10`，含 `test_eval_frame_gap.py`、`sim/3dgs/test_collect_eval.py`（28 项，70 s）、`tools/test_preprocess_split.py`（41 s） | 全部 rc=0；`test_collect` 120 项通过（改前 108 项）、`test_eval_blocks` 33 项、`test_eval_frame_gap` 5 项、`test_action_latency` 24 项 |
| `actions.jsonl` 行差分：同一夹具下真的 `collect_one` 加 `action_row`，旧树与新树各写一行（教师、前向、学生 probs=None、学生） | 前 3 行逐字节相同；学生行只多 `"probs": [0.1235, 0.5, 0.2, 0.1765]`（原始输入 `[0.1234567, 0.5000004, 0.2000009, 0.176542]`），位置在 `dispatch_lag_s` 之前 |
| `summarize_episode` 差分：`exp-long-course-20261110` 的 96 集（7 个臂有逐集轨迹，`guard2-s0` 被中断，0 集），旧树与新树各重算，hold 取 `eval.json` 的 `hold_s` | 旧键的键集合与取值 0 处差异；旧树重算值与 `eval.json` 记录 0 处不一致；新树多 `max_frame_gap_s`：最小 0.900、中位 0.900、最大 1.000 s，0 集打旗 |
| 同上，`e1-pipeline-20261107-0417-eval4` 的 8 个臂 159 集 | 旧键 0 处差异，0 处与记录不一致；4 集打旗：教师臂 seed 536（2.4 s）、e1-new seed 929（2.2 s）与 990（5.5 s）、截断臂 seed 230（2.9 s）；e1-new seed 992 的 1.1 s 不超过 1.2 s，不打旗。与 `docs/reviews/2026-11-10/data-algo.md` C8 列的间隔（教师臂 2.40 s，e1-new 最长 5.50 s，截断臂 2.90 s）一致 |
| AST 逐个顶层单元比对（旧树对新树） | `collect.py`：只有 `collect_one`、`action_row` 变，其余 92 个单元不变；`eval_closed_loop.py`：只有 `summarize_episode`、`run_episode` 变，新增 `FRAME_GAP_FLAG_FACTOR`、`max_frame_gap_s`、`frame_gap_reference_s`，其余 63 个单元不变 |
| `git diff --check`、新文件行尾空白 | 通过、无 |
| `UAV_RGB_PYTHON=... bash tools/check.sh --python-only`（守护进程，W1-0 第二轮 `DONE`（00:31:43）之后在 `check.lock` 上排队，`nice -n 10`；跑的是 HEAD `bbeffa3`，工作树干净，检查期间没有改任何文件） | rc=0（00:58:06 拿到锁，01:02:03 结束）；日志里 34 个 unittest 汇总行合计 574 项，没有 Traceback；末行 `PASS（有 SKIP）：所选检查完成，但未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl；模拟采集、真实训练和学生闭环须独立验收。` |
| 同一命令的第一次（00:33:31 到 00:38:44，rc=0） | 不引用：这次跑的时候我正在补一项测试与注释并改本文档（提交 `a1a4fc7` 前后），工作树不是静止的；代码与测试内容与 HEAD 相同的可能性大，但没有逐项核对。日志留在草稿目录 `check.draft1.log` |

## 风险与遗留

- 真实学生评测下 `probs` 的数值未核实（没有跑 Gazebo 与真实推理）。`src/infer.cpp` 输出的是四类概率（softmax），原始概率之和为 1 时，行里保留 4 位小数后四项之和与 1 的差不超过 2e-4（4 × 5e-5）；小于 5e-5 的概率记为 0.0。
- 基线行的 `probs` 是 one-hot，不是置信度。以后用 `probs` 做置信度分析时，要先把零样本基线的臂（`--infer-bin` 指向 `baselines/launch_*.sh`）排除。
- 每个学生步的行多约 40 个字符；80 步的一集多约 3 KB。
- `max_frame_gap_s` 与 W1-0 的 `dispatch_lag_s` / `late_dispatch` 是两个量：前者是相邻两个观测帧的间隔，后者是派发的仿真戳减观测戳，互不替代。
- 打旗只说明相邻两帧间隔超过 1.2 × hold，不判定这一集是否作废。像 DAgger round 3 那样把停顿集剔除，由主会话定。
- 读取方用 `.get("frame_gap_flag")`：键只在为真时出现。旧实验目录里冻结的 `code/sim/tools` 快照没有这两个键，新实验起跑时冻结的代码才带。
- 3DGS 评测的 `per_episode` 多键，见「待确认」第 3 条。
