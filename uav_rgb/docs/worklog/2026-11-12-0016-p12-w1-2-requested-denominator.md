# 卡 W1-2：主表分母改请求集，写入主比较协议

- 时间：2026-11-12 00:16（`check.sh --python-only` 的结果在文末「更新」小节追加）
- 范围：`tools/stats_report.py`、`tools/test_stats_report.py`、`docs/EVAL_STANDARD.md`
- 分支：`p12-w1-2`，基于 main `a714b54`
- 结论：代码、单测、文档完成。无失败集时输出与改前逐位相同；有失败集时 SR 等越高越好的指标分母改为请求集。`check.sh --python-only` 按 11-11 23:15 的排队规矩延后，见「验证到哪一步」。

## 做了什么

- `tools/stats_report.py`
  - 新增 `uses_requested(metric)`：二值且越高越好的指标（SR、task_success、reach_ever、stop_success）取请求集为分母。越低越好的二值指标（CR、stuck、false_forward）与 SPL 仍用完成集。
  - 新增 `run_denominator(ev, n_done, requested)`：取 `eval.json` 的 `episodes_requested`，失败集与跳过集记 0；旧文件没有该字段时退回完成集；请求数小于完成数（不应出现）时取完成数，比例不超过 1。
  - `summarize_cell` 与 `summarize_all_scenes` 加 `requested` 参数。单 run 的 Wilson 用请求集作 n；多 run 的 IQM 与跨场景 IQM 按每个 run 各自的分母算比例。格里的 `n` 就是分母，分母与完成数不同时另记 `n_done`。
  - `build_report` 在报告 JSON 里加 `denominators`（每个指标是 `requested` 还是 `done`）与 `failed_total`（每个方法全部 run、全部场景的失败集数之和）。
  - `to_markdown` 在主表末列加「失败集数」，表注改写为两种分母的说明。模块文档的口径段同步更新。
- `tools/test_stats_report.py`
  - `_fake_eval` 的默认值改为 `episodes_requested = 完成数`、`failed_count = 0`，既有断言里的 n 不变。
  - 新增 `test_success_metrics_use_requested_denominator`：请求 20、完成 19、失败 1 时，SR 为 15/20（`n=20`、`n_done=19`，Wilson 区间按 20 算），task_success、reach_ever、stop_success 的 n 为 20，CR、false_forward、stuck、SPL 的 n 为 19，`failed_total` 为 1；无失败集时格里没有 `n_done`；缺 `episodes_requested` 的旧文件退回完成集；多 run 的 IQM 与跨场景 IQM 数值；Markdown 末列与表注内容。
- `docs/EVAL_STANDARD.md`：在 §5.5 之后、§6 之前追加 §5.6「主比较协议与统计表分母」。内容：统计表分母规则（写明覆盖 §5 开头列表「SR 分母 = `episodes_done`」、§5.3 指标表 task_success 行、§5.3「统计方法」第一条这三处的分母写法，原文不改）；主比较规模（20 个冻结几何 × 3 个训练种子 = 60 集，配对精确 McNemar 加 Wilson 区间，T1 起再报逐块存活）；预注册（起跑前写下预测与证伪条件，主比较配对不超过 3 个，不在评测 seed 上调参）；单次 n=20 的比较不下结论。

## 没做什么

- 没改配对检验。`paired_outcomes` 仍在两方法都完成的共同 seed 上配对，失败 seed 被剔除而不是记 0。原因：卡只要求主表分母。配对里失败 seed 怎么算是设计问题，见「待确认」。
- 没动 §5.5（W1-0 加的 `--action-latency` 口径）、§5 开头列表与 §5.3 的原文。原因：任务要求。与新分母冲突的三处（§5 开头列表「SR 分母 = `episodes_done`」、§5.3 指标表 task_success 行、§5.3「统计方法」第一条）留作原文，由 §5.6 的第一条声明覆盖。`eval.json` 的 `success_rate` 字段仍是完成集分母，没改。
- 没改 `sim/tools/eval_closed_loop.py`。原因：它的 `aggregate` 已经输出 `requested_success_rate` 与 `wilson_ci.requested`，本卡只改读端。
- 没重生成历史统计表，没重评任何 run。原因：卡未要求；也不占 gz。
- 没改 `tools/compare_datasets.py`、`dagger_gz_pipeline.py`。原因：它们只调用 `build_report`、`to_markdown`、`mcnemar`、`paired_outcomes`；表里多一列，配对函数没变。`test_compare_datasets.py` 与 `test_dagger_gz_pipeline.py` 在改后通过。
- 没改 `progress.md`、`CLAUDE.md`、`ui/`、主仓库工作树。原因：执行规则。

## 待确认

- 配对检验里失败 seed 的算法。现在两方法共同完成的 seed 才配对，一方在某 seed 上基础设施失败时，该 seed 不进配对。若按「失败记 0」处理，配对表的分母才与主表一致，这是卡之外的设计变更，由主会话定。
- 所有集都失败的 run（完成数为 0）仍显示「—」，不显示 `0.00 (n=请求数)`。原因：该格在改前就是「—」，「失败集数」列和下表「失败」列携带这个信息。要不要改成 0/请求数，由主会话定。
- §5.6 里「同权重同 seed 重评的逐集不一致率约 24%」取自任务卡，出处写的是 `docs/reviews/2026-11-10-phase12/longcourse.md` 第 103 行的「自己重跑约 76%」。该处是预注册文字，这个一致率有没有被实测，未核实。
- §5.6 紧接在 W1-0 加的 §5.5 之后。W1-0 若同时改 §5.5 末尾或 §6 开头，合并时可能在相邻行产生文本冲突，需要人工解。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python tools/test_stats_report.py`（新测试 + 旧实现） | 红：新用例 SR 的 n 为 19、点估计 0.789；其余 9 项 PASS（`$D/tmp/phase12-cards/w1-2/red.txt`） |
| `python tools/test_stats_report.py`（新实现） | 绿：10 项全部 PASS（`green.txt`） |
| `python tools/test_compare_datasets.py` | PASS（调用 `build_report`、`to_markdown`） |
| 真实输入 1：`$D/exp-swa-calib-20261110/calswa-s0/eval.json`（20 请求、20 完成、0 失败）跑 `stats_report.py` | SR `0.60 [0.39, 0.78] (20)`，CR 与 false_forward `0.20 [0.08, 0.42] (20)`，SPL `0.57 [0.38, 0.77] (20)`，失败集数 0（`real-run.txt`、`real/`） |
| 真实输入 2：`$D/exp-action-latency-20261110/acc-L0.16-k1-s1/eval.json`（20 请求、19 完成、1 失败、14 成功），main 的与改后的 `stats_report.py` 各跑一次 | 改前 SR `0.74 [0.51, 0.88] (19)`；改后 SR `0.70 [0.48, 0.85] (20)`，CR 仍 `0.21 [0.09, 0.43] (19)`，失败集数 1（`failed-old.txt`、`failed-new.txt`） |
| `bash tools/check.sh --python-only` | 未跑：排队规矩要求等 W1-0 第二轮结束后用守护脚本串行拿 `check.lock`，结果见下方「更新」小节 |

草稿与日志目录：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-2/`。

## 风险与遗留

- 读 `stats_report.json` 的下游若依赖成功类指标的 `n` 等于完成集，会在有失败集的输入上读到不同的值。仓库内的调用方只有 `compare_datasets.py` 与 `dagger_gz_pipeline.py`（后者起子进程只落盘不解析），仓库外的脚本未核实。
- 同一张表里 SR 与 CR 的 n 不同时（有失败集），读者容易把两列当同一分母。表注与「失败集数」列已写明；`n_done` 只在 JSON 里。
- `uses_requested` 把「越高越好的二值指标」一概视为成功类。以后新增二值指标时，要在 `METRICS` 里按方向登记，否则分母会按方向自动归类。

## 更新（2026-11-12 01:25）：`check.sh --python-only`

守护脚本串行等 `check.lock`，01:14:01 拿到锁（排在 W1-1、W1-5 的两次之后，中间夹着别的会话的几次）。

| 项 | 结果 |
| --- | --- |
| `UAV_RGB_PYTHON=<主仓库 .venv-tools>/bin/python bash tools/check.sh --python-only`（`nice -n 10`，`PYTHONDONTWRITEBYTECODE=1`，`TMPDIR` 在数据盘；工作树 `p12-w1-2`，HEAD `3ab7aaa`） | rc=0；311 s（01:14:01 到 01:19:12，不含排队）；起跑时负载均值 13.35、16.83、19.31 |
| 末行 | `PASS（有 SKIP）：所选检查完成，但未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl；模拟采集、真实训练和学生闭环须独立验收。` |
| 日志里本卡的测试 | `test_stats_report: PASS`，其中有 `ok test_success_metrics_use_requested_denominator`；表下的说明行是新的分母口径（原文：`越高越好的指标（SR、task_success、reach_ever、stop_success）n = 请求集 episodes_requested，失败集与跳过集记 0`） |
| 日志里的异常词 | `Traceback`、`^FAIL` 都是 0 行；`^SKIP` 2 行：第 1 行是 `--python-only` 的声明，第 878 行是 `test_nurec_to_ply` 依赖 msgpack 的用例未跑（本解释器没有 msgpack），与本卡无关，W1-1 的日志里也有同样的两行 |

日志：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-2/check-python-only.log`（1273 行），完成标记 `check-python-only.done`。
