# 2026-11-02 ~ 11-03 特权蒸馏 A/B 与 1000 集扩量复盘

- 范围：教师 warm-up 修复、首轮 400 集蒸馏 A/B、第二轮 1000 集三组训练与 a50 OOM。
- 性质：流程复盘，不改代码、不训练、不跑评测。所有时间均为本机 CST（UTC+8）。
- 证据约定：`transcript` 指 `/root/.claude/projects/-root-claude-code-backup-20261028-code/c15974aa-cbdb-450a-a2e8-a7c6ee6b2784.jsonl`，时间戳为 UTC，文中已换算为 CST。
- 结论先行：A 的根因描述准确，排查从闸门失败到根因 57 分钟，但此前已有一条被搁置的同类警告；
  B 的 20 集闭环在配对设计下**最小可检测差异约 30 pp**（低于此测不出），跑之前没有做最小可检测差异（MDE）计算；
  C 没有代码泄漏证据，最可能是同机外部进程抢占，但缺少崩溃时刻的 per-process 显存证据，**无法证实**。

## 1. 事实时间线

| 时间 | 事件 | 证据 |
| --- | --- | --- |
| 11-02 04:46 | 3 集教师闭环冒烟：全部在距航点 ~1.8 m 提前停，`SAFE 9 / FORWARD 38 / YAW 0`。已写下“可能是时序缓冲开头用最早帧填充”的警告，但决定等 400 集重训后再查 | transcript `2026-11-01T20:46:24Z`；`docs/worklog/2026-11-02-0830-teacher-closed-loop-diagnosis.md:16-20` |
| 11-02 06:19–07:25 | 400 集教师训练（旧 `dataset.py`，只用满 8 帧真历史） | `/tmp/train_teacher_400.sh` mtime 06:19；transcript `2026-11-01T23:25:38Z` |
| 11-02 07:25–07:43 | test split 20 集教师闭环闸门：SR 0.0，316 步 0 次转向，每集 14–17 步就因连续 3 次 SAFE 结束 | `docs/worklog/2026-11-02-0830-teacher-closed-loop-diagnosis.md:5-20` |
| 11-02 07:45–08:25 | 依次排除：yaw 单位（一致）、输入管线（闭环帧离线重放 17/17 一致）、场景泛化（train split 闭环 SR 0.1、仍 0 转向）；同时启动类别加权重训 | worklog `:22-30`、`:72-90`；transcript `2026-11-02T00:25:41Z` |
| 11-02 08:40 | warm-up 探针（35 s）：val 33 集 582 步，step 0 专家转向率 0.697，模型预测转向率 0.000，准确率 0.303；step 3 以后恢复到 0.88–1.00 | transcript `2026-11-02T00:40:53Z`–`00:41:34Z`；worklog `:77-90` |
| 11-02 08:41 | 修 `teacher/dataset.py`：每集在常规滑窗前补 `seq_len-1` 个 warm-up 样本；加回归测试。杀掉已跑到 ep21 的类别加权重训（macro-F1 0.674 < 未加权 0.728） | `teacher/dataset.py:266-284`；`teacher/test_dataset_windows.py:59-74`；transcript `2026-11-02T00:42:22Z` |
| 11-02 08:42–09:49 | warm-up 版教师重训（5906 train / 582 val，20 epoch，约 67 min），best val macro-F1 0.8198 | `/root/rivermind-data/uav_rgb/teacher-run-400-warmup.log`；worklog `:96-108` |
| 11-02 09:50–10:09 | step 0 准确率 0.303→0.788、预测转向率 0→0.909；test 20 集闸门 SR 0.65（13/20），碰撞 0.25 | worklog `:96-119`；transcript `2026-11-02T01:50:03Z`、`02:09:37Z` |
| 11-02 10:11–11:32 | 教师补机身系目标向量（pose 6→8 维），训练到 ep10（val macro-F1 0.8991）后因 GPU 被其他会话占用提前收尾；同 20 场景闸门 SR 0.95（19/20） | worklog `:110-122`；transcript `2026-11-02T02:46:18Z`、`03:15:24Z` |
| 11-02 11:34–12:46 | 首轮蒸馏 A/B：328 train / 33 val（400 集实为 361 集可用，39 集 test 未入包），导出 5906+582 条软标签（6488/6488 覆盖）；base 2076.7 s，distill α=0.5 2089.0 s | `/tmp/distill_pipeline.log:1-53`；`distill-ab.md:9-14` |
| 11-02 12:46–18:50 | **6 h 04 min 空转**：流水线 12:46 已完成，等待器 `pgrep -f "distill_pipeline.sh"` 匹配到自己的 shell，永不退出；直到用户 18:49 追问才启动闭环 | `/tmp/distill_pipeline.log:53`；`/tmp/eval_ab.log:1`；transcript 摘要 `2026-11-02T23:28:41Z` |
| 11-02 18:50–19:24 | 首轮闭环 20 集：base SR ever 0.600（12/20），distill 0.650（13/20），McNemar 精确 p=1.0000；碰撞 0.25 vs 0.30 | `/tmp/eval_ab.log`；`docs/experiments/2026-11-02-distill-ab.md:45-56` |
| 11-03 00:14–00:21 | 400+600 集合并为 1000 集（815 train / 92 val / 93 test），打包 `ds-teacher1000`：18056 样本（14733/1607/1715）；软标签 18056/18056 全覆盖，T=2.0 | `../docs/PLAN_2026-11-02_uav_rgb_restructure_distill_gru.md:730-745`；`ds-teacher1000/soft_targets.json` |
| 11-03 00:22–01:44 | `student-base` 20 轮完成：294660 样本，4966.2 s，best dev macro-F1 0.8139（ep17），test acc 0.8688 | `/tmp/train_students_1000.log:1-25`；`student-base/metrics.jsonl` |
| 11-03 01:44:53 | `student-a50` 启动（α=0.5） | `/tmp/train_students_1000.log:24` |
| 11-03 01:47:56 | 同机另一会话的 YOLO 训练 `glp_50k_fix4_e20` 第 6 轮结束（由 results.csv 累计时间反推；该 run 启动于 21:42:04） | `/root/rivermind-data/YOLOVOmodal_3871538535/runs/test/glp_50k_fix4_e20/results.csv` |
| 11-03 01:48:50 | a50 第 1 轮结束，`ckpt.bin` / `ckpt_best.bin` 落盘（dev macro-F1 0.6311） | `student-a50_oom-20261103T0148/metrics.jsonl`；文件 mtime |
| 11-03 01:49:03 | 第 2 轮第 15 步（step 246，累计 249.68 s）抛 `CUDA error src/nn_cuda.cu:45: out of memory` | `/tmp/train_students_1000.log:26`；OOM run `metrics.jsonl` 尾行 |
| 11-03 01:49–07:30 | **5 h 41 min 空转**：等待器是 `until grep -q "全部完成"`，a50 失败后脚本 `set -eu` 直接退出，a80 根本没启动，等待器没有失败分支，永不退出 | `/tmp/train_students_1000.sh:4`、`:11-17`；transcript `2026-11-02T17:45:37Z`、`23:27:55Z` |
| 11-03 07:27 | 上一会话结束，2 个后台 shell waiter 被标记 stopped（`ba1en2vcx`、`bqbvrk5ru`）；**不是训练进程** | transcript `2026-11-02T23:27:55Z` |
| 11-03 07:30:22 | a50 重启；30 s 显存采样前 8 个点：1 → 159 → 275 → 201 → 273 → 159 → 253 → 219 MiB（21.5 GiB 卡） | `/tmp/train_students_1000_b.log`；`/tmp/gpu_mem_a50.log` |

## 2. 根因

### A. 教师闭环 SR=0

**症状**：离线 val macro-F1 0.73，闭环 20 集 SR 0.0，316 个动作里 YAW 为 0，每集 14–17 步就结束。

**排除掉的假设**（都有实测，不是“觉得”）：

| 假设 | 排除证据 |
| --- | --- |
| yaw 单位/坐标系错 | `collect._yaw_deg` 存度，`dataset.py` 与 `teacher_policy.py` 都转弧度；闭环重放一致 |
| 输入管线 RGB/深度预处理错 | 用 `--save-depth` 留下闭环帧，按同顺序离线重放，17/17 动作完全一致（worklog `:24-28`） |
| 场景泛化不足 | 在训练过的 train split 上闭环：SR 0.1（2/20），仍然 0 次转向（worklog `:72-75`） |
| 类别不平衡导致转向被淹没 | 类别加权重训到 ep21，macro-F1 0.674 < 未加权 0.728，SAFE 召回从 0.889 掉到 0.737，放弃（worklog `:100-103`） |
| 停止判据有问题 | 停止确实是直接死因（连续 3 SAFE 结束），但 SAFE 只在距航点 ≤0.5 m 出现；模型在 1.7 m 判 SAFE 是“转向缺失后的下游症状”，不是独立根因（worklog `:16-20`） |

**真正根因**：训练样本只产“满 `seq_len` 帧真历史”的滑窗；闭环 `_FrameBuffer.build` 在缓冲未满时用
`frames[0]` 重复补齐（`sim/tools/teacher_policy.py:112`）。闭环第 0/1 步的窗口是
`[f0]×(seq_len-i-1)+[f0..fi]`，训练从没见过这种拼接。专家恰好在这两步有 70%/58% 是转向，
模型在这两步预测转向率 0.000 → 一开局直飞 → 进入专家从不经过的状态 → 再也回不来。
修复是在常规滑窗前补 `seq_len-1` 个同拼接的 warm-up 样本（`teacher/dataset.py:266-284`），
顺序与 `_FrameBuffer` 逐帧一致，回归测试断言了窗口内容（`teacher/test_dataset_windows.py:59-74`）。

**描述核对**：任务里给的描述与代码/文档一致。两点补充：
1. 该 commit（`aa9bf85`）同时加了“机身系目标向量”，所以 SR 0→0.95 是两步：warm-up 单独到 0.65，
   再加目标向量到 0.95。不要把 0.95 全记在 warm-up 上。
2. warm-up 同时把训练样本 3610→5906、YAW 占比 12.6%→约 18%，**没有做“只补窗口、不增 YAW 占比”的消融**，
   所以 macro-F1 0.728→0.8198 中有多少来自拼接、多少来自类别平衡，未查清。

**排查耗时与弯路**：

- 闸门失败（07:43）→ 根因锁定（08:40）：**57 分钟**；探针本身只跑了 **35 秒**。
- 但 04:46 的 3 集冒烟已经点名“时序缓冲开头用最早帧填充”，当时判断“可能是 43 集样本不足”，
  先花 66 min 重训 400 集、再花约 18 min 跑闸门，才回到同一个假设。
- 闸门失败后又先查 yaw 单位、再启动类别加权重训（57 min，跑到 ep21 放弃），
  之后才做 35 秒的 warm-up 探针。
- 可归因于“没先测已写下的假设”的 GPU 时间约 **2 小时**（66 min 被推翻的旧教师 + 57 min 类别加权），
  外加约 1.5 h 墙钟。探针成本只有 35 秒，这是本轮最贵的一次“先跑后想”。

### B. 首轮 A/B 的统计效力

**症状**：20 集闭环，base 0.600 vs distill 0.650（差 1 集），McNemar 精确 p=1.0000；
离线报的是 val macro-F1 0.7648→0.7860（+2.1 pp），而 val 正是选权重的 split。

**排除“代码/配置不一致”**：两组命令只差 `--soft-targets/--distill-alpha`，seed/batch/lr/epochs/
`--drop-collision` 完全相同；软标签 6488/6488 覆盖，行和归一化通过。
所以 p=1.0 不是实现 bug，是样本量问题。

**根因（统计）**：这是配对设计，McNemar 只看不一致对。n=20 时，精确双侧检验要 p<0.05，
最少需要 **6 个同向不一致对（6:0，p=0.03125）**；5:0 已经是 p=0.0625。
也就是说，**净差异不到 6 集（30 pp）时，n=20 根本不可能显著**。
观测到的差异是 1 集（5 pp），p=1.0 是设计决定的，不是“蒸馏无效”的证据。

| 闭环集数 n | 6:0 时对应的最小可检测差异 | 说明 |
| --- | --- | --- |
| 20 | 30 pp | 本轮实际规模；<30 pp 测不出 |
| 40 | 15 pp | 第二轮计划规模；仍测不出 <10 pp |
| 60 | 10 pp | 最乐观情形（无反向不一致对） |

以上都是 6:0 的最乐观情形；80% 功效需要更大的 n，或更多同向不一致对。

**流程根因**：跑之前没有算 MDE。计划文档已经把“只有 20 个 test 场景，统计力不足”
列为 40% 概率的风险，并写了“扩到 ≥47 个、报 Wilson”（`../docs/PLAN...md:552`），
但没有把它变成“n 不达标就不跑闭环”的门槛。Wilson CI ±20 pp 是在结果出来后才补算的
（transcript `2026-11-02T13:32:49Z`）。第二轮把闭环加到 40 集，但 40 集的最小可检测差异仍约 15 pp，
计划里“40+ 集”不足以支撑 <10 pp 的结论（`../docs/PLAN...md:744-745`）。

**离线口径问题**：首轮 400 集包只有 train+val，**0 test**（`distill-ab.md:9-14`）；
学生 best checkpoint 按 dev/val macro-F1 选（`src/trainer.cpp:313-317`），
再用同一个 val 报 +2.1 pp，属于“选完再报”，偏乐观且没有置信区间。
这不是“数字错了”，是这个数字不能作为蒸馏有收益的证据。

**代价**：两组训练 69.4 min（2076.7+2089.0 s）+ 40 集闭环 34 min（18:50:13–19:24:10），
合计约 1.75 h GPU/仿真。产出不是零（打通了软标签→C++ 蒸馏链路，暴露了终点前撞柱），
但作为“有没有收益”的判据无效。

### C. 第二轮 a50 OOM

**症状**：`student-a50` 01:44:53 启动，第 1 轮正常（dev macro-F1 0.6311），
第 2 轮第 15 步（step 246，累计 249.68 s）在 `nn_cuda.cu:45` 的 `DeviceBuf` 构造里
`cudaMalloc` 失败。base 同脚本连跑 20 轮（4620 步）无异常。

**先排除“a50 的软标签路径更吃显存”**：

- `src/ops_soft.cpp` 是纯 CPU：读 `logits.data()`（`Tensor` 是 host `std::vector<float>`）
  和 host 的 `q`，没有任何 `cudaMalloc`（`src/ops_soft.cpp:58-107`）。
- 但任务里的“只会更少”略强：a50 的 `q` 非空，训练循环每步会额外调用
  `softmax_ce_loss_cuda(logits, y)` 写 CSV 的 `train_ce`（`src/trainer.cpp:265-268`），
  这会分配 3 个很小的 device buffer（64×4 float + 64 int + 8 B ≈ 1.3 KB/步）。
  246 步约 0.3 MB，**不足以解释 22 GB**；但它确实是一个 GPU 路径差异，不能声称“严格更少”。
- `DeviceBuf` 是栈上 RAII，析构 `cudaFree`，没有 static/thread_local 缓存
  （`src/nn_cuda.cu:42-50`）。base 用同一套 conv/linear/nn 分配跑了 4620 步，
  若这些路径泄漏，base 先崩。

**代码泄漏嫌疑：低。** 没有证据支持；`DeviceBuf` 无持久缓存，a50 的额外分配是 KB 级且有界。
唯一无法从代码排除的是“每步 `cudaMalloc/cudaFree` 的碎片化 + 外部占用”叠加，但这需要显存曲线证据。

**外部抢占：最可能，但未证实。** 本轮找到一条时间高度吻合的外部进程：

- 同机另一会话在跑 YOLO 训练 `/opt/conda/bin/python train.py`，run 目录
  `/root/rivermind-data/YOLOVOmodal_3871538535/runs/test/glp_50k_fix4_e20/`。
- 由 `results.csv` 累计时间反推：该 run 启动 21:42:04，epoch 6 结束 **01:47:56**；
  a50 在 **01:48:50** 存完第 1 轮 checkpoint、**01:49:03** OOM，相差 67 秒。
- 该会话在 19:22 实测 GPU `16186 MiB / 22528 MiB`，且其自己的 review 记录了
  “每 epoch 约 30 万个 GPU 张量常驻”的累积问题（transcript `c88acd31...`，`2026-11-02T11:22Z`、
  `13:36:35Z`）。它的 epoch 边界/验证/checkpoint 与 a50 的 epoch 边界正好撞上。

但这仍不是证明：YOLO 常规占用 16.2 GB 时还留约 6 GB，a50 只需 <300 MiB；
要让它成为充分原因，需要证明它在 01:48 的验证/保存阶段冲到了接近 22 GB，或叠加了第三个进程。
**崩溃时刻没有 `nvidia-smi` 采样，无法事后证伪。**

**当前重跑基线**：`/tmp/gpu_mem_a50.log` 的采样点全在 **159–275 MiB**，
占 22528 MiB 的约 1%，且没有单调上升。这只能说明“这条代码路径正常时很小”，
不能反推 01:48 的外部占用；需要跑满 20 轮看形态。

**区分外部/泄漏需要的证据**：

1. 采样必须带 **per-process 归属**：
   `nvidia-smi --query-compute-apps=pid,used_memory,process_name`，而不只是 `memory.used`。
2. OOM 报错必须带 **free/total/requested bytes**：在 `DeviceBuf` 构造失败时调
   `cudaMemGetInfo`，把 `free/total` 和本次 `bytes` 写进异常。
3. 判据：
   - 外部：total used ≈22 GB，本进程 PID 只占 ~0.2 GB，另一个 PID 占 ~20 GB；本进程曲线平。
   - 泄漏：本进程 PID 随 step/epoch 单调上升，即使没有别的进程 free 也持续下降。
   - 碎片：本进程 used 不高但 free 很小、且大块分配失败、小块成功。
4. 预检：启动前 `memory.used > 1024 MiB` 就等待或报错，不要和别人的训练抢卡。

## 3. 流程问题

1. **先跑后想，没有先算 MDE**（B）。20 集配对设计的最小可检测差异约 30 pp，却在跑完才补 Wilson/McNemar；
   计划里的“统计力不足”风险没有变成门槛。
2. **用参与选权重的 val 报离线收益**（B）。`src/trainer.cpp:313-317` 按 dev macro-F1 存 best，
   `distill-ab.md:36-41` 又用同一 val 报 +2.1 pp；没有独立 test，也没有置信区间。
3. **等待器不可靠，造成两段共约 11 h 45 min 空转**（本轮最大代价）：
   - 首轮 `pgrep -f "distill_pipeline.sh"` 匹配到自己的 shell，12:46 完成后不退出，
     直到 18:50 用户追问才启动闭环 → **6 h 04 min**。
   - 第二轮 `until grep -q "全部完成"` 没有失败分支，a50 失败后永远等不到 → **5 h 41 min**。
   两段都不是“训练被会话杀掉”，是**完成检测写错了**。
4. **驱动脚本全在 `/tmp`，没有进仓库**。本轮至少 **20 个** shell 驱动
   （`collect_teacher_{pilot,400,600}.sh`、`train_teacher_{pilot,400,400_cw,goal}.sh`、
   `eval_teacher_{400,goal,train_split,warmup}.sh`、`distill_pipeline.sh`、`eval_ab.sh`、
   `eval_student_same20.sh`、`train_students_1000{,_b}.sh`、`eval_test_offline.sh`、
   `eval_1000.sh`、`export_join_1000.sh`、`run_v4_goalreach.sh`），
   `git ls-files experiments` 里只有 `cuda_smoke/` 和 `pipeline_profile/`。
   脚本还硬编码 `/tmp/build-soft/uavrgb_train`（`/tmp/train_students_1000.sh:7`），
   干净 clone 无法从仓库复现这个二进制。
5. **一个组失败拖垮整条链**。`train_students_1000.sh` 用 `set -eu` 顺序跑 base/a50/a80，
   a50 OOM 后脚本直接退出，a80 根本没启动；没有 per-group 状态、没有失败继续、没有自动重试。
6. **后台任务在会话结束确实会丢，但本轮丢的不是训练**。transcript 只记录了
   `ba1en2vcx`、`bqbvrk5ru` 两个 waiter 被标记 stopped；400 集 A/B 和 1000 集 base 都完成了。
   真正导致 a50/a80 重跑的是 OOM + `set -eu` + 无失败分支的等待器。
   用户口径里的“两轮训练被会话杀掉”在本轮证据里**不成立**；历史上 R01/D4 同版本复跑
   曾在 14/20 被会话中断、需要重跑（`docs/HANDOFF.md:32`），那是另一回事。
7. **教师排查没有先做最小探针**。04:46 已经写下 warm-up 假设，仍先花 66 min 重训、
   再花 57 min 试类别加权；35 秒的逐帧重放探针最后一次命中。
8. **GPU 无预检、无 per-process 采样**。a50 崩溃时没有“谁在占卡”的记录；
   OOM 报错只有 `out of memory`，没有 free/total/bytes，导致 C 只能停在“最可能”。
9. **第二轮计划用 test 选模型**：`../docs/PLAN...md:744-745` 写“先用 test split 比 macro-F1，
   只把最好的送闭环”。如果再用这个 test 数字报离线指标，被选中组的 test 就被选择污染了。
   test 应只用于最终报告，筛选应在 val 上做（或明确标注 test 是筛选集）。

## 4. 下次怎么改

### 4.1 驱动脚本入库（最小 3 个文件）

新增 `experiments/scripts/`，只放**参数化驱动**，不放数据/权重/日志：

- `train_students.sh`：参数 `--data <uavr> --out <dir> --groups base,a50,a80 --alpha 0.5,0.8`；
  `source env.sh`，用 `$DATA_ROOT` 和 `$REPO/build/uavrgb_train`，不再硬编码 `/tmp/build-soft`。
- `eval_offline.sh`：参数 `--data --ckpt --split --out`，只报指定 split。
- `eval_closed_loop.sh`：参数 `--ckpt --episodes --seeds-file --out`；把 `eval_1000.sh` 里
  “扫 seed 找 test 几何”的逻辑收进来，seed 列表落盘成 `seeds.txt` 随实验文档提交。

每个实验文档必须写：完整命令、数据/ckpt 路径与 sha256、seed 文件、n、MDE。
这样 `/tmp` 被清掉也能从仓库重建；数据/权重仍按仓库规则留在 `$DATA_ROOT`。

### 4.2 把“等完成”改成状态文件

不要再用 `pgrep -f` 或 `grep "全部完成"` 当完成信号（前者会自匹配，后者没有失败分支）。
每个 driver 在每 stage 结束写 `run_status.json`：

```json
{"stage":"student-a50","status":"ok|failed","exit_code":0,"started":"...","ended":"...","log":"..."}
```

`experiments/scripts/wait_run.sh` 只读这个文件，支持 `ok`、`failed`、超时三种退出；
失败时立刻打印最后 20 行日志和 `nvidia-smi`，不要静默睡到天亮。

### 4.3 `train_students.sh` 的失败隔离

去掉“一组失败就整体退出”的 `set -e` 语义；每组独立：

```bash
for g in base a50 a80; do
  if ! "$BIN" ... --out "$OUT/student-$g"; then
    echo "FAILED $g" | tee -a "$LOG"
    continue
  fi
done
```

a50 失败时 a80 仍能跑；失败组保留目录（本轮已把 OOM run 改名为
`student-a50_oom-20261103T0148`，这个做法对）。

### 4.4 统计门槛写进 `docs/EVAL_STANDARD.md`

在闭环 A/B 前强制填一张小表：

| 字段 | 要求 |
| --- | --- |
| 对照集 n | 当前 n |
| 基线 SR | 来自 eval.json |
| 预期效应 | 例如 5–10 pp |
| 最小可检测差异 | 由 n 和配对设计算；n=20 约 30 pp，n=40 约 15 pp，n=60 约 10 pp |
| 决策规则 | n 不达标时只报“探索性”，不下“有效/无效”结论 |

低于 MDE 的实验可以跑，但必须标 `exploratory`，不能写成“蒸馏无收益”。
离线筛选用 val，test 只在最后报一次；不要用 test 选三组里的“最好”。

### 4.5 GPU 预检与 OOM 证据

- 所有训练 driver 启动前：`nvidia-smi --query-gpu=memory.used`，>1024 MiB 就等待/报错。
- 训练中每 30 s 记：`memory.used` + `--query-compute-apps=pid,used_memory,process_name`。
- `src/nn_cuda.cu` 的 `DeviceBuf` 构造失败时，调 `cudaMemGetInfo` 并把
  `requested_bytes/free/total` 写进异常；这样下一次 OOM 一行就能区分“外部占用”和“自己泄漏”。
- 训练失败时 driver 立刻 dump `nvidia-smi` 和最后一条 `metrics.jsonl`，不要只留一句 `out of memory`。

### 4.6 教师类问题：先跑最小探针

本次的 warm-up 探针只有 35 秒（33 集 582 步逐帧重放），却能一次命中。
以后遇到“离线好、闭环坏”的时序模型，先做同协议逐帧重放，再决定是否重训；
重训前把探针结果写进 worklog，避免用 66 分钟训练去验证一个 35 秒就能验证的假设。

## 5. 没做的 / 没查清的

- **C 的外部抢占没有证实**。没有 01:48 的 `nvidia-smi`；YOLO 训练的时间吻合是强线索，
  但“它当时占了多少显存、是否叠加了第三个进程”无法从现有日志证明。
- **warm-up 与类别平衡的贡献没有拆开**。warm-up 同时增加了样本数和 YAW 占比，
  0.728→0.8198 中有多少来自“见过首帧重复拼接”、多少来自类别平衡，没有消融。
- **首轮 6 h 空转的“等待器自杀”来自 transcript 摘要**（`2026-11-02T23:28:41Z`），
  没有该 waiter 被 kill 的逐秒记录；时间线（12:46 完成、18:50 才开闭环）是确定的。
- **两个 stopped waiter 的具体死因不明**。通知说可能是 UI、Monitor timeout 或进程退出，
  没有 completion record；只能确认它们不是训练进程。
- **a50 的 OOM 分配大小未知**。`nn_cuda.cu:45` 只报 `out of memory`，没有 `bytes`/`free`；
  无法判断是“请求过大”还是“free 接近 0”。
- **1000 集数据集的 split 独立性没有重跑审计**。本复盘只读了文档和 `soft_targets.json`，
  没有重跑 `audit_dataset.py`（约束禁止跑评测/训练；审计也未做）。
- **第二轮结果尚未产生**。a50 在 07:34 左右只到 step 154，a80 未开始；本复盘无法评价
  1000 集蒸馏是否有效。
- **没有跑任何构建/CTest/GPU 任务**。按任务约束，本文件只基于代码、文档、日志和 transcript。
- **没有逐一检查全部 `/tmp` 脚本**（机器上共 623 个 `.sh/.py`）；只核对了本轮 20 个驱动
  和 3 个探针脚本，其他临时文件可能还有未记录的复现依赖。
