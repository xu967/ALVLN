# 第十二阶段任务卡（给执行模型：deepseek 或 Sonnet 5.5 Max）

- 立：2026-11-10，主会话。配套文档：计划 `docs/PLAN_2026-11-10_phase12.md`（为什么做），公共规则 `docs/briefs/2026-11-10-phase12-common.md`（怎么做），
  上一阶段计划 `docs/PLAN_2026-11-10_phase11.md`（方案与预注册的原文）。
- 用法：按 §2 的顺序取卡。一张卡 = 一个分支 = 一份报告。卡里写了「先读」的文件必须先读，不猜接口。
  卡里的行号来自 2026-11-10 的 main，只作定位提示，以 `grep -n` 的实际结果为准。
- 做不到或与卡的描述对不上时：停下，把现象与证据写进报告，不要自行改设计。预注册的阈值、seed、规则参数不许改。

## 1. 开工前核对（每次会话做一遍）

```bash
R=/root/claude-code-backup-20261028/code/uav_rgb; D=/root/rivermind-data/uav_rgb
cd $R && git status --short && git log --oneline -3          # 主工作树必须干净
nvidia-smi -L                                                # 两张 2080 Ti；uav_rgb 只用 GPU-dbc6435f-…（按 UUID 钉）
df -h / $D | tail -2                                         # 根分区与数据盘余量，小于 1.5 GB / 5 GB 时先停下报告
ps -eo pid,pgid,etimes,cmd | grep -E "gz sim|eval_closed_loop|driver.py|uavrgb_train" | grep -v grep   # 谁在占 gz 与 GPU
source $R/env.sh && UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only     # 必须 rc=0
```

建分支与工作树（主仓库只读，不在里面改文件）：

```bash
NAME=<卡号小写，如 w1-1>
git -C $R worktree add -b p12-$NAME /root/claude-code-backup-20261028/code/wt/p12-$NAME main
mkdir -p $D/tmp/phase12-cards/$NAME            # 草稿、日志；报告写 $D/tmp/phase12-cards/reports/$NAME.md
```

## 2. 顺序与依赖

| 波次 | 卡 | 一句话 | 前置 | 占 gz | 占 GPU | 难度 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | W1-0 | 动作派发时刻固定到仿真时间（Gazebo 线每步动力学不再随策略段耗时变化）；最先做 | 无 | 3 个（标定） | 否 | 难，建议 Sonnet 5.5 Max |
| 1 | W1-1 | 走廊判据进两条流水线的默认评测命令 | 无 | 否 | 否 | 易 |
| 1 | W1-2 | 统计表分母改请求集数，重复评测协议写进 EVAL_STANDARD | 无 | 否 | 否 | 易 |
| 1 | W1-3 | 学生概率与帧间隔落盘 | 无 | 否 | 否 | 易 |
| 1 | W1-4 | 逐块存活与每百米碰撞数进 eval.json | 无 | 否 | 否 | 易 |
| 1 | W1-5 | 流水线交付权重改成权重平均加校准 | 无 | 否 | 是（校准推理） | 中 |
| 1 | W1-6 | 3DGS 测试进门禁 | 无 | 否 | 否 | 易 |
| 1 | W1-7 | 监督清理：教师失败集不进训练、教师目标从点改线、无路降余量重试 | 无 | 2 个（验收） | 否 | 中 |
| 1 | W1-8 | 解析几何标签 `--depth-source analytic` | 无 | 1 个（对拍） | 否 | 中 |
| 1 | W1-9 | 航向码输入（打包、闭环、流水线三处同一个函数） | 无 | 否 | 否 | 中 |
| 1 | W1-10 | 满电档参数副本、Gazebo 插件核对、K=1 对拍 | 无 | 3 个 | 是 | 中 |
| 1 | W1-11 | 3DGS 教师的航向容差随转角步推导（消除 hold 1.0 下的左右来回转） | 无 | 否 | 是（3DGS） | 中 |
| 2 | W2-1 | 基线 v2：清理后的数据 + 新配方，3 个训练种子，T0 与 T1 评测 | W1-1、W1-4、W1-5、W1-7 | 6 个 | 是 | 中 |
| 2 | W2-2 | 航向码学生 + 带闸门的 DAgger | W2-1、W1-9 | 6 个 | 是 | 难 |
| 2 | W2-3 | 几何头辅助监督训练 | W2-1、W1-8 | 6 个 | 是 | 中 |
| 2 | W2-4 | 公开小模型同结构重训臂 | 基线包装器已入库 | 3 个 | 是 | 中 |
| 2 | W2-5 | Depth-Anything-V2 Small 米制加规则 | 基线包装器已入库 | 3 个 | 是 | 中 |
| 2 | K1 | 3DGS 上的规则层上限：占据或渲染深度的列距离 + 规则 v1 + 路线方位，室内外各 120 集（零训练） | 规则 v1 已入库 | 否 | 是（3DGS） | 难，建议 Sonnet 5.5 Max |
| 3 | W3-x | 3DGS 程序化障碍、多航点路线、执行器与路书、SAGE-3D 完整试点、数据集 B 放量 | 第 1、2 波的结论 | | | 难；规格见 §5，细化后再开工 |

第 1 波的卡互不依赖，可以并行；W1-3、W1-4、W1-9 都改 `sim/tools/collect.py` 或 `sim/tools/eval_closed_loop.py`，并行时各自只动卡里写的函数，合并由主会话处理。
gz 全机同时不超过 7 个，起实验前先看当前占用。

## 3. 第一波任务卡

### 卡 W1-0：动作派发时刻固定到仿真时间（第十一阶段 F4 的 Gazebo 部分，11-10 晚升为最高优先级）

- 背景（11-10 晚实测，计划 phase12 §2.4；分段打点的证据在 `docs/reviews/2026-11-10-phase12/longcourse.md` 第 6–9 节）：
  `--window-from observation` 下动作窗口从观测帧的时间戳起算，动作的实际生效时长 = hold −（观测帧到派发的延迟）。
  延迟 = 帧龄（相机 10 Hz，固定约 0.114 s）+ 策略段的墙钟耗时折成的仿真时间。策略段里占大头的是每步重跑的 Python A*：
  教师臂里它是动作本身；学生臂与规则臂里它是影子教师标签 `action_expert`，评测不用它，但在派发之前算。
  柱阵 K=1（8 根柱）策略段约 35 ms，每步生效 0.871 s；3 块长课程（25 根柱）约 145 ms，每步生效 0.77–0.79 s，每步位移与转角小 15–20%。
  等新帧 0 ms、解码 7–8 ms、推理 1 ms，都与场景无关；与机器负载、辅助相机也无关。今晚的止血是 A* 栅格可走性缓存（提交 fcbf2fe，K=3 回到 0.872 s）。
  还剩两处没有解决：每集第一步缓存是冷的，K=3 的第一步多出约 0.45 s 的延迟（第一步动作只生效约 0.4 s）；
  场景再大（森林、K=6）时 A* 的堆操作仍会变慢。场景一复杂，动力学就变，跨场景的结论都被它混杂。
- 目标：给定 `--action-latency L`（仿真秒）后，每一步都在「观测戳 + L」这个仿真时刻派发动作，动作生效时长恒为 hold − L，与场景、负载、策略的耗时无关；
  做不到的步要被记下来，不能悄悄变成另一种动力学。
- 先读：`sim/tools/collect.py` 的 `collect_one()`（取观测帧、`exec_ts`、`action_start_ns`、`sil.dispatch(...)`、`hold_end_ns` 的等待循环、`timing` 字段）；
  `sim/tools/cf_sitl_exec.py`（`dispatch` 与 `until_from_window_start`）；`sim/tools/scene_gen.py` 生成世界时物理步长与实时因子写在哪；
  `docs/reviews/2026-11-10/arch.md` §F2 与 `sim-real.md` §D4-3（两域时序与延迟灵敏度）；`docs/reviews/2026-11-10-phase12/longcourse.md` 里时序复核的一节。
- 改动：
  1. `collect.py` 加 `--action-latency L`（默认不给 = 现状，逐位不变）。给了之后：策略出动作后，等位姿话题的仿真戳 ≥ 观测戳 + L 再派发；
     执行器的截止时刻仍是观测戳 + hold。动作就绪时仿真戳已经超过观测戳 + L + 0.02 s 的步记为迟到。
  2. 每步在 actions.jsonl 里记 `dispatch_lag_s`（派发时的仿真戳减观测戳；不给 `--action-latency` 时也记，只加键）；
     manifest 记 `action_latency_s`、`late_dispatch_steps`。`eval_closed_loop.py` 透传参数，`per_episode` 记迟到步数，
     汇总记各集 `dispatch_lag_s` 的中位数与最大值；迟到步数大于 0 的集在 eval.json 里单列，不计入成功（记为基础设施失败并写原因）。
  3. `policy == "student"` 且 β = 0 时（闭环评测、纯学生采集、经 `--infer-bin` 接入的外部基线都走这个分支），影子教师标签
     （`collect_one()` 里 `label, tie_margin = plan_action_global_tiebreak(...)` 那一句）挪到 `sil.dispatch` 之后算，落在 hold 窗口里，与 PNG 落盘同一个位置。
     `action_expert` 的值不变（同一个 `pose_before`、`yaw_before`、同一个函数），只是算的时刻变了。β > 0（DAgger 混合执行）保持现状：执行哪个动作要先看标签。
     教师策略不动；rule-oracle 分支本来就不算影子标签。这一条去掉学生臂策略段的大头。只在给了 `--action-latency` 时生效：
     不给时派发时刻会因此提前约 20 ms（每步生效长约 2%），旧数字就复现不了，所以默认路径保持现状。
  4. 每集第一步之前预热（同样只在给了 `--action-latency` 时做）：对整个栅格填一遍 `teacher._free_cells`，学生与外部基线先做一次空推理。
     预热的墙钟耗时不进任何动作窗口。
  5. 策略太慢、赶不上 L 时的处理：`scene_gen.py` 加 `--rtf R`（写进世界的实时因子，默认不给 = 模板原值），评测入口透传。
     慢策略（外部基线里跑 PyTorch 的那几个、大场景上的教师）用较小的实时因子，让策略耗时折算成的仿真时间小于 L − 帧龄。
  6. 口径：给了 `--action-latency` 的数据在 manifest 的协议键里要能与旧数据区分（看 `PROTOCOL_KEYS` 与 preprocess 的 `STRICT_KEYS` 怎么管 hold 与 window_from，照样加一项），
     默认不混包。
- 测试（先红后绿）：用现有 `test_collect.py` 的替身话题构造三种情形：动作在 L 之前就绪（派发戳 = 观测戳 + L）；动作在 L 之后就绪（记迟到）；不给参数（行与 manifest 除新增的 `dispatch_lag_s` 外逐位不变）。
- 标定与验收（守护进程，3 个 gz，预注册写在入口脚本头）：
  1. L 取 0.16 s：帧龄 0.114 s 加教师在 K=3 上的策略段（缓存后沿航线均值 16–26 ms 墙钟，折仿真约 13–21 ms）再留余量。
     柱阵现状的延迟中位数是 0.129–0.137 s，所以新协议下每步生效 0.84 s，比现在短约 3.5%，旧的 K=1 参照要在新协议下重跑一遍（见第 2 条）。
     教师策略，柱阵 K=1 与长课程 K=3 各 10 个 train 分区 seed：每步动作生效时长、稳态前进位移、转向角度三项，K=3 与 K=1 的相对差都不超过 2%；
     迟到步占比为 0，含每集第一步。做不到时先查预热，再试 L = 0.20，再考虑 `--rtf`；试过的每个取值的迟到步占比都写进报告，最后用哪个由主会话定。
  2. 学生 calswa 种子 1 加 `--heading-guard 2`，柱阵 20 个冻结 seed，带 `--action-latency 0.16`：走廊内成功数与 11-10 的两次同配置评测
     （下午 17/20，晚上对照见计划 §1.3）配对比较。预测：差别在同权重同 seed 重评的波动之内（逐集一致率不低于 70%）。这一组同时是新协议下的 K=1 参照。
  3. 森林 `forest_e1` 三档各量一次延迟分布，给出各自需要的 `--rtf`。
  4. 长课程第 1 块的对照（回答 11-10 晚留下的问题，计划 §1.3「长柱阵课程」）：学生 calswa 三个训练种子加 `--heading-guard 2`，带 `--action-latency`，
     柱阵 K=1 与长课程 K=3 各 20 个冻结 seed × 3 个种子。比较量是 K=3 的第 1 块通过率对 K=1 的成功率（同权重同 seed 配对，精确 McNemar）。
     11-10 晚的旧协议下是 39/60 对 50/60，但 K=3 的第一步因为冷缓存只生效了约一半时长，分不清是时序还是画面。
     预注册：新协议下两者相差不超过 0.08，则那次下降来自第一步的时序；K=3 仍低 0.12 以上，则来自后面几块的柱子进了画面（学生没见过远处的柱子）。
- 交付：`docs/experiments/<日期>-action-latency.md`；`docs/EVAL_STANDARD.md` 加一节：新实验一律带 `--action-latency`，报数时附 `dispatch_lag_s` 的中位数与迟到步数。
- 提交：`feat(collect): --action-latency（派发时刻固定到观测戳 + L，迟到记账）`、`feat(scene): --rtf`、实验入口。

### 卡 W1-1：走廊判据进流水线默认（第十一阶段 A1）

- 目标：`tools/e1_pipeline.py` 与 `tools/dagger_gz_pipeline.py` 起的每一条柱阵评测命令默认带 `--corridor-half-width 3.0`。
- 先读：`tools/e1_pipeline.py` 的 `CFG` 与 `evaluate()`；`tools/dagger_gz_pipeline.py` 的 `CFG` 与 `eval_cmd()`；
  `tools/test_e1_pipeline.py`、`tools/test_dagger_gz_pipeline.py` 里 dry-run 断言的写法；`sim/tools/eval_closed_loop.py` 里 `--corridor-half-width` 的帮助文字。
- 改动：
  1. 两个 `CFG` 各加一行 `corridor_half_width=float(os.environ.get("E1_CORRIDOR_HALF_WIDTH", "3.0")),`，行尾注释写：
     柱阵评测的横向边界，等于 `scene_gen.Y_RANGE`；11-10 发现不加它时 20/20 里有 8 集是绕出柱阵；设 0 回到旧口径。
  2. `evaluate()` 与 `eval_cmd()` 在该值大于 0 时给命令追加 `["--corridor-half-width", f"{值:g}"]`。
  3. 不改 `eval_closed_loop.py` 的默认值（保持 None）。不改 `ui/`。
- 测试（先红后绿）：两个测试文件各加一条：dry-run 输出的评测命令包含 `--corridor-half-width 3`；环境变量设 0 时不包含。
- 验收：`python tools/test_e1_pipeline.py`、`python tools/test_dagger_gz_pipeline.py`、`check.sh --python-only` 全过。
- 提交：`feat(pipeline): 柱阵评测默认带 --corridor-half-width 3.0（E1_CORRIDOR_HALF_WIDTH 可覆盖）`。

### 卡 W1-2：主表分母与重复评测协议（第十一阶段 A2）

- 目标：统计表不再把基础设施失败的集从分母里拿掉；评测标准文档写明主比较的规模。
- 先读：`tools/stats_report.py` 的文件头、`load_eval()`、`summarize_cell()`、`to_markdown()`；`tools/test_stats_report.py`；`docs/EVAL_STANDARD.md` §5。
- 改动：
  1. 方向为「越高越好」的二值指标（SR、task_success、reach_ever、stop_success）：分母改为 `episodes_requested`，失败集记 0。
     方向为「越低越好」的二值指标（CR、stuck、false_forward）：分母仍是完成集，表注写明。表里新增一列「失败集数」。
  2. `docs/EVAL_STANDARD.md` §5 追加一小节：主比较 = 20 个冻结几何 × 每方法 3 个训练种子（60 集），T1 起再报逐块存活；
     起跑前写下不超过 3 个配对比较；单次 n=20 的比较不下结论（同权重同 seed 重评的逐集不一致率约 24%）。
- 测试：`tools/test_stats_report.py` 加一例：一个 eval 完成 19 集、失败 1 集，SR 的分母是 20，CR 的分母是 19。
- 验收：测试与 `check.sh --python-only` 全过；用 `$D/exp-swa-calib-20261110/calswa-s0/eval.json` 跑一次 `stats_report.py`，输出贴进报告。
- 提交：`fix(stats): 成功类指标分母改请求集数，失败集单列；EVAL_STANDARD 写入主比较协议`。

### 卡 W1-3：学生概率与帧间隔落盘（第十一阶段 A6）

- 目标：闭环评测的每一步能事后看到学生的四类概率；每集能看到相邻两帧的最大时间间隔。
- 先读：`sim/tools/collect.py` 的 `collect_one()` 里 `policy == "student"` 分支（`student.act()` 返回 `(action, infer_ms, probs)`，现在概率被丢掉）
  与它末尾组装 `row` 的地方（搜 `"infer_ms": infer_ms`）；`sim/tools/eval_closed_loop.py` 的 `summarize_episode()`；`sim/tools/test_collect.py`、`sim/tools/test_eval_summary.py`。
- 改动：
  1. `collect_one()`：学生分支把概率保留下来，`row` 里只在学生策略下加键 `"probs": [四个数，保留 4 位小数]`。其它策略的行逐键不变。
  2. `summarize_episode()`：由 actions 的 `rgb_ts`（纳秒）算相邻两步之差的最大值，写 `max_frame_gap_s`；大于 `1.2 × hold` 时再写 `frame_gap_flag: true`。
     函数拿不到 hold 时用行里的 `sim_dt`。
- 测试：`test_collect.py` 加：学生策略的行有 4 个概率且和为 1（±1e-3），教师策略的行没有 `probs` 键。
  `test_eval_summary.py` 加：合成一条相邻帧间隔 2.4 s 的轨迹，`frame_gap_flag` 为真；间隔 1.0 s 的为假或缺省。
- 验收：相关测试与 `check.sh --python-only` 全过。
- 提交：`feat(eval): actions.jsonl 记学生概率；per_episode 记 max_frame_gap_s`。

### 卡 W1-4：逐块存活与每百米碰撞数进 eval.json（批次 J2）

- 目标：长课程 `--task course_base` 的评测结果里直接有逐块存活，不靠实验目录里的汇总脚本。
- 先读：`sim/tools/scene_course.py`（meta 的 `course.block_goal_x`）；`experiments/2026-11-10-long-course/summarize.py` 里算逐块存活的函数；
  `sim/tools/eval_closed_loop.py` 的 `summarize_episode()`、`run_episode()`、`write_report()`。
- 改动：
  1. 把「第 k 块通过 = 首次到达 `x >= block_goal_x[k]` 之前无碰撞且 `|y|` 不超过走廊半宽」做成 `eval_closed_loop.py` 里的一个纯函数
     `blocks_passed(oracle, actions, block_goal_x, corridor_half_width)`，返回通过的块数。`summarize.py` 改为 import 它，删掉自己的那份。
  2. `per_episode` 在场景 meta 带 `course` 时加 `blocks_passed` 与 `blocks_total`；汇总加 `block_survival`（长度 K 的列表，分母是请求集数）。
  3. 所有任务的汇总都加 `collisions_per_100m` = 100 × 碰撞集数 ÷ 各集飞行路程之和（路程算到首次碰撞、越界或过线为止，单位 m）。
  4. base 任务（meta 没有 `course`）不出现 `blocks_*` 键，旧 eval.json 的其它键与值不变。
- 测试：新增 `sim/tools/test_eval_blocks.py`：合成 3 块课程的轨迹，分别在第 1 块撞、第 2 块越界、全程通过，断言 `blocks_passed` 为 0、1、3；
  base 任务的汇总没有 `block_survival` 键。
- 验收：对 `$D/exp-long-course-20261110/` 里已有的集离线重算，逐块存活与该实验 `summary.md` 的数一致，贴进报告。
- 提交：`feat(eval): course_base 逐块存活 block_survival 与 collisions_per_100m 进 eval.json`。

### 卡 W1-5：流水线交付权重改成权重平均加校准（第十一阶段 B4 的接入部分）

- 背景：11-10 闭环证实「末 10 轮权重平均 + FORWARD 先验校准」比 `ckpt_best` 好（60 集 27 → 39，三个训练种子的极差 6 → 3）；
  只平均不校准反而更差（20/60）。工具是 `tools/average_ckpt.py`。
- 先读：`tools/average_ckpt.py` 的文件头与配置区；`tools/e1_pipeline.py` 的 `train()`、`evaluate()`；`tools/dagger_gz_pipeline.py` 里训练与评测取权重的地方
  （搜 `ckpt_best.bin`）；两条流水线各自的测试。
- 改动：
  1. 两条流水线的训练命令加 `--save-every-epoch`（平均要用末 10 轮的 `ckpt_epNN.bin`）。
  2. 训练步成功后调用 `average_ckpt.py --run <run> --data <ds>/dataset.uavr --out <run>/ckpt_calswa.bin --last-k 10 --infer-bin <BUILD>/uavrgb_infer`，
     带 `CUDA_VISIBLE_DEVICES=<uav_rgb 的那张卡>`。失败就让流水线停下（不静默退回 `ckpt_best`）。
  3. `CFG` 加 `deliver="calswa"`（环境变量可切回 `"best"`），评测与后续步骤按它取 `ckpt_calswa.bin` 或 `ckpt_best.bin`。
     训练轮数小于 10 时报错并提示改 `deliver`。
- 测试：两条流水线的 dry-run 测试断言：训练命令含 `--save-every-epoch`；训练步之后有 `average_ckpt.py` 这条命令；评测命令的 `--ckpt` 指向 `ckpt_calswa.bin`；
  `deliver=best` 时三条都回到旧行为。
- 验收：测试与 `check.sh --python-only` 全过；另用 `$D/exp-student-select-20261109/e1-noise/seed0/run` 实际跑一次 `average_ckpt.py`，
  产物 sha256 等于 `$D/exp-swa-calib-20261110/inputs.txt` 里 `cal_swa_s0_k10.bin` 的值（工具入库时已核对过，这里是回归）。
- 提交：`feat(pipeline): 交付权重默认 calswa（末 10 轮平均 + 校准），训练加 --save-every-epoch`。

### 卡 W1-6：3DGS 测试进门禁（第十一阶段 A5 的前半）

- 先读：`tools/check.sh` 末尾收集 `tools/test_*.py`、`sim/tools/test_*.py` 的两个循环；`$D/tmp/phase12-20261110/bearing3dgs/run_3dgs_tests.sh`（3DGS 测试怎么跑、用哪个解释器）。
- 改动：`check.sh` 在两个循环之后加一段：`$DATA_ROOT/uav_rgb/3dgs-venv/bin/python` 存在时逐个跑 `sim/3dgs/test_*.py`（纯 CPU，mock 渲染）；
  不存在时把「3DGS 测试」记进 SKIP 列表（脚本末行 PASS 会列出未检查的段）。`--python-only` 模式也要跑这一段。
- 验收：`check.sh --python-only` rc=0，输出里看得到 7 个 3DGS 测试文件各自通过；把解释器路径临时改错时，这一段显示为 SKIP 而不是通过。
- 提交：`chore(check): sim/3dgs/test_*.py 进 check.sh（3dgs-venv 不在时 SKIP 并列出）`。

### 卡 W1-7：监督清理（第十一阶段 B1、B2、B3）

- 背景〔`docs/reviews/2026-11-10/data-algo.md` §A1、§A2、§A4〕：训练集里有教师自己没过线的示范；教师的目标是点 `(最远柱 x + 2.0, 0)`，
  过了柱阵之后为了回到 y=0 还在转向，这些转向标签占出口区的 17.7%，而学生看不到目标点；A* 偶尔无路，教师退回局部规则并左右摆。
- 先读：`tools/preprocess.py` 的 `load_episode()`（`exclude_collision_episodes` 那个开关的写法）；`sim/tools/teacher.py` 的 `_global_path()`、`_costmap_goal()`、
  `astar_path()`、`TEACHER_VERSION` 与文件末尾的自检；`sim/tools/audit_dataset.py` 的 label_recompute；`sim/tools/test_teacher_*.py`。
- 改动：
  1. B1：`preprocess.py` 加开关 `--exclude-failed-teacher`（默认关）：策略是教师的集，如果 oracle 的最大 x 小于「最远柱 x + 1.0」，整集跳过，
     并像 `--exclude-collision-episodes` 那样打印跳过了几集。带 waypoint 的任务不适用，遇到就不跳。
  2. B2：`_global_path()` 与 `_costmap_goal()` 的 base 目标，在机体 x 已超过最远柱后表面（最远柱 x + 柱半宽）时，目标 y 取当前 y（目标从点变成线）。
  3. B3：`_global_path()` 里 A* 无路时，按余量 0.30、0.20 m 各重试一次，仍无路才返回 None。
  4. `TEACHER_VERSION` 升一版，注释写清 B2、B3。`audit_dataset.py` 的标签复算按版本分派：旧版本的数据仍按旧规则复算。
- 测试（先红后绿）：`teacher.py` 自检或 `test_teacher_*.py` 加：机体在 `(最远柱 x + 0.5, 1.2)`、机头朝 +x 时，新教师出 FORWARD（旧教师出 YAW）；
  构造一个标准余量下无路、0.30 m 余量下有路的柱阵，新教师给出路径。`tools/test_preprocess_split.py` 或新文件加 B1 的正反例。
- 验收：
  1. `check.sh --python-only` 全过。
  2. 闭环：教师在 20 个冻结 seed 上仍是 20/20（`eval_closed_loop.py --policy teacher --task base --split test --corridor-half-width 3.0`，其余参数照
     `experiments/2026-11-10-heading-guard/run.sh`），2 条车道，守护进程。出口区（x 大于最远柱 x）的 YAW 步占比从约 17.7% 降到 5% 以下，数字贴进报告。
  3. 对 `$D/e1-pipeline-20261107-0417/collection` 打包时加 `--exclude-failed-teacher`，日志里跳过的集数贴进报告（预期约 26 集）。
- 提交：`fix(teacher): 过柱阵后目标取线不取点；A* 无路降余量重试（TEACHER_VERSION 升版）`、`feat(preprocess): --exclude-failed-teacher`。

### 卡 W1-8：解析几何标签（第十一阶段 D1）

- 目标：已有的 Gazebo 数据没存深度图，但每帧有位姿、每集有柱子真值。打包时由它们解析算出每个网格单元的最近深度，写进现有的 UAVRAUX1 侧车，
  这样几何头可以直接在现有数据上训练，不用重采。
- 先读：`tools/preprocess.py` 的 `aux_payload()`、`cell_min_depth()`、`dilate_by_envelope()`、`depth_bands()`、`episode_focal_px()` 与 `DEPTH_GRID_DOWN`；
  `sim/tools/rule_policy.py` 的 `column_distances()`（水平面内对方柱与圆柱求交的写法与角度约定）；`tools/DATASET_FORMAT.md` §6；`sim/dynamics/e1_mini.json` 的 `camera_mount`。
- 改动：
  1. `preprocess.py` 加参数 `--depth-source {file,analytic}`，默认 `file`（现状，逐位不变）。
  2. `analytic`：对每个样本，用该步动作前的位姿（`pose_before` 与 `yaw_before`，单位以 `tools/DATASET_FORMAT.md` 为准）、相机安装位（前 0.045 m，高 0.015 m）、
     相机内参（`episode_focal_px()`，320×240）与 manifest 的 `pillars`，对网格的每个单元取 3×3 条射线：
     方位角 φ 的射线在水平面内到最近柱面的距离记 r（复用 `rule_policy` 的求交），该射线的深度（沿光轴）= r·cos φ；
     仰角为负的射线先算打到地面（z=0）的水平距离，取两者较近的；仰角为正的射线在距离 r 处的高度超过柱高 3.0 m 时算没打到（记最大深度）。
     单元取 9 条射线里最小的深度，得到与 `cell_min_depth()` 同形状的数组，之后原样走 `dilate_by_envelope()` 与 `depth_bands()`。
  3. 机体的滚转与俯仰不计（前飞倾角不超过 10°），写 `shortcut` 注释。森林场景的圆柱一并支持。带 waypoint 的任务与 3DGS 的集不走 analytic（报错）。
  4. 数据集 meta 记 `depth_source`。
- 测试：新增 `tools/test_preprocess_analytic_depth.py`：正前方 1.0 m 一根 0.4 m 方柱，中间列的地平线行落在 0.8–1.2 m 所在的分带，最左与最右列落在最远分带；
  相机高度 2.0 m 时最下一行落在地面对应的分带；`--depth-source file` 的输出与改动前逐字节相同（用现有夹具）。
- 验收：
  1. 测试与 `check.sh --python-only` 全过。
  2. 对拍：用 `collect.py --save-depth` 采 1 集教师轨迹（1 个 gz），同一集分别用 `file` 与 `analytic` 打包，比较柱子所在单元的分带一致率，不低于 95%；
     不一致的单元按「柱边缘 / 地面 / 其它」分类，贴进报告。
  3. 对 `$D/e1-pipeline-20261107-0417/collection` 全量打包一次，报告耗时与各分带占比。
- 提交：`feat(preprocess): --depth-source analytic（由位姿与柱子真值解析算深度分带，现有数据不用重采）`。

### 卡 W1-9：航向码输入（第十一阶段 C2）

- 目标：把「已经偏了多少」作为 4 值条件码送进学生，走现有的 prev-action 通路（训练器的 `--prev-action`、推理协议里的 1 字节），不改 C++。
- 码的定义：n = 该步之前已执行的 YAW_LEFT 数减 YAW_RIGHT 数，再减去航向设定值 H（柱阵与长课程任务 H 恒为 0）。
  码 0：|n| ≤ 1；码 2：n ≥ 2（偏左了）；码 3：n ≤ −2（偏右了）；码 1 不用。首帧 n=0。
- 先读：`tools/preprocess.py` 第 230–290 行附近（`prev_action` 怎么从上一步的 `action_executed` 取）；`sim/tools/collect.py` 里 `student.act(rgb_msg, prev_action=prev_executed)`
  与主循环的 `net_yaw`；`sim/tools/student.py` 的 `act()`；`tools/dagger_gz_pipeline.py` 的采集、打包、训练三处命令；`docs/reviews/2026-11-10/data-algo.md` §D1。
- 改动：
  1. 新文件 `sim/tools/nav_cond.py`：`heading_code(n)` 一个纯函数加自检。打包与闭环都 import 它，不许各写一份。
  2. `preprocess.py` 加 `--cond {prev-action,heading-code}`，默认 `prev-action`（现状）。`heading-code` 时侧车里 prev-action 那一列写航向码
     （按集从头累计已执行动作），数据集 meta 记 `cond: "heading_code.v1"`。
  3. `collect.py` 加 `--student-cond {prev-action,heading-code}`，默认现状；`heading-code` 时传给 `student.act()` 的是 `heading_code(net_yaw)`。manifest 记同名键。
     `eval_closed_loop.py` 加同名参数透传，eval.json 记录。
  4. `dagger_gz_pipeline.py` 的 `CFG` 加 `student_cond`，采集、打包、评测三处透传。
  5. 一致性闸门：`eval_closed_loop.py` 起跑前读 ckpt 同目录的训练配置或数据集 meta（看现有的 protocol 校验是怎么做的），
     条件类型与 `--student-cond` 不一致时报错退出。拿不到训练时的条件类型就打印警告并写进 eval.json，不要猜。
- 测试：`nav_cond` 自检：动作序列 F L L F R 的码序列是 0 0 0 2 2（码取执行前的 n）。`test_collect.py`：`heading-code` 时传给学生的值与 `nav_cond` 一致；
  默认时与改动前逐位相同。`preprocess` 测试：同一集两种 `--cond` 打包，标签与图像相同，只有条件列不同。
- 验收：测试与 `check.sh --python-only` 全过。训练与闭环在卡 W2-2 做，本卡不训练。
- 提交：`feat(cond): 航向码条件输入（nav_cond.heading_code；preprocess --cond、collect --student-cond、流水线透传）`。

### 卡 W1-10：满电档（批次 J5）

- 背景〔`docs/reviews/2026-11-10-phase12/complexity.md` §5〕：SITL 每集从 50% 电量起飞，约 100 s 后推力余量耗尽，一集最多约 100 步；
  长任务需要从满电起飞。起始电量只能通过 params JSON 的副本覆盖，动力学指纹会从 `1bc8fedf` 变成 `06e5f017`。
- 先读：`sim/cf_sitl/README.md` 的电池一节；`sim/tools/cf_sitl_exec.py` 的 `params_values_sha256()`；`sim/tools/collect.py` 的 `--cf-sitl-params`；
  `sim/cf_sitl/gz/` 插件怎么读 `<params>`；`sim/tools/scene_gen.py` 生成 E1 世界时 `<params>` 写的是什么路径；`sim/3dgs/collect_3dgs.py` 的 `--dynamics-params`。
- 改动：
  1. 新文件 `tools/make_params_variant.py`：读 `sim/dynamics/e1_mini.json`，按 `--set sim.battery_soc_init=1.0` 这类点路径改值，
     写到 `--out`（放数据盘 `$D/params/`），并打印新指纹。带自检。
  2. 查清 Gazebo 线：`collect.py --cf-sitl-params` 给了副本之后，gz 插件实际读的是不是副本。不是的话做最小改动让它生效
     （例如生成世界时把 `<params>` 改写成副本路径），并用测试锁住「不给参数时世界文件逐字节不变」。
  3. `eval_closed_loop.py` 若没有透传该参数，加上透传；eval.json 记 `dynamics_profile`。
- 验收（实验，守护进程，预注册写在 `experiments/<日期>-soc100-parity/run.sh` 头部）：
  1. Gazebo，柱阵 20 个冻结 seed，calswa 种子 1 加 `--heading-guard 2`，满电档跑一遍（3 个 gz）。对照是 `$D/exp-heading-guard-20261110/guard2-calswa-s1/`。
  2. 报告：两档的每步前进位移中位数、每步转角中位数（预期差不超过 5%）、起飞到就绪的时间、悬停油门（如果遥测里有）、走廊内成功数（配对 McNemar）。
  3. 预测：成功数与 50% 档无可检出的差别；每步位移与转角差在 5% 以内。差超过 10% 则两档的数据与结论不能互用，报告里写明并停下。
- 提交：`feat(tools): make_params_variant.py（params 副本与指纹）`、`fix(collect): --cf-sitl-params 对 Gazebo 插件生效`（如果需要）、实验入口。

### 卡 W1-11：3DGS 教师的航向容差随转角步推导（第十一阶段 E1）

- 背景（11-10 晚实测，`docs/reviews/2026-11-10-phase12/sage3d.md` §4.5–§4.6）：`sim/3dgs/planner.py` 的 `OccAstarTeacher` 在 `|航向误差| > YAW_TOL_DEG = 12°` 时转向。
  hold 1.0 下一个转角步实测 27–35°，死区宽 24° 小于一个转角步，误差落在 12°–16° 时转一步就过冲到另一侧，形成左右来回转的极限环。
  室内 5 个场景教师到点成功 76/120 = 0.63（超时 29 集里 25 集含交替段）；只把容差改成 18° 的诊断实验是 101/120 = 0.84。室外 4 个场景教师 0.83 的失败（超时率 0.16）是同一个原因。
- 先读：`sim/3dgs/planner.py` 的 `OccAstarTeacher`（`__init__` 的 `tol_deg`、`__call__` 里两处转向判定）；`sim/3dgs/test_teacher_occ.py` 是否在 main 上（在途工作树
  `$D/tmp/wt-teacher-3dgs` 里有一版没做完的 `astar_occ_tol`，只作参考，不合并它）；`experiments/2026-11-10-bearing-guard-3dgs/driver.py` 的 `teacher_tol_deg` 配置键；
  `sim/3dgs/collect_3dgs.py` 与 `tools/demo_3dgs_routes.py` 里构造教师的地方。
- 改动：
  1. `OccAstarTeacher` 的容差默认值改成由 hold 推导：`tol = max(12°, 0.55 × 该 hold 的名义转角步)`。名义转角步取 `executor.yaw_rate_rps × hold` 换成度
     （0.6 rad/s：hold 0.5 约 17°，hold 1.0 约 34°），所以 hold 0.5 仍是 12°，hold 1.0 是约 19°。显式传 `tol_deg` 时照传入值。
  2. 构造教师的调用方把 hold 传进去；`teacher_version`（或 manifest 里记录教师参数的字段）要能区分新旧容差，旧数据的标签复算按记录的值走。
  3. 「前向被堵时按误差符号转」那一条是否也参与极限环没有拆开。先只改容差；改完仍有交替段的集单独列出来。
- 测试：桩规划器加一个每步转 33° 的运动学模拟，目标在正前方偏 14° 处：旧容差下出现 6 步以上的左右交替，新容差下不出现；hold 0.5 的行为与改动前逐步相同。
- 验收（守护进程，3DGS，1 个驱动进程）：室外 4 个场景 × 6 路线 × 5 集（与 `exp-bearing-guard-3dgs-20261110` 同 seed）教师到点成功不低于 0.92（现在 0.83）；
  室内 5 个场景 40 条路线 × 3 集（与 `exp-sage3d-pilot-20261110` 同 seed）不低于 0.80；两处都报超时集里含交替段的集数。
- 提交：`fix(3dgs): OccAstarTeacher 航向容差随 hold 的转角步推导（hold 1.0 下不再左右来回转）`。

## 4. 第二波任务卡

### 卡 W2-1：基线 v2（第十一阶段 B6）

- 数据：`$D/e1-pipeline-20261107-0417-eval3/` 的截断集 300 集与 `$D/exp-dagger-gz-20261109/full-1109/more_demos/` 的 340 集（先核对两处目录与 manifest 的
  `dynamics_profile`；不一致时按第十一阶段 §6 的说明处理并写进报告）。打包：`--exclude-failed-teacher --drop-collision`。标签按 W1-7 的新教师离线重标
  （看 `tools/relabel_costmap.py` 的做法，A* 教师没有现成的重标脚本就写一个 `tools/relabel_teacher.py`，带测试）。
- 训练：3 个训练种子；`--augment photo --save-every-epoch --class-weights 0,1,2,2 --epochs 30`，其余同现有流水线。交付权重 calswa（W1-5）。
- 评测（守护进程，6 个 gz）：T0 = 柱阵 20 个冻结 seed × 3 个训练种子，无护栏与 `--heading-guard 2` 各一遍；T1 = `--task course_base --blocks 3`，同样两遍。
- 预注册（原文见第十一阶段 §6 B6）：参考臂是 11-10 的 39/60（无护栏）与 50/60（加护栏）。预测：基线 v2 无护栏不低于 42/60，加护栏不低于 50/60，
  三个种子的极差不超过 3，撞柱不多于参考臂。两个口径都不高于参考臂，则清理与新配方只保留稳定性结论。T1 只报告，不设门槛。
- 交付：`docs/experiments/<日期>-baseline-v2.md`（命令、环境、原始输出、配对检验）；权重与数据留数据盘。

### 卡 W2-2：航向码学生与带闸门的 DAgger（第十一阶段 C3、C4）

- 前置：W2-1 的数据与配方，W1-9。
- 第零步：`tools/average_ckpt.py` 现在对带条件输入的模型（ckpt 的 `prev_dim > 0`）只能平均、不能校准（会报「暂不支持」）。
  先扩展它：校准推理时从数据集侧车（UAVRAUX1）读每帧的条件列，按 `sim/tools/student.py` 的协议随帧多发 1 字节；加测试；
  用一个 `--prev-action` 训出的小模型核对「工具算的预测」与 `uavrgb_train --eval-only` 的预测逐帧一致。
- 第一步：同一份数据用 `--cond heading-code` 重新打包，训练加 `--prev-action`，3 个种子，评测加 `--student-cond heading-code`，不带护栏。
- 第二步（DAgger v2）：`dagger_gz_pipeline.py` 的学生自采加 `--stop-on-collision`；出阵（|y| 超过 3 m）即停；采集步数与评测相同；
  验证集固定为第 0 轮的 val 加第一轮学生状态的 val；β 取 0.5、0.3、0；每轮之后比较在线到线率，比上一轮低 15 个点就停。
  对照臂：同帧数只加教师示范。
- 预注册（原文见第十一阶段 §7 C4）：在线到线率逐轮不降；最终臂不带护栏时走廊内成功不低于 48/60、越界不超过 3/60。
  不带护栏时不高于 42/60，则学出来的航向条件没有接近手写规则：保留护栏，航向码输入停掉。
- 另报：T1（K=3）的逐块存活；反事实探针（把码 2 与码 3 对调，学生的转向方向是否跟着变）。

### 卡 W2-3：几何头辅助监督（第十一阶段 D2）

- 前置：W2-1，W1-8。
- 做法：基线 v2 的数据用 `--depth-source analytic` 打包，训练加 `--depth-aux --aux-lambda 0.5`，其余同 W2-1，3 个种子。
- 预注册：加护栏口径下撞柱不多于 6/60（参考臂 10/60），走廊内成功不低于 50/60。同时报告几何头在 val 上的分带准确率，
  以及它与闭环成功率在各 checkpoint 之间的秩相关（与 val macro-F1 的秩相关作对比）。
- 推理端输出列距离（第十一阶段 D4）涉及 C++（`src/infer.cpp`），不在本卡里做。

### 卡 W2-4：公开小模型同结构重训臂

- 先读：`docs/reviews/2026-11-10-phase12/baselines.md` §5.1 的 B1 与 §5.3；`sim/tools/baselines/README.md`。
- 做法：PULP-Dronet v3 的两个尺寸（320,928 与 2,784 参数），网络结构与损失沿用官方（转向回归加碰撞二分类），在我们的训练数据（与 W2-1 相同的帧）上训练。
  标签：转向 = 教师动作为 YAW_LEFT 取 +0.278、YAW_RIGHT 取 −0.278、其余 0；碰撞 = 解析几何标签里中间条带最近距离小于 1.5 m 记 1（用 W1-8 的函数）。
  训练脚本放 `sim/tools/baselines/train_pulp.py`，用 `$D/teacher-venv` 的 torch；检查点放 `$D/baselines/pulp-dronet/retrain/`。
  超参与早停只看 val，不看 20 个冻结 seed。
- 评测：`BASELINE_WEIGHTS=<检查点>`，柱阵 20 个冻结 seed × 3 个训练种子，无护栏与加护栏各一遍，映射参数用零样本实验里定下的值。
- 预注册的主比较：我们的学生（基线 v2，calswa）对 PULP v3 重训臂（32 万参数），60 对配对 McNemar。不预设方向；两个结果都入档。

### 卡 W2-5：Depth-Anything-V2 Small 米制加规则

- 先读：`baselines.md` §5.1 的 B3（下载命令、目录、适配器现状）。
- 做法：下载两个米制权重（各 99.2 MB，先 HEAD 核对体积）与官方代码到 `$D/baselines/depth-anything-v2/`；在 `sim/tools/baselines/baseline_infer.py` 里补全 `da2_rule`：
  估计深度图 → 按 `rule_policy` 的列划分取地平线带每列最小深度 → `rule_policy.rule_v0`（与规则层上限实验同一条规则、同一组参数）。
  先在 100 张闭环帧上报告估计深度与解析深度的相对误差（中位数、p90）与单步时延。
- 评测：室内权重、室外权重各一臂，柱阵 20 个冻结 seed × 3 次重复；对照是规则层上限实验的 `rule60`（真值列距离加同一规则）。
- 预注册的主比较：我们的学生对「室内权重加规则」。预测：灰色无纹理柱体上米制尺度偏差大，估计深度臂低于真值臂 10 集以上。

### 卡 K1：3DGS 上的规则层上限（零训练；11-10 晚从第三波提前）

- 回答的问题：在真实外观的室内场景里，「各列可通行距离 + 固定规则 + 路线方向」这套分解能到多高。11-10 晚只量了另外两个角：
  路线知识完整但不看图（盲飞加路线护栏）室内 0.60、10 m 以上 0.30、碰撞率 0.43；教师（特权 A*）0.63，容差修正后 0.84。
  规则层在 Gazebo 柱阵上是 57/60、撞柱 0（规则 v1）。这张卡量的是第三个角：感知完美时规则层在室内外 3DGS 场景上的到点成功。
- 先读：`sim/tools/rule_policy.py`（`RULE_V1`、`rule_v1_step`、`obstacle_points`、`corridor_clear`；Gazebo 侧的接线在 `collect.py` 的 `rule-oracle` 分支）；
  `sim/3dgs/guard.py`（`OracleRouteGuide`、`bearing_error_deg`，角度约定写在文件头）；`sim/3dgs/collect_3dgs.py` 的 `run_episode`（策略签名
  `policy(rgb, depth, state) -> (action, infer_ms)`，`state` 是动力学对象，带 x / y / yaw 弧度）；`sim/3dgs/collision.py` 的 `Occupancy`（`collides`、`sweep`、`clearance`）；
  `sim/3dgs/renderer.py` 文件头（深度是 expected depth，米，0 = 无效）；`experiments/2026-11-10-bearing-guard-3dgs/{driver.py,run.sh,summarize.py}`
  与 `experiments/2026-11-10-sage3d-pilot/{run.sh,select_routes.py}`（两个实验共用这一个驱动，臂 = 底座 + 可选护栏）；
  报告 `docs/reviews/2026-11-10-phase12/{bearing3dgs,sage3d,rulev1}.md`。
- 策略（新文件 `sim/3dgs/rule_oracle.py`，类名带 Oracle）：每步
  1. 取 20 列（60° 视场，每列 3°）的可通行距离 d，上限 4.0 m。两个来源，各做一个臂：
     `occ` = 在碰撞判定用的那个 `Occupancy` 上按飞行高度沿各列射线步进（步长不大于体素边长的一半），第一个 `collides` 为真的距离，
     这是与碰撞判定完全一致的真值，对应 Gazebo 的解析列距离；
     `depth` = 渲染深度图里，把像素反投影到机体系，只留高度落在机体包络上下各加 0.10 m 之内的点，按方位角分到 20 列，每列取最近的水平距离。
     深度是 z 深度还是沿射线的距离，用一个已知几何的位姿对拍后再定，不猜。
  2. 航向计数换成路线方位：`err = bearing_error_deg(x, y, yaw, guide.target(x, y))`（正 = 目标在左），`n = round(−err / yaw_step_deg)`，
     与 Gazebo 的 n（已执行左转数减右转数，正 = 机头偏左）同号。另做一个不给路线的臂：目标点取 waypoint 本身。
  3. `rule_policy.rule_v1_step(memory, step, x, y, yaw_deg, d, n, last_turn, hfov_deg=60.0, params=...)`，规则与 Gazebo 上是同一个函数。
     记忆按集清零。不输出 SAFE_NEUTRAL；到点判定仍由 `run_episode` 做。
- 参数：`yaw_step_deg` 取 3DGS 动力学在该 hold 下实测的转角步中位数（从 `$D/exp-bearing-guard-3dgs-20261110/` 与 `$D/exp-sage3d-pilot-20261110/` 已有轨迹量，
  hold 0.5 的另量）。`r_safe` 与 `d_look` 只许在开发路线上选：室内 5 个场景里没进试点 40 条的合格路线每场景取 4 条，
  `r_safe ∈ {0.20, 0.25, 0.35}` × `d_look ∈ {0.8, 1.2}` 各跑 1 集，取到点成功最高的一组（并列取 `r_safe` 大的），写进入口脚本后冻结。
  其余参数（`mem_steps 8`、`n_cap 3`、`tie_frac`、`sum_clip`）不动。室外场景用同一组，不另调。
- 实验（守护进程，只用 3DGS 线，不占 gz；入口 `experiments/<日期>-rule-oracle-3dgs/`，输出 `$D/exp-rule-oracle-3dgs-<日期>/`）：
  室内 = 试点的 5 个场景、同 40 条路线、每条 3 集、同 seed；室外 = garden、bicycle、Wanderland、Callinstrasse 各 6 条路线 × 5 集、同 seed。hold 1.0。

  | 臂 | 列距离 | 航向参照 | 说明 |
  | --- | --- | --- | --- |
  | `forward+route` | 无 | 路线（护栏阈值 45°，与前两个实验相同） | 参照，重跑一遍用于配对 |
  | `rule-occ+route` | 占据真值 | 路线 | 主臂 |
  | `rule-depth+route` | 渲染深度 | 路线 | 图像可得的几何能到哪 |
  | `rule-occ+goal` | 占据真值 | 终点方位 | 不给路线，只靠局部避障 |
  | `rule-occ+route@0.5`、`forward+route@0.5` | 同上 | 路线 | 只跑室内，hold 0.5，步数预算乘 2 |

- 预注册（写进入口脚本头，起跑前定死）：
  主臂室内到点成功不低于 0.80，10 m 以上的路线不低于 0.60，碰撞率不高于 0.15；室外不低于 0.84（参照 0.89 减 0.05），碰撞率不高于 0.10。
  证伪：主臂室内低于 0.70。含义是地平线带的列距离加这条规则在室内不够，要么动作粒度太粗（看 hold 0.5 的臂），要么需要更多几何（门框、家具高度）。
  主比较三个：`rule-occ+route` 对 `forward+route`（室内，配对 McNemar）；`rule-depth+route` 对 `rule-occ+route`（室内）；`rule-occ+goal` 对 `rule-occ+route`（室内）。
- 另报：`depth` 与 `occ` 两种列距离在同一位姿上的一致性（每列绝对差的中位数与 90 分位；`depth` 比 `occ` 远 0.3 m 以上的列占比，即漏检）。
  这个数同时决定 3DGS 数据的几何头标签用渲染深度够不够（计划 §4.4 第 1 条）。失败集按「撞、超时（含左右交替的集数）、没到点」分类，按路线长度分桶。
- 测试（先红后绿）：桩占据（一堵墙加一个门洞）上 `occ` 列距离与解析值一致；桩深度图上 `depth` 列距离与构造值一致；n 的符号（目标在左时 n 为负，规则朝左转回）；
  驱动不给新臂时行为与配置逐位不变。
- 提交：`feat(3dgs): rule_oracle 策略（占据 / 渲染深度的列距离 + 规则 v1 + 路线方位）`、实验入口。

## 5. 第三波规格（细化后再开工，不要照这一节直接写代码）

| 项 | 要点 | 验收的形状 |
| --- | --- | --- |
| K2：3DGS 程序化障碍 | `scene.json` 加 `inserts`（圆柱 / 方柱列表）；渲染时把插入物的高斯与场景高斯拼接（基础张量常驻 GPU，每集只换插入部分）；碰撞与规划加解析障碍的叠加层；按 seed 在路线走廊内撒柱并做连通校验 | 插入柱在 RGB 与深度里一致（柱心列深度与解析距离差 ≤ 5 cm）；教师在 20 个课程上不低于 19/20；一集不超过 15 s |
| J4：多航点路线 | `routes[].waypoints`；到达当前航点后切到下一个；评测报「按序到达数」与 ISR / CSR / CGT | 现有单航点路线的结果逐位不变；5 个场景各有 2 条以上含 45° 弯的 2–3 段路线 |
| L3、L4：执行器与路书 | 程序格式（JSON）；执行器纯函数；把 A* 路径编译成「航向设定值加距离」的分段；执行时只用航向计数（或固件姿态估计）与步数里程 | 预注册：四个室外场景到点成功不低于 0.6（真值位姿的护栏是 0.89）；按路线长度分桶报告 |
| N：SAGE-3D 完整试点 | 20 个场景；多段拼接成 27–45 m 的任务；1.2 m 巡航；满电档 | 教师到点成功不低于 0.8；每个场景 8 条以上合格路线 |
| E：3DGS 数据集 B 放量 | 5 个室外场景加室内场景的教师数据，带渲染深度；几何头加条件码训练 S_B、S_AB | 插入障碍课程上 S_B 高于盲飞 0.3 以上 |
| L5、L6：模板指令与地标头 | 指令文法、编译到程序、闭集地标头 | 正确指令对打乱指令的成功率增益 p < 0.01 |

## 6. 每张卡的交付

1. 分支 `p12-<卡号>` 上的提交（逐文件 `git add`，中文提交说明，带 `fix:` / `feat:` / `test:` / `docs:` 前缀）。不合并、不推送。
2. 报告 `$D/tmp/phase12-cards/reports/<卡号>.md`：结论一段；改动清单（文件、行数）；先红后绿的真实输出；验收表（命令 / 结果）；
   「没做什么」逐条带原因；起了实验的写 PGID、日志、DONE、怎么停。
3. 实验数字另写 `docs/experiments/<日期>-<名字>.md`（命令、环境、时间、原始输出、配对检验），随分支提交。
4. 最后给用户或主会话的回复不超过 30 行。
