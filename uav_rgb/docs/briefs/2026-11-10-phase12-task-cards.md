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
nvidia-smi -L                                                # 以它列出的为准。11-11 起只有 GPU-b544b42e-… 一张，uav_rgb 就用它（公共规则 §4）；按 UUID 钉
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
| 2 | W2-5 | Depth-Anything-V2 Small 米制加规则 v1；先做「规则 v1 的记忆改用航位推算」的零训练对照 | 基线包装器已入库 | 3 个 | 是 | 中 |
| 2 | W2-6 | 多块课程与带扰动的教师轨迹进训练分布，T0 与 T1 评测 | 协议拍板、W1-0、W1-4、W1-7 | 6 个 | 是 | 中 |
| 2 | K1 | 3DGS 上的规则层上限：占据或渲染深度的列距离 + 规则 v1 + 路线方位，室内外各 120 集（零训练）。11-11 已完成并合入 | 规则 v1 已入库 | 否 | 是（3DGS） | 难，建议 Sonnet 5.5 Max |
| 2 | K3 | 规则 v2：两侧候选、分级通道、禁忌航向；先在 3DGS 室内的开发路线上选变体，再在留出路线上与 v1 逐集配对（零训练；11-11 晚立） | K1 已合入 | 否 | 是（3DGS） | 难，建议 Sonnet 5.5 Max |
| 2 | W1-12 | 3DGS 教师的前向净空检查改形状、出生判空可选两层检查、导引指针核查（11-11 23:30 细化） | W1-11 已合入；与 K3 只在 driver 的一处相邻改动 | 否 | 是（3DGS） | 中 |
| 2 | W1-13 | 新协议的记账与口径闭环；动作持续到下一次派发的协议修订与重标定（审核 A 的发现，11-12 立；02:15 开工） | 前置已合并 | 2 个（标定独占） | 是 | 难，建议 Sonnet 5.5 Max |
| 2 | S1a | 3DGS 线的条件通路与划分：学生收到的上一步动作取实际执行的；`eval_3dgs` 不再把 split 写死成 test（审核 C2、G1、F1、D1；E4 的前置；11-12 02:15 开工） | 前置已合并 | 否 | 是（3DGS 冒烟） | 中 |
| 2 | S1b | 3DGS 线的碰撞口径（斜向膨胀、高度跳层、逐段扫掠）加版本键 `collision_model`，新旧口径下重定基线（审核 B、E-12；11-12 02:15 开工） | 前置已合并 | 否 | 是（3DGS） | 难，建议 Sonnet 5.5 Max |
| 2 | W1-14 | 统计口径：按几何聚类的检验与区间、多 run 配对、协议一致性键、分母（审核 D、A4；11-12 02:15 开工） | 前置已合并 | 否 | 否 | 中 |
| 2 | W1-15a | 实验入口的公共脚本 `tools/exp_lib.sh`（锁、冻结代码、起跑记录、池的起跑条件、完成标记）与核对后再停的工具（审核 E；11-12 02:15 开工） | 前置已合并 | 否 | 否 | 中 |
| 2 | W1-15b | 三条 Python 流水线：输出目录锁、「评完」的统一判据、半成品续跑、分母、透传协议开关（审核 E、A2；11-12 02:15 开工） | 前置已合并 | 否 | 否 | 中 |
| 2 | C1a | 训练器的小修与数值测试（C++，不动 checkpoint 格式；审核 H8、H9、C6 与上一轮 D2、D3、D6、D9；另量训练进程的 CPU 占用） | T1 合并之后 | 否 | 是（编 C++） | 中 |
| 2 | T1 | 把交叉审核 F 提议的用例入库：标签来源与图像对齐、推理输入布局、翻转增强、判据边界（只加测试；11-12 立，立即开工） | 无 | 否 | 是（编 C++ 测试） | 中 |
| 端到端 | E2E-C1 | 条件通路进契约（C++：checkpoint 记条件种类与前处理契约；规格初稿） | E2E-1 出结果之后 | 否 | 是 | 难，建议 Sonnet 5.5 Max |
| 端到端 | E2E-0 | 可判定性审计：真值列距离与动作历史能决定多少教师标签（离线，11-12 立） | 无 | 否 | 可选 | 中 |
| 端到端 | E2E-1 | 航向码端到端学生：W1-9 的实现，加「示范 / DAgger 聚合 × 有无航向码」2×2 试点，不带护栏（11-12 立） | W1-0 第二轮的 L | 2 个 | 是（训练） | 难，建议 Sonnet 5.5 Max |
| 端到端 | E2E-P0 | PyTorch 原型线的地基：模型复刻、checkpoint 读写、行协议包装器、训练配方复刻，逐项与 C++ 对拍（11-12 立） | 无 | 2 个 | 是（训练） | 中 |
| 端到端 | E2E-R1 | 训练配方的稳定性筛选（现成开关：余弦退火、显式类权重）与工作点的闭环扫描（审核 H；11-12 立，立即开工） | 无 | 2 个 | 是（训练） | 中 |
| 端到端 | E2E-1b | 带条件模型的按码校准（规格初稿） | E2E-1 的第二步与训练权重 | 2 个 | 否 | 中 |
| 端到端 | E2E-P1 | 原型线上的结构与配方探索：logit adjustment 与 EMA、列结构头、FiLM 前移、列特征缓存（规格初稿） | E2E-P0 对拍通过；E2E-0、E2E-R1 的结果 | 2 个 | 是 | 难 |
| 3 | W3-x | 3DGS 程序化障碍、多航点路线、执行器与路书、SAGE-3D 完整试点、数据集 B 放量。11-12 起「执行器与路书」不作为交付路线（计划 §10.3） | 第 1、2 波的结论 | | | 难；规格见 §5，细化后再开工 |

第 1 波的卡互不依赖，可以并行；W1-3、W1-4、W1-9 都改 `sim/tools/collect.py` 或 `sim/tools/eval_closed_loop.py`，并行时各自只动卡里写的函数，合并由主会话处理。
gz 的上限随核数定（24 核 7 个，12 核 3 个，公共规则 §4），起实验前先看当前占用；下表「占 gz」一栏是按 24 核写的，12 核时按比例减并把实验排队。

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
- 状态（11-11）：代码与测试已合入 main。第一轮标定门没过；离线归因后第二轮（`experiments/2026-11-10-action-latency/run_r2.sh`，位移门用巡航步口径）
  11-11 21:42 起跑，输出 `$D/exp-action-latency-r2-20261111/`。报告 `docs/reviews/2026-11-10-phase12/w1-0.md`，计划 §1.3。

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
- 11-11 更新（主会话；上面「验收」一节的对照与规模以这里为准）：验收改在新协议下做。W1-0 第二轮（`$D/exp-action-latency-r2-20261111/`）的标定在 L=0.20 过门
  （教师跑 3 块长课程要带 `--rtf 0.5`，学生臂不带）。对照用该目录下 50% 电量的六个学生臂：`acc-L0.20-k1-s{0,1,2}` 与 `acc-L0.20-k3-s{0,1,2}`
  （各 20 个冻结 seed，calswa 三个训练种子，`--heading-guard 2`，`--action-latency 0.20`）。满电档跑同样的六个臂：命令行照 `run_r2.sh` 的 `acc_job`，
  只多满电副本的参数；2 条车道（与第二轮相同）；含迟到步的集按 `run_r2.sh` 的条款 R6 重跑一次。K=1 60 集、K=3 60 集，与对照逐集配对。约 1.5–2 小时，拿机器锁。
  预注册：K=1 的每步前飞位移与转角（口径同 `summarize_r2.py`）与 50% 档相差不超过 5%，成功数配对无可检出的差别（精确 McNemar p > 0.05）；
  K=3 满电档里飞到电量掉高的集数为 0，第 1 块通过率与 50% 档相差不超过 0.08，三块全过的集数不少于 50% 档。证伪：K=1 的位移或转角相差超过 10%。
  第二轮没有 `DONE` 或 `DONE` 的 rc 不是 0 时先停下报告，不要自己起对照臂。
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
  2026-11-12 07:30 更正：训练器不接受 0 权重，写成 `--class-weights 1e-6,1,2,2`；并加 `--cosine-lr`（卡 E2E-R1 第一部分：R3e 末 10 轮翻转率 0.012、三种子一致率 0.926，不加余弦的恒定学习率是 0.236、0.671）。
  `--augment photo` 等卡 C1a 修好增强的随机数种子（审核 H8）之后再开。配方最终取 R2e 还是 R3e，看 E2E-R1 第二部分的闭环结果。
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
- 另报（不设门槛）：T1（K=3）的逐块存活。几何头的标签把 4 m 以外的柱子归到最远一档，如果它起作用，第 1 块的通过率应当比不带几何头的臂更接近 T0。
  W2-6 的混合数据出来之后，用混合数据再训一组，同样报 T0 与 T1。
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
  估计深度图 → 按 `rule_policy` 的列划分取地平线带每列最小深度 → `rule_policy.rule_v1_step`（与规则层上限实验 `rule60v1` 同一条规则、同一组参数 `RULE_V1`；
  规则 v0 在 11-10 晚已判负，不再用）。规则 v1 的障碍记忆要位姿：这里不许用真值，用航位推算（每个已执行的前进步沿当前计数航向走 0.25 m，
  每个转向步 26.1°，数字来自 11-10 的实测），包装器自己维护。
  先在 100 张闭环帧上报告估计深度与解析深度的相对误差（中位数、p90）与单步时延。
- 先做一个不用下载的对照臂（零训练，3 个 gz）：真值列距离加规则 v1，但记忆用同样的航位推算位姿（`collect.py` 的 rule-oracle 分支加一个开关，默认关）。
  对照是 `$D/exp-rule-oracle-v1-20261110/rule60v1-rep{0,1,2}`（真值位姿的记忆，57/60）。预测相差不超过 3/60；超过就说明规则层的记忆离不开里程，
  要先解决里程来源（光流或固件估计）再谈任何「估计几何加规则」的臂。
- 评测：室内权重、室外权重各一臂，柱阵 20 个冻结 seed × 3 次重复；对照是上一条的航位推算臂与 `rule60v1`。
- 预注册的主比较：我们的学生对「室内权重加规则」。预测：灰色无纹理柱体上米制尺度偏差大，估计深度臂低于真值臂 10 集以上。

### 卡 W2-6：多块课程进训练分布（11-10 晚新增）

- 背景（计划 §1.3「长柱阵课程」）：只在单块柱阵上训练的学生，到了 3 块课程里每块的通过率从 0.83 掉到 0.65–0.70，整条 0.32。
  第 0 步出生位姿相同，单块 140 集全部先前进，3 块课程 120 集里 9 集先右转：远处的柱子改变了网络的输出。训练数据里没有 5 m 以外的柱子，也没有不居中的入口位姿。
- 前置：计划 §8 第 1 条的协议拍板与 W1-0（新数据按新协议采，免得重采两遍）；W1-7（新教师）；W1-4。
- 改动：
  1. `collect.py` 加 `--action-noise EPS`（默认 0，逐位不变；只对教师策略有效）：每步以概率 EPS 改执行 {前进, 左转, 右转} 里随机一个，
     标签仍是教师动作，行里 `executed_by` 记 `noise`。目的是让数据覆盖偏离教师航线的状态。几何标签只取决于位姿，谁在飞都可以用；
     动作标签是教师在该状态下的动作，与 DAgger 的影子教师同义。`audit_dataset.py` 对这种行的处理照 DAgger 的行（看 `executed_by` 的现有分派）。
  2. `tools/e1_pipeline.py` 或新的入口脚本支持混合任务清单：`base`、`course_base --blocks 2`、`course_base --blocks 3`，各自的 train / val 分区。
- 数据（守护进程，6 个 gz，train 分区）：`base` 300 集、`--blocks 2` 150 集、`--blocks 3` 150 集，其中一半带 `--action-noise 0.15`；另采 val 各 30 集。
  体积先估（每集 80–240 帧 PNG），超过 3 GB 先停下报告。
- 训练：与 W2-1 同配方，3 个种子；另训一组只用 `base` 的同帧数对照。
- 评测：T0（柱阵 60 集）与 T1（K=3，60 集），无护栏与加护栏各一遍。
- 预注册：混合数据臂在 T1 的第 1 块通过率不低于它在 T0 的成功率减 0.08；T1 整条不低于 0.50（现在 0.32）；T0 不低于只用 `base` 的对照臂减 3/60。
  T1 整条仍低于 0.40，则「加数据覆盖」不够，转向几何头加规则（W2-3 的后续 D5）。

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
- 结果（11-11，已合入 main）：室外主臂 117/120 = 0.97；室内主臂 80/120 = 0.67，预注册的 0.80 没有达到，证伪线成立。数字与失败形态见计划 §1.3 与
  `docs/reviews/2026-11-10-phase12/k1.md`。后续是卡 K3。

### 卡 K3：规则 v2（两侧候选、分级通道、禁忌航向；零训练；11-11 晚立）

- 回答的问题：K1 量到规则 v1 在 3DGS 室内只有 80/120 = 0.67，40 个失败集里 23 个以「再转开一步」与「到顶转回」两步循环收尾，另有 9 个含 6 步以上的左右交替；
  循环时三个候选航向的通道全被判堵。Gazebo 的 3 块长课程上规则 v1 的 9 个失败里有 3 个是同一种循环。
  这张卡把循环去掉，量同样的感知下规则层在室内能到多高，剩下的失败是哪一类。
- v1 为什么循环（主会话读 `rule_policy.rule_v1_entry` 与 K1 的逐步记录得到的结论；开工前自己核对一遍，对不上就停下报告）：
  1. 每步只在三个航向里选（朝回一步、保持、再转开一步）；转开只沿 n 的符号那一侧，另一侧从不尝试。
  2. 「再转开一步」（V1-more）不看要转去的航向通不通；到 |n| = n_cap 时「到顶转回」（V1-cap）无条件转回一步，转回的航向刚判过是堵的。两条互相抵消。
  3. 视场 60°，一步转 31.7°，转一步之后的航向有一部分在视场外。没看过的方向没有障碍点，按畅通算，面对它之后才发现是堵的。
  4. 通道只有一种尺寸（半宽 r_safe、长 d_look）。室内家具与门洞附近，1.2 m 长的通道在不少位置对所有航向都不通。
- 先读：`sim/tools/rule_policy.py` 全文；`sim/3dgs/rule_oracle.py`；`experiments/2026-11-10-bearing-guard-3dgs/driver.py`（K1 加的规则臂与 `rule_params`）；
  `experiments/2026-11-10-rule-oracle-3dgs/` 下的 `run.sh`、`summarize.py`、`select_dev_routes.py`、`select_params.py`、`formal_failure_analysis.py`、
  `README.md`、`RESULT.md`、`dev_failures.md`、`formal_failures.md`；报告 `docs/reviews/2026-11-10-phase12/k1.md`（§6、§11、§16、§18）
  与 `w1-11.md`（§7.4 切换步的位移，§8.2 driver 传 hold）；规则 v1 的离线运动学模拟在 `experiments/2026-11-10-rule-oracle-v1/`（看有没有能直接复用的模拟器）。
- 规则 v2：在 `sim/tools/rule_policy.py` 里新增 `RULE_V2`、`V2_ENTRIES`、`rule_v2_entry`、`rule_v2_step`。v1 的常量与函数一行不改，对已有行的 diff 必须是 0。
  状态由调用方按集清零、逐步传入传出：`memory`（同 v1 的世界系障碍点）与 `tabu`（禁忌表）。
  记号：α = `yaw_step_deg`；n = 机头偏离导引方向的整数步数（同 v1，正 = 偏左）；e = 同一个量的浮点值（3DGS 里是 −err/α，调用方不给就取 n）；
  m = 相对当前航向再转的步数（正 = 左转），第 m 个候选航向相对机头的角度是 m·α，它的 n 是 n + m。
  1. 记忆：同 v1，保留步数取 `RULE_V2["mem_steps"]`（24）。
  2. 通道分级：`levels` 是一个列表，每级（r_safe、d_look、是否只许静止起步）。要求逐级不放宽（r_safe 与 d_look 都不增大），否则抛 ValueError。
     于是某一级畅通则更靠后的级也畅通，某一级被堵则更靠前的级也被堵。「只许静止起步」的级只在上一步执行的不是 FORWARD_SLOW 时可用
     （转向后的第一个前进步位移约 0.14 m，连续前进约 0.30 m，见 w1-11 §7.4）。
  3. 禁忌表：每步先看当前航向（m = 0）。它在第 0 级被堵时记一条：世界航向 yaw_deg、位置 x 与 y、步号、lb（被堵的最靠后的级）。
     候选航向 ψ 在第 ℓ 级被禁，当且仅当表里有一条同时满足：航向差不超过 α/2、位置差不超过 `tabu_radius`、步龄不超过 `tabu_steps`、ℓ ≤ lb。
     超龄的条目每步清掉。禁忌表按世界航向记，不按 n 记：导引目标点会随位置跳，n 的编号不稳（w1-11 §6 里教师的瞄准点就有这种跳变）。
  4. 候选是否可用：`ok(m, ℓ)` = `corridor_clear(记忆点换到机体系, m·α, 第 ℓ 级的 r_safe 与 d_look)` 为真，且该航向在第 ℓ 级没有被禁。
     m = 0 不查禁忌表（它是本步直接看到的）。
  5. 决策顺序有两种，由参数 `order` 选。三个基本动作：
     a(ℓ)：n ≠ 0 且 `ok(朝回一步, ℓ)` → 朝回转一步，条目 `V2-back<ℓ>`；
     b(ℓ)：`ok(0, ℓ)` → FORWARD_SLOW，条目 `V2-fwd<ℓ>`；
     c(ℓ)：在 m ≠ 0、|m| ≤ `m_max`、|n + m| ≤ max(`n_cap`, |n|) 的候选里取 `ok(m, ℓ)` 为真的，按（|m| 小、|e + m| 小、v1 的左右两半求和偏向的那一侧、
     `last_turn` 那一侧、左）依次比较取最优，朝它转一步，条目 `V2-seek<ℓ>`。
     `lexi`（逐级）：对 ℓ = 0, 1, … 依次做 a(ℓ)、b(ℓ)、c(ℓ)，第一个命中的返回。含义：短通道只在长通道上两侧都找不到路时才用。
     `near`（先前进）：a(0)、b(0)、b(1)、b(2)、…、c(0)、a(1)、c(1)、a(2)、c(2)、…。含义：当前航向在长通道上不通、在短通道上通时先往前走，不去绕路。
     当前不可用的级（只许静止起步而上一步是前进）整级跳过。全都没有命中 → 沿 `last_turn` 再转一步（没有则左转），条目 `V2-stuck`。
  6. 不变量（都要有测试）：永不输出 SAFE_NEUTRAL；FORWARD_SLOW 只在当前航向某一可用级的通道畅通时输出；同一位置上面对过并判为堵的航向，
     在禁忌有效期内不会再被选为候选；不出现周期为 2 的原地转向循环（构造 K1 的循环场景：v1 循环 20 步以上，v2 在 2·n_cap + 3 步内前进或落到 `V2-stuck`）。
  7. `rule_v2_step` 返回（动作、条目、新记忆、新禁忌表、info）。info 至少含 `mem_n`、命中的级、选中的 m、禁忌表长度、
     m = −n_cap … n_cap 各候选在每一级上的 `ok` 位图。
- 默认参数 `RULE_V2`（Gazebo 侧的默认，3DGS 里由实验入口覆盖）：沿用 `RULE_V1` 的 `r_safe` 0.35、`d_look` 1.2、`yaw_step_deg` 26.0、`n_cap` 3、`tie_frac`、`sum_clip`；
  新增 `mem_steps` 24、`levels` 只有一级（0.35、1.2、否）、`order` lexi、`tabu_radius` 0.30、`tabu_steps` 24、`m_max` 6。
- 3DGS 接线：`RuleOraclePolicy` 加 `rule_version`（缺省 v1，逐位不变）；v2 时把 e = −err/α 传给规则，`rows` 多记 v2 的 info；`planner_name` 与 `version` 按版本区分。
  driver 的规则臂加可选键 `rule_version`；`rule_params` 按版本校验键名；不给新键时旧臂逐字节不变（K1 的对拍测试保持绿）。
  Gazebo 侧（`collect.py --rule-version`）这张卡不接，等 3DGS 的结果。
- 顺带的一处改动（单独一个提交；w1-11 §8.2）：driver 构造教师时传 `hold_s=CFG["hold"]`，显式的 `teacher_tol_deg` 仍然优先。加测试；
  在 `experiments/2026-11-10-bearing-guard-3dgs/` 的说明里写一句：不带 `teacher_tol_deg` 的旧配置在 hold 1.0 下重跑会得到 18.9° 的教师，复现旧结果要显式写 12.0。
- 实验（入口 `experiments/2026-11-11-rule-v2-3dgs/`，输出 `$D/exp-rule-v2-3dgs-20261111/`；沿用 K1 入口的结构与共用的 driver；hold 1.0；`yaw_step_deg` 31.7；
  一个守护进程把「开发 → 自动选变体 → 正式」串起来，中间不需要人改文件）。
  路线分三份，互不重叠，都来自试点的 5 个室内场景，用 K1 的 `select_dev_routes.py` 同一套合格条件与按 A* 长度分位的取法：
  正式 40 条 = K1 的同一批（同 seed，8000 + 100·路线下标 + k）；
  开发 40 条 = K1 的 20 条开发路线加新取的 20 条（每场景 4 条；seed 9000 + 100·路线下标）；
  留出路线 = 再新取的每场景 8 条（不够就有多少取多少；合计少于 24 条就停下报告）；seed 5000 + 100·路线下标 + k。
  留出路线在这张卡之前没有被任何规则跑过，是主终点。正式 40 条的失败形态在设计 v2 时已经看过，单列为次终点。
  1. 开发阶段（40 条 × 1 集）七个臂：`v1`（r_safe 0.25、d_look 1.2，K1 的冻结值）与六个 v2 变体。

     | 变体 | levels（r_safe, d_look；带 * 的级只许静止起步） | order | 想分清什么 |
     | --- | --- | --- | --- |
     | `v2a` | (0.25, 1.2) | lexi | 只改决策逻辑 |
     | `v2b` | (0.25, 1.2)、(0.25, 0.8) | lexi | 加一级短通道，绕不开时才用 |
     | `v2c` | (0.25, 1.2)、(0.25, 0.8) | near | 加一级短通道，先前进 |
     | `v2d` | (0.25, 1.2)、(0.25, 0.8)、(0.20, 0.55)* | near | 再加一级静止起步的更短通道 |
     | `v2e` | (0.15, 1.2) | lexi | occ 列距离量到的是膨胀后的边界，半宽按它扣掉 0.10 |
     | `v2f` | (0.15, 1.2)、(0.15, 0.8)、(0.12, 0.55)* | near | 窄半宽加分级 |

     选变体的规则（写进入口脚本，自动执行）：撞在前多于 3/40 的臂出局；剩下的取到点成功 SR(stop) 最高的；与最高相差不超过 1 集的算并列，并列取表里靠前的。
     闸门：选中的变体比 `v1` 在同一批 40 条上多到点 3 集以上才进正式阶段；否则停，DONE 写明「开发路线上 v2 没有比 v1 好」，照实报告。
  2. 正式阶段（变体选定后冻结，写进 `$OUT/selected.json`，正式阶段只读它）：
     留出路线 × 3 集：`forward+route`、`rule-occ+route`（v1）、`rule2-occ+route`（主臂）、`rule2-depth+route`；
     正式 40 条 × 3 集：`rule-occ+route`（v1 重跑，先与 K1 的结果同键对拍，应逐集一致，不一致就停）、`rule2-occ+route`、`rule2-depth+route`；
     室外 24 条 × 5 集（同 K1 的 seed）：`rule2-occ+route`、`rule2-depth+route`；v1 取 K1 的结果按键配对。
     depth 臂的 levels：选中的是 `v2e` 或 `v2f` 时，depth 臂用同结构的 0.25 一档（`v2a` 或 `v2d`），因为渲染深度量到的是表面，不是膨胀后的边界。
- 预注册（写进入口脚本头，起跑前定死；主会话 11-11 定，不许改）：
  主终点（留出路线，`rule2-occ+route` 对 `rule-occ+route`，同路线同 seed 逐集配对）：到点成功率 SR(stop) 高 0.10 以上，精确 McNemar 双侧 p < 0.05；
  主臂碰撞率（整集任一步）不高于 0.10，撞在前不高于 0.08。
  证伪：到点成功率的差低于 0.05，或主臂碰撞率高于 0.15。含义是两侧搜索加分级通道补不上室内的缺口，下一步要的是更细的航向或更多几何，不是规则顺序。
  落在两者之间记「介于预测与证伪之间」。
  次终点（正式 40 条，同样的比较，只报告）：另报主臂是否到 0.80（K1 没达到的那条预测）。
  室外（不劣）：主臂到点成功不低于 0.93；对 v1 逐集配对，「v1 成而 v2 败」的集数减去「v2 成而 v1 败」的集数不超过 5；碰撞率不高于 0.10。
  depth 臂只报告。判读口径沿用 K1（SR(stop)、碰撞率含到点之后、点估计对阈值并列 Wilson 95% 区间）。
- 另报：失败形态（沿用 K1 的分类，加 `V2-stuck` 步占比、各级 `V2-fwd` 的占比、每集转向步占比与平均步数）；按路线长度分桶，10 m 以上单列；出生即碰撞的集单列；
  导引目标点不前进的集数；以 `V2-stuck` 收尾的集逐集列出位置与到占据的最小间隙（这些是规则顺序之外的缺口）。
- 测试（先红后绿，带变异检查）：第 6 条的四个不变量；禁忌的三个失效条件（走远、超龄、级别）；levels 不逐级收紧时报错；两种 `order` 的先后；
  v1 逐位不变（同一批随机输入分别喂 main 上的旧文件与新文件里的 `rule_v1_step`，输出逐字相同）；3DGS 接线里 e 与 n 的符号；driver 不给新键时旧臂逐字节不变；
  有现成的离线运动学模拟器就把 v2（Gazebo 默认参数）在 20 个 train 几何上跑一遍，过线数与撞柱数不差于 v1（18/20、0），结果写进报告，差了也照实写。
- 资源与排队：只用 `GPU-b544b42e-…`。开发与正式阶段都拿机器锁（阻塞式 flock，同 K1 的 formal）：W1-0 的第二轮标定正占着这把锁，它对墙钟敏感。
  第二轮的标定阶梯结束之前（`$D/exp-action-latency-r2-20261111/` 下既没有 `L_STAR` 也没有 `DONE`）不跑任何超过几秒的命令，只读写文件；
  阶梯结束之后可以跑单进程的测试，可以不拿锁跑一次单车道、不超过 3 分钟的 GPU 冒烟，起止时间写进报告。`tools/check.sh` 不跑，合并时由主会话跑。
  11-11 23:55 更新（用户指示实验可以并行；以公共规则 §4 同一时刻的那一条为准）：3DGS 批量实验改拿 3DGS 池的锁 `$D/tmp/phase12-cards/gs3d.lock`，不再排 `machine.lock`，
  与 Gazebo 实验同时跑；单进程测试随时可以跑。
- 11-12 定位更新（计划 §10）：交付路线改为端到端，规则层不作为部署模块。这张卡的结果有两个用途：感知完美时固定规则的上限（端到端学生的对照基线）；
  训练期可模仿教师的候选。报告里加一小节，逐条列出规则 v2 的决策用到了哪些学生拿不到的量（真值位姿下的障碍点记忆、路线方位等）。
- 提交：`feat(rule): 规则 v2（两侧候选、分级通道、禁忌航向），v1 不变`、`feat(3dgs): rule_oracle 与 driver 接规则 v2`、`fix(exp): driver 构造教师时传 hold`、实验入口与结果文档。

### 卡 W1-12：3DGS 教师的前向净空检查改形状、出生判空、导引指针（11-11 23:30 细化，可以开工）

- 依据：`docs/reviews/2026-11-10-phase12/w1-11.md` §6、§7、§8 与 `k1.md` §6.1、§16、§18。新教师（容差 18.9°）室内 19 个没到点的集全是撞在前：
  3 集是出生点的问题；15 集撞在前进与转向互相切换的那一步；1 集撞在连续前进时。超时 0。
  教师的前向检查在 `OccAstarTeacher.__call__` 末尾：航向对准后沿机头做一条 `FORWARD_CHECK_M` = 0.4 m 的 `occ.sweep`（零宽线段）。
  转向后的第一个前进步，实际位移方向偏离机头中位 25.5°（p90 34.4°），位移约 0.14 m；前进后的转弯步前滑中位 0.066 m（p90 0.134 m）。
- 先读：`sim/3dgs/planner.py`（`OccAstarTeacher`、`OccPlanner.aim_point`）；`sim/3dgs/collision.py` 的 `Occupancy.collides`、`sweep`、`clearance`；
  `sim/3dgs/collect_3dgs.py` 的 `_spawn_for_episode`、`run_episode`（每步写进 `actions.jsonl` 与 `oracle.jsonl` 的字段）、`main()` 里构造教师的地方；
  `sim/3dgs/dynamics_sitl.py`（动力学对象对策略暴露了哪些量，有没有水平速度）；`sim/3dgs/guard.py` 的 `OracleRouteGuide.target`；
  `experiments/2026-11-10-teacher-tol-3dgs/`（W1-11 的验收入口，`analyze.py` 的回放与逐集配对；这张卡沿用它的结构）；
  草稿目录 `$D/tmp/phase12-cards/w1-11/` 里的 `fail_replay.py`、`fail_verify.py`、`step_types.py`（只读，可以照着写）。
- 第一部分：前向净空检查（主体）。分三步，前一步的结论决定后一步，不许跳。
  1. 量（离线，只读 `$D/exp-teacher-tol-3dgs-20261110/` 的新教师臂 240 集，不起仿真）：把每一步的实际位移（`pose_before` → `pose_gt`）换到动作前的机体系。比两个预测器。
     P1「上下文表」：按（上一步执行的动作类，本步动作类）分组取位移向量的中位数；转向后的前进步按上一步的转向方向分左右两组（位移偏向哪一侧要量出来，不猜）。
     P2「速度外推」：只在记录里或动力学对象上拿得到动作前的水平速度时做。对每类动作用最小二乘拟合「位移 = A·机体系速度 + b」。
     按路线下标的奇偶分成拟合与检验两半，报告检验那一半上预测误差（预测位移与实际位移之差的模）的中位数与 p90。取 p90 小的；两者相差不到 0.01 m 取 P1。
  2. 判（离线反事实）：检查 = 沿「预测位移，再顺着它延长 `reserve`」的线段做 `occ.sweep`。`reserve` = 前进后转弯步前滑的 p90（约 0.13 m）加第 1 步量到的预测误差 p90。
     只对 FORWARD_SLOW 判；转向不否决（转向没有替代动作）。在同一批 240 集上逐步重算，报告两个数：
     命中 = 16 个非出生点的撞在前集里，碰撞步或它之前 2 步内有前进步被否决的集数（预期不少于 12）；
     误拒 = 没有碰撞的集里被否决的前进步占全部前进步的比例（预期不高于 5%）。
     命中不到 10 集或误拒高于 10%：停下，把两个数与逐集表写进报告交回，不往下做。
  3. 改：`OccAstarTeacher` 加可选参数 `clearance`，缺省 `"ray"`（与现在逐步相同）；`"swept"` 时在原来的机头线段检查通过之后再做第 2 步的检查。
     被否决时：|err| 不小于容差的一半就按 err 的符号转（分支名 `turn_swept`）；否则出 SAFE_NEUTRAL 等一步（分支名 `wait_swept`）。
     连续等待不超过 2 步，第 3 步按原规则（只看机头线段）放行，免得卡死。goal-stop 的「连续 3 步 SAFE 提前结束」不能被等待步误触发，用测试锁住。
     `clearance="swept"` 的教师版本记 `uavrgb.teacher_occ.v3`，manifest 多写 `teacher_clearance`。`collect_3dgs` 加 `--teacher-clearance {ray,swept}`，缺省 ray。
     实验 driver（`experiments/2026-11-10-bearing-guard-3dgs/driver.py`）只加一个可选键 `teacher_clearance`，放在最后做，改动控制在构造教师的那一处；
     卡 K3 正在改同一处（给教师传 `hold_s`），合并冲突由主会话处理，报告里写明你改了哪几行。
- 第二部分：出生判空。`_spawn_for_episode` 用 `scene.spawn.z` 判空，起飞后的飞行高度在另一层栅格（w1-11 §7.2：120/120 集两层不同，2 集起点在碰撞栅格里）。
  加开关，缺省关（出生位姿与随机数的消耗顺序逐位不变，用测试锁住）。开的时候：抖动出的候选点要在出生高度与巡航高度两层都不碰撞，
  并且 `occ.clearance` 不低于机体半径加 0.05 m。manifest 记开关值。
- 第三部分：导引指针，先复现再定改不改。`OracleRouteGuide.target` 的指针只在「下一个顶点不更远」时前进一格。K1 的开发与正式各有 1 集目标点停住
  （k1 §6.1 的 841150 r13 与 §16.1）。用那两集的轨迹离线重放 `target()`，说清指针停在哪、为什么。
  改法（加可选参数，缺省旧行为）：从当前指针往前、沿路径累计不超过 `lookahead_m` 的范围内，取离飞机最近的顶点。复现不了就只写报告，不改代码。
- 验收实验（入口 `experiments/2026-11-11-teacher-clearance-3dgs/`，输出 `$D/exp-teacher-clearance-3dgs-20261111/`；守护进程；拿机器锁；
  室内 = 试点 40 条 × 3 集，室外 = 4 个场景各 6 条 × 5 集，seed 与 W1-11 的验收相同，hold 1.0，容差 18.9°）：
  臂 A：`swept`，出生开关关，与 W1-11 的新教师臂逐集配对。臂 B：`swept` 加出生开关开，只报告。回归臂：`ray`，抽室内 24 集、室外 30 集，与 W1-11 的新教师臂逐步一致。
  预注册（写进入口脚本头，不许改）：臂 A 室内到点成功率不低于 0.90，撞在前不高于 0.06，超时不高于 0.03；室外到点不低于 0.97；
  对 W1-11 的新教师臂逐集配对，室内「新成旧败」减「旧成新败」不少于 8 集。证伪：室内到点低于 0.87，或超时高于 0.05。落在之间记「介于预测与证伪之间」。
  另报：`wait_swept` 与 `turn_swept` 的步占比；每集平均步数；SAFE_NEUTRAL 标签的占比（教师数据的类别分布会不会被等待步拉偏）；仍然撞在前的集逐集归类。
- 测试（先红后绿，带变异检查）：`clearance="ray"` 与改动前逐步相同；合成占据上「机头线段通、预测位移线段不通」时 swept 否决而 ray 放行；连续等待的上限；
  出生开关关闭时随机数消耗与出生位姿逐位不变；导引的新参数缺省时 `target()` 的输出不变。
- 不做：不改 A* 的规划余量，不改 hold，不改航向容差，不动 `rule_oracle.py`、`rule_policy.py` 与 driver 里的规则臂（卡 K3 在改）。
- 提交：`feat(3dgs): 教师前向净空按预测位移做扫掠检查（clearance=swept，缺省不变）`、`feat(3dgs): 出生判空可选两层检查`、
  `fix(3dgs): 导引指针在前视范围内取最近顶点`（复现了才有）、实验入口与结果文档。

### 卡 W1-13：新协议的记账、口径闭环与步间刹停（审核 A 的发现 1–5；11-12 立，等第一波合并后开工）

- 依据：`docs/reviews/2026-11-12-cross-review/review-a.md` 与同目录 `FIXLIST.md` 的 A1–A5。要点：
  `--action-latency L` 固定了「观测到派发」，没有管「上一动作截止到下一动作派发」。执行器截止 = 观测戳 + hold，下一次派发 = 下一观测戳 + L，
  相邻观测相隔 frame_dt（现在恒为 0.9 s：相机 10 Hz，动作窗口结束时到手的最新一帧是 0.1 s 前拍的）。于是每两步之间有 L − (hold − frame_dt) 的空档，执行器回 SAFE 刹停：
  L=0.20 时恒为 0.100 s，旧协议中位 0.015–0.025 s。新协议每步前飞位移比旧协议小 17% 的原因在这里。这段空档不落盘，迟到判据也看不到步间停顿。
- 前置：W1-3、W1-4、W1-9、W1-10、W1-2 合并之后（都改 `sim/tools/collect.py`、`sim/tools/eval_closed_loop.py` 或 `tools/stats_report.py`）。
- 先读：审核报告的发现 1–5 与它的两个统计脚本（`$D/tmp/phase12-cards/review-a/gap_stats.py`、`stalls.py`）；`sim/tools/collect.py` 的 `collect_one`（取帧、等派发、`sil.dispatch`）；
  `sim/tools/cf_sitl_exec.py` 的 `dispatch` 与插件侧对 `until` 的处理（`sim/cf_sitl/gz/`：到期后回 SAFE 是位置保持还是滑行，新动作到来时旧动作是否立刻被取代）；
  `experiments/2026-11-10-action-latency/` 的 `run_r2.sh` 与 `summarize_r2.py`（标定门的写法，这张卡的标定沿用）；`docs/EVAL_STANDARD.md` §5.5。
- 第一部分，记账（不改飞行行为；默认输出只多键，用测试锁住其余逐位不变）：
  1. 每步行加 `safe_gap_s`（本步派发戳减上一步动作截止，首步为 null）与 `obs_gap_s`（本观测戳减上一观测戳）。
  2. `summarize_episode` 报 `safe_gap_s` 的中位数与最大值、`obs_gap_s` 的最大值；`obs_gap_s` 超过名义步周期 0.2 s 以上的步记进 `stall_steps`。
     带 `--action-latency` 时，含停顿步的集与含迟到步的集同等处理（进 `failed[]`，占请求名额，不计成功）；不带开关时只记不判。
  3. 插件侧的实际生效时刻：订阅 state 话题，记每步动作字段第一次等于本步动作时的仿真时刻 `plugin_ack_sim_t`。先量后定：在空闲与满载（另起 8 个占核进程）两种条件下各跑教师 5 集，
     报 `plugin_ack_sim_t` 减（观测戳 + L）的分布。分布的 p99 小于 10 ms 就只记录；否则迟到判据改用它，阈值按量到的分布定并写明依据。
- 第二部分，口径闭环：
  4. `tools/stats_report.py` 的协议一致性键加 `action_latency_s`、`heading_guard`、`rtf`。缺键等于旧口径，与带键的不能进同一张表。加测试（两份只差这一个键的 eval.json 应当报错）。
  5. `eval_closed_loop.py`：训练数据的协议快照没有 `action_latency_s` 而本次给了 L 时，在 eval.json 里记 `protocol_notes`，不判 OOD，报表能看到。
  6. `collect_parallel.py`、`tools/e1_pipeline.py`、`tools/dagger_gz_pipeline.py` 透传 `--action-latency` 与 `--rtf`（默认不给，命令行逐字不变）；`preprocess` 加开关丢掉迟到步与停顿步的行（默认关）。
- 第三部分，协议修订：动作持续到下一次派发（新开关，默认关，关的时候逐位不变）：
  7. `collect.py` 与 `eval_closed_loop.py` 加 `--action-until {deadline,next-dispatch}`，缺省 `deadline`（现状）。`next-dispatch` 只能与 `--action-latency` 同用：
     执行器截止取「派发戳 + 名义步周期」，名义步周期先量后定（现在 frame_dt 恒为 0.9 s），相邻两步首尾相接，步间不回 SAFE。
     SAFE_NEUTRAL 动作、每集最后一步、goal-stop、迟到步、停顿步各自怎么处理，逐条写进 `docs/EVAL_STANDARD.md` 并各有测试。manifest 与 eval.json 记这个键，进协议一致性检查。
  8. 标定（Gazebo，守护进程，排 Gazebo 池的锁；这一步独占机器，起跑条件照 `run_r2.sh` 的条款 R7）：教师 K=1 与 K=3（K=3 带 `--rtf 0.5`）各 10 个 train 分区 seed，L=0.20，`next-dispatch`。
     预注册的门：迟到步 0，停顿步 0；K=3 对 K=1 的逐步生效时长、转角、巡航步位移相对差都不超过 2%；`safe_gap_s` 中位数为 0、最大不超过 0.02 s。
     另报（不进门）：每步前飞位移与转角。预期回到旧协议附近（前飞 0.25–0.27 m，转角 26° 上下）；偏离旧协议 5% 以上要写明原因。
  9. 学生参照（门过了才跑）：calswa 三个训练种子带护栏，K=1 与 K=3 各 60 集，与第二轮的六个臂逐集配对，只报告。
- 不做：改 hold、改相机频率、改 L 的取值；改 3DGS 线（它的步进由动力学对象同步推进，没有这段空档；这一句开工时先核实，核实不了就写「未核实」）。
- 提交：`feat(collect): 每步记 safe_gap_s / obs_gap_s 与插件侧生效时刻，停顿步进失败集`、`fix(stats): 协议一致性键补全`、`feat(pipeline): 透传 --action-latency`、
  `feat(collect): --action-until next-dispatch（动作持续到下一次派发，默认关）`、标定入口与结果文档。
- 2026-11-12 02:15 更新（第一波已合并，本卡开工）：
  1. 前置的分支都已进 main。W1-3 已经记了 `max_frame_gap_s` 与 `frame_gap_flag`（只记不判），第一部分在它上面接。
  2. 分工调整，避免与同时在做的卡改同一个文件：第 4 条（`tools/stats_report.py` 的协议一致性键）移给卡 W1-14；第 6 条里两条流水线的透传移给卡 W1-15b。
     本卡留 `sim/tools/collect.py`、`cf_sitl_exec.py`、`gz_io.py`、`eval_closed_loop.py`、`collect_parallel.py` 与 `tools/preprocess.py` 的过滤开关。
  3. 第 8 条的起跑条件不再照条款 R7。标定对墙钟敏感，要独占四个池（`machine.lock`、`gs3d.lock`、`train.lock`、`check.lock`）。四把锁要么同时拿到，要么都不拿：
     每分钟非阻塞地试一遍，有一把拿不到就把已经拿到的放掉（不许占着一把等另一把：那样会把别的池卡住，11-12 凌晨 Gazebo 池空转 45 分钟就是持锁空等造成的）。
     拿齐之后看 30 s 窗口内容器的 CPU，低于 3 核才起跑（十几个 agent 同时在跑单测，1.5 核等不到）；10 分钟内不满足就放掉四把锁回到轮询。
     总共最多等 8 小时，超时退出（rc=14），不在有负载的时候标定。起跑后四把锁持有到标定结束。入口写好并用桩演练过之后就可以起守护进程，不用等主会话。
     第 9 条的学生参照臂不独占，只拿 `machine.lock`，起跑条件照公共规则 §4「2026-11-12 01:40 起」。
  4. 新增一项离线核对（第一部分做完就做，不起 Gazebo）：卡 W1-5 的报告披露，11-11 23:42:29 到约 23:45 它的一个测试误起了 4 组真实评测（其中 2 组起了 gz），
     当时第二轮的 `acc-L0.20-k1-s2` 与 `acc-L0.20-k3-s0` 两个臂在跑。用第一部分的 `obs_gap_s`、停顿步与每步位移，对这两个臂按「集的起止时刻是否落在该窗口」分两组比较，
     报两组的停顿步数、`obs_gap_s` 的最大值、巡航步位移与转角的分布，给出「要不要重评这两个臂」的依据。只报告，不重评。
  5. 提交按公共规则 §5 的 11-12 四条：A1、A3、A5 各自的记账一条一个提交，`--action-until` 一个提交，标定入口一个提交。

### 卡 S1：3DGS 线的碰撞口径、条件通路与划分修正，重定基线（审核 B、C2、D1；11-12 立；等 K3 与 W1-12 合并后开工；端到端 E4 的前置）

- 依据：`docs/reviews/2026-11-12-cross-review/` 的 `review-b.md`（发现 1–7）、`review-c.md` 的 C2、`review-d.md` 的 D1、`review-e.md` 的 E-12，与同目录 `FIXLIST.md`。
- 要改的（都带版本键；缺省保持现状逐位不变，新实验的入口显式选新口径，`metric_version` 升版）：
  1. 膨胀核：结构元素按「格中心到原始体素盒的欧氏距离不超过机体半径」取，补 0.1 m 体素下斜向的测试（B1）。
  2. 高度口径：碰撞判定、教师规划的硬栅格、规则臂的列距离、出生检查统一成一个函数：z ±（机体半高 + 高度余量）覆盖到的层取并（B2、B4、B7）。高度余量先量后定（稳态飞行 z 的 p1–p99）。
  3. 每步碰撞沿 SITL 窗口内的轨迹点逐段扫，不只扫首尾的弦（B3）。
  4. 渲染深度与占据的门限：先核实 garden 里「深度读到占据里没有的障碍」是不是由 opacity 在 0.05–0.3 之间的高斯造成（按门限过滤后重渲同一批位姿比较），核实了再定对齐方式（B5）。
  5. 室内场景选了 `cf_blackbox` 动力学时报错（B6）。
  6. 学生适配器显式传条件值：实际执行的上一步动作或条件码，DAgger 的教师替换与护栏改写之后都要更新；加测试（C2）。
  7. `eval_3dgs` 不再把 split 写死成 test：按场景与路线的划分文件核对，对不上报错；划分文件写明哪些场景、哪些路线是留出的（D1）。
  8. 3DGS 实验读路线目录的通配排除 `*.incomplete-*`（E-12）。
- 重定基线（3DGS 池，保留每集的轨迹文件）：室内 = K1 的 40 条 × 3 集加 K3 的留出路线 × 3 集，室外 = 4 个场景各 6 条 × 5 集，seed 与原实验相同。
  臂：教师（容差 18.9°；W1-12 合并了就再加 `swept`）、盲前进加路线方位、规则 v1（occ）、规则 v2 的选中变体（occ 与 depth）。每个臂与原实验逐集配对，报新旧口径下的到点、撞在前、超时。
  起跑前写下的预期：规则臂在新口径下的到点成功数变化不超过 3/120（主会话已按精确欧氏距离重判过 K3 的已录轨迹，0 集翻转）；教师的变化方向不预测，照实报。
  规则 v2 的到点成功下降超过 6/120，就说明它的窄通道（半宽 0.15 m）依赖了旧口径的宽松，E4 的教师要改用半宽 0.25 m 的变体。
- 2026-11-12 02:15 细化（K3 与 W1-12 已合并）：拆成两张卡并行，文件分开。
  - **S1a（条件通路与划分；小）**：第 6、7 条。
    - 第 6 条：`sim/3dgs/collect_3dgs.py` 的 `StudentAdapter`、`DaggerPolicy` 与 `sim/3dgs/guard.py` 的改写路径。要求：喂给学生的「上一步动作」是上一步实际执行的动作；
      DAgger 换成教师动作、护栏改写之后的下一步都要反映出来；`manifest` 记喂的是哪一种条件。审核 F 写好的用例 `$D/tmp/phase12-cards/review-f/proposed/test_3dgs_student_prev_action.py`
      在 main 上 2/2 红，随修复入库并变绿。纯图像模型（`prev_dim = 0`）的行为逐位不变，用测试锁住。航向码在 3DGS 线怎么定义不在这张卡里（留给 E4 的卡）。
    - 第 7 条：照审核 D1 的最小修法。`eval_3dgs` 读场景身份里按哈希算出的 split，不是 test 时默认拒绝；显式 `--allow-train-scene` 才放行，并把 `split_policy` 写成 `diag-train-scene`，
      产物里不出现 `split = "test"`。`experiments/2026-11-10-bearing-guard-3dgs/driver.py` 是 K1、K3、W1-12 共用的驱动，它自己写 `split="test"`、`split_policy="diag-eval-fixed-test"`：
      加一个 config 键让新实验写真实的 split 与 `diag-train-scene`，缺省不变（旧实验的配置能逐位复现）。报告里列出 9 个评测场景各自的哈希 split（审核 D 的 `scene_splits.py` 可复用）。
    - 不碰：`collision.py`、`planner.py`、`rule_oracle.py`、`dynamics.py`，以及 `collect_3dgs.py` 里每步碰撞判定的那一段（S1b 的）。
  - **S1b（碰撞口径与重定基线；大）**：第 1–5、8 条与重定基线。
    - 先复现：跑审核 B 草稿目录 `$D/tmp/phase12-cards/review-b/` 的 `kernel_replay.py`、`z_layers.py`、`trace_vs_chord.py`，数字与报告对上再动代码；对不上停下报告。
    - 口径用一个键选：`collision_model`，`v1` = 现状（缺省，逐位不变），`v2` = 第 1、2、3 条都生效。这个键进 `manifest`、`metrics.json` 与 `done.json`，`metric_version` 随之升版；
      教师规划、出生检查、规则臂的列距离在 `v2` 下用同一个高度口径函数（W1-12 的 `spawn_two_layer` 与 `SPAWN_CRUISE_DZ_M` 在 `v2` 下由这个函数取代，`v1` 下保留）。
    - 第 8 条：写一个读路线目录的公共函数（排除 `*.incomplete-*`），本卡的汇总脚本用它；已经跑完的实验的汇总脚本不改，只在报告里列出哪些脚本有这个通配。
    - 重定基线照上面一段，加两点：臂里的教师再加 `swept`（W1-12 已合并）；每个臂同时在 `v1` 与 `v2` 下各跑一遍并逐集配对，`v1` 的结果要与原实验逐集一致（不一致先查原因）。
    - 不碰：`StudentAdapter`、`DaggerPolicy`、`guard.py`、`eval_3dgs.py` 的 split 逻辑（S1a 的）。
  - 两张卡都改 `collect_3dgs.py`，各自只动上面写的那一段；合并时的冲突由主会话处理。提交按公共规则 §5 的 11-12 四条，一条发现一个提交。

### 卡 W1-14：统计口径（审核 D2、D4、D5、D7、A4 与卡 W1-2 的遗留；11-12 02:15 细化，可以开工）

- 依据：`docs/reviews/2026-11-12-cross-review/review-d.md` 的 D2、D4、D5、D7，`review-a.md` 的发现 4（口径混表），卡 W1-2 报告的「需要主会话定的事」第 1–4 条（`$D/tmp/phase12-cards/reports/w1-2.md`）。
  审核 D 的复现脚本在 `$D/tmp/phase12-cards/review-d/`（`cluster_stats.py`、`sr_multirun.py`）。
- 只动：`tools/stats_report.py`、`tools/test_stats_report.py`、新文件 `tools/paired_stats.py` 与它的测试、`docs/EVAL_STANDARD.md` §5。不起 Gazebo，不用 GPU。
- 要做的（一条一个提交）：
  1. D2：新文件 `tools/paired_stats.py`（纯标准库）：按簇的符号翻转精确检验（每个几何把各训练种子的成败差加总，再对几何做符号翻转；簇数不超过 20 时枚举，超过时用固定种子的蒙特卡洛并写明次数）、
     按簇重抽样的 bootstrap 区间、精确 McNemar、Wilson。`stats_report` 的配对表在多个 run 共用几何时加这两列，逐集 McNemar 保留作参考。簇键用 `per_episode[*].geometry_gid`，没有时退回 seed 并在表注里写明。
  2. D4：多 run 的格，区间改成按几何整簇重抽样（现在只在 run 维度重抽样，3 个 run 的区间偏窄）；多 run 组也出配对检验。
  3. D5：按 seed 作键互相覆盖的地方改成按（run，seed）；同一个 run 里 seed 重复时报错。
  4. 卡 W1-2 的遗留：配对检验里一方基础设施失败的 seed 按失败记 0，与主表分母同口径；全部失败的格显示 `0.00 (n=请求数)`。
  5. A4：协议一致性键加 `action_latency_s`、`heading_guard`、`rtf`、`student_cond`、`dynamics_profile`（给了 `--cf-sitl-params` 时）。缺键等于旧口径；同一张表里这些键不一致就报错并列出是哪几个键，
     `--allow-mixed-protocol` 才放行，放行时表头标「混合协议」。加测试：两份只差一个键的 eval.json 报错。
  6. D7：`paired_stats.py` 里给一个取成功数与分母的函数（请求集数、完成数、失败数），新实验的汇总脚本用它；已经跑完的实验的汇总脚本不改。
  7. `docs/EVAL_STANDARD.md`：§5 开头列表、§5.3 指标表与「统计方法」第一条按新分母直接改写（不再靠 §5.6 的声明盖住）；§5.6 加按几何聚类的检验；
     「同权重同 seed 重评的逐集不一致率约 24%」这个数用现有数据核实（找两次配置完全相同的评测），核实不了就改成「未核实」并写明查过哪些目录。
- 验收：对 `$D/exp-swa-calib-20261110/` 与 `$D/exp-heading-guard-20261110/` 重算，按几何的符号翻转 p 值与审核 D 的表一致（calswa 对 best 0.0887，calbest 对 best 0.1390，
  guard2-best 对 best 0.0034，calswa 对 calbest 0.672）。
  2026-11-12 07:30 更正（执行者查明）：原来写的 calswa 对 calbest 0.664 是 60 集逐集精确 McNemar 的数，按几何的符号翻转 p 是 0.672；前三个数是符号翻转 p。两个数都报。没有多 run、没有失败集、协议键齐全且一致的旧输入，报表的 `cells` 与 `paired` 逐项不变（用现有 eval.json 实测并写进报告）。
- 不做：改各实验目录里已有的 `summarize.py`；改 `tools/dagger_pipeline.py` 的分母（在卡 W1-15b）。

### 卡 W1-15：入口脚本与流水线的健壮性（审核 E、A2 与卡 W1-1、W1-5 的遗留；11-12 02:15 细化成两张，可以开工）

- 依据：`docs/reviews/2026-11-12-cross-review/review-e.md`（E-1 到 E-13，每条有位置与最小修法）；公共规则 §6 的 11-12 五条与 §4「2026-11-12 01:40 起」；
  卡 W1-1、W1-5 报告的「需要主会话定的事」（`$D/tmp/phase12-cards/reports/w1-1.md`、`w1-5.md`）。
- 为什么拆出公共脚本：11-11 夜里每张卡的入口都是把 `run_r2.sh`（约 400 行）抄一份再改，条款 R7 的「容器低于 1.5 核才起跑」被一起抄走。分池并行之后这个条件不会满足，
  Gazebo 池 11-12 00:56 到 01:40 空转了 45 分钟。锁、冻结代码、起跑记录、起跑条件、完成标记这几样收到一处并用测试锁住，新入口只写自己的臂。
- **W1-15a（实验入口的公共脚本）**。只动：新文件 `tools/exp_lib.sh`、`tools/stop_daemon.sh`、`tools/test_exp_lib.py`、`experiments/_template/run.sh` 与 `README.md`，以及 `tools/check.sh` 里 E-13 的两行。
  1. `tools/exp_lib.sh`（bash，被入口脚本 `source`）提供：实例锁（先拿锁，再写任何状态文件；E-5）；`run.pid` 记 PID、PGID 与启动时刻（E-4）；起跑记录只追加（HEAD、输入文件的 sha256、
     被环境变量覆盖过的配置项逐个列出；E-2、E-10、E-11）；代码冻结到 `$OUT/code`（`git archive`，续跑时核对冻结副本与起跑记录一致，之后只从副本里跑）；
     拿池的锁（`gazebo`、`gs3d`、`train` 三种，阻塞等待并按间隔记日志），Gazebo 池带公共规则 §4「01:40 起」的起跑条件（门槛与最长等待是配置项，缺省 8 核、600 s，到点照样起跑并记下核数）；
     每个臂起止的 cgroup 快照（`usage_usec`、`nr_throttled`、`throttled_usec`）与一行健康摘要（平均核数、限流时长占墙钟的比例）；半成品目录改名 `*.partial-<时刻>` 后重做（E-9）；
     `finish <rc> <说明>` 落 DONE，rc 反映臂的失败（E-6）；被信号终止时落 rc=130 的 DONE。
  2. `tools/stop_daemon.sh <输出目录>`：读 `run.pid`，核对该 PID 的启动时刻与记录一致、命令行是入口脚本，才 `kill -- -<PGID>`；对不上就说明守护进程已经不在，报告并退出，不 kill（E-4）。
  3. `experiments/_template/run.sh`：不超过 80 行的样板，配置区在顶部，演示上面每一项怎么用；`README.md` 写新实验怎么起步。
  4. `tools/test_exp_lib.py`：用假的臂命令（`sleep`、`false`）在临时目录里演练：重复起实例被拒、续跑跳过已完成的臂、半成品改名、覆盖变量进起跑记录、臂失败时 DONE 的 rc 非零、
     被 SIGTERM 时落 DONE、起跑条件到点照起、容器重启后的旧 `run.pid`（伪造一个启动时刻对不上的）不被 `stop_daemon.sh` 杀。每条先红后绿。
  5. E-13：`tools/check.sh` 的临时目录跟 `TMPDIR` 走，并设 `PYTHONDONTWRITEBYTECODE=1`。
  - 不做：改任何已有实验的入口脚本（冻结记录）；改公共规则（把要改的句子写在报告里，主会话改）。
- **W1-15b（三条 Python 流水线）**。只动：`tools/e1_pipeline.py`、`tools/dagger_gz_pipeline.py`、`tools/dagger_pipeline.py`、`tools/pipeline_common.py`、它们的测试、
  `tools/compare_datasets.py`、`tools/train_student_with_video.py`。
  1. E-3：三条流水线起跑时对输出目录加锁，重复实例直接退出。
  2. E-8：「评完」的判据收成 `pipeline_common` 里的一个函数：被跳过或截短的评测不算评完；已有 eval.json 的 `metric_version` 或权重路径与本次配置不一致时也不算（卡 W1-1 第 3 条、W1-5 第 5 条），
     挪到 `<目录>.stale-<时刻>` 后重评，并在汇总里列出。
  3. E-9：采集与训练的半成品目录改名 `*.partial-<时刻>` 后重做，原命令能续跑，不用人手删。
  4. E-7：`dagger_pipeline` 的成功率分母改请求集数。
  5. A2：`e1_pipeline` 与 `dagger_gz_pipeline` 的 `CFG` 加 `action_latency`、`rtf` 两项（缺省 None，命令行逐字不变），给了就透传给采集与评测。
  6. 卡 W1-1 第 2 条、W1-5 第 4 条：`compare_datasets.py` 与 `train_student_with_video.py` 起的柱阵评测带走廊判据（同名环境变量，设 0 回旧口径）。先 grep 这两个工具近期有没有被用过；
     没人用就只在它们的输出里加一行「旧口径，未带走廊」的提示，不扩功能，并在报告里写明依据。
  7. E-4：流水线的 `run.pid` 同时记启动时刻。
  - 不做：把两条流水线里重复的 `deliver_check` / `deliver_ckpt` 合并（不在审核发现里；要合并另立卡）。
- 两张卡都按公共规则 §5 的 11-12 四条提交，一条发现一个提交。

### 卡 T1：把交叉审核 F 提议的用例入库（只加测试；11-12 立，立即开工）

- 依据：`docs/reviews/2026-11-12-cross-review/review-f.md` 的 F2–F8 与第 3 节「最要紧的 5 个函数与缺的用例」；审核者写好的提议用例在 `$D/tmp/phase12-cards/review-f/proposed/`
  （`test_preprocess_label_alignment.py`、`cpp_cases.cpp`、`test_3dgs_student_prev_action.py`），它的变异脚本在同目录上一级（`mutate.py`、`check_proposed.py`）。
- 为什么现在做：今晚有三个在途分支在改 `tools/preprocess.py`（W1-7、W1-8、W1-9）。审核 F 证明了把标签改成取执行动作、让帧号错一位，现有的 `--python-only` 门禁都抓不到。
  这张卡的测试要赶在那三个分支合并之前进 main。
- 要做的（只加测试；生产代码一行不改，例外只有 `tools/check.sh` 的那个负例与 CMake 里登记新测试）：
  1. 打包的标签来源与图像对齐（F3）：新测试文件 `tools/test_preprocess_label_alignment.py`。合成集里教师标签与执行动作要不同、每一步的图像内容要能认出步号。
     验收：在 HEAD 上通过；在草稿副本里把 `preprocess.py` 的标签改成取 `action_executed`、把帧号错一位，各自变红。
  2. 部署推理路径（F2）：`run_infer_loop` 的 HWC→CHW、双帧顺序（先当前帧、后上一帧）、条件字节，用非方形输入与手算的参考值断言。把推理端的两帧对调、把下标写成转置时要变红。
  3. 水平翻转增强（F4）：翻转样本的标签左右互换、条件列 2 与 3 互换、深度目标按列镜像，各有带参考值的断言。
  4. 双帧训练读的是 `prev_index` 那一帧（F5）：构造「标签只由上一帧决定」的小数据集，读错帧时学不会。
  5. `check.sh` 的几何划分负例（F7）：现在的夹具被 split 复算那道门先挡下。改夹具，让它走到「同一个几何出现在两个 split」那一行；原来那道门另留一个负例。
  6. 闭环判据的边界（F8）：走廊半宽上的等号、过线的等号，各锁一个用例。
  7. `sim/3dgs` 的那个提议用例（F1）在 HEAD 上是红的，不放进这张卡：把它在 HEAD 上的失败输出记进报告，用例留给卡 S1 随修复入库。
- 新写的测试在 HEAD 上变红，说明测到了真问题：停下，把现象写进报告交回，不要去改生产代码。
- C++ 测试要编译：构建目录用 `$D/tmp/build-p12-t1`（数据盘），只编只跑新增的测试与受影响的目标，`UAV_RGB_BUILD_JOBS` 不超过 4。全量 `check.sh` 由主会话合并时跑。
- 每个新测试都做变异检查（人为改错被测代码的副本，看测试是否变红），结果写进报告。
- 提交：`test(preprocess): 标签来源与图像对齐的回归测试`、`test(infer): 推理输入布局与双帧顺序的参考值测试`、`test(train): 翻转增强与双帧读帧`、`fix(check): 几何划分负例走到它要测的那道门`、`test(eval): 走廊与过线判据的边界`。

### 卡 E2E-C1：条件通路进契约（C++；审核 C3–C6、C8、C9、G7；规格初稿；等 E2E-1 出结果后开工）

- checkpoint 升版：记条件输入的种类（上一步动作、按步数的航向码、按角度的航向码）与维度；推理进程的握手行带出来，客户端核对，对不上拒绝启动。条件维度可配（现在写死成 4），翻转增强的镜像规则跟着条件的种类走。
- 握手行与 checkpoint 带前处理契约：输入的视场、裁剪、缩放方式（G7）；辅助头的分带语义与文本词表的指纹（C8）。
- 深度辅助头可选接在 FiLM 之前（C5）；`--eval-only` 读不到训练配置时不再静默回落（C6）；双帧顺序与条件字节加「训练路径等于推理路径」的回归测试（C9）。
- 结构上的选择（辅助头接在哪）先在 PyTorch 原型线上比过再移植。
- 11-12 09:13 加（第二轮审核 N-3、N-5、L-3、I1-6 第 3 点）：权重升版时记条件类型（无、上一步动作、按步数的航向码、按角度的航向码）与合法取值的掩码，推理端收到不合法的码当场拒收（航向码模型没见过码 1，喂进去会把 14%–22% 的动作翻掉）；握手行输出条件类型；`sim/tools/student.py` 在模型带条件而调用方没给条件时报错，去掉回落到自己上一步输出的分支；
  条件维度可配（训练器 `--cond-dim`，缺省同现在；侧车校验、`student.py`、`average_ckpt.py` 都按权重头里的维度走）；推理的行协议加「新的一集开始」的标志，预热帧之后自动复位；
  `eval_closed_loop.py --student-cond` 的缺省值改成空，显式给 `prev-action` 而模型是纯图像时判对不上。

### 卡 C1a：训练器的小修与数值测试（C++；审核 H8、H9、C6 与上一轮 `src.md` 的 D2、D3、D6、D9；11-12 02:15 立；等 T1 合并后开工）

- 为什么先做这一部分：不动 checkpoint 格式，不与在途的 E2E-1、E2E-P0 冲突；E2E-R1 正在用 `--cosine-lr`，而学习率调度没有数值测试。
- 要做的（缺省行为逐位不变；一条一个提交）：
  1. D3：warmup、余弦退火、AdamW 的数值测试（给定步数与基础学习率，逐步的学习率与手算值相等；AdamW 的一步更新与手算值相等）。
  2. H8：增强的随机数种子随 `--seed` 变（现在写死，三个训练种子看到同一串增强参数）。不开增强时逐位不变；开增强时的旧行为不保留，在报告里写明。
  3. H9（即上一轮 D2）：`--log-grad-norm`（默认关）把每轮裁剪前的梯度范数的中位数与最大值、被裁剪的步数占比写进 `metrics.jsonl`。
  4. D6：续训时继承 `--max-grad-norm`。
  5. C6：`--eval-only` 读不到训练时的碰撞过滤口径时报错，不再静默回落到命令行缺省。
  6. D9：卷积反传的 dx 做有限差分检查。
  7. 训练进程的 CPU 占用：`uavrgb_train` 每个进程占满 1 个核（11-12 实测 99%），其中多少是在自旋等 GPU 先量清楚；是自旋的话加一个开关（环境变量或命令行，默认关）让 CUDA 用阻塞等待，
     报开关前后的 CPU 占用与每轮用时。机器只有 12 核配额，Gazebo 池吃 CPU，训练池让出的核直接换成评测的吞吐。
- 不做：改 checkpoint 格式、条件维度、辅助头位置（在卡 E2E-C1）；末批不满照常更新（上一轮 D5，会改变缺省训练轨迹，归到配方探索）。
- 构建目录 `$D/tmp/build-p12-c1a`，`UAV_RGB_BUILD_JOBS` 不超过 2。

## 4b. 端到端线的任务卡（11-12 立；背景与路线见计划 §10）

端到端的定义：部署时只有一个网络。输入是机载图像、指令 token、由自己已执行的动作算出的历史码（不用仿真真值）；输出是 4 个动作。
评测主表不带航向护栏、不带路线护栏、不带规则层；带护栏的数字只当参照列。

### 卡 E2E-0：可判定性审计（离线分析；不起仿真，不占任何池的锁）

- 回答的问题：特权 A* 教师的动作标签里，有多少能由端到端学生拿得到的信息决定；补哪一种输入能把缺口补上。结果用来定 E2E-1 之后条件码该带什么（计划 §10.4 的 E2）。
  已有的旁证：加评测端航向护栏后越界 23/60 → 0；DAgger 聚合臂 1/20；只截掉「过线后朝看不见的目标回正」的帧，成功 10 → 13/20。
- 先读：`sim/tools/rule_policy.py` 的 `column_distances`（真值列距离，角度约定在文件头）；`sim/tools/collect.py` 每步写进 `actions.jsonl` 与 `oracle.jsonl` 的字段
  （动作前位姿、`action_expert`、`action_executed`、`executed_by`）与 `episode_manifest.json` 的 `pillars`；`tools/preprocess.py` 的 `load_episode()`（哪些步进训练集、截断与丢碰撞的规则）；
  `sim/tools/split_manifest.py`（按几何分组的 split）；`docs/reviews/2026-11-10/data-algo.md` §A（上一轮对标签的分析，已有的结论不重复做）。
- 数据（都是现成的采集，只读）：示范 = `$D/tmp/agent6-distill/trunc/col`（参照学生的训练数据）与 `$D/exp-dagger-gz-20261109/full-1109/more_demos/collect`；
  学生驱动的状态 = `$D/exp-dagger-gz-20261109/full-1109/round_{1,2,3}/collect`（影子教师给的标签）。进训练集的步按 `preprocess` 的同一套规则取，不另立口径。
- 特征（都由仿真真值算，只用于分析，名字带 oracle；一步一行）：
  `d`：动作前位姿上的 20 列距离（60° 视场，上限 4 m），代表「一个完美的视觉前端能从当前这一帧读出的几何」；
  `n`：这一步之前已执行的左转数减右转数，截到 ±4；
  `hist`：上一步执行的动作、最近一次转向的方向、距最近一次转向过了几步（封顶 5）；
  `d_wide`：同一位姿上 180° 视场的 60 列距离，代表「对周围障碍的完美短期记忆」；
  `priv`：横向位置 y、到最远柱的纵向距离（学生拿不到，当作参照的上界）。
  特征组：S1 = {d}；S2 = {d, 上一步动作}；S3 = {d, n}；S4 = {d, n, hist}；S5 = {d_wide, n, hist}；S6 = S5 加 priv。
- 做法：每个特征组拟合一个小分类器预测教师动作（两层各 64 个单元的 MLP，类别加权交叉熵；用 `$D/teacher-venv` 的 torch，这是分析工具，不进主线；
  或者纯 numpy 实现，二者取简单的）。按几何分组划分：train 拟合，val 定停止轮数，test 报数；每组 3 个随机种子，报均值与极差。
  另做一个不依赖拟合器的核对：把特征粗分箱后查「同一个箱里教师动作的多数占比」，与 MLP 的准确率对一下量级。
- 报告的数（分三个子集：教师驱动的帧、学生驱动的帧、画面里 4 m 内没有障碍的帧）：整体准确率、macro-F1、每类召回；「该转的时候转没转」与「转向帧里左右对不对」分开报。
- 起跑前写下的预期（不是门槛，照实对照）：学生驱动的帧上，S1 的左右正确率不高于 0.70，S3 不低于 0.85；教师驱动的帧上 S1 与 S3 的整体准确率相差不到 5 个点；
  S5 比 S4 高不到 5 个点。S5 比 S4 高 10 个点以上，就说明端到端学生需要学出来的记忆，历史码不够。
- 入口 `experiments/2026-11-12-decidability-audit/`（脚本、测试、README、结果表），输出 `$D/exp-decidability-audit-20261112/`。特征提取要有测试（合成两根柱的场景上列距离、n、hist 的值）。
  预计一小时以内的 CPU 活，守护进程跑，不拿锁；拟合可以用 GPU（显存很小）。
- 提交：`feat(exp): 可判定性审计（真值列距离与动作历史能决定多少教师标签）`。

### 卡 E2E-1：航向码端到端学生（卡 W1-9 的实现，加 2×2 试点；不带任何评测端规则）

- 回答的问题：把「自己已经偏了多少」作为输入交给网络之后，端到端学生不靠护栏能到多少；DAgger 聚合数据的塌陷（在线到线率 87 → 40 → 1）是不是因为标签在图像上不可判定。
- 第一步：完成卡 W1-9 的全部改动、测试与提交，分支 `p12-w1-9`（工作树已建）。
- 第二步（分支 `p12-e2e-1`，从 `p12-w1-9` 的 HEAD 起；内容就是卡 W2-2 的第零步）：`tools/average_ckpt.py` 的校准支持带条件输入的模型（ckpt 的 `prev_dim > 0`）：
  校准推理时从数据集侧车读每帧的条件列，按 `sim/tools/student.py` 的协议随帧发给 `uavrgb_infer`。加测试；
  用一个 `--prev-action` 训出的小模型核对工具算的预测与 `uavrgb_train --eval-only` 逐帧一致。不带条件的模型的输出逐位不变（W1-5 的回归值可以直接用来核对）。
- 第三步：实验。入口 `experiments/2026-11-12-e2e-heading-code/`，输出 `$D/exp-e2e-heading-code-20261112/`。
  数据两份，都是现成的采集，不重采：
  `demo` = `$D/tmp/agent6-distill/trunc/col`（截断的教师示范；现有数据集 `…/trunc/ds/dataset.uavr`，打包参数照 `…/trunc/preprocess.log`）；
  `dagger` = DAgger 第 3 轮训练用的聚合集（592 集、20,291 样本；现有数据集 `$D/exp-dagger-gz-20261109/full-1109/round_3/ds/dataset.uavr`，
  集清单与打包参数照 `round_3/ds.log` 与 `tools/dagger_gz_pipeline.py` 复原）。
  先核对复原得对不对：不带新开关重打一遍，样本数、各 split 的样本数、标签直方图与现有数据集一致（逐字节相同最好），对不上就停下报告。
  四个臂，每臂 3 个训练种子（0、1、2）：

  | 臂 | 训练数据 | 输入 | 权重从哪来 |
  | --- | --- | --- | --- |
  | A0 `img/demo` | demo | 图像 | 现成的 `$D/exp-swa-calib-20261110/ckpt/cal_swa_s{0,1,2}_k10.bin`，不重训 |
  | A1 `code/demo` | demo，`--cond heading-code` 重打包 | 图像 + 航向码 | 新训，`--prev-action` |
  | A2 `code/dagger` | dagger，`--cond heading-code` 重打包 | 图像 + 航向码 | 新训，`--prev-action` |
  | A3 `img/dagger` | dagger（现有数据集） | 图像 | 新训（11-09 的塌陷臂只有 1 个训练种子，这里补齐 3 个） |

  训练配方与 A0 相同（照 `$D/exp-student-select-20261109/e1-noise/seed0/run/config.json`：30 轮、batch 128、lr 0.001、类权重 auto、`--drop-collision`），
  加 `--save-every-epoch`；交付权重都是末 10 轮平均加校准（`tools/average_ckpt.py`，A1、A2 用第二步的扩展）。训练排训练池的锁，3 个种子可以同时跑，单个约 7 分钟。
  评测（Gazebo，守护进程，排 Gazebo 池的锁，2 条车道）：新协议，命令行照 `experiments/2026-11-10-action-latency/run_r2.sh` 的 `acc_job`，去掉 `--heading-guard`；
  `--action-latency` 取 `$D/exp-action-latency-r2-20261111/L_STAR`；`--task base --steps 80`；20 个冻结 seed；A1、A2 加 `--student-cond heading-code`。
  12 个评测臂共 240 集。含迟到步的集按 `run_r2.sh` 条款 R6 重跑一次。评测顺序按训练种子轮转（先四个臂的种子 0，再种子 1，再种子 2），中途被打断时四个臂的进度相同。
  推理二进制用 `$D/bin-snapshots/main-97ae04b/uavrgb_infer`；它不支持条件字节的话停下报告，不要自己换二进制。
- 预注册（主会话 11-12 定，写进入口脚本头，不许改；成功 = 走廊内成功，口径同第二轮）。三个主比较，各 60 对逐集配对，精确 McNemar：
  M1：A2 对 A3（同一份聚合数据，有无航向码）。预测：A2 的成功数比 A3 多 15 集以上，p < 0.01。证伪：多不到 6 集。
  含义：预测成立，则 DAgger 的塌陷来自标签不可判定，补上航向码后学生自己走到的状态可以拿来学；证伪成立，则航向不是塌陷的主因，或者这个码带的信息不够。
  M2：A1 对 A0（同一份示范数据，有无航向码）。预测：A1 的越界集数不超过 A0 的一半，成功数不低于 A0。证伪：A1 的越界集数比 A0 少不到 3 集。
  M3：A2 对 A0。预测：A2 的成功数比 A0 多 6 集以上。证伪：A2 不高于 A0。
  只报告、不设判决的一条：航向码臂里成功数高的那个，是否达到卡 W2-2 的判据（不带护栏成功不低于 48/60、越界不超过 3/60）。带护栏的参照是第二轮的 K=1 三个臂（53/60）。
- 另报：每臂的撞柱、越界、未到线；三个训练种子各自的数；动作占比；每步生效时长与迟到步；
  离线反事实探针（val 上）：把码 2 与码 3 对调，转向帧的预测方向翻转的比例；把码恒置 0，macro-F1 的变化。这两个数说明网络有没有在用这个码。
- 11-12 01:10 补充（交叉审核 D2、D3、A2、C1；写在评测数据出来之前，由主会话在审报告时执行，执行者已经写好的入口不用改）：
  1. 三个主比较另报按几何聚类的符号翻转精确检验（20 个几何，每个几何把三个训练种子的成败差加总）。判决仍按预注册的集数阈值；聚类检验的 p 值与逐集 McNemar 并列，结论不一致时照实写。
  2. 这 20 个 seed 对航向码的阈值来说是样本内：阈值 2 沿用护栏，护栏是看着这批 seed 的轨迹定的。胜出的航向码臂与 A0 在另外 40 个 test 分区几何上各补一遍，作为样本外的数。
  3. 报告的局限一节写明：训练数据是旧协议与旧动力学库，评测是新协议（每步多 0.1 s 保持，位移小 17%）。四个臂受同样的偏移，臂间比较公平；绝对数不能与旧协议的数直接比。
  4. 另报 A1、A2「只平均不校准」的离线指标，看单一校准量对带条件的模型有没有副作用。
- 11-12 09:09 补充（第二轮审核 M-4、M-6、M-1，报告 `docs/reviews/2026-11-12-review-round2/review2-m.md`；写在闭环开始之前：此刻训练池一侧已完成，闭环一集都还没跑，Gazebo 池在跑 E2E-R1 的最后两个臂。
  离线报告 `offline.md` 主会话已经看过，闭环数据没有。冻结在输出目录里的代码不改，下面几条由收尾的执行者用单独的脚本做，脚本调 `tools/paired_stats.py`）：
  1. 判决以冻结的汇总器为准（M1 是逐集精确 McNemar 加集数阈值，M2、M3 是集数阈值），这一点不变。
  2. 聚类检验的做法定死：配对单元是 60 个（训练种子，评测 seed）对；簇是 20 个 `geometry_gid`（先核对与 20 个 seed 一一对应）；用 `paired_stats.pair_summary`，报双侧符号翻转精确 p 与整簇 bootstrap 的 95% 区间。
     三个主比较的成功数之差都报。
  3. 聚类检验按 α = 0.05（双侧）读，不照搬 M1 的 0.01。理由：只有 20 个簇时，双侧符号翻转要有 8 个以上非零差的几何全部同向才能到 p < 0.01（7 个全同向是 0.0156）；0.01 是给逐集检验定的。
     写结论时的四种情形：预注册判「预测成立」且聚类 p < 0.05，写「成立，按几何聚类也显著」；预注册判「预测成立」而聚类 p ≥ 0.05，写「逐集检验下成立，按几何聚类不显著，效应集中在少数几何」，
     这种情形在另外 40 个几何的样本外复核出来之前不当成已经立住的结论引用；其余两种（证伪、不判）照预注册写，聚类的数并列。
  4. M2 的越界集数只在两个臂都干净完成的单元上比（任一侧是基础设施失败的单元两侧一起剔除：重跑后仍迟到、停顿步、采集进程崩溃），剔了几个单元照实报。没有剔除时就是冻结汇总器的数；有剔除时两种算法都报，判决按剔除后的。
  5. 成功数的分母恒为每臂 60：`failed[]` 里的集（含不是迟到的崩溃集）一律记失败。冻结的汇总器遇到崩溃集会少算分母并且拒绝出判决（M-6）；发生时用单独的脚本按同样的阈值重算，并在报告里写明是重算的。
  6. 校准的混杂（M-1：对 A1 的三个权重，全局校准量是 +1.73 到 +1.89 nat，val 上码非 0 的 15 帧里朝回转的预测从 12、13、10 帧变成 9、6、10 帧；样本很小，方向与第一轮 H1 的推断一致）：
     M2 照预注册跑、照预注册判。但 M2 判「证伪」或「不判」时，不下「示范数据上加航向码没有用」的结论，等卡 E2E-1b 的按码校准臂。
     M3（A2 对 A0）同时差了数据、航向码与工作点（两份 val 的前进占比目标 0.734 对 0.873），报告里写明这三样分不开。
     离线部分补一张表：A1、A2、A3 的「只平均不校准、全局校准、只按码 0 校准」三种权重在 val 与 test 上分码（码 0、码非 0）的指标（即 01:10 补充的第 4 条，离线报告里漏了）。
- 不做：不带护栏之外的评测端规则；3 块课程（等这张卡的结果）；改网络结构；改 C++。
- 并行改动的提醒：今晚别的卡也在改 `tools/preprocess.py`（W1-7、W1-8）、`sim/tools/collect.py` 的学生分支与 `sim/tools/eval_closed_loop.py`（W1-3、W1-4）、
  两条流水线（W1-1、W1-5）。只动卡里写的地方，改动尽量集中在新函数与新参数上，合并冲突由主会话处理。
- 提交：W1-9 的提交照该卡；`feat(tools): average_ckpt 的校准支持带条件输入的模型`；实验入口与结果文档。

### 卡 E2E-P0：PyTorch 原型线的地基（用户 11-12 允许结构探索用 PyTorch 做原型）

- 目的：以后要试循环状态、更省参数的卷积这类结构，先在 PyTorch 里试，闭环评测走 `--infer-bin` 包装器，结构定了再移植到 C++。
  这张卡只搭地基，并证明「原型线的数与主线的数可比」。不做任何结构探索。
- 规矩（公共规则 §4）：原型代码只放新目录 `proto/`；`src/`、`tools/`、`sim/` 的主线代码不 import 它；原型的权重与数字都标 `proto`，不进主表。
  解释器用 `$D/teacher-venv/bin/python`（只读借用，不往里装包；缺包就停下报告）。
- 先读：`src/include/uavrgb/model.hpp` 与 `src/model.cpp`（前向的每一步：卷积的填充、池化、FiLM 的公式 y' = (1+γ)·y + β、展平的顺序）；`src/checkpoint.cpp`（checkpoint 的二进制格式与版本）；
  `src/include/uavrgb/infer.hpp`、`src/infer_main.cpp`、`sim/tools/student.py`（行协议：结构头一行、每帧一行 JSON、双帧与 prev-action 字节的顺序）；
  `sim/tools/baselines/baseline_infer.py` 与同目录 README（现成的 PyTorch 包装器与启动器写法）；`tools/DATASET_FORMAT.md`、`tools/read_dataset.cpp`、`tools/select_ckpt.py`（UAVR 与侧车的读法）；
  `src/trainer*.cpp`、`src/class_weights.cpp`（训练配方：Adam 的超参、类权重 auto 的算法、丢碰撞帧、`ckpt_best` 按什么选、每轮存档）；`teacher/dataset.py`（已有的 PyTorch 数据读取，能复用就复用）。
- 第一步，前向对拍：`proto/model.py` 复刻 `CnnModel`（含 FiLM 的文本段与 prev-action 段、辅助头、双帧）；`proto/ckpt_io.py` 读写 C++ 的 checkpoint。
  验收：载入 `$D/exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin`，在它训练数据集的 val 分区取 2,000 帧，四类概率与 `uavrgb_infer` 的输出逐帧相差不超过 1e-5；
  写出再读回的 checkpoint 与原文件逐字节相同。带 prev-action 的模型同样对拍（卡 E2E-1 的权重出来之前，用 `uavrgb_train --prev-action` 现训一个 1 轮的小模型）。
- 第二步，包装器：`proto/infer_wrap.py` 说 `uavrgb_infer` 的行协议，加启动器。验收：载入同一个 C++ checkpoint，离线回放一集的帧序列，动作序列与 `uavrgb_infer` 逐步相同；
  Gazebo 里用 calswa 种子 1 跑 3 个冻结 seed（排 Gazebo 池的锁，新协议，带护栏），与 W1-0 第二轮 `acc-L0.20-k1-s1` 的同 seed 集对照动作序列，
  有分歧就写明第一处分歧在第几步、两边的派发时延各是多少。推理时延的 p50 与 p95 写进报告（包装器比 C++ 推理慢多少，会不会造成迟到步）。
- 第三步，训练对拍：`proto/train.py` 复刻主线配方（与 `$D/exp-student-select-20261109/e1-noise/seed0/run/config.json` 逐项相同：30 轮、batch 128、lr 0.001、Adam 的 β 与 ε、
  类权重 auto、丢碰撞帧、无增强、每轮存档）；`proto/export.py` 把 PyTorch 权重写成 C++ checkpoint，之后的平均加校准与闭环都走主线工具。
  验收（a，离线）：在同一份数据集（`$D/tmp/agent6-distill/trunc/ds/dataset.uavr`）上训 3 个种子，val 与 test 的 macro-F1、每类召回落在 C++ 三个参照种子的范围上下各放宽 0.03 之内。
  验收（b，闭环，排 Gazebo 池的锁，2 条车道，新协议，带护栏，20 个冻结 seed × 3 个种子）：导出、平均加校准之后用 `uavrgb_infer` 评测。
  预注册：与第二轮的三个 K=1 臂（53/60）相比，成功数相差不超过 6 集。超过就说明两套训练不等价，停下查原因，不往下用。
  训练排训练池的锁。
  2026-11-12 07:00 主会话的决定（在看到任何原型闭环数据之前）：验收 (a) 的字面判据立不住。它对 33 个样本的类只容得下 1 个样本的差；同一条判据套在 C++ 自己的 5 个新种子上只过 1 个，
  任取 3 个 C++ 种子作参照、其余当候选的逐个通过率是 16%（`$D/exp-proto-parity-20261112/step3-train/accept_with_control.md`）。改按分布读法：原型三个种子的每项指标落在 8 个 C++ 种子的最小到最大之间
  （24 格全部满足，|z| 最大 1.31；`step3-analysis/distribution_check.md`），(a) 记通过，(b) 带 `ALLOW_OFFLINE_FAIL=1` 起跑。
  2026-11-12 08:31 更正（第二轮审核 L-1，已复现）：上面这条「落在 8 个参照种子的最小到最大之间」的读法也立不住，作废，不许当以后的对拍判据。
  两条线完全等价时，3 个候选种子的 24 格全部落在 8 个参照种子的极值之内的概率只有 3.3%（11 个真实 run 的 165 种分法里 5 种），这一次过了是碰上的；
  它量的又是没校准的 `ckpt_best`，工作点的漂移把种子间的方差放大了，单个转向类的召回要差 9–18 个点才检得出。放行 (b) 的决定仍然成立，依据换成审核者实跑的更有力的检查：
  8 个 C++ run 与 3 个原型 run 都换成交付口径（末 10 轮平均加校准），在 val 加 test 的 1,458 帧上比逐帧预测，跨线一致率 0.9446（C++ 内部 0.9374，原型内部 0.9479），
  双样本置换检验（165 种标号分法，精确）p = 0.78；概率的全变差距离同样检验 p = 0.87；九项指标的两样本 t 值绝对值都不超过 1.04（`$D/tmp/phase12-cards/review2-l/verify_all.log`）。
  以后的对拍判据：在交付口径上，对逐帧预测做跨线对线内的双样本置换检验（p 不低于 0.05 记不可分），并报 test macro-F1 的均值差与区间；参照种子不少于 8 个，候选不少于 3 个；
  参照种子的个数在看结果之前定死，不许判不过再加种子。
- 不做：任何新结构、新损失、新数据、新的评测端规则。
- 交付：`proto/README.md`（规矩、怎么训、怎么导出、怎么接闭环）；测试（前向对拍、checkpoint 往返、包装器协议）；报告。
- 提交：`feat(proto): PyTorch 原型线地基（模型复刻、checkpoint 读写、行协议包装器、训练配方复刻，逐项与 C++ 对拍）`。

### 卡 E2E-R1：训练配方的稳定性筛选（离线）与工作点扫描（闭环）（审核 H 的 H2、H3、H6；11-12 立，立即开工）

- 回答的问题：审核 H 用已有的逐轮预测复现了一件事：训练种子之间、相邻轮之间的不稳定，大部分是「前进对转向」这一个工作点在漂，网络对帧的排序是稳的
  （每轮单独做一次前进校准，相邻轮翻转率从 0.21–0.24 降到约 0.05，三个种子逐帧一致率从 0.67 升到 0.90）。这张卡问两件事：
  用训练器现成的开关能不能在配方层把漂移消掉；这个工作点在闭环上放在哪最好（它现在取的是教师状态分布上的前进占比 0.873，从来没有在闭环上调过）。
- 先读：`docs/reviews/2026-11-12-cross-review/review-h.md` 的 H2、H3、H6、H8 与第 3 节改动一；审核者的脚本 `$D/tmp/phase12-cards/review-h/h1_bias_drift.py` 与它的日志（指标的算法照它来，能复用就复用）；
  `tools/select_ckpt.py`（逐轮 val 预测是怎么落盘的：`$D/exp-student-select-20261109/select/seed{0,1,2}/preds_epNN.npy`）；`tools/average_ckpt.py`（`shift_forward_bias`、`calibrate`）；
  参照配方 `$D/exp-student-select-20261109/e1-noise/seed{0,1,2}/run/config.json`；`src/train_main.cpp` 的开关清单（`--cosine-lr`、`--class-weights`、`--warmup`）。
- 第一部分，离线筛选（只训练与离线推理；排训练池的锁，3 个训练进程可以同时跑，单个约 7 分钟；不起 Gazebo）。
  数据同参照学生：`$D/tmp/agent6-distill/trunc/ds/dataset.uavr`。训练种子 0、1、2，30 轮，`--save-every-epoch`，其余开关与参照配方逐项相同，只改表里写的：

  | 臂 | 与参照配方的差别 | 来源 |
  | --- | --- | --- |
  | R0 | 无（参照配方：恒定学习率，类权重 auto） | 现成的三个 run 与它们的逐轮预测，不重训 |
  | R1 | 加 `--cosine-lr` | 新训 |
  | R2 | `--cosine-lr`，类权重改成显式的 `0,1,1,1`（SAFE 不参与，三类不加权） | 新训 |
  | R3 | `--cosine-lr`，类权重改成显式的 `0,1,2,2`（卡 W2-1 的计划配方加余弦退火） | 新训 |

  每个臂报（val 574 帧；算法同 `h1_bias_drift.py`）：末 10 轮的相邻轮预测翻转率（原始，以及每轮单独校准之后）；末轮与末 10 轮平均权重的三种子逐帧全一致率（原始与校准之后）；
  逐轮需要的校准量 δ 的范围；val 的无权交叉熵曲线；末 10 轮平均权重的 macro-F1 与每类召回。
  起跑前写下的预期（审核 H 的预测，不许改）：R1 的原始翻转率降到 0.08 以下（R0 是 0.21–0.24）。
  通过线：不做事后校准时，末 10 轮翻转率不高于 0.06，并且三种子全一致率不低于 0.88。没有臂过线也照实报，第二部分只做工作点扫描。
  不开增强：增强的随机数种子现在写死、不随 `--seed` 变（审核 H8），开了之后三个训练种子看到同一串增强参数，种子之间的一致率会被高估。
  2026-11-12 07:30 更正：训练器不接受显式的 0 权重（R2、R3 没有跑成）。执行者改跑替代臂 R2e（`1e-6,1,1,1`）与 R3e（`1e-6,1,2,2`），主会话接受这个写法：SAFE 类 56 个样本，
  1e-6 的权重对加权损失的相对贡献约 7e-9（估算）。之后的卡里凡是写「SAFE 权重 0」的地方都按 1e-6 做（卡 W2-1 同）。结果：R2e、R3e 过线，R1 不过线（见报告）。
- 第二部分，闭环（Gazebo，守护进程，排 Gazebo 池的锁，2 条车道；新协议，命令行照 `experiments/2026-11-10-action-latency/run_r2.sh` 的 `acc_job`，去掉 `--heading-guard`；`--task base --steps 80`）：
  评测几何用 val 分区的 20 个（`--split val`；从一段没被别的实验用过的 seed 区间里按 val 分区取前 20 个，清单冻结在入口脚本里）。工作点是在调参，不能在 20 个冻结的 test seed 上调（公共规则 §2）。
  工作点扫描：现成的 `cal_swa_s{0,1,2}_k10.bin`，前进偏置在现有校准值上再加 −0.5、0、+0.5 nat 三档（用 `shift_forward_bias` 生成，不训练），3 个训练种子 × 20 个几何，每档 60 集。
  第一部分有臂过线时，再加一个臂：那个配方的末 10 轮平均加校准，60 集。含迟到步的集按 `run_r2.sh` 条款 R6 重跑一次。
  预注册（写进入口脚本头，不许改）：三档里成功数最高与最低相差 8 集以上，说明工作点在闭环上是一个要显式调的量；相差不到 4 集，说明在正负 0.5 nat 之内闭环对它不敏感。
  过线的配方臂：三个训练种子的成功数极差不超过 3，总数不低于同批几何上 0 档的总数减 3。
  另报：每档的撞柱、越界、未到线，动作占比；三个训练种子各自的数。主比较按几何聚类报检验（公共规则 §2 的 11-12 一条），逐集 McNemar 并列。
- 不做：改 C++；logit adjustment 与权重 EMA（要写代码，在原型线上做，卡 E2E-P1）；在冻结的 test seed 上跑任何一档。
- 入口 `experiments/2026-11-12-recipe-stability/`，输出 `$D/exp-recipe-stability-20261112/`。
- 提交：`feat(exp): 训练配方的稳定性筛选与工作点扫描`。
- 11-12 09:13 加（第二轮审核 M-5、M-10；这时第二部分 12 个臂跑完 10 个，主会话看过运行日志里各臂的成功数。下面三条只收窄能下的结论，不改任何判据；冻结的代码不动）：
  1. 加臂 R2e 是看了第一部分的结果之后加的，它在第二部分的数标成「事后探索臂」，不作预注册的过线结论。三档工作点扫描（P1–P3）的预注册效力不受影响。
  2. R2e 相对 R0 的稳定性提升写成「余弦退火与类权重（`1e-6,1,1,1`）的联合效应」；没有只改类权重的消融臂，不拆开归因。
  3. R2e 的交付权重被事后校准减了 0.78–0.98 nat（0 档是加 1.56–1.70 nat）。R2e 的闭环数不好时，先对照三档扫描给出的工作点方向，看是不是负向校准压低的，不就此否定这个配方。
  4. 收尾报告附 12 个臂的表：被限流时长占墙钟的比例、相邻观测间隔的最大值、停顿步数、巡航步位移的均值（审核者已核 6 个臂：4,192 步里停顿步 0）。区间与检验统一用 `tools/paired_stats.py` 重算一遍并列。

### 卡 E2E-1b：带条件模型的按码校准（审核 H1、C1、D6；规格初稿；等 E2E-1 的第二步提交与训练权重出来后开工）

- 背景：`average_ckpt` 的校准只有一个标量，加在所有帧的前进 logit 上。对带航向码的模型，这个量由码 0 的帧决定（示范数据里码 2、码 3 的帧很少），却同样压在需要回转的帧上。
  E2E-1 的 M2、M3 如果不成立，原因可能是校准，不是航向码。E2E-1 照预注册跑完，这张卡补上分辨这两种原因的臂。
- 做法：`tools/average_ckpt.py` 加按码校准的选项：只在码 0 的帧上求校准量，也只施加在码 0 上。
  不改推理协议：FiLM 的 β 经全连接之后就是一张按条件值分行的类偏置表（审核 H5 的推导），把码 0 那一行按最小范数解平移，使码 0 的前进 logit 增加校准量、其余三类不变。
  加测试：平移之后码 0 的帧的 logit 变化等于校准量（数值误差 1e-4 以内），码 2、码 3 的帧的输出逐位不变。
- 离线（不起 Gazebo）：对 E2E-1 的 A1、A2，分码统计三种权重（只平均、全局校准、按码校准）在 val 上码 2、码 3 帧的「朝回转」预测率。
  审核 H 的预测：全局校准让这个比率下降 10 个点以上。
- 闭环：A1c、A2c（按码校准）各 3 个训练种子 × 20 个冻结 seed，协议与 E2E-1 相同，与 A1、A2 逐集配对。只报告，不设判决；与 E2E-1 的三个主比较并排读。

### 卡 E2E-P1：原型线上的结构与配方探索（审核 H 的第 3、4 节；规格初稿；等 E2E-P0 对拍通过、E2E-0 与 E2E-R1 出结果后细化）

- 候选（每个都先用离线稳定性指标筛，过了才进闭环；指标与通过线照审核 H 第 3 节）：
  1. 配方：logit adjustment 替代类权重，余弦退火，权重 EMA；工作点做成一个显式的数。
  2. 结构：列结构头替换 19,200 → 4 的全连接（第四层卷积、沿高度的逐通道聚合、列间卷积、小线性层，约 7 万参数，各层加 BN，导出时折叠）。
  3. 条件输入：FiLM 前移到第二层卷积之后；训练时以 0.2–0.3 的概率把条件置成「未知」；校准按码分开。条件码按步数与按角度两种定义做对照（计划 §10.7 第 6 条）。
  4. 记忆：先看 E2E-0 里「完美短期记忆」比历史码多出多少。多出 10 个点以上，做「按已执行偏航平移的列特征缓存」；小循环单元放在最后。
- 胜出的结构移植到 C++（卡 E2E-C1 之后的移植卡），移植后要在闭环上复现原型线的数才算。
- 2026-11-12 07:30 细化：**第一轮，离线筛选**（可以开工；只用训练池与 GPU，不起 Gazebo；分支 `p12-e2e-p1` 从 `p12-e2e-p0` 的 HEAD 起，只改 `proto/` 与本卡的实验目录）。
  - 依据的三份结果：E2E-0（学生驱动的帧上，只看画面判转向方向是 0.51，补上已执行的净转向计数 n 是 0.86，三值航向码 0.80，上一步动作没有作用；`experiments/2026-11-12-decidability-audit/RESULT.md`）；
    E2E-R1 第一部分（余弦学习率加显式类权重 `1e-6,1,1,1`，即臂 R2e，末 10 轮相邻轮翻转率 0.008、三种子逐帧一致率 0.942，不靠事后校准；`$D/tmp/phase12-cards/reports/e2e-r1.md`）；
    E2E-P0（原型线前向逐帧对拍通过，训练配方统计上与 C++ 不可分；`proto/README.md`）。审核 H 的第 3、4 节是候选的出处（`docs/reviews/2026-11-12-cross-review/review-h.md`）。
  - 数据（都只读，划分用数据集自带的按几何哈希的 split；筛选只看 val，test 只在最后对选中的臂看一次）：
    示范集 `$D/tmp/agent6-distill/trunc/ds/dataset.uavr`（教师驱动，9,663 样本）；DAgger 第 3 轮聚合集 `$D/exp-dagger-gz-20261109/full-1109/round_3/ds/dataset.uavr`（592 集、20,291 样本，含学生驱动的帧）；
    带三值航向码的两个包 `$D/exp-e2e-heading-code-20261112/data/{demo-hc,dagger-hc}/`（卡 E2E-1 打的，码在 `dataset.aux` 里）。
  - 指标（val；每个臂 3 个训练种子，报均值与最小到最大）：
    稳定性照 E2E-R1 的算法（末 10 轮相邻轮翻转率取三种子最大、三种子逐帧全一致率，原始与校准后各一份；能复用 `experiments/2026-11-12-recipe-stability/stability.py` 就复用）；
    macro-F1（F/L/R 三类）与每类召回，取末 10 轮平均的权重，不校准与校准后各报一份；「该转时转了」与「转向帧左右对」照 E2E-0 的定义（`RESULT.md` 第 22 行），另在学生驱动的帧上单独报一份（帧的选法照 E2E-0 的 `audit_features.py`）；
    带条件的臂加两个反事实探针：把 n 的符号取反，转向帧上预测方向翻转的比例；把码置成 0，macro-F1 的变化；参数量与乘加数。
  - 臂（每个 3 个训练种子，30 轮，逐轮留档；配方除表里写的之外与 R2e 相同）：

    | 组 | 臂 | 内容 | 数据 |
    | --- | --- | --- | --- |
    | A 配方（基础结构，纯图像） | A0 | 参照配方 R0（恒定学习率，类权重 auto）。原型线已有三个种子的 run（`$D/exp-proto-parity-20261112/step3-train/seed{0,1,2}`），逐轮预测由逐轮权重补算，不重训 | 示范集 |
    | | A1 | R2e：余弦学习率，类权重 `1e-6,1,1,1` | 示范集 |
    | | A2 | A1 加权重 EMA（衰减 0.999，交付用 EMA 权重） | 示范集 |
    | | A3 | logit adjustment（τ = 1，先验取训练集 F/L/R 三类的频率；SAFE 的权重 1e-6、调整量与 FORWARD 相同，使它不被抬高），余弦学习率，不加类权重 | 示范集 |
    | B 结构（纯图像，配方 A1） | B0 | 基础结构（三层卷积加 19,200 → 4 的全连接）。示范集上就是 A1，不重训 | 示范集、聚合集 |
    | | B1 | 基础结构，每层卷积后加 BN（导出时折叠进卷积） | 示范集、聚合集 |
    | | B2 | 列结构头：第三层池化之后接第四层 3×3 卷积加 BN、沿高度的逐通道可学权重聚合（每通道 15 个权重，得到 20 列的特征）、列方向的一维卷积（核 5）加 BN、小线性层出 4 类。总参数不超过 7.5 万 | 示范集、聚合集 |
    | | B3 | B2 的对照：沿高度改成取平均（没有可学权重） | 示范集、聚合集 |
    | C 条件输入（聚合集，配方 A1；基础结构与 B 组里胜出的结构各做一遍） | C0 | 不给条件（基础结构上就是 B0 的聚合集那一份） | 聚合集 |
    | | C1 | 三值航向码，FiLM 接在线性层之前（主线现在的接法） | `dagger-hc` |
    | | C2 | 净转向计数 n 截到 ±4 共 9 档，FiLM 接在线性层之前 | 聚合集加自算的 n |
    | | C3 | n 的 9 档，FiLM 前移到第二层卷积之后（码过一个小嵌入出每通道的 γ、β） | 同上 |
    | | C4 | C3 加条件随机置空：训练时以 0.25 的概率把码换成第 10 个取值「未知」 | 同上 |

    n 的算法：一集里到当前步为止已执行的左转次数减右转次数（用已执行的动作，不用教师标签，不用位姿）。先写一条核对：按 n 的阈值 ±2 折成三值之后，必须与 `dagger-hc`、`demo-hc` 两个包里的码逐样本相同
    （`sim/tools/nav_cond.py` 是三值码的唯一实现，折法照它）。对不上就停下报告，不往下训练。
  - 起跑前写下的预期（写进入口脚本头，不改）：
    P1（闸门）：原型线上的 A1 复现 C++ 的 R2e：末 10 轮翻转率不高于 0.03、三种子全一致率不低于 0.90。不满足说明原型线在这个配方上与主线不等价，停下报告，B、C 两组不跑。
    P2（码有没有被用上）：学生驱动的帧上「转向帧左右对」，C0 不高于 0.60（E2E-0 里用真值列距离也只有 0.51）；C1 比 C0 高 0.15 以上；C2 不低于 C1。证伪：C1 比 C0 高不到 0.05（图像学生离线上用不上这个码）。
    P3（FiLM 的位置）：C3 的符号取反探针翻转比例比 C2 高 0.10 以上。相差不到 0.05 记「位置在离线指标上没有区别」。
    P4（结构，只报告不设判决）：B2 用不到基础结构 75% 的参数，val macro-F1 不低于 B0 减 0.01，「转向帧左右对」不低于 B0。
  - 另外要做的两件：
    包装器能直接载入原型线自己的模型文件（新结构写不成 C++ 的 checkpoint）。验收：基础结构的同一份权重，走 `.bin` 与走原型模型文件两条路，`proto/replay_compare.py` 的动作序列逐步相同。这是第二轮上闭环的前提。
    训练按批拿训练池的锁：一批最多 3 个训练进程，跑完就放掉锁再拿下一批（卡 C1a 的短训练要插得进来）；每个「臂 × 种子」落完成标记，同一条命令能续跑。
  - 交付：入口与汇总脚本（`proto/runs/` 下，带测试）、结果表 `experiments/2026-11-12-proto-p1/RESULT.md`、报告、过程文档；报告末尾给「建议进第二轮闭环的臂（不超过 3 个）」与每个臂上闭环还缺什么
    （主线的 `--student-cond` 认不认 9 档码、评测端怎么算 n）。第二轮（闭环，20 个 val 几何 × 3 个训练种子，几何清单用 E2E-R1 冻结的那 20 个）由主会话看过第一轮结果之后定臂。
  - 不做：按偏航平移的列特征缓存与循环单元（等第一轮结果；E2E-0 量到宽视场只多 0.04–0.09）；条件码按角度定义的对照（要固件的偏航估计进数据，另立卡）；改 `proto/` 之外的任何文件；起 Gazebo；在 test 上挑臂。

## 4c. 第二轮交叉审核的修复卡（11-12 08:19 起；清单 `docs/reviews/2026-11-12-review-round2/FIXLIST2.md`）

规矩同第一轮的修复卡：公共规则 §5 的 11-12 四条（一条发现一个提交，说明第一行带发现编号，代码与测试同一个提交）与 §8；缺省行为逐位不变，变了的在报告里单列；每条先红后绿。

- 审核 K 给后面几张卡的约束（11-12 08:53 记）：
  - 卡 W1-10 的验收不改参数副本（指纹 `06e5f017` 已预注册）。收尾报告要报每个臂的起飞重试计数（`takeoff_failsafe_retries`）；K-3：这份副本没有改悬停前馈，独立积分模式下起飞峰值越过 2.50 m 的量程上限。
  - E3 与卡 W2-6 采满电数据之前，换成前馈匹配的满电副本（`tools/make_params_variant.py --set sim.battery_soc_init=1.0 --set hover_duty_estimate=0.734`，新指纹），
    先在 `cf_sitl_proc` 上核对两件事：40 集起飞峰值都低于 2.50 m；巡航段每步位移与转角同 `06e5f017` 的差在 1% 以内（差得多就说明 W1-10 的验收结论不能搬到新副本上，要补跑）。
  - K-8：单集不超过 120 s（K=3 约 90 s，在包络内）。更长的课程要先在固件估计器里启用飞行中的重力修正并升库（lib32），不在第十二阶段的范围里。

### 卡 W1-14b：统计工具的三处补丁（审核 I2-1、I2-3；11-12 08:19 立，可以开工）

- 依据：`docs/reviews/2026-11-12-review-round2/review2-i2.md` 的 I2-1、I2-3（各有最小修法与复现脚本 `$D/tmp/phase12-cards/review2-i2/verify_i2.py`）。
- 只动 `tools/stats_report.py`、`tools/paired_stats.py` 与它们的测试、`docs/EVAL_STANDARD.md` §5。分支 `p12-w1-14b` 从 main 起。
  1. I2-1：`load_eval` 在顶层没有 `dynamics_profile` 时从 `per_episode` 取（单值取它，多值拼成排序后的串让一致性检查报错）；`per_episode` 为空而顶层有 `cf_sitl_params` 时用它的 `values_sha256` 前 8 位参与比较。
     这一条会让以前能出表的输入开始报错（不同固件库的评测混在一张表里）：在报告里列出数据盘上现有的 eval.json 按 `dynamics_profile` 分成哪几桶、哪些已有的出表命令会因此要加 `--allow-mixed-protocol`。
  2. I2-3：`load_eval` 取出去重后的未完成 seed（`failed` 与 `skipped` 合并），`run_outcomes` 与 `cell_clusters` 共用；`pair_units` 遇到两侧都没有几何号的共同未完成 seed 时只让它单独成簇，不把整张表退回按 seed 聚类；失败记录重复不重复累加分母。
     验收：审核者的 `TEST 3c`（2 个 run、4 个 seed）下主表与配对表的成功率、聚类键一致。
  3. 没有跳过集、没有共同失败 seed、`dynamics_profile` 一致的输入，报表逐项不变（沿用卡 W1-14 的对比脚本 `$D/tmp/phase12-cards/w1-14/check_invariance_full.py`，把结果写进报告）。
- 不做：改 `eval_closed_loop.py`（全失败时顶层写什么，并入卡 W1-13 第二轮）；改三条流水线里自带的简表（卡 W1-15c）。

### 卡 PK1：打包的健壮性（审核 I2-4、I2-6、I2-8 第 2 点、I2-9 第 2 点，第一轮 C7；11-12 08:19 立，可以开工）

- 依据：`review2-i2.md` 的 I2-4、I2-6、I2-8、I2-9（复现脚本同上，`TEST 2a`–`2f`）；第一轮 `review-c.md` 的 C7。
- 只动 `tools/preprocess.py`、`tools/test_preprocess_*.py`（新测试文件可以加）、`tools/DATASET_FORMAT.md`。分支 `p12-pk1` 从 main 起。
  1. I2-4：`teacher_crossed_goal` 区分「文件不存在或不是柱阵任务」（仍是不适用）与「文件在但读不懂、数值不是有限数」（报错停下，写清是哪个文件哪一行）；`actions.jsonl` 行数与 `manifest` 不符的检查挪到两个整集过滤之前；
     某个 split 被过滤跳空时报错（train、val 任一为 0），给一个显式开关才放行。
  2. I2-6：整集的步被行级过滤全部滤掉时，这一集不进 `episodes`、不进指令侧车，单独计数并打印；每一行自己的 `action_executed` 直接校验是 0–3 的整数，不再靠查上一步；步号有缺口时两种条件（上一步动作、航向码）的处理写成同一个规则并写进 `DATASET_FORMAT.md`。
  3. C7：行级过滤丢掉某一步之后，下一步的「上一帧」指向自己而条件值不是 SAFE 的组合。先量现有数据集里这种样本有多少（示范集、DAgger 聚合集、两个航向码包），再定是保留现状加文档，还是给这种样本的条件值一个明确的规则；只报数与建议，改不改由主会话定。
  4. I2-8 第 2 点：解析深度里圆柱补上顶盖与底盖的求交，与方柱一致；相机低于柱顶时（现有全部数据）结果逐位不变，用现有的对拍数据核一遍。
  5. I2-9 第 2 点：给 `build_meta` 的「几何组跨 split」那道门加一条直接调用的测试。
- 缺省行为：合法输入下打包结果逐字节不变（用审核者的四集夹具与 0417 的 6 集真实数据各核一遍）。变严的地方（坏文件报错、空 split 报错）在报告里逐条列出。

### 卡 S1b 第二轮：碰撞口径 v2 定稿（审核 J-1、J-2、J-3、J-4、J-9 与 B5；11-12 08:31 立，可以开工）

- 依据：`docs/reviews/2026-11-12-review-round2/review2-j.md` 的 J-1、J-2、J-3、J-4、J-9（复现脚本 `$D/tmp/phase12-cards/review2-j/verify_all.py`）；卡 S1b 第一轮报告里 B5 的核实结果。
  第一轮合进 main 的 v2（合并 `2b3eee7`）是没定稿的：格中心的膨胀核在 0.1 m 体素下比机体多判 3.9 cm、在 0.2 m 体素下仍漏判 10.2 cm；高度余量只按稳态量，起飞过冲时判定看到的层比规划多。还没有任何数是用它出的，`v2` 这个名字与 `+coll.v2` 的后缀留给定稿后的口径。
- 定稿的口径，一句话：机体是一个圆柱（半径 = `body_radius + inflate`，半高 = `body_half_height`），原始占据的每个体素是一个实心方块，圆柱与任何一个方块相交就是碰撞。不再有高度余量。
  1. 连续坐标上的查询（`collides`、`sweep`、`sweep_path`、规则臂的列距离，都走 `_grid_hit`）在 v2 下：竖直方向，机体的高度区间 [z − 半高, z + 半高] 与哪些层的高度区间相交就查哪些层；
     水平方向，粗筛之后取点到邻域里原始占据体素方块的精确距离，不超过半径才算碰（审核 J 的实现 `verify_all.py:29-112` 可以参考）。`HEIGHT_MARGIN_M` 去掉。
     测试：与暴力公式在随机点上逐点一致；0.1 m 体素下斜向贴角判碰、离角 0.11 m 不判；0.2 m 体素下斜向离角 0.05 m 判碰、轴向离面 0.15 m 不判（第一轮的 v2 在这两处都是反的）。
  2. 按一个名义高度做的两件事（教师规划的硬栅格、出生检查）用整集的高度包络取层：`measure_z_margin.py` 扩成量全部步（含第 0 步动作前的位姿与窗口内的轨迹点）的高度减出生高度的最小与最大，
     各向外取到 5 mm，定出 `Z_ENVELOPE_DOWN_M`、`Z_ENVELOPE_UP_M`（审核 J 量到上沿 +0.070 m）。规划的硬栅格 = 高度区间 [名义高度 − 下包络 − 半高, 名义高度 + 上包络 + 半高] 碰到的层取并，再用格中心的核（现有的 `_dilate_euclid`）。
     出生检查 = 候选点在这些层上到原始体素方块的精确水平距离不小于半径加 `SPAWN_DRIFT_MARGIN_M`（起飞的水平漂移同样先量后定；卡 W1-12 量到最大 8.2 cm）。v2 下仍不与 `spawn_two_layer` 叠用。
     这样规划看到的障碍是判定可能判到的障碍的超集：教师不会规划穿过起飞时或高度波动时机体够得到的地方。
  3. J-9：名义高度取路线自己的出生高度。共用驱动与 `planner.py` 的命令行入口按高度缓存规划器（三维占据只建一次）。
  4. J-3：`sim/3dgs/eval_3dgs.py` 加 `--collision-model`（缺省 v1，逐位不变），`eval.json` 记 `collision_model`，`metric_version` 随口径加后缀。
  5. J-4：重定基线入口的 `SPLIT_FROM_IDENTITY` 缺省改成 true；补一条 `split_from_identity` 与 `collision_model` 两个键同开时 `done.json` 键顺序的测试（合并时定的顺序：split 的两个键在前）。
  6. B5：渲染器加可选的 `depth_opacity_min`（缺省不给 = 现状逐位不变）：RGB 照旧，深度只用 opacity 不低于门限的高斯另渲一遍。重定基线加一个诊断臂：规则 v2（深度）带 `depth_opacity_min = 0.30`，只在 v2 下跑。
  7. 定稿之后先离线重判，再谈在线的实验：用已录的轨迹（卡 W1-11、W1-12、K1、K3 各臂）在定稿的 v2 下重判，与审核 J 报的「精确距离加层带加折线」那一列对上（扫掠教师 98/120，碰撞 23）；对不上就停下报告。
  8. 重定基线的臂表与预注册不动。汇总里加两列：每个臂在 v2 下判成碰撞而 v1 下没有的集数；规划与判定的层不一致的步数（应当是 0）。正式实验这一轮仍不起，等主会话审完这一轮的代码。
- 缺省 v1 的一切仍然逐位不变（5 组金标）。一条发现一个提交。分支 `p12-s1b2` 从 main 起。

### 卡 P1b：原型线的更正（审核 L-2、L-3 的原型线一侧、L-4、L-5、L-6 第 1 条、L-7、L-10；11-12 08:31 立，可以开工）

- 依据：`docs/reviews/2026-11-12-review-round2/review2-l.md`（每条有位置、复现与最小修法；复现脚本 `$D/tmp/phase12-cards/review2-l/verify_all.py`）。
- 分支 `p12-e2e-p1b`，从 `p12-e2e-p1` 的 `48766e8` 起，新工作树 `wt/p12-e2e-p1b`；只改 `proto/`。`wt/p12-e2e-p1`（卡 E2E-P1 的执行者在用）与 `wt/p12-e2e-p0`（闭环验收要从它冻结代码）一个字节都不动。
  1. L-2：带 BN 的模型，逐轮留档存没折叠的参数与 BN 的缓冲；交付权重的做法改成：末 K 轮的参数取平均（或取 EMA）→ BN 的统计量清零，在训练分区上不带梯度跑一遍精确重估 → 再折叠导出。
     回归测试要在旧做法上红：平均后的模型在 val 上 logit 的标准差与单轮模型相比不应缩到一半以下（审核者实测旧做法缩到 33%）。不带 BN 的结构逐位不变。
  2. L-3 的原型线一侧：三值码开条件置空时条件维度的笔误；交付权重旁写 `<ckpt>.json`（run 目录的绝对路径、条件类型、数据集的绝对路径与 sha256）。主线客户端认更大的条件维度、行协议加集首标志不在这张卡（卡 E2E-C1）。
  3. L-4：`read_aux` 核对侧车头里的样本数、数据集哈希、载荷哈希、文件长度、上一帧下标与条件值的范围；`Uavr` 缺省拒绝诊断性的 split，给显式开关。
  4. L-5：`proto/train.py` 里 `DATA_ROOT` 的含义与 `env.sh` 一致（不带 `/uav_rgb`）。
  5. L-6 第 1 条：新写的余弦学习率与显式类权重（配方 A1）与 C++ 的同配方对拍：种子 0 的前 5 步 loss 相差不超过 1e-5（卡 E2E-R1 的 C++ run 在 `$D/exp-recipe-stability-20261112/runs/`；逐步 loss 取不到就用同一个二进制重跑 5 步）。
  6. L-7：汇总对带条件的臂按 n = 0 与 |n| ≥ 2 分桶，报不校准与校准两种权重的转向预测占比。
  7. L-10：`encode_checkpoint` 写之前校验类权重、最佳指标、契约里的学习率是有限数且在合法范围；`test_eval.jsonl` 补 `run_id`。
- 这一轮不跑整轮训练（回归测试可以用 GPU，单进程两分钟以内）。交回后主会话把它并进 E2E-P1 的分支，带 BN 的臂用改好的代码重跑。

### 卡 S2：3DGS 线给 E4 用的教师与学生接线（审核 J-5、J-6 第 1 点、J-7；规格初稿，等卡 S1b 第二轮合并后开工）

- J-5：3DGS 线的三个入口（`collect_3dgs.py`、`eval_3dgs.py`、共用驱动）构造学生之后调 `nav_cond.check_student_cond`，对不上就拒绝，不再把航向码模型当成上一步动作模型喂。
- J-6 第 1 点、J-7 第 1 点：教师一侧的集内状态也按实际执行的动作更新（与卡 S1a 给学生做的是同一件事）：`OccAstarTeacher` 与 `RuleOraclePolicy` 各加 `note_executed`，`DaggerPolicy` 定下执行的动作之后通知教师。
- J-7 其余：规则 v2 的入口计数按分级数生成；`DaggerPolicy` 容忍没有 `stop_radius` 的教师；`collect_3dgs.py` 加规则策略的选项，名字带 oracle；到点即结束本集，不录转圈的尾巴；打包端登记这个教师名（等卡 PK1 合并后改 `preprocess.py`）。
- J-8 不改规则 v2（冻结）。E4 的采集丢掉没到点的规则集；要更好的教师另起版本。

### 卡 W1-13 第二轮：采集与评测端的几处口径（审核 I1-2、I1-3、I1-4、I1-5、I1-7、I1-9，I2-1 第 2 点、I2-2 第 2 点，K-2；11-12 08:39 立，08:56 开工）

- 这一轮不含「动作持续到下一次派发」与重标定（那是第三轮，等审核 K 的报告与插件侧生效时刻的测量）。但 I1-4 是重标定的前提，要在这一轮修掉。
- 只动 `sim/tools/collect.py`、`collect_parallel.py`、`eval_closed_loop.py`、`teacher.py` 里的预热函数（只加预热的范围，不改任何路径的数值）与它们的测试、`docs/EVAL_STANDARD.md`。分支 `p12-w1-13b` 从 main 起。
  1. I1-4：带 `--action-latency` 时，教师的栅格预热覆盖标准余量与两档重试余量。验收：审核者的测量（3 块课程冷触发 456–719 ms，预热后 70 ms 上下）自己重量一遍；不带开关时行为不变。
  2. I1-2 与 K-2：`collect_parallel.py` 加 `--cf-sitl-params`（校验与绝对路径同评测端），透传给 `collect.py`，`collection_manifest.json` 记这个键；不给时命令行逐字不变。
     防呆放在 `collect.py`：`--dynamics cf_sitl` 且给了 `--scene-meta` 时，场景 meta 里的 `cf_sitl_params` 与自己拿到的参数文件（都缺省时是缺省文件）按真实路径比，不一致就退出。
  3. I1-3 与 I2-2 第 2 点：`collect.py --policy student` 构造学生之后调 `nav_cond.check_student_cond`，对不上就退出；三处入口在带 waypoint 的任务上拒绝 `--student-cond heading-code`（与打包端一致）。
  4. I1-7 与 I2-1 第 2 点：`eval_closed_loop.py` 在参数校验阶段就读一遍 `--cf-sitl-params` 的文件（坏文件当场报错）；完成集为空时顶层的 `dynamics_profile` 由参数文件算出，不写 null；
     显式传缺省参数文件与不传这两种情形在报表里要算同一种协议：主会话 11-12 08:56 定在统计端解决（顶层没有键时回退读逐集的 `dynamics_profile`；归卡 W1-14b 的后续），评测端不为它改缺省输出，默认不传时 eval.json 逐位不变。
  5. I1-5：每百米碰撞数的分子与分母取同一个窗口。定义（主会话 11-12 08:56 定）：窗口是「从起点到第一个终局事件」，终局事件指第一次越出走廊、第一次进目标区或过终点线、碰撞；
     分母是窗口内的路程（现状），分子只数窗口内的碰撞（同一步既碰撞又越界算碰撞，与逐块存活的规则一致），越界之后才发生的接触不进分子。理由：越界那一刻这一集的结局已经定了，之后飞到没电落地的接触与任务无关。
     `collision_rate`（整集有没有碰撞）不动。改的是指标的定义，新旧数不能混：照仓库现有的做法记定义版本。报告里给出现有两个长课程实验各臂在新旧定义下的数，定义写进 `EVAL_STANDARD`。
  6. I1-9：断帧旗与停顿步用同一个判据（`collect.is_stall_gap`）；旧键保留，现有 eval.json 重算时哪些集的旗会变，列出来。
- 每条一个提交，先红后绿；缺省产物逐位不变的测试照卡 W1-13 第一轮的写法。

### 卡 W1-13 第三轮：动作持续到下一次派发，规格更正（审核 K-1、K-4、K-5；11-12 08:53 立；等第二轮合并、插件侧生效时刻的测量 `$D/exp-w1-13-ack-20261112/summary.md` 出来之后开工）

卡 W1-13 第三部分（第 7–9 条）照下面几条更正，没提到的照原文。依据在 `docs/reviews/2026-11-12-review-round2/review2-k.md` 的「给 W1-13 第三部分的事实」一节，开工前读完。

1. 截止的取法（K-1）。原文第 7 条的「派发戳 + 名义步周期」做不到步间不回 SAFE：下一条动作到达插件时旧动作已经到期。改成：发给插件的截止 = 派发戳 + 名义步周期 P + 保持余量 `guard_s`。
   正常的步里下一条动作在旧动作到期之前到达，直接取代，执行器一拍都不回 SAFE；下一条没来（停顿、集末、进程出事）时最多多执行 `guard_s` 就回 SAFE。
   `guard_s` 先量后定：取插件侧生效时刻减（观测戳 + L）在满载条件下的 p99 加一个位姿周期，上限 0.10 s（要小于停顿判据的余量 0.20 s）；取值与依据写进 `EVAL_STANDARD`，进 manifest 与协议一致性键。
   测量 11-12 09:47 出来了（`$D/exp-w1-13-ack-20261112/summary.md`）：空闲与满载共 362 步加冒烟 68 步，插件侧确认时刻减派发戳全部是 5.0 ms，也就是状态帧的一格，真实延迟在 0 到 5 ms 之间。
   按上面的取法下限是 10 ms；主会话定 `guard_s = 0.05 s`（十格，留出满载时偶发的抖动；仍只有停顿判据余量的四分之一）。不加余量时每步会漏出至多 5 ms 的 SAFE（审核 K 量到 5 ms 的空档让每步位移少 0.7%）。
   迟到判据不改（两种条件的 p99 都小于预注册的 10 ms）。
   真机上对应的语义：每条动作收到之后有效 P + `guard_s`，新动作一到就取代。
2. 记账分两项（K-1）。Python 一侧的 `safe_gap_s` 在这个模式下恒为 0，不能当门。另记：`dispatch_period_err_s`（本步派发戳减上一步派发戳减 P）；
   `plugin_safe_gap_s`（本步在插件侧的生效时刻减上一步发给插件的截止，首尾相接时为负）。标定门里「`safe_gap_s` 中位数为 0、最大不超过 0.02 s」换成「`plugin_safe_gap_s` 全部为负」，外加原有的迟到步 0、停顿步 0。
3. 先修 `ack_sim_t()`（K-4），它是上面那道门的量具：记已确认的位点（同一个截止不配到上一步；两步的截止相同时给后一步加 2 ms 的微偏或者拒绝同戳派发两步，选一种并写明）；
   窗口末两条管道没对齐时等状态帧追上再扫一遍（有超时）；缓冲随 `--hold` 变；200 Hz 状态帧带来的 0 到 4 ms 单向偏差写进汇总的说明，判据的阈值扣掉它。每条一个测试，用审核报告给的两个构造。
4. 标定的口径（K-5）。前飞与转向都报两种量：行内的（派发到窗口末，0.8 s）与整周期的（相邻两个观测之间，0.9 s）。转向分「首步」（前一步不是同向转向）与「连续」（前一步是同向转向）两列。
   审核者在独立积分模式下量到的参照：前飞行内 0.2420 m、整周期 0.2721 m；转向首步整周期 28.24°、连续 30.87°。原文第 8 条的「预期转角 26° 上下」作废，偏离这组参照 5% 以上要写明原因。
   K=3 对 K=1 的 2% 门按同一口径、同一类步比。
5. 下游要跟着改的常量，标定出来之后列清单交主会话，不在这张卡里改：规则层的 `yaw_step_deg`（现在 26°）、航向护栏按步数的阈值、航向码「净转向步数」对应的物理角度（卡 E2E-1 的模型是在旧协议的数据上训的）。
6. 学生参照臂（原文第 9 条）照旧：门过了才跑，只报告。旧协议数据训的学生在新协议下每步位移与转角都变了，属于换了被控对象，结果单列，不与旧协议的数合并。
7. 观测帧的选取改成按时间戳的确定性规则（审核 P-5，11-12 09:18 加）。现在取的是「窗口末已经到达 Python 的最新一帧」，帧间隔恒为 0.900 s 靠的是图像管道比位姿管道慢：窗口末那一帧图像只比位姿帧早 5 ms 拍下，
   被 JSON 管道拖慢 70–175 ms 之后赶不上。图像传输一变快，帧间隔就会跳到 1.000 s。新开关下按时间戳取：取时间戳不晚于「上一步的派发戳 + P − 相机周期的一半」的最新一帧（具体的界与依据写进 `EVAL_STANDARD`），
   缺省开关下行为不变。标定的门加一条：相邻观测间隔全部等于 P。

### 卡 TCH1：教师 v8（审核 I1-1、I1-8、I1-10；规格初稿，E3 的采集开工之前做；等并发的执行者少一些就开工）

- I1-1：降余量重试命中之后，前向 0.4 m 检查用的余量不大于规划命中的那一档；重试的档位要大于机身半宽加 5 cm（x3 不用 0.20 m 这一档）。I1-8：左右两个假想动作与主动作共用同一个目标。
- 改的是教师的动作，要升版：`uavrgb.teacher.v8`，v7 的规则留在按版本分派的表里供审计复算；v7 的数据与 v8 的数据不能并集打包（`provenance_uniform` 会拦）。
- 验收：审核者的重演（0417 数据里被重试救活的 349 步）在 v8 下机头转准后能前进的比例、两步来回的比例；20 个冻结 seed 的教师闭环不低于 v7（Gazebo 池，排在今天的队列后面）；近柱状态的影子标签不再左右交替。
- I1-10：`make_counterfactual.py` 按源集的载具与教师版本设好再重算标签。

### 卡 W1-15c：流水线与交付配方的条件通路（审核 I2-2 第 1、3 点、I2-5、I2-7、I1-6，第一轮 C6 的工具一侧；`p12-e2e-1` 的代码 09:03 合进 main、卡 W1-15b 第二轮 08:58 合并，11-12 09:04 开工）

- 只动三条流水线、`tools/pipeline_common.py`、`tools/average_ckpt.py`、`sim/tools/nav_cond.py` 里读 `<ckpt>.json` 的那一段与它们的测试。
  1. I2-2 第 1 点：两条流水线的 `deliver_check` 在 `preflight` 里就拒绝 `average_ckpt` 不支持的配置（文本、双帧；带条件的在 `dc8e5ff` 之后支持），不等采集与训练跑完才报。
  2. I2-2 第 3 点、I2-7：`dagger_gz_pipeline` 的汇总按请求集报分母（用 `paired_stats.success_counts`），自带的配对检验把失败集记 0；调 `stats_report` 时基线与新臂的协议键不同就传 `--allow-mixed-protocol`，报表照出并带混合协议的表头；`stats_report` 非零退出要进日志并在汇总里写明没有出表。
  3. I2-5 第 1、3 点：`average_ckpt` 写出的 `.json` 记数据集的条件类型（`meta.cond`）与数据集的绝对路径，改成先写临时文件再改名；流水线判断「校准做完了」时要读得出这个 `.json`；`nav_cond.check_student_cond` 在 `.json` 里有记录时优先用它，不再因为相对路径或数据集被挪走而退化成只警告。
- 11-12 09:04 加的三条（文件范围相应加上 `tools/select_ckpt.py`、`tools/exp_lib.sh`、`experiments/_template/README.md`）：
  4. I1-6：`nav_cond.find_train_config` 在每个候选目录先看 `config.resume.json` 再看 `config.json`（同目录续训换过数据集的情形）；checkpoint 头里记的数据集 sha256 与配置里的不同就判对不上；
     显式要求 `prev-action` 而模型是纯图像（`prev_dim` 为 0）时判对不上。最后这一点在 `nav_cond.py` 里做成函数的一个参数（调用方说明这个取值是不是用户显式给的），
     `eval_closed_loop.py` 里 `--student-cond` 的缺省值从 `prev-action` 改成空的那一行不在这张卡里改（卡 W1-13 第二轮正在改这个文件），把要改的那一处写进报告。
  5. 第一轮 C6 的工具一侧（I2-5 第 2 点；C++ 一侧卡 C1a 已修，`e7ee56d`）：`average_ckpt` 的校准剔不剔碰撞步跟 checkpoint 训练契约里的 `drop_collision` 位走（现在恒剔），`.json` 里记下用的是哪种；
     `select_ckpt.py` 在没有 `config.json` 时读 checkpoint 的契约位，两处都没有就拒绝，不再告警后按不剔算。现有交付权重（都是 `drop_collision` 为真训的）重新校准的结果逐字节不变，用测试锁住。
  6. 训练进程等 GPU 时不自旋（卡 C1a 的 `25c1d03`：一个训练进程从 0.93 核降到 0.10 核，每轮耗时不变，权重逐字节相同）：三条流水线起 `uavrgb_train` 时环境里缺省带 `UAVRGB_CUDA_BLOCKING_SYNC=1`（外面已经设了就照外面的），
     `tools/exp_lib.sh` 的 `exp_start` 同样缺省导出；日志里记一行这个开关的取值。命令行与 dry-run 的输出逐字不变。顺手修 `exp_lib.sh` 时序健康那一行：被限流时长按核累计，除以墙钟会超过 100%，改成除以「墙钟乘核数配额」或者不报比例，选一种写明。
- 11-12 09:36 追加一条（审核 O-1 的代码部分；执行者交回这一轮之后续做）：`tools/train_student_with_video.py` 给 3DGS 评测拼命令时补上 `--data`（或 `--hold`），缺省的构建目录改成与别的工具一致；
  测试要真的走一遍 `eval_3dgs` 的参数解析（现在只查了打印出来的字符串，所以这个必定 rc=2 的路径没被发现）。

### 卡 P1c：原型线的选胜规则、对照臂与校准（审核 M-2、M-3、M-7、M-8；11-12 09:13 立；等卡 P1b 交回后由同一个执行者续做，在 E2E-P1 的筛选结果被采用之前合到 `p12-e2e-p1` 上）

- 只动 `proto/p1_summarize.py`、`proto/train.py` 与它们的测试（分支接着 `p12-e2e-p1b`）。筛选守护进程按旧规则自动选出的 B 组胜者不采用，改完之后用已经训好的权重重选；胜者变了才补训 C 组。
  1. M-2：B 组选胜先把候选权重校准到同一个前进占比再比，主要看学生驱动帧上转向帧的左右正确率与校准后的 macro-F1（F/L/R）；两臂的差在可检出量之内（见第 3 条）算分不开，分不开时取参数少的。
     预期 P3 的翻转率只在码非 0 的帧上算（码 0 的帧取反之后输入不变），有柱帧与空画面帧分开报。
  2. M-3：条件臂的报告加一行零训练的「只看码」基线（按净转向计数的反方向猜：E2E-0 的测试帧上 0.791，三值码 0.721；在本卡的 val 上重算）。胜出的条件臂在有柱的转向帧上要高于它，说明网络把图像与码合起来用了。
  3. M-7：所有臂间的差附按 val 几何簇（13 到 14 个）配对 bootstrap 的 95% 区间；差落在区间之内的不做硬性淘汰。预期 P4 同时列 B2 − B0 与 B2 − B1（后者才是只换头的差）。
     C 组的交付校准改成只在码 0 的帧上求、只加在码 0 的帧上（做法写进报告：FiLM 在全连接之前的结构可以把按码的偏置折进 FiLM 的偏移项）。补一个臂 C1b：三值码 + FiLM 放在第二层卷积之后 + 条件置空 0.25。
     三值码不用改主线就能进闭环，第二轮闭环用得上。
  4. M-8：权重 EMA 改成带步数修正的衰减（`min(decay, (1 + t) / (10 + t))`），A2 重训；或者保留 0.999 并在报告里写明 30 轮结束时还混着 14.2% 的初值、不据此否定 EMA。选前一种。
  5. 部署一侧的约束（审核 N-2）：FiLM 放在 p3 之前的臂（C3、C4、C1b）要在报告里给出部署办法（按码把 FiLM 折进该层卷积的权重，每个码一份主干；或者说明切图的位置与 CPU 一侧的算量），
     并报对称 int8 伪量化下转向帧的翻转率（照 `$D/tmp/phase12-cards/review2-n/quant_probe.py` 的口径，门槛 2%）。过不了门的结构不进第二轮闭环。

### 卡 DEP1：串口动作工具的四个毛病与接收端参考实现（审核 N-1、N-7；11-12 09:13 立，可以开工；不需要硬件）

- 只动 `tools/uart_action.py`、`tools/test_uart_action.py`、`config/uart_protocol.example.json`、`docs/uart_action.md`，可以新增 `tools/uart_codec.py`。分支 `p12-dep1` 从 main 起。
  1. N-1 的四条各一个提交，先用审核者的探针（`$D/tmp/phase12-cards/review2-n/uart_probe.py`）改写成会失败的测试：
     一次写入多行时每一行都按到达顺序发出；上游写半行时心跳不断；上游关闭管道（EOF）时发一帧 SAFE_NEUTRAL 之后退出，把链路超时交还飞控（退出码写进文档；加 `--upstream-timeout-ms`，这么久没有新动作就发 SAFE_NEUTRAL 并告警）；
     动作帧之后同一轮不再追发心跳。心跳帧的语义定为「只保活，不改当前动作与它的截止」，接收端忽略心跳帧里的动作字段；动作帧可带有效时长字段（卡 W1-13 第三轮定的真机语义：每条动作收到之后有效一段时间，新动作一到就取代）。
  2. N-7：接收端的参考实现（纯 Python，写法要能逐行对应成固件里无动态分配的 C：帧头滑窗重同步、CRC 校验、半窗口的序号回绕比较、心跳与动作的区分、失联倒计时）；
     示例配置缺省改成 CRC-16 与 2 字节序号（旧配置仍能加载）；用伪终端做故障注入测试：随机比特翻转、粘包拆包、丢帧、乱序、序号回绕、断流 350 ms 触发失联。
- 不做：上行回执帧的版式、固件一侧的代码、打开真实串口设备（这些等硬件的事实：ESP32 的型号、用官方固件还是改开源固件、板子的哪一路串口）；`sim/tools/collect.py` 里模拟丢帧的诊断开关（另排）。
- 验收：审核者的四个探针用例全部变绿；现有 `tools/test_uart_action.py` 的用例不改断言仍通过（单帧编码逐字节不变）；`check.sh --python-only` 通过。

### 卡 DEP2：参考前向、按 p3 切图的导出与 int8 门禁（审核 N-2、N-4；规格初稿，等卡 DEP1 交回后由同一个执行者做；不需要硬件）

- 新增 `tools/ref_forward.py`（纯 numpy，读现行版本的权重：三层卷积池化、FiLM、全连接、严格大于的 argmax；单帧、双帧、带条件都支持），与 `uavrgb_infer` 在真实样本上逐帧对拍（含集首帧）。
- 新增导出：主干到 p3 的 ONNX 图加一个侧车（FiLM 与全连接的权重、argmax 的规则、输入的视场与尺寸、条件的维度），外加 20 帧金标向量供板端程序自检。
- 新增 int8 门禁脚本（对称有符号 int8 伪量化，激活按最大值校准）：总动作一致率不低于 99%，转向帧（FP32 预测为转向）的翻转率不高于 2%；整网量化与按 p3 切图两种都报。
- `tools/check_input_parity.py` 扩到双帧与带条件。权重格式升版（卡 E2E-C1）之后侧车里的条件类型从权重里读，这张卡先从 `<ckpt>.json` 读。

### 卡 PERF1：Gazebo 闭环的三处无损提速（审核 P-2、P-3、P-6；11-12 09:18 立；等卡 W1-13 第二轮合并后开工，两张卡都动 `collect.py` 与 `eval_closed_loop.py`；由审核 P 的执行者做）

- 目标：每集省 9–10 s（K=1；满载时 13–17 s），每条车道省约 0.3 核，学生看到的图像、派发时刻、每步位移与转角都不变。在跑与已冻结的实验（E2E-R1、E2E-1、E2E-P0、W1-10）不受影响，它们用自己冻结的代码跑完。
  1. P-2：`eval_closed_loop.py` 在 `e1_mini` 且不录像时缺省不带辅助相机（留一个开关能带回来）；`collect.py` 读完相机内参就关掉内参话题的订阅。eval.json 里已有 `aux_cameras` 键，照实记。
  2. P-3：固件在环路径跳过 `SIM_SETTLE_S` 的静置；就绪探针合成一次；集末最短路的多终点搜索加剪枝，结果逐位相同（20 个冻结 seed 的 K=1 与 K=3 共 40 个场景上用测试锁住）；
     `collect.py` 加 `--no-shadow-teacher`（缺省关），`eval_closed_loop.py --policy student` 时传它，每步的动作、概率、时序不变（测试锁住），`action_expert` 一列为空。
  3. P-6：两条 Gazebo 流水线的定量评测缺省不开三视图录像（这一条动流水线文件，等卡 W1-15c 合并）。
  4. 记账：每集的 `performance` 里记采集进程与子进程的 CPU 秒、`gz sim` 的 CPU 秒（标准库，约 15 行），之后的实验能按臂算核数。
- 验证（不占 Gazebo 池的部分先做）：同一个 seed 带与不带辅助相机的静止首帧逐字节相同；最短路逐位相等；学生动作序列不变的单测。
  占 Gazebo 池的 A/B（入口先写好并用桩演练，排在今天队列的后面，由主会话放行）：同一个学生权重、20 个冻结 seed，改动前后各跑一遍，逐集配对。门：迟到步 0、停顿步 0；相邻观测间隔全部 0.900 s；
  步间空档中位数 0.100 s；巡航步位移与转角的均值差不超过 1%；成功数之差在同配置重评的波动之内（给出重评基线）。另报每集墙钟与每条车道的核数，对照审核者的估计。
- 协议键：带不带辅助相机现在不在协议一致性键里。A/B 过门之后维持「只记录不拦」，并在 `EVAL_STANDARD` 写明依据；过不了门就把它加进协议键，缺省改回带辅助相机。

### 卡 PERF2：订阅改成进程内、学生评测关深度相机、加车道（审核 P-4、P-5、P-7；规格初稿；等卡 W1-13 第三轮把「取哪一帧」改成按时间戳的确定性规则并重标定之后）

- 现在帧间隔恒为 0.900 s 靠的是图像管道慢：窗口末那一帧图像比位姿帧只早 5 ms 拍下，被 JSON 管道拖慢之后赶不上。图像一快（进程内订阅、少渲一路相机），取到的就是晚 0.1 s 的那一帧，步周期与步间空档都变。
  所以先在卡 W1-13 第三轮里把观测帧的选取写成按时间戳的规则（不依赖到达的先后），再做这张卡。
- 内容：`gz_io.py` 优先用进程内的 `gz.transport13` 订阅，没有绑定时回退；学生评测与只用真值位姿的教师臂不渲深度相机（`scene_gen.py --no-depth-camera`、`collect.py --no-depth`，训练采集保持原样）；
  每条车道降到 0.75 核上下之后实测 3 条、4 条车道（带并发的训练负载），迟到步与停顿步为 0、被限流时长不到墙钟的 5% 才改缺省车道数。

### 卡 T2：`check.sh --python-only` 的慢测试（审核 P-10；拆成两半：T2a 打包测试四个文件，PK1 合并后 11-12 09:52 派给 PK1 的执行者，分支 `p12-t2a`；T2b 3DGS 的 `test_collect_eval.py`，等卡 S1b 第二轮合并）

- 544 s 里 `sim/3dgs/test_collect_eval.py` 占 98 s、`tools/test_preprocess_*.py` 四个文件占 84 s，都是每个用例重新造数据并起子进程。改成类级夹具造一次、进程内调入口（各留两个真子进程的用例锁命令行的退出码）；
  `sim/tools/` 下标准 `unittest` 的文件合成一次解释器启动。不减覆盖（改前改后的用例名单与断言数对得上），不并行（`check.lock` 的预算是 1 核）。目标 180–220 s。

### 卡 DOC1：入口文档与现行代码对齐（审核 O-1 的文档部分、O-3、O-5、O-6、O-7、O-8；11-12 09:36 立；等清理分支 `p12-cleanup`、卡 W1-13 第二轮、卡 PK1 合并之后，由审核 O 的执行者做）

- 目标：新人照 `README.md` 从零做一遍（建环境、跑检查、编译、采一个小包、训一轮、跑一集闭环评测）每一步都能走通；入口文档里不再有与现行口径相反的说法。只改文档，不改代码（发现代码的毛病写进报告）。
  1. `README.md` 重写「从零跑通」一节：命令用 `$UAV_RGB_PYTHON`，输出写数据盘，载具用 `e1_mini` 加固件在环，评测带走廊判据与固定派发延迟，交付权重走 `tools/average_ckpt.py`；
     写明前置条件（显示服务、`CF-Drone` 源码、Gazebo 插件要开 `CF_SITL_BUILD_GZ`、`3dgs-venv` 没建时 3DGS 的测试会被跳过）。每条命令实际跑过（要占 Gazebo 池的那一条排队跑，或者注明没跑）。
  2. `docs/HANDOFF.md`、`docs/TASKS.md`、`docs/ROADMAP.md`、`sim/README.md`：顶上置一段现行入口的指引，旧章节标成历史；与现行事实相反的句子（唯一基线、没修的缺陷、CTest 项数、缺省 EGL）改掉或删掉。
  3. `docs/HANDOFF_3DGS_20261105.md` 的第 4、5、6 步改成能跑通的命令。
  4. `docs/EVAL_STANDARD.md` 补上走廊判据、停顿集、逐块存活、每百米碰撞数、断帧旗、交付口径；`tools/DATASET_FORMAT.md` 更正 `split_policy` 的合同，写上 C++ 读端的两条硬约束与两个过滤开关。
  5. 审核报告第 4 节的六组文档逐组处置（删、移进归档目录、或在顶上标历史），处置之后入口文档里的链接跟着改；因为撤掉链接而变成无引用的旧文档按清理任务书的规则一并删。
- 验收：报告里给一张表，逐步列「文档写的命令、实际跑的结果」；`grep` 不到裸 `python3` 加 `/tmp/` 的例子；入口文档无死链。

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
