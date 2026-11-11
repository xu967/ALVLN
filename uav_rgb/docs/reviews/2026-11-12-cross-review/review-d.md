# 交叉审核 D：数据链路与统计（main `97c4228`）

审核者：角度 D。只读；所有草稿脚本与日志在 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review-d/`，下文写「草稿 X」即指该目录下的文件。上一轮 `docs/reviews/2026-11-10/data-algo.md` 已报的不重复，仍未修的标明。

## 1 结论

最要紧的三条：

1. **3DGS 端到端（E4）会在训练场景上评测，代码不拦。** 9 个评测场景按 `collect_3dgs.scene_identity` 的哈希划分是 8 个 train、1 个 val、0 个 test；`eval_3dgs.py` 把 split 写死成 `"test"`，不读也不比对 identity 里的 split。E4 用 `collect_3dgs` 在这些场景采数、再用 `eval_3dgs` 在同一批场景评，就是场景级泄漏，K3 的「留出路线」只是同场景内的路线留出。
2. **60 集主比较把 60 对当独立样本，几何聚类被忽略。** 20 个几何 × 3 个训练种子；按几何做符号翻转精确检验后，交付配方的核心证据「calswa 对 ckpt_best 20 对 8，p=0.036」变成 p=0.089，「只校准对 ckpt_best」0.078 变成 0.139。公共规则 §2 与 `EVAL_STANDARD.md:319` 规定的就是 60 对合并 McNemar，所以这是口径问题，写进计划的 p 值要重算。另外「平均」这一半没有闭环证据：calswa 对 calbest 12 对 9，p=0.664。
3. **航向护栏的阈值、规则 v1 的设计都看过冻结 test seed 上的轨迹，再在同一组 seed 上报结果。** 护栏的依据是 eval4 学生轨迹的分析（eval4 的 seed 就是 `EVAL_SEEDS_BASE_20`），规则 v1 是看过 v0 在这 20 个 seed 上的失败集后设计的。47/60、50/60、57/60 这几个数是样本内的。`EVAL_SEEDS_BASE_60` 里另外 40 个 seed 没参与这两项设计，可以拿来做样本外复评。

## 2 发现（按严重度）

### D1【高】3DGS 评测场景全部不在 test 分区，`eval_3dgs` 强写 split="test" 不校验
- 位置：`sim/3dgs/eval_3dgs.py:219-223`（注释说明 split_policy=diag-eval-fixed-test）、`:248`（`split="test"`）、`:328`；`sim/3dgs/collect_3dgs.py:76-99`（`scene_identity` 用 `sm.split_for_gid(gid, gver)` 给 split，`:98`）。
- 现象：草稿 `scene_splits.py` 按 `collect_3dgs` 的同一公式（PLY sha256 加 nav_to_world 得到 gid，salt = geometry_version）算 9 个评测场景：sage3d_839992 = val；garden、bicycle、callinstrasse、wanderland_1MlUucug、sage3d_841012、sage3d_840865、sage3d_840139、sage3d_841150 = train；没有一个是 test。`eval_3dgs` 的 per-episode 记录一律写 `split="test"`，identity 里的真实 split 不参与任何判断。
- 为什么是缺陷：计划 §10.4 的 E4（3DGS 室内条件模仿）走 `collect_3dgs` 采数 → `preprocess` 按 identity split 打包 → 训练只用 train 分区，而 8/9 场景正好在 train 分区。之后用 `eval_3dgs` 在同几个场景评测，场景外观、占据栅格全都见过，得到的成功率是训练集成功率，报告里却标成 test。K3 的留出路线只防同一路线重复，挡不住这个。
- 复现：`PYTHONDONTWRITEBYTECODE=1 .venv-tools/bin/python review-d/scene_splits.py`（12 s，只读 PLY 头与 sha）。
- 最小修法：`eval_3dgs` 读 identity["split"]，不是 test 时默认拒绝，必须显式 `--allow-train-scene` 才放行并把 split_policy 写成 `diag-train-scene`；E4 采数前先定一份场景级留出表（或换一个能把评测场景划进 test 的 salt，并把 salt 写进 meta），任务卡写明「评测场景不进训练」。
- 状态：划分结果已复现；泄漏后果是推断（E4 还没开工）。

### D2【高】60 集主比较的 McNemar 与 Wilson 忽略几何聚类，计划里的 p 值偏小
- 位置：口径来源 `docs/briefs/2026-11-10-phase12-common.md:24`、`docs/EVAL_STANDARD.md:319`；实现 `experiments/2026-11-10-swa-calib/summarize.py:120,134`、`experiments/2026-11-10-heading-guard/summarize.py:105` 及其配对段、`experiments/2026-11-10-rule-oracle-v1/summarize.py`；引用处 `docs/PLAN_2026-11-10_phase12.md:89`、`docs/PLAN_2026-11-10_phase11.md:81`。
- 现象：同一几何的 3 个训练种子结果正相关。草稿 `cluster_stats.py`（日志 `cluster_stats.log`）按几何算组内相关 ICC：best +0.28、calbest +0.26、calswa +0.21、guard2-best +0.23。配对检验改为「每个几何把 3 个训练种子的差求和，再做符号翻转精确检验」后：

  | 比较 | 60 对 McNemar p | 几何符号翻转 p |
  |---|---|---|
  | calswa 对 best（20:8） | 0.0357 | 0.0887 |
  | calbest 对 best（15:6） | 0.0784 | 0.1390 |
  | guard2-calswa 对 calswa（15:4） | 0.0192 | 0.0415 |
  | guard2-best 对 best（23:3） | 0.0001 | 0.0034 |
  | calswa 对 swa（28:9） | 0.0026 | 0.0022 |
  | rule60v1 对 guard2-calswa（8:1） | 0.0391 | 0.0312 |

  单臂区间：best 27/60 的 Wilson [0.33,0.58]，按几何整簇 bootstrap [0.30,0.62]；calbest [0.47,0.71] 对 [0.45,0.75]。
- 为什么是缺陷：计划 §0 把「权重平均 + 校准」写成已证实的交付配方，依据是 p=0.036。按几何聚类后不过 0.05；「只校准」不过 0.1。平均这一半：exp-swa-calib `summary.md:38` 的 calswa 对 calbest 12 对 9，p=0.664，闭环里看不出平均有额外作用。规则 v1 的 summarize 已经写了「60 对不独立，p 只作参考」，但计划正文没有照这个口径降级。护栏相对 best 的提升（0.0034）与 calswa 相对 swa 的差（0.0022）在聚类口径下依然显著，结论不变。
- 复现：`PYTHONDONTWRITEBYTECODE=1 .venv-tools/bin/python review-d/cluster_stats.py`（读各实验 eval.json，几秒）。
- 最小修法：公共规则 §2 与 EVAL_STANDARD 的主检验改成「按几何聚合的配对符号翻转（或置换）精确检验」，区间改成按几何整簇 bootstrap；60 对 McNemar 只作附表。计划 §0 里 calswa 的表述改成「方向一致，几何层面 p=0.089，未达显著」，并注明平均部分无闭环证据。
- 状态：已复现。

### D3【中】护栏阈值与规则 v1 都看过冻结 test seed 的轨迹，再在同一组 seed 上评
- 位置：`experiments/2026-11-10-heading-guard/run.sh:3-5,27,30`（依据是 data-algo 报告 B.2、D1 对 eval4 学生轨迹的分析；`GUARD=2` 的注释引用了教师臂与阵内成功集的 |n|≥2 比例）；`experiments/2026-11-10-rule-oracle-v1/run.sh:18,51`（「规则 v1 由主会话在看过 v0 的失败集之后设计」，评测 seed 即 `EVAL_SEEDS_BASE_20`）。
- 现象：`/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417-eval4/eval-forward/eval.json` 的 seed 集合正是 `tools/eval_seeds.py:20` 的 20 个冻结 seed（eval4-extra 也是这 20 个）。护栏的规则形式（|n|≥2 朝回转）、阈值 2 都来自这些轨迹，然后在同 20 个 seed 上跑出 47/60、50/60。规则 v1 的参数虽然冻结在起跑前、冒烟只用 train 分区，但规则结构是对着这 20 个 seed 上 v0 的失败集改出来的，57/60 也在同一组 seed 上。
- 为什么是缺陷：违反公共规则「开发与调参只用别的 seed」。这些数字被计划当成「缺航向是出阵主因」「可判定的规则能到 95%」的证据，进入 E2E-1 的立项理由。样本内估计偏乐观的幅度未核实。
- 复现：`python3 -c "import json;e=json.load(open('/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417-eval4/eval-forward/eval.json'));print(sorted({x['seed'] for x in e['per_episode']}))"`，与 `tools/eval_seeds.py:20` 对比。
- 最小修法：护栏（guard2-calswa）与 rule60v1 在 `EVAL_SEEDS_BASE_60` 去掉 BASE_20 后的 40 个 seed 上各补跑一次（这 40 个没参与两项设计），作为样本外数字写进计划；以后基于失败集的设计改动在任务卡里登记「看过哪些 seed」。
- 状态：seed 重合已核实；偏差幅度未核实。

### D4【中】`tools/stats_report.py` 的多 run 格区间偏窄，多 run 组不做配对
- 位置：`tools/stats_report.py:174-198`（`stratified_bootstrap_ci` 只在 run 维度重采样）、多 run 格调用处约 `:286-290`、`build_report` 中多 run 组跳过配对约 `:361-367`。
- 现象：3 个训练种子的格子，区间只来自 3 个 run 成功率的重采样，不含每个 run 内 20 集的二项噪声。草稿 `sr_multirun.py`（日志 `sr_multirun.log`）直接调用该函数：calswa（12/12/15）区间 [0.600,0.750] 宽 0.150，60 集 Wilson [0.524,0.758] 宽 0.235；guard2-calswa（15/17/18）[0.750,0.900] 宽 0.150，Wilson 宽 0.187。3 个 run 时重采样均值只有 10 种取值，区间覆盖不足。主比较（3 种子 × 20 几何）是多 run 组，`build_report` 直接不出配对检验，所以共享工具实现不了公共规则规定的主比较；各实验只好各自复制 wilson/mcnemar（swa-calib、heading-guard、rule-oracle-v1 三份）。
- 为什么是缺陷：DAgger 流水线（`tools/dagger_gz_pipeline.py:693`）调用 stats_report 出完整表，多 run 臂的区间会比实际不确定度窄；以后的端到端比较若改用该工具，主检验缺失。
- 复现：`PYTHONDONTWRITEBYTECODE=1 .venv-tools/bin/python review-d/sr_multirun.py`（<1 s）。
- 最小修法：多 run 格改为「run 内按集、run 间按几何」的两级 bootstrap，或直接用按几何整簇 bootstrap；多 run 组按 (几何, 训练种子) 对齐后做 D2 的几何符号翻转检验。
- 状态：已复现。

### D5【中，上一轮 C1 仍未修】`stats_report` 的 Wilson 分母仍是完成集，`metric_values` 按 seed 作键会覆盖
- 位置：`tools/stats_report.py:8`（文档写明分母 = episodes_done）、`:244`、`:417`；`:263` `out[ep.get("seed", i)] = v`。
- 现象：公共规则要求分母用请求集数，stats_report 仍用 episodes_done（上一轮已报，未修）。另外同一 eval.json 里 seed 重复（同 seed 多次重复的协议）时，后一集覆盖前一集，配对与比例都会少算，代码不报警。
- 影响：失败集多的臂成功率被抬高；重复 seed 协议下配对样本静默变少。
- 复现：读代码即可判断；未跑。
- 最小修法：分母取 episodes_requested，失败集按失败计；键改为 (seed, 重复序号) 或 episode_id，发现重复键时报错。
- 状态：读代码判断（推断），分母一项与上一轮结论一致。

### D6【低】`average_ckpt` 的校准只拟合全局 FORWARD 占比，对航向码模型约束不到关键帧
- 位置：`tools/average_ckpt.py:7`（校准目标说明）、`:131,137,141`；条件输入拒绝在 `:206-212`（`prev_dim/text_dim/in_c` 不支持，已在 E2E-1 步骤 2 计划内，不算新问题）。
- 现象：δ 是单个标量，目标是 val（教师状态、剔除碰撞步）上 FORWARD 标签占比。E2E-1 的航向码模型真正要学的是 |n|≥2 时朝回转，而教师臂只有 2.7% 的步 |n|≥2（`heading-guard/run.sh:30` 的注释），这些帧在 val 里比例小，全局 FORWARD 占比几乎不受它们影响。
- 影响：E2E-1 若照搬「平均 + 校准」配方，校准对码 2/3 帧上的行为没有约束，可能把这些帧上的 FORWARD 也一起推高。
- 最小修法：E2E-1 解禁条件输入时，按码分组报校准前后的 FORWARD 占比；若码 2/3 帧上的 FORWARD 占比偏离标签，改为分码校准或不校准。
- 状态：推断，未跑。

### D7【低】各实验 summarize 的分母口径不一致，当前数据上没有数值影响
- 位置：`experiments/2026-11-10-swa-calib/summarize.py:95,120`、`experiments/2026-11-10-heading-guard/summarize.py:105` 用 `len(per)`/`len(rows)`；rule-oracle-v1、rule-oracle-3dgs、bearing-guard-3dgs 用请求集数；`experiments/2026-11-10-teacher-tol-3dgs/analyze.py:92-93,267` 用「写了 done.json 的路线数 × k」。
- 现象：前两份用完成集作分母，teacher-tol 的分母会把没写 done.json 的路线整条排除。
- 已核对：swa-calib、heading-guard 各臂 failed=0；teacher-tol 的 acceptance.json 两臂 n_req=120、failed=0（4 场景 × 10 路线 × 3）。所以已发布数字不受影响。
- 最小修法：三份脚本统一用请求集数，缺 done.json 的路线按失败计并在表里列出。
- 状态：已核实无数值影响。

## 3 查过没问题的地方

- **标签与图像对齐**：`tools/preprocess.py:259-267` 的 prev_action 取上一行 action_executed，首帧 0；aux 的 prev_index 在上一帧被丢时指向自己（`:859-861`）。与 `sim/tools/student.py`（reset 后 _prev_rgb=None、_last_action=0，PIL 双线性缩放转 RGB）和 `collect.py`（每集重置 prev_executed）一致。`tools/DATASET_FORMAT.md` §1.1、§6 与代码里的结构体字段一致。
- **Gazebo 柱阵的 train/eval 隔离**：split 由 geometry gid 哈希决定（`sim/tools/split_manifest.py`，80/10/10），test 几何即使被采进数据集也落在 test 分区，训练只读 train 分区，所以 DAgger 不会把评测几何学进去；K=3 课程的 block 0 与基础几何同 gid，spawn 不依赖柱子（`make_spawn` 默认值），配对成立。
- **航向码时序（工作树 `wt/p12-w1-9`，未提交改动）**：`sim/tools/nav_cond.py:43-49` 的 `heading_codes` 先取码再累加，第 t 步的码只用第 0..t−1 步的已执行动作；`collect.py:963-966` 取 `heading_code(net_yaw)` 后才执行动作并更新 net_yaw（main `collect.py:1816-1817`）。两侧口径一致，无错位。
- **W1-0 动作延迟验收 4**（`experiments/2026-11-10-action-latency/summarize_r2.py` 与 `long-course/summarize.py`）：草稿 `w10_recount.py` 独立复算 K=1 53/60、K=3 第 1 块 36/60，差 +0.283，与 summary 一致。K=1 用块判据重算也是 53。有 14 集 K=3 在过第 1 块线后立刻碰撞，被计为过块，即 K=3 判据偏宽，差值没有被夸大。`lc.load_arm` 把失败集记为 aborted，分母是请求集数。学生臂没带 `--rtf 0.5`，但各 acc 臂 late_dispatch 均为 0，不影响结论。
- **K1 rule-oracle-3dgs 正式配对**：草稿 `k1_recount.py` 复算室内 rule-occ+route 80/120（碰撞 4）、室外 117/120、forward+route 72/120；各臂 120 个唯一键，success_stop 与「到点早于碰撞」一致；碰撞率 52/120=0.43 → 4/120=0.03 与计划相符。
- **teacher-tol-3dgs**：分母与配对键核对过，两臂 120/120 完整。
- **stats_report 的公式本身**：精确 McNemar、Holm、Cochran Q、功效近似公式逐行核对，无误（问题在 D4、D5 的用法与分母）。
- **rule-oracle-v1 summarize**：分母为请求集数（course3 17 完成 + 3 失败），已注明 60 对不独立。

## 4 没查的地方与原因

- `sim/tools/audit_dataset.py` 只看了与 aux、碰撞丢帧相关的部分，label_recompute 的全部分支没有逐条核对（时间）。
- `analyze_round1.py`、baselines-zeroshot、rule-oracle v0 的 summarize：结论已被后续实验取代或不在本阶段主线上，未看。
- C++ 侧 `--drop-collision` 与 aux prev_index 的交互：只从 Python 侧推断一致，C++ 打包代码未读。
- bearing-guard-3dgs 的 `--gate` 场景选择模式是否引入选择偏差：看到了，没有深入。
- 没有跑任何 Gazebo、训练或 `tools/check.sh`；D3 的样本内偏差幅度需要补跑 40 个 seed 才能知道，按只读约束未做。

## 做了什么 / 没做什么 / 验证到哪一步

- 做了：读任务书与上一轮报告；逐个读角度 D 范围内的脚本；写 6 个只读草稿脚本（`scene_splits.py`、`cluster_stats.py`、`cluster_stats2.py`、`sr_multirun.py`、`k1_recount.py`、`w10_recount.py`）并保存日志；每次只跑一个进程，`nice -n 15`，均在 1 分钟内结束。
- 没做：未改仓库任何文件，未提交，未建分支，未占锁，未动进程；未补跑 D3 的样本外评测（只读约束）。
- 验证到：D1 的划分、D2、D4 的数字已复现；D3 的 seed 重合已核实；D5、D6 为读代码推断；D7 已核实无数值影响。
