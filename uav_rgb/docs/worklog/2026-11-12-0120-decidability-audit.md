# 可判定性审计（卡 E2E-0）：真值特征提取与对账、特征组拟合、分箱核对、与预期对照

- 时间：2026-11-12 00:19 写出首个测试（红），00:36:46 提交 `37ff4eb`（含起跑前写定的预期与读法），00:36:50 起跑，00:57:34 首次运行结束，01:00:56 续跑 bins 与 report（01:01:05 结束），01:20 起写本文
- 范围：新增 `experiments/2026-11-12-decidability-audit/`（`run.sh`、`audit_features.py`、`audit_fit.py`、`audit_report.py`、`recheck_tables.py`、`mech_check.py`、三个测试文件、`README.md`、`RESULT.md`）与本文；`sim/tools/`、`tools/`、`src/` 下的库文件没有改动
- 分支：`p12-e2e-0`（基于 main `8378762`；main 之后由其他会话的提交前进，写本文时为 `f6c159f`；本分支未合并、未推送）
- 结论：完成。4 种拟合配置共 84 次拟合已跑完（`pooled_w` 24、`pooled_u` 24、`teacher_w` 18、`student_w` 18），结果在 5 个测试子集上取数，报告与结果表已写出。学生驱动的帧上，只看当前画面距离（S1）判不出转向方向（转向帧左右对 0.508，95% 区间 [0.411, 0.601]），补上 n（S3）到 0.861（[0.768, 0.921]）；S5 比 S4 的 macro-F1 高 0.089（[0.053, 0.134]），落在预期的「不到 5 个点」与「10 个点以上」之间。逐条对照、没做什么与待定事项见报告 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/e2e-0.md`。

## 做了什么

- 读卡 E2E-0、计划 §10、公共规则、`sim/tools/rule_policy.py::column_distances`、`sim/tools/collect.py` 写出的字段、`tools/preprocess.py::load_episode`、`sim/tools/split_manifest.py`、`docs/reviews/2026-11-10/data-algo.md`。发现卡与数据有一处出入：柱子真值在每集的 `scene.json`，不在 `episode_manifest.json`；动作前位姿在 `actions.jsonl` 的 `pose_before`、`yaw_before`，`oracle.jsonl` 的 `pose_gt` 是动作后的位姿（已写进 README 与报告）。
- `audit_features.py`：选步直接调 `preprocess.load_episode(drop_collision=True)`，列距离直接调 `rule_policy.column_distances`；特征 `d`（20 列）、`d_wide`（60 列）、`n`、`hist`、`priv`，另算三值航向码与真实航向 sin、cos（卡外附加的 S3c、S7）；历史量（n、hist）按该集全部步算，被丢弃的碰撞步不当样本，但仍计入「已执行」；提取后与 5 份打包日志逐集、逐 split、逐类对账，对不上就停；抽 100 帧核对 `d` 与 60° 视场直接调库函数的结果逐位相同。
- `audit_fit.py`：8 个特征组的输入矩阵、5 个子集掩码、2×64 的 ReLU MLP（torch，CPU 单线程，Adam，类别权重 N/(C·n_c)，train 拟合，val 上同一个交叉熵最低的那一轮停止，patience 10，最多 80 轮）、numpy 实现的混淆矩阵与指标（整体准确率、macro-F1、每类召回、该转时转了、转向帧左右对、前进帧被判成转向）、按几何簇的混淆矩阵、不依赖拟合器的分箱核对。4 种配置：`pooled_w`（主，8 组）、`pooled_u`（不加权，权重全为 1，8 组）、`teacher_w`、`student_w`（只用对应子集的训练帧拟合，只拟合卡里的 6 组）。
- `audit_report.py`：测试几何簇自助法 1000 次的 95% 区间（三个种子取均值，特征组之间配对）、与起跑前预期逐条对照（含换配置、换指标、换子集的读数）、`tables.md` 与 `results.json`。01:00 补了主表的区间表、`pooled_u` 的区间表与相邻组差值表（提交 `c9a1ef7`，含 `interval_rows` 的测试）。
- `run.sh`：守护进程入口（`setsid nohup`，`run.pid`、`run.log`、`DONE`、`.lock` 互斥），每个阶段 1 个 python 进程、1 线程、`nice 10`，CPU only；头部原样抄入卡的预期，并写定三处卡没写清的口径与 E1 到 E4 的判定读法；这些随 `37ff4eb` 在 00:36:46 提交，早于起跑的 00:36:50，起跑时工作树无改动（`inputs_run1.txt`）。
- 测试：`test_audit_features.py` 19 项（手算的合成场景：正前方柱、镜像场景、左转后柱的列位置；卡里的例子 F L L F R 的 n 与 hist；碰撞步被丢但仍计入历史；打包日志对账的各种不一致）、`test_audit_fit.py` 20 项、`test_audit_report.py` 9 项，先红后绿；对特征与对账模块做了 14 种突变检查。
- 试跑：只含 trunc 的干跑（提取、对账、拟合 2 个种子最多 6 轮、分箱、报告）三阶段连通，`dry/tables.md` 411 行，00:33:57 结束。干跑只含 trunc 的帧（教师驱动），没有学生驱动的帧，所以起跑前没有看过学生驱动帧上的任何拟合结果（`run.sh` 头部同样写明）。
- 正式运行：提取 8 分 51 秒、拟合 11 分 46 秒、bins 与 report 合计 3 秒，`run.log` 第 139 行 `全部结束 rc=0`（00:57:34）。01:00:56 续跑（跳过 extract 与 fit），`DONE` 改写为 `rc=0 2026-11-12 01:01:05`，`results.json` 与首次逐字节相同。
- 独立核对：`recheck_tables.py` 只用 `features.npz` 与 `preds_*.npz`（不调 `audit_*`）重算 `pooled_w`、`pooled_u` 全部 8 组在三个子集上的整体准确率、macro-F1（F/L/R）、该转时转了，并解析 `tables.md` 的主表逐格比对（144 格，不一致 0 个；对一份改了一格的副本报 1 个），另打印相邻组的 macro-F1 差值，结论段引用的 macro-F1 与差值都与之相同；`mech_check.py` 核对转向标签的方向与 n 的符号的关系（`RESULT.md` §7 引用）。
- 文档：`README.md`（口径、协议、指标、局限、复现）、`RESULT.md`（结果表，表由 `tables.md` 原样摘录）、本文、报告 `e2e-0.md`。
- 提交：`37ff4eb`、`9885e16`、`79a9f38`（起跑前）；`c9a1ef7`（报告阶段补区间表）；`98efab2`（两个核对脚本）；`e775def`（`RESULT.md` 与 README 补文件说明）；`450981b`（本文初版）；`26f50f2`（`recheck_tables.py` 扩到 144 格逐格比对）；本文的更新是最后一个提交。逐文件 `git add`，没有用 `git add -A`。

## 没做什么

- 没训练学生、没起 Gazebo、没改库代码。原因：卡的边界。
- 没用 GPU。原因：2×64 的 MLP，单线程每次拟合 1.8 到 20.9 秒，84 次合计 704 秒（`fitmeta_*.json` 的 `wall_s`）；GPU 对这个规模没有收益，也避免占显卡。
- 没装包、没用 sklearn。原因：`teacher-venv` 是只读借用的环境；指标与自助法用 numpy 实现。
- 没拟合卡外的特征组 {d, 真实航向角}。原因：卡里的六组不许改，S3c 与 S7 已是卡外附加，再加会扩大范围；角度码对步数码（计划 §10.7 第 6 条）只有 S7 − S6 一条间接证据，是否补由主会话定。
- 没模拟视觉前端的噪声。原因：卡定义 `d` 为完美视觉前端读出的几何，这里的可判定率是上界。
- 没拆分「不可判定」的来源（特征缺信息、教师在近平局帧上的 tiebreak 跳变、MLP 拟合不到的结构）。原因：需要对教师做扰动或另做可观测性受限的教师，超出卡的范围。
- 没做换布局族或新几何的泛化检验。原因：只有这些现成采集，卡只要求按几何分组切分。
- 没重复 data-algo B.2 的 kNN。原因：卡写明已有结论不重复；报告 §5 并列了两边的读数，两边的测试帧是否重合未核实。
- 没对拟合与报告两个模块做突变检查（只做了特征与对账模块）。原因：时间盒；这两个模块靠手算期望值的测试，以及对表里数字的独立重算。
- 没有独立重算「转向帧左右对」。原因：它用 L、R 两个 logit 比，`preds_*.npz` 只存 argmax，要重算得重新拟合。旁证有两条：分箱口径独立算的左右对（S1 0.497、S3 的三值码 0.793）与 MLP（S1 0.508，S3c 0.803）同量级；`audit_fit.py` 里这一指标有手算期望值的测试（`test_lr_ok_counts_only_turn_frames`、`test_metrics_from_counts_by_hand`）。
- 没给 `recheck_tables.py`、`mech_check.py` 写单元测试。原因：它们本身是对 `audit_*` 输出的独立重算与计数，输出与表逐位相同。
- 没把文档装配脚本入库。原因：`RESULT.md` 与报告的表由一个一次性脚本从 `tables.md` 原样摘录，脚本在 `$D/tmp/phase12-cards/e2e-0/docs/`，不属于分析流程；`RESULT.md` 开头已写明表摘自 `tables.md`。
- 没跑 `tools/check.sh`。原因：没改任何库文件，它只收 `sim/tools/test_*.py` 与 `tools/test_*.py`，不收 `experiments/` 下的测试；三个测试文件单独跑，结果见下。
- 没做多重比较校正。原因：区间是逐格 95% 区间，读的时候看方向与量级，不按单个阈值的穿越下结论。
- 没改 `progress.md`、`CLAUDE.md`，没合并、没推送，没动主仓库工作树，没删除任何既有文件、分支或工作树。原因：卡的边界。

## 待确认

- S5 比 S4 的 macro-F1 高 0.089（加权）、0.038（不加权）、0.074（只用学生帧拟合）。没有一个读数落在稳定的「10 个点以上」，也没有稳定落在「5 个点以下」。要不要把它当成需要循环状态的依据，需要主会话定。
- E2 起的条件码内容：三值航向码比 9 档的 n 左右对低 0.057（[−0.104, −0.017]），macro-F1 低 0.009（区间含 0；不加权低 0.027）；hist 在学生帧上没有增益，在教师帧与画面空的帧上补 5 到 12 个点的方向正确率。是否加宽由主会话定。
- 拟合池与加权：E3 的判定随之翻转（加权不成立，不加权成立）。卡后续版本要不要写明，需要定。
- round_1（β=0.5，1,644 帧由专家执行）归入学生驱动是按卡原文；纯学生执行的 round_2、round_3 作为附加子集，读数同方向。
- 要不要补 {d, 真实航向角} 一组（估计约半小时，未实测）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `test_audit_features.py` | 先红 `ModuleNotFoundError: No module named 'audit_features'`（00:19:05）；后绿 `Ran 19 tests in 0.046s OK` |
| `test_audit_fit.py` | 先红 `ModuleNotFoundError: No module named 'audit_fit'`（00:27:34）；中间一次 `AssertionError: 0.0 not greater than 0.3`、`FAILED (failures=1)`（00:30:16，测试前提有误，见下）；后绿 `Ran 20 tests in 3.375s OK` |
| `test_audit_report.py` | 先红 `ModuleNotFoundError: No module named 'audit_report'`（00:31:40）；补 `interval_rows` 的测试后先红 `AttributeError: module 'audit_report' has no attribute 'interval_rows'`、`FAILED (errors=1)`（01:00:14）；后绿 `Ran 9 tests in 0.059s OK` |
| 三个测试的环境 | 后绿一栏是 HEAD `e775def`（之后的提交只改了 `recheck_tables.py`、README 与本文，三个测试文件和被测的三个模块没有变），`env` 里没有 `AUDIT_*` 变量，`teacher-venv` 的 python，`nice -n 10`；日志 `$D/tmp/phase12-cards/e2e-0/test_{features,fit,report}_green_final2.log`；HEAD `c9a1ef7` 上的同一组测试为 0.044s、3.399s、0.057s，全绿 |
| 特征与对账模块突变检查（14 种） | 最终一轮 `漏网 0 / 14`（`mutants_features.log`）；首轮有 2 种漏网，其一是对账漏 split（补了 `test_check_against_log_catches_swapped_splits_even_when_totals_agree`），另一种补了 1 处消息断言；首轮日志已被覆盖，另一种漏网的名称未保留 |
| 只含 trunc 的干跑 | 三阶段连通，`dry/tables.md` 411 行 |
| 5 份打包日志对账 | 全部通过（日志是累计口径）：trunc 300 集 9663 样本；more_demos 640 集 20491；round_1 400 集 13018；round_2 500 集 16693；round_3 592 集 20291；去重后 932 集 31119 帧（`run.log` 第 25 到 31 行） |
| `d` 与 60° 直接调库函数 | 100 个真实帧逐位相同（`run.log` 00:45:44 前后） |
| 正式运行 | `run.log` 第 139 行 `全部结束 rc=0`（00:57:34）；84 次拟合没有一次的最佳轮数顶到 80 轮上限（`tables.md` §7），最佳轮数 11 到 78，另有 10 次拟合跑满 80 轮（最佳轮数在 71 以后） |
| 续跑 | `DONE`：`rc=0 2026-11-12 01:01:05`；`cmp results.json results_run1.json` 无输出，sha256 前 20 位同为 `d940117736fc390e8d1f` |
| 表里数字的独立重算 | `recheck_tables.py`（`26f50f2`）：`检查 144 个格子（应为 2 × 3 × 8 × 3 = 144），不一致 0 个`；对改了一格的副本：`不一致 1 个`（`('pooled_w', 'student', 'S3', 'acc'), '重算 0.751，表 0.752'`）；日志 `recheck_tables.log`、`recheck_tables_neg.log`；副本目录 `$D/tmp/phase12-cards/e2e-0/neg_control/` 只含指向 `features.npz`、`preds_*.npz` 的符号链接和改了一格的 `tables.md` |
| 正文数字的出处 | 报告与 `RESULT.md` 里的三位小数，除 0.157（分箱口径 S3 与 S1 留出准确率的差，0.773 − 0.616）与测试耗时（0.044、0.057、3.399 秒）外，都能在 `tables.md` 或两份核对日志里找到；这个检查只核数字是否出现过，不核它出现在哪个格子，格子的对应关系靠逐条人工对表 |
| `git diff --stat 8378762..HEAD`（本文提交前） | 11 个文件，全在 `experiments/2026-11-12-decidability-audit/` 下，无删除；本文提交后只多 `docs/worklog/` 下的本文 |
| 主仓库工作树 | `git -C /root/claude-code-backup-20261028/code/uav_rgb status --short` 无输出 |
| 残留进程 | `ps -g 341522`、`ps -g 485386`（两次运行的 PGID）都没有进程 |
| `tools/check.sh` | 未跑，原因见「没做什么」 |

过程中出过的问题：

- `test_class_weight_moves_the_decision_toward_the_rare_class` 的前提有误：权重恰好抵消先验的构造里，平局处的 argmax 是噪声。改成比较 R 与 L 两个 logit 的平均差（`test_class_weight_shifts_the_logits_toward_the_rare_class`）。这是测试的问题，不是实现的问题。另有两处测试自身的问题，都改了测试：`test_bin_check_by_hand` 的手算注释与夹具对不上；`fmt_ci` 的用例落在四舍五入的边界上，改成 0.0457。
- 核对脚本最初从环境变量 `AUDIT_OUT` 取输出目录，我的 shell 里残留着干跑时导出的 `AUDIT_*`（`AUDIT_OUT` 指向 `dry/`、`AUDIT_FORCE=1`），脚本读到了干跑的数据而失败；我随后用 `mv` 把两份核对日志换成了失败的输出，再清掉变量、改成命令行参数、用入库版本重新生成日志，数字与之前逐位相同。
  正式运行不受影响：`run.sh` 自己导出全部 `AUDIT_*`，续跑日志里 fit 阶段 1 秒结束（跳过已有的 stats），`results.json` 逐字节不变；三个测试文件在清空环境后重跑仍为绿。
- 等待时在 shell 里用过两次 `sleep`，之后改用 `AwaitShell`；几处文件编辑用了一次性的 python 补丁脚本而不是编辑工具，内容已逐处核对。

## 风险与遗留

- 结果是同一生成器下的同分布读数（几何簇 541 个：训练 462、验证 39、测试 40），换布局族的可判定性没有数据。
- 学生驱动的测试帧只有 925 帧、19 个几何簇，S3 的左右对区间 [0.768, 0.921] 跨过 0.85；E4 的区间 [0.053, 0.134] 跨过 0.10。要更窄的区间需要补采学生驱动的帧。
- 拟合池与加权方式改变读数：E3 翻转，E4 的差在 0.038 到 0.089 之间。结论里已并列给出。
- 分箱核对只支持 S1 到 S3 的跳升，不重现 S4 到 S5 的增益（箱划分不可比），S5 的增益只有 MLP 一个口径支持。
- `student_w` 的 S3 与 S4 三个种子的最佳轮数相同（22 / 64 / 54），验证损失不同（0.5806 对 0.5538，0.5330 对 0.4821，0.5099 对 0.4996），输入维数 29 与 41，两组的统计量在表里也不同，所以不是同一次拟合的重复；轮数相同的原因未核实（猜测是同一种子下批次顺序相同，验证损失由权重 37.6 的 SAFE 类少数几帧主导，未验证）。
- `run.log` 是追加写的，首次与续跑的记录在同一个文件里；首次运行用的输入与产物另存为 `inputs_run1.txt`、`tables_run1.md`、`results_run1.json`。
- `run.pid` 里的 485386 是已结束的 PGID，容器重启后可能被复用；停进程前先核对。
- 根分区剩 3.0 GB（`df -h /`，01:23 读到）；本卡输出 3.8 MB（`du -sh`），在数据盘，临时文件放 `$OUT/tmp`（8 KB）。数据盘已用 95%、剩 18 GB，别的卡要大输出时留意。
