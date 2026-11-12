# 第二轮交叉审核的修复清单（2026-11-12 08:19 起，主会话汇总）

- 来源：`docs/briefs/2026-11-12-review-round2-brief.md` 的九个角度（I1、I2、J、K、L、M、N、O、P），Gemini 4 Carbon High 的只读审核 agent。各角度的原报告在本目录 `review2-<代号>.md`。
- 用法同第一轮的 `docs/reviews/2026-11-12-cross-review/FIXLIST.md`：每条发现一行；「状态」照抄审核者的说法；「处理」写清并入哪张卡。修复卡的规格在任务卡文档「4c. 第二轮交叉审核的修复卡」。
- 到一份归档一份。已到：I2（08:16）。

## 怎么看 diff

- 这一轮修复的基线标签：`base2-20261112`（主会话在第一张第二轮修复卡合并之前打）。全部改动：`git diff base2-20261112..main -- src sim tools proto`。
- 一张卡一个合并提交，一条发现一个提交，说明第一行带发现编号（公共规则 §5）。

## I2：11-12 合进 main 的工具链改动（`review2-i2.md`，11-12 08:16 交回；审的是 `ada0ca5`，收尾复核到 `035b398`）

审核者另外实跑确认了一件事：`preprocess.py` 的三个新开关（`--cond heading-code`、`--depth-source analytic`、`--exclude-failed-teacher`）加 `--drop-collision` 同时开，在合法输入下像素、标签、航向码、解析深度与 C++ 读端逐项对齐；
`tools/` 下三次合并冲突的解决没有丢逻辑；`paired_stats.py` 的数学实现无误。

| 编号 | 问题 | 位置 | 状态 | 影响 | 处理 |
| --- | --- | --- | --- | --- | --- |
| I2-1 | 卡 W1-14 的协议一致性门禁对 `dynamics_profile` 这一个键没有生效：`stats_report` 只读 eval.json 顶层，常规 Gazebo 评测只把它写在 `per_episode` 里。带 `--cf-sitl-params` 的臂全部失败时顶层写 null，反而与同批正常臂报冲突 | `tools/stats_report.py:287`、`:570-578`；`sim/tools/eval_closed_loop.py:1481-1487`、`:1577-1579` | 已复现 | 不同固件库、不同动力学参数的 Gazebo 评测可以进同一张表而不报错。第一轮清单的 A4 写「已修」要改成「五个键里修了四个」 | 卡 W1-14b（`stats_report` 从 `per_episode` 取）；评测端全失败时的顶层值并入卡 W1-13 第二轮 |
| I2-2 | `dagger_gz_pipeline` 开航向码又用默认的交付配方时三处断链：训练完才被 `average_ckpt` 拒绝（main 上它还不支持带条件的模型），`preflight` 不拦；采集端不核对学生权重的条件类型，忘换第 0 轮权重时第 1 轮由纯图像模型采集而没有任何报错；汇总调 `stats_report` 被新门禁拒绝，报表悄悄没了 | `tools/dagger_gz_pipeline.py:57-82`、`:505-525`、`:752-757`；`tools/average_ckpt.py:210`；`sim/tools/collect.py:966-970` | 第 1、3 点已复现，第 2 点读代码确认 | 端到端线的 E1 正式版与 E3（卡 W2-2，带航向码的 DAgger）走的就是这条路 | 第 1 点：分支 `p12-e2e-1` 的 `dc8e5ff` 先合进 main，`deliver_check` 在 `preflight` 里拦不支持的配置（卡 W1-15c）；第 2 点：`collect.py` 接上 `check_student_cond`（卡 W1-13 第二轮）；第 3 点：汇总传 `--allow-mixed-protocol`（卡 W1-15c） |
| I2-3 | `stats_report` 的三处口径不一致：跳过集在主表里记 0，在配对表里被漏掉；两侧共同失败的 seed 没有几何号，让整张配对表从按几何聚类退回按 seed 聚类；失败记录重复时簇的分母被累加 | `tools/stats_report.py:280`、`:330-339`、`:398`、`:418-425`、`:478-482` | 第 1、2 点已复现，第 3 点读代码确认 | 已发布的柱阵主表不受影响（没有跳过集，每个 seed 一个几何）。森林高密度档、校园、3DGS 这类有跳过集或多 seed 同几何的评测会出现主表与配对表两个成功率 | 卡 W1-14b |
| I2-4 | `preprocess.py --exclude-failed-teacher` 把读坏的 `oracle.jsonl` / `scene.json` 当成「不适用」原样打进训练集；首行是 NaN 时把过了线的集误删；检查排在 `actions.jsonl` 截断检查之前，开了它截断的集反而不报错；val 被跳空不报 | `tools/preprocess.py:163-183`、`:303-314`、`:536` | 第 1–3 点已复现，第 4 点读代码确认 | 0417 的打包结果不受影响（文件完整）。之后大批量新采的教师数据一旦混进写了一半的文件，会被悄悄放过 | 卡 PK1 |
| I2-5 | 分支 `p12-e2e-1` 的 `dc8e5ff`：平均后的权重旁边的 `.json` 不记数据集的条件类型，训练时 `--data` 用相对路径或数据集被清理之后，评测端的条件一致性闸门退化成只警告；`.json` 不是原子写，写到一半被杀时流水线会当成校准做完了；校准不看训练契约里的 `drop_collision` 位（同第一轮 C6） | `tools/average_ckpt.py`（`dc8e5ff`）`:168-169`、`:258-263`、`:289-290`；`sim/tools/nav_cond.py:95-141` | 第 1 点已复现，第 2、3 点读代码确认 | E2E-1 这次用的是绝对路径，数据集也在，没有触发 | 合并 `p12-e2e-1` 之后补第 1、3 点（卡 W1-15c）；第 2 点仍归卡 E2E-C1 |
| I2-6 | 整集的步被 `--drop-collision` 或 `--drop-late-and-stall` 全部滤掉时，这一集仍以 0 个样本留在数据集里；带指令侧车且有两个这样的集时 C++ 读端直接报错。步号有缺口时 `--cond heading-code` 把读不懂的动作当成不转向，不报错 | `tools/preprocess.py:327-336`、`:350-373`、`:485-492`、`:719-723`；`src/dataset.cpp:376-380` | 已复现 | 现有数据不受影响。W1-13 的过滤开关合并之后整集被滤空更容易出现；E5 的语言任务带指令侧车 | 卡 PK1（与第一轮 C7 一起） |
| I2-7 | `dagger_gz_pipeline` 的汇总写着「分母 = 请求集」，打印的是成功数比完成数；它自己算的配对检验没有把失败集记 0 | `tools/dagger_gz_pipeline.py:742-754`、`:771`、`:790-796` | 已复现 | 有失败集时同一格里的分数、区间、检验用了两套分母 | 卡 W1-15c |
| I2-8 | `make_params_variant.py` 给整数参数传 `25.0` 会写成浮点，指纹跟着变；解析深度里圆柱不算顶盖，相机高过柱顶时射线穿过圆柱打到地面，方柱则会打到顶面 | `tools/make_params_variant.py:114-119`；`tools/preprocess.py:1021-1046` | 已复现 | 现在改的参数是浮点、柱子都比飞行高度高，已有数字不受影响 | 第 2 点并入卡 PK1；第 1 点随下一次动满电档的卡修 |
| I2-9 | 门禁里没有任何一处让 C++ 读端读过带解析深度或航向码的侧车（`read_dataset --selfcheck` 写死「合成集没有深度」）；几何 split 的第三层防线 `build_meta` 仍没有测试 | `tools/check.sh:57-66`、`:106-146`；`tools/read_dataset.cpp:306`；`tools/preprocess.py:631` | 读代码确认（审核者自己编了读端实跑过一次，现在的格式没有漂） | 以后改侧车布局时门禁抓不到 | 第 2 点并入卡 PK1；第 1 点（全量 `check.sh` 加一段）并入卡 E2E-C1 之前的 C++ 小卡，等卡 C1a 交回后定 |

审核者核对第一轮清单的结论：D2、D5、D7、F3、W1-6 与今早合并的 E-3、E-4、E-7、E-8、E-9、E-13、A2 修对了；A4、D4 与 W1-2 的遗留、F7 修了一半（见 I2-1、I2-3、I2-9）；
另记一条不在九条里的：三条 Python 流水线仍在活的工作树上跑，没有冻结代码，续跑会重写 `config.json`（第一轮 E-2 的流水线部分；`exp_lib.sh` 只管 shell 入口）。这一条先记着，等 Gazebo 线的入口迁到 `exp_lib.sh` 时一起定。
