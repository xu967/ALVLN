# 航向码端到端学生 2×2 试点（任务卡 E2E-1）：平均校准工具的条件输入、实验入口、训练与闭环评测

- 时间：2026-11-12（步骤 2 提交 01:02，入口与测试提交 01:52，本文 02:05 起写，随进度更新）
- 范围：步骤 1 是任务卡 W1-9（见 `docs/worklog/2026-11-12-0040-p12-w1-9-heading-code.md`，分支 `p12-w1-9`）；步骤 2：`tools/average_ckpt.py`、`tools/test_average_ckpt.py`；步骤 3：`experiments/2026-11-12-e2e-heading-code/`（14 个入口文件、7 个测试文件）
- 分支：`p12-e2e-1`（基于 `p12-w1-9`），提交 `dc8e5ff`（步骤 2）、`f6e8bd0`（入口）、`7ec2069`（入口测试）；未合并、未推送
- 结论：进行中。步骤 1、2 完成；步骤 3 的入口已提交并冻结在 `7ec2069`，测试与冒烟通过，训练池守护进程已起，打包阶段完成，训练在 `train.lock` 上排队，闭环评测未起。实验数字与预注册判定见 `docs/experiments/2026-11-12-e2e-heading-code.md`（出结果后补）。

## 做了什么

- 步骤 1（W1-9）：`sim/tools/nav_cond.py` 与 `preprocess --cond heading-code`、`collect --student-cond heading-code`、评测端一致性闸门，提交 `ba9b630`、`402fdcb`。
- 步骤 2，`tools/average_ckpt.py`（`dc8e5ff`，+53/−8 行）：`prev_dim=4` 的 ckpt（`--prev-action` 训出的 prev-action 模型与航向码模型）校准时，每帧后多发 1 字节条件，取自 `--data` 同名 `.aux` 侧车的 `prev_action` 列，协议同 `sim/tools/student.py`。侧车与数据集的绑定逐项核对（N、flags、头里的 uavr sha256、文件长、payload sha256、列值 < 4）；推理进程 hello 的 `prev_action_dim` 与 ckpt 不符先拦；结果 json 多一个 `cond` 记录。文本指令、双帧、`prev_dim` 不在 {0,4} 的 ckpt 仍拒绝校准。不带条件的 ckpt 走原路径：W1-5 的 6 个权重用新工具重新生成，sha256 与记录相同。`tools/test_average_ckpt.py` +243/−2 行，共 25 例。
- 步骤 3，入口 `experiments/2026-11-12-e2e-heading-code/`：
  - `prepare.sh`（训练池一侧）：重建核对 → 航向码重打包 → 训练 9 个新 run（A1、A2、A3 各 3 个训练种子，`--save-every-epoch`）→ 配方逐键核对 → 末 10 轮平均加校准 → 与 `--eval-only` 对拍 → 离线报告。
  - `run.sh`（Gazebo 池一侧）：12 个评测臂 × 20 集，训练种子优先的顺序，迟到集重跑一次，写 `DONE`、`summary.md`、`verdicts.json`、`episodes.csv`。预注册 M1–M3 逐字写在文件头。
  - `summarize_e2e.py`：配对精确 McNemar、M1–M3 判定、W2-2 只报告、2×2 与逐种子、动作占比、cond_code 自检、每步生效时长、每臂 cgroup 记录、离线探针。
  - `compare_packs.py`、`check_cond_codes.py`、`check_recipe.py`、`verify_infer_vs_evalonly.py`、`offline_report.py`、`freeze.sh`、`lib.sh`、`e2e_arms.py`、`make_dagger_dirs.py`、`stub_eval_e2e.py`、`README.md`。
- 测试：7 个文件共 100 例（`test_summarize_e2e` 25、`test_pack_checks` 13、`test_e2e_arms` 14、`test_verify_offline` 10、`test_freeze` 7、`test_run_rehearsal` 15、`test_prepare_rehearsal` 16）；33 处单点变异（汇总器阈值与口径、`run.sh`、`prepare.sh`、冻结与对拍）各使至少一个对应测试变红，3 个对照变绿。
- 冒烟（任务卡「起跑前」要求，S1–S6）：demo 用 `--cond heading-code` 重打包并与原包对账 → 带 `--prev-action` 训 2 轮 → 平均加校准 → 推理路与训练器 `--eval-only` 对拍（航向码权重与 A0 权重各一份）→ Gazebo 1 集 → 核对送进推理进程的逐步码、评测行的 `cond_code`、`nav_cond.heading_codes(executed)` 三者逐步相同。
- 按主会话 01:40 的公共规则（容器低于 8 核起跑，最多等 10 分钟，到点照样起跑并记核数）改了 `run.sh`、冒烟脚本与汇总器：起跑记录 `start.txt` 多 `idle_max_cores`、`idle_limit_s`、`idle_forced`、`idle_cores` 等键，汇总每臂一行 cgroup（平均核数、被限流时长占墙钟的比例、限流次数增量、迟到步）。
- 冻结代码到 `$D/exp-e2e-heading-code-20261112/code`（`7ec2069`），起 `prepare.sh` 守护进程（PGID 755089，01:52:49）。打包阶段完成：demo 与 dagger 的默认重打包与现有数据集逐项相同；demo-hc 与 dagger-hc 通过与原始动作的独立重算（非零码样本 464 与 2545）。
- 草稿、日志、变异脚本：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/e2e-1/`；报告 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/e2e-1.md`。

## 没做什么

- 没改 C++（`src/` 0 行）、没改网络结构。原因：任务卡。
- 没重新采集数据；demo 与 dagger 用现成采集重打包。原因：任务卡的设计。
- 没带 `--heading-guard`，没做 3 块课程。原因：预注册与任务卡「不做」。
- 没动 `progress.md`、`CLAUDE.md`，没合并、没推送、没删既有文件、分支、工作树。原因：交办。
- 没在冒烟里让一个会转向的模型走 Gazebo：冒烟模型只训 2 轮，8 步全部 FORWARD_SLOW，码全为 0。非零码的接线靠 W1-9 的单元测试，并在正式评测的 A1 前几集里核对。
- 没做任务卡 11-12 01:10 补充的 1、2、4（按几何聚类的符号翻转检验、40 个 test 几何的样本外补评、「只平均不校准」的离线指标）。原因：卡里写明由主会话在审报告时执行，入口不用改；入口输出的 `episodes.csv` 与 `OUT/runs/` 里的逐轮权重够主会话自行计算。第 3 条（局限）写进了实验文档与报告。

## 待确认

- 配对键：我的汇总器按（训练种子，评测 seed）配对，3 × 20 = 60 对。任务卡写「60 对逐集配对」，没指定怎么配。
- 「越界集数」取失败分类里的 `corridor` 类；eval.json 的 `corridor_violation` 标志另列一行。两个口径在 A0 的第二轮数据上是否一致，以实际结果为准（未核实）。
- M2 的预测（A1 越界 ≤ A0 越界的一半，成功不低于 A0）与证伪（A0 越界 − A1 越界 < 3）在 A0 越界很少时可以同时成立，汇总器输出 CONFLICT，不裁决。M1、M3 的中间区间输出「不判」。
- 起跑条件：入口按卡草稿写的是 1.5 核、3 h，已按主会话 01:40 的规则改为 8 核、600 s。

## 验证到哪一步

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 步骤 2 先红后绿 | `python tools/test_average_ckpt.py`（新测试对旧工具、对新工具） | 对旧工具 25 例，failures=2、errors=4、skipped=1；对新工具 25 例 OK |
| W1-5 权重逐位复现 | `UAVRGB_REPRO_AVGCKPT=1` 的 `TestReproduction` | 6 个权重 sha256 与记录相同 |
| 入口测试（已提交状态） | `E/step3-logs/run_all_tests.sh final-7ec2069` | 7 个文件共 100 例 OK，`ALL rc=0 2026-11-12 02:03:12` |
| 变异检验 | `E/step3-logs/mutate.py` | 33 处变异全部使对应测试变红，3 个对照变绿（`mutation.json`） |
| `check.sh --python-only` | `flock check.lock bash tools/check.sh --python-only`（`7ec2069` 的本地克隆） | `rc=0 2026-11-12 01:56:46` |
| 冒烟 S1–S4 | `E/smoke/s1_pack.sh`、`s2_train.sh`、`s34_avg_verify.sh` | 重打包对账全部相同；推理路与训练器 `--eval-only` 的 `confusion` 逐格相同（航向码权重与 A0 权重各一份） |
| 冒烟 S5、S6 | `E/smoke/s5_gazebo.sh`、`s6_check.py` | `eval_closed_loop` rc=0，1 集 8 步；线上码 = `cond_code` = `nav_cond.heading_codes`，长度 8/8；`student_cond_check.status: ok` |
| 重建与码核对 | `prepare.sh` 的 pack 阶段 | 4 份包对账 `核对项全部相同`；两份航向码包独立重算 `RESULT OK`，0 集不符 |
| 离线报告演练 | `offline_report.py`（冒烟产物拼的目录布局） | 出曲线、val 指标、探针；码 0 帧预测变化自检 0/559、0/848 |
| 训练、配方核对、校准、对拍、闭环评测 | 未完成 | 见实验文档 |

## 风险与遗留

- 非零码经 Gazebo 链路送进推理，冒烟没有覆盖；正式评测开始后看 `a1-code-demo-s0` 的前几集（结果写进实验文档）。
- 训练 run 的 `config.json` 里 `source_git` 记 `7ec2069`；之后若改评测端文件需手动重新冻结，并在报告写明。
- 机器负载高（`/proc/loadavg` 21–27，另有其他任务的训练与 Gazebo）。评测的时间健康由 cgroup 记录与迟到集重跑兜；没有兜住的臂在汇总里带 ⚠。
- 训练数据是旧协议与旧动力学库，评测是新协议；四个臂受同样的偏移，臂间比较成立，绝对数不能与旧协议的数直接比（卡 01:10 补充第 3 条）。
