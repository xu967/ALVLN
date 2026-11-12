# 2026-11-12 航向码端到端学生：2×2 试点（卡 E2E-1 第 3 步）

四个臂：A0 图像/demo（现成权重）、A1 航向码/demo、A2 航向码/dagger、A3 图像/dagger；每个臂 3 个训练种子，评测 20 个冻结 seed，不带 `--heading-guard`，动作延迟 L* = 0.20。
预注册的三个比较（M1、M2、M3）与判据逐字写在 `run.sh` 的文件头，起跑后不改。方案背景：`docs/PLAN_2026-11-10_phase12.md` §10；任务卡：`docs/briefs/2026-11-10-phase12-task-cards.md` 的 E2E-1。

## 文件

| 文件 | 作用 |
| --- | --- |
| `lib.sh` | 路径、臂表、日志、锁、cgroup 记录，`prepare.sh` 与 `run.sh` 共用；带 `${X:-}` 的项只在演练时覆盖 |
| `freeze.sh` | 把已提交的 HEAD 冻结到 `$OUT/code`（`git archive`），写 `code_head.txt`；工作树有未提交改动时拒绝 |
| `prepare.sh` | 训练池一侧（守护进程）：重建核对 → 航向码重打包 → 训练 9 个新 run → 配方核对 → 末 10 轮平均 + 校准 → 与 eval-only 对拍 → 离线报告；写 `PREP_DONE` |
| `run.sh` | Gazebo 池一侧（守护进程）：12 个评测臂（4 臂 × 3 训练种子，各 20 集），训练种子优先的顺序，迟到集重跑一次，写 `DONE`、`summary.md`、`verdicts.json` |
| `summarize_e2e.py` | 汇总与判定（M1–M3、W2-2 只报告、2×2、逐种子、动作占比、cond_code 自检、每步生效时长、离线探针）；随时可跑，缺的臂标「未完成」 |
| `e2e_arms.py` | 臂与路径约定的 Python 版（`lib.sh` 里有 bash 版，`test_e2e_arms.py` 核对两边一致） |
| `compare_packs.py`、`check_cond_codes.py`、`check_recipe.py`、`make_dagger_dirs.py` | 重建核对、航向码独立重算、训练配方逐键核对、dagger 的 592 个集目录清单 |
| `verify_infer_vs_evalonly.py`、`offline_report.py` | 交付权重的逐帧推理与 `uavrgb_train --eval-only` 对拍；训练曲线、val 指标、反事实探针 |
| `stub_eval_e2e.py` | 演练用的假评测入口（不起 Gazebo） |
| `test_*.py` | 见下文「测试」 |

## 运行顺序

```bash
cd /root/claude-code-backup-20261028/code/wt/p12-e2e-1 && bash experiments/2026-11-12-e2e-heading-code/freeze.sh   # 冻结已提交的代码
cd /root/rivermind-data/uav_rgb/exp-e2e-heading-code-20261112/code/experiments/2026-11-12-e2e-heading-code
setsid nohup bash prepare.sh > /dev/null 2>&1 < /dev/null &                                                     # 训练池一侧；日志 ../../../prepare.log，结束写 PREP_DONE
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &                                                         # PREP_DONE 写 rc=0 之后；日志 run.log，结束写 DONE
```

- 停：`kill -- -$(cat $OUT/prepare.pid)`（或 `run.pid`），只按进程组，不用 `pkill -f`。续跑：同一条命令重起，完成的阶段与评完的臂跳过，半成品挪到 `<名>.partial-<时刻>`，不删。
- 进度：`tail -f $OUT/prepare.log`、`tail -f $OUT/run.log`；汇总随时可跑：`<解释器> $OUT/code/experiments/2026-11-12-e2e-heading-code/summarize_e2e.py $OUT`。
- 锁：训练拿 `train.lock`（阻塞等待，≤ 3 个训练进程）；任何 Gazebo 评测拿 `machine.lock`，拿到后核对没有 gz sim 进程且 30 s 窗口内容器低于 8 核才起跑，不满足就每分钟复查，最多等 600 s，到点照样起跑并把当时的核数记进 `start.txt`（公共规则 §4「2026-11-12 01:40 起」，主仓库提交 89b7804）。
- rc 表在两个脚本的文件头（`prepare.sh`：0/3/4/12/20/21/22/23/30/31/32/33/34/130；`run.sh`：0/3/4/5/6/7/11/12/13/130）。

## 输出目录 `$D/exp-e2e-heading-code-20261112/`

`code/`（冻结代码）、`code_head.txt`、`bin/uavrgb_train`、`recon/`（默认重建对账、码独立重算、配方核对、SHA256SUMS）、`data/{demo,dagger}-hc/`（航向码数据集与 `dataset.aux` 侧车）、`runs/`（9 个新训练 run）、`ckpt/`（9 个交付权重 + `.json` 侧车）、`offline/`（曲线、val 指标、探针、12 份对拍）、`<臂>-s<种子>/`（评测）、`<臂>-s<种子>-retry1/`（迟到集重跑）、`<臂>.cg`（cgroup 记录）、`summary.md`、`verdicts.json`、`episodes.csv`、`start.txt`、`inputs.txt`、`config.env`、`PREP_DONE`、`DONE`。

## 测试

均在仓库根运行，单进程，不起 Gazebo、不用 GPU（解释器 `.venv-tools/bin/python`；跑时建议 `nice -n 10`、`OMP_NUM_THREADS=1`）。

| 测试 | 覆盖 | 时长 |
| --- | --- | --- |
| `test_summarize_e2e.py`（25） | 配对与精确 McNemar、M1–M3 与 W2-2 的阈值边界、失败分类、航向码统计、完整世界与缺臂「不判」、迟到重跑的三种结果、W2-2 参照读到 53/60 | 约 10 s |
| `test_pack_checks.py`（13） | 打包对账、侧车核对、码独立重算（含篡改必须被抓住） | 约 5 s |
| `test_e2e_arms.py`（14） | `lib.sh` 与 `e2e_arms.py` 的路径约定一致 | 约 5 s |
| `test_verify_offline.py`（10） | 指标定义（缺类按 0、除以 4）、对拍容差、码 2↔3 对调、探针统计、曲线读取 | 约 1 s |
| `test_run_rehearsal.py`（15） | `run.sh` 全流程（桩评测）：命令行、顺序、重跑、续跑、各个拒绝 rc、等机器锁 | 约 40 s |
| `test_prepare_rehearsal.py`（16） | `prepare.sh` 全流程（桩工具）：阶段与标记、续跑、各个 rc、训练池锁、SIGTERM 与续跑 | 约 45 s |
| `test_freeze.py`（7） | `freeze.sh` 与 `verify_frozen`：脏工作区、冻结后换代码、冻结副本被改或缺文件 | 约 2 s |

变异检验（把脚本故意改坏一处，对应测试必须变红）的脚本与日志在草稿目录，见报告 `reports/e2e-1.md`。
