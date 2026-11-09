# STUDENT-SELECT：`--save-every-epoch`、选模判据、训练噪声、双帧反事实

- 时间：2026-11-09 17:50 开工，18:45 收工（子 agent，worktree `tmp/wt-student-select`，分支 student-select @ 2899428，未提交、未推送）
- 范围：`src/`（只加 `--save-every-epoch`）、`tools/select_ckpt.py`、`tools/probe_two_frame_cf.py` 及两份测试、`docs/experiments/2026-11-09-student-select.md`；实验产物 `$DATA_ROOT/uav_rgb/exp-student-select-20261109/`
- 结论：完成（代码 + 三组离线实验 + 报告）；闭环部分按任务书交主会话排 eval4 之后

## 做了什么

- `src/include/uavrgb/trainer.hpp`（+4）、`src/train_main.cpp`（+6/−2）、`src/trainer.cpp`（+6）：`--save-every-epoch`，每轮在 `save_checkpoint(ckpt.bin)` 之后另存 `<run>/ckpt_ep{NN}.bin`（NN = `state_.epoch`，与 results.csv 对齐），v6 格式不变，默认关。
- `src/tests/test_save_every_epoch.cpp`（新）+ `src/CMakeLists.txt`（+5）：3 轮合成训练后恰有 3 个 `ckpt_ep*.bin`、每个可加载且 epoch / train_step 对、`ckpt_ep03.bin` 与 `ckpt.bin` 字节相同、默认关时 0 个文件。实现前 FAIL 5 项，实现后 PASS。
- `tools/select_ckpt.py`（新）：对一个 run 的全部 `ckpt_epNN.bin` 用真实 `uavrgb_infer` 子进程跑 val，输出逐轮 acc / macro-F1 / bal_acc / 四类召回 / L-R 差 / 空画面 YAW 率（`teacher.visible_pillar_side(...) is None`，位姿从 `source_dir/actions.jsonl` 回读）/ 相邻轮预测翻转率；显式规则 `select_epoch()`（两条硬约束 + ±1 轮平滑）；逐轮 macro-F1 与 `metrics.jsonl` 对账。支持双帧 run（读 `dataset.aux`）。
- `tools/probe_two_frame_cf.py`（新）：三个反事实侧车（`self` / `next` / `shuffle`，`preprocess.write_aux_sidecar` 重写 `prev_index`，uavr 硬链接）→ 逐帧预测 + 四件套 + YAW 帧 L:R 预测比 + 单帧对照；每个侧车与 `uavrgb_train --eval-only` 对账。
- `tools/test_select_ckpt.py`、`tools/test_probe_two_frame_cf.py`（新，纯 Python，`tools/check.sh` 自动收）。
- 实验：E1 三 seed（trunc/ds）、E2 逐轮选模（3 seed + 0417 单帧 / 双帧）、E3 0417 重打包带 UAVRAUX1 → 单帧 / 双帧训练 → 反事实探针 + 切换帧分析。数字与结论见 `docs/experiments/2026-11-09-student-select.md`。
- 守护进程脚本（在 `tmp/agent10-student-select/`，不进仓库）：`run_e1_noise.sh`、`run_preprocess_0417.sh`、`run_e2_select.sh`、`run_e2b_select_0417.sh`、`run_e3_two_frame.sh`；分析脚本 `summarize_e1.py`、`analyze_e3.py`。

## 没做什么

- 闭环评测（三个 seed 的 ckpt_best、select_epoch 选的三个、双帧 vs 单帧）。原因：任务书规定由主会话排在 eval4 之后；本文只给路径与 sha。
- 没改 split 规则 / val 比例。原因：PLAN §8.1 未拍板，任务书只要建议；建议与依据在实验文档 §5。
- 没把 `select_epoch` 接进 `tools/e1_pipeline.py` 或训练器的 best-ckpt 逻辑。原因：判据对不对要闭环裁决；接进流水线属于改训练行为。
- 没做末 k 轮权重平均（SWA 类）。原因：训练侧改动，本轮禁止（只许加 `--save-every-epoch`）。
- 没修 `sim/tools/teacher.py` 对 numpy 标量的 `(b > 0) - (b < 0)` TypeError。原因：不在我的文件域；在 `select_ckpt.empty_frame_mask` 里转 Python float 绕开。
- 0417 重打包的 `dataset.uavr` sha 与原 ds 不同（meta 多了 11-07 后 preprocess 新增字段），没有去对齐。原因：images / labels / collision_flags / 300 集 split 逐位相同，单帧训练逐值复现，训练输入等价。
- 双帧反事实只做了 0417 数据 seed 0 一对。原因：任务书只问「双帧是不是捷径」，一对加三个反事实足以回答方向性问题；增益大小（+0.022 < seed 间极差 0.032）本来就不能用单 seed 下结论。
- 没跑 `tools/check.sh` 全量。原因：Python 侧两份新测试单独跑过；C++ 侧 CTest 31/31 在 `tmp/build-student-select` 跑过；全量 check.sh 含 cf_sitl 等与本改动无关的长项，留给合并时。

## 待确认

- 选模判据的两条阈值（空画面 YAW 率 ≤ 标签率 + 0.05、L/R 差 ≤ 0.15）是按 trunc 三 seed 的分布定的，没有闭环证据；是否采用看闭环。
- val 比例 15–20% 的建议会减少训练集（300 集本就少），是否接受要主会话权衡。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `tmp/build-student-select` 构建 + `ctest`（`tmp/agent10-student-select/logs/ctest1.log`） | `100% tests passed, 0 tests failed out of 31`，`31/31 Test #31: test_save_every_epoch ............ Passed 3.10 sec` |
| 红测 | 实现前 `test_save_every_epoch` FAIL 5 项（无 ckpt_ep 文件） |
| `python tools/test_select_ckpt.py` | `PASS: select_ckpt 四件套 / UAVR 读取与过滤 / 样本→位姿 / 空画面 / 选模规则` |
| `python tools/test_probe_two_frame_cf.py` | `PASS: probe_two_frame_cf 反事实映射 / 侧车重写 / 方向计数` |
| seed 0 复现 `tmp/agent6-distill/trunc/run` | 31 条 validation 事件逐值相同；ckpt_best 权重与 Adam m/v 逐 float 相同（文件 sha 不同：v5 → v6 头） |
| 0417 单帧复现 `e1-pipeline-20261107-0417/run` | 31 条 validation 逐值相同；best ep14 mF1 0.396405438，test acc 0.6904 |
| `select_ckpt.py` 推理路 vs 训练器 val | 5 个 run × 30 轮 macro-F1 最大差 1.5e-8 |
| `probe_two_frame_cf.py` 逐帧 vs `--eval-only` | 4 个侧车 × 2 split 最大差 1.4e-8 |
| `git -C tmp/wt-student-select status --short` | 4 个 M（src）+ 7 个 ??（1 测试、2 工具、2 测试、1 实验文档、本文） |
| 残留进程 | 0（`pgrep -af agent10-student-select|uavrgb_train|select_ckpt|probe_two_frame` 只剩别的 agent 的闭环评测进程） |

## 风险与遗留

- `trainer.cpp` 的 6 行加在 `Trainer::run()` 末段 `save_checkpoint(ckpt_path…)` 之后；FIX-B 若也改这一段，合并时以这 6 行为准（已在 `tmp/agent10-student-select/PROGRESS.md` 写明）。
- `select_ckpt.py` 每个 ckpt 起一个 `uavrgb_infer` 子进程，30 轮 × 574 样本约 5 min；val 变大后线性增长。
- `$DATA_ROOT/uav_rgb/exp-student-select-20261109/` 842 MB，其中 `ds-0417-aux/dataset.uavr` 681 MB（E3 可复现所需，三个 cf-* 目录是硬链接不占空间）。闭环跑完后可删 `e1-noise/seed*/run/ckpt_ep*.bin`（90 × 1.2 MB）。
