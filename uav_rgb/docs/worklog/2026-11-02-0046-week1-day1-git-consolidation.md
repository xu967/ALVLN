# Week1 Day1：工作树清盘 + 镜像同步

- 时间：2026-11-02 00:10–00:46
- 范围：`docs/`、git 提交、ALVLN 镜像同步
- 未触碰：`src/`、`sim/tools/` 的任何代码逻辑（只把它们已有的改动提交）

## 做了什么

### 1. 更正 HANDOFF 的评测口径（`docs/HANDOFF.md` §4.5）

新增「2026-11-02 更正」小节：`0.426` 是按 `goal_reach` 规则对旧 `per_episode`
**手工复算**的值；`l1-v4-closed-20261101/student-keep/eval.json` 实际是
`success_criterion="goal_stop"`、`success_rate=0.3404`（16/47）。
明确写「v4 未过 50% 线，两种口径都不达标」。

### 2. 取消一笔无关暂存

工作树里 `experiments/gpu_bench/` 的 6 个删除原本已进索引，会导致 Commit 1
混入清理内容。执行 `git restore --staged experiments/gpu_bench` 取消暂存
（文件仍在工作树删除状态），让提交按主题干净分离。

### 3. `.gitignore` 加 `.ipynb_checkpoints/`

原先该目录是未跟踪垃圾，会被 `git status` 一直列出来。

### 4. 6 个主题提交（本地 `uav_rgb`）

| commit | 主题 | 文件数 |
| --- | --- | --- |
| `7e4e15f` | feat(L1) 语言任务基础设施 | 10 |
| `205332c` | feat(L2) FiLM 文本分支 + checkpoint v4 | 19 |
| `9ddeff1` | feat(L1) 反事实与探针工具 | 4 |
| `9824143` | fix(L1) goal_reach 判据 + 去重口径 | 12 |
| `03c7eeb` | docs L1 v3/v4 实验记录与 worklog | 19 |
| `22bc58e` | docs+chore 交接文档 + 清理 | 17 |

工作树从 **82 个改动**降到 **1 个未跟踪目录（`teacher/`）**。

### 5. 镜像同步与推送

`rsync --delete`（排除 `.git`、`build`、`src/build`、`.venv-tools`、
`.ipynb_checkpoints`、`teacher`）→ `ALVLN` commit `e9a921f` → push。
远端 `refs/heads/main = e9a921fd…`，本地与远端同步。

> 注意：README 里的 rsync 命令**没有** `--exclude=.ipynb_checkpoints`，
> 直接照抄会把垃圾目录同步到镜像。本次加了这两个排除项，建议把 README
> 里的命令一并更新（未做，见「没做什么」）。

### 6. 创建 Teacher 脚手架（未提交）

`teacher/{model.py,dataset.py,train.py,export_soft_targets.py,README.md}`，
只新增文件、未改任何已有文件。**故意不提交**：数据格式假设未确认（见风险）。

## 没做什么（含原因）

1. **没有删任何数据**。原计划「整目录删除 `l1-stop-500-173307` / `l1-full-400-0803`」
   经清点后**不安全**：前者含 6 组 L2 对照 ckpt 与 `eval-film-ft-keep-49/eval.json`，
   是 `experiments/2026-11-01-l2-film-finetune.md` 全部数字的唯一来源。已改为
   「只删 ds/raw、保留 run/eval/audit」的保守方案（约 9.4 G），**等你确认后再执行**。
2. **没有重跑 v4 三组闭环**（80 min）。需要独占 Gazebo 窗口，且当前结论不依赖它。
3. **没有实现 `collect.py --save-depth`**。Teacher 的数据集读不到深度：现有采集
   只在内存解码深度、不落盘，manifest 也无深度引用。这是 Day 2 的前置改动。
4. **没有训练 Teacher**。本机未装 torch/torchvision；且深度缺失，数据集会是 0 样本。
5. **没有提交 `teacher/`**。`dataset.py` 对 depth/pose 的字段名是假设值
   （`depth_file`/`depth_ref`、`pose_before`+`yaw_before`），需先确认。
6. **没有更新 README 的 rsync 示例**。应补 `--exclude=.ipynb_checkpoints --exclude=teacher`。

## 验证到哪一步（真实输出）

- `cmake --build src/build`：`ninja: no work to do.`（产物 Nov 1 07:02 晚于源码 00:24）；
  未做 clean 全量重建。
- `ctest --test-dir src/build`：**100% tests passed, 0 tests failed out of 21**，
  耗时 124.78 s。
- `test_text_branch` 单独运行：`text gradcheck max_err=1.576e-05`、`PASS`、`EXIT=0`。
- `src/build/uavrgb_train`、`src/build/uavrgb_infer` 均存在。
- `nvidia-smi -L` 列出 RTX 2080 Ti。
- `git ls-remote --heads origin` → `e9a921fd…  refs/heads/main`。

## 风险与遗留

1. `teacher/` 的深度格式假设未落实；实际采集**没有深度文件**，直接跑会 0 样本。
2. 构建验证依赖 Ninja 时间戳判断，未做 clean 重建；若担心，另开空目录全量编一次。
3. 数据区清点有 `du` 两次读数不一致（519M vs 552M），精确容量建议用
   `du --apparent-size` 复核。
4. `pubdata/airsim10k.zip` 删除前需确认 Kaggle token 有效（文档曾建议吊销），
   否则无法重下。
