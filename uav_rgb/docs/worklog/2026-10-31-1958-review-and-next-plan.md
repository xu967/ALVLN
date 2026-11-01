# 全仓代码审核（4 路只读）与下一阶段计划

- 时间：2026-10-31 19:00–20:05
- 范围：`uav_rgb/` 的未提交改动（src、sim/tools、tools、docs）；只读审核 + 修复 + 计划
- 结论：完成审核与计划；修掉 1 个评测口径洞、1 个数据解析边界、3 处假绿测试、
  4 个门禁/脚本缺口；v2 500 集仍在采集（约 489/500），audit 尚未跑。未 commit/push。

## 做了什么

- 4 路并行只读审核：C++ 核心、采集链路、评测/预处理/测试、文档一致性。
  覆盖 `git diff HEAD` 的 34 个改动文件 + 未跟踪的 `text.hpp` / `test_text_branch.cpp`。
- **评测口径（H）**：`src/infer_main.cpp` hello 增加 `text_dim`；`student.py` 解析并暴露；
  `eval_closed_loop.py` 在 `blank/shuffle` 前启动一次 infer 探针，纯视觉 ckpt 直接报错，
  并在 `eval.json` 记录 `text_dim` / `success_criterion`。此前纯视觉 ckpt 上
  blank/shuffle 与 keep 等价，L/I/S 会被误读成「语言无增益」。
- **侧车解析（M）**：`src/dataset.cpp` 不再用 `stoul`；自累加并拒绝 >uint32，
  token 数组只接受数字/空格/逗号（`[-1]` 不再被读成 1）。
- **audit 门禁**：新增 `safe_full_length`（safe 集必须跑满请求步数）与
  `provenance_uniform`（teacher/task 版本混包 FAIL）；`teacher_stop` 容差 0.3→0.2，
  有 `stop_radius` 但无 astar 集时 FAIL 而不是 SKIP。
- **preprocess**：有指令时先删旧 `instructions.jsonl` 再替换 `dataset.uavr`，
  侧车用 `mkstemp`；失败时宁可没有侧车，也不留「新数据集 + 旧侧车」。
- **eval shuffle**：`scene_gen.py` 写 `distractor_color`；shuffle 优先用它，
  不再换成场景里不存在的颜色。
- **check.sh**：加文本长度/词表跨语言门禁（`text.hpp` vs `preprocess.py` /
  `instruction.py`）；`ctest --no-tests=error`；safe 剔除负例按 `episode_id` 断言；
  `--python-only` 的 PASS 行标注未验证 C++/GPU。
- **测试**：修 `test_text_branch` 的 pad 假绿（token 换位置）、补 text_dim 专项拒绝
  用例；`test_trainer` 的 NaN 注入加越界保护。
- **文档**：按审核结果修正 38 处矛盾中的 H/M 项：旧 L1 400 集全部标废
  （TASKS/HANDOFF/plan/3 篇 experiment）；`traverse_rate` 在 v2 上 0/40 是假失败，
  §9.3 门禁改为 `teacher_stop`/`safe_full_length`；`M0` 验收补可执行阈值；
  文档不再把「再加数据」当第一选项。新增 plan §10 执行计划。

## 没做什么

- 没跑 500 集 audit / preprocess：采集仍在进行（19:58 约 489/500），跑不了。
- 没训练 L2、没跑离线/闭环 L/I/S：依赖 500 集 audit 通过。
- 没修以下 M/L（排期在 plan §10.1 末尾）：`collect_parallel` 只信
  `stopped_on_goal` 布尔、student eval 的 3 连 SAFE 提前停未写终止原因、
  `src/device/` 状态层不支持 text_dim、B4（infer 语义对拍）、
  `eval_closed_loop` 的 actions/oracle 行数校验。
- 没改 `src/device/*`、CUDA kernel、UI；没深审 `collect_parallel` 的进程管理。
- 没 commit/push：用户未授权。
- 没删除任何旧数据/旧 run；v1 400 集保留为错误样本。

## 待确认

- 500 集 audit 通过后是否直接按 plan §10 跑 M0（训练 10 轮 + L/I/S）。
- SAFE 步占比约 46% 是否接受；若闭环 SAFE 预测过高，是否在打包时抽样 safe 集
  （不重采，只改输入目录列表）。
- M2 前是否补 `collect_parallel` 终止距离校验和 student eval 终止原因。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `cmake --build src/build` | PASS（28/28） |
| `cmake --build build/check` + `ctest` | PASS，21/21（96 s） |
| `test_audit_dataset` / `test_eval_summary` / `test_eval_closed_loop_args` / `test_collect` / `test_student` / `test_instruction` | PASS |
| infer hello | `{"in_h":120,"in_w":160,"in_c":3,"num_classes":4,"text_dim":16}`（50 集文本 ckpt） |
| 消融前置校验 | 纯视觉 ckpt + `--text-mode blank` → `exit 2`，报 `text_dim=0` |
| `bash tools/check.sh` | PASS（rc=0；21/21 CTest + 全部 Python 测试；第一次构建失败是 `test_trainer` 误用 `CHECK` 宏，已修后重跑） |
| 500 集 audit / preprocess / 训练 / 闭环 | 未跑（采集中） |

## 风险与遗留

- 文本分支的端到端回归仍只有合成数据；真实训练后必须先用 `--text-mode keep/blank`
  在 1–2 个 seed 上确认输出确实不同，再跑 36 集。
- SAFE 46% 与 pilot 的 21.5% 不同：`class_share` 只卡 5% 下限，不会报这种偏移；
  训练后要看动作分布和 stuck。
- audit 新增两项后总数从 16 变 18；引用旧「16/16 PASS」的文档已同步，但历史产物
  仍显示当时的项数。
- 文档改动面大；如有并发会话同时在改 docs，提交时需逐文件确认归属。

## 更新 2026-10-31 20:25：500 集 audit / preprocess / 训练

- 采集完成：500 集（train 393 / val 58 / test 49），10509 步，0 失败。
- 首次 audit **17 PASS / 1 FAIL**：`safe_collision` 3/5205 个 SAFE 步带 collision，
  样本 `train/700268`；另有 `train/700483` 含 9 个 collision 步中的其余部分。
- 处置：两个集移到 `l1-stop-500-173307/excluded/`，从 `collection_manifest.json`
  的 `splits` 移除并把 task 标为 `excluded_collision`，`excluded` 记录原因；
  原始 manifest 备份 `collection_manifest.raw.json`。**只丢 2/500 集，不重采。**
- 二次 audit：**498 集 / 10442 步，18/18 PASS**（`teacher_stop` 0/400 失败、
  `safe_full_length` 0/100 失败、`label_recompute` 0/10442 不一致）。
- preprocess：`ds/dataset.uavr`（601,745,837 B，sha256 `99c70ed8…`）+
  `instructions.jsonl`（498 行，sha256 `e8f865ec…`）；meta `task_version=l1.v2`、
  `stop_radius=0.5`。
- 数据卡：`docs/experiments/2026-10-31-l1-v2-500-data-card.md`。
- 训练已启动：`run-l2-text16`（10 轮 / text-dim 16 / batch 64 / lr 1e-3 /
  seed 1234 / class-weights auto / drop-collision），结果待出。
- 类分布：SAFE 49.7%、FORWARD 39.3%、YAW_L 5.5%、YAW_R 5.4%；100 个 safe 集
  跑满 40 步，约 4000 步为原地悬停；val 的 YAW 支持仅 57/56 步。
- 验证：audit 二次 PASS；preprocess 日志正常；训练未完成，离线/闭环未跑。
- 风险：SAFE 占比高可能让模型偏保守；若闭环 SAFE >60% 或 stuck 上升，
  按计划在打包时抽样 safe 集（不重采）。

## 更新 2026-10-31 21:00：M0 文本模型训练出「语言未被使用」

- `run-l2-text16`（498 集、10 轮）完成：dev 最佳 epoch 5 macro_f1 0.8575；
  `ckpt_best` 离线 test **macro_f1 0.8595 / acc 0.9106 / bal 0.8972**。
- 离线 L/I/S（`ds` 的 blank/换干扰色 sidecar 变体）：
  L 0.8595 / I 0.8621 / S 0.8595 —— **无差异**。
- 直接探针（147 张 test 图，同一图换 `--text`）：
  蓝→红概率 L1 差 **2e-4**，argmax 动作 **0 次变化**；打空 1.6e-3、0 次变化。
  结论：文本分支被优化器忽略，不是离线指标不敏感。
- 反事实探测：换目标色后 **2274/6442 步（35.3%）** 专家动作会变，
  398 个 astar 集全部有差异——任务需要文本，缺的是「同图不同文本→不同标签」的监督对。
- 处置（不重采）：新增 `sim/tools/make_counterfactual.py`，用 teacher 对同一轨迹
  重算另一根彩色柱的标签，硬链接图像，生成 311 集 / 5045 步（1783 步标签不同，
  35.3%）。只加 train，test 不动。
- 重新打包 `ds-cf`：809 集 / 15487 步；train 类分布 SAFE 31% / FWD 52% /
  YAW 各 8.3%。训练 `run-l2-text16-cf` 已启动（同超参）。
- 8-seed 闭环（未加反事实的模型，keep）：**SR 0.75（6/8）**，collision 0、
  stuck 0、false_forward 0；2 个失败集 `final_wp` 1.99/1.74 m——像停在了错误目标柱。
- 验证：离线/探针/8-seed 均有真实产物；反事实模型的离线/闭环与探针待出。
- 风险：dev 仍是原始数据（无配对），模型选择可能偏向忽略文本的 epoch；
  若反事实模型仍无视文本，下一步是文本编码/损失，而不是继续加数据。

## 更新 2026-10-31 21:25：反事实模型仍无视文本 → 改位置感知编码

- `run-l2-text16-cf`（反事实训练集）完成：dev macro_f1 0.7815（epoch 10），
  test 0.7579。决定性帧（换色会改标签的 255 帧）上：
  `acc(correct text)=0.514`、`acc(alt text)=0.467`、`flip=0.016`；
  train 决定性帧也一样（0.515 / 0.02）——不是过拟合，是文本分支没被用。
- checkpoint 检查：`text_emb` std 0.242（Xavier 初始化 0.229，几乎没动）；
  fc 文本列 RMS 0.104 vs 图像列 0.0395（文本列被放大但输入太小）。
  根因：mean-pool 把颜色 token 稀释成 1/7，且图像分支已把 softmax 推到 0.99，
  文本梯度被饱和吞掉。
- 处置（计划 §4.3 的回退项）：`src/model.cpp` 文本分支从 mean-pool 改为
  **逐位置拼接** `[kTextLen × text_dim]`（pad 槽位为 0），`fc.in += 128`；
  反向按位置散回 embedding。`text.hpp`/`model.hpp` 注释同步。
- 测试：`test_text_branch` 的 pad 用例改为「同一 token 换位置结果必须不同」+
  「补零槽位不改变结果」；CTest 21/21 PASS。
- `run-l2-pos16-cf` 已启动（同超参）；决定性帧探针待出。

## 更新 2026-10-31 22:20：M0 结论——late fusion 无法做“按文本选目标”

- `run-l2-pos16-cf`（位置感知 + 反事实）：test 决定性帧 flip 3.9%、acc≈0.5；
  train 决定性帧同样 0.48/0.039——连训练集都拟合不了，不是泛化问题。
- `run-l2-pos16-cf-init`（再把 fc 文本列按自身 fan-in 初始化，放大约 12 倍）：
  flip **2.75%**，更差。位置编码/初始化都不是根因。
- 安全出生集（局部策略采的、与目标无关的 80 集 3200 步）：同一状态按两个
  彩色目标各标一次，动作 **0% 不同**——近障碍状态由避障主导，文本仍然无用。
- **根因（结构性）**：当前是 late fusion，`logits = W_img·img + W_txt·text`。
  同一张图换文本，logit 差是常数，**与目标在画面哪一侧无关**；模型没法表达
  “文本说红 → 去红柱（它可能在左也可能在右）”。这不是数据量或初始化能修的，
  需要文本与图像特征相乘（FiLM/双线性），让文本调制视觉通道。
- 另有一条数据侧偏差：训练状态全部来自朝目标飞的轨迹，图像本身泄漏了目标是
  哪根柱（无文本模型在决定性帧上 0.776），会进一步压低文本的梯度。
- 已改而未提交：`model.cpp` 位置感知拼接 + 文本列单独初始化；
  `test_text_branch` 位置用例；新增 `sim/tools/make_counterfactual.py`、
  `sim/tools/probe_text_effect.py`。CTest 21/21 PASS。
- 结论：v2 任务/数据可保留，**L2 的 fusion 结构需要改**；M2 之前必须先过
  “决定性帧 flip ≥ 50% 且 acc_correct ≥ 0.9”这个语言门槛，否则 SR 差异
  只是导航噪声。

## 更新 2026-11-01 00:55：FiLM 落地 + 审核修复

- `src/model.cpp`：文本分支改为 FiLM（`[kTextLen*text_dim]` → γ/β → 调制 conv
  最后一层），反向按 γ/β 反传；参数 11 个（fc、text_emb、film.W、film.b）。
  `test_text_branch` 增加 FiLM 数值梯度检查（通过）。
- 训练结果（`run-l2-film16-cf`）：决定性帧 flip **12.2%**（concat 3.9%、
  mean-pool 1.6%），仍未达 50%；val 上换色 CE 0.3106 ≈ 原文本 0.3104，
  空文本 CE 1.7281 —— 模型用了“有指令”，但没用“颜色”。
- 训练集里决定性样本只占 13%，被 87% 与颜色无关的样本稀释。新增
  `sim/tools/make_decisive.py`：每个决定性状态生成两条样本（原色/另一色），
  311 源集 → 622 集 / 3566 步，训练集 100% 需要文本；`run-film-dec` 训练中。
- 审核修复：
  - `probe_text_effect.py` 侧车缺行不再静默当空文本，直接退出；
  - `dataset.cpp` 侧车整数先检查再乘，挡住 uint64 回绕；
  - `model.cpp` 无文本时不再深拷贝池化特征（指针复用）；
  - `infer.cpp` 文本模型缺 `--text` 打 stderr 警告，token id 越界直接报错；
  - `make_counterfactual`/`make_decisive` 缺位姿的集直接跳过（不再保留原标签）；
  - 新增 `test_make_decisive.py`；清掉 `cf-smoke`/`dec-smoke`。
- 验证：完整 `bash tools/check.sh` **PASS（rc=0，21/21 CTest + 全部 Python
  测试）**；`test_make_decisive`、`test_audit_dataset`、`test_eval_closed_loop_args`、
  `test_collect` 单独复跑也全 PASS。
- 实验：`run-film-dec`（决定性集，20 轮）完成。决定性帧探针：
  - test：flip **49.4%**、follow 25.5%、acc_correct 0.596；
  - train：flip 58.8%、follow 50.2%、acc_correct 0.787。
  文本第一次真正进入决策（门槛 flip≥0.5 基本达到），但正确文本下的动作准确率
  仍低（0.60 < 0.9），因为只训练了决定性帧、没学导航。
- 高 LR（5e-3）那一路发散（step 23 loss 1.94），已停掉。
- 注意：`ds-dec` 只有 train，没有 val/test，训练器不会写 `ckpt_best.bin`
  （本轮用的是 `ckpt.bin`）；后续决定性集应留出 val 才能做 best 选择。
- 下一步：从 `run-film-dec/ckpt.bin` 在 `ds-cf` 上以 lr 5e-4 微调 10 轮
  （`run-film-dec-ft`），把导航能力找回来，再复测决定性帧与闭环 L/I/S。
