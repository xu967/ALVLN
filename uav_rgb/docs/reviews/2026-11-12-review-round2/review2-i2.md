# 第二轮交叉审核报告 I2：11-12 合进 main 的工具链改动

- 审核角度：I2（打包、流水线、权重平均与校准、统计、门禁，及 `p12-e2e-1` 分支未合并的 `dc8e5ff`）
- 审核基线：开工时主仓库 `main` 为 `ada0ca5`（2026-11-12 07:56:49 CST）；收尾前复核主仓库 `main` 已前进至 `035b398`（期间新合入 W1-15a `f2c1107`、W1-13 第一二部分 `e40c693`、W1-15b `035b398`）。下文行号以开工基线 `ada0ca5` 为准，每条同步注明在 `035b398` 上是否仍存在。
- 起止时刻：2026-11-12 07:56:49 CST 至 2026-11-12 08:15 CST
- 草稿与复现日志：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-i2/`（`verify_i2.py`、`verify_i2.log`、`verify_i2_part2.py`、`verify_i2_part2.log`）

## 1. 结论

本角度范围内最要紧的 3 件事如下：
1. **卡 W1-14（`0d7f3b5`）声称已修的 A4 协议一致性门禁，对常规 Gazebo 评测的 `dynamics_profile` 失效（`I2-1`，已复现）**：`eval_closed_loop.py` 仅在显式传入 `--cf-sitl-params` 时才在 `eval.json` 顶层写 `dynamics_profile`，常规 Gazebo 评测只将其写在 `per_episode[*]` 中，而 `stats_report.load_eval` 只读顶层字段（恒为 `None`），导致跨固件或跨动力学参数的 Gazebo 评测混入同表时 `rc=0` 放行且无混合协议提示。
2. **卡 W1-5（`8fadc83`）、W1-9（`d816d2c`）与 W1-14（`0d7f3b5`）在 `dagger_gz_pipeline.py` 上组合断链，且刚合入的 W1-15b（`035b398`）未覆盖（`I2-2`，已复现 / 读代码确认）**：开启 `student_cond="heading-code"` 并保持默认 `deliver="calswa"` 时，`preflight` 放行但训练后被 main 的 `average_ckpt.py` 拒绝（需先合入 `dc8e5ff`）；`collect.py` 未接入条件一致性闸门，若忘换默认 `round0_ckpt`（`prev_dim=0`）会使第 1 轮自采由纯视觉模型盲跑而最终评测闸门检不出；`summary()` 混入无 `student_cond` 基线调 `stats_report.py` 时触发 A4 `rc=2` 静默丢掉 `stats/` 报表。
3. **统计工具与打包清洗在异常边界下存在口径分裂与静默放行（`I2-3`、`I2-4`，均已复现）**：`stats_report.py` 未提取 `skipped_seeds` 且在两侧共同失败的 `seed` 上将配对表整表误退化回按 `seed` 聚类，使同一方法在主表与配对表的成功率及聚类键不一致；`preprocess.py --exclude-failed-teacher` 用宽泛异常捕获吞掉损坏的 `oracle.jsonl` / `scene.json`（把未过线坏集当「不适用」打包进训练集），并在首行 `NaN` 时把已过线正常集误删。

---

## 2. 发现清单（按严重程度排序）

### I2-1 `stats_report.py` 的 A4 协议门禁只读顶层 `dynamics_profile`，常规 Gazebo 评测跨动力学混表不报错

- **位置**：`tools/stats_report.py:287, 570-578`（合并提交 `0d7f3b5`，子提交 `087d2a8`）；对照 `sim/tools/eval_closed_loop.py:1481-1487, 1577-1579`（合并提交 `41c474d`，子提交 `b436c90`）。在 `035b398` 上原样存在。
- **现象**：
  1. `eval_closed_loop.py` 仅在指定 `--cf-sitl-params` 时才向 `eval.json` 顶层写 `"dynamics_profile"`，常规 Gazebo 评测（不传 `--cf-sitl-params`）顶层无此键，`dynamics_profile` 只存在于 `per_episode[*]["dynamics_profile"]`。`stats_report.load_eval` 仅执行 `e.get("dynamics_profile")`，对常规 Gazebo 产物恒得 `None`（由 `protocol_value` 映射为 `（无）`）。
  2. 带 `--cf-sitl-params` 的满电臂若请求的集全部失败（`results == []`），`dynamics_profile_of([])` 返回 `None`，顶层写出 `"dynamics_profile": null`，被映射为 `（无）`。
- **为什么是缺陷，影响哪些结论或实验**：
  1. `FIXLIST.md` 第 102 行将 A4 标为「已修」，但五个严格协议键中的 `dynamics_profile` 对所有不带 `--cf-sitl-params` 的 Gazebo 评测并未生效。将旧动力学库 `cf_sitl_v1:bd0dcd43`（如 `e1-pipeline-20261107-0417`）与新库 `cf_sitl_v1:1bc8fedf:lib31`（如 `e1-pipeline-20261107-0417-eval4`、`exp-swa-calib-20261110`）混入同一张表同一场景，`stats_report.py` 返回 `rc=0` 且不标混合协议。卡 W1-14 报告（`reports/w1-14.md` 第 93 行）扫描全盘 162 个 `eval.json` 时，120 个横跨多个动力学版本的 Gazebo 文件全部落入同一个「无键」桶，原因就在这里。
  2. 在满电档实验（W1-10）中，若某个带 `--cf-sitl-params` 的臂全部集失败，它会与同批正常满电臂报 `ProtocolMismatch`，反而与未带 `--cf-sitl-params` 的半电旧协议臂被判为协议一致。
- **复现与验证**：草稿脚本 `verify_i2.py` 的 `TEST 3a` 与 `TEST 3b`（日志 `verify_i2.log`）：
  ```bash
  nice -n 15 /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python \
    /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-i2/verify_i2.py
  ```
  实测输出：`e1-pipeline-20261107-0417/eval-student-e1-new/eval.json`（逐集 `cf_sitl_v1:bd0dcd43`）与 `e1-pipeline-20261107-0417-eval4/eval-student-e1-new/eval.json`（逐集 `cf_sitl_v1:1bc8fedf:lib31`）同传给 `build_report`，未抛 `ProtocolMismatch`，`mixed_protocol == {}`。满电正常臂对满电全败臂抛 `ProtocolMismatch`，半电旧臂对满电全败臂放行。
- **建议的最小修法**：
  1. `stats_report.load_eval` 中若顶层 `e.get("dynamics_profile")` 为 `None`，从 `per_episode` 提取 `{ep.get("dynamics_profile") for ep in per if ep.get("dynamics_profile")}`（单值取该字符串，多值拼成排序串以触发不一致报错）；若顶层有 `cf_sitl_params.values_sha256` 而 `per_episode` 为空，回退用 `values_sha256[:8]` 参与比较。
  2. `eval_closed_loop.py` 在带 `--cf-sitl-params` 且 `results` 为空时，由 `cf_sitl_params` 结合 `CF_SITL_LIB_VERSION` 填顶层 `dynamics_profile`（复用 `tools/make_params_variant.py` 的 `profile_of`）。
- **状态**：已复现。
- **归属建议**：新立小修复项（或并入 W1-13 第二轮更新 `stats_report.py` 一致性键时一并修），同步更新 `FIXLIST.md` A4 条目说明。

---

### I2-2 `dagger_gz_pipeline.py` 组合默认 `deliver="calswa"` 与 `student_cond="heading-code"` 时，校准、自采闸门与汇总三处断链

- **位置**：`tools/dagger_gz_pipeline.py:57, 76, 80-82, 505, 524-525, 752-757`（合并提交 `8fadc83`、`d816d2c`）；`tools/e1_pipeline.py:86, 239-245`（合并提交 `8fadc83`）；`tools/average_ckpt.py:210`；`sim/tools/collect.py:966-970`（合并提交 `d816d2c`）。在 `035b398`（W1-15b 合入后）三处均仍存在（对应 `dagger_gz_pipeline.py:59, 78, 82-84, 540, 560-561, 800-805`）。
- **现象**：
  1. **训练后校准崩溃且 `preflight` 不拦截**：W1-5 将两条流水线的 `CFG["deliver"]` 默认设为 `"calswa"`；W1-9 在 `dagger_gz_pipeline.py` 增加 `CFG["student_cond"]`，设为 `"heading-code"` 时训练追加 `--prev-action`（产出 `prev_dim=4` 的 ckpt）。`deliver_check()` 在 `preflight` 不检查条件或文本维度，等第 1 轮采集与 30 轮训练跑完后，main 上的 `tools/average_ckpt.py:210` 因 `(in_c, text_dim, prev_dim) != (3, 0, 0)` 抛 `ValueError` 停机。同理，`e1_pipeline.py` 若设 `CFG["text_dim"] = 16`（l1 任务），即便合入 `p12-e2e-1` 的 `dc8e5ff`，`deliver_check()` 仍不在 `preflight` 拦截，要等采集和训练结束后才在 `calibrate_swa` 报错。`test_dagger_gz_pipeline.py` 未捕获此冲突是因为 `MainTests.setUp` 将 `deliver` 强制改成了 `"best"`。
  2. **`round0_ckpt` 缺条件一致性闸门，污染第 1 轮状态分布**：`dagger_gz_pipeline.py:81` 注释写「非默认时 `round0_ckpt` 也必须是同类条件训的模型（本流水线不校验；评测端闸门兜底）」。但 `round0_ckpt` 只传给 `round_1.collect`（`collect.py --policy student --student-cond heading-code`），从不带 `--student-cond` 进入 `eval_closed_loop.py`；而 `collect.py` 未调用 `nav_cond.check_student_cond`，且 `StudentPolicy.act`（`sim/tools/student.py:121`）在 `self.prev_dim == 0` 时静默丢弃传入的 `prev_action=cond_in`。若开启 `student_cond="heading-code"` 时忘换默认的 `TRUNC_CKPT`（`prev_dim=0`），第 1 轮 100 集实际由纯视觉模型盲采，随后重训出的 `round_1..3` 模型却是真正的 `heading-code` 模型并顺利通过 `eval20` 闸门，全程无报错无警告。
  3. **汇总调 `stats_report.py` 必被 A4 门禁拒绝**：`student_cond="heading-code"` 时，`eval20` 给 `dagger` 与 `more_demos` 传 `--student-cond heading-code`（`eval.json` 含 `student_cond="heading-code"`），给 6 个基线不传（`eval.json` 缺该键）。`summary()` 调用 `stats_report.py` 时未传 `--allow-mixed-protocol`，W1-14 合入后 `stats_report.py` 以 `rc=2` 报错退出，`summary()` 将非零退出码当作不致命忽略，不生成 `out/stats/stats_report.{md,json}`。
- **为什么是缺陷，影响哪些结论或实验**：计划 §10.4 的 E1 正式版与 E3（卡 W2-2，带航向码的 DAgger）直接走 `dagger_gz_pipeline.py` 的 `student_cond="heading-code"`。不修会导致：(1) 未合 `dc8e5ff` 前必在训练末尾崩；(2) 操作者漏改 `round0_ckpt` 时第 1 轮 DAgger 状态分布静默失真；(3) 汇总缺失完整统计配对表。
- **复现与验证**：草稿脚本 `verify_i2_part2.py` 的 `TEST 5a` 与 `TEST 5b`（日志 `verify_i2_part2.log`）：
  - `TEST 5a`：`deliver_check()` 在 `deliver="calswa", student_cond="heading-code"` 下通过，`average_ckpt.calibrate_and_write` 对 `prev_dim=4` 抛 `ValueError`。
  - `TEST 5b`：`dagger_gz_pipeline.summary()` 运行后 `stats/stats_report.md` 不存在，`stats.log` 末行为 `stats_report.py: error: 协议不一致（A4）：场景 gazebo：student_cond：heading-code [dagger] / （无） [trunc]`。
- **建议的最小修法**：
  1. 合入 `p12-e2e-1` 的 `dc8e5ff`；同时在 `e1_pipeline.deliver_check` 与 `dagger_gz_pipeline.deliver_check` 中检查当前 `average_ckpt` 不支持的配置（如 `text_dim > 0`），在 `preflight` 提前拒绝。
  2. 在 `sim/tools/collect.py` 的 `--policy student` 初始化处（拿到 `student.prev_dim` 后）同样调用 `nav_cond.check_student_cond(args.student_cond, args.student_ckpt, student.prev_dim)`，`mismatch` 直接报错退出；并在 `dagger_gz_pipeline.preflight` 核对 `round0_ckpt`。
  3. `dagger_gz_pipeline.summary()` 在 `CFG["student_cond"] != "prev-action"`（或基线与新臂协议不同）时向 `stats_report.py` 传 `--allow-mixed-protocol`，让报表正常产出并带上混合协议表头提示。
- **状态**：第 1、3 点已复现；第 2 点读代码确认。刚合入的 `p12-w1-15b`（`035b398`）未涉及这三处。
- **归属建议**：第 1 点随 `p12-e2e-1`（`dc8e5ff`）合入并补 `deliver_check`；第 2、3 点并入卡 W2-2 开工前的前置修复或卡 E2E-C1。

---

### I2-3 `stats_report.py` 对 `skipped` 集与共同失败 `seed` 处理不一致，导致主表、整簇区间与配对表三处口径分裂

- **位置**：`tools/stats_report.py:280, 330-339, 398, 418-425, 478-482`（合并提交 `0d7f3b5`，子提交 `21ab8bb`、`a6adb57`）。在 `035b398` 上原样存在。
- **现象**：
  1. `eval_closed_loop.py:1107` 在 `eval["skipped"]` 中记录了 `{"seed": seed, "reason": skip}`。主表 `summarize_cell`（及 `paired_stats.success_counts`）以 `episodes_requested` 为分母，将跳过集记 0；但 `load_eval`（`:280`）只提取 `failed_seeds` 未提取 `skipped_seeds`：
     - 配对函数 `run_outcomes`（`:335-339`）只给 `failed_seeds` 补 `False`，漏掉 `skipped` 的 `seed`，导致配对表的 `n_common` 变小、`rate_A / rate_B` 高于主表的 `SR`；
     - 多 run 整簇区间 `cell_clusters`（`:420-424`）把不在 `failed_seeds` 里的 `missing` 差额逐 run 生成 `a[("pad", i, j)] = [0.0, 1]`（簇键含 run 下标 `i`），若多个训练种子在同一几何上跳过，不会聚成同一几何簇，而是裂成多个单 run 有分母、其余 run 分母为 0 的独立假簇。
  2. `pair_units`（`:479-480`）在判断聚类键时执行 `g = ga.get(s, geom.get(s)); use_gid = use_gid and g is not None and g == gb.get(s, geom.get(s))`。若某一个 `seed` 在 A、B 两侧全部 run 中均基础设施失败（不在任何 `per_episode` 中，仅在 `failed` 中），`geom.get(s)` 为 `None`，`use_gid` 变为 `False`，导致整张配对表放弃其余全部已完成几何的 `geometry_gid`，退回按 `seed` 聚类（与 `cell_clusters:398` 只检查完成集的规则不一致）。
  3. `cell_clusters:420` 构造 `fs` 时未对 `failed_seeds` 去重；若 `failed` 列表含同一 `seed` 的重复记录且 `missing > 1`，该 `seed` 在单个 run 内的簇分母会被累加多次，而 `run_outcomes` 按字典键只记一次。
- **为什么是缺陷，影响哪些结论或实验**：W1-2 与 W1-14 的目标是让主表与配对表同分母、同按几何聚类。一旦评测存在 `skipped` 集（如森林高密度档无合格走廊的 seed，见 `exp-action-latency-r2-20261111/forest-L0.20-dense`）或多 seed 同几何场景（campus、3DGS）出现单个共同失败 seed，同一份报表的主表与配对表会出现不同的成功率点估计与不同的聚类粒度。现有柱阵主表实验（`swa-calib`、`heading-guard`、`action-latency-r2` 的 `k1/k3` 学生臂）`skipped=0` 且每 seed 独立 `gid`，已发布数字不受影响。
- **复现与验证**：草稿脚本 `verify_i2.py` 的 `TEST 3c`（日志 `verify_i2.log`）：2 个 run、请求 4 集（`seed` 1、2 共享 `geometry_gid="G_12"`，`seed` 3 两侧共同失败，`seed` 4 在方法 A 跳过、在方法 B 完成失败），实跑 `build_report` 输出：
  - 主表方法 A：`SR = 0.500`（`n = 4`，`cluster_key = "geometry_gid"`，`n_clusters = 4`）；
  - 配对表方法 A：`SR_A = 0.667`（`n_common = 6`，`cluster_key = "seed"`，`n_clusters = 3`）。
- **建议的最小修法**：
  1. `load_eval` 同时提取去重后的 `missing_seeds`（合并 `failed` 与 `skipped` 中的 `seed`，保持顺序去重），`run_outcomes` 与 `cell_clusters` 共用它补齐未完成集。
  2. `pair_units` 对两侧均无 `geometry_gid` 的共同未完成 `seed`，与 `cell_clusters:404` 保持同一规则：其余完成集 `gid` 一致时保持 `key="geometry_gid"`，仅将该未完成 `seed` 以 `("seed", s)` 单独成簇，不推翻整表已完成集的 `geometry_gid` 聚类。
- **状态**：第 1、2 点已复现；第 3 点读代码确认。
- **归属建议**：新立统计补丁（或并入 W1-14 后续维护）。

---

### I2-4 `preprocess.py --exclude-failed-teacher` 静默吞掉损坏的 `oracle.jsonl` / `scene.json`、受首行 `NaN` 污染误丢过线集，且绕过截断检查与空 `val` 检查

- **位置**：`tools/preprocess.py:152-172, 292-303, 502`（合并提交 `f383584`，子提交 `1dad4aa`）。在 `035b398` 上同样存在（对应 `preprocess.py:163-183, 303-314, 536`）。
- **现象**：
  1. `teacher_crossed_goal` 使用两处宽泛异常捕获 `except (OSError, ValueError, TypeError, KeyError, AttributeError): return None`。当 `scene.json` 或 `oracle.jsonl` 存在但内容损坏（例如磁盘满或写中断留下半行截断 JSON，触发 `JSONDecodeError`，属 `ValueError` 子类）时，函数静默返回 `None`（计入 `failed_teacher_na`，即「不适用，未跳」），把原本未过线且文件已损坏的失败教师集原样放进训练集。
  2. `teacher_crossed_goal` 未做 `math.isfinite` 校验：若 `oracle.jsonl` 首行 `pose_gt.x` 为 `NaN`（或 `pillars[0]["x"]` 为 `NaN`），`max_x` 被置为 `nan`，后续正常步 `float(x) > max_x` 恒为 `False`，最终 `max_x >= goal_x` 为 `False`，将实际已过线的正常集静默当成失败教师集整集跳过。
  3. `load_episode`（`:292-303`）在 `:301` 的 `len(rows) != expected`（`actions.jsonl` 行数与 `manifest.num_steps` 一致性检查）之前执行 `exclude_failed_teacher` 并提前返回 `SKIP_FAILED_TEACHER`。若某集因写盘中断导致 `actions.jsonl` 截断（前半截轨迹自然未到成功线），关闭该开关会报错拦截，打开该开关反而静默跳过不报文件损坏。
  4. `scan()` 只检查总样本数 `total == 0`（`:502`），不检查 `val` 分区是否被 `--exclude-failed-teacher` 全部跳空。产出 `splits.val == 0` 的数据集后，若 `E1_DELIVER=best`，`uavrgb_train`（`src/trainer.cpp:458-459`）静默把 `train_idx` 当作 `dev` 选 `ckpt_best.bin`；若默认 `E1_DELIVER=calswa`，训完 30 轮后在 `average_ckpt.py:171` 抛 `val 分区剔除碰撞步后没有样本` 停机。
- **为什么是缺陷，影响哪些结论或实验**：违反工程规范「不静默降级」。0417 数据集文件完整（W1-7 报告核对跳过 26 集、`val` 剩 720 帧），已有打包结果未受影响；后续大批量新采教师数据打包时，一旦混入中断或损坏文件会被静默掩盖或反向混入训练集。
- **复现与验证**：草稿脚本 `verify_i2.py` 的 `TEST 2a`、`TEST 2b`、`TEST 2c`（日志 `verify_i2.log`）：
  - `2a`：向未过线集 `ep2`（`max_x=1.0 < 1.5`）的 `oracle.jsonl` 末尾追加半截 JSON，`preprocess.py --exclude-failed-teacher` 以 `rc=0` 退出并将 `ep2` 打包进 `dataset.uavr`。
  - `2b`：将已过线集 `ep1`（末步 `x=3.0 >= 1.5`）首行 `pose_gt.x` 设为 `NaN`，`preprocess.py --exclude-failed-teacher` 将 `ep1` 误判跳过。
  - `2c`：将 `ep2` 的 `actions.jsonl` 截断为 2 行（`manifest.num_steps=4`），不带 `--exclude-failed-teacher` 报 `rc=1`（文件被截断），带该开关返回 `rc=0` 静默跳过。
- **建议的最小修法**：
  1. 区分「文件不存在（`FileNotFoundError`）或无 `pillars` 键（非柱阵任务）」与「文件存在但 JSON 解析失败 / 数值非有限」：后者直接调用 `die()` 报错退出，不返回 `None`。
  2. 对 `pillar_xs` 与 `pose_gt.x` 校验 `math.isfinite`，非有限值直接 `die()`。
  3. 将 `len(rows) != expected` 截断检查移到 `exclude_collision_episodes` 与 `exclude_failed_teacher` 之前。
- **状态**：第 1、2、3 点已复现；第 4 点结合 `src/trainer.cpp:458` 读代码确认。
- **归属建议**：新立打包健壮性修复项（或并入 W1-13 第二轮）。

---

### I2-5 分支 `p12-e2e-1` 的 `dc8e5ff`（`average_ckpt.py`）未记条件语义标签、忽略契约 `drop_collision` 位且非原子写 `.json`，令 `check_student_cond` 在相对路径或归档后漏拦错配

- **位置**：分支 `p12-e2e-1` 提交 `dc8e5ff` 的 `tools/average_ckpt.py:168-169, 258-263, 289-290`（`main` 同文件 `:168-169, 249-250`）；`sim/tools/nav_cond.py:95-141`（合并提交 `d816d2c`）；对照 `src/class_weights.cpp:242` 与 `src/checkpoint.cpp:266-268`。
- **现象**：
  1. `uavrgb_train`（`src/class_weights.cpp:242`）把命令行的 `--data` 字符串原样写入 `run/config.json`（不转绝对路径）。`nav_cond.check_student_cond` 只从 `config.json` 读取 `cfg["data"]` 并按当前工作目录打开，不按 `config.json` 所在目录或仓库根解析，也不读取 `<ckpt>.json` 里已验证过训练契约 SHA256 的绝对路径 `"data"`；同时 `dc8e5ff` 在 `<out>.json` 新增的 `"cond"` 字典只写了 `"column": "prev_action"` 与 `"val_value_counts"`，未写入数据集 `meta.get("cond")`（即 `"heading_code.v1"` 或 `None`）。只要训练用了相对 `--data` 并在另一目录跑评测，或者因数据盘空间吃紧（当前剩余 14 GB，清理任务在推进）归档删除了数百 MB 的训练 `.uavr` 文件，`check_student_cond` 立即返回 `status="unverified"` 仅打警告放行。
  2. `average_ckpt.py` 已读入 113 字节训练契约 `last["contract"]` 并校验了前 32 字节 `dataset_sha256`，但忽略了第 33 字节 `last["contract"][32] & 1` 的 `drop_collision` 标志（`src/checkpoint.cpp:268`），在 `read_val` 中无条件剔除 `collision_flags[i] == "1"` 的样本（延续 `FIXLIST.md` C6）。
  3. `average_ckpt.py` 写模型权重用临时文件加 `os.replace` 原子替换，但写旁路 `<out>.json` 直接 `open(out + ".json", "w")` 非原子写。若进程在写 `.json` 时被 `SIGKILL` 中断留下空文件或半截文件，两条流水线的续跑判定 `os.path.isfile(swa) and os.path.isfile(swa + ".json")`（`e1_pipeline.py:260`、`dagger_gz_pipeline.py:503`）会误判校准已完成并直接跳过。
- **为什么是缺陷，影响哪些结论或实验**：航向码模型训练时码 `1` 从不出现（只用码 `0, 2, 3`）；若评测时漏传 `--student-cond heading-code`（默认 `prev-action`）且触发第 1 点的 `unverified` 退化，评测端不会拦截，而是把上一步 `FORWARD_SLOW(1)` 当作码 `1` 持续喂给模型，产出分布外失真结果。E2E-1 当前 `prepare.sh` 使用绝对路径且保留了 `demo-hc/dataset.uavr`，在跑实验未触发该退化。
- **复现与验证**：草稿脚本 `verify_i2_part2.py` 的 `TEST 6`（日志 `verify_i2_part2.log`）：模拟以相对路径 `--data ds/dataset.uavr` 训练航向码模型并写出 `dc8e5ff` 格式的 `cal.bin.json`，切到仓库根目录对 `cal.bin` 调用 `nav_cond.check_student_cond("prev-action", cal_ckpt, 4)`，实测返回 `status='unverified'`（未能拦下 `"prev-action"` 与 `"heading-code"` 的错配）。
- **建议的最小修法**：
  1. `dc8e5ff` 的 `read_val` 顺带返回数据集 `meta.get("cond")`，写入 `<out>.json` 的 `cond["dataset_cond"]`，并将 `<out>.json` 改为写 `.tmp` 后 `os.replace`；流水线 `calibrate_swa` / `calibrate_into` 用 `read_json` 校验 `.json` 可解析再跳过。
  2. `nav_cond.check_student_cond` 在 `<ckpt>.json` 存在且记录了 `data` / `dataset_cond` 时优先用绝对路径 `data`（或核对 SHA 后用 `dataset_cond`）判定。
  3. `read_val` 按 `(last["contract"][32] & 1) != 0` 决定是否过滤 `collision_flags`（与训练时验证集口径对齐）。
- **状态**：第 1 点已复现；第 2、3 点读代码确认。
- **归属建议**：第 1、3 点在合并 `p12-e2e-1` 的 `dc8e5ff` 时顺手补齐（不超过 15 行）；第 2 点与 checkpoint 升版一起并入卡 E2E-C1。

---

### I2-6 `preprocess.py` 行级过滤把整集滤空时保留 `sample_count=0` 空壳集（带指令侧车时令 C++ `load_uavr` 崩溃），且跳号步下 `--cond heading-code` 静默吞错

- **位置**：`tools/preprocess.py:327-336, 350-373, 485-492, 719-723`（合并提交 `d816d2c`、`e41703e`），对照 `src/dataset.cpp:376-380` 与 `035b398` 中刚合入的卡 W1-13 提交 `aa9a6ff`（`--drop-late-and-stall`）。
- **现象**：
  1. 当某个 episode 的全部步被 `--drop-collision` 或刚合入的 `--drop-late-and-stall`（机器高负载下整集迟到或停顿易发生）全部跳过时，`load_episode` 返回 `samples == []`，`scan()` 仍将该集加入 `episodes`（`sample_count = 0`），写入 `meta["episodes"]`、`meta["scenes"]`、`geometry_versions[*]["n_episodes"]` 与 `instructions.jsonl`。若包内出现两个及以上 `sample_count == 0` 的相邻空壳集（两者 `sample_start` 相同、`sample_count` 均为 `0`）且带 `instructions.jsonl`，C++ 读端 `load_text_sidecar`（`src/dataset.cpp:376-379`）因 `(start, count)` 匹配多个 episode 直接抛 `std::runtime_error("UAVR: instructions.jsonl 第 ... 行区间匹配多个 episode")` 崩溃。
  2. `load_episode` 主循环（`:327-335`）仅在 `by_step.get(step - 1)` 非空时校验上一步的 `action_executed`。若输入 `actions.jsonl` 因离线裁剪存在 `step` 跳号（如 `step = 0, 2`），`step=0` 的 `action_executed` 未被主循环校验；`:366` 的 `executed_or_none` 捕获 `TypeError / ValueError` 静默返回 `None`（且越界整数如 `99` 也不报错），当作非转向处理；同时默认 `--cond prev-action` 将 `step=2` 的 `prev_action` 置 `0`（当首帧），`--cond heading-code` 却累加了 `step=0` 的转向，两种条件模式在跳号数据上语义不一致。
- **为什么是缺陷，影响哪些结论或实验**：`collect.py` 原始产出的 `step` 连续且柱阵无指令，当前 E2E-1 打包不受影响；但 W1-13 合入 `--drop-late-and-stall` 后整集被滤空的概率上升，后续 E5 语言任务重采打包一旦出现两个全滤空集即会导致 C++ 训练器无法加载数据集。
- **复现与验证**：草稿脚本 `verify_i2.py` 的 `TEST 2d` 与 `TEST 2e`（日志 `verify_i2.log`）：
  - `2d`：`ep2`、`ep3` 全碰撞加 `--drop-collision` 与 `instruction`，写出 `episodes` 与 `instructions.jsonl` 区间均为 `[(1, 0, 5), (2, 5, 0), (3, 5, 0)]`，对应 `src/dataset.cpp:376-379` 的重复区间异常条件。
  - `2e`：`step = 0, 2` 且 `step=0` 的 `action_executed="BAD_STR"`，`preprocess.py --cond heading-code` 返回 `rc=0` 未报错。
- **建议的最小修法**：
  1. `scan()` 中当 `load_episode` 返回的 `ep["samples"]` 为空时，将其作为被行级过滤整集淘空的集跳过并计数打印，不写入 `episodes` 与 `instructions.jsonl`。
  2. `load_episode` 主循环对每一行自身存在的 `action_executed` 直接校验 `int` 与 `0..3` 范围（不依赖 `step - 1` 查找），删除 `executed_or_none` 的异常吞没。
- **状态**：已复现（C++ 异常触发条件读 `src/dataset.cpp:376-379` 确认）。
- **归属建议**：并入卡 W1-13 第二轮（与 `FIXLIST.md` C7 打包丢步侧车处理一起收口）。

---

### I2-7 `dagger_gz_pipeline.py` 的 `summary()` 声明「SR 分母 = 请求集」，实际打印 `success / episodes_done` 且配对检验漏算失败集

- **位置**：`tools/dagger_gz_pipeline.py:694-706, 723, 741, 744, 747`（在 `035b398` 对应 `:742-754, 771, 790, 793, 796`）。
- **现象**：
  1. `summary()` 第 747 行写明「口径：SR 分母 = 请求集」，`wilson95` 也取 `wilson_ci.requested`；但第 695 行令 `rows[name]["n"] = e.get("episodes_done")`，第 741 行打印 `| {name} | {r['success']}/{r['n']} |`（分母用完成集），第 723 与 744 行 `eval60` 打印 `{r['SR']} ({r['n']})` 时 `SR` 用 `requested_success_rate` 而括号内 `n` 却是 `episodes_done`。
  2. 第 694 与 705 行计算 `mcnemar_vs_dagger` 时，`per[name]` 仅收集 `per_episode`，未按卡 W1-14（`a6adb57`）口径将 `failed` 中的失败 `seed` 记为 `False`（0）。
- **为什么是缺陷，影响哪些结论或实验**：W1-2（`670c4b4`）与 W1-14（`0d7f3b5`）统一了全仓统计分母，唯独漏了 `dagger_gz_pipeline.py` 内置简表（W1-15b `035b398` 只修了 `dagger_pipeline.py` 的 E-7）。当复用旧评测或存在失败集时，同一个单元格内的分数、Wilson 区间与配对检验使用两套不同分母。
- **复现与验证**：草稿脚本 `verify_i2_part2.py` 的 `TEST 5b`（日志 `verify_i2_part2.log`）：请求 20 集、完成 18 集、失败 2 集、成功 9 集时，`summary.md` 打印 `| dagger | 9/18 | [0.26, 0.66] |`，与 `trunc`（10/20）配对的 `b,c` 从请求集口径的 `0,3` 变成 `0,1`。
- **建议的最小修法**：`summary()` 改调 `paired_stats.success_counts(e)` 取 `k` 与 `denominator`（`n` 记 `denominator`），配对字典改调 `stats_report.run_outcomes(stats_report.load_eval(path), "success", requested=True)`。
- **状态**：已复现。
- **归属建议**：并入卡 W2-2 或 W1-15b 后续收尾。

---

### I2-8 `make_params_variant.py` 允许整数字段被改为浮点数并改变动力学指纹；`preprocess.py` 解析深度在越过柱顶时圆柱与方柱求交不一致

- **位置**：`tools/make_params_variant.py:114-119`（合并提交 `41c474d`，子提交 `786fc3f`）；`tools/preprocess.py:1021-1046`（合并提交 `e41703e`，子提交 `7dd319d`）。在 `035b398` 上均存在。
- **现象**：
  1. `make_params_variant._coerce` 写为 `return float(new) if isinstance(old, float) else new`。当源参数 `old` 为 `int`、命令行传入浮点字面量（如 `25.0` 或 `25.5`）时，不会转为 `int` 也不会拒绝小数，而是直接写入 `float`。对整数项传 `25.0` 会使 JSON 文本由 `25` 变为 `25.0`，导致规范化 `values_sha256` 与 `dynamics_profile` 改变，违背文件头第 20 行「同一个数不该有两个指纹」与 `self_check`「原值是整数就保持整数」的约定。
  2. `preprocess._pillar_hit_t` 在相机高度 `oz` 高于柱顶 `zc + zh` 时：方柱分支做三维 slab 求交，向下射线会命中顶面 `z = zc + zh`；圆柱分支只解无限高圆柱侧面二次方程再判交点高度 `z_hit <= zc + zh`，不求顶盖圆盘交点，同一条打向柱顶的射线判为未命中 `+inf`，直接穿透整根圆柱打到地面。
- **为什么是缺陷，影响哪些结论或实验**：`e1_mini.json` 现有修改项 `sim.battery_soc_init` 原值为 `float`（`0.5`），且当前柱阵与 `forest_e1` 柱高 2.4–3.0 m 高于稳态飞行高度（1.0–1.5 m），W1-8 与 W1-10 已有实验数字不受影响。后续若对整型参数做变体扫描，或无人机爬升越顶、场景引入矮圆柱障碍，会产生冗余指纹或深度跳变。
- **复现与验证**：草稿脚本 `verify_i2.py` 的 `TEST 4` 与 `TEST 2f`（日志 `verify_i2.log`）：
  - `TEST 4`：原值 `steps=25`（`int`），传 `steps=25.0` 后变为 `25.0`（`float`），`values_sha256[:8]` 由 `9b6bd79f` 变为 `58831714`；传 `steps=25.5` 未报错。
  - `TEST 2f`：`x=1.0 m` 处高 `2.0 m` 的同外接尺寸方柱与圆柱，相机在 `z=2.1 m`（`oz=2.115 m`）处，`row=9, col=10` 格方柱最小深度 `0.854 m`，圆柱穿透至地面得 `15.700 m`。
- **建议的最小修法**：
  1. `_coerce` 中若 `isinstance(old, int)`：当 `isinstance(new, int)` 或（`isinstance(new, float)` 且 `new.is_integer()`）时返回 `int(new)`，否则抛 `ValueError` 拒绝给整型参数赋非整数。
  2. `_pillar_hit_t` 圆柱分支在 `dz != 0` 时补算顶底圆盘面 `z = zc ± zh` 交点（满足 `(ox + t*dx - px)^2 + (oy + t*dy - py)^2 <= r^2` 且 `t >= 0`），与柱面交点取最小正值（或明确按相机始终低于柱顶限制方柱同样不计顶面，使两形状一致）。
- **状态**：均已复现。
- **归属建议**：低优先级维护项，随下次触及对应文件时修正。

---

### I2-9 门禁 `tools/check.sh` 对 `--depth-source analytic` 与 `--cond heading-code` 侧车的 C++ 读端零覆盖，且 `build_meta` 第三层隔离门无独立测试

- **位置**：`tools/check.sh:57-66, 106-146`（合并提交 `ceac416`），对照 `tools/read_dataset.cpp:306` 与 `tools/preprocess.py:631`。在 `035b398` 上仍存在（原 `ada0ca5` 上 `check.sh:40` 的 `/tmp` 与字节码问题 E-13 已在 `035b398` 由卡 W1-15a 的 `4b43a9e` 修好）。
- **现象**：
  1. `check.sh:57-66` 的「Python `preprocess.py` 产出 → C++ `read_dataset.cpp` 校验」链路只在全量模式（非 `--python-only`）下运行，且只用默认参数打包（无深度、`prev-action`）；同时 `read_dataset.cpp:306` 在 `--selfcheck` 下硬编码 `if (has_depth) return fail("aux selfcheck: 合成集没有深度")`。因此 11-12 合入的 `--depth-source analytic`（`UAVRAUX1` `flags=1` 深度段）与 `--cond heading-code`（`meta.cond`）从未在门禁中经过任何 C++ 读端（`read_dataset.cpp` 或 `src/aux_sidecar.cpp`）校验。
  2. 卡 T1（`54ae6bb`）修复 F7 时拆出了前两层几何隔离负例（`load_episode` 的 split 复算门与 `scan` 的 `gid` 唯一性门），但 `preprocess.build_meta:631` 的第三层门（「几何组 … 跨 split」，供 `airsim_to_uavr.py` 等绕过 `scan` 直接构造 episode 字典的调用方防泄漏）仍无单测锁住，单独注释掉 `:631-633` 时 `check.sh` 全绿。
- **为什么是缺陷，影响哪些结论或实验**：门禁存在跨语言契约盲区，后续修改 `UAVRAUX1` 深度段布局或 `meta` 字段顺序时，`--python-only` 与全量 `check.sh` 的合成集自检都不会覆盖带深度侧车的 Python→C++ 通路。本次审核已在草稿 `verify_i2.py` 的 `TEST 1` 中编译 `read_dataset.cpp` 实跑三新开关组合产物并通过校验，确认当前 `main` 格式无漂移。
- **复现与验证**：读 `tools/check.sh:57-66` 与 `tools/read_dataset.cpp:306` 确认；三开关组合过 `read_dataset` 的正向验证见 `verify_i2.py` `TEST 1`。
- **建议的最小修法**：
  1. 在全量 `check.sh` 第 66 行之后，对加了 `scene.json` 与位姿的合成集跑一次 `preprocess.py --cond heading-code --depth-source analytic --exclude-failed-teacher`，再用 `$uav_check_fmt/read_dataset <dataset.uavr>`（不带 `--selfcheck` 或扩展 `--selfcheck`）与 C++ `load_uavr` 校验。
  2. 在 `tools/test_preprocess_split.py` 增加一条直接调用 `preprocess.build_meta` 传入跨 split 同 `gid` 字典的单元测试，锁住第三层防线。
- **状态**：读代码确认。
- **归属建议**：并入卡 T1 后续批次或卡 E2E-C1。

---

## 3. 第一轮修复清单（`FIXLIST.md`）中本角度相关条目的核对结果

### 3.1 修对了的条目

- **D2（`0d7f3b5` / `401be6b`）**：`tools/paired_stats.py` 的动态规划精确符号翻转检验（`k <= 20`）、蒙特卡洛保守修正 `(hit + 1) / (n_draws + 1)`、零差簇剔除、整簇有放回 bootstrap 百分位区间，以及 `stats_report.py` 多 run 配对表的接入，公式与实现核对无误。
- **D5（`670c4b4`、`0d7f3b5` / `ceb9d2d`）**：单 run 内重复 `seed` 在 `load_eval` 与 `metric_values` 均抛 `InputError`（CLI `rc=2`）；多 run 按 `(run, seed)` 对齐不再互相覆盖。
- **D7（`0d7f3b5` / `86c64be`）**：`paired_stats.success_counts` 分母与 `stats_report.run_denominator` 对齐，支持字典、文件路径与目录路径。
- **F3（`ceac416` / `0a0cd5d`）**：`tools/test_preprocess_label_alignment.py` 锁住了标签取 `action_expert`、丢碰撞步后图像与原 `step` 对齐及乱序 `step` 排序。
- **W1-6 / F6 的 3DGS 部分（`499bec9` / `0c10dd1`）**：`tools/check.sh` 在 `--python-only` 与全量模式下均遍历 `sim/3dgs/test_*.py`，解释器缺失时打印 `SKIP` 并记入末行 `uav_check_skipped`。
- **收尾前刚合入 `035b398` 的条目（核对 `e40c693`、`f2c1107`、`035b398`）**：E-3（三条流水线 `.pipeline.lock`）、E-4（`run.pid` 四行格式与 `stop_daemon.sh`）、E-7（`dagger_pipeline.py` 请求集分母）、E-8（`pipeline_common.eval_state` 核对 `skipped_count==0`、`done==requested`、`metric_version` 与 `ckpt`，退居 `.failed-` / `.stale-` 并写入 `retired_dirs.jsonl`）、E-9（采集/打包/训练半成品退居 `.partial-`）、E-13（`check.sh` 设 `PYTHONDONTWRITEBYTECODE=1` 与数据盘 `TMPDIR`）、A2 的流水线与 `collect_parallel` 透传部分，代码逻辑核对正确。

### 3.2 修了一半或存在漏洞的条目

- **A4（`0d7f3b5` / `087d2a8`，`FIXLIST.md` 标「已修」）**：`action_latency_s`、`heading_guard`、`rtf`、`student_cond` 四个键有效，但第五个键 `dynamics_profile` 因只读 `eval.json` 顶层，对所有不传 `--cf-sitl-params` 的常规 Gazebo 评测完全漏检（见 **I2-1**）。
- **D4 与 W1-2 遗留（`0d7f3b5` / `21ab8bb`、`a6adb57`，`FIXLIST.md` 标「已修」）**：无 `skipped` 且无全 run 共同失败 `seed` 时正确；存在 `skipped` 集或共同失败 `seed` 时，主表、整簇区间与配对表口径分裂（见 **I2-3**）；且 `dagger_gz_pipeline.summary()` 内置表未同步（见 **I2-7**）。
- **C1（`p12-e2e-1` 未合并的 `dc8e5ff`）**：已按计划补齐单帧无文本 `prev_dim=4` 模型的侧车条件读取与校准，无条件模型逐字节不变；但 `<out>.json` 未记数据集 `cond` 标签且非原子写，在相对 `--data` 或 `.uavr` 清理后会使评测闸门退化为 `unverified`（见 **I2-5**），且未合进 `main` 导致当前 `main` 上 `dagger_gz_pipeline` 的 `heading-code + calswa` 组合断链（见 **I2-2**）。双帧与文本按计划留给后续卡。
- **F7（`ceac416` / `54ae6bb`）**：前两道门（`load_episode` 与 `scan`）已拆开并各自用独有报错文本断言；第三道门（`build_meta:631`）仍无单测覆盖（见 **I2-9**）。

### 3.3 尚未修（按计划排期在后续卡）的条目

- **C6**：`average_ckpt.py` 硬编码剔除碰撞步，不读 checkpoint 训练契约的 `drop_collision` 位（排卡 E2E-C1，见 **I2-5** 第 2 点）。
- **C7**：打包丢步后 `prev_index` 指向自身而 `prev_action` 保留真实上一步动作（`035b398` 已加 `shortcut` 注释备案，主线单帧不受影响；另见 **I2-6** 的整集滤空问题）。
- **D6 / H1**：全局单一标量 `δ` 校准可能压制航向码非零帧的回转（按计划排在卡 E2E-1b）。
- **D1 的三个 `tools/` 调用方遗留**：`tools/dagger_pipeline.py`、`tools/train_student_with_video.py`、`tools/compare_datasets.py` 调用 `eval_3dgs` 时未传 `--allow-train-scene`（`FIXLIST.md:139` 已登记归下一轮）。
- **E-2 的 Python 流水线部分**：三条 Python 流水线仍运行在活工作树上（未做 `git archive` 代码冻结），且续跑重写 `config.json` / `inputs.json`（W1-15a 只为 shell 入口实现了 `exp_lib.sh` 冻结与追加记录，W1-15b 范围不含此项）。

---

## 4. 查过但没有发现问题的地方

1. **`tools/preprocess.py` 三个新开关（`--cond heading-code`、`--depth-source analytic`、`--exclude-failed-teacher`）与 `--drop-collision` 同时开启在合法输入下的对齐**：通过 `verify_i2.py` 的 `TEST 1` 构造含过线教师集（含碰撞转向步）、未过线教师集、未过线 DAgger 学生自采集、边界过线教师集的 4 集夹具并实跑，逐样本核对了 14 个保留样本的 RGB 像素、`action_expert` 标签、跨丢弃步累计的 `heading_code`、`prev_index` 回指链、按保留步位姿计算的 `15×20` 解析深度分带、`meta` 的 `cond` / `aux_sidecar.depth_source` / `collision_filter`，并用编译的 C++ `tools/read_dataset.cpp` 校验文件头、双 SHA256 与侧车载荷，全部逐项对齐。三个新开关全部缺省关闭时，输出与改动前逐字节一致。
2. **`tools/` 下三次合并冲突的解决（`d816d2c`、`f383584`、`e41703e`）**：逐个对比 `git diff <merge>^1 <merge>` 与分支原提交：`d816d2c` 在 `dagger_gz_pipeline.py` 完整保留了 W1-5 的 `deliver/avg_last_k/gpu/--save-every-epoch` 与 W1-9 的 `student_cond/--prev-action`；`f383584` 在 `preprocess.py` 同时保留了 `scan(..., exclude_failed_teacher, skip_stats)` 与随后的 `heading-code` waypoint 拦截；`e41703e` 在 `aux_payload` 同时保留了 `cond` 与 `depth_source` 关键字传参，无逻辑丢失。
3. **`preprocess.py` 解析深度的核心几何投影与焦距/包络膨胀方向**：`analytic_ray_depth` 中射线方向 `d = fwd - xr*left - yd*up` 在光轴 `fwd` 上的投影恒为 `1`，参数 `t` 直接等于沿光轴的 z-深度（无需额外乘 `cos φ`）；方位角左正、yaw 逆时针为正、`camera_mount` `(0.045, 0.015)` 与 `sim/dynamics/e1_mini.json` 一致；多集不同 `camera.fx` 时逐集独立做包络膨胀。
4. **`tools/e1_pipeline.py` 与 `tools/dagger_gz_pipeline.py` 的走廊判据透传（W1-1 `25e4b28`）**：`corridor_half_width` 默认 `3.0`，`> 0` 时统一传 `--corridor-half-width 3`，设 `E1_CORRIDOR_HALF_WIDTH=0` 时不传开关回到旧口径；`dagger_gz_pipeline.reusable` 按 `metric_version` 自动拒绝复用无走廊的旧 `EVAL4` 基线。
5. **`p12-e2e-1` 提交 `dc8e5ff` 的侧车校验与行协议实现**：`read_aux_cond` 对 `UAVRAUX1` 的 magic、`N`、`flags`、`unknown_code`、`uavr_sha256`、文件总长、`payload_sha256` 及 `cond < 4` 逐项校验，与 `src/aux_sidecar.cpp` 一致；`infer_probs` 先核对 `hello["prev_action_dim"]` 再在每帧 HWC 字节后追加 1 字节条件，与 `sim/tools/student.py` 及 `src/infer.cpp` 一致；`src/infer.cpp:108` 以默认浮点科学计数法输出 `double` 概率，不会因定点数小数位截断把小概率变成 `0`。
6. **W1-7 的 `tools/export_costmap_soft.py` 与 `tools/relabel_costmap.py`**：`export_costmap_soft.py` 要求全部输入集同属单一可复算 costmap 教师版本（拒绝 v1 与 v2 混导），并将实际版本写入 npz `meta`；`relabel_costmap.py` 显式传入 `version=T.TEACHER_COST_VERSION`，与 `audit_dataset.py` 版本分派一致。
7. **`tools/paired_stats.py` 的数学实现（W1-14 `0d7f3b5`）**：精确符号翻转 DP 分布、非整数簇差拦截、整簇有放回重采样线性插值分位数及单簇退化处理核对无误，且无第三方库依赖。

---

## 5. 没有查的地方与原因

1. **未启动 Gazebo、未运行 3DGS 批量渲染、未运行 GPU 训练或 `uavrgb_infer` 真实 GPU 推理、未跑 `tools/check.sh`**：遵守任务书第 3 节只读与算力隔离规则（机器 12 核已满载运行闭环与训练实验）。`average_ckpt.py` 的 GPU 推理路径通过读源码及核对 `e2e-1.md`、`w1-5.md` 的实跑 sha256 记录审查。
2. **未对 `035b398` 新合入的 `tools/exp_lib.sh`、`tools/stop_daemon.sh`（卡 W1-15a）做全量独立逐行审计**：本角度任务书指定范围为开工时已合并的 11-12 工具链改动加 `dc8e5ff`，W1-15a 在开工时尚未合并；仅核对了它对 E-4 与 E-13 的修复状态。
3. **未审查 `experiments/2026-11-12-*` 下的实验专属汇总脚本（如 `summarize_e2e.py`、`recheck_tables.py`）与 `src/` C++ 训练核心**：分属审核角度 M 与角度 K/L/N 的范围。
