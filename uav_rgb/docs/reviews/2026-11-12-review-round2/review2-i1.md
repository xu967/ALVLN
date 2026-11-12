# 第二轮交叉审核报告：I1（11-12 合进 main 的 Gazebo 线改动）

## 1. 审核基准与结论

- 审核对象：主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` 分支 `main`，开工提交 `ada0ca5`（完整 SHA：`ada0ca5ecd357850b7df63c7a7b8c705ff8f324c`），对比基线标签 `base-20261112`。审核期间主会话继续向 `main` 合入了 `f2c1107`（W1-15a）、`e40c693`（W1-13 第一、二部分）与 `035b398`（W1-15b）；下文文件与行号一律对齐开工基准 `ada0ca5`（只读快照在 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-i1/tree-ada0ca5/`），已被 `p12-w1-13`（`e40c693`）覆盖或未覆盖之处逐条注明。
- 起止时刻：2026-11-12 07:59 CST 至 2026-11-12 08:23 CST。
- 证据脚本：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-i1/verify_batch1.py`、`verify_batch2.py`。

结论：本角度下最要紧的 3 件事如下。第一，教师 v7 的 B3（A\* 无路降余量重试，`f383584`）只降低了 `_global_path` 的栅格膨胀余量，未同步放宽 `plan_action_global` 第 581 行前向 0.4 m 的 `LOCAL_CLEARANCE_M` 检查；在 `0417` 真实数据集被 B3 救活的 349 个非碰撞步中，245 步（70.2%）因处在 `LOCAL_CLEARANCE_M` 内，机头转准路径后仍被否决回退局部规则，165 步（47.3%）陷入 `YAW_LEFT` 与 `YAW_RIGHT` 两步死锁摆振，且 `--action-latency` 预热未覆盖重试余量，K=3 长课程冷触发 B3 单步达 456–719 ms，会直接造成迟到废集（`I1-1`、`I1-4`）。第二，满电档开关 `--cf-sitl-params`（`41c474d`）在 `collect_parallel.py` 只接了 `scene_gen_extra`，漏接命令行参数与 `collect_command`，批量满电采集命令行不可用，按模块传入参数时还会产生「世界 SDF 跑 100% 满电物理、manifest 记 50% 半电指纹」的静默错配（`I1-2`）。第三，航向码 `--student-cond`（`d816d2c`）在采集端 `collect.py` 与 `collect_parallel.py` 缺少一致性闸门，三处入口均未拦截带 `waypoint` 的任务（`l1` / `campus`），评测端闸门 `nav_cond.check_student_cond` 忽略同目录续训的 `config.resume.json` 且不核对 ckpt 内嵌的数据集契约哈希（`I1-3`、`I1-6`）。

---

## 2. 发现清单（按严重程度排序，共 10 条）

### `I1-1`：教师 v7 的 B3（A\* 无路降余量重试）未同步放宽前向 0.4 m 净空检查，65.3% 的救活步对准路径后无法前进、47.3% 陷入两步死锁，且 `x3` 载具重试余量低于机身半宽

- 位置：`sim/tools/teacher.py:183`、`:502-516`、`:581-587`、`:722`（合并提交 `f383584`，卡 W1-7 提交 `009bca2`）。
- 现象：
  1. `_astar_retry` 在标准余量（`e1_mini` 为 0.40 m，`x3` 为 0.50 m）无路时按 `ASTAR_RETRY_CLEARANCES = (0.30, 0.20)` 重试，只返回 `path` 而不返回命中的余量档位；`plan_action_global` 在 `|err| <= 25°` 后，第 581 行固定用 `LOCAL_CLEARANCE_M`（`e1_mini` 为 0.30 m，`x3` 为 0.36 m）执行 `ray_clear(x, y, yaw_deg, 0.4, pillars, LOCAL_CLEARANCE_M)`。
  2. 当机体距柱表面小于 `LOCAL_CLEARANCE_M` 时，`ray_clear` 在起点 `t=0` 恒返回 `False`，B3 路径在机头对准的瞬间被否决并退回 `plan_action`。在 `/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417/collection` 的 300 集 12,000 步上实测：B3 救活的非碰撞无路步共 349 步，其中 245 步（70.2%）落在 `LOCAL_CLEARANCE_M` 内；将机头直接置为 B3 胡萝卜方向（`err = 0`）时，仅 121 步（34.7%）输出 `FORWARD_SLOW`，其余 228 步（65.3%）输出 `YAW_LEFT`（127）、`YAW_RIGHT`（66）或 `SAFE_NEUTRAL`（35）；按 26°/步从原航向做纯偏航推进 8 步，165 步（47.3%，如成功过线集 `3137` 步 9–10、失败集 `3149` 步 12–14）因全局 A\* 与局部 `plan_action` 选边相反，在 `|err| = 25°` 边界交替输出 `YAW_LEFT` 与 `YAW_RIGHT`。`teacher.py:1209` 自检自己的样例 `wall` 在 `(1.10, 0.0)` 从 `yaw=0°` 连转 3 步到 `78°` 对准路径（胡萝卜 `79.7°`）后，第 4 步起同样退回 `SAFE_NEUTRAL` 原地停死。
  3. `ASTAR_RETRY_CLEARANCES = (0.30, 0.20)` 未按载具过滤：`x3` 的真实机身半宽 `BODY_HALF_BY_VEHICLE["x3"] = 0.235 m`（`teacher.py:153`），0.20 m 档位低于机身半宽；同时 `x3` 的 `ESCAPE_V2 = False`（第 584 行不进 `plan_action`），在 `(1.05, 0.0)` 面对单柱时从 v5 的 `SAFE_NEUTRAL` 变为持续输出 `YAW_RIGHT`。
- 为什么是缺陷及影响：W1-7 离线统计只看 `_global_path(...) is not None` 与单帧静态重标，未检出转准后被第 581 行拦截；20 个冻结 seed 的短柱阵教师闭环未进入 `< 0.30 m` 区域所以 20/20 未暴露。后续 E3（新采长课程教师数据、DAgger 影子教师用 v7 给偏离靠近柱子的学生状态打标签）会在近柱区产生左右交替或 `SAFE_NEUTRAL` 标签，污染监督。分支 `p12-w1-13` 不涉及此文件。
- 复现与验证：
  `env TMPDIR=/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-i1/tmp PYTHONDONTWRITEBYTECODE=1 nice -n 15 /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-i1/verify_batch1.py`（第 1、2 节）及 `verify_batch2.py`（第 A 节）。
- 建议的最小修法：让 `_astar_retry` 与 `_global_path` 带回命中的余量 `used_clearance`，`plan_action_global` 第 581 行的前向检查余量取 `min(LOCAL_CLEARANCE_M, used_clearance)`（或对降余量命中的步传 `skip_clearance_m=0.2` 仅查 `_body_hits_box`）；重试档位增加 `c > BODY_HALF_M + 0.05` 约束以排除 `x3` 的 0.20 m。因改变教师动作，须升版为 `uavrgb.teacher.v8` 并在 `TEACHER_RULES_BY_VERSION` 保留 `v7` 规则供审计复算。
- 状态：已复现。
- 归属建议：在 E3 采集开工前新立教师修订小卡（或并入 W2-2 / W2-6 的前置步骤）。

### `I1-2`：`collect_parallel.py` 的 `--cf-sitl-params` 只接了 `scene_gen_extra` 未接 CLI 与 `collect_command`，批量满电采集跑不通，模块调用会造出「满电物理 + 半电指纹」

- 位置：`sim/tools/collect_parallel.py:204-205`、`:403-443`、`:600-674`、`:872-961`（合并提交 `41c474d`，卡 W1-10 提交 `b436c90`）。
- 现象：W1-10 报告（`w1-10.md:10`）与探针文档（`probe_plugin_params.py:6`）写明「`collect_parallel` / `eval_closed_loop` 透传 `--cf-sitl-params`」，但实际改动只在 `collect_parallel.scene_gen_extra(args)`（供 `eval_closed_loop.py:828` 调用）加了 `--cf-sitl-params`；`collect_parallel.py` 的 `main()` 没有定义 `--cf-sitl-params` 参数，`collect_command(task, args)` 没有向 `collect.py` 透传 `--cf-sitl-params`，`collection_manifest.json` 也没有记录该字段。
- 为什么是缺陷及影响：
  1. 计划 §10.4 的 E3 / W2-6 要求用满电档新采多块课程与 DAgger 数据，`collect_parallel.py` 命令行传 `--cf-sitl-params` 会直接 `rc=2` 报未知参数退出。
  2. 若以 Python 调用方式构造含 `cf_sitl_params` 的 `args`（或后续只补 `argparse` 漏补 `collect_command`），`scene_gen.py` 会把世界 SDF 的 `<params>` 改成满电副本（插件按 100% 满电运算），而 `collect.py` 未收到该开关，按 `cf_sitl_exec.DEFAULT_PARAMS`（`collect.py:1585-1586`）在 `episode_manifest.json` 写入 50% 半电指纹 `cf_sitl_v1:1bc8fedf:lib31`。`tools/preprocess.py` 只核对 `episode_manifest.json` 的 `dynamics_profile` 而不核对 `scene.json` 的 `cf_sitl_params`，满电物理轨迹会与半电数据静默混包。
  3. 分支 `p12-w1-13`（合入 `e40c693` 的提交 `90ed8d7`）只给 `collect_parallel.py` 透传了 `--action-latency` 与 `--rtf`，未修此项。
- 复现与验证：运行 `verify_batch1.py` 第 6 节：`cp.scene_gen_extra(ns)` 输出 `['--cf-sitl-params', '/tmp/e1_soc100.json']`，而 `"--cf-sitl-params" in cp.collect_command(task, ns)` 为 `False`；命令行跑 `collect_parallel.py --cf-sitl-params ...` 报 `unrecognized arguments`。
- 建议的最小修法：`collect_parallel.py` 的 `main()` 增加 `--cf-sitl-params` 参数及与 `eval_closed_loop.py` 相同的校验与 `os.path.abspath` 转换；`collect_command` 在 `getattr(args, "cf_sitl_params", None)` 非空时追加 `["--cf-sitl-params", args.cf_sitl_params]`；`collection_manifest.json` 在非空时记录 `cf_sitl_params`。
- 状态：已复现。
- 归属建议：并入卡 W1-13 第三部分或 W1-15b 后续（两条 Gazebo 流水线若需满电档也在此处衔接）。

### `I1-3`：`collect.py`（及 `collect_parallel.py`）缺少学生条件一致性闸门，且三处 Gazebo 入口均未拦截 waypoint 任务上的 `--student-cond heading-code`

- 位置：`sim/tools/collect.py:966-969`、`:1532-1533`、`:1778-1783`、`:2034-2035`；`sim/tools/collect_parallel.py:707-708`；`sim/tools/eval_closed_loop.py:1325-1326`（合并提交 `d816d2c`，卡 W1-9 提交 `ba9b630`）。
- 现象：
  1. W1-9 只在 `eval_closed_loop.py:1415` 调用了 `nav_cond.check_student_cond`，未在 `collect.py` 调用。用 `collect.py` / `collect_parallel.py --policy student` 做 DAgger 自采时：
     - 若模型 `prev_dim == 0`（纯视觉）却传了 `--student-cond heading-code`：`student.py:121`（`if self.prev_dim:`）静默丢弃 `cond_in`，模型实际未接收航向码，而 `collect.py` 照常在 `actions.jsonl` 每行写 `cond_code`、在 `episode_manifest.json` 写 `"student_cond": "heading-code"`；
     - 若模型是 `heading-code` 训出的（`prev_dim == 4`）却漏传 `--student-cond heading-code`（或 `dagger_gz_pipeline.py` 的 `round0_ckpt` 与 `CFG["student_cond"]` 条件类型不一致）：采集全程不报错，将上一步动作 `0/1/2/3`（含航向码未使用的码 `1`）当成航向码喂给模型。
  2. `tools/preprocess.py:1257-1261` 对含 `waypoint` 的集开 `--cond heading-code` 直接报错退出（航向设定值 `H` 在 waypoint 任务未定义），但 `collect.py`、`collect_parallel.py`、`eval_closed_loop.py` 均允许 `--task l1/campus/campus_assets`（或含 `waypoint` 的 `scene.json`）配 `--student-cond heading-code`。
- 为什么是缺陷及影响：DAgger 自采（卡 W2-2）若配错 `round0_ckpt` 或漏传开关，会跑完数小时采集与训练才在末尾评测报错或产出污染轨迹；在 E5（`l1` / `campus`）误开 `--student-cond heading-code` 时，评测闸门对柱阵航向码模型返回 `"ok"` 并全程按 `H=0` 喂错条件，采集则会在事后 `preprocess.py` 打包时整批作废。另注：`ada0ca5` 上 `dagger_gz_pipeline.py` 合并了 W1-5（默认 `deliver="calswa"`）与 W1-9（`heading-code` 加 `--prev-action`），而 `main` 的 `tools/average_ckpt.py:210` 拒绝 `prev_dim != 0`；未合并分支 `p12-e2e-1` 的 `dc8e5ff` 已在修 `average_ckpt.py` 校准部分，但未修本条的采集闸门与 waypoint 拦截。
- 复现与验证：`verify_batch2.py` 第 C.3 节验证 `collect_parallel.py --policy student --student-cond heading-code --task l1 --spawn-mode random` 通过全部业务参数校验（直到模板存在性检查才停）。
- 建议的最小修法：`collect.py` 构造 `StudentPolicy` 后调用 `nav_cond.check_student_cond(args.student_cond, args.student_ckpt, student.prev_dim)`，`status == "mismatch"` 时报错退出；`collect.py`（当 `waypoint is not None`）、`collect_parallel.py` 与 `eval_closed_loop.py`（当 `args.task in ("l1", "campus", "campus_assets")`）拒绝 `--student-cond heading-code`。
- 状态：已复现（waypoint 参数放行已运行复现；采集无闸门读代码确认）。
- 归属建议：并入卡 E2E-C1 或 W2-2 开工前修补。

### `I1-4`：教师 v7 的 `--action-latency` 栅格预热未覆盖 B3 重试余量，长课程首次触发 B3 单步耗时 456–1093 ms，突破 `L=0.20 s` 预算造成迟到废集

- 位置：`sim/tools/collect.py:1795-1801`；`sim/tools/teacher.py:387-403`、`:502-516`（合并提交 `f383584`，卡 W1-7 提交 `009bca2`）。
- 现象：`collect.py:1800` 在开启 `--action-latency` 时调用 `_teacher.warm_free_cells(pillars)`，仅预热默认 `ASTAR_CLEARANCE_M`（0.40 m）一档缓存，未预热 `ASTAR_RETRY_CLEARANCES`（0.30 m、0.20 m）。长课程中首次遇到 0.40 m 与 0.30 m 均无路的步时，`_astar_retry` 在观测到派发之间同步冷算两张完整长课程栅格。实测（`verify_batch2.py` 第 B 节，`nice -n 15`）：`course_base --blocks 3`（12,740 格）冷触发 B3 单步耗时 718.5 ms（`legacy`）/ 455.7 ms（`visible`），`--blocks 6`（22,490 格）耗时 1093.1 ms / 1000.5 ms；而将 3 个余量档位全部预热后，同一步耗时降至 69.9 ms（`blocks=3`）与 116.6 ms（`blocks=6`）。
- 为什么是缺陷及影响：在 `--action-latency 0.20` 下，策略从观测位姿戳到派发的允许预算为 `0.20 + 0.02 = 0.22 s`（仿真秒），教师策略与 `beta > 0` 的 DAgger 自采都在派发前同步算 A\*（`defer_label` 仅对 `policy == "student" and beta == 0.0` 生效，`collect.py:870`）。W1-7 验收只跑了单块 `K=1` 且未带 `--action-latency`，未暴露此耗时。后续跑带 `--action-latency 0.20` 的长课程教师对照或 `beta > 0` 长课程 DAgger 时，凡靠近柱子首次触发 B3 的集都会产生 `late_dispatch=True` 被判为基础设施失败，造成困难集的选择性丢弃或反复重跑。分支 `p12-w1-13` 未修此项。
- 复现与验证：运行 `verify_batch2.py` 第 B 节。
- 建议的最小修法：`warm_free_cells` 新增 `include_retries=True`（或在 `collect.py:1800` 循环 `(ASTAR_CLEARANCE_M,) + tuple(c for c in ASTAR_RETRY_CLEARANCES if c < ASTAR_CLEARANCE_M)`），在起飞前一次填满至多 3 张栅格（`_FREE_CACHE_MAX_SCENES = 4` 容量足够，不改变任何路径数值）。
- 状态：已复现。
- 归属建议：并入卡 W1-13 第三部分（重标定要跑带 `--action-latency` 的长课程教师标定臂 `cal-*-k3`！如果触发 B3 会直接迟到不过门）。

### `I1-5`：`eval_closed_loop.py` 的 `collisions_per_100m` 分子计整集碰撞、分母在首次越界或进目标圆处截断，越界后碰撞使指标虚高（已有真实臂路程分母少算 19%–31%）

- 位置：`sim/tools/eval_closed_loop.py:477-501`（`flight_path_m`）、`:590`（`summarize_episode`）、`:733-743`（`collisions_per_100m`）（合并提交 `dd98f89`，卡 W1-4 提交 `f9bfbfa`）。
- 现象：
  1. `eval_closed_loop.py` 启动 `collect.py` 时带 `--stop-on-collision` 与 `--stop-at-goal`，但横向越出走廊（`|y| > corridor_half_width`）或 waypoint 任务首次进入 1.5 m 到达圆后并不会停止采集。
  2. `flight_path_m` 在首次 `outside`（第 494–496 行）或首次进入目标区（第 498 行 `goal_path_length_m`）处截断分母路程，而 `collisions_per_100m` 的分子 `sum(1 for e in episodes if e.get("collision"))` 统计整集所有步的接触碰撞。
  3. 扫描现有评测目录，在 3 个真实集中检出「先越界、继续飞行数十步后发生接触碰撞（含长课程越界后飞到第 137 步电池耗尽落地接触）」：
     - `exp-long-course-20261110/calswa-s0` `seed 555`：第 23 步越界（分母截在 3.896 m），第 137 步碰撞（到碰撞处实际路程 30.484 m），分子记 1 次碰撞，分母少算 26.588 m（占 W1-4 报告表 4.1 该臂总路程 85.584 m 的 31.1%）；
     - `exp-long-course-20261110/guard2-calswa-s2` `seed 992`：第 98 步越界（17.441 m），第 137 步碰撞（57.170 m），分母少算 39.729 m（占该臂总路程 204.503 m 的 19.4%）；
     - `e1-pipeline-20261107-0417-eval4/eval-student-e1-new` `seed 929`：第 22 步越界（4.135 m），第 64 步撞柱（13.872 m）。
  4. 另有单步即撞边界：`oracle.jsonl` 不含起飞初始点，若全部完成集都在第 0 步碰撞终止（`len(oracle) == 1`），`flight_path_m` 返回 `0.0`，`collisions_per_100m` 因 `total <= 0.0` 返回 `None`。
- 为什么是缺陷及影响：W1-4 报告中的离线数字及今后所有开启 `--corridor-half-width 3` 的 `eval.json` 均受影响；易越界的无护栏学生臂在走廊外耗尽电量落地的接触被算进走廊内碰撞分子，放大了与有护栏臂的 `collisions_per_100m` 差距。
- 复现与验证：运行 `verify_batch1.py` 第 4 节。
- 建议的最小修法：对齐分子与分母的事件窗口。若定义为「有效走廊/目标前路程的碰撞密度」，分子改为只在首次越界与首次进目标区之前（含当步）发生的接触碰撞；若分子保持与 `collision_rate` 相同（计整集碰撞），则 `flight_path_m` 不在 `outside` 处截断。单步集可用 `(actions[0]["pose_before"], oracle[0]["pose_gt"])` 计算第 0 步折线长。
- 状态：已复现。
- 归属建议：新立统计口径修补小卡（并在 `docs/EVAL_STANDARD.md` 写明定义）。

### `I1-6`：`nav_cond.check_student_cond` 存在三处核验盲区：忽略 `config.resume.json`、不核对 ckpt 内嵌契约哈希、显式 `--student-cond prev-action` 对纯视觉模型静默放行

- 位置：`sim/tools/nav_cond.py:97-155`；`sim/tools/eval_closed_loop.py:1235`、`:1415`（合并提交 `d816d2c`，卡 W1-9 提交 `ba9b630`）。
- 现象：
  1. `find_train_config`（`nav_cond.py:97-108`）只查找 `config.json`。而 C++ 训练器在同目录 `--resume` 续训时不覆盖 `config.json`，把最新训练数据集与 `dataset_sha256` 写入 `config.resume.json`（`src/class_weights.cpp:126, 286-290`）。同目录续训切换数据集后，闸门读旧 `config.json`，会把合法模型误拒为 `mismatch` 或把不匹配模型放行（`e40c693` 合入的 `eval_closed_loop.train_dataset_meta_for_ckpt` 复用了 `find_train_config`，同样受影响）。
  2. `check_student_cond` 不读取 `ckpt` 二进制文件头内嵌的 32 字节 `contract.dataset_sha256`（`src/checkpoint.cpp:166`，`tools/average_ckpt.py:60, 216` 已实现 Python 解析）。若将另一个运行产出的 `ckpt.bin` 拷入已有 `config.json` 的目录评测，闸门只要看到 `config.json` 与其所指 `.uavr` 自洽即返回 `"ok"`，不会发现 `ckpt.bin` 并非该配置训出。此外，`<ckpt>.json` 侧车本身记有绝对路径 `"data"` 与 `"data_sha256"`（`average_ckpt.py:229`），`find_train_config` 未直接读取，当 `config.json` 中 `"data"` 为相对路径且评测在不同工作目录运行时会退化为 `"unverified"` 警告放行。
  3. `eval_closed_loop.py:1235` 将 `--student-cond` 设为 `default="prev-action"`（而非 `default=None`）：用户显式传 `--student-cond prev-action` 但误传 `prev_dim=0` 的纯视觉权重时，`check_student_cond` 返回 `status="ok"`，无报错且不写 `eval.json`。
- 为什么是缺陷及影响：在卡 E2E-C1 给 checkpoint 增加 `cond_kind` 字段之前，`check_student_cond` 是阻止条件喂错的唯一自动防线；上述三处盲区在续训、拷贝权重或混用纯视觉与 `prev-action` 权重时会失效或误判。
- 复现与验证：运行 `verify_batch2.py` 第 C.1、C.2 节（同目录 `config.resume.json` 为 `heading-code`、原 `config.json` 为 `prev-action` 时，请求 `heading-code` 实测返回 `mismatch`）。
- 建议的最小修法：`find_train_config` 在每个候选目录先查 `config.resume.json` 再查 `config.json`，并支持从 `<ckpt>.json` 直接取 `data`/`data_sha256`；解析 `ckpt` 头部的 `contract.dataset_sha256` 与配置的 `dataset_sha256` 比对；`--student-cond` 改 `default=None`，显式传入 `"prev-action"` 而 `prev_dim == 0` 时返回 `"mismatch"`。
- 状态：已复现（第 1、3 点运行复现；第 2 点读代码确认）。
- 归属建议：并入卡 E2E-C1。

### `I1-7`：`--cf-sitl-params` 在 `eval_closed_loop.py` 与 `stats_report.py` 之间的两处协议边界错配及坏参数文件晚崩溃

- 位置：`sim/tools/eval_closed_loop.py:83-84`、`:1338-1343`、`:1481-1495`、`:1577-1579`（合并提交 `41c474d`，卡 W1-10 提交 `b436c90`）；交互位置 `tools/stats_report.py:72`、`:259-263`（合并提交 `0d7f3b5`，卡 W1-14 提交 `087d2a8`）。
- 现象：
  1. 满电评测臂若因迟到步或超时导致完成集为空（`results == []`），`dynamics_profile_of(results)` 返回 `None`，`eval.json` 顶层写 `"dynamics_profile": null`。`stats_report.py` 将 `None` 当作 `NO_VALUE`（旧默认 50% 档），配合 W1-14「失败集记 0 参与主表与配对」的新规则，100% 满电全败臂与 50% 半电臂同表汇总时不会触发 `ProtocolMismatch` 报错。
  2. 不传 `--cf-sitl-params` 时 `eval.json` 顶层无 `dynamics_profile`（`NO_VALUE`），显式传默认参数 `--cf-sitl-params sim/dynamics/e1_mini.json` 时顶层写 `"cf_sitl_v1:1bc8fedf:lib31"`，物理参数相同却被 `stats_report.py` 判为 `ProtocolMismatch`（`rc=2`）。
  3. `eval_closed_loop.py:1338-1343` 启动时只检查 `os.path.isfile`，不检查 `CF_SITL_PARAMS_BAD_CHARS` 也不调 `CfSitlParams.from_json`；若传入损坏或 schema 不符的 JSON，会等跑完/失败全部请求集后，在收尾 `write_report` 第 1577 行 `cf_sitl_params_record` 抛未捕获异常，无法写出 `eval.json`。此外 `PROTOCOL_KEYS` 未纳入 `dynamics_profile`，`--protocol` 快照不检查电量档变更。
- 为什么是缺陷及影响：影响含失败臂的多档动力学报表防混门禁（A4）与长评测的快速失败保护。
- 复现与验证：见草稿核验输出（`stats_report.build_report` 对默认无开关与显式 `e1_mini.json` 抛 `ProtocolMismatch`，对默认 50% 与 `dynamics_profile=None` 的满电全败臂静默通过）。
- 建议的最小修法：`eval_closed_loop.py` 参数校验阶段即调用 `cf_sitl_exec.CfSitlParams.from_json(args.cf_sitl_params)` 并校验路径字符；`dynamics_profile_of` 在 `results` 为空时用该参数对象的 `values_sha256[:8]` 构造前缀标识（不为 `None`）；`stats_report.load_eval` 在顶层无 `dynamics_profile` 时回退读取 `per_episode` 的 `dynamics_profile`（或把默认 `1bc8fedf` 归一处理）。
- 状态：已复现（第 1、2 点运行复现；第 3 点读代码确认）。
- 归属建议：并入 W1-15b 后续或新立小卡。

### `I1-8`：教师 v7 的 B2（过柱阵后目标取线）在 `tiebreak="visible"` 下让左右滚出在最远柱后表面边界处指向不同目标，造成 6.6 步虚假代价差

- 位置：`sim/tools/teacher.py:519-532`、`:624-639`（合并提交 `f383584`，卡 W1-7 提交 `009bca2`）。
- 现象：`_q_after_yaw` 对左/右转向各滚出一步 `(nx, ny)`（平移 `TIE_YAW_DRIFT_M = 0.11 m`）后调用 `_global_path(nx, ny, ...)`，而 v7 在 `_global_path` 内按 `nx > _pillars_back_x(pillars)` 决定目标是点 `(goal_x, 0.0)` 还是 `(goal_x, ny)`。当机体在最远柱后表面前 `0 ~ 0.11 m` 且 `|y| > 0` 时，一侧滚出越过 `back_x`（目标跳为 `(goal_x, ny)`），另一侧未越过（目标仍为 `(goal_x, 0.0)`）。在单门柱 `gate`（`back_x = 1.70`）的 `(1.65, 1.5, yaw=50°)` 处实测：`Q_L = 14.406`、`Q_R = 7.833`，`|Q_L - Q_R| = 6.573` 步（对比 `plan_costs` 的 `_costmap_goal` 按当前步 `(x, y)` 固定一次目标后四动作共用）。
- 为什么是缺陷及影响：破坏了 `_global_path` 文档注释所写的「保证 `Q_L`/`Q_R` 与教师看的是同一张图、同一个目标」，使 `--tiebreak visible` 在柱阵出口边界前 0.11 m 带内的平局判定失效。默认 `tiebreak="legacy"` 不调用 `_q_after_yaw`，已有实验不受影响；后续若用 v7 加 `--tiebreak visible` 采数会受影响。
- 复现与验证：运行 `verify_batch1.py` 第 3 节。
- 建议的最小修法：`_global_path` 增加可选参数 `ref_xy=None`（默认取 `(x, y)`），`_q_after_yaw` 传入当前步的 `ref_xy=(x, y)`，使主动作与 `Q_L`、`Q_R` 共用同一目标（改标签须随教师下一版本升版）。
- 状态：已复现。
- 归属建议：与 `I1-1` 一并纳入下一版教师（v8）修订。

### `I1-9`：W1-3 的 `frame_gap_flag` 以 `1.2 × hold`（1.20 s）而非名义步周期（0.90 s）为基准，漏标 1.10–1.20 s 停顿集，并与 W1-13 的 `stall` 判据自相矛盾

- 位置：`sim/tools/eval_closed_loop.py:505-526`、`:682-688`（合并提交 `bbc6132`，卡 W1-3 提交 `309fcc4`）。
- 现象：Gazebo 10 Hz 相机在 `hold = 1.0 s` 下的名义相邻 `rgb_ts` 间隔恒为 `0.900 s`（`hold - 0.1 s`）。`FRAME_GAP_FLAG_FACTOR = 1.2` 按 `gap > 1.2 * hold`（严格大于 `1.200 s`）打旗，导致停顿 0.20–0.30 s（`gap` 为 1.10–1.20 s）的集全部不打旗。在真实实验目录中，`exp-action-latency-20261110/acc-L0.16-k1-s1` 的 4 集（`seed 113, 122, 153, 176`，`max_frame_gap_s = 1.2 s`）、`exp-action-latency-r2-20261111/cal-L0.20rtf0.5-k3` `seed 8`（`1.2 s`）以及 `e1-pipeline-20261107-0417-eval4/eval-student-e1-new` `seed 992`（`1.1 s`）均未被打旗；反之在 `hold = 0.2 s` 档下，`1.2 * 0.2 = 0.24 s` 会将 10 Hz 相机正常的 `0.30 s` 间隔误标打旗。同时，卡 W1-13（合入 `e40c693` 的提交 `51ba557`、`4d8a4ff`）对同一物理量定义了 `is_stall_gap`（`> max(0.1, hold - 0.1) + 0.2 + 1e-6`，`hold=1.0` 时 `> 1.100001 s`），导致合并后同一集在 `1.15 s` 或 `1.20 s` 时既被标为 `stall_steps > 0`（停顿），又没有 `frame_gap_flag`（未断帧）。
- 为什么是缺陷及影响：在旧协议（不带 `--action-latency`，`stall_steps` 不进 `failed[]`）评测中，靠 `frame_gap_flag` 排查断帧会漏掉 0.20–0.30 s 的真实停顿；新旧两套指标出现在同一份 `per_episode` 中口径矛盾。
- 复现与验证：运行 `verify_batch1.py` 第 5 节。
- 建议的最小修法：在 `eval_closed_loop.py` 中对 Gazebo 集统一调用 `collect.is_stall_gap(gap, hold)` 决定 `frame_gap_flag`（无 `hold` 而走 `sim_dt` 后备的 3DGS 集无相机相位差，保持原判断或单列）。
- 状态：已复现。
- 归属建议：并入卡 W1-13 第三部分。

### `I1-10`：`sim/tools/make_counterfactual.py` 未调用 `teacher.set_vehicle` 切换载具与版本，在教师默认升 v7 后会产生版本伪造与余量错配

- 位置：`sim/tools/make_counterfactual.py:60-76`、`:86-103`、`:128-137`（受合并提交 `f383584` 教师升 v7 影响，卡 W1-7 未改该文件）。
- 现象：`make_counterfactual.py` 导入 `teacher` 后未调 `teacher.set_vehicle(...)`，固定使用导入时的全局状态（载具 `x3`、余量 `0.50/0.36`、v7 规则 `ASTAR_RETRY_CLEARANCES = (0.30, 0.20)`），并在输出 `episode_manifest.json` 中原样复制源集的 `teacher_version` 与 `vehicle`。
- 为什么是缺陷及影响：对旧 `v5` 的 L1 数据运行会写出按 `v7`（含 B3）计算的标签却标 `v5`；在计划 §10.4 的 E5 阶段对 `e1_mini` 新采的 L1 数据运行，会用 `x3` 的 `0.50 m` 余量重算 `e1_mini`（应为 `0.40 m`）的标签，送入 `audit_dataset.py` 的 `label_recompute` 会因不一致失败。（同类遗留：`experiments/2026-11-10-action-latency/profile_late_steps.py:377` 调 `set_vehicle("e1_mini", task)` 未传 `version=man.get("teacher_version")`，重跑 11-10 的 v5 数据会改按 v7 复算。）
- 复现与验证：读代码确认（对照 `make_counterfactual.py:69` 与 `audit_dataset.py:936`）。
- 建议的最小修法：在 `make_one` 开头调用 `teacher.set_vehicle(man.get("vehicle") or "x3", teacher.task_of_scene(scene), version=man.get("teacher_version"))`。
- 状态：读代码确认。
- 归属建议：并入 E5（语言导航）开工前的数据准备卡。

---

## 3. 第一轮修复清单（`FIXLIST.md`）相关条目核对

核对基准为 `ada0ca5`，括号内附注审核期间 `e40c693`（W1-13 第一、二部分）与 `035b398`（W1-15b）的新进展：

- **修对了的**：
  - `F8`（`eval_closed_loop.py` 闭环判据的走廊半宽与过线边界锁测试）：卡 T1 第一批（合并 `ceac416`）新增 `sim/tools/test_eval_criteria_boundary.py`，已锁住 `|y| == 3.0` 与 `x == goal_x` 的包含边界。
- **修了一半的**：
  - `A3`（迟到判据看不到步间停顿；`max_frame_gap_s`）：`ada0ca5` 上仅合入卡 W1-3（`bbc6132`），只记不判，且 `frame_gap_flag` 用 `1.2 * hold` 漏标了 1.10–1.20 s 的停顿（见 `I1-9`）。（注：`e40c693` 的 `51ba557`、`4d8a4ff` 已补齐 `obs_gap_s`、`stall_steps` 及 `--action-latency` 下的失败判定，但未统一 `frame_gap_flag` 阈值。）
  - `A4`（不同协议的 `eval.json` 混进 `stats_report` 不报错）：卡 W1-14（合并 `0d7f3b5`，提交 `087d2a8`）对 `action_latency_s`、`heading_guard`、`rtf`、`student_cond` 四个键拦截正确；但 `dynamics_profile` 与 W1-10（`41c474d`）在 `eval_closed_loop.py` 的写入条件配合时，存在满电全败臂 `dynamics_profile=null` 混入 50% 臂不报错、以及显式传默认 `e1_mini.json` 对不传开关误报错两处边界漏洞（见 `I1-7`）。
- **在 `ada0ca5` 上没修的（符合 `FIXLIST.md` 当时的在途排期）**：
  - `A1`（步间 SAFE 空档不受控且不落盘）：`ada0ca5` 未修。（注：`e40c693` 的 `e9822de`、`5460af6` 已完成 `safe_gap_s` 落盘与汇总；「动作持续到下一次派发并重标定」仍在 W1-13 第三部分待做。）
  - `A2`（训练旧协议评测新协议不提示、`collect_parallel` 与流水线不透传 `--action-latency`、`preprocess` 不过滤迟到步）：`ada0ca5` 未修。（注：`e40c693` 的 `46cd88c`、`90ed8d7`、`aa9a6ff` 与 `035b398` 的 `402096a` 已补齐四处；其中 `46cd88c` 的 `train_dataset_meta_for_ckpt` 复用 `nav_cond.find_train_config`，同样受 `I1-6` 的 `config.resume.json` 盲区影响，且读不到配置时静默不写 `protocol_notes`。）
  - `A5`（`dispatch_lag_s` 用到达 Python 的位姿戳而非插件收到动作的仿真时刻）：`ada0ca5` 未修。（注：`e40c693` 的 `579901d`、`f4a2315` 已增加 `plugin_ack_sim_t` 落盘与测量入口，按计划只记录、未切换迟到判据。）
  - `A6`（`rule_v1` 与 `heading_guard` 的 26° 常量按旧协议标定）：按 `FIXLIST.md` 不单独修（K3 新增的 `RULE_V2` 默认 `yaw_step_deg` 同样为 26.0°）。
  - `C2` / `G1` 后半（`sim/tools/student.py:122` 在 `prev_dim > 0` 且调用方不传 `prev_action` 时回落 `_last_action` 不报错）及 `C3`、`C4`、`G7`（checkpoint 与握手行无 `cond_kind`、条件维度写死 4）：`ada0ca5` 未修，归待建卡 `E2E-C1`。

---

## 4. 查过但没有发现问题的地方

- **四次合并冲突解决**：逐个核对了 `git show --cc`（`d816d2c` 的 `collect.py` 与 `dagger_gz_pipeline.py`、`f383584` 的 `preprocess.py`、`e41703e` 的 `preprocess.py`、`dfb98c4` 的 `driver.py`）。除 `I1-3` 已指出的 `dagger_gz_pipeline.py` 在 `main` 上受制于 `average_ckpt.py` 尚未合入 `dc8e5ff` 外，四次手动解冲突均完整保留了两侧代码逻辑与参数，没有丢行或顺序错位。
- **「缺省逐位不变」成立的部分**：
  - `collect.py`（W1-3 `probs`、W1-9 `--student-cond`）：教师、盲前进、规则层策略下 `probs` 与 `cond_code` 均为 `None`，`action_row` 不多键，默认 `episode_manifest.json` 不多键；学生策略下仅按卡要求新增 `probs`。
  - `eval_closed_loop.py`（W1-4、W1-3、W1-9、W1-10）：`base` 任务不出现 `blocks_passed`、`blocks_total`、`block_survival`；不传 `--student-cond`（且 `prev_dim=0` 或闸门返回 `ok`）与不传 `--cf-sitl-params` 时顶层不新增对应键；`aggregate()` 未改动，不影响共用它的 `eval_3dgs.py` 与 `compare_datasets.py`。
  - `scene_gen.py`（W1-10）：不传 `--cf-sitl-params` 时世界文本、meta 键集合与 `world_sha256` 逐位不变；传开关时正则 `CF_SITL_PARAMS_RE` 锚定在 `<plugin filename="CfSitlSystem"...>...</plugin>` 内部，不误改模板注释中的 `<params>` 字样，且 `campus`、`campus_assets`、`forest_e1`、`course_base`、`l1`、`base` 六种任务在 E1 模板下都保留该插件块，匹配次数均为 1。
  - `teacher.py`（W1-7）：标准余量下有路且未越过 `_pillars_back_x(pillars)` 时，`_astar_retry` 只调一次 `astar_path`，路径与 v5 逐位一致；每个余量档位在 `_free_cells` 中有独立缓存键 `(clearance, step, pillars_tuple)`，不会串档；`audit_dataset.py` 按 `manifest.teacher_version` 分派 `v5`、`v6`、`v7` 与 `cost v1`、`cost v2` 的逻辑清晰，`forest_e1` 档保持 `v6`（B2、B3 关闭）未变。
- **航向码 `nav_cond.py` 与 `--heading-guard` / `--beta` 组合**：`heading_code` 与 `heading_codes` 取本步执行前的净转向步数 `n`，码值集合 `{0, 2, 3}` 与训练器 `flip_side_inputs` 的 `2 ↔ 3` 镜像互换兼容；`collect.py` 主循环按 `action_executed`（含 `beta` 专家接管与 `--heading-guard` 改写后的实际执行动作）更新 `net_yaw`，与 `preprocess.py --cond heading-code`（含保留被 `--drop-collision` 丢弃步的转向计数）口径一致。
- **W1-4 的 `block_outcomes` / `blocks_passed`**：同一步既碰撞又越界按碰撞记，过线当步有碰撞或越界不算通过该块，`stuck` 不算接触碰撞，参数顺序反转 `(oracle, actions)` 在调用处 `run_episode:991` 传参正确。
- **K3 在 `sim/tools/rule_policy.py` 与 `rule_sim.py` 的改动**：`rule_policy.py` 前 218 行（`v0`、`v1`）零改动；`rule_v2_step` 对 `memory` 与 `tabu` 均返回新列表，无跨集状态残留；`check_v2_params` 保证了 `levels` 逐级不放宽，使禁忌表的 `lb` 单调性成立；`rule_sim.py` 传入 `tabu` 与 `prev_action` 的顺序正确。

---

## 5. 没有查的地方与原因

- 没有启动 Gazebo、没有跑真实学生或教师闭环、没有跑 GPU 推理或训练、没有运行 `tools/check.sh`：严格遵守任务书第 3 节只读与零抢核约束（机器 12 核配额已满，Gazebo 池与训练池有实验在跑）。因此 `I1-1` 在真实 Gazebo 闭环采集中的触发频率基于 `0417` 真实轨迹状态的确定性重演，新采轨迹上的闭环碰撞率变化未核实。
- `sim/cf_sitl/` C/C++ 源码内部实现（除为核对 `--cf-sitl-params` 路径解析只读查看了 `CfSitlSystem.cc:27-84` 外）：归角度 K 审核。
- `tools/preprocess.py` 的 `--depth-source analytic`（W1-8）内部几何求交、`tools/average_ckpt.py`、`tools/stats_report.py` 的聚类统计实现细节：归角度 I2 审核（本文仅查了它们与 `sim/tools` 交界处的契约）。
- `sim/3dgs/` 下的 `rule_oracle.py`、`planner.py`、`collect_3dgs.py`、`eval_3dgs.py`：归角度 J 审核。
