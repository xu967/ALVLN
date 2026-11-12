# 卡 PK1：打包的健壮性（I2-4、I2-6、I2-8 第 2 点、I2-9 第 2 点，第一轮 C7）

- 时间：2026-11-12 08:21
- 范围：`tools/preprocess.py`、`tools/DATASET_FORMAT.md`、`tools/test_preprocess_*.py`
- 结论：完成

## 做了什么

- 修复 `I2-4a`（提交 `e4ff163`）：在 `tools/preprocess.py` 新增 `_read_goal_pillar_xs` 与 `_read_oracle_max_x`，区分「文件缺失或无柱子 / 无 `pose_gt.x`」（仍记为不适用）与「文件存在但 JSON 损坏或字段类型非法」（报错退出并指明文件路径与行号）；在 `tools/test_preprocess_failed_teacher.py` 新增对应回归测试。
- 修复 `I2-4b`（提交 `d559654`）：在 `tools/preprocess.py` 对 `scene.json` 的 `pillars[*].x` 与 `oracle.jsonl` 的 `pose_gt.x` 校验 `math.isfinite`，拒绝 `NaN` 或无穷大，防止首行 `NaN` 污染 `max_x` 误跳已过线的正常集；在 `tools/test_preprocess_failed_teacher.py` 新增测试。
- 修复 `I2-4c`（提交 `6691baa`）：在 `tools/preprocess.py` 新增 `_read_actions_jsonl` 并将 `len(rows) != manifest.num_steps` 的截断检查移到 `exclude_collision_episodes` 与 `exclude_failed_teacher` 之前，报错指明文件与行号；在 `tools/test_preprocess_failed_teacher.py` 新增测试。
- 修复 `I2-4d`（提交 `3b06867`）：在 `tools/preprocess.py` 新增 `_check_nonempty_splits` 与显式放行开关 `--allow-empty-split`，当输入中存在 `train` 或 `val` 集目录但被整集或行级过滤清空时默认报错退出并指明示例文件路径与行号；在 `tools/test_preprocess_failed_teacher.py` 新增测试。
- 修复 `I2-6a`（提交 `243f355`）：在 `tools/preprocess.py` 的 `scan()` 中将行级过滤（`--drop-collision`、`--drop-late-and-stall`）整集淘空后剩余 0 步的 episode 跳过，不写入 `meta.episodes`、`meta.scenes`、`geometry_versions` 与 `instructions.jsonl`，单独计数并打印 `preprocess: 跳过整集 N 个（行级过滤后剩余 0 步）`；新增 `tools/test_preprocess_empty_episode.py`。
- 修复 `I2-6b`（提交 `aa5b95e`）：在 `tools/preprocess.py` 新增 `_require_action_int` 与 `_cond_by_step`，逐行直接校验自身 `action_executed` 为 `0..3` 的整数并删除异常吞没，源 `actions.jsonl` 自身步号存在缺口时统一将 `prev-action` 与 `heading-code` 在缺口处断链重置为 `0`；同步更新 `tools/DATASET_FORMAT.md` §6.2–§6.3 与 `tools/test_preprocess_cond.py`。
- 修复 `I2-8` 第 2 点（提交 `070f488`）：在 `tools/preprocess.py` 的 `_pillar_hit_t` 圆柱分支补上顶盖与底盖圆盘面求交，与方柱三维 slab 求交一致，相机处于柱底与柱顶之间（现有全部数据）时输出逐字节不变；同步更新 `tools/DATASET_FORMAT.md` 与 `tools/test_preprocess_analytic_depth.py`。
- 补充 `I2-9` 第 2 点测试（提交 `8173851`）：在 `tools/test_preprocess_split.py` 新增直接调用 `preprocess.build_meta` 拦截跨 split 同 `geometry_gid` 的单元测试，并用草稿副本变异验证先红后绿。
- 完成第一轮 `C7` 四个数据集的实盘测量与方案建议，记录于报告 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/pk1.md`（不改打包行为）。

## 没做什么

- 没有修改打包期行级过滤丢步后的 `C7` 条件列取值行为（`prev_index[i] == i` 而 `prev_action` / `heading_code` 非 0）。
  - 原因：任务卡与交办第 4 条规定本卡对 `C7` 只量数与给建议，不改行为，改不改由主会话定。
- 没有修改 `tools/make_synthetic_episode.py`。
  - 原因：新测试用临时目录副本构造异常与几何夹具即可覆盖全部场景，无需给 `make_synthetic_episode.py` 增加新选项。
- 没有修改 `src/` 下 C++ 代码、`sim/` 下仿真代码、三条流水线（`tools/e1_pipeline.py`、`tools/dagger_gz_pipeline.py`、`tools/dagger_pipeline.py`）、`tools/make_params_variant.py`（`I2-8` 第 1 点）或全量 `tools/check.sh` 的 C++ 侧车自检（`I2-9` 第 1 点）。
  - 原因：交办明确规定只动 `tools/preprocess.py`、`tools/test_preprocess_*.py`、`tools/DATASET_FORMAT.md`，其余分属别的卡范围。
- 没有跑全量 `bash tools/check.sh`（含 C++/CUDA 编译与 GPU CTest），没有启动 Gazebo。
  - 原因：本卡不改 C++、不起 Gazebo、不用 GPU，按公共规则 §5 与交办第 6 条只跑 `tools/check.sh --python-only`。

## 待确认

- `C7` 处理方式由主会话拍板：四个数据集里 `prev_index[i] == i` 且条件非 `0` 的样本占比在 `0.021%`–`0.108%`（最多 `22/20291` 帧），且当前单帧主线完全不读 `prev_index`。建议当前单帧阶段保留现状（`DATASET_FORMAT.md` §6.3 与 `preprocess.py` 的 `shortcut` 注释已记录）；后续重启双帧加条件训练时，优先让侧车保留被丢帧像素供 `prev_index` 指向真实上一帧（或仅对 `--cond prev-action` 将断链帧条件置 `0`，`--cond heading-code` 绝不能置 `0`，否则破坏机体相对初始走廊的累积偏航角真值）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 审核复现脚本 `repro_i2.py`（`TEST 1` 与 `2a`–`2f`） | 改前全部复现属实；改后 `TEST 1` 14/14 对齐，`2a`–`2e` 均 `rc=1` 报出具体文件与行号，`2f` 方柱与圆柱同为 `0.854 m` |
| 7 组基线逐字节不变核对（`baseline_pack.py --verify`） | 通过（`TEST 1` 缺省/四开关全开、`0417` 真实 6 集缺省/四开关全开、W1-8 对拍集 `file`/`analytic`、森林圆柱四开关全开：固定 UTC 整文件 `dataset.uavr` / `dataset.aux` 与真实 CLI 各段 SHA256 全部逐位一致） |
| `tools/test_preprocess_*.py` 全部 7 个文件（含新增 `test_preprocess_empty_episode.py`） | 通过（`nice -n 10` 逐个串行运行，全部 `OK`） |
| `tools/test_e1_pipeline.py`、`tools/test_dagger_gz_pipeline.py`（带 `guard-bin` 防起 `gz`） | 通过（`test_e1_pipeline: PASS`，`test_dagger_gz_pipeline: Ran 36 tests OK`） |
| `tools/check.sh --python-only`（守护脚本排 `check.lock`） | 通过（`rc=0 2026-11-12 09:35:16 CST head=8173851`，`PASS（有 SKIP）`，日志 `$D/tmp/phase12-cards/pk1/check/check_python_only.log`） |

## 风险与遗留

- 变严项对下游的影响：若后续小规模冒烟命令在全部 `val` 集均未过线时开 `--exclude-failed-teacher`，或行级过滤将整个 `train` / `val` 清空，打包会报 `split=train/val ... 被过滤跳空` 退出；诊断用途需显式传 `--allow-empty-split`。两条正常流水线（`e1_pipeline.py`、`dagger_gz_pipeline.py`）在正常数据上不会触发报错。
- `tools/preprocess.py` 当前共 1,605 行，本次新增校验已拆为独立小函数（`_pillar_lineno`、`_read_goal_pillar_xs`、`_read_oracle_max_x`、`_read_actions_jsonl`、`_require_action_int`、`_cond_by_step`、`_fill_cond_segment`、`_check_nonempty_splits`），未重构其余既有代码。
