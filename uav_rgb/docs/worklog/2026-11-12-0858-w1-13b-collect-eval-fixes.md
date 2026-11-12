# 卡 W1-13 第二轮：采集与评测端的六处口径与防呆修正

- 时间：2026-11-12 08:58
- 范围：`sim/tools/{teacher,collect,collect_parallel,eval_closed_loop}.py`、对应测试文件与 `docs/EVAL_STANDARD.md`（分支 `p12-w1-13b`）
- 结论：完成

## 做了什么

- 修复 `I1-4`（`sim/tools/teacher.py`、`sim/tools/collect.py`、`sim/tools/test_astar_cache.py`、`sim/tools/test_action_latency.py`，提交 `aac03d7`）：带 `--action-latency` 时起跑前预热一并覆盖标准余量与小于它的 `ASTAR_RETRY_CLEARANCES` 两档重试余量；核实 `_FREE_CACHE_MAX_SCENES = 4` 单集最多占 3 槽无淘汰，K=3 长课程 5 种子实测首次触发重试的单步 CPU 中位从 `230.7–260.5 ms` 降至 `42.2–45.1 ms`，88 组状态动作数值逐点一致。
- 修复 `I1-2` 与 `K-2`（`sim/tools/collect_parallel.py`、`sim/tools/collect.py`、`sim/tools/test_scene_cf_sitl_params.py`，提交 `05ab71a`）：`collect_parallel.py` 增加 `--cf-sitl-params` 参数校验、向 `collect.py` 透传并记入 `collection_manifest.json`；`collect.py` 在 `--dynamics cf_sitl` 且有 `--scene-meta` 时按 `realpath` 核对自身与场景的参数文件路径，单边传参立即 `rc=2` 退出，同时验证 `eval_closed_loop.py` 原有行为不受误伤。
- 修复 `I1-3` 与 `I2-2` 第 2 点（`sim/tools/collect.py`、`sim/tools/collect_parallel.py`、`sim/tools/eval_closed_loop.py`、`sim/tools/test_collect.py`、`sim/tools/test_nav_cond_wiring.py`，提交 `f843d29`）：`collect.py` 构造学生后调用 `nav_cond.check_student_cond`，不匹配立即 `rc=2` 退出并关闭推理子进程；三处入口在带 `waypoint` 的任务上统一拒绝 `--student-cond heading-code`。
- 修复 `I1-7` 与 `I2-1` 第 2 点（`sim/tools/eval_closed_loop.py`、`sim/tools/test_scene_cf_sitl_params.py`，提交 `87f7928`）：`eval_closed_loop.py` 在参数校验阶段即检查路径特殊字符并解析 `--cf-sitl-params`（坏文件当场 `rc=2`）；完成集为空时由参数文件加 `CF_SITL_LIB_VERSION` 算出顶层 `dynamics_profile`，不再写 `null`。
- 修复 `I1-5`（`sim/tools/eval_closed_loop.py`、`sim/tools/test_eval_blocks.py`、`sim/tools/test_eval_frame_gap.py`，提交 `8c5b4d0`）：`collisions_per_100m` 分子与分母同取「从起点到第一个终局事件」窗口，越界或进目标区后才发生的接触不进分子，单步即撞集用第 0 步 `pose_before -> pose_gt` 折线长计路程，顶层写 `collisions_per_100m_def = "v2.first_terminal_event"`；离线重算 `exp-long-course-20261110`、`exp-long-course-v2-20261110`、`exp-action-latency-r2-20261111` 共 27 个臂。
- 修复 `I1-9`（`sim/tools/eval_closed_loop.py`、`sim/tools/test_eval_summary.py`、`sim/tools/test_eval_frame_gap.py`，提交 `af23cb8`）：Gazebo `frame_gap_flag` 与停顿步共用 `collect.is_stall_gap`，无 `hold` 的 3DGS 集保留原判断；全盘扫描 238 份 `eval.json` 列出全部 21 集旗翻转并核对审核者点名的 6 集。
- 更新 `docs/EVAL_STANDARD.md` §5.3 与 §5.5（提交 `36995b4`）。

## 没做什么

- 未实现卡 W1-13 第三轮（`--action-until next-dispatch` 加 `guard_s`、`ack_sim_t()` 边界修正、四锁标定与学生参照）
  - 原因：超出本轮交办范围，按计划等本轮合并及 `exp-w1-13-ack-20261112` 测量完成后在第三轮执行。
- 未修改教师规则与版本号（`I1-1`、`I1-8`、`I1-10`）
  - 原因：归卡 TCH1（教师 v8），本轮限定只扩展 `teacher.py` 预热范围且不改路径数值。
- 未修改 `sim/tools/nav_cond.py`（`I1-6`）、两条 Gazebo 流水线（`I2-2` 第 1、3 点，`I2-7`）、`tools/stats_report.py`（`I2-1` 第 1 点）与 `tools/preprocess.py`
  - 原因：分别归属卡 W1-15c、W1-15b、W1-14b、PK1，不在本轮允许改动文件白名单内。
- 未启动 Gazebo、未占 GPU、未跑全量 `tools/check.sh`（C++/CUDA 构建与 CTest）
  - 原因：本轮仅改动 Python 工具与文档，按规范使用单元测试、离线实验目录扫描与排 `check.lock` 的 `tools/check.sh --python-only` 验证。

## 待确认

- 统计端（卡 W1-14b 后续）：不传 `--cf-sitl-params` 且全部请求集失败（`per_episode == []`）时，`eval.json` 的顶层与 `failed[]` 均无 `dynamics_profile`，需定统计端对该空集边缘情形的处理策略。
- 历史长课程数字引用：`exp-long-course-v2-20261110` 中 `calswa-s2`（`6.8941 -> 3.7122`，`-46.15%`）与 `calswa-s1`（`8.0408 -> 5.5667`，`-30.77%`）在新定义下显著下降，后续报告引用时请标注 `v2.first_terminal_event` 口径。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `test_astar_cache.py`（14 项）、`test_action_latency.py`（27 项）与 `bench_i1_4.py` | 通过（先红后绿，`rc=0`） |
| `test_scene_cf_sitl_params.py`（24 项）、`test_collect_parallel.py`（8 项） | 通过（先红后绿，`rc=0`） |
| `test_nav_cond_wiring.py`（28 项）、`test_heading_guard.py`（10 项）、`test_collect.py`（130 项） | 通过（先红后绿，`rc=0`） |
| `test_eval_blocks.py`（37 项）、`test_eval_summary.py`、`test_eval_frame_gap.py`（5 项）、`test_step_gaps.py`（70 项） | 通过（先红后绿，`rc=0`） |
| 排 `check.lock` 执行 `bash tools/check.sh --python-only` | 通过（`rc=0 2026-11-12 09:59:31`，含 `sim/tools`、`tools`、`sim/3dgs` 与 `ui` 全部 Python 测试，日志 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-13b/logs/check-python-only/check.log`） |

## 风险与遗留

- 旧实验目录已落盘的 `eval.json` 为冻结记录未原地改写，其中 `collisions_per_100m` 无 `collisions_per_100m_def` 键、`frame_gap_flag` 为旧 `1.2 * hold` 阈值；跨批次对比这两项指标时需看 `collisions_per_100m_def` 或用本轮离线脚本按 `episodes/seed_*/` 轨迹重算。
