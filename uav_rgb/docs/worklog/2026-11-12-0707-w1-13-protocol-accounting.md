# 卡 W1-13 第一轮：新协议记账、口径闭环、打包过滤与离线核对

- 时间：2026-11-12 07:07
- 范围：`sim/tools/{collect,cf_sitl_exec,eval_closed_loop,collect_parallel}.py`、`tools/preprocess.py`、对应测试与 `experiments/2026-11-12-w1-13-newprotocol/`（分支 `p12-w1-13`）
- 结论：部分完成（第一部分与第二部分代码、测试、离线核对、第三部分设计要点已完成；第 3 条 Gazebo 实测守护进程已启动并在 `machine.lock` 队列等待，第三部分按交办本轮不动）

## 做了什么

- 接手审查前任 4 个提交（`e9822de`、`51ba557`、`579901d`、`5460af6`），清理工作树根目录失效占位文件 `OWNER-3fdfa650.txt`。
- 修改 `sim/tools/collect.py`、`sim/tools/eval_closed_loop.py` 与 `sim/tools/test_step_gaps.py`：更正 `51ba557` 硬编码名义步周期 `0.9 s` 的问题，改为按 `max(0.1, hold - 0.1)` 计算，修复非 `1.0 s` 窗口下的停顿步误判与漏判（提交 `4d8a4ff`）。
- 修改 `sim/tools/test_step_gaps.py`：补充文件落盘级回归测试并通过变异检查，锁住默认不带 `--action-latency` 时 `episode_manifest.json`、`oracle.jsonl` 及剔除新增键后的 `actions.jsonl`、`eval.json` 逐字节不变（提交 `7353e10`）。
- 补完并提交 `experiments/2026-11-12-w1-13-newprotocol/{README.md,run_smoke_ack.sh,rehearse_smoke_ack.sh,stub_eval_ack.py,summarize_ack.py}` 与 `sim/tools/test_w113_summaries.py`：补齐起跑记录追加不覆盖、`env_overrides` 记录、`EXIT/TERM/INT/HUP` trap 清理 8 个占核进程组、半成品目录披露与测量轨迹数校验，桩演练通过后启动守护进程排 `machine.lock`（提交 `f4a2315`）。
  - 守护进程信息：启动命令 `cd /root/rivermind-data/uav_rgb/exp-w1-13-ack-20261112 && setsid nohup bash run_smoke_ack.sh > /dev/null 2>&1 < /dev/null &`；PID/PGID `78842`（启动时刻 `Thu Nov 12 07:19:12 2026`）；日志 `/root/rivermind-data/uav_rgb/exp-w1-13-ack-20261112/run.log`；进度查看 `tail -f .../run.log`；完成标记 `.../DONE`；停止命令（核对 `run.meta` 启动时刻后）`kill -- -78842`。
- 修改 `sim/tools/eval_closed_loop.py` 与 `sim/tools/test_action_latency.py`：当 `--protocol` 快照或学生训练数据集 `meta` 无 `action_latency_s` 而本次评测给了 `--action-latency L` 时，`eval.json` 顶层记录 `protocol_notes: ["train_protocol_without_action_latency"]`，不判 OOD（提交 `46cd88c`）。
- 修改 `sim/tools/collect_parallel.py` 与 `sim/tools/test_collect_parallel.py`：透传 `--action-latency` 与 `--rtf`，默认不传时命令行与 `collection_manifest.json` 逐字不变（提交 `90ed8d7`）。
- 修改 `tools/preprocess.py` 并新增 `tools/test_preprocess_timing_filter.py`：增加 `--drop-late-and-stall` 开关过滤迟到步与停顿步行，默认关闭时 `dataset.uavr` 与 `dataset.aux` 逐字节不变，并用与 main 逐字节相同的 `tools/test_preprocess_label_alignment.py` 验证通过（提交 `aa9a6ff`）。
- 独立重跑核实 W1-5 误起窗口对第二轮 `acc-L0.20-k1-s2` 与 `acc-L0.20-k3-s0` 的离线对比，通读 `sim/cf_sitl/gz/CfSitlSystem.cc`、`sim/cf_sitl/exec/executor.c` 与 `sim/3dgs/{collect_3dgs,dynamics}.py` 核实第三部分实现要点与 3DGS 无步间空档，写入 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/w1-13.md`。

## 没做什么

- 未等待 Gazebo 守护进程（PGID `78842`）跑完空闲与满载各 5 集的真机仿真测量
  - 原因：Gazebo 池 `machine.lock` 当前被 E2E-R1（PGID `18148`）及主会话排队器（PGID `23019`）持有，按公共规则 §6 与交办要求不阻塞会话干等。
- 未修改 `sim/tools/collect.py` 的迟到判据 `late_dispatch`
  - 原因：交办要求「这一轮不改迟到判据」，需等 PGID `78842` 测出空闲与满载 `p99` 后由主会话决策。
- 未实现任务卡第三部分（`--action-until next-dispatch`、四锁独占标定、学生参照臂）
  - 原因：交办明确本轮只做第一部分与第二部分，第三部分不动、只写实现要点。
- 未修改 `tools/stats_report.py` 与 `tools/{e1_pipeline,dagger_gz_pipeline,dagger_pipeline}.py`
  - 原因：任务卡 02:15 更新已分别移交卡 W1-14 与卡 W1-15b。
- 未重评第二轮的 `acc-L0.20-k1-s2` 与 `acc-L0.20-k3-s0`
  - 原因：交办要求只报告不重评，且离线核对证实无需重评。
- 未跑全量 `tools/check.sh`（C++/CUDA 构建与 GPU CTest）
  - 原因：本轮只修改 Python 工具与脚本，未触碰 C++/CUDA 代码，按规范执行排 `check.lock` 的 `tools/check.sh --python-only`。

## 待确认

- 待守护进程 PGID `78842` 落盘 `/root/rivermind-data/uav_rgb/exp-w1-13-ack-20261112/DONE` 与 `summary.md` 后，根据两条件 `plugin_ack_sim_t - (pose_ts + L)` 的 `p99` 对 `10 ms` 的结果，拍板是否改迟到判据及具体阈值。
- 确认第二轮 `acc-L0.20-k1-s2` 与 `acc-L0.20-k3-s0` 采纳「不重评」结论。
- 确认报告第 6 节「第三部分的实现要点」的边界规则与双口径标定汇报方式，并在第三部分落地时同步通知卡 W1-14 将 `action_until` 加入 `stats_report.py` 协议一致性键。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/tools/test_step_gaps.py`（70 项，含变异检查） | 通过（`rc=0`） |
| `sim/tools/test_w113_summaries.py`（18 项）与 `rehearse_smoke_ack.sh`（5 场景桩演练） | 通过（`rc=0`） |
| `sim/tools/test_action_latency.py`（27 项）、`test_collect_parallel.py`（8 项）、`test_collect.py`（130 项）、`test_eval_summary.py`、`test_eval_provenance.py`、`test_eval_closed_loop_args.py`、`test_scene_rtf.py` | 通过（`rc=0`） |
| `tools/test_preprocess_timing_filter.py`（5 项）、`tools/test_preprocess_label_alignment.py`（2 项，与 main 一致）及其余 4 个 `tools/test_preprocess_*.py` | 通过（`rc=0`） |
| `second-executor/test_offline_check.py`（23 项）与 `offline_check.py` 复跑对拍 | 通过（`rc=0`，产物零差异） |
| 排 `check.lock` 执行 `bash tools/check.sh --python-only` | 通过（`rc=0 2026-11-12 07:54:32`，含 `sim/tools`、`tools`、11 个 `sim/3dgs` 与 `ui` 全部测试，日志 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-13/logs/check-python-only/check.log`） |

## 风险与遗留

- 守护进程 PGID `78842` 仍在等待 `machine.lock`，若容器再次意外重启需用原命令在 `/root/rivermind-data/uav_rgb/exp-w1-13-ack-20261112` 下续跑（脚本支持幂等续跑并自动保留半成品目录）。
- `tools/preprocess.py` 在 `--drop-collision` 或 `--drop-late-and-stall` 丢掉某步后，下一步的双帧 `prev_index` 退回指向自身而 `prev_action` 保留真实执行动作（审核 C7）；当前主线单帧模型不受影响，双帧模型升级时需评估是否保留被丢帧像素供 `prev_index` 引用。
