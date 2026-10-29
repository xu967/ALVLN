# E6-2~E6-4：学生闭环评测链路

- 时间：2026-10-29 22:32
- 范围：`sim/tools/`（student / collect / eval）、`src/`（复用 E6-1 推理进程）
- 结论：完成。真实 Gazebo 跑通 1 集学生闭环，指标 JSON 与退出码可用；
  当前 checkpoint 的 SR=0 只说明链路能测出失败，不是模型质量结论。

## 做了什么

- `sim/tools/student.py`：常驻 `uavrgb_infer` 子进程；把 gz RGB 帧按
  preprocess 同口径（PIL bilinear，320×240→模型尺寸）缩放后喂模型。
- `sim/tools/collect.py`：`--policy student` + `--student-ckpt` + `--infer-bin` +
  `--stop-on-collision`；actions.jsonl 增 `infer_ms`；manifest 的 `teacher=student`，
  新增 `stopped_on_collision`（`validity` 仍按「跑满请求步数」严格填）。
- `sim/tools/eval_closed_loop.py`：每个 episode 生成固定 seed 世界 → 只保留目标
  split（几何组隔离）→ 独立 `GZ_PARTITION` 起 sim → collect 学生闭环 →
  解析 actions/oracle → 汇总 SR / 碰撞率 / 误前向 / 推理 p50·p95 / 动作分布。
- `sim/tools/test_student.py`、`test_eval_metrics.py`：协议/缩放/异常格式、指标纯函数。
- `docs/experiments/2026-10-29-closed-loop-smoke.md`：首集真实输出。

## 没做什么

- **没有仿真时间动作窗口**（D1 未做）。原因：独立改造；因此 `hold` 仍是墙钟秒，
  评测结果随 RTF 漂移，不能跨机复现。
- **没有 20 集基线**（E6-5 未做）。原因：先交付链路；1 集只验证端到端。
- **没有近碰撞 / 未响应 / 动作抖动 / time_to_collision_warning**。原因：
  需要深度时序与更多状态记录，当前协议没有这些量。
- **没有多实例并行评测**。原因：串行一集一 sim 先保证干净；并行会放大
  Gazebo 资源冲突，等 D1 后再做。
- **没有接入 check.sh**。原因：闭环需要 Gazebo/显示资源，属 L3；check.sh 只跑纯函数自检。
- **没有验证模型质量**。原因：现有 checkpoint 只在小样本上训练，评测场景按 salt v2
  属 test，但旧 checkpoint 的训练数据用旧 split，几何是否见过无法保证。

## 待确认

- 成功定义：当前 `max_x >= 最远柱 x + 1.0 m` 且无碰撞。是否改成「穿过最远柱后
  再前进 0.5 m」或按目标点判定。
- `false_forward_rate` 当前是 episode 级（该集是否发生过「前进且碰撞」）；
  是否还要 step 级比例。
- 评测是否需要保留 RGB PNG（当前 collect 仍写图，磁盘占用随集数增长）。

## 验证到哪一步

| 项 | 真实输出 |
| --- | --- |
| `test_student.py` | PASS |
| `test_eval_metrics.py` | PASS |
| `test_collect.py` | 14 项通过 |
| `bash tools/check.sh`（DISPLAY=:20） | **EXIT=0**，19/19 CTest，Python 自检全过 |
| 真实闭环 1 集（seed=5, split=test, 20 步上限） | `steps=8 collision=True success=False max_x=1.06`，`collect_rc=0`，wall 30.7 s |
| 推理延时（同一集） | p50=9.14 ms，p95=11.28 ms（warm n=7），首帧 310.65 ms |
| 残留进程 | `ps` 无 gz sim / ruby / uavrgb_infer |

## 风险与遗留

- `eval_closed_loop.py` 复用了 `collect_parallel` 的私有 `_start_sim/wait_ready/stop_group`；
  对方重构会静默打断评测。等评测稳定后再提取公共模块。
- `hold` 是墙钟秒；同 seed 两次评测可能不同。D1 完成前不要把 SR 写进对外结论。
- 旧 checkpoint 的训练 split 与 salt v2 不一致；E6-5 的 20 集基线只能标「链路验证」。
- 首帧 310 ms 是 CUDA 初始化，已单列；p50/p95 用 warm 帧。
