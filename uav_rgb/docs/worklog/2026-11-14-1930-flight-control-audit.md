# 飞行控制与抖动审计

- 时间：2026-11-14 19:30
- 范围：`sim/3dgs` 控制策略入口、A* 教师净空检查、`sim/cf_sitl` 执行器和动力学 profile，以及现有 VLN/净空实验记录。
- 结论：完成只读审计；确认可先做软件层测量和策略改进，不足以据此调整真机 PID 或执行器参数。

## 做了什么

- 对照 Git 状态和 `run_vln.py` 的未提交 lexical 改动，确认控制器接收 grounder 目标，但评估仍使用场景原始目标；`teacher` 仍通过默认 `clearance="ray"`。
- 阅读 `planner.py`、`executor.c`、SITL README 与 `e1_mini.json`，梳理固定动作窗口、SAFE/YAW 位置保持、前进速度参考和参数校准状态。
- 核对 M1 碰撞归因、旧 swept 净空实验、reactive 配对实验、Wanderland smoke 与 VLN 计划卡 C，整理目前抖动的证据边界和下一步最小实验。
- 验证现有轻量测试：`test_dynamics.py` 6 项、`test_dynamics_sitl.py` 12 项、`test_guard.py` 24 项、`test_teacher_clearance.py` 53 项、`test_teacher_tol.py` 18 项均通过。

## 没做什么

- 未改代码、控制参数、固件或用户的未提交 lexical 改动；本次子任务明确要求只审计。
- 未启动长路线仿真、重训练或真机飞行；先报告控制稳定性核心结论，避免没有必要的算力负载。
- 未跑全量 pytest；环境的系统 Python 没有安装 pytest，改用项目现有测试脚本直接运行。
- 未把 SITL 的姿态噪声或碰撞数据称作真机测量；动力学 profile 标记为 `calibrated=false`，多项物理参数仍是假设值。

## 待确认

- 后续是否按计划卡 C 优先采集不同动作切换、速度和 hold 下的刹停距离、侧滑和终速度，再决定是否改教师净空模型。
- 只有证据显示主要抖动来自执行器参考量突变时，才需进一步讨论改 `executor.c` 并同步真机固件。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| Git 状态、`run_vln.py` diff 与控制调用链检查 | 已检查；工作树原有 13 个文件修改，未改动 |
| `python3 sim/3dgs/test_dynamics.py` | 通过，6 项 |
| `python3 sim/3dgs/test_dynamics_sitl.py` | 通过，12 项 |
| `python3 sim/3dgs/test_guard.py` | 通过，24 项 |
| `python3 sim/3dgs/test_teacher_clearance.py` | 通过，53 项 |
| `python3 sim/3dgs/test_teacher_tol.py` | 通过，18 项 |
| 长时仿真、训练、真机测试 | 未运行；不在本次只读审计范围内 |

## 风险与遗留

- VLN 默认 `hold=0.5`，而 swept 位移模型只在 `hold=1.0` 标定，当前入口不得直接启用该模式。
- 旧 swept 有限等待后会放行，且 SAFE 等待可能继续漂移；重用前需重新设计并配对验证。
- M1 的原地转向碰撞和 stop-turn-go 切换提示可先从软件策略优化，但 SITL 参数未标定，不能据此宣称实机稳定。
- 方向锁存配对样本各只有两集；路线 smoke 和视频镜头低通也不能作为飞行平顺性证据。
