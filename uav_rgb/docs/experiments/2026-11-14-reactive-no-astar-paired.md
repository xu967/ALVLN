# 2026-11-14：无 A* 在线控制配对实验

- 时间：2026-11-14
- 环境：RTX 2080 Ti；`/root/rivermind-data/uav_rgb/3dgs-venv`；cf_sitl；hold=0.5 s；显式 STOP
- 场景：`sage3d_841980` 的 `ts_2` / `ts_3` 局部控制探针，以及 `sage3d_840865` 的 `tr_11` / `tr_49` 同起点、不同终点正式配对（统一 seed=141114）
- 目的：验证在线控制不调用 A* 时，局部控制层是否能完成目标条件飞行

## 实验口径

`reactive` 策略由 `sim/3dgs/reactive.py` 实现。它读取当前深度、当前位置、航向和目标方向，输出四动作及独立 STOP；不读取 `OccPlanner` 或全局 occupancy。目标方向仍是评测器提供的 waypoint，因此这是控制层 G0 诊断，不是语言理解实验。

SPL 仍按现有评测器的离线 A* 参考路计算，不能解释为“在线用了 A*”；后续无 A* 主线应改用 SAGE 参考轨迹长度并单列旧口径。

## 结果

| 策略 | SR | OSR | NE 均值(m) | 碰撞率 | STOP率 | 结局 |
|---|---:|---:|---:|---:|---:|---|
| reactive 首版 | 0/2 | 0/2 | 5.159 | 0/2 | 0/2 | 2 集预算耗尽 |
| reactive + 3 帧方向锁存 | 0/2 | 0/2 | 5.115 | 0/2 | 0/2 | 2 集预算耗尽 |
| A* teacher（同场景另一组 paired smoke） | 0/2 | 0/2 | 4.388 | 2/2 | 0/2 | 2 集碰撞 |

上表 841980 路线在起点附近容易局部卡住，且 A* 参考也碰撞，因此另选同起点双目标任务做固定 seed 对照：

| 场景/策略 | SR | OSR | NE 均值(m) | 碰撞率 | STOP率 | 逐集结果 |
|---|---:|---:|---:|---:|---:|---|
| 840865 reactive | 1/2 | 1/2 | 3.056 | 0/2 | 1/2 | `tr_11` 超时；`tr_49` 成功，NE=0.793 m |
| 840865 A* teacher | 2/2 | 2/2 | 0.464 | 0/2 | 2/2 | 两集成功 |

逐集结果保存在：

- `/root/rivermind-data/uav_rgb/tmp/exp-reactive-paired-20261114/reactive/vln_eval.json`
- `/root/rivermind-data/uav_rgb/tmp/exp-reactive-paired-20261114/reactive-latched3/vln_eval.json`
- `/root/rivermind-data/uav_rgb/tmp/exp-reactive-paired-20261114/astar-reference/vln_eval.json`
- `/root/rivermind-data/uav_rgb/tmp/exp-reactive-paired-20261114/sage840865-reactive/vln_eval.json`
- `/root/rivermind-data/uav_rgb/tmp/exp-reactive-paired-20261114/sage840865-astar-same-seed/vln_eval.json`

补充的同起点 paired smoke（`sage3d_840865`，`tr_11`/`tr_49`）此前已完成：A* SR=1/2、OSR=2/2、碰撞=1/2；reactive SR=1/2、OSR=1/2、碰撞=0/2，但该结果只有两集，不能作为性能结论。

## 结论

固定 seed 的 840865 配对显示，reactive 无碰撞完成 1/2，低于 A* 特权教师的 2/2；841980 又显示两者都可能受局部困境或教师碰撞影响。方向锁存降低了明显左右摆振，但没有解决局部最小和目标不可见。继续微调三扇区阈值会对单条路线过拟合；下一轮应改成带短期记忆的局部轨迹/速度策略，并将语言目标落地单独作为 grounder 评测。这些各 2 集的数字只用于选方向，不是性能结论。
