# 无 A* 在线控制原型与配对验证

- 时间：2026-11-14 07:30
- 范围：`sim/3dgs` VLN runner、局部控制基线、SAGE 配对 smoke
- 结论：部分完成

## 做了什么

- 新增 `sim/3dgs/reactive.py`：目标条件深度扇区局部控制器，带独立 STOP、方向滞回和三帧避障方向锁存；在线不访问 A* 全局规划。
- 在 `sim/3dgs/run_vln.py` 增加 `--policy reactive`，并更新策略说明，明确它是控制层诊断，不是语言模型。
- 在 `sim/3dgs/test_vln.py` 增加反应式转向、目标停止、清障方向和方向锁存测试。
- 跑了 `sage3d_841980` 同起点不同终点的 reactive、锁存 reactive、A* 参考配对实验；结果单独记录到 `docs/experiments/2026-11-14-reactive-no-astar-paired.md`。
- 更新 `docs/PLAN_2026-11-13_vln.md §10.7`，明确下一阶段转向“语言 grounder + 短历史局部视觉策略 + 安全层”。

## 没做什么

- 没有把 waypoint/目标方向伪装成语言理解；当前 reactive 不读取文本，不能报告语言 SR。
- 没有移除仓库中的 A*；A* 仍用于历史教师、离线路线质检和现有 SPL 参考。彻底移除导入器/评测器中的 A* 需要单独定义新的可达性与参考路径口径。
- 没有训练 CNN/GRU、grounder、连续速度策略或 DAgger；本轮先验证最小控制层原型。
- 没有修改真实飞控 PID、执行器协议或硬件参数；当前证据来自 3DGS + cf_sitl。
- 没有把两个 841980 路线的失败当成算法排名；A* 同组也发生碰撞，样本太小且动力学问题未分离。

## 待确认

- 下一步是先实现 SAGE 轨迹局部切线/速度标签与短历史控制器，还是先完成 grounder 的候选地标接口；计划建议先做控制器 G0，再接 G1 grounder。
- 无 A* 主线的 SPL 是否改用 SAGE 参考轨迹长度；在口径冻结前继续保留旧 A* SPL 但单独标注。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /root/rivermind-data/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_vln.py` | 通过，31 tests，OK |
| reactive 841980 配对 | 运行完成，SR 0/2，OSR 0/2，碰撞 0/2，2 集超时 |
| reactive 锁存 841980 配对 | 运行完成，SR 0/2，OSR 0/2，碰撞 0/2，2 集超时 |
| A* 841980 参考配对 | 运行完成，SR 0/2，OSR 0/2，碰撞 2/2 |
| GPU | RTX 2080 Ti 已检测并用于真实 renderer/cf_sitl |

## 风险与遗留

- 目标不可见时，纯局部反应式控制会陷入局部最小；方向锁存只减少摆振，不提供全局探索和重定位。
- reactive 当前通过 state 真值获得目标方向，不能直接部署；必须在接入语言 grounder 后再做正式 VLN 反事实评测。
