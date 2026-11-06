# 2026-11-07 FAST Lab 规划侧精读 → E1 动力学感知教师（teacher_kino）

- 时间：2026-11-07 00:20–01:10 · 范围：`sim/tools/teacher_kino.py`、`test_teacher_kino.py`、`compare_teachers.py`、`sim/dynamics/e1_mini.json`（复制）· 分支 `fastlab-20261106`（不推送）
- 结论：旧全局 A* 教师（X3 常量）在 E1 0.3 m/s 的离线闭环里**没坏**（20 seed × 3 组响应常数 100% 到达 0 碰撞）；新教师持平并多出速度感知/必撞 fail-closed；差距在 0.6 m/s + 慢响应时出现（旧 25% vs 新 100%）。采集管线暂不建议切教师。
- 详细报告：`/root/rivermind-data/uav_rgb/tmp/phase2-20261106/reports/FASTLAB_A_planner_20261107.md`

## 做了什么

- 精读 Fast-Planner `kinodynamic_astar.cpp`、visPlanner `tracking_astar.hpp`、ego-planner `bspline_optimizer.cpp / ego_replan_fsm.cpp / planner_manager.cpp / dyn_a_star.cpp`、EGO-v2 `traj_server.cpp / planner_manager.cpp`、TGK `pos_checker.cpp / krrtplanner.cpp / fsm.cpp`、EVA `high_mpcc_optimizer.cpp`、Assistive-Racing `risk_monitor.cpp`、GPA `ego_replan_fsm.cpp / poly_traj_optimizer.cpp`；file:line 与 8 条借鉴（B1–B8）写在报告 §1–§2。
- `sim/dynamics/e1_mini.json`：从主仓库原样复制到工作树同路径（teacher_kino 读包络 0.102 m、执行器 0.3 m/s / 0.6 rad/s、相机延迟 0.10 s）。
- `sim/tools/teacher_kino.py`（纯 stdlib，~400 行）：四动作运动原语（0.1 s 悬停=上一条过期 + 0.9 s 动作，一阶响应 τ_up 0.313 / τ_down 0.393 / τ_yaw 0.091 由 SITL 实测折算）→ kinodynamic A*（0.1 m × 10° 剪枝、8 步视野、1500 展开上限）；碰撞按包络 + 0.08 m 硬余量逐点查 0.05 m 距离栅格；每个节点再查「悬停 1 s 能停住」（必撞剪枝）；目标根 Dijkstra 距离场做启发；0.5 m 软余量带 W=10、换动作罚 0.15 s；停靠模式用悬停后停点判到达。接口与 `teacher.plan_action_global` 对齐，多可选 `vx/vy/yaw_rate/prev_action`。
- `sim/tools/test_teacher_kino.py`：70 项（参数来源、原语物理对照实测、门柱闭环、必撞 → SAFE、漂进余量脱困、停靠语义、20 test seed 闭环、耗时、确定性），18 s；`tools/check.sh` 的 `test_*.py` 循环自动包含。
- `sim/tools/compare_teachers.py`：独立欧拉积分被控对象（dt 0.01、延迟期间悬停、世界系速度一阶响应），两任务 corridor / l1_stop × 三组 plant × 4 教师，输出 summary.md/json + 轨迹 PNG；用守护进程跑全量（`agent-fastlab-a/compare/{run.log,run.pid,DONE}`）。
- 提交（按主题）：8f6d540 e1_mini.json 复制；325de7c teacher_kino + 单测；后续 compare 脚本、docs 各一笔（见 git log）。

## 没做什么（原因）

- 没改 `collect.py`、没切教师：任务只要原型 + 离线对比；且结论是当前 E1 工况下不必切。
- 没把 τ 写进 `e1_mini.json`：SITL-core 的文件，schema 改动留主会话决定；代码标 `shortcut:`。
- 没跑 Gazebo/SITL/GPU（任务约束）；被控对象没有高度/风/位姿噪声。
- 没借 EVA 速度方向因子、GPA 机头-速度约束：离散原语天然满足。
- kino_sampling_with_regional_opti 只对照结构（与 TGK 同骨架），没逐行读。
- 没把 PNG 入库，留在数据盘。

## 待确认

- 旧全局教师在 E1 上"够用"是离线几何结论；主会话若在 Gazebo/SITL 上看到失败，先查偏航跟踪/延迟/感知，而不是路径几何。
- `teacher_kino` 是否进 `collect.py` 作为可选 `--teacher kino`：建议等 SITL 回放验证一窗位移/转角误差后再定。
- l1_stop 里 kino 的最小净空 0.35–0.38 m（旧 0.47–0.50）是否可接受：由 waypoint 停靠圈贴柱 0.1 m 的几何决定，调大 `CLEAR_SOFT` 可换。

## 验证到哪一步

- `python sim/tools/teacher_kino.py`：8/8 通过（FORWARD 一窗 0.181 m、刹停滑行 0.118 m、YAW 一窗 27.7°、seed 180 首步 YAW_LEFT 0.60 s 含建场、热缓存 0.016 s）。
- `python sim/tools/test_teacher_kino.py`：70/70 通过，18.3 s；20 seed 闭环最小净空 ≥0.44 m，热缓存单步 max 0.038 s。
- `compare_teachers.py` 全量（20 seed）：
  - corridor nominal：astar_x3 1.00/0 碰撞/22.1 步；local_x3 0.85/0/26.1（142 次左右换向）；predictive_cfstab 0.00（SAFE 死锁）；kino_e1 1.00/0/22.0。
  - corridor sluggish：astar 1.00；local 0.10（911 次换向）；kino 1.00。heavy：astar/local/kino 1.00，predictive 0。
  - l1_stop 三组 plant：astar 1.00（15.2/17.2/19.4 步）；kino 1.00（14.1/16.9/19.2 步）；predictive 0。
  - 0.6 m/s 变体 corridor：heavy plant 下 astar_x3 0.25（288 次换向、28% SAFE），kino_e1 1.00/0 碰撞；nominal 下 predictive 25% 碰撞。
- predictive_cfstab 0 到达的根因已单独复现：其 `DistanceField` 在 seed 180 x∈[1,2]、|y|≤0.5 全为 60（不可达）。

## 风险与遗留

- 被控对象与教师共用同一组响应常数（故意各写一份），是"模型对模型"的验证；真实差距要等 SITL 回放。
- kino 单步 35–45 ms（纯 Python）；若 hold 缩到 0.5 s 以下或栅格再细，要把距离栅格/Dijkstra 换 numpy。
- `scene_gen.sample_pillars` 的 20 个 test seed 只有约 6 种布局且都留有直线走廊，对比区分度有限；更密的柱阵（缝 <0.8 m）没测。
- 守护进程产物在 `/root/rivermind-data/uav_rgb/tmp/agent-fastlab-a/{compare,compare-fast06}/`，含 `e1_mini_fast06.json`（0.6 m/s 副本，仅实验用）。
