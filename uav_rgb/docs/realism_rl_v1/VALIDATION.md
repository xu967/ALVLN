# 已执行验证（打包时，2026-11-04）

容器环境只能完成离线 Python 逻辑测试，不存在 Gazebo GUI/服务或 NVIDIA CUDA 运行条件；没有安装 `gymnasium` 与 `stable_baselines3`，因此没有实际训练 PPO，也没有虚构任何成功率。下面两条 teacher/test_dataset_*.py 需带可用 torch 的解释器（如 $DATA_ROOT/uav_rgb/teacher-venv/bin/python）；系统 python3 的 torch 缺 libcudnn.so.9 时会 ImportError，不属于代码回归失败。

已通过的检查：

- `python sim/tools/test_asset_campus.py`：160 个新场景 SDF，XML 可解析、确定性、资源路径存在、拆分不泄漏、目标颜色/相对方向、A* 可达、碰撞模型与元数据一致；PASS。
- `python -m rl_teacher.test_core`：数值积分、延迟、空间净空、测地距离、PBRS 两步折扣恒等式、奖励拆分、13维观测；PASS。
- `python teacher/test_dataset_windows.py`：原有滑窗回归，PASS。
- `python teacher/test_dataset_step_gap.py`：新增丢帧/原始 step 回归，PASS。
- `python sim/tools/test_campus.py`：旧校园场景兼容，PASS。
- `python sim/tools/test_collect.py`：46 项原有纯函数自检，PASS。
- `python sim/tools/test_eval_closed_loop_args.py`、`test_collect_parallel.py`：PASS。
- `python sim/tools/test_eval_metrics.py`、`test_eval_summary.py`、`test_instruction.py`、`test_teacher_policy.py`：PASS。
- Python `compileall`：新旧仿真工具、Teacher 模块语法检查通过。
- `scene_gen.py --task campus_assets --seed 19`：输出 SDF 和 JSON 元数据；文件存在且内容具备完整字段。

**待在 Ubuntu + Gazebo 环境执行：** 实际 mesh/PBR 的可见性与材质显示；三视图性能；T0/T1/T2 在真实 Gazebo 服务上的闭环；随机 seed 的跨地图泛化；PPO 在装好依赖后的实际训练及模型保存/加载；CUDA 编译、Jetson 真机部署。

> 这些离线测试不证明新算法提升了导航性能，最终论文必须报告独立闭环结果和相同设置的基线对照。
