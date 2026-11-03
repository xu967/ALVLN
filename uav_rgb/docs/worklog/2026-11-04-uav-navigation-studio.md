# 2026-11-04 UAV Navigation Studio 桌面工作台

## 目标

用户要求把训练监控、实验留档和 Gazebo 三视角仿真整合到可直接在 Ubuntu 桌面运行的工作台，不以网页静态原型或预录视频替代实时数据。

## 实施

- 新增 `ui/studio.py`：Tkinter 原生桌面程序，包含 Dashboard、Training、Simulation、Experiments 四页。
- Training 默认调用真实 `ui/trainer_native.py` → `build/check/uavrgb_train`，不默认启动 `trainer_demo.py`；支持选择 `.uavr`、Epochs/Batch/LR/Seed/Text dim、class weights、drop-collision；实时读取 `results.csv`、`metrics.jsonl` 和 trainer stdout/stderr。
- 每次 run 写入独立 `studio_runs/<timestamp>-<id>/`；训练完成自动生成 `studio_summary.json`、`experiment_report.md`，同级 `.monitor/` 保存配置、lifecycle 和日志；记录离线 test 四件套，并明确与闭环导航 SR 分开。
- 新增实验历史、best checkpoint 选择、正式 `eval_closed_loop.py` 启动与结果归档；支持桌面关闭后子进程继续运行，重开 Studio 时恢复/归档 detached run。
- Simulation 页新增三路 Gazebo 渲染相机：原 `/uav/camera` RGB、`/uav/chase_camera` 虚拟追踪相机、`/uav/top_camera` 固定顶视相机；另保留 `/model/x3/pose` + SDF collision footprints 的二维轨迹地图。
- Simulation 页支持从桌面选择/生成 `base/l1/campus` 场景、启动或连接 Gazebo、运行现有 `collect.py --policy student` 做单集学生策略演示。单集演示明确不冒充正式 SR。
- `sim/worlds/uav_rgb.sdf` 新增后上方虚拟 chase camera（640×360 @15 Hz）和固定顶视 camera（480×360 @10 Hz）；均无质量、无碰撞，不进入 student observation；原 RGB 相机不变。
- `trainer_native.py` 默认 trainer 路径修正到 `build/check/uavrgb_train`，并兼容 `UAV_RGB_TRAIN_BIN` / 历史 `UAVRGB_TRAIN_BIN`。
- 新增 `ui/launch_studio.sh`、`ui/install_desktop_entry.sh`、`ui/README_STUDIO.md`、`ui/test_studio.py`；`tools/check.sh` 加入 Studio 单测。
- 更新 `test_spawn.py` 的 legacy world golden：变化仅来自新增虚拟相机传感器；更新 `test_student.py` 的启动超时测试，避免 300 ms 进程冷启动抖动误判。

## 验证

通过：

- `python3 -m py_compile ui/studio.py ui/trainer_native.py ui/test_studio.py`
- `python3 ui/test_studio.py`：6 项通过。
- `python3 ui/test_monitor.py`：74 项通过、1 项 GUI 环境跳过。
- `python3 ui/test_trainer_demo.py`：92 项通过。
- `python3 sim/tools/test_spawn.py`：通过（20 seeds × 2 spawn modes；legacy golden 已按 chase sensor 更新）。
- `python3 sim/tools/test_student.py`：通过。
- `python3 sim/tools/test_teacher_policy.py`：单独运行通过。
- `xvfb-run` 桌面初始化冒烟：4 个 tab、3 个视角控件成功创建。
- `xvfb-run` 假 trainer 集成测试：真实 launcher → trainer_native → fake binary → summary/report → dashboard test Macro-F1 显示，端到端通过。
- `xvfb-run` PNG camera 解码/PhotoImage 和 top-down 绘制冒烟通过。
- `scene_gen.py` base/campus 生成测试：SDF collision footprints 可解析，metadata `world_sha256` 与输出 SDF 一致。
- `bash -n`：两个新增 shell launcher 通过；SDF XML 静态解析通过。

未完成的验收：

- 本次工作容器没有 `gz` 可执行文件，也没有连接用户的 RTX 2080 Ti / Gazebo 8 桌面会话；因此 chase sensor 的真实渲染、`gz topic` 订阅、GPU 帧率和实际学生策略飞行尚未进行 L3 实测。
- `tools/check.sh --python-only` 聚合执行在该环境中超时，未跑完整个测试循环；相关新旧单测分开运行通过，但不能宣称完整 L2 聚合检查通过。
- 未运行 C++/CUDA 构建、CTest、GPU 训练或真实闭环评测。

## 使用

```bash
cd <project>/uav_rgb
bash ui/launch_studio.sh
# 可选：安装 Ubuntu 应用菜单和桌面快捷方式
bash ui/install_desktop_entry.sh
```

旧 world SDF 不含 chase sensor；必须用更新后的模板重新生成场景。第一人称输入保持 `/uav/camera`。正式导航成功率必须通过单独的闭环评测生成 `eval.json`，不能由离线分类指标或单集 live demo 推断。
