# UAV Navigation Studio（桌面端）

这是 `uav_rgb/` 主线的原生桌面工作台，使用 Python Tkinter，不依赖浏览器、FastAPI 或云服务。
它读取现有 C++/CUDA trainer 的真实输出，并通过 Gazebo Transport 订阅相机和位姿。

## 启动

在 Ubuntu 桌面会话中：

```bash
cd <project>/uav_rgb
bash ui/launch_studio.sh
```

也可以安装应用菜单项和桌面快捷方式（仅修改当前用户目录，不需要 sudo）：

```bash
bash ui/install_desktop_entry.sh
```

之后可以从 Ubuntu 应用菜单搜索 `UAV Navigation Studio`，或双击桌面快捷方式启动。`launch_studio.sh` 会加载 `env.sh`，并寻找带 Tkinter 的系统 Python。默认实验目录：

```text
$DATA_ROOT/uav_rgb/studio_runs/<timestamp>-<run-id>/
```

如果没有设置 `DATA_ROOT`，桌面程序使用 `~/rivermind-data`。训练产物、实验摘要和日志不会写入源码目录。

## 训练工作台

1. 在「训练 Training」选择真实 `.uavr` 数据集。选择文件后，Studio 会根据同目录的 `instructions.jsonl` 自动把 Text dim 预设为 16 或 0。
2. 按数据协议确认 `Text dim`：`0` 表示显式纯视觉，当前文本分支通常为 `16`；如果没有 instruction 侧车，Studio 会阻止以 `text_dim>0` 启动。
3. 选择 Epochs、Batch、学习率、Seed、类别权重和是否剔除碰撞样本。
4. 「训练后保存视频（闭环 1 集）」默认勾选：训练成功且存在 `ckpt_best.bin` 时，Studio 在后台自动跑 1 集闭环录像评测，不阻塞界面。配置里的 `save_video` / `video_episodes` 只供 Studio 使用，`trainer_native.py` 会忽略这两个未知键。
5. 点击「开始真实训练」。程序调用 `ui/trainer_native.py`，再由它执行 `build/check/uavrgb_train`（或 `UAV_RGB_BUILD_DIR` / `UAV_RGB_TRAIN_BIN` 指定的可执行文件）。不会默认调用 `trainer_demo.py`。

运行期间实时读取：

- `metrics.jsonl`：train step、ETA、validation、Macro-F1、Balanced Accuracy、错误和 checkpoint 状态；
- `results.csv`：每轮 train/dev CE 与 dev accuracy；
- stdout/stderr 合并日志：保存在 `<run-dir>.monitor/train.log`。

训练结束后自动生成：

- `studio_summary.json`：运行状态、配置、离线验证/测试指标和 checkpoint 存在性；
- `experiment_report.md`：可直接阅读的实验报告；
- `<run-dir>.monitor/job.json`、`lifecycle.json`、`train_config.json` 和原始日志；
- `$DATA_ROOT/uav_rgb/studio_runs/index.jsonl`：实验索引。

离线 test accuracy / Macro-F1 / Balanced Accuracy 是动作分类指标，不等于闭环导航成功率。缺失的 test 指标保持 null/“—”。

### 训练后自动录像

勾选「训练后保存视频（闭环 1 集）」时，训练退出码为 0 且 `ckpt_best.bin` 存在后，Studio 会用训练数据集的 `.uavr` 尾部 meta 自动推导评测参数：

- **task**：`task_version` 为 `campus.assets*` → `campus_assets`，`campus*` → `campus`，`l1*` → `l1`；没有 `task_version` 但 `geometry_version` 是 `uavrgb.geom.*` → `base`；
- **split**：`splits.test > 0` 用 `test`，否则 `splits.val > 0` 用 `val`，两者都空则跳过；
- **dynamics**：`x3_v1*` → `x3_velocity`，`cf_stab*` / `cf_sitl*` / `cf_sil*` / `cf_blackbox*` 保留原名；旧数据缺该字段按 `x3_velocity`。`cf_blackbox`（或 meta `renderer_backend=gsplat`）走 3DGS 评测器 `sim.3dgs.eval_3dgs`。`forest_e1` 数据集没有 `task_version`，自动录像推不出任务会跳过；手动评测页选 `forest_e1` 会自动补 `--density`、`--dynamics cf_sitl --vehicle e1_mini`。

任务无法确定时跳过并写告警。录像评测命令为 `eval_closed_loop.py --policy student --record-video --episodes 1`，输出到 `<run>/closed_loop_eval_<UTCstamp>/`。视频合成依赖 `cv2`（`.venv-tools` 已安装 `opencv-python-headless`）；缺 cv2 时评测指标仍会保存，只是单集出现 `video_error`。评测失败只写入 `studio_summary.json` 的 `notes` 和训练日志，不会把训练状态改成失败。完成后 `studio_summary.json` 记录 `video_dir` / `videos`，`experiment_report.md` 追加视频相对链接。手动「运行正式闭环评测」的行为不变（仍会询问并读取界面参数）。

也可以不打开桌面程序，直接用 CLI 包装训练并录像：

```bash
python3 tools/train_student_with_video.py --data <dataset.uavr> --out <run_dir> \
    [uavrgb_train 参数...] [--dry-run]
```

`--dry-run` 只打印训练/评测命令，不执行。

## 三视角仿真

「仿真 Simulation」页面提供三路独立的 Gazebo 渲染相机流（各自有更新率和 header stamp，**不是同步采样的同一时刻**），并额外提供一张二维轨迹分析地图：

1. **第三人称跟随视角**：订阅 `/uav/chase_camera`。这是新增的虚拟追踪相机，刚体固定在 X3 的 `base_link` 后上方（相对位姿 `(-2.5, 0, 1.2 m)`），随无人机运动和姿态一起倾斜，不是稳定云台；不参与学生模型输入。
2. **第一人称 RGB**：订阅原有 `/uav/camera`，原始输出 320×240 RGB；学生模型实际输入为预处理后的 160×120。
3. **真实顶视相机**：订阅 `/uav/top_camera`。这是固定在世界坐标 `(2.5, 0, 6 m)` 的 Gazebo camera sensor，输出 480×360 渲染图像；80° HFOV 在 z=0 地面覆盖约 10.1×7.6 m。更大地图需要调整 SDF 中的相机高度、中心点或 FOV；它只用于观察整个场景，不参与学生模型输入。
4. **二维轨迹分析地图（补充）**：订阅 `/model/<vehicle>/pose`（载具按所选 world 推断：X3 世界 `x3`，E1 世界 `e1_mini`），并从当前 SDF 的 collision 几何提取障碍物俯视轮廓，绘制无人机航迹、朝向和可用的 waypoint。

以上相机位姿与分辨率是 X3 世界 `sim/worlds/uav_rgb.sdf` 的。E1 世界 `sim/worlds/uav_rgb_e1.sdf`（`scene_gen.py --vehicle e1_mini` / `forest_e1`）不同：顶视相机在 `(3.25, 0, 7 m)`、640×360；追踪相机是独立的 `chase_cam` 模型，由 `CfSitlSystem` 插件按位置 + 偏航跟随（不随姿态倾斜）；`--no-aux-cameras` 生成的世界没有这两路。

操作：

- 可先在「快速生成场景」中选择 `base/l1/campus/forest_e1` 与 seed（`forest_e1` 还要选 Density，生成的是 E1 世界）点击「生成 SDF + metadata」；生成成功后会自动填入对应 world 和 scene metadata；也可以手动选择已有文件；
- 点击「启动 Gazebo GUI」，或先在其他终端启动 Gazebo 后点击「连接当前仿真」；
- 如连接外部默认分区 Gazebo，`GZ_PARTITION` 留空；如连接自定义分区，填入相同值。由 Studio 启动 Gazebo 时，留空会自动生成一个隔离分区并回填到输入框；
- 旧的 world SDF 不包含新增 chase/top cameras。需要用更新后的 `sim/worlds/uav_rgb.sdf` 重新生成/启动场景，`/uav/chase_camera` 和 `/uav/top_camera` 才会出现；Studio 启动 Gazebo 时会把 `sim/worlds` 加入 `GZ_SIM_RESOURCE_PATH`；
- 如果连接的是外部已运行的 Gazebo，Studio 无法从 Transport 反查其磁盘 world 文件，因此用户需要确保当前仿真确实由所选 SDF 启动。

第三人称画面是刚体固定在 `base_link`、随姿态倾斜的 chase sensor，不是额外的物理无人机，也不是云台稳定画面。真实顶视相机是固定在世界坐标中的静态 Gazebo sensor；二维地图则是基于碰撞几何的辅助投影。三路 RGB 来自 Gazebo 的三路独立渲染流，不是同步帧，也不使用预录视频。

### 学生策略实时演示

选择 `ckpt_best.bin` 和对应场景的 `scene metadata JSON`，点击「在当前仿真运行学生策略」。程序调用现有 `sim/tools/collect.py --policy student`，通过真实学生推理进程控制当前 Gazebo 无人机；界面同步显示动作、第一人称 RGB、追踪相机、顶视相机和轨迹地图。

单次 live demo 的 `live_demo_summary.json` 只记录该 episode 的步数、动作分布、碰撞和 manifest。它是单集演示记录，不是正式导航 SR。

### 正式闭环评测

「实验 Experiments」页可以启动 `sim/tools/eval_closed_loop.py`，默认评测 best checkpoint，使用独立 headless Gazebo 分区，结果写入该 run 下的 `closed_loop_eval_<timestamp>/eval.json`。完成后会把 success rate、`success_criterion`、`requested_success_rate`、`completion_rate`、`failed_count`、碰撞率、stuck、false-forward、episode 数和 unique geometry 追加到 `studio_summary.json` 与 `experiment_report.md`。主卡显示的是**该判据下的成功率**，不同 `task`/`success_criterion` 之间不可直接比较。

评测 Task 必须与训练任务/数据协议匹配；不要为了得到更高 SR 随意切换 task、split、window 语义或 appearance。闭环评测可能持续较长时间并占用 GPU。

## 可靠性边界

- 训练目录由 C++ `RunOwner` 独占，Studio 的 job/log/lifecycle 资料放在同级 `.monitor/`，避免破坏 trainer 的空目录约束。
- 关闭桌面程序时，如果任务仍运行，会询问是否发送 SIGINT；选择不停止时，子进程继续运行，日志仍落盘。
- Studio 只停止自己启动的 Gazebo 进程，不会按进程名清理其他用户的 Gazebo。
- 相机流需要桌面图形会话、Gazebo Sensors/Ogre2 和 `gz topic` 可用。没有运行 Gazebo 时，视角显示等待状态，不显示伪造视频。
- 如果相机没有帧，分别执行 `gz topic -i -t /uav/camera`、`gz topic -i -t /uav/chase_camera`、`gz topic -i -t /uav/top_camera` 检查话题；旧 world SDF 需要按新模板重新生成。三路相机和 pose 必须处于同一个 `GZ_PARTITION`。
- chase/top 两个新增 sensor 已完成 SDF/XML 静态检查；真实相机渲染和 GPU 帧率仍需在目标 Ubuntu + Gazebo 8 桌面环境实测。
