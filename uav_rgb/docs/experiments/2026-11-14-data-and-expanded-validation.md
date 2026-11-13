# 数据集尝试与扩展验证（2026-11-14）

## 数据集准备

当前数据卷有约 243 GB 可用；`/root/rivermind-fs` 是 200 GB 卷，约 165 GB 可用。目标目录为 `/root/rivermind-data/uav_datasets/`。

- **SAGE-3D VLN 标注**：公开卡片标注 CC-BY-4.0，数据页约 9.82 GB，含语言指令与轨迹，可用于语言任务分析；`huggingface-cli download spatialverse/SAGE-3D_VLN_Data` 本轮未完成，目录无产物。[数据卡](https://huggingface.co/datasets/spatialverse/SAGE-3D_VLN_Data)
- **MatrixCity aerial_street_fusion**：本轮尝试按 `aerial_street_fusion/*` 下载未成功；单个 `street.tar` 约 620 MB，许可 CC-BY-NC-4.0，仅在接受非商业限制后用于实验。[street.tar 与许可](https://huggingface.co/datasets/BoDai/MatrixCity/blob/main/aerial_street_fusion/street.tar)
- **AirVLN**：官方 README 说明 AerialVLN/AerialVLN-S 指令标注各小于 100 MB，模拟器约 35 GB。Kaggle API 本轮返回 404，官方 S3 超时，GitHub clone 未能取得数据。先不要下 35 GB 模拟器；小标注可达且许可确认后再评估其语言导航基准价值。[官方仓库与下载说明](https://github.com/AirVLN/AirVLN)

当前已经有约 7 GB 的 3DGS 场景以及一份 Wanderland 场景。优先在现有场景上补齐路线、相机视角覆盖和 3DGS 质量测试；等网络恢复后再补公开数据，避免数据重复。已确认的冷备份 ZIP 已迁移到 `/root/rivermind-fs/archive/rivermind-data-archive-20261114/`，恢复信息在对应 `*.move-manifest.json` 中；仍被文档引用的 `archive/idea` 保留原位。

## 测试结果

### 3DGS 回归

命令：`python -m unittest sim.3dgs.test_*`（`3dgs-venv`，真实 GPU 环境）。

- 368 项通过，2 项 skip，用时 145.7 s。
- 覆盖 VLN 指标/STOP、渲染、碰撞、动力学、教师、视频和学生输入合同。
- SAGE-3D 教师 2 集真实 GPU smoke 已通过：SR=1、SPL=0.8742、NE mean=0.327 m、无碰撞。属于 A* 特权教师结果，不能当作语言策略结果。
- 抽查输出视频：指令横幅、目标、top occupancy/path、位姿和 SUCCESS 停止状态可见；onboard/chase 视角存在运动模糊/重影，建议单独比较运动期间相机采样与渲染帧质量。

### CPU quota 敏感用例

命令：`bash tools/check.sh --python-only`，并单独复跑 `tools.test_exp_lib.Arms.test_health_line_comes_from_the_start_and_end_snapshots`。

- Python/工具/3DGS 测试整体通过；一个资源敏感断言失败。并行时均值 2.36 核，独立复测 1.68 核，仍未达到测试的 2.5–3.5 核区间；日志显示 CPU quota throttling。保留阈值，记录为环境敏感失败。

### Wanderland 12.83 m SITL 路线

命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.collect_3dgs \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/wanderland_1MlUucug/scene.json \
  --out /root/rivermind-data/uav_rgb/tmp/wanderland-long-20261114/route0 \
  --route 0 --episodes 1 --policy astar --dynamics cf_sitl --hold 0.5 \
  --seed 20261114 --stop-on-collision
```

实际结果：87/126 步，`completed=true`，`terminal_reason=legacy_goal_stop`，无碰撞；起点至目标 12.141 m，轨迹最近距离 0.356 m，末端距离 0.473 m，最小净空 0.831 m。渲染 87 帧，低质量帧 0，p50 10.93 ms、p95 14.83 ms。动作数量（action id）：SAFE 3、FORWARD 79、左转 3、右转 2。

同一场景又跑了 route 1（15.24 m，99/147 步）和 route 2（12.8 m，88/129 步）：两条均无碰撞、`legacy_goal_stop`，末端误差分别 0.388 m 和 0.507 m，最低净空分别 1.023 m 和 0.736 m，低质量帧均为 0。三条运行均为 A* 特权教师，`vln=null`，且 SITL 参数 `calibrated=false`，所以只是当前 3DGS/LiDAR 代理下的几何与固件在环冒烟，不是语言理解、完整长航线泛化或真机验证。

## 后续测试优先级

1. 在采集器与 VLN 评测之间加 STOP 语义一致性回归：训练标签必须来自任务终止意图，不能把连续 SAFE/安全等待直接当语言任务完成。
2. 在 Wanderland 多条路线、正反向与多个 seed 上做教师对照，报告路线长度、终点误差、碰撞、净空、停顿、动作切换、渲染帧质量和 3DGS 相机覆盖；不要只跑开阔的 route 0。
3. 将视频抽查中看到的运动模糊转成量化对照：相邻帧清晰度、相机快门/曝光假设、动作窗口前后帧，区分原始渲染模糊与视频合成/播放引入的拖影。
4. 有语言地面器与目标条件学生后，再跑错指令、空指令、反事实地标、STOP 提前/过晚和 unseen 场景闭环；没有这些接口前，不做“语言提升”结论。
