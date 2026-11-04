# 校园导航 pilot（campus.v1）

这是可验证的定高四动作基线：保留 Gazebo X3 动力学、自研 C++/CUDA 学生，
加入建筑通道、树木道路、广场三类低模场景及自有程序化材质。它不是照片级数字孪生。

## 任务和划分

- 指令示例：`go to the red pillar on your left`。左右以出生朝向为基准，两个候选柱同色。
- 目标区是彩柱前 1.1 m 处、半径 0.5 m 的水平圆盘；已有彩柱和地面圆盘标识。
  相机能观察彩柱，用已知地标尺度定位目标邻域；**不能声称地面圆盘边界在全部视角可见**。
  审计检查出生/目标中心的彩柱视野、遮挡、出生及整个目标区的机体净空和 A* 可达性。
- `visible_goal_reach_v1`：全程无 contact，轨迹采样曾进入目标区；不要求停稳。
  这不是“pass … and stop”，也不与旧 L1 的 1.5 m 成绩比较。
- 默认动作窗口为 `dispatch`，每次执行完整仿真时间窗口；旧 base/l1 默认 `observation` 不变。
- `seed // 8` 决定几何样本，余数决定四颜色×左右；`map_id` 按**拓扑族**分组。
  building_lane=train、tree_road=val、plaza=test，尺寸微扰与语言/光照变体不能跨组。
  这是很小的跨族 pilot，仅三个组；不代表完整校园泛化数据集。
- 所有实体障碍均有碰撞体并进入教师 `pillars`。教师已读取每个长方体的实际尺寸。
  地面标识只有视觉，地面已有碰撞平面。学生仍只接收 RGB 和文本。

## 使用

```bash
source env.sh
bash tools/check.sh

# 生成单场景，不启动仿真。
"$UAV_RGB_PYTHON" sim/tools/scene_gen.py --task campus --seed 16 \
  --out "$DATA_ROOT/uav_rgb/campus-scene-16.sdf"

# 三族各一对左右任务，保留世界、帧、轨迹、失败和 summary.json；输出目录必须新建。
"$UAV_RGB_PYTHON" sim/tools/smoke_campus.py \
  --out "$DATA_ROOT/uav_rgb/campus-pilot-new"
"$UAV_RGB_PYTHON" sim/tools/audit_dataset.py \
  --dataset "$DATA_ROOT/uav_rgb/campus-pilot-new/episodes"

# 正式采集接口；每组包含全部八个颜色/方向变体。
"$UAV_RGB_PYTHON" sim/tools/collect_parallel.py --task campus \
  --instances 1 --seeds 0:24 --steps 80 --keep-worlds \
  --out "$DATA_ROOT/uav_rgb/campus-collection-new"

# 新任务须用对应数据训练权重。keep/blank/shuffle 用同一组 --seeds；输出目录各自唯一。
"$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py --task campus --policy student \
  --ckpt /path/to/campus/ckpt_best.bin --split test --seeds 16:24 \
  --text-mode keep --keep-work --out "$DATA_ROOT/uav_rgb/campus-eval-new"
```

`--task campus` 固定使用校园材质/随机光照。自 2026-11-04 起 base/l1 的
`--appearance` 默认是 `v1`（删除模板里 3 个固定纯色地标、随机化光照/天空）；
`--appearance legacy` 才逐字节复现带固定地标的旧场景。旧 legacy 数据与新 v1 数据
不能混包（`appearance_version` 不同）。
运行时副本在 `sim/assets/campus_textures`，SDF 写 `model://campus_textures/...`；
`GZ_SIM_RESOURCE_PATH` 指向 `sim/assets`，不再写开发机绝对路径。
纹理由 `make_campus_textures.py` 确定性生成（当前仍输出到 `sim/worlds/campus_textures`，
需要同步复制到运行时目录），无需网络或用户照片。
模板哈希覆盖相机和动力学，材质哈希与各集 world 哈希随产物落盘。
外观、几何、任务、动作窗口、相机/动力学版本不一致的 episode 拒绝混包。

## 工程修复与兼容性

- 接触事件在订阅线程锁存，跨步骤传递游标；推理期间、零速后、步间碰撞不会因缓冲淘汰丢失。
  起飞地面接触单独排除，不以“无新消息”推断碰撞传感器健康。
- 默认训练/推理/检查统一到 `build/check`，或显式 `UAV_RGB_BUILD_DIR`；评测保留程序和权重哈希。
- 学生进程每次通信默认 30 s 超时，包含写入、完整行接收和 warmup；超时清理子进程。
- 软标签缺失、重复、非有限值、样本序不符直接失败。旧裸数组不再接受，必须从原 NPZ
  重导出至新路径；格式见 `../tools/SOFT_TARGETS_FORMAT.md`。旧文件不自动覆盖。
- 报告同时给 completed-only SR、请求集成功比例、完成比例、失败和跳过明细。
  自动 seed 扫描的请求集比例是运行完成性诊断；正式配对比较须使用事先冻结的有效 seed 清单。
- `DeviceTrainState` 和 `BatchReader` 保留为实验组件；没有删除其测试，也没有把未接入代码宣传成优化。

## 后续实验边界

先用成对数据完成学生训练与 keep/blank/shuffle，再扩展更多独立地图。
成像噪声、畸变、风、控制延迟和机体参数需逐因素试验，校准范围依赖目标相机/真机参数；
本版本不把任意噪声数值写成“真实参数”。真实照片不是当前运行前提。
若转向特定园区重建，再采多视角重叠影像与尺度；照片/3DGS 本身不保证可靠碰撞几何。

2026-11-03 已将默认启动改为 NVIDIA EGL，真实相机确认 RTX 2080 Ti；显式 software
模式保留 llvmpipe 对照。详见 [GPU 仿真实验](experiments/2026-11-03-gpu-rendering.md)。
教师小样本成功、单元测试通过、图片更清晰均不能代替学生未见地图闭环评测。
