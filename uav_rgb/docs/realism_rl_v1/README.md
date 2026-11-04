# Gazebo 写实资产 + 动力学 Teacher / RL 实验包（2026-11-04）

> **实验代码，非已验证的论文结果。** 本包以 `code_code_only_20261104 (2).zip` 为基础，尽量保持 `base/l1/campus`、历史训练器、旧 Teacher 默认配置不变。新能力为**显式 opt-in**，不要把新旧训练集、成功率和控制周期混报。

## 0. 先读：已经完成什么、没有完成什么

| 功能 | 状态 | 具体含义 |
|---|---|---|
| `--task campus_assets` | **已实现，离线单测通过** | 多目标颜色-方向指令、网格渲染、独立碰撞几何、稳定 map 分组与 split |
| 模型网格导入工具 | **已实现** | 允许人工验证 CC0 许可后导入 `.obj/.dae` 与纹理，保留来源与 SHA-256 |
| ZIP 内真实第三方扫描资产 | **未包含** | 当前环境无外网，不能下载 Poly Haven/Fuel；只有明确标为 *offline demo* 的原创示例网格 |
| T1 动力学预测 Teacher | **已实现算法与采集接入，未实机验证** | `cf_stab` 近似状态、A* 测地距离场、四动作 1.2 秒预测 |
| T2 PPO Teacher | **已实现训练和离线评测脚本，未训练模型** | 使用简化 2.5D 动力学环境，Gymnasium + SB3 可选依赖；无预训练权重 |
| 真实 Gazebo 闭环评测 | **未运行** | 本环境没有可用 `gz sim`；用户在 Ubuntu 环境运行后才能报告成功率 |
| 相机真机标定/真机飞行 | **未完成** | 不得宣称 Crazyflie 真机可行或 Sim-to-Real 已实现 |
| 历史数据 step/gap 逻辑 | **已修正，单测通过** | 修复 filtered row 索引覆盖源 step、跨坏帧伪时序拼接 |

## 1. 结构变更

```
uav_rgb/
  sim/tools/
    asset_campus.py           # campus.assets.v1 实景模型+程序化地图
    import_asset.py           # 登记用户验证许可的本地 CC0 网格
    register_ground_texture.py # 登记本地 CC0 地面纹理
    build_demo_assets.py      # 生成原创离线演示网格和纹理（不是实景模型）
    teacher_predictive.py     # T1: 短时域动力学预测的离散专家
    test_asset_campus.py      # 离线 160 seed 一致性、重现性、数据分组与碰撞检查
    scene_gen.py              # 新增 --task campus_assets，默认任务保持不变
    split_manifest.py         # 新 map_id 按整个几何组分配 split
    collect.py                # 新的可选专家入口；默认 A* 不变
    collect_parallel.py       # 批量生成 + 传新 Teacher 参数 + 导入本地资产资源路径
    eval_closed_loop.py       # 可选新场景、新 Teacher 闭环对比
  sim/assets/
    asset_catalog.json        # 本地模型类型、尺寸、来源/哈希
    uavrgb_assets/            # GZ_SIM_RESOURCE_PATH 的 model:// 包
      meshes/
      materials/
  rl_teacher/
    core.py                   # 简化动力学/占据几何/测地距离场/PBRS/奖励
    gym_env.py                # Gymnasium 接口
    reward_v1.json            # 奖励权重版本
    train_ppo.py              # PPO 训练命令
    evaluate.py               # val/test 复现指标
    test_core.py              # 纯 Python 回归测试
  teacher/dataset.py          # 保留动作原始 step；不跨 rgb 丢帧组 GRU 时序
  teacher/test_dataset_step_gap.py
  docs/realism_rl_v1/README.md
```

原 `code/ALVLN`、`code/CF-Drone` 与历史数据/文档保留，用于独立追溯。**不要**直接把新 Teacher 产生的标签混入旧 `uavrgb.teacher.v5` 的审计结果；没有训练新模型之前，旧 Student 也不会自动获得改进。

## 2. 第一步：检验可复现场景（无 Gazebo 也能执行）

从 `code/uav_rgb` 目录：

```bash
python3 sim/tools/build_demo_assets.py
python3 sim/tools/scene_gen.py --task campus_assets --seed 0 --out /tmp/asset_demo.sdf
python3 sim/tools/test_asset_campus.py
python3 -m rl_teacher.test_core
```

**资源路径：** 使用资源 URI `model://uavrgb_assets/...`；需要把 `sim/assets` 作为模型目录加入 `GZ_SIM_RESOURCE_PATH`，不要把开发机绝对路径写死到 SDF。

```bash
export GZ_SIM_RESOURCE_PATH="$(pwd)/sim/assets${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim -r /tmp/asset_demo.sdf
```

部分 Gazebo 环境下在 GUI 启动前需配置 `DISPLAY`/EGL/渲染设备，参照原 `sim/README.md`。在所有实验中记录 OGRE2 的实际渲染后端，而不是只记录用户选择了“gpu”。

### 2.1 真正的路线 B：替换为真实 3D 资产

本包提供离线示范网格以确保在无网络时能够生成合法 SDF；它们不是 Poly Haven 真实模型。要获得真正的高质量效果，请从拥有合法授权的来源获取 `.obj/.dae` 资产并在 Blender 中预处理：

1. **场景尺寸：** 单位米、Z 向上、模型中心 XY=0、底面 Z=0；旋转和模型 scale 应当应用变换；不要在网格内部保留随意的偏移。
2. **网格简化：** 输出轻量的视觉 LOD；避免直接导入超高面数的 3D 扫描模型，控制大型场景并行采样开销。
3. **纹理：** 导出 UV 与 albedo PNG/JPG。普通 OBJ 如果引用 `.mtl` 的外部文件，先在 Blender 烘焙/整理后再导出；导入器会拒绝带外部 mtllib 的 OBJ，避免静默缺贴图。
4. **许可：** 逐项确认 CC0 才可加入本 ZIP 的可分发资产目录；记录来源、作者或官网、下载日期与 SHA-256。可参考 Poly Haven、ambientCG 的 CC0 资源；**Gazebo Fuel 的第三方资产并非全部 CC0**。

示例：假设 `tree.obj` 几何真实尺寸是 2.0×2.0×4.0 米。

```bash
python3 sim/tools/import_asset.py \
  --type tree --mesh /path/to/tree.obj \
  --dimensions 2.0 2.0 4.0 --source "https://原资源地址" --license CC0 \
  --albedo /path/to/tree_albedo.png

python3 sim/tools/register_ground_texture.py \
  --image /path/to/pavement.png --source "https://原资源地址" --license CC0

# 重新生成场景与 scene.json；不能只换网格却沿用老 world_sha256。
python3 sim/tools/scene_gen.py --task campus_assets --seed 0 --out /tmp/asset_demo.sdf
```

**警告：** `build_demo_assets.py` 会重新生成同名示例资产。真实资产导入后不要再运行此脚本，否则会覆盖已经登记的示范文件，应该在代码管理中冻结版本。

### 2.2 视觉与物理几何契约

`asset_campus.layout()` 返回一个**统一几何源**，`pillars` 同时供碰撞模型与 Teacher 使用：

- 视觉：mesh + PBR albedo。树冠示例在 z≈2.8m 以上、巡航高度 z≈2.1m，树干碰撞 box 代表当前固定高度通道；高枝没有建立完整碰撞模型，**因此本场景只用于定高与演示**。
- 物理：每个障碍物独立 SDF collision box。不要把高多边形 mesh 当成默认物理碰撞体；真实树枝、路灯横杆等如果进入飞行空间，必须另外增加碰撞代理和规划几何。
- Teacher：仍使用这些 box 作为 A* 障碍；不能仅增加视觉物件而忘记物理与规划地图。
- 场景唯一性：每 8 个 seed 为同一地图几何，颜色/指令等变体进入同一 split；split 在 map_key hash 上固定，不按 seed 位置切分。

**这不是物理场景“自动重建”：** 资产导入器只更新外观，不尝试根据任意复杂网格自动推断完美的安全通道。真实数据进来后需要在 Gazebo 验证碰撞体、传感器深度和规划障碍的一致性。

### 2.3 指令任务不再仅靠左右词

每个地图同时有左右各两个目标柱，同一侧两个不同颜色。`target_name`、`target_color`、`relation`、`waypoint` 明确写入 scene.json。仅根据 left/right 不再足以区分目标，必须结合颜色；数据集的实际语言因果性仍需通过改变目标/指令的反事实实验验证。

## 3. 第二步：T0/T1 动力学专家基线

**T0：** 保留已有 `plan_action_global()` 作为 `--teacher-planner astar`，不改历史算法版本。

**T1：** 新 `teacher_predictive.PredictiveTeacher` 使用场景 A* 距离场，读取 `cf_stab` 的当前世界系水平速度，分别预测 SAFE/FORWARD/LEFT/RIGHT 持续 1.2 秒后位置与净空，通过预测进展、转向、碰撞风险和速度代价选择动作。它仍是**近似控制规划器，不保证安全，也不是 MPC 的严格最优解**。新策略和原 `cf_stab` 都使用参数：2° 倾角、阻力系数 1/s、控制延迟 0.15s、偏航速度 0.6rad/s；这些参数还未校准真机。

用同一新地图、同一 `cf_stab`、同一 `--hold 0.2` 比较专家：

```bash
python3 sim/tools/eval_closed_loop.py --task campus_assets --policy teacher \
  --teacher-planner astar --dynamics cf_stab --hold 0.2 \
  --split test --episodes 20 --out /tmp/t0_asset

python3 sim/tools/eval_closed_loop.py --task campus_assets --policy teacher \
  --teacher-planner predictive --dynamics cf_stab --hold 0.2 \
  --split test --episodes 20 --out /tmp/t1_asset
```

**注意：** 这里的 T1 仅通过离线公式单测，未在真实 Gazebo 跑过；不能宣称 T1 优于 T0。对比时必须固定 seed 清单、相机、起点/目标、渲染后端、物理档和动作窗口；不能把旧 X3 的成功率混入。

并行采集范例（首次使用可以 `--instances 1 --episodes-per-instance 1` 冒烟）：

```bash
python3 sim/tools/collect_parallel.py --task campus_assets \
  --policy teacher --teacher-planner predictive \
  --dynamics cf_stab --hold 0.2 --steps 160 \
  --instances 1 --episodes-per-instance 1 --base-seed 0 \
  --keep-worlds --out /tmp/t1_collection
```

`collect_parallel.py` 会把本地资产目录注入 `GZ_SIM_RESOURCE_PATH`，因此它无需在工作目录下载网格。

## 4. 第三步：T2 特权 PPO Teacher（可选依赖）

环境是一个**离线简化物理平台**，与 Gazebo 的 sensor/world simulation 完全分离。先训练，再在 Gazebo 上做独立验证：

```bash
# 在虚拟环境中安装；严格按本地环境兼容版本管理
python3 -m pip install gymnasium stable-baselines3 numpy

# 从 code/uav_rgb 根目录运行，使用 --seed 确定可复现的随机源
python3 -m rl_teacher.train_ppo --steps 100000 --n-envs 4 \
  --seed 1234 --reward-config rl_teacher/reward_v1.json \
  --out runs/teacher_ppo

# SB3 model.save 输出 runs/teacher_ppo.zip
python3 -m rl_teacher.evaluate --checkpoint runs/teacher_ppo.zip \
  --split val --episodes 20 --out runs/teacher_val.json

# 仅在确定训练和调参完成之后执行未见测试，防止测试集调参泄漏
python3 -m rl_teacher.evaluate --checkpoint runs/teacher_ppo.zip \
  --split test --episodes 20 --out runs/teacher_test.json

# 再去真实 Gazebo；即使离线成功，也不等于 Gazebo 成功
python3 sim/tools/eval_closed_loop.py --task campus_assets --policy teacher-rl \
  --rl-checkpoint runs/teacher_ppo.zip --dynamics cf_stab --hold 0.2 \
  --split test --episodes 20 --out /tmp/t2_gazebo
```

PPO 使用 13 维特权观测：机体系目标相对向量(2)、机体系速度(2)、航向 sin/cos(2)、四方向安全净空探针(4)、目标距离(1)、上一生效动作(1)、剩余延迟(1)。输出 0..3 四动作，**和旧 Student 的动作契约一致**。

真实速度由训练内的动力学状态直接提供；Gazebo 使用 `cf_stab` 模型的速度估算。此接口存在 sim-sim 差异，必须对相同动作序列的轨迹进行定量对照。**不能**把特权速度/净空直接作为 RGB Student 的推理输入；Student 的可观测性须另行研究。

## 5. 奖励函数：哪些是理论塑形，哪些是有意加入的任务偏好？

代码在 `rl_teacher/core.py`，权重在 `reward_v1.json`。给定目标地形测地距离场 `D(s,g)`，势函数 `Phi(s)=-alpha*D(s,g)`，折扣 `gamma=0.99`，基础形式：

```text
r = task + [ gamma * Phi(next) - Phi(current) ]
         - time_penalty * dt
         - risk_weight * max(0, safety_distance - future_min_clearance)^2
         - action_switch_weight * 1[action != previous]
```

- **PBRS**：距离场 D 是从目标格反向 Dijkstra 求得，8 邻域且禁止斜穿障碍角；有限回合终止态令 `Phi(terminal)=0`。该项用来解决绕墙时欧氏距离暂时变大的问题。奖励中 `gamma` 必须与 PPO 训练折扣一致。
- **真正成功**：位于 0.5m 目标半径内、水平速度 `<0.12m/s`、连续保持 `0.6s`、无碰撞；不是“曾经过目标半径”。
- **安全项**：基于已走过的动作窗口及继续执行当前动作 0.8 秒的预测最小净空，保留现有动作延迟和阻力假设。风险惩罚**不具有 PBRS 的策略不变性保证**。
- **碰撞**：中心到 box 的距离减去无人机半径（0.356m）`<=0` 判碰撞。真实 X3 机体/四桨盘外形与 2.5D 球圆近似并不完全相同，下一阶段必须做相同状态下的碰撞判据对照。
- **超时**：每集最多 250 个 0.2 秒动作，总计 50s 仿真时间；`truncated=True`，不能算成功。

建议**先固定权重**，观察策略是否学到“停住/绕远/贴墙/不断转向”等坏模式，再依次消融 risk、PBRS、control，不要盲目改十几个权重。可在训练前编辑 `reward_v1.json`；生成的 `.meta.json` 保存完整奖励配置，评测复用。

**重点风险**：四动作中的 `SAFE_NEUTRAL` 只是去掉俯仰输入，依赖阻力自然滑停，不是主动反向制动。即使设置了稳定停靠奖励，也不能保证所有进站速度下可刹停。若 T1/T2 均在终点失败，应先测控物理，而不是无限加大成功奖励。

## 6. 数据集与可信度

历史 `teacher/dataset.py` 在剔除 `rgb_ok=False` 行后，用过滤后索引生成样本 `step`；本包改为保留**原始 step**，并跳过跨越丢帧的滑窗。新增 `teacher/test_dataset_step_gap.py` 回归验证。旧源文件已经改变，因此需要将其作为**新实现修复版本**记录，不应使用该版本重新解释历史训练复现实验而不标注代码哈希。

数据采集阶段应保留每条轨迹是否碰撞、退出原因、目标距离、速度、执行端延迟与渲染版本。碰撞失败的轨迹不适合直接作为行为克隆的成功示范，但可以单独保留用于故障诊断与 DAgger 回溯；新包**没有自动完成 DAgger**，不能宣称已经完成纠偏学习。

**禁止的结论**：`T2 success 90% in light dynamics` => `真实 Gazebo 90%`；`Gazebo 更真实` => `Student 必然更稳`；`新资产场景 T0 教师能飞` => `Student 理解颜色指令`。

## 7. 本地测试与验收顺序

```bash
cd code/uav_rgb
python3 sim/tools/test_asset_campus.py
python3 -m rl_teacher.test_core
python3 teacher/test_dataset_windows.py
python3 teacher/test_dataset_step_gap.py
python3 sim/tools/test_campus.py
python3 sim/tools/test_collect.py
python3 sim/tools/test_eval_closed_loop_args.py
python3 sim/tools/test_collect_parallel.py

# 原项目的 C++/CUDA 工具链和 CTest 按 README 单独测试，
# 本次容器未配置 Gazebo、Jetson、CUDA GPU，不能以纯 Python 单测代替这些验证。
```

注：teacher/test_dataset_windows.py 与 teacher/test_dataset_step_gap.py 依赖 torch；
系统 python3 的 torch 不可用时（如缺 libcudnn.so.9），改用
$DATA_ROOT/uav_rgb/teacher-venv/bin/python 执行，否则会直接 ImportError。

**Gazebo 环境必须额外人工核查：**

1. SDF 引用的 mesh/albedo 在日志中确实加载成功，画面没有“隐形模型/黑色模型”。
2. 视角变化时纹理贴图、地面尺度、阴影方向合理；记录 GPU 推理/渲染时间。
3. 相同地图的视觉、深度、碰撞与 A* 地图一致，特别是树冠、建筑屋檐和目标前安全缓冲。
4. 同样 seed 在不同机器的 SDF/meta 确定性；world SHA-256、资源 SHA-256、相机配置及软件版本全部记录。
5. 用原版 `--task campus` 跑一次旧任务回归；确认新任务并未污染历史数据。
6. 至少 20 个独立地图（不是同一地图 20 种颜色）的 T0/T1/T2 配对测试；报告目标最终速度、最小净空、碰撞率、完成率及置信区间。

## 8. 已知限制与后续待办

- 模型库未包含真实第三方下载资产；只有离线原创示例 + 经过用户许可核实的导入路径。
- 当前模型网格和简化碰撞代理只是新实验场景的试点，并不代表真实摄影测量。
- 未建立照片级相机曝光/镜头畸变/运动模糊模型；这些需要后续相机实拍与参数标定。
- 仍是二维/定高路径规划；复杂冠层、上下穿越和动态障碍需要 3D 占据/轨迹规划。
- T1 是单动作恒定短时预测，不是完全的多步 MPC，也无安全可行性证明。
- RL 模型未包含训练好的权重。训练必须在用户可用的 Python 环境执行。
- 与 Gazebo 的控制延迟、加速度、阻力、接触判据可能不一致；真机性能**完全未验证**。
- `collect.py --policy teacher-rl` 与 `--teacher-planner predictive` 是实验开关，历史 label_recompute 只认可旧 Teacher；若要作为最终训练集，应另行实现按版本校准的标签审计与示范筛选。
- 当前 Student 仍是四类动作/有限指令模型；这套代码没有假装已解决时序可观测性、完整 DAgger、连续动作控制及真机部署。

## 9. 调研参考（可公开验证的外部文献与说明）

- Gazebo Harmonic 模型资源目录： https://gazebosim.org/docs/harmonic/fuel_insert/ 及 https://github.com/gazebosim/docs/blob/master/harmonic/sdf_worlds.md
- Poly Haven（下载时逐项验证 CC0）：https://polyhaven.com/models ；公开 API：https://github.com/Poly-Haven/Public-API
- ambientCG CC0 纹理： https://ambientcg.com/
- Stable-Baselines3 自定义环境： https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
- ZJU FAST Lab / EGO-Planner-Swarm： https://github.com/ZJU-FAST-Lab/ego-planner-swarm
- ZJU FAST Lab / GCOPTER： https://github.com/ZJU-FAST-Lab/GCOPTER
- Ng 等, *Policy Invariance Under Reward Transformations* (ICML 1999)：势函数奖励塑形；需注意有限回合边界条件。

以上外部来源用于技术方案论证，不代表它们的代码或资产被偷偷复制进本包。

### 动力学环境边界补充

`rl_teacher/core.py` 将状态限制在测试区域 `x ∈ [-2,12]m, y ∈ [-6,6]m`，越界属于任务失败并终止、执行与碰撞同级的惩罚。这是**离线训练防止逃逸**的抽象边界，不是真实 Gazebo 的物理墙；做 Gazebo 对照时必须统一任务区域定义。
