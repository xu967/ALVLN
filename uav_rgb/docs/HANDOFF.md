# uav_rgb 交接文档

> **2026-10-29 本轮接手复核**：下方保留 01:10 的历史记录，当前入口见
> `../README.md` 与 `worklog/2026-10-29-1145-project-review.md`。
> 现已通过 13/13 GPU CTest；常驻 publisher、8–9 根密障碍场景和配置落盘已实现。
> 新修复 venv 搬迁路径、读端场景/样本区间校验、MaxPool 小输入越界、验证指标权重及
> split 标记、独立采集运行分区冲突。新增 setup/doctor/check 工具。
> 真实 Gazebo **5 帧采集 → 预处理 → GPU 训练 → checkpoint → resume** 已实测通过，
> 但仅单场景、两个动作类、没有 val，评测事件明确为 `split=train`，不能算泛化成绩。
> **P0 几何泄漏已处理**（2026-10-29，见 `worklog/2026-10-29-1820-geometry-split-manifest.md`）：
> 新增 `sim/tools/split_manifest.py`，split 由几何组 ID 的固定 salt hash 决定，采集端已接线。
> 仍遗留：场景族只有 1 个（`family_holdout=false`），几何去重不等于泛化。
> 传感器停更处理、学生 SimulatorEval、NPU/ESP32 合同仍未完成。旧 89.2% FORWARD
> 和 12 实例吞吐均为历史数据，不能作为当前密场景验收结论。

- 更新时间：2026-10-29 01:10
- 项目：单目 RGB 无人机避障（4 离散动作），自研 C++/CUDA
- **硬约束：禁止 PyTorch / TensorRT / cuDNN / cuBLAS / ROS 2**
- 设计 spec：`../TinyAerialVLN_Final_V8/references/reviews/UBUNTU_BASE_GUIDE_V1.md` + `../TinyAerialVLN_Final_V8/docs/`
- 本机：RTX 2080 Ti（sm_75，22 GiB）、CUDA 12.8、Gazebo Harmonic 8.15、Ubuntu 24.04、56 核 / 503 GB

---

## 〇、接手前必读

1. **已建立 Git**（2026-10-29）。本地仓库在 `uav_rgb/.git`；远端是
   `git@github.com:xu967/ALVLN.git` 的 **`uav_rgb/` 子目录**（deploy key 只能访问这一个仓库）。
   推送方式：`code/ALVLN`（完整克隆，**不要删**，重新克隆约 12 分钟）里替换 `uav_rgb/` 后
   commit + push。**本仓库有并发会话在写，提交前逐文件确认归属，不要 `git add -A`。**
2. **`ui/` 归另一个会话维护**。改 UI 前先确认，否则会覆盖对方工作。本文档只覆盖 `src/`、`sim/`、`tools/`、`docs/`。
3. **`sim/tools/collect.py` 可能被并发修改**。动它之前先看 `mtime` 和是否有 agent 在跑。
4. 所有任务结束必须写 `docs/worklog/YYYY-MM-DD-HHMM-<slug>.md`，含「做了什么 / 没做什么 / 验证到哪一步 / 风险遗留」。

---

## 一、当前状态总览

| 层 | 状态 | 位置 |
| --- | --- | --- |
| 环境（GPU/CUDA/Gazebo/OpenCV/Eigen/Catch2） | ✅ 可用 | `env.sh`、`docs/环境说明.md` |
| 接口冻结 | **6/8**（缺 `NPUModelIO`；`TeacherPlanner` 只有实现没有头文件契约） | `src/include/uavrgb/` |
| 数值算子（Linear/Conv/softmax-CE/ReLU/MaxPool/Adam/梯度裁剪/**加权 CE**） | ✅ CPU + CUDA，12/12 ctest 全绿 | `src/` |
| CNN 模型 + 训练器 + checkpoint + resume | ✅ 参数量 100,388，过拟合 100% | `src/model.cpp`、`src/trainer.cpp` |
| **类别权重 + 碰撞过滤** | ✅ 少数类召回 0 → 0.5（合成不平衡集） | `src/class_weights.cpp`、`--drop-collision` |
| 数据预处理（320×240 → 160×120，UAVR 二进制） | ✅ 端到端自检通过 | `tools/preprocess.py` |
| E1 仿真世界（四旋翼 + RGB + 深度 + contact） | ✅ 实测出图 | `sim/worlds/uav_rgb.sdf` |
| E2 专家策略（深度 → 4 动作） | ✅ 22 项自测；**有几个已知缺陷，见 §五** | `sim/tools/teacher.py` |
| E3 采集器 | ✅ 有效轨迹 + 碰撞真值 + RGB/动作因果正确 | `sim/tools/collect.py` |
| 场景随机化 + 并行编排 | ✅ 随机 8–9 根车道式障碍；**可用上限 8 实例**（12 实例起飞全超时） | `sim/tools/scene_gen.py`、`collect_parallel.py` |
| **数据类别分布** | ✅ FORWARD **80.0%** / LEFT 6.7% / RIGHT 13.3%（非 FORWARD 20%） | 见 §五 |
| E6 SimulatorEval（闭环评测） | ❌ 未开始 | — |
| 训练监控台（原生 Tkinter 程序） | ✅ 可监视；**启动按钮与 CLI 不兼容** | `ui/`（另一会话） |
| 硬件合同（荔枝派型号 / ESP32 协议） | ❌ 信息缺失，阻塞部署 | — |

### 最近一批改动（还在进行中）

- ✅ **常驻订阅**：读 I/O 1.75 s → **0.016 s**；配对残差 232 ms → 0 ms
- ✅ **常驻 publisher**：`gz topic -p` 每次固定 0.96 s（Gazebo CLI 硬编码 sleep）→ 常驻 `gz.transport13` 后 **0.003 s**；
  每步墙钟 1.518 → **0.520 s**（实测 `total=0.520 I/O=0.016 动作=0.501 发布=0.003`）
- ✅ **场景密度**：8–9 根车道式障碍，非 FORWARD 10.8% → **20.0%**
- ✅ **类别权重 + 碰撞过滤**：加权 CE（CPU+CUDA）+ `--drop-collision`，少数类召回 0 → 0.5

---

## 二、已冻结的合同（改之前必须想清楚）

### 2.1 四动作协议 `src/include/uavrgb/action.hpp`

```cpp
enum class Action : std::uint8_t {
  SAFE_NEUTRAL = 0, FORWARD_SLOW = 1, YAW_LEFT = 2, YAW_RIGHT = 3
};
inline constexpr int kNumActions = 4;
```

**安全边界（不可违反）**：模型只输出这 4 个离散动作，绝不输出 PWM/转速/姿态角；ESP32 飞控保留最终否决权。

### 2.2 学生输入 `src/include/uavrgb/observation.hpp`

```cpp
inline constexpr int kObsW = 160, kObsH = 120, kObsC = 3;
struct SimObservation {
  std::uint8_t rgb[160*120*3];  // 只有 RGB，HWC，未归一化
  std::int64_t timestamp_ns;
  std::int64_t episode_id;
};
```

**数据隔离铁律**：深度 / 位姿 / 障碍几何 / 碰撞真值 = 特权信息，只能进专家和 oracle 文件，**永远不进学生输入**。

**分辨率决策（偏离 spec，需最终验证）**：spec 原写 128×128，但 320×240→128×128 是把 4:3 横向压扁到 1:1
（水平角分辨率 0.469°/px，10m 外 0.4m 立柱仅 4.9px）。现用 **160×120**：精确一半、保持 4:3、像素只多 17%、
角分辨率 0.375°/px（同条件 6.1px）。**模型超参数**，改它必须同步改 `tools/preprocess.py --size` 和 `src/model.cpp`。
最终尺寸应由「128 / 160 / 224 同数据 A/B + 闭环 SR」实测决定。

**布局差异**：数据集是 **NCHW**，`SimObservation` 是 **HWC**，由训练器/部署适配层负责转换。漏了会得到「形状对但画面错乱」的静默错误。

**细障碍（2cm 线缆/树枝）在任何分辨率下都是亚像素**（160×120 下 5m 外 0.61px），是单目 RGB 的物理极限。

### 2.3 Episode 字段 `src/include/uavrgb/episode.hpp`

`EpisodeStep{timestamp_ns, action_expert, action_executed, collision}`；
`EpisodeMeta{episode_id, scene_id, seed, version}`。

**划分铁律**：train/val/test 按 `scene_id` 隔离，禁止同一场景随机拆帧。已实测强制拒绝（exit=1 且不留产物）。

### 2.4 事件/指标 `src/include/uavrgb/event.hpp`

`MetricsEvent`（train_step / validation / status / error）+ `ResultsRow{epoch, train_ce, dev_ce, dev_accuracy}`。
**训练器写出的 `results.csv` / `metrics.jsonl` 字段必须与此一致**，否则 `ui/monitor.py` 读不出来。

### 2.5 数据集格式 `tools/DATASET_FORMAT.md`

UAVR 版本 1，小端：48 B 头（magic `UAVR` / version / header_size / N / H / W / C / dtype / offsets）
→ N×C×H×W uint8 NCHW → N 个 uint8 标签 → 紧凑 JSON meta（含 split_policy=scene_id、逐 episode 的 sample_start）。

---

## 三、怎么跑（复制即用）

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
source env.sh

# 1) 核心库 + 全部测试（当前 9 个，约 95 s）
cmake -S src -B src/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build src/build
ctest --test-dir src/build --output-on-failure

# 2) 必须只有 sm_75
cuobjdump -lelf src/build/libuavrgb_core.a | grep -oE 'sm_[0-9]+' | sort -u

# 3) 专家自测
python3 sim/tools/teacher.py

# 4) 单实例采集（纯 headless 会 segfault，必须 DISPLAY=:20）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg gz sim -s -r sim/worlds/uav_rgb.sdf &
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect.py --steps 20 --hold 1.0 --out /tmp/ep0

# 5) 并行采集（推荐，自动生成随机场景 + 起停实例）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg \
  python3 sim/tools/collect_parallel.py --instances 6 --episodes-per-instance 1 \
  --steps 20 --hold 1.0 --base-seed 0 --out /tmp/dataset

# 6) 预处理成训练集
python3 tools/preprocess.py --out /tmp/ds /tmp/dataset/train/*/        # 默认 160×120

# 7) 训练
./src/build/uavrgb_train --data /tmp/ds/dataset.uavr --epochs 30 --batch 128 --lr 1e-3 --out /tmp/run1

# 8) 监控（原生窗口）
python3 ui/monitor.py --run-dir /tmp/run1
```

---

## 四、关键环境事实（踩过的坑，别再踩）

1. **纯 headless 相机渲染会 segfault**。容器无 NVIDIA EGL，Ogre2 初始化 `/dev/dri/card1` 失败后崩溃。
   必须 `DISPLAY=:20`（软件 GL / llvmpipe）。GUI 模式 `gz sim -r` 同样崩，只用 `-s`。
2. **sm_75 必须在 `project()` 之前 FORCE**，否则 CMake 3.28 用 nvcc 默认值 52。绝不能编 sm_120。
3. **`pkill -f 'gz sim'` 会杀掉你自己** —— shell 命令行里含同样的字符串，正则自匹配。用 `setsid` + `kill -- -PGID`；
   残留 `ruby` 用 `ps -eo pid,comm --no-headers | awk '$2=="ruby"{print $1}'` 找 PID 再 kill。
4. **多实例必须用 `GZ_PARTITION` 隔离**，否则多个 server 互相污染话题（实测过 3 个 server 串台，读出 y=41 这种不可能位姿）。
5. **深度图底边照到地面**（~0.13m）。专家若整列取最小深度会永远判「前方被挡」，只取中间水平带（行 [0.4h,0.6h)）。
6. **`/model/x3/pose` 是 `gz.msgs.Pose_V`**，JSON 是 `{"pose":[{position,...}]}`，不是 `msg["position"]`。
7. **`/uav/camera_info` 有两个 publisher**（RGB 和 depth），必须按 `header.data` 的 `frame_id` 区分。
   实测内参：`fx=fy=277.128, cx=160, cy=120, 畸变全 0`（与 `(w/2)/tan(hfov/2)` 理论值完全一致）。
8. **`gz topic -e` 不带 `-n` 时按行持续输出 JSON**，一帧一行 —— 这是常驻订阅可行性的依据。
9. **Fuel 模型下载极慢**（~15KB/s），世界里的四旋翼是 X3 等价 inline 体（物理参数一致，网格换成 box/cylinder）。
10. **一步的实际时长约 1.7 s，其中动作窗口只有 hold 秒** —— `gz topic -e` 子进程开销主导。这也是并行能线性扩展的原因。
11. **`Tensor::zeros` 和 `Tensor(dims)` 在本仓库完全等价**（都做零初始化），没有「不初始化分配」的接口。
12. **CUDA 算子每次调用临时 `cudaMalloc/cudaFree` + 全量同步**，是 parity 测试的 shortcut，不是生产实现。

---

## 五、已知问题（按严重性）

### 5.1 数据类别严重不平衡（**当前头号阻塞**）

**已修**。障碍从 2–5 根稀疏改成 **8–9 根车道式蛇形**（`LANE_SEP=1.302`，缝 0.902 m）后，实测 6 条 episode / 120 步：
**FORWARD 80.0% / RIGHT 13.3% / LEFT 6.7%**（非 FORWARD **20%**，目标 ≥15%），碰撞步 1/120。

配套：训练侧类别权重（`--class-weights auto`，少数类召回 0 → 0.5）+ 碰撞过滤（`--drop-collision`）。

**仍存在的隐患**：6 条 episode 的动作序列**高度规整**（都是「转 2 步 → 直飞 4 步 → 转 2 步 → 直飞」），
因为障碍落在固定车道 grid 上（x 只有 9 个取值）。模型可能记住节奏而非学会看画面。
若闭环评测发现泛化差，要把车道布局改回随机分布。

**加权只解决「学得到」，不解决「样本够」**：LEFT 类只占 6.7%，仍偏少。

### 5.2 专家会撞墙，碰撞时的标签不可信

实测 6 条里 1 条碰撞 5 步（左右摆振后卡死）。**专家撞墙时给出的动作标签本身就是错的**，
混进训练集会教模型学错。修法：`--drop-collision` 过滤或降权。

### 5.3 专家策略的三个设计缺陷

1. 单个近处障碍会同时污染左/中/右三条带 → 曾导致「三面堵死」永久卡死（已用两级阈值 `turn_dist=1.5 / emergency=0.8` 绕过，治标）
2. 转向后常常一路飞走，缺乏「贴障穿行」的困难样本
3. 三分带太粗，没有利用机体宽度和间隙几何

研究线给出的改造方案（R1 列级深度剖面 / R2 机体宽度分级安全距离 / R3 最小转向评分，约 +110–130 行）
**故意排在数据采集之后** —— 否则同一批数据里专家规则变过，无法对比改造效果。

### 5.4 训练吞吐受限于 host 往返

实测 59–62 样本/秒，且**对 batch 大小几乎不敏感** —— 瓶颈是每层固定的 `cudaMalloc + 同步 + pageable 拷贝`。
单层 32→64、batch 256 前向 991.6 ms 中：host 清零 599 ms（60%）、拷贝 220 ms（22%）、**kernel 只有 167 ms（17%）**。
真解是 device-resident 引擎（权重常驻、层间不落 host，预计 3×），属单独一轮工作。

### 5.5 采集吞吐（正在修）

**单实例 0.52 s/步**（hold=0.5）。但**并行有硬上限**：

| 实例数 | 结果 |
| --- | --- |
| 1 / 3 / 6 | RTF 稳定 **0.97**，全部成功 |
| 8 | **8/8 成功**，3.03 步/s（hold=1.0） |
| **12** | **12/12 起飞超时**（30 s 内 z 只到 1.09，目标 2.1）——软件 GL 渲染把 CPU 打满，RTF 崩溃 |

**实际采集能力**：8 实例 ≈ 3.0 步/s（hold=1.0）。
2000 episode × 20 步 = 40,000 步 → **约 3.7 小时**；200 episode → 约 22 分钟。

**并行铁律**：同一时间只能有一个采集任务在跑。实测两个任务各跑 6 个实例（合计 12）时 RTF 掉到 0.15–0.28，
导致起飞超时和 `stuck` 误报碰撞。

### 5.6 其他

- 监控台的「启动训练」按钮会追加 `--config`，而 `uavrgb_train` 不接受 → 能监视、不能按钮启动。
- `CLAUDE.md` 的测试数量描述会随改动过期（以 `ctest -N` 为准）。
- 相机外参、sim-time 对齐、episode reset 均未做。

---

## 六、没做的（按优先级）

### P0
1. **修好类别分布**（进行中）—— 不解决就没法训练
2. **第一次用真实数据训练** —— 训练器至今只在合成数据上验证过
3. **端到端跑通**：采集 → 预处理 → 训练 → 出 checkpoint

### P1
4. **SimulatorEval 闭环评测**：无碰撞完成率 / 碰撞率 / 误前向率 / 推理 p50·p95（离线准确率不算数）
5. **device-resident 训练引擎**（预计 3× 吞吐）
6. **`teacher.py` R1–R3 改造** + 与基线数据对比

### P2
7. **NPUModelIO 接口冻结**（spec §3 最后一个）
8. **硬件合同**：荔枝派准确型号、ESP32 四动作→UART 帧协议（**必须等用户提供**）
9. **安全监督器契约**：显式优先级（飞手急停 > 指令 stale > 位姿 stale > 深度/RGB stale > 策略异常 > 电池 > 碰撞）+ 三级降级
   阈值按 30 Hz 传感器 3 帧≈100 ms 标定，**不能抄 PX4 的 0.5 s**

### P3 工程
10. `-Wall -Wextra` + `CMAKE_EXPORT_COMPILE_COMMANDS`（**目前的「0 warning」是在默认警告级别下测的，基本没意义**）
11. `src/.clang-format`（自写 Google 基底 + 100 列；照搬参考仓库的 Mozilla/80 列会产生 2960 处改动）
12. `tools/check.sh` 一键构建+测试；Catch2 3.4.0 已装，新测试增量迁移
13. 生效配置落盘到 `<out_dir>/config.json`

---

## 七、借鉴参考实现的结论（浙大高飞团队 9 个仓库）

参考代码在 `../idea/repos/`。**只读思想，不要抄代码**：

| 仓库 | 许可证 | 结论 |
| --- | --- | --- |
| GCOPTER | **MIT** | 唯一可抄代码的，但我们不需要 |
| ego-planner / EGO-Planner-v2 / Elastic-Tracker / Fast-Racing / Implicit-SDF-Planner / Swarm-Formation / ego-planner-swarm | GPL-3.0 | **传染性**，抄了项目就得 GPL |
| Fast-Drone-250 | **无 LICENSE**，README 明写禁止商用转载 | 风险最高，只看思想 |

**三条最有价值的结论**：
1. 这个 `ego-planner` 快照其实是 **ESDF-free** 的（README 明写），别被名字误导；真正用 ESDF 的只有 Implicit-SDF-Planner
2. 参考仓库的**工程实践比我们落后**（9/9 ROS1 catkin、0/9 CI、0/9 现代 CMake），没有可抄的
3. 值得借鉴的只有：列级深度剖面找可通行区间、机体宽度/间隙判据、常驻订阅 + sim-time 配对、安全降级优先级

详细报告：`docs/worklog/2026-10-29-0037~0038-ref-*.md`

---

## 八、代码规则（doc 11）

- C++ 单文件 ≤280 行，CUDA 单文件 ≤360 行（超了拆文件，不许压注释）
- 接口按顺序冻结：SimObservation → TeacherPlanner → ActionAdapter → DatasetWriter → NativeTrainer → SimulatorEval → EventSchema → NPUModelIO
- 每个新数值算子必须有：CPU FP32 参考 + FP64 有限差分 gradcheck + CPU/CUDA 对拍
- 不许假结果：**FOUND ≠ VERIFIED**，必须真跑真报数
- 不许把「至少一条不是全 F」这种能糊弄过去的指标当验收标准 —— 要用**分布/比例**类指标

---

## 九、下一步

```
修类别分布 ──→ 采 200 episode ──→ 预处理 ──→ 训练 ──→ 看是否学得动
                    │                                    │
                    └── 达标后采 2000 episode ───────────┘
                                                         ↓
                                            闭环评测（SR / 碰撞率 / 时延）
                                                         ↓
                                            改专家 R1–R3 → 采第二批 → 对比
```

关键决策点：**先用 200 episode（约半小时训练）验证链路，再决定要不要花 1–3 小时采 2000 条**。
不要一步到位采完才发现模型学不动。

---

## 十、交接更新 2026-10-29 22:00

### 现在的真实状态

| 项 | 事实 |
| --- | --- |
| CTest | **18/18 通过**（RTX 2080 Ti，全新构建目录，零编译警告）。本文档 §一 里的 "12/12" 已过期 |
| 专家策略 | **换了**。`collect_parallel` 会给每集传 `--scene-meta`，走 `teacher.py:plan_action`（柱子真值 carrot planner），不再是 §五 描述的深度反应式 |
| 专家通过率 | seed 0..7 × 80 步：**5/8 穿过 x=5.0**（第 68–76 步），失败 3 条都撞柱后无法倒车 |
| 采集默认步数 | `--steps` 默认 **80**（原 20）。实测每 FORWARD 步只前进 0.108 m，20 步只有约 2 m |
| 训练入口保护 | `--lr` 非有限负值拒绝；`--class-weights` 空项/全 0 拒绝；非空 `--out` 拒绝（`RunOwner` 已接入） |
| 数据可追溯 | episode manifest 新增 `teacher` / `hold_s` / `scene_meta_sha256` |
| 工程规范 | 见 [`ENGINEERING_RULES.md`](ENGINEERING_RULES.md)（动手前必读） |

### 本轮改了什么（按影响排序）

1. **专家换成了 GT carrot planner**：`sim/tools/teacher.py` 新增 `ray_clear` /
   `plan_action`；`collect.py` 新增 `--scene-meta`。真值只决定标签，不进学生输入。
2. **修掉一批「跑完还报成功」的缺陷**：`--lr` 负数做梯度上升、`--class-weights`
   空 token 静默变 0 权重、同一 `--out` 连跑两次产生 `1,2,1,2` 的 results.csv。
3. **`nn_cuda.cu` 从 `std::exit` 改为抛异常**：训练中途显存出错不再直接杀进程跳过快照。
4. **碰撞/半截数据不再静默进训练集**：`preprocess.py` 拒收 `error` 非空的 episode、
   拒绝 `collision` 非 bool、行数与 `num_steps` 不符报错；`dataset.cpp` 在没开
   `--drop-collision` 时打印碰撞步占比。
5. **补测试**：`src/tests/test_eval_metrics.cpp`（宏指标，原先零覆盖）；
   `tools/check.sh` 增加 preprocess→read_dataset 的格式契约链路。

全部证据见 `docs/worklog/2026-10-29-2103-audit-fixes.md`、
`docs/experiments/2026-10-29-teacher-traverse-ablations.md`。

### 下一任最该先做的两件事

0. ~~**加 best-checkpoint**~~ **已完成（2026-10-29 21:55）**：`Trainer::evaluate` 拆到
   `src/trainer_eval.cpp`，每 epoch 按 dev 宏 F1 保存 `ckpt_best.bin`；
   best 值写入 checkpoint v3（v2 仍可读）。详见本轮 worklog。
1. **决定车道间距要不要放宽**。缝 0.902 m，机体外接直径 0.711 m。实测把净空从
   0.36 提到 0.40 会把穿越率从 5/8 打到 2/8——**继续调参数没用，要改几何**。
2. **实现 §19.3 仿真时间动作窗口**。`--hold` 是墙钟秒，同 seed 两次采集终点
   x 分别是 1.40 / 2.36，数据集目前不可复现。

### 不要重复踩的坑

- `--scene-meta` 不给会**静默**退回深度专家（现在有 manifest 的 `teacher` 字段 +
  stderr 警告，但仍然会降级）。
- 采集必须 `DISPLAY=:20`；纯 headless 相机会 segfault。
- 提交前逐文件确认归属，不要 `git add -A`；CMakeLists 与新测试文件必须同一次提交，
  否则干净 clone 在 configure 阶段就失败。
- 推送用 `../ALVLN` 完整克隆，**不要用 partial clone**（会卡死在 `git-upload-pack`）。
