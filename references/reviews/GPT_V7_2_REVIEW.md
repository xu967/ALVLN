# TinyAerialVLN V7.1 新一轮审核：GPT 独立复核与 V7.2 行动方案

> 日期：2026-10-28  
> 文档类型：**复核报告及下一阶段执行决策**，不是已实现模型或实验结果。  
> 审查输入：4 份 V7.1 独立评审、1 份 CF-Drone 源码专项审计、`review_v2(1).patch`、`review_v2_repro(1).py`、V7.1 原工程 ZIP、CF-Drone 原 ZIP、V7.1 主方案及 CODE_RULES。  
> **固件冻结**：本次没有修改、编译、烧录 CF-Drone ESP32 固件；所有应用补丁都在隔离的 V7.1 工程副本中。  
> 证据级别：`SRC`=本次直接核对源码；`TEST`=本次实际运行；`EXT`=官方公开来源；`REVIEW`=仅评审声称；`PROPOSAL`=建议/未验证。  
> 主原则：评审建议不等于事实；文件存在不等于功能验收；通过基础设施测试不等于自主导航成功。

## 0. 执行摘要：本轮先修什么

**建议暂不进入 V8 或继续扩写 1,000 行新方案；下一轮应是 V7.2 以可运行的研究最小闭环为中心。**

- P0-SCI：ECPT 缺定义、没有可证伪的独立机制；必须先写出数学定义与同容量对照，否则删除方法名称。
- P0-DATA：目前 SDF 障碍物布局、固定起终点及可见目标使任务可能被“直行+计步”解决；禁止用此场景训练成绩论证视觉/语言导航。
- P0-EXEC：没有真正的学生 C++ 训练源文件、Gazebo 四旋翼 RGB 闭环与板端实测。停止扩大 UI 功能，先实现最小动力学/相机/动作 loop 和可训练的学生基线。
- P0-SAFETY：CF-Drone 现有 MAVLink/CLI 高权限接口、可到达的未实现模式与陀螺积分姿态问题必须列入实体测试硬阻断；**现在只记录并隔离，暂不改固件**。
- P1-ENG：评审补丁的 18 项基础设施修正具有实测依据，但未解决场景退化、JSONL schema 漂移和硬件验证。应经过代码审查后合并，不得作为“导航系统完成”的依据。

## 1. 实测证据与方法

### 1.1 采用的文件

本次直接检查了以下已挂载原始文件：

- `/mnt/data/TinyAerialVLN_V7_1_工程与可视化UI.zip`：V7.1 原工程（49 个归档条目）。
- `/mnt/data/CF-Drone.zip`：CF-Drone 源码，含 `.git`；执行 `git rev-parse --short HEAD` 得到 `e7f507f`。
- `/mnt/data/review_v2_repro(1).py`：Python 复现脚本（查看其行为后运行）。
- `/mnt/data/review_v2(1).patch`：统一 diff，修改 8 个既有文件并新增 1 个回归测试文件。
- 此轮用户上传的评审报告 MD 文件，以及 V7.1 研究方案/规则文档。

没有直接访问 RTX 5090、LicheeRV Nano、ESP32、真实无人机，也没有运行 Gazebo，因为当前执行环境缺少 `gz`。不声称任何飞行、模型训练或 NPU 速度成绩。

### 1.2 复核命令与原始结果

```bash
# 对 ZIP 解压出的原工程副本赋予 run.sh 执行位后：
python -m unittest discover -s code/tests -p 'test_*.py'
# Ran 13 tests: OK
python -m unittest discover -s sim/tests -p 'test_*.py'
# Ran 2 tests: OK

python 'review_v2_repro(1).py' '/path/to/original/project'
# 汇总：存在 21，已修复 0，跳过 0

# 把补丁应用到完全独立的第二份工程副本：
patch -p1 --dry-run -i 'review_v2(1).patch'
patch -p1 -i 'review_v2(1).patch'
python -m unittest discover -s code/tests -p 'test_*.py'
# Ran 23 tests: OK (expected failures=1)
python -m unittest discover -s sim/tests -p 'test_*.py'
# Ran 4 tests: OK
python 'review_v2_repro(1).py' '/path/to/patched/project'
# 汇总：存在 3，已修复 18，跳过 0
```

**重要限制**：原补丁的 1 项 `expectedFailure` 不是问题已解决，恰恰标识了尚需修复的场景问题。脚本中的“已修复”仅表示其指定复现条件消失；不等于全面安全审计。

### 1.2.1 本次补做的电机预检反例（无需安装 Gazebo）

复现脚本的 B15 原始输出 `exit=127` 是因为本机没有 `gz`，**不能单凭这条输出就断言它真的接受了错误电机布局**。为消除该混杂因素，我另外在隔离临时目录构造一个只返回 0 的假 `gz` 命令，使程序运行到 SDF 静态检查分支，得到：

| 构造的 SDF | 原版 `sim smoke --dry-run` | 补丁版 | 解释 |
|---|---:|---:|---|
| 只有 `MulticopterVelocityControl`，没有电机模型 | **exit=0** | **exit=4** | 证实原子串匹配存在缺陷且补丁修复了这个复现点 |
| 四个 `MulticopterMotorModel`，但 actuator_number 全为 0 | 未另跑 | **exit=0** | **证实补丁仍无法发现电机编号重复** |

这是纯静态输入校验的独立反例，**没有启动 Gazebo 动力学**。后续还要加入 actuator ID 的 `0/1/2/3` 唯一性、转子旋向、关节/link 引用及模型动态状态检查；并以真正的 Gazebo 物理 trace 作为 GZ0 通过依据。


### 1.3 21 条复现结果归并

| 类别 | 具体问题 | 目前结论 |
|---|---|---|
| 场景 | 固定直通道、`mixed` 同 `forest` | **未修**，仍为 2 项 |
| JSONL 契约 | 文档 `kind/loss_total` 与代码 `event/loss` 不一致 | **未修**，仍为 1 项 |
| Dashboard | 嵌套 NaN、过深 JSON、坏 manifest、Host 头、IO 错误状态行、80 run 截断 | 补丁在复现脚本覆盖场景中有效 |
| Launcher | run_id 校验不统一、train/eval 同名冲突、缺命令、工具版本假阳性 | 补丁在复现脚本覆盖场景中有效 |
| C++ 日志 | locale 影响 JSON、篡改 cerr 状态、非有限数丢失错误事件 | 补丁在 C++ 实编测试中有效 |
| 测试可复现性 | fresh clone 隐藏夹具依赖、测试误启动训练器、忽略 build | 补丁在复现脚本覆盖场景中有效 |
| Gazebo 检查 | 仅靠文件名含 `multicopter` 就通过 | 补丁要求 4 个已知电机模型插件；仍不足以证明四旋翼布局正确 |

## 2. 对评审内容逐条下判定（避免照单全收）

| 意见 | 决定 | 理由及限制 |
|---|---|---|
| “ECPT 没有可计算定义” | **采纳（P0）** | 当前只有概念，不足以展开严格消融及论文方法节 |
| “目前模型的研究代码等于零” | **采纳（P0）** | 原工程 `code/src`、`code/cuda`、`sim/src` 确无实作源文件；UI/logging 不是训练器 |
| “现有场景可以直飞，不需要语言” | **采纳（P0）** | 复现脚本跨 preset、seed 计算出中央最小障碍净空 1.15 m；真实策略成绩尚未测 |
| “Flix 可省去大量 CF SIL 代码” | **有条件采纳** | Flix 官方确有原始 Arduino 代码参与的 Gazebo 仿真，但需核实相似性、Gazebo 版本及授权后再决定引用方式 |
| “必须先 AerialVLN-S 全链才允许做 Gazebo” | **不采纳为唯一顺序** | 大型旧模拟器可能与目标 5090/Ubuntu 栈冲突，且用户要求小数据；应有 **限时 1–2 天探针**，Gazebo 小闭环独立推进 |
| “物理轨只能二选一，不可能同时做 CF SIL” | **过度绝对** | 可用标准四旋翼控制器验证相机/动力学，再将 CF 控制律作为不同层级的替换/对照；但 CF 不得阻塞第一条闭环 |
| “扩展板有气压计，补几个驱动就能定高” | **不接受为事实** | `board_config.h` 规划不等于实物焊装、驱动、融合、控制器和安全降落都存在 |
| “SG2002 的 NPU 不缺算力，二次推理几乎免费” | **只接受 MAC 估算，不接受速度结论** | 官方标称 1 TOPS、共享 256 MB，但实际瓶颈与 p95 必须板端测；一次/两次前向都要真实测 |
| “LAG 7.2% 是 Val-Unseen 的目标线” | **事实错误** | ICCV 2023 AerialVLN-S 原表 LAG **Val-Seen 7.2%、Val-Unseen 5.1%**；且官方模态与拟议纯 RGB 学生并不严格相同 |
| “每组 531 条一定检测不了 3pp/5pp” | **过度绝对** | 功效取决于策略配对分歧率、基线成功率、场景内相关与 seed 方差；不能用通用阈值取代预实验估计 |
| “只要采用独立 STOP head 就有创新” | **拒绝** | L2Stop（2020）、X-Fly（2026）等已有相似分离终止机制；需额外机制与对照 |
| “LightVLN 与 AirAnchor 完全覆盖了本研究” | **部分接受** | 覆盖历史/局部聚合/进度的许多技术要素；但 0.5B/MLLM 与 1M 级 NPU 预算是不同约束，不可推定同构或新颖 |
| “UI 只读与空状态没有价值” | **不接受这种贬低** | 可靠报告接口有工程价值；但现在应冻结新增 UI，先修契约漏洞并让研究生产真实指标 |

**注意两个外部基准数字：** ICCV 2023 的 AerialVLN-S S8(LAG) Val-Seen SR=7.2%、Val-Unseen SR=5.1%；官方 LAG 与纯 RGB、超轻学生不属于完全匹配设置，应用相同输入重新跑 baseline。来源：ICCV 2023 AerialVLN 原文及项目官方表格。

## 3. 最紧迫的安全复核：CF-Drone（本次源码直接验证）

以下结论由用户提供的 `CF-Drone.zip`、commit `e7f507f` 中直接核得，**不是单靠第三方报告**。

### 3.1 高权限 MAVLink 消息未见有效命令权限隔离

- `mavlink.ino:163–171` 的 `SERIAL_CONTROL` 分支把收到的数据交给 `doCommand(data, true)`，没有按命令权限设置独立限制或明确飞行状态门禁。
- `mavlink.ino:173–207` 的另外两个目标接口在 `AUTO` 下可影响姿态或归一化电机值。不能把这两条接口当作安全导航 UART。
- `CF-Drone.ino:12` 将 `WIFI_ENABLED` 绑定到 `BOARD_WIFI_ENABLED`；`board_config.h` 中 C3 默认禁用 Wi-Fi，ESP32/S3 分支默认启用。`wifi.ino` 中确实存在公开的默认 AP 凭据；其危险可达性取决于实物配置和是否已修改密码。

**立即执行的非固件措施：** 实体机停止无线高权限接入测试；隔离飞控测试网络、核实实际编译宏与 AP 配置；禁止带桨状态下测试不可信控制命令。后续固件阶段须对远程 shell、任意电机命令、解锁与参数写入分权限，添加上锁状态校验、合法数据范围、超时保护和审计日志。

### 3.2 ALTHOLD 可被选择，但缺失实现分支

- `control.ino:43` 直接注释悬停推力是 stub；`control.ino:50–53` 定义 ALTHOLD=3 与可配置模式列表。
- `interpretControls()` 可从 `flightModes[]` 取模式；控制分支只处理 STAB、ACRO、RAW，另对 AUTO 早退。未处理 ALTHOLD 的情况下，部分控制目标可能保留旧值。
- WebRC 针对 ALTHOLD 的回退逻辑仅覆盖 WebRC 路径，并不能证明其他配置入口安全。

**验收要求：** 未实现模式不可进入有效飞行状态；以后改固件前，至少有模式合法性与故障恢复的无桨回归测试。任何“中立动作=悬停”假设均不得成立。

### 3.3 姿态积分与失效保护的局限

- `estimate.ino:51–56`：`applyAcc()` 只有 `landed` 时提供加速度校正，飞行中直接返回；`levelWeight=0`、`levelBiasGain=0`，默认飞行姿态实质上依赖陀螺积分。真实漂移量尚未测量，不接受评审中假设的具体“30 秒漂 10–20 度”为测量事实。
- `safety.ino` 的 `descend()` 使用未标定的 `ALTHOLD_HOVER_THRUST` 限制推力；它是开环推力下降，不是经过验证的定高降落。
- `CF-Drone.ino` 初始化调用 `disableBrownOut()`；在负载与电源尚未测量前，这还意味着供电跌落后的复位/异常行为需专门验证。

### 3.4 扩展板与 UART：配置不是实物

- `board_config.h` 对 BMP388、QMC5883L、VL53L1X、PMW3901 预留了配置与引脚；未见相应驱动及有效数据融合路径。**不能据此认定用户的板上装有这些传感器。**
- C3 的 RC 使用 `Serial1`；其他 ESP32/S3 分支使用 `Serial2`。实际 NAV 串口是否可用须由芯片、固件配置、引脚复用、接收机及 PCB 决定。
- 现在不修改 ESP32：优先交付芯片型号/PCB/传感器/UART 引脚资源记录，后续单独审批固件改动。

### 3.5 原始许可证与 Flix 引用边界

- 本次实际读取 `CF-Drone/LICENSE`：文件**同时包含 GNU GPL v3 全文及 2026-07-10 版本的额外商业限制和品牌归属条款**。因此称其为“纯 Apache-2.0”与原文件不符。
- **GPLv3 与附加商业限制的可执行性/兼容性存在法律问题；本项目不作法律结论。** 仅凭源代码可读取，不代表能够自由复制修改后向外再许可。
- Flix 的公开仓库证明它有 Gazebo/Arduino 仿真方案，但尚未用完整文件级 diff 证明 CF-Drone 与 Flix 存在何种派生关系；引用其控制思想或代码前须先核对来源、许可证、署名与差异。

## 4. 当前数据任务为何不合格，以及怎样最小修复

### 4.1 可复核的现象

1. 多个 preset 和种子下中央道路直接畅通；现有障碍物不强迫路径选择。
2. `forest` 与 `mixed` 进入同一随机分布逻辑，不能视为两个独立场景类别。
3. 起点、终点和地标过于固定，可能产生数据泄漏与盲策略高成功率。
4. 静态 SDF 只生成几何，不提供动态四旋翼、真实相机、控制执行及轨迹记录。
5. 缺少不同颜色/类别/相对顺序的语言地标和可解性检验，因此尚不能生成真正的多阶段 VLN 任务。

### 4.2 最小 S0–S2 课程：先验证语言确实有用

| 级别 | 场景与指令 | 测试重点 | 放行条件 |
|---|---|---|---|
| S0 | 两个相似但颜色不同的目标，随机目标位置 | 视觉目标选择 | 同图不同指令能让动作显著变化 |
| S1 | 必经转弯/遮挡的两个地标，目标隐藏于路口之后 | 指令顺序与空间参照 | 盲策略不能稳定成功；完整策略优于指令打乱 |
| S2 | 三阶段子目标、相似干扰地标、短时丢失地标 | 记忆与阶段切换、正确 STOP | 相比无记忆/无进度对照有可区分优势 |

**任务生成算法建议：** seed → 生成障碍与候选语义地标 → 基于机体安全余量做占用网格和 A*/几何连通检验 → 检查直线可达性/可见性 → 采样起点与目标 → 生成与路线一致的指令 → 重放专家路径 → 发布 scene/task/episode 三份不同合同。即使 A* 在 2D 上证明连通，也不等于真实四旋翼动力学可行；必须进行物理专家重放。

**严禁泄漏：** 完整地图、真值目标和隐藏目标坐标只存在仿真/teacher/eval 数据层，不能传入 student 在线输入；若视觉目标地标可见，则必须是任务里合法可观察的真实物体，而不是为了调试额外画出的真值球。

### 4.3 配对评测与反捷径测试

强制至少比较：全输入、空指令、乱指令、RGB 置零、历史记忆清空、只执行固定动作或固定时间 STOP、RGB-only GRU 基线。所有策略使用同一 `pair_id=(scene_id,task_id,start_state,render_seed)`；记录真实成功率、路径效率、碰撞、超时、提前终止、过站漏停、语言干预效应。不得把 episode 内的帧当独立样本；按独立场景/地标配置聚类，训练 seed 只是重复估计，不是独立场景数量。

## 5. ECPT 应先变成可计算候选，而不是论文结论

下面是 **PROPOSAL / 尚未实现** 的最小规格草案，仅供审核确定是否值得写代码。不要把草案描述为已有成果或已证实的算法创新。

### 5.1 研究假设

在相同参数量、相同数据、相同可用在线信息与固定延迟预算下，**以视觉证据门控的进度状态更新，同时控制动作和终止判断**，能否减少“仅凭历史累积就误判指令已完成”的错误？其可能的增益应集中在遮挡、回访、相似地标和已通过目标却未 STOP 的场景，不能只用全局 SR 证明。

### 5.2 输入/状态/输出（候选值，非已定工程规格）

- 在线输入：`rgb_t: uint8[B,3,160,160]`，`tokens: int[B,L]`（L 用完整 train 分布 P99 决定，不以 48 截断），`a_{t-1}: int[B]`；禁止深度、IMU、位姿、参考路线等学生外部输入。
- 视觉：`V_t: fp32[B,N,64]`，从静态 CNN 的 N 个区域 token 得到；N 要以模型可编译结果决定。
- 指令：按合法文本切分为 K 个 clauses，`E: fp32[B,K,64]`；初始只对受控模板指令做确定切分，不宣称可以可靠解析任意自然语言。
- 内部状态：运动/视觉记忆 `h_t∈R^64`；子指令进度 `q_t∈Δ^K`（单调前移的概率质量），当前证据 `e_t∈[0,1]^K`。重置 episode 时必须全清。
- 输出：官方离散基准的 7 个移动类 logits 与 1 个 STOP 候选（或保持官方 8 logits 以协议兼容），独立的 `p_stop`，以及仅用于训练/解释的 `q_t` 和 `e_t`；实体机由独立连续动作适配器处理，STOP 请求不是电机断电。

### 5.3 计算定义（初始可实现版本）

令当前进度聚合语言向量：

```math
r_t = \sum_{j=1}^{K}q_{t-1,j}E_j
```

令当前指令相关视觉证据为：

```math
z_t=\operatorname{Attn}(r_t,V_t,V_t),\quad
\eta_t=\sigma(w_e^T[z_t;r_t;h_{t-1}])
```

用本次证据门控推进指令状态，`Shift` 把概率质量往下一子句移一格，并保持最后子句吸收：

```math
q_t=(1-\eta_t)q_{t-1}+\eta_t\operatorname{Shift}(q_{t-1})
```

更新视觉历史、动作与终止预测：

```math
h_t=\operatorname{GRU}([z_t;r_t;\operatorname{Emb}(a_{t-1})],h_{t-1})
```

```math
p_{stop}=\sigma\bigl(w_s^T[h_t;z_t;q_{t,K};\eta_t]\bigr),\qquad
\pi_t=\operatorname{Softmax}(W_a[h_t;z_t;r_t])
```

**审稿风险主动声明：** `η_t` 同时担任“证据可靠”和“状态推进”未必足够，视觉证据可能是误匹配；单调推进假设不能正确表达回溯、重复路口或指令纠正。上式可能退化成普通进度 head + STOP 头，所以必须做因果干预实验；若无法实证其独立作用，删掉 ECPT 名称。

### 5.4 标签、损失与禁止偷换

- `L_action`：训练动作标签（BC 或 rollout 专家）交叉熵；记录是否专家态/学生态。
- `L_stop`：在正确到达判据下得到的终止标签，用 class weighting/BCE 与独立阈值验证。官方 20m 离散 STOP 标准不得偷换成真实物理悬停。
- `L_progress`：**默认关闭弱伪标签监督**；仅当场景中明确的地标事件或人工验证的子任务事件可信时才开启。教师注意力分布和未来路线索引不能直接冒充在线进度真值。
- `L_consistency`（可选）：同一 RGB 切换合法相反指令，或同一指令改变相关地标时，证据/动作输出应合理改变；具体约束必须不会鼓励无意义的大动作波动。
- “校准”如果只是独立 STOP head，不应使用该术语；必须在 held-out calibration split 上报告 ECE/Brier/reliability diagram 等，并严格冻结阈值后再测 unseen。

### 5.5 必须打败的不是大型 VLM，而是针对性对照

(1) 只保留普通 GRU；(2) GRU + 独立 STOP（L2Stop 风格）；(3) GRU + 进度辅助（Self-Monitoring 风格）；(4) 相同容量但无 `η_t` 证据门控；(5) ECPT 完整版；(6) 全部 INT8 版本。参数控制要求总参数量尽量对齐（建议 ±1% 目标，无法对齐则报告差值），包括文本 embedding、动作头和温度标定参数。若仅利用更大的 K 或多一次前向带来收益，不能归功于机制。

## 6. 物理轨如何真正闭环，而不让 CF-Drone 阻塞研究

**主路线选择：** 先独立仿真基准控制器 + Gazebo Harmonic 真正 RGB/动力学闭环（GZ0–GZ3），再在旁路完成 CF 数学核对拍/SIL（CF0），最终是否移植 CF 取决于真实板卡/电源/传感器及固件审批。基准控制器不是 CF 的飞控结果，论文不得混写。

- GZ0：固定插件版本的动态四旋翼；动作→姿态/位姿有非零物理响应；可复位；记录 100 次相同 seed 的数值偏差。
- GZ1：机载 RGB Camera，明确光学系与机体系变换、FOV、裁剪、曝光与实际相机；同步 `sim_time_ns`、图像帧序号；同世界不同机体方向应有可解释像素变化。
- GZ2：有界导航意图，例如水平前向速度 m/s、偏航角速度 rad/s、有效执行时长 ms（初版仅低速、定高的仿真控制）；每个字段确定单位、坐标系、失效规则。与 AerialVLN 5 m/15°离散步完全隔离。
- GZ3：场景任务、语言、RGB、专家、记录器、学生动作、连续运动和后续 RGB 形成 `observation→policy→action→physics→next observation` 闭环；加入碰撞与机体半径，拒绝以瞬移/忽略碰撞作弊。
- CF0（旁路）：在不改 `.ino` 的前提下，对原来的姿态 PID/mixer 做 Linux 数值对拍；若使用 Flix 代码，需要许可证与来源核对；不把原厂 demo 的通过当作 CF-Drone SIL 通过。
- CF1（后续）：实测本机硬件后再选择是否开发定高/速度闭环和 UART NAV 接口。当前不进入自由飞行。

**仿真时钟重点：** 物理积分、内环、相机曝光、导航决策和命令有效时长使用独立频率；所有线程通过 `sim_time_ns` 和明确的采样/控制保持规则对齐；timeout 不能让旧命令在下一次物理步继续无限执行。

## 7. 补丁接收建议：可以合，但还需这些审核

| 类别 | 补丁解决了什么 | 尚缺什么 |
|---|---|---|
| `dashboard.py` | 坏行容错、Host 校验、场景损坏状态、run 列表真实总数 | 事件合同版本、认证边界、UI 分页面错误隔离、完整日志告警 |
| `launcher.py` | ID 一致、评测目录、缺命令返回错误、版本探针 | 训练进程崩溃后的恢复语义、checkpoint 明确选择、配置实质校验 |
| `training_monitor.hpp` | 递归非有限数对应展示容错，C locale、流状态隔离、错误事件 | 多线程日志竞争、真实 loss/梯度/吞吐字段、resume 与 checkpoint 对齐 |
| `sim_preflight.py` | 能区分速度控制插件与四个电机模型 | **4 个不同 actuator 编号、旋向/关节/动态 model、include 模型解析**，更不能替代实际 Gazebo 物理验收 |
| `code/tests`/`sim/tests` | fresh clone 与伪造执行器问题部分修复 | 真正的 SDF 动力学、RGB 时间戳与训练数值测试 |
| 场景 | 基本没改 | 起终点、遮挡、语言地标、可解性、盲策略门禁 |

**推荐合并方式**：把这个第三方补丁作为单独 `fix/infrastructure-audit-v2` 审查分支；重新跑自有测试与对方复现，再以小 PR 合入。由于补丁改动范围达到 8 个文件，仍需逐文件代码审查；此处没有替用户修改主工程。

### 7.1 新发现的一个残留契约矛盾

补丁中 `sim_preflight.py` 的 `sim doctor` 若检测到 `gz` 可执行文件，仍打印 `[BLOCKED][GZ_PHYSICS_NOT_VERIFIED]` 而返回 `0`。这对“doctor 仅做依赖查询”可以成立，但若自动化把 `sim doctor` 的退出码当作 GZ0 PASS 就是严重语义错误。应将 `doctor` 明确限定为“依赖存在探针”，另设 `sim verify` 使用物理 trace 与相机 trace 作为通过条件。

## 8. 工程合同：先写 3 份，不再继续增加 UI 文档

1. `EPISODE_SCHEMA.md`：`schema_version`、`scene_id`、`task_id`、`pair_id`、`instruction`、合法 observation、`action_contract`、episode 终止原因、teacher/student 来源、时间戳、哈希、数据划分。与 `metrics.jsonl` 的事件合同分离但互相引用。
2. `ACTION_PROTOCOL.md`：区分 `AerialDiscreteAction`、`SimNavIntent`、未来 `NavFrame`，每个字段写单位/上下限/时间基准/状态转换/错误码；当前只定义，**不修改 ESP32**。
3. `ECPT_SPEC.md`：5.2–5.5 的精确定义、参数/MAC 计算脚本、数值正确性 oracle、两类失败反例、预注册消融。填不出来就不写 `ecpt.cc`。

`CODE_RULES` 暂停继续扩充大段政策，只保留强制项：可复现、清晰模块边界、无虚假状态、严格动作类型、无越界参数、无高权限调用、独立测试、单一实际配置来源。文件数量上限可以随真实必要的模块作有证据修订，不为了守 6×300 的数字挤出巨型难维护文件。

## 9. 下一阶段的两周“可验收任务”建议（不是已经完成，也不是承诺后台执行）

| 优先级 | 任务 | 验收物 | 不通过处理 |
|---|---|---|---|
| D0 | 审查并合并基础设施补丁 | 原/补丁 21 项对照和全新 clone 测试 | 不继续增 UI；先修真实失效 |
| D0 | 最小数据合同 | `EPISODE_SCHEMA`, `ACTION_PROTOCOL`, schema 单元测试 | 不开始采集大批样本 |
| D0 | 场景去捷径 | 多颜色地标、随机起终点、无直行可达子集、`mixed` 修正 | 不能宣布视觉/语言任务成立 |
| D1 | Gazebo GZ0–GZ1 | 电机→位姿 trace，第一视角 RGB 流和同步记录 | 返回动力学/相机接口；不训练模型 |
| D1 | CF 安全资料收集 | 实物 ESP32 型号、扩展板在否、UART 资源、固件配置、供电载荷记录 | 保持固件冻结、禁止实飞 |
| D1 | 官方 AerialVLN 探针 | 仅 1–2 天，能/不能在 RTX 5090 跑 10 条 episode，原生分辨率和吞吐 | 失败则暂缓 35GB 大基准，不阻塞轻量 Gazebo |
| D1 | 最小 C++ 学生训练器 | Linear/Conv/CE/AdamW、有限差分、数值 oracle、恢复训练、小集合过拟合 | 暂不实现 ECPT/KD/ROI |
| D2 | ECPT 是否存在 | 一页数学规格、预注册对照、可实现的事件标签 | 若无法区别普通 GRU+STOP，就删除 ECPT 名称 |
| D2 | 仅在闭环后生成小样本 | 16/64 smoke → 100–300 episode、时长与吞吐报告 | 若 RTF 不满足预算，重新计算采样量 |

**决定研发主线的条件判断**：

- 小型 Gazebo 语言场景 + 连续动力学可行 → 先让它产生真实闭环数据，训练一个 TinyVLN 基线；无需等待官方 35GB 模拟器。
- AerialVLN 旧模拟器在 RTX 5090 上可运行 → 并行建立 RGB-only 官方协议对照。二者不是同一个任务与动作空间，严禁直接把 SR 并列。
- AerialVLN 不可用 → 记录证据，不编造官方闭环 SR；独立小数据集仍可支撑方法原理研究，但论文外部效度需要补充后续基准。
- CF-Drone 真实硬件有传感器 → 仅说明未来可研发估计/控制，不代表现阶段能悬停；没有传感器 → 继续仿真开发，不以 CF 作为训练阻断项。

## 10. 主要学术比较与统计纪律

**现有先例必须引用：**

- AerialVLN (ICCV 2023)：离散空中 VLN、LAG 与官方分割。
- Self-Monitoring (2019)、L2Stop (2020)：进度与 STOP 分离不是新的概念。
- LightVLN (2026-10)：紧凑历史、指令条件聚合和机载 HIL，实际使用 0.5B 语言骨干；不能把其 Val-Seen 25.83% 与本项目 Val-Unseen 相比。
- AirAnchor (2026-09)：空间锚点、持久地标记忆与进度反思。
- X-Fly (2026-10)：真实闭环恢复与 Bernoulli STOP 训练，是独立终止头的直接先例。
- Flix：原始 Arduino/Gazebo 仿真存在，但**未证明**与 CF-Drone 的代码授权关系。

**统计协议：**

- 主对比先定协议/split、输入合法性、阈值校准集、指标排序和停止条件。相同 scene/task/start/render seed 配对；报告场景级 bootstrap，不能把同场景的帧/多指令视为独立场景。
- `SR`（0/1）与 `NE`/`SDTW`（连续）各有价值；连续指标不意味着完全摆脱样本相关问题。选 STOP 阈值只能在开发/校准集，`val_unseen` 不可反复选择最优点。
- 用实际成功率、配对分歧率和场景内相关估计所需 episode 数；评审中的固定“531 只能检测 Xpp”或“5pp 永远不显著”不能作为普适定理。
- 不把“通过 15 项测试”当训练方法的实验次数。若 ECPT 因消融无效而被删，工程仍保留为有意义的负结果与可复现实验记录，但不能保证负结果一定被顶会接收。

## 11. 必须向其他 AI 追问的 12 个核心问题

1. 我们提出的 ECPT 能否在 10 行数学内给出不同于 Self-Monitoring、AirAnchor、L2Stop/X-Fly 的独立信息操作？
2. 同输入同参数的 GRU + progress + calibrated STOP 能否复现完全一样的行为？
3. 视觉证据门控 `η_t` 在遮挡或误识别时如何避免“错误进度吸收态”？
4. AerialVLN-S 旧模拟器在 RTX 5090 上是否真实兼容？实测 log 和版本是什么？
5. 当前场景有没有不看语言/不看图像就高成功率的捷径？
6. 小型 Gazebo 物理闭环是否真的产生动作导致的下一帧，而不是位置瞬移？
7. CF-Drone `SERIAL_CONTROL`、未实现 ALTHOLD、姿态积分这三项是否满足实体测试阻断标准？
8. 实物 ESP32 是 C3/S3/经典款？真实 RC/NAV UART 是否冲突？扩展传感器是否实际存在？
9. `CF-Drone/LICENSE` 和 Flix 的来源、授权、可分发性是否经独立法律与源码 diff 核实？
10. 量化后是否仍具有可比较的端到端 latency p50/p95/p99、内存和闭环成功率？
11. `review_v2.patch` 的安全修复是否经独立测试与 clean clone 验证，而非只是复现脚本返回“已修复”？
12. 在 2 周后，能否展示至少一个由当前代码**实际运行产生的**可验证研究结果（camera trace、trajectory trace、模型 loss），而不只是新文档和 UI？

## 12. 公开资料（可自行复核）

- AerialVLN ICCV 2023：https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html
- AirVLN 仿真器与数据：https://github.com/AirVLN/AirVLN
- LightVLN：https://arxiv.org/abs/2610.05024
- AirAnchor：https://arxiv.org/abs/2609.08442
- X-Fly：https://arxiv.org/abs/2610.21450
- L2Stop：https://aclanthology.org/2020.findings-emnlp.62/
- Flix：https://github.com/okalachev/flix
- LicheeRV Nano / SG2002 官方规格：https://wiki.sipeed.com/hardware/zh/lichee/RV_Nano/1_intro.html

## 13. 最终审查决议

**有条件接受评审结论，不照单全收；基础设施补丁建议审查合入，科研主线必须重整。**

下一阶段唯一有意义的跃迁是从“已有只读看板 + 静态障碍 XML”升级到“有物理反馈的 RGB 导航 episode + 能真正训练的最小学生模型”。这两个可运行对象首次出现以前，不继续开发 UI，不把 ECPT 称为已实现算法，也不改 ESP32 固件。
