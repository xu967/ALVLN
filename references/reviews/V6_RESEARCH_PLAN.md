# TinyAerialVLN V6.0：小数据集起步、专用 C++/CUDA 训练、SG2002 部署与工程约束的完整方案

> 文档日期：2026-10-28  
> 状态：**研究设计与实施规格（未开始新代码实现，未经板端/仿真实测）**  
> 继承：用户 V3、V4、V5 与五份独立评审；**保留真实的空中视觉语言导航（Aerial VLN）目标**，不退化为无语言的四动作避障分类。  
> 最终形态：Ubuntu + RTX 5090 训练 → LicheeRV Nano（SG2002）视觉推理 → UART → ESP32 飞控。视觉学生允许 RGB、自然语言、历史动作、内部记忆；**导航模型不使用 IMU / 深度 / GNSS / 真实位姿作在线输入**。ESP32 可以使用原本已有的惯性稳定控制。  
> 重要：本文一切参数、预计工期、性能目标、数据容量目标均是**待验证的设计值**。尚未在用户的 Ubuntu、SG2002、ESP32 上执行训练与实机测试。

---

## 0. 结论先行：停止依赖“大型基准先行”，采用三层数据与两层评测

项目的关键矛盾不是“学生太小”，而是：**AerialVLN-S 自身标注小，约 35 GB 的 AirSim 模拟器大；旧 UE4 在 RTX 5090 上的实际兼容性未知。** 不应为了验证一个 64 维 GRU 而先搭建全套旧城市模拟器。因此将资源需求拆开：

1. **L0 / 最小功能验证：BabyAI（MiniGrid）**。天然有自然语言 mission 和可交互动作，可程序化产生数百条任务，几乎不需购买大场景文件；但是默认 `7×7×3` 是对象编码，**不是 RGB 像素**，必须用 `RGBImgPartialObsWrapper` 获得渲染 RGB，并保留原任务文本字段。只作为学习算法和数据合同测试，**不能作为空中导航成绩**。
2. **L1 / 第一阶段主开发平台：MiniWorld 2.1.0**。公开约 39.4 MB 的 wheel、室内三维第一视角 RGB，可程序化构建任务。我们需要自己生成“路过 A、右转经过 B、在 C 附近停”的语言与参考动作；暂定生成 **TinyWorld-Lang-v0**（项目自建数据，不是公开论文已发布的官方数据集）。MiniWorld 2025-08 已宣布停止后续维护，所以**锁版本 + 封装 Adapter + 失败降级**；其物理模型简单，**不等于真实四旋翼动力学**。
3. **L2 / 学术正式基准：AerialVLN-S**。等小模型完成闭环、SG2002 跑通后再部署 35 GB 场景。保留官方八动作与评测协议；不能把 TinyWorld 的成功率当 AerialVLN 的 SR。

**建议最终论文的两个独立证据链：** (A) L1 上的小数据、语言干预、进度/STOP、训练蒸馏与内存消融；(B) AerialVLN-S 正式复现（若模拟器兼容性通过）+ SG2002 真实端到端测量。物理飞行安全另作 C 轨，不能以 A/B 直接推出。

**本次 V6 最重要的决定：不强制一开始实现 ROI、3DGS、PPO、通用自动微分、ROS 或 PX4。** 科研主线先验证可观察证据—进度—终止机制；蒸馏作为增量、动态采样作为可否证分支。

---

## 1. 固定的研究问题与可证伪假设

### 1.1 研究问题

**RQ：** 对于只能部署约百万参数级视觉语言模型、在线仅可见单目 RGB 与自然语言、不能使用惯性导航状态的微型无人机端侧计算平台，能否使用**证据可靠性约束的紧凑语义进度记忆**，改善长指令阶段切换和终止决策，并通过强教师学习进一步改善未见场景的闭环导航？

如果只研究“普通小 CNN + GRU + KD”，学术新意较弱。V6 重点研究**模型在错误或含糊视觉证据下何时允许推进语义阶段，以及终止判断与阶段证据如何相互约束**。同时要承认 Self-Monitoring、LookasideVLN、QAD、AirAnchor 和 LightVLN 已经探索过其中的部分要素，本方案**不是已获确认的原创算法**。

### 1.2 三条递进假设

- **H0（可行性）**：在 L1 自建任务和 L2 官方协议里，小 CNN + 文本编码 + 紧凑 GRU 能够真正利用语言信息，获得优于随机 / 语言打乱的闭环表现，并通过 SG2002 INT8 部署。
- **H1（方法主假设）**：对显式“已见地标证据、当前阶段候选、完成确认”建模，优于**同参数量**普通 GRU、Self-Monitoring 式回归或硬单调进度指针，尤其减少中间阶段误切换、早停和到达未停。
- **H2（可选蒸馏）**：只有在视觉教师 T1 被证实闭环优于学生时，从其动作/STOP 分布蒸馏，能在相同部署结构和相同数据预算下改善泛化；否则取消 KD。

**H3 可选**：语言或历史驱动的非均匀视觉采样（single-pass fovea / ROI）是否在**真实板端同 P95、同容量、同原始帧信息**下有收益；只有在原生高清帧可获得后才研究。

### 1.3 明确的否证规则

只要观察到任一情形，就不得宣称相应机制成功：相同输入时替换/打乱自然语言无影响；去掉进度机制与等容量 GRU 无差别；STOP 头只提高 SR 却大幅恶化 NE/SDTW；KD 仅改善 train 模仿损失而无闭环改善；采样机制获益完全来自更高原始分辨率而非观察决策；INT8 下模型不再保持关键阶段/终止行为。

---

## 2. 公开数据集和小型平台重新选型（解决“大”）

### 2.1 候选资源对照：必须区分发布大小、真实图像、语言、交互和物理性

| 资源 | 语言指令 | 可用视觉 | 动作/交互 | 获取成本与关键事实 | 建议定位 |
|---|---|---|---|---|---|
| **BabyAI / MiniGrid** | 有自然语言 mission | 默认符号网格；可加 RGB 渲染 wrapper | 现成 7 动作等任务 | 任务程序化生成，基本不需下载场景素材；非三维真实相机 | **D0：训练器/语言/记忆烟测** |
| **MiniWorld 2.1.0** | **需要自建语言任务**，官方不是 Aerial VLN 语料库 | 第一视角 3D RGB，亦可查询深度作训练期标签 | 可交互第一视角导航、简化物理 | 官方 PyPI wheel **39.4 MB**；2025 年已宣布停止维护 | **D1：首个完整 RGB—指令—动作闭环** |
| **CHAI（2018）** | 有人类导航语句 | 配合原 CIFF/Unity 生成第一视角 | 原论文交互式连续段落 | 约 **1,596** 个 paragraph；需要旧 Unity/CIFF；**总下载体积未核实** | 人类语言辅助，**不作为首选可执行平台** |
| **LANI（2018）** | 有人类长指令 | 需原 Unity 场景实时渲染 | 前进/转向/停止 | **6,000** paragraph；旧环境与运行库兼容性不确定 | D1 备选/语料对照，不要求先跑通 Unity |
| **AVDN（ACL 2023）** | 有对话导航 | 基于 xView 的卫星/航拍影像 | 非同款前视 AirSim 八动作 | 任务集相对小，但**还需下载 xView 图像**，总体资源不确定 | 不选为主训练集：视角/协议不匹配 |
| **AerialVLN-S** | 有正式 Aerial VLN 长指令 | 运行时 AirSim RGB/Depth，不是轻量现成逐帧包 | 官方离散 8 动作 | 标注 <100MB，模拟器官方 README 约 **35GB** | **D2：正式空中学术评测** |
| **IndoorUAV（AAAI 2026）** | 有，约 16k 路径 | Habitat + MP3D/HM3D 等三维素材 | 室内 UAV 4DoF | 论文包含 **1000+ 场景**，实际场景下载不是轻量起步 | 未来连续飞行/跨数据扩展 |
| **OpenFly** | 有 | 多渲染引擎与场景 | 自己的动作合同 | 约 100k 轨迹、场景资产复杂 | 第一轮明确不接入 |
| **DroNet** | **无自然语言导航任务** | 第一视角图像 | 转向/风险回归 | 可做低层视觉风险对照 | 只能做感知辅助，不可作为 VLN 主数据 |

**注意**：以上平台中真正同时满足**公开、人类语言、RGB 第一视角、低资源、现成可交互空中评测**的理想候选并未查实。不能虚构出一个“几百 MB 且各项齐备的 AerialVLN-S”。V6 采用诚实的组合策略：**公开轻量引擎 + 自建小型语言导航 episode + 后期官方对照**。

**一手入口：** [MiniGrid BabyAI](https://minigrid.farama.org/environments/babyai/)；[RGB wrapper](https://minigrid.farama.org/main/api/wrapper/)；[MiniWorld](https://github.com/Farama-Foundation/Miniworld) / [PyPI](https://pypi.org/project/miniworld/)；[LANI/CHAI EMNLP 2018](https://aclanthology.org/D18-1287/)；[AVDN 官方代码](https://github.com/UCSB-AI/Aerial-Vision-and-Dialog-Navigation)；[AirVLN](https://github.com/AirVLN/AirVLN)；[IndoorUAV AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39562)。

### 2.2 推荐数据路线：D0 → D1 → D2，不采用“按帧随机划分”

**D0：BabyAI-Visual-Smoke（自己生成）**

- 使用 BabyAI `GoTo`, `GoToSeq`, `OpenDoorsOrder` 等任务；`RGBImgPartialObsWrapper` 必须显式保留 `mission`，不使用把 mission 丢弃的 `ImgObsWrapper`。
- 先生成 **200 train / 40 dev / 60 test** episode（**建议配额，不是 BabyAI 官方 split**）；语言指令与 agent 视图/动作等一起存档。
- 控制组：模型看到 RGB + mission；只看 RGB；交换 mission；随机 mission；只看 mission。交换指令后决策应随任务变化，证明文本确实有用。
- 风险：这是 2D 网格目标任务的 RGB 渲染，任务动作 7 类，不允许直接用这里的 STOP 定义替代 AerialVLN `STOP=0`，也不能以 D0 的表现说无人机会飞。

**D1：TinyWorld-Lang-v0（本项目新生成）**

- 平台用 MiniWorld 2.1.0（固定源 commit、依赖版本、许可证），以 3D 第三人称内部场景配置和**agent 第一人称 RGB**采样。先使用平面导航，保留改变观察方向、路径遮挡与连续帧，不假设其拥有四旋翼动力学。
- 任务模板由严格可验证的程序产生：`Go toward the blue marker, pass to its left, then stop near the red marker.`；或 `Go through the doorway, turn right, and approach the green pillar.` 这里“目标、顺序、方位、STOP”都必须存在于生成场景的可校验元信息中。
- 给出三级语言复杂度：K1 单目标 GoTo；K2 两阶段 Pass+Turn；K3 三阶段 Pass+Turn+Stop / 相似地标干扰。环境随机化改变纹理、颜色、障碍布局与起终点；**测试场景采用全新布局模板与对象组合**，不是只换随机种子。语言先支持英语，中文以后独立扩展，不混淆词表。
- 生成 expert：训练期从全部地图、对象语义与已知目标计算**按地标顺序约束的参考路径**，不是简单起终点最短路。将其映射为平台动作，用于行为克隆；对偏离状态做局部纠正。**专家可读真值，学生在线绝不读真值。**
- 规模：先 30 条作渲染/格式/时间戳 smoke；再 200 条作 overfit；随后 **1,000 train / 200 dev / 300 test**；如有增益才扩为 **3,000 train / 500 dev / 600 test**。数字是建议实验配额；不要将这些称为官方发表数据集。
- 存储：优先单份高质量原始 RGB 或流式渲染，离线缓存需预算；按 `scene_seed, layout_family, language_template_id, episode_id` 分组拆分。可另留 **100 条相机域扰动压力测试**，与 main test 不重复。
- MiniWorld 已停止维护：尽可能只扩展关卡生成和观察 Adapter，**禁止 fork 后全面改写渲染器**。若遇无法运行的 OpenGL/Pyglet 兼容问题，在 3 个工作日内评估是否改为自建最小 OpenGL 观察器或 Gazebo 小场景；不允许连续数周只修模拟器。

**D2：AerialVLN-S 正式数据**

- D1 学到的是“小型三维语言导航”，不是正式无人机 VLN 成绩。D2 使用 AirVLN 官方 train 10,113 / val_seen 333 / val_unseen 531（论文 test 771；下载版本可能有 770，须核对包哈希与统计），原八动作训练/评测。
- 官方默认 RGB 分辨率一般代码中设 224×224；研究新分辨率时设独立观察协议（更换观测必须重跑对照），不可引用官方 RGB+Depth 作为同输入公平基线。
- D2 资源风险：**不得先下载 35GB 然后发现 GPU/UE4 不兼容**。先做分发文件/显卡/画面/随机 10 条 episode 的可运行性 probe；不能兼容时保留 D1 成果，另寻能运行旧模拟器的独立环境，绝不把离线分类准确率冒充 D2 的闭环 SR。

### 2.3 不能选择为主数据集的理由

- R2R / REVERIE / RxR：虽有文本导航，但常依赖大型室内三维扫描或全景导航图，和前视小型飞行器观察/动作协议不同；不能仅凭“标注文件小”下结论。
- AVDN：虽然名字有 aerial 与 dialog，观测基于 xView 卫星影像，是另一种视角与任务；可作为语言对话研究补充，**不满足第一视角机载相机闭环同构**。
- 普通视频/VisDrone 分类或检测：只有视频、目标框却没有自然语言与可执行动作，不能作为端到端 VLN 主训练数据。
- 只有 LiDAR/Depth/GPS 动作监督的飞行论文数据：可在训练时作特权标签来源，不能不加说明就给纯 RGB 学生喂入状态信息。

### 2.4 episode 数据合同（四类字段，任何环节不可删）

**A. 观察字段（学生合法输入）**：`episode_id, step, rgb_ref, frame_timestamp, instruction_utf8, previous_action, previous_action_age, valid_image_mask, reset_flag`。MiniWorld 的 step 观察要复现 “只看当前+过去”，不能暴露 agent 精确位置。

**B. 训练专用标签（privileged）**：`expert_action, route_waypoint_index, route_progress_label(optional), stop_label, goal_reach_event, landmark_visibility_label(optional), collision_gt, off_route_flag, teacher_logits(optional)`。标签不存在时写 `null + valid_mask=0`，**严禁自动填 0 并参与损失**。

**C. 仿真元信息（不可进学生推理）**：`scene_id, layout_family, spawn_pose_gt, landmarks_gt, camera_intrinsics, action_duration, collision_geometry_version, sim_commit, renderer_version`。

**D. 实验可重复性**：`schema_version, sample_sha256, renderer_seed, instruction_template_seed, split_manifest_hash, tokenizer_hash, preprocessing_hash, code_commit, expert_version, license_id`。

**每次重放的强制校验**：相同 seed 与动作序列的重置状态、动作映射、终止与异常标签；若图片渲染浮点细节不逐字节确定，至少检查几何、帧尺寸、轨迹/指标在预设容差内可重复，不擅自要求 bitwise 图像相同。

---

## 3. 仿真与训练环境的现实配置

### 3.1 建议的环境拓扑

| 层 | 环境 | 安装/依赖约束 | 做什么 |
|---|---|---|---|
| Host | Ubuntu 24.04 + NVIDIA 驱动 + RTX 5090 | 先 `nvidia-smi` 核对，CUDA 工具链要支持 `sm_120`；精确版本由官方兼容矩阵固定 | 专用 C++/CUDA 学生训练 |
| `sim-lite` | Python 3.11 虚拟环境 | 仅 MiniGrid/BabyAI、MiniWorld（按 2.1.0 和 Gymnasium 锁版本）；无学生 PyTorch 依赖 | 程序化场景/渲染/专家标签；**Python 是环境胶水，不是学生训练框架** |
| `toolchain` | 厂商认可的 TPU-MLIR 容器 | 固定工具链 commit、SG2002/CV181x 后端、SDK/runtime 匹配 | ONNX → 校准 → `.cvimodel` |
| `airsim-legacy` | 独立旧版本环境（条件启用） | 官方 AirSim/Python 等旧依赖与图形 ABI，运行成功后锁版本 | D2 标准闭环，不污染 Host |
| Board | SG2002 Buildroot/Debian + NPU Runtime | 实测 PCB 70405/70415 与系统内存 | C++ 视觉推理、CPU 文本/记忆、UART |
| Controller | ESP32 自带飞控 | 现有固件版本固定、执行能力记录 | 接高层命令，独立稳定/超时/人工接管 |

RTX 5090 的 CPU/显卡驱动与 OpenGL 兼容须实测；不要在未验证的 Ubuntu 上混装 ROS Noetic、Humble、Jazzy 或用于 AirSim 的古老 GPU 用户库。当前阶段不需要 ROS。

### 3.2 最先执行的两个探针

**Probe-S（轻量模拟器）**：能以 `rgb_array` 拿到一帧实际 RGB、有一条英语 mission、有可重复动作序列、能 reset；抽查 RGB 通道顺序、画面尺寸、step_time、episode 终止；输出 30 条可复核 episode。MiniWorld 若工作需要图形环境，先测试 EGL/Xvfb 兼容，而非假设无显示器必能跑。

**Probe-H（Nano 硬件）**：读 PCB 丝印、确认主核模式、摄像头 ISP 路径、内核和 NPU 运行库；用厂商官方静态 Conv 示例，检查 ONNX 输入 `[1,3,224,224]` / `[1,3,192,320]`、DWConv、BN folding、两个输出、INT8 标定，再测 camera→preprocess→NPU→CPU→UART 的 P50/P95/P99 与 RSS/ION。**如果 Probe-H 失败，模型规模暂定不是性能结论。**

### 3.3 为什么不用 Isaac Sim / Gazebo 作为第一天主平台？

它们更适合连续物理飞行和传感器模拟，但小语料验证的第一目标是神经网络是否理解指令、记忆是否有效、序列状态能否复位。先搭重仿真会使验证被渲染/飞控依赖阻断。等 L1 成立后再用 Gazebo Harmonic + 自家 ESP32 控制等价动力学系统进行物理迁移；这里不直接声称 MiniWorld 具备四旋翼动力学。

---

## 4. 总架构：工程执行链和研究模型分离

### 4.1 两条互不偷换的执行链

**开发/学术链**：`MiniGrid/MiniWorld/AerialVLN` → `episode adapter` → `C++/CUDA trainer` → `FP32 checkpoint` → `ONNX exporter` → `TPU-MLIR` → `SG2002 cvimodel` → `端侧策略输出`。

**真实控制链**：`前视 RGB + 用户指令` → `LicheeRV Nano：NPU 视觉特征 + CPU 文本与状态记忆` → `高层有界意图` → `UART 帧` → `ESP32 已有姿态/定高/接管系统`。

AerialVLN 官方 `Forward=5m` 是**离散位姿更新**，而真实串口命令必须指定方向/幅度/有效期且与 ESP32 能力一致。必须通过 `action_adapter`，不能把八动作原封不动映射成控制时间。

### 4.2 M0 学生模型：TinyVLN-Base

输入为 `(RGB_t, text_tokens, history_state, prev_action)`，不增加其他在线观测。

| 部件 | 建议张量 / 状态 | 实现位置 | 初期依据 |
|---|---|---|---|
| VisionStem | `[B,3,224,224] → [B,16,112,112]` | NPU 静态 Conv | 官方默认观测首选 |
| DW/PW blocks | 16→24→32→48 | NPU Conv/DWConv/PW | 较小参数、可导出；实际支持待 Probe-H |
| SpatialMap | `[B,48,14,14] → [B,49,64]` | NPU 特征输出，CPU reshape | 保留地标方位，而非整图 GAP |
| TextEncoder | `tokens [B,L] → E [B,L,64]` | 训练时 C++/CUDA，部署时 CPU 仅 episode 起始 | 避免强制 48 token，L 由训练集分位数决定 |
| SpatialFusion | `E + SpatialMap → z_t∈R^64` | CPU/小矩阵计算 | 单头/低秩可对拍实现 |
| GRU | `h_t∈R^64` | CPU 逐步 | 维护历史，禁止跨 episode 污染 |
| ActionHead | `logits∈R^8`（仅 D2 协议） | CPU | Aerial 8 动作，STOP=0 |
| StopHead | `logit_stop∈R` | CPU | 单独处理不平衡终止条件 |

**严格区分输出头：** 在 BabyAI 和 MiniWorld 使用各自的 `TaskActionSpace`、`STOP` 定义和 head 的维度；到 AerialVLN-S 才使用八动作 head。**共享的是编码与记忆设计，不共享硬编码动作枚举和分数**。

参数建议从总计 0.3M、0.7M、1.2M 的三档**实算**起步，百万参数上限暂定。训练前输出逐层 param/MAC/activation 表，不许仅凭网络名称声称符合板端限制。

### 4.3 M1 主方法：Evidence-Calibrated Progress & Termination（ECPT）

先明确 K 个任务阶段来自指令本身可解释的 span 或**训练期结构化合成任务标签**；对于没有可靠人工对齐的 AerialVLN-S，标注必须设置弱监督/掩码，不能假冒官方具备逐步子指令标签。

状态拆成：

- `h_t^v (64)`：时间上的视觉—动作摘要。
- `b_t(k) (K+1)`：当前阶段分布，附 DONE；不盲目使用硬单调索引。
- `e_t (32)`：本阶段观测证据摘要、累计可信度/反证信息。
- `s_t`：上一合法**实际发送**动作与自上次发出以来的时间；发送不等于已经执行完毕。

用小网络计算当前观测对阶段 k 的相关性：`r_t(k)=sigmoid(f(z_t, E_k, h_{t-1}^v))`。更新时加入一致性/冲突信息，使一次误检测不立即推进。提出**待确认（PENDING）→ 已确认（CONFIRMED）**两阶段语义状态：只有可观察的连续证据满足开发集标定条件才推进；如果新帧否定候选地标，撤回 PENDING；对于真正不可见的目标应保持不确定状态，不伪造“确定完成”。

最终 STOP 取决于独立 `p_stop`、任务进入最终阶段的可观察证据、有限历史一致性和合法 action mask。**不允许在线拿“距离目标<20m”判 STOP**，因为该距离是 benchmark 评测真值，学生没有看到。

**核心候选创新**：不是“加一个进度预测头”，而是**对低容量记忆的事件更新施加可观察证据确认与撤回机制**。是否真的有新意，要对照 Self-Monitoring / 单调对齐 / 普通双 GRU / 其他 2026 工作，实测并报告负结果；在文献核查前不可声称首创。

### 4.4 图像分辨率与可选 ROI

- D0 和 D1：仅用固定原生 RGB 视图，Resize/Letterbox 方式作为实验合同冻结。
- D2：官方默认 224×224 为正式协议；320×192、384×224 需要**新原始图/自建观察协议**并重跑同协议基线。
- ROI 只有在确定具备**高于学生画布的信息**后才允许开始；严禁放大低清缓存冒充“新增信息”。
- 先测单次 3 通道全图、固定 foveated mosaic、受语言阶段条件的 mosaic、同 shape 二次 ROI。对无实验收益的方案立即删除。单次 mosaic 若利用上一帧阶段信息选择区域，必须量化滞后/转向失配；若利用当前帧 CNN 得到区域，则不是“一次视觉前向”，需计入额外推理。

---

## 5. 教师与学生：蒸馏顺序以及不做负迁移的策略

### 5.1 不能把三种“教师”混称为大模型

- **T0，符号/几何专家**：在 D0/D1 上读取地图真值、语义对象、目标顺序生成专家动作、事件真值、短期后果；它是**监督生成器**，不是部署时 RGB 策略，也没有可直接用于 KL 的神经 logits。
- **T1，学习型视觉教师（可选）**：可复用较大的视觉语言编码器或沿现有 RGB-only CMA/DINOv3 路线训练。必须记录完整闭环成功率，且合法输入与学生一致（如果教师用特权信息，应报告特权设定差异）。教师原来只做动作分类且没有闭环成绩，不能默认是强教师。
- **S，TinyVLN-ECPT 学生**：轻量网络，要求部署输入严格合法，只有学生运行在 SG2002。

### 5.2 四阶段学习，而不是一上来堆全损失

| 阶段 | 数据 | 损失 | 验收/退出 |
|---|---|---|---|
| P0 最小 BC | D0/D1 专家 episode | `CE(action_logits,y)` | 16–64 episode 可过拟合，语言干预有效 |
| P1 STOP | 加权/分层终止正负样本 | `CE + λ_s BCE(stop)`，或 focal 二选一 | 早停率/漏停率下降，NE/路径不恶化 |
| P2 事件证据 | 自建 D1 **明确可验证**阶段事件标签 | `P1 + λ_e CE(event) + λ_c consistency` | 比等参数 GRU/自监控基线有稳定收益 |
| P3 KD 可选 | 已经强于学生的 T1 + 学生 rollout | `P2 + λ_k τ²KL(p_teacher^τ||p_student^τ)`，对低置信伪标签降权 | 闭环/迁移有净收益，未通过即取消 |

STOP 类不平衡不能只靠高权重一项解决。开发集需要扫独立 STOP 阈值，但不能在正式 `val_unseen` 上无限次阈值搜索。对于人为控制 STOP 的对照，还要检查“到达率”和“到达后成功停”的条件概率。

### 5.3 防止学生学到自己根本看不见的“教师作弊信息”

举例：T0 知道墙后有第二个目标，学生当前 RGB 完全不可见，强逼学生拟合 T0 确定动作，会让训练目标不一致。D1 因为生成器控制真值，可定义 `visibility_mask` 以及 `oracle_only_event`；对不可观察状态禁止进行一帧确定性强监督，或使用历史证据加权。**这不等于知道不可见状态时学生一定应该 STOP**，应让模型表达不确定、由任务策略决定是否继续获取观察。

### 5.4 小数据也能严谨使用蒸馏

推荐在 D1 使用 1,000/3,000 条轨迹先筛查 T1；若样本不足且 T1 学习不稳定，优先 T0 监督 + 恢复数据（DAgger 类），不要消耗大算力训练 72B VLM。只在强教师通过配对闭环对照后增加 KL 蒸馏。冻结教师若用 PyTorch 离线导出 logits，需写明仅作为**教师离线工具**，学生正式训练继续保持 C++/CUDA。

---

## 6. 自研 C++/CUDA 训练引擎：从数学到模块的实现顺序

### 6.1 原则：写的是专用训练器，不是“替代 PyTorch 的万能框架”

第一版只支持固定模型、固定张量 shape family、batch-major 训练、可变长度 mask/episode，明确前向和手写反向；不做通用 autograd DAG，不做通用编译器、算子注册宇宙、多 GPU、分布式、在线网页 UI 和任意 Python 模型导入。要模仿高飞团队自主实现关键 C++ 数学/优化与软件边界的思路，而不是简单以“没用框架”证明创新。

### 6.2 内核职责拆分

**Core：** `TensorView`（指针、dtype、dims、strides、owner/lifetime）、`Arena`（一步内临时激活分配）、`StaticGraphExecutor`（严格按固定拓扑顺序调用）、`SeedManager`（CPU/GPU RNG 和复现记录）、`CheckpointWriter`（版本、shape、CRC/hash）。首个版本必须有可读取的错误码、无非法越界、清楚的 CPU/GPU 所有权。

**Ops：** Linear/GEMM，Conv2D，DepthwiseConv2D，pointwise Conv，ReLU/SiLU（选择一种）、pool，Embedding、Softmax/CE、BCE、GRU、masked attention、AdamW。只有确有实验需求才增加 BN/QAT/KL 等。卷积优先通过自研 tensor/调度接口调用 cuDNN/cuBLAS；可提供慢速 CPU 参考以便对拍，不需要自己造 GEMM 的每一行。

**Model：** VisionEncoder / TextEncoder / Fusion / GRU / ActionHead / StopHead / ECPT；各模块只有显式输入输出接口，不允许数据处理模块直接修改权重，也不允许仿真 Adapter 读取学习率和优化器状态。

**Data：** RendererAdapter 负责 Python 环境 IPC 或写入 episode 序列化；DatasetReader 负责 schema 和 mask；Tokenizer 负责词表；SplitChecker 负责按路径/场景阻断泄漏。**训练器读统一 episode 格式，不直接 import miniworld、AirSim 或 PyTorch。**

**Deployment：** ONNXWriter 静态导出视觉 NPU 图，TPU-MLIR 生成匹配 `cvimodel`；文本 encoder / GRU / ECPT 小矩阵及 safety / UART 先由板端 C++ CPU 运行。CPU/NPU 输出 parity 是固定门禁。

### 6.3 数据接口示例（规格描述，不是实现代码）

```text
EpisodeHeader: episode_id, scene_id, split, instruction, action_space_id,
               camera_contract_id, tokenizer_id, schema_version, hash
FrameRecord:   step_id, timestamp_ns, rgb_hwc_ref, prev_action,
               student_visible_mask, expert_action, expert_action_valid,
               stop_label, stop_label_valid, event_label, event_label_valid,
               done, truncated
ModelStep:     input_rgb_nchw, token_ids, text_length_mask,
               prev_action, reset_flag, hidden_in
ModelOutput:   action_logits, stop_logit, hidden_out, optional_event_logits
```

不在公开 API 中直接传 God-view 地图/GT 位姿；有权限读取这些标签的模块仅在 `teacher/` 或 `sim_adapters/`，并且必须通过学生输入白名单测试。

### 6.4 CUDA 静态图/梯度实现的详细顺序

1. **数值参考**：用独立 NumPy 或一次性 PyTorch reference oracle 生成固定 FP64/FP32 测试向量。正式训练器不依赖这些包；CPU/CUDA 只互相比对不足以证明数学公式正确。
2. **Linear/Softmax CE/BCE**：采用稳定 `logsumexp`；验证极端 logits、全 PAD、空有效 batch、类别失衡与 mask。loss=Σ有效样本/有效数，不得把 PAD 帧算进均值。
3. **Conv/DWConv**：明确 padding/stride/dilation/NCHW 与布局；forward/backward weights/input/bias 三套梯度；用小张量中心差分和独立前向 golden 对拍。
4. **Embedding 和 Tokenizer**：PAD=0、UNK=1，PAD embedding 无梯度；词表从 train 统计并冻结；处理 UTF-8、ASCII 标点、大小写；文本不完整不得静默截尾。
5. **GRU/BPTT**：明确定义 gate 顺序 `(z,r,n)` 或 `(r,z,n)` 并统一导出；序列 padding 不更新隐藏状态；`done/reset` 立即清空，不跨 episode 反传；多步与逐步推理输出一致。
6. **Cross-modal fusion**：先一个轻量 query-to-spatial map 交互，正确处理 attention mask 与数值稳定；空间区域位置编码不可省，否则左右含义被全局池化抹除。
7. **AdamW/训练循环**：学习率、decoupled weight decay、梯度裁剪、梯度累积、loss scale（仅后期用混合精度），每步检查 NaN/Inf；checkpoint 既保存权重也保存 optimizer/RNG/全局 step。
8. **静态导出**：首先只导出 CNN（固定 `[1,3,H,W]` 到 `[1,C,h,w]`），CPU 融合模块保留可移植 C++；等 SG2002 实测后再谈算子上推 NPU。
9. **端到端 parity**：同一个 episode、完全相同文本和帧，CUDA FP32 → ONNX → SG2002 BF16/INT8，比较逐步 action logits、stop、hidden 更新、final path；不是只比一个 argmax。

### 6.5 是否坚持完全不使用 PyTorch？

**交付的学生训练进程和上板进程：不依赖 PyTorch。** 允许仅测试目录内的 PyTorch/NumPy golden oracle（或冻结大型教师提取 logits），这不是“用 C++ 包一层 PyTorch”。若第一版手写训练器超时：保留 C++/CUDA 稳定核心，削掉非必要高级特性；**任何降级都需明示，不能暗中将 PyTorch 当主训练框架**。

### 6.6 持续可靠性检查

每次新算子必须同时有：shape/stride 契约、独立 forward golden、有限差分 gradient、CPU/CUDA tolerance、极端值/随机输入、checkpoint 恢复检查。新模块必须先证明在 32–64 条 episode 可以过拟合，不再添加其它功能蒙混过去。

---

## 7. `code/` 的强制目录约束（防止无限膨胀）

### 7.1 目录结构：只允许这些一级目录

```text
TinyAerialVLN/
├── README.md                  # 仅做启动与数据声明
├── V6_RESEARCH_PLAN.md        # 本文
├── CODE_RULES.md              # 单独发布的工程门禁
├── CMakeLists.txt             # 全项目唯一顶层构建入口
├── config/
│   ├── experiment.yaml       # 唯一模型/训练参数入口
│   └── deployment.yaml       # 唯一硬件/UART 参数入口
├── code/
│   ├── include/tinyvln/      # 公共 API：允许 10 个以内头文件
│   ├── src/
│   │   ├── core/             # tensor / arena / RNG / checkpoint
│   │   ├── ops/              # 数值算子 + 优化器
│   │   ├── model/            # 视觉、文本、融合、记忆、头、ECPT
│   │   ├── data/             # tokenizer / episode reader
│   │   ├── train/            # 单个训练可执行目标
│   │   ├── export/           # ONNX/权重导出
│   │   └── board/            # SG2002 策略与 UART 集成
│   ├── cuda/                 # 仅必要 CUDA kernels/dispatch
│   └── tests/                # 单元、梯度、parity、协议回归
├── adapters/
│   ├── babyai/               # 薄环境胶水，不参与学生训练
│   ├── miniworld/            # 薄环境胶水，不 fork 整套平台
│   └── aerialvln/           # D2 才建立，沿官方评测协议
├── tools/                     # 数据 manifest、评测桥、转换检查
├── manifests/                 # 小型文本清单、hash、版本
├── docs/                      # 至多 5 份长期维护合同
├── third_party/               # Git submodule / 外部路径链接，不复制源码
└── artifacts/                 # 训练结果、缓存、checkpoint，本地忽略提交
```

**不允许**在 `code/` 下出现 `experiments1/、new_model2/、backup/、final/、latest/、temp/、copy/` 或任务名堆叠目录。训练实验差异用同一套模型的 `experiment.yaml` + experiment ID 表达；下游 artifact 用哈希区分。

### 7.1.1 第一批允许创建的**确切源文件清单**（上限 24 个）

> 这是**规划名册**，不是已经存在或通过编译的代码。根据实际探针，可删除文件，不应不经审查增加第 25 个文件。

| 子目录 | 计划文件 | 数量 | 关键职责 / 测试 |
|---|---|---:|---|
| `src/core/` | `tensor.cc`, `runtime.cc`, `checkpoint.cc` | 3 | shape、stride、arena、RNG、持久化；边界/恢复 |
| `src/ops/` | `linear.cc`, `conv.cc`, `depthwise.cc`, `loss.cc`, `gru.cc`, `optimizer.cc`, `attention.cc` | 7 | 前后向、mask、finite-difference、独立 oracle |
| `src/model/` | `visual.cc`, `text.cc`, `fusion.cc`, `recurrent.cc`, `ecpt.cc`, `tinyvln.cc` | 6 | 模型前后向、隐态 reset、进度状态、接口一致 |
| `src/data/` | `episode_reader.cc`, `tokenizer.cc` | 2 | train/dev/test 隔离、完整指令、动作协议 |
| `src/train/` | `train_main.cc`, `eval_bridge.cc` | 2 | 唯一训练入口 + 官方/小平台评测适配 |
| `src/export/` | `onnx_export.cc` | 1 | 模型图静态导出、张量布局/权重 parity |
| `src/board/` | `infer_runtime.cc`, `uart_bridge.cc`, `safety_fsm.cc` | 3 | NPU/CPU 运行、CRC/TTL/序号、故障降级 |
| **合计** | **24 个计划源文件** | **24** | 不是必须一次性全部创建 |

建议 4 个 CUDA 文件作为初始上限：`conv_kernels.cu`, `pointwise_kernels.cu`, `rnn_kernels.cu`, `optim_kernels.cu`；**若 cuDNN/cuBLAS 已足够，相关 CUDA 文件可以不写**。建议公共头文件首先只建立 `types.hpp`, `tensor.hpp`, `ops.hpp`, `model.hpp`, `episode.hpp`, `tokenizer.hpp`, `protocol.hpp`, `runtime.hpp`, `export.hpp`, `errors.hpp`，总计不超过 10 个。必须通过接口和编译依赖检查才可以在 `src` 建实文件，严禁提前填空式生成所有类。

### 7.2 首版文件数量和体积的硬上限

| 项目 | M1 上限 | 意图 | 例外机制 |
|---|---:|---|---|
| `code/src/` 下 `.cc/.cpp` | **≤ 24** | 强制合并细碎职责而不做巨型单文件 | 超出需 ADR（架构决策记录）+ 删除/替换旧文件 |
| `code/cuda/` 下 `.cu` | **≤ 6** | 内核按功能族聚合，禁止每个算子单建十几个文件 | 数值测试证实必要时 ADR |
| `code/include/tinyvln` 头文件 | **≤ 10** | 稳定 ABI，避免 include 层级爆炸 | 公开 API 审查 |
| 业务 C++ 单文件 | **≤ 280 有效行** | 避免 1,000+ 行 `fit.py` 重演 | 仅自动生成代码和权重数据外；必做拆分 ADR |
| CUDA 单文件 | **≤ 360 有效行** | 必要 kernel 可稍长 | 超出需 reviewer 同意 |
| C++ 单函数 | **≤ 70 有效行** | 单一职责 | 算法推导的特殊循环需注明不拆理由 |
| `docs/` 非自动文档 | **≤ 5 份** | 禁止文档无限堆叠 | 更新原合同，不创 `v2-final` |
| 实验 YAML | **恰好 2 个** | 防配置漂移 | 不同实验只生成记录快照到 artifacts |
| 运行时依赖 | **不得包含 torch/torchvision/transformers** | 学生主链确实自研 | 仅 teacher-offline 或 oracle-tests 沙箱允许 |
| 可执行程序 | **初期 3 个** | `tinyvln_train`、`tinyvln_eval_bridge`、`tinyvln_board` | 导出可为 train 子命令，不要再造第 4/5 主程序 |

**有效行**定义：去掉空白、纯注释后计数；实现时由仓库本地静态检查脚本自动报告。阈值是项目自我管理规则，不是 C++ 社区通用质量标准；不应为了机械凑行数拆坏紧密相连的算法，也不应以美化文件计数为由绕过职责边界。

### 7.3 依赖方向（绝对禁止循环）

```text
adapters -> data schema <- train/model
                             |      |
                             v      v
                            ops -> core
board -> exported model contract / runtime SDK
```

规范解释：`core` 不能引用 `model/train/board`；`ops` 只能引用 core；`model` 只引用 ops/core；`train` 引用 model/data；`board` 只能调用导出合同和设备 SDK；`adapters` 只能输出规范化 episode，**不能调用训练器内部实现**。**官方 AirVLN 和 MiniWorld 源码不得直接复制进 `code/src/`。**

### 7.4 数据与模型不放 `code/`

- 禁止 `code/` 存 `.jpg/.png/.mp4/.zip/.pth/.pt/.ckpt/.onnx/.cvimodel/.npy/.npz` 与大规模日志；测试 golden 小样本除外，必须写明许可证及大小。
- 所有 dataset、预训练权重、缓存、运行日志、转换中间文件放仓库外 `data_root`/`artifact_root`，或忽略提交的 `artifacts/`。
- 单次 PR 的大文件需要 Git LFS 或外部存储；**不准**把第三方整个大仓库复制成项目目录。
- 禁止任何训练/评测代码硬编码 `/home/xxx`、Windows 路径、串口 `/dev/tty*`、数据集根目录。只能从 `config/`、环境变量和 CLI 合法参数读取。

### 7.5 开发者/Agent 的任务分工约束（即使未来并行也适用）

**角色 A：算子与正确性。** 独占 `core/ops/cuda/`，只改有明确需求的算子，提交数值报告。

**角色 B：数据协议与环境。** 独占 `adapters/data/manifests/`，保证泄漏隔离、可重放、RGB/动作合同；不得为适配模拟器修改模型代码。

**角色 C：模型与评测。** 独占 `model/train/`，只在公共头文件合同通过审查后集成；不得绕过数值门禁偷偷更改算子语义。

**角色 D：板端与飞控。** 独占 `board/export/`，只处理 ONNX/cvimodel、CPU/NPU、UART 的安全状态机；不得改变数据集评分指标。

注意：这是**未来的协作边界设计**，不表示目前已经实际调用 4 个 Agent。发生公共接口变更必须先写 ADR，锁定与回归测试通过后再跨目录合并；每个角色最多一个正在进行的特性分支。

### 7.6 Pull Request 必须通过的门禁

1. **目标**：关联一个 RQ/Hx 或确凿 bug，说明为何需要新增文件；不可无目的地“重构一下”。
2. **变更范围**：默认单 PR 不超过 8 个代码文件、净新增不超过 500 有效行；突破需要 ADR。
3. **测试**：单元、gradcheck、schema、动作协议、tokenizer、隐态 reset、训练 checkpoint；板端影响的修改须包含转换/parity。
4. **数据**：不得以 val_unseen 进行调参；训练/开发/最终评测必须按 scene/path 分组。
5. **性能**：影响部署的 PR 必须附 NPU 编译记录和 camera→UART 分段测量；没有硬件条件时标记“阻塞”，不得假装验证成功。
6. **依赖**：新增第三方库说明许可证、版本、二进制大小、为什么不能用既有工具。
7. **质量**：无未使用参数、隐式全局状态、未经说明的 magic number、硬编码动作 ID、静默截断、把 NaN 替换为成功数据。
8. **文档**：只更新现有合同 / README，除非确有新的长期维护边界；实验笔记写入 artifacts，不新建十几份 MD。

---

## 8. 实施里程碑：任务、产物、时间边界与失败路线

> 计划按单人主导开发估计，可并行安排硬件/数据探针；这些是**相对工作预算与决策门槛**，不是任何已经执行的进度承诺。

| 门禁 | 时间盒（建议） | 最小产物 | 通过条件 | 不通过的处理 |
|---|---:|---|---|---|
| G0：资料与环境 | 3–5 工作日 | 板卡 PCB/固件调查；MiniGrid & MiniWorld 安装验证；数据源表 | 两个平台至少一个产出 RGB+语言+动作，板端识别成功 | 先修环境或换平台，不训练 |
| G1：数据合同 | 3–5 工作日 | D0 200 条 + D1 30 条、hash/split/语言事件标签 | 图像、语言、动作、STOP、reset 一致，可回放 | 修 schema，禁止搭更复杂模型 |
| G2：NPU 提前探针 | 3–5 工作日，与 G1 并行 | 最小静态 CNN INT8 `.cvimodel`、实际延迟和 RAM | 真实板上稳定加载与推理 | 缩输入/换算子，先别造学生大网络 |
| G3：专用训练最小集 | 2–4 周 | 固定模型训练器、独立 golden、gradcheck、checkpoint | 16–64 条过拟合且外部数值测试通过 | 缩训练器为 Conv+GRU+CE，不造通用 Autograd |
| G4：D0/D1 基线 | 1–3 周 | D0/D1 闭环结果、语言 swap、跨场景验证 | 语言有效，模型高于无语言/随机 | 修专家、场景、训练，不加 ECPT |
| G5：ECPT 与 STOP | 2–4 周 | 3 seed 等参数多阶段/STOP 消融 | 两个以上可靠机制指标改善，主指标不明显退化 | 删除额外阶段机制，保留简单 STOP |
| G6：蒸馏可选 | 1–3 周 | T1 完整成绩和 KD 对照 | 教师显著更强且学生确有净收益 | 不做 KD，保留 BC/纠偏 |
| G7：Nano 导出与 HIL | 1–3 周，可交叉 | ONNX/cvimodel parity、30min 热稳态、UART 故障注入 | 板端持续运行，无重复/过期危险命令 | 仅离线演示，不接动力执行 |
| G8：AerialVLN-S 正式 | 条件启动；先 1 周可行性探针 | 10–20 条随机官方闭环，随后全量 train/val 证据 | 原始观测、八动作、计分器统一且能跑 | 不宣称官方成绩，D1 作为工程/研究试验 |
| G9：连续飞行安全 | 独立任务，不同论文证据 | 物理仿真、ESP32 定高/接管、安全域与低速受控测试 | 足够的独立安全验收 | **禁止自由飞行** |

**优先级：G0+G1+G2 先走**，不用在准备数据之前写出 20 个未测试算子。只有 G3/G4 证明小型 VLN 可学，G5 方法创新才有研究意义。

---

## 9. 实验设计：小数据可以调通，论文不能以烟测代替证明

### 9.1 主对照矩阵（最多 7 个核心模型）

| ID | 架构 | 假设与控制变量 |
|---|---|---|
| E0 | 随机 / 多数动作 / 不看语言 | 评测器/类别先验最下界 |
| E1 | 小 CNN + 文本池化 + GRU + 任务动作头 | 可学 VLN 任务最小下界 |
| E2 | E1 + 独立 STOP Head | 停止不平衡单独对照 |
| E3 | E2 + 标准 Self-Monitoring/软进度状态 | 已知进度机制对照 |
| E4 | E2 + **ECPT 待确认-撤回证据更新** | 主方法；与 E3 **等参数** |
| E5 | E4 + T1 KD（条件启用） | 蒸馏独立增量 |
| E6 | E4/E5 INT8 on SG2002 | 学术模型能否真实部署、量化后的闭环一致性 |

分辨率、模型容量（0.3/0.7/1.2M）和输入历史长度不要乘到一个巨大三维网格造成实验爆炸。先用简单 mini 数据扫两个最可能的容量点，最终对照**固定一个选定容量**，并附一个紧邻容量点作为鲁棒性检查。

### 9.2 指标与分解

- **D0/D1**：语言条件 Success、按阶段地标顺序成功率、错误提前 STOP、过站漏停、路径长度比、失败恢复率、指令打乱退化幅度、跨场景/跨模板 SR；此外报告教师正确性和训练标签覆盖率。
- **D2**：`SR, OSR, NE, SDTW`（按官方定义）、早停、迟停、错误阶段切换、每个 unseen 场景分数；SPL/nDTW 如另加必须声明为扩展指标，不能写成原论文主官方全部指标。
- **SG2002**：`P50/P95/P99(camera→UART), NPU/CPU/resize/文本时延, RSS/ION 峰值, 温度/降频, 量化 action flip-rate, STOP reliability, 30min 稳定运行`。核对“接收 ACK”与“实际完成动作”非同一事件。
- **统计单位**：episode、轨迹、场景与训练 seed 分开；3 个关键 seed（最终关键对照建议 5）；配对比较时 cluster bootstrap 与逐场景结果并报。逐帧准确率只能作机制诊断，不替代闭环成功率。

### 9.3 必须出示的反例与干预实验

1. 同一观察换成相反指令；阶段/动作应随指令变化。
2. 同一指令移开关键地标；模型不应沿用记忆强行判 DONE。
3. 从 episode 1 切到 episode 2；记忆与 STOP 滞后状态必须完全 reset。
4. 人为插入假颜色/假地标，区分“一帧误检测”与“稳定经过”。
5. 终点被遮挡或绕路：模型应允许不确定，不因 teacher GT 提前终止。
6. 仅用固定分辨率比较 E1/E4，不能通过给 E4 更多图片或更多教师信息产生虚假收益。
7. 保留语言尾部内容 vs 强制截断 48 token，量化真实截断与 STOP 错误关系。
8. 历史被交换、重复动作连续输入、观测延迟增大，检验模型是否只是记住动作频率。

### 9.4 论文新颖性评审提前准备

必须在目标会议/期刊论文写作前，系统比较：Self-Monitoring 2019、Sub-Instruction Aware VLN、Input-Adaptive VLN 2025、LookasideVLN CVPR 2026、QAD TCSVT 2026、AirAnchor 2026、LightVLN 2026。若 ECPT 在等容量与等数据量下只是已有进度监控的功能等价实现，**承认并转为系统部署/受约束研究，不应包装成全新方法**。

---

## 10. 部署规范：SG2002 与 ESP32 的真实约束

### 10.1 SG2002 的分工与可测预算

- 厂商 Wiki：SG2002 主核 1GHz C906 或 A53（两种模式选一）、小核 700MHz C906、256MB DDR3、标称 1 TOPS INT8/BF16；PCB 70405 早期版本 1T 稳定性有限。不能由 TOPS 推断 CNN 延迟。
- **NPU：** 仅静态 VisionCNN / 必要可编译卷积，先按 batch 1、224×224 报告。CPU：episode 级一次 TextEncoder、逐帧 GRU/ECPT/Stop、安全门控与 UART。用真实时间戳测定 ISP/VPSS、颜色空间、resize、DMA、NPU 与 CPU。
- **初始工程预算**：学生权重优先 < 1.2M 参数、整体关键服务 RSS 争取 < 128 MB、P95 camera→UART 先争取 < 250ms、持续运行 30 min；这些仅为工作目标，实测后改；不将 250ms 直接等同可安全飞行速度。
- **量化路径**：自研静态 C++/CUDA 导出 ONNX → TPU-MLIR → 模型校准（只用 train/dev 合法资料）→ SG2002 可用 CV18xx `.cvimodel`，先验证 BF16 / INT8 和 NPU 编译兼容。基于 GPU FP32 的 argmax 一致不足，必须检查 STOP 阈值和整条动作序列的闭环漂移。

### 10.2 UART 应是可靠控制契约，而不是传字符串 `left`

建议帧：`protocol_version, seq, monotonic_timestamp, command_type, bounded_value, duration_ms, valid_until, checksum`。ESP32 回 `ACK_RECEIVED`、`EXEC_DONE`、`CURRENT_SAFETY_STATE`；三者不能混淆。相机采集**不能**等 ACK 后停止，而应持续取新帧；动作下发按飞控可执行条件串行/防覆盖。

推荐状态机：`DISARMED → IDLE_READY → EXECUTING → WAIT_DONE/REPLAN → SAFE_DEGRADED → MANUAL_OVERRIDE`，任何 Linux 进程卡住、串口丢包或过期动作都应由 **ESP32** 的硬件侧 watchdog 和已验证的飞控降级处理，不得默认 STOP 会悬停，也不能默认“没有 IMU 输给模型”意味着 ESP32 无需惯性稳定。

**实际自由飞行前必须**：ESP32 具备可靠的必要稳定能力、安全低速控制、人工接管和场地防护；无人桨台架和故障注入验证通过；评测的 5m 传送式 Aerial 动作不可裸映射真实无人机。

---

## 11. 项目完成的最低验收定义与失败退出路线

### 11.1 工程成功（最先追求）

- D0 的语言/图像/动作闭环和 D1 的第一视角小型 VLN 学习是有效的；
- C++/CUDA 学生训练能在小集上稳定下降并过拟合，独立 forward/gradient oracle 通过；
- LicheeRV Nano 静态 CNN 能运行，CPU 语言/记忆正确，串口协议有日志和可恢复异常；
- 同样输入的 FP32/INT8 输出具有可接受的一致性，全链 P95 有真实数据。

### 11.2 方法成功（有证据才能主张）

在匹配参数量、图像协议、训练数据和开发调参次数下，E4 相对 E2/E3 在未见布局/多阶段条件下有可重复的机制指标/闭环收益，STOP 的改善不是依赖更短路径或看了隐藏标签。如果没有，不应把代码体积和完全不使用 PyTorch 当作方法成果。

### 11.3 论文层级与现实飞行成功明确区分

D1 上高成功率只证明合成语言与简化 3D 环境理解；AerialVLN-S 正式闭环成绩证明该基准的语言路径跟随；SG2002 测量证明设备能推理；**没有一项单独证明真实自由飞行安全**。真正实机成功需要独立的连续动力学、ESP32 控制能力、视觉时延和人工接管实验。

### 11.4 关键备用路线

- MiniWorld 由于停止维护出现兼容问题：优先锁老依赖/环境；仍不行则改轻量纯 C++ 渲染器或最小 Gazebo 场景（**须记录任务变化**）。
- Aerial UE4 在 RTX 5090 上失败：保留 D1 成果及有诚实限制的论文，不假装做过官方 Aerial；等待兼容资源或选择公开可运行对照。
- C++ 训练核实现过慢：缩到专用静态图最小算子，保留 PyTorch/NumPy 为测试数值 oracle，不默许作为正式学生训练的隐藏后端。
- ECPT 无效：发布干净的 tiny VLN + 定量资源/STOP 研究，不再堆 ROI/KD 强拗论文创新。
- KD 无效：直接删除蒸馏支线；T0 标签 + 少量纠偏已足以形成有效实验对照。
- ESP32 无定高/可靠控制：只作无桨台架、视频回放和 UART/HIL，不进行自主自由飞行。

---

## 12. 参考资料、版本风险与结论出处

### 12.1 用户给定材料（V6 继承并核对）

- `TinyAerialVLN_V5_七份评审综合复核与完整研究方案_20261028.md`：三轨任务、默认 Aerial 协议、ECPT、C++/CUDA 最小专用训练器与 SG2002 资源门禁。
- `TinyAerialVLN_研究方案与跨模型评审稿_V3_20261028(1).md`、`TinyAerialVLN_研究方案_V4_修订执行稿_20261028.md`：自然语言、历史动作、进度、ROI、教师蒸馏、自研约束。
- 五份独立评审：令牌截断、官方渲染资产、RGB-only 公平性、SG2002 真正成本、STOP、强教师、统计功效和现实控制差距。本稿并未照单全收其时延估算或“未见过的方法必能发表”等强断言。
- `web_code_review_20261027.zip`：旧工程的 Aerial 八动作和 AirSim 位姿跳转作为参考。旧代码**不是**可直接用在 SG2002 的小学生模型。

### 12.2 外部一手链接（可按需要点击核对）

1. [MiniGrid/BabyAI 任务](https://minigrid.farama.org/environments/babyai/)；[RGB 观察 wrapper](https://minigrid.farama.org/main/api/wrapper/)。
2. [Farama MiniWorld](https://github.com/Farama-Foundation/Miniworld)；[PyPI MiniWorld 2.1.0](https://pypi.org/project/miniworld/2.1.0/)：wheel 约 39.4 MB，注意官方 2025-08 宣布 deprecated。
3. [LANI/CHAI, EMNLP 2018](https://aclanthology.org/D18-1287/)：LANI 6,000 paragraphs、CHAI 1,596 paragraphs；不是“下载即得官方无人机数据”的材料。
4. [AVDN, ACL 2023 官方代码](https://github.com/UCSB-AI/Aerial-Vision-and-Dialog-Navigation)：依赖 xView 卫星图像，非机载前视同协议。
5. [AirVLN 官方代码与 README](https://github.com/AirVLN/AirVLN)；[AerialVLN ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.pdf)：标注与仿真器分开；官方约 35GB 场景。
6. [IndoorUAV, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39562)：16k+ 轨迹/1000+ 室内扫描场景，不适合第一周下载。
7. [Sipeed LicheeRV Nano 官方 Wiki](https://wiki.sipeed.com/licheerv-nano)：SG2002 内存/NPU/PCB 版本差异。
8. [SOPHGO TPU-MLIR CV18xx 指南](https://doc.sophgo.com/sdk-docs/v25.03.01/docs_latest_release/docs/tpu-mlir/quick_start/html/Appx.02_cv18xx_guide.html)：ONNX/INT8/BF16 → `.cvimodel`，具体 SG2002 处理器 ID 和支持算子以板上 SDK 验证为准。
9. [Self-Monitoring, ICLR 2019](https://arxiv.org/abs/1901.03035)；[LookasideVLN CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ning_LookasideVLN_Direction-Aware_Aerial_Vision-and-Language_Navigation_CVPR_2026_paper.html)；[QAD TCSVT 2026](https://doi.org/10.1109/TCSVT.2026.3727199)；[LightVLN 2026](https://arxiv.org/abs/2610.05024)；[AirAnchor 2026](https://arxiv.org/abs/2609.08442)。

### 12.3 本稿证据强度说明

**已由公开官网核对**：MiniWorld 发布包体积和废弃公告、BabyAI 的任务/RGB wrapper、AerialVLN-S 标注与约 35GB 模拟器分离、Sipeed SG2002 规格、CHAI/LANI 论文段落数、AVDN xView 依赖、IndoorUAV 规模。**根据用户材料与先前审查沿用**：V5 的源码行为和八动作；当前未在 RTX 5090、AirSim、MiniWorld、SG2002、ESP32 上进行实际测量。**新建议**：TinyWorld-Lang-v0 子集数、ECPT 事件确认式状态、文件数量/行数门禁、工作时间盒等。这些不应写为已有研究事实。

---

## 附录 A. 开始写代码之前的 20 项检查（本稿建议的真实工作顺序）

- [ ] 已读取并固定 V6 方法边界，不将新项目退化为“无语言避障”。
- [ ] 已为 D0 选择正确 BabyAI RGB wrapper，语言 mission 未被丢弃。
- [ ] 已验证 MiniWorld 2.1.0 在 Ubuntu 上能 reset/step/取第一视角 RGB。
- [ ] 已记录 MiniWorld deprecated 风险和退出方案。
- [ ] 已将 TinyWorld-Lang-v0 明确标注为自建，而非官方数据集。
- [ ] 已按场景布局与路线分组形成 train/dev/test 清单。
- [ ] 已验证“沿指定地标顺序”专家路线而非无指令最短路径。
- [ ] 已将每个学生在线特征纳入观察白名单。
- [ ] 已单独保存训练特权标签并用 mask 控制损失。
- [ ] 已测量词长分布和真实 token 截断率，完整保留任务尾部。
- [ ] 已固定动作枚举与各平台独立 ActionProtocol，D2 STOP=0。
- [ ] 已复核 SG2002 PCB、SDK 和可编译的 CNN INT8 probe。
- [ ] 已将 Python/平台依赖限制在 adapters 和测试工具而非正式训练。
- [ ] 已完成独立 numerical oracle + gradcheck。
- [ ] 已验证 episode reset、padding mask 和 resume 一致性。
- [ ] 已保留语言打乱/相反指令/地标反证的消融。
- [ ] 已把 KD 列为教师实际验证优于学生后的增量分支。
- [ ] 已约束 `code/src ≤24`、`cuda ≤6`、`headers ≤10`、文件行数与依赖方向。
- [ ] 已为 SG2002–ESP32 定义 TTL、ACK、完成反馈、人工接管与飞控端安全。
- [ ] 已知“小平台成功”“正式 Aerial 得分”“真实飞行安全”是三种不同证据。
