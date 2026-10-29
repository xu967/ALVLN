# TinyAerialVLN V5.0：七份评审综合复核与完整研究方案

> 日期：2026-10-28  
> 性质：**研究设计、证据审查和实施规格**；不是已经训练、跑分、板端部署或真实飞行的实验报告。  
> 项目约束：单目 RGB + 自然语言 + 合法历史动作/内部记忆；**学生网络不接 IMU、深度、地图或真值位姿**；RTX 5090 训练；LicheeRV Nano（SG2002）部署；UART 对接 ESP32 已有飞控；学生主训练实现以专用 C++/CUDA 为主，禁止把 PyTorch 包一层 C++ 冒充自研。  
> 本次审查依据：上传的 V3 原稿、V4 执行稿和五份独立评审；本地 `web_code_review_20261027.zip` 的部分关键源码；并核对公开 AirVLN 仓库、ICCV 2023 论文入口、2025–2026 文献、Sipeed/TPU-MLIR 官方资料。  
> **所有未在用户机器上执行的内容均属计划**，所有板端 FPS、RAM、SR、可编译性及真实导航能力，均不得写成实验事实。

## 0. 最终决策摘要

本项目应分为**三条相互约束，但不混淆结论的实验轨道**：

- **A 轨：AerialVLN-S 标准语言导航**。主要学术证据；遵守官方离散 8 动作、场景、观测默认参数和 SR/OSR/NE/SDTW 计分。学生只能看 RGB 与指令，禁止把官方 RGB+Depth 成绩当作同输入公平基线。
- **B 轨：SG2002 板端闭环软件/HIL**。验证训练模型能从真实摄像头读帧、NPU 执行、CPU 更新记忆、UART 发送有时效的动作，完整测量尾延迟、内存、动作一致性及故障处理。板上跑了不代表无人机能飞。
- **C 轨：受控连续运动 / 真实迁移**。仿真 ESP32 等价控制接口、短时安全运动、视觉时延、姿态/定高能力、接管机制。第一篇主论文可先交付 A+B；**没有 C 轨合格证据绝不能宣称实机自主导航成功或避障安全**。

**核心方法建议**：先做可复现的轻量 RGB-only VLN 基线，之后优先研究**“可验证视觉证据驱动的指令阶段/终止联合决策”**，尤其是 STOP 与中间事件误判、记忆更新是否相互干扰。在 SG2002 固定资源预算下比较“紧凑显式状态”与“等容量普通 GRU / 注意力”的收益。主动 ROI / 非均匀采样为明确可删除的后续支线；蒸馏是提升方法而非默认主创新。

这一定义避免了两个陷阱：(i) 为一个复杂但不确定的数据前提（高清 ROI）孤注一掷；(ii) 因 STOP 重要，就把复杂多阶段语言理解删掉，变成另一种障碍分类。

## 1. 七份材料的交叉复核：哪些评审是对的，哪些需要修正

### 1.1 高优先级问题判定

| 争议 | 多份评审给出的说法 | 本次结论 | 执行修订 / 验证 |
|---|---|---|---|
| 48-token 上限 | 可用 / 大幅截断 | **采纳批评**：官方论文的平均指令词数远高于 48；token 数须按实际 tokenizer 统计，不能从平均词数直接计算精确截断率 | 原始 train 统计 P50/P90/P95/P99/最大；建议初始 192 word tokens 或分句全覆盖；冻结预处理与词表 |
| 2k 词表 | 足够 / 严重 OOV | **风险成立，但不能未经统计断言 OOV 比例** | 训练集构词表，统计 unseen OOV；尽量保留颜色、方位、数词、地标词；按体积和性能选 word/subword |
| 224×224 | 是不可改的协议 / 完全可任意调整 | **都过强**：官方代码 RGB 默认 224×224；提高渲染尺寸是自建观测条件 | A 轨忠实复现官方默认配置；HD 新建 B-obs 协议，所有模型重跑；不得拿 HD 得分直接声称超越默认公开基线 |
| 官方有逐帧高清图 | 可直接下载 / 只能再渲染 | 现有官方 README 只明确标注及模拟器分发，RGB 经 AirSim 运行时抓图；无已验证的官方全量高清序列缓存 | 先跑官方渲染/采集试验，留帧、位姿、图像配置与重放哈希；失败则暂停 ROI |
| LAG 是 5.1% SR 的专家 | 是 / 否 | **否**：LAG 是辅助生成训练引导动作的策略；发表的数值是使用 LAG 训练的 CMA 模型表现，不能给几何教师冠上该 SR | 定义 T0“基于参考路径的动作标签服务”、T1“实际训练出的策略教师”，各自评估适当指标 |
| STOP 很重要 | 因 OSR≫SR，所有失败都由 STOP 主导 | **一半对**：OSR−SR 仅指“曾进入阈值范围但未成功在那里 STOP”的 episode；没有进入目标区域的失败无法靠 STOP 解决 | 单独报告 `P(reach radius)`、`P(stop correctly|reached)`、early/late/no STOP、NE 与路径偏离 |
| NPU 计算几乎免费 | 1 TOPS 所以 CNN 0.1–3 ms | **不成立为实测结论**：参数、MAC 和理论吞吐不能推断调度、DMA、算子分解、相机到 UART 的 P95；也不能武断断言算力一定是瓶颈 | 实板 P50/P95/P99、PMU、热稳态、模型 RSS/ION、camera→UART 分段计时；以测量决定优化对象 |
| 单次 fovea 一定优于双次 ROI | 硬件绝对优选 | **候选，不是定理**：6 通道输入、额外插值、取景时间滞后及 NPU 支持都可能有成本 | 先比较恒定 3 通道全图、3 通道拼接画布、双次同 shape ROI；确认像素/延迟预算匹配 |
| 进度标签 | 官方必有 / 官方必无 | 论文写有子路径对齐，但**不等于当前发行 JSON 已包含字段**；不同评审对字段审计说法不完全一致 | 对具体下载版本全量枚举字段；找不到不允许宣称官方监督，改为弱监督或仅学习隐式状态 |
| 单调指针无需标签 | 只用单调约束就一定有效 | 不能保证：指令可重复、偏航、重访、错误地标识别，硬单调会锁死错误阶段 | 比较 unconstrained、soft monotonic、可回退的事件确认三类状态机；对每种加 STOP 诊断 |
| 教师和蒸馏 | 先造 72B 教师 / 立即做 KD | 这两者都不合理 | 先专家标签 BC/有限次偏轨纠偏；T1 教师只有在合法 RGB-only 闭环且明显优于学生才用于 KD |
| 评测只看 SR | 531 例无法做任何方法比较 / 可用单一百分点阈值宣布胜负 | 均不科学；配对差、每轨迹 3 指令聚类、仅 5 unseen 场景、种子方差都重要 | 全量 val_unseen、配对 cluster bootstrap、scene-wise 结果与稳健性分析；SR 与连续指标并报；不预设“<3.7pp 一律无效” |
| 自研 C++/CUDA | 造完整通用框架 / 放弃自研转 PyTorch | 两个极端都不采纳 | 专用静态图、显式反向、独立数值 oracle、数值和导出 parity；用第三方库做 GEMM/编译器不影响系统自研 |
| ACK 收到后才能拍下一帧 | 必须串行阻塞图像采集 | **不采纳**：ACK 表示接收，不表示动作完成；封锁采集会丢失实时观测 | 分离“命令收到 ACK”“执行完成 EXEC_DONE”“状态估计/安全反馈”；相机持续采集但决策有序，禁止旧指令覆盖新状态 |
| 8051 做看门狗 | 可直接使用 | 只能作为潜在工程探索，不能默认有可用固件与串口控制路径 | **安全主权必须在 ESP32**（TTL、降级、接管）；低功耗核仅在官方 SDK 与验证完成后选用 |

### 1.2 三项特别需要避免的计算/逻辑错误

1. **531 条指令和 177 条路径不能随意互换统计单位。** 1% × 531 ≈ 5.31 条“指令 episode”；若按独立参考路径算约 1.77 条路径，但一条路径的三种指令不是完全相同的随机变量。评审中“1pp≈1.8 条 episode”只有特定路径级归约才能成立，不能用于指令级 SR 的直接计数。
2. **标准独立二项置信区间不是已配对、多种子、场景聚类实验的“普适可分辨效应”**。例如 p=5%、n=531 时，简单 Wald 95% 半宽近似 1.85pp；这不是对任意两模型 ΔSR 的显著性阈值。应从逐 episode 配对差与聚类结构估计不确定性。
3. **STOP 上界不可包装成一套在线 Oracle 模型结果。** 沿已经生成的轨迹在最佳时刻停止是诊断（类似 OSR），不能证明真实学生可在当时知道已到达，或真实无人机安全地停止。

### 1.3 已查证的旧源码相关事项（有条件）

在 `/mnt/data/web_code_review_20261027.zip` 中，本次重新查看 `source_snapshot/uavnav/avln/data.py` 的 `preprocess_rgb()`：确实使用 PIL `resize((imgsz,imgsz), Image.BICUBIC)`；`airsim_env.py` 存在 `simSetVehiclePose(..., ignore_collision=True)` 调用。它们支持“旧工程有正方形预处理、部分流程使用传送式位姿设置”的结论，但**不支持“每个流程、每种评测都因此没有碰撞”的无限推广**。要将其用于新论文，需要固定调用路径和评测配置。新方案的 48 token 等是 V3 的拟议值，而非在这次源码核查中认定的旧代码现状。

## 2. 任务定义：先完成“语言导航”，再单独验证“物理自主飞行”

### 2.1 A 轨正式任务（可发表的受控问题）

输入：`o_t = (RGB_t, instruction, compressed history, previous action)`；输出官方 8 动作 logits（与官方 enum 顺序一一映射）；约束：没有在线深度、IMU、GPS、真值位姿、目标坐标、参考路径或未来帧。

AerialVLN-S 使用离散 5 m 级平移、15° 级转向、2 m 级升降以及 STOP 的抽象步进。它可验证**多阶段语言理解、视觉地标证据与历史决策**，却不等价于适用于真实微型四旋翼的安全轨迹控制。

### 2.2 B 轨：LicheeRV Nano 部署

固定同一个学生模型权重和 tokenizer 版本，测相机实际采集/预处理、NPU、CPU 文本与记忆、策略输出、UART 写入、ESP32 ACK/完成反馈。板端模型版本、PCB、SDK、固件、温度/频率、校准集哈希全部随实验归档。

### 2.3 C 轨：受控物理化与安全

首先在独立物理仿真里定义与 ESP32 一致的动作：有限倾角、有限偏航速率、有限控制持续时间；只在飞控具备可信定高/姿态稳定及已验证的安全降级能力后探索。在 C 轨不能简单把 A 轨的 `Forward 5 m` 等同于串口“向前”命令；这会导致未建模的加速、漂移、延迟和高度损失。**没有可靠飞控保护，最终仅做无桨台架和相机离线回放。**

## 3. 2026 相关工作的真实竞争位置

| 工作（官方链接） | 已有贡献 | 对 TinyAerialVLN 的限制 | 应做的对照 |
|---|---|---|---|
| [AerialVLN, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html) | 任务、CMA、LAG、DA、基准 | 不能把训练引导和 8 动作协议当新意 | 同 RGB-only 的 Seq2Seq/CMA/LAG 训练版本 |
| [Self-Monitoring, ICLR 2019](https://arxiv.org/abs/1901.03035) | 指令—视觉 grounding + progress monitor | 单纯“进度监督”没有新意 | 标准进度回归/软指针 |
| [AdaFocus, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Adaptive_Focus_for_Efficient_Video_Recognition_ICCV_2021_paper.html) | 全局粗看 + 局部精看 | 不能宣称 ROI 思想原创 | 固定、置信度、语言条件裁剪 |
| [Input-Adaptive VLN, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Harnessing_Input-Adaptive_Inference_for_Efficient_VLN_ICCV_2025_paper.html) | 空间/时间/退出自适应推理 | 不能把动态计算本身当新意 | 固定帧率、置信度触发缓存等 |
| [LookasideVLN, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ning_LookasideVLN_Direction-Aware_Aerial_Vision-and-Language_Navigation_CVPR_2026_paper.html) | 地标与方向图及知识库 | 地标方位账本也不是未经尝试的思想 | 紧凑记忆/方向先验的等容量版本 |
| [QAD, IEEE TCSVT 2026](https://doi.org/10.1109/TCSVT.2026.3727199) | 全局/局部 query、指令对齐、证据与动作蒸馏 | “蒸馏哪里看与怎么做”不是新意 | 低复杂度 query/action KD 对照 |
| [AirAnchor, 2026-09 arXiv](https://arxiv.org/abs/2609.08442) | 空间锚点、持久化语义记忆、进度反思 | 阶段 + 地标证据组合已拥挤 | 结构化状态对比：无图 vs 小状态 |
| [LightVLN, 2026-10 arXiv](https://arxiv.org/abs/2610.05024) | 压缩历史、历史/指令局部聚合，0.5B 语言主干，Orin NX HIL | “轻量 + 历史 + 局部聚合”不新；0.5B 对 Nano 仍很大 | 学生容量级、板端真实预算、输入与协议差异 |

**新颖性边界**：目前没有足够证据把“单调指针/地标方位/ROI/KD”任一要素单独宣布为新算法。可提出**一个受硬件严格约束、针对错误进度与终止的可观察证据更新机制**并作等参数、等输入、等训练数据的因果消融；即使如此，投稿前仍需检索最新细分工作并检查方法同构。

## 4. 数据、环境、协议：先完成两个可行性探针

### 4.1 AerialVLN-S 官方数据

论文给出的 AerialVLN-S 分布：train 10,113 条指令（12 场景）；val_seen 333；val_unseen 531（5 个新场景、177 条参考路径）；test 论文表述 771。部分第三方/下载版本出现 test=770，**必须本地审计下载包的真实计数与过滤口径**，且不可默认存在可公开提交 test 的评分服务器。正式结果以可执行、可审核的 val_unseen 为锚点。

论文报告指令长度平均约 82 词，显著高于 V3 的 48 token 候选。实际 token/子词数量要由正式 tokenizer 全量统计。`train` 中同一参考路径的 3 条指令要保持同一分组；原始路径/场景 ID、样本哈希、重放版本、随机种子全部记录，避免帧级随机划分泄漏。

### 4.2 默认观测与高清观测必须双轨

**Protocol A（默认重现实验）：** 按官方开源客户端默认 RGB 224×224、对应相机/FOV/动作/采样重建输入。所有新 RGB-only 基线都在此协议下重新训练与测试。官方已发表 RGB-D 基线只作模态不同的参照，不进行不公平 SOTA 宣称。

**Protocol H（自建高清/保比例试验）：** 在确认 AirSim 场景允许并正确输出高分辨率 RGB 的前提下，记录原始宽高/FOV/畸变并按保比例 resize/letterbox 或空间采样转换。H 轨中的固定输入尺寸 320×192 / 384×224 仅是候选画布，必须全部配备重新训练的 H 轨基线。H 轨仍可使用官方的 episode 和计分公式，但**“观察协议”已经改变**，不能与 A 轨公开分数作一行直接横比。

### 4.3 双探针（最高优先级）

**G-A：仿真可运行性。** 在目标 Ubuntu/RTX 5090 机器上验证官方约 35 GB AirSim 场景能否启动；必须以默认观测跑完 10–20 条随机策略闭环 episode，能正确 reset、渲染、反馈 8 动作、计算 NE/SR。检查 RGB/BGR 与摄像头配置，并测每条 episode 耗时、显存、帧时间戳及回放一致性；HD 能力作为额外可行性测试。**UE4 旧二进制兼容性未实际验证**；Docker 不能保证修复 GPU/图形 ABI 问题。失败则考虑独立旧系统/兼容机器，或更换基准并声明更改任务，不能靠离线 accuracy 冒充原 benchmark 闭环。

**G-B：板端能力。** 查 PCB 70405/70415、SoC 固件、Linux CPU 模式、官方相机接口/ISP、TPU SDK 版本；从最小 Conv→`.cvimodel`→INT8 在线推理开始，测试 1/3/6 输入通道、DWConv、残差、Pool、多输出、静态形状；GRU/动态 If/Loop 暂不放 NPU。记录相机采集到 UART 的真实延迟与内存水位。在 G-A、G-B 都通过之前，**不写完整 ROI 或大型自研训练器**。

### 4.4 数据获取与缓存

第一批仅获取 AerialVLN-S 标注与必要场景；先 Debug-Overfit（16–64 条训练指令，按路径组），然后 mini-train（500、2,000 条训练指令），最后全量 10,113。Mini 用于程序正确性/模型学习迹象，不能据此宣布泛化性能。

每一张采集图必须伴随 `scene_id/episode_id/step/id/timestamp/action_protocol/original_size/FOV/render_config`，并校验相机颜色通道及与目标路径对应关系；图像中间缓存和评测轨迹都要存版本号。高分辨率缓存只有在 G-A 验证可重渲染并且确有额外像素信息时才建立。

LANI 只可作为可选、一次性的 tokenizer/序列 reset 语义测试，不强制维护旧 Unity/Python2.7 运行栈；优先以 AerialVLN-S train 内部构建 smoke set，减少第四套环境风险。OpenFly 保留后续跨场景/不同动作协议外推，不在第一轮同步接入。

## 5. 新主方法：Evidence-Calibrated Progress & Termination（暂名 ECPT-VLN）

> 本节是**候选新设计，不是已证明的原创成果**。关键目标不是把三个已有模块相加，而是用可验证、可撤销的证据更新压缩状态，研究在长指令/少参数约束下何时推进阶段与何时 STOP。若等容量消融不优于传统 GRU/进度回归，则删除机制，不包装负结果为创新。

### 5.1 第一阶段最小学生 B1

- **视觉**：官方观测 `[B,3,224,224]` → Conv(3→16, stride2) → 轻量 DW/PW (16→24→32→48, 后三次 stride2) → `[B,48,14,14]` → pool+1×1 到 `[B,64,7,7]`（49 个局部区域特征）；BatchNorm 在部署端折叠。输入 384×224 时同类 CNN 对应 `[B,48,14,24]`，需在另一协议单独训练。
- **语言**：从 train 冻结词表，Embedding(32) + 轻量文本编码器(64)；初始完整长度预算 L=192，但先由分词统计决定最终上限。只在 episode 开始对文本编码一次，缓存 token-level 64 维特征；需要保留尾部目标和 STOP 条件，不做强制前 48 词截断。
- **跨模态**：视觉区域与当前指令的低秩/单头轻量相关性（CPU 侧运行，明确 mask、方位编码、有效像素）；必须证明输入变化时行为确实改变。
- **记忆**：单 GRU(64)，输入融合特征 + 上一次**合法已发送动作**，episode reset 必须清空。
- **动作**：官方 8 个 logits，训练时对 STOP 做类不平衡处理；首阶段保持原协议。

参数预算建议首先目标 `<1M`，不是所有部分都强行降到 10k 权重；整机可用内存、CPU 工作集和 NPU 显存分配必须另核。相比只报告参数量，更重视全链路板端是否跑得动。

### 5.2 ECPT 的核心机制（在 B1 完整闭环之后）

**输入**：当前局部视觉证据 `z_t`、完整文本 token 缓存 `E`、上步动作、历史状态。**内部结构**：

1. **阶段信念 `b_t(k)`**：对指令的 K 个顺序 span/子事件加一个 `DONE` 状态（K 不是强制按逗号数）。训练期仅用经确认的路线事件或弱对齐；未确认时可完全不加阶段监督。学生在线不能得到未来路径进度。
2. **证据账本 `m_t`**：只保留固定小维度（如 32–64）的视觉证据摘要：刚刚观察到的地标/方向、证据置信度、最近动作。初期不执行全局几何地图/真实三维距离回归；可只用方位分桶和观测次数。
3. **阶段更新**：比较 `b_t` 与观测匹配度，生成“保留当前阶段/进入下一阶段/证据不足暂缓”的门控；不同于硬单调 pointer，在虚假地标、绕行和重访时允许低置信度候选被推翻，但不随意回到已经可靠完成的早期阶段。
4. **终止判断**：独立 `p_stop` 二分类头，条件是当前视觉证据 + 终端指令阶段 + 历史转移；使用固定 dev 上校准的阈值和短时稳定确认（例如连续几帧，而非零延迟强制 STOP），输出仍遵循 8 动作 benchmark；将误触发、漏停、经过目标但未停分开归因。

**为什么可能有研究价值？** 传统小 GRU 可能把视觉地标、进度与终止压进同一个小状态；ECPT 显式分析“观察是否足够推进语义状态”，避免一次误识别导致后续长程错误。但相似的 Self-Monitoring、AirAnchor、Lookaside 和 QAD 已经研究过相关结构；只有对**低容量、错误证据、终止行为三者的联合机制**做出明确新公式并通过相同参数量对照，才有可能声称方法增量。

### 5.3 可检查的训练损失（递增，不一次叠加）

- **B1**：`L_BC = weighted_CE(action_logits, action_gt)`，STOP 权重仅在 train/dev 标定，不能在 val_unseen 调。
- **B2（终止）**：`L = L_BC + λ_s BCE/Focal(stop_head, stop_gt)`；STOP 标签来自合法当前/过去及训练监督，终止距离信息可以只作为损失，不在线输入。
- **B3（阶段）**：`L + λ_e L_event + λ_c L_consistency`，其中 `L_event` 只使用可信的阶段弱标签；`L_consistency` 约束同一事件的相邻视觉片段与动作具有一致证据。事件监督不可信则置 λ_e=0。
- **B4（KD，可选）**：在独立验证比学生更强的教师策略出现时 `L + λ_k T^2 KL(softmax(teacher_logits/T) || softmax(student_logits/T))`，配合教师不确定性门控。不能用弱教师输出的 attention 强行充当真值。

优先记录各损失的梯度范数、梯度相互冲突、STOP 召回/精度、学到的阶段转移概率。禁止因为加更多 loss 而忽视闭环退化。

### 5.4 STOP 的科学归因设计

对每个 episode 分组：未进入 20m 区域；进入 20m 却在外停止；进入后离开且超出时间/步数；过早停；合法停在区域。报告：`SR`、`OSR`、`OSR−SR`、`P(stop_in_radius | ever_enter_radius)`、NE、SDTW，以及每类错误的改变。改变阈值有时会牺牲路径/终止行为，因此不能只挑最高 SR。

## 6. 可选观察研究：先问数据有没有额外信息

**G-HD 前提**：模拟器重渲染确实生成新的高频细节；验证不同高清图与低清图的像素级对应。完成以下 3 个初步试验：

- HD 全图缩至固定 224/320/384；保持几何一致性与各协议内基线对照；首先判断纯增大有效分辨率是否值得。
- 固定 ROI / 固定 fovea；观察离线机制指标是否有收益，且与短期成对闭环回放收益相关，而不是只凭教师 CE 下降。
- **单次采样画布**：建议最初使用 **3 通道静态输入** 的拼接画布（左半全局、右半更精细的 ROI），避免未经实测的 6/7 通道 NPU 假设；ROI 的位置/尺度以小向量交给 CPU 融合模块。必做同参数、同宏观视野、同推理次数、同实测 P95 的均匀采样对照。

**隐藏的逻辑陷阱**：如果 ROI 是根据当前帧图像语义选择，而 NPU 又只允许一次前向，不能先用同一个模型看当前图再决定如何裁当前图。须明确 ROI 来源为 `t−1` 的历史预测/指令方位先验，或者增加一个额外便宜的当前帧窥视步骤并计入成本。还要测快速相机转向时的 ROI 跟踪丢失率。若“oracle ROI”从未来 GT 获得信息，只能用于离线上界分析，不能进入在线对照。

**否证规则**：若高清帧不可得、固定 ROI 不改善短程闭环、单次画布同预算不优于全图、或者自适应比固定/置信度策略更差 → 删除观察分支；不要强行“用更多蒸馏救 ROI”。

## 7. 教师/蒸馏的正确顺序

- **T0（数据标签器，不是训练出的大模型）**：沿参考路径的专家动作与 LAG/偏轨引导的生成规则，能对离轨状态提供修正监督。但它读取真值，无法算作真实 RGB-only 策略；其“Oracle-STOP”也只用于诊断。
- **T1（必须有独立闭环报告的视觉教师）**：现有 DINOv3/GRU、官方 RGB-only CMA 或可训练大型视觉语言教师任选一条成本可控路径；要求在相同 RGB 输入规则的完整 train 和 val_unseen 上得到逐 episode SR/NE/SDTW 后才考虑作为蒸馏教师。教师精度和稳定性比参数量重要。
- **S（学生）**：单目 RGB + 指令 + 合法过去输入；先 BC、有限轮纠偏，再对 T1 动作分布 KD；必要时用 train-only 目标标签做特权辅助监督。学生可以先只蒸馏动作和 STOP 置信度，视觉 query 蒸馏应对 QAD 做等价对照后再引入。

教师如果没有稳定的正收益，**主方法不能依靠 KD 才成立**。先让 ECPT 在没有 KD 时超过同容量 GRU/标准进度监督；否则研究核心尚未建立。

## 8. 自研 C++/CUDA：只造我们必要的算法与执行系统

### 8.1 从底层实现的正确目标

自研：TensorView + shape/strides 与生命周期、静态图 Executor、Conv/DW/PW 与 Linear 的前后向调度、Embedding/GRU/单头融合的显式梯度、mask/CE/BCE/KL、AdamW、episode Loader、动态序列截断与 reset、checkpoint 恢复、数据/结果 manifest、ONNX 导出，以及与板端相同的文本/几何预处理规则。

复用：编译器、CUDA Runtime、cuBLAS/cuDNN、OpenCV、Protobuf/ONNX 标准、官方 TPU-MLIR 编译器与 SG2002 SDK；必要时可在仅测试目录使用 NumPy 或 PyTorch 生成**只用于比对**的 golden tensors。正式学生优化/训练和板端运行不调用 PyTorch；预训练视觉教师允许冻结离线特征生成。

### 8.2 推荐的开发次序

1. CPU 标量数值正确性：Linear、Embedding、CE/BCE、Conv/DW 与反向；有限差分、独立 NumPy golden 测试。
2. 加 GRU + mask + sequence reset + checkpoint 恢复；能在 16–64 条 train 指令做确定性 overfit。
3. GPU 静态图与 cuBLAS/cuDNN 实现；层级 parity、混合精度差异和跨设备种子不要求 bitwise 完全相同，但要能解释。
4. 最小视觉 CNN ONNX 输出及两端 parity；用真实 SG2002 `cvimodel` 验证层输出、INT8 calibration/STOP 分布。
5. 只有 B1 在 Aerial 官方闭环有效以后才加 ECPT；后续仅按消融缺口补写算子，不构建自动图编译器、多 stream 通用内存调度或自家 TPU 指令生成器。

### 8.3 强制测试合同

- 一切输入都用同一 schema：通道顺序、颜色空间、letterbox/resize、归一化、padding mask、tokenizer、动作 enum、episode reset、frame age。
- `Gradcheck`：小尺寸 float64 有限差分；静态前向 vs 独立数值参考；关键算子注入人为错误时应能检测。
- `Sequence parity`：整段 vs 逐帧更新 logits 对拍；reset、两条 episode 同 batch 不串状态；GRU 隐态量化积累误差检查。
- `Deployment parity`：CUDA FP32 → ONNX → INT8 `.cvimodel`，比较 8 logits、STOP logits、顶层动作、整条闭环轨迹，而不只看 Top-1。

**禁止**：将“完全零第三方库”作为学术创新；如果 3–4 周仍无法让静态学生在小集上稳定训练，可保留 C++ 关键模块与板端部署，自研训练主链按能力砍算子，不能让完整课题永久卡死在手写 BLAS。

## 9. SG2002 平台预算与验证

官方 SG2002：主核 1GHz C906 或 A53 二选一、辅助 C906 700MHz、256MB DDR3、INT8 标称 1 TOPS，支持 BF16；70405 早期 PCB 在 1T 下稳定性不保证。硬件算力是峰值指标，**对任何网络实际 FPS 都没有保证**。

### 9.1 初始约束（工程目标而非保证）

| 项目 | V5 起点 | 实测后决策 |
|---|---|---|
| 总学生参数 | 优先 ≤1M，允许 ≤2M 的消融上界 | 由编译成功、内存、准确率与吞吐联合选取 |
| 静态视觉图 | batch=1，224×224，首选 INT8 | DW/PW/残差、输出 taps 必须实际编译 |
| 常驻系统内存 | 初始希望关键服务峰值 RSS ≤ 128MB；全系统可用内存需单独测 | 数值非承诺，不能从总 256MB 减一个常数推导 |
| 全链延迟 | 先争取 camera→UART P95 ≤ 250ms（实验目标） | 真实可安全速度须根据延迟、制动和实际控制能力另定 |
| 决策更新 | 初期 2–5Hz 的 HIL 软件基线 | **不能把这当成真实飞控必须的速度或安全保证** |
| 时间稳定性 | 连续至少 30 分钟无错帧、积压、热失稳 | 记录前后温度、RSS、P95 变化 |

先测采集、ISP/VPSS 缩放、DMA、NPU、CPU 文本/融合/GRU、UART 排队、ESP32 ACK、完整被动端到端通路的 P50/P95/P99。允许使用真实硬件图像的 LED/GPIO 校准测量相机到 UART 物理时延；性能必须附 PCB、固件、输入、功耗/热、计时方法。

### 9.2 量化

- 先完成 BF16/INT8 的兼容性和性能小探针；不要假设 GRU/attention 因能被 ONNX 前端识别就能被 CV181x NPU 后端执行。
- 把静态 CNN 放 NPU，缓存文本、门控、GRU、动作状态机先由 CPU 执行；若 CPU 成瓶颈再评估合并或优化。
- 用训练场景中的真实分布样本校准 INT8；测试场景不得进入量化校准集。
- STOP 置信度特别容易因量化发生阈值漂移，阈值必须在合法 dev 上重新校准并报告。

## 10. UART/ESP32 与软件在环规格

**本节必须独立于 Aerial 八动作，且不可把“STOP”解释成切断电机。**

推荐消息概念字段：`version, seq, command_type, bounded_value, duration_ms, expires_at, crc`；接收端返回 `ACK_RECEIVED(seq)`、`EXEC_DONE(seq,status)`，以及已验证的安全状态。系统只保留一个未完成运动命令，图像采集持续发生，动作发射须有执行时序/过期逻辑。相机帧超过年龄阈值、串口心跳丢失、ESP32 姿态/控制异常时进入其**已经实现并验证的**安全处理（不能假设无定高还能悬停）。无桨台架先验证 TTL、重复帧、乱序、CRC、Linux 卡死和人工接管优先级。

A 轨评测仍使用官方离散动作；C 轨另写动作适配层和环境合同。不可把 benchmark 的 20 m 成功半径宣传为真实避障精度，也不可把 HIL 设备推理宣传为实体机自主飞行成绩。

## 11. 实验矩阵：优先保留可归因的 6–8 组

| 编号 | 模型/设置 | 关键研究问题 | 通过判断 |
|---|---|---|---|
| E0 | Random/Action Frequency/STOP 固定基线 + 官方评测烟测 | 评测器和动作分布正确吗 | 结果可复现，官方协议动作与指标正确 |
| E1 | RGB-only Seq2Seq/CMA 复现（必要时沿用公开基线训练代码） | 模态公平参照点 | 完整闭环可复现；不要求机械地等于公开数字 |
| E2 | 自研 Tiny 基线（CNN+完整 Text+单 GRU） | 能否仅靠小学生学习 VLN | 相比 E0 有稳定导航表现；语言干预有效 |
| E3 | E2 + 独立 STOP head | STOP 是否是可改进瓶颈 | 条件成功率、提前/漏 STOP、SDTW/NE 无明显反向退化 |
| E4 | E3 + ECPT 证据确认式阶段记忆 | 是否优于等参数 GRU / 自监控式进度回归 | 重点比较多阶段、误识别、STOP，且整体指标不退化 |
| E5 | E4 + 有限纠偏数据（DA/LAG） | 分布漂移是否是障碍 | 相同训练预算多种子对照 |
| E6（条件） | E4/E5 + 有效视觉布局或 KD（二者分开做） | ROI/KD 有独立净收益吗 | 通过 G-HD 或教师门禁，预算匹配 |
| E7 | 最优模型 FP32/BF16/INT8 + SG2002 HIL | 能否真实运行且闭环行为保真 | 同权重/同输入逐层和逐 episode 比对，通过资源预算 |

实验量：调试 16–64、mini 500/2,000 仅检查可训练性；主结果全量 10,113 训练，val_unseen 531 作为最终关键比较。每个关键模型至少 3 个 seed，建议 5 个；同路径/episode 配对、报告各个 unseen 场景、路径/指令长度分层与 CI。场景仅 5 个时，scene-level bootstrap 非常不稳定，应报告 leave-one-scene-out 和明确外推限制。teacher-forcing 的 step accuracy 可作机制诊断，**不能代替在线 SR**。

**val_unseen 不应用于无限次阈值扫描**：对训练内部 held-out/dev 选择超参、STOP 阈值和模型；官方 val_seen 作为辅助观察；最终将 val_unseen 用作有限且预登记的确认评测。所有训练/参数搜索决策均记录，避免基准集过拟合。

### 11.1 必做的反证实验

1. 同一图像和历史，交换指令：若动作/阶段无有意义变化，语言分支没有生效。
2. 同一指令，改变可见地标方向：若预测不变，空间 grounding 没生效。
3. 删掉历史/随机重置状态：若完全无影响，所谓记忆可能是装饰。
4. 假阳性地标干扰、折返、重复地标：检验 ECPT 是否真的比硬单调指针稳健。
5. 对照 `no-stop-head / separate-stop / separate-stop+ECPT`，区分终止分类收益与阶段收益。
6. teacher-only、student-only、KD：确认 KD 在任务指标上有净增益，不只 logits 接近。
7. A 轨 vs HD 轨不能混排名；INT8 vs FP32 必须同步报告闭环变化、延迟和内存。

## 12. 实施门禁、排期和优先级（单人 + 一张 5090 + 一块 Nano 的粗略估计）

| 里程碑 | 规划工作量 | 交付物 | Go/No-Go 与替代路径 |
|---|---:|---|---|
| M−1 / M0 双探针 | 1–2 周 | AirSim 随机闭环日志；SG2002 CNN `cvimodel`、整链 trace | 任一失败先解决环境，不进行主线模型实现；旧 UE4 不兼容可换独立机器/系统 |
| M1 数据合同 | 1 周 | JSON 字段/动作 enum、token 长度、test 770/771、预处理全审计 | 缺图像则先渲染；禁止随机划分泄漏 |
| M2 专用训练 MVP | 3–5 周 | CPU/CUDA parity、梯度检查、16–64 overfit、ONNX | 训练器受阻时缩减算子和网络，不造通用自动微分 |
| M3 公开基线+Tiny 基线 | 3–5 周 | RGB-only baseline、E2 val_seen/dev + val_unseen、失败归因 | 若 Tiny 未优于随机且不依赖文本，不启动 ECPT |
| M4 STOP/ECPT | 3–4 周 | 配对多种子实验、阶段错误和 STOP 诊断 | 无独立收益则删除主机制或回到最简学生 |
| M5 KD/ROI 条件分支 | 2–4 周可选 | 教师闭环、高清上界/固定 ROI、板端代价 | 未达门禁即不做，不挤占 A+B 主实验 |
| M6 SG2002+UART HIL | 2–3 周，与前面交叉 | 30分钟热稳态、P99、模型量化闭环 parity、ESP32 故障注入 | 不通过仅报告离线 benchmark；不进行自由飞行 |
| M7 物理化验证 | 另立项目 | 连续运动仿真与定高/接管证据 | 这不是 AerialVLN 主论文成绩，不能偷偷合并 |

时间是工作量预估而非未来交付承诺，实际受 GPU/仿真兼容、板卡固件、个人开发经验影响很大。**第一条实际执行任务是双探针，第二条是数据/动作合同，第三条才是自研 Tensor。**

## 13. 环境拓扑与依赖隔离

- **Ubuntu 24.04 主机/RTX 5090**：NVIDIA 驱动、与 Blackwell 兼容的 CUDA、CMake、Ninja、OpenCV、C++/CUDA 编译器；自研学生训练。
- **旧版 AirVLN 仿真沙箱**：按官方 Python 3.8 + AirSim 1.7.0 等版本锁定，用独立容器/conda；GPU 图形兼容实际试验结果决定能否在同一主机。**容器不能替代一个确实不兼容的 UE4 渲染后端。**
- **SG2002 TPU-MLIR 编译沙箱**：官方验证的容器/工具链与目标 `cv181x`/`cvimodel`，固定 SDK 版本与校准配置；不要因 TPU-MLIR 主分支多芯片支持就默认板子上全部支持。
- **LicheeRV Nano 板端**：Buildroot/对应固件、摄像头、ISP/VPSS、NPU Runtime、CPU 状态机和 UART；不运行训练器。
- **ESP32 飞控**：保留已有控制固件，另建立完整的命令接收、TTL、接管与故障注入验收。
- **连续物理实验**：若进 M7，再独立 Gazebo/PX4 或与自家 ESP32 控制等价的 C++ 物理仿真环境；不在 M0 时混入 ROS Noetic/Humble/Jazzy 多套系统。

## 14. 主要风险登记册

| 风险 | 最坏结果 | 降低风险的证据 |
|---|---|---|
| AirSim 旧 UE4 在 RTX 5090/Ubuntu 24.04 无法运行 | 主 benchmark 无闭环 | M−1 官方场景启动/重放日志；替代独立旧系统 |
| 语言没有被模型真实使用 | 退化为图像动作先验 | 指令 swap/shuffle、地标转移、长指令尾部测试 |
| 误以为 LAG 是已验证教师模型 | KD 没有可靠教师 | 分开标签器与真实闭环教师，先 BC 后 KD |
| ROI 没有额外高清信息 | 无效模块吞掉工期 | HD 可用性 + oracle/fixed ROI 机制探针 |
| 参数/FLOPs 推算板端延迟失真 | 方案无法上板 | 实测 P50/P95/P99、热稳定、内存水位 |
| STOP 改善掩盖路径质量下降 | 指标造假 | 同报 SR/OSR/NE/SDTW 及长度/失败类型 |
| 主方法与 2026 工作同构 | 方法论文新颖性不足 | 与 Self-Monitoring、LightVLN、QAD 式相同预算对照 |
| RGB-only 在真实环境尺度和运动状态不足 | 自由飞行碰撞 | 不宣称现实安全；C 轨单独验证飞控控制能力/时延/接管 |
| 自研框架数值错误 | 训练效果无法解释 | 独立黄金数值、有限差分、静态 parity、回归测试 |
| 过度以 val_unseen 调参 | 不可信泛化结果 | 训练内部 dev、预登记实验、有限次确认评测 |

## 15. 一份可检查的“研究成功”定义

**最低工程成功**：官方 AerialVLN-S RGB-only 闭环可跑；自研静态学生能训练并表现明显优于随机；语言干预有效；在真实 SG2002 上 INT8 推理、记忆和 UART 正确并有实际全链 P95/内存/热稳定 trace。

**方法成功**：新机制在相同输入、参数、训练预算和可比延迟下，比强 GRU/Self-Monitoring 基线提升至少一个预注册主指标，且不让关键 SR/安全/STOP 指标系统性退化；效果不依赖调参时滥用 val_unseen。

**现实导航成功**：需要独立连续动力学环境、准确控制协议、可靠定高/接管与多次受控实机无碰撞飞行；**前两项成功不蕴含第三项成功**。

## 16. 核验来源与资料边界

**用户材料（上传文件）**：
1. `TinyAerialVLN_研究方案与跨模型评审稿_V3_20261028(1).md`（原始拟议方案）
2. `TinyAerialVLN_研究方案_V4_修订执行稿_20261028.md`（聚焦执行稿）
3. `TinyAerialVLN V3.0 独立评审：收窄创新点并修正三处硬伤.md`
4. `TinyAerialVLN_方案评审意见_20261028.md`
5. `TinyAerialVLN_独立评审意见_V1_20261028.md`
6. `TinyAerialVLN_V3_独立评审报告_证据核查版_20261028.md`
7. `TinyAerialVLN_V3_独立评审_20261028.md`

**外部一手资料（本次核对）**：
- [AirVLN 开源仓库和下载/依赖说明](https://github.com/AirVLN/AirVLN)
- [AirVLN 官方客户端图像抓取实现](https://github.com/AirVLN/AirVLN/blob/main/airsim_plugin/AirVLNSimulatorClientTool.py)
- [AerialVLN ICCV 2023 论文主页](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.html)
- [Sipeed LicheeRV Nano 官方规格与 PCB 注意事项](https://wiki.sipeed.com/licheerv-nano)
- [TPU-MLIR 编译器与 CV181x](https://github.com/sophgo/tpu-mlir)
- [Self-Monitoring Agent](https://arxiv.org/abs/1901.03035)
- [AdaFocus](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Adaptive_Focus_for_Efficient_Video_Recognition_ICCV_2021_paper.html)
- [Input-Adaptive Inference for Efficient VLN, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Kang_Harnessing_Input-Adaptive_Inference_for_Efficient_VLN_ICCV_2025_paper.html)
- [LookasideVLN, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ning_LookasideVLN_Direction-Aware_Aerial_Vision-and-Language_Navigation_CVPR_2026_paper.html)
- [Query, Align, and Distill, TCSVT 2026](https://doi.org/10.1109/TCSVT.2026.3727199)
- [AirAnchor, 2026](https://arxiv.org/abs/2609.08442)
- [LightVLN, 2026](https://arxiv.org/abs/2610.05024)

> 尚未开展：用户 RTX 5090 上的真实 AirSim 启动、AerialVLN-S 数据包逐字段复查、LicheeRV Nano PCB 实物辨认、NPU 算子实测、FP32/INT8 推理、闭环成绩与实机试飞。上述内容在本文中一律为计划或验证门槛，不是已经完成的工作。
