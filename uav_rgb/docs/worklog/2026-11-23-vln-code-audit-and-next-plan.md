# 无人机视觉语言导航代码审查与后续计划

日期：2026-11-23
审查方式：主代理与并行 subagents 分别检查导航语义、控制与可视化、训练运行契约、Git 基线和代码缺陷。此次只读审查，不改源码、不清理实验文件、不启动训练或仿真。

## 结论

项目目标是无人机根据自然语言和实时视觉完成有序导航，并在正确目标处自主停止。仓库已经有场景、任务、grounder、局部策略、训练和评测组件，但目前不能把它们概括成已验证的端到端视觉语言导航算法：3DGS 路径主要由 VLM grounder 提供方位与 `near_goal`，C++ 学生使用 RGB 和 15 维动态导航条件，且学生不直接消费原始语言。多阶段任务还会把标注阶段指令提供给 grounder。当前 11 月 21 日阶段 1 冻结闭环门禁未通过，不能用现有结果证明语言改变了无人机路线或成功完成导航。

论文的可检验目标应当具体到一条任务，例如“从起点穿过左侧门洞，进入指定房间，在蓝色椅子旁停下”。评测必须判断阶段顺序、目标实例、碰撞和自主停止。飞行距离、经过目标附近、视频平滑或单独动作分类正确都不是任务成功。

## 优先问题

### P0：训练阶段复用契约会漏掉实验输入变化

当前未提交的 `train.py` 运行契约对复用 S1 示范集只纳入 `vln_eval.json` 与 `episodes.jsonl`，没有纳入帧级 RGB/Depth 内容；VLM 模型目录只纳入 JSON，没有覆盖权重分片；OpenFly 契约没有覆盖被调用的外部运行时代码。场景 `scene.json` 也不足以代表其引用的 PLY 和碰撞网格。与此同时，带 `.stage_complete` 且必需文件存在的阶段会直接复用，不验证产物所对应的输入哈希。

因此，同一运行名下即使示范图像、VLM 权重或外部代码已经替换，旧阶段仍可能被静默当作当前配置的结果。这是昂贵训练或恢复实验前的首要风险。下一步先在现有 `train.py` 补齐输入清单或固定数据包哈希，并让阶段复用记录及核对输入/产物契约，再用替换一个输入文件的 CPU 测试确认旧阶段会拒绝复用。代码位置见 [`train.py`](../../train.py) 328-345、357-393；交接文档也已记录场景资产和模型权重指纹缺口，见 [`HANDOFF.md`](../HANDOFF.md) 7-11。

### P1：后方目标方位被解释成前进

`ReactiveBearingPolicy` 的 8 桶约定中，0 是正前方、1-3 左侧、4 后方、5-7 右侧、8 搜索。策略只把 1-3/8 映射为左转、5-7 映射为右转，其余值前进，所以 4（目标在正后方）会返回 `FORWARD_SLOW`。我用现有策略做了无 GPU 直接复现，结果为 `bearing_bin=4 action=1 FORWARD_SLOW`。见 [`reactive.py`](../../sim/3dgs/reactive.py) 164-180。应先修正映射并为 0-8 每个桶补回归用例，再跑固定输入回放。

### P1：当前闭环已出现动作塌缩

论文计划记录的 dropout=0.2 冻结闭环中，真值局部子目标、终点直线方位和黑屏三臂均为 `SR=0/2`、`OSR=0/2`、碰撞 `2/2`、显式停止 `0/2`，34 个动作全为 `FORWARD_SLOW`。这比泛泛讨论模型结构更直接：下一步先定位学生为何不能按真值局部子目标转弯，不能在这个状态下继续扩数据或开展正式语言实验。详细记录在 [`2026-11-21-vln-paper-plan.md`](../experiments/2026-11-21-vln-paper-plan.md) 25-34。

### P1：视觉语言输入与执行策略之间不是端到端语言闭环

`run_vln.py` 会按策略构造学生，VLM student 分支要求 `prev_dim=15`，且错误提示明确说明当前权重不会消费 VLM 方位。`StudentAdapter` 将方位、near-goal、上一动作作为动态条件传给学生；学生动作本身不是由原始语言直接条件化。分层设计可以作为研究方案，但论文必须明确语言在哪个模块改变目标选择及路线，不能仅凭使用了 VLM 就称端到端 VLN。见 [`run_vln.py`](../../sim/3dgs/run_vln.py) 689-697、[`collect_3dgs.py`](../../sim/3dgs/collect_3dgs.py) 235-250、[`student.py`](../../sim/tools/student.py) 和 [`nav_cond.py`](../../sim/tools/nav_cond.py)。

多阶段任务还会将任务总体指令和标注的当前阶段指令一起发给 grounder，见 [`run_vln.py`](../../sim/3dgs/run_vln.py) 103-109、[`vln.py`](../../sim/3dgs/vln.py)。因此当前这类结果属于带阶段标注提示的分层导航，不等价于只给一条原始语言指令的自主阶段推断。建议分别定义“原始语言端到端组”和“阶段标注诊断组”，后者不能进入主结果。

### P1：STOP 证据不足以确认目标实例

`VlmBeliefTracker` 用重复的 `reason` 文本判断两帧是否指向同一目标，并将 `near_goal` 与中心方位组合确认目标。它没有显式目标框/掩码、实例 ID、跨帧视觉匹配或独立几何距离验证；VLM 置信度被记录但不用于确认门控。VLM grounder 存在时，`collect_3dgs.py` 的最终停止由 `confirmed_target` 覆盖模型 STOP 头。因此需把停止结果标为 grounder/rule stop 或 student stop，不能将前者归因给学生 STOP 头。见 [`run_vln.py`](../../sim/3dgs/run_vln.py) 127-158、[`collect_3dgs.py`](../../sim/3dgs/collect_3dgs.py) 247-259。区域与实例证据是既有论文计划提出的候选假设，目前尚未实现或验证。

### P2：可视化适合诊断，但部分实时画面不适合当因果证据

离线录像顶视图已经标记 `[DIAGNOSTIC/ORACLE]`，并区分真值目标与预测目标，这个边界处理是正确的。Studio 实时地图显示 `WAYPOINT` 时没有标明它是 GT/参考点还是策略预测点。实时面板又分别读取最新视频帧、最新位姿和最新 action JSONL，未按同一时间戳对齐，所以只能作为运行监控，不能据此解释模型在某帧做某动作。见 [`video.py`](../../sim/3dgs/video.py)、[`studio.py`](../../ui/studio.py) 2403-2447、2609-2612、2727-2745。

下一步在地图标记中显式区分 `GT/reference`、`grounder prediction` 和 `executed target`，记录统一的仿真时间/单调时间戳，并在取证视频按时间对齐帧、位姿、动作、模型输出和安全层干预。可视化必须让人看出任务目标、阶段、STOP 来源和最终成功/失败原因。

### P2：飞行抖动尚不能归因于硬件 PID

现有控制是离散动作及保持时间，方向决策或净空门控来回切换都可能产生抖动。当前材料没有同步姿态/角速度/速度、动作输出和动力学状态的证据，不能据此直接改真机 PID。先在固定仿真动力学 profile 上记录左右反转次数、动作切换/米、每米停顿、偏航角速度、横滚/俯仰标准差、动作延迟和碰撞前残余速度。把策略抖动与执行器/PID 跟踪误差分开分析。旧计划也明确要求在未校准 profile 下不要修改硬件 PID，见 [`PLAN_2026-11-13_vln.md`](../PLAN_2026-11-13_vln.md) 452-457。

## 建议研究方案与执行门禁

研究假设沿用已有论文计划，不在此阶段另造“新算法”名称：显式视觉区域证据支持阶段推进，跨帧目标实例证据支持正确 STOP，局部策略负责可执行避障与接近。创新性要等公平基线和消融结果出来后再判断。

1. **先保证可复现，不用 GPU。** 补足运行契约覆盖的数据帧、权重、外部运行时代码、PLY/碰撞网格和场景元数据；验证所有复用产物与契约一致。固定当前任务清单、场景/路线划分、seed、控制器、VLM 版本、推理预算与阈值。
2. **先修确定性控制错误，不用 GPU。** 修复 bearing 4，补齐 0-8 桶测试。用相同 RGB/depth/state 轨迹分别回放当前学生、真值局部子目标、教师和盲飞，报告动作分布、首个需要转向的时刻、转向是否发生、动作反转和碰撞原因。必须先证明真值局部目标能驱动学生通过首个门洞，再进入语言实验。
3. **验证语言是否真的改变路线，先冻结权重。** 对同起点、不同目标的配对任务回放正确指令、目标互换、空指令和关系词反事实，初始观测和预算一致。先用缓存帧测 grounding，再做小规模闭环。主检验包括首个分岔、目标实例、阶段顺序与自主 STOP。若语言互换没有改变首个分岔和最终目标，停止扩规模并修接口/数据。
4. **通过局控门禁后再实现候选证据门控。** 同一个局控 checkpoint、同一个 VLM、同一推理预算对比 G3 文本输出、区域证据阶段门控、实例证据 STOP 门控和完整方案；每次只移除一个模块做消融。正确目标与错目标、同名异实例、近墙无目标、目标短暂遮挡和错误门洞都纳入开发集。标注阶段提示单独作为诊断臂。
5. **最后做正式配对评测。** 固定未参与调参与选模的场景和路线簇，按同一任务/动作预算比较教师上界、盲飞下界、现有 G3 和候选方案。已有计划设定正式 seed `20261117, 20261118, 20261119`，除非修订实验计划，否则保持一致。一次只变化一个预登记因素，记录 GPU UUID、精度、显存、batch、更新数、学习率曲线、数据和 checkpoint 哈希、推理预算及动力学 profile。单卡串行，仅使用用户指定的 GPU0；训练前向用户说明并确认可复现实验门禁已通过。

每集至少报告请求数及分子/分母、SR、OSR、NE、SPL、nDTW、SDTW、阶段依序完成率、错误 STOP、STOP precision/recall、碰撞、超时和延迟。对不同场景或动作接口不兼容的公开结果不做直接排名。主指标应是“按序经过要求的阶段、无碰撞并在正确目标实例自主停止”的请求成功率。

## 开源基线与论文定位

已有文档核对过 CityNavAgent、See, Point, Fly、VLM-Nav、GRaD-Nav++ 和 SAGE-3D，也整理了 VLN-CE、SAGE-Bench、LH-VLN、AerialVLN/OpenFly 等指标与尺度差异。但先前实测的 PULP-Dronet、DroNet、Depth Anything 主要是避障比较，不能作为 VLN 基线；当前仓库尚无同任务预算、同场景/动作接口的公平 VLN 基线闭环。后续至少选择一个可合理适配的公开 VLN 方法，在同一任务集和渲染/动力学下评测，并列出输入、动作接口、预训练数据和推理预算差异。相关梳理见 [`PLAN_2026-11-13_vln.md`](../PLAN_2026-11-13_vln.md) 257-280、[`2026-11-21-vln-paper-plan.md`](../experiments/2026-11-21-vln-paper-plan.md) 5-17。

## Git、清理与交接边界

- `uav_rgb` 当前 HEAD 为 `463be09bb121001fe159e824e64d07c936d13426`。审计开始时工作区有 422 项状态变化，包括 39 个修改文件、375 个删除、8 个未跟踪，均按用户已有内容处理。本次没有暂存、清理、还原或提交这些变化。
- 源仓库没有配置 remote。兄弟目录 `code/ALVLN` 的 HEAD 与 `origin/main` 均为 `06d01d9203e3cdba98b362a3de3e6eb8c4a03297`，源仓库 `.mirror_src_commit` 指向当前源 HEAD；检查时 GitHub `ALVLN/main` 也为该镜像提交。没有 push，因为本次没有代码修复或可提交的源码变更。
- 用户工作区中大量删除项尚未逐个做引用、实验价值和恢复审查，不能把它们认定为死代码。本轮没有删除日志、测试、旧文档、模型或实验结果。后续清理前先按入口引用、Git 历史、论文证据和复现实验依赖分类，并逐批核验。
- 个人 skill 已明确禁止擅自新增训练包装脚本、`run_*.sh` 或替代入口，并要求优先复用现有 `train.py`；无需再次重复修改 skill。

## 本轮验证

以下均为 CPU/既有构建产物验证，没有训练、仿真闭环或 GPU 推理：

| 验证 | 结果 |
|---|---|
| `sim.3dgs.test_grounder`、`sim.3dgs.test_vln`、`sim.3dgs.test_student_inputs` | 97 tests passed |
| `sim/tools/test_instruction.py`、`sim/tools/test_nav_cond.py` | 23 tests passed；有未关闭文件的 ResourceWarning |
| `tools/test_train_entry.py` | 11 tests passed；`train.py --check` 预检明确未导入 torch、未触碰 GPU |
| `ui/test_studio.py` | 55 tests passed、3 skipped；无 Tk/display，另有评测日志文件 ResourceWarning |
| bearing bin 4 直接回放 | 复现为 `FORWARD_SLOW`，确认上文缺陷 |
| `git diff --check` | 通过 |

本轮没有运行完整 CTest、真实 VLM 推理、3DGS/cf_sitl 闭环或新训练；这些验证需要后续门禁和算力安排，当前不构成模型任务成功证据。

## 做了什么

- 并行审查无人机 VLN 任务语义、训练/复用契约、控制与可视化、Git/实验基线及代码缺陷。
- 阅读当前 handoff、论文实验计划、历史 VLN 计划与公开基线梳理，并确认论文计划中的阶段 1 当前未放行。
- 做了上述 CPU 单元测试、契约预检和 bearing 4 缺陷复现。
- 新增本交接文档，供后续 AI 按门禁继续工作。

## 没做什么与待确认

- 没有修改源码或 skill；用户要求先阅读和整理想法，本轮以审计和计划为交付。
- 没有运行 GPU 训练、GPU 推理或仿真闭环；当前控制门禁失败且复用契约不完整，先消除这两个风险。
- 没有删除或恢复用户已有文件；422 项现有 worktree 变化未逐文件审完，清理风险和实验价值尚待确认。
- 没有验证真机 PID、飞行平滑度、完整 CTest 或真实 VLM 行为；现有 CPU 测试不覆盖这些硬件与集成路径。
- 没有实测公开 VLN 基线；算法新颖性尚未成立，需在控制与语言因果门禁通过后做同口径适配比较。
- 没有提交或推送 Git；下一轮若按用户要求修复代码，应先基于本地差异审查修复，再提交单独、可回溯的 commit 并同步远端。

## 更新 2026-11-23 07:31 CST

按上述门禁继续执行后，已经修复可确定的问题，并用既有失败轨迹做了单帧推理诊断。以下结果不改变“阶段 1 未放行”的结论。

### 做了什么

- 扩展 `train.py` 运行输入指纹，覆盖被复用的采集 episode 文件、场景引用的 PLY/碰撞网格、模型目录权重、OpenFly 清单引用的图像与启动资产。阶段完成标记现在记录命令、配置和必需产物哈希，旧的空 marker 会拒绝复用。OpenFly ResNet 预训练权重预检路径也补上了 Torch Hub 的 `hub` 目录。
- 修复 `ReactiveBearingPolicy` 对后方目标 `bearing_bin=4` 的处理，按左右净空选择转向，安全数据缺失时悬停，并以 0.25 米滞回减少方向来回切换。添加覆盖 0 至 8 桶、后方目标和无效深度的测试。
- 在 `train.py` 将训练设备固定为物理 GPU 0。`UAVRGB_CUDA_VISIBLE_DEVICES` 纳入运行配置与阶段配置，其他取值会在预检阶段拒绝，子进程不继承外部的多卡可见设置。
- 对失败门禁首帧用两个既有 checkpoint 做了 10 次固定图像推理，进程仅暴露 GPU 0。新 checkpoint 在 `(prev=0, bearing=1, near=0)` 下仍选择前进，概率为 `[SAFE=0.003, FORWARD=0.498, LEFT=0.291, RIGHT=0.208]`；把方位改成 unknown 8 时选择右转，概率为 `[0.000, 0.008, 0.451, 0.541]`。前进历史会进一步把前进概率推到 0.673。条件输入能够改变输出，双帧及 VLN 条件编码在训练端和 C++ 推理端一致，未发现接口字节错位。
- 同起点教师轨迹在相同 `bearing=1` 首步执行 `YAW_LEFT`，学生执行 `FORWARD_SLOW`。学生碰撞前预测中央深度约 2.6 至 2.8 米、near-obstacle 概率约 0.14 至 0.16，而碰撞代理净空约 0.14 至 0.20 米。两种距离的定义不同，这提示深度辅助头可能没有为当前避障门提供可靠信号，尚不足以认定某段实现有 bug。当前最强证据指向已训练策略在闭环状态上的动作/深度预测失效，是否由数据分布或损失权重引起仍未证实。
- CPU 验证结果为 `tools.test_train_entry` 15 项通过，`sim.3dgs.test_vln`、`sim.3dgs.test_grounder` 与 `sim.3dgs.test_student_inputs` 共 100 项通过。`py_compile` 和 `git diff --check` 通过。

### 没做什么

- 阶段 1 门禁仍为 0/2 成功、2/2 碰撞、0 次显式停止，不能推进语言因果测试、正式对照或新训练。没有新建训练脚本，也没有启动训练或新闭环仿真。
- 单帧诊断只用了 GPU 0 的缓存帧，没有使用 GPU 1。未重跑 311 秒的三臂门禁，也未消耗新训练预算。
- 没有删除日志、旧实验或文件。当前工作区原有 424 项变化仍未逐项审核，不能认定为本轮死代码清理结果。
- 没有提交或推送。源仓库仍有 424 项既有未提交变化，且本轮修改的 `train.py`、两个测试文件等本身也包含这些既有差异；直接按文件提交会一并发布未审核代码。镜像脚本会从已提交快照整体替换 `ALVLN/uav_rgb`，所以目前没有生成混合提交或推送不完整快照。
- 未验证真实自然语言输入下的路线变化、真实 VLM grounding、硬件 PID 抖动和公开 VLN 基线。当前 S1 门禁 checkpoint 的 `text_dim=0`，这是局部控制诊断而不是语言输入实验。

## 更新 2026-11-23 10:00 CST

### 做了什么

- 并行审查训练数据/增强、现有 checkpoint 轨迹和论文实验顺序；重新核对 GPU、输出盘和固定输入哈希。
- 解析 `dataset.uavr` 与 `dataset.aux`，独立得到 train/val 样本数、动作分布、深度分带和 `(prev_action,bearing,near_goal) -> expert_action` 分层计数。训练集 `prev=FORWARD,bearing=1,near=0` 有 1,191 条、其中 965 条监督 FORWARD；train 与 val 合计 1,278 条、其中 1,032 条监督 FORWARD。`prev=SAFE,bearing=1,near=0` train+val 共 16 条。此前 checkpoint 在相同首帧将 prev 从 SAFE 改为 FORWARD 后，FORWARD 概率由 0.498 上升到 0.673。
- 核对增强 kernel：双帧 RGB 做相同几何平移，15×20 深度目标没有随 ±4 px 平移。将统一改用已有 `photo,flip` 配方做成对诊断。
- 新增实验登记 [S1 配对诊断计划](../experiments/2026-11-23-s1-cleanaug-prev-dropout-pair.md)，预登记数据、固定超参数、唯一变量、GPU 0 和后续门槛。

### 没做什么

- 本次没有改算法实现或训练代码。已启动登记的成对训练，输出目录为 `/root/rivermind-data/uav_rgb/exp-s1-history-cleanaug-paired-20261123`，PID/PGID `271493/271493`，只暴露 GPU 0；GPU 1 未使用。
- 两组预计串行总耗时约 8 分钟，预计 10:11 CST 完成。按实验守护规范，启动后只确认了一次进程存活，本轮不继续轮询。
- 尚无新 checkpoint、离线 104 帧对照或新增闭环结果；不据此声称算法有效。

## 更新 2026-11-23 10:07 CST

### 做了什么

- 核验原训练进程仍存活。dropout 0 组已完成 `rc=0`，8,375 updates，209.3 秒；dropout 0.2 组正在进行，观察时为 607/8,375 updates，训练器 ETA 约 194 秒。
- 更新数据盘 progress 和本工作日志。训练仍固定 GPU 0，GPU 1 未使用。

### 没做什么

- 第二组尚未完成，本配对实验没有结果；未启动缓存帧推理或闭环仿真，也未启动其他 GPU 工作。
- 该成对诊断只改 dropout 因素，两组都采用修正过的 `photo,flip`。因此可比较该配方下 dropout 的影响，不能把结果解释为单独的增强修复效应。

## 更新 2026-11-23 10:19 CST

### 做了什么

- 两组配对训练均以 `rc=0` 完成。逐字段比较训练配置后确认只有 `prev_action_dropout` 不同，数据哈希、source commit、seed、增强和 8,375 updates 一致。两组最佳 checkpoint 都在第 3 轮；dropout 0 的 dev macro-F1 / balanced accuracy 为 0.67154 / 0.67600，dropout 0.2 为 0.65805 / 0.65778。末轮 macro-F1 为 0.61601 / 0.61731，当前结果不支持 dropout 0.2 有改善。
- 用两个 checkpoint 在同一批 104 个教师缓存帧回放，actual previous action 条件下的教师动作一致数为 69/104 和 68/104。只统计 `near_goal=0` 且 bearing=1/7 的 18 帧时，正确方向分别为 dropout 0 的 12/18、dropout 0.2 的 11/18。分母很小，此回放只作为闭环启动门禁。
- 已按计划启动两组同路线 privileged-bearing 闭环诊断，使用 `sage3d_839992/ti_0_0` 与 `sage3d_841980/ts_0_0`，seed 0、hold 0.5、相同 `subgoal` 方位输入和深度落盘。任务于 10:18:35 CST 启动，串行只暴露 GPU 0，输出在 `/root/rivermind-data/uav_rgb/exp-s1-history-cleanaug-paired-20261123/closed-loop/`。按既往同类评测耗时估计，约 10:22 CST 完成。
- 更新配对实验计划和结果记录，保留训练、缓存帧和闭环输出路径及限制说明。

### 没做什么

- 闭环任务仍在运行，当前还没有路线成功、碰撞、首个转向门洞或深度渲染结果。闭环完成前不据分类或缓存帧结果判断航线通过。
- 本轮没有修改训练/控制源码，也没有增加训练包装脚本。该闭环使用场景 waypoint/A* 子目标提供方位，指令没有传入学生，不能作为端到端视觉语言导航结果。
- 尚未评测未见路线簇、多 seed、目标语言反事实、公开 VLN 基线或真实 VLM grounding；当前结果不能支持论文算法有效性结论。

## 更新 2026-11-23 10:22 CST

### 做了什么

- 核验首次闭环启动后发现 `run_vln.py` 要求 `--out` 为空目录。由于启动时预建了组输出目录，命令在 GPU 初始化前以 `rc=2` 退出，没有运行仿真。保留首次启动日志，没有删除输出或日志。
- 改用全新的输出目录 `/root/rivermind-data/uav_rgb/exp-s1-history-cleanaug-paired-20261123/closed-loop/retry-1/` 重启两组同路线评测，PID 为 `284783`，启动时间 10:22:14 CST。仍按 GPU 0 串行运行，预计约 10:26 CST 完成。

### 没做什么

- 重跑仍在进行，尚未得到闭环路线指标。首次失败没有触及 GPU；重跑结束前不报告路线成功或碰撞结论。

## 更新 2026-11-23 10:42 CST

### 做了什么

- 核对两组 subgoal 闭环完成标记、`vln_eval.json`、逐步动作和 `--save-depth` 文件。两组 SR/OSR 均 0/2 且 STOP 0/2。dropout 0 在 140/140 步全输出转向，未碰撞但耗尽预算；dropout 0.2 在 34/34 步全前进，并在第 24、8 步发生 voxel proxy 碰撞。
- `dataset.uavr` 元数据表明训练方位来自 `direct`。为控制该条件语义差异，使用同两个 checkpoint、路线、seed 和其余配置，另跑 direct bearing 闭环。结果仍均为 SR/OSR 0/2 和 STOP 0/2；dropout 0 为 140 步左右转向且未碰撞，dropout 0.2 为 34 步全前进并发生两次碰撞。
- 对 rollout 图像做 previous-action 反事实回放，强制 SAFE 后仅 8/174 步 argmax 改变；固定比较 bearing 1 与 7 仅 3/174 步改变。actual previous action 回放逐步复现全部 174 个在线动作，动作 ID 与推理协议一致。两组策略在 rollout 视觉状态上都几乎没有利用 bearing。
- 用训练预处理重新构造碰撞帧的深度辅助标签，口径与预测网格一致。两次碰撞中央目标深度约为模型预测 2.38/2.08 m 对真实分带期望 0.42/0.20 m；near-obstacle 概率 0.187/0.261 对真实近障网格占比 0.778/1.0。相机图像正常，保存帧与教师首帧匹配。现有证据指向闭环状态上的策略输出塌缩和深度头泛化误差，没有找到动作 ID、bearing 编码或网格单位实现错误。
- 并行审查确认现有 10,708 train 帧含 75 个教师 episode 与 45 个 DAgger episode，train/val geometry group 分离，测试场景未进入该训练集。计划按既有入口在三个 train 场景采集新一轮 DAgger 状态，不碰测试场景。

### 没做什么

- 当时尚未启动 DAgger 第二轮采集或后续训练。需要先固定采集方位语义、β、seed 和新增样本下的优化更新数，再运行单卡实验。
- 当前模型不是端到端 VLN。上述闭环没有将自然语言输入学生，不能作为语言任务成果。
- 没有修改源码、测试或 skill，没有删除旧数据，也没有提交或推送 Git。

## 更新 2026-11-23 10:50 CST

### 做了什么

- 固定下一项数据采集的条件为 `g0-mode=direct`，与现有 `dataset.uavr` 元数据 `vln_bearing_source=direct` 对齐；使用三个既有 train scenes、`split=train`、`beta=0.3`、round 2、seed 401、每场景 limit 15。采集标签由 A* 教师提供，执行策略以当前 clean-augmentation dropout 0 checkpoint 为基础。
- 已启动 DAgger round 2 采集，PID 为 `303291`，10:49:53 CST 开始，限制 CUDA 到物理 GPU 0。输出路径为 `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/collect_r2/`，日志与完成标记在同实验目录。
- 依据同样 45 集的既有采集耗时 559.6 秒加 15% 余量，预计约 11:00:37 CST 完成。采集完成后先核实 route failures、有效样本数及碰撞过滤，再定后续训练更新数。

### 没做什么

- DAgger 采集仍在运行，第二轮数据集尚未生成，后续训练未启动。旧训练有 8,375 updates，但新数据会改变每 epoch batch 数；在核对新样本数前不启动 GPU 训练。
- 不会把这次训练场景采集并入 test split，也不会用刚才已反复使用的测试路线声称最终泛化。当前所有路线试验仍是 privileged-bearing 控制诊断。
- 没有添加训练或采集脚本，没有修改源码、skill 或配置。

## 更新 2026-11-23 11:07 CST

### 做了什么

- 核验 DAgger round 2 报告为 45/45 集、三个 train scenes 各 15 集、0 个采集错误；`train.py` 的 `_collection_episodes()` 校验通过，45 集均有 manifest 和深度文件。报告记录 SR/OSR `8/45`、SPL `0.1503`、平均导航误差 `3.85 m`、碰撞率 `12/45`。这些是学生采集轨迹结果，不是最终 VLN 评测。
- 汇总逐步标签得到 5,369 行、12 个碰撞行、学生动作与教师动作不同 2,468 行（45.97%）；β=0.3 的教师介入标记为 719 行（13.39%）。介入标记只在教师动作改变学生动作时写入，比例与采集实现一致。
- 确认 45 个新增 episode 全部为 `executed_by=student`、DAgger round 2、β=0.3、训练 split，使用 clean-augmentation dropout 0 checkpoint（SHA256 `dee74160842b729c14d266c3507274dd12c434a6a8618b9b6d227f9e0a4c1d5f`）；报告的方位模式为 direct。并验证原训练数据仍可由入口枚举为 75 个教师训练集、45 个 round 1 DAgger 集和 15 个独立 validation 集。
- 并行代码审查确认预处理监督标签来自 `action_expert`，previous-action 条件来自实际执行动作，碰撞行按既定规则过滤；31 项定向 CPU 测试通过。修正 `DaggerPolicy` 注释中已过期的 beta 描述。

### 没做什么

- 未打包新数据集或训练模型。阶段 1 的同路线 direct/subgoal 仍未完成控制门槛，先补跑同 checkpoint 的 blind-RGB 对照；训练更新预算保持待定。
- 采集报告和 episode 清单完整且验证通过，但外层 supervisor 日志只留下 `rc=`，没有 `DONE` 标记，因此没有把该日志当作退出码证据，也没有重复 45 集采集。
- 没有新增训练包装脚本，没有触碰 GPU 1，也没有提交或推送 Git。

## 更新 2026-11-23 11:52 CST

### 做了什么

- 延续阶段 1 的局部控制诊断。blind-RGB 对照在两条冻结路线均为 SR/OSR `0/2`、STOP `0/2`、碰撞 `0/2`；两集全程 140 步都是 `YAW_RIGHT`。这说明旧 checkpoint 在黑屏与正常 direct-bearing 条件下都失败，不能据此分离 RGB 的因果作用。
- 为新增 R2 数据的诊断训练接入 `--max-updates`。预算参与 lr 总步数、运行配置与 resume 继承，达到预算后正常验证并保存 checkpoint；新增定向测试覆盖精确更新数、部分 epoch 验证、checkpoint 续训、超容量预算、epoch 总步数溢出及 CLI 负数输入。
- 修复默认 epoch 训练的总更新数乘法溢出检查。`uavrgb_train`、`test_trainer`、`test_train_cli` 编译通过；trainer 和 CLI 测试均 `PASS`。`git diff --check` 通过。
- 通过仓库 `_collection_episodes()` 逐组验证 75 teacher train、15 teacher validation、45 R1 DAgger、45 R2 DAgger，共 180 个唯一 episode。R2 supervisor 没有可信退出码，因此使用采集报告及 episode 清单逐集校验，没有重跑采集。
- 用现有 `tools/preprocess.py` 打包 R2 聚合数据，保持 839992 validation，test split 为 0。总样本 16,980，其中 train 16,065、validation 915；过滤 38 条 collision step。`dataset.uavr` SHA256 为 `08ca281ceb00ef618f38fe85cdf643aab66c6aa1ac97978fdc347c5e2b2a7358`，`dataset.aux` SHA256 为 `01c45f1569012ff74b1e3b9dedd58bd1884ed234cf951fe7d81a6bf75989a564`。
- 固定更新训练以 `train_rc=0` 完成，metrics 精确包含 8,375 步，所有 train_step 均记录 `total_steps=8375`。R2 best dev macro-F1 / balanced accuracy 为 `0.645596/0.650809`（step 503），低于旧 dropout-0 组的 `0.671536/0.676000`（step 1,005）。最佳 checkpoint SHA256 为 `11e67512be11e7d97c8f19fc5110b3470b7a8ff56ef6a06f69f737216e5bb719`。配置比较确认除数据集哈希/新增 R2 episode 与固定预算记录外，原训练超参数一致。
- 第一次自动闭环因误传 `--teacher-clearance braking` 被 `run_vln.py` 在启动仿真前以 `eval_rc=2` 拒绝。根据 CLI 约束去掉该参数后已重启相同两条路线，使用默认 `ray` clearance；GPU0 only，PID/PGID `358942/358942`，于 11:59:21 CST 开始，预计 12:01 CST 完成。其日志和独立完成标记记录在实验目录。
- 更新实验登记 [R2 诊断计划](../experiments/2026-11-23-s1-dagger-r2-fixed-updates.md) 和数据盘 [progress.md](/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/progress.md)。

### 没做什么

- 训练已结束且 dev 分数低于基线；修正后的 10 集闭环已完成。全量任务 SR/OSR/STOP `0/10`，碰撞 `3/10`。两条与基线完全匹配的路线 SR/OSR/STOP `0/2`，无碰撞，平均 NE 改善 `0.717 m`，但反转次数从 1 次升到 26 次。
- 阶段 1 门禁仍未通过。本次只是一次性数据分布诊断，不代表开始正式论文训练，不放行语言因果实验，也不能支持端到端 VLN 主张。
- 没有删除用户已有日志、实验、文档或模型；没有新增辅助脚本。源仓库 remote 仍为空且工作区有大量既有差异，因此没有提交或推送。
- 修正后 closed-loop 的 `DONE` 已核验为 `rc=0`。全量10集仍无成功或显式 STOP，matched两条路线的NE有小幅改善，转向反转明显增加。仍需解释转向振荡及低前进率，不能据此声称局部控制有效。

### 12:00 CST 补充

同一 915 条验证样本的混淆矩阵按“行是真值、列是预测”读取。R2 的 SAFE/FORWARD/LEFT/RIGHT 召回分别为 `14/14, 520/648, 51/134, 50/119`；dropout-0 基线为 `14/14, 523/648, 56/134, 57/119`。R2 中 LEFT→RIGHT 错分 22 条、RIGHT→LEFT 23 条，基线分别为 16 和 14。训练集 turn 标签比例较高，但单 seed 对照不能说明新增 DAgger 数据造成了方向混淆；后续可按 bearing、前序动作和场景分层检查。

第一次闭环调用因 clearance 参数不合法在仿真前退出，已记录并修正。第二次评测成功。该命令未设 `--limit 1`，实际运行了两个场景的全部 10 条 eligible routes；与旧基线共享的两条 task keys 已逐一确认起点和朝向一致。新模型在 `ti_0` 的 NE 从 5.347 m 降至 3.783 m，在 `ts_0` 从 2.913 m 变为 3.044 m。整体动作仍有强烈左右振荡，10集中的转向方向反转合计多，阶段1门禁继续保持未通过。

## 更新 2026-11-23 12:20 CST

### 做了什么

- 并行审查 R2 的动作 trace 与学生执行适配器。两条匹配任务每一步 `action_proposed == action_executed`，在线干预字段为 null，飞控每个 0.5 秒窗口的动作与记录一致。没有证据表明动作映射或深度护栏造成了反转。
- CPU-only 定向测试通过：`sim.3dgs.test_vln` 67 项、`sim.3dgs.test_3dgs_student_prev_action` 9 项、`sim/tools/test_student.py` 通过。离线 debounce 串显示可以靠插入 SAFE 降低直接反转，但总切换增加；改写动作后的闭环路线不能由原 trace 评分。
- 预先登记只把动作窗口从 0.5 秒改为 0.25 秒的两路线评测，固定 checkpoint、seed、路线、bearing/STOP 条件和仿真总时长。见 [action-window diagnostic](../experiments/2026-11-23-s1-action-window-diagnostic.md)。

### 登记时尚未完成

- 新闭环时序诊断尚未执行。未改执行器或训练源码，也没有训练新权重。结果出来前不根据离线改写动作声称导航改善。
- 未运行自然语言反事实、独立路线簇、多 seed 或公开 VLN 基线。阶段 1 仍未通过。
- 没有删除既有实验文件或 Git 变化，也没有提交或推送。

## 更新 2026-11-23 12:26 CST

### 做了什么

- 固定 R2 checkpoint、seed 0、direct bearing、显式 STOP、两个同起点匹配任务和 `cf_sitl_v1:1bc8fedf:lib31`，只将 `--hold` 从 0.5 秒改为 0.25 秒。`run_vln.py` 按 hold 将步数翻倍，保持请求的总仿真时间。GPU 仅暴露物理 GPU 0，评测 `rc=0`，12:26 CST 完成。
- 两集 SR/OSR/SPL 和显式 STOP 均为 `1/2`，碰撞 `0/2`，平均 NE `1.952 m`。`ti_0` 从 R2 的 `3.783 m` 改善到 `0.905 m` 并成功 STOP，FORWARD 从 11 次变为 60 次，turn reversal 从 16 次变为 2 次。`ts_0` 仍无前进、成功或 STOP，NE 从 `3.044 m` 变为 `2.999 m`，turn reversal 从 10 次升为 12 次。
- 检查 10 Hz SITL trace。`ti_0` 各动作窗累计绝对 yaw 从 `958.6°` 降为 `103.5°`；`ts_0` 仍转向近 `1,067°`。这说明时序效应集中在一条任务，不能推断飞控 PID 已修好。
- 结果、精确命令和指标写入 [action-window diagnostic](../experiments/2026-11-23-s1-action-window-diagnostic.md)，实验总计划和数据盘进度也已更新。

### 没做什么

- 没有训练新模型、改执行器、添加脚本或运行更多路线。短 hold 同时改变决策频率和模型时序输入，未隔离单一的机体控制因素。
- 没有把这两条复用的开发路线当作论文结果，也未放行阶段 1/2。自然语言因果、多 seed、独立测试路线和公开基线仍未完成。
- 没有删除用户已有实验文件或 Git 变化，也没有提交或推送。

## 更新 2026-11-23 12:29 CST

### 做了什么

- 用原 `r2-direct` 的 10 条 route/instruction 配置登记全量配对时序验证。固定 R2 checkpoint、seed、两个 split、direct bearing、显式 STOP、SITL profile 和总仿真时长，只改 `--hold 0.5 -> 0.25` 秒。精确命令见 [action-window diagnostic](../experiments/2026-11-23-s1-action-window-diagnostic.md)。
- 依据 `hold=0.5s` 全量 10 集实测 `130.7s`，按决策步骤翻倍及 15% 余量估算总时长约 5 分钟，ETA 12:35 CST。新输出目录为 `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/closed-loop/hold-025-all10/`，只暴露 GPU 0。

### 没做什么

- 该全量配对评测登记后尚未启动。未训练模型或更改控制实现。完成前不会根据两路线 pilot 宣称时序改进有效。
- 评测仍复用已用于开发诊断的路线，不用于正式测试或泛化主张。

## 更新 2026-11-23 12:35 CST

### 做了什么

- 仅将配对闭环评测扩大到已有 `r2-direct` 的 10 条 task keys，checkpoint、seed、路线/指令选择、direct bearing、显式 STOP、SITL profile 和模拟总时长固定。动作窗口设为 0.25 秒，不录视频。
- GPU0 only 守护进程已于约 12:34:50 CST 启动，PID/PGID `383117/383117`。输出根目录为 `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/closed-loop/hold-025-all10/`，日志 `run.log`，完成标记 `DONE`，停止命令 `kill -- -383117`。
- 按旧 10 集 `130.7s` 实测、决策步数翻倍及 15%余量估算，预计 12:40 CST 完成。守护启动后只确认一次进程和首个仿真 step，不做重复轮询。

### 没做什么

- 全量结果尚未产生。未启动训练、未修改执行器，也未运行其他 GPU 工作。
- 路线均为复用的开发诊断，不作为论文泛化证据。

## 更新 2026-11-23 12:52 CST

### 做了什么

- 核验 `hold=0.25s` 全量闭环以 `rc=0` 完成，`DONE` 时间为 12:35:20 CST，`vln_eval.json` 的 `wall_s=51.8`。与 `hold=0.5s` 基线逐项比对后，确认两个评测包含相同 10 个 task key、相同 checkpoint、seed、起点/朝向、direct bearing、显式 STOP 和 SITL profile，只有 hold 不同。
- 重新读取两组 `episodes.jsonl` 和汇总指标。SR/OSR/SPL/STOP `0/10 -> 1/10`，碰撞 `3/10 -> 4/10`，平均 NE `6.709m -> 6.281m`。唯一新成功为 `ti_0`，`ti_2` 新增碰撞。前进总动作从 20 增至 97，但转向反转从 159 降至 115 的总数受碰撞和 STOP 提前终止影响，不能作为轨迹平滑证据。
- 结合并行 trace 审查，确认没有一般性偏航率下降。三个前进碰撞的中心距离预测为 `1.94–2.88m`、近障概率为 `0.125–0.347`，未触发当前前进门控；另一次是左转碰撞。低质量帧占比为 `49.45%`，主要集中于 `sage3d_841980`，两种 hold 下同样存在。
- 已把全量结果、逐路线表格、风险解释和门禁状态同步到 action-window 诊断、R2 训练记录、论文实验计划和数据盘 `progress.md`。

### 没做什么

- 没改模型、执行器或训练超参数，也没启动新训练；本轮闭环使用 GPU0，未使用 GPU1。低质量帧只做统计，没有剔除路线或样本；没有根据 4 次碰撞调整安全阈值。
- 阶段 1 仍未通过。当前学生使用 privileged direct bearing，不消费自然语言，路线也已用于开发诊断，不能作为论文 VLN 或独立泛化结果。
- 未做独立路线/多 seed 评测、语言反事实或公开基线对比。检查中未找到可直接输出逐障碍漏报/误报的现成 CPU 评估入口，需先审查模型已有验证指标和 trace 字段，再决定是否需要新增受控评估。
- 没有删除现存实验与日志，没有新增脚本，也没有提交或推送。仓库 remote 为空，工作区包含大量既有改动与删除。

## 更新 2026-11-23 13:00 CST

### 做了什么

- 按阶段 1 计划登记同 checkpoint、同 10 条任务、同 seed、`hold=0.5s`、显式 STOP 和 SITL profile 的 `direct` 对 `subgoal` 配对评测，只改变 `g0_mode`。使用已有 `r2-direct` 为 direct 基线，不训练权重。
- 评测已用物理 GPU0 启动，PID/PGID `403491/403491`，输出目录 `/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/closed-loop/r2-subgoal-all10/`。一次启动检查看到首条路线第 7 步开始 FORWARD。依据最近无视频全量评测估计约 1 分钟，ETA 13:02 CST。

### 没做什么

- 评测尚未完成，当前没有路线汇总或阶段 1 门禁结论。GPU1 未使用。
- 没有训练模型、改代码、改超参数、改安全阈值或删除实验产物；没有新增脚本或提交 Git。

## 更新 2026-11-23 13:12 CST

### 做了什么

- 检查 `r2-subgoal-all10` 的完成标记为 `rc=0`、完成时间 13:03:56 CST，`wall_s=138.4`。与 `r2-direct` 核对相同 10 个 task keys、route/instruction index、逻辑初始 spawn、逐集 seed（每个场景 0–4）、checkpoint 路径、hold、显式 STOP 和 SITL profile。
- 全量结果 `direct -> subgoal`：SR/OSR/SPL/STOP `0/10 -> 0/10`，碰撞 `3/10 -> 5/10`，平均 NE `6.709m -> 6.917m`，中位 NE `6.930m -> 6.929m`。前进动作 `20 -> 52`、左右转 `402/392 -> 389/389`、反转 `159 -> 146`。新增碰撞为 `ti_0` 和 `ti_4`，`ts_4` 碰撞消失，其余三次碰撞持续。
- 逐集 trace 的起始 pose_before 与朝向在两臂完全一致。方法语义审查确认 subgoal 是由占据图 A* 和速度相关 lookahead 产生的特权局部瞄准点，不是视觉模型定位出的路标。
- 审查数据合同发现 R2 聚合数据元数据是 `vln_bearing_source=direct`，预处理器当前只支持 `recorded/direct`。因此该闭环结果说明未经 subgoal 条件训练的 checkpoint 未改善，不能代表 subgoal 监督模型的能力上限。
- 修复 VLN 评测产物的溯源缺口：`vln_eval.json` 新增基础 seed 与学生 checkpoint SHA256，逐集 `episodes.jsonl` 新增实际 seed。已有 episode manifest 仍保存 seed 和 SITL profile。
- 运行 `/root/rivermind-data/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_vln.py`，67 项通过。

### 没做什么

- 阶段 1 仍未通过，学生的 SR/STOP 均为 `0/10`；没有启动新的训练。下一项训练前需先在 CPU 上定义、测试 subgoal bearing 标签的生成合同，保证离线监督与在线占据图 lookahead 一致。
- 没有调碰撞阈值、删除困难路线、运行 GPU1、添加脚本或提交/推送 Git。direct/subgoal 两臂均属 privileged-bearing 控制诊断，不输入自然语言，也不能作为论文 VLN 结果。

## 更新 2026-11-23 14:55 CST

### 做了什么

- 按现有 runtime `OccPlanner.plan + aim_point` 口径，在 `sim/3dgs/vln.py` 提供共享 bearing 计算，并让在线 `run_vln.py` 与离线预处理共用它。预处理的首步速度通过 CF-SITL reset 重放恢复，要求首帧 pose/yaw 匹配；后续速度来自前一动作窗口末 trace，并校验时间戳和位置。
- 让 direct 与 subgoal 共用基于 direct bearing 的 search-dropout 决策，避免实验变量同时改变输入 bearing 和 dropout mask。清除了重放函数中未使用的参数与局部变量。
- 将单次实验优先调用现有入口、不为监控或日志临时造脚本的规则补充到 `/root/.codex/skills/personal-skill/SKILL.md`。skill-creator 校验通过。
- 使用原数据包记录的 180 个 `source_dir` 顺序重建 direct 和 subgoal 数据包。CPU 逐字节审计确认原包与 direct 重包完全一致；两种重包只在未屏蔽的 bearing 条件不同，保持了图像、动作、sample order、previous action/frame、depth、STOP、near-goal 和 dropout mask。数据包哈希和分项结果见 [R2 subgoal-bearing experiment](../experiments/2026-11-23-s1-dagger-r2-fixed-updates.md) 与数据盘 `train_r2_subgoal/progress.md`。
- `test_vln.py` 69 项、`test_preprocess_cond.py` 15 项、`test_preprocess_label_alignment.py` 3 项及 `py_compile` 全部通过。
- GPU0 空闲确认后启动 8,375-update subgoal 诊断训练，保持原 R2 超参数。训练后会在同一 10 条开发任务上顺序评估 direct 和 subgoal 输入。守护任务 PID/PGID 为 `482099/482099`，只暴露物理 GPU0。

### 没做什么

- 训练和两项闭环评测当时仍在后台运行，预计约 15:03 CST 完成；后续结果见下方更新。
- 没有改训练超参数、删除既有实验数据或日志、使用 GPU1、运行独立测试路线或改变阶段 1 门禁。新模型仍为 `text_dim=0` 的 privileged-bearing 控制诊断，不消费语言指令。
- 没有提交或推送 Git。当前仓库没有配置 remote，工作区还存在大量既有修改与删除，不能安全地将整个工作区当作本次改动上传。

## 更新 2026-11-23 15:46 CST

### 做了什么

- 核验固定 8,375-update subgoal-bearing 训练于 15:00:53 CST 完成，`rc=0` 且更新步数完整。配置除数据包路径和 SHA 外与 R2 一致；best validation macro-F1 / balanced accuracy 为 `0.787543/0.736004`，checkpoint SHA256 为 `513e95a7a26709ab38a0157453138ea28ef0dac663e8f267919c2c64e397e667`。
- 汇总同一 10 条开发路线上的 direct/subgoal 训练标签 × 在线 bearing 2x2。subgoal/subgoal 为 7/10 成功、1/10 碰撞、2/10 超时；direct/direct 为 0/10 成功、3/10 碰撞。两种交叉分布均失败，其中 subgoal 训练/direct 输入为 10/10 碰撞。结果说明 bearing 条件分布强烈影响控制，但不是自然语言导航结果。
- 按计划做冻结帧 VLM correct/swap/zero 探针。首轮 8 次非空查询有 4 次因小数 near 值或截断触发 fail-closed；显式整数提示及 20-token 上限调整后同帧复跑，8/8 可解析。只有一个正确门目标查询输出 `final`，所有查询均未给出 `near_goal=1`，两个分岔样本的正确/交换输入都输出同一 `R` 方位。
- 修复评测来源元数据歧义，报告新增 `active_bearing_source`，保留 `g0_mode` 原字段。补充 `0.8`、`0.8055` 与截断输出必须安全回退的测试。
- 验证 `test_grounder.py` 25 项、`test_vln.py` 70 项、`py_compile` 与 `git diff --check` 通过。GPU probe 只暴露物理 GPU0；没有训练或启动闭环仿真。

### 没做什么

- 阶段 1 未正式通过，阶段 2 语言闭环与正式对照仍不放行。没有独立门洞通过指标，开发路线已重复用于选型，`text_dim=0` 学生不读指令。
- 没有删除实验数据、旧日志或其他既有结果，没有使用 GPU1，没有启动新训练或 G3 闭环。VLM probe 只验证冻结帧响应，不能支持目标实例定位或 STOP 有效性。
- 没有提交或推送 Git。仓库当前无 remote，工作区含大量既有变更和删除。

## 更新 2026-11-23 16:35 CST

### 做了什么

- 修复 `parse_vlm_output` 自动补齐 JSON 结束符后接受截断停止信号的问题。增加 near 字段完整但尾部截断的回归用例，并拒绝非字符串目标标签。
- 为 `VlmBeliefTracker` 增加指令目标标签门禁。只有最终可见目标短语与 VLM 标签吻合、且连续两次确认时才会通过 STOP。提示词明确输出字段类型和最终目标名称。英文开发集的 10 条表述和中文目的地样例均通过 CPU 检查。
- 并行 subagent 复核了目标标签、路线分岔帧和下一阶段实验。`ts_2` / `ts_3` 参考路径前 20 点相同，原 branch 帧仍在共享前缀；先前相同 `R` 的比较不足以判断语言选路。
- 在物理 GPU0 对 4 帧执行一次新的 correct/swap/zero 探针，耗时 18.86 秒。新提示后 8/8 非空结果为可解析 JSON，near-door 正指令和错误 lighting 指令在 `final/near=1` 与 `subgoal/near=0` 上不同。实验结果见冻结帧记录。
- `test_grounder.py` 系统 Python 与 VLM 环境均 27 项通过，`test_vln.py` 在 VLM 环境 70 项通过；`py_compile` 和 `git diff --check` 通过。系统 Python 的 OpenCV 与 NumPy 二进制不兼容，故完整 VLN 套件在兼容的 VLM 环境验证。

### 没做什么

- 未启动 G3 闭环或训练。分岔帧尚未有效验证，near-door 正样本只有一帧，且模型 `R` 方位下 tracker 不会确认 STOP。
- 未实现图像框/掩码、跨帧目标跟踪或同类实例消歧；当前目标匹配依据指令文本和 VLM 标签。目标可见性与近距离估计仍需更大的独立正负样本集验证。
- 未使用 GPU1，未删除现有实验数据或日志，未新建脚本。Git 未配置 remote，工作区有 433 项既有状态变化，因此没有提交或推送。
