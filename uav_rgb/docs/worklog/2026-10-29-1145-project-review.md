# 并行接手审查、环境修复与真实数据链路冒烟

- 时间：2026-10-29，Asia/Shanghai
- 范围：`code/uav_rgb` 主项目；关联 V8 设计/管理台、CF-Drone 与参考资料只读核对
- 结论：完成本轮接手审查、最小修复和链路冒烟；完整自主导航系统仍是部分实现
- 使用规范：`/root/claude-code-backup-20261028/claude-skills/skills/personal-skill/SKILL.md`
- 用户明确要求 agents 并行：主任务负责环境、实验代码、集成实测与汇总；三个子任务分别负责核心、仿真、UI/设计/硬件。

## 做了什么

### 环境与操作入口

- 先执行 `nvidia-smi -L`，确认当前为 GPU 模式（RTX 2080 Ti），再构建和训练。
- 发现备份 venv 的 `activate`、pip 等入口仍指向 `/root/code/uav_rgb/.venv-tools`。
  `tools/setup_env.sh` 用系统 Python 在原目录刷新 venv，并仅修正该 venv bin 内迁移前的
  Python shebang。保留原有包；numpy 2.5.4、Pillow 13.0.0 已满足依赖，无需下载新包。
- 新增 `tools/requirements.txt`，仅声明数据工具实际所需 numpy/Pillow；不引入模型框架。
- `env.sh` 增加 `DATA_ROOT`、`UAV_RGB_PYTHON`、`UAVRGB_GZ_PYTHON`，移除无用的
  PyTorch 架构变量。重复 source 不重复 PATH/LD_LIBRARY_PATH。
- 新增 `tools/doctor.py`：输出 JSON，检查 GPU/工具链/数据解释器/Gazebo/Tk/数据位置；
  `--require-gpu/--require-sim` 可以把对应缺失变成非零失败。工具存在与功能实测明确分开。
- 新增 `tools/check.sh`：构建、CTest、teacher/scene/gz_io/采集工具/UI 自检；
  显式 `--python-only` 会说明跳过核心构建/GPU，不伪称全链路通过。
- 新增项目 README，纠正 CLAUDE.md“没有 CNN/trainer/Adam”的陈旧描述，并在 HANDOFF
  顶部增加当前复核。保留历史测试和决策记录，旧环境清单不冒充本轮新测。

### 核心与仿真修复

- UAVR 训练读端检查同 scene_id 不跨 split，区间不得重叠/遗漏，拒绝溢出、负偏移、小数。
- CPU/CUDA MaxPool 在尺寸除法前拒绝输入小于窗口，修复越界读取。
- 验证损失按合同使用无权 CE；无 val 时仍允许冒烟训练，但评测事件明确 `split=train`。
- CMake 开启 CXX `-Wall -Wextra` 和 compile_commands；新增/扩充实际失败场景的回归测试。
- 并行采集每个 episode 使用 UUID 分区，并记录分区，避免独立运行共用 `dq0` 串台。
- 完整改动及位置见两个子任务记录，不把同 ID 校验描述为已解决“不同 ID 同几何”问题。

### 真实链路

产物根目录：`/root/rivermind-data/uav_rgb/review-sim-20261029-115357/`。

1. 用修改后的并行编排启动一个独立 Gazebo，seed=42、5 步、hold=0.5，退出 0。
2. 5 张真实 RGB → `preprocessed/dataset.uavr`（288824 bytes），训练集 5、val/test 0。
3. 最终新构建 `build/review-core/uavrgb_train` 执行 1 epoch、batch=5、lr=0.001、
   `--class-weights auto --drop-collision`，输出 `train/`，退出 0。
4. 从 `train/ckpt.bin` 恢复，再训练 1 epoch 到新目录 `resume/`，退出 0，epoch 从 1 接到 2。
5. 检查配置、checkpoint、CSV、事件，记录文件 SHA256 与实际 argv。

| 项 | 初训 | 恢复 |
| --- | --- | --- |
| 训练样本数 | 5 | 5 |
| 加权训练 CE | 1.38612103 | 0.77025795 |
| 无权训练集回看 CE | 0.792164564 | 0.573536813 |
| 训练集回看准确率 | 0.6 | 0.8 |
| epoch | 1 | 2 |
| CLI exit code | 0 | 0 |

这组指标只有链路诊断意义：同一场景、两个动作类别，无独立 val，没有学生闭环。
事件同时包含 `DEV_SPLIT_EMPTY` 和 `validation.split=train`；CSV 的历史列名
`dev_ce/dev_accuracy` 不能脱离事件当作真实验证成绩。

## 阅读范围与证据等级

- `uav_rgb/src` 自研 C++/CUDA、头文件、CMake、测试逐文件读；数据预处理、格式和读取器已读。
- `uav_rgb/sim` 全部现有 Python/SDF/README、相关历史 worklog 已读。
- `uav_rgb/ui` 全部 Python/shell/demo 文本与监控交接已读。
- 主任务补读两个 experiments 的 CMake/CUDA 源码、GPU benchmark 报告、`tools/test_event.cpp`、
  synthetic episode 工具、顶层 README/START_HERE/决策/审核交接与顶层 worklog。
- 实际打开查看本轮首帧 `rgb_00000.png`，确认渲染图包含立柱和场景；目视检查不替代帧hash/传感器验收。
- 核心/仿真/UI 三份子任务记录含具体阅读文件清单；关联 V8、CF 和上游资料的补读范围在其中更新。
- 补读已完成：CF-Drone 枚举的35个源码/驱动/README/LICENSE文本；V8 当前server/tools/tests/cpp/scripts/run.sh/config
  的16个文本文件、frontend三份源码与三份合同/README、docs/00–14。历史preview图片和legacy归档未算当前源码。
- 上游顶层 README/许可证补读确认：九仓库为7个GPLv3、GCOPTER为MIT、Fast-Drone无顶层LICENSE；
  顶层旧README的“8个GPLv3”计数有误，本轮未改目标目录外文件。规划器研究伪代码不能直接照搬，
  航向符号和无效深度处理的问题已在核心子任务更新中说明。
- 阅读源码为 `source_observed`；本轮执行得到结果为 `locally_tested`；旧报告和并发会话报告
  单独标注，不转化为本轮实测。没有把文件枚举、二进制 hash 或 grep 命中当作全文精读。

## 验证到哪一步

| 命令/检查 | 本轮真实结果 |
| --- | --- |
| `nvidia-smi -L` | RTX 2080 Ti |
| `bash tools/setup_env.sh` | exit 0；修复 10 个 console script 的旧 shebang；依赖 already satisfied |
| doctor `--require-gpu --require-sim` | PASS；nvcc 12.8.93、CMake 3.28.3、GCC 13.3、Gazebo 8.15.0、系统 gz 绑定导入成功 |
| `.venv-tools/bin/pip --version` 与 activate 后 `sys.executable` | 均指向当前备份路径 |
| env 重复 source、`bash -n` | PATH/LD_LIBRARY_PATH 不变；三个 shell 脚本语法通过 |
| doctor 设置 `DATA_ROOT=$UAV_RGB_ROOT/data` | 预期 FAIL、退出 1，拒绝源码树内数据根 |
| 核心新构建 + 最终完整 CTest | 13/13 通过，90.86 s；日志在 `build/review-core/Testing/Temporary/LastTest.log` |
| `cuobjdump -lelf build/review-core/libuavrgb_core.a` | 仅 sm_75 |
| `bash tools/check.sh --python-only` | exit 0；teacher 22、分区 2、collector 13、图像 10、gz_io 自检、monitor 56+1跳过、demo 89 |
| 新增 scene 检查命令单独执行 | 100/100 seeds，PASS；已加入 check.sh |
| `DISPLAY=:20 .venv-tools/bin/python ui/test_monitor.py` | 85 项通过，exit 0，含实际 Tk 窗口/进程控制；其中训练进程测试是 DEMO |
| 真实仿真采集 | 5/5 步，exit 0；动作 [2,2,2,1,1]，记录碰撞 0/5，RGB-depth gap 0ms |
| 真实数据预处理/训练/恢复 | 全部 exit 0；最终证据检查 `REAL_DATA_TRAIN_AND_RESUME_PASS` |
| 关联 V8 当前实现补读与验证 | discover 23/23；直接运行测试文件仅21项；独立C++ gradcheck最大误差1.39362e-10，未改V8源码 |

完整默认 `check.sh` 没有再次从头串跑：其构建/CTest 由核心子任务在新目录完成，
Python 部分由主任务运行，新增 scene 行单独验证，避免重复运行约 90 秒的已通过训练测试。

本轮失败也保留说明：初次训练后的证据脚本误用 `type` 而非实际 JSON `event` 字段，
导致断言失败；训练 CLI 已退出 0。读源码确认字段后修正检查并验证原产物，没有重复初训。
另一次 doctor 负例命令误用了工作目录前缀，未启动脚本；修正后得到上述预期退出 1。

## 当前程度与优先级

当前是“有真实数据短链路证据的 RGB 四动作避障研究原型”。不适合给一个总百分比：
训练算子和采集可运行，而完整导航验收的关键模块仍不存在。

| 优先级 | 缺口 | 后续可验收动作 |
| --- | --- | --- |
| P0 | 不同 ID 同几何跨 split | 规范化几何 hash 去重/分组划分；跨集合交集必须为零 |
| P0 | 深度/contact/位姿停更处理不完整 | 冻结新鲜度合同，注入停更并验证拒收/降级 |
| P1 | 正式多场景数据质量与模型训练 | 排除泄漏后采小批，报告四类分布/碰撞/失败率/独立val |
| P1 | SimulatorEval 无实现 | 学生实时 RGB → 动作 → 物理仿真，记录闭环成功/碰撞/误前向/时延 |
| P1 | UI 启动追加 --config 与原生CLI不兼容 | UI维护会话对齐 dataset 输入和启动合同；当前用CLI训练、UI监视 |
| P2 | NPU/ESP32 未联调 | 获取实物型号与协议，按合同验证，不猜 UART/帧率 |

几何泄漏实测：seed 0..1999 只有 1124 种布局，241 组跨 split；seed4/train 与1618/val
完全相同。完整复现见仿真记录。世界文本 hash 包含 seed 注释，不能证明几何不同。

审查过程中另一个会话新增 `2026-10-29-1211-scene-density.md`，报告 120 步中 FORWARD=80%、
非前进20%、碰撞1/120。本任务读到该记录，但未重新核验其全部原始产物；不计入本轮5步
实测，也不覆盖几何泄漏/单一车道结构的结论。源码 mtime 显示场景生成器仍为既有10:05版本。

## 没做什么

- 未声称整个 code 树所有内容都完成全文审计：第三方约1GB仓库、论文PDF、ZIP/legacy快照、
  venv第三方包、build生成物、全部历史日志/图片没有逐字逐页审读；本轮先覆盖当前自研主线。
- 没有开展200/2000 episode正式训练：几何跨split与传感器质量门禁未完成，先确认短链路。
- 没有改 `ui/` 和 `sim/tools/collect.py`：交接明确并发维护，本次分工保持只读，记录具体问题。
- 没有修改第三方/CF固件、刷机、接串口、操作电机、部署服务或发外部消息；当前任务是接手开发环境。
- 没有实现语言/GRU/ECPT、SimulatorEval、NPU导出或device-resident引擎；这些是后续独立验收模块。
- 没有重跑完整GPU性能基准、计算消毒器/所有畸形输入/真实无卡机器测试；现有基准仅作为历史参考。
- 没有升级系统CUDA/驱动/内核，已有工具满足本轮；没有安装被禁止的训练框架。
- 没有建立Git仓库或提交：当前uav_rgb无Git，不能把改后快照冒充改前基线；改动记录与实际文件留存。

## 待确认

- 荔枝派具体型号、板卡系统和NPU SDK；ESP32板型、UART接线及四动作协议，仓内缺实物证据。
- 正式数据分组去重策略与传感器允许残差/停更阈值，需要后续与实现一起冻结和验证。

## 风险与遗留

五帧回看准确率没有泛化意义；无碰撞短轨迹也不是碰撞率验收。
当前元数据仍是最小扫描器，checkpoint/CLI数值边界、resume随机状态等遗留见核心记录。
任何后续改动都需要按当时源码和运行证据更新本记录，不能沿用旧“已完成”标签。

## 更新 2026-10-29 13:00：GPU空闲与并行架构专项复核

### 做了什么

- 延续用户提出的GPU闲置/难并行问题，两个并行只读agent分别核对执行架构与并发正确性；
  主任务构建整网探针，重新实测，不用旧128×128单层benchmark推断当前CNN。
- 新增 `experiments/pipeline_profile/CMakeLists.txt`、`pipeline_profile.cpp` 和 README。
  链接实际核心函数，分段测量120×160输入的读取/前向/CE反向/模型反向/裁剪/Adam。
- 先确认GPU仍为2080Ti，使用当前CUDA12.8；建立独立build/pipeline-profile。
- 创建256张均衡标签的合成噪声UAVR，仅用于吞吐测试；数据与Nsight证据位于
  `/root/rivermind-data/uav_rgb/pipeline-profile-20261029/`，保存关键源码和二进制SHA256。
- 读取当前源码确认Tensor仍为host vector，算子每次显存分配、全设备同步及全量往返；
  读取器mutable ifstream不能直接供多个worker并发seek/read；输出目录缺少单writer所有权。
- 重新核实几何跨split与停更问题：2000 seeds仍1124布局/241组跨split；假死depth订阅
  仍可返回99秒前帧，假死contact订阅返回False。这是纯函数故障复现，不是实际停更物理实验。

### 验证到哪一步

| 检查 | 结果 |
| --- | --- |
| 探针新构建 | PASS；初次因漏include algorithm失败，补齐后编译通过，无核心源码更改 |
| batch=1/8/32/128，2步热身+4步 | 均exit0，有限loss及参数检查通过；39.84/47.80/55.77/59.21样本/s |
| batch32，热身后Nsight捕获8步 | exit0；单步505.66ms；已导出SQLite与4类CSV |
| 非法batch=0 | 预期exit1，`profile: 参数须为正整数` |
| API统计 | 每步malloc/free各121，memcpy146，device-sync47，kernel47 |
| 传输 | 每步H2D+D2H约766.4MB；对应输入float32仅7.37MB |
| 时间线 | 8步kernel814.79ms/copy1304.19ms；本进程GPU活动并集2119.95ms/窗口4033.85ms |
| kernel瓶颈 | 卷积dW62.5%+db28.3%=90.8% kernel时间；dW实际网格2/18/72blocks，db均1block |

准确口径：本进程kernel覆盖时间窗约20.2%，约47.4%时间窗没有捕获到本进程GPU操作；
不能称作全设备空闲率或SM利用率。CPU API耗时包含GPU等待，不与kernel/copy耗时再相加。
batch32四步短测573.77ms，另一次Nsight八步505.66ms存在运行抖动，不能解释为profiler加速。
完整方法、数据和复跑命令见 `experiments/pipeline_profile/README.md`。

### 判断与改造边界

当前host-in/host-out算子适合作为数值参考，没有形成设备常驻训练执行器。
需要同时改内存所有权/执行模型和归约算法：单compute stream常驻整步 → dW/db并行归约
→ 双pinned buffer预取和copy/compute重叠 → 融合与调度优化。当前热缓存读取仅约0.5%，
只加加载线程不是首要收益来源。不可同时并发更新同一模型/Adam，不可异步保存可变状态。

采集并发还有语义问题：`collect.py`按墙钟hold控制，RTF变低时同一动作模拟时长变短；
需先固定sim-time动作窗口，比较有效模拟秒/墙钟秒和合格样本率，不能只看steps/s。
传感器unknown被当无碰撞、不同ID同几何泄漏，会令更快训练产生不可信实验结论。

并发维护事实：本轮期间另一个会话修改trainer并新增eval_metrics。最终读取源码已看到
CSV train_ce改为无权口径，评测增加混淆矩阵/宏指标/test。上文旧“加权train_ce汇总错误”
属于先前快照，不能作为当前未修缺陷。当前仍存在每batch按权重和归一化的目标语义：
batch=1时正权重会约掉；性能调batch不能不控制目标/梯度累积。探针使用无权CE并单独说明。

### 没做什么

- 没有直接重写生产训练器或修改CUDA kernel：本轮是先定位架构和正确性问题，新增可复跑探针；
  不把新的性能证据声称为已提升性能。resident路径需独立梯度/更新/checkpoint等价验收。
- 没有重跑核心完整CTest：核心未由本任务修改；探针新构建、实际执行与边界检查已完成。
  不能把上一轮13项数字保证给其他会话随后修改的源码。
- 没有测试硬件occupancy、冷盘大型数据、并发模型更新/同目录写竞争、真实传感器停更；
  分别缺本轮采样范围或无需冒着污染训练输出的风险证明源码已有的所有权缺口。
- 没有宣称预计3倍或某个固定加速比；下一轮必须以固定输入/批序列、热身、重复测量和整步吞吐验收。

### 待确认与遗留

本轮无需额外用户许可；后续实现应明确设备buffer/view/context/workspace的合同和错误检查边界。
resume随机状态、checkpoint部分加载、CLI有限值校验等正确性遗留仍需单独处理。

## 更新 2026-10-29 13:10：并行训练改造方案交付

### 做了什么

- 按用户“出一套方案”请求新增 `docs/PARALLEL_TRAINING_PLAN.md`，并从项目README链接。
- 复核当前CMake、trainer、checkpoint、dataset和已有profile记录；现有宏指标/test评测及无权
  train_ce已写入方案基线，不将其他会话已完成能力重复列为待实现。
- 明确P0到P4、设备对象所有权、device view/stream/workspace合同、归约与预取实施顺序、
  数据正确性并行工作线、恢复兼容、逐阶段验收和4槽agent分工。
- 区分结构性硬目标（稳态无分配、无层间往返）和暂定吞吐目标（同条件2倍，未实测承诺）。
- 规定3次重复、固定批序列、正确性先行、GPU性能实验排队、CPU工作并行，以及失败/回退行为。

### 没做什么

- 没有新增resident执行器、改CUDA kernel或启动新的改写agent：本次请求是方案，设计接口均标为拟实现。
- 没有重跑GPU基准或完整CTest：本次只改文档，复用上一轮保存的实测证据并写清限制。
- 没有建立Git、修改数据集/硬件/环境或决定传感器阈值：相关动作作为实施步骤，阈值需先实测冻结。
- 没有保证2倍性能或GPU满载：它是工程目标，须按固定工作负载和重复测量验收。

### 验证到哪一步

- 当前关键接口、profile口径与方案描述人工核对完成。
- README相对文件链接检查通过；方案Markdown代码围栏成对检查通过。
- 未运行代码测试（文档修改，不改变执行逻辑）。

### 待确认与风险遗留

方案不需要额外信息即可开始P0/P1；硬件型号仍不影响服务器侧执行器改造。
将来改变CLI、数据质量合同、checkpoint格式时需要版本化兼容和回归；不能只依据文档视为已实现。

## 更新 2026-10-29 16:15：方案细化到接口、算法与任务验收

### 做了什么

- 根据用户“方案详细点”的要求，在原 `docs/PARALLEL_TRAINING_PLAN.md` 扩充第10–22节，
  保留原1–9节总体方案并增加角色阅读指引，不另建互相分叉的方案。
- 复读当前nn/checkpoint/episode/action合同，细化device view、容量/别名/错误处理、
  张量生命周期与完整训练step提交边界；新API和CLI仍明确标为拟实现。
- 根据实际120×160 CNN尺寸计算forward缓存、反向临时区、双输入槽和parameter/gradient/m/v
  预算，说明是规划公式而非显存实测。
- 给出db/dW两级归约、norm/FC/loss算法实施顺序，读取/stream事件状态机、输出锁、
  checkpoint v3分区字段草案、事务保存/加载与旧格式兼容行为。
- 增加仿真几何分组/unknown碰撞/sim-time控制细则、测试矩阵、benchmark JSON格式，
  T00–T10依赖表和12批提交顺序，明确共享文件与GPU测试的单一负责人。

### 没做什么

- 未改生产代码、运行训练或重测性能：本次是扩写方案，所有性能数字引用已保存实测。
- 未把接口示例编译成真实API、未冻结checkpoint字节偏移或数据新鲜度数值阈值：
  这些须在对应实施任务中用合同/测试确定，不能从设计文档推导已可运行。
- 未建立Git、创建工作树、安装依赖或启动实现agent：用户当前要求详细方案，不是立即执行重构。

### 验证到哪一步

- 用Python按实际网络shape核算缓存与参数数量：100388参数；batch32的forward缓存约171.094MiB，
  两个反向最大临时区75MiB，额外输入槽7.031MiB，四组参数状态合计约1.532MiB。
- 文档围栏、相对链接、章节编号以及T00–T10任务表完整性检查通过。
- 人工复核step编号仅在更新结果有效后提交、pinned/device复用边界、v2非精确恢复等约定。
- 未运行代码测试；文档变更不改变现有执行逻辑。

### 待确认与风险遗留

实施时仍须以当前源码与实际硬件状态重建基线；本文显存公式未包含全部scratch/context开销。
新归约容差须先按数值分析和固定用例冻结，具体性能、恢复确定性和传感器阈值都须实测，
不可把详细方案当作已完成的工程能力。

## 更新 2026-10-29 16:42：补齐可落代码的实现过程

### 做了什么

- 按用户明确要求“怎么实现”，在原方案新增23–31节，首页指引直接阅读实现部分。
- 对照现有conv/nn/softmax/model源码，写明host返回值接口如何改为device输出参数，
  launch层禁止哪些分配/拷贝/同步调用，以及Trainer如何接入backend。
- 给出8组参数精确偏移、arena分配方法、forward图和反向A/B显存读写表。
- 提供256线程block归约、dW partial与最终归约CUDA骨架，说明grid、R/S映射、
  scratch预算、db复用和边界检查；补齐loss/clip/Adam的有效性屏障和step提交顺序。
- 写明双缓冲stream/event实际调用顺序、pread读满、flock和事务快照的实现步骤。
- 最小版本拆到8个可以分别编码/运行的步骤，并规定4组性能对照拆分收益。

### 没做什么

- 未把文档骨架写入生产源码或编译CUDA示例：用户本次仍要求方案实现细节，
  示例依赖的检查/封装类型尚待对应任务实现，不声称可直接运行。
- 未重跑训练、GPU性能或修改数据：本次只扩写文档和验证索引公式。

### 验证到哪一步

- Python核验参数offset/count连续且总数100388。
- 多组R/S检查归约分片完整覆盖[0,R)，无重叠遗漏；遍历三层全部权重验证flatten索引往返。
- 文档第1–31节编号连续、代码围栏成对检查通过。
- CUDA骨架未编译、未GPU执行；上述仅为CPU侧公式检查，不代替数值/内存正确性测试。

### 待确认与风险遗留

两级归约扩大grid不等于必然加速，tile/split需分别测量；文档给出的是首版算法与可调参数。
新增接口、例外路径和硬件执行错误处理必须随实现测试，不得以文档检查通过代替代码验收。

## 更新 2026-10-29 17:22：补齐实现接线、状态提交与失败语义

### 做了什么

- 读取personal-skill/ponytail、当前CLAUDE/HANDOFF和指标合同；重新核对trainer、dataset、
  checkpoint、model、nn/linear/softmax与既有测试调用。安排两个只读agent分别审查数学接口和流水线。
- 在原方案`docs/PARALLEL_TRAINING_PLAN.md`补充第32–38节：逐文件修改位置、异步backend、
  host模型回填、完整调度循环、尾批、评测聚合、恢复游标、SIGINT和写入失败的处理规则。
- 直接修正前文：submit/finish拆分；大logits的CE先减再加；ReLU原地反传不复用冲突的restrict签名；
  rename后目录fsync失败不能声称旧目标未变；明确v2已有m/v/t但不能精确恢复。
- 明确P1 float argmax及容量限制、P2 uint32迁移；首个resident用v2兼容桥接，v3精确恢复先支持epoch边界。
- 增加具体测试输入/断言、待基线校准的数值容差、未来构建与sanitizer命令，写明新target当前不存在。
- 将第一批实现工单限定到buffer/context/state与往返测试，明确agent文件归属及GPU验证串行执行。

### 没做什么

- 未改生产代码、安装依赖、运行训练或重测性能：用户当前要求补充方案细节。
- 未编译文档kernel/接口、未执行新增测试或故障注入：这些仍是实现规格，不能当已交付能力。
- 未冻结v3字节级格式和校准数值容差：分别留给codec实现和固定版本基线；文档显式限定精确恢复承诺。
- 未修改UI、采集脚本及其他会话代码：本次修改仅为现有方案与本工作记录。

### 验证到哪一步

- `nvidia-smi -L`确认当前有RTX 2080 Ti；本次无GPU计算任务。
- Python文档检查实际输出：`PASS: headings 1-38; paired fences; revised interface/formula/failure contracts; referenced existing paths`。
- 检查了章节连续、围栏成对、旧同步接口/错误公式/错误保存承诺已移除、关键合同存在及当前源码引用路径存在。
- 人工对照两个agent意见，统一argmax阶段选择、停止时禁止发布半个epoch、v2与v3恢复能力表述。
- 没有新增性能/训练实验，因此本次不新增实验结果数字；以上静态检查不证明CUDA代码正确。

### 待确认与风险遗留

本次无须用户额外确认。详细方案仍未实现resident训练器，实际吞吐、浮点容差与恢复确定性须编码后验收。
旧文件可能被其他会话修改，实现前必须重新读取；网络文件系统卡死和GPU驱动失联的有界退出不在首版保证范围。
