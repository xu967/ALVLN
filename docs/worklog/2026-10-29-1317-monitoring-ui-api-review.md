# 监控软件 UI 与 API 只读审查

- 时间：2026-10-29 13:17
- 范围：TinyAerialVLN_Final_V8 的 frontend、server、接口契约、测试及现有预览图
- 结论：完成只读审查、CSV/worker 收尾、任务参数校验、审核包一致性、托管任务跨文件静止快照及 stopped 生命周期修复；Git 仓库已初始化并提交

## 做了什么

- 并行安排两个审查代理：一个检查 UI、响应式、可访问性和截图，另一个核对后端路由、前端请求、接口契约和任务生命周期。
- 阅读 frontend/index.html、frontend/styles.css、frontend/app.js、server/app.py、server/core.py、相关契约和 tests/test_engineering.py。
- 检查现有 frontend/preview_*.png，定位首屏信息优先级、低对比度、移动端图表和状态反馈问题。
- 枚举前端 fetch/EventSource 调用并核对后端路由；确认主要训练/仿真路径和字段基本匹配，但状态处理、参数校验和异步竞态存在缺口。
- 运行 ./run.sh test，23 项测试全部通过。

## 没做什么

- 首轮只读审查没有修改 frontend、server、配置或测试代码；后续 CSV/worker 修复见下方更新。
- 没有启动持久化监控服务或连接真实训练、仿真后端；当前默认注册表为空，真实 profile 行为无法在本轮验证。
- 没有做浏览器动态交互、SSE 断线重连和快速切换的自动化复现；这些竞态由代码路径审查确认，仍需修复后补浏览器测试。
- 没有运行 GPU、Gazebo、C++ 或真实飞行验证；不属于本次 UI/API 审查范围。
- 没有实施 UI 改版；视觉方向和信息架构需用户确认后再落地。

## 待确认

- UI 是否按“Run/Job 为中心的监控操作台”方向改版，而不是仅调整配色和卡片样式。
- 第一轮是否同时修复高优先级 API 问题，建议至少覆盖 CSV 异常导致任务卡死、服务端参数校验、审核包一致性、停止状态和空 Run 演示状态混淆。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 两个只读审查代理 | 完成，分别返回带文件/行号的 UI 与 API 结论 |
| ./run.sh test | 通过：Ran 23 tests in 3.089s，OK |
| 浏览器动态复现 | 未跑：本轮为只读审查，未启动持久化服务 |
| 真实训练/仿真联调 | 未跑：默认注册表为空且超出本轮范围 |

## 风险与遗留

- server/core.py 的 CSV 解析异常可能逃出 worker，使任务停留在 running 且进程清理不完整。
- 后端缺少完整任务参数校验，浏览器校验可被直接 API 请求绕过。
- 运行中构建审核包时，ZIP 内容与 manifest 哈希/大小可能来自不同文件时刻。
- 用户停止可能被记录为 failed 或 finished，停止过程还存在进程退出竞态。
- 空服务器自动连接后仍可能保留演示 Run，造成真实/演示数据来源混淆。
- 前端断线、导入和快速切换缺少统一的订阅清理与旧响应失效机制。
- UI 的 9–11px 低对比文字、大面积装饰 Hero 和移动端整图缩放，直接影响监控可读性。
- 现有测试全部通过，但没有覆盖上述多数异常、竞态和浏览器状态场景，不能把测试通过等同于这些问题不存在。

## 更新 2026-10-29 16:05

### 做了什么

- server/core.py：worker 的 CSV 同步增加 csv.Error 捕获，超长字段等解析异常不再直接杀死监控线程。
- server/core.py：监控循环和最终同步放入 try/finally；即使发生未预期监控异常，也会等待已退出进程、写入 finished/failed、ended_at、exit_code 和状态事件。
- server/core.py：生命周期收尾增加内层 finally，确保状态写入或事件记录失败时仍清理 _processes 中的任务句柄。
- server/app.py：GET /runs/{id}/metrics 的 csv.Error 统一映射为 HTTP 500 READ_FAILED JSON，不再无响应断连。
- tests/test_engineering.py：新增超长 CSV 字段、未预期监控异常和 HTTP JSON 错误响应三项回归测试。

### 没做什么

- 没有修改 CSV 字段上限或自动接受超长字段；原因是解析失败时应保留原始证据并明确报错，而不是静默放宽输入限制。
- 没有处理停止状态、审核包一致性、参数校验或前端竞态；原因是用户本轮只要求优先修复 CSV 异常和任务收尾。
- 没有运行真实长训练任务；原因是默认注册表为空，使用 Mock 覆盖 worker 异常路径，并保留现有真实子进程集成测试验证。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 修复前三项定向回归 | 按预期失败：两个 ERROR、一个 FAIL，复现 csv.Error、断连和 _processes 未清理 |
| 修复后三项定向回归 | 通过：Ran 3 tests，OK |
| ./run.sh test | 通过：Ran 26 tests in 3.044s，OK |

### 风险与遗留

- 永久性畸形 CSV 会在后续轮询中继续被跳过；worker 和任务状态不再卡死，但畸形指标本身仍需修复数据来源。
- 其他已审查 API 与 UI 问题仍保留，未因本轮修复而视为解决。

## 更新 2026-10-29 16:30

### 做了什么

- server/core.py：新增统一 _validate_payload，在查询 profile 和创建任务目录前校验训练/仿真请求。
- 训练请求现在要求合同中的完整字段；校验名称和 ID、epochs 1–5000、正整数 sequence_chunk_size、有限正数 lr、非负整数 seed，并拒绝 bool 冒充整数。
- 仿真请求校验资源 ID 和 mode；auto_launch、record_observations、audit_artifacts 只接受真正布尔值，字符串 false 不再被当成真。
- tests/test_engineering.py：新增训练字段缺失/类型/范围、仿真布尔类型和 HTTP 400 回归；现有有效启动测试补齐合同字段。
- docs/14_UI_BACKEND_OPERATION.md、docs/TEST_REPORT_LOCAL.md：同步实际校验规则与 29 项测试结果。

### 没做什么

- 没有执行 Git commit：TinyAerialVLN_Final_V8 及其父目录均不是 Git 仓库；未擅自 git init，避免形成不包含完整项目历史的异常初始提交。
- 没有为 sequence_chunk_size、seed 或 lr 增加未在项目合同中定义的上限；目前只落实已有类型和范围约束。
- 没有处理审核包一致性、停止状态、前端竞态或 UI 改版；这些属于后续独立问题。
- 没有运行真实训练或仿真；默认注册表为空，使用 HTTP 测试、Mock 和既有受控子进程集成测试验证。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 修复前参数校验定向回归 | 按预期失败：两个 AttributeError、两个 HTTP 状态断言失败 |
| 修复后五项定向回归 | 通过：Ran 5 tests，OK；包含有效子进程启动回归 |
| ./run.sh test | 通过：Ran 29 tests in 3.026s，OK |
| Git 提交 | 阻塞：git rev-parse 返回 not a git repository |

### 风险与遗留

- 训练完整字段现在是服务端硬要求；直接调用旧客户端若省略 name、sequence_chunk_size、lr 或 seed，会收到 400，需要按合同补齐。
- 其他 API/UI 审查项仍未解决，不能因 29 项测试通过而视为全部完成。

## 更新 2026-10-29 16:50

### 做了什么

- server/core.py：活动审核任务先暂停独占进程组，使用 Linux /proc 确认领导进程进入停止态；随后读取全部证据文件，finally 中恢复进程组，再在任务恢复后压缩 ZIP。
- 快照读取和 worker 的 sync_csv 现在共用全局锁，防止训练进程已暂停但服务自身仍写 lifecycle/events 而混入不同状态。
- 暂停、等待或读取失败会恢复进程组并抛出错误，不返回声称原子但实际降级的审核包。
- tests/test_engineering.py：真实受控子进程测试确认每个文件读取时进程处于 T 状态、归档返回后恢复；另测读取异常和 /proc 暂停确认失败的恢复路径。
- docs/14_UI_BACKEND_OPERATION.md、docs/TEST_REPORT_LOCAL.md：同步静止快照语义、实时暂停风险和 33 项测试结果。

### 没做什么

- 没有实现文件系统级快照，也不能保证外部程序、脱离任务进程组的写入者或服务重启后的孤儿进程停止写入；这些场景需要 btrfs/ZFS/LVM 快照或写入方协议配合。
- 没有对真实实时仿真或硬件做暂停安全验证；SIGSTOP 不暂停墙钟时间，可能触发 RPC、看门狗或控制超时。
- 没有实现流式 ZIP 或总大小限制；快照读取期间会同时持有纳入归档的文件字节，既有内存风险仍在。
- 没有执行 Git commit；目录仍不是 Git 仓库。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 活动进程与异常路径定向回归 | 通过：4 项测试，包含真实子进程停止态读取、恢复及既有审核包流程 |
| 暂停失败恢复 | 通过：模拟 /proc 不可用，验证 SIGSTOP 后发送 SIGCONT |
| ./run.sh test | 通过：Ran 33 tests in 3.058s，OK |
| 真实实时仿真暂停安全性 | 未跑：超出安全验证范围，不能据此宣称无超时风险 |

### 风险与遗留

- 原子性边界是当前服务、当前活动任务进程组及 worker；不能扩大解释为任意外部写入者的文件系统原子快照。
- 下载审核包会短暂暂停活动任务；实时关键阶段应避免下载，后续如需零暂停能力应改用文件系统快照。

## 更新 2026-10-29 16:54

### 做了什么

- server/core.py：stop_job 在发送 SIGTERM 成功后持久化 status=stopping、stop_requested=true 和 stop_requested_at。
- worker 收尾优先检查持久化停止意图；进程返回 0 或 SIGTERM 负码都记录 stopped，不再误报 finished/failed。
- stop_job 捕获 killpg 的 ProcessLookupError 并映射 JOB_NOT_ACTIVE；进程已在信号前退出时不写入 stopping，避免自然结束被误标。
- tests/test_engineering.py：新增真实 SIGTERM 停止、优雅退出码 0 映射、进程退出竞态三项回归；三个测试修复前分别失败并复现问题。
- docs/14_UI_BACKEND_OPERATION.md、docs/TEST_REPORT_LOCAL.md：同步 stopped 状态语义、竞态行为和 36 项测试结果。

### 没做什么

- 没有实现 SIGTERM 超时后升级 SIGKILL；忽略 SIGTERM 的任务仍会停在 stopping，需要后续策略。
- 没有修改前端停止按钮或轮询终态；前端当前未使用 stop API，该问题仍在后续范围。
- 没有用真实训练器验证其 SIGTERM 处理行为；使用受控子进程验证生命周期逻辑。
- 该节点尚未执行 Git commit，因为目录不是仓库；仓库初始化与提交见下一节。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 修复前停止定向回归 | 按预期失败：graceful stop 记成 finished、竞态抛 ProcessLookupError、真实停止未写 stopping |
| 修复后三项定向回归 | 通过：Ran 3 tests，OK |
| ./run.sh test | 通过：Ran 36 tests in 3.066s，OK |
| 真实训练器信号处理 | 未跑：默认注册表为空，且不属于生命周期单测范围 |

### 风险与遗留

- 停止请求只发送一次 SIGTERM；若外部程序忽略信号，状态会长期保持 stopping，运维仍需人工确认。
- 用户停止与崩溃几乎同时发生时，只要 SIGTERM 发送成功，最终会记 stopped；原始 exit_code 仍保留用于审计。

## 更新 2026-10-29 2026-10-29 17:07

### 做了什么

- 在 TinyAerialVLN_Final_V8 初始化 Git 仓库，默认分支 main。
- 设置仓库级提交身份 CodeBuddy <codebuddy@localhost>，未修改全局 Git 配置。
- 将当前项目 122 个非忽略文件纳入初始提交 921b48c，包含源码、文档、预览图、审核记录和 references 快照 ZIP。
- 提交信息记录本次生命周期、参数校验、审核快照和 stopped 状态修复，并注明 36 项测试通过。

### 没做什么

- 没有配置远端仓库，也没有 push；当前仅本地提交。
- 没有重建历史提交或生成修复前基线；首次导入包含当前已修复状态，无法在 Git 历史中单独展示修复 diff。
- 没有排除 24.5 MB 的 references/snapshots/OpenFly-Platform.zip；该文件未被现有 .gitignore 忽略，保留为项目快照证据。
- 没有修改 .gitignore 或清理 .ipynb_checkpoints；保持原包内容不变。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| git init | 成功：main 分支 |
| git commit | 成功：921b48c，root-commit，122 files changed，16201 insertions |
| 提交前测试 | 通过：Ran 36 tests in 3.066s，OK |
| 远端推送 | 未执行：未配置 remote |

### 风险与遗留

- 初始提交无法替代真实开发历史；后续修复应从该提交开始正常增量提交。
- 本地提交身份为 CodeBuddy，若需要归属个人或团队，应在后续提交前调整仓库级 user.name/user.email。

## 更新 2026-10-29 16:38

### 做了什么

- server/core.py：审核包不再分别执行 z.write、sha(path)、stat，而是每个证据文件只 read_bytes 一次。
- ZIP 写入、SHA256 和 bytes 字段现在统一使用该次读取的 data；读取后再次检查 20 MB 上限，覆盖读取期间文件增长的情况。
- tests/test_engineering.py：新增文件在归档与哈希之间被修改的确定性回归，逐项验证 ZIP 字节与 manifest 大小、哈希一致。
- docs/14_UI_BACKEND_OPERATION.md、docs/TEST_REPORT_LOCAL.md：同步快照语义与 30 项测试结果。

### 没做什么

- 没有实现流式 ZIP、总大小限制或并发下载限制；这些属于独立资源风险，不是本次一致性修复所需。
- 没有把运行中文件标记为不可变或停止训练后再归档；当前保证的是包内内容与 manifest 描述同一份字节，不代表整个任务目录是同一时刻的全局快照。
- 没有处理停止状态、前端竞态或其他 API/UI 问题；超出本次范围。
- 没有执行 Git commit；项目仍不是 Git 仓库，阻塞原因与前次相同。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 修复前一致性回归 | 按预期失败：ZIP 为 6 字节，manifest 记录 16 字节 |
| 修复后两项定向回归 | 通过：一致性测试及真实受控子进程审核包测试均 OK |
| ./run.sh test | 通过：Ran 30 tests in 3.064s，OK |

### 风险与遗留

- 此前不同文件可能在各自时刻读取；后续更新已在当前服务托管写入者范围内实现跨文件静止快照，外部写入者限制见最新一节。
