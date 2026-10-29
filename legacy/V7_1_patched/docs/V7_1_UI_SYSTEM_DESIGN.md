# TinyAerialVLN V7.1：研究工作台 UI 与可观测性设计说明

> 文档属性：本次实际修改说明。相关原理讨论/模拟器设计保留在根目录 `V7_1_RESEARCH_PLAN.md`，本文件只陈述 V7.1 UI 系统的事实与设计合同。
>
> 状态约束：此次 **没有改动 CF-Drone ESP32 固件**；没有新训练器、没有通过 GZ0–GZ3、没有真实 SG2002 设备测量。

## 1. 修改动机与设计目标

原 V7 单页 dashboard 已有 `artifacts/runs/<id>/metrics.jsonl` 训练曲线，但存在五个明显空缺：

1. 看板只提供训练进度和两幅曲线，无法查看仿真/板端的独立状态。
2. 场景文件存在容易造成“Gazebo 已经闭环”的错觉。
3. 无直接用于第三方 AI 的结构化工程进度报告。
4. 日志异常、日志空值、尚未训练没有明确区分。
5. 原布局信息密度有限，训练、仿真、部署和评审混在一页。

V7.1 目标：统一视觉风格；不依赖 React/Vue/Node；后端仅本地绑定；数据可审计；五个页面真实可用；训练指标从 JSONL 原始记录来；未完成任务明确标为“尚未实现”。

## 2. 信息架构

| 页面 | 用途 | 输入来源 | 明确不承担的责任 |
|---|---|---|---|
| 项目总览 | 实验记录总数、静态场景数、开发门禁、快速命令 | `/api/status`,`/api/runs`,`/api/scenes` | 不证明动力学或训练功能 |
| 训练监控 | Step/Total、Loss、Accuracy、Val SR、STOP F1、两张曲线 | `/api/metrics` | 不能从静态场景推断训练成绩 |
| 场景与仿真 | 生成场景 Manifest、Gazebo/CF 控制器状态 | `/api/scenes`,`/api/status` | 不在网页执行 Gazebo 命令 |
| 边缘部署 | SG2002、UART、ESP32 固件状态与未来资源合同 | `/api/status` | 不发 UART 指令、不刷机 |
| 日志与评审 | 事件过滤、独立审计 JSON 导出 | `/api/metrics`,`/api/audit` | 不代替运行测试或学术审稿 |

导航用原生按钮和 hash 页面切换；桌面侧边栏，窄屏变成横向导航。缩放与响应式仅是前端静态逻辑，必须在实际 Ubuntu 浏览器复测。所有业务表格使用 `textContent` 构造，避免原始日志文本被解释为 HTML。

## 3. 指标渲染合同（不能伪造）

训练 JSONL 示例仅说明格式，不意味着工程已有该事件：

```json
{"event":"train_step","step":8,"total_steps":120,"loss":1.338,"action_accuracy":0.44,"lr":0.0003,"elapsed_sec":10.2,"eta_sec":142.8}
```

```json
{"event":"validation","step":120,"sr":0.032,"stop_f1":0.21}
```

**重要：以上数值是格式说明例子，不是真实训练记录。** 本项目 ZIP 不包含这些示例事件，也不会在页面填充虚构曲线。

- `train_step`：从最新非空完整事件读取 loss、action_accuracy、step/total、ETA、learning_rate。
- `validation`：从独立验证事件读取 SR 与 STOP F1，不将训练准确率冒充 SR。
- 缺失数据：显示 `—`，并明确“暂无真实指标”。
- loss、accuracy、SR、STOP F1 不得从看板自行生成。
- 进度条是 `clamp(step/total,0,1)`，只有 `total_steps>0` 且 step 数值合法才有具体百分比。
- ETA 仅显示训练器记录的 `eta_sec`，不臆测速度。
- 图表 X 轴优先使用记录里的 step，空集合显示文字。
- 2MB bounded tail：避免刷新时将巨量日志全部读入内存；若截断，页面显示警告。
- 末行无换行：视为进行中的不完整写入，跳过；不误判模型失败。
- 日志行格式损坏、非有限数值：返回明确警告；不是零值替代。

## 4. 后端 API 合同

服务位置 `tools/dashboard.py`，默认 127.0.0.1:8765。无 POST/PUT/DELETE 功能。

| API | Method | 主要字段 | 错误行为 |
|---|---|---|---|
| `/api/health` | GET | `status, read_only` | 服务不可用时网络错误 |
| `/api/status` | GET | `version,generated_at,checks,run_count,scene_count` | I/O 问题 500 |
| `/api/runs` | GET | `runs[]` | 无数据给空数组 |
| `/api/metrics?run=<id>` | GET | `events,warnings,truncated,source` | 不合法 run 404 |
| `/api/scenes` | GET | `scenes[]` | 破损 manifest 标为 invalid |
| `/api/audit` | GET | `evidence_levels,explicit_limitations,status` | I/O 问题 500 |

API 不执行 shell 命令；不会启动训练、Gazebo 或运行板端驱动；只读指标与本地工程信息。运行目录名称要求 `^[A-Za-z0-9_]{1,70}$`，排除符号链接目录与日志文件。当前版本预留未来鉴权扩展，但**仅允许回环地址访问**；不要通过反向代理对公网暴露。

日志裁剪：取文件尾至多 2MiB，拒绝过长单行，最多返回 2000 个事件；这不是全量实验导出 API。完整指标应使用原始 `metrics.jsonl` 与哈希核对。

## 5. 状态术语严格定义

- `ready`：本地文件或只读服务就绪，不等于物理功能已通过实验。
- `available_unverified`：发现可用命令或目标文件，但未进行对应系统验收。
- `blocked`：缺少运行前置条件，如 Gazebo/配置文件。
- `not_implemented`：缺实现，不能绕过。
- `frozen`：明确禁止本轮变更，如 CF-Drone 固件。

例如 Gazebo CLI 存在，最多使 `gazebo=available_unverified`；绝不能据此让 `physics=ready`。静态 forest SDF 存在，最多使 `scenes=ready`；不意味着前视 RGB 和四电机物理闭环。

## 6. UI 交互与可用性

- “立即刷新”调用真实只读 GET。
- “暂停自动刷新”只暂停浏览器轮询，不暂停训练进程。
- 下拉实验列表由本地指标文件实时建立，避免演示记录混入。
- “复制”只复制 shell 命令；网页不会执行命令。
- “导出审核 JSON”把 `/api/audit` 结果保存到本地，不能替代源代码审核。
- “事件筛选”只过滤当前已加载的日志尾部，不影响训练端文件。
- 图表支持任意窗口尺寸、视网膜屏渲染；空数据只显示空状态。
- 导航按钮支持键盘 focus，缺失的指标不显示为 0%。

## 7. 为什么不做启动训练按钮

当前 C++/CUDA 自研训练器未完成，`./run.sh train` 会因 `BINARY_NOT_BUILT` 失败。现在增加“开始训练”按钮会制造接口已具备的错误预期；允许将 shell 命令复制到终端，且只在验证 G3/G4 后考虑经过权限控制的本地任务调度功能。若未来需要 UI 一键训练，必须独立审查：任务队列、进程组生命周期、日志权限、Stop/Resume 与 checkpoint 一致性，禁止网页启动外部飞控。

## 8. 错误与警告示例

| 代码 | 含义 | 表现 |
|---|---|---|
| `DASHBOARD_FETCH_ERROR` | 前端无法访问 API | 全局黄色通知；不显示伪数据 |
| `DASHBOARD_METRICS_ERROR` | 当前 run 指标请求失败 | 保留说明，并提示检查目录与权限 |
| `RUN_NOT_FOUND` | run 不存在或命名不合法 | HTTP 404 |
| `LOG_JSON_INVALID` | 某行 JSON 损坏 | warnings；其余完整事件可显示 |
| `LOG_PARTIAL_WRITE` | 最新一行未写完 | 跳过等待下一轮 |
| `LOG_NONFINITE` | NaN、Inf 等非法值 | 该行跳过并记录警告 |
| `LOG_LINE_TOO_LONG` | 单行超过 64 KiB | 跳过并告警 |
| `SCENE_MANIFEST_INVALID` | 场景 manifest 解析失败 | 场景表标记异常 |
| `READ_ONLY_DASHBOARD` | 外部发送 POST | HTTP 405 |
| `AUDIT_EXPORT_ERROR` | 导出 JSON 失败 | UI 提示，保留原始资源 |

## 9. 测试范围与证据等级

本轮在容器中执行：

```bash
python3 -m unittest discover -s code/tests -p 'test_*.py' -v
python3 -m unittest discover -s sim/tests -p 'test_*.py' -v
./run.sh doctor
./run.sh sim doctor
```

结果：代码基础设施 13 个 unittest 通过，仿真 preflight 2 个 unittest 通过；doctor 报告没有 `nvcc` 和 `gz`，`sim doctor` 按设计报告 `GZ_NOT_INSTALLED`。这些结果不意味着已在 Ubuntu 5090 用户设备运行。

浏览器静态 UI 测试：Chromium headless 使用 `set_content()` 载入，注入后端 `status()` 与 `scene_list()` 的真实本地数据，五页中的总览、训练、仿真切换无 JS pageerror，并导出三张真实空状态截图。由于沙盒浏览器阻止了对 127.0.0.1 的直接导航，**未完成真实浏览器到本地 HTTP API 的端到端浏览器测量**；API 本身通过 Python HTTP 客户端测试和函数测试。

## 10. 下一阶段与明确门禁

1. GZ0：固定 Gazebo 版本，校验 X3 电机、pose、动力学和日志。
2. GZ1：将 RGB 摄像机安装在动态机体的正确 link、统一 FOV，验证 image time 与 pose。
3. GZ2：随机障碍物转成带真实 collider 的实体，不允许 SetPose 代替飞行。
4. CF0：先抽纯 PID/math kernel，数值对拍后软件在环；不动 ESP32 原固件。
5. G3：C++ 静态训练器通过 golden、gradcheck、过拟合与 checkpoint 测试。
6. G4：闭环 VLN 与语言干预测试；再引入 ECPT 方法。
7. M0-Edge：SG2002 最小图转模型、实测 P50/P95/P99；在有真实飞控控制接口和安全验证后才开始 HIL。
8. CF-FW：只有后续单独批准且无桨台架准备好，才创建 ESP32 固件变更分支。

以上步骤相互有前置条件。不能将 UI 状态卡“已就绪”作为任何物理 gate 的 PASS。
