# 14｜NavLab Studio UI v3 + 本机任务 API 操作合同

## 0. 对两个新上传文件的继承

原始 `NavLab_Studio_UI_v3(1).zip` 包含 `index.html`, `styles.css`, `app.js`, 七张 UI 预览及 `API_LOGGING_CONTRACT.md`/`SIMULATION_ORCHESTRATION.md`。**原始 UI 是静态原型，并无与之完全匹配的服务器程序**。本最终包保留前端页面与源代码，通过 `server/app.py` 新增少量已经真实测试的 API。文档 `06_SIMULATION_ORCHESTRATION.md` 继续作为更完整的 future supervisor 需求说明。

本服务端**已真实实现**：

- `GET /api/v1/health`, `GET /api/v1/capabilities`, `/datasets`, `/checkpoints`, `/runs`, `/runs/{id}`, `/runs/{id}/metrics`, `/runs/{id}/events`, `/runs/{id}/artifacts`, `/runs/{id}/audit-bundle`；
- `GET /api/v1/simulation/capabilities?task=...`, `/simulation-jobs/{id}`, `/simulation-jobs/{id}/events`, `/simulation-jobs/{id}/artifacts`, `/simulation-jobs/{id}/audit-bundle`；
- `POST /api/v1/runs`, `POST /api/v1/simulation-jobs`，仅在管理员登记完整、预检 verified 的可执行 profile/数据/权重/场景后实际启动进程；现有默认注册为空，所以**安全阻断**。服务端在 profile 查询前校验请求：训练必须提供合同中的完整字段，`epochs` 为 1–5000 的整数，`sequence_chunk_size` 为正整数，`lr` 为有限正数，`seed` 为非负整数；仿真资源 ID 必须合法，`auto_launch/record_observations/audit_artifacts` 只接受布尔值，`mode` 仅支持 `diagnostic/official_split`。无效请求返回 `400 INVALID_REQUEST`。
- `POST .../{id}/stop` 只对本服务亲自启动且仍存活的任务进程组发送 SIGTERM；绝不按 PID 匹配外部共享仿真服务；所有操作记录审计事件。
- 静态 `frontend/` 自动同源连接；离线还是 DEMO，连接真实 API 后未有真实 run 就空值。

**没有实现**：Aerial UE supervisor、真正 Unity 自动启动的一体化工作流、已受控的动作实时视频流、单步图像推理会话、基于独立 evaluator 的真实导航 SR 计算、外网权限管理、浏览器多人协同。未实现接口返回 `FEATURE_NOT_IMPLEMENTED`；浏览器不能创建虚构运行记录。

## 1. 创建真实训练任务前要做的事

服务器管理员编辑 `config/registry.json`（参照 `config/registry.example.json`），其中至少有：可信 C++ 训练程序路径、`task/architecture`、严格 argv 模板、运行必备文件、独立 verified 的 train/validation dataset IDs。训练程序**必须已经用 CLI 手工验证过**，并且能接受 `--output` 或等价参数输出真实 `results.csv`。

允许的 argv 插值变量仅 `{run_dir}`, `{config_file}`, `{sim_dir}`。前端既不能上传任意脚本，也不能注入命令行字符串。外部程序的工作目录在项目根下，日志自动写入独占的 `artifacts/runs/run_<uuid>/ui_service`。stdout/stderr 为原样的证据文件；任务结束状态由 `returncode` 决定。

注册项 `ready=true` **不是自动获得安全认证**；只表示管理员在本机完成了依赖和命令合同确认。对正式实验还需人工审查源码/数据协议与日志。当前服务的 profiles 是单机开发者工具，暂不支持公网协作。

## 2. AerialVLN 的一键仿真不能跳过 Scene Supervisor

`SIMULATION_ORCHESTRATION.md` 的约束在此原样继承：旧 `evaluate.py --backend airsim` 需要一个**已经真实启动且与正确 object/scene provenance 对齐的 AirSim 环境**。用户勾选 `auto_launch` 并不证明 UE 场景可用。管理员在准备 AirSim profile 前必须实现独立 SceneSupervisor，记录执行文件哈希、scene id、settings、RPC、相机、spawn provenance、reset，再由 trusted wrapper 提交 `evaluate.py`。

已存在 LANI Unity `runtime.real_server()` 的进程控制思路可参考并重用，但完整资产/参数/平台的实际契约必须再次检查。LANI 和 Aerial 不共用成功率定义，`replay_only` 不产生闭环成绩。

## 3. 服务端允许/禁止的状态

- `queued/preflight/launching/waiting_rpc/scene_verified/running/finalizing/finished` 仅在执行管理层真生成相应事件后显示；本 V8 的 generic job runner 目前只会记录 `preflight`, `running`, `finished/failed/stopped`，没有假造 SceneSupervisor 的中间状态。主动停止会在锁内发送 SIGTERM 后持久化 `stopping/stop_requested`，worker 无论进程最终返回 0 还是信号负码都记录 `stopped`；若进程已在发送信号前退出，返回 `409 JOB_NOT_ACTIVE` 且不改写为 stopped。当前只发送一次 SIGTERM，不自动升级 SIGKILL。
- 如果注册缺失，`409 PRECONDITION_FAILED` 并带机器可读原因 `PROFILE_NOT_REGISTERED/EXECUTABLE_NOT_READY/CLOSED_LOOP_NOT_VERIFIED`。
- 已创建进程如果服务器异常重启，目录会标记 `interrupted`，不会盲目 kill 它；**运维人员需先确认孤儿进程**，这是尚未实现可靠收养/recovery 的明确限制。
- `results.csv` 只按 LANI/Aerial 已知字段转换 train/dev CE 和 action accuracy；未见真实评测的 `navigation_sr=null`；绝不根据单步准确率推算官方 SR。

## 4. 审计 ZIP

通过 `GET /api/v1/runs/<id>/audit-bundle` 或 `/api/v1/simulation-jobs/<id>/audit-bundle` 生成只读 ZIP。包含小型真实文件、SHA256 manifest 和 `AI_REVIEW_START_HERE.md`；每单文件大小有上限，权重和大型视频默认不进入审核包（需外部 URI/hash 另行存档）。每个证据文件只读取一次，ZIP 内容、manifest 哈希和字节数均来自同一份快照。对当前服务仍托管的活动任务，归档会先向任务独占进程组发送 SIGSTOP，通过 Linux `/proc` 确认进程已停止，并在全局锁内阻止 worker 写入；全部文件读取后在 finally 中发送 SIGCONT，压缩在恢复后进行，因此托管写入者范围内的所有文件来自同一个静止阶段。暂停失败或读取异常也会恢复进程并返回错误，不静默降级为非原子归档。该机制不能约束外部程序、已脱离进程组的写入者或服务重启后的孤儿进程；暂停期间墙钟时间仍前进，实时仿真可能触发 RPC/控制超时，不应在实时关键阶段下载。**归档 ZIP 是对已有文件的集合，不代表所有科学门禁已通过。**

## 5. UI / 数据保护

- UI 主界面保持原配色和页面设计；可在项目 CSS 内局部更新，但不引入未经批准的云服务或前端自动假预测。
- API 只允许 loopback Host/Origin，默认不支持公网。`Host` 与 `Origin` 校验可降低本地开发环境 DNS rebinding/CSRF 风险，但**不是**企业级认证。
- 默认无 job registry；不向 UI 提供物理飞控启停按钮。
- 日志 JSON 必须严格处理嵌套 NaN、深层 JSON 和坏行；单行损坏不能让整个 run 数据消失。接口输出严格 `allow_nan=False`。
- `run_id` 必须统一遵循 ASCII `[A-Za-z0-9_-]{1,70}`；所有运行目录和文件均由服务端生成。

## 6. 一次最小运营验收

```bash
./run.sh ui
curl -s http://127.0.0.1:8765/api/v1/health
curl -s http://127.0.0.1:8765/api/v1/runs
curl -s 'http://127.0.0.1:8765/api/v1/simulation/capabilities?task=aerialvln_s_v1'
```

期望 health 为 management_only；没有真实 run 时 `items=[]`；sim capability 默认 ready=false。**绝对不能**因为页面连接成功就宣称已训练或已启动仿真器。
