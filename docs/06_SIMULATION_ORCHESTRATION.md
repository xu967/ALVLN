# NavLab Studio v3：一键仿真推理接口设计与源码核对

> 状态：**UI 原型与建议契约，后端尚未实现统一 HTTP 服务**。核对依据：`web_code_review_20261027.zip` → `source_snapshot/evaluate.py`、`uavnav/avln/airsim_env.py`、`uavnav/avln/rollout.py`、`uavnav/lani/runtime.py`、`scripts/verify_lani.py`。不要将原型占位视作真实观测，也不要把 `replay` 当成闭环导航。

## 1. 为什么推理页应该支持「一键仿真」

前端同时支持两条路径：

- **默认：自动仿真推理**。选择任务、认证过的 checkpoint、评测拆分和场景 → 预检 → 后端启动或连接真实仿真进程 → 运行 Episode → 保存观测、动作、状态及终止原因 → 持久化日志与评测证据 → 出具报告。
- **次级：单步调试**。用于核查 `RGB / (depth) + instruction + recurrent history` 到模型动作输出的连接，不代表导航成绩。

「自动」代表**一个显式启动操作触发完整后端工作流**，不是前端加载时自行启动仿真软件，也不是把每次单帧推理强制升级为昂贵的场景评测。

## 2. 已有代码能力与缺口（重点）

### LANI：Unity / NavDrone

`uavnav/lani/runtime.py:real_server()` **已有真实 Unity 启动代码**：构造 `simulators/nav_drone/NavDroneLinuxBuild.x86_64` 命令并附带 `--port` 和 `-logFile`；进程 stdout 写入 `runtime_attempt_{n}/unity_stdout.log`，环境元数据进入 `environment.json`，退出时清理 Unity 与 NavDrone Server。`runtime.evaluate()` 将 Episode 事件写为 `episodes/{task_id}/journal.jsonl`，并写 `status.json`、`evaluation_contract.json` 等。LANI 应由服务端调用已有 `scripts/verify_lani.py --mode evaluate` 路径（合法参数需逐项白名单校验），**不要重复造 Unity 控制器**。

- 前置：官方 asset root / upstream root、相应平台可执行文件及渲染依赖、reference manifest、正确格式的 LANI checkpoint、split。
- `UnityLinuxBuild.x86_64` 仅是现有 Linux 路径；若用户在 Windows 部署，需要 Linux 容器/主机或另行适配合法的 Windows 构建，不能假装当前源码已支持 Windows 本地启动。

### AerialVLN-S：TravelUAV / AirSim

`evaluate.py` 的 `--backend airsim` 会加载已启动环境中的 `AirSimEnv` 并执行闭环；但 `evaluate.py` 的 `--scene-ready` 只是一种**调用方确认状态**，且 `uavnav/avln/airsim_env.py` 明确指出它**不启动场景、不替换官方对象生成逻辑**。代码已提供 `scene-id`、`scene-endpoints`、IP/port、scene provenance、评测协议、录制观测等约束，要求匹配真实场景、对象布局、相机和冻结评测合同。

要实现 Aerial 的一键自动启动，**必须新增独立 Scene Supervisor**，而非修改算法：

1. 查询已注册的官方场景 build、settings、对象生成/预布置流程；
2. 启动官方 AirSim/UE 构建或连接可信的已启动端点；
3. 核对场景 ID、build SHA256、settings SHA256、相机外参/FOV、目标对象及 spawn provenance；
4. 实际做 RPC 心跳、相机观测与 reset 就绪检测；
5. **全部成功后**才通过受控参数让 `evaluate.py` 执行，赋予 `--scene-ready`；
6. 逐 Episode 记录 `rollout.json` / `error.json`、`ori_info.json`、`scene_provenance.json` 等原生文件，不跳过异常审计直接重试；
7. 按配置决定是否清理由该任务启动的仿真进程，不要杀掉其他任务共享的场景服务。

如果注册构建/设备不可用，API 返回 `blocked` 与原因，前端不允许以 replay 替代并打出 SR。

### 评测身份与口径

- 单场景/少数 Episode 是 **diagnostic**，必须标注与正式基准成绩不同。
- 正式 Aerial 评测需要符合现有源码的真实 AirSim、完整 `val_seen / val_unseen`、显式 `eval_adapter_v2`、`max_steps=500` 与冻结清单等约束；是否具备论文可复现性还需数据与环境来源核验。
- LANI 官方 Unity 有自己的冻结评测合同。两个任务不能共享一个不加区分的 `success_rate` 计算实现。

## 3. 新增服务端管理 API（并非现有接口）

### 查询可运行资源

`GET /api/v1/simulation/capabilities?task=lani_v1`

示例响应 **仅表示 schema，非当前机器状态**：

```json
{
  "task": "lani_v1",
  "ready": false,
  "preflight": {"scene": null, "model": null, "data": null, "rpc": null},
  "checkpoints": [],
  "splits": [],
  "scenes": [],
  "auto_launch_supported": false,
  "missing_reasons": ["Backend implementation not installed"]
}
```

资源列表中 `id` 必须是服务端授权注册 ID，不得允许用户从页面传入任意可执行文件路径或可执行命令。

### 创建异步仿真任务

`POST /api/v1/simulation-jobs`

```json
{
  "task": "aerialvln_s_v1",
  "checkpoint_id": "registered_checkpoint_id",
  "split_id": "registered_dev_route_subset",
  "scene_id": "registered_scene_episode",
  "auto_launch": true,
  "mode": "diagnostic",
  "record_observations": true,
  "audit_artifacts": true
}
```

建议立即返回 `202 Accepted` 与：

```json
{"job_id":"sim_...","status":"queued"}
```

强制状态机：`queued → preflight → launching → waiting_rpc → scene_verified → running → finalizing → finished`，任意阶段可进入 `blocked / failed / stopped`，并写完整原因。正式运行与开发诊断采用不同协议，不能因为前端选择了 `mode=diagnostic` 就绕过源代码现有安全校验。

### 获取运行状态与直播

- `GET /api/v1/simulation-jobs/{job_id}` → `{status,episode_index,step,action,termination_reason,observation_stream_status,events,artifacts,...}`。UI v3 以 **2.5s 状态轮询** 为可运行基础，不需要强制先实现视频推流。
- `GET /api/v1/simulation-jobs/{job_id}/events` → SSE。未来可增加按 `Last-Event-ID` 断线重放。
- `GET /api/v1/simulation-jobs/{job_id}/observations` → 提供经权限校验的 JPEG/MJPEG/WS 观测流；没有真实观测，UI 必须显示占位状态。
- `POST /api/v1/simulation-jobs/{job_id}/stop` → 安全终止单次任务，保留未完成证据。
- `GET /api/v1/simulation-jobs/{job_id}/artifacts` → 列表，包含 SHA256/路径/大小/生成阶段；`GET /api/v1/simulation-jobs/{job_id}/audit-bundle` → AI 审核 ZIP。

服务器进程负责全部仿真与日志生命周期。浏览器刷新/关闭不应终止任务，也不能清除源日志。要防止重复提交、共享 GPU/同场景端口竞争、无授权启动与强杀其他人的进程。

## 4. AI 审核证据：训练和仿真两份相互关联的目录

```text
experiments/
  {run_id}/
    results.csv                          # 原有训练输出
    weights/best.pt
    ui_service/stdout.log
    ui_service/stderr.log
    ui_service/events.jsonl
    ui_service/lifecycle.json
    ui_service/audit_manifest.json
  simulations/
    {simulation_job_id}/
      simulation_request.json            # 后端白名单化后的请求
      launch.json                        # 进程/端口/环境来源/时间线
      preflight.json                     # 构建、RPC、资源、协议检查
      simulator_stdout.log               # 外部程序原始 stdout
      simulator_stderr.log               # 外部程序原始 stderr
      supervisor_events.jsonl            # 场景控制与重启等事件
      evaluation_stdout.log              # 评测器原始输出
      evaluation_stderr.log
      evaluation/                       # 源码已有 rollout/ori_info/error/journal 等
      observations/                      # 可选：真实 RGB/深度及时间戳
      checkpoint_ref.json               # 相应 run_id 和 weights SHA256
      audit_manifest.json               # 每个文件大小/hash/状态
      AI_REVIEW_START_HERE.md
```

所有事件带 `run_id`（若来自某训练）和 `simulation_job_id`、UTC 时间戳、单调序号、阶段、错误类型、原始日志索引。异常也必须有完整 stderr、exit code 与最后成功阶段。`audit-bundle` 打包原始输出、评测协议与原始轨迹记录；大权重默认只包含 URI/sha256/元信息，不自动复制数 GB 权重。

**关键禁止事项**：把没有执行过 AirSim 的 Replay 输出标记为闭环 SR；自动重跑未封口或被污染的 Episode 而不做审计；把单条 Episode 的诊断成绩当作官方 Val；修改或重写原有 Source 生成的日志；用假视频/伪深度来演示真实执行。

## 5. 前端验收

1. 默认打开“推理与仿真”时看见自动仿真模式，显式切换才看单步推理。
2. 没连接管理服务时按钮提示不可启动，不改变状态为 Running，不产生虚构数据。
3. 任务 + checkpoint + split + scene 完整才能提交；必须收到服务端 job_id 才显示已受理。
4. 已受理后可以展示状态、Episode、动作、终止原因与带序号事件，不把上传图片作为仿真观测。
5. 可跳转 AI 审核中心找到对应训练与仿真文件；真实 ZIP 完全由后端持久归档生成。
6. 对 Aerial 未提供 supervisor 或 scene provenance 的场景应显示阻塞原因，不得绕过 `--scene-ready`。
7. 后端按任务协议输出结果；`diagnostic`、`official_split`、`replay_only` 具有不同语义和视觉状态。
