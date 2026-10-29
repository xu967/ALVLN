# NavLab Studio：UI 接口与 AI 审核归档契约（设计规范）

> **状态：待后端实现。** 这是 UI 与后端之间的拟议协议，不是 `web_code_review_20261027.zip` 中已存在的 HTTP API。检查依据为 2026-10-27 代码快照中的 `train.py`、`uavnav/lani/fit.py`、`uavnav/avln/fit.py`、`uavnav/avln/inference.py`、`evaluate.py`。UI 不得宣称未连接时已启动任何真实任务。

## 一、源码已支持的产物

**LANI** (`train.py` → `fit_lani`)：`results.csv` 字段为 `epoch, train_ce, dev_ce, dev_accuracy, dev_stop_recall, dev_macro_recall, updates, lr, elapsed_seconds`；有 `hparams.json`、`task_config.json`、`dev_action_confusion.json`、`results.png`、`weights/last.pt`、`weights/best.pt`、训练结束写入的 `metrics.json`。按完整 dev 无权 Teacher CE 选择 `best.pt`；`metrics.json` 导航指标（NE/SR/OSR/nDTW/sDTW/SPL）在独立闭环前为 null。四动作按顺序 `Forward / TurnRight / TurnLeft / Stop`。

**AerialVLN-S** (`train.py` → `fit_aerial`)：`results.csv` 字段中有 `epoch`、`train/loss`、`val/loss`、`val/accuracy`、`val/stop_recall` 等；此外会写 `training_diagnostics.jsonl`、`runtime.json`、`source_manifest.json`、`task_config.json`、`split_manifest.json`、`metrics.json`、`action_confusion.json`、`weights/last.pt`、`weights/best.pt`、有条件保存的 `candidate_inventory.json` 等。候选留存和 resume 机制有特定认证约束，不能通过 UI 随意启用。主线八动作，动作 0 是 STOP；2/3 是偏航、6/7 是侧移，前端不可只显示“左右转”并混淆含义。验证损失/动作准确率不等于 SR。

**推理**：Aerial 路线可经 `uavnav.avln.inference.load_predictor` / `predict_single` 预测（包括循环状态）；Aerial 循环推理器要求 episode 语言指令一致，换指令前需 reset；RGB-D 模型必须有真实深度，不能用零填充。LANI 需单独适配其模型/图像处理/动作协议，不能直接假设 Aerial predictor 接口支持 LANI checkpoint。

**闭环评测**：`evaluate.py` 提供 replay/AirSim 等入口，replay 只能作为离线分类诊断，**不得宣称其为正式导航成功率**；完整评测需独立协议、环境和相应权重 SHA。

## 二、统一任务管理 API（建议）

| 方法 | URL | 用途 | 关键行为 |
|---|---|---|---|
| GET | `/api/v1/capabilities` | 列出任务、架构、能力、可选参数 | 数据由后端提供，前端禁用不支持的组合 |
| GET | `/api/v1/datasets` | 查询已注册训练/验证缓存 | 返回缓存 ID、版本、任务、校验状态 |
| POST | `/api/v1/runs` | 新建异步训练 | 验证配置，创建独占 run_dir，返回 202 + run_id |
| GET | `/api/v1/runs` | 实验历史 | 分页 / 支持状态筛选 |
| GET | `/api/v1/runs/{id}` | 训练状态和配置 | 包括状态 queued/running/finished/failed/stopped |
| POST | `/api/v1/runs/{id}/stop` | 安全终止 | 先发终止信号，再根据策略升级，不声称支持断点续训 |
| GET | `/api/v1/runs/{id}/metrics` | 训练指标 | 后端对 LANI/Aerial 进行归一化，只返回已真实写入数据 |
| GET | `/api/v1/runs/{id}/events` | SSE 实时事件 | `id:` 使用单调事件序号，支持 `Last-Event-ID` 断线回放 |
| GET | `/api/v1/runs/{id}/artifacts` | 文件清单 | 文件名、存在性、sha256、字节数、生成时刻 |
| GET | `/api/v1/runs/{id}/audit-bundle` | AI 审核归档 | streaming ZIP，原始证据、审计 manifest、README |
| GET | `/api/v1/checkpoints` | 根据 task 查询可用权重 | 需绑定 `task`, `architecture`, `checkpoint_format`, `sha256` |
| POST | `/api/v1/inference/sessions` | 新建推理 Episode | 明确选择权重，创建受控循环状态 |
| POST | `/api/v1/inference/sessions/{id}/steps` | 单步推理 | multipart RGB + instruction（按架构增加 depth），回动作及分数 |
| POST | `/api/v1/inference/sessions/{id}/reset` | 重置 Episode | 新指令前先 reset，不要静默复用 GRU 状态 |
| POST | `/api/v1/evaluations` | 创建异步评测 | 必须绑定真实环境和冻结评分协议 |

`POST /api/v1/runs` 建议请求：

```json
{
  "task": "lani_v1",
  "architecture": "lani_dino_recurrent_v1",
  "name": "lani_dino_seed0_runA",
  "train_cache_id": "cache_train_verified",
  "validation_cache_id": "cache_dev_verified",
  "epochs": 20,
  "sequence_chunk_size": 16,
  "lr": 0.0001,
  "seed": 0
}
```

后端需要将前端字段映射为已存在的 `train.train(dataset=..., overrides=...)` 能接受的特定配置。不能把任意用户输入拼接成 shell 命令，不允许绕过 `train.py` 与各模型的校验。Aerial 还需校验 `validation_cache_manifest`、视觉模型、冻结权重和多模态输入等组合限制。`lr=1e-4, seed=0` 等代码冻结值在 UI 中应只读。

SSE 例子（**新协议示例，不是已有训练事件**）：

```
id: 302
retry: 5000
event: metric
data: {"kind":"metric","seq":302,"run_id":"run_abc","timestamp":"2026-10-28T08:00:00Z","epoch":4,"metrics":{"epoch":4,"train_ce":1.3,"dev_ce":1.4,"dev_accuracy":0.58}}

```

注意：上述数字为 schema 示例而非真实训练结果。**事件必须先落盘，再推送浏览器**。断网时浏览器可通过 `Last-Event-ID` 重放；服务器不得只把 SSE 当临时日志缓存。

### 指标规范化

| 前端标准字段 | LANI 列 | AerialVLN-S 列 | 语义 |
|---|---|---|---|
| `epoch` | `epoch` | `epoch` | Epoch 从 1 起 |
| `train_ce` | `train_ce` | `train/loss` | 训练损失，明确来源和计数基础 |
| `dev_ce` | `dev_ce` | `val/loss` | 验证 CE（选模时需区分权重 CE 协议） |
| `dev_accuracy` | `dev_accuracy` | `val/accuracy` | 教师状态动作分类准确率 |
| `dev_stop_recall` | `dev_stop_recall` | `val/stop_recall` | 教师 STOP 召回率 |

对 Aerial，`training_diagnostics.jsonl` 已区分 `predictions_during_parameter_updates` 与 `fixed_epoch_end_weights_eval`。**训练时累积预测与固定末权重在 eval 模式下重算的精度不可混用。** 单步推理和独立闭环导航指标另设接口和协议，不从训练 CSV 构造 SR。

## 三、日志保存：UI 只展示，服务端必须持久归档

每次启动生成不可复用的 `run_id` 与独占目录。推荐结构（现有文件直接沿用，不覆盖，不改原语义；`ui_service/` 为新增旁路资料）：

```text
runs/
└── <run_id>/
    ├── results.csv                     # 原始训练指标，保持现有格式
    ├── metrics.json                    # 原始指标摘要（导航未评测则 null）
    ├── hparams.json / hparams.yaml     # 依具体训练分支
    ├── task_config.json
    ├── runtime.json                    # 若源码分支产生
    ├── source_manifest.json            # 若源码分支产生
    ├── training_diagnostics.jsonl      # 若源码分支产生
    ├── training_failure*.json          # 异常时才有
    ├── weights/last.pt, best.pt, ...   # 原始权重（审核包通常不含大文件）
    └── ui_service/                     # 任务管理层新增文件
        ├── job.json                    # CLI 参数、白名单配置、启动方式
        ├── stdout.log                  # 原样持续写入（stdout 重定向）
        ├── stderr.log                  # 原样持续写入（stderr 重定向）
        ├── events.jsonl                # 结构化、单调 seq 事件
        ├── lifecycle.json              # 进程状态、启动/终止/退出码
        ├── audit_manifest.json         # 文件清单、路径、大小、sha256
        └── ai_review_readme.md         # 用自然语言解释训练和证据
```

必须明确 **训练进程 stdout/stderr 与结构化事件是两层独立证据**。stderr 要包含完整 Traceback，事件文件可提供摘要但不能替代原始错误输出。UI 异常告警展示可追踪的 `source_file/line/epoch/run_id`；后端失联时不得丢已有原始日志。

### 结构化事件建议 schema

```json
{
  "schema_version": "navlab_event_v1",
  "run_id": "run_abc",
  "seq": 302,
  "timestamp": "2026-10-28T08:00:00Z",
  "level": "ERROR",
  "source": "train_process",
  "phase": "validation",
  "epoch": 4,
  "step": null,
  "message": "Nonfinite validation loss",
  "error_type": "FloatingPointError",
  "traceback_ref": "ui_service/stderr.log#L182",
  "raw_artifact_ref": "training_failure.json"
}
```

生产环境：写入时行追加并 flush；崩溃时完整目录仍可下载。进程结束后计算 SHA256；正在写的文件标记 `in_progress`，不得错当最终校验。保留 UTC 时戳 + 单调序号，避免仅靠前端本地时间排序。记录环境和数据来源而不泄露密钥、访问令牌、PII；用户可以配置日志脱敏和保留期。对长日志建议按大小轮转，但审核包按顺序合并且保留切片 manifest。

## 四、AI 审核包 ZIP 内容

推荐由后端生成单个 `run_abc_ai_audit.zip`，不用让用户手动挨个找文件：

```text
run_abc_ai_audit/
  START_HERE.md              # 实验目标、任务、模型、训练配置、审核重点
  manifest.json              # 所有归档文件大小、SHA256、原路径及产生时间
  run_summary.json           # run_id、状态、退出码、最优轮次/指标来源、已知限制
  config/                    # 训练与数据协议原文件副本
  metrics/                   # results.csv、metrics.json、训练诊断文件
  logs/                      # stdout.log、stderr.log、events.jsonl（必要时分卷）
  failures/                  # training_failure*.json（若存在）
  checkpoints/               # 权重元数据/sha256/来源，默认不打包大体积 .pt
  evaluation/                # 若有真实独立闭环结果，连同评测协议和轨迹文件索引
```

需要 **保留 null**，绝对不要把尚未完成的 NE/SR/OSR/nDTW/sDTW 伪造为 0 或优秀结果。缺文件在 `manifest` 写明 `not_produced` / `failed_to_collect` 等状态；`finished` 与 `failed` 都应可下载审核包。可选提供“精简 AI 包”（不含权重、图片或大日志）与“完整证据包”（保留全部原始记录，体积可能很大）。

`START_HERE.md` 应帮助 AI 分辨：哪些指标由真实模型输出、哪些是 replay 诊断、哪些是正式闭环评测、哪个 checkpoint 用于哪些结果；并记录源码提交 SHA、数据 manifest SHA、评测协议版本、缺失项，以及运行状态。

## 五、可靠性与安全验收

1. 页面关闭、前端刷新、SSE 断开不影响服务端训练，也不停止日志写入。
2. 任务在任意时刻崩溃仍保留 stdout/stderr/部分 CSV/traceback/退出码；可以生成含 `incomplete` 标记的审核包。
3. 重新选择实验后补发过去事件，UI 保持 epoch + metric 去重，不重复追加。
4. 训练运行目录不允许与旧 run 重名覆盖；对源码已经有 resume 锁和认证的机制保持尊重。
5. 所有对 GPU 训练进程的操作需鉴权、任务隔离、速率限制，不允许任意路径读取或 shell 注入。
6. 多种架构的模型输入兼容性在后端确认；RGB-D 缺真实深度、循环模型未 reset 指令等情况应拒绝推理。
7. 对关闭页面后的任务和事故能完整导出审核包；ZIP 中 manifest 校验通过，AI 能独立审核日志与指标。
8. 禁止基于 teacher accuracy 计算导航 SR 或将 replay 分类直接作为闭环成绩。

## 六、仿真自动启动及审计

UI v3 中「推理与仿真」增加一键自动仿真模式，接口 `POST /api/v1/simulation-jobs`。详见 [`SIMULATION_ORCHESTRATION.md`](SIMULATION_ORCHESTRATION.md)。仿真任务必须和训练 run 关联并记录独立的 simulator/评测器 stdout、stderr、preflight、原始 rollout 与 SHA256 manifest。
