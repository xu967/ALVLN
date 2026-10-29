# NavLab Studio UI v3 — 静态交互原型

这是一套以当前 `web_code_review_20261027.zip` 中 `train.py`、`uavnav/lani/fit.py`、`uavnav/avln/fit.py`、`uavnav/avln/inference.py` 为依据设计的 **前端原型**。不修改训练算法，也不包含训练服务端。

## 运行

直接双击 `index.html`；建议在目录下运行 `python -m http.server 8080`，浏览器打开 `http://localhost:8080`。

页面包括训练工作台、推理与仿真（默认一键自动仿真模式 + 单步调试）、闭环评测、实验记录、AI 审核归档。支持响应式布局、训练配置草稿保存、导入 LANI/Aerial 的 results.csv 更新图表、导入 JSONL/LOG 查看事件、本地 JSON 审核数据导出、预留 API 连接、SSE 订阅及推理会话管理。

## 极其重要的真实/演示边界

- 初始 UI 中所有数值、曲线和事件都清楚标注 **DEMO**，不能当作模型结果。
- 离线模式不会启动训练、推理、评测；界面动作会明确提示服务端未连接。
- 本地导入的文件仅在当前页面会话内保留；浏览器的草稿配置保存在 localStorage。**没有后台持久存储，不构成可靠日志归档。**
- 连接真实服务需要服务端按 `API_LOGGING_CONTRACT.md` 实现 API，且满足鉴权、CORS、任务约束检查。具体 JSON 契约是建议而非已经实现的接口。
- `GET /api/v1/runs/{id}/audit-bundle` 应下载由后端组织的正式证据 ZIP。离线只导出本地示例/导入数据 JSON，不输出虚假合规 ZIP。
- 不在前端创建假的推理动作概率；RGB-D 模型缺少真实深度将被阻止。

## 导入训练日志

进入“AI 审核归档” → 导入 `results.csv`、`runtime.json`、`training_diagnostics.jsonl` 等现有文件。单文件限 8 MB，过大日志请走服务端归档。导入后会识别 Aerial（`train/loss`, `val/loss`）与 LANI（`train_ce`, `dev_ce`）两个 CSV schema。

## 内容边界

项目快照并未提供基于 FastAPI 的统一任务 API。本原型 API Client 是针对后续新增的管理层，**不是直接对应已有的 Python HTTP 路由**。

## v3：自动仿真接口（待后端实现）

在「推理与仿真」中支持先选任务/权重/拆分/场景，再选择自动启动仿真；点击启动后调用新拟议接口 `POST /api/v1/simulation-jobs`。离线不会启动实际进程。LANI 可复用已存在 Unity 构建启动逻辑；Aerial 目前只支持连接已启动 AirSim，需要新建外部场景 supervisor。详细见 `SIMULATION_ORCHESTRATION.md`。
