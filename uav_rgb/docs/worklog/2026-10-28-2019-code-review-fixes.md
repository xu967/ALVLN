# 代码审核修复 + 个人规范补充

- 时间：2026-10-28 20:19
- 范围：uav_rgb 的 `src/`、`ui/`；个人 skill（`personal-skill`）
- 结论：完成

## 做了什么

- 更新 `personal-skill`（`/root/.claude/skills/personal-skill/SKILL.md`）
  - §3 增加「及时删除测试脚本」：一次性验证脚本/调试残留用完即删，只有正式单测/自检保留
  - 新增 §6「实验结果单独管理」：实验/训练/基准/smoke 结果单独落一份 md，与 worklog 分开
- 审核修复（6 项全部落地）
  - `ui/static/app.js`：监控页加 5s 轮询重拉 `/metrics` + `/events`（原来只加载一次）；删除硬编码 `DEMO_ROWS`/`DEMO_EVENTS` 两份假数据，离线改为空态
  - `ui/server.py`：`DEFAULT_RUN_DIR` 改为从 `__file__` 派生，不再硬编码绝对路径
  - `src/ops_cpu.cpp`：`softmax_ce_loss`/`softmax_ce_grad` 增加 `ndim()!=2` 校验，异常统一为 `invalid_argument`
  - `src/tests/test_tensor.cpp`：修正默认构造的过时注释（1 元素全零，非 0 元素）
  - `ui/static/index.html`：banner/footer 文案更新（不再提「内置 DEMO 数据」）
  - 新增 `ui/test_server.py`：起真实 HTTP 服务的自检，覆盖 health/metrics/events/路径穿越
- `CLAUDE.md`：补充 src 核心库构建 + ctest、UI 启动/自检命令

## 没做什么

- 没做真实浏览器端 e2e（打开页面确认 5s 轮询视觉表现）
  - 原因：无头环境；已用 `node --check` 验 JS 语法 + HTTP 层自检兜底
- 没删 `ui/demo/results.csv` / `metrics.jsonl`
  - 原因：仍是服务端默认 run_dir 的样例，保留是为了连上时能看到曲线；已标注「示例数据非真实」
- 轮询里没重查 `/health`
  - 原因：最小改动；后果是服务端中途宕机时 banner 不会自动变「未连接」，需手动刷新
- 没动 CUDA 内核（E4 未开始）、没动 E1–E3 仿真层
  - 原因：本轮只做审核修复，不新增功能
- 没有真正删除任何测试脚本
  - 原因：仓库内无一次性残留脚本；新建的 `ui/test_server.py` 是正式自检，按新规范保留
- 本轮无新实验，未新建实验结果 md
  - 原因：只有 smoke 测试和 demo 数据；结果管理规范已写入 skill §6，后续跑实验再落地

## 待确认

- 轮询间隔 5s 是否合适；是否需要在轮询里一并重查 `/health` 更新连接状态
- `ui/demo/` 示例数据是否要保留（当前保留并标注非真实）

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `cmake --build src/build` | 通过（6 个 target 重编成功） |
| `ctest --test-dir src/build` | 3/3 通过（test_action / test_tensor / test_gradcheck） |
| `python3 ui/test_server.py` | EXIT=0（health/metrics=8行/events=4条/路径穿越404 全过） |
| `node --check ui/static/app.js` | syntax OK |

## 风险与遗留

- 前端自动刷新仅验证到语法 + API 层，未在真机浏览器点开确认；5s 轮询在后端未启动时会持续发空请求，无副作用
- 轮询不重查 health：服务端宕机时 UI 仍显示「已连接」直到手动刷新
- `send_error(404)` 会触发 stdlib 的双日志（log_request + log_error），属 Python 标准库行为，不影响功能
