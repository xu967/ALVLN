# 3DGS 长航线候选审计

- 时间：2026-11-14
- 范围：`uav_rgb/tools/audit_route_candidates.py`、路线候选审计测试和六个 SAGE-3D 场景的只读审计
- 结论：完成

## 做了什么

- 新增纯标准库审计器，读取 `scene.json` 与 `routes_check.json`，检查候选记录、正式路线、重复名称和过滤结果的一致性。
- 输出稳定的 `pair_key`、`route_index`、`max_steps`、路径长度、净空、转向、指令数和过滤原因，按固定排序列出长航线候选，结果写到 `/root/rivermind-data/uav_rgb/tmp/route-candidate-audit-20261114/`。
- 新增 3 个单测，覆盖长路线筛选、缺记录完整性错误和非法顶层格式。
- 将实际六场景审计结果单独记在 `docs/experiments/2026-11-14-route-candidate-audit.md`。

## 没做什么

- 没有改动任何原始 `scene.json`、`routes_check.json`、PLY、路线轨迹或实验输出；本任务只做只读审计。
- 没有修改 `run_vln.py` 以直接消费清单；当前结果供后续配对脚本按 `pair_key` 固定请求集，避免扩大本次改动。
- 没有运行教师、reactive 或 SITL 长航线实验；本任务目标是先固定可复现候选集，策略结果需另行记录。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `python -m unittest tools.test_audit_route_candidates -v` | 3 项通过 |
| 六场景 `audit_route_candidates.py` | 249/249 正式路线与通过候选一致；79 条路线长度不少于 10 m；完整性错误 0 |
| `git diff --check` | 通过 |

## 风险与遗留

- 长路线阈值 10 m 是本次审计参数，不等于公开基准阈值；跨场景对比仍需按 split、`agl_model_unverified` 和动作预算分层。
- 清单目前不会自动约束 `run_vln.py` 的请求集，后续运行必须保存实际 `pair_key` 列表并检查两种策略完全相同。
