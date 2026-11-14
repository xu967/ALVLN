# VLN 计划执行：profile、语言 grounder 与验证

- 时间：2026-11-14 09:00
- 范围：`proto/` profile 配置与 Torch 基线、`sim/3dgs/` 语言 grounder、OpenFly/路线审计工具、交接文档
- 结论：部分完成

## 做了什么

- 在 `proto/config.py` 注册 `nano_proto_v6` 与 `base_torch_v1`，支持 YAML/JSON、严格字段校验、配置 hash 和 resolved config；修复默认配置被运行时修改的问题。
- 给 `nano_proto_v6` 增加配置尺寸检查和 CLI 显式覆盖；拒绝当前模型尚未实现的语言、历史帧和 prev-action 设置。
- 新增 `proto/base_model.py` 与 `proto/train_base.py`，提供 Torch-only 较大基线的最小真实训练入口、输出锁、验证集选 best、test 指标和 `model.pt`；不触碰 C++ checkpoint。
- 新增 `config/vln/*.yaml`，让 profile 选择可以通过人可读配置完成。
- 新增 `sim/3dgs/grounder.py`：指令到外部候选地标的可解释 lexical G1 起点，增加目标短语加权、实例编号忽略和歧义拒识；新增 6 个单测。
- 新增 `tools/audit_vln_grounder.py`，对六个 SAGE 场景做端点词汇诊断，并在报告中明确标注标注泄漏。
- 保留并纳入 OpenFly 标注审计、SAGE 路线候选审计及对应文档；没有修改原始数据和场景。

## 没做什么

- 没有把 grounder 接入 `run_vln`、学生模型或飞控：当前 `scene.json` 没有独立物体 catalog/坐标，接入会把 waypoint 真值伪装成语言理解。
  - 原因：先保留可审计边界，等独立地图/感知候选表后再做正式 G1 闭环。
- 没有把语言输入加入 `nano_proto_v6` C++ 模型：这会改变现有 checkpoint/动作契约，计划中的端到端条件学生仍是下一阶段。
  - 原因：本次先完成 profile 与语言落地接口，不在没有训练标签的情况下改部署主线。
- 没有下载 OpenFly 全量数据、权重或 3DGS 场景，也没有运行官方远程评测。
  - 原因：官方数据约 2.97 TB，现有动作/动力学和场景资产不匹配；当前完成本地 schema 审计。
- 没有删除数据盘旧实验、日志或废弃文档。
  - 原因：本轮没有逐项确认可恢复性，避免误删用户实验；仅新增审计报告和交接说明。
- 没有运行长航线控制仿真或修改 3DGS 渲染器/教师控制律。
  - 原因：路线审计先固定了 249 条正式路线和 79 条 10 m 以上候选，控制实验应使用这份固定清单另开实验记录。

## 待确认

- 独立物体 catalog 的来源（InteriorGS labels 权限、SAGE 标注扩展或自采地图）以及是否允许地面站 VLM 作为 G2。
- 是否把 `base_torch_v1` 作为地面站比较基线，还是只保留用于结构 smoke；当前没有宣称它达到任何效果门槛。
- OpenFly 是否只做本地 evaluator 迁移对照；当前没有发现可直接上传模型的官方远程提交接口。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `proto.test_config proto.test_base_model proto.test_train` | 通过，29 项，跳过 4 项 |
| `sim.3dgs.test_vln test_guard test_renderer test_grounder` | 通过，77 项，跳过 2 项 |
| `tools.test_audit_vln_grounder test_audit_route_candidates test_audit_openfly_annotations` | 通过，6 项 |
| `ctest --test-dir build/check --output-on-failure` | 通过，34/34 |
| 真实 UAVR `proto.train_base --max-steps 1` | 通过，生成 `model.pt`、`result.json`、`DONE` |
| 真实 UAVR `proto.train --config nano_proto_v6.yaml --max-steps 1` | 通过，验证配置训练入口；按 debug 语义不落 checkpoint |
| 六场景端点词汇诊断 | 通过生成报告；仅诊断，不是正式 G1 |
| OpenFly 官方完整环境/远程评测 | 未跑；资产和接口未准备 |

## 风险与遗留

- 当前 grounder 是词汇启发式，不能处理真正的视觉关系和未知物体；报告中的 18.74% 命中率受端点候选泄漏影响。
- `base_torch_v1` 是 Python 原型，不兼容 C++ v6 checkpoint，也没有闭环 VLN 适配器。
- 长航线候选的 `agl_model_unverified` 路线仍需在正式实验中分层，不能直接当作可靠真值。
