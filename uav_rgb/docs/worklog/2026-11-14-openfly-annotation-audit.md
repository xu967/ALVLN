# OpenFly 标注审计工具

- 时间：2026-11-14
- 范围：`tools/`、OpenFly 本地评测接入说明
- 结论：完成

## 做了什么

- 新增 `tools/audit_openfly_annotations.py`，只读 JSON/JSONL，统计记录、split、动作 token、指令长度、轨迹长度、字段出现次数和数组长度不一致。
- 新增 `tools/test_audit_openfly_annotations.py`，覆盖 JSON/JSONL 混合输入、路径长度和输入不被修改。
- 新增 `docs/OPENFLY_LOCAL_EVAL.md`，记录官方本地评测边界、动作空间差异和当前不下载 2.97 TB 数据的原因。

## 没做什么

- 没有下载 OpenFly 全量数据、模型权重或 3DGS 场景：当前磁盘和动作/动力学不匹配，先做 schema 审计。
- 没有修改 `sim/3dgs` evaluator，也没有声称当前代码已经获得 OpenFly 官方分数：官方环境资产和远程提交服务不在现有归档中。
- 没有把工具接入训练流程：本次只建立可复用的数据检查入口，避免引入额外依赖。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `python -m unittest tools.test_audit_openfly_annotations` | 通过（2 tests） |
| `python tools/audit_openfly_annotations.py --input <OpenFly eval_test.json> --output /root/rivermind-data/uav_rgb/tmp/openfly-audit-eval-test.json` | 通过（240 records，动作/位置字段齐全） |
| 本地 OpenFly 归档 `configs/eval_test.json` 的只读格式检查 | 已检查，未运行官方 evaluator |

## 风险与遗留

- OpenFly Annotation 的完整目录尚未下载；工具会对未知顶层结构报错，避免静默漏统计。
- 真正的官方分数仍需隔离环境、对应场景资产和官方 evaluator。
