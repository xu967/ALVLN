# 3DGS 长航线候选审计（2026-11-14）

## 目的

为后续控制器配对实验固定一份只读路线候选清单。审计器同时读取 `scene.json` 与其旁边的 `routes_check.json`，检查正式路线与候选过滤结果是否一致，再按 SAGE 参考路径长度筛选长航线。它不重建场景、不改原始路线、不渲染帧。

## 命令与环境

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -B \
  tools/audit_route_candidates.py \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_839992/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_840139/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_840865/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841012/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841150/scene.json \
  --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841980/scene.json \
  --min-path-m 10 --top-k 10 \
  --out /root/rivermind-data/uav_rgb/tmp/route-candidate-audit-20261114/sage3d-route-candidates.json
```

环境为 RTX 2080 Ti；本工具只用标准库，实际运行没有加载 PLY、CUDA 或修改场景。

## 结果

输出：`/root/rivermind-data/uav_rgb/tmp/route-candidate-audit-20261114/sage3d-route-candidates.json`

| 项目 | 数值 |
|---|---:|
| 场景 | 6 |
| `scene.routes` 正式路线 | 249 |
| `routes_check.json` 候选记录 | 676 |
| 通过既有过滤的路线 | 249 |
| SAGE 路径不少于 10 m 的正式路线 | 79 |

六个场景均无完整性错误：每条正式路线都有候选记录，且没有一条通过候选过滤但未写入 `scene.routes`。审计结果带有 `pair_key=scene/route`、`route_index` 和 `max_steps`，后续教师/控制策略应使用同一批 `pair_key`，再比较 SR、OSR、NE、碰撞、停顿和动作切换。

最长候选包括 `sage3d_840139/tr_16`（SAGE 22.243 m）和 `sage3d_840139/tr_3`（21.513 m）。它们属于 train，不能冒充留出测试；`agl_model_unverified` 也应在正式实验中单独分层报告。

## 限制

这是一份候选清单和数据完整性审计，不是控制器性能结果；没有执行任何策略、渲染或动力学仿真。`routes_check.json` 中被淘汰的路线仍保留在完整输出中，未移动或删除。
