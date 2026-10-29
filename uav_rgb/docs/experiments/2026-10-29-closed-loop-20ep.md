# 闭环评测 20 集基线（链路验证，不作验收）

- 时间：2026-10-29 22:32–22:47
- 环境：RTX 2080 Ti（GPU 模式）、Gazebo Harmonic 8.15、DISPLAY=:20
- 模型：`/root/rivermind-data/uav_rgb/review-sim-20261029-115357/train/ckpt.bin`
- split：`test`（几何组隔离，20 集 gid 互不相同）
- 命令：

```bash
source env.sh && source .venv-tools/bin/activate
python3 sim/tools/eval_closed_loop.py \
  --ckpt /root/rivermind-data/uav_rgb/review-sim-20261029-115357/train/ckpt.bin \
  --episodes 20 --steps 80 --out /root/rivermind-data/uav_rgb/eval-20261029-2235 \
  --keep-work --max-seeds 1000
```

- 真实输出（`eval.json`，schema `uavrgb.eval.v1`）：

```json
{
  "episodes_requested": 20, "episodes_done": 20, "failed": [],
  "success_rate": 0.0, "collision_rate": 1.0, "false_forward_rate": 1.0,
  "wall_s": 681.3, "steps": 80, "hold_s": 1.0,
  "inference_ms": {"p50": 9.75957, "p95": 12.9561, "n_warm": 154,
                   "p50_all": 10.1198, "p95_all": 322.879},
  "action_distribution": {"SAFE_NEUTRAL": 0, "FORWARD_SLOW": 169,
                          "YAW_LEFT": 5, "YAW_RIGHT": 0}
}
```

| 指标 | 值 |
| --- | --- |
| 成功率 | 0 / 20 |
| 碰撞率 | 20 / 20 |
| false_forward（集级） | 20 / 20 |
| 推理 p50 / p95（warm n=154） | 9.76 / 12.96 ms |
| 首帧（CUDA 初始化） | 292.8–350.5 ms，20 集均 ≈ 320 ms |
| 每集步数 | 7–14（撞柱即停），中位 8 |
| 每集 wall | 28.9–38.4 s |
| `max_x` | 1.003–2.107 m（goal_x = 6.0 m） |

## 结论

**链路可用，模型不可用。** 20 集全部在 1.0–2.1 m 处撞柱，动作 169/174 是
`FORWARD_SLOW`，`SAFE_NEUTRAL` 与 `YAW_RIGHT` 一次都没出现。这是小样本
checkpoint 的行为退化，与训练集标签分布（`/tmp/ds200_bin/dataset.uavr`：
`SAFE_NEUTRAL` 3726 个样本里只有 1 条）一致。

**不得**把这组数字当作算法结论或对外结果。原因有三：
1. `--hold` 仍是墙钟秒，动作窗口随 RTF 漂移（见 `2026-10-29-2241-dataset-audit.md`）。
2. 旧 checkpoint 的训练 split 用旧策略（`scene_id`），与 salt v2 几何 split 不一致。
3. 训练数据本身未过验收闸门（5 项 FAIL）。

## 复现与残留

- `eval.json` 与原始 episode 保留在 `/root/rivermind-data/uav_rgb/eval-20261029-2235/`
  （`.work/` 下 20 个 ep 目录，含 RGB PNG、actions/oracle、场景 SDF）。
- 全程串行，无残留 `gz sim` / `ruby` / `uavrgb_infer` 进程。
