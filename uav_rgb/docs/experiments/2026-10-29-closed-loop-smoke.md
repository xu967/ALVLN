# 闭环评测首集（链路验证）

- 时间：2026-10-29 22:30
- 环境：RTX 2080 Ti（GPU 模式）、Gazebo Harmonic 8.15、DISPLAY=:20
- 模型：`/root/rivermind-data/uav_rgb/review-sim-20261029-115357/train/ckpt.bin`
  （小样本训练，仅用于验证链路，不代表泛化）
- 命令：

```bash
source env.sh && source .venv-tools/bin/activate
python3 sim/tools/eval_closed_loop.py \
  --ckpt /root/rivermind-data/uav_rgb/review-sim-20261029-115357/train/ckpt.bin \
  --episodes 1 --steps 20 --out /tmp/uavrgb_eval_smoke --keep-work --max-seeds 200
```

- 真实输出（eval.json 摘要）：

```json
{
  "split": "test",
  "episodes_done": 1,
  "success_rate": 0.0,
  "collision_rate": 1.0,
  "false_forward_rate": 1.0,
  "inference_ms": {"p50": 9.13559, "p95": 11.2789, "n_warm": 7,
                   "cold_first_ms": [310.648]},
  "action_distribution": {"SAFE_NEUTRAL": 0, "FORWARD_SLOW": 8,
                          "YAW_LEFT": 0, "YAW_RIGHT": 0}
}
```

- 单集：seed=5，geometry_gid=`c8afd118245c4551`，split=test，9 根柱，
  goal_x=6.0；学生 8 步全 FORWARD_SLOW，x=1.06 m 撞柱，collect_rc=0，
  wall 30.7 s。每步 total≈1.063 s（I/O 0.062 s，动作窗口 1.000 s）。
- 结论：链路可用且能测出失败；模型行为退化（只会前进），符合「小样本 checkpoint」
  的预期。**不得**把 SR=0 当作算法结论；等 D1（sim-time 动作窗口）与正式数据后再报。
