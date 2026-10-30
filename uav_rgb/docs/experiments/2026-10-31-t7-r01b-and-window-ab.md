# T7-R 同版本复跑 + 动作窗口语义 A/B

- 时间：2026-10-30 23:20–2026-10-31 00:15
- 范围：`sim/tools/collect.py`、`sim/tools/eval_closed_loop.py`、
  `sim/tools/test_collect.py`；产物在 `$DATA_ROOT/uav_rgb/t6-full-1640/`
- 环境：RTX 2080 Ti、CUDA 12.8、Gazebo 8.15、`DISPLAY=:20`、12 核 cgroup
- 结论：**R01 的动作窗口语义是 T7 SR 从 0.90 掉到 0.55 的主因**。
  旧模型是在 observation-window 数据上训练的；dispatch-window 评测是
  分布外（OOD）。A/B：observation 2/2 成功，dispatch 0/2 碰撞。

## 1. 命令

```bash
# T7-R 同版本复跑（dispatch 默认）
eval_closed_loop.py --policy student --ckpt .../run/ckpt_best.bin \
  --steps 80 --hold 1.0 --split test --seeds 23,...,180 --keep-work \
  --out .../eval-student-r01b

# A/B
eval_closed_loop.py ... --seeds 49,57 --window-from observation \
  --keep-work --out .../ab-observation
eval_closed_loop.py ... --seeds 49,57 --window-from dispatch \
  --keep-work --out .../ab-dispatch
```

## 2. 结果

| 项 | 旧 T7（observation） | 新 T7-R（dispatch） |
| --- | --- | --- |
| episodes_done | 20/20 | 20/20 |
| SR | **0.90**（18/20） | **0.55**（11/20） |
| collision_rate | 0.05 | 0.45 |
| stuck_rate | 0.00 | 0.00 |
| false_forward_rate | 0.05 | 0.30 |
| 动作分布 | SAFE 57 / FWD 1095 / LEFT 178 / RIGHT 211 | SAFE 14 / FWD 709 / LEFT 263 / RIGHT 351 |
| inference p50/p95 | 40.42 / 59.28 ms | 10.94 / 15.24 ms |
| wall | 3475.8 s | 2501.4 s |

A/B（同一 ckpt、同一环境、紧邻运行）：

| 语义 | seed 49/57 | SR | collision | false_forward | p50 |
| --- | --- | --- | --- | --- | --- |
| observation（旧） | 2/2 成功 | **1.0** | 0.0 | 0.0 | 11.35 ms |
| dispatch（新） | 0/2 碰撞 | **0.0** | 1.0 | 1.0 | 10.97 ms |

两次 A/B 的推理耗时几乎相同，**排除资源竞争/机器差异**。

## 3. T7-R 失败分类（dispatch，20 seed）

| 类型 | seed | 说明 |
| --- | --- | --- |
| 起步早碰撞 | 49、57、99 | collision step 12–16，max_x 1.0–1.24；49/57 场景只差 5 mm（gid 不同是 3 位舍入），动作逐位相同 |
| 过线后碰撞 | 25、36、114、116、137 | max_x 6.46–7.49 > goal_x 6.0，但 80 步内仍碰撞；成功要求全程无碰撞 |
| 未到线碰撞 | 50 | max_x 4.42，未到 goal_x |
| 成功 | 23、45、48、54、84、88、129、147、171、173、180 | 11 集 |

## 4. 解释

- 旧 `collect.py` 的窗口起点是观测位姿 `pose_before_ts`，hold 终点 =
  观测时刻 + 1.0 s；推理/解码耗时从执行窗口里扣掉。R01 改为派发位姿
  `exec_ts` 起算，窗口不再被扣。
- 旧模型是在被扣过的窗口数据上训练的；用 dispatch 评测时每步实际飞得更久
  （A/B 中表现为同一策略从成功变碰撞），属于训练/评测执行语义不一致。
- **0.90 只适用于 observation 语义**；dispatch 是新执行合同，必须用 dispatch
  数据重采重训后才能报数。两者不能混报。

## 5. 没做什么

- **没有改模型/重新训练**：本次只做诊断与 A/B。
- **没有用 observation 语义跑完整 20 seed**：A/B 只跑了 49/57；
  完整 v1 observation 基线已启动（`eval-student-v1-observation`），结果另记。
- **没有启动 T5b 全量重采**：等完整 v1 observation 基线跑完再启动（G 车道互斥）。
- **没有修 D-GEODUP**：seed 49/57 近重复几何只记录，未改 gid/容差。

## 6. 风险

- 20 个 test seed 含近重复几何（49/57），有效独立场景数 <20；旧 0.90 的
  统计置信度没有想象中高。
- Gazebo 非逐位确定；A/B 只有 2 个 seed，结论方向明确，但正式数字仍需
  完整 20 seed + 重复。
- observation 语义有「推理耗时从窗口里扣掉」的已知缺陷；把它作为 v1
  baseline 是兼容训练数据的权宜合同，不是最终执行合同。
