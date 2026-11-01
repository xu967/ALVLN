# L1 v4 闭环按 goal_reach 重跑（2026-11-02）

- 起因：`l1-v4-closed-20261101` 的 `eval.json` 实际是 `goal_stop` 口径（SR 0.3404），
  文档里的 0.426 是对旧 `per_episode` 手工复算的 goal_reach，**没有评测文件支撑**。
- 目的：拿到有 `eval.json` 背书的 goal_reach 数字。
- 产物：`$DATA_ROOT/uav_rgb/l1-v4-closed-goalreach-20261102/`

## 命令

```bash
python3 sim/tools/eval_closed_loop.py --policy student \
  --ckpt $DATA_ROOT/uav_rgb/l1-v4-cf/run-film-cf/ckpt_best.bin \
  --task l1 --split test --keep-work \
  --seeds 900004,...,900493 --text-mode keep \
  --out $DATA_ROOT/uav_rgb/l1-v4-closed-goalreach-20261102/student-keep
# shuffle 同上改 --text-mode shuffle；forward 换 --policy forward 且用 44 个 seed
```

`--window-from` 未传——2026-11-02 起默认已是 `observation`（唯一被采纳基线），
`eval.json` 里记录 `window_from=observation`。

## 结果

| 组 | SR | 碰撞率 | false_forward | stuck | 完成 | 墙钟 |
| --- | --- | --- | --- | --- | --- | --- |
| student-keep | **0.4043**（19/47） | 0.319 | 0.277 | 0.000 | 47 | 2266 s |
| student-shuffle | **0.000** | 0.298 | 0.255 | 0.000 | 47 | 2263 s |
| forward | **0.000** | 0.463 | 0.463 | 0.000 | 41 | 3731 s |

- keep 的 Wilson 95% CI：**[0.276, 0.547]**
- 共同完成的 41 集：keep SR **0.4146**
- keep 19 成功 vs shuffle 0 成功，McNemar 精确 **p < 1e-5**
- 推理 p50：keep 15.9 ms / shuffle 17.4 ms（本次运行期间有 torch 安装并发，
  历史同模型为 10.26 ms；SR 不受影响，时延数字不宜作为基准）

## 失败分解（keep，47 集）

| 类别 | 集数 |
| --- | --- |
| 成功 | 19 |
| **撞柱** | **15（15/15 的 `final_waypoint_dist` ≤1.5 m）** |
| 未到达 waypoint（未碰撞） | 13 |

与 2026-11-01 的结论一致：**失败集中在「到了但停不下来」**，不是选边选错。
最终距离 p25/p50/p75 = 0.33 / 0.77 / 1.63 m。

## 与旧数字的关系

| 数字 | 性质 |
| --- | --- |
| 0.3404（`l1-v4-closed-20261101`） | 实际 `eval.json`，goal_stop 口径 |
| 0.4255 | 对旧 per_episode 手工复算 goal_reach，**无文件背书** |
| **0.4043**（本次） | 实际 `eval.json`，goal_reach + observation，**唯一可引用** |

0.4043 与 0.4255 的差异来自 Gazebo 非逐位确定性（同 seed 两次采集轨迹可不同，
HANDOFF §八 坑 #10），不是口径变化。

## 其他

- forward 组有 3 集起飞超时（900031 / 900054 / 900426），rc=1，不计入分母。
  历史上同一现象出现在另外 3 个 seed（900032 / 900052 / 900101），
  说明是环境偶发，不是特定场景。
- 验收状态：**仍未过 50% 线**。按计划推进 B（改几何）/ C（改任务）或
  GRU + 特权蒸馏路线。
