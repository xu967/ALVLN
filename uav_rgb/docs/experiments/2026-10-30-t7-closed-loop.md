# T7 学生闭环验收（20 个冻结 test seed）

- 时间：2026-10-30 18:31–19:29（3475.8 s）
- 权重：`$DATA_ROOT/uav_rgb/t6-full-1640/run/ckpt_best.bin`（T6 best，epoch 8）
- 场景：固定 20 个 test split seed，与盲前进基线同一批
  `23,25,36,45,48,49,50,54,57,84,88,99,114,116,129,137,147,171,173,180`
- 产物：`$DATA_ROOT/uav_rgb/t6-full-1640/eval-student/eval.json`
- 状态：**完成**；满足 T7 验收线（SR ≥ 50% 且 false_forward 低于盲前进）

## 1. 命令

```bash
source env.sh
BASE=/root/rivermind-data/uav_rgb/t6-full-1640
"$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py \
  --policy student --ckpt "$BASE/run/ckpt_best.bin" \
  --steps 80 --hold 1.0 --split test \
  --seeds 23,25,36,45,48,49,50,54,57,84,88,99,114,116,129,137,147,171,173,180 \
  --out "$BASE/eval-student"
```

每个场景最多 80 步、`hold=1.0` 仿真秒；碰撞即停。碰撞按 contact 计，stuck 单列
（`EVAL_STANDARD §5.1`）。

## 2. 结果

| 指标 | 盲前进基线（同 20 seed） | **T6 best 学生** |
| --- | --- | --- |
| success_rate | 0.00 | **0.90**（18/20） |
| collision_rate | 1.00 | **0.05**（1/20） |
| stuck_rate | 0.00 | **0.00** |
| false_forward_rate | 1.00 | **0.05**（1/20） |
| 总步数 | 140 | 1541 |
| 动作分布 | FWD 140 | SAFE 57 / FWD 1095 / LEFT 178 / RIGHT 211 |
| inference p50 / p95 | — | 40.4 / 59.3 ms（warm n=1521） |
| 冷启动首帧 | — | 0.34–1.70 s（每集第一条） |

离线多数类先验（test）：acc 0.6492 / macro_f1 0.1968 / bal_acc 0.2500；
T6 best 离线：acc 0.9186 / macro_f1 0.8589 / bal_acc 0.8497。闭环同时超过两条基线。

### 失败集

| seed | 步数 | 结束原因 | max_x | goal_x | 动作分布 |
| --- | --- | --- | --- | --- | --- |
| 45 | 80 | 无碰撞，但一直悬停（SAFE 50 次），未到终点 | 3.26 | 6.0 | SAFE 50 / FWD 17 / LEFT 8 / RIGHT 5 |
| 99 | 21 | FORWARD 撞柱（false_forward） | 1.46 | 6.0 | FWD 11 / LEFT 4 / RIGHT 6 |

成功的 18 集 max_x 最小 7.63（goal 6.0），不是擦线通过。seed 45 是「学成悬停」
的失败模式，不是碰撞，也不计 stuck（stuck 只统计 FORWARD 位移不足）。

## 3. 口径与注意

- **本轮跑在 R01 修复之前**：旧 `collect.py` 把观测戳当动作窗口起点，推理耗时
  （本轮 p50 40.4 ms，共享 GPU 上比冒烟时的 6.9 ms 慢）会从每步执行窗口里扣掉，
  首步冷启动扣得更多。R01 修复后窗口从派发位姿起算，执行时间不再被扣。但窗口
  变长也可能让每步走得更远、更容易撞，**SR 的变化方向不确定**；本轮只代表修复前
  版本，没有为此重跑 20 集（SR 已远高于 50% 验收线，严格验收需同版本重跑）。
- 推理 p50 40 ms 包含共享 GPU 上的竞争；R01 修复后另给 `StudentPolicy` 加了
  启动 warmup，把每集首帧的 CUDA 首启开销移出第一步。
- test 的 SAFE 样本全部来自 safe 出生悬停集（T6 实验记录 §3）。闭环里学生
  输出 57 次 SAFE，说明它不是只会在悬停画面里停；但 seed 45 也暴露了
  「不确定时一直 SAFE」的失败模式，后续若要提高 SR，需要针对这种状态加数据
  或加代价。
- 本轮没有对失败集做逐帧可视化/重规划分析。

## 4. 没做什么

- 没有在 R01 修复后重跑 T7：本轮已满足验收且失败集不是擦线，重跑成本
  （约 58 min）与信息增益不成比例；本条已在 worklog 记为已知差异。
- 没有增加更多 test seed：20 个是 TASKS 预先冻结的口径。
- 没有调整 `--steps/--hold/--goal-margin`：与盲前进基线完全一致。
- 没有 commit：用户未授权。

## 更新 2026-10-30 22:35：R01/D4 复跑中断

21:29 启动的同版本复跑在 14/20 集被会话结束连带杀掉；`eval-student-r01/`
无 `eval.json`，日志停在 seed 116。中间快照 7 成功 / 7 碰撞，**不是结论**。
下一步用 `eval-student-r01b` + `--keep-work` 重跑；完成前 §3 的 0.90 仍只适用于
R01 前版本。
