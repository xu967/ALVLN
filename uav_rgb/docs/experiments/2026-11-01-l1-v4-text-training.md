# L1 v4 外侧 waypoint 任务的文本训练与闭环

- 时间：2026-11-01 19:20 起
- 数据：`$DATA_ROOT/uav_rgb/l1-v4-relation-500`（499 集 / 8966 步）、
  `l1-v4-cf`（451 个反事实集 / 8092 步）
- 结论：离线与文本探针都明显好于 v3 同配方（D 组），**但两组任务定义不同，
  不能直接比较谁更好**——v4 换掉了任务本身（waypoint 改到柱子外侧）。
  闭环结果见文末，完成前不下最终结论。

## 任务定义变化（v3 → v4）

v3 的两个候选 waypoint 都落在出生航线上（横移中位 0.09 m），274 个 test 场景
**没有一个**要求按关系词横移，闭环 SR 只由「敢不敢飞 + 会不会停」决定。
v4 把 waypoint 移到目标柱外侧 0.8 m，横移变成 1.6 m。详见
`docs/worklog/2026-11-01-1739-code-review-waypoint-degeneracy.md`。

## 数据

```bash
# 采集（5 实例 × 100 集 × 60 步上限，与 v3 同 seed 基 900000）
python3 sim/tools/collect_parallel.py --instances 5 --episodes-per-instance 100 \
  --steps 60 --hold 1.0 --base-seed 900000 --task l1 --spawn-mode random \
  --window-from observation --out $DATA_ROOT/uav_rgb/l1-v4-relation-500

python3 sim/tools/audit_dataset.py --dataset $DATA_ROOT/uav_rgb/l1-v4-relation-500
# 17/17 PASS

python3 tools/preprocess.py --drop-collision \
  --out $DATA_ROOT/uav_rgb/l1-v4-relation-500/ds \
  .../l1-v4-relation-500/{train,val,test}/*/

python3 sim/tools/make_counterfactual.py \
  --src $DATA_ROOT/uav_rgb/l1-v4-relation-500 \
  --out $DATA_ROOT/uav_rgb/l1-v4-cf --splits train,val

python3 tools/preprocess.py --drop-collision \
  --out $DATA_ROOT/uav_rgb/l1-v4-cf/ds \
  .../l1-v4-relation-500/{train,val}/*/ .../l1-v4-cf/{train,val}/*/
```

| 项 | v3 | v4 |
| --- | --- | --- |
| 集数 / 步数（可用） | 494 / 7104 | 499 / 8966 |
| train / val / test 集 | 401 / 47 / 49 | 404 / 47 / 48 |
| test 样本 | 697 | 874 |
| 实际横向位移 \|Δy\| p50 | 0.13 m | 1.58 m |
| 采集步数 p50 / max | 14 / 28 | 17 / 28 |
| 反事实标签变化率 | 59.9% | **92.6%** |
| 配对训练集样本 | 12814 | 16184（原集 8092 + 反事实 8092） |
| 真/反事实 waypoint 间距 | p25 0.88 / p50 1.66 m | **min 3.23 / p50 3.32 m** |

`audit_dataset.py` 17/17 PASS。v3 的三个 FAIL 全部消失：`class_share`
（YAW 两类从 3.8%/4.0% 升到 8.5%/9.2%）、`time_pairing`（max_gap 100 ms → 36 ms）、
`safe_collision`（1 个 SAFE 碰撞步 → 0）。

## 训练

```bash
src/build/uavrgb_train --data l1-v4-cf/ds/dataset.uavr --text-dim 16 \
  --epochs 20 --batch 64 --lr 5e-4 --seed 1234 --class-weights auto \
  --drop-collision --out l1-v4-cf/run-film-cf
# 289920 样本 / 20 轮 / 4864.9 s（81.1 min）
# class_weights auto = [2.990, 0.734, 0.868, 0.869]
# ckpt_best.bin 落在 E17（dev macro_f1 0.8702），md5 d91cd0f1…
```

`ckpt_best.bin` 与本轮末轮 `ckpt.bin`（md5 `22188929…`）不同，选权重正常
（v3 的 `carry_resume_best` 冻结问题本轮回 resume，碰不到）。

## 离线（`l1-v4-relation-500/ds` 的 test split，874 样本，未参与训练与模型选择）

| 权重 | acc | macro_f1 | bal_acc | 每类 recall |
| --- | --- | --- | --- | --- |
| 多数类先验（全猜 FORWARD_SLOW） | 0.6590 | 0.1986 | 0.2500 | 0 / 1 / 0 / 0 |
| **`ckpt_best.bin`（E17）** | **0.8478** | **0.7740** | **0.7623** | 0.833 / 0.903 / 0.676 / 0.637 |
| 末轮 `ckpt.bin`（E20） | 0.8467 | 0.7633 | 0.7405 | 0.764 / 0.924 / 0.662 / 0.613 |

## 文本探针（test split 的 801 个决定性帧）

```bash
python3 sim/tools/probe_text_effect.py \
  --src $DATA_ROOT/uav_rgb/l1-v4-relation-500 \
  --ckpt l1-v4-cf/run-film-cf/ckpt_best.bin --split test \
  --out l1-v4-cf/probe-best.json
```

| 模型 | flip（≥0.50） | acc_correct（≥0.90） | acc_alt | follow |
| --- | --- | --- | --- | --- |
| v3 D `run-film-cf2/ckpt_best` | 0.8280 | 0.7890 | 0.7294 | 0.6055 |
| **v4 `run-film-cf/ckpt_best`** | **0.9638** | **0.8814** | 0.9026 | **0.8090** |

三项都涨了：flip +13.6pp、acc_correct +9.2pp、follow +20.4pp。v3 里
「flip 高必然 macro_f1 低」的权衡被推开——v3 A 组 flip 0.970 时 macro_f1 只有
0.479，v4 在 flip 0.964 下拿到 macro_f1 0.774。

`acc_correct ≥ 0.90` 仍未过（0.8814）。这个阈值出自内部计划
（`experiments/2026-10-31-basic-vln-plan.md`），不是公开基准。

## 闭环（48 个 test seed，去重后 47 集）

```bash
SEEDS=$(ls l1-v4-relation-500/test | paste -sd,)
python3 sim/tools/eval_closed_loop.py --policy forward --task l1 --split test \
  --seeds "$SEEDS" --episodes 48 --steps 40 --hold 1.0 \
  --window-from observation --out .../l1-v4-closed-20261101/forward
python3 sim/tools/eval_closed_loop.py --policy student --ckpt run-film-cf/ckpt_best.bin \
  --text-mode keep|shuffle ... --out .../student-keep|student-shuffle
```

| 策略 | 完成 | SR（goal_reach） | 碰撞率 | stuck | false_forward | 推理 p50/p95 |
| --- | --- | --- | --- | --- | --- | --- |
| `student` + keep | 47/48 | **0.426** | 0.277 | 0.000 | 0.234 | 13.3 / 35.4 ms |
| `student` + shuffle | 47/48 | **0.000** | 0.383 | 0.000 | 0.362 | 13.9 / 36.3 ms |
| 盲前进 | 44/48 | **0.000** | 0.455 | 0.000 | 0.455 | — |

三组失败集不同（盲前进有 3 集 infra 超时，学生集全部完成），所以另取**共同完成的
44 集**再对一次：

| 策略 | SR | 碰撞率 |
| --- | --- | --- |
| 盲前进 | 0.000 | 0.455 |
| student + keep | **0.409（18/44）** | **0.295** |
| student + shuffle | 0.000 | 0.409 |

keep 的 Wilson 95% CI（47 集）：`[0.295, 0.567]`。McNemar（keep 成功 20 集、
shuffle 0 集）精确 **p < 1e-5**。

### 口径变更：goal_stop → goal_reach

同一批 episode 按旧口径（0.7 m + 末动作 SAFE）是 **0.340**，按新口径（1.5 m、
不看末动作）是 **0.426**。差别只来自 4 集「停在了 1.23–1.47 m 处」。
两种口径不可混报，`eval.json` 的 `success_criterion` 已分开。

上表的 SR 是**用已记录 episode 的 `final_waypoint_dist`/`collision` 复算**的
（新判据是这两个字段的纯函数，不依赖轨迹）。代码改完后另跑了 2 集端到端
（900160 / 900101）确认 `success_criterion=goal_reach` 且判定与复算一致；
三组全量没有重跑（要 80 min，数值必然相同）。

**keep 47 集的失败分解（三者互斥，刚好 47）**：

| 结局 | 集数 | 最终距 waypoint |
| --- | --- | --- |
| 成功 | 20 | ≤1.5 m 且无碰撞 |
| **到了却撞柱** | 13 | **13/13 都在 ≤1.5 m 内撞的**——已经飞到目标位置，停不下来 |
| 到了但没进 1.5 m | 14 | 1.62, 1.67, 1.69, 1.85, 1.86, 1.89, 1.90, 2.23, 2.33, 2.40, 2.49, 2.49, 2.87, 6.73 |

换成不同终点的容差（无碰撞且终点 ≤X m）：

| X | 0.7 | 1.0 | 1.5 | 2.0 |
| --- | --- | --- | --- | --- |
| keep | 34% | 34% | 43% | 57% |
| shuffle | **0%** | **0%** | **0%** | **0%** |

### 结论：指令学会了，停不会

- **指令确实在起作用**：shuffle 下没有任何一集能靠近真 waypoint 2 m 以内
  （两个候选 waypoint 相距 3.2 m），keep 与 shuffle 的分离度是满的。
- **瓶颈是「停」不是「选边」**：13 集碰撞全部发生在距 waypoint 1.5 m 以内——
  无人机已经飞到目标位置，停不下来，撞在目标柱上；另有 14 集停在 1.6–2.9 m 处。
  两类加起来 27/47，全部与「停得准」有关，与选边无关。
- **原因是可观测性**：详见 `docs/worklog/2026-11-01-2130-stop-point-unobservable.md`。
  waypoint 在柱阵外侧 ±2.4 m，相机 60° 视场在**真停止点上 40% 的帧一根柱子都看不到**；
  同样的空画面在飞行途中也出现（13% 的 FORWARD 帧）。模型学成了「柱子消失 → 停」，
  于是提前停 1–2 m，或撞上还没让开的柱子。
