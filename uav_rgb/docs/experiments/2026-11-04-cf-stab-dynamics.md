# cf_stab 动力学档：Gazebo 冒烟（2026-11-04）

- 环境：GPU 模式（RTX 2080 Ti），Gazebo Harmonic 8.15，`DISPLAY=:20` 软件 GL
- 参数（全部是占位值，等实测替换）：`fwd_tilt_deg=2`、`drag_k=1.0 /s`、`latency_s=0.15`、`hold=0.2 s`
- 档名：`cf_stab_v1:tilt2:k1:lat0.15`；高度按用户决定继续用真值定高
- 结论：采集、audit、preprocess、闭环评测四条链路都能跑通；X3 底层跟踪模型曲线有约 0.4 s 滞后，稳态速度跟得上

## 1. 采集冒烟

命令：

```bash
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 2 --episodes-per-instance 2 --steps 60 --hold 0.2 \
  --dynamics cf_stab --base-seed 0 --out /tmp/cfs_smoke2
```

输出：`== 采集完成: 4/4 episodes, 240/240 steps`，每集约 29 s 墙钟；`src` 统计 `Counter({None: 240})`，即没有误判卡滞，也没有碰撞。
单集节奏（seed 0，40 步）：`total=0.196s/步`，`sim_dt mean=0.202s cv=2.7%`，相当于 5 Hz 连续决策。

## 2. X3 对模型速度曲线的跟踪（seed 0，40 步，按 oracle 位姿差计算）

| 模板 | MulticopterVelocityControl 增益 | 实际/预测 中位数 | 最小 | 卡滞误报（旧阈值 0.5×预测） |
|---|---|---|---|---|
| 仓库原模板 | velocity 2.7，attitude yaw 0.15，rate yaw 0.18 | 0.83 | 0.43 | 2 |
| /tmp 副本 g1 | velocity 6，attitude yaw 1.0，rate yaw 0.5 | 0.93 | 0.29 | 0 |

- 起步段（前 3 步）实际速度明显落后于预测（例如 0.13 预测对 0.02 实际），之后稳态能跟到约 0.30 对 0.33。
- 调高增益能减小稳态误差，但起步段的滞后依旧。仓库模板**没有改**，g1 只是在 /tmp 做的对照。
- 卡滞判定因此改成：模型预测位移达到 `STUCK_EPS=3 cm` 才开始判，且实际位移不到预测的 1/4 才算卡住。改完后 240 步的误报为 0。

## 3. audit（`sim/tools/audit_dataset.py --dataset /tmp/cfs_smoke2`）

- `forward_motion` / `transition_lag`：SKIP（cf_stab 没有固定前进速度，定速判据不适用）。
- `yaw_direction`：第一版 FAIL（13/74）。逐步检查后发现，14 个异常步全部落在偏航的头两步（0.15 s 延迟加跟踪滞后，0.2 s 窗口内机体还没转起来）。改成只核验同向连续第 3 步起，结果 PASS（0/47，起始步 29 个未验）。
- `class_share`：FAIL，SAFE_NEUTRAL 为 0。用旧档 x3_velocity 采 8 步也同样 FAIL（SAFE 和 YAW_LEFT 都缺），说明这是 base 任务加小样本的数据属性，与 cf_stab 无关。

## 4. preprocess 混包门禁

- 只有 cf_stab 集：`wrote /tmp/cfs_ds/dataset.uavr (10371714 bytes)`。
- cf_stab 集和 x3 集混在一起（把 x3 副本的 hold_s 改成 0.2，单独考察动力学字段）：
  `preprocess: episode 的 dynamics_profile 不一致: ['cf_stab_v1:tilt2:k1:lat0.15', 'x3_v1']（不同口径不能混包）`。

## 5. 闭环评测冒烟（盲前进，2 集）

```bash
python3 sim/tools/eval_closed_loop.py --policy forward --episodes 2 --steps 150 \
  --hold 0.2 --dynamics cf_stab --out /tmp/cfs_eval
```

`episodes=2/2 SR=0.0 collision=1.0 stuck=0.0 false_forward=1.0`。seed 1 和 seed 37 都在第 34 步、`max_x≈1.07 m` 处被接触传感器判为碰撞，说明撞柱能被正确检出，没有被误判成卡滞。`eval.json` 中有 `dynamics: cf_stab`，每集带 `dynamics_profile`。

这两集只用来检查链路，不是 baseline 数字（只有 2 集，参数是占位值）。

## 对照评测：x3_velocity vs cf_stab（2026-11-04 14:30）

- 场景：`geom.v5` test split 前 20 个 seed（`1,37,58,113,122,127,132,153,158,172,176,180,190,206,209,230,250,256,259,270`），
  同几何去重后 17 个独立场景，6 组完全同一批。
- 预算：x3 `--hold 1.0 --steps 80`；cf_stab `--hold 0.2 --steps 400`（同为 80 s 仿真时间）。
- student ckpt：`$DATA_ROOT/uav_rgb/t6-full-1640/run/ckpt_best.bin`（x3 数据训练）。
- 结果目录：`$DATA_ROOT/uav_rgb/cfstab-eval-20261104/<组>/eval.json`。

| 组 | SR（17 集） | requested SR（20） | 碰撞 | 卡滞 | false_forward |
|---|---|---|---|---|---|
| x3 teacher（GT planner） | 0.94（16） | 0.80 | 0.06 | 0 | 0 |
| x3 student | 0.59（10） | 0.50 | 0.41 | 0 | 0.24 |
| x3 盲前进 | 0（0） | 0 | 1.00 | 0 | 1.00 |
| cf_stab teacher | 0.65（11） | 0.55 | 0.35 | 0 | 0 |
| cf_stab student | 0.06（1） | 0.05 | 0.76 | 0 | 0.35 |
| cf_stab 盲前进 | 0（0） | 0 | 1.00 | 0.12 | 1.00 |

SR 为 reach 口径，本批 `stop_success_rate` 与 `success_rate` 相同。不是 AvoidBench（PO/EO/AGV/MP 未实现）。
x3 student 0.59 与历史 v1 的 0.90 不可比：场景集不同（旧 seed 在 geom.v5 下已不在 test）。

### 读数

- **cf_stab 下 GT 教师自己就撞 6/17**：3 集在第一根柱（x≈1.1–1.2 m，seed 1/153/172），3 集在 x≈3.8–5.0 m。
  `plan_action` 按「每步可停、原地转」设计（lookahead 1.6 m、clearance 0.36 m），没算延迟 0.15 s 和松杆滑行，
  转向命令发出时机身还在惯性前冲。教师不先修，用它采 cf_stab 数据会把碰撞写进标签。
- **x3 训练的 student 换到 cf_stab 几乎全挂（0.59 → 0.06）**：SAFE_NEUTRAL 占 948/2787 步，3 集原地耗尽 400 步，
  false_forward 0.35。决策频率（1 Hz→5 Hz）和「每步刹停」的训练分布都变了，不能直接迁移。
- 盲前进两档都 100% 撞，碰撞检测在两档下都正常。

### 结论（下一步顺序）

1. 先改教师：用 `CfStab` 前向预测延迟+滑行后的位置，按预测轨迹查净空；目标 cf_stab teacher ≥ 0.9（同 17 场景）。
2. 教师过线后再采 cf_stab 数据（注意 SAFE_NEUTRAL 几乎为 0 的类别分布），重训 student，对照 cf teacher。
3. 真机参数（倾角/阻力/延迟）到位后替换占位值并重跑本表；参数敏感性可在第 1 步后做小扫描。
