# T5b：dispatch 语义全量重采（600 集）

- 时间：2026-10-31 01:20–05:00（world_gen 35.1 s + collect 13258.6 s）
- 范围：`sim/tools/collect_parallel.py`，4 实例 × 150 集，40 步，hold=1.0，
  base-seed 200000，teacher，safe 出生 20%
- 输出：`$DATA_ROOT/uav_rgb/t5b-dispatch-20261031/collection`
- 结论：采集完成；隔离 4 集 SAFE+碰撞后 audit **PASS**（596 集 / 23840 步）。

## 1. 命令

```bash
BASE5=$DATA_ROOT/uav_rgb/t5b-dispatch-20261031
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg "$UAV_RGB_PYTHON" sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 150 --steps 40 --hold 1.0 \
  --base-seed 200000 --policy teacher --spawn-mode default --safe-spawn-fraction 0.2 \
  --out "$BASE5/collection"
```

## 2. 采集结果

| 项 | 结果 |
| --- | --- |
| 集数 / 步数 | 600/600，24000/24000 |
| split | train 463 / val 70 / test 67 |
| 吞吐 | 1.810 步/s（4 实例端到端）；collect 13258.6 s（3.68 h） |
| 残留 | 0 ruby 进程；四实例全部正常收尾 |

## 3. Audit

首轮 `audit_dataset.py`：**FAIL safe_collision** —— 10 个 SAFE 步带 contact
（dispatch 窗口变长，教师贴着柱子输出 SAFE），另有已排除 10 步。

按 T5 的处置：整集隔离 4 个受影响集到 `quarantine/`，不删、不改标签：

| split | episode | SAFE+碰撞步 |
| --- | --- | --- |
| train | 200067 | 4 |
| train | 200314 | 1 |
| train | 200587 | 1 |
| val | 200298 | 4 |

隔离后 audit：**PASS**，596 集 / 23840 步，其余检查全过：

- `time_pairing` max_gap 36 ms；`action_window` cv 0.1%
- `forward_motion` 0/12022；`transition_lag` 47%（阈值 20%）
- `yaw_direction` 0/4491（跳过 33 个反向接续步）
- `class_share` 无 <5% 类、无 split 整类缺席
- `safe_collision` 0/4824
- `geometry_version` v3，常量核对 596/596
- `split_geometry` 交集 0（test 49 / train 325 / val 44）
- `label_recompute` 0/23840 不一致
- `instruction_coverage` SKIP（T5b 无 instruction；L1 才有）

审计 JSON：`audit-after-quarantine.json`；隔离清单：`quarantine.json`。

## 4. 没做什么

- 没有重采被隔离的 4 集：它们的问题是 SAFE+碰撞，重采同一 seed 也可能复现；
  直接隔离，保留原始数据以备分析。
- 没有改 teacher/动作窗口：dispatch 是执行合同，teacher 的 SAFE+碰撞属
  真实行为，不用调参掩盖。
- 没有训练/复评：下一步 preprocess → train → eval。

## 5. 下一步

```bash
"$UAV_RGB_PYTHON" tools/preprocess.py --drop-collision --out "$BASE5/ds" \
  "$BASE5/collection/train"/*/ "$BASE5/collection/val"/*/ "$BASE5/collection/test"/*/
src/build/uavrgb_train --data "$BASE5/ds/dataset.uavr" --epochs 10 --batch 64 \
  --lr 1e-3 --class-weights auto --drop-collision --out "$BASE5/run"
# eval-only + 20 seed 闭环（dispatch），与 v1 observation 0.90 对比
```

## 6. 风险

- 596 集里可能有未触发 SAFE+碰撞的 contact 步；preprocess 用 `--drop-collision` 过滤。
- dispatch 合同下教师 SAFE 比例约 20%；训练后闭环是否稳定要看 20 seed 复评。
- 与 v1（observation）不可混报；T5b 是新执行合同的数据。

## 7. 训练与离线复评（2026-10-31 05:30）

```bash
src/build/uavrgb_train --data "$BASE5/ds/dataset.uavr" --epochs 10 --batch 64 \
  --lr 1e-3 --class-weights auto --drop-collision --out "$BASE5/run"
src/build/uavrgb_train --data "$BASE5/ds/dataset.uavr" \
  --eval-only "$BASE5/run/ckpt_best.bin" --out "$BASE5/eval-best" \
  --batch 128 --split all --drop-collision
```

| 项 | 结果 |
| --- | --- |
| 训练 | 183770 样本 × 10 轮，2632.0 s，69.8 样本/秒 |
| best | epoch 5，dev macro_f1 0.8357 |
| val（best 复评） | ce 0.3196 / acc 0.9025 / **macro_f1 0.8357** / bal_acc 0.8572 / recall [1.000 0.912 0.773 0.744] |
| test（best 复评） | ce 0.3217 / acc 0.8940 / **macro_f1 0.8287** / bal_acc 0.8494 / recall [1.000 0.914 0.747 0.737] |

对比 T6（observation 数据）：test macro_f1 0.8589 → T5b 0.8287，低 3.0 pp。
dispatch 数据的类别分布/执行窗口不同，属预期内的轻微下降，不改变合同判断。

闭环复评（dispatch 20 seed，`eval-student-dispatch`）进行中，结果另记。

## 8. dispatch 闭环复评（2026-10-31 06:10）

命令：20 个冻结 test seed，`--window-from dispatch --keep-work`，
`ckpt_best.bin`（T5b 训练，dispatch 数据）。

| 项 | 结果 |
| --- | --- |
| SR | **0.60**（12/20） |
| collision_rate | 0.35 |
| false_forward_rate | 0.25 |
| stuck_rate | 0.00 |
| inference p50/p95 | 10.10 / 11.84 ms |
| failed | [] |

失败集：23、25、57、84、88、171、180 碰撞（其中 23/57/84/88/171 为 false_forward）；
45 未成功（max_x 5.63，未到 goal 6.0）。

### 结论

- v1（observation 模型 + observation 评测）：**0.90**
- T7-R（observation 模型 + dispatch 评测，OOD）：**0.55**
- T5b（dispatch 模型 + dispatch 评测）：**0.60**

即使按 dispatch 语义重采重训，SR 也只从 0.55 提到 0.60，远低于「v1 − 5 pp」
（0.85）的采纳线。**dispatch（hold=1.0）不作为 baseline 采纳**；v1 observation
仍是当前唯一稳定基线，语言线（L1/L2）按 observation 语义采集/评测。
dispatch 合同若要继续，应先调 hold/动作适配/教师，而不是直接换数据。
T5b 数据与模型保留作证据，不与 v1 数据混训/混报。
