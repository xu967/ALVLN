# uav_rgb 交接文档

- 更新时间：**2026-11-01 21:52**
- **当前状态**：T1–T7 已完成（v1 observation baseline `SR 0.90`）。
  语言线走到 **L1 任务 v4**：FiLM 文本分支 + 外侧 waypoint 任务，
  离线 `macro_f1 0.7740`、文本探针 `follow 0.8090`，闭环 keep `SR 0.426` /
  shuffle `0.000` / 盲前进 `0.000`。
  **指令确实进决策了，但「停得准」做不到**——13 次碰撞全部落在距 waypoint
  1.5 m 以内。原因已定位：v4 的停止点在柱阵外侧 ±2.4 m，相机 60° 视场在
  真停止点上 40% 的帧一根柱子都看不到，「停」不可观测。
  用户 2026-11-01 拍板方案 A（去掉「必须停」→ goal-reach），已实现；
  **SR 0.426 仍未过 50% 验收线**。
- **接手第一件事**：本文件 → **§4.5（当前任务 + 下一步命令）** →
  `docs/EVAL_STANDARD.md` → `docs/TASKS.md` → `docs/ENGINEERING_RULES.md`
- **口径唯一来源**：`docs/EVAL_STANDARD.md`。报任何数字前必读，违反口径的数字等于造假。
  **2026-11-01 起 L1 闭环口径是 `goal_reach`（1.5 m、不看末动作），
  与历史的 `goal_stop`（0.7 m + 末动作 SAFE）不可混报。**
- ⚠️ **工作树有大量未提交改动**（含并发会话的），`ALVLN/uav_rgb` 仍是旧版。
  提交前必须逐文件确认归属，见 §5「Git」。

### 30 秒上手

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb && source env.sh
nvidia-smi -L          # 期望列出 RTX 2080 Ti；列不出就是无卡模式，见 §五
bash tools/check.sh    # 期望最后 EXIT=0，CTest 21/21
```

三条都过，说明环境是活的；然后跳到 §4.5 看当前任务，再从 §七 看下一步。

---

## 一、项目是什么

单目 RGB 无人机避障：机载相机 160×120 → 四个离散动作的 CNN。

```cpp
// src/include/uavrgb/action.hpp
enum class Action : std::uint8_t {
  SAFE_NEUTRAL = 0, FORWARD_SLOW = 1, YAW_LEFT = 2, YAW_RIGHT = 3
};
```

**硬约束（不可违反）**

- 模型**只**输出这 4 个离散动作，绝不输出 PWM/转速/姿态角；ESP32 飞控保留最终否决权。
- 训练/推理**全部自写 C++/CUDA**。**禁止** PyTorch / TensorRT / cuDNN / cuBLAS / ROS 2。
- 深度 / 位姿 / 障碍几何 / 碰撞真值 = **特权信息**，只能进专家和 oracle 文件，
  **永远不进学生模型输入**。

管线：`scene_gen`（按 seed 生成柱阵世界）→ `collect`（Gazebo 里专家飞、存 RGB+标签）
→ `preprocess`（打包 `.uavr`）→ `uavrgb_train`（自写 CUDA 训练）
→ `uavrgb_infer`（常驻推理进程）→ `eval_closed_loop`（Gazebo 里学生自己飞）。

## 二、环境

| 项 | 事实 |
| --- | --- |
| 机器 | Linux x86_64，**RTX 2080 Ti（sm_75，22 GiB）** |
| CUDA | 12.8（`/usr/local/cuda-12.8`），driver 580.119.02。**不要装 CUDA 13.x，绝不编 sm_120** |
| Gazebo | Harmonic 8.15，`DISPLAY=:20`（软件 GL / llvmpipe） |
| 编译 | CMake 3.28 + Ninja，GCC 13.3 |
| Python | `.venv-tools`（3.12.3），**只用于数据工具** |
| 故意没装 | cuDNN、PyTorch、TensorRT、ROS 2 —— **不要装** |

**每个 shell 的第一件事**（必须从仓库根目录）：

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb
source env.sh                                   # env.sh 自定位 UAV_RGB_ROOT
source .venv-tools/bin/activate                 # 仅在跑 Python 数据工具时
nvidia-smi -L                                   # 先判定 GPU 模式，别凭印象
```

> **踩过的坑**：在仓库外的目录 `source env.sh` 会**静默失败**（文件不存在，不报错），
> 后面的 python 命令全部不执行。写脚本时永远先 `cd` 回仓库根。

## 三、仓库地图

| 路径 | 作用 |
| --- | --- |
| `src/` | C++/CUDA 训练推理栈：`train_main.cpp`（CLI）、`trainer.cpp` / `trainer_eval.cpp`、`infer_main.cpp`（常驻推理进程）、`dataset.cpp`、`checkpoint.cpp`（v4，best-checkpoint + text_dim）、`src/tests/`（**21 个 CTest**） |
| `sim/tools/` | `scene_gen.py`（柱阵世界生成）、`collect.py` / `collect_parallel.py`（采集）、`teacher.py`（`plan_action` GT planner / `expert_action` 深度专家）、`audit_dataset.py`（**18 条数据验收闸门**）、`eval_closed_loop.py`（闭环评测）、`split_manifest.py`（几何 gid + split）、13 个 `test_*.py` |
| `tools/` | `preprocess.py`（采集目录 → `.uavr`）、`airsim_to_uavr.py`（公开数据接入）、`check.sh`（**L2 全量验证**）、`doctor.py` |
| `ui/` | Tkinter 训练监控台（**另一个会话维护，改前先确认**） |
| `docs/` | `TASKS.md`（任务板，**唯一任务来源**）、`ENGINEERING_RULES.md`（执行细则）、`EVAL_STANDARD.md`（口径）、`worklog/`、`experiments/` |
| 仓库外 | 数据与产物在 `/root/rivermind-data/uav_rgb/` |

**三级验证**：L1 相关单测（每次改）→ **L2 `bash tools/check.sh`**（每个任务完成，不许
只跑子集）→ L3 真实 Gazebo / 训练（里程碑，先在 TASKS.md 登记，同一时刻只一个会话跑）。

## 四、当前状态与关键数字

> **2026-11-01 21:52 更新。§4.1 / §4.2 是历史记录（T1–T7，v1 基线）；**
> **当前可用数字看 §4.3，当前任务看 §4.5。** 更早的「闭环 0 / 20」
> 「公开数据 SAFE=0」等已移出，需要时看 `docs/experiments/` 与 `docs/worklog/`。

### 4.1 执行语义三条合同：v1 / T7-R / T5b

**T7 同版本复跑（dispatch 默认）已完成**：20/20 集，`SR 0.55`（11/20）、
collision 0.45、false_forward 0.30、stuck 0；产物
`$BASE/eval-student-r01b/eval.json`，实验记录
`experiments/2026-10-31-t7-r01b-and-window-ab.md`。

**A/B（同一 ckpt、同一环境、紧邻运行）**：`observation` 语义 seed 49/57
**2/2 成功**；`dispatch` 语义 **0/2 碰撞**。推理 p50 两次都约 11 ms，
排除资源竞争。结论：**R01 的动作窗口语义是 0.90 → 0.55 的主因**；旧模型
训练在 observation 数据上，dispatch 评测是分布外。

| 项 | 值 |
| --- | --- |
| 旧 T7（observation 语义） | SR 0.90（18/20）；collision 0.05；p50 40.4 ms |
| **v1 复测（observation，当前环境）** | **SR 0.90（18/20）**；collision 0.05；p50 10.26 ms；失败 seed 23/88 |
| T7-R（dispatch 语义） | SR 0.55（11/20）；collision 0.45；p50 10.9 ms |
| **T5b（dispatch 重采重训）** | **SR 0.60（12/20）**；collision 0.35；p50 10.1 ms；离线 test macro_f1 0.8287 |
| A/B observation | seed 49/57，2/2 成功 |
| A/B dispatch | seed 49/57，0/2 碰撞 |

**合同结论（2026-10-31 06:15）**：

1. **v1 observation 是唯一 baseline**（SR 0.90）；语言线（L1/L2）按 observation
   语义采集/评测。
2. dispatch（hold=1.0）重采重训后只有 0.60，低于「v1 − 5 pp」的采纳线（0.85）；
   **不采纳为 baseline**。若继续，先调 hold/动作适配/教师，不是换数据。
3. T5b 数据/模型保留作证据，**不与 v1 混训/混报**；dispatch 数字必须标注合同。
4. L1 用 observation 语义：`collect_parallel.py` 需透传 `--window-from`；
   评测用 `eval_closed_loop.py --window-from observation`。

**禁止**：同时启动第二个 Gazebo 采集/评测/训练；先 `mkdir` 输出目录
（`eval_closed_loop.py` 拒绝非空目录）。日志按集 flush，日志不动不等于卡死，
先 `ps -p <pid>`。

### 4.2 已完成

| 任务 | 结果 |
| --- | --- |
| T1 | v3 几何 + A* 教师，20/20 test 场景无碰撞；见 `experiments/2026-10-30-t1-lane-geometry.md` |
| T2 | safe 出生 + 52 集/2080 步，0 碰撞，audit 11/11 PASS；见 `experiments/2026-10-30-t2-safe-neutral.md` |
| T3 | 10 轮端到端放行，test 四类 recall 全 >0；见 `worklog/2026-10-30-0500-t3-rollgate.md` |
| T4 | `uavrgb_train --eval-only`；CTest 20/20；见 `experiments/2026-10-30-t4-eval-only.md` |
| T5 | 599 集 / 23960 步，audit 11/11 PASS；见 `experiments/2026-10-30-t5-scaleup.md` |
| T6 | best 权重 test `acc 0.9186 / macro_f1 0.8589 / bal_acc 0.8497`；见 `experiments/2026-10-30-t6-train.md` |
| T7（旧版） | 20 个冻结 test seed：`SR 0.90 / collision 0.05 / stuck 0.00 / false_forward 0.05`；**R01 前版本**，见 `experiments/2026-10-30-t7-closed-loop.md` |
| D4 | depth 与 RGB sim-time 差 >100 ms、contact 订阅进程退出即失败；`test_collect` 20/20；见 `worklog/2026-10-30-2129-d4-staleness-gate.md` |
| R01 | 动作窗口从派发位姿起算 + `StudentPolicy` warmup；见 `worklog/2026-10-30-1935-deepseek-review-fixes.md` |
| L1 | **v1 400 集作废**（route-choice，teacher 0/400 到 goal）；v2 goal-stop 500 集采集中。见 `experiments/2026-10-31-basic-vln-plan.md §2.5/§8` |
| D-STACK | 用户拍板**全自研 C++/CUDA**；文本分支（token 查表 → 按位置拼接 → FiLM）已实现，CTest 21/21；见 `worklog/2026-10-31-1115-l2-text-branch.md` |

### 4.3 关键数字

#### 4.3.1 当前主线：L1 v4（2026-11-01）——**只有这组数能当现状**

数据：`l1-v4-relation-500`（499 集 / 8966 步）+ `l1-v4-cf`（451 反事实集 / 8092 步），
配对训练集 16184 样本。

**审计口径（2026-11-02 更正）**：此前写的「全部过 17/17」只对基础集成立，**反事实集
并不通过**。修正后的实测：

| 数据集 | 结果 |
| --- | --- |
| `l1-v4-relation-500` | **19/19 PASS**（含新增 `waypoint_side`、`window_semantics`） |
| `l1-v4-cf` | **12 PASS / 7 SKIP / 0 FAIL** |

反事实集复用原集的 RGB/oracle/轨迹、只重写 action 与 waypoint，所以
`teacher_waypoint` / `teacher_stop` / `forward_motion` / `transition_lag` /
`yaw_direction` / `class_share` 这几项**在 cf 上不适用**，现在判 SKIP；
`label_recompute` 仍执行且 `0/8092` 不一致。修复前的审计会把 cf 判成 5 项 FAIL。

| 项 | 数字 | 出处 |
| --- | --- | --- |
| 训练 | 20 轮 / 4864.9 s，best = **E17** | `l1-v4-cf/run-film-cf` |
| 离线 test（874 样本） **best** | acc **0.8478** / macro_f1 **0.7740** / bal_acc **0.7623**；recall `[0.833, 0.903, 0.676, 0.637]` | `eval-best-test` |
| 离线 test 末轮（E20） | acc 0.8467 / macro_f1 0.7633 / bal_acc 0.7405 | `eval-last-test` |
| 多数类先验 test | acc 0.6590 / macro_f1 0.1986 / bal_acc 0.2500 | — |
| 文本探针（801 决定性帧） | flip 0.9638 / acc_correct 0.8814 / acc_alt 0.9026 / **follow 0.8090** | `probe-best.json` |
| 闭环 keep（47 集，**2026-11-02 重跑**） | **SR 0.4043**（goal_reach）/ 碰撞 0.319 / false_forward 0.277 / p50 15.9 ms | `l1-v4-closed-goalreach-20261102/student-keep` |
| 闭环 shuffle（47 集，重跑） | **SR 0.000** / 碰撞 0.298 / false_forward 0.255 | `.../student-shuffle` |
| 盲前进（41 集完成 / 47 请求，重跑） | **SR 0.000** / 碰撞 0.463 / false_forward 0.463 | `.../forward` |
| 共同 41 集 | keep SR 0.4146 | 复算 |
| keep Wilson 95% CI | `[0.276, 0.547]`（47 集） | — |
| keep vs shuffle | McNemar 精确 **p < 1e-5**（keep 19 成功 vs shuffle 0） | — |

**失败分解（keep，47 集）**：成功 19、**撞柱 15（15/15 都在 ≤1.5 m 内）**、
未达 waypoint 13。

> **旧数字作废**：此前表里的 `SR 0.426 / 碰撞 0.277` 出自
> `l1-v4-closed-20261101`，其 `eval.json` 实际是 `goal_stop` 口径、值 `0.3404`；
> 0.426 是对旧 per_episode 手工复算的 goal_reach，**重跑不可复现**。
> 2026-11-02 按 `goal_reach` + `observation` 重跑后真实值是 **0.4043**。
> 两者差异来自 Gazebo 非逐位确定性（HANDOFF §八 坑 #10），不是口径问题。
> 论文/汇报只能用 `l1-v4-closed-goalreach-20261102` 的数字。

#### 4.3.2 历史基线（v1 / T6，**不要与新数字混报**）

| 项 | 数字 |
| --- | --- |
| T6 训练数据 | 599 集 / 23960 步；train 18520 / val 2800 / test 2640 |
| T6 best 离线 test | acc 0.9186 / macro_f1 0.8589 / balanced_acc 0.8497；recall `[1.000, 0.952, 0.700, 0.747]` |
| 多数类先验 test | acc 0.6492 / macro_f1 0.1968 / balanced_acc 0.2500 |
| **v1 复测（observation，唯一 baseline）** | **SR 0.90（18/20）** / collision 0.05 / false_forward 0.05；p50 10.26 ms / p95 13.10 ms；失败 seed 23/88 |
| 盲前进基线（v1 场景） | SR 0 / collision 1.00 / false_forward 1.00 |
| T7-R（dispatch） | SR 0.55（11/20）/ collision 0.45 / false_forward 0.30；p50 10.9 ms |
| T5b（dispatch 重训） | SR 0.60（12/20）/ collision 0.35；离线 test macro_f1 0.8287 |

> **T6 数字的出处**：`0.8589` 出自 `eval-best/metrics.jsonl`（ckpt_best 复评）；
> `run/metrics.jsonl` 末条是末轮 `ckpt.bin` 的 0.8042，抓错文件会差 5 个点。

> **v1 的 0.90 不是「比 v4 好」。** v1 是「飞到 goal_x」的走廊穿越任务；
> v1 任务里没有文本、没有选边、停止可观测。v4 SR 0.426 是在「必须按指令选边 +
> 停止点不可观测」的任务上测的。**两个任务的 SR 不可比，也不可混报。**

### 4.4 数据与产物路径

| 路径 | 内容 |
| --- | --- |
| `$DATA_ROOT/uav_rgb/t5-full-1155/` | T5 原始采集（599 集，`train/ val/ test/` + `audit.json`） |
| `$DATA_ROOT/uav_rgb/t6-full-1640/ds/dataset.uavr` | T6 打包数据（23960 步，1.38 GB） |
| `$DATA_ROOT/uav_rgb/t6-full-1640/run/ckpt_best.bin` | T6 best 权重（epoch 8） |
| `$DATA_ROOT/uav_rgb/t6-full-1640/eval-best/` | T6 best 离线复评 |
| `$DATA_ROOT/uav_rgb/t6-full-1640/eval-student/` | T7 旧版闭环（R01 前） |
| `$DATA_ROOT/uav_rgb/t6-full-1640/eval-forward/` | 盲前进基线 |
| `$DATA_ROOT/uav_rgb/t6-full-1640/eval-student-r01b/` | T7-R dispatch 复跑（SR 0.55，`--keep-work`） |
| `$DATA_ROOT/uav_rgb/t6-full-1640/ab-observation/`、`ab-dispatch/` | 窗口语义 A/B（49/57） |
| `$DATA_ROOT/uav_rgb/t6-full-1640/eval-student-v1-observation/` | 完整 observation v1 基线（SR 0.90，已完成） |
| `$DATA_ROOT/uav_rgb/l1-full-400-0803/{train,val,test}/` | L1 语言任务 400 集原始采集（每集 `episode_manifest.json` + `scene.json` + `rgb_*.png`） |
| `.../l1-full-400-0803/ds/` | L1 打包：`dataset.uavr`（16000 步）+ `instructions.jsonl`（400 行，带 `text_ids`） |
| `.../l1-full-400-0803/run/` | L1 image-only 基线权重（10 轮 / batch 64 / lr 1e-3 / seed 1234 / auto 权重 / drop_collision） |
| `.../l1-full-400-0803/eval-student-image-only/eval.json` | L1 v1 闭环（SR 0/34，**废数据，禁止引用**） |
| `$DATA_ROOT/uav_rgb/l1-stop-500-173307/` | L1 **v2** goal-stop 数据（498 集 / 10442 步）。**已被 v3/v4 取代，不再用于新实验** |
| `$DATA_ROOT/uav_rgb/l1-v3-relation-500/` | L1 **v3** 同色柱对 + 关系词数据（499 集），waypoint 朝走廊中心偏 |
| `$DATA_ROOT/uav_rgb/l1-v3-closed-20261101/` | v3 闭环（goal_stop 口径，**不可与 v4 混报**） |
| `$DATA_ROOT/uav_rgb/l1-v4-relation-500/` | **L1 v4 原始采集**（499 集 / 8966 步，train 404 / val 47 / test 48）。waypoint 在目标柱外侧 0.8 m |
| `.../l1-v4-relation-500/ds/` | v4 打包：`dataset.uavr` + `instructions.jsonl` |
| `$DATA_ROOT/uav_rgb/l1-v4-cf/` | **v4 反事实集**（451 集 / 8092 步）→ `ds/` 配对训练集 16184 样本 |
| `.../l1-v4-cf/run-film-cf/` | **当前主线权重**：`ckpt_best.bin`（E17，md5 `d91cd0f1…`）+ `ckpt.bin`（E20） |
| `.../l1-v4-cf/eval-best-test` / `eval-last-test` | v4 离线复评（874 样本） |
| `.../l1-v4-cf/probe-best.json` | v4 文本探针 |
| `$DATA_ROOT/uav_rgb/l1-v4-closed-20261101/` | **v4 闭环三组**：`student-keep` / `student-shuffle` / `forward`（goal_reach，已复算；未重跑） |

### 4.5 语言线（L1 / L2）现状 —— **接手先看这一节**

> 2026-11-01 21:52。一句话：**L1 走到 v4，指令学会了（shuffle 0.000 vs keep 0.426），
> 但「停得准」做不到（SR 0.426 < 50%）。两种修法待拍板。**

#### 数据版本沿革（四代，每一代都不可混包）

| 版本 | `task_version` | 作废/在用 | 原因 |
| --- | --- | --- | --- |
| v1 | `l1.v1` | **作废** | route-choice；teacher 自己 0/400 到 goal，闭环 SR 0/34 |
| v2 | `l1.v2` | 历史 | goal-stop + 两色柱：图像里目标色唯一 → 模型可完全忽略文本（换词只改 2e-4 概率） |
| v3 | `l1.v3` | 历史 | 同色柱对 + 关系词（文本必须读）。但 waypoint 朝**走廊中心**偏，274/274 个 test 场景都**不需要横移** → 测不到语言 |
| **v4** | `l1.v4` | **当前** | waypoint 改到目标柱**外侧** 0.8 m，横移变成 1.6 m；几何仍是 `uavrgb.geom.v5` |

几何版本 `uavrgb.geom.v5`（柱子 / 出生点 / split 都没变，只改目标点与关系词映射）。
`preprocess.py` 按 `task_version` 拦混包。

#### 当前结论：**指令学会了，停不会**

- **指令确实在起作用**：shuffle（把关系词 left↔right 反过来）下**没有任何一集**
  能靠近真 waypoint 2 m 以内（两个候选 waypoint 相距 3.2 m）。keep vs shuffle
  McNemar `p < 1e-5`。文本探针 `follow 0.8090`（v3 只有 0.6055）。
- **瓶颈是「停」不是「选边」**：13 次碰撞 **13/13 都发生在距 waypoint 1.5 m 以内**
  ——飞到了、停不下来，撞在目标柱上；另有 14 集停在 1.6–2.9 m 处。27/47 与
  「停得准」有关，与选边无关。
- **根因已定位**：`waypoint_for` 把停止点放在 `target.y ± 0.8`，柱子只在
  `y ∈ {0, ±1.6}`，于是停止点落在 **±2.4 m——已经出了柱阵**。实测 v4 test split：
  **40%（57/144）的 SAFE 帧视野里一根柱子都没有**（v3 是 0%），飞行途中也有
  13%（77/576）的 FORWARD 帧是空的。同一张空图在数据里既标 FORWARD 又标 SAFE。
  模型学成了「柱子消失 → 停」，于是提前停 1–2 m，或撞上还没让开的柱子。
- 另有一层：巷道网格是重复的，**单帧图像无法唯一确定沿 x 的位置**。追踪 seed
  900101：前 12 步与教师逐帧一致，第 12 步（距 waypoint 1.77 m，**视野里有 3 根柱子**）
  突然输出 SAFE，教师要求 YAW_R。所以不只是「看不见」，还有「看得见也分不出远近」。

完整证据：`docs/worklog/2026-11-01-2130-stop-point-unobservable.md`、
`docs/experiments/2026-11-01-l1-v4-text-training.md`。

#### 已拍板并落地的方案 A（2026-11-01）：判据改 goal-reach

用户选了 **A：去掉「必须停」的判定**。已实现：

| 文件 | 改动 |
| --- | --- |
| `sim/tools/eval_closed_loop.py` | `REACH_RADIUS = 1.5`；L1 分支 success = 无碰撞 + 末态 ≤1.5 m，**末动作不参与**；`success_criterion="goal_reach"` |
| `sim/tools/test_eval_summary.py` | L1 断言跟改 + 1.4/1.6 m 边界 + 「到位但撞柱」两条守卫 |
| `docs/EVAL_STANDARD.md §5.1` | 补 goal_reach 口径与 1.5 m 的几何来历；修正关系词定义 |

**1.5 m 是几何卡死的，不是调出来的**：必须 **< 两候选 waypoint 最小间距 3.23 m 的一半
（1.615）**，否则停在中间的无人机对 keep 和 shuffle 都算成功、反事实失效；
同时 **< 出生点到 waypoint 的最小距离 2.50 m**，否则原地不动也算成功。

**诚实修正**：早前估计方案 A 能到 57%，实际是 **43%（0.426）**。原因两条：
(a) 半径取 2.0 m 不安全（候选 waypoint 中点到两边各 1.62 m，会同时算成功）；
(b) 单纯去掉 `stopped` 一条**一集都多不出来**（16 个旧成功全部 `stopped=True`）。

#### 2026-11-02 更正：0.426 是复算值，不是评测文件

上面写的 **0.426 是按 goal_reach 规则对旧 `per_episode` 数据手工复算**得到的，
`eval.json` 本身**没有**按新判据重跑：

| 项 | 实际值 |
| --- | --- |
| `l1-v4-closed-20261101/student-keep/eval.json` 的 `success_criterion` | `"goal_stop"` |
| 同一文件 `success_rate` | **0.3404**（16/47） |
| goal_reach 复算 | 20/47 = 0.4255 |

**验收状态：v4 未过 50% 线，两种口径都不达标。**

**2026-11-02 已按 goal_reach + observation 重跑**（产物
`$DATA_ROOT/uav_rgb/l1-v4-closed-goalreach-20261102/`，三组共 138 min）：

| 组 | SR | 碰撞 | false_forward | 完成 |
| --- | --- | --- | --- | --- |
| student-keep | **0.4043**（19/47） | 0.319 | 0.277 | 47 |
| student-shuffle | **0.000** | 0.298 | 0.255 | 47 |
| forward | **0.000** | 0.463 | 0.463 | 41（3 集起飞超时） |

`eval.json` 的 `success_criterion` 均已是 `goal_reach`、`window_from=observation`。
**旧的 0.426 作废**——重跑不可复现（Gazebo 非逐位确定），
论文与汇报一律引用 0.4043。

#### L2 文本分支（**已训练，v4 主线用的就是它**）

自研 C++ 文本分支，零 PyTorch：

```
token ids [N,9] → emb[22,16]（CPU 查表）→ 按位置拼接 [N,144]（<pad> 槽位保持 0）
                                                 ↓ film 线性层
                                          γ/β [N,2×64]
                                                 ↓ 调制 c3 每个空间位置
image [N,3,120,160] → conv×3 → pool → c3 [N,64,20,15]
                                          y' = (1+γ)·y + β
                                                 ↓ flatten
                                          [N,19200] → fc → [N,4]
```

> 上图是初版的 mean-pool + concat，**已被 FiLM 取代**（见
> `experiments/2026-11-01-l2-film-finetune.md`）。现行实现（`model.cpp:164-199`）：
> token 查表后**按位置拼接**成 `[kTextLen × text_dim]`（**没有**均值池化，
> `<pad>` 槽位保持 0），过 `film` 线性层出逐通道 γ/β，
> 对 `c3` 的每个空间位置做 `(1+γ)·y + β`；`fc` 仍是 `[19200→4]`。
> **pre-FiLM 的 concat checkpoint 不兼容**：`fc.in=19216` 与当前 `19200` 不符，
> `load_checkpoint` 会按 shape 直接拒绝。

| 位置 | 作用 |
| --- | --- |
| `src/include/uavrgb/text.hpp` | `kTextLen=9` / `kTextVocab=22`，必须与 `sim/tools/instruction.py` 一致 |
| `src/model.cpp` | 前向查表 → 按位置拼接 → `film` 出逐通道 γ/β 调制 `c3`；**无** concat/mean-pool 分支，pre-FiLM ckpt 不兼容 |
| `src/dataset.cpp` | 读 `instructions.jsonl` 的 `text_ids`，按 `sample_start/count` 展开 |
| `src/checkpoint.cpp` | **v4**：`text_dim` 追加在 `best_macro_f1` 之后，v2/v3 偏移不变 |
| `train_main` / `trainer*` | `--text-dim N` 开文本分支；`--eval-only` 以 checkpoint 记录为准 |
| `infer*` | `--text id,id,...`（每个 id 必须 `< 22`）；无文本分支的 ckpt 仍忽略 `--text` |
| `tools/preprocess.py` | 侧车写 `text_ids`；`--exclude-safe-episodes` 见下面的警告 |

#### 下一步（**需要用户拍板，不要自行开工**）

方案 A 只把不可观测的判定从指标里拿掉了，**没有修根因**。SR 0.426 仍低于 50%
验收线。要真修只有两条路：

- **B：改几何让停止可观测**——把 waypoint 拉回柱阵内侧，或让最后一排柱子出现在
  停止点视野里。改 `landmarks.waypoint_for` + `scene_gen`，**必须 bump
  `SPLIT_SALT` 并重采重训**（约 1.5 h 采集 + 80 min 训练）。副作用：横移距离会变小，
  可能又回到 v3 的「不需要横移、测不到语言」。
- **C：改任务定义**——把「停」换成一个**可观测**的终止信号（例如「经过目标柱后
  再前进 N 步」），或允许序列输入（多帧）。这是改任务不是改几何，代价最大。

不做决定前**不要重跑三组闭环**（80 min，数值必然相同）。

#### v4 复现命令（已验证跑通）

```bash
# 采集
python3 sim/tools/collect_parallel.py --instances 5 --episodes-per-instance 100 \
  --steps 60 --hold 1.0 --base-seed 900000 --task l1 --spawn-mode random \
  --window-from observation --out $DATA_ROOT/uav_rgb/l1-v4-relation-500
python3 sim/tools/audit_dataset.py --dataset $DATA_ROOT/uav_rgb/l1-v4-relation-500

# 打包 + 反事实
python3 tools/preprocess.py --drop-collision --out .../l1-v4-relation-500/ds \
  .../l1-v4-relation-500/{train,val,test}/*/
python3 sim/tools/make_counterfactual.py --src .../l1-v4-relation-500 \
  --out .../l1-v4-cf --splits train,val
python3 tools/preprocess.py --drop-collision --out .../l1-v4-cf/ds \
  .../l1-v4-relation-500/{train,val}/*/ .../l1-v4-cf/{train,val}/*/

# 训练（81 min）
src/build/uavrgb_train --data .../l1-v4-cf/ds/dataset.uavr --text-dim 16 \
  --epochs 20 --batch 64 --lr 5e-4 --seed 1234 --class-weights auto \
  --drop-collision --out .../l1-v4-cf/run-film-cf

# 离线复评（注意：cf/ds 只有 train+val，test 要用基础数据集的 ds）
src/build/uavrgb_train --data .../l1-v4-relation-500/ds/dataset.uavr \
  --eval-only .../l1-v4-cf/run-film-cf/ckpt_best.bin --split test \
  --batch 128 --out .../l1-v4-cf/eval-best-test

# 文本探针
python3 sim/tools/probe_text_effect.py --src .../l1-v4-relation-500 \
  --ckpt .../run-film-cf/ckpt_best.bin --split test --out .../probe-best.json
```

闭环三组的完整命令见 `docs/experiments/2026-11-01-l1-v4-text-training.md`。

#### 闭环口径注意事项

- **必须 `--window-from observation`**（dispatch 语义的 SR 只有 0.55，未采纳）。
- 两个候选 waypoint 相距 **min 3.23 m**，这是「停在中间不可能同时算 keep 和 shuffle
  成功」的几何依据。
- **`shuffle SR > 0` 不能单独证明模型读懂了指令**：shuffle 只换 instruction、不换
  场景 waypoint，一个忽略文本的策略在 shuffle 下仍可能漂到原始 waypoint。语言证据
  必须来自 **keep − shuffle 的配对差异（McNemar）** 或文本探针的 `follow`，
  不能只引用某一组的 SR。
- `--text-mode blank/shuffle` 会先问 infer 的 `text_dim`，纯视觉 ckpt 上直接报错；
  `eval.json` 同时记 `text_dim` / `success_criterion` / `stop_radius`。
- **`stopped` 仍返回但不参与 goal_reach 判定**；`goal_stop` 与 `goal_reach` 的 SR
  不可混报。
- 分母是 `episodes_done`。盲前进那组有 3 集 infra 超时（`900032/900052/900101`），
  所以同时报「共同完成的 44 集」。
- **`flip` 只说明「文本会改变动作」**，不说明改对了；严格指标是 `follow`。
  两个数要一起给。

#### 未提交清单（2026-11-01 21:52 的工作树）

`git status --short` 实测（2026-11-01 21:52）：**39 改 + 5 删 + 6 暂存删除 + 32 未跟踪**。
全部未提交。

- 未跟踪里**必须一起提交**的代码：`src/include/uavrgb/text.hpp` 和
  `src/tests/test_text_branch.cpp` 已被 `src/CMakeLists.txt` 引用，**漏了构建直接失败**。
- 三个新的 Python 工具：`sim/tools/make_counterfactual.py`、`make_decisive.py`、
  `probe_text_effect.py`（+ `test_make_decisive.py`）。
- **未跟踪里绝大多数是文档**（`docs/experiments/` 10 个 + `docs/worklog/` 15 个），
  这些是语言线（v1→v4）的全部留痕，**没有其它备份**，一个都不能丢。
- 工作树里有**并发会话的改动**（`ui/`、部分 `src/`）。**禁止 `git add -A`**，
  逐文件确认归属再 `git add <path>`。
- `.ipynb_checkpoints/` 是垃圾目录，尚未加入 `.gitignore`，别误提交。
- **`ALVLN/uav_rgb` 仍是旧版**，本地的东西一个字节都没上 GitHub。

**坑（踩过的）**：

- `instructions.jsonl` 若没有 `text_ids`（L2 之前打包的），现在**当作没有侧车**处理：
  image-only 照跑，文本模型会在 trainer 报"没有侧车"。旧数据集不用重打包也能继续用。
- 指令超过 9 个 token 会**直接报错**（不截断）：`pass the red pillar on your right and stop`
  正好是 9 个，是 v5 的正式模板（`kTextLen` / `TEXT_LEN` 都已抬到 9）。
- `--exclude-safe-episodes` 不是"去噪"，是**把 SAFE 类整个删掉**（16000 → 12800，SAFE 归零），
  对照组会变成 3 类模型，结论只能读成"没有 SAFE 类会怎样"。
- **`audit_dataset.py` 的 17/17 PASS 认证不了图像**：17 条检查只读
  `actions/oracle/scene` 的 jsonl，**从不打开 PNG**；`label_recompute` 是同一个
  teacher、同一个 `pose_before` 重算，对未改动数据**数学上恒等**。所以它只证明
  「原始采集内部自洽」，既不覆盖实际训练的配对包（反事实集的 oracle/collision/
  validity 是从原集复制的，按原集口径审必然 FAIL），也发现不了「空帧同图标
  FORWARD/SAFE」这类问题。要验图像必须另外开图看。
- **反事实集 `l1-v4-cf/ds` 只有 train+val**，没有 test。`--eval-only --split test`
  必须对基础数据集的 `ds` 跑，否则报 `eval_only: test split 为空`。
- **训练末尾那次 test 用的是末轮模型，不是 `ckpt_best`**（`src/trainer.cpp:284-295`），
  直接读 `metrics.jsonl` 最后一条会偏低。评 best 必须用 `--eval-only`。
- **`carry_resume_best` 跨数据集冻结 best**（`src/trainer.cpp:40-72`），
  resume 还会静默继承旧 `class_weights`（`src/class_weights.cpp:134-142`）。
  **v4 没有用 `--resume`，不受影响**；以后 resume 前先修这两处。
- **EVAL_STANDARD 的关系词定义曾经和代码相反**：v4 的 `left` 是**经过时**目标柱在
  机身左侧，不是「出生时刻」。v4 的 waypoint 在柱子外侧，目标柱出生时位于关系词的
  **另一侧**。文档已按 500/500 集实测改正，读旧版 EVAL_STANDARD 会理解反。

## 五、工作纪律（必须遵守）

### 留痕

- 每个任务结束写 `docs/worklog/YYYY-MM-DD-HHMM-<slug>.md`，必含四块：
  **做了什么 / 没做什么（每条给原因）/ 验证到哪一步（真实输出）/ 风险与遗留**。
- 实验数字（训练、基准、smoke）另落 `docs/experiments/<实验名>.md`，不塞进 worklog。
- 同一任务继续迭代**不新建文件**，在原文追加 `## 更新 YYYY-MM-DD HH:MM` 小节；
  结论变了要改开头那行「结论」，不要让文档自相矛盾。

### Git

- **只有用户明确要求提交或推送时才做 Git 写操作。** 修 bug、审查、「继续」都不等于授权。
- 一个主题一个提交；精确 `git add <path>`；**不 `git add -A`**。
- 不提交密钥/Token/私钥/`.env`/缓存/日志/临时脚本/生成产物。
- 不改已推送历史，**禁 `git push --force` / `--force-with-lease`**。
- 远端是 `git@github.com:xu967/ALVLN.git` 的 **`uav_rgb/` 子目录**。推送走
  `../ALVLN`（完整克隆，**不要删**，重克隆约 12 分钟）：rsync 替换 `uav_rgb/`
  → `git add uav_rgb` → commit → push。不要用 partial clone（会卡在 `git-upload-pack`）。
  推送后核对 `git status --short --branch` 与 `git ls-remote --heads origin`。

### 其他

- 简体中文输出；代码标识符/路径/命令/日志保持原样。
- **不删用户没让删的东西**；发现可疑状态先查清楚再动手。
- **临时脚本用完即删**（`/tmp/*.py`、临时进程），只有正式单测才留在
  `src/tests/` 或对应模块下。
- 不静默降级：跳过校验、关掉检查、绕过权限必须写进文档并说明原因。
- **不写「应该能过」「理论上没问题」**。没跑就写「未跑（原因）」。

## 六、验收口径（阶段目标，**已拍板**）

> 2026-10-30 用户拍板：对标对象选 **(a) 自建基线阶梯**。
> 不是 AvoidBench 对外对标，也不是换数据集。

### 离线

必须**同时**报四件套：`action_accuracy` + `macro_f1` + `balanced_accuracy` +
**每类 recall**。定义按 `EVAL_STANDARD.md §1.1`（scikit-learn 宏指标）。

**两条违规**（写了就等于造假）：

1. 只报准确率。
2. 按「有样本的类」算宏 F1（例如 3 类 0.5140）——这个数只能当参考，不能写进结论。

### 闭环

按 `EVAL_STANDARD.md §1.2`（AvoidBench ICRA 2023 口径）：SR / PO / EO，
外加碰撞率、`false_forward_rate`、动作分布、推理 p50/p95。
**PO / EO / AGV / MP 仍未实现**（缺 `d_min` A*），只有 SR 是完整的。

**L1 的 SR 有两种口径，`eval.json` 的 `success_criterion` 区分，不可混报**：

| 口径 | 判据 | 用于 |
| --- | --- | --- |
| `goal_reach`（**2026-11-01 起**） | 无碰撞 + 末态距 waypoint ≤ **1.5 m** | L1 v4 |
| `goal_stop`（历史） | 无碰撞 + 末态 ≤ `stop_radius+0.2`（0.7 m）+ 末动作 `SAFE_NEUTRAL` | L1 v2/v3 |

**v1 走廊任务的 SR（0.90）和 L1 的 SR（0.426）不可比**，任务定义不同。

### 基线阶梯（对标对象）

| 基线 | 口径 | 怎么跑 |
| --- | --- | --- |
| 盲前进 policy | 闭环 SR / 碰撞率 / false_forward | `eval_closed_loop.py --policy forward` 已实现（T7 冒烟时加的） |
| 多数类先验 | 离线 `macro_f1` / `balanced_accuracy` | `tools/baseline_priors.py` 已实现（T4 起在用）：读 `.uavr` 标签列，多数类只从 train 选，报同款宏指标；T2 数据 test `macro_f1=0.1543` |
| 我们的模型 | 两个都要 | 同场景同口径 |

**建议的阶段验收线**（用户可改，改这里即可）：

- 离线 `macro_f1` 显著高于多数类先验；
- 闭环 SR ≥ 50%（≥20 个 test 场景）且 `false_forward_rate` 低于盲前进。

### 汇报规则

每次汇报对标结果，必须同时写清三件套：**「我们的数字」/「对标数字」/
「对标数字的出处（论文/榜单/仓库 + 日期）」**。三者缺一不算完成。

## 七、任务路线

> **T1–T7 已于 2026-10-30 全部完成，T7 同版本复跑也已完成。**
> 当前唯一在做的是 **L1 语言任务**（§4.5），卡在「停得准」上，等用户拍板 B/C。
> 下面 T1–T7 各节是**历史记录与约束**，不再作为待办，但里面的几何契约
> （改几何要联动哪些参数、必须 bump SPLIT_SALT）仍然有效。
> **状态与验收以 `docs/TASKS.md` 为准，本节只讲理由与约束。**
>
> **新 AI 不要自行开工的项目**：T8（鲁棒性/失败模式）、D5（device 层）、
> ECPT、公开数据预训练——都未立项，等用户拍板。

### T1 = D2 车道几何决策（P1）

> **2026-10-30 已完成并复核**：几何 v3（车道缝 1.2 m、出生点 x=-0.5）+ A* 全局教师；
> 05:xx 用当前 teacher 重采同 20 个 test seed，**20/20、0 碰撞、label_recompute
> 0/800**（旧 run 是中间版本教师，已不作为验收依据）。
> （普通集 `planner=astar`，safe 集 `planner=plan_action`）。24 个 test 场景的
> 穿越率、根因与改动见 `docs/experiments/2026-10-30-t1-lane-geometry.md`。

**问题**：车道缝 0.902 m，机体外接直径 0.711 m，单侧余量仅 9.5 cm。

**方向已定，别重复验证**：`docs/experiments/2026-10-29-teacher-traverse-ablations.md`
第 10 轮已证伪「收紧 planner 余量」路线（clearance 0.36→0.40 时穿越率 5/8→2/8，
max|y| 从 0.69 涨到 1.86）。`teacher.py:112-119` 注释同结论：**真正的解法是放宽车道间距**。
缺的只是定数字 + 改后重测。

**参数位置**：`sim/tools/scene_gen.py:41-54`
（`LANE_SEP=1.302`、`MIN_GAP=0.9`、`PILLAR_COUNT_RANGE=(8,9)`、`GATE_Y_ABS_MAX=0.25`、
`PILLAR_Y_RANGE=(-1.5,1.5)`、`Y_RANGE=(-3.0,3.0)`）。

**改之前必须重推这些约束（不能只改一个数）**：

- 三条车道装进 ±1.5 m 要求 `2×LANE_SEP + 中心摆动 ≤ 3.0`。LANE_SEP 加到 1.7 就装不下
  → 要么加宽密布区（`PILLAR_Y_RANGE`、`Y_RANGE` 联动），要么改两车道。
- 柱数上限 9 的推导（任意两柱 Chebyshev ≥ LANE_SEP）随间距变，`PILLAR_COUNT_RANGE` 重推。
- `GATE_Y_ABS_MAX=0.25`（门柱必须落在 teacher 正前方 ±10.9° 视锥内）与
  `LANE_CENTER_ABS_MAX` 契约都要复查。
- **改几何必须 bump `SPLIT_SALT`**（`split_manifest.py:26`，v2→v3）并开新数据版本，
  否则新旧几何会静默混进同一 split。已有数据不原地重标。
- 同步更新 `scene_gen --self-check` 断言与 `test_split_manifest.py`。

**建议目标**：缝 ≥ 1.2 m（单侧余量 ≥ 0.24 m，给 160×120 的视觉定位留误差带）。
具体数字按下面的实测结果定。

**验收**：改后跑 `plan_action` 教师策略 ≥20 个 test 场景，穿越率 ≥90%
（成功 = 无碰撞且 `max_x ≥ goal_x`，`goal_x` = 最右柱 x + `--goal-margin` 默认 1.0）。
教师不走 `eval_closed_loop.py`（那是学生专用），用采集器跑：

```bash
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 5 --steps 40 --hold 1.0 \
  --seeds 23,25,36,45,48,49,50,54,57,84,88,99,114,116,129,137,147,171,173,180 \
  --split test --policy teacher --out /tmp/t1-teacher
# 然后跑 sim/tools/traverse_rate.py --min-rate 0.9（safe 出生集自动剔除）
```

这就是 `2026-10-29-teacher-traverse-ablations.md` 第 8/10 轮用的同一套口径
（那里是 6/8 过 x=4.0、5/8 过 x=5.0）。决策记入 `docs/experiments/`（数字 + 理由）；
相关自测通过。

### T2 = D3 SAFE_NEUTRAL 数据 / 四类定义（P1，核心）

> **2026-10-30 已完成**：safe 出生（`scene_gen --spawn-mode safe`，机身+桨盘碰撞检查）
> + 52 集实采：2080 步 0 碰撞，audit 11 项全 PASS，SAFE 21.2%，每 split 四类都有。
> `transition_lag` 阈值按 164 个过渡实测改为 20%。见
> `docs/experiments/2026-10-30-t2-safe-neutral.md`。

**根因**：`plan_action` 只有一条返回 SAFE 的分支（`teacher.py:156-158`：35 条射线全堵
**且** base 方向也被堵）。而当前几何保证出生区 `x<1.0` 无遮挡、全对缝隙 ≥0.9 m →
「全堵」几乎不发生 → SAFE 占比 ≈0。

**出生位姿是固定的**：`sim/worlds/uav_rgb.sdf:140` `<pose>0 0 0.15 0 0 0</pose>`，
随机性全部来自 `scene_gen` 生成的场景，`collect.py` 自己不做起点随机化。

**设计空间（方案由你定并验证）**：给采集加「初始位姿随机化 / 定向摆放」，让一部分集的
起点或中途位姿正对近距离柱群（lookahead 1.6 m 内全方向堵）→ `plan_action` 天然输出 SAFE；
或在 `scene_gen` 增「近距障碍开局」场景族。

**三条硬约束**：

1. 标签必须由 `plan_action` 在该位姿**真算出来**——`audit_dataset.py` 的
   `label_recompute` 会逐条复算，造假过不了闸门。
2. SAFE 步必须**无碰撞**（碰撞步的专家标签不可信）。
3. `collect.py` 的 `actions.jsonl` 是**字段白名单投影**，新字段必须显式加进去才落盘。

**顺带解决**：小数据集 `class_share` 该报「样本不足」独立状态还是硬 FAIL。

**验收**：实采 ≥50 集过 `audit_dataset.py` **全部**闸门（含 `class_share` 每类 ≥5%、
`label_recompute` 0 不一致）；类别分布与 SAFE 位姿设计记入 `docs/experiments/`；
7 个 Python 自测 + 新用例全过。

> 为什么是 50 集而不是 T3 的 20 集：`class_share` 要求每类 ≥5%，若 SAFE 只占 10%，
> 20 集 × 80 步 = 1600 步里只有 ~160 个 SAFE 步，压不住统计噪声；50 集才够用
> 这个闸门做「类别分布是否合理」的判定。T3 的 20 集**不**要求过 `class_share`，
> 它只验端到端链路通（四类 recall >0）。两个数字服务于两个不同的目的。

### T3 = 小规模端到端放行闸门（P1）

> **2026-10-30 已完成，T5 放行**：T2 的 52 集打包成 2080 步 → 10 轮训练 →
> eval-only：test recall `[1.000 0.504 0.278 0.941]` 全 >0，macro_f1 0.5859 /
> bal_acc 0.6807（多数类先验 0.1543 / 0.2500）。见 `docs/worklog/2026-10-30-0500-t3-rollgate.md`。

新几何 + 新采集方案下采 ~20 集（`collect_parallel.py` 2 实例）→ 过闸门 →
`preprocess.py` 打包 `.uavr` → 10 轮短训练 → **四类 recall 全部 >0**。

**这一步不过，不放行 T5。** 它的存在就是为了避免「采完 8 万帧才发现标签坏」。

**与 T2 的分工**：T2 用小批量（≥50 集）验**标签分布是否合理**（要不要过
`class_share`）；T3 验**整条链路能不能跑出可用模型**（只看四类 recall 是否 >0，
不卡 `class_share` 的比例）。T3 用的采参数应与 T5 完全一致，否则放行的不是同一个东西。

### T4 = 训练器加 eval-only 模式（P1，小改动，前置收益高）

> **2026-10-30 已完成**：`uavrgb_train --eval-only <ckpt> --split val|test|all`；
> CTest 20/20。airsim10k `ckpt_best.bin` test `macro_f1=0.5173`（末轮 0.3855），
> 见 `docs/experiments/2026-10-30-t4-eval-only.md`。

**现状**：test 只在训练末尾用**末轮权重**跑一次；`ckpt_best.bin` 从未被评过。

**最小实现**：`src/train_main.cpp` 加 `--eval-only <ckpt.bin>`（配 `--data`/`--out`/
`--batch`），复用 `src/trainer_eval.cpp` 现有评估路径，不建优化器、不更新权重。

**输出契约（照这个做，别自创）**：复用训练时 validation 事件的字段（`src/include/uavrgb/event.hpp`
的 `MetricsEvent`），向 `<out>/metrics.jsonl` 追加一条 `split=val|test` 的评估记录，
stdout 打同样内容；`--split` 选评哪个 split（不给就 val+test 都评）。
**字段必须与训练期一致**，否则 `ui/monitor.py` 读不出来。

**验收**：对 `airsim10k-run/ckpt_best.bin` 跑出 val/test 四件套并与训练时
`metrics.jsonl` 里末轮的 val 记录对账（同权重同数据，数字应可复现）；CTest 补 1 条
（可仿 `src/tests/test_overfit.cpp` 的 eval 辅助）；L2 全绿。

> **为什么排在 T1–T3 之后**：它是独立的小改动，不阻塞任何数据侧工作，也不影响
> T1–T3 的结论。但它必须在 T6 之前完成——T6 要用它评 `ckpt_best.bin`。
> 若 T1/T2 卡住等结果，可以插队先做 T4。

### T5 = 大规模采集（P0）

**步数用 40，不要用 80**（2026-10-30 实测）：v3 几何 + A* 每步约 0.27 m，T2 的
41 个 A* 集在 29–38 步过线（中位 30）；80 步会让约一半帧在柱区外原地转向
（T2 数据里线外 22.8% 是 YAW）。`collect*.py` 的默认已改为 40。
吞吐实测 1.95 步/s（4 实例、40 步/集）→ 500 集 ≈ 2.8 h、1000 集 ≈ 5.7 h。

**顺序**：先 50 集小批过闸门并**实测吞吐**（集/小时、墙钟、类别分布），再按目标窗口
（建议 overnight ≤12 h）反推实例数与集数。目标量级 500–1000 集 × 40 步 ≈ 2–4 万帧。

**排期参考（2026-10-30 T2 实测，以此为准）**：**4 实例 = 1.95 步/s**；
**8 实例会把 RTF 打到 0.1、起飞超时**（软件 GL + 12 核 cgroup 被其它会话共享）。
所以**并行上限按 4 个实例规划**。T2 的 52 集实测 1068 s（175 集/h）；本轮 pilot
在其它会话抢 CPU 时约 150 集/h。按 150–175 集/h：600 集 ≈ 3.4–4.0 h、
1000 集 ≈ 5.7–6.7 h（含每集约 40 s 的起 sim + 起飞固定开销）。

**验收**：全量过 `audit_dataset.py` 闸门；采集统计（类别分布、`sim_dt`、墙钟、实例数）
记入 `docs/experiments/`。

### T6 = 训练 + 离线验收（P0）

打包（默认 `split_policy="scene_id"`，几何 gid 保证跨 split 隔离）→ 训练
（`--class-weights auto`，盯 dev_ce 防过拟合，信赖 `ckpt_best.bin`）→
用 T4 的 eval-only 模式评**best 权重**。

**排期参考（旧实测，需复核）**：训练吞吐 **59–62 样本/秒**，且对 batch 大小几乎不敏感
（瓶颈是每层固定的 `cudaMalloc` + 同步 + pageable 拷贝：batch 256 前向 991.6 ms 里
host 清零 599 ms、拷贝 220 ms、**kernel 只有 167 ms**）。按此估算 8 万帧 × 10 轮
≈ 3.7 h。真解是 device-resident 引擎（预计 3×），列为 P3 池的 D5，**不阻塞主线**。

**验收**：四件套齐全 + 与多数类先验的对比表成文。

### T7 = 闭环验收 vs 基线阶梯（P0）

按 §六 的表跑三个对象：盲前进 / 多数类先验 / 我们的模型，同场景同口径。

**验收**：三件套汇报齐；全量数字落 `docs/experiments/` 总结 md；TASKS 全部 done 并链 worklog。

> 当前状态：同版本复跑已完成。v1 observation **SR 0.90（18/20）**、dispatch
> T7-R **0.55（11/20）**；0.90 只适用于 observation 语义，两者禁止混报（§4.1）。

### T8 候选（未立项，等用户拍板）

方向是**模型鲁棒性 / 失败模式**：T7 旧版已暴露 seed 45「一直 SAFE 悬停」和
seed 99「FORWARD 撞柱」；复跑若出现新的碰撞集，先做逐帧可视化与动作序列分析，
再决定是补数据（近距障碍/贴障穿行）、改教师，还是加推理侧安全约束。
**验收线未定，不要直接开工**；用户拍板后先写进 `docs/TASKS.md`。

### P3 池（不阻塞主线）

D4 depth/contact 停更门禁**已完成（2026-10-30）**；D5 device 层接入训练路径、G1 CUDA 多 block parity、
G2 resume 签名校验、G3 fsync/64 位溢出/目录锁、G4 `-Wall -Wextra` 全开、
G5 并发 TOCTOU 加锁、G6 生效配置落盘。定义见 `docs/TASKS.md`。

**另有一个研究线待办（不在表上，按需启动）**：`expert_action`（深度反应式专家）的
三个设计缺陷——单个近处障碍同时污染左/中/右三带、转向后一路飞走缺「贴障穿行」困难样本、
三分带太粗没用上机体宽度。改造方案 R1 列级深度剖面 / R2 机体宽度分级安全距离 /
R3 最小转向评分，约 +110–130 行。**故意排在数据采集之后**：否则同一批数据里专家规则
变过，无法对比改造效果。

## 八、已知陷阱（照做，别重新发现）

### 工具链

1. **`tools/check.sh` 建在 `build/check`，不保证 `src/build` 是新的。**
   跑真训练前单独 `cmake --build src/build`。曾因此整轮训练用的是旧二进制、没产出
   `ckpt_best.bin`。
2. **sm_75 必须在 `project()` 之前 FORCE**，否则 CMake 3.28 用 nvcc 默认值 52。
3. **`pkill -f 'gz sim'` 会杀掉你自己**（命令行自匹配）。用 `setsid` + `kill -- -PGID`；
   残留 `ruby` 用 `ps -eo pid,comm --no-headers | awk '$2=="ruby"{print $1}'`。
   **SIGKILL 采集进程还会留下 `gz-transport-topic` 订阅者（ppid=1）**：实测 30+
   个残留把 4 实例 RTF 压到 0.1（10 s/步），清掉后 1.4 s/步。`collect_parallel`
   现在启动/收尾会 sweep 掉 `/uav/`、`/model/x3` 话题的孤儿订阅者。
4. **纯 headless 相机渲染会 segfault**（无 NVIDIA EGL）。必须 `DISPLAY=:20`，且只用 `-s`。
5. **多实例必须用 `GZ_PARTITION` 隔离**，否则 server 互相污染话题。
6. **同一时间只能有一个采集任务在跑**。两任务各 6 实例时 RTF 掉到 0.15–0.28，
   导致起飞超时和 `stuck` 误报碰撞。

### 数据

7. **`collect.py` 的 `actions.jsonl` 是字段白名单投影**——新增字段不改写盘处不会落盘
   （第一次改 `pose_before` 就漏了）。
8. **`--hold` 是仿真时间秒**（D1 之后）。旧文档说「`--hold` 是墙钟秒」的一律是过时说法。
   注意区分：**动作窗口**按仿真时间计，但**吞吐/耗时**永远按墙钟算（RTF≈0.25 时
   1 s 仿真时间要 4 s 墙钟）。两者不是一回事，不要混着说。
9. **`label_recompute` 只覆盖 `teacher=plan_action`**。`expert_action` / `forward` /
   `student` 的集永远是 SKIP——拿柱子真值去复算深度专家等于换教师，比出来没意义。
10. **同 seed 两次采集动作序列可能分叉**（Gazebo 不是逐位确定的）。依赖「同 seed 同轨迹」
    的下游实验不能假设位级一致。`sim_dt` 这条口径是满足的（cv 0.0%）。
11. **碰撞步的专家标签不可信**——专家撞墙时给的动作本身就是错的。用 `--drop-collision`
    过滤或降权。
12. **场景族只有 1 个**（`family_holdout=false`）。几何去重 ≠ 泛化。

### 模型/数据格式

13. 数据集是 **NCHW**，`SimObservation` 是 **HWC**，由适配层转换。漏了会得到
    「形状对但画面错乱」的静默错误。
14. `.uavr` 的 C++ reader **硬性要求 `split_policy == "scene_id"`**（`src/dataset.cpp:257`）。
    不要为了绕开它去弱化这个 guard。
15. 深度图底边照到地面（~0.13 m），专家若整列取最小深度会永远判「前方被挡」，
    只取中间水平带 `[0.4h, 0.6h)`。

### 安全

16. **Kaggle token 在 `~/.kaggle/access_token`（mode 600），需要用户吊销重发。**
    不要写进文档、不要 echo、不要提交。公开数据已下载完，不再需要它。
17. **采集中不要跑 CTest/构建/其它 GPU 任务。** 2026-10-30 T1 重跑时并发跑
    `ctest`（含 `test_overfit` 200 步 CUDA），8 实例里 21/24 集起飞超时，
    z 只到 1.4–1.8 m（目标 2.1）；停掉并发任务后同一命令 24/24 起飞。
    采集窗口是独占的，不只是「同一时间一个采集任务」。
18. **本会话的 cgroup CPU 配额是 12 核，不是 `nproc` 的 88。** 用
    `cat /sys/fs/cgroup/cpu.max` 看（1200000/100000=12）。8 个软件 GL sim
    在 12 核里已接近上限，叠加别的会话就会把 RTF 打到 0.1、80 步一集要
    10+ 分钟。实际用 **4–6 实例**；8 只在机器空闲时用。起飞 timeout 已从
    30s 提到 120s（RTF 0.2 时爬升 1.95 m 要 ~60s 墙钟）。
19. **`eval_closed_loop.py --out` 不能先 `mkdir` 再往里写日志**：脚本拒绝非空目录，
    日志要放 `"$OUT.log"` 这种同级路径。`collect_parallel.py` 同理。
20. **长训练用 `nohup ... &` 或后台任务跑满**：前台命令被中断会留下半个 run
    （results.csv 停在第 8 轮、ckpt_best 不可信）。中断后重跑要换新目录或
    `--resume`，不要往半成品目录追加。

## 九、悬而未决（遇到先停下来问用户）

| 项 | 现状 | 卡在哪 |
| --- | --- | --- |
| **L1 闭环 SR 0.426 < 50%** | 方案 A 已落地（goal_reach），根因未修 | **等用户拍板 B（改几何，重采重训）还是 C（改任务定义）**，见 §4.5 |
| **13 次碰撞全在 1.5 m 内** | 已定位为「停止点不可观测 + 单帧无法定位 x」 | 同上，B/C 二选一 |
| `audit_dataset.py` 不验图像 | 17 条检查只读 jsonl，`label_recompute` 是自比 | 要不要加 PNG 抽样校验，未立项 |
| `carry_resume_best` / resume 继承 class_weights | 代码缺陷已确认，v4 未触发 | 下次用 `--resume` 前必须先修 |
| `transition_lag` 阈值 | **已定 20%**（2026-10-30：52 集/164 过渡中位 44%，50% 误杀） | 已解决 |
| AirSim 离散化阈值 | 10 deg/s vs 我们 `YAW=0.6 rad/s=34.4 deg/s`，差 3 倍多 | 两者是否同一动作口径无依据 |
| 公开数据是否当预训练 | 能训、训得出来，但不知道有没有用 | 郊区街道 vs 柱阵走廊，视觉域差得远；要证明有用得预训练+微调+闭环对比 |
| 小样本 `class_share` | **52 集/2080 步已 PASS**（每类 ≥5%）；<20 步的小样本仍会 FAIL | 未给「样本不足」独立状态，小样本继续按 FAIL 处理 |
| 硬件合同 | 荔枝派型号、ESP32 UART 帧协议 | **必须等用户提供** |

## 十、已冻结的合同（改之前必须想清楚）

### 10.1 学生输入 `src/include/uavrgb/observation.hpp`

```cpp
inline constexpr int kObsW = 160, kObsH = 120, kObsC = 3;
struct SimObservation {
  std::uint8_t rgb[160*120*3];  // 只有 RGB，HWC，未归一化
  std::int64_t timestamp_ns;
  std::int64_t episode_id;
};
```

**分辨率决策（偏离 spec，需最终验证）**：spec 原写 128×128，但 320×240→128×128 是把 4:3
横向压扁到 1:1（水平角分辨率 0.469°/px，10 m 外 0.4 m 立柱仅 4.9 px）。现用 **160×120**：
精确一半、保持 4:3、像素只多 17%、角分辨率 0.375°/px（同条件 6.1 px）。
**改它必须同步改 `tools/preprocess.py --size` 和 `src/model.cpp`。**
最终尺寸应由「128 / 160 / 224 同数据 A/B + 闭环 SR」实测决定。

**细障碍（2 cm 线缆/树枝）在任何分辨率下都是亚像素**（160×120 下 5 m 外 0.61 px），
是单目 RGB 的物理极限。

### 10.2 Episode 字段 `src/include/uavrgb/episode.hpp`

`EpisodeStep{timestamp_ns, action_expert, action_executed, collision}`；
`EpisodeMeta{episode_id, scene_id, seed, version}`。

**划分铁律**：train/val/test 按场景/几何组隔离，**禁止同一场景随机拆帧**。
实测强制拒绝（exit=1 且不留产物）。

### 10.3 事件/指标 `src/include/uavrgb/event.hpp`

`MetricsEvent`（train_step / validation / status / error）+
`ResultsRow{epoch, train_ce, dev_ce, dev_accuracy}`。
**训练器写出的 `results.csv` / `metrics.jsonl` 字段必须与此一致**，否则 `ui/monitor.py` 读不出来。

### 10.4 数据集格式 `tools/DATASET_FORMAT.md`

UAVR 版本 1，小端：48 B 头（magic `UAVR` / version / header_size / N / H / W / C / dtype /
offsets）→ N×C×H×W uint8 NCHW → N 个 uint8 标签 → 紧凑 JSON meta
（含 `split_policy=scene_id`、逐 episode 的 `sample_start`）。

### 10.5 代码规则

- C++ 单文件 ≤280 行，CUDA 单文件 ≤360 行（超了拆文件，不许压注释）。
- 每个新数值算子必须有：CPU FP32 参考 + FP64 有限差分 gradcheck + CPU/CUDA 对拍。
- **不许假结果**：FOUND ≠ VERIFIED，必须真跑真报数。
- 不许把「至少一条不是全 F」这种能糊弄过去的指标当验收标准——要用**分布/比例**类指标。

## 十一、参考实现（浙大高飞团队 9 个仓库）

参考代码在 `$DATA_ROOT/archive/idea/repos/`（原 `code/idea/`，2026-10-30 移出以释放根盘）。**只读思想，不要抄代码**：

| 仓库 | 许可证 | 结论 |
| --- | --- | --- |
| GCOPTER | **MIT** | 唯一可抄代码的，但我们不需要 |
| ego-planner / EGO-Planner-v2 / Elastic-Tracker / Fast-Racing / Implicit-SDF-Planner / Swarm-Formation / ego-planner-swarm | GPL-3.0 | **传染性**，抄了项目就得 GPL |
| Fast-Drone-250 | **无 LICENSE**，README 明写禁止商用转载 | 风险最高，只看思想 |

**三条结论**：

1. 这个 `ego-planner` 快照其实是 **ESDF-free** 的，别被名字误导。
2. 参考仓库的工程实践比我们落后（9/9 ROS1 catkin、0/9 CI、0/9 现代 CMake），没有可抄的。
3. 值得借鉴的只有：列级深度剖面找可通行区间、机体宽度/间隙判据、常驻订阅 + sim-time
   配对、安全降级优先级。

## 十二、常用命令速查

```bash
cd /root/claude-code-backup-20261028/code/uav_rgb && source env.sh

# L2 全量验证（每个任务完成必跑）。通过判据：EXIT=0 且末行是
#   PASS: 所选检查完成；模拟采集、真实训练和学生闭环须独立验收。
# 它做的事：doctor --require-gpu → ctest → 合成 episode 打包格式契约 →
#   几何 split 门禁负例 → teacher/scene_gen/gz_io/airsim_to_uavr 自检 →
#   逐个跑 sim/tools/test_*.py → ui 两个测试。
# 它在 build/check 里构建，不是 src/build。
bash tools/check.sh
# 无卡时：bash tools/check.sh --python-only（会打印跳过 C++/CUDA 的说明）

# 单独构建 + 全量 CTest（21 项）
cmake -S src -B src/build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build src/build
ctest --test-dir src/build --output-on-failure

# Python 自测（不用 pytest，逐个直接跑；用 UAV_RGB_PYTHON 而不是裸 python3）
for t in sim/tools/test_*.py; do "$UAV_RGB_PYTHON" "$t" || echo "FAIL $t"; done

# 专家自测 / 场景自检
python3 sim/tools/teacher.py
python3 sim/tools/scene_gen.py --self-check

# 并行采集（必须 DISPLAY=:20）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 1 --steps 40 --hold 1.0 --base-seed 0 --out /tmp/dataset

# 打包 + 训练
python3 tools/preprocess.py --out /tmp/ds /tmp/dataset/train/*/
src/build/uavrgb_train --data /tmp/ds/dataset.uavr --epochs 10 --batch 128 --lr 1e-3 \
  --class-weights auto --out /tmp/run1

# 数据验收闸门（FAIL 即非零退出）
python3 sim/tools/audit_dataset.py --dataset /tmp/dataset

# 闭环评测（学生；T7 的 20 个冻结 seed）
python3 sim/tools/eval_closed_loop.py --policy student --ckpt <ckpt.bin> \
  --steps 80 --hold 1.0 --split test \
  --seeds 23,25,36,45,48,49,50,54,57,84,88,99,114,116,129,137,147,171,173,180 \
  --out /tmp/eval1
# 盲前进基线：把 --policy student --ckpt ... 换成 --policy forward
```

**上面示例里的 `/tmp/...` 都是临时验证目录，产物可以随时丢。** 正式数据集/权重/运行
产物一律放 `/root/rivermind-data/uav_rgb/`（`$DATA_ROOT/uav_rgb`），不要放进 `code/`。

## 十三、参数与路径速查

### 场景几何（改 T1 前必看）

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `PILLAR_SIZE` | 0.4 | 柱子是 0.4×0.4×3.0 m 的方柱（不是圆） |
| `LANE_SEP` | 1.6 | **车道中心距**，不是缝宽 |
| 横向车道缝 | **1.2** | `LANE_SEP − PILLAR_SIZE = 1.6 − 0.4`（T1，2026-10-30） |
| `MIN_GAP` | 0.9 | 任意两柱至少一条缝 ≥ 0.9 m 的采样契约；同车道纵向缝仍按 0.9125 m 校验 |
| 机体外接直径 | 0.711 | `(hypot(0.13,0.22) + 0.1) × 2`；横向缝单侧余量 **0.2445 m**。**不是碰撞盒**（盒子是 0.47 m 见方） |
| `PILLAR_Y_RANGE` | ±1.7 | 三条车道 ±1.6 + 中心摆动 ±0.1 |
| `PILLAR_COUNT_RANGE` | (8, 9) | 上限 9 的推导：3.5×3.0 m 区 + 任意两柱 Chebyshev ≥ 1.3 → 最多 3×3 |
| `GATE_Y_ABS_MAX` | 0.25 | 第一根「门柱」的 \|y\| 上限 |
| 出生点 | (−0.5, 0, 0.15) | `SPAWN_X=-0.5`，给门柱前多 0.5 m；`--spawn-mode random` 另加 y/yaw 随机 |
| 教师 | A* + 纯追踪 | 全部 `planner=astar`；SAFE 来自到达 waypoint 与被围住两种情况（占 18.5%） |

> **2026-11-01 几何 v4**：safe 出生模式已删除。`plan_action` 只在「±85° 内每个方向
> 都被柱子挡住」时判 SAFE，而那种位姿的相机必然贴着柱面（实测 100/100 seed 距离
> 0.100 m），产出 4000/10442 步纯色 SAFE 帧，是闭环 34.7% 提前停车的首要嫌疑。
> 旧的 v3 数据（含 safe 集）不能再与新数据混包；`geometry_gid` 已随 SPLIT_SALT 重排。

> **2026-11-01 几何 v5（任务语义）**：L1 landmark 改成「两根**同色**柱分居左右 +
> 关系词指令」`pass the <color> pillar on your <left|right> and stop`（9 token，
> `kTextLen` 8→9）。原因是 v2 的两根柱异色，图像里目标色唯一，模型可以完全忽略
> 文本（实测换词只改变 2e-4 概率、argmax 0 次变化）。v5 的图像只给出「有两个候选」，
> 选哪根必须读指令。同源变化：`task_version` → `l1.v3`，`--task l1` 必须配
> `--spawn-mode random`（default 的 y=0 选不出装得进 60° 视场的左右柱对），
> S 组消融从「换色」改成「关系词取反」，`distractor_color` 字段删除。
> v4 及更早的数据不能与 v5 混包。
>
> **2026-11-01 任务 v4（`task_version=l1.v4`）**：几何版本仍是 `uavrgb.geom.v5`
> （柱子/出生点/split 都没变），只改目标点与关系词映射：
> `landmarks.waypoint_for` 从「朝走廊中心偏 0.8 m」改成「朝目标柱外侧偏 0.8 m」，
> `assign_landmarks` 里 relation 与目标柱的映射随之翻反（`left` 仍表示「经过时
> 目标柱在机身左侧」）。原因见 `docs/worklog/2026-11-01-1739-code-review-waypoint-degeneracy.md`：
> v3 的 274 个 test 场景**没有一个**要求横移，闭环 SR 只由「敢不敢飞 + 会不会停」
> 决定。`l1.v3` 与 `l1.v4` 数据不能混包（`preprocess.py` 按 `task_version` 拦）。

### split 机制

- **`geometry_gid`**：柱子坐标/尺寸 round 到 3 位 → 排序 → sha256 前 16 位。
  seed / scene_id / 文件名都不参与，所以**同几何必同 gid**。
- **`SPLIT_SALT`**（当前 `uavrgb.geom.v5`，与 `scene_gen.GEOMETRY_VERSION` 同源）：
  `sha256(salt + ":" + gid)` 前 8 位 %100，<80 train / <90 val / 其余 test。
  salt 的作用是让「同一份几何」在换了生成器之后落到**不同的** split，从而不会
  新旧混在一起。
- **改几何 → 必须 bump salt（v2→v3）+ 开新数据版本**。gid 变了但 salt 没变，新旧
  几何会静默混进同一 split，audit 只查 gid 交集，查不出来。
- **「开新数据版本」的落地做法**：换一个新的 `--out` 目录（如
  `.../sim-v3-train/`），不要往旧目录里追加；旧数据不原地重标。

### 数据与产物路径

> **L1 v3/v4 的路径见 §4.4**（那里是当前主线）。本表只列不在 §4.4 里的历史产物。

| 路径 | 内容 |
| --- | --- |
| `/root/rivermind-data/uav_rgb/t5-full-1155/` | T5 原始采集（599 集 / 23960 步，`train/ val/ test/` + `audit.json`） |
| `/root/rivermind-data/uav_rgb/t6-full-1640/ds/dataset.uavr` | T6 打包数据（23960 步，1.38 GB） |
| `/root/rivermind-data/uav_rgb/t6-full-1640/run/ckpt_best.bin` | T6 best 权重（epoch 8） |
| `/root/rivermind-data/uav_rgb/t6-full-1640/eval-best/` | T6 best 离线复评 |
| `/root/rivermind-data/uav_rgb/t6-full-1640/eval-student/` | T7 旧版闭环（R01 前） |
| `/root/rivermind-data/uav_rgb/t6-full-1640/eval-student-r01b/` | T7-R dispatch 复跑（SR 0.55） |
| `/root/rivermind-data/uav_rgb/airsim10k-uavr/dataset.uavr` | 公开数据打包，10000 样本 549 MiB |
| `/root/rivermind-data/uav_rgb/airsim10k-run/` | 公开数据的训练 run（含 `ckpt.bin` / `ckpt_best.bin`） |
| `/root/rivermind-data/uav_rgb/d1-verify-a` `d1-verify-b` | 新格式自采验证量（各 2 集 × 8 步） |
| `/tmp/ds200_bin/dataset.uavr` | 旧 3726 样本。**`/tmp` 随时可能被清掉，别依赖它**；它本来就不可用于训练 |
| `/root/rivermind-data/uav_rgb/pubdata/` | 公开数据原始 zip 2.0 GiB + 解压 3.5 GiB，确认不需要可删 |

### 其他符号

- **`gz sim -s`** 的 `-s` = **server only**（不开 GUI）。纯 headless 会 segfault，
  所以必须配 `DISPLAY=:20`。不加 `-s` 的 GUI 模式同样崩。
- **`--goal-margin`**（`eval_closed_loop.py`，默认 1.0）：`goal_x = 最右柱 x + margin`。
  成功判据是「无碰撞 **且** `max_x ≥ goal_x`」。

---

## 附：历史交接记录

- 2026-10-29 22:10 初版；2026-10-30 03:00 重写；2026-10-30 21:55 更新（T1–T7、D4、T7 复跑）。
- 2026-10-31 20:10 更新（L2 文本分支、v2 500 集数据）。
- **2026-11-01 21:52 更新（本次）**：L1 任务走到 v4；§4.3 换成 v4 数字；
  §4.5 重写（数据四代沿革 + 停止点不可观测 + 方案 A 已落地 + B/C 待拍板）；
  §六补 `goal_reach` 口径；§八补 6 条新坑；§九补 4 条待确认。
  更早的实质内容已并入上文（合同、环境事实、坑），未并入的部分见
  git 历史与 `docs/worklog/`。

**给接手 AI 的最后三条**：

1. **报数字前先读 `docs/EVAL_STANDARD.md`**，离地四件套（acc + macro_f1 +
   bal_acc + 每类 recall）缺一不可，只报准确率等于造假。
2. **不要自行重跑实验**。v4 的三组闭环要 80 min，数值不会变；训练要 81 min。
   下一步是 B/C 的选择题，需要用户拍板。
3. **不要 `git add -A`**。工作树里有并发会话的改动，逐文件确认归属。
