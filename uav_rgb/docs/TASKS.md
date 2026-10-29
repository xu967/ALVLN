# 任务板

- 建立：2026-10-29 22:19
- **唯一任务来源**：开工把状态改 `doing`，完成改 `done` 并链 worklog。HANDOFF / ENGINEERING_RULES 只保留优先级理由，不再各存一份状态。
- 文件域：`code-4b` = `src/ sim/tools/ tools/ docs/`；`code-3b` = `ui/`；子 agent 只读，或只写**新文件**。跨域先发消息。
- 验证三级：**L1** 相关单测（每次改）；**L2** `bash tools/check.sh`（每个任务完成，不许只跑子集）；**L3** 真实 Gazebo/训练（里程碑，先在本表登记，同一时刻只一个会话跑）。
- Git：一个主题一个提交，精确 `git add <path>`；首行「改了什么 + 为什么」，正文列行为变更与验证命令；文档单独提交；不 `git add -A`；用户明确授权后才 commit/push。

## 进行中

| ID | P | 状态 | owner | 文件域 | 任务 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | P0 | done | code-3b + code-4b | git | 清 37 文件积压，按主题提交 | 工作树干净；5 个主题提交 |
| E6-1 | P0 | done | code-4b | `src/infer_main.cpp`、`src/CMakeLists.txt`、`src/tests/` | 常驻推理进程：stdin 收 H×W RGB uint8，stdout 回 action/probs/infer_ms | 加载现有 ckpt；同帧 100 次输出一致；JSON 可解析（`2026-10-29-2226-infer-process.md`） |
| E6-2 | P0 | done | code-4b | `sim/tools/eval_closed_loop.py`、`sim/tools/collect.py` | 复用 collect 的 gz 连接，学生动作闭环飞，碰撞即停 | 1 个 test 场景跑通并输出 JSON（`2026-10-29-2232-closed-loop-eval.md`） |
| E6-3 | P0 | done | code-4b | `sim/tools/eval_closed_loop.py` | 固定 test split 场景集（几何 gid 与训练不重叠） | seed 5 / gid `c8afd118…`，split=test |
| E6-4 | P1 | done | code-4b | `sim/tools/eval_closed_loop.py` | SR、碰撞率、false_forward、推理 p50/p95、动作分布 | JSON + `docs/experiments/2026-10-29-closed-loop-smoke.md` |
| E6-5 | P1 | done | code-4b | `docs/` | 现有 checkpoint 基线 20 episode | 20/20 碰撞、SR=0；`2026-10-29-closed-loop-20ep.md` |
| E6-6 | P1 | done | code-3b | `ui/` | 只读载入 `eval.json`：ckpt 文件名 + split、SR/碰撞/false_forward、p50/p95(warm)、四动作分布、per_episode 的 max_x/碰撞 | `aa8838ac` + 真实 JSON 回归 `71fdc0cf`（未 push）；GUI 116 项 EXIT=0 |

## 阶段目标（2026-10-30 立）

**第一阶段目标：指标超过公开方案的公开数字。**

口径固定为 `docs/EVAL_STANDARD.md`：离线同时报 `acc` + `macro_f1` +
`balanced_accuracy` + 每类召回；闭环用 AvoidBench 的 SR/PO/EO。

**当前卡点：对标对象未定。** Kaggle AirSim 10k 这份数据**没有任何公开基线**
（`kernelCount=0`、无论文、无仓库、无 split、无指标），所以「超过他们」对它
没有定义。三条可选路（验收口径完全不同，需先拍板）：

| 选项 | 对标什么 | 我们能立刻做的 |
| --- | --- | --- |
| (a) 自建基线阶梯 | 盲前进 / 频率先验 / 我们的模型，比 macro-F1 | 能，只要 sim 数据够 |
| (b) AvoidBench 口径 | Ego-planner / MBPlanner / Agile-Autonomy 的闭环 SR | 要在 Gazebo 里重建同类场景，成本高 |
| (c) 换有榜单的数据集 | 公开 leaderboard | 有重做一次接入的成本 |

每次汇报必须同时写清「我们的数字」「对标数字」「对标数字出处（论文/榜单/仓库 + 日期）」。

## 数据来源决策（2026-10-29 22:30 立，22:51 改，10-30 01:30 有实测结论）

**实测结论：公开数据集可用但不能单独用——`SAFE_NEUTRAL` 为 0 条。**

- Kaggle AirSim 10k 已下载（2.0 GiB）、校验（30000 文件序号连续无缺口）、
  清点、离散化、打包（`/root/rivermind-data/uav_rgb/airsim10k-uavr/dataset.uavr`，
  10000 样本 549 MiB）并完成一次训练。详见 `2026-10-30-airsim10k.md`。
- **能过的一条**：标签与深度几何强相关（`YAW_LEFT` 左边更近 82.1%、
  `YAW_RIGHT` 16.8%，差 25.8 sigma）→ 图像/深度/标签是对齐的。
- **过不了的一条**：`vx` 最小 1.00 m/s，势场法从不输出「停」，
  `SAFE_NEUTRAL` = 0 条。四类接口里的第一类它教不了。
- 因此：公开数据只能当预训练/链路验证，**四类模型仍必须靠 sim 自采**。
  D1 修好后自采的 `sim_dt` 已是 0.0% 变异，且 `label_recompute` 可逐条复算。

历史决策记录（保留）：

- 首版结论是「不接入公开数据集」，理由是 ICL corridor / AirGoal-10k /
  drone-vla-toolkit 的标签语义与相机对不上我们的 4 类接口。同日闸门实测推翻
  了「自采可信」这个前提（旧 200 集 4 项 FAIL），所以两条线并行推进。
- 候选：Kaggle `lukpellant/droneflight-obs-avoidanceairsimrgb10k-320x320`
  （AirSim，10k 样本，RGB 320×320 + depth + 连续指令 `(vx,vy,vz,yaw_rate)`，
  CC BY 4.0）。标签由确定性势场法脚本生成，可复算。
- **卡点**：下载需 Kaggle API token（`~/.kaggle/kaggle.json`），本机没有，匿名 403。
  HuggingFace 全站在本机不可达（curl 000）；Padhy 走廊数据集无公开下载地址。
- 拿到后的顺序：`audit_dataset.py` 审类别分布与标签-物理一致性 → 定义
  `(vx,vy,vz,yaw_rate)` → 4 类的映射 → 转 `.uavr` → 训练。审不过就不训。
- 已知风险：势场法可能从不输出「停」，`SAFE_NEUTRAL` 可能又为空；320×320 方形与
  本机 320×240 / 160×120 不同，需裁剪或缩放；AirSim 世界不是我们的柱阵。
- sim 自采仍是主线备份：D1 → D2 → D3 修完 `sim/tools/` 后多实例采，
  1000 集 × 20 步 ≈ 2 万帧，4 实例 ~3.5 h / 8 实例 ~1.7 h。

## 待排（E6 后）

| ID | P | 状态 | owner | 文件域 | 任务 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | P0 | done | code-4b | `sim/tools/audit_dataset.py` | 数据集验收闸门：8 条不变量，FAIL 即非零退出 | `/tmp/ds200` 跑出 5 项 FAIL（`2026-10-29-2241-dataset-audit.md`） |
| D0b | P0 | done | code-4b | `sim/tools/collect.py`、`audit_dataset.py` | actions 记 `pose_before`/`yaw_before`，audit 真实现 `label_recompute` | 新采集 16 步 0 不一致；depth 存盘**未做**（无消费方），见 `2026-10-29-2338-simtime-hold-and-pose-before.md` |
| D1 | P0 | done | code-4b | `sim/tools/collect*.py`、`eval_closed_loop.py` | 仿真时间动作窗口（`--hold` 改 sim-time） | 同 seed 两次采集 `sim_dt` 全为 1.000 s，cv 0.0%（旧 18.3%）；同上 worklog |
| D0c | P1 | done | code-4b | `tools/airsim_to_uavr.py`、`tools/preprocess.py` | 公开数据集接入：离散化 + 裁剪 + 打包 `.uavr` | 10000 样本打包成功，11 项自检通过，训练跑通（`2026-10-30-airsim10k.md`） |
| D2 | P1 | todo | code-4b | `sim/tools/scene_gen.py` | 车道几何决策（缝 0.902 m vs 机体 0.711 m） | 目标穿越率 + 实测 |
| D3 | P1 | todo | code-4b | `sim/tools/collect.py`、`teacher.py` | SAFE_NEUTRAL 数据/四类定义 | 每类样本量 + 宏 F1 可解释 |
| D4 | P2 | todo | code-4b | `sim/tools/collect.py` | depth/contact 停更门禁 | 停更注入测试失败 |
| D5 | P2 | todo | code-4b | `src/device/`、`src/trainer.cpp` | device 层接入训练路径、RNG/best 状态 | 吞吐对比 + resume 一致 |
| G1 | P3 | todo | 子 agent（只写新文件） | `src/tests/` | CUDA 多 block parity | 新增用例通过 |
| G2 | P3 | todo | code-4b | `src/checkpoint.cpp` | resume 数据签名校验（升 v4） | 换数据 resume 被拒 |
| G3 | P3 | todo | code-4b | `src/checkpoint.cpp`、`dataset.cpp`、`run_owner.cpp` | fsync、64 位溢出、目录 fd 锁 | 单测 |
| G4 | P3 | todo | code-4b | `src/CMakeLists.txt` | `-Wall -Wextra` 全开、`.clang-format` | 0 warning（真） |
| G5 | P3 | todo | code-4b | `sim/tools/collect_parallel.py` | 同 `--out` 并发 TOCTOU 加锁 | 并发负例被拒 |
| G6 | P3 | todo | code-4b | `src/train_main.cpp` | 生效配置落盘 `<out>/config.json` | 字段与 CLI 一致 |

## 已完成（2026-10-29）

| 项 | worklog |
| --- | --- |
| best-checkpoint + checkpoint v3 | `2026-10-29-2155-best-checkpoint.md` |
| 数据完整性 P1×3（-0.0 / 几何门禁 / 位姿停更） | `2026-10-29-2200-data-integrity-fixes.md` |
| C++/CUDA 静默失败 P1×4 | `2026-10-29-2215-src-silent-failure-fixes.md` |
| 本地监控台（无 web、真实训练接入） | `2026-10-29-2140-monitor-review2-fixes.md`、`2026-10-29-2205-real-trainer-integration.md` |
