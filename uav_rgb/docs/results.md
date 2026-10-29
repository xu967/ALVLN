# uav_rgb 验证与实验结果

本文件只记录真实跑出的命令、输出与指标，并与 `docs/worklog/` 的过程记录分开；追加不覆盖。

## 2026-10-29 19:00 归因分析：数据集问题还是训练问题（CPU）

- 环境：GPU 模式已确认；本条纯统计 + 源码审读，不占 GPU。
- 数据：`/tmp/ds200` 200 条 episode；模型指标取自 30 epoch 训练结果。

### 一、训练链路是有效的（三条证据）

| 证据 | 数字 |
| --- | --- |
| image-free 先验基线 vs 模型（test，N=400 / 375） | 0.8300 → **0.9733** |
| 先验基线错误分解 | step 0–2：N=60 错 **28**；step 3–19：N=340 错 **40** |
| 模型 test 总错误 | **10** |

先验在早期步（真需要看画面的步）错 28 个、全程错 68 个，模型只错 10 个。
**先验错的地方模型基本都改对了**，说明它在读图像，不是在读步号。

另外 `test_overfit`（16 个合成样本 200 步 → 训练准确率 >95%）通过，
gradcheck、CPU/CUDA 对拍、16/16 CTest 全绿 —— 梯度与优化器正确。

### 二、训练侧的确定缺陷（真实存在，但不是本次结果的原因）

| 编号 | 缺陷 | 证据 | 影响 |
| --- | --- | --- | --- |
| T1 | 只保存最后一个 epoch，没有最优 checkpoint 选择 | `src/trainer.cpp:253` 每 epoch 覆盖 | dev acc 峰值 0.9784@ep27，交付的是 0.9757@ep30，白丢 0.27 pt |
| T2 | `auto` 类别权重在 `n_c=1` 时退化 | 权重 `[745.25, 0.339, 1.827, 1.982]`，Σw=2980.3 | 单个 SAFE 样本占 **25%** 的 loss |
| T3 | 过拟合无任何抑制 | train_ce 0.029 vs dev_ce 0.083 | 无 weight decay / dropout / 增强 / LR 衰减 |
| T4 | 无早停 | 同上 | 与 T1 叠加 |

### 三、数据集侧的缺陷

| 编号 | 缺陷 | 证据 |
| --- | --- | --- |
| D1 | 从未进入障碍场 | 0/200 穿场，终点 x 中位数 1.109 m vs 场地 1.5–5.0 m |
| D2 | 动作序列退化 | 200 条 episode 只有 79 种序列 |
| D3 | 有效样本量远小于 4000 | 200 条独立轨迹，test 只有 20 条 |
| D4 | SAFE 类实际不存在 | train 1 / val 0 / test 0 |
| D5 | 场景族只有 1 个 | `family_holdout=false` |
| D6 | 3 组几何跨 split | 见 18:20 条目 |
| D7 | **三个固定彩色地标** | 红/绿/蓝 1 m 立方体固定在 (3,0)/(3,1.5)/(3,−1.5)，200/200 场景不变；且 **200/200 场景都有柱子陷在地标内部**（每个场景 3–4 根）。`validate_pillars` 只检查柱间缝隙与边界，不检查地标 |

D7 补充：地标 z∈[0.5,1.5]，巡航高度 2.3 m，**不阻挡飞行**。所以它不解释模型表现，
但它是场景噪声源，也是历史上 teacher 深度带被填满导致 SAFE_NEUTRAL 卡死的原因。

### 结论

**主要矛盾在数据集（D1 为根因，D2–D7 是其衍生物或独立污染）。训练链路本身有效。**

T1/T2 必须在采集新数据之前修掉，否则新一轮实验仍会被「交付的是过拟合末轮」
和「单样本占 25% loss」污染，届时无法把失败归因到数据或模型。



- 环境：GPU 模式已确认，本条纯 CPU 统计，不占 GPU。
- 数据：`/tmp/ds200` 全部 200 条 episode、4000 步、200 个 `oracle.jsonl`。
- 命令：`python3 -I` 内联统计（读 `oracle.jsonl` / `actions.jsonl` / `rgb_*.png`）。

**终点位置 vs 柱子场深度**

| 项 | 值 |
| --- | --- |
| 柱子场 x 范围 | 1.5 – 5.0 m（纵深 3.5 m） |
| 终点 x 分位 | p10 0.79 / p25 0.95 / **p50 1.11** / p75 1.28 / p90 1.37 / max 2.05 |
| x ≥ 1.30（首排柱前表面） | 42 / 200 |
| x ≥ 1.50（首排柱中心） | 6 / 200 |
| **x ≥ 5.00（穿场）** | **0 / 200** |
| 实飞 `d_trav` | 1.124 – 3.476 m，中位数 1.713 m |

**每步仿真时间增量**（`oracle.jsonl` timestamp 差，3800 个样本）

| 项 | 值 |
| --- | --- |
| 中位数 | **0.420 s** |
| 均值 / min / max | 0.434 / 0.160 / 0.880 s |

`--hold 1.0` 是墙钟窗口，8 路并行下 RTF≈0.42，所以每步只推进 0.42 s 仿真时间；
`SPEED_FWD=0.3 m/s` → 每步 0.126 m → 20 步约 1.7 m。**要从 x=0 到 x=5.0 需 40 步纯前进。**

**动作序列退化**

| 项 | 值 |
| --- | --- |
| 不同动作序列数 | **79** / 200 条 episode |
| 最常见序列 | `L L L` + `F`×17，14 条 |
| 前 5 名序列覆盖 | 55 / 200 |

**不看图的基线**

全部 4000 步上：全预测 FORWARD 0.7520；只按 step 序号取众数 **0.8127**。

在 test split 上（N=400，未过滤碰撞步；用 train 的逐 step 众数预测）：

| 预测方式 | test 准确率 |
| --- | --- |
| 全预测 FORWARD | 0.7600 |
| **只按 step 序号取众数，完全不用图像** | **0.8300** |
| 模型 | 0.9733 |

结论：图像带来的真实增量是 **14.3 个百分点**（0.8300 → 0.9733），不是 22。
数据集的物理前提不成立——它不是避障数据集，是「起飞后转向 3 次再直飞 1 米」的数据集。



- 环境：GPU 模式已确认（RTX 2080 Ti），但本条全部是 CPU/Python 计算，不占 GPU。
- 几何 = `scene_gen.sample_pillars(seed)` 的 x/y/z/size，按坐标排序后取 sha256 前 16 位；
  排除 scene_id / seed / 文件名 / 注释。
- 命令：`python3 sim/tools/split_manifest.py --verify <collection_manifest.json>`

| 数据集 / seed 范围 | seed 数 | 不同几何 | 跨 split 几何组 | 旧策略 |
| --- | ---: | ---: | ---: | --- |
| `/tmp/ds200`（实际训练的 200 条） | 200 | 192 | **3** | `scene_id % 10` |
| seed 0..1999（HANDOFF 记录值复算） | 2000 | 1124 | **241** | `scene_id % 10` |
| seed 1000..1199（新策略） | 200 | 192 | **0** | 几何组 hash |

`/tmp/ds200` 的三组泄漏（真实出现在已训练数据里）：

```
8ba41c6d30fc1277  train↔test   seeds=[1023, 1069]
4bfb34754ed02c52  train↔val    seeds=[1024, 1188]
e71bf144dce144d9  train↔test   seeds=[1192, 1129]
```

`--verify` 对旧数据 exit=1，对新策略 exit=0（3 条冒烟 episode，split 与目录、episode manifest 三处一致）。

新策略在 1000..1199 上的分配：train 157 组 / val 17 组 / test 18 组，重复 seed 8 个。

结论：几何泄漏真实存在，但 200 条时只占 1.5%（8/200），**解释不了 test 准确率 0.9733**；
真正的病是场景族只有 1 个（`family_holdout=false`），200 条全是同一蛇形车道模板。
泄漏随扩量变严重（2000 seed 是 241 组），所以分组必须在扩量前做。

## 2026-10-28 23:29 监控台增量读取基准（CPU）

- 环境：Ubuntu 24.04 容器，96 核；GPU 模式已确认（RTX 2080 Ti），但本基准只测 Python 解析，不占 GPU。
- 数据：合成 `metrics.jsonl`，200,000 行、14.6 MiB，每行一个 `train_step` 事件；非真实训练产物。
- 对照：旧实现每次刷新全文件读取 + 全量 `json.loads`；新实现首次只读末尾 1 MiB，之后按字节偏移读新增行。
- 命令：`python3` 内联脚本，`sys.path.insert(0, 'ui')`，调用 `monitor.tail_events`；`full_old` 为旧逻辑复刻。

| 项 | 第 1 次 | 第 2 次 | 第 3 次 |
| --- | ---: | ---: | ---: |
| 旧实现全量重读 | 1.042 s | 1.015 s | 1.010 s |
| 新实现首次打开（末 1 MiB） | 0.068 s | 0.066 s | 0.067 s |
| 新实现无新数据 tick | 0.149 ms | 0.036 ms | 0.029 ms |
| 新实现 +1 事件 tick | 0.053 ms（1 条，step=200000） | — | — |

结论：首次打开约快 15 倍；训练长跑时每 5 s 的 tick 从约 1 s 降到亚毫秒级，且不再阻塞 Tk 主线程。旧实现下 200k 行日志每 5 s 卡顿约 1 s；1M 行时按实测线性外推约 5 s/次，会持续卡死 UI。

## 2026-10-28 23:27 监控台 GUI 动态验证（DISPLAY=:20）

- 环境：GPU 模式；Tk 8.6；Xorg 软件 GL 桌面 `:20`；`python3` = conda 3.11.15（另有系统 3.12.3 + python3-tk 验证通过）。
- 测试：`ui/test_monitor.py`（48 项）与内联 live 脚本。

| 项 | 真实输出 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | `test_monitor: 48 项通过`，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | `test_monitor: 31 项通过，1 项跳过`，EXIT=0（无显示时跳过 GUI） |
| `DISPLAY=:20 /usr/bin/python3 ui/test_monitor.py` | `test_monitor: 48 项通过`，EXIT=0（系统 3.12.3 + python3-tk） |
| live：启动后 | 4 events / 8 rows |
| live：追加 1 条 event + 1 行 results.csv 后（500 ms 刷新） | 5 events / 9 rows；status `9 epochs · 5 事件 · 数据 1s 前 · 刷新 23:26:51` |
| live：暂停后追加 1 条 | 仍 5 events（暂停生效） |
| live：继续后（500 ms 刷新） | 6 events；`LIVE_GUI_OK` |
| `./launch_ui.sh --run-dir demo --refresh 1`（XFCE 会话 PATH，`timeout 6`） | EXIT=124（6 s 时仍在运行，超时终止）；stderr 为空 |
| 截图 | `ui/evidence/monitor_demo.png`（980×660，真实窗口抓取） |

修复前已复现的失败（旧代码真实输出）：

| 输入 | 旧行为 |
| --- | --- |
| `results.csv` 中 epoch 为空 | `TypeError: '<' not supported between instances of 'int' and 'NoneType'` at `_line_chart` |
| `metrics.jsonl` 含 `[1,2,3]` | `AttributeError: 'list' object has no attribute 'get'` in `fmt_event` |
| `metrics.jsonl` 含非 UTF-8 字节 | `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff` |
| 200k 行 / 14.6 MiB 日志 | 每次刷新约 1.0–1.1 s（旧 `load_events`） |

## 2026-10-28 23:19 环境回归（GPU 模式）

- 环境：`nvidia-smi -L` → `GPU 0: NVIDIA GeForce RTX 2080 Ti`；CUDA 12.8；驱动 580.119.02。
- 命令与真实输出：

| 命令 | 结果 |
| --- | --- |
| `cmake --build src/build` | `ninja: no work to do.`（已是最新） |
| `ctest --test-dir src/build --output-on-failure` | `100% tests passed, 0 tests failed out of 6`（1.13 s） |
| `experiments/cuda_smoke/build/cuda_smoke` | `RESULT: PASS`，max abs error 0，kernel 0.0328 ms |
| `cuobjdump -lelf build/cuda_smoke` | 仅 `sm_75` |
| `python3 sim/tools/teacher.py` | 6/6 通过，EXIT=0 |
| `python3-tk` 安装后 `/usr/bin/python3 -c 'import tkinter'` | 3.12.3 Tk 8.6 |

## 2026-10-28 23:23 环境发现（未完成项）

- 旧 web 监控进程仍在运行：`python3 .../ui/server.py --port 8799`，PID 133198，已运行 2h41m，监听 `127.0.0.1:8799`，`/api/v1/health` 返回 `{"status":"ok","service":"uavrgb-monitor",...}`。`ui/server.py` 文件已删除，进程为历史残留。
- 尝试 `kill -TERM 133198` 被权限分类器拒绝（未获用户授权终止该进程）；未自行绕过。需要用户执行 `kill 133198` 或重启会话清理。该进程不影响本地 Tk 监控台，但会继续占用 8799 端口。

## 更新 2026-10-28 23:30（最终回归）

- 监控自检新增 `csv.Error 保留已解析行` 与 `GUI 缩放后曲线重画` 两项：`DISPLAY=:20 python3 ui/test_monitor.py` → **50 项通过**，EXIT=0；无显示 → **37 项通过、1 项跳过**，EXIT=0。
- 畸形输入 6 s 压力：临时 run-dir 同时含空 epoch、非 UTF-8 字节、非 object JSON、孤立代理字符；`DISPLAY=:20 timeout 6 python3 ui/monitor.py --run-dir <dir> --refresh 1` → EXIT=124（6 s 时仍运行），stderr 为空。
- 最终环境回归（并行会话在审核期间新增了 `test_nn_cuda`）：
  - `ctest --test-dir src/build` → **7/7 通过**（1.56 s；会话开始为 6/6）
  - `cuda_smoke` → `RESULT: PASS`，avg kernel 0.0457 ms，仅 sm_75
  - `teacher.py` → 6/6 通过；`gz sdf -p` → EXIT=0；`py_compile`（ui + sim tools）→ OK
- 残留：PID 133198 仍在 `127.0.0.1:8799`；未获授权终止。Gazebo 审核进程已清理（`pgrep -af 'gz sim'` 无输出）。

## 2026-10-29 00:25 监控台审核修复回归（本地 Tk）

- 环境：本机 Xorg `:20`，Tk 8.6；conda Python 3.11.15 与系统 Python 3.12.3 都跑。
- 命令与真实输出：

| 命令 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | `test_monitor: 60 项通过`，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | `42 项通过，1 项跳过`（GUI 无显示），EXIT=0 |
| `DISPLAY=:20 /usr/bin/python3 ui/test_monitor.py` | 60 项通过，EXIT=0；`Tk 8.6` |
| `env -u DISPLAY ./ui/launch_ui.sh` | `launch_ui.sh: 没有 DISPLAY；请在 XFCE 桌面会话（X11）里运行`，EXIT=1 |
| `DISPLAY=:20 timeout 5 ./ui/launch_ui.sh --run-dir demo` | EXIT=124（5 s 时仍在运行），无 traceback |

修复对照（真实测得的旧行为 → 新行为）：

| 项 | 旧 | 新 |
| --- | --- | --- |
| 同 inode 写回更大文件 | 316 B/4 事件 → 594 B/6 事件，`tail_events` 返回 2 条、`reset=False`，旧事件不清 | 首 64 字节指纹变化 → `reset=True`，6 条全读；测试覆盖 |
| 文件删除 | `([], False)`，界面保留旧事件 | `([], True)`，清空并显示“等待 metrics.jsonl” |
| `load_events(limit=0)` | 返回全部 | 返回空 |
| 单 epoch 曲线 | x 轴伪造 `1, 2` 两个刻度 | 只画 `1` 一个真实刻度 |
| 表格重建 | 选中行丢失 | 按 epoch 恢复选中行；滚动位置 Tk 保留 |
| 不可解码路径 | `realpath`/Tk 抛 `UnicodeEncodeError` | 显示为 `?`，不崩溃 |
| 无 DISPLAY 启动 | Tk traceback，桌面无提示 | 启动脚本直接报错退出 |

## 2026-10-29 00:20 仿真采集审核修复（takeoff / out-dir / 接触 / 深度）

- 环境：GPU 模式；Gazebo Harmonic 8.15，`DISPLAY=:20` 软件 GL，server-only；`Z_TARGET=2.1`。
- 命令：`gz sim -s -r sim/worlds/uav_rgb.sdf` → `collect.py --steps 10 --hold 0.5 --out /tmp/uavrgb_review_ep --scene-id 1 --seed 42`。

| 项 | 真实输出 |
| --- | --- |
| 起飞 | `起飞完成: z=2.20 m`（旧实现 2.5 s wall 只到 z≈1.15，10 步里 7 步 SAFE_NEUTRAL） |
| 10 步 | `写入 10/10 步`，`COLLECT_RC=0`；动作 `[1,1,1,1,1,1,1,1,2,1]`（8×FORWARD、1×YAW_LEFT、1×FORWARD），无 SAFE_NEUTRAL |
| z 高度 | step 0 `2.636 m`（起飞零速后有过冲），step 9 `2.314 m`；不再卡在低空 |
| `action_wall` | 0.47–0.59 s（`--hold 0.5`） |
| `depth_rgb_gap_ns` | 168–308 ms（深度/RGB 仍是两次独立读取，已落盘，未同帧） |
| `actions.jsonl` 字段 | `depth_ts` / `depth_rgb_gap_ns` / `action_start_ns` 均在 |
| manifest | `camera_verified/physics_verified/episode_executed = true/true/true`，`error=null` |
| 非空 `--out` | `collect.py: error: --out /tmp/uavrgb_review_ep 不是空目录；请换新目录（或先清空）`，EXIT=2 |
| world 未复位 | 第二次采集起飞前 `z=2.30 m` → `RuntimeError: world 未复位…`；manifest 三项 validity 全 false，EXIT=1 |
| 残留 | `pgrep -af 'gz sim|ruby'` 无输出；测试目录 `/tmp/uavrgb_review_ep*` 为本次临时产物 |

- 纯函数自检：`python3 sim/tools/test_collect.py` → **13 项通过**；`python3 sim/tools/test_save_frame.py` → **PASS（10 项）**；`python3 sim/tools/teacher.py` → **22/22 通过**（并行会话新增卡死用例）；`py_compile` 与 `gz sdf -p` 均 EXIT=0。
- 说明：`teacher.py` 的 `turn_dist=1.5/emergency=0.8` 是并行会话的改动，不在本次 collect 修改内；两边的实跑是同一版 `collect.py`。

## 2026-10-29 00:50 本地训练控制台（Tk）+ DEMO trainer

- 环境：`DISPLAY=:20`，Tk 8.6；conda Python 3.11.15；系统 Python 3.12.3。
- 自检：

| 命令 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | **85 项通过**，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 56 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |

- 控制台内真实启动默认 DEMO trainer（不是直接跑 CLI）：
  - 3 epoch / 10 steps：`rows=3 events=35 demo_lines=11 rc=0`，状态“训练完成 rc=0”；run 目录 `logs/runs/20261029-004013-demo`，包含 `metrics.jsonl`、`results.csv`、`train.log`、`train_command.txt`、`train_config.json`。
  - 10 epoch / 100 steps：`rows=10 events=112 demo_lines=32 rc=0`。
  - 假 trainer：启动 → running；停止 → `已停止 rc=0`；崩溃 rc=3 → 自动切到“训练输出”页。
- 截图：`ui/evidence/monitor_control.png`（980×660，真实窗口，DEMO 运行中）。
- 说明：DEMO 的 loss/accuracy 是 `seed` 驱动的**模拟值**，不代表任何模型效果；真实 C++/CUDA trainer 尚未接入，只有控制层与 `--config` 合同。

## 2026-10-29 12:45 训练控制审核修复回归

- 环境：`DISPLAY=:20`，Tk 8.6；conda Python 3.11.15。

| 命令 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | **89 项通过**，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 55 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |
| `python3 -m py_compile ui/monitor.py ui/trainer_demo.py ui/test_monitor.py` | OK |

修复对照（均为真实窗口内实跑）：

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 命令写错后改好重试同一目录 | `启动失败: No such file...` → 目录留下 `train.log`, `train_command.txt`, `train_config.json` → 重试报 `运行目录不是空目录` | 失败后目录为空（`[]`），同目录重试进入 `running` 并 `rc=0` |
| 选中已跑完的 run 目录 | `train.log` 2122 B，「训练输出」0 字符 | 1887 字符，首行 `[DEMO] 模拟训练（非真实训练）开始，run_dir=...` |
| 一次训练结束后再点开始 | 输入框仍是刚用过的目录 → 必报「不是空目录」 | 自动换新默认目录（实测 `20261029-124456-demo`） |
| 死代码 | `prepare_train_log` 只在空目录（`train.log` 必然不存在）上调用，永不触发 | 已删函数、`TRAIN_LOG_CAP` 与对应自检（净减 13 行） |

## 2026-10-29 21:40 监控台二审修复回归

- 环境：`DISPLAY=:20`，Tk 8.6；系统 Python 3.12.3。

| 命令 | 结果 |
| --- | --- |
| `DISPLAY=:20 python3 ui/test_monitor.py` | **94 项通过**，EXIT=0 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 57 项通过，1 项跳过，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |
| `python3 -m py_compile ui/monitor.py ui/trainer_demo.py ui/test_monitor.py` | OK |

修复对照（真实窗口内实跑，1 epoch / 1 step 的 DEMO）：

| 场景 | 修复前 | 修复后 |
| --- | --- | --- |
| 快跑（同整秒内起停）结束后再点开始 | 默认目录仍是 `20261029-211951-demo` → `启动失败: 运行目录不是空目录` | 默认变成 `20261029-213653-2-demo`（不存在），再点直接 `运行中 pid=...`，`rc=0` |
| 186 KB 中文 `train.log`（6000 行）分 3 tick 读 | 块边界切在行中间，出现半行与 `�` | 2000 行（视图上限）全部是完整行，`含替换符: False` |
| 写盘中途失败留下的 `train_config.json.tmp` | 不在回滚名单 → 重试报「不是空目录」 | 回滚后目录为空（自检里用假 `write_train_config` 制造该文件） |

## 2026-10-29 21:55 best-checkpoint 回归

- 环境：GPU 模式（RTX 2080 Ti，sm_75）；CUDA 12.8；合成数据集走 GPU 训练器。

| 命令 | 结果 |
| --- | --- |
| `cmake --build build/check --parallel 8` | 成功，无 warning 输出 |
| `ctest --test-dir build/check --output-on-failure` | **18/18 passed**，99.04 s |
| `./build/check/test_trainer` | `best dev macro_f1=0.136364 state=0.136364`，`RESULT: PASS` |
| 同一测试末轮 | dev macro_f1=0.0294（若交付末轮，会比峰值差 4.6 倍） |
| checkpoint v2 兼容 | 由 v3 原地降级后读回：参数逐位一致、`best_macro_f1=-1` |

- 新增产物：`<run_dir>/ckpt_best.bin`；checkpoint 版本 2 → 3（新增 `best_macro_f1`）。
- 未验证：真实采集数据、跨版本续训、磁盘满。

## 2026-10-29 22:00 数据完整性修复回归

- 环境：GPU 模式；本组为纯 Python/预处理测试，未启动 Gazebo。

| 命令 | 结果 |
| --- | --- |
| `python3 sim/tools/test_split_manifest.py` | 9 tests OK（含 seed 337/3651 同 gid 同 split） |
| `python3 sim/tools/test_collect.py` | 14 项通过（含 pose 停更 fail-closed） |
| `python3 sim/tools/teacher.py` | 全部通过（含全堵且 base 堵 → SAFE_NEUTRAL） |
| `python3 -m py_compile`（collect/preprocess/split_manifest/teacher/tests） | OK |
| 同 gid 跨 train/val → `tools/preprocess.py` | rc=1，`几何组 g_same 同时出现在 split=train 与 val`，不产 dataset |
| 无 gid 的合成 episode → preprocess | rc=0，`dataset.uavr` 634630 B |

修复前复现：`seed 337` y=`-0.0` → gid `e8c96d776cd0ba5c` → test；
`seed 3651` y=`0.0` → gid `cc45e868b6e1e474` → train，物理布局相同。
`salt` 已从 `uavrgb.geom.v1` 升到 `v2`，v1 数据不得与 v2 混用。

## 2026-10-29 22:05 真实训练接入（界面里跑 uavrgb_train）

- 环境：`DISPLAY=:20`，Tk 8.6；真实 trainer `src/build/uavrgb_train`（18:03 构建）；数据集 `/tmp/bench_ds.uavr`（29 MB，448 样本）。

| 项 | 实测 |
| --- | --- |
| 界面启动真实训练（3 epoch / batch 64） | `运行中 pid=619634` → `训练完成 rc=0`；表格 3 行、事件 26 条 |
| 启动瞬间 run 目录 | `[]`（RunOwner 空目录要求） |
| run 目录产物 | `ckpt.bin` / `config.json` / `metrics.jsonl` / `results.csv`，无 train.log |
| `<run>.monitor/` | `train.log` / `train_command.txt` / `train_config.json` |
| 跑 4 s 后点停止 | `已停止 rc=-2`；已完成产物保留，无半截文件 |
| 命令行 resume 1 epoch | `rc=0`；`results.csv` 4 行（epoch 1–4）、事件 36 条，监控台正常解析 |
| `env -u DISPLAY python3 ui/test_monitor.py` | 61 项通过，1 项跳过，EXIT=0 |
| `DISPLAY=:20 python3 ui/test_monitor.py` | **100 项通过**，EXIT=0 |
| `python3 ui/test_trainer_demo.py` | 89 项通过，EXIT=0 |

说明：这次接的是真实 `uavrgb_train`（`ui/trainer_native.py` 把 `uavrgb.train.v1` 转成 CLI 后 execv），不再依赖 DEMO。
