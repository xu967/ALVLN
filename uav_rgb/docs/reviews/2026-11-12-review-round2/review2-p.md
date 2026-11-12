# 第二轮交叉审核报告（角度 P：Gazebo 闭环一集的时间与 CPU 拆分及无损提速）

- 审核仓库：`/root/claude-code-backup-20261028/code/uav_rgb`（分支 `main`）
- 审核基准提交：开工时 `1d92be0`（`2026-11-12 08:01:22 +0800`）；审核期间主会话合入卡 W1-15a（`f2c1107`）、卡 W1-13 第一二部分（`e40c693`）、卡 W1-15b（`035b398`），收尾核对至 `3a1db2f`（`2026-11-12 08:44:51 +0800`）。下文涉及被改文件的行号同时标出 `1d92be0` 与合并后 `7b6e71a`/`3a1db2f` 的位置。
- 草稿与只读统计产物：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-p/`（`analyze_all.py` → `analysis_summary.json`，`detail_checks.py` → `detail_checks.json`，`extra_checks.py` → `extra_checks.json`，以及间隔 30 s 以上的 7 次只读 `ps` / `/proc/<pid>/stat` / `cgroup` / `gz sim` 线程采样）。

## 0. 结论

柱阵单集闭环评测（K=1，约 31–34 步，`--action-latency 0.20`）在低竞争 2 车道下平均耗时 `54.3 s`（中位 `56.1 s`），容器满载被限流时升至 `60.0–84.7 s`。把墙钟拆开后有两个核心事实：第一，非步进固定开销占了 `20.3 s/集`（占整集 `37.5%`），其中每集重起 `gz sim` 等就绪 `5.7 s`、硬编码 `SIM_SETTLE_S` 空等 `3.0 s`、起 `collect.py` 与 `uavrgb_infer` 预热 `2.4 s`、固件起飞爬升 `7.3 s`、集末重算 SPL 栅格 A* `1.2 s`（K=3 长课程升至 `5.9 s`）；第二，步进循环占 `33.9 s/集`（`62.5%`，`1.09 s/步`），其中 `89.9%`（`30.5 s`）都在等 Gazebo 走满 `0.80 s` 仿真动作窗口，原因是评测默认未传 `--no-aux-cameras`，软件渲染线程 `llvmpipe-0` 每秒多渲了 `89.5%` 的无用像素（`chase` 640×360@15Hz、`top` 640×360@10Hz、`depth` 320×240@10Hz），把锁 1.0 的实际实时因子压低到了 `0.82`（满载时 `0.65`）。在不改变学生输入图像、不改变派发时序与数值的前提下，最要紧的 3 项改动是：（1）评测与非录像采集默认关掉辅助相机并及时关闭 `camera_info` 订阅，同时去掉 `cf_sitl` 不需要的 `3.0 s` 死等与重复的 SPL/影子教师 A* 开销（`P-2` + `P-3`，低风险改约 30 行，单集立省 `8.5–10.5 s`，单车道降约 `0.30 核`）；（2）同臂跨集复用 `gz sim`（世界复位 + 重摆柱子）与常驻 `StudentPolicy`（`P-1`，单集再省 `8.0–9.0 s` 冷启动）；（3）把 `gz_io.py` 的 6 个 `gz topic --json-output` 子进程改为进程内 `gz.transport13` 订阅并在学生评测时关掉深度相机（`P-4` + `P-5`，单车道再降 `0.32–0.38 核`，使单车道稳态 CPU 从 `1.40 核` 降至约 `0.75 核`，从而在 12 核配额下安全把 Gazebo 池从 2 车道扩到 3–4 车道 `P-7`，600 集总墙钟可从 5.5 小时以上压到 1.8–2.2 小时）。需警惕一处隐藏竞态：当前 `frame_dt = 0.9 s` 依赖第 `0.900 s` 相机帧晚于 `1.005 s` 位姿帧到达 Python（仅差 `5 ms` 仿真时间加传输延迟），任何削减图像传输/渲染延迟的改动都必须用 W1-13 刚合入的 `obs_gap_s` / `safe_gap_s` 验证未触发 `0.9 s → 1.0 s` 跳变。

---

## 1. 一集闭环的墙钟与 CPU 拆分

### 1.1 数据来源与测量方法

1. **低竞争基准样本（2 车道，无并行训练）**：`$D/exp-action-latency-r2-20261111/acc-L0.20-k1-s{0,1,2}/`（3 个臂 × 20 集 = 60 集，共 `1,876` 步，平均 `31.27` 步/集，中位 `32–33` 步/集）与长课程 `acc-L0.20-k3-s{0,1,2}/`（60 集，均值 `53.6` 步/集）。读取每集 `eval.json` 的 `wall_s`、`performance.mean_step_s`、`action_rtf`，每集 `actions.jsonl` 的 `rgb_ts`、`pose_ts`、`infer_ms`，以及臂级 `.cg` 的 `usage_usec` 与 `throttled_usec`（汇总于草稿 `analysis_summary.json`）。
2. **高竞争在跑样本（2 车道 + 3 个 `uavrgb_train` 训练进程 + 其它工作树单测）**：`$D/exp-recipe-stability-20261112/closed-loop/` 已评完的 5 个臂 `lvl-0-s0`、`lvl-0-s1`、`lvl-m0.5-s0`、`lvl-p0.5-s0`、`x-R2e-s0`（5 × 20 = 100 集，均值 `33.0–42.5` 步/集）。
3. **进程启动偏移与逐进程/逐线程 CPU 采样**：对在跑的两条车道（`eval_closed_loop.py` PID `217223`、`266079` 及其子进程树）做 7 次只读采样（`ps` + `/proc/<pid>/stat` + `/proc/<gz_pid>/task/<tid>/stat`），相邻两次稳态配对窗口分别为 `45.125 s`（`sample3→sample4`）和 `30.244 s`（`sample6→sample7`），跨集累计回收窗口为 `270.334 s`（`sample1→sample6`）；另由 `n=10` 个在跑集的 `/proc/<pid>/stat` 第 22 列 `starttime` 精确算出一条车道内各子进程的启动先后时刻。
4. **单函数微基准（纯 CPU、单进程、`nice -n 15`）**：测量 `eval_closed_loop.episode_shortest_path_m`、`teacher.warm_free_cells`（`detail_checks.json`）以及 320×240 帧 `json.loads`、`decode_depth`、`write_rgb_png`（`extra_checks.json`）。

### 1.2 单集墙钟拆分表（柱阵 K=1 学生评测，`--action-latency 0.20`）

注：`eval.json` 的 `per_episode[].wall_s` 从 `_start_sim` 之后起算，包含 `episode_shortest_path_m`，不含集前 `scene_gen.py` 子进程（约 `0.13 s`，由 `(arm_wall_s - sum(wall_s)) / 20` 核实：`s0` 为 `(1109.32 - 1107.0) / 20 = 0.12 s`）。

| 阶段 | 细项与代码位置（`main` `1d92be0` / `3a1db2f`） | 低竞争 60 集均值（K=1，31.3 步） | 占单集比例 | 满载限流 100 集范围（K=1，33–42 步） | 依据与说明 |
| --- | --- | --- | --- | --- | --- |
| **集前生成** | `scene_gen.py` 子进程生成 `.sdf` 与 `.json`（`eval_closed_loop.py:826/879`） | **0.13 s** | 0.2% | 0.15–0.18 s | `.cg` 臂总墙钟减 `sum(wall_s)` 除以 20 集 |
| **启动段 ①** | `gz sim -s -r` 启动到首帧就绪 `cp.wait_ready`（`collect_parallel.py:350/350`） | **5.73 s** | 10.6% | 5.5–6.0 s | `/proc/<pid>/stat` 实测 `collect_minus_gz_s` 均值 `8.73 s`（`n=10`）减去固定 `3.00 s` |
| **启动段 ②** | `time.sleep(cp.SIM_SETTLE_S)` 硬编码静置（`eval_closed_loop.py:875/928`） | **3.00 s** | 5.5% | **3.00 s** | 常量 `SIM_SETTLE_S = 3.0`，期间物理与 4 路相机空转 |
| **启动段 ③** | `collect.py` 导入、起 5 路订阅、读内参、`StudentPolicy` 冷启动 + `_warmup`、`warm_free_cells`（`collect.py:1750-1802`） | **2.37 s** | 4.4% | 2.1–2.8 s | `/proc` 实测 `state` 订阅晚于 `collect` 启动 `2.37 s`（`n=10`）：订阅与内参 `~0.8 s`、`uavrgb_infer` 启动预热 `~1.3 s`、K=1 栅格预热 `0.23 s`（K=3 为 `1.13 s`） |
| **起飞段 ④** | `cf_sitl` 建发布 helper、`reset`、`arm_takeoff 2.0` 爬升稳定至 `EXEC_READY`（`cf_sitl_exec.py:450-483`） | **7.32 s** | 13.5% | 8.5–12.0 s | 非步进墙钟 `20.22 s` 扣除 ①②③⑤⑥；仿真爬升约 `7.5 s`（首帧 `rgb_ts` 中位 `13.6 s`），按起飞段 `RTF≈1.0` 折算 |
| **每步循环：观测与推理** | `timing.io`：取缓冲 RGB（等帧 `0.0 ms`）+ `decode_depth`（`6.9 ms`）+ 配对位姿（`0.7 ms`）+ 学生缩放与推理（`5.8 ms`） | **0.42 s**（`13.4 ms/步`） | 0.8% | 0.6–1.0 s（`19–25 ms/步`） | `performance.mean_step_s.io`；C++ `infer_ms` 暖帧中位低负载 `1.00 ms`、满载 `5.0–6.0 ms` |
| **每步循环：等固定派发** | 等位姿仿真戳走到 `pose_before_ts + 0.20 s`（`collect.py:1036-1050/1069-1083`） | **3.00 s**（`96.0 ms/步`） | 5.5% | 3.3–5.1 s（`100–120 ms/步`） | `total - io - window - tail`；RGB 到达时已有约 `110 ms` 帧龄，故只需再等约 `80 ms` 仿真时间 |
| **每步循环：动作窗口** | `timing.window`：派发后并行算影子 A*（`~17 ms`）与写 RGB PNG（`4.8 ms`），轮询等仿真走满 `0.807 s`（`collect.py:1075-1178`） | **30.54 s**（`0.977 s/步`） | **56.3%** | 34.0–53.0 s（`1.02–1.27 s/步`） | `performance.mean_step_s.window`；时长由 `0.807 s / action_rtf` 决定（低竞争 `RTF=0.828`，满载 `0.65–0.80`） |
| **每步循环：步尾记账** | `timing.tail` + `actions.jsonl` / `oracle.jsonl` 行写出与 `flush` | **0.01 s**（`<0.3 ms/步`） | <0.1% | 0.01–0.02 s | `performance.mean_step_s.tail` 均值 `0.025 ms` |
| **集末收尾 ⑤** | `stop_group` 杀 `collect` 与 `gz sim` 进程组并 `ps` 轮询退出（`eval_closed_loop.py:934-938`） | **0.57 s** | 1.0% | 0.6–1.0 s | 每次 `stop_group` 至少跑 2–3 轮 `0.1 s` 睡眠加 `ps -eo pgid=,stat=` 子进程 |
| **集末复算 ⑥** | `episode_shortest_path_m` 对约 27 个终点跑 A* 算 SPL 的 `l_i`（`eval_closed_loop.py:806,954`） | **1.23 s** | 2.3% | 1.2–1.6 s | `detail_checks.json` 实测 K=1 为 `1.20–1.26 s`；**K=3 长课程实测 `5.56–6.23 s/集`**（解释了 K=3 非步进墙钟为何升至 `26.0 s`） |
| **单集合计** | 臂级单集平均墙钟（`arm_wall_s / 20`；括号内为 `eval.json` `wall_s`） | **54.28 s**（`54.15 s`） | **100.0%** | **60.1–84.9 s**（`60.0–84.7 s`） | 低竞争 60 集单集 `wall_s` 中位 `55.5 s`（p10 `38.5 s`，p90 `60.2 s`） |

### 1.3 单车道与容器的 CPU 拆分表

| 层级 / 进程（或线程） | 稳态活跃期核数（单车道，`sample6→7` 实测，高 RTF） | 满载竞争期核数（单车道均值，`sample3→4` 实测） | 占稳态单车道比例 | 说明与出处 |
| --- | --- | --- | --- | --- |
| **`gz sim -s -r` 进程合计** | **1.019 核** | **0.789 核** | **73.0%** | `/proc/<gz_pid>/stat` 的 `(Δutime+Δstime)/Δt` |
| ↳ 线程 `llvmpipe-0`（软件光栅化，`LP_NUM_THREADS=1`） | 0.419 核 | 未单拆 | 30.0% | 渲染 `chase`、`top`、`camera`、`depth` 共 4 路传感器（`7.30 MPix/s`） |
| ↳ 线程 `ruby` 主物理循环（`tid == pid`） | 0.277 核 | 未单拆 | 19.8% | 1 kHz DART 刚体/接触求解 + `CfSitlSystem::PreUpdate`（固件 tick + 每 20 ms 挪 `chase_cam` + 200 Hz 发 state） |
| ↳ 线程 Ogre2 传感器更新调度 + 5 个传感器线程 | 0.274 核 | 未单拆 | 19.6% | `tid 332339`（`0.105 核`）+ `tid 332434..332438`（合计 `0.169 核`）：图像拷贝、Protobuf 序列化与发布 |
| ↳ 线程 ZeroMQ 传输（`ZMQbg/IO/0` 等 3 线程） | 0.043 核 | 未单拆 | 3.1% | 跨进程发出图像、位姿、state 消息 |
| **6 个 `gz-transport-topic` 订阅子进程合计** | **0.258 核** | **0.216 核** | **18.5%** | 每个进程把二进制 Protobuf 转 JSON 文本写管道，约一半 CPU 在内核态 `stime` |
| ↳ `/e1_mini/state`（200 Hz，40 个 double） | 0.059 核 | 0.048 核 | 4.2% | 频率最高，JSON 行约 600 B × 200/s |
| ↳ `/model/e1_mini/pose`（200 Hz Pose_V） | 0.057 核 | 0.046 核 | 4.1% | 200 Hz 位姿流 |
| ↳ `/uav/depth`（10 Hz，320×240 float32 base64） | 0.047 核 | 0.039 核 | 3.4% | 每帧 JSON `400.1 KB`，`4.0 MB/s` 管道文本流 |
| ↳ `/uav/camera`（10 Hz，320×240 RGB base64） | 0.046 核 | 0.038 核 | 3.3% | 每帧 JSON `300.1 KB`，`3.0 MB/s` 管道文本流 |
| ↳ `/uav/camera_info`（20 Hz，RGB+depth 各 10 Hz） | 0.029 核 | 0.025 核 | 2.1% | **起飞前读 1 次后全程闲置**，子进程与读线程却一直解析到集末 |
| ↳ `/uav/contact`（无碰撞时 0 Hz，ZMQ 心跳/发现） | 0.020 核 | 0.020 核 | 1.4% | 空转监听开销 |
| **`collect.py` 主进程（含 6 个读线程）** | **0.113 核** | **0.082 核** | **8.1%** | 6 路 `json.loads` + `decode_depth` + `write_rgb_png` + 影子 A* + 20 ms 轮询 |
| **`cf_sitl_exec.py --pub-server` 辅助进程** | **0.020 核** | **0.019 核** | **1.4%** | `/usr/bin/python3` 常驻动作发布桥 |
| **`uavrgb_infer` 常驻推理进程** | **0.007 核** | **0.004 核** | **0.5%** | CUDA 前向中位 `1.0 ms/步`，CPU 端极轻 |
| **单车道稳态合计 / 跨集全周期均值** | **1.397 核**（跨集 `~1.85 核` 低竞争） | **1.110 核**（跨集 `1.33 核` 满载） | **100%** | 低竞争 2 车道 `.cg` 实测容器共 `3.66–3.84 核`（含无竞争时更高 RTF 及冷启动峰值）；`Xorg :20` 全机仅 `0.004 核` |

- **未核实项与下次实验的精确记账埋点**：现成历史数据（`exp-action-latency-r2-20261111` 与 `exp-recipe-stability-20261112`）只有容器级 `.cg`（`usage_usec`），没有单集或逐进程的 CPU 记录；多池并行时 `.cg` 混入了训练池（3 个 `uavrgb_train` 实测共占 `2.48 核`）与单测进程的 CPU，因此历史数据里各臂单独吃了多少核属于**未核实**。下次实验不借助外部采样器即可精确记账的最小改法：（a）在 `collect.py` 写 `episode_manifest.json` 的 `performance` 时，记录 `resource.getrusage(resource.RUSAGE_SELF)` 与 `resource.RUSAGE_CHILDREN` 的 `ru_utime + ru_stime`（拿到 `collect.py` 与全部订阅/推理子进程的核秒）；（b）在 `collect_parallel.py:stop_group` 终止 `gz sim` 前读一次 `/proc/<sim.pid>/stat` 的 `utime + stime + cutime + cstime` 写进 `per_episode[].performance.sim_cpu_s`。两处共约 15 行标准库代码，零运行时开销。

---

## 2. 提速点与候选方向逐项判定

排序规则：不改变学生看到的图像、不改变派发时序与评测数值的提速点排在最前，内部按「单集墙钟收益 + 单车道降核收益」从高到低排列（`P-1` 至 `P-7`）；会改变学生输入图像或触碰时序协议的候选方向单独归入 `P-8`；最后两项（`P-9`、`P-10`）回答任务书顺带核查的 3DGS 加载缓存与 `tools/check.sh --python-only` 耗时拆分。

### P-1：同臂跨集复用常驻 `StudentPolicy` 与 `gz sim` 实例（候选方向 ③）

- **位置**：`sim/tools/eval_closed_loop.py:869-938,1402-1410`（`3a1db2f` 为 `:922-991,1472-1480`）；`sim/tools/collect_parallel.py:470-516`；`sim/tools/collect.py:1750-1806`（`:1802-1858`）；`sim/cf_sitl/gz/CfSitlSystem.cc:109-129`。
- **现在花多少**：每集重起一次 `gz sim` 等就绪 `5.73 s`（不含 `3.0 s` 静置）+ 新起 `collect.py`、建 6 个订阅并冷启动 `uavrgb_infer` 加载 CUDA 权重做 `_warmup` `2.14 s` + 集末 `stop_group` 杀进程组轮询退出 `0.57 s`，合计 **`8.44 s/集`**（占 K=1 单集 `15.5%`）。另有固件 `reset + arm_takeoff 2.0` 爬升稳定 `7.32 s/集`。
- **依据**：草稿 `analysis_summary.json` 的 `proc_startup_offsets`（`n=10` 个在跑集 `/proc/<pid>/stat` 启动时刻差）与 `gz_threads_sample6.txt`（`llvmpipe-0` 渲染线程在 `gz sim` 启动后 `+2.34 s` 才初始化完成）；`nm -D` 核实 `libgz-sim-sensors-system.so` 已实现 `ISystemReset`，`CfSitlSystem.cc:116-124` 已实现运行期 `reset <episode>` 清空动作队列并重设 `base_s_`。
- **改法**：分两层推进，互不阻塞：
  1. **第一层（进程内复用 `collect` 与 `StudentPolicy`，约 25 行）**：`eval_closed_loop.py` 开头探针已经起过一次 `StudentPolicy(args.ckpt, infer_bin)`（`:1402`），改为不关闭它，把 `collect.py` 主循环作为模块函数在同一 Python 进程内调用，每集开头调 `student.reset()` 清空双帧与上一步动作缓存（与 `sim/3dgs/eval_3dgs.py:265,274` 现行做法一致）。这直接省掉每集 fork Python、重导库和 20 次 CUDA 上下文冷启动。
  2. **第二层（同任务跨集复用 `gz sim` 与话题订阅，约 80 行）**：柱阵任务每集最多 9 根柱子（不足 9 根的槽位放到地下 `z=-10`），模板已加载 `gz-sim-user-commands-system`（`uav_rgb_e1.sdf:37`）。集间不杀 `gz sim` 和常驻订阅，经 `/world/uav_rgb/set_pose` 更新柱子与 `e1_mini` 出生位姿、更新太阳方向光，然后照旧走 `CfSitlExecutor` 的 `reset <ep>` + `arm_takeoff 2.0`（**保留起飞段**：因为固件从 50% 电量起飞消耗约 7.5 s 后的电压压降、高度环积分器初值都是物理状态的一部分，跳过起飞直接空中悬停会改变动力学初态）。注意：`appearance=v1` 每集随机化 `<scene>` 的 `ambient` 与 `background`（`scene_gen.py:398-401`），必须通过 Gazebo 场景服务同步更新天空背景色与环境光，否则画面像素会变。
- **估计收益**：第一层单集省 **`1.8–2.0 s`**；第一层加第二层合计单集省 **`8.0–8.5 s`**（单集快约 `15%`，600 集 2 车道省约 `40–43 分钟`），同时消除每集启动 `gz sim` 时的瞬时多核毛刺。
- **会不会改变协议或数值**：第一层完全不改图像、物理与时序，数值不变。第二层若未同步更新 `v1` 的天空 `background` 与 `ambient`，会改变学生看到的图像（违反无损前提，此时只上第一层）；若通过场景服务同步更新且调世界重置清零 DART 接触态、保留完整 `reset + arm_takeoff`，则不改变图像与派发时序。
- **怎样验证没有改变**：取 5 个冻结 seed 顺序连跑，把第 2..5 集出生静止处渲染的帧与单起 `gz sim` 对应 seed 的帧逐像素比对（要求 `max|Δpixel| == 0`），并在 20 个冻结 seed 上核对 `safe_gap_s_median == 0.100 s`、巡航前飞位移差 `< 1%`、逐集成功率一致率不低于同配置重评基线（`0.75–0.80`）。
- **状态与建议归属**：冷启动耗时为**已复现**；第一层改法为**读代码确认**；第二层场景背景动态更新为**推断**（需写 20 行探针验证 Ogre2 服务端是否响应运行期 `Scene` 背景修改）。建议新立卡 `P12-PERF-1`（第一层立即做，第二层探针验证后做）。

### P-2：评测默认开启 `--no-aux-cameras` 并在读完内参后关闭 `/uav/camera_info` 订阅

- **位置**：`sim/worlds/uav_rgb_e1.sdf:86-110,399-439`；`sim/cf_sitl/gz/CfSitlSystem.cc:182-200,245`；`sim/tools/eval_closed_loop.py:1195`（`3a1db2f` 为 `:1263`）；`sim/tools/collect.py:556,1754`（`:589,1807`）。
- **现在花多少**：所有已跑和在跑的闭环评测臂（`exp-action-latency-r2-20261111` 全部 12 个臂、`exp-recipe-stability-20261112` 在跑全部臂）均未传 `--no-aux-cameras`（`eval.json` 顶层 `"aux_cameras": true`）。模板里 4 路渲染传感器总吞吐为 `7.296 MPix/s`，其中 `chase_camera`（640×360@15Hz = `3.456 MPix/s`）与 `top_camera`（640×360@10Hz = `2.304 MPix/s`）占了 **`78.9%` 的像素量**。反汇编 `libgz-sensors8-camera.so` 的 `CameraSensor::HasConnections()` 查明：即使没人订阅 `/uav/chase_camera` 与 `/uav/top_camera` 图像话题，4 路相机因共享 `/uav/` 前缀而共用同一个内参话题 `/uav/camera_info`；`collect.py:556` 全集保持 `TopicSub("/uav/camera_info")` 常驻连接，导致两路辅助相机的 `HasInfoConnections()` 恒为真，迫使 `llvmpipe-0` 全程满负荷渲染这两路高清画面！此外，`CfSitlSystem` 每 20 ms 调一次 `PlaceChaseCam` 改写 ECS，闲置的 `camera_info` 订阅子进程白占 `0.025–0.029 核`。这使软件渲染下 `action_rtf` 从接近 `1.0` 掉到 `0.82`（满载时 `0.65`），31.3 步窗口墙钟从理论 `25.3 s` 膨胀到 `30.5 s`（满载 `39–42 s`）。
- **依据**：（1）`gz_threads_sample6→7` 实测 `llvmpipe-0` 占 `0.419 核`（`gz sim` 第一大线程）、`camera_info` 子进程占 `0.029 核`；（2）`extra_checks.json` 读取已有实跑数据：单车道锁 RTF 1.0 下 `aux` `1.09 s/步`（`RTF=0.85`）对 `noaux` `0.90 s/步`（`RTF=1.01`）（`2026-11-07-0140-gz-e1-mini.md:48-49`）；`exp-2026-11-12-analytic-depth-parity` 带 `--no-aux-cameras` 实测 `action_rtf = 0.9625`、`0.941 s/步`，`39/39` 个步对 `frame_dt == 0.9 s`；（3）`longcourse.md:253` 之前测出 `--no-aux-cameras` 缩短生效时长，是因为当时在**无 `--action-latency` 的旧协议**下跑（RTF 升高把 Python A* 墙钟折成了更多仿真延迟），而在新协议 `--action-latency 0.20` 下派发已锚定在仿真时刻 `pose_before_ts + 0.20 s`。
- **改法**：（1）`eval_closed_loop.py`（及各评测入口脚本）在 `vehicle == "e1_mini"` 且 `not args.record_video` 时默认启用 `--no-aux-cameras`（保留 `--with-aux-cameras` 供桌面调试；`CfSitlSystem::PlaceChaseCam` 已在首步找不到 `chase_cam` 时直接返回，无需改 C++）；（2）`collect.py` 在起飞前 `read_camera_info` 返回后立刻关闭并移除 `subs["camera_info"]`。共约 12 行。
- **估计收益**：单集墙钟在低竞争下省 **`4.5–5.5 s`**（起飞段与步进段的 `RTF` 从 `0.82` 回到 `0.96–1.00`），在当前满载竞争下省 **`8.0–12.0 s/集`**；单车道稳态 CPU 下降 **`0.25–0.35 核`**（直接为 `P-7` 加车道腾出核数）。
- **会不会改变协议或数值**：不改变。`chase_cam` 与 `studio_top_camera` 均为无碰撞、无质量的 `static` 纯视觉传感器，删掉后物理步、光照、阴影与 `/uav/camera` 机载图像不变；保留现有订阅通道时 `k1-noaux`（60/60 步）与 `analytic-depth-parity`（39/39 步）已验证 `frame_dt` 恒为 `0.900 s`，派发与截止时序不变。
- **怎样验证没有改变**：（1）同 seed 对比开关 `--no-aux-cameras` 下静止首帧 `rgb_00000.png` 逐字节一致；（2）在 `--action-latency 0.20` 下跑 5–10 集，检查 `eval.json` 中 `safe_gap_s_median == 0.100`、`obs_gap_s_max == 0.900`、`late_dispatch_steps == 0`、`stall_steps == 0`。
- **状态与建议归属**：**已复现**（数据盘三组 `noaux` 实跑记录 + `libgz-sensors8` 反汇编确认）。现成开关已就绪，建议排在最前立即采用（归入 `P12-PERF-1` 或实验入口模板 `experiments/_template/run.sh`）。

### P-3：去掉 `cf_sitl` 的 `SIM_SETTLE_S = 3.0 s` 死等、裁剪集末 SPL 重复 A* 与评测期无用的影子教师 A*

- **位置**：`sim/tools/collect_parallel.py:53,350-368,472`；`sim/tools/eval_closed_loop.py:336-346,806,874-875`（`3a1db2f` 为 `:369-379,859,927-928`）；`sim/tools/collect.py:977-991,1097-1107,1795-1802`（`:1009-1023,1132-1142,1848-1855`）。
- **现在花多少**：三处纯开销合计占 K=1 单集 **`4.7–5.0 s`**、占 K=3 长课程单集 **`9.5–10.5 s`**：
  1. `time.sleep(cp.SIM_SETTLE_S)` 每集硬睡 **`3.00 s`**（旧 X3 速度控制器防丢首包的补丁；`cf_sitl` 的 `_reset_until_ack` 已有 `1.0 s` 幂等重发确认，不需要这 3 s）。
  2. `cp.wait_ready` 串行起 3 次 Ruby `gz topic` CLI（`-l` 加两次 `-e -n 1`），探针本身花 **`0.8–1.0 s/集`**。
  3. `eval_closed_loop.episode_shortest_path_m` 每集结束对 `x=goal_x` 上约 27 个 `y` 候选终点循环跑 `teacher.astar_path`：`detail_checks.json` 实测 K=1 耗 **`1.20–1.26 s/集`**，**K=3 耗 `5.56–6.23 s/集`**！此外，纯学生闭环评测（`eval_closed_loop --policy student`，不打包训练数据）每集仍花 `0.23 s`（K=1）/ `1.13 s`（K=3）预热 `warm_free_cells` 并在每步跑影子教师 A*，而评测报告根本不读 `action_expert`。
- **依据**：代码逐行核对与 `detail_checks.json` 在 `seed 1`、`seed 113`（K=1 与 K=3）上的离线实测计时；K=3 与 K=1 在 `r2` 数据里的非步进墙钟差（`26.0 s - 20.2 s = +5.8 s`）与 `warm_free_cells` + `episode_shortest_path_m` 的实测增量（`+0.90 s + 4.67 s = +5.57 s`）吻合。
- **改法**：
  1. `dynamics == "cf_sitl"` 时跳过 `time.sleep(cp.SIM_SETTLE_S)`；`wait_ready` 改为单次轻量探针（或直接交由 `collect.py` 内部的常驻订阅 `wait_ready` 等待首帧）。
  2. `teacher_shortest_path_m` 优化且保持**逐位严格等价**：先调一次 `warm_free_cells(pillars, clearance=clearance, step=step)`，按离 `(traj[0][0], traj[0][1])` 横向更近的候选终点排序，并用八邻接 octile 距离下界剪枝（若某候选 `goal` 到起点的无障碍 octile 路长已 `>= best_grid_len` 则直接跳过 `astar_path`，因为有障碍栅格路长不可能短于无障碍 octile 路长）；若想更彻底，跑一次从 `traj[0]` 出发的 Dijkstra 求全终点精确 `gscore`。
  3. `collect.py` 增加可选开关 `--no-shadow-teacher`（默认关，DAgger 采集照旧算标签；仅 `eval_closed_loop.py --policy student` 在不需要专家对比列时开启，跳过 `warm_free_cells` 与每步影子 A*）。
- **估计收益**：K=1 单集省 **`4.5–5.0 s`**，**K=3 长课程单集省 `8.5–9.5 s`**（600 集 K=1 2 车道省 `23–25 分钟`，K=3 省 `42–48 分钟`），并去掉 `collect.py` 内的 A* CPU 消耗。
- **会不会改变协议或数值**：不改变。`CfSitlSystem` 在收到 `reset` 时才建立 `base_s_ = sim_s` 零点，去掉 `3.0 s` 静置不改变固件内部时间线、障碍物或光照；octile 下界剪枝在数学上保证 `teacher_shortest_path_m` 返回值与原实现逐位相同；`--no-shadow-teacher` 只在纯学生评测里省掉不参与控制也不进 `eval.json` 的 `action_expert`。
- **怎样验证没有改变**：（1）在 20 个冻结 seed（K=1 与 K=3 共 40 个场景）上断言新旧 `teacher_shortest_path_m` 返回值逐位相等；（2）单测断言 `--no-shadow-teacher` 下学生每步发出的动作、概率与时序不变；（3）5 集冒烟核对 `safe_gap_s_median == 0.100` 与位移分布。
- **状态与建议归属**：**已复现 / 读代码确认**。改动小（约 25 行）、无外部依赖，建议并入 `P12-PERF-1` 第一批落地。

### P-4：学生评测与真值规划臂按需关闭深度相机渲染、订阅与解码（候选方向 ②）

- **位置**：`sim/worlds/uav_rgb_e1.sdf:356-374`；`sim/tools/scene_gen.py:98-101,323`；`sim/tools/collect.py:551,577,885-898,939,1020,1112`（`3a1db2f` 为 `:584,610,918-931,972,1055,1147`）；`sim/tools/audit_dataset.py:603-620`。
- **现在花多少**：
  1. 在 `P-2` 去掉 `chase/top` 后，世界里剩 `/uav/camera` 与 `/uav/depth` 两路相机，深度相机（320×240@10Hz，`R_FLOAT32`）占了**剩余软件渲染通道的 50%**。
  2. `/uav/depth` 的 `gz-transport-topic --json-output` 子进程每秒序列化 `4.0 MB` base64 JSON（`400.1 KB/帧`），稳态占 **`0.039–0.047 核/车道`**。
  3. `collect.py` 每步无条件调 `decode_depth`（base64 解码加 240 次 `struct.unpack("<320f")`），`extra_checks.json` 微基准实测耗时 **`6.92 ms/步`**，占学生每步 `timing.io`（`13.4 ms`）的 **`51.6%`**。
- **依据（逐类核查谁真正用深度图）**：全仓检索 `decode_depth`、`depth_ts`、`depth_file`、`expert_action` 确认：
  - **不需要深度图的路径**：（a）学生闭环评测（`eval_closed_loop --policy student`），学生只看 RGB，影子教师 `plan_action_global_tiebreak` 只吃真值位姿与 `scene.json` 的 `pillars`（`collect.py:984`），完全不读 `d`；（b）不带 `--save-depth` 的 DAgger 采集，以及带解析深度开关 `preprocess.py --depth-source analytic`（卡 W1-8 `e41703e`）的采集；（c）带 `--scene-meta` 的特权教师臂（`astar` / `costmap` / `predictive`）、规则上限臂（`rule-oracle`）与盲飞臂（`forward`），全部只用真值位姿与 `pillars`。
  - **真正需要深度图的路径仅四处**：（a）未传 `--scene-meta` 退回深度反应式专家 `expert_action`；（b）`--policy teacher-nn`；（c）显式传 `--save-depth` 落盘深度 PNG；（d）需要过 `audit_dataset.py` 第 2 条 `time_pairing` 门禁的训练采集数据集（`audit_dataset.py:603` 要求每步都有非空 `depth_rgb_gap_ns`）。
- **改法**：`scene_gen.py` 加 `--no-depth-camera` 剥离 `<sensor name="depth">` 传感器块；`collect.py` 加 `--no-depth`（与 `--save-depth`、`teacher-nn`、无 `scene-meta` 的 `teacher` 互斥，启动即校验报错；开启后不起 `/uav/depth` 订阅、不调 `decode_depth`，`depth_ts` 与 `depth_rgb_gap_ns` 填 `None`）；`eval_closed_loop.py` 在 `policy in ("student", "forward", "rule-oracle", "teacher")` 且 `not args.save_depth` 时默认启用（训练采集 `collect_parallel.py` 默认保持原样，不动 `audit_dataset.py` 契约）。
- **估计收益**：单车道稳态 CPU 再降 **`0.12–0.18 核`**（软件渲染少一半通道省 `~0.08–0.12 核` + 省掉 `/uav/depth` 子进程 `0.047 核` + 省掉 Python 端 `4 MB/s` JSON 解析与 `6.9 ms/步` 解码）；学生每步 `timing.io` 从 `13.4 ms` 降到 **`~6.5 ms`**；多车道并发受渲染制约时单集墙钟再省 **`1.5–3.0 s`**。
- **会不会改变协议或数值**：不改变学生看到的 `/uav/camera` 图像；需防范的唯一耦合是去掉同频深度渲染后 RGB 帧到达 Python 变快约 `5–15 ms`，必须确认没有越过窗口末 `+1.005 s` 位姿与 `+1.000 s` RGB 之间的 `5 ms` 相位边界（见 `P-5`）。
- **怎样验证没有改变**：（1）同 seed 对比开关 `--no-depth-camera` 时 `rgb_00000.png` 逐字节相同；（2）跑 10 集学生评测，核对 `obs_gap_s_max == 0.900`、`safe_gap_s_median == 0.100`、`late_dispatch_steps == 0`。
- **状态与建议归属**：`decode_depth` 与订阅开销**已实测确认**，调用链依赖**读代码确认**。建议归入性能卡 `P12-PERF-1`（排在 `P-2`、`P-3` 之后同批或次批落地）。

### P-5：`gz_io.py` 改用进程内 `gz.transport13` 原生订阅，消除 7 个通讯子进程与 JSON/base64 编解码（候选方向 ①）

- **位置**：`sim/tools/gz_io.py:59-157`；`sim/tools/collect.py:548-561,877`（`3a1db2f` 为 `:581-594,910`）；`sim/tools/cf_sitl_exec.py:207-287,452`（`:214-294,459`）。
- **现在花多少**：每条车道在集内常驻跑着 **6 个** `gz-transport-topic -e --json-output` 子进程与 **1 个** `cf_sitl_exec.py --pub-server` 子进程。`analysis_summary.json` 实测 6 个订阅子进程合计占 **`0.258 核/车道`**（占单车道总核数的 `18.5%`，近一半是管道读写的内核态 `stime`），加上 `pub-server`（`0.020 核`）和 `collect.py` 6 个后台读线程解析 `7.0 MB/s` JSON+base64 字符串（约 `0.05–0.07 核`），跨进程文本桥合计消耗 **`0.33–0.35 核/车道`**（2 车道约 `0.7 核`）。
- **依据**：只读探测确认系统 `/usr/lib/python3/dist-packages` 已安装针对本机 Python `3.12.3` 编译的 `gz.transport13` 与 `gz.msgs10`（含 `image_pb2`、`pose_v_pb2`、`contacts_pb2`、`double_v_pb2`、`camera_info_pb2`）；`.venv-tools` 解释器就是同一个 `/usr/bin/python3.12`（因 `include-system-site-packages = false` 默认被隔离），在 `.venv-tools` 中挂入系统 `gz` 包后可直接 `Node().subscribe(...)` 拿二进制 Protobuf `msg.data` 字节，同时 `cf_sitl_exec.open_publisher()` 会自动切入已有的 `_InProcPublisher`（不再起 `pub-server` 子进程）。
- **关键时序耦合（必须同步处理）**：现行协议下第 `k` 步 `rgb_ts = t_0 + 0.9k`、`pose_before_ts = t_0 + 0.9k + 0.005 s`，窗口末等位姿走到 `hold_end_ns = t_0 + 0.9k + 1.005 s` 后，下一步立刻调 `subs["rgb"].latest_after(prev_rgb_ts)`。目前之所以 `4,972/4,972` 步对恒有 `frame_dt = 0.900 s`，正是因为 `t_0 + 0.9k + 1.000 s` 刚拍下的那帧 RGB 被 `gz-transport-topic` 的 base64+JSON 管道拖慢了 `70–175 ms`，在 `1.005 s` 位姿到达时尚未进缓冲！若直接把 `/uav/camera` 换成 `< 2 ms` 延迟的进程内二进制订阅而不加约束，一旦软件渲染在 `~5–20 ms` 内跑完，`latest_after` 就会抓到 `+1.000 s` 那一帧，使 `frame_dt` 从 `0.900 s` 跳到 `1.000 s`、`safe_gap_s` 从 `0.100 s` 变成 `0.200 s`（单步前飞位移缩减约 `16%`）！
- **改法**：
  1. 在 `.venv-tools` 的 `.pth`（或 `gz_io.py` 局部路径）引入系统 `gz` 包；`gz_io.TopicSub` 优先使用进程内 `gz.transport13.Node().subscribe`（直接存轻量结构或原始 Protobuf 字节，按需转 dict，无绑定时回退原 CLI）。
  2. **防相位跳变**：在卡 W1-13 第三部分（「动作持续到下一次派发」）落地前，若要在现行 `0.900 s` 步周期协议下启用进程内订阅，`collect_one` 在 `prev_until_ns is not None and window_from == "observation"` 时应从缓冲中选取时间戳 `< prev_until_ns - 50_000_000`（即窗口截止前拍的 `prev_rgb_ts + 0.9 s` 帧）的最新帧；等 W1-13 第三部分把步间空档消除并重标定后，则可直接取最新帧。
- **估计收益**：单车道消除 7 个子进程，稳态 CPU 下降 **`0.28–0.33 核`**（与 `P-4` 部分重叠在 `/uav/depth` 子进程的 `0.047 核`，两者叠加净省约 `0.38–0.45 核/车道`），每集起停订阅还省约 `0.4 s`。
- **会不会改变协议或数值**：不加相位上界时会因图像提早到达而改变 `frame_dt` 与单步位移（属改变协议数值）；加上相位上界（或随 W1-13 第三部分重标定一起上）后，图像字节与派发时序均不改变。
- **怎样验证没有改变**：（1）单测喂同一帧 Protobuf 消息，断言进程内解码得到的 `(w, h, nch, rows)` 与 JSON 路径逐字节一致；（2）实跑 20 集核对 `obs_gap_s` 全部为 `0.900 s`、`safe_gap_s_median == 0.100 s`、巡航位移差 `< 1%`。
- **状态与建议归属**：开销与绑定可用性**已复测确认**；`5 ms` 竞态由代码与 `actions.jsonl` 时间戳确认。建议并入卡 W1-13 第三部分或单独立卡 `P12-PERF-2`。

### P-6：评测临时目录跳过每步 RGB PNG 压缩写盘，且定量评测与三视图录像分离（候选方向 ⑤）

- **位置**：`sim/tools/collect.py:1110-1117`（`3a1db2f` 为 `:1145-1152`）；`sim/tools/eval_closed_loop.py:1012-1018,1110-1118,1465-1468`（`:1065-1071,1163-1171,1539-1542`）；`tools/e1_pipeline.py:325`。
- **现在花多少**：
  1. **每步 `rgb_%05d.png` 落盘**：`eval_closed_loop.py` 默认不开 `--keep-work`，集末 `keep_episode_logs` 只保留 `actions.jsonl`、`oracle.jsonl`、`episode_manifest.json`（「帧不留」），随后把临时目录里刚写出的 30–80 张 PNG 全部删掉。`extra_checks.json` 对真实评测帧微基准实测：`write_rgb_png`（`zlib` level 6，均值 `5.8 KB/张`）耗时 **`4.84 ms/步`**（一集约 `0.15 s` CPU），且位于 `sil.dispatch` 之后的 `0.98 s` 窗口睡眠期内，因此**对不录像评测的墙钟影响为 `0.0 s`**（仅占 `~0.005 核` 与目录元数据 I/O）。
  2. **`--record-video` 三视图录像**：当前排队的批量实验（`r2`、`recipe-stability`、`e2e-heading-code`）都没开它；但流水线 `tools/e1_pipeline.py:325` 默认评测开着它。`extra_checks.json` 读 `exp-viz3-20261110/gz-base/eval/eval.json`（`seed 113`，30 步）实测：开 `--record-video` 后单集 `wall_s` 达 **`115.0 s`**（比不录像的 `55 s` **慢 `60 s/集`，即 `2.1 倍`**，`action_rtf` 跌至 `0.330`），原因是保留并放大辅助相机分辨率、多起 3 路图像订阅实时写原始帧（`~10 MB/仿真秒`）、集末 `ffmpeg` 合成视频。
- **改法**：（1）`collect.py` 加 `--no-save-rgb`（仅校验 `save_frame.decode(rgb_msg)` 置 `rgb_ok=True`，不压缩写盘；`eval_closed_loop` 在 `not args.keep_work and not args.record_video` 时自动传）；（2）`e1_pipeline.py` 把 20/60 集定量评测改为 `--no-aux-cameras` 不录像，需要视频时另对 2–3 个固定 seed 单跑录像任务。
- **估计收益**：对当前不录像的批量实验墙钟收益约 **`0 s/集`**（只减磁盘小文件擦写）；对 `e1_pipeline.py` 流水线评测省 **`60–70 s/集`（评测阶段提速约 `2×`）**。
- **会不会改变协议或数值**：`--no-save-rgb` 在动作派发之后，不改任何数值；关掉 `--record-video` 恢复高 `RTF`，在 `--action-latency 0.20` 下与 `P-2` 同口径。
- **怎样验证没有改变**：单测锁住 `--no-save-rgb` 下 `validity.camera_verified == True` 且 `actions.jsonl` 行内容逐字不变。
- **状态与建议归属**：微基准与 `viz3` 实跑日志**已核实**。优先级低于 `P-1..P-5`，顺手并入流水线维护即可。

### P-7：单车道降核后将 Gazebo 池从 2 车道扩至 3–4 车道（候选方向 ⑥）

- **位置**：公共规则 `docs/briefs/2026-11-10-phase12-common.md:62-77`；`tools/exp_lib.sh`（卡 W1-15a `f2c1107`）；各闭环实验入口的 `LANES=2` 配置。
- **现在花多少（为什么现在不能直接加车道）**：容器配额 12 核。未优化前单车道稳态占 `1.40 核`、低竞争跨集均值约 `1.85 核`（`r2` 两车道独跑时容器占 `3.66–3.84 核`）。当训练池（3 个 `uavrgb_train` 实测共 `2.48 核`）、3DGS 池（`3–4 核`）与 `check.sh`（`1 核`）并行时，今天在跑的 `exp-recipe-stability-20261112` 仅开 2 条车道就已经把容器推到 `8.85–11.73 核`，每臂 `nr_throttled` 暴涨 `1,253–12,805` 次、`throttled_usec` 达 `3,615–19,218 s`（`analysis_summary.json` `rs_cg`），单集墙钟从 `54 s` 被拖慢到 `60–85 s`。在未降单车道 CPU 前盲目开 3–4 车道，只会触发更重的内核 CFS 限流，触发 W1-13 的迟到步/停顿步重跑，净吞吐不增反降。
- **降核后的新 CPU 账与改法**：
  1. 落地 `P-2`（关辅助相机与 `camera_info`，省 `~0.30 核`）+ `P-4`（关深度相机，省 `~0.15 核`）+ `P-5`（进程内 `gz.transport13`，扣除重叠净省 `~0.25 核`）后，单车道稳态 CPU 从 **`1.40 核` 降至约 `0.70–0.75 核`**（含冷启动的跨集均值从 `1.85 核` 降至 **`1.00–1.10 核`**）。
  2. 此时开 **3 条车道仅占 `3.0–3.3 核`**，比优化前 2 条车道的 `3.7 核` 还低！即使训练池（`2.5 核`）+ 3DGS 池（`3.5 核`）+ `check.sh`（`1.0 核`）同时满跑，全容器合计 `10.0–10.3 核 < 12 核`，仍留有约 `2 核` 缓冲；当训练池或 3DGS 池任一空闲（起跑前 30 s 窗口非 Gazebo 占用 `<= 5.5 核`）时，可自适应放行 **4 条车道（约 `4.0–4.4 核`）**。
- **估计收益**：单集墙钟经 `P-1..P-4` 从 `55–75 s` 压到 **`35–40 s`** 后，车道数 `2 → 3` 再省 `33%` 墙钟、`2 → 4` 再省 `45–50%` 墙钟。合算下来，600 集闭环实验总墙钟从现在的 **5.4 小时以上** 降到 **`1.6–2.1 小时`**（整体提速约 **`2.6–3.3 倍`**）。
- **会不会改变协议或数值**：各车道通过独立 `GZ_PARTITION` 隔离；只要容器未发生严重 CFS 限流，图像与时序完全不变。万一外部突发负载抢核，卡 W1-13 已合入的 `late_dispatch_steps` 与 `stall_steps` 会将该集拦入 `failed[]` 按规则重跑，不会污染成功率数字。
- **怎样验证没有改变**：用卡 W1-13 的 `experiments/2026-11-12-w1-13-newprotocol/` 在 `LANES=3` 与 `LANES=4`（配并发训练负载）下实测：要求 `late_dispatch_steps == 0`、`stall_steps == 0`、`plugin_ack_sim_t` 偏差 `< 15 ms`、`.cg` 的 `throttled_usec / arm_wall_s < 0.05`，核实通过后再改 `exp_lib.sh` 默认车道数。
- **状态与建议归属**：现行 2 车道限流数据**已实测确认**，降核后 3–4 车道容量为基于实测分项核数的**推断**（成立条件：`P-2`、`P-4`、`P-5` 先落地且无额外进程突破 12 核配额）。建议作为性能卡第二阶段（等 `P-2..P-5` 合入后开闸验证）。

### P-8：【单列：会改变图像或时序协议，不作无损提速采纳】改相机分辨率、改相机帧率、解锁 `real_time_factor`（候选方向 ④ 等）

以下三个方向经常被想到，但经代码与已有实验数据核实，它们都会改变学生看到的像素或改变派发时序，**不属于「不改变协议与数值」**：

1. **把 Gazebo 相机渲染分辨率从 `320×240` 降到网络输入 `160×120`**（位置：`uav_rgb_e1.sdf:341-345`、`student.py:117`、`eval_closed_loop.py:82`）：
   - **为什么改变数值**：现在 Ogre2 单采样光栅化 `320×240` 无抗锯齿原图，再由 `PIL` 双线性缩放到 `160×120`（等价于 `2×2` 超采样抗锯齿）。若直接渲染 `160×120`，远柱边缘会出现锯齿跳变，改变输入张量像素值，且违反冻结协议键 `CAMERA_PROFILE = "rgb320x240_fov60_v1"`。
   - **为什么不值得做**：在 `P-2` 与 `P-4` 关掉 `chase/top/depth` 后，唯一的 `320×240@10Hz` RGB 相机每秒仅 `0.768 MPix`（占 `llvmpipe-0` 不到 `0.06 核`），降到 `160×120` 最多省 `~0.04 核`、单集省不到 `0.5 s`，收益微乎其微却破坏与全部历史训练数据的像素一致性。
2. **把相机帧率从 `10 Hz` 降到 `5 Hz`（或其它频率）**（位置：`scene_gen.py:277`、`uav_rgb_e1.sdf:337`）：
   - **为什么改变协议与数值**：`extra_checks.json` 读取 `exp-long-course-v2-20261110/calib/k3-noaux-cam5-s{1,113}` 实测确认，改成 `5 Hz`（周期 `0.200 s`）后，`207/207` 个步对的 `frame_dt` 全部从 `0.900 s` 变成 **`0.800 s`**！在 `--action-latency 0.20` 下，步间刹停空档 `safe_gap_s = L - (hold - frame_dt)` 从 `0.100 s` 变成 `0.000 s`，步周期缩短 `11%`、每步初速度与位移全部改变。同时 `10 Hz` 下单路 `320×240` 负担本就极小，不需要靠降帧率省算力。
   - **补充一处按需触发的设想**：`gz::sensors::Sensor::SetTriggered`（`Sensor.hh:234`）理论支持每步发信号只渲 1 帧，但 SDF 模板与 `collect.py` 都需改写握手，且会改变帧捕获相位，收益同样只有 `< 0.05 核`，不值得动。
3. **解锁世界 `<real_time_factor>`（设 `0` 或 `> 1.0`）**（位置：`uav_rgb_e1.sdf:33`、`scene_gen.py:287`）：
   - **为什么改变协议与数值**：`2026-11-07-0140-gz-e1-mini.md:39-61` 测得解锁后物理可跑到 `RTF = 1.7–2.3`（单步墙钟 `0.52 s`），但墙钟抖动折算成仿真时间的倍率也同步放大 `2.3 倍`，不仅容易让 A* 步冲破 `L=0.20 s + 0.02 s` 迟到线，还会因渲染线程跟不上物理线程导致 `depth_rgb_gap` 跳至 `100 ms`、`frame_dt` 在 `0.9 s` 与 `1.0 s` 间抖动。在 W1-13 第三部分彻底消除相机相位对运动学的影响并重做标定前，正式评测必须保持 `real_time_factor = 1.0`。

### P-9：【顺带核查 1】3DGS 线单次启动重读 PLY 与重建占据代理的缓存方案

- **位置**：`sim/3dgs/scene.py:38-115`（`read_gaussian_ply`）；`sim/3dgs/collision.py:252-427,534-605`（`Occupancy`、`from_scene`、`terrain_from_scene`）；`sim/3dgs/renderer.py:91-109`（`GsplatRenderer`）；`sim/3dgs/eval_3dgs.py:236-260`；`experiments/2026-11-10-bearing-guard-3dgs/driver.py:165-173`。
- **现在花多少（108 份 `driver-*.log` 实测汇总）**：
  - **大室外场景（PLY `1.17–1.45 GB`）**：`bicycle`（`1450.3 MB`）`driver.py` 场景加载中位 **`41.5 s`**（均值 `51.3 s`，磁盘/CPU 忙时最高 **`109 s`**，`n=12`）；`garden`（`1380.0 MB`）中位 **`44.0 s`**（最高 **`106 s`**，`n=12`）；`stump`（`1173.5 MB`）中位 **`43.5 s`**（最高 `57 s`，`n=4`）。
  - **单调 `eval_3dgs.py` 为何高达 `150–200 s`**：读代码发现 `eval_3dgs.py:236-253` 与 `driver.py:167-170` 都把同一个 `1.45 GB` PLY 重复读了 **3 遍**！（1）`scene_identity` / `_sha256` 全文件流式算 sha256 读第 1 遍；（2）`collision.from_scene` 调 `read_gaussian_ply` 读第 2 遍，做 600 万高斯的旋转椭球体素化与膨胀（`20–60 s`）；（3）`make_renderer` 内部 `GsplatRenderer.__init__`（`renderer.py:91`）不收已读好的字典，又调 `read_gaussian_ply` 读第 3 遍并重排 45 列球谐数组（`15–25 s`），最后拷上 GPU 本身只需 `< 0.6 s`！
  - **中型室外与室内场景**：`callinstrasse`（`488 MB`）中位 `17.5 s`、`wanderland`（`451 MB`）中位 `22.5 s`；5 个室内 `sage3d_*`（`93–328 MB`）中位 **`8.0–13.5 s`**。
- **不改数值的三步缓存法（缓存在 `$D/3dgs-scenes/.cache/`，不进仓库）**：
  1. **消掉同进程第 2、3 遍重复读（约 15 行，零磁盘占用）**：`GsplatRenderer.__init__` 增加可选参数 `gaussians=None`（与 `collision.from_scene` 对齐），`eval_3dgs` / `collect_3dgs` / `driver.py` 只调 1 次 `read_gaussian_ply` 同时喂给碰撞、地形层与渲染器；PLY 的 `sha256` 按 `(abspath, size, mtime_ns)` 缓存在 `.cache/ply_sha256.json`。
  2. **缓存占据栅格 `.npz`（磁盘占用极小，收益最大）**：以 `sha256(ply_sha256 + canonical_json(collision_cfg, nav_to_world, envelope, PROXY_VERSION))` 为键（卡 S1b 升级碰撞口径改 `PROXY_VERSION` 时自动失效，不会误用旧栅格），把 `Occupancy` 的 `raw`、`grid`、`_occ_xy`、`origin`、`shape` 与统计字段存成不压缩 `.npz`。9 个场景合计不到 `80 MB`，命中后 `coll.from_scene` + `terrain_from_scene` 从 **`25–60 s` 降到 `< 0.1 s`**！
  3. **按需 `mmap` 缓存大场景预处理高斯张量（注意数据盘配额）**：把过滤并转置好的 float32 数组（`means/quats/scales/opacities/sh`）存为 `.npy`，命中时 `np.load(..., mmap_mode="r")` 直传 `torch.from_numpy(...).to("cuda")`，省掉 48 列解析与球谐转置。注意：`bicycle` 单场景 3 阶球谐 `.npy` 约 `1.35 GB`，全缓 9 个场景约 `4.2 GB` 会触碰数据盘 3 GB 预算线；因此建议**默认只开第 1、2 步**（已把 `eval_3dgs` 从 `150–200 s` 压到 `12–18 s`），第 3 步设可选环境变量或 LRU 上限 `2 GB`（进一步压到 `2–3 s`）。
- **会不会改变协议或数值**：不改变。栅格与预处理张量均为确定性纯函数产物的原样序列化，键绑定文件哈希与版本号。
- **怎样验证没有改变**：单测对同一场景对比冷跑与命中缓存后的 `occ.raw`、`occ.grid`、`occ.summary()` 逐项相等，且 `renderer.render_pose` 的 `rgb`、`depth` 数组 `np.array_equal` 为真。
- **状态与建议归属**：重复读 3 遍 PLY **读代码确认**，各场景加载耗时由 108 份日志**已核实**。建议等卡 S1b（碰撞口径 v2）合入稳定后单独立小卡实施。

### P-10：【顺带核查 2】`tools/check.sh --python-only` 耗时 5–9 分钟的慢点拆分与提速法

- **位置**：`tools/check.sh:179-215`；`sim/3dgs/test_collect_eval.py`；`tools/test_preprocess_*.py`；`sim/tools/test_astar_cache.py`；`sim/tools/test_teacher_v7.py`。
- **现在花多少（真实日志逐项拆解）**：读取 `$D/tmp/phase12-cards/w1-14/check-python-only.log`（`HEAD=5de1a00`，起止 `07:26:33–07:35:37`，总墙钟 **`544 s` = 9 分 04 秒**，期间容器 `nr_throttled` 增 `3,661` 次）：
  - 全程串行启动约 **110 个独立 Python 进程**（`sim/tools/test_*.py` 60 个、`tools/test_*.py` 31 个、`sim/3dgs/test_*.py` 13 个、`ui/` 3 个及自检与门禁负例脚本）。其中 56 个 `unittest` 文件合计 **`374.1 s`**（占总时长 `68.8%`），其余非 `unittest` 脚本（`scene_gen`/`scene_course` 自检、`test_asset_campus.py` 160 世界校验、`test_collect.py` 108 项、打包负例等）加进程启动与限流占 **`169.9 s`**（`31.2%`）。此外，`08:03–08:11` 新合入的 `tools/test_exp_lib.py`（1,102 行）与 `sim/tools/test_step_gaps.py`（943 行）又增加了约 5 个重测试文件。
  - **最慢的前 10 个测试文件 / 簇**（已占全部 `unittest` 374.1 s 的 **`79.6%` = `297.6 s`**）：
    1. `sim/3dgs/test_collect_eval.py`（28 项）：**`98.47 s`**（单文件占整个 `3dgs` 段的 `65.1%`！原因是每个用例各自调 `make_corridor` 生成数万个表面高斯写二进制 PLY、重做椭球体素化，并起 20 余次 `python -m sim.3dgs.collect_3dgs / eval_3dgs` 子进程，每次重导 `numpy` 与重建 `Occupancy`）。
    2. `tools/test_preprocess_*.py` 四件套合计 **`84.34 s`**：`test_preprocess_split.py`（22 项 **`40.24 s`**）、`test_preprocess_analytic_depth.py`（22 项 **`17.00 s`**）、`test_preprocess_failed_teacher.py`（10 项 **`14.05 s`**）、`test_preprocess_cond.py`（9 项 **`13.05 s`**）。慢在每个 `setUp()` 起 `make_synthetic_episode.py` 子进程，每个用例再起 `preprocess.py` 子进程，累计 fork 超过 120 次 Python + PIL。
    3. `sim/3dgs/test_teacher_clearance.py`（53 项）：**`22.16 s`**（多组 CLI 子进程 + 占据规划测试）。
    4. `sim/tools/test_astar_cache.py`（11 项）：**`22.10 s`**（`test_paths_identical_to_reference` 在无缓存旧实现 `astar_reference` 上跑 8 个 seed × K=1/K=3 × 2 档余量 × 6 起点共 192 次全格 `_point_blocked`）。
    5. `sim/tools/test_teacher_v7.py`（31 项）：**`20.48 s`**（`pose_grid_digests` 对 v5 与 v7 各扫一遍全状态网格 A*，每次约 9–10 s）。
    6. `sim/3dgs/test_teacher_tol.py`（18 项）：**`14.79 s`**。
    7. `tools/test_scene_import_mipnerf.py`（9 项）：**`12.54 s`**。
    8. `sim/tools/test_scene_course.py`（15 项）：**`12.43 s`**（加上前面 `check.sh:181` 单独跑过一次 `scene_course.py --self-check` 扫 120 个课程，存在重复覆盖）。
    9. `sim/tools/test_rule_v2.py`（71 项）：**`11.49 s`**。
- **不减门禁覆盖的提速改法**：
  1. **`test_collect_eval.py` 夹具复用（立省 `~75 s`）**：在 `setUpClass` 里只生成 **1 份**只读合成走廊 `seed=2` 与 `seed=3` 的 PLY 及 `Occupancy`，各测试拷 `scene.json` 或在进程内直接调 `collect_3dgs.main` / `eval_3dgs.main`（只留 2 个真 `subprocess` 用例锁 CLI 入口码），单文件可从 `98.5 s` 降到 `15–20 s`。
  2. **`test_preprocess_*.py` 夹具复用与进程内调用（立省 `~55 s`）**：`make_synthetic_episode` 在 `setUpClass` 跑一次后按用例内存拷贝/软链 PNG，`preprocess.main(argv)` 改进程内捕获 `SystemExit`（保留 `check.sh:95-178` 的 CLI 负例锁子进程退出码），四件套从 `84 s` 降到 `20–25 s`。
  3. **同解释器合并进程启动（立省 `~20–30 s`）**：`sim/tools/test_*.py` 里的 30 个标准 `unittest` 文件用单次 `python -m unittest ...` 批量跑（注意测试模块间的全局状态如 `teacher._FREE_CACHE` 已在各自 `setUp/tearDown` 清理），减少 30 次解释器与 `numpy/PIL` 导入。
  - **注意并发约束**：`check.sh` 内部**不要**改成多进程 `-j4` 并行跑测试，因为公共规则 §4 给 `check.lock` 的 CPU 预算固定为 **1 核**；做完上面三条纯单核减负，即可在不增加核数的前提下把 `--python-only` 从 **`544 s`（9 分钟）压回 `180–220 s`（3–3.5 分钟）**。

---

## 3. 第一轮交叉审核清单（`FIXLIST.md`）相关条目核对

本角度涉及第一轮角度 A（Gazebo 时序与记账）与角度 E（`check.sh` 临时目录及实验入口记账）：

1. **修对了的条目（已在 `main` `3a1db2f` 核对代码与单测）**：
   - **A1（记账部分，`e9822de`、`5460af6`，合并 `e40c693`）**：`collect.py` 每步计算并落盘 `safe_gap_s = (exec_ts - prev_until_ns) / 1e9`（首步 `None`），`eval_closed_loop.py` 汇总 `safe_gap_s_median` 与 `safe_gap_s_max`；合成旧行不带该键时不凭空加键。
   - **A2（`46cd88c`、`90ed8d7`、`aa9a6ff` 合并 `e40c693`；流水线部分 `402096a` 合并 `035b398`）**：`eval_closed_loop.py` 在训练协议无 `action_latency_s` 而评测给 `L` 时记 `protocol_notes: ["train_protocol_without_action_latency"]`；`collect_parallel.py` 与两条 Gazebo 流水线（`e1_pipeline.py`、`dagger_gz_pipeline.py`）均已透传 `--action-latency` 与 `--rtf`；`preprocess.py` 已加 `--drop-late-and-stall`。
   - **A3（`51ba557`、`4d8a4ff`，合并 `e40c693`）**：`collect.py` 每步记 `obs_gap_s`，名义步周期按 `max(0.1, hold - 0.1)` 计算（修了 `hold != 1.0` 时的误判），带 `--action-latency` 时 `stall_steps > 0` 的集与 `late_dispatch_steps > 0` 同等计入 `failed[]` 并单列 `stall_episodes`。这对本报告各项提速建议是直接的安全护栏：任何改变帧间隔的副作用都会在 `obs_gap_s` 与 `stall_steps` 上立即显形。
   - **A5（记账部分，`579901d`，合并 `e40c693`）**：`cf_sitl_exec.py` 把 `STATE_BUF` 从 `4` 扩到 `600`（覆盖 3 s 仿真），窗口末由 `ack_sim_t()` 回扫本步 `until` 首次出现在插件 `state` 帧的 `IDX_SIM_T` 记入 `plugin_ack_sim_t`。`STATE_BUF=600` 只在内存 deque 里多留 596 个字典引用，不增加子进程或序列化开销。
   - **E-13（`4b43a9e`，合并 `f2c1107`）**：`tools/check.sh:6-8` 已设 `export PYTHONDONTWRITEBYTECODE=1` 并在未设 `TMPDIR` 时默认指到 `$DATA_ROOT/uav_rgb/tmp`，不再把合成集与测试临时目录写到只剩 2.8 GB 的根分区。
2. **修了一半 / 按计划待下一轮完成的条目**：
   - **A1 第三部分与 G5（动作持续到下一次派发并重标定）**：`FIXLIST.md` 已注明归卡 W1-13 下一轮；目前 `cf_sitl_exec.dispatch` 仍发 `until = t_ns/1e9 + hold_s`，因此本文第 2 节所有提速分析均严格建立在保持现行 `frame_dt = 0.900 s`、`safe_gap_s = 0.100 s` 不变的前提之上。
   - **A5 后半（空闲与满载下 `plugin_ack_sim_t` 实测分布与迟到判据切换）**：测量入口 `experiments/2026-11-12-w1-13-newprotocol/` 已合入并在排 Gazebo 池锁，尚未出数。

---

## 4. 查过但没有发现问题的地方

1. **`StudentPolicy` 的推理开销与预热（`sim/tools/student.py:88-99`）**：C++ `uavrgb_infer` 常驻子进程在构造时已做 1 帧全零 `_warmup()`，低竞争下第 0 步 `infer_ms` 中位 `0.99–1.00 ms` 与暖帧中位 `1.00 ms` 一致，没有首帧 CUDA JIT 毛刺；GPU 推理只占每步墙钟的 `< 0.1%`，不是瓶颈。
2. **`teacher.py` 的 A* 栅格缓存与每集预热（`sim/tools/teacher.py:373-404`）**：`warm_free_cells` 在起飞前一次性填满 `_free_cells`（K=1 约 `0.23 s`），使 `r2` 标定与学生臂在 K=1 上实现了 `0` 迟到步；缓存键完整覆盖 `(clearance, step, pillars)`，教师 v7 降余量重试（`0.40 → 0.30 → 0.20`）各自独立命中对应槽位（`_FREE_CACHE_MAX_SCENES = 4` 够单集 3 档余量加 SPL `0.102` 余量共 4 把钥匙同时驻留，恰好不互相挤出！但若再加第 5 档余量则会发生缓存抖动，留作备忘）。
3. **软件渲染线程数控制（`sim/tools/collect_parallel.py:148`）**：`gz_env` 在 `software` 后端下已显式设 `LP_NUM_THREADS=1`，线程采样核实每个 `gz sim` 只有 1 个 `llvmpipe-0` 线程，没有在 88 逻辑核宿主机上默认爆出几十个 LLVM 渲染线程抢光 12 核 cgroup 配额。
4. **`Xorg :20` 显示服务开销**：`ps` 与 `/proc` 采样确认容器共享的 `Xorg :20` CPU 占用仅 `0.4%`（`0.004 核`），不是瓶颈。
5. **步尾 JSONL 刷盘与接触轮询**：每步 `actions_f.flush()`、`oracle_f.flush()` 与 `timing.tail` 实测均值 `0.025 ms/步`，`CONTACT_POLL = 20 ms` 睡眠轮询在 Python 主线程开销可忽略。

---

## 5. 没有查的地方与原因

1. **没有实际启动 Gazebo 实跑 `--no-aux-cameras`、`--no-depth-camera`、进程内 `gz.transport13` 或跨集 `gz sim` 复用的 A/B 计时与像素对拍**：严格遵守任务书第 3 节「只读：不起 Gazebo、不拿池锁、不动在跑进程」的硬边界；其中 `--no-aux-cameras` 的提速与 `frame_dt=0.9 s` 已由数据盘现成的三组历史实跑数据核实，进程内订阅的 `5 ms` 竞态与跨集复用的背景色更新在各条目中明确列为需上机验证项并给出了验证命令。
2. **没有用 `perf`、`py-spy`、`strace` 对运行中的 `gz sim` 做符号级火焰图采样**：任务书明确禁止使用会打扰在跑闭环实验的工具（被抢核会出迟到步）；改用只读读取 `/proc/<gz_pid>/task/<tid>/stat` 拆到了 `llvmpipe-0`、主物理线程、传感器线程与 ZMQ 线程四层，粒度已足够支撑全部提速决策。
3. **没有实跑测量 3DGS `.npz`/`.npy` 缓存命中后的 GPU 端耗时**：GPU 当前正在跑训练池与原型线任务，按只读与不抢卡边界只做了代码链路核查与 108 份已有日志统计。
4. **没有重跑 `tools/check.sh --python-only`**：任务书明确禁止跑 `check.sh`；直接解析了当天上午刚跑完的 `w1-14/check-python-only.log` 真实日志。

---

## 做了什么

- 按任务书顺序读完 `CLAUDE.md`、`2026-11-10-phase12-common.md` §1–§4、`PLAN_2026-11-10_phase12.md` §0 与 §10、`FIXLIST.md` 全文、`review-a.md`、`2026-11-07-0140-gz-e1-mini.md`、`w1-0.md`、`longcourse.md` 及角度 P 范围内的全部核心源码（`collect.py`、`gz_io.py`、`cf_sitl_exec.py`、`eval_closed_loop.py`、`collect_parallel.py`、`scene_gen.py`、`student.py`、`save_frame.py`、`record_views.py`、`uav_rgb_e1.sdf`、`CfSitlSystem.cc`、`tools/check.sh`、`sim/3dgs/{scene,collision,renderer,collect_3dgs,eval_3dgs,test_collect_eval}.py`、`driver.py`）。
- 编写并单进程低优先级（`nice -n 15`）运行三个只读分析脚本（`analyze_all.py`、`detail_checks.py`、`extra_checks.py`），汇总 `exp-action-latency-r2-20261111`（12 个臂）与 `exp-recipe-stability-20261112/closed-loop`（已完成 5 个臂）共 200+ 集的墙钟、步段、帧间隔与 `.cg` 记录，结合间隔 30 s 以上的只读 `/proc` 进程/线程采样，完成单集墙钟与单车道逐进程/逐线程 CPU 精确拆分。
- 反汇编核实 `libgz-sensors8-camera.so` 的 `CameraSensor::HasConnections()` 与 `/uav/camera_info` 话题共享导致辅助相机在评测中持续软件渲染的根因，探测核实本机 `/usr/lib/python3/dist-packages` 的 `gz.transport13` / `gz.msgs10` Python 3.12 绑定可直接被 `.venv-tools` 复用，并推导出图像提速触发 `0.9 s → 1.0 s` 帧间隔跳变的 `5 ms` 竞态条件与防护办法。
- 统计数据盘 108 份 3DGS 日志的场景加载耗时并定位单次调用重复读 3 遍 `1.45 GB` PLY 的问题；解析 `w1-14/check-python-only.log`（`544 s`）定位前 10 个最慢测试文件。

## 没做什么（每条带原因）

- 未修改主仓库或任何工作树中的任何文件、未建分支、未提交、未写仓库内 `docs/worklog/` 或 `progress.md`：原因：严格执行任务书「只读审核 agent」边界。
- 未启动 Gazebo、未跑闭环实验或 3DGS 渲染、未跑 `tools/check.sh`、未触碰任何 `.lock` 锁文件与在跑守护进程：原因：任务书明令禁止，避免干扰正在运行的闭环评测与训练任务。
- 未从历史实验目录拆分出单臂独占 CPU 核数：原因：历史 `.cg` 只记录容器整体 `usage_usec`，多池并行期间无法事后剥离训练池与其它子进程的核数，已在报告 1.3 节标明「未核实」并给出下次实验 15 行 `getrusage` + `/proc/<sim.pid>/stat` 埋点方案。

## 验证到哪一步

| 检查项 / 命令 | 结果 |
| --- | --- |
| `nice -n 15 .venv-tools/bin/python .../review2-p/analyze_all.py` | 通过（输出 `analysis_summary.json`，覆盖 `r2` 12 臂、`recipe-stability` 5 臂、7 次 `/proc` 进程与 `gz sim` 线程拆分、108 份 3DGS 加载日志） |
| `nice -n 15 .venv-tools/bin/python .../review2-p/detail_checks.py` | 通过（实测 K=1 / K=3 的 `warm_free_cells` 与 `episode_shortest_path_m` 耗时，映射 `check.sh --python-only` 慢测排行） |
| `nice -n 15 .venv-tools/bin/python .../review2-p/extra_checks.py` | 通过（核实 `calib` 中 `aux/noaux/cam5` 与 `analytic-depth-parity` 的 `RTF`/`frame_dt`，实测 `decode_depth` `6.92 ms`、`write_rgb_png` `4.84 ms`、`viz3` 录像 `115.0 s/集`） |
| `/usr/bin/python3 -I` 与 `.venv-tools/bin/python` 导入 `gz.transport13` / `gz.msgs10` 探测 | 通过（`Node.subscribe` 与全部 5 类消息的 `pb2` 绑定均在 `/usr/lib/python3/dist-packages` 可用） |
| `objdump -d -C` / `nm -D -C` 核查 `libgz-sensors8-camera.so` 与 `libgz-sim-sensors-system.so` | 通过（确认 `HasConnections` 含 `camera_info` 订阅判定，确认 `Sensors` 插件实现 `ISystemReset`） |
| `git -C /root/claude-code-backup-20261028/code/uav_rgb status --short` | 通过（输出为空，主仓库完全干净未改动） |
