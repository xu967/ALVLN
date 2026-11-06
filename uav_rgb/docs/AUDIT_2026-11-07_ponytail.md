# uav_rgb 代码质量审计（ponytail-audit）— 2026-11-06

**审计基线：git worktree 快照 `/root/rivermind-data/uav_rgb/tmp/wt-baseline-20261106/`，HEAD `9d764d9`，工作树干净。**
主仓库 `/root/claude-code-backup-20261028/code/uav_rgb/` 今晚已提交 cf_sitl 库、3DGS 修复、UI 修复，本报告不追，所有行号以 9d764d9 为准。
不重复 `docs/AUDIT_2026-11-05.md` #1–#20、`docs/REVIEW_2026-11-05_real_drone_sim_viz.md` #1–#20、`TinyAerialVLN.md` SIM-01–20 / D01–D18 的编号问题；需要时写「同 XX #n」。
复现脚本与输出都在 `/root/rivermind-data/uav_rgb/tmp/agent-audit/`（纯 Python，没起 Gazebo，没用 GPU）。标「未复现」的条目只有代码与仓库文档作证据。

---

**这套代码做什么：** 用一张机载 RGB 图加一句短指令，让无人机在四个离散动作（停/前进/左转/右转）里选一个。数据来自 Gazebo 或 3DGS 仿真里的「教师」飞行（`sim/tools/collect*.py`），打包成自定义二进制（`tools/preprocess.py`），由自写的 C++/CUDA 训练器训练一个小 CNN（`src/`），再由常驻推理进程在仿真里闭环飞（`sim/tools/eval_closed_loop.py`）。

**假设的负载：** 一个人、一台单 GPU 服务器、Studio 单用户。采集是多进程（`collect_parallel.py` 同时开 4–8 个 Gazebo，软件渲染每步约 3 s）；采集/训练/评测动辄跑几小时、跨会话，按 BRIEF 规定用 `setsid nohup` 起守护进程，用 `kill -- -PGID` 停。

---

## 必须修

### Q1. `collect_parallel.py` 没有 SIGTERM 处理器：按规定方式停采集会留下一整排孤儿 Gazebo（`sim/tools/collect_parallel.py:194-214, 290-310, 613-626`；对照 `eval_closed_loop.py:722-724`）

- **是什么：** 并行采集器。每个 Gazebo 和每个 `collect.py` 都用 `start_new_session=True` 起在自己的进程组里，登记进 `_PROC_GROUPS`，由 `main()` 的 `finally: cleanup_all()` 收尾。
- **问题：** `main()` 只处理 `KeyboardInterrupt`，没有 `signal.signal(SIGTERM, …)`。Python 收到 SIGTERM 的默认行为是立刻死亡、不跑 `finally`。而子进程全在别的 session，所以 BRIEF 规定的停法 `kill -- -<PGID>`（或普通 `kill <pid>`）只杀掉 Python 本身，4–8 个 `gz sim`（每个占满 1 核以上）和 `collect.py` 继续跑到 `--episode-timeout`，之后也无人收尸。同一仓库的 `eval_closed_loop.py:722-724` 就装了处理器（`_term` → `SystemExit(143)` → `finally`），两处写法不一致。**已复现**（`agent-audit/repro_sigterm_orphan.py`，用 `sleep 300` 代替 `gz sim`，其余照抄 `_start_sim`/`_register`/`finally: cleanup_all()`）：
  ```
  no-handler    kill-pid : collect_parallel 退出 rc= -15  子进程(sleep=90055) 仍在运行 <-- 孤儿
  no-handler    kill-pgid: collect_parallel 退出 rc= -15  子进程(sleep=90061) 仍在运行 <-- 孤儿
  with-handler  kill-pid : collect_parallel 退出 rc= 143  子进程(sleep=90065) 已被 cleanup_all 收掉
  with-handler  kill-pgid: collect_parallel 退出 rc= 143  子进程(sleep=90082) 已被 cleanup_all 收掉
  ```
- **修：** 把 `eval_closed_loop.py:722-724` 那三行原样搬进 `collect_parallel.main()` 开头：`def _term(signum, frame): raise SystemExit(143)` + `signal.signal(signal.SIGTERM, _term)`。3 行，`finally` 原封不动。
- **不修的后果：** 每次按规矩停采集都会留下几个吃满 CPU 的 Gazebo；BRIEF 又禁止 `pkill -f 'gz sim'`（会误杀别的 agent 的仿真），只能逐个手动找 PID。和 Q2 叠加时见下。

### Q2. Ctrl-C 之后 worker 继续「失败重试」并起新 Gazebo，主线程被 `shutdown(wait=True)` 挂住；连按三次也退不干净（`sim/tools/collect_parallel.py:377-404, 407-408, 613-626`）

- **是什么：** 每个实例是一个 future，顺序跑完自己的所有 episode（`run_instance`）；`run_episode` 对任何失败都重试一次（重启 sim）。Ctrl-C 分支先 `cleanup_all()` 杀掉在飞的 sim/collect，再 `executor.shutdown(wait=True, cancel_futures=True)`。
- **问题：** 被 `cleanup_all` 杀掉的 collect 返回 rc=-15 → `task.ok=False` → `run_episode` 无条件进入 attempt 2，**重新起一个 Gazebo**；跑完还继续该实例的下一集。`cancel_futures=True` 只取消没开始的 future，而每个实例就是一个已在跑的 future，所以一个都取消不了；主线程卡在 `shutdown(wait=True)`。再按 Ctrl-C 会从 `shutdown` 里抛第二个 `KeyboardInterrupt`，落进 `finally` 的第二个 `shutdown(wait=True)` 继续卡；第三次 Ctrl-C 才真正退出，但此时 worker 刚起的 sim 没人收。**已复现**（`repro_ctrlc_retry.py`、`repro_ctrlc_x3.py`、`repro_ctrlc_orphans.py`，真实 `cp.run_episode`/`cp.run_instance` + `main()` 同样的 executor 结构，`_run_episode_once` 换成 0.2–0.3 s 假实现返回 rc=-15）：
  ```
  Ctrl-C 在 t=0.10s；主线程直到 t=1.20s 才从 shutdown(wait=True) 返回
  _run_episode_once 总调用 12 次；其中 Ctrl-C 之后仍发起 10 次（含重试 attempt=2 的 6 次）
  ```
  三次 Ctrl-C 版本（子进程用 `start_new_session` 的 `sleep 300`，`finally: stop_group`）：
  ```
  collect_parallel 替身退出 rc=130
  pid=100356 scene=103 attempt=1 t=+0.62  <-- 孤儿
  pid=100357 scene=102 attempt=1 t=+0.62  <-- 孤儿
  孤儿数=2
  ```
  真实规模（8 实例 × 几十集 × 40 步 × 3 s）下，Ctrl-C 之后仿真还会继续跑几个小时。
- **修：** 模块级 `_STOP = threading.Event()`；`KeyboardInterrupt` 分支（和 Q1 的 `_term`）第一句 `_STOP.set()`；`run_episode` 的 `for attempt` 里重试前加 `if _STOP.is_set(): break`；`run_instance` 改成 `for t in tasks: if _STOP.is_set(): break; …`。约 6 行；不加线程、不加配置。
- **不修的后果：** 「中断」不是中断：按下 Ctrl-C 后 GPU/CPU 还被仿真占着几小时，别的 agent 按 BRIEF 规则不能启动 Gazebo；半截 episode 还会被搬进 `failed/` 当失败统计（见 SIM-19 的失败偏差）。

---

## 应该修

### Q3. 训练慢的主因已经在仓库里被量过，但没人收口；卷积反向的 dW/db kernel 几乎是单 SM 串行（`src/conv_cuda.cu:95-140`，`src/model.cpp:278`）—— **未复现**（本机 GPU 被占用，不允许本轮使用）

- **是什么：** 自写 CUDA 卷积。前向每线程算一个输出元素；反向 `conv2d_dw_kernel` 每线程算一个**权重**元素、串行扫完全部 N×H'×W'，`conv2d_db_kernel` 每线程一个**输出通道**。
- **问题：** 「28.5 样本/秒」出自 `docs/experiments/2026-10-30-t6-train.md:28`，同期 `t5-scaleup.md:80` 注明是「采集中并行，CPU 被抢」；空载实测是 52.9 / 56.3 / 59–62 / 68.6 / 69.8 样本/秒（`docs/worklog/2026-10-30-0500`、`experiments/2026-10-31-l1-full-400`、`worklog/2026-10-29-0051`、`experiments/2026-11-01-l2-film-finetune`、`experiments/2026-10-31-t5b-dispatch`）。所以 28.5 一半是调度问题（和 6 个 Gazebo 抢 12 个核），不是代码。代码侧仓库唯一一次 nsys 分解（`docs/worklog/2026-10-28-2337-gpu-bench.md:54`，batch 256，同一设计的 kernel）：一步 5.35 s，前向包装 1851 ms 中 kernel 只有 216 ms（其余是每层 `cudaMalloc`/H2D/D2H/同步，这部分同 TinyAerialVLN D13，不重复）；**反向 kernel 本身 1940 ms**，按 FLOPs 反向≈2×前向只该 ≈430 ms。静态可数的原因：训练形状下 `c1` 的 dW 只有 16·3·9 = **432 个线程（2 个 block）**，每线程循环 64·120·160 = 1.23M 次；`c1` 的 db 是 **16 个线程（1 个 block）**各循环 1.23M 次；`c2` dW 4,608 线程 × 307k、db 32 线程；`c3` dW 18,432 线程 × 76.8k、db 64 线程。对比前向 `c1` 是 19.7M 线程 / 76,800 个 block。一张 68–80 SM 的卡上 1–18 个 block 等于只用一两个 SM。另外 `model.cpp:278` 对第一层也算 `dx`（531M 次乘加，与 c1 前向同量级），结果没人用。
- **修：** 不重写 `.cu`：① 采集和训练不要同时跑（零代码，吞吐从 28.5 回到 50–70）；② `conv2d_dw_kernel`/`conv2d_db_kernel` 的 grid 加一维 `blockIdx.y` 切 `n`（或 `oh`），每线程只累加自己那一片，末尾 `atomicAdd` 进 `dw[idx]`/`db[oc]`，host 侧先 `cudaMemset` 为 0——每个 kernel 约 10 行，`test_conv_cuda` 的 CPU/CUDA 对拍直接覆盖；③ `conv2d_backward_cuda` 加一个 `bool need_dx`，`model.cpp` 对 `c1` 传 false 跳过 dx kernel（3 行）。改完用 nsys 再量一次，再决定 D13 的持久显存值不值得做。
- **不修的后果：** 训练按「样本/秒」继续被包装开销和两个几乎串行的 kernel 卡着；数据量再翻一倍（L1 已 16k 样本/轮）就是 2–4 小时一轮，迭代节奏被训练而不是被想法决定。

### Q4. 所有 CPU/CUDA 对拍测试的张量都塞得进一个 block，多 block 索引路径从未被对拍过（`src/tests/test_conv_cuda.cpp:98-102`、`test_nn_cuda.cpp:44,90,115`、`test_linear_cuda.cpp:62-64`、`test_softmax_cuda.cpp:54-55`）

- **是什么：** 四个数值对拍测试（CPU 参考 vs CUDA 同输入比最大误差）。kernel 都是 `kThreads = 256`，一个 block 256 个线程。
- **问题：** 测试里最大的输出是 `test_conv_cuda` 的 y = 2·3·6·7 = **252 个元素**；ReLU 60、MaxPool 140、Adam 120、Linear 16×4 = 64、softmax 6×5 = 30。全部 `blockIdx.x == 0`。任何只在第 2 个 block 起才出错的 bug（把 `blockIdx.x*blockDim.x` 写丢、`idx` 的 int 溢出、grid 少算一个 block）都会让这四个测试照样 PASS；`blockIdx.x ≥ 1` 只被 `test_overfit`/`test_trainer`（160×120、batch 16）间接碰到，而那两个只断言「收敛」，不对拍 CPU。现有 kernel 我逐行看过是对的，这是「有风险的代码没测试」，不是现有 bug。**未复现**（无 GPU；结论来自测试里的尺寸常量）。
- **修：** 改数字：`test_conv_cuda.cpp:98` 的 x 从 `{2,2,6,7}` 改成 `{2,2,16,17}`（y = 1,632 元素 = 7 个 block）并同步 dy 形状；`test_nn_cuda.cpp:44` 的 `{3,4,5}` 改 `{3,40,5}`；`test_linear_cuda.cpp:64` 的 `n = 16` 改 `n = 300`；softmax `n = 6` 改 `n = 300`。参考值都是 CPU 算的，不用手写期望。
- **不修的后果：** 下一次有人「顺手」改 kernel 的索引算法（例如做 Q3），唯一能抓住错误的是训练发散，而不是 CTest。

### Q5. `--policy teacher-rl` 整条路径是死的：采出来的数据打不了包、依赖没装、没有任何 checkpoint（`sim/tools/collect.py:1266-1267, 1344-1347`，`tools/preprocess.py:166-169`，`collect_parallel.py:303-304, 479-481, 508-514, 682`，`eval_closed_loop.py:453-454, 551-557, 670-675, 772`，`rl_teacher/`）

- **是什么：** 用 PPO（强化学习）在轻量物理近似（`rl_teacher/core.py`，cf_stab 语义）上训练一个教师，再让它在 Gazebo 里采数据。三个 CLI 都接了 `--policy teacher-rl --rl-checkpoint`。
- **问题：** `collect.py:1267` 给这种集写 `teacher="teacher_rl"`，而 `preprocess.py:167` 只放行 `plan_action / expert_action / synthetic`。**已复现**（合成集把 manifest 的 teacher 改成 `teacher_rl` 后打包）：
  ```
  preprocess: .../ep1/episode_manifest.json: teacher='teacher_rl' 不是教师标签（forward/student 产物不能当训练数据打包）
  rc=1
  ```
  同时 `stable_baselines3` 在 `.venv-tools` 里没装（`ModuleNotFoundError`），`README.md:102` 写明「未训练 PPO」，数据盘下也没有任何 PPO/rl_teacher 产物。全树 grep：`teacher_rl` 的消费者只有这三个 CLI 的参数透传，没有打包器、没有评测报表认它。
- **修：** 删 `rl_teacher/`（6 个文件，468 行）、三个 CLI 的 `teacher-rl` 分支和 `--rl-checkpoint`（collect.py 约 25 行、collect_parallel.py 约 10 行、eval_closed_loop.py 约 10 行）、`test_realism_rl_guards.py:106` 那一个用例。若想保留「以后可能训 PPO」，至少把 `preprocess.py:167` 的白名单和 `collect.py:1267` 的值对齐——但在没有 checkpoint 之前这是给不存在的东西写代码。
- **不修的后果：** 读者（和下一轮 agent）会以为有一条 RL 教师路线可用，花时间去配 sb3、跑 PPO，最后发现产物进不了训练集。

### Q6. REVIEW #15（三套占位动力学）的**执行清单**：删哪些文件、改哪 21 个引用点（本轮只补清单，不重复判断）

- **是什么：** cf_stab（`collect.py` 内嵌）、cf_sil（`cf_sil.py`）、cf_blackbox（`cf_blackbox.py`）三种「模仿 CF-Drone」的执行模型，REVIEW #15 已判定互相矛盾、全是猜的，SITL 验收后删。
- **问题：** 没有清单就删不干净——`collect.py` 里 122 处引用散在 9 段，另有 8 个文件、2 份 JSON、1 个 3DGS 包装依赖它。全树 grep（含测试、文档、字符串）结果如下。
- **修（清单）：**
  整文件删除：`sim/tools/cf_sil.py`(982)、`test_cf_sil.py`(293)、`cf_blackbox.py`(295)、`test_cf_blackbox.py`(232)、`teacher_predictive.py`(59)、`sim/dynamics/cf_sil_v1.json`、`sim/dynamics/cf_blackbox_v1.json`、`rl_teacher/`(468，同 Q5)；3DGS 接上 SITL 后再删 `sim/3dgs/dynamics.py`(45) + `test_dynamics.py`(73)。
  引用点（文件外的 21 处）：
  1. `sim/tools/collect.py:70-71` `import cf_sil / cf_blackbox`
  2. `collect.py:141-231` `CfStab` 类及 `DYNAMICS / DYNAMICS_EXPERIMENTAL / CF_STAB_PROFILE / DEFAULT_CF_SIL_PARAMS / DEFAULT_CF_BLACKBOX_PARAMS`
  3. `collect.py:307-313` 安全网的「cf_sil 停桨」分支
  4. `collect.py:664-691` `take_off_cf_sil`
  5. `collect.py:722-1000` `collect_one` 的 `dyn`/`sil` 分支（775、791、904、924-927、987、994）
  6. `collect.py:1127-1145` `--dynamics` 选项、`--cf-sil-params`、`--flight-log`、`--flight-log-csv`、`--teacher-planner predictive`
  7. `collect.py:1176-1222` 对应参数校验
  8. `collect.py:1266-1308, 1334, 1357-1371` 执行器初始化（`CfStab()`、`CfSilMotorThread`、`PredictiveTeacher`）
  9. `collect.py:1511-1537` manifest 的 `dynamics_profile / experimental_policy / rl_checkpoint`
  10. `sim/tools/collect_parallel.py:54-57, 303-304, 463-464, 479-481, 508-517, 682`
  11. `sim/tools/eval_closed_loop.py:62, 451-454, 495, 551-559, 571-572, 647-675, 772`
  12. `sim/tools/audit_dataset.py:603-672`（cf_stab 跳过定速判据的整段）
  13. `sim/tools/scene_gen.py:337, 550`（为 cf_sil 删 `MulticopterVelocityControl` 的选项）
  14. `sim/tools/test_collect.py:348-549`（cf_stab/cf_sil 用例，约 200 行）
  15. `sim/tools/test_eval_closed_loop_args.py:60-75`
  16. `sim/tools/test_audit_dataset.py:591-599`
  17. `sim/tools/test_realism_rl_guards.py:106`
  18. `tools/train_student_with_video.py:19, 120, 132` 与 `tools/test_train_student_with_video.py:38-89`
  19. `ui/test_studio.py:230, 324, 333` 与 `ui/README_STUDIO.md`（`ui/studio.py` 本身不含 cf_ 字面量，按 manifest 透传）
  20. `sim/3dgs/collect_3dgs.py:290` 及 `make_blackbox` 的导入（3DGS 重写中，只标不改）
  21. `docs/EVAL_STANDARD.md`（同 AUDIT #20）与 `CLAUDE.md` / `docs/HANDOFF.md` 里的 `--dynamics` 说明
- **不修的后果：** 同 REVIEW #15；多出来的是：清单不在手上时，删一半会让 `test_collect.py:533-549` 和 `test_eval_closed_loop_args.py:60-63` 这类「断言 cf_sil 在/不在矩阵里」的测试先红，拖住删除。

---

## 可以顺手修

### Q7. 前进速度 `SPEED_FWD = 0.3` 手抄了两份，审计门用自己那份（`sim/tools/collect.py:97`，`sim/tools/audit_dataset.py:79`；另 `rl_teacher/core.py:108` 的 `yaw_rate=.6`）

- **是什么：** `collect.py` 用它发 twist；`audit_dataset.py` 的 forward_motion 门用它算「一步至少该走多远」（0.4×SPEED_FWD×dt），行尾注释写着「与 collect.py 一致」。
- **问题：** 两处是独立常量，没有任何检查。把 `collect.py` 的 0.3 改成 0.4 重采，审计门仍按 0.3 算阈值，少走 25% 的步也 PASS；反过来改小则全部假 FAIL。`tools/check.sh:13-40` 已经对 `kObsW/kTextLen/kTextVocab` 做了「两份手抄必须相等」的检查，唯独没管这一个。
- **修：** 把 `SPEED_FWD`/`SPEED_YAW` 移到无依赖的 `sim/tools/teacher.py`（`audit_dataset.py` 已经 `from teacher import …`），`collect.py` 和 `audit_dataset.py` 各改成 import。净 -1 行。`rl_teacher/core.py:108` 随 Q5/Q6 一起删。
- **不修的后果：** 下次调速度，数据集审计的 `forward_motion` 门默默失效，而且它是最像「物理合理性」的那条。

### Q8. 文档里会误导操作的数字和路径（`CLAUDE.md:16, 59, 67-76`，`docs/HANDOFF.md:8, 52, 75-76, 902-916`）

- **是什么：** 两份入口文档。
- **问题（只列会让人做错事的）：**
  - `CLAUDE.md:59`「ctest 当前 20 项」、`HANDOFF.md:75`「21 个 CTest」——实际 `src/CMakeLists.txt` 注册 **25** 条（`grep -c add_test` = 25；`README.md:27` 的 25/25 是对的）。少报的正好是 device/batch_reader/run_owner 那几条，读者会以为它们没进 CTest。
  - `HANDOFF.md:76`「13 个 `test_*.py`」——`sim/tools/` 下实际 **25** 个；「18 条数据验收闸门」——`audit_dataset.py` 里有 **20** 个门名（`grep -o 'check("[a-z_0-9]*"' | sort -u`），按任务类型启用不同子集（同一文档 `HANDOFF.md:154` 写 L1 v4 集「19/19 PASS」），这个数随数据集变，写成固定数字只会对不上。
  - `CLAUDE.md:16`、`HANDOFF.md:8, 52`「RTX 2080 Ti（sm_75，22 GiB，68 SM）」——BRIEF 核实本机是 RTX 4080 SUPER（sm_89，用 sm_75 PTX JIT 跑，D09 已报架构问题）。按 22 GiB 估 batch 会把显存算多。
  - `CLAUDE.md:67-76`、`HANDOFF.md:902-916` 的示例全部 `--out /tmp/...`——BRIEF 禁止写 /tmp，且 `df -h /` 现在只剩 **1.7 GB**；按示例采 500 集（约 2 GB PNG）会在中途把根分区写满，`collect.py` 以「写 RGB 失败」整集报错。
- **修：** 改成「25 项（以 `ctest -N` 为准）」「25 个 `test_*.py`」「门数以 `audit_dataset.py` 输出为准」；机器一行改成 4080 SUPER / sm_89（PTX JIT）；示例路径统一成 `$DATA_ROOT/uav_rgb/...`（HANDOFF 别处已在用这个变量）。纯文本改动。
- **不修的后果：** 新接手的人按文档走第一遍就踩到磁盘满或 ctest 数量对不上，先怀疑环境再怀疑自己，浪费半天。

### Q9. 没人引用的文件：三份世界模板、一个 profiling 实验、一个 GRU 原型（`sim/worlds/templates/*.sdf`，`experiments/pipeline_profile/`，`teacher/student_gru_proto.py`）

- **是什么：** `sim/worlds/templates/uav_rgb.{3267bb7f,70ef2c0a,7d7f9472}.sdf`（1,413 行）；`experiments/pipeline_profile/`（CMake 项目 + 115 行 .cpp）；`teacher/student_gru_proto.py`（78 行）+ 它自己的测试（64 行）。
- **问题：** 全树 grep（代码、脚本、文档、字符串）：`templates/` 零引用，`scene_gen.py`/`collect_parallel.py`/`studio.py` 都只用 `sim/worlds/uav_rgb.sdf`；其中 `70ef2c0a` 与 `uav_rgb.sdf` **md5 相同**（`065e1eed…`），另两份是旧版本。`pipeline_profile` 除自身和历史 worklog 外零引用（`cuda_smoke` 有 `CLAUDE.md:38-48` 引用，保留）。`student_gru_proto.py` 只有 `teacher/test_student_gru_proto.py` 引用它，而学生被文档明确定义为「无时序记忆」。`data/ models/ logs/ config/ third_party/` 的 `.gitkeep` 是空占位（数据在 `/root/rivermind-data`），`__pycache__` 未被 git 跟踪，这两项不算问题。
- **修：** `git rm -r sim/worlds/templates experiments/pipeline_profile teacher/student_gru_proto.py teacher/test_student_gru_proto.py`。约 -1,670 行，没有调用方要改。
- **不修的后果：** 很小——主要是每个读仓库的人都要花几分钟确认「这三份 sdf 哪份是真的」。

---

## 补充：采集「1.5 步/秒」的每步耗时分解（结论，不计入 finding）

仓库自己的测量（`docs/experiments/2026-11-03-gpu-rendering.md`，hold=1.0）：软件渲染 **每步 4.23 s：I/O 0.059 s、动作窗口 4.1 s（RTF 0.24）、发布开销 ≈0、其他 <0.001 s**；GPU 渲染每步 1.07 s（`io=0.0586 window=1.0144 block=1.0147 tail=0.00004`，RTF 1.0）。本轮在 `.venv-tools` 下实测纯函数开销：`plan_action_global`（A*）中位 0.4 ms / 均值 4.4 ms / 最大 31 ms；`decode_depth` 5.8 ms；PNG 编码 9.0 ms。即 **每步 ≥99% 的墙钟是在等 Gazebo 推进仿真时间**；Python 全部优化到 0 也只省 1–2%。「1.52 步/秒」（`docs/results.md:415`）= 6 个实例 × RTF ÷ hold 的结果，杠杆只有两个：实例数（受 12 核 cgroup 限制）和 RTF（本容器 GPU 渲染段错误，只能软件渲染）。

## 没有当成 finding 的事项

- `src/` 四个 `.cu` 的边界、`N>256`、越界、同步、`cudaError`、溢出逐行核过：多 block 索引正确、元素数先按 int64 校验、每次 launch 后 `cudaGetLastError + cudaDeviceSynchronize`、所有 API 过 `CUDA_CHECK` 抛异常、`N=0` 不 launch、softmax 先减 max、范数 NaN/超 float 都抛。没有新 bug（只有 Q4 的测试覆盖问题）。
- `dataset.cpp` 的 UAVR 读端与 `preprocess.py:360-452, 484-508` 的写端逐字段对过（头偏移自洽、`collision_flags`、episodes[] 扁平化——preprocess 已注明「C++ meta 扫描器假定 episode 项内无嵌套对象」并把 waypoint 拆成 `waypoint_x/y`）；`check.sh` 已用 `read_dataset.cpp --selfcheck` 接上真实产物。
- `infer.cpp` ↔ `student.py` 协议（hello 行、HWC 原始字节、一行 JSON、argmax 取首个最大、`/255`、PIL BILINEAR）两端一致。
- `collect.py --stop-on-collision` 把「卡滞」也当停止条件而 `eval_closed_loop.summarize_episode` 不把卡滞算碰撞：口径不同但结果保守（卡滞集记 STUCK、不记成功），不算错。
- `eval_closed_loop` 的 `timed_out` 永远为 False（超时走失败分支）、`collect_parallel --out` 非空检查有竞态：单用户下无后果。
- `shortcut:` 注释（`collect.py:926` 等）都写明了边界且当前负载没越过，按规则算决定不算 finding。
- `src/device/`（512 行 + 3 个测试 ≈ 520 行）与 `BatchReader`（149 行 + 178 行测试）未接入训练：AUDIT 已登记为债务。本轮引用情况：除各自测试、`CMakeLists.txt` 和文档外零调用方；它们服务的并行训练方案已归档（`docs/archive/PARALLEL_TRAINING_PLAN.md`）。`RunOwner` **不是**未接入——`train_main.cpp:120-126, 186-192` 用它做 run 目录单写者锁。
- 快照上跑过的纯 Python 自检全部通过：`teacher.py`、`scene_gen.py --self-check`、`gz_io.py --self-check`、`test_collect.py`（58 项）、`test_process_cleanup.py`、`test_eval_closed_loop_args.py`、`test_audit_dataset.py`、`test_split_manifest.py`。

---

**Verdict：** 核心数值与数据链没有发现正确性 bug；先修 Q1+Q2（约 10 行），否则按 BRIEF 规定方式停采集会留下一排孤儿 Gazebo、Ctrl-C 后仿真还要跑几小时。

**Lean：** -2,600 行（Q5+Q6：cf_sil/cf_blackbox/teacher_predictive/rl_teacher 及其测试、collect.py 内嵌 CfStab；REVIEW #15 估 -1,900，本轮把 rl_teacher 和 3DGS 包装算进去）+ -1,670 行（Q9）可行；另 -1,200 行（`src/device/` + `BatchReader` 及测试）取决于并行训练方案是否复活；-1 个从未安装的可选依赖（`stable_baselines3`）。

**Not checked：** 没有用 GPU，Q3/Q4 未在本机复现，`ctest` 未跑；没起 Gazebo，Q1/Q2 用 `sleep` 替身复现。没读：`src/ops_cpu.cpp`、`nn_cpu.cpp`、`conv_cpu.cpp`、`conv_gradcheck.cpp`、`soft_targets.cpp`、`metrics_json.cpp`、`sha256.cpp`、`device/*.cpp`、`batch_reader.cpp` 与 `run_owner.cpp` 正文、`src/tests/` 中除 `test_nn_cuda`/`test_conv_cuda` 外的测试正文；`ui/studio.py`（3,060 行，重写中）、`ui/monitor.py`、`tools/train_student_with_video.py`、`tools/doctor.py`、`tools/airsim_to_uavr.py`、`tools/baseline_priors.py`；`sim/3dgs/` 只读了 `collect_3dgs.py` 的 manifest 部分（`collision.py`/`renderer.py`/`scene.py`/`eval_3dgs.py` 未读，且正在重写）；`sim/tools/scene_gen.py`、`campus.py`、`asset_campus.py`、`import_asset.py`、`instruction.py` 只读了自检结果和被引用的函数；`teacher/`（PyTorch 教师）、`firmware/`、`docs/worklog/` 大部分、`docs/archive/`。
