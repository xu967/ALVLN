# 第十二阶段 W1-0：动作派发时刻固定到仿真时间（`--action-latency`）

- 时间：2026-11-10 22:46（标定与验收守护进程 22:42 起跑）
- 范围：`sim/tools/{collect,teacher,eval_closed_loop,scene_gen,collect_parallel}.py`、`tools/preprocess.py`、相关测试、`experiments/2026-11-10-action-latency/`、`docs/EVAL_STANDARD.md` 5.5 节、`docs/experiments/2026-11-10-action-latency.md`
- 分支：`p12-w1-0`（从 main 4498177 起），提交 `69a5515`（collect）、`b85aa6b`（scene --rtf）、`8dae28f`（实验入口）、本条与文档（见 `git log`）。不合并、不推送，`progress.md` 由主会话在 main 上维护，这里没动。
- 详细报告：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/w1-0.md`；实验文档 `docs/experiments/2026-11-10-action-latency.md`。

## 做了什么

- `collect.py --action-latency L`：策略结果出来后等位姿话题仿真戳 ≥ 观测戳 + L 再派发，执行器截止仍是观测戳 + hold；就绪时仿真戳已超过 观测戳 + L + 0.02 s 记迟到步，立即派发。
  每步行记 `dispatch_lag_s`（不给开关也记），manifest 记 `action_latency_s` 与 `late_dispatch_steps`。
- 学生 + β=0 + 带开关：影子 A\* 标签挪到派发之后；带开关时每集起飞前填满教师的整张 A\* 自由栅格（`teacher.warm_free_cells`）。
- `eval_closed_loop.py`：透传 `--action-latency` 与 `--rtf`；`per_episode` 记延迟中位 / 最大与迟到步数；含迟到步的集单列在 `late_dispatch_episodes`，计作基础设施失败、不计成功。
- `scene_gen.py --rtf R`；`PROTOCOL_KEYS` 与 `STRICT_KEYS` 各加 `action_latency_s`，带开关的数据与旧数据可区分。
- 测试：`test_action_latency.py`（24）、`test_scene_rtf.py`（7）、`test_astar_cache.py`（+4）、`test_preprocess_split.py`（+1）；先在旧代码树里红，再在新代码里绿；新旧差分 8 个场景输出 sha256 相同（默认路径逐位不变）。
- 实验入口 `run.sh`（预注册抄自任务卡，标定门自动判定，L=0.16 未过试 L=0.20，过门才进验收 4）、`summarize.py`、演练脚本（假评测入口，四条分支都走通）。
- 冒烟 3 条（教师 K=1 / K=3、学生 K=1，各 1–2 集，L=0.16）：K=1 无迟到步；K=3 79 步里 3 步迟到（最大延迟 0.195 s）。
- 起跑正式守护进程；前几集的 `dispatch_lag_s` 与迟到步见报告第 5 节。

## 没做什么

- `collect_parallel.py` 与 `tools/e1_pipeline.py` 没有透传 `--action-latency`：任务卡只要求 collect 与 eval；训练数据是否带开关要主会话定。
- 学生与外部基线没有新加「每集一次空推理」：`StudentPolicy.__init__` 已经空推理一次（每集一个 collect 进程构造一次），再加会重复。
- 森林三档没有跑带开关的评测，也没有用 `--rtf` 实际跑；需要的 `--rtf` 只按公式推算。
- 没有排查 K=3 冒烟里预热后仍出现的 3 个迟到步（步 0、3、8）的原因：冒烟没有分段墙钟，标定的 10 集出来后再看。
- 标定与验收 2–4 的最终数字：守护进程在跑，没等。

## 待确认（主会话）

1. 标定「过」的读法：我按 L=0.16、再 L=0.20 的顺序，第一个过标定门的 L 作为阶段 2（验收 4）的 L；两个都不过则不进验收 4，DONE 写原因。卡里「最后用哪个 L 由主会话定」我照办，没替你选。
2. L=0.20 的生效时长按 hold − L + 约 7.5 ms 过冲推算为约 0.807 s（L=0.16 的冒烟实测 0.847 s，同一推算 0.840 + 0.0075），比旧协议 K=1 的 0.871 s 短约 7%，L=0.16 短约 3%；数字以标定表为准。选哪个 L 都意味着新口径的每步动作比旧口径短，SR 与旧数字不可直接比。
3. 训练数据采集是否带 `--action-latency`（学生训练分布与评测协议是否一致），`PROTOCOL_KEYS` 对此已有检查。

## 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `python sim/tools/test_action_latency.py` | 旧代码 23 项失败 / 新代码 24 项 OK |
| `python sim/tools/test_scene_rtf.py` | 旧代码 5 项失败 / 新代码 7 项 OK |
| `test_astar_cache.py`、`tools/test_preprocess_split.py`、`test_collect.py`、两个 wiring 测试 | 11、22、108、16、18 项 OK |
| `UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only` | rc=0（C++/CUDA 构建、CTest、GPU 测试、cf_sitl 被跳过） |
| 新旧差分 `diff_default.py`（8 个场景） | 输出 sha256 相同 |
| 演练 `rehearse.sh ladder / pass016 / fail / fwd` | 四条分支的 DONE 与 `L_STAR` 符合设计 |
| 冒烟（真 Gazebo，3 条） | K=1 无迟到；K=3 3/79 步迟到 |
| 正式守护进程 | 已起跑，前几集见报告第 5 节；全部结果未出 |

## 风险与遗留

- 同 seed 的两次 K=1 运行（冒烟与守护进程首集，都是 seed 2、L=0.16）稳态前飞位移 0.2455 对 0.2596 m/步，相差 5.7%（样本各约 24 步）；标定门的 2% 阈值作用在 10 集合并量上，单集噪声可能把「该过」判成「未过」。阈值是预注册的，没改。
- 两个守护进程和其他 agent 共用机器：本任务 2 个 gz，加上基线实验的 5 个，正好 7 个；任何一边再加 gz 就超配额。
- 标定里 K=3 的迟到步若集中在步 0，要先看预热是否生效，再考虑 L=0.20 或 `--rtf`（预注册顺序）。

## 更新 2026-11-11 20:55

第一轮 11-10 23:33:48 结束，`DONE rc=10`：标定门在 L=0.16 与 L=0.20 都没过，没有进入验收 4（数字见 `docs/experiments/2026-11-10-action-latency.md` 第 4 节）。
11-11 主会话把我恢复两次：先做离线归因，GPU 回来后（19:37 起）补第二轮入口、演练与冒烟。**第二轮没有起跑**，原因见下面「待确认」第 1 条。
本次新增提交（分支 `p12-w1-0`）：`dd8c0f3` 修汇总在森林一节的崩溃；`0600d00` 第一轮离线归因脚本与单测；`b605af3` 第二轮汇总与单测；`e43043a` 汇总 K=3 说明改引弹性；
`9b132ee` 第二轮入口 + 桩评测 + 演练 + README；`8b82856` 演练驱动加 `RUN_R2_SH`；文档与本条见 `git log`。不合并、不推送，`progress.md` 没动。

### 做了什么

- `summarize.py` 在森林一节崩溃（`pillar_count` 对 int 取 `len`）：修复并加回归测试，用修复后的脚本重新生成第一轮 `summary.md`（实验目录里起跑时的副本没改）。
- 第一轮离线归因（`analyze_round1.py`、`profile_late_steps.py`、`test_latency_analysis.py` 21 项）：迟到步慢在教师那一步的 A\*，不是 `lr_tie_margin_steps`（第一轮 `teacher_tiebreak=legacy`，只跑一次 A\*）；
  K=3 对 K=1 的稳态位移差里步段 / 电量只解释 25%–28%，主要来自连续前进的构成；验收 2 逐集读数；森林三档读数。结果在 `exp-action-latency-20261110/analysis-round1.md`、`profile-late-steps.md`。
- 第二轮：`summarize_r2.py`（标定阶梯 A/B/C、重跑合并、验收 2 与 4、每臂 cgroup 记录；43 项单测，12 个注入缺陷全部变红）；`run_r2.sh`（预注册 R1–R7 写在文件头）；
  `stub_eval_r2.py` + `rehearse_r2.py`（22 个场景、216 条断言全过；对 `run_r2.sh` 注入 15 个缺陷的结果见报告）。
- 冒烟（1 条车道、不占机器锁、seed 2、L=0.20）：教师 K=1 与学生 K=1 各 1 集都成功、迟到步 0、`uavrgb_infer` 在新卡上有显存占用。
- 改 `README.md`、实验文档第 4–8 节、报告 `w1-0.md`。

### 没做什么

- 没起第二轮守护进程：任务卡给定的前提「位移差来自步段或电量状态不同」没有被离线归因支持，按卡的规定写进报告交回，位移条款的口径由主会话重定。
- 没做 A\* 提速：慢源是 A\*，但任务卡要的「平局推演提速」没有证据支持（平局推演第一轮没跑）。
- 没跑 `check.sh`（主会话说这次不用）；只跑了我改过的测试文件。
- 没碰主仓库的 `progress.md`、`CLAUDE.md`、公共规则里的 GPU 条目，没改第一轮的 `run.sh` 与 `summarize.py`。
- 第二轮的重跑规则只覆盖学生臂；标定臂的迟到步是测量对象，森林臂只报分布，都不重跑。这是我对「含迟到步的集自动重跑一次」的读法，没找到别的解释依据。

### 待确认

1. 位移条款的口径与是否起跑。`cruise`（连续前进第 4 步起，步 3–30）在 L=0.20 对第一轮数据是 +0.0%（区间 -0.8, +0.9）；任务卡口径 `steady`（上一步前进，步 3–30）在 L=0.20 为 -1.8%（区间 -3.8, -0.2），L=0.16 为 -2.5%。推荐 `cruise`。口径写在起跑命令的环境变量 `DISP_CLAUSE` 里，不给不起跑。
2. 「K=3 多飞约 8% 的步」偏低：旧协议 K=3 稳态位移 0.2621 m/步（学生）对新协议 L=0.20 的 K=3 教师 0.2182 m/步低 16.7%，按弹性外推约多飞 20%–22% 的步（未核实）。汇总已改为报实测步数与 sim_s。
3. 起跑条件 ④ 要求拿锁后 30 s 容器平均 CPU < 1.5 核；其他 agent 的常驻进程（`llama-server` 等）可能使它等满 3 小时上限（rc=14），是否放宽由主会话定。

### 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `python sim/tools/test_latency_summary_r2.py` | 43 项 OK；对汇总脚本注入 12 个缺陷，每个至少 1 项变红 |
| `python sim/tools/test_latency_analysis.py` | 21 项 OK |
| `python experiments/2026-11-10-action-latency/rehearse_r2.py all`（提交 `8b82856` 上） | 22/22 场景、216 条断言通过 |
| 对 `run_r2.sh` 注入 15 个缺陷各跑对应场景 | 见报告第 6 节 |
| 冒烟（真 Gazebo，新卡） | 教师 K=1 38 步成功、学生 K=1 73 步成功，两集 `dispatch_lag_s` 恒为 0.200 s，迟到步 0 |
| 第二轮守护进程 | 没起跑 |

### 风险与遗留

- 同 seed 重跑不逐位复现：起飞段结束时刻（第 0 步窗口起点）在第一轮 40 个 K=1 / K=3 集里为 12.7–16.9 s，冒烟的 13.1 s 与 13.9 s 在这个范围内；同 seed 两次的轨迹从第 0 步起就差 0.1 m。标定门作用在 10 集合并量上，单集噪声仍可能把「该过」判成「未过」。
- 冒烟期间容器被限流（`nr_throttled` 增 63），同机还有别的 agent 的 `llama-server` 与 GPU 上的 python 任务，耗时不支持时序结论。
- 数据盘只剩 9.7 GB（已用 97%）；第一轮输出 16 MB（`du -sh`），第二轮的体量只是估计，按同量级算（未核实），远低于 3 GB 的停下线。
