# 2026-11-10 动作派发时刻固定到仿真时间：标定与验收（任务卡 W1-0）

入口 `run.sh`（预注册写在文件头，起跑时冻结代码到 `$OUT/code`），汇总 `summarize.py`。输出目录 `/root/rivermind-data/uav_rgb/exp-action-latency-20261110/`。

| 文件 | 作用 |
|---|---|
| `run.sh` | 两阶段守护进程：阶段 1 = 标定（教师 K=1 / K=3 各 10 个 train 分区 seed，L=0.16）+ 验收 2（学生 K=1 种子 1）+ 验收 3（森林三档延迟）；标定门（`summarize.py --gate`）；L=0.16 未过则试 L=0.20；过门才进阶段 2 = 验收 4（学生三个训练种子，K=1 对 K=3 第 1 块）。配额闸等 `exp-rule-oracle-v1-20261110/DONE`，`LANES=2`。 |
| `summarize.py` | 写 `summary.md`：标定表（每个 L 的生效时长、稳态前飞位移、转角、`dispatch_lag_s` 中位与最大、迟到步占比、每集第 0 步）、验收 2 的逐集配对、森林延迟分布与需要的 `--rtf`、验收 4 的 McNemar 与预注册判定、全部臂的延迟与迟到步总表。`--gate L` 只判标定门（退出码 0 PASS / 1 FAIL / 2 INCOMPLETE）。 |
| `stub_eval.py`、`rehearse.sh` | 演练 `run.sh` 的流程用：假评测入口按真实格式写 `eval.json` 与轨迹，不起 Gazebo、不占 gz。`bash rehearse.sh ladder\|pass016\|fail\|fwd` 各走一条分支。 |

运行（仓库根之外任意处）：

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-action-latency-20261110
mkdir -p $OUT && cp experiments/2026-11-10-action-latency/{run.sh,summarize.py} $OUT/ \
  && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
```

- 进度：`tail -f $OUT/run.log`；各臂日志 `$OUT/<臂>.log`；结束标记 `$OUT/DONE`（`rc=… 时间 原因`）。
- 随时汇总：`python $OUT/summarize.py`（写 `$OUT/summary.md`）；只看标定门：`python $OUT/summarize.py --gate 0.16`。
- 停：`kill -- -$(cat $OUT/run.pid)`（只按进程组，不用 `pkill -f`）。续跑：同一条命令重起，评完的臂跳过。
- 臂目录：`cal-L<L>-k<K>`（教师标定）、`acc-L<L>-k<K>-s<种子>`（学生验收）、`forest-<密度>`。`L_STAR` 文件记阶段 2 用的 L。

## 第二轮（2026-11-11）：标定阶梯重跑、学生验收、森林

第一轮的标定门在 L=0.16 与 L=0.20 都没过，`DONE rc=10`，没有进入验收 4。第二轮是看到这个结果之后设计的，预注册（条款 R1–R7）写在 `run_r2.sh` 文件头，起跑后不改。
第一轮的 `run.sh`、`summarize.py`、`stub_eval.py`、`rehearse.sh` 保持原样，第二轮的数据不与第一轮混报。输出目录 `/root/rivermind-data/uav_rgb/exp-action-latency-r2-20261111/`。

| 文件 | 作用 |
|---|---|
| `run_r2.sh` | 守护进程。起跑条件（配置的 GPU 在 `nvidia-smi -L` 里、cgroup 配额 ≥ 12 核、拿到机器锁、拿锁后没有 gz sim 进程且 30 s 窗口内容器 CPU < 1.5 核）→ 冻结代码 → 标定阶梯（教师 K=1 / K=3 各 10 个 train seed；A 组 L=0.20，B 组 K=3 带 `--rtf 0.5`，C 组 L=0.24）→ 过门后学生 calswa 三个训练种子 + 护栏 2，K=1 与 K=3 各 20 个冻结 seed（验收 2、4；含迟到步的学生集自动重跑一次）→ 森林三档。`LANES=2`。位移条款的口径用环境变量 `DISP_CLAUSE=cruise\|steady` 显式给，不给不起跑（rc=5）。 |
| `summarize_r2.py` | 写 `summary.md`：阶梯表（每组每条款的判定）、验收 2 与 4 的配对检验和预注册判定、每臂时序健康（平均核数、限流增量）、K=3 的 S_1/S_2/S_3 与掉高集数、旧协议并排。`--gate A\|B\|C` 只判一组（退出码 0 PASS / 1 未过且含迟到步 / 2 数据不全 / 3 未过且没有迟到步）；`--late-seeds <臂目录>` 列出含迟到步的 seed。 |
| `stub_eval_r2.py`、`rehearse_r2.py` | 演练 `run_r2.sh` 的 22 个场景（阶梯三组、只因位移停、没给口径、GPU 不在、配额不足、等锁、等空闲、迟到集重跑、限流标记、续跑、重复实例、信号终止、半成品…），每个场景带自动断言，一个场景不过退出码 1；不起 Gazebo、不碰真实机器锁。`python rehearse_r2.py all`，或只跑某个场景。 |
| `analyze_round1.py`、`profile_late_steps.py` | 第一轮的事后分析，只读第一轮输出目录：迟到步清单与集中度、稳态位移差的归因、验收 2 的逐集配对、森林读数；迟到步的离线计时与 A* 展开量。结果在 `exp-action-latency-20261110/analysis-round1.md` 与 `profile-late-steps.md`。 |

对应的测试在 `sim/tools/`：`test_latency_summary_r2.py`（43 条）、`test_latency_analysis.py`（21 条）。

运行（主会话定了位移口径、核对过起跑条件之后；只用已提交的版本，脚本起跑时拿自己与 `CODE_HEAD` 里的同名文件比哈希，不一致 rc=4）：

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-action-latency-r2-20261111
mkdir -p $OUT && cp experiments/2026-11-10-action-latency/run_r2.sh $OUT/ \
  && cd $OUT && DISP_CLAUSE=cruise setsid nohup bash run_r2.sh > /dev/null 2>&1 < /dev/null &
```

- 进度：`tail -f $OUT/run.log`（每个臂开始与结束各记一次容器 `usage_usec` 与 `nr_throttled`）；结束标记 `$OUT/DONE`（`rc=… 时间 原因`，rc 含义见 `run_r2.sh` 文件头）；`$OUT/start.txt` 记起跑条件的实测值。
- 随时汇总：`.venv-tools/bin/python $OUT/code/experiments/2026-11-10-action-latency/summarize_r2.py $OUT`（写 `$OUT/summary.md`，缺的臂标「未完成」）；只看某组标定门：同一脚本加 `--gate A|B|C`。
- 停：`kill -- -$(cat $OUT/run.pid)`（只按进程组，不用 `pkill -f`）。续跑：同一条命令重起，评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>`（不删），沿用起跑时的 `CODE_HEAD` 与位移口径。
- 臂目录：`cal-L0.20-k1|k3`、`cal-L0.20rtf0.5-k3`、`cal-L0.24-k1|k3`（教师标定）、`acc-L<L*>-k<1|3>-s<0|1|2>`（学生验收，含迟到步的集在 `<臂>-retry1`）、`forest-L<L*>-<sparse|medium|dense>`。`L_STAR`（与 `RTF_STAR`）文件记过门的参数。
- 演练：`python experiments/2026-11-10-action-latency/rehearse_r2.py all`（临时目录在 `$TMPDIR` 下的 `rehearse-r2-<场景>-*`，不自动删）。演练用已提交的 `HEAD` 冻结代码，改了 `summarize_r2.py` 要先提交再演练；场景 `selfcheck` 只在 `run_r2.sh` 已提交时有意义。
