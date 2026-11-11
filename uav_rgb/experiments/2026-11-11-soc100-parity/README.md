# 满电档（SOC=100%）对 50% 档的验收实验

任务卡 W1-10（`docs/briefs/2026-11-10-phase12-task-cards.md`「11-11 更新」）。同一批权重、同一批 seed、同一协议（`--window-from observation --action-latency <L_STAR>`），只换起始电量：
SITL 默认从 50% 电量起飞，一集最多约 100 步；满电副本把 `sim.battery_soc_init` 改成 1.0，用来做 K=3 长课程。
本目录回答一件事：满电档的单步动力学与成败，是否与已有的 50% 档数据可以互用（预注册判据逐字在 `run.sh` 头部）。

## 文件

| 文件 | 作用 |
|---|---|
| `run.sh` | 守护进程入口：起跑条件、冻结代码、探针、六个学生臂、配对汇总与判决。头部写了预注册、条款、运行命令、DONE 的 rc 表 |
| `summarize_parity.py` | 配对汇总与预注册判决（`judge` 是纯函数，有单测）。随时可跑，缺的臂标「未完成」 |
| `probe_plugin_params.py` | Gazebo 上的 A/B 探针：default 与 copy 两种世界各起 3 个 `gz sim -v 3`，读静止电压、READY 遥测与插件日志的 `params=` 行，证明插件读的是副本 |
| `test_soc100_parity.py` | 单测：判决阈值与边界、配对计数、位移与转角口径与 `summarize_r2.py` / `summarize.py` 一致、探针的纯函数、`run.sh` 头部预注册与任务卡逐字一致 |
| `stub_eval.py`、`rehearse.py` | 演练：假评测入口加 25 个场景，走 `run.sh` 的每个分支（不起 Gazebo） |

## 运行（实验）

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-soc100-parity-20261111
mkdir -p $OUT && cp experiments/2026-11-11-soc100-parity/run.sh $OUT/ && cd $OUT
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
```

- 起跑前 `run.sh` 必须是已提交的版本（脚本拿自己与 `CODE_HEAD` 里的同名文件比哈希，不一致 rc=4）。
- 进度：`tail -f $OUT/run.log`。结束落 `$OUT/DONE`（rc、时间、原因、判决）、`VERDICT`、`summary.md`。
- 停：`kill -- -$(cat $OUT/run.pid)`（只按进程组）。续跑：同一条命令重起，评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>`，探针通过后不重跑。
- 对照在第二轮目录 `$D/exp-action-latency-r2-20261111`（只读）；第二轮没有 `DONE` 或 rc 不是 0 时本脚本 rc=20，不起任何臂。
- 重实验：脚本阻塞等 `$D/tmp/phase12-cards/machine.lock`，持有到退出；同机别的卡的 Gazebo 也排这把锁。

## 运行（自检）

单测、演练都不起 Gazebo，但 `tools/check.sh` 只扫 `sim/tools/test_*.py` 与 `tools/test_*.py`，本目录的测试要单独跑：

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$D/tmp/<自己的草稿目录>/tmp
PY=/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python
nice -n 10 $PY experiments/2026-11-11-soc100-parity/test_soc100_parity.py                        # 单测，约 20 s
cd experiments/2026-11-11-soc100-parity && REHEARSE_SRC=<已提交的 git 仓库> $PY rehearse.py all   # 25 个场景，约 4 分钟
```

演练里冻结代码用 `git archive HEAD`，`summarize_parity.py` 也从冻结副本里跑，所以改了脚本要先提交再演练；`REHEARSE_SRC` 可指向另一个 git 仓库（工作树快照），
`RUN_SH` 指向另一份 `run.sh`（变异检查）。临时目录在 `$TMPDIR` 下的 `rehearse-soc100-<场景>-*`，不自动删。

## 判决

`PASS`：P1..P6 全过；`FAIL`：有项没过且没触发证伪；`FALSIFIED`：K=1 位移或转角相对差大于 10%，两档数据与结论不能互用；`INCOMPLETE`：臂没跑完；
`INVALID`：指纹（对照 `1bc8fedf`、满电 `06e5f017`）或配对协议键不一致，不给 P1..P6。判决写入 `DONE` 文本、`VERDICT`、`summary.md`；`DONE` 的 rc 不随判决变，rc 表见 `run.sh` 头部。
