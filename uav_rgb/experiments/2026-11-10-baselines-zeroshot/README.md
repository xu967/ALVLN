# baselines-zeroshot：公开权重零样本基线在柱阵闭环里的表现

问题：PULP-Dronet v3、Tiny-PULP-Dronet v3、DroNet 的公开权重不训练、不微调，直接放进我们的柱阵闭环，走廊内成功率是多少。
这只量域差异。它不能说明哪种网络结构更好：结构对比要用「同结构在我们数据上重训」的臂（`sim/tools/baselines/README.md` 的 `BASELINE_WEIGHTS`）。

预注册（预测、证伪、主比较、参照）写在 `run.sh` 文件头，起跑前冻结。结果出来后在 `RESULT.md` 里对照写，本目录目前没有 `RESULT.md`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 入口。先对 6 个臂的正式命令行做不起 Gazebo 的预检，再同时等 `exp-rule-oracle-20261110/DONE` 与 `exp-long-course-20261110/DONE`（上限 3 小时），然后 5 条车道评测，最后跑 `summarize.py` 并落 `DONE` |
| `summarize.py` | 汇总：每臂走廊内成功、撞柱、越界、未到线、SPL、动作占比、护栏改写占比，配对 McNemar，对参照臂（学生、教师、盲前进） |
| `offline_check.py` | 起跑前在 train 分区教师帧上的离线核对：`meta`（逐帧最近柱方位与距离）、`infer <基线>`（各基线逐帧输出，含左右翻转输入）、`analyze`（镜像检验、转向符号、碰撞概率分布、动作占比）、`dronet-verify`（DroNet 前向的独立 NumPy 复核） |
| `check_h5_vs_npz.py` | DroNet 的 npz 权重与 Keras 原 h5 逐张量对拍（需 `$D/baselines/venv-h5`） |

## 臂（顺序即优先级，每臂 20 集，test 分区的 20 个冻结 seed）

| 臂 | 启动器 | 评测端护栏 |
| --- | --- | --- |
| `pulp_v3` | `launch_pulp_v3.sh` | 无 |
| `pulp_tiny` | `launch_pulp_tiny.sh` | 无 |
| `pulp_v3-guard2` / `pulp_tiny-guard2` | 同上 | `--heading-guard 2` |
| `dronet` | `launch_dronet.sh` | 无 |
| `dronet-recal`（次要） | `launch_dronet_recal.sh` | 无 |

5 条车道 6 个臂：前 5 个同时起跑，`dronet-recal` 接在第一个收尾的臂之后。`dronet` 没有护栏臂：离线核对显示它忠实映射下单帧 FORWARD 占比为 0，护栏对只会悬停的策略没有作用空间。
这是 2026-11-10 起跑前对预注册的修订（当时没有任何闭环数据），修订时间、理由与原文都记在 `run.sh` 文件头。

评测命令的其余部分与 `2026-11-10-heading-guard` 相同：`--task base --split test --steps 80 --hold 1.0 --window-from observation --dynamics cf_sitl --vehicle e1_mini --corridor-half-width 3.0`，口径 `sim09.v2.coast1s.corridor3m`。

## 运行

```bash
D=/root/rivermind-data/uav_rgb; mkdir -p $D/exp-baselines-zeroshot-20261110 && cd $D/exp-baselines-zeroshot-20261110
cp <仓库>/experiments/2026-11-10-baselines-zeroshot/{run.sh,summarize.py} .
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
```

输出在 `$D/exp-baselines-zeroshot-20261110/`：`run.log`（进度）、`<臂>/eval.json`、`<臂>.log`、`<臂>.baseline.jsonl`（闭环里基线每步的 steer 与 p_coll，按 pid 切开各集）、`inputs.txt`（提交号与文件哈希）、`summary.md`、`DONE`。
`run.sh` 里的 `COMMIT` 是冻结代码的提交，用 `git archive` 解到 `$D/tmp/phase12-20261110/baselines-run/code-<哈希前 7 位>`，之后分支怎么动都不影响本实验。
