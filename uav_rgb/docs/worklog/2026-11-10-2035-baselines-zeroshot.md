# 外部基线包装器入库、离线核对、零样本实验排队

- 时间：2026-11-10 20:04 到 20:5x（50 分钟时间盒）
- 范围：`sim/tools/baselines/`、`sim/tools/test_baseline_infer.py`、`experiments/2026-11-10-baselines-zeroshot/`；分支 `p12-baselines`（基于 main 155e377）
- 结论：包装器与实验入口完成并提交；守护进程在等 `exp-rule-oracle-20261110/DONE` 与 `exp-long-course-20261110/DONE`（起跑前已按主会话意见修订，见「关键决策」），闭环结果还没有。
- 完整报告（含离线核对全部表格与两张后续任务卡）：`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/reports/baselines-run.md`

## 做了什么

- 把草稿包装器整理入库：配置区（环境变量覆盖、每参数一行带出处）、映射纯函数 `map_steer_coll` / `map_velocity`、PULP v3 / Tiny / DroNet 适配器；`BASELINE` 无默认值，漏设报错退出 2；去掉了没跑通的 `da2_rule`、`trailnet`、队列规则；`BASELINE_LOG` 每行带 `pid`，按集切开。
- 5 个一行启动器、README（协议、映射与阈值出处、符号证据、怎么加基线、下载与 h5 转 npz 命令、`stub.ckpt` 一行命令）、测试 9 个函数。
- 离线核对（train 分区 255 集 10,200 帧，不起 Gazebo）：镜像检验、转向符号、碰撞概率分布、默认映射动作占比、DroNet NumPy 独立复核与 h5 对拍。结果写进启动器：PULP v3 与 Tiny 符号 +1（实测），DroNet 符号判不出沿用 +1，重标定臂 `P_CRIT=0.9994456`、S=90。
- 实验入口 `run.sh`（预注册在文件头；冻结代码；6 个臂的不起 gz 预检；排队等 rule-oracle 与 long-course 两个 `DONE`；5 车道）与 `summarize.py`。最初是 7 个臂、等 rule-oracle、2 车道，起跑前修订成现状（见「关键决策」）。
- 提交：57cf134 `feat(baselines)`，8991e00 `feat(experiments)`，96774f0 `docs(worklog)`；起跑前修订在其后的 `fix(experiments)` 提交里（哈希见 `git log`），该提交同时修订了本文件。
- 守护进程：第一个（PGID 1433245，20:33 起）只做过预检与等待，没有起过任何臂，按修订停掉；用修订后的脚本重起，新 PGID 与时间见 `/root/rivermind-data/uav_rgb/exp-baselines-zeroshot-20261110/run.pid` 与 `run.log` 的「开始」行，也记在报告第四节。

## 关键决策

- 映射用最近邻量化：`|steer·S·hold| >= 12.5°`（半个 25° 步长）转向优先，其余看碰撞概率 `<= 0.7`；S 取 PULP 90 °/s、DroNet 20 °/s，出处在 README。
- 符号按「裁剪视场内右减左的 steer 均值差，按集自助法区间是否含 0」判定；判不出就如实写「无有效转向信号」并沿用约定 +1，不硬凑。
- 预检不起 gz：用正式命令行跑 `eval_closed_loop.py` 到非空 `--out` 的报错为止，覆盖参数解析、启动器、权重、协议握手与预热。
- 重标定臂按派工原样保留（阈值取校准帧中位数、S=90），离线预测它在原地左转；主会话确认保留，排在队尾。
- 起跑前修订预注册（2026-11-10 20:44，主会话 20:41 批准，当时没有任何闭环数据；原文与理由写在 `run.sh` 文件头）：① 去掉 `dronet-guard2` 臂（DroNet 忠实映射下单帧 FORWARD 占比 0，护栏没有作用空间，每集空耗 80 步），主比较 ③ 只对 PULP 两个基线；② 起跑条件改为同时等 rule-oracle 与 long-course 两个 `DONE`，等待上限仍 3 小时（从重起算起）；③ 车道 5，臂序 pulp_v3、pulp_tiny、pulp_v3-guard2、pulp_tiny-guard2、dronet、dronet-recal。冻结代码不变：修订只改 `experiments/`，`sim/` 与 `tools/` 下的代码字节不变，仍用 57cf134 的冻结目录。
- 第三个提交放本文件：派工要求两次提交，过程文档按个人规范必须随代码提交，单独成 `docs:` 提交保持一个主题一次提交。

## 没做什么

- `BASELINE_PREP` 的敏感性闭环臂与离线对比：派工臂表里没有，时间盒内先保证 6 臂预检与排队。
- DroNet 与 Keras/TF 原模型的数值对拍：本机没有 TF 1.x。
- 任何 Gazebo 闭环：派工要求冒烟不起 Gazebo，gz 配额被 rule-oracle 与 long-course 占满。
- `da2_rule`、三条带适配器、同结构重训：留给两张后续任务卡（报告第六节）。
- 修改 `collect.py`、`eval_closed_loop.py`、`scene_gen.py`：派工禁止。
- 改仓库 `progress.md`：多个 worktree 并行，同一文件追加会在合并时冲突；进度记在本文件，合并时汇入。
- `RESULT.md`：闭环结果还没有。
- 推送、动 main、删除任何已有文件：没有。

## 待确认

- 闭环结果出来后，是否按结果启动卡 A（同结构重训）与卡 B（DA-V2 米制 + 规则）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/tools/test_baseline_infer.py` | PASS，含 5 个真实启动器经 `StudentPolicy` 的往返 |
| 红到绿 | 探针测试 `No module named 'torch'` 修后通过；启动器检查 `could not convert string to float: '0.9994456：DroNet'` 修后通过 |
| 变异检查 | 5 个故意改坏的映射版本全部被抓到，存活 0 |
| DroNet 权重与前向 | h5 对 npz 48 数据集一致；torch 前向对 NumPy 解释器最大绝对差 ≤ 5.0e-6；与 Keras/TF 对拍没做 |
| 预检（不起 gz） | 修订前 7 个臂通过（`run.log` 20:33:32 到 20:34:03）；修订后 6 个臂的预检见 `run.log` 与报告第四节 |
| `run.sh` 控制流 | 假 python 副本跑通：预检、等待、车道、汇总、`DONE`。修订后重跑：6 个臂预检通过；两个起跑文件缺一个时 `run.log` 只列缺的那个；都到之后 5 个臂同时起、`dronet-recal` 接在后面；3 小时上限的超时路径落 `DONE` rc=5 并列出缺的文件、没有起任何臂 |
| `tools/check.sh --python-only` | rc=0（20:31:55，含 `test_baseline_infer: PASS`） |
| 闭环第一集与各臂成功率 | 未验证：守护进程还在等两个 `DONE` |

## 风险与遗留

- 守护进程的起跑时间是两个 `DONE` 里较晚的那个：rule-oracle 约 20:50，long-course 预计 21:40 到 22:00（均为主会话估计，未核实）；3 小时上限从重起算起，超时则 `DONE` 写 rc=5。
- DroNet 的两个臂（dronet、dronet-recal）预计前进很少（recal 预计原地左转），每集跑满 80 步，耗时偏长。
- DroNet 的符号与数值正确性只靠自洽核对，没有第三方实现验证。
- 停止：`kill -- -$(cat run.pid)`；续跑：同一条启动命令（评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>`）。
