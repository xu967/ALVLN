# 教师 v7 闭环对拍（任务卡 W1-7 验收 2）

教师 v7 = `TEACHER_VERSION uavrgb.teacher.v7`：B2（机体越过最远柱后表面后，A* 目标的 y 取当前 y，目标由点变线）加 B3（A* 在标准余量无路时按 0.30、0.20 m 各重试一次）。
本目录回答两个问题：教师在 20 个冻结 seed 上是否仍 20/20；出口区（x > 最远柱 x）的 YAW 步占比是否从约 17.7% 降到 5% 以下。

预注册（预测、证伪条件、口径、起跑条件）写在 `run.sh` 文件头，起跑前提交，起跑后不改。

| 文件 | 作用 |
|---|---|
| `run.sh` | 守护脚本：冻结代码（`git archive`）、阻塞式机器锁、两个臂各两半（`v7-a v7-b v5-a v5-b`，每半 10 个 seed，2 条车道）、汇总、`DONE` |
| `summarize.py` | 合并两半，算成功率与出口区 YAW 占比，写 `summary.md`、`summary.json`；`--self-check` 用合成数据自检 |
| `offline_0417.py` | 离线证据：0417 的 12000 个状态上，v5 与 v7 规则的标签对比（同状态重标，不是新轨迹），含 path_none 步数与耗时 |
| `kinematic_preview.py` | 起跑前的预演：教师规则 + 运动基元 + 噪声的运动学代理（不是 Gazebo），20 个冻结 seed × N 次重复，给出 v5、v7 出口区 YAW 占比与「单次 20-seed 评测 < 5%」的概率；预注册 P2 的依据 |

两个臂：`v7` 主臂用分支 `p12-w1-7` 的冻结代码；`v5` 对照臂用 main 开工时的 HEAD（`a714b54`），同 seed、同参数、同一个守护进程里先后跑，只作配对参照，不设门。

输出：`$D/exp-2026-11-12-teacher-v7-parity/`（`$D=/root/rivermind-data/uav_rgb`）。

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-2026-11-12-teacher-v7-parity
mkdir -p $OUT && cp run.sh $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
tail -f $OUT/run.log                     # 进度；排队期间每 10 分钟一行
kill -- -$(cat $OUT/run.pid)             # 停（只按进程组）
# 续跑：同一条 setsid nohup 命令；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>
```

结果与解读见 `docs/experiments/2026-11-12-teacher-v7-parity.md`。
