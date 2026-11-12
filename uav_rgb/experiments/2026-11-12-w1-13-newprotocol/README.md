# W1-13 插件侧动作生效时刻测量（先量后定）

本目录对应任务卡 W1-13 第一部分第 3 条（审核 A 发现 5）：在 Gazebo 池锁（`machine.lock`）内先后跑冒烟（教师 K=1，train 分区 seed `2,3`）与两个测量条件：
- `idle`：教师 K=1，train 分区 seed `2,3,4,6,7`（共 5 集，不用冻结评测 seed），`--action-latency 0.20`，不加占核进程；
- `load`：同上 5 集，在测量窗口内另起 8 个纯 CPU 占核进程（同一进程组，退出或信号终止时由 `trap` 按 PGID 核对命令行后清理）。

## 运行与监控

```bash
OUT=/root/rivermind-data/uav_rgb/exp-w1-13-ack-20261112
mkdir -p "$OUT"
cp experiments/2026-11-12-w1-13-newprotocol/run_smoke_ack.sh "$OUT/"
cd "$OUT" && setsid nohup bash run_smoke_ack.sh > /dev/null 2>&1 < /dev/null &
```

- 查看进度：`tail -f $OUT/run.log`
- 随时生成汇总：`/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python $OUT/code/experiments/2026-11-12-w1-13-newprotocol/summarize_ack.py $OUT`
- 完成标记：`$OUT/DONE`（`rc=0` 为全部完成且数据达标）
- 停止守护进程：先用 `ps -o lstart=,args= -p $(cat $OUT/run.pid)` 核对与 `$OUT/run.meta` 一致，再执行 `kill -- -$(cat $OUT/run.pid)`

## 离线测试与演练

```bash
nice -n 10 /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python sim/tools/test_w113_summaries.py
nice -n 10 bash experiments/2026-11-12-w1-13-newprotocol/rehearse_smoke_ack.sh /root/rivermind-data/uav_rgb/tmp/phase12-cards/w1-13/tmp/rehearse_check
```
