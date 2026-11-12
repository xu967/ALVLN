# 新实验入口起步说明（`experiments/_template/`）

本目录是守护进程实验入口的最小样板，配合 `tools/exp_lib.sh` 与 `tools/stop_daemon.sh` 使用。锁、冻结代码、起跑记录、池排队与起跑条件、每臂时序健康、完成标记统一由 `tools/exp_lib.sh` 处理，入口脚本只写预注册、配置区与各臂命令。

## 1. 新实验起步步骤

1. **复制样板（保持目录深度不变）**
   ```bash
   mkdir -p experiments/<YYYY-MM-DD>-<slug>
   cp experiments/_template/run.sh experiments/<YYYY-MM-DD>-<slug>/run.sh
   ```
   目录深度固定为 `experiments/<目录>/run.sh`，以便脚本内 `$HERE/../../tools/exp_lib.sh` 定位公共库。

2. **在文件头写预注册，改顶部配置区**
   - 起跑前在文件头注释写清预注册：预测、证伪条件、主比较（不超过 3 个）。
   - 在配置区逐行改 `exp_cfg` 字面量，每行末尾注明取值依据：
     - `OUT`：数据盘上的独立输出目录（例：`/root/rivermind-data/uav_rgb/exp-<slug>-<YYYYMMDD>`）。
     - `POOL`：必须显式选池：`gazebo`（排 `machine.lock`，带 8 核 / 600 s 起跑条件）、`gs3d`（排 `gs3d.lock`）、`train`（排 `train.lock`）、`none`（纯离线分析或自管多把锁，不碰单池锁）。留空会被拒绝（`rc=2`）。
     - `INPUTS`：依赖的权重、数据集 manifest 或路线文件路径（空格分隔）；起跑时记 sha256，续跑时比对，不一致则拒绝（`rc=4`）。
     - `GPU`：用 CUDA 时填 `nvidia-smi -L` 查到的显卡 UUID，并在行尾注明核实日期与来历；不用 GPU 则留空。
   - 在 `main()` 内用 `exp_arm <臂名> <命令...>` 串起各臂。此时已切入冻结副本 `$CODE`，命令里的脚本与模块一律从 `$CODE` 读取，不回活工作树取代码，也不往 `$CODE` 里写产物。

3. **先提交代码，再以守护进程起跑**
   `SELFCHECK=1`（缺省）要求入口脚本与 `tools/exp_lib.sh` 同 `CODE_HEAD` 提交里的内容逐字节一致。先在自己的分支提交，再用 `setsid` 启动：
   ```bash
   setsid nohup bash experiments/<YYYY-MM-DD>-<slug>/run.sh > /dev/null 2>&1 < /dev/null &
   ```
   启动后核对进程存活：
   ```bash
   cat <OUT>/run.pid
   ps -o pid,pgid,etime,cmd -p "$(sed -n 's/^pid=//p' <OUT>/run.pid)"
   ```

## 2. 查看进度、停止与续跑

- **看进度与记录**：
  - 实时日志：`tail -f <OUT>/run.log`（`exp_start` 缺省导出 `UAVRGB_CUDA_BLOCKING_SYNC=1` 让训练器等 GPU 时不自旋，外部已设 `0` 或 `1` 则沿用外部值并在日志记录）
  - 起跑与续跑追加记录（提交号、冻结副本 sha256、环境变量覆盖项、`cuda_blocking_sync`、输入 sha256、等锁时长、Gazebo 起跑核数、各臂退出码）：`cat <OUT>/launches.log`
  - 每臂 cgroup 时序健康摘要（墙钟、平均核数、`nr_throttled` 增量、被限流时长及占「墙钟×核数配额」比例）：`cat <OUT>/health.txt`（原始起止快照在 `<OUT>/<臂名>.cg`）
  - 完成标记：`cat <OUT>/DONE`（`rc=0` 为全部臂成功；有臂失败自动改为 `rc=11` 并列出失败臂名；被信号终止落 `rc=130`；脚本中途异常退出也由 `EXIT` 钩子补写非零 `DONE`）
- **停止守护进程**：
  ```bash
  bash tools/stop_daemon.sh <OUT>
  ```
  `stop_daemon.sh` 会核对 `<OUT>/run.pid` 中的 PID 启动时刻（`/proc/<pid>/stat` 第 22 列）、命令行与进程组一致后，才执行 `kill -TERM -- -<PGID>`；容器重启后旧 `run.pid` 指向别的进程时会拒绝发送信号（`rc=1`）。禁止用 `pkill -f`。
- **断点续跑与汇总要求**：
  用原命令重新起跑即可。已有 `<OUT>/<臂名>/.ok` 的臂自动跳过；未跑完的旧目录自动改名为 `<臂名>.partial-<时刻>` 保留并在 `launches.log` 留痕，随后重做该臂。实验的汇总脚本必须扫描并列出 `*.partial-*`，交代哪些臂重评过以及半成品里已跑完的集数与结果。

## 3. 演练与冒烟约定

正式运行不靠环境变量临时改配置；确因冒烟或演练覆盖了 `exp_cfg` 变量（如 `SELFCHECK=0`、`OUT=...`、`LOCK_DIR=...`），所有覆盖项都会逐条记入 `<OUT>/launches.log` 的 `override` 行。演练时不占用真实的 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/*.lock`，应将 `LOCK_DIR` 指到自己的草稿临时目录。
