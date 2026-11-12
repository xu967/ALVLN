#!/usr/bin/env bash
# 新实验入口的样板（任务卡 W1-15a）。复制到 experiments/<日期>-<名字>/run.sh（目录深度不变），只改配置区与「臂」那一段，其余由 tools/exp_lib.sh 负责。
# 起：setsid nohup bash experiments/<目录>/run.sh > /dev/null 2>&1 &    看：tail -f <OUT>/run.log    停：tools/stop_daemon.sh <OUT>    说明：README.md
# 流程：配置区 → exp_start（实例锁、核对、冻结代码、run.pid、起跑记录、按池拿锁、Gazebo 池的起跑条件）→ exp_arm … → exp_finish
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../tools/exp_lib.sh"
D=/root/rivermind-data                                   # 数据盘（常量，不可被环境变量改）

# ---------------------------------------------------------------- 配置区：每项一行；同名环境变量可覆盖，覆盖了的项逐个写进 $OUT/launches.log
exp_cfg OUT "$D/uav_rgb/exp-template-demo"               # 输出目录：一个实验一个新目录，续跑沿用同一个
exp_cfg SRC "$HERE/../.."                                # 代码所在的 git 工作树根
exp_cfg CODE_HEAD ""                                     # 冻结哪个提交；空 = 首次取 HEAD，续跑沿用首次的
exp_cfg SELFCHECK 1                                      # 1 = 本脚本与 tools/exp_lib.sh 必须与 CODE_HEAD 里的一致（先提交再起跑）；0 只给演练用
exp_cfg INPUTS ""                                        # 输入文件（空格分隔）：sha256 进起跑记录，续跑时与首次的比，不同就拒绝
exp_cfg POOL ""                                          # 必须自己选：gazebo | gs3d | train | none（machine.lock | gs3d.lock | train.lock | 不拿锁）
exp_cfg LOCK_DIR "$D/uav_rgb/tmp/phase12-cards"          # 池锁所在目录（公共规则 §4 各卡共用）
exp_cfg LOCK_LOG_S 600                                   # 排队等池锁时每 10 分钟记一行，避免刷日志（公共规则 §4）
exp_cfg GPU ""                                           # 用哪张卡（nvidia-smi -L 里的 UUID，旁边写日期与来历）；空 = 不用 GPU，不核对
exp_cfg MIN_CORES 0                                      # cgroup 配额低于这个核数就拒绝起跑（rc=13）；按本实验最低核数填
exp_cfg CPU_MAX_FILE /sys/fs/cgroup/cpu.max              # cgroup v2 配额文件（配额 ÷ 周期 = 核数）；测试里换成假文件
exp_cfg CPU_STAT_FILE /sys/fs/cgroup/cpu.stat            # cgroup v2 用量与限流计数（容器真实忙闲来源，不用宿主机 loadavg）
exp_cfg IDLE_MAX_CORES 8                                 # Gazebo 池起跑条件：窗口内容器平均核数低于它（12 核减 2 条车道约 3.5 核，留 0.5 核）
exp_cfg IDLE_WINDOW_S 30                                 # 测容器平均核数的窗口秒数（公共规则 §4 01:40 条款）
exp_cfg IDLE_RECHECK_S 60                                # 起跑条件不满足时的复查间隔秒数（公共规则 §4 01:40 条款）
exp_cfg IDLE_LIMIT_S 600                                 # 持池锁等空闲的最长秒数（10 分钟）；到点照样起跑并记核数，避免占锁空转
exp_cfg GZ_PATTERN '(^|[ /])gz sim( |$)'                 # 数 gz sim 进程用的 pgrep 模式（避开命令行里带括号的模式串自身）
exp_cfg LANES 2                                          # 本实验自己的参数从这里往下加（Gazebo 池一次不超过 2 条车道，约 3.5 核）

# 臂的命令可以是本脚本里的函数（看得到上面所有配置变量），也可以是外部命令（外部命令要用的变量得 export 或写在命令行上）。环境里有 $ARM 与 $ARM_DIR。
arm_demo() { mkdir -p "$ARM_DIR" && echo "lanes=$LANES arm=$ARM" > "$ARM_DIR/result.txt" && sleep "$1"; }

# 起跑之后的所有步骤包在 main() 里，末行 main 与 exit 写在同一行：bash 执行完外部命令会按旧偏移量重读脚本文件，
# 起跑后文件被原地改写（编辑器、工具直接写文件）会让后面的行读错位；包进函数就整段先解析完，改文件不影响已启动的实例。
main() {
  exp_start                                              # 之后有 $OUT、$CODE（冻结副本，臂的命令从这里跑）、$TMPDIR
  cd "$CODE" || exp_finish 2 "进不了冻结副本 $CODE"

  # 臂：exp_arm <名> <命令…>。完成过的臂（$ARM_DIR/.ok）续跑时跳过；没跑完的旧目录改名 <名>.partial-<时刻> 留着；
  # 起止的 cgroup 快照与时序健康一行自动记。臂不要往 $CODE 里写文件（续跑会核对冻结副本）。
  exp_arm demo-a arm_demo 0
  exp_arm demo-b arm_demo 1
  # 并行的臂：exp_arm x cmd & exp_arm y cmd & wait（exp_finish 之前必须 wait）

  exp_finish 0 "全部臂跑完"                              # 有臂失败时 DONE 的 rc 自动改成 11 并列出失败的臂；被 kill 进程组时落 rc=130
}
main "$@"; exit
