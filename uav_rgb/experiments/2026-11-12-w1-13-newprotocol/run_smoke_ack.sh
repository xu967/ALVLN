#!/usr/bin/env bash
# 任务卡 W1-13：Gazebo 冒烟 + 插件确认时刻 plugin_ack_sim_t 的测量（先量后定）。一个守护进程，三个臂，顺序执行，只拿 machine.lock。
#
# ---------------------------------------------------------------- 预注册（起跑前写下，起跑后不改；要改就换文件名与输出目录）
#   冒烟（臂 smoke）：教师 K=1（--task base --steps 80），train 分区 seed 2,3 各 1 集，--action-latency 0.20，默认 RTF，单进程。
#     通过条件（summarize_ack.py --smoke-check，与该脚本文件头一致）：eval 正常结束（有 eval.json 且不是 partial）；每一行都有 safe_gap_s、obs_gap_s、
#     plugin_ack_sim_t 三个键；每集第 0 步前两个为 null、其余非 null；plugin_ack_sim_t 非 null 的行 ≥ 95%；非 null 的确认时刻不早于 Python 侧派发戳减 6 ms。
#     不通过：DONE rc=20，不进入测量（测量没有意义），释放锁。
#   测量（臂 idle 与 load）：教师 K=1，train 分区 seed 2,3,4,6,7 各 1 集（共 5 集，两个臂同 seed），--action-latency 0.20，默认 RTF。
#     idle = 不加占核进程（容器里别的 agent 的负载照旧，平均核数记在 <臂>.cg 与 summary.md，所谓「空闲」就是这个读数）；
#     load = 另起 8 个占核进程（每个一个死循环，同一个进程组，臂结束按 PGID 停，停之前核对命令行）。
#     量：每步 plugin_ack_sim_t − (pose_ts + L)，pose_ts 是观测位姿戳（迟到判据的 target_ns = 观测戳 + L 用的就是它）；另报 Python 侧派发 − (观测戳 + L)、确认 − Python 侧派发。
#   判定（任务卡第一部分第 3 条）：每个条件分别看确认差的 p99：两个条件都 < 10 ms，只记录，迟到判据不动；任一条件 ≥ 10 ms，迟到判据改用 plugin_ack_sim_t，
#     阈值按量到的分布定并写明依据（后续提交，不在本脚本里）。预期（先写下）：空闲 p99 在 5–10 ms 一带（state 5 ms 一帧，加动作消息到插件的排队）；满载下尾部可能变长，未核实。
#   起跑条件（公共规则 §4「2026-11-12 01:40 起」，主会话定）：拿到 machine.lock 之后，先核对没有 gz sim 进程，再看 30 s 窗口内容器 CPU（cpu.stat usage_usec 增量）：
#     < 8 核（IDLE_MAX_CORES）就起跑；不满足每分钟复查，最多等 10 分钟（IDLE_LIMIT_S），到点照样起跑，把当时的核数与 gz sim 进程数写进 start.txt（forced_start=是）。
#     另：cgroup 配额 < 12 核不起跑（DONE rc=13）。本脚本是教师臂，不用 GPU，不查 nvidia-smi（GPU 信息只记进 start.txt）。
#
# 运行：mkdir -p $OUT && cp run_smoke_ack.sh $OUT/ && cd $OUT && setsid nohup bash run_smoke_ack.sh > /dev/null 2>&1 < /dev/null &
#   起跑前必须是已提交的版本：脚本拿自己与 CODE_HEAD 里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive CODE_HEAD，解到 $OUT/code，逐文件核对 sim/ 的 git blob 哈希，评测与汇总都从冻结副本里跑。
#   续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删）。
# 进度：tail -f $OUT/run.log；汇总随时可跑：<解释器> $OUT/code/experiments/2026-11-12-w1-13-newprotocol/summarize_ack.py $OUT（缺的臂标「未完成」）。
#   结束落 DONE（rc、时间、原因）与 summary.md。停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f；停之前核对 run.pid 对应进程的启动时刻与命令行）。
# DONE 的 rc：0 冒烟过且两个测量臂跑完且数据达标；3 已有实例（不写 DONE）；4 冻结代码或脚本版本不一致；11 有臂没跑完；12 测量臂有效轨迹不足；13 cgroup 配额不足；20 冒烟没过；130 被信号终止。
# 时间估计（未核实）：冒烟 2 集约 3 分钟，每个测量臂 5 集约 6–10 分钟，合计约 20–25 分钟，另加等锁。
set -u
export LC_ALL=C

# 捕获外部环境变量对配置项的覆盖（公共规则 §6 E-10：逐项记进 start.txt 与 inputs.txt）
OVERRIDE_KEYS=(OUT SRC CODE_HEAD L HOGS K1_EPT MIN_CORES IDLE_MAX_CORES IDLE_WINDOW_S IDLE_RECHECK_S IDLE_LOG_S IDLE_LIMIT_S LOCK_LOG_S MACHINE_LOCK NVIDIA_SMI CPU_MAX_FILE CPU_STAT_FILE GZ_PATTERN GZ_PROBE EVAL SELFCHECK)
ENV_OVERRIDES=""
for _k in "${OVERRIDE_KEYS[@]}"; do
  if [[ -n "${!_k+x}" ]]; then
    ENV_OVERRIDES+="${ENV_OVERRIDES:+; }${_k}=${!_k}"
  fi
done
ENV_OVERRIDES=${ENV_OVERRIDES:-none}

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=${OUT:-$D/exp-w1-13-ack-20261112}
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-w1-13}   # 代码源（分支 p12-w1-13）：只用来 git archive 冻结代码
CODE_HEAD=${CODE_HEAD:-}                                 # 冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 教师臂不用推理二进制，只记哈希
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740             # 只设环境变量与记录；教师臂不用 GPU
L=${L:-0.20}                                             # 预注册：L = 0.20
SMOKE_SEEDS=2,3
SMOKE_EPISODES=2
ACK_SEEDS=2,3,4,6,7
ACK_EPISODES=5
HOGS=${HOGS:-8}                                          # 满载臂的占核进程数（任务卡：8 个）
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
K1_ARGS="--task base --steps 80"                         # 柱阵 K=1
K1_EPT=${K1_EPT:-1500}                                   # 每集墙钟上限（秒）：同第二轮
# 起跑条件（公共规则 §4 01:40 起）
MIN_CORES=${MIN_CORES:-12}                               # cgroup 配额下限（核）
IDLE_MAX_CORES=${IDLE_MAX_CORES:-8}                      # 窗口内容器平均核数上限
IDLE_WINDOW_S=${IDLE_WINDOW_S:-30}                       # 测量窗口（秒）
IDLE_RECHECK_S=${IDLE_RECHECK_S:-30}                     # 不满足时的复查间隔（秒）；窗口 30 s 加间隔 30 s = 每分钟一次
IDLE_LOG_S=${IDLE_LOG_S:-600}                            # 等待期间每多少秒记一行
IDLE_LIMIT_S=${IDLE_LIMIT_S:-600}                        # 最长等 10 分钟，到点照样起跑
LOCK_LOG_S=${LOCK_LOG_S:-600}                            # 等锁期间每多少秒记一行（排队期间每 10 分钟看一次，不空转刷日志）
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # Gazebo 池的锁（exec 8>> 锁文件; flock 8；持有到脚本退出）
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}                     # 只记录
CPU_MAX_FILE=${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}     # 配额 微秒 周期
CPU_STAT_FILE=${CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}  # usage_usec、nr_throttled、throttled_usec
GZ_PATTERN=${GZ_PATTERN:-'(^|[ /])gz sim( |$)'}          # pgrep -f 的模式（匹配 gz sim 进程的命令行）
GZ_PROBE=${GZ_PROBE:-}                                   # 非空时用这条命令的第一行输出当 gz sim 进程数（演练用）
EVAL=${EVAL:-}                                           # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练流程时才覆盖）
SELFCHECK=${SELFCHECK:-1}                                # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（演练时才关）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python CPU_STAT_FILE
REL=experiments/2026-11-12-w1-13-newprotocol
SUMM=$CODE/$REL/summarize_ack.py
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg

log() { echo "$(date '+%F %T') $*"; }
HOG_PGID=""

hog_stop() {   # 只按 PGID 停；停之前核对该 PGID 的组长命令行是我们起的那条
  [[ -n "$HOG_PGID" ]] || return 0
  local args; args=$(ps -o args= -p "$HOG_PGID" 2>/dev/null)
  if [[ "$args" == bash*"while :; do :; done"* ]]; then
    kill -TERM -- "-$HOG_PGID" 2>/dev/null; sleep 1; kill -KILL -- "-$HOG_PGID" 2>/dev/null
    log "占核进程组 PGID=$HOG_PGID 已停（核对命令行：${args:0:60}…）；剩余进程 $(pgrep -g "$HOG_PGID" 2>/dev/null | wc -l) 个"
    [[ -f "$OUT/$HOG_ARM.hog" ]] && echo "stop=$(date '+%F %T') remaining=$(pgrep -g "$HOG_PGID" 2>/dev/null | wc -l)" >> "$OUT/$HOG_ARM.hog"
  else
    log "警告：PGID=$HOG_PGID 的命令行不是预期的占核进程（${args:-已不存在}），不杀"
  fi
  HOG_PGID=""
}

finish() {   # $1 = rc，$2 = 原因：停占核进程，刷新 summary.md（冻结代码已就位时），写 DONE，退出
  trap - EXIT TERM INT HUP
  hog_stop
  [[ -f "$SUMM" ]] && "$PY" "$SUMM" "$OUT" --latency "$L" --note "$(date '+%F %T') 生成，DONE rc=$1" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }

exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
{ echo "pid=$$"; echo "pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"; echo "lstart=$(ps -o lstart= -p $$ 2>/dev/null)"; echo "cmd=${BASH_SOURCE[0]} $*"; } > "$OUT/run.meta"
trap 'hog_stop' EXIT
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；评完的臂保留，同一条命令重起可续跑"' TERM INT HUP
log "开始 PGID=$$ L=$L 冒烟 seed $SMOKE_SEEDS，测量 seed $ACK_SEEDS，占核进程 $HOGS 个，机器锁=$MACHINE_LOCK，env_overrides=$ENV_OVERRIDES"

# ---------------------------------------------------------------- 配额检查（不满足立刻退出）
read -r quota period < "$CPU_MAX_FILE"
if [[ "$quota" == max ]]; then cores=max; quota_ok=1
else
  cores=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN{printf "%.2f", q/p}')
  quota_ok=$(awk -v c="$cores" -v m="$MIN_CORES" 'BEGIN{print (c>=m) ? 1 : 0}')
fi
[[ "$quota_ok" == 1 ]] || finish 13 "cgroup 配额 $cores 核 < $MIN_CORES 核（$CPU_MAX_FILE：$quota ${period:-?}），没有起跑"
log "cgroup 配额 $cores 核（要求 ≥ $MIN_CORES）"
gpu_list=$("$NVIDIA_SMI" -L 2>&1 | head -n 3 | tr '\n' ';')

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露；E-2/E-11：核对通过后才写 code_head.txt，续跑拒绝换提交）
if [[ -f "$OUT/code_head.txt" ]]; then
  prev_head=$(cat "$OUT/code_head.txt")
  if [[ -n "$CODE_HEAD" ]]; then
    want_head=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
    [[ "$want_head" == "$prev_head" ]] || finish 4 "续跑 CODE_HEAD ($want_head) 与已有 code_head.txt ($prev_head) 不一致，拒绝混入不同代码"
  fi
  CODE_HEAD=$prev_head
elif [[ -z "$CODE_HEAD" ]]; then
  CODE_HEAD=$(git -C "$SRC" rev-parse HEAD 2>/dev/null) || finish 4 "无法读取 $SRC 的 HEAD"
else
  CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
fi
if [[ "$SELFCHECK" == 1 ]]; then
  want=$(git -C "$SRC" show "$CODE_HEAD:$REL/run_smoke_ack.sh" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $REL/run_smoke_ack.sh 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
[[ -f "$SUMM" ]] || finish 4 "冻结副本里没有 $REL/summarize_ack.py（CODE_HEAD $CODE_HEAD 之前的提交？）"
[[ -f "$OUT/code_head.txt" ]] || echo "$CODE_HEAD" > "$OUT/code_head.txt"
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
export EVAL
{ echo "=== attempt $(date '+%F %T') pid=$$ ==="
  echo "code_head $CODE_HEAD"
  echo "env_overrides $ENV_OVERRIDES"
  echo "gpu_list $gpu_list"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/cf_sitl_exec.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$INFER" "$PLUGIN/libCfSitlSystem.so"; } >> "$OUT/inputs.txt" 2>&1

# ---------------------------------------------------------------- 机器锁，起跑条件（等到满足或到点才开始）
mkdir -p "$(dirname "$MACHINE_LOCK")"
exec 8>> "$MACHINE_LOCK"
t_lock=$SECONDS
if ! flock -n 8; then
  log "机器锁 $MACHINE_LOCK 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行）"
  until flock -w "$LOCK_LOG_S" 8; do log "仍在等机器锁（已等 $((SECONDS - t_lock)) s）"; done
fi
lock_wait=$((SECONDS - t_lock))
log "拿到机器锁，等了 $lock_wait s；持有到脚本退出"

gz_count() {
  local n
  if [[ -n "$GZ_PROBE" ]]; then n=$(bash -c "$GZ_PROBE" 2>/dev/null | head -n1); else n=$(pgrep -fc "$GZ_PATTERN" 2>/dev/null); fi
  echo "${n:-0}"
}
usage_usec() { awk '$1=="usage_usec"{print $2+0}' "$CPU_STAT_FILE"; }
t_idle=$SECONDS; last_idle_log=-999999; idle_cores=""; gz_n=""; forced=否
while true; do
  u0=$(usage_usec); w0=$(date +%s.%N); sleep "$IDLE_WINDOW_S"; u1=$(usage_usec); w1=$(date +%s.%N)
  idle_cores=$(awk -v a="$u0" -v b="$u1" -v t0="$w0" -v t1="$w1" 'BEGIN{printf "%.2f", (b-a)/1e6/(t1-t0)}')
  gz_n=$(gz_count)
  idle_ok=$(awk -v c="$idle_cores" -v m="$IDLE_MAX_CORES" -v g="$gz_n" 'BEGIN{print (g==0 && c<m) ? 1 : 0}')
  [[ "$idle_ok" == 1 ]] && break
  waited=$((SECONDS - t_idle))
  if (( waited >= IDLE_LIMIT_S )); then
    forced=是
    log "到点照样起跑：等了 $waited s 仍不满足（gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核，要求 0 个且 < $IDLE_MAX_CORES 核）"
    break
  fi
  if (( SECONDS - last_idle_log >= IDLE_LOG_S )); then
    log "起跑条件不满足：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（要求 0 个且 < $IDLE_MAX_CORES 核）；已等 $waited s，每 $((IDLE_WINDOW_S + IDLE_RECHECK_S)) s 复查"
    last_idle_log=$SECONDS
  fi
  sleep "$IDLE_RECHECK_S"
done
idle_wait=$((SECONDS - t_idle))
log "起跑：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（阈值 $IDLE_MAX_CORES）；强制起跑=$forced；等了 $idle_wait s"
cat >> "$OUT/start.txt" <<EOF
=== attempt $(date '+%F %T') pid=$$ ===
cpu_quota_cores=$cores
min_cores=$MIN_CORES
lock_wait_s=$lock_wait
gz_count=$gz_n
idle_window_s=$IDLE_WINDOW_S
idle_cores=$idle_cores
idle_max_cores=$IDLE_MAX_CORES
idle_wait_s=$idle_wait
forced_start=$forced
started_at=$(date '+%F %T')
code_head=$CODE_HEAD
L=$L
hogs=$HOGS
env_overrides=$ENV_OVERRIDES
gpu_list=$gpu_list
EOF

# ---------------------------------------------------------------- 臂
cg_snapshot() {   # 一行：epoch usage_usec nr_throttled throttled_usec（cgroup 整个容器）
  awk -v t="$(date +%s.%N)" '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} $1=="throttled_usec"{h=$2+0} END{printf "%s %d %d %d\n", t, u, n, h}' "$CPU_STAT_FILE"
}
cg_mark() {       # $1 = start|end，$2 = 臂名：追加到 <臂>.cg（start 时重写）
  if [[ $1 == start ]]; then echo "start $(cg_snapshot)" > "$OUT/$2.cg"; else echo "end $(cg_snapshot)" >> "$OUT/$2.cg"; fi
}
cg_brief() { awk 'END{printf "usage_usec=%s nr_throttled=%s", $2, $3}' <<< "$(cg_snapshot)"; }
arm_complete() {  # $1 = 臂名，$2 = 请求集数：eval.json 不是 partial 且 出结果 + 采集失败（含迟到集、停顿集）= 请求数
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$1/eval.json" "$2" 2>/dev/null
}

hog_start() {   # $1 = 臂名：另起 $HOGS 个死循环，同一个进程组（setsid），记 PGID、启动时刻、命令行
  HOG_ARM=$1
  setsid bash -c "for _ in \$(seq $HOGS); do ( while :; do :; done ) & done; wait" > /dev/null 2>&1 < /dev/null &
  local pid=$!
  sleep 1
  HOG_PGID=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [[ -z "$HOG_PGID" ]]; then log "占核进程没起来"; return 1; fi
  { echo "pgid=$HOG_PGID"; echo "start=$(date '+%F %T')"; echo "lstart_args=$(ps -o lstart=,args= -p "$HOG_PGID")"
    echo "processes_in_group=$(pgrep -g "$HOG_PGID" | wc -l)"; } > "$OUT/$1.hog"
  log "占核进程组 PGID=$HOG_PGID 已起，组内 $(pgrep -g "$HOG_PGID" | wc -l) 个进程"
}

run_arm() {   # $1 = 臂名，$2 = 集数，$3 = seed 列表，$4 = 有无占核进程（0|1）
  local name=$1 episodes=$2 seeds=$3 hog=${4:-0}
  if arm_complete "$name" "$episodes"; then log "$name 已评完，跳过"; return 0; fi
  local ts; ts=$(date +%H%M%S)
  if [[ -d "$OUT/$name" ]]; then   # 半成品挪开留证据，不删
    mv "$OUT/$name" "$OUT/$name.partial-$ts"
    for ext in log cg hog; do [[ -f "$OUT/$name.$ext" ]] && mv "$OUT/$name.$ext" "$OUT/$name.partial-$ts.$ext"; done
  fi
  [[ "$hog" == 1 ]] && hog_start "$name"
  cg_mark start "$name"
  log "$name 开始（$(cg_brief)）"
  # shellcheck disable=SC2086
  "$PY" "$EVAL" --policy teacher $K1_ARGS --split train --seeds "$seeds" --corridor-half-width $CORRIDOR --action-latency "$L" \
      --episodes "$episodes" --sim-timeout 120.0 --episode-timeout "$K1_EPT" \
      --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  cg_mark end "$name"
  [[ "$hog" == 1 ]] && hog_stop
  log "$name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)（$(cg_brief)）"
  return 0
}

# ---------------------------------------------------------------- 阶段 1：冒烟
run_arm smoke "$SMOKE_EPISODES" "$SMOKE_SEEDS" 0
smoke_text=$("$PY" "$SUMM" --smoke-check "$OUT/smoke" --latency "$L"); smoke_rc=$?
log "冒烟判定 rc=$smoke_rc $smoke_text"
if [[ $smoke_rc -ne 0 ]]; then
  finish 20 "冒烟没过，没有进入测量：${smoke_text#* }"
fi

# ---------------------------------------------------------------- 阶段 2：测量（空闲、满载）
run_arm idle "$ACK_EPISODES" "$ACK_SEEDS" 0
run_arm load "$ACK_EPISODES" "$ACK_SEEDS" 1

missing=""
arm_complete smoke "$SMOKE_EPISODES" || missing="$missing smoke"
arm_complete idle "$ACK_EPISODES" || missing="$missing idle"
arm_complete load "$ACK_EPISODES" || missing="$missing load"
[[ -z "$missing" ]] || finish 11 "冒烟已过，但有臂没跑完：$missing；读 summary.md，同一条命令重起可续跑"
mchk_text=$("$PY" "$SUMM" --measurement-check "$OUT" --latency "$L" --min-episodes "$ACK_EPISODES"); mchk_rc=$?
log "测量数据检查 rc=$mchk_rc $mchk_text"
if [[ $mchk_rc -ne 0 ]]; then
  finish 12 "测量臂有效轨迹不足：${mchk_text#* }"
fi
finish 0 "冒烟过，idle 与 load 两个测量臂跑完且数据达标；读 summary.md 的判定（两个条件的 p99 对 10 ms）"
