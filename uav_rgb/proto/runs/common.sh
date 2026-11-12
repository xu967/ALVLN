#!/usr/bin/env bash
# proto/runs 下各守护进程入口共用的函数；入口脚本先设好 OUT（输出目录）再 source 本文件。
# 约定（公共任务书 §4）：守护进程阻塞等所属池的锁（exec 8>> 锁文件; flock 8，持有到退出），
# 等锁期间每 LOCK_LOG_S 秒记一行，开始与结束各记一次 cgroup 的 usage_usec 与 nr_throttled。
# 停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。结束落 $OUT/DONE（rc、时间、原因）。
CPU_STAT_FILE=${CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}
LOCK_LOG_S=${LOCK_LOG_S:-600}      # 等锁期间每 10 分钟记一行

log() { echo "$(date '+%F %T') $*"; }

cg_snapshot() {   # 一行：epoch usage_usec nr_throttled throttled_usec（整个容器的 cgroup，不是本进程）
  awk -v t="$(date +%s.%N)" '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} $1=="throttled_usec"{h=$2+0} END{printf "%s %d %d %d\n", t, u, n, h}' "$CPU_STAT_FILE"
}
cg_mark() {       # $1 = start|end，$2 = 名字：追加到 $OUT/<名字>.cg
  if [[ $1 == start ]]; then echo "start $(cg_snapshot)" > "$OUT/$2.cg"; else echo "end $(cg_snapshot)" >> "$OUT/$2.cg"; fi
}
cg_brief() { awk 'END{printf "usage_usec=%s nr_throttled=%s", $2, $3}' <<< "$(cg_snapshot)"; }

finish() {        # $1 = rc，$2 = 原因
  trap - TERM INT
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "结束 rc=$1 $2"
  exit "$1"
}

daemon_begin() {  # 单实例 + 日志 + pid；$1 = 一句话说明
  mkdir -p "$OUT"
  exec 9> "$OUT/.lock"
  flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
  exec >> "$OUT/run.log" 2>&1
  rm -f "$OUT/DONE"
  echo $$ > "$OUT/run.pid"
  trap 'finish 130 "收到终止信号（kill 进程组），停在这里"' TERM INT
  log "开始 PGID=$$ $1"
}

take_lock() {     # $1 = 锁文件，$2 = 池名；阻塞等待，持有到脚本退出（fd 8）
  mkdir -p "$(dirname "$1")"
  exec 8>> "$1"
  local t0=$SECONDS
  if ! flock -n 8; then
    log "$2：锁 $1 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行）"
    until flock -w "$LOCK_LOG_S" 8; do log "$2：仍在等锁（已等 $((SECONDS - t0)) s）"; done
  fi
  LOCK_WAIT_S=$((SECONDS - t0))
  log "$2：拿到锁，等了 $LOCK_WAIT_S s；持有到脚本退出"
}
