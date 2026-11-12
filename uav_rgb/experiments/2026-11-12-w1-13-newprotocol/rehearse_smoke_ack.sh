#!/usr/bin/env bash
# run_smoke_ack.sh 的演练（不起 Gazebo、不碰真实锁与真实 cgroup）：用桩评测、假 cgroup 文件、临时锁、临时 git 仓库（冻结代码与哈希自检也走一遍）。
# 场景：A 全流程（等锁、强制起跑、占核进程起停）、B 冒烟没过（rc=20，不进测量）、C 满载臂里被终止（rc=130，占核进程组清干净）、
#       D 第二个实例被拒（rc=3）、E 续跑（评完的臂跳过，rc=0）。
# 用法：rehearse_smoke_ack.sh <演练目录>   退出码 0 = 全部场景符合预期；失败时打印哪一条不符。
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL=experiments/2026-11-12-w1-13-newprotocol
W=${1:?用法: rehearse_smoke_ack.sh <演练目录>}
mkdir -p "$W"; W=$(cd "$W" && pwd)
fail=0
ok() { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fail=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1  （条件：$2）"; fi; }

# 临时 git 仓库：桩评测放在 sim/tools/eval_closed_loop.py 的位置，实验目录用真文件
SRC=$W/src
if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$SRC/sim/tools" "$SRC/$REL"
  cp "$HERE/stub_eval_ack.py" "$SRC/sim/tools/eval_closed_loop.py"
  cp "$HERE/run_smoke_ack.sh" "$HERE/summarize_ack.py" "$SRC/$REL/"
  git -C "$SRC" init -q && git -C "$SRC" add -A && git -C "$SRC" -c user.name=rehearse -c user.email=r@x commit -q -m rehearse
fi
export PYTHONDONTWRITEBYTECODE=1

# 假 cgroup：cpu.max 12 核；cpu.stat 由后台计数器每秒加 FAKE_CORES 核的 usage_usec
echo "1200000 100000" > "$W/cpu.max"
ticker() {   # $1 = 每秒核数
  local u=0
  while sleep 1; do u=$((u + $1 * 1000000)); printf 'usage_usec %d\nnr_throttled 0\nthrottled_usec 0\n' "$u" > "$W/cpu.stat.tmp" && mv "$W/cpu.stat.tmp" "$W/cpu.stat"; done
}
printf 'usage_usec 0\nnr_throttled 0\nthrottled_usec 0\n' > "$W/cpu.stat"
setsid bash -c "$(declare -f ticker); W=$W; ticker 12" > /dev/null 2>&1 < /dev/null &
TICK_PGID=$(ps -o pgid= -p $! | tr -d ' ')
trap 'kill -- -$TICK_PGID 2>/dev/null' EXIT

run_daemon() {   # $1 = 场景名；其余 = 额外环境变量（K=V）。后台起（自成会话，PGID = run.pid 里的数）
  local name=$1; shift
  mkdir -p "$W/$name"; cp "$HERE/run_smoke_ack.sh" "$W/$name/"
  ( cd "$W/$name" || exit 1
    env OUT="$W/$name" SRC="$SRC" MACHINE_LOCK="$W/machine.lock" CPU_MAX_FILE="$W/cpu.max" CPU_STAT_FILE="$W/cpu.stat" \
      NVIDIA_SMI=true GZ_PROBE='echo 0' HOGS=2 IDLE_WINDOW_S=1 IDLE_RECHECK_S=1 IDLE_LIMIT_S=3 LOCK_LOG_S=1 "$@" \
      setsid bash run_smoke_ack.sh > "$W/$name/stdout.txt" 2>&1 < /dev/null & )
  sleep 0.5
}
wait_done() {   # $1 = 场景名，$2 = 最长秒数
  local i; for ((i = 0; i < $2 * 10; i++)); do [[ -f "$W/$1/DONE" ]] && return 0; sleep 0.1; done; return 1
}

echo "场景 A：全流程（锁被别人占 3 s，CPU 假读数 12 核 ≥ 8 → 3 s 后强制起跑，满载臂起停 2 个占核进程）"
exec 7>> "$W/machine.lock"; flock -n 7 || { echo "演练锁被占"; exit 2; }
run_daemon A STUB_SLEEP_S=1 STUB_DELAY_MS=5 STUB_DELAY_LOAD_MS=30
sleep 3; flock -u 7; exec 7>&-
wait_done A 60 || bad "A 没有在 60 s 内结束"
check "A DONE rc=0" "grep -q '^rc=0 ' $W/A/DONE"
check "A 等过锁（run.log 有「仍在等机器锁」）" "grep -q '仍在等机器锁' $W/A/run.log"
check "A 强制起跑记录" "grep -q 'forced_start=是' $W/A/start.txt && grep -q '到点照样起跑' $W/A/run.log"
check "A 三个臂都有 eval.json 与 .cg" "[[ -f $W/A/smoke/eval.json && -f $W/A/idle/eval.json && -f $W/A/load/eval.json && -f $W/A/idle.cg && -f $W/A/load.cg ]]"
check "A 满载臂有 .hog 记录且已停" "grep -q '^pgid=' $W/A/load.hog && grep -q '^stop=' $W/A/load.hog && grep -q 'remaining=0' $W/A/load.hog"
check "A 占核进程组没有残留" "! pgrep -g \$(sed -n 's/^pgid=//p' $W/A/load.hog) > /dev/null"
check "A 汇总判定：load p99 30 ms ≥ 10 ms，给建议判据交主会话定" "grep -q '本轮不改迟到判据' $W/A/summary.md && grep -q '由主会话定' $W/A/summary.md"
check "A 冻结代码与 code_head、env_overrides" "[[ -f $W/A/code/sim/tools/eval_closed_loop.py && -s $W/A/code_head.txt ]] && grep -q 'env_overrides=' $W/A/start.txt"

echo "场景 E：续跑（同一条命令重起，评完的臂跳过，start.txt 追加不覆盖）"
cp "$HERE/run_smoke_ack.sh" "$W/A/"
( cd "$W/A" && env OUT="$W/A" SRC="$SRC" MACHINE_LOCK="$W/machine.lock" CPU_MAX_FILE="$W/cpu.max" CPU_STAT_FILE="$W/cpu.stat" \
    NVIDIA_SMI=true GZ_PROBE='echo 0' HOGS=2 IDLE_WINDOW_S=1 IDLE_RECHECK_S=1 IDLE_LIMIT_S=3 LOCK_LOG_S=1 STUB_SLEEP_S=1 \
    setsid bash run_smoke_ack.sh > "$W/A/stdout2.txt" 2>&1 < /dev/null )
check "E 续跑 DONE rc=0 且三个臂都跳过" "grep -q '^rc=0 ' $W/A/DONE && [[ \$(grep -c '已评完，跳过' $W/A/run.log) -eq 3 ]]"
check "E start.txt 与 inputs.txt 追加了两段记录" "[[ \$(grep -c '^=== attempt ' $W/A/start.txt) -eq 2 && \$(grep -c '^=== attempt ' $W/A/inputs.txt) -eq 2 ]]"

echo "场景 B：冒烟没过（桩让冒烟臂缺 plugin_ack_sim_t 键）"
run_daemon B STUB_BREAK_SMOKE=1 STUB_SLEEP_S=0
wait_done B 60 || bad "B 没有在 60 s 内结束"
check "B DONE rc=20" "grep -q '^rc=20 ' $W/B/DONE"
check "B 没有进入测量" "[[ ! -d $W/B/idle && ! -d $W/B/load ]]"
check "B 释放了机器锁" "flock -n $W/machine.lock true"

echo "场景 D：第二个实例被拒（A 的 OUT 里有实例锁；先手工占住实例锁，再起一个）"
mkdir -p "$W/D"; cp "$HERE/run_smoke_ack.sh" "$W/D/"
exec 6> "$W/D/.lock"; flock -n 6
( cd "$W/D" && env OUT="$W/D" SRC="$SRC" MACHINE_LOCK="$W/machine.lock" CPU_MAX_FILE="$W/cpu.max" CPU_STAT_FILE="$W/cpu.stat" bash run_smoke_ack.sh > "$W/D/stdout.txt" 2>&1 < /dev/null ); rc=$?
check "D 退出码 3、不写 DONE" "[[ $rc -eq 3 && ! -f $W/D/DONE ]]"
exec 6>&-

echo "场景 C：满载臂里被终止（kill 进程组），占核进程组清干净"
run_daemon C STUB_SLEEP_S=0 STUB_SLEEP_LOAD_S=30
for ((i = 0; i < 300; i++)); do [[ -f "$W/C/load.hog" ]] && break; sleep 0.1; done
cpgid=$(cat "$W/C/run.pid"); hogpg=$(sed -n 's/^pgid=//p' "$W/C/load.hog" 2>/dev/null)
check "C 满载臂已起占核进程（组内 ≥ 2 个）" "[[ -n '$hogpg' ]] && [[ \$(pgrep -g '$hogpg' | wc -l) -ge 2 ]]"
kill -- "-$cpgid"
wait_done C 20 || bad "C 没有在 20 s 内写 DONE"
check "C DONE rc=130" "grep -q '^rc=130 ' $W/C/DONE"
check "C 占核进程组已清（没有残留进程）" "[[ -n '$hogpg' ]] && ! pgrep -g '$hogpg' > /dev/null"
check "C 机器锁已释放" "flock -n $W/machine.lock true"

[[ $fail -eq 0 ]] && echo "演练全部通过" || echo "演练有不符预期的条目"
exit $fail
