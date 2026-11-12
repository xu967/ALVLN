#!/usr/bin/env bash
# 按 run.pid 停一个实验入口（任务卡 W1-15a，交叉审核 E-4）。用法：tools/stop_daemon.sh <输出目录>
# 保证：run.pid 里记的 PID 现在仍是当初那个入口（启动时刻、命令行、进程组都对得上）才 kill -TERM -- -<PGID>；对不上就报告、不杀、rc 不为 0。
#   容器重启后 PID 会被别的进程用掉，只凭 PID 停进程会误杀（E-4）。不用 pkill -f，不自己升级成 SIGKILL。
# rc：0 已停止，或入口本来就不在；1 身份对不上（不杀）；2 用法错误、没有 run.pid、或是只有 PID 的旧格式（无法核对，不杀）；
#     3 发了 TERM，等 $STOP_WAIT_S 秒后进程组里仍有进程；4 入口已不在，但进程组里还有进程（无法核对身份，不杀，人工处理）。
set -u
WAIT_S=${STOP_WAIT_S:-30}            # 发 TERM 之后最多等几秒让进程组退干净

stat_field() {  # 打印 /proc/$1/stat 第 $2 列（3 = 状态，5 = PGID，22 = 启动时刻）；与 exp_lib.sh 的 exp_stat 同一写法
  local s a; { s=$(< "/proc/$1/stat"); } 2>/dev/null || return 1
  read -ra a <<< "${s##*) }"; echo "${a[$2-3]}"
}
group_alive() {  # 进程组 $1 里是否还有没退出的进程（僵尸算已退出：等父进程回收而已）
  ps -eo pgid=,stat= | awk -v g="$1" '$1 == g && $2 !~ /^Z/ { f = 1 } END { exit !f }'
}
group_list() { ps -eo pid=,pgid=,stat=,args= | awk -v g="$1" '$2 == g && $3 !~ /^Z/'; }

[[ $# == 1 && -f "$1/run.pid" ]] || { echo "用法：tools/stop_daemon.sh <输出目录>（目录里要有 run.pid）" >&2; exit 2; }
declare -A r
while IFS== read -r k v; do r[$k]=$v; done < "$1/run.pid"
if [[ ! ${r[pid]:-} =~ ^[0-9]+$ || ! ${r[pgid]:-} =~ ^[0-9]+$ || -z ${r[start]:-} || -z ${r[cmd]:-} ]]; then
  echo "$1/run.pid 是只有 PID 的旧格式或内容不全，无法核对身份，不杀；人工确认后自己停。内容：$(head -c 200 "$1/run.pid" | tr '\n' ' ')" >&2; exit 2
fi
pid=${r[pid]}; pgid=${r[pgid]}

if [[ ! -d /proc/$pid || "$(stat_field "$pid" 3)" == Z ]]; then
  if group_alive "$pgid"; then
    echo "入口 $pid 已不在，但进程组 $pgid 里还有进程（无法核对身份，不杀；人工确认后再停）：" >&2; group_list "$pgid" >&2; exit 4
  fi
  echo "run.pid 里的进程 $pid 已不存在（入口已结束或已被停掉），无需停"; exit 0
fi

live_start=$(stat_field "$pid" 22); live_pgid=$(stat_field "$pid" 5); live_cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | sed 's/ *$//')
why=
if [[ $live_start != "${r[start]}" ]]; then why="启动时刻不符（记录 ${r[start]}，现在 $live_start）"
elif [[ $live_cmd != "${r[cmd]}" ]]; then why="命令行不符（记录 ${r[cmd]}，现在 $live_cmd）"
elif [[ $live_pgid != "$pgid" || $pgid != "$pid" ]]; then why="进程组不符（记录 pgid $pgid，现在 $live_pgid，pid $pid）"
fi
[[ -z $why ]] || { echo "不停：run.pid 里的 PID $pid 现在不是当初起的那个入口：$why" >&2; exit 1; }

echo "核对通过：PID $pid，启动时刻 ${r[start]}，命令行 ${r[cmd]}；kill -TERM -- -$pgid"
kill -TERM -- "-$pgid" || { echo "kill 失败" >&2; exit 3; }
for ((i = 0; i < WAIT_S * 2; i++)); do group_alive "$pgid" || break; sleep 0.5; done
if group_alive "$pgid"; then
  echo "发了 TERM，等了 $WAIT_S s，进程组 $pgid 里仍有进程（不自动升级成 SIGKILL；要用请人工核对后 kill -KILL -- -$pgid）：" >&2; group_list "$pgid" >&2; exit 3
fi
echo "进程组 $pgid 已停止；$(cat "$1/DONE" 2>/dev/null || echo "DONE 不存在")"
