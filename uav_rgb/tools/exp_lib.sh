# shellcheck shell=bash
# 实验入口的公共脚本（任务卡 W1-15a，交叉审核 E）：被入口 source，不单独运行。样板与用法：experiments/_template/。
# 入口的流程：配置区（exp_cfg）→ exp_start → exp_arm … → exp_finish。本脚本只读入口配置区里的变量，自己不带路径与门槛的缺省值。
# 保证：先拿实例锁再写任何状态文件；run.pid 记 PID、PGID、启动时刻与命令行（stop_daemon.sh 靠它核对）；被信号终止时落 rc=130 的 DONE，没走到 exp_finish 就退出也补 DONE；
#   代码冻结到 $OUT/code（git archive），续跑沿用首次的提交并核对冻结副本；起跑记录 launches.log 只追加，覆盖过的配置项与输入文件的 sha256 逐个写进去；
#   按池拿锁（排队是唯一不设上限的等待），Gazebo 池拿锁后的起跑条件有上限，到点照样起并把当时的核数记下来；
#   臂：完成过的跳过，半成品目录改名留着再做，起止各记一次 cgroup 快照并打一行时序健康；有臂失败时 DONE 的 rc 不为 0。
# DONE 的 rc：0 全部跑完；2 配置或输入有问题；3 已有实例（不写 DONE）；4 提交、冻结副本或输入与首次起跑记录不一致；11 有臂失败；12 配置的 GPU 不在；13 cgroup 配额不够；130 被信号终止。
export LC_ALL=C
EXP_ENTRY=$(readlink -f "${BASH_SOURCE[1]}"); EXP_LIB=$(readlink -f "${BASH_SOURCE[0]}")   # 入口与本文件的真实路径：自检与记录用
EXP_OVERRIDES=()

exp_log() { echo "$(date '+%F %T') $*"; }

exp_cfg() {  # 保证：变量 $1 取环境里的同名值（环境里设了就不动它，并记进覆盖清单），没有就取默认值 $2；「被覆盖」= 调用时这个变量已经设过
  if [[ -n "${!1+x}" ]]; then EXP_OVERRIDES+=("$1=${!1}（缺省 $2）"); else printf -v "$1" %s "$2"; fi
}

exp_stat() {  # 保证：打印进程 $1 的 /proc/$1/stat 第 $2 列（5 = PGID，22 = 启动时刻：自宿主机开机起的节拍，容器重启不归零）；进程不存在返回 1
  local s a; { s=$(< "/proc/$1/stat"); } 2>/dev/null || return 1
  read -ra a <<< "${s##*) }"; echo "${a[$2-3]}"     # comm 里可能有空格和括号，从最后一个「) 」之后数，第 3 列在下标 0
}

exp_cmdline() {  # 保证：打印进程 $1 的命令行（参数间一个空格）；起跑时写进 run.pid，stop_daemon.sh 用同一个函数核对
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | sed 's/ *$//'
}

exp_rec() { echo "$*" >> "$OUT/launches.log"; }   # 保证：只往起跑记录末尾追加一行，从不改写已有内容

exp_first() {  # 保证：打印起跑记录里以「$1 」开头的第一行的其余部分（首次起跑时写下的值）；没有记录就什么也不打印
  awk -v p="$1 " 'index($0, p) == 1 { print substr($0, length(p) + 1); exit }' "$OUT/launches.log" 2>/dev/null
}

exp_code_sum() {  # 保证：打印冻结副本 $CODE 全部文件（不含 __pycache__）的汇总 sha256，副本被改过或缺文件就对不上
  (cd "$CODE" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
}

exp_cores() {  # 保证：睡 $1 秒，打印这段时间里容器（cgroup cpu.stat 的 usage_usec 增量）平均用了几个核，两位小数
  local u0 u1 t0 t1
  u0=$(awk '$1 == "usage_usec" { print $2 }' "$CPU_STAT_FILE"); t0=$(date +%s.%N); sleep "$1"
  u1=$(awk '$1 == "usage_usec" { print $2 }' "$CPU_STAT_FILE"); t1=$(date +%s.%N)
  awk -v a="$u0" -v b="$u1" -v x="$t0" -v y="$t1" 'BEGIN { printf "%.2f", (b - a) / 1e6 / (y - x) }'
}

exp_lock_instance() {  # 保证：同一个 $OUT 同一时刻只有一个实例；拿不到锁 rc=3 退出，不写任何文件（只往 stderr 说一句），盖不掉在跑的实例的状态
  mkdir -p "$OUT" && exec 9> "$OUT/.lock" || exit 2
  flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
}

exp_refuse() {  # 保证：$1=rc $2=原因：原因追加进 refused.txt 并打印；DONE 只在还没有时才写（不盖掉上一次的结果）；然后退出
  echo "$2" >&2; echo "$(date '+%F %T') $2" >> "$OUT/refused.txt"
  [[ -f "$OUT/DONE" ]] || echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  exit "$1"
}

exp_check_config() {  # 保证（只读，不过就 exp_refuse 2）：配置区的变量齐全，池名认识，数值是数，cgroup 文件读得了；库自己不带缺省值，缺了就拒绝
  local v
  for v in SRC CODE_HEAD INPUTS SELFCHECK POOL LOCK_DIR LOCK_LOG_S GPU MIN_CORES CPU_MAX_FILE CPU_STAT_FILE \
           IDLE_MAX_CORES IDLE_WINDOW_S IDLE_RECHECK_S IDLE_LIMIT_S GZ_PATTERN; do
    [[ -n "${!v+x}" ]] || exp_refuse 2 "入口配置区缺 $v"
  done
  case $POOL in gazebo | gs3d | train | none) ;; *) exp_refuse 2 "POOL=$POOL 不认识（gazebo | gs3d | train | none）" ;; esac
  for v in MIN_CORES IDLE_MAX_CORES IDLE_RECHECK_S; do [[ ${!v} =~ ^[0-9]+(\.[0-9]+)?$ ]] || exp_refuse 2 "$v=${!v} 不是非负数"; done
  for v in LOCK_LOG_S IDLE_WINDOW_S; do [[ ${!v} =~ ^[0-9]+(\.[0-9]+)?$ && ${!v} =~ [1-9] ]] || exp_refuse 2 "$v=${!v} 不是正数"; done
  [[ $IDLE_LIMIT_S =~ ^[0-9]+$ ]] || exp_refuse 2 "IDLE_LIMIT_S=$IDLE_LIMIT_S 不是非负整数（秒）"
  [[ -r $CPU_STAT_FILE && -r $CPU_MAX_FILE ]] || exp_refuse 2 "读不了 $CPU_STAT_FILE 或 $CPU_MAX_FILE"
}

exp_preflight() {  # 保证（只读，不过就 exp_refuse）：入口是进程组长、输入在且与首次一致、提交有效且与首次一致、入口与库已提交、冻结副本没被改过、GPU 在、配额够
  local f head src have want first_head first_sum gpus quota period
  exp_check_config
  [[ "$(exp_stat $$ 5)" == "$$" ]] || exp_refuse 2 "入口不是进程组长：请用 setsid nohup bash <入口> 起（停进程按 PGID）"
  for f in $INPUTS; do
    [[ -f $f ]] || exp_refuse 2 "缺输入文件 $f"
    have=$(sha256sum "$f" | cut -d' ' -f1); want=$(exp_first "input $f")
    [[ -z $want || $want == "$have" ]] || exp_refuse 4 "输入 $f 的 sha256 与首次起跑记录不一致（记录 $want，现在 $have）；输入变了就换新的输出目录"
  done
  first_head=$(exp_first code_head); first_sum=$(exp_first code_sha256); src=$(readlink -f "$SRC")
  head=$(git -C "$src" rev-parse "${CODE_HEAD:-${first_head:-HEAD}}^{commit}" 2>/dev/null) || exp_refuse 4 "$src 里没有提交 ${CODE_HEAD:-${first_head:-HEAD}}"
  [[ -z $first_head || $head == "$first_head" ]] || exp_refuse 4 "续跑沿用首次起跑的提交 $first_head，本次给的是 $head"
  if [[ $SELFCHECK == 1 ]]; then for f in "$EXP_ENTRY" "$EXP_LIB"; do
    [[ "$(git -C "$src" show "$head:${f#"$src"/}" 2>/dev/null | sha256sum)" == "$(sha256sum < "$f")" ]] \
      || exp_refuse 4 "$f 不在 $src 里，或与 $head 里的同名文件不一致：先提交再起跑（SELFCHECK=0 只给演练用，会进起跑记录）"
  done; fi
  [[ -z $first_sum || "$(exp_code_sum)" == "$first_sum" ]] || exp_refuse 4 "冻结副本 $CODE 与首次起跑记录不一致（被改过或不全）；不要改它，换新的输出目录重跑"
  EXP_GPU_LINE=
  if [[ -n $GPU ]]; then
    gpus=$(nvidia-smi -L 2>&1); EXP_GPU_LINE=$(grep -F "$GPU" <<< "$gpus" | head -n1)
    [[ -n $EXP_GPU_LINE ]] || exp_refuse 12 "配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpus" | tr '\n' ';')）"
  fi
  read -r quota period < "$CPU_MAX_FILE"
  EXP_CORES=max; [[ $quota == max ]] || EXP_CORES=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN { printf "%.2f", q / p }')
  [[ $EXP_CORES == max ]] || awk -v c="$EXP_CORES" -v m="$MIN_CORES" 'BEGIN { exit !(c >= m) }' \
    || exp_refuse 13 "cgroup 配额 $EXP_CORES 核 < $MIN_CORES 核（$CPU_MAX_FILE：$quota $period）"
  EXP_HEAD=$head; EXP_SRC=$src
}

exp_begin() {  # 保证：到这里才开始写状态：日志进 run.log、旧 DONE 删掉、run.pid 原子写好、终止信号落 rc=130 的 DONE、代码冻结、起跑记录追加一段
  local v f
  export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$OUT/tmp"; mkdir -p "$TMPDIR"     # 根分区只剩几 GB，临时文件与字节码不往上面放
  EXP_FAILED="$TMPDIR/failed.$$"; : > "$EXP_FAILED"                          # 本次起跑里失败的臂名，一行一个（并行的臂在子进程里也往这里写）
  exec >> "$OUT/run.log" 2>&1; rm -f "$OUT/DONE"
  printf 'pid=%s\npgid=%s\nstart=%s\ncmd=%s\n' "$$" "$(exp_stat $$ 5)" "$(exp_stat $$ 22)" "$(exp_cmdline $$)" > "$OUT/run.pid.tmp"
  mv -f "$OUT/run.pid.tmp" "$OUT/run.pid"
  trap 'exp_finish 130 "收到终止信号（kill 进程组），停在这里；已完成的臂保留，同一条命令重起可续跑"' TERM INT
  trap exp_guard EXIT
  if [[ -z "$(exp_first code_sha256)" ]]; then     # 首次，或首次起跑死在解包途中（没记 sha256）：半成品改名留着，重新冻结
    [[ ! -e $CODE ]] || mv -T "$CODE" "$CODE.partial-$(date +%m%d-%H%M%S-%3N)" || exp_finish 4 "冻结代码失败：旧的 $CODE 改名失败，不在它上面解包"
    mkdir -p "$CODE" && ( set -o pipefail; git -C "$EXP_SRC" archive "$EXP_HEAD" | tar -x -C "$CODE" ) || exp_finish 4 "冻结代码失败（git archive $EXP_HEAD）"
  fi
  exp_rec "== 起跑 $(date '+%F %T') pid=$$"
  exp_rec "code_head $EXP_HEAD"; exp_rec "code_sha256 $(exp_code_sum)"
  exp_rec "entry $EXP_ENTRY $(sha256sum < "$EXP_ENTRY" | cut -d' ' -f1)"; exp_rec "lib $EXP_LIB $(sha256sum < "$EXP_LIB" | cut -d' ' -f1)"
  for v in "${EXP_OVERRIDES[@]}"; do exp_rec "override $v"; done
  exp_rec "gpu ${EXP_GPU_LINE:-无}"; exp_rec "cpu_quota_cores $EXP_CORES min_cores=$MIN_CORES"
  for f in $INPUTS; do exp_rec "input $f $(sha256sum "$f" | cut -d' ' -f1)"; done
  exp_log "开始 PGID=$$ OUT=$OUT 提交 $EXP_HEAD 覆盖：${EXP_OVERRIDES[*]:-无}"
}

exp_lock_pool() {  # 保证：POOL 对应的池锁（gazebo→machine.lock、gs3d→gs3d.lock、train→train.lock，在 $LOCK_DIR 下）拿到才返回，持有到本进程退出；POOL=none 不碰锁
  local f=$LOCK_DIR/machine.lock t0=$SECONDS
  case $POOL in none) return 0 ;; gs3d) f=$LOCK_DIR/gs3d.lock ;; train) f=$LOCK_DIR/train.lock ;; esac
  mkdir -p "$LOCK_DIR" && exec 8>> "$f" || exp_finish 2 "打不开池锁 $f"
  if ! flock -n 8; then
    exp_log "池 $POOL：锁 $f 被占用，排队（每 ${LOCK_LOG_S}s 记一行）"
    until flock -w "$LOCK_LOG_S" 8; do exp_log "池 $POOL：仍在排队（已等 $((SECONDS - t0))s）"; done   # 唯一不设上限的等待：排队时自己还没占任何东西
  fi
  exp_rec "pool $POOL lock=$f lock_wait_s=$((SECONDS - t0))"; exp_log "池 $POOL：拿到 $f，等了 $((SECONDS - t0))s，持有到本进程退出"
}

exp_idle() {  # 保证：Gazebo 池拿到锁后，没有 gz sim 进程且 $IDLE_WINDOW_S 秒窗口内容器低于 $IDLE_MAX_CORES 核才往下走；不满足每 $IDLE_RECHECK_S 秒复查，
              # 最长等 $IDLE_LIMIT_S 秒（整秒计，最后一次复查多出至多一个窗口），到点照样往下走并把当时的核数记进起跑记录；等的时候占着池锁，所以必须有上限
  local t0=$SECONDS n c left late=0
  while :; do
    c=$(exp_cores "$IDLE_WINDOW_S"); n=$(pgrep -fc "$GZ_PATTERN" 2>/dev/null)
    [[ $n =~ ^[0-9]+$ ]] || exp_finish 2 "pgrep 没有给出 gz sim 进程数（给出：$n）"
    awk -v c="$c" -v m="$IDLE_MAX_CORES" -v g="$n" 'BEGIN { exit !(g == 0 && c < m) }' && break
    left=$((IDLE_LIMIT_S - (SECONDS - t0)))
    (( left > 0 )) || { late=1; exp_log "等了 $((SECONDS - t0))s 仍不满足（gz sim $n 个，$c 核），到点照样起跑"; break; }
    exp_log "起跑条件不满足：gz sim $n 个，${IDLE_WINDOW_S}s 窗口内 $c 核（要求 0 个且低于 $IDLE_MAX_CORES 核）；已等 $((SECONDS - t0))s，最多再等 ${left}s"
    sleep "$(awk -v a="$IDLE_RECHECK_S" -v b="$left" 'BEGIN { print (a < b) ? a : b }')"
  done
  exp_rec "start gz_count=$n idle_cores=$c idle_max_cores=$IDLE_MAX_CORES idle_window_s=$IDLE_WINDOW_S idle_wait_s=$((SECONDS - t0)) idle_timed_out=$late"
}

exp_cg() {  # 保证：打印一行 cgroup 快照「时刻 usage_usec nr_throttled throttled_usec」（整个容器，读 $CPU_STAT_FILE）
  awk -v t="$(date +%s.%N)" '$1 == "usage_usec" { u = $2 } $1 == "nr_throttled" { n = $2 } $1 == "throttled_usec" { h = $2 } END { print t, u + 0, n + 0, h + 0 }' "$CPU_STAT_FILE"
}

exp_health() {  # 保证：把臂 $1 的时序健康一行（墙钟、平均核数、nr_throttled 增量、被限流时长及占墙钟的比例）打印并追加进 health.txt；数据取自起止快照 $OUT/$1.cg
  awk -v a="$1" '$1 == "start" { t0 = $2; u0 = $3; n0 = $4; h0 = $5 } $1 == "end" { t1 = $2; u1 = $3; n1 = $4; h1 = $5 }
    END { w = t1 - t0; if (w <= 0) { print a, "没有起止快照"; exit }
          printf "%s 墙钟 %.0fs 平均 %.2f 核 nr_throttled +%d 被限流 %.1fs（占墙钟 %.0f%%）\n", a, w, (u1 - u0) / 1e6 / w, n1 - n0, (h1 - h0) / 1e6, 100 * (h1 - h0) / 1e6 / w }' "$OUT/$1.cg" | tee -a "$OUT/health.txt"
}

exp_arm() {  # 保证：$1=臂名，其余是命令，环境里有 ARM 与 ARM_DIR（=$OUT/$1）。有 $ARM_DIR/.ok 就跳过；没有 .ok 的旧目录改名 $1.partial-<时刻> 留着再做；
             # 起止各记一次 cgroup 快照并打一行时序健康；命令 rc=0 才写 .ok；失败只记下来不停队列（exp_finish 据此给 rc=11）；返回命令的 rc
  local name=$1 rc t; shift
  if [[ -f $OUT/$name/.ok ]]; then exp_log "臂 $name：已完成，跳过"; return 0; fi
  if [[ -e $OUT/$name ]]; then
    t=$name.partial-$(date +%m%d-%H%M%S-%3N)
    mv -T "$OUT/$name" "$OUT/$t" || exp_finish 2 "臂 $name：旧目录改名失败，不在旧目录上重做"
    exp_rec "partial $name -> $t"; exp_log "臂 $name：上次没跑完，旧目录改名 $t 留着，重做"
  fi
  echo "start $(exp_cg)" > "$OUT/$name.cg"; exp_log "臂 $name：开始"
  ARM=$name ARM_DIR=$OUT/$name "$@"; rc=$?
  echo "end $(exp_cg)" >> "$OUT/$name.cg"
  if (( rc == 0 )); then mkdir -p "$OUT/$name" && touch "$OUT/$name/.ok"; else echo "$name" >> "$EXP_FAILED"; fi
  exp_rec "arm $name rc=$rc"; exp_log "臂 $name：结束 rc=$rc"; exp_health "$name"
  return "$rc"
}

exp_finish() {  # 保证：$1=rc $2=说明：落下 DONE（rc、时间、说明）、起跑记录收一行、日志记一行，然后以 rc 退出；rc 给 0 而本次有臂失败时改成 11 并列出失败的臂
  local rc=$1 msg=$2 bad
  trap - TERM INT
  bad=$(sort -u "${EXP_FAILED:-/dev/null}" 2>/dev/null | tr '\n' ' ')
  if [[ $rc == 0 && -n $bad ]]; then rc=11; msg="有臂失败：${bad}其余臂保留，改好后同一条命令重起，只重跑失败的臂"; fi
  echo "rc=$rc $(date '+%F %T') $msg" > "$OUT/DONE"; exp_rec "end rc=$rc $(date '+%F %T')"; exp_log "全部结束 rc=$rc $msg"; exit "$rc"
}

exp_guard() {  # 保证：入口没走到 exp_finish 就退出（脚本出错、未定义变量、臂函数里的 exit）时补一个 DONE，rc 取退出码（0 改成 1），等 DONE 的脚本不会一直等
  local rc=$?
  [[ -f $OUT/DONE ]] || exp_finish "$((rc == 0 ? 1 : rc))" "入口没走到 exp_finish 就退出了（退出码 $rc），看 run.log 末尾"
}

exp_start() {  # 保证：先拿实例锁，再只读核对，然后才开始写状态（冻结、记录），最后按池排队拿锁、过 Gazebo 池的起跑条件；核对不过都在写日志与删旧 DONE 之前退出
  [[ -n "${OUT:-}" ]] || { echo "入口配置区缺 OUT" >&2; exit 2; }
  export OUT CODE="$OUT/code"
  exp_lock_instance; exp_preflight; exp_begin; exp_lock_pool
  if [[ $POOL == gazebo ]]; then exp_idle; fi
}
