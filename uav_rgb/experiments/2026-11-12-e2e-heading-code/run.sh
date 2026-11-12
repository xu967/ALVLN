#!/usr/bin/env bash
# 卡 E2E-1 第 3 步，Gazebo 池一侧：4 个臂 × 3 个训练种子 × 20 个冻结 seed 的闭环评测（12 个评测臂、240 集），新协议（--action-latency L*），不带 --heading-guard。
# 训练池一侧是 prepare.sh；本脚本要等 $OUT/PREP_DONE 写着 rc=0 才起跑。冻结代码由 freeze.sh 做，本脚本从 $OUT/code/experiments/2026-11-12-e2e-heading-code/ 启动。
#
# ---------------------------------------------------------------- 预注册（卡 E2E-1 原文，主会话 11-12 定，起跑前写进本文件，起跑后不改；要改就换新的文件名与输出目录）
#   成功 = 走廊内成功，口径同第二轮（metric_version sim09.v2.coast1s.corridor3m：无碰撞、过成功线、|y| ≤ 3 m）。三个主比较，各 60 对逐集配对，精确 McNemar：
#   M1：A2 对 A3（同一份聚合数据，有无航向码）。预测：A2 的成功数比 A3 多 15 集以上，p < 0.01。证伪：多不到 6 集。
#     含义：预测成立，则 DAgger 的塌陷来自标签不可判定，补上航向码后学生自己走到的状态可以拿来学；证伪成立，则航向不是塌陷的主因，或者这个码带的信息不够。
#   M2：A1 对 A0（同一份示范数据，有无航向码）。预测：A1 的越界集数不超过 A0 的一半，成功数不低于 A0。证伪：A1 的越界集数比 A0 少不到 3 集。
#   M3：A2 对 A0。预测：A2 的成功数比 A0 多 6 集以上。证伪：A2 不高于 A0。
#   只报告、不设判决的一条：航向码臂里成功数高的那个，是否达到卡 W2-2 的判据（不带护栏成功不低于 48/60、越界不超过 3/60）。带护栏的参照是第二轮的 K=1 三个臂（53/60）。
#   另报：每臂的撞柱、越界、未到线；三个训练种子各自的数；动作占比；每步生效时长与迟到步；
#     离线反事实探针（val 上）：把码 2 与码 3 对调，转向帧的预测方向翻转的比例；把码恒置 0，macro-F1 的变化。这两个数说明网络有没有在用这个码。
#   不做：不带护栏之外的评测端规则；3 块课程（等这张卡的结果）；改网络结构；改 C++。
#
# 实现口径（卡里没写，执行者在起跑前定下，写在这里供主会话复核；判定代码在 summarize_e2e.py，阈值常量与上面逐条对应）：
#   · 配对键 =（训练种子，评测 seed）：同一个训练种子下两个臂在同一个评测 seed 的成败配对，3 个训练种子 × 20 个 seed = 60 对。
#   · 越界集数 = 失败分类为「越界」的集数（collision 优先；与第二轮、长课程汇总的 lc.block_outcomes 同口径）；eval.json 的 corridor_violation 标记另列，M2 另给一行按该标记计的判定，只作对照。
#   · 预测与证伪之间的取值（M1 差 6–14；M2 两条都不成立；M3 差 1–5）写「介于预测与证伪之间，不判」；M2 的预测条款与证伪条款同时成立（A0 越界集数 ≤ 5 时会出现）写「冲突」，不判。
#   · 含迟到步的集（基础设施失败）按 run_r2.sh 条款 R6 重跑一次：新输出子目录 <臂>-retry1，只含迟到的 seed，汇总以重跑结果为准；重跑仍迟到按没成功算，首次记录另列。电池掉压造成的采集中止是测量结果，不重跑。
#   · 评测顺序：训练种子优先，种子 0 的四个臂（A0、A1、A2、A3），再种子 1，再种子 2；两条车道按这个顺序取臂。中途被打断时四个臂的进度相同。
#   · 起跑条件按公共规则 §4「2026-11-12 01:40 起」（主仓库提交 89b7804）：① nvidia-smi -L 里有配置的 GPU；② cgroup 配额 ≥ 12 核；③ 拿到机器锁（阻塞等待，每 10 分钟记一行）；
#     ④ 拿到锁后核对没有 gz sim 进程，再看 30 s 窗口内容器 CPU（cpu.stat usage_usec 增量）：低于 8 核就起跑；不低于 8 核就每分钟复查，最多等 600 s，到点照样起跑，当时的核数与 gz 进程数记进 start.txt（idle_forced=1）。
#     改动说明：本入口 01:05 起草时抄的是第二轮 R7 的「< 1.5 核、最长 3 小时」，主会话 01:40 指出分池并行后这个条件不会满足、等待期间还占着锁，要求改成上述条件；门槛与最长等待在配置区（IDLE_MAX_CORES、IDLE_LIMIT_S）。
#     时序健康不靠起跑时的空闲保证：每个臂开始与结束各记一次 usage_usec、nr_throttled、throttled_usec（<臂>.cg），汇总每臂给平均核数、被限流时长占墙钟的比例、迟到步数，限流次数增量大于 0 的臂标 ⚠，不自动作废；迟到的集按 R6 重跑。
#   · L = $L_STAR_FILE 的内容（第二轮 B 组过门的 0.20）；推理二进制 main-97ae04b/uavrgb_infer（与 11-10、11-11 的评测同一个；src/infer.cpp 核过支持条件字节）。
#
# 运行：先 freeze.sh（把已提交的 HEAD 冻结到 $OUT/code），prepare.sh 写出 PREP_DONE rc=0，再
#   cd $OUT/code/experiments/2026-11-12-e2e-heading-code && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#   续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删）。
# 进度：tail -f $OUT/run.log；汇总随时可跑：<解释器> $OUT/code/experiments/2026-11-12-e2e-heading-code/summarize_e2e.py $OUT（缺的臂标「未完成」）。
#   结束落 DONE（rc、时间、原因）与 summary.md、verdicts.json、episodes.csv。停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# DONE 的 rc：0 全部跑完；3 已有实例（不写 DONE）；4 冻结代码与提交不一致；5 L_STAR 读不到或格式不对；6 PREP_DONE 缺失或不是 rc=0；7 输入缺失（权重、推理二进制、插件）；
#   11 有评测臂没评完；12 配置的 GPU 不在；13 cgroup 配额不足；130 被信号终止。（14 曾用于等空闲超时，01:40 起不再使用：到点照样起跑。）
# 时间估计（按第二轮 K=1 学生臂 20 集约 20 分钟折算，仅为估计）：12 个臂 / 2 条车道 ≈ 2–2.5 小时，加重跑，另加等锁与等空闲的时间。
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$HERE/lib.sh"

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
EPISODES=20                                              # 每个评测臂的请求集数 = 冻结 seed 个数
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
K1_ARGS="--task base --steps 80"                         # 柱阵 K=1
EPT=1500                                                 # 每集墙钟上限（秒）：同第二轮 K=1 学生臂
LANES=${LANES:-2}                                        # 2 条车道
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}        # 评测入口；只有演练时才覆盖
SUMM=$ENTRY/summarize_e2e.py
MIN_CORES=${MIN_CORES:-12}
IDLE_MAX_CORES=${IDLE_MAX_CORES:-8}                      # 公共规则 §4（01:40 起）：容器低于 8 核就起跑（12 核配额减 2 条车道的 3.5 核，留 0.5 核）
IDLE_WINDOW_S=${IDLE_WINDOW_S:-30}
IDLE_RECHECK_S=${IDLE_RECHECK_S:-60}
IDLE_LOG_S=${IDLE_LOG_S:-600}
IDLE_LIMIT_S=${IDLE_LIMIT_S:-600}                        # 最多等 10 分钟，到点照样起跑
GZ_PATTERN=${GZ_PATTERN:-'(^|[ /])gz sim( |$)'}          # ④ 探针：pgrep -f 的模式
GZ_PROBE=${GZ_PROBE:-}                                   # 非空时用这条命令的第一行输出当 gz sim 进程数（演练用）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT CODE PY EVAL SUMM CPU_STAT_FILE
mkdir -p /tmp/xdg && chmod 700 /tmp/xdg

finish() {   # $1 = rc，$2 = 原因：刷新 summary.md，写 DONE，退出
  trap - TERM INT
  [[ -f "$SUMM" ]] && "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 生成，DONE rc=$1" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

exec 9> "$OUT/.run.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.run.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；评完的臂保留，同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ lanes=$LANES gpu=$GPU 机器锁=$MACHINE_LOCK CODE_HEAD=$CODE_HEAD"

# ---------------------------------------------------------------- 先决条件：冻结代码、训练池一侧、L*、输入文件
if [[ "$SELFCHECK" == 1 ]]; then
  verify_frozen || finish 4 "冻结副本与 CODE_HEAD $CODE_HEAD 不一致：$FROZEN_BAD（先运行 freeze.sh，再从 \$OUT/code 里启动）"
fi
[[ -f "$OUT/PREP_DONE" ]] && grep -q '^rc=0 ' "$OUT/PREP_DONE" || finish 6 "训练池一侧没有写出 rc=0 的 $OUT/PREP_DONE（$(cat "$OUT/PREP_DONE" 2>/dev/null || echo 不存在)）；先让 prepare.sh 跑完"
L=$(tr -d '[:space:]' < "$L_STAR_FILE" 2>/dev/null)
[[ "$L" =~ ^[0-9]+(\.[0-9]+)?$ ]] || finish 5 "$L_STAR_FILE 读不到或格式不对（内容 '$L'）"
for a in $ARMS; do
  for s in $SEEDS_TRAIN; do
    c=$(arm_ckpt "$a" "$s")
    [[ -f "$c" ]] || finish 7 "交付权重缺失：$c"
    # 航向码臂的评测端一致性闸门靠 <权重>.json 侧车找训练 run 与数据集（average_ckpt 写）；A0 是 W1-5 的现成权重，没有侧车，图像臂的闸门只看 prev_dim = 0
    if arm_is_code "$a"; then [[ -f "$c.json" ]] || finish 7 "航向码臂的权重缺 .json 侧车：$c.json"; fi
  done
done
for f in "$INFER" "$PLUGIN/libCfSitlSystem.so" "$EVAL"; do
  [[ -f "$f" ]] || finish 7 "输入缺失：$f"
done

# ---------------------------------------------------------------- 起跑条件 ①②（不满足立刻退出）
gpu_list=$("$NVIDIA_SMI" -L 2>&1)
if ! grep -qF "$GPU" <<< "$gpu_list"; then
  finish 12 "条件 ① 不满足：配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpu_list" | tr '\n' ';')），没有起跑"
fi
log "条件 ① 满足：$(grep -F "$GPU" <<< "$gpu_list" | head -n1)"
read -r quota period < "${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}"
if [[ "$quota" == max ]]; then
  cores=max; quota_ok=1
else
  cores=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN{printf "%.2f", q/p}')
  quota_ok=$(awk -v c="$cores" -v m="$MIN_CORES" 'BEGIN{print (c>=m) ? 1 : 0}')
fi
if [[ "$quota_ok" != 1 ]]; then
  finish 13 "条件 ② 不满足：cgroup 配额 $cores 核 < $MIN_CORES 核，没有起跑"
fi
log "条件 ② 满足：cgroup 配额 $cores 核（要求 ≥ $MIN_CORES）"

# ---------------------------------------------------------------- 起跑记录（输入、配置）
{ echo "code_head $CODE_HEAD"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/student.py" "$CODE/sim/tools/nav_cond.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  for a in $ARMS; do for s in $SEEDS_TRAIN; do sha256sum "$(arm_ckpt "$a" "$s")"; done; done
  sha256sum "$INFER" "$PLUGIN/libCfSitlSystem.so" "$TRAIN" "$OUT/data/demo-hc/dataset.uavr" "$OUT/data/dagger-hc/dataset.uavr" "$DAGGER_DS" "$DEMO_DS"; } > "$OUT/inputs.txt" 2>&1
[[ -f "$OUT/config.env" ]] || cat > "$OUT/config.env" <<EOF
# run.sh 起跑配置
L_STAR=$L
CODE_HEAD=$CODE_HEAD
GPU=$GPU
LANES=$LANES
SEEDS_EVAL=$SEEDS_EVAL
EPISODES=$EPISODES
EOF

# ---------------------------------------------------------------- 起跑条件 ③ 机器锁，④ 空闲（等到满足才开始）
acquire_lock 8 "$MACHINE_LOCK" "条件 ③ 机器锁"
lock_wait=$LOCK_WAITED
gz_count() {
  local n
  if [[ -n "$GZ_PROBE" ]]; then n=$(bash -c "$GZ_PROBE" 2>/dev/null | head -n1); else n=$(pgrep -fc "$GZ_PATTERN" 2>/dev/null); fi
  echo "${n:-0}"
}
usage_usec() { awk '$1=="usage_usec"{print $2+0}' "$CPU_STAT_FILE"; }
t_idle=$SECONDS; last_idle_log=-999999; idle_cores=""; gz_n=""; idle_forced=0
while true; do
  u0=$(usage_usec); w0=$(date +%s.%N); sleep "$IDLE_WINDOW_S"; u1=$(usage_usec); w1=$(date +%s.%N)
  idle_cores=$(awk -v a="$u0" -v b="$u1" -v t0="$w0" -v t1="$w1" 'BEGIN{printf "%.2f", (b-a)/1e6/(t1-t0)}')
  gz_n=$(gz_count)
  idle_ok=$(awk -v c="$idle_cores" -v m="$IDLE_MAX_CORES" -v g="$gz_n" 'BEGIN{print (g==0 && c<m) ? 1 : 0}')
  [[ "$idle_ok" == 1 ]] && break
  waited=$((SECONDS - t_idle))
  if (( waited + IDLE_RECHECK_S + IDLE_WINDOW_S > IDLE_LIMIT_S )); then idle_forced=1; break; fi   # 下一轮复查会超过最长等待：现在起跑
  if (( SECONDS - last_idle_log >= IDLE_LOG_S )); then
    log "条件 ④ 不满足：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（要求 0 个且 < $IDLE_MAX_CORES 核）；已等 $waited s，每 $IDLE_RECHECK_S s 复查"
    last_idle_log=$SECONDS
  fi
  sleep "$IDLE_RECHECK_S"
done
idle_wait=$((SECONDS - t_idle))
if [[ $idle_forced == 1 ]]; then
  log "条件 ④ 等了 $idle_wait s 仍不满足（gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核，要求 0 个且 < $IDLE_MAX_CORES 核），按公共规则到点起跑，核数记进 start.txt"
else
  log "条件 ④ 满足：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（< $IDLE_MAX_CORES）；等了 $idle_wait s"
fi
cat > "$OUT/start.txt" <<EOF
gpu_ok=是
cpu_quota_cores=$cores
min_cores=$MIN_CORES
lock_wait_s=$lock_wait
gz_count=$gz_n
idle_window_s=$IDLE_WINDOW_S
idle_cores=$idle_cores
idle_max_cores=$IDLE_MAX_CORES
idle_limit_s=$IDLE_LIMIT_S
idle_forced=$idle_forced
idle_wait_s=$idle_wait
started_at=$(date '+%F %T')
gpu=$GPU
lanes=$LANES
L_STAR=$L
EOF

# ---------------------------------------------------------------- 评测臂
arm_complete() {  # $1 = 臂名，$2 = 请求集数：eval.json 不是 partial 且 出结果 + 采集失败（含迟到集）= 请求数
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$1/eval.json" "$2" 2>/dev/null
}
export -f log cg_snapshot cg_mark cg_brief arm_complete

# 一个评测臂：名字|评测参数串|请求集数|单集墙钟上限|种类（acc|none）。单臂失败不停整个队列，汇总里按 failed 标出来。
run_job() {
  local name=$1 args=$2 episodes=$3 ept=$4 kind=${5:-none}
  [[ -n "$name" && -n "$args" ]] || return 0
  if arm_complete "$name" "$episodes"; then
    log "$name 已评完，跳过"
  else
    local ts; ts=$(date +%H%M%S)
    if [[ -d "$OUT/$name" ]]; then   # 半成品挪开留证据，不删
      mv "$OUT/$name" "$OUT/$name.partial-$ts"
      for ext in log cg; do [[ -f "$OUT/$name.$ext" ]] && mv "$OUT/$name.$ext" "$OUT/$name.partial-$ts.$ext"; done
    fi
    cg_mark start "$name"
    log "$name 开始（$(cg_brief)）"
    # shellcheck disable=SC2086
    "$PY" "$EVAL" $args --episodes "$episodes" --sim-timeout 120.0 --episode-timeout "$ept" \
        --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --out "$OUT/$name" > "$OUT/$name.log" 2>&1
    local rc=$?
    cg_mark end "$name"
    log "$name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)（$(cg_brief)）"
  fi
  [[ $kind == acc ]] && retry_job "$name" "$args" "$ept"
  return 0
}
retry_job() {   # 含迟到步的集重跑一次（条款 R6）：新输出子目录 <臂>-retry1，只含迟到的 seed，原目录保留
  local name=$1 args=$2 ept=$3 late n rargs
  late=$("$PY" "$SUMM" --late-seeds "$OUT/$name" 2>/dev/null)
  [[ -n "$late" ]] || return 0
  n=$(tr ',' '\n' <<< "$late" | grep -c .)
  rargs=$(sed -E "s/--seeds [^ ]+/--seeds $late/" <<< "$args")
  log "$name 含迟到步的集 [$late]，重跑一次 → $name-retry1"
  run_job "$name-retry1" "$rargs" "$n" "$ept" none
}
export -f run_job retry_job

run_queue() {   # 标准输入：每行一个臂 名字|评测参数串|请求集数|单集墙钟上限|种类；顺序即优先级
  xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n a e t k <<< "{}"; run_job "$n" "$a" "$e" "$t" "$k"'
}

acc_job() {   # $1 = 臂，$2 = 训练种子：学生 + 交付权重，K=1，不带 --heading-guard；A1、A2 加 --student-cond heading-code
  local cond=""
  arm_is_code "$1" && cond=" --student-cond heading-code"
  echo "$(arm_name "$1")-s$2|--policy student --ckpt $(arm_ckpt "$1" "$2") --infer-bin $INFER$cond $K1_ARGS --split test --seeds $SEEDS_EVAL --corridor-half-width $CORRIDOR --action-latency $L|$EPISODES|$EPT|acc"
}
refresh_summary() { "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 进行中：$1" > "$OUT/summarize.log" 2>&1; }

log "评测：4 个臂 × 训练种子 $SEEDS_TRAIN × $EPISODES 个冻结 seed，L=$L；顺序 = 训练种子优先（种子 0 的 a0 a1 a2 a3，再种子 1，再种子 2）"
{ for s in $SEEDS_TRAIN; do for a in $ARMS; do acc_job "$a" "$s"; done; done; } | run_queue
refresh_summary "评测队列结束"

missing=""
for a in $ARMS; do for s in $SEEDS_TRAIN; do arm_complete "$(arm_name "$a")-s$s" "$EPISODES" || missing="$missing $(arm_name "$a")-s$s"; done; done
[[ -z "$missing" ]] || finish 11 "有评测臂没评完：$missing；读 summary.md，同一条命令重起可续跑"
finish 0 "12 个评测臂全部跑完（L=$L）；读 summary.md 与 verdicts.json，判定与报告由执行者写、由主会话复核"
