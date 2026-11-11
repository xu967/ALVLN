#!/usr/bin/env bash
# 任务卡 W1-7 验收 2：教师 v7（B2 过柱阵后目标取线 + B3 A* 无路降余量重试）闭环对拍。教师在 20 个冻结 seed 上是否仍 20/20，
# 出口区（x > 最远柱 x）的 YAW 步占比是否从约 17.7% 降到 5% 以下。零训练，不用推理二进制。
#
# 两个臂（同 seed、同参数、同一个守护进程里先后跑，所以同时段）：
#   v7  主臂：冻结代码 = 分支 p12-w1-7 起跑时的 HEAD（TEACHER_VERSION uavrgb.teacher.v7）。
#   v5  对照臂：冻结代码 = main 在开工时的 HEAD（CTRL_REF，默认 a714b54，TEACHER_VERSION uavrgb.teacher.v5）。不设门，只给同口径的出口区 YAW 占比与逐 seed 配对。
#       任务卡写的「约 17.7%」来自 0417 离线数据的另一种口径（视锥内无柱的训练帧），这里在同一批冻结 seed 上用同一口径重测旧教师。
# 每臂的 20 个冻结 seed 拆成两半：-a 取前 10 个、-b 取后 10 个，两条车道（LANES=2）并行；v7 两半先跑，v5 两半后跑。
#
# 参数：eval_closed_loop.py --policy teacher --task base --split test --episodes 10 --steps 80 --sim-timeout 120.0 --episode-timeout 1500.0
#       --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --corridor-half-width 3.0
#   与 experiments/2026-11-10-heading-guard/run.sh 同一套，去掉学生专用的 --ckpt --infer-bin --heading-guard，不加 --action-latency（任务卡要求）。
#
# ---------------------------------------------------------------- 预注册（2026-11-12，起跑前写下，起跑后不改；要改就换新文件名与输出目录）
#   P1（任务卡验收 2 的前半）：v7 臂 20/20 成功（撞柱 0、越界 0）。
#     证伪：任一集不成功。判不通过，逐集给原因（撞柱 / 越界 / 超时）并与 v5 对照臂同 seed 的结果并排；不调参数、不重跑。
#     要看的风险：B3 把 A* 余量降到 0.20 m（机身半宽 0.102 m，余量下的空隙只剩约 0.10 m），可能让教师擦柱。
#   P2（任务卡验收 2 的后半）：v7 臂出口区 YAW 步占比 < 5%。
#     起跑前已知（草稿 w1-7/evidence/，全文见报告；都不是 Gazebo 闭环）：
#       ① 离线，0417 旧轨迹（v5 标签 12000 步，300 集，所有 split）里「最远柱后表面之后、成功线之前」一档的 YAW 占比 17.78%（208/1170），
#          与任务卡的 17.7%（data-algo 评审：train、视锥内无柱，175/987）是相近的两个口径，未逐帧对齐。
#       ② 同状态重标（旧轨迹的状态不动，只换规则）预测不了闭环：该档里机头偏离 +x ≤ 10° 的 515 步，YAW 占比 39.22% → 0.00%；
#          偏离 > 25° 的 429 步，0.47% → 85.08%（v7 的目标在正前方，要把机头转回来）；该档合计 17.78% → 34.79%。旧轨迹的机头偏差来自采集时的扰动，
#          不代表 v7 闭环里出口区起点的机头偏差。
#       ③ 运动学代理（experiments/2026-11-12-teacher-v7-parity/kinematic_preview.py：教师规则 + 运动基元 + 噪声，20 个冻结 seed × 10 次重复，不是 Gazebo 动力学）：
#          v5 出口区 YAW 17.77%（906 步，161 次），v7 3.65%（821 步，30 次；Wilson 95% [2.57%, 5.17%]），两臂都 200/200 成功。
#          按「每个 seed 随机抽 1 次、池化 20 集」模拟单次 20-seed 评测，v7 的出口区占比 < 5% 的概率约 79.7%，v5 为 0。
#     预测：点估计约 3–4%，< 5% 的概率约 0.8（来自代理，cf_sitl 的动力学与噪声与代理不同，未核实）。单次评测的出口区约 80 步，YAW 步 ≤ 4 步才过线，对个别 seed 的转向敏感，
#       所以没过线不能直接推出 B2 无效，要看出口区 YAW 步的状态。
#     证伪：点估计 ≥ 5%。判不通过，如实写，并分析出口区 YAW 步的状态（机头偏离、离最远柱后表面的距离）；不改 B2。
#   P3（参照，不设门）：v5 对照臂出口区 YAW 占比高于 v7 臂。数字只报告，不与 17.7% 比大小（口径不同）。
#   口径：出口区 = 步的 pose_before.x > 最远柱 x（= per_episode.goal_x − goal_margin）；YAW = action_expert ∈ {YAW_LEFT, YAW_RIGHT}；占比按步池化。细节见 summarize.py 文件头。
#   对照臂若有不成功的集（旧教师在这批 seed 上不一定 20/20，未核实），v7 臂的门槛仍按任务卡判，对照臂的失败原因单列。
#
# 起跑条件：① nvidia-smi -L 里有配置的 GPU；② cgroup 配额 ≥ 12 核；③ 阻塞式拿机器锁 $MACHINE_LOCK（每 10 分钟记一行，持有到脚本退出）。
#   拿到锁后记录 gz sim 进程数与 30 s 窗口里容器的平均核数（只记录，不设门；本实验不带 --action-latency，对墙钟不敏感）。
#   ①② 不满足：立刻退出并写 DONE。③ 不设等待上限（只是一个阻塞的 flock）。
# 运行：mkdir -p $OUT && cp run.sh $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#   起跑前必须是已提交的版本：脚本会拿自己与 CODE_HEAD 里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive：v7 解到 $OUT/code，v5 解到 $OUT/code-v5，并逐文件核对 sim/ 的 git blob 哈希。
# 进度：tail -f $OUT/run.log；汇总随时可跑：<解释器> $OUT/code/experiments/2026-11-12-teacher-v7-parity/summarize.py $OUT。结束落 DONE 与 summary.md。
# 停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删），沿用起跑时的 CODE_HEAD。.lock 互斥，重复启动的那个立即退出（rc=3，不写 DONE）。
# DONE 的 rc：0 全部跑完；4 冻结代码或脚本版本不一致；11 有臂没跑完；12 配置的 GPU 不在；13 cgroup 配额不足；130 被信号终止。
# 时间估计（按 11-10 学生臂 1 条车道约 56 s/集折算，教师一集的步数相近，仅为估计）：每对两半约 10–12 分钟，两对约 25 分钟，另加等锁。
set -u
export LC_ALL=C

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器，对照臂的代码从它的 git 里 archive（只读）
D=/root/rivermind-data/uav_rgb
NAME=2026-11-12-teacher-v7-parity
OUT=${OUT:-$D/exp-$NAME}
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-w1-7}   # 代码源（分支 p12-w1-7）：只用来 git archive 冻结代码
CODE_HEAD=${CODE_HEAD:-}                                 # v7 臂冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
CTRL_REF=${CTRL_REF:-a714b54}                            # v5 对照臂的冻结代码：main 在开工时的 HEAD
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740             # 11-11 容器重启后 uav_rgb 用的这张卡（RTX 2080 Ti）；教师不用推理，仍按公共任务书钉住
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
STEPS=80                                                 # 同 heading-guard；到成功线即停
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
EPT=1500                                                 # 每集墙钟上限（秒）：同 heading-guard 实验
LANES=${LANES:-2}                                        # 任务卡：2 条车道
MIN_CORES=${MIN_CORES:-12}                               # ② cgroup 配额下限（核）
IDLE_WINDOW_S=${IDLE_WINDOW_S:-30}                       # 拿到锁后记录容器核数的窗口（秒）
LOCK_LOG_S=${LOCK_LOG_S:-600}                            # ③ 等锁期间每多少秒记一行
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # 重实验的整机锁（主会话定的惯用法：exec 8>> 锁文件; flock 8；持有到脚本退出）
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}                     # ① 探针（演练时换成桩）
CPU_MAX_FILE=${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}     # ② 探针：配额 微秒 周期
CPU_STAT_FILE=${CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}  # 记录用：usage_usec、nr_throttled
GZ_PATTERN=${GZ_PATTERN:-'(^|[ /])gz sim( |$)'}          # 记录用：pgrep -f 的模式
EVAL=${EVAL:-}                                           # 评测入口；缺省 = 各臂冻结代码里的 eval_closed_loop.py（演练时才覆盖）
SELFCHECK=${SELFCHECK:-1}                                # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（演练时才关）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT NAME STEPS CORRIDOR EPT PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg

log() { echo "$(date '+%F %T') $*"; }
export -f log

finish() {   # $1 = rc，$2 = 原因：刷新 summary.md（冻结代码已就位时），写 DONE，退出
  trap - TERM INT
  [[ -f "${SUMM:-/nonexistent}" ]] && "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 生成，DONE rc=$1" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；评完的臂保留，同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ lanes=$LANES gpu=$GPU 对照=$CTRL_REF 机器锁=$MACHINE_LOCK"

# ---------------------------------------------------------------- 起跑条件 ①②（不满足立刻退出）
gpu_list=$("$NVIDIA_SMI" -L 2>&1)
if ! grep -qF "$GPU" <<< "$gpu_list"; then
  finish 12 "条件 ① 不满足：配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpu_list" | tr '\n' ';')），没有起跑"
fi
log "条件 ① 满足：$(grep -F "$GPU" <<< "$gpu_list" | head -n1)"
read -r quota period < "$CPU_MAX_FILE"
if [[ "$quota" == max ]]; then
  cores=max; quota_ok=1
else
  cores=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN{printf "%.2f", q/p}')
  quota_ok=$(awk -v c="$cores" -v m="$MIN_CORES" 'BEGIN{print (c>=m) ? 1 : 0}')
fi
if [[ "$quota_ok" != 1 ]]; then
  finish 13 "条件 ② 不满足：cgroup 配额 $cores 核 < $MIN_CORES 核（$CPU_MAX_FILE：$quota ${period:-?}），没有起跑"
fi
log "条件 ② 满足：cgroup 配额 $cores 核（要求 ≥ $MIN_CORES）"

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露，冻结的是起跑这一刻的提交）
CODE=$OUT/code; CODE_V5=$OUT/code-v5
SUMM=$CODE/experiments/$NAME/summarize.py
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
CTRL_HEAD=$(git -C "$R" rev-parse "$CTRL_REF^{commit}" 2>/dev/null) || finish 4 "CTRL_REF=$CTRL_REF 在 $R 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=experiments/$NAME/run.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
freeze() {   # $1 = 仓库，$2 = 提交，$3 = 目标目录：git archive 解开，再逐文件核对 sim/ 的 blob 哈希
  if [[ ! -f "$3/sim/tools/eval_closed_loop.py" ]]; then
    mkdir -p "$3" && git -C "$1" archive "$2" | tar -x -C "$3" || finish 4 "冻结代码失败：$1 $2"
  fi
  local bad
  bad=$(git -C "$1" ls-tree -r "$2" -- sim | while read -r _m _t sha path; do [[ "$(git hash-object "$3/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
  [[ -z "$bad" ]] || finish 4 "冻结副本与 $2 不一致：$bad"
}
freeze "$SRC" "$CODE_HEAD" "$CODE"
freeze "$R" "$CTRL_HEAD" "$CODE_V5"
echo "$CODE_HEAD" > "$OUT/code_head.txt"
echo "$CTRL_HEAD" > "$OUT/ctrl_head.txt"
[[ -f "$SUMM" ]] || finish 4 "冻结副本里没有 summarize.py（CODE_HEAD $CODE_HEAD 之前的提交？）"
grep -q 'TEACHER_VERSION = "uavrgb.teacher.v7"' "$CODE/sim/tools/teacher.py" || finish 4 "v7 臂的冻结代码里 TEACHER_VERSION 不是 v7"
grep -q 'TEACHER_VERSION = "uavrgb.teacher.v5"' "$CODE_V5/sim/tools/teacher.py" || finish 4 "v5 对照臂的冻结代码里 TEACHER_VERSION 不是 v5"
{ echo "code_head $CODE_HEAD"; echo "ctrl_head $CTRL_HEAD"; echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$CODE_V5/sim/tools/collect.py" "$CODE_V5/sim/tools/eval_closed_loop.py" "$CODE_V5/sim/tools/teacher.py" "$CODE_V5/sim/tools/scene_gen.py" "$CODE_V5/sim/dynamics/e1_mini.json"
  sha256sum "$PLUGIN/libCfSitlSystem.so"; } > "$OUT/inputs.txt" 2>&1

# ---------------------------------------------------------------- 起跑条件 ③ 机器锁（阻塞等待，持有到脚本退出）
mkdir -p "$(dirname "$MACHINE_LOCK")"
exec 8>> "$MACHINE_LOCK"
t_lock=$SECONDS
if ! flock -n 8; then
  log "条件 ③：机器锁 $MACHINE_LOCK 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行）"
  until flock -w "$LOCK_LOG_S" 8; do log "条件 ③：仍在等机器锁（已等 $((SECONDS - t_lock)) s）"; done
fi
lock_wait=$((SECONDS - t_lock))
log "条件 ③ 满足：拿到机器锁，等了 $lock_wait s；持有到脚本退出"
usage_usec() { awk '$1=="usage_usec"{print $2+0}' "$CPU_STAT_FILE"; }
u0=$(usage_usec); w0=$(date +%s.%N); sleep "$IDLE_WINDOW_S"; u1=$(usage_usec); w1=$(date +%s.%N)
idle_cores=$(awk -v a="$u0" -v b="$u1" -v t0="$w0" -v t1="$w1" 'BEGIN{printf "%.2f", (b-a)/1e6/(t1-t0)}')
gz_n=$(pgrep -fc "$GZ_PATTERN" 2>/dev/null); gz_n=${gz_n:-0}
log "起跑记录：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（只记录，不设门）"
cat > "$OUT/start.txt" <<EOF
gpu_ok=是
cpu_quota_cores=$cores
lock_wait_s=$lock_wait
gz_count=$gz_n
idle_window_s=$IDLE_WINDOW_S
idle_cores=$idle_cores
started_at=$(date '+%F %T')
gpu=$GPU
lanes=$LANES
EOF

# ---------------------------------------------------------------- 评测臂
cg_brief() { awk '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} END{printf "usage_usec=%s nr_throttled=%s", u, n}' "$CPU_STAT_FILE"; }
arm_complete() {  # $1 = 臂名，$2 = 请求集数：eval.json 不是 partial 且 出结果 + 采集失败 = 请求数
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$1/eval.json" "$2" 2>/dev/null
}
export -f cg_brief arm_complete
export CPU_STAT_FILE CODE CODE_V5 EVAL

# 一个评测臂：名字|冻结代码目录|seed 串|请求集数。单臂失败不停整个队列，汇总里按 failed 标出来。
run_arm() {
  local name=$1 codedir=$2 seeds=$3 n=$4 ev
  [[ -n "$name" && -n "$seeds" ]] || return 0
  if arm_complete "$name" "$n"; then log "$name 已评完，跳过"; return 0; fi
  if [[ -d "$OUT/$name" ]]; then   # 半成品挪开留证据，不删
    local ts; ts=$(date +%H%M%S)
    mv "$OUT/$name" "$OUT/$name.partial-$ts"
    [[ -f "$OUT/$name.log" ]] && mv "$OUT/$name.log" "$OUT/$name.partial-$ts.log"
  fi
  ev=${EVAL:-$codedir/sim/tools/eval_closed_loop.py}
  log "$name 开始（$(cg_brief)）代码 $codedir seeds $seeds"
  "$PY" "$ev" --policy teacher --task base --split test --seeds "$seeds" --episodes "$n" --steps "$STEPS" \
      --sim-timeout 120.0 --episode-timeout "$EPT" --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  log "$name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)（$(cg_brief)）"
  return 0
}
export -f run_arm

SEEDS_A=$(cut -d, -f1-10 <<< "$SEEDS"); SEEDS_B=$(cut -d, -f11-20 <<< "$SEEDS")
[[ $(tr ',' '\n' <<< "$SEEDS_A" | grep -c .) == 10 && $(tr ',' '\n' <<< "$SEEDS_B" | grep -c .) == 10 ]] || finish 4 "SEEDS 不是 20 个：$SEEDS"
log "阶段 1：v7 主臂（冻结代码 $CODE_HEAD），两半各 10 个 seed，$LANES 条车道"
{ echo "v7-a|$CODE|$SEEDS_A|10"; echo "v7-b|$CODE|$SEEDS_B|10"; } \
  | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n c s e <<< "{}"; run_arm "$n" "$c" "$s" "$e"'
"$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 进行中：v7 臂结束，v5 对照臂未开始或进行中" > "$OUT/summarize.log" 2>&1
log "阶段 2：v5 对照臂（冻结代码 $CTRL_HEAD），同 seed 同参数"
{ echo "v5-a|$CODE_V5|$SEEDS_A|10"; echo "v5-b|$CODE_V5|$SEEDS_B|10"; } \
  | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n c s e <<< "{}"; run_arm "$n" "$c" "$s" "$e"'

missing=""
for a in v7-a v7-b v5-a v5-b; do arm_complete "$a" 10 || missing="$missing $a"; done
[[ -z "$missing" ]] || finish 11 "有臂没跑完：$missing；读 summary.md 与各臂 .log，同一条命令重起可续跑"
finish 0 "四个臂（v7-a v7-b v5-a v5-b）全部跑完；读 summary.md"
