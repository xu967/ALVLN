#!/usr/bin/env bash
# 卡 E2E-P0 第三关验收 (b)：PyTorch 原型训练出的权重，经主线的权重平均与校准后，在 Gazebo 闭环里的成功数对 W1-0 第二轮三个 K=1 臂。
#
# 做什么：
#   1. 前提：第三关离线部分（run_step3_train.sh）已跑完，且验收 (a) 的字面读法是 PASS（卡规定 (a) 没过就不跑 (b)）。
#      主会话明确决定要在 (a) 没过时仍跑 (b) 时，才可用 ALLOW_OFFLINE_FAIL=1 起跑；默认不允许。
#   2. 每个训练种子（0、1、2）的原型 run（ckpt_ep01..30.bin 是 export.py 写出的 C++ 格式）交给主线工具
#      tools/average_ckpt.py --run --data --out --last-k 10 --infer-bin：末 10 轮权重平均，再用真实 uavrgb_infer 在 val 帧上校准 FORWARD 先验。
#      工具对照：同一条命令对 C++ 参照 run e1-noise/seed0，产物的 sha256 必须等于第二轮用的 cal_swa_s0_k10.bin（2fb25b30…），否则 rc=15 不起跑。
#   3. 三个臂 proto-calswa-s{0,1,2}：20 个冻结 seed × 新协议（--action-latency 0.20、--heading-guard 2、--task base --steps 80、corridor 3.0），
#      --infer-bin 与第二轮同一个 C++ 二进制；参数与第二轮 acc_job（run_r2.sh）逐项相同，只换 --ckpt 与 --out。2 条车道：先起两个臂，一个结束再起第三个。
#      含迟到步的集按第二轮条款 R6 重跑一次（<臂>-retry1，只含迟到的 seed，原目录保留）。
#   4. 出汇总（proto/closedloop_summary.py）：逐训练种子、逐集配对，总成功数对第二轮 acc-L0.20-k1-s{0,1,2}（18+16+19 = 53/60）。
#
# 预注册（任务卡，起跑前定下，不改）：原型三个训练种子的总成功数与 53/60 相差不超过 6 集（47..59）；超过 6 集说明两条线的训练不等价，
#   停下来查，不再继续用。判据只看总成功数；逐种子、McNemar、迟到集、平均核数只报告。迟到后重跑仍迟到或缺失的集按没成功。
#   第二轮三个臂没有含迟到步的集（eval.json 的 late_dispatch_episodes 都为空），所以 53/60 不含 R6 调整。
#
# 运行（先提交再起跑：脚本拿自己与 CODE_HEAD 里的同名文件比哈希，不一致 rc=4）：
#   mkdir -p $OUT && cd $SRC && setsid nohup bash proto/runs/run_step3_closedloop.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；结束落 $OUT/DONE；停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# 续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删），沿用起跑时的 CODE_HEAD；已做好的平均与校准产物不重做。
# DONE 的 rc：0 三个臂都评完并出了汇总（预注册通过与否写在原因里与 $OUT/VERDICT）；3 已有实例；4 冻结代码或脚本版本不一致；
#   5 离线部分没跑完或 (a) 没过；10 有臂没跑完；12 配置的 GPU 不在；13 cgroup 配额不足；15 权重平均与校准失败或工具对照不一致；130 被信号终止。
# 离线部分的守护进程还没结束时，本脚本先等它的 DONE（WAIT_TRAIN=1，缺省），再检查 (a) 的结论；两个守护进程可以先后脚起，本脚本不占任何锁地等。
# 在 Gazebo 池的锁上阻塞等待（不限时；等待期间每 10 分钟记一行）。

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
D=/root/rivermind-data/uav_rgb
R=/root/claude-code-backup-20261028/code/uav_rgb          # 主仓库：只借用 .venv-tools 解释器
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-e2e-p0}   # 本卡 worktree：只用来 git archive 冻结代码
OUT=${OUT:-$D/exp-proto-parity-20261112/step3-closedloop}
TRAIN_OUT=${TRAIN_OUT:-$D/exp-proto-parity-20261112/step3-train}   # 第三关离线部分的输出（run_step3_train.sh）
CODE_HEAD=${CODE_HEAD:-}                                  # 缺省：续跑沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
DATA=${DATA:-$D/tmp/agent6-distill/trunc/ds/dataset.uavr} # 训练与校准平均用的数据集（average_ckpt 要求 sha256 与 ckpt 契约一致）
INFER=${INFER:-$D/bin-snapshots/main-97ae04b/uavrgb_infer} # 与第二轮同一个 C++ 推理二进制（sha256 b5458338…）
PLUGIN=${PLUGIN:-$D/tmp/build-cf-sitl-lib31/gz}           # SITL 插件，与第二轮同一个（公共任务书：不要动这个目录）
REF_ROOT=${REF_ROOT:-$D/exp-action-latency-r2-20261111}   # 第二轮输出（只读）：acc-L0.20-k1-s{0,1,2}
REF_CAL_S0=$D/exp-swa-calib-20261110/ckpt/cal_swa_s0_k10.bin     # 工具对照的目标产物
REF_RUN_S0=${REF_RUN_S0:-$D/exp-student-select-20261109/e1-noise/seed0/run}   # 工具对照的输入（C++ 参照 run）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740              # 卡指定的 GPU（RTX 2080 Ti）
TRAIN_SEEDS=${TRAIN_SEEDS:-"0 1 2"}                       # 原型的三个训练种子，对应第二轮 s0、s1、s2
SEEDS=${SEEDS:-1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654}   # tools/eval_seeds.py EVAL_SEEDS_BASE_20（与第二轮相同）
EPISODES=${EPISODES:-20}                                  # = seed 个数
LAST_K=${LAST_K:-10}                                      # 原型权重平均的轮数（第二轮 cal_swa_*_k10；演练时用环境变量改小）
CTRL_LAST_K=10                                            # 工具对照固定用第二轮的 k=10
FINAL_EPOCH=${FINAL_EPOCH:-30}                            # 原型训练的轮数（任务卡：30 轮；演练时改小）
L=0.20                                                    # 新协议的动作延迟（第二轮 L_STAR）
GUARD=2                                                   # 航向护栏阈值（同第二轮）
CORRIDOR=3.0                                              # 横向边界（同第二轮）
K1_ARGS="--task base --steps 80"                          # K=1（同第二轮）
EPT=1500                                                  # 每集墙钟上限（秒，同第二轮 K1_EPT）
LANES=${LANES:-2}                                         # 卡规定每个实验 ≤2 条车道（演练时可改成 1；大于 2 拒绝起跑）
TOL=6                                                     # 任务卡预注册：总成功数与第二轮相差不超过 6 集
MIN_CORES=${MIN_CORES:-12}                                # cgroup 配额下限（核，同第二轮）
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # Gazebo 池的锁（阻塞等待，持有到退出）
EVAL=${EVAL:-}                                            # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练时才覆盖）
SELFCHECK=${SELFCHECK:-1}                                 # 1 = 起跑前核对本脚本与 CODE_HEAD 里同名文件一致（演练时才关）
ALLOW_OFFLINE_FAIL=${ALLOW_OFFLINE_FAIL:-0}               # 1 = (a) 没过也跑 (b)；只有主会话明确决定后才设
WAIT_TRAIN=${WAIT_TRAIN:-1}                               # 1 = 离线部分的守护进程还没结束就等它（它自己在训练池上排队）；0 = 没结束就 rc=5 退出
WAIT_POLL_S=${WAIT_POLL_S:-60}                            # 等离线部分结束时的轮询间隔（秒）
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}                      # GPU 探针（演练时换桩）
CPU_MAX_FILE=${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}
# ----------------------------------------------------------------
set -u
export LC_ALL=C
export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" "$OUT/ckpt" /tmp/xdg && chmod 700 /tmp/xdg
source "$SRC/proto/runs/common.sh"
daemon_begin "闭环验收 (b)：原型训练种子 [$TRAIN_SEEDS] 的 calswa 权重，$EPISODES 个冻结 seed，L=$L，护栏 $GUARD，车道 $LANES，锁 $MACHINE_LOCK"

# ---------------------------------------------------------------- 起跑条件
(( LANES >= 1 && LANES <= 2 )) || finish 4 "LANES=$LANES：卡规定每个实验最多 2 条车道，没有起跑"
gpu_list=$("$NVIDIA_SMI" -L 2>&1)
grep -qF "$GPU" <<< "$gpu_list" || finish 12 "配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpu_list" | tr '\n' ';')），没有起跑"
read -r quota period < "$CPU_MAX_FILE"
if [[ "$quota" != max ]]; then
  cores=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN{printf "%.2f", q/p}')
  awk -v c="$cores" -v m="$MIN_CORES" 'BEGIN{exit (c>=m) ? 0 : 1}' || finish 13 "cgroup 配额 $cores 核 < $MIN_CORES 核，没有起跑"
fi
log "GPU 在、cgroup 配额 ${cores:-max} 核"

# 第三关离线部分：守护进程结束（rc=0），三个训练都跑完，(a) 的字面读法 PASS
if [[ ! -f "$TRAIN_OUT/DONE" ]]; then
  [[ "$WAIT_TRAIN" == 1 ]] || finish 5 "离线部分还没结束（没有 $TRAIN_OUT/DONE）且 WAIT_TRAIN=0，没有起跑"
  log "离线部分还没结束，等 $TRAIN_OUT/DONE（每 $LOCK_LOG_S s 记一行）"
  t_wait=$SECONDS; t_last=$SECONDS
  while [[ ! -f "$TRAIN_OUT/DONE" ]]; do
    if [[ -f "$TRAIN_OUT/run.pid" ]] && ! kill -0 "$(cat "$TRAIN_OUT/run.pid")" 2>/dev/null; then
      finish 5 "离线部分的守护进程（pid $(cat "$TRAIN_OUT/run.pid")）已不在，也没有 DONE；没有起跑"
    fi
    sleep "$WAIT_POLL_S"
    if (( SECONDS - t_last >= LOCK_LOG_S )); then
      log "仍在等离线部分结束（已等 $((SECONDS - t_wait)) s）：$(tail -n 1 "$TRAIN_OUT/run.log" 2>/dev/null | cut -c1-160)"
      t_last=$SECONDS
    fi
  done
  log "离线部分已结束：$(cat "$TRAIN_OUT/DONE")"
fi
tr_rc=$(sed -n 's/^rc=\([0-9]*\).*/\1/p' "$TRAIN_OUT/DONE")
[[ "$tr_rc" == 0 ]] || finish 5 "离线部分的 DONE 不是 rc=0（$(cat "$TRAIN_OUT/DONE")），没有起跑"
for S in $TRAIN_SEEDS; do
  [[ -f "$TRAIN_OUT/seed$S/run/DONE" ]] || finish 5 "原型种子 $S 的训练没有 DONE（$TRAIN_OUT/seed$S/run/DONE）；离线部分没跑完，没有起跑"
  [[ -f "$TRAIN_OUT/seed$S/run/ckpt_ep$(printf %02d "$FINAL_EPOCH").bin" ]] || finish 5 "原型种子 $S 缺 ckpt_ep$(printf %02d "$FINAL_EPOCH").bin，没有起跑"
done
[[ -f "$TRAIN_OUT/OFFLINE_VERDICT" ]] || finish 5 "没有 $TRAIN_OUT/OFFLINE_VERDICT：离线验收 (a) 还没出结果，没有起跑"
lit=$(sed -n 's/^literal=//p' "$TRAIN_OUT/OFFLINE_VERDICT" | head -n1)
if [[ "$lit" != PASS ]]; then
  if [[ "$ALLOW_OFFLINE_FAIL" == 1 ]]; then
    log "警告：验收 (a) 字面读法是 ${lit:-?}，ALLOW_OFFLINE_FAIL=1，按主会话的决定继续跑 (b)"
  else
    finish 5 "验收 (a) 字面读法是 ${lit:-?}（$TRAIN_OUT/OFFLINE_VERDICT）；卡规定 (a) 没过不跑 (b)，没有起跑。主会话决定要跑时用 ALLOW_OFFLINE_FAIL=1"
  fi
fi
cp -f "$TRAIN_OUT/OFFLINE_VERDICT" "$OUT/OFFLINE_VERDICT.copy"

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露）
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=proto/runs/run_step3_closedloop.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
echo "$CODE_HEAD" > "$OUT/code_head.txt"
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim proto tools | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
AVG=$CODE/tools/average_ckpt.py
SUMM=$CODE/experiments/2026-11-10-action-latency/summarize_r2.py
SUMMARY=$CODE/proto/closedloop_summary.py
[[ -f "$AVG" && -f "$SUMM" && -f "$SUMMARY" ]] || finish 4 "冻结副本里缺 average_ckpt.py、summarize_r2.py 或 closedloop_summary.py"

# ---------------------------------------------------------------- 权重平均与校准（先于等锁：只占 GPU 几秒，早暴露问题）
average() {   # $1 = run 目录，$2 = 产物路径，$3 = --last-k；产物与 .json 都在就跳过。工具拒绝覆盖已有的 --out，所以半成品要先挪开
  local run=$1 out=$2 k=$3
  if [[ -f "$out" && -f "$out.json" ]]; then log "$(basename "$out") 已在，跳过"; return 0; fi
  [[ -e "$out" ]] && mv "$out" "$out.partial-$(date +%H%M%S)"
  [[ -e "$out.json" ]] && mv "$out.json" "$out.json.partial-$(date +%H%M%S)"
  nice -n 10 "$PY" "$AVG" --run "$run" --data "$DATA" --out "$out" --last-k "$k" --infer-bin "$INFER" > "$out.log" 2>&1
  local rc=$?
  log "average_ckpt $(basename "$(dirname "$run")")/$(basename "$run") → $(basename "$out") rc=$rc：$(tail -n 2 "$out.log" | tr '\n' ' ' | cut -c1-200)"
  return $rc
}
for S in $TRAIN_SEEDS; do
  average "$TRAIN_OUT/seed$S/run" "$OUT/ckpt/cal_swa_p${S}_k10.bin" "$LAST_K" || finish 15 "原型种子 $S 的权重平均与校准失败，读 $OUT/ckpt/cal_swa_p${S}_k10.bin.log"
done
average "$REF_RUN_S0" "$OUT/ckpt/control_cal_swa_s0_k10.bin" "$CTRL_LAST_K" || finish 15 "工具对照（C++ 参照 run e1-noise/seed0）失败，读 $OUT/ckpt/control_cal_swa_s0_k10.bin.log"
want_sha=$(sha256sum < "$REF_CAL_S0" | cut -d' ' -f1)
got_sha=$(sha256sum < "$OUT/ckpt/control_cal_swa_s0_k10.bin" | cut -d' ' -f1)
echo "tool_control want=$want_sha got=$got_sha match=$([[ $want_sha == "$got_sha" ]] && echo yes || echo no)" > "$OUT/tool_control.txt"
[[ "$want_sha" == "$got_sha" ]] || finish 15 "工具对照不一致：average_ckpt 对 C++ 参照 run 的产物 sha256 $got_sha ≠ 第二轮的 cal_swa_s0_k10.bin $want_sha；闭环没有起跑"
log "工具对照一致：average_ckpt 对 C++ 参照 run 的产物与第二轮的 cal_swa_s0_k10.bin 逐字节相同"
{ echo "code_head $CODE_HEAD"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/student.py" "$AVG" "$SUMMARY" "$DATA" "$INFER" "$PLUGIN/libCfSitlSystem.so" "$OUT"/ckpt/*.bin
  for S in $TRAIN_SEEDS; do sha256sum "$TRAIN_OUT/seed$S/run/ckpt_ep"*.bin; done
  cat "$TRAIN_OUT/OFFLINE_VERDICT"; } > "$OUT/inputs.txt" 2>&1
R2CODE=$REF_ROOT/code
if [[ -d "$R2CODE/sim" ]]; then diff -rq "$R2CODE/sim" "$CODE/sim" > "$OUT/sim_diff_vs_r2.txt" 2>&1; echo "（diff 退出码 $?，空文件 = 无差异）" >> "$OUT/sim_diff_vs_r2.txt"; fi

# ---------------------------------------------------------------- Gazebo 池的锁
take_lock "$MACHINE_LOCK" "Gazebo 池"
cg_mark start all
u0=$(awk '$1=="usage_usec"{print $2+0}' "$CPU_STAT_FILE"); w0=$(date +%s.%N); sleep "${IDLE_WINDOW_S:-15}"
u1=$(awk '$1=="usage_usec"{print $2+0}' "$CPU_STAT_FILE"); w1=$(date +%s.%N)
cat > "$OUT/start.txt" <<EOF
lock_wait_s=$LOCK_WAIT_S
started_at=$(date '+%F %T')
gpu=$GPU
lanes=$LANES
gz_sim_processes=$(pgrep -fc '(^|[ /])gz sim( |$)' 2>/dev/null || true)
cpu_quota_cores=${cores:-max}
idle_window_s=${IDLE_WINDOW_S:-15}
container_cores_in_window=$(awk -v a="$u0" -v b="$u1" -v t0="$w0" -v t1="$w1" 'BEGIN{printf "%.2f", (b-a)/1e6/(t1-t0)}')
EOF
log "起跑时负载：$(tr '\n' ' ' < "$OUT/start.txt")"

# ---------------------------------------------------------------- 评测臂
arm_complete() {  # $1 = 臂名，$2 = 请求集数
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$1/eval.json" "$2" 2>/dev/null
}
run_arm() {   # $1 = 臂名，$2 = --ckpt，$3 = seed 列表，$4 = 集数
  local name=$1 ckpt=$2 seeds=$3 n=$4
  if arm_complete "$name" "$n"; then log "$name 已评完，跳过"; return 0; fi
  if [[ -d "$OUT/$name" ]]; then
    local ts; ts=$(date +%H%M%S)
    mv "$OUT/$name" "$OUT/$name.partial-$ts"
    for ext in log cg; do [[ -f "$OUT/$name.$ext" ]] && mv "$OUT/$name.$ext" "$OUT/$name.partial-$ts.$ext"; done
  fi
  cg_mark start "$name"
  log "$name 开始（$(cg_brief)）"
  # 参数与第二轮 acc_job（run_r2.sh）逐项相同，只换 --ckpt、--seeds、--episodes、--out
  "$PY" "$EVAL" --policy student --ckpt "$ckpt" --infer-bin "$INFER" --heading-guard "$GUARD" $K1_ARGS --split test --seeds "$seeds" \
      --corridor-half-width "$CORRIDOR" --action-latency "$L" --episodes "$n" --sim-timeout 120.0 --episode-timeout "$EPT" \
      --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  cg_mark end "$name"
  log "$name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)（$(cg_brief)）"
  return 0
}
retry_arm() {   # 含迟到步的集重跑一次（第二轮条款 R6）
  local name=$1 ckpt=$2 late n
  late=$("$PY" "$SUMM" --late-seeds "$OUT/$name" 2>/dev/null)
  [[ -n "$late" ]] || return 0
  n=$(tr ',' '\n' <<< "$late" | grep -c .)
  log "$name 含迟到步的集 [$late]，重跑一次 → $name-retry1"
  run_arm "$name-retry1" "$ckpt" "$late" "$n"
}
arm_job() { run_arm "$1" "$2" "$SEEDS" "$EPISODES"; retry_arm "$1" "$2"; }
wait_slot() { while (( $(jobs -rp | wc -l) >= LANES )); do sleep 5; done; }   # 轮询而不用 wait -n：后者会吞掉已结束任务的 rc

for S in $TRAIN_SEEDS; do
  wait_slot
  arm_job "proto-calswa-s$S" "$OUT/ckpt/cal_swa_p${S}_k10.bin" &
  sleep 2
done
wait
cg_mark end all
log "三个臂都已结束，出汇总"

# ---------------------------------------------------------------- 汇总
args=()
for S in $TRAIN_SEEDS; do args+=(--proto "s$S=$OUT/proto-calswa-s$S" --ref "s$S=$REF_ROOT/acc-L$(printf %.2f "$L")-k1-s$S"); done
"$PY" "$SUMMARY" "${args[@]}" --seeds "$SEEDS" --tol "$TOL" --out-json "$OUT/closedloop.json" --out-md "$OUT/closedloop.md" > "$OUT/closedloop.log" 2>&1
src=$?
log "closedloop_summary rc=$src（0 通过且完整；1 差超过容差；2 数据不全）"
verdict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUT/closedloop.json" 2>/dev/null)
echo "$verdict" > "$OUT/VERDICT"
missing=""
for S in $TRAIN_SEEDS; do
  name=proto-calswa-s$S
  late=$("$PY" "$SUMM" --late-seeds "$OUT/$name" 2>/dev/null)
  if [[ -n "$late" ]]; then
    arm_complete "$name-retry1" "$(tr ',' '\n' <<< "$late" | grep -c .)" || missing="$missing $name-retry1"
  fi
  arm_complete "$name" "$EPISODES" || missing="$missing $name"
done
[[ -n "$verdict" ]] || finish 10 "汇总没有产出，读 $OUT/closedloop.log"
[[ -z "$missing" ]] || finish 10 "有臂没跑完：$missing；读 $OUT/*.log，同一条命令重起可续跑。当前汇总：$verdict"
finish 0 "三个臂都评完：$verdict；读 $OUT/closedloop.md"
