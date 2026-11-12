#!/usr/bin/env bash
# 卡 E2E-P0 第二关（闭环部分）：Gazebo 里 PyTorch 包装器与 C++ 推理对同一批冻结 seed 的逐步动作序列对照。
#
# 做什么：学生 calswa 种子 1（cal_swa_s1_k10.bin）+ 新协议（--action-latency 0.20、--heading-guard 2、K=1 的 --task base --steps 80），
#   与 W1-0 第二轮臂 acc-L0.20-k1-s1（C++ 推理，$REF）的同 seed 集对照 action_executed 序列。
#   两个臂同时跑（2 条车道）：
#     wrap-calswa-s1   --infer-bin = 冻结副本里的 proto/uavrgb_infer_proto（本卡的被测物）
#     cpp-calswa-s1    --infer-bin = 与第二轮同一个 C++ 二进制（对照臂：同一台机器、同一批冻结代码、同一时刻的 C++ 重跑；
#                      没有它，包装器与第二轮的分歧里分不清「推理端」与「仿真本身的不可复现」。卡没要求这个臂，是附加的，报告里单列）
#   含迟到步的集按第二轮条款 R6 重跑一次（<臂>-retry1，只含迟到的 seed，原目录保留）。
#   结束后用 proto/compare_episodes.py 对 $REF 出 compare.json / compare.md：序列是否一致、第一处分歧的步号、
#   分歧步两边的 dispatch_lag_s 与 infer_ms、推理时延 p50/p95、迟到步数。
# seed 的选法（起跑前定，没看过第二轮这三集的结果）：EVAL_SEEDS_BASE_20 的前 3 个，1、113、122。
#
# 运行（先提交再起跑：脚本拿自己与 CODE_HEAD 里的同名文件比哈希，不一致 rc=4）：
#   mkdir -p $OUT && cd $SRC && setsid nohup bash proto/runs/run_step2_gazebo.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；结束落 $OUT/DONE；停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# 续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删），沿用起跑时的 CODE_HEAD。
# DONE 的 rc：0 全部完成并出了对照；3 已有实例；4 冻结代码或脚本版本不一致；10 有臂没跑完；12 配置的 GPU 不在；13 cgroup 配额不足；130 被信号终止。

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
D=/root/rivermind-data/uav_rgb
R=/root/claude-code-backup-20261028/code/uav_rgb          # 主仓库：只借用 .venv-tools 解释器
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-e2e-p0}   # 本卡 worktree：只用来 git archive 冻结代码
OUT=${OUT:-$D/exp-proto-parity-20261112/step2-gazebo}
CODE_HEAD=${CODE_HEAD:-}                                  # 缺省：续跑沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
INFER_CPP=$D/bin-snapshots/main-97ae04b/uavrgb_infer      # 与第二轮同一个 C++ 推理二进制（sha256 b5458338…）
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                      # SITL 插件，与第二轮同一个（公共任务书：不要动这个目录）
CKPT=$D/exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin    # calswa 种子 1
REF=${REF:-$D/exp-action-latency-r2-20261111/acc-L0.20-k1-s1}   # 第二轮参照臂（只读）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740              # 卡指定的 GPU（RTX 2080 Ti）
SEEDS=1,113,122                                           # 冻结 seed 的前 3 个（见上）
EPISODES=3                                                # = seed 个数
L=0.20                                                    # 新协议的动作延迟（第二轮 L_STAR）
GUARD=2                                                   # 航向护栏阈值（同第二轮）
CORRIDOR=3.0                                              # 横向边界（同第二轮）
K1_ARGS="--task base --steps 80"                          # K=1（同第二轮）
EPT=1500                                                  # 每集墙钟上限（秒，同第二轮 K1_EPT）
LANES=2                                                   # 卡规定每个实验 ≤2 条车道：两个臂各占一条
MIN_CORES=${MIN_CORES:-12}                                # cgroup 配额下限（核，同第二轮）
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # Gazebo 池的锁（阻塞等待，持有到退出）
EVAL=${EVAL:-}                                            # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练时才覆盖）
SELFCHECK=${SELFCHECK:-1}                                 # 1 = 起跑前核对本脚本与 CODE_HEAD 里同名文件一致（演练时才关）
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}                      # GPU 探针（演练时换桩）
CPU_MAX_FILE=${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}
# ----------------------------------------------------------------
set -u
export LC_ALL=C
export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export UAV_RGB_PROTO_DEVICE=cuda UAV_RGB_PROTO_PYTHON=$D/teacher-venv/bin/python   # 包装器显式用 CUDA（不可用则报错，不静默退回 CPU）
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
source "$SRC/proto/runs/common.sh"
daemon_begin "Gazebo 对照：包装器与 C++ 推理，calswa s1，seed $SEEDS，L=$L，护栏 $GUARD，车道 $LANES，锁 $MACHINE_LOCK"

# ---------------------------------------------------------------- 起跑条件
gpu_list=$("$NVIDIA_SMI" -L 2>&1)
grep -qF "$GPU" <<< "$gpu_list" || finish 12 "配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpu_list" | tr '\n' ';')），没有起跑"
read -r quota period < "$CPU_MAX_FILE"
if [[ "$quota" != max ]]; then
  cores=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN{printf "%.2f", q/p}')
  awk -v c="$cores" -v m="$MIN_CORES" 'BEGIN{exit (c>=m) ? 0 : 1}' || finish 13 "cgroup 配额 $cores 核 < $MIN_CORES 核，没有起跑"
fi
log "GPU 在、cgroup 配额 ${cores:-max} 核"

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露）
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=proto/runs/run_step2_gazebo.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
echo "$CODE_HEAD" > "$OUT/code_head.txt"
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim proto | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
WRAP=$CODE/proto/uavrgb_infer_proto
SUMM=$CODE/experiments/2026-11-10-action-latency/summarize_r2.py
[[ -x "$WRAP" && -f "$SUMM" && -f "$CODE/proto/compare_episodes.py" ]] || finish 4 "冻结副本里缺启动器、summarize_r2.py 或 compare_episodes.py"
{ echo "code_head $CODE_HEAD"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/student.py" "$CODE/proto/infer_wrap.py" \
            "$CODE/proto/model.py" "$CODE/proto/ckpt_io.py" "$WRAP" "$CKPT" "$INFER_CPP" "$PLUGIN/libCfSitlSystem.so"
  echo "torch $("$UAV_RGB_PROTO_PYTHON" -c 'import torch,numpy;print(torch.__version__, numpy.__version__)' 2>&1)"; } > "$OUT/inputs.txt" 2>&1
# 冻结副本的 sim/ 与第二轮冻结副本的 sim/ 的差异（解释分歧时要看）
R2CODE=$(dirname "$REF")/code
if [[ -d "$R2CODE/sim" ]]; then diff -rq "$R2CODE/sim" "$CODE/sim" > "$OUT/sim_diff_vs_r2.txt" 2>&1; echo "（diff 退出码 $?，空文件 = 无差异）" >> "$OUT/sim_diff_vs_r2.txt"; fi

# ---------------------------------------------------------------- Gazebo 池的锁
take_lock "$MACHINE_LOCK" "Gazebo 池"
# 起跑时的负载只记录不卡：第二轮的「容器 < 1.5 核才起跑」是它自己的起跑条件，本卡的规则只要求排锁；
# 迟到步由 R6 重跑兜底，cgroup 的 usage_usec 与 nr_throttled 在每个臂的 .cg 里，供事后判断负载是否扰动了时延
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
run_arm() {   # $1 = 臂名，$2 = --infer-bin，$3 = seed 列表，$4 = 集数
  local name=$1 infer=$2 seeds=$3 n=$4
  if arm_complete "$name" "$n"; then log "$name 已评完，跳过"; return 0; fi
  if [[ -d "$OUT/$name" ]]; then
    local ts; ts=$(date +%H%M%S)
    mv "$OUT/$name" "$OUT/$name.partial-$ts"
    for ext in log cg; do [[ -f "$OUT/$name.$ext" ]] && mv "$OUT/$name.$ext" "$OUT/$name.partial-$ts.$ext"; done
  fi
  cg_mark start "$name"
  log "$name 开始（$(cg_brief)）"
  # 参数与第二轮 acc_job 逐项相同（run_r2.sh），只换 --infer-bin、--seeds、--episodes、--out
  "$PY" "$EVAL" --policy student --ckpt "$CKPT" --infer-bin "$infer" --heading-guard "$GUARD" $K1_ARGS --split test --seeds "$seeds" \
      --corridor-half-width "$CORRIDOR" --action-latency "$L" --episodes "$n" --sim-timeout 120.0 --episode-timeout "$EPT" \
      --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  cg_mark end "$name"
  log "$name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)（$(cg_brief)）"
  return 0
}
retry_arm() {   # 含迟到步的集重跑一次（第二轮条款 R6）
  local name=$1 infer=$2 late n
  late=$("$PY" "$SUMM" --late-seeds "$OUT/$name" 2>/dev/null)
  [[ -n "$late" ]] || return 0
  n=$(tr ',' '\n' <<< "$late" | grep -c .)
  log "$name 含迟到步的集 [$late]，重跑一次 → $name-retry1"
  run_arm "$name-retry1" "$infer" "$late" "$n"
}
arm_job() { run_arm "$1" "$2" "$SEEDS" "$EPISODES"; retry_arm "$1" "$2"; }

arm_job wrap-calswa-s1 "$WRAP" &
arm_job cpp-calswa-s1 "$INFER_CPP" &
wait
log "两个臂都已结束，出对照"

# ---------------------------------------------------------------- 对照
"$PY" "$CODE/proto/compare_episodes.py" --ref "$REF" --arm wrap="$OUT/wrap-calswa-s1" --arm cpp="$OUT/cpp-calswa-s1" --seeds "$SEEDS" \
  --out "$OUT/compare.json" --md "$OUT/compare.md" > "$OUT/compare.log" 2>&1
crc=$?
log "compare rc=$crc"
missing=""
for a in wrap-calswa-s1 cpp-calswa-s1; do
  arm_complete "$a" "$EPISODES" || arm_complete "$a-retry1" "$(tr ',' '\n' <<< "$("$PY" "$SUMM" --late-seeds "$OUT/$a" 2>/dev/null)" | grep -c .)" || missing="$missing $a"
done
[[ $crc -eq 0 ]] || finish 10 "compare 失败 rc=$crc，读 $OUT/compare.log"
[[ -z "$missing" ]] || finish 10 "有臂没跑完：$missing；读 $OUT/*.log，同一条命令重起可续跑"
finish 0 "完成：$OUT/compare.md、compare.json"
