#!/usr/bin/env bash
# 卡 E2E-P0 第三关（离线部分）：PyTorch 原型线用主线配方训 3 个种子，与 C++ 三个参照种子对拍离线指标（验收 a）。
#
# 做什么：
#   1. 冻结代码（git archive 本卡 HEAD）；核对数据集 sha256 与 e1-noise 参照 config.json 一致；
#   2. 排训练池的锁（阻塞等待，持有到退出）；
#   3. proto/train.py 训 SEEDS（同时跑，组内最多 3 个训练进程），每个 30 轮，产物与导出的 C++ checkpoint 在 $OUT/seed<S>/run/；
#   4. 每个种子的 ckpt_best.bin 交给主线 uavrgb_train --eval-only 读回，核对混淆矩阵与 proto 自己算的逐格相同（crosscheck_evalonly.py）；
#   5. accept_offline.py 对 e1-noise 三个参照种子出验收 (a) 的对拍表，判定落 $OUT/OFFLINE_VERDICT；
#   6. 对照臂（C++ 自己与自己对拍，用来读懂判据）：
#        - C++ 种子 0 重跑一次（主线二进制 + 参照同配置）：核对它与 e1-noise seed0 逐步 loss 是否逐位相同（同一二进制同种子是否可复现）；总是跑
#        - C++ 额外种子 CONTROL_SEEDS：只在字面判据没过时跑（RUN_CONTROL=auto），量判据对「等价配方」的误判率；RUN_CONTROL=1 强制、0 禁止
#      对照臂也在同一次持锁内完成，同样最多 3 个进程同时跑。
#
# 运行（先提交再起跑：脚本拿自己与 CODE_HEAD 里的同名文件比哈希，不一致 rc=4）：
#   mkdir -p $OUT && cd $SRC && setsid nohup bash proto/runs/run_step3_train.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；结束落 $OUT/DONE；停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# 续跑：同一条命令重起，已有 DONE 的种子跳过，半成品挪到 run.partial-<时刻>（不删）。
# DONE 的 rc：0 全部完成；3 已有实例；4 冻结代码、脚本版本或数据集不一致；10 有训练或核对失败；12 配置的 GPU 不在；130 被信号终止。
# 验收 (a) 的判定与后续：$OUT/OFFLINE_VERDICT 第一行 literal=PASS|FAIL；PASS 才去起闭环（run_step3_closedloop.sh），FAIL 不起（任务卡）。

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
D=/root/rivermind-data/uav_rgb
R=/root/claude-code-backup-20261028/code/uav_rgb          # 主仓库：只借用 .venv-tools 解释器（numpy，给对拍脚本用）
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-e2e-p0}   # 本卡 worktree：只用来 git archive 冻结代码
OUT=${OUT:-$D/exp-proto-parity-20261112/step3-train}
CODE_HEAD=${CODE_HEAD:-}                                  # 缺省：续跑沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
PY_PROTO=$D/teacher-venv/bin/python                       # 原型训练用（torch 2.14.1+cu126；只读借用，不装包）
PY=$R/.venv-tools/bin/python                              # 对拍脚本用（numpy）
TRAIN_CPP=$D/tmp/build-check-main-1111/uavrgb_train       # 卡指定：只读借用，不在该目录重新 cmake
DATA=$D/tmp/agent6-distill/trunc/ds/dataset.uavr          # 任务卡验收 (a) 指定的数据集
REF_ROOT=$D/exp-student-select-20261109/e1-noise          # C++ 三个参照种子：seed0..2/run
SEEDS=${SEEDS:-"0 1 2"}                                   # 原型种子：与 C++ 参照种子同号（同号 = 同初值、同打乱，轨迹对比有意义）
EPOCHS=${EPOCHS:-30}                                      # 参照 config.json: epochs=30（演练时用环境变量改小）
BATCH=128                                                 # 同上: batch=128
LR=0.001                                                  # 同上: lr=0.001
MAX_PAR=3                                                 # 公共任务书 §4：训练池组内最多 3 个训练进程同时跑
RUN_CONTROL=${RUN_CONTROL:-auto}                          # 额外 C++ 种子：auto = 字面判据没过才跑；1 强制；0 禁止
CONTROL_SEEDS=${CONTROL_SEEDS:-"3 4 5 6 7"}               # 额外 C++ 种子（凑到 8 个 C++ 种子，才能估判据的误判率）
RERUN_SEED=0                                              # C++ 重跑的种子（对 e1-noise seed0）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740              # 卡指定的 GPU（RTX 2080 Ti）
TRAIN_LOCK=${TRAIN_LOCK:-$D/tmp/phase12-cards/train.lock} # 训练池的锁（阻塞等待，持有到退出）
SELFCHECK=${SELFCHECK:-1}                                 # 1 = 起跑前核对本脚本与 CODE_HEAD 里同名文件一致（演练时才关）
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
# ----------------------------------------------------------------
set -u
export LC_ALL=C CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1               # 每个训练进程约 1 核（公共任务书 §4）
export OUT CODE=$OUT/code
mkdir -p "$OUT/tmp"
source "$SRC/proto/runs/common.sh"
daemon_begin "原型训练 seeds [$SEEDS] × $EPOCHS 轮 → $OUT/seed<S>/run；对拍 e1-noise；锁 $TRAIN_LOCK"

# ---------------------------------------------------------------- 起跑条件与冻结代码
gpu_list=$("$NVIDIA_SMI" -L 2>&1)
grep -qF "$GPU" <<< "$gpu_list" || finish 12 "配置的 GPU $GPU 不在 nvidia-smi -L 的列表里，没有起跑"
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=proto/runs/run_step3_train.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/proto/train.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
echo "$CODE_HEAD" > "$OUT/code_head.txt"
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- proto | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
ds_sha=$(sha256sum "$DATA" | cut -d' ' -f1)
ref_sha=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_sha256"])' "$REF_ROOT/seed0/run/config.json")
[[ "$ds_sha" == "$ref_sha" ]] || finish 4 "数据集 sha256 $ds_sha 与参照 config.json 的 $ref_sha 不一致"
{ echo "code_head $CODE_HEAD"
  echo "dataset_sha256 $ds_sha"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$TRAIN_CPP" "$PY_PROTO" 2>&1
  echo "torch $("$PY_PROTO" -c 'import torch,numpy;print(torch.__version__, numpy.__version__, torch.version.cuda)' 2>&1)"
  echo "numpy(PY) $("$PY" -c 'import numpy;print(numpy.__version__)' 2>&1)"; } > "$OUT/inputs.txt" 2>&1

# ---------------------------------------------------------------- 训练池的锁
take_lock "$TRAIN_LOCK" "训练池"
cg_mark start all
log "持锁开始（$(cg_brief)）"

# 并发池：最多 MAX_PAR 个后台任务
declare -A PIDS=()      # 任务名 → pid
declare -A RCS=()
wait_slot() { while (( $(jobs -rp | wc -l) >= MAX_PAR )); do sleep 5; done; }   # 轮询而不用 wait -n：后者会吞掉已结束任务的 rc
reap() {        # 等全部后台任务结束，把各自的 rc 记进 RCS
  local name
  for name in "${!PIDS[@]}"; do wait "${PIDS[$name]}"; RCS[$name]=$?; done
  PIDS=()
}
fresh_dir() {   # $1 = run 目录：有 DONE 返回 1（已完成，跳过）；半成品挪走；其余新建
  local d=$1
  if [[ -f "$d/DONE" ]]; then return 1; fi
  if [[ -d "$d" ]]; then mv "$d" "$d.partial-$(date +%H%M%S)"; fi
  mkdir -p "$(dirname "$d")"
  return 0
}

# ---------------------------------------------------------------- 3. 原型训练
fail=0
cg_mark start proto
for S in $SEEDS; do
  RUN=$OUT/seed$S/run
  if ! fresh_dir "$RUN"; then log "seed $S 已训完（$RUN/DONE），跳过"; continue; fi
  wait_slot
  (
    cd "$CODE" && nice -n 10 "$PY_PROTO" -W ignore proto/train.py --seed "$S" --out "$RUN" --data "$DATA" --epochs "$EPOCHS" --batch "$BATCH" --device cuda \
      > "$OUT/seed$S/train.log" 2>&1
  ) &
  PIDS[proto-$S]=$!
  log "seed $S 起训 pid=${PIDS[proto-$S]}（$(cg_brief)）"
done
reap
for S in $SEEDS; do
  rc=${RCS[proto-$S]:-0}
  if [[ ! -f "$OUT/seed$S/run/DONE" ]]; then log "seed $S 训练失败 rc=$rc，读 $OUT/seed$S/train.log"; fail=1; else log "seed $S 训练完成 rc=$rc：$(tail -n 1 "$OUT/seed$S/train.log")"; fi
done
cg_mark end proto
log "原型训练全部结束（$(cg_brief)）"

# ---------------------------------------------------------------- 4. C++ 读回核对
for S in $SEEDS; do
  [[ -f "$OUT/seed$S/run/DONE" ]] || continue
  EV=$OUT/seed$S/evalonly
  if grep -q '"identical": true' "$OUT/seed$S/crosscheck.json" 2>/dev/null; then log "seed $S 的读回核对已通过，跳过"; continue; fi
  rm -rf "$EV"
  nice -n 10 "$TRAIN_CPP" --data "$DATA" --eval-only "$OUT/seed$S/run/ckpt_best.bin" --out "$EV" --split all --drop-collision > "$OUT/seed$S/evalonly.log" 2>&1
  erc=$?
  "$PY" "$CODE/proto/crosscheck_evalonly.py" --run "$OUT/seed$S/run" --eval-dir "$EV" --out "$OUT/seed$S/crosscheck.json" > "$OUT/seed$S/crosscheck.log" 2>&1
  crc=$?
  log "seed $S 的 ckpt_best.bin 交给 C++ eval-only：rc=$erc，混淆矩阵核对 rc=$crc（0 = 逐格一致）"
  [[ $erc -eq 0 && $crc -eq 0 ]] || fail=1
done

# ---------------------------------------------------------------- 5. 验收 (a)
refs=(); protos=()
for s in 0 1 2; do refs+=(--ref "s$s=$REF_ROOT/seed$s/run"); done
for S in $SEEDS; do [[ -f "$OUT/seed$S/run/DONE" ]] && protos+=(--proto "p$S=$OUT/seed$S/run"); done
if (( ${#protos[@]} == 0 )); then finish 10 "没有训完的原型种子"; fi
"$PY" "$CODE/proto/accept_offline.py" "${refs[@]}" "${protos[@]}" --out-json "$OUT/accept.json" --out-md "$OUT/accept.md" > "$OUT/accept.log" 2>&1
arc=$?
case $arc in 0) lit=PASS ;; 1) lit=FAIL ;; *) lit=ERROR ;; esac
if (( ${#protos[@]} < 3 )); then lit=INCOMPLETE; fi
{ echo "literal=$lit"; echo "accept_rc=$arc"; echo "seeds_done=${#protos[@]}"; echo "decided_at=$(date '+%F %T')"; } > "$OUT/OFFLINE_VERDICT"
log "验收 (a) 字面读法：$lit（accept_offline.py rc=$arc，详见 $OUT/accept.md）"

# ---------------------------------------------------------------- 6. C++ 对照臂
cpp_train() {   # $1 = 种子，$2 = run 目录
  nice -n 10 "$TRAIN_CPP" --data "$DATA" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --seed "$1" --class-weights auto \
      --drop-collision --save-every-epoch --out "$2" > "$2.log" 2>&1
}
do_control=0
case "$RUN_CONTROL" in 1) do_control=1 ;; 0) do_control=0 ;; *) [[ "$lit" == FAIL ]] && do_control=1 ;; esac
CTRL=$OUT/cpp-control
mkdir -p "$CTRL"
cg_mark start control
RERUN=$CTRL/seed$RERUN_SEED-rerun/run
if fresh_dir "$RERUN"; then
  wait_slot
  ( cpp_train "$RERUN_SEED" "$RERUN" && echo "rc=0 $(date '+%F %T')" > "$RERUN/DONE" ) &
  PIDS[cpp-rerun]=$!
  log "C++ 种子 $RERUN_SEED 重跑起训 pid=${PIDS[cpp-rerun]}"
fi
if (( do_control )); then
  log "字面判据没过（或强制），起额外 C++ 种子 [$CONTROL_SEEDS]"
  for S in $CONTROL_SEEDS; do
    CR=$CTRL/seed$S/run
    fresh_dir "$CR" || continue
    wait_slot
    ( cpp_train "$S" "$CR" && echo "rc=0 $(date '+%F %T')" > "$CR/DONE" ) &
    PIDS[cpp-$S]=$!
    log "C++ 种子 $S 起训 pid=${PIDS[cpp-$S]}"
  done
fi
reap
cg_mark end control
log "C++ 对照臂结束（$(cg_brief)）"
for name in "${!RCS[@]}"; do [[ $name == cpp-* ]] && log "$name rc=${RCS[$name]}"; done
# 重跑与 e1-noise seed0 逐步 loss 比对（标准库 python3，只读 metrics.jsonl）
if [[ -f "$RERUN/DONE" ]]; then
  python3 - "$REF_ROOT/seed$RERUN_SEED/run/metrics.jsonl" "$RERUN/metrics.jsonl" > "$CTRL/rerun_vs_ref.txt" 2>&1 <<'PY'
import json, sys
def steps(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if '"train_step"' in l]
a, b = steps(sys.argv[1]), steps(sys.argv[2])
n = min(len(a), len(b))
same = sum(1 for x, y in zip(a, b) if x["loss"] == y["loss"] and x["action_accuracy"] == y["action_accuracy"])
first = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if x["loss"] != y["loss"]), None)
mx = max((abs(x["loss"] - y["loss"]) for x, y in zip(a, b)), default=0.0)
print(f"steps ref={len(a)} rerun={len(b)}; identical loss+acc {same}/{n}; first differing step {first}; max |dloss| {mx:.3e}")
PY
  log "C++ 重跑 vs e1-noise seed$RERUN_SEED：$(cat "$CTRL/rerun_vs_ref.txt")"
fi
if (( do_control )); then
  ctrl=()
  for S in $CONTROL_SEEDS; do [[ -f "$CTRL/seed$S/run/DONE" ]] && ctrl+=(--control "c$S=$CTRL/seed$S/run"); done
  if (( ${#ctrl[@]} > 0 )); then
    "$PY" "$CODE/proto/accept_offline.py" "${refs[@]}" "${protos[@]}" "${ctrl[@]}" --out-json "$OUT/accept_with_control.json" --out-md "$OUT/accept_with_control.md" > "$OUT/accept_with_control.log" 2>&1
    log "带 C++ 对照的对拍表：$OUT/accept_with_control.md（rc=$?）"
  fi
fi

cg_mark end all
(( fail )) && finish 10 "有训练或读回核对失败，读 $OUT/seed*/train.log、$OUT/seed*/crosscheck.log"
finish 0 "完成：$OUT/accept.md；验收 (a) 字面读法 $lit"
