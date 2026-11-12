#!/usr/bin/env bash
# 卡 E2E-1 第 3 步，训练池一侧：重建核对 → 航向码重打包 → 训练 A1 / A2 / A3（各 3 个训练种子）→ 末 10 轮平均 + 校准 → 与 eval-only 对拍 → 离线指标与探针。
# 评测一侧（Gazebo 池）是 run.sh，要等本脚本写出 PREP_DONE（rc=0）才会起跑。预注册的三个比较与判据写在 run.sh 的文件头，本脚本不改动任何一条。
#
# 数据与臂（卡 E2E-1，不重采）：demo = $D/tmp/agent6-distill/trunc/col（300 集，现有数据集 …/trunc/ds/dataset.uavr）；
#   dagger = DAgger 第 3 轮聚合集（592 集、20,291 样本，现有数据集 $D/exp-dagger-gz-20261109/full-1109/round_3/ds/dataset.uavr）。
#   A0 img/demo：现成权重 $SWA/cal_swa_s{0,1,2}_k10.bin，不重训；A1 code/demo、A2 code/dagger：重打包 --cond heading-code，训练带 --prev-action；
#   A3 img/dagger：现有 dagger 数据集，图像输入，新训 3 个种子。配方与 A0 的 config.json 逐项相同（30 轮、batch 128、lr 0.001、类权重 auto、--drop-collision），加 --save-every-epoch。
#   交付权重 = 末 10 轮平均 + FORWARD 先验校准（tools/average_ckpt.py；A1、A2 用卡 E2E-1 第二步的扩展，条件列取自数据集侧车）。
#
# 阶段（每个阶段完成写 $OUT/stage/<名>.done，重起同一条命令从没完成的阶段续跑）：
#   preflight  GPU 在、二进制在、A0 权重与 demo 数据集的 sha256 与 W1-5 记录一致、磁盘够
#   pack       先不带新开关重打两份默认包，与现有数据集逐项对账（compare_packs.py）；对不上 rc=20 停，不往下；
#              通过后重打 --cond heading-code 的两份，与现有数据集对账并核对侧车（prev_index 同、prev_action 只含 0/2/3），
#              再用原始 actions.jsonl 独立重算码，与侧车逐样本比对（check_cond_codes.py，不走 preprocess 的累计逻辑）
#   train      拿训练池锁 train.lock（阻塞等待），A1、A2、A3 依次各 3 个种子同时跑（同时 ≤ 3 个训练进程），训练结束就释放锁
#   recipe     12 个训练 run（A0 现成 3 个 + 新训 9 个）的 config.json 与 A0 的训练配置逐键比较（check_recipe.py）；不一致 rc=34 停
#   calibrate  9 个新 run 各做末 10 轮平均 + 校准
#   verify     12 份交付权重（含 A0 的 3 份）各对 uavrgb_train --eval-only 对拍；真实条件一路必须相同，否则 rc=32 停
#   offline    训练曲线、交付权重的 val 指标、反事实探针（offline_report.py）
# 恢复：同一条命令重起；半成品（run、包）挪到 <名>.partial-<时刻>，不删。停：kill -- -$(cat $OUT/prepare.pid)（只按进程组，不用 pkill -f）。
# 进度：tail -f $OUT/prepare.log。结束写 $OUT/PREP_DONE（rc、时间、原因）。
# rc：0 全部完成；3 已有实例；4 冻结代码与提交不一致；12 GPU 不在；20 数据集重建与现有不一致；21 输入哈希与记录不一致；22 输入缺失；23 磁盘不足；30 训练失败；31 校准失败；
#     32 对拍不一致；33 离线报告失败；34 训练配方与 A0 不一致；130 被信号终止。
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$HERE/lib.sh"
MIN_FREE_GB=${MIN_FREE_GB:-8}
STAGE=$OUT/stage

mkdir -p "$OUT" "$STAGE" "$OUT/bin" "$OUT/data" "$OUT/recon" "$OUT/runs" "$OUT/ckpt" "$OUT/offline"
exec 7> "$OUT/.prepare.lock"
flock -n 7 || { echo "已有实例在跑（$OUT/.prepare.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/prepare.log" 2>&1
rm -f "$OUT/PREP_DONE"
echo $$ > "$OUT/prepare.pid"

finish() {   # $1 = rc，$2 = 原因
  trap - TERM INT
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/PREP_DONE"
  log "结束 rc=$1 $2"
  exit "$1"
}
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；同一条命令重起可续跑"' TERM INT

CODE_HEAD=$(cat "$OUT/code_head.txt" 2>/dev/null || echo unknown)
log "开始 PGID=$$ CODE_HEAD=$CODE_HEAD CODE=$CODE OUT=$OUT"
stage_done() { [[ -f "$STAGE/$1.done" ]]; }
stage_mark() { echo "$(date '+%F %T') $2" > "$STAGE/$1.done"; log "阶段 $1 完成：$2"; }

# ---------------------------------------------------------------- preflight
if ! stage_done preflight; then
  if [[ "$SELFCHECK" == 1 ]]; then
    verify_frozen || finish 4 "冻结副本与 CODE_HEAD $CODE_HEAD 不一致：$FROZEN_BAD（先运行 freeze.sh，再从 \$OUT/code 里启动）"
  fi
  gpu_list=$("$NVIDIA_SMI" -L 2>&1)
  grep -qF "$GPU" <<< "$gpu_list" || finish 12 "配置的 GPU $GPU 不在 nvidia-smi -L 的列表里"
  for f in "$TRAIN_SRC" "$INFER" "$DEMO_DS" "$DAGGER_DS" "$SWA/cal_swa_s0_k10.bin" "$SWA/cal_swa_s1_k10.bin" "$SWA/cal_swa_s2_k10.bin" \
           "$CODE/tools/preprocess.py" "$CODE/tools/average_ckpt.py" "$CODE/sim/tools/nav_cond.py"; do
    [[ -f "$f" ]] || finish 22 "输入缺失：$f"
  done
  if [[ -f "$TRAIN" ]]; then
    [[ "$(sha_of "$TRAIN")" == "$(sha_of "$TRAIN_SRC")" ]] || finish 21 "$TRAIN 与 $TRAIN_SRC 不是同一个文件"
  else
    cp "$TRAIN_SRC" "$TRAIN" && chmod +x "$TRAIN"
  fi
  if [[ "$SKIP_SHA_CHECK" != 1 ]]; then
    s=0
    for want in "${A0_SHA_EXPECT[@]}"; do
      got=$(sha_of "$SWA/cal_swa_s${s}_k10.bin")
      [[ "$got" == "$want" ]] || finish 21 "A0 权重 cal_swa_s${s}_k10.bin 的 sha256 $got 与 W1-5 记录 $want 不一致"
      s=$((s + 1))
    done
    got=$(sha_of "$DEMO_DS")
    [[ "$got" == "$DEMO_SHA_EXPECT" ]] || finish 21 "demo 现有数据集的 sha256 $got 与 A0 训练契约里的 $DEMO_SHA_EXPECT 不一致"
  fi
  free_gb=$(df --output=avail -BG "$OUT" | tail -n1 | tr -dc 0-9)
  (( free_gb >= MIN_FREE_GB )) || finish 23 "数据盘剩余 ${free_gb} GB < ${MIN_FREE_GB} GB"
  {
    echo "code_head $CODE_HEAD"
    echo "gpu $(grep -F "$GPU" <<< "$gpu_list" | head -n1)"
    sha256sum "$TRAIN" "$TRAIN_SRC" "$INFER" "$DEMO_DS" "$DAGGER_DS" "$SWA"/cal_swa_s{0,1,2}_k10.bin
    sha256sum "$CODE/tools/preprocess.py" "$CODE/tools/average_ckpt.py" "$CODE/sim/tools/nav_cond.py" "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py"
    echo "free_gb_at_start $free_gb"
  } > "$OUT/inputs-prepare.txt" 2>&1
  stage_mark preflight "A0 权重三份与 demo 数据集的 sha256 与记录一致；磁盘剩余 ${free_gb} GB"
fi

# ---------------------------------------------------------------- pack
pack_one() {   # $1 = demo|dagger，$2 = default|hc，$3 = 输出目录（里面写 dataset.uavr 与 dataset.aux）
  local name=$1 mode=$2 out=$3 dirs=() flags=(--drop-collision)
  if [[ -d "$out" ]]; then mv "$out" "$out.partial-$(date +%H%M%S)"; fi
  if [[ $name == demo ]]; then
    mapfile -t dirs < <(ls -d "$DEMO_COL"/train/*/ "$DEMO_COL"/val/*/ "$DEMO_COL"/test/*/)
  else
    mapfile -t dirs < "$OUT/dagger_dirs.txt"
    flags+=(--allow-mixed-dynamics)
  fi
  [[ $mode == hc ]] && flags+=(--cond heading-code)
  mkdir -p "$(dirname "$out")"
  log "打包 $name/$mode：${#dirs[@]} 个集目录 → $out（preprocess.py ${flags[*]}）"
  ( cd "$(dirname "$out")" && nice -n 10 "$PY" "$CODE/tools/preprocess.py" --out "$(basename "$out")" "${flags[@]}" "${dirs[@]}" ) > "$out.log" 2>&1
}
compare_one() {   # $1 = 名字，其余参数原样交给 compare_packs.py；输出写 $OUT/recon/compare-<名字>.log，rc 非 0 停
  local label=$1; shift
  "$PY" "$ENTRY/compare_packs.py" "$@" > "$OUT/recon/compare-$label.log" 2>&1
  local rc=$?
  log "核对 $label rc=$rc：$(grep -E '^RESULT' "$OUT/recon/compare-$label.log" | head -n1)"
  (( rc == 0 )) || finish 20 "数据集重建核对 $label 不一致，见 $OUT/recon/compare-$label.log；按卡要求停下，不往下做"
}
check_codes() {   # $1 = 名字，$2 = 航向码数据集：与原始 actions.jsonl 独立重算的码逐样本比对；rc 非 0 停
  "$PY" "$ENTRY/check_cond_codes.py" "$2" --json "$OUT/recon/codes-$1.json" > "$OUT/recon/codes-$1.log" 2>&1
  local rc=$?
  log "码独立重算 $1 rc=$rc：$(grep -E '^RESULT' "$OUT/recon/codes-$1.log" | head -n1)"
  (( rc == 0 )) || finish 20 "航向码与原始动作独立重算的结果不一致（$1），见 $OUT/recon/codes-$1.log；按卡要求停下"
}
drop_default_pack() {   # 默认包核对通过且哈希记录后，删掉自己生成的 dataset.uavr（磁盘预算 ≤ 3 GB）；侧车、日志、哈希留着
  sha256sum "$1/dataset.uavr" >> "$OUT/recon/SHA256SUMS"
  rm -f "$1/dataset.uavr"
}
if ! stage_done pack; then
  "$PY" "$ENTRY/make_dagger_dirs.py" "$OUT/dagger_dirs.txt" || finish 22 "生成 dagger 集目录清单失败"
  [[ "$(wc -l < "$OUT/dagger_dirs.txt")" == 592 ]] || finish 20 "dagger 集目录清单不是 592 项（$(wc -l < "$OUT/dagger_dirs.txt")）：与卡的 592 集不符"
  for name in demo dagger; do
    orig=$DEMO_DS; [[ $name == dagger ]] && orig=$DAGGER_DS
    pack_one "$name" default "$OUT/recon/$name-default" || finish 20 "默认参数重打 $name 失败，见 $OUT/recon/$name-default.log"
    compare_one "$name-default" "$orig" "$OUT/recon/$name-default/dataset.uavr"
    drop_default_pack "$OUT/recon/$name-default"
  done
  for name in demo dagger; do
    orig=$DEMO_DS; [[ $name == dagger ]] && orig=$DAGGER_DS
    pack_one "$name" hc "$OUT/data/$name-hc" || finish 20 "--cond heading-code 重打 $name 失败，见 $OUT/data/$name-hc.log"
    compare_one "$name-hc" "$orig" "$OUT/data/$name-hc/dataset.uavr" --cond-pack --aux-ref "$OUT/recon/$name-default/dataset.aux"
    check_codes "$name-hc" "$OUT/data/$name-hc/dataset.uavr"
  done
  stage_mark pack "demo 与 dagger 的默认重打包与现有数据集一致；航向码包两份已打好并通过核对（含与原始动作独立重算的码比对）；默认包的 dataset.uavr 已删（哈希在 recon/SHA256SUMS）"
fi

# ---------------------------------------------------------------- train
run_complete() { [[ -f "$1.done" ]] && grep -q '^rc=0 ' "$1.done" && [[ -f "$1/ckpt_ep30.bin" && -f "$1/config.json" ]]; }
run_group() {   # $1 = 臂：3 个训练种子同时跑
  local a=$1 s run data extra pids=() seeds_run=()
  data=$(arm_data "$a")
  extra=""; arm_is_code "$a" && extra="--prev-action"
  for s in $SEEDS_TRAIN; do
    run=$(arm_run "$a" "$s")
    if run_complete "$run"; then log "$(arm_name "$a") s$s 已训完，跳过"; continue; fi
    if [[ -d "$run" ]]; then
      ts=$(date +%H%M%S); mv "$run" "$run.partial-$ts"
      for ext in log done; do [[ -f "$run.$ext" ]] && mv "$run.$ext" "$run.partial-$ts.$ext"; done
    fi
    log "$(arm_name "$a") s$s 开始训练（$(cg_brief)）"
    (
      export UAVRGB_SOURCE_GIT=$CODE_HEAD
      # shellcheck disable=SC2086
      "$TRAIN" --data "$data" --out "$run" $TRAIN_FLAGS --seed "$s" $extra > "$run.log" 2>&1
      echo "rc=$? $(date '+%F %T')" > "$run.done"
    ) &
    pids+=($!); seeds_run+=("$s")
  done
  (( ${#pids[@]} == 0 )) || wait "${pids[@]}"
  for s in "${seeds_run[@]}"; do
    run=$(arm_run "$a" "$s")
    run_complete "$run" || finish 30 "$(arm_name "$a") s$s 训练失败：$(cat "$run.done" 2>/dev/null)，见 $run.log"
    log "$(arm_name "$a") s$s 训练完成：$(tail -n 2 "$run.log" | tr '\n' ' ' | cut -c1-200)"
  done
}
if ! stage_done train; then
  acquire_lock 8 "$TRAIN_LOCK" "训练池"
  cg_mark start train
  for a in a1 a2 a3; do run_group "$a"; done
  cg_mark end train
  log "训练结束（$(cg_brief)），释放训练池锁"
  exec 8>&-
  stage_mark train "A1、A2、A3 各 3 个种子训练完成"
fi

# ---------------------------------------------------------------- recipe
if ! stage_done recipe; then
  specs=()
  for s in $SEEDS_TRAIN; do specs+=("$(arm_name a0)-s$s=$(arm_run a0 "$s"):img:$s:demo"); done
  for a in a1 a2 a3; do
    kind=img; arm_is_code "$a" && kind=code
    ds=dagger; [[ $a == a1 ]] && ds=demo
    for s in $SEEDS_TRAIN; do specs+=("$(arm_name "$a")-s$s=$(arm_run "$a" "$s"):$kind:$s:$ds"); done
  done
  "$PY" "$ENTRY/check_recipe.py" --ref "$A0_RUN_DIR/seed0/run/config.json" "${specs[@]}" > "$OUT/recon/recipe.log" 2>&1
  rc=$?
  log "训练配方核对 rc=$rc：$(grep -E '^RESULT' "$OUT/recon/recipe.log" | head -n1)"
  (( rc == 0 )) || finish 34 "训练配方与 A0 不一致，见 $OUT/recon/recipe.log；按卡要求停下，不往下做"
  stage_mark recipe "12 个训练 run 的 config.json 与 A0 逐键一致（允许的差异：数据集、种子、prev_action、source_git）"
fi

# ---------------------------------------------------------------- calibrate
if ! stage_done calibrate; then
  for a in a1 a2 a3; do
    for s in $SEEDS_TRAIN; do
      out=$(arm_ckpt "$a" "$s")
      if [[ -f "$out" && -f "$out.json" ]]; then log "$(arm_name "$a") s$s 已有交付权重，跳过"; continue; fi
      log "$(arm_name "$a") s$s 平均 + 校准 → $out"
      nice -n 10 "$PY" "$CODE/tools/average_ckpt.py" --run "$(arm_run "$a" "$s")" --data "$(arm_data "$a")" --out "$out" --infer-bin "$INFER" --last-k "$LAST_K" \
        > "$out.log" 2>&1 || finish 31 "$(arm_name "$a") s$s 校准失败，见 $out.log"
      log "$(tail -n 2 "$out.log" | tr '\n' ' ' | cut -c1-260)"
    done
  done
  stage_mark calibrate "9 份新权重已平均 + 校准"
fi

# ---------------------------------------------------------------- verify
if ! stage_done verify; then
  for a in $ARMS; do
    for s in $SEEDS_TRAIN; do
      vdir=$OUT/offline/verify/$(arm_name "$a")-s$s
      mkdir -p "$vdir"
      nice -n 10 "$PY" "$ENTRY/verify_infer_vs_evalonly.py" --ckpt "$(arm_ckpt "$a" "$s")" --data "$(arm_data "$a")" --train-bin "$TRAIN" --infer-bin "$INFER" \
        --out "$vdir" --tools-dir "$CODE/tools" > "$vdir.log" 2>&1
      rc=$?
      log "对拍 $(arm_name "$a") s$s rc=$rc：$(grep -E '^RESULT' "$vdir.log" | head -n1)"
      (( rc == 0 )) || finish 32 "$(arm_name "$a") s$s 的逐帧推理与 eval-only 不一致，见 $vdir.log"
    done
  done
  stage_mark verify "12 份交付权重的推理与 eval-only 一致"
fi

# ---------------------------------------------------------------- offline
if ! stage_done offline; then
  nice -n 10 "$PY" "$ENTRY/offline_report.py" --out-root "$OUT" --tools-dir "$CODE/tools" --infer-bin "$INFER" > "$OUT/offline/offline_report.log" 2>&1 \
    || finish 33 "离线报告失败，见 $OUT/offline/offline_report.log"
  stage_mark offline "曲线、val 指标、探针已写入 $OUT/offline/"
fi

finish 0 "训练池一侧全部完成：9 个新权重 + 12 份对拍 + 离线报告；可以起 run.sh"
