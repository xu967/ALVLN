#!/usr/bin/env bash
# 卡 E2E-P0 第一关的补充：用 C++ 训练器训一个 1 轮的 prev-action 小模型，给 PyTorch 前向对拍用
# （E2E-1 的权重还没有；卡规格：带 prev-action 的模型同样对拍）。
# 数据集带 dataset.aux 侧车（prev_action 取自侧车），1 轮，其余配方同主线 e1-noise：batch 128、lr 0.001、类别权重 auto、drop-collision。
#
# 运行：setsid nohup bash proto/runs/run_cpp_prevaction.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；结束落 $OUT/DONE；停：kill -- -$(cat $OUT/run.pid)。训练排训练池的锁。

# ---------------------------------------------------------------- 配置（改这里）
D=/root/rivermind-data/uav_rgb
OUT=${OUT:-$D/exp-proto-parity-20261112/cpp-prevaction-1ep}
TRAIN=$D/tmp/build-check-main-1111/uavrgb_train          # 卡指定：只读借用，不在该目录重新 cmake
DATA=$D/exp-student-select-20261109/ds-0417-aux/dataset.uavr   # 有 dataset.aux 侧车的数据集（train 10032 / val 720 / test 1066）
EPOCHS=1                                                 # 卡规格：1 轮
BATCH=128                                                # 同主线 e1-noise
LR=0.001                                                 # 同主线 e1-noise
SEED=0                                                   # 任选；对拍只要权重是真训练产物
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740             # 卡指定的 GPU
TRAIN_LOCK=$D/tmp/phase12-cards/train.lock               # 训练池的锁
# ----------------------------------------------------------------
set -u
export LC_ALL=C CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$D/tmp/phase12-cards/e2e-p0/tmp
mkdir -p "$TMPDIR"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
daemon_begin "C++ 训练 1 轮 prev-action 小模型 → $OUT/run"
take_lock "$TRAIN_LOCK" "训练池"
cg_mark start train
log "训练开始（$(cg_brief)）"
nice -n 10 "$TRAIN" --data "$DATA" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --seed "$SEED" \
  --class-weights auto --drop-collision --prev-action --out "$OUT/run" > "$OUT/train.log" 2>&1
rc=$?
cg_mark end train
log "训练 rc=$rc（$(cg_brief)）"
[[ $rc -eq 0 && -s "$OUT/run/ckpt.bin" ]] || finish 20 "uavrgb_train rc=$rc 或没有 ckpt.bin，读 $OUT/train.log"
finish 0 "完成：$OUT/run/ckpt.bin"
