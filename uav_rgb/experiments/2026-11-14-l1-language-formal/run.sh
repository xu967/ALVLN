#!/usr/bin/env bash
# 已有 L1 模板语言基线复训；不宣称为 3DGS 开放语言学生。
set -euo pipefail
cd "$(dirname "$0")/../.."
source env.sh
OUT="$DATA_ROOT/uav_rgb/exp-l1-language-formal-20261114"
DATA="$DATA_ROOT/uav_rgb/l1-v4-cf/ds/dataset.uavr"
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 3
exec >>"$OUT/run.log" 2>&1
trap 'rc=$?; echo "rc=$rc" > "$OUT/DONE"' EXIT
source tools/exp_lib.sh
printf 'pid=%s\npgid=%s\nstart=%s\ncmd=%s\n' "$$" "$(exp_stat $$ 5)" "$(exp_stat $$ 22)" "$(exp_cmdline $$)" > "$OUT/run.pid"
export CUDA_VISIBLE_DEVICES=GPU-4910bb04-2d00-ca81-f64a-2031568758ad
export OMP_NUM_THREADS=4
sha256sum "$DATA" "$(dirname "$DATA")/instructions.jsonl" build/check/uavrgb_train > "$OUT/inputs.sha256"
git rev-parse HEAD > "$OUT/source_commit"
cp build/check/uavrgb_train "$OUT/uavrgb_train"
for seed in 1234 1235 1236; do
  "$OUT/uavrgb_train" --data "$DATA" --text-dim 16 --epochs 20 --batch 64 --lr 0.0005 --seed "$seed" --max-grad-norm 5 --drop-collision --out "$OUT/seed$seed"
done
