#!/usr/bin/env bash
# 三个视觉对照并发，与已有语言基线使用相同数据和训练预算。
set -euo pipefail
cd "$(dirname "$0")/../.."
source env.sh
source tools/exp_lib.sh
OUT="$DATA_ROOT/uav_rgb/exp-l1-vision-parallel-20261114"
BASE="$DATA_ROOT/uav_rgb/exp-l1-language-formal-20261114"
DATA="$DATA_ROOT/uav_rgb/l1-v4-cf/ds/dataset.uavr"
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9 || exit 3
# 不覆盖已经运行过的实验。
[[ ! -e "$OUT/source_commit" ]] || exit 4
exec >>"$OUT/run.log" 2>&1
trap 'rc=$?; echo "rc=$rc" > "$OUT/DONE"' EXIT
printf 'pid=%s\npgid=%s\nstart=%s\ncmd=%s\n' "$$" "$(exp_stat $$ 5)" "$(exp_stat $$ 22)" "$(exp_cmdline $$)" > "$OUT/run.pid"
export OMP_NUM_THREADS=4
cp "$BASE/uavrgb_train" "$OUT/uavrgb_train"
sha256sum "$DATA" "$OUT/uavrgb_train" > "$OUT/inputs.sha256"
git rev-parse HEAD > "$OUT/source_commit"
pids=()
for seed in 1234 1235 1236; do
  (
    set +e
    "$OUT/uavrgb_train" --data "$DATA" --vision-only --epochs 20 --batch 64 --lr 0.0005 --seed "$seed" --max-grad-norm 5 --drop-collision --out "$OUT/seed$seed" > "$OUT/seed$seed.log" 2>&1
    rc=$?
    echo "rc=$rc" > "$OUT/seed$seed.exit"
    exit "$rc"
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
