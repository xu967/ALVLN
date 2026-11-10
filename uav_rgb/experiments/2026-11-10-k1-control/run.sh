#!/usr/bin/env bash
# K=1 对照（主会话 2026-11-10 21:27 起）：长课程第二轮里学生第 1 块的通过率（约 0.5）明显低于下午 K=1 的 0.83，每步生效时长已基本恢复（0.85 对 0.87）。
# 要分清原因是「下午之后入库的改动」（新 E1 模板 2e554bcd 的保护罩外形、A* 栅格缓存）还是「后续块的柱子入画」。
# 做法：与长课程第二轮同一份冻结代码（$D/exp-long-course-v2-20261110/code，提交 fcbf2fe），柱阵 K=1，同权重同 seed 同口径，只跑一个臂。
# 对照臂：$D/exp-heading-guard-20261110/guard2-calswa-s1（下午，旧模板 21c8bc9a，17/20）。
# 预测：若今晚的 K=1 仍在 17/20 上下（逐集一致率不低于 70%），原因是后续块入画；若明显变差（不高于 12/20），原因在下午之后的改动，先查模板。
# 运行：cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &   停：kill -- -$(cat run.pid)
set -u
R=/root/claude-code-backup-20261028/code/uav_rgb
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-k1-control-20261110
CODE=$D/exp-long-course-v2-20261110/code                   # 与长课程第二轮同一份冻结代码
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654
CKPT=$D/exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin
export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp
exec 9> "$OUT/.lock"; flock -n 9 || { echo "已有实例在跑" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"; echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ code=$(cat $D/exp-long-course-v2-20261110/inputs.txt | head -1)"
"$R/.venv-tools/bin/python" "$CODE/sim/tools/eval_closed_loop.py" --policy student --ckpt "$CKPT" --infer-bin "$INFER" \
    --task base --split test --seeds "$SEEDS" --episodes 20 --steps 80 \
    --sim-timeout 120.0 --episode-timeout 1500.0 --dynamics cf_sitl --vehicle e1_mini \
    --hold 1.0 --window-from observation --corridor-half-width 3.0 --heading-guard 2 \
    --out "$OUT/guard2-calswa-s1" > "$OUT/guard2-calswa-s1.log" 2>&1
rc=$?
echo "rc=$rc $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 结束 rc=$rc $(grep -E '^requested:' $OUT/guard2-calswa-s1.log | cut -c1-80)"
