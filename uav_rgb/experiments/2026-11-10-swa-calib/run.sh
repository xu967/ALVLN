#!/usr/bin/env bash
# 闭环对照：现行 ckpt_best 对「末 10 轮权重平均 + 类先验校准」（零代码改动，只换推理权重）。
# 背景：src 审查（$D/tmp/cursor-1110/agents/src/REPORT.md B.4–B.6）离线发现恒定 lr 的 Adam 在末期来回过冲，
#       「FWD 对 YAW」的类先验每轮摆 0.55–0.65 nat；权重平均 + 校准把跨种子预测分歧率从 22–29% 压到 5–7%。闭环没测过，这里测。
# 运行：cd /root/rivermind-data/uav_rgb/exp-swa-calib-20261110 && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#       （日志由脚本自己追加进 run.log；.lock 互斥，重复启动的那个立即退出，不会两个实例写同一批目录）
# 进度：tail -f run.log；每臂一个 <臂>.log 与 <臂>/eval.json；汇总随时可跑：python summarize.py
# 结束：落 DONE（rc + 时间）与 summary.md。停：kill -- -$(cat run.pid)。
# 续跑：同一条命令重起；20 集评完且无失败集的臂跳过，半成品（含 SIGTERM 写出的 partial eval.json）挪到 <臂>.partial-<时刻> 后重跑。
#
# 预注册（2026-11-10 17:20，起跑前写下）：
#   判据都用走廊口径 sim09.v2.coast1s.corridor3m 的 success，三个训练种子合并成 60 个配对（训练种子, 评测 seed）。
#   H1 calswa 对 best：走廊内成功数不低于 best（配对 McNemar 的 b <= c），且「越界 + 超时」之和不增加。
#   H2 一致性：calswa 三个训练种子的走廊内成功数极差 <= best 的极差。
#   证伪：calswa 的走廊内成功数比 best 少 5 集以上（60 对里），说明把先验推向 FORWARD 在闭环里有害，
#         校准这条路停掉，只保留「权重停稳」（余弦退火 / 权重平均）并改用别的办法定先验。
#   calbest（只校准）与 swa（只平均）两组用来拆开「先验」与「停稳」各自的作用，排在后面跑。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-swa-calib-20261110
WT=$D/tmp/cursor-1110/wt                                 # 冻结代码：main c85a1ff + 14:48 的走廊判据补丁（全量 check.sh 已过）
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与 eval4 同一个推理二进制
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
STEPS=80                                                 # 同 eval4；到成功线即停
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
LANES=3                                                  # 同时 3 个 gz；另一会话的 DAgger eval60 占 1 个，合计 4（历史上限）
BASE=$D/exp-student-select-20261109/e1-noise             # 三个训练种子的原始 run
CAND=$D/tmp/cursor-1110/agents/src/ckpt                  # src 审查生成的候选权重（起跑时拷进 $OUT/ckpt，之后不再依赖草稿目录）
# 臂：名字|权重。顺序即优先级，前 6 个是预注册的主对照。
ARMS=(
  "best-s0|$BASE/seed0/run/ckpt_best.bin"    "calswa-s0|$OUT/ckpt/cal_swa_s0_k10.bin"
  "best-s1|$BASE/seed1/run/ckpt_best.bin"    "calswa-s1|$OUT/ckpt/cal_swa_s1_k10.bin"
  "best-s2|$BASE/seed2/run/ckpt_best.bin"    "calswa-s2|$OUT/ckpt/cal_swa_s2_k10.bin"
  "calbest-s0|$OUT/ckpt/cal_s0_best.bin"     "calbest-s1|$OUT/ckpt/cal_s1_best.bin"   "calbest-s2|$OUT/ckpt/cal_s2_best.bin"
  "swa-s0|$OUT/ckpt/swa_s0_k10.bin"          "swa-s1|$OUT/ckpt/swa_s1_k10.bin"        "swa-s2|$OUT/ckpt/swa_s2_k10.bin"
)
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.6 GB，临时文件放数据盘
export OUT WT INFER SEEDS STEPS CORRIDOR PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" "$OUT/ckpt" /tmp/xdg && chmod 700 /tmp/xdg
# 互斥：2026-11-10 17:18 这个脚本被启动了两次，两个实例往同一批目录写，数据作废（discarded-double-launch/）。
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES"

# 1) 候选权重入档 + 记 sha256
for f in cal_swa_s{0,1,2}_k10.bin cal_s{0,1,2}_best.bin swa_s{0,1,2}_k10.bin; do
  [[ -f "$OUT/ckpt/$f" ]] || cp "$CAND/$f" "$OUT/ckpt/$f" || { echo "缺候选权重 $f"; echo "rc=2 $(date '+%F %T') 缺权重" > "$OUT/DONE"; exit 2; }
done
{ sha256sum "$OUT"/ckpt/*.bin "$BASE"/seed{0,1,2}/run/ckpt_best.bin "$INFER"; git -C "$WT" rev-parse HEAD; git -C "$WT" diff --cached --stat | tail -1; } > "$OUT/inputs.txt" 2>&1

# 2) 逐臂闭环（每臂 20 集串行，臂之间 LANES 路并行）
run_arm() {
  local name=$1 ckpt=$2
  [[ -n "$name" && -f "$ckpt" ]] || { echo "$(date '+%F %T') 臂 '$name' 的权重不存在：$ckpt"; return 0; }
  # 评完 = eval.json 不是 partial、20 集都完成、没有失败集（与 tools/pipeline_common.eval_finished 同口径）
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done")==20 and e.get("failed_count")==0) else 1)' "$OUT/$name/eval.json" 2>/dev/null; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  # 上次中断留下的半成品挪开留证据，不删
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"
  echo "$(date '+%F %T') $name 开始"
  "$PY" "$WT/sim/tools/eval_closed_loop.py" --policy student --ckpt "$ckpt" --infer-bin "$INFER" \
      --task base --split test --seeds "$SEEDS" --episodes 20 --steps "$STEPS" \
      --sim-timeout 120.0 --episode-timeout 1500.0 --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') $name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-60)"
  return 0                                   # 单臂失败不停整个队列，汇总里按 failed 标出来
}
export -f run_arm
printf '%s\n' "${ARMS[@]}" | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n c <<< "{}"; run_arm "$n" "$c"'

# 3) 汇总
"$PY" "$OUT/summarize.py" > "$OUT/summarize.log" 2>&1
rc=$?
echo "rc=$rc $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束 rc=$rc"
