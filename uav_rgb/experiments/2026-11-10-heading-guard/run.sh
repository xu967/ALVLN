#!/usr/bin/env bash
# R1 判别实验：不重训，评测端加「航向护栏」，看「学生缺航向」是不是出阵的主因。
# 规则：n = 已执行的 YAW_LEFT 数减 YAW_RIGHT 数（机体自己数得出来，不是特权信息）；|n| >= 2 时强制朝回转一步。
# 依据：data-algo 审查（$D/tmp/cursor-1110/agents/data-algo/REPORT.md B.2、D1）：学生闭环 47.5–54.7% 的步视锥内没有柱子，
#       这些步上规则「|航向|>25° 就朝回转」与影子教师一致 94.2%；54 个出过阵的集里 53 个在出阵前 |n|>=2 已成立，中位提前 9 步。
# 代码：$D/tmp/cursor-1110/wt-r1 = main c85a1ff + 走廊判据补丁 + 航向护栏补丁（$D/tmp/cursor-1110/r1-heading-guard.patch，
#       3 个文件 +55 −5，--heading-guard 默认 0 时行为逐位不变；10 组相关测试通过）。
# 运行：cd /root/rivermind-data/uav_rgb/exp-heading-guard-20261110 && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f run.log；汇总随时可跑：python summarize.py。结束落 DONE 与 summary.md。停：kill -- -$(cat run.pid)。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
#
# 预注册（2026-11-10 17:30，起跑前写下；对照臂是 exp-swa-calib-20261110 的 best-s0/1/2：同 seed、同口径、同时段、护栏关）：
#   预测：guard2 三个训练种子合并 60 集，「越界」集数比 best 少一半以上，走廊内成功数高于 best（配对 McNemar 的 b > c），
#         每个训练种子的走廊内成功数不低于 14/20（data-algo 的估计值）。
#   证伪：guard2 的走廊内成功数不高于 best（b <= c）。那样「缺航向」不是出阵的主因，航向码输入这条线不做。
#   副作用要看：护栏把机头按回去以后撞柱数是否上升；每集 L<->R 相邻换向是否上升（学生想转、护栏按回的拉锯）。
#   次要臂：护栏 + 权重平均校准（guard2-calswa），护栏 + more_demos 权重（guard2-moredemos），排在主臂后面。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-heading-guard-20261110
WT=$D/tmp/cursor-1110/wt-r1                              # 冻结代码（见文件头）
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与 eval4 同一个推理二进制
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
STEPS=80                                                 # 同 eval4；到成功线即停
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
GUARD=2                                                  # |n| >= 2 即朝回转：教师臂只有 2.7% 的步 |n|>=2，阵内成功集 4.3%
LANES=3                                                  # 同时 3 个 gz；加上 exp-swa-calib 的 3 个与 DAgger eval60 的 1 个共 7 个（24 核）
BASE=$D/exp-student-select-20261109/e1-noise             # 三个训练种子的原始 run
SWA=$D/exp-swa-calib-20261110/ckpt                       # 权重平均 + 校准的候选权重（那个实验起跑时已入档）
MD=$D/exp-dagger-gz-20261109/full-1109/more_demos/run/ckpt_best.bin   # 多采教师示范的对照臂权重（走廊内 14/20）
# 臂：名字|权重。顺序即优先级，前 3 个是预注册的主臂。
ARMS=(
  "guard2-s0|$BASE/seed0/run/ckpt_best.bin"  "guard2-s1|$BASE/seed1/run/ckpt_best.bin"  "guard2-s2|$BASE/seed2/run/ckpt_best.bin"
  "guard2-calswa-s0|$SWA/cal_swa_s0_k10.bin" "guard2-calswa-s1|$SWA/cal_swa_s1_k10.bin" "guard2-calswa-s2|$SWA/cal_swa_s2_k10.bin"
  "guard2-moredemos|$MD"
)
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.6 GB，临时文件放数据盘
export OUT WT INFER SEEDS STEPS CORRIDOR GUARD PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES guard=$GUARD"
{ for a in "${ARMS[@]}"; do sha256sum "${a#*|}"; done; sha256sum "$INFER"; git -C "$WT" rev-parse HEAD; git -C "$WT" diff --stat | tail -1; } > "$OUT/inputs.txt" 2>&1

run_arm() {
  local name=$1 ckpt=$2
  [[ -n "$name" && -f "$ckpt" ]] || { echo "$(date '+%F %T') 臂 '$name' 的权重不存在：$ckpt"; return 0; }
  # 评完 = eval.json 不是 partial、20 集都完成、没有失败集（与 tools/pipeline_common.eval_finished 同口径）
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done")==20 and e.get("failed_count")==0) else 1)' "$OUT/$name/eval.json" 2>/dev/null; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"   # 半成品挪开留证据，不删
  echo "$(date '+%F %T') $name 开始"
  "$PY" "$WT/sim/tools/eval_closed_loop.py" --policy student --ckpt "$ckpt" --infer-bin "$INFER" \
      --task base --split test --seeds "$SEEDS" --episodes 20 --steps "$STEPS" \
      --sim-timeout 120.0 --episode-timeout 1500.0 --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" --heading-guard "$GUARD" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "$(date '+%F %T') $name rc=$? $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-60)"
  return 0                                   # 单臂失败不停整个队列，汇总里按 failed 标出来
}
export -f run_arm
printf '%s\n' "${ARMS[@]}" | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n c <<< "{}"; run_arm "$n" "$c"'

"$PY" "$OUT/summarize.py" > "$OUT/summarize.log" 2>&1
rc=$?
echo "rc=$rc $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束 rc=$rc"
