#!/usr/bin/env bash
# 长课程第二轮的时序标定：教师策略，K=3 / K=1 世界 × 不同配置，每种 2 集（seed 1、113），2 条车道。
# 为什么：第一轮 K=3 世界每步动作生效时长 0.77–0.79 s，今天下午 K=1 是 0.871 s（--window-from observation 下动作窗口从观测帧起算，
#         观测到派发的延迟越长，动作生效越短）；这里量哪种配置能把 K=3 的生效时长拉回 K=1 水平（0.854–0.888）。只用教师，不碰学生臂。
# 配置：前 4 个是任务书指定的（去辅助相机 / 保留，K=3 / K=1），用第一轮冻结的代码；k3-noaux-cam5 是任务书的第二个手段（降相机频率）；
#       k3-cache / k1-cache 用第二轮冻结的代码（teacher.astar_path 加了栅格缓存，路径逐位不变）。结论与数字见报告「更新 2026-11-10 21:xx」。
# 运行：setsid nohup bash calibrate_timing.sh > /dev/null 2>&1 < /dev/null &        已评完的 job 跳过，可反复追加配置再起。
# 进度：tail -f $ROOT/calib/run.log。读结果：python summarize.py --timing $ROOT/calib/<配置>-s1 $ROOT/calib/<配置>-s113（打印时序健康表）。
# 停：kill -- -$(cat $ROOT/calib/run.pid)。.lock 互斥。每个配置的每个 seed 一个输出目录（独立 eval_closed_loop 进程 = 一条车道）。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
ROOT=$D/exp-long-course-v2-20261110                      # 第二轮输出目录；标定放在 $ROOT/calib
SRC=/root/claude-code-backup-20261028/code/wt/p12-longcourse   # 代码源（分支 p12-longcourse）；第二轮冻结其 HEAD
CODE1=$D/exp-long-course-20261110/code                   # 第一轮冻结的代码（A* 无缓存）
CODE2=$ROOT/code                                         # 第二轮冻结的代码（A* 有缓存）；缺就从 SRC 的 HEAD 冻结，run_v2.sh 复用同一份
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（教师不用 GPU，仍按规则钉住）
SEED_LIST=(1 113)                                        # 与任务书一致
LANES=2                                                  # 标定只占 2 条车道
# 配置：名字|代码目录|评测参数。xargs 按序派发，已评完的跳过。
CONFIGS=(
  "k3-noaux|$CODE1|--task course_base --blocks 3 --steps 240 --no-aux-cameras"                      # 任务书标定 1：K=3 去掉 chase / top 两路辅助相机
  "k1-noaux|$CODE1|--task base --steps 80 --no-aux-cameras"                                         # 任务书标定 2：K=1 去掉辅助相机（参照）
  "k3-aux|$CODE1|--task course_base --blocks 3 --steps 240"                                         # 对照：同负载下 K=3 保留辅助相机（第一轮的配置）
  "k1-aux|$CODE1|--task base --steps 80"                                                            # 对照：同负载下 K=1 保留辅助相机
  "k3-cache|$CODE2|--task course_base --blocks 3 --steps 240"                                       # 修法：A* 栅格缓存，保留辅助相机（与 K=1 参照同一渲染配置）
  "k1-cache|$CODE2|--task base --steps 80"                                                          # 参照：同一份代码的 K=1
  "k3-noaux-cam5|$CODE1|--task course_base --blocks 3 --steps 240 --no-aux-cameras --camera-rate 5" # 任务书第二个手段：在去辅助相机之上再把学生相机降到 5 Hz
)
# ----------------------------------------------------------------

OUT=$ROOT/calib
export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$ROOT/tmp
export OUT PY=$R/.venv-tools/bin/python
mkdir -p "$OUT" "$ROOT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES"

# 冻结第二轮代码：已有副本就复用；工作树里有未提交的改动（已跟踪文件）就不冻结（git archive 只带 HEAD，会悄悄丢改动）
if [[ ! -f "$CODE2/sim/tools/eval_closed_loop.py" ]]; then
  if [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]; then
    echo "$(date '+%F %T') $SRC 有未提交的改动，先提交再起跑"; exit 4
  fi
  mkdir -p "$CODE2" && git -C "$SRC" archive HEAD | tar -x -C "$CODE2" || { echo "冻结代码失败"; exit 4; }
  echo "$(date '+%F %T') 冻结第二轮代码 $(git -C "$SRC" rev-parse --short HEAD) -> $CODE2"
fi

run_job() {   # $1 = 配置名，$2 = 代码目录，$3 = 评测参数串，$4 = seed
  local name=$1 code=$2 args=$3 seed=$4 dir="$OUT/$1-s$4"
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)==1) else 1)' "$dir/eval.json" 2>/dev/null; then
    echo "$(date '+%F %T') $name s$seed 已评完，跳过"; return 0
  fi
  [[ -d "$dir" ]] && mv "$dir" "$dir.partial-$(date +%H%M%S)"
  echo "$(date '+%F %T') $name s$seed 开始"
  # shellcheck disable=SC2086
  "$PY" "$code/sim/tools/eval_closed_loop.py" --policy teacher $args --split test --seeds "$seed" --episodes 1 \
      --sim-timeout 120.0 --episode-timeout 2520 --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width 3.0 --out "$dir" > "$dir.log" 2>&1
  echo "$(date '+%F %T') $name s$seed rc=$? $(grep -E '^\[seed' "$dir.log" | cut -c1-100)"
  return 0
}
export -f run_job

for c in "${CONFIGS[@]}"; do for s in "${SEED_LIST[@]}"; do printf '%s|%s\n' "$c" "$s"; done; done \
  | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n c a s <<< "{}"; run_job "$n" "$c" "$a" "$s"'

echo "rc=0 $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束"
