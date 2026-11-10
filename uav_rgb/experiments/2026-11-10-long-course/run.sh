#!/usr/bin/env bash
# 长课程实验（phase 12 longcourse）：把 K 个 base 柱阵块首尾相接成一条长课程，不训练，用既有权重逐块测存活率。
# 为什么：现有柱阵任务约 6.5 m、绕一根柱就过线，量不出成功率随航程怎么衰减。
# 课程：scene_course.py（--task course_base --blocks K）；第 0 块 = 同 seed 的 base 柱阵，第 k 块是同 split 的另一个 base 几何平移 5.0·k m。
#       成功线 = 6.0 + 5.0·(K−1)；逐块存活的判据与定义见 summarize.py 文件头。
# 运行：mkdir -p $OUT && cp run.sh summarize.py $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；汇总随时可跑：python $OUT/summarize.py。结束落 DONE 与 summary.md。停：kill -- -$(cat $OUT/run.pid)。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
# 代码：起跑时把 SRC 的 git HEAD 用 git archive 冻结到 $OUT/code，之后只用冻结副本（分支被合并或工作树被清理都不影响在跑的实验）。
#
# 预注册（2026-11-10，起跑前写下；K=1 参照 = 今天同权重、同 seed、同口径的 K=1 评测）：
#   K=1 参照成功率（走廊内）：calswa + 护栏 2 为 50/60（p1=0.83），calswa 无护栏 39/60（0.65），ckpt_best + 护栏 2 为 47/60（0.78）。
#   P1 calswa + 护栏 2 的 S_K 落在独立块预测 p1^K ± 0.15 内（K=3：0.57；K=2：0.69）。
#      明显低于窗口 = 存在独立块模型之外的长航程失败机制（护栏计数与真实航向的漂移、块间衔接、电池），用轨迹定位；
#      明显高于窗口 = 风险集中在第一排柱子。
#   P2 无护栏的 S_K <= p1^K + 0.15（0.65^K）且失败以越界为主；护栏对无护栏的配对 McNemar（S_K）b > c。b <= c 则 P2 证伪。
#   P3 第 1 块通过与今天 K=1 同权重同 seed 的逐集一致率 >= 70%（同权重同 seed 自己重跑约 76%）；
#      偏低时先查：新 E1 模板（护栏形状变更，sha 2e554bcd）、下一块的柱子是否进入视野。
#   教师臂必须 >= 19/20，否则先查课程生成与电池，再看学生臂。
# 起跑前已知（写进来，免得事后当发现）：
#   · 教师冒烟 K=3、2 个 seed：81 / 83 步全部过 3 条线，末 20 步高度 1.98–2.01 m、每步位移 0.2–0.3 m 无衰减，末动作终点 87.4 / 87.6 s
#     （起飞前约 13.5 s）。11-09 的电池探针：READY 后约 106 s 开始掉高，约 123 s 进 FAILSAFE（READY 约在复位后 8–9 s）。
#   · 学生 K=1 过线集的中位步数 31–33（教师 27），按 3 倍估算 K=3 要约 95–100 步 ≈ 复位后 100 s，已逼近掉压起点（约 115 s）；
#     超过约 110 步的集会进入掉压段，约 130 步进 FAILSAFE、采集中止。summarize.py 把这类集单列（采集中止），计入没通过，
#     并给出每集 sim_s、末 10 步最低高度，用来把「电池」与「导航失败」分开。
#   · 起始 SoC 保持 0.5 不变（同指纹 cf_sitl_v1:1bc8fedf:lib31，与今天的 K=1 结果同一动力学档）。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-long-course-20261110
SRC=/root/claude-code-backup-20261028/code/wt/p12-longcourse   # 代码源（分支 p12-longcourse）；起跑时冻结其 HEAD
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与今天 K=1 评测同一个推理二进制
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
K=3                                                      # 块数；教师冒烟 K=3 正常，所以不降到 2（依据见上）
STEPS=$((80 * K))                                        # 每块 80 步的预算，到成功线即停
EPISODES=20                                              # 每臂请求集数 = 分母
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
GUARD=2                                                  # 与 R1 实验同一个护栏阈值
LANES=5                                                  # 同时 5 个 gz；另一个 agent 占 2 个，机器上限 7
EP_TIMEOUT=$((600 + 8 * STEPS))                          # 每集墙钟上限（秒）：实测 1.1–1.8 s/步，留约 4 倍
BASE=$D/exp-student-select-20261109/e1-noise             # 三个训练种子的原始 run（ckpt_best）
SWA=$D/exp-swa-calib-20261110/ckpt                       # 权重平均 + 校准权重
# 臂：名字|策略|权重|护栏阈值。顺序即优先级（xargs 按序派发）：教师 > calswa+护栏（主臂）> calswa 无护栏 > ckpt_best+护栏（次要）
ARMS=(
  "teacher|teacher||0"
  "guard2-calswa-s0|student|$SWA/cal_swa_s0_k10.bin|$GUARD" "guard2-calswa-s1|student|$SWA/cal_swa_s1_k10.bin|$GUARD" "guard2-calswa-s2|student|$SWA/cal_swa_s2_k10.bin|$GUARD"
  "calswa-s0|student|$SWA/cal_swa_s0_k10.bin|0" "calswa-s1|student|$SWA/cal_swa_s1_k10.bin|0" "calswa-s2|student|$SWA/cal_swa_s2_k10.bin|0"
  "guard2-s0|student|$BASE/seed0/run/ckpt_best.bin|$GUARD" "guard2-s1|student|$BASE/seed1/run/ckpt_best.bin|$GUARD" "guard2-s2|student|$BASE/seed2/run/ckpt_best.bin|$GUARD"
)
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.4 GB，临时文件放数据盘
export OUT INFER SEEDS K STEPS EPISODES CORRIDOR EP_TIMEOUT CODE=$OUT/code PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES K=$K steps=$STEPS guard=$GUARD"

# 冻结代码：已有副本就复用（续跑与首跑必须同一份）；工作树里有未提交的改动就不起跑（git archive 只带 HEAD，会悄悄丢改动）
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  if [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]; then
    echo "$(date '+%F %T') $SRC 有未提交的改动，先提交再起跑"; exit 4
  fi
  mkdir -p "$CODE" && git -C "$SRC" archive HEAD | tar -x -C "$CODE" || { echo "冻结代码失败"; exit 4; }
fi
{ echo "code_head $(git -C "$SRC" rev-parse HEAD)"; sha256sum "$CODE/sim/tools/scene_course.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/dynamics/e1_mini.json"
  for a in "${ARMS[@]}"; do c=${a#*|}; c=${c#*|}; c=${c%|*}; [[ -n "$c" ]] && sha256sum "$c"; done; sha256sum "$INFER"; } > "$OUT/inputs.txt" 2>&1

run_arm() {
  local name=$1 policy=$2 ckpt=$3 guard=$4
  if [[ "$policy" == student && ! -f "$ckpt" ]]; then echo "$(date '+%F %T') 臂 '$name' 的权重不存在：$ckpt"; return 0; fi
  # 评完 = eval.json 不是 partial，且请求的集都跑过（出结果 + 采集失败 = 请求数）。电池掉压造成的采集中止是测量结果，不重跑。
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$name/eval.json" "$EPISODES" 2>/dev/null; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"   # 半成品挪开留证据，不删
  echo "$(date '+%F %T') $name 开始"
  local extra=(--policy teacher)
  if [[ "$policy" == student ]]; then
    extra=(--policy student --ckpt "$ckpt" --infer-bin "$INFER")
    [[ "$guard" -gt 0 ]] && extra+=(--heading-guard "$guard")
  fi
  "$PY" "$CODE/sim/tools/eval_closed_loop.py" "${extra[@]}" \
      --task course_base --blocks "$K" --split test --seeds "$SEEDS" --episodes "$EPISODES" --steps "$STEPS" \
      --sim-timeout 120.0 --episode-timeout "$EP_TIMEOUT" --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') $name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)"
  return 0                                   # 单臂失败不停整个队列，汇总里按 failed 标出来
}
export -f run_arm
printf '%s\n' "${ARMS[@]}" | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n p c g <<< "{}"; run_arm "$n" "$p" "$c" "$g"'

"$PY" "$OUT/summarize.py" > "$OUT/summarize.log" 2>&1
rc=$?
echo "rc=$rc $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束 rc=$rc"
