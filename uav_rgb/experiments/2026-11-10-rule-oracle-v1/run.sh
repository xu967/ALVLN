#!/usr/bin/env bash
# 规则层上限实验第二轮（计划 docs/PLAN_2026-11-10_phase11.md §4、§8 的 D3 → D6）：规则 v1 = v0 + 短期障碍记忆 + 候选航向选择。零训练。
# 柱子真值算「完美的列距离」，列距离变成机体系障碍点，按真值位姿存进世界系、保留 mem_steps 步；每步在「转回一步 / 保持 / 再转开」里
# 用记忆里的点判断通道是否畅通（sim/tools/rule_policy.py，参数 RULE_V1 在该文件顶部，起跑前冻结）：
#   clear(θ) = 记忆点里没有点满足 0 < u < d_look 且 |v| < r_safe（u = x·cosθ + y·sinθ，v = −x·sinθ + y·cosθ，θ 相对当前航向，正 = 左）
#   1) n ≠ 0 且 clear(转回一步) → 转回（V1-back）；2) clear(0) → FORWARD_SLOW（V1-fwd）；
#   3) 否则转开：n≠0 且 |n| < n_cap → 沿 n 的符号再转一步（V1-more）；n=0 → 左右两半 Σmin(d, sum_clip) 取较大一侧（V1-side），
#      相差不到 tie_frac 沿用上次转向（V1-last），没有则向左（V1-left）；|n| >= n_cap → 转回（V1-cap）。永不输出 SAFE_NEUTRAL。
# 运行：D=/root/rivermind-data/uav_rgb; mkdir -p $D/exp-rule-oracle-v1-20261110 && cd $D/exp-rule-oracle-v1-20261110 \
#         && cp <仓库>/experiments/2026-11-10-rule-oracle-v1/{run.sh,summarize.py} . && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f run.log；汇总随时可跑：python summarize.py（缺的臂标「进行中」）。结束落 DONE 与 summary.md。停：kill -- -$(cat run.pid)。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
# gz 配额：本任务最多 2 个实例。起跑前等 v0 守护进程收掉（$V0/DONE 出现）、冒烟结束，所以先于它们起跑也不会超配额。
#
# 预注册（2026-11-10，起跑前写下）：
#   这是第二轮。规则 v0（exp-rule-oracle-20261110）在 test 上 rule60 的三次重复判负（预注册证伪线 < 48/60），失败集几乎都是 80 步跑满、
#   max_x 停在 1.3–1.5 m（门柱前）、无碰撞：规则 1 回转例外分支恒走 SAFE_NEUTRAL 的悬停（v0 报告 §5.1 事前指出）。
#   规则 v1 由主会话在看过 v0 的失败集之后设计，参数（RULE_V1）冻结在起跑前，没有在 test seed 上调；
#   离线检查与 Gazebo 冒烟只用 train 分区的几何（离线：train 分区 20 个几何的运动学模拟；冒烟：train 分区 4 个 seed）。
#   预测（计划 D3）：rule60v1 三次重复合并 60 集，走廊内成功 >= 54/60，撞柱 <= 3/60。
#   证伪：rule60v1 走廊内成功 < 48/60，说明「真值列距离 + 固定规则」在 60° 视场下就不够，计划的 D 线收窄为只把几何头当辅助监督。
#   次要预测：① rule81v1 撞柱数 <= rule60v1；② 长课程 K=3（rule60v1-course3，20 集）整条课程成功 >= 16/20
#            （每块存活 0.93 的独立模型给 0.93^3 = 0.80），同时给逐块存活。
#   主比较（3 个）：① rule60v1 走廊内成功对 54/60 与 48/60 两条线（Wilson 95% 区间）；② rule60v1 撞柱数对 3/60；
#                   ③ rule60v1 对 v0 的 rule60（60 集，同 seed、同重复序号配对，精确 McNemar；60 对不独立，p 只作参考）。
#   并列参照，不做检验结论：学生最好配置 calswa + 护栏 50/60（同口径，配对 McNemar 仅供参考）、教师 A* 20/20（11-09 v1 口径，无走廊判据）。
#   重复 3 次的理由同 v0：规则是确定的，固件在环的噪声与时序每次不同；60 集 = 20 个几何 × 3 次，不是 60 个独立几何。
#   臂顺序即优先级：① rule60v1-rep{0,1,2}；② rule60v1-course3；③ rule81v1-rep{0,1,2}。2 路并行。
# 起跑前已知（写进来，免得事后当发现）：
#   · 离线（rule_sim.py，前进 0.25 m / 转 26° 的运动学模拟，不起 Gazebo）：train 分区 20 个几何过线 18/20、撞柱 0/20；没过线的 seed 3、14 是
#     「V1-more ↔ V1-cap」在 |n| = 2 与 3 之间摆动（左右各转 26°、52°、78° 的通道都被门柱挡住），原地转 80 步，max_x 1.76 m。
#     同一模拟里 v0 过线 1/20（悬停死锁），而 Gazebo 里 v0 在 test 上约 13/20：模拟没有悬停漂移，对原地不动类死锁偏悲观，对 v1 的估计未核实偏向哪边。
#   · Gazebo 冒烟（train 分区 seed 2、10、3、14，rule60v1，单路）：train 分区 seed 2、10、3、14 全部过线（31 / 39 / 31 / 29 步），撞柱 0、越界 0；条目只出现 V1-fwd / V1-back / V1-side / V1-left / V1-last / V1-more，
#     没有 V1-cap，最长连续换向 1 次，每步记忆点数 0–122（出了视场的柱子仍在记忆里）。seed 3、14 在 Gazebo 里没有复现离线模拟的 V1-more ↔ V1-cap 摆动
#     （转向步实际带位移、首个转向方向也与模拟不同）：这只说明模拟对这类摆动偏悲观，不说明规则没有这个缺陷；test 几何上的发生率未核实，看汇总的 V1-cap 占比与停滞形态。
#   · v0 的失败机制（事后核对 v0 的 60 集）：未到线 14 集里 13 集末 20 步换向 15–19 次、SAFE 占全部步 0.000、含 >=3 连续 SAFE 的集 0 个，
#     也就是 R1（|n| >= 2 回转）与 R2-sum（受阻转向）左右拉扯在门柱前（max_x 1.3–1.5 m），不是 SAFE 悬停；另 6 集越界（R3 前进，n = ±1 的航向没被纠正）。
#     v1 的 V1-more ↔ V1-cap 摆动与它同一类结构，这是 v1 最该看的失败形态。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-rule-oracle-v1-20261110
V0=$D/exp-rule-oracle-20261110                           # v0 守护进程的目录：其 DONE 出现后才起跑（配额）
REPO=/root/claude-code-backup-20261028/code/wt/p12-rulev1   # 只读：用来 git archive 冻结代码
COMMIT=cafb826de56d28c3539b1bd1e8282f3f1bd43de8                                          # feat(rule) 提交：规则 v1 + collect/eval 接线 + 测试
WT=$D/tmp/phase12-20261110/rulev1/code-${COMMIT:0:7}     # 冻结代码目录（git archive 解出，之后分支怎么动都不影响本实验）
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）；规则臂不用 GPU，钉住是防 gz 误选 0 号卡
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
EPISODES=20                                              # 与上面 seed 个数一致
STEPS=80                                                 # 同 v0 与 heading-guard；到成功线即停，步数上限只管不到线的集
BLOCKS=3                                                 # 长课程块数 K（--task course_base --blocks 3，成功线 6.0 + 5.0·2 = 16.0 m）
COURSE_STEPS=240                                         # 80·K：到成功线即停
COURSE_EP_TIMEOUT=2520                                   # 每集墙钟上限（秒）：600 + 8·240，同 long-course 实验
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
REPS=3                                                   # 每个视场重复次数：固件在环噪声与时序每次不同
LANES=2                                                  # 同时 2 个 gz：本任务的配额（总共 7 个，另一个 agent 占 5 个）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.4 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT WT SEEDS EPISODES STEPS BLOCKS COURSE_STEPS COURSE_EP_TIMEOUT CORRIDOR PY=$R/.venv-tools/bin/python RULE_TOOLS=$WT/sim/tools
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES reps=$REPS commit=${COMMIT:0:7}"

# 配额闸：v0 守护进程收掉（DONE 出现）且冒烟的 gz 退出之后才起跑；只读检查，不杀任何进程
waited=0
while [[ ! -f "$V0/DONE" ]] || ps -eo cmd | grep -F "$OUT/smoke" | grep -v grep | grep -q .; do
  (( waited % 300 == 0 )) && echo "$(date '+%F %T') 等 v0 的 DONE 与冒烟结束（配额）"
  sleep 20; waited=$((waited + 20))
done

# 冻结代码：解档一次，已有则复用（目录名带提交哈希，内容不会变）
if [[ ! -f "$WT/sim/tools/rule_policy.py" ]]; then
  mkdir -p "$WT" && git -C "$REPO" archive "$COMMIT" | tar -x -C "$WT" || { echo "$(date '+%F %T') 解档冻结代码失败"; echo "rc=4 $(date '+%F %T')" > "$OUT/DONE"; exit 4; }
fi
{ echo "commit $COMMIT"; (cd "$WT" && sha256sum sim/tools/rule_policy.py sim/tools/rule_sim.py sim/tools/collect.py sim/tools/eval_closed_loop.py sim/tools/teacher.py sim/tools/scene_gen.py sim/tools/scene_course.py)
  sha256sum "$PLUGIN/libCfSitlSystem.so"; "$PY" --version; } > "$OUT/inputs.txt" 2>&1

run_arm() {
  local name=$1 hfov=$2 kind=$3
  [[ -n "$name" && -n "$hfov" ]] || return 0
  local task=(--task base) steps=$STEPS ept=1500.0 want_failed0=1
  if [[ "$kind" == course ]]; then
    task=(--task course_base --blocks "$BLOCKS"); steps=$COURSE_STEPS; ept=$COURSE_EP_TIMEOUT
    want_failed0=0      # 长课程：电池掉压造成的采集中止是测量结果，不重跑（同 long-course 实验）
  fi
  # 评完 = eval.json 不是 partial，且请求的集都跑过（出结果 + 采集失败 = 请求数）；基础臂另要求没有失败集（与 v0 同口径）
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); done=e.get("episodes_done",0); fl=e.get("failed_count",0); sys.exit(0 if (not e.get("partial") and done+fl==int(sys.argv[2]) and (fl==0 or sys.argv[3]=="0")) else 1)' "$OUT/$name/eval.json" "$EPISODES" "$want_failed0" 2>/dev/null; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"   # 半成品挪开留证据，不删
  echo "$(date '+%F %T') $name 开始 hfov=$hfov kind=$kind"
  "$PY" "$WT/sim/tools/eval_closed_loop.py" --policy rule-oracle --rule-version v1 --rule-hfov-deg "$hfov" \
      "${task[@]}" --split test --seeds "$SEEDS" --episodes "$EPISODES" --steps "$steps" \
      --sim-timeout 120.0 --episode-timeout "$ept" --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "$(date '+%F %T') $name rc=$? $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-60)"
  return 0                                   # 单臂失败不停整个队列，汇总里按 failed 标出来
}
export -f run_arm
# 臂：名字|视场|类型。顺序即优先级（xargs 按序派发）：rule60v1 三次 → 长课程 → rule81v1 三次
ARMS=()
for ((r = 0; r < REPS; r++)); do ARMS+=("rule60v1-rep${r}|60|base"); done
ARMS+=("rule60v1-course3|60|course")
for ((r = 0; r < REPS; r++)); do ARMS+=("rule81v1-rep${r}|81|base"); done
printf '%s\n' "${ARMS[@]}" | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n h k <<< "{}"; run_arm "$n" "$h" "$k"'

"$PY" "$OUT/summarize.py" > "$OUT/summarize.log" 2>&1
rc=$?
echo "rc=$rc $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束 rc=$rc"
