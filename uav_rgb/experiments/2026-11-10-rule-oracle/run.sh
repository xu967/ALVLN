#!/usr/bin/env bash
# 规则层上限实验（计划 docs/PLAN_2026-11-10_phase11.md §4、§8 D3）：零训练。
# 柱子真值算「完美的列距离」，接固定规则 v0，在固件在环的 Gazebo 闭环里评，量规则层本身的天花板，再决定要不要训练几何头。
# 规则（sim/tools/rule_policy.py，参数 RULE_V0 在该文件顶部，起跑前定死）：
#   blocked = 存在列 j，d_j·|sin φ_j| < r_safe 且 d_j·cos φ_j < d_look（机体前方 2·r_safe 宽、d_look 长的通道里有柱子）
#   1) |n| >= n_max 朝回转（n = 已执行左转数减右转数；n>0 向右，n<0 向左），回转侧最外 2 列 d < side_near 时不转（blocked 则 SAFE，否则 FORWARD）
#   2) blocked 朝左右两半 Σmin(d, sum_clip) 较大的一侧转；相差不到 10% 沿用上次转向；没有上次转向则朝能使 n 归零的方向，n=0 固定朝左
#   3) 否则 FORWARD_SLOW
# 运行：cd /root/rivermind-data/uav_rgb/exp-rule-oracle-20261110 && cp <仓库>/experiments/2026-11-10-rule-oracle/{run.sh,summarize.py} . \
#         && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f run.log；汇总随时可跑：python summarize.py（缺的臂标「进行中」）。结束落 DONE 与 summary.md。停：kill -- -$(cat run.pid)。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
#
# 预注册（2026-11-10 20:05，起跑前写下；规则参数不在这 20 个 test seed 上调，冒烟只用 train 分区 seed 2、3）：
#   预测（计划 D3 原文）：rule60 三次重复合并 60 集，走廊内成功 >= 54/60，撞柱 <= 3/60。
#   证伪：rule60 走廊内成功 < 48/60，说明规则层本身表达力不够（输入再完美也过不了），
#         先给规则层加短期记忆（保留最近 3 步列距离，按每步位移转到当前机体系）再评，不直接训练几何头。
#   次要预测：rule81 撞柱数 <= rule60（data-algo 审查：学生 18 次撞柱里 16 次被撞柱中心在 60° 视锥外，视场加宽应减少侧擦）。
#   主比较（3 个）：① rule60 走廊内成功对 54/60 与 48/60 两条线（Wilson 95% 区间）；② rule60 撞柱数对 3/60；
#                   ③ rule81 对 rule60 的撞柱数（同 seed 同重复序号配对，精确 McNemar；60 对不独立，p 只作参考）。
#   并列参照，不做检验结论：教师 A* 20/20（11-09 v1 口径，无走廊判据）、学生最好配置 calswa + 护栏 50/60（同口径）。
#   重复 3 次的理由：规则是确定的，但固件在环的噪声与时序每次不同，3 次独立重跑估计这部分方差；60 集 = 20 个几何 × 3 次，不是 60 个独立几何。
#   臂顺序即优先级：先 rule60 的 3 次，再 rule81 的 3 次。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-rule-oracle-20261110
REPO=/root/claude-code-backup-20261028/code/wt/p12-ruleoracle   # 只读：用来 git archive 冻结代码
COMMIT=18f2ec051221caee928d1fe4bd651473b67c5b67          # feat(rule) 提交：规则层 + collect/eval 接线 + 测试
WT=$D/tmp/phase12-20261110/ruleoracle/code-${COMMIT:0:7}  # 冻结代码目录（git archive 解出，之后分支怎么动都不影响本实验）
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）；规则臂不用 GPU，钉住是防 gz 误选 0 号卡
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
EPISODES=20                                              # 与上面 seed 个数一致
STEPS=80                                                 # 同 eval4 与 heading-guard；到成功线即停，步数上限只管不到线的集
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
HFOVS=(60 81)                                            # 60 = 学生相机（20 列）；81 = 真机 GC4653 水平视场（27 列），Gazebo 相机仍是 60°
REPS=3                                                   # 每个视场重复次数：固件在环噪声与时序每次不同
LANES=2                                                  # 同时 2 个 gz：本任务的配额（总共 7 个，另一个 agent 占 5 个）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.4 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT WT SEEDS EPISODES STEPS CORRIDOR PY=$R/.venv-tools/bin/python RULE_TOOLS=$WT/sim/tools
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES hfovs=${HFOVS[*]} reps=$REPS commit=${COMMIT:0:7}"

# 冻结代码：解档一次，已有则复用（目录名带提交哈希，内容不会变）
if [[ ! -f "$WT/sim/tools/rule_policy.py" ]]; then
  mkdir -p "$WT" && git -C "$REPO" archive "$COMMIT" | tar -x -C "$WT" || { echo "$(date '+%F %T') 解档冻结代码失败"; echo "rc=4 $(date '+%F %T')" > "$OUT/DONE"; exit 4; }
fi
{ echo "commit $COMMIT"; (cd "$WT" && sha256sum sim/tools/rule_policy.py sim/tools/collect.py sim/tools/eval_closed_loop.py sim/tools/teacher.py sim/tools/scene_gen.py)
  sha256sum "$PLUGIN/libCfSitlSystem.so"; "$PY" --version; } > "$OUT/inputs.txt" 2>&1

run_arm() {
  local name=$1 hfov=$2
  [[ -n "$name" && -n "$hfov" ]] || return 0
  # 评完 = eval.json 不是 partial、集数都完成、没有失败集（与 tools/pipeline_common.eval_finished 同口径）
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done")==int(sys.argv[2]) and e.get("failed_count")==0) else 1)' "$OUT/$name/eval.json" "$EPISODES" 2>/dev/null; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"   # 半成品挪开留证据，不删
  echo "$(date '+%F %T') $name 开始 hfov=$hfov"
  "$PY" "$WT/sim/tools/eval_closed_loop.py" --policy rule-oracle --rule-hfov-deg "$hfov" \
      --task base --split test --seeds "$SEEDS" --episodes "$EPISODES" --steps "$STEPS" \
      --sim-timeout 120.0 --episode-timeout 1500.0 --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" \
      --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "$(date '+%F %T') $name rc=$? $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-60)"
  return 0                                   # 单臂失败不停整个队列，汇总里按 failed 标出来
}
export -f run_arm
ARMS=()
for h in "${HFOVS[@]}"; do for ((r = 0; r < REPS; r++)); do ARMS+=("rule${h}-rep${r}|${h}"); done; done
printf '%s\n' "${ARMS[@]}" | xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n h <<< "{}"; run_arm "$n" "$h"'

"$PY" "$OUT/summarize.py" > "$OUT/summarize.log" 2>&1
rc=$?
echo "rc=$rc $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束 rc=$rc"
