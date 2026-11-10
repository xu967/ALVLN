#!/usr/bin/env bash
# 公开权重零样本基线对比：PULP-Dronet v3 ResBlock-1.0（320,928 参数）、Tiny-PULP-Dronet v3 dw-pw-0.125（2,784 参数）、DroNet 原版（约 32 万）
# 不训练、不微调，公开权重原样放进 Gazebo 柱阵闭环（sim/tools/baselines/baseline_infer.py 说 uavrgb_infer 的行协议，输出映射成 4 个动作）。
# 目的：回答「公开小模型直接用能到哪」，量域差异有多大。它不回答「我们的网络结构是否更好」：那要靠「同结构在我们数据上重训」的臂，本轮不做。
# 运行：cd /root/rivermind-data/uav_rgb/exp-baselines-zeroshot-20261110 && cp <仓库>/experiments/2026-11-10-baselines-zeroshot/{run.sh,summarize.py} . \
#         && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f run.log；汇总随时可跑：python summarize.py（缺的臂标「未完成」）。结束落 DONE 与 summary.md。停：kill -- -$(cat run.pid)。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
# 排队：起跑前同时等 exp-rule-oracle-20261110/DONE 与 exp-long-course-20261110/DONE（全机 gz 上限 7 个：rule-oracle 占 2、long-course 占 5），
#       之后用 5 条车道（届时规则层 v1 实验占 2 个）；等待期间每 10 分钟在 run.log 写一行（列出还缺哪个文件）；
#       等待上限 3 小时，超时退出并落 DONE 写明原因。
#
# 预注册（2026-11-10 20:30 写下，起跑前冻结；映射参数与符号只来自 train 分区帧的离线核对 offline_check.py，不看 20 个 test seed）：
#   预测（派工消息）：零样本公开权重在我们的柱阵上走廊内成功不超过 6/20（训练域是真实走廊与道路，视场与外观都不同），
#         且低于我们的学生加护栏（配对 McNemar；学生取 calswa-s1 + 护栏 2 = exp-heading-guard-20261110/guard2-calswa-s1 的同 seed 结果，17/20）。
#   证伪：任一零样本臂达到 12/20 以上。那说明公开小模型直接可用，我们的训练管线要重新审视。
#   离线核对给出的臂级预期（train 分区 10,200 帧，单帧动作占比，不是闭环；详见报告）：
#     pulp_v3：转向符号 +1（实测，裁剪视场内右减左 steer 差 +0.162），单帧 SAFE 0.205 / FORWARD 0.563 / 左 0.110 / 右 0.122，p_coll 中位数 0.35、>0.7 占 34%；预期能前进也会转，失败以撞柱或越界为主；
#     pulp_tiny：符号 +1（实测，+0.193），单帧 SAFE 0.309 / FORWARD 0.636 / 左 0.037 / 右 0.017，steer 标准差只有 0.064、5% 的帧过转向门槛；预期前进占比高、转得少，撞柱多；
#     dronet：符号判不出（沿用 +1），p_coll 中位数 0.9994、99.97% 的帧 > 0.7，单帧 FORWARD 占比 0.000；预期 FORWARD 占比 < 5%，走廊内成功 0/20，失败以「未到线」为主（原地不动，不是撞）；
#     dronet-recal：单帧 SAFE 0.066 / FORWARD 0.156 / 左 0.735 / 右 0.043；预期大多在原地左转。
#     （来源：experiments/2026-11-10-baselines-zeroshot/offline_check.py analyze，输出 $D/tmp/phase12-20261110/baselines-run/offline/offline_check.json）
#   主比较（3 个）：① 每个零样本臂走廊内成功数（Wilson 95% 区间）对 6/20 与 12/20 两条线；
#                   ② 零样本里成功最多的臂对学生 calswa-s1 + 护栏（同 seed 配对精确 McNemar，报 b、c 与 p）；
#                   ③ 每个基线的护栏臂对其无护栏臂（同 seed 配对 McNemar）：护栏是评测端规则，给基线也加上才公平，看它能救回多少。
#   参照（不重跑，路径在 summary.md）：学生 calswa 39/60、calswa + 护栏 50/60，教师 20/20，盲前进 0/20。
#   诊断量（不做检验）：各臂 FORWARD 占比（区分「原地不动」与「飞了但撞」）、护栏改写占比、闭环里 p_coll 与 steer 的分布。
#   次要臂：dronet-recal（碰撞阈值取校准帧中位数、S 取 90，两个改动偏离原文），排在最后；离线单帧动作占比预示它大多在原地左转，结果只作旁证。
#   不能下的结论：零样本臂的成败不能证明网络结构优劣；n=20 的单次比较不下结论。
#
# 起跑前修订（2026-11-10 20:44，主会话 20:41 批准；此时没有任何闭环数据。上面的预测、证伪线、主比较 ①② 不变，下面三处改了，原文留作记录）：
#   ① 去掉 dronet-guard2 臂。理由：离线核对显示 DroNet 忠实映射下单帧 FORWARD 占比 0.000（87% 的帧 SAFE，13% 左转），
#      护栏只在策略前进时才有作用空间，对只会悬停的策略没有，跑它每集空耗 80 步。dronet（忠实）与 dronet-recal 保留，排在队尾。
#      主比较 ③ 随之只对 PULP 的两个基线（pulp_v3-guard2 对 pulp_v3，pulp_tiny-guard2 对 pulp_tiny）。
#   ② 起跑条件由「等 rule-oracle」改为「同时等 exp-rule-oracle-20261110/DONE 与 exp-long-course-20261110/DONE」。理由：long-course 占 5 个 gz，
#      两个都让出之后才有 5 条车道可用；long-course 预计 21:40 到 22:00 结束（主会话估计，未核实）。等待上限仍是 3 小时，从本次重起算起。
#   ③ 车道数 2 改为 5（届时另有规则层 v1 实验占 2 个 gz，全机上限 7）。臂序改为 pulp_v3、pulp_tiny、pulp_v3-guard2、pulp_tiny-guard2、dronet、dronet-recal，
#      PULP 的 4 个臂并行先出数（主会话估计 25 到 30 分钟），dronet 同时起跑，dronet-recal 接在第一个收尾的臂之后。
#   冻结代码不变：本次修订只改 experiments/ 下的脚本，sim/ 下的代码与 COMMIT 逐字节相同（git diff --stat COMMIT HEAD -- sim tools 为空），仍用同一个冻结目录。
#   旧守护进程（PGID 1433245，20:33 起，只做过预检和等待，没有起过任何臂）随本次修订按 PGID 停掉，用本脚本重起；重起时刻与新 PGID 见 run.log 的「开始」行。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-baselines-zeroshot-20261110
REPO=/root/claude-code-backup-20261028/code/wt/p12-baselines   # 只读：用来 git archive 冻结代码
COMMIT=57cf1341f15e245c6fafd9bdf7d3eb9f44929276                                # feat(baselines) 提交：包装器 + 启动器 + 测试；之后分支怎么动都不影响本实验
WT=$D/tmp/phase12-20261110/baselines-run/code-${COMMIT:0:7}   # 冻结代码目录（git archive 解出）
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）；基线在 CPU 上跑，钉住是防 gz 误选 0 号卡
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
EPISODES=20                                              # 与上面 seed 个数一致
STEPS=80                                                 # 同 eval4、heading-guard、rule-oracle；到成功线即停，步数上限只管不到线的集（原地不动的基线每集要跑满 80 步 ≈ 160 s）
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
GUARD=2                                                  # 与 heading-guard 实验同值：|n| >= 2 即朝回转
LANES=5                                                  # 同时 5 个 gz：本任务的配额（全机上限 7 个，届时规则层 v1 实验占 2 个）
WAIT_FOR=(                                               # 起跑条件：这些文件都出现才起跑（两个实验各自让出 gz 的标志）
  $D/exp-rule-oracle-20261110/DONE                       # 规则层 v0 实验，占 2 个 gz，约 20:50 收尾（主会话估计）
  $D/exp-long-course-20261110/DONE                       # 长课程实验，占 5 个 gz，预计 21:40 到 22:00 结束（主会话估计）
)
WAIT_POLL_S=60                                           # 每 60 s 查一次
WAIT_LOG_S=600                                           # 等待期间每 10 分钟写一行日志
WAIT_MAX_S=10800                                         # 等待上限 3 小时（从本次启动算起）
# 臂：名字|启动器|护栏阈值（0 = 关）。顺序即优先级。启动器里写死各基线的 STEER_SIGN / 阈值 / 尺度，见 sim/tools/baselines/README.md
# dronet 没有护栏臂：见文件头「起跑前修订」①。5 条车道 6 个臂：前 5 个同时起跑，dronet-recal 接在第一个收尾的臂之后。
ARMS=(
  "pulp_v3|launch_pulp_v3.sh|0"             "pulp_tiny|launch_pulp_tiny.sh|0"
  "pulp_v3-guard2|launch_pulp_v3.sh|$GUARD" "pulp_tiny-guard2|launch_pulp_tiny.sh|$GUARD"
  "dronet|launch_dronet.sh|0"               "dronet-recal|launch_dronet_recal.sh|0"
)
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.4 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT WT SEEDS EPISODES STEPS CORRIDOR PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES guard=$GUARD commit=${COMMIT:0:7}"
finish() { echo "rc=$1 $2 $(date '+%F %T')" > "$OUT/DONE"; echo "$(date '+%F %T') 结束 rc=$1 $2"; exit "$1"; }

# 冻结代码：解档一次，已有则复用（目录名带提交哈希，内容不会变）
if [[ ! -f "$WT/sim/tools/baselines/baseline_infer.py" ]]; then
  mkdir -p "$WT" && git -C "$REPO" archive "$COMMIT" | tar -x -C "$WT" || finish 4 "解档冻结代码失败（$REPO @ $COMMIT）"
fi
[[ -f "$OUT/stub.ckpt" ]] || printf 'stub ckpt (占位文件，外部基线只检查它存在；不要当权重)\n' > "$OUT/stub.ckpt"
{ echo "commit $COMMIT"; echo "repo_head $(git -C "$REPO" rev-parse HEAD)"; echo "wait_for ${WAIT_FOR[*]}"
  sha256sum "$OUT/run.sh" "$OUT/summarize.py"
  (cd "$WT" && sha256sum sim/tools/baselines/baseline_infer.py sim/tools/baselines/launch_*.sh sim/tools/student.py sim/tools/collect.py sim/tools/eval_closed_loop.py)
  sha256sum "$OUT/stub.ckpt" "$D/baselines/pulp-dronet/src/tiny-pulp-dronet-v3/model/"*.pth "$D/baselines/dronet/dronet_best_weights.npz" "$D/baselines/pulp-dronet/src/tiny-pulp-dronet-v3/model/dronet_v3.py"
} > "$OUT/inputs.txt" 2>&1

# 一条臂的评测命令（预检与正式跑共用，保证预检验的就是正式的命令行）。第 4 个参数是输出目录，第 5 个是 BASELINE_LOG 路径（预检用别处）。
eval_arm() {
  local name=$1 launcher=$2 guard=$3 out=$4 blog=${5:-$OUT/$1.baseline.jsonl}
  local guard_flag=()
  [[ "$guard" != 0 ]] && guard_flag=(--heading-guard "$guard")
  export BASELINE_LOG="$blog"                            # 沿 eval → collect → StudentPolicy → 启动器 → 包装器继承；每步追加 steer / p_coll
  "$PY" "$WT/sim/tools/eval_closed_loop.py" --policy student --ckpt "$OUT/stub.ckpt" --infer-bin "$WT/sim/tools/baselines/$launcher" \
      --task base --split test --seeds "$SEEDS" --episodes "$EPISODES" --steps "$STEPS" \
      --sim-timeout 120.0 --episode-timeout 1500.0 --dynamics cf_sitl --vehicle e1_mini \
      --hold 1.0 --window-from observation --corridor-half-width "$CORRIDOR" ${guard_flag[@]+"${guard_flag[@]}"} \
      --out "$out"
}
# 评完 = eval.json 不是 partial、20 集都完成、没有失败集（与 tools/pipeline_common.eval_finished 同口径）
arm_finished() {
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done")==20 and e.get("failed_count")==0) else 1)' "$OUT/$1/eval.json" 2>/dev/null
}
run_arm() {
  local name launcher guard
  IFS="|" read -r name launcher guard <<< "$1"
  [[ -x "$WT/sim/tools/baselines/$launcher" ]] || { echo "$(date '+%F %T') 臂 '$name' 的启动器不可执行：$launcher"; return 0; }
  if arm_finished "$name"; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"   # 半成品挪开留证据，不删
  [[ -f "$OUT/$name.baseline.jsonl" ]] && mv "$OUT/$name.baseline.jsonl" "$OUT/$name.baseline.partial-$(date +%H%M%S).jsonl"
  echo "$(date '+%F %T') $name 开始"
  eval_arm "$name" "$launcher" "$guard" "$OUT/$name" > "$OUT/$name.log" 2>&1
  echo "$(date '+%F %T') $name rc=$? $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-60)"
  return 0                                   # 单臂失败不停整个队列，汇总里按未完成/失败标出来
}
export -f eval_arm run_arm arm_finished

# 预检（不起 gz）：对每个臂用正式命令行跑 eval_closed_loop.py，--out 指向一个非空目录。
# eval_closed_loop 在「参数解析 → 权重/启动器存在 → StudentPolicy 探针（hello + 预热 + 关闭，退出码 0）」都通过之后才检查 --out，
# 所以看到「非空」说明启动器、解释器、权重、协议握手全部可用，而且没有任何 gz 被拉起；看到别的报错就是起跑前就该修的问题。
mkdir -p "$OUT/preflight" && echo sentinel > "$OUT/preflight/NONEMPTY"
for a in "${ARMS[@]}"; do
  IFS="|" read -r name launcher guard <<< "$a"
  msg=$(eval_arm "$name" "$launcher" "$guard" "$OUT/preflight" "$OUT/preflight/$name.baseline.jsonl" 2>&1 < /dev/null)
  if [[ "$msg" == *"非空"* ]]; then echo "$(date '+%F %T') 预检通过 $name（启动器 $launcher，护栏 $guard）"
  else echo "$(date '+%F %T') 预检失败 $name：$(echo "$msg" | tail -n 3)"; finish 6 "预检失败：$name，原因见上一行"; fi
done

# 排队：等两个实验都让出 gz 配额（WAIT_FOR 里的文件全部出现）
missing_waits() { local f; for f in "${WAIT_FOR[@]}"; do [[ -f "$f" ]] || printf '%s ' "$f"; done; }   # 仍缺的文件，空 = 全到
t0=$SECONDS; last=-$WAIT_LOG_S
while [[ -n "$(missing_waits)" ]]; do
  if (( SECONDS - t0 >= WAIT_MAX_S )); then finish 5 "等待超过 ${WAIT_MAX_S}s（3 小时），仍缺：$(missing_waits)，没有起跑任何臂"; fi
  if (( SECONDS - last >= WAIT_LOG_S )); then echo "$(date '+%F %T') 等待中，仍缺：$(missing_waits)（已等 $((SECONDS - t0)) s，上限 ${WAIT_MAX_S} s）"; last=$SECONDS; fi
  sleep "$WAIT_POLL_S"
done
echo "$(date '+%F %T') 起跑条件满足：$(for f in "${WAIT_FOR[@]}"; do printf '%s=[%s] ' "$f" "$(cat "$f")"; done)lanes=$LANES"

printf '%s\n' "${ARMS[@]}" | xargs -P "$LANES" -I{} bash -c 'run_arm "{}"'

"$PY" "$OUT/summarize.py" > "$OUT/summarize.log" 2>&1
rc=$?
ok=0; for a in "${ARMS[@]}"; do IFS="|" read -r n _ _ <<< "$a"; arm_finished "$n" && ok=$((ok + 1)); done
finish "$rc" "全部臂结束，评完 $ok/${#ARMS[@]} 个臂，summarize rc=$rc"
