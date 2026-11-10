#!/usr/bin/env bash
# 长课程实验第二轮（phase 12 longcourse，round 2）：K=3 个 base 柱阵块首尾相接，不训练，用既有权重逐块测存活率。第一轮见 run.sh。
# 为什么重跑：第一轮（输出 exp-long-course-20261110）在 20:46 被主会话中止，原因是时序混杂。--window-from observation 下动作窗口从观测帧起算，
#   观测到派发的仿真时延越长，动作实际生效越短。K=3 世界每步 A* 重规划（教师臂是动作本身，学生臂是影子教师标签）耗 145 ms，K=1 约 35 ms
#   （25 根柱 vs 8 根），打点副本实测（diag_v1_table.txt，seed 1 各一集）：策略段墙钟 K=3 145–147 ms / K=1 33–38 ms，观测到派发 K=3 219–220 ms /
#   K=1 132–137 ms，等帧 0 ms、深度解码 7–8 ms、帧龄 114–116 ms 与 K 无关。第一轮每步动作生效 0.77–0.79 s（K=1 参照 0.871 s），
#   前飞位移 0.20–0.22 m（K=1 0.26 m），转角 22–23.4°（K=1 26.1°）：学生在 K=1 学到的「一步多远、转多少」在 K=3 世界不成立，第一轮 S_k 与 K=1 参照不可比。
# 改了什么（唯一的参数变化）：teacher.astar_path 的栅格可走性按场景缓存（提交 fcbf2fe，路径逐位不变，sim/tools/test_astar_cache.py 锁定）。
#   冻结代码 = 提交 fcbf2fe，与标定（calibrate_timing.sh）用的是同一份。标定（教师，seed 1 / 113 各 2 集，K=3 生效时长中位）：
#   第一轮代码 保留辅助相机 0.784 / 去辅助相机 0.763 / 去辅助相机 + 相机 5 Hz 0.643；第二轮代码（A* 缓存，保留辅助相机）0.872。K=1 对应 0.875 / 0.865 / - / 0.887。
#   任务书的两个手段都没用：去辅助相机使仿真更快（RTF 升高），同样的墙钟计算折算成更长的仿真时延；相机降频使帧更旧。本轮保留辅助相机，渲染配置与 K=1 参照一致。
# 课程：scene_course.py（--task course_base --blocks K）；第 0 块 = 同 seed 的 base 柱阵，第 k 块是同 split 的另一个 base 几何平移 5.0·k m。
#       成功线 = 6.0 + 5.0·(K−1)；逐块存活的判据与定义见 summarize.py 文件头。
# 运行：mkdir -p $OUT && cp run_v2.sh summarize.py $OUT/ && cd $OUT && setsid nohup bash run_v2.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；汇总随时可跑：python $OUT/summarize.py [--present-only]。结束落 DONE 与 summary.md。停：kill -- -$(cat $OUT/run.pid)。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
# 代码：冻结到 $OUT/code（提交 CODE_HEAD），之后只用冻结副本；已有副本（标定时冻结的）就复用，并逐文件校验与 CODE_HEAD 一致。
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
# 第二轮新增（不改上面任何一条预测）：
#   · 第一轮的数据（时序混杂，只作参考，不参与 P1–P3 判定）留在 exp-long-course-20261110，用 summarize.py --present-only 汇总。
#   · 有效性条件：每个臂的动作生效时长中位数须在 K=1 参照 0.871 s 的 ±3% 内（0.845–0.897），转角在 26.1° 的 ±3% 内；
#     summarize.py 第 8 节逐臂标注，超出的臂其 P1–P3 判定作废并写明原因。起跑后先看头几集的 0.854–0.888。
#   · 第一轮 FAILSAFE 采集中止的原因已核实：电池压降，复位后约 115 s 掉高、约 131 s 进 FAILSAFE（manifest error：「step 137: cf_sitl 执行器故障: 固件进入 FAILSAFE」）。
#     标定的教师过 3 条线用 70 / 68 步（第一轮代码同 seed 77 / 81 步），掉压段的集预计少于第一轮；采集中止仍计入没通过。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=$D/exp-long-course-v2-20261110
SRC=/root/claude-code-backup-20261028/code/wt/p12-longcourse   # 代码源（分支 p12-longcourse）
CODE_HEAD=fcbf2fe                                        # 冻结代码的提交：teacher.astar_path 栅格缓存；标定用的就是它
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与今天 K=1 评测同一个推理二进制
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
K=3                                                      # 块数
STEPS=$((80 * K))                                        # 每块 80 步的预算，到成功线即停
EPISODES=20                                              # 每臂请求集数 = 分母
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
GUARD=2                                                  # 与 R1 实验同一个护栏阈值
LANES=5                                                  # 同时 5 个 gz；另一个 agent 占 2 个，机器上限 7
EP_TIMEOUT=$((600 + 8 * STEPS))                          # 每集墙钟上限（秒）：实测 1.1–1.8 s/步，留约 4 倍
SWA=$D/exp-swa-calib-20261110/ckpt                       # 权重平均 + 校准权重
# 臂：名字|策略|权重|护栏阈值。顺序即优先级（xargs 按序派发）：教师 > calswa+护栏（主臂）> calswa 无护栏。
# guard2-s{0,1,2}（ckpt_best + 护栏 2，次要臂）这一轮不跑（任务书）。
ARMS=(
  "teacher|teacher||0"
  "guard2-calswa-s0|student|$SWA/cal_swa_s0_k10.bin|$GUARD" "guard2-calswa-s1|student|$SWA/cal_swa_s1_k10.bin|$GUARD" "guard2-calswa-s2|student|$SWA/cal_swa_s2_k10.bin|$GUARD"
  "calswa-s0|student|$SWA/cal_swa_s0_k10.bin|0" "calswa-s1|student|$SWA/cal_swa_s1_k10.bin|0" "calswa-s2|student|$SWA/cal_swa_s2_k10.bin|0"
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
echo "$(date '+%F %T') 开始（第二轮）PGID=$$ lanes=$LANES K=$K steps=$STEPS guard=$GUARD code=$CODE_HEAD"

# 冻结代码：已有副本就复用（标定与续跑必须同一份），没有就从 CODE_HEAD 解出；然后逐文件核对 sim/ 下的 git blob 哈希
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || { echo "冻结代码失败"; exit 4; }
fi
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || { echo "$(date '+%F %T') 冻结副本与 $CODE_HEAD 不一致：$bad"; exit 4; }
{ echo "code_head $(git -C "$SRC" rev-parse "$CODE_HEAD")"; sha256sum "$CODE/sim/tools/scene_course.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/dynamics/e1_mini.json"
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
