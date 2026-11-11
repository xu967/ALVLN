#!/usr/bin/env bash
# 演练 run.sh 的流程（不起 Gazebo、不占 gz）：假评测入口 stub_eval.py 写真实格式的 eval.json 与轨迹，run.sh 照常走
# 阶段 1 → 标定门 → L 阶梯 → 阶段 2 → summarize.py → DONE。用法：bash rehearse.sh <场景>，场景见下；输出在 $TMPDIR 下的临时目录。
#   ladder   默认：K=3 在 L=0.16 第 0 步迟到、0.20 不迟到 → 门在 0.16 不过、0.20 过 → 进阶段 2，DONE rc=0，L_STAR=0.20
#   pass016  K=3 不迟到 → 门在 0.16 就过 → 不跑 0.20，阶段 2 复用验收 2 那一组（k1-s1 已评完，跳过）
#   fail     K=3 在 0.16 与 0.20 都迟到 → 不进阶段 2，DONE rc=10 并写原因
#   fwd      K=3 稳态前飞位移缩 5%（不迟到）→ 两个 L 都因 2% 门不过，DONE rc=10
# 结束时打印 DONE、L_STAR、各臂目录与 summarize.py 的标定门行，供人核对。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SCENARIO=${1:-ladder}
D=/root/rivermind-data/uav_rgb
R=/root/claude-code-backup-20261028/code/uav_rgb
TMP=${TMPDIR:-$D/tmp}/rehearse-$SCENARIO-$$
mkdir -p "$TMP/v1"
echo "rc=0 rehearsal" > "$TMP/v1/DONE"
export OUT=$TMP/out V1=$TMP/v1 EVAL=$HERE/stub_eval.py LANES=2
case "$SCENARIO" in
  ladder)  export STUB_K3_LATE_AT=0.16 ;;
  pass016) export STUB_K3_LATE_AT=0.99 ;;
  fail)    export STUB_K3_LATE_AT=0.16,0.20 ;;
  fwd)     export STUB_K3_LATE_AT=0.99 STUB_K3_FWD_SCALE=0.95 ;;
  *) echo "未知场景 $SCENARIO"; exit 2 ;;
esac
mkdir -p "$OUT" && cp "$HERE/run.sh" "$HERE/summarize.py" "$OUT/"
# 冻结代码用真实的提交（git archive，不起任何仿真）；summarize.py 要它里面的 11-10 长课程汇总函数
( cd "$OUT" && bash run.sh ) > "$TMP/rehearse.stdout" 2>&1
echo "--- run.log 末尾"; tail -n 14 "$OUT/run.log"
echo "--- DONE: $(cat "$OUT/DONE" 2>/dev/null)"
echo "--- L_STAR: $(cat "$OUT/L_STAR" 2>/dev/null)"
echo "--- 臂目录: $(ls "$OUT" | grep -E '^(cal|acc|forest)-' | tr '\n' ' ')"
"$R/.venv-tools/bin/python" "$OUT/summarize.py" --gate 0.16 "$OUT"
"$R/.venv-tools/bin/python" "$OUT/summarize.py" --gate 0.20 "$OUT"
echo "--- 输出目录 $OUT"
