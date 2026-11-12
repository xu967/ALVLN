#!/usr/bin/env bash
# 卡 E2E-1 第 3 步起跑前的代码冻结：把分支 p12-e2e-1 已提交的 HEAD 解到 $OUT/code，记 $OUT/code_head.txt。
# 之后 prepare.sh、run.sh 都从 $OUT/code/experiments/2026-11-12-e2e-heading-code/ 启动，起跑时再逐文件核对冻结副本与 CODE_HEAD 的提交一致（lib.sh 的 verify_frozen）。
# 用法（在 p12-e2e-1 工作树里，工作区干净时）：bash experiments/2026-11-12-e2e-heading-code/freeze.sh
# 拒绝的情形（rc=4）：sim/、tools/、experiments/ 里有未提交的改动或未跟踪的文件；$OUT/code_head.txt 已存在且与当前 HEAD 不同（冻结后不换代码；确要换，先由执行者人工处理并写进报告）。
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib.sh
source "$HERE/lib.sh"

head_sha=$(git -C "$SRC" rev-parse HEAD) || { echo "读不到 $SRC 的 HEAD" >&2; exit 4; }
dirty=$(git -C "$SRC" status --porcelain -- sim tools experiments | head -n 5)
if [[ -n "$dirty" ]]; then
  echo "拒绝冻结：$SRC 的 sim/、tools/、experiments/ 里有未提交的改动或未跟踪的文件（前 5 条）：" >&2
  echo "$dirty" >&2
  exit 4
fi
if [[ -f "$OUT/code_head.txt" && "$(cat "$OUT/code_head.txt")" != "$head_sha" ]]; then
  echo "拒绝冻结：$OUT/code_head.txt 记的是 $(cat "$OUT/code_head.txt")，当前 HEAD 是 $head_sha；冻结后不换代码" >&2
  exit 4
fi
mkdir -p "$OUT/code"
git -C "$SRC" archive "$head_sha" | tar -x -C "$OUT/code" || { echo "git archive 失败" >&2; exit 4; }
echo "$head_sha" > "$OUT/code_head.txt"
CODE_HEAD=$head_sha
CODE=$OUT/code
if ! verify_frozen; then
  echo "冻结副本与 $head_sha 不一致：$FROZEN_BAD" >&2
  exit 4
fi
echo "已冻结 $head_sha → $OUT/code（sim/、tools/、本入口目录逐文件核对一致）"
echo "下一步：cd $OUT/code/experiments/2026-11-12-e2e-heading-code && setsid nohup bash prepare.sh > /dev/null 2>&1 < /dev/null &"
