# 交接文档刷新（T7 复跑进行中）

- 时间：2026-10-30 21:55
- 范围：`docs/HANDOFF.md`、`docs/TASKS.md`、`CLAUDE.md`
- 背景：旧 HANDOFF 的「当前状态」停在 T5 放行前，仍写「闭环 0/20」「CTest 19」；
  T7 的 0.90 是 R01 修复前数字，容易让接手的人误当当前结论。

## 做了什么

| # | 文件 | 改动 |
| --- | --- | --- |
| 1 | `docs/HANDOFF.md` | 重写 §四：当前唯一 L3 任务（T7 同版本复跑）的 PID/命令/产物/进度/跑完后动作；T1–T7、D4、R01 完成表；关键数字与数据路径 |
| 2 | `docs/HANDOFF.md` | 顶部状态、CTest 20/20、§七 标题改为「T1–T7 已完成」，新增 T8 候选（未立项）；常用命令的闭环评测改用 `--seeds`；§十三补 T5/T6/T7 产物路径 |
| 3 | `docs/TASKS.md` | 「进行中」新增 `T7-R`（doing）：R01+D4 同版本复跑，完成前 T7 严格验收不关闭 |
| 4 | `CLAUDE.md` | Project state 标明 0.90 是 R01 前版本、复跑进行中；CTest 13→20 |

## 没做什么

- **没有写 T7 复跑的结论**：进程仍在跑（21:54 日志 12/20），中间快照只作进度，
  不作实验结论；跑完后必须按 HANDOFF §4.1 的清单补实验记录和 worklog。
- **没有改代码**，所以没跑 L1/L2；文档改动只用 grep 核对了旧数字/旧路径是否残留。
- **没有 commit**：用户未授权。

## 验证

- `grep -n "19/19\|T1–T7，按序\|闭环仍 0\|11/20\|CTest 19\|13 项" docs/HANDOFF.md CLAUDE.md docs/TASKS.md`
  无输出（人工核对）。
- T7 进程仍在：21:55 `ps -p 1054313` 显示运行 25:30；`eval-student-r01.log` 12 行，
  `eval.json` 尚未生成。
- `git diff --check -- docs/HANDOFF.md docs/TASKS.md CLAUDE.md docs/worklog/2026-10-30-2155-handoff-refresh.md`
  返回 0。

## 风险与遗留

- 交接文档里的进度快照会很快过期；接手的人应以 `eval-student-r01/eval.json` 和
  进程状态为准，不能引用快照当结果。
- 若复跑进程被杀/失败，§4.1 的「禁止重启」要改为先看日志、确认是否残留
  `gz sim`/订阅者，再决定是否重跑；不要直接覆盖 `eval-student-r01`。
- T8 只是候选，TASKS 未立项；用户拍板前不要开工。
