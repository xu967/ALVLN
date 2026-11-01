# 交接文档刷新（L1 v4）

- 时间：2026-11-01 21:52
- 范围：只改文档，未改代码、未跑实验
- 结论：完成。`docs/HANDOFF.md` 与 `docs/TASKS.md` 已更新到 v4
  现状（SR 0.426、停止点不可观测、方案 A 已落地、B/C 待拍板）。

## 做了什么

- `docs/HANDOFF.md`
  - 头部「当前状态」整段重写：v4 数字 + 停止点不可观测 + 方案 A + 未过 50% 线；
    补「goal_reach 与 goal_stop 不可混报」的醒目提示。
  - §4.3 拆成两节：4.3.1 是 v4 现状（**唯一能当现状引用的数字**），
    4.3.2 是 v1/T6 历史基线，并写明「v1 的 0.90 不比 v4 的 0.426 好，任务不同」。
  - §4.4 补 v3/v4 六条路径（原始采集、ds、cf、run、eval、probe、闭环）。
  - §4.5 整节重写：数据四代沿革表（v1 作废 / v2 历史 / v3 历史 / v4 在用）、
    「指令学会了停不会」的证据、失败分解、方案 A 的落地清单与 `REACH_RADIUS=1.5`
    的几何推导、**对早前 57% 估计的诚实更正（实际 43%）**、下一步 B/C 二选一、
    v4 复现命令。
  - §4.5 末段换成「闭环口径注意事项」+「未提交清单（实测 39 改 / 5 删 /
    6 暂存删 / 31 未跟踪）」。
  - §六补 `goal_reach` vs `goal_stop` 对照表，并说明 v1 走廊 SR 与 L1 SR 不可比。
  - §七标题与开头改成「T1–T7 已完成，当前是 L1 v4，等 B/C 拍板」，并列出
    **不要自行开工**的项目（T8 / D5 / ECPT / 公开数据预训练）。
  - §八补 6 条新坑：`audit_dataset` 不验图像、反事实包无 test split、
    训练末尾 test 用末轮模型、`carry_resume_best`、resume 继承 class_weights、
    EVAL_STANDARD 关系词曾与代码相反。
  - §九补 4 条待确认：SR < 50%、13 次碰撞、audit 不验图像、resume 两处缺陷。
  - 文末加「给接手 AI 的最后三条」。
- `docs/TASKS.md`：L1 行改成 v4 状态 + 卡点 + 待拍板；L2 行从 doing 改 done
  （FiLM 已落地并训练，语言门槛已过）。

## 没做什么

- **没改任何代码**，没重跑 v4 的三组闭环、离线、探针
  - 原因：本次只要求写交接文档，重跑要 80 min 且数值必然相同。
- 没修 §八/§九 里列出的三处代码缺陷（`carry_resume_best`、resume 继承
  `class_weights`、训练末尾 test 用末轮模型）
  - 原因：v4 没有用 `--resume`，不影响当前结论；属独立小任务。
- 没给 `audit_dataset.py` 补图像校验
  - 原因：超出本次范围，且需要重新设计检查项，未立项。
- 没做 PO / EO / AGV / MP（缺 `d_min` A*）
  - 原因：仍未实现，`EVAL_STANDARD §5.2` 已记录。
- **没有 commit / push**
  - 原因：用户未授权；工作树里有并发会话的改动。
- 没动 `.ipynb_checkpoints/`（未跟踪的垃圾目录）
  - 原因：不是本次任务产生的，不删用户没让删的东西。

## 待确认

- **L1 下一步是 B（改几何让停止可观测，重采重训）还是 C（改任务定义）**——
  这是唯一阻塞项，需要用户拍板。
- `acc_correct ≥ 0.90` 这条阈值是否保留（出自内部计划，不是公开基准）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `git status --short` 计数 | 39 改 / 5 删 / 6 暂存删除 / 31 未跟踪（已写进 §4.5） |
| `grep REACH_RADIUS sim/tools/eval_closed_loop.py` | `REACH_RADIUS = 1.5`（第 40 行），`success_criterion` goal_reach 分支在第 478 行 |
| v4 数据目录存在性 | `l1-v4-relation-500`（含 ds/train/val/test）、`l1-v4-cf`（含 ds/run-film-cf/eval-best-test/probe-best.json）、`l1-v4-closed-20261101` 均在 |
| 文档内所有数字 | 全部抄自 `experiments/2026-11-01-l1-v4-text-training.md` 与 `worklog/2026-11-01-2130-stop-point-unobservable.md`，未新增未跑过的数 |
| 代码测试 | **未跑**——本次没有改代码 |

## 风险与遗留

- HANDOFF 有 800+ 行，历史章节（§7 T1–T7 的任务定义）仍然很长。新 AI 如果
  从头读到尾会浪费上下文；头部已加「先看 §4.5」的指引。
- §4.3.1 的 v4 SR 是**用已记录 episode 复算**的（goal_reach 是
  `final_waypoint_dist`/`collision` 的纯函数）。只跑了 2 集端到端确认一致，
  三组全量没重跑。
- 工作树里 31 个未跟踪文件中有 25 个是文档，是语言线的全部留痕。
  如果有人 `git clean -fd`，这些会全部消失，**没有任何备份**。
- `docs/HANDOFF.md` 与 `docs/TASKS.md` 本身也在未提交列表里。
