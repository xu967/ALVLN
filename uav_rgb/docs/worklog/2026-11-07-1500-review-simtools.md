# REVIEW2-SIMTOOLS：Gazebo 侧仿真工具整仓审核（第六阶段 R2）

- 时间：2026-11-07 15:00（14:20 开工）
- 范围：`sim/tools/` 全部（不含 `record_views.py`）、`tools/e1_pipeline.py`、`tools/pick_test_seeds.py`、`tools/compare_datasets.py`、`tools/stats_report.py`、
  `tools/dagger_pipeline.py`、`tools/join_soft_targets.py` 调用侧；`552daac..28f0f78` 域内 11 个提交逐块读 diff，基线前代码从头读
- 结论：完成（报告交付；1 条 Must fix 已在 worktree 修、未 commit；2 条口径问题待主会话定）
- 报告：`/root/rivermind-data/uav_rgb/tmp/phase3-20261107/reports/REVIEW2_SIMTOOLS_20261107.md`
- 脚本与真实输出：`/root/rivermind-data/uav_rgb/tmp/agent7-r2-simtools/`（`run_tests.sh` → `tests.log` / `testlogs/`，`latency_check.py`，`run_gz.sh` → `gz-teacher-s180/`，`diff_scope.patch`，`PROGRESS.md`）

## 做了什么

- `tools/e1_pipeline.py`（1 行）：`evaluate()` 先把所有评测组 `wait()` 完再逐组查 eval.json。原来第一组缺 eval.json 就抛出，`main` 写 `DONE rc=1` 退出，其余组的 gz 留成孤儿，续跑再起 4 个 gz。
- 新增 `tools/test_e1_pipeline.py`：替身子进程，第一组不写 eval.json；修前只等了 `['teacher']`（红），修后四组都等完且仍报错（绿）。`e1_pipeline --dry-run` 修后仍 rc=0。
- 审核报告（ponytail-review 格式）：1 条 Must fix（已修）、8 条 Should fix、6 条 Nice to have，对 BRIEF_REVIEW §3 七问逐条回答，验证表与「没检查什么」。
- 重跑域内全部测试：29 份 `sim/tools/test_*.py` + `tools/test_pick_test_seeds.py` / `test_stats_report.py` / `test_join_soft_targets.py` 全部 rc=0；自检里只有 `teacher_kino.py` 的墙钟断言在负载下失败（1.73 s > 1.5 s）。
- 1 个 gz 实跑 teacher seed 180 一集（software GL，`--keep-work`）：rc=0，M1 40/40，`action_start == pose_before_ts` 40/40，sim_dt 1.000–1.015，SPL 0.9636；跑完确认没有我的 gz 残留。
- 真数据对拍：0417 采集 300 集派发延迟均值 0.131 s（max 0.149）；eval2 里 student-e1-new seed 127 过线后撞记失败（口径问题，报告第 3 条）；gid 复算与 scene.json 一致（0417 集 95/95、t5-full-1155 536/536）。

## 没做什么

- 没有 commit：按 BRIEF_REVIEW §4 只在 worktree 改，由主会话合并。
- 没改 `eval_closed_loop` 的 workdir / SIGTERM 路径（报告第 2 条，REVIEW_SITL 第 4 条未修项）：约 15 行、涉及产物布局，超出「≤30 行且无口径影响」的直接修范围，给了修法。
- 没改成功率口径（报告第 3 条：过线后撞计失败、不在线处停）：改哪一种都动 `docs/EVAL_STANDARD.md`，由主会话定。
- 没统一三份 seed 列表（报告第 5 条）：按 BRIEF_REVIEW2 不改文档里的列表，代码里是否统一一并由主会话定。
- 没改 `tools/check.sh`（报告第 9 条，三份 tools 测试不在门禁）：不在我的文件域。
- 没抽 `tools/pipeline_common.py`（报告第 6 条）：三条流水线同时改动，等主会话排期。
- 没跑 `--policy student` 的 gz、没真跑 `compare_datasets` / `dagger_pipeline`、forest_e1 没有实跑；eval3 只读目录与日志，没等终值。
- 基线前的 `cf_sil.py` / `cf_blackbox.py` / `teacher_policy.py` / `campus*.py` 等只抽查头部与调用点。

## 待确认

- 报告第 3 条的成功率口径选 (a) 过线即停 还是 (b) 第一次接触前过线。
- 报告第 5 条：`compare_datasets` / `dagger_pipeline` 的 seed 列表是否改为引用 `e1_pipeline.CFG["eval_seeds"]`。

## 验证到哪一步

| 项 | 结果 |
|---|---|
| `tools/test_e1_pipeline.py` | 修前红（只等了 teacher），修后 PASS |
| `tools/e1_pipeline.py --dry-run` | rc=0 |
| `tools/test_pick_test_seeds.py` | PASS（20 seeds, 20 gids） |
| 域内 29 + 3 份测试 | 全部 rc=0（`agent7-r2-simtools/tests.log`） |
| gz 实跑 1 集 | rc=0，无残留 gz |
| `git status --short` | ` M tools/e1_pipeline.py`、`?? tools/test_e1_pipeline.py`、本 worklog |

## 风险与遗留

- 报告第 1 条修的是等待顺序，没有加「失败就杀掉其余组」：按现有约定其余组继续跑完并各自写 eval.json，更符合 docstring「四组全跑完再一起看」。
- 评测中断仍会在 TMPDIR 留下 `uavrgb_eval_*`（本机现有 1 个 13:07 的、`/tmp` 下 6 个旧的），清理前先确认不属于正在跑的 eval3。
