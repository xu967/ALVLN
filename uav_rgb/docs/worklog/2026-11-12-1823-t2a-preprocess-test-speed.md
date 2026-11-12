# T2a：`check.sh --python-only` 里打包测试提速（7 个 `tools/test_preprocess_*.py`）

- 时间：2026-11-12 18:23
- 范围：`tools/test_preprocess_*.py`（7 个）与新增的 `tools/preprocess_testlib.py`；生产代码、`tools/check.sh`、`env.sh` 没动
- 结论：完成。7 个文件 CPU 合计 508.8 s 降到 49.4 s（9.7%，目标 ≤30%）；覆盖没减（见下）；`check.sh --python-only` 整套 rc=0，总耗时 500 s（18:11:58 到 18:20:18）
- 报告：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/t2a.md`；草稿与全部原始数据：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/t2a/`
- 分支 `p12-t2a`，工作树 `/root/claude-code-backup-20261028/code/wt/p12-pk1`；没有合并，没有推送

## 做了什么

- 新增 `tools/preprocess_testlib.py`（提交 `dfd8fad`、`b1199b6`）：只读共享夹具 `SharedFixture`（TMPDIR 下造一次、记每个文件的 sha256、`copy_to` 拷出去改、`verify()` 核对哈希没变、`cleanup()`）；进程内入口 `run_preprocess`（恢复 `sys.argv`、工作目录与环境变量，把 `SystemExit` 折成退出码，返回与 `subprocess.run` 同形的结果）；真子进程入口 `run_preprocess_cli`。名字不以 `test_` 开头，`check.sh` 的通配不会当测试跑。
- 7 个测试文件各一个提交，夹具改成 `setUpModule` 造一次，用例拷贝后再改，每个用例的 `tearDown` 与 `tearDownModule` 核对夹具哈希，打包改进程内调用，每个文件留两个真子进程用例（锁命令行退出码与 stderr 关键字）：`005934d` label_alignment、`87d0bf0` empty_episode、`8edc89b` timing_filter、`c4ab0b5` cond、`9877444` failed_teacher、`125c9c4` analytic_depth、`055fa6e` split。
- `9bb44e1`：failed_teacher 文件头注释更正，只改两行注释，行为不变。
- label_alignment 新增 1 个用例 `test_cli_missing_frame_exits_1_with_stderr_keyword`（缺帧时退出码 1、stderr 含「缺帧」与文件名），因为该文件原来只有成功路径的用例。
- empty_episode 与 failed_teacher 的各 1 个失败用例，断言从「stdout 加 stderr 里找关键字」收紧为「只在 stderr 里找」。
- 量了基线与改后（`bench/before.json`、`bench/after.json`），另量了加 `OPENBLAS_NUM_THREADS=1` 的两组（`bench/before_cap.json`、`bench/after_cap.json`）：根因是每个导入 numpy 的进程在 88 核主机上烧约 1.8 s CPU（容器配额 12 核）。
- 覆盖核对：改前改后导出用例清单与断言数；22 个变异在改前树与改后树各跑一遍；稳定性（每文件连跑 5 遍、3 个随机顺序种子、全走真子进程）；夹具哈希自检的有效性实验。脚本与结果都在 `t2a/` 下。
- 提速建议写进报告第九节（线程上限放 `tools/check.sh`，不放 `env.sh`）。

## 没做什么

- 没改 `tools/preprocess.py` 与任何生产代码。原因：卡的「不做」；`git diff --stat f22647e..HEAD` 只有 7 个测试文件和 1 个新辅助模块（`t2a/evidence/diffstat.txt`）。
- 没改 `tools/check.sh` 与 `env.sh`。原因：卡的「不做」；线程上限的建议留给主会话（报告第九节第 1 条）。
- 没在 `preprocess_testlib.py` 里设 `OPENBLAS_NUM_THREADS`。原因：OpenBLAS 在导入 numpy 时读变量，辅助模块里设只对先导入它的文件生效；两个真子进程用例会继承该变量，被测命令行的环境与用户手敲时不同；这是全局环境问题，归 `check.sh`。
- 没碰 T2b（`sim/3dgs/test_collect_eval.py`）。原因：卡的「不做」，另有执行者。
- 没有跑「整套 `check.sh --python-only` 加线程上限」。原因：卡里没有要求，且要再占一次共享的 `check.lock`（排队已有多个执行者）；只有 7 个文件级的数据（`bench/after_cap.json`）。线程上限对其余测试有没有影响（例如依赖多线程 BLAS 求和顺序的 `GOLDEN` 哈希）未核实。
- 没有在同一时段对主干再跑一遍 `check.sh` 做同场对比。原因：卡指定的对比对象是审核 P 的 544 s；同场对比要再占一次 9 到 18 分钟的锁。报告第八节给了 7 个文件的前后对照，来自不同时刻，不能当同场数据。
- 没有给 `make_synthetic_episode.py` 与 `preprocess_testlib.py` 做变异检查。原因：它们不是被测对象。
- 没有把进程内调用与共享夹具推广到 7 个文件之外的测试。原因：超出本卡范围（T2b 与审核 P-10 的第三条另算）。
- 没有读 OpenBLAS 源码核实「线程空转」的机制，只测了现象（CPU 约为墙钟的 6 倍，设线程上限后降到约 0.14 s）。原因：不影响结论，机制只写成推断。
- 没有合并、没有推送、没有改 `progress.md` 与 `CLAUDE.md`。原因：卡与交接说明的「不做」。
- 没有停用任何别人的进程。我自己起的守护进程都已结束（`logs/*.DONE`）。

## 待确认

- label_alignment 新增的那个用例要不要留（报告第九节第 4 条）：删了该文件就只剩 1 个真子进程用例。
- 线程上限加在 `tools/check.sh` 第 5 行之后，是否采纳，以及采纳前谁来跑整套核对。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 7 个文件各连续跑 5 遍（`python tools/<文件>`，默认环境） | 35 次全部退出码 0（`t2a/stability/stability.json`） |
| 随机顺序（种子 1、20261112、7）× 7 个文件 | 全绿 |
| 全部用例改走真子进程（`--cli-mode`）× 7 个文件 | 全绿（进程内与命令行等价） |
| 用例清单与断言数改前对改后 | 82 个旧用例 81 个三项断言数相同，1 个只是断言写法的位置变了（动态数相同）；新增 1 个；没有删除（`t2a/coverage/compare.md`） |
| 22 个变异（任务指定的 6 类各一个，外加 16 个） | 改前 22/22，改后 22/22，集合相同，改后没有漏（`t2a/mut/compare.json`） |
| 夹具哈希自检实验（改、加、删各一版） | 库级与端到端都会红，对照全绿（`t2a/evidence/fixture_selfcheck.txt`） |
| 改后基准（2 遍取最小） | CPU 49.4 s（改前 508.8 s，9.7%），子进程 202 → 14（`bench/after.json`） |
| `check.sh --python-only`（守护进程，`flock check.lock`） | rc=0，总墙钟 500 s；7 个文件单项墙钟合计 20.0 s，审核 P 点名的四个老文件 84.34 s 降到 14.0 s（`t2a/check/digest.md`）；日志末行 `PASS（有 SKIP）` 是 `--python-only` 的正常末行 |
| `git status --short` | 空 |
| 残留进程 | 无 |

## 风险与遗留

- 提交信息里的「CPU a s → b s」取自逐文件迭代测量（合计 60.3 s），收尾测量是 49.4 s；两次差异来自 OpenBLAS 空转的 CPU 读数受容器负载影响。不改写历史。
- 墙钟读数受容器内其他任务限流影响，跨时刻不可比；判断提速以 CPU 为准。
- 每个文件下限是 3 个导入 numpy 的进程（约 5.7 到 6.6 s CPU）；三个小文件（empty_episode、label_alignment、timing_filter）CPU 仍是改前的 37% 到 51%，要再降只能靠线程上限或合并进程。
- `cond` 的两个真子进程用例都是失败路径，成功路径的命令行退出码由其余 6 个文件的真子进程用例锁住；变异 M20（成功时退出码变 1）改后 `cond` 自己不再红，其余 6 个文件仍红。
- `sim/tools/test_w113_summaries.py` 用 `tempfile.mkdtemp` 建目录后不清理，每跑一次整套在 `TMPDIR` 留 7 个目录约 1 MB（不是本卡的文件，没改，写进报告第九节）。
- 清理：已删除变异工作副本 `t2a/mut/before`、`t2a/mut/after` 与 `t2a/tmp/` 下的临时文件（可由 `mutate.py` 重新生成）；`t2a/before_snap/` 是上一任留下的，没动。
