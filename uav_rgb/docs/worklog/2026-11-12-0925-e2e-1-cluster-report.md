# 卡 E2E-1 闭环前聚类补报脚本（审核 M-4、M-6）

- 时间：2026-11-12 09:25
- 范围：`experiments/2026-11-12-e2e-heading-code/cluster_report.py`、`test_cluster_report.py`、`README.md`
- 结论：完成（在 E2E-1 闭环出结果前盲态开发、验证并提交）

## 做了什么

- `experiments/2026-11-12-e2e-heading-code/cluster_report.py`（新增，提交 `521112e`）：
  - 只读读取 12 个评测臂（4 臂 × 3 训练种子，每臂 20 个冻结 seed），迟到集重跑合并规则与冻结汇总器 `summarize_e2e.py` 一致。
  - 每臂分母恒为 60：非迟到 `failed[]`、缺 `episode_manifest.json` 的崩溃集及未记录 `seed` 的失败集均记失败并占名额；触发异常集或 M2 剔除时按预注册同阈值重算 M1–M3 判决，并列明具体集与冻结汇总器原始结果。
  - 核对 20 个冻结 `seed` 与 20 个 `geometry_gid` 严格一一对应（缺 `geometry_gid`、同 `seed` 多 `gid`、多 `seed` 共用 `gid` 均报错退出，不猜测）；调用 `tools/paired_stats.pair_summary` 对 60 个 `(train_seed, eval_seed)` 单元做 20 几何簇双侧符号翻转精确检验（$\alpha = 0.05$）与整簇 bootstrap 95% 区间，输出四种判读写法之一及非零差几何数、同向几何数。
  - M2 越界集数在两侧均干净完成的配对单元上比较（任一侧发生重跑仍迟到、`actions.jsonl` 相邻 `rgb_ts` 差 `> 1.1 s` 停顿步、采集崩溃时两侧一起剔除），同时报告越界簇差符号翻转检验与无法读取逐步 `rgb_ts` 的集；未跑完的臂标「未完成」且不出判决；原子写出 `cluster_report.md` 与 `cluster_report.json`（含开头四条限定声明）。
- `experiments/2026-11-12-e2e-heading-code/test_cluster_report.py`（新增，提交 `521112e`）：全部采用桩数据覆盖 8 个用例（干净数据与冻结汇总器逐项一致、崩溃集保持 60 分母重算、7/8 几何全同向离散分辨率、15:0 集中于 5 几何判「成立而聚类不显著」、M2 剔除与 1.1 s 边界及缺逐步记录标注、`seed`/`gid` 失配报错、未完臂不判、配对键按训练种子对齐），并对脚本副本做 6 项变异检查（6/6 变红）。
- `experiments/2026-11-12-e2e-heading-code/README.md`（提交 `4673619`）：补充 `cluster_report.py` 用法与单测表。

## 没做什么

- 没读取 `/root/rivermind-data/uav_rgb/exp-e2e-heading-code-20261112/` 下任何评测臂的 `eval.json`、`run.log` 成功数或 `summary` 文件，没在真实输出目录上运行脚本。
  - 原因：严格遵守闭环出数前的盲态预注册约束。
- 没改 `summarize_e2e.py`、`run.sh`、`offline_report.py` 等入口已有文件，没改 `$OUT/code` 冻结副本，没改 `tools/`。
  - 原因：遵守任务边界，保持冻结汇总器与原入口不变以供对照。
- 没补 E2E-1 离线报告漏掉的「只平均不校准 vs 全局校准 vs 按码 0 校准」对照表（审核 M-1），也没做另外 40 个几何的样本外闭环补跑（审核 D3 / M-4）。
  - 原因：前者属 E2E-1 结项报告 / 卡 E2E-1b 范围，后者需等 E2E-1 闭环结束后另起 Gazebo 评测。
- 没跑 `test_run_rehearsal.py` 与 `test_prepare_rehearsal.py`，没启动 Gazebo 或占用 GPU。
  - 原因：`test_run_rehearsal.py` 头部注明会起多车道桩进程、不要与 Gazebo 评测同时跑；当前 Gazebo 池在跑实验、12 核满载。

## 待确认

- M2 在存在基础设施剔除单元时，`verdict_m2` 中非劣条件 `succ_a1 >= succ_a0` 的分母：本脚本主判决按细则第 4、5 条字面组合（「M2 越界集数在干净单元上比」+「成功数分母恒为 60」）使用 60 集总成功数，同时在 JSON（`clean_verdict_if_clean_succ`）与 Markdown 并列给出干净子集内成功数供核对。
- M2 的四种聚类判读写法：按提示第 3–5 条（「越界数的簇差符号翻转检验，只报告」），第 2 节主表统一按 `pair_summary` 的 60 对成功数聚类 $p < 0.05$ 套四分法，当成功差聚类不显著而越界减少聚类 $p < 0.05$ 时在第 3 节自动追加说明。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `nice -n 10 $PY experiments/2026-11-12-e2e-heading-code/test_cluster_report.py` | 通过（先红 `ModuleNotFoundError`，实现后 8 项 PASS，18.4 s） |
| 副本 6 项变异检查（`$D/tmp/phase12-cards/e2e-1-cluster/run_mutations.py`） | 通过（6/6 变红 `rc=1`） |
| 入口已有 5 个单测文件（`test_summarize_e2e`、`test_pack_checks`、`test_e2e_arms`、`test_verify_offline`、`test_freeze`，共 69 项） | 通过（全部 PASS） |
| `tools/check.sh --python-only`（守护脚本排 `check.lock`） | 已启动守护脚本排队运行（日志 `$D/tmp/phase12-cards/e2e-1-cluster/check-python-only.log`，完成标记 `check.DONE`） |

## 风险与遗留

- E2E-1 闭环结束后，收尾执行者需从合并后的主仓库（而非早先冻结的 `$OUT/code`）调用 `experiments/2026-11-12-e2e-heading-code/cluster_report.py $OUT` 生成 `$OUT/cluster/cluster_report.{md,json}`。
