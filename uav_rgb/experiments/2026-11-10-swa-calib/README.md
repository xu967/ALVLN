# 2026-11-10 权重平均与类先验校准闭环实验

不重训，只换推理权重：现行 `ckpt_best` 对「末 10 轮权重平均 + 类先验校准」，另设只校准、只平均两组，拆开两者各自的作用。
口径 `sim09.v2.coast1s.corridor3m`，20 个冻结 seed，三个训练种子合并 60 集，共 12 个臂。

- 预注册（H1、H2、证伪条件）：`run.sh` 文件头。
- 数据盘输出：`/root/rivermind-data/uav_rgb/exp-swa-calib-20261110/`（每臂 `<臂>/eval.json` 与 `<臂>.log`，候选权重在 `ckpt/`，汇总 `summary.md`）。
- 结果全表：`RESULT.md`（即上述 `summary.md` 的副本）。
- 结论：末 10 轮平均 + 校准 39/60，现行 `ckpt_best` 27/60（配对 McNemar b=20, c=8, p=0.036）；只校准 36/60，只平均不校准 20/60。

`run.sh` 与 `summarize.py` 是冻结的实验记录，不是维护中的入口：里面的路径指向当时的冻结工作树（`tmp/cursor-1110/wt`）与数据盘。
`tools/{swa,ck,calib}.py` 是生成这些权重的脚本，原样复制自 `tmp/cursor-1110/agents/src/`；`calib.py` 还 import 未入库的 `analyze.py`、`analyze2.py`，不能单独运行。
权重平均与校准工具计划整理成 `tools/average_ckpt.py`（计划 B4）。
正式工具：`tools/average_ckpt.py`（用法见文件头，测试 `tools/test_average_ckpt.py`）。用它重新生成上述 6 个权重（`swa_s{0,1,2}_k10`、`cal_swa_s{0,1,2}_k10`），sha256 与 `inputs.txt` 逐位一致（`UAVRGB_REPRO_AVGCKPT=1` 跑 `TestReproduction`）。
