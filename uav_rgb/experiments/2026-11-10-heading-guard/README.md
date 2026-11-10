# 2026-11-10 航向护栏闭环实验（R1）

不重训，只在评测端加航向护栏：n = 已执行的 YAW_LEFT 数减 YAW_RIGHT 数，|n| ≥ 2 时强制朝回转一步（`collect.py --heading-guard 2`），
用来判别「学生缺航向」是不是出阵的主因。口径 `sim09.v2.coast1s.corridor3m`，20 个冻结 seed，三个训练种子合并 60 集。

- 预注册（预测、证伪条件、主比较）：`run.sh` 文件头。
- 数据盘输出：`/root/rivermind-data/uav_rgb/exp-heading-guard-20261110/`（每臂 `<臂>/eval.json` 与 `<臂>.log`，汇总 `summary.md`）。
- 结果全表：`RESULT.md`（即上述 `summary.md` 的副本）。
- 结论：`ckpt_best` 走廊内成功 27/60，加护栏 47/60（配对 McNemar b=23, c=3），越界 23 → 0；权重平均校准 39/60，加护栏 50/60。

`run.sh` 与 `summarize.py` 是冻结的实验记录，不是维护中的入口：里面的路径指向当时的冻结工作树
（`/root/rivermind-data/uav_rgb/tmp/cursor-1110/wt-r1`）与数据盘，内容与数据盘上的原件逐字节相同。
