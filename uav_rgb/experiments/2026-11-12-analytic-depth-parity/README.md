# 解析深度分带 对拍（任务卡 W1-8 验收 2）

`preprocess.py --depth-source analytic` 由位姿与柱子真值解析算深度分带。本目录用 Gazebo 的真实深度图检验它：采 1 集带 `--save-depth` 的教师轨迹，
同一集分别用 `file`（读深度图）与 `analytic`（解析）打包，比较「柱子所在单元」的分带一致率，门槛 ≥ 95%。

预注册（主比较、门槛、分类规则、预测）写在 `run.sh` 文件头，起跑前提交，起跑后不改。

| 文件 | 作用 |
|---|---|
| `run.sh` | 守护脚本：冻结代码（`git archive`）、阻塞式机器锁（Gazebo 池）、`collect_parallel.py` 采 1 集（seed 3100，40 步，`--save-depth`）、`DONE` |
| `compare.py` | 读 `file` 与 `analytic` 两个包的 `dataset.aux` 与 episode 源目录，逐步逐格比较，输出一致率、分带差直方图、不一致格的分类（柱边缘 / 地面 / 其它）和每步明细 |
| `posthoc_rays.py` | 事后分析（不在预注册之内，不参与判定）：同一集、同一个 `file` 包，只改 analytic 每格射线数（1、2、3、4、5、8、16），输出柱子格一致率表 |

输出：`$D/exp-2026-11-12-analytic-depth-parity/`（`$D=/root/rivermind-data/uav_rgb`）。

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-2026-11-12-analytic-depth-parity; PY=/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python
mkdir -p $OUT && cp run.sh $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
tail -f $OUT/run.log                     # 进度；排队期间每 10 分钟一行
kill -- -$(cat $OUT/run.pid)             # 停（只按进程组）
# 采完（DONE rc=0）之后，两种打包与比较（CPU 小活，几秒）：
EP=$OUT/collection/train/3100
$PY tools/preprocess.py --out $OUT/ds-file     $EP                              # 默认 --depth-source file
$PY tools/preprocess.py --out $OUT/ds-analytic --depth-source analytic $EP
$PY experiments/2026-11-12-analytic-depth-parity/compare.py --episode $EP --file $OUT/ds-file --analytic $OUT/ds-analytic --out $OUT/compare.json
```

射线数扫描（事后分析）：`$PY experiments/2026-11-12-analytic-depth-parity/posthoc_rays.py $EP $OUT/ds-file`。

结果与解读见 `docs/experiments/2026-11-12-analytic-depth-parity.md`。
