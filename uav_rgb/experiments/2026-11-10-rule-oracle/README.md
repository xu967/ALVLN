# 2026-11-10 规则层上限实验（rule-oracle）

计划 `docs/PLAN_2026-11-10_phase11.md` §4、§8 的 D1、D3：先量规则层本身的天花板，再决定要不要训练几何头。
零训练：柱子真值解析算出「完美的列距离」，接固定规则 v0，在固件在环的 Gazebo 闭环里评。

## 问什么

规则层输入是完美列距离时，20 个冻结评测 seed 上能走多远。教师 A* 20/20，学生最好配置（calswa + 航向护栏）50/60，剩下的失败全是撞柱。
- 规则 v0 + 完美列距离达到预测（rule60 走廊内成功 >= 54/60，撞柱 <= 3/60）：规则层够用，下一步训练只学列距离（D3 的 geometry head）。
- 低于 48/60：规则层表达力不够，先给规则层加短期记忆（最近 3 步列距离按位移转到当前机体系）再评。
- rule81：同一规则、同一相机，只把列距离的视场从 60° 放宽到 81°（真机 GC4653 水平视场，27 列），看侧擦是否减少。

预注册（预测、证伪、主比较）写在 `run.sh` 文件头。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 守护进程入口：flock 互斥、冻结代码（`git archive` 指定提交）、2 路并行、6 个臂（rule60 / rule81 各重复 3 次）、续跑、DONE |
| `summarize.py` | 汇总：走廊内成功、撞柱、越界、未到线、SPL、动作占比、换向、规则条目触发占比、撞柱集逐集（被撞柱、方位、撞前 3 步列距离），并列教师与学生参照行 |

规则实现在 `sim/tools/rule_policy.py`，接线在 `sim/tools/collect.py`（`--policy rule-oracle`、`--rule-hfov-deg`）与 `sim/tools/eval_closed_loop.py`。

## 运行

```bash
D=/root/rivermind-data/uav_rgb; mkdir -p $D/exp-rule-oracle-20261110 && cd $D/exp-rule-oracle-20261110
cp <仓库>/experiments/2026-11-10-rule-oracle/{run.sh,summarize.py} .
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
tail -f run.log                       # 进度；每臂 20 集，约 1 分钟一集
python summarize.py                   # 随时可跑，用 .venv-tools 的解释器；结果写 summary.md
kill -- -$(cat run.pid)               # 停：只按进程组
```

输出：`$D/exp-rule-oracle-20261110/{rule60,rule81}-rep{0,1,2}/`（每臂一个 eval.json 与 episodes/seed_N/{actions.jsonl,oracle.jsonl,episode_manifest.json}；
actions.jsonl 每步多记 `rule_cols`（各列距离，2 位小数）与 `rule_entry`（触发的规则条目））。

## 结果

实验结束后补 `RESULT.md`。
