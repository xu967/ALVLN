# 2026-11-10 规则层上限实验第二轮（rule-oracle v1）

计划 `docs/PLAN_2026-11-10_phase11.md` §4、§8 的 D3 → D6：规则 v0（`experiments/2026-11-10-rule-oracle`）在 test 上判负后，给规则层加短期障碍记忆再评。
零训练：柱子真值解析算出「完美的列距离」，接固定规则 v1，在固件在环的 Gazebo 闭环里评。

## 问什么

v0 的失败集几乎都是 80 步跑满、`max_x` 停在 1.3–1.5 m（门柱前）、无碰撞：回转例外分支恒走 SAFE_NEUTRAL，悬停（v0 报告 §5.1 事前指出）。
v1 把列距离变成机体系障碍点，按真值位姿存进世界系保留 8 步，每步在「转回一步 / 保持 / 再转开」里用记忆点判断通道是否畅通，永不输出 SAFE_NEUTRAL。

- rule60v1 三次重复合并 60 集，走廊内成功 >= 54/60 且撞柱 <= 3/60：「真值列距离 + 固定规则」够用。
- 低于 48/60：这个组合在 60° 视场下就不够，计划的 D 线收窄为只把几何头当辅助监督。
- 次要：rule81v1 撞柱 <= rule60v1；长课程 K=3 整条课程成功 >= 16/20（每块存活 0.93 的独立模型给 0.80），逐块存活另给。

预注册（预测、证伪、主比较、起跑前已知）写在 `run.sh` 文件头。规则参数冻结在 `sim/tools/rule_policy.py` 的 `RULE_V1`，没有在 test seed 上调。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 守护进程入口：flock 互斥、等 v0 的 DONE 与冒烟结束（gz 配额）、冻结代码（`git archive` 指定提交）、2 路并行、7 个臂（rule60v1 ×3、rule60v1-course3、rule81v1 ×3）、续跑、DONE |
| `summarize.py` | 汇总：走廊内成功、撞柱、越界、未到线、SPL、动作占比、换向、最大 \|n\|、V1 条目占比、记忆点数、长课程逐块存活、撞柱集逐集、失败集末 10 步（条目、n、三个 clear），与 v0、学生的并列与配对 McNemar |

规则实现在 `sim/tools/rule_policy.py`（`RULE_V1`、`obstacle_points`、`points_to_world`、`points_to_body`、`update_memory`、`corridor_clear`、`rule_v1_entry`、`rule_v1_step`），
接线在 `sim/tools/collect.py`（`--rule-version {v0,v1}`，记忆列表由主循环按集清零）与 `sim/tools/eval_closed_loop.py`；离线运动学模拟在 `sim/tools/rule_sim.py`
（`python sim/tools/rule_sim.py --rule v1 --count 20` 即 train 分区 20 个几何的离线检查）。

## 运行

```bash
D=/root/rivermind-data/uav_rgb; mkdir -p $D/exp-rule-oracle-v1-20261110 && cd $D/exp-rule-oracle-v1-20261110
cp <仓库>/experiments/2026-11-10-rule-oracle-v1/{run.sh,summarize.py} .
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
tail -f run.log                       # 进度；每臂 20 集，约 1–2.5 分钟一集（长课程更长）
python summarize.py                   # 随时可跑，用 .venv-tools 的解释器；结果写 summary.md
kill -- -$(cat run.pid)               # 停：只按进程组
```

输出：`$D/exp-rule-oracle-v1-20261110/{rule60v1-rep0..2,rule60v1-course3,rule81v1-rep0..2}/`（每臂一个 eval.json 与 episodes/seed_N/{actions.jsonl,oracle.jsonl,episode_manifest.json}；
actions.jsonl 每步多记 `rule_cols`、`rule_entry`、`rule_mem_n`（记忆点数）与 `rule_clear`（[左转一步, 保持, 右转一步] 三个通道是否畅通））。

## 结果

实验结束后补 `RESULT.md`。
