#!/usr/bin/env python3
"""exp-swa-calib-20261110 汇总：读各臂 eval.json，写 summary.md（随时可跑，缺的臂标「未完成」）。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py
口径：success 取 eval.json 的 per_episode.success（走廊口径 sim09.v2.coast1s.corridor3m，已含 |y|<=3 m）。
配对：同一 (训练种子, 评测 seed) 上两个变体各一集，精确 McNemar（双侧）。
"""
import json
import math
import os
from math import comb

OUT = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["best", "calswa", "calbest", "swa"]          # 臂名 = <变体>-s<训练种子>
LABEL = {"best": "现行 ckpt_best", "calswa": "末 10 轮平均 + 校准", "calbest": "ckpt_best + 校准", "swa": "末 10 轮平均（不校准）"}
TRAIN_SEEDS = [0, 1, 2]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_arm(name):
    """-> (eval.json 顶层, {评测 seed: per_episode 行}) 或 None。"""
    path = os.path.join(OUT, name, "eval.json")
    if not os.path.exists(path):
        return None
    e = json.load(open(path, encoding="utf-8"))
    return e, {int(x["seed"]): x for x in e.get("per_episode", [])}


def outcome(x):
    """单集归类：成功 / 碰撞 / 越界（无碰撞、出过走廊，不论是否过线）/ 未到线（留在走廊内但步数用完）。"""
    if x.get("success"):
        return "success"
    if x.get("collision"):
        return "collision"
    if x.get("corridor_violation"):
        return "corridor"
    return "not_reached"


def lr_flips(ep_dir):
    """执行动作里相邻 YAW_LEFT<->YAW_RIGHT 的次数（左右来回转）。"""
    p = os.path.join(ep_dir, "actions.jsonl")
    if not os.path.exists(p):
        return None
    seq = [json.loads(l).get("action_executed") for l in open(p, encoding="utf-8") if l.strip()]
    return sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {2, 3})


def main():
    arms = {}
    for v in VARIANTS:
        for s in TRAIN_SEEDS:
            arms[(v, s)] = load_arm(f"{v}-s{s}")
    lines = ["# exp-swa-calib-20261110：ckpt_best 对「权重平均 + 类先验校准」闭环对照", "",
             "口径 `sim09.v2.coast1s.corridor3m`（成功 = 无碰撞、过成功线、全程 |y| ≤ 3 m）；20 个冻结评测 seed；"
             "hold 1.0，observation，cf_sitl lib31，E1 MINI，80 步上限。预注册见 `run.sh` 文件头。", "",
             "## 逐臂", "",
             "| 臂 | 完成集 | 走廊内成功 | 碰撞 | 越界 | 未到线 | 失败集(基础设施) | SPL | 动作占比 S/F/L/R | 每集 L↔R 换向 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (v, s), got in arms.items():
        name = f"{v}-s{s}"
        if got is None:
            lines.append(f"| {name} | 未完成 | | | | | | | | |")
            continue
        e, per = got
        cnt = {"success": 0, "collision": 0, "corridor": 0, "not_reached": 0}
        flips = []
        for x in per.values():
            cnt[outcome(x)] += 1
            # episode_dir 是相对 eval 目录的路径（episodes/seed_N）
            f = lr_flips(os.path.join(OUT, name, x.get("episode_dir") or os.path.join("episodes", f"seed_{x['seed']}")))
            if f is not None:
                flips.append(f)
        ad = e.get("action_distribution", {})
        tot = max(sum(ad.values()), 1)
        share = "/".join(f"{ad.get(k, 0) / tot:.2f}" for k in ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT"))
        spl = e.get("spl")
        lines.append(f"| {name} | {len(per)} | {cnt['success']} | {cnt['collision']} | {cnt['corridor']} | {cnt['not_reached']} | "
                     f"{e.get('failed_count', '?')} | {spl if spl is None else round(spl, 2)} | {share} | "
                     f"{(sum(flips) / len(flips)) if flips else float('nan'):.1f} |")
    lines += ["", "## 按变体合并三个训练种子", "",
              "| 变体 | 配对集数 | 走廊内成功 | Wilson 95% | 碰撞 | 越界 | 未到线 | 三个训练种子各自的成功数 | 极差 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    pooled = {}
    for v in VARIANTS:
        rows, per_seed = {}, []
        for s in TRAIN_SEEDS:
            got = arms[(v, s)]
            if got is None:
                per_seed.append(None)
                continue
            _e, per = got
            per_seed.append(sum(1 for x in per.values() if x.get("success")))
            for es, x in per.items():
                rows[(s, es)] = x
        pooled[v] = rows
        if not rows:
            lines.append(f"| {LABEL[v]} | 未完成 | | | | | | | |")
            continue
        cnt = {"success": 0, "collision": 0, "corridor": 0, "not_reached": 0}
        for x in rows.values():
            cnt[outcome(x)] += 1
        lo, hi = wilson(cnt["success"], len(rows))
        done = [k for k in per_seed if k is not None]
        lines.append(f"| {LABEL[v]} | {len(rows)} | {cnt['success']} | [{lo:.2f}, {hi:.2f}] | {cnt['collision']} | {cnt['corridor']} | "
                     f"{cnt['not_reached']} | {' / '.join('-' if k is None else str(k) for k in per_seed)} | "
                     f"{(max(done) - min(done)) if len(done) == len(TRAIN_SEEDS) else '未齐'} |")
    lines += ["", "## 配对检验（精确 McNemar；b = 前者成后者败，c = 前者败后者成）", "",
              "| A 对 B | 共同配对 | b | c | p |", "| --- | --- | --- | --- | --- |"]
    for a, b_ in (("calswa", "best"), ("calbest", "best"), ("swa", "best"), ("calswa", "calbest"), ("calswa", "swa")):
        keys = sorted(set(pooled[a]) & set(pooled[b_]))
        if not keys:
            lines.append(f"| {LABEL[a]} 对 {LABEL[b_]} | 0 | | | |")
            continue
        b = sum(1 for k in keys if pooled[a][k].get("success") and not pooled[b_][k].get("success"))
        c = sum(1 for k in keys if not pooled[a][k].get("success") and pooled[b_][k].get("success"))
        lines.append(f"| {LABEL[a]} 对 {LABEL[b_]} | {len(keys)} | {b} | {c} | {mcnemar_exact(b, c):.3f} |")
    lines += ["", "说明：60 个配对在同一评测 seed 上不独立（三个训练种子共用 20 个几何），p 值只作参考；"
              "结论看方向与失败类型的变化。"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
