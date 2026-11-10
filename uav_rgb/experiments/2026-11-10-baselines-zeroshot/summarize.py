#!/usr/bin/env python3
"""exp-baselines-zeroshot-20261110 汇总：公开权重的零样本基线在柱阵闭环里的走廊内成功、失败类型、动作占比，写 summary.md。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py   （随时可跑，没评完的臂标「未完成」）
口径：success 取 eval.json 的 per_episode.success（sim09.v2.coast1s.corridor3m，已含 |y|<=3 m）；分母 = 请求集数 20（运行失败的集按失败算）。
参照臂不重跑，直接读今天同口径的 eval.json（路径在下面 REFS，写进 summary.md）。预注册见 run.sh 文件头。
"""
import json
import math
import os
from math import comb

OUT = os.path.dirname(os.path.abspath(__file__))   # 本脚本被复制到实验输出目录里运行；各臂的子目录与 BASELINE_LOG 都在这里

# ---------------------------------------------------------------- 配置（改这里）
D = "/root/rivermind-data/uav_rgb"                 # 数据盘根：参照臂的 eval.json 在这下面
# 臂：目录名 -> (表里的名字, 参数量, 映射说明)。顺序 = run.sh 的优先级。dronet 没有护栏臂：见 run.sh 文件头「起跑前修订」①。
ARMS = [
    ("pulp_v3", "PULP v3 ResBlock-1.0 零样本", "320,928", "S=90 °/s，p>0.7 刹停，门槛 12.5°"),
    ("pulp_tiny", "Tiny-PULP v3 dw-pw-0.125 零样本", "2,784", "同上"),
    ("pulp_v3-guard2", "PULP v3 零样本 + 护栏 2", "320,928", "同 pulp_v3，评测端加 --heading-guard 2"),
    ("pulp_tiny-guard2", "Tiny-PULP v3 零样本 + 护栏 2", "2,784", "同 pulp_tiny，评测端加 --heading-guard 2"),
    ("dronet", "DroNet 零样本", "约 32 万（321,634 个张量元素，含 BN 统计）", "S=20 °/s，p>0.7 刹停，门槛 12.5°"),
    ("dronet-recal", "DroNet 重标定（次要）", "约 32 万", "阈值 = 校准帧碰撞概率中位数，S=90 °/s（两处偏离原文）"),
]
# 参照（不重跑）：名字 -> 该参照的各个 eval 目录（多个训练种子合并）；PAIR_REF 是配对检验用的那一个
REFS = {
    "学生 calswa（无护栏，3 个训练种子）": [f"{D}/exp-swa-calib-20261110/calswa-s{s}" for s in (0, 1, 2)],
    "学生 calswa + 护栏 2（3 个训练种子）": [f"{D}/exp-heading-guard-20261110/guard2-calswa-s{s}" for s in (0, 1, 2)],
    "学生 calswa-s1 + 护栏 2（配对检验的参照）": [f"{D}/exp-heading-guard-20261110/guard2-calswa-s1"],
    "教师 A*（11-07 eval4，sim09.v2.coast1s，无走廊判据）": [f"{D}/e1-pipeline-20261107-0417-eval4/eval-teacher"],
    "盲前进（11-07 eval4，sim09.v2.coast1s，无走廊判据）": [f"{D}/e1-pipeline-20261107-0417-eval4/eval-forward"],
}
PAIR_REF = "学生 calswa-s1 + 护栏 2（配对检验的参照）"
PRED_MAX = 6      # 预注册预测：每个零样本臂走廊内成功不超过 6/20
FALSIFY_MIN = 12  # 预注册证伪：任一零样本臂达到 12/20
# ----------------------------------------------------------------------------------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_eval(edir):
    """-> (eval 字典, {seed: 行}) ；行含 outcome / 各动作步数 / 护栏改写步数。没有完整 eval.json 返回 (None, {})。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None, {}
    e = json.load(open(path, encoding="utf-8"))
    if e.get("partial"):
        return None, {}
    rows = {}
    for x in e.get("per_episode", []):
        ep = os.path.join(edir, x.get("episode_dir") or os.path.join("episodes", f"seed_{x['seed']}"))
        acts = read_jsonl(os.path.join(ep, "actions.jsonl"))
        executed = [a.get("action_executed") for a in acts]
        success = bool(x.get("success"))
        outcome = ("success" if success else "collision" if x.get("collision")
                   else "corridor" if x.get("corridor_violation") else "not_reached")
        rows[int(x["seed"])] = {
            "success": success, "outcome": outcome, "steps": len(executed), "spl": x.get("spl"),
            "act": [sum(1 for a in executed if a == k) for k in range(4)],
            "guard": sum(1 for a in acts if a.get("executed_by") == "heading_guard"),
        }
    return e, rows


def merge(dirs):
    rows, evals = {}, []
    for i, d in enumerate(dirs):
        e, r = load_eval(d)
        evals.append(e)
        for s, v in r.items():
            rows[(i, s)] = v
    return evals, rows


def baseline_log_stats(name):
    """BASELINE_LOG（run.sh 给每个臂单独一个文件）里的闭环 steer / p_coll：只看分布，不含预热帧（n==0）。"""
    path = os.path.join(OUT, f"{name}.baseline.jsonl")
    rows = [r for r in read_jsonl(path) if r.get("n", 1) > 0 and "p_coll" in r]
    if not rows:
        return None
    p = sorted(r["p_coll"] for r in rows)
    s = sorted(r["steer"] for r in rows)
    q = lambda v, f: v[min(int(f * len(v)), len(v) - 1)]
    return {"n": len(rows), "p_med": q(p, 0.5), "p_gt07": sum(1 for x in p if x > 0.7) / len(p),
            "steer_med": q(s, 0.5), "steer_p10": q(s, 0.1), "steer_p90": q(s, 0.9)}


def progress_line(name):
    path = os.path.join(OUT, f"{name}.log")
    if not os.path.exists(path):
        return "未开始"
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace") if l.strip()]
    return lines[-1][:120] if lines else "已起（日志为空）"


def main():
    data = {}
    for key, *_ in ARMS:
        ev, rows = merge([os.path.join(OUT, key)])
        data[key] = (ev[0], rows)
    refs = {name: merge(dirs) for name, dirs in REFS.items()}
    ref_rows = refs[PAIR_REF][1]
    requested = 20

    L = ["# exp-baselines-zeroshot-20261110：公开权重零样本基线在柱阵闭环里的表现", "",
         "口径 `sim09.v2.coast1s.corridor3m`；20 个冻结评测 seed（test 分区）；hold 1.0，observation，cf_sitl lib31，E1 MINI，80 步上限。"
         "**零样本**：公开权重原样使用，没有在我们的数据上训练或微调；输出经 `sim/tools/baselines/baseline_infer.py` 映射成 4 个动作。"
         "参数量对照：我们的学生约 10 万。预注册见 `run.sh` 文件头。", "",
         "## 主表", "",
         "| 臂 | 参数量 | 映射 | 集数 | 走廊内成功 | Wilson 95% | 撞柱 | 越界 | 未到线 | 运行失败 | SPL | SAFE | FORWARD | YAW | 护栏改写占比 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    best = {}
    for key, label, params, mapping in ARMS:
        ev, rows = data[key]
        if ev is None:
            L.append(f"| {label} | {params} | {mapping} | 未完成（{progress_line(key)}） | | | | | | | | | | | |")
            continue
        cnt = {k: sum(1 for r in rows.values() if r["outcome"] == k) for k in ("success", "collision", "corridor", "not_reached")}
        lo, hi = wilson(cnt["success"], requested)
        steps = max(sum(r["steps"] for r in rows.values()), 1)
        act = [sum(r["act"][k] for r in rows.values()) / steps for k in range(4)]
        guard = sum(r["guard"] for r in rows.values()) / steps
        spl = ev.get("spl")
        best[key] = cnt["success"]
        L.append(f"| {label} | {params} | {mapping} | {len(rows)}/{requested} | {cnt['success']}/{requested} | [{lo:.2f}, {hi:.2f}] | "
                 f"{cnt['collision']} | {cnt['corridor']} | {cnt['not_reached']} | {ev.get('failed_count', 0)} | "
                 f"{'-' if spl is None else f'{spl:.2f}'} | {act[0]:.3f} | {act[1]:.3f} | {act[2] + act[3]:.3f} | {guard:.3f} |")
    L += ["", "动作占比 = 全部集、全部步的 `action_executed`（含护栏改写后的动作），分母是实际执行的步数；"
          "FORWARD 占比低说明「原地不动」，高而成功少说明「飞了但撞或偏」。SPL 只在算出最短路的集上平均（分母见 eval.json 的 spl_n）。", ""]

    L += ["## 参照（不重跑，读已有 eval.json）", "", "| 参照 | 集数 | 走廊内成功 | Wilson 95% | 撞柱 | 越界 | 来源 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, (evals, rows) in refs.items():
        if not rows:
            L.append(f"| {name} | 缺 eval.json | | | | | {', '.join(REFS[name])} |")
            continue
        n = len(rows)
        cnt = {k: sum(1 for r in rows.values() if r["outcome"] == k) for k in ("success", "collision", "corridor")}
        lo, hi = wilson(cnt["success"], n)
        L.append(f"| {name} | {n} | {cnt['success']} | [{lo:.2f}, {hi:.2f}] | {cnt['collision']} | {cnt['corridor']} | "
                 f"{', '.join(os.path.relpath(p, D) for p in REFS[name])} |")
    L += ["", "教师与盲前进来自 11-07 eval4（20 个同 seed，指标 sim09.v2.coast1s，没有走廊判据；教师不越界、盲前进不过线，走廊判据不改变这两个数）。", ""]

    ref_ok = sum(1 for r in ref_rows.values() if r["success"])
    L += ["## 预注册检查", "", f"- 预测：每个零样本臂走廊内成功不超过 {PRED_MAX}/20，且低于学生 calswa-s1 + 护栏（{ref_ok}/20，见下表配对）。",
          f"- 证伪：任一零样本臂达到 {FALSIFY_MIN}/20 以上。"]
    done = {k: v for k, v in best.items() if k != "dronet-recal"}
    if done:
        over = {k: v for k, v in done.items() if v > PRED_MAX}
        fals = {k: v for k, v in done.items() if v >= FALSIFY_MIN}
        L.append(f"- 已完成臂（不含次要的重标定臂）：{', '.join(f'{k} {v}/20' for k, v in done.items())}。"
                 f"超过预测上限的臂：{', '.join(over) or '无'}；触发证伪的臂：{', '.join(fals) or '无'}。")
    L.append("")

    L += ["## 配对检验（精确 McNemar；b = 前者成后者败，c = 前者败后者成；同 seed 20 对，运行失败的集记失败）", "",
          "| A 对 B | 共同配对 | b | c | p |", "| --- | --- | --- | --- | --- |"]
    def paired(a_rows, b_rows):
        seeds = sorted({s for (_, s) in a_rows} | {s for (_, s) in b_rows})
        sa = {s: a_rows.get((0, s), {}).get("success", False) for s in seeds}
        sb = {s: b_rows.get((0, s), {}).get("success", False) for s in seeds}
        b = sum(1 for s in seeds if sa[s] and not sb[s])
        c = sum(1 for s in seeds if not sa[s] and sb[s])
        return len(seeds), b, c
    for key, label, *_ in ARMS:
        if data[key][0] is None:
            continue
        n, b, c = paired(data[key][1], ref_rows)
        L.append(f"| {label} 对 学生 calswa-s1 + 护栏 | {n} | {b} | {c} | {mcnemar_exact(b, c):.3g} |")
    for g, base in (("pulp_v3-guard2", "pulp_v3"), ("pulp_tiny-guard2", "pulp_tiny")):   # dronet 没有护栏臂
        if data[g][0] is None or data[base][0] is None:
            continue
        n, b, c = paired(data[g][1], data[base][1])
        L.append(f"| {g} 对 {base}（护栏的作用） | {n} | {b} | {c} | {mcnemar_exact(b, c):.3g} |")
    L += ["", "n=20 的单次比较不下结论：只有差距很大才有检出力；p 值只作参考。零样本臂只说明公开权重在我们的画面上的域差异，不能用来证明网络结构的优劣。", ""]

    L += ["## 闭环里基线自己的输出分布（BASELINE_LOG，含全部步，不含预热帧）", "",
          "| 臂 | 记录步数 | p_coll 中位数 | p_coll > 0.7 的占比 | steer 中位数 | steer p10 | steer p90 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for key, label, *_ in ARMS:
        st = baseline_log_stats(key)
        if st:
            L.append(f"| {label} | {st['n']} | {st['p_med']:.3f} | {st['p_gt07']:.3f} | {st['steer_med']:.3f} | {st['steer_p10']:.3f} | {st['steer_p90']:.3f} |")
    L.append("")
    text = "\n".join(L)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
