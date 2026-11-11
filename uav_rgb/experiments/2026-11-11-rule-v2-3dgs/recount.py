#!/usr/bin/env python3
"""独立复算：只读各集 metrics.json，不引用 summarize.py，复核 summary.json 的主要数字。

用法：python3 recount.py [OUT] [K1]
  OUT  本实验输出目录（缺省 $D/exp-rule-v2-3dgs-20261111）
  K1   K1 输出目录（缺省 $D/exp-rule-oracle-3dgs-20261110，只用来取室外 v1 的逐集结果）
退出码：0 全部一致；1 有不一致；2 输入缺失。
"""
import glob
import json
import math
import os
import sys

D = os.environ.get("D", "/root/rivermind-data/uav_rgb")
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{D}/exp-rule-v2-3dgs-20261111"
K1 = sys.argv[2] if len(sys.argv) > 2 else f"{D}/exp-rule-oracle-3dgs-20261110"
V1, V2, V2D, FWD = "rule-occ+route", "rule2-occ+route", "rule2-depth+route", "forward+route"


def episodes(root):
    """{(scene, route, seed): metrics}，读 <root>/<scene>/r*/ep_*/metrics.json。"""
    eps = {}
    for p in sorted(glob.glob(f"{root}/*/r*/ep_*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        key = (m["scene"], m["route"], m["seed"])
        if key in eps:
            sys.exit(f"重复的键 {key}: {p}")
        eps[key] = m
    return eps


def mcnemar_p(b, c):
    """精确双侧 McNemar（二项，p=0.5）。"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def rate(eps, field):
    return sum(1 for m in eps.values() if m[field])


bad = []


def check(name, mine, theirs, tol=1e-9):
    ok = abs(mine - theirs) <= tol
    print(f"  {'OK  ' if ok else 'DIFF'} {name}: 复算 {mine!r}  summary.json {theirs!r}")
    if not ok:
        bad.append(name)


def main():
    if not os.path.isfile(f"{OUT}/summary.json"):
        sys.exit(2)
    summ = json.load(open(f"{OUT}/summary.json"))
    data = {}
    for cls, arms in (("holdout", (FWD, V1, V2, V2D)), ("formal40", (V1, V2, V2D)), ("outdoor", (V2, V2D))):
        for arm in arms:
            data[cls, arm] = episodes(f"{OUT}/{cls}/{arm}")
    data["outdoor", "K1:" + V1] = episodes(f"{K1}/formal/outdoor/{V1}")

    for cls in ("holdout", "formal40", "outdoor"):
        print(f"[{cls}] 各臂")
        for arm in summ["classes"][cls]["arms"]:
            eps = data[cls, arm]
            ov = summ["classes"][cls]["overall"][arm]
            print(f" {arm}: n={len(eps)} 到点={rate(eps, 'success_stop')} 碰撞={rate(eps, 'collision')} "
                  f"撞在前={rate(eps, 'collide_first')} 超时={rate(eps, 'timeout')}")
            check(f"{cls}/{arm} n", len(eps), ov["n"])
            check(f"{cls}/{arm} 到点数", rate(eps, "success_stop"), ov["k_stop"])
    for cls, a_arm, b_arm, ref in (("holdout", V2, V1, "pairs"), ("formal40", V2, V1, "pairs"), ("outdoor", V2, "K1:" + V1, "pairs")):
        a, b_ = data[cls, a_arm], data[cls, b_arm]
        if set(a) != set(b_):
            print(f"  DIFF {cls} 两臂的键集合不同")
            bad.append(f"{cls} keys")
            continue
        wins = sum(1 for k in a if a[k]["success_stop"] and not b_[k]["success_stop"])
        loss = sum(1 for k in a if not a[k]["success_stop"] and b_[k]["success_stop"])
        n = len(a)
        p = mcnemar_p(wins, loss)
        print(f"[{cls}] {a_arm} vs {b_arm}: 配对 {n}，A 成 B 败 {wins}，A 败 B 成 {loss}，Δ={(wins - loss) / n:+.4f}，p={p:.3g}")
        rec = next(x for x in summ["pairs"][cls] if x["b"] == b_arm and x["a"] == a_arm)
        check(f"{cls} A成B败 b", wins, rec["b_count"])
        check(f"{cls} A败B成 c", loss, rec["c_count"])
        check(f"{cls} p", p, rec["p"], 1e-12)
        if cls == "outdoor":
            # 结论字段按室外判据命名：b_v1_only = v1 成 v2 败（A = v2，所以是 loss），c_v2_only = v2 成 v1 败
            check("室外 verdict.b_v1_only", loss, summ["verdict"]["outdoor"]["b_v1_only"])
            check("室外 verdict.c_v2_only", wins, summ["verdict"]["outdoor"]["c_v2_only"])

    h = data["holdout", V2]
    n = len(h)
    coll, cf = rate(h, "collision") / n, rate(h, "collide_first") / n
    wins = sum(1 for k in h if h[k]["success_stop"] and not data["holdout", V1][k]["success_stop"])
    loss = sum(1 for k in h if not h[k]["success_stop"] and data["holdout", V1][k]["success_stop"])
    delta, p = (wins - loss) / n, mcnemar_p(wins, loss)
    verdict = ("达到预测" if delta >= 0.10 and p < 0.05 and coll <= 0.10 and cf <= 0.08
               else "证伪" if delta < 0.05 or coll > 0.15 else "介于")
    print(f"[主终点] Δ={delta:+.4f} p={p:.3g} 碰撞={coll:.4f} 撞在前={cf:.4f} -> {verdict}")
    if verdict != summ["verdict"]["primary"]["verdict"]:
        bad.append("主终点结论")
    o = data["outdoor", V2]
    ob = sum(1 for k in o if data["outdoor", "K1:" + V1][k]["success_stop"] and not o[k]["success_stop"])
    oc = sum(1 for k in o if o[k]["success_stop"] and not data["outdoor", "K1:" + V1][k]["success_stop"])
    osr, ocoll = rate(o, "success_stop") / len(o), rate(o, "collision") / len(o)
    ok = osr >= 0.93 and ob - oc <= 5 and ocoll <= 0.10
    print(f"[室外] SR={osr:.4f} (v1成v2败){ob}-(v2成v1败){oc}={ob - oc} 碰撞={ocoll:.4f} -> {'不劣' if ok else '劣'}")
    if ok != (summ["verdict"]["outdoor"]["verdict"] == "不劣"):
        bad.append("室外结论")
    print("全部一致" if not bad else f"不一致：{bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
