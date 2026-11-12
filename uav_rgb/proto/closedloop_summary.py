#!/usr/bin/env python3
"""卡 E2E-P0 验收 (b)：原型训练出的权重经平均与校准后在 Gazebo 闭环的成功数，对 W1-0 第二轮三个 K=1 臂（53/60）。

预注册（任务卡）：原型三个训练种子 × 20 个冻结 seed = 60 集，成功数与第二轮 acc-L0.20-k1-s{0,1,2} 的 53/60 相差不超过 6 集；
超过 6 集说明两条线的训练不等价，停下来查，不再继续用。判据只看总成功数；逐种子、逐集配对与 McNemar 只报告。

含迟到步的集按第二轮条款 R6：基础设施失败，首次不计成功；<臂>-retry1 里重跑干净则以重跑为准；重跑仍迟到、重跑中止或缺失，按没成功。
读 eval_closed_loop.py 的输出：<臂>/eval.json 的 per_episode（seed、success）、late_dispatch_episodes、failed。

用法（标准库即可）：
  python proto/closedloop_summary.py --proto s0=<臂目录> --proto s1=... --proto s2=... \\
      --ref s0=<R2 臂目录> --ref s1=... --ref s2=... --seeds 1,113,... [--tol 6] --out-json <json> --out-md <md>
  名字相同的 --proto 与 --ref 配成一对（同一个训练种子）。
退出码：0 总成功数差在容差内且 60 集都有结果；1 差超过容差；2 输入不全（读不到 eval.json 或有集没结果，且差在容差内时也返回 2）。
"""
import argparse
import json
import math
import os
import sys

TOL = 6                                         # 任务卡预注册：与第二轮 K=1 三个臂的成功数相差不超过 6 集
RETRY_SUFFIX = "-retry1"                        # 与 run_r2.sh 一致

SUCCESS, FAIL, LATE, FAILED, MISSING = "success", "fail", "late", "failed", "missing"
SYMBOL = {SUCCESS: "✓", FAIL: "✗", LATE: "迟到", FAILED: "中止", MISSING: "缺"}


def read_eval(arm_dir):
    path = os.path.join(arm_dir, "eval.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_pass(ev):
    """一份 eval.json 里每个 seed 的结果：{seed: success|fail|late|failed}。"""
    out = {}
    for e in ev.get("per_episode", []):
        out[int(e["seed"])] = SUCCESS if e.get("success") else FAIL
    late = {int(x["seed"]) for x in ev.get("late_dispatch_episodes") or []}
    for x in ev.get("failed") or []:
        s = int(x["seed"])
        out[s] = LATE if s in late else FAILED
    for s in late:
        out[s] = LATE
    return out


def arm_outcomes(arm_dir, seeds):
    """按 R6 合并首次与重跑。返回 (outcomes {seed: 结果}, 明细 dict)；arm 没有 eval.json 返回 (None, ...)。"""
    ev = read_eval(arm_dir)
    if ev is None:
        return None, {"state": "没有 eval.json"}
    first = first_pass(ev)
    retry_ev = read_eval(arm_dir.rstrip("/") + RETRY_SUFFIX)
    retry = first_pass(retry_ev) if retry_ev is not None else {}
    out, how = {}, {}
    for s in seeds:
        r = first.get(s, MISSING)
        if r == LATE:                                 # R6：首次迟到的集以重跑为准；重跑仍迟到 / 中止 / 缺失都不算成功
            out[s] = retry.get(s, MISSING)
            how[s] = "retry"
        else:
            out[s] = r
            how[s] = "first"
    return out, {"state": "完成" if not ev.get("partial") else "partial", "episodes_done": ev.get("episodes_done"),
                 "failed_count": ev.get("failed_count"), "first_late_seeds": sorted(s for s in seeds if first.get(s) == LATE),
                 "how": how, "has_retry_dir": retry_ev is not None}


def counts(outcomes):
    c = {k: 0 for k in (SUCCESS, FAIL, LATE, FAILED, MISSING)}
    for v in outcomes.values():
        c[v] += 1
    return c


def mcnemar_exact(b, c):
    """双侧精确 McNemar（二项检验，p=0.5）：b、c 为两种不一致格的数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def read_cg(path):
    """<臂>.cg 两行 start/end：epoch usage_usec nr_throttled throttled_usec → 平均核数与限流增量；不全返回 None。"""
    if not os.path.isfile(path):
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 5 and p[0] in ("start", "end"):
                rows[p[0]] = [float(x) for x in p[1:]]
    if "start" not in rows or "end" not in rows:
        return None
    s, e = rows["start"], rows["end"]
    wall = e[0] - s[0]
    return {"wall_s": wall, "avg_cores": (e[1] - s[1]) / 1e6 / wall if wall > 0 else None, "throttled": int(e[2] - s[2])}


def summarize(proto, ref, seeds, tol=TOL):
    names = list(proto)
    res = {"seeds": list(seeds), "tolerance": tol, "arms": {}, "pairs": {}}
    tot_p = tot_r = 0
    complete = True
    for n in names:
        po, pd = arm_outcomes(proto[n], seeds)
        ro, rd = arm_outcomes(ref[n], seeds) if n in ref else (None, {"state": "没有给参照臂"})
        if po is None or ro is None:
            res["arms"][n] = {"proto_state": pd.get("state"), "ref_state": rd.get("state")}
            complete = False
            continue
        pc, rc = counts(po), counts(ro)
        a = sum(1 for s in seeds if po[s] == SUCCESS and ro[s] == SUCCESS)
        b = sum(1 for s in seeds if po[s] == SUCCESS and ro[s] != SUCCESS)    # 仅原型成功
        c = sum(1 for s in seeds if po[s] != SUCCESS and ro[s] == SUCCESS)    # 仅参照成功
        d = len(seeds) - a - b - c
        res["arms"][n] = {"proto": {"dir": proto[n], "outcomes": {str(s): v for s, v in po.items()}, "counts": pc, "detail": pd,
                                    "cg": read_cg(proto[n].rstrip("/") + ".cg")},
                          "ref": {"dir": ref[n], "outcomes": {str(s): v for s, v in ro.items()}, "counts": rc, "detail": rd}}
        res["pairs"][n] = {"both": a, "proto_only": b, "ref_only": c, "neither": d, "mcnemar_p": mcnemar_exact(b, c),
                           "proto_successes": pc[SUCCESS], "ref_successes": rc[SUCCESS]}
        tot_p += pc[SUCCESS]
        tot_r += rc[SUCCESS]
        if pc[MISSING] or pc[FAILED] or pc[LATE] or rc[MISSING]:
            complete = False
    n_ep = len(seeds) * len(names)
    delta = tot_p - tot_r
    res["total"] = {"episodes": n_ep, "proto_successes": tot_p, "ref_successes": tot_r, "delta": delta, "within_tolerance": abs(delta) <= tol,
                    "all_arms_present": all("proto" in v for v in res["arms"].values()), "complete": complete}
    if not res["total"]["all_arms_present"]:
        res["verdict"] = "未下判：有臂还没有 eval.json（总数只含已有的臂）"
    elif abs(delta) <= tol:
        res["verdict"] = "预注册通过：总成功数相差 %+d 集，不超过 %d" % (delta, tol)
    else:
        res["verdict"] = "预注册未通过：总成功数相差 %+d 集，超过 %d；两条线的训练不等价，停下来查，不再继续用" % (delta, tol)
    return res


def render_md(res):
    seeds = res["seeds"]
    L = ["# 验收 (b)：原型训练权重的闭环成功数对第二轮 K=1 三臂", "",
         f"- 判据（预注册）：{len(res['arms'])} 个训练种子 × {len(seeds)} 个冻结 seed 的总成功数，与第二轮 acc-L0.20-k1-s* 相差不超过 {res['tolerance']} 集。",
         f"- 结果：原型 {res['total']['proto_successes']}/{res['total']['episodes']}，第二轮 {res['total']['ref_successes']}/{res['total']['episodes']}，"
         f"差 {res['total']['delta']:+d}。**{res['verdict']}**。",
         f"- 数据完整性：{'60 集都有干净结果' if res['total']['complete'] else '有集缺结果、迟到未过重跑或中止（按没成功计），见下表的标注'}。", ""]
    L += ["## 1. 逐训练种子", "", "| 训练种子 | 原型成功 | 第二轮成功 | 两边都成功 | 仅原型成功 | 仅第二轮成功 | 都没成功 | 精确 McNemar p（仅报告） | 原型首次迟到集 | 原型平均核数 / 限流增量 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, p in res["pairs"].items():
        arm = res["arms"][n]["proto"]
        cg = arm.get("cg")
        L.append(f"| {n} | {p['proto_successes']}/{len(seeds)} | {p['ref_successes']}/{len(seeds)} | {p['both']} | {p['proto_only']} | {p['ref_only']} | {p['neither']} | "
                 f"{p['mcnemar_p']:.3f} | {arm['detail'].get('first_late_seeds') or '无'} | "
                 + (f"{cg['avg_cores']:.2f} / {cg['throttled']}" if cg and cg['avg_cores'] is not None else "-") + " |")
    for n, v in res["arms"].items():
        if "proto" not in v:
            L.append(f"| {n} | 未完成（原型：{v.get('proto_state')}；第二轮：{v.get('ref_state')}） | | | | | | | | |")
    L += ["", "## 2. 逐集（原型 / 第二轮）", "", "| seed | " + " | ".join(res["arms"]) + " |", "|---|" + "---|" * len(res["arms"])]
    for s in seeds:
        cells = []
        for n, v in res["arms"].items():
            if "proto" not in v:
                cells.append("-")
                continue
            a, b = v["proto"]["outcomes"][str(s)], v["ref"]["outcomes"][str(s)]
            retried = v["proto"]["detail"]["how"][s] == "retry"
            cells.append(f"{SYMBOL[a]}{'（重跑）' if retried else ''} / {SYMBOL[b]}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += [""]
    return "\n".join(L)


def parse_named(items):
    out = {}
    for it in items or []:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"参数 {it!r} 应写成 名字=目录")
        out[name] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--proto", action="append")
    ap.add_argument("--ref", action="append")
    ap.add_argument("--seeds", required=True, help="冻结 seed，逗号分隔")
    ap.add_argument("--tol", type=int, default=TOL)
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    proto, ref = parse_named(a.proto), parse_named(a.ref)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    if not proto or not seeds:
        print("至少要 1 个 --proto 和 --seeds", file=sys.stderr)
        return 2
    res = summarize(proto, ref, seeds, a.tol)
    md = render_md(res)
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
            f.write("\n")
    if a.out_md:
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    if not res["total"]["all_arms_present"]:
        return 2
    if not res["total"]["within_tolerance"]:
        return 1
    return 0 if res["total"]["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
