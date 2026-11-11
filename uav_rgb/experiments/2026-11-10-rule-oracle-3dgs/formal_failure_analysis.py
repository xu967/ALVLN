#!/usr/bin/env python3
"""正式（或 dev / 冒烟）输出里规则臂失败集的失败形态分析（只读，不重跑，不改任何参数 / 规则 / 预注册）。只用标准库。

输入 <OUT>/formal（含 indoor/ outdoor/ 两类的 config.json 与各臂各路线的集目录）；每个有 rule.jsonl 的臂（rule-*）单独统计。
输出（打印，--out 同时写 markdown）：
  1. 失败形态：没到点的集分 撞在前 / 超时；超时再按「末 20 步严格 more↔cap 两步循环」「目标点停滞」「其他交替」「其他」分（互斥，按此顺序）；
  2. 每步三个候选航向通道全堵（clear == [0,0,0]）的步占比：失败集 vs 到点集，全部步 / 到 nav_end 为止；V1-cap / V1-more 占比；
  3. 撞在前的集逐个列出：碰撞步、碰撞步的动作与规则条目、出生时与碰撞时的飞机到占据栅格距离、出生点到碰撞点距离、碰撞时到 waypoint 距离、
     碰撞前一步 occ / depth 两路正前方 6 列的最小距离（depth 是否看到 occ 看到的障碍）；
  4. 没有一个臂到点的路线（全臂失败）与只有主臂失败的路线。
口径同 dev_failure_analysis.py：clear = [clear(+α), clear(0), clear(-α)]；动作 1 前进 / 2 YAW_LEFT / 3 YAW_RIGHT；
循环 = 末 20 步全是 V1-more / V1-cap 且相邻严格互换；目标点停滞 = 末 20 步目标点不变且末位姿离它不到 0.5 m（只数没到点的集，
到点之后规则臂不停，在 waypoint 附近徘徊，目标点本来就是 waypoint）。
用法：formal_failure_analysis.py <OUT>/formal [--out 结果.md] [--last 20]
"""
import argparse
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


summ = _load("summarize")
dfa = _load("dev_failure_analysis")

FWD, YAW_LEFT, YAW_RIGHT = 1, 2, 3
ACT = {FWD: "前进", YAW_LEFT: "YAW_LEFT", YAW_RIGHT: "YAW_RIGHT"}
CENTER = slice(7, 13)           # 20 列里正前方 6 列（约 ±9°）


def pct(a, b):
    return "—" if not b else "%.1f%%" % (100.0 * a / b)


def load_ep(m):
    d = m["_dir"]
    return {"m": m, "rule": dfa.read_jsonl(os.path.join(d, "rule.jsonl")) or [], "act": dfa.read_jsonl(os.path.join(d, "actions.jsonl")) or []}


def failure_form(ep):
    """没到点的集的形态（互斥）。返回 撞在前 / 2 步循环收尾 / 目标点停滞 / 其他交替 / 其他超时 / 无规则日志。"""
    m = ep["m"]
    if m["success_stop"]:
        return "到点"
    if not ep["rule"]:
        return "无规则日志"
    if m["collide_first"]:
        return "撞在前"
    if dfa.ended_in_cycle(ep):
        return "2 步循环收尾"
    if dfa.target_stalled(ep):
        return "目标点停滞"
    seq = [a["action_executed"] for a in ep["act"]]
    if summ.alt_runs(seq):
        return "其他交替"
    return "其他超时"


FORMS = ["撞在前", "2 步循环收尾", "目标点停滞", "其他交替", "其他超时", "无规则日志"]


def step_stats(eps):
    """一组集合计：全堵 / V1-cap / V1-more 步占比（全部步、到 nav_end）。"""
    out = {"all": {"n": 0, "blk": 0, "cap": 0, "more": 0}, "pre": {"n": 0, "blk": 0, "cap": 0, "more": 0}}
    for ep in eps:
        ne = dfa.nav_end(ep["m"])
        for x in ep["rule"]:
            for w in ("all", "pre"):
                if w == "pre" and x["step"] > ne:
                    continue
                o = out[w]
                o["n"] += 1
                o["blk"] += x["clear"] == [0, 0, 0]
                o["cap"] += x["entry"] == "V1-cap"
                o["more"] += x["entry"] == "V1-more"
    return out


def collision_row(ep):
    """撞在前一集的碰撞位置描述。"""
    m, rule, act = ep["m"], ep["rule"], ep["act"]
    cs = m["collision_step"]
    row = {"scene": m["scene"], "route": m["route"], "episode": m["episode"], "step": cs, "steps": m["steps"]}
    a = next((x for x in act if x["step"] == cs), None)
    r = next((x for x in rule if x["step"] == cs), None)
    r_prev = next((x for x in rule if x["step"] == cs - 1), None)
    row["action"] = ACT.get(a["action_executed"], "?") if a else "?"
    row["entry"] = r["entry"] if r else "?"
    row["clr0"] = act[0].get("clearance_m") if act else None
    row["clr_hit"] = a.get("clearance_m") if a else None
    sp, cp = m.get("start_pose"), a.get("collision_point") if a else None
    row["spawn_to_hit"] = math.hypot(cp[0] - sp["x"], cp[1] - sp["y"]) if (cp and sp) else None
    row["wp_dist"] = m.get("final_wp_dist_m")
    row["occ_c_prev"] = min(r_prev["d_occ"][CENTER]) if r_prev else None
    row["dep_c_prev"] = min(r_prev["d_depth"][CENTER]) if r_prev else None
    row["occ_c"] = min(r["d_occ"][CENTER]) if r else None
    row["dep_c"] = min(r["d_depth"][CENTER]) if r else None
    return row


def analyze(cname, c, last):
    o = []
    P = o.append
    rule_arms = [a for a in c["arms"] if c["arm_cfg"][a].get("base", "").startswith("rule-")]
    by_arm = {a: [load_ep(m) for m in c["rows"] if m["arm"] == a] for a in rule_arms}
    P("## %s\n" % cname)
    # 1 失败形态
    P("### 1. 没到点的集的形态（互斥，按表头顺序判）\n")
    P("| 臂 | 集数 | 到点 | " + " | ".join(FORMS) + " |")
    P("| --- | --- | --- | " + " | ".join("---" for _ in FORMS) + " |")
    forms = {}
    for a in rule_arms:
        f = {}
        for ep in by_arm[a]:
            k = failure_form(ep)
            f[k] = f.get(k, 0) + 1
        forms[a] = f
        P("| `%s` | %d | %d | %s |" % (a, len(by_arm[a]), f.get("到点", 0), " | ".join(str(f.get(k, 0)) for k in FORMS)))
    P("")
    # 2 步占比
    P("### 2. 三通道全堵步占比：失败集 vs 到点集\n")
    P("| 臂 | 组 | 集数 | 全堵步占比（全部步） | V1-cap（全部步） | V1-more（全部步） | 全堵步占比（到 nav_end） | V1-cap（到 nav_end） |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for a in rule_arms:
        for lab, sel in (("到点", [e for e in by_arm[a] if e["m"]["success_stop"]]), ("失败（含撞在前）", [e for e in by_arm[a] if not e["m"]["success_stop"]]),
                         ("超时", [e for e in by_arm[a] if e["m"]["timeout"]])):
            s = step_stats(sel)
            al, pr = s["all"], s["pre"]
            P("| `%s` | %s | %d | %s（%d / %d） | %s | %s | %s（%d / %d） | %s |" % (
                a, lab, len(sel), pct(al["blk"], al["n"]), al["blk"], al["n"], pct(al["cap"], al["n"]), pct(al["more"], al["n"]),
                pct(pr["blk"], pr["n"]), pr["blk"], pr["n"], pct(pr["cap"], pr["n"])))
    P("")
    # 3 撞在前
    P("### 3. 撞在前的集：碰撞位置\n")
    P("碰撞步 = metrics 的 collision_step；「出生 / 碰撞时间隙」= actions.jsonl 的 clearance_m（飞机位姿到原始占据栅格的距离，步 0 与碰撞步）；"
      "正前方 6 列 = 20 列里第 7–12 列（约 ±9°）；occ / depth 取该步或前一步各自的最小列距离，两者差 ≥ 0.3 m 即任务卡定义的漏检。\n")
    for a in rule_arms:
        cf = [e for e in by_arm[a] if e["m"]["collide_first"]]
        P("**`%s`：撞在前 %d / %d 集**\n" % (a, len(cf), len(by_arm[a])))
        if not cf:
            continue
        rows = [collision_row(e) for e in cf]
        n_turn = sum(r["action"] in ("YAW_LEFT", "YAW_RIGHT") for r in rows)
        n_fwd = sum(r["action"] == "前进" for r in rows)
        early = sum(1 for r in rows if r["step"] is not None and r["step"] <= 3)
        P("- 碰撞步的动作：前进 %d、转向 %d；碰撞发生在前 4 步（步号 ≤ 3）的 %d 集；规则条目 %s" % (
            n_fwd, n_turn, early, "，".join("%s %d" % (k, v) for k, v in sorted({e: sum(1 for r in rows if r["entry"] == e) for e in {r["entry"] for r in rows}}.items(), key=lambda kv: -kv[1]))))
        P("")
        P("| 场景 | 路线 | 集 | 碰撞步 / 总步 | 碰撞步动作 | 碰撞步条目 | 出生时间隙 m | 碰撞时间隙 m | 出生点到碰撞点 m | 碰撞时到 waypoint m | 前一步 occ / depth 正前方最小 m | 碰撞步 occ / depth 正前方最小 m |")
        P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        def f2(v):
            return "—" if v is None else "%.2f" % v

        for r in rows:
            P("| %s | %s | %s | %s / %s | %s | %s | %s | %s | %s | %s | %s / %s | %s / %s |" % (
                r["scene"], r["route"], r["episode"], r["step"], r["steps"], r["action"], r["entry"], f2(r["clr0"]), f2(r["clr_hit"]),
                f2(r["spawn_to_hit"]), f2(r["wp_dist"]), f2(r["occ_c_prev"]), f2(r["dep_c_prev"]), f2(r["occ_c"]), f2(r["dep_c"])))
        P("")
    # 4 路线级
    P("### 4. 路线级：全臂失败与只有主臂失败的路线\n")
    keyf = lambda m: (m["scene"], m["route"])
    allm = [m for m in c["rows"]]
    arms_all = c["arms"]
    groups = {}
    for m in allm:
        groups.setdefault(keyf(m), {}).setdefault(m["arm"], []).append(bool(m["success_stop"]))
    main = "rule-occ+route"
    rows = []
    for k, d in sorted(groups.items()):
        rate = {a: (sum(d[a]), len(d[a])) for a in d}
        rows.append((k, rate))
    none_ok = [k for k, rate in rows if all(v[0] == 0 for v in rate.values())]
    main_only = [k for k, rate in rows if main in rate and rate[main][0] == 0 and any(v[0] > 0 for a, v in rate.items() if a != main)]
    P("- 路线（场景, 路线）共 %d 条；**所有臂 0 集到点**的 %d 条：%s" % (len(rows), len(none_ok), "，".join("%s r%s" % (k[0].replace("sage3d_", ""), k[1]) for k in none_ok) or "无"))
    P("- 主臂 `%s` 0 集到点、但别的臂有到点的 %d 条：%s" % (main, len(main_only), "，".join("%s r%s" % (k[0].replace("sage3d_", ""), k[1]) for k in main_only) or "无"))
    P("")
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", help="<OUT>/formal（含 indoor/ outdoor/）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--last", type=int, default=20)
    a = ap.parse_args(argv)
    text = ["# 正式实验规则臂失败形态分析（只读；不改参数 / 规则 / 预注册）\n",
            "- 输出目录：`%s`；clear = [clear(+α), clear(0), clear(-α)]；全堵 = `[0, 0, 0]`；循环 = 末 %d 步全是 V1-more / V1-cap 且严格互换。\n" % (a.out_dir, a.last)]
    for name, sub in (("室内", "indoor"), ("室外", "outdoor")):
        d = os.path.join(a.out_dir, sub)
        c = summ.load_class(d)
        if c is None:
            continue
        text.append(analyze(name, c, a.last))
    s = "\n".join(text)
    sys.stdout.write(s)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
