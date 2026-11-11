#!/usr/bin/env python3
"""dev 阶段的失败形态离线分析（只读 dev 输出，不重跑，不改任何参数 / 规则 / 预注册；给规则 v2 设计用）。只用标准库。

输入 <DEV>/indoor（config.json、routes_dev.json、<臂>/<场景>/r<N>/ep_00000/{metrics,rule,actions}）。
输出（打印，--out 同时写 markdown）：
  A. 选定臂（默认 rocc-r0.25-d1.2）里没到点的每一集：场景、路线、A* 长度、结局、末距离、末 20 步规则条目序列、
     V1-more / V1-cap 是否交替、最长左右交替段、路线方位 n 与规则转向是否相反、卡住位置附近的最小间隙；
  B. 6 组参数各自的：每步三个候选航向通道全堵（clear == [0,0,0]）的占比、V1-cap 步占比、超时率（按全部步、按到 nav_end 为止各一份），
     以及每条开发路线在各组参数下的结局矩阵与路线最小间隙；
  C. `forward+route@0.5` 的 dev 结果（参照）。
口径：
  - clear = rule_v1_step 的 info["clear"] = [clear(+α), clear(0), clear(-α)]（左转一步 / 保持航向 / 右转一步的通道是否畅通），写进 rule.jsonl。
  - 动作码 1 前进 / 2 YAW_LEFT / 3 YAW_RIGHT；n = round(-err/yaw_step_deg)，err > 0 = 目标在左，n > 0 = 目标在右。
    回转方向 = n > 0 时 YAW_RIGHT，n < 0 时 YAW_LEFT；转向与回转方向相同记「朝路线」，相反记「背离路线」，n = 0 无回转方向另计。
  - 最长左右交替段 = 整集里严格 YAW_LEFT / YAW_RIGHT 交替的最长连续段（和 summarize.alt_runs 同一个定义，min_run 取 2 以求最长）。
  - 卡住位置的最小间隙：actions.jsonl 的 clearance_m（该步位姿到原始占据栅格的距离）在末 20 步里的最小值；
    路线整条的最小间隙取 routes_dev.json 的 min_clearance_m。路线上「卡住位置那一点」的栅格间隙要载入占据栅格重算 A* 折线，这里不做（写未核实）。
用法：dev_failure_analysis.py <DEV/indoor> [--arm rocc-r0.25-d1.2] [--last 20] [--out 结果.md]
"""
import argparse
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("summarize", os.path.join(HERE, "summarize.py"))
summ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summ)

FWD, YAW_LEFT, YAW_RIGHT = 1, 2, 3
SHORT = {"V1-fwd": "fwd", "V1-back": "back", "V1-more": "more", "V1-cap": "cap", "V1-side": "side", "V1-last": "last", "V1-left": "left"}
ARROW = {FWD: "^", YAW_LEFT: "<", YAW_RIGHT: ">"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    except (OSError, ValueError):
        return None
    return out


def load_episodes(dev):
    """{臂: {(场景, 路线): {"m": metrics, "rule": [行], "act": [行]}}}，只读 ep_00000（dev 的 K = 1）。"""
    cfg = read_json(os.path.join(dev, "config.json"))
    eps = {}
    for arm in cfg["arms"]:
        g = arm["name"]
        eps[g] = {}
        for s in cfg["scenes"]:
            for r in cfg["routes"][s]:
                d = os.path.join(dev, g, s, "r%d" % r, "ep_00000")
                m = read_json(os.path.join(d, "metrics.json"))
                if m is None:
                    continue
                eps[g][(s, r)] = {"m": m, "rule": read_jsonl(os.path.join(d, "rule.jsonl")), "act": read_jsonl(os.path.join(d, "actions.jsonl"))}
    return cfg, eps


def outcome(m):
    if m["success_stop"]:
        return "到点"
    if m["collide_first"]:
        return "撞在前"
    return "超时" if m["timeout"] else "其他"


def nav_end(m):
    """到点 / 首次碰撞的步号（含）较早者；都没有取最后一步。"""
    cands = [x for x in (m.get("reach_step"), m.get("collision_step")) if x is not None]
    return min(cands) if cands else m["steps"] - 1


def back_dir(n):
    return YAW_RIGHT if n > 0 else YAW_LEFT if n < 0 else None


def longest_alt(seq):
    runs = summ.alt_runs(seq, min_run=2)
    if not runs:
        return 0, None
    i, j = max(runs, key=lambda ij: ij[1] - ij[0])
    return j - i + 1, i


def clear_counter(rows):
    c = {}
    for x in rows:
        k = "".join(str(int(v)) for v in x["clear"])
        c[k] = c.get(k, 0) + 1
    return c


def alt_more_cap(entries):
    """entries = 条目名序列；返回 (相邻 more/cap 互换的对数, 严格 more/cap 交替的最长连续段长度)。"""
    pairs = sum(1 for a, b in zip(entries, entries[1:]) if {a, b} == {"V1-more", "V1-cap"})
    best = cur = 0
    for i, e in enumerate(entries):
        if e in ("V1-more", "V1-cap") and (cur == 0 or {entries[i - 1], e} == {"V1-more", "V1-cap"}):
            cur += 1
        else:
            cur = 1 if e in ("V1-more", "V1-cap") else 0
        best = max(best, cur)
    return pairs, best


def tail_target_run(rule):
    """从最后一步往回数，目标点坐标（取 2 位小数）不变的连续步数。"""
    if not rule:
        return 0
    last_t = tuple(round(v, 2) for v in rule[-1]["target"])
    n = 0
    for x in reversed(rule):
        if tuple(round(v, 2) for v in x["target"]) != last_t:
            break
        n += 1
    return n


def final_target_dist(ep):
    """最后一步位姿（pose_before）到该步目标点的距离。"""
    a, x = ep["act"][-1]["pose_before"], ep["rule"][-1]["target"]
    return math.hypot(x[0] - a["x"], x[1] - a["y"])


def ended_in_cycle(ep, last=20):
    """末 last 步全是 V1-more / V1-cap 且相邻严格互换。"""
    ents = [x["entry"] for x in (ep["rule"] or [])[-last:]]
    pairs, run = alt_more_cap(ents)                # 不足 last 步时 run 到不了 last，自然判否
    return run == last and pairs == last - 1


def target_stalled(ep, last=20, near_m=0.5):
    """末 last 步目标点一直是同一个点，且飞机末位姿离它不到 near_m。"""
    if not ep["rule"] or not ep["act"]:
        return False
    return tail_target_run(ep["rule"]) >= last and final_target_dist(ep) < near_m


def episode_report(key, ep, route_info, last):
    m, rule, act = ep["m"], ep["rule"], ep["act"]
    s, r = key
    ri = route_info.get((s, r), {})
    tail = rule[-last:]
    seq_all = [a["action_executed"] for a in act]
    run_len, run_at = longest_alt(seq_all)
    entries = [x["entry"] for x in tail]
    pairs, mc_run = alt_more_cap(entries)
    toward = away = zero = 0
    for x in tail:
        if x["action"] not in (YAW_LEFT, YAW_RIGHT):
            continue
        b = back_dir(x["n"])
        if b is None:
            zero += 1
        elif x["action"] == b:
            toward += 1
        else:
            away += 1
    pos = [(a["pose_before"]["x"], a["pose_before"]["y"]) for a in act[-last:]]
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    clr = [a["clearance_m"] for a in act[-last:] if a.get("clearance_m") is not None]
    cnt = {}
    for e in entries:
        cnt[SHORT.get(e, e)] = cnt.get(SHORT.get(e, e), 0) + 1
    return {
        "scene": s, "route": r, "name": ri.get("name"), "astar_m": ri.get("astar_path_m"), "route_min_clr": ri.get("min_clearance_m"),
        "bucket": ri.get("bucket"), "outcome": outcome(m), "steps": m["steps"], "final_wp_dist": m.get("final_wp_dist_m"),
        "min_wp_dist": m.get("min_wp_dist_m"), "collision_step": m.get("collision_step"),
        "tail_seq": " ".join("%s%s" % (SHORT.get(x["entry"], x["entry"]), ARROW.get(x["action"], "?")) for x in tail),
        "tail_n": [x["n"] for x in tail], "tail_err": [x["err_deg"] for x in tail], "tail_counts": cnt,
        "more_cap_pairs": pairs, "more_cap_run": mc_run, "alt_run_len": run_len, "alt_run_at": run_at,
        "toward": toward, "away": away, "n_zero_turn": zero,
        "bbox": (max(xs) - min(xs), max(ys) - min(ys)) if pos else None,
        "min_clr_tail": min(clr) if clr else None, "last_clr": clr[-1] if clr else None,
        "last_entry": tail[-1]["entry"] if tail else None, "last_clear": tail[-1]["clear"] if tail else None,
        "tgt_run": tail_target_run(rule), "tgt_dist": final_target_dist(ep), "cycle_end": ended_in_cycle(ep, last),
        "clear_dist": clear_counter(tail),
    }


def arm_stats(eps_arm):
    """某臂 20 集合计：全部步 / 到 nav_end 为止的步里，clear == [0,0,0] 的占比、各条目占比；超时率等。"""
    n = k = 0
    tot = {"all": 0, "pre": 0}
    blk = {"all": 0, "pre": 0}
    ent = {"all": {}, "pre": {}}
    blk_by = {"timeout": [0, 0], "other": [0, 0]}      # [全堵步数, 步数]，全部步
    eps_with_blk = 0
    for key, ep in eps_arm.items():
        m, rule = ep["m"], ep["rule"] or []       # forward 臂没有 rule.jsonl
        n += 1
        k += bool(m["success_stop"])
        ne = nav_end(m)
        b_ep = 0
        for x in rule:
            for w in ("all", "pre"):
                if w == "pre" and x["step"] > ne:
                    continue
                tot[w] += 1
                ent[w][x["entry"]] = ent[w].get(x["entry"], 0) + 1
                if x["clear"] == [0, 0, 0]:
                    blk[w] += 1
            grp = "timeout" if m["timeout"] else "other"
            blk_by[grp][1] += 1
            if x["clear"] == [0, 0, 0]:
                blk_by[grp][0] += 1
                b_ep += 1
        eps_with_blk += b_ep > 0
    # 只数没到点的集：到点之后规则臂不停，在 waypoint 附近徘徊，目标点本来就是 waypoint，不算停滞
    cyc = sum(bool(ended_in_cycle(e)) for e in eps_arm.values() if not e["m"]["success_stop"])
    stall = sum(bool(target_stalled(e)) for e in eps_arm.values() if not e["m"]["success_stop"])
    return {"n": n, "k": k, "tot": tot, "blk": blk, "ent": ent, "blk_by": blk_by, "eps_with_blk": eps_with_blk, "cyc": cyc, "stall": stall,
            "timeout": sum(bool(e["m"]["timeout"]) for e in eps_arm.values()), "cf": sum(bool(e["m"]["collide_first"]) for e in eps_arm.values())}


def pct(a, b):
    return "—" if not b else "%.1f%%" % (100.0 * a / b)


def render(cfg, eps, route_info, arm, last):
    o = []
    P = o.append
    P("# dev 失败形态离线分析（只读 dev 输出；不改参数 / 规则 / 预注册）\n")
    P("- 数据：`%s` 下的 dev 阶段输出，5 个室内场景 × 4 条开发路线 × K = 1 = 20 集每臂（seed0 9000）；条目 / clear / n / err 取 `rule.jsonl`，动作与间隙取 `actions.jsonl`，结局取 `metrics.json`。" % cfg.get("_dir", "dev/indoor"))
    P("- clear = [clear(+α), clear(0), clear(-α)] = 左转一步 / 保持航向 / 右转一步三个候选通道是否畅通；全堵 = `[0, 0, 0]`。动作 `^` 前进 `<` YAW_LEFT `>` YAW_RIGHT；序列写法「条目+动作」。\n")

    # --- A
    fails = [(k, e) for k, e in sorted(eps[arm].items()) if not e["m"]["success_stop"]]
    P("## A. `%s` 没到点的 %d 集（共 %d 集）\n" % (arm, len(fails), len(eps[arm])))
    P("| 场景 | 路线 | A* 长度 m | 结局 | 步数 | 末距离 m（最近距离） | 路线最小间隙 m | 末 %d 步 位置包围盒 m | 末 %d 步最小间隙 m | 末 %d 步 more↔cap 互换对数 / 最长交替段 | 整集最长左右交替段（起点步） | 末 %d 步转向：朝路线 / 背离 / n=0 | 目标点末尾连续不变步数 / 末距离 m |" % (last, last, last, last))
    P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    reps = []
    for key, ep in fails:
        x = episode_report(key, ep, route_info, last)
        reps.append(x)
        P("| %s | %s（%s） | %s | %s | %d | %.2f（%.2f） | %s | %.2f × %.2f | %s | %d / %d | %d（第 %s 步） | %d / %d / %d | %d / %.2f |" % (
            x["scene"], x["route"], x["name"], x["astar_m"], x["outcome"], x["steps"], x["final_wp_dist"], x["min_wp_dist"], x["route_min_clr"],
            x["bbox"][0], x["bbox"][1], "%.3f" % x["min_clr_tail"] if x["min_clr_tail"] is not None else "未核实",
            x["more_cap_pairs"], x["more_cap_run"], x["alt_run_len"], x["alt_run_at"], x["toward"], x["away"], x["n_zero_turn"],
            x["tgt_run"], x["tgt_dist"]))
    P("")
    P("末 %d 步规则条目序列（旧 → 新）：\n" % last)
    for x in reps:
        P("- %s r%d（%s，%s）：`%s`" % (x["scene"], x["route"], x["outcome"], "条目计数 " + ", ".join("%s %d" % kv for kv in sorted(x["tail_counts"].items(), key=lambda kv: -kv[1])), x["tail_seq"]))
        P("  - n：`%s`；err_deg：`%s`" % (" ".join("%+d" % v for v in x["tail_n"]), " ".join("%.0f" % v for v in x["tail_err"])))
        P("  - 末 %d 步 clear 分布（左 / 保持 / 右，1 = 畅通）：%s；最后一步：条目 %s，clear %s；整集最长左右交替段 %d 步（第 %s 步起）" % (
            last, "，".join("%s × %d" % kv for kv in sorted(x["clear_dist"].items(), key=lambda kv: -kv[1])), x["last_entry"], x["last_clear"], x["alt_run_len"], x["alt_run_at"]))
    P("")
    P("路线上卡住位置那一点的栅格间隙：**未核实**（routes_dev.json / routes_check.json 只存整条路线的最小间隙，没有折线；要在每一集卡住的位置重算要载入占据栅格重跑 A*，开发期间不做）。上表「路线最小间隙」是整条路线的最小值，「末 %d 步最小间隙」是飞机自己位姿到原始占据栅格的距离。\n" % last)

    # --- B
    arms = [a["name"] for a in cfg["arms"] if a["name"].startswith("rocc-")]
    P("## B. 6 组参数：超时率随 r_safe 上升与三通道全堵、V1-cap 的关系\n")
    P("全部步 = 整集所有步（含到点之后的徘徊）；到 nav_end = 只数到点 / 首次碰撞（无则全集）为止的步，去掉到点后徘徊的影响。\n")
    P("| 臂 | SR(stop) | 撞在前 | 超时率 | 全部步数 | 全堵步占比（全部） | V1-cap 步占比（全部） | V1-more 步占比（全部） | 全堵步占比（到 nav_end） | V1-cap 占比（到 nav_end） | 有全堵步的集数 | 全堵步里属于超时集的占比 | 没到点且末 20 步严格 more↔cap 交替的集数 | 没到点且目标点停滞的集数（末 20 步不变且末距离 < 0.5 m） |")
    P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    st = {}
    for g in arms:
        s = arm_stats(eps[g])
        st[g] = s
        a, p = s["tot"]["all"], s["tot"]["pre"]
        bt = s["blk_by"]["timeout"][0]
        ba = s["blk"]["all"]
        P("| `%s` | %d/%d = %.2f | %s | %s | %d | %s（%d） | %s | %s | %s（%d） | %s | %d | %s | %d | %d |" % (
            g, s["k"], s["n"], s["k"] / s["n"], pct(s["cf"], s["n"]), pct(s["timeout"], s["n"]), a,
            pct(ba, a), ba, pct(s["ent"]["all"].get("V1-cap", 0), a), pct(s["ent"]["all"].get("V1-more", 0), a),
            pct(s["blk"]["pre"], p), s["blk"]["pre"], pct(s["ent"]["pre"].get("V1-cap", 0), p), s["eps_with_blk"], pct(bt, ba),
            s["cyc"], s["stall"]))
    P("")
    P("每条开发路线在各组参数下的结局（到 = 到点，超 = 超时，撞 = 撞在前）与路线最小间隙：\n")
    P("| 场景 | 路线 | A* m | 路线最小间隙 m | " + " | ".join("`%s`" % g.replace("rocc-", "") for g in arms) + " |")
    P("| --- | --- | --- | --- | " + " | ".join("---" for _ in arms) + " |")
    short = {"到点": "到", "超时": "超", "撞在前": "撞", "其他": "?"}
    keys = sorted(eps[arm].keys())
    rows = []
    for key in keys:
        ri = route_info.get(key, {})
        cells = [short[outcome(eps[g][key]["m"])] if key in eps[g] else "—" for g in arms]
        rows.append((ri.get("min_clearance_m"), key, ri, cells))
        P("| %s | %d | %s | %s | %s |" % (key[0], key[1], ri.get("astar_path_m"), ri.get("min_clearance_m"), " | ".join(cells)))
    P("")
    inv = [r for r in rows if len(set(r[3])) == 1]
    inv_fail = [r for r in inv if r[3][0] != "到"]
    var = [r for r in rows if len(set(r[3])) > 1]
    P("6 组参数下结局完全相同的路线 %d 条（其中全部没到点 %d 条：%s；全部到点 %d 条）；结局随参数变化的路线 %d 条（%s）。\n" % (
        len(inv), len(inv_fail), "，".join("%s r%d（%s）" % (r[1][0].replace("sage3d_", ""), r[1][1], r[3][0]) for r in inv_fail), len(inv) - len(inv_fail),
        len(var), "，".join("%s r%d" % (r[1][0].replace("sage3d_", ""), r[1][1]) for r in var)))
    # 间隙分档：路线最小间隙 vs 各组参数的到点率
    P("按路线最小间隙分档的到点数（分母 = 该档路线数）：\n")
    cuts = [(0.0, 0.30), (0.30, 0.40), (0.40, 9.0)]
    P("| 档 | 路线数 | " + " | ".join("`%s`" % g.replace("rocc-", "") for g in arms) + " |")
    P("| --- | --- | " + " | ".join("---" for _ in arms) + " |")
    for lo, hi in cuts:
        sel = [r for r in rows if r[0] is not None and lo <= r[0] < hi]
        cells = ["%d/%d" % (sum(1 for r in sel if r[3][i] == "到"), len(sel)) for i in range(len(arms))]
        P("| [%.2f, %.2f) | %d | %s |" % (lo, hi, len(sel), " | ".join(cells)))
    P("")

    # --- C
    ref = "forward+route@0.5"
    if ref in eps:
        s = arm_stats(eps[ref])
        P("## C. 参照：`%s`（不看距离，只用路线方位；hold 0.5 s，步数 ×2）的 dev 结果\n" % ref)
        n = s["n"]
        coll = sum(bool(e["m"]["collision"]) for e in eps[ref].values())
        P("- 到点 SR(stop) %d/%d = %.2f；撞在前 %d/%d = %.2f；超时 %d/%d = %.2f；碰撞（整集任一步）%d/%d = %.2f。" % (
            s["k"], n, s["k"] / n, s["cf"], n, s["cf"] / n, s["timeout"], n, s["timeout"] / n, coll, n, coll / n))
        P("- 逐集：" + "，".join("%s r%d %s" % (k[0].replace("sage3d_", ""), k[1], outcome(eps[ref][k]["m"])) for k in sorted(eps[ref])))
        P("")
    return "\n".join(o) + "\n", reps, st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dev", help="<OUT>/dev/indoor")
    ap.add_argument("--arm", default="rocc-r0.25-d1.2")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg, eps = load_episodes(a.dev)
    cfg["_dir"] = a.dev
    rs = read_json(os.path.join(a.dev, "routes_dev.json"))
    route_info = {(s, x["route"]): x for s, xs in rs["scenes"].items() for x in xs}
    text, _, _ = render(cfg, eps, route_info, a.arm, a.last)
    sys.stdout.write(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
