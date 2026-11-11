#!/usr/bin/env python3
"""任务卡 W1-12 第一部分第 2 步「判」：离线反事实。在 W1-11 新教师臂的 240 集上逐步重算「预测位移再延长 reserve 的线段扫掠」会不会否决前进步。

只读 `exp-teacher-tol-3dgs-20261110`，不起仿真。占据栅格按场景读 pickle 缓存（`--occ-cache`），缓存缺失时从 `3dgs-scenes/<场景>/scene.json` 现建（慢，每个室外场景要十几秒）。

口径（任务卡原文）：
  - 检查 = 沿「预测位移（所选预测器给出），再顺着它的方向延长 reserve」的线段做 `occ.sweep`，起点 = 动作前位置，高度 = 动作前的 z。
  - 只对教师给 FORWARD_SLOW 的步判；转向不否决。
  - 命中 = 16 个非出生点（碰撞步 > 0）的室内撞在前集里，碰撞步或它之前 2 步内有前进步被否决的集数（预期不少于 12，不到 10 停）。
  - 误拒 = 全程没有碰撞的集里被否决的前进步占全部前进步的比例（预期不高于 5%，高于 10% 停）。
预测器变体（都用第 1 步的结论，P2 = 速度外推、P1 = 上下文表）：
  - P2 交叉拟合（主口径）：路线下标偶数的集用奇数半拟合的模型，奇数的集用偶数半拟合的模型，每一集都由没见过它的模型预测。
  - P2 拟合半：全部集用偶数半拟合的模型（第 1 步量误差用的那一个），拟合半的集是样本内。
  - P2 全量：全部 240 集拟合一次（上线会用的系数的样本内下限）。
  - P1 交叉拟合：只作旁证。
reserve：P2 变体用第 1 步的 reserve（前进后转弯步位移模 p90 + P2 在检验半前进步上预测误差 p90）；P1 变体用它自己的前进步检验半 p90。
用法：python counterfactual.py --out <目录> [--occ-cache <目录>]
"""
import argparse
import importlib
import json
import math
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import measure_disp as md  # noqa: E402

D_DEFAULT = "/root/rivermind-data/uav_rgb"
HIT_PRESET = 12          # 预期命中数（任务卡）
HIT_FLOOR = 10           # 低于它停
FALSE_PRESET = 0.05      # 预期误拒率上限
FALSE_CEIL = 0.10        # 高于它停
LOOKBACK = 2             # 命中：碰撞步及其之前 2 步


def _repo_imports():
    for p in (REPO, os.path.join(REPO, "sim", "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module("sim.3dgs.scene"), importlib.import_module("sim.3dgs.collision")


def load_occ(scene, cache_dir, scenes_dir):
    scene_mod, coll = _repo_imports()
    if cache_dir:
        p = os.path.join(cache_dir, f"{scene}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return coll.from_scene(scene_mod.load_scene(os.path.join(scenes_dir, scene, "scene.json")))


def body_to_world(dx, dy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (c * dx - s * dy, s * dx + c * dy)


def swept_veto(occ, s, d_body, reserve):
    """预测位移（机体系）换到世界系，再顺着它的方向延长 reserve，做 occ.sweep。返回 (被否决?, 首个碰撞点距起点的距离或 None)。"""
    wx, wy = body_to_world(d_body[0], d_body[1], s["yaw"])
    n = math.hypot(wx, wy)
    if n > 1e-6:
        ux, uy = wx / n, wy / n
    else:                                         # 预测位移为 0：顺机头方向
        ux, uy = math.cos(math.radians(s["yaw"])), math.sin(math.radians(s["yaw"]))
    p0 = (s["x"], s["y"], s["z"])
    p1 = (s["x"] + wx + reserve * ux, s["y"] + wy + reserve * uy, s["z"])
    hit, pt = occ.sweep(p0, p1)
    return bool(hit), (None if pt is None else math.hypot(pt[0] - p0[0], pt[1] - p0[1]))


def collide_class(e, c):
    """撞在前集的类：碰撞步是转向 = B；是前进：位移弦偏机头 >= 20° = C，否则 D（W1-11 §7.2 的口径，阈值 20° 取 C 类 29.7–37.2°、D 类 <= 5.1° 的中间）。"""
    s = e["samples"][c]
    if s["cur"] in ("L", "R"):
        return "B"
    if s["cur"] != "F":
        return "其他"
    ang = abs(math.degrees(math.atan2(s["d"][1], s["d"][0])))
    return "C" if ang >= 20.0 else "D"


def build_variants(eps, meas):
    """返回 {名称: (predict(e, s) -> 机体系位移, reserve)}。"""
    fit_eps, test_eps = md.split_fit_test(eps)
    s_fit = [s for e in fit_eps for s in e["samples"]]
    s_test = [s for e in test_eps for s in e["samples"]]
    s_all = [s for e in eps for s in e["samples"]]
    t1_fit, t1_test, t1_all = md.fit_p1(s_fit), md.fit_p1(s_test), md.fit_p1(s_all)
    m2_fit, m2_test, m2_all = md.fit_p2(s_fit), md.fit_p2(s_test), md.fit_p2(s_all)
    fit_ids = {id(e) for e in fit_eps}
    slip = meas["slip"]["F_to_T_all"]["p90"]
    r_p2 = meas["reserve_m"]
    r_p1 = slip + meas["errors"]["P1"]["forward"]["p90"]
    return {
        "P2 交叉拟合（主口径）": (lambda e, s: md.predict_p2(m2_test if id(e) in fit_ids else m2_fit, s, t1_test if id(e) in fit_ids else t1_fit), r_p2),
        "P2 拟合半": (lambda e, s: md.predict_p2(m2_fit, s, t1_fit), r_p2),
        "P2 全量": (lambda e, s: md.predict_p2(m2_all, s, t1_all), r_p2),
        "P1 交叉拟合（旁证）": (lambda e, s: md.predict_p1(t1_test if id(e) in fit_ids else t1_fit, s), r_p1),
    }


def run_variant(eps, occs, predict, reserve):
    """返回 {id(e): [(step, 被否决?, 命中点距起点距离)]}，只含教师给前进的步。"""
    res = {}
    for e in eps:
        occ = occs[e["scene"]]
        rows = []
        for s in e["samples"]:
            if s["expert"] != "F":
                continue
            v, dist = swept_veto(occ, s, predict(e, s), reserve)
            rows.append((s["i"], v, dist))
        res[id(e)] = rows
    return res


def summarize(eps, res):
    """命中与误拒。"""
    out = {"hits": 0, "collide_first_nonspawn": 0, "episodes": [], "false": {}}
    for e in eps:
        m = e["metrics"]
        rows = res[id(e)]
        if e["suite"] == "indoor" and m.get("collide_first") and m.get("collision_step", 0) >= 1:
            c = int(m["collision_step"])
            out["collide_first_nonspawn"] += 1
            near = [(i, d) for (i, v, d) in rows if v and c - LOOKBACK <= i <= c]
            earlier = [i for (i, v, d) in rows if v and i < c - LOOKBACK]
            hit = bool(near)
            out["hits"] += int(hit)
            s = e["samples"][c]
            out["episodes"].append({
                "scene": e["scene"], "route": e["route"], "episode": e["episode"], "collision_step": c, "class": collide_class(e, c),
                "prev": s["prev"], "cur": s["cur"], "hit": hit,
                "veto_steps_rel": [i - c for (i, _) in near], "veto_dist_m": [None if d is None else round(d, 3) for (_, d) in near],
                "earlier_veto_steps": earlier, "forward_steps": len(rows)})
    for key, pred in (("all", lambda e: True), ("indoor", lambda e: e["suite"] == "indoor"), ("outdoor", lambda e: e["suite"] == "outdoor")):
        nf = nv = ne = nev = 0
        by_prev = {pv: [0, 0] for pv in md.NAMES}
        for e in eps:
            if e["metrics"].get("collision") or not pred(e):
                continue
            ne += 1
            ev = 0
            sam = {s["i"]: s for s in e["samples"]}
            for (i, v, _) in res[id(e)]:
                nf += 1
                nv += int(v)
                ev += int(v)
                by_prev[sam[i]["prev"]][0] += 1
                by_prev[sam[i]["prev"]][1] += int(v)
            nev += int(ev > 0)
        out["false"][key] = {"episodes": ne, "episodes_with_veto": nev, "forward_steps": nf, "vetoed": nv,
                             "rate": (nv / nf if nf else None), "by_prev": {pv: {"forward": a, "vetoed": b} for pv, (a, b) in by_prev.items()}}
    return out


def gate(summary):
    hits = summary["hits"]
    rate = summary["false"]["all"]["rate"]
    stop = hits < HIT_FLOOR or rate > FALSE_CEIL
    return {"hits": hits, "false_rate": rate, "stop": stop,
            "hit_ok_preset": hits >= HIT_PRESET, "false_ok_preset": rate <= FALSE_PRESET}


def render(results, meas, n_mismatch, order):
    pc = lambda v: "—" if v is None else f"{100 * v:.2f}%"      # noqa: E731
    L = ["# 任务卡 W1-12 第 2 步：离线反事实（预测位移扫掠否决）", "",
         f"- 240 集（室外 120、室内 120）；教师给的动作与执行的动作不一致的步数：{n_mismatch}（预期 0）。",
         f"- reserve（P2）= {meas['reserve_m']:.4f} m = 前进后转弯步位移模 p90 {meas['slip']['F_to_T_all']['p90']:.4f} + P2 检验半前进步误差 p90 "
         f"{meas['errors']['P2']['forward']['p90']:.4f}。", "",
         "## 1. 汇总", "", "| 变体 | reserve（m） | 命中（/16） | 误拒 全部 | 误拒 室内 | 误拒 室外 | 门槛判定 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in order:
        r = results[name]
        f = r["summary"]["false"]
        g = r["gate"]
        verdict = "**停**" if g["stop"] else ("达预期" if g["hit_ok_preset"] and g["false_ok_preset"] else "过门槛、未全达预期")
        L.append(f"| {name} | {r['reserve']:.4f} | {r['summary']['hits']} | {pc(f['all']['rate'])}（{f['all']['vetoed']}/{f['all']['forward_steps']}） | "
                 f"{pc(f['indoor']['rate'])}（{f['indoor']['vetoed']}/{f['indoor']['forward_steps']}） | "
                 f"{pc(f['outdoor']['rate'])}（{f['outdoor']['vetoed']}/{f['outdoor']['forward_steps']}） | {verdict} |")
    main = order[0]
    r = results[main]
    L += ["", f"门槛：命中 < {HIT_FLOOR} 或 误拒（全部）> {int(FALSE_CEIL * 100)}% 即停；预期命中 >= {HIT_PRESET}、误拒 <= {int(FALSE_PRESET * 100)}%。判定按主口径（{main}）。", "",
          f"## 2. 逐集表（{main}；16 个非出生点的室内撞在前集）", "",
          "| # | 场景 路线 集 | 碰撞步 | 类 | 上一步 → 碰撞步动作 | 否决的前进步（相对碰撞步） | 否决点距起点（m） | 更早的否决步 | 命中 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, ep in enumerate(r["summary"]["episodes"], 1):
        L.append(f"| {k} | {ep['scene'].replace('sage3d_', '')} r{ep['route']} ep{ep['episode']} | {ep['collision_step']} | {ep['class']} | {ep['prev']}→{ep['cur']} | "
                 f"{ep['veto_steps_rel'] or '无'} | {ep['veto_dist_m'] or '—'} | {ep['earlier_veto_steps'] or '无'} | {'是' if ep['hit'] else '否'} |")
    L += ["", "按类的命中：" + "；".join(
        f"{cl} 类 {sum(1 for e in r['summary']['episodes'] if e['class'] == cl and e['hit'])}/{sum(1 for e in r['summary']['episodes'] if e['class'] == cl)}"
        for cl in ("B", "C", "D")), "",
          f"## 3. 误拒按上一步类拆开（{main}，全程无碰撞的集）", "", "| 范围 | 上一步 | 前进步 | 否决 | 比例 |", "| --- | --- | --- | --- | --- |"]
    for key, label in (("all", "全部"), ("indoor", "室内"), ("outdoor", "室外")):
        for pv in md.NAMES:
            b = r["summary"]["false"][key]["by_prev"][pv]
            if b["forward"]:
                L.append(f"| {label} | {pv} | {b['forward']} | {b['vetoed']} | {pc(b['vetoed'] / b['forward'])} |")
    ff = r["summary"]["false"]
    L += ["", f"有至少 1 步被否决的集：全部 {ff['all']['episodes_with_veto']}/{ff['all']['episodes']}，室内 {ff['indoor']['episodes_with_veto']}/{ff['indoor']['episodes']}，"
          f"室外 {ff['outdoor']['episodes_with_veto']}/{ff['outdoor']['episodes']}（无碰撞的集）。"]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=os.path.join(D_DEFAULT, "exp-teacher-tol-3dgs-20261110"))
    ap.add_argument("--arm", default=md.ARM_DEFAULT)
    ap.add_argument("--scenes-dir", default=os.path.join(D_DEFAULT, "3dgs-scenes"))
    ap.add_argument("--occ-cache", default=os.path.join(D_DEFAULT, "tmp", "phase12-cards", "w1-12", "occ_cache"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    eps = md.load_episodes(args.src, args.arm)
    meas = md.measure(eps)
    n_mismatch = sum(1 for e in eps for s in e["samples"] if s["expert"] != s["cur"])
    occs = {sc: load_occ(sc, args.occ_cache, args.scenes_dir) for sc in sorted({e["scene"] for e in eps})}
    variants = build_variants(eps, meas)
    order = list(variants)
    results = {}
    for name, (predict, reserve) in variants.items():
        res = run_variant(eps, occs, predict, reserve)
        sm = summarize(eps, res)
        results[name] = {"reserve": reserve, "summary": sm, "gate": gate(sm)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "counterfactual.json"), "w", encoding="utf-8") as f:
        json.dump({"source": {"dir": args.src, "arm": args.arm}, "reserve_p2": meas["reserve_m"], "expert_ne_executed_steps": n_mismatch,
                   "results": results}, f, ensure_ascii=False, indent=1)
    lines = render(results, meas, n_mismatch, order)
    with open(os.path.join(args.out, "counterfactual.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
