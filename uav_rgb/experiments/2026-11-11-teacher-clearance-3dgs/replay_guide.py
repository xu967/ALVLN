#!/usr/bin/env python3
"""卡 W1-12 第三部分：用 K1 两个「目标点停滞」集的真实位姿轨迹离线重放 OracleRouteGuide.target，说清指针停在哪、为什么。

K1 的 rule.jsonl 逐步记了导引给出的目标点；这里在同一个占据栅格、同一个 A*、同一个前视距离上重建导引，
按 actions.jsonl 的 pose_before（= 策略看到的位姿）逐步喂给 target()，先核对重放的目标点与 rule.jsonl 记的一致，
再逐步记指针 k、到当前顶点与下一个顶点的距离，以及不受「只前进到局部最小」限制时离飞机最近的顶点。
如果 guard.OracleRouteGuide 带 nearest_in_lookahead 参数，同一条位姿序列也用它重放一遍，对比目标点。

只读：占据缓存（pickle，没有就现建）、K1 输出、场景 scene.json。不碰任何实验产物。
用法：python replay_guide.py --out DIR [--occ-cache DIR]
"""
import argparse
import importlib
import json
import math
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = "/root/rivermind-data/uav_rgb"
K1 = f"{D}/exp-rule-oracle-3dgs-20261110"
EPISODES = {
    "dev": f"{K1}/dev/indoor/rocc-r0.25-d1.2/sage3d_841150/r13/ep_00000",
    "formal": f"{K1}/formal/indoor/rule-occ+route/sage3d_841150/r4/ep_00002",
}
CONFIGS = {"dev": f"{K1}/dev/indoor/config.json", "formal": f"{K1}/formal/indoor/config.json"}
SCENE = "sage3d_841150"


def imports():
    sys.path.insert(0, ROOT)
    return (importlib.import_module("sim.3dgs.scene"), importlib.import_module("sim.3dgs.collision"),
            importlib.import_module("sim.3dgs.planner"), importlib.import_module("sim.3dgs.guard"))


def load_occ(scene_mod, coll, cache_dir, scenes_dir):
    if cache_dir:
        p = os.path.join(cache_dir, f"{SCENE}.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    return coll.from_scene(scene_mod.load_scene(os.path.join(scenes_dir, SCENE, "scene.json")))


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def d2(p, x, y):
    return (p[0] - x) ** 2 + (p[1] - y) ** 2


def seg_dist(a, b, x, y):
    """点到线段 ab 的距离。"""
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    vx, vy = bx - ax, by - ay
    n2 = vx * vx + vy * vy
    t = 0.0 if n2 <= 0 else max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / n2))
    return math.hypot(x - (ax + t * vx), y - (ay + t * vy))


def replay(guard_mod, plan_fn, wp, lookahead, poses, **guide_kw):
    """逐步喂位姿给 OracleRouteGuide，返回 (行, 导引对象)。行记 target() 之后的指针与相关距离。"""
    g = guard_mod.OracleRouteGuide(plan_fn, wp, lookahead, **guide_kw)
    rows = []
    for i, (x, y) in enumerate(poses):
        t = g.target(x, y)
        p = g.path or []
        k = g.k
        row = {"step": i, "x": x, "y": y, "k": k, "target": [t[0], t[1]]}
        if len(p) >= 2:
            row["d_k"] = math.sqrt(d2(p[k], x, y))
            row["d_k1"] = math.sqrt(d2(p[k + 1], x, y)) if k + 1 < len(p) else None
            row["nearest_global"] = min(range(len(p)), key=lambda j: d2(p[j], x, y))
            row["off_path_m"] = min(seg_dist(p[j], p[j + 1], x, y) for j in range(len(p) - 1))
        rows.append(row)
    return rows, g


def path_arclen(p):
    s = [0.0]
    for a, b in zip(p, p[1:]):
        s.append(s[-1] + math.dist(a, b))
    return s


def explain(path, rows, lookahead):
    """停滞：k 在末段不再变。报它第一次到达终值的步、停在哪个顶点、为什么不前进。"""
    k_final = rows[-1]["k"]
    first = next(r["step"] for r in rows if r["k"] == k_final)
    s = path_arclen(path)
    last = rows[-1]
    x, y = last["x"], last["y"]
    window = []
    for j in range(max(0, k_final - 2), min(len(path), k_final + 26)):
        window.append({"j": j, "xy": [round(path[j][0], 3), round(path[j][1], 3)], "arc_from_k_m": round(s[j] - s[k_final], 3),
                       "dist_to_aircraft_m": round(math.sqrt(d2(path[j], x, y)), 3)})
    # 窗口内（从 k 起沿路径不超过 lookahead）离飞机最近的顶点
    j_win = min((j for j in range(k_final, len(path)) if s[j] - s[k_final] <= lookahead), key=lambda j: (d2(path[j], x, y), -j))
    return {"k_final": k_final, "first_step_at_k_final": first, "n_vertices": len(path), "path_length_m": round(s[-1], 3),
            "pointer_vertex_xy": [round(v, 3) for v in path[k_final]], "pointer_arc_m": round(s[k_final], 3),
            "final_pose": [round(x, 3), round(y, 3)], "d_k": round(last["d_k"], 3), "d_k1": None if last["d_k1"] is None else round(last["d_k1"], 3),
            "nearest_global_final": last["nearest_global"], "nearest_in_window_final": j_win,
            "window": window}


def run(args):
    scene_mod, coll, planner_mod, guard_mod = imports()
    base_path = os.path.join(args.scenes_dir, SCENE, "scene.json")
    occ = load_occ(scene_mod, coll, args.occ_cache, args.scenes_dir)
    plan_fn = planner_mod.OccPlanner(occ).plan
    has_new = "nearest_in_lookahead" in guard_mod.OracleRouteGuide.__init__.__code__.co_varnames
    out = {}
    for name, ep in EPISODES.items():
        man = json.load(open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8"))
        cfg = json.load(open(CONFIGS[name], encoding="utf-8"))
        lookahead = float(cfg["route_lookahead_m"])
        sc = scene_mod.load_scene(base_path)
        scene_mod.apply_route(sc, int(man["route"]["index"]))
        wp = (float(sc["waypoint"]["x"]), float(sc["waypoint"]["y"]))
        act, rule = read_jsonl(os.path.join(ep, "actions.jsonl")), read_jsonl(os.path.join(ep, "rule.jsonl"))
        poses = [(a["pose_before"]["x"], a["pose_before"]["y"]) for a in act]
        rows, g = replay(guard_mod, plan_fn, wp, lookahead, poses)
        rec = [tuple(r["target"]) for r in rule]
        diff = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip((tuple(r["target"]) for r in rows), rec))
        n_equal = sum(1 for a, b in zip((tuple(round(v, 3) for v in r["target"]) for r in rows), rec) if a == b)
        k_changes = [(r["step"], r["k"]) for i, r in enumerate(rows) if i == 0 or r["k"] != rows[i - 1]["k"]]
        res = {"episode": ep, "route": man["route"], "waypoint": wp, "lookahead_m": lookahead, "n_steps": len(rows), "spawn_pose": poses[0],
               "plan_failed": g.plan_failed, "n_vertices": len(g.path),
               "replay_vs_recorded": {"steps_exact_after_rounding_3dp": n_equal, "max_abs_diff_m": diff},
               "pointer_changes": k_changes,
               "recorded_target_changes": [(i, rec[i]) for i in range(len(rec)) if i == 0 or rec[i] != rec[i - 1]],
               "explain": explain(g.path, rows, lookahead), "rows": rows}
        if has_new:
            rows2, g2 = replay(guard_mod, plan_fn, wp, lookahead, poses, nearest_in_lookahead=True)
            res["nearest_in_lookahead"] = {
                "pointer_changes": [(r["step"], r["k"]) for i, r in enumerate(rows2) if i == 0 or r["k"] != rows2[i - 1]["k"]],
                "target_changes": [(r["step"], [round(v, 3) for v in r["target"]]) for i, r in enumerate(rows2)
                                   if i == 0 or r["target"] != rows2[i - 1]["target"]][:40],
                "final_k": rows2[-1]["k"], "final_target": [round(v, 3) for v in rows2[-1]["target"]], "rows": rows2}
        out[name] = res
    return out


def render(out):
    L = ["# W1-12 第三部分：OracleRouteGuide 指针重放（K1 的两个目标点停滞集）", ""]
    for name, r in out.items():
        e = r["explain"]
        L += [f"## {name}", "", f"- 集：`{r['episode']}`", f"- 路线 {r['route']}，waypoint {tuple(round(v, 3) for v in r['waypoint'])}，前视 {r['lookahead_m']} m，{r['n_steps']} 步",
              f"- 出生位姿（第 0 步 pose_before）{tuple(round(v, 3) for v in r['spawn_pose'])}，A* 路径 {r['n_vertices']} 个顶点，总长 {e['path_length_m']} m，plan_failed = {r['plan_failed']}",
              f"- 重放目标点与 `rule.jsonl` 记录：四舍五入到 3 位后逐步相同的步数 {r['replay_vs_recorded']['steps_exact_after_rounding_3dp']} / {r['n_steps']}，最大差 {r['replay_vs_recorded']['max_abs_diff_m']:.4f} m",
              f"- 指针变化（步, k）：{r['pointer_changes']}",
              f"- 记录的目标点变化（步, 目标点）：{r['recorded_target_changes']}",
              f"- 指针终值 k = {e['k_final']}（第 {e['first_step_at_k_final']} 步起不再变），顶点 {e['pointer_vertex_xy']}，沿路径 {e['pointer_arc_m']} m；"
              f"末位姿 {e['final_pose']}，到顶点 k 的距离 {e['d_k']} m，到顶点 k+1 的距离 {e['d_k1']} m",
              f"- 末位姿的全局最近顶点 {e['nearest_global_final']}；从 k 起沿路径不超过前视距离内的最近顶点 {e['nearest_in_window_final']}", ""]
        L += ["| 顶点 j | 坐标 | 自 k 起沿路径 m | 到末位姿 m |", "| --- | --- | --- | --- |"]
        for w in e["window"]:
            L.append(f"| {w['j']} | {w['xy']} | {w['arc_from_k_m']} | {w['dist_to_aircraft_m']} |")
        L.append("")
        pr = [(x["step"], x["k"], round(x["d_k"], 3), None if x["d_k1"] is None else round(x["d_k1"], 3), x["nearest_global"], round(x["off_path_m"], 3))
              for x in r["rows"]]
        L += ["逐步（步, k, d(k), d(k+1), 全局最近顶点, 离折线距离 m）：", "", "```", *[str(t) for t in pr[::max(1, len(pr) // 31)]], "```", ""]
        if "nearest_in_lookahead" in r:
            n = r["nearest_in_lookahead"]
            L += [f"新参数 `nearest_in_lookahead=True`：指针变化（步, k）{n['pointer_changes'][:30]}，末 k = {n['final_k']}，末目标点 {n['final_target']}；"
                  f"目标点变化 {len(n['target_changes'])} 次，前几次 {n['target_changes'][:8]}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--occ-cache", default=f"{D}/tmp/phase12-cards/w1-12/occ_cache")
    ap.add_argument("--scenes-dir", default=f"{D}/3dgs-scenes")
    args = ap.parse_args()
    out = run(args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "replay_guide.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    md = render(out)
    with open(os.path.join(args.out, "replay_guide.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
