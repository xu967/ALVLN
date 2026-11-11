#!/usr/bin/env python3
"""K3 的室内路线选取：三批互不相交的路线，每场景一份。

  formal40 = K1 的同一批（试点 sage3d-pilot 选定的 40 条，不在这里重选，只用来排除）；
  dev      = K1 的开发路线 20 条（K1 的 routes_dev.json）+ 新选的 dev-new（每场景 DEV_NEW_PER_SCENE 条）；
  holdout  = 新选的（每场景 HOLDOUT_PER_SCENE 条，从排除试点、K1 开发、dev-new 之后剩下的合格路线里取，不够就有多少取多少）。
合格条件与按 A* 长度升序的等间隔分位取法全部复用 K1 的 select_dev_routes.select_dev（它又复用 sage3d-pilot/select_routes.py），不另写一份：
dev-new 与 holdout 各调用一次 select_dev，后一次把前面已用的路线加进排除名单。
自检：用同样的参数在只排除试点的条件下重算 K1 开发路线，必须与 K1 的文件逐条相同（证明「同一套取法」成立），否则退出码 2。
held-out 合计少于 MIN_HOLDOUT 条退出码 3（任务卡：合计少于 24 条就停下报告）。

用法（run.sh 配置区是参数的唯一来源）：
  python select_routes_v2.py --scenes-dir $D/3dgs-scenes --scenes "sage3d_841150 ..." --pilot-routes <试点 routes_selected.json> \
      --k1-dev-routes <K1 routes_dev.json> --out-dir <OUT_ROOT>/routes --dev-new-per-scene 4 --holdout-per-scene 8 --min-holdout 24 \
      --step-frac 0.5 --step-pad 15 --step-cap 100
写 routes_dev.json、routes_holdout.json、routes_formal40.json（= 试点文件原样）、counts.md。
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("select_dev_routes", os.path.join(HERE, "..", "2026-11-10-rule-oracle-3dgs", "select_dev_routes.py"))
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)
sr = sd.sr


def _merge(*lists_by_scene):
    out = {}
    for d in lists_by_scene:
        for s, routes in d.items():
            out.setdefault(s, []).extend(routes)
    return out


def select_v2(scene_routes, pilot_scenes, k1_dev_scenes, *, dev_new_per_scene, holdout_per_scene, step_frac, step_pad, step_cap, len_edges):
    """返回 {"dev": routes_dev 格式, "holdout": routes_holdout 格式, "checks": {...}}。"""
    kw = dict(step_frac=step_frac, step_pad=step_pad, step_cap=step_cap, len_edges=len_edges)
    # 自检：只排除试点、取 4 条，应当复现 K1 的开发路线
    k1_again = sd.select_dev(scene_routes, pilot_scenes, per_scene=max((len(v) for v in k1_dev_scenes.values()), default=0), **kw)
    k1_same = {s: [r["route"] for r in k1_again["scenes"].get(s, [])] == [r["route"] for r in k1_dev_scenes.get(s, [])] for s in scene_routes}
    new = sd.select_dev(scene_routes, _merge(pilot_scenes, k1_dev_scenes), per_scene=dev_new_per_scene, **kw)
    hold = sd.select_dev(scene_routes, _merge(pilot_scenes, k1_dev_scenes, new["scenes"]), per_scene=holdout_per_scene, **kw)
    dev_scenes = {}
    for s in scene_routes:
        tagged = [{**r, "origin": "k1"} for r in k1_dev_scenes.get(s, [])] + [{**r, "origin": "new"} for r in new["scenes"][s]]
        dev_scenes[s] = sorted(tagged, key=lambda r: (r["astar_path_m"], r["route"]))
    dev = {**{k: v for k, v in new.items() if k not in ("scenes", "counts", "excluded_pilot", "params")},
           "params": {**new["params"], "per_scene_new": dev_new_per_scene, "per_scene_k1": max((len(v) for v in k1_dev_scenes.values()), default=0)},
           "scenes": dev_scenes,
           "counts": {s: {"k1": len(k1_dev_scenes.get(s, [])), "new": len(new["scenes"][s]), "total": len(dev_scenes[s]),
                          "rest_before_new": new["counts"][s]["rest"]} for s in scene_routes},
           "excluded_pilot": new["excluded_pilot"]}
    holdout = {**{k: v for k, v in hold.items() if k not in ("params",)}, "params": {**hold["params"], "per_scene": holdout_per_scene}}
    holdout["counts"] = {s: {**hold["counts"][s], "dev_excluded": len(k1_dev_scenes.get(s, [])) + len(new["scenes"][s])} for s in scene_routes}
    return {"dev": dev, "holdout": holdout, "checks": {"k1_dev_reproduced": k1_same}}


def disjoint_problems(pilot_scenes, dev, holdout):
    """三批路线按场景两两相交的 (场景, 路线) 列表；空 = 互不相交。"""
    bad = []
    for s in set(pilot_scenes) | set(dev["scenes"]) | set(holdout["scenes"]):
        sets = {"formal40": {r["route"] for r in pilot_scenes.get(s, [])}, "dev": {r["route"] for r in dev["scenes"].get(s, [])},
                "holdout": {r["route"] for r in holdout["scenes"].get(s, [])}}
        for a, b in (("formal40", "dev"), ("formal40", "holdout"), ("dev", "holdout")):
            bad += [(s, a, b, r) for r in sorted(sets[a] & sets[b])]
    return bad


def counts_table(pilot_scenes, dev, holdout):
    lines = ["| 场景 | formal40 | dev（K1 + 新） | holdout | 合格池 | 试点 + K1 开发 + dev-new 之后剩 |", "| --- | --- | --- | --- | --- | --- |"]
    tot = [0, 0, 0, 0, 0, 0]
    for s in dev["scenes"]:
        cd, ch = dev["counts"][s], holdout["counts"][s]
        row = [len(pilot_scenes.get(s, [])), f"{cd['total']}（{cd['k1']} + {cd['new']}）", len(holdout["scenes"][s]), ch["pool"], ch["rest"]]
        lines.append(f"| {s} | " + " | ".join(str(x) for x in row) + " |")
        tot[0] += len(pilot_scenes.get(s, []))
        tot[1] += cd["total"]
        tot[2] += len(holdout["scenes"][s])
        tot[3] += ch["pool"]
        tot[4] += ch["rest"]
    lines.append(f"| 合计 | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {tot[4]} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes-dir", required=True)
    p.add_argument("--scenes", required=True, help="空格分隔的场景目录名")
    p.add_argument("--pilot-routes", required=True, help="试点的 routes_selected.json（formal40）")
    p.add_argument("--k1-dev-routes", required=True, help="K1 的 routes_dev.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dev-new-per-scene", type=int, required=True)
    p.add_argument("--holdout-per-scene", type=int, required=True)
    p.add_argument("--min-holdout", type=int, required=True)
    p.add_argument("--step-frac", type=float, required=True)
    p.add_argument("--step-pad", type=int, required=True)
    p.add_argument("--step-cap", type=int, required=True)
    a = p.parse_args(argv)
    with open(a.pilot_routes, encoding="utf-8") as f:
        pilot_raw = f.read()
    with open(a.k1_dev_routes, encoding="utf-8") as f:
        k1_raw = f.read()
    pilot, k1_dev = json.loads(pilot_raw)["scenes"], json.loads(k1_raw)["scenes"]
    scene_routes = {}
    for s in a.scenes.split():
        with open(os.path.join(a.scenes_dir, s, "scene.json"), encoding="utf-8") as f:
            scene_routes[s] = json.load(f).get("routes") or []
    res = select_v2(scene_routes, pilot, k1_dev, dev_new_per_scene=a.dev_new_per_scene, holdout_per_scene=a.holdout_per_scene,
                    step_frac=a.step_frac, step_pad=a.step_pad, step_cap=a.step_cap, len_edges=sr.LEN_EDGES)
    dev, holdout = res["dev"], res["holdout"]
    for obj in (dev, holdout):
        obj["params"]["pilot_routes_sha256"] = hashlib.sha256(pilot_raw.encode("utf-8")).hexdigest()
        obj["params"]["k1_dev_routes_sha256"] = hashlib.sha256(k1_raw.encode("utf-8")).hexdigest()
    os.makedirs(a.out_dir, exist_ok=True)
    table = counts_table(pilot, dev, holdout)
    print(table, flush=True)
    bad_repro = [s for s, ok in res["checks"]["k1_dev_reproduced"].items() if not ok]
    if bad_repro:
        print(f"自检失败：只排除试点重算 K1 开发路线，与 K1 的文件不同的场景 {bad_repro}（同一套取法不成立，停下报告）", file=sys.stderr)
        return 2
    bad = disjoint_problems(pilot, dev, holdout)
    if bad:
        print(f"三批路线有重叠 {bad[:10]}（停下报告）", file=sys.stderr)
        return 2
    n_hold = sum(len(v) for v in holdout["scenes"].values())
    outputs = {"routes_dev.json": dev, "routes_holdout.json": holdout}
    for name, obj in outputs.items():
        tmp = os.path.join(a.out_dir, name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(a.out_dir, name))
    with open(os.path.join(a.out_dir, "routes_formal40.json"), "w", encoding="utf-8") as f:
        f.write(pilot_raw)
    with open(os.path.join(a.out_dir, "counts.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    if n_hold < a.min_holdout:
        print(f"held-out 合计 {n_hold} 条，少于 {a.min_holdout}（任务卡：停下报告）", file=sys.stderr)
        return 3
    print(f"dev {sum(len(v) for v in dev['scenes'].values())} 条，holdout {n_hold} 条，formal40 {sum(len(v) for v in pilot.values())} 条", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
