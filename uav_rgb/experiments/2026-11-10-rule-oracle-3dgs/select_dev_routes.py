#!/usr/bin/env python3
"""室内开发路线选取：每个场景里「合格、但没进试点 40 条」的路线中取 N 条（A* 长度等间隔分位），写 routes_dev.json。

r_safe / d_look 只许在这批路线上选（任务卡 K1），正式的 40 条（试点那批）和室外 24 条一律不碰。
合格 = 与试点同一个过滤：步数预算 steps_h1 = ceil(max_steps × step_frac) + step_pad 不超过 step_cap（超的标「需满电档」）。
复用 sage3d-pilot/select_routes.py 的 select_scene / pick_indices / bucket_of，不另写一份过滤与分位规则。
一个场景剩下的合格路线不足 N 条就全取（counts 里记剩余数与实取数）。

输出结构同 routes_selected.json（scenes / need_full_battery / counts / params），多记 excluded_pilot（试点已选的路线，按场景）。
用法（参数全部必填，run.sh 配置区是唯一来源）：
  python select_dev_routes.py --scenes-dir $D/3dgs-scenes --scenes "sage3d_841150 ..." --pilot-routes <试点 routes_selected.json> \
      --out <OUT>/routes_dev.json --per-scene 4 --step-frac 0.5 --step-pad 15 --step-cap 100
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("select_routes", os.path.join(HERE, "..", "2026-11-10-sage3d-pilot", "select_routes.py"))
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def select_dev(scene_routes, pilot_scenes, *, per_scene, step_frac, step_pad, step_cap, len_edges):
    """scene_routes = {场景: scene.json 的 routes}；pilot_scenes = {场景: [试点已选的路线 dict]}。返回 routes_dev.json 的内容。"""
    scenes, full, counts, excluded = {}, {}, {}, {}
    for s, routes in scene_routes.items():
        # per_scene = 路线总数：select_scene 的 pick_indices 在 n <= m 时全取，得到整个合格池（按 A* 长度升序）
        res = sr.select_scene(s, routes, per_scene=max(len(routes), 1), step_frac=step_frac, step_pad=step_pad,
                              step_cap=step_cap, len_edges=len_edges)
        pilot = {x["route"] for x in pilot_scenes.get(s, [])}
        rest = [x for x in res["selected"] if x["route"] not in pilot]
        picked = [rest[j] for j in sr.pick_indices(len(rest), per_scene)]
        scenes[s], full[s], excluded[s] = picked, res["need_full_battery"], sorted(pilot)
        counts[s] = {"total": res["total"], "pool": res["pool"], "pilot_excluded": len(pilot & {x["route"] for x in res["selected"]}),
                     "rest": len(rest), "selected": len(picked)}
    return {"params": {"per_scene": per_scene, "step_frac": step_frac, "step_pad": step_pad, "step_cap": step_cap,
                       "len_edges": list(len_edges), "bucket_labels": sr.bucket_labels(len_edges)},
            "scenes": scenes, "need_full_battery": full, "counts": counts, "excluded_pilot": excluded}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes-dir", required=True)
    p.add_argument("--scenes", required=True, help="空格分隔的场景目录名")
    p.add_argument("--pilot-routes", required=True, help="试点的 routes_selected.json")
    p.add_argument("--out", required=True)
    p.add_argument("--per-scene", type=int, required=True)
    p.add_argument("--step-frac", type=float, required=True)
    p.add_argument("--step-pad", type=int, required=True)
    p.add_argument("--step-cap", type=int, required=True)
    a = p.parse_args(argv)
    with open(a.pilot_routes, encoding="utf-8") as f:
        raw = f.read()
    pilot = json.loads(raw)["scenes"]
    scene_routes = {}
    for s in a.scenes.split():
        with open(os.path.join(a.scenes_dir, s, "scene.json"), encoding="utf-8") as f:
            scene_routes[s] = json.load(f).get("routes") or []
    res = select_dev(scene_routes, pilot, per_scene=a.per_scene, step_frac=a.step_frac, step_pad=a.step_pad,
                     step_cap=a.step_cap, len_edges=sr.LEN_EDGES)
    res["params"]["pilot_routes_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    for s in scene_routes:
        c, sel = res["counts"][s], res["scenes"][s]
        print(f"{s}: 共 {c['total']} 条，合格池 {c['pool']}，试点已选 {c['pilot_excluded']}，剩 {c['rest']}，取 {c['selected']}；"
              f"路线 {[(r['route'], round(r['astar_path_m'], 1)) for r in sel]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
