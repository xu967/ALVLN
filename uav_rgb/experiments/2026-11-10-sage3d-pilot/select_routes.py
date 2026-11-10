#!/usr/bin/env python3
"""SAGE-3D 室内试点的选路线：读每个场景 scene.json 的 routes，按 A* 长度分层取，写 routes_selected.json。

只用标准库（不依赖 numpy / torch）。run.sh 在起跑前调用；选取函数可导入，单测见 test_select_routes.py。
用法（参数全部必填，run.sh 配置区是唯一来源，这里不放第二份默认值）：
  python select_routes.py --scenes-dir $D/3dgs-scenes --scenes "sage3d_841150 sage3d_840865" --out $OUT/routes_selected.json \
      --per-scene 8 --step-frac 0.5 --step-pad 15 --step-cap 100

规则：
  1. 步数预算 steps_h1 = ceil(max_steps × step_frac) + step_pad（max_steps 是 hold 0.5 口径，与 bearing-guard 同口径）。
  2. steps_h1 > step_cap 的路线标「需满电档」：不进选取池、不跑，单列（场景 / 路线 / 未封顶步数）。先剔除，再分层。
  3. 池里按 sage.astar_path_m 升序（同长按路线下标），n <= per_scene 全取；否则取 per_scene 个等间隔分位，
     索引 floor(i × (n-1)/(per_scene-1) + 0.5)，含最短与最长；per_scene=1（冒烟）取最短。
  4. bucket 按 LEN_EDGES 记，左闭右开：<5 m / 5–10 m / >=10 m。
"""
import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------- 配置区
LEN_EDGES = (5.0, 10.0)   # 长度分桶边界（A* 米）。每步约 0.25 m，5 m ≈ 20 步、10 m ≈ 40 步；边界值是任务书定的，没有按真实路线长度分布核实
# ----------------------------------------------------------------


def steps_h1(max_steps, step_frac, step_pad):
    return math.ceil(max_steps * step_frac) + step_pad


def bucket_labels(edges):
    e = [f"{v:g}" for v in edges]
    return [f"<{e[0]} m"] + [f"{a}–{b} m" for a, b in zip(e, e[1:])] + [f">={e[-1]} m"]


def bucket_of(length_m, edges):
    return bucket_labels(edges)[sum(length_m >= v for v in edges)]


def pick_indices(n, m):
    """从 n 个有序元素里取 m 个等间隔分位的下标。n <= m 全取；m = 1 取最短。n > m >= 2 时相邻分位的间隔 > 1，取整后不会重复。"""
    if n <= m:
        return list(range(n))
    if m <= 1:
        return [0]
    return [math.floor(i * (n - 1) / (m - 1) + 0.5) for i in range(m)]


def _num(scene, i, route, key, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{scene} routes[{i}]（{route.get('name')}）的 {key} 不是有限数：{value!r}")
    return value


def select_scene(scene, routes, *, per_scene, step_frac, step_pad, step_cap, len_edges):
    """返回 {"selected": [...], "need_full_battery": [...], "total": n, "pool": p}；route 字段是 scene.json routes 的原下标。"""
    pool, full = [], []
    for i, r in enumerate(routes):
        sage = r.get("sage") or {}
        length = _num(scene, i, r, "sage.astar_path_m", sage.get("astar_path_m"))
        max_steps = int(_num(scene, i, r, "max_steps", r.get("max_steps")))
        uncapped = steps_h1(max_steps, step_frac, step_pad)
        if uncapped > step_cap:
            full.append({"route": i, "name": r.get("name"), "max_steps": max_steps, "steps_uncapped": uncapped,
                         "astar_path_m": length})
            continue
        pool.append({"route": i, "name": r.get("name"), "split": sage.get("split"), "astar_path_m": length,
                     "sage_path_m": sage.get("sage_path_m"), "min_clearance_m": sage.get("min_clearance_m"),
                     "max_steps": max_steps, "steps_h1": uncapped, "bucket": bucket_of(length, len_edges)})
    pool.sort(key=lambda x: (x["astar_path_m"], x["route"]))
    return {"selected": [pool[j] for j in pick_indices(len(pool), per_scene)], "need_full_battery": full,
            "total": len(routes), "pool": len(pool)}


def select_all(scene_routes, *, per_scene, step_frac, step_pad, step_cap, len_edges):
    """scene_routes = {场景名: routes 列表} -> routes_selected.json 的内容。"""
    res = {s: select_scene(s, rs, per_scene=per_scene, step_frac=step_frac, step_pad=step_pad,
                           step_cap=step_cap, len_edges=len_edges) for s, rs in scene_routes.items()}
    return {"params": {"per_scene": per_scene, "step_frac": step_frac, "step_pad": step_pad, "step_cap": step_cap,
                       "len_edges": list(len_edges), "bucket_labels": bucket_labels(len_edges)},
            "scenes": {s: v["selected"] for s, v in res.items()},
            "need_full_battery": {s: v["need_full_battery"] for s, v in res.items()},
            "n_filtered": {s: len(v["need_full_battery"]) for s, v in res.items()},
            "counts": {s: {"total": v["total"], "pool": v["pool"], "selected": len(v["selected"])}
                       for s, v in res.items()}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes-dir", required=True)
    p.add_argument("--scenes", required=True, help="空格分隔的场景目录名")
    p.add_argument("--out", required=True, help="routes_selected.json 路径")
    p.add_argument("--per-scene", type=int, required=True)
    p.add_argument("--step-frac", type=float, required=True)
    p.add_argument("--step-pad", type=int, required=True)
    p.add_argument("--step-cap", type=int, required=True)
    a = p.parse_args(argv)
    scene_routes = {}
    for s in a.scenes.split():
        with open(os.path.join(a.scenes_dir, s, "scene.json"), encoding="utf-8") as f:
            scene_routes[s] = json.load(f).get("routes") or []
    res = select_all(scene_routes, per_scene=a.per_scene, step_frac=a.step_frac, step_pad=a.step_pad,
                     step_cap=a.step_cap, len_edges=LEN_EDGES)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(tmp, a.out)
    for s in scene_routes:
        c, sel = res["counts"][s], res["scenes"][s]
        by_bucket = {b: sum(r["bucket"] == b for r in sel) for b in res["params"]["bucket_labels"]}
        print(f"{s}: 共 {c['total']} 条，需满电档剔除 {res['n_filtered'][s]}，池 {c['pool']}，选 {c['selected']}；"
              f"分桶 {by_bucket}；路线 {[r['route'] for r in sel]}", flush=True)
        for r in res["need_full_battery"][s]:
            print(f"  需满电档 {s} r{r['route']}（{r['name']}）：未封顶 {r['steps_uncapped']} 步 > {a.step_cap}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
