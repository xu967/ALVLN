#!/usr/bin/env python3
"""F0 对拍（任务卡 K3：正式 40 条上 v1 重跑，先与 K1 的结果同键对拍，应逐集一致，不一致就停）。

键 = (场景, 路线, seed)。逐集比：success_stop、collision、collide_first、timeout、steps、reach_step、collision_step、出生位姿（x、y、yaw_deg，容差 1e-6）。
K1 的臂目录 <K1 formal/indoor>/<臂>/<场景>/r<N>/ep_*/metrics.json；本次的臂目录 <OUT>/formal40/<臂>/...，同结构，路线与集数取本次 config.json。
用法：python check_k1_match.py --new <OUT_ROOT>/formal40 --k1 <K1 formal/indoor> [--arm rule-occ+route] [--k1-arm rule-occ+route] [--out-json f] [--out-md f]
退出码：0 全部一致；1 有不一致（markdown 里列出前 20 条；有不一致时即使数据不全也是 1）；2 数据不全（本次没跑完、K1 缺键）。
同一个函数 `compare_dirs` 也给 summarize.py 在「室外」用（v1 取 K1 的结果按键配对时先看 K1 的键是否齐）。
"""
import argparse
import glob
import json
import os
import sys

FIELDS = ("success_stop", "collision", "collide_first", "timeout", "steps", "reach_step", "collision_step")
POSE_TOL = 1e-6


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def routes_of(cfg, scene):
    r = cfg["routes"]
    return r.get(scene, []) if isinstance(r, dict) else r


def load_arm(cdir, arm, scenes_routes):
    """{(场景, 路线, seed): metrics}；scenes_routes = [(场景, 路线)]；某条路线没有 done.json 记入 missing_routes。"""
    rows, missing = {}, []
    for s, r in scenes_routes:
        d = os.path.join(cdir, arm, s, f"r{r}")
        if read_json(os.path.join(d, "done.json")) is None:
            missing.append((s, r))
            continue
        for p in sorted(glob.glob(os.path.join(d, "ep_*", "metrics.json"))):
            m = read_json(p)
            if m is not None:
                rows[(s, r, m["seed"])] = m
    return rows, missing


def load_all(cdir, arm):
    """不看 config：glob 目录下该臂全部集（K1 目录用，不依赖它的 config 写法）。"""
    rows = {}
    for p in sorted(glob.glob(os.path.join(cdir, arm, "*", "r*", "ep_*", "metrics.json"))):
        m = read_json(p)
        if m is not None:
            rows[(m["scene"], m["route"], m["seed"])] = m
    return rows


def pose_equal(a, b, tol=POSE_TOL):
    return all(abs(a[f] - b[f]) <= tol for f in ("x", "y", "yaw_deg"))


def compare_rows(new, old):
    """new / old = {键: metrics}；返回 {matched, equal, diffs（[{key, fields}]）, only_new, only_old}。"""
    keys = sorted(set(new) & set(old))
    diffs = []
    for k in keys:
        bad = [f for f in FIELDS if new[k].get(f) != old[k].get(f)]
        if not pose_equal(new[k]["start_pose"], old[k]["start_pose"]):
            bad.append("start_pose")
        if bad:
            diffs.append({"key": list(k), "fields": bad, "new": {f: new[k].get(f) for f in FIELDS}, "old": {f: old[k].get(f) for f in FIELDS}})
    return {"matched": len(keys), "equal": len(keys) - len(diffs), "diffs": diffs,
            "only_new": sorted(set(new) - set(old)), "only_old": sorted(set(old) - set(new))}


def render_md(res, arm, new_dir, k1_dir):
    L = ["# F0：v1 重跑与 K1 逐集对拍", "",
         f"- 本次 `{new_dir}` 的臂 `{arm}`；K1 `{k1_dir}`。键 = （场景, 路线, seed）；比较 {', '.join(FIELDS)} 与出生位姿（容差 {POSE_TOL}）。",
         f"- 本次 {res['n_new']} 集，K1 同键 {res['matched']} 集，逐项一致 {res['equal']} 集，不一致 {len(res['diffs'])} 集；本次有而 K1 没有 {len(res['only_new'])} 集，"
         f"本次路线没写 done.json {len(res['missing_routes'])} 条。", "",
         f"**判定：{'一致' if res['ok'] else '不一致或数据不全，停'}**"]
    if res["diffs"]:
        L += ["", "| 场景 | 路线 | seed | 不一致的项 | 本次 | K1 |", "| --- | --- | --- | --- | --- | --- |"]
        for d in res["diffs"][:20]:
            s, r, seed = d["key"]
            L.append(f"| {s} | {r} | {seed} | {', '.join(d['fields'])} | {json.dumps({f: d['new'][f] for f in d['fields'] if f in d['new']}, ensure_ascii=False)} "
                     f"| {json.dumps({f: d['old'][f] for f in d['fields'] if f in d['old']}, ensure_ascii=False)} |")
        if len(res["diffs"]) > 20:
            L.append(f"（共 {len(res['diffs'])} 条，只列前 20）")
    return "\n".join(L) + "\n"


def run(new_dir, k1_dir, arm, k1_arm):
    cfg = read_json(os.path.join(new_dir, "config.json"))
    if cfg is None:
        return None, f"{new_dir}/config.json 不存在"
    pairs = [(s, r) for s in cfg["scenes"] for r in routes_of(cfg, s)]
    new, missing = load_arm(new_dir, arm, pairs)
    old = load_all(k1_dir, k1_arm)
    cmp = compare_rows(new, old)
    res = {"arm": arm, "k1_arm": k1_arm, "n_new": len(new), "n_expected": len(pairs) * cfg["k"], "n_k1": len(old), "missing_routes": missing, **cmp}
    res["complete"] = not missing and len(new) == res["n_expected"] and not res["only_new"]
    res["ok"] = res["complete"] and not res["diffs"] and res["matched"] == res["n_expected"]
    return res, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True)
    ap.add_argument("--k1", required=True)
    ap.add_argument("--arm", default="rule-occ+route")
    ap.add_argument("--k1-arm", default="rule-occ+route")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args(argv)
    res, err = run(a.new, a.k1, a.arm, a.k1_arm)
    if res is None:
        print(err, file=sys.stderr)
        return 2
    md = render_md(res, a.arm, a.new, a.k1)
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if a.out_md:
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    if res["diffs"]:
        return 1
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
