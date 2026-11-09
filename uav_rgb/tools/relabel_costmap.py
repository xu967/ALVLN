#!/usr/bin/env python3
"""离线重标：把一个 collection 的 action_expert 按 costmap 教师（teacher.plan_action_costmap 的 argmin）重标到副本目录。

副本结构同 agent6 的 exp_truncate_post_line.py：<out>/<split>/<id>/ 下 rgb_*.png / scene.json / oracle.jsonl 硬链接，
actions.jsonl 重写（action_expert=costmap 标签，多 q_costs 与 legacy_action_expert），episode_manifest.json 改
planner=costmap / teacher_version=TEACHER_COST_VERSION / relabel_from=legacy（audit 的 label_recompute 按 planner 选复算器）。
action_executed 不动：轨迹是 legacy 教师飞出来的，重标只换「该帧该做什么」。

统计（--stats-out，JSON + 同名 .md）：与 legacy 的一致率与混淆矩阵；分区域（approach x<1.0 / in_field ≤ max_px+0.2 /
exit ≤ max_px+1.0 / post_success_line，与 agent6 q2_fov_explainability.zone_of 同口径）、分 legacy 航向误差 |err| 段、
分「学生视锥内有无柱」（--fov-csv q2_fov_frames.csv 的 n_vis，按 (episode, step) 对齐）；标签分布；fallback / tie 计数。

用法：
  python3 tools/relabel_costmap.py --src <collection> --out <copy_dir> --stats-out <stats.json> [--tol 25] [--lookahead-steps 1]
  python3 tools/relabel_costmap.py --src <collection> --no-copy --stats-out <stats.json> --tol 0     # 只算统计（变体）
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim", "tools"))
import teacher as T  # noqa: E402

NAMES = ("SAFE", "FWD", "L", "R")
ERR_BINS = ((0, 10), (10, 25), (25, 45), (45, 90), (90, 181))
DEFAULT_FOV_CSV = "/root/rivermind-data/uav_rgb/tmp/agent6-distill/q2_fov_frames.csv"


def zone_of(x, max_px):
    if x < 1.0:
        return "approach"
    if x <= max_px + 0.2:
        return "in_field"
    if x <= max_px + 1.0:
        return "exit"
    return "post_success_line"


def err_bin(err):
    if err is None:
        return "no_path"
    a = abs(err)
    for lo, hi in ERR_BINS:
        if lo <= a < hi:
            return f"[{lo},{hi if hi < 181 else 180}{')' if hi < 181 else ']'}"
    return "?"


def relabel_episode(job):
    """一集：返回 (split, eid, 每帧记录列表, 写出的副本路径或 None)。"""
    ep, split, out_dir, tol, steps_ahead, do_copy = job
    scene = json.load(open(os.path.join(ep, "scene.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8"))
    pillars = scene["pillars"]
    max_px = max(float(p["x"]) for p in pillars)
    T.set_vehicle(man.get("vehicle") or "x3", T.task_of_scene(scene))
    waypoint, stop_radius = man.get("waypoint"), man.get("stop_radius")
    rows = [json.loads(line) for line in open(os.path.join(ep, "actions.jsonl"), encoding="utf-8") if line.strip()]
    recs, new_rows = [], []
    for r in rows:
        pb, yb = r.get("pose_before"), r.get("yaw_before")
        rec = {"split": split, "episode": int(man.get("seed", os.path.basename(ep))), "step": int(r["step"]),
               "legacy": r["action_expert"], "collision": bool(r.get("collision"))}
        if not isinstance(pb, dict) or yb is None:
            rec.update(costmap=None, fallback="no_pose", tie=False, zone=None, err=None, q=None)
            recs.append(rec)
            new_rows.append(r)
            continue
        x, y = float(pb["x"]), float(pb["y"])
        a, info = T.plan_action_costmap(x, y, float(yb), pillars, tol=tol, lookahead_steps=steps_ahead,
                                        waypoint=waypoint, stop_at_waypoint=bool(stop_radius),
                                        stop_radius=stop_radius or 0.5)
        path = T._global_path(x, y, pillars, T.ASTAR_CLEARANCE_M, 0.1, waypoint=waypoint,
                              stop_at_waypoint=bool(stop_radius))
        err = T._pursuit_err_deg(path, x, y, float(yb), 0.7) if path else None
        rec.update(costmap=a, fallback=info["fallback"], tie=info["tie"], zone=zone_of(x, max_px),
                   err=err, q=info["q"])
        recs.append(rec)
        nr = dict(r)
        nr["action_expert"] = a
        nr["legacy_action_expert"] = r["action_expert"]
        if info["q"] is not None:
            nr["q_costs"] = info["q"]
        if info["fallback"] is not None:
            nr["costmap_fallback"] = info["fallback"]
        new_rows.append(nr)
    dst = None
    if do_copy:
        dst = os.path.join(out_dir, split, os.path.basename(ep))
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(ep):
            if f.startswith("rgb_") or f in ("scene.json", "oracle.jsonl") or f.startswith("depth"):
                src, d = os.path.join(ep, f), os.path.join(dst, f)
                if os.path.isdir(src):
                    continue
                if not os.path.exists(d):
                    try:
                        os.link(os.path.realpath(src), d)
                    except OSError:          # 跨文件系统：退回符号链接（preprocess 只读内容）
                        os.symlink(os.path.realpath(src), d)
        with open(os.path.join(dst, "actions.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(nr) + "\n" for nr in new_rows)
        man2 = dict(man)
        man2.update({"planner": "costmap", "teacher_version": T.TEACHER_COST_VERSION,
                     "relabel_from": "legacy", "relabel_from_planner": man.get("planner"),
                     "relabel_from_teacher_version": man.get("teacher_version"),
                     "relabel_source_dir": os.path.abspath(ep),
                     "costmap_tol_deg": tol, "costmap_lookahead_steps": steps_ahead})
        with open(os.path.join(dst, "episode_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man2, f, indent=1, ensure_ascii=False)
    return split, os.path.basename(ep), recs, dst


def load_fov(path):
    """q2_fov_frames.csv → {(episode, step): n_vis}；文件缺失返回 None。"""
    if not path or not os.path.isfile(path):
        return None
    import csv
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(int(row["episode"]), int(row["step"]))] = int(row["n_vis"])
    return out


def block(recs):
    n = len(recs)
    agree = sum(r["legacy"] == r["costmap"] for r in recs)
    return {"n": n, "agree": agree, "agreement": round(agree / n, 4) if n else None}


def summarize(recs, fov):
    valid = [r for r in recs if r["costmap"] is not None]
    out = {"frames": len(recs), "no_pose": len(recs) - len(valid), "overall": block(valid),
           "non_collision": block([r for r in valid if not r["collision"]]),
           "confusion_legacy_x_costmap": [[sum(1 for r in valid if r["legacy"] == i and r["costmap"] == j)
                                           for j in range(4)] for i in range(4)],
           "label_dist_legacy": [sum(1 for r in valid if r["legacy"] == i) for i in range(4)],
           "label_dist_costmap": [sum(1 for r in valid if r["costmap"] == i) for i in range(4)],
           "fallback": dict(Counter(r["fallback"] for r in valid if r["fallback"])),
           "ties": sum(1 for r in valid if r["tie"]),
           "by_zone": {}, "by_abs_err": {}, "by_visible_pillar": {}}
    for z in ("approach", "in_field", "exit", "post_success_line"):
        sub = [r for r in valid if r["zone"] == z]
        if sub:
            out["by_zone"][z] = block(sub) | {"label_dist_legacy": [sum(1 for r in sub if r["legacy"] == i) for i in range(4)],
                                              "label_dist_costmap": [sum(1 for r in sub if r["costmap"] == i) for i in range(4)]}
    for b in [f"[{lo},{hi if hi < 181 else 180}{')' if hi < 181 else ']'}" for lo, hi in ERR_BINS] + ["no_path"]:
        sub = [r for r in valid if err_bin(r["err"]) == b]
        if sub:
            out["by_abs_err"][b] = block(sub)
    if fov is None:
        out["by_visible_pillar"] = {"note": "没有 q2_fov_frames.csv，未分视锥"}
    else:
        groups = {"n_vis=0": [], "n_vis>=1": [], "unknown": []}
        for r in valid:
            nv = fov.get((r["episode"], r["step"]))
            groups["unknown" if nv is None else ("n_vis=0" if nv == 0 else "n_vis>=1")].append(r)
        for k, sub in groups.items():
            if sub:
                out["by_visible_pillar"][k] = block(sub) | {
                    "label_dist_legacy": [sum(1 for r in sub if r["legacy"] == i) for i in range(4)],
                    "label_dist_costmap": [sum(1 for r in sub if r["costmap"] == i) for i in range(4)]}
    return out


def to_md(stats, title):
    L = [f"## {title}", "", f"帧 {stats['frames']}（缺位姿 {stats['no_pose']}）；"
         f"一致率 {stats['overall']['agreement']}（{stats['overall']['agree']}/{stats['overall']['n']}），"
         f"非碰撞帧 {stats['non_collision']['agreement']}；fallback {stats['fallback']}；平局 {stats['ties']}", "",
         "| legacy \\ costmap | " + " | ".join(NAMES) + " |", "|---|" + "---|" * 4]
    for i, row in enumerate(stats["confusion_legacy_x_costmap"]):
        L.append(f"| {NAMES[i]} | " + " | ".join(str(v) for v in row) + " |")
    L += ["", "| 标签分布 | " + " | ".join(NAMES) + " |", "|---|" + "---|" * 4,
          "| legacy | " + " | ".join(str(v) for v in stats["label_dist_legacy"]) + " |",
          "| costmap | " + " | ".join(str(v) for v in stats["label_dist_costmap"]) + " |", ""]
    for name, key in (("区域", "by_zone"), ("legacy |err|", "by_abs_err"), ("视锥内柱", "by_visible_pillar")):
        d = stats[key]
        if not d or "note" in d:
            L += [f"{name}：{d.get('note', '无')}", ""]
            continue
        L += [f"| {name} | n | 一致率 | legacy 分布 | costmap 分布 |", "|---|---|---|---|---|"]
        for k, v in d.items():
            L.append(f"| {k} | {v['n']} | {v['agreement']} | {v.get('label_dist_legacy', '')} | {v.get('label_dist_costmap', '')} |")
        L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="collection 根目录（<src>/<split>/<id>/）")
    p.add_argument("--out", default=None, help="副本目录（--no-copy 时可不给）")
    p.add_argument("--no-copy", action="store_true", help="只算统计，不写副本")
    p.add_argument("--stats-out", required=True, help="统计 JSON 路径（同名 .md 一并写）")
    p.add_argument("--tol", type=float, default=25.0, help="costmap 航向罚死区（度）")
    p.add_argument("--lookahead-steps", type=int, default=1, help="costmap 前瞻步数")
    p.add_argument("--fov-csv", default=DEFAULT_FOV_CSV, help="q2_fov_frames.csv（n_vis）；不存在则不分视锥")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits", default="train,val,test")
    args = p.parse_args()
    if not args.no_copy and not args.out:
        p.error("--out 或 --no-copy 二选一")
    if args.out and os.path.realpath(args.out) == os.path.realpath(args.src):
        p.error("--out 不能是 --src 本身")
    if args.lookahead_steps < 1 or args.tol < 0:
        p.error("--lookahead-steps ≥ 1、--tol ≥ 0")
    jobs = []
    for split in args.splits.split(","):
        for ep in sorted(glob.glob(os.path.join(args.src, split, "*"))):
            if os.path.isfile(os.path.join(ep, "actions.jsonl")):
                jobs.append((ep, split, args.out, args.tol, args.lookahead_steps, not args.no_copy))
    if not jobs:
        p.error(f"{args.src} 下没有集")
    recs = []
    with Pool(args.workers) as pool:
        for i, (split, eid, rs, dst) in enumerate(pool.imap_unordered(relabel_episode, jobs), 1):
            recs.extend(rs)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} 集，{len(recs)} 帧", file=sys.stderr, flush=True)
    fov = load_fov(args.fov_csv)
    stats = {"schema": "uavrgb.relabel_costmap.v1", "src": os.path.abspath(args.src),
             "out": os.path.abspath(args.out) if args.out and not args.no_copy else None,
             "teacher_version": T.TEACHER_COST_VERSION, "tol_deg": args.tol, "lookahead_steps": args.lookahead_steps,
             "episodes": len(jobs), "fov_csv": os.path.abspath(args.fov_csv) if fov is not None else None,
             "all": summarize(recs, fov)}
    for split in args.splits.split(","):
        sub = [r for r in recs if r["split"] == split]
        if sub:
            stats[split] = summarize(sub, fov)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_out)), exist_ok=True)
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    title = f"{os.path.basename(os.path.abspath(args.src))} tol={args.tol} lookahead_steps={args.lookahead_steps}"
    with open(os.path.splitext(args.stats_out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write(to_md(stats["all"], title) + "\n")
    o = stats["all"]["overall"]
    print(f"{title}: 一致率 {o['agreement']} ({o['agree']}/{o['n']})，fallback {stats['all']['fallback']}，"
          f"costmap 分布 {stats['all']['label_dist_costmap']}", flush=True)


if __name__ == "__main__":
    main()
