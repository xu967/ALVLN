#!/usr/bin/env python3
"""教师 v7 闭环对拍的汇总：两臂（v7 主臂、v5 对照臂）各两半（-a / -b，各 10 个冻结 seed）合并，写 summary.md 与 summary.json。

用法：
  python summarize.py <OUT> [--note 文字]      读 <OUT>/{v7,v5}-{a,b}/eval.json 与 episodes/seed_*/actions.jsonl
  python summarize.py --self-check            合成数据自检（不碰 Gazebo、不读真实输出）

口径（起跑前写定，与 README.md 一致）：
  成功    = eval.json per_episode[].success（metric_version 带 corridor3m：到线、无碰撞、没出走廊），每臂 20 个冻结 seed。
  出口区  = 步的 pose_before.x > 最远柱 x，最远柱 x = per_episode.goal_x − 顶层 goal_margin（eval 的 goal_x = 最远柱 x + goal_margin）。
            集在 x ≥ goal_x 即止，所以出口区是柱阵后 goal_margin（1.0 m）的那一段。
  YAW 步  = actions.jsonl 的 action_expert ∈ {2, 3}（YAW_LEFT、YAW_RIGHT）；教师臂 action_executed 与 action_expert 相同，
            汇总里同时数 action_executed 作校验。
  占比    = 出口区 YAW 步数 / 出口区步数，两半合并后按步池化；附 Wilson 95% 区间（样本只有一百多步，区间宽）。
门槛（任务卡 W1-7 验收 2）：v7 臂成功 20/20；v7 臂出口区 YAW 占比 < 5%。v5 对照臂只作同 seed、同时段的配对参照，不设门。
另报（只报告，不设门）：相邻 L↔R 换向次数（口径同 data-algo 的 a10_teacher_flipflop.py：本步与上一步都非碰撞、都是 YAW、方向不同）、
SAFE 步数（action_expert == 0）。A* 是否无路（path_none）评测输出里没有记录，闭环汇总里没有这一项；它的离线数字在 offline_0417.py。
"""
import argparse
import json
import math
import os
import sys
import tempfile

ARMS = ("v7", "v5")
HALVES = ("a", "b")
EXPECTED_EPISODES = 20
YAW_ACTIONS = (2, 3)        # YAW_LEFT、YAW_RIGHT
FORWARD, SAFE = 1, 0
GATE_SHARE = 0.05


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def episode_rows(ep_dir):
    path = os.path.join(ep_dir, "actions.jsonl")
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def exit_zone_counts(rows, pillar_x_max):
    """出口区步的统计：{n, yaw, fwd, safe, yaw_executed}；没有 pose_before.x 的步不计入。"""
    c = {"n": 0, "yaw": 0, "fwd": 0, "safe": 0, "yaw_executed": 0}
    for r in rows:
        pb = r.get("pose_before")
        if not isinstance(pb, dict) or not isinstance(pb.get("x"), (int, float)):
            continue
        if pb["x"] > pillar_x_max:
            c["n"] += 1
            a = r.get("action_expert")
            c["yaw"] += a in YAW_ACTIONS
            c["fwd"] += a == FORWARD
            c["safe"] += a == SAFE
            c["yaw_executed"] += r.get("action_executed") in YAW_ACTIONS
    return c


def count_flips(rows):
    """相邻 L↔R 换向：本步与上一步都非碰撞、都是 YAW、方向不同（a10_teacher_flipflop.py 的口径）。"""
    n = 0
    for i in range(1, len(rows)):
        cur, prev = rows[i], rows[i - 1]
        if cur.get("collision") or prev.get("collision"):
            continue
        a, b = cur.get("action_expert"), prev.get("action_expert")
        if a in YAW_ACTIONS and b in YAW_ACTIONS and a != b:
            n += 1
    return n


def load_arm(out, arm):
    """合并两半：返回 dict；缺任何一半的 eval.json 时 complete=False，已有的照样汇总。"""
    res = {"arm": arm, "halves": {}, "episodes": [], "complete": True, "teacher_versions": set(), "code_head": None}
    goal_margin = None
    for h in HALVES:
        name = f"{arm}-{h}"
        ev = load_json(os.path.join(out, name, "eval.json"))
        if ev is None:
            res["halves"][name] = "无 eval.json"
            res["complete"] = False
            continue
        res["halves"][name] = ("partial" if ev.get("partial") else "完成") + \
            f"（episodes_done={ev.get('episodes_done')}，failed_count={ev.get('failed_count')}）"
        if ev.get("partial") or ev.get("failed_count"):
            res["complete"] = False
        gm = ev.get("goal_margin")
        goal_margin = gm if goal_margin is None else goal_margin
        if gm != goal_margin:
            raise SystemExit(f"{name} 的 goal_margin={gm} 与另一半的 {goal_margin} 不同")
        for pe in ev.get("per_episode", []):
            seed = pe.get("seed")
            ep_dir = os.path.join(out, name, "episodes", f"seed_{seed}")
            rows = episode_rows(ep_dir)
            man = load_json(os.path.join(ep_dir, "episode_manifest.json")) or {}
            if man.get("teacher_version"):
                res["teacher_versions"].add(man["teacher_version"])
            ep = {"half": h, "seed": seed, "success": bool(pe.get("success")), "collision": bool(pe.get("collision")),
                  "corridor_violation": bool(pe.get("corridor_violation")), "timed_out": bool(pe.get("timed_out")),
                  "steps": pe.get("steps"), "goal_x": pe.get("goal_x"), "spl": pe.get("spl"), "max_x": pe.get("max_x"),
                  "exit": None, "flips": None, "safe_steps": None}
            if rows is not None:
                ep["flips"] = count_flips(rows)
                ep["safe_steps"] = sum(1 for r in rows if r.get("action_expert") == SAFE)
            if rows is not None and pe.get("goal_x") is not None and goal_margin is not None:
                ep["exit"] = exit_zone_counts(rows, float(pe["goal_x"]) - float(goal_margin))
            res["episodes"].append(ep)
    res["goal_margin"] = goal_margin
    res["episodes"].sort(key=lambda e: (e["seed"] is None, e["seed"]))
    return res


def arm_stats(arm):
    eps = arm["episodes"]
    n = len(eps)
    s = {"episodes": n, "success": sum(e["success"] for e in eps), "collision": sum(e["collision"] for e in eps),
         "corridor_violation": sum(e["corridor_violation"] for e in eps), "timed_out": sum(e["timed_out"] for e in eps),
         "mean_steps": (sum(e["steps"] for e in eps if e["steps"] is not None) / max(1, sum(e["steps"] is not None for e in eps)))
         if eps else None}
    z = {"n": 0, "yaw": 0, "fwd": 0, "safe": 0, "yaw_executed": 0}
    n_with_exit = 0
    for e in eps:
        if e["exit"] is not None:
            n_with_exit += 1
            for k in z:
                z[k] += e["exit"][k]
    s["exit"] = z
    s["episodes_with_exit_record"] = n_with_exit
    s["exit_yaw_share"] = z["yaw"] / z["n"] if z["n"] else None
    s["exit_yaw_ci95"] = wilson(z["yaw"], z["n"]) if z["n"] else None
    s["episodes_with_exit_yaw"] = sum(1 for e in eps if e["exit"] and e["exit"]["yaw"] > 0)
    s["flips"] = sum(e["flips"] for e in eps if e["flips"] is not None)
    s["episodes_with_flip"] = sum(1 for e in eps if e["flips"])
    s["safe_steps"] = sum(e["safe_steps"] for e in eps if e["safe_steps"] is not None)
    s["steps_total"] = sum(e["steps"] for e in eps if e["steps"] is not None)
    return s


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def build(out, note=""):
    arms = {a: load_arm(out, a) for a in ARMS}
    stats = {a: arm_stats(arms[a]) for a in ARMS}
    v7, v5 = stats["v7"], stats["v5"]
    gate_sr = v7["episodes"] == EXPECTED_EPISODES and v7["success"] == EXPECTED_EPISODES and arms["v7"]["complete"]
    gate_share = v7["exit_yaw_share"] is not None and v7["exit_yaw_share"] < GATE_SHARE and v7["exit"]["n"] > 0
    summary = {
        "note": note, "out": os.path.abspath(out),
        "gate": {"success_20_of_20": gate_sr, "exit_yaw_share_lt_5pct": gate_share, "threshold_share": GATE_SHARE},
        "arms": {a: {"complete": arms[a]["complete"], "halves": arms[a]["halves"],
                     "teacher_versions": sorted(arms[a]["teacher_versions"]), "goal_margin": arms[a]["goal_margin"],
                     "stats": stats[a], "episodes": arms[a]["episodes"]} for a in ARMS},
    }
    L = []
    L.append("# 教师 v7 闭环对拍汇总")
    L.append("")
    if note:
        L.append(f"> {note}")
        L.append("")
    L.append("## 1. 验收（任务卡 W1-7 验收 2）")
    L.append("")
    L.append("| 项 | 门槛 | 实测 | 结论 |")
    L.append("|---|---|---|---|")
    L.append(f"| v7 臂成功（20 个冻结 seed，2 条车道） | 20/20 | {v7['success']}/{v7['episodes']}"
             f"{'' if arms['v7']['complete'] else '（未完成）'} | {'通过' if gate_sr else '未过'} |")
    ci = v7["exit_yaw_ci95"]
    ci_txt = "—" if ci is None else f"Wilson 95% [{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"
    L.append(f"| v7 臂出口区 YAW 步占比 | < 5% | {v7['exit']['yaw']}/{v7['exit']['n']} = {pct(v7['exit_yaw_share'])}，{ci_txt} "
             f"| {'通过' if gate_share else '未过'} |")
    L.append("")
    L.append("## 2. 两臂对照（同 seed、同参数、同时段；v5 对照臂只作参照，不设门）")
    L.append("")
    L.append("| 臂 | manifest 教师版本 | 集数 | 成功 | 撞柱 | 越界 | 超时 | 平均步数 | 出口区步数 | 出口区 YAW | YAW 占比 | 95% 区间 | 出口区 FWD | 出口区 SAFE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        s, arm = stats[a], arms[a]
        ci = s["exit_yaw_ci95"]
        ci_txt = "—" if ci is None else f"[{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"
        ms = "—" if s["mean_steps"] is None else f"{s['mean_steps']:.1f}"
        L.append(f"| {a}{'' if arm['complete'] else '（未完成）'} | {', '.join(sorted(arm['teacher_versions'])) or '—'} | {s['episodes']} "
                 f"| {s['success']} | {s['collision']} | {s['corridor_violation']} | {s['timed_out']} | {ms} | {s['exit']['n']} "
                 f"| {s['exit']['yaw']} | {pct(s['exit_yaw_share'])} | {ci_txt} | {s['exit']['fwd']} | {s['exit']['safe']} |")
    L.append("")
    for a in ARMS:
        L.append(f"- {a}：" + "；".join(f"{k} {v}" for k, v in arms[a]["halves"].items()))
    L.append("")
    L.append("其他读数（只报告，不设门）：")
    L.append("")
    L.append("| 臂 | 总步数 | 相邻 L↔R 换向 | 有换向的集 | SAFE 步 | 出口区里有 YAW 的集 |")
    L.append("|---|---|---|---|---|---|")
    for a in ARMS:
        s = stats[a]
        L.append(f"| {a} | {s['steps_total']} | {s['flips']} | {s['episodes_with_flip']} | {s['safe_steps']} | {s['episodes_with_exit_yaw']} |")
    L.append("")
    L.append("## 3. 逐 seed（出口区 YAW 步数 / 出口区步数）")
    L.append("")
    L.append("| seed | v7 成功 | v7 出口区 YAW/步 | v5 成功 | v5 出口区 YAW/步 |")
    L.append("|---|---|---|---|---|")
    by = {a: {e["seed"]: e for e in arms[a]["episodes"]} for a in ARMS}

    def cell(e):
        if e is None:
            return "—", "—"
        z = e["exit"]
        return ("是" if e["success"] else "否"), ("—" if z is None else f"{z['yaw']}/{z['n']}")
    for seed in sorted(set(by["v7"]) | set(by["v5"]), key=lambda s: (s is None, s)):
        a7, b7 = cell(by["v7"].get(seed))
        a5, b5 = cell(by["v5"].get(seed))
        L.append(f"| {seed} | {a7} | {b7} | {a5} | {b5} |")
    L.append("")
    L.append("## 4. 口径")
    L.append("")
    L.append("- 出口区 = 步的 `pose_before.x` > 最远柱 x（`goal_x − goal_margin`）；集在 `x ≥ goal_x` 即止，所以只有柱阵后约 1 m 一段。")
    L.append("- YAW 步 = `action_expert` ∈ {YAW_LEFT, YAW_RIGHT}；教师臂 `action_executed` 与之相同"
             f"（v7 臂出口区 `action_executed` YAW 数 {v7['exit']['yaw_executed']}，v5 臂 {v5['exit']['yaw_executed']}）。")
    L.append("- 占比按步池化（不是每集占比的均值）。区间是 Wilson 95%，样本只有一百多步，区间宽；门槛判的是点估计。")
    L.append("- 成功 = eval.json 的 `success`（metric_version 带 corridor3m）。v5 对照臂的代码是开工时 main 的 HEAD（见 inputs.txt）。")
    L.append("- 相邻 L↔R 换向 = 本步与上一步都非碰撞、都是 YAW、方向不同（a10_teacher_flipflop.py 口径）；A* 无路（path_none）评测输出里没有记录，离线数字见 offline_0417.py。")
    L.append("")
    return "\n".join(L) + "\n", summary


def self_check():
    """合成两臂两半：v7 出口区 1 个 YAW / 12 步（8.3%，未过门），v5 出口区 5/12；成功 v7 3/4、v5 4/4。"""
    def act(x, a):
        return {"pose_before": {"x": x, "y": 0.0}, "action_expert": a, "action_executed": a, "collision": False}

    def write_arm(root, name, eps, goal_margin=1.0):
        per = []
        for seed, success, rows, gx in eps:
            d = os.path.join(root, name, "episodes", f"seed_{seed}")
            os.makedirs(d)
            with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"teacher_version": "uavrgb.teacher.v7" if name.startswith("v7") else "uavrgb.teacher.v5"}, f)
            per.append({"seed": seed, "success": success, "collision": not success, "corridor_violation": False,
                        "timed_out": False, "steps": len(rows), "goal_x": gx, "spl": 0.9, "max_x": gx})
        with open(os.path.join(root, name, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"partial": False, "episodes_done": len(eps), "failed_count": 0, "goal_margin": goal_margin,
                       "per_episode": per}, f)

    def rows(n_in, n_exit, yaw_in_exit, gx=6.0, flip=False):
        # 阵内 n_in 步（x ≤ 5.0 = gx − 1）含 2 个 YAW（flip=True 时一左一右，算 1 次相邻换向）；出口区 n_exit 步，其中 yaw_in_exit 个 YAW
        r = [act(1.0 + 0.1 * i, (3 if (flip and i == 1) else 2) if i < 2 else 1) for i in range(n_in)]
        r += [act(5.0 + 0.05 * (i + 1), 3 if i < yaw_in_exit else 1) for i in range(n_exit)]
        return r

    with tempfile.TemporaryDirectory() as t:
        write_arm(t, "v7-a", [(1, True, rows(10, 3, 0), 6.0), (2, True, rows(10, 3, 1), 6.0)])
        write_arm(t, "v7-b", [(3, True, rows(10, 3, 0), 6.0), (4, False, rows(10, 3, 0), 6.0)])
        write_arm(t, "v5-a", [(1, True, rows(10, 3, 2, flip=True), 6.0), (2, True, rows(10, 3, 1), 6.0)])
        write_arm(t, "v5-b", [(3, True, rows(10, 3, 1), 6.0), (4, True, rows(10, 3, 1), 6.0)])
        md, s = build(t, "自检")
    st7, st5 = s["arms"]["v7"]["stats"], s["arms"]["v5"]["stats"]
    assert st7["episodes"] == 4 and st7["success"] == 3 and st7["collision"] == 1, st7
    assert st7["exit"] == {"n": 12, "yaw": 1, "fwd": 11, "safe": 0, "yaw_executed": 1}, st7["exit"]
    assert abs(st7["exit_yaw_share"] - 1 / 12) < 1e-12
    assert st5["exit"]["n"] == 12 and st5["exit"]["yaw"] == 5 and st5["success"] == 4, st5
    assert st7["flips"] == 0 and st5["flips"] == 1 and st5["episodes_with_flip"] == 1, (st7["flips"], st5["flips"])
    assert st7["safe_steps"] == 0 and st7["steps_total"] == 4 * 13, st7
    assert s["gate"]["success_20_of_20"] is False and s["gate"]["exit_yaw_share_lt_5pct"] is False   # 4 集 ≠ 20，8.3% ≥ 5%
    assert s["arms"]["v7"]["teacher_versions"] == ["uavrgb.teacher.v7"] and s["arms"]["v5"]["teacher_versions"] == ["uavrgb.teacher.v5"]
    assert "| v7 臂成功" in md and "未过" in md
    lo, hi = wilson(1, 12)
    assert 0.0 < lo < 1 / 12 < hi < 1.0
    print("summarize.py 自检通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--note", default="")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.out:
        ap.error("需要 <OUT> 目录，或 --self-check")
    md, summary = build(args.out, args.note)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False, default=list)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
