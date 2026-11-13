#!/usr/bin/env python3
"""视觉语言导航（VLN）任务层：一集 = 一句自然语言指令 + 一个目标，智能体自己决定什么时候停。

指令来自 SAGE-3D，tools/scene_import_sage3d.py 已经导进 scene.json：routes[i].sage.instructions 每条是
{instruction_type, start, end, text}，start / end 是起点、终点物体名（如 track_light_1）。
目标 = 航线 waypoint（SAGE 轨迹终点，就在终点物体旁边），成功半径 = scene.reach_radius_m。

STOP：四个动作里没有单独的 STOP，约定连续 STOP_STREAK 步 SAFE_NEUTRAL = 「我到了」
（与 A* 教师的 goal-stop 同口径），collect_3dgs.run_episode(stop_after_safe=STOP_STREAK) 在那一步结束本集。

指标与 VLN-CE / Anderson et al. 2018 同口径，距离都在导航系 xy 平面上量：
  SR   自己停下时离目标 <= 半径，且全程没撞
  OSR  撞之前轨迹上任一点进过半径（停对了地方就能成功的上限）
  NE   结束时离目标的距离，米
  SPL  SR * l / max(p, l)；l = 起点到目标圆盘的最短路（eval_3dgs.route_shortest_path_m），p = 实际飞过的路

旧的 3dgs.pilot.v2 成功判据（轨迹进过半径就算，不要求停）在这里就是 OSR。
"""
import math
import statistics

STOP_STREAK = 3


def episodes(scene, splits=None, types=None):
    """scene.json 里所有 (航线, 指令) 组合，按航线、指令顺序。splits / types 给了就只留 SAGE split / 句式在其中的。"""
    out = []
    for r, route in enumerate(scene.get("routes") or []):
        sage = route.get("sage") or {}
        if splits and sage.get("split") not in splits:
            continue
        for k, ins in enumerate(sage.get("instructions") or []):
            if types and ins.get("instruction_type") not in types:
                continue
            out.append({"key": f"{scene.get('name')}/{route.get('name')}/{k}",
                        "route_index": r, "route_name": route.get("name"), "instruction_index": k,
                        "split": sage.get("split"), "type": ins.get("instruction_type"),
                        "text": ins["text"], "start_object": ins.get("start"), "goal_object": ins.get("end")})
    return out


def one_per_route(eps, seed=0):
    """每条航线留一条指令，句式按航线轮换（seed 决定起点），让各句式在小样本里都出现。
    各航线里指令的排列顺序不一致，所以先按句式名排序再轮换。"""
    by_route = {}
    for ep in eps:
        by_route.setdefault(ep["route_index"], []).append(ep)
    picked = []
    for n, r in enumerate(sorted(by_route)):
        group = sorted(by_route[r], key=lambda e: (str(e["type"]), e["instruction_index"]))
        picked.append(group[(n + seed) % len(group)])
    return picked


def trajectory(actions, oracle):
    """[(x, y)]：第一步动作前的位姿，再接每步结束时的真值位姿。"""
    if not actions:
        return []
    p0 = actions[0]["pose_before"]
    return [(float(p0["x"]), float(p0["y"]))] + [(float(o["pose_gt"]["x"]), float(o["pose_gt"]["y"])) for o in oracle]


def episode_metrics(actions, oracle, goal, radius, stopped, shortest_m=None):
    """单集 VLN 指标。actions / oracle 是 run_episode 写的逐步记录；stopped = 智能体自己喊了停。"""
    pts = trajectory(actions, oracle)
    if not pts:
        raise ValueError("这一集没有任何一步")
    gx, gy = float(goal["x"]), float(goal["y"])
    dist = [math.hypot(x - gx, y - gy) for x, y in pts]
    first_hit = next((i for i, a in enumerate(actions) if a.get("collision")), None)
    collided = first_hit is not None
    safe = dist if first_hit is None else dist[:first_hit + 1]      # 撞的那一步结束时已在障碍里，不算进过
    path = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    ne = dist[-1]
    success = bool(stopped and not collided and ne <= radius)
    spl = None
    if shortest_m is not None:
        spl = float(success) * shortest_m / max(path, shortest_m, 1e-9)
    return {"success": success, "oracle_success": min(safe) <= radius, "ne_m": ne,
            "spl": spl, "path_m": path, "shortest_m": shortest_m, "stopped": bool(stopped),
            "collision": collided, "steps": len(actions)}


def summarize(rows):
    """rows：episode_metrics 的结果列表。SPL 只在有 l 的集上平均（n_spl 记有几集）。"""
    n = len(rows)
    if not n:
        return {"n": 0}
    spl = [r["spl"] for r in rows if r["spl"] is not None]
    ne = [r["ne_m"] for r in rows]
    return {"n": n,
            "sr": sum(r["success"] for r in rows) / n,
            "osr": sum(r["oracle_success"] for r in rows) / n,
            "spl": sum(spl) / len(spl) if spl else None, "n_spl": len(spl),
            "ne_mean_m": sum(ne) / n, "ne_median_m": statistics.median(ne),
            "collision_rate": sum(r["collision"] for r in rows) / n,
            "stop_rate": sum(r["stopped"] for r in rows) / n}


def summarize_by(rows, key):
    """按 rows[i][key]（如句式 type、SAGE split）分组汇总。"""
    groups = {}
    for r in rows:
        groups.setdefault(r.get(key), []).append(r)
    return {str(k): summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}
