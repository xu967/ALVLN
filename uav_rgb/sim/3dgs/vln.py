#!/usr/bin/env python3
"""视觉语言导航（VLN）任务层：一集 = 一句自然语言指令 + 一个目标，智能体自己决定什么时候停。

指令来自 SAGE-3D，tools/scene_import_sage3d.py 已经导进 scene.json：routes[i].sage.instructions 每条是
{instruction_type, start, end, text}，start / end 是起点、终点物体名（如 track_light_1）。
目标 = 航线 waypoint（SAGE 轨迹终点，就在终点物体旁边），成功半径 = scene.reach_radius_m。

STOP：四个飞控动作不变，高层策略用独立 bool stop_requested 表示「任务完成」。
collect_3dgs.run_episode(stop_on_request=True) 将它执行为一个 SAFE 窗口，随后结束并按末位姿评测。
安全等待/避障 SAFE 不结束任务；这不是稳定悬停验收。旧 v1/v2 连续 STOP_STREAK 步 SAFE 只用于显式 legacy-safe 复现。

参考 VLN-CE / Anderson et al. 2018 的指标名称；本项目用 xy 平面、目标圆盘、无碰撞约束，不能直接比公开榜单：
  SR   自己停下时离目标 <= 半径，且全程没撞
  OSR  撞之前轨迹上任一点进过半径（停对了地方就能成功的上限）
  NE   结束时离目标的距离，米
  SPL  SR * l / max(p, l)；l = 起点到目标圆盘的最短路（eval_3dgs.route_shortest_path_m），p = 实际飞过的路

旧的 3dgs.pilot.v2 成功判据（轨迹进过半径就算，不要求停）在这里就是 OSR。
"""
import json
from itertools import chain, islice, zip_longest
import math
import os
import statistics

STOP_STREAK = 3     # 旧 v1/v2 兼容约定；新显式 STOP 不使用
SAFE_NEUTRAL, YAW_LEFT, YAW_RIGHT = 0, 2, 3
NDTW_STEP_M = 0.25  # 参考路径等距重采样步长（与 E1 单步位移同量级，消除源数据点密度差异）
SUBGOAL_BEARING_CONTRACT = {
    "planner": "OccPlanner.plan+aim_point",
    "lookahead_base_m": 0.65,
    "lookahead_speed_gain_s": 0.8,
    "los_max_m": 3.0,
    "hard_los_max_m": 1.0,
    "near_goal_radius": "scene.reach_radius_m",
    "speed_source": "horizontal_nav_velocity_mps",
}


def bearing_bin(yaw_rad, pos_xy, aim_xy, n_bins=8):
    """机头朝向到瞄准点的相对方位桶（0..n_bins-1，逆时针，桶 0 以正前方为中心）。"""
    if n_bins < 1:
        raise ValueError("n_bins 必须 >= 1")
    rel = (math.atan2(float(aim_xy[1]) - float(pos_xy[1]),
                      float(aim_xy[0]) - float(pos_xy[0])) - float(yaw_rad)) % (2.0 * math.pi)
    width = 2.0 * math.pi / n_bins
    return int(((rel + 0.5 * width) % (2.0 * math.pi)) // width)


def subgoal_bearing_bin(occ_planner, yaw_rad, pos_xy, goal_xy, speed_mps, reach_radius_m):
    """按运行时 A* 子目标、速度 lookahead 与任务到达半径计算 VLN 条件。"""
    px, py = float(pos_xy[0]), float(pos_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    if math.hypot(px - gx, py - gy) <= float(reach_radius_m):
        return bearing_bin(yaw_rad, (px, py), (gx, gy)), 1
    if occ_planner is not None:
        path = occ_planner.plan((px, py), (gx, gy))
        if path is not None:
            lookahead = SUBGOAL_BEARING_CONTRACT["lookahead_base_m"] + \
                SUBGOAL_BEARING_CONTRACT["lookahead_speed_gain_s"] * float(speed_mps)
            tx, ty = occ_planner.aim_point(
                path, px, py, lookahead=lookahead,
                los_max=SUBGOAL_BEARING_CONTRACT["los_max_m"],
                hard_los_max=min(SUBGOAL_BEARING_CONTRACT["hard_los_max_m"], lookahead))
            return bearing_bin(yaw_rad, (px, py), (tx, ty)), 0
    return bearing_bin(yaw_rad, (px, py), (gx, gy)), 0


def resample_polyline(points, step_m=NDTW_STEP_M):
    """按固定弧长 step_m 对 2D/3D 折线等距重采样（保留首尾点）。"""
    pts = [tuple(float(v) for v in p) for p in (points or ())]
    if len(pts) < 2 or step_m <= 0:
        return pts
    out, acc = [pts[0]], 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        while acc + seg >= step_m and seg > 1e-12:
            t = (step_m - acc) / seg
            a = tuple(a[k] + t * (b[k] - a[k]) for k in range(len(a)))
            out.append(a)
            seg = math.dist(a, b)
            acc = 0.0
        acc += seg
    if math.dist(out[-1], pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


def ndtw_sdtw(pred_pts, ref_pts, threshold_m, success):
    """精确动态规划计算 nDTW 与 SDTW（Ilharco et al. 2019）。"""
    if not pred_pts or not ref_pts or threshold_m <= 0:
        return None, None
    prev = [math.inf] * (len(ref_pts) + 1)
    prev[0] = 0.0
    for p in pred_pts:
        curr = [math.inf] * (len(ref_pts) + 1)
        for j, q in enumerate(ref_pts, 1):
            curr[j] = math.dist(p, q) + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    ndtw = math.exp(-prev[-1] / (len(ref_pts) * float(threshold_m)))
    return ndtw, (float(bool(success)) * ndtw)


def _matching_reference_path(route, points):
    """截断任务可能沿用旧参考路径。单段路径首尾须与当前任务各相距不超过 5 cm。"""
    if not isinstance(points, list) or len(points) < 2:
        return None
    try:
        path = [(float(p[0]), float(p[1])) for p in points]
        start = tuple(float(route["spawn"][k]) for k in ("x", "y"))
        goal = tuple(float(route["waypoint"][k]) for k in ("x", "y"))
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in path):
        return None
    if math.dist(start, path[0]) <= 0.05 and math.dist(goal, path[-1]) <= 0.05:
        return path
    return None


def sage_ref_paths(scene, sage_vln_root=None):
    """按航线索引返回已对齐的单段参考路径，内嵌路径优先，其次查 SAGE 原始标注。"""
    routes = scene.get("routes") or []
    out = {}
    for idx, r in enumerate(routes):
        path = _matching_reference_path(r, r.get("ref_path"))
        if path is not None:
            out[idx] = path
    if len(out) == len(routes):
        return out
    name = str(scene.get("name") or "")
    sid = name.split("sage3d_", 1)[1] if name.startswith("sage3d_") else None
    root = sage_vln_root or os.path.join(
        os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "pubdata", "sage3d", "vln")
    scene_vln = os.path.join(root, sid) if sid else None
    if not scene_vln or not os.path.isdir(scene_vln):
        return out
    by_key = {}
    for split in sorted(os.listdir(scene_vln)):
        split_dir = os.path.join(scene_vln, split)
        if not os.path.isdir(split_dir):
            continue
        for fn in sorted(os.listdir(split_dir)):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(split_dir, fn), encoding="utf-8") as f:
                doc = json.load(f)
            for sc in doc.get("scenes", []):
                for smp in sc.get("samples", []):
                    pts = [p["position"] for p in smp.get("points", []) if "position" in p]
                    if len(pts) >= 2:
                        by_key[(split, int(smp["trajectory_id"]))] = pts
    for idx, r in enumerate(routes):
        if idx in out:
            continue
        sage = r.get("sage") or {}
        path = _matching_reference_path(r, by_key.get((sage.get("split"), sage.get("trajectory_id"))))
        if path is not None:
            out[idx] = path
    return out


def episodes(scene, splits=None, types=None):
    """返回单段指令集和显式版本化的 SAGE 多阶段任务。"""
    out = []
    for r, route in enumerate(scene.get("routes") or []):
        sage = route.get("vln") or route.get("sage") or {}
        if splits and sage.get("split") not in splits:
            continue
        for k, ins in enumerate(sage.get("instructions") or []):
            if types and ins.get("instruction_type") not in types:
                continue
            out.append({"key": f"{scene.get('name')}/{route.get('name')}/{k}",
                        "route_index": r, "route_name": route.get("name"), "instruction_index": k,
                        "route_key": f"route:{r}",
                        "split": sage.get("split"), "type": ins.get("instruction_type"),
                        "text": ins["text"], "start_object": ins.get("start"), "goal_object": ins.get("end")})
    for index, route in enumerate(scene.get("multi_stage_routes") or []):
        if route.get("route_contract") != "sage.multi_stage.v2":
            continue
        stages = route.get("stages") or []
        if len(stages) < 2 or not str(route.get("instruction") or "").strip():
            continue
        if splits and route.get("split") not in splits:
            continue
        if types and "Multi-stage" not in types:
            continue
        stage_waypoints = [
            {"stage": int(stage.get("stage", stage_index)),
             "waypoint": {axis: float(stage["waypoint"][axis]) for axis in ("x", "y")},
             "goal_object": stage.get("goal_object"), "instruction": stage.get("instruction"),
             "path_m": float(stage.get("path_m") or 0.0)}
            for stage_index, stage in enumerate(stages)
        ]
        out.append({"key": f"{scene.get('name')}/{route.get('name')}/multi_stage",
                    "route_index": len(scene.get("routes") or []) + index,
                    "route_name": route.get("name"), "instruction_index": 0,
                    "route_key": f"multi_stage:{index}", "route_contract": "sage.multi_stage.v2",
                    "multi_stage_index": index, "stage_waypoints": stage_waypoints,
                    "total_path_m": float(route.get("total_path_m") or
                                          sum(stage["path_m"] for stage in stage_waypoints)),
                    "ref_path": route.get("ref_path"), "split": route.get("split"),
                    "type": "Multi-stage", "text": route["instruction"],
                    "start_object": (stages[0] or {}).get("start_object"),
                    "goal_object": stages[-1].get("goal_object")})
    return out


def one_per_route(eps, seed=0):
    """每条航线留一条指令，句式按航线轮换（seed 决定起点），让各句式在小样本里都出现。
    各航线里指令的排列顺序不一致，所以先按句式名排序再轮换。"""
    by_route = {}
    for ep in eps:
        by_route.setdefault(ep.get("route_key", f"route:{ep['route_index']}"), []).append(ep)
    picked = []
    def route_order(key):
        kind, index = str(key).split(":", 1)
        return (0 if kind == "route" else 1, int(index))

    for n, r in enumerate(sorted(by_route, key=route_order)):
        group = sorted(by_route[r], key=lambda e: (str(e["type"]), e["instruction_index"]))
        picked.append(group[(n + seed) % len(group)])
    return picked


def select_requests(episodes, *, all_instructions, limit, seed):
    """各臂使用相同名单。全部指令模式有截断时，先遍历各航线的第一句，再取第二句。"""
    if not all_instructions:
        picked = one_per_route(episodes, seed)
        return picked[:limit] if limit else picked
    if not 0 < limit < len(episodes):
        return episodes
    routes = {}
    for ep in episodes:
        routes.setdefault(ep["route_key"], []).append(ep)
    ordered = (ep for turn in zip_longest(*routes.values()) for ep in turn if ep is not None)
    return list(islice(ordered, limit))


def swap_partners(scene, requested, candidates):
    """从当前 split 的完整指令池找相同实验条件的异目标。优先从本次请求中互换。"""
    conditions = {}
    routes = (scene.get("routes") or []) + (scene.get("multi_stage_routes") or [])
    for i, route in enumerate(routes):
        try:
            spawn = route["spawn"]
            pose = tuple(float(spawn[k]) for k in ("x", "y", "z")) + (float(spawn.get("yaw_deg", 0)),)
            budget = int(route.get("max_steps", scene.get("max_steps")))
            xy = tuple(float(route["waypoint"][k]) for k in ("x", "y"))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if budget > 0 and all(math.isfinite(value) for value in (*pose, *xy)):
            conditions[i] = (pose, budget)

    keys = {ep["key"] for ep in requested}
    ordered = chain(requested, (ep for ep in candidates if ep["key"] not in keys))
    groups = {}
    for ep in ordered:
        condition = conditions.get(ep["route_index"])
        labels = (ep.get("split"), ep.get("type"), ep.get("goal_object"), ep.get("text"))
        if condition is None or not all(isinstance(s, str) and s.strip() for s in labels):
            continue
        group = (condition, ep.get("route_contract"), ep.get("split"), ep.get("type"),
                 len(ep.get("stage_waypoints") or ()))
        groups.setdefault(group, []).append(ep)

    partners = {}
    for group in groups.values():
        if len({ep["goal_object"] for ep in group}) < 2:
            continue
        selected = [ep for ep in group if ep["key"] in keys]
        outside = group[len(selected):]
        for i, ep in enumerate(selected):
            # 相邻请求优先互换；同一目标的改写不能当作语言换目标。
            direction = 1 if i % 2 == 0 else -1
            nearby = (selected[(i + direction * offset) % len(selected)] for offset in range(1, len(selected)))
            for other in chain(nearby, outside):
                if other["goal_object"] != ep["goal_object"] and other["text"] != ep["text"]:
                    partners[ep["key"]] = other
                    break
    return partners


def trajectory(actions, oracle):
    """[(x, y)]：第一步动作前的位姿，再接每步结束时的真值位姿。"""
    if not actions:
        return []
    p0 = actions[0]["pose_before"]
    return [(float(p0["x"]), float(p0["y"]))] + [(float(o["pose_gt"]["x"]), float(o["pose_gt"]["y"])) for o in oracle]


def doorway_passages(trace, actions, doors):
    """按模拟时间返回每扇 CAD 门首次正向穿越的 {door, step, t}。碰撞窗口整段排除。"""
    if not actions or not trace:
        return []
    cutoff = next((a.get("step", i) for i, a in enumerate(actions) if a.get("collision")), math.inf)
    start = actions[0]["pose_before"]
    samples = [(0.0, start["x"], start["y"], 0)]
    for row in trace:
        if row["step"] >= cutoff:
            break
        if row["t"] <= samples[-1][0]:
            raise ValueError("穿门评测需要时间严格递增的轨迹")
        samples.append((row["t"], row["pos"][0], row["pos"][1], row["step"]))

    events = []
    for name, door in doors.items():
        (ax, ay), (bx, by) = door["line_xy_m"]
        nx, ny = door["forward_normal_xy"]
        dx, dy = bx - ax, by - ay
        for (t0, x0, y0, _), (t1, x1, y1, step) in zip(samples, samples[1:]):
            side0 = (x0 - ax) * nx + (y0 - ay) * ny
            side1 = (x1 - ax) * nx + (y1 - ay) * ny
            if not side0 < 0 <= side1:
                continue
            fraction = -side0 / (side1 - side0)
            x, y = x0 + fraction * (x1 - x0), y0 + fraction * (y1 - y0)
            # 门框端点不算通过。刚好落在门线上的观测归于到达该点的轨迹段。
            along = (x - ax) * dx + (y - ay) * dy
            if 0 < along < dx * dx + dy * dy:
                events.append({"door": name, "step": step, "t": t0 + fraction * (t1 - t0)})
                break
    return sorted(events, key=lambda event: (event["t"], event["door"]))


def _segment_circle_entry(start, end, goal, radius, min_fraction=0.0):
    """返回线段上从 min_fraction 起进入目标圆的最早比例，没有进入则返回 None。"""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return min_fraction if math.dist(start, goal) <= radius else None
    projection = ((goal[0] - start[0]) * dx + (goal[1] - start[1]) * dy) / length_sq
    closest_x, closest_y = start[0] + projection * dx, start[1] + projection * dy
    offset = math.hypot(closest_x - goal[0], closest_y - goal[1])
    if offset > radius:
        return None
    half_width = math.sqrt(max(0.0, radius * radius - offset * offset) / length_sq)
    entry, exit_at = max(0.0, projection - half_width), min(1.0, projection + half_width)
    if exit_at < min_fraction or entry > 1.0:
        return None
    return max(entry, min_fraction)


def _action_diagnostics(actions, path_m):
    """统计动作切换与 SAFE 动作窗；未知动作不跨缺口配对，也不伪造 0。

    SAFE 这里只表示动作协议中的 ``SAFE_NEUTRAL`` 窗口，包含显式 STOP 被改写的
    SAFE 窗口，不等价于物理上已经停稳或悬停了多久。
    """
    values = []
    for row in actions:
        value = row.get("action_executed", row.get("action"))
        valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3
        values.append(value if valid else None)
    observed = sum(value is not None for value in values)
    pairs = [(a, b) for a, b in zip(values, values[1:]) if a is not None and b is not None]
    safe_steps = sum(value == SAFE_NEUTRAL for value in values if value is not None) if observed else None
    switches = sum(a != b for a, b in pairs) if pairs else None
    reversals = sum({a, b} == {YAW_LEFT, YAW_RIGHT} for a, b in pairs) if pairs else None
    return {
        "action_steps_observed": observed,
        "action_pairs_observed": len(pairs),
        "action_switches": switches,
        "action_switches_per_m": switches / path_m if switches is not None and path_m > 1e-9 else None,
        "safe_steps": safe_steps,
        "safe_ratio": safe_steps / observed if safe_steps is not None else None,
        "safe_steps_per_m": safe_steps / path_m if safe_steps is not None and path_m > 1e-9 else None,
        "turn_reversals": reversals,
    }


def episode_metrics(actions, oracle, goal, radius, stopped, shortest_m=None,
                    ref_path=None, ndtw_step_m=NDTW_STEP_M, stage_goals=None):
    """单集 VLN 指标。actions / oracle 是 run_episode 写的逐步记录；stopped = 智能体自己喊了停。"""
    pts = trajectory(actions, oracle)
    if not pts:
        raise ValueError("这一集没有任何一步")
    gx, gy = float(goal["x"]), float(goal["y"])
    dist = [math.hypot(x - gx, y - gy) for x, y in pts]
    first_hit = next((i for i, a in enumerate(actions) if a.get("collision")), None)
    collided = first_hit is not None
    safe_pts = pts if first_hit is None else pts[:first_hit + 1]
    safe = dist if first_hit is None else dist[:first_hit + 1]      # 撞的那一步结束时已在障碍里，不算进过
    path = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    ne = dist[-1]
    stage_goals = stage_goals or []
    stage_count = len(stage_goals)
    stage_goals_xy = [(float(stage["x"]), float(stage["y"])) for stage in stage_goals]
    stage_hits = [False] * stage_count
    stage_reached = 0
    for start_point, end_point in zip(safe_pts, safe_pts[1:]):
        fraction = 0.0
        while stage_reached < stage_count:
            for index, stage_goal in enumerate(stage_goals_xy):
                if _segment_circle_entry(start_point, end_point, stage_goal, radius) is not None:
                    stage_hits[index] = True
            fraction = _segment_circle_entry(start_point, end_point, stage_goals_xy[stage_reached], radius, fraction)
            if fraction is None:
                break
            stage_hits[stage_reached] = True
            stage_reached += 1
    if safe_pts:
        for index, stage_goal in enumerate(stage_goals_xy):
            if math.dist(safe_pts[-1], stage_goal) <= radius:
                stage_hits[index] = True
    all_stages_reached = stage_reached == stage_count
    stage_csr = None
    if stage_count:
        csr_total = 0.0
        for index, hit in enumerate(stage_hits):
            previous_hit = stage_hits[index - 1] if index else True
            csr_total += float(hit) * (1.0 + (stage_count - 1) * float(previous_hit))
        stage_csr = csr_total / (stage_count * stage_count)
    success = bool(stopped and not collided and ne <= radius and all_stages_reached)
    # 控制平顺性诊断不参与 SR/SPL；动作缺口保持在原位置，不能跨缺口统计切换。
    motion = _action_diagnostics(actions, path)
    step_count = len(actions)
    spl = None
    if shortest_m is not None:
        spl = float(success) * shortest_m / max(path, shortest_m, 1e-9)
    ndtw, sdtw = (ndtw_sdtw(resample_polyline(safe_pts, ndtw_step_m),
                            resample_polyline(ref_path, ndtw_step_m), radius, success)
                  if ref_path else (None, None))
    metrics = {"success": success, "oracle_success": (all_stages_reached if stage_count else min(safe) <= radius), "ne_m": ne,
            "spl": spl, "ndtw": ndtw, "sdtw": sdtw, "path_m": path, "shortest_m": shortest_m,
            "stopped": bool(stopped), "collision": collided, "steps": step_count, **motion}
    if stage_count:
        metrics.update({"stage_count": stage_count, "stage_reached_count": stage_reached,
                        "stage_hit_count": sum(stage_hits),
                        "stage_hit_flags": stage_hits,
                        "stage_sr": sum(stage_hits) / stage_count,
                        "stage_csr": stage_csr,
                        "stage_ordered_fraction": stage_reached / stage_count,
                        "stage_task_success": success})
    return metrics


def summarize_stages(rows, failures=()):
    """多阶段任务使用 LH-VLN ISR/CSR，并另报严格端到端 stage_task_sr。"""
    stage_rows = [row for row in rows if row.get("route_contract") == "sage.multi_stage.v2"]
    stage_failures = [row for row in failures if row.get("route_contract") == "sage.multi_stage.v2"]
    requested = len(stage_rows) + len(stage_failures)
    total_stages = sum(row.get("stage_count", len(row.get("stage_waypoints") or [])) for row in stage_rows)
    total_stages += sum(len(row.get("stage_waypoints") or []) for row in stage_failures)
    reached = sum(row.get("stage_hit_count", 0) for row in stage_rows)
    complete = sum(bool(row.get("stage_task_success")) for row in stage_rows)
    csr_values = [row.get("stage_csr") for row in stage_rows if row.get("stage_csr") is not None]
    return {"n_requested": requested, "n_valid": len(stage_rows), "n_stages_requested": total_stages,
            "stage_sr": reached / total_stages if total_stages else None,
            "stage_csr": sum(csr_values) / requested if requested else None,
            "stage_task_sr": complete / requested if requested else None,
            "definitions": {"stage_sr": "不要求次序，各阶段命中数 / 请求阶段数",
            "stage_csr": "LH-VLN CSR；按任务平均 Σ s_i(1+(n-1)s_(i-1))/n²，首阶段前驱取 1。失败集计 0",
                            "stage_task_sr": "全阶段按序达成、无碰撞且显式停在末阶段的任务数 / 请求任务数"}}


def summarize(rows, failures=()):
    """SR/OSR 分母为请求集，异常集记未成功；SPL/NE/CR/nDTW/SDTW 只量有效集（EVAL_STANDARD §5.6）。"""
    n = len(rows)
    requested = n + len(failures)
    spl = [r["spl"] for r in rows if r.get("spl") is not None]
    ndtw = [r["ndtw"] for r in rows if r.get("ndtw") is not None]
    sdtw = [r["sdtw"] for r in rows if r.get("sdtw") is not None]
    ne = [r["ne_m"] for r in rows]
    return {"n": n, "n_requested": requested, "n_failed": len(failures),
            "error_rate": len(failures) / requested if requested else None,
            "denominators": {"sr": requested, "osr": requested, "spl": len(spl),
                             "ndtw": len(ndtw), "sdtw": len(sdtw),
                             "ne": n, "collision_rate": n, "stop_rate": n},
            "sr": sum(r["success"] for r in rows) / requested if requested else None,
            "osr": sum(r["oracle_success"] for r in rows) / requested if requested else None,
            "spl": sum(spl) / len(spl) if spl else None, "n_spl": len(spl),
            "ndtw": sum(ndtw) / len(ndtw) if ndtw else None,
            "sdtw": sum(sdtw) / len(sdtw) if sdtw else None, "n_ndtw": len(ndtw),
            "ne_mean_m": sum(ne) / n if n else None, "ne_median_m": statistics.median(ne) if n else None,
            "collision_rate": sum(r["collision"] for r in rows) / n if n else None,
            "stop_rate": sum(r["stopped"] for r in rows) / n if n else None}


def summarize_by(rows, key, failures=()):
    """按 rows[i][key]（如句式 type、SAGE split）分组汇总。"""
    groups, failed_groups = {}, {}
    for r in rows:
        groups.setdefault(r.get(key), []).append(r)
    for r in failures:
        failed_groups.setdefault(r.get(key), []).append(r)
    return {str(k): summarize(groups.get(k, []), failed_groups.get(k, []))
            for k in sorted(groups.keys() | failed_groups.keys(), key=str)}
