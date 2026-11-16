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
import math
import os
import statistics

STOP_STREAK = 3     # 旧 v1/v2 兼容约定；新显式 STOP 不使用
SAFE_NEUTRAL, YAW_LEFT, YAW_RIGHT = 0, 2, 3
NDTW_STEP_M = 0.25  # 参考路径等距重采样步长（与 E1 单步位移同量级，消除源数据点密度差异）


def bearing_bin(yaw_rad, pos_xy, aim_xy, n_bins=8):
    """机头朝向到瞄准点的相对方位桶（0..n_bins-1，逆时针，桶 0 以正前方为中心）。"""
    if n_bins < 1:
        raise ValueError("n_bins 必须 >= 1")
    rel = (math.atan2(float(aim_xy[1]) - float(pos_xy[1]),
                      float(aim_xy[0]) - float(pos_xy[0])) - float(yaw_rad)) % (2.0 * math.pi)
    width = 2.0 * math.pi / n_bins
    return int(((rel + 0.5 * width) % (2.0 * math.pi)) // width)


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


def sage_ref_paths(scene, sage_vln_root=None):
    """返回 route_index -> [(x, y), ...] 参考路径表。优先读 route['ref_path']，否则查本地 pubdata/sage3d/vln。"""
    routes = scene.get("routes") or []
    out = {}
    for idx, r in enumerate(routes):
        if isinstance(r.get("ref_path"), list) and len(r["ref_path"]) >= 2:
            out[idx] = [(float(p[0]), float(p[1])) for p in r["ref_path"]]
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
                    pts = [(float(p["position"][0]), float(p["position"][1]))
                           for p in smp.get("points", []) if "position" in p]
                    if len(pts) >= 2:
                        by_key[(split, int(smp["trajectory_id"]))] = pts
    for idx, r in enumerate(routes):
        if idx in out:
            continue
        sage = r.get("sage") or {}
        pts = by_key.get((sage.get("split"), sage.get("trajectory_id")))
        if not pts:
            continue
        sp, wp = r.get("spawn") or {}, r.get("waypoint") or {}
        if ("x" in sp and math.hypot(pts[0][0] - float(sp["x"]), pts[0][1] - float(sp["y"])) <= 0.05
                and "x" in wp and math.hypot(pts[-1][0] - float(wp["x"]), pts[-1][1] - float(wp["y"])) <= 0.05):
            out[idx] = pts
    return out


def episodes(scene, splits=None, types=None):
    """scene.json 里所有 (航线, 指令) 组合，按航线、指令顺序。splits / types 给了就只留 SAGE split / 句式在其中的。"""
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
                    ref_path=None, ndtw_step_m=NDTW_STEP_M):
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
    success = bool(stopped and not collided and ne <= radius)
    # 控制平顺性诊断不参与 SR/SPL；动作缺口保持在原位置，不能跨缺口统计切换。
    motion = _action_diagnostics(actions, path)
    step_count = len(actions)
    spl = None
    if shortest_m is not None:
        spl = float(success) * shortest_m / max(path, shortest_m, 1e-9)
    ndtw, sdtw = (ndtw_sdtw(resample_polyline(safe_pts, ndtw_step_m),
                            resample_polyline(ref_path, ndtw_step_m), radius, success)
                  if ref_path else (None, None))
    return {"success": success, "oracle_success": min(safe) <= radius, "ne_m": ne,
            "spl": spl, "ndtw": ndtw, "sdtw": sdtw, "path_m": path, "shortest_m": shortest_m,
            "stopped": bool(stopped), "collision": collided, "steps": step_count, **motion}


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
