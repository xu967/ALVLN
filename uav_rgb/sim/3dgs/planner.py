#!/usr/bin/env python3
"""占据栅格 A* 教师（3DGS 多路线采集用）+ 路线离线可达性检查。

障碍来自 collision.Occupancy 的**膨胀栅格**（与碰撞判定同一份数据，机体包络已含），
不是 Gazebo 柱子真值；决策形状照搬 sim/tools/teacher.plan_action_global：
A* 路径 -> 纯追踪胡萝卜 -> 航向差超 tol 先转 -> 前向净空不够先转 -> 否则 FORWARD；
离 waypoint <= STOP_RADIUS_M 输出 SAFE_NEUTRAL（goal-stop，teacher v5 口径）。
A* 失败（被围死 / waypoint 在占据里）fail closed 输出 SAFE_NEUTRAL。

深度反应式 collect_3dgs.DepthTeacher 不知道 waypoint，开阔路线全程 FORWARD；
这里的教师是特权信息（占据栅格 + waypoint），只决定标签，不进学生输入。

自检：python -m sim.3dgs.planner
"""
import heapq
import math

import numpy as np

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
TEACHER_VERSION = "uavrgb.teacher_occ.v1"
# 软余量：硬栅格 = 机体包络膨胀后的占据；再向外 MARGIN_M 这一圈可走但每米加 MARGIN_COST 倍代价，
# 路径贴走廊中线、窄缝仍可通过。0.4 与 E1 Gazebo A* 余量同值（teacher.CLEARANCE_BY_VEHICLE）。
MARGIN_M = 0.4
MARGIN_COST = 4.0
LOOKAHEAD_M = 0.7      # 纯追踪胡萝卜（Gazebo 教师同值）；cf_sitl 每 0.5 s 步前进约 0.15 m
# 视线抽捷径上限：8 邻接 A* 的 0°/45° 阶梯路径会让纯追踪每 3–4 步换向（smoke_r1 实测 ±34° 来回摆），
# 先瞄最远的、与当前位置连线不穿软余量的路径点（<= LOS_MAX_M），其次不穿硬栅格的，都没有才退回胡萝卜。
LOS_MAX_M = 2.5
# cf_sitl 每 0.5 s 步偏航 17°（首步 14.5°，tmp/agent-3dgs-routes/probe_yaw.py 实测）；
# tol 取 0.7 步：小于半步会左右过冲（T1 门柱振荡的根因），大于一步会斜着飞出走廊。
YAW_TOL_DEG = 12.0
FORWARD_CHECK_M = 0.4  # 前向净空（硬栅格 sweep）；Gazebo 教师同值
STOP_RADIUS_M = 0.5    # goal-stop 半径（teacher v5 同值）；到达判定仍是 scene.reach_radius_m
# 步数预算：每步前进 0.15 m / 偏航 17°，加速与 goal-stop 约 8 步，再留 25%。
STEP_FORWARD_M, STEP_YAW_DEG, STEP_OVERHEAD, STEP_MARGIN = 0.15, 17.0, 8, 1.25
_MOVES = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))


def _dilate2d(mask, k):
    """2D 方形结构元素膨胀 k 格（纯 numpy 位移或），k=0 原样返回。"""
    out = mask.copy()
    nx, ny = mask.shape
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            if di == 0 and dj == 0:
                continue
            src = mask[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj)]
            out[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj)] |= src
    return out


def astar(blocked, cost, start, goal):
    """8 邻接 A*（禁止斜穿角点）。blocked 2D bool，cost 2D 额外代价（每格 >= 0）。

    起点强制视为自由（机体漂进膨胀带时还能规划出去，与 teacher.astar_path 同）；
    终点被占据返回 None。返回 [(i, j), ...] 含起终点。
    """
    nx, ny = blocked.shape
    if not (0 <= start[0] < nx and 0 <= start[1] < ny and 0 <= goal[0] < nx and 0 <= goal[1] < ny):
        return None
    if blocked[goal]:
        return None

    def free(i, j):
        return (i, j) == start or not blocked[i, j]

    def heur(i, j):
        dx, dy = abs(i - goal[0]), abs(j - goal[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)     # octile，代价 >= 1/格 所以可采纳

    g = {start: 0.0}
    parent = {}
    heap = [(heur(*start), 0.0, start)]
    while heap:
        _f, gc, cur = heapq.heappop(heap)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if gc > g[cur]:
            continue
        i, j = cur
        for di, dj, w in _MOVES:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not free(ni, nj):
                continue
            if di and dj and not (free(i + di, j) and free(i, j + dj)):
                continue
            ng = gc + w * (1.0 + cost[ni, nj])
            if ng < g.get((ni, nj), math.inf):
                g[(ni, nj)] = ng
                parent[(ni, nj)] = cur
                heapq.heappush(heap, (ng + heur(ni, nj), ng, (ni, nj)))
    return None


def _carrot(path, x, y, lookahead):
    """离当前点最近的路径点之后沿路径累计 lookahead 米的那一点（teacher._path_carrot 同）。"""
    best = min(range(len(path)), key=lambda k: (path[k][0] - x) ** 2 + (path[k][1] - y) ** 2)
    acc, (px, py) = 0.0, path[best]
    for qx, qy in path[best + 1:]:
        acc += math.hypot(qx - px, qy - py)
        px, py = qx, qy
        if acc >= lookahead:
            break
    return px, py


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


class OccPlanner:
    """Occupancy -> 2D 硬栅格（膨胀占据，沿 z 任一层）+ 软余量代价；plan() 在裁剪窗口上跑 A*。"""

    def __init__(self, occ, margin_m=MARGIN_M, margin_cost=MARGIN_COST):
        if occ.grid.size == 0:
            raise ValueError("占据栅格为空，无法规划")
        self.occ = occ
        self.voxel = float(occ.voxel)
        self.origin = np.asarray(occ.origin[:2], dtype=np.float64)
        self.hard = occ.grid.any(axis=2)
        k = int(round(margin_m / self.voxel))
        self.soft = _dilate2d(self.hard, k) & ~self.hard
        self.cost = np.where(self.soft, float(margin_cost), 0.0)
        self.margin_m, self.margin_cost = float(margin_m), float(margin_cost)

    def cell(self, x, y):
        c = np.floor((np.array([x, y], dtype=np.float64) - self.origin) / self.voxel).astype(int)
        return int(c[0]), int(c[1])

    def xy(self, i, j):
        return (float(self.origin[0] + (i + 0.5) * self.voxel),
                float(self.origin[1] + (j + 0.5) * self.voxel))

    def blocked_at(self, x, y):
        i, j = self.cell(x, y)
        return not (0 <= i < self.hard.shape[0] and 0 <= j < self.hard.shape[1]) or bool(self.hard[i, j])

    def line_free(self, mask, p, q):
        """2D 线段 p->q 按半格步距采样，是否全部落在 mask=False 的格子里（出栅格算堵）。"""
        p, q = np.asarray(p[:2], dtype=np.float64), np.asarray(q[:2], dtype=np.float64)
        n = int(math.ceil(np.linalg.norm(q - p) / (0.5 * self.voxel))) + 1
        pts = p[None, :] + np.linspace(0.0, 1.0, max(2, n))[:, None] * (q - p)[None, :]
        c = np.floor((pts - self.origin) / self.voxel).astype(int)
        ok = (c >= 0).all(axis=1) & (c[:, 0] < mask.shape[0]) & (c[:, 1] < mask.shape[1])
        return bool(ok.all() and not mask[c[:, 0], c[:, 1]].any())

    def aim_point(self, path, x, y, lookahead=LOOKAHEAD_M, los_max=LOS_MAX_M):
        """纯追踪目标点：视线抽捷径（软余量 -> 硬栅格），都不通时退回 _carrot。"""
        best = min(range(len(path)), key=lambda k: (path[k][0] - x) ** 2 + (path[k][1] - y) ** 2)
        far, acc = best, 0.0
        for k in range(best + 1, len(path)):
            acc += math.hypot(path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1])
            if acc > los_max:
                break
            far = k
        for mask in (self.hard | self.soft, self.hard):
            for k in range(far, best, -1):
                if self.line_free(mask, (x, y), path[k]):
                    return path[k]
        return _carrot(path, x, y, lookahead)

    def plan(self, start_xy, goal_xy, pad_m=6.0):
        """返回 [(x, y), ...]（导航系米，首点为起点格中心）或 None。

        先在起终点包围盒外扩 pad_m 的窗口里找；找不到再放大到 3 倍窗口（绕远的路），仍无则 None。
        """
        s, g = self.cell(*start_xy), self.cell(*goal_xy)
        for pad in (pad_m, 3.0 * pad_m):
            k = int(math.ceil(pad / self.voxel))
            i0, i1 = max(0, min(s[0], g[0]) - k), min(self.hard.shape[0], max(s[0], g[0]) + k + 1)
            j0, j1 = max(0, min(s[1], g[1]) - k), min(self.hard.shape[1], max(s[1], g[1]) + k + 1)
            if not (i0 <= s[0] < i1 and j0 <= s[1] < j1 and i0 <= g[0] < i1 and j0 <= g[1] < j1):
                return None
            cells = astar(self.hard[i0:i1, j0:j1], self.cost[i0:i1, j0:j1],
                          (s[0] - i0, s[1] - j0), (g[0] - i0, g[1] - j0))
            if cells is not None:
                return [self.xy(i + i0, j + j0) for i, j in cells]
        return None


class OccAstarTeacher:
    """占据 A* + 纯追踪教师；collect_3dgs 的 policy 接口 (rgb, depth, state) -> (action, infer_ms)。

    state 需有 x / y / z / yaw（弧度），CfBlackbox 与 CfSitlDynamics 都满足。
    """

    # teacher 名沿用 Gazebo 的 plan_action（特权规划教师；preprocess 只认 plan_action/expert_action/
    # synthetic 为教师标签）；实现差异记在 planner="astar_occ" 和 teacher_version 里。
    name = "plan_action"
    planner_name = "astar_occ"
    version = TEACHER_VERSION

    def __init__(self, planner, waypoint, lookahead=LOOKAHEAD_M, tol_deg=YAW_TOL_DEG,
                 stop_radius=STOP_RADIUS_M, forward_check_m=FORWARD_CHECK_M):
        self.planner = planner
        self.wp = (float(waypoint["x"]), float(waypoint["y"]))
        self.lookahead, self.tol = float(lookahead), float(tol_deg)
        self.stop_radius, self.forward_check = float(stop_radius), float(forward_check_m)
        self.prev = None          # collect 每集会重置（滞回接口），这里不用
        self.last_path = None
        self.n_plan_fail = 0

    def set_waypoint(self, waypoint):
        self.wp = (float(waypoint["x"]), float(waypoint["y"]))

    def __call__(self, rgb, depth, state):
        x, y, z = float(state.x), float(state.y), float(getattr(state, "z", 2.0))
        yaw_deg = math.degrees(float(state.yaw))
        if math.hypot(x - self.wp[0], y - self.wp[1]) <= self.stop_radius:
            return SAFE_NEUTRAL, None
        path = self.planner.plan((x, y), self.wp)
        self.last_path = path
        if path is None:                      # 被围死 / waypoint 不可达：fail closed
            self.n_plan_fail += 1
            return SAFE_NEUTRAL, None
        tx, ty = self.planner.aim_point(path, x, y, self.lookahead)
        err = _wrap(math.degrees(math.atan2(ty - y, tx - x)) - yaw_deg)
        if err > self.tol:
            return YAW_LEFT, None
        if err < -self.tol:
            return YAW_RIGHT, None
        # 航向对准也不能盲进：机体有惯性，前向 FORWARD_CHECK_M 内撞硬栅格就先按 err 方向转。
        yaw = math.radians(yaw_deg)
        hit, _ = self.planner.occ.sweep(
            (x, y, z), (x + self.forward_check * math.cos(yaw), y + self.forward_check * math.sin(yaw), z))
        if hit:
            return YAW_LEFT if err >= 0 else YAW_RIGHT, None
        return FORWARD_SLOW, None


def terrain_under_path(terrain_occ, pts):
    """沿 pts（导航系 (x, y) 序列）查地形层原始占据：正下方那一列最高占据格的格顶高度（nav 米）；没有返回 None。

    terrain_occ 来自 collision.terrain_from_scene（[TOF_TERRAIN_Z_MIN, z_slice 下沿) 那一层）。只查机体正下方
    一列（ToF 光斑几厘米），不含机体半径。
    """
    if terrain_occ is None or terrain_occ.raw.size == 0 or not pts:
        return None
    nx, ny, nz = terrain_occ.shape
    cells = terrain_occ._cell([(float(x), float(y), terrain_occ.z_range[0]) for x, y in pts])
    ok = (cells[:, 0] >= 0) & (cells[:, 0] < nx) & (cells[:, 1] >= 0) & (cells[:, 1] < ny)
    if not ok.any():
        return None
    cols = terrain_occ.raw[cells[ok, 0], cells[ok, 1], :]              # [M, nz]
    if not cols.any():
        return None
    k_top = int(np.max(np.nonzero(cols.any(axis=0))[0]))               # 最高的占据层
    return round(float(terrain_occ.origin[2] + (k_top + 1) * terrain_occ.voxel), 3)


def segment_points(p, q, step):
    """直线段 p->q 按 step 米取点（含两端），给没有 A* 路径的策略（teacher/forward）查地形用。"""
    p, q = np.asarray(p[:2], dtype=np.float64), np.asarray(q[:2], dtype=np.float64)
    n = max(2, int(math.ceil(np.linalg.norm(q - p) / step)) + 1)
    return [tuple(map(float, p + t * (q - p))) for t in np.linspace(0.0, 1.0, n)]


def agl_flags(terrain_occ, pts, route_name=None):
    """AUDIT_SIM #2 的打标：tof_terrain_m（沿 pts 的地形最高格顶，None = 飞行层以下没东西）；
    agl_model_unverified = 有地形 或 路线名含 over_（场地文档里「飞越桌面/长椅」的命名约定）；reason 按这个顺序列出。
    只是标记：ToF 仍按平地模型算，真机在这些路线上的高度/俯仰会与仿真不同。"""
    terrain_m = terrain_under_path(terrain_occ, pts)
    reason = []
    if terrain_m is not None:
        reason.append("tof_terrain")
    if route_name and "over_" in str(route_name):
        reason.append("route_name")
    return {"tof_terrain_m": terrain_m, "agl_model_unverified": bool(reason), "agl_model_unverified_reason": reason}


def route_check(planner, spawn, waypoint, body_radius_m=None, margin_m=0.3, terrain_occ=None, route_name=None):
    """离线可达性：A* 路径是否存在、航程、转向总量、沿路最小净空、步数预算，以及 ToF 地形打标（agl_flags）。

    净空 = occ.clearance（到最近原始占据格中心的水平距离）；要求 >= 机体半径 + margin_m 才算 clear_ok。
    转向量在 2×LOOKAHEAD_M 抽稀后的折线上累计（8 邻接栅格路径的 45° 锯齿大部分被抹掉，仍偏保守）。
    terrain_occ（collision.terrain_from_scene）给了就沿 A* 路径查飞行层以下的桌面/长椅；没给只按路线名判。
    """
    occ = planner.occ
    body = float(body_radius_m if body_radius_m is not None else occ.body_radius)
    sx, sy, syaw = float(spawn["x"]), float(spawn["y"]), float(spawn.get("yaw_deg", 0.0))
    z = float(spawn.get("z", 2.0))
    out = {"feasible": False, "spawn_free": not planner.blocked_at(sx, sy),
           "waypoint_free": not planner.blocked_at(float(waypoint["x"]), float(waypoint["y"]))}
    path = planner.plan((sx, sy), (float(waypoint["x"]), float(waypoint["y"])))
    if path is None:
        out["reason"] = "A* 无路径"
        out.update(agl_flags(terrain_occ, segment_points((sx, sy), (waypoint["x"], waypoint["y"]), planner.voxel),
                             route_name))
        return out
    out.update(agl_flags(terrain_occ, path, route_name))
    # 抽稀：每 2×LOOKAHEAD_M 取一点，首尾保留（按 LOOKAHEAD_M 抽会把 45° 锯齿计成 200°+ 的转向量）
    thin, acc = [path[0]], 0.0
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        acc += math.hypot(bx - ax, by - ay)
        if acc >= 2.0 * LOOKAHEAD_M:
            thin.append((bx, by))
            acc = 0.0
    if thin[-1] != path[-1]:
        thin.append(path[-1])
    headings = [math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) for a, b in zip(thin, thin[1:])]
    turn = abs(_wrap(headings[0] - syaw)) if headings else 0.0
    turn += sum(abs(_wrap(h1 - h0)) for h0, h1 in zip(headings, headings[1:]))
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
    clear = [occ.clearance(px, py, z) for px, py in path]
    min_clear = float(min(clear)) if clear else math.inf
    steps = int(math.ceil((length / STEP_FORWARD_M + turn / STEP_YAW_DEG + STEP_OVERHEAD) * STEP_MARGIN))
    out.update({"feasible": bool(out["spawn_free"] and out["waypoint_free"]),
                "path_m": round(length, 2), "turn_deg": round(turn, 1),
                "initial_heading_err_deg": round(_wrap(headings[0] - syaw), 1) if headings else 0.0,
                "min_clearance_m": round(min_clear, 3),
                "clear_ok": bool(min_clear >= body + margin_m),
                "steps_estimate": steps, "n_path_pts": len(path), "path": path})
    return out


def _selftest():
    import importlib

    Occupancy = importlib.import_module("sim.3dgs.collision").Occupancy
    # 一堵 x=5、y∈[-3,1] 的墙，起点 (2,0) 朝 +x，目标 (8,0)：必须绕墙的 +y 端。
    ys = np.linspace(-3.0, 1.0, 21)
    means = np.array([(5.0, y, 2.0) for y in ys])
    occ = Occupancy(means, np.tile([0.01, 0.2, 0.3], (len(ys), 1)), None, np.full(len(ys), 0.95),
                    z_range=(1.65, 2.35), voxel_m=0.1, body_radius_m=0.102, body_half_height_m=0.028, pad_m=4.0)
    pl = OccPlanner(occ)
    r = route_check(pl, {"x": 2.0, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}, {"x": 8.0, "y": 0.0})
    assert r["feasible"] and r["clear_ok"], r
    assert max(p[1] for p in r["path"]) > 1.2, "路径应从墙的 +y 端绕过"
    assert r["turn_deg"] > 40 and r["path_m"] > 6.5, r
    assert pl.plan((2.0, 0.0), (5.0, 0.0)) is None, "目标在墙里应无路径"

    class S:  # 最小状态
        def __init__(self, x, y, yaw_deg):
            self.x, self.y, self.z, self.yaw = x, y, 2.0, math.radians(yaw_deg)

    t = OccAstarTeacher(pl, {"x": 8.0, "y": 0.0})
    assert t(None, None, S(2.0, 0.0, 0.0)) == (YAW_LEFT, None)       # 路径先往 +y 绕，左转
    assert t(None, None, S(2.0, 2.0, 0.0)) == (FORWARD_SLOW, None)    # 墙的 +y 端外侧直飞
    assert t(None, None, S(7.8, 0.1, 0.0)) == (SAFE_NEUTRAL, None)    # goal-stop
    assert t(None, None, S(4.6, 0.0, 0.0))[0] in (YAW_LEFT, YAW_RIGHT)  # 正对墙 0.4 m：先转
    assert t(None, None, S(5.0, 0.0, 0.0)) == (SAFE_NEUTRAL, None)     # 机体在墙里：四邻全堵 fail closed
    print("planner selftest PASS", {k: v for k, v in r.items() if k != "path"})


def main(argv=None):
    """离线检查 scene.routes（或顶层 spawn/waypoint）：A* 可达、航程、转向、最小净空、步数预算。

    python -m sim.3dgs.planner --scene <scene.json> [--route 1 3] [--json]；全部 feasible 且 clear_ok 退出 0。
    """
    import argparse
    import importlib
    import json

    p = argparse.ArgumentParser(description="3DGS 路线离线可达性检查（占据代理 A*）")
    p.add_argument("--scene", required=True)
    p.add_argument("--route", type=int, nargs="*", default=None,
                   help="routes 索引（默认全部；scene 没有 routes 时查顶层 spawn/waypoint）")
    p.add_argument("--json", action="store_true", help="单行 JSON 输出（给脚本解析）")
    args = p.parse_args(argv)
    scene_mod = importlib.import_module("sim.3dgs.scene")
    collision = importlib.import_module("sim.3dgs.collision")
    scene = scene_mod.load_scene(args.scene)
    # 高斯源只读一次 PLY（飞行层占据 + 飞行层以下的 ToF 地形层共用）；点云/网格源由 from_scene 自己读两遍。
    gaussians = (scene_mod.read_gaussian_ply(scene["ply_path"])
                 if scene.get("collision", {}).get("source", "gaussians") == "gaussians" else None)
    planner = OccPlanner(collision.from_scene(scene, gaussians=gaussians))
    terrain = collision.terrain_from_scene(scene, gaussians=gaussians)
    idxs = args.route if args.route else list(range(max(1, len(scene.get("routes") or []))))
    rows = []
    for i in idxs:
        sc = scene_mod.apply_route(dict(scene), i)
        r = route_check(planner, sc["spawn"], sc["waypoint"], terrain_occ=terrain,
                        route_name=(sc.get("route") or {}).get("name"))
        r.pop("path", None)
        rows.append({"route": i, "name": (sc.get("route") or {}).get("name"), "spawn": sc["spawn"],
                     "waypoint": sc["waypoint"], "max_steps": sc.get("max_steps"), **r})
    print(json.dumps(rows, ensure_ascii=False, indent=None if args.json else 1))
    return 0 if all(r["feasible"] and r.get("clear_ok") for r in rows) else 1


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main() if len(sys.argv) > 1 else _selftest())
