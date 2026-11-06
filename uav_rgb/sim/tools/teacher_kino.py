#!/usr/bin/env python3
"""teacher_kino：E1 MINI 的动力学感知教师（训练时特权，学生不可用）。

接口与 teacher.plan_action / plan_action_global 对齐，多了可选的运动状态：
  plan_action(x, y, yaw_deg, pillars, waypoint=None, stop_at_waypoint=False,
              stop_radius=0.5, vx=0.0, vy=0.0, yaw_rate=0.0, prev_action=None) -> 0..3
  vx/vy 是世界系水平速度（m/s），yaw_rate 是偏航角速率（rad/s，左正）；不传就按静止。
  动作编号同 teacher.py：0=SAFE_NEUTRAL 1=FORWARD_SLOW 2=YAW_LEFT 3=YAW_RIGHT。

和旧教师的区别：旧教师在 2D 网格上找几何路径再纯追踪，假设每步都能停住、能转到
任意航向（X3 几何常量 0.245/0.36/0.5）。本教师直接在四个动作张成的运动原语空间里做
kinodynamic A*（Fast-Planner kinodynamic_astar.cpp 的结构：离散输入 × 固定时长 →
状态转移 → 栅格剪枝 → 时间代价 + 启发），每条原语按 E1 SITL 实测的一阶响应积分一个
hold 窗口，碰撞按 E1 包络沿原语逐点查；每个节点还要求「从这里悬停能停住且不撞」
（Assistive-Racing risk_monitor::SafetyCheck 的紧急停轨迹检查），不进入必撞状态。
启发用目标根 Dijkstra 距离场（ego-planner 用 A* 几何路径引导优化，同一思路）。

纯 stdlib。参数读 sim/dynamics/e1_mini.json（collision_envelope / executor /
camera_latency）；响应时间常数来自 E1 SITL 实测（见配置区）。
自检：python3 sim/tools/teacher_kino.py
"""
import heapq
import json
import math
import os

from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, ACTIONS

TEACHER_VERSION = "uavrgb.teacher_kino.v1"

# ---------------------------------------------------------------- 配置区
SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.environ.get("UAV_RGB_E1_PARAMS",
                             os.path.join(SIM_DIR, "dynamics", "e1_mini.json"))
STEP_S = 1.0        # 决策周期 = hold 窗口（BRIEF §4）；collect --hold 改了这里要跟着改
TAU_V_UP = 0.313    # FORWARD 速度一阶响应：E1 SITL 实测 0.72 s 到 90% → τ = 0.72/ln10
TAU_V_DOWN = 0.393  # SAFE 刹停：实测滑行 0.118 m ÷ 0.3 m/s；对应 0.64 s 时 v≈0.06 m/s
TAU_YAW = 0.091     # 偏航率一阶响应：实测 0.21 s 到 90%（0.598 rad/s）
# shortcut: 三个 τ 是 SITL 实测折算的常数，e1_mini.json 还没有这几项；SITL-core 把
# 执行器响应写进 JSON 后改成从文件读。
SIM_DT = 0.05       # 原语积分步长；0.3 m/s 下每步 1.5 cm，检查误差并入 CLEAR_HARD
STOP_CHECK_S = 1.0  # 紧急停检查时长：τ=0.393 时 1 s 后余速 0.3·e^-2.5≈0.02 m/s
CLEAR_HARD = 0.08   # 包络外硬余量：位姿噪声 + 栅格最近格误差（0.05 m 格 → 0.035 m）
CLEAR_SOFT = 0.50   # 软惩罚带：距柱面不足此值加代价（EGO calcDistanceCostRebound 的 dist0）；
                    # 1.2 m 缝的中线离柱面 0.6 m，取 0.5 让中线零代价、贴边飞有代价
W_SOFT = 10.0       # 软余量权重（代价单位=秒）：距柱面 0.3 m 每步 +0.4 s、0.2 m 每步 +0.9 s；
                    # 4.0 时实测门柱场景会拖到 0.3 m 才转（一窗 1 s 的时间代价压过了它）
W_SWITCH = 0.15     # 换动作惩罚（秒）：抑制左右摆（EGO-v2 traj_server 限偏航加速度同一目的）
GRID = 0.05         # 距离栅格分辨率：柱 0.4 m、包络 0.1 m，再细纯 Python 建场太慢
PRUNE_XY = 0.10     # 搜索剪枝：位置格（Fast-Planner 用地图分辨率剪枝）
PRUNE_YAW_DEG = 10.0  # 搜索剪枝：航向格；一步偏航约 34°，10° 不会把左右转剪成同一节点
HORIZON_STEPS = 8   # 搜索视野：8 步 ≈ 8 s ≈ 2.4 m，覆盖一次车道变换（1.3 m 纵向缝）
MAX_EXPAND = 1500   # 展开上限：到上限取启发最小的节点（Fast-Planner run-out 的处理）
GOAL_AHEAD_M = 2.0  # 默认目标 = 最远柱 x + 2.0 m，y=0（与 teacher.plan_action_global 同）
GOAL_RADIUS = 0.5   # 默认目标的到达半径；l1 任务的 stop_radius 由调用方传
FIELD_CACHE_MAX = 4  # 距离场缓存条数：collect 一次只有一个场景，compare 交替用几个


def _load_params(path):
    """只取本教师要的几项；缺项直接报错，不静默用默认值。"""
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    try:
        env = p["collision_envelope"]["value"]
        ex = p["executor"]["value"]
        return {
            "radius": float(env["radius_from_center"]),     # 0.102 m（装保护罩）
            "v_fwd": float(ex["forward_speed_mps"]),        # 0.3 m/s
            "yaw_rate": float(ex["yaw_rate_rps"]),          # 0.6 rad/s
            "latency": float(p["camera_latency"]["value"]),  # 0.10 s 曝光→动作发出
        }
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path} 缺少 teacher_kino 需要的字段: {e}") from e


PARAMS = _load_params(PARAMS_PATH)
ENVELOPE_R = PARAMS["radius"]
V_FWD = PARAMS["v_fwd"]
YAW_RATE = PARAMS["yaw_rate"]
# 窗口前 LATENCY_S 秒上一条指令已过期（observation 窗口语义：until_s=拍图+hold），
# 执行器回悬停；新指令生效后只剩 STEP_S-LATENCY_S。
LATENCY_S = PARAMS["latency"]
CLEAR_PLAN = ENVELOPE_R + CLEAR_HARD


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _boxes(pillars):
    out = []
    for p in pillars:
        sx, sy = (float(v) for v in p.get("size", [0.4, 0.4])[:2])
        out.append((float(p["x"]), float(p["y"]), sx / 2.0, sy / 2.0))
    return tuple(out)


def _box_dist(x, y, boxes):
    """点到最近柱方框表面的距离（框内为负）。"""
    best = math.inf
    for bx, by, hx, hy in boxes:
        dx, dy = abs(x - bx) - hx, abs(y - by) - hy
        if dx > 0.0 or dy > 0.0:
            d = math.hypot(max(dx, 0.0), max(dy, 0.0))
        else:
            d = max(dx, dy)
        if d < best:
            best = d
    return best


class _Field:
    """柱阵距离栅格 + 目标根 Dijkstra 距离场。

    距离栅格给碰撞检查（最近格查表，误差 ≤ GRID·√2/2 并入 CLEAR_HARD）；
    Dijkstra 场给 A* 启发：几何最短路长度 / 巡航速度是到达时间的下界，比欧氏距离
    少走很多死胡同（ego-planner 也是先用 A* 几何路径给优化器指方向）。
    """

    def __init__(self, boxes, goal, clear):
        self.boxes, self.goal = boxes, goal
        xs = [goal[0]] + [b[0] for b in boxes]
        ys = [abs(goal[1])] + [abs(b[1]) + b[3] for b in boxes]
        self.x0 = min(-1.5, min(xs)) - 1.5
        x1 = max(xs) + 1.5
        y1 = max(3.0, max(ys)) + 1.0
        self.y0 = -y1
        self.nx = int(math.ceil((x1 - self.x0) / GRID)) + 1
        self.ny = int(math.ceil((y1 - self.y0) / GRID)) + 1
        self.dist = [_box_dist(self.x0 + i * GRID, self.y0 + j * GRID, boxes)
                     for j in range(self.ny) for i in range(self.nx)]
        self.geo = self._dijkstra(clear)

    def _cell(self, x, y):
        i = int(round((x - self.x0) / GRID))
        j = int(round((y - self.y0) / GRID))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def clearance(self, x, y):
        c = self._cell(x, y)
        # 栅格外没有柱子（柱子都在格内），按精确距离算，不会把外面判堵。
        return self.dist[c[1] * self.nx + c[0]] if c else _box_dist(x, y, self.boxes)

    def _dijkstra(self, clear):
        nx, ny = self.nx, self.ny
        free = [d >= clear for d in self.dist]
        g = self._cell(*self.goal)
        if g is None:
            raise ValueError(f"目标 {self.goal} 落在距离栅格外")
        gi = g[1] * nx + g[0]
        if not free[gi]:
            # 目标点贴柱：挪到最近的自由格（TGK krrtplanner.cpp:126-149 也是给终点找替代）。
            cand = [(abs(k % nx - g[0]) + abs(k // nx - g[1]), k)
                    for k in range(len(free)) if free[k]]
            if not cand:
                raise ValueError("距离栅格没有自由格")
            gi = min(cand)[1]
        geo = [math.inf] * (nx * ny)
        geo[gi] = 0.0
        heap = [(0.0, gi)]
        moves = ((1, 0, GRID), (-1, 0, GRID), (0, 1, GRID), (0, -1, GRID),
                 (1, 1, GRID * 1.41421356), (1, -1, GRID * 1.41421356),
                 (-1, 1, GRID * 1.41421356), (-1, -1, GRID * 1.41421356))
        while heap:
            d, k = heapq.heappop(heap)
            if d > geo[k]:
                continue
            i, j = k % nx, k // nx
            for di, dj, w in moves:
                ni, nj = i + di, j + dj
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                nk = nj * nx + ni
                if not free[nk]:
                    continue
                # 禁止斜穿角点：两个正交邻格有一个堵就不能斜走。
                if di and dj and not (free[j * nx + ni] and free[nj * nx + i]):
                    continue
                nd = d + w
                if nd < geo[nk]:
                    geo[nk] = nd
                    heapq.heappush(heap, (nd, nk))
        return geo

    def geodesic(self, x, y):
        """(到目标的几何最短路长度, 下降方向单位向量)。不可达/格外退回欧氏 + 罚项。"""
        c = self._cell(x, y)
        nx = self.nx
        if c is not None:
            k = c[1] * nx + c[0]
            d = self.geo[k]
            if math.isfinite(d):
                best, bi, bj = d, 0, 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = c[0] + di, c[1] + dj
                        if 0 <= ni < nx and 0 <= nj < self.ny:
                            v = self.geo[nj * nx + ni]
                            if v < best:
                                best, bi, bj = v, di, dj
                if bi or bj:
                    n = math.hypot(bi, bj)
                    return d, bi / n, bj / n
                gx, gy = self.goal[0] - x, self.goal[1] - y
                n = math.hypot(gx, gy) or 1.0
                return d, gx / n, gy / n
        gx, gy = self.goal[0] - x, self.goal[1] - y
        n = math.hypot(gx, gy) or 1.0
        # 不可达格：欧氏距离再加 5 m，让搜索明显偏向可达区域而不是直接放弃。
        return n + 5.0, gx / n, gy / n


_FIELD_CACHE = {}


def _field(boxes, goal, clear):
    key = (boxes, goal, clear)
    f = _FIELD_CACHE.get(key)
    if f is None:
        if len(_FIELD_CACHE) >= FIELD_CACHE_MAX:
            _FIELD_CACHE.clear()
        f = _FIELD_CACHE[key] = _Field(boxes, goal, clear)
    return f


def advance(state, action, duration, field=None, clear=0.0, dt=SIM_DT):
    """按一阶响应积分 duration 秒；返回 (末状态, 途中最小净空, 是否撞)。

    state=(x, y, yaw, vx, vy, yaw_rate)。速度在世界系积分：偏航时动量不随机头转
    （collect.py CfStab 同一约定），FORWARD 的目标速度沿当前机头，所以边转边加速
    会画弧。field=None 时不查碰撞。
    """
    x, y, yaw, vx, vy, w = state
    v_cmd = V_FWD if action == FORWARD_SLOW else 0.0
    w_cmd = YAW_RATE if action == YAW_LEFT else -YAW_RATE if action == YAW_RIGHT else 0.0
    tau_v = TAU_V_UP if v_cmd > 0.0 else TAU_V_DOWN
    min_clear = math.inf
    t = 0.0
    while t < duration - 1e-9:
        h = min(dt, duration - t)
        kw = 1.0 - math.exp(-h / TAU_YAW)
        kv = 1.0 - math.exp(-h / tau_v)
        w_new = w + (w_cmd - w) * kw
        yaw += 0.5 * (w + w_new) * h
        w = w_new
        tx, ty = v_cmd * math.cos(yaw), v_cmd * math.sin(yaw)
        vx_new, vy_new = vx + (tx - vx) * kv, vy + (ty - vy) * kv
        x += 0.5 * (vx + vx_new) * h
        y += 0.5 * (vy + vy_new) * h
        vx, vy = vx_new, vy_new
        t += h
        if field is not None:
            c = field.clearance(x, y)
            if c < min_clear:
                min_clear = c
            if c < clear:
                return (x, y, _wrap(yaw), vx, vy, w), min_clear, True
    return (x, y, _wrap(yaw), vx, vy, w), min_clear, False


def primitive(state, action, field=None, clear=0.0):
    """一个决策步的运动原语：前 LATENCY_S 悬停（上一条已过期），再执行 action。"""
    s1, c1, hit = advance(state, SAFE_NEUTRAL, LATENCY_S, field, clear)
    if hit:
        return s1, c1, True
    s2, c2, hit = advance(s1, action, STEP_S - LATENCY_S, field, clear)
    return s2, min(c1, c2), hit


# 稳态巡航时一窗的位移（m）与一窗的偏航量（rad），给启发函数用；由上面的模型算出，
# 不手抄：0.3 m/s 稳态 0.27 m，静止起转 0.48 rad（27.7°）。
FWD_PER_STEP = primitive((0.0, 0.0, 0.0, V_FWD, 0.0, 0.0), FORWARD_SLOW)[0][0]
YAW_PER_STEP = abs(primitive((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), YAW_LEFT)[0][2])


def _heuristic(state, field, goal_r):
    """到达时间下界（秒）：几何最短路 ÷ 每窗巡航位移 + 航向差 ÷ 每窗偏航量，再乘窗长。

    用一窗实际位移/转角而不是 0.3 m/s、0.6 rad/s：窗口前 0.1 s 悬停、响应有 τ，
    一窗只走 0.27 m、只转 28°。YAW 原语不前进，转向和平移不能重叠，直接相加仍是下界。
    """
    d, gx, gy = field.geodesic(state[0], state[1])
    h = d / FWD_PER_STEP
    if d > goal_r:
        h += abs(_wrap(math.atan2(gy, gx) - state[2])) / YAW_PER_STEP
    return h * STEP_S


def _search(start, field, goal, goal_r, stop_mode, clear_first, prev_action):
    """运动原语 kinodynamic A*；返回动作序列（可能为空=已在目标）或 None（起点必撞）。"""
    # 节点: [state, action_from_parent, parent_idx, g, depth, stop_state]
    nodes = [[start, prev_action, -1, 0.0, 0, None]]
    counter = 0
    heap = [(_heuristic(start, field, goal_r), counter, 0)]
    seen = {}
    best = None   # (h, idx) 预算耗尽时的退路
    expanded = 0

    def at_goal(idx):
        s = nodes[idx][5] if stop_mode else nodes[idx][0]
        return s is not None and math.hypot(s[0] - goal[0], s[1] - goal[1]) <= goal_r

    def path(idx):
        acts = []
        while nodes[idx][2] >= 0:
            acts.append(nodes[idx][1])
            idx = nodes[idx][2]
        return acts[::-1]

    while heap and expanded < MAX_EXPAND:
        f, _, idx = heapq.heappop(heap)
        state, act, _, g, depth, _ = nodes[idx]
        if idx > 0 and at_goal(idx):
            return path(idx)
        if depth >= HORIZON_STEPS:
            return path(idx)
        expanded += 1
        clear = clear_first if depth == 0 else CLEAR_PLAN
        for a in (FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, SAFE_NEUTRAL):
            end, min_clear, hit = primitive(state, a, field, clear)
            if hit:
                continue
            # 必撞状态剪枝：从末状态悬停 STOP_CHECK_S 还得不撞（Assistive-Racing SafetyCheck）。
            stop, c_stop, hit = advance(end, SAFE_NEUTRAL, STOP_CHECK_S, field, clear)
            if hit:
                continue
            key = (int(round(end[0] / PRUNE_XY)), int(round(end[1] / PRUNE_XY)),
                   int(round(math.degrees(end[2]) / PRUNE_YAW_DEG)))
            soft = max(0.0, CLEAR_SOFT - min(min_clear, c_stop))
            g2 = g + STEP_S + W_SOFT * soft * soft + (W_SWITCH if a != act else 0.0)
            if seen.get(key, math.inf) <= g2:
                continue
            seen[key] = g2
            h = _heuristic(end, field, goal_r)
            nodes.append([end, a, idx, g2, depth + 1, stop])
            counter += 1
            heapq.heappush(heap, (g2 + h, counter, len(nodes) - 1))
            if best is None or h < best[0]:
                best = (h, len(nodes) - 1)
    if best is None:
        return None
    return path(best[1])


def plan_action(x, y, yaw_deg, pillars, waypoint=None, stop_at_waypoint=False,
                stop_radius=0.5, vx=0.0, vy=0.0, yaw_rate=0.0, prev_action=None):
    """返回 0..3。waypoint/stop_at_waypoint/stop_radius 语义同 teacher.plan_action_global。

    stop_at_waypoint=True：距 waypoint ≤ stop_radius 直接 SAFE（与旧标签口径一致）；
    否则目标判定用「悬停后停住的位置」是否进半径，所以会提前一步刹车让滑行停在圈里。
    起点已漂进规划余量时第一步只查真实包络（teacher.py 脱困回退同一做法）；
    起点连悬停都停不住（必撞）→ SAFE，fail closed。
    """
    boxes = _boxes(pillars)
    if waypoint is not None:
        goal = (float(waypoint["x"]), float(waypoint["y"]))
        goal_r = float(stop_radius) if stop_at_waypoint else GOAL_RADIUS
        if stop_at_waypoint and math.hypot(x - goal[0], y - goal[1]) <= goal_r:
            return SAFE_NEUTRAL
    elif boxes:
        goal, goal_r = (max(b[0] for b in boxes) + GOAL_AHEAD_M, 0.0), GOAL_RADIUS
    else:
        goal, goal_r = (x + 5.0, 0.0), GOAL_RADIUS
    field = _field(boxes, goal, CLEAR_PLAN)
    start = (float(x), float(y), math.radians(yaw_deg), float(vx), float(vy), float(yaw_rate))
    # 已经漂进余量：第一步放宽到真实包络 + 1 cm，先离开再要求 full clearance。
    clear_first = CLEAR_PLAN if field.clearance(x, y) >= CLEAR_PLAN else ENVELOPE_R + 0.01
    acts = _search(start, field, goal, goal_r, stop_at_waypoint, clear_first, prev_action)
    if acts is None:
        return SAFE_NEUTRAL
    if not acts:
        # 根节点已满足目标：非停靠模式继续前进（旧教师到终点也不停）；前方不净空则 SAFE。
        if not stop_at_waypoint:
            _, _, hit = primitive(start, FORWARD_SLOW, field, clear_first)
            return SAFE_NEUTRAL if hit else FORWARD_SLOW
        return SAFE_NEUTRAL
    return acts[0]


def plan_sequence(x, y, yaw_deg, pillars, **kw):
    """调试/画图用：返回整条规划的动作序列（同 plan_action 的参数）。"""
    boxes = _boxes(pillars)
    waypoint = kw.get("waypoint")
    if waypoint is not None:
        goal = (float(waypoint["x"]), float(waypoint["y"]))
        goal_r = float(kw.get("stop_radius", 0.5)) if kw.get("stop_at_waypoint") else GOAL_RADIUS
    elif boxes:
        goal, goal_r = (max(b[0] for b in boxes) + GOAL_AHEAD_M, 0.0), GOAL_RADIUS
    else:
        goal, goal_r = (x + 5.0, 0.0), GOAL_RADIUS
    field = _field(boxes, goal, CLEAR_PLAN)
    start = (float(x), float(y), math.radians(yaw_deg), float(kw.get("vx", 0.0)),
             float(kw.get("vy", 0.0)), float(kw.get("yaw_rate", 0.0)))
    clear_first = CLEAR_PLAN if field.clearance(x, y) >= CLEAR_PLAN else ENVELOPE_R + 0.01
    return _search(start, field, goal, goal_r, bool(kw.get("stop_at_waypoint")),
                   clear_first, kw.get("prev_action")) or []


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene_gen
    ok = True

    def check(name, cond, detail=""):
        global ok
        ok = ok and bool(cond)
        print(f"  {'ok' if cond else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")

    check("e1_mini.json 包络半径 0.102", abs(ENVELOPE_R - 0.102) < 1e-9, f"{ENVELOPE_R}")
    rest = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    s, _, _ = primitive(rest, FORWARD_SLOW)
    check("FORWARD 一窗位移 0.15–0.30 m", 0.15 <= s[0] <= 0.30, f"dx={s[0]:.3f} v={s[3]:.3f}")
    s, _, _ = advance((0.0, 0.0, 0.0, V_FWD, 0.0, 0.0), SAFE_NEUTRAL, 3.0)
    check("SAFE 刹停滑行 ≈0.118 m", abs(s[0] - 0.118) < 0.012, f"dx={s[0]:.3f}")
    s, _, _ = primitive(rest, YAW_LEFT)
    check("YAW_LEFT 一窗转 25–36°", 25.0 <= math.degrees(s[2]) <= 36.0, f"{math.degrees(s[2]):.1f}°")
    check("无柱子航向对准 -> FORWARD", plan_action(0.0, 0.0, 0.0, []) == FORWARD_SLOW)
    check("无柱子航向偏左 40° -> YAW_RIGHT", plan_action(0.0, 0.0, 40.0, []) == YAW_RIGHT)
    # 避障/停靠/必撞等行为检查在 test_teacher_kino.py（闭环回放），这里只做冒烟。
    t0 = time.time()
    pillars, _, _ = scene_gen.sample_pillars(180)
    got = plan_action(scene_gen.SPAWN_X, 0.0, 0.0, pillars)
    dt = time.time() - t0
    check("seed 180 出生点首步非 SAFE 且 <1.5 s（含建场）", got != SAFE_NEUTRAL and dt < 1.5,
          f"{ACTIONS[got]} {dt:.2f}s")
    t0 = time.time()
    plan_action(0.5, 0.3, 10.0, pillars, vx=0.25, vy=0.05)
    check("seed 180 热缓存单步 <0.5 s", time.time() - t0 < 0.5, f"{time.time() - t0:.3f}s")
    print("teacher_kino: " + ("全部通过" if ok else "有失败"))
    raise SystemExit(0 if ok else 1)
