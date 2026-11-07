#!/usr/bin/env python3
"""teacher_kino：E1 MINI 的动力学感知教师（训练时特权，学生不可用）。

接口与 teacher.plan_action / plan_action_global 对齐，多了可选的运动状态：
  plan_action(x, y, yaw_deg, pillars, waypoint=None, stop_at_waypoint=False,
              stop_radius=0.5, vx=0.0, vy=0.0, yaw_rate=0.0, prev_action=None) -> 0..3
  vx/vy 是世界系水平速度（m/s），yaw_rate 是偏航角速率（rad/s，左正）；不传就按静止。
  动作编号同 teacher.py：0=SAFE_NEUTRAL 1=FORWARD_SLOW 2=YAW_LEFT 3=YAW_RIGHT。
  plan_costs(同参数) -> {动作: cost_to_go}：同一次 A*（v1 的堆、剪枝、首步净空都不变）
  给四个首原语各记下自己子树第一个终点的 f（见 search_costs），
  代价单位是**秒**（每步 step_s + 软余量罚 + 换向罚；视野/预算截断处加启发剩余时间），
  必撞/停不住/无路 = inf。plan_action 就是它的 argmin（并列按 FORWARD/LEFT/RIGHT/SAFE）；
  与 v1 的 20 seed 离线对比（tmp/agent4-teacher-p1/compare_plan_action_v1_v2.py）3198/3200
  一致，仅有的 2 处是四个首原语全 inf 的必撞状态：v1 返回启发最小节点的首原语，这里 SAFE。
  model=Model(step_s, latency_s, drift) 给 3DGS（hold 0.5、无窗口延迟）和信念教师
  （teacher_belief 采样延迟/漂移）用；不传就是 Gazebo 名义值。

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
from typing import NamedTuple

from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, ACTIONS

TEACHER_VERSION = "uavrgb.teacher_kino.v1"
# argmin 并列时的优先序：和 v1 搜索里子节点的入堆顺序一致（同 f 先弹先入的）。
ACTION_ORDER = (FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, SAFE_NEUTRAL)

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
HORIZON_S = HORIZON_STEPS * STEP_S  # 视野按时间定：hold 0.5 的 3DGS 数据用 16 步，覆盖同样 2.4 m
MAX_EXPAND = 1500   # 展开上限：到上限取启发最小的节点（Fast-Planner run-out 的处理）；plan_costs 每个首原语各一份
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


class Model(NamedTuple):
    """一次规划用的执行模型：决策步长、窗口首部的失效延迟、悬停漂移（世界系 m/s）。

    名义值 = Gazebo observation 语义（hold 1.0，前 0.1 s 上一条已过期）。3DGS cf_sitl 是
    until_s = t_obs + hold 的 dispatch 语义：step_s=hold、latency_s=0。teacher_belief 按假设
    采样 latency/drift；drift 加在速度设定点上（悬停时机体随之漂，前飞时叠加侧偏）。
    """
    step_s: float = STEP_S
    latency_s: float = LATENCY_S
    drift_vx: float = 0.0
    drift_vy: float = 0.0

    @property
    def horizon_steps(self):
        return max(1, int(round(HORIZON_S / self.step_s)))


NOMINAL = Model()


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

    构造函数接现成的行优先距离表（teacher_belief 用 numpy 算 3DGS 占据栅格的 EDT 后传进来，
    grid 可不是 GRID）；柱阵走 from_boxes，与 v1 完全同一套格点和距离。
    """

    def __init__(self, dist, x0, y0, nx, ny, goal, clear, grid=GRID, boxes=()):
        self.dist, self.x0, self.y0, self.nx, self.ny = dist, x0, y0, nx, ny
        self.grid, self.goal, self.boxes = grid, goal, boxes
        self.geo = self._dijkstra(clear)

    @staticmethod
    def extent(boxes, goal):
        """柱阵栅格范围 (x0, y0, nx, ny)：两侧各留 1.5 m，y 至少盖到 ±3 走廊再加 1 m。"""
        xs = [goal[0]] + [b[0] for b in boxes]
        ys = [abs(goal[1])] + [abs(b[1]) + b[3] for b in boxes]
        x0 = min(-1.5, min(xs)) - 1.5
        x1 = max(xs) + 1.5
        y1 = max(3.0, max(ys)) + 1.0
        return x0, -y1, int(math.ceil((x1 - x0) / GRID)) + 1, int(math.ceil((2.0 * y1) / GRID)) + 1

    @classmethod
    def from_boxes(cls, boxes, goal, clear):
        x0, y0, nx, ny = cls.extent(boxes, goal)
        dist = [_box_dist(x0 + i * GRID, y0 + j * GRID, boxes)
                for j in range(ny) for i in range(nx)]
        return cls(dist, x0, y0, nx, ny, goal, clear, GRID, boxes)

    def _cell(self, x, y):
        i = int(round((x - self.x0) / self.grid))
        j = int(round((y - self.y0) / self.grid))
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
        grid = self.grid
        moves = ((1, 0, grid), (-1, 0, grid), (0, 1, grid), (0, -1, grid),
                 (1, 1, grid * 1.41421356), (1, -1, grid * 1.41421356),
                 (-1, 1, grid * 1.41421356), (-1, -1, grid * 1.41421356))
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
        f = _FIELD_CACHE[key] = _Field.from_boxes(boxes, goal, clear)
    return f


def advance(state, action, duration, field=None, clear=0.0, dt=SIM_DT, drift=(0.0, 0.0)):
    """按一阶响应积分 duration 秒；返回 (末状态, 途中最小净空, 是否撞)。

    state=(x, y, yaw, vx, vy, yaw_rate)。速度在世界系积分：偏航时动量不随机头转
    （collect.py CfStab 同一约定），FORWARD 的目标速度沿当前机头，所以边转边加速
    会画弧。field=None 时不查碰撞。drift 是速度设定点上的世界系常偏（悬停漂移），
    名义 (0,0) 时算式逐位不变。
    """
    x, y, yaw, vx, vy, w = state
    v_cmd = V_FWD if action == FORWARD_SLOW else 0.0
    w_cmd = YAW_RATE if action == YAW_LEFT else -YAW_RATE if action == YAW_RIGHT else 0.0
    tau_v = TAU_V_UP if v_cmd > 0.0 else TAU_V_DOWN
    dx_drift, dy_drift = drift
    min_clear = math.inf
    t = 0.0
    while t < duration - 1e-9:
        h = min(dt, duration - t)
        kw = 1.0 - math.exp(-h / TAU_YAW)
        kv = 1.0 - math.exp(-h / tau_v)
        w_new = w + (w_cmd - w) * kw
        yaw += 0.5 * (w + w_new) * h
        w = w_new
        tx, ty = v_cmd * math.cos(yaw) + dx_drift, v_cmd * math.sin(yaw) + dy_drift
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


def primitive(state, action, field=None, clear=0.0, model=NOMINAL):
    """一个决策步的运动原语：前 latency_s 悬停（上一条已过期），再执行 action 到窗口末。"""
    drift = (model.drift_vx, model.drift_vy)
    s1, c1, hit = advance(state, SAFE_NEUTRAL, model.latency_s, field, clear, drift=drift)
    if hit:
        return s1, c1, True
    s2, c2, hit = advance(s1, action, model.step_s - model.latency_s, field, clear, drift=drift)
    return s2, min(c1, c2), hit


_PER_STEP = {}


def per_step(model=NOMINAL):
    """(稳态巡航一窗位移 m, 静止起转一窗偏航 rad)，给启发函数用；由上面的模型算出，不手抄。

    名义模型：0.3 m/s 稳态 0.27 m，静止起转 0.48 rad（27.7°）。按 model 缓存。
    """
    v = _PER_STEP.get(model)
    if v is None:
        v = _PER_STEP[model] = (
            primitive((0.0, 0.0, 0.0, V_FWD, 0.0, 0.0), FORWARD_SLOW, model=model)[0][0],
            abs(primitive((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), YAW_LEFT, model=model)[0][2]))
    return v


FWD_PER_STEP, YAW_PER_STEP = per_step()


def _heuristic(state, field, goal_r, model=NOMINAL):
    """到达时间下界（秒）：几何最短路 ÷ 每窗巡航位移 + 航向差 ÷ 每窗偏航量，再乘窗长。

    用一窗实际位移/转角而不是 0.3 m/s、0.6 rad/s：窗口前 0.1 s 悬停、响应有 τ，
    一窗只走 0.27 m、只转 28°。YAW 原语不前进，转向和平移不能重叠，直接相加仍是下界。
    """
    fwd, yaw = per_step(model)
    d, gx, gy = field.geodesic(state[0], state[1])
    h = d / fwd
    if d > goal_r:
        h += abs(_wrap(math.atan2(gy, gx) - state[2])) / yaw
    return h * model.step_s


def search_costs(start, field, goal, goal_r, stop_mode, clear_first, prev_action, model=NOMINAL,
                 max_expand=MAX_EXPAND):
    """运动原语 kinodynamic A*，一次搜索给出四个首原语各自的 cost_to_go。

    返回 {动作: (秒, 动作序列或 None)}；首步必撞/停不住、或子树全部撞死 = (inf, None)。
    搜索结构与 v1 完全一致（同一个堆、同一个 (x,y,yaw) 格剪枝、同样的首步 clear_first），
    只是弹出第一个终点/视野节点后不停：记下它所属首原语的代价，继续弹，直到每个首原语都
    有自己的终点（或子树耗尽 / 总预算到）。
      - 代价 = 该首原语第一个终点弹出时的 f 层级：f = g + h（到达节点 h 是到目标圆心的
        残余，和 v1 一样当作「离圆心更近更好」的细分），再取到此刻为止弹出过的最大 f。
        启发函数不满足一致性（几何最短路方向的航向差一步内能大降），弹出 f 不单调；取
        运行最大值等于把 f 按弹出顺序单调化（pathmax 的同一思路），这样第一个弹出的终点
        一定是 argmin，plan_action 与 v1 逐位一致，而其它首原语的代价仍是各自子树的下界估计。
      - 子树被别的首原语剪掉（同格但 g 更大，即「先悬停/先转再做同样的事」）：代价 = 剪掉它
        的那条路的代价 + 两者 g 差。同格剪枝的含义本来就是「这条路只是晚到」，这样算出的
        正是晚到的代价；若各分支独立搜索，悬停一步的分支会因为启发偏松而看起来更便宜
        （门柱场景实测 SAFE 19.3 s < FORWARD 19.6 s），v1 的 argmin 就复现不出来。
      - 预算耗尽仍没终点的子树：取它已生成节点里最小的 f（v1 的退路取最小 h，这里取 f 才能
        与其它分支的代价同口径）。
    field 只需 clearance(x, y) 与 geodesic(x, y)（teacher_belief 传自己的假设场）。
    """
    horizon = model.horizon_steps
    drift = (model.drift_vx, model.drift_vy)
    # 节点: [state, action_from_parent, parent_idx, g, depth, stop_state, h, root_action]
    nodes = [[start, prev_action, -1, 0.0, 0, None, 0.0, None]]
    heap, seen, counter = [], {}, 0
    # 每个首原语的子树状态：done=终点代价、path、open=堆里的节点数、best_f、dom=[(g 差, 剪它的首原语)]
    sub = {}

    def at_goal(idx):
        s = nodes[idx][5] if stop_mode else nodes[idx][0]
        return s is not None and math.hypot(s[0] - goal[0], s[1] - goal[1]) <= goal_r

    def path(idx):
        acts = []
        while nodes[idx][2] >= 0:
            acts.append(nodes[idx][1])
            idx = nodes[idx][2]
        return acts[::-1]

    def expand(idx):
        nonlocal counter
        state, act, _, g, depth, _, _, root = nodes[idx]
        clear = clear_first if depth == 0 else CLEAR_PLAN
        for a in ACTION_ORDER:
            end, min_clear, hit = primitive(state, a, field, clear, model)
            if hit:
                continue
            # 必撞状态剪枝：从末状态悬停 STOP_CHECK_S 还得不撞（Assistive-Racing SafetyCheck）。
            stop, c_stop, hit = advance(end, SAFE_NEUTRAL, STOP_CHECK_S, field, clear, drift=drift)
            if hit:
                continue
            key = (int(round(end[0] / PRUNE_XY)), int(round(end[1] / PRUNE_XY)),
                   int(round(math.degrees(end[2]) / PRUNE_YAW_DEG)))
            soft = max(0.0, CLEAR_SOFT - min(min_clear, c_stop))
            g2 = g + model.step_s + W_SOFT * soft * soft + (W_SWITCH if a != act else 0.0)
            r = a if root is None else root
            # 首步可行就先登记子树：hold 0.5 时 FORWARD 从静止只走 0.075 m，和 SAFE 落同一格，
            # 后入堆的那个首原语会在这里就被剪掉，只能靠 dom 估代价。
            st = sub.setdefault(r, {"done": None, "path": None, "open": 0, "best_f": math.inf, "dom": []})
            prev = seen.get(key)
            if prev is not None and prev[0] <= g2:
                if prev[1] != r:
                    st["dom"].append((g2 - prev[0], prev[1]))
                continue
            seen[key] = (g2, r)
            h2 = _heuristic(end, field, goal_r, model)
            nodes.append([end, a, idx, g2, depth + 1, stop, h2, r])
            counter += 1
            heapq.heappush(heap, (g2 + h2, counter, len(nodes) - 1))
            st["open"] += 1
            st["best_f"] = min(st["best_f"], g2 + h2)

    expand(0)
    budget = max_expand * max(1, len(sub))
    expanded = 1
    f_level = 0.0
    while heap and expanded < budget and any(
            st["done"] is None and st["open"] > 0 for st in sub.values()):
        f, _, idx = heapq.heappop(heap)
        f_level = max(f_level, f)
        st = sub[nodes[idx][7]]
        st["open"] -= 1
        if st["done"] is not None:        # 该首原语已有终点，剩余节点不再展开
            continue
        if at_goal(idx) or nodes[idx][4] >= horizon:
            st["done"], st["path"] = f_level, path(idx)
        else:
            expanded += 1
            expand(idx)
    # 被剪的子树：代价 = 剪它的分支的代价 + g 差（可能链式，最多迭代 4 轮）。
    for _ in range(len(sub)):
        for r, st in sub.items():
            if st["done"] is None:
                ests = [dg + sub[r2]["done"] for dg, r2 in st["dom"] if sub[r2]["done"] is not None]
                if ests:
                    st["done"] = min(ests)
    out = {}
    for a in ACTION_ORDER:
        st = sub.get(a)
        if st is None:
            out[a] = (math.inf, None)                  # 首步必撞或停不住
        elif st["done"] is not None:
            out[a] = (st["done"], st["path"] or [a])
        elif st["open"] > 0:
            out[a] = (max(st["best_f"], f_level), [a])  # 预算耗尽的退路：子树里最小 f，不低于已弹出层级
        else:
            out[a] = (math.inf, None)                  # 子树全部撞死
    return out


def argmin_action(costs):
    """{动作: cost} -> 代价最小的动作；并列按 ACTION_ORDER；全 inf -> None。"""
    best, best_c = None, math.inf
    for a in ACTION_ORDER:
        c = costs[a][0] if isinstance(costs[a], tuple) else costs[a]
        if c < best_c:
            best, best_c = a, c
    return best


def _setup(x, y, yaw_deg, pillars, waypoint, stop_at_waypoint, stop_radius, vx, vy, yaw_rate):
    """plan_* 共用：目标、距离场、起始状态、首步净空要求。"""
    boxes = _boxes(pillars)
    if waypoint is not None:
        goal = (float(waypoint["x"]), float(waypoint["y"]))
        goal_r = float(stop_radius) if stop_at_waypoint else GOAL_RADIUS
    elif boxes:
        goal, goal_r = (max(b[0] for b in boxes) + GOAL_AHEAD_M, 0.0), GOAL_RADIUS
    else:
        goal, goal_r = (x + 5.0, 0.0), GOAL_RADIUS
    field = _field(boxes, goal, CLEAR_PLAN)
    start = (float(x), float(y), math.radians(yaw_deg), float(vx), float(vy), float(yaw_rate))
    # 已经漂进余量：第一步放宽到真实包络 + 1 cm，先离开再要求 full clearance。
    clear_first = CLEAR_PLAN if field.clearance(x, y) >= CLEAR_PLAN else ENVELOPE_R + 0.01
    return field, goal, goal_r, start, clear_first


def plan_costs(x, y, yaw_deg, pillars, waypoint=None, stop_at_waypoint=False,
               stop_radius=0.5, vx=0.0, vy=0.0, yaw_rate=0.0, prev_action=None, model=NOMINAL):
    """四个首原语的 cost_to_go（秒）：{0: c_SAFE, 1: c_FWD, 2: c_L, 3: c_R}，不可行 = inf。

    停靠模式下已在 stop_radius 内：SAFE 代价 0、其余 inf（与 plan_action 的直接 SAFE 口径一致）。
    """
    if waypoint is not None and stop_at_waypoint and math.hypot(
            x - float(waypoint["x"]), y - float(waypoint["y"])) <= float(stop_radius):
        return {a: (0.0 if a == SAFE_NEUTRAL else math.inf) for a in ACTION_ORDER}
    field, goal, goal_r, start, clear_first = _setup(
        x, y, yaw_deg, pillars, waypoint, stop_at_waypoint, stop_radius, vx, vy, yaw_rate)
    costs = search_costs(start, field, goal, goal_r, stop_at_waypoint, clear_first, prev_action, model)
    return {a: c for a, (c, _) in costs.items()}


def plan_action(x, y, yaw_deg, pillars, waypoint=None, stop_at_waypoint=False,
                stop_radius=0.5, vx=0.0, vy=0.0, yaw_rate=0.0, prev_action=None, model=NOMINAL):
    """返回 0..3 = plan_costs 的 argmin。waypoint/stop_at_waypoint/stop_radius 语义同 teacher.plan_action_global。

    stop_at_waypoint=True：距 waypoint ≤ stop_radius 直接 SAFE（与旧标签口径一致）；
    否则目标判定用「悬停后停住的位置」是否进半径，所以会提前一步刹车让滑行停在圈里。
    起点已漂进规划余量时第一步只查真实包络（teacher.py 脱困回退同一做法）；
    起点连悬停都停不住（四个首原语全 inf）→ SAFE，fail closed。
    """
    costs = plan_costs(x, y, yaw_deg, pillars, waypoint, stop_at_waypoint, stop_radius,
                       vx, vy, yaw_rate, prev_action, model)
    a = argmin_action(costs)
    return SAFE_NEUTRAL if a is None else a


def plan_sequence(x, y, yaw_deg, pillars, **kw):
    """调试/画图用：返回 argmin 首原语对应的整条规划动作序列（同 plan_action 的参数）。"""
    waypoint, stop = kw.get("waypoint"), bool(kw.get("stop_at_waypoint"))
    stop_radius = float(kw.get("stop_radius", 0.5))
    if waypoint is not None and stop and math.hypot(
            x - float(waypoint["x"]), y - float(waypoint["y"])) <= stop_radius:
        return [SAFE_NEUTRAL]
    field, goal, goal_r, start, clear_first = _setup(
        x, y, yaw_deg, pillars, waypoint, stop, stop_radius,
        kw.get("vx", 0.0), kw.get("vy", 0.0), kw.get("yaw_rate", 0.0))
    costs = search_costs(start, field, goal, goal_r, stop, clear_first, kw.get("prev_action"),
                         kw.get("model", NOMINAL))
    a = argmin_action(costs)
    return [] if a is None else costs[a][1]


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
    c = plan_costs(0.0, 0.0, 0.0, [])
    # 空场 5 m 目标：FORWARD ≈ 18–19 s；先悬停一步再走 = 多一步 + 换动作罚；先转更贵。
    check("plan_costs 单位是秒：空场 FORWARD 15–22 s", 15.0 <= c[FORWARD_SLOW] <= 22.0, f"{c[FORWARD_SLOW]:.2f}")
    check("plan_costs SAFE - FORWARD ≈ 一步 + 换动作罚",
          abs(c[SAFE_NEUTRAL] - c[FORWARD_SLOW] - STEP_S - W_SWITCH) < 1e-6,
          f"{c[SAFE_NEUTRAL] - c[FORWARD_SLOW]:.3f}")
    check("plan_costs 偏航 > SAFE > FORWARD", c[FORWARD_SLOW] < c[SAFE_NEUTRAL] < min(c[YAW_LEFT], c[YAW_RIGHT]))
    gate = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    c = plan_costs(1.18, 0.0, 0.0, gate, vx=0.3)
    check("贴柱必撞：四个首原语全 inf，plan_action fail closed = SAFE",
          all(math.isinf(v) for v in c.values()) and plan_action(1.18, 0.0, 0.0, gate, vx=0.3) == SAFE_NEUTRAL)
    c = plan_costs(0.85, 0.0, 0.0, gate, vx=0.3)
    check("门柱前 0.65 m 带速：FORWARD inf，偏航可行", math.isinf(c[FORWARD_SLOW]) and math.isfinite(c[YAW_RIGHT]))
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
