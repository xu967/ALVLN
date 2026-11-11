#!/usr/bin/env python3
"""oracle 方位护栏：评测端规则层，把「上层给的粗方向」加到任意策略上（学生或盲前进），不训练。

用的是真值位姿（动力学对象的 x / y / yaw）和占据 A* 的全局路径，都是特权信息，不是学生输入，
所以类名都带 Oracle；评测报告里「+route / +goal」的臂是学生/盲前进 + 这层规则，不能当成学生自己的能力。
动机：Gazebo 柱阵上「航向护栏」（机体自己数净转向步数）把学生 27/60 提到 47/60；3DGS 路线带转弯，
学生既不知道目标在哪也没见过真实外观，这里把两个因素拆开，只补前一个。

约定（与 planner.OccAstarTeacher / collect_3dgs.run_episode 同一套）：导航系 ENU，yaw 0 = +x，
正方向逆时针（YAW_LEFT 使 yaw 增大），policy 的 state 是动力学对象（x / y / yaw 弧度，窗口起点的真值位姿）。
误差 = 目标方位 − 机头方位，折到 [-180, 180)，正 = 目标在左 -> 护栏出 YAW_LEFT。

接口：OracleGuardedPolicy 就是一个 policy (rgb, depth, state) -> (action, infer_ms)，不需要改 run_episode。
每集新建一个对象（路线只在该集第一次调用时规划，不跨集复用）。
"""
import math

from .planner import YAW_LEFT, YAW_RIGHT, _wrap


def bearing_error_deg(x, y, yaw_deg, target_xy):
    """目标方位减机头方位，[-180, 180)，正 = 目标在左（与 OccAstarTeacher 的 err 同号同公式）。"""
    return _wrap(math.degrees(math.atan2(target_xy[1] - y, target_xy[0] - x)) - yaw_deg)


def bearing_guard_action(err_deg, threshold_deg):
    """|err| >= 阈值：返回朝目标转的动作；否则 None。阈值 <= 0 表示关闭。"""
    if threshold_deg <= 0 or abs(err_deg) < threshold_deg:
        return None
    return YAW_LEFT if err_deg > 0 else YAW_RIGHT


class OracleRouteGuide:
    """oracle 路线前视点：占据 A* 全局路径只规划一次（第一次 target() 时，从起飞后的实际位姿到 waypoint），之后不重规划。

    plan_fn(start_xy, goal_xy) -> [(x, y), ...] 或 None（planner.OccPlanner.plan 就是这个签名）。
    target(x, y) = 路径上「最近点再往前 lookahead_m（沿路径累计）」的点；剩余路径不足前视距离时返回 waypoint。
    最近点索引 self.k 只增不减：每次从上一次的索引往前走，只要下一个点不更远就前进。路径绕墙折回时前后两段
    只隔 1 m 左右，全局取最近点会跳回已走过的那段，朝前一段取全局最小则会跳到后面更近的一段（前视点反指、穿墙）。
    规划失败（被围死 / waypoint 在占据里）时一直直指 waypoint，plan_failed=True 留给汇总统计。
    shortcut: 最近点按顶点算（A* 栅格路径 0.1 m 一点，误差 < 0.05 m），稀疏折线不适用，要的话改线段投影。

    nearest_in_lookahead（W1-12，缺省 False = 上面的旧行为，target() 逐位不变）：
    旧规则停在第一个「下一个顶点更远」的顶点，也就是顶点距离序列的第一个局部极小。A* 栅格路径开头常有一个侧向的小凹口
    （例如 (0, 0) -> (0.1, -0.1) -> (0.2, 0)，对角走一步再对角走回来），飞机在凹口北侧起飞，凹口底部的顶点永远比前一个顶点远，
    指针停在凹口之前，目标点留在原地（K1 的 841150 r13 与 r4，复现见 experiments/2026-11-11-teacher-clearance-3dgs/replay_guide.py）。
    True：每次从当前指针往前，沿路径累计不超过 lookahead_m 的范围内取离飞机最近的顶点（并列取更靠前的），指针仍只增不减。
    范围限在前视距离内，是为了不让指针跳到折回的后一段上；折回的另一段若落在这个范围内（墙很薄、路径绕墙端不到 lookahead_m），
    指针仍会跳过去，调用方打开前要知道这一点。
    """

    def __init__(self, plan_fn, waypoint_xy, lookahead_m, nearest_in_lookahead=False):
        if not (math.isfinite(lookahead_m) and lookahead_m > 0):
            raise ValueError(f"lookahead_m 必须是有限正数，得到 {lookahead_m!r}")
        self.plan_fn, self.waypoint = plan_fn, (float(waypoint_xy[0]), float(waypoint_xy[1]))
        self.lookahead = float(lookahead_m)
        self.nearest_in_lookahead = bool(nearest_in_lookahead)
        self.path, self.k, self.plan_failed = None, 0, False

    def target(self, x, y):
        if self.path is None:
            self.path = self.plan_fn((x, y), self.waypoint) or []
            self.plan_failed = not self.path
        p = self.path
        if len(p) < 2:
            return self.waypoint

        def d2(i):
            return (p[i][0] - x) ** 2 + (p[i][1] - y) ** 2

        if self.nearest_in_lookahead:
            best, best_d, run = self.k, d2(self.k), 0.0
            for j in range(self.k + 1, len(p)):
                run += math.dist(p[j - 1], p[j])
                if run > self.lookahead:
                    break
                if d2(j) <= best_d:
                    best, best_d = j, d2(j)
            self.k = best
        else:
            while self.k + 1 < len(p) and d2(self.k + 1) <= d2(self.k):
                self.k += 1
        acc = 0.0
        for i in range(self.k, len(p) - 1):
            seg = math.dist(p[i], p[i + 1])
            if seg > 0 and acc + seg >= self.lookahead:
                f = (self.lookahead - acc) / seg
                return (p[i][0] + f * (p[i + 1][0] - p[i][0]), p[i][1] + f * (p[i + 1][1] - p[i][1]))
            acc += seg
        return self.waypoint


class OracleGuardedPolicy:
    """包住任意内层策略：先让内层出动作，再按当前真值位姿算目标方位误差，|err| >= threshold_deg 就改写成朝目标转。

    target_fn(x, y) -> (tx, ty)：路线护栏传 OracleRouteGuide.target，目标方位护栏传返回 waypoint 的函数。
    threshold_deg <= 0 = 关闭：不算目标（路线也不规划），原样放行，行为与不包装逐位一致。
    name 沿用内层：run_episode 靠它填 manifest 的 executed_by / action_expert（学生集不写教师标签）；
    故意不带 stop_radius / prev 等教师属性，否则 run_episode 会把它当 goal-stop 教师。
    统计：n_steps 总步数，n_triggered = |err| 超阈值的步数，n_rewritten = 其中内层动作与护栏动作不同、真被改写的步数；
    trace 每步一行（step / inner / action / err_deg / target / triggered / rewritten），调用方自己决定写不写旁路文件。
    改写之后，把真正执行的动作告诉内层（内层有 note_executed 就调）：prev-action 学生下一步看到的是护栏改写后的动作，
    不是它自己的输出（C2）。student_cond 照搬内层的，只给 run_episode 写 manifest 用。
    """

    def __init__(self, inner, target_fn, threshold_deg):
        self.inner, self.target_fn, self.threshold = inner, target_fn, float(threshold_deg)
        self.name = inner.name
        self.student_cond = getattr(inner, "student_cond", None)
        self.n_steps = self.n_triggered = self.n_rewritten = 0
        self.trace = []

    def reset(self):
        """集边界（run_episode 每集开头会调）：清统计，并把 reset 转给内层（学生的帧 / 动作缓存）。"""
        self.n_steps = self.n_triggered = self.n_rewritten = 0
        self.trace = []
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def __call__(self, rgb, depth, state):
        action, infer_ms = self.inner(rgb, depth, state)
        step = self.n_steps
        self.n_steps += 1
        if self.threshold <= 0:
            return action, infer_ms
        x, y = float(state.x), float(state.y)
        target = self.target_fn(x, y)
        err = bearing_error_deg(x, y, math.degrees(float(state.yaw)), target)
        turn = bearing_guard_action(err, self.threshold)
        rewritten = turn is not None and turn != action
        self.n_triggered += turn is not None
        self.n_rewritten += rewritten
        out = turn if rewritten else action
        if rewritten and hasattr(self.inner, "note_executed"):
            self.inner.note_executed(out)
        self.trace.append({"step": step, "inner": int(action), "action": int(out), "err_deg": round(err, 2),
                           "target": [round(target[0], 3), round(target[1], 3)],
                           "triggered": turn is not None, "rewritten": rewritten})
        return out, infer_ms


# 规格里的名字：同一批类（oracle 前缀是为了让特权信息在名字里可见）
RouteGuide, GuardedPolicy = OracleRouteGuide, OracleGuardedPolicy
