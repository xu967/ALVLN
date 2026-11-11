#!/usr/bin/env python3
"""TeacherPlanner：专家避障策略（训练时特权，学生不可用）。

输出：动作枚举值（与 src/include/uavrgb/action.hpp 对齐）：
  0=SAFE_NEUTRAL, 1=FORWARD_SLOW, 2=YAW_LEFT, 3=YAW_RIGHT。

三个教师：
  - expert_action：深度反应式（三竖带），用于没有柱子真值的对照集；
  - plan_action：局部射线 carrot planner，safe 出生集用它保留真 SAFE 标签；
  - plan_action_global：A* 全局路径 + 纯追踪，普通采集集默认用它。
    T1 实测局部规划在门柱前因 34°/步的偏航过冲而摆振，24 集里 15 集擦柱；
    A* 先找一条净空路径再跟，才能稳定穿过。

纯 stdlib，不依赖 numpy；合成深度/柱阵即可单测。
"""
import heapq
import json
import math

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")

# 教师实现版本：改动作选择（A* 参数/前向安全门/局部容差/目标定义）必须 bump，并在 TEACHER_RULES_BY_VERSION 加一行。
# collect 写进 manifest；audit 的 label_recompute 按 manifest 里的版本分派规则（set_vehicle(..., version=)），
# 旧版本的集按旧规则复算：不跳过，也不会用新算法复算旧标签造成假 FAIL。
TEACHER_VERSION = "uavrgb.teacher.v7"
# v7（第十二阶段 W1-7）：
#   B2 机体越过最远柱后表面（最远柱 x + 柱半宽）后，base 目标 (最远柱 x + 2.0, 0) 的 y 取当前 y（点 → 线）。
#      点目标让出口区的教师为回到 y=0 继续转向，这些 YAW 标签占出口区 17.7%，而学生看不到目标点（data-algo §A2）。
#   B3 A* 在标准余量下无路（起点落在余量内、八邻全堵）时，按 0.30、0.20 m 各重试一次，只试严格小于标准余量的档位，
#      仍无路才返回 None、由调用方退回局部规划（无路步原先退回局部规则左右摆，data-algo §A4）。
#   costmap 教师的 B2 同步生效，版本是 TEACHER_COST_VERSION（v2）。
# v6：forest_e1 档（余量 0.20/0.15 + 圆柱按圆判定），版本按任务档位分开：旧森林集（写 v5、按 0.40/0.30 方框算）
#   不能被它复算。forest_e1 档仍是旧规则（v6 = 不开 B2、B3）：B2、B3 只在 base 档的 20 个冻结 seed 上做过闭环验证，
#   森林档的标签逐位不变。
# v5：goal-stop（waypoint 作为最终目标）。
# 已知口径问题：5af664a 把 e1_mini 的 base 余量从 0.5/0.36 改到 0.40/0.30 并加脱困 v2 却没 bump，
# 0357 与 0417 两次 pilot 都写 v5；此前的 e1 集只能靠采集时间区分。
TEACHER_VERSION_BY_TASK = {"base": TEACHER_VERSION, "forest_e1": "uavrgb.teacher.v6"}


def teacher_version(task=None):
    """当前（或指定）任务档位的教师版本；未单列的任务都走 base 档。"""
    return TEACHER_VERSION_BY_TASK.get(task if task is not None else TEACHER_TASK, TEACHER_VERSION)


def task_of_scene(scene_meta):
    """从 scene.json 推断教师任务档位：forest_e1 族 → "forest_e1"，其余 → "base"。

    forest_e1 的 scene.json 没有 task 字段，只有 family / geometry_version 前缀；collect 和 audit
    都用这一个函数推断，避免两边各写一份判断。
    """
    if not isinstance(scene_meta, dict):
        return "base"
    if scene_meta.get("family") == "forest_e1":
        return "forest_e1"
    gver = scene_meta.get("geometry_version")
    if isinstance(gver, str) and gver.startswith("uavrgb.forest_e1."):
        return "forest_e1"
    return "base"

# 只取竖直方向的中间带（正前方走廊），比例范围 [0.4, 0.6)。
BAND_LO, BAND_HI = 0.4, 0.6

# carrot planner 把机器拉回走廊中线的增益；collect 与 audit 都走这个默认值，
# 改它必须两边一起变（否则 label_recompute 会把新数据全判不一致）。
# T1 实测 25 会在门柱前反复回中线、把已选的净空方向拉掉；普通集改用 A* 后
# 局部规划只剩 safe 出生集使用，保留 0 让 SAFE 位姿稳定。
Y_GAIN = 0.0


def _strip_min(depth, w, y0, y1, x0, x1):
    """行 [y0,y1) × 列 [x0,x1) 的最小有效深度和有效样本数。

    无效读数（<=0、NaN、±inf）跳过。返回 (inf, 0) 表示该区域一个有效样本都没有——
    调用方必须把它当成「不知道」，不能当成「空旷」。
    """
    best = math.inf
    n_valid = 0
    for y in range(y0, y1):
        base = y * w
        for x in range(x0, x1):
            d = depth[base + x]
            if math.isfinite(d) and d > 0.0:  # NaN/±inf/0/负数都是无效读数
                n_valid += 1
                if d < best:
                    best = d
    return best, n_valid


def expert_action(depth, w, h, turn_dist=1.5, emergency=0.8, prev_action=None,
                  switch_margin=0.75):
    """depth: 行主序深度（米），w*h 长度。返回 0..3 动作。

    prev_action/switch_margin 是**滞回**：两侧都可转时保持上次的转向方向，
    只有另一侧比当前侧空出 switch_margin 米以上才反向。没有滞回时，无人机一边转
    一边看深度，「哪侧更空」会在临界点来回翻转，实测出现左-右-左摆振（z 也从
    1.78 掉到 1.23），会污染训练标签。

    深度失效时 fail closed：中心带无有效样本 -> SAFE_NEUTRAL；某侧无有效样本
    -> 该侧视为堵。小尺寸深度（h<5 / w<3）和短 buffer 直接抛 ValueError。
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid depth size {w}x{h}")
    if len(depth) < w * h:
        raise ValueError(f"depth buffer too short: {len(depth)} < {w * h}")
    # h<5 时中间带 [0.4h,0.6h) 可能为空或只采一行，专家会看错方向。
    if h < 5:
        raise ValueError(f"depth height too small: h={h} (need h>=5)")
    if w < 3:
        raise ValueError(f"depth width too small: w={w} (need w>=3)")

    y0, y1 = int(h * BAND_LO), int(h * BAND_HI)
    xl = w // 3
    xr = w * 2 // 3
    left, left_n = _strip_min(depth, w, y0, y1, 0, xl)
    center, center_n = _strip_min(depth, w, y0, y1, xl, xr)
    right, right_n = _strip_min(depth, w, y0, y1, xr, w)

    if center_n == 0:
        # 深度相机没出图/全 0/全 NaN：不能假设前方空旷。
        return SAFE_NEUTRAL
    if center >= turn_dist:
        return FORWARD_SLOW

    if left_n == 0:
        left = 0.0
    if right_n == 0:
        right = 0.0
    # 只有三面都进了紧急距离才算真被围死。旧版要求「某一整侧 >= min_clear 才敢转」，
    # 一个 1m 宽的地标在 1.88m 处就会同时占满三条带，于是永远 SAFE_NEUTRAL（实测卡死）。
    if center < emergency and left < emergency and right < emergency:
        return SAFE_NEUTRAL
    left_ok = left >= emergency
    right_ok = right >= emergency
    if left_ok and right_ok:
        # 滞回：默认保持上次方向，除非另一侧明显更空。
        if prev_action == YAW_LEFT:
            return YAW_RIGHT if right > left + switch_margin else YAW_LEFT
        if prev_action == YAW_RIGHT:
            return YAW_LEFT if left > right + switch_margin else YAW_RIGHT
        return YAW_LEFT if left >= right else YAW_RIGHT
    if left_ok:
        return YAW_LEFT
    if right_ok:
        return YAW_RIGHT
    return SAFE_NEUTRAL


# 机身半宽（m）：只用于脱困段的真实碰撞盒判定，不改 clearance/lookahead 等规划几何。
# x3 0.235 是历史默认（标签逐字节不变）；e1_mini 取 sim/dynamics/e1_mini.json 的
# collision_envelope.radius_from_center 0.102（按方框半宽用，角部比圆保守 0.042 m）。
BODY_HALF_BY_VEHICLE = {"x3": 0.235, "e1_mini": 0.102}
BODY_HALF_M = BODY_HALF_BY_VEHICLE["x3"]
# 规划余量（m）：(A* 栅格膨胀, 局部射线/前向检查净空)，按 (载具, 任务档位) 取；未单列的任务走 (载具, "base")。
# x3 0.5/0.36 是历史默认（标签逐字节不变）。
# E1 沿用 X3 余量时 1.2 m 缝隙的可行走廊只剩 0.2 m：一步 0.25 m 带 20° 航向偏差就出走廊，
# 进余量后 A* 起点四邻全堵、局部脱困也找不到 1.6 m 净空射线 → 永远 SAFE，悬停漂移 2.4 cm/s 撞柱
# （E1 pilot 20 集 8 集这样失败）。E1 按「机身半宽差 0.133 m」缩，再各留 0.03/0.07 给漂移。
# forest_e1：缝只有 0.45/0.35/0.28 m，0.40 膨胀下 A* 把整片森林当墙、从带外绕（离线 20 seed：sparse 20/20 绕）。
# 0.20 时 sparse 20/20 穿过、medium 10/20、dense 3/20；0.15 时 medium 20/20、dense 9/20
# （tmp/agent-forest/offline_astar.json，2026-11-06）。取 A* 0.20 / 局部 0.15：局部 0.15 = E1 半径 0.102 + 0.048，
# 比森林走廊自身的 clearance 0.127 略宽。medium/dense 仍会部分绕行，是已知口径（见报告待确认）。
CLEARANCE_BY_VEHICLE = {
    ("x3", "base"): (0.5, 0.36),
    ("e1_mini", "base"): (0.40, 0.30),
    ("e1_mini", "forest_e1"): (0.20, 0.15),
}
TEACHER_TASK = "base"
ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M = CLEARANCE_BY_VEHICLE[("x3", "base")]
# 脱困阶段 (前这么多米只查机身方框, 射线长度 None=lookahead)：已漂进余量时按顺序试，第一阶段有解就停。
# x3 只有历史那一档。E1 多一档短射线：柱阵里离柱 0.26 m 时 1.6 m 全净空的射线几乎不存在
# （pilot 实测 142/173 个 SAFE 步在新余量下仍无解），0.6 m 够离开余量后让 A* 重新接手。
ESCAPE_STAGES_BY_VEHICLE = {"x3": ((0.2, None),), "e1_mini": ((0.2, None), (0.2, 0.6))}
ESCAPE_STAGES = ESCAPE_STAGES_BY_VEHICLE["x3"]
# 脱困 v2（x3 False = 历史行为，v5 标签逐字节不变）：① 全局规划里「已在余量内」时交给局部脱困而不是
# 按 err 原地左右换向；② 局部脱困选出射线后，前向 0.4 m 检查用同样的 skip，否则 t=0 永远被判堵、
# 脱困射线永远走不出去（只会 YAW）。E1 pilot 两种卡法都实测到了。
ESCAPE_V2 = False
# 教师规则档（按 TEACHER_RULES_BY_VERSION 取，set_vehicle 设置）。导入时的缺省 = 当前 TEACHER_VERSION 的规则：
# GOAL_LINE_AFTER_PILLARS = B2（过柱阵后目标取线）；ASTAR_RETRY_CLEARANCES = B3（A* 无路时依次重试的余量，空 = 不重试）。
GOAL_LINE_AFTER_PILLARS = True
ASTAR_RETRY_CLEARANCES = (0.30, 0.20)


def set_vehicle(name, task="base", version=None):
    """按 (载具, 任务档位) 切换机身半宽、规划余量与脱困策略，并按教师版本切换规则档（B2、B3）；未知载具/版本报错。

    collect.py 在解析 --vehicle 与 scene.json 后调用一次；audit 复算每集前按 manifest.vehicle +
    task_of_scene(scene.json) + manifest.teacher_version 调。任务没有单列余量时用该载具的 base 档（l1/safe/campus 都是柱阵）。
    version=None：取该任务档位当前的教师版本（teacher_version(task)）；给定时取该版本的规则，用来按旧规则复算旧数据。
    规则在改任何全局状态之前先校验：未知版本报错时不会留下改了一半的状态。
    """
    global BODY_HALF_M, ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M, ESCAPE_STAGES, ESCAPE_V2, TEACHER_TASK
    global GOAL_LINE_AFTER_PILLARS, ASTAR_RETRY_CLEARANCES
    if name not in BODY_HALF_BY_VEHICLE:
        raise ValueError(f"未知载具 {name!r}，可选 {sorted(BODY_HALF_BY_VEHICLE)}")
    if not isinstance(task, str) or not task:
        raise ValueError(f"任务档位必须是非空字符串，得到 {task!r}")
    rules = rules_for_version(version if version is not None else teacher_version(task))
    BODY_HALF_M = BODY_HALF_BY_VEHICLE[name]
    ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M = CLEARANCE_BY_VEHICLE.get(
        (name, task), CLEARANCE_BY_VEHICLE[(name, "base")])
    ESCAPE_STAGES = ESCAPE_STAGES_BY_VEHICLE[name]
    ESCAPE_V2 = name != "x3"
    TEACHER_TASK = task
    GOAL_LINE_AFTER_PILLARS, ASTAR_RETRY_CLEARANCES = rules["goal_line"], rules["retry"]
    return BODY_HALF_M


def _body_hits_box(px, py, yaw_deg, pillar, half):
    """方形机身（半宽 BODY_HALF_M，含 1 cm 余量）与柱方框是否重叠（2D 分离轴）。

    X3 为 0.47 m 方。机身随 yaw 旋转。按轴向包围盒会把 45° 机身的半宽当成 0.245 m，
    实际方角伸到 0.332 m，角部漏判；膨胀到外接圆又会在斜向误报、把脱困路径挡死。
    两条方框的分离轴只有 4 条：世界 x/y 与机身局部 x/y。
    带 radius 的圆柱（forest_e1 树干）：把柱心转到机身系，夹到方框上取最近点，距离 ≤ 半径即重叠。
    """
    a = BODY_HALF_M + 0.01
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = px - pillar["x"], py - pillar["y"]
    r = pillar.get("radius")
    if r is not None:
        lx, ly = -dx * cy - dy * sy, dx * sy - dy * cy   # 柱心在机身系的坐标
        qx, qy = max(-a, min(a, lx)), max(-a, min(a, ly))
        return (lx - qx) ** 2 + (ly - qy) ** 2 <= float(r) * float(r)
    hx, hy = (v / 2 for v in pillar.get("size", [half * 2, half * 2])[:2])
    # 世界 x/y 轴：机身投影半宽 = a*(|cos|+|sin|)
    body_w = a * (abs(cy) + abs(sy))
    if abs(dx) > body_w + hx or abs(dy) > body_w + hy:
        return False
    # 机身局部 x/y 轴：柱框投影半宽 = half*(|cos|+|sin|)（轴同为正交旋转）
    for ux, uy in ((cy, sy), (-sy, cy)):
        if abs(dx * ux + dy * uy) > a + hx * abs(ux) + hy * abs(uy):
            return False
    return True


def _pillar_blocked(px, py, pillar, clearance, half=0.2):
    """点 (px,py) 到柱**表面**的距离 < clearance 即堵。

    带 radius 的圆柱（forest_e1 树干，scene_forest 同时落盘外接方框 size=[2r,2r]）按圆：
    圆心距 < r + clearance。按外接方框判会把 45° 方向多算 0.41r 的障碍（r=0.15 时 6 cm），
    0.28–0.45 m 的缝会被判死。没有 radius 的方柱保持原来的方框算式（x3/E1 柱阵标签逐字节不变）。
    """
    r = pillar.get("radius")
    if r is not None:
        lim = float(r) + clearance
        return (px - pillar["x"]) ** 2 + (py - pillar["y"]) ** 2 < lim * lim
    hx, hy = (v / 2 for v in pillar.get("size", [half * 2, half * 2])[:2])
    dx = max(0.0, abs(px - pillar["x"]) - hx)
    dy = max(0.0, abs(py - pillar["y"]) - hy)
    return dx * dx + dy * dy < clearance * clearance


def ray_clear(x, y, th_deg, length, pillars, clearance, half=0.2, samples=33,
              skip_clearance_m=0.0, yaw_deg=0.0):
    """从 (x,y) 沿 th_deg 走 length 米，线段是否离所有柱子的**方框表面** ≥ clearance。

    不能用外接圆：0.4×0.4 方的外接圆半径 0.283 m，而车道缝隙只有 1.2 m，
    按外接圆算会把大部分窄缝判成堵死。

    采样精度：点到方框的距离是 1-Lipschitz，相邻采样点之间真实距离最多再降半个
    间隔。samples=33 时 1.6/32 = 0.05 m 间隔 → 最坏低算 25 mm（实测统计约 14 mm）。
    原先这里写"误差 < 1 mm"是错的。

    skip_clearance_m：前这么多米只查**真实碰撞盒**（0.47 m 机身方框 + 1 cm 余量），
    不查 clearance。用于 plan_action 的脱困回退：无人机已经漂进 0.36 m 安全余量时，
    从当前点出发的每条射线都会在 t=0 被判堵，旧逻辑会永远 SAFE 卡死（T1 实测
    8/8 场景卡在 x≈1.1–1.45）。真实碰撞仍然禁止，只是允许先离开余量。
    碰撞盒按 yaw 旋转的方框做分离轴判定（`_body_hits_box`）；外接圆或轴向包围盒
    都会在角部漏判/误判。
    """
    # shortcut: 脱困段只查机身方框，不查四个桨盘（外接半径 0.356 m）；
    # safe 集实测 0 碰撞，若出现擦桨再补逐盘圆判据。
    c, s = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
    for i in range(samples):
        t = length * i / (samples - 1)
        px, py = x + t * c, y + t * s
        for p in pillars:
            if t < skip_clearance_m:
                if _body_hits_box(px, py, yaw_deg, p, half):
                    return False
                continue
            if _pillar_blocked(px, py, p, clearance, half):
                return False
    return True


def plan_action(x, y, yaw_deg, pillars, lookahead=1.6, clearance=None,
                y_gain=Y_GAIN, y_max=30.0, tol=15.0, turn_weight=0.25, grid=5):
    """GT carrot planner：挑一条净空视线，代价 = 偏离目标角 + 0.25×当前转角。

    深度反应式专家解不了这条走廊（60 步 × 8 场景实测只有 1 个穿过，其余卡在
    门柱前左右摆）：它没有目标方向，转开之后只能直飞，航向一偏就斜着飞出
    ±1.7 m 柱区。这里改用柱子真值（特权信息，只决定标签、不进学生输入）。

    目标角 base = clamp(-y_gain*y, ±y_max) 把机器拉回走廊中线。**不能**简单地
    「从 base 向外试、取第一条空视线」：净空锥很窄时它会从 +43° 一步跳到 -47°，
    机器跟着左右各转 100° 以上、净位移为零（实测 4/8 场景这样卡死）。所以对
    -85°..85° 每 5° 求一次代价取最小；turn_weight 只用来在同分时偏向小转角，
    不    影响「回中线」的方向偏好。全都堵就退回 base。
    """
    if clearance is None:
        clearance = LOCAL_CLEARANCE_M   # 按载具（set_vehicle）；x3 0.36
    base = max(-y_max, min(y_max, -y_gain * y))

    def best_ray(skip_m, length=lookahead):
        aim, best = base, None
        for th in range(-85, 86, grid):
            if not ray_clear(x, y, th, length, pillars, clearance,
                             skip_clearance_m=skip_m, yaw_deg=yaw_deg):
                continue
            cost = (abs(th - base)
                    + turn_weight * abs((yaw_deg - th + 180.0) % 360.0 - 180.0))
            if best is None or cost < best:
                aim, best = th, cost
        return aim, best

    aim, best = best_ray(0.0)
    skip_used = 0.0
    for skip_m, length in ESCAPE_STAGES:
        if best is not None:
            break
        # 脱困：无人机可能已经漂进安全余量（T1 实测 8/8 场景因此卡死）。
        # 允许前 skip_m 米只查真实碰撞盒，离开余量后再要求 full clearance。
        aim, best = best_ray(skip_m, length or lookahead)
        skip_used = skip_m
    # 所有方向都堵时不能默认 base 可走：base 自己可能正对着柱子。
    if best is None and not ray_clear(x, y, base, lookahead, pillars, clearance):
        return SAFE_NEUTRAL
    err = (yaw_deg - aim + 180.0) % 360.0 - 180.0
    if abs(err) > tol:
        return YAW_LEFT if err < 0 else YAW_RIGHT
    # 局部回退同样不能盲目前进：0.4 m 内会进余量就先转。脱困 v2 下前 skip_used 米只查机身方框，
    # 否则在余量内 t=0 必堵，脱困射线永远走不出去。
    if not ray_clear(x, y, yaw_deg, 0.4, pillars, clearance,
                     skip_clearance_m=skip_used if ESCAPE_V2 else 0.0, yaw_deg=yaw_deg):
        return YAW_LEFT if err >= 0 else YAW_RIGHT
    return FORWARD_SLOW


def _point_blocked(x, y, pillars, clearance):
    """点到柱表面（方框；带 radius 时按圆）的距离 < clearance 即不可走（用于 A* 栅格）。"""
    for p in pillars:
        if _pillar_blocked(x, y, p, clearance):
            return True
    return False


# A* 与测地场共用的 8 邻接移动集（dx, dy, 步长倍数）；斜向 1.41421356 是历史常量，改成 math.sqrt(2) 会让
# 路长与 tie_margin 的数值变化（sgvis 行内 tie_margin_steps 逐位回归会炸）。
_MOVES = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
          (1, 1, 1.41421356), (1, -1, 1.41421356),
          (-1, 1, 1.41421356), (-1, -1, 1.41421356))


def _grid_spec(pillars, step):
    """astar_path / geodesic_field 共用的栅格 (x0, y0, nx, ny)：边界由柱子派生，避免改 PILLAR_X_RANGE 后 A* 静默退回局部规划。"""
    x0 = min(-1.5, min(float(p["x"]) for p in pillars) - 1.5)
    x1 = max(7.0, max(float(p["x"]) for p in pillars) + 3.0)
    y1 = max(2.8, max(abs(float(p["y"])) for p in pillars) + 1.5)
    y0 = -y1
    nx = int(round((x1 - x0) / step)) + 1
    ny = int(round((y1 - y0) / step)) + 1
    return x0, y0, nx, ny


# 栅格格子的「可走」只由 柱子几何 / 余量 / 栅格步长 决定。评测一集内每步都从当前位姿重跑 A*（教师臂是动作本身，
# 学生臂是影子教师标签），柱子不变；K=3 长课程里 85% 的 A* 耗时在逐格逐柱的 _point_blocked，每步 50–250 ms，
# 把动作生效时长压短约 11%（experiments/2026-11-10-long-course 第二轮）。按场景缓存格子，之后每步只剩堆操作；路径逐位不变。
_FREE_CACHE_MAX_SCENES = 4   # 一个采集 / 评测进程一次只飞一集；多留几个给离线批量重标（relabel 逐集换场景）
_FREE_CACHE = {}


def _free_cells(pillars, clearance, step):
    """(余量, 步长, 柱子几何) 对应的 {(i, j): 可走} 字典；键含 _pillar_blocked 用到的全部字段（x、y、radius、size）。"""
    key = (clearance, step, tuple((p["x"], p["y"], p.get("radius"), tuple(p["size"]) if "size" in p else None) for p in pillars))
    cells = _FREE_CACHE.get(key)
    if cells is None:
        if len(_FREE_CACHE) >= _FREE_CACHE_MAX_SCENES:
            _FREE_CACHE.pop(next(iter(_FREE_CACHE)))   # 字典保持插入序：丢最早的
        cells = _FREE_CACHE[key] = {}
    return cells


def warm_free_cells(pillars, clearance=None, step=0.1):
    """把整张 A* 栅格的「可走」一次填进缓存（--action-latency 的每集预热），返回格子数。

    astar_path 按需填格子：一集的第一步还没有缓存，A* 比后续步慢约 0.45 s（K=3 长课程），那一步的动作就晚了。
    预热按 astar_path 同一把钥匙（余量、步长、柱子几何）、同一个栅格范围（_grid_spec）、同一个判定（_point_blocked）
    逐格填满；格子的值只由钥匙决定，所以之后 astar_path 的路径与按需填的逐位相同。clearance 缺省取当前载具档
    （set_vehicle 之后的 ASTAR_CLEARANCE_M，与 plan_action_global 的缺省一致）。
    """
    if clearance is None:
        clearance = ASTAR_CLEARANCE_M
    x0, y0, nx, ny = _grid_spec(pillars, step)
    free = _free_cells(pillars, clearance, step)
    for i in range(nx):
        for j in range(ny):
            if (i, j) not in free:
                free[(i, j)] = not _point_blocked(x0 + i * step, y0 + j * step, pillars, clearance)
    return nx * ny


def astar_path(pillars, start, goal, clearance=0.5, step=0.1):
    """柱区 2D 栅格 A*，返回 [(x,y), ...] 或 None。

    局部射线规划在门柱前会左右摆（34°/步的偏航一次就跨过 10° 容差），
    T1 实测 24 集里 15 集擦柱。A* 从全局找一条净空路径，代价函数只是距离；
    障碍按柱方框 + clearance 膨胀，8 邻接且禁止斜穿角点。
    """
    x0, y0, nx, ny = _grid_spec(pillars, step)

    def idx(px, py):
        return int(round((px - x0) / step)), int(round((py - y0) / step))

    sx, sy = idx(*start)
    gx, gy = idx(*goal)
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    free = _free_cells(pillars, clearance, step)
    forced = [None]   # 起点格强制可走只对本次调用有效，不能写进跨调用共享的 free

    def is_free(i, j):
        k = (i, j)
        if k == forced[0]:
            return True
        if k not in free:
            free[k] = not _point_blocked(x0 + i * step, y0 + j * step, pillars, clearance)
        return free[k]

    if not is_free(gx, gy):
        return None
    # 起点已经在 0.5 m 规划余量内时不能让 A* 直接失败退回 0.36 m 的局部规划：
    # 强制把起点当自由格，路径先离开余量；纯追踪的前向 0.36 m 检查兜底。
    forced[0] = (sx, sy)
    open_heap = [(0.0, 0, sx, sy)]
    gscore = {(sx, sy): 0.0}
    parent = {}
    moves = _MOVES
    while open_heap:
        _f, _g, i, j = heapq.heappop(open_heap)
        if (i, j) == (gx, gy):
            path = []
            cur = (i, j)
            while cur in parent:
                path.append((x0 + cur[0] * step, y0 + cur[1] * step))
                cur = parent[cur]
            path.append((x0 + sx * step, y0 + sy * step))
            return path[::-1]
        gi = gscore[(i, j)]
        for di, dj, w in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not is_free(ni, nj):
                continue
            if di and dj and not (is_free(i + di, j) and is_free(i, j + dj)):
                continue
            ng = gi + w * step
            if ng < gscore.get((ni, nj), math.inf):
                gscore[(ni, nj)] = ng
                parent[(ni, nj)] = (i, j)
                h = math.hypot((ni - gx) * step, (nj - gy) * step)
                heapq.heappush(open_heap, (ng + h, ng, ni, nj))
    return None


def _path_carrot(path, x, y, lookahead):
    """离当前点最近的路径点之后，沿路径累计 lookahead 米的那一点。"""
    best = min(range(len(path)),
               key=lambda k: (path[k][0] - x) ** 2 + (path[k][1] - y) ** 2)
    acc = 0.0
    px, py = path[best]
    for qx, qy in path[best + 1:]:
        acc += math.hypot(qx - px, qy - py)
        px, py = qx, qy
        if acc >= lookahead:
            break
    return px, py


def _pursuit_err_deg(path, x, y, yaw_deg, lookahead):
    """纯追踪的航向误差（度，(-180,180]）：lookahead 米处胡萝卜的方位角减当前航向；正 = 目标在左。"""
    tx, ty = _path_carrot(path, x, y, lookahead)
    return (math.degrees(math.atan2(ty - y, tx - x)) - yaw_deg + 180.0) % 360.0 - 180.0


def _pillar_back_x(p):
    """柱的后表面 x：圆柱 x + radius；方柱 x + size[0]/2（缺 size 按 0.4 m 方柱的半宽 0.2）。"""
    r = p.get("radius")
    if r is not None:
        return float(p["x"]) + float(r)
    return float(p["x"]) + p.get("size", [0.4, 0.4])[0] / 2


def _pillars_back_x(pillars):
    """所有柱的后表面里最靠后的 x：机体 x 大于它，才算越过整个柱阵（B2 的触发线）。"""
    return max(_pillar_back_x(p) for p in pillars)


def _astar_retry(pillars, start, goal, clearance, step):
    """B3：标准余量下 A* 无路时，按 ASTAR_RETRY_CLEARANCES 里严格小于标准余量的档位依次重试，第一个有路的就用。

    无路的典型成因是起点已漂进余量、八邻全堵（起点格强制可走也出不去）。重试放宽的是「离柱面多远算可走」，
    路径仍是 A* 最短路；每个余量用自己的可走性缓存钥匙（_free_cells 的钥匙含余量），不会串档。
    标准余量下有路时只调一次 astar_path，与改动前逐位相同。ASTAR_RETRY_CLEARANCES 为空（旧版本规则、forest 档）时不重试。
    """
    path = astar_path(pillars, start, goal, clearance=clearance, step=step)
    if path is None:
        for c in ASTAR_RETRY_CLEARANCES:
            if c < clearance:
                path = astar_path(pillars, start, goal, clearance=c, step=step)
                if path is not None:
                    break
    return path


def _global_path(x, y, pillars, clearance, step, waypoint=None, stop_at_waypoint=False):
    """plan_action_global 跟踪的那条 A* 路径；None = 无路或 waypoint 字段坏（调用方退回局部 plan_action）。

    base 任务目标 (最远柱 x + 2.0, 0)：比验收线（最远柱 x + 1.0）再远 1 m，目标正好落在验收线上时
    飞控会在终点前转头/减速，max_x 常停在 5.97 而判失败。B2（v7）：机体 x 越过所有柱的后表面后，目标 y 取当前 y
    （点 → 线）：过了柱阵不再为回到 y=0 转向，学生也看不到那个点。waypoint 任务：stop_at_waypoint 时 waypoint 就是
    终点；旧 route-choice 口径是「先到 waypoint 再到走廊终点」两段拼接（B2 只作用于 base 目标，waypoint 任务不受影响）。
    tie-break 的滚出位姿也走这里，保证 Q_L/Q_R 与教师看的是同一张图、同一个目标。
    B3（v7）：每次 A* 无路时按 ASTAR_RETRY_CLEARANCES 降余量重试（见 _astar_retry）。
    """
    goal = (max(float(p["x"]) for p in pillars) + 2.0, 0.0)
    if waypoint is None:
        if GOAL_LINE_AFTER_PILLARS and x > _pillars_back_x(pillars):
            goal = (goal[0], float(y))
        return _astar_retry(pillars, (x, y), goal, clearance, step)
    try:
        wp = (float(waypoint["x"]), float(waypoint["y"]))
    except (KeyError, TypeError, ValueError):
        return None
    if stop_at_waypoint:
        return _astar_retry(pillars, (x, y), wp, clearance, step)
    first = _astar_retry(pillars, (x, y), wp, clearance, step)
    rest = (_astar_retry(pillars, wp, goal, clearance, step)
            if first is not None else None)
    return (first + rest[1:]) if first is not None and rest is not None else None


def plan_action_global(x, y, yaw_deg, pillars, clearance=None, step=0.1,
                       lookahead=0.7, tol=25.0, waypoint=None,
                       stop_at_waypoint=False, stop_radius=0.5):
    """A* 全局路径 + 纯追踪；失败时退回局部 plan_action。

    tol=25°：偏航一步约 34°，容差小于半步会左右过冲（T1 门柱振荡的根因）。
    路径目标比 T1 验收终点（最右柱 x+1.0）再远 1 m：目标正好落在验收线上时，
    飞控会在终点前转头/减速，max_x 常停在 5.97 而判失败；多飞 1 m 才真正越过。
    v7：越过柱阵后目标取线（B2），A* 无路时降余量重试（B3），细节见 _global_path / _astar_retry。

    stop_at_waypoint=True（L1 v2 goal-stop）：waypoint 就是最终目标，距其
    ≤ stop_radius 时输出 SAFE_NEUTRAL；不再去走廊终点。旧 route-choice 的
    “先到 waypoint 再飞终点”保留为默认，只用于复现旧数据。
    """
    if clearance is None:
        clearance = ASTAR_CLEARANCE_M   # 按载具（set_vehicle）；x3 0.5
    if not pillars:
        return plan_action(x, y, yaw_deg, [], tol=tol)
    if waypoint is not None and stop_at_waypoint:
        try:
            wp = (float(waypoint["x"]), float(waypoint["y"]))
        except (KeyError, TypeError, ValueError):
            return plan_action(x, y, yaw_deg, pillars, tol=tol)
        if math.hypot(x - wp[0], y - wp[1]) <= stop_radius:
            return SAFE_NEUTRAL
    path = _global_path(x, y, pillars, clearance, step, waypoint, stop_at_waypoint)
    if path is None:
        return plan_action(x, y, yaw_deg, pillars, tol=tol)
    err = _pursuit_err_deg(path, x, y, yaw_deg, lookahead)
    if err > tol:
        return YAW_LEFT
    if err < -tol:
        return YAW_RIGHT
    # 航向已对准也不能盲目前进：机体有惯性，A* 路径的余量（x3 0.5 m）会被一步
    # 0.27 m 的漂移吃掉（X3 实测；E1 cf_sitl hold 1.0 s 前飞稳态 0.25 m/步、首步 0.12 m，
    # SAFE 后还要滑行 0.07–0.21 m）。前向 0.4 m 若进机身余量（x3 0.36 m），先转再走。
    if not ray_clear(x, y, yaw_deg, 0.4, pillars, LOCAL_CLEARANCE_M):
        # 已经在余量内时这条检查对任何航向都不过，按 err 换向会左右摆到漂进柱子
        # （E1 pilot 实测 "LRLRLR…"）；交给局部规划的脱困阶段先离开余量。
        if ESCAPE_V2 and _point_blocked(x, y, pillars, LOCAL_CLEARANCE_M):
            return plan_action(x, y, yaw_deg, pillars, tol=tol)
        return YAW_LEFT if err >= 0 else YAW_RIGHT
    return FORWARD_SLOW


# ---------------------------------------------------------------- L/R 近平局按可见几何决定（DISTILL §2.1 / §3.3）
# plan_action_global 的左右决定有两处与图像无关：A* 对等长路径按 heap 顺序取一条；前向 0.4 m 被堵且 |err|≤tol 时
# 按 `err >= 0` 的符号换向。DISTILL 2026-11-07 在 300 集上量到：转弯帧 11.6%（approach 区 24.6%）两侧代价差
# < 0.5 步，学生拿到的 L/R 监督互相矛盾，val 上 L/R 召回逐轮翻转 0.19，ckpt_best 把 test 19.5% 的 YAW 帧预测成反向。
# 近平局的度量照搬 agent6-distill/q1_cost_to_go.py：对 YAW_LEFT / YAW_RIGHT 各滚出一步，
#   Q(a) = 1 + D(x′,y′)/v + max(0, |err′| − tol)/ω，D = 滚出点到目标的 A* 路长（同栅格、同膨胀、同目标）。
# 下面五个常量是「一步原语模型」的唯一来源：tie-break 的 |Q_L−Q_R|（lr_tie_margin_steps）与 costmap 教师（plan_costs）
# 都经 _rollout / _q_from 用它们；改任何一个都会同时改两边（sgvis 的行内 tie_margin_steps 回归会报）。
TIE_V_STEP_M = 0.25          # v：E1 cf_sitl hold 1.0 s 前飞稳态位移/步（DATA 报告实测中位）。X3 约 0.27，未按载具分：
TIE_YAW_STEP_DEG = 27.0      # ω：E1 偏航转角/步（同上）。X3 约 34°。平局阈值 0.5 步很粗，10% 的尺度差不改判定
TIE_YAW_DRIFT_M = 0.11       # 偏航一步沿新航向的平移（E1 实测；SAFE 后仍滑行 0.07–0.21 m）
TIE_COLLISION_STEPS = 40.0   # 滚出一步就撞机身方框或 A* 无路：记一集长度（40 步）
TIE_MARGIN_STEPS = 0.5       # |Q_L − Q_R| 小于它算近平局（DISTILL §2.1 用的阈值）
STUDENT_HFOV_DEG = 60.0      # 学生相机水平视场：sim/worlds 模板 <horizontal_fov>1.047</horizontal_fov>
TIEBREAKS = ("legacy", "visible")


def _rollout(x, y, yaw_deg, action):
    """一步原语后的位姿 (x', y', yaw')：FORWARD 沿航向 TIE_V_STEP_M；YAW ±TIE_YAW_STEP_DEG 并沿新航向平移 TIE_YAW_DRIFT_M；SAFE 原地。"""
    if action == FORWARD_SLOW:
        return (x + TIE_V_STEP_M * math.cos(math.radians(yaw_deg)),
                y + TIE_V_STEP_M * math.sin(math.radians(yaw_deg)), yaw_deg)
    if action in (YAW_LEFT, YAW_RIGHT):
        nyaw = yaw_deg + (TIE_YAW_STEP_DEG if action == YAW_LEFT else -TIE_YAW_STEP_DEG)
        return (x + TIE_YAW_DRIFT_M * math.cos(math.radians(nyaw)),
                y + TIE_YAW_DRIFT_M * math.sin(math.radians(nyaw)), nyaw)
    return x, y, yaw_deg


def _q_from(dist_m, err_deg, tol):
    """Q = 1 步 + 剩余路长/v + 航向罚 max(0, |err| − tol)/ω（单位：步）。"""
    return 1.0 + dist_m / TIE_V_STEP_M + max(0.0, err_deg - tol) / TIE_YAW_STEP_DEG


def _q_after_yaw(x, y, yaw_deg, action, pillars, clearance, step, lookahead, tol,
                 waypoint=None, stop_at_waypoint=False):
    """DISTILL §2.1 的 Q(a)（单位：步）：滚出一步 YAW 后的 A* 路长 + 航向罚；撞柱 / 无路 = TIE_COLLISION_STEPS。

    剩余路长用 A* 路径（_global_path）而不是 geodesic_field：两者最优长度只差浮点舍入，但 0.7 m 胡萝卜方向在
    等长路径之间可能不同；sgvis 集的行内 tie_margin_steps 是按这条算的，必须逐位不变。
    """
    nx, ny, nyaw = _rollout(x, y, yaw_deg, action)
    if any(_body_hits_box(nx, ny, nyaw, p, 0.2) for p in pillars):
        return TIE_COLLISION_STEPS
    path = _global_path(nx, ny, pillars, clearance, step, waypoint, stop_at_waypoint)
    if path is None:
        return TIE_COLLISION_STEPS
    length = sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(path, path[1:]))
    err = abs(_pursuit_err_deg(path, nx, ny, nyaw, lookahead))
    return _q_from(length, err, tol)


def lr_tie_margin_steps(x, y, yaw_deg, pillars, clearance=None, step=0.1, lookahead=0.7, tol=25.0,
                        waypoint=None, stop_at_waypoint=False):
    """|Q_L − Q_R|（步）：左转一步与右转一步之后到目标的代价差。小于 TIE_MARGIN_STEPS 即近平局。

    两边都撞/无路时为 0（两边一样坏，也算平局）。多跑两次 A*（E1 柱阵实测每次约 21 ms）。
    """
    if clearance is None:
        clearance = ASTAR_CLEARANCE_M
    q = [_q_after_yaw(x, y, yaw_deg, a, pillars, clearance, step, lookahead, tol,
                      waypoint, stop_at_waypoint) for a in (YAW_LEFT, YAW_RIGHT)]
    return abs(q[0] - q[1])


def visible_pillar_side(x, y, yaw_deg, pillars, hfov_deg=STUDENT_HFOV_DEG):
    """学生视锥内最近的柱在机头哪一侧：+1 左（方位角 > 0）、-1 右、0 正前方；视锥内一根都没有 → None。

    可见 = 柱外接圆有一部分落在 ±hfov/2 内（|方位角| − 角半径 ≤ hfov/2）。不做遮挡：最近的柱不会被更远的挡住。
    只看水平：柱高 3 m，巡航高度 2.1 m 处始终在竖直视场内（DISTILL §5 同一简化）。
    """
    best = None
    for p in pillars:
        dx, dy = float(p["x"]) - x, float(p["y"]) - y
        dist = math.hypot(dx, dy)
        r = p.get("radius")
        outer = float(r) if r is not None else math.hypot(*(v / 2 for v in p.get("size", [0.4, 0.4])[:2]))
        half_ang = 90.0 if dist <= outer else math.degrees(math.asin(outer / dist))
        bearing = (math.degrees(math.atan2(dy, dx)) - yaw_deg + 180.0) % 360.0 - 180.0
        if abs(bearing) - half_ang > hfov_deg / 2:
            continue
        if best is None or dist < best[0]:
            best = (dist, bearing)
    if best is None:
        return None
    return (best[1] > 0) - (best[1] < 0)


def plan_action_global_tiebreak(x, y, yaw_deg, pillars, tiebreak="legacy", clearance=None, step=0.1,
                                lookahead=0.7, tol=25.0, waypoint=None, stop_at_waypoint=False,
                                stop_radius=0.5):
    """plan_action_global + 近平局时按可见几何定左右。返回 (动作, tie_margin_steps)。

    legacy：原样返回 plan_action_global，margin None，不多跑一次 A*（默认采集路径逐位不变）。
    visible：每步算 |Q_L − Q_R|；只有 legacy 动作是 YAW 且 margin < TIE_MARGIN_STEPS 时才改向，改成
    「视锥内最近柱所在侧的反向」（柱在左就右转）：学生看得见这根柱，标签就成了图像的函数。视锥内无柱、
    或最近柱正对机头（方位角 0）时沿用 legacy。FORWARD / SAFE 不改（平局只关乎左右）。
    """
    if tiebreak not in TIEBREAKS:
        raise ValueError(f"tiebreak 必须是 {TIEBREAKS}，得到 {tiebreak!r}")
    action = plan_action_global(x, y, yaw_deg, pillars, clearance=clearance, step=step,
                                lookahead=lookahead, tol=tol, waypoint=waypoint,
                                stop_at_waypoint=stop_at_waypoint, stop_radius=stop_radius)
    if tiebreak == "legacy" or not pillars:
        return action, None
    margin = lr_tie_margin_steps(x, y, yaw_deg, pillars, clearance=clearance, step=step,
                                 lookahead=lookahead, tol=tol, waypoint=waypoint,
                                 stop_at_waypoint=stop_at_waypoint)
    if action in (YAW_LEFT, YAW_RIGHT) and margin < TIE_MARGIN_STEPS:
        side = visible_pillar_side(x, y, yaw_deg, pillars)
        if side:
            action = YAW_RIGHT if side > 0 else YAW_LEFT
    return action, margin


# ---------------------------------------------------------------- 统一代价模型教师（BRIEF_ALGO 2026-11-07 §1；DISTILL §2.1）
# legacy 教师（A* + 纯追踪 25° 容差）不是任何自然时间代价的 argmin：q1_cost_to_go 在 0417 的 12000 帧上量到逐动作
# cost-to-go 的 argmin 与 legacy 标签只有 72.2% 一致，P1 的失败根因是「示范专家 ≠ 标注专家」。这里让示范教师本身
# 就是 argmin_a Q(s,a)：Q 建在同一张测地场上（栅格 / 膨胀 / 移动集与 astar_path 同一套 helper），软标签只从这一个 Q 来。
# 版本独立于 TEACHER_VERSION：costmap 集的 manifest 写这个，audit 按 planner 字段选复算函数。
# v2（第十二阶段 W1-7 的 B2）：越过柱阵后目标取线（见 _costmap_goal）；v1 = 目标永远是点 (最远柱 x + 2.0, 0)。
TEACHER_COST_VERSION = "uavrgb.teacher_cost.v2"


# 版本 → 规则档。audit 的 label_recompute 按 manifest.teacher_version 取这张表（set_vehicle(..., version=)），
# 旧版本的集按旧规则复算；collect 与评测不传版本，取当前版本的规则。
#   goal_line：B2，越过柱阵后 base 目标的 y 取当前 y（A* 教师的 _global_path 与 costmap 教师的 _costmap_goal 共用这一个开关）。
#   retry：B3，A* 无路时依次重试的余量（只试严格小于标准余量的档位；costmap 教师不走 A*，用不到）。
# 加新版本必须加一行：set_vehicle 缺省取当前版本，表里没有会直接报错，不会悄悄沿用旧规则。
TEACHER_RULES_BY_VERSION = {
    "uavrgb.teacher.v5": {"goal_line": False, "retry": ()},
    "uavrgb.teacher.v6": {"goal_line": False, "retry": ()},      # forest_e1 档：旧规则
    TEACHER_VERSION: {"goal_line": True, "retry": (0.30, 0.20)},
    "uavrgb.teacher_cost.v1": {"goal_line": False, "retry": ()},
    TEACHER_COST_VERSION: {"goal_line": True, "retry": ()},
}
# audit 能按旧规则逐位复算的版本（按任务档位；最新的在最后）。写进这张表 = 承诺上面的规则档能复现该版本的标签。
RECOMPUTABLE_VERSIONS = {"base": ("uavrgb.teacher.v5", TEACHER_VERSION),
                         "forest_e1": (TEACHER_VERSION_BY_TASK["forest_e1"],)}
RECOMPUTABLE_COST_VERSIONS = ("uavrgb.teacher_cost.v1", TEACHER_COST_VERSION)


def rules_for_version(version):
    """教师版本对应的规则档 {"goal_line": bool, "retry": 余量元组}；版本不认识报错（不回落到当前规则）。"""
    rules = TEACHER_RULES_BY_VERSION.get(version) if isinstance(version, str) else None
    if rules is None:
        raise ValueError(f"未知教师版本 {version!r}，可选 {sorted(TEACHER_RULES_BY_VERSION)}")
    return dict(rules)


def recomputable_versions(task=None, costmap=False):
    """audit 能复算的教师版本：costmap 集看 RECOMPUTABLE_COST_VERSIONS，其余按任务档位（未单列的任务走 base 档）。"""
    if costmap:
        return RECOMPUTABLE_COST_VERSIONS
    return RECOMPUTABLE_VERSIONS.get(task if task is not None else TEACHER_TASK, RECOMPUTABLE_VERSIONS["base"])
COST_TIE_EPS = 1e-6          # |ΔQ| 小于它算平局（先按可见柱侧，再按动作枚举顺序）
_FIELD_CACHE_MAX = 16        # 每个场景一张场（柱子静态）；relabel 逐集扫 300 个场景时只保留最近的几张
_FIELD_CACHE = {}
_FIELD_ORDER = []


class GeodesicField:
    """目标出发的 Dijkstra 测地距离场（米）；不可行格 = inf。栅格 / 膨胀 / 8 邻接禁斜穿与 astar_path 同一套。"""
    __slots__ = ("x0", "y0", "nx", "ny", "step", "goal", "dist")

    def __init__(self, pillars, goal, clearance, step):
        self.x0, self.y0, self.nx, self.ny = _grid_spec(pillars, step)
        self.step, self.goal = step, (float(goal[0]), float(goal[1]))
        free = [[not _point_blocked(self.x0 + i * step, self.y0 + j * step, pillars, clearance)
                 for j in range(self.ny)] for i in range(self.nx)]
        dist = [[math.inf] * self.ny for _ in range(self.nx)]
        gi, gj = self.cell(*self.goal)
        if 0 <= gi < self.nx and 0 <= gj < self.ny and free[gi][gj]:
            dist[gi][gj] = 0.0
            heap = [(0.0, gi, gj)]
            while heap:
                d, i, j = heapq.heappop(heap)
                if d > dist[i][j]:
                    continue
                for di, dj, w in _MOVES:
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < self.nx and 0 <= nj < self.ny) or not free[ni][nj]:
                        continue
                    if di and dj and not (free[i + di][j] and free[i][j + dj]):
                        continue
                    nd = d + w * step
                    if nd < dist[ni][nj]:
                        dist[ni][nj] = nd
                        heapq.heappush(heap, (nd, ni, nj))
        self.dist = dist

    def cell(self, x, y):
        return int(round((x - self.x0) / self.step)), int(round((y - self.y0) / self.step))

    def lookup(self, x, y):
        """(D, 起算格)：落在膨胀带内时取 3×3 邻域里「D + 一跳」最小的格（A* 教师把起点强制为自由，等价于先离开余量）；出界 → (inf, None)。"""
        i, j = self.cell(x, y)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return math.inf, None
        if not math.isinf(self.dist[i][j]):
            return self.dist[i][j], (i, j)
        best, cell = math.inf, None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < self.nx and 0 <= jj < self.ny and not math.isinf(self.dist[ii][jj]):
                    d = self.dist[ii][jj] + self.step * math.hypot(di, dj)
                    if d < best:
                        best, cell = d, (ii, jj)
        return best, cell

    def descent_dir(self, cell, lookahead):
        """D 在 cell 处的「胡萝卜方向」（度）：沿最陡下降走到累计 lookahead 米处，取起点→终点的方位角；已在目标格 → None。

        8 邻域单步下降方向被量化到 45°，直接用会让航向项误差 ±22°、把开阔走廊里的 FWD 判成该转；
        沿测地线走 0.7 m 再取方向与 _path_carrot 的做法一致。
        下降率按**单位长度**比（octile 距离下直走与斜走每米降得一样多），同率先直走：若按邻格 D 最小取，
        只要有一点横向偏差就永远选斜格，开阔走廊里 7 cm 偏差会给出 45° 的胡萝卜（0417 复算时 2297 帧
        legacy FWD 被判成 YAW 的来源）；先直后斜与 astar_path 的阶梯顺序一致，远处目标的方向 ≈ 真实方位。
        """
        i, j = cell
        ci, cj, acc = i, j, 0.0
        while acc < lookahead:
            here, best_slope, nxt, w_best = self.dist[ci][cj], 1e-9, None, 0.0
            for di, dj, w in _MOVES:          # 轴向 4 个在前：同率时保留先遇到的直走
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.nx and 0 <= nj < self.ny:
                    slope = (here - self.dist[ni][nj]) / w
                    if slope > best_slope + 1e-9:
                        best_slope, nxt, w_best = slope, (ni, nj), w
            if nxt is None:
                break
            ci, cj = nxt
            acc += w_best * self.step
        if (ci, cj) == (i, j):
            return None
        return math.degrees(math.atan2(cj - j, ci - i))


def _pillars_key(pillars):
    return json.dumps(pillars, sort_keys=True, default=str)


def geodesic_field(pillars, goal, clearance, step=0.1):
    """按 (柱子, 目标, 膨胀, 栅格) 缓存的 GeodesicField；每个场景只算一次（E1 柱阵约 5000 格、10 ms）。"""
    key = (_pillars_key(pillars), (float(goal[0]), float(goal[1])), float(clearance), float(step))
    field = _FIELD_CACHE.get(key)
    if field is None:
        field = GeodesicField(pillars, goal, clearance, step)
        _FIELD_CACHE[key] = field
        _FIELD_ORDER.append(key)
        while len(_FIELD_ORDER) > _FIELD_CACHE_MAX:
            _FIELD_CACHE.pop(_FIELD_ORDER.pop(0), None)
    return field


def _costmap_goal(pillars, waypoint, stop_at_waypoint, x=None, y=None):
    """costmap 的目标点：base 任务 (最远柱 x + 2.0, 0)（与 _global_path 同）；goal-stop 任务就是 waypoint。

    B2（cost v2）：给了机体位置 (x, y) 且 x 已越过所有柱的后表面时，base 目标的 y 取当前 y（点 → 线，与 _global_path 同）。
    测地场按目标缓存，出口区每个不同的 y 一张场（E1 柱阵约 5000 格、10 ms）。不给 (x, y) 时仍是点目标。
    """
    if waypoint is None:
        goal_x = max(float(p["x"]) for p in pillars) + 2.0
        if GOAL_LINE_AFTER_PILLARS and x is not None and y is not None and x > _pillars_back_x(pillars):
            return (goal_x, float(y))
        return (goal_x, 0.0)
    if not stop_at_waypoint:
        raise ValueError("costmap 教师只支持 goal-stop 的 waypoint 任务，不支持旧 route-choice 两段目标")
    return (float(waypoint["x"]), float(waypoint["y"]))


def _q_value(field, pillars, x, y, yaw_deg, lookahead, tol, remaining):
    """位姿 (x,y,yaw) 的 Q（步）：撞机身方框 / 落进不可行格 → TIE_COLLISION_STEPS；remaining>1 时再向前枚举一步取最小。"""
    if any(_body_hits_box(x, y, yaw_deg, p, 0.2) for p in pillars):
        return TIE_COLLISION_STEPS
    d, cell = field.lookup(x, y)
    if cell is None or math.isinf(d):
        return TIE_COLLISION_STEPS
    if remaining <= 1:
        ddir = field.descent_dir(cell, lookahead)
        err = 0.0 if ddir is None else abs((ddir - yaw_deg + 180.0) % 360.0 - 180.0)
        return _q_from(d, err, tol)
    return 1.0 + min(_q_value(field, pillars, *_rollout(x, y, yaw_deg, a), lookahead, tol, remaining - 1)
                     for a in range(4))


def plan_costs(x, y, yaw_deg, pillars, clearance=None, step=0.1, lookahead=0.7, tol=25.0,
               waypoint=None, stop_at_waypoint=False, lookahead_steps=1):
    """四个原语各滚一步后的 cost-to-go（单位：步）。返回 (Q[4], info)。

    Q(a) = 1 + D(x',y')/v + max(0, |err'| − tol)/ω：D 是目标出发的测地场（栅格 0.1、膨胀 ASTAR_CLEARANCE_M、与 A* 同移动集），
    err' 是滚出后航向与 D 在 (x',y') 处 0.7 m 下降方向之差，v/ω 与 tie-break 同一组 TIE_* 常量。撞机身方框或落进不可行格
    → TIE_COLLISION_STEPS。tol 是显式参数（默认 25° = legacy 容差；tol=0 为允许的变体）；lookahead_steps=2 时
    Q(a) = min_{a2} [2 + D(after a,a2)/v + 罚]（16 条序列查表，仍是毫秒级）。接口只吃 (位姿, 柱子/目标)，3DGS 占据栅格
    可以换一张 field 复用（见报告接口说明）。
    """
    if not pillars:
        raise ValueError("plan_costs 需要柱子真值；没有柱子时用 plan_action")
    if lookahead_steps < 1:
        raise ValueError("lookahead_steps 必须 ≥ 1")
    if clearance is None:
        clearance = ASTAR_CLEARANCE_M
    goal = _costmap_goal(pillars, waypoint, stop_at_waypoint, x, y)
    field = geodesic_field(pillars, goal, clearance, step)
    q = [_q_value(field, pillars, *_rollout(x, y, yaw_deg, a), lookahead, tol, lookahead_steps)
         for a in range(4)]
    d_now, _cell = field.lookup(x, y)
    return q, {"goal": goal, "d_now_m": d_now, "tol": tol, "lookahead_steps": lookahead_steps}


def plan_action_costmap(x, y, yaw_deg, pillars, clearance=None, step=0.1, lookahead=0.7, tol=25.0,
                        waypoint=None, stop_at_waypoint=False, stop_radius=0.5, lookahead_steps=1):
    """argmin_a Q(s,a) 的示范教师。返回 (动作, info)，info = {"q": Q[4] 或 None, "tie": bool, "fallback": None|原因}。

    平局（|ΔQ| < COST_TIE_EPS）：左右都在平局集里时按 visible_pillar_side 取「视锥内最近柱的反侧」（柱在左就右转，
    标签成为图像的函数），看不到柱或柱正对机头时按动作枚举顺序取第一个；其它平局同样按枚举顺序。
    FORWARD / YAW_L / YAW_R 三个运动原语全是 TIE_COLLISION_STEPS（已漂进余量 ≥ 2 格或贴柱）时退回 plan_action 做局部脱困
    并标 fallback：SAFE 原地不会撞、Q 常常有限，若只在「四个全撞」时退回，机器会在柱前 0.4 m 原地保持到超时
    （门柱正对 x=0.9 实测 Q=[15.9, 40, 40, 40]）。不做换动作惩罚、不做滞回（teacher_kino 死在前者上；后者是 hack）。
    """
    info = {"q": None, "tie": False, "fallback": None}
    if not pillars:
        return plan_action(x, y, yaw_deg, [], tol=tol), dict(info, fallback="no_pillars")
    if waypoint is not None and stop_at_waypoint:
        try:
            wp = (float(waypoint["x"]), float(waypoint["y"]))
        except (KeyError, TypeError, ValueError):
            return plan_action(x, y, yaw_deg, pillars, tol=tol), dict(info, fallback="bad_waypoint")
        if math.hypot(x - wp[0], y - wp[1]) <= stop_radius:
            return SAFE_NEUTRAL, dict(info, fallback="goal_stop")
    q, _ = plan_costs(x, y, yaw_deg, pillars, clearance=clearance, step=step, lookahead=lookahead, tol=tol,
                      waypoint=waypoint, stop_at_waypoint=stop_at_waypoint, lookahead_steps=lookahead_steps)
    info["q"] = q
    if min(q[FORWARD_SLOW], q[YAW_LEFT], q[YAW_RIGHT]) >= TIE_COLLISION_STEPS:
        return plan_action(x, y, yaw_deg, pillars, tol=tol), dict(info, fallback="all_collide")
    q_min = min(q)
    cands = [a for a in range(4) if q[a] - q_min < COST_TIE_EPS]
    if len(cands) > 1:
        info["tie"] = True
        if YAW_LEFT in cands and YAW_RIGHT in cands:
            side = visible_pillar_side(x, y, yaw_deg, pillars)
            if side:
                return (YAW_RIGHT if side > 0 else YAW_LEFT), info
    return cands[0], info


def _make(w, h, left_d, center_d, right_d):
    """构造 w*h 深度图：左/中/右三条竖带各填一个深度值（所有行一致）。"""
    depth = []
    for _ in range(h):
        depth += [left_d] * (w // 3) + [center_d] * (w // 3) + [right_d] * (w - 2 * (w // 3))
    return depth


def _make_ground(w, h, band_d, ground_d):
    """构造 w*h 深度图：中间带 [0.4h,0.6h) 填 band_d，其余行填 ground_d（模拟地面）。"""
    y0, y1 = int(h * BAND_LO), int(h * BAND_HI)
    depth = []
    for y in range(h):
        d = band_d if y0 <= y < y1 else ground_d
        depth += [d] * w
    return depth


if __name__ == "__main__":
    w, h = 9, 10
    cases = [
        # (left, center, right, 期望动作)
        (5.0, 5.0, 5.0, FORWARD_SLOW, "全空 -> 前进"),
        (5.0, 0.3, 3.0, YAW_LEFT, "前方挡，左更空 -> 左转"),
        (3.0, 0.3, 5.0, YAW_RIGHT, "前方挡，右更空 -> 右转"),
        (0.3, 0.3, 0.3, SAFE_NEUTRAL, "三面堵死 -> 中立"),
        (0.3, 3.0, 0.3, FORWARD_SLOW, "仅前方通 -> 前进"),
    ]
    ok = True
    for left_d, center_d, right_d, want, name in cases:
        d = _make(w, h, left_d, center_d, right_d)
        got = expert_action(d, w, h)
        status = "ok" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {status}  {name}: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 地面排除：底/顶行都是 0.15（地面级近距），但中间带全 5.0 -> 应前进，不误判挡。
    d = _make_ground(w, h, band_d=5.0, ground_d=0.15)
    got = expert_action(d, w, h)
    want = FORWARD_SLOW
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  底边地面排除（中间带空）: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 滞回：两侧都能转、右略优于左，但上次是左转 -> 保持左转（不来回摆）。
    d = _make(w, h, 3.0, 0.3, 3.5)
    got = expert_action(d, w, h, prev_action=YAW_LEFT)
    want = YAW_LEFT
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  滞回-小幅差异保持原向: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 滞回：右侧明显更空（超出 switch_margin）-> 必须转向右。
    d = _make(w, h, 2.1, 0.3, 5.0)
    got = expert_action(d, w, h, prev_action=YAW_LEFT)
    want = YAW_RIGHT
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  滞回-明显更空则换向: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 滞回不能压过安全：上次左转，但左侧已被堵 -> 必须转右。
    d = _make(w, h, 1.0, 0.3, 5.0)
    got = expert_action(d, w, h, prev_action=YAW_LEFT)
    want = YAW_RIGHT
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  滞回不覆盖安全: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 回归：一个 1m 地标在 1.88m 处会填满三条带。旧版在此 SAFE_NEUTRAL 卡死（实测 7/10 步）。
    d = _make(w, h, 1.88, 1.88, 1.88)
    got = expert_action(d, w, h)
    want = FORWARD_SLOW
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  三带都 1.88m 不卡死 -> 前进: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 回归：再近一点（1.2m，已进转向距离但没到紧急距离）必须转向而不是停住。
    d = _make(w, h, 1.2, 1.2, 1.2)
    got = expert_action(d, w, h)
    want = YAW_LEFT   # 三侧相同 -> 确定性地先左转
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  三带都 1.2m -> 转向而非停住: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # GT carrot planner：正前方有柱子时必须选出一条让开的视线，而不是硬顶。
    gate = [{"x": 1.5, "y": 0.0}]
    ok = ok and not ray_clear(0.0, 0.0, 0.0, 1.6, gate, 0.36)
    ok = ok and ray_clear(0.0, 0.0, 0.0, 1.6, [], 0.36)
    got = plan_action(0.0, 0.0, 0.0, gate)
    ok = ok and got in (YAW_LEFT, YAW_RIGHT)   # 正中门柱左右对称，任一侧都对
    print(f"  {'ok' if got in (YAW_LEFT, YAW_RIGHT) else 'FAIL'}  门柱正前方 -> 让开: got {ACTIONS[got]}")
    plan_cases = [
        (dict(x=0.0, y=0.0, yaw_deg=0.0, pillars=[]), FORWARD_SLOW, "无柱子 -> 前进"),
        (dict(x=0.0, y=0.0, yaw_deg=40.0, pillars=[]), YAW_RIGHT, "航向偏左 40° -> 右转"),
        # 回中线依赖 y_gain；显式给 25，避免调全局 Y_GAIN 时这条用例失真。
        (dict(x=0.0, y=-1.2, yaw_deg=0.0, pillars=[], y_gain=25.0),
         YAW_LEFT, "偏在右侧 -> 左转回中线"),
    ]
    for kw, want, name in plan_cases:
        got = plan_action(**kw)
        ok = ok and got == want
        print(f"  {'ok' if got == want else 'FAIL'}  {name}: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 所有射线都堵、且 base 自身也被堵：必须 SAFE_NEUTRAL，不能默认前进撞柱。
    orig_ray_clear = ray_clear
    ray_clear = lambda *a, **k: False
    try:
        got = plan_action(0.0, 0.0, 0.0, [])
    finally:
        ray_clear = orig_ray_clear
    want = SAFE_NEUTRAL
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  全堵且 base 堵 -> 中立: "
          f"got {ACTIONS[got]}, want {ACTIONS[want]}")

    # T1 回归：无人机漂进 0.36 m 安全余量（距柱面 0.344 m，但 AABB 未碰撞）时，
    # 旧逻辑在 t=0 就把所有射线判堵，永远 SAFE；脱困回退必须给出转向/前进。
    gate = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    got = plan_action(1.1, 0.48, 0.0, gate)
    ok = ok and got != SAFE_NEUTRAL
    print(f"  {'ok' if got != SAFE_NEUTRAL else 'FAIL'}  已进安全余量仍能脱困: "
          f"got {ACTIONS[got]}, want != SAFE_NEUTRAL")

    # 脱困碰撞盒：轴向方框的角部（旧圆漏判）必须判堵；45° 机身方角伸到 0.346 m，
    # 正前方 0.5 m 的柱子也必须判堵（轴向 0.245 会漏）。
    corner = [{"x": 0.4, "y": 0.4, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    hit = not ray_clear(0.0, 0.0, 0.0, 0.2, corner, 0.36,
                        skip_clearance_m=0.2, yaw_deg=0.0)
    ok = ok and hit
    print(f"  {'ok' if hit else 'FAIL'}  角部碰撞盒：旧圆漏判的 (0.4,0.4) 必须判堵")
    # length=0：只判当前位姿（射线继续前进 0.2 m 后 yaw=0 也会真实撞上，测不出航向差）。
    front = [{"x": 0.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    clear0 = ray_clear(0.0, 0.0, 0.0, 0.0, front, 0.36,
                       skip_clearance_m=0.2, yaw_deg=0.0)
    hit45 = not ray_clear(0.0, 0.0, 0.0, 0.0, front, 0.36,
                          skip_clearance_m=0.2, yaw_deg=45.0)
    ok = ok and clear0 and hit45
    print(f"  {'ok' if clear0 and hit45 else 'FAIL'}  45° 机身角：yaw=0 判空、"
          f"yaw=45 判堵（轴向包围盒会漏）")

    # 深度失效必须 fail closed（M5）。
    extra = [
        ("全 0 深度 -> SAFE_NEUTRAL", [0.0] * (w * h), SAFE_NEUTRAL),
        ("全 NaN 深度 -> SAFE_NEUTRAL", [float("nan")] * (w * h), SAFE_NEUTRAL),
        ("全 inf 深度 -> SAFE_NEUTRAL", [float("inf")] * (w * h), SAFE_NEUTRAL),
        ("全负深度 -> SAFE_NEUTRAL", [-1.0] * (w * h), SAFE_NEUTRAL),
        ("中心带全 0、两侧 0.3 -> SAFE_NEUTRAL",
         _make(w, h, 0.3, 0.0, 0.3), SAFE_NEUTRAL),
        ("左侧无有效样本视为堵 -> YAW_RIGHT",
         _make(w, h, 0.0, 1.0, 5.0), YAW_RIGHT),
    ]
    for name, depth, want in extra:
        got = expert_action(depth, w, h)
        ok = ok and got == want
        print(f"  {'ok' if got == want else 'FAIL'}  {name}: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 小尺寸/短 buffer 必须拒绝，不能静默给出错误动作（S1）。
    bad = [
        ("h=1", _make(w, 1, 0.3, 5.0, 0.3), w, 1),
        ("h=2", _make(w, 2, 0.3, 5.0, 0.3), w, 2),
        ("h=3", _make(w, 3, 0.3, 5.0, 0.3), w, 3),
        ("w=2", _make(2, h, 0.3, 5.0, 0.3), 2, h),
        ("w=0", [], 0, h),
        ("短 buffer", [1.0] * (w * h - 1), w, h),
    ]
    for name, depth, ww, hh in bad:
        try:
            expert_action(depth, ww, hh)
        except ValueError:
            print(f"  ok  {name}: ValueError")
        else:
            ok = False
            print(f"  FAIL  {name}: 未抛 ValueError")

    # T1 回归：局部规划在门柱前会摆振；A* 必须能找到绕行路径并给出非 SAFE 动作。
    gate = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    path = astar_path(gate, (0.0, 0.0), (2.5, 0.0), clearance=0.5)
    ok = ok and path is not None
    got = plan_action_global(0.0, 0.0, 0.0, gate)
    ok = ok and got != SAFE_NEUTRAL
    print(f"  {'ok' if path is not None and got != SAFE_NEUTRAL else 'FAIL'}  "
          f"A* 绕门柱: path={len(path) if path else None}, action={ACTIONS[got]}")

    # v5 goal-stop：waypoint 半径内必须 SAFE；半径外不能误停。
    wp = {"x": 2.0, "y": 1.0}
    got_stop = plan_action_global(2.1, 1.05, 0.0, gate, waypoint=wp,
                                  stop_at_waypoint=True, stop_radius=0.5)
    got_go = plan_action_global(0.0, 0.0, 0.0, gate, waypoint=wp,
                                stop_at_waypoint=True, stop_radius=0.5)
    ok = ok and got_stop == SAFE_NEUTRAL and got_go != SAFE_NEUTRAL
    print(f"  {'ok' if got_stop == SAFE_NEUTRAL and got_go != SAFE_NEUTRAL else 'FAIL'}  "
          f"goal-stop: 半径内={ACTIONS[got_stop]}, 半径外={ACTIONS[got_go]}")

    # forest_e1 档：(载具, 任务) 余量与版本。base 档的余量与脱困策略一字不改（v7 只改目标与无路重试，见下面的 v7 自检）。
    try:
        ok_prof = set_vehicle("x3") == 0.235 and (ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M) == (0.5, 0.36)
        ok_prof = ok_prof and teacher_version() == TEACHER_VERSION == "uavrgb.teacher.v7"
        set_vehicle("e1_mini")
        ok_prof = ok_prof and (ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M) == (0.40, 0.30)
        ok_prof = ok_prof and teacher_version() == TEACHER_VERSION
        set_vehicle("e1_mini", "l1")   # 未单列的任务走 base 档
        ok_prof = ok_prof and (ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M) == (0.40, 0.30)
        ok_prof = ok_prof and teacher_version() == TEACHER_VERSION
        set_vehicle("e1_mini", "forest_e1")
        ok_prof = ok_prof and (ASTAR_CLEARANCE_M, LOCAL_CLEARANCE_M) == (0.20, 0.15)
        ok_prof = ok_prof and teacher_version() != TEACHER_VERSION and teacher_version("forest_e1") == teacher_version()
        ok_prof = ok_prof and task_of_scene({"family": "forest_e1"}) == "forest_e1"
        ok_prof = ok_prof and task_of_scene({"geometry_version": "uavrgb.forest_e1.v1"}) == "forest_e1"
        ok_prof = ok_prof and task_of_scene({"geometry_version": "uavrgb.geom.v5"}) == "base"
        ok_prof = ok_prof and task_of_scene(None) == "base"
        try:
            set_vehicle("e1_mini", "")
        except ValueError:
            pass
        else:
            ok_prof = False
        ok = ok and ok_prof
        print(f"  {'ok' if ok_prof else 'FAIL'}  (载具, 任务) 档位：forest_e1 0.20/0.15 + 独立版本，其余走 base 档")

        # 圆柱按圆判定：45° 方向上外接方框会多算 0.41r。r=0.15 的柱在 (0,0)，点 (0.28,0.28) 到圆面 0.246 m、
        # 到外接方框 0.184 m；余量 0.2 下圆判空、方框判堵。轴向上两者一致（0.34,0 → 0.19 m 都堵）。
        cyl = [{"x": 0.0, "y": 0.0, "z": 1.5, "radius": 0.15, "shape": "cylinder", "size": [0.3, 0.3, 3.0]}]
        box = [{"x": 0.0, "y": 0.0, "z": 1.5, "size": [0.3, 0.3, 3.0]}]
        ok_c = (not _point_blocked(0.28, 0.28, cyl, 0.2)) and _point_blocked(0.28, 0.28, box, 0.2)
        ok_c = ok_c and _point_blocked(0.34, 0.0, cyl, 0.2) and _point_blocked(0.34, 0.0, box, 0.2)
        ok_c = ok_c and ray_clear(0.28, 0.28, 90.0, 1.0, cyl, 0.2) and not ray_clear(0.28, 0.28, 90.0, 1.0, box, 0.2)
        # 脱困段的机身碰撞盒也按圆：E1 机身半宽 0.112，机身中心在 (0.21,0.21) 时柱心到机身方框 0.139 < 0.15 → 撞；
        # (0.26,0.26) 时 0.209 > 0.15 → 不撞，而外接方框（0.26 ≤ 0.112+0.15）仍判撞。
        ok_c = ok_c and _body_hits_box(0.21, 0.21, 0.0, cyl[0], 0.15) and not _body_hits_box(0.26, 0.26, 0.0, cyl[0], 0.15)
        ok_c = ok_c and _body_hits_box(0.26, 0.26, 0.0, box[0], 0.15)
        ok = ok and ok_c
        print(f"  {'ok' if ok_c else 'FAIL'}  圆柱按圆判定（45° 方向不再多算外接方框的角）")

        # forest_e1 档能穿过两根树干之间 0.6 m 的缝（圆面距），base 档（0.40 膨胀）A* 无路。
        pair = [{"x": 1.5, "y": 0.45, "z": 1.5, "radius": 0.15, "size": [0.3, 0.3, 3.0]},
                {"x": 1.5, "y": -0.45, "z": 1.5, "radius": 0.15, "size": [0.3, 0.3, 3.0]}]
        ok_f = astar_path(pair, (0.0, 0.0), (3.0, 0.0), clearance=ASTAR_CLEARANCE_M) is not None
        got = plan_action_global(0.0, 0.0, 0.0, pair)
        ok_f = ok_f and got == FORWARD_SLOW
        # 走廊外没有墙：A* 的边界是柱子派生的，目标在 (max_x+2, 0)；穿缝比绕 0.45+0.15+0.2 的柱更短，
        # 路径里 |y| 应留在缝内。
        path = astar_path(pair, (0.0, 0.0), (3.5, 0.0), clearance=ASTAR_CLEARANCE_M)
        ok_f = ok_f and path is not None and max(abs(py) for _px, py in path) < 0.3
        set_vehicle("e1_mini")   # base 档 0.40 膨胀：缝被判死，A* 只能从柱对外侧绕（离线 20/20 绕行的机理）
        path = astar_path(pair, (0.0, 0.0), (3.5, 0.0), clearance=ASTAR_CLEARANCE_M)
        ok_f = ok_f and path is not None and max(abs(py) for _px, py in path) > 0.6
        ok = ok and ok_f
        print(f"  {'ok' if ok_f else 'FAIL'}  forest_e1 档穿 0.6 m 缝（{ACTIONS[got]}），base 档 0.40 膨胀只能绕行")

        # L/R 近平局按可见几何（DISTILL §2.1）。两根柱 (1.5,0.35)、(2.4,-0.3)，机头 -10°：legacy A* 取左转，
        # |Q_L − Q_R| = 0.22 步 < 0.5；视锥内最近柱 (1.5,0.35) 在左 → visible 改右转。legacy 必须与
        # plan_action_global 相同且 margin None（不多跑 A*）。单柱 (1.0,0.3) 右绕明显更短（margin 1.4 步）→ 不改。
        tie = [{"x": 1.5, "y": 0.35, "z": 1.5, "size": [0.4, 0.4, 3.0]},
               {"x": 2.4, "y": -0.3, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
        leg, m_leg = plan_action_global_tiebreak(0.0, 0.0, -10.0, tie, "legacy")
        vis, m_vis = plan_action_global_tiebreak(0.0, 0.0, -10.0, tie, "visible")
        ok_t = leg == plan_action_global(0.0, 0.0, -10.0, tie) == YAW_LEFT and m_leg is None
        ok_t = ok_t and vis == YAW_RIGHT and m_vis < TIE_MARGIN_STEPS
        ok_t = ok_t and visible_pillar_side(0.0, 0.0, -10.0, tie) == 1
        ok_t = ok_t and visible_pillar_side(0.0, 0.0, 180.0, tie) is None          # 都在身后：视锥内无柱
        ok_t = ok_t and visible_pillar_side(0.0, 0.0, 0.0, [{"x": 1.0, "y": 0.0, "size": [0.4, 0.4, 3.0]}]) == 0
        one = [{"x": 1.0, "y": 0.3, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
        got_one, m_one = plan_action_global_tiebreak(0.0, 0.0, 0.0, one, "visible")
        ok_t = ok_t and got_one == plan_action_global(0.0, 0.0, 0.0, one) == YAW_RIGHT and m_one > TIE_MARGIN_STEPS
        ok_t = ok_t and plan_action_global_tiebreak(0.0, 0.0, 0.0, [], "visible") == (FORWARD_SLOW, None)
        try:
            plan_action_global_tiebreak(0.0, 0.0, 0.0, tie, "nope")
        except ValueError:
            pass
        else:
            ok_t = False
        ok = ok and ok_t
        print(f"  {'ok' if ok_t else 'FAIL'}  tiebreak=visible 近平局（{m_vis:.2f} 步）改向可见柱反侧，legacy 不变、非平局不改")

        # v7（W1-7）：B2 越过柱阵后目标取线——机体在 (最远柱 x + 0.5, 1.2)、机头 +x：v7 前进，v5 规则还在转向回 y=0；
        # B3 标准余量下无路（起点离柱前表面 0.25 m，0.40 m 余量八邻全堵）、0.30 m 余量有路：v7 给出路径，v5 规则无路 → 局部规划 SAFE。
        # 完整用例在 test_teacher_v7.py；这里只留一条最短的，让 check.sh 跑 teacher 自检时也覆盖规则开关。
        wall = [{"x": 1.55, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
        set_vehicle("e1_mini")
        ok_v7 = plan_action_global(2.0, 1.2, 0.0, gate) == FORWARD_SLOW
        ok_v7 = ok_v7 and _global_path(1.1, 0.0, wall, ASTAR_CLEARANCE_M, 0.1) is not None
        set_vehicle("e1_mini", "base", version="uavrgb.teacher.v5")
        ok_v7 = ok_v7 and plan_action_global(2.0, 1.2, 0.0, gate) == YAW_RIGHT
        ok_v7 = ok_v7 and _global_path(1.1, 0.0, wall, ASTAR_CLEARANCE_M, 0.1) is None
        ok_v7 = ok_v7 and plan_action_global(1.1, 0.0, 0.0, wall) == SAFE_NEUTRAL
        set_vehicle("e1_mini", "forest_e1")          # forest 档缺省 v6：B2、B3 都关
        ok_v7 = ok_v7 and not GOAL_LINE_AFTER_PILLARS and ASTAR_RETRY_CLEARANCES == ()
        ok_v7 = ok_v7 and all(rules_for_version(v) for v in TEACHER_RULES_BY_VERSION)
        ok_v7 = ok_v7 and TEACHER_VERSION in TEACHER_RULES_BY_VERSION and TEACHER_COST_VERSION in TEACHER_RULES_BY_VERSION
        ok_v7 = ok_v7 and recomputable_versions("base") == ("uavrgb.teacher.v5", TEACHER_VERSION)
        try:
            set_vehicle("e1_mini", "base", version="uavrgb.teacher.v99")
        except ValueError:
            pass
        else:
            ok_v7 = False
        ok = ok and ok_v7
        print(f"  {'ok' if ok_v7 else 'FAIL'}  v7：过柱阵后目标取线（FORWARD，v5 规则 YAW_RIGHT）、无路降余量重试（v5 规则 SAFE）、forest 档不开、版本表")
    finally:
        set_vehicle("x3")

    print("teacher: " + ("全部通过" if ok else "有失败"))
    raise SystemExit(0 if ok else 1)
