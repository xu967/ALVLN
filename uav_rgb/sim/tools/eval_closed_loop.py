#!/usr/bin/env python3
"""E6 闭环评测：学生在未知场景里自己飞，记录 SR / 碰撞率 / 误前向 / 推理延时。

每个 episode：生成固定 seed 的世界 -> 只保留指定 split 的几何 -> 起独立
GZ_PARTITION 的 gz sim -> collect.py --policy student --stop-on-collision ->
解析 actions.jsonl / oracle.jsonl。评测场景按几何组隔离，不会和训练几何重复。

用法（需要 .venv-tools 的 PIL；student 还要 uavrgb_infer 已构建）：
  python3 sim/tools/eval_closed_loop.py --policy student --ckpt <ckpt.bin> \
      --episodes 3 --out /tmp/eval1 --split test
  python3 sim/tools/eval_closed_loop.py --policy forward --episodes 3 \
      --out /tmp/eval-blind --split test
  # rule-oracle：柱子真值算完美列距离 + 固定规则 v0（零训练的规则层上限，不要 --ckpt）；
  # --rule-hfov-deg 60 = 学生相机，81 = 真机 GC4653 的水平视场
  python3 sim/tools/eval_closed_loop.py --policy rule-oracle --rule-hfov-deg 60 --episodes 3 \
      --out /tmp/eval-rule --split test
  # teacher-nn 用训练好的特权教师闭环（必须用带 torch 的解释器跑，见 --teacher-ckpt）
  $DATA_ROOT/uav_rgb/teacher-venv/bin/python sim/tools/eval_closed_loop.py \
      --policy teacher-nn --teacher-ckpt $DATA_ROOT/uav_rgb/teacher-run-pilot/ckpt_best.pth \
      --episodes 3 --out /tmp/eval-teacher --split test
  # T7 三个基线用同一批场景：把 student 的 seeds 回传给 --seeds。
  # 动作窗口默认 observation（唯一被采纳基线）；--window-from dispatch 仅用于复现
  # T7-R，其 SR/碰撞率不可与 observation 结果混报。
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(TOOLS)
REPO_ROOT = os.path.dirname(SIM_DIR)
SCENE_GEN = os.path.join(TOOLS, "scene_gen.py")
COLLECT = os.path.join(TOOLS, "collect.py")
RECORD = os.path.join(TOOLS, "record_views.py")
DEFAULT_TEMPLATE = os.path.join(SIM_DIR, "worlds", "uav_rgb.sdf")
from student import DEFAULT_INFER as DEFAULT_INFER_BIN
ACTION_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
WAYPOINT_RADIUS = 1.0
# L1 goal-reach 的到达半径。上限由几何定死，不是调出来的：
#   必须 < 两个候选 waypoint 最小间距 3.23 m 的一半（1.615），否则停在中间的无人机
#   对 keep 和 shuffle 都算成功，反事实就失效了；
#   必须 < 出生点到 waypoint 的最小距离 2.50 m，否则原地不动也算成功。
REACH_RADIUS = 1.5
# 新指标合同版本（SIM-09）。旧 success 定义不变，新字段单独按此版本解释。
# v2（2026-11-09，FIXLIST_R2 A1）：route_choice 任务（base/forest_e1）首次 x ≥ goal_x 即结束本集
# （collect.py --stop-at-goal），过线后不再有碰撞/超时，SR 不随 --steps 变；每集多记
# stopped_on_goal / steps_to_goal。v1 的数字过线后继续飞，碰撞可能发生在过线之后，不可与 v2 混报。
METRIC_VERSION = "sim09.v2"
# 柱阵任务的横向边界（2026-11-10）：route_choice 原先只判 max_x >= goal_x，世界里没有墙，
# 学生从柱阵旁边绕到 y=-17 m 再横穿成功线也会被记成成功（eval4 截断 seed1 的 20/20 里 8 集）。
# --corridor-half-width 给值后要求 oracle pose_gt 全程 |y| <= 该值，否则 success=False；
# 默认 None 逐位保持旧口径，metric_version 加 .corridor<N>m 后缀，新旧数字不可混报。
CORRIDOR_HALF_WIDTH_DEFAULT = None
# 逐块存活判据 block_outcomes 缺省的走廊半宽（m），等于 scene_gen.Y_RANGE 的边界（summarize.py 沿用多年的缺省值）。
# 评测命令行的 --corridor-half-width 默认仍是 None（关），与这个缺省无关；blocks_passed 一律显式传评测自己的值。
BLOCK_CORRIDOR_HALF_WIDTH_M = 3.0
# collisions_per_100m 的分母：run_episode 把每集量到首次碰撞 / 越界 / 过线的路程放进这个键，只给 write_report 汇总用，
# 不写进 per_episode（任务卡 W1-4 没列这个键；infer_ms 也是这样只供汇总、不落盘）。原始轨迹在 <out>/episodes/seed_N。
FLIGHT_PATH_KEY = "_flight_path_m"
# cf_sitl（固件在环）下 FORWARD 到期后机体还会滑行，碰撞步的 action 多半已是 SAFE/YAW；
# false_forward 改成「碰撞前 1 s 仿真内下过 FORWARD」。metric_version 带后缀，数字不与旧档混报。
COAST_WINDOW_S = {"cf_sitl": 1.0}
# stable_termination：末步真值速度 < 0.05 m/s（只有 oracle 带 speed_mps 的档能算，其余 None）。
STABLE_SPEED_MPS = 0.05
# SPL（Anderson 2018）的 l_i = 教师 A*（teacher.astar_path）参考路径长。base 任务的目标是验收线
# x=goal_x 上的任意点，A* 只认点目标，所以沿 y 每 SPL_GOAL_Y_STEP 取一个候选终点取最短
# （0.2 m：30 个候选 0.85 s/集；0.1 m 只少算 <1 cm）。只在整张障碍图就是 pillars 的任务上算，
# campus 的建筑不在 pillars 里，算出来会错，留 None。
SPL_GOAL_Y_STEP = 0.2
# course_base（长课程）的障碍图也就是 pillars，成功线 / SPL 最短路由 meta 推出，与 base 同一条路径。
SPL_TASKS = ("base", "l1", "forest_e1", "course_base")
# scene_gen/campus/asset_campus 目前都写死这个相机档；协议比对只用它做本次运行的实际值。
CAMERA_PROFILE = "rgb320x240_fov60_v1"
PROTOCOL_KEYS = ("window_from", "hold_s", "steps", "task",
                 "appearance_version", "dynamics", "camera_profile", "action_latency_s")

sys.path.insert(0, TOOLS)
import collect_parallel as cp  # noqa: E402
import instruction as ins  # noqa: E402
import split_manifest as sm  # noqa: E402
import campus  # noqa: E402
import asset_campus  # noqa: E402
import teacher  # noqa: E402
import collect  # noqa: E402  - 只为 GOAL_MARGIN_M：成功线与 collect --stop-at-goal 的停线必须是同一条
import rule_policy  # noqa: E402  - 只为 RULE_V0 / COL_DEG：参数随 eval.json 留证、视场范围校验

# 与 collect_parallel 共用同一份默认矩阵（cf_sil 已退出；cf_blackbox 只留名字）。
DYNAMICS_PROFILES = cp.DYNAMICS_CHOICES
# route_choice 成功线 = 最远柱 x + GOAL_MARGIN_M（单一来源 collect.GOAL_MARGIN_M = 1.0）。
GOAL_MARGIN_M = collect.GOAL_MARGIN_M


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(xs, p):
    """最近秩百分位；空列表返回 None。"""
    if not xs:
        return None
    ordered = sorted(xs)
    k = max(0, min(len(ordered) - 1, int(math.ceil(p / 100.0 * len(ordered))) - 1))
    return ordered[k]


def parse_seeds(text):
    """`--seeds` 文本（`23,25:28`）-> seed 列表；格式错误/重复 seed 抛 ValueError。"""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition(":")
        try:
            start = int(a)
            end = int(b) if b else start + 1
        except ValueError:
            raise ValueError(f"seed 不是整数: {part!r}（示例 23,25:28）") from None
        if start < 0 or end <= start:
            raise ValueError(f"seed 区间无效: {part!r}")
        seeds.extend(range(start, end))
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:
        raise ValueError(f"seed 重复: {dup}")
    return seeds


def dedup_key(meta, task):
    """唯一标识一个评测任务。

    base 用几何 gid；L1 必须把彩色柱的位置+颜色算进去。只用
    (gid, target_color) 会把「同几何、同目标色、但目标柱位置不同」的合法任务
    当成重复丢掉，也会让反事实对少一半。

    同色柱对之后 marks 不再能区分目标：左右两个任务的彩色柱完全一样，
    所以还要带上关系词，否则 left/right 两个任务会被去重成一个。
    """
    gid = sm.geometry_gid(meta["pillars"])
    if task not in ("l1", "campus", "campus_assets"):
        return gid
    marks = tuple(sorted(
        (round(float(p["x"]), 3), round(float(p["y"]), 3), p.get("color"))
        for p in meta["pillars"] if p.get("color")))
    if not marks:
        raise ValueError("L1 scene 没有彩色柱，无法确定评测任务")
    return (gid, marks, meta.get("relation"), meta.get("target_name"))


TEXT_MODES = ("keep", "blank", "shuffle")


def override_instruction(meta, mode):
    """L2 消融：keep=原指令；blank=空文本；shuffle=把 left/right 反过来。

    只改交给学生策略的 instruction，不改场景几何/标签/waypoint。
    同色柱对之后颜色两边一样，唯一的可控变量就是关系词——换色已经不可观测。
    """
    if mode == "keep":
        return meta
    out = dict(meta)
    if mode == "blank":
        out["instruction"] = ""
        return out
    color, relation = meta.get("target_color"), meta.get("relation")
    if color not in ins.COLORS or relation not in ins.RELATIONS:
        raise ValueError(
            f"shuffle 需要 target_color/relation，实际 {color!r}/{relation!r}")
    maker = ins.make_instruction
    # 按前缀判，不按具体版本：v1 与 v2 的指令生成器相同，旧数据仍要能评测。
    if str(meta.get("task_version", "")).startswith("campus."):
        from campus import instruction as maker
    out["instruction"] = maker(color, "right" if relation == "left" else "left")
    return out


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_target(episodes, seed_list):
    """未显式给 --episodes 时，显式 seed 列表就是目标集数。

    否则传 20 个 seed 会按默认 3 集跑完就退出，SR 只在 3 集上算。
    """
    return episodes if episodes is not None else (len(seed_list) or 3)


def appearance_version_for(task, appearance):
    return ("campus.v1" if task == "campus" else
            asset_campus.VERSION if task == "campus_assets" else appearance)


def load_protocol(path):
    """读协议快照 JSON；必须是对象，坏文件直接报错而不是静默忽略。"""
    with open(path, "r", encoding="utf-8") as f:
        protocol = json.load(f)
    if not isinstance(protocol, dict):
        raise ValueError("protocol JSON 顶层必须是对象")
    return protocol


def protocol_mismatches(protocol, actual):
    """协议中出现的字段与本次实际参数逐项比对，返回 (key, expected, got)。

    未知字段忽略：协议文件可以带 protocol_id 等备注，不能因此判 OOD。
    """
    return [(key, protocol[key], actual.get(key)) for key in PROTOCOL_KEYS
            if key in protocol and protocol[key] != actual.get(key)]


def wilson_ci(successes, n, z=1.96):
    """Wilson 95% 区间；n=0 返回 None。用于 requested 与 done-set 两个分母。"""
    if not n:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _src(a):
    """collision 来源；旧 episode 没有 src 字段时返回空串（按 contact 处理，保持旧口径）。"""
    return a.get("src") or ""


def coast_false_forward(actions, window_s):
    """碰撞（contact）步之前 window_s 仿真秒内是否下过 FORWARD（含碰撞步本身）。

    按 action_start_ns 计时；缺时间戳的行退回「碰撞步动作 == FORWARD」。
    """
    starts = [a.get("action_start_ns") for a in actions]
    for k, a in enumerate(actions):
        if not (bool(a.get("collision")) and _src(a) != "stuck"):
            continue
        if starts[k] is None:
            if int(a.get("action_executed", -1)) == 1:
                return True
            continue
        for j in range(k, -1, -1):
            if starts[j] is None:
                continue
            if (int(starts[k]) - int(starts[j])) > window_s * 1e9:
                break
            if int(actions[j].get("action_executed", -1)) == 1:
                return True
    return False


def trajectory_xy(oracle):
    """oracle 里带 x/y 的 pose_gt 序列 -> [(x, y), ...]。p_i 与 l_i 的起点都取它的首点。"""
    return [(float(o["pose_gt"]["x"]), float(o["pose_gt"]["y"])) for o in oracle
            if isinstance(o, dict) and isinstance(o.get("pose_gt"), dict)
            and "x" in o["pose_gt"] and "y" in o["pose_gt"]]


def path_length_m(points):
    """折线长（xy 平面；z 固定高度飞行，不计起飞段）。少于 2 点为 0。"""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def shortcut_path(path, segment_free):
    """视线抽捷径（string pulling）：从当前点起，沿路径往后看，直线段不穿障碍就继续往后，到第一段穿障碍为止跳过去。

    8 邻接栅格 A* 只有 0°/45° 两种走向，阶梯路径比欧氏最短路最长多 8%（octile/欧氏）；不抽的话多数成功集
    p_i < l_i，SPL 全被 max(p_i, l_i) 截成 1，列就白加了。抽完仍只走原路径经过的自由区，不会穿墙。
    """
    if len(path) < 3:
        return list(path)
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = i + 1
        while j + 1 < len(path) and segment_free(path[i], path[j + 1]):
            j += 1
        out.append(path[j])
        i = j
    return out


def _pillar_segment_free(pillars, clearance, sample_m=0.05):
    """线段 p->q 每 sample_m 采一点，全部离柱表面 >= clearance 才算通（与 A* 栅格同一个 _point_blocked）。"""
    def free(p, q):
        n = max(2, int(math.ceil(math.hypot(q[0] - p[0], q[1] - p[1]) / sample_m)) + 1)
        for k in range(n):
            t = k / (n - 1)
            if teacher._point_blocked(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), pillars, clearance):
                return False
        return True
    return free


def teacher_shortest_path_m(pillars, start, goal_x=None, waypoint=None, clearance=None,
                            step=0.1, goal_y_step=SPL_GOAL_Y_STEP, radius=None):
    """SPL 的 l_i：教师 A*（teacher.astar_path，柱方框膨胀 clearance）从 start 到目标的路径，抽捷径后的长度；无路返回 None。

    waypoint 任务（l1）目标是 waypoint 点，radius 给了就把路径截到进入半径那一点（与 p_i 同口径，
    见 truncate_at_goal）；base/forest 目标是验收线 x=goal_x 上的任意点，对 y 每 goal_y_step 取候选终点、
    取栅格路径最短的那条再抽捷径（A* 边界与 astar_path 内部同一算式）。
    clearance 由调用方给：评测器传机体半径（spl_clearance，E1 0.102 / X3 0.235），l_i 就是碰撞模型下可行的
    最短路下界（0.1 m 栅格 + 抽捷径，离散误差几 cm）；p_i < l_i 只会差在离散误差上，SPL 的 max(p_i, l_i) 兜住。
    不给 clearance 时退回教师规划余量 ASTAR_CLEARANCE_M（旧「教师参考路径」口径，离线脚本用）。
    """
    if not pillars:
        return None
    if clearance is None:
        clearance = teacher.ASTAR_CLEARANCE_M
    if waypoint is not None:
        goals = [(float(waypoint["x"]), float(waypoint["y"]))]
    else:
        if goal_x is None:
            return None
        y1 = max(2.8, max(abs(float(p["y"])) for p in pillars) + 1.5)
        n = int((2.0 * y1) / goal_y_step)
        goals = [(float(goal_x), -y1 + goal_y_step * (k + 1)) for k in range(n)
                 if -y1 + goal_y_step * (k + 1) < y1 - goal_y_step / 2]
    best = None
    for goal in goals:
        path = teacher.astar_path(pillars, start, goal, clearance=clearance, step=step)
        if path is None:
            continue
        if best is None or path_length_m(path) < path_length_m(best):
            best = path
    if best is None:
        return None
    pulled = shortcut_path(best, _pillar_segment_free(pillars, clearance))
    if waypoint is not None and radius is not None:
        pulled = truncate_at_goal(pulled, waypoint=waypoint, radius=radius)
    return path_length_m(pulled)


def _crossing_t(a, b, goal_x=None, waypoint=None, radius=None):
    """线段 a->b 上第一次进入目标区的参数 t∈[0,1]；整段都在区外返回 None（a 已在区内返回 0）。

    目标区：waypoint 任务 = 以 waypoint 为心、radius 为半径的圆盘；验收线任务 = 半平面 x >= goal_x。
    """
    if isinstance(waypoint, dict) and radius is not None:
        cx, cy, r = float(waypoint["x"]), float(waypoint["y"]), float(radius)
        fx, fy = a[0] - cx, a[1] - cy
        if math.hypot(fx, fy) <= r:
            return 0.0
        dx, dy = b[0] - a[0], b[1] - a[1]
        qa, qb, qc = dx * dx + dy * dy, 2.0 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r
        if qa <= 0.0:
            return None
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            return None
        t = (-qb - math.sqrt(disc)) / (2.0 * qa)      # 较小根 = 进入圆的那一点
        return t if 0.0 <= t <= 1.0 else None
    if goal_x is None:
        return None
    gx = float(goal_x)
    if a[0] >= gx:
        return 0.0
    if b[0] < gx or b[0] == a[0]:
        return None
    return (gx - a[0]) / (b[0] - a[0])


def truncate_at_goal(points, goal_x=None, waypoint=None, radius=None):
    """折线截到**第一次进入目标区**的那个点（线性插值到边界）；从未进入则原样返回。

    p_i 与 l_i 都用它截：两边都量到「目标区边界」，否则 waypoint 任务里 p_i 在半径 r 处就停、
    l_i 却量到圆心，p_i 系统性短 r，SPL 全被 max(p_i, l_i) 截成 1（3DGS r=2.0 m 对 11 m 路线差 18%）。
    """
    if len(points) < 2:
        return list(points)
    for i, (a, b) in enumerate(zip(points, points[1:])):
        t = _crossing_t(a, b, goal_x, waypoint, radius)
        if t is not None:
            out = list(points[:i + 1])
            if t > 0.0:
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            return out
    return list(points)


def goal_path_length_m(traj, goal_x=None, waypoint=None, radius=None):
    """SPL 的 p_i：轨迹从首点到**第一次进入目标区**（插值到边界）的折线长；从未到达则整条轨迹长。

    到达条件和 success 同一口径：waypoint 任务 = 到 waypoint 的欧氏距离 <= radius；
    验收线任务（base/forest）= x >= goal_x。截断的原因：harness 到线不停（base 没有 goal-stop，
    SAFE 在 E1 上是滑行），整条轨迹会把过线后继续飞的那段也算进分母，教师走直线也只剩 0.7。
    """
    if not traj:
        return None
    return path_length_m(truncate_at_goal(traj, goal_x, waypoint, radius))


def first_goal_step(oracle, goal_x=None, waypoint=None, radius=None):
    """steps_to_goal：第一次进入目标区的那一步的步数（oracle 行号 + 1，pose_gt 是该步动作后的真值位姿）。

    目标区与 success / p_i 同口径：waypoint 任务 = 距 waypoint <= radius；验收线任务 = x >= goal_x。
    从未进入返回 None。sim09.v2 下 route_choice 集在这一步就结束，所以它也等于 steps。
    """
    for i, o in enumerate(oracle):
        pg = o.get("pose_gt") if isinstance(o, dict) else None
        if not (isinstance(pg, dict) and pg.get("x") is not None):
            continue
        if waypoint is not None:
            if pg.get("y") is None:
                continue
            if math.hypot(float(pg["x"]) - waypoint["x"], float(pg["y"]) - waypoint["y"]) <= radius:
                return i + 1
        elif goal_x is not None and float(pg["x"]) >= goal_x:
            return i + 1
    return None


def spl_of(success, path_m, shortest_m):
    """单集 SPL：S_i · l_i / max(p_i, l_i)；缺任一长度为 None。两长度都为 0 时退化成 S_i。"""
    if path_m is None or shortest_m is None:
        return None
    denom = max(float(path_m), float(shortest_m))
    if denom <= 0.0:
        return float(bool(success))
    return float(bool(success)) * float(shortest_m) / denom


def block_outcomes(actions, oracle, goals, half_width=BLOCK_CORRIDOR_HALF_WIDTH_M, aborted=False):
    """一集的逐块结果（纯函数）。goals = 每块的过线 x（block_goal_x）。

    返回 {"pass": [bool]*K, "fail": None 或 (块序号从 0 起, "collision"|"corridor"|"not_reached"), "cross_step": [首次过线的步号或 None]*K,
          "cross_y": [过线那一步的 |y|（m）或 None]*K}。
    同一步既碰撞又越界按撞柱记；过线那一步本身有碰撞 / 越界，这一块不算过（与 K=1 判据：停线前任何碰撞都算失败 同口径）。
    aborted=True 是采集中途失败的集（多为固件 FAILSAFE：电池掉压后推力不够）：没有坏事件时记「采集中止」而不是「未到线」。
    half_width=None：不查横向（评测没开 --corridor-half-width 时 success 也不查，逐块判据与它同口径）。
    原在 experiments/2026-11-10-long-course/summarize.py；W1-4 起移到这里，eval.json 的 blocks_passed 与汇总脚本共用这一份。
    """
    k_total = len(goals)
    passed, cross, cross_y = [False] * k_total, [None] * k_total, [None] * k_total
    cur, fail = 0, None   # cur = 已通过的块数 = 正在穿越的块
    for i in range(max(len(actions), len(oracle))):
        a = actions[i] if i < len(actions) else {}
        o = oracle[i] if i < len(oracle) else {}
        pg = o.get("pose_gt") if isinstance(o, dict) else None
        has_pose = isinstance(pg, dict) and "x" in pg and "y" in pg
        collided = bool(a.get("collision")) and a.get("src") != "stuck"   # stuck（前飞位移不足）不是碰撞
        outside = has_pose and half_width is not None and abs(float(pg["y"])) > half_width
        if fail is None and (collided or outside):
            fail = (cur, "collision" if collided else "corridor")
        if fail is None and has_pose:
            while cur < k_total and float(pg["x"]) >= goals[cur]:
                passed[cur], cross[cur], cross_y[cur] = True, i, abs(float(pg["y"]))
                cur += 1
    if fail is None and cur < k_total:
        fail = (cur, "aborted" if aborted else "not_reached")
    return {"pass": passed, "fail": fail, "cross_step": cross, "cross_y": cross_y}


def blocks_passed(oracle, actions, block_goal_x, corridor_half_width):
    """这一集通过的块数（0..K）：第 k 块通过 = 首次到达 x >= block_goal_x[k] 之前（含那一步）无接触碰撞，且 |y| 不超过走廊半宽。

    参数顺序是 (oracle, actions)，与本文件 summarize_episode / block_outcomes 的 (actions, oracle) 相反（任务卡 W1-4 定的签名）。
    corridor_half_width=None（评测没开走廊）时不查横向，与该次评测的 success 同口径。
    """
    return sum(block_outcomes(actions, oracle, block_goal_x, corridor_half_width)["pass"])


def flight_path_m(oracle, actions, goal_x, waypoint=None, stop_radius=None, goal_radius=None,
                  corridor_half_width=None):
    """collisions_per_100m 的分母（一集）：oracle xy 折线，从首点量到首次接触碰撞、越界（|y| > 走廊半宽）或进入目标区为止，单位 m。

    碰撞 / 越界发生的那一步算在内；stuck 不是碰撞。目标区与 summarize_episode 的 p_i 同一个（验收线任务 = x >= goal_x 半平面，
    量到边界的插值点；waypoint 任务 = 到达圆）。越界只对验收线任务生效：waypoint 任务的 waypoint 本身可以在走廊外
    （见 summarize_episode 的 success 判据）。没有任何 pose_gt 时 None。
    """
    waypoint_task = stop_radius is not None
    half = None if waypoint_task else corridor_half_width
    end = len(oracle)
    for i, o in enumerate(oracle):
        a = actions[i] if i < len(actions) else {}
        pg = o.get("pose_gt") if isinstance(o, dict) else None
        has_pose = isinstance(pg, dict) and "x" in pg and "y" in pg
        outside = has_pose and half is not None and abs(float(pg["y"])) > half
        if (bool(a.get("collision")) and _src(a) != "stuck") or outside:
            end = i + 1   # 事件那一步的位姿也算进路程
            break
    return goal_path_length_m(trajectory_xy(oracle[:end]), goal_x=goal_x,
                              waypoint=waypoint if waypoint_task else None,
                              radius=(REACH_RADIUS if goal_radius is None else goal_radius) if waypoint_task else None)


def summarize_episode(actions, oracle, goal_x, waypoint=None,
                      waypoint_radius=WAYPOINT_RADIUS, stop_radius=None, goal_radius=None,
                      timed_out=False, coast_window_s=None, shortest_path_m=None,
                      corridor_half_width=None):
    """单集指标；纯函数，便于单测。actions/oracle 是 jsonl 行。

    EVAL_STANDARD §5.1：stuck（FORWARD 位移 < 阈值）不是碰撞，碰撞率只看 contact。
    新 collect 在 actions.jsonl 写 src（contact/stuck/contact+stuck）；旧数据没有该
    字段时退回原 collision 口径，不会把历史结果改成另一个数。

    stop_radius 非空（L1）：同时算两个口径，都写进每集记录。

    success（goal_reach_ever，2026-11-02 起的判据）= 无碰撞 + **轨迹上曾**进入
    waypoint 的 REACH_RADIUS 球内。理由：waypoint 是柱阵外侧一个没有视觉标记的点，
    把「末位姿落在 1.5 m 内」当判据实际在考「停的时机」，而 fit 到 0.03 m 却飞走的
    集会被记成完全没到，和「根本没走过去」混为一谈，掩盖了导航能力。
    stop_success（旧 goal_reach）= 无碰撞 + **末位姿**在 REACH_RADIUS 内，保留用于
    对照与历史可比，不再作为主指标。
    两者都不要求末动作 SAFE_NEUTRAL（原因见下方分支注释）；`stopped` 仍是诊断字段。
    旧 route-choice 数据仍走“max_x≥goal 且经过 waypoint”的旧口径，保持历史结果可复现。

    SIM-09 起另写一组字段（旧 success/stop_success 逐位不变）：
      reach_ever = 无 contact 且轨迹曾进入 goal_radius_reach（1.5 m）
      final_in_radius = 无 contact 且末位姿在 goal_radius_stop（scene.meta.stop_radius）
      stop_action_issued = 最近 waypoint 点前后各 2 步内出现过 SAFE_NEUTRAL；
        不只看最后一步。没有可靠速度数据，因此不用速度阈值。
      task_success = reach_ever + final_in_radius + stop_action_issued + 无 contact + 未超时；
        stable_termination：oracle 末行带 speed_mps（cf_sitl）时 = 速度 < STABLE_SPEED_MPS，
        否则 None（不知道，不等于通过）。
    coast_window_s（cf_sitl）：false_forward 改为「碰撞前该窗口内下过 FORWARD」，
    metric_version 加后缀 .coast<N>s。
    shortest_path_m（2026-11-07 起，只加不改）：调用方算好的 l_i（teacher_shortest_path_m /
    3DGS planner.route_check）；写 path_length_m（oracle xy 折线，截到首次到达，见 goal_path_length_m）、
    shortest_path_m、spl = success · l_i / max(p_i, l_i)。没给 l_i 时 spl 为 None，旧字段逐位不变。
    steps_to_goal（sim09.v2）：第一次进入目标区用的步数（first_goal_step），从未进入 None。
    """
    goal_radius_stop = float(stop_radius) if stop_radius is not None else None
    metric_version = (METRIC_VERSION if coast_window_s is None
                      else f"{METRIC_VERSION}.coast{coast_window_s:g}s")
    if corridor_half_width is not None:
        metric_version += f".corridor{corridor_half_width:g}m"
    if not actions:
        return {"steps": 0, "collision": False, "stuck": False, "false_forward": False,
                "success": False, "stop_success": False, "max_x": None,
                "waypoint_min_dist": None, "final_waypoint_dist": None,
                "stopped": False, "infer_ms": [],
                "reach_ever": False, "final_in_radius": False,
                "stop_action_issued": False, "collision_free": True,
                "stuck_free": True, "task_success": False,
                "stable_termination": None, "timed_out": bool(timed_out),
                "metric_version": metric_version,
                "y_max_m": None, "corridor_violation": False,
                "goal_radius_reach": REACH_RADIUS,
                "goal_radius_stop": goal_radius_stop,
                "path_length_m": None, "shortest_path_m": shortest_path_m, "spl": None,
                "steps_to_goal": None}
    collision = any(bool(a.get("collision")) and _src(a) != "stuck" for a in actions)
    stuck = any(bool(a.get("collision")) and "stuck" in _src(a) for a in actions)
    if coast_window_s is None:
        false_forward = any(bool(a.get("collision")) and _src(a) != "stuck" and
                            int(a.get("action_executed", -1)) == 1 for a in actions)
    else:
        false_forward = coast_false_forward(actions, coast_window_s)
    final_speed = oracle[-1].get("speed_mps") if oracle and isinstance(oracle[-1], dict) else None
    stable_termination = (bool(float(final_speed) < STABLE_SPEED_MPS)
                          if final_speed is not None else None)
    xs = [float(o["pose_gt"]["x"]) for o in oracle
          if isinstance(o.get("pose_gt"), dict) and "x" in o["pose_gt"]]
    max_x = max(xs) if xs else None
    # 横向越界只看有真值 pose_gt 的集；waypoint 任务（l1/campus）的 waypoint 本身就在柱阵外侧
    # （|y| 可达 2.4 m），只对 route_choice 这种「穿越验收线」的判据生效。
    ys = [abs(float(o["pose_gt"]["y"])) for o in oracle
          if isinstance(o.get("pose_gt"), dict) and "y" in o["pose_gt"]]
    y_max = max(ys) if ys else None
    corridor_violation = bool(corridor_half_width is not None and y_max is not None
                              and y_max > float(corridor_half_width))
    wp_min = wp_final = None
    nearest_i = None
    if isinstance(waypoint, dict):
        ds = []
        best = None
        for i, o in enumerate(oracle):
            pg = o.get("pose_gt")
            if isinstance(pg, dict) and "x" in pg and "y" in pg:
                d = ((pg["x"] - waypoint["x"]) ** 2 +
                     (pg["y"] - waypoint["y"]) ** 2) ** 0.5
                ds.append(d)
                if best is None or d < best:
                    best, nearest_i = d, i
        wp_min = min(ds) if ds else None
        wp_final = ds[-1] if ds else None
    stopped = int(actions[-1].get("action_executed", -1)) == 0
    # 最近点前后各 2 步：SAFE 常出现在到达点前一两步，只看末步会把「到点后继续飞走」
    # 全判成没停。窗口按 actions 行与 oracle 行一一对应取索引。
    stop_action_issued = bool(nearest_i is not None and any(
        int(a.get("action_executed", -1)) == 0
        for a in actions[max(0, nearest_i - 2):nearest_i + 3]))
    if stop_radius is not None:
        # 2026-11-01 起不再要求末动作 SAFE_NEUTRAL：waypoint 在柱阵外侧 ±2.4 m，
        # 相机 60° 视场在真停止点上 40% 的帧一根柱子都看不到，"停"不可观测
        # （worklog 2026-11-01-2130-stop-point-unobservable.md）。改成"到达"。
        radius = REACH_RADIUS if goal_radius is None else goal_radius
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("goal_radius 必须有限且为正")
        success = bool(not collision and wp_min is not None
                       and wp_min <= radius)
        stop_success = bool(not collision and wp_final is not None
                            and wp_final <= radius)
    else:
        reached = wp_min is None or wp_min <= waypoint_radius
        success = bool(not collision and max_x is not None and max_x >= goal_x and reached
                       and not corridor_violation)
        stop_success = success
    collision_free = not collision
    reach_ever = bool(collision_free and wp_min is not None
                      and wp_min <= REACH_RADIUS)
    final_in_radius = bool(collision_free and wp_final is not None
                           and goal_radius_stop is not None
                           and wp_final <= goal_radius_stop)
    task_success = bool(reach_ever and final_in_radius and stop_action_issued
                        and collision_free and not timed_out)
    # p_i 截到第一次到达（与 success 同口径）：waypoint 任务用 success 用的那个 radius，验收线任务用 goal_x。
    goal_region = dict(goal_x=goal_x,
                       waypoint=waypoint if stop_radius is not None else None,
                       radius=(REACH_RADIUS if goal_radius is None else goal_radius)
                       if stop_radius is not None else None)
    p_i = goal_path_length_m(trajectory_xy(oracle), **goal_region)
    steps_to_goal = first_goal_step(oracle, **goal_region)
    infer = [float(a["infer_ms"]) for a in actions if a.get("infer_ms") is not None]
    return {"steps": len(actions), "collision": collision, "stuck": stuck,
            "false_forward": false_forward, "success": success,
            "stop_success": stop_success,
            "max_x": max_x, "waypoint_min_dist": wp_min,
            "final_waypoint_dist": wp_final, "stopped": stopped, "infer_ms": infer,
            "reach_ever": reach_ever, "final_in_radius": final_in_radius,
            "stop_action_issued": stop_action_issued,
            "collision_free": collision_free, "stuck_free": not stuck,
            "task_success": task_success,
            # 只有 oracle 带真值速度（cf_sitl）才能判；其余档 None，不等于通过。
            "stable_termination": stable_termination, "timed_out": bool(timed_out),
            "metric_version": metric_version,
            "y_max_m": y_max, "corridor_violation": corridor_violation,
            "goal_radius_reach": REACH_RADIUS,
            "goal_radius_stop": goal_radius_stop,
            # SPL 三件：p_i（oracle xy 折线，截到首次到达）、l_i（调用方给的教师参考路长）、S_i·l_i/max(p_i,l_i)。
            "path_length_m": p_i, "shortest_path_m": shortest_path_m,
            "spl": spl_of(success, p_i, shortest_path_m),
            "steps_to_goal": steps_to_goal}


def aggregate(episodes, requested=None, attempted=None, failed_count=0, skipped_count=0):
    """汇总。分母写清楚：
    - success_rate / 新指标率 = episodes_done；
    - requested_success_rate = episodes_requested（固定请求集，含 failed/skipped 的非成功）；
    - completion_rate = 实际尝试数（done+failed，skipped 不算）；
    - execution_rate = episodes_done / episodes_requested（skipped 表现为未执行）。
    比率都落在 [0,1]：失败后补扫 seed 不会把 execution_rate 推过 1。
    """
    done = [e for e in episodes if e.get("steps", 0) > 0]
    n = len(done)
    if attempted is None:
        attempted = n
    infer_all = [ms for e in done for ms in e["infer_ms"]]
    infer_warm = [ms for e in done for ms in e["infer_ms"][1:]]  # 首帧含 CUDA 初始化
    dist = {name: 0 for name in ACTION_NAMES}
    for e in done:
        for k, v in e.get("actions", {}).items():
            if 0 <= k < len(ACTION_NAMES):
                dist[ACTION_NAMES[k]] += v

    def rate(key):
        return (sum(1 for e in done if e.get(key)) / n) if n else None

    successes = sum(1 for e in done if e.get("success"))
    corridor_violations = sum(1 for e in done if e.get("corridor_violation"))
    # SPL 只在算出 l_i 的集上平均（A* 无路 / 旧 eval.json 没有该字段的集不进分母，spl_n 单列）；
    # requested_spl 与 requested_success_rate 同口径：失败/跳过集记 0，只有全部完成集都有 spl 时才给。
    spls = [float(e["spl"]) for e in done if e.get("spl") is not None]
    return {
        "episodes_done": n,
        "success_rate": rate("success"),
        # 越界集数：开了 --corridor-half-width 才有意义，没开时应为 0。
        "corridor_violations": corridor_violations,
        "spl": (sum(spls) / len(spls)) if spls else None,
        "spl_n": len(spls),
        "requested_spl": (sum(spls) / requested) if requested and spls and len(spls) == n else None,
        "requested_success_rate": (successes / requested) if requested else None,
        # skipped 不进 completion 分母：只看真正尝试过的集（失败算尝试）。
        "completion_rate": (n / attempted) if attempted else None,
        "execution_rate": (n / requested) if requested else None,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "unique_gids": len({e["geometry_gid"] for e in done
                            if e.get("geometry_gid") is not None}),
        "wilson_ci": {
            "success": wilson_ci(successes, n),
            "requested": wilson_ci(successes, requested) if requested else None,
        },
        # 旧「末位姿在 1.5 m 内」口径，保留对照；主指标改看 success_rate。
        "stop_success_rate": rate("stop_success"),
        # SIM-09 新口径：分别报告，旧 success_rate 不动。
        "reach_ever_rate": rate("reach_ever"),
        "final_in_radius_rate": rate("final_in_radius"),
        "stop_action_issued_rate": rate("stop_action_issued"),
        "task_success_rate": rate("task_success"),
        "collision_free_rate": rate("collision_free"),
        "stuck_free_rate": rate("stuck_free"),
        # 只在有真值速度的集上算（cf_sitl）；一集都没有就是 None，不是 0。
        "stable_termination_rate": (
            (sum(1 for e in done if e.get("stable_termination")) /
             sum(1 for e in done if e.get("stable_termination") is not None))
            if any(e.get("stable_termination") is not None for e in done) else None),
        "collision_rate": rate("collision"),
        # stuck 是「被挡住没动」而非接触；EVAL_STANDARD 要求单列，不能混进碰撞率。
        "stuck_rate": rate("stuck"),
        "false_forward_rate": rate("false_forward"),
        "inference_ms": {
            "p50": percentile(infer_warm, 50),
            "p95": percentile(infer_warm, 95),
            "n_warm": len(infer_warm),
            "cold_first_ms": [round(e["infer_ms"][0], 3) for e in done if e["infer_ms"]],
            "p50_all": percentile(infer_all, 50),
            "p95_all": percentile(infer_all, 95),
        },
        "action_distribution": dist,
    }


def collisions_per_100m(episodes):
    """每百米接触碰撞数 = 100 × 有接触碰撞的集数 ÷ 各集 FLIGHT_PATH_KEY（m）之和；集数口径同 collision_rate 的分子。

    episodes 是完成集（run_episode 的返回，failed[] / skipped[] 的集不在内，没有路程可量）。总路程为 0（含没有完成集）时 None，
    不是 0：没飞过不能说每百米 0 次碰撞。零碰撞但飞过时是 0.0。
    """
    total = sum(e.get(FLIGHT_PATH_KEY) or 0.0 for e in episodes)
    if total <= 0.0:
        return None
    return 100.0 * sum(1 for e in episodes if e.get("collision")) / total


def block_survival(episodes, requested, blocks):
    """逐块存活率：长度 blocks 的列表，第 k 项（k 从 1 起）= blocks_passed >= k 的集数 ÷ requested（请求集数）。

    分母同 requested_success_rate：采集失败 / 跳过的集不在 episodes 里，等于没通过，但占请求名额。非增；最后一项 = 全程通过率。
    """
    return [sum(1 for e in episodes if (e.get("blocks_passed") or 0) >= k) / requested
            for k in range(1, blocks + 1)]


def spl_clearance(vehicle, task):
    """l_i 用的 A* 膨胀余量 = 机体半径 teacher.BODY_HALF_BY_VEHICLE[vehicle]（e1_mini 0.102 / x3 0.235）：
    机心离柱表面 ≥ 这个值就是碰撞模型下可行的路，所以 l_i 是物理可行最短路的下界（栅格 0.1 m 离散误差内），
    不是教师规划余量（CLEARANCE_BY_VEHICLE，E1 base 0.40）那条「教师参考路径」。task 不影响机体尺寸，参数留着
    是为了调用方把 (vehicle, task) 一起传、spl_reference 里两者并列好对照（见 teacher_clearance）。"""
    del task
    return teacher.BODY_HALF_BY_VEHICLE[vehicle]


def teacher_clearance(vehicle, task):
    """该 (载具, 任务) 的教师 A* 规划余量（CLEARANCE_BY_VEHICLE 第一项）；只写进 spl_reference 供对照，不参与 l_i。"""
    return teacher.CLEARANCE_BY_VEHICLE.get(
        (vehicle, task), teacher.CLEARANCE_BY_VEHICLE[(vehicle, "base")])[0]


def episode_shortest_path_m(args, meta, oracle, goal_x):
    """SPL 的 l_i：从轨迹首点到任务目标的教师 A* 路长；非 SPL_TASKS / 无轨迹 / 无路时 None。"""
    traj = trajectory_xy(oracle)
    if args.task not in SPL_TASKS or not traj:
        return None
    vehicle = getattr(args, "vehicle", None) or "x3"
    waypoint = meta.get("waypoint") if meta.get("stop_radius") is not None else None
    # 到达半径与 summarize_episode 判 success 用的同一个：meta.goal_radius 没给就是 REACH_RADIUS。
    radius = (REACH_RADIUS if meta.get("goal_radius") is None else float(meta["goal_radius"])) \
        if waypoint is not None else None
    return teacher_shortest_path_m(meta.get("pillars") or [], traj[0],
                                   goal_x=None if waypoint is not None else goal_x,
                                   waypoint=waypoint, clearance=spl_clearance(vehicle, args.task),
                                   radius=radius)


def run_episode(seed, args, workdir, seen_gids):
    """返回 (summary 或 None, 跳过原因)。None 表示该 seed 不在目标 split/已评测。"""
    world = os.path.join(workdir, f"scene_{seed}.sdf")
    meta_path = os.path.join(workdir, f"scene_{seed}.json")
    cmd = [sys.executable, SCENE_GEN, "--seed", str(seed), "--scene-id", str(seed),
           "--out", world, "--template", args.template, "--task", args.task,
           "--appearance", args.appearance] + cp.scene_gen_extra(args)
    if args.task == "l1":
        # v5 关系任务要求两根同色柱横跨出生点，默认 spawn 在 y=0 时
        # assign_landmarks 找不到柱对会直接抛错（采集也是 random）。
        cmd += ["--spawn-mode", "random"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"scene_gen seed={seed} 失败：{r.stderr.strip()}")
    video_patch = None
    if args.record_video:
        # 录像时只对这一集生成的 world 打追尾相机补丁（机位更近、HFOV 60°、分辨率随画质档）；
        # 模板文件不动，template_sha256 不变；补丁内容记进 eval.json 的 video_patch。
        import record_views
        video_patch = record_views.patch_chase_cam(world, args.video_quality)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gid = sm.geometry_gid(meta["pillars"])
    split = sm.split_for_scene(meta)
    if split != args.split:
        return None, f"split={split}"
    # 在起 sim 之前去重：key 必须唯一标识「几何 + 彩色柱位置/颜色」，
    # 否则同几何不同目标柱的合法任务会被当成重复，反事实对也会少一半。
    dedup = dedup_key(meta, args.task)
    if dedup in seen_gids:
        return None, f"已评测 {dedup}"

    eval_meta = override_instruction(meta, args.text_mode)
    eval_meta_path = meta_path
    if args.text_mode != "keep":
        eval_meta_path = os.path.join(workdir, f"scene_{seed}.text.json")
        with open(eval_meta_path, "w", encoding="utf-8") as f:
            json.dump(eval_meta, f, ensure_ascii=False)

    goal_x = max(float(p["x"]) for p in meta["pillars"]) + args.goal_margin
    # sim09.v2：route_choice 任务（没有 waypoint/stop_radius 的场景）让 collect.py 在首次 x ≥ goal_x
    # 结束本集；waypoint 任务（l1/campus）仍按 stop_radius 的 goal-stop 结束，collect 也拒绝两者同开。
    stop_at_goal = meta.get("stop_radius") is None and meta.get("waypoint") is None
    partition = f"uavrgb_eval_{os.getpid()}_{seed}"
    log_path = os.path.join(workdir, "logs", f"sim_{seed}.log")
    ep_dir = os.path.join(workdir, f"ep_{seed}")
    os.makedirs(ep_dir, exist_ok=True)
    sim = cp._start_sim(world, partition, log_path)
    t0 = time.monotonic()
    rec = None
    frames_dir = os.path.join(args.video_dir, f".frames_{seed}") if args.record_video else None
    try:
        cp.wait_ready(sim, partition, log_path, args.sim_timeout)
        time.sleep(cp.SIM_SETTLE_S)
        if frames_dir:
            # 录像与 collect 并行订阅相机；只读话题，不影响动作窗口（窗口按仿真时间计）。
            rec = subprocess.Popen([sys.executable, RECORD, "record", "--out", frames_dir],
                                   env=cp.gz_env(partition), stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
            cp._register(rec)
        cmd = [sys.executable, COLLECT, "--out", ep_dir,
               "--steps", str(args.steps), "--hold", str(args.hold),
               "--policy", args.policy,
               "--stop-on-collision",
               "--scene-id", str(seed), "--seed", str(seed),
               "--scene-meta", eval_meta_path, "--split", split,
               "--window-from", args.window_from, "--dynamics", args.dynamics]
        if stop_at_goal:
            cmd += ["--stop-at-goal"]
        if args.save_depth:
            cmd += ["--save-depth"]
        if args.policy == "student":
            cmd += ["--student-ckpt", args.ckpt]
            if args.infer_bin:
                cmd += ["--infer-bin", args.infer_bin]
            if args.heading_guard:
                cmd += ["--heading-guard", str(args.heading_guard)]   # 0 不传：旧命令行逐字不变
        elif args.policy == "rule-oracle":
            cmd += ["--rule-hfov-deg", str(args.rule_hfov_deg)]       # 只给 rule-oracle 传：其它策略的命令行逐字不变
            if args.rule_version != "v0":
                cmd += ["--rule-version", args.rule_version]          # v0 不传：v0 的命令行与 v0 实验逐字相同
        elif args.policy == "teacher-nn":
            cmd += ["--teacher-ckpt", args.teacher_ckpt]
        elif args.policy == "teacher-rl":
            cmd += ["--rl-checkpoint", args.rl_checkpoint]
        if args.policy == "teacher" and args.teacher_planner != "astar":
            cmd += ["--teacher-planner", args.teacher_planner]
        if args.policy == "teacher" and args.teacher_planner == "costmap":
            if args.costmap_tol != 25.0:
                cmd += ["--costmap-tol", str(args.costmap_tol)]
            if args.costmap_lookahead_steps != 1:
                cmd += ["--costmap-lookahead-steps", str(args.costmap_lookahead_steps)]
        if getattr(args, "vehicle", None) and args.vehicle != "x3":
            cmd += ["--vehicle", args.vehicle]   # x3 不传：旧命令行逐字不变
        if args.action_latency is not None:
            cmd += ["--action-latency", f"{args.action_latency:g}"]   # 没给不传：旧命令行逐字不变
        env = cp.gz_env(partition)
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        cp._register(proc)   # Ctrl-C/SIGTERM 时 stop_group 能兜底杀掉 collect
        try:
            out, err = proc.communicate(timeout=args.episode_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
            rc = "timeout"
        finally:
            cp.stop_group(proc, f"eval collect seed {seed}")
    finally:
        cp.stop_group(rec, f"eval recorder seed {seed}")
        cp.stop_group(sim, f"eval seed {seed}")

    actions_path = os.path.join(ep_dir, "actions.jsonl")
    oracle_path = os.path.join(ep_dir, "oracle.jsonl")
    actions = load_jsonl(actions_path) if os.path.isfile(actions_path) else []
    oracle = load_jsonl(oracle_path) if os.path.isfile(oracle_path) else []
    # 默认会删除临时采集目录，先将性能与请求后端带入持久报告。
    # 旧集/采集失败没有这些字段时保留 null，不推断实际 renderer。
    try:
        with open(os.path.join(ep_dir, "episode_manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        manifest = {}
    summary = summarize_episode(actions, oracle, goal_x, meta.get("waypoint"),
                                WAYPOINT_RADIUS, meta.get("stop_radius"), meta.get("goal_radius"),
                                timed_out=(rc == "timeout"),
                                coast_window_s=COAST_WINDOW_S.get(args.dynamics),
                                shortest_path_m=episode_shortest_path_m(args, meta, oracle, goal_x),
                                corridor_half_width=args.corridor_half_width)
    summary.update({
        "seed": seed, "geometry_gid": gid, "split": split, "policy": args.policy,
        "task": args.task, "window_from": args.window_from,
        "dynamics": args.dynamics,
        "vehicle": getattr(args, "vehicle", None) or "x3",
        "text_mode": args.text_mode,
        "task_version": meta.get("task_version"),
        **{key: meta.get(key) for key in sm.PROVENANCE_KEYS},
        # 场景只写 x3_v1；cf_stab 的真实档名在采集 manifest 里。
        "dynamics_profile": manifest.get("dynamics_profile") or meta.get("dynamics_profile"),
        "target_color": meta.get("target_color"),
        "instruction": eval_meta.get("instruction"),
        "waypoint": meta.get("waypoint"),
        "waypoint_radius": WAYPOINT_RADIUS,
        "stop_radius": meta.get("stop_radius"),
        "pillars": meta.get("pillar_count"), "goal_x": goal_x,
        # sim09.v2：collect 是否在目标处提前结束（manifest 字段：route_choice = 首次过成功线；
        # waypoint 任务 = goal-stop 连续 3 步 SAFE；采集失败/旧 collect 没有该字段为 False）。
        "stopped_on_goal": bool(manifest.get("stopped_on_goal", False)),
        "collect_rc": rc, "wall_s": round(time.monotonic() - t0, 1),
        "performance": manifest.get("performance"),
        "render_backend_requested": manifest.get("render_backend_requested"),
        "actions": {},
        "stderr_tail": err.strip().splitlines()[-5:],
        "dedup_key": dedup,
        # 地图标识与评测任务标识不是一回事：campus 每 8 个 seed 是同一张图，
        # dedup_key 还含颜色/左右，不能拿来数地图覆盖。
        "geometry_gid": sm.geometry_gid(meta["pillars"]),
    })
    if meta.get("course"):   # course_base：块边界线（每块最远柱 x + 1.0）随集记下，逐块存活统计直接读它
        summary["block_goal_x"] = meta["course"]["block_goal_x"]
        # W1-4：本集通过的块数与总块数；走廊半宽用评测自己的（--corridor-half-width，没开为 None 即不查横向），
        # 与这一集 success 的横向判据同口径。base 等没有 course 的任务不写这两个键。
        summary["blocks_total"] = len(meta["course"]["block_goal_x"])
        summary["blocks_passed"] = blocks_passed(oracle, actions, meta["course"]["block_goal_x"],
                                                 args.corridor_half_width)
    # W1-4：collisions_per_100m 的分母（所有任务）。私有键，write_report 汇总后从 per_episode 里去掉。
    summary[FLIGHT_PATH_KEY] = flight_path_m(oracle, actions, goal_x, meta.get("waypoint"),
                                             meta.get("stop_radius"), meta.get("goal_radius"),
                                             args.corridor_half_width)
    # W1-0：观测戳到派发的仿真秒（collect 每步都记 dispatch_lag_s；旧 collect 的集没有，不写键）。
    lags = [a["dispatch_lag_s"] for a in actions if a.get("dispatch_lag_s") is not None]
    if lags:
        summary["dispatch_lag_s_median"] = statistics.median(lags)
        summary["dispatch_lag_s_max"] = max(lags)
    if args.action_latency is not None:
        # 迟到步 = 动作就绪时仿真戳已超过 观测戳 + L + 0.02 s（collect 逐步记 late_dispatch）。数行而不是读 manifest：
        # 采集中途失败的集没有 manifest 的计数，行是权威。有迟到步的集在 scan_episodes 里算基础设施失败。
        late_ids = [a.get("step", i) for i, a in enumerate(actions) if a.get("late_dispatch")]
        summary["late_dispatch_steps"] = len(late_ids)
        if late_ids:
            summary["late_step_ids"] = late_ids
    for a in actions:
        k = int(a.get("action_executed", -1))
        summary["actions"][k] = summary["actions"].get(k, 0) + 1
    if getattr(args, "episodes_dir", None):
        keep_dir = os.path.join(args.episodes_dir, f"seed_{seed}")
        keep_episode_logs(ep_dir, keep_dir)
        summary["episode_dir"] = os.path.relpath(keep_dir, os.path.dirname(args.episodes_dir))
    if frames_dir:
        _write_video(summary, frames_dir, ep_dir, world, args, seed, goal_x, meta)
        summary["video_patch"] = video_patch
    return summary, None


def scan_episodes(seed_list, episodes, run_fn, log=print, state=None):
    """固定请求集扫描（AUDIT #11）：成功与采集失败都占一个名额，不再为失败补扫新 seed。

    之前按 len(results) 停：1 集失败就多扫 1 个 seed 补上，requested_SR 的分母名义上是
    请求数、实际成功集却来自更大的集合（4 请求 1 失败 → SR 1.0 / execution 1.0 /
    completion 0.75 自相矛盾）。现在按 len(attempted) 停：失败进分母、单列 failed。
    跳过（split 不符 / 重复几何）不占名额。run_fn(seed, seen) -> (summary, skip_reason)。
    返回 (results, failures, skips, scanned, attempted)。
    state（可选 dict）：开扫前就挂上这五个列表和正在跑的 current seed，SIGTERM/Ctrl-C
    （SystemExit/KeyboardInterrupt 不在这里捕获）打断后调用方仍拿得到已完成的部分（A2）。
    """
    results, seen, failures, skips, scanned, attempted = [], set(), [], [], [], []
    if state is not None:
        state.update(results=results, failures=failures, skips=skips, scanned=scanned,
                     attempted=attempted, current=None)
    for seed in seed_list:
        if len(attempted) >= episodes:
            break
        scanned.append(seed)
        if state is not None:
            state["current"] = seed
        try:
            summary, skip = run_fn(seed, seen)
        except Exception as e:  # noqa: BLE001 - 单集失败要记录，不能拖垮整轮
            log(f"[seed {seed}] 失败: {type(e).__name__}: {e}", flush=True)
            attempted.append(seed)
            failures.append({"seed": seed, "error": f"{type(e).__name__}: {e}"})
            continue
        if skip:
            log(f"[seed {seed}] 跳过: {skip}", flush=True)
            skips.append({"seed": seed, "reason": skip})
            continue
        attempted.append(seed)
        late = summary.get("late_dispatch_steps") or 0   # 只有给了 --action-latency 才有这个键
        if summary["collect_rc"] != 0 or summary["steps"] == 0:
            # 采集中途失败不是「学生飞得差」，不计进 SR 分子，但占请求名额。
            failures.append({"seed": seed, "rc": summary["collect_rc"],
                             "stderr_tail": summary.get("stderr_tail"),
                             "performance": summary.get("performance"),
                             "render_backend_requested": summary.get("render_backend_requested")})
            log(f"[seed {seed}] 采集失败 rc={summary['collect_rc']}，占请求名额、不计入成功",
                flush=True)
            continue
        if late > 0:
            # 迟到步 = 派发时刻偏离了固定的 观测戳 + L：这一集的动作生效时长不可比。与采集失败同样处理：
            # 基础设施失败，占请求名额、不计入成功，原因与迟到明细随 failed[] 留证。
            failures.append({"seed": seed, "rc": summary["collect_rc"],
                             "error": f"late_dispatch: {late} 步迟到（动作就绪时已超过 观测戳 + L + 0.02 s），"
                                      "派发时刻不再固定，本集不计成功",
                             "late_dispatch_steps": late, "late_step_ids": summary.get("late_step_ids", []),
                             "dispatch_lag_s_median": summary.get("dispatch_lag_s_median"),
                             "dispatch_lag_s_max": summary.get("dispatch_lag_s_max"),
                             "stderr_tail": summary.get("stderr_tail"),
                             "performance": summary.get("performance"),
                             "render_backend_requested": summary.get("render_backend_requested")})
            log(f"[seed {seed}] 迟到派发 {late} 步（步号 {summary.get('late_step_ids')}），基础设施失败，"
                "占请求名额、不计入成功", flush=True)
            continue
        seen.add(summary["dedup_key"])
        results.append(summary)
        log(f"[seed {seed}] steps={summary['steps']} "
            f"collision={summary['collision']} success={summary['success']} "
            f"max_x={summary['max_x']} rc={summary['collect_rc']}", flush=True)
    if state is not None:
        state["current"] = None
    return results, failures, skips, scanned, attempted


def interruption(exc, state):
    """把打断扫描的 SystemExit(143)/KeyboardInterrupt/异常翻译成 eval.json 的 interrupted 记录 + 退出码。

    正在跑的 seed 记进 failed（error=interrupted: …）并占请求名额：它被尝试过但没有结果。
    """
    if isinstance(exc, SystemExit) and exc.code == 143:
        reason, rc = "SIGTERM", 143
    elif isinstance(exc, KeyboardInterrupt):
        reason, rc = "KeyboardInterrupt", 130
    elif isinstance(exc, SystemExit):
        reason, rc = f"SystemExit({exc.code})", exc.code if isinstance(exc.code, int) else 1
    else:
        reason, rc = f"{type(exc).__name__}: {exc}", 1
    seed = state.get("current")
    if seed is not None:
        state["failures"].append({"seed": seed, "error": f"interrupted: {reason}"})
        state["attempted"].append(seed)
    return {"seed": seed, "reason": reason, "rc": rc}


def keep_episode_logs(ep_dir, keep_dir):
    """正常结束也保留每集的逐步日志（actions/oracle/manifest；帧不留）到 <out>/episodes/seed_<seed>/，
    SPL 等指标事后可复算（EVAL_STANDARD §5.3「历史 eval.json 不能补算 SPL」）。"""
    os.makedirs(keep_dir, exist_ok=True)
    for name in ("actions.jsonl", "oracle.jsonl", "episode_manifest.json"):
        src = os.path.join(ep_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(keep_dir, name))


def record_views_quality(name):
    """录像画质档的生效名（环境变量可覆盖），写进 eval.json；不录像的评测不导入 record_views。"""
    import record_views
    return record_views.vl.quality(name)["name"]


def _write_video(summary, frames_dir, ep_dir, world, args, seed, goal_x, meta):
    """三视图视频 -> <out>/videos/seed_<seed>.mp4。失败只记 video_error，不影响评测结论。"""
    import record_views
    result = ("SUCCESS" if summary.get("success") else
              "COLLISION" if summary.get("collision") else
              "STUCK" if summary.get("stuck") else "NOT REACHED")
    if args.task in ("campus", "campus_assets"):
        criterion, goal_radius = "visible_goal_reach_v1", summary.get("goal_radius_stop")
    elif meta.get("stop_radius") is not None:
        criterion, goal_radius = "goal_reach_ever", summary.get("goal_radius_reach")
    else:
        criterion, goal_radius = "route_choice", None
    path = os.path.join(args.video_dir, f"seed_{seed}.mp4")
    try:
        n = record_views.compose(frames_dir, ep_dir, world, path,
                                 title=f"{args.policy} | {args.dynamics} | seed {seed}",
                                 result=result, goal_x=goal_x, waypoint=meta.get("waypoint"),
                                 success_criterion=criterion, goal_radius=goal_radius,
                                 quality=args.video_quality,
                                 log=lambda m: print(f"[seed {seed}] {m}", flush=True))
        summary["video"] = os.path.relpath(path, os.path.dirname(args.video_dir))
        summary["video_frames"] = n
        summary["video_quality"] = record_views_quality(args.video_quality)
    except Exception as e:  # noqa: BLE001 - 视频是诊断产物，不能让评测失败
        summary["video_error"] = f"{type(e).__name__}: {e}"
    if not args.keep_work:
        shutil.rmtree(frames_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="学生 checkpoint（--policy student 必填）")
    p.add_argument("--policy", choices=("student", "forward", "teacher-nn", "teacher", "teacher-rl", "rule-oracle"),
                   default="student",
                   help="student=模型闭环；forward=盲前进基线（T7）；"
                        "teacher-nn=特权教师网络闭环（可解性闸门）；"
                        "teacher=柱子真值 A* 专家闭环（新动力学下的可解性上界）；"
                        "rule-oracle=柱子真值列距离 + 固定规则 v0（零训练的规则层上限，不需要 --ckpt）")
    p.add_argument("--teacher-planner", choices=("astar", "predictive", "costmap"), default="astar",
                   help="--policy teacher 的规划器：costmap=测地场代价 argmin（teacher.plan_action_costmap）")
    p.add_argument("--costmap-tol", type=float, default=25.0,
                   help="透传给 collect.py：costmap 教师航向罚死区（度）；仅 --teacher-planner costmap")
    p.add_argument("--costmap-lookahead-steps", type=int, default=1,
                   help="透传给 collect.py：costmap 教师前瞻步数；仅 --teacher-planner costmap")
    p.add_argument("--rl-checkpoint", default="")
    p.add_argument("--teacher-ckpt", default=None,
                   help="PrivilegedTeacher checkpoint（--policy teacher-nn 必填）")
    p.add_argument("--episodes", type=int, default=None,
                   help="请求评测的 episode 数 = SR 分母（失败集也占名额并计入分母，列在 failed；"
                        "缺省：给了 --seeds 用其个数，否则 3）")
    p.add_argument("--steps", type=int, default=80, help="每集最大步数")
    p.add_argument("--hold", type=float, default=1.0, help="每动作仿真时间秒")
    # default=None 只为识别「用户显式给了 dispatch」；缺省解析成 observation。
    p.add_argument("--window-from", choices=("dispatch", "observation"),
                   default=None,
                   help="动作窗口起点：observation=观测位姿（默认，唯一被采纳基线）；"
                        "dispatch=派发位姿（仅复现 T7-R，数字不可与 observation 混报）")
    p.add_argument("--dynamics", choices=DYNAMICS_PROFILES, default="x3_velocity",
                   help="透传给 collect.py：x3_velocity=旧行为（唯一历史基线）；"
                        "cf_stab=CF-Drone 摇杆模型（建议配 --hold 0.1~0.2）；"
                        "cf_blackbox=纯 Python 黑盒（3DGS collector 直接使用；Gazebo "
                        "闭环会明确拒绝，不拿 x3 物理冒充）；"
                        "cf_sitl=E1 MINI 固件在环（必须 --vehicle e1_mini；false_forward 按滑行窗口"
                        f"{COAST_WINDOW_S['cf_sitl']:g}s 判，metric_version 带 .coast 后缀）")
    p.add_argument("--vehicle", choices=("x3", "e1_mini"), default=None,
                   help="世界载具：缺省 x3（旧行为）；e1_mini 选 E1 模板并透传给 collect.py")
    p.add_argument("--camera-rate", type=float, default=None,
                   help="透传给 scene_gen：学生 RGB/深度相机 update_rate（Hz）；缺省用模板值")
    p.add_argument("--no-aux-cameras", action="store_true",
                   help="透传给 scene_gen：不生成 chase/top 辅助相机（仅 E1 模板；--record-video 时不要加）")
    p.add_argument("--protocol", default=None,
                   help="可选协议快照 JSON；比对 window_from/hold_s/steps/task/"
                        "appearance_version/dynamics/camera_profile，不一致写 ood=true 并警告")
    p.add_argument("--strict-protocol", action="store_true",
                   help="协议与本次参数不一致时直接失败（默认只警告，保证旧 run 可复现）")
    p.add_argument("--text-mode", choices=TEXT_MODES, default="keep",
                   help="L2 消融：keep=原指令；blank=空文本；shuffle=把 left/right 反过来")
    p.add_argument("--task", choices=("base", "l1", "campus", "campus_assets", "forest_e1", "course_base"), default="base",
                   help="base=原柱阵；l1=彩色 landmark + instruction/waypoint；"
                        "forest_e1=E1 随机圆柱森林（需 --density 与 --vehicle e1_mini）；"
                        "course_base=K 个 base 柱阵块首尾相接的长课程（需 --blocks；成功线 = 6.0+5.0·(K−1)）")
    p.add_argument("--density", choices=("sparse", "medium", "dense"), default=None,
                   help="透传给 scene_gen：forest_e1 密度档（最窄缝 0.45/0.35/0.28 m）；forest_e1 必填")
    p.add_argument("--blocks", type=int, default=None,
                   help="透传给 scene_gen：course_base 块数 K（2–6）；course_base 必填，其它任务不接受")
    p.add_argument("--appearance", choices=("legacy", "v1"), default="v1",
                   help="透传给 scene_gen：v1 删固定地标并随机化光照/天空（几何不变）；旧数据复现请显式 legacy")
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=500,
                   help="最多扫描多少个 seed 来凑够 --episodes（test split 约 10%%）")
    p.add_argument("--seeds", default="",
                   help="固定 seed 列表（逗号分隔或 a:b）；T7 三个基线用同一批场景")
    p.add_argument("--out", required=True)
    p.add_argument("--template", default=None,
                   help="模板 SDF（缺省按 --vehicle 选；x3 为 sim/worlds/uav_rgb.sdf）")
    p.add_argument("--infer-bin", default=None)
    p.add_argument("--goal-margin", type=float, default=GOAL_MARGIN_M,
                   help="成功 = 无碰撞且 max_x >= 最远柱 x + 该余量。sim09.v2 起 collect.py --stop-at-goal "
                        f"在同一条线上结束本集，余量只接受 collect.GOAL_MARGIN_M={GOAL_MARGIN_M:g}（改它会让"
                        "停线与评测线不一致）")
    p.add_argument("--corridor-half-width", type=float, default=CORRIDOR_HALF_WIDTH_DEFAULT,
                   help="route_choice 任务的横向边界（m）：oracle pose_gt 全程 |y| 超过它就判本集不成功。"
                        "默认关（旧口径逐位不变）；建议 3.0（scene_gen.Y_RANGE 的走廊边界）。"
                        "开了以后 metric_version 带 .corridor<N>m，新旧 SR 不可混报")
    p.add_argument("--heading-guard", type=int, default=0,
                   help="透传给 collect.py（仅 --policy student）：已执行的左转数减右转数的绝对值达到 N 就朝回转一步；"
                        "0=关闭。它改的是策略不是判据，eval.json 顶层记 heading_guard，不动 metric_version")
    p.add_argument("--action-latency", type=float, default=None, metavar="L",
                   help="透传给 collect.py：把动作派发时刻固定到仿真时间（观测戳 + L 秒），动作生效时长固定为 hold - L，"
                        "迟到步（就绪时已超过 L + 0.02 s）记账。含迟到步的集算基础设施失败（eval.json failed[] 与 "
                        "late_dispatch_episodes 单列），不计入成功。默认关（旧口径逐位不变）；开了以后 eval.json 顶层记 "
                        "action_latency_s，新旧 SR 不可混报。只能配 --window-from observation，且 0 < L < --hold")
    p.add_argument("--rtf", type=float, default=None,
                   help="透传给 scene_gen：世界物理的目标实时因子（只能压低，缺省用模板值 1.0）。eval.json 顶层记 rtf")
    p.add_argument("--rule-version", choices=("v0", "v1"), default="v0",
                   help="透传给 collect.py（仅 --policy rule-oracle）：v0=固定规则（默认，命令行与输出逐位不变）；"
                        "v1=短期障碍记忆 + 候选航向选择（计划 D6）。v1 时 eval.json 顶层另记 rule_version")
    p.add_argument("--rule-hfov-deg", type=float, default=60.0,
                   help="透传给 collect.py（仅 --policy rule-oracle）：列距离视场（度），60=学生相机（20 列），"
                        "81=真机 GC4653 的水平视场（27 列）；范围 [6, 180]。eval.json 顶层记 rule_hfov_deg 与 rule_params")
    p.add_argument("--sim-timeout", type=float, default=120.0)
    p.add_argument("--episode-timeout", type=float, default=600.0)
    p.add_argument("--keep-work", action="store_true", help="保留 world/每集原始文件")
    p.add_argument("--record-video", action="store_true",
                   help="每集录 机载/追尾/顶视 三视图视频到 <out>/videos/seed_<seed>.mp4"
                        "（原始帧约 10 MB/仿真秒，暂存 <out>/videos/.frames_*，合成后删除；"
                        "本集 world 的追尾相机会被打补丁：机位更近、HFOV 60°，模板文件不动）")
    p.add_argument("--video-quality", default=None,
                   help="录像画质档 low/medium/high（sim/tools/view_layout.VIDEO_QUALITY；"
                        "默认 medium，环境变量 UAVRGB_VIDEO_QUALITY 覆盖）")
    p.add_argument("--save-depth", action="store_true",
                   help="保留每步 16 位深度 PNG（诊断闭环失败用；配合 --keep-work）")
    args = p.parse_args()
    if args.window_from is None:
        args.window_from = "dispatch" if args.task in ("campus", "campus_assets") else "observation"
    elif args.window_from == "dispatch" and args.task not in ("campus", "campus_assets"):
        print("警告: --window-from dispatch 不是被采纳的基线语义（唯一基线是 "
              "observation）；本次 SR/碰撞率不可与 observation 结果混报。",
              file=sys.stderr, flush=True)

    actual_protocol = {
        "window_from": args.window_from, "hold_s": args.hold, "steps": args.steps,
        "task": args.task,
        "appearance_version": appearance_version_for(args.task, args.appearance),
        "dynamics": args.dynamics, "camera_profile": CAMERA_PROFILE,
        "action_latency_s": args.action_latency,   # None = 旧口径；协议快照写了它而本次没给（或值不同）= OOD
    }
    protocol, mismatches = None, []
    if args.protocol:
        try:
            protocol = load_protocol(args.protocol)
        except (OSError, ValueError) as e:
            p.error(f"--protocol 无法读取: {e}")
        mismatches = protocol_mismatches(protocol, actual_protocol)
        if mismatches:
            detail = "; ".join(f"{k}: protocol={v!r} actual={g!r}"
                               for k, v, g in mismatches)
            if args.strict_protocol:
                p.error(f"protocol 与本次参数不一致: {detail}")
            print(f"警告: protocol 不一致（本次结果 ood=true）: {detail}",
                  file=sys.stderr, flush=True)
    ood = bool(mismatches)

    try:
        seed_list = parse_seeds(args.seeds) if args.seeds else []
    except ValueError as e:
        p.error(str(e))
    if args.seeds and not seed_list:
        p.error("--seeds 没有有效 seed")
    args.episodes = episode_target(args.episodes, seed_list)
    if args.episodes < 1 or args.steps < 1:
        p.error("episodes/steps 必须 >=1")
    if not seed_list and args.max_seeds < args.episodes:
        p.error("max-seeds >= episodes")
    if not (math.isfinite(args.hold) and args.hold > 0):
        p.error("--hold 必须是有限正数")
    if args.action_latency is not None:
        if not (math.isfinite(args.action_latency) and args.action_latency > 0):
            p.error("--action-latency 必须是有限正数")
        if args.action_latency >= args.hold:
            p.error(f"--action-latency 必须小于 --hold（动作生效时长 = hold - L；L={args.action_latency:g} hold={args.hold:g}）")
        if args.window_from != "observation":
            p.error("--action-latency 只能配 --window-from observation（执行器截止 = 观测戳 + hold）")
    if args.rtf is not None and not (math.isfinite(args.rtf) and args.rtf > 0):
        p.error("--rtf 必须是有限正数")
    if args.heading_guard < 0 or (args.heading_guard and args.policy != "student"):
        p.error("--heading-guard 必须是非负整数，且只能和 --policy student 一起用")
    if args.policy != "rule-oracle" and args.rule_hfov_deg != 60.0:
        p.error("--rule-hfov-deg 只能和 --policy rule-oracle 一起用")
    if args.policy != "rule-oracle" and args.rule_version != "v0":
        p.error("--rule-version 只能和 --policy rule-oracle 一起用")
    if args.policy == "rule-oracle" and not (math.isfinite(args.rule_hfov_deg)
                                             and 2 * rule_policy.COL_DEG <= args.rule_hfov_deg <= 180.0):
        p.error(f"--rule-hfov-deg 必须在 [{2 * rule_policy.COL_DEG:g}, 180] 度内")
    if args.dynamics == "cf_blackbox":
        p.error("cf_blackbox 是纯 Python 模型档，Gazebo 闭环未接入；"
                "请让 3DGS collector 直接调用 cf_blackbox，或改用 cf_stab/x3_velocity")
    cp.resolve_vehicle(args, p.error)
    if (args.task == "forest_e1") != (args.density is not None):
        p.error("--density 与 --task forest_e1 必须同时给")
    if args.task == "forest_e1" and args.vehicle != "e1_mini":
        p.error("--task forest_e1 需要 --vehicle e1_mini（--dynamics cf_sitl）")
    if (args.task == "course_base") != (args.blocks is not None):
        p.error("--blocks 与 --task course_base 必须同时给")
    if args.blocks is not None and not 2 <= args.blocks <= 6:   # 同 scene_course.BLOCKS_RANGE（这里不为一个常量导入它）
        p.error("--blocks 必须在 2–6")
    if args.record_video and args.no_aux_cameras:
        p.error("--record-video 需要 chase/top 相机，不能同时 --no-aux-cameras")
    for name in ("goal_margin", "sim_timeout", "episode_timeout"):
        value = getattr(args, name)
        if not (math.isfinite(value) and value > 0):
            p.error(f"--{name.replace('_', '-')} 必须是有限正数")
    if abs(args.goal_margin - GOAL_MARGIN_M) > 1e-9:
        p.error(f"--goal-margin 只接受 {GOAL_MARGIN_M:g}：collect.py --stop-at-goal 的停线是最远柱 x + "
                f"collect.GOAL_MARGIN_M，评测线必须与它重合（sim09.v2）")
    if args.policy == "teacher-nn":
        if not args.teacher_ckpt:
            p.error("--policy teacher-nn 需要 --teacher-ckpt")
        if not os.path.isfile(args.teacher_ckpt):
            p.error(f"teacher checkpoint 不存在: {args.teacher_ckpt}")
        # collect.py 由 run_episode 用 sys.executable 起，所以这里探的就是子进程解释器。
        # 不能只看 find_spec：本机 conda 的 torch 装了但缺 libcudnn，import 会失败。
        try:
            torch_ok = subprocess.run([sys.executable, "-c", "import torch"],
                                      capture_output=True,
                                      timeout=120).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            torch_ok = False
        if not torch_ok:
            p.error("teacher-nn 需要带 torch 的解释器，例如 "
                    "$DATA_ROOT/uav_rgb/teacher-venv/bin/python")
    if (args.teacher_planner == "predictive" or args.policy == "teacher-rl") and (
            args.dynamics != "cf_stab" or args.task not in ("campus", "campus_assets")
            or abs(args.hold-0.2)>1e-6):
        p.error("新 Teacher 需要 --dynamics cf_stab --hold 0.2 --task campus_assets/campus")
    if args.policy == "teacher-rl" and not args.rl_checkpoint:
        p.error("teacher-rl 需要 --rl-checkpoint")
    if args.teacher_planner == "costmap" and args.policy != "teacher":
        p.error("--teacher-planner costmap 只能和 --policy teacher 一起用")
    if args.teacher_planner != "costmap" and (args.costmap_tol != 25.0 or args.costmap_lookahead_steps != 1):
        p.error("--costmap-tol/--costmap-lookahead-steps 只能和 --teacher-planner costmap 一起用")
    if args.text_mode != "keep" and args.policy != "student":
        p.error("--text-mode 只对 --policy student 有意义")
    infer_bin = None
    text_dim = None
    if args.policy == "student":
        if not args.ckpt:
            p.error("--policy student 需要 --ckpt")
        if not os.path.isfile(args.ckpt):
            p.error(f"checkpoint 不存在: {args.ckpt}")
        infer_bin = os.path.abspath(args.infer_bin or DEFAULT_INFER_BIN)
        if not os.path.isfile(infer_bin):
            p.error(f"推理程序不存在: {infer_bin}")
        # 无文本分支的 ckpt 上 blank/shuffle 与 keep 完全等价：不先探一次，
        # 会把「消融没接上」读成「语言没有增益」。
        import student as stu
        try:
            probe = stu.StudentPolicy(args.ckpt, infer_bin)
        except RuntimeError as e:
            p.error(str(e))
        try:
            text_dim = probe.text_dim
        finally:
            probe.close()
        if args.text_mode != "keep" and text_dim <= 0:
            p.error(f"ckpt text_dim={text_dim}，--text-mode {args.text_mode} 不会改变推理")
        if args.text_mode != "keep" and args.task not in ("l1", "campus", "campus_assets"):
            p.error("--text-mode 只对 --task l1/campus 有意义")
        args.infer_bin = infer_bin  # 让 run_episode 与 report 用同一个路径
    out = os.path.abspath(args.out)
    if os.path.isdir(out) and os.listdir(out):
        p.error(f"--out {out} 非空；请用新目录")
    os.makedirs(out, exist_ok=True)
    args.video_dir = os.path.join(out, "videos")
    args.episodes_dir = os.path.join(out, "episodes")   # 每集 actions/oracle/manifest 的保留处（帧不留）
    if args.record_video:
        os.makedirs(args.video_dir, exist_ok=True)
    try:
        os.makedirs(cp.XDG_RUNTIME_DIR, mode=0o700, exist_ok=True)
        os.chmod(cp.XDG_RUNTIME_DIR, 0o700)
    except OSError as e:
        print(f"警告: 无法准备 XDG_RUNTIME_DIR: {e}", flush=True)

    workdir = (os.path.join(out, ".work") if args.keep_work
               else tempfile.mkdtemp(prefix="uavrgb_eval_"))
    os.makedirs(os.path.join(workdir, "logs"), exist_ok=True)

    def _term(_sig, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _term)
    if not seed_list:
        seed_list = list(range(args.base_seed, args.base_seed + args.max_seeds))
    started = time.monotonic()
    # A2：SIGTERM(SystemExit 143)/Ctrl-C/意外异常都先写出部分 eval.json（partial=true）再退出；
    # 工作目录无论怎么结束都删（--keep-work 除外）。第二次信号会打断写报告——那就没有 eval.json。
    state = dict(results=[], failures=[], skips=[], scanned=[], attempted=[], current=None)
    interrupted = None
    try:
        try:
            scan_episodes(seed_list, args.episodes,
                          lambda seed, seen: run_episode(seed, args, workdir, seen), state=state)
        except BaseException as e:  # noqa: BLE001 - 见上
            interrupted = interruption(e, state)
            print(f"中断：{interrupted['reason']}（seed {interrupted['seed']}）；"
                  f"写出部分结果 eval.json partial=true", file=sys.stderr, flush=True)
        finally:
            cp.cleanup_all()
        rc = write_report(args, out, state, interrupted, protocol, mismatches, ood, text_dim, infer_bin, started)
    finally:
        if args.keep_work:
            print(f"原始文件保留在 {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    return rc


def report_metric_version(args):
    """eval.json 顶层的 metric_version：与 summarize_episode 同一套后缀规则（coast / corridor）。"""
    v = (METRIC_VERSION if args.dynamics not in COAST_WINDOW_S
         else f"{METRIC_VERSION}.coast{COAST_WINDOW_S[args.dynamics]:g}s")
    if args.corridor_half_width is not None:
        v += f".corridor{args.corridor_half_width:g}m"
    return v


def write_report(args, out, state, interrupted, protocol, mismatches, ood, text_dim, infer_bin, started):
    """汇总 + 写 <out>/eval.json + 打印；返回进程退出码（中断时是 143/130/1）。"""
    results, failures, skips = state["results"], state["failures"], state["skips"]
    scanned_seeds, attempted_seeds = state["scanned"], state["attempted"]
    agg = aggregate(results, requested=args.episodes,
                    attempted=len(attempted_seeds),
                    failed_count=len(failures), skipped_count=len(skips))
    stop_radius_used = next((e.get("stop_radius") for e in results
                             if e.get("stop_radius") is not None), None)
    report = {"schema": "uavrgb.eval.v1", "policy": args.policy,
              "teacher_planner": args.teacher_planner,
              "rl_checkpoint": args.rl_checkpoint if args.policy == "teacher-rl" else None,
              "task": args.task, "window_from": args.window_from,
              "dynamics": args.dynamics,
              "vehicle": args.vehicle or "x3",
              "camera_rate_hz": args.camera_rate,   # None = 模板默认（X3 30 Hz / E1 10 Hz）
              "aux_cameras": not args.no_aux_cameras,
              "text_mode": args.text_mode, "text_dim": text_dim,
              # 新指标合同（SIM-09）；旧 success/stop_success 定义不变。
              # cf_sitl 带 .coast 后缀：false_forward 按滑行窗口判，数字不与旧档混报。
              "metric_version": report_metric_version(args),
              "coast_window_s": COAST_WINDOW_S.get(args.dynamics),
              # 成功语义跟着数据走：L1 是 goal_reach_ever「轨迹上曾进入 waypoint
              # 1.5 m 且无碰撞」，旧数据是 route_choice「越过终点」。不写死会让新旧
              # SR 被直接混报。历史口径：2026-11-01 前是 goal_stop（含 SAFE 判定），
              # 2026-11-01~11-02 是 goal_reach（**末位姿**在 1.5 m 内），三者都不可比。
              "success_criterion": ("visible_goal_reach_v1" if args.task in ("campus", "campus_assets") else
                                    "goal_reach_ever" if stop_radius_used is not None
                                    else "route_choice"),
              "appearance_version": appearance_version_for(args.task, args.appearance),
              "goal_radius": (0.5 if args.task in ("campus", "campus_assets") else
                              REACH_RADIUS if stop_radius_used is not None else None),
              # L1 训练目标是 0.5 m 停、评测按 1.5 m 到达：两个半径同时写。
              "goal_radius_reach": REACH_RADIUS,
              "goal_radius_stop": stop_radius_used,
              # SPL 口径（EVAL_STANDARD §5.3）：l_i 的来源与膨胀余量写死在报告里，免得跨载具混报。
              "spl_reference": (None if args.task not in SPL_TASKS else {
                  "l_i": "teacher.astar_path + LOS shortcut, truncated at goal region", "step_m": 0.1,
                  # 2026-11-09 起 l_i 的膨胀 = 机体半径（物理可行下界）；之前是教师规划余量（E1 0.40），
                  # 两版 l_i 不同、SPL 不可直接比：对照时先看这两个字段。
                  "clearance_m": spl_clearance(args.vehicle or "x3", args.task),
                  "clearance_basis": "body half-width teacher.BODY_HALF_BY_VEHICLE (physically feasible lower bound)",
                  "teacher_clearance_m": teacher_clearance(args.vehicle or "x3", args.task),
                  # SPL_TASKS 里只有 l1 走 waypoint 分支，到达半径就是 REACH_RADIUS（campus 不算 SPL）。
                  "goal": (f"disk r={REACH_RADIUS:g} around waypoint" if stop_radius_used is not None
                           else f"line x=max_pillar_x+{args.goal_margin:g} (y step {SPL_GOAL_Y_STEP:g})"),
                  "p_i": "oracle pose_gt xy polyline, first pose -> first entry into goal region"}),
              "template_sha256": sha256_file(args.template),
              # 录像档位；每集 world 的追尾相机补丁在 episodes[i].video_patch（模板 sha 不受影响）
              "video_quality": (record_views_quality(args.video_quality) if args.record_video else None),
              # requested_seeds 只含真正尝试过的 seed；扫描过的另存，避免把未处理
              # 的扫描尾巴当成已请求集。
              "requested_seeds": list(attempted_seeds),
              "scanned_seeds": scanned_seeds,
              "attempted_seeds": list(attempted_seeds),
              "protocol": protocol,
              "protocol_mismatches": [{"key": k, "protocol": v, "actual": g}
                                      for k, v, g in mismatches],
              "ood": ood,
              "strict_protocol": args.strict_protocol,
              "goal_margin": args.goal_margin,
              "corridor_half_width": args.corridor_half_width,
              **({"heading_guard": args.heading_guard} if args.heading_guard else {}),
              # W1-0：动作派发固定到仿真时间。只在给了开关时写键（默认 eval.json 逐位不变）：action_latency_s 是口径
              # （新旧 SR 不可混报）；含迟到步的集不在 per_episode，单列在 late_dispatch_episodes，明细同在 failed[]。
              **({"action_latency_s": args.action_latency,
                  "late_dispatch_episodes": [
                      {k: f.get(k) for k in ("seed", "late_dispatch_steps", "late_step_ids",
                                             "dispatch_lag_s_median", "dispatch_lag_s_max")}
                      for f in failures if f.get("late_dispatch_steps")]}
                 if args.action_latency is not None else {}),
              **({"rtf": args.rtf} if args.rtf is not None else {}),
              # rule-oracle 的视场与规则参数：预注册的参数随结果留证（其它策略不写这两个键）
              **({"rule_hfov_deg": args.rule_hfov_deg,
                  "rule_params": rule_policy.RULE_V0 if args.rule_version == "v0" else rule_policy.RULE_V1,
                  **({"rule_version": args.rule_version} if args.rule_version != "v0" else {})}
                 if args.policy == "rule-oracle" else {}),
              **({"blocks": args.blocks} if args.blocks is not None else {}),   # course_base 的块数；其它任务不写键，旧输出逐位不变
              "waypoint_radius": WAYPOINT_RADIUS,
              "stop_radius": stop_radius_used,
              "geometry_version": cp.task_geometry_version(args.task),
              "density": args.density,   # forest_e1 的密度档；其它任务 None
              "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
              "ckpt_sha256": sha256_file(args.ckpt) if args.ckpt else None,
              "infer_bin": infer_bin,
              "infer_bin_sha256": (sha256_file(infer_bin)
                                   if args.policy == "student" else None),
              "split": args.split, "episodes_requested": args.episodes,
              "steps": args.steps, "hold_s": args.hold,
              "seeds": [e["seed"] for e in results],
              # 集数不等于独立地图数：campus 每 8 个 seed 是同一张图，扫出来的
              # 20 集可能只覆盖 3 张图（2026-11-03 审计）。不写出来会被当 20 个样本。
              "unique_maps": len({e["geometry_gid"] for e in results}),
              "wall_s": round(time.monotonic() - started, 1),
              # A2：中断（SIGTERM/Ctrl-C/异常）写出的部分结果；partial=true 的 eval.json 不得当完整结果报。
              "partial": interrupted is not None,
              "interrupted": (None if interrupted is None else
                              {"seed": interrupted["seed"], "reason": interrupted["reason"]}),
              **agg,
              # W1-4：每百米接触碰撞数（所有任务；路程只量到首次碰撞 / 越界 / 过线，见 flight_path_m）。
              # block_survival 只在 course_base 写（args.blocks 非空），分母 = 请求集数，与 requested_success_rate 同口径。
              "collisions_per_100m": collisions_per_100m(results),
              **({"block_survival": block_survival(results, args.episodes, args.blocks)}
                 if args.blocks is not None else {}),
              "failed": failures, "skipped": skips,
              "per_episode": [{k: v for k, v in e.items() if k not in ("infer_ms", FLIGHT_PATH_KEY)}
                              for e in results]}
    path = os.path.join(out, "eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    successes = sum(1 for e in results if e.get("success"))
    # 首行先报 requested（分母 = 本次请求集数），再报 done-set（分母 = 完成集数）。
    print(f"requested: SR={agg['requested_success_rate']} "
          f"({successes}/{args.episodes}); "
          f"completion={agg['completion_rate']} "
          f"({agg['episodes_done']}/{len(attempted_seeds)} attempted); "
          f"execution={agg['execution_rate']} "
          f"({agg['episodes_done']}/{args.episodes} requested); "
          f"failed={agg['failed_count']} skipped={agg['skipped_count']} "
          f"ood={ood}")
    print(f"done-set: SR={agg['success_rate']} ({successes}/{agg['episodes_done']}); "
          f"SR_stop={agg['stop_success_rate']}  collision={agg['collision_rate']}  "
          f"stuck={agg['stuck_rate']}  "
          f"false_forward={agg['false_forward_rate']}")
    print(f"new ({METRIC_VERSION}): reach_ever={agg['reach_ever_rate']} "
          f"final_in_radius={agg['final_in_radius_rate']} "
          f"stop_action={agg['stop_action_issued_rate']} "
          f"task_success={agg['task_success_rate']}  "
          f"radii reach={REACH_RADIUS}m stop={stop_radius_used}m  "
          f"spl={agg['spl']} (n={agg['spl_n']})")
    print(f"wilson95: success={agg['wilson_ci']['success']} "
          f"requested={agg['wilson_ci']['requested']}")
    im = agg["inference_ms"]
    print(f"inference p50={im['p50']}ms p95={im['p95']}ms "
          f"(warm n={im['n_warm']}; cold first={im['cold_first_ms']})")
    if args.corridor_half_width is not None:
        print(f"corridor: half_width={args.corridor_half_width:g}m "
              f"violations={agg['corridor_violations']}/{agg['episodes_done']}")
    print(f"action distribution: {agg['action_distribution']}")
    print(f"-> {path}" + ("  （partial=true：中断写出的部分结果）" if interrupted else ""))
    if interrupted:
        return interrupted["rc"]
    return 0 if agg["episodes_done"] == args.episodes else 1


if __name__ == "__main__":
    sys.exit(main())
