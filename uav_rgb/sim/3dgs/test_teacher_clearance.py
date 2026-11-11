#!/usr/bin/env python3
"""OccAstarTeacher 前向净空检查的形状（clearance = ray / swept）与出生两层判空（任务卡 W1-12）的单测：纯 Python + numpy，不需要 torch / GPU。

背景：新教师（容差 18.9°）室内 19 个撞在前的集里，15 个撞在前进与转向互相切换的那一步。教师的前向检查是沿机头的一条 0.4 m 零宽线段，
看不到转向后首个前进步的侧向位移（偏机头中位 25.5°）和前进后转弯步的前滑（中位 0.066 m）。swept 在机头线段通过之后，再沿预测位移
（动作前机体系速度的线性函数，系数来自 W1-11 的 240 集）延长 reserve 做一次扫掠。

锁定的行为：
  - clearance="ray"（缺省）与改动前逐步相同（test 里内嵌改动前 __call__ 的逐字拷贝作对照）；
  - 合成占据上机头线段通、预测位移线段不通：swept 否决（|err| 不小于容差的一半转向、否则等待），ray 放行；reserve、速度、机体系换算各有用例；
  - 连续等待不超过 2 步、第 3 步放行；不连续的等待不累计；reset() 清零；
  - goal-stop 的「连续 3 步 SAFE 提前结束」不被等待步误触发；
  - 出生两层判空开关关闭时，出生位姿与随机数消耗逐位不变；
  - 教师版本 v3、manifest 的 teacher_clearance / spawn_two_layer、CLI 开关；
  - 导引指针（第三部分）：OracleRouteGuide 的 nearest_in_lookahead 缺省关，target() 与改动前逐步相同（内嵌旧实现的逐字拷贝）；
    路径开头有凹口时旧指针停住、新参数前进；取点范围不超过 lookahead_m；并列取更靠前的顶点；指针只增不减。

用 3dgs-venv 或 .venv-tools 跑（要 numpy）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_teacher_clearance.py
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import importlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
planner = importlib.import_module("sim.3dgs.planner")
collect = importlib.import_module("sim.3dgs.collect_3dgs")
scene_mod = importlib.import_module("sim.3dgs.scene")
guard = importlib.import_module("sim.3dgs.guard")

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
HOLD = 1.0


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 替身
class State:
    """policy 接口里的 state（= 动力学对象）：x / y / z / yaw（弧度）；vx / vy 是世界系水平速度，缺省不给 = 旧 stub（没有速度属性）。"""

    def __init__(self, x, y, yaw_deg, z=2.0, vx=None, vy=None):
        self.x, self.y, self.z, self.yaw = x, y, z, math.radians(yaw_deg)
        if vx is not None:
            self.vx, self.vy = vx, vy


def world_vel(vbx, vby, yaw_deg):
    """机体系速度 -> 世界系（state.vx / vy 的口径）。"""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return c * vbx - s * vby, s * vbx + c * vby


class BoxWorld:
    """OccAstarTeacher 只用到 plan / aim_point / occ.sweep：路径 = 起点到 waypoint 的直线，瞄准点 = waypoint；
    sweep 把线段按 1 cm 采样，落进任一轴对齐矩形 (x0, x1, y0, y1) 就算撞（与 Occupancy.sweep 同样返回 (撞?, 首个撞点)）。"""

    def __init__(self, boxes=(), blocked=False, no_path=False):
        self.boxes, self.blocked, self.no_path, self.occ = list(boxes), blocked, no_path, self
        self.calls = []

    def plan(self, start, goal):
        return None if self.no_path else [start, goal]

    def aim_point(self, path, x, y, lookahead):
        return path[-1]

    def sweep(self, a, b):
        self.calls.append((tuple(a), tuple(b)))
        if self.blocked:
            return True, None
        n = max(2, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.01)) + 1)
        for i in range(n):
            t = i / (n - 1)
            px, py = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            if any(x0 <= px <= x1 and y0 <= py <= y1 for (x0, x1, y0, y1) in self.boxes):
                return True, [px, py, a[2]]
        return False, None


def predicted_body(vbx, vby):
    """规格：前进步的预测位移 d = A·v + b（机体系）。常数取自 planner。"""
    A, B = planner.SWEPT_PRED_A, planner.SWEPT_PRED_B
    return A[0][0] * vbx + A[0][1] * vby + B[0], A[1][0] * vbx + A[1][1] * vby + B[1]


def box_on_predicted_line(vbx, vby, dist, yaw_deg=0.0, half=0.02):
    """放在「预测位移方向上距起点 dist」处的小方块（世界系，起点在原点）。"""
    dx, dy = predicted_body(vbx, vby)
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    bx, by = c * ux - s * uy, s * ux + c * uy
    cx, cy = dist * bx, dist * by
    return (cx - half, cx + half, cy - half, cy + half)


def mk(world, clearance="swept", wp=None, hold_s=HOLD, **kw):
    wp = wp or {"x": 1000.0, "y": 0.0}
    return planner.OccAstarTeacher(world, wp, hold_s=hold_s, clearance=clearance, **kw)


def wp_at_bearing(deg, dist=1000.0):
    return {"x": dist * math.cos(math.radians(deg)), "y": dist * math.sin(math.radians(deg))}


# 转向后的首个前进步：动作前机体系速度约 (−0.14, ∓0.09)（W1-11 数据里 R→F、L→F 步的典型值），预测位移偏机头约 24–27°
V_AFTER_RIGHT = (-0.14, -0.09)


# ------------------------------------------------------------------ 改动前的实现（逐字拷贝，作 ray 模式的对照）
def old_decision(t, state):
    """W1-12 之前 OccAstarTeacher.__call__ 的逐字拷贝（self 换成 t）。ray 模式必须与它逐步相同。返回 (动作, 分支)。"""
    x, y, z = float(state.x), float(state.y), float(getattr(state, "z", 2.0))
    yaw_deg = math.degrees(float(state.yaw))
    if math.hypot(x - t.wp[0], y - t.wp[1]) <= t.stop_radius:
        return SAFE_NEUTRAL, "stop"
    path = t.planner.plan((x, y), t.wp)
    if path is None:
        return SAFE_NEUTRAL, "plan_fail"
    tx, ty = t.planner.aim_point(path, x, y, t.lookahead)
    err = planner._wrap(math.degrees(math.atan2(ty - y, tx - x)) - yaw_deg)
    if err > t.tol:
        return YAW_LEFT, "turn_err"
    if err < -t.tol:
        return YAW_RIGHT, "turn_err"
    yaw = math.radians(yaw_deg)
    hit, _ = t.planner.occ.sweep(
        (x, y, z), (x + t.forward_check * math.cos(yaw), y + t.forward_check * math.sin(yaw), z))
    if hit:
        return (YAW_LEFT if err >= 0 else YAW_RIGHT), "turn_blocked"
    return FORWARD_SLOW, "forward"


def old_spawn_for_episode(scene, occ, jitter_m, rng):
    """W1-12 之前 collect_3dgs._spawn_for_episode 的逐字拷贝。"""
    s = scene["spawn"]
    x, y, yaw = float(s["x"]), float(s["y"]), float(s.get("yaw_deg", 0.0))
    if jitter_m <= 0:
        return x, y, yaw
    for _ in range(100):
        nx = x + rng.uniform(-jitter_m, jitter_m)
        ny = y + rng.uniform(-jitter_m, jitter_m)
        if not occ.collides(nx, ny, float(s["z"])):
            return nx, ny, yaw + rng.uniform(-10.0, 10.0)
    raise RuntimeError(f"spawn 抖动 {jitter_m} m 内找不到自由点")


# ------------------------------------------------------------------ ray：与改动前逐步相同
class TestRayIsUnchanged(unittest.TestCase):
    def test_default_is_ray(self):
        t = planner.OccAstarTeacher(BoxWorld(), {"x": 1e4, "y": 0.0})
        self.assertEqual(t.clearance, "ray")
        self.assertEqual(planner.CLEARANCE_MODES, ("ray", "swept"))

    def test_ray_stepwise_identical_to_old_implementation(self):
        """随机状态 × 随机 waypoint × 前向是否被堵 × 三种容差构造；states 带随机速度属性：ray 模式不读速度，动作与分支都与旧实现相同。"""
        rng = random.Random(11)
        n_forward = n_blocked = n_turn = 0
        for i in range(1500):
            x, y, yaw = rng.uniform(-9, 9), rng.uniform(-9, 9), rng.uniform(-540, 540)
            wp = {"x": x + rng.uniform(-6, 6), "y": y + rng.uniform(-6, 6)}
            if rng.random() < 0.6:          # 机头大致对准目标，前进 / 前向被堵才有足够的样本
                yaw = math.degrees(math.atan2(wp["y"] - y, wp["x"] - x)) + rng.uniform(-25.0, 25.0)
            kw = rng.choice(({}, {"hold_s": 1.0}, {"tol_deg": 25.0}, {"hold_s": 0.5}))
            st = State(x, y, yaw, z=rng.uniform(1.0, 2.4), vx=rng.uniform(-0.5, 0.5), vy=rng.uniform(-0.5, 0.5))
            mode = rng.random()
            if mode < 0.25:
                world = BoxWorld(blocked=True)
            elif mode < 0.65:               # 机头线段上 0.2–0.7 m 处放一个小方块：ray 的线段长度（0.4 m）决定看不看得到它
                d = rng.uniform(0.2, 0.7)
                bx, by = x + d * math.cos(math.radians(yaw)), y + d * math.sin(math.radians(yaw))
                world = BoxWorld(boxes=[(bx - 0.03, bx + 0.03, by - 0.03, by + 0.03)])
            else:
                world = BoxWorld()
            for explicit in (False, True):
                t = planner.OccAstarTeacher(world, wp, **kw, **({"clearance": "ray"} if explicit else {}))
                got = t(None, None, st)
                want = old_decision(t, st)
                self.assertEqual((got[0], t.last_branch), want, (i, kw, explicit))
            n_forward += want[1] == "forward"
            n_blocked += want[1] == "turn_blocked"
            n_turn += want[1] == "turn_err"
        self.assertGreater(min(n_forward, n_blocked, n_turn), 50, (n_forward, n_blocked, n_turn))    # 三条分支都被走到

    def test_ray_segment_ends_at_forward_check_m(self):
        """机头线段长 0.4 m：0.36、0.39 m 处的方块看得到（turn_blocked，目标在左 → 左转），0.43 m 处看不到。线段被改短或改长都会翻。
        swept 模式的机头线段也是这一条（先于预测位移检查）。"""
        self.assertEqual(planner.FORWARD_CHECK_M, 0.4)
        for clearance in ("ray", "swept"):
            for dist, want in ((0.36, (YAW_LEFT, "turn_blocked")), (0.39, (YAW_LEFT, "turn_blocked")), (0.43, (FORWARD_SLOW, "forward"))):
                world = BoxWorld(boxes=[(dist - 0.005, dist + 0.005, -0.05, 0.05)])
                t = mk(world, clearance=clearance, wp=wp_at_bearing(5.0))
                a, _ = t(None, None, State(0.0, 0.0, 0.0, vx=0.0, vy=0.0))
                self.assertEqual((a, t.last_branch), want, (clearance, dist))

    def test_ray_ignores_velocity_and_works_without_velocity_attributes(self):
        wp = wp_at_bearing(0.0)
        world = BoxWorld(boxes=[box_on_predicted_line(*V_AFTER_RIGHT, 0.25)])                     # 预测线上有方块、机头线上没有
        for st in (State(0, 0, 0), State(0, 0, 0, vx=-0.14, vy=-0.09)):
            t = planner.OccAstarTeacher(world, wp, hold_s=HOLD)
            self.assertEqual(t(None, None, st), (FORWARD_SLOW, None))
            self.assertEqual(t.last_branch, "forward")

    def test_ray_version_and_branches_unchanged(self):
        wp = {"x": 1e4, "y": 0.0}
        v = lambda **kw: planner.OccAstarTeacher(BoxWorld(), wp, **kw).version                       # noqa: E731
        self.assertEqual(planner.TEACHER_VERSION, "uavrgb.teacher_occ.v1")
        self.assertEqual(planner.TEACHER_VERSION_HOLD_TOL, "uavrgb.teacher_occ.v2")
        self.assertEqual(v(), planner.TEACHER_VERSION)
        self.assertEqual(v(hold_s=1.0), planner.TEACHER_VERSION_HOLD_TOL)
        self.assertEqual(v(hold_s=1.0, clearance="ray"), planner.TEACHER_VERSION_HOLD_TOL)
        self.assertEqual(v(hold_s=0.5, clearance="ray"), planner.TEACHER_VERSION)

    def test_ray_never_emits_swept_branches(self):
        rng = random.Random(3)
        seen = set()
        for _ in range(500):
            world = BoxWorld(boxes=[(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
                                    for _ in range(3)])
            t = planner.OccAstarTeacher(world, wp_at_bearing(rng.uniform(-30, 30)), hold_s=HOLD)
            t(None, None, State(0, 0, 0, vx=rng.uniform(-.4, .4), vy=rng.uniform(-.4, .4)))
            seen.add(t.last_branch)
        self.assertFalse(seen & {"turn_swept", "wait_swept"}, seen)


# ------------------------------------------------------------------ swept：构造与版本
class TestSweptConstruction(unittest.TestCase):
    def test_version_is_v3(self):
        self.assertEqual(planner.TEACHER_VERSION_SWEPT, "uavrgb.teacher_occ.v3")
        t = mk(BoxWorld())
        self.assertEqual((t.clearance, t.version), ("swept", "uavrgb.teacher_occ.v3"))
        self.assertAlmostEqual(t.tol, planner.yaw_tol_deg(1.0), places=9)
        t2 = mk(BoxWorld(), tol_deg=12.0)                                # 显式 12° 也是 v3：版本跟着 clearance
        self.assertEqual(t2.version, "uavrgb.teacher_occ.v3")

    def test_swept_needs_hold_one_second(self):
        for bad in (None, 0.5, 0.99, 1.5, 2.0):
            with self.assertRaises(ValueError, msg=repr(bad)):
                planner.OccAstarTeacher(BoxWorld(), {"x": 1.0, "y": 0.0}, hold_s=bad, clearance="swept")
        planner.OccAstarTeacher(BoxWorld(), {"x": 1.0, "y": 0.0}, hold_s=1.0, clearance="swept")
        planner.OccAstarTeacher(BoxWorld(), {"x": 1.0, "y": 0.0}, hold_s=1, clearance="swept")

    def test_rejects_unknown_clearance(self):
        for bad in ("rays", "SWEPT", "", None, 1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                planner.OccAstarTeacher(BoxWorld(), {"x": 1.0, "y": 0.0}, hold_s=1.0, clearance=bad)

    def test_constants_are_the_measured_ones(self):
        """常数与 experiments/2026-11-11-teacher-clearance-3dgs/measure_disp.py 的输出一致（W1-12 第 1 步，P2 全量重拟合与 reserve）。"""
        self.assertEqual(planner.SWEPT_PRED_A, ((0.4259, 0.0065), (0.0007, 0.6662)))
        self.assertEqual(planner.SWEPT_PRED_B, (0.1809, 0.0003))
        self.assertEqual(planner.SWEPT_RESERVE_M, 0.1731)
        self.assertEqual((planner.SWEPT_HOLD_S, planner.SWEPT_MAX_WAITS), (1.0, 2))


# ------------------------------------------------------------------ swept：机头线段通、预测位移线段不通
class TestSweptVeto(unittest.TestCase):
    def setUp(self):
        self.d_pred = math.hypot(*predicted_body(*V_AFTER_RIGHT))
        self.vel = V_AFTER_RIGHT
        self.world_box = box_on_predicted_line(*self.vel, 0.25)

    def _decide(self, clearance, boxes, err_deg=0.0, vel=None, yaw=0.0, **kw):
        world = BoxWorld(boxes=boxes, **kw)
        t = mk(world, clearance=clearance, wp=wp_at_bearing(yaw + err_deg))
        vx, vy = world_vel(*(vel or self.vel), yaw)
        a, _ = t(None, None, State(0.0, 0.0, yaw, vx=vx, vy=vy))
        return a, t.last_branch, world

    def test_heading_line_clear_predicted_line_blocked(self):
        """任务卡：机头线段通、预测位移线段不通。ray 放行；swept 在 |err| < 容差一半时等一步。"""
        a, br, world = self._decide("ray", [self.world_box])
        self.assertEqual((a, br), (FORWARD_SLOW, "forward"))
        # 机头线段（0.4 m）确实没撞到方块：ray 只做了这一次 sweep
        self.assertEqual(len(world.calls), 1)
        a, br, world = self._decide("swept", [self.world_box])
        self.assertEqual((a, br), (SAFE_NEUTRAL, "wait_swept"))
        self.assertEqual(len(world.calls), 2)                              # 机头线段 + 预测位移线段

    def test_veto_turns_when_error_is_at_least_half_tolerance(self):
        tol = planner.yaw_tol_deg(1.0)
        for err, want in ((0.5 * tol + 0.05, (YAW_LEFT, "turn_swept")), (-(0.5 * tol + 0.05), (YAW_RIGHT, "turn_swept")),
                          (0.5 * tol - 0.05, (SAFE_NEUTRAL, "wait_swept")), (-(0.5 * tol - 0.05), (SAFE_NEUTRAL, "wait_swept")),
                          (tol - 0.2, (YAW_LEFT, "turn_swept")), (-(tol - 0.2), (YAW_RIGHT, "turn_swept"))):
            a, br, _ = self._decide("swept", [self.world_box], err_deg=err)
            self.assertEqual((a, br), want, err)

    def test_error_exactly_half_tolerance_turns(self):
        """边界：|err| 恰为容差的一半归「转」。目标 (1, ±1)、机头 0°：err = ±atan2(1, 1) = ±45.0（浮点上恰好），容差 90° 的一半也是 45.0。"""
        vx, vy = world_vel(*self.vel, 0.0)
        for sign, want in ((1.0, YAW_LEFT), (-1.0, YAW_RIGHT)):
            world = BoxWorld(boxes=[self.world_box])
            t = mk(world, wp={"x": 1.0, "y": sign}, tol_deg=90.0)
            a, _ = t(None, None, State(0.0, 0.0, 0.0, vx=vx, vy=vy))
            self.assertEqual((a, t.last_branch), (want, "turn_swept"), sign)

    def test_no_veto_when_predicted_line_is_clear(self):
        for clearance in ("ray", "swept"):
            a, br, _ = self._decide(clearance, [])
            self.assertEqual((a, br), (FORWARD_SLOW, "forward"), clearance)

    def test_prediction_uses_velocity(self):
        """同一个方块：速度为 0 时预测位移沿机头，方块不在线上，不否决；有转向残留速度时预测位移偏机头，否决。"""
        a, br, _ = self._decide("swept", [self.world_box], vel=(0.0, 0.0))
        self.assertEqual((a, br), (FORWARD_SLOW, "forward"))
        a, br, _ = self._decide("swept", [self.world_box], vel=self.vel)
        self.assertEqual(br, "wait_swept")
        # 反号的侧向速度把预测位移推向另一侧：同一个方块不再挡路
        a, br, _ = self._decide("swept", [self.world_box], vel=(self.vel[0], -self.vel[1]))
        self.assertEqual((a, br), (FORWARD_SLOW, "forward"))

    def test_reserve_extends_the_segment_beyond_predicted_displacement(self):
        """方块在预测位移终点之外、reserve 之内：否决；超出 reserve：放行。"""
        inside = box_on_predicted_line(*self.vel, self.d_pred + 0.5 * planner.SWEPT_RESERVE_M)
        beyond = box_on_predicted_line(*self.vel, self.d_pred + planner.SWEPT_RESERVE_M + 0.05)
        self.assertEqual(self._decide("swept", [inside])[1], "wait_swept")
        self.assertEqual(self._decide("swept", [beyond])[:2], (FORWARD_SLOW, "forward"))
        before_end = box_on_predicted_line(*self.vel, 0.5 * self.d_pred)
        self.assertEqual(self._decide("swept", [before_end])[1], "wait_swept")

    def test_body_frame_conversion_is_rotation_invariant(self):
        """整个场景绕原点转 yaw：速度（世界系）与方块一起转，决策不变。换错坐标系（把世界系速度当机体系）会在 yaw != 0 时露出来。"""
        base = self._decide("swept", [self.world_box])[:2]
        for yaw in (30.0, 90.0, -120.0, 171.0):
            box = box_on_predicted_line(*self.vel, 0.25, yaw_deg=yaw)
            self.assertEqual(self._decide("swept", [box], yaw=yaw)[:2], base, yaw)
            self.assertEqual(self._decide("ray", [box], yaw=yaw)[:2], (FORWARD_SLOW, "forward"), yaw)

    def test_sweep_uses_current_height_and_starts_at_the_vehicle(self):
        world = BoxWorld(boxes=[self.world_box])
        t = mk(world, wp=wp_at_bearing(0.0))
        vx, vy = world_vel(*self.vel, 0.0)
        t(None, None, State(0.0, 0.0, 0.0, z=1.23, vx=vx, vy=vy))
        a, b = world.calls[-1]
        self.assertEqual((a, b[2]), ((0.0, 0.0, 1.23), 1.23))
        dx, dy = predicted_body(*self.vel)
        n = math.hypot(dx, dy)
        want = (dx + planner.SWEPT_RESERVE_M * dx / n, dy + planner.SWEPT_RESERVE_M * dy / n)
        self.assertAlmostEqual(b[0], want[0], places=9)
        self.assertAlmostEqual(b[1], want[1], places=9)

    def test_only_forward_is_vetoed(self):
        """转向不否决：|err| 超容差时仍是 turn_err；机头线段被堵时仍是 turn_blocked（先于 swept 检查）；stop / plan_fail 不受影响。"""
        tol = planner.yaw_tol_deg(1.0)
        self.assertEqual(self._decide("swept", [self.world_box], err_deg=tol + 3)[:2], (YAW_LEFT, "turn_err"))
        self.assertEqual(self._decide("swept", [self.world_box], err_deg=-tol - 3)[:2], (YAW_RIGHT, "turn_err"))
        a, br, world = self._decide("swept", [], blocked=True, err_deg=5.0)
        self.assertEqual((a, br), (YAW_LEFT, "turn_blocked"))
        self.assertEqual(len(world.calls), 1)                              # 机头线段已堵，不再做预测位移扫掠
        a, br, _ = self._decide("swept", [], err_deg=-5.0, blocked=True)
        self.assertEqual((a, br), (YAW_RIGHT, "turn_blocked"))
        t = mk(BoxWorld(boxes=[self.world_box], no_path=True))
        self.assertEqual((t(None, None, State(0, 0, 0, vx=-.14, vy=-.09))[0], t.last_branch), (SAFE_NEUTRAL, "plan_fail"))
        t = mk(BoxWorld(boxes=[self.world_box]), wp={"x": 0.3, "y": 0.0})
        self.assertEqual((t(None, None, State(0, 0, 0, vx=-.14, vy=-.09))[0], t.last_branch), (SAFE_NEUTRAL, "stop"))

    def test_swept_needs_velocity_attributes(self):
        """没有 vx / vy 的 state 会让 swept 直接报错（不悄悄当成速度 0）；ray 不受影响。"""
        with self.assertRaises(AttributeError):
            mk(BoxWorld())(None, None, State(0, 0, 0))
        planner.OccAstarTeacher(BoxWorld(), wp_at_bearing(0.0), hold_s=HOLD)(None, None, State(0, 0, 0))


# ------------------------------------------------------------------ swept：连续等待上限
class TestWaitCap(unittest.TestCase):
    def setUp(self):
        self.world = BoxWorld(boxes=[box_on_predicted_line(*V_AFTER_RIGHT, 0.25)])
        self.vx, self.vy = world_vel(*V_AFTER_RIGHT, 0.0)

    def _teacher(self, bearing=0.0):
        return mk(self.world, wp=wp_at_bearing(bearing))

    def _call(self, t):
        a, _ = t(None, None, State(0.0, 0.0, 0.0, vx=self.vx, vy=self.vy))
        return a, t.last_branch

    def test_two_waits_then_release(self):
        """否决一直成立、|err| 小：等、等、第 3 步按原规则放行（只看机头线段，这里通），之后重新开始计数。"""
        t = self._teacher()
        got = [self._call(t) for _ in range(7)]
        W, F = (SAFE_NEUTRAL, "wait_swept"), (FORWARD_SLOW, "forward")
        self.assertEqual(got, [W, W, F, W, W, F, W])

    def test_cap_constant_is_two(self):
        self.assertEqual(planner.SWEPT_MAX_WAITS, 2)

    def test_reset_clears_the_counter(self):
        t = self._teacher()
        self._call(t)
        self._call(t)
        t.reset()
        self.assertEqual(self._call(t), (SAFE_NEUTRAL, "wait_swept"))
        self.assertEqual(self._call(t), (SAFE_NEUTRAL, "wait_swept"))
        self.assertEqual(self._call(t), (FORWARD_SLOW, "forward"))

    def test_waits_must_be_consecutive(self):
        """两次等待之间插进任何一个不是等待的步（转向、放行），计数清零：不是累计。"""
        t = self._teacher()
        tol = planner.yaw_tol_deg(1.0)
        W = (SAFE_NEUTRAL, "wait_swept")
        self.assertEqual(self._call(t), W)
        t.set_waypoint(wp_at_bearing(tol + 5.0))                                      # 误差超容差：turn_err，清零
        self.assertEqual(self._call(t), (YAW_LEFT, "turn_err"))
        t.set_waypoint(wp_at_bearing(0.0))
        self.assertEqual([self._call(t) for _ in range(3)], [W, W, (FORWARD_SLOW, "forward")])

    def test_blocked_ray_step_also_breaks_the_streak(self):
        t = self._teacher()
        W = (SAFE_NEUTRAL, "wait_swept")
        self.assertEqual(self._call(t), W)
        self.world.blocked = True
        self.assertEqual(self._call(t)[1], "turn_blocked")
        self.world.blocked = False
        self.assertEqual([self._call(t) for _ in range(3)], [W, W, (FORWARD_SLOW, "forward")])

    def test_turn_swept_is_not_a_wait(self):
        """|err| 不小于容差一半时否决 = 转向，不占等待额度；转向改变了几何，不是卡死。"""
        t = self._teacher(bearing=12.0)
        got = [self._call(t) for _ in range(5)]
        self.assertEqual(got, [(YAW_LEFT, "turn_swept")] * 5)
        t = self._teacher()
        self.assertEqual(self._call(t), (SAFE_NEUTRAL, "wait_swept"))        # 转向之后等待额度仍是满的
        t.set_waypoint(wp_at_bearing(12.0))
        self.assertEqual(self._call(t), (YAW_LEFT, "turn_swept"))
        t.set_waypoint(wp_at_bearing(0.0))
        self.assertEqual([self._call(t) for _ in range(3)][-1], (FORWARD_SLOW, "forward"))


# ------------------------------------------------------------------ goal-stop 不被等待步误触发
class TestGoalStopNotTriggeredByWaits(unittest.TestCase):
    def test_wait_never_emitted_inside_stop_radius(self):
        """run_episode 的 goal-stop 只在动作前位姿进了 stop 半径时数「连续 3 步 SAFE」：教师的 stop 分支先于否决，半径内永远是 stop，
        不会出 wait_swept；半径外同样的状态（机头对准目标、预测线段被挡）才是 wait_swept，所以等待步的 SAFE 不会累计进 goal-stop 的计数。
        状态：机头对准 waypoint（误差 < 容差一半，否决时会等待）、动作前速度带转向残留、预测线段上有方块，到 waypoint 的距离随机跨过半径。"""
        rng = random.Random(5)
        n_in = n_out = 0
        for _ in range(400):
            wp = {"x": rng.uniform(-3, 3), "y": rng.uniform(-3, 3)}
            ang = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0.2, 0.9)
            x, y = wp["x"] + r * math.cos(ang), wp["y"] + r * math.sin(ang)
            yaw = math.degrees(math.atan2(wp["y"] - y, wp["x"] - x)) + rng.uniform(-3.0, 3.0)
            vel = (-0.14, rng.choice((-0.09, 0.09)))
            vx, vy = world_vel(*vel, yaw)
            box = box_on_predicted_line(*vel, 0.25, yaw_deg=yaw)
            world = BoxWorld(boxes=[(box[0] + x, box[1] + x, box[2] + y, box[3] + y)])
            t = mk(world, wp=wp)
            a, _ = t(None, None, State(x, y, yaw, vx=vx, vy=vy))
            if r <= t.stop_radius:
                n_in += 1
                self.assertEqual((a, t.last_branch), (SAFE_NEUTRAL, "stop"), (r, yaw))
            else:
                n_out += 1
                self.assertEqual((a, t.last_branch), (SAFE_NEUTRAL, "wait_swept"), (r, yaw))
        self.assertGreater(min(n_in, n_out), 100, (n_in, n_out))

    def _run(self, policy, spawn_x, steps, name):
        with tempfile.TemporaryDirectory() as tmp:
            scene, renderer, occ, params, identity = _mock_env(tmp)
            scene["spawn"] = {"x": spawn_x, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}
            scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}
            m, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, policy(scene, occ), steps, HOLD,
                                         split=identity["split"], identity=identity)
            self.assertIsNone(err)
            with open(os.path.join(tmp, name, "actions.jsonl"), encoding="utf-8") as f:
                return m, [json.loads(x)["action_executed"] for x in f if x.strip()]

    def test_run_episode_goal_stop_counts_only_safe_steps_inside_the_radius(self):
        """run_episode 的 goal-stop：连续 3 步 SAFE 且动作前位姿在 stop 半径内才提前结束。脚本策略恒出 SAFE：
        在半径外（距 waypoint 0.6 m）出多少步 SAFE 都不结束（等待步的 SAFE 不会累计）；在半径内（0.3 m）3 步后结束（对照：开关有效）。"""
        class AlwaysSafe:
            name, planner_name, version, stop_radius = "plan_action", "astar_occ", "test", planner.STOP_RADIUS_M

            def __call__(self, rgb, depth, state):
                return SAFE_NEUTRAL, None
        m, seq = self._run(lambda scene, occ: AlwaysSafe(), 9.4, 8, "outside")
        self.assertEqual((len(seq), m["completed"], m.get("stopped_on_goal")), (8, True, False), seq)
        m, seq = self._run(lambda scene, occ: AlwaysSafe(), 9.7, 8, "inside")
        self.assertEqual((len(seq), m["completed"], m.get("stopped_on_goal")), (3, True, True), seq)

    def test_swept_teacher_waits_in_run_episode_then_moves(self):
        """真的 swept 教师放进 run_episode：前两步否决（等待，距 waypoint 0.6 m，半径外），第 3 步放行前进；episode 不因 SAFE 提前结束。"""
        def mk_t(scene, occ):
            t = planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"], hold_s=HOLD, clearance="swept")
            veto = iter([True, True] + [False] * 200)
            t._swept_blocked = lambda state, x, y, z: next(veto)
            return t
        m, seq = self._run(mk_t, 9.4, 3, "waits")
        self.assertEqual(seq, [SAFE_NEUTRAL, SAFE_NEUTRAL, FORWARD_SLOW])
        self.assertEqual((m["num_steps"], m.get("stopped_on_goal")), (3, False))


# ------------------------------------------------------------------ manifest / DaggerPolicy / CLI
def _mock_env(tmp, seed=2):
    """合成走廊 + mock 渲染器 + cf_blackbox，直接调 run_episode 用（同 test_teacher_tol._mock_env）。"""
    scene_dir = os.path.join(tmp, "scene")
    scene_mod.make_corridor(scene_dir, seed=seed, length=12.0)
    scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
    renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
    occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)
    params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
    return scene, renderer, occ, params, collect.scene_identity(scene)


class FakeStudent:
    """student.StudentPolicy 的最小替身：恒定动作，不起 uavrgb_infer。"""

    def reset(self):
        pass

    def act(self, msg):
        return 0, 1.0, [0.0] * 4


class TestManifest(unittest.TestCase):
    def _episode(self, tmp, policy, name, **kw):
        scene, renderer, occ, params, identity = _mock_env(tmp)
        scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}
        pol = policy(scene, occ)
        m, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, pol, 3, HOLD,
                                     split=identity["split"], identity=identity, **kw)
        self.assertIsNone(err)
        return m

    def test_teacher_manifest_records_clearance_and_version(self):
        mk_t = lambda **kw: (lambda scene, occ: planner.OccAstarTeacher(                         # noqa: E731
            planner.OccPlanner(occ), scene["waypoint"], hold_s=HOLD, **kw))
        with tempfile.TemporaryDirectory() as tmp:
            m = self._episode(tmp, mk_t(clearance="swept"), "swept")
            self.assertEqual((m["teacher_clearance"], m["teacher_version"]), ("swept", planner.TEACHER_VERSION_SWEPT))
            self.assertAlmostEqual(m["teacher_yaw_tol_deg"], planner.yaw_tol_deg(1.0), places=9)
            m = self._episode(tmp, mk_t(), "ray")
            self.assertEqual((m["teacher_clearance"], m["teacher_version"]), ("ray", planner.TEACHER_VERSION_HOLD_TOL))

    def test_dagger_manifest_records_expert_clearance_and_resets_expert(self):
        with tempfile.TemporaryDirectory() as tmp:
            made = {}

            def mk_d(scene, occ):
                made["t"] = planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"], hold_s=HOLD, clearance="swept")
                return collect.DaggerPolicy(FakeStudent(), made["t"], 0.0, random.Random(0))
            m = self._episode(tmp, mk_d, "dagger", dagger={"executed_by": "student", "dagger_round": 1, "beta": 0.0})
            self.assertEqual((m["teacher_clearance"], m["teacher_version"]), ("swept", planner.TEACHER_VERSION_SWEPT))
            t = made["t"]
            t._waits = 2
            pol = collect.DaggerPolicy(FakeStudent(), t, 0.0, random.Random(0))
            pol.reset()
            self.assertEqual(t._waits, 0)                                   # 集边界清教师的等待计数

    def test_non_astar_policies_manifest_has_no_clearance_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, mk_p in (("fwd", lambda scene, occ: collect.ForwardPolicy()),
                               ("stu", lambda scene, occ: collect.StudentAdapter(FakeStudent())),
                               ("depth", lambda scene, occ: collect.DepthTeacher())):
                m = self._episode(tmp, mk_p, name)
                self.assertNotIn("teacher_clearance", m, name)
                self.assertIs(m["spawn_two_layer"], False, name)

    def test_run_episode_resets_teacher_between_episodes(self):
        """同一个教师对象跑两集：第一集结束时等待计数不留给第二集。"""
        with tempfile.TemporaryDirectory() as tmp:
            scene, renderer, occ, params, identity = _mock_env(tmp)
            scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}
            t = planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"], hold_s=HOLD, clearance="swept")
            t._waits = 2
            collect.run_episode(scene, os.path.join(tmp, "e0"), renderer, occ, params, t, 1, HOLD, split=identity["split"], identity=identity)
            self.assertEqual(t._waits, 0)


class TestCli(unittest.TestCase):
    def _scene_json(self, tmp):
        scene_dir = os.path.join(tmp, "scene")
        scene = scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        scene["routes"] = [{"name": "r0", "spawn": scene["spawn"], "waypoint": scene["waypoint"]},
                           {"name": "r1", "spawn": {"x": 3.0, "y": 0.0, "z": 2.0, "yaw_deg": 90.0},
                            "waypoint": {"x": 10.0, "y": 0.0, "z": 2.0}, "max_steps": 4}]
        path = os.path.join(scene_dir, "scene.json")
        scene_mod.save_scene({k: v for k, v in scene.items() if not k.startswith("_") and k not in ("path", "dir", "ply_path")}, path)
        return path

    def _run(self, tmp, out, *extra):
        return subprocess.run([sys.executable, "-m", "sim.3dgs.collect_3dgs", "--scene", self._scene_json(tmp), "--out", os.path.join(tmp, out),
                               "--mock-renderer", "--dynamics", "cf_blackbox", "--policy", "astar", "--route", "1", *extra],
                              cwd=ROOT, capture_output=True, text=True, timeout=180)

    def test_cli_default_is_ray_and_switches_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, "dflt", "--hold", "1.0")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load_json(os.path.join(tmp, "dflt", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["teacher_clearance"], m["spawn_two_layer"], m["teacher_version"]), ("ray", False, planner.TEACHER_VERSION_HOLD_TOL))
            run = _load_json(os.path.join(tmp, "dflt", "run.json"))
            self.assertEqual((run["teacher_clearance"], run["spawn_two_layer"]), ("ray", False))
            r = self._run(tmp, "both", "--hold", "1.0", "--teacher-clearance", "swept", "--spawn-jitter", "0.1", "--spawn-two-layer")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load_json(os.path.join(tmp, "both", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["teacher_clearance"], m["spawn_two_layer"], m["teacher_version"]),
                             ("swept", True, planner.TEACHER_VERSION_SWEPT))

    def test_cli_rejects_inconsistent_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, "h05", "--hold", "0.5", "--teacher-clearance", "swept")
            self.assertEqual(r.returncode, 2, r.stderr[-1000:])
            self.assertIn("hold", r.stderr)
            r = self._run(tmp, "nojit", "--hold", "1.0", "--spawn-two-layer")
            self.assertEqual(r.returncode, 2, r.stderr[-1000:])
            self.assertIn("spawn-jitter", r.stderr)
            r = subprocess.run([sys.executable, "-m", "sim.3dgs.collect_3dgs", "--scene", self._scene_json(tmp), "--out", os.path.join(tmp, "dep"),
                                "--mock-renderer", "--dynamics", "cf_blackbox", "--policy", "teacher", "--hold", "1.0",
                                "--teacher-clearance", "swept"], cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 2, r.stderr[-1000:])


# ------------------------------------------------------------------ 出生两层判空
class SpawnOcc:
    """_spawn_for_episode 只用 collides / clearance / body_radius。"""

    body_radius = 0.102

    def __init__(self, collide=None, clearance=None):
        self._collide = collide or (lambda x, y, z: False)
        self._clearance = clearance or (lambda x, y, z: 9.0)
        self.n_collide = self.n_clearance = 0

    def collides(self, x, y, z=2.0):
        self.n_collide += 1
        return self._collide(x, y, z)

    def clearance(self, x, y, z=2.0):
        self.n_clearance += 1
        return self._clearance(x, y, z)


SCENE = {"spawn": {"x": 1.0, "y": -2.0, "z": 1.2, "yaw_deg": 30.0}}


class TestSpawnSwitchOff(unittest.TestCase):
    def test_default_off_is_bit_identical_in_pose_and_rng_consumption(self):
        """缺省关：出生位姿与随机数消耗顺序与旧实现逐位相同（多种占据：自由、一半拒绝、几乎全拒绝），之后再抽一个数也相同。"""
        occs = {"free": SpawnOcc(),
                "half": SpawnOcc(collide=lambda x, y, z: x > 1.0),
                "mostly": SpawnOcc(collide=lambda x, y, z: (x - 1.0) ** 2 + (y + 2.0) ** 2 > 0.0004),
                "layered": SpawnOcc(collide=lambda x, y, z: z > 1.21 and x > 1.0)}      # 只在巡航层上有障碍：关闭时看不见
        for name, occ in occs.items():
            for jitter in (0.05, 0.2, 0.5):
                for seed in range(40):
                    ra, rb, rc = (np.random.default_rng(seed) for _ in range(3))
                    try:
                        want = old_spawn_for_episode(SCENE, occ, jitter, ra)
                    except RuntimeError:
                        with self.assertRaises(RuntimeError):
                            collect._spawn_for_episode(SCENE, occ, jitter, rb)
                        with self.assertRaises(RuntimeError):
                            collect._spawn_for_episode(SCENE, occ, jitter, rc, two_layer=False)
                        continue
                    got_default = collect._spawn_for_episode(SCENE, occ, jitter, rb)
                    got_explicit = collect._spawn_for_episode(SCENE, occ, jitter, rc, two_layer=False)
                    self.assertEqual((got_default, got_explicit), (want, want), (name, jitter, seed))
                    self.assertEqual(ra.bit_generator.state, rb.bit_generator.state, (name, jitter, seed))
                    self.assertEqual(ra.bit_generator.state, rc.bit_generator.state, (name, jitter, seed))

    def test_off_never_queries_cruise_layer_or_clearance(self):
        occ = SpawnOcc()
        collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(1))
        self.assertEqual((occ.n_collide, occ.n_clearance), (1, 0))

    def test_zero_jitter_returns_spawn_untouched_with_or_without_switch(self):
        for two in (False, True):
            rng = np.random.default_rng(4)
            before = rng.bit_generator.state
            self.assertEqual(collect._spawn_for_episode(SCENE, SpawnOcc(collide=lambda *a: True), 0.0, rng, two_layer=two), (1.0, -2.0, 30.0))
            self.assertEqual(rng.bit_generator.state, before)


class TestSpawnSwitchOn(unittest.TestCase):
    def test_cruise_layer_collision_rejects_candidate(self):
        """候选在出生高度那层空、巡航高度那层撞：关闭时被接受，打开时被拒绝。"""
        occ = SpawnOcc(collide=lambda x, y, z: z > 1.2 + 1e-9 and x > 1.0)
        off = [collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(s)) for s in range(60)]
        on = [collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(s), two_layer=True) for s in range(60)]
        self.assertTrue(any(p[0] > 1.0 for p in off))
        self.assertTrue(all(p[0] <= 1.0 for p in on))

    def test_cruise_height_is_spawn_z_plus_constant(self):
        seen = []
        occ = SpawnOcc(collide=lambda x, y, z: seen.append(round(z, 6)) or False)
        collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(0), two_layer=True)
        self.assertEqual(sorted(set(seen)), [1.2, round(1.2 + collect.SPAWN_CRUISE_DZ_M, 6)])

    def test_clearance_threshold_is_body_radius_plus_margin(self):
        thr = SpawnOcc.body_radius + collect.SPAWN_CLEARANCE_MARGIN_M
        self.assertAlmostEqual(thr, 0.152, places=9)
        ok = lambda c: collect._spawn_two_layer_ok(SpawnOcc(clearance=lambda x, y, z: c), 0.0, 0.0, 1.2)     # noqa: E731
        self.assertTrue(ok(thr))                                           # 恰等于下限：通过
        self.assertTrue(ok(thr + 0.01))
        self.assertFalse(ok(thr - 0.001))
        self.assertFalse(ok(0.0))

    def test_low_clearance_candidate_rejected(self):
        occ = SpawnOcc(clearance=lambda x, y, z: 0.05 if y > -2.0 else 1.0)
        on = [collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(s), two_layer=True) for s in range(60)]
        self.assertTrue(all(p[1] <= -2.0 for p in on))
        off = [collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(s)) for s in range(60)]
        self.assertTrue(any(p[1] > -2.0 for p in off))

    def test_no_free_point_raises(self):
        occ = SpawnOcc(clearance=lambda x, y, z: 0.0)
        with self.assertRaises(RuntimeError):
            collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(0), two_layer=True)
        collect._spawn_for_episode(SCENE, occ, 0.3, np.random.default_rng(0), two_layer=False)         # 关闭时同一个占据可以出生


class TestSpawnInRunEpisode(unittest.TestCase):
    def test_run_episode_pose_unchanged_when_off_and_manifest_records_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, renderer, occ, params, identity = _mock_env(tmp)
            scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}
            want = old_spawn_for_episode(scene, occ, 0.3, np.random.default_rng(7))
            res = {}
            for name, kw in (("dflt", {}), ("off", {"spawn_two_layer": False}), ("on", {"spawn_two_layer": True})):
                t = planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"], hold_s=HOLD)
                m, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, t, 2, HOLD, split=identity["split"],
                                             identity=identity, seed=7, spawn_jitter=0.3, **kw)
                self.assertIsNone(err)
                with open(os.path.join(tmp, name, "actions.jsonl"), encoding="utf-8") as f:
                    acts = [json.loads(x) for x in f if x.strip()]
                res[name] = (m["spawn_two_layer"], (acts[0]["pose_before"]["x"], acts[0]["pose_before"]["y"], acts[0]["yaw_before"]))
            self.assertEqual((res["dflt"][0], res["off"][0], res["on"][0]), (False, False, True))
            for name in ("dflt", "off"):
                self.assertAlmostEqual(res[name][1][0], want[0], places=12)
                self.assertAlmostEqual(res[name][1][1], want[1], places=12)
                self.assertAlmostEqual(res[name][1][2], want[2], places=9)


    def test_run_episode_hands_the_flag_to_the_spawn_check(self):
        """缺省 / False：run_episode 从不调用两层检查；True：调用。开关值被取反或没传下去，这里会翻。"""
        calls = []
        orig = collect._spawn_two_layer_ok
        collect._spawn_two_layer_ok = lambda occ, x, y, z: calls.append((x, y, z)) or True
        try:
            with tempfile.TemporaryDirectory() as tmp:
                scene, renderer, occ, params, identity = _mock_env(tmp)
                scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}
                for name, kw in (("dflt", {}), ("off", {"spawn_two_layer": False}), ("on", {"spawn_two_layer": True})):
                    del calls[:]
                    t = planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"], hold_s=HOLD)
                    _, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, t, 1, HOLD, split=identity["split"],
                                                 identity=identity, seed=7, spawn_jitter=0.3, **kw)
                    self.assertIsNone(err)
                    self.assertEqual(len(calls) > 0, name == "on", (name, len(calls)))
        finally:
            collect._spawn_two_layer_ok = orig


# ------------------------------------------------------------------ 导引指针：nearest_in_lookahead（第三部分）
class OldRouteGuide:
    """W1-12 之前 guard.OracleRouteGuide 的逐字拷贝（构造与 target()）。新实现缺省参数时必须与它逐步相同。"""

    def __init__(self, plan_fn, waypoint_xy, lookahead_m):
        self.plan_fn, self.waypoint = plan_fn, (float(waypoint_xy[0]), float(waypoint_xy[1]))
        self.lookahead = float(lookahead_m)
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


def dip_path():
    """K1 的 841150 r13 / r4 两集的 A* 路径开头（顶点取自离线重放，米）：沿 +x 走两格后向南凹 0.1 m（第 3 个顶点），回到原高度再向北、向东北。"""
    p = [(-4.829, 3.778), (-4.729, 3.778), (-4.629, 3.678), (-4.529, 3.778)]
    p += [(-4.529, round(3.778 + 0.1 * i, 3)) for i in range(1, 8)]
    p += [(round(-4.529 + 0.1 * i, 3), round(4.478 + 0.1 * i, 3)) for i in range(1, 13)]
    return p


def random_grid_path(rng, n=70):
    """0.1 m 栅格上的 8 邻域随机游走路径，偏向 +x / +y 前进，偶尔向后或侧向凹一格（制造距离序列的局部极大）。"""
    x, y, out = 0.0, 0.0, [(0.0, 0.0)]
    moves = [(1, 0), (1, 1), (0, 1), (1, 1), (1, 0), (0, 1), (1, -1), (-1, 1), (0, -1), (-1, 0)]
    for _ in range(n):
        dx, dy = rng.choice(moves)
        x, y = round(x + 0.1 * dx, 3), round(y + 0.1 * dy, 3)
        out.append((x, y))
    return out


class TestGuideNearestInLookahead(unittest.TestCase):
    def test_default_target_identical_to_old_implementation(self):
        """缺省（不传新参数）与显式 False：每一步的 target() 与指针 k 都与旧实现逐位相同。随机路径 × 随机位姿序列（贴着路径、偏出路径、突然跳远）× 三种前视。"""
        rng = random.Random(21)
        n_steps = n_moved = 0
        for case in range(120):
            path = random_grid_path(rng)
            wp = (path[-1][0] + 0.03, path[-1][1] - 0.02)
            la = rng.choice((0.5, 1.5, 3.0))
            old = OldRouteGuide(lambda s, e, path=path: path, wp, la)
            gd = guard.OracleRouteGuide(lambda s, e, path=path: path, wp, la)
            ge = guard.OracleRouteGuide(lambda s, e, path=path: path, wp, la, nearest_in_lookahead=False)
            for step in range(60):
                base = path[min(len(path) - 1, step * rng.choice((1, 1, 2)))]
                mode = rng.random()
                x, y = (base[0] + rng.uniform(-0.05, 0.05), base[1] + rng.uniform(-0.05, 0.05)) if mode < 0.6 else \
                       (base[0] + rng.uniform(-0.6, 0.6), base[1] + rng.uniform(-0.6, 0.6)) if mode < 0.9 else \
                       (rng.uniform(-1, 8), rng.uniform(-1, 8))
                want = old.target(x, y)
                self.assertEqual(gd.target(x, y), want, (case, step))
                self.assertEqual(ge.target(x, y), want, (case, step))
                self.assertEqual((gd.k, ge.k), (old.k, old.k), (case, step))
                n_steps += 1
                n_moved += old.k > 0
        self.assertGreater(n_moved, n_steps // 4)                 # 指针确实走动过，不是一直停在 0

    def test_flag_default_false_and_stored(self):
        g = guard.OracleRouteGuide(lambda s, e: [], (1.0, 0.0), 1.0)
        self.assertIs(g.nearest_in_lookahead, False)
        self.assertIs(guard.OracleRouteGuide(lambda s, e: [], (1.0, 0.0), 1.0, nearest_in_lookahead=True).nearest_in_lookahead, True)
        self.assertIs(guard.OracleRouteGuide(lambda s, e: [], (1.0, 0.0), 1.0, True).nearest_in_lookahead, True)

    @staticmethod
    def _pursue(guide_target, start, steps=60, step_m=0.25):
        """理想的闭环：飞机每步朝当前目标点直线移动 step_m（到不了就停在目标点上），这是规则层跟随目标点的最简模型。"""
        x, y = start
        traj = []
        for _ in range(steps):
            tx, ty = guide_target(x, y)
            traj.append((x, y, (tx, ty)))
            d = math.hypot(tx - x, ty - y)
            if d > 1e-9:
                f = min(1.0, step_m / d)
                x, y = x + f * (tx - x), y + f * (ty - y)
        return traj

    def test_dip_at_the_start_stalls_the_old_pointer_and_not_the_new_one(self):
        """复现 K1 的停滞：路径开头有个凹口，飞机朝目标点直飞（切过路径的拐角），到了凹口北侧后，「下一个顶点更远」对它的所有位置都成立，
        旧指针停在凹口之前，目标点一动不动，飞机飞到目标点后原地不走（K1 里是原地转向）；新参数让指针越过凹口，目标点随飞机前进，飞到路径末端。"""
        path = dip_path()
        wp = (path[-1][0], path[-1][1])
        start = (-4.79, 3.80)
        old = OldRouteGuide(lambda s, e: path, wp, 1.5)
        new = guard.OracleRouteGuide(lambda s, e: path, wp, 1.5, nearest_in_lookahead=True)
        t_old, t_new = self._pursue(old.target, start), self._pursue(new.target, start)
        self.assertLessEqual(old.k, 1)                                  # 停在凹口之前（顶点 2 是凹口底部）
        self.assertEqual(len({t for _, _, t in t_old[1:]}), 1, t_old[:4])   # 第 1 步起目标点一动不动
        x, y, t = t_old[-1]
        self.assertLess(math.hypot(x - t[0], y - t[1]), 0.05)           # 飞机到了停滞的目标点，没有继续前进
        self.assertGreater(math.hypot(x - wp[0], y - wp[1]), 1.0)       # 离 waypoint 还有 1 m 以上
        self.assertGreaterEqual(new.k, len(path) - 3)                   # 新指针跟到路径末端
        x, y, _ = t_new[-1]
        self.assertLess(math.hypot(x - wp[0], y - wp[1]), 0.3)
        self.assertGreater(len({t for _, _, t in t_new}), 3)            # 目标点随飞机前进（路径只有 2.9 m，剩余不足前视后目标点就是 waypoint）

    def test_window_is_bounded_by_lookahead_path_length(self):
        """取点范围 = 从当前指针沿路径累计不超过 lookahead_m。U 形路径（去程 x=0、回程 x=0.3，相隔 0.3 m）的顶端附近，飞机在两段之间偏向回程段：
        回程段的对应顶点沿路径 1.3 m，前视 1.0 m 时范围够不到，指针不跳；前视 3.0 m 时落进范围，会跳过去（已知的取舍，docstring 写明）。"""
        u = [(0.0, round(0.1 * i, 3)) for i in range(0, 41)] + [(round(0.1 * i, 3), 4.0) for i in range(1, 4)] + \
            [(0.3, round(4.0 - 0.1 * i, 3)) for i in range(1, 41)]
        poses = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.2, 3.5)]
        ks = {}
        for la in (1.0, 3.0):
            g = guard.OracleRouteGuide(lambda s, e: u, (0.3, 0.0), la, nearest_in_lookahead=True)
            for q in poses:
                g.target(*q)
            ks[la] = g.k
        self.assertEqual(ks[1.0], 35)                                   # 去程段上 (0, 3.5) 的顶点
        self.assertGreater(ks[3.0], 43)                                 # 回程段（下标 >= 44）上 (0.3, 3.5) 的顶点

    def test_ties_take_the_later_vertex(self):
        """飞机恰在两个顶点正中间：并列取更靠前的（与旧规则「不更远就前进」同向）。"""
        path = [(round(0.1 * i, 3), 0.0) for i in range(0, 30)]
        for flag in (False, True):
            g = guard.OracleRouteGuide(lambda s, e: path, (2.9, 0.0), 1.0, nearest_in_lookahead=flag)
            g.target(0.25, 0.0)
            self.assertEqual(g.k, 3, flag)                              # 0.25 在顶点 2 与 3 之间正中：取 3（旧规则也取 3）

    def test_window_is_measured_from_the_current_pointer(self):
        """窗口的路径长度从当前指针算起，不含指针之前的那一段：直线路径（1 m 一格）、前视 2 m，指针在顶点 1 时，顶点 3（沿路径 2 m）仍在窗口内，
        飞机在顶点 3 上，指针要一次跳到 3。"""
        path = [(float(i), 0.0) for i in range(6)]
        g = guard.OracleRouteGuide(lambda s, e: path, (5.0, 0.0), 2.0, nearest_in_lookahead=True)
        g.target(1.0, 0.0)
        self.assertEqual(g.k, 1)
        g.target(3.0, 0.0)
        self.assertEqual(g.k, 3)
        g.target(5.9, 0.0)                                              # 窗口 2 m：从 3 出发够到顶点 5（沿路径 2 m）
        self.assertEqual(g.k, 5)

    def test_pointer_never_decreases_and_stays_in_range(self):
        rng = random.Random(8)
        for case in range(60):
            path = random_grid_path(rng)
            g = guard.OracleRouteGuide(lambda s, e, path=path: path, path[-1], rng.choice((0.5, 1.5, 3.0)), nearest_in_lookahead=True)
            k_prev = 0
            for step in range(80):
                g.target(rng.uniform(-1, 8), rng.uniform(-1, 8)) if step % 3 == 0 else g.target(*path[min(step, len(path) - 1)])
                self.assertGreaterEqual(g.k, k_prev, (case, step))
                self.assertLess(g.k, len(path))
                k_prev = g.k

    def test_plan_failure_and_short_path_behave_as_before(self):
        for flag in (False, True):
            g = guard.OracleRouteGuide(lambda s, e: None, (5.0, 5.0), 3.0, nearest_in_lookahead=flag)
            self.assertEqual(g.target(0.0, 0.0), (5.0, 5.0))
            self.assertTrue(g.plan_failed)
            g = guard.OracleRouteGuide(lambda s, e: [(0.0, 0.0)], (5.0, 5.0), 3.0, nearest_in_lookahead=flag)
            self.assertEqual(g.target(0.0, 0.0), (5.0, 5.0))


if __name__ == "__main__":
    unittest.main()
