#!/usr/bin/env python3
"""OccAstarTeacher 航向容差随 hold 推导（任务卡 W1-11）的单测：纯 Python + numpy，不需要 torch / GPU / 学生权重。

背景：hold 1.0 下一个转角步实测 27–35°，死区宽 2 × 12° = 24° 小于一个转角步，航向误差落在 12°–16° 时转一步就过冲到
另一侧的 12° 之外，再转回来，左右来回转（极限环）。容差改为 max(12°, 0.55 × 名义转角步)，名义转角步 = yaw_rate_rps × hold。

用 3dgs-venv 或 .venv-tools 跑（要 numpy）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_teacher_tol.py
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

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
MIN_RUN = 6                    # 「交替段」= 左右严格交替连续 >= 6 步（sage3d 报告 §4.5 的判据）
NOMINAL_STEP_HOLD1 = math.degrees(0.6 * 1.0)      # 34.38°：e1_mini.json executor.yaw_rate_rps × hold 1.0


class State:
    """policy 接口里的 state（= 动力学对象）：x / y / z / yaw（弧度）。"""

    def __init__(self, x, y, yaw_deg, z=2.0):
        self.x, self.y, self.z, self.yaw = x, y, z, math.radians(yaw_deg)


class StubPlanner:
    """OccAstarTeacher 只用到 plan / aim_point / occ.sweep：路径 = 起点到 waypoint 的直线，瞄准点 = waypoint。"""

    def __init__(self, blocked=False, no_path=False):
        self.blocked, self.no_path, self.occ = blocked, no_path, self

    def plan(self, start, goal):
        return None if self.no_path else [start, goal]

    def aim_point(self, path, x, y, lookahead):
        return path[-1]

    def sweep(self, a, b):
        return self.blocked, None


def const_step(deg):
    return lambda prev, act: deg


def measured_step_hold1(prev, act):
    """hold 1.0 的实测转角步（exp-bearing-guard / exp-sage3d-pilot 教师臂的 dyaw_deg，按上一步动作分）：
    直飞或悬停之后 31.6°，同向连转 34.4°（= 名义值），反向 27.5°（偏航角速度要先反号）。"""
    if prev in (YAW_LEFT, YAW_RIGHT):
        return 34.4 if prev == act else 27.5
    return 31.6


def measured_step_hold05(prev, act):
    """hold 0.5：直飞后首步 14.5°，连转 17°（planner.py 注释里 cf_sitl 的实测）。"""
    return 17.0 if prev in (YAW_LEFT, YAW_RIGHT) else 14.5


def rollout(factory, e0_deg, turn_step, steps=30, forward_m=0.3, target_m=1000.0):
    """运动学闭环：目标在 target_m 远处、方位比机头偏 e0（正 = 左），所以前进几步误差基本不变，只有转向改变它。
    FORWARD 沿机头前进 forward_m，YAW 转 turn_step(上一步动作, 本步动作) 度，SAFE 不动。返回动作序列。"""
    wp = {"x": target_m * math.cos(math.radians(e0_deg)), "y": target_m * math.sin(math.radians(e0_deg))}
    teacher = factory(wp)
    x = y = yaw = 0.0
    acts, prev = [], FORWARD_SLOW
    for _ in range(steps):
        a, _ms = teacher(None, None, State(x, y, yaw))
        acts.append(a)
        if a == FORWARD_SLOW:
            x += forward_m * math.cos(math.radians(yaw))
            y += forward_m * math.sin(math.radians(yaw))
        elif a in (YAW_LEFT, YAW_RIGHT):
            yaw += turn_step(prev, a) * (1.0 if a == YAW_LEFT else -1.0)
        prev = a
    return acts


def longest_alternation(acts):
    """最长的「左右严格交替」转向段的步数（单独一次转向算 1）。"""
    best = run = 0
    for i, a in enumerate(acts):
        if a not in (YAW_LEFT, YAW_RIGHT):
            run = 0
        elif run and acts[i - 1] in (YAW_LEFT, YAW_RIGHT) and acts[i - 1] != a:
            run += 1
        else:
            run = 1
        best = max(best, run)
    return best


def old_teacher(wp):
    return planner.OccAstarTeacher(StubPlanner(), wp)                 # 不给 hold = 旧行为（容差 12°）


def new_teacher(hold_s):
    return lambda wp: planner.OccAstarTeacher(StubPlanner(), wp, hold_s=hold_s)


class TestYawTolForHold(unittest.TestCase):
    def test_values(self):
        self.assertEqual(planner.yaw_tol_deg(0.5), 12.0)                 # 0.55 × 17.2° = 9.5° < 12°：下限生效
        self.assertEqual(planner.yaw_tol_deg(0.25), 12.0)
        self.assertAlmostEqual(planner.yaw_tol_deg(1.0), 0.55 * NOMINAL_STEP_HOLD1, places=9)
        self.assertAlmostEqual(planner.yaw_tol_deg(1.0), 18.9076, places=4)    # 0.55 × degrees(0.6 rad/s × 1 s)
        self.assertAlmostEqual(planner.yaw_tol_deg(2.0), 0.55 * 2 * NOMINAL_STEP_HOLD1, places=9)

    def test_floor_then_linear_and_monotonic(self):
        holds = np.arange(0.1, 3.0, 0.05)
        tols = [planner.yaw_tol_deg(float(h)) for h in holds]
        self.assertTrue(all(b >= a for a, b in zip(tols, tols[1:])))
        self.assertTrue(all(t >= planner.YAW_TOL_DEG for t in tols))
        # 下限与线性段的交点：0.55 × 0.6 rad/s × h = 12° 在 h ≈ 0.635 s
        self.assertEqual(planner.yaw_tol_deg(0.63), 12.0)
        self.assertGreater(planner.yaw_tol_deg(0.64), 12.0)

    def test_rejects_bad_hold(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                planner.yaw_tol_deg(bad)

    def test_yaw_rate_constant_is_the_params_value(self):
        """planner 不读 json（不依赖路径与 cf_sitl_proc），常量与 e1_mini.json 的 executor.yaw_rate_rps 靠这条测试锁住。"""
        with open(os.path.join(ROOT, "sim", "dynamics", "e1_mini.json"), encoding="utf-8") as f:
            ex = json.load(f)["executor"]["value"]
        self.assertEqual(planner.YAW_RATE_RPS, ex["yaw_rate_rps"])

    def test_threshold_is_at_least_half_the_nominal_step(self):
        """容差不小于名义转角步的一半：否则最大的转角步（同向连转）也会让误差一步过冲到对侧容差之外。"""
        for hold in (0.5, 0.75, 1.0, 1.5):
            nominal = math.degrees(planner.YAW_RATE_RPS * hold)
            self.assertGreaterEqual(2 * planner.yaw_tol_deg(hold), nominal, hold)


class TestTeacherTolerance(unittest.TestCase):
    def test_constructor_defaults(self):
        wp = {"x": 1e4, "y": 0.0}
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), wp).tol, 12.0)                    # 不给 hold：旧行为
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), wp, hold_s=0.5).tol, 12.0)
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), wp, hold_s=1.0).tol, planner.yaw_tol_deg(1.0))

    def test_explicit_tol_wins_over_hold(self):
        wp = {"x": 1e4, "y": 0.0}
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), wp, tol_deg=12.0, hold_s=1.0).tol, 12.0)
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), wp, tol_deg=25.0, hold_s=0.5).tol, 25.0)
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), wp, tol_deg=18.0).tol, 18.0)

    def test_bad_hold_rejected(self):
        for bad in (0.0, -0.5, float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                planner.OccAstarTeacher(StubPlanner(), {"x": 1.0, "y": 0.0}, hold_s=bad)

    def test_card_scenario_14deg_33deg_step(self):
        """任务卡的场景：每步转 33°、目标在偏 14° 处。旧容差 12° 左右交替 6 步以上，新容差（hold 1.0 → 18.9°）一步不转。"""
        step = const_step(33.0)
        old = rollout(old_teacher, 14.0, step)
        self.assertGreaterEqual(longest_alternation(old), MIN_RUN, old)
        self.assertEqual(old[:6], [YAW_LEFT, YAW_RIGHT] * 3)
        new = rollout(new_teacher(1.0), 14.0, step)
        self.assertLess(longest_alternation(new), MIN_RUN, new)
        self.assertEqual(new, [FORWARD_SLOW] * 30)                       # 14° < 18.9°：直接前进
        explicit_old = rollout(lambda wp: planner.OccAstarTeacher(StubPlanner(), wp, tol_deg=12.0, hold_s=1.0), 14.0, step)
        self.assertEqual(explicit_old, old)                              # 显式 12° 照旧

    def test_no_limit_cycle_for_any_initial_error(self):
        """e0 扫 [-180, 180]：新容差在三种转角步模型（定值 33°、名义值 34.4°、实测的分上下文步长）下都不出现 6 步以上的交替；
        旧容差在同一批 e0 里有检出（模型有检出力），其中包含误差直接落进 (tol, 步长 − tol) 的 e0。
        远处的 e0 也会被旧容差带进环：一路同向转，转到误差落进那一带为止，所以不检查旧容差的 e0 范围。"""
        e0s = [i * 0.5 for i in range(-360, 361)]
        for name, step in (("const33", const_step(33.0)), ("measured1.0", measured_step_hold1),
                           ("nominal34.4", const_step(NOMINAL_STEP_HOLD1))):
            bad_new = [e for e in e0s if longest_alternation(rollout(new_teacher(1.0), e, step, steps=40)) >= MIN_RUN]
            self.assertEqual(bad_new, [], name)
            bad_old = [e for e in e0s if longest_alternation(rollout(old_teacher, e, step, steps=40)) >= MIN_RUN]
            self.assertTrue(any(12.0 < abs(e) < 20.0 for e in bad_old), (name, len(bad_old)))   # 三种转角步都大于 24° 的死区宽

    def test_hold_half_second_is_stepwise_identical_to_old(self):
        """hold 0.5：容差仍是 12°，每一步动作与改动前（不给 hold）相同：e0 扫描 × 两种转角步模型，再加随机状态（含前向被堵）。"""
        e0s = [i * 0.5 for i in range(-360, 361)]
        for step in (const_step(17.2), measured_step_hold05):
            for e0 in e0s:
                self.assertEqual(rollout(new_teacher(0.5), e0, step, steps=25), rollout(old_teacher, e0, step, steps=25), e0)
        rng = random.Random(7)
        for _ in range(400):
            blocked = rng.random() < 0.3
            x, y, yaw = rng.uniform(-9, 9), rng.uniform(-9, 9), rng.uniform(-540, 540)
            wp = {"x": x + rng.uniform(-6, 6), "y": y + rng.uniform(-6, 6)}
            old = planner.OccAstarTeacher(StubPlanner(blocked=blocked), wp)
            new = planner.OccAstarTeacher(StubPlanner(blocked=blocked), wp, hold_s=0.5)
            self.assertEqual(new(None, None, State(x, y, yaw)), old(None, None, State(x, y, yaw)), (x, y, yaw, wp, blocked))

    def test_no_hold_default_is_the_old_12_degrees(self):
        """调用方不传 hold 时不猜：退回 12°，旧调用方（test_guard、驱动脚本）的行为不变。"""
        self.assertEqual(planner.OccAstarTeacher(StubPlanner(), {"x": 1.0, "y": 0.0}).tol, planner.YAW_TOL_DEG)
        self.assertEqual(planner.YAW_TOL_DEG, 12.0)


class TestVersionAndBranch(unittest.TestCase):
    def test_version_distinguishes_old_and_new_tolerance(self):
        wp = {"x": 1e4, "y": 0.0}
        self.assertEqual(planner.TEACHER_VERSION, "uavrgb.teacher_occ.v1")                          # 旧数据全是它，字符串不动
        self.assertNotEqual(planner.TEACHER_VERSION_HOLD_TOL, planner.TEACHER_VERSION)
        v = lambda **kw: planner.OccAstarTeacher(StubPlanner(), wp, **kw).version                    # noqa: E731
        self.assertEqual(v(), planner.TEACHER_VERSION)                                              # 12°
        self.assertEqual(v(hold_s=0.5), planner.TEACHER_VERSION)                                    # 12°：与旧教师逐步相同，版本也不变
        self.assertEqual(v(tol_deg=12.0, hold_s=1.0), planner.TEACHER_VERSION)
        self.assertEqual(v(hold_s=1.0), planner.TEACHER_VERSION_HOLD_TOL)                           # 18.9°
        self.assertEqual(v(tol_deg=18.0), planner.TEACHER_VERSION_HOLD_TOL)
        # 只有容差恰好 = 12° 才是 v1：显式给更小的容差（变异检查里 `<=` 会把它们误标成 v1）和 12° 附近的值都是 v2
        for tol in (8.0, 11.999, 12.001):
            self.assertEqual(v(tol_deg=tol), planner.TEACHER_VERSION_HOLD_TOL, tol)

    def test_last_branch_names_the_rule_that_fired(self):
        """last_branch 记本步动作是哪条规则给的：交替段里的转向是「误差超容差」还是「前向被堵时按误差符号转」，离线复算时要分开。"""
        def call(e0, blocked=False, no_path=False, wp_dist=1000.0):
            wp = {"x": wp_dist * math.cos(math.radians(e0)), "y": wp_dist * math.sin(math.radians(e0))}
            t = planner.OccAstarTeacher(StubPlanner(blocked=blocked, no_path=no_path), wp, hold_s=1.0)
            a, _ = t(None, None, State(0.0, 0.0, 0.0))
            return a, t.last_branch
        self.assertEqual(call(40.0), (YAW_LEFT, "turn_err"))
        self.assertEqual(call(-40.0), (YAW_RIGHT, "turn_err"))
        self.assertEqual(call(5.0), (FORWARD_SLOW, "forward"))
        self.assertEqual(call(5.0, blocked=True), (YAW_LEFT, "turn_blocked"))          # err >= 0 左转
        self.assertEqual(call(-5.0, blocked=True), (YAW_RIGHT, "turn_blocked"))
        self.assertEqual(call(40.0, blocked=True), (YAW_LEFT, "turn_err"))             # 误差超容差优先，不看前向
        self.assertEqual(call(0.0, wp_dist=0.3), (SAFE_NEUTRAL, "stop"))               # goal-stop
        self.assertEqual(call(0.0, no_path=True), (SAFE_NEUTRAL, "plan_fail"))


def _mock_env(tmp, seed=2):
    """合成走廊 + mock 渲染器 + cf_blackbox，直接调 run_episode 用（同 test_collect_eval._mock_env）。"""
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
    def _episode(self, tmp, policy, hold, name, dagger=None):
        scene, renderer, occ, params, identity = _mock_env(tmp)
        scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}              # 走廊默认 waypoint x=30 在尽头墙里，A* 无路
        pol = policy(scene, occ)
        m, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, pol, 3, hold,
                                     dagger=dagger, split=identity["split"], identity=identity)
        self.assertIsNone(err)
        return m

    def test_teacher_manifest_records_tolerance_and_version(self):
        mk = lambda hold_s: (lambda scene, occ: planner.OccAstarTeacher(                    # noqa: E731
            planner.OccPlanner(occ), scene["waypoint"], **({} if hold_s is None else {"hold_s": hold_s})))
        with tempfile.TemporaryDirectory() as tmp:
            m1 = self._episode(tmp, mk(1.0), 1.0, "h10")
            self.assertAlmostEqual(m1["teacher_yaw_tol_deg"], planner.yaw_tol_deg(1.0), places=9)
            self.assertEqual(m1["teacher_version"], planner.TEACHER_VERSION_HOLD_TOL)
            m5 = self._episode(tmp, mk(0.5), 0.5, "h05")
            self.assertEqual((m5["teacher_yaw_tol_deg"], m5["teacher_version"]), (12.0, planner.TEACHER_VERSION))
            m0 = self._episode(tmp, mk(None), 1.0, "nohold")                  # 调用方没传 hold：12°，版本 v1（旧驱动脚本的行为）
            self.assertEqual((m0["teacher_yaw_tol_deg"], m0["teacher_version"]), (12.0, planner.TEACHER_VERSION))

    def test_dagger_manifest_records_expert_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            mk = lambda scene, occ: collect.DaggerPolicy(                                    # noqa: E731
                FakeStudent(), planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"], hold_s=1.0),
                0.0, random.Random(0))
            m = self._episode(tmp, mk, 1.0, "dagger", dagger={"executed_by": "student", "dagger_round": 1, "beta": 0.0})
            self.assertAlmostEqual(m["teacher_yaw_tol_deg"], planner.yaw_tol_deg(1.0), places=9)
            self.assertEqual(m["teacher_version"], planner.TEACHER_VERSION_HOLD_TOL)

    def test_non_astar_policies_manifest_has_no_tolerance_key(self):
        """forward / 学生 / 深度教师的 manifest 逐键不变：不出现 teacher_yaw_tol_deg。"""
        with tempfile.TemporaryDirectory() as tmp:
            for name, mk in (("fwd", lambda scene, occ: collect.ForwardPolicy()),
                             ("stu", lambda scene, occ: collect.StudentAdapter(FakeStudent())),
                             ("depth", lambda scene, occ: collect.DepthTeacher())):
                m = self._episode(tmp, mk, 0.5, name)
                self.assertNotIn("teacher_yaw_tol_deg", m, name)

    def test_cli_astar_passes_hold_to_teacher(self):
        """collect_3dgs --policy astar：构造教师的调用方把 --hold 传进去。"""
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene = scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene["routes"] = [{"name": "r0", "spawn": scene["spawn"], "waypoint": scene["waypoint"]},
                               {"name": "r1", "spawn": {"x": 3.0, "y": 0.0, "z": 2.0, "yaw_deg": 90.0},
                                "waypoint": {"x": 10.0, "y": 0.0, "z": 2.0}, "max_steps": 4}]
            scene_json = os.path.join(scene_dir, "scene.json")
            scene_mod.save_scene({k: v for k, v in scene.items()
                                  if not k.startswith("_") and k not in ("path", "dir", "ply_path")}, scene_json)
            got = {}
            for hold in ("0.5", "1.0"):
                out = os.path.join(tmp, "h" + hold)
                r = subprocess.run([sys.executable, "-m", "sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", out,
                                    "--mock-renderer", "--dynamics", "cf_blackbox", "--policy", "astar", "--route", "1",
                                    "--hold", hold], cwd=ROOT, capture_output=True, text=True, timeout=180)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                with open(os.path.join(out, "ep_00000", "episode_manifest.json"), encoding="utf-8") as f:
                    m = json.load(f)
                got[hold] = (m["hold_s"], m["teacher_yaw_tol_deg"], m["teacher_version"])
            self.assertEqual(got["0.5"], (0.5, 12.0, planner.TEACHER_VERSION))
            self.assertEqual(got["1.0"][0], 1.0)
            self.assertAlmostEqual(got["1.0"][1], planner.yaw_tol_deg(1.0), places=9)
            self.assertEqual(got["1.0"][2], planner.TEACHER_VERSION_HOLD_TOL)


if __name__ == "__main__":
    unittest.main()
