#!/usr/bin/env python3
"""guard.py（oracle 方位护栏）单测：纯 Python + numpy，不需要 torch / GPU / 学生权重。

用 3dgs-venv 或 .venv-tools 跑：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_guard.py
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import importlib
import json
import math
import os
import random
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
guard = importlib.import_module("sim.3dgs.guard")
planner = importlib.import_module("sim.3dgs.planner")
collect = importlib.import_module("sim.3dgs.collect_3dgs")
scene_mod = importlib.import_module("sim.3dgs.scene")
dynamics = importlib.import_module("sim.3dgs.dynamics")

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3


def line(p, q, step=0.1):
    """p -> q 的稠密折线：模拟 A* 栅格路径（0.1 m 一个点）。"""
    n = max(1, int(round(math.dist(p, q) / step)))
    return [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n + 1)]


class State:
    """policy 接口里的 state（= 动力学对象）：x / y / z / yaw（弧度）。"""

    def __init__(self, x, y, yaw_deg, z=2.0):
        self.x, self.y, self.z, self.yaw = x, y, z, math.radians(yaw_deg)


class TestBearingConvention(unittest.TestCase):
    """坐标与角度约定：yaw 0 = +x，正方向 = 逆时针（YAW_LEFT 使 yaw 增大），误差正 = 目标在左。"""

    def test_target_due_left_gives_plus_90_and_yaw_left(self):
        err = guard.bearing_error_deg(0.0, 0.0, 0.0, (0.0, 1.0))
        self.assertAlmostEqual(err, 90.0, places=9)
        self.assertEqual(guard.bearing_guard_action(err, 45.0), YAW_LEFT)

    def test_target_due_right_gives_minus_90_and_yaw_right(self):
        err = guard.bearing_error_deg(0.0, 0.0, 0.0, (0.0, -1.0))
        self.assertAlmostEqual(err, -90.0, places=9)
        self.assertEqual(guard.bearing_guard_action(err, 45.0), YAW_RIGHT)

    def test_target_ahead_is_zero_and_no_action(self):
        err = guard.bearing_error_deg(1.0, 2.0, 0.0, (5.0, 2.0))
        self.assertAlmostEqual(err, 0.0, places=9)
        self.assertIsNone(guard.bearing_guard_action(err, 45.0))

    def test_heading_is_respected(self):
        # 机头朝 +y（yaw 90°）：+x 方向的目标在右手边，-x 方向的在左手边。
        self.assertAlmostEqual(guard.bearing_error_deg(1.0, 1.0, 90.0, (2.0, 1.0)), -90.0, places=9)
        self.assertAlmostEqual(guard.bearing_error_deg(1.0, 1.0, 90.0, (0.0, 1.0)), 90.0, places=9)

    def test_wraps_across_180_and_stays_in_half_open_range(self):
        # 机头 170°、目标方位 -170°：该左转 20°，不是右转 340°。
        tgt = (math.cos(math.radians(-170.0)), math.sin(math.radians(-170.0)))
        self.assertAlmostEqual(guard.bearing_error_deg(0.0, 0.0, 170.0, tgt), 20.0, places=9)
        tgt = (math.cos(math.radians(170.0)), math.sin(math.radians(170.0)))
        self.assertAlmostEqual(guard.bearing_error_deg(0.0, 0.0, -170.0, tgt), -20.0, places=9)
        # 正后方恰好是 -180（区间 [-180, 180) 含左端不含右端）；任何航向组合都落在区间里。
        self.assertAlmostEqual(guard.bearing_error_deg(0.0, 0.0, 0.0, (-1.0, 0.0)), -180.0, places=9)
        for yaw in range(-360, 361, 15):
            for b in range(-360, 361, 15):
                tgt = (math.cos(math.radians(b)), math.sin(math.radians(b)))
                e = guard.bearing_error_deg(0.0, 0.0, float(yaw), tgt)
                self.assertTrue(-180.0 <= e < 180.0 + 1e-9, (yaw, b, e))

    def test_threshold_boundary_and_disabled(self):
        self.assertIsNone(guard.bearing_guard_action(44.99, 45.0))
        self.assertEqual(guard.bearing_guard_action(45.0, 45.0), YAW_LEFT)       # |err| >= 阈值
        self.assertEqual(guard.bearing_guard_action(-45.0, 45.0), YAW_RIGHT)
        self.assertEqual(guard.bearing_guard_action(-179.0, 45.0), YAW_RIGHT)
        for thr in (0.0, -1.0, -45.0):                                          # 阈值 <= 0 = 关闭
            for err in (-180.0, -90.0, 0.0, 90.0, 179.0):
                self.assertIsNone(guard.bearing_guard_action(err, thr), (thr, err))

    def test_sign_matches_planner_teacher(self):
        """护栏的转向号与 planner.OccAstarTeacher 同一套约定：随机位姿、随机目标，教师转哪边护栏就转哪边。"""
        class StubPlanner:      # OccAstarTeacher 只用到 plan / aim_point / occ.sweep；目标点由测试直接给
            def __init__(self):
                self.target, self.occ = None, self

            def plan(self, start, goal):
                return [start, self.target]

            def aim_point(self, path, x, y, lookahead):
                return self.target

            def sweep(self, a, b):
                return False, None

        stub = StubPlanner()
        teacher = planner.OccAstarTeacher(stub, {"x": 1e4, "y": 1e4})           # waypoint 远在天边，不触发 goal-stop
        tol, rng, seen = teacher.tol, random.Random(0), set()
        for _ in range(400):
            x, y, yaw = rng.uniform(-9, 9), rng.uniform(-9, 9), rng.uniform(-540, 540)
            stub.target = (x + rng.uniform(-6, 6), y + rng.uniform(-6, 6))
            if math.dist((x, y), stub.target) < 0.5:
                continue
            err = guard.bearing_error_deg(x, y, yaw, stub.target)
            if abs(abs(err) - tol) < 0.5:
                continue                                                         # 教师是 > tol，护栏是 >= 阈值，边界上不比
            want, _ = teacher(None, None, State(x, y, yaw))
            got = guard.bearing_guard_action(err, tol)
            self.assertEqual(got if got is not None else FORWARD_SLOW, want, (x, y, yaw, stub.target, err))
            seen.add(want)
        self.assertEqual(seen, {FORWARD_SLOW, YAW_LEFT, YAW_RIGHT})

    def test_sign_matches_cf_blackbox_dynamics(self):
        """动力学一侧：YAW_LEFT 使 yaw 增大（逆时针），按护栏的转向动作闭环后误差收敛到阈值以内。"""
        params = dynamics.CfBlackboxParams.from_json(None)
        for target, first in (((0.0, 5.0), YAW_LEFT), ((0.0, -5.0), YAW_RIGHT)):
            d = dynamics.CfBlackbox(params, x=0.0, y=0.0, yaw_deg=0.0)
            err0 = guard.bearing_error_deg(d.x, d.y, math.degrees(d.yaw), target)
            self.assertEqual(guard.bearing_guard_action(err0, 45.0), first)
            for _ in range(6):
                err = guard.bearing_error_deg(d.x, d.y, math.degrees(d.yaw), target)
                a = guard.bearing_guard_action(err, 20.0)
                if a is None:
                    break
                d.step(a, 1.0)
            err = guard.bearing_error_deg(d.x, d.y, math.degrees(d.yaw), target)
            self.assertLess(abs(err), 20.0, ("闭环后航向误差应收进阈值", target, err))


class TestRouteGuide(unittest.TestCase):
    def test_plans_once_from_first_pose_to_waypoint(self):
        calls = []

        def plan(start, goal):
            calls.append((start, goal))
            return line(start, goal)

        g = guard.OracleRouteGuide(plan, (10.0, 0.0), 3.0)
        self.assertEqual(calls, [])                        # 构造时不规划：实际出生位姿要等第一次 target()
        g.target(0.3, 0.1)
        for x in (0.6, 1.0, 1.5, 2.0, 2.5):
            g.target(x, 0.0)
        self.assertEqual(calls, [((0.3, 0.1), (10.0, 0.0))])   # 只规划一次，之后不重规划

    def test_target_is_lookahead_beyond_nearest_point(self):
        g = guard.OracleRouteGuide(lambda s, e: line((0.0, 0.0), (10.0, 0.0)), (10.0, 0.0), 3.0)
        tx, ty = g.target(2.0, 0.3)                       # 最近点 (2.0, 0)，再往前 3 m
        self.assertAlmostEqual(tx, 5.0, places=6)
        self.assertAlmostEqual(ty, 0.0, places=6)

    def test_end_of_path_returns_the_waypoint(self):
        wp = (10.2, 0.1)                                  # 与路径末格中心不同：返回的必须是 waypoint 本身
        g = guard.OracleRouteGuide(lambda s, e: line((0.0, 0.0), (10.0, 0.0)), wp, 3.0)
        g.target(0.0, 0.0)
        self.assertEqual(g.target(8.0, 0.0), wp)           # 剩余路径 2 m < 前视 3 m
        self.assertEqual(g.target(9.9, 0.0), wp)

    # U 形路径：A 段 x=0 向北，绕过 y=6 的墙端，B 段 x=1 向南，两段相距 1 m（路径自相近）。
    U_PATH = line((0.0, 0.0), (0.0, 6.0)) + line((0.0, 6.0), (1.0, 6.0))[1:] + line((1.0, 6.0), (1.0, 0.0))[1:]
    N_LEG_A = len(line((0.0, 0.0), (0.0, 6.0)))

    def test_nearest_index_never_decreases(self):
        g = guard.OracleRouteGuide(lambda s, e: self.U_PATH, (1.0, 0.0), 1.0)
        for p in line((0.0, 0.0), (0.0, 6.0), 0.25) + line((0.0, 6.0), (1.0, 6.0), 0.25) + line((1.0, 6.0), (1.0, 5.0), 0.25):
            g.target(*p)                                   # 机体沿路径走过拐角，进入 B 段
        k = g.k
        self.assertGreater(k, self.N_LEG_A)
        tx, ty = g.target(0.0, 1.0)                        # 位置突然离 A 段起点最近：索引不许回跳
        self.assertGreaterEqual(g.k, k)
        self.assertAlmostEqual(tx, 1.0, places=6)          # 前视点仍在 B 段（x=1），沿路径往南
        self.assertLess(ty, 5.0)

    def test_does_not_jump_ahead_to_a_nearby_later_leg(self):
        # 在 A 段偏向 B 段 0.6 m：B 段更近（0.4 m），但最近点不能跳到 B 段（那会让前视点朝回指，穿过墙）。
        g = guard.OracleRouteGuide(lambda s, e: self.U_PATH, (1.0, 0.0), 3.0)
        for p in line((0.0, 0.0), (0.0, 1.5), 0.25):
            g.target(*p)
        tx, ty = g.target(0.6, 2.0)
        self.assertLess(g.k, self.N_LEG_A)
        self.assertAlmostEqual(tx, 0.0, places=6)
        self.assertAlmostEqual(ty, 5.0, places=6)          # A 段上最近点 (0, 2) 往前 3 m

    def test_plan_failure_falls_back_to_waypoint_without_replanning(self):
        calls = []

        def plan(start, goal):
            calls.append(start)
            return None

        g = guard.OracleRouteGuide(plan, (5.0, 5.0), 3.0)
        self.assertEqual(g.target(0.0, 0.0), (5.0, 5.0))
        self.assertEqual(g.target(1.0, 1.0), (5.0, 5.0))
        self.assertTrue(g.plan_failed)
        self.assertEqual(len(calls), 1)

    def test_rejects_non_positive_lookahead(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                guard.OracleRouteGuide(lambda s, e: line(s, e), (1.0, 0.0), bad)

    def test_spec_names_are_aliases_of_the_oracle_classes(self):
        self.assertIs(guard.RouteGuide, guard.OracleRouteGuide)
        self.assertIs(guard.GuardedPolicy, guard.OracleGuardedPolicy)

    def test_real_occ_planner_goes_around_a_wall_where_goal_bearing_does_not(self):
        """占据 A* 的真路径：一堵 x=5、y∈[-3,1] 的墙挡在 (2,0) 与 (8,0) 之间。
        目标方位护栏正对墙（误差 0，什么都不做）；路线护栏看到的是绕 +y 端的前视点（目标在左，误差 > 0）。"""
        Occupancy = importlib.import_module("sim.3dgs.collision").Occupancy
        ys = np.linspace(-3.0, 1.0, 21)
        occ = Occupancy(np.array([(5.0, y, 2.0) for y in ys]), np.tile([0.01, 0.2, 0.3], (len(ys), 1)), None,
                        np.full(len(ys), 0.95), z_range=(1.65, 2.35), voxel_m=0.1, body_radius_m=0.102,
                        body_half_height_m=0.028, pad_m=4.0)
        pl = planner.OccPlanner(occ)
        g = guard.OracleRouteGuide(pl.plan, (8.0, 0.0), 3.0)
        tx, ty = g.target(2.0, 0.0)
        self.assertGreater(guard.bearing_error_deg(2.0, 0.0, 0.0, (tx, ty)), 10.0)
        self.assertAlmostEqual(guard.bearing_error_deg(2.0, 0.0, 0.0, (8.0, 0.0)), 0.0, places=9)
        self.assertEqual(g.target(7.8, 0.1), (8.0, 0.0))   # 末段直指 waypoint
        self.assertFalse(g.plan_failed)


class Fixed:
    """内层策略：永远出同一个动作；记录 reset 次数。name 与学生 / 盲前进一样会被 run_episode 读。"""

    def __init__(self, action, name="forward"):
        self.action, self.name, self.resets, self.calls = action, name, 0, 0

    def __call__(self, rgb, depth, state):
        self.calls += 1
        return self.action, 1.25

    def reset(self):
        self.resets += 1


class TestGuardedPolicy(unittest.TestCase):
    TARGET = (0.0, 5.0)        # 机体在原点朝 +x：目标正左（err = +90）

    def test_disabled_threshold_is_pass_through(self):
        for thr in (0.0, -10.0):
            for a in (SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT):
                g = guard.OracleGuardedPolicy(Fixed(a), lambda x, y: self.TARGET, thr)
                self.assertEqual(g(None, None, State(0.0, 0.0, 0.0)), (a, 1.25))
                self.assertEqual(g.n_rewritten, 0)

    def test_rewrites_only_beyond_threshold_and_toward_target(self):
        g = guard.OracleGuardedPolicy(Fixed(FORWARD_SLOW), lambda x, y: self.TARGET, 45.0)
        # 目标正左：|err|=90 >= 45，前进被改写成左转
        self.assertEqual(g(None, None, State(0.0, 0.0, 0.0)), (YAW_LEFT, 1.25))
        # 机头已转到 60°：err = 30 < 45，保留内层动作
        self.assertEqual(g(None, None, State(0.0, 0.0, 60.0)), (FORWARD_SLOW, 1.25))
        # 内层要右转（背离目标）也被改成左转；机头朝 +y 右侧很远时反过来
        g2 = guard.OracleGuardedPolicy(Fixed(YAW_RIGHT), lambda x, y: self.TARGET, 45.0)
        self.assertEqual(g2(None, None, State(0.0, 0.0, 0.0))[0], YAW_LEFT)
        g3 = guard.OracleGuardedPolicy(Fixed(FORWARD_SLOW), lambda x, y: self.TARGET, 45.0)
        self.assertEqual(g3(None, None, State(0.0, 0.0, 170.0))[0], YAW_RIGHT)   # err = 90 - 170 = -80

    def test_counts_rewrites_and_records_every_step(self):
        g = guard.OracleGuardedPolicy(Fixed(YAW_LEFT), lambda x, y: self.TARGET, 45.0)
        g(None, None, State(0.0, 0.0, 0.0))        # 触发，但内层本来就左转：不算改写
        g(None, None, State(0.0, 0.0, 80.0))       # err=10，不触发
        g.inner.action = FORWARD_SLOW
        g(None, None, State(0.0, 0.0, 0.0))        # 触发且改写
        self.assertEqual((g.n_steps, g.n_triggered, g.n_rewritten), (3, 2, 1))
        self.assertEqual([r["step"] for r in g.trace], [0, 1, 2])
        self.assertEqual([r["triggered"] for r in g.trace], [True, False, True])
        self.assertEqual([r["rewritten"] for r in g.trace], [False, False, True])
        self.assertEqual([r["action"] for r in g.trace], [YAW_LEFT, YAW_LEFT, YAW_LEFT])
        self.assertEqual([r["inner"] for r in g.trace], [YAW_LEFT, YAW_LEFT, FORWARD_SLOW])
        self.assertAlmostEqual(g.trace[0]["err_deg"], 90.0, places=2)
        self.assertEqual(g.trace[0]["target"], list(self.TARGET))
        json.dumps(g.trace)                         # 旁路文件要能直接写 jsonl

    def test_run_episode_contract(self):
        """run_episode 靠 name / 属性探测决定行为：name 要与内层一致，且不能冒出 stop_radius 等教师属性。"""
        for name in ("forward", "student"):
            g = guard.OracleGuardedPolicy(Fixed(FORWARD_SLOW, name=name), lambda x, y: self.TARGET, 45.0)
            self.assertEqual(g.name, name)
            for attr in ("stop_radius", "prev", "n_plan_fail", "n_expert_exec", "last_expert", "planner_name"):
                self.assertFalse(hasattr(g, attr), attr)

    def test_reset_clears_episode_state_and_forwards(self):
        inner = Fixed(FORWARD_SLOW)
        g = guard.OracleGuardedPolicy(inner, lambda x, y: self.TARGET, 45.0)
        g(None, None, State(0.0, 0.0, 0.0))
        g.reset()
        self.assertEqual((g.n_steps, g.n_triggered, g.n_rewritten, g.trace), (0, 0, 0, []))
        self.assertEqual(inner.resets, 1)
        # 内层没有 reset（StudentAdapter / ForwardPolicy）也不能报错
        class Bare:
            name = "forward"

            def __call__(self, rgb, depth, state):
                return FORWARD_SLOW, None

        guard.OracleGuardedPolicy(Bare(), lambda x, y: self.TARGET, 45.0).reset()


class TestGuardInRunEpisode(unittest.TestCase):
    """真 run_episode（mock 渲染 + cf_blackbox）：不改 run_episode 就能用；关闭时与不包装逐位一致。"""

    def _run(self, tmp, policy, spawn, steps=8, hold=1.0):
        renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(
            scene_mod.load_scene(os.path.join(tmp, "scene", "scene.json")), True)
        scene = scene_mod.load_scene(os.path.join(tmp, "scene", "scene.json"))
        scene["spawn"].update({"x": spawn[0], "y": spawn[1], "yaw_deg": spawn[2]})
        occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)
        params = dynamics.CfBlackboxParams.from_json(None)
        identity = collect.scene_identity(scene)
        out = os.path.join(tmp, "ep%d" % len(os.listdir(tmp)))
        manifest, err = collect.run_episode(scene, out, renderer, occ, params, policy, steps, hold,
                                            split=identity["split"], identity=identity)
        self.assertIsNone(err)
        def rows(name):
            with open(os.path.join(out, name), encoding="utf-8") as f:
                return [json.loads(x) for x in f]

        return manifest, rows("actions.jsonl"), rows("oracle.jsonl")

    def test_closed_loop_turns_toward_target_and_keeps_heading_inside_cone(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(os.path.join(tmp, "scene"), seed=2, length=12.0)
            for target, first, sign in (((1.0, 3.0), YAW_LEFT, 1), ((1.0, -3.0), YAW_RIGHT, -1)):
                g = guard.OracleGuardedPolicy(collect.ForwardPolicy(), lambda x, y, t=target: t, 45.0)
                man, actions, oracle = self._run(tmp, g, (1.0, 0.0, 0.0), steps=4)
                self.assertEqual(actions[0]["action_executed"], first)
                self.assertGreater(sign * (oracle[0]["pose_gt"]["yaw_deg"] - actions[0]["yaw_before"]), 5.0)
                self.assertEqual(man["teacher"], "forward")                   # name 沿用内层
                self.assertEqual(man["executed_by"], "forward")
                self.assertEqual(len(g.trace), len(actions))                  # 旁路逐步对齐 actions.jsonl
                self.assertEqual([r["action"] for r in g.trace], [a["action_executed"] for a in actions])
                last = oracle[-1]["pose_gt"]
                err = guard.bearing_error_deg(last["x"], last["y"], last["yaw_deg"], target)
                self.assertLess(abs(err), 45.0 + 35.0, err)                   # 容许一步过冲

    def test_disabled_guard_is_bit_for_bit_the_inner_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(os.path.join(tmp, "scene"), seed=2, length=12.0)
            _, a0, o0 = self._run(tmp, collect.ForwardPolicy(), (1.0, 0.0, 20.0))
            g = guard.OracleGuardedPolicy(collect.ForwardPolicy(), lambda x, y: (1.0, 3.0), 0.0)
            _, a1, o1 = self._run(tmp, g, (1.0, 0.0, 20.0))
            self.assertEqual(o0, o1)
            keys = ("step", "action_executed", "action_expert", "pose_before", "yaw_before", "pose_gt", "collision")
            self.assertEqual([{k: r[k] for k in keys} for r in a0], [{k: r[k] for k in keys} for r in a1])
            self.assertEqual(g.n_rewritten, 0)


if __name__ == "__main__":
    unittest.main()
