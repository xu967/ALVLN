#!/usr/bin/env python3
"""规则 v1（计划 D6：短期障碍记忆 + 候选航向选择）的纯函数测试与离线运动学模拟。

不起 Gazebo。坐标与角度约定同 rule_policy / teacher（x 前、y 左；yaw 度，0 = +x，正 = 左转；方位正 = 左）。
覆盖：障碍点、机体系 ↔ 世界系换算、记忆过期、候选航向通道 clear(θ)、决策顺序与各条目、n_cap、81° 点数、
v1 不输出 SAFE_NEUTRAL 且 FORWARD 只在 clear(0) 时输出、v0 不变；以及 rule_sim（前进 0.25 m / 转 26° 的运动学模拟）上
单柱绕行与 20 个 train 分区几何的离线检查（过线数 < 18/20 就先停下，不起 Gazebo 正式实验）。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_rule_v1.py
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect
import rule_policy as rp
import rule_sim
import scene_course
import split_manifest as sm
from rule_policy import (COL_DEG, RULE_V0, RULE_V1, V1_ENTRIES, SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT,
                         column_distances, corridor_clear, obstacle_points, points_to_body, points_to_world,
                         rule_v0_entry, rule_v1_entry, rule_v1_step, update_memory)

CAM = 0.045          # e1_mini.json camera_mount.xyz[0]
ALPHA = RULE_V1["yaw_step_deg"]
OPEN20 = [4.0] * 20


def square(x, y, size=0.4):
    return {"name": "p", "x": x, "y": y, "z": 1.5, "size": [size, size, 3.0]}


class ParamsTests(unittest.TestCase):
    def test_frozen_values(self):
        self.assertEqual(RULE_V1, {"r_safe": 0.35, "d_look": 1.2, "yaw_step_deg": 26.0, "mem_steps": 8,
                                   "n_cap": 3, "tie_frac": 0.10, "sum_clip": 4.0})

    def test_v0_params_untouched(self):
        self.assertEqual(RULE_V0, {"r_safe": 0.35, "d_look": 1.2, "n_max": 2, "side_near": 0.5, "tie_frac": 0.10,
                                   "sum_clip": 4.0})

    def test_v0_decision_untouched(self):
        d = column_distances(0.0, 0.0, 0.0, [square(0.8, 0.0)])
        self.assertEqual(rule_v0_entry(d, 0, None), (YAW_LEFT, "R2-left"))
        self.assertEqual(rule_v0_entry(OPEN20, 0, None), (FORWARD_SLOW, "R3"))
        self.assertEqual(rule_v0_entry(OPEN20, 2, None), (YAW_RIGHT, "R1"))


class ObstaclePointsTests(unittest.TestCase):
    def test_point_per_column_with_column_centre_bearing(self):
        d = [4.0] * 20
        d[0], d[9], d[10], d[19] = 1.0, 2.0, 2.5, 3.0
        pts = obstacle_points(d, 60.0)
        self.assertEqual(len(pts), 4, "d_j == max_range 的列不出点")
        for (px, py), j in zip(pts, (0, 9, 10, 19)):
            phi = math.radians((9.5 - j) * COL_DEG)
            self.assertAlmostEqual(px, d[j] * math.cos(phi) + CAM, places=9)
            self.assertAlmostEqual(py, d[j] * math.sin(phi), places=9)
        self.assertGreater(pts[0][1], 0, "第 0 列在最左：方位 +28.5°，y > 0")
        self.assertLess(pts[3][1], 0, "第 19 列在最右：y < 0")
        self.assertEqual(obstacle_points(OPEN20, 60.0), [])

    def test_81_deg_has_27_columns(self):
        pts = obstacle_points([1.0] * 27, 81.0)
        self.assertEqual(len(pts), 27)
        self.assertAlmostEqual(pts[13][1], 0.0, places=9)             # 正中间一列方位 0
        self.assertAlmostEqual(pts[13][0], 1.0 + CAM, places=9)
        self.assertAlmostEqual(math.degrees(math.atan2(pts[0][1], pts[0][0] - CAM)), 39.0, places=6)   # (13 − 0)·3°
        self.assertAlmostEqual(math.degrees(math.atan2(pts[26][1], pts[26][0] - CAM)), -39.0, places=6)

    def test_column_count_must_match_hfov(self):
        with self.assertRaises(ValueError):
            obstacle_points([1.0] * 20, 81.0)
        with self.assertRaises(ValueError):
            obstacle_points([1.0] * 27, 60.0)

    def test_points_sit_on_the_pillar_surface_in_the_world_frame(self):
        pillars = [square(3.0, 1.0)]
        x, y, yaw = 1.0, 1.0, 0.0
        d = column_distances(x, y, yaw, pillars)
        world = points_to_world(obstacle_points(d, 60.0), x, y, yaw)
        self.assertTrue(world)
        for wx, wy in world:       # 近面 x = 2.8；列取最近一条射线、方位取列中心，横向最多偏 d·sin(1.5°)
            self.assertAlmostEqual(wx, 2.8, delta=0.06)
            self.assertLessEqual(abs(wy - 1.0), 0.2 + 0.06)


class TransformTests(unittest.TestCase):
    def test_known_rotations(self):
        self.assertEqual([tuple(round(v, 9) for v in p) for p in points_to_world([(1.0, 0.0), (0.0, 1.0)], 1.0, 2.0, 90.0)],
                         [(1.0, 3.0), (0.0, 2.0)])       # 机头朝 +y：机体前 = 世界 +y，机体左 = 世界 −x
        self.assertEqual([tuple(round(v, 9) for v in p) for p in points_to_world([(1.0, 0.5)], 3.0, -1.0, 0.0)], [(4.0, -0.5)])
        body = points_to_body([(0.0, 1.0)], 0.0, 0.0, 90.0)[0]
        self.assertAlmostEqual(body[0], 1.0, places=9)           # 世界 (0,1) 对机头朝 +y 的机体正是正前方
        self.assertAlmostEqual(body[1], 0.0, places=9)

    def test_round_trip(self):
        rng = random.Random(7)
        for _ in range(200):
            x, y, yaw = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-180, 180)
            pts = [(rng.uniform(-1, 4), rng.uniform(-3, 3)) for _ in range(6)]
            back = points_to_body(points_to_world(pts, x, y, yaw), x, y, yaw)
            for (a, b), (c, e) in zip(pts, back):
                self.assertAlmostEqual(a, c, places=9)
                self.assertAlmostEqual(b, e, places=9)
            again = points_to_world(points_to_body(pts, x, y, yaw), x, y, yaw)    # 反方向也闭合
            for (a, b), (c, e) in zip(pts, again):
                self.assertAlmostEqual(a, c, places=9)
                self.assertAlmostEqual(b, e, places=9)


class MemoryTests(unittest.TestCase):
    def test_expiry_boundary(self):
        mem = update_memory([], 0, [(1.0, 0.0)], 8)
        self.assertEqual(mem, [(1.0, 0.0, 0)])
        mem8 = update_memory(mem, 8, [], 8)
        self.assertEqual(mem8, [(1.0, 0.0, 0)], "年龄 8 步 = mem_steps：还在")
        mem9 = update_memory(mem, 9, [], 8)
        self.assertEqual(mem9, [], "年龄 9 步 > mem_steps：丢弃")

    def test_new_points_always_kept_and_old_mixed(self):
        mem = update_memory([], 0, [(1.0, 0.0)], 8)
        mem = update_memory(mem, 5, [(2.0, 1.0), (3.0, 1.0)], 8)
        self.assertEqual(len(mem), 3)
        mem = update_memory(mem, 9, [(9.0, 9.0)], 8)
        self.assertEqual(sorted(mem), [(2.0, 1.0, 5), (3.0, 1.0, 5), (9.0, 9.0, 9)], "第 0 步的点过期，第 5 步的还在")

    def test_pure(self):
        mem = [(1.0, 0.0, 0)]
        update_memory(mem, 20, [(2.0, 2.0)], 8)
        self.assertEqual(mem, [(1.0, 0.0, 0)], "不改传入的列表")

    def test_remembered_point_is_converted_with_the_current_pose(self):
        # 第 0 步在原点看见正前方 1.0 m 的点；第 3 步机体前进到 (0.5, 0)：同一点现在在机体前方 0.5 m
        mem, _ = rule_v1_step([], 0, 0.0, 0.0, 0.0, [1.0 - CAM if j in (9, 10) else 4.0 for j in range(20)], 0, None)[2:]
        _a, _e, mem2, info = rule_v1_step(mem, 3, 0.5, 0.0, 0.0, OPEN20, 0, None)
        self.assertEqual(info["mem_n"], 2)
        body = points_to_body([(X, Y) for X, Y, _t in mem2], 0.5, 0.0, 0.0)
        for bx, _by in body:
            self.assertAlmostEqual(bx, 0.5, delta=0.02)


class CorridorClearTests(unittest.TestCase):
    def test_membership_boundaries(self):
        self.assertTrue(corridor_clear([], 0.0))
        self.assertFalse(corridor_clear([(1.19, 0.34)], 0.0))
        self.assertTrue(corridor_clear([(1.20, 0.0)], 0.0), "u = d_look 不算（严格小于）")
        self.assertTrue(corridor_clear([(1.0, 0.35)], 0.0), "|v| = r_safe 不算（严格小于）")
        self.assertTrue(corridor_clear([(0.0, 0.0)], 0.0), "u = 0 不算（严格大于）")
        self.assertTrue(corridor_clear([(-0.5, 0.0)], 0.0), "身后的点不挡")

    def test_candidate_heading_rotation(self):
        p = [(1.0, 0.5)]
        self.assertTrue(corridor_clear(p, 0.0), "横向 0.5 > 0.35")
        self.assertFalse(corridor_clear(p, ALPHA), "朝左转 26° 后这点正对机头")
        self.assertTrue(corridor_clear(p, -ALPHA))
        q = [(1.0, -0.5)]
        self.assertFalse(corridor_clear(q, -ALPHA))
        self.assertTrue(corridor_clear(q, ALPHA))

    def test_matches_literal_formula(self):
        rng = random.Random(11)
        for _ in range(3000):
            pts = [(rng.uniform(-0.5, 2.0), rng.uniform(-1.0, 1.0)) for _ in range(rng.randint(0, 6))]
            th = rng.choice((0.0, ALPHA, -ALPHA, rng.uniform(-90, 90)))
            c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
            blocked = any(0 < x * c + y * s < RULE_V1["d_look"] and abs(-x * s + y * c) < RULE_V1["r_safe"] for x, y in pts)
            self.assertEqual(corridor_clear(pts, th), not blocked)


class DecisionTests(unittest.TestCase):
    def entry(self, pts, n, last=None, d=None):
        return rule_v1_entry(pts, OPEN20 if d is None else d, n, last)

    BLOCK_BACK_RIGHT = (1.0, -0.5)    # 只挡「朝右转 26°」的通道
    BLOCK_BACK_LEFT = (1.0, 0.5)      # 只挡「朝左转 26°」的通道
    AHEAD = (1.0, 0.0)                # 挡当前航向，不挡左右各转 26° 的通道

    def test_forward_when_clear(self):
        self.assertEqual(self.entry([], 0), (FORWARD_SLOW, "V1-fwd"))

    def test_turn_back_when_back_clear(self):
        self.assertEqual(self.entry([], 2), (YAW_RIGHT, "V1-back"))
        self.assertEqual(self.entry([], 1), (YAW_RIGHT, "V1-back"))
        self.assertEqual(self.entry([], -1), (YAW_LEFT, "V1-back"))
        self.assertEqual(self.entry([self.AHEAD], 1), (YAW_RIGHT, "V1-back"), "转回通道通畅就转回，哪怕当前航向被挡")

    def test_forward_when_back_blocked_but_hold_clear(self):
        self.assertEqual(self.entry([self.BLOCK_BACK_RIGHT], 1), (FORWARD_SLOW, "V1-fwd"))
        self.assertEqual(self.entry([self.BLOCK_BACK_LEFT], -1), (FORWARD_SLOW, "V1-fwd"))

    def test_more_when_both_blocked_below_cap(self):
        self.assertEqual(self.entry([self.BLOCK_BACK_RIGHT, self.AHEAD], 1), (YAW_LEFT, "V1-more"))
        self.assertEqual(self.entry([self.BLOCK_BACK_LEFT, self.AHEAD], -1), (YAW_RIGHT, "V1-more"))
        self.assertEqual(self.entry([self.BLOCK_BACK_RIGHT, self.AHEAD], 2), (YAW_LEFT, "V1-more"), "n=2 < n_cap=3 还能再转")

    def test_cap_turns_back(self):
        self.assertEqual(self.entry([self.BLOCK_BACK_RIGHT, self.AHEAD], 3), (YAW_RIGHT, "V1-cap"))
        self.assertEqual(self.entry([self.BLOCK_BACK_LEFT, self.AHEAD], -3), (YAW_LEFT, "V1-cap"))
        self.assertEqual(self.entry([self.BLOCK_BACK_RIGHT, self.AHEAD], 4), (YAW_RIGHT, "V1-cap"), "超过上限同样回转")
        self.assertEqual(self.entry([], 3), (YAW_RIGHT, "V1-back"), "到上限但转回通道通畅：仍是 V1-back（顺序 1 在先）")
        self.assertEqual(self.entry([self.BLOCK_BACK_RIGHT], 3), (FORWARD_SLOW, "V1-fwd"))

    def test_side_choice_when_n_zero_and_blocked(self):
        left_open = [4.0] * 10 + [1.0] * 10
        self.assertEqual(self.entry([self.AHEAD], 0, d=left_open), (YAW_LEFT, "V1-side"))
        self.assertEqual(self.entry([self.AHEAD], 0, d=left_open[::-1]), (YAW_RIGHT, "V1-side"))

    def test_tie_follows_last_turn_then_left(self):
        sym = [2.0] * 20
        self.assertEqual(self.entry([self.AHEAD], 0, YAW_RIGHT, sym), (YAW_RIGHT, "V1-last"))
        self.assertEqual(self.entry([self.AHEAD], 0, YAW_LEFT, sym), (YAW_LEFT, "V1-last"))
        self.assertEqual(self.entry([self.AHEAD], 0, None, sym), (YAW_LEFT, "V1-left"))

    def test_tie_boundary(self):
        left = [4.0] * 10
        self.assertEqual(self.entry([self.AHEAD], 0, YAW_RIGHT, left + [4.0] * 9 + [0.0]), (YAW_LEFT, "V1-side"),
                         "差 4.0 = 较大者 40 的 10%：不算平局")
        self.assertEqual(self.entry([self.AHEAD], 0, YAW_RIGHT, left + [4.0] * 9 + [0.4]), (YAW_RIGHT, "V1-last"),
                         "差 3.6 < 4.0：平局")
        self.assertEqual(self.entry([self.AHEAD], 0, YAW_RIGHT, [10.0] * 10 + [4.0] * 10), (YAW_RIGHT, "V1-last"),
                         "sum_clip：左半 10 与 4.0 截断后同为 4.0 一列，平局")

    def test_odd_middle_column_is_excluded_from_the_halves(self):
        d27 = [4.0] * 27
        d27[13] = 0.5                          # 只有正中间一列近
        self.assertEqual(rule_v1_entry([self.AHEAD], d27, 0, YAW_RIGHT), (YAW_RIGHT, "V1-last"))

    def test_never_safe_and_forward_only_when_clear(self):
        rng = random.Random(3)
        seen = set()
        for _ in range(6000):
            pts = [(rng.uniform(-0.5, 2.0), rng.uniform(-1.0, 1.0)) for _ in range(rng.randint(0, 10))]
            d = [rng.choice((4.0, rng.uniform(0.3, 4.0))) for _ in range(20)]
            n, last = rng.randint(-4, 4), rng.choice((None, YAW_LEFT, YAW_RIGHT))
            action, entry = rule_v1_entry(pts, d, n, last)
            seen.add(entry)
            self.assertIn(action, (FORWARD_SLOW, YAW_LEFT, YAW_RIGHT), "v1 永不输出 SAFE_NEUTRAL")
            self.assertIn(entry, V1_ENTRIES)
            if action == FORWARD_SLOW:
                self.assertTrue(corridor_clear(pts, 0.0), "FORWARD 只在 clear(0) 时输出（兼作护盾）")
            if entry == "V1-back":
                self.assertEqual(action, YAW_RIGHT if n > 0 else YAW_LEFT)
            if entry in ("V1-more",):
                self.assertEqual(action, YAW_LEFT if n > 0 else YAW_RIGHT)
        self.assertEqual(seen, set(V1_ENTRIES), "随机输入覆盖全部 7 个条目")


class StepTests(unittest.TestCase):
    def test_step_returns_clear_triple_left_hold_right(self):
        d = column_distances(0.0, 0.0, 0.0, [square(1.2, 0.0)])
        action, entry, mem, info = rule_v1_step([], 0, 0.0, 0.0, 0.0, d, 0, None)
        self.assertEqual((action, entry), (YAW_LEFT, "V1-left"))
        self.assertEqual(info["mem_n"], len(mem))
        self.assertGreater(info["mem_n"], 3)
        self.assertEqual(info["clear"], [False, False, False], "单柱正前方 1.0 m：左转 26°、保持、右转 26° 的通道都被挡")

    def test_clear_triple_order_is_left_hold_right_for_asymmetric_scenes(self):
        # 柱在左前（面上点约 (1.0, 0.45..0.53)）：只有「左转 26°」的通道被挡；镜像柱只挡「右转 26°」
        d = column_distances(0.0, 0.0, 0.0, [square(1.2, 0.65)])
        _a, _e, _m, info = rule_v1_step([], 0, 0.0, 0.0, 0.0, d, 0, None)
        self.assertEqual(info["clear"], [False, True, True])
        d = column_distances(0.0, 0.0, 0.0, [square(1.2, -0.65)])
        _a, _e, _m, info = rule_v1_step([], 0, 0.0, 0.0, 0.0, d, 0, None)
        self.assertEqual(info["clear"], [True, True, False])

    def test_memory_keeps_a_pillar_that_left_the_field_of_view(self):
        # 原地依次左转到 78°（n 在这里固定传 0，只验记忆）：柱 (1.2, 0) 在 yaw=0、26° 时可见，52° 起出了视场（±30°）
        pillars = [square(1.2, 0.0)]
        mem = []
        for step, yaw in enumerate((0.0, 26.0, 52.0, 78.0)):
            d = column_distances(0.0, 0.0, yaw, pillars)
            _a, _e, mem, info = rule_v1_step(mem, step, 0.0, 0.0, yaw, d, 0, None)
        self.assertTrue(all(v == 4.0 for v in d), "第 4 步朝 78°：柱在视场外，列距离全空")
        self.assertGreater(info["mem_n"], 0)
        body = points_to_body([(X, Y) for X, Y, _t in mem], 0.0, 0.0, 78.0)
        self.assertFalse(corridor_clear(body, -78.0), "正对柱子的那个方向仍被记忆里的点挡住")
        self.assertTrue(corridor_clear(body, 0.0), "当前航向 78° 这边没有柱子")


class SimMechanicsTests(unittest.TestCase):
    def test_goal_line_and_start(self):
        pillars = [square(1.5, 0.0), square(3.0, 1.0)]
        r = rule_sim.simulate(pillars, "forward", steps=3)
        self.assertEqual(r["goal_x"], collect.goal_line_x(pillars))
        self.assertEqual(r["traj"][0], (rule_sim.START_X, 0.0, 0.0))

    def test_forward_step_and_turn_step(self):
        r = rule_sim.simulate([square(30.0, 0.0)], "forward", steps=2)
        self.assertAlmostEqual(r["traj"][-1][0], rule_sim.START_X + 2 * 0.25, places=9)
        r = rule_sim.simulate([square(30.0, 5.0)], "left", steps=2)
        self.assertAlmostEqual(r["traj"][-1][2], 52.0, places=9)
        self.assertAlmostEqual(r["traj"][-1][0], rule_sim.START_X, places=9, msg="转向步不位移")

    def test_forward_into_pillar_collides(self):
        # 柱面 x = 0.3；出发 x = −0.5：第 3 步走到 0.25 时离柱面 0.05 m，已进入 0.102 m 以内
        r = rule_sim.simulate([square(0.5, 0.0)], "forward", steps=10)
        self.assertTrue(r["collided"])
        self.assertFalse(r["success"])
        self.assertLess(r["min_gap"], rule_sim.BODY_RADIUS_M)

    def test_gap_to_square_surface(self):
        self.assertAlmostEqual(rule_sim.gap_to_pillar(square(1.0, 0.0), 0.5, 0.0), 0.3, places=9)       # 近面 x=0.8
        self.assertAlmostEqual(rule_sim.gap_to_pillar(square(1.0, 0.0), 0.5, 0.5), math.hypot(0.3, 0.3), places=9)
        self.assertEqual(rule_sim.gap_to_pillar(square(1.0, 0.0), 1.0, 0.0), 0.0, "在柱内")

    def test_reversal_run(self):
        L, R, F = YAW_LEFT, YAW_RIGHT, FORWARD_SLOW
        self.assertEqual(rule_sim.max_reversal_run([L, L, F, R, R]), 0)
        self.assertEqual(rule_sim.max_reversal_run([L, R, L, R]), 3)
        self.assertEqual(rule_sim.max_reversal_run([L, R, F, L, R, L]), 2)
        self.assertEqual(rule_sim.max_reversal_run([L, R, L, R, L]), 4)
        self.assertEqual(rule_sim.max_reversal_run([]), 0)


class DetourTests(unittest.TestCase):
    def check_detour(self, pillar):
        r = rule_sim.simulate([pillar], "v1", steps=40, stop_at_goal=False)
        goal = pillar["x"] + 1.0
        reached = [i for i, (x, _y, _yaw) in enumerate(r["traj"]) if x >= goal]
        self.assertTrue(reached, "要越过柱 x + 1.0")
        self.assertLessEqual(reached[0], 40)
        self.assertFalse(r["collided"])
        self.assertGreaterEqual(r["min_gap"], rule_sim.BODY_RADIUS_M, "离柱面始终 ≥ 0.102 m")
        self.assertEqual(r["n_final"], 0, "绕过后净转向回到 0")
        self.assertLess(rule_sim.max_reversal_run(r["actions"]), 4, "不出现连续 4 次以上相邻 L/R 换向")
        self.assertTrue(all(a != SAFE_NEUTRAL for a in r["actions"]))
        self.assertTrue(all(c[1] for a, c in zip(r["actions"], r["clear"]) if a == FORWARD_SLOW), "每个前进步都过 clear(0)")
        return r

    def test_single_pillar_surface_one_metre_ahead(self):
        self.check_detour(square(rule_sim.START_X + 1.2, 0.0))       # 柱面离出生点 1.0 m

    def test_single_pillar_centre_one_metre_ahead(self):
        self.check_detour(square(rule_sim.START_X + 1.0, 0.0))       # 柱心离出生点 1.0 m（柱面 0.8 m）

    def test_small_lateral_offsets_still_pass(self):
        for y in (-0.15, 0.15):
            r = self.check_detour(square(rule_sim.START_X + 1.2, y))
            self.assertTrue(rule_sim.simulate([square(rule_sim.START_X + 1.2, y)], "v1", steps=40)["success"], y)
            self.assertIsNotNone(r)


class OfflineTrainCheckTests(unittest.TestCase):
    def test_train_geometries_selection(self):
        geoms = rule_sim.train_geometries(20)
        self.assertEqual(len(geoms), 20)
        self.assertEqual(len({sm.geometry_gid(p) for _s, p in geoms}), 20, "几何互不相同")
        for seed, pillars in geoms:
            self.assertEqual(sm.split_for_seed(seed), "train")

    def test_offline_gate_on_20_train_geometries(self):
        geoms = rule_sim.train_geometries(20)
        results = [rule_sim.simulate(p, "v1", steps=80) for _s, p in geoms]
        passed = sum(r["success"] for r in results)
        collisions = sum(r["collided"] for r in results)
        print(f"\n[离线检查] train 分区 20 个几何（seed {[s for s, _ in geoms]}）：过线 {passed}/20，撞柱 {collisions}/20，"
              f"步数 {[r['steps'] for r in results]}", file=sys.stderr)
        self.assertGreaterEqual(passed, 18, "离线过线数 < 18/20：停下写报告，不起 Gazebo 正式实验")

    def test_course_blocks_use_all_pillars(self):
        pillars, info = scene_course.sample_course(2, 3)
        self.assertGreater(len(pillars), 20)
        r = rule_sim.simulate(pillars, "v1", steps=240)
        self.assertEqual(r["goal_x"], collect.goal_line_x(pillars))
        self.assertGreater(max(x for x, _y, _yaw in r["traj"]), 6.0, "长课程上规则能走出第 0 块")


if __name__ == "__main__":
    unittest.main()
