#!/usr/bin/env python3
"""教师 v7（第十二阶段 W1-7 的 B2、B3）回归：过柱阵后目标取线、A* 无路降余量重试、版本分派。

背景（docs/reviews/2026-11-10/data-algo.md §A2、§A4）：
  B2：教师目标是点 (最远柱 x + 2.0, 0)，过了柱阵为了回到 y=0 还在转向，出口区 17.7% 的训练帧是这种 YAW，
      而学生看不到目标点。机体 x 越过最远柱后表面（柱 x + 柱半宽）后，目标 y 取当前 y（点 → 线）。
  B3：A* 在标准余量下偶尔无路（起点在余量内、四邻全堵），教师退回局部规则并左右摆。无路时按 0.30、0.20 m 各重试一次，
      只试严格小于标准余量的档位，仍无路才返回 None。
  版本：TEACHER_VERSION v5 → v7（v6 已被 forest_e1 档占用）；旧版本数据按旧规则复算（teacher.set_vehicle(..., version=)）。

覆盖：版本表与分派 / B2（FORWARD、对称、边界、航向仍回正、waypoint 不受影响、圆柱、costmap 目标）/
B3（两档重试、重试顺序与「严格小于」、无路仍 None、waypoint 两段）/ 缓存钥匙按余量分开且标准余量结果不被污染 /
旧版规则与 a714b54 的 teacher.py 逐位相同（位姿网格摘要，常量由 a714b54 的 teacher.py 算出）。
"""
import functools
import hashlib
import json
import os
import random
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene_course as sc  # noqa: E402
import scene_gen as sg  # noqa: E402
import teacher as T  # noqa: E402
from test_astar_cache import astar_reference  # noqa: E402  旧 A* 的原样拷贝：逐位对照的参照物

V5, V6, V7 = "uavrgb.teacher.v5", "uavrgb.teacher.v6", "uavrgb.teacher.v7"
COST_V1, COST_V2 = "uavrgb.teacher_cost.v1", "uavrgb.teacher_cost.v2"
BOX = [0.4, 0.4, 3.0]
GATE = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": BOX}]      # 最远柱 x=1.5，后表面 1.7，目标 (3.5, 0)
FAR_X = 1.5
BACK_X = 1.7

# B3 夹具：单柱 x=1.55（前表面 1.35，不在栅格点上，避免余量判定落在浮点边界）。
# 栅格点是 0.1 的整数倍：起点吸附到 1.1 / 1.2 / 1.3，离前表面 0.25 / 0.15 / 0.05 m。
WALL = [{"x": 1.55, "y": 0.0, "z": 1.5, "size": BOX}]
WALL_GOAL = (3.55, 0.0)
NEAR_030 = (1.1, 0.0)    # 0.40 无路；0.30 有路
NEAR_020 = (1.2, 0.0)    # 0.40、0.30 无路；0.20 有路
NEAR_NONE = (1.3, 0.0)   # 三档都无路

# 位姿网格摘要（旧版规则必须与 a714b54 的 teacher.py 逐位相同）：常量由 a714b54 的 teacher.py 按下面同一个函数算出
# （草稿 $D/tmp/phase12-cards/w1-7/golden_probe2.py、golden_old_v5.txt）。
GOLDEN_V5_ALL = "652ac5350b8b45fdc7ab4887fbe614c14a4fa16ea61acbc6e615e90ab4f2a007"
GOLDEN_PRE_EXIT = "dace94057f34881017e26d7145bd4c66c75f5c912ce42bf80dfbcdf03d862d2e"
GOLDEN_TIEBREAK = "49e286550104c61427b5a8f3693978a6a968ab916489fde2d59b37323d4801c5"
GRID_SEEDS = (1, 113)
GRID_YS = (-1.4, -0.5, 0.0, 0.35, 1.2)
GRID_YAWS = (-60.0, 0.0, 45.0)


def grid_xs(far, back):
    return (-1.0, 0.0, 0.8, far - 0.6, far - 0.1, far + 0.05, back + 0.1, back + 0.5, back + 1.0)


@functools.lru_cache(maxsize=None)
def pose_grid_digests(version):
    """(全部状态摘要, 出口区之前且标准余量有路的状态摘要, tiebreak=visible 摘要, 状态数, 出口区状态数, 无路状态数)。

    按 set_vehicle(..., version=version) 的规则算；一次约 9 s，同一版本只算一次（lru_cache）。
    """
    h_all, h_pre, h_tie = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    n_all = n_exit = n_none = 0
    for veh in ("x3", "e1_mini"):
        T.set_vehicle(veh, "base", version=version)
        for seed in GRID_SEEDS:
            pillars = sg.sample_pillars(seed)[0]
            far = max(p["x"] for p in pillars)
            back = far + 0.2
            goal = (far + 2.0, 0.0)
            for x in grid_xs(far, back):
                for y in GRID_YS:
                    for yaw in GRID_YAWS:
                        act = T.plan_action_global(x, y, yaw, pillars)
                        path = T._global_path(x, y, pillars, T.ASTAR_CLEARANCE_M, 0.1)
                        rec = json.dumps([veh, seed, x, y, yaw, act, path], sort_keys=True).encode()
                        h_all.update(rec)
                        n_all += 1
                        n_none += path is None
                        n_exit += x > back
                        std = T.astar_path(pillars, (x, y), goal, clearance=T.ASTAR_CLEARANCE_M)
                        if x <= back and std is not None:
                            h_pre.update(rec)
                        if veh == "e1_mini" and seed == 1 and yaw != 0.0:
                            tb = T.plan_action_global_tiebreak(x, y, yaw, pillars, tiebreak="visible")
                            h_tie.update(json.dumps([x, y, yaw, list(tb)]).encode())
    return h_all.hexdigest(), h_pre.hexdigest(), h_tie.hexdigest(), n_all, n_exit, n_none


def astar_calls(fn):
    """执行 fn()，返回 (fn 的返回值, 这期间 teacher.astar_path 被调用时的 clearance 序列)。"""
    with mock.patch.object(T, "astar_path", wraps=T.astar_path) as spy:
        out = fn()
    return out, [c.kwargs["clearance"] for c in spy.call_args_list]


class TeacherCase(unittest.TestCase):
    """每个用例从 e1_mini + 当前版本规则 + 空缓存起步；收尾回到模块缺省（x3 + 当前版本规则）。"""

    def setUp(self):
        T.set_vehicle("e1_mini")
        T._FREE_CACHE.clear()

    def tearDown(self):
        T.set_vehicle("x3")
        T._FREE_CACHE.clear()


class VersionTable(TeacherCase):
    def test_version_constants(self):
        self.assertEqual(T.TEACHER_VERSION, V7)
        self.assertEqual(T.teacher_version("base"), V7)
        self.assertEqual(T.teacher_version("l1"), V7)             # 未单列的任务走 base 档
        self.assertEqual(T.teacher_version("forest_e1"), V6)      # forest 档规则不变，版本不动
        self.assertEqual(T.TEACHER_COST_VERSION, COST_V2)
        self.assertNotEqual(T.TEACHER_COST_VERSION, T.TEACHER_VERSION)

    def test_rules_table(self):
        self.assertEqual(T.rules_for_version(V5), {"goal_line": False, "retry": ()})
        self.assertEqual(T.rules_for_version(V6), {"goal_line": False, "retry": ()})
        self.assertEqual(T.rules_for_version(V7), {"goal_line": True, "retry": (0.30, 0.20)})
        self.assertEqual(T.rules_for_version(COST_V1), {"goal_line": False, "retry": ()})
        self.assertEqual(T.rules_for_version(COST_V2), {"goal_line": True, "retry": ()})
        with self.assertRaises(ValueError):
            T.rules_for_version("uavrgb.teacher.v99")
        with self.assertRaises(ValueError):
            T.rules_for_version(None)

    def test_recomputable_versions(self):
        self.assertEqual(T.recomputable_versions("base"), (V5, V7))
        self.assertEqual(T.recomputable_versions("l1"), (V5, V7))
        self.assertEqual(T.recomputable_versions("forest_e1"), (V6,))
        self.assertEqual(T.recomputable_versions("base", costmap=True), (COST_V1, COST_V2))
        self.assertEqual(T.recomputable_versions("forest_e1", costmap=True), (COST_V1, COST_V2))
        self.assertIn(T.TEACHER_VERSION, T.recomputable_versions("base"))
        self.assertIn(T.TEACHER_COST_VERSION, T.recomputable_versions("base", costmap=True))

    def test_set_vehicle_selects_rules(self):
        T.set_vehicle("e1_mini", "base")
        self.assertEqual((T.GOAL_LINE_AFTER_PILLARS, T.ASTAR_RETRY_CLEARANCES), (True, (0.30, 0.20)))
        T.set_vehicle("e1_mini", "base", version=V5)
        self.assertEqual((T.GOAL_LINE_AFTER_PILLARS, T.ASTAR_RETRY_CLEARANCES), (False, ()))
        T.set_vehicle("e1_mini", "base", version=V7)
        self.assertEqual((T.GOAL_LINE_AFTER_PILLARS, T.ASTAR_RETRY_CLEARANCES), (True, (0.30, 0.20)))
        T.set_vehicle("e1_mini", "forest_e1")        # forest 档缺省版本 v6：旧规则，标签逐位不变
        self.assertEqual((T.GOAL_LINE_AFTER_PILLARS, T.ASTAR_RETRY_CLEARANCES), (False, ()))
        T.set_vehicle("e1_mini", "base", version=COST_V1)
        self.assertEqual(T.GOAL_LINE_AFTER_PILLARS, False)
        T.set_vehicle("e1_mini", "base", version=COST_V2)
        self.assertEqual(T.GOAL_LINE_AFTER_PILLARS, True)

    def test_unknown_version_raises_before_touching_state(self):
        T.set_vehicle("e1_mini", "base")
        before = (T.BODY_HALF_M, T.ASTAR_CLEARANCE_M, T.LOCAL_CLEARANCE_M, T.TEACHER_TASK,
                  T.GOAL_LINE_AFTER_PILLARS, T.ASTAR_RETRY_CLEARANCES)
        with self.assertRaises(ValueError):
            T.set_vehicle("x3", "forest_e1", version="uavrgb.teacher.v99")
        after = (T.BODY_HALF_M, T.ASTAR_CLEARANCE_M, T.LOCAL_CLEARANCE_M, T.TEACHER_TASK,
                 T.GOAL_LINE_AFTER_PILLARS, T.ASTAR_RETRY_CLEARANCES)
        self.assertEqual(before, after)


class GoalLine(TeacherCase):
    """B2：越过最远柱后表面后，目标 y 取当前 y。"""

    def test_forward_after_pillars_with_lateral_offset(self):
        for veh in ("e1_mini", "x3"):
            T.set_vehicle(veh)
            for sign, name in ((1, "左偏"), (-1, "右偏")):
                got = T.plan_action_global(FAR_X + 0.5, 1.2 * sign, 0.0, GATE)
                self.assertEqual(got, T.FORWARD_SLOW, f"{veh} {name}：{T.ACTIONS[got]}")

    def test_old_rules_still_yaw_back_toward_axis(self):
        for veh in ("e1_mini", "x3"):
            T.set_vehicle(veh, "base", version=V5)
            self.assertEqual(T.plan_action_global(FAR_X + 0.5, 1.2, 0.0, GATE), T.YAW_RIGHT, veh)
            self.assertEqual(T.plan_action_global(FAR_X + 0.5, -1.2, 0.0, GATE), T.YAW_LEFT, veh)

    def test_not_triggered_before_back_surface(self):
        for x in (-0.5, 1.0, BACK_X - 0.1, BACK_X - 1e-6):
            new = T._global_path(x, 1.2, GATE, T.ASTAR_CLEARANCE_M, 0.1)
            T.set_vehicle("e1_mini", "base", version=V5)
            old = T._global_path(x, 1.2, GATE, T.ASTAR_CLEARANCE_M, 0.1)
            T.set_vehicle("e1_mini", "base", version=V7)
            self.assertEqual(new, old, f"x={x}（还没越过后表面 {BACK_X}）路径不能变")
        # 后表面之外 1 cm 才算越过
        past = T._global_path(BACK_X + 0.01, 1.2, GATE, T.ASTAR_CLEARANCE_M, 0.1)
        self.assertAlmostEqual(past[-1][1], past[0][1], places=6)

    def test_goal_becomes_a_line_in_the_path(self):
        new = T._global_path(FAR_X + 0.5, 1.2, GATE, T.ASTAR_CLEARANCE_M, 0.1)
        self.assertAlmostEqual(new[-1][0], FAR_X + 2.0, places=6)
        self.assertAlmostEqual(new[-1][1], new[0][1], places=6)       # 终点与起点同一行
        self.assertTrue(all(abs(py - new[0][1]) < 1e-6 for _px, py in new), "路径应当是直线")
        T.set_vehicle("e1_mini", "base", version=V5)
        old = T._global_path(FAR_X + 0.5, 1.2, GATE, T.ASTAR_CLEARANCE_M, 0.1)
        self.assertAlmostEqual(old[-1][1], 0.0, places=6)             # 旧版目标是点 (3.5, 0)

    def test_heading_is_still_corrected(self):
        # 目标变成线以后，机头偏离 +x 超过 25° 仍要回正（航向守卫的前提不能丢）
        self.assertEqual(T.plan_action_global(FAR_X + 0.5, 1.2, 60.0, GATE), T.YAW_RIGHT)
        self.assertEqual(T.plan_action_global(FAR_X + 0.5, 1.2, -60.0, GATE), T.YAW_LEFT)
        self.assertEqual(T.plan_action_global(FAR_X + 0.5, -1.2, 60.0, GATE), T.YAW_RIGHT)
        self.assertEqual(T.plan_action_global(FAR_X + 0.5, -1.2, -60.0, GATE), T.YAW_LEFT)

    def test_waypoint_tasks_are_not_affected(self):
        wp = {"x": 3.0, "y": 1.0}
        for stop in (True, False):
            new = T._global_path(2.2, 0.8, GATE, T.ASTAR_CLEARANCE_M, 0.1, waypoint=wp, stop_at_waypoint=stop)
            T.set_vehicle("e1_mini", "base", version=V5)
            old = T._global_path(2.2, 0.8, GATE, T.ASTAR_CLEARANCE_M, 0.1, waypoint=wp, stop_at_waypoint=stop)
            T.set_vehicle("e1_mini", "base", version=V7)
            self.assertEqual(new, old, f"stop_at_waypoint={stop}")
            self.assertIsNotNone(new)

    def test_round_pillar_back_surface_uses_radius(self):
        trees = [{"x": 1.0, "y": 0.4, "z": 1.5, "radius": 0.15, "size": [0.3, 0.3, 3.0]},
                 {"x": 1.5, "y": -0.4, "z": 1.5, "radius": 0.15, "size": [0.3, 0.3, 3.0]}]
        self.assertAlmostEqual(T._pillars_back_x(trees), 1.65)
        self.assertAlmostEqual(T._pillars_back_x(GATE), BACK_X)
        self.assertAlmostEqual(T._pillars_back_x([{"x": 2.0, "y": 0.0}]), 2.2)    # 缺 size 按 0.2 半宽
        # 柱后表面之后：线；之前：点
        past = T._global_path(1.8, 1.2, trees, T.ASTAR_CLEARANCE_M, 0.1)
        self.assertAlmostEqual(past[-1][1], past[0][1], places=6)
        before = T._global_path(1.6, 1.2, trees, T.ASTAR_CLEARANCE_M, 0.1)
        self.assertAlmostEqual(before[-1][1], 0.0, places=6)

    def test_forest_profile_keeps_old_rules(self):
        trees = [{"x": 1.0, "y": 0.4, "z": 1.5, "radius": 0.15, "size": [0.3, 0.3, 3.0]}]
        T.set_vehicle("e1_mini", "forest_e1")
        path = T._global_path(1.6, 1.2, trees, T.ASTAR_CLEARANCE_M, 0.1)
        self.assertAlmostEqual(path[-1][1], 0.0, places=6)           # 仍然是点目标
        self.assertEqual(T.plan_action_global(1.6, 1.2, 0.0, trees), T.YAW_RIGHT)

    def test_costmap_goal_line(self):
        T.set_vehicle("e1_mini", "base", version=COST_V2)
        q, info = T.plan_costs(2.0, 1.5, 0.0, GATE)
        self.assertEqual(info["goal"], (FAR_X + 2.0, 1.5))
        act, _ = T.plan_action_costmap(2.0, 1.5, 0.0, GATE)
        self.assertEqual(act, T.FORWARD_SLOW)
        # 还没越过后表面：点目标
        _, info_in = T.plan_costs(1.6, 1.5, 0.0, GATE)
        self.assertEqual(info_in["goal"], (FAR_X + 2.0, 0.0))
        # waypoint 的 goal-stop 任务：目标仍是 waypoint
        _, info_wp = T.plan_costs(2.0, 1.5, 0.0, GATE, waypoint={"x": 3.0, "y": 0.5}, stop_at_waypoint=True)
        self.assertEqual(info_wp["goal"], (3.0, 0.5))

    def test_costmap_old_version_keeps_point_goal(self):
        T.set_vehicle("e1_mini", "base", version=COST_V1)
        _, info = T.plan_costs(2.0, 1.5, 0.0, GATE)
        self.assertEqual(info["goal"], (FAR_X + 2.0, 0.0))
        act, _ = T.plan_action_costmap(2.0, 1.5, 0.0, GATE)
        self.assertEqual(act, T.YAW_RIGHT)                            # 旧版的出口区 YAW


class RetryMargins(TeacherCase):
    """B3：A* 在标准余量下无路时，按 0.30、0.20 m 各重试一次。"""

    def goal_path(self, start, clearance=None, **kw):
        c = T.ASTAR_CLEARANCE_M if clearance is None else clearance
        return T._global_path(start[0], start[1], WALL, c, 0.1, **kw)

    def test_fixture_documents_the_three_regimes(self):
        for start, lens in ((NEAR_030, (False, True, True)), (NEAR_020, (False, False, True)),
                            (NEAR_NONE, (False, False, False))):
            got = tuple(T.astar_path(WALL, start, WALL_GOAL, clearance=c) is not None for c in (0.40, 0.30, 0.20))
            self.assertEqual(got, lens, f"start={start}")

    def test_path_found_at_030(self):
        want = astar_reference(WALL, NEAR_030, WALL_GOAL, clearance=0.30)
        self.assertIsNotNone(want)
        self.assertEqual(self.goal_path(NEAR_030), want)
        T.set_vehicle("e1_mini", "base", version=V5)
        self.assertIsNone(self.goal_path(NEAR_030))                    # 旧规则：无路 → 调用方退回局部

    def test_path_found_at_020(self):
        want = astar_reference(WALL, NEAR_020, WALL_GOAL, clearance=0.20)
        self.assertIsNotNone(want)
        self.assertEqual(self.goal_path(NEAR_020), want)
        T.set_vehicle("e1_mini", "base", version=V5)
        self.assertIsNone(self.goal_path(NEAR_020))

    def test_still_none_when_no_margin_works(self):
        self.assertIsNone(self.goal_path(NEAR_NONE))
        # 仍无路 → plan_action_global 退回局部规则，结果与直接调 plan_action 相同
        for yaw in (-30.0, 0.0, 30.0):
            self.assertEqual(T.plan_action_global(NEAR_NONE[0], NEAR_NONE[1], yaw, WALL),
                             T.plan_action(NEAR_NONE[0], NEAR_NONE[1], yaw, WALL, tol=25.0))

    def test_action_follows_the_retried_path(self):
        # 旧教师在这两个位姿 A* 无路，退回局部规则只能 SAFE；新教师沿重试出的路径走（路径一开始向后退出余量，是转向）
        for start in (NEAR_030, NEAR_020):
            T.set_vehicle("e1_mini", "base", version=V5)
            self.assertEqual(T.plan_action_global(start[0], start[1], 0.0, WALL), T.SAFE_NEUTRAL, start)
            T.set_vehicle("e1_mini", "base", version=V7)
            got = T.plan_action_global(start[0], start[1], 0.0, WALL)
            self.assertIn(got, (T.YAW_LEFT, T.YAW_RIGHT), f"{start}: {T.ACTIONS[got]}")

    def test_retry_order_and_strictly_below_standard(self):
        _, seq = astar_calls(lambda: self.goal_path(NEAR_NONE))
        self.assertEqual(seq, [0.40, 0.30, 0.20])                      # 三档都试，仍无路
        _, seq = astar_calls(lambda: self.goal_path(NEAR_030))
        self.assertEqual(seq, [0.40, 0.30])                            # 第一个有路的档位就停
        _, seq = astar_calls(lambda: self.goal_path(NEAR_020))
        self.assertEqual(seq, [0.40, 0.30, 0.20])
        _, seq = astar_calls(lambda: self.goal_path((-0.5, 0.0)))
        self.assertEqual(seq, [0.40])                                  # 标准余量有路：不重试
        _, seq = astar_calls(lambda: self.goal_path(NEAR_NONE, clearance=0.30))
        self.assertEqual(seq, [0.30, 0.20])                            # 标准 0.30：只试比它小的 0.20
        _, seq = astar_calls(lambda: self.goal_path(NEAR_NONE, clearance=0.20))
        self.assertEqual(seq, [0.20])                                  # 标准 0.20（forest 档那样）：不会向更大的余量重试
        T.set_vehicle("e1_mini", "base", version=V5)
        _, seq = astar_calls(lambda: self.goal_path(NEAR_NONE))
        self.assertEqual(seq, [0.40])                                  # 旧规则：不重试

    def test_forest_profile_never_retries(self):
        T.set_vehicle("e1_mini", "forest_e1")
        _, seq = astar_calls(lambda: self.goal_path(NEAR_NONE))
        self.assertEqual(seq, [0.20])

    def test_x3_profile_retries_from_050(self):
        T.set_vehicle("x3")
        self.assertEqual(T.ASTAR_CLEARANCE_M, 0.5)
        _, seq = astar_calls(lambda: self.goal_path(NEAR_NONE))
        self.assertEqual(seq, [0.50, 0.30, 0.20])

    def test_waypoint_legs_retry_too(self):
        wp = {"x": 3.0, "y": 0.9}
        want = astar_reference(WALL, NEAR_030, (3.0, 0.9), clearance=0.30)
        self.assertIsNotNone(want)
        self.assertEqual(self.goal_path(NEAR_030, waypoint=wp, stop_at_waypoint=True), want)
        T.set_vehicle("e1_mini", "base", version=V5)
        self.assertIsNone(self.goal_path(NEAR_030, waypoint=wp, stop_at_waypoint=True))

    def test_standard_margin_paths_are_untouched(self):
        # 标准余量有路的状态：路径逐位等于旧 A*（不是重试档的路径）
        rng = random.Random(20261111)
        n = 0
        for seed in (1, 113, 172):
            pillars = sg.sample_pillars(seed)[0]
            goal = (max(p["x"] for p in pillars) + 2.0, 0.0)
            back = max(p["x"] for p in pillars) + 0.2
            for _ in range(8):
                start = (rng.uniform(-1.0, back - 0.2), rng.uniform(-2.0, 2.0))
                want = astar_reference(pillars, start, goal, clearance=T.ASTAR_CLEARANCE_M)
                if want is None:
                    continue
                self.assertEqual(T._global_path(start[0], start[1], pillars, T.ASTAR_CLEARANCE_M, 0.1), want)
                n += 1
        self.assertGreater(n, 10)


class RetryCache(TeacherCase):
    """不同余量不能共用同一份可走性缓存；标准余量下的结果不被重试污染。"""

    def test_cache_keys_are_per_clearance(self):
        T._global_path(NEAR_NONE[0], NEAR_NONE[1], WALL, 0.40, 0.1)    # 三档都被访问
        self.assertEqual(sorted({k[0] for k in T._FREE_CACHE}), [0.20, 0.30, 0.40])
        self.assertEqual(len(T._FREE_CACHE), 3)
        std, mid, low = (T._free_cells(WALL, c, 0.1) for c in (0.40, 0.30, 0.20))
        self.assertIsNot(std, mid)
        self.assertIsNot(mid, low)
        # 同一格在三档里的可走性不同：格 (1.0, 0) 离柱前表面 0.35 m，标准 0.40 堵、0.30 与 0.20 通。
        # 用 warm_free_cells 把三档各自整张填满（它按各自的钥匙填），再逐格对照。
        for c in (0.40, 0.30, 0.20):
            T.warm_free_cells(WALL, clearance=c)
        x0, y0, _nx, _ny = T._grid_spec(WALL, 0.1)
        i, j = 25, 28
        self.assertAlmostEqual(x0 + i * 0.1, 1.0)
        self.assertAlmostEqual(y0 + j * 0.1, 0.0)
        self.assertEqual((std[(i, j)], mid[(i, j)], low[(i, j)]), (False, True, True))
        self.assertEqual(len(T._FREE_CACHE), 3)

    def test_standard_margin_result_not_polluted_by_retry(self):
        T._global_path(NEAR_030[0], NEAR_030[1], WALL, 0.40, 0.1)      # 重试已把 0.30 档填进缓存
        self.assertIsNone(T.astar_path(WALL, NEAR_030, WALL_GOAL, clearance=0.40))
        self.assertEqual(T.astar_path(WALL, NEAR_030, WALL_GOAL, clearance=0.30),
                         astar_reference(WALL, NEAR_030, WALL_GOAL, clearance=0.30))
        # 重试之后再走标准余量有路的状态，路径逐位等于旧 A*
        far_start = (-0.5, 0.3)
        self.assertEqual(T._global_path(far_start[0], far_start[1], WALL, 0.40, 0.1),
                         astar_reference(WALL, far_start, WALL_GOAL, clearance=0.40))

    def test_retry_paths_equal_reference_on_scenes(self):
        # 起点放在柱前 0.12–0.33 m（余量内），期望 = 旧 A* 在 (标准, 0.30, 0.20) 里第一个有路的结果；每个起点冷缓存
        n = hit = 0
        for seed in (1, 113, 172):
            pillars = sg.sample_pillars(seed)[0]
            goal = (max(p["x"] for p in pillars) + 2.0, 0.0)
            for p in pillars[:3]:
                for d in (0.12, 0.20, 0.26, 0.33):
                    start = (p["x"] - p["size"][0] / 2 - d, p["y"])
                    want = None
                    for c in (0.40, 0.30, 0.20):
                        want = astar_reference(pillars, start, goal, clearance=c)
                        if want is not None:
                            break
                    T._FREE_CACHE.clear()
                    got = T._global_path(start[0], start[1], pillars, 0.40, 0.1)
                    self.assertEqual(got, want, f"seed={seed} start={start}")
                    n += 1
                    hit += want is not None and astar_reference(pillars, start, goal, clearance=0.40) is None
        self.assertGreater(n, 30)
        self.assertGreater(hit, 3, "场景扫描里应当出现过「标准余量无路、重试有路」的状态")


class BitIdentical(TeacherCase):
    """旧规则与 a714b54 的 teacher.py 逐位相同；新规则在没触发 B2、B3 的状态上同样逐位相同。"""

    def test_v5_rules_reproduce_the_old_teacher(self):
        h_all, _h_pre, h_tie, _n_all, n_exit, n_none = pose_grid_digests(V5)
        self.assertGreater(n_exit, 100)       # 网格里包含出口区与无路状态：旧规则在这两类状态上也要逐位相同
        self.assertGreater(n_none, 50)
        self.assertEqual(h_all, GOLDEN_V5_ALL)
        self.assertEqual(h_tie, GOLDEN_TIEBREAK)

    def test_v7_rules_equal_old_outside_exit_zone_when_standard_path_exists(self):
        _h_all, h_pre, _h_tie, *_ = pose_grid_digests(V7)
        self.assertEqual(h_pre, GOLDEN_PRE_EXIT)

    def test_v7_differs_from_old_in_exit_zone_and_no_path_states(self):
        self.assertNotEqual(pose_grid_digests(V7)[0], GOLDEN_V5_ALL)
        self.assertNotEqual(pose_grid_digests(V7)[2], GOLDEN_TIEBREAK)


if __name__ == "__main__":
    unittest.main()
