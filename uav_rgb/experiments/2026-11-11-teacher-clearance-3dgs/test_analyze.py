#!/usr/bin/env python3
"""teacher-clearance-3dgs 的 analyze.py 单测：只用标准库，合成数据，不加载场景、不占 GPU。

  PYTHONDONTWRITEBYTECODE=1 python test_analyze.py
没设 TMPDIR 时临时目录落数据盘（根分区只剩 3 GB，不往 /tmp 写）。
覆盖：预注册判决的每个边界（Fraction，不用浮点）、撞在前的归类、步分支与标签的计数、回放（桩占据 + 真 OccAstarTeacher：否决、连续等待上限、
缺 trace、预测位移误差）、出生位姿对比、整套读盘与汇总（配对 / 臂 B / 分场景 / 分桶 / 逐集归类）、acceptance 与 regress 的命令行。
"""
import contextlib
import importlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def load(name):
    spec = importlib.util.spec_from_file_location("w112_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = load("analyze")
sys.path.insert(0, REPO)
planner = importlib.import_module("sim.3dgs.planner")

S, F, L, R = 0, 1, 2, 3          # SAFE_NEUTRAL / FORWARD_SLOW / YAW_LEFT / YAW_RIGHT


def summ(k, cf, to, n=120):
    return {"n_req": n, "k_stop": k, "collide_first": cf, "timeouts": to}


def pair_of(b, c):
    return {"pairs": 120, "b": b, "c": c, "p": 1.0, "delta": 0.0}


class TestVerdict(unittest.TestCase):
    """预注册：室内到点 >= 0.90、撞在前 <= 0.06、超时 <= 0.03；室外到点 >= 0.97；室内配对净增 >= 8。证伪：室内到点 < 0.87 或超时 > 0.05。"""

    def v(self, indoor=(108, 7, 3), outdoor=117, b=8, c=0, complete=True, n=120):
        return analyze.verdict_a(summ(*indoor, n=n), summ(outdoor, 0, 0, n=n), pair_of(b, c), complete)

    def test_all_predictions_met_at_the_boundaries(self):
        r = self.v()                                           # 108/120 = 0.90，7/120 = 0.0583 <= 0.06，3/120 = 0.025，117/120 = 0.975，8 − 0 = 8
        self.assertEqual(r["verdict"], "预测成立")
        self.assertTrue(all(c["ok"] for c in r["checks"]))
        self.assertFalse(any(f["hit"] for f in r["falsify"]))

    def test_indoor_reach_just_below_prediction_is_in_between(self):
        r = self.v(indoor=(107, 7, 3))                         # 0.8917 < 0.90，但 >= 0.87
        self.assertEqual(r["verdict"], "介于预测与证伪之间")
        self.assertFalse(r["checks"][0]["ok"])

    def test_collide_first_limit(self):
        self.assertEqual(self.v(indoor=(108, 7, 3))["verdict"], "预测成立")
        r = self.v(indoor=(108, 8, 3))                         # 8/120 = 0.0667 > 0.06
        self.assertEqual((r["verdict"], r["checks"][1]["ok"]), ("介于预测与证伪之间", False))

    def test_timeout_prediction_limit_and_falsification_limit(self):
        self.assertEqual(self.v(indoor=(108, 7, 4))["verdict"], "介于预测与证伪之间")     # 4/120 = 0.0333 > 0.03，不高于 0.05
        self.assertEqual(self.v(indoor=(108, 7, 6))["verdict"], "介于预测与证伪之间")     # 6/120 = 0.05：不高于 0.05，不算证伪
        r = self.v(indoor=(108, 7, 7))                                                   # 7/120 = 0.0583 > 0.05
        self.assertEqual(r["verdict"], "证伪成立")
        self.assertTrue(r["falsify"][1]["hit"])

    def test_reach_falsification_limit(self):
        self.assertEqual(self.v(indoor=(105, 7, 3))["verdict"], "介于预测与证伪之间")     # 105/120 = 0.875 >= 0.87
        r = self.v(indoor=(104, 7, 3))                                                   # 104/120 = 0.8667 < 0.87
        self.assertEqual(r["verdict"], "证伪成立")
        self.assertTrue(r["falsify"][0]["hit"])

    def test_falsification_wins_over_other_conditions(self):
        r = self.v(indoor=(100, 0, 0), outdoor=120, b=30, c=0)
        self.assertEqual(r["verdict"], "证伪成立")

    def test_outdoor_limit(self):
        self.assertEqual(self.v(outdoor=117)["verdict"], "预测成立")
        r = self.v(outdoor=116)                                # 116/120 = 0.9667 < 0.97
        self.assertEqual((r["verdict"], r["checks"][3]["ok"]), ("介于预测与证伪之间", False))

    def test_pair_net_limit_uses_b_minus_c(self):
        self.assertEqual(self.v(b=8, c=0)["verdict"], "预测成立")
        self.assertEqual(self.v(b=10, c=2)["verdict"], "预测成立")                       # 净增 8
        r = self.v(b=10, c=3)                                                            # 净增 7
        self.assertEqual((r["verdict"], r["checks"][4]["ok"]), ("介于预测与证伪之间", False))
        self.assertEqual(self.v(b=3, c=11)["checks"][4]["value"], "3 − 11 = -8")

    def test_comparisons_are_exact_not_float(self):
        # n = 100：到点 0.90、撞在前 0.06、超时 0.03、室外 0.97 恰好等于阈值，都算满足；证伪的 0.87 / 0.05 是严格不等号
        r = self.v(indoor=(90, 6, 3), outdoor=97, n=100)
        self.assertTrue(all(c["ok"] for c in r["checks"][:4]))
        self.assertEqual(self.v(indoor=(87, 6, 3), n=100)["falsify"][0]["hit"], False)   # 恰好 0.87：不是「低于 0.87」
        self.assertEqual(self.v(indoor=(86, 6, 3), n=100)["falsify"][0]["hit"], True)
        self.assertEqual(self.v(indoor=(90, 6, 5), n=100)["falsify"][1]["hit"], False)   # 恰好 0.05：不是「高于 0.05」
        self.assertEqual(self.v(indoor=(90, 6, 6), n=100)["falsify"][1]["hit"], True)

    def test_incomplete_data_is_not_judged(self):
        self.assertEqual(self.v(complete=False)["verdict"], "未判（数据不全）")
        r = analyze.verdict_a(None, summ(117, 0, 0), pair_of(8, 0), True)
        self.assertEqual(r["verdict"], "未判（数据不全）")
        r = analyze.verdict_a(summ(0, 0, 0, n=0), summ(117, 0, 0), pair_of(8, 0), True)
        self.assertEqual(r["verdict"], "未判（数据不全）")
        self.assertEqual(analyze.verdict_a(summ(108, 7, 3), summ(117, 0, 0), None, True)["verdict"], "未判（数据不全）")

    def test_text_carries_verdict_and_numbers(self):
        r = self.v(indoor=(104, 7, 3))
        self.assertTrue(r["text"].startswith("证伪成立"))
        self.assertIn("104/120", r["text"])
        self.assertIn("证伪条件命中", r["text"])


class TestCounts(unittest.TestCase):
    def rows(self):
        return {("s", 0, 0): {"steps": 6, "labels": [F, F, L, F, S, S], "reach_step": 3, "collision_step": None,
                              "replay": {"branch": ["forward", "forward", "turn_err", "forward", "stop", "stop"]}},
                ("s", 0, 1): {"steps": 3, "labels": [F, S, F], "reach_step": None, "collision_step": 2,
                              "replay": {"branch": ["forward", "wait_swept", "forward"]}},
                ("s", 1, 0): {"steps": 2, "labels": [R, R], "reach_step": None, "collision_step": None, "replay": None}}

    def test_nav_end_is_first_reach_or_collision(self):
        r = self.rows()
        self.assertEqual(analyze.nav_end(r[("s", 0, 0)]), 3)
        self.assertEqual(analyze.nav_end(r[("s", 0, 1)]), 2)
        self.assertEqual(analyze.nav_end(r[("s", 1, 0)]), 1)                          # 都没有 = 整集
        self.assertEqual(analyze.nav_end({"steps": 9, "reach_step": 5, "collision_step": 2}), 2)   # 取较早的

    def test_label_counts_all_and_nav(self):
        cnt, total = analyze.label_counts(self.rows())
        self.assertEqual((cnt, total), ({0: 3, 1: 5, 2: 1, 3: 2}, 11))
        cnt, total = analyze.label_counts(self.rows(), nav=True)
        self.assertEqual((cnt, total), ({0: 1, 1: 5, 2: 1, 3: 2}, 9))                 # 去掉第一集末尾 2 步悬停

    def test_branch_counts_skip_rows_without_replay(self):
        cnt, total = analyze.branch_counts(self.rows())
        self.assertEqual(total, 9)
        self.assertEqual((cnt["forward"], cnt["turn_err"], cnt["stop"], cnt["wait_swept"]), (5, 1, 2, 1))
        cnt, total = analyze.branch_counts(self.rows(), nav=True)
        self.assertEqual((total, cnt["stop"], cnt["forward"]), (7, 0, 5))


class TestClassifyCollision(unittest.TestCase):
    @staticmethod
    def rep(branch, blocked=None, match=None, err=None):
        n = len(branch)
        return {"branch": branch, "blocked": blocked or [False] * n, "match": match or [True] * n, "pred_err": err or [None] * n}

    def test_spawn_collision_wins(self):
        cls, det = analyze.classify_collision(0, [F], None)
        self.assertEqual(cls, analyze.CLS_SPAWN)
        self.assertEqual(analyze.classify_collision(0, [L], self.rep(["turn_err"]))[0], analyze.CLS_SPAWN)

    def test_turn_step_and_hover_step(self):
        cls, det = analyze.classify_collision(2, [F, F, L], self.rep(["forward", "forward", "turn_blocked"]))
        self.assertEqual((cls, det["branch"], det["prev"]), (analyze.CLS_TURN, "turn_blocked", F))
        self.assertEqual(analyze.classify_collision(1, [F, R], None)[0], analyze.CLS_TURN)          # 转向步不需要回放
        self.assertEqual(analyze.classify_collision(2, [F, F, S], self.rep(["forward", "forward", "stop"]))[0], analyze.CLS_HOVER)

    def test_forward_step_after_two_waits_is_a_cap_release(self):
        rep = self.rep(["wait_swept", "wait_swept", "forward"], blocked=[True, True, True])
        cls, det = analyze.classify_collision(2, [S, S, F], rep)
        self.assertEqual(cls, analyze.CLS_CAP)
        self.assertEqual(det["prev_branches"], ["wait_swept", "wait_swept"])

    def test_forward_step_the_swept_check_passed(self):
        rep = self.rep(["turn_err", "forward"], blocked=[None, False], err=[None, 0.31])
        cls, det = analyze.classify_collision(1, [L, F], rep)
        self.assertEqual((cls, det["pred_err_m"]), (analyze.CLS_PASS, 0.31))

    def test_forward_step_without_usable_replay_is_unclassified(self):
        self.assertEqual(analyze.classify_collision(3, [F, F, F, F], None)[0], analyze.CLS_NOREPLAY)
        rep = self.rep(["forward"] * 4, match=[True, True, True, False])
        self.assertEqual(analyze.classify_collision(3, [F, F, F, F], rep)[0], analyze.CLS_NOREPLAY)        # 撞击步复算与记录不一致


# ---- 回放：桩占据（一堵垂直于 x 轴的墙）+ 真 OccAstarTeacher
class WallOcc:
    """沿 +x 的线段只要终点 x >= wall_x 就算撞；另记每次 sweep 的终点（看检查的线段多长）。"""

    def __init__(self, wall_x):
        self.wall_x, self.calls = wall_x, []

    def sweep(self, a, b):
        self.calls.append((a, b))
        return b[0] >= self.wall_x, None


class WallPlanner:
    def __init__(self, wall_x):
        self.occ = WallOcc(wall_x)

    def plan(self, start, goal):
        return [start, goal]

    def aim_point(self, path, x, y, lookahead):
        return path[-1]


WP = {"x": 1e4, "y": 0.0}


def action_rows(n, yaw=0.0, x0=0.0, expert=None):
    """n 步：位姿恒在 x0（只回放决策，位移用 pose_gt）；pose_gt 比 pose_before 前进 0.4 m。"""
    return [{"pose_before": {"x": x0, "y": 0.0, "z": 2.0}, "yaw_before": yaw, "action_expert": (expert or [F] * n)[i],
             "pose_gt": {"x": x0 + 0.4, "y": 0.0, "z": 2.0}} for i in range(n)]


def trace_of(vels):
    """vels[i] = 第 i 步末样本的世界系速度；每步给 2 个样本，末样本才是决策用的。"""
    out = []
    for i, v in enumerate(vels):
        out += [{"step": i, "vel": [9.0, 9.0, 0.0]}, {"step": i, "vel": [v[0], v[1], 0.0]}]
    return out


class TestReplayEpisode(unittest.TestCase):
    def swept(self, wall_x):
        return planner.OccAstarTeacher(WallPlanner(wall_x), WP, tol_deg=18.9, hold_s=1.0, clearance="swept")

    def test_velocity_from_previous_step_vetoes_forward_and_cap_releases(self):
        # 墙在 0.5 m：机头线段（0.4 m）通过；速度 0.5 m/s 时预测位移 0.4259·0.5 + 0.1809 = 0.394，加 reserve 0.1731 = 0.567 > 0.5 被否决
        trace = trace_of([(0.5, 0.0)] * 4)
        res = analyze.replay_episode(self.swept(0.5), action_rows(4, expert=[F, S, S, F]), trace, planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
        self.assertEqual(res["branch"], ["forward", "wait_swept", "wait_swept", "forward"])    # 第 0 步速度 0：预测终点 0.354 < 0.5 通过
        self.assertEqual(res["blocked"], [False, True, True, True])      # 第 3 步检查仍否决，由连续等待上限放行
        self.assertEqual(res["match"], [True, True, True, True])
        self.assertEqual((res["mismatch"], res["n_steps"]), (0, 4))
        res = analyze.replay_episode(self.swept(0.5), action_rows(4, expert=[S, S, F, F]), trace, planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
        self.assertEqual(res["match"], [False, True, False, True])       # 记录与复算不一致的步被数出来
        self.assertEqual(res["mismatch"], 2)

    def test_step_zero_uses_zero_velocity_even_if_the_trace_has_samples(self):
        t = self.swept(0.5)
        rows = action_rows(1)
        res = analyze.replay_episode(t, rows, trace_of([(5.0, 0.0)]), planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
        self.assertEqual(res["branch"], ["forward"])             # 用第 0 步末样本（5 m/s）当决策速度会被否决

    def test_missing_trace_cannot_replay_swept_but_ray_does_not_need_it(self):
        self.assertIsNone(analyze.replay_episode(self.swept(0.5), action_rows(3), None, planner.SWEPT_PRED_A, planner.SWEPT_PRED_B))
        self.assertIsNone(analyze.replay_episode(self.swept(0.5), action_rows(3), trace_of([(0.0, 0.0)]),
                                                 planner.SWEPT_PRED_A, planner.SWEPT_PRED_B))      # 第 2 步缺第 1 步的样本
        ray = planner.OccAstarTeacher(WallPlanner(0.5), WP, tol_deg=18.9)
        res = analyze.replay_episode(ray, action_rows(3), None)
        self.assertEqual(res["branch"], ["forward"] * 3)
        self.assertEqual(res["blocked"], [None] * 3)
        self.assertEqual(res["pred_err"], [None] * 3)

    def test_pred_error_is_prediction_minus_actual_in_the_body_frame(self):
        t = self.swept(10.0)
        rows = action_rows(2)                                    # 实际位移 (0.4, 0)
        res = analyze.replay_episode(t, rows, trace_of([(0.5, 0.0)] * 2), planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
        a, b = planner.SWEPT_PRED_A, planner.SWEPT_PRED_B
        pred1 = (a[0][0] * 0.5 + b[0], a[1][0] * 0.5 + b[1])    # 第 1 步速度 0.5 m/s（世界 = 机体，yaw 0）
        self.assertAlmostEqual(res["pred_err"][1], math.hypot(pred1[0] - 0.4, pred1[1] - 0.0), places=12)
        pred0 = (b[0], b[1])
        self.assertAlmostEqual(res["pred_err"][0], math.hypot(pred0[0] - 0.4, pred0[1] - 0.0), places=12)

    def test_pred_error_rotates_velocity_and_displacement_into_the_body_frame(self):
        t = self.swept(10.0)
        rows = action_rows(2, yaw=90.0)                          # 机头朝 +y
        rows[1]["pose_gt"] = {"x": 0.0, "y": 0.3, "z": 2.0}     # 实际位移：世界 (0, 0.3) = 机体 (0.3, 0)
        res = analyze.replay_episode(t, rows, trace_of([(0.0, 0.5)] * 2), planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
        a, b = planner.SWEPT_PRED_A, planner.SWEPT_PRED_B
        pred = (a[0][0] * 0.5 + b[0], a[1][0] * 0.5 + b[1])      # 世界速度 (0, 0.5) = 机体 (0.5, 0)
        self.assertAlmostEqual(res["pred_err"][1], math.hypot(pred[0] - 0.3, pred[1]), places=12)

    def test_decision_velocity_is_the_previous_steps_last_sample(self):
        # 墙在 0.5 m：速度 0.5 m/s 才会被否决。第 i 步用第 i−1 步的末样本：第 1 步用 0.5（否决），第 2 步用 0.0（通过）
        vels = [(0.5, 0.0), (0.0, 0.0), (0.5, 0.0)]
        res = analyze.replay_episode(self.swept(0.5), action_rows(3), trace_of(vels), planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
        self.assertEqual(res["branch"], ["forward", "wait_swept", "forward"])
        self.assertEqual(res["blocked"], [False, True, False])

    def test_each_episode_needs_a_fresh_teacher_so_waits_do_not_leak(self):
        rows = action_rows(3, expert=[S, S, F])
        trace = trace_of([(0.5, 0.0)] * 3)
        for _ in range(2):
            res = analyze.replay_episode(self.swept(0.5), rows, trace, planner.SWEPT_PRED_A, planner.SWEPT_PRED_B)
            self.assertEqual(res["branch"][:3], ["forward", "wait_swept", "wait_swept"])


class TestMakeTeacher(unittest.TestCase):
    def test_swept_gets_hold_and_clearance_ray_gets_the_old_constructor(self):
        sw = analyze.make_teacher(planner, WallPlanner(1.0), WP, 18.9, "swept", 1.0)
        self.assertEqual((sw.clearance, sw.tol, sw.version), ("swept", 18.9, planner.TEACHER_VERSION_SWEPT))
        ray = analyze.make_teacher(planner, WallPlanner(1.0), WP, 18.9, "ray", 1.0)
        self.assertEqual((ray.clearance, ray.tol), ("ray", 18.9))
        self.assertEqual(analyze.make_teacher(planner, WallPlanner(1.0), WP, 12.0, "ray", 0.5).version, planner.TEACHER_VERSION)   # ray 不带 hold_s，容差取记录值
        with self.assertRaises(ValueError):
            analyze.make_teacher(planner, WallPlanner(1.0), WP, 18.9, "swept", 0.5)      # swept 要 hold 1.0


class _FakeTeacher:
    """记录构造参数的桩教师：回放接线（场景 / 路线 / 容差 / hold / 臂角色）的测试用。"""
    made = []

    def __init__(self, planner_, waypoint, tol_deg=None, hold_s=None, clearance="ray"):
        self.args, self.clearance, self.last_branch = (waypoint, tol_deg, hold_s, clearance), clearance, None
        _FakeTeacher.made.append(self)

    def _swept_blocked(self, st, x, y, z):
        return False

    def __call__(self, rgb, depth, st):
        self.last_branch = "forward"
        return F, None


class TestReplayRowsWiring(unittest.TestCase):
    """replay_rows 的接线：每个臂按角色建教师（swept 带 hold_s 与 clearance，W1-11 臂用旧构造），容差优先取 episode_manifest 的记录值，
    场景只加载一次占据，路线下标交给 apply_route。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.suite = os.path.join(self._td.name, "indoor")
        for arm in ("swept", "teacher-tol"):
            write_episode(self.suite, arm, "sA", 3, 0, GOOD, "reach")
        with open(os.path.join(self.suite, "swept", "sA", "r3", "ep_00000", "episode_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"teacher_yaw_tol_deg": 17.5}, f)
        self.rows = {arm: {("sA", 3, 0): {"arm": arm, "scene": "sA", "route": 3, "dir": os.path.join(self.suite, arm, "sA", "r3", "ep_00000")}}
                     for arm in ("swept", "teacher-tol")}

    def test_wiring(self):
        _FakeTeacher.made = []
        loads, routes, froms = [], [], []
        scene = types.SimpleNamespace(load_scene=lambda path: loads.append(path) or {"waypoint": {"x": 9.0, "y": 0.0}},
                                      apply_route=lambda sc, r: routes.append(r))
        coll = types.SimpleNamespace(from_scene=lambda sc: froms.append(1) or object())
        fake_planner = types.SimpleNamespace(OccPlanner=lambda occ: object(), OccAstarTeacher=_FakeTeacher,
                                             SWEPT_PRED_A=planner.SWEPT_PRED_A, SWEPT_PRED_B=planner.SWEPT_PRED_B)
        cfg = {"scenes_dir": "/scenes", "hold": 1.0, "teacher_tol_deg": 18.9}
        with mock.patch.dict(sys.modules, {"sim.3dgs.scene": scene, "sim.3dgs.collision": coll, "sim.3dgs.planner": fake_planner}):
            analyze.replay_rows(cfg, self.rows, {"swept": "swept", "teacher-tol": "ray"}, repo=self._td.name)
        by_clearance = {t.clearance: t.args for t in _FakeTeacher.made}
        self.assertEqual(by_clearance["swept"], ({"x": 9.0, "y": 0.0}, 17.5, 1.0, "swept"))   # 容差取 manifest 的记录值，带 hold_s
        self.assertEqual(by_clearance["ray"], ({"x": 9.0, "y": 0.0}, 18.9, None, "ray"))                               # 无 manifest 用 config 的值；不带 hold_s
        self.assertEqual(len(_FakeTeacher.made), 2)
        self.assertEqual(routes, [3, 3])
        self.assertEqual(len(froms), 1)                                                                               # 同一场景的两臂只建一次占据
        self.assertEqual(set(loads), {"/scenes/sA/scene.json"})
        self.assertIsNone(self.rows["swept"][("sA", 3, 0)]["replay"])                      # swept 没有 trace 速度：无法回放，记 None
        self.assertEqual(self.rows["teacher-tol"][("sA", 3, 0)]["replay"]["n_steps"], 5)   # ray 不需要 trace


class TestSpawnStats(unittest.TestCase):
    def test_counts_pairs_with_different_start_and_step0_collisions(self):
        def row(x, cs=None):
            return {"start_pose": {"x": x, "y": 0.0, "yaw_deg": 0.0}, "collision_step": cs}
        a = {("s", 0, 0): row(0.0, 0), ("s", 0, 1): row(1.0), ("s", 0, 2): row(2.0, 0)}
        b = {("s", 0, 0): row(0.5), ("s", 0, 1): row(1.0), ("s", 0, 2): row(2.0, 0), ("s", 1, 0): row(9.0)}
        r = analyze.spawn_stats(a, b)
        self.assertEqual(r, {"pairs": 3, "start_pose_differs": 1, "collision_step0_a": 2, "collision_step0_b": 1})


# ---- 合成实验目录：<out>/<suite>/<arm>/<scene>/r<route>/{done.json, ep_*/{metrics.json, actions.jsonl}}
def write_episode(suite_dir, arm, scene, route, ep, seq, outcome, start_x=0.0, cstep=None, expert=None, reach_at=None):
    """seq = action_executed 序列；expert = action_expert 序列（缺省与 seq 相同）；outcome = reach / collide / timeout；reach_at = 到点步下标（缺省末步）。"""
    d = os.path.join(suite_dir, arm, scene, f"r{route}", "ep_%05d" % ep)
    os.makedirs(d, exist_ok=True)
    cs = (len(seq) - 1 if cstep is None else cstep) if outcome == "collide" else None
    expert = expert or seq
    rows = [{"step": i, "action_expert": expert[i], "action_executed": a, "collision": cs == i,
             "pose_before": {"x": float(i), "y": 0.0, "z": 2.0}, "yaw_before": 10.0 * i, "dyaw_deg": 0.0, "displacement_m": 0.1,
             "pose_gt": {"x": float(i) + 1.0, "y": 0.0, "z": 2.0, "yaw_deg": 10.0 * i}} for i, a in enumerate(seq)]
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    reach = outcome == "reach"
    m = {"success_stop": reach, "success_v2": reach, "collision": outcome == "collide", "collide_first": outcome == "collide",
         "timeout": outcome == "timeout", "steps": len(seq), "reach_step": (len(seq) - 1 if reach_at is None else reach_at) if reach else None,
         "collision_step": cs,
         "spl": 0.5, "spl_stop": 0.5 if reach else 0.0, "final_wp_dist_m": 0.4 if reach else 3.0, "min_wp_dist_m": 0.4,
         "arm": arm, "scene": scene, "route": route, "episode": ep, "seed": 8000 + 100 * route + ep,
         "start_pose": {"x": start_x, "y": 0.0, "yaw_deg": 0.0}}
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)


def finish_route(suite_dir, arm, scene, route, k):
    d = os.path.join(suite_dir, arm, scene, f"r{route}")
    with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": [{"episode": i} for i in range(k)], "failures": []}, f)


GOOD = [L, F, F, F, F]


class Synthetic(unittest.TestCase):
    """室内合成：1 场景 × 2 路线 × K=3。W1-11 臂 teacher-tol：r0 = 撞、到点、到点，r1 = 到点、超时、到点；
    臂 A swept：r0 = 到点 × 3，r1 = 撞、到点、到点；臂 B swept-spawn：6 集全到点，r0 ep0 起点与臂 A 不同。
    配对（A 对 W1-11）：r0 ep0 新成旧败，r1 ep1 新成旧败，r1 ep0 旧成新败 -> b = 2，c = 1。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.out = self._td.name
        self.suite = os.path.join(self.out, "indoor")
        self.cfg = {"scenes": ["sA"], "routes": {"sA": [0, 1]}, "k": 3, "hold": 1.0, "seed0": 8000, "jitter": 0.2, "step_frac": 0.5,
                    "step_pad": 15, "step_cap": 100, "teacher_tol_deg": 18.9, "arm_a": "swept", "arm_b": "swept-spawn", "w111_arm": "teacher-tol"}
        os.makedirs(self.suite)
        with open(os.path.join(self.suite, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)
        data = {"teacher-tol": {0: [(GOOD, "collide"), (GOOD, "reach"), (GOOD, "reach")], 1: [(GOOD, "reach"), ([S] * 6, "timeout"), (GOOD, "reach")]},
                "swept": {0: [(GOOD, "reach")] * 3, 1: [([F, F, F, F, L], "collide"), (GOOD, "reach"), (GOOD, "reach")]},
                "swept-spawn": {0: [(GOOD, "reach")] * 3, 1: [(GOOD, "reach")] * 3}}
        for arm, routes in data.items():
            for route, eps in routes.items():
                for i, (seq, outcome) in enumerate(eps):
                    write_episode(self.suite, arm, "sA", route, i, seq, outcome, start_x=0.5 if (arm == "swept-spawn" and route == 0 and i == 0) else 0.0)
                finish_route(self.suite, arm, "sA", route, 3)

    def test_summaries_pairs_and_discordant_episodes(self):
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        s = res["arms"]
        self.assertEqual((s["swept"]["k_stop"], s["swept"]["collide_first"], s["swept"]["timeouts"]), (5, 1, 0))
        self.assertEqual((s["teacher-tol"]["k_stop"], s["teacher-tol"]["collide_first"], s["teacher-tol"]["timeouts"]), (4, 1, 1))
        self.assertEqual((s["swept-spawn"]["k_stop"], s["swept-spawn"]["failed"]), (6, 0))
        p = res["pairs"]["A_vs_W"]
        self.assertEqual((p["pairs"], p["b"], p["c"]), (6, 2, 1))
        d = res["discordant"]
        self.assertEqual(d["A_success_W_fail"], [["sA", 0, 0], ["sA", 1, 1]])
        (x,) = d["A_fail_W_success"]
        self.assertEqual((x["key"], x["A_outcome"], x["W_outcome"]), (["sA", 1, 0], "撞在前", "到点"))
        self.assertEqual((x["first_diff"]["step"], x["first_diff"]["key"]), (0, "action_expert"))     # [L, F, ...] 对 [F, F, ...]：第 0 步标签就不同
        self.assertEqual((len(d["A_success_W_fail"]), len(d["A_fail_W_success"])), (p["b"], p["c"]))

    def test_arm_b_pairs_and_spawn_stats(self):
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        self.assertEqual(res["pairs"]["B_vs_W"]["pairs"], 6)
        self.assertEqual(res["pairs"]["B_vs_A"]["pairs"], 6)
        self.assertEqual(res["spawn"]["start_pose_differs"], 1)

    def test_collide_first_table_without_replay_classifies_what_it_can(self):
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        (t,) = res["collide_first"]["swept"]
        self.assertEqual((t["scene"], t["route"], t["episode"], t["step"]), ("sA", 1, 0, 4))
        self.assertEqual(t["class"], analyze.CLS_TURN)               # 撞击步动作 = 左转，不需要回放
        self.assertEqual(res["collide_first"]["swept-spawn"], [])
        self.assertNotIn("teacher-tol", res["collide_first"])        # W1-11 臂不归类
        self.assertEqual(res["branches"], {})

    def test_timeout_table_lists_the_timeout_episode(self):
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        self.assertEqual(res["timeouts"]["swept"], [])
        (t,) = res["timeouts"]["teacher-tol"]
        self.assertEqual((t["route"], t["episode"], t["steps"], t["replayed"]), (1, 1, 6, False))

    def test_label_distribution_counts_safe_steps(self):
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        lb = res["labels"]["teacher-tol"]
        self.assertEqual(lb["total"], 5 * 5 + 6)                       # 5 集各 5 步 + 超时集 6 步
        self.assertEqual(lb["all"]["n_safe"], 6)                       # 超时集 6 步全是 SAFE
        self.assertEqual((lb["all"]["n_left"], lb["all"]["n_forward"], lb["all"]["n_right"]), (5, 20, 0))
        self.assertEqual(res["labels"]["swept"]["all"]["n_safe"], 0)
        self.assertEqual(res["labels"]["swept"]["total"], 6 * 5)        # 6 集各 5 步（撞的一集也是 5 步）
        self.assertEqual((lb["nav_total"], lb["nav"]["n_safe"]), (5 * 5 + 6, 6))   # 合成集都在末步到点 / 撞 / 超时，导航阶段 = 整集

    def test_unfinished_route_makes_the_suite_incomplete(self):
        os.remove(os.path.join(self.suite, "swept", "sA", "r1", "done.json"))
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        self.assertFalse(res["complete"])
        self.assertEqual(res["arms"]["swept"]["n_req"], 3)

    def test_acceptance_writes_report_json_and_an_unjudged_verdict_without_outdoor(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = analyze.main(["acceptance", self.out, "--suites", "indoor"])
        self.assertEqual(rc, 0)
        with open(os.path.join(self.out, "analysis", "acceptance.json"), encoding="utf-8") as f:
            j = json.load(f)
        self.assertEqual(j["verdict_a"]["verdict"], "未判（数据不全）")           # 没有室外
        self.assertEqual(j["suites"]["indoor"]["arms"]["swept"]["k_stop"], 5)
        with open(os.path.join(self.out, "analysis", "acceptance.md"), encoding="utf-8") as f:
            md = f.read()
        for head in ("预注册判决", "到点成功", "McNemar", "仍然撞在前的集逐集归类", "超时的集", "教师标签", "未回放"):
            self.assertIn(head, md)

    def test_arm_b_can_be_absent(self):
        self.cfg["arm_b"] = None
        with open(os.path.join(self.suite, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)
        res = analyze.analyze_suite(self.out, "indoor", replay=False)
        self.assertNotIn("B_vs_W", res["pairs"])
        self.assertNotIn("spawn", res)
        self.assertEqual(res["arms_order"], ["swept", "teacher-tol"])


class TestNavPhaseAndLabels(unittest.TestCase):
    """到点后教师还会连出悬停（goal-stop 的 3 步 SAFE）：导航阶段的数字要截到到点步，标签用教师给的 action_expert。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.out = self._td.name
        self.suite = os.path.join(self.out, "indoor")
        self.cfg = {"scenes": ["sA"], "routes": {"sA": [0]}, "k": 1, "hold": 1.0, "seed0": 8000, "jitter": 0.2, "teacher_tol_deg": 18.9,
                    "arm_a": "swept", "arm_b": None, "w111_arm": "teacher-tol"}
        os.makedirs(self.suite)
        with open(os.path.join(self.suite, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)
        # 臂 A：5 步，第 2 步到点，后 2 步悬停；教师标签与执行动作故意不同（第 1 步标签 L 执行 F）
        write_episode(self.suite, "swept", "sA", 0, 0, [F, F, F, S, S], "reach", reach_at=2, expert=[F, L, F, S, S])
        write_episode(self.suite, "teacher-tol", "sA", 0, 0, [F, F, F, F], "reach", reach_at=3)
        for arm in ("swept", "teacher-tol"):
            finish_route(self.suite, arm, "sA", 0, 1)

    def test_nav_counts_stop_at_the_reach_step(self):
        lb = analyze.analyze_suite(self.out, "indoor", replay=False)["labels"]["swept"]
        self.assertEqual((lb["total"], lb["all"]["n_safe"]), (5, 2))
        self.assertEqual((lb["nav_total"], lb["nav"]["n_safe"]), (3, 0))

    def test_labels_are_the_experts_not_the_executed(self):
        lb = analyze.analyze_suite(self.out, "indoor", replay=False)["labels"]["swept"]
        self.assertEqual((lb["all"]["n_left"], lb["all"]["n_forward"]), (1, 2))      # 执行的是 F F F，标签是 F L F

    def test_replay_arguments_follow_the_arm_roles(self):
        calls = []
        orig = analyze.replay_rows
        analyze.replay_rows = lambda cfg, rows_by_arm, clearance_of, occ_cache=None, repo=None: calls.append((sorted(rows_by_arm), dict(clearance_of)))
        try:
            analyze.analyze_suite(self.out, "indoor", replay=True)
        finally:
            analyze.replay_rows = orig
        self.assertEqual(calls, [(["swept", "teacher-tol"], {"swept": "swept", "teacher-tol": "ray"})])      # W1-11 的教师是 ray，不能按 swept 回放


class TestRegressCommand(unittest.TestCase):
    def test_regress_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            for arm in ("new", "old"):
                write_episode(os.path.join(td, arm), "x", "sA", 0, 0, GOOD, "reach")
            new, old = os.path.join(td, "new", "x"), os.path.join(td, "old", "x")
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                self.assertEqual(analyze.main(["regress", new, old]), 0)
            self.assertIn("确定性回归：1 集，逐步逐字段完全一致 1 集", buf.getvalue())
            p = os.path.join(new, "sA", "r0", "ep_00000", "actions.jsonl")
            rows = [json.loads(x) for x in open(p, encoding="utf-8")]
            rows[2]["pose_before"]["x"] += 1e-9
            with open(p, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(["regress", new, old]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
