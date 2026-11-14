#!/usr/bin/env python3
"""vln.py（VLN 集与指标）、run_episode 的智能体喊停、录像指令横幅的小测（mock 渲染器，纯 CPU）。

  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_vln.py
"""
import importlib
import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
vln = importlib.import_module("sim.3dgs.vln")
collect = importlib.import_module("sim.3dgs.collect_3dgs")
scene_mod = importlib.import_module("sim.3dgs.scene")
runner = importlib.import_module("sim.3dgs.run_vln")
reactive = importlib.import_module("sim.3dgs.reactive")
dyn3dgs = importlib.import_module("sim.3dgs.dynamics")

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

SCENE = {"name": "s", "routes": [
    {"name": "r0", "sage": {"split": "train", "instructions": [
        {"instruction_type": "Attribute-based", "start": "sofa_1", "end": "lamp_2", "text": "Go to the tall lamp."},
        {"instruction_type": "Area-based", "start": "sofa_1", "end": "lamp_2", "text": "Head to the reading corner."}]}},
    {"name": "r1", "sage": {"split": "test_Instruction_Unseen", "instructions": [
        {"instruction_type": "Attribute-based", "start": "bed_1", "end": "tv_1", "text": "Find the black TV."}]}},
    {"name": "old"}]}     # 不是 SAGE 导入的航线：没有指令，不出集
GOAL = {"x": 4.0, "y": 0.0}


def _steps(xy, collide_at=None):
    """xy[0] 是出发点，之后每步一个终点；collide_at 起每步都标撞（run_episode 撞后整段标撞的语义）。"""
    actions = [{"pose_before": {"x": x, "y": y}, "collision": collide_at is not None and i >= collide_at}
               for i, (x, y) in enumerate(xy[:-1])]
    oracle = [{"pose_gt": {"x": x, "y": y}} for x, y in xy[1:]]
    return actions, oracle


class TestEpisodes(unittest.TestCase):
    def test_every_instruction_is_an_episode(self):
        eps = vln.episodes(SCENE)
        self.assertEqual([e["key"] for e in eps], ["s/r0/0", "s/r0/1", "s/r1/0"])
        self.assertEqual((eps[2]["route_index"], eps[2]["goal_object"], eps[2]["split"]),
                         (1, "tv_1", "test_Instruction_Unseen"))

    def test_filters(self):
        self.assertEqual([e["key"] for e in vln.episodes(SCENE, splits=["test_Instruction_Unseen"])], ["s/r1/0"])
        self.assertEqual([e["key"] for e in vln.episodes(SCENE, types=["Area-based"])], ["s/r0/1"])

    def test_one_per_route_rotates_by_type_name(self):
        # r0 按句式名排序是 [Area-based(1), Attribute-based(0)]，与指令在航线里的原顺序无关
        eps = vln.episodes(SCENE)
        self.assertEqual([e["key"] for e in vln.one_per_route(eps, seed=0)], ["s/r0/1", "s/r1/0"])
        self.assertEqual([e["key"] for e in vln.one_per_route(eps, seed=1)], ["s/r0/0", "s/r1/0"])

    def test_external_grounder_changes_policy_goal_and_rejects_unknown(self):
        selector = importlib.import_module("sim.3dgs.grounder").LexicalGrounder()
        catalog = {"lamp_2": {"x": 7.0, "y": -1.0}}
        result, goal = runner.resolve_grounding(selector, catalog, "Go to the tall lamp.")
        self.assertTrue(result.accepted)
        self.assertEqual(goal, catalog["lamp_2"])
        rejected, no_goal = runner.resolve_grounding(selector, catalog, "Fly to the red plant.")
        self.assertFalse(rejected.accepted)
        self.assertIsNone(no_goal)


class TestMetrics(unittest.TestCase):
    def test_stop_inside_radius_is_success(self):
        a, o = _steps([(0, 0), (1, 0), (2, 0), (3.5, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True, shortest_m=3.0)
        self.assertTrue(m["success"] and m["oracle_success"])
        self.assertAlmostEqual(m["ne_m"], 0.5)
        self.assertAlmostEqual(m["path_m"], 3.5)
        self.assertAlmostEqual(m["spl"], 3.0 / 3.5)

    def test_flying_through_without_stop_is_only_oracle_success(self):
        # 旧 3dgs.pilot.v2 判据会算成功；VLN 要求自己停在半径里
        a, o = _steps([(0, 0), (2, 0), (4, 0), (6, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=False, shortest_m=3.0)
        self.assertEqual((m["success"], m["oracle_success"], m["spl"]), (False, True, 0.0))
        self.assertAlmostEqual(m["ne_m"], 2.0)

    def test_stop_short_of_goal_fails(self):
        a, o = _steps([(0, 0), (1, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True)
        self.assertEqual((m["success"], m["oracle_success"], m["spl"]), (False, False, None))

    def test_motion_diagnostics_report_switches_and_safe_time(self):
        actions = [
            {"pose_before": {"x": 0, "y": 0}, "action_executed": 1, "collision": False},
            {"pose_before": {"x": 1, "y": 0}, "action_executed": 2, "collision": False},
            {"pose_before": {"x": 1, "y": 0}, "action_executed": 3, "collision": False},
            {"pose_before": {"x": 1, "y": 0}, "action_executed": 0, "collision": False},
        ]
        oracle = [{"pose_gt": {"x": 1, "y": 0}}, {"pose_gt": {"x": 1, "y": 0}},
                  {"pose_gt": {"x": 1, "y": 0}}, {"pose_gt": {"x": 1, "y": 0}}]
        m = vln.episode_metrics(actions, oracle, GOAL, 1.0, stopped=False)
        self.assertEqual((m["action_steps_observed"], m["action_pairs_observed"], m["action_switches"],
                          m["safe_steps"], m["turn_reversals"]), (4, 3, 3, 1, 1))
        self.assertAlmostEqual(m["action_switches_per_m"], 3.0)
        self.assertAlmostEqual(m["safe_steps_per_m"], 1.0)

    def test_unknown_action_does_not_bridge_or_report_zero(self):
        actions = [
            {"pose_before": {"x": 0, "y": 0}, "action_executed": 1, "collision": False},
            {"pose_before": {"x": 1, "y": 0}, "action_executed": 99, "collision": False},
            {"pose_before": {"x": 1, "y": 0}, "action_executed": 3, "collision": False},
        ]
        oracle = [{"pose_gt": {"x": 1, "y": 0}}] * 3
        m = vln.episode_metrics(actions, oracle, GOAL, 1.0, stopped=False)
        self.assertEqual((m["action_steps_observed"], m["action_pairs_observed"], m["action_switches"],
                          m["safe_steps"], m["turn_reversals"]), (2, 0, None, 0, None))
        self.assertIsNone(m["action_switches_per_m"])

    def test_old_actions_without_action_field_do_not_claim_zero_diagnostics(self):
        a, o = _steps([(0, 0), (1, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=False)
        self.assertEqual((m["action_steps_observed"], m["action_pairs_observed"], m["action_switches"],
                          m["safe_steps"], m["safe_ratio"]), (0, 0, None, None, None))

    def test_zero_path_does_not_report_per_meter_rate(self):
        actions = [{"pose_before": {"x": 0, "y": 0}, "action_executed": 0, "collision": False}]
        oracle = [{"pose_gt": {"x": 0, "y": 0}}]
        m = vln.episode_metrics(actions, oracle, GOAL, 1.0, stopped=False)
        self.assertEqual(m["safe_steps"], 1)
        self.assertAlmostEqual(m["safe_ratio"], 1.0)
        self.assertIsNone(m["safe_steps_per_m"])

    def test_poses_after_collision_do_not_count(self):
        # 第 1 步（0 起）撞：它的终点 (4, 0) 已在障碍里，不算进过半径
        a, o = _steps([(0, 0), (2.5, 0), (4, 0), (4, 0)], collide_at=1)
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True)
        self.assertEqual((m["success"], m["oracle_success"], m["collision"]), (False, False, True))

    def test_summarize(self):
        rows = [{"success": True, "oracle_success": True, "spl": 0.8, "ne_m": 0.5, "collision": False, "stopped": True, "type": "A"},
                {"success": False, "oracle_success": True, "spl": 0.0, "ne_m": 2.5, "collision": True, "stopped": False, "type": "B"},
                {"success": False, "oracle_success": False, "spl": None, "ne_m": 4.0, "collision": False, "stopped": True, "type": "B"}]
        s = vln.summarize(rows)
        self.assertEqual((s["n"], s["n_spl"], s["ne_median_m"]), (3, 2, 2.5))
        for key, want in (("sr", 1 / 3), ("osr", 2 / 3), ("spl", 0.4), ("ne_mean_m", 7 / 3),
                          ("collision_rate", 1 / 3), ("stop_rate", 2 / 3)):
            self.assertAlmostEqual(s[key], want, msg=key)
        self.assertEqual(vln.summarize_by(rows, "type")["B"]["n"], 2)
        self.assertIsNone(vln.summarize([])["sr"])


class TestReactiveGoalPolicy(unittest.TestCase):
    def _state(self, x=0.0, y=0.0, yaw=0.0):
        return SimpleNamespace(x=x, y=y, yaw=yaw)

    def test_turns_towards_goal_without_global_planner(self):
        policy = reactive.ReactiveGoalPolicy({"x": 0.0, "y": 5.0})
        depth = np.full((10, 12), 4.0, dtype=np.float32)
        action, _ = policy(None, depth, self._state())
        self.assertEqual(action, dyn3dgs.YAW_LEFT)

    def test_same_start_changes_action_when_grounded_goal_changes_side(self):
        depth = np.full((10, 12), 4.0, dtype=np.float32)
        state = self._state()
        left, right = (reactive.ReactiveGoalPolicy({"x": 0.0, "y": y})
                       for y in (5.0, -5.0))
        action_left, _ = left(None, depth, state)
        action_right, _ = right(None, depth, state)
        self.assertEqual((action_left, action_right), (dyn3dgs.YAW_LEFT, dyn3dgs.YAW_RIGHT))

    def test_stops_inside_goal_radius(self):
        policy = reactive.ReactiveGoalPolicy({"x": 0.5, "y": 0.0})
        depth = np.full((10, 12), 4.0, dtype=np.float32)
        action, _ = policy(None, depth, self._state())
        self.assertEqual((action, policy.stop_requested), (collect.SAFE_NEUTRAL, True))

    def test_turns_to_the_clearer_side_when_blocked(self):
        policy = reactive.ReactiveGoalPolicy({"x": 5.0, "y": 0.0})
        depth = np.full((10, 12), 4.0, dtype=np.float32)
        depth[4:6, 4:8] = 0.5
        depth[4:6, 0:4] = 2.0
        action, _ = policy(None, depth, self._state())
        self.assertEqual(action, dyn3dgs.YAW_RIGHT)

    def test_keeps_avoidance_turn_until_front_opens(self):
        policy = reactive.ReactiveGoalPolicy({"x": 5.0, "y": 0.0})
        first = np.full((10, 12), 4.0, dtype=np.float32)
        first[4:6, 4:8] = 0.5
        first[4:6, 0:4] = 1.0
        first[4:6, 8:12] = 2.0
        second = first.copy()
        second[4:6, 0:4] = 3.0
        second[4:6, 8:12] = 1.0
        a1, _ = policy(None, first, self._state())
        a2, _ = policy(None, second, self._state())
        self.assertEqual((a1, a2), (dyn3dgs.YAW_RIGHT, dyn3dgs.YAW_RIGHT))

    def test_requires_three_clear_frames_before_releasing_avoidance(self):
        policy = reactive.ReactiveGoalPolicy({"x": 0.0, "y": 5.0})
        blocked = np.full((10, 12), 4.0, dtype=np.float32)
        blocked[4:6, 4:8] = 0.5
        blocked[4:6, 0:4] = 1.0
        blocked[4:6, 8:12] = 2.0
        clear = np.full((10, 12), 4.0, dtype=np.float32)
        policy(None, blocked, self._state())
        actions = [policy(None, clear, self._state())[0] for _ in range(3)]
        self.assertEqual(actions[:2], [dyn3dgs.YAW_RIGHT, dyn3dgs.YAW_RIGHT])
        self.assertEqual(actions[2], dyn3dgs.YAW_LEFT)

    def test_all_failed_is_zero_success_without_invented_motion_metrics(self):
        s = vln.summarize([], [{"error": "render failed"}])
        self.assertEqual((s["n"], s["n_requested"], s["n_failed"], s["error_rate"]), (0, 1, 1, 1.0))
        self.assertEqual((s["sr"], s["osr"]), (0.0, 0.0))
        self.assertIsNone(s["spl"])
        self.assertIsNone(s["collision_rate"])


class Scripted:
    """按顺序出动作的策略；序列用完就重复最后一个。"""
    name = "forward"

    def __init__(self, seq, stop_at=None):
        self.seq, self.i = list(seq), 0
        self.stop_at, self.stop_requested = stop_at, False

    def __call__(self, rgb, depth, state):
        a = self.seq[min(self.i, len(self.seq) - 1)]
        self.stop_requested = self.i == self.stop_at
        self.i += 1
        return a, None


class TestAgentStop(unittest.TestCase):
    def test_safe_streak_ends_episode_only_when_enabled(self):
        F, S = collect.FORWARD_SLOW, collect.SAFE_NEUTRAL
        seq = [F, S, S, F, S, S, S, F]
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
            renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
            occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)
            params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
            identity = collect.scene_identity(scene)
            scene["vln"] = {"text": "Go to the end of the corridor."}
            m, err = collect.run_episode(scene, os.path.join(tmp, "stop"), renderer, occ, params, Scripted(seq), 10, 0.5,
                                         split=identity["split"], identity=identity, stop_after_safe=vln.STOP_STREAK)
            self.assertIsNone(err)
            self.assertEqual((m["num_steps"], m["stopped_by_agent"], m["completed"]), (7, True, True))
            self.assertEqual(m["vln"], {"text": "Go to the end of the corridor."})
            m, _ = collect.run_episode(scene, os.path.join(tmp, "old"), renderer, occ, params, Scripted(seq), 10, 0.5,
                                       split=identity["split"], identity=identity)
            self.assertEqual((m["num_steps"], m["stopped_by_agent"], m["stop_after_safe"]), (10, False, None))


class TestExplicitStop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        scene_dir = os.path.join(self.tmp.name, "scene")
        scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        self.scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
        self.renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(self.scene, True)
        self.occ = importlib.import_module("sim.3dgs.collision").from_scene(self.scene)
        self.params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
        self.identity = collect.scene_identity(self.scene)

    def run_policy(self, policy, **kw):
        out = os.path.join(self.tmp.name, "episode")
        manifest, err = collect.run_episode(
            self.scene, out, self.renderer, self.occ, self.params, policy, 10, 0.5,
            split=self.identity["split"], identity=self.identity, stop_on_request=True, **kw)
        self.assertIsNone(err)
        actions = runner._read_jsonl(os.path.join(out, "actions.jsonl"))
        oracle = runner._read_jsonl(os.path.join(out, "oracle.jsonl"))
        metrics = vln.episode_metrics(actions, oracle, self.scene["waypoint"], 1.0,
                                      manifest["stopped_by_agent"])
        return manifest, actions, metrics

    def test_safe_at_goal_or_away_never_implies_completion(self):
        for at_goal in (False, True):
            with self.subTest(at_goal=at_goal):
                if at_goal:
                    self.scene["waypoint"] = dict(self.scene["spawn"])
                policy = Scripted([collect.SAFE_NEUTRAL])
                policy.stop_radius = 1.0      # 旧教师自动 goal-stop 在显式模式中必须禁用
                man, actions, metrics = self.run_policy(policy)
                self.assertEqual(man["num_steps"], 10)
                self.assertFalse(man["stopped_on_goal"] or man["stopped_by_agent"])
                self.assertEqual(man["terminal_reason"], "budget_exhausted")
                self.assertFalse(any(a["stop_requested"] for a in actions))
                self.assertFalse(metrics["success"])

    def test_explicit_stop_maps_to_safe_and_wrong_goal_fails(self):
        man, actions, metrics = self.run_policy(Scripted([collect.FORWARD_SLOW], stop_at=0))
        self.assertEqual((man["num_steps"], man["terminal_reason"]), (1, "agent_stop"))
        self.assertTrue(man["stopped_by_agent"] and man["stop_on_request"])
        self.assertTrue(actions[0]["stop_requested"])
        self.assertEqual(actions[0]["action_executed"], collect.SAFE_NEUTRAL)
        self.assertEqual((metrics["success"], metrics["stopped"]), (False, True))

    def test_explicit_stop_at_goal_succeeds(self):
        self.scene["waypoint"] = dict(self.scene["spawn"])
        man, _, metrics = self.run_policy(Scripted([collect.SAFE_NEUTRAL], stop_at=0))
        self.assertEqual(man["terminal_reason"], "agent_stop")
        self.assertTrue(metrics["success"])

    def test_collision_overrides_simultaneous_stop(self):
        self.scene["waypoint"] = dict(self.scene["spawn"])
        with mock.patch.object(collect, "_step_collision", return_value=(True, [0, 0, 2])):
            man, actions, metrics = self.run_policy(Scripted([collect.SAFE_NEUTRAL], stop_at=0))
        self.assertTrue(actions[0]["stop_requested"])
        self.assertFalse(man["stopped_by_agent"] or metrics["success"])
        self.assertEqual(man["terminal_reason"], "collision")

    def test_conflicting_or_missing_stop_contract_rejected(self):
        with self.assertRaisesRegex(ValueError, "互斥"):
            self.run_policy(Scripted([0]), stop_after_safe=3)
        guarded = importlib.import_module("sim.3dgs.guard").OracleGuardedPolicy(
            collect.ForwardPolicy(), lambda x, y: (x + 1, y), 0)
        with self.assertRaisesRegex(ValueError, "bool stop_requested"):
            self.run_policy(guarded)

    def test_teacher_reset_failure_and_safety_wait_do_not_request_stop(self):
        planner = mock.Mock()
        teacher = runner.planner_mod.OccAstarTeacher(planner, {"x": 0, "y": 0}, hold_s=1.0, clearance="swept")
        state = SimpleNamespace(x=0.0, y=0.0, z=2.0, yaw=0.0)
        self.assertEqual(teacher(None, None, state)[0], collect.SAFE_NEUTRAL)
        self.assertTrue(teacher.stop_requested)
        teacher.reset()
        self.assertFalse(teacher.stop_requested)
        teacher.set_waypoint({"x": 10, "y": 0})
        planner.plan.return_value = None
        self.assertEqual(teacher(None, None, state)[0], collect.SAFE_NEUTRAL)
        self.assertEqual(teacher.last_branch, "plan_fail")
        self.assertFalse(teacher.stop_requested)
        planner.plan.return_value = [(0, 0), (10, 0)]
        planner.aim_point.return_value = (10, 0)
        planner.occ.sweep.return_value = (False, None)
        with mock.patch.object(teacher, "_swept_blocked", return_value=True):
            self.assertEqual(teacher(None, None, state)[0], collect.SAFE_NEUTRAL)
        self.assertEqual(teacher.last_branch, "wait_swept")
        self.assertFalse(teacher.stop_requested)

    def test_dagger_does_not_leak_expert_stop_into_student_execution(self):
        expert = runner.planner_mod.OccAstarTeacher(mock.Mock(), {"x": 0, "y": 0})
        student = mock.Mock(student_cond=None, stop_requested=False)
        student.return_value = (collect.SAFE_NEUTRAL, None)
        state = SimpleNamespace(x=0.0, y=0.0, z=2.0, yaw=0.0)
        with mock.patch.object(collect, "StudentAdapter", return_value=student):
            for beta in (0.0, 1.0):
                dagger = collect.DaggerPolicy(None, expert, beta, np.random.default_rng(0))
                dagger(None, None, state)
                self.assertTrue(expert.stop_requested)
                self.assertEqual(dagger.stop_requested, bool(beta))
                dagger.reset()
                self.assertFalse(dagger.stop_requested)


class TestRunner(unittest.TestCase):
    def run_scene(self, outcome=None, records=None):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        for i, route in enumerate(scene["routes"]):
            route.update(spawn={"x": 0.0, "y": 0.0, "z": 2.0},
                         waypoint={"x": 4.0 + i, "y": 0.0, "z": 2.0})
        scene_mod.apply_route(scene, 0)
        actions, oracle = _steps([(0, 0), (1, 0), (4, 0)])
        manifest = {"stopped_by_agent": True, "validity": {"physics_verified": True, "camera_verified": True}}
        args = SimpleNamespace(split=None, all_instructions=False, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=0, stop_mode="explicit")
        with ExitStack() as stack:
            for module, attr, value in ((scene_mod, "load_scene", scene),
                                       (scene_mod, "read_gaussian_ply", None),
                                       (runner.collision_mod, "from_scene", None),
                                       (runner.renderer_mod, "make_renderer", None),
                                       (runner.planner_mod, "OccPlanner", None),
                                       (runner, "make_policy", None)):
                stack.enter_context(mock.patch.object(module, attr, return_value=value))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode",
                side_effect=outcome or [(manifest, None), (manifest, None)]))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=records if records is not None else [actions, oracle] * 2))
            shortest = stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=3.0))
            rows, failures = runner.run_scene("fixture.json", args, None)
        return rows, failures, run.call_args_list, shortest.call_args_list

    def test_eval_is_not_training_and_task_id_tracks_each_route(self):
        _, _, calls, _ = self.run_scene()
        self.assertEqual([c.kwargs.get("split_policy") for c in calls], ["diag-vln-eval"] * 2)
        self.assertNotEqual(calls[0].kwargs["identity"]["task_id"], calls[1].kwargs["identity"]["task_id"])
        self.assertTrue(all(c.kwargs["stop_on_request"] for c in calls))
        self.assertTrue(all(c.kwargs["stop_after_safe"] is None for c in calls))

    def test_lexical_goal_changes_with_same_start_and_reports_both_targets(self):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        scene["routes"] = [{"name": "r0", "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                             "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0},
                             "sage": {"split": "train", "instructions": [
                                 {"instruction_type": "A", "start": "start", "end": "lamp_2",
                                  "text": "Go to the tall lamp."},
                                 {"instruction_type": "B", "start": "start", "end": "tv_1",
                                  "text": "Go to the black TV."}]}}]
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=0, stop_mode="explicit")
        goals = []
        runs = [_steps([(0, 0), (7, -1), (7, -1)]),
                _steps([(0, 0), (8, 2), (8, 2)])]
        manifest = {"stopped_by_agent": True,
                    "validity": {"physics_verified": True, "camera_verified": True}}
        selector = importlib.import_module("sim.3dgs.grounder").LexicalGrounder()
        catalog = {"lamp_2": {"x": 7.0, "y": -1.0}, "tv_1": {"x": 8.0, "y": 2.0}}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            stack.enter_context(mock.patch.object(runner, "make_policy",
                side_effect=lambda name, planner, waypoint, hold, stop_radius=1.0:
                    goals.append(dict(waypoint)) or SimpleNamespace(name="teacher")))
            stack.enter_context(mock.patch.object(collect, "run_episode", side_effect=[(manifest, None), (manifest, None)]))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[item for pair in runs for item in pair]))
            stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=3.0))
            rows, failures = runner.run_scene("fixture.json", args, None,
                                              lexical_grounder=selector, catalog=catalog)
        self.assertEqual(goals, [catalog["lamp_2"], catalog["tv_1"]])
        self.assertEqual(len(rows), 2)
        self.assertFalse(failures)
        self.assertTrue(all(row["target_success"] for row in rows))
        self.assertTrue(all(row["grounding"]["correct"] for row in rows))
        self.assertFalse(any(row["success"] for row in rows))  # route waypoint 与 catalog 目标不同

    def test_route_waypoint_success_does_not_hide_catalog_goal_miss(self):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        scene["routes"] = [{"name": "r0", "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                             "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0},
                             "sage": {"split": "train", "instructions": [
                                 {"instruction_type": "A", "start": "start", "end": "lamp_2",
                                  "text": "Go to the tall lamp."}]}}]
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=0, stop_mode="explicit")
        actions, oracle = _steps([(0, 0), (4, 0), (4, 0)])
        manifest = {"stopped_by_agent": True,
                    "validity": {"physics_verified": True, "camera_verified": True}}
        selector = importlib.import_module("sim.3dgs.grounder").LexicalGrounder()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            stack.enter_context(mock.patch.object(runner, "make_policy", return_value=SimpleNamespace(name="teacher")))
            stack.enter_context(mock.patch.object(collect, "run_episode", return_value=(manifest, None)))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[actions, oracle]))
            stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=3.0))
            rows, failures = runner.run_scene(
                "fixture.json", args, None, lexical_grounder=selector,
                catalog={"lamp_2": {"x": 8.0, "y": 0.0}})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["success"])
        self.assertFalse(rows[0]["target_success"])
        self.assertTrue(rows[0]["grounding"]["correct"])
        self.assertEqual(len(failures), 0)

    def test_accepted_grounding_survives_episode_failure(self):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        scene["routes"] = [{"name": "r0", "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                             "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0},
                             "sage": {"split": "train", "instructions": [
                                 {"instruction_type": "A", "start": "start", "end": "lamp_2",
                                  "text": "Go to the tall lamp."}]}}]
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=0, stop_mode="explicit")
        selector = importlib.import_module("sim.3dgs.grounder").LexicalGrounder()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            stack.enter_context(mock.patch.object(collect, "run_episode", side_effect=RuntimeError("render failed")))
            failures = runner.run_scene(
                "fixture.json", args, None, lexical_grounder=selector,
                catalog={"lamp_2": {"x": 8.0, "y": 0.0}})[1]
        self.assertTrue(failures[0]["grounding"]["accepted"])
        self.assertTrue(failures[0]["grounding"]["correct"])
        summary = runner._grounding_summary([], failures)
        self.assertEqual((summary["n_requested"], summary["n_grounded"], summary["n_accepted"],
                          summary["n_correct"], summary["coverage"]), (1, 1, 1, 1, 1.0))

    def test_rejected_instruction_stays_in_requested_denominator(self):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        scene["routes"] = [{"name": "r0", "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                             "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0},
                             "sage": {"split": "train", "instructions": [
                                 {"instruction_type": "A", "start": "start", "end": "plant_1",
                                  "text": "Go to the red plant."}]}}]
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=0, stop_mode="explicit")
        selector = importlib.import_module("sim.3dgs.grounder").LexicalGrounder()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode"))
            rows, failures = runner.run_scene(
                "fixture.json", args, None, lexical_grounder=selector,
                catalog={"lamp_2": {"x": 8.0, "y": 0.0}})
        self.assertEqual((len(rows), len(failures)), (0, 1))
        self.assertEqual(failures[0]["grounding"]["reason"], "below_min_score")
        self.assertEqual(vln.summarize(rows, failures)["n_requested"], 1)
        run.assert_not_called()

    def test_spl_reference_starts_before_first_action(self):
        _, _, _, calls = self.run_scene()
        self.assertEqual(calls[0].args[1][0]["pose_gt"]["x"], 0.0)

    def test_exception_error_and_invalid_physics_are_failures(self):
        for outcome in (RuntimeError("renderer broke"), ({"stopped_by_agent": True}, "copy failed"),
                        ({"stopped_by_agent": True, "validity": {"physics_verified": False}}, None)):
            with self.subTest(outcome=outcome):
                rows, failures, _, _ = self.run_scene([outcome, outcome])
                self.assertEqual((len(rows), len(failures)), (0, 2))
                self.assertEqual(failures[0]["scene"], "s")
                self.assertIn("split", failures[0])

    def test_empty_or_truncated_trace_is_a_failure(self):
        for records in ([[], []] * 2, [[{"pose_before": {}}], []] * 2):
            rows, failures, _, _ = self.run_scene(records=records)
            self.assertEqual((len(rows), len(failures)), (0, 2))
            self.assertIn("长度一致", failures[0]["error"])

    def test_failed_episodes_stay_in_requested_denominator_and_groups(self):
        rows, failures, _, _ = self.run_scene()
        failures = [dict(rows.pop(), error="renderer broke")]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "run_scene", return_value=(rows, failures)), \
                mock.patch.object(runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")):
            out = os.path.join(tmp, "out")
            self.assertEqual(runner.main(["--scene", "fixture", "--policy", "teacher", "--out", out]), 1)
            with open(os.path.join(out, "vln_eval.json")) as f:
                report = json.load(f)
        self.assertEqual(report["overall"]["sr"], 0.5)
        self.assertEqual(report["by_split"]["test_Instruction_Unseen"]["sr"], 0.0)
        self.assertEqual(report["metric_version"], "3dgs.vln.v3")
        self.assertEqual(report["stop_mode"], "explicit")
        self.assertIsNone(report["stop_streak"])

    def test_legacy_report_keeps_v2_and_streak_metadata(self):
        rows, _, _, _ = self.run_scene()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "run_scene", return_value=(rows, [])), \
                mock.patch.object(runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")):
            self.assertEqual(runner.main(["--scene", "fixture", "--policy", "teacher", "--out", tmp,
                                          "--stop-mode", "legacy-safe"]), 0)
            with open(os.path.join(tmp, "vln_eval.json")) as f:
                report = json.load(f)
        self.assertEqual((report["metric_version"], report["stop_mode"], report["stop_streak"]),
                         ("3dgs.vln.v2", "legacy-safe", 3))

    def test_empty_output_directory_is_allowed_but_empty_selection_fails(self):
        rows, _, _, _ = self.run_scene()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "run_scene", return_value=(rows, [])), \
                mock.patch.object(runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")):
            self.assertEqual(runner.main(["--scene", "fixture", "--policy", "teacher", "--out", tmp]), 0)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "run_scene", return_value=([], [])), \
                mock.patch.object(runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")):
            with self.assertRaises(SystemExit) as caught:
                runner.main(["--scene", "fixture", "--policy", "teacher", "--out", os.path.join(tmp, "out")])
            self.assertEqual(caught.exception.code, 2)

    def test_negative_limits_are_rejected(self):
        for flag in ("--limit", "--videos"):
            with self.assertRaises(SystemExit) as caught:
                runner.main(["--scene", "fixture", "--policy", "teacher", "--out", "unused", flag, "-1"])
            self.assertEqual(caught.exception.code, 2)


class TestBanner(unittest.TestCase):
    def test_wraps_and_keeps_at_most_four_lines(self):
        import cv2
        video = importlib.import_module("sim.3dgs.video")
        canvas = np.full((360, 640, 3), 200, np.uint8)
        video._banner(cv2, canvas, "word " * 80, 0, 0, 640)
        band = canvas[32:32 + 90]                       # 4 行 * 20 px + 10 px 边距
        self.assertEqual(band.max(), 255)               # 白字
        self.assertLess(np.median(band), 100)           # 黑底
        self.assertTrue((canvas[:32] == 200).all() and (canvas[32 + 90:] == 200).all())


if __name__ == "__main__":
    unittest.main()
