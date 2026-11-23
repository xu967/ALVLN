#!/usr/bin/env python3
"""vln.py（VLN 集与指标）、run_episode 的智能体喊停、录像指令横幅的小测（mock 渲染器，纯 CPU）。

  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_vln.py
"""
import importlib
import copy
import contextlib
import io
import json
import math
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

    def test_only_versioned_sage_multi_stage_routes_are_included(self):
        scene = copy.deepcopy(SCENE)
        scene["multi_stage_routes"] = [
            {"name": "long", "route_contract": "sage.multi_stage.v2", "split": "train",
             "instruction": "Go past the sofa, then stop at the lamp.",
             "ref_path": [[0, 0], [2, 0]], "stages": [
                 {"stage": 0, "waypoint": {"x": 1, "y": 0}, "goal_object": "sofa"},
                 {"stage": 1, "waypoint": {"x": 2, "y": 0}, "goal_object": "lamp"}]},
            {"name": "old", "route_contract": "old", "split": "train", "instruction": "ignore",
             "stages": [{"waypoint": {"x": 1, "y": 0}}, {"waypoint": {"x": 2, "y": 0}}]},
        ]
        eps = vln.episodes(scene)
        self.assertEqual(eps[-1]["route_contract"], "sage.multi_stage.v2")
        self.assertEqual(eps[-1]["stage_waypoints"][-1]["goal_object"], "lamp")
        self.assertEqual([ep["type"] for ep in vln.episodes(scene, types=["Multi-stage"])], ["Multi-stage"])
        self.assertEqual(len(vln.episodes(scene, splits=["test_Instruction_Unseen"])), 1)

    def test_one_per_route_rotates_by_type_name(self):
        # r0 按句式名排序是 [Area-based(1), Attribute-based(0)]，与指令在航线里的原顺序无关
        eps = vln.episodes(SCENE)
        self.assertEqual([e["key"] for e in vln.one_per_route(eps, seed=0)], ["s/r0/1", "s/r1/0"])
        self.assertEqual([e["key"] for e in vln.one_per_route(eps, seed=1)], ["s/r0/0", "s/r1/0"])

    def test_swapped_instruction_requires_matching_split_type_and_budget(self):
        spawn = {"x": 0.0, "y": 1.0, "z": 2.0, "yaw_deg": 90.0}
        scene = {"routes": [
            {"spawn": spawn, "max_steps": 20},
            {"spawn": {**spawn, "x": 4.0}, "max_steps": 20},
            {"spawn": dict(spawn), "max_steps": 20},
            {"spawn": dict(spawn), "max_steps": 20},
            {"spawn": dict(spawn), "max_steps": 30},
            {"spawn": dict(spawn), "max_steps": 20},
        ]}
        episodes = [
            {"route_index": 0, "goal_object": "lamp", "text": "Find the lamp.",
             "split": "train", "type": "A"},
            {"route_index": 1, "goal_object": "chair", "text": "Find the chair.",
             "split": "train", "type": "A"},
            {"route_index": 2, "goal_object": "chair", "text": "Go to the chair.",
             "split": "train", "type": "B"},
            {"route_index": 3, "goal_object": "chair", "text": "Fly to the chair.",
             "split": "test", "type": "A"},
            {"route_index": 4, "goal_object": "chair", "text": "Navigate to the chair.",
             "split": "train", "type": "A"},
            {"route_index": 5, "goal_object": "chair", "text": "Go to that chair.",
             "split": "train", "type": "A"},
        ]

        swapped = runner._swapped_instruction(episodes, 0, scene)

        self.assertIs(swapped, episodes[5])
        self.assertIsNone(runner._swapped_instruction(episodes[:2], 0, scene))

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
    def test_first_forward_doorway_passages_follow_crossing_time(self):
        doors = {"far": {"line_xy_m": [(1, 0), (1, 2)], "forward_normal_xy": (1, 0)},
                 "near": {"line_xy_m": [(0, 2), (0, 0)], "forward_normal_xy": (1, 0)}}
        actions = [{"step": 0, "pose_before": {"x": -1, "y": 1}}]
        trace = [{"step": 0, "t": .1, "pos": (1, 1, 1.2)},
                 {"step": 0, "t": .2, "pos": (-1, 1, 1.2)},
                 {"step": 0, "t": .3, "pos": (2, 1, 1.2)}]
        events = vln.doorway_passages(trace, actions, doors)
        self.assertEqual(events, [{"door": "near", "step": 0, "t": .05},
                                  {"door": "far", "step": 0, "t": .1}])

    def test_touching_jamb_reverse_and_collision_window_do_not_count(self):
        doors = {"door": {"line_xy_m": [(0, 1), (2, 1)], "forward_normal_xy": (0, 1)}}
        actions = [{"step": 0, "pose_before": {"x": 0, "y": 0}}, {"step": 1, "collision": True}]
        trace = [{"step": 0, "t": .1, "pos": (0, 2, 1.2)},
                 {"step": 0, "t": .2, "pos": (2, 0, 1.2)},
                 {"step": 1, "t": .3, "pos": (0, 2, 1.2)}]
        self.assertEqual(vln.doorway_passages(trace, actions, doors), [])

    def test_stop_inside_radius_is_success(self):
        a, o = _steps([(0, 0), (1, 0), (2, 0), (3.5, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True, shortest_m=3.0)
        self.assertTrue(m["success"] and m["oracle_success"])
        self.assertAlmostEqual(m["ne_m"], 0.5)
        self.assertAlmostEqual(m["path_m"], 3.5)
        self.assertAlmostEqual(m["spl"], 3.0 / 3.5)

    def test_multistage_requires_each_waypoint_in_order_and_explicit_final_stop(self):
        goals = [{"x": 2.0, "y": 1.0}, {"x": 4.0, "y": 0.0}]
        actions, oracle = _steps([(0, 0), (2, 0), (4, 0)])
        complete = vln.episode_metrics(actions, oracle, GOAL, 0.2, stopped=True,
                                       stage_goals=[{"x": 2.0, "y": 0.0}, goals[-1]])
        self.assertTrue(complete["success"])
        self.assertEqual((complete["stage_reached_count"], complete["stage_sr"], complete["stage_csr"]),
                         (2, 1.0, 1.0))
        missed = vln.episode_metrics(actions, oracle, GOAL, 0.2, stopped=True, stage_goals=goals)
        self.assertFalse(missed["success"])
        self.assertEqual(missed["stage_reached_count"], 0)
        self.assertEqual((missed["stage_hit_count"], missed["stage_csr"]), (1, 0.25))
        reverse, reverse_oracle = _steps([(0, 0), (2, 0), (4, 0)])
        unordered = vln.episode_metrics(reverse, reverse_oracle, GOAL, 0.2,
                                        stopped=True, stage_goals=[{"x": 4.0, "y": 0.0},
                                                                   {"x": 2.0, "y": 0.0}])
        self.assertEqual((unordered["stage_hit_count"], unordered["stage_reached_count"]), (2, 1))
        no_stop = vln.episode_metrics(actions, oracle, GOAL, 0.2, stopped=False,
                                      stage_goals=[{"x": 2.0, "y": 0.0}, goals[-1]])
        self.assertTrue(no_stop["oracle_success"])
        self.assertFalse(no_stop["success"])
        self.assertEqual(no_stop["stage_csr"], 1.0)
        self.assertFalse(no_stop["stage_task_success"])
        skipped = vln.episode_metrics(actions, oracle, GOAL, 0.2, stopped=True,
                                      stage_goals=[{"x": 3.0, "y": 0.0}, {"x": 2.0, "y": 0.0}])
        self.assertEqual((skipped["stage_hit_count"], skipped["stage_reached_count"], skipped["stage_csr"]),
                         (2, 1, 1.0))
        self.assertFalse(skipped["success"])

    def test_stage_summary_uses_local_names_and_requested_denominator(self):
        rows = [{"route_contract": "sage.multi_stage.v2", "stage_count": 2,
                 "stage_reached_count": 1, "stage_hit_count": 1, "stage_csr": 0.25,
                 "stage_task_success": False}]
        failures = [{"route_contract": "sage.multi_stage.v2", "stage_waypoints": [{}, {}]}]
        report = vln.summarize_stages(rows, failures)
        self.assertEqual((report["n_requested"], report["n_stages_requested"]), (2, 4))
        self.assertEqual(report["stage_sr"], 0.25)
        self.assertEqual(report["stage_csr"], 0.125)
        self.assertEqual(report["stage_task_sr"], 0.0)

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

    def test_ndtw_sdtw_and_bearing_bin(self):
        ref = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        a, o = _steps([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (4.0, 0.0)])
        m_ok = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True, shortest_m=4.0, ref_path=ref)
        self.assertGreater(m_ok["ndtw"], 0.90)
        self.assertAlmostEqual(m_ok["sdtw"], m_ok["ndtw"])
        m_nostop = vln.episode_metrics(a, o, GOAL, 1.0, stopped=False, shortest_m=4.0, ref_path=ref)
        self.assertAlmostEqual(m_nostop["ndtw"], m_ok["ndtw"])
        self.assertEqual(m_nostop["sdtw"], 0.0)
        self.assertEqual(vln.bearing_bin(0.0, (0, 0), (5, 0), 8), 0)
        self.assertEqual(vln.bearing_bin(0.0, (0, 0), (0, 5), 8), 2)
        self.assertEqual(vln.bearing_bin(0.0, (0, 0), (-5, 0), 8), 4)
        self.assertEqual(vln.bearing_bin(0.0, (0, 0), (0, -5), 8), 6)

    def test_subgoal_bearing_uses_runtime_lookahead_and_goal_radius(self):
        planner = SimpleNamespace(
            plan=mock.Mock(return_value=[(0.0, 0.0), (2.0, 0.0)]),
            aim_point=mock.Mock(return_value=(0.0, 2.0)),
        )
        self.assertEqual(vln.subgoal_bearing_bin(planner, 0.0, (0, 0), (5, 0), 0.5, 1.0), (2, 0))
        planner.plan.assert_called_once_with((0.0, 0.0), (5.0, 0.0))
        planner.aim_point.assert_called_once_with(
            [(0.0, 0.0), (2.0, 0.0)], 0.0, 0.0,
            lookahead=1.05, los_max=3.0, hard_los_max=1.0)

        planner.plan.reset_mock()
        planner.aim_point.reset_mock()
        self.assertEqual(vln.subgoal_bearing_bin(planner, 0.0, (4.5, 0), (5, 0), 0.5, 1.0), (0, 1))
        planner.plan.assert_not_called()
        planner.aim_point.assert_not_called()

    def test_subgoal_bearing_falls_back_to_direct_goal_when_astar_fails(self):
        planner = SimpleNamespace(plan=mock.Mock(return_value=None), aim_point=mock.Mock())
        self.assertEqual(vln.subgoal_bearing_bin(planner, 0.0, (0, 0), (0, 5), 0.2, 1.0), (2, 0))
        planner.aim_point.assert_not_called()

    def test_phrase_grounder_and_wrong_goal_video_badge(self):
        grounder_mod = importlib.import_module("sim.3dgs.grounder")
        pg = grounder_mod.RelationPhraseGrounder()
        catalog = {"track_light_1": {"x": 7.0, "y": -1.0}, "wall_art_2": {"x": 2.0, "y": 3.0}}
        res = pg.ground("Go to the track light located directly above the decorative wall art.", catalog)
        self.assertTrue(res.accepted)
        self.assertEqual(res.target, "track_light_1")
        # 若 grounder 选错目标且飞到了错误目标，视频徽标必须是 WRONG GOAL 而不能伪装成 SUCCESS
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        scene["routes"] = [{"name": "r0", "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                            "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0},
                            "sage": {"split": "train", "instructions": [
                                {"instruction_type": "A", "start": "start", "end": "tv_1",
                                 "text": "Go to the tall lamp."}]}}]
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=1, stop_mode="explicit")
        actions, oracle = _steps([(0, 0), (7, -1), (7, -1)])
        manifest = {"stopped_by_agent": True, "validity": {"physics_verified": True, "camera_verified": True}}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            stack.enter_context(mock.patch.object(runner, "make_policy", return_value=SimpleNamespace(name="teacher")))
            run_ep = stack.enter_context(mock.patch.object(collect, "run_episode", return_value=(manifest, None)))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[actions, oracle]))
            stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=3.0))
            vid = stack.enter_context(mock.patch.object(runner.video_mod, "compose_views_video",
                                                        return_value={"video": "unused/v.mp4", "frames": 10, "fps": 10.0}))
            rows, _ = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}),
                                       lexical_grounder=grounder_mod.LexicalGrounder(),
                                       catalog={"lamp_2": {"x": 7.0, "y": -1.0}})
        self.assertFalse(rows[0]["grounding"]["correct"])
        self.assertTrue(rows[0]["target_success"])
        self.assertFalse(rows[0]["success"])
        self.assertEqual(run_ep.call_args.args[0]["waypoint"], {"x": 4.0, "y": 0.0, "z": 2.0})
        self.assertTrue(vid.call_args.kwargs["result"].startswith("WRONG GOAL"))

    def test_braking_teacher_brakes_before_high_speed_turn_and_commits_hysteresis(self):
        planner = mock.Mock()
        planner.plan.return_value = [(0.0, 0.0), (0.0, 4.0)]
        planner.aim_point.return_value = (0.0, 4.0)
        # 当前速度 0.30 m/s 沿 +x，前方滑行段撞墙 -> 先输出 SAFE_NEUTRAL 刹停再左转
        planner.occ.sweep.side_effect = lambda a, b: (b[0] > 0.10, None)
        teacher = runner.planner_mod.OccAstarTeacher(planner, {"x": 0.0, "y": 4.0}, hold_s=0.5, clearance="braking")
        self.assertEqual(teacher.version, runner.planner_mod.TEACHER_VERSION_BRAKING)
        act_brake, _ = teacher(None, None, SimpleNamespace(x=0.0, y=0.0, z=1.2, yaw=0.0, vx=0.30, vy=0.0))
        self.assertEqual((act_brake, teacher.last_branch), (collect.SAFE_NEUTRAL, "brake_before_turn"))
        # 速度降到 0 后正常左转，且在误差 10°（介于 tol_exit=9.0° 与 tol=12° 之间）保持转向承诺不回跳
        planner.occ.sweep.side_effect = lambda a, b: (False, None)
        act_turn1, _ = teacher(None, None, SimpleNamespace(x=0.0, y=0.0, z=1.2, yaw=0.0, vx=0.0, vy=0.0))
        act_turn2, _ = teacher(None, None, SimpleNamespace(x=0.0, y=0.0, z=1.2, yaw=math.radians(80.0), vx=0.0, vy=0.0))
        act_fwd, _ = teacher(None, None, SimpleNamespace(x=0.0, y=0.0, z=1.2, yaw=math.radians(85.0), vx=0.0, vy=0.0))
        self.assertEqual((act_turn1, act_turn2, act_fwd),
                         (dyn3dgs.YAW_LEFT, dyn3dgs.YAW_LEFT, collect.FORWARD_SLOW))


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
        with self.assertRaisesRegex(ValueError, "bool stop_requested"):
            self.run_policy(lambda _rgb, _depth, _st: (collect.FORWARD_SLOW, None))

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
    def run_scene(self, outcome=None, records=None, videos=0):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        for i, route in enumerate(scene["routes"]):
            route.update(spawn={"x": 0.0, "y": 0.0, "z": 2.0},
                         waypoint={"x": 4.0 + i, "y": 0.0, "z": 2.0})
        scene_mod.apply_route(scene, 0)
        actions, oracle = _steps([(0, 0), (1, 0), (4, 0)])
        manifest = {"stopped_by_agent": True, "validity": {"physics_verified": True, "camera_verified": True}}
        args = SimpleNamespace(split=None, all_instructions=False, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=videos, stop_mode="explicit")
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
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}))
        return rows, failures, run.call_args_list, shortest.call_args_list

    def test_video_defaults_off(self):
        with mock.patch.object(runner.video_mod, "compose_views_video") as compose:
            rows, failures, _, _ = self.run_scene()
        compose.assert_not_called()
        self.assertEqual((len(rows), len(failures)), (2, 0))
        self.assertTrue(all("video" not in row and "video_error" not in row for row in rows))

    def test_video_limit_and_success_metadata(self):
        video = {"video": "unused/s/r0_1/views.mp4", "frames": 25, "fps": 10.0,
                 "backend": "ffmpeg libx264 crf 22", "quality": "normal", "keyframes": []}
        with mock.patch.object(runner.video_mod, "compose_views_video", return_value=video) as compose:
            rows, failures, _, _ = self.run_scene(videos=1)
        compose.assert_called_once()
        self.assertEqual((len(rows), len(failures)), (2, 0))
        self.assertEqual(rows[0]["video"], "s/r0_1/views.mp4")
        self.assertEqual(rows[0]["video_metadata"], {k: v for k, v in video.items() if k != "video"})
        self.assertNotIn("video", rows[1])

    def test_video_failure_preserves_metrics_continues_and_writes_report(self):
        video = {"video": "unused/s/r1_0/views.mp4", "frames": 25, "fps": 10.0}
        with mock.patch.object(runner.video_mod, "compose_views_video",
                               side_effect=[RuntimeError("encoder unavailable"), video]) as compose:
            rows, failures, calls, _ = self.run_scene(videos=2)
        self.assertEqual((len(rows), len(failures), len(calls), compose.call_count), (2, 0, 2, 2))
        self.assertTrue(all(row["success"] for row in rows))
        self.assertEqual(rows[0]["video_error"], "RuntimeError: encoder unavailable")
        self.assertNotIn("video", rows[0])
        self.assertEqual(rows[1]["video"], "s/r1_0/views.mp4")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, "run_scene", return_value=(rows, failures)), \
                mock.patch.object(runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")):
            self.assertEqual(runner.main(["--scene", "fixture", "--policy", "teacher", "--out", tmp,
                                          "--videos", "2"]), 0)
            with open(os.path.join(tmp, "episodes.jsonl")) as f:
                saved = [json.loads(line) for line in f]
            with open(os.path.join(tmp, "vln_eval.json")) as f:
                report = json.load(f)
        self.assertEqual(saved, rows)
        self.assertEqual((report["overall"]["n"], report["overall"]["n_failed"], report["overall"]["sr"]), (2, 0, 1.0))

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

    def test_swapped_instruction_reports_original_and_supplied_goal_scores(self):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=8)
        shared_spawn = {"x": 0.0, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}
        scene["routes"] = [
            {"name": "lamp_route", "spawn": shared_spawn, "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0},
             "max_steps": 8, "sage": {"split": "train", "instructions": [
                 {"instruction_type": "A", "start": "start", "end": "lamp_1", "text": "Go to lamp_1."}]}},
            {"name": "tv_route", "spawn": dict(shared_spawn), "waypoint": {"x": 8.0, "y": 0.0, "z": 2.0},
             "max_steps": 8, "sage": {"split": "train", "instructions": [
                 {"instruction_type": "A", "start": "start", "end": "tv_1", "text": "Go to tv_1."}]}},
        ]
        actions, oracle = _steps([(0, 0), (4, 0), (8, 0), (8, 0)])
        manifest = {"stopped_by_agent": True,
                    "validity": {"physics_verified": True, "camera_verified": True}}
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=2,
                               stop_mode="explicit", counterfactual="swap")
        episode_waypoints = []
        grounder_mod = importlib.import_module("sim.3dgs.grounder")
        selector = grounder_mod.LexicalGrounder()
        catalog = {"lamp_1": {"x": 4.0, "y": 0.0}, "tv_1": {"x": 8.0, "y": 0.0}}

        def run_episode(scene_arg, *_args, **_kwargs):
            episode_waypoints.append(scene_arg["waypoint"]["x"])
            return manifest, None

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            stack.enter_context(mock.patch.object(runner, "make_policy", return_value=SimpleNamespace(name="teacher")))
            stack.enter_context(mock.patch.object(collect, "run_episode", side_effect=run_episode))
            videos = stack.enter_context(mock.patch.object(
                runner.video_mod, "compose_views_video",
                return_value={"video": "unused.mp4", "frames": 10, "fps": 10.0}))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[actions, oracle] * 2))
            stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=4.0))
            rows, failures = runner.run_scene(
                "fixture.json", args, SimpleNamespace(camera_mount={}),
                lexical_grounder=selector, catalog=catalog)

        self.assertFalse(failures)
        self.assertEqual(len(rows), 2)
        first, second = rows
        self.assertEqual((first["success"], first["supplied_goal_success"]), (False, True))
        self.assertEqual((second["success"], second["supplied_goal_success"]), (True, False))
        self.assertEqual(first["supplied_goal_object"], "tv_1")
        self.assertEqual(first["supplied_goal_waypoint"]["x"], 8.0)
        self.assertEqual(second["supplied_goal_object"], "lamp_1")
        self.assertEqual(second["supplied_goal_waypoint"]["x"], 4.0)
        self.assertTrue(first["grounding"]["correct"])
        self.assertTrue(second["grounding"]["correct"])
        self.assertEqual(runner._grounding_summary(rows, failures)["n_correct"], 2)
        self.assertEqual(videos.call_args_list[0].kwargs["result"], "SUCCESS stop 0.00 m")
        self.assertEqual(videos.call_args_list[1].kwargs["result"], "WRONG STOP 4.0 m")
        self.assertEqual(episode_waypoints, [4.0, 8.0])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runner, "run_scene", return_value=(rows, failures)), \
                mock.patch.object(runner.CfSitlParams, "from_json",
                                  return_value=SimpleNamespace(profile=lambda: "test")):
            self.assertEqual(runner.main(["--scene", "fixture", "--policy", "teacher",
                                          "--counterfactual", "swap", "--out", tmp]), 0)
            with open(os.path.join(tmp, "vln_eval.json"), encoding="utf-8") as report_file:
                report = json.load(report_file)
        self.assertEqual(report["overall_goal_scope"], "original_episode_goal")
        self.assertEqual(report["supplied_goal_metric_scope"], "instruction_episode_waypoint")
        self.assertEqual(report["supplied_goal_metrics"]["n_requested"], 2)
        self.assertEqual(report["supplied_goal_metrics"]["sr"], 0.5)

    def test_swap_missing_partner_stays_in_supplied_goal_denominator(self):
        failure = {"key": "s/r0/0", "split": "test", "type": "A",
                   "error": "ValueError: no compatible swap partner"}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(runner, "run_scene", return_value=([], [failure])), \
                mock.patch.object(runner.CfSitlParams, "from_json",
                                  return_value=SimpleNamespace(profile=lambda: "test")):
            rc = runner.main(["--scene", "fixture", "--policy", "teacher",
                              "--counterfactual", "swap", "--out", tmp])
            with open(os.path.join(tmp, "vln_eval.json"), encoding="utf-8") as report_file:
                report = json.load(report_file)

        self.assertEqual(rc, 1)
        self.assertEqual(report["overall"]["n_requested"], 1)
        self.assertEqual(report["supplied_goal_metrics"]["n_requested"], 1)
        self.assertEqual(report["supplied_goal_metrics"]["sr"], 0.0)

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

    def test_multistage_runner_uses_route_geometry_but_vlm_bearing_policy(self):
        scene = {"name": "s", "ply_path": "unused.ply", "reach_radius_m": 0.5,
                 "max_steps": 8, "routes": [], "collision": {"source": "mesh"},
                 "multi_stage_routes": [{
                     "name": "long", "route_contract": "sage.multi_stage.v2", "split": "train",
                     "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                     "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0}, "max_steps": 20,
                     "instruction": "First go to the chair, then stop at the lamp.", "path_m": 4.0,
                     "ref_path": [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
                     "stages": [
                         {"stage": 0, "waypoint": {"x": 2.0, "y": 0.0}, "goal_object": "chair",
                          "instruction": "Go to the chair.", "path_m": 2.0},
                         {"stage": 1, "waypoint": {"x": 4.0, "y": 0.0}, "goal_object": "lamp",
                          "instruction": "Then go to the lamp.", "path_m": 2.0},
                     ]}], "step_s": 0.5}
        actions, oracle = _steps([(0, 0), (2, 0), (4, 0)])
        manifest = {"stopped_by_agent": True,
                    "validity": {"physics_verified": True, "camera_verified": True}}
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0, out="unused", hold=0.5,
                               policy="reactive", videos=0, stop_mode="explicit", grounder="vlm",
                               counterfactual="correct", blind_rgb=False, collect=False,
                               teacher_clearance="ray")

        class Tracker:
            trace = []

            def __call__(self, _rgb, _state):
                return 0, 0

        tracker = None
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode", return_value=(manifest, None)))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[actions, oracle]))
            shortest = stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=4.0))
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}),
                                               vlm_grounder=object())
        self.assertEqual((len(rows), len(failures)), (1, 0), failures)
        call = run.call_args
        self.assertEqual((call.args[0]["spawn"]["x"], call.args[0]["waypoint"]["x"], call.args[0]["max_steps"]),
                         (0.0, 4.0, 20))
        policy = call.args[5]
        self.assertIsInstance(policy, reactive.ReactiveBearingPolicy)
        self.assertIsInstance(policy.goal_fn, runner.VlmBeliefTracker)
        self.assertFalse(hasattr(policy, "goal_x"))
        self.assertTrue(rows[0]["success"])
        self.assertEqual((rows[0]["stage_reached_count"], rows[0]["stage_csr"],
                          rows[0]["stage_task_success"]), (2, 1.0, True))
        self.assertEqual(policy.goal_fn.stage_instructions,
                         [stage["instruction"] for stage in scene["multi_stage_routes"][0]["stages"]])
        self.assertEqual((rows[0]["shortest_m"], shortest.call_count), (4.0, 0))

    def test_manual_instruction_uses_vlm_without_waypoint_scoring(self):
        scene = {"name": "s", "ply_path": "unused.ply", "reach_radius_m": 0.5,
                 "max_steps": 8, "routes": [{"name": "stored", "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                                             "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0}}],
                 "collision": {"source": "mesh"}, "step_s": 0.5,
                 "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                 "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0}}
        actions, oracle = _steps([(0, 0), (1, 0), (2, 0)])
        manifest = {"stopped_by_agent": True, "terminal_reason": "agent_stop",
                    "validity": {"physics_verified": True, "camera_verified": True}}
        args = SimpleNamespace(split=None, all_instructions=False, seed=0, limit=0, out="unused", hold=0.5,
                               policy="reactive", videos=1, stop_mode="explicit", grounder="vlm",
                               counterfactual="correct", blind_rgb=False, collect=False,
                               teacher_clearance="ray", instruction="飞到走廊尽头，然后悬停",
                               spawn=(1.0, 2.0, 2.0), spawn_yaw_deg=270.0)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            apply_route = stack.enter_context(mock.patch.object(scene_mod, "apply_route"))
            occupancy = SimpleNamespace(raw=np.zeros((50, 50, 10), dtype=bool), origin=np.zeros(3),
                                        shape=(50, 50, 10), voxel=0.1, collision_model="mesh",
                                        collides=mock.Mock(return_value=False))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=occupancy))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=None))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode", return_value=(manifest, None)))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[actions, oracle]))
            shortest = stack.enter_context(mock.patch.object(runner, "route_shortest_path_m"))
            compose = stack.enter_context(mock.patch.object(
                runner.video_mod, "compose_views_video", return_value={"video": "unused.mp4"}))
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}),
                                               vlm_grounder=object())
        self.assertEqual((len(rows), failures), (1, []))
        apply_route.assert_not_called()
        shortest.assert_not_called()
        self.assertEqual(run.call_args.args[0]["spawn"],
                         {"x": 1.0, "y": 2.0, "z": 2.0, "yaw_deg": 270.0})
        self.assertEqual(run.call_args.args[0]["vln"]["supplied_text"], args.instruction)
        self.assertIsInstance(run.call_args.args[5], reactive.ReactiveBearingPolicy)
        self.assertEqual(run.call_args.args[5].goal_fn.instruction, args.instruction)
        self.assertIsNone(rows[0]["success"])
        self.assertIsNone(rows[0]["ne_m"])
        self.assertEqual(rows[0]["path_m"], 2.0)
        self.assertFalse(rows[0]["collision"])
        self.assertEqual(compose.call_args.kwargs["goal_visible"], False)

    def test_manual_spawn_collision_is_reported_before_simulator_starts(self):
        scene = {"name": "s", "ply_path": "unused.ply", "max_steps": 8,
                 "routes": [], "collision": {"source": "mesh"},
                 "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                 "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0}}
        args = SimpleNamespace(split=None, all_instructions=False, seed=0, limit=0, out="unused", hold=0.5,
                               policy="reactive", videos=0, stop_mode="explicit", grounder="vlm",
                               counterfactual="correct", blind_rgb=False, collect=False,
                               teacher_clearance="ray", instruction="飞到走廊尽头", spawn=(1.0, 2.0, 2.0))
        occ = SimpleNamespace(raw=np.zeros((50, 50, 10), dtype=bool), origin=np.zeros(3),
                              shape=(50, 50, 10), voxel=0.1, collision_model="v2",
                              spawn_ok=mock.Mock(return_value=False))
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=occ))
            renderer = stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode"))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}),
                                               vlm_grounder=object())
        self.assertEqual(rows, [])
        self.assertIn("碰撞代理范围内或与占据代理冲突", failures[0]["error"])
        renderer.assert_not_called()
        run.assert_not_called()

    def test_manual_spawn_outside_occupancy_bounds_is_rejected(self):
        scene = {"name": "s", "ply_path": "unused.ply", "max_steps": 8,
                 "routes": [], "collision": {"source": "mesh"},
                 "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                 "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0}}
        args = SimpleNamespace(split=None, all_instructions=False, seed=0, limit=0, out="unused", hold=0.5,
                               policy="reactive", videos=0, stop_mode="explicit", grounder="vlm",
                               counterfactual="correct", blind_rgb=False, collect=False,
                               teacher_clearance="ray", instruction="Fly to the sofa.", spawn=(8.0, 8.0, 2.0))
        occ = SimpleNamespace(raw=np.zeros((50, 50, 10), dtype=bool), origin=np.zeros(3),
                              shape=(50, 50, 10), voxel=0.1, collision_model="v2",
                              spawn_ok=mock.Mock(return_value=True))
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=occ))
            renderer = stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode"))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}),
                                               vlm_grounder=object())
        self.assertEqual(rows, [])
        self.assertIn("碰撞代理范围内", failures[0]["error"])
        renderer.assert_not_called()
        run.assert_not_called()

    def test_manual_spawn_rejects_insufficient_takeoff_drift_clearance(self):
        scene = {"name": "s", "ply_path": "unused.ply", "max_steps": 8,
                 "routes": [], "collision": {"source": "mesh"},
                 "spawn": {"x": 0.0, "y": 0.0, "z": 2.0},
                 "waypoint": {"x": 4.0, "y": 0.0, "z": 2.0}}
        args = SimpleNamespace(split=None, all_instructions=False, seed=0, limit=0,
                               out="unused", hold=0.5, policy="reactive", videos=0,
                               stop_mode="explicit", grounder="vlm", counterfactual="correct",
                               blind_rgb=False, collect=False, teacher_clearance="ray",
                               instruction="Fly through the doorway.", spawn=None)
        occupancy = SimpleNamespace(
            raw=np.zeros((50, 50, 10), dtype=bool), origin=np.zeros(3), shape=(50, 50, 10),
            voxel=0.1, collision_model="v1", collides=mock.Mock(return_value=False),
            clearance=mock.Mock(return_value=0.18), body_radius=0.102, spawn_drift_margin=0.105)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=occupancy))
            renderer = stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer"))
            run = stack.enter_context(mock.patch.object(collect, "run_episode"))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}),
                                               vlm_grounder=object())
        self.assertEqual(rows, [])
        self.assertIn("起飞净空不足", failures[0]["error"])
        renderer.assert_not_called()
        run.assert_not_called()

    def test_manual_student_rejects_checkpoint_without_vln_condition_before_vlm_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "student.bin")
            with open(ckpt, "wb") as f:
                f.write(b"fixture")
            student = SimpleNamespace(prev_dim=4, close=mock.Mock())
            fake_module = SimpleNamespace(StudentPolicy=mock.Mock(return_value=student))
            vlm = mock.Mock()
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(sys.modules, {"student": fake_module}))
                stack.enter_context(mock.patch.object(runner.grounder_mod, "VisualLanguageGrounder", vlm))
                stack.enter_context(mock.patch.object(
                    runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")))
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
                    runner.main(["--scene", "fixture", "--policy", "student", "--ckpt", ckpt,
                                 "--grounder", "vlm", "--instruction", "Go to the sofa.",
                                 "--out", os.path.join(tmp, "out")])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("prev_dim=15", stderr.getvalue())
            vlm.assert_not_called()
            student.close.assert_called_once_with()

    def test_multistage_runner_rejects_coordinate_teacher_policy(self):
        scene = {"name": "s", "ply_path": "unused.ply", "collision": {"source": "mesh"},
                 "routes": [], "multi_stage_routes": [{
            "name": "long", "route_contract": "sage.multi_stage.v2", "split": "train",
            "instruction": "go then stop", "stages": [
                {"stage": 0, "waypoint": {"x": 1, "y": 0}},
                {"stage": 1, "waypoint": {"x": 2, "y": 0}}]}]}
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="teacher", videos=0,
                               stop_mode="explicit", grounder="none")
        with mock.patch.object(scene_mod, "load_scene", return_value=scene), \
                mock.patch.object(collect, "_sha256", return_value="fixture-ply"):
            rows, failures = runner.run_scene("fixture.json", args, None)
        self.assertEqual(rows, [])
        self.assertEqual(failures[0]["route_contract"], "sage.multi_stage.v2")
        self.assertIn("需要 --grounder vlm", failures[0]["error"])

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

    def test_spawn_yaw_requires_finite_value_and_manual_instruction(self):
        with self.assertRaises(SystemExit) as caught:
            runner.main(["--scene", "fixture", "--policy", "reactive", "--grounder", "vlm",
                         "--instruction", "Find the sofa.", "--spawn-yaw-deg", "nan", "--out", "unused"])
        self.assertEqual(caught.exception.code, 2)
        with self.assertRaises(SystemExit) as caught:
            runner.main(["--scene", "fixture", "--policy", "teacher", "--spawn-yaw-deg", "90", "--out", "unused"])
        self.assertEqual(caught.exception.code, 2)

    def test_vlm_belief_tracker_one_miss_hysteresis_prevents_7_8_limit_cycle(self):
        grounder_mod = importlib.import_module("sim.3dgs.grounder")
        seq = [
            grounder_mod.VisualGrounding(8, 0, 0.90, "unseen_spawn", 4.0, "search"),    # step 0: 未见过目标 -> 立即 8 扫描
            grounder_mod.VisualGrounding(7, 0, 0.88, "doorway_right", 4.0, "subgoal"),  # step 2: 右侧见门洞子目标 -> 7 (不锁 final)
            grounder_mod.VisualGrounding(8, 0, 0.60, "transient_miss", 4.0, "search"),  # step 4: 首次漏检 -> 滞回置 0 推进，不回甩 8
            grounder_mod.VisualGrounding(8, 0, 0.85, "second_miss", 4.0, "search"),     # step 6: 连续二次漏检 -> 切回 8 搜索
            grounder_mod.VisualGrounding(0, 1, 0.94, "sofa", 4.0, "final"),             # step 8: 首次命中只记录证据
            grounder_mod.VisualGrounding(0, 1, 0.94, "sofa", 4.0, "final"),             # step 10: 同一目标新观测确认 STOP
        ]
        mock_vlm = mock.Mock()
        mock_vlm.ground_step.side_effect = seq
        tracker = runner.VlmBeliefTracker(mock_vlm, "Fly to the sofa.", update_every=2)
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        outputs = []
        for step in range(11):
            outputs.append(tracker(rgb, None))
            if step == 2:
                tracker.note_step(3)
        self.assertEqual(outputs, [
            (8, 0),  # step 0: 初始扫描
            (8, 0),  # step 1: 延续扫描
            (7, 0),  # step 2: 右转对准子目标门洞
            (0, 0),  # step 3: 奇数步回正防过冲
            (0, 0),  # step 4: 首次漏检保持 0 前进（消除 7<->8 极限环）
            (0, 0),  # step 5: 延续 0 前进
            (8, 0),  # step 6: 连续漏检 >= 2 进入搜索 8
            (8, 0),  # step 7: 延续搜索 8
            (0, 0),  # step 8: 首次最终目标近景只记录证据
            (0, 0),  # step 9: VLM 更新前沿用非停止动作
            (0, 1),  # step 10: 同一目标第二次近景证据确认 STOP
        ])
        self.assertEqual(tracker.trace[1]["goal_mode"], "subgoal")
        self.assertFalse(tracker.trace[1]["confirmed_target"])
        self.assertEqual(tracker.trace[-1]["goal_mode"], "final")
        self.assertTrue(tracker.trace[-1]["confirmed_target"])
        self.assertEqual([t["miss_streak"] for t in tracker.trace], [1, 0, 1, 2, 0, 0])

    def test_reactive_bearing_commits_executed_turn_to_tracker(self):
        goal_fn = mock.Mock(return_value=(1, 0))
        policy = reactive.ReactiveBearingPolicy(goal_fn)
        depth = np.full((10, 12), 4.0, dtype=np.float32)
        state = SimpleNamespace(x=0.0, y=0.0, yaw=0.0)
        first_action, _ = policy(None, depth, state)
        self.assertEqual(first_action, dyn3dgs.YAW_LEFT)
        policy.note_executed(dyn3dgs.YAW_LEFT)
        goal_fn.note_step.assert_called_once_with(dyn3dgs.YAW_LEFT)
        second_action, _ = policy(None, depth, state)
        self.assertEqual(second_action, dyn3dgs.FORWARD_SLOW)

    def test_reactive_bearing_records_depth_intervention(self):
        grounder_mod = importlib.import_module("sim.3dgs.grounder")
        goal_fn = mock.Mock(return_value=(0, 0))
        goal_fn.last_vg = grounder_mod.VisualGrounding(0, 0, 0.7, "doorway", 1.0, "subgoal")
        policy = reactive.ReactiveBearingPolicy(goal_fn)
        depth = np.full((10, 12), 4.0, dtype=np.float32)
        depth[4:6, 4:8] = 0.5
        depth[4:6, 0:4] = 1.0
        depth[4:6, 8:12] = 2.0
        action, _ = policy(None, depth, SimpleNamespace())
        self.assertEqual(action, dyn3dgs.YAW_RIGHT)
        self.assertEqual(policy.last_decision, {
            "action_proposed": dyn3dgs.FORWARD_SLOW,
            "action_intervention": "depth_avoidance",
            "stop_source": None,
            "grounding_mode": "subgoal",
            "grounding_reason": "doorway",
            "grounding_conf": 0.7,
        })

    def test_vlm_tracker_advances_named_stages_before_final_stop(self):
        grounder_mod = importlib.import_module("sim.3dgs.grounder")
        mock_vlm = mock.Mock()
        mock_vlm.ground_step.side_effect = [
            grounder_mod.VisualGrounding(0, 1, 0.9, "chair", 1.0, "final"),
            grounder_mod.VisualGrounding(0, 1, 0.9, "chair", 1.0, "final"),
            grounder_mod.VisualGrounding(0, 1, 0.9, "lamp", 1.0, "final"),
            grounder_mod.VisualGrounding(0, 1, 0.9, "lamp", 1.0, "final"),
        ]
        tracker = runner.VlmBeliefTracker(
            mock_vlm, "First go to the chair, then stop at the lamp.", update_every=1,
            stage_instructions=["Go to the chair.", "Then go to the lamp."])
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        outputs = []
        task_stop = []
        for _ in range(4):
            outputs.append(tracker(rgb, None))
            task_stop.append(tracker.confirmed_target)
        self.assertEqual(outputs, [(0, 0), (0, 0), (0, 0), (0, 1)])
        self.assertEqual(task_stop, [False, False, False, True])
        self.assertTrue(tracker.confirmed_target)
        self.assertEqual(tracker.stage_index, 1)
        self.assertEqual([row["stage_confirmed"] for row in tracker.trace], [False, True, False, True])
        self.assertEqual([row["confirmed_target"] for row in tracker.trace], [False, False, False, True])
        self.assertEqual([row["near_goal"] for row in tracker.trace], [0, 0, 0, 1])
        self.assertEqual([row["stage_index"] for row in tracker.trace], [0, 1, 1, 1])
        self.assertEqual([row["stage_confirmed_index"] for row in tracker.trace], [None, 0, None, 1])
        self.assertEqual([row["task_complete"] for row in tracker.trace], [False, False, False, True])
        prompts = [call.args[1] for call in mock_vlm.ground_step.call_args_list]
        self.assertEqual(prompts[0], prompts[1])
        self.assertIn("Current stage 1/2: Go to the chair.", prompts[0])
        self.assertIn("Current stage 2/2: Then go to the lamp.", prompts[2])

    def test_g0_mode_direct_uses_pure_euclidean_bearing_without_planner(self):
        scene = copy.deepcopy(SCENE)
        scene.update(ply_path="unused.ply", reach_radius_m=1.0, max_steps=4)
        scene["routes"] = [{"name": "r0", "spawn": {"x": 0.0, "y": 0.0, "z": 1.2},
                             "waypoint": {"x": 0.0, "y": 4.0, "z": 1.2},
                             "sage": {"split": "train", "instructions": [
                                 {"instruction_type": "A", "start": "s", "end": "lamp_2",
                                  "text": "Go to the tall lamp."}]}}]
        forbidden_planner = mock.Mock()
        forbidden_planner.plan.side_effect = AssertionError("direct 模式严禁调用 OccPlanner.plan")
        forbidden_planner.aim_point.side_effect = AssertionError("direct 模式严禁调用 OccPlanner.aim_point")

        captured_goal_fns = []

        def fake_adapter(_proc, goal_fn=None, rgb_transform=None):
            captured_goal_fns.append(goal_fn)
            return SimpleNamespace(name="student", stop_requested=True)

        actions, oracle = _steps([(0, 0), (0, 3.5)])
        manifest = {"stopped_by_agent": True, "validity": {"physics_verified": True, "camera_verified": True}}
        args = SimpleNamespace(split=None, all_instructions=True, seed=0, limit=0,
                               out="unused", hold=0.5, policy="student", ckpt="dummy.bin",
                               videos=0, stop_mode="explicit", g0_mode="direct", grounder="none")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(scene_mod, "load_scene", return_value=scene))
            stack.enter_context(mock.patch.object(scene_mod, "read_gaussian_ply", return_value=None))
            stack.enter_context(mock.patch.object(runner.collision_mod, "from_scene", return_value=None))
            stack.enter_context(mock.patch.object(runner.renderer_mod, "make_renderer", return_value=None))
            stack.enter_context(mock.patch.object(runner.planner_mod, "OccPlanner", return_value=forbidden_planner))
            stack.enter_context(mock.patch.object(collect, "_sha256", return_value="fixture-ply"))
            stack.enter_context(mock.patch.object(collect, "StudentAdapter", side_effect=fake_adapter))
            stack.enter_context(mock.patch.object(collect, "run_episode", return_value=(manifest, None)))
            stack.enter_context(mock.patch.object(runner, "_read_jsonl", side_effect=[actions, oracle]))
            stack.enter_context(mock.patch.object(runner, "route_shortest_path_m", return_value=4.0))
            rows, failures = runner.run_scene("fixture.json", args, SimpleNamespace(camera_mount={}), student_proc=object())
        self.assertEqual((len(rows), len(failures), len(captured_goal_fns)), (1, 0, 1))
        gfn = captured_goal_fns[0]
        # (0,0), yaw=0 朝 +x，终点 (0,4) 在正左 (+y) -> 纯直线欧氏方位桶 2，距终点 4.0m > 1.0m -> near_goal=0
        self.assertEqual(gfn(None, SimpleNamespace(x=0.0, y=0.0, yaw=0.0)), (2, 0))
        # (0,3.5), yaw=pi/2 朝 +y，距终点 0.5m <= 1.0m -> 方位桶 0，near_goal=1
        self.assertEqual(gfn(None, SimpleNamespace(x=0.0, y=3.5, yaw=math.pi / 2)), (0, 1))

        # 验证 CLI 解析 --g0-mode direct 并写入 vln_eval.json
        with tempfile.TemporaryDirectory() as tmp:
            dummy_ckpt = os.path.join(tmp, "ckpt.bin")
            with open(dummy_ckpt, "wb") as f:
                f.write(b"ckpt")
            out_dir = os.path.join(tmp, "eval_direct")
            fake_student_mod = SimpleNamespace(StudentPolicy=lambda *_a, **_kw: SimpleNamespace(close=lambda: None))
            with mock.patch.dict(sys.modules, {"student": fake_student_mod}), \
                    mock.patch.object(runner, "run_scene", return_value=(rows, [])), \
                    mock.patch.object(runner.CfSitlParams, "from_json", return_value=SimpleNamespace(profile=lambda: "test")):
                rc = runner.main(["--scene", "fixture", "--policy", "student", "--ckpt", dummy_ckpt,
                                  "--g0-mode", "direct", "--out", out_dir])
            self.assertEqual(rc, 0)
            with open(os.path.join(out_dir, "vln_eval.json"), encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["g0_mode"], "direct")


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
