#!/usr/bin/env python3
"""video.py 标题/结尾字的离线单测（不渲染、不需要 cv2）。FIXLIST D7：
标题策略名取 executed_by（以前只要 manifest 有 teacher 字段就写 teacher，学生 eval 集也被标成 teacher）；
没有 goal-stop 的集（深度教师 / forward / 学生）按 eval_3dgs 同口径：oracle 轨迹任一点到 waypoint <= reach_radius 就是 SUCCESS。
运行：$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m unittest sim.3dgs.test_video
"""
import importlib
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
video = importlib.import_module("sim.3dgs.video")

WP = {"x": 10.0, "y": 0.0, "z": 2.0}


def _oracle(*xy):
    return [{"step": i, "pose_gt": {"x": x, "y": y, "z": 2.0, "yaw_deg": 0.0}} for i, (x, y) in enumerate(xy)]


class TestVideoPolicyName(unittest.TestCase):
    def test_top_map_is_marked_as_oracle_diagnostic(self):
        self.assertIn("DIAGNOSTIC/ORACLE", video.TOP_LABEL)

    def test_executed_by_wins(self):
        self.assertEqual(video.video_policy({"teacher": "student", "executed_by": "student"}), "student")
        self.assertEqual(video.video_policy({"teacher": "plan_action", "executed_by": "teacher"}), "teacher")
        self.assertEqual(video.video_policy({"teacher": "plan_action", "executed_by": "student", "beta": 0.5}), "student")
        self.assertEqual(video.video_policy({"teacher": "forward", "executed_by": "forward"}), "forward")

    def test_old_manifest_without_executed_by(self):
        self.assertEqual(video.video_policy({"teacher": "plan_action"}), "teacher")
        self.assertEqual(video.video_policy({"teacher": "expert_action"}), "teacher")
        self.assertEqual(video.video_policy({"teacher": "student"}), "student")     # 以前这种集被写成 teacher
        self.assertEqual(video.video_policy({"teacher": "forward"}), "forward")
        self.assertEqual(video.video_policy({}), "student")
        self.assertEqual(video.video_policy({"teacher": "reactive_bearing", "executed_by": "teacher"}),
                         "reactive")
        self.assertEqual(video.video_policy({"teacher": "reactive_goal"}), "reactive")


class TestEpisodeResult(unittest.TestCase):
    def test_manual_instruction_result_stays_unscored(self):
        man = {"vln": {"manual_demo": True}, "waypoint": None, "stopped_by_agent": True}
        actions = [{"pose_before": {"x": 0.0, "y": 0.0}, "collision": False}]
        self.assertEqual(video.episode_result(man, _oracle((10.0, 0.0)), 1.0, actions=actions),
                         "VLM STOP | UNSCORED MANUAL DEMO")
        self.assertEqual(video.episode_result(man, _oracle((10.0, 0.0)), 1.0, collided=True, actions=actions),
                         "COLLISION | UNSCORED MANUAL DEMO")
        self.assertEqual(video.episode_result(dict(man, error="render failure")), "INVALID episode error")
        self.assertEqual(video.episode_result(dict(man, validity={"camera_verified": False})),
                         "INVALID observation")
        self.assertEqual(video.episode_result(man, [], 1.0), "INVALID trajectory")

    def test_vln_invalid_episode_cannot_show_success(self):
        man = {"vln": {"text": "go to chair"}, "waypoint": WP, "stopped_by_agent": True}
        actions = [{"pose_before": {"x": 0., "y": 0.}, "collision": False}]
        near = _oracle((10., 0.))
        for fields in ({"error": "render failed"}, {"validity": {"physics_verified": False}},
                       {"validity": {"camera_verified": False}}):
            with self.subTest(fields=fields):
                self.assertTrue(video.episode_result(dict(man, **fields), near, 1., actions=actions).startswith("INVALID"))
        self.assertEqual(video.episode_result(man, near, 1., actions=[]), "INVALID trajectory")
        # 历史录像没有validity字段时，仍可依靠完整轨迹和STOP判定。
        self.assertEqual(video.episode_result(man, near, 1., actions=actions), "SUCCESS stop 0.00 m")

    def test_vln_requires_stop_at_final_goal(self):
        man = {"vln": {"text": "go to chair"}, "waypoint": WP, "stopped_by_agent": False}
        actions = [{"pose_before": {"x": 0., "y": 0.}, "collision": False}]
        near = _oracle((10., 0.))
        self.assertEqual(video.episode_result(man, near, 1., actions=actions), "NO STOP 0.0 m")
        man["stopped_by_agent"] = True
        self.assertEqual(video.episode_result(man, near, 1., actions=actions), "SUCCESS stop 0.00 m")
        self.assertEqual(video.episode_result(man, _oracle((12., 0.)), 1., actions=actions), "WRONG STOP 2.0 m")
        actions[0]["collision"] = True
        self.assertEqual(video.episode_result(man, near, 1., actions=actions), "COLLISION")

    def test_goal_stop_and_collision_words_unchanged(self):
        self.assertEqual(video.episode_result({"stopped_on_goal": True}), "SUCCESS reached")
        self.assertEqual(video.episode_result({"stopped_on_collision": True}), "COLLISION")
        self.assertEqual(video.episode_result({"truncated_on_collision": True}), "COLLISION")
        self.assertEqual(video.episode_result({"num_steps": 5}, collided=True), "COLLISION")
        self.assertTrue(video.episode_result({"dynamics_anomaly": {"kind": "backward_under_forward"}}).startswith("ANOMALY"))
        self.assertEqual(video.episode_result({"num_steps": 7}), "NOT REACHED (7 steps)")

    def test_swap_uses_the_supplied_target_for_the_map_and_stop_score(self):
        supplied = {"x": 2., "y": 0.}
        task = {"counterfactual": "swap", "supplied_goal_object": "lamp",
                "supplied_goal_waypoint": supplied, "grounding": {"target": "lamp", "correct": False}}
        man = {"vln": task, "waypoint": WP, "stopped_by_agent": True}
        actions = [{"pose_before": {"x": 0., "y": 0.}, "collision": False}]
        self.assertEqual(video._video_waypoint(task, WP), supplied)
        self.assertEqual(video.episode_result(man, _oracle((2., 0.)), 1., actions=actions), "SUCCESS stop 0.00 m")
        self.assertEqual(video.episode_result(man, _oracle((10., 0.)), 1., actions=actions), "WRONG STOP 8.0 m")
        del task["supplied_goal_waypoint"]
        self.assertIsNone(video._video_waypoint(task, WP))
        self.assertEqual(video.episode_result(man, _oracle((10., 0.)), 1., actions=actions),
                         "UNSCORED | goal unavailable")

    def test_empty_instruction_hides_original_goal_even_when_agent_stops_there(self):
        actions = [{"pose_before": {"x": 0., "y": 0.}, "collision": False}]
        for task in ({"counterfactual": "zero"}, {"text": "go to chair", "supplied_text": ""}):
            with self.subTest(task=task):
                self.assertIsNone(video._video_waypoint(task, WP))
                self.assertEqual(video.instruction_banner(task), "[empty instruction]")
                man = {"vln": task, "waypoint": WP, "stopped_by_agent": True}
                self.assertEqual(video.episode_result(man, _oracle((10., 0.)), 1., actions=actions),
                                 "STOP | UNSCORED EMPTY INSTRUCTION")
                self.assertEqual(video.episode_result(man, _oracle((10., 0.)), 1.,
                                                      actions=[dict(actions[0], collision=True)]), "COLLISION")

    def test_swapped_multistage_replay_requires_ordered_partner_stages(self):
        partner = {"name": "other", "route_contract": "sage.multi_stage.v2", "instruction": "Go via the lamp.",
                   "stages": [{"waypoint": {"x": 0., "y": 4.}}, {"waypoint": WP}]}
        scene = {"name": "s", "multi_stage_routes": [partner]}
        task = {"counterfactual": "swap", "route_contract": "sage.multi_stage.v2",
                "supplied_goal_waypoint": WP, "supplied_instruction_key": "s/other/multi_stage"}
        man = {"vln": task, "waypoint": {"x": -5., "y": 0.}, "stopped_by_agent": True}
        actions = [{"pose_before": {"x": 0., "y": 0.}}, {"pose_before": {"x": 0., "y": 4.}}]
        for via, success in [((0., 4.), True), ((0., 0.), False)]:
            score = video.episode_result(man, _oracle(via, (10., 0.)), 1., actions=actions, scene=scene)
            self.assertEqual(score.startswith("SUCCESS"), success)
        self.assertEqual(video.episode_result(man, _oracle((0., 4.), (10., 0.)), 1., actions=actions),
                         "UNSCORED | supplied stages unavailable")

    def test_legacy_goal_stop_cannot_override_failure(self):
        man = {"stopped_on_goal": True}
        self.assertEqual(video.episode_result(man, collided=True), "COLLISION")
        self.assertEqual(video.episode_result(dict(man, stopped_on_collision=True)), "COLLISION")
        self.assertTrue(video.episode_result(dict(man, dynamics_anomaly={"kind": "backward_under_forward"})).startswith("ANOMALY"))

    def test_reach_by_oracle_min_distance_without_goal_stop(self):
        man = {"num_steps": 3, "waypoint": WP, "stopped_on_goal": False}
        near = _oracle((0.0, 0.0), (9.0, 0.5), (12.0, 0.0))        # 中途进过 1.5 m 圈（距 1.12 m）
        self.assertEqual(video.episode_result(man, near, 1.5), "SUCCESS reached (min 1.12 m)")
        far = _oracle((0.0, 0.0), (5.0, 0.0), (7.0, 2.0))
        self.assertEqual(video.episode_result(man, far, 1.5), "NOT REACHED (3 steps)")
        self.assertEqual(video.episode_result(man, near, 1.0), "NOT REACHED (3 steps)")        # 半径更小就没进圈
        # 撞了之后穿模飞到目标不算到达
        self.assertEqual(video.episode_result(man, near, 1.5, collided=True), "COLLISION")
        # 没有 oracle / 半径就退回老行为
        self.assertEqual(video.episode_result(man), "NOT REACHED (3 steps)")


class TestVideoTime(unittest.TestCase):
    def test_regular_trace_preserved(self):
        trace = [{"t": .1 * k} for k in range(1, 11)]
        self.assertEqual(video._video_trace(trace), trace)

    def test_partial_window_does_not_slow_playback(self):
        # SITL 的 .25 s 窗口各给 .1/.2/.25 三个状态，30 条应播放 2.5 s。
        trace = [{"t": .25 * k + dt} for k in range(10) for dt in (.1, .2, .25)]
        frames = video._video_trace(trace)
        self.assertEqual(len(frames), 25)
        self.assertIs(frames[-1], trace[-1])
        for i, row in enumerate(frames):
            self.assertLessEqual(abs(row["t"] - (i + 1) / video.VIEW_FPS), .05 + 1e-9)

    def test_subframe_window_and_terminal_state(self):
        trace = [{"t": .05 * k} for k in range(1, 12)]
        frames = video._video_trace(trace)
        self.assertEqual(len(frames), 6)
        self.assertIs(frames[-1], trace[-1])

    def test_invalid_times_rejected(self):
        for times in ([], [0.], [float("nan")], [.2, .1], [.1, .1]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                video._video_trace([{"t": t} for t in times])


class TestVlnReplay(unittest.TestCase):
    def test_composed_swap_video_shows_the_supplied_goal_on_map_and_hud(self):
        supplied = {"x": 2., "y": 0.}
        scene = {"spawn": {"x": 0., "y": 0.}, "waypoint": WP,
                 "vln": {"counterfactual": "swap", "text": "go to chair",
                         "supplied_text": "go to lamp", "supplied_goal_waypoint": supplied}}
        renderer = importlib.import_module("sim.3dgs.renderer").MockRenderer(scene)
        trace = {"step": 0, "t": .1, "pos": [2., 0., 2.], "vel": [0., 0., 0.], "quat": [1, 0, 0, 0]}
        with (tempfile.TemporaryDirectory() as ep,
              mock.patch.object(video, "_TopMap", wraps=video._TopMap) as top_map,
              mock.patch.object(video.vl, "panel", wraps=video.vl.panel) as panel,
              mock.patch.object(video, "_banner_mask", wraps=video._banner_mask) as banner):
            with open(os.path.join(ep, "trace.jsonl"), "w") as f:
                f.write(json.dumps(trace) + "\n")
            video.compose_views_video(ep, scene, renderer, None, SimpleNamespace(raw=np.empty(0)),
                                      out_path=os.path.join(ep, "views.webm"), result="SUCCESS stop 0.00 m",
                                      quality="low")
            self.assertEqual(top_map.call_args.args[0]["waypoint"], supplied)
            self.assertIn("GT goal 0.00m", panel.call_args.args[4][2])
            self.assertEqual(panel.call_args.args[-1], "SUCCESS stop 0.00 m")
            self.assertEqual(banner.call_args.args[0], '"go to lamp"')
            self.assertEqual(scene["waypoint"], WP)

    def test_instruction_banner_uses_supplied_text_including_empty(self):
        self.assertEqual(video.instruction_banner({"text": "original", "supplied_text": "swap"}), '"swap"')
        self.assertEqual(video.instruction_banner({"text": "original", "supplied_text": ""}), "[empty instruction]")
        self.assertEqual(video.instruction_banner({"text": "original"}), '"original"')
        self.assertEqual(video.instruction_banner({"text": "original", "counterfactual": "swap"}),
                         "[swapped instruction unavailable]")

    def test_decision_frame_rejects_path_escape_and_reads_saved_rgb(self):
        import cv2

        with tempfile.TemporaryDirectory() as ep:
            expected = np.full((12, 16, 3), (21, 86, 143), np.uint8)
            cv2.imwrite(os.path.join(ep, "rgb_00000.png"), expected)
            frame = video._decision_rgb(cv2, ep, {"frame": "rgb_00000.png"})
            self.assertTrue(np.array_equal(frame, expected))
            self.assertIsNone(video._decision_rgb(cv2, ep, {"frame": "../outside.png"}))
            self.assertIsNone(video._decision_rgb(cv2, ep, {"frame": "rgb_00000.png", "rgb_ok": False}))

    def test_cli_keeps_supplied_instruction_and_stop_requirement(self):
        with tempfile.TemporaryDirectory() as ep:
            scene_path = os.path.join(ep, "scene.json")
            scene = {"dir": ep, "waypoint": WP, "reach_radius_m": 1.}
            man = {"vln": {"text": "go to chair", "goal_object": "sofa"}, "waypoint": WP,
                   "stopped_by_agent": False, "scene": {}, "dynamics_params": {}, "seed": 0}
            with open(scene_path, "w") as f:
                json.dump(scene, f)
            for name, rows in (("oracle.jsonl", _oracle((10., 0.))),
                               ("actions.jsonl", [{"pose_before": {"x": 0., "y": 0.}, "collision": False}])):
                with open(os.path.join(ep, name), "w") as f:
                    f.writelines(json.dumps(row) + "\n" for row in rows)
            renderer = importlib.import_module("sim.3dgs.renderer")
            collision = importlib.import_module("sim.3dgs.collision")
            with (mock.patch.object(sys, "argv", ["video", "--scene", scene_path, "--ep", ep]),
                  mock.patch.object(video.scene_mod, "load_scene", return_value=scene),
                  mock.patch.object(renderer, "make_renderer"),
                  mock.patch.object(collision, "from_scene"),
                  mock.patch.object(video, "compose_views_video", return_value={}) as compose,
                  mock.patch("builtins.print")):
                cases = [({}, '"go to chair"', "NO STOP 0.0 m"),
                         ({"counterfactual": "swap", "supplied_text": "go to the window",
                           "supplied_goal_waypoint": {"x": 4., "y": 0.}}, '"go to the window"', "NO STOP 6.0 m"),
                         ({"supplied_text": ""}, "[empty instruction]", "NO STOP | UNSCORED EMPTY INSTRUCTION")]
                for fields, banner, result in cases:
                    with self.subTest(fields=fields):
                        with open(os.path.join(ep, "episode_manifest.json"), "w") as f:
                            json.dump(dict(man, vln=dict(man["vln"], **fields)), f)
                        video.main()
                        self.assertEqual(compose.call_args.kwargs["instruction"], banner)
                        self.assertEqual(compose.call_args.kwargs["result"], result)


if __name__ == "__main__":
    unittest.main()
