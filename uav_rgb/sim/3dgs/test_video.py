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
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
video = importlib.import_module("sim.3dgs.video")

WP = {"x": 10.0, "y": 0.0, "z": 2.0}


def _oracle(*xy):
    return [{"step": i, "pose_gt": {"x": x, "y": y, "z": 2.0, "yaw_deg": 0.0}} for i, (x, y) in enumerate(xy)]


class TestVideoPolicyName(unittest.TestCase):
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


class TestEpisodeResult(unittest.TestCase):
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
    def test_cli_keeps_instruction_and_stop_requirement(self):
        with tempfile.TemporaryDirectory() as ep:
            scene_path = os.path.join(ep, "scene.json")
            scene = {"dir": ep, "waypoint": WP, "reach_radius_m": 1.}
            man = {"vln": {"text": "go to chair", "goal_object": "chair"}, "waypoint": WP,
                   "stopped_by_agent": False, "scene": {}, "dynamics_params": {}, "seed": 0}
            for name, obj in (("scene.json", scene), ("episode_manifest.json", man)):
                with open(os.path.join(ep, name), "w") as f:
                    json.dump(obj, f)
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
                video.main()
            self.assertEqual(compose.call_args.kwargs["instruction"], '"go to chair"   goal: chair')
            self.assertEqual(compose.call_args.kwargs["result"], "NO STOP 0.0 m")


if __name__ == "__main__":
    unittest.main()
