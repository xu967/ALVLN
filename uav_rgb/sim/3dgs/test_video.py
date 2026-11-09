#!/usr/bin/env python3
"""video.py 标题/结尾字的离线单测（不渲染、不需要 cv2）。FIXLIST D7：
标题策略名取 executed_by（以前只要 manifest 有 teacher 字段就写 teacher，学生 eval 集也被标成 teacher）；
没有 goal-stop 的集（深度教师 / forward / 学生）按 eval_3dgs 同口径：oracle 轨迹任一点到 waypoint <= reach_radius 就是 SUCCESS。
运行：$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m unittest sim.3dgs.test_video
"""
import importlib
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
