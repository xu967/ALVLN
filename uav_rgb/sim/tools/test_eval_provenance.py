#!/usr/bin/env python3
"""不启动 Gazebo，走完整评测汇总与默认临时目录删除，验证元数据留存与新口径。"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import campus
import eval_closed_loop as ec


class EvalProvenanceTests(unittest.TestCase):
    @staticmethod
    def scene_meta(seed=16):
        family = campus.FAMILIES[(seed // 8) % len(campus.FAMILIES)]
        return {
            "schema": "navlab.scene.v1", "scene_id": seed, "seed": seed,
            "geometry_version": campus.VERSION, "task_version": campus.VERSION,
            "map_id": campus.map_id(seed), "family": family,
            "split_policy": "campus_map_v1", "appearance_version": "campus.v1",
            "pillars": [
                {"x": 1.0, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0],
                 "color": "red"},
                {"x": 2.0, "y": 1.6, "z": 1.5, "size": [0.4, 0.4, 3.0]},
            ],
            "waypoint": {"x": 2.0, "y": 1.0}, "stop_radius": 0.5, "goal_radius": 0.5,
            "instruction": "pass the red pillar on your left and stop",
            "target_color": "red", "relation": "left",
            "camera_profile": ec.CAMERA_PROFILE, "dynamics_profile": "x3_v1",
            "success_criterion": "visible_goal_reach_v1",
        }

    def run_eval(self, manifest, collect_rc=0, seeds="16", split=None,
                 expected_rc=None, protocol=None):
        workdirs = []
        seed_split = campus.split(16)
        split = seed_split if split is None else split

        def generate(cmd, **_kwargs):
            world = Path(cmd[cmd.index("--out") + 1])
            world.write_text("<sdf/>")
            world.with_suffix(".json").write_text(json.dumps(self.scene_meta()))
            return Mock(returncode=0)

        def collect(cmd, **_kwargs):
            episode = Path(cmd[cmd.index("--out") + 1])
            workdirs.append(episode.parent)
            # 到达 waypoint 并发出 SAFE：旧 success 与新 task_success 同时成立。
            (episode / "actions.jsonl").write_text(json.dumps({
                "action_executed": 0, "collision": False, "infer_ms": None}) + "\n")
            (episode / "oracle.jsonl").write_text(json.dumps({
                "pose_gt": {"x": 2.0, "y": 1.0}}) + "\n")
            if manifest is not None:
                (episode / "episode_manifest.json").write_text(json.dumps(manifest))
            proc = Mock(returncode=collect_rc)
            proc.communicate.return_value = ("", "collector diagnostic")
            return proc

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "eval"
            argv = ["eval_closed_loop.py", "--policy", "forward", "--task", "campus",
                    "--split", split, "--seeds", seeds, "--steps", "1",
                    "--out", str(out)]
            if protocol is not None:
                protocol_path = Path(directory) / "protocol.json"
                protocol_path.write_text(json.dumps(protocol))
                argv += ["--protocol", str(protocol_path)]
            stdout = io.StringIO()
            with patch.object(ec.sys, "argv", argv), \
                 patch.object(ec.subprocess, "run", side_effect=generate), \
                 patch.object(ec.subprocess, "Popen", side_effect=collect), \
                 patch.object(ec.cp, "_start_sim", return_value=Mock()), \
                 patch.object(ec.cp, "wait_ready"), patch.object(ec.cp, "_register"), \
                 patch.object(ec.cp, "stop_group"), patch.object(ec.cp, "cleanup_all"), \
                 patch.object(ec.cp, "XDG_RUNTIME_DIR", str(Path(directory) / "xdg")), \
                 patch.object(ec.time, "sleep"), patch.object(ec.signal, "signal"), \
                 contextlib.redirect_stdout(stdout):
                rc = ec.main()
            self.last_stdout = stdout.getvalue()
            want = expected_rc if expected_rc is not None else (0 if collect_rc == 0 else 1)
            self.assertEqual(rc, want)
            if split == seed_split:
                self.assertEqual(len(workdirs), 1)
                self.assertFalse(workdirs[0].exists(), "必须验证默认清理后的持久报告")
            else:
                self.assertEqual(workdirs, [], "split 跳过的 seed 不应启动采集")
            return json.loads((out / "eval.json").read_text())

    def test_default_cleanup_retains_performance_and_requested_backend(self):
        performance = {"steps": 1, "action_rtf": .97,
                       "mean_step_s": {"total": 1.07}}
        report = self.run_eval({"performance": performance,
                                "render_backend_requested": "software"})
        episode = report["per_episode"][0]
        self.assertEqual(episode["performance"], performance)
        self.assertEqual(episode["render_backend_requested"], "software")
        self.assertNotIn("renderer", episode)  # 请求路径不能冒充渲染器实测。

    def test_legacy_manifest_retains_explicit_unknowns(self):
        episode = self.run_eval({"num_steps": 1})["per_episode"][0]
        self.assertIsNone(episode["performance"])
        self.assertIsNone(episode["render_backend_requested"])

    def test_failed_collection_retains_available_metadata(self):
        performance = {"steps": 1, "action_rtf": None}
        report = self.run_eval({"performance": performance,
                                "render_backend_requested": "gpu"}, collect_rc=1)
        self.assertEqual(report["per_episode"], [])
        self.assertEqual(report["failed"][0]["performance"], performance)
        self.assertEqual(report["failed"][0]["render_backend_requested"], "gpu")
        # 失败算尝试，不算 skipped；requested/completion 分母为请求集数。
        self.assertEqual(report["attempted_seeds"], [16])
        self.assertEqual(report["scanned_seeds"], [16])
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["skipped_count"], 0)
        self.assertEqual(report["requested_success_rate"], 0.0)
        self.assertEqual(report["completion_rate"], 0.0)
        self.assertEqual(report["execution_rate"], 0.0)  # 尝试了 1 集但没完成

    def test_failed_collection_without_manifest_still_has_report(self):
        failure = self.run_eval(None, collect_rc=1)["failed"][0]
        self.assertIsNone(failure["performance"])
        self.assertIsNone(failure["render_backend_requested"])

    def test_skipped_is_not_failed_and_not_in_completion_denominator(self):
        other = next(s for s in ("train", "val", "test") if s != campus.split(16))
        report = self.run_eval({}, split=other, expected_rc=1)
        self.assertEqual(report["skipped_count"], 1)
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(report["scanned_seeds"], [16])
        self.assertEqual(report["attempted_seeds"], [])
        self.assertEqual(report["requested_seeds"], [])
        self.assertIsNone(report["completion_rate"])  # 0 次尝试，不是 0 成功率
        self.assertEqual(report["execution_rate"], 0.0)
        self.assertEqual(report["requested_success_rate"], 0.0)
        self.assertEqual(report["per_episode"], [])

    def test_new_metric_fields_and_requested_first_line(self):
        report = self.run_eval({})
        # 分母与 seed 账本分开写：requested_seeds 只含实际尝试过的 seed。
        self.assertEqual(report["requested_seeds"], [16])
        self.assertEqual(report["attempted_seeds"], [16])
        self.assertEqual(report["scanned_seeds"], [16])
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(report["skipped_count"], 0)
        self.assertEqual(report["requested_success_rate"], 1.0)
        self.assertEqual(report["completion_rate"], 1.0)
        self.assertEqual(report["execution_rate"], 1.0)
        self.assertEqual(report["unique_gids"], 1)
        self.assertEqual(report["metric_version"], ec.METRIC_VERSION)
        self.assertEqual(report["goal_radius_reach"], 1.5)
        self.assertEqual(report["goal_radius_stop"], 0.5)
        self.assertIsNotNone(report["wilson_ci"]["success"])
        self.assertIsNotNone(report["wilson_ci"]["requested"])
        self.assertIsNone(report["stable_termination_rate"])
        self.assertEqual(report["dynamics"], "x3_velocity")

        episode = report["per_episode"][0]
        for key in ("reach_ever", "final_in_radius", "stop_action_issued",
                    "collision_free", "stuck_free", "task_success",
                    "stable_termination", "metric_version",
                    "goal_radius_reach", "goal_radius_stop", "timed_out", "dynamics"):
            self.assertIn(key, episode)
        self.assertTrue(episode["reach_ever"])
        self.assertTrue(episode["final_in_radius"])
        self.assertTrue(episode["stop_action_issued"])
        self.assertTrue(episode["collision_free"])
        self.assertTrue(episode["stuck_free"])
        self.assertTrue(episode["task_success"])
        self.assertIsNone(episode["stable_termination"])
        self.assertEqual(episode["metric_version"], ec.METRIC_VERSION)
        self.assertEqual(episode["dynamics"], "x3_velocity")

        # 汇总第一行必须是 requested（带分母），done-set 在其后。
        lines = self.last_stdout.splitlines()
        requested_i = next(i for i, line in enumerate(lines)
                           if line.startswith("requested:"))
        done_i = next(i for i, line in enumerate(lines)
                      if line.startswith("done-set:"))
        self.assertLess(requested_i, done_i)
        self.assertIn("(1/1)", lines[requested_i])

    def test_protocol_match_is_not_ood(self):
        protocol = {"protocol_id": "campus.v2-test", "window_from": "dispatch",
                    "hold_s": 1.0, "steps": 1, "task": "campus",
                    "appearance_version": "campus.v1",
                    "dynamics": "x3_velocity",
                    "camera_profile": ec.CAMERA_PROFILE}
        report = self.run_eval({}, protocol=protocol)
        self.assertFalse(report["ood"])
        self.assertEqual(report["protocol_mismatches"], [])
        self.assertEqual(report["protocol"], protocol)

    def test_protocol_mismatch_marks_ood_by_default(self):
        report = self.run_eval({}, protocol={"steps": 2, "protocol_id": "old"})
        self.assertTrue(report["ood"])
        self.assertEqual(report["protocol_mismatches"],
                         [{"key": "steps", "protocol": 2, "actual": 1}])

    def test_strict_protocol_mismatch_fails_before_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path = Path(directory) / "protocol.json"
            protocol_path.write_text(json.dumps({"steps": 2}))
            argv = ["eval_closed_loop.py", "--policy", "forward", "--task", "campus",
                    "--split", "test", "--seeds", "16", "--steps", "1",
                    "--out", str(Path(directory) / "eval"),
                    "--protocol", str(protocol_path), "--strict-protocol"]
            with patch.object(ec.subprocess, "run") as run, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    ec.main()
            self.assertNotEqual(raised.exception.code, 0)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
