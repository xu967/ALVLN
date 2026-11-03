#!/usr/bin/env python3
"""不启动 Gazebo，走完整评测汇总与默认临时目录删除，验证元数据留存。"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import eval_closed_loop as ec
import scene_gen as sg


class EvalProvenanceTests(unittest.TestCase):
    def run_eval(self, manifest, collect_rc=0):
        workdirs = []

        def generate(cmd, **_kwargs):
            world = Path(cmd[cmd.index("--out") + 1])
            sdf, meta = sg.build_scene(sg._read_template(sg.DEFAULT_TEMPLATE), 16, 16,
                                       task="campus")
            world.write_text(sdf)
            world.with_suffix(".json").write_text(json.dumps(meta))
            return Mock(returncode=0)

        def collect(cmd, **_kwargs):
            episode = Path(cmd[cmd.index("--out") + 1])
            workdirs.append(episode.parent)
            (episode / "actions.jsonl").write_text(json.dumps({
                "action_executed": 1, "collision": False, "infer_ms": None}) + "\n")
            (episode / "oracle.jsonl").write_text(json.dumps({
                "pose_gt": {"x": 0., "y": 0.}}) + "\n")
            if manifest is not None:
                (episode / "episode_manifest.json").write_text(json.dumps(manifest))
            proc = Mock(returncode=collect_rc)
            proc.communicate.return_value = ("", "collector diagnostic")
            return proc

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "eval"
            argv = ["eval_closed_loop.py", "--policy", "forward", "--task", "campus",
                    "--split", "test", "--seeds", "16", "--steps", "1", "--out", str(out)]
            with patch.object(ec.sys, "argv", argv), \
                 patch.object(ec.subprocess, "run", side_effect=generate), \
                 patch.object(ec.subprocess, "Popen", side_effect=collect), \
                 patch.object(ec.cp, "_start_sim", return_value=Mock()), \
                 patch.object(ec.cp, "wait_ready"), patch.object(ec.cp, "_register"), \
                 patch.object(ec.cp, "stop_group"), patch.object(ec.cp, "cleanup_all"), \
                 patch.object(ec.cp, "XDG_RUNTIME_DIR", str(Path(directory) / "xdg")), \
                 patch.object(ec.time, "sleep"), patch.object(ec.signal, "signal"), \
                 contextlib.redirect_stdout(io.StringIO()):
                rc = ec.main()
            self.assertEqual(rc, 0 if collect_rc == 0 else 1)
            self.assertEqual(len(workdirs), 1)
            self.assertFalse(workdirs[0].exists(), "必须验证默认清理后的持久报告")
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

    def test_failed_collection_without_manifest_still_has_report(self):
        failure = self.run_eval(None, collect_rc=1)["failed"][0]
        self.assertIsNone(failure["performance"])
        self.assertIsNone(failure["render_backend_requested"])


if __name__ == "__main__":
    unittest.main()
