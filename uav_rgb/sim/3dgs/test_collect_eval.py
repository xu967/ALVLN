#!/usr/bin/env python3
"""collect_3dgs / eval_3dgs 端到端小测（mock renderer，不需要 torch/学生权重）。"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
scene_mod = importlib.import_module("sim.3dgs.scene")


def _run(args):
    return subprocess.run([sys.executable, "-m"] + args, cwd=ROOT,
                          capture_output=True, text=True, timeout=180)


class TestCollect3dgs(unittest.TestCase):
    def test_forward_mock_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            out = os.path.join(tmp, "ep")
            r = _run(["sim.3dgs.collect_3dgs", "--scene",
                      os.path.join(scene_dir, "scene.json"), "--out", out,
                      "--steps", "3", "--policy", "forward", "--mock-renderer",
                      "--save-depth"])
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            with open(os.path.join(out, "ep_00000", "episode_manifest.json")) as f:
                m = json.load(f)
            self.assertEqual(m["renderer"]["backend"], "mock")
            self.assertEqual(m["num_steps"], 3)
            self.assertEqual(m["dynamics_params"]["z_target_m"], 2.0)
            self.assertEqual(m["dynamics_profile"], m["dynamics_params"]["profile"])
            self.assertTrue(m["validity"]["episode_executed"])
            with open(os.path.join(out, "ep_00000", "actions.jsonl")) as f:
                rows = [json.loads(x) for x in f]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["action_executed"] == 1 for r in rows))
            self.assertTrue(all(r["pose_gt"]["z"] == 2.0 for r in rows))
            for i in range(3):
                png = os.path.join(out, "ep_00000", "rgb_%05d.png" % i)
                with open(png, "rb") as f:
                    self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
                self.assertTrue(os.path.isfile(
                    os.path.join(out, "ep_00000", "depth", "depth_%05d.npy" % i)))
            with open(os.path.join(out, "run.json")) as f:
                run = json.load(f)
            self.assertEqual(run["dynamics_params"]["z_target_m"], 2.0)


class TestEval3dgs(unittest.TestCase):
    def test_forward_mock_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=3, length=12.0)
            out = os.path.join(tmp, "eval")
            r = _run(["sim.3dgs.eval_3dgs", "--scene",
                      os.path.join(scene_dir, "scene.json"), "--out", out,
                      "--policy", "forward", "--mock-renderer",
                      "--episodes", "1", "--steps", "3"])
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            with open(os.path.join(out, "eval.json")) as f:
                e = json.load(f)
            self.assertEqual(e["episodes_done"], 1)
            self.assertIn("success_rate", e)
            self.assertIn("collision_rate", e)
            self.assertEqual(e["metric_version"], "3dgs.pilot.v1")
            self.assertEqual(e["renderer"]["backend"], "mock")
            self.assertEqual(len(e["per_episode"]), 1)


if __name__ == "__main__":
    unittest.main()
