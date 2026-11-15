import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_openfly_subset as evaluator


class ActionTests(unittest.TestCase):
    def test_native_actions_and_yaw_wrap(self):
        expected = [[0, 0, 0, 0], [3, 0, 0, 0], [0, 0, 0, math.pi / 6],
                    [0, 0, 0, -math.pi / 6], [0, 0, 3, 0], [0, 0, -3, 0],
                    [0, 3, 0, 0], [0, -3, 0, 0], [6, 0, 0, 0], [9, 0, 0, 0]]
        for action, target in enumerate(expected):
            with self.subTest(action=action):
                for value, wanted in zip(evaluator.next_pose([0, 0, 0, 0], action), target):
                    self.assertAlmostEqual(value, wanted)
        rotated = evaluator.next_pose([0, 0, 0, math.pi / 2], 9)
        self.assertAlmostEqual(rotated[0], 0)
        self.assertAlmostEqual(rotated[1], 9)
        self.assertAlmostEqual(evaluator.next_pose([0, 0, 0, math.pi], 2)[3], -5 * math.pi / 6)

    def test_invalid_actions_never_become_stop(self):
        for action in (-1, 10, True, 1.0):
            with self.assertRaises(ValueError):
                evaluator.next_pose([0, 0, 0, 0], action)
        with self.assertRaises(ValueError):
            evaluator.next_pose([math.nan, 0, 0, 0], 0)

    def test_video_defaults_on_and_can_be_disabled(self):
        args = ["--checkpoint", "c", "--manifest", "m", "--airvln-root", "a", "--out", "o"]
        self.assertTrue(evaluator.parser().parse_args(args).record_video)
        self.assertFalse(evaluator.parser().parse_args(args + ["--no-record-video"]).record_video)

    def test_protocol_failures_exit_nonzero_but_valid_task_terminations_do_not(self):
        cases = [("pose_application_error", 1), ("stale_rgb_timestamp", 1),
                 ("collision_reported", 0), ("step_budget", 0),
                 ("distance_budget", 0), ("time_budget", 0), ("predicted_stop", 0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route = "env_airsim_18/astar_data/medium_short/fixture"
            source = root / "train.json"
            source.write_text(json.dumps([{"image_path": route, "pos": [[0, 0, 0], [3, 0, 0]], "yaw": [0, 0]}]))
            manifest = {"annotation_source": str(source), "episodes": [
                {"split": "validation", "route_id": route, "instruction": "go forward"}]}
            for reason, expected in cases:
                with self.subTest(reason=reason):
                    cli = evaluator.parser().parse_args(["--checkpoint", "unused", "--manifest", "unused",
                           "--airvln-root", "unused", "--out", str(root / reason)])
                    result = {"termination": reason, "error": None, "video_error": None}
                    with mock.patch.object(evaluator, "load_manifest", return_value=manifest), \
                         mock.patch.object(evaluator, "Policy", return_value=SimpleNamespace(checkpoint_sha256="fixture", contract={})), \
                         mock.patch.object(evaluator, "Simulator"), mock.patch.object(evaluator, "Recorder"), \
                         mock.patch.object(evaluator, "rollout", return_value=result), mock.patch("builtins.print"):
                        self.assertEqual(evaluator.run(cli), expected)
                    summary = json.loads((cli.out / "summary.json").read_text())
                    self.assertEqual(summary["protocol_errors"], expected)
                    self.assertEqual(summary["errors"], 0)
                    self.assertEqual(summary["episodes"][0]["termination"], reason)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "需要 numpy")
class RolloutTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        class Simulator:
            def __init__(self):
                self.actions, self.observations = [], 0
                self.collided = False
            def begin(self, pose):
                self.pose = list(pose)
            def advance(self, pose):
                self.actions.append(list(pose))
                self.pose = list(pose)
            def observe(self):
                self.observations += 1
                return np.full((4, 6, 3), self.observations, dtype=np.uint8), self.pose, self.collided, self.observations
        class Policy:
            def __init__(self):
                self.outputs, self.observed = iter([1, 0]), []
            def reset(self, instruction):
                self.instruction = instruction
            def step(self, rgb):
                self.observed.append(int(rgb[0, 0, 0]))
                return next(self.outputs), [0.0] * 10
        class Recorder:
            def __init__(self):
                self.rows, self.closed = [], False
            def write(self, rgb, row):
                self.rows.append((int(rgb[0, 0, 0]), row.copy()))
            def close(self):
                self.closed = True
        self.simulator, self.policy, self.recorder = Simulator(), Policy(), Recorder()
        self.cli = SimpleNamespace(max_steps=10, max_distance=100, timeout=120, language="correct")

    def run_episode(self):
        return evaluator.rollout(self.simulator, self.policy, self.recorder, io.StringIO(),
                                 [0, 0, 0, 0], [3, 0, 0], "go to the building", self.cli)

    def test_stop_saves_latest_feedback_without_an_extra_action(self):
        result = self.run_episode()
        self.assertEqual(result["termination"], "predicted_stop")
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["executed_actions"], 1)
        self.assertEqual(self.policy.observed, [1, 2])
        self.assertEqual(self.recorder.rows[-1][1]["action"], 0)
        self.assertFalse(self.recorder.rows[-1][1]["action_requested"])
        self.assertTrue(self.recorder.closed)

    def test_step_budget_saves_post_action_final_frame(self):
        self.cli.max_steps = 1
        result = self.run_episode()
        self.assertEqual(result["termination"], "step_budget")
        self.assertEqual(self.policy.observed, [1])
        self.assertEqual([r[0] for r in self.recorder.rows], [1, 2])
        self.assertIsNone(self.recorder.rows[-1][1]["action"])

    def test_collision_and_distance_budget_never_execute_forbidden_move(self):
        self.simulator.collided = True
        self.assertEqual(self.run_episode()["termination"], "collision_reported")
        self.assertEqual(self.policy.observed, [])
        self.simulator.collided = False
        self.cli.max_distance = 2
        self.assertEqual(self.run_episode()["termination"], "distance_budget")
        self.assertEqual(self.simulator.actions, [])

    def test_yaw_and_stale_timestamp_stop_before_using_bad_feedback(self):
        observe = self.simulator.observe
        def wrong_yaw():
            rgb, pose, collision, stamp = observe()
            return rgb, [*pose[:3], pose[3] + math.pi / 2], collision, stamp
        self.simulator.observe = wrong_yaw
        self.assertEqual(self.run_episode()["termination"], "pose_application_error")
        self.assertEqual(self.policy.observed, [])
        def stale():
            rgb, pose, collision, stamp = observe()
            return rgb, pose, collision, 1
        self.simulator.observe = stale
        self.assertEqual(self.run_episode()["termination"], "stale_rgb_timestamp")
        self.assertEqual(len(self.policy.observed), 1)

    def test_initialization_and_video_failure_do_not_erase_results(self):
        self.simulator.begin = mock.Mock(side_effect=RuntimeError("reset failed"))
        result = self.run_episode()
        self.assertEqual(result["termination"], "error")
        self.assertIn("reset failed", result["error"])
        self.assertTrue(self.recorder.closed)
        self.simulator.begin = lambda pose: setattr(self.simulator, "pose", list(pose))
        self.recorder.write = mock.Mock(side_effect=RuntimeError("encoder failed"))
        self.recorder.close = mock.Mock(side_effect=RuntimeError("finalize failed"))
        result = self.run_episode()
        self.assertEqual(result["termination"], "predicted_stop")
        self.assertEqual(result["frames"], 2)
        self.assertIsNone(result["error"])
        self.assertIn("encoder failed", result["video_error"])
        self.assertIn("finalize failed", result["video_error"])

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "需要 Pillow")
    def test_video_disabled_still_saves_first_last_and_preview(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(evaluator, "open_video") as video:
            recorder = evaluator.Recorder(Path(tmp), "go left", "go left", False)
            recorder.write(np.zeros((4, 6, 3), dtype=np.uint8),
                           {"frame": 0, "action": 0, "goal_distance_m": 1,
                            "language": "correct", "termination": "predicted_stop"})
            recorder.close()
            video.assert_not_called()
            self.assertEqual({p.name for p in Path(tmp).iterdir()},
                             {"first_rgb.png", "last_rgb.png", "preview_first.png", "preview_last.png"})

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "需要 Pillow")
    def test_encoder_failure_preserves_pngs_and_disables_only_video(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(evaluator, "open_video",
                  side_effect=RuntimeError("ffmpeg missing")) as video:
            recorder = evaluator.Recorder(Path(tmp), "go left", "go left", True)
            row = {"frame": 0, "action": 0, "goal_distance_m": 1, "language": "correct"}
            recorder.write(np.zeros((4, 6, 3), dtype=np.uint8), row)
            recorder.write(np.zeros((4, 6, 3), dtype=np.uint8), row)
            recorder.close()
            self.assertEqual(video.call_count, 1)
            self.assertIn("ffmpeg missing", recorder.video_error)
            self.assertEqual(recorder.count, 2)
            self.assertTrue((Path(tmp) / "preview_last.png").is_file())


@unittest.skipUnless(importlib.util.find_spec("torch"), "需要 torch")
class RecurrentTests(unittest.TestCase):
    def test_white_list_hidden_carry_and_episode_reset(self):
        import numpy as np
        import torch
        class Cnn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
            def forward(self, x):
                return self.avgpool(x)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                cnn = Cnn()
                self.rgb_encoder = SimpleNamespace(cnn=cnn, layer_extract=cnn.avgpool)
                self.num_recurrent_layers = 1
                self.state_encoder = SimpleNamespace(hidden_size=2)
                self.calls = []
            def forward(self, obs, hidden, prev, mask):
                self.calls.append((hidden.clone(), prev.clone(), mask.clone(), sorted(obs)))
                hidden = hidden * mask.view(1, 1, 1) + 1
                return hidden[:, 0, :], hidden
        net = Net()
        def distribution(features):
            logits = torch.zeros(1, 10)
            logits[:, 1] = features[:, 0]
            return SimpleNamespace(logits=logits)
        agent = evaluator.Policy.__new__(evaluator.Policy)
        agent.torch, agent.device = torch, "cpu"
        agent.policy = SimpleNamespace(net=net, action_distribution=distribution)
        agent.tokenizer = SimpleNamespace(split_sentence=lambda s: s.split(),
                                         encode_sentence=lambda s: [2, 3, 0])
        agent.reset("go left")
        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        agent.step(rgb)
        agent.step(rgb)
        self.assertEqual(net.calls[0][0].sum().item(), 0)
        self.assertFalse(net.calls[0][2].item())
        self.assertEqual(net.calls[1][0].sum().item(), 2)
        self.assertTrue(net.calls[1][2].item())
        self.assertEqual(net.calls[1][1].item(), 1)
        self.assertEqual(net.calls[1][3], evaluator.INPUTS)
        agent.reset("go right")
        agent.step(rgb)
        self.assertEqual(net.calls[-1][0].sum().item(), 0)
        self.assertFalse(net.calls[-1][2].item())
        with self.assertRaisesRegex(ValueError, "白名单"):
            evaluator.recurrent_step(agent.policy, {"instruction": 1, "rgb_features": 1, "goal": 2}, None, None, None)


if __name__ == "__main__":
    unittest.main()
