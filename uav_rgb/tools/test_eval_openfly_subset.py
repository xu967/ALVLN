import importlib.util
import copy
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

STUDENT = {"policy": "student", "training_mode": "rgb_language",
           "active_inputs": evaluator.INPUTS, "blind_baseline": None}


def blind_manifest():
    return {"episodes": [
        {"split": "train", "route_id": "train_a", "actions": [9, 2, 0]},
        {"split": "train", "route_id": "train_b", "actions": [1, 9, 9, 0]},
        {"split": "validation", "route_id": "validation", "actions": [1, 0]}]}


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
        self.assertEqual(evaluator.parser().parse_args(args).policy, "student")

    def test_training_mode_defaults_for_old_checkpoints_and_rejects_unknown(self):
        self.assertEqual(evaluator.checkpoint_training_mode({}), "rgb_language")
        for mode in ("rgb_language", "rgb_only"):
            self.assertEqual(evaluator.checkpoint_training_mode({"training_mode": mode}), mode)
        with self.assertRaises(ValueError):
            evaluator.checkpoint_training_mode({"training_mode": "unknown"})

    def test_nonfinite_budgets_fail_before_loading_data(self):
        for name in ("max_distance", "timeout"):
            for value in (math.inf, -math.inf, math.nan):
                cli = evaluator.parser().parse_args(["--policy", "blind_forward",
                                                     "--manifest", "unused", "--out", "unused"])
                setattr(cli, name, value)
                with self.subTest(name=name, value=value), \
                     mock.patch.object(evaluator, "load_manifest") as load, self.assertRaises(ValueError):
                    evaluator.run(cli)
                load.assert_not_called()

    def test_blind_cli_does_not_require_or_accept_checkpoint(self):
        args = ["--policy", "blind_forward", "--manifest", "m", "--out", "o"]
        cli = evaluator.parser().parse_args(args)
        self.assertIsNone(cli.checkpoint)
        self.assertIsNone(cli.airvln_root)
        for extra in (["--checkpoint", "unused"], ["--airvln-root", "unused"], ["--language", "swap"]):
            with self.subTest(extra=extra), mock.patch.object(evaluator, "load_manifest") as load, self.assertRaises(ValueError):
                evaluator.run(evaluator.parser().parse_args(args + extra))
            load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "student 需要"):
            evaluator.run(evaluator.parser().parse_args(["--manifest", "m", "--out", "o"]))

    def test_protocol_failures_exit_nonzero_but_valid_task_terminations_do_not(self):
        cases = [("pose_application_error", 1), ("stale_rgb_timestamp", 1),
                 ("collision_reported", 0), ("step_budget", 0),
                 ("distance_budget", 0), ("time_budget", 0), ("predicted_stop", 0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route = "env_airsim_18/astar_data/medium_short/fixture"
            source = root / "train.json"
            source.write_text(json.dumps([{"image_path": route, "pos": [[0, 0, 0], [3, 0, 0]], "yaw": [0, 0]}]))
            manifest = {"annotation_source": str(source), "annotation_sha256": "fixture", "episodes": [
                {"split": "validation", "route_id": route, "instruction": "go forward"}]}
            for reason, expected in cases:
                with self.subTest(reason=reason):
                    cli = evaluator.parser().parse_args(["--checkpoint", "unused", "--manifest", "unused",
                           "--airvln-root", "unused", "--out", str(root / reason)])
                    result = {"termination": reason, "error": None, "video_error": None}
                    with mock.patch.object(evaluator, "load_manifest", return_value=manifest), \
                         mock.patch.object(evaluator, "Policy", return_value=SimpleNamespace(
                             checkpoint_sha256="fixture", contract={}, details=STUDENT, input_whitelist=evaluator.INPUTS)), \
                         mock.patch.object(evaluator, "sha256", return_value="fixture"), \
                         mock.patch.object(evaluator, "Simulator"), mock.patch.object(evaluator, "Recorder"), \
                         mock.patch.object(evaluator, "rollout", return_value=result), mock.patch("builtins.print"):
                        self.assertEqual(evaluator.run(cli), expected)
                    summary = json.loads((cli.out / "summary.json").read_text())
                    self.assertEqual(summary["protocol_errors"], expected)
                    self.assertEqual(summary["errors"], 0)
                    self.assertEqual(summary["episodes"][0]["termination"], reason)


class BlindForwardTests(unittest.TestCase):
    def test_initialization_failures_keep_summary_and_video_failure_keeps_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = blind_manifest()
            route = "env_airsim_18/astar_data/medium_short/fixture"
            manifest["episodes"][-1].update(route_id=route, instruction="go forward")
            source = root / "train.json"
            source.write_text(json.dumps([{"image_path": route, "pos": [[0, 0, 0], [3, 0, 0]], "yaw": [0, 0]}]))
            manifest.update(annotation_source=str(source), annotation_sha256="fixture")
            for stage in ("simulator", "recorder", "cleanup"):
                cli = evaluator.parser().parse_args(["--policy", "blind_forward", "--manifest", "unused",
                                                    "--out", str(root / stage)])
                with self.subTest(stage=stage), \
                     mock.patch.object(evaluator, "load_manifest", return_value=manifest), \
                     mock.patch.object(evaluator, "sha256", return_value="fixture"), \
                     mock.patch.object(evaluator, "Simulator") as simulator, \
                     mock.patch.object(evaluator, "Recorder") as recorder, \
                     mock.patch.object(evaluator, "rollout", return_value={"termination": "fixed_stop",
                         "error": None, "video_error": None}) as rollout, mock.patch("builtins.print"):
                    if stage == "simulator":
                        simulator.side_effect = RuntimeError("RPC unavailable")
                    elif stage == "recorder":
                        recorder.side_effect = OSError("font unavailable")
                    else:
                        simulator.return_value.close.side_effect = RuntimeError("cleanup unavailable")
                    self.assertEqual(evaluator.run(cli), 1)
                    summary = json.loads((cli.out / "summary.json").read_text())
                    if stage == "simulator":
                        self.assertIn("RPC unavailable", summary["run_error"])
                        rollout.assert_not_called()
                    else:
                        rollout.assert_called_once()
                        self.assertEqual(summary["episodes"][0]["termination"], "fixed_stop")
                        simulator.return_value.close.assert_called_once()
                        if stage == "recorder":
                            self.assertIsNone(rollout.call_args.args[2])
                            self.assertIn("font unavailable", summary["episodes"][0]["video_error"])
                        else:
                            self.assertIn("cleanup unavailable", summary["cleanup_error"])

    def test_run_does_not_load_checkpoint_and_records_untrained_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = blind_manifest()
            route = "env_airsim_18/astar_data/medium_short/fixture"
            manifest["episodes"][-1].update(route_id=route, instruction="go forward")
            source = root / "train.json"
            source.write_text(json.dumps([{"image_path": route, "pos": [[0, 0, 0], [3, 0, 0]], "yaw": [0, 0]}]))
            manifest.update(annotation_source=str(source), annotation_sha256="fixture")
            cli = evaluator.parser().parse_args(["--policy", "blind_forward", "--manifest", "unused",
                                                "--out", str(root / "out"), "--no-record-video"])
            with mock.patch.object(evaluator, "load_manifest", return_value=manifest), \
                 mock.patch.object(evaluator, "Policy") as student, \
                 mock.patch.object(evaluator, "sha256", return_value="fixture"), \
                 mock.patch.object(evaluator, "Simulator"), mock.patch.object(evaluator, "Recorder"), \
                 mock.patch.object(evaluator, "rollout", return_value={"termination": "fixed_stop",
                     "error": None, "video_error": None}), mock.patch("builtins.print"):
                self.assertEqual(evaluator.run(cli), 0)
            student.assert_not_called()
            contract = json.loads((cli.out / "contract.json").read_text())
            summary = json.loads((cli.out / "summary.json").read_text())
            self.assertIsNone(contract["training_contract"])
            self.assertEqual(contract["input_whitelist"], [])
            for document in (contract, summary):
                self.assertEqual(document["policy"], "blind_forward")
                self.assertIsNone(document["checkpoint_sha256"])
                self.assertIsNone(document["training_mode"])
                self.assertEqual(document["blind_baseline"]["stop_after_forward_actions"], 3)

    def test_rules_use_only_train_routes_and_no_observation(self):
        manifest = blind_manifest()
        original = evaluator.BlindForward(manifest)
        changed = copy.deepcopy(manifest)
        changed["episodes"][-1].update(actions=[1] * 1000 + [0], instruction="stop now",
                                         goal=[1, 2, 3], pos=[[0, 0, 0], [1, 2, 3]])
        changed["episodes"].append({"split": "validation", "route_id": "extra", "actions": [8] * 999 + [0]})
        baseline = evaluator.BlindForward(changed)
        self.assertEqual(baseline.details, original.details)
        self.assertEqual((baseline.action, baseline.stop_after), (9, 3))
        self.assertIsNone(baseline.checkpoint_sha256)
        self.assertIsNone(baseline.contract)
        self.assertEqual(baseline.input_whitelist, [])
        for _ in range(2):
            baseline.reset(object())
            self.assertEqual([baseline.step(object()) for _ in range(5)],
                             [(9, None)] * 3 + [(0, None)] * 2)

    def test_forward_token_tie_is_deterministic_and_training_required(self):
        manifest = {"episodes": [{"split": "train", "route_id": "a", "actions": [9, 1, 0]}]}
        self.assertEqual(evaluator.BlindForward(manifest).action, 1)
        for episodes in ([], [{"split": "validation", "actions": [9, 0]}],
                         [{"split": "train", "route_id": "a", "actions": [2, 0]}]):
            with self.assertRaises(ValueError):
                evaluator.BlindForward({"episodes": episodes})


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
            details = STUDENT
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

    def test_blind_fixed_stop_does_not_use_goal_and_keeps_latest_feedback(self):
        self.policy = evaluator.BlindForward(blind_manifest())
        result = self.run_episode()
        self.assertEqual(result["termination"], "fixed_stop")
        self.assertEqual((result["executed_actions"], result["frames"]), (3, 4))
        self.assertEqual([row[1]["action"] for row in self.recorder.rows], [9, 9, 9, 0])
        self.assertIsNone(self.recorder.rows[-1][1]["action_log_probs"])
        self.assertEqual(result["policy"], "blind_forward")
        self.assertIsNone(result["training_mode"])
        self.assertFalse(result["stopped_within_20m_diagnostic"])
        self.recorder.rows.clear()
        close = evaluator.rollout(self.simulator, self.policy, self.recorder, io.StringIO(),
                                  [0, 0, 0, 0], [27, 0, 0], "turn left", self.cli)
        self.assertTrue(close["stopped_within_20m_diagnostic"])
        self.assertEqual([row[1]["action"] for row in self.recorder.rows], [9, 9, 9, 0])

    def test_collision_and_distance_budget_never_execute_forbidden_move(self):
        self.simulator.collided = True
        self.assertEqual(self.run_episode()["termination"], "collision_reported")
        self.assertEqual(self.policy.observed, [])
        self.simulator.collided = False
        self.cli.max_distance = 2
        self.assertEqual(self.run_episode()["termination"], "distance_budget")
        self.assertEqual(self.simulator.actions, [])

    def test_invalid_start_or_goal_never_reaches_simulator(self):
        for field in ("start", "goal"):
            for value in (math.nan, math.inf, -math.inf):
                start, goal = [0, 0, 0, 0], [3, 0, 0]
                (start if field == "start" else goal)[0] = value
                with self.subTest(field=field, value=value), mock.patch.object(self.simulator, "begin") as begin:
                    result = evaluator.rollout(self.simulator, self.policy, self.recorder, io.StringIO(),
                                               start, goal, "go forward", self.cli)
                self.assertEqual(result["termination"], "error")
                self.assertIsNone(result["final_goal_distance_m"])
                json.dumps(result, allow_nan=False)
                begin.assert_not_called()

    def test_expired_inference_or_recording_never_dispatches_an_action(self):
        for stage in ("step", "write"):
            self.setUp()
            clock, trace = [0.0], io.StringIO()
            target = self.policy if stage == "step" else self.recorder
            original = getattr(target, stage)
            def slow(*args):
                result = original(*args)
                clock[0] = self.cli.timeout
                return result
            with self.subTest(stage=stage), mock.patch.object(target, stage, side_effect=slow), \
                 mock.patch.object(evaluator.time, "monotonic", side_effect=lambda: clock[0]):
                result = evaluator.rollout(self.simulator, self.policy, self.recorder, trace,
                                           [0, 0, 0, 0], [3, 0, 0], "go forward", self.cli)
            self.assertEqual(result["termination"], "time_budget")
            self.assertEqual(result["executed_actions"], 0)
            self.assertEqual(self.simulator.actions, [])
            row = json.loads(trace.getvalue().splitlines()[-1])
            self.assertEqual(row["termination"], "time_budget")
            self.assertFalse(row["action_requested"])

    def test_first_invalid_timestamp_never_reaches_policy(self):
        observe = self.simulator.observe
        for stamp in (0, -1, math.nan, math.inf, True):
            with self.subTest(stamp=stamp):
                self.simulator.observe = lambda: (*observe()[:3], stamp)
                result = self.run_episode()
                self.assertEqual(result["termination"], "stale_rgb_timestamp")
                self.assertEqual(self.policy.observed, [])
                self.assertEqual(self.simulator.actions, [])

    def test_nonfinite_feedback_never_reaches_policy(self):
        observe = self.simulator.observe
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.simulator.observe = lambda: (lambda row: (row[0], [value, 0, 0, 0], *row[2:]))(observe())
                result = self.run_episode()
                self.assertEqual(result["termination"], "error")
                self.assertIn("反馈位姿", result["error"])
                self.assertIsNone(result["final_goal_distance_m"])
                self.assertEqual(self.policy.observed, [])
                self.assertEqual(self.simulator.actions, [])
                json.dumps(result, allow_nan=False)

    def test_stop_at_distance_limit_is_valid_but_late_stop_is_timeout(self):
        self.cli.max_distance = 3
        result = self.run_episode()
        self.assertEqual(result["termination"], "predicted_stop")
        self.assertEqual(result["executed_actions"], 1)
        self.setUp()
        self.policy.outputs = iter([0])
        clock, step = [0.0], self.policy.step
        def late(rgb):
            output = step(rgb)
            clock[0] = self.cli.timeout
            return output
        with mock.patch.object(self.policy, "step", side_effect=late), \
             mock.patch.object(evaluator.time, "monotonic", side_effect=lambda: clock[0]):
            result = self.run_episode()
        self.assertEqual(result["termination"], "time_budget")
        self.assertFalse(result["stopped_within_20m_diagnostic"])
        self.assertEqual(self.simulator.actions, [])

    def test_timely_stop_is_not_revoked_by_final_video_encoding(self):
        self.policy.outputs = iter([0])
        clock, write = [0.0], self.recorder.write
        def slow(rgb, row):
            write(rgb, row)
            clock[0] = self.cli.timeout
        with mock.patch.object(self.recorder, "write", side_effect=slow), \
             mock.patch.object(evaluator.time, "monotonic", side_effect=lambda: clock[0]):
            result = self.run_episode()
        self.assertEqual(result["termination"], "predicted_stop")
        self.assertTrue(result["stopped_within_20m_diagnostic"])
        self.assertEqual(self.simulator.actions, [])

    def test_collision_precedes_timeout_and_never_calls_policy(self):
        self.simulator.collided = True
        with mock.patch.object(evaluator.time, "monotonic", side_effect=[0.0, 120.0]):
            result = self.run_episode()
        self.assertEqual(result["termination"], "collision_reported")
        self.assertEqual(self.policy.observed, [])
        self.assertTrue(self.recorder.closed)

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
                           {**STUDENT, "frame": 0, "action": 0, "goal_distance_m": 1,
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
            row = {**STUDENT, "frame": 0, "action": 0, "goal_distance_m": 1, "language": "correct"}
            recorder.write(np.zeros((4, 6, 3), dtype=np.uint8), row)
            recorder.write(np.zeros((4, 6, 3), dtype=np.uint8), row)
            recorder.close()
            self.assertEqual(video.call_count, 1)
            self.assertIn("ffmpeg missing", recorder.video_error)
            self.assertEqual(recorder.count, 2)
            self.assertTrue((Path(tmp) / "preview_last.png").is_file())


@unittest.skipUnless(importlib.util.find_spec("torch"), "需要 torch")
class RecurrentTests(unittest.TestCase):
    def test_nonfinite_logits_or_hidden_are_rejected(self):
        import torch
        observation = {"instruction": None, "rgb_features": None}
        for target in ("logits", "hidden"):
            for value in (math.nan, math.inf, -math.inf):
                logits, hidden = torch.zeros(1, 10), torch.zeros(1, 1, 2)
                (logits if target == "logits" else hidden).view(-1)[0] = value
                policy = SimpleNamespace(net=lambda *args: (torch.zeros(1, 2), hidden),
                                         action_distribution=lambda features: SimpleNamespace(logits=logits))
                with self.subTest(target=target, value=value), self.assertRaisesRegex(ValueError, "非有限"):
                    evaluator.recurrent_step(policy, observation, None, None, None)

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
