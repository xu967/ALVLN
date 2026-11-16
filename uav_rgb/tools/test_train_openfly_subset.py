import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_openfly_subset as adapter


class TrainingModeTests(unittest.TestCase):
    def test_mode_is_explicit_and_only_changes_instruction_switch(self):
        args = SimpleNamespace(ablate_instruction=False, ablate_depth=True, SEQ2SEQ_use_prev_action=False)
        adapter.configure_training_mode(args, "rgb_only")
        self.assertTrue(args.ablate_instruction)
        self.assertTrue(args.ablate_depth)
        self.assertFalse(args.SEQ2SEQ_use_prev_action)
        adapter.configure_training_mode(args, "rgb_language")
        self.assertFalse(args.ablate_instruction)
        with self.assertRaisesRegex(ValueError, "training_mode"):
            adapter.configure_training_mode(args, "zero")

    def test_policy_feature_contract_keeps_cma_spatial_information(self):
        self.assertEqual(adapter.rgb_feature_shape("seq2seq"), (2048, 1, 1))
        self.assertEqual(adapter.rgb_feature_shape("cma"), (2048, 4, 4))
        with self.assertRaisesRegex(ValueError, "policy_type"):
            adapter.rgb_feature_shape("attention")


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        originals, episodes = [], []
        for i in range(4):
            route = f"env_airsim_18/route_{i}"
            images = []
            for j in range(3):
                image = self.root / route / f"frame_{j}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"fixture {i} {j}".encode())
                images.append(str(image))
            row = {"image_path": route, "gpt_instruction": f"go to building {i}",
                   "action": [9, 2, 0], "index_list": [f"frame_{j}" for j in range(3)],
                   "pos": [[i, j, 1] for j in range(3)], "yaw": [0, 0, 0]}
            originals.append(row)
            episodes.append({"route_id": route, "split": "train" if i < 2 else "validation",
                             "instruction": row["gpt_instruction"], "actions": row["action"],
                             "index_list": row["index_list"], "images": images,
                             "image_sha256": [adapter.sha256(p) for p in images]})
        self.source = self.root / "train.json"
        self.source.write_text(json.dumps(originals))
        self.manifest = {"schema": adapter.SCHEMA, "annotation_source": str(self.source),
                         "annotation_sha256": adapter.sha256(self.source), "episodes": episodes}
        self.path = self.root / "manifest.json"

    def load(self, manifest=None):
        self.path.write_text(json.dumps(self.manifest if manifest is None else manifest))
        return adapter.load_manifest(self.path)

    def test_complete_source_matching_routes_load_without_changing_inputs(self):
        before = self.source.read_bytes()
        self.assertEqual(len(self.load()["episodes"]), 4)
        self.assertEqual(self.source.read_bytes(), before)

    def test_rejects_action_stop_language_and_frame_contract_errors(self):
        patches = [{"actions": [9, -1, 0]}, {"actions": [9, True, 0]},
                   {"actions": [9, 0, 0]}, {"actions": [9, 2, 1]},
                   {"actions": [9, 3, 0]}, {"instruction": ""},
                   {"instruction": "invented instruction"}, {"index_list": ["frame_0"]},
                   {"images": ["relative.png"] * 3}, {"image_sha256": ["0" * 64] * 3},
                   {"split": "test"}]
        for patch in patches:
            with self.subTest(patch=patch):
                manifest = copy.deepcopy(self.manifest)
                manifest["episodes"][0].update(patch)
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_rejects_duplicate_route_and_physical_route_leakage(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["episodes"][2] = copy.deepcopy(manifest["episodes"][0])
        manifest["episodes"][2]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "route_id"):
            self.load(manifest)
        rows = json.loads(self.source.read_text())
        rows[2]["pos"] = rows[0]["pos"]
        self.source.write_text(json.dumps(rows))
        self.manifest["annotation_sha256"] = adapter.sha256(self.source)
        with self.assertRaisesRegex(ValueError, "物理路线"):
            self.load()

    def test_rejects_annotation_changes_and_single_route_ablation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["annotation_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Annotation SHA256"):
            self.load(manifest)
        self.manifest["episodes"] = self.manifest["episodes"][:3]
        with self.assertRaisesRegex(ValueError, "至少两条"):
            self.load()

    def test_numeric_representation_cannot_hide_duplicate_geometry(self):
        originals = json.loads(self.source.read_text())
        for four_dimensional in (False, True):
            with self.subTest(four_dimensional=four_dimensional):
                rows = copy.deepcopy(originals)
                rows[2]["pos"] = [[float(v) for v in p] + ([0.0] if four_dimensional else [])
                                  for p in rows[0]["pos"]]
                rows[2]["yaw"] = [float(v) for v in rows[0]["yaw"]]
                self.source.write_text(json.dumps(rows))
                self.manifest["annotation_sha256"] = adapter.sha256(self.source)
                with self.assertRaisesRegex(ValueError, "物理路线"):
                    self.load()

    def test_source_geometry_must_be_finite_and_aligned(self):
        originals = json.loads(self.source.read_text())
        patches = [{"pos": [[0, 0, float("nan")]] * 3},
                   {"pos": [[0, 0, True]] * 3}, {"pos": [[0, 0]] * 3},
                   {"pos": [[0, 0, 1]]}, {"yaw": [0]},
                   {"yaw": [0, float("inf"), 0]}]
        for patch in patches:
            with self.subTest(patch=patch):
                rows = copy.deepcopy(originals)
                rows[0].update(patch)
                self.source.write_text(json.dumps(rows))
                self.manifest["annotation_sha256"] = adapter.sha256(self.source)
                with self.assertRaisesRegex(ValueError, "pos/yaw"):
                    self.load()

    def test_existing_run_is_never_overwritten(self):
        self.load()
        output = self.root / "previous_run"
        output.mkdir()
        sentinel = output / "result.json"
        sentinel.write_text("preserve me")
        with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
            adapter.run(SimpleNamespace(epochs=1, manifest=self.path, out=output))
        self.assertEqual(sentinel.read_text(), "preserve me")

    def test_wrong_route_images_rejected_even_with_matching_stems_and_hashes(self):
        self.manifest["episodes"][0]["images"] = self.manifest["episodes"][1]["images"]
        self.manifest["episodes"][0]["image_sha256"] = self.manifest["episodes"][1]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "完整路线"):
            self.load()


class CausalManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        originals, episodes = [], []
        for route_index in range(4):
            route = f"env_airsim_18/route_{route_index}"
            folder = self.root / route
            folder.mkdir(parents=True)
            raw, x, y, yaw = [], route_index * 100.0, 0.0, 0.0
            for i, token in enumerate([1, 1, 1, 2, 1, 1, 3, 1, 0]):
                kind, value = {1: ("go straight", 3), 2: ("turn left", 30),
                               3: ("turn right", 30), 0: ("stop", 0)}[token]
                raw.append({"action": {"imageid": i, "type": kind, "value": value,
                                       "pos": [x, y, 1.0], "yaw": yaw}})
                (folder / f"frame_{i}.png").write_bytes(f"image {route_index}:{i}".encode())
                if token == 1:
                    x, y = x + 3 * math.cos(yaw), y + 3 * math.sin(yaw)
                elif token in (2, 3):
                    yaw += math.radians(30 if token == 2 else -30)
            raw_path = folder / "pose.jsonl"
            raw_path.write_text("\n".join(json.dumps(row, indent=2) for row in raw)
                                + '\n{"aim_landmark": {}}\n')
            bounds = [(0, 2), (3, 3), (4, 5), (6, 6), (7, 7), (8, 8)]
            indices = [f"frame_{first}" for first, _ in bounds]
            source_indices = [f"frame_{last}" for _, last in bounds]
            images = [str(folder / f"{index}.png") for index in indices]
            instruction, actions = f"fly to building {route_index}", [9, 2, 8, 3, 1, 0]
            originals.append({"image_path": route, "gpt_instruction": instruction, "action": actions,
                              "index_list": source_indices,
                              "pos": [raw[last]["action"]["pos"] for _, last in bounds],
                              "yaw": [raw[last]["action"]["yaw"] for _, last in bounds]})
            episodes.append({"route_id": route, "split": "train" if route_index < 2 else "validation",
                             "instruction": instruction, "actions": actions,
                             "index_list": indices, "source_index_list": source_indices,
                             "images": images, "image_sha256": [adapter.sha256(p) for p in images],
                             "raw_pose_path": str(raw_path), "raw_pose_sha256": adapter.sha256(raw_path),
                             "groups": [{"first_imageid": first, "last_imageid": last,
                                         "start_pose": raw[first]["action"]["pos"] + [raw[first]["action"]["yaw"]]}
                                        for first, last in bounds]})
        self.source = self.root / "train.json"
        self.source.write_text(json.dumps(originals))
        self.manifest = {"schema": adapter.CAUSAL_SCHEMA, "supervision": "causal_group_start_macro_bc",
                         "annotation_source": str(self.source), "annotation_sha256": adapter.sha256(self.source),
                         "episodes": episodes}
        self.path = self.root / "manifest.json"

    def load(self, manifest=None):
        self.path.write_text(json.dumps(self.manifest if manifest is None else manifest))
        return adapter.load_manifest(self.path)

    def test_causal_groups_preserve_source_and_use_pre_action_images(self):
        before = self.source.read_bytes()
        loaded = self.load()
        self.assertEqual(loaded, self.manifest)
        self.assertEqual(adapter.manifest_action_schema(loaded), adapter.CAUSAL_ACTION_SCHEMA)
        self.assertEqual(adapter.manifest_action_schema({"schema": adapter.SCHEMA}), adapter.ACTION_SCHEMA)
        self.assertEqual(loaded["episodes"][0]["groups"][0]["start_pose"], [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(self.source.read_bytes(), before)

    def test_rejects_changed_source_contract_or_group_boundaries(self):
        ep = self.manifest["episodes"][0]
        bad_groups = copy.deepcopy(ep["groups"])
        bad_groups[0]["first_imageid"] = 1
        bad_pose = copy.deepcopy(ep["groups"])
        bad_pose[0]["start_pose"][0] = 6
        bad_nan = copy.deepcopy(ep["groups"])
        bad_nan[0]["start_pose"][0] = float("nan")
        patches = [{"source_index_list": ep["index_list"]}, {"instruction": "changed task"},
                   {"raw_pose_sha256": "0" * 64}, {"groups": ep["groups"][:-1]},
                   {"groups": bad_groups}, {"groups": bad_pose}, {"groups": bad_nan},
                   {"actions": [9, 0, 8, 3, 1, 0]}]
        for patch in patches:
            with self.subTest(patch=patch):
                manifest = copy.deepcopy(self.manifest)
                manifest["episodes"][0].update(patch)
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_group_end_image_cannot_be_relabelled_as_group_start(self):
        ep = self.manifest["episodes"][0]
        ep["index_list"][0] = ep["source_index_list"][0]
        image = Path(ep["images"][0]).with_name(ep["index_list"][0] + ".png")
        ep["images"][0], ep["image_sha256"][0] = str(image), adapter.sha256(image)
        with self.assertRaisesRegex(ValueError, "图像索引"):
            self.load()

    def test_raw_internal_action_is_checked_even_with_updated_digest(self):
        ep = self.manifest["episodes"][0]
        raw_path = Path(ep["raw_pose_path"])
        original_text = raw_path.read_text()
        for before, after in [('"imageid": 1', '"imageid": 99'),
                              ('"value": 3', '"value": 2'),
                              ('3.0,', '4.0,')]:
            with self.subTest(after=after):
                self.assertIn(before, original_text)
                raw_path.write_text(original_text.replace(before, after, 1))
                ep["raw_pose_sha256"] = adapter.sha256(raw_path)
                with self.assertRaises(ValueError):
                    self.load()


@unittest.skipUnless(importlib.util.find_spec("torch"), "需要训练环境中的 torch")
class TensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        class Language(torch.nn.Module):
            def forward(self, obs):
                return obs["instruction"].float().sum(1, keepdim=True)
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Module()
                self.net.instruction_encoder = Language()
                self.net.num_recurrent_layers = 1
                self.net.state_encoder = SimpleNamespace(hidden_size=2)
                self.inputs = []
            def build_distribution(self, obs, hidden, prev, masks):
                self.inputs.append(obs["instruction"].clone())
                language = self.net.instruction_encoder(obs)
                logits = torch.zeros(len(prev), 10)
                logits[:, 2:3] = language
                return SimpleNamespace(logits=logits)
        self.trainer = SimpleNamespace(policy=Policy(), device="cpu")
        self.batches = [adapter.episode_tensors({"actions": [9, 2, 0]},
                        torch.zeros(3, 2048, 1, 1), [i, 0, 0], "cpu") for i in (2, 5)]

    def test_no_teacher_actions_and_reset_mask_are_available(self):
        obs, prev, masks, targets, weights = self.batches[0]
        self.assertEqual(sorted(obs), adapter.INPUTS)
        self.assertEqual(prev.view(-1).tolist(), [0, 0, 0])
        self.assertEqual(masks.view(-1).tolist(), [0, 1, 1])
        self.assertEqual(targets.view(-1).tolist(), [9, 2, 0])
        self.assertEqual(weights.view(-1).tolist(), [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "白名单"):
            adapter.logits_for(self.trainer, ({**obs, "pos": 1}, prev, masks, targets, weights))

    def test_shared_factory_disables_cma_previous_action_embedding(self):
        torch = self.torch
        args = SimpleNamespace(policy_type="seq2seq", SEQ2SEQ_use_prev_action=True,
                               ablate_depth=False, PROGRESS_MONITOR_use=True)
        calls = []
        def make_trainer(load, observation_space, action_space):
            calls.append((load, observation_space, action_space, args.policy_type))
            net = torch.nn.Module()
            net.prev_action_embedding = torch.nn.Embedding(11, 32)
            return SimpleNamespace(policy=SimpleNamespace(net=net))
        modules = {"src.common.param": SimpleNamespace(args=args),
                   "Model.il_trainer": SimpleNamespace(VLNCETrainer=make_trainer)}
        with patch.dict(sys.modules, modules):
            trainer = adapter.build_trainer("rgb-only-space", "ten-actions", "cma")
            embedding = trainer.policy.net.prev_action_embedding
            first, changed = embedding(torch.tensor([0, 1, 9])), embedding(torch.tensor([10, 7, 2]))
            self.assertTrue(torch.equal(first, torch.zeros(3, 32)))
            self.assertTrue(torch.equal(first, changed))
            self.assertFalse(first.requires_grad)
            self.assertFalse(args.SEQ2SEQ_use_prev_action)
            self.assertTrue(args.ablate_depth)
            self.assertFalse(args.PROGRESS_MONITOR_use)
            default = adapter.build_trainer("rgb-only-space", "ten-actions")
            self.assertEqual(len(default.policy.net.prev_action_embedding._forward_hooks), 0)
        self.assertEqual([call[-1] for call in calls], ["cma", "seq2seq"])

    def test_state_hash_identifies_values_and_is_independent_of_dict_order(self):
        torch = self.torch
        first = {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([3.0])}
        reordered = {"bias": first["bias"].clone(), "weight": first["weight"].clone()}
        self.assertEqual(adapter.state_sha256(first), adapter.state_sha256(reordered))
        reordered["weight"][0, 0] += 1
        self.assertNotEqual(adapter.state_sha256(first), adapter.state_sha256(reordered))
        renamed = {"another_weight": first["weight"], "bias": first["bias"]}
        self.assertNotEqual(adapter.state_sha256(first), adapter.state_sha256(renamed))

    def test_zero_language_removes_hook_and_swap_preserves_sources(self):
        original = [batch[0]["instruction"].clone() for batch in self.batches]
        normal = adapter.evaluate(self.trainer, self.batches)
        zero = adapter.evaluate(self.trainer, self.batches, zero_language=True)
        self.assertNotEqual(normal["cross_entropy"], zero["cross_entropy"])
        self.assertEqual(len(self.trainer.policy.net.instruction_encoder._forward_hooks), 0)
        self.assertEqual(normal, adapter.evaluate(self.trainer, self.batches))
        adapter.evaluate(self.trainer, self.batches, permute=True)
        self.assertTrue(self.torch.equal(self.trainer.policy.inputs[-2], original[1]))
        self.assertTrue(self.torch.equal(self.trainer.policy.inputs[-1], original[0]))
        for before, batch in zip(original, self.batches):
            self.assertTrue(self.torch.equal(before, batch[0]["instruction"]))

    def test_zero_language_hook_removed_on_failure(self):
        obs, *rest = self.batches[0]
        with self.assertRaisesRegex(ValueError, "白名单"):
            adapter.evaluate(self.trainer, [({**obs, "future_image": 1}, *rest)], zero_language=True)
        self.assertEqual(len(self.trainer.policy.net.instruction_encoder._forward_hooks), 0)
        with self.assertRaisesRegex(ValueError, "至少需要两条"):
            adapter.evaluate(self.trainer, self.batches[:1], permute=True)


if __name__ == "__main__":
    unittest.main()
