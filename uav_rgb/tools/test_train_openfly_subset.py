import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_openfly_subset as adapter


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

    def test_only_past_actions_and_reset_mask_are_available(self):
        obs, prev, masks, targets, weights = self.batches[0]
        self.assertEqual(sorted(obs), adapter.INPUTS)
        self.assertEqual(prev.view(-1).tolist(), [0, 9, 2])
        self.assertEqual(masks.view(-1).tolist(), [0, 1, 1])
        self.assertEqual(targets.view(-1).tolist(), [9, 2, 0])
        self.assertEqual(weights.view(-1).tolist(), [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "白名单"):
            adapter.logits_for(self.trainer, ({**obs, "pos": 1}, prev, masks, targets, weights))

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
