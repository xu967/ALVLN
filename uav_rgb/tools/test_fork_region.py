import contextlib
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import train
from tools import fork_region as region


class TinyEncoder:
    def __init__(self, cfg, device):
        pass

    def text_vectors(self, texts):
        return {t: np.array([len(t), 1.0]) / np.hypot(len(t), 1) for t in texts}

    def image_vectors(self, images):
        angles = np.array([im.getpixel((0, 0))[0] / 7 for im in images])[:, None] + [0.0, 1.0, 2.0]
        return np.stack([np.cos(angles), np.sin(angles)], axis=-1), np.tile([1, 0], (len(images), 1))


class TestForkRegion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = dict(train.FORK_REGION, checkpoint=str(self.root / "clip"), labels=str(self.root / "labels"),
                        original_manifests="original")
        checkpoint = Path(self.cfg["checkpoint"])
        checkpoint.mkdir()
        for name in region.MODEL_FILES:
            (checkpoint / name).write_text(name)
        fingerprint = "".join(f"{name} {region.sha(checkpoint / name)}\n" for name in sorted(region.MODEL_FILES))
        self.cfg["model_files_digest"] = hashlib.sha256(fingerprint.encode()).hexdigest()
        self.labels = Path(self.cfg["labels"])
        self.labels.mkdir()
        self.rows = {"train": [], "development": []}
        rooms = self.cfg["train_rooms"] + self.cfg["development_rooms"] * 2
        for i, (room, bins) in enumerate(zip(rooms, [(0, 1), (7, 0), (1, 7), (1, 7), (0, 4)])):
            split = "train" if i < 3 else "development"
            png = self.labels / f"{i}.png"
            image = Image.new("RGB", (8, 6), (i + 1, 5, 9))
            image.save(png)
            arms = [dict(instruction=f"fly to the {'lamp' if j else 'doorway'} {room}", planning_status="astar",
                         local_bearing_bin=b, inside_actual_image=(j == 0 or i % 2 == 1)) for j, b in enumerate(bins)]
            self.rows[split].append(dict(split=split, scene=room, pair=f"route_{i}", spawn_group=f"spawn_{i}",
                                         step=i, arm=0, verified_shared_prefix=i >= 3, rgb=png.name,
                                         rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(), labels=arms))
        self.originals = {}
        archive = self.root / self.cfg["original_manifests"]
        archive.mkdir()
        for split, rows in self.rows.items():
            for pool, portion in ((split, rows[:-1]), (f"equal_{split}", rows[-1:])):
                original = []
                for number, row in enumerate(portion, 1):
                    row.update(source_pool=pool, source_line=number)
                    original.append(dict(row, instructions=[a["instruction"] for a in row["labels"]]))
                path = archive / f"{pool}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in original))
                self.originals[str(path)] = region.sha(path)
        corpus = self.root / "registration.json"
        corpus.write_text(json.dumps(dict(registered_inputs={Path(p).name: h for p, h in self.originals.items()},
                                          splits={s: self.cfg[f"{s}_rooms"] for s in self.rows},
                                          denominators={s + "_states": len(r) for s, r in self.rows.items()})))
        self.cfg["corpus_registration_sha256"] = region.sha(corpus)
        protocol = self.root / "region_registration.json"
        protocol.write_text("fixture protocol")
        self.cfg["protocol_sha256"] = region.sha(protocol)
        self.write_sources()
        self.cfg["label_sha256"] = {s: region.sha(self.labels / f"{s}.jsonl") for s in self.rows}

    def write_sources(self):
        hashes = {}
        for split, rows in self.rows.items():
            path = self.labels / f"{split}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            hashes[path.name] = region.sha(path)
        (self.labels / "summary.json").write_text(json.dumps(dict(schema="uavrgb.vln.fork_bearings.v1",
                                                                  output_sha256=hashes, sources_sha256=self.originals)))

    def test_manifest_overlap_and_pixel_checks(self):
        checked = region.check_inputs(self.cfg)["label_sha256"]
        rows = region.read_frames(self.cfg, checked)["development"]
        Image.new("RGB", (8, 6)).save(rows[0].rgb)
        with self.assertRaisesRegex(ValueError, "Decoded RGB"):
            region.load_rgb(rows[0])
        self.rows["development"][0]["rgb_sha256"] = self.rows["train"][0]["rgb_sha256"]
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "changed after preflight"):
            region.read_frames(self.cfg, checked)
        with self.assertRaisesRegex(ValueError, "frozen export"):
            region.check_inputs(self.cfg)
        self.cfg["label_sha256"]["development"] = region.sha(self.labels / "development.jsonl")
        with self.assertRaisesRegex(ValueError, "sample identity"):
            region.check_inputs(self.cfg)
        modified = {s: region.sha(self.labels / f"{s}.jsonl") for s in self.rows}
        with self.assertRaisesRegex(ValueError, "Duplicate RGB"):
            region.read_frames(self.cfg, modified)
        (self.labels / "development.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "Label manifest SHA256"):
            region.check_inputs(self.cfg)

    def test_partial_reregistered_export_and_changed_protocol_fail(self):
        self.rows["development"].pop()
        self.write_sources()
        self.cfg["label_sha256"]["development"] = region.sha(self.labels / "development.jsonl")
        with self.assertRaisesRegex(ValueError, "Incomplete registered coverage"):
            region.check_inputs(self.cfg)
        (self.root / "region_registration.json").write_text("different protocol")
        with self.assertRaisesRegex(ValueError, "protocol or original"):
            region.check_inputs(self.cfg)

    def test_controls_and_denominators(self):
        dev = region.read_frames(self.cfg, region.check_inputs(self.cfg)["label_sha256"])["development"]
        features = region.extract_features(dev, TinyEncoder(self.cfg, "cpu"), 2)
        blank, glob = features["region_blank"], features["global_text"]
        np.testing.assert_array_equal(blank[0], blank[1])
        self.assertFalse(np.array_equal(blank[0], blank[2]))
        np.testing.assert_array_equal(glob[:, 3:], np.ones((4, 3)))
        np.testing.assert_array_equal(glob[:, 0], glob[:, 2])
        score = region.score(dev, np.array([[1, 2], [0, 0]]))
        self.assertEqual(score["single"]["accuracy_all_requests"], 0.75)
        self.assertEqual(score["single"]["accuracy_eligible"], 1)
        self.assertEqual(score["single"]["macro_recall"], 1)
        self.assertEqual(score["pairs"]["all"]["states"], 2)
        self.assertEqual((score["pairs"]["shared"]["correct"], score["pairs"]["shared"]["states"]), (1, 2))
        for key in ("both_expressible", "different", "both_inside", "expressible_shared", "different_both_inside_shared"):
            self.assertEqual((score["pairs"][key]["correct"], score["pairs"][key]["states"]), (1, 1))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "linear fitting requires SciPy")
    def test_standardization_and_loss_ignore_unsupported_arms(self):
        x = np.array([[2, 1], [3, 1], [4, 1], [1e6, 0], [5, 1]], dtype=float)
        params = region.fit_classifier(x, np.array([0, 0, 1, -1, 2]), self.cfg)
        self.assertEqual(params["train_class_counts"], [2, 1, 1])
        self.assertEqual(params["mean"], [3.5, 1.0])
        self.assertEqual(params["scale"][1], 1.0)
        self.assertEqual(region.predict(json.loads(json.dumps(params)), x).shape, (5,))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "linear fitting requires SciPy")
    def test_entry_cpu_outputs_and_overwrite_guard(self):
        try:
            __import__("torch")
        except (ImportError, OSError) as exc:
            self.skipTest(f"PyTorch runtime unavailable: {exc}")
        env = {k: v for k, v in os.environ.items() if not k.startswith("UAVRGB_")}
        env.update(UAVRGB_OUT_ROOT=str(self.root), UAVRGB_NAME="unit", CUDA_VISIBLE_DEVICES="")
        runtime = dict(train._paths(), py_vlm=Path(sys.prefix) / "bin/python")
        with patch.dict(os.environ, env, clear=True), patch.object(train, "FORK_REGION", self.cfg), \
                patch.object(train, "_paths", return_value=runtime), \
                patch.object(region, "FrozenClip", TinyEncoder), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(train.main(["fork_region", "--check"]), 0)
            self.assertFalse((self.root / "unit").exists())
            self.assertEqual(train.main(["fork_region"]), 0)
            dst = self.root / "unit" / "fork_region"
            ready = json.loads((dst / "summary.json").read_text())
            self.assertEqual(ready["eligible_train_requests"], 6)
            self.assertEqual({k: region.sha(dst / k) for k in ready["output_sha256"]}, ready["output_sha256"])
            entries = [json.loads(line) for line in (dst / "development_predictions.jsonl").read_text().splitlines()]
            self.assertEqual([len(r["labels"]) for r in entries], [2, 2])
            self.assertIsNone(entries[-1]["labels"][1]["truth"])
            self.assertEqual(set(entries[0]["labels"][0]["predicted"]), set(region.MODELS) | {"always_center"})
            with self.assertRaisesRegex(SystemExit, "目标目录已完成"):
                train.main(["fork_region", "--check"])


if __name__ == "__main__":
    unittest.main()
