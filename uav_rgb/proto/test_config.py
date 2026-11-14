#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
try:
    import torch
    from base_model import BaseCnn
except ImportError:
    torch = None


class TestConfig(unittest.TestCase):
    def test_default_and_cli_override(self):
        cfg = config.resolve({"train": {"epochs": 2, "seed": 7}}, cli={"batch": 4})
        self.assertEqual(cfg["model"]["profile"], "nano_proto_v6")
        self.assertEqual((cfg["train"]["epochs"], cfg["train"]["batch"], cfg["train"]["seed"]), (2, 4, 7))
        self.assertEqual(config.resolve({"train": {"epochs": 2}}, cli={"epochs": 30})["train"]["epochs"], 30)

    def test_yaml_subset_and_hash_are_stable(self):
        text = "model:\n  profile: nano_proto_v6\ntrain:\n  epochs: 1\n  seed: 3\n"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(text)
            path = f.name
        try:
            cfg = config.resolve(config.load(path))
            self.assertEqual(cfg["train"]["epochs"], 1)
            self.assertEqual(config.digest(cfg), config.digest(json.loads(config.canonical_json(cfg))))
        finally:
            os.unlink(path)

    def test_rejects_unknown_profile_and_fields(self):
        with self.assertRaises(config.ConfigError):
            config.resolve({"model": {"profile": "made_up"}})
        with self.assertRaises(config.ConfigError):
            config.resolve({"train": {"mystery": 1}})
        with self.assertRaises(config.ConfigError):
            config.resolve({"model": []})

    def test_resolve_does_not_mutate_defaults_and_rejects_unused_inputs(self):
        first = config.resolve()
        first["model"]["backend"] = "changed"
        self.assertEqual(config.resolve()["model"]["backend"], "cpp")
        with self.assertRaises(config.ConfigError):
            config.resolve({"model": {"profile": "base_torch_v1", "language": {"enabled": True}}})
        with self.assertRaises(config.ConfigError):
            config.resolve({"train": {"lr": 0.0}})

    def test_base_is_registered_but_torch_only(self):
        cfg = config.resolve({"model": {"profile": "base_torch_v1"}})
        self.assertEqual(cfg["model"]["backend"], "torch")
        self.assertFalse(config.PROFILES["base_torch_v1"]["export"])

    @unittest.skipIf(torch is None, "没有 torch")
    def test_base_forward_shape(self):
        m = BaseCnn(in_h=16, in_w=16)
        y = m(torch.zeros(2, 3, 16, 16, dtype=torch.uint8))
        self.assertEqual(tuple(y.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
