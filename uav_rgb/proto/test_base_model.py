"""base_torch_v1 的最小形状和输入校验测试。"""
import importlib.util
import fcntl
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import torch
    from base_model import BaseCnn
    import train_base
except ImportError:  # pragma: no cover - 没有 torch 的工具环境跳过
    torch = None


@unittest.skipUnless(torch is not None, "没有 torch")
class TestBaseCnn(unittest.TestCase):
    def test_uint8_forward_shape(self):
        model = BaseCnn(16, 24)
        out = model(torch.zeros(2, 3, 16, 24, dtype=torch.uint8))
        self.assertEqual(tuple(out.shape), (2, 4))

    def test_wrong_shape_is_rejected(self):
        model = BaseCnn(16, 24)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 3, 15, 24))

    def test_run_lock_rejects_a_second_trainer(self):
        with tempfile.TemporaryDirectory() as out:
            lock_path = os.path.join(out, ".run.lock")
            with open(lock_path, "a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                cfg = {"train": {"out": out}}
                with self.assertRaisesRegex(ValueError, "另一个训练进程"):
                    train_base.run(cfg, max_steps=1)


if __name__ == "__main__":
    unittest.main()
