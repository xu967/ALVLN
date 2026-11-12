#!/usr/bin/env python3
"""proto/ckpt_io.py 的测试：只用 numpy（check.sh 的 .venv-tools 解释器也能跑）。

  python proto/test_ckpt_io.py                       # 合成 checkpoint 的读写往返、负例
  UAVRGB_PROTO_REAL=1 python proto/test_ckpt_io.py   # 另测数据盘上的真实 C++ checkpoint 写后读回与原文件逐字节一致
"""
import hashlib
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io as cio  # noqa: E402

D = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
REAL = os.environ.get("UAVRGB_PROTO_REAL") == "1"
CALSWA = os.path.join(D, "exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin")
CALSWA_SHA = "5cfe51f58f4f4a1d1afb4f79db0f11e777fd67271b9605f722b6bdf8372a6900"   # exp-swa-calib-20261110/inputs.txt

STRUCTURES = {
    "base": dict(in_h=16, in_w=24, in_c=3, num_classes=4),
    "two_frame": dict(in_h=16, in_w=24, in_c=6, num_classes=4),
    "prev": dict(in_h=16, in_w=24, in_c=3, num_classes=4, prev_dim=4),
    "text": dict(in_h=16, in_w=24, in_c=3, num_classes=4, text_dim=5),
    "aux": dict(in_h=16, in_w=24, in_c=3, num_classes=4, aux_classes=8),
    "all": dict(in_h=16, in_w=24, in_c=6, num_classes=4, text_dim=3, prev_dim=4, aux_classes=8),
}


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def make_ck(seed=0, **st):
    rng = np.random.default_rng(seed)
    shapes = cio.expected_shapes(**st)
    f = lambda: [rng.standard_normal(s).astype(np.float32) for s in shapes]   # noqa: E731
    return cio.Ckpt(**st, params=f(), adam_m=f(), adam_v=[np.abs(a) for a in f()], adam_t=65 * 3, epoch=3,
                    class_weights=np.array([36.6, 0.3, 3.2, 3.3], np.float32), best_macro_f1=0.4,
                    contract=cio.make_contract("ab" * 32, True, 0.001, 7, 128, "cd" * 32))


class TestRoundTrip(unittest.TestCase):
    def test_write_read_write_byte_identical(self):
        for name, st in STRUCTURES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as td:
                ck = make_ck(1, **st)
                a, b = os.path.join(td, "a.bin"), os.path.join(td, "b.bin")
                cio.write_checkpoint(a, ck)
                back = cio.read_checkpoint(a)
                cio.write_checkpoint(b, back)
                self.assertEqual(read_bytes(a), read_bytes(b))
                self.assertEqual(back.structure(), ck.structure())
                self.assertEqual((back.adam_t, back.epoch, back.contract), (ck.adam_t, ck.epoch, ck.contract))
                self.assertEqual(back.best_macro_f1, np.float32(0.4))
                for x, y in zip(ck.params + ck.adam_m + ck.adam_v, back.params + back.adam_m + back.adam_v):
                    np.testing.assert_array_equal(x, y)
                self.assertEqual(cio.read_meta(a).__dict__, {**dict(text_dim=0, aux_classes=0, prev_dim=0), **st})

    def test_param_counts(self):
        # 基础结构 100,388 个参数（model.cpp 的 params=100388），FiLM 层与 embedding 另加
        n = lambda **kw: sum(int(np.prod(s)) for s in cio.expected_shapes(**kw))   # noqa: E731
        self.assertEqual(n(in_h=120, in_w=160, in_c=3, num_classes=4), 100388)
        self.assertEqual(n(in_h=120, in_w=160, in_c=6, num_classes=4), 100388 + 3 * 16 * 9)
        self.assertEqual(n(in_h=120, in_w=160, in_c=3, num_classes=4, prev_dim=4), 100388 + 128 * 4 + 128)

    def test_contract_helpers(self):
        raw = cio.make_contract("ab" * 32, True, 0.001, 7, 128, "cd" * 32)
        self.assertEqual(len(raw), 113)
        c = cio.parse_contract(raw)
        self.assertEqual((c["dataset_sha256"], c["drop_collision"], c["seed"], c["batch"], c["split_scheme_sha256"]),
                         ("ab" * 32, True, 7, 128, "cd" * 32))
        self.assertEqual(c["soft_targets_sha256"], "")
        self.assertAlmostEqual(c["lr"], 0.001, places=9)


class TestRejects(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "c.bin")
        cio.write_checkpoint(self.path, make_ck(2, **STRUCTURES["prev"]))
        self.raw = read_bytes(self.path)

    def tearDown(self):
        self.td.cleanup()

    def put(self, data):
        with open(self.path, "wb") as f:
            f.write(data)

    def assert_rejects(self, data, fragment):
        self.put(data)
        with self.assertRaises(cio.CkptError) as cm:
            cio.read_checkpoint(self.path)
        self.assertIn(fragment, str(cm.exception))

    def test_bad_magic(self):
        self.assert_rejects(b"UAVRCK02" + self.raw[8:], "magic")

    def test_old_version(self):
        self.assert_rejects(self.raw[:8] + struct.pack("<I", 5) + self.raw[12:], "version")

    def test_trailing_bytes(self):
        self.assert_rejects(self.raw + b"\0", "多余数据")

    def test_truncated(self):
        self.assert_rejects(self.raw[:-4], "读取不足")

    def test_param_count(self):
        self.assert_rejects(self.raw[:12] + struct.pack("<I", 7) + self.raw[16:], "参数个数")

    def test_nan_param_rejected_on_read(self):
        ck = make_ck(2, **STRUCTURES["prev"])
        data = bytearray(cio.encode_checkpoint(ck))
        # 第一个参数 c1.W 的数据区起点：头 + 类别权重 + 契约 + ndim/dims
        off = 8 + 8 + 16 + 12 + 16 + 4 + 4 + 113 + 8 + (4 + 8 * 4)
        data[off:off + 4] = struct.pack("<f", float("nan"))
        self.assert_rejects(bytes(data), "非有限值")

    def test_nan_rejected_on_write(self):
        ck = make_ck(2, **STRUCTURES["base"])
        ck.params[0][0, 0, 0, 0] = np.inf
        with self.assertRaises(cio.CkptError):
            cio.encode_checkpoint(ck)
        ck = make_ck(2, **STRUCTURES["base"])
        ck.adam_v[3][0] = np.nan
        with self.assertRaises(cio.CkptError):
            cio.encode_checkpoint(ck)

    def test_wrong_dims_in_file(self):
        ck = make_ck(2, **STRUCTURES["base"])
        data = bytearray(cio.encode_checkpoint(ck))
        off = 8 + 8 + 16 + 12 + 16 + 4 + 4 + 113 + 8 + 4          # c1.W 的第一个 dim（16）
        data[off:off + 8] = struct.pack("<q", 17)
        self.assert_rejects(bytes(data), "形状")

    def test_class_weight_count(self):
        ck = make_ck(2, **STRUCTURES["base"])
        ck.class_weights = np.ones(3, np.float32)
        with self.assertRaises(cio.CkptError):
            cio.encode_checkpoint(ck)

    def test_bad_structure(self):
        with self.assertRaises(cio.CkptError):
            cio.expected_shapes(in_h=120, in_w=160, in_c=4, num_classes=4)
        with self.assertRaises(cio.CkptError):
            cio.expected_shapes(in_h=121, in_w=160, in_c=3, num_classes=4)
        with self.assertRaises(cio.CkptError):
            cio.expected_shapes(in_h=120, in_w=160, in_c=3, num_classes=4, prev_dim=65)

    def test_shape_mismatch_on_write(self):
        ck = make_ck(2, **STRUCTURES["base"])
        ck.params[1] = np.zeros(15, np.float32)
        with self.assertRaises(cio.CkptError):
            cio.encode_checkpoint(ck)


@unittest.skipUnless(REAL and os.path.isfile(CALSWA), "UAVRGB_PROTO_REAL=1 且数据盘有 cal_swa_s1_k10.bin 才测")
class TestRealCheckpoint(unittest.TestCase):
    def test_real_roundtrip_byte_identical(self):
        orig = read_bytes(CALSWA)
        self.assertEqual(hashlib.sha256(orig).hexdigest(), CALSWA_SHA)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "rt.bin")
            cio.write_checkpoint(out, cio.read_checkpoint(CALSWA))
            self.assertEqual(read_bytes(out), orig)

    def test_real_fields(self):
        ck = cio.read_checkpoint(CALSWA)
        self.assertEqual((ck.epoch, ck.adam_t, ck.in_c, ck.num_classes), (30, 1950, 3, 4))
        self.assertEqual(sum(a.size for a in ck.params), 100388)
        self.assertEqual(cio.parse_contract(ck.contract)["seed"], 1)


if __name__ == "__main__":
    unittest.main()
