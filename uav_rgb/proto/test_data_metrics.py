#!/usr/bin/env python3
"""proto/data.py 与 proto/metrics.py 的测试：只用 numpy，合成 UAVR 文件，不需要 GPU。
UAVRGB_PROTO_REAL=1 时另测数据盘上的 trunc 数据集（划分大小、sha256、auto 类别权重与 C++ 参照 run 的 config.json 一致）。
"""
import json
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as dsmod  # noqa: E402
import metrics as mt  # noqa: E402

D = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
REAL = os.environ.get("UAVRGB_PROTO_REAL") == "1"
TRUNC = os.path.join(D, "tmp/agent6-distill/trunc/ds/dataset.uavr")
REF_CONFIG = os.path.join(D, "exp-student-select-20261109/e1-noise/seed0/run/config.json")


def write_uavr(path, episodes, collisions=None, h=8, w=8, scheme="geometry_gid", seed=0):
    """episodes：[(split, scene_id, gid, count)]。图像第 i 个样本的所有像素 = i % 256，标签 = i % 4。"""
    n = sum(e[3] for e in episodes)
    imgs = np.zeros((n, 3, h, w), np.uint8)
    imgs[:] = (np.arange(n) % 256).astype(np.uint8)[:, None, None, None]
    labels = (np.arange(n) % 4).astype(np.uint8)
    eps, start = [], 0
    for k, (split, scene, gid, count) in enumerate(episodes):
        eps.append({"episode_id": k, "scene_id": scene, "geometry_gid": gid, "split": split,
                    "sample_start": start, "sample_count": count, "source_dir": f"/x/{k}"})
        start += count
    meta = {"schema": "uavrgb.dataset.v1", "split_policy": scheme, "split_scheme": scheme, "episodes": eps,
            "collision_flags": collisions if collisions is not None else "0" * n}
    mb = json.dumps(meta, separators=(",", ":")).encode()
    lab_off = 48 + imgs.size
    hdr = b"UAVR" + struct.pack("<HHIHHBBBBQQQI", 1, 48, n, h, w, 3, 1, 1, 0, 48, lab_off, lab_off + n, len(mb))
    with open(path, "wb") as f:
        f.write(hdr + imgs.tobytes() + labels.tobytes() + mb)
    return n


EPS = [("train", 1, "g1", 5), ("val", 2, "g2", 4), ("train", 3, "g3", 6), ("test", 4, "g4", 3)]


class TestUavr(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.td.name, "dataset.uavr")

    def tearDown(self):
        self.td.cleanup()

    def test_splits_follow_episode_order(self):
        n = write_uavr(self.path, EPS)
        ds = dsmod.Uavr(self.path, drop_collision=False)
        self.assertEqual((ds.n, ds.h, ds.w), (n, 8, 8))
        self.assertEqual(ds.splits["train"].tolist(), [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14])   # meta 里的 episode 顺序
        self.assertEqual(ds.splits["val"].tolist(), [5, 6, 7, 8])
        self.assertEqual(ds.splits["test"].tolist(), [15, 16, 17])
        x, y = ds.take([7, 2])
        self.assertEqual((x[0].min(), x[0].max(), x[1].min()), (7, 7, 2))
        self.assertEqual(y.tolist(), [3, 2])

    def test_drop_collision_removes_flagged_steps_from_all_splits(self):
        flags = "".join("1" if i in (1, 6, 16) else "0" for i in range(18))
        write_uavr(self.path, EPS, collisions=flags)
        ds = dsmod.Uavr(self.path, drop_collision=True)
        self.assertNotIn(1, ds.splits["train"].tolist())
        self.assertNotIn(6, ds.splits["val"].tolist())
        self.assertNotIn(16, ds.splits["test"].tolist())
        self.assertEqual(sum(len(v) for v in ds.splits.values()), 15)

    def test_drop_collision_requires_flags(self):
        write_uavr(self.path, EPS, collisions="")
        with self.assertRaises(dsmod.DatasetError):
            dsmod.Uavr(self.path, drop_collision=True)

    def test_auto_class_weights(self):
        write_uavr(self.path, EPS)
        ds = dsmod.Uavr(self.path, drop_collision=False)
        cnt = ds.class_counts("train")
        w = ds.auto_class_weights()
        for k in range(4):
            self.assertEqual(w[k], np.float32(len(ds.splits["train"]) / (4.0 * cnt[k])))
        self.assertEqual(w.dtype, np.float32)

    def test_rejects_scene_or_gid_across_splits(self):
        write_uavr(self.path, [("train", 1, "g1", 3), ("val", 1, "g2", 3)])
        with self.assertRaises(dsmod.DatasetError):
            dsmod.Uavr(self.path, drop_collision=False)
        write_uavr(self.path, [("train", 1, "g1", 3), ("val", 2, "g1", 3)])
        with self.assertRaises(dsmod.DatasetError):
            dsmod.Uavr(self.path, drop_collision=False)

    def test_rejects_truncated_and_bad_header(self):
        write_uavr(self.path, EPS)
        with open(self.path, "rb") as f:
            raw = f.read()
        for name, data in (("trunc", raw[:-5]), ("magic", b"XXXX" + raw[4:]), ("trail", raw + b"\0")):
            with self.subTest(name):
                with open(self.path, "wb") as f:
                    f.write(data)
                with self.assertRaises(dsmod.DatasetError):
                    dsmod.Uavr(self.path, drop_collision=False)

    def test_aux_sidecar(self):
        write_uavr(self.path, EPS)
        self.assertIsNone(dsmod.read_aux(self.path))
        n = 18
        hdr = b"UAVRAUX1" + struct.pack("<QIHHBBH", n, 0, 0, 0, 0, 255, 0) + bytes(32) + bytes(32) + bytes(36)
        self.assertEqual(len(hdr), 128)
        prev_index = np.maximum(np.arange(n) - 1, 0).astype("<u4")
        prev_action = (np.arange(n) % 4).astype(np.uint8)
        with open(os.path.splitext(self.path)[0] + ".aux", "wb") as f:
            f.write(hdr + prev_index.tobytes() + prev_action.tobytes())
        aux = dsmod.read_aux(self.path)
        self.assertEqual(aux["prev_index"].tolist(), prev_index.tolist())
        self.assertEqual(aux["prev_action"].tolist(), prev_action.tolist())


class TestMetrics(unittest.TestCase):
    # C++ 参照 run（e1-noise/seed0）最后一轮的 val 混淆矩阵与它在 metrics.jsonl 里记的指标
    CONF = [0, 0, 0, 0, 5, 421, 33, 42, 1, 17, 16, 6, 0, 15, 1, 17]

    def test_matches_cpp_logged_values(self):
        c = np.array(self.CONF).reshape(4, 4)
        self.assertAlmostEqual(mt.macro_f1(c), 0.396273464, places=6)
        self.assertAlmostEqual(mt.balanced_accuracy(c), 0.438867718, places=6)
        np.testing.assert_allclose(mt.per_class_recall(c), [0, 421 / 501, 16 / 40, 17 / 33])

    def test_empty_class_and_zero_division(self):
        c = np.zeros((4, 4), np.int64)
        c[1, 1] = 3
        self.assertEqual(mt.macro_f1(c), (2 * 1.0 * 1.0 / 2.0) / 4)   # 只有类 1 有分，其余 F1 记 0，分母仍是 4
        self.assertEqual(mt.per_class_recall(c).tolist(), [0, 1, 0, 0])
        c2 = np.zeros((4, 4), np.int64)
        c2[2, 3] = 5     # 类 2 有真值但全错、类 3 有预测但无命中：P+R=0 → F1 0
        self.assertEqual(mt.macro_f1(c2), 0.0)

    def test_confusion_builder(self):
        c = mt.confusion([0, 1, 1, 2], [0, 1, 2, 2], 3)
        self.assertEqual(c.tolist(), [[1, 0, 0], [0, 1, 1], [0, 0, 1]])


@unittest.skipUnless(REAL and os.path.isfile(TRUNC) and os.path.isfile(REF_CONFIG), "UAVRGB_PROTO_REAL=1 且有 trunc 数据集")
class TestRealDataset(unittest.TestCase):
    def test_trunc_splits_hash_and_weights(self):
        ds = dsmod.Uavr(TRUNC, drop_collision=True)
        self.assertEqual({k: len(v) for k, v in ds.splits.items()}, {"train": 8205, "val": 574, "test": 884})
        self.assertEqual(ds.class_counts("train").tolist(), [56, 6885, 639, 625])
        self.assertEqual(ds.class_counts("val").tolist(), [0, 501, 40, 33])
        self.assertEqual(ds.sha256, "534ca5904ea027b99a59137f1985a7b1b67646492bed124fb6bcae44309dfcac")
        with open(REF_CONFIG, encoding="utf-8") as f:
            ref = json.load(f)
        # config.json 里的权重是 float32 以 9 位有效数字打印的，比较时同样转 float32
        np.testing.assert_array_equal(ds.auto_class_weights(), np.array(ref["class_weights"], np.float32))


if __name__ == "__main__":
    unittest.main()
