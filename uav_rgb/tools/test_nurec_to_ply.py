#!/usr/bin/env python3
"""nurec_to_ply 的测试：合成极小 NuRec（几个高斯）-> usdz -> INRIA PLY，查字段顺序、f_rest 转置、读回一致。

运行：
  /opt/conda/bin/python3 tools/test_nurec_to_ply.py     # 全部用例（msgpack 只在这个解释器里有）
  $UAV_RGB_PYTHON tools/test_nurec_to_ply.py            # check.sh 用的解释器没有 msgpack：依赖 msgpack 的用例 SKIP 并打印，其余照跑
"""
import gzip
import importlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
nurec = importlib.import_module("nurec_to_ply")
scene_mod = importlib.import_module("sim.3dgs.scene")

try:
    import msgpack
except ImportError:
    msgpack = None
if msgpack is None:
    # 不静默：check.sh 的日志里要能看到这几个用例没跑
    print("SKIP: test_nurec_to_ply 中依赖 msgpack 的用例未跑（本解释器没有 msgpack；用 /opt/conda/bin/python3 跑全部）",
          file=sys.stderr)
needs_msgpack = unittest.skipIf(msgpack is None, "SKIP: 本解释器没有 msgpack")

P = nurec.STATE_PREFIX
N = 3                                   # 合成高斯数
# 取 fp16 能精确表示的值（0.25 的倍数），读回后可以逐位比较
POS = np.array([[1.0, 2.0, -0.5], [0.25, -3.0, 1.5], [4.0, 0.0, 0.75]], dtype=np.float32)
ROT = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
SCA = np.array([[-4.0, -4.5, -5.0], [-3.0, -3.0, -3.0], [0.25, 0.5, 0.75]], dtype=np.float32)
DEN = np.array([[2.0], [-1.5], [0.0]], dtype=np.float32)
DC = np.array([[0.5, 0.25, -0.25], [1.0, 0.0, 0.75], [-1.0, 1.5, 0.5]], dtype=np.float32)
# NuRec 的 features_specular 布局 [N, 15 个系数, 3 个通道]；每个元素编号成 n*100 + k*3 + c，便于认出转置错在哪
REST = (np.arange(N)[:, None, None] * 100 + np.arange(15)[None, :, None] * 3 + np.arange(3)[None, None, :]).astype(np.float32) / 4.0


def arrays():
    return {"positions": POS, "rotations": ROT, "scales": SCA, "densities": DEN,
            "features_albedo": DC, "features_specular": REST.reshape(N, 45)}


def nurec_bytes(dtype=np.float16):
    """合成的 .nurec 内容：gzip(msgpack({"nre_data": {"state_dict": {...}}}))，数组是 fp16/fp32 字节串 + .shape 项。"""
    sd = {}
    for name, a in arrays().items():
        a = a.astype(dtype)
        sd[P + name] = a.tobytes()
        sd[P + name + ".shape"] = list(a.shape)
    return gzip.compress(msgpack.packb({"nre_data": {"state_dict": sd}}, use_bin_type=True))


def make_usdz(path, nurec_raw, nurec_name="demo.usdz.nurec"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("default.usda", "#usda 1.0\n")
        z.writestr(nurec_name, nurec_raw)


def ply_header_props(path):
    props = []
    with open(path, "rb") as f:
        for line in f:
            t = line.decode("ascii").strip()
            if t.startswith("property float "):
                props.append(t.split()[2])
            if t == "end_header":
                break
    return props


class TestTable(unittest.TestCase):
    """纯 numpy 部分：字段顺序与 f_rest 转置（不需要 msgpack）。"""

    def test_property_order_matches_repo_reader(self):
        names, table = nurec.to_inria_table(arrays())
        self.assertEqual(list(names), list(scene_mod.PLY_PROPS))        # x y z nx ny nz f_dc_0-2 f_rest_0-44 opacity scale_0-2 rot_0-3
        self.assertEqual(table.shape, (N, 62))
        self.assertEqual(table.dtype, np.float32)

    def test_columns(self):
        names, table = nurec.to_inria_table(arrays())
        col = {k: i for i, k in enumerate(names)}
        self.assertTrue(np.array_equal(table[:, [col["x"], col["y"], col["z"]]], POS))
        self.assertTrue((table[:, [col["nx"], col["ny"], col["nz"]]] == 0).all())
        self.assertTrue(np.array_equal(table[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]], DC))
        self.assertTrue(np.array_equal(table[:, col["opacity"]], DEN[:, 0]))
        self.assertTrue(np.array_equal(table[:, [col["scale_0"], col["scale_1"], col["scale_2"]]], SCA))
        self.assertTrue(np.array_equal(table[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]], ROT))

    def test_f_rest_is_transposed_to_channel_major(self):
        """NuRec [N,15 系数,3 通道] -> INRIA f_rest_{c*15+k}。不转置时墙面出粉绿斑（bench 报告 2.3 实测）。"""
        names, table = nurec.to_inria_table(arrays())
        col = {k: i for i, k in enumerate(names)}
        for n in range(N):
            for k in range(15):
                for c in range(3):
                    self.assertEqual(table[n, col[f"f_rest_{c * 15 + k}"]], REST[n, k, c], (n, k, c))

    def test_rest_given_as_n_by_15_by_3(self):
        a = arrays()
        a["features_specular"] = REST                                   # 3 维形状也认
        _, t3 = nurec.to_inria_table(a)
        _, t2 = nurec.to_inria_table(arrays())
        self.assertTrue(np.array_equal(t3, t2))

    def test_rejects_wrong_rest_size(self):
        a = arrays()
        a["features_specular"] = np.zeros((N, 24), dtype=np.float32)    # 不是 15*3
        with self.assertRaises(ValueError):
            nurec.to_inria_table(a)

    def test_write_then_read_with_repo_reader(self):
        names, table = nurec.to_inria_table(arrays())
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "g.ply")
            nurec.write_inria_ply(p, names, table)
            self.assertEqual(ply_header_props(p), list(scene_mod.PLY_PROPS))
            g = scene_mod.read_gaussian_ply(p)
        self.assertTrue(np.array_equal(g["means"], POS))
        self.assertEqual(g["sh_degree"], 3)
        self.assertEqual(g["sh"].shape, (N, 16, 3))
        self.assertTrue(np.array_equal(g["sh"][:, 0, :], DC))
        self.assertTrue(np.array_equal(g["sh"][:, 1:, :], REST))        # 读端 [N,15,3] 与 NuRec 原布局一致
        self.assertTrue(np.allclose(g["opacities"], 1.0 / (1.0 + np.exp(-DEN[:, 0]))))
        self.assertTrue(np.allclose(g["scales"], np.exp(SCA)))


class TestMsgpackMissing(unittest.TestCase):
    def test_clear_error_when_msgpack_missing(self):
        with mock.patch.dict(sys.modules, {"msgpack": None}):           # import msgpack -> ImportError
            with self.assertRaises(SystemExit) as cm:
                nurec.require_msgpack()
        msg = str(cm.exception)
        self.assertIn("msgpack", msg)
        self.assertIn("/opt/conda/bin/python3", msg)                    # 提示该用哪个解释器
        self.assertIn("不", msg)                                        # 并说明不往现有 venv 装包


@needs_msgpack
class TestDecode(unittest.TestCase):
    def test_decode_fp16_exact(self):
        got = nurec.decode_nurec(nurec_bytes(np.float16))
        for name, a in arrays().items():
            self.assertEqual(got[name].dtype, np.float32)
            self.assertTrue(np.array_equal(got[name].reshape(a.shape), a), name)

    def test_decode_fp32(self):
        got = nurec.decode_nurec(nurec_bytes(np.float32))
        self.assertTrue(np.array_equal(got["positions"], POS))

    def test_decode_missing_key(self):
        sd = {P + "positions": POS.astype(np.float16).tobytes(), P + "positions.shape": [N, 3]}
        raw = gzip.compress(msgpack.packb({"nre_data": {"state_dict": sd}}, use_bin_type=True))
        with self.assertRaises(ValueError) as cm:
            nurec.decode_nurec(raw)
        self.assertIn("rotations", str(cm.exception))

    def test_decode_bad_itemsize(self):
        sd = {}
        for name, a in arrays().items():
            sd[P + name] = a.astype(np.float16).tobytes() + b"\x00"     # 长度对不上 shape
            sd[P + name + ".shape"] = list(a.shape)
        raw = gzip.compress(msgpack.packb({"nre_data": {"state_dict": sd}}, use_bin_type=True))
        with self.assertRaises(ValueError):
            nurec.decode_nurec(raw)

    def test_usdz_to_ply_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            usdz, ply = os.path.join(tmp, "demo.usdz"), os.path.join(tmp, "demo.ply")
            make_usdz(usdz, nurec_bytes())
            info = nurec.convert(usdz, ply)
            self.assertEqual(info["n_gaussians"], N)
            self.assertEqual(info["props"], 62)
            self.assertEqual(os.path.getsize(ply), info["ply_bytes"])
            g = scene_mod.read_gaussian_ply(ply)
        self.assertTrue(np.array_equal(g["means"], POS))
        self.assertTrue(np.array_equal(g["sh"][:, 1:, :], REST))

    def test_usdz_without_nurec(self):
        with tempfile.TemporaryDirectory() as tmp:
            usdz = os.path.join(tmp, "empty.usdz")
            with zipfile.ZipFile(usdz, "w") as z:
                z.writestr("default.usda", "#usda 1.0\n")
            with self.assertRaises(ValueError) as cm:
                nurec.convert(usdz, os.path.join(tmp, "o.ply"))
            self.assertIn(".nurec", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
