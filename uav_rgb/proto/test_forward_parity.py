#!/usr/bin/env python3
"""第一关验收：PyTorch 前向与真实 uavrgb_infer 逐帧对拍（需要 GPU、C++ 推理二进制、数据盘上的权重与数据集）。
默认跳过；开关 UAVRGB_PROTO_REAL=1：

  UAVRGB_PROTO_REAL=1 CUDA_VISIBLE_DEVICES=<UUID> $D/teacher-venv/bin/python proto/test_forward_parity.py

三项：
  1. cal_swa_s1_k10.bin + 它训练集（trunc/ds）里的 2000 帧，四类概率逐帧相差 ≤ 1e-5，动作无不同
  2. C++ 训练的 1 轮 prev-action 模型（run_cpp_prevaction.sh 的产物）同样对拍
  3. 双帧 / prev-action / 文本 / 辅助头的随机权重变体：写出的 checkpoint C++ 读得进去，概率一致
可用环境变量换路径：UAVRGB_PROTO_INFER（推理二进制）、UAVRGB_PROTO_PREV_CKPT / UAVRGB_PROTO_PREV_DATA。
"""
import argparse
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

D = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
REAL = os.environ.get("UAVRGB_PROTO_REAL") == "1"
INFER = os.environ.get("UAVRGB_PROTO_INFER", os.path.join(D, "bin-snapshots/main-97ae04b/uavrgb_infer"))
CALSWA = os.path.join(D, "exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin")
TRUNC = os.path.join(D, "tmp/agent6-distill/trunc/ds/dataset.uavr")
PREV_CKPT = os.environ.get("UAVRGB_PROTO_PREV_CKPT", os.path.join(D, "exp-proto-parity-20261112/cpp-prevaction-1ep/run/ckpt.bin"))
PREV_DATA = os.environ.get("UAVRGB_PROTO_PREV_DATA", os.path.join(D, "exp-student-select-20261109/ds-0417-aux/dataset.uavr"))

try:
    import torch  # noqa: F401
    import parity_forward as pf
    HAVE = True
except ImportError:   # pragma: no cover
    HAVE = False


def args(**kw):
    return argparse.Namespace(**kw)


@unittest.skipUnless(REAL and HAVE and os.path.isfile(INFER), "UAVRGB_PROTO_REAL=1 且有 torch 与 uavrgb_infer 才测")
class TestForwardParity(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(CALSWA) and os.path.isfile(TRUNC), "缺 calswa 权重或 trunc 数据集")
    def test_calswa_s1_two_thousand_val_frames(self):
        r = pf.run_real(args(ckpt=CALSWA, data=TRUNC, infer_bin=INFER, frames=2000))
        self.assertTrue(r["dataset_sha256_matches_contract"])
        for bs in (1, 100):
            self.assertLessEqual(r[f"batch{bs}"]["max_abs_prob_diff"], pf.TOL)
            self.assertEqual(r[f"batch{bs}"]["argmax_mismatch"], 0)

    @unittest.skipUnless(os.path.isfile(PREV_CKPT) and os.path.isfile(PREV_DATA), "缺 prev-action 1 轮权重或带侧车的数据集")
    def test_cpp_trained_prev_action_model(self):
        r = pf.run_real(args(ckpt=PREV_CKPT, data=PREV_DATA, infer_bin=INFER, frames=2000))
        self.assertEqual(r["hello"]["prev_action_dim"], 4)
        for bs in (1, 100):
            self.assertLessEqual(r[f"batch{bs}"]["max_abs_prob_diff"], pf.TOL)
            self.assertEqual(r[f"batch{bs}"]["argmax_mismatch"], 0)

    @unittest.skipUnless(os.path.isfile(TRUNC), "缺 trunc 数据集")
    def test_structure_variants_loaded_by_cpp(self):
        r = pf.run_variants(args(data=TRUNC, infer_bin=INFER, frames=60))
        for name, v in r.items():
            if name == "pass":
                continue
            with self.subTest(name):
                self.assertLessEqual(v["max_abs_prob_diff"], pf.TOL)
                self.assertEqual(v["argmax_mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
