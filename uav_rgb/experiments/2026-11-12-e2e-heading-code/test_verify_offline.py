#!/usr/bin/env python3
"""verify_infer_vs_evalonly.py 与 offline_report.py 里纯函数的回归（卡 E2E-1）：指标定义（缺类按 0、除以 4）、对拍容差、
码 2↔3 对调、探针统计、训练曲线读取。不起任何二进制，纯 CPU。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_verify_offline.py
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_report as off  # noqa: E402
import verify_infer_vs_evalonly as vf  # noqa: E402


def onehot_probs(pred, strength=0.9):
    p = np.full((len(pred), 4), (1 - strength) / 3)
    p[np.arange(len(pred)), pred] = strength
    return p


class MetricsTests(unittest.TestCase):
    def test_hand_computed_with_absent_class(self):
        # y = [0,0,1,1,2,3]，预测 = [0,1,1,1,2,2]：类 3 没被预测到
        y = np.array([0, 0, 1, 1, 2, 3])
        pred = np.array([0, 1, 1, 1, 2, 2])
        m = vf.metrics_from_probs(onehot_probs(pred), y)
        self.assertEqual(m["confusion"], [1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])
        self.assertAlmostEqual(m["acc"], 4 / 6)
        recall = [0.5, 1.0, 1.0, 0.0]
        precision = [1.0, 2 / 3, 0.5, 0.0]
        f1 = [2 * r * p / (r + p) if r + p else 0.0 for r, p in zip(recall, precision)]
        self.assertAlmostEqual(m["macro_f1"], sum(f1) / 4)
        self.assertAlmostEqual(m["balanced_acc"], sum(recall) / 4)
        self.assertAlmostEqual(m["ce"], float(-np.log(np.where(pred == y, 0.9, 0.1 / 3)).mean()))

    def test_absent_true_class_counts_as_zero_and_divides_by_four(self):
        y = np.array([0, 0, 1, 1])
        m = vf.metrics_from_probs(onehot_probs(y), y)
        self.assertAlmostEqual(m["macro_f1"], 0.5)           # 两个类满分，两个缺类按 0，除以 4
        self.assertAlmostEqual(m["balanced_acc"], 0.5)
        self.assertAlmostEqual(m["acc"], 1.0)


class CompareTests(unittest.TestCase):
    BASE = {"confusion": [5, 1, 0, 0, 0, 6, 0, 0, 0, 0, 3, 1, 0, 0, 1, 2], "acc": 0.8118466, "ce": 0.55, "macro_f1": 0.7, "balanced_acc": 0.72}

    def test_identical_and_within_tolerance(self):
        _, ok = vf.compare(self.BASE, dict(self.BASE))
        self.assertTrue(ok)
        near = dict(self.BASE, ce=self.BASE["ce"] + 0.9e-4, acc=self.BASE["acc"] + 0.5e-6)
        _, ok = vf.compare(self.BASE, near)
        self.assertTrue(ok)

    def test_each_item_fails_alone(self):
        for key, bad in (("confusion", [5, 1, 0, 0, 0, 5, 1, 0, 0, 0, 3, 1, 0, 0, 1, 2]), ("acc", 0.82), ("ce", 0.5502), ("macro_f1", 0.7001), ("balanced_acc", 0.7201)):
            items, ok = vf.compare(self.BASE, dict(self.BASE, **{key: bad}))
            self.assertFalse(ok, key)
            self.assertEqual(sum(1 for good, _ in items.values() if not good), 1, key)


class ProbeTests(unittest.TestCase):
    def test_swap_codes(self):
        cond = np.array([0, 2, 3, 1, 2, 0], dtype=np.uint8)
        np.testing.assert_array_equal(off.swap_codes(cond), [0, 3, 2, 1, 3, 0])
        np.testing.assert_array_equal(cond, [0, 2, 3, 1, 2, 0])         # 不改原数组

    def test_probe_stats_hand_computed(self):
        y = np.array([2, 3, 2, 0, 1, 2])
        cond = np.array([2, 2, 3, 0, 0, 3], dtype=np.uint8)
        real = onehot_probs(np.array([3, 2, 2, 0, 1, 3]))
        swap = onehot_probs(np.array([2, 2, 3, 0, 1, 2]))
        zero = onehot_probs(np.array([3, 2, 2, 0, 1, 3]))
        st = off.probe_stats(y, cond, real, swap, zero)
        self.assertEqual(st["frames"], 6)
        self.assertEqual(st["code_counts"], [2, 0, 2, 2])
        sw = st["swap"]
        self.assertEqual((sw["turn_pred_all"]["flip"], sw["turn_pred_all"]["n"]), (3, 4))
        self.assertEqual((sw["turn_pred_code_nonzero"]["flip"], sw["turn_pred_code_nonzero"]["n"]), (3, 4))
        self.assertEqual((sw["turn_label_and_pred_all"]["flip"], sw["turn_label_and_pred_all"]["n"]), (3, 4))
        self.assertEqual((sw["any_change_code_nonzero"]["flip"], sw["any_change_code_nonzero"]["n"]), (3, 4))
        self.assertEqual((sw["any_change_code_zero"]["flip"], sw["any_change_code_zero"]["n"]), (0, 2))     # 码 0 的帧输入没变，自检为 0
        self.assertAlmostEqual(st["zero"]["delta_macro_f1"], 0.0)                                              # 置 0 后预测不变
        self.assertEqual(st["zero"]["pred_changed_frames"], 0)

    def test_probe_zero_changes_macro_f1(self):
        y = np.array([2, 3, 2, 3])
        cond = np.array([2, 3, 2, 3], dtype=np.uint8)
        real = onehot_probs(np.array([2, 3, 2, 3]))
        zero = onehot_probs(np.array([0, 0, 2, 3]))
        st = off.probe_stats(y, cond, real, real, zero)
        self.assertLess(st["zero"]["delta_macro_f1"], 0)
        self.assertEqual(st["zero"]["pred_changed_frames"], 2)
        self.assertEqual(st["swap"]["turn_pred_all"]["flip"], 0)

    def test_empty_denominator_gives_none(self):
        y = np.array([0, 1])
        cond = np.array([0, 0], dtype=np.uint8)
        p = onehot_probs(np.array([0, 1]))
        st = off.probe_stats(y, cond, p, p, p)
        self.assertIsNone(st["swap"]["turn_pred_all"]["frac"])
        self.assertEqual(st["swap"]["turn_pred_all"]["n"], 0)


class CurveTests(unittest.TestCase):
    def write_run(self, d, epochs, val_events):
        with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
            f.write("epoch,train_ce,dev_ce,dev_accuracy\n")
            for e in range(1, epochs + 1):
                f.write(f"{e},{1.0 / e:.4f},{1.1 / e:.4f},{0.5 + 0.01 * e:.4f}\n")
        with open(os.path.join(d, "metrics.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "train_epoch", "epoch": 1}) + "\n")
            for i in range(val_events):
                f.write(json.dumps({"event": "validation", "split": "val", "macro_f1": 0.1 * (i + 1), "balanced_accuracy": 0.2 * (i + 1)}) + "\n")
                f.write(json.dumps({"event": "validation", "split": "test", "macro_f1": 9.9, "balanced_accuracy": 9.9}) + "\n")   # 不是 val，不能混进来
            f.write(json.dumps({"event": "log", "msg": "best checkpoint: epoch 3 (dev macro-F1 0.3)"}) + "\n")

    def test_read_curves_aligned(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_run(d, 5, 5)
            rows, note, best = off.read_curves(d)
            self.assertEqual(len(rows), 5)
            self.assertEqual(note, "")
            self.assertEqual(best, 3)
            self.assertAlmostEqual(rows[2]["dev_macro_f1"], 0.3)
            self.assertAlmostEqual(rows[4]["dev_bal_acc"], 1.0)
            self.assertAlmostEqual(rows[0]["train_ce"], 1.0)
            self.assertAlmostEqual(off.window_mean(rows, "dev_macro_f1", 2), (0.4 + 0.5) / 2)

    def test_read_curves_misaligned_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_run(d, 5, 4)
            rows, note, _ = off.read_curves(d)
            self.assertIn("对不上", note)
            self.assertTrue(all(r["dev_macro_f1"] is None for r in rows))
            self.assertIsNone(off.window_mean(rows, "dev_macro_f1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
