#!/usr/bin/env python3
"""audit_report.py 的测试（卡 E2E-0）：自助法权重、按特征组/子集取指标与区间、差值、预期判定。

合成的 stats（几何簇 × 4×4 混淆矩阵），数字都能手算。
运行（仓库根）：
  TMPDIR=$D/tmp/phase12-cards/e2e-0/tmp PYTHONDONTWRITEBYTECODE=1 $D/teacher-venv/bin/python experiments/2026-11-12-decidability-audit/test_audit_report.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_report as rep  # noqa: E402


def make_stats(cms, lr_oks, groups=("S1", "S3"), subsets=("student",), seeds=(0, 1, 2)):
    """cms[g][seed][cluster] = 4×4；lr_oks[g][seed][cluster] = 整数。每个特征组、每个种子、每个簇各给一份。"""
    cm = np.array(cms, np.int32)[:, :, None]          # [G, seeds, 1 subset, clusters, 4, 4]
    lr = np.array(lr_oks, np.int32)[:, :, None]
    return {"cm": cm, "lr_ok": lr, "groups": list(groups), "subsets": list(subsets), "seeds": list(seeds),
            "clusters": ["c%d" % i for i in range(cm.shape[3])], "weights": np.ones(4), "fit_subset": "all"}


def cm_from(acc_f, acc_l, acc_r, n_f=10, n_l=5, n_r=5):
    """FORWARD / LEFT / RIGHT 各自的召回给定，错的全判成 FORWARD（前进帧错的判成 LEFT）。"""
    cm = np.zeros((4, 4), int)
    cm[1, 1] = round(acc_f * n_f); cm[1, 2] = n_f - cm[1, 1]
    cm[2, 2] = round(acc_l * n_l); cm[2, 1] = n_l - cm[2, 2]
    cm[3, 3] = round(acc_r * n_r); cm[3, 1] = n_r - cm[3, 3]
    return cm


class BootstrapWeightTests(unittest.TestCase):
    def test_multiplicities(self):
        valid = np.array([True, False, True, True, False])
        m = rep.boot_weights(valid, 50, seed=1)
        self.assertEqual(m.shape, (50, 5))
        self.assertTrue((m.sum(1) == 3).all(), "每次抽 3 个有帧的簇")
        self.assertTrue((m[:, ~valid] == 0).all(), "没有帧的簇不参与")
        np.testing.assert_array_equal(m, rep.boot_weights(valid, 50, seed=1))
        self.assertFalse(np.array_equal(m, rep.boot_weights(valid, 50, seed=2)))


class MetricAccessTests(unittest.TestCase):
    def setUp(self):
        # 4 个簇、3 个种子，每个簇都一样：S1 前进召回 1.0、左转 0.2、右转 0.2；S3 全对
        s1 = [[cm_from(1.0, 0.2, 0.2)] * 4] * 3
        s3 = [[cm_from(1.0, 1.0, 1.0)] * 4] * 3
        self.st = make_stats([s1, s3], [[[1] * 4] * 3, [[10] * 4] * 3])

    def test_point_values(self):
        mean, lo, hi = rep.point(self.st, "S1", "student", "acc")
        # 每簇 20 帧：对 10 + 1 + 1 = 12
        self.assertAlmostEqual(mean, 12 / 20)
        self.assertEqual((lo, hi), (mean, mean), "种子间一样 → 极差为 0")
        self.assertAlmostEqual(rep.point(self.st, "S3", "student", "acc")[0], 1.0)
        self.assertAlmostEqual(rep.point(self.st, "S1", "student", "recall_L")[0], 0.2)
        self.assertAlmostEqual(rep.point(self.st, "S1", "student", "lr_acc")[0], 4 / 40, msg="每簇 10 个转向帧里 lr_ok = 1")
        self.assertAlmostEqual(rep.point(self.st, "S3", "student", "lr_acc")[0], 1.0)

    def test_seed_range(self):
        base = [cm_from(1.0, 1.0, 1.0)] * 2
        worse = [cm_from(1.0, 0.0, 1.0)] * 2
        st = make_stats([[base, worse, base]], [[[10, 10], [5, 5], [10, 10]]], groups=("S1",))
        mean, lo, hi = rep.point(st, "S1", "student", "recall_L")
        self.assertAlmostEqual(mean, 2 / 3)
        self.assertEqual((lo, hi), (0.0, 1.0))

    def test_identical_clusters_give_zero_width_interval_and_exact_contrast(self):
        valid = np.ones(4, bool)
        mult = rep.boot_weights(valid, 200, seed=3)
        lo, hi = rep.interval(self.st, "S1", "student", "acc", mult)
        self.assertAlmostEqual(lo, 12 / 20)
        self.assertAlmostEqual(hi, 12 / 20)
        diff, dlo, dhi = rep.contrast(self.st, "S3", "S1", "student", "acc", mult)
        self.assertAlmostEqual(diff, 1.0 - 12 / 20)
        self.assertAlmostEqual(dlo, diff)
        self.assertAlmostEqual(dhi, diff)
        self.assertEqual(rep.contrast(self.st, "S1", "S1", "student", "acc", mult), (0.0, 0.0, 0.0))

    def test_heterogeneous_clusters_widen_the_interval_around_the_point(self):
        good, bad = cm_from(1.0, 1.0, 1.0), cm_from(1.0, 0.0, 0.0)
        seeds = [[good, bad, good, bad, good, good]] * 3
        st = make_stats([seeds], [[[10, 0, 10, 0, 10, 10]] * 3], groups=("S1",))
        mult = rep.boot_weights(np.ones(6, bool), 500, seed=4)
        point, _, _ = rep.point(st, "S1", "student", "acc")
        lo, hi = rep.interval(st, "S1", "student", "acc", mult)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        self.assertGreater(hi - lo, 0.05)
        self.assertTrue(0.0 <= lo and hi <= 1.0)


class TableRowTests(unittest.TestCase):
    def test_interval_rows_follow_group_and_metric_order(self):
        s1 = [[cm_from(1.0, 0.2, 0.2)] * 4] * 3
        s3 = [[cm_from(1.0, 1.0, 1.0)] * 4] * 3
        st = make_stats([s1, s3], [[[1] * 4] * 3, [[10] * 4] * 3])
        mult = rep.boot_weights(np.ones(4, bool), 100, seed=7)
        rows = rep.interval_rows(st, ["S3", "S1"], "student", (("acc", "整体准确率"), ("lr_acc", "转向帧左右对")), mult)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0][0].startswith("S3") and rows[1][0].startswith("S1"))
        self.assertEqual(rows[0][1:], ["[1.000, 1.000]", "[1.000, 1.000]"])
        self.assertEqual(rows[1][1:], ["[0.600, 0.600]", "[0.100, 0.100]"])


class JudgeTests(unittest.TestCase):
    def test_judge_threshold(self):
        self.assertEqual(rep.judge(0.65, 0.60, 0.69, "<=", 0.70), "成立（区间在阈值同一侧）")
        self.assertEqual(rep.judge(0.68, 0.62, 0.74, "<=", 0.70), "成立（区间跨过阈值，未能分辨）")
        self.assertEqual(rep.judge(0.80, 0.78, 0.83, ">=", 0.85), "不成立（区间在阈值同一侧）")
        self.assertEqual(rep.judge(0.86, 0.83, 0.88, ">=", 0.85), "成立（区间跨过阈值，未能分辨）")
        self.assertEqual(rep.judge(0.03, 0.01, 0.04, "<", 0.05), "成立（区间在阈值同一侧）")
        self.assertEqual(rep.judge(0.12, 0.09, 0.15, ">=", 0.10), "成立（区间跨过阈值，未能分辨）")

    def test_judge_abs_less_than(self):
        self.assertEqual(rep.judge_abs_lt(0.02, -0.01, 0.04, 0.05), "成立（区间在阈值同一侧）")
        self.assertEqual(rep.judge_abs_lt(0.04, 0.01, 0.07, 0.05), "成立（区间跨过阈值，未能分辨）")
        self.assertEqual(rep.judge_abs_lt(0.08, 0.06, 0.10, 0.05), "不成立（区间在阈值同一侧）")
        self.assertEqual(rep.judge_abs_lt(-0.08, -0.10, -0.06, 0.05), "不成立（区间在阈值同一侧）")

    def test_format_cells(self):
        self.assertEqual(rep.fmt(0.8731, 0.8681, 0.8793), "0.873 [0.868, 0.879]")
        self.assertEqual(rep.fmt(float("nan"), float("nan"), float("nan")), "—")
        self.assertEqual(rep.fmt_ci(0.0123, -0.0201, 0.0457), "+0.012 [−0.020, +0.046]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
