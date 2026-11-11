#!/usr/bin/env python3
"""audit_fit.py 的测试（卡 E2E-0）：特征组矩阵、子集掩码、混淆矩阵与指标、小 MLP 拟合、分箱核对。

全部用手算得出的小例子，不读真实采集。torch 不在时拟合类用例跳过（teacher-venv 里有）。
运行（仓库根）：
  TMPDIR=$D/tmp/phase12-cards/e2e-0/tmp PYTHONDONTWRITEBYTECODE=1 $D/teacher-venv/bin/python experiments/2026-11-12-decidability-audit/test_audit_fit.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_fit as fit  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

S, F_, L, R = 0, 1, 2, 3


def toy_features(n=6):
    """n 帧的最小特征字典：与 features.npz 的字段同名。第 0 帧中间 20 列全是 4.0（画面里没有障碍）。"""
    rng = np.random.RandomState(0)
    d_wide = rng.uniform(0.5, 3.9, size=(n, 60)).astype(np.float32)
    d_wide[0, 20:40] = 4.0
    return {"d_wide": d_wide,
            "n_raw": np.array([0, 1, 2, 5, -3, -9][:n], np.int16),
            "prev": np.array([0, 1, 2, 3, 1, 2][:n], np.int8),
            "last_dir": np.array([0, 0, 1, 1, 2, 2][:n], np.int8),
            "since": np.array([5, 5, 1, 3, 2, 5][:n], np.int8),
            "x": np.zeros(n, np.float32),
            "y": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0][:n], np.float32),
            "x_rem": np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0][:n], np.float32),
            "yaw": np.array([0.0, 90.0, 180.0, -90.0, 45.0, 10.0][:n], np.float32)}


class GroupMatrixTests(unittest.TestCase):
    def test_class_weights_auto(self):
        w = fit.class_weights_auto([10, 80, 5, 5])
        np.testing.assert_allclose(w, [100 / 40, 100 / 320, 100 / 20, 100 / 20])
        self.assertEqual(fit.class_weights_auto([0, 90, 5, 5])[0], 0.0, "没有的类权重 0（与 C++ 训练器一致）")

    def test_onehot(self):
        oh = fit.onehot(np.array([0, 2, 1]), 3)
        self.assertEqual(oh.dtype, np.float32)
        self.assertEqual(oh.tolist(), [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    def test_group_dimensions_and_composition(self):
        feats = toy_features()
        train = np.array([True, True, True, True, False, False])
        g = fit.build_groups(feats, train)
        dims = {k: v.shape[1] for k, v in g.items()}
        # d 20；上一步动作 4；n 的 9 个取值；hist = 上一步 4 + 最近转向方向 3 + 距上次转向步数 5；priv 2；航向码 3；sin/cos 2
        self.assertEqual(dims, {"S1": 20, "S2": 24, "S3": 29, "S4": 41, "S5": 81, "S6": 83, "S3c": 23, "S7": 85})
        self.assertEqual(list(g), ["S1", "S2", "S3", "S4", "S5", "S6", "S3c", "S7"])
        for k, v in g.items():
            self.assertEqual(v.shape[0], 6)
            self.assertEqual(v.dtype, np.float32)
        d = feats["d_wide"][:, 20:40] / 4.0
        np.testing.assert_allclose(g["S1"], d)
        np.testing.assert_allclose(g["S2"][:, :20], d)
        np.testing.assert_allclose(g["S5"][:, :60], feats["d_wide"] / 4.0)
        self.assertTrue((g["S1"][0] == 1.0).all(), "第 0 帧中间 20 列 = 4 m 封顶 → 归一化后 1.0")

    def test_n_is_clipped_to_plus_minus_4_before_one_hot(self):
        g = fit.build_groups(toy_features(), np.ones(6, bool))
        n_block = g["S3"][:, 20:29]                      # n = −4…4 的 9 位独热
        self.assertEqual(n_block.sum(1).tolist(), [1.0] * 6)
        self.assertEqual(n_block.argmax(1).tolist(), [4, 5, 6, 8, 1, 0], "n_raw 0,1,2,5,−3,−9 → 截断后 0,1,2,4,−3,−4 → 下标 +4")

    def test_heading_code_onehot_for_s3c(self):
        g = fit.build_groups(toy_features(), np.ones(6, bool))
        code = g["S3c"][:, 20:23]                        # 列顺序：码 0、码 2、码 3
        self.assertEqual(code.argmax(1).tolist(), [0, 0, 1, 1, 2, 2], "n=0,1 → 码 0；2,5 → 码 2（偏左）；−3,−9 → 码 3（偏右）")

    def test_hist_block_layout(self):
        g = fit.build_groups(toy_features(), np.ones(6, bool))
        hist = g["S4"][:, 29:41]
        np.testing.assert_allclose(hist[2], np.r_[fit.onehot([2], 4)[0], fit.onehot([1], 3)[0], fit.onehot([0], 5)[0]],
                                   err_msg="第 2 帧：上一步 2、最近转向左(1)、距上次转向 1 步 → 下标 0")

    def test_priv_standardised_with_train_statistics_only(self):
        feats = toy_features()
        train = np.array([True, True, True, True, False, False])
        g = fit.build_groups(feats, train)
        priv = g["S6"][:, 81:83]
        self.assertAlmostEqual(float(priv[train, 0].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(priv[train, 0].std()), 1.0, places=5)
        self.assertAlmostEqual(float(priv[train, 1].mean()), 0.0, places=5)
        self.assertGreater(float(priv[5, 0]), float(priv[3, 0]), "测试帧用训练统计量缩放，不重新标准化")

    def test_s7_adds_true_heading(self):
        g = fit.build_groups(toy_features(), np.ones(6, bool))
        sc = g["S7"][:, 83:85]
        np.testing.assert_allclose(sc[1], [np.sin(np.pi / 2), np.cos(np.pi / 2)], atol=1e-6)
        np.testing.assert_allclose(sc[3], [-1.0, 0.0], atol=1e-6)


class SubsetTests(unittest.TestCase):
    def test_masks(self):
        feats = toy_features(6)
        feats.update(ep_idx=np.array([0, 1, 2, 3, 4, 4], np.int32), ep_source=np.array([0, 1, 2, 3, 4], np.int8),
                     source_names=np.array(["trunc", "more_demos", "round_1", "round_2", "round_3"]))
        m = fit.subset_masks(feats, teacher=("trunc", "more_demos"), student=("round_1", "round_2", "round_3"),
                             student_pure=("round_2", "round_3"))
        self.assertEqual(m["teacher"].tolist(), [True, True, False, False, False, False])
        self.assertEqual(m["student"].tolist(), [False, False, True, True, True, True])
        self.assertEqual(m["student_r23"].tolist(), [False, False, False, True, True, True])
        self.assertEqual(m["empty"].tolist(), [True, False, False, False, False, False], "画面（60°）里 20 列全是 4 m 封顶才算没有障碍")
        self.assertTrue(m["all"].all())


class MetricTests(unittest.TestCase):
    CM = np.array([[1, 0, 0, 0],
                   [0, 80, 5, 5],
                   [0, 3, 6, 1],
                   [0, 2, 2, 6]])

    def test_confusion_by_cluster(self):
        y = np.array([F_, F_, L, R, L])
        pred = np.array([F_, L, L, R, R])
        cl = np.array([0, 0, 1, 1, 1])
        cm = fit.confusion_by_cluster(y, pred, cl, 2)
        self.assertEqual(cm.shape, (2, 4, 4))
        self.assertEqual(cm[0, F_, F_], 1)
        self.assertEqual(cm[0, F_, L], 1)
        self.assertEqual((cm[1, L, L], cm[1, R, R], cm[1, L, R]), (1, 1, 1))
        self.assertEqual(int(cm.sum()), 5)
        mask = np.array([True, False, True, True, False])
        self.assertEqual(int(fit.confusion_by_cluster(y, pred, cl, 2, mask).sum()), 3)

    def test_lr_ok_counts_only_turn_frames(self):
        y = np.array([F_, L, L, R, R])
        lr = np.array([5.0, 1.0, -1.0, -2.0, 3.0])        # L 对数几率 − R 对数几率：>0 判左
        cl = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(fit.lr_ok_by_cluster(y, lr, cl, 2), [1, 1])

    def test_metrics_from_counts_by_hand(self):
        m = fit.metrics_from_counts(self.CM, 16)
        self.assertAlmostEqual(float(m["acc"]), 93 / 111)
        np.testing.assert_allclose(m["recall"], [1.0, 80 / 90, 0.6, 0.6])
        self.assertAlmostEqual(float(m["f1_frl"]), (160 / 175 + 12 / 23 + 12 / 22) / 3)
        self.assertAlmostEqual(float(m["f1_4"]), (1.0 + 160 / 175 + 12 / 23 + 12 / 22) / 4)
        self.assertAlmostEqual(float(m["turn_recall"]), 15 / 20, msg="该转时转了：转向帧里被判成 L 或 R 的比例")
        self.assertAlmostEqual(float(m["lr_acc"]), 16 / 20)
        self.assertAlmostEqual(float(m["lr_cond"]), 12 / 15, msg="只看判成转向的帧，方向对的比例")
        self.assertAlmostEqual(float(m["false_turn"]), 10 / 90)
        self.assertEqual((int(m["n"]), int(m["n_turn"])), (111, 20))

    def test_metrics_broadcast_over_leading_axes_and_nan_when_absent(self):
        cm = np.stack([self.CM, np.zeros((4, 4), int)])
        m = fit.metrics_from_counts(cm, np.array([16, 0]))
        self.assertEqual(m["acc"].shape, (2,))
        self.assertAlmostEqual(float(m["acc"][0]), 93 / 111)
        self.assertTrue(np.isnan(m["acc"][1]) and np.isnan(m["lr_acc"][1]))

    def test_majority_baseline_rows(self):
        cm = fit.majority_baseline_counts(self.CM)
        np.testing.assert_array_equal(cm.sum(1), self.CM.sum(1))
        self.assertEqual(int(cm[:, F_].sum()), 111, "恒选 FORWARD_SLOW：所有预测落在第 1 列")


@unittest.skipUnless(HAVE_TORCH, "没有 torch")
class MlpTests(unittest.TestCase):
    CFG = {"hidden": 64, "layers": 2, "lr": 1e-2, "batch": 64, "max_epochs": 40, "patience": 6, "threads": 1}

    def _data(self, n, seed):
        rng = np.random.RandomState(seed)
        X = rng.uniform(0, 1, size=(n, 4)).astype(np.float32)
        y = np.where(X[:, 0] > 0.5, L, R).astype(np.int64)
        return X, y

    def test_learns_a_separable_rule_and_reports_epochs(self):
        Xtr, ytr = self._data(800, 1)
        Xva, yva = self._data(200, 2)
        Xte, yte = self._data(300, 3)
        w = np.array([0.0, 0.0, 1.0, 1.0])
        net, info = fit.fit_mlp(Xtr, ytr, Xva, yva, w, seed=0, cfg=self.CFG)
        logits = fit.predict_logits(net, Xte)
        self.assertEqual(logits.shape, (300, 4))
        self.assertGreater(float((logits.argmax(1) == yte).mean()), 0.95)
        self.assertTrue(1 <= info["best_epoch"] <= info["epochs_run"] <= self.CFG["max_epochs"], info)
        self.assertEqual(info["layers"], [4, 64, 64, 4], "输入 → 64 → 64 → 4 类")

    def test_same_seed_same_result_different_seed_differs(self):
        Xtr, ytr = self._data(400, 1)
        Xva, yva = self._data(100, 2)
        w = np.array([1.0, 1.0, 1.0, 1.0])
        a, _ = fit.fit_mlp(Xtr, ytr, Xva, yva, w, seed=3, cfg=self.CFG)
        b, _ = fit.fit_mlp(Xtr, ytr, Xva, yva, w, seed=3, cfg=self.CFG)
        c, _ = fit.fit_mlp(Xtr, ytr, Xva, yva, w, seed=4, cfg=self.CFG)
        la, lb, lc = (fit.predict_logits(n, Xva) for n in (a, b, c))
        np.testing.assert_array_equal(la, lb)
        self.assertFalse(np.array_equal(la, lc))

    def test_class_weight_shifts_the_logits_toward_the_rare_class(self):
        # 同一批特征上 L:R = 9:1 且完全不可分（标签与特征无关）。无权重：R 相对 L 的 logit 差 ≈ ln(1/9) = −2.2；
        # 按 N/(C·n_c) 加权：权重比 9:1 把先验拉平，差 ≈ 0。比平均 logit 差而不比 argmax，因为拉平后两类打平，argmax 取谁是噪声决定的
        rng = np.random.RandomState(5)
        X = rng.uniform(0, 1, size=(2000, 3)).astype(np.float32)
        y = np.where(rng.uniform(size=2000) < 0.9, L, R).astype(np.int64)
        Xv, yv = X[:500], y[:500]
        cnt = np.bincount(y, minlength=4)
        plain, _ = fit.fit_mlp(X, y, Xv, yv, np.array([0, 0, 1.0, 1.0]), seed=0, cfg=self.CFG)
        wtd, _ = fit.fit_mlp(X, y, Xv, yv, fit.class_weights_auto(cnt), seed=0, cfg=self.CFG)
        gap = lambda net: float((lambda lg: (lg[:, R] - lg[:, L]).mean())(fit.predict_logits(net, X)))  # noqa: E731
        self.assertLess(gap(plain), -1.5)
        self.assertGreater(gap(wtd) - gap(plain), 1.5)


class BinningTests(unittest.TestCase):
    def test_sector_levels(self):
        d = np.full((4, 20), 4.0, np.float32)
        d[1, 7] = 0.5          # 扇区 1（列 5–9）里最近 0.5 m → 近
        d[2, 12] = 1.0         # 扇区 2（列 10–14）正好 1.0 m → 中（边界归上一档：≥1.0 为中）
        d[3, 17] = 2.5         # 扇区 3（列 15–19）正好 2.5 m → 远
        lv = fit.sector_levels(d, n_sectors=4, edges=(1.0, 2.5))
        self.assertEqual(lv.tolist(), [[2, 2, 2, 2], [2, 0, 2, 2], [2, 2, 1, 2], [2, 2, 2, 2]])

    def test_mixed_radix_ids_are_unique(self):
        ids = fit.mixed_radix([np.array([0, 1, 2, 0]), np.array([0, 0, 1, 1])], [3, 2])
        self.assertEqual(ids.tolist(), [0, 2, 5, 1])
        self.assertEqual(len(set(ids.tolist())), 4)

    def test_bin_check_by_hand(self):
        # 训练帧：箱 0 的标签 F,F,L → 多数 F；箱 1 的标签 R,R,L → 多数 R；箱 5 的标签 L,R → 左右平局；箱 7 全是 F。训练集总体多数类是 F（6 帧）
        tr_bin = np.array([0, 0, 0, 1, 1, 1, 5, 5, 7, 7, 7, 7])
        tr_y = np.array([F_, F_, L, R, R, L, L, R, F_, F_, F_, F_])
        te_bin = np.array([0, 1, 1, 2, 5])
        te_y = np.array([F_, R, L, F_, L])
        res = fit.bin_check(tr_bin, tr_y, te_bin, te_y)
        self.assertAlmostEqual(res["purity_train"], (2 + 2 + 1 + 4) / 12, msg="每箱多数类的帧数之和 / 总帧数")
        self.assertEqual(res["bins_train"], 4)
        self.assertAlmostEqual(res["heldout_acc"], 4 / 5,
                               msg="箱 0 判 F 对；箱 1 判 R：R 对 L 错；箱 2 没见过，退回训练集多数类 F 对；箱 5 平局取类号小的 L 对")
        self.assertAlmostEqual(res["coverage"], 4 / 5, msg="测试帧里 4 帧所在的箱训练集见过")
        self.assertAlmostEqual(res["heldout_lr"], (1.0 + 0.0 + 0.5) / 3, msg="转向帧 R/L/L：箱 1 判右（对）、箱 1 判右（错）、箱 5 平局记 0.5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
