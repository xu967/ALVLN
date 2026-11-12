#!/usr/bin/env python3
"""stability.py 的测试。合成数据上的参考值都是手算的（算法在每个用例的注释里），不是拿被测函数的输出当答案。
运行：python -m unittest experiments/2026-11-12-recipe-stability/test_stability.py -v（或 cd 到本目录 python test_stability.py）。
最后一组用例读真实的 R0 逐轮预测与审核 H 的日志，数据不在时跳过。
"""
import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stability as st  # noqa: E402

# 合成 val：10 帧，标签 = 8 个 FORWARD(1) + L(2) + R(3)；FORWARD 目标占比 0.8。
Y = np.array([1] * 8 + [2, 3])
TARGET = 0.8


def make_preds(gaps, y=Y):
    """每帧 logit = [0, g, -10, -10]：g > 0 预测 FORWARD(1)，否则预测 SAFE(0)。FORWARD 对 SAFE 的差恰为 g。返回 [idx, y, pred, p0..p3]。"""
    z = np.zeros((len(y), 4))
    z[:, 1] = gaps
    z[:, 2:] = -10.0
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return np.c_[np.arange(len(y)), y, p.argmax(1), p]


# 三个训练种子 × 三轮的 FORWARD 差 g（手算见各用例）。种子 2 与种子 0 相同；种子 1 只在第 3 轮把第 0 帧的 g 改成 −1。
G0 = [[4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, -0.5, -1],
      [4, 3.5, 3, 2.5, 2, 1.5, -0.2, -0.4, -0.6, -0.8],
      [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0.2, -0.3]]
G1 = [G0[0], G0[1], [-1, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0.2, -0.3]]
GS = {0: G0, 1: G1, 2: G0}


def synth_runs():
    return {s: {e + 1: make_preds(g) for e, g in enumerate(GS[s])} for s in GS}


class CorePrimitives(unittest.TestCase):
    def test_flip_series_reference(self):
        # e1 = 0,1,1,1,2,3；e2 与 e1 只差第 3 帧；e3 与 e2 只差第 0 帧 → 每对 1/6
        e1, e2, e3 = np.array([0, 1, 1, 1, 2, 3]), np.array([0, 1, 1, 2, 2, 3]), np.array([1, 1, 1, 2, 2, 3])
        np.testing.assert_allclose(st.flip_series([e1, e2, e3]), [1 / 6, 1 / 6])
        # 只看标签为 L/R 的帧（第 4、5 帧）：两对都没有翻转
        np.testing.assert_allclose(st.flip_series([e1, e2, e3], np.array([0, 0, 0, 0, 1, 1], bool)), [0.0, 0.0])
        # 只看第 3 帧：e1→e2 翻转，e2→e3 不翻
        np.testing.assert_allclose(st.flip_series([e1, e2, e3], np.array([0, 0, 0, 1, 0, 0], bool)), [1.0, 0.0])

    def test_flip_summary_windows(self):
        # 12 个相邻对 0.0, 0.1, …, 1.1：全程均值 0.55；末 9 对（下标 3..11）均值 0.7；末 10 对（下标 2..11）均值 0.65
        series = np.arange(12) / 10
        got = st.flip_summary(series)
        self.assertAlmostEqual(got["all"], 0.55)
        self.assertAlmostEqual(got["last9"], 0.7)
        self.assertAlmostEqual(got["last10"], 0.65)
        # 对数不足 9 时用全部
        self.assertAlmostEqual(st.flip_summary(np.array([0.2, 0.4]))["last9"], 0.3)

    def test_flip_segments(self):
        # 两个种子，各 5 对：s0 = 0.1,0.2,0.3,0.4,0.5；s1 = 0.3,0.3,0.3,0.3,0.3。n=5 → 前 2 对、后 2 对、最后 3 对，中间那对不算
        got = st.flip_segments({0: [0.1, 0.2, 0.3, 0.4, 0.5], 1: [0.3] * 5})
        self.assertAlmostEqual(got["first"], (0.15 + 0.3) / 2)        # s0 前 2 对均值 0.15，s1 0.3
        self.assertAlmostEqual(got["second"], (0.45 + 0.3) / 2)       # s0 后 2 对均值 0.45
        self.assertAlmostEqual(got["last3"], (0.4 + 0.3) / 2)         # s0 末 3 对均值 0.4
        self.assertEqual(got["pairs_half"], 2)
        # 29 对（30 轮）：前 14 对、后 14 对
        ser = list(range(29))
        got = st.flip_segments({0: ser})
        self.assertAlmostEqual(got["first"], np.mean(ser[:14]))
        self.assertAlmostEqual(got["second"], np.mean(ser[-14:]))
        self.assertEqual(got["pairs_half"], 14)

    def test_all_agree_and_pair_flip(self):
        a, b, c = np.array([1, 1, 2, 3]), np.array([1, 1, 2, 2]), np.array([1, 0, 2, 3])
        self.assertAlmostEqual(st.all_agree([a, b, c]), 0.5)                              # 第 0、2 帧三者相同
        self.assertAlmostEqual(st.all_agree([a, b, c], np.array([1, 0, 1, 0], bool)), 1.0)  # 只看第 0、2 帧
        self.assertAlmostEqual(st.all_agree([a, b, c], np.array([0, 1, 0, 1], bool)), 0.0)  # 第 1、3 帧都不一致
        # 两两翻转：(a,b) 1 帧，(a,c) 1 帧，(b,c) 2 帧 → (0.25 + 0.25 + 0.5) / 3
        self.assertAlmostEqual(st.pair_flip_mean([a, b, c]), 1 / 3)

    def test_class_metrics_reference(self):
        # y = 0,1,1,1,2,2,3,3；p = 1,1,1,0,2,1,3,3
        # 召回 = [0/1, 2/3, 1/2, 2/2]；类 0：P=0 R=0 → F1 0；类 1：P=2/4 R=2/3 → F1 4/7；类 2：P=1 R=1/2 → F1 2/3；类 3：F1 1
        y, p = np.array([0, 1, 1, 1, 2, 2, 3, 3]), np.array([1, 1, 1, 0, 2, 1, 3, 3])
        m = st.class_metrics(y, p)
        np.testing.assert_allclose(m["recall"], [0, 2 / 3, 0.5, 1.0])
        np.testing.assert_allclose(m["f1"], [0, 4 / 7, 2 / 3, 1.0])
        self.assertAlmostEqual(m["macro_f1"], (0 + 4 / 7 + 2 / 3 + 1) / 4)
        self.assertAlmostEqual(m["balanced_acc"], (0 + 2 / 3 + 0.5 + 1) / 4)
        self.assertAlmostEqual(m["acc"], 5 / 8)

    def test_class_metrics_absent_class(self):
        # val 里没有 SAFE：召回记 0，预测过 SAFE 的话 P=0 → F1 0，macro 仍除以 4（sklearn 口径，与 select_ckpt 同）
        # p = 1×9 + 0；类 1：P=8/9 R=1 → F1 16/17；类 0、2、3：F1 0
        m = st.class_metrics(Y, np.array([1] * 9 + [0]))
        self.assertAlmostEqual(m["macro_f1"], (16 / 17) / 4)
        self.assertAlmostEqual(m["balanced_acc"], 0.25)
        self.assertAlmostEqual(m["acc"], 0.8)
        self.assertEqual(m["recall"][0], 0.0)

    def test_calib_delta_reference(self):
        # 每帧 logit [0, g, −10, −10]：校准后 FORWARD 占比不低于目标的最小 δ = −（第 k 大的 g），k = ceil(目标 × 帧数)
        # δ 恰好落在某一帧的翻转点上，二分取区间中点，浮点误差会让那一帧落在哪一边不确定（h1 的算法同样），所以不断言 δ 处的预测，
        # 而断言 δ 本身，以及「δ 以下 1e-9 占比不足、δ 以上 1e-9 占比达标」这条最小性。
        g = np.array([3.0, 1.0, -0.5, -2.0, -4.0])
        probs = make_preds(g, np.ones(5, int))[:, 3:7]

        def share_at(shift):
            z = np.log(probs)
            z[:, 1] += shift
            return float((z.argmax(1) == 1).mean())

        for target, k, want in ((0.6, 3, 0.5), (0.4, 2, -1.0), (1.0, 5, 4.0), (0.2, 1, -3.0)):
            _, d = st.calib_pred(probs, target)
            self.assertAlmostEqual(d, want, places=9, msg=f"目标 {target}：第 {k} 大的 g = {-want}")
            self.assertLess(share_at(d - 1e-9), target)
            self.assertGreaterEqual(share_at(d + 1e-9), target)
        # 偏离翻转点时预测是确定的：目标 0.5 不是 1/5 的整数倍，k = 3，第 3 大的 g = −0.5 → δ = +0.5，前 3 帧是 FORWARD
        pred, d = st.calib_pred(probs, 0.5)
        self.assertAlmostEqual(d, 0.5, places=9)
        self.assertEqual(int((pred == 1).sum()) in (2, 3), True)

    def test_pass_verdict_boundaries(self):
        ok = {"worst_last9": 0.06, "mean_last9": 0.05, "worst_last10": 0.06}
        agree = {"raw_last10": 0.88, "raw_last_epoch": 0.9, "avg_raw": 0.9}
        v = st.pass_verdict(ok, agree)
        self.assertTrue(v["primary"] and v["pass_flip"] and v["pass_agree"] and v["consistent"])   # 0.06、0.88 正好在线上算过
        v = st.pass_verdict({**ok, "worst_last9": 0.0601}, agree)
        self.assertFalse(v["primary"] or v["pass_flip"])
        self.assertTrue(v["pass_agree"])
        v = st.pass_verdict(ok, {**agree, "raw_last10": 0.8799})
        self.assertFalse(v["primary"] or v["pass_agree"])
        # 判据读法过线、别的读法不过线 → primary 为真，consistent 为假
        v = st.pass_verdict({**ok, "worst_last10": 0.07}, agree)
        self.assertTrue(v["primary"])
        self.assertFalse(v["consistent"])

    def test_choose_extra_arm(self):
        def rep(flip, passed):
            return {"verdict": {"primary": passed, "flips": {"worst_last9": flip}}}
        arms = {"R0": rep(0.01, True), "R1": rep(0.05, True), "R2": rep(0.03, True), "R3": rep(0.02, False), "R2e": rep(0.001, True)}
        self.assertEqual(st.choose_extra_arm(arms), "R2")        # R0 与替代臂不参选；R3 没过线
        arms["R1"] = rep(0.03, True)
        self.assertEqual(st.choose_extra_arm(arms), "R1")        # 并列取编号小的
        self.assertIsNone(st.choose_extra_arm({"R1": rep(0.1, False), "R2e": rep(0.0, True)}))

    def test_choose_extra_arm_run(self):
        def rep(flip, passed):
            return {"verdict": {"primary": passed, "flips": {"worst_last9": flip}}}
        # 卡里的臂有过线者：取它，即使替代臂的翻转率更低
        self.assertEqual(st.choose_extra_arm_run({"R1": rep(0.05, True), "R2e": rep(0.001, True)}), ("R1", False))
        # 卡里的臂都不过线：在过线的替代臂里取翻转率最小的
        self.assertEqual(st.choose_extra_arm_run({"R1": rep(0.1, False), "R2e": rep(0.008, True), "R3e": rep(0.012, True)}), ("R2e", True))
        self.assertEqual(st.choose_extra_arm_run({"R1": rep(0.1, False), "R2e": rep(0.02, True), "R3e": rep(0.012, True)}), ("R3e", True))
        self.assertEqual(st.choose_extra_arm_run({"R2e": rep(0.01, True), "R3e": rep(0.01, True)}), ("R2e", True))        # 并列取编号小的
        self.assertEqual(st.choose_extra_arm_run({"R2e": rep(0.0, False), "R3e": rep(0.5, False)}), (None, False))      # 替代臂没过线
        self.assertEqual(st.choose_extra_arm_run({"R0": rep(0.0, True)}), (None, False))                                 # R0 不参选

    def test_noise_floor(self):
        a = {1: make_preds(G0[0]), 2: make_preds(G0[1])}
        b = {1: make_preds(G0[0]), 2: make_preds(G0[1])}
        b[2] = b[2].copy()
        b[2][0, 2] = 3                                           # 第 2 轮第 0 帧的预测改掉：1 帧 / 10 帧
        nf = st.noise_floor(a, b)
        np.testing.assert_allclose(nf["per_epoch"], [0.0, 0.1])
        self.assertAlmostEqual(nf["mean"], 0.05)
        self.assertAlmostEqual(nf["max"], 0.1)
        self.assertEqual(nf["zero_epochs"], 1)


class SyntheticArm(unittest.TestCase):
    """三种子 × 三轮的合成臂，全部数字手算。

    原始预测（g > 0 预测 FORWARD=1，否则 SAFE=0）：
      种子 0/2：ep1 = 1×8,0,0   ep2 = 1×6,0×4   ep3 = 1×9,0
      种子 1  ：ep1、ep2 同上   ep3 = 0,1×8,0
    原始翻转：种子 0/2：ep1→ep2 第 6、7 帧 = 0.2；ep2→ep3 第 6、7、8 帧 = 0.3 → 均值 0.25。种子 1：0.2；第 0、6、7、8 帧 = 0.4 → 均值 0.3。
    校准 δ = −（第 8 大的 g）：种子 0/2：[−0.5, +0.4, −0.5]；种子 1：[−0.5, +0.4, −0.2]。校准后预测：
      种子 0/2：三轮都是 1×8,0,0（翻转率 0）；种子 1：ep1、ep2 同，ep3 = 0,1×8,0
    """

    @classmethod
    def setUpClass(cls):
        cls.arm = st.analyse_arm(synth_runs(), Y, TARGET)

    def test_flip_rates(self):
        s = self.arm["seeds"]
        for seed, raw_mean in ((0, 0.25), (1, 0.3), (2, 0.25)):
            self.assertAlmostEqual(s[seed]["flip"]["raw"]["all"], raw_mean)
            self.assertAlmostEqual(s[seed]["flip"]["raw"]["last9"], raw_mean)       # 只有 2 对，末 9 对 = 全部
        np.testing.assert_allclose(s[0]["flip_series_raw"], [0.2, 0.3])
        np.testing.assert_allclose(s[1]["flip_series_raw"], [0.2, 0.4])
        self.assertAlmostEqual(s[0]["flip"]["cal"]["all"], 0.0)
        # 种子 1 校准后 ep2→ep3：ep2 = 1×8,0,0；ep3 = 0,1×8,0 → 第 0、8 帧不同 = 0.2；ep1→ep2 为 0 → 均值 0.1
        self.assertAlmostEqual(s[1]["flip"]["cal"]["all"], 0.1)

    def test_yaw_frame_flips(self):
        # YAW 帧 = 第 8、9 帧。种子 0 原始：预测 (0,0) → (0,0) → (1,0)：0 与 0.5，均值 0.25；校准后全程 (0,0)，翻转 0
        s0 = self.arm["seeds"][0]
        self.assertAlmostEqual(s0["flip_yaw"]["raw"]["all"], 0.25)
        self.assertAlmostEqual(s0["flip_yaw"]["cal"]["all"], 0.0)

    def test_delta(self):
        d0, d1 = self.arm["seeds"][0]["delta"], self.arm["seeds"][1]["delta"]
        np.testing.assert_allclose(d0["per_epoch"], [-0.5, 0.4, -0.5], atol=1e-9)
        np.testing.assert_allclose(d1["per_epoch"], [-0.5, 0.4, -0.2], atol=1e-9)
        self.assertAlmostEqual(d0["median"], -0.5, places=9)
        self.assertAlmostEqual(d0["min"], -0.5, places=9)
        self.assertAlmostEqual(d0["max"], 0.4, places=9)
        self.assertAlmostEqual(d1["last"], -0.2, places=9)
        self.assertAlmostEqual(d0["last10_max"] - d0["last10_min"], 0.9, places=9)   # 三轮都在「末 10 轮」窗口里

    def test_fwd_share(self):
        # 种子 0 原始 FORWARD 预测占比：8/10、6/10、9/10
        np.testing.assert_allclose(self.arm["seeds"][0]["fwd_share_raw"]["per_epoch"], [0.8, 0.6, 0.9])
        self.assertAlmostEqual(self.arm["seeds"][0]["fwd_share_raw"]["min"], 0.6)

    def test_agreement(self):
        g = self.arm["agree"]
        # 原始：ep1、ep2 三者全同 = 1；ep3 只有第 0 帧不一致（1,0,1）= 0.9 → 均值 (1+1+0.9)/3
        np.testing.assert_allclose(g["raw_per_epoch"], [1.0, 1.0, 0.9])
        self.assertAlmostEqual(g["raw_last10"], 2.9 / 3)
        self.assertAlmostEqual(g["raw_last_epoch"], 0.9)
        # 校准：ep3 第 0 帧（1,0,1）与第 8 帧（0,1,0）不一致 = 0.8 → 均值 (1+1+0.8)/3
        np.testing.assert_allclose(g["cal_per_epoch"], [1.0, 1.0, 0.8])
        self.assertAlmostEqual(g["cal_last10"], 2.8 / 3)
        # YAW 帧（第 8、9 帧）：原始三轮都一致 = 1；校准 ep3 第 8 帧不一致 = 0.5 → 均值 (1+1+0.5)/3
        self.assertAlmostEqual(g["raw_yaw_last10"], 1.0)
        self.assertAlmostEqual(g["cal_yaw_last10"], 2.5 / 3)
        # 种子两两翻转（原始）：ep3 的 (0,1)=0.1、(0,2)=0、(1,2)=0.1 → 均值 0.2/3；ep1、ep2 为 0 → 三轮均值 (0.2/3)/3
        self.assertAlmostEqual(g["pair_flip_raw_last10"], 0.2 / 9)

    def test_verdict_fails_on_flip(self):
        v = self.arm["verdict"]
        self.assertAlmostEqual(v["flips"]["worst_last9"], 0.3)       # 三个种子里最大的是种子 1
        self.assertAlmostEqual(v["flips"]["mean_last9"], (0.25 + 0.3 + 0.25) / 3)
        self.assertAlmostEqual(v["flip_all_worst"], 0.3)
        self.assertFalse(v["pass_flip"])
        self.assertTrue(v["pass_agree"])                              # 一致率 (1 + 1 + 0.9) / 3 = 0.967 ≥ 0.88
        self.assertFalse(v["primary"])                                # 翻转率不过线，整体不过线

    def test_stable_arm_passes(self):
        # 三个种子三轮完全相同：翻转率 0、一致率 1，过线
        same = {s: {e + 1: make_preds(G0[0]) for e in range(3)} for s in (0, 1, 2)}
        arm = st.analyse_arm(same, Y, TARGET)
        self.assertAlmostEqual(arm["verdict"]["flips"]["worst_last9"], 0.0)
        self.assertAlmostEqual(arm["agree"]["raw_last10"], 1.0)
        self.assertTrue(arm["verdict"]["primary"])

    def test_averaged_weights_metrics(self):
        # 平均权重的预测给种子 0/2 用 ep3 的表，种子 1 用它自己的 ep3：
        # 种子 0/2 预测 1×9,0：召回 [0,1,0,0]、类 1 的 P=8/9 R=1 → F1 16/17，macro-F1 (16/17)/4，acc 0.8
        # 种子 1 预测 0,1×8,0（第 0 帧、第 8 帧错，第 9 帧错）：类 1 的 tp=7、召回 7/8、预测 8 个 → P=7/8，F1 7/8，macro-F1 (7/8)/4 = 0.21875
        avgs = {s: {"raw": make_preds(GS[s][2]), "cal": make_preds(GS[s][2])} for s in (0, 1, 2)}
        arm = st.analyse_arm(synth_runs(), Y, TARGET, avgs)
        raw = arm["avg"]["raw"]
        self.assertAlmostEqual(raw["macro_f1"][0], (16 / 17) / 4)
        self.assertAlmostEqual(raw["macro_f1"][1], 0.21875)
        self.assertAlmostEqual(raw["macro_f1_mean"], ((16 / 17) / 4 * 2 + 0.21875) / 3)
        np.testing.assert_allclose(raw["recall"][0], [0, 1, 0, 0])
        self.assertAlmostEqual(raw["agree"], 0.9)                       # 第 0 帧（1,0,1）不一致
        self.assertAlmostEqual(arm["agree"]["avg_raw"], 0.9)
        self.assertAlmostEqual(arm["avg"]["raw"]["fwd_share"][0], 0.9)

    def test_dev_ce_summary(self):
        ces = {s: [1.0, 0.8, 0.9] for s in (0, 1, 2)}
        arm = st.analyse_arm(synth_runs(), Y, TARGET, None, ces)
        ce = arm["seeds"][0]["ce"]
        self.assertEqual(ce["min_epoch"], 2)
        self.assertAlmostEqual(ce["min"], 0.8)
        self.assertAlmostEqual(ce["last"], 0.9)
        self.assertAlmostEqual(ce["last10_mean"], 0.9)


class ReportOnDisk(unittest.TestCase):
    """用合成数据落盘，跑 build_report → render_md → 图，核对路径约定、缺臂的处理与 R0 对 R0b 的噪声底。"""

    def test_build_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, out = os.path.join(tmp, "ref"), os.path.join(tmp, "out")

            def put(select_dir, run_dir, runs, ce):
                os.makedirs(select_dir)
                os.makedirs(run_dir, exist_ok=True)
                for e, a in runs.items():
                    np.save(os.path.join(select_dir, f"preds_ep{e:02d}.npy"), a)
                with open(os.path.join(run_dir, "results.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch", "train_ce", "dev_ce", "dev_accuracy"])
                    for i, v in enumerate(ce):
                        w.writerow([i + 1, 1.0, v, 0.5])

            runs = synth_runs()
            for s in (0, 1, 2):
                put(f"{ref}/select/seed{s}", f"{ref}/e1-noise/seed{s}/run", runs[s], [1.0, 0.8, 0.9])
                put(f"{out}/select/R1-s{s}", f"{out}/runs/R1-s{s}", runs[0], [1.0, 0.7, 0.6])
                put(f"{out}/select/R2e-s{s}", f"{out}/runs/R2e-s{s}", runs[0], [1.0, 0.7, 0.6])
                for tag in (f"R0-s{s}", f"R1-s{s}", f"R2e-s{s}"):
                    os.makedirs(f"{out}/avg/{tag}")
                    for k in ("raw", "cal"):
                        np.save(f"{out}/avg/{tag}/preds_avg_{k}.npy", make_preds(GS[s][2]))
            put(f"{out}/select/R0b-s0", f"{out}/runs/R0b-s0", runs[0], [1.0, 0.8, 0.9])
            os.makedirs(f"{out}/blocked")
            with open(f"{out}/blocked/R2.txt", "w", encoding="utf-8") as f:
                f.write("blocked 训练器拒绝 0 权重")
            res = st.build_report(out, ref, list(st.ARM_ORDER))
            self.assertEqual(sorted(res["arms"]), ["R0", "R1", "R2e"])
            self.assertIn("R2", res["missing"])
            self.assertIn("训练器拒绝", res["missing"]["R2"])
            self.assertIn("R3", res["missing"])
            self.assertEqual(res["epochs"], 3)
            self.assertEqual(res["label_counts"], [0, 8, 1, 1])
            self.assertAlmostEqual(res["forward_target"], 0.8)
            self.assertEqual(res["noise_floor"]["flip"]["zero_epochs"], 3)      # R0b 与 R0 种子 0 的预测相同
            self.assertIsNone(res["extra_arm"])                                   # 合成臂的翻转率 0.25，都不过线
            self.assertIsNone(res["extra_arm_card"])
            self.assertFalse(res["extra_arm_is_substitute"])
            md = st.render_md(res)
            for needle in ("R2e（替代）", "未跑或无数据的臂", "过线判决", "同配方同种子两次训练", "原始翻转率分段"):
                self.assertIn(needle, md)
            png = st.make_figures(res, os.path.join(tmp, "fig"))
            self.assertTrue(os.path.getsize(png) > 1000)
            st.write_ce_csv(res, os.path.join(tmp, "ce.csv"))
            with open(os.path.join(tmp, "ce.csv"), encoding="utf-8") as f:
                self.assertEqual(len(f.read().strip().split("\n")), 1 + 3 * 3 * 3)   # 表头 + 3 臂 × 3 种子 × 3 轮
            json.dumps(res)                                                        # 结果可序列化


class SubstituteExtraArm(unittest.TestCase):
    """卡里的臂都没过线、替代臂 R2e 过线（三个种子、各轮预测相同）：第二部分加的臂是 R2e，并标明替代；R1 也过线时取 R1。"""

    @staticmethod
    def build(tmp, stable_arms):
        ref, out = os.path.join(tmp, "ref"), os.path.join(tmp, "out")
        runs = synth_runs()
        for s in (0, 1, 2):
            dirs = [(f"{ref}/select/seed{s}", runs[s])] + [(f"{out}/select/{a}-s{s}", {e + 1: make_preds(G0[0]) for e in range(3)}) for a in stable_arms]
            for d, rr in dirs:
                os.makedirs(d)
                for e, a in rr.items():
                    np.save(os.path.join(d, f"preds_ep{e:02d}.npy"), a)
        return st.build_report(out, ref, ["R0"] + list(stable_arms))

    def test_card_arm_has_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.build(tmp, ["R1", "R2e"])
            self.assertEqual(res["extra_arm_card"], "R1")
            self.assertEqual(res["extra_arm"], "R1")
            self.assertFalse(res["extra_arm_is_substitute"])
            self.assertNotIn("替代臂；卡里没有规定", st.render_md(res))

    def test_substitute_arm_is_added_and_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.build(tmp, ["R2e"])
            self.assertTrue(res["arms"]["R2e"]["verdict"]["primary"])
            self.assertIsNone(res["extra_arm_card"])
            self.assertEqual(res["extra_arm"], "R2e")
            self.assertTrue(res["extra_arm_is_substitute"])
            md = st.render_md(res)
            self.assertIn("卡里的臂中过线者（第二部分加臂的依据；按末 10 轮原始翻转率三种子最大者最小，并列取编号小的）：无", md)
            self.assertIn("第二部分加的臂：R2e（替代臂", md)


REF = "/root/rivermind-data/uav_rgb/exp-student-select-20261109"
H1_LOG = "/root/rivermind-data/uav_rgb/tmp/phase12-cards/review-h/h1_bias_drift.log"


@unittest.skipUnless(os.path.isdir(f"{REF}/select/seed0") and os.path.exists(H1_LOG), "真实数据不在，跳过")
class RealR0(unittest.TestCase):
    def test_reproduces_review_h_log_line_by_line(self):
        P = {s: st.load_epoch_preds(f"{REF}/select/seed{s}") for s in (0, 1, 2)}
        mine = st.h1_lines(P)
        with open(H1_LOG, encoding="utf-8") as f:
            theirs = f.read().rstrip("\n").split("\n")
        self.assertEqual(mine, theirs)

    def test_class_metrics_match_select_ckpt_epochs_csv(self):
        P = st.load_epoch_preds(f"{REF}/select/seed0")
        with open(f"{REF}/select/seed0/epochs.csv", encoding="utf-8") as f:
            rows = {int(r["epoch"]): r for r in csv.DictReader(f)}
        for e in (1, 15, 30):
            a = P[e]
            m = st.class_metrics(a[:, 1].astype(int), a[:, 2].astype(int))
            r = rows[e]
            self.assertAlmostEqual(m["acc"], float(r["acc"]), places=12)
            self.assertAlmostEqual(m["macro_f1"], float(r["macro_f1"]), places=12)
            self.assertAlmostEqual(m["balanced_acc"], float(r["balanced_acc"]), places=12)
            for k, name in enumerate(("recall_safe", "recall_fwd", "recall_l", "recall_r")):
                self.assertAlmostEqual(m["recall"][k], float(r[name]), places=12)


if __name__ == "__main__":
    unittest.main()
