#!/usr/bin/env python3
"""tools/stats_report.py 自检：每个统计量对一个有已知答案的小例子。

  python tools/test_stats_report.py
"""
import json
import math
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats_report as sr  # noqa: E402


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_distributions():
    # 卡方生存函数的教科书临界值（α=0.05）：df=1 3.841、df=2 5.991、df=3 7.815、df=9 16.919。
    for x, df in ((3.841459, 1), (5.991465, 2), (7.814728, 3), (16.918978, 9)):
        assert close(sr.chi2_sf(x, df), 0.05, 2e-5), (x, df, sr.chi2_sf(x, df))
    assert sr.chi2_sf(0.0, 3) == 1.0
    assert close(sr.chi2_sf(2.0, 2), math.exp(-1.0))          # df=2 时 sf = exp(-x/2)
    assert sr.chi2_sf(200.0, 1) < 1e-40
    assert close(sr.norm_ppf(0.975), 1.959964, 1e-6) and close(sr.norm_ppf(0.8), 0.841621, 1e-6)


def test_wilson():
    lo, hi = sr.wilson_ci(19, 47)    # HANDOFF §4.3 / test_eval_metrics 同一个例子
    assert round(lo, 3) == 0.276 and round(hi, 3) == 0.547, (lo, hi)
    assert sr.wilson_ci(0, 0) is None
    assert sr.wilson_ci(0, 10)[0] == 0.0 and sr.wilson_ci(10, 10)[1] == 1.0


def test_mcnemar():
    # 精确二项：b=5, c=1 -> n=6，P[X<=1] = 7/64，双侧 p = 14/64 = 0.21875。
    t = sr.mcnemar(5, 1)
    assert t["method"] == "exact_binomial" and close(t["p_value"], 0.21875), t
    assert sr.mcnemar(3, 3)["p_value"] == 1.0           # 对称：无差异
    assert sr.mcnemar(0, 0) == {"b": 0, "c": 0, "n_discordant": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    # 经典大样本例（Wikipedia McNemar's test）：b=121, c=59 -> (|121-59|-1)^2/180 = 20.672，p ≈ 5.5e-6。
    t = sr.mcnemar(121, 59)
    assert t["method"] == "chi2_cc" and close(t["statistic"], 3721.0 / 180.0), t
    assert t["p_value"] < 1e-5, t
    # 精确二项单侧极端：b=10, c=0 -> p = 2 * 1/1024。
    assert close(sr.mcnemar(10, 0)["p_value"], 2.0 / 1024.0)
    # paired_outcomes 只看共同 seed。
    a = {1: True, 2: True, 3: False, 4: True, 9: True}
    b = {1: True, 2: False, 3: True, 4: False, 8: False}
    assert sr.paired_outcomes(a, b) == (2, 1, 4)


def test_cochran_holm():
    # 手算例：4 受试 × 3 方法，列和 [4,2,1]，行和 [2,1,3,1] -> Q = 2*(3*21-49)/(3*7-15) = 28/6。
    q = sr.cochran_q([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]])
    assert close(q["Q"], 28.0 / 6.0) and q["df"] == 2 and close(q["p_value"], math.exp(-28.0 / 12.0)), q
    # k=2 时 Q 等于不校正的 McNemar 卡方 (b-c)^2/(b+c)：b=3, c=1 -> 4/4 = 1。
    q2 = sr.cochran_q([[1, 0], [1, 0], [1, 0], [0, 1], [1, 1], [0, 0]])
    assert close(q2["Q"], 1.0) and q2["df"] == 1, q2
    assert sr.cochran_q([[1, 1], [0, 0]])["p_value"] == 1.0      # 全同：无差异可检
    # Holm：[0.01, 0.04, 0.03] -> 排序 [0.01, 0.03, 0.04]，乘 [3,2,1] -> [0.03, 0.06, 0.04] 单调化 -> [0.03, 0.06, 0.06]。
    assert [round(x, 6) for x in sr.holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]
    assert sr.holm([0.5, 0.9]) == [1.0, 1.0]
    assert sr.holm([]) == []


def test_iqm_bootstrap():
    # IQM：对称数据等于均值；离群点被截掉；n<4 退化为均值。
    assert sr.iqm(range(1, 9)) == 4.5
    assert sr.iqm([0, 0, 0, 1, 1, 1, 1, 100]) == 0.75 and sum([0, 0, 0, 1, 1, 1, 1, 100]) / 8 == 13.0
    assert sr.iqm([1, 2, 3]) == 2.0
    assert close(sr.iqm([[0.9, 0.5], [0.7, 0.6]]), 0.65)     # 矩阵先拉平：去掉 0.5 与 0.9，剩 0.6/0.7
    # 分层 bootstrap：常数矩阵区间为点；1 个 run 区间退化为点；有变化时 lo <= point <= hi 且可复现。
    assert sr.stratified_bootstrap_ci([[0.5], [0.5], [0.5]]) == (0.5, 0.5, 0.5)
    assert sr.stratified_bootstrap_ci([[0.3, 0.9]]) == (0.6, 0.6, 0.6)
    m = [[0.9, 0.8, 0.7], [0.6, 0.5, 0.8], [0.7, 0.9, 0.4]]
    p1 = sr.stratified_bootstrap_ci(m, n_boot=500, seed=1)
    p2 = sr.stratified_bootstrap_ci(m, n_boot=500, seed=1)
    assert p1 == p2 and p1[1] <= p1[0] <= p1[2] and p1[1] < p1[2], p1
    assert close(p1[0], sr.iqm(m))
    # 非 iqm 统计量走逐样本回退：均值。
    p3 = sr.stratified_bootstrap_ci(m, stat=lambda x: float(x.mean()), n_boot=200, seed=0)
    assert close(p3[0], sum(sum(r) for r in m) / 9) and p3[1] <= p3[0] <= p3[2]
    # 单组均值 bootstrap。
    b = sr.bootstrap_mean_ci([0.2, 0.4, 0.6, 0.8], n_boot=300, seed=0)
    assert close(b[0], 0.5) and b[1] <= 0.5 <= b[2]
    assert sr.bootstrap_mean_ci([]) is None


def test_power():
    # 0.80 vs 0.90，α=0.05，功效 0.8：正态近似 ≈ 199/臂，Fleiss 校正 ≈ 219/臂（ALGO 调研里的 "约 400" 是两臂合计）。
    r = sr.power_two_proportions(0.8, 0.9)
    assert 198 <= r["n_per_arm"] <= 200 and 217 <= r["n_per_arm_corrected"] <= 221, r
    # 差距翻倍，样本量约缩到 1/4。
    r2 = sr.power_two_proportions(0.7, 0.9)
    assert r2["n_per_arm"] < r["n_per_arm"] / 3, r2
    for bad in ((0.5, 0.5), (0.0, 0.5), (0.5, 1.0)):
        try:
            sr.power_two_proportions(*bad)
            assert False, bad
        except ValueError:
            pass


def _fake_eval(path, seeds, success, collision=None, spl=None, **top):
    per = []
    for i, s in enumerate(seeds):
        ep = {"seed": s, "success": bool(success[i]), "collision": bool(collision[i]) if collision else False,
              "false_forward": False, "task_success": False, "stuck": False}
        if spl is not None:
            ep["spl"] = spl[i]
        per.append(ep)
    doc = {"per_episode": per, "episodes_done": len(per), "episodes_requested": len(per) + 1,
           "failed_count": 1, "skipped_count": 0, "hold_s": 1.0, "metric_version": "sim09.v1.coast1s",
           "success_criterion": "route_choice", "dynamics": "cf_sitl", "vehicle": "e1_mini",
           "window_from": "observation", "task": "base", **top}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def test_report_end_to_end():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = list(range(1, 13))
        t = _fake_eval(os.path.join(tmp, "t.json"), seeds, [1] * 11 + [0], spl=[0.9] * 11 + [0.0])
        s = _fake_eval(os.path.join(tmp, "s.json"), seeds, [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                       collision=[0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
        f = _fake_eval(os.path.join(tmp, "f.json"), seeds, [0] * 12, collision=[1] * 12)
        # 第二个场景只有两组，且 forward 多一个 run（训练种子）-> 该场景不做 forward 的配对检验。
        g1 = _fake_eval(os.path.join(tmp, "g1.json"), seeds[:6], [1, 1, 0, 1, 1, 0], hold_s=0.5)
        g2 = _fake_eval(os.path.join(tmp, "g2.json"), seeds[:6], [1, 1, 1, 1, 1, 0], hold_s=0.5)
        g3 = _fake_eval(os.path.join(tmp, "g3.json"), seeds[:6], [1, 0, 1, 1, 0, 0], hold_s=0.5)
        rep = sr.build_report([("teacher", "gazebo", t), ("student", "gazebo", s), ("forward", "gazebo", f),
                               ("teacher", "3dgs", g1), ("forward", "3dgs", g2), ("forward", "3dgs", g3)],
                              n_boot=300, seed=0, title="单测")
        c = rep["cells"]["student"]["gazebo"]
        assert c["SR"]["point"] == 0.5 and c["SR"]["n"] == 12 and c["SR"]["kind"] == "wilson", c
        assert c["CR"]["point"] == 0.5 and c["SPL"] is None, c          # 该组 eval.json 没有 spl -> 空格
        tc = rep["cells"]["teacher"]["gazebo"]
        assert close(tc["SPL"]["point"], 0.9 * 11 / 12) and tc["SPL"]["kind"] == "bootstrap_mean", tc
        # 3dgs 场景 forward 两个 run -> IQM 分层 bootstrap；teacher 单 run。
        fc = rep["cells"]["forward"]["3dgs"]["SR"]
        assert fc["kind"] == "iqm_stratified" and fc["runs"] == 2 and fc["n"] == 6, fc
        assert close(fc["point"], (5 / 6 + 3 / 6) / 2), fc
        # 全场景 IQM：teacher 两场景各 1 run -> 矩阵 1×2，区间退化为点 = 两场景均值。
        allt = rep["all_scenes"]["teacher"]["SR"]
        assert close(allt["point"], (11 / 12 + 4 / 6) / 2) and allt["lo"] == allt["hi"], allt
        assert "forward" not in rep["all_scenes"] or rep["all_scenes"]["forward"]["SR"] is None  # run 数不齐
        # 配对检验：gazebo 三组 -> Cochran's Q + 3 对 McNemar + Holm；3dgs 只剩 teacher 单 run -> 不做。
        pg = rep["paired"]["gazebo"]["SR"]
        assert pg["cochran_q"]["k"] == 3 and pg["cochran_q"]["n"] == 12 and pg["cochran_q"]["p_value"] < 0.01, pg
        assert len(pg["pairwise"]) == 3 and all("p_holm" in r for r in pg["pairwise"])
        ts = next(r for r in pg["pairwise"] if r["A"] == "teacher" and r["B"] == "student")
        assert (ts["b"], ts["c"], ts["n_common"]) == (6, 1, 12) and ts["method"] == "exact_binomial", ts
        assert "3dgs" not in rep["paired"], rep["paired"].keys()
        assert rep["protocol"]["gazebo"]["inconsistent"] == []
        assert rep["protocol"]["3dgs"]["values"]["hold_s"] == ["0.5"]
        # Markdown：每格 `x.xx [lo, hi] (n)`，多 run 为 (2×6)。
        md = sr.to_markdown(rep)
        assert re.search(r"\| student \| 0\.50 \[0\.2\d, 0\.7\d\] \(12\)", md), md
        assert "(2×6)" in md and "Cochran's Q" in md and "p Holm" in md, md
        # 落盘 + CLI。
        out = os.path.join(tmp, "out")
        rc = sr.main(["--eval", "teacher", "gazebo", t, "--eval", "student", "gazebo", s, "--out-dir", out,
                      "--n-boot", "100", "--power", "0.8", "0.9", "--metrics", "SR,CR,SPL"])
        assert rc == 0 and os.path.isfile(os.path.join(out, "stats_report.md"))
        saved = json.load(open(os.path.join(out, "stats_report.json"), encoding="utf-8"))
        assert saved["power"]["n_per_arm"] == rep.get("power", saved["power"])["n_per_arm"]
        assert saved["metrics"] == ["SR", "CR", "SPL"]
        # 目录形式的路径也接受。
        assert sr.load_eval(os.path.dirname(t) if False else t)["episodes_done"] == 12
        try:
            sr.build_report([("a", "x", t)], metrics=("nope",))
            assert False
        except ValueError:
            pass


def main():
    for fn in (test_distributions, test_wilson, test_mcnemar, test_cochran_holm, test_iqm_bootstrap,
               test_power, test_report_end_to_end):
        fn()
        print("ok", fn.__name__)
    print("test_stats_report: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
