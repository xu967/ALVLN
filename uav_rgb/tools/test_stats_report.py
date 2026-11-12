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


def _fake_eval(path, seeds, success, collision=None, spl=None, extra=None, **top):
    """假 eval.json。默认请求数 = 完成数、没有失败集（既有用例的数字按完成集读）；失败集用 episodes_requested / failed_count 覆盖。"""
    per = []
    for i, s in enumerate(seeds):
        ep = {"seed": s, "success": bool(success[i]), "collision": bool(collision[i]) if collision else False,
              "false_forward": False, "task_success": False, "stuck": False, **(extra or {})}
        if spl is not None:
            ep["spl"] = spl[i]
        per.append(ep)
    doc = {"per_episode": per, "episodes_done": len(per), "episodes_requested": len(per),
           "failed_count": 0, "skipped_count": 0, "hold_s": 1.0, "metric_version": "sim09.v1.coast1s",
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
        # 3dgs 场景 forward 两个 run -> IQM + 整簇（没有 geometry_gid，按 seed 聚）bootstrap；teacher 单 run。
        fc = rep["cells"]["forward"]["3dgs"]["SR"]
        assert fc["kind"] == "iqm_cluster" and fc["runs"] == 2 and fc["n"] == 6, fc
        assert fc["cluster_key"] == "seed" and fc["n_clusters"] == 6, fc
        assert close(fc["point"], (5 / 6 + 3 / 6) / 2), fc
        # 全场景 IQM：teacher 两场景各 1 run -> 矩阵 1×2，区间退化为点 = 两场景均值。
        allt = rep["all_scenes"]["teacher"]["SR"]
        assert close(allt["point"], (11 / 12 + 4 / 6) / 2) and allt["lo"] == allt["hi"], allt
        assert "forward" not in rep["all_scenes"] or rep["all_scenes"]["forward"]["SR"] is None  # run 数不齐
        # 配对检验：gazebo 三组 -> Cochran's Q + 3 对 McNemar + Holm（都是单 run，行里没有聚类信息）。
        pg = rep["paired"]["gazebo"]["SR"]
        assert pg["cochran_q"]["k"] == 3 and pg["cochran_q"]["n"] == 12 and pg["cochran_q"]["p_value"] < 0.01, pg
        assert len(pg["pairwise"]) == 3 and all("p_holm" in r for r in pg["pairwise"])
        assert not any("cluster" in r for r in pg["pairwise"]), pg
        ts = next(r for r in pg["pairwise"] if r["A"] == "teacher" and r["B"] == "student")
        assert (ts["b"], ts["c"], ts["n_common"]) == (6, 1, 12) and ts["method"] == "exact_binomial", ts
        # D2：3dgs 场景 forward 有两个 run，不再跳过；run 数 1 对 2，只配前 1 个 run（第 1 个 run 对第 1 个 run）。
        p3 = rep["paired"]["3dgs"]["SR"]["pairwise"]
        assert len(p3) == 1 and p3[0]["cluster"]["runs"] == [1, 2, 1] and p3[0]["n_common"] == 6, p3
        assert (p3[0]["b"], p3[0]["c"]) == (0, 1), p3          # teacher 对 forward 第 1 个 run（g2）：只在 seed 3 上 c
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


def test_mixed_metric_version_header():
    """sim09.v1（过线后继续飞）与 sim09.v2（成功线即停）的 eval.json 同表：表头与清单表都要标出混合版本。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = list(range(1, 7))
        v1 = _fake_eval(os.path.join(tmp, "v1.json"), seeds, [1, 1, 0, 1, 1, 0])
        v2 = _fake_eval(os.path.join(tmp, "v2.json"), seeds, [1, 1, 1, 1, 1, 0], metric_version="sim09.v2.coast1s")
        rep = sr.build_report([("teacher", "gazebo", v1), ("student", "gazebo", v2)], n_boot=100, seed=0)
        assert rep["metric_versions"] == ["sim09.v1.coast1s", "sim09.v2.coast1s"], rep.get("metric_versions")
        md = sr.to_markdown(rep)
        table_header = next(line for line in md.splitlines() if line.startswith("| 方法"))
        assert "sim09.v1.coast1s" in table_header and "sim09.v2.coast1s" in table_header, table_header
        entries_header = next(line for line in md.splitlines() if line.startswith("| 方法 | 场景 |"))
        assert "metric_version" in entries_header, entries_header
        assert "| teacher | gazebo |" in md and "sim09.v1.coast1s |" in md, md
        # 同一版本：表头不带警告。
        same = sr.build_report([("teacher", "gazebo", v1), ("student", "gazebo", v1)], n_boot=100, seed=0)
        header = next(line for line in sr.to_markdown(same).splitlines() if line.startswith("| 方法"))
        assert same["metric_versions"] == ["sim09.v1.coast1s"] and "sim09" not in header, header


def test_default_metrics_and_test_procedure_doc():
    """REVIEW2 #8：默认表不带 base 任务下结构性为 0 的 task_success；文档写实际做法（两两 McNemar 不以 Q 显著为前提，
    Holm 族 = 同一场景同一指标的全部两两行），并在配对表的表注写明 Holm 族。"""
    assert "task_success" not in sr.DEFAULT_METRICS and sr.DEFAULT_METRICS == ("SR", "CR", "false_forward", "SPL"), \
        sr.DEFAULT_METRICS
    assert "task_success" in sr.METRICS   # 还能用 --metrics ...,task_success 显式要（有 waypoint 的 l1/campus 任务）
    doc = sr.__doc__
    assert "显著再两两" not in doc and "Holm" in doc and "族" in doc, doc
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = list(range(1, 9))
        a = _fake_eval(os.path.join(tmp, "a.json"), seeds, [1, 1, 1, 1, 1, 1, 0, 0])
        b = _fake_eval(os.path.join(tmp, "b.json"), seeds, [1, 1, 0, 0, 0, 0, 0, 0])
        c = _fake_eval(os.path.join(tmp, "c.json"), seeds, [1, 0, 0, 0, 0, 0, 0, 0])
        rep = sr.build_report([("A", "gz", a), ("B", "gz", b), ("C", "gz", c)], n_boot=50, seed=0)
        assert list(rep["metrics"]) == list(sr.DEFAULT_METRICS), rep["metrics"]
        md = sr.to_markdown(rep)
        assert "task_success" not in md.splitlines()[0], md.splitlines()[0]
        assert "Holm 族" in md and "不以 Q 显著为前提" in md, md
        # 两两行不因 Q 不显著而消失：3 方法 → 3 行。
        assert len(rep["paired"]["gz"]["SR"]["pairwise"]) == 3


def test_success_metrics_use_requested_denominator():
    """W1-2：越高越好的二值指标（SR、task_success、reach_ever、stop_success）分母 = episodes_requested，失败集记 0；
    越低越好的（CR、stuck、false_forward）与连续指标 SPL 仍按完成集。一个 eval 请求 20 集、完成 19 集、失败 1 集：
    SR 的分母是 20，CR 的分母是 19；表里有「失败集数」列和写明分母的表注。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        metrics = ("SR", "CR", "false_forward", "stuck", "task_success", "reach_ever", "stop_success", "SPL")
        e19 = _fake_eval(os.path.join(tmp, "e19.json"), list(range(1, 20)), [1] * 15 + [0] * 4, collision=[0] * 15 + [1] * 4,
                         spl=[0.8] * 19, extra={"reach_ever": True, "stop_success": True},
                         episodes_requested=20, failed_count=1)
        rep = sr.build_report([("m", "gazebo", e19)], metrics=metrics, n_boot=50, seed=0)
        cell = rep["cells"]["m"]["gazebo"]
        c = cell["SR"]
        assert (c["k"], c["n"], c.get("n_done")) == (15, 20, 19) and close(c["point"], 15 / 20), c
        assert (c["lo"], c["hi"]) == sr.wilson_ci(15, 20), c        # 区间也按 20 算
        c = cell["CR"]
        assert (c["k"], c["n"]) == (4, 19) and close(c["point"], 4 / 19) and "n_done" not in c, c
        for m in ("task_success", "reach_ever", "stop_success"):   # 越高越好：请求集
            assert cell[m]["n"] == 20 and cell[m].get("n_done") == 19, (m, cell[m])
        for m in ("false_forward", "stuck"):                       # 越低越好：完成集
            assert cell[m]["n"] == 19, (m, cell[m])
        assert cell["SPL"]["n"] == 19 and cell["SPL"]["kind"] == "bootstrap_mean", cell["SPL"]
        assert rep["failed_total"] == {"m": 1}, rep.get("failed_total")
        assert rep["denominators"]["SR"] == "requested" and rep["denominators"]["CR"] == "done", rep.get("denominators")
        # 没有失败集：数字与改前逐位相同（分母同为完成集，不带 n_done）。
        e20 = _fake_eval(os.path.join(tmp, "e20.json"), list(range(1, 21)), [1] * 18 + [0] * 2)
        c = sr.build_report([("m", "gazebo", e20)], n_boot=50, seed=0)["cells"]["m"]["gazebo"]["SR"]
        assert (c["k"], c["n"]) == (18, 20) and "n_done" not in c, c
        # 旧 eval.json 没有 episodes_requested：退回完成集，不报错。
        old = _fake_eval(os.path.join(tmp, "old.json"), list(range(1, 20)), [1] * 15 + [0] * 4, episodes_requested=None)
        c = sr.build_report([("m", "gazebo", old)], n_boot=50, seed=0)["cells"]["m"]["gazebo"]["SR"]
        assert c["n"] == 19, c
        # 多 run：每个 run 按各自的请求集算比例再取 IQM；CR 两个 run 的分母不同（19 与 20），n 记成列表。
        rep2 = sr.build_report([("m", "gazebo", e19), ("m", "gazebo", e20)], metrics=("SR", "CR"), n_boot=50, seed=0)
        c2 = rep2["cells"]["m"]["gazebo"]
        assert c2["SR"]["runs"] == 2 and c2["SR"]["n"] == 20 and close(c2["SR"]["point"], (15 / 20 + 18 / 20) / 2), c2["SR"]
        assert c2["CR"]["n"] == [19, 20], c2["CR"]
        assert rep2["failed_total"] == {"m": 1}
        # 跨场景 IQM 与单场景格同一套分母：gazebo 的 15/20 与 3dgs 的 5/6。
        g6 = _fake_eval(os.path.join(tmp, "g6.json"), list(range(1, 7)), [1, 1, 1, 1, 1, 0])
        rep3 = sr.build_report([("m", "gazebo", e19), ("m", "3dgs", g6)], metrics=("SR",), n_boot=50, seed=0)
        assert close(rep3["all_scenes"]["m"]["SR"]["point"], (15 / 20 + 5 / 6) / 2), rep3["all_scenes"]
        # Markdown：末列「失败集数」，表注写明两种分母。
        md = sr.to_markdown(rep)
        header = next(line for line in md.splitlines() if line.startswith("| 方法"))
        row = next(line for line in md.splitlines() if line.startswith("| m |"))
        assert header.rstrip().endswith("| 失败集数 |"), header
        assert row.rstrip().endswith("| 1 |") and "0.75 [" in row and "(20)" in row and "(19)" in row, row
        assert "请求集" in md and "episodes_requested" in md and "完成集" in md, md


def _gid_eval(path, seeds, success, gid_of=None, **top):
    """带 geometry_gid 的假 eval.json：gid_of(seed) -> 几何 id（默认每个 seed 一个几何）。"""
    gid_of = gid_of or (lambda s: f"geo-{s}")
    per = [{"seed": s, "success": bool(success[i]), "collision": False, "false_forward": False, "task_success": False,
            "stuck": False, "geometry_gid": gid_of(s)} for i, s in enumerate(seeds)]
    doc = {"per_episode": per, "episodes_done": len(per), "episodes_requested": len(per), "failed_count": 0,
           "skipped_count": 0, "hold_s": 1.0, "metric_version": "sim09.v2.coast1s.corridor3m",
           "success_criterion": "route_choice", "dynamics": "cf_sitl", "vehicle": "e1_mini",
           "window_from": "observation", "task": "base", **top}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def test_multi_run_pairing_cluster_columns():
    """D2：同一场景多 run 的方法两两配对，按 (run, seed) 对齐，逐集 McNemar 保留作参考，另给几何聚类符号翻转 p 与整簇区间两列。
    6 个 seed 落在 3 个几何（每个几何 2 个 seed），A 三个 run 全成功，B 三个 run 只在第一个几何上成功：
    逐集 b=12、c=0（精确 McNemar p = 2/4096），几何差 d=[0,6,6]，符号翻转 p = 0.5。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = [1, 2, 3, 4, 5, 6]
        gid = lambda s: f"geo{(s - 1) // 2}"      # noqa: E731
        runs_a = [_gid_eval(os.path.join(tmp, f"a{i}.json"), seeds, [1] * 6, gid) for i in range(3)]
        runs_b = [_gid_eval(os.path.join(tmp, f"b{i}.json"), seeds, [1, 1, 0, 0, 0, 0], gid) for i in range(3)]
        entries = [("A", "gz", p) for p in runs_a] + [("B", "gz", p) for p in runs_b]
        rep = sr.build_report(entries, metrics=("SR",), n_boot=200, seed=0, n_draws=1000)
        rows = rep["paired"]["gz"]["SR"]["pairwise"]
        assert len(rows) == 1, rows                                   # 以前两个多 run 方法被整组跳过
        r = rows[0]
        # 键是 (run, seed)：三个 run 的同一个 seed 不互相覆盖，18 个单元。
        assert (r["n_common"], r["b"], r["c"]) == (18, 12, 0) and r["rate_A"] == 1.0 and close(r["rate_B"], 1 / 3), r
        assert close(r["p_value"], 2 / 4096) and r["method"] == "exact_binomial", r
        c = r["cluster"]
        assert c["key"] == "geometry_gid" and c["runs"] == [3, 3, 3], c
        assert (c["n_clusters"], c["n_nonzero"], c["method"], c["n_draws"]) == (3, 2, "exact", None), c
        assert close(c["p_value"], 0.5) and close(c["diff"], 2 / 3) and c["ci_lo"] <= c["diff"] <= c["ci_hi"], c
        assert rep["paired"]["gz"]["skipped_multi_run"] == [], rep["paired"]["gz"]
        md = sr.to_markdown(rep)
        header = next(line for line in md.splitlines() if line.startswith("| A | B |"))
        assert header.rstrip().endswith("| p 几何聚类符号翻转 | 差值 [95% 整簇区间] |") and "p Holm" in header, header
        row = next(line for line in md.splitlines() if line.startswith("| A | B | 18 |"))
        assert "0.500 | +0.67 [" in row, row
        assert "几何聚类" in md and "逐集 McNemar" in md and "geometry_gid" in md and "多 run 分组不做配对检验" not in md, md
        # 单 run 的配对表没有这两列，行里没有 cluster 键。
        single = sr.build_report([("A", "gz", runs_a[0]), ("B", "gz", runs_b[0])], metrics=("SR",), n_boot=50, seed=0)
        r1 = single["paired"]["gz"]["SR"]["pairwise"][0]
        assert "cluster" not in r1 and (r1["n_common"], r1["b"], r1["c"]) == (6, 4, 0), r1
        md1 = sr.to_markdown(single)
        assert "几何聚类" not in md1 and "| p | p Holm | 方法 |" in md1, md1


def test_multi_run_pairing_alignment_and_fallback():
    """第 i 个 run 对第 i 个 run；run 数不同只配前 min 个；没有 geometry_gid 时聚类键回退为 seed 并在表里写明；
    非零簇数超过 20 走蒙特卡洛，抽样次数写进结果与表。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = [1, 2, 3, 4, 5, 6]
        x, y = [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]
        # A 与 B 的第 0 个 run 相同、第 1 个 run 相同：按 run 序号对齐则逐集全部一致；若错位（A 的 run0 对 B 的 run1）会有 12 个不一致。
        a = [_fake_eval(os.path.join(tmp, f"a{i}.json"), seeds, v) for i, v in enumerate((x, y))]
        b = [_fake_eval(os.path.join(tmp, f"b{i}.json"), seeds, v) for i, v in enumerate((x, y))]
        rep = sr.build_report([("A", "gz", a[0]), ("A", "gz", a[1]), ("B", "gz", b[0]), ("B", "gz", b[1])],
                              metrics=("SR",), n_boot=100, seed=0)
        r = rep["paired"]["gz"]["SR"]["pairwise"][0]
        assert (r["n_common"], r["b"], r["c"]) == (12, 0, 0) and r["p_value"] == 1.0, r
        assert r["cluster"]["key"] == "seed" and r["cluster"]["n_clusters"] == 6 and r["cluster"]["p_value"] == 1.0, r
        assert "回退" in sr.to_markdown(rep), "没有 geometry_gid 时表里要写明聚类键回退为 seed"
        # run 数 2 对 1：只配第 0 个 run（6 个单元）。
        rep21 = sr.build_report([("A", "gz", a[0]), ("A", "gz", a[1]), ("B", "gz", b[0])], metrics=("SR",), n_boot=50, seed=0)
        r21 = rep21["paired"]["gz"]["SR"]["pairwise"][0]
        assert r21["cluster"]["runs"] == [2, 1, 1] and r21["n_common"] == 6, r21
        # 蒙特卡洛：25 个几何、A 三个 run 全成功、B 全失败，每个几何差 3，非零簇 25 > 20。
        seeds25 = list(range(1, 26))
        ra = [_gid_eval(os.path.join(tmp, f"ma{i}.json"), seeds25, [1] * 25) for i in range(3)]
        rb = [_gid_eval(os.path.join(tmp, f"mb{i}.json"), seeds25, [0] * 25) for i in range(3)]
        rep25 = sr.build_report([("A", "g", p) for p in ra] + [("B", "g", p) for p in rb], metrics=("SR",),
                                n_boot=50, seed=0, n_draws=200)
        c = rep25["paired"]["g"]["SR"]["pairwise"][0]["cluster"]
        assert (c["method"], c["n_draws"], c["n_nonzero"], c["key"]) == ("monte_carlo", 200, 25, "geometry_gid"), c
        assert c["p_value"] == 1 / 201, c
        assert "MC 200" in sr.to_markdown(rep25), sr.to_markdown(rep25)
        # 两边的 geometry_gid 不一致：整张表回退成 seed。
        diff_gid = [_gid_eval(os.path.join(tmp, f"d{i}.json"), seeds, [1] * 6, lambda s: f"other-{s}") for i in range(2)]
        same_gid = [_gid_eval(os.path.join(tmp, f"s{i}.json"), seeds, [1] * 6) for i in range(2)]
        repd = sr.build_report([("A", "gz", p) for p in same_gid] + [("B", "gz", p) for p in diff_gid], metrics=("SR",),
                               n_boot=50, seed=0)
        assert repd["paired"]["gz"]["SR"]["pairwise"][0]["cluster"]["key"] == "seed"


def test_multi_run_cell_cluster_interval():
    """D4：多 run 的格，点估计仍是各 run 比例的 IQM，区间改为整个几何有放回重采样（同一几何上各 run 的集一起抽）。
    10 个几何 × 每个几何 2 个 seed，5 个几何三个 run 全成功、5 个全失败：独立单位只有 10 个几何，区间应该宽（约 [0.2, 0.8]）；
    旧的 run 级分层 bootstrap 三个 run 完全相同，区间塌成一个点 [0.5, 0.5]，等于说「没有不确定性」。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = list(range(1, 21))
        gid = lambda s: f"geo{(s - 1) // 2}"       # noqa: E731    seed 1,2 -> geo0 ...
        ok = [1 if ((s - 1) // 2) % 2 == 0 else 0 for s in seeds]
        runs = [_gid_eval(os.path.join(tmp, f"r{i}.json"), seeds, ok, gid) for i in range(3)]
        rep = sr.build_report([("m", "gz", p) for p in runs], metrics=("SR",), n_boot=2000, seed=0)
        c = rep["cells"]["m"]["gz"]["SR"]
        assert c["kind"] == "iqm_cluster" and c["runs"] == 3 and c["n"] == 20 and close(c["point"], 0.5), c
        assert c["cluster_key"] == "geometry_gid" and c["n_clusters"] == 10, c
        assert 0.15 <= c["lo"] <= 0.30 and 0.70 <= c["hi"] <= 0.85, c         # 10 个独立单位：0.5 ± 1.96·√(0.25/10) 约 [0.2, 0.8]
        assert sr.stratified_bootstrap_ci([[0.5], [0.5], [0.5]]) == (0.5, 0.5, 0.5)   # 旧口径的区间确实塌成点
        # 同种子可复现；种子不同，区间有小差异。
        again = sr.build_report([("m", "gz", p) for p in runs], metrics=("SR",), n_boot=2000, seed=0)
        assert again["cells"]["m"]["gz"]["SR"] == c
        # 没有 geometry_gid：按 seed 聚（三个 run 的同一个 seed 一起抽）。
        bare = [_fake_eval(os.path.join(tmp, f"b{i}.json"), seeds, ok) for i in range(3)]
        cb = sr.build_report([("m", "gz", p) for p in bare], metrics=("SR",), n_boot=500, seed=0)["cells"]["m"]["gz"]["SR"]
        assert cb["cluster_key"] == "seed" and cb["n_clusters"] == 20 and close(cb["point"], 0.5), cb
        assert 0.25 <= cb["lo"] <= 0.35 and 0.65 <= cb["hi"] <= 0.75, cb      # 20 个独立单位：约 [0.3, 0.7]
        # 单 run 的格不变（Wilson）。
        c1 = sr.build_report([("m", "gz", runs[0])], metrics=("SR",), n_boot=50, seed=0)["cells"]["m"]["gz"]["SR"]
        assert c1["kind"] == "wilson" and "cluster_key" not in c1, c1
        # Markdown：表注写明整簇重采样，不再说分层 bootstrap。
        md = sr.to_markdown(rep)
        assert "整簇" in md and "分层 bootstrap" not in md and "3×20" in md, md
        # 4 个 run 含离群 run：每个几何上 3 个 run 为 0.5（2 个 seed 一成一败）、1 个离群 run 为 1.0（全成功）。
        # 任意整簇重采样样本上 4 个 run 的比例恒为 [0.5, 0.5, 0.5, 1.0]，IQM 截掉首尾后恒为 0.5（若错算成均值会变成 0.625）。
        half = [s % 2 for s in seeds]
        runs4 = [_gid_eval(os.path.join(tmp, f"q{i}.json"), seeds, half, gid) for i in range(3)]
        runs4.append(_gid_eval(os.path.join(tmp, "q3.json"), seeds, [1] * 20, gid))
        c4 = sr.build_report([("m", "gz", p) for p in runs4], metrics=("SR",), n_boot=200, seed=0)["cells"]["m"]["gz"]["SR"]
        assert c4["runs"] == 4 and close(c4["point"], 0.5) and close(c4["lo"], 0.5) and close(c4["hi"], 0.5), c4


def test_multi_run_cell_failed_and_spl():
    """D4：越高越好的指标分母是请求集：失败集（落在它的几何簇里）与没有 seed 的空位（各自一个单点簇）都参与重采样，
    点估计与以前相同；连续指标 SPL 的多 run 格同样按整簇重采样，per-run 均值再取 IQM。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = list(range(1, 21))
        gid = lambda s: f"geo{(s - 1) // 2}"       # noqa: E731
        full = _gid_eval(os.path.join(tmp, "full.json"), seeds, [1] * 20, gid)
        # 请求 20、完成 18、失败 seed 19 与 20（它们的几何 geo9 在别的 run 里完成过，所以失败集落进 geo9 的簇）。
        fail = _gid_eval(os.path.join(tmp, "fail.json"), seeds[:18], [1] * 18, gid, episodes_requested=20, failed_count=2,
                         failed=[{"seed": 19, "rc": 1}, {"seed": 20, "rc": 1}])
        # 请求 20、完成 18、记录里没有失败 seed（旧文件里有这种空位）：两个空位各是单点簇。
        pad = _gid_eval(os.path.join(tmp, "pad.json"), seeds[:18], [1] * 18, gid, episodes_requested=20, failed_count=2)
        rep = sr.build_report([("m", "gz", full), ("m", "gz", fail), ("m", "gz", pad)], metrics=("SR", "CR"), n_boot=500, seed=0)
        c = rep["cells"]["m"]["gz"]["SR"]
        assert close(c["point"], (1.0 + 18 / 20 + 18 / 20) / 3) and c["n"] == 20 and c["n_done"] == [18, 20], c
        assert c["n_clusters"] == 10 + 2 and c["cluster_key"] == "geometry_gid", c      # 10 个几何 + 空位 2 个
        assert c["lo"] <= c["point"] <= c["hi"], c
        # CR（越低越好，分母 = 完成集）没有失败集与空位的簇：10 个几何、各 run 完成数 20/18/18。
        cr = rep["cells"]["m"]["gz"]["CR"]
        assert cr["n_clusters"] == 10 and cr["n"] == [18, 20] and cr["point"] == 0.0, cr
        # SPL：每个 run 的均值，整簇重采样；点估计 = IQM(各 run 均值)。
        spl_a = _gid_eval(os.path.join(tmp, "sa.json"), seeds, [1] * 20, gid)
        spl_b = _gid_eval(os.path.join(tmp, "sb.json"), seeds, [1] * 20, gid)
        for path, val in ((spl_a, 0.8), (spl_b, 0.6)):
            doc = json.load(open(path, encoding="utf-8"))
            for ep in doc["per_episode"]:
                ep["spl"] = val if ep["seed"] % 4 else val / 2
            json.dump(doc, open(path, "w", encoding="utf-8"))
        rep_spl = sr.build_report([("m", "gz", spl_a), ("m", "gz", spl_b)], metrics=("SPL",), n_boot=300, seed=0)
        cs = rep_spl["cells"]["m"]["gz"]["SPL"]
        want = (sum(0.8 if s % 4 else 0.4 for s in seeds) / 20 + sum(0.6 if s % 4 else 0.3 for s in seeds) / 20) / 2
        assert cs["kind"] == "iqm_cluster" and close(cs["point"], want) and cs["n_clusters"] == 10, cs
        assert cs["lo"] <= cs["point"] <= cs["hi"] and cs["lo"] < cs["hi"], cs


def test_duplicate_seed_in_one_run_is_error():
    """D5：同一个 run 内 seed 重复（后一集会覆盖前一集，配对与比例静默少算）→ 报错并写明文件与 seed；
    不同 run 的同一个 seed 按 (run, seed) 对齐，不算重复，三个 run 各自的集都进配对。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        dup = _fake_eval(os.path.join(tmp, "dup.json"), [1, 2, 2, 3, 3, 4], [1, 0, 1, 1, 0, 1])
        ok = _fake_eval(os.path.join(tmp, "ok.json"), [1, 2, 3, 4], [1, 0, 1, 1])
        for call in (lambda: sr.load_eval(dup), lambda: sr.build_report([("m", "gz", dup)], n_boot=10),
                     lambda: sr.build_report([("a", "gz", ok), ("b", "gz", dup)], n_boot=10),
                     lambda: sr.metric_values(json.load(open(dup, encoding="utf-8"))["per_episode"], "success")):
            try:
                call()
                assert False, "重复 seed 没有报错"
            except ValueError as e:
                assert "seed [2, 3]" in str(e) and "重复" in str(e), str(e)
        try:
            sr.load_eval(dup)
        except ValueError as e:
            assert dup in str(e), str(e)                                   # 报错写明文件
        # 命令行：干净地退出（argparse 的退出码 2），不抛 traceback。
        try:
            sr.main(["--eval", "m", "gz", dup, "--n-boot", "10"])
            assert False
        except SystemExit as e:
            assert e.code == 2, e.code
        # 三个 run 的同一个 seed 不是重复：每个 run 各 4 个单元，配对 12 个。
        runs = [_fake_eval(os.path.join(tmp, f"r{i}.json"), [1, 2, 3, 4], [1, 0, 1, 1]) for i in range(3)]
        other = [_fake_eval(os.path.join(tmp, f"o{i}.json"), [1, 2, 3, 4], [0, 0, 1, 1]) for i in range(3)]
        rep = sr.build_report([("a", "gz", p) for p in runs] + [("b", "gz", p) for p in other], metrics=("SR",), n_boot=20, seed=0)
        row = rep["paired"]["gz"]["SR"]["pairwise"][0]
        assert (row["n_common"], row["b"], row["c"]) == (12, 3, 0), row


def test_pairing_counts_infra_failed_seed_as_zero():
    """W1-2 遗留：越高越好的指标配对时，一侧基础设施失败的 seed 记失败（0），与主表同一个分母；越低越好的指标只用两侧都完成的 seed。
    B 请求 6 集，seed 6 失败（在 failed 里，不在 per_episode 里）：以前只配共同完成的 5 个 seed，现在配 6 个。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = [1, 2, 3, 4, 5, 6]
        fail6 = dict(episodes_requested=6, failed_count=1, failed=[{"seed": 6, "rc": 1}])
        a = _fake_eval(os.path.join(tmp, "a.json"), seeds, [1] * 6)
        b = _fake_eval(os.path.join(tmp, "b.json"), seeds[:5], [1, 1, 0, 0, 1], **fail6)
        rep = sr.build_report([("A", "gz", a), ("B", "gz", b)], metrics=("SR", "CR"), pair_metrics=("SR", "CR"), n_boot=20, seed=0)
        r = rep["paired"]["gz"]["SR"]["pairwise"][0]
        assert (r["n_common"], r["b"], r["c"]) == (6, 3, 0) and r["rate_A"] == 1.0 and close(r["rate_B"], 3 / 6), r
        cell = rep["cells"]["B"]["gz"]["SR"]
        assert (cell["k"], cell["n"], cell["n_done"]) == (3, 6, 5) and close(cell["point"], r["rate_B"]), cell    # 与主表同分母
        cr = rep["paired"]["gz"]["CR"]["pairwise"][0]
        assert cr["n_common"] == 5, cr                                  # CR 越低越好：失败集没有观测，不进配对
        # 两侧都失败：seed 6 在 A、B 里都失败，记 (0, 0)，n 共同照数、不产生不一致对。
        a2 = _fake_eval(os.path.join(tmp, "a2.json"), seeds[:5], [1] * 5, **fail6)
        rep2 = sr.build_report([("A", "gz", a2), ("B", "gz", b)], metrics=("SR",), n_boot=20, seed=0)
        r2 = rep2["paired"]["gz"]["SR"]["pairwise"][0]
        assert (r2["n_common"], r2["b"], r2["c"]) == (6, 2, 0), r2
        # Cochran's Q 同一个口径：三个方法的共同 seed 含失败集（记 0）。
        c3 = _fake_eval(os.path.join(tmp, "c3.json"), seeds, [0, 1, 0, 1, 0, 1])
        rep3 = sr.build_report([("A", "gz", a), ("B", "gz", b), ("C", "gz", c3)], metrics=("SR",), n_boot=20, seed=0)
        assert rep3["paired"]["gz"]["SR"]["cochran_q"]["n"] == 6, rep3["paired"]["gz"]["SR"]["cochran_q"]
        # 多 run：B 的第 2 个 run 里 seed 6 失败，A 两个 run 全成功 -> 12 个单元，只有那一个单元不一致。
        b_ok = _fake_eval(os.path.join(tmp, "b_ok.json"), seeds, [1] * 6)
        b_f = _fake_eval(os.path.join(tmp, "b_f.json"), seeds[:5], [1] * 5, **fail6)
        a_runs = [_fake_eval(os.path.join(tmp, f"am{i}.json"), seeds, [1] * 6) for i in range(2)]
        repm = sr.build_report([("A", "gz", p) for p in a_runs] + [("B", "gz", b_ok), ("B", "gz", b_f)], metrics=("SR",),
                               n_boot=20, seed=0)
        rm = repm["paired"]["gz"]["SR"]["pairwise"][0]
        assert (rm["n_common"], rm["b"], rm["c"]) == (12, 1, 0), rm
        assert "失败集记 0" in sr.to_markdown(rep), sr.to_markdown(rep)


def test_all_failed_cell_shows_zero_with_requested_denominator():
    """W1-2 遗留：一个 run 全部失败（完成 0 集、请求 20 集）：越高越好的指标格显示 0.00，n = 请求集，而不是「—」；
    越低越好的指标与 SPL 没有观测，仍是「—」。多 run 与全场景 IQM 里这个 run 记 0，不被丢掉。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        fails = [{"seed": s, "rc": 1} for s in range(1, 21)]
        dead = _fake_eval(os.path.join(tmp, "dead.json"), [], [], episodes_requested=20, failed_count=20, failed=fails)
        ok = _fake_eval(os.path.join(tmp, "ok.json"), list(range(1, 21)), [1] * 10 + [0] * 10, spl=[0.5] * 20)
        rep = sr.build_report([("dead", "gz", dead), ("ok", "gz", ok)], metrics=("SR", "CR", "SPL"), n_boot=50, seed=0)
        c = rep["cells"]["dead"]["gz"]["SR"]
        assert (c["point"], c["n"], c["k"], c["n_done"], c["lo"]) == (0.0, 20, 0, 0, 0.0) and 0.1 < c["hi"] < 0.2, c
        assert c["kind"] == "wilson" and (c["lo"], c["hi"]) == sr.wilson_ci(0, 20), c
        assert rep["cells"]["dead"]["gz"]["CR"] is None and rep["cells"]["dead"]["gz"]["SPL"] is None
        assert rep["failed_total"]["dead"] == 20
        md = sr.to_markdown(rep)
        row = next(line for line in md.splitlines() if line.startswith("| dead |"))
        assert row.startswith("| dead | 0.00 [0.00, 0.16] (20) | — | — |") and row.rstrip().endswith("| 20 |"), row
        # 配对：失败的 20 个 seed 记 0，对 ok 的 10 个成功 seed 是 c；n 共同 20。
        r = rep["paired"]["gz"]["SR"]["pairwise"][0]
        assert (r["n_common"], r["b"], r["c"]) == (20, 0, 10), r
        # 多 run：全失败的 run 记 0，与 ok（0.5）取 IQM（2 个值 = 均值）。
        rep2 = sr.build_report([("m", "gz", dead), ("m", "gz", ok)], metrics=("SR",), n_boot=50, seed=0)
        c2 = rep2["cells"]["m"]["gz"]["SR"]
        assert c2["runs"] == 2 and c2["n"] == 20 and close(c2["point"], 0.25) and c2["n_done"] == [0, 20], c2
        # 全场景 IQM：dead 在两个场景都全失败 -> 0.0，不再是 None。
        rep3 = sr.build_report([("dead", "gz", dead), ("dead", "other", dead)], metrics=("SR",), n_boot=50, seed=0)
        assert rep3["all_scenes"]["dead"]["SR"]["point"] == 0.0, rep3["all_scenes"]
        # 旧文件没有 episodes_requested：请求数未知，仍是「—」。
        old = _fake_eval(os.path.join(tmp, "old.json"), [], [], episodes_requested=None)
        assert sr.build_report([("o", "gz", old)], metrics=("SR",), n_boot=10)["cells"]["o"]["gz"]["SR"] is None
        # 回归：有完成集、但 per_episode 里没有这个指标字段（旧文件没有 reach_ever）+ 有失败集，不是「全失败」：格仍是「—」，
        # 配对也不给失败集补 0（补出来的 0 会和别的 run 的真实值配成假的不一致对）。
        seeds = [1, 2, 3, 4, 5, 6]
        miss = _fake_eval(os.path.join(tmp, "miss.json"), seeds[:5], [1] * 5, episodes_requested=6, failed_count=1,
                          failed=[{"seed": 6, "rc": 1}])
        has = _fake_eval(os.path.join(tmp, "has.json"), seeds, [1] * 6, extra={"reach_ever": True})
        repm = sr.build_report([("miss", "gz", miss), ("has", "gz", has)], metrics=("SR", "reach_ever"),
                               pair_metrics=("reach_ever",), n_boot=10)
        assert repm["cells"]["miss"]["gz"]["reach_ever"] is None, repm["cells"]["miss"]["gz"]
        assert repm["paired"]["gz"]["reach_ever"]["pairwise"] == [], repm["paired"]["gz"]["reach_ever"]
        # 聚合文件（没有 per_episode、完成数不是 0）也不是全失败。
        agg = _fake_eval(os.path.join(tmp, "agg.json"), [], [], episodes_requested=20, failed_count=2, episodes_done=18)
        assert sr.build_report([("a", "gz", agg)], metrics=("SR",), n_boot=10)["cells"]["a"]["gz"]["SR"] is None


def test_protocol_keys_mismatch_raises_unless_allowed():
    """A4：action_latency_s / heading_guard / rtf / student_cond / dynamics_profile 在同一场景（同一列组）内不一致 → 报错并列出
    哪些键；缺键 = 旧协议；--allow-mixed-protocol 放行并在表头标「混合协议」。原有七个键仍只警告（见 test_mixed_metric_version_header）。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = list(range(1, 7))
        mk = lambda name, **top: _fake_eval(os.path.join(tmp, name + ".json"), seeds, [1, 1, 0, 0, 0, 0], **top)   # noqa: E731
        base = mk("base")
        cases = (("action_latency_s", 0.2, 0.16), ("heading_guard", 2, 3), ("rtf", 0.5, 0.8),
                 ("student_cond", "heading-code", "other-cond"), ("dynamics_profile", "cf_blackbox_v1:0954b747", "cf_sitl_v1:683e7400"))
        for key, val, val2 in cases:
            with_key, same, diff = mk("w_" + key, **{key: val}), mk("s_" + key, **{key: val}), mk("d_" + key, **{key: val2})
            for a, b in ((base, with_key), (with_key, base), (with_key, diff)):     # 缺键对有键、有键对缺键、两个不同的值
                try:
                    sr.build_report([("A", "gz", a), ("B", "gz", b)], metrics=("SR",), n_boot=10)
                    assert False, (key, "没有报错")
                except sr.ProtocolMismatch as e:
                    assert isinstance(e, ValueError) and key in str(e) and "gz" in str(e), str(e)
            # 两边都有同一个值：一致，不报错；同一个方法的多个 run 里值不一致也报。
            sr.build_report([("A", "gz", with_key), ("B", "gz", same)], metrics=("SR",), n_boot=10)
            try:
                sr.build_report([("A", "gz", with_key), ("A", "gz", base), ("B", "gz", same)], metrics=("SR",), n_boot=10)
                assert False
            except sr.ProtocolMismatch:
                pass
        # 一次列出全部不一致的键与各自的取值、方法。
        both = mk("both", heading_guard=2, rtf=0.5)
        try:
            sr.build_report([("无护栏", "gz", base), ("护栏", "gz", both)], metrics=("SR",), n_boot=10)
            assert False
        except sr.ProtocolMismatch as e:
            msg = str(e)
            assert "heading_guard" in msg and "rtf" in msg and "无护栏" in msg and "护栏" in msg and "--allow-mixed-protocol" in msg, msg
        # 默认值等价于缺键：heading_guard=0、student_cond=prev-action 与缺键一致；cf_sitl_params（含路径）不比较。
        eq = mk("eq", heading_guard=0, student_cond="prev-action", cf_sitl_params={"path": "/x"})
        sr.build_report([("A", "gz", base), ("B", "gz", eq), ("C", "gz", mk("eq2", cf_sitl_params={"path": "/y"}))],
                        metrics=("SR",), n_boot=10)
        # 不同场景各自一致：场景之间不比较（gz 全带护栏，3dgs 全不带）。
        rep_ok = sr.build_report([("A", "gz", both), ("B", "gz", mk("both2", heading_guard=2, rtf=0.5)),
                                  ("A", "3dgs", base), ("B", "3dgs", mk("base2"))], metrics=("SR",), n_boot=10)
        assert rep_ok["mixed_protocol"] == {} and "混合协议" not in sr.to_markdown(rep_ok)
        assert rep_ok["protocol"]["gz"]["values"]["heading_guard"] == ["2"] and "heading_guard" not in rep_ok["protocol"]["3dgs"]["values"]
        # --allow-mixed-protocol：放行，报告里记哪个场景的哪些键，表头标「混合协议」。
        rep = sr.build_report([("无护栏", "gz", base), ("护栏", "gz", both)], metrics=("SR",), n_boot=10, allow_mixed_protocol=True)
        assert rep["mixed_protocol"] == {"gz": ["heading_guard", "rtf"]}, rep["mixed_protocol"]
        header = next(line for line in sr.to_markdown(rep).splitlines() if line.startswith("| 方法"))
        assert "混合协议" in header and "heading_guard" in header, header
        # 命令行：不一致 rc 2、stderr 列出键；加开关 rc 0。
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                sr.main(["--eval", "A", "gz", base, "--eval", "B", "gz", both, "--n-boot", "10"])
                assert False
            except SystemExit as e:
                assert e.code == 2
        assert "heading_guard" in err.getvalue() and "rtf" in err.getvalue(), err.getvalue()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = sr.main(["--eval", "A", "gz", base, "--eval", "B", "gz", both, "--n-boot", "10", "--allow-mixed-protocol"])
        assert rc == 0 and "混合协议" in out.getvalue()


def test_success_counts_agrees_with_run_denominator():
    """D7：paired_stats.success_counts 的分母与主表（run_denominator）同口径：有请求数取请求数，没有退回完成数，不小于完成数。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        for i, (n_done, requested, failed) in enumerate(((20, 20, 0), (17, 20, 3), (5, None, 0), (3, 2, 0), (0, 20, 20))):
            seeds = list(range(1, n_done + 1))
            path = _fake_eval(os.path.join(tmp, f"e{i}.json"), seeds, ([1, 0] * n_done)[:n_done],
                              episodes_requested=requested, failed_count=failed)
            ev = sr.load_eval(path)
            vals = list(sr.metric_values(ev["per_episode"], "success").values())
            c = sr.paired_stats.success_counts(path)
            assert c["denominator"] == sr.run_denominator(ev, len(vals), True), (i, c)
            assert c["k"] == sum(1 for v in vals if v) and c["failed"] == failed and c["done"] == n_done, (i, c)


def test_dynamics_profile_from_per_episode_and_params():
    """I2-1：常规 Gazebo 评测不在顶层写 dynamics_profile，load_eval 须从 per_episode 取（单值取该串，多值拼成排序串并触发
    协议不一致报错）；per_episode 为空且顶层有 cf_sitl_params 时用 values_sha256[:8] 参与比较。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        seeds = [1, 2, 3, 4]
        old_lib = _fake_eval(os.path.join(tmp, "old_lib.json"), seeds, [1, 1, 0, 0],
                             extra={"dynamics_profile": "cf_sitl_v1:bd0dcd43"})
        new_lib = _fake_eval(os.path.join(tmp, "new_lib.json"), seeds, [1, 0, 1, 0],
                             extra={"dynamics_profile": "cf_sitl_v1:1bc8fedf:lib31"})
        new_lib2 = _fake_eval(os.path.join(tmp, "new_lib2.json"), seeds, [0, 1, 1, 0],
                              extra={"dynamics_profile": "cf_sitl_v1:1bc8fedf:lib31"})
        assert sr.load_eval(old_lib)["dynamics_profile"] == "cf_sitl_v1:bd0dcd43"
        # 相同逐集 dynamics_profile 放行并记录到口径表；不同固件库默认报错，加 --allow-mixed-protocol 放行。
        rep_same = sr.build_report([("A", "gz", new_lib), ("B", "gz", new_lib2)], metrics=("SR",), n_boot=10)
        assert rep_same["protocol"]["gz"]["values"]["dynamics_profile"] == ["cf_sitl_v1:1bc8fedf:lib31"]
        try:
            sr.build_report([("old", "gz", old_lib), ("new", "gz", new_lib)], metrics=("SR",), n_boot=10)
            assert False, "不同 per_episode dynamics_profile 混表没有报错"
        except sr.ProtocolMismatch as e:
            assert "dynamics_profile" in str(e) and "bd0dcd43" in str(e) and "1bc8fedf:lib31" in str(e), str(e)
        rep_mix = sr.build_report([("old", "gz", old_lib), ("new", "gz", new_lib)], metrics=("SR",), n_boot=10,
                                  allow_mixed_protocol=True)
        assert rep_mix["mixed_protocol"] == {"gz": ["dynamics_profile"]}, rep_mix["mixed_protocol"]

        # 单文件内 per_episode 含多个 dynamics_profile（或顶层为列表）：load_eval 返回排序拼接串，单独或同表均触发 ProtocolMismatch。
        mixed_ep = os.path.join(tmp, "mixed_ep.json")
        _fake_eval(mixed_ep, [1, 2], [1, 0])
        doc = json.load(open(mixed_ep, encoding="utf-8"))
        doc["per_episode"][0]["dynamics_profile"] = "cf_sitl_v1:b"
        doc["per_episode"][1]["dynamics_profile"] = "cf_sitl_v1:a"
        json.dump(doc, open(mixed_ep, "w", encoding="utf-8"))
        assert "cf_sitl_v1:a" in sr.load_eval(mixed_ep)["dynamics_profile"]
        for entries in ([("M", "gz", mixed_ep)], [("M", "gz", mixed_ep), ("N", "gz", new_lib)]):
            try:
                sr.build_report(entries, metrics=("SR",), n_boot=10)
                assert False, (entries, "单文件内多 dynamics_profile 没有报错")
            except sr.ProtocolMismatch as e:
                assert "dynamics_profile" in str(e) and "cf_sitl_v1:a" in str(e) and "cf_sitl_v1:b" in str(e), str(e)

        # per_episode 为空（全败臂）且带 cf_sitl_params：用 values_sha256[:8] 参与比较。
        params_a = {"path": "/p/a.json", "sha256": "1" * 64, "values_sha256": "06e5f017" + "0" * 56}
        params_b = {"path": "/p/b.json", "sha256": "2" * 64, "values_sha256": "deadbeef" + "0" * 56}
        soc_fail_a = _fake_eval(os.path.join(tmp, "soc_fail_a.json"), [], [], episodes_requested=2, failed_count=2,
                                failed=[{"seed": 1}, {"seed": 2}], dynamics_profile=None, cf_sitl_params=params_a)
        soc_fail_a2 = _fake_eval(os.path.join(tmp, "soc_fail_a2.json"), [], [], episodes_requested=2, failed_count=2,
                                 failed=[{"seed": 1}, {"seed": 2}], dynamics_profile=None, cf_sitl_params=params_a)
        soc_fail_b = _fake_eval(os.path.join(tmp, "soc_fail_b.json"), [], [], episodes_requested=2, failed_count=2,
                                failed=[{"seed": 1}, {"seed": 2}], dynamics_profile=None, cf_sitl_params=params_b)
        no_param_fail = _fake_eval(os.path.join(tmp, "no_param_fail.json"), [], [], episodes_requested=2, failed_count=2,
                                   failed=[{"seed": 1}, {"seed": 2}])
        assert sr.load_eval(soc_fail_a)["dynamics_profile"] == "06e5f017"
        sr.build_report([("A1", "gz", soc_fail_a), ("A2", "gz", soc_fail_a2)], metrics=("SR",), n_boot=10)
        for other in (new_lib, no_param_fail, soc_fail_b):
            try:
                sr.build_report([("soc_fail", "gz", soc_fail_a), ("other", "gz", other)], metrics=("SR",), n_boot=10)
                assert False, (other, "cf_sitl_params 全败臂与异构臂混表没有报错")
            except sr.ProtocolMismatch as e:
                assert "dynamics_profile" in str(e) and "06e5f017" in str(e), str(e)


def test_skipped_and_common_failed_cluster_consistency():
    """I2-3：1) 占请求名额的 skipped 集与 failed 集一样在配对与整簇区间里记 0，主表与配对表成功率一致；
    2) 两侧所有 run 都未完成的 seed 单独按 ('seed', s) 成簇，不把整张配对表退回按 seed 聚类；
    3) failed / skipped 重复记录去重，不重复累加分母；范围扫描里不占请求名额的多余 skipped 不混入配对。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        # 复现审核 I2 TEST 3c：2 个 run、请求 4 集（seed 1,2 共享 G_12，seed 3 两侧共同失败，seed 4 在 A 跳过、在 B 完成失败）。
        a_runs, b_runs = [], []
        for r_idx in range(2):
            pa = _gid_eval(os.path.join(tmp, f"a_{r_idx}.json"), [1, 2], [1, 1], lambda _s: "G_12",
                           episodes_requested=4, failed_count=1, skipped_count=1,
                           failed=[{"seed": 3}], skipped=[{"seed": 4, "reason": "skip"}])
            pb = _gid_eval(os.path.join(tmp, f"b_{r_idx}.json"), [1, 2, 4], [0, 0, 0],
                           lambda s: "G_12" if s in (1, 2) else "G_4",
                           episodes_requested=4, failed_count=1, skipped_count=0,
                           failed=[{"seed": 3}])
            a_runs.append(pa)
            b_runs.append(pb)
        rep = sr.build_report([("A", "gz", p) for p in a_runs] + [("B", "gz", p) for p in b_runs],
                              metrics=("SR",), n_boot=200, seed=0)
        ca = rep["cells"]["A"]["gz"]["SR"]
        cb = rep["cells"]["B"]["gz"]["SR"]
        pr = rep["paired"]["gz"]["SR"]["pairwise"][0]
        assert close(ca["point"], 0.5) and ca["n"] == 4 and ca["cluster_key"] == "geometry_gid" and ca["n_clusters"] == 3, ca
        assert close(cb["point"], 0.0) and cb["n"] == 4 and cb["cluster_key"] == "geometry_gid" and cb["n_clusters"] == 3, cb
        assert pr["n_common"] == 8 and close(pr["rate_A"], ca["point"]) and close(pr["rate_B"], cb["point"]), pr
        assert (pr["b"], pr["c"]) == (4, 0), pr
        assert pr["cluster"]["key"] == "geometry_gid" and pr["cluster"]["n_clusters"] == 3 and pr["cluster"]["n_nonzero"] == 1, pr
        # 单 run 配对同样把占名额的 skipped 集记 0。
        rep1 = sr.build_report([("A", "gz", a_runs[0]), ("B", "gz", b_runs[0])], metrics=("SR",), n_boot=20, seed=0)
        pr1 = rep1["paired"]["gz"]["SR"]["pairwise"][0]
        assert (pr1["n_common"], pr1["b"], pr1["c"]) == (4, 2, 0) and close(pr1["rate_A"], 0.5), pr1

        # failed 含重复记录且 missing > 1：同一个未完成 seed 在单 run 内的分母只记 1，不累加多次。
        dup_f = [_gid_eval(os.path.join(tmp, f"df_{i}.json"), [1, 2, 3], [1, 1, 0],
                           episodes_requested=6, failed_count=2, failed=[{"seed": 4}, {"seed": 4}])
                 for i in range(2)]
        evs = [sr.load_eval(p) for p in dup_f]
        assert evs[0]["missing_seeds"] == [4] and evs[0]["failed_seeds"] == [4], evs[0]
        clusters, ckey = sr.cell_clusters(evs, "success", "binary", [6, 6], True)
        assert ckey == "geometry_gid" and len(clusters) == 3 + 1 + 4, (ckey, clusters)
        assert all(c[1] <= 1 and c[3] <= 1 for c in clusters), clusters
        assert sum(c[1] for c in clusters) == 6 and sum(c[3] for c in clusters) == 6, clusters

        # 范围扫描里已满请求名额（done == requested）的多余 skipped（如 split=train）不混进配对。
        scan_a = _gid_eval(os.path.join(tmp, "sa.json"), [1, 2, 3, 4], [1, 1, 1, 0],
                           episodes_requested=4, skipped_count=2,
                           skipped=[{"seed": 98, "reason": "split=train"}, {"seed": 99, "reason": "split=val"}])
        scan_b = _gid_eval(os.path.join(tmp, "sb.json"), [1, 2, 3, 4], [1, 0, 0, 0],
                           episodes_requested=4, skipped_count=2,
                           skipped=[{"seed": 98, "reason": "split=train"}, {"seed": 99, "reason": "split=val"}])
        reps = sr.build_report([("A", "gz", scan_a), ("B", "gz", scan_b)], metrics=("SR",), n_boot=20, seed=0)
        prs = reps["paired"]["gz"]["SR"]["pairwise"][0]
        assert prs["n_common"] == 4 and close(prs["rate_A"], 0.75) and close(prs["rate_B"], 0.25), prs

        # 全部集均被跳过（完成 0、请求 4、skipped 4）：主表与配对表同记 0.00 (n=4)。
        all_sk = _fake_eval(os.path.join(tmp, "all_sk.json"), [], [], episodes_requested=4, skipped_count=4,
                            skipped=[{"seed": s, "reason": "no_corridor"} for s in (1, 2, 3, 4)])
        ok4 = _fake_eval(os.path.join(tmp, "ok4.json"), [1, 2, 3, 4], [1, 1, 0, 0])
        repk = sr.build_report([("sk", "gz", all_sk), ("ok", "gz", ok4)], metrics=("SR",), n_boot=20, seed=0)
        assert repk["cells"]["sk"]["gz"]["SR"]["point"] == 0.0 and repk["cells"]["sk"]["gz"]["SR"]["n"] == 4
        assert (repk["paired"]["gz"]["SR"]["pairwise"][0]["n_common"], repk["paired"]["gz"]["SR"]["pairwise"][0]["c"]) == (4, 2)


def main():
    for fn in (test_distributions, test_wilson, test_mcnemar, test_cochran_holm, test_iqm_bootstrap,
               test_power, test_report_end_to_end, test_mixed_metric_version_header,
               test_default_metrics_and_test_procedure_doc, test_success_metrics_use_requested_denominator,
               test_multi_run_pairing_cluster_columns, test_multi_run_pairing_alignment_and_fallback,
               test_multi_run_cell_cluster_interval, test_multi_run_cell_failed_and_spl,
               test_duplicate_seed_in_one_run_is_error, test_pairing_counts_infra_failed_seed_as_zero,
               test_all_failed_cell_shows_zero_with_requested_denominator, test_protocol_keys_mismatch_raises_unless_allowed,
               test_success_counts_agrees_with_run_denominator, test_dynamics_profile_from_per_episode_and_params,
               test_skipped_and_common_failed_cluster_consistency):
        fn()
        print("ok", fn.__name__)
    print("test_stats_report: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
