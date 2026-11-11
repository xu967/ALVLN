#!/usr/bin/env python3
"""tools/paired_stats.py 自检：每个统计量对一个手算或暴力枚举得到的已知答案。

  python tools/test_paired_stats.py
"""
import itertools
import json
import math
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paired_stats as ps  # noqa: E402
import pipeline_common  # noqa: E402


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def brute_signflip_p(diffs):
    """暴力枚举 2^k 种符号：双侧 p = #{|S| >= |观测|} / 2^k（只枚举非零簇）。"""
    nz = [abs(x) for x in diffs if x != 0]
    obs = abs(sum(diffs))
    if not nz:
        return 1.0
    hit = sum(1 for signs in itertools.product((1, -1), repeat=len(nz)) if abs(sum(s * x for s, x in zip(signs, nz))) >= obs)
    return hit / 2.0 ** len(nz)


def test_mcnemar_exact():
    # b=5, c=1：n=6，P[X<=1] = 7/64，双侧 p = 14/64。
    assert close(ps.mcnemar_exact(5, 1), 14 / 64)
    assert ps.mcnemar_exact(3, 3) == 1.0 and ps.mcnemar_exact(0, 0) == 1.0
    assert close(ps.mcnemar_exact(10, 0), 2 / 1024)
    # 不一致对很多时仍是精确二项（不换卡方）：b=40, c=10，n=50。
    want = min(1.0, 2 * sum(math.comb(50, i) for i in range(11)) / 2.0 ** 50)
    assert close(ps.mcnemar_exact(40, 10), want) and want < 1e-4
    # 评审的第四个数：calswa 对 calbest 逐集 b=12, c=9 -> 0.6636。
    assert round(ps.mcnemar_exact(12, 9), 4) == 0.6636


def test_wilson_single_source():
    assert ps.wilson_ci is pipeline_common.wilson_ci, "Wilson 公式只在 pipeline_common 里有一份"
    lo, hi = ps.wilson_ci(19, 47)
    assert round(lo, 3) == 0.276 and round(hi, 3) == 0.547, (lo, hi)


def test_stdlib_only():
    """任务卡 W1-14 第 1 条：paired_stats.py 只用标准库（屏蔽 numpy 仍可导入并运行全部核心函数）。"""
    import subprocess
    code = (
        "import sys; sys.modules['numpy'] = None; "
        "import paired_stats as ps; "
        "assert ps.signflip_test([1, 1, -1])['n_nonzero'] == 3; "
        "assert ps.pair_summary([('g1', 1, 0), ('g2', 0, 0)], n_boot=20)['n_clusters'] == 2; "
        "assert ps.success_counts({'episodes_requested': 4, 'per_episode': [{'success': True}]})['rate'] == 0.25"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_signflip_hand_examples():
    # [1,1,1]：8 种符号，|S|=3 的 2 种 -> 2/8。
    t = ps.signflip_test([1, 1, 1])
    assert close(t["p_value"], 0.25) and t["method"] == "exact" and t["n_nonzero"] == 3 and t["n_draws"] is None, t
    # [2,1,1]：和为 ±4 的 2 种 -> 2/8。
    assert close(ps.signflip_test([2, 1, 1])["p_value"], 0.25)
    # 观测和为 0：所有符号都满足 |S| >= 0 -> 1。
    assert ps.signflip_test([1, -1])["p_value"] == 1.0
    # [3,-1,-1]：观测 |1|，±3±1±1 的 8 个和都 >= 1 -> 1。
    assert ps.signflip_test([3, -1, -1])["p_value"] == 1.0
    # [4,1,1,1]：观测 7；16 种符号里 |S|>=7 的只有全正全负 -> 2/16。
    assert close(ps.signflip_test([4, 1, 1, 1])["p_value"], 2 / 16)
    # 方向不影响双侧 p。
    assert ps.signflip_test([-4, -1, -1, -1])["p_value"] == ps.signflip_test([4, 1, 1, 1])["p_value"]


def test_signflip_matches_brute_force():
    """动态规划得到的精确分布与逐个符号枚举逐位相同（随机小例，固定种子）。"""
    rng = random.Random(7)
    for _ in range(60):
        k = rng.randint(1, 11)
        d = [rng.randint(-3, 3) for _ in range(k)]
        got = ps.signflip_test(d)["p_value"]
        assert got == brute_signflip_p(d), (d, got, brute_signflip_p(d))


def test_signflip_zero_clusters_dropped():
    a = ps.signflip_test([0, 2, 0, 1, -1, 0])
    b = ps.signflip_test([2, 1, -1])
    assert a["p_value"] == b["p_value"] and a["n_clusters"] == 6 and a["n_nonzero"] == 3, (a, b)
    z = ps.signflip_test([0, 0, 0])
    assert z["p_value"] == 1.0 and z["method"] == "none" and z["n_nonzero"] == 0, z
    # 非整数的差无法做精确分布：报错，不悄悄取整。
    try:
        ps.signflip_test([0.5, 1])
        assert False
    except ValueError:
        pass


def test_signflip_reviewer_cases():
    """评审 review-d 的四个几何差向量（每个几何 3 个训练种子的成功差之和，20 个几何）；p 取四位小数。"""
    cases = {
        "calswa 对 calbest": ([0, 2, 1, 0, -1, 0, 0, 0, 0, -2, 1, 0, 2, 0, 2, 0, -1, -1, 0, 0], 0.6719),
        "calswa 对 best": ([2, 2, 2, 0, -1, -1, -1, 1, 0, -1, 0, 2, 3, 1, 1, 3, 0, 0, -1, 0], 0.0887),
        "calbest 对 best": ([2, 0, 1, 0, 0, -1, -1, 1, 0, 1, -1, 2, 1, 1, -1, 3, 1, 1, -1, 0], 0.1390),
        "guard2-best 对 best": ([2, 2, 3, -1, 0, -1, 0, 1, 0, 1, 2, 2, 2, 2, 1, 3, 1, 0, 0, 0], 0.0034),
    }
    for name, (d, want) in cases.items():
        t = ps.signflip_test(d)
        assert round(t["p_value"], 4) == want and t["method"] == "exact" and t["n_clusters"] == 20, (name, t)


def test_signflip_monte_carlo():
    """非零簇数超过 20：固定种子蒙特卡洛，记抽样次数；与精确分布接近，同种子可复现。"""
    # 25 个非零簇、每个差 ±1：S = 25 − 2X，X ~ Bin(25, 1/2)，精确 p 有闭式。
    for pos, want in ((14, 1 - 2 * math.comb(25, 12) / 2.0 ** 25),            # 和 3：p = 1 − P(X=12) − P(X=13) ≈ 0.69
                      (18, 2 * sum(math.comb(25, i) for i in range(8)) / 2.0 ** 25)):    # 和 11：p = 2·P(X ≤ 7) ≈ 0.043
        d = [1] * pos + [-1] * (25 - pos)
        assert len(d) > ps.EXACT_MAX_CLUSTERS
        exact = sum(c for s, c in ps._sum_counts([1] * 25).items() if abs(s) >= abs(sum(d))) / 2.0 ** 25
        assert close(exact, want), (pos, exact, want)                           # 动态规划与闭式一致
        t1 = ps.signflip_test(d, n_draws=20000, seed=3)
        t2 = ps.signflip_test(d, n_draws=20000, seed=3)
        t3 = ps.signflip_test(d, n_draws=20000, seed=4)
        assert t1 == t2 and t1["method"] == "monte_carlo" and t1["n_draws"] == 20000, t1
        assert abs(t1["p_value"] - exact) < 0.015 and abs(t3["p_value"] - exact) < 0.015, (pos, t1["p_value"], t3["p_value"], exact)
    # 抽样次数写进结果，p = (命中 + 1) / (抽样 + 1)，永远 > 0。
    t0 = ps.signflip_test([5] * 25, n_draws=100, seed=0)
    assert t0["p_value"] == 1 / 101 and t0["n_draws"] == 100, t0
    # 阈值按非零簇计：25 个簇里只有 15 个非零，仍走精确。
    t4 = ps.signflip_test([1] * 15 + [0] * 10)
    assert t4["method"] == "exact" and t4["n_clusters"] == 25 and t4["n_nonzero"] == 15, t4


def test_group_units_and_percentile():
    g = ps.group_units([("g1", 1, 0), ("g2", 0, 0), ("g1", 1, 1), ("g1", 0, 1)])
    assert g == {"g1": [2, 2, 3], "g2": [0, 0, 1]}, g
    assert list(g) == ["g1", "g2"]
    assert ps._percentile([1, 2, 3, 4, 5], 50) == 3.0
    assert ps._percentile([0, 10], 25) == 2.5
    assert close(ps._percentile([0, 10, 20, 30], 2.5), 0.75)
    assert ps._percentile([7], 97.5) == 7.0


def test_cluster_bootstrap():
    clusters = [(1.0,)] * 10 + [(0.0,)] * 10
    stat = lambda s: sum(c[0] for c in s) / len(s)       # noqa: E731
    lo, hi, nv = ps.cluster_bootstrap(clusters, stat, n_boot=4000, seed=1)
    # 20 个簇、点估计 0.5：整簇重采样的 95% 区间应接近正态近似 0.5 ± 1.96·√(0.25/20) = [0.30, 0.70]。
    assert nv == 4000 and 0.25 <= lo <= 0.36 and 0.64 <= hi <= 0.75, (lo, hi)
    assert (lo, hi, nv) == ps.cluster_bootstrap(clusters, stat, n_boot=4000, seed=1)       # 同种子可复现
    # 簇全相同：区间是点；少于 2 个簇：没有簇间方差可估，不装样子。
    assert ps.cluster_bootstrap([(1.0,)] * 5, stat, n_boot=100)[:2] == (1.0, 1.0)
    assert ps.cluster_bootstrap([(1.0,)], stat, n_boot=100) == (None, None, 0)
    assert ps.cluster_bootstrap([], stat, n_boot=100) == (None, None, 0)
    # 统计量返回 None 的重采样丢弃，并计入有效次数。
    flip = iter(range(10 ** 6))
    lo2, hi2, nv2 = ps.cluster_bootstrap([(1.0,), (2.0,)], lambda s: None if next(flip) % 2 else 1.0, n_boot=10)
    assert nv2 == 5 and lo2 == hi2 == 1.0, (lo2, hi2, nv2)


def _units_from_d(diffs, runs=3):
    """每个几何 runs 个单元：a 成功 max(d,0) 个、b 成功 max(-d,0) 个，a−b 之和 = d。"""
    units = []
    for gi, d in enumerate(diffs):
        sa, sb = max(d, 0), max(-d, 0)
        for j in range(runs):
            units.append((f"g{gi}", int(j < sa), int(j < sb)))
    return units


def test_pair_summary():
    d = [2, 2, 2, 0, -1, -1, -1, 1, 0, -1, 0, 2, 3, 1, 1, 3, 0, 0, -1, 0]
    s = ps.pair_summary(_units_from_d(d), n_boot=800, seed=0)
    assert s["n_units"] == 60 and s["n_clusters"] == 20 and s["n_nonzero"] == 14, s
    assert round(s["p_value"], 4) == 0.0887 and s["method"] == "exact", s
    b, c = sum(x for x in d if x > 0), sum(-x for x in d if x < 0)
    assert (s["b"], s["c"]) == (b, c) == (17, 5), s
    assert close(s["mcnemar_p"], min(1.0, 2 * sum(math.comb(22, i) for i in range(6)) / 2.0 ** 22)), s
    assert close(s["rate_a"] - s["rate_b"], s["diff"]) and close(s["diff"], sum(d) / 60), s
    assert s["ci_lo"] < s["diff"] < s["ci_hi"] and s["n_boot"] == 800 and s["n_boot_valid"] == 800, s
    # 锁分母（按总单元数 60 除而非簇数 20）：率差 +0.20 的整簇区间在 [+0.017, +0.400] 附近，绝不超过 [-1, 1]。
    assert 0.0 <= s["ci_lo"] <= 0.06 and 0.34 <= s["ci_hi"] <= 0.45, (s["ci_lo"], s["ci_hi"])
    assert s == ps.pair_summary(_units_from_d(d), n_boot=800, seed=0)
    # 锁重采样单位（整簇而非逐单元）：10 个簇 × 每簇 6 个全同单元，5 个簇全 (1,0)、5 个簇全 (0,0)。
    # 独立单位只有 10 个簇，95% 区间宽约 0.5 ± 1.96·√(0.25/10) = [0.20, 0.80]；若错按 60 个单元抽会窄成约 [0.37, 0.63]。
    corr_units = [(f"g{gi}", int(gi < 5), 0) for gi in range(10) for _ in range(6)]
    sc = ps.pair_summary(corr_units, n_boot=2000, seed=0)
    assert close(sc["diff"], 0.5) and sc["ci_lo"] <= 0.31 and sc["ci_hi"] >= 0.69, sc
    # 只有一个簇：p 仍给，区间退化为点，n_boot_valid = 0。
    one = ps.pair_summary([("g", 1, 0), ("g", 1, 0), ("g", 0, 0)], n_boot=50)
    assert one["n_clusters"] == 1 and one["ci_lo"] == one["ci_hi"] == one["diff"] and one["n_boot_valid"] == 0, one
    # 单元数 = 簇数（每个簇 1 个单元）：符号翻转 p 等于精确 McNemar（旧单 run 配对的那个数）。
    u = [(i, 1, 0) for i in range(7)] + [(10 + i, 0, 1) for i in range(2)] + [(20 + i, 1, 1) for i in range(5)]
    s1 = ps.pair_summary(u, n_boot=50)
    assert close(s1["p_value"], ps.mcnemar_exact(7, 2)) and (s1["b"], s1["c"]) == (7, 2), s1


def test_success_counts():
    """D7：新实验的 summarize 脚本取成功数与分母。成功率的分母是请求集（基础设施失败与跳过的集记 0），不是完成集。"""
    def ev(succ, **top):
        return {"per_episode": [{"seed": i, "success": bool(v)} for i, v in enumerate(succ)], **top}

    # 20 集全完成：14 成功。
    r = ps.success_counts(ev([1] * 14 + [0] * 6, episodes_requested=20, episodes_done=20))
    assert r == {"k": 14, "requested": 20, "done": 20, "failed": 0, "skipped": 0, "denominator": 20, "rate": 0.7}, r
    # 3 集基础设施失败：完成 17、成功 10，比例除以请求数 20（0.5），不是除以完成数 17（0.588）。
    r = ps.success_counts(ev([1] * 10 + [0] * 7, episodes_requested=20, episodes_done=17, failed_count=3))
    assert (r["k"], r["requested"], r["done"], r["failed"], r["denominator"], r["rate"]) == (10, 20, 17, 3, 20, 0.5), r
    # failed_count / skipped_count 缺省时数 failed / skipped 列表（eval_closed_loop 写的是列表）。
    r = ps.success_counts(ev([1, 0, 1], episodes_requested=6, failed=[{"seed": 7}, {"seed": 8}], skipped=[{"seed": 9}]))
    assert (r["failed"], r["skipped"], r["done"], r["denominator"], r["rate"]) == (2, 1, 3, 6, 2 / 6), r
    # 全部失败：没有 per_episode，请求 20 集，比例 0.0（不是 None）。
    r = ps.success_counts({"episodes_requested": 20, "failed_count": 20, "episodes_done": 0})
    assert (r["k"], r["done"], r["failed"], r["denominator"], r["rate"]) == (0, 0, 20, 20, 0.0), r
    # 旧文件没有 episodes_requested：requested 为 None，分母退回观测到的集数。
    r = ps.success_counts(ev([1, 1, 0, 0]))
    assert (r["requested"], r["denominator"], r["rate"]) == (None, 4, 0.5), r
    # 请求数小于观测数（不应出现）：分母取观测数，比例不超过 1。
    r = ps.success_counts(ev([1, 1, 1], episodes_requested=2))
    assert r["denominator"] == 3 and r["rate"] == 1.0, r
    # 空文件、没有请求数：分母 0，比例 None，不抛除零。
    assert ps.success_counts({})["rate"] is None and ps.success_counts({})["denominator"] == 0
    # 指标字段可选；字段是 None 的集不是观测（旧文件没有 reach_ever）。
    per = [{"seed": 0, "success": True, "task_success": True}, {"seed": 1, "success": True, "task_success": False},
           {"seed": 2, "success": True}]
    r = ps.success_counts({"per_episode": per, "episodes_requested": 5}, field="task_success")
    assert (r["k"], r["denominator"], r["rate"]) == (1, 5, 0.2), r
    assert ps.success_counts({"per_episode": per, "episodes_requested": 5})["k"] == 3
    # 路径：eval.json 文件，或含它的目录。
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        path = os.path.join(tmp, "eval.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ev([1, 0, 1, 1], episodes_requested=5, episodes_done=4, failed_count=1), f)
        a, b = ps.success_counts(path), ps.success_counts(tmp)
        assert a == b and (a["k"], a["denominator"], a["failed"]) == (3, 5, 1), (a, b)


def main():
    for fn in (test_mcnemar_exact, test_wilson_single_source, test_stdlib_only, test_signflip_hand_examples,
               test_signflip_matches_brute_force, test_signflip_zero_clusters_dropped, test_signflip_reviewer_cases,
               test_signflip_monte_carlo, test_group_units_and_percentile, test_cluster_bootstrap, test_pair_summary,
               test_success_counts):
        fn()
        print("ok", fn.__name__)
    print("test_paired_stats: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
