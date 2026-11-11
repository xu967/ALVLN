#!/usr/bin/env python3
"""配对统计（只用标准库）：几何聚类的符号翻转检验、整簇 bootstrap 区间、精确 McNemar、Wilson。

为什么要聚类：闭环评测的 60 集 = 20 个冻结几何 × 3 个训练种子，同一个几何上的三集不独立。把 60 集当 60 个独立配对
（逐集 McNemar），p 低于按几何聚类的 p；独立单位是几何。约定（docs/EVAL_STANDARD.md §5.6）：
  单元   一次配对评测（某个 run 的某个 seed 上，方法 A 与方法 B 各一个 0/1 结果）；
  簇     同一个几何上的全部单元（geometry_gid；没有该字段时按 seed 聚）；
  差     每个簇内 A 成功数减 B 成功数，记 d_g（整数）；
  检验   对非零簇做双侧符号翻转：H0 下每个 d_g 的正负号等可能，统计量 |Σ d_g|，p = #{|S| ≥ |观测|} / 2^k。
         非零簇数 ≤ EXACT_MAX_CLUSTERS 时精确（对和的分布做动态规划，与逐个枚举 2^k 种符号逐位相同，
         测试里用暴力枚举对拍），否则固定种子蒙特卡洛，结果里写明抽样次数，p = (命中 + 1) / (抽样 + 1)；
  区间   整个簇有放回重采样（簇数不变）、百分位 95% 区间（线性插值，与 numpy.percentile 默认同）；
  参考   逐集精确 McNemar（把单元当独立样本，p 低于按几何聚类的 p）与 Wilson（pipeline_common.wilson_ci，公式只有一份）。
另有 success_counts：新实验的 summarize 脚本用它取成功数与三个分母（请求 / 完成 / 失败），不再各写各的 n。

库用法见 tools/stats_report.py（配对表的两列）；单测 tools/test_paired_stats.py。
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_common  # noqa: E402  - wilson_ci 只此一份

EXACT_MAX_CLUSTERS = 20       # 非零簇数不超过此值走精确，否则蒙特卡洛（任务卡 W1-14 的规定）
DEFAULT_DRAWS = 100_000       # 蒙特卡洛抽样次数的默认值
wilson_ci = pipeline_common.wilson_ci   # Wilson 95% (lo, hi)；n=0 → None


def mcnemar_exact(b, c):
    """精确 McNemar 双侧 p：b = A 成功 B 失败的单元数，c = A 失败 B 成功的单元数；p = min(1, 2·P[X ≤ min(b,c)])，X ~ Bin(b+c, 1/2)。
    b+c = 0 时无信息，p = 1。与 stats_report.mcnemar 的区别：这里不论 b+c 多大都用精确二项（不换卡方）。"""
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def _as_int(x):
    """几何差是成功数之差，必须是整数；非整数（如按几何取均值）不走精确分布，直接报错。"""
    if isinstance(x, (bool, int)):
        return int(x)
    if isinstance(x, float) and math.isfinite(x) and x == int(x):
        return int(x)
    raise ValueError(f"几何差必须是整数（成功数之差），收到 {x!r}")


def _sum_counts(abs_values):
    """{符号分配的和: 分配个数}：对每个 |d| 取 + 或 −，累计和的分布（动态规划，等价于枚举全部 2^k 种符号）。"""
    counts = {0: 1}
    for x in abs_values:
        nxt = {}
        for s, c in counts.items():
            nxt[s + x] = nxt.get(s + x, 0) + c
            nxt[s - x] = nxt.get(s - x, 0) + c
        counts = nxt
    return counts


def signflip_test(diffs, n_draws=DEFAULT_DRAWS, seed=0, exact_max=EXACT_MAX_CLUSTERS):
    """每个簇的整数差 d_g（簇内 A 成功数减 B 成功数）的双侧符号翻转检验。零簇不改变和，先丢掉。

    返回 {p_value, method("exact" | "monte_carlo" | "none"), n_clusters, n_nonzero, observed_sum, n_draws}。
    n_draws 只在蒙特卡洛时有值。阈值按非零簇数算（零簇不增加枚举量）。
    """
    d = [_as_int(x) for x in diffs]
    nz = [abs(x) for x in d if x != 0]
    obs = abs(sum(d))
    out = {"n_clusters": len(d), "n_nonzero": len(nz), "observed_sum": sum(d), "n_draws": None}
    if not nz:
        return {**out, "p_value": 1.0, "method": "none"}
    if len(nz) <= exact_max:
        tail = sum(c for s, c in _sum_counts(nz).items() if abs(s) >= obs)
        return {**out, "p_value": tail / 2.0 ** len(nz), "method": "exact"}
    # shortcut: 非零簇数超过 exact_max 时蒙特卡洛是任务卡的规定；上面的动态规划对任意簇数都精确，要全程精确只需去掉这个分支。
    rng = random.Random(seed)
    hit = sum(1 for _ in range(n_draws) if abs(sum(x if rng.random() < 0.5 else -x for x in nz)) >= obs)
    return {**out, "p_value": (hit + 1) / (n_draws + 1), "method": "monte_carlo", "n_draws": n_draws}


def group_units(units):
    """[(簇, a, b)] -> {簇: [A 成功数, B 成功数, 单元数]}（保持首次出现的顺序）。a、b 是 0/1 或布尔。"""
    out = {}
    for g, a, b in units:
        e = out.setdefault(g, [0, 0, 0])
        e[0] += int(bool(a))
        e[1] += int(bool(b))
        e[2] += 1
    return out


def _percentile(sorted_vals, q):
    """已排序数组的 q 百分位（0 到 100），线性插值，与 numpy.percentile 默认同。"""
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def cluster_bootstrap(clusters, statistic, n_boot=2000, alpha=0.05, seed=0):
    """整簇有放回重采样的百分位区间。clusters 是簇的列表（元素由 statistic 自己解释）；
    statistic(重采样出的簇列表) -> float，或 None（这一次不可算，丢弃）。

    返回 (lo, hi, n_valid)；不足 2 个簇没有簇间方差可估，返回 (None, None, 0)，由调用方退化成点估计。
    """
    k = len(clusters)
    if k < 2:
        return None, None, 0
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        v = statistic([clusters[j] for j in rng.choices(range(k), k=k)])
        if v is not None:
            boots.append(v)
    if not boots:
        return None, None, 0
    boots.sort()
    return _percentile(boots, 100.0 * alpha / 2.0), _percentile(boots, 100.0 * (1.0 - alpha / 2.0)), len(boots)


def success_counts(ev, field="success"):
    """新实验的 summarize 脚本用：eval.json（读好的 dict，或 eval.json / 含它的目录的路径）-> 成功数与各分母。
    比例的分母是请求集 episodes_requested，基础设施失败与跳过的集记 0（docs/EVAL_STANDARD.md §5.1、§5.6，与 stats_report 同口径）；
    旧文件没有 episodes_requested 就退回观测到的集数，请求数小于观测数（不应出现）时取观测数，所以比例不超过 1。
    返回 {k: 成功数, requested: 请求集（旧文件为 None）, done: 完成集, failed: 失败集, skipped: 跳过集,
          denominator: 比例的分母, rate: k / denominator（分母为 0 时 None）}。field 取 per_episode 的字段名，None 的集不算观测。"""
    if not isinstance(ev, dict):
        path = os.fspath(ev)
        if os.path.isdir(path):
            path = os.path.join(path, "eval.json")
        with open(path, encoding="utf-8") as f:
            ev = json.load(f)
    per = ev.get("per_episode") or []
    obs = [ep[field] for ep in per if ep.get(field) is not None]
    k = sum(1 for v in obs if v)
    requested = ev.get("episodes_requested")
    denominator = max(int(requested), len(obs)) if requested else len(obs)
    return {"k": k, "requested": requested, "done": ev.get("episodes_done", len(per)),
            "failed": ev.get("failed_count", len(ev.get("failed") or [])),
            "skipped": ev.get("skipped_count", len(ev.get("skipped") or [])),
            "denominator": denominator, "rate": k / denominator if denominator else None}


def pair_summary(units, n_boot=2000, alpha=0.05, seed=0, n_draws=DEFAULT_DRAWS):
    """一对方法的聚类配对汇总。units = [(簇, a, b)]（a、b 为 0/1）。

    给出符号翻转 p（见 signflip_test）、A 率减 B 率的差与整簇 bootstrap 区间、逐集 b/c 与精确 McNemar p（参考）。
    只有 1 个簇时区间退化为点（n_boot_valid = 0）。
    """
    units = list(units)
    clusters = list(group_units(units).values())          # [A 成功数, B 成功数, 单元数]
    n = sum(c[2] for c in clusters)
    sa, sb = sum(c[0] for c in clusters), sum(c[1] for c in clusters)
    test = signflip_test([c[0] - c[1] for c in clusters], n_draws=n_draws, seed=seed)

    def diff_of(sample):
        m = sum(c[2] for c in sample)
        return (sum(c[0] for c in sample) - sum(c[1] for c in sample)) / m if m else None

    lo, hi, n_valid = cluster_bootstrap(clusters, diff_of, n_boot=n_boot, alpha=alpha, seed=seed)
    diff = (sa - sb) / n
    if lo is None:
        # shortcut: 不足 2 个簇没有簇间方差可估，区间取点估计（与 stats_report 单 run 区间退化为点是同一个约定）。
        lo = hi = diff
    b = sum(1 for _g, a, bb in units if a and not bb)
    c = sum(1 for _g, a, bb in units if not a and bb)
    return {"n_units": n, "n_clusters": test["n_clusters"], "n_nonzero": test["n_nonzero"],
            "p_value": test["p_value"], "method": test["method"], "n_draws": test["n_draws"],
            "observed_sum": test["observed_sum"], "rate_a": sa / n, "rate_b": sb / n, "diff": diff,
            "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot, "n_boot_valid": n_valid,
            "b": b, "c": c, "mcnemar_p": mcnemar_exact(b, c)}
