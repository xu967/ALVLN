#!/usr/bin/env python3
"""闭环评测统计汇总（库 + CLI；只用 numpy 与标准库，不装 scipy/rliable）。

读若干 eval.json（sim/tools/eval_closed_loop.py 或 sim/3dgs/eval_3dgs.py 的产物）+ 分组名，输出论文主表
（行 = 方法，列 = 场景 × 指标，每格 `x.xx [lo, hi] (n)`）与配对检验表，Markdown + JSON 各一份。

统计口径（docs/EVAL_STANDARD.md §5.3，分母见 §5.6）：
  二值指标: 比例 + Wilson 95%。越高越好的（SR/task_success/reach_ever/stop_success）分母 = 请求集 episodes_requested，
    失败/跳过集记 0；越低越好的（CR/false_forward/stuck）分母 = 完成集 episodes_done。表里单列「失败集数」；
  连续指标（SPL）: 完成集上的均值 + 百分位 bootstrap 95%；
  同一 (方法, 场景) 多个 run（训练种子）: IQM + 分层 bootstrap 95%（Agarwal et al. NeurIPS 2021）；
  两方法配对（同 seed 集）: McNemar（b+c<25 精确二项，否则带连续性校正的卡方）；
  ≥3 方法: 另给 Cochran's Q（全体共同 seed）作总体检验；两两 McNemar 照做、不以 Q 显著为前提；
    Holm 族 = 同一场景、同一指标的全部两两行（k 方法 → k(k-1)/2 个 p）；
  样本量: 两比例功效估算（正态近似 + Fleiss 连续性校正）。
默认列 SR/CR/false_forward/SPL；task_success 要显式 --metrics 加（base 任务没有 waypoint，恒为 0）。

用法（仓库根，.venv-tools 的 python）：
  python tools/stats_report.py --out-dir $T/agent4-eval-matrix/e1-four-groups \
      --eval teacher gazebo $OUT/eval-teacher/eval.json \
      --eval student-e1-new gazebo $OUT/eval-student-e1-new/eval.json \
      --eval forward gazebo $OUT/eval-forward/eval.json
  同一方法多个训练种子：同名多次 --eval（如 --eval S_A gazebo run-s0/eval.json --eval S_A gazebo run-s1/eval.json）。
  python tools/stats_report.py --power 0.8 0.9            # 检出 0.80 vs 0.90 每臂要多少集
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_common  # noqa: E402  - wilson_ci 只此一份

# 表头名 -> (per_episode 字段, 类型, 方向)。CR 就是 collision（contact 口径，stuck/超时不算碰撞），不另起字段。
METRICS = {
    "SR": ("success", "binary", "↑"),
    "CR": ("collision", "binary", "↓"),
    "stuck": ("stuck", "binary", "↓"),
    "false_forward": ("false_forward", "binary", "↓"),
    "task_success": ("task_success", "binary", "↑"),
    "reach_ever": ("reach_ever", "binary", "↑"),
    "stop_success": ("stop_success", "binary", "↑"),
    "SPL": ("spl", "mean", "↑"),
}
DEFAULT_METRICS = ("SR", "CR", "false_forward", "SPL")   # task_success 只在有 waypoint 的任务（l1/campus）有意义，显式 --metrics 加
PILOT_N = 20            # e1-demo-plan §3：n<20 标 pilot
EXACT_MCNEMAR_MAX = 25  # 不一致对 b+c 小于此用精确二项，否则卡方（Agresti 的常用阈值）
_NORMAL = statistics.NormalDist()


# ───────────────────────────── 基础分布函数 ─────────────────────────────
def norm_ppf(q):
    return _NORMAL.inv_cdf(q)


def _gammp_series(a, x):
    """正则化下不完全伽马 P(a, x)，级数展开（x < a+1 时收敛快）。Numerical Recipes gser。"""
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(10000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-16:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammq_cf(a, x):
    """正则化上不完全伽马 Q(a, x)，Lentz 连分式（x >= a+1）。Numerical Recipes gcf。"""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    """卡方分布生存函数 P(X > x) = Q(df/2, x/2)。"""
    if df <= 0:
        raise ValueError("df 必须为正")
    if x <= 0.0:
        return 1.0
    a, hx = df / 2.0, x / 2.0
    return (1.0 - _gammp_series(a, hx)) if hx < a + 1.0 else _gammq_cf(a, hx)


# ───────────────────────────── 区间与检验 ─────────────────────────────
wilson_ci = pipeline_common.wilson_ci   # Wilson 95% (lo, hi)；n=0 → None。一份公式（与 eval_closed_loop.wilson_ci 同式）


def mcnemar(b, c):
    """配对二值结果的 McNemar 检验。b = A 成功 B 失败的集数，c = A 失败 B 成功的集数。

    b+c < EXACT_MCNEMAR_MAX：精确二项双侧 p = min(1, 2·P[X ≤ min(b,c)]，X~Bin(b+c, 1/2)]；
    否则 (|b−c|−1)²/(b+c) ~ χ²(1)（带连续性校正）。b+c=0 时无信息，p=1。
    """
    b, c = int(b), int(c)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if n < EXACT_MCNEMAR_MAX:
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
        return {"b": b, "c": c, "n_discordant": n, "statistic": None,
                "p_value": min(1.0, 2.0 * tail), "method": "exact_binomial"}
    stat = (abs(b - c) - 1.0) ** 2 / n
    return {"b": b, "c": c, "n_discordant": n, "statistic": stat,
            "p_value": chi2_sf(stat, 1), "method": "chi2_cc"}


def paired_outcomes(a, b):
    """两个 {seed: bool} 在共同 seed 上的 (b, c, n_common)：b = a 真 b 假，c = a 假 b 真。"""
    common = sorted(set(a) & set(b))
    bb = sum(1 for s in common if a[s] and not b[s])
    cc = sum(1 for s in common if not a[s] and b[s])
    return bb, cc, len(common)


def cochran_q(table):
    """Cochran's Q：table[i][j] 为第 i 个受试（seed）在第 j 个方法上的 0/1。k=2 时等于不校正的 McNemar 卡方。"""
    m = np.asarray(table, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("table 需为 n×k（k≥2）")
    n, k = m.shape
    col, row = m.sum(axis=0), m.sum(axis=1)
    denom = k * row.sum() - (row ** 2).sum()
    if denom <= 0:   # 所有受试在各方法上结果全同：无差异可检
        return {"Q": 0.0, "df": k - 1, "p_value": 1.0, "k": k, "n": n}
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denom
    return {"Q": float(q), "df": k - 1, "p_value": chi2_sf(float(q), k - 1), "k": k, "n": n}


def holm(pvalues):
    """Holm 逐步校正，返回与输入同序的校正 p（单调、截到 1）。"""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ───────────────────────────── 多 run 聚合（Agarwal 2021） ─────────────────────────────
def iqm(values):
    """四分位均值：排序后去掉两端各 int(0.25·n) 个再平均（与 scipy.stats.trim_mean(x, 0.25) 同定义）。

    n<4 时不截，退化成均值；二值数据不要直接喂（0/1 截尾后只剩多数类）。
    """
    x = np.sort(np.asarray(values, dtype=float).ravel())
    if x.size == 0:
        raise ValueError("iqm 需要至少 1 个值")
    cut = int(0.25 * x.size)
    return float(x[cut:x.size - cut].mean())


def stratified_bootstrap_ci(matrix, stat=iqm, n_boot=2000, alpha=0.05, seed=0):
    """runs × tasks 分数矩阵的聚合统计量 + 分层 bootstrap 百分位区间：每个 task 内独立对 runs 有放回重采样。

    返回 (point, lo, hi)。1 个 run 时区间退化为点（没有 run 间方差可估，不装样子）。
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 1:
        m = m[:, None]
    runs, tasks = m.shape
    if runs == 0 or tasks == 0:
        raise ValueError("矩阵为空")
    point = stat(m)
    if runs == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, runs, size=(n_boot, runs, tasks))
    samples = m[idx, np.arange(tasks)[None, None, :]]          # (n_boot, runs, tasks)
    if stat is iqm:   # 向量化：排序后截尾
        flat = np.sort(samples.reshape(n_boot, runs * tasks), axis=1)
        cut = int(0.25 * flat.shape[1])
        boots = flat[:, cut:flat.shape[1] - cut].mean(axis=1)
    else:
        boots = np.array([stat(s) for s in samples])
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return point, float(lo), float(hi)


def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """单组连续值（SPL）均值的百分位 bootstrap 95%。"""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(x.mean()), float(lo), float(hi)


# ───────────────────────────── 样本量 ─────────────────────────────
def power_two_proportions(p1, p2, alpha=0.05, power=0.8):
    """两独立比例、双侧 α、功效 1−β 下每臂需要的集数（正态近似；corrected 为 Fleiss 连续性校正）。

    n = [z_{1−α/2}·√(2·p̄·(1−p̄)) + z_{1−β}·√(p1(1−p1)+p2(1−p2))]² / (p1−p2)²，p̄ = (p1+p2)/2。
    配对设计（同 seed 集）用 McNemar 通常更省，这里给的是非配对上界。
    """
    for p in (p1, p2):
        if not 0.0 < p < 1.0:
            raise ValueError("p1、p2 必须在 (0, 1) 内")
    if p1 == p2:
        raise ValueError("p1 == p2 时无穷大")
    za, zb = norm_ppf(1.0 - alpha / 2.0), norm_ppf(power)
    pbar = (p1 + p2) / 2.0
    n = (za * math.sqrt(2.0 * pbar * (1.0 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 \
        / (p1 - p2) ** 2
    corrected = n / 4.0 * (1.0 + math.sqrt(1.0 + 4.0 / (n * abs(p1 - p2)))) ** 2
    return {"p1": p1, "p2": p2, "alpha": alpha, "power": power,
            "n_per_arm": int(math.ceil(n)), "n_per_arm_corrected": int(math.ceil(corrected))}


# ───────────────────────────── 读 eval.json、汇总成表 ─────────────────────────────
def load_eval(path):
    """eval.json（或含它的目录）-> 报表需要的字段。失败/跳过集不在 per_episode 里，另记数。"""
    if os.path.isdir(path):
        path = os.path.join(path, "eval.json")
    with open(path, encoding="utf-8") as f:
        e = json.load(f)
    per = e.get("per_episode") or []
    hold = e.get("hold_s")
    return {
        "path": os.path.abspath(path), "per_episode": per,
        "episodes_done": e.get("episodes_done", len(per)),
        "episodes_requested": e.get("episodes_requested"),
        "failed": e.get("failed_count", len(e.get("failed") or [])),
        "skipped": e.get("skipped_count", len(e.get("skipped") or [])),
        "hold_s": hold, "metric_version": e.get("metric_version"),
        "success_criterion": e.get("success_criterion"), "dynamics": e.get("dynamics"),
        "vehicle": e.get("vehicle"), "window_from": e.get("window_from"),
        "task": e.get("task"), "route": (e.get("scene") or {}).get("route") if isinstance(e.get("scene"), dict) else None,
        "spl_reference": e.get("spl_reference"),
    }


def metric_values(per_episode, field):
    """per_episode -> {seed: 值}，None 的集丢掉（如旧 eval.json 没有 spl）。"""
    out = {}
    for i, ep in enumerate(per_episode):
        v = ep.get(field)
        if v is None:
            continue
        out[ep.get("seed", i)] = v
    return out


def uses_requested(metric):
    """越高越好的二值指标（SR/task_success/reach_ever/stop_success）分母取请求集：基础设施失败的集不能从分母里拿掉，否则成功率虚高。
    越低越好的二值指标（CR/stuck/false_forward）失败集没有碰撞/乱飞的观测，记 0 会把它们压低，分母仍是完成集；连续指标（SPL）同。"""
    _field, kind, direction = METRICS[metric]
    return kind == "binary" and direction == "↑"


def run_denominator(ev, n_done, requested):
    """一个 run 的比例分母：requested 时取 episodes_requested（失败/跳过集记 0），否则取完成集。
    旧 eval.json 没有 episodes_requested 就退回完成集；请求数小于完成数（不应出现）时取完成数，比例不超过 1。"""
    req = ev.get("episodes_requested")
    return max(int(req), n_done) if requested and req else n_done


def _run_value(vals, kind, n=None):
    return (sum(1 for v in vals if v) / (n or len(vals))) if kind == "binary" else (sum(vals) / len(vals))


def summarize_cell(runs, field, kind, n_boot, seed, requested=False):
    """一个 (方法, 场景) 格：1 个 run 给 Wilson / bootstrap 均值；多个 run 给 IQM + 分层 bootstrap。
    requested=True（越高越好的二值指标）时比例的分母是请求集；格里的 n 就是分母，分母与完成数不同时另记 n_done。"""
    per_run = [(list(metric_values(r["per_episode"], field).values()), r) for r in runs]
    per_run = [(v, r) for v, r in per_run if v]
    if not per_run:
        return None
    denoms = [run_denominator(r, len(v), requested and kind == "binary") for v, r in per_run]
    if len(per_run) == 1:
        vals, n_done = per_run[0][0], len(per_run[0][0])
        if kind == "binary":
            n, k = denoms[0], sum(1 for v in vals if v)
            lo, hi = wilson_ci(k, n)
            cell = {"point": k / n, "lo": lo, "hi": hi, "n": n, "k": k, "runs": 1, "kind": "wilson"}
            if n != n_done:
                cell["n_done"] = n_done
            return cell
        point, lo, hi = bootstrap_mean_ci(vals, n_boot=n_boot, seed=seed)
        return {"point": point, "lo": lo, "hi": hi, "n": n_done, "runs": 1, "kind": "bootstrap_mean"}
    scores = [[_run_value(v, kind, d)] for (v, _r), d in zip(per_run, denoms)]          # runs × 1 task
    point, lo, hi = stratified_bootstrap_ci(scores, n_boot=n_boot, seed=seed)
    ns, dones = sorted(set(denoms)), sorted({len(v) for v, _r in per_run})
    cell = {"point": point, "lo": lo, "hi": hi, "n": ns[0] if len(ns) == 1 else ns, "runs": len(per_run),
            "kind": "iqm_stratified"}
    if ns != dones:
        cell["n_done"] = dones[0] if len(dones) == 1 else dones
    return cell


def summarize_all_scenes(runs_by_scene, field, kind, n_boot, seed, requested=False):
    """一个方法跨场景的 IQM：矩阵 runs × scenes，元素 = 该 run 在该场景的指标值（场景间 run 数须相同）。
    分母规则与 summarize_cell 一致（requested 见 uses_requested），同一张表里不留两套分母。"""
    cols = []
    for scene_runs in runs_by_scene.values():
        col = []
        for r in scene_runs:
            vals = list(metric_values(r["per_episode"], field).values())
            col.append(_run_value(vals, kind, run_denominator(r, len(vals), requested and kind == "binary")) if vals else None)
        cols.append(col)
    runs = {len(c) for c in cols}
    if len(runs) != 1 or any(v is None for c in cols for v in c):
        return None
    matrix = np.asarray(cols, dtype=float).T                    # runs × scenes
    point, lo, hi = stratified_bootstrap_ci(matrix, n_boot=n_boot, seed=seed)
    return {"point": point, "lo": lo, "hi": hi, "runs": matrix.shape[0], "scenes": matrix.shape[1],
            "kind": "iqm_stratified"}


def paired_tests(single_runs, field):
    """同一场景内单 run 方法两两 McNemar（共同 seed）+ Holm；≥3 方法先 Cochran's Q（全体共同 seed）。"""
    names = list(single_runs)
    outcomes = {n: {s: bool(v) for s, v in metric_values(single_runs[n]["per_episode"], field).items()}
                for n in names}
    out = {"cochran_q": None, "pairwise": []}
    if len(names) >= 3:
        common = sorted(set.intersection(*(set(o) for o in outcomes.values())))
        if len(common) >= 2:
            out["cochran_q"] = cochran_q([[int(outcomes[n][s]) for n in names] for s in common])
            out["cochran_q"]["methods"] = names
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            bb, cc, n = paired_outcomes(outcomes[a], outcomes[b])
            if n == 0:
                continue
            common = set(outcomes[a]) & set(outcomes[b])
            rows.append({"A": a, "B": b, "n_common": n,
                         "rate_A": sum(1 for s in common if outcomes[a][s]) / n,
                         "rate_B": sum(1 for s in common if outcomes[b][s]) / n,
                         **mcnemar(bb, cc)})
    if rows:
        for r, p in zip(rows, holm([r["p_value"] for r in rows])):
            r["p_holm"] = p
    out["pairwise"] = rows
    return out


def build_report(entries, metrics=DEFAULT_METRICS, pair_metrics=("SR",), n_boot=2000, seed=0, title=None):
    """entries: [(method, scene, eval.json 路径)]。返回可直接 json.dump 的 dict（表 + 检验 + 口径）。"""
    for m in tuple(metrics) + tuple(pair_metrics):
        if m not in METRICS:
            raise ValueError(f"未知指标 {m}，可选 {sorted(METRICS)}")
    loaded = [(method, scene, load_eval(path)) for method, scene, path in entries]
    methods = list(dict.fromkeys(m for m, _, _ in loaded))
    scenes = list(dict.fromkeys(s for _, s, _ in loaded))
    groups = {}
    for method, scene, ev in loaded:
        groups.setdefault(method, {}).setdefault(scene, []).append(ev)
    cells, all_scenes = {}, {}
    for method in methods:
        cells[method] = {}
        for scene in scenes:
            runs = groups[method].get(scene, [])
            cells[method][scene] = {m: (summarize_cell(runs, METRICS[m][0], METRICS[m][1], n_boot, seed, uses_requested(m))
                                        if runs else None) for m in metrics}
        if len(scenes) >= 2 and len(groups[method]) == len(scenes):
            all_scenes[method] = {m: summarize_all_scenes(groups[method], METRICS[m][0], METRICS[m][1], n_boot, seed,
                                                          uses_requested(m))
                                  for m in metrics}
    paired = {}
    for scene in scenes:
        single = {m: groups[m][scene][0] for m in methods if len(groups[m].get(scene, [])) == 1}
        skipped = [m for m in methods if len(groups[m].get(scene, [])) > 1]
        if len(single) >= 2:
            paired[scene] = {pm: paired_tests(single, METRICS[pm][0]) for pm in pair_metrics}
            paired[scene]["skipped_multi_run"] = skipped
    protocol = {}
    for scene in scenes:
        keys = ("metric_version", "success_criterion", "hold_s", "dynamics", "vehicle", "window_from", "task")
        vals = {k: sorted({str(ev[k]) for m in methods for ev in groups[m].get(scene, [])}) for k in keys}
        protocol[scene] = {"values": vals, "inconsistent": [k for k, v in vals.items() if len(v) > 1]}
    return {
        "schema": "uavrgb.stats_report.v1", "title": title, "generated": time.strftime("%F %T"),
        "metrics": list(metrics), "pair_metrics": list(pair_metrics), "methods": methods, "scenes": scenes,
        "n_boot": n_boot, "seed": seed,
        # 比例的分母：requested = episodes_requested（失败/跳过集记 0），done = episodes_done；failed_total = 方法的全部 run、全部场景的失败集数之和
        "denominators": {m: "requested" if uses_requested(m) else "done" for m in metrics},
        "failed_total": {m: sum(ev["failed"] for runs in groups[m].values() for ev in runs) for m in methods},
        # 全表出现过的 metric_version；>1 个就是混合（sim09.v1 过线后继续飞 vs v2 成功线即停），表头要标。
        "metric_versions": sorted({str(ev["metric_version"]) for _, _, ev in loaded}),
        "cells": cells, "all_scenes": all_scenes, "paired": paired, "protocol": protocol,
        "entries": [{"method": m, "scene": s, **{k: v for k, v in ev.items() if k != "per_episode"}}
                    for m, s, ev in loaded],
    }


# ───────────────────────────── Markdown ─────────────────────────────
def fmt_cell(c):
    if c is None:
        return "—"
    n = c.get("n")
    n_txt = f"{c['runs']}×{n}" if c.get("runs", 1) > 1 and not isinstance(n, list) else (
        f"{c['runs']} runs" if c.get("runs", 1) > 1 else str(n))
    if "scenes" in c:
        n_txt = f"{c['runs']}×{c['scenes']} scenes"
    return f"{c['point']:.2f} [{c['lo']:.2f}, {c['hi']:.2f}] ({n_txt})"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def to_markdown(rep):
    lines = []
    if rep.get("title"):
        lines += [f"# {rep['title']}", ""]
    versions = rep.get("metric_versions") or []
    mixed = (f" ⚠ metric_version 混合：{' / '.join(versions)}（v1 过线后继续飞、碰撞可能发生在过线之后；"
             "v2 成功线即停；SR/CR 不可直接比，见 EVAL_STANDARD §5）" if len(versions) > 1 else "")
    header = ["方法" + mixed] + [f"{s} {m}{METRICS[m][2]}" for s in rep["scenes"] for m in rep["metrics"]]
    if rep["all_scenes"]:
        header += [f"全场景 IQM {m}{METRICS[m][2]}" for m in rep["metrics"]]
    header.append("失败集数")
    lines += ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for method in rep["methods"]:
        row = [method] + [fmt_cell(rep["cells"][method][s][m]) for s in rep["scenes"] for m in rep["metrics"]]
        if rep["all_scenes"]:
            row += [fmt_cell((rep["all_scenes"].get(method) or {}).get(m)) for m in rep["metrics"]]
        row.append(str((rep.get("failed_total") or {}).get(method, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "每格 `点估计 [95% 区间] (n)`：二值指标 Wilson，分母按方向分两种。越高越好的指标（SR、task_success、reach_ever、stop_success）"
                  "n = 请求集 episodes_requested，失败集与跳过集记 0；越低越好的指标（CR、stuck、false_forward）n = 完成集 episodes_done。"
                  f"SPL 为完成集上的百分位 bootstrap 均值区间（B={rep['n_boot']}）；多 run 的格为 IQM + 分层 bootstrap（Agarwal 2021），"
                  f"每个 run 按各自的分母算比例。n<{PILOT_N} 的格只算 pilot。"
                  " 「失败集数」= 该方法全部 run、全部场景的 failed_count 之和，分场景明细见下表「失败」列。"
                  " CR = contact 碰撞比例（stuck/超时不算）。SPL 的 l_i 口径见各 eval.json 的 spl_reference。"]
    for scene in rep["scenes"]:
        pr = rep["protocol"][scene]
        kv = "；".join(f"{k}={'/'.join(v)}" for k, v in pr["values"].items())
        warn = f" **协议不一致：{', '.join(pr['inconsistent'])}**" if pr["inconsistent"] else ""
        lines.append(f"- 场景 `{scene}` 口径：{kv}{warn}")
    lines += ["", "| 方法 | 场景 | 请求 | 完成 | 失败 | 跳过 | metric_version | eval.json |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in rep["entries"]:
        lines.append(f"| {e['method']} | {e['scene']} | {e['episodes_requested']} | {e['episodes_done']} | "
                     f"{e['failed']} | {e['skipped']} | {e['metric_version']} | `{e['path']}` |")
    for scene, tests in rep["paired"].items():
        for pm in rep["pair_metrics"]:
            t = tests[pm]
            lines += ["", f"## 配对检验：`{scene}` × {pm}（共同 seed 集）", ""]
            if t["cochran_q"]:
                q = t["cochran_q"]
                lines.append(f"Cochran's Q（{q['k']} 方法 × {q['n']} 共同 seed）：Q={q['Q']:.2f}，df={q['df']}，"
                             f"p={_p(q['p_value'])}")
                lines.append("")
            # b/c 的 ✓ 指该二值指标为 1（SR 是成功，CR 是碰撞），不是"更好"。
            lines += [f"| A | B | n 共同 | {pm}_A | {pm}_B | b (A=1,B=0) | c (A=0,B=1) | p | p Holm | 方法 |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for r in t["pairwise"]:
                lines.append(f"| {r['A']} | {r['B']} | {r['n_common']} | {r['rate_A']:.2f} | {r['rate_B']:.2f} | "
                             f"{r['b']} | {r['c']} | {_p(r['p_value'])} | {_p(r['p_holm'])} | {r['method']} |")
            lines.append(f"\n注：两两 McNemar 不以 Q 显著为前提（Q 只是总体检验）；Holm 族 = 本表全部 {len(t['pairwise'])} 行"
                         f"（场景 `{scene}` × {pm}），跨指标/跨场景不合并校正。")
        if tests.get("skipped_multi_run"):
            lines.append(f"\n多 run 分组不做配对检验：{', '.join(tests['skipped_multi_run'])}")
    if rep.get("power"):
        pw = rep["power"]
        lines += ["", f"两比例功效估算：p1={pw['p1']} vs p2={pw['p2']}，α={pw['alpha']}，功效 {pw['power']}："
                      f"每臂 {pw['n_per_arm']} 集（Fleiss 校正 {pw['n_per_arm_corrected']}）；配对设计（同 seed）用 McNemar 可更少。"]
    return "\n".join(lines) + "\n"


def write_report(rep, out_dir, stem="stats_report"):
    os.makedirs(out_dir, exist_ok=True)
    md = to_markdown(rep)
    with open(os.path.join(out_dir, stem + ".md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return md


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", nargs=3, action="append", metavar=("METHOD", "SCENE", "EVAL_JSON"), default=[],
                   help="一条评测产物；同 METHOD+SCENE 多次给 = 多个训练种子 run")
    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help=f"逗号分隔，可选 {sorted(METRICS)}")
    p.add_argument("--pair-metrics", default="SR", help="做 McNemar/Cochran 的二值指标，逗号分隔")
    p.add_argument("--out-dir", default=None, help="写 <out-dir>/stats_report.{md,json}；缺省只打印 Markdown")
    p.add_argument("--stem", default="stats_report")
    p.add_argument("--title", default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--power", nargs=2, type=float, metavar=("P1", "P2"), default=None,
                   help="两比例功效估算：每臂集数（可与 --eval 同用，结果附在报表末尾）")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--target-power", type=float, default=0.8)
    a = p.parse_args(argv)
    if not a.eval and a.power is None:
        p.error("至少给一个 --eval 或 --power")
    power = power_two_proportions(a.power[0], a.power[1], a.alpha, a.target_power) if a.power else None
    if not a.eval:
        print(json.dumps(power, ensure_ascii=False))
        return 0
    for _m, _s, path in a.eval:
        if not os.path.exists(path):
            p.error(f"找不到 {path}")
    rep = build_report([tuple(e) for e in a.eval], metrics=[m.strip() for m in a.metrics.split(",") if m.strip()],
                       pair_metrics=[m.strip() for m in a.pair_metrics.split(",") if m.strip()],
                       n_boot=a.n_boot, seed=a.seed, title=a.title)
    rep["power"] = power
    md = write_report(rep, a.out_dir, a.stem) if a.out_dir else to_markdown(rep)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
