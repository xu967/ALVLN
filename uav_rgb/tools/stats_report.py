#!/usr/bin/env python3
"""闭环评测统计汇总（库 + CLI；只用 numpy 与标准库，不装 scipy/rliable）。

读若干 eval.json（sim/tools/eval_closed_loop.py 或 sim/3dgs/eval_3dgs.py 的产物）+ 分组名，输出论文主表
（行 = 方法，列 = 场景 × 指标，每格 `x.xx [lo, hi] (n)`）与配对检验表，Markdown + JSON 各一份。

统计口径（docs/EVAL_STANDARD.md §5.3，分母见 §5.6）：
  二值指标: 比例 + Wilson 95%。越高越好的（SR/task_success/reach_ever/stop_success）分母 = 请求集 episodes_requested，
    失败/跳过集记 0；越低越好的（CR/false_forward/stuck）分母 = 完成集 episodes_done。表里单列「失败集数」；
  连续指标（SPL）: 完成集上的均值 + 百分位 bootstrap 95%；
  同一 (方法, 场景) 多个 run（训练种子）: 各 run 比例的 IQM（Agarwal et al. NeurIPS 2021），区间 = 整个几何有放回重采样的
    百分位 95%（同一几何上各 run 的集一起抽，簇键同下；run 级分层 bootstrap 把同一几何上的集当独立样本，区间比整簇区间窄）；
    跨场景的 IQM（全场景列）区间仍是 run 级分层 bootstrap；
  两方法配对（同 seed 集）: McNemar（b+c<25 精确二项，否则带连续性校正的卡方）；
    多 run 的方法也配对：第 i 个 run 对第 i 个 run，按 (run, seed) 对齐（run 数不同只配前 min 个）；这样的行另加两列：
    几何聚类符号翻转 p（每个几何上各 run 的成功差相加，对几何做双侧符号翻转；≤20 个非零几何精确，否则固定种子蒙特卡洛并写明
    抽样次数）与整簇 bootstrap 的 A−B 率差区间（tools/paired_stats.py，只用标准库）；逐集 McNemar 把同一几何上的各 run 当
    独立样本，只作参考。聚类键 = per_episode 的 geometry_gid，没有就回退为 seed（表里写明）；
  ≥3 个单 run 方法: 另给 Cochran's Q（全体共同 seed）作总体检验；两两 McNemar 照做、不以 Q 显著为前提；
    Holm 族 = 同一场景、同一指标的全部两两行（k 方法 → k(k-1)/2 个 p；只校正逐集 McNemar 的 p）；
  样本量: 两比例功效估算（正态近似 + Fleiss 连续性校正）。
协议一致性（A4）：同一场景（同一列组）内，action_latency_s / heading_guard / rtf / student_cond / dynamics_profile 五个键
  必须一致，缺键 = 旧协议；不一致就报错并列出哪些键、各取什么值；确实要混表（例如有护栏对无护栏）加 --allow-mixed-protocol，
  表头标「混合协议」。原有七个键（metric_version、success_criterion、hold_s、dynamics、vehicle、window_from、task）仍只警告。
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
import paired_stats  # noqa: E402  - 几何聚类的配对检验与区间（标准库）
import pipeline_common  # noqa: E402  - wilson_ci 只此一份


class InputError(ValueError):
    """输入不合法（run 内 seed 重复、未知指标、协议不一致）：命令行干净退出（rc 2），库调用方按 ValueError 处理。"""


class ProtocolMismatch(InputError):
    """同一场景内五个协议键不一致（A4）；allow_mixed_protocol=True 或 --allow-mixed-protocol 放行。"""


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
# A4：不一致就报错的协议键。eval_closed_loop.py 只在给了对应参数（或非默认值）时才写这些键，所以缺键 = 旧协议或默认值。
# cf_sitl_params 含路径，不比较（dynamics_profile 里的哈希已经覆盖参数内容）。
STRICT_PROTOCOL_KEYS = ("action_latency_s", "heading_guard", "rtf", "student_cond", "dynamics_profile")
NO_VALUE = "（无）"
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
def protocol_value(key, v):
    """严格协议键的比较形式：缺键（旧协议）与等价于缺键的默认值（heading_guard=0、student_cond=prev-action）都记 NO_VALUE，
    其余取 str(值)。"""
    if v is None or (key == "heading_guard" and v == 0) or (key == "student_cond" and v == "prev-action"):
        return NO_VALUE
    return str(v)


def resolve_dynamics_profile(e, per):
    """eval.json 的 dynamics_profile（I2-1）：顶层有值时优先用顶层（列表去重排序）；常规 Gazebo 评测顶层不写该键，
    从 per_episode 取（单值取该串，多值拼成排序后的 ' + ' 串触发一致性检查报错）；per_episode 为空且顶层有 cf_sitl_params 时，
    回退用它的 values_sha256[:8] 参与比较。"""
    top = e.get("dynamics_profile")
    if isinstance(top, list):
        profiles = sorted({str(x) for x in top if x})
    elif top:
        profiles = [str(top)]
    else:
        profiles = sorted({str(ep["dynamics_profile"]) for ep in per if isinstance(ep, dict) and ep.get("dynamics_profile")})
        if not profiles and not per and isinstance(e.get("cf_sitl_params"), dict):
            vsha = str(e["cf_sitl_params"].get("values_sha256") or "")[:8]
            if vsha:
                profiles = [vsha]
    if not profiles:
        return None
    return profiles[0] if len(profiles) == 1 else " + ".join(profiles)


def _unique_seeds(*records_lists):
    """按出现顺序提取去重后的未完成 seed（I2-3：先 failed 后 skipped，重复记录不累加）。"""
    seen, out = set(), []
    for records in records_lists:
        for r in records or ():
            if isinstance(r, dict) and r.get("seed") is not None and r["seed"] not in seen:
                seen.add(r["seed"])
                out.append(r["seed"])
    return out


def load_eval(path):
    """eval.json（或含它的目录）-> 报表需要的字段。失败/跳过集不在 per_episode 里，另记数。"""
    if os.path.isdir(path):
        path = os.path.join(path, "eval.json")
    with open(path, encoding="utf-8") as f:
        e = json.load(f)
    per = e.get("per_episode") or []
    check_unique_seeds(per, path)
    hold = e.get("hold_s")
    failed_cnt = e.get("failed_count")
    skipped_cnt = e.get("skipped_count")
    return {
        "path": os.path.abspath(path), "per_episode": per,
        "episodes_done": e.get("episodes_done", len(per)),
        "episodes_requested": e.get("episodes_requested"),
        "failed": len(e.get("failed") or []) if failed_cnt is None else failed_cnt,
        "failed_seeds": _unique_seeds(e.get("failed")),
        "skipped": len(e.get("skipped") or []) if skipped_cnt is None else skipped_cnt,
        "missing_seeds": _unique_seeds(e.get("failed"), e.get("skipped")),
        "hold_s": hold, "metric_version": e.get("metric_version"),
        "success_criterion": e.get("success_criterion"), "dynamics": e.get("dynamics"),
        "vehicle": e.get("vehicle"), "window_from": e.get("window_from"),
        "task": e.get("task"), "route": (e.get("scene") or {}).get("route") if isinstance(e.get("scene"), dict) else None,
        "spl_reference": e.get("spl_reference"),
        **{k: e.get(k) for k in STRICT_PROTOCOL_KEYS},
        "dynamics_profile": resolve_dynamics_profile(e, per),
    }


def check_unique_seeds(per_episode, where):
    """同一个 run 内 seed 必须唯一：{seed: 值} 的写法里后一集会覆盖前一集，配对与比例会静默少算，所以重复就报错（D5）。
    没有 seed 字段的集按序号算，不会重复。不同 run 的同一个 seed 不算重复，配对时按 (run, seed) 对齐。"""
    seen, dups = set(), []
    for i, ep in enumerate(per_episode):
        sd = ep.get("seed", i)
        if sd in seen and sd not in dups:
            dups.append(sd)
        seen.add(sd)
    if dups:
        raise InputError(f"{where}: per_episode 里 seed {sorted(dups)} 重复；同一个 run 内 seed 必须唯一"
                         "（不同 run 的同一个 seed 按 (run, seed) 对齐，不算重复）。要重复评测同一个 seed，请拆成多个 eval.json 当多个 run 给")


def metric_values(per_episode, field):
    """per_episode -> {seed: 值}，None 的集丢掉（如旧 eval.json 没有 spl）。run 内 seed 重复 → ValueError（check_unique_seeds）。"""
    check_unique_seeds(per_episode, "per_episode")
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


def all_failed(ev):
    """请求的集全部未完成（失败或跳过）：完成 0 集、有失败或跳过记录、请求数已知。没有 per_episode 的聚合文件（完成数不是 0）
    与缺这个指标字段的旧文件都不是全失败，它们的格仍是「—」，不能被写成 0。"""
    return ev["episodes_done"] == 0 and (ev["failed"] + ev.get("skipped", 0)) > 0 and bool(ev.get("episodes_requested"))


def missing_seeds_for(ev, done_seeds, denom):
    """需要记 0 的未完成 seed（I2-3）：从去重后的 missing_seeds（先 failed 后 skipped）去掉已完成集，最多补满请求分母缺口
    denom − len(done_seeds)。范围扫描里已满请求名额（done + failed == denom）的多余 skipped 不会混进来。"""
    need = denom - len(done_seeds)
    if need <= 0:
        return []
    raw = ev["missing_seeds"] if "missing_seeds" in ev else ev.get("failed_seeds", ())
    return [sd for sd in raw if sd not in done_seeds][:need]


def run_outcomes(ev, field, requested):
    """一个 run 的 {seed: 值}，配对用。requested（越高越好的二值指标）时未完成（失败或占请求名额的跳过）的 seed 记 False，
    与主表的请求集分母同口径；越低越好的指标未完成集没有观测，不进。run 里没有这个字段（旧文件）而且不是全失败时，
    不给未完成集补 False（没有观测可对）。同一个 seed 既完成又在 failed 里（重试成功）以完成的为准。没有 seed 的空位
    （episodes_requested 大于完成加未完成记录，旧文件里有）配不上对，不进。"""
    out = dict(metric_values(ev["per_episode"], field))
    if requested and (out or all_failed(ev)):
        denom = run_denominator(ev, len(out), True)
        for sd in missing_seeds_for(ev, out, denom):
            out[sd] = False
    return out


def run_denominator(ev, n_done, requested):
    """一个 run 的比例分母：requested 时取 episodes_requested（失败/跳过集记 0），否则取完成集。
    旧 eval.json 没有 episodes_requested 就退回完成集；请求数小于完成数（不应出现）时取完成数，比例不超过 1。"""
    req = ev.get("episodes_requested")
    return max(int(req), n_done) if requested and req else n_done


def _run_value(vals, kind, n=None):
    return (sum(1 for v in vals if v) / (n or len(vals))) if kind == "binary" else (sum(vals) / len(vals))


def summarize_cell(runs, field, kind, n_boot, seed, requested=False):
    """一个 (方法, 场景) 格：1 个 run 给 Wilson / bootstrap 均值；多个 run 给各 run 比例的 IQM + 整簇（几何）重采样区间。
    requested=True（越高越好的二值指标）时比例的分母是请求集；格里的 n 就是分母，分母与完成数不同时另记 n_done。"""
    per_run = [(list(metric_values(r["per_episode"], field).values()), r) for r in runs]
    with_requested = requested and kind == "binary"
    # 没有完成集的 run 丢掉；但越高越好的指标分母是请求集，全部失败的 run（请求 > 0）记 0 留下，格显示 0.00、n = 请求集。
    per_run = [(v, r) for v, r in per_run if v or (with_requested and all_failed(r))]
    if not per_run:
        return None
    denoms = [run_denominator(r, len(v), with_requested) for v, r in per_run]
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
    point = iqm([_run_value(v, kind, d) for (v, _r), d in zip(per_run, denoms)])        # 各 run 比例的 IQM，点估计与以前相同
    clusters, key = cell_clusters([r for _v, r in per_run], field, kind, denoms, requested and kind == "binary")

    def stat(sample):    # 重采样出的簇 -> 各 run 在这些簇上的比例 -> IQM；某个 run 一个簇都没抽到（分母 0）就不算它
        rates = [sum(c[2 * i] for c in sample) / den for i in range(len(per_run))
                 if (den := sum(c[2 * i + 1] for c in sample))]
        return iqm(rates) if rates else None

    lo, hi, _nb = paired_stats.cluster_bootstrap(clusters, stat, n_boot=n_boot, seed=seed)
    if lo is None:      # shortcut: 不足 2 个簇没有簇间方差可估，区间取点（与单 run 区间退化为点是同一个约定）
        lo = hi = point
    ns, dones = sorted(set(denoms)), sorted({len(v) for v, _r in per_run})
    cell = {"point": point, "lo": lo, "hi": hi, "n": ns[0] if len(ns) == 1 else ns, "runs": len(per_run),
            "kind": "iqm_cluster", "cluster_key": key, "n_clusters": len(clusters)}
    if ns != dones:
        cell["n_done"] = dones[0] if len(dones) == 1 else dones
    return cell


def cell_clusters(evs, field, kind, denoms, with_missing):
    """多 run 格的整簇重采样数据：每个簇一个元组 (run0 分子, run0 分母, run1 分子, run1 分母, ...)，返回 (簇列表, 聚类键)。
    簇 = geometry_gid（所有完成集都有且同 seed 一致），否则 seed（同一个 seed 在各 run 里同一簇）。分子分母与点估计同一套：
    二值指标 = 成功数 / 集数，连续指标 = 值之和 / 集数；with_missing（越高越好的二值指标，分母取请求集）时，失败与占名额的跳过集记 0
    落进它所在的几何簇（几何由别的 run 里同 seed 的集给出，查不到就是以 seed 为键的独立簇），记录里没有 seed 的空位各自一个单点簇。"""
    obs_eps = [ep for ev in evs for ep in ev["per_episode"] if ep.get(field) is not None]
    gids_by_seed = {}
    for ev in evs:
        for sd, g in run_geometries(ev).items():
            gids_by_seed.setdefault(sd, set()).add(g)
    use_gid = (bool(obs_eps) and all(ep.get("geometry_gid") is not None for ep in obs_eps)
               and all(len(gs) == 1 for gs in gids_by_seed.values()))
    geom = {sd: next(iter(gs)) for sd, gs in gids_by_seed.items() if len(gs) == 1}

    def cid(sd, gid):
        return ("gid", gid) if use_gid and gid is not None else ("seed", sd)

    acc = []
    for i, ev in enumerate(evs):
        a, done = {}, set()
        for j, ep in enumerate(ev["per_episode"]):
            v = ep.get(field)
            if v is None:
                continue
            sd = ep.get("seed", j)
            done.add(sd)
            e = a.setdefault(cid(sd, ep.get("geometry_gid")), [0.0, 0])
            e[0] += (1 if v else 0) if kind == "binary" else v
            e[1] += 1
        if with_missing:
            ms = missing_seeds_for(ev, done, denoms[i])
            for sd in ms:
                a.setdefault(cid(sd, geom.get(sd)), [0.0, 0])[1] += 1
            for j in range(denoms[i] - len(done) - len(ms)):
                a[("pad", i, j)] = [0.0, 1]
        acc.append(a)
    ids = list(dict.fromkeys(c for a in acc for c in a))
    return [tuple(x for a in acc for x in a.get(c, (0.0, 0))) for c in ids], ("geometry_gid" if use_gid else "seed")


def summarize_all_scenes(runs_by_scene, field, kind, n_boot, seed, requested=False):
    """一个方法跨场景的 IQM：矩阵 runs × scenes，元素 = 该 run 在该场景的指标值（场景间 run 数须相同）。
    分母规则与 summarize_cell 一致（requested 见 uses_requested），同一张表里不留两套分母。"""
    cols = []
    for scene_runs in runs_by_scene.values():
        col = []
        for r in scene_runs:
            vals = list(metric_values(r["per_episode"], field).values())
            den = run_denominator(r, len(vals), requested and kind == "binary")
            col.append(_run_value(vals, kind, den)
                       if (vals or (requested and kind == "binary" and all_failed(r))) else None)   # 全部失败的 run 记 0，与单场景格同口径
        cols.append(col)
    runs = {len(c) for c in cols}
    if len(runs) != 1 or any(v is None for c in cols for v in c):
        return None
    matrix = np.asarray(cols, dtype=float).T                    # runs × scenes
    point, lo, hi = stratified_bootstrap_ci(matrix, n_boot=n_boot, seed=seed)
    return {"point": point, "lo": lo, "hi": hi, "runs": matrix.shape[0], "scenes": matrix.shape[1],
            "kind": "iqm_stratified"}


def run_geometries(ev):
    """一个 run 的 {seed: geometry_gid}；没有 gid 的集不进。"""
    out = {}
    for i, ep in enumerate(ev["per_episode"]):
        g = ep.get("geometry_gid")
        if g is not None:
            out[ep.get("seed", i)] = g
    return out


def pair_units(runs_a, runs_b, field, requested=False):
    """两个方法的配对单元 [(簇, a, b)]（a、b 为 0/1）：第 i 个 run 对第 i 个 run（按 --eval 给出的顺序），每个 run 内按共同 seed 配对。
    键是 (run, seed)，不同 run 的同一个 seed 各是一个单元，不会互相覆盖。requested 见 run_outcomes（失败与跳过集记 0，共同 seed =
    两侧请求 seed 的交集）。簇 = geometry_gid（全部已完成单元都有 geometry_gid 且同 seed 一致；未完成集取同 seed 在别的 run 里的
    geometry_gid，两侧所有 run 都未完成的 seed 单独按 ('seed', seed) 成簇，不把整张表退回 seed），否则整张表回退成 seed。
    返回 (units, 聚类键, 配对的 run 数)。"""
    # shortcut: 两边 run 数不同时只配前 min(nA, nB) 个 run，多出来的 run 不参与；改成按几何取均值再配对，几何差就不再是整数，
    #   精确符号翻转不适用。
    m = min(len(runs_a), len(runs_b))
    gids_by_seed = {}
    for ev in list(runs_a) + list(runs_b):
        for sd, g in run_geometries(ev).items():
            gids_by_seed.setdefault(sd, set()).add(g)
    rows, any_gid, gid_ok = [], False, True
    for i in range(m):
        da = metric_values(runs_a[i]["per_episode"], field)
        db = metric_values(runs_b[i]["per_episode"], field)
        va = run_outcomes(runs_a[i], field, requested)
        vb = run_outcomes(runs_b[i], field, requested)
        ga, gb = run_geometries(runs_a[i]), run_geometries(runs_b[i])
        for s in sorted(set(va) & set(vb)):
            if (s in da and s not in ga) or (s in db and s not in gb):
                gid_ok = False
            gs = gids_by_seed.get(s, set())
            if len(gs) > 1:
                gid_ok = False
            g = next(iter(gs)) if len(gs) == 1 else None
            if g is not None:
                any_gid = True
            rows.append((s, g, int(bool(va[s])), int(bool(vb[s]))))
    use_gid = gid_ok and any_gid
    return [((("gid", g) if use_gid and g is not None else ("seed", s)), a, b) for s, g, a, b in rows], (
        "geometry_gid" if use_gid else "seed"), m


def paired_tests(runs, field, n_boot=2000, seed=0, n_draws=paired_stats.DEFAULT_DRAWS, requested=False):
    """同一场景内方法两两配对 + Holm；runs = {方法: [run, ...]}；≥3 个单 run 方法先 Cochran's Q（全体共同 seed）。
    两个都是单 run：逐集 McNemar（共同 seed）。有多 run 的一方：按 pair_units 配成 (run, seed) 单元，逐集 McNemar 保留作参考，
    行里另加 cluster：几何聚类符号翻转 p 与整簇 bootstrap 的 A−B 率差区间（paired_stats.pair_summary）。
    requested = uses_requested(指标)：越高越好的二值指标一侧基础设施失败的 seed 记 0（run_outcomes），与主表同分母。"""
    names = list(runs)
    single = [n for n in names if len(runs[n]) == 1]
    outcomes = {n: {s: bool(v) for s, v in run_outcomes(runs[n][0], field, requested).items()} for n in single}
    out = {"cochran_q": None, "pairwise": []}
    if len(single) >= 3:
        common = sorted(set.intersection(*(set(outcomes[n]) for n in single)))
        if len(common) >= 2:
            out["cochran_q"] = cochran_q([[int(outcomes[n][s]) for n in single] for s in common])
            out["cochran_q"]["methods"] = single
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if len(runs[a]) == 1 and len(runs[b]) == 1:
                bb, cc, n = paired_outcomes(outcomes[a], outcomes[b])
                if n == 0:
                    continue
                common = set(outcomes[a]) & set(outcomes[b])
                rows.append({"A": a, "B": b, "n_common": n,
                             "rate_A": sum(1 for s in common if outcomes[a][s]) / n,
                             "rate_B": sum(1 for s in common if outcomes[b][s]) / n,
                             **mcnemar(bb, cc)})
                continue
            units, key, m = pair_units(runs[a], runs[b], field, requested)
            if not units:
                continue
            bb = sum(1 for _g, x, y in units if x and not y)
            cc = sum(1 for _g, x, y in units if not x and y)
            n = len(units)
            cs = paired_stats.pair_summary(units, n_boot=n_boot, seed=seed, n_draws=n_draws)
            rows.append({"A": a, "B": b, "n_common": n,
                         "rate_A": sum(x for _g, x, _y in units) / n, "rate_B": sum(y for _g, _x, y in units) / n,
                         **mcnemar(bb, cc),
                         "cluster": {"key": key, "runs": [len(runs[a]), len(runs[b]), m],
                                     **{k: cs[k] for k in ("n_clusters", "n_nonzero", "p_value", "method", "n_draws",
                                                           "diff", "ci_lo", "ci_hi", "n_boot", "n_boot_valid")}}})
    if rows:
        for r, p in zip(rows, holm([r["p_value"] for r in rows])):
            r["p_holm"] = p
    out["pairwise"] = rows
    return out


def build_report(entries, metrics=DEFAULT_METRICS, pair_metrics=("SR",), n_boot=2000, seed=0, title=None,
                 n_draws=paired_stats.DEFAULT_DRAWS, allow_mixed_protocol=False):
    """entries: [(method, scene, eval.json 路径)]。返回可直接 json.dump 的 dict（表 + 检验 + 口径）。
    n_draws：多 run 配对里非零几何超过 paired_stats.EXACT_MAX_CLUSTERS 时蒙特卡洛的抽样次数。
    同一场景内 STRICT_PROTOCOL_KEYS 不一致：默认抛 ProtocolMismatch（列出键、取值与方法）；allow_mixed_protocol=True 放行，
    结果里 mixed_protocol = {场景: [不一致的键]}，表头标「混合协议」。"""
    for m in tuple(metrics) + tuple(pair_metrics):
        if m not in METRICS:
            raise InputError(f"未知指标 {m}，可选 {sorted(METRICS)}")
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
        scene_runs = {m: groups[m][scene] for m in methods if groups[m].get(scene)}
        if len(scene_runs) >= 2:
            paired[scene] = {pm: paired_tests(scene_runs, METRICS[pm][0], n_boot, seed, n_draws, uses_requested(pm))
                             for pm in pair_metrics}
            paired[scene]["skipped_multi_run"] = []     # 键保留给旧读者：多 run 的方法现在也配对，恒为空
    protocol, mixed_protocol, problems = {}, {}, []
    for scene in scenes:
        keys = ("metric_version", "success_criterion", "hold_s", "dynamics", "vehicle", "window_from", "task")
        vals = {k: sorted({str(ev[k]) for m in methods for ev in groups[m].get(scene, [])}) for k in keys}
        bad = {}
        for k in STRICT_PROTOCOL_KEYS:
            by_value = {}
            for m in methods:
                for ev in groups[m].get(scene, []):
                    pv = protocol_value(k, ev.get(k))
                    parts = pv.split(" + ") if k == "dynamics_profile" and " + " in pv else (pv,)
                    for part in parts:
                        by_value.setdefault(part, set()).add(m)
            if set(by_value) != {NO_VALUE}:         # 整个场景都缺键 = 旧协议，不进口径行
                vals[k] = sorted(by_value)
            if len(by_value) > 1:
                bad[k] = by_value
        protocol[scene] = {"values": vals, "inconsistent": [k for k, v in vals.items() if len(v) > 1]}
        if bad:
            mixed_protocol[scene] = list(bad)
            problems.append(f"场景 {scene}：" + "；".join(
                f"{k}：" + " / ".join(f"{v} [{', '.join(sorted(ms))}]" for v, ms in sorted(by_value.items()))
                for k, by_value in bad.items()))
    if problems and not allow_mixed_protocol:
        raise ProtocolMismatch("协议不一致（A4）：" + "。".join(problems)
                               + "。同一场景内这些键必须相同才能比较，缺键表示旧协议；确实要混表（例如有护栏对无护栏）"
                                 "加 --allow-mixed-protocol，表头会标「混合协议」")
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
        "mixed_protocol": mixed_protocol,        # {场景: [不一致的协议键]}，只有 allow_mixed_protocol 时才可能非空
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


def _cluster_p(c):
    return _p(c["p_value"]) + (f" (MC {c['n_draws']})" if c["method"] == "monte_carlo" else "")


def _cluster_diff(c):
    return f"{c['diff']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]"


def to_markdown(rep):
    lines = []
    if rep.get("title"):
        lines += [f"# {rep['title']}", ""]
    versions = rep.get("metric_versions") or []
    mixed = (f" ⚠ metric_version 混合：{' / '.join(versions)}（v1 过线后继续飞、碰撞可能发生在过线之后；"
             "v2 成功线即停；SR/CR 不可直接比，见 EVAL_STANDARD §5）" if len(versions) > 1 else "")
    mp = rep.get("mixed_protocol") or {}
    mixed += (" ⚠ 混合协议（--allow-mixed-protocol）：" + "；".join(f"{sc} 的 {', '.join(ks)}" for sc, ks in mp.items())
              + " 在各行不同，行间差异不能全归给方法") if mp else ""
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
                  f"SPL 为完成集上的百分位 bootstrap 均值区间（B={rep['n_boot']}）；多 run 的格为各 run 比例的 IQM（Agarwal 2021），"
                  "区间按几何整簇重采样（同一几何上各 run 的集一起抽，没有 geometry_gid 时按 seed 聚），每个 run 按各自的分母算比例"
                  + ("；全场景 IQM 的区间仍是 run 级分层 bootstrap" if rep["all_scenes"] else "")
                  + f"。n<{PILOT_N} 的格只算 pilot。"
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
            cl = any("cluster" in r for r in t["pairwise"])     # 有多 run 的行才加聚类两列；单 run 的表与以前逐字相同
            head = ["A", "B", "n 共同", f"{pm}_A", f"{pm}_B", "b (A=1,B=0)", "c (A=0,B=1)",
                    "p 逐集 McNemar（参考）" if cl else "p", "p Holm（逐集）" if cl else "p Holm", "方法"]
            if cl:
                head += ["p 几何聚类符号翻转", "差值 [95% 整簇区间]"]
            lines += ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
            for r in t["pairwise"]:
                cells = [r["A"], r["B"], str(r["n_common"]), f"{r['rate_A']:.2f}", f"{r['rate_B']:.2f}",
                         str(r["b"]), str(r["c"]), _p(r["p_value"]), _p(r["p_holm"]), r["method"]]
                if cl:
                    c = r.get("cluster")
                    cells += [_cluster_p(c), _cluster_diff(c)] if c else ["—", "—"]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append(f"\n注：两两 McNemar 不以 Q 显著为前提（Q 只是总体检验）；Holm 族 = 本表全部 {len(t['pairwise'])} 行"
                         f"（场景 `{scene}` × {pm}），跨指标/跨场景不合并校正。"
                         + (" 越高越好的指标：一侧基础设施失败或跳过的 seed 记失败（失败集记 0、跳过集记 0，与主表同分母），n 共同 = 两侧请求 seed 的交集。"
                            if uses_requested(pm) else
                            " 越低越好的指标：失败与跳过集没有观测，只配两侧都完成的 seed。"))
            if cl:
                first = next(r["cluster"] for r in t["pairwise"] if "cluster" in r)
                lines += ["", "多 run 的行：第 i 个 run 对第 i 个 run，按 (run, seed) 对齐，一个 (run, seed) 是一个单元。"
                              "几何聚类 p = 每个几何上各 run 的 A、B 成功数之差相加，对非零几何做双侧符号翻转检验"
                              f"（非零几何不超过 {paired_stats.EXACT_MAX_CLUSTERS} 个时精确，否则固定种子蒙特卡洛，抽样次数写在该格）；"
                              f"区间 = 整个几何有放回重采样（B={first['n_boot']}）的 A 率减 B 率的百分位 95% 区间。"
                              "逐集 McNemar 把同一几何上的各 run 当独立样本，p 低于几何聚类 p，只作参考；"
                              "Holm 只校正逐集 McNemar 的 p，几何聚类 p 不做多重比较校正。"]
                for r in t["pairwise"]:
                    c = r.get("cluster")
                    if not c:
                        continue
                    ra, rb, m = c["runs"]
                    note = ("" if ra == rb else f"，两边 run 数不同，只配前 {m} 个")
                    key = ("geometry_gid" if c["key"] == "geometry_gid"
                           else "seed（eval.json 没有 geometry_gid 或两边不一致，回退为 seed）")
                    lines.append(f"- {r['A']}（{ra} run）对 {r['B']}（{rb} run）：配 {m} 个 run{note}，{r['n_common']} 个单元，"
                                 f"{c['n_clusters']} 个簇，聚类键 {key}")
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
    p.add_argument("--n-draws", type=int, default=paired_stats.DEFAULT_DRAWS,
                   help="多 run 配对里非零几何超过 %d 个时，符号翻转蒙特卡洛的抽样次数" % paired_stats.EXACT_MAX_CLUSTERS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-mixed-protocol", action="store_true",
                   help="同一场景内 action_latency_s / heading_guard / rtf / student_cond / dynamics_profile 不一致时不报错（默认报错），"
                        "表头标「混合协议」")
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
    try:
        rep = build_report([tuple(e) for e in a.eval], metrics=[m.strip() for m in a.metrics.split(",") if m.strip()],
                           pair_metrics=[m.strip() for m in a.pair_metrics.split(",") if m.strip()],
                           n_boot=a.n_boot, seed=a.seed, title=a.title, n_draws=a.n_draws,
                           allow_mixed_protocol=a.allow_mixed_protocol)
    except InputError as e:      # 重复 seed、未知指标等输入问题：干净退出（rc 2），不抛 traceback；别的 ValueError 是 bug，照常抛
        p.error(str(e))
    rep["power"] = power
    md = write_report(rep, a.out_dir, a.stem) if a.out_dir else to_markdown(rep)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
