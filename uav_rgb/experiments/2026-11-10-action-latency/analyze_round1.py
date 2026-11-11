#!/usr/bin/env python3
"""第一轮（2026-11-10）标定与验收的事后分析：迟到步清单与集中度、稳态前飞位移差的归因、验收 2 的逐集配对、森林三档读数。

只读 $OUT 下已有的 episodes/ 与 eval.json，不起 gz、不碰实验目录里的其它文件；结果写 $OUT/analysis-round1.md。
运行：.venv-tools/bin/python analyze_round1.py [输出目录，默认 $UAVRGB_DATA/exp-action-latency-20261110] [--note 说明]
量的定义沿用 summarize.py（同目录）：生效时长 = (oracle.action_end_ns − action_start_ns)/1e9 − actions.dispatch_lag_s；
稳态前飞位移 = 本步与上一步都是 FORWARD_SLOW 且本步无碰撞的步的 displacement_m。
置信区间：按 seed 成对重抽样（K=1 与 K=3 的 train 分区 seed 相同，K=3 第 0 块 = 同 seed 的 base 柱阵），重抽 2000 次，百分位 2.5 / 97.5。
"""
import argparse
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize as sm  # noqa: E402

ACTION_NAMES = {0: "SAFE_NEUTRAL", 1: "FORWARD_SLOW", 2: "YAW_LEFT", 3: "YAW_RIGHT"}
FWD, SAFE = 1, 0
TURNS = (2, 3)
GOALS = [6.0, 11.0, 16.0]          # K=3 各块的成功线 x（scene_course：6.0 + 5.0·k）
STEP_BINS = ((0, 3), (4, 20), (21, 60), (61, 10 ** 6))   # 迟到步按步号分组：起飞后头 4 步 / 前 20 步 / 中段 / 后段
SEG = 10                           # 位移按 10 步一段列
COMMON_LO, COMMON_HI = 3, 30       # 主会话指定的共同步段（K=1 的集 30–37 步，K=3 的集 80–120 步）
N_BOOT = 2000
BOOT_SEED = 20261111


def name(a):
    return ACTION_NAMES.get(a, str(a))


def block_of(x):
    """K=3 的位置所在块：x < 6.0 → 0，< 11.0 → 1，其余 2（以成功线分块，与 long-course 汇总的块判据同口径）。"""
    if x is None:
        return None
    return sum(1 for g in GOALS[:-1] if x >= g)


# ---------------------------------------------------------------- 迟到步清单与集中度
def late_rows(arm, edir, ctx=2):
    """迟到步（actions.late_dispatch 为 true）逐条：集、步号、lag、该步动作、前后各 ctx 步的 lag、位姿。"""
    rows = []
    for seed, actions, oracle, man in sm.episode_records(edir):
        lags = [a.get("dispatch_lag_s") for a in actions]
        L = man.get("action_latency_s")
        for i, a in enumerate(actions):
            if not a.get("late_dispatch"):
                continue
            pose = a.get("pose_before") or {}
            rows.append({"arm": arm, "seed": seed, "step": i, "lag": lags[i], "L": L, "action": a.get("action_executed"),
                         "before": lags[max(0, i - ctx):i], "after": lags[i + 1:i + 1 + ctx],
                         "x": pose.get("x"), "y": pose.get("y"), "n_steps": len(actions)})
    return rows


def breakdown(edir):
    """全部步按 步号分组 / 动作 / 块 的 [迟到数, 总数]。没带 --action-latency 的臂没有 late_dispatch，按 0 迟到算。"""
    by_step = {b: [0, 0] for b in STEP_BINS}
    by_action, by_block = {}, {}
    for seed, actions, oracle, man in sm.episode_records(edir):
        for i, a in enumerate(actions):
            late = 1 if a.get("late_dispatch") else 0
            for b in STEP_BINS:
                if b[0] <= i <= b[1]:
                    by_step[b][0] += late
                    by_step[b][1] += 1
            cell = by_action.setdefault(a.get("action_executed"), [0, 0])
            cell[0] += late
            cell[1] += 1
            blk = block_of((a.get("pose_before") or {}).get("x"))
            cell = by_block.setdefault(blk, [0, 0])
            cell[0] += late
            cell[1] += 1
    return {"step": by_step, "action": by_action, "block": by_block}


def overshoot_ms(row):
    """lag 超出迟到阈值（L + 0.02 s）的毫秒数。"""
    return (row["lag"] - (row["L"] + sm.LATE_TOL_S)) * 1000.0


def fmt_lags(xs):
    return " ".join(f"{x:.3f}" for x in xs) if xs else "-"


def late_section(arms):
    """arms = [(标签, 目录)]。"""
    lines = ["## 1. 迟到步清单（lag = 观测戳到派发的仿真秒；阈值 = L + 0.02 s；超出 = lag − 阈值）", ""]
    allrows = []
    for label, edir in arms:
        allrows += late_rows(label, edir)
    lines += ["| 臂 | seed | 步 | lag s | 超出阈值 ms | 该步动作 | 前 2 步 lag | 后 2 步 lag | x m | 块 | 本集步数 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        x = r["x"]
        lines.append(f"| {r['arm']} | {r['seed']} | {r['step']} | {r['lag']:.3f} | {overshoot_ms(r):.0f} | {name(r['action'])} | "
                     f"{fmt_lags(r['before'])} | {fmt_lags(r['after'])} | {'-' if x is None else f'{x:.2f}'} | "
                     f"{'-' if x is None else block_of(x)} | {r['n_steps']} |")
    lines.append("")
    lines += ["### 集中度（迟到数 / 总数）", "",
              "| 臂 | 步 0–3 | 步 4–20 | 步 21–60 | 步 61 起 | FORWARD | YAW（左+右） | SAFE | 块 0 | 块 1 | 块 2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(c):
        return f"{c[0]}/{c[1]}（{100.0 * c[0] / c[1]:.1f}%）" if c and c[1] else "-"
    for label, edir in arms:
        b = breakdown(edir)
        yaw = [sum(b["action"].get(k, [0, 0])[0] for k in TURNS), sum(b["action"].get(k, [0, 0])[1] for k in TURNS)]
        lines.append(f"| {label} | " + " | ".join(cell(b["step"][s]) for s in STEP_BINS) + f" | {cell(b['action'].get(FWD))} | {cell(yaw)} | "
                     f"{cell(b['action'].get(SAFE))} | " + " | ".join(cell(b["block"].get(k)) for k in (0, 1, 2)) + " |")
    lines.append("")
    amounts = {}
    for r in allrows:
        amounts.setdefault(r["arm"], []).append(overshoot_ms(r))
    lines += ["### 超出阈值的幅度", ""]
    for label, _ in arms:
        xs = sorted(amounts.get(label, []))
        if xs:
            lines.append(f"- {label}：{len(xs)} 步；超出 ≤ 5 ms {sum(1 for x in xs if x <= 5.0)}、≤ 10 ms {sum(1 for x in xs if x <= 10.0)}、"
                         f"≤ 20 ms {sum(1 for x in xs if x <= 20.0)}、最大 {xs[-1]:.0f} ms（中位 {statistics.median(xs):.0f} ms）；位姿话题 5 ms 一格。")
    lines.append("")
    return lines, allrows


# ---------------------------------------------------------------- 稳态前飞位移差的归因
def fwd_samples(actions):
    """[(步号, 上一步动作, displacement_m, 连续前进位置 r)]：本步 FORWARD、无碰撞、有位移的样本；第 0 步没有上一步，不作样本，但计入连续前进的计数。
    r = 本步是连续 FORWARD 的第几个（含本步）：r=1 是转向 / SAFE 之后的第一个前进，r≥2 的上一步必是前进。"""
    out, run = [], 0
    for i, a in enumerate(actions):
        run = run + 1 if a.get("action_executed") == FWD else 0
        if i == 0:
            continue
        if a.get("action_executed") == FWD and not a.get("collision") and a.get("displacement_m") is not None:
            out.append((i, actions[i - 1].get("action_executed"), a["displacement_m"], run))
    return out


def per_episode_samples(edir):
    """{seed: [(步号, 上一步动作, 位移, 连续前进位置)]}"""
    return {seed: fwd_samples(actions) for seed, actions, _o, _m in sm.episode_records(edir) if actions}


def select(samples, lo=0, hi=10 ** 6, prev=None, run_lo=1, run_hi=10 ** 6):
    """prev：None 不限；"fwd" 上一步前进；"turn" 上一步转向；"safe" 上一步 SAFE。run_lo / run_hi 限连续前进位置 r。"""
    want = {None: None, "fwd": (FWD,), "turn": TURNS, "safe": (SAFE,)}[prev]
    return [d for i, p, d, r in samples if lo <= i <= hi and (want is None or p in want) and run_lo <= r <= run_hi]


def pooled(per_ep, **kw):
    xs = [d for s in per_ep.values() for d in select(s, **kw)]
    return (sum(xs) / len(xs) if xs else float("nan")), len(xs)


def boot_rel(a_eps, b_eps, n_boot=N_BOOT, seed=BOOT_SEED, **kw):
    """b 对 a 的相对差 Σb/Σa − 1 的点估计与 95% 区间；按 seed 成对重抽样（只用两边都有样本的 seed）。返回 (rel, lo, hi, 成对 seed 数)。"""
    stats = {}
    for s in sorted(set(a_eps) & set(b_eps)):
        xa, xb = select(a_eps[s], **kw), select(b_eps[s], **kw)
        if xa and xb:
            stats[s] = (sum(xa), len(xa), sum(xb), len(xb))
    if not stats:
        return float("nan"), float("nan"), float("nan"), 0

    def rel(ss):
        sa, na = sum(stats[s][0] for s in ss), sum(stats[s][1] for s in ss)
        sb, nb = sum(stats[s][2] for s in ss), sum(stats[s][3] for s in ss)
        return (sb / nb) / (sa / na) - 1.0
    keys = list(stats)
    rng = random.Random(seed)
    draws = sorted(rel([rng.choice(keys) for _ in keys]) for _ in range(n_boot))
    return rel(keys), draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1], len(keys)


RUN_STRATA = ((2, 2, "r=2"), (3, 3, "r=3"), (4, 10 ** 6, "r≥4（巡航）"))   # 稳态（上一步是前进）按连续前进位置分层
CRUISE_MIN_RUN = 4                                                      # 巡航 = 本步及前 3 步都是 FORWARD


def std_rel(a_eps, b_eps, lo=COMMON_LO, hi=COMMON_HI, strata=RUN_STRATA, n_boot=N_BOOT, seed=BOOT_SEED):
    """b 按 a 的连续前进位置构成（strata）加权后，对 a 的相对差（直接标准化）：点估计与按 seed 成对重抽样的 95% 区间。
    只用步号在 [lo, hi]、上一步是前进的样本；某层 b 没有样本时该层从两边的加权里去掉。
    返回 (rel, 区间下界, 区间上界, 成对 seed 数, a 的各层占比, b 的各层占比, 标准化后 b 的均值, a 的均值)。"""
    per = {}
    for s in sorted(set(a_eps) & set(b_eps)):
        cells = []
        for r_lo, r_hi, _lbl in strata:
            xa = select(a_eps[s], lo=lo, hi=hi, prev="fwd", run_lo=r_lo, run_hi=r_hi)
            xb = select(b_eps[s], lo=lo, hi=hi, prev="fwd", run_lo=r_lo, run_hi=r_hi)
            cells.append((sum(xa), len(xa), sum(xb), len(xb)))
        if any(c[1] for c in cells) and any(c[3] for c in cells):
            per[s] = cells
    nan = float("nan")
    if not per:
        return nan, nan, nan, 0, [], [], nan, nan

    def parts(ss):
        agg = [[0.0, 0, 0.0, 0] for _ in strata]
        for s in ss:
            for j, c in enumerate(per[s]):
                for k in range(4):
                    agg[j][k] += c[k]
        valid = [j for j, c in enumerate(agg) if c[1] and c[3]]
        tot_n = sum(agg[j][1] for j in valid)
        if not valid or not tot_n:
            return nan, nan, agg
        mean_a = sum(agg[j][1] / tot_n * agg[j][0] / agg[j][1] for j in valid)
        mean_b = sum(agg[j][1] / tot_n * agg[j][2] / agg[j][3] for j in valid)
        return mean_a, mean_b, agg

    def rel(ss):
        ma, mb, _agg = parts(ss)
        return mb / ma - 1.0 if ma == ma and ma else nan
    keys = list(per)
    rng = random.Random(seed)
    draws = sorted(d for d in (rel([rng.choice(keys) for _ in keys]) for _ in range(n_boot)) if d == d)
    ma, mb, agg = parts(keys)
    na, nb = sum(c[1] for c in agg), sum(c[3] for c in agg)
    share_a = [c[1] / na if na else nan for c in agg]
    share_b = [c[3] / nb if nb else nan for c in agg]
    lo_ci = draws[int(0.025 * len(draws))] if draws else nan
    hi_ci = draws[max(int(0.975 * len(draws)) - 1, 0)] if draws else nan
    return rel(keys), lo_ci, hi_ci, len(keys), share_a, share_b, mb, ma


def decompose(k1_eps, k3_eps):
    """K=3 对 K=1 的稳态前飞位移相对差，口径一路收窄：门的口径（全部步）→ 共同步段 → 按 K=1 的连续前进构成加权 → 只比巡航步。
    返回 {口径名: (rel, lo, hi, n_pairs)} 与标准化结果。"""
    out = {"all": boot_rel(k1_eps, k3_eps, prev="fwd"),
           "common": boot_rel(k1_eps, k3_eps, lo=COMMON_LO, hi=COMMON_HI, prev="fwd"),
           "cruise": boot_rel(k1_eps, k3_eps, lo=COMMON_LO, hi=COMMON_HI, prev="fwd", run_lo=CRUISE_MIN_RUN),
           "cruise_all": boot_rel(k1_eps, k3_eps, prev="fwd", run_lo=CRUISE_MIN_RUN)}
    out["std"] = std_rel(k1_eps, k3_eps)
    return out


def fmt_mean(mean, n, nd=4):
    return "-" if not n else f"{mean:.{nd}f}（{n}）"


def fmt_rel(r, lo=None, hi=None):
    if r != r:
        return "-"
    ci = "" if lo is None or lo != lo else f" [{100 * lo:+.1f}%, {100 * hi:+.1f}%]"
    return f"{100 * r:+.1f}%{ci}"


def segment_table(k1_eps, k3_eps, width=SEG):
    """稳态（上一步也是 FORWARD）前飞位移按步号分段；返回 markdown 行。"""
    last = max([i for eps in (k1_eps, k3_eps) for s in eps.values() for i, _p, _d, _r in s] or [0])
    segs = [(lo, lo + width - 1) for lo in range(0, last + 1, width)]
    head = "| 臂 | " + " | ".join(f"步 {lo}–{hi}" for lo, hi in segs) + " | 全部 | 步 3–30 |"
    lines = [head, "|---|" + "---|" * (len(segs) + 2)]
    for label, eps in (("K=1", k1_eps), ("K=3", k3_eps)):
        cells = [fmt_mean(*pooled(eps, lo=lo, hi=hi, prev="fwd")) for lo, hi in segs]
        cells += [fmt_mean(*pooled(eps, prev="fwd")), fmt_mean(*pooled(eps, lo=COMMON_LO, hi=COMMON_HI, prev="fwd"))]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def attribution_section(L, k1_dir, k3_dir):
    k1, k3 = per_episode_samples(k1_dir), per_episode_samples(k3_dir)
    lines = [f"### L = {L:.2f} s", "",
             "稳态前飞位移按 10 步一段（单元格：池化均值 m/步（样本数））：", ""]
    lines += segment_table(k1, k3) + [""]
    all_rel = boot_rel(k1, k3, prev="fwd")
    common_rel = boot_rel(k1, k3, lo=COMMON_LO, hi=COMMON_HI, prev="fwd")
    lines += ["K=3 对 K=1 的相对差（稳态；括号内是按 seed 成对重抽样的 95% 区间）：", "",
              "| 范围 | K=1 m/步（样本数） | K=3 m/步（样本数） | 相对差 [95% 区间] | 成对 seed 数 |", "|---|---|---|---|---|"]
    for label, kw, rel in (("全部步（门的口径）", {}, all_rel), (f"共同步段 {COMMON_LO}–{COMMON_HI}", {"lo": COMMON_LO, "hi": COMMON_HI}, common_rel)):
        lines.append(f"| {label} | {fmt_mean(*pooled(k1, prev='fwd', **kw))} | {fmt_mean(*pooled(k3, prev='fwd', **kw))} | "
                     f"{fmt_rel(*rel[:3])} | {rel[3]} |")
    lines += ["", "按「上一步是什么动作」分层（本步都是无碰撞的 FORWARD；前进后前进 = 门的稳态口径）：", "",
              "| 范围 | 上一步 | K=1 m/步（样本数） | K=3 m/步（样本数） | 相对差 [95% 区间] |", "|---|---|---|---|---|"]
    for rlabel, kw in (("全部步", {}), (f"步 {COMMON_LO}–{COMMON_HI}", {"lo": COMMON_LO, "hi": COMMON_HI})):
        for plabel, prev in (("前进", "fwd"), ("转向", "turn"), ("SAFE", "safe")):
            a, b = pooled(k1, prev=prev, **kw), pooled(k3, prev=prev, **kw)
            if not a[1] and not b[1]:
                continue
            rel = boot_rel(k1, k3, prev=prev, **kw)
            lines.append(f"| {rlabel} | {plabel} | {fmt_mean(*a)} | {fmt_mean(*b)} | {fmt_rel(*rel[:3])} |")
    lines.append("")
    lines += run_strata_lines(k1, k3)
    dec_lines, dec = decomposition_lines(L, k1, k3)
    lines += dec_lines
    return lines, {"all": all_rel, "common": common_rel, **dec}


def run_strata_lines(k1, k3):
    """步 3–30 的前进步按连续前进位置 r 分层：r=1 是转向 / SAFE 之后的第一个前进（起步，不算稳态）；r≥2 合起来是门的稳态口径。"""
    lines = [f"按连续前进位置 r 分层（步 {COMMON_LO}–{COMMON_HI}；r=1 是转向或 SAFE 之后的第一个前进，起步，不属于门的稳态口径；占比 = 占 r≥2 的比例）：", "",
             "| r | K=1 m/步（样本数；占比） | K=3 m/步（样本数；占比） | 相对差 [95% 区间] |", "|---|---|---|---|"]
    kw = {"lo": COMMON_LO, "hi": COMMON_HI}
    n1 = sum(len(select(s, run_lo=2, **kw)) for s in k1.values())
    n3 = sum(len(select(s, run_lo=2, **kw)) for s in k3.values())
    for r_lo, r_hi, label in ((1, 1, "r=1（起步）"),) + tuple(RUN_STRATA):
        a, b = pooled(k1, run_lo=r_lo, run_hi=r_hi, **kw), pooled(k3, run_lo=r_lo, run_hi=r_hi, **kw)
        rel = boot_rel(k1, k3, run_lo=r_lo, run_hi=r_hi, **kw)
        sh = (lambda n, tot: f"；{100.0 * n / tot:.0f}%" if r_lo >= 2 and tot else "")
        lines.append(f"| {label} | {a[0]:.4f}（{a[1]}{sh(a[1], n1)}） | {b[0]:.4f}（{b[1]}{sh(b[1], n3)}） | {fmt_rel(*rel[:3])} |")
    lines.append("")
    return lines


def decomposition_lines(L, k1, k3):
    """相对差分解：门口径 → 共同步段 3–30 → 按 K=1 的连续前进构成加权；另列巡航步（r≥4）。"""
    dec = decompose(k1, k3)
    std = dec["std"]
    common_k1, common_k3 = pooled(k1, prev="fwd", lo=COMMON_LO, hi=COMMON_HI), pooled(k3, prev="fwd", lo=COMMON_LO, hi=COMMON_HI)
    all_k1, all_k3 = pooled(k1, prev="fwd"), pooled(k3, prev="fwd")
    cru_k1 = pooled(k1, prev="fwd", lo=COMMON_LO, hi=COMMON_HI, run_lo=CRUISE_MIN_RUN)
    cru_k3 = pooled(k3, prev="fwd", lo=COMMON_LO, hi=COMMON_HI, run_lo=CRUISE_MIN_RUN)
    all_r, common_r, std_r = dec["all"][0], dec["common"][0], std[0]
    step_pp = 100.0 * (common_r - all_r)
    comp_pp = 100.0 * (std_r - common_r)
    lines = [f"相对差分解（L={L:.2f}；K=3 对 K=1，稳态前飞位移）：", "",
             "| 口径 | K=1 m/步（样本数） | K=3 m/步（样本数） | 相对差 [95% 区间] | 比上一行变化 |", "|---|---|---|---|---|",
             f"| 门的口径：全部步，上一步前进 | {fmt_mean(*all_k1)} | {fmt_mean(*all_k3)} | {fmt_rel(*dec['all'][:3])} | - |",
             f"| 只比共同步段 {COMMON_LO}–{COMMON_HI} | {fmt_mean(*common_k1)} | {fmt_mean(*common_k3)} | {fmt_rel(*dec['common'][:3])} | {step_pp:+.1f} 个百分点（步段 / 电量） |",
             f"| 再按 K=1 的 r=2 / r=3 / r≥4 构成给 K=3 加权 | {std[7]:.4f}（{common_k1[1]}） | {std[6]:.4f}（{common_k3[1]}） | {fmt_rel(*std[:3])} | {comp_pp:+.1f} 个百分点（连续前进构成） |",
             f"| 只比巡航步（r≥4），步 {COMMON_LO}–{COMMON_HI} | {fmt_mean(*cru_k1)} | {fmt_mean(*cru_k3)} | {fmt_rel(*dec['cruise'][:3])} | - |",
             f"| 只比巡航步（r≥4），全部步 | {fmt_mean(*pooled(k1, prev='fwd', run_lo=CRUISE_MIN_RUN))} | {fmt_mean(*pooled(k3, prev='fwd', run_lo=CRUISE_MIN_RUN))} | {fmt_rel(*dec['cruise_all'][:3])} | - |",
             ""]
    if std[4] and std[5]:
        lines += [f"r≥2 的步在 步 {COMMON_LO}–{COMMON_HI} 里的构成（r=2 / r=3 / r≥4）：K=1 " + " / ".join(f"{100 * x:.0f}%" for x in std[4])
                  + "，K=3 " + " / ".join(f"{100 * x:.0f}%" for x in std[5]) + "。", ""]
    lines += [f"一句话：L={L:.2f} 门口径相对差 {100 * all_r:+.1f}%；收窄到共同步段后 {100 * common_r:+.1f}%，步段 / 电量只解释 {abs(step_pp):.1f} 个百分点"
              f"（占 {100 * abs(step_pp) / abs(100 * all_r):.0f}%）；再按连续前进构成加权后 {100 * std_r:+.1f}%，构成再解释 {abs(comp_pp):.1f} 个百分点；"
              f"巡航步（r≥4）直接比是 {100 * dec['cruise'][0]:+.1f}%。", ""] if all_r == all_r and all_r else []
    return lines, {"std": std, "cruise": dec["cruise"], "cruise_all": dec["cruise_all"]}


def eff_stats(edir, lo=0, hi=10 ** 6):
    """生效时长的逐步池化均值（可限步段）。"""
    xs = []
    for seed, actions, oracle, man in sm.episode_records(edir):
        for i, (a, o) in enumerate(zip(actions, oracle)):
            if lo <= i <= hi and a.get("dispatch_lag_s") is not None:
                xs.append((o["action_end_ns"] - o["action_start_ns"]) / 1e9 - a["dispatch_lag_s"])
    return (sum(xs) / len(xs) if xs else float("nan")), len(xs)


def elasticity_text(k1_016, k1_020):
    """位移对生效时长的弹性：同一批 K=1 教师 seed 在 L=0.16 与 0.20 的稳态位移与生效时长的对数比（两个 L 之间只有 L 不同）。"""
    d16, _ = pooled(per_episode_samples(k1_016), prev="fwd")
    d20, _ = pooled(per_episode_samples(k1_020), prev="fwd")
    e16, _ = eff_stats(k1_016)
    e20, _ = eff_stats(k1_020)
    if any(x != x or x <= 0 for x in (d16, d20, e16, e20)) or e16 == e20:
        return float("nan"), []
    el = math.log(d20 / d16) / math.log(e20 / e16)
    return el, [f"K=1 教师稳态位移 {d16:.4f} → {d20:.4f} m/步（{100 * (d20 / d16 - 1):+.1f}%），逐步生效时长均值 {e16:.4f} → {e20:.4f} s（{100 * (e20 / e16 - 1):+.1f}%），"
                f"弹性 = ln(位移比) / ln(时长比) = {el:.2f}。"]


def duration_explains(L, k1_dir, k3_dir, elasticity):
    """K=3 与 K=1 的生效时长相对差乘弹性 = 时长差能解释的位移相对差。"""
    e1, _ = eff_stats(k1_dir)
    e3, _ = eff_stats(k3_dir)
    if e1 != e1 or e3 != e3 or elasticity != elasticity:
        return "-"
    d = e3 / e1 - 1.0
    return f"L={L:.2f}：K=3 对 K=1 逐步生效时长 {100 * d:+.2f}%，按弹性 {elasticity:.2f} 只能解释位移 {100 * d * elasticity:+.2f}%。"


# ---------------------------------------------------------------- 验收 2 的逐集读数
def outcome(e, seed):
    """eval.json 里该 seed 的结局文字。"""
    for x in e.get("per_episode", []):
        if int(x["seed"]) == seed:
            if x.get("success"):
                return "成功"
            return "碰撞" if x.get("collision") else ("步数用完未过线" if x.get("timed_out") or x.get("steps", 0) >= (e.get("steps") or 10 ** 9) else "未过线")
    for f in e.get("failed", []):
        if int(f["seed"]) == seed:
            return "迟到（基础设施失败）" if f.get("late_dispatch_steps") else "采集失败"
    return "无记录"


def item2_readout(L=0.16):
    new_dir = sm.acc_dir(L, 1, 1)
    refs = (("下午", sm.REF_K1_AFTERNOON_S1), ("晚上对照", sm.REF_K1_EVENING_S1))
    lines = ["## 4. 验收 2 的逐集读数（学生 calswa 种子 1 + 护栏 2，K=1，20 个冻结 seed，L=0.16）", ""]
    new_e = sm.read_json(os.path.join(new_dir, "eval.json"))
    evs = {lbl: sm.read_json(os.path.join(p, "eval.json")) for lbl, p in refs}
    lines += ["| seed | 下午 | 晚上对照 | 新口径 L=0.16 |", "|---|---|---|---|"]
    for s in sm.FROZEN_SEEDS:
        lines.append(f"| {s} | {outcome(evs['下午'], s)} | {outcome(evs['晚上对照'], s)} | {outcome(new_e, s)} |")
    lines.append("")
    new = sm.success_by_seed(new_dir)
    aft, eve = (sm.success_by_seed(p) for _l, p in refs)
    lines += ["逐集配对（b = 只有前者成功，c = 只有后者成功；精确 McNemar p 只作参考）：", "",
              "| 配对 | 对数 | 逐集一致 | 一致率 | b | c | p |", "|---|---|---|---|---|---|---|"]
    late = {int(f["seed"]) for f in new_e.get("failed", []) if f.get("late_dispatch_steps")}
    for label, a, b, drop in (("新口径 对 下午（20 对，迟到集记没成功）", new, aft, set()), ("新口径 对 晚上对照（20 对，迟到集记没成功）", new, eve, set()),
                              ("新口径 对 下午（去掉迟到集）", new, aft, late), ("新口径 对 晚上对照（去掉迟到集）", new, eve, late),
                              ("下午 对 晚上对照（20 对）", aft, eve, set())):
        a2 = {s: v for s, v in a.items() if s not in drop}
        b2 = {s: v for s, v in b.items() if s not in drop}
        same, n, bb, cc = sm.agreement(a2, b2)
        lines.append(f"| {label} | {n} | {same} | {same / n:.2f} | {bb} | {cc} | {sm.mcnemar_exact(bb, cc):.3f} |")
    lines.append("")
    lines += ["未完成的那一集：" + "；".join(f"seed {f['seed']}：{f.get('error')}（lag 中位 {f.get('dispatch_lag_s_median')} s，最大 {f.get('dispatch_lag_s_max')} s，"
                                         f"迟到步 {f.get('late_step_ids')}）" for f in new_e.get("failed", [])) or "无", ""]
    lines += ["每步时序（生效时长为各集均值的集间中位；稳态前飞位移与转角为池化均值）：", "", *sm.TIMING_HEAD,
              sm.timing_row("新口径学生 K=1 s1 L=0.16", sm.timing_stats([new_dir])),
              sm.timing_row("旧协议下午 s1", sm.timing_stats([sm.REF_K1_AFTERNOON_S1])),
              sm.timing_row("旧协议晚上对照 s1", sm.timing_stats([sm.REF_K1_EVENING_S1])),
              sm.timing_row("旧协议下午 s0..2（60 集）", sm.timing_stats(sm.REF_K1)), ""]
    return lines


# ---------------------------------------------------------------- 森林三档
def forest_readout():
    lines = ["## 5. 森林三档（教师，不带 --action-latency；每档 3 集）", "",
             "lag 为自然延迟（没有开关，不等到 L）。第 0 步没有预热（预热只在带开关时做），单列。"
             "假设迟到 = lag > L + 0.02 s。策略段仿真秒 = lag − 帧龄 0.114 s（任务卡 W1-0 背景的固定值），墙钟 = 仿真秒 ÷ RTF0。", "",
             "| 密度 | 柱子数（每集） | 步数 | 稳态 lag 中位 / p90 / 最大 s | 第 0 步 lag s | 假设迟到 L=0.16：稳态 / 第 0 步 | 假设迟到 L=0.20：稳态 / 第 0 步 | "
             "逐步生效时长均值 s（最小–最大） | RTF0 | 策略段墙钟 中位 / 最大 ms（推算） |", "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for dens in ("sparse", "medium", "dense"):
        edir = os.path.join(sm.OUT, f"forest-{dens}")
        recs = [r for r in sm.episode_records(edir) if r[1]]
        e = sm.read_json(os.path.join(edir, "eval.json"))
        pill = [sm.pillar_count(x.get("pillars")) for x in e.get("per_episode", [])]
        steady, first, rtfs = [], [], []
        for seed, actions, oracle, man in recs:
            lag = [a["dispatch_lag_s"] for a in actions]
            first.append(lag[0])
            steady += lag[1:]
            rtf = (man.get("performance") or {}).get("action_rtf")
            if rtf:
                rtfs.append(rtf)
        t = sm.timing_stats([edir])
        rtf0 = sm.mean(rtfs)
        cells = []
        for L in sm.LADDER:
            thr = L + sm.LATE_TOL_S
            cells.append(f"{sum(1 for x in steady if x > thr)}/{len(steady)} / {sum(1 for x in first if x > thr)}/{len(first)}")
        pol = [max(0.0, x - sm.FRAME_AGE_S) / rtf0 * 1000.0 for x in steady]
        out[dens] = {"pillars": pill, "steady": steady, "first": first, "rtf0": rtf0, "policy_ms": pol}
        lines.append(f"| {dens} | {', '.join(str(p) for p in pill)} | {len(steady) + len(first)} | "
                     f"{sm.median(steady):.3f} / {sm.percentile(steady, 90):.3f} / {max(steady):.3f} | {', '.join(f'{x:.3f}' for x in first)} | {cells[0]} | {cells[1]} | "
                     f"{t['eff_mean']:.4f}（{t['eff_min']:.3f}–{t['eff_max']:.3f}） | {rtf0:.2f} | {sm.median(pol):.0f} / {max(pol):.0f} |")
    lines.append("")
    return lines, out


def rtf_need(L, lag_max, rtf0):
    """需要的 --rtf（世界目标实时因子，只能压低）= RTF0 × (L + 0.02 − 0.114) / (lag_max − 0.114)；lag_max 不超过阈值时 None。"""
    thr = L + sm.LATE_TOL_S
    if lag_max <= thr:
        return None
    return rtf0 * (thr - sm.FRAME_AGE_S) / (lag_max - sm.FRAME_AGE_S)


# ---------------------------------------------------------------- 总装
def main(out=None, note=None):
    if out:
        sm.OUT = os.path.abspath(out)
    OUT = sm.OUT
    arms016 = [("cal-L0.16-k1", sm.cal_dir(0.16, 1)), ("cal-L0.16-k3", sm.cal_dir(0.16, 3)), ("acc-L0.16-k1-s1（学生）", sm.acc_dir(0.16, 1, 1))]
    arms020 = [("cal-L0.20-k1", sm.cal_dir(0.20, 1)), ("cal-L0.20-k3", sm.cal_dir(0.20, 3))]
    lines = ["# 第一轮标定与验收的事后分析（任务卡 W1-0）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["只读 episodes/ 与 eval.json 算出，方法与定义见 analyze_round1.py 文件头。", ""]
    late, _rows = late_section(arms016 + arms020)
    lines += late
    lines += ["## 3. 稳态前飞位移差的归因（K=3 对 K=1 低 2.6%–3.3%，而逐步生效时长只差 0.1%–0.2%）", ""]
    el, el_lines = elasticity_text(sm.cal_dir(0.16, 1), sm.cal_dir(0.20, 1))
    lines += ["弹性（位移对生效时长）：" + "".join(el_lines), ""]
    for L in sm.LADDER:
        sec, _r = attribution_section(L, sm.cal_dir(L, 1), sm.cal_dir(L, 3))
        lines += sec + [duration_explains(L, sm.cal_dir(L, 1), sm.cal_dir(L, 3), el), ""]
    lines += item2_readout()
    flines, fout = forest_readout()
    lines += flines
    lines += ["需要的 `--rtf`（推算：RTF0 × (L + 0.02 − 0.114) / (稳态最大 lag − 0.114)；稳态最大 lag 不超过阈值则不需要。样本每档只有 3 集，最大值只当下限）：", ""]
    for dens, d in fout.items():
        mx = max(d["steady"])
        parts = []
        for L in sm.LADDER:
            need = rtf_need(L, mx, d["rtf0"])
            parts.append(f"L={L:.2f}：" + ("不需要" if need is None else f"{need:.2f}"))
        lines.append(f"- {dens}：稳态最大 lag {mx:.3f} s，RTF0 {d['rtf0']:.2f}；" + "；".join(parts))
    lines.append("")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "analysis-round1.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", default=os.path.join(sm.D, "exp-action-latency-20261110"))
    ap.add_argument("--note", help="写在标题下的一行说明（生成时间、提交哈希）")
    args = ap.parse_args(argv)
    return main(args.out, args.note)


if __name__ == "__main__":
    sys.exit(cli())
