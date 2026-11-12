#!/usr/bin/env python3
"""训练配方稳定性指标（任务卡 E2E-R1 第一部分的分析）。只读逐轮 val 预测，不碰 GPU、不碰 test 数据。

子命令
  report     读各臂的逐轮预测与末 10 轮平均权重的预测，算翻转率、一致率、校准量 δ、val 无加权 CE、平均权重的 macro-F1 与逐类召回、过线判决，写 json、markdown 与图。
  repro-h1   用本文件的函数复现审核 H 的 h1_bias_drift.py 对 R0 的输出，与它的日志逐行比对（口径对不上就不往下做）。

输入布局（run_offline.sh 产出；R0 是 11-09 现成的三个 run，只读）
  <ref>/select/seed{0,1,2}/preds_epNN.npy        R0 逐轮预测        <ref>/e1-noise/seed{0,1,2}/run/results.csv   R0 逐轮 val 无加权 CE（dev_ce）
  <out>/select/<臂>-s<种子>/preds_epNN.npy        新训各臂            <out>/runs/<臂>-s<种子>/results.csv
  <out>/avg/<臂>-s<种子>/preds_avg_{raw,cal}.npy  末 k 轮平均权重（不校准 / 加校准）的 val 预测，R0 的是 <out>/avg/R0-s<种子>
  <out>/gate.json                                 复训对账（gate_check.py）          <out>/blocked/<臂>.txt   卡里的臂被训练器拒绝时的证据
preds 的列：[idx, y, pred, p0, p1, p2, p3]；val 574 帧，标签 S/F/L/R = [0, 501, 40, 33]。

口径（与审核 H 的 h1_bias_drift.py 同；起跑前写进 run_offline.sh 的预注册）
  翻转率 = 同一种子相邻两轮预测不同的帧占比，逐对算再平均。「全程」= 所有相邻对（30 轮时 29 对）；「末 10 轮」= 第 21–30 轮内的 9 对（h1 的 slice(-9, None)）；
  「末 10 对」= 最后 10 个相邻对（第 20→30 轮）。过线判据 = 三个种子里「末 10 轮」最大的那个。
  三种子全一致率 = 同一轮三个种子预测在同一帧上相同的帧占比；「末 10 轮均值」= 第 21–30 轮各轮一致率的平均（h1 口径，过线判据）。
  校准 = FORWARD logit 加 δ，使 val 上 FORWARD 预测占比不低于 val 标签里的 FORWARD 占比（二分 60 步，δ 取区间中点，同 h1 的 calib_pred 与 tools/average_ckpt.calibrate）。
  macro-F1、召回、balanced acc 与 tools/select_ckpt.py、src/eval_metrics.cpp 同定义（sklearn 口径：无真值样本的类召回记 0，P+R=0 的类 F1 记 0，macro 对 4 类取平均）。
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import warnings

import numpy as np

FWD = 1
YAW = (2, 3)
N_CLASS = 4
CLASS_NAMES = ("S", "F", "L", "R")
LAST = 10                  # 「末 10 轮」
PASS_FLIP = 0.06           # 任务卡通过线：末 10 轮翻转率不高于 0.06
PASS_AGREE = 0.88          # 任务卡通过线：三种子全一致率不低于 0.88
EXPECT_R1_FLIP = 0.08      # 审核 H 的预期（任务卡抄录）：R1 的原始翻转率低于 0.08
SEEDS = (0, 1, 2)
CARD_ARMS = ("R0", "R1", "R2", "R3")      # 卡里的臂
SUBST_ARMS = ("R2e", "R3e")               # 替代臂：卡里的 R2、R3 被训练器拒绝（SAFE 权重 0）时用，SAFE 权重 1e-6；等价性待主会话确认
ARM_ORDER = CARD_ARMS + SUBST_ARMS


# ---------------------------------------------------------------- 纯函数（有测试）
def calib_pred(probs, target, fwd=FWD):
    """h1_bias_drift.calib_pred 同：z = log(max(p, 1e-300))，在 [-30, 30] 里二分 60 步找最小的 δ 使 FORWARD 预测占比不低于 target，δ 取区间中点。返回 (预测, δ)。"""
    z = np.log(np.maximum(np.asarray(probs, dtype=np.float64), 1e-300))
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        zz = z.copy()
        zz[:, fwd] += mid
        if (zz.argmax(1) == fwd).mean() < target:
            lo = mid
        else:
            hi = mid
    d = 0.5 * (lo + hi)
    z[:, fwd] += d
    return z.argmax(1), d


def confusion(y, p, c=N_CLASS):
    m = np.zeros((c, c), dtype=np.int64)
    np.add.at(m, (np.asarray(y, dtype=int), np.asarray(p, dtype=int)), 1)
    return m


def class_metrics(y, p):
    """acc、macro-F1、balanced acc、逐类召回与 F1（口径见模块说明）。"""
    m = confusion(y, p)
    tp = np.diag(m).astype(float)
    row, col = m.sum(1).astype(float), m.sum(0).astype(float)
    recall = np.where(row > 0, tp / np.maximum(row, 1), 0.0)
    prec = np.where(col > 0, tp / np.maximum(col, 1), 0.0)
    f1 = np.where(prec + recall > 0, 2 * prec * recall / np.maximum(prec + recall, 1e-300), 0.0)
    return {"acc": float(tp.sum() / max(m.sum(), 1)), "macro_f1": float(f1.mean()), "balanced_acc": float(recall.mean()),
            "recall": [float(r) for r in recall], "f1": [float(v) for v in f1]}


def recall_of(y, p, k):
    m = np.asarray(y) == k
    return float((np.asarray(p)[m] == k).mean()) if m.any() else float("nan")


def flip_series(preds, mask=None):
    """preds：按轮序排好的预测向量列表。返回相邻两轮预测不同的帧占比（长度 = 轮数 − 1）；给 mask 时只在这些帧上算。"""
    out = []
    for a, b in zip(preds[:-1], preds[1:]):
        d = np.asarray(a) != np.asarray(b)
        out.append(float(d[mask].mean()) if mask is not None else float(d.mean()))
    return np.array(out)


def flip_summary(series, last=LAST):
    """all = 全部相邻对均值；last9 = 末 last 轮内的 last−1 个相邻对（h1 口径 slice(-9, None)）；last10 = 最后 last 个相邻对。"""
    s = np.asarray(series, dtype=float)
    return {"all": float(s.mean()), "last9": float(s[-(last - 1):].mean()), "last10": float(s[-last:].mean())}


def flip_segments(series_by_seed):
    """翻转率分段（看学习率退火的机械成分）：series_by_seed = {种子: 相邻轮翻转率序列}，每个种子长度 n。
    返回三个种子均值的 前半程（前 n//2 对）、后半程（后 n//2 对；n 为奇数时中间那一对两边都不算）、最后 3 对。"""
    ser = [np.asarray(v, dtype=float) for v in series_by_seed.values()]
    h = len(ser[0]) // 2
    return {"first": float(np.mean([v[:h].mean() for v in ser])), "second": float(np.mean([v[-h:].mean() for v in ser])),
            "last3": float(np.mean([v[-3:].mean() for v in ser])), "pairs_half": int(h)}


def all_agree(vectors, mask=None):
    """各向量在同一帧上完全相同的帧占比。"""
    v = np.stack([np.asarray(x) for x in vectors])
    same = (v == v[0]).all(0)
    return float(same[mask].mean()) if mask is not None else float(same.mean())


def pair_flip_mean(vectors):
    """所有两两组合的预测不同的帧占比的平均。"""
    n = len(vectors)
    return float(np.mean([(np.asarray(vectors[i]) != np.asarray(vectors[j])).mean() for i in range(n) for j in range(i + 1, n)]))


def pass_verdict(flips, agrees):
    """flips：{读法: 翻转率}（读法 'worst_last9' 是判据）；agrees：{读法: 一致率}（读法 'raw_last10' 是判据）。
    primary = 判据读法下「翻转率不高于 PASS_FLIP 且一致率不低于 PASS_AGREE」；combos 列出所有读法组合；consistent = 各组合与 primary 同判。"""
    combos = {f"{fk}|{ak}": bool(fv <= PASS_FLIP and av >= PASS_AGREE) for fk, fv in flips.items() for ak, av in agrees.items()}
    primary = combos["worst_last9|raw_last10"]
    return {"primary": primary, "pass_flip": bool(flips["worst_last9"] <= PASS_FLIP), "pass_agree": bool(agrees["raw_last10"] >= PASS_AGREE),
            "flips": dict(flips), "agrees": dict(agrees), "combos": combos, "consistent": all(v == primary for v in combos.values())}


def choose_extra_arm(arm_reports, candidates=CARD_ARMS):
    """第二部分加的臂：在过线的候选臂（只含卡里的臂，R0 不算）里取末 10 轮原始翻转率（三种子最大）最小的，并列取编号小的；没有过线的返回 None。"""
    passed = [(r["verdict"]["flips"]["worst_last9"], a) for a, r in arm_reports.items() if a in candidates and a != "R0" and r.get("verdict", {}).get("primary")]
    return min(passed)[1] if passed else None


def choose_extra_arm_run(arm_reports):
    """第二部分实际加的臂：返回 (臂, 是否替代臂)。卡里的臂有过线者就取它（choose_extra_arm）；没有，再在过线的替代臂里按同一规则取一个（替代臂过线时才有）；都没有返回 (None, False)。
    卡里没有规定替代臂的加臂：这一档是执行者在看到第一部分结果后加的（run_offline.sh 起跑前写的是替代臂不自动加臂），结果里标明，主会话可以不采用。"""
    card = choose_extra_arm(arm_reports)
    if card:
        return card, False
    sub = choose_extra_arm(arm_reports, candidates=SUBST_ARMS)
    return (sub, True) if sub else (None, False)


# ---------------------------------------------------------------- 读盘
def load_epoch_preds(dirpath):
    """<dir>/preds_epNN.npy → {轮: 数组[n, 7]}，轮序升序；各轮的 idx、y 两列必须一致。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "preds_ep*.npy"))):
        m = re.fullmatch(r"preds_ep(\d+)\.npy", os.path.basename(p))
        if m:
            out[int(m.group(1))] = np.load(p)
    if not out:
        raise FileNotFoundError(f"{dirpath} 里没有 preds_epNN.npy")
    first = out[min(out)]
    for e, a in out.items():
        if a.shape != first.shape or not (a[:, :2] == first[:, :2]).all():
            raise ValueError(f"{dirpath} 第 {e} 轮的帧序号或标签与第 {min(out)} 轮不一致")
    return dict(sorted(out.items()))


def load_dev_ce(run_dir):
    """results.csv 的 dev_ce 列（val 无加权 CE），按轮序。"""
    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8") as f:
        return [float(r["dev_ce"]) for r in csv.DictReader(f)]


def arm_paths(arm, seed, out, ref):
    if arm == "R0":
        return {"select": f"{ref}/select/seed{seed}", "run": f"{ref}/e1-noise/seed{seed}/run", "avg": f"{out}/avg/R0-s{seed}"}
    tag = f"{arm}-s{seed}"
    return {"select": f"{out}/select/{tag}", "run": f"{out}/runs/{tag}", "avg": f"{out}/avg/{tag}"}


# ---------------------------------------------------------------- 分析
def analyse_seed(P, y, target, ce=None):
    """一个训练种子的逐轮指标。返回 (指标 dict, 原始预测列表, 校准后预测列表)；列表都按轮序。"""
    eps = sorted(P)
    raw = [P[e][:, 2].astype(int) for e in eps]
    cal, delta = [], []
    for e in eps:
        c, d = calib_pred(P[e][:, 3:7], target)
        cal.append(c)
        delta.append(float(d))
    yaw = np.isin(y, YAW)
    fwd_share = [float((r == FWD).mean()) for r in raw]
    lastk = slice(-LAST, None)
    lr_raw = np.mean([recall_of(y, r, 2) + recall_of(y, r, 3) for r in raw[lastk]]) / 2
    lr_cal = np.mean([recall_of(y, c, 2) + recall_of(y, c, 3) for c in cal[lastk]]) / 2
    res = {"epochs": [int(e) for e in eps],
           "flip": {"raw": flip_summary(flip_series(raw)), "cal": flip_summary(flip_series(cal))},
           "flip_yaw": {"raw": flip_summary(flip_series(raw, yaw)), "cal": flip_summary(flip_series(cal, yaw))},
           "flip_series_raw": [float(v) for v in flip_series(raw)], "flip_series_cal": [float(v) for v in flip_series(cal)],
           "delta": {"per_epoch": delta, "median": float(np.median(delta)), "min": float(min(delta)), "max": float(max(delta)),
                     "last10_min": float(min(delta[lastk])), "last10_max": float(max(delta[lastk])), "last": float(delta[-1])},
           "fwd_share_raw": {"per_epoch": fwd_share, "min": float(min(fwd_share)), "max": float(max(fwd_share)),
                             "last10_min": float(min(fwd_share[lastk])), "last10_max": float(max(fwd_share[lastk]))},
           "lr_recall_last10": {"raw": float(lr_raw), "cal": float(lr_cal)},
           "last_epoch": class_metrics(y, raw[-1])}
    if ce is not None:
        ce = [float(v) for v in ce]
        res["ce"] = {"per_epoch": ce, "last": ce[-1], "min": min(ce), "min_epoch": int(np.argmin(ce)) + 1,
                     "last10_mean": float(np.mean(ce[lastk])), "last10_min": float(min(ce[lastk])), "last10_max": float(max(ce[lastk]))}
    return res, raw, cal


def analyse_arm(runs, y, target, avgs=None, ces=None):
    """runs：{种子: {轮: 预测数组}}；avgs：{种子: {'raw': 数组, 'cal': 数组}}（末 k 轮平均权重的预测）；ces：{种子: [dev_ce]}。"""
    seeds = sorted(runs)
    per, rawp, calp = {}, {}, {}
    for s in seeds:
        per[s], rawp[s], calp[s] = analyse_seed(runs[s], y, target, None if ces is None else ces.get(s))
    n_ep = {len(rawp[s]) for s in seeds}
    if len(n_ep) != 1 or len({tuple(per[s]["epochs"]) for s in seeds}) != 1:
        raise ValueError(f"各种子的轮数或轮号不一致：{ {s: per[s]['epochs'][-1] for s in seeds} }")
    n = n_ep.pop()
    yaw = np.isin(y, YAW)
    lastk = range(max(0, n - LAST), n)
    ag = {k: [all_agree([P[s][i] for s in seeds]) for i in range(n)] for k, P in (("raw", rawp), ("cal", calp))}
    ag_y = {k: [all_agree([P[s][i] for s in seeds], yaw) for i in range(n)] for k, P in (("raw", rawp), ("cal", calp))}
    pf = {k: [pair_flip_mean([P[s][i] for s in seeds]) for i in range(n)] for k, P in (("raw", rawp), ("cal", calp))}
    agree = {"raw_last10": float(np.mean([ag["raw"][i] for i in lastk])), "cal_last10": float(np.mean([ag["cal"][i] for i in lastk])),
             "raw_last_epoch": ag["raw"][-1], "cal_last_epoch": ag["cal"][-1],
             "raw_yaw_last10": float(np.mean([ag_y["raw"][i] for i in lastk])), "cal_yaw_last10": float(np.mean([ag_y["cal"][i] for i in lastk])),
             "pair_flip_raw_last10": float(np.mean([pf["raw"][i] for i in lastk])), "pair_flip_cal_last10": float(np.mean([pf["cal"][i] for i in lastk])),
             "raw_per_epoch": ag["raw"], "cal_per_epoch": ag["cal"]}
    avg = None
    if avgs:
        avg = {"per_seed": {}}
        for k in ("raw", "cal"):
            preds = {s: avgs[s][k][:, 2].astype(int) for s in seeds}
            ms = {s: class_metrics(y, preds[s]) for s in seeds}
            avg[k] = {"agree": all_agree([preds[s] for s in seeds]), "agree_yaw": all_agree([preds[s] for s in seeds], yaw),
                      "fwd_share": {s: float((preds[s] == FWD).mean()) for s in seeds},
                      "macro_f1": {s: ms[s]["macro_f1"] for s in seeds}, "macro_f1_mean": float(np.mean([ms[s]["macro_f1"] for s in seeds])),
                      "balanced_acc_mean": float(np.mean([ms[s]["balanced_acc"] for s in seeds])), "acc_mean": float(np.mean([ms[s]["acc"] for s in seeds])),
                      "recall": {s: ms[s]["recall"] for s in seeds},
                      "recall_mean": [float(np.mean([ms[s]["recall"][c] for s in seeds])) for c in range(N_CLASS)]}
        agree["avg_raw"], agree["avg_cal"] = avg["raw"]["agree"], avg["cal"]["agree"]
    flips = {"worst_last9": max(per[s]["flip"]["raw"]["last9"] for s in seeds), "mean_last9": float(np.mean([per[s]["flip"]["raw"]["last9"] for s in seeds])),
             "worst_last10": max(per[s]["flip"]["raw"]["last10"] for s in seeds)}
    agrees = {"raw_last10": agree["raw_last10"], "raw_last_epoch": agree["raw_last_epoch"]}
    if avg:
        agrees["avg_raw"] = avg["raw"]["agree"]
    verdict = pass_verdict(flips, agrees)
    verdict["flip_all_worst"] = max(per[s]["flip"]["raw"]["all"] for s in seeds)   # 预期（R1 低于 0.08）的另一个读法：全程所有相邻对
    return {"seeds": {int(s): per[s] for s in seeds}, "agree": agree, "avg": avg, "verdict": verdict}


def noise_floor(P_a, P_b):
    """同配方同种子的两次训练：逐轮比较两边预测，返回 {每轮翻转率、均值、最大值、为 0 的轮数}。"""
    eps = sorted(set(P_a) & set(P_b))
    fl = [float((P_a[e][:, 2] != P_b[e][:, 2]).mean()) for e in eps]
    return {"epochs": len(eps), "per_epoch": fl, "mean": float(np.mean(fl)), "max": float(max(fl)), "zero_epochs": int(sum(1 for v in fl if v == 0.0))}


# ---------------------------------------------------------------- 复现审核 H 的输出
def h1_lines(P_by_seed):
    """与 h1_bias_drift.py 同一组公式、同一组打印格式，输入 {种子: {轮: 预测数组}}。用来逐行比对它的日志。"""
    seeds = {s: P for s, P in P_by_seed.items()}
    eps = sorted(set.intersection(*[set(v) for v in seeds.values()]))
    y = seeds[0][eps[0]][:, 1].astype(int)
    target = float((y == FWD).mean())
    lines = [f"val n={len(y)} 标签 S/F/L/R = {np.bincount(y, minlength=4).tolist()}  FWD 目标占比 {target:.4f}"]
    raw, cal, deltas = {}, {}, {}
    for s, d in seeds.items():
        for e in eps:
            raw[s, e] = d[e][:, 2].astype(int)
            cal[s, e], deltas[s, e] = calib_pred(d[e][:, 3:7], target)
    yaw = np.isin(y, YAW)
    for s in (0, 1, 2):
        fr = [(raw[s, e] != raw[s, e - 1]).mean() for e in eps[1:]]
        fc = [(cal[s, e] != cal[s, e - 1]).mean() for e in eps[1:]]
        fry = [(raw[s, e] != raw[s, e - 1])[yaw].mean() for e in eps[1:]]
        fcy = [(cal[s, e] != cal[s, e - 1])[yaw].mean() for e in eps[1:]]
        last10 = slice(-9, None)
        lines.append(f"seed{s}: 相邻轮翻转率 全帧 原始 {np.mean(fr):.3f}（末10轮 {np.mean(fr[last10]):.3f}）→ 逐轮校准 {np.mean(fc):.3f}（末10轮 {np.mean(fc[last10]):.3f}）；"
                     f"YAW 帧 {np.mean(fry):.3f} → {np.mean(fcy):.3f}")
        ds = [deltas[s, e] for e in eps]
        lines.append(f"   δ 逐轮: 中位 {np.median(ds):+.2f}，范围 [{min(ds):+.2f}, {max(ds):+.2f}]，末10轮 [{min(ds[-10:]):+.2f}, {max(ds[-10:]):+.2f}]")
        fwd_frac = [float((raw[s, e] == FWD).mean()) for e in eps]
        lines.append(f"   原始 FWD 预测占比 范围 [{min(fwd_frac):.3f}, {max(fwd_frac):.3f}]；末10轮 [{min(fwd_frac[-10:]):.3f}, {max(fwd_frac[-10:]):.3f}]")
        rl_raw = [recall_of(y, raw[s, e], 2) + recall_of(y, raw[s, e], 3) for e in eps[-10:]]
        rl_cal = [recall_of(y, cal[s, e], 2) + recall_of(y, cal[s, e], 3) for e in eps[-10:]]
        lines.append(f"   末10轮 (recall_L+recall_R)/2: 原始 {np.mean(rl_raw) / 2:.3f} 校准 {np.mean(rl_cal) / 2:.3f}")
    for tag, P in (("原始", raw), ("逐轮校准", cal)):
        agree = [float(((P[0, e] == P[1, e]) & (P[1, e] == P[2, e])).mean()) for e in eps[-10:]]
        agree_y = [float(((P[0, e] == P[1, e]) & (P[1, e] == P[2, e]))[yaw].mean()) for e in eps[-10:]]
        lines.append(f"三种子同轮全一致率（末10轮均值）{tag}: 全帧 {np.mean(agree):.3f}，YAW 帧 {np.mean(agree_y):.3f}")
    for s in (0, 1, 2):
        lshare = []
        for e in eps[-10:]:
            p = cal[s, e]
            m = np.isin(p, YAW)
            lshare.append(float((p[m] == 2).mean()))
        lines.append(f"seed{s} 校准后 YAW 预测中 L 占比（末10轮）范围 [{min(lshare):.2f}, {max(lshare):.2f}]（标签 L/(L+R) = {np.sum(y == 2) / np.sum(yaw):.2f}）")
    return lines


# ---------------------------------------------------------------- 报告
def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def arm_label(arm):
    return f"{arm}（替代）" if arm in SUBST_ARMS else arm


def render_md(res):
    arms = [a for a in ARM_ORDER if a in res["arms"]]
    miss = {a: res["missing"][a] for a in res["missing"]}
    L = []
    L.append(f"val {res['n_val']} 帧，标签 S/F/L/R = {res['label_counts']}；FORWARD 目标占比 {res['forward_target']:.4f}。轮数：{res['epochs']}。"
             "「末 10 轮」= 第 21–30 轮内的 9 个相邻对；「末 10 对」= 最后 10 个相邻对；「全程」= 全部相邻对。")
    L.append("\n### 翻转率（相邻两轮预测不同的帧占比；每个种子一行）\n")
    rows = []
    for a in arms:
        ar = res["arms"][a]
        for s, sr in ar["seeds"].items():
            fr, fc = sr["flip"]["raw"], sr["flip"]["cal"]
            rows.append([arm_label(a), s, f3(fr["all"]), f3(fr["last9"]), f3(fr["last10"]), f3(fc["all"]), f3(fc["last9"]), f3(fc["last10"])])
        rows.append([f"**{arm_label(a)} 三种子最大 / 均值（末 10 轮，原始）**", "", "", f"{f3(ar['verdict']['flips']['worst_last9'])} / {f3(ar['verdict']['flips']['mean_last9'])}", "", "", "", ""])
    L.append(md_table(["臂", "种子", "原始·全程", "原始·末 10 轮", "原始·末 10 对", "校准·全程", "校准·末 10 轮", "校准·末 10 对"], rows))
    L.append("\n### 原始翻转率分段（三个种子的均值；前半程 = 前 14 对，后半程 = 后 14 对，最后 3 对 = 第 27 轮到第 30 轮之间；用来看学习率退火的机械成分）\n")
    rows = []
    for a in arms:
        sg = flip_segments({s: sr["flip_series_raw"] for s, sr in res["arms"][a]["seeds"].items()})
        rows.append([arm_label(a), f3(sg["first"]), f3(sg["second"]), f3(sg["last3"])])
    L.append(md_table(["臂", "前半程", "后半程", "最后 3 对"], rows))
    L.append(f"\n### YAW 帧（标签为 L/R 的 {res['label_counts'][2] + res['label_counts'][3]} 帧）上的翻转率\n")
    rows = [[arm_label(a), s, f3(sr["flip_yaw"]["raw"]["all"]), f3(sr["flip_yaw"]["raw"]["last9"]), f3(sr["flip_yaw"]["cal"]["all"]), f3(sr["flip_yaw"]["cal"]["last9"])]
            for a in arms for s, sr in res["arms"][a]["seeds"].items()]
    L.append(md_table(["臂", "种子", "原始·全程", "原始·末 10 轮", "校准·全程", "校准·末 10 轮"], rows))
    L.append("\n### 三种子全一致率（同一帧三个种子预测相同的占比）\n")
    rows = []
    for a in arms:
        g = res["arms"][a]["agree"]
        rows.append([arm_label(a), f3(g["raw_last10"]), f3(g["cal_last10"]), f3(g["raw_last_epoch"]), f3(g["cal_last_epoch"]), f3(g.get("avg_raw")), f3(g.get("avg_cal")),
                     f3(g["raw_yaw_last10"]), f3(g["cal_yaw_last10"]), f3(g["pair_flip_raw_last10"])])
    L.append(md_table(["臂", "原始·末 10 轮均值", "校准·末 10 轮均值", "原始·末轮", "校准·末轮", "平均权重·不校准", "平均权重·加校准", "YAW 帧原始·末 10 轮", "YAW 帧校准·末 10 轮", "种子两两翻转·原始·末 10 轮"], rows))
    L.append("\n### 逐轮校准量 δ 与原始 FORWARD 预测占比（val）\n")
    rows = []
    for a in arms:
        for s, sr in res["arms"][a]["seeds"].items():
            d, fs = sr["delta"], sr["fwd_share_raw"]
            rows.append([arm_label(a), s, f"{d['median']:+.2f}", f"[{d['min']:+.2f}, {d['max']:+.2f}]", f"[{d['last10_min']:+.2f}, {d['last10_max']:+.2f}]",
                         f"{d['last10_max'] - d['last10_min']:.2f}", f"[{fs['last10_min']:.3f}, {fs['last10_max']:.3f}]"])
    L.append(md_table(["臂", "种子", "δ 中位", "δ 全程范围", "δ 末 10 轮范围", "末 10 轮 δ 极差", "原始 FWD 占比 末 10 轮范围"], rows))
    ce_arms = [a for a in arms if all("ce" in sr for sr in res["arms"][a]["seeds"].values())]
    if ce_arms:
        L.append("\n### val 无加权交叉熵（results.csv 的 dev_ce；三个种子的均值，逐轮曲线见 ce_curves.csv 与图）\n")
        pick = [e for e in (1, 5, 10, 15, 20, 25, 30) if e <= res["epochs"]]
        rows = []
        for a in ce_arms:
            seeds = res["arms"][a]["seeds"]
            mean = [float(np.mean([seeds[s]["ce"]["per_epoch"][e] for s in seeds])) for e in range(res["epochs"])]
            rows.append([arm_label(a)] + [f"{mean[e - 1]:.3f}" for e in pick] + [f"{min(mean):.3f}（第 {int(np.argmin(mean)) + 1} 轮）",
                         f"{np.mean(mean[-LAST:]):.3f}", " / ".join(f"{seeds[s]['ce']['last']:.3f}" for s in seeds)])
        L.append(md_table(["臂"] + [f"第 {e} 轮" for e in pick] + ["均值曲线最小", "末 10 轮均值", "末轮 各种子"], rows))
    avg_arms = [a for a in arms if res["arms"][a].get("avg")]
    if avg_arms:
        L.append("\n### 末 10 轮平均权重（val；召回按 S / F / L / R；macro-F1 对 4 类取平均，S 类 val 无样本，上限 0.75）\n")
        rows = []
        for a in avg_arms:
            for k, name in (("raw", "不校准"), ("cal", "加校准")):
                v = res["arms"][a]["avg"][k]
                rows.append([arm_label(a), name, " / ".join(f3(v["macro_f1"][s]) for s in v["macro_f1"]), f3(v["macro_f1_mean"]),
                             " / ".join(f3(x) for x in v["recall_mean"]), f3(v["balanced_acc_mean"]), f3(v["acc_mean"]),
                             " / ".join(f"{v['fwd_share'][s]:.3f}" for s in v["fwd_share"])])
        L.append(md_table(["臂", "权重", "macro-F1 各种子", "macro-F1 均值", "召回均值 S/F/L/R", "balanced acc 均值", "acc 均值", "FORWARD 预测占比 各种子"], rows))
    L.append("\n### 过线判决（任务卡：不做事后校准，末 10 轮翻转率 ≤ 0.06 且三种子全一致率 ≥ 0.88）\n")
    rows = []
    for a in arms:
        if a == "R0":
            continue
        v = res["arms"][a]["verdict"]
        avg = res["arms"][a].get("avg")
        lr = f"{avg['raw']['recall_mean'][2]:.3f} / {avg['raw']['recall_mean'][3]:.3f}" if avg else "—"
        rows.append([arm_label(a), f3(v["flips"]["worst_last9"]), f3(v["flips"]["mean_last9"]), f3(v["flips"]["worst_last10"]), f3(v["agrees"]["raw_last10"]),
                     f3(v["agrees"]["raw_last_epoch"]), f3(v["agrees"].get("avg_raw")), "是" if v["pass_flip"] else "否", "是" if v["pass_agree"] else "否",
                     "**过线**" if v["primary"] else "不过线", "各读法同判" if v["consistent"] else "读法之间不一致：" + ", ".join(k for k, x in v["combos"].items() if x != v["primary"]), lr])
    L.append(md_table(["臂", "翻转率·三种子最大（判据）", "三种子均值", "末 10 对·最大", "一致率·末 10 轮均值（判据）", "一致率·末轮", "一致率·平均权重", "翻转率达标", "一致率达标", "判决", "读法一致性",
                       "平均权重（不校准）L / R 召回均值"], rows))
    L.append("\n通过线只度量稳定，不度量可用：全部预测 FORWARD 的权重，翻转率为 0、一致率为 1，也会过线。判读时同时看最后一列与上一张表的 macro-F1。")
    r1 = res["arms"].get("R1")
    if r1:
        v = r1["verdict"]
        L.append(f"\nR1 的预期（审核 H：原始翻转率低于 {EXPECT_R1_FLIP}）：全程所有相邻对读法最大 {f3(v['flip_all_worst'])}（{'低于' if v['flip_all_worst'] < EXPECT_R1_FLIP else '不低于'} {EXPECT_R1_FLIP}）；"
                 f"末 10 轮读法最大 {f3(v['flips']['worst_last9'])}（{'低于' if v['flips']['worst_last9'] < EXPECT_R1_FLIP else '不低于'} {EXPECT_R1_FLIP}）。")
    if miss:
        L.append("\n未跑或无数据的臂：" + "；".join(f"{a}：{why}" for a, why in miss.items()))
    L.append(f"\n卡里的臂中过线者（第二部分加臂的依据；按末 10 轮原始翻转率三种子最大者最小，并列取编号小的）：{res['extra_arm_card'] or '无'}。")
    pend = [a for a in SUBST_ARMS if a in res["arms"] and res["arms"][a]["verdict"]["primary"]]
    if pend:
        L.append(f"替代臂过线：{', '.join(pend)}（替代臂不进卡里的过线判决）。")
    if res["extra_arm"]:
        L.append(f"第二部分加的臂：{res['extra_arm']}" + ("（替代臂；卡里没有规定替代臂的加臂，这是看到本表之后按同一规则加的，主会话可以不采用）。" if res["extra_arm_is_substitute"] else "。"))
    else:
        L.append("第二部分加的臂：无。")
    nf = res.get("noise_floor")
    if nf:
        L.append("\n### 同配方同种子两次训练（R0 种子 0 对 R0b 种子 0）\n")
        rows = [["逐轮预测翻转率", f"均值 {nf['flip']['mean']:.4f}，最大 {nf['flip']['max']:.4f}，为 0 的轮数 {nf['flip']['zero_epochs']}/{nf['flip']['epochs']}"]]
        if nf.get("gate"):
            g = nf["gate"]
            rows += [["末轮 val macro-F1（参照 / 复训 / 差）", f"{g['last']['ref']['macro_f1']:.6f} / {g['last']['new']['macro_f1']:.6f} / {g['last']['d_macro_f1']:+.6f}"],
                     ["对账的轮数；权重逐位相同的轮数", f"{g['epochs_compared']}；{len(g['weights_identical_epochs'])}"],
                     ["全程 val macro-F1 最大差；val CE 最大差", f"{g['max_abs_d_macro_f1_all_epochs']:.3g}；{g['max_abs_d_dev_ce_all_epochs']:.3g}"],
                     ["闸门（差的绝对值 < 0.05）", "过" if g["gate"]["pass"] else "不过"]]
        L.append(md_table(["项", "值"], rows))
        L.append(f"\n对照：R0 三个种子之间同一轮的预测两两翻转率（末 10 轮均值，原始）{f3(res['arms']['R0']['agree']['pair_flip_raw_last10'])}。")
    return "\n".join(L) + "\n"


def make_figures(res, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    arms = [a for a in ARM_ORDER if a in res["arms"]]
    colors = dict(zip(ARM_ORDER, ["#444444", "#1f77b4", "#2ca02c", "#d62728", "#98df8a", "#ff9896"]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for a in arms:
        seeds = res["arms"][a]["seeds"]
        n = res["epochs"]
        for ax, key, getter in ((axes[0], "val CE（无加权）", lambda sr: sr.get("ce", {}).get("per_epoch")),
                                (axes[1], "相邻轮翻转率（原始）", lambda sr: [None] + sr["flip_series_raw"]),
                                (axes[2], "逐轮校准量 δ", lambda sr: sr["delta"]["per_epoch"])):
            ys = [getter(sr) for sr in seeds.values()]
            if any(v is None for v in ys):
                continue
            arr = np.array([[np.nan if x is None else x for x in v] for v in ys], dtype=float)
            x = np.arange(1, n + 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # 第 1 轮没有「与上一轮的翻转率」，整列 nan
                lo, hi, mid = np.nanmin(arr, 0), np.nanmax(arr, 0), np.nanmean(arr, 0)
            ax.fill_between(x, lo, hi, color=colors[a], alpha=0.15)
            ax.plot(x, mid, color=colors[a], label=arm_label(a), lw=1.6, ls="--" if a in SUBST_ARMS else "-")
            ax.set_title(key)
            ax.set_xlabel("训练轮")
    axes[1].axhline(PASS_FLIP, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("各臂三个种子的均值（线）与最小到最大（带）", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def build_report(out, ref, arms, seeds=SEEDS):
    res = {"arms": {}, "missing": {}}
    y = target = None
    for a in arms:
        runs, avgs, ces, why = {}, {}, {}, None
        for s in seeds:
            p = arm_paths(a, s, out, ref)
            try:
                runs[s] = load_epoch_preds(p["select"])
            except (FileNotFoundError, ValueError) as e:
                note = os.path.join(out, "blocked", f"{a}.txt")
                if os.path.exists(note):
                    with open(note, encoding="utf-8") as nf:
                        why = nf.read().strip()[:300]
                else:
                    why = f"{type(e).__name__}: {e}"
                break
            try:
                ces[s] = load_dev_ce(p["run"])
            except OSError:
                pass
            if all(os.path.exists(os.path.join(p["avg"], f"preds_avg_{k}.npy")) for k in ("raw", "cal")):
                avgs[s] = {k: np.load(os.path.join(p["avg"], f"preds_avg_{k}.npy")) for k in ("raw", "cal")}
        if why is not None:
            res["missing"][a] = why
            continue
        if y is None:
            first = runs[seeds[0]][min(runs[seeds[0]])]
            y, idx0 = first[:, 1].astype(int), first[:, 0]
            target = float((y == FWD).mean())
        for s in seeds:
            first = runs[s][min(runs[s])]
            if not (first[:, 0] == idx0).all() or not (first[:, 1].astype(int) == y).all():
                raise ValueError(f"{a} 种子 {s} 的 val 帧序号或标签与 R0 不同")
        res["arms"][a] = analyse_arm(runs, y, target, avgs if len(avgs) == len(seeds) else None, ces if len(ces) == len(seeds) else None)
    if y is None:
        raise SystemExit("report: 没有任何臂有数据")
    first_arm = next(iter(res["arms"].values()))
    res.update({"n_val": int(len(y)), "label_counts": np.bincount(y, minlength=N_CLASS).tolist(), "forward_target": target,
                "epochs": len(next(iter(first_arm["seeds"].values()))["epochs"])})
    res["extra_arm_card"] = choose_extra_arm(res["arms"])                      # 卡里的臂（R1、R2、R3）中过线者
    res["extra_arm"], res["extra_arm_is_substitute"] = choose_extra_arm_run(res["arms"])   # 第二部分实际加的臂（run_closed_loop.sh 读这一项）
    # 噪声底：R0 种子 0 对复训的 R0b 种子 0
    rb = os.path.join(out, "select", "R0b-s0")
    if "R0" in res["arms"] and os.path.isdir(rb):
        P_new = load_epoch_preds(rb)
        P_ref = load_epoch_preds(arm_paths("R0", 0, out, ref)["select"])
        nf = {"flip": noise_floor(P_ref, P_new)}
        gj = os.path.join(out, "gate.json")
        if os.path.exists(gj):
            with open(gj, encoding="utf-8") as f:
                nf["gate"] = json.load(f)
        res["noise_floor"] = nf
    return res


def write_ce_csv(res, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "epoch", "dev_ce", "flip_raw_vs_prev", "delta"])
        for a, ar in res["arms"].items():
            for s, sr in ar["seeds"].items():
                for i, e in enumerate(sr["epochs"]):
                    w.writerow([a, s, e, "" if "ce" not in sr else f"{sr['ce']['per_epoch'][i]:.6f}",
                                "" if i == 0 else f"{sr['flip_series_raw'][i - 1]:.6f}", f"{sr['delta']['per_epoch'][i]:.6f}"])


def cmd_report(args):
    arms = args.arms.split(",")
    res = build_report(args.out_dir, args.ref, arms)
    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, "stability.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    md = render_md(res)
    with open(os.path.join(args.result_dir, "stability.md"), "w", encoding="utf-8") as f:
        f.write(md)
    write_ce_csv(res, os.path.join(args.result_dir, "ce_curves.csv"))
    try:
        make_figures(res, args.result_dir)
    except ImportError as e:   # 没有 matplotlib 时只缺图
        print(f"report: 未画图（{e}）", file=sys.stderr)
    print(md)
    return 0


def cmd_repro(args):
    P = {s: load_epoch_preds(arm_paths("R0", s, args.out_dir, args.ref)["select"]) for s in SEEDS}
    mine = h1_lines(P)
    with open(args.h1_log, encoding="utf-8") as f:
        theirs = f.read().rstrip("\n").split("\n")
    diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    ok = len(mine) == len(theirs) and not diff
    print("\n".join(mine))
    print(f"\nrepro-h1: 本文件算出 {len(mine)} 行，审核 H 的日志 {len(theirs)} 行，" + ("逐行一致" if ok else f"不一致 {len(diff)} 行：{diff[:3]}"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "repro-h1"):
        sp = sub.add_parser(name)
        sp.add_argument("--out-dir", required=True, help="run_offline.sh 的 $OUT")
        sp.add_argument("--ref", required=True, help="11-09 参照 run 的目录（exp-student-select-20261109）")
        if name == "report":
            sp.add_argument("--arms", default=",".join(ARM_ORDER))
            sp.add_argument("--result-dir", required=True, help="stability.json / stability.md / ce_curves.csv / curves.png 写到这里")
        else:
            sp.add_argument("--h1-log", required=True)
    args = ap.parse_args(argv)
    return cmd_report(args) if args.cmd == "report" else cmd_repro(args)


if __name__ == "__main__":
    sys.exit(main())
