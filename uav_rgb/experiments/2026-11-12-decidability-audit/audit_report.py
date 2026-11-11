#!/usr/bin/env python3
"""可判定性审计（卡 E2E-0）第三步：由 stats_*.npz、bins.json 等生成 tables.md 与 results.json。

区间：对测试集的几何簇（同一 geometry_gid 的集合并作一簇）做有放回重抽样（AUDIT_BOOT_REPS 次，种子 AUDIT_BOOT_SEED），
每次重抽样里三个种子各自的指标取均值，再取 2.5% 与 97.5% 分位数；两个特征组的差值用同一批重抽样（配对）。
区间只反映「测试几何有限」带来的波动，不含「换一批训练几何」的波动（只有约 800 种几何，测试几何与训练几何同分布）。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_features as af  # noqa: E402
import audit_fit as fit  # noqa: E402

RECALL_IDX = {"recall_S": 0, "recall_F": 1, "recall_L": 2, "recall_R": 3}
HEAD_METRICS = (("acc", "整体准确率"), ("f1_frl", "macro-F1（F/L/R）"), ("turn_recall", "该转时转了"), ("lr_acc", "转向帧左右对"))
MORE_METRICS = (("recall_S", "召回 SAFE"), ("recall_F", "召回 FWD"), ("recall_L", "召回 LEFT"), ("recall_R", "召回 RIGHT"),
                ("lr_cond", "判成转向后方向对"), ("false_turn", "前进帧被判成转向"), ("f1_4", "macro-F1（4 类）"))
SUBSET_TITLES = {"teacher": "教师驱动的帧（示范集）", "student": "学生驱动的帧（DAgger 第 1–3 轮）", "empty": "画面（60°）里 4 m 内没有障碍的帧",
                 "student_r23": "附加：纯学生执行的轮（第 2–3 轮，β=0）", "all": "附加：全部测试帧"}
GROUP_NOTES = {"S1": "d", "S2": "d + 上一步动作", "S3": "d + n", "S4": "d + n + hist", "S5": "d_wide + n + hist", "S6": "S5 + priv",
               "S3c": "卡外附加：d + 三值航向码", "S7": "卡外附加：S6 + 真实航向（天花板参照）"}
SUBSET_SHORT = {"teacher": "教师驱动的帧", "student": "学生驱动的帧", "empty": "画面里 4 m 内没有障碍的帧"}
CONTRASTS = (("S2", "S1"), ("S3", "S1"), ("S4", "S3"), ("S5", "S4"), ("S6", "S5"), ("S3c", "S3"), ("S7", "S6"))


# ---------------------------------------------------------------- 读 stats、取指标、自助法
def load_stats(path):
    z = np.load(path, allow_pickle=False)
    st = {k: z[k] for k in z.files}
    for k in ("groups", "subsets", "clusters"):
        st[k] = [str(v) for v in st[k]]
    st["seeds"] = [int(v) for v in st["seeds"]]
    st["fit_subset"] = str(st["fit_subset"])
    return st


def boot_weights(valid, B, seed):
    """[B, 簇数] 的重抽样次数：只在 valid 的簇里有放回抽 valid.sum() 个，没有帧的簇次数恒为 0。"""
    valid = np.asarray(valid, bool)
    idx = np.nonzero(valid)[0]
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(idx), size=(B, len(idx)))
    flat = (np.arange(B)[:, None] * len(valid) + idx[draws]).ravel()
    return np.bincount(flat, minlength=B * len(valid)).reshape(B, len(valid))


def _pick(m, name):
    return m["recall"][..., RECALL_IDX[name]] if name in RECALL_IDX else m[name]


def metric_values(st, g, s, name, mult=None):
    """[种子数]（点估计）或 [B, 种子数]（给了重抽样次数）。"""
    gi, si = st["groups"].index(g), st["subsets"].index(s)
    cm = st["cm"][gi, :, si].astype(np.float64)
    lr = st["lr_ok"][gi, :, si].astype(np.float64)
    if mult is None:
        return _pick(fit.metrics_from_counts(cm.sum(1), lr.sum(1)), name)
    agg = np.einsum("bc,scij->bsij", mult, cm)
    agg_lr = np.einsum("bc,sc->bs", mult, lr)
    return _pick(fit.metrics_from_counts(agg, agg_lr), name)


def point(st, g, s, name):
    v = metric_values(st, g, s, name)
    return float(np.mean(v)), float(np.min(v)), float(np.max(v))


def interval(st, g, s, name, mult):
    v = metric_values(st, g, s, name, mult).mean(-1)
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def contrast(st, g1, g0, s, name, mult):
    """g1 − g0（种子均值之差）与配对自助法区间。"""
    d = float(np.mean(metric_values(st, g1, s, name)) - np.mean(metric_values(st, g0, s, name)))
    b = metric_values(st, g1, s, name, mult).mean(-1) - metric_values(st, g0, s, name, mult).mean(-1)
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return d, float(lo), float(hi)


def valid_clusters(st, s):
    si = st["subsets"].index(s)
    return st["cm"][:, 0, si].sum((0, 2, 3)) > 0


# ---------------------------------------------------------------- 判定与格式
_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, ">": lambda a, b: a > b}


def judge(pt, lo, hi, op, thr):
    f = _OPS[op]
    same = f(lo, thr) == f(hi, thr)
    return ("成立" if f(pt, thr) else "不成立") + ("（区间在阈值同一侧）" if same else "（区间跨过阈值，未能分辨）")


def judge_abs_lt(pt, lo, hi, thr):
    inside = max(abs(lo), abs(hi)) < thr
    outside = lo >= thr or hi <= -thr
    return ("成立" if abs(pt) < thr else "不成立") + ("（区间在阈值同一侧）" if inside or outside else "（区间跨过阈值，未能分辨）")


def fmt(mean, lo, hi):
    if not np.isfinite(mean):
        return "—"
    return "%.3f [%.3f, %.3f]" % (mean, lo, hi)


def fmt_ci(d, lo, hi):
    if not np.isfinite(d):
        return "—"
    return ("%+.3f [%+.3f, %+.3f]" % (d, lo, hi)).replace("-", "−")


def fmt1(v):
    return "—" if not np.isfinite(v) else "%.3f" % v


# ---------------------------------------------------------------- 表
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def group_label(g):
    return "%s（%s）" % (g, GROUP_NOTES[g])


def baseline_metrics(st, s):
    """恒选 FORWARD_SLOW 的指标（同一批测试帧；与特征组无关，取第一组第一个种子的行合计）。"""
    si = st["subsets"].index(s)
    cm = fit.majority_baseline_counts(st["cm"][0, 0, si].sum(0))
    return fit.metrics_from_counts(cm, 0)


def metric_table(st, subset, metrics, groups, with_baseline=True):
    rows = []
    if with_baseline:
        b = baseline_metrics(st, subset)
        rows.append(["恒选前进（基线）"] + [fmt1(float(_pick(b, m))) if m not in ("lr_acc", "lr_cond") else "—" for m, _ in metrics])
    for g in groups:
        rows.append([group_label(g)] + [fmt(*point(st, g, subset, m)) for m, _ in metrics])
    return md_table(["特征组"] + [t for _, t in metrics], rows)


def interval_rows(st, groups, subset, metrics, mult):
    """主表对应的 95% 区间：行 = 特征组，列 = 指标，单元格 = [下, 上]（测试几何簇自助法，三个种子取均值）。"""
    return [[group_label(g)] + ["[%.3f, %.3f]" % interval(st, g, subset, m, mult) for m, _ in metrics] for g in groups]


def subset_counts_line(st, subset):
    si = st["subsets"].index(subset)
    cm = st["cm"][0, 0, si].sum(0)
    return "测试帧 %d（SAFE %d / FWD %d / LEFT %d / RIGHT %d），其中转向帧 %d；几何簇 %d" % (
        cm.sum(), *cm.sum(1), cm[2:].sum(), int(valid_clusters(st, subset).sum()))


def expectation_rows(stats, mults):
    """逐条对照起跑前的预期。每条先给卡的原句对应的主读法（pooled_w），再给同一句话换拟合池/加权方式的读法。"""
    rows = []

    def add(tag, text, cfg, kind, g, s, name, op=None, thr=None, g0=None):
        st = stats.get(cfg)
        if st is None or g not in st["groups"] or s not in st["subsets"] or (g0 and g0 not in st["groups"]):
            return
        mult = mults[(cfg, s)]
        if kind == "point":
            pt, lo_s, hi_s = point(st, g, s, name)
            lo, hi = interval(st, g, s, name, mult)
            verdict = judge(pt, lo, hi, op, thr)
            val = "%.3f（种子 %.3f–%.3f）" % (pt, lo_s, hi_s)
        else:
            pt, lo, hi = contrast(st, g, g0, s, name, mult)
            _, lo_s, hi_s = point(st, g, s, name)
            verdict = judge_abs_lt(pt, lo, hi, thr) if op == "abs<" else judge(pt, lo, hi, op, thr)
            val = "%+.3f" % pt
        rows.append([tag, text, cfg, val.replace("-", "−"), ("[%.3f, %.3f]" % (lo, hi)).replace("-", "−"), verdict])

    add("E1", "学生驱动的帧上，S1 的左右正确率 ≤ 0.70", "pooled_w", "point", "S1", "student", "lr_acc", "<=", 0.70)
    add("E2", "学生驱动的帧上，S3 的左右正确率 ≥ 0.85", "pooled_w", "point", "S3", "student", "lr_acc", ">=", 0.85)
    add("E3", "教师驱动的帧上，S3 与 S1 的整体准确率相差不到 5 个点", "pooled_w", "contrast", "S3", "teacher", "acc", "abs<", 0.05, g0="S1")
    add("E4", "学生驱动的帧上，S5 比 S4 的 macro-F1（F/L/R）高不到 5 个点", "pooled_w", "contrast", "S5", "student", "f1_frl", "<", 0.05, g0="S4")
    add("E4'", "学生驱动的帧上，S5 比 S4 的 macro-F1（F/L/R）高 10 个点以上（需要学出来的记忆）", "pooled_w", "contrast", "S5", "student", "f1_frl", ">=", 0.10, g0="S4")
    # 同一句话的其他读法
    for cfg in ("pooled_u", "student_w"):
        add("E1 换配置", "学生驱动的帧上，S1 的左右正确率 ≤ 0.70", cfg, "point", "S1", "student", "lr_acc", "<=", 0.70)
        add("E2 换配置", "学生驱动的帧上，S3 的左右正确率 ≥ 0.85", cfg, "point", "S3", "student", "lr_acc", ">=", 0.85)
    for cfg in ("pooled_u", "teacher_w"):
        add("E3 换配置", "教师驱动的帧上，S3 与 S1 的整体准确率相差不到 5 个点", cfg, "contrast", "S3", "teacher", "acc", "abs<", 0.05, g0="S1")
    for cfg in ("pooled_w", "pooled_u", "student_w"):
        for name, title in (("acc", "整体准确率"), ("lr_acc", "转向帧左右对"), ("turn_recall", "该转时转了")):
            add("E4 换指标" if cfg == "pooled_w" else "E4 换配置", "学生驱动的帧上，S5 比 S4 的%s高不到 5 个点" % title, cfg, "contrast", "S5", "student",
                name, "<", 0.05, g0="S4")
        if cfg != "pooled_w":
            add("E4 换配置", "学生驱动的帧上，S5 比 S4 的 macro-F1（F/L/R）高不到 5 个点", cfg, "contrast", "S5", "student", "f1_frl", "<", 0.05, g0="S4")
    for s in ("teacher", "empty"):
        add("E4 换子集", "%s上，S5 比 S4 的 macro-F1（F/L/R）高不到 5 个点" % SUBSET_SHORT[s], "pooled_w", "contrast", "S5", s, "f1_frl", "<", 0.05, g0="S4")
    return rows


def bins_tables(bins, stats):
    out = []
    for sub in ("teacher", "student", "empty"):
        rows = []
        for g in fit.CARD_GROUPS:
            if g not in bins:
                continue
            p, o = bins[g][sub]["pooled"], bins[g][sub]["own"]
            mlp_u = fmt1(point(stats["pooled_u"], g, sub, "acc")[0]) if "pooled_u" in stats else "—"
            mlp_w = fmt1(point(stats["pooled_w"], g, sub, "acc")[0]) if "pooled_w" in stats else "—"
            lr_u = fmt1(point(stats["pooled_u"], g, sub, "lr_acc")[0]) if "pooled_u" in stats else "—"
            rows.append([group_label(g), p["bins_train"], "%.1f" % p["frames_per_bin"], fmt1(o["purity_train"]), fmt1(p["heldout_acc"]),
                         fmt1(p["coverage"]), mlp_u, mlp_w, fmt1(p["heldout_lr"]), lr_u])
        out += ["", "#### " + SUBSET_TITLES[sub], ""]
        out += md_table(["特征组", "占用箱数（全部训练帧）", "训练帧/箱", "箱内多数占比（该子集训练帧）", "分箱留出准确率", "测试帧落在见过的箱",
                         "MLP 准确率（不加权）", "MLP 准确率（加权）", "分箱留出左右对", "MLP 左右对（不加权）"], rows)
    return out


def run_report(out_dir):
    B = af.env("AUDIT_BOOT_REPS", int)
    seed0 = af.env("AUDIT_BOOT_SEED", int)
    configs = [c for c in af.env("AUDIT_FIT_CONFIGS").split(",") if c.strip()]
    stats = {c: load_stats(os.path.join(out_dir, "stats_%s.npz" % c)) for c in configs
             if os.path.exists(os.path.join(out_dir, "stats_%s.npz" % c))}
    mults = {}
    for c, st in stats.items():
        for si, s in enumerate(st["subsets"]):
            mults[(c, s)] = boot_weights(valid_clusters(st, s), B, seed0 + si)
    with open(os.path.join(out_dir, "data_summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(out_dir, "selection_check.json"), encoding="utf-8") as f:
        sel = json.load(f)
    bins = json.load(open(os.path.join(out_dir, "bins.json"), encoding="utf-8")) if os.path.exists(os.path.join(out_dir, "bins.json")) else None
    lines = ["# 可判定性审计结果表（由 audit_report.py 生成）", "",
             "单元格 = 3 个种子的均值 [最小, 最大]。差值与阈值判定的区间 = 测试几何簇自助法 %d 次的 95%% 区间（种子均值、配对）。" % B,
             "主配置 pooled_w：所有训练帧一起拟合、类别加权交叉熵；其余配置见 §5。", "",
             "## 1. 选步核对与数据量", ""]
    lines += md_table(["打包日志", "对账", "集数", "样本数", "丢弃的碰撞步 / 总步数"],
                      [[os.path.relpath(p, af.env("AUDIT_DATA_ROOT")), "通过" if r["ok"] else "失败", r["episodes"], r["samples"], "%d / %d" % (r["dropped"], r["seen"])]
                       for p, r in sel.items()])
    lines += ["", "各 split × 子集的帧数（类别数 = SAFE / FWD / LEFT / RIGHT）：", ""]
    lines += md_table(["split / 子集", "帧数", "集数", "几何簇", "类别数"],
                      [[k, v["frames"], v["episodes"], v["geometry_clusters"], " / ".join(map(str, v["class_counts"]))] for k, v in summary.items()])
    main = stats["pooled_w"]
    lines += ["", "## 2. 主表：特征组 × 子集（pooled_w）", ""]
    for sub in ("teacher", "student", "empty", "student_r23", "all"):
        lines += ["### %s" % SUBSET_TITLES[sub], "", subset_counts_line(main, sub), ""]
        lines += metric_table(main, sub, HEAD_METRICS, fit.ALL_GROUPS)
        lines += [""] + metric_table(main, sub, MORE_METRICS, fit.ALL_GROUPS, with_baseline=True) + [""]
    lines += ["### 主表的 95% 区间（pooled_w）", ""]
    for sub in ("teacher", "student", "empty"):
        lines += ["**%s**" % SUBSET_TITLES[sub], ""]
        lines += md_table(["特征组"] + [t for _, t in HEAD_METRICS], interval_rows(main, fit.ALL_GROUPS, sub, HEAD_METRICS, mults[("pooled_w", sub)])) + [""]
    lines += ["## 3. 与起跑前预期逐条对照", "",
              "主读法（pooled_w）在前，同一句话换配置/指标/子集的读法在后。「区间跨过阈值」= 点估计与阈值的关系在 95% 区间里不稳定。", ""]
    lines += md_table(["条", "预期", "配置", "测得", "95% 区间", "判定"], expectation_rows(stats, mults))
    lines += ["", "## 4. 相邻特征组的差值（Δ = 前者 − 后者，95% 区间配对自助法）", ""]
    for cfg in [c for c in ("pooled_w", "pooled_u") if c in stats]:
        lines += ["### 配置 %s" % cfg, ""]
        for sub in ("teacher", "student", "empty"):
            rows = []
            for g1, g0 in CONTRASTS:
                rows.append(["%s − %s" % (g1, g0)] + [fmt_ci(*contrast(stats[cfg], g1, g0, sub, m, mults[(cfg, sub)])) for m, _ in HEAD_METRICS])
            lines += ["#### %s" % SUBSET_TITLES[sub], ""] + md_table(["差值"] + [t for _, t in HEAD_METRICS], rows) + [""]
    lines += ["## 5. 敏感性：其他拟合配置", ""]
    for cfg, text in (("pooled_u", "全部训练帧一起拟合，不加权"), ("teacher_w", "只用教师驱动的训练帧拟合（加权），只在教师驱动的测试帧上取数"),
                      ("student_w", "只用学生驱动的训练帧拟合（加权），只在学生驱动的测试帧上取数")):
        if cfg not in stats:
            continue
        lines += ["### %s：%s" % (cfg, text), ""]
        st = stats[cfg]
        for sub in [s for s in ("teacher", "student", "empty") if s in st["subsets"]] + ([s for s in ("student_r23",) if s in st["subsets"]]):
            lines += ["**%s**" % SUBSET_TITLES[sub], ""]
            lines += metric_table(st, sub, HEAD_METRICS, [g for g in fit.ALL_GROUPS if g in st["groups"]]) + [""]
            if cfg == "pooled_u" and sub in ("teacher", "student", "empty"):
                lines += ["95% 区间：", ""]
                lines += md_table(["特征组"] + [t for _, t in HEAD_METRICS], interval_rows(st, fit.ALL_GROUPS, sub, HEAD_METRICS, mults[(cfg, sub)])) + [""]
    lines += ["## 6. 分箱核对（不依赖拟合器）", "",
              "箱的定义见 README。「箱内多数占比」是训练帧上每个箱里教师动作的多数占比，箱多帧少时偏高；「分箱留出准确率」是用训练帧的箱内多数类去判测试帧"
              "（没见过的箱退回训练集的多数类），箱多帧少时偏低；MLP 准确率应落在两者之间或与留出准确率同量级。分箱规则是不加权的多数规则，对量级用「不加权」的 MLP 一列。"]
    if bins:
        lines += bins_tables(bins, stats)
    lines += ["", "## 7. 拟合诊断（val 加权交叉熵最低的那一轮）", ""]
    diag = []
    for cfg in stats:
        fp = os.path.join(out_dir, "fitmeta_%s.json" % cfg)
        if not os.path.exists(fp):
            continue
        meta = json.load(open(fp, encoding="utf-8"))
        cap = meta["cfg"]["max_epochs"]
        for g in stats[cfg]["groups"]:
            fs = [m for m in meta["fits"] if m["group"] == g]
            diag.append([cfg, g, fs[0]["in_dim"], fs[0]["n_train"], fs[0]["n_val"],
                         " / ".join(str(m["best_epoch"]) for m in fs), sum(m["best_epoch"] >= cap for m in fs), "%.0f" % sum(m["wall_s"] for m in fs)])
    lines += md_table(["配置", "特征组", "输入维数", "训练帧", "验证帧", "各种子的最佳轮数", "最佳轮数顶到上限的拟合数", "耗时（秒，3 种子合计）"], diag)
    with open(os.path.join(out_dir, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    res = {}
    for cfg, st in stats.items():
        res[cfg] = {g: {s: {m: dict(zip(("mean", "min", "max"), point(st, g, s, m))) for m in ("acc", "f1_frl", "f1_4", "turn_recall", "lr_acc", "lr_cond", "false_turn",
                                                                                           "recall_S", "recall_F", "recall_L", "recall_R")}
                        for s in st["subsets"]} for g in st["groups"]}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, allow_nan=True)
    af.log("写 tables.md（%d 行）与 results.json" % len(lines))


def main():
    run_report(af.env("AUDIT_OUT"))


if __name__ == "__main__":
    main()
