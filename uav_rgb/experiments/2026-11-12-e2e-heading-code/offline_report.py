#!/usr/bin/env python3
"""卡 E2E-1 的离线部分：训练曲线、12 份交付权重的 val 指标、航向码臂的反事实探针。不起仿真。

读：$OUT/runs/*（A1、A2、A3 的训练 run）与 A0 的 e1-noise run（只读）：results.csv、metrics.jsonl；
    $OUT/offline/verify/<臂>-s<种子>/verify.json（verify_infer_vs_evalonly.py 的结果，训练器 eval-only 路的 val 指标）；
    交付权重 + 数据集（含侧车）：探针用 average_ckpt 的 read_aux_cond / infer_probs 逐帧推理。
写：$OUT/offline/{curves.csv, curves.png, offline.json, offline.md}。

探针（卡 E2E-1「另报」，A1 与 A2 的交付权重，val 分区剔除碰撞步；test 分区同样算一份只作补充，不是卡里写的口径）：
  对调：把条件码 2 与 3 互换（码 0 不变），重新推理。
    转向预测帧 = 真实码下预测为 YAW_LEFT(2) 或 YAW_RIGHT(3) 的帧；方向翻转 = 对调后预测变成另一个转向（2↔3）。
    给三个分母：全部转向预测帧；其中码非 0 的帧（对调只对这些帧有输入上的变化，码 0 的帧输入不变，翻转为 0）；
    以及标签是转向且预测也是转向的帧（全部 / 码非 0）。同时给码非 0 帧里预测有任何变化的比例。
  恒置 0：把全部帧的码置 0，重新推理；报 macro-F1、acc、CE 对真实码的变化量。
  这两个数说明网络有没有在用这个码：不用，翻转比例与 macro-F1 变化都接近 0。
macro-F1 与训练器同定义（缺类按 0 计、除以 4），见 verify_infer_vs_evalonly.metrics_from_probs。
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_arms as arms  # noqa: E402
import verify_infer_vs_evalonly as vf  # noqa: E402

TURN = (2, 3)
EPOCH_MARKS = (1, 5, 10, 15, 20, 25, 30)
LAST_K = 10


# ---------------------------------------------------------------- 训练曲线
def read_curves(run):
    """训练 run 目录 → ([{"epoch", "train_ce", "dev_ce", "dev_acc", "dev_macro_f1", "dev_bal_acc"}], 说明, best_epoch)。
    results.csv 给逐轮 train_ce / dev_ce / dev_acc；metrics.jsonl 里 split=val 的 validation 事件按顺序对应各轮，给 macro-F1 与 balanced accuracy。
    两边条数对不上时 macro-F1 与 balanced accuracy 记 None 并写进说明，不猜对应关系。"""
    rows = []
    with open(os.path.join(run, "results.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"epoch": int(r["epoch"]), "train_ce": float(r["train_ce"]), "dev_ce": float(r["dev_ce"]), "dev_acc": float(r["dev_accuracy"]),
                         "dev_macro_f1": None, "dev_bal_acc": None})
    events, best_epoch = [], None
    with open(os.path.join(run, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"validation"' in line:
                e = json.loads(line)
                if e.get("event") == "validation" and e.get("split") == "val":
                    events.append(e)
            elif "best checkpoint" in line:
                m = re.search(r"best checkpoint: epoch (\d+)", line)
                if m:
                    best_epoch = int(m.group(1))
    note = ""
    if len(events) == len(rows):
        for r, e in zip(rows, events):
            r["dev_macro_f1"], r["dev_bal_acc"] = float(e["macro_f1"]), float(e["balanced_accuracy"])
    else:
        note = f"metrics.jsonl 的 val 事件 {len(events)} 条与 results.csv 的 {len(rows)} 轮对不上，macro-F1 与 balanced accuracy 缺"
    return rows, note, best_epoch


def window_mean(rows, key, k=LAST_K):
    vals = [r[key] for r in rows[-k:] if r[key] is not None]
    return float(np.mean(vals)) if vals else None


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def curves_section(curves):
    """curves：{(臂, 种子): (rows, note, best_epoch)} → markdown 行列表。"""
    out = ["## 1. 训练曲线（每轮 val：dev_ce 为无权 CE，macro-F1 缺类按 0 计；三个训练种子取均值，括号内为种子间极差）", ""]
    for key, title in (("dev_ce", "dev CE"), ("dev_macro_f1", "dev macro-F1")):
        out += [f"### {title}（均值，括号内极差）", "", "| 臂 | " + " | ".join(f"epoch {e}" for e in EPOCH_MARKS) + f" | 末 {LAST_K} 轮均值 |", "| --- |" + " --- |" * (len(EPOCH_MARKS) + 1)]
        for a in arms.ARMS:
            cells = []
            for e in EPOCH_MARKS:
                vs = [next((r[key] for r in curves[(a, s)][0] if r["epoch"] == e), None) for s in arms.SEEDS_TRAIN if (a, s) in curves]
                vs = [v for v in vs if v is not None]
                cells.append(f"{np.mean(vs):.3f} ({max(vs) - min(vs):.3f})" if vs else "-")
            wm = [window_mean(curves[(a, s)][0], key) for s in arms.SEEDS_TRAIN if (a, s) in curves]
            wm = [v for v in wm if v is not None]
            cells.append(f"{np.mean(wm):.3f} ({max(wm) - min(wm):.3f})" if wm else "-")
            out.append(f"| {arms.LABEL[a]} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["### ckpt_best 所在轮（trainer 按 dev macro-F1 选；交付权重不用它，用末 10 轮平均）", "", "| 臂 | 种子 0 | 种子 1 | 种子 2 |", "| --- | --- | --- | --- |"]
    for a in arms.ARMS:
        out.append(f"| {arms.LABEL[a]} | " + " | ".join(str(curves[(a, s)][2]) if (a, s) in curves else "-" for s in arms.SEEDS_TRAIN) + " |")
    notes = [f"{arms.LABEL[a]} s{s}：{curves[(a, s)][1]}" for a in arms.ARMS for s in arms.SEEDS_TRAIN if (a, s) in curves and curves[(a, s)][1]]
    if notes:
        out += ["", "说明：" + "；".join(notes)]
    return out + [""]


def write_curves(out_dir, curves):
    path = os.path.join(out_dir, "curves.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "epoch", "train_ce", "dev_ce", "dev_acc", "dev_macro_f1", "dev_bal_acc"])
        for (a, s), (rows, _n, _b) in sorted(curves.items()):
            for r in rows:
                w.writerow([a, s, r["epoch"], r["train_ce"], r["dev_ce"], r["dev_acc"], "" if r["dev_macro_f1"] is None else r["dev_macro_f1"],
                            "" if r["dev_bal_acc"] is None else r["dev_bal_acc"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return path, None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"a0": "tab:gray", "a1": "tab:blue", "a2": "tab:red", "a3": "tab:green"}
    for ax, key, title in ((axes[0], "dev_ce", "dev CE (val, unweighted)"), (axes[1], "dev_macro_f1", "dev macro-F1 (val)")):
        for a in arms.ARMS:
            series = [[r[key] for r in curves[(a, s)][0]] for s in arms.SEEDS_TRAIN if (a, s) in curves]
            series = [x for x in series if x and all(v is not None for v in x)]
            for x in series:
                ax.plot(range(1, len(x) + 1), x, color=colors[a], alpha=0.25, linewidth=0.8)
            if series and len({len(x) for x in series}) == 1:
                ax.plot(range(1, len(series[0]) + 1), np.mean(series, axis=0), color=colors[a], linewidth=2, label=arms.LABEL[a])
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    png = os.path.join(out_dir, "curves.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return path, png


# ---------------------------------------------------------------- 交付权重的 val 指标
def recall_from_confusion(conf):
    c = np.asarray(conf, dtype=np.int64).reshape(4, 4)
    row = c.sum(1)
    return [None if row[k] == 0 else float(c[k, k] / row[k]) for k in range(4)], [int(x) for x in c.sum(0)]


def delivered_section(out_root, verify):
    out = ["## 2. 交付权重的离线指标（val，剔除碰撞步；数取自训练器 eval-only 一路，与逐帧推理一路的对拍结果见各 verify.json）", "",
           "val 是校准 δ 用过的分区，不是独立的测试集；A0、A3 与 A1、A2 的数据集不同（demo 与 dagger），val 样本不同，不要跨数据集比绝对数。", "",
           "| 臂 | 种子 | val 帧数 | acc | macro-F1 | balanced acc | dev CE | 召回 FORWARD / LEFT / RIGHT | 预测占比 SAFE / FWD / LEFT / RIGHT | 校准 δ |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in arms.ARMS:
        for s in arms.SEEDS_TRAIN:
            v = verify.get((a, s))
            if v is None:
                out.append(f"| {arms.LABEL[a]} | {s} | 未完成 | | | | | | | |")
                continue
            t = v["trainer"]
            rec, cols = recall_from_confusion(t["confusion"])
            n = sum(cols)
            sidecar = arms.ckpt_path(out_root, a, s) + ".json"
            delta = None
            if os.path.isfile(sidecar):
                with open(sidecar, encoding="utf-8") as f:
                    delta = json.load(f).get("delta")
            out.append(f"| {arms.LABEL[a]} | {s} | {v['val_frames']} | {t['acc']:.4f} | {t['macro_f1']:.4f} | {t['balanced_acc']:.4f} | {t['ce']:.4f} | "
                       f"{fmt(rec[1], 3)} / {fmt(rec[2], 3)} / {fmt(rec[3], 3)} | " + " / ".join(f"{c / n:.3f}" for c in cols) + f" | {fmt(delta, 4)} |")
    return out + [""]


# ---------------------------------------------------------------- 反事实探针（纯函数，test_offline_report.py 覆盖）
def swap_codes(cond):
    """码 2 与 3 互换；其余（0、1）不变。"""
    out = np.array(cond, copy=True)
    out[cond == 2], out[cond == 3] = 3, 2
    return out


def _frac(num, den):
    return {"n": int(den), "flip": int(num), "frac": None if den == 0 else float(num / den)}


def probe_stats(y, cond, probs_real, probs_swap, probs_zero):
    """一个权重在一个分区上的探针。y、cond 是 u8[n]；probs_* 是 [n,4]，分别为真实码、对调码、恒置 0 下的概率。"""
    pr, ps = probs_real.argmax(1), probs_swap.argmax(1)
    turn_pred = np.isin(pr, TURN)
    flipped = turn_pred & np.isin(ps, TURN) & (ps != pr)          # 转向预测帧里，对调后变成另一个转向
    nz = cond != 0
    turn_label = np.isin(y, TURN)
    both = turn_label & turn_pred
    m_real, m_zero = vf.metrics_from_probs(probs_real, y), vf.metrics_from_probs(probs_zero, y)
    return {
        "frames": int(len(y)),
        "code_counts": [int((cond == v).sum()) for v in range(4)],
        "swap": {
            "turn_pred_all": _frac(flipped.sum(), turn_pred.sum()),
            "turn_pred_code_nonzero": _frac((flipped & nz).sum(), (turn_pred & nz).sum()),
            "turn_label_and_pred_all": _frac((flipped & both).sum(), both.sum()),
            "turn_label_and_pred_code_nonzero": _frac((flipped & both & nz).sum(), (both & nz).sum()),
            "any_change_code_nonzero": _frac(((ps != pr) & nz).sum(), nz.sum()),
            "any_change_code_zero": _frac(((ps != pr) & ~nz).sum(), (~nz).sum()),   # 码 0 的帧输入没变，应为 0（自检）
        },
        "zero": {"real": m_real, "zeroed": m_zero,
                 "delta_macro_f1": m_zero["macro_f1"] - m_real["macro_f1"], "delta_acc": m_zero["acc"] - m_real["acc"], "delta_ce": m_zero["ce"] - m_real["ce"],
                 "pred_changed_frames": int((probs_zero.argmax(1) != pr).sum())},
    }


def read_split(ac, path, split):
    """dataset.uavr 的某个 split（剔除碰撞步，口径同 average_ckpt.read_val）→ (images NCHW memmap, 样本下标, 标签)。"""
    with open(path, "rb") as f:
        magic, ver, hs, n, h, w, c, idt, ldt, _r, io, lo, mo, ml = ac.DS_HEAD.unpack(f.read(ac.DS_HEAD.size))
        if (magic, ver, hs, c, idt, ldt) != (b"UAVR", 1, 48, 3, 1, 1):
            raise ValueError(f"{path}: 不是 UAVR v1 的 RGB uint8 数据集")
        f.seek(mo)
        meta = json.loads(f.read(ml).decode("utf-8"))
    flags = meta.get("collision_flags")
    if not isinstance(flags, str) or len(flags) != n or set(flags) - {"0", "1"}:
        raise ValueError(f"{path}: meta 缺 collision_flags 或格式不符")
    idx = np.asarray([i for ep in meta["episodes"] if ep["split"] == split
                      for i in range(ep["sample_start"], ep["sample_start"] + ep["sample_count"]) if flags[i] == "0"], dtype=np.int64)
    if not idx.size:
        raise ValueError(f"{path}: {split} 分区剔除碰撞步后没有样本")
    return np.memmap(path, np.uint8, "r", io, (n, c, h, w)), idx, np.fromfile(path, dtype=np.uint8, count=n, offset=lo)[idx]


def run_probe(ac, ckpt, data, infer_bin, split):
    images, idx, y = read_split(ac, data, split)
    col, _aux = ac.read_aux_cond(data, vf.sha256_file(data))
    cond = col[idx]
    p_real = ac.infer_probs(infer_bin, ckpt, images, idx, cond)
    p_swap = ac.infer_probs(infer_bin, ckpt, images, idx, swap_codes(cond))
    p_zero = ac.infer_probs(infer_bin, ckpt, images, idx, np.zeros_like(cond))
    return probe_stats(y, cond, p_real, p_swap, p_zero)


def pct(d):
    return "-" if d["frac"] is None else f"{d['flip']}/{d['n']}={100 * d['frac']:.1f}%"


def probes_section(probes):
    out = ["## 3. 反事实探针（A1、A2 的交付权重）", "",
           "对调 = 码 2 与 3 互换，量转向预测帧里方向翻转的比例；恒置 0 = 全部码置 0，量 macro-F1 的变化。码非 0 的帧在 demo 里很少，分母见每格。", ""]
    for split in ("val", "test"):
        title = "val（卡里写的口径）" if split == "val" else "test（补充，不是卡里写的口径）"
        out += [f"### {title}", "",
                "| 臂 | 种子 | 帧数 | 码计数 0/1/2/3 | 转向预测帧翻转：全部 | 其中码非 0 | 标签与预测都转向：全部 | 其中码非 0 | 码非 0 帧预测有变化 | 码 0 帧预测有变化（自检，应为 0） | 恒置 0：macro-F1 真实 → 置 0（变化） | acc 变化 | CE 变化 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for a in ("a1", "a2"):
            for s in arms.SEEDS_TRAIN:
                p = probes.get((a, s, split))
                if p is None:
                    out.append(f"| {arms.LABEL[a]} | {s} | 未完成 | | | | | | | | | | |")
                    continue
                w, z = p["swap"], p["zero"]
                out.append(f"| {arms.LABEL[a]} | {s} | {p['frames']} | {'/'.join(map(str, p['code_counts']))} | {pct(w['turn_pred_all'])} | {pct(w['turn_pred_code_nonzero'])} | "
                           f"{pct(w['turn_label_and_pred_all'])} | {pct(w['turn_label_and_pred_code_nonzero'])} | {pct(w['any_change_code_nonzero'])} | {pct(w['any_change_code_zero'])} | "
                           f"{z['real']['macro_f1']:.4f} → {z['zeroed']['macro_f1']:.4f}（{z['delta_macro_f1']:+.4f}） | {z['delta_acc']:+.4f} | {z['delta_ce']:+.4f} |")
        out.append("")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="实验输出目录 $OUT")
    ap.add_argument("--tools-dir", default=os.path.join(HERE, "..", "..", "tools"), help="含 average_ckpt.py 的目录")
    ap.add_argument("--infer-bin", required=True)
    ap.add_argument("--skip-probes", action="store_true", help="只出曲线与 val 指标（调试用）")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.tools_dir))
    import average_ckpt as ac

    out_root = os.path.abspath(args.out_root)
    out_dir = os.path.join(out_root, "offline")
    os.makedirs(out_dir, exist_ok=True)
    curves = {(a, s): read_curves(arms.run_dir(out_root, a, s)) for a in arms.ARMS for s in arms.SEEDS_TRAIN if os.path.isfile(os.path.join(arms.run_dir(out_root, a, s), "results.csv"))}
    csv_path, png_path = write_curves(out_dir, curves)
    verify = {}
    for a in arms.ARMS:
        for s in arms.SEEDS_TRAIN:
            p = os.path.join(out_dir, "verify", f"{arms.NAME[a]}-s{s}", "verify.json")
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    verify[(a, s)] = json.load(f)
    probes = {}
    if not args.skip_probes:
        for a in ("a1", "a2"):
            for s in arms.SEEDS_TRAIN:
                for split in ("val", "test"):
                    probes[(a, s, split)] = run_probe(ac, arms.ckpt_path(out_root, a, s), arms.data_path(out_root, a), args.infer_bin, split)
                    print(f"探针 {arms.NAME[a]} s{s} {split}：{probes[(a, s, split)]['swap']['turn_pred_code_nonzero']}", flush=True)
    md = ["# E2E-1 离线报告（训练曲线、交付权重的 val 指标、反事实探针）", "",
          f"曲线图 {png_path or '（没有 matplotlib，未出图）'}；逐轮数据 {csv_path}。", ""]
    md += curves_section(curves) + delivered_section(out_root, verify) + (probes_section(probes) if probes else [])
    with open(os.path.join(out_dir, "offline.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "offline.json"), "w", encoding="utf-8") as f:
        json.dump({"verify": {f"{a}-s{s}": v for (a, s), v in verify.items()},
                   "probes": {f"{a}-s{s}-{sp}": v for (a, s, sp), v in probes.items()},
                   "curves": {f"{a}-s{s}": {"rows": r[0], "note": r[1], "best_epoch": r[2]} for (a, s), r in curves.items()}}, f, indent=1, ensure_ascii=False)
    print(f"写入 {out_dir}/offline.md、offline.json、curves.csv" + (f"、curves.png" if png_path else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
