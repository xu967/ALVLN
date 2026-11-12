#!/usr/bin/env python3
"""卡 E2E-P0 第三关：原型训练与 C++ 训练的逐项对照。定位工具，不是验收判据。

同一个种子的两条线（原型 vs C++）从逐位相同的初始权重和逐位相同的打乱顺序出发，
所以差异只来自浮点舍入顺序（cuDNN 卷积 vs 手写卷积，Adam 把微小梯度的噪声放大）。本脚本回答三个问题：
  1. 前几步 loss 一致到几位，第一次超过 1e-5 / 1e-4 / 1e-3 / 1e-2 是第几步；
  2. 逐轮 val 宏 F1、val CE、train CE 两条线差多少；
  3. 同种子两条线的权重距离，与 C++ 自己「不同种子」之间的权重距离相比有多大。
     距离 = ||θa - θb||₂ / ||θb||₂（全部参数拼成 float64 向量）。

用法（任一 Python 3 + numpy）：
  python proto/compare_training.py --pair p0:<原型 run>:<C++ run> [--pair ...] [--extra c3:<C++ run> ...] \\
      --out-json <json> --out-md <md>
  --pair NAME:PROTO_DIR:REF_DIR  同种子的一对（两边都是 run 目录：含 metrics.jsonl、results.csv、ckpt_epNN.bin）。
                                 C++ 重跑对 C++ 参照也可以用这个形式。
  --extra NAME:DIR               只进「C++ 不同种子」基线的额外 C++ run。
退出码：0 正常；2 输入不全。
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io  # noqa: E402

N_FIRST = 12                                          # 表里列出的前几步
THRESHOLDS = (1e-5, 1e-4, 1e-3, 1e-2)                 # 「第一次超过」的 |Δloss| 阈值
EPOCHS_SHOWN = (1, 2, 3, 5, 10, 20, 30)               # 权重距离与逐轮表里列出的轮
LAYER_NAMES = ("c1.W", "c1.b", "c2.W", "c2.b", "c3.W", "c3.b", "fc.W", "fc.b")   # 前 8 个参数张量（无 FiLM/文本/上一动作的结构）


def read_events(run):
    out = []
    with open(os.path.join(run, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def read_results(run):
    with open(os.path.join(run, "results.csv"), encoding="utf-8", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def step_losses(run):
    """{step: (loss, action_accuracy)}"""
    return {e["step"]: (e["loss"], e["action_accuracy"]) for e in read_events(run) if e.get("event") == "train_step"}


def val_f1(run):
    return [e["macro_f1"] for e in read_events(run) if e.get("event") == "validation" and e.get("split") == "val"]


def compare_steps(proto_run, ref_run):
    a, b = step_losses(proto_run), step_losses(ref_run)
    steps = sorted(set(a) & set(b))
    d = np.array([abs(a[s][0] - b[s][0]) for s in steps]) if steps else np.zeros(0)
    first = {}
    for t in THRESHOLDS:
        idx = np.flatnonzero(d > t)
        first[f"{t:g}"] = int(steps[idx[0]]) if idx.size else None
    acc_same = sum(1 for s in steps if a[s][1] == b[s][1])
    return {
        "steps_compared": len(steps),
        "first_rows": [{"step": s, "proto": a[s][0], "ref": b[s][0], "diff": a[s][0] - b[s][0]} for s in steps[:N_FIRST]],
        "first_exceed": first,
        "max_abs_diff_first_epoch": float(d[:65].max()) if d.size else None,
        "max_abs_diff_all": float(d.max()) if d.size else None,
        "mean_abs_diff_all": float(d.mean()) if d.size else None,
        "accuracy_identical_steps": acc_same,
    }


def compare_epochs(proto_run, ref_run):
    pr, rr = read_results(proto_run), read_results(ref_run)
    pf, rf = val_f1(proto_run), val_f1(ref_run)
    n = min(len(pr), len(rr), len(pf), len(rf))
    rows = []
    for i in range(n):
        rows.append({
            "epoch": i + 1,
            "train_ce": [pr[i]["train_ce"], rr[i]["train_ce"]],
            "dev_ce": [pr[i]["dev_ce"], rr[i]["dev_ce"]],
            "val_macro_f1": [pf[i], rf[i]],
        })
    def mad(key):
        return float(np.mean([abs(r[key][0] - r[key][1]) for r in rows])) if rows else None
    return {"epochs_compared": n, "rows": rows,
            "mean_abs_diff": {"train_ce": mad("train_ce"), "dev_ce": mad("dev_ce"), "val_macro_f1": mad("val_macro_f1")}}


def flat(ck):
    return np.concatenate([np.asarray(p, np.float64).reshape(-1) for p in ck.params])


def rel_dist(ca, cb):
    a, b = flat(ca), flat(cb)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def layer_dist(ca, cb):
    out = {}
    for name, pa, pb in zip(LAYER_NAMES, ca.params, cb.params):
        a, b = np.asarray(pa, np.float64).reshape(-1), np.asarray(pb, np.float64).reshape(-1)
        nb = np.linalg.norm(b)
        out[name] = float(np.linalg.norm(a - b) / nb) if nb > 0 else None
    return out


def ckpt_at(run, epoch):
    p = os.path.join(run, f"ckpt_ep{epoch:02d}.bin")
    return ckpt_io.read_checkpoint(p) if os.path.exists(p) else None


def compare_weights(proto_run, ref_run, epochs):
    out = {}
    for e in epochs:
        ca, cb = ckpt_at(proto_run, e), ckpt_at(ref_run, e)
        if ca is None or cb is None:
            continue
        out[e] = {"rel_l2": rel_dist(ca, cb), "per_tensor": layer_dist(ca, cb)}
    return out


def baseline_weights(runs, epochs):
    """C++ 不同种子之间：{epoch: {"pairs": {"a|b": d}, "mean": m}}；每对取两个方向距离的平均（分母不同）。"""
    names = sorted(runs)
    out = {}
    for e in epochs:
        cks = {n: ckpt_at(runs[n], e) for n in names}
        pairs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if cks[a] is None or cks[b] is None:
                    continue
                pairs[f"{a}|{b}"] = 0.5 * (rel_dist(cks[a], cks[b]) + rel_dist(cks[b], cks[a]))
        if pairs:
            out[e] = {"pairs": pairs, "mean": float(np.mean(list(pairs.values())))}
    return out


def build(pairs, extras, epochs):
    result = {"pairs": {}, "baseline_other_seeds": {}, "epochs_shown": list(epochs)}
    base_runs = {}
    seen = set()                                      # 同一个 C++ run 出现在多个对里（如参照种子 0 与它的重跑）只算一次
    for name, (pr, rr) in pairs.items():
        result["pairs"][name] = {
            "proto_run": pr, "ref_run": rr,
            "steps": compare_steps(pr, rr),
            "epochs": compare_epochs(pr, rr),
            "weights": compare_weights(pr, rr, epochs),
        }
        if os.path.realpath(rr) not in seen:
            seen.add(os.path.realpath(rr))
            base_runs[f"ref:{name}"] = rr
    for n, d in extras.items():
        if os.path.realpath(d) not in seen:
            seen.add(os.path.realpath(d))
            base_runs[f"extra:{n}"] = d
    result["baseline_other_seeds"] = baseline_weights(base_runs, epochs)
    return result


def fmt(x, nd=6):
    return "-" if x is None else f"{x:.{nd}g}"


def render_md(res):
    L = ["# 第三关：原型训练与 C++ 训练的逐项对照", "",
         "同种子两条线从逐位相同的初始权重与打乱顺序出发；差异只来自浮点舍入顺序。本表是定位依据，不是验收判据。", ""]
    L += ["## 1. 前几步 loss（原型 vs C++）", ""]
    for name, p in res["pairs"].items():
        s = p["steps"]
        L += [f"### {name}", "", f"- 对比的步数 {s['steps_compared']}；动作准确率逐步一致的步数 {s['accuracy_identical_steps']}。",
              f"- |Δloss| 第一次超过阈值的步：" + "；".join(f"> {t} 在第 {v} 步" if v else f"> {t} 未出现" for t, v in s["first_exceed"].items()) + "。",
              f"- |Δloss| 最大值：第 1 轮内 {fmt(s['max_abs_diff_first_epoch'], 3)}；全部步 {fmt(s['max_abs_diff_all'], 3)}；全部步均值 {fmt(s['mean_abs_diff_all'], 3)}。", "",
              "| step | 原型 loss | C++ loss | 差 |", "|---|---|---|---|"]
        for r in s["first_rows"]:
            L.append(f"| {r['step']} | {r['proto']:.9g} | {r['ref']:.9g} | {r['diff']:+.2e} |")
        L.append("")
    L += ["## 2. 逐轮指标（原型 / C++）", ""]
    for name, p in res["pairs"].items():
        e = p["epochs"]
        m = e["mean_abs_diff"]
        L += [f"### {name}", "", f"- 对比的轮数 {e['epochs_compared']}；逐轮平均绝对差：train CE {fmt(m['train_ce'], 3)}，val CE {fmt(m['dev_ce'], 3)}，val 宏 F1 {fmt(m['val_macro_f1'], 3)}。", "",
              "| 轮 | train CE | val CE | val 宏 F1 |", "|---|---|---|---|"]
        for r in e["rows"]:
            if r["epoch"] in res["epochs_shown"]:
                L.append(f"| {r['epoch']} | {r['train_ce'][0]:.4f} / {r['train_ce'][1]:.4f} | {r['dev_ce'][0]:.4f} / {r['dev_ce'][1]:.4f} | "
                         f"{r['val_macro_f1'][0]:.4f} / {r['val_macro_f1'][1]:.4f} |")
        L.append("")
    L += ["## 3. 权重距离 ||θa-θb||/||θb||", "",
          "同种子（原型 vs C++）与 C++ 不同种子之间的距离。不同种子的初始权重不同，距离是「完全独立」的量级；"
          "同种子的距离随轮数变大，是浮点混沌放大，不是配方差异的直接证据。", ""]
    head = "| 轮 | " + " | ".join(res["pairs"]) + " | C++ 不同种子均值（对数） |"
    L += [head, "|---|" + "---|" * (len(res["pairs"]) + 1)]
    for e in res["epochs_shown"]:
        cells = []
        for name, p in res["pairs"].items():
            w = p["weights"].get(e) or p["weights"].get(str(e))
            cells.append(f"{w['rel_l2']:.4f}" if w else "-")
        b = res["baseline_other_seeds"].get(e) or res["baseline_other_seeds"].get(str(e))
        L.append(f"| {e} | " + " | ".join(cells) + f" | {b['mean']:.4f} ({len(b['pairs'])}) |" if b else f"| {e} | " + " | ".join(cells) + " | - |")
    L += ["", "括号里是基线用到的 C++ 种子对数。", ""]
    return "\n".join(L)


def parse_pair(s):
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--pair 需要 NAME:PROTO_DIR:REF_DIR，收到 {s!r}")
    return parts


def parse_extra(s):
    parts = s.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--extra 需要 NAME:DIR，收到 {s!r}")
    return parts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", action="append", type=parse_pair, default=[])
    ap.add_argument("--extra", action="append", type=parse_extra, default=[])
    ap.add_argument("--epochs", type=int, nargs="*", default=list(EPOCHS_SHOWN))
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    args = ap.parse_args(argv)
    if not args.pair:
        print("至少要一个 --pair", file=sys.stderr)
        return 2
    pairs = {}
    for name, pr, rr in args.pair:
        for d in (pr, rr):
            for f in ("metrics.jsonl", "results.csv"):
                if not os.path.exists(os.path.join(d, f)):
                    print(f"缺 {os.path.join(d, f)}", file=sys.stderr)
                    return 2
        pairs[name] = (pr, rr)
    res = build(pairs, dict(args.extra), tuple(args.epochs))
    md = render_md(res)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    if not (args.out_json or args.out_md):
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
