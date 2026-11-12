#!/usr/bin/env python3
"""复训对拍（任务卡 E2E-R1 第一部分的闸门）：新 run 与参照 run（同数据、同配方、同种子）逐轮对账。

用法：gate_check.py --ref <参照 run 目录> --new <复训 run 目录> --out <gate.json> [--max-dmf1 0.05]
读两边的 results.csv（train_ce、dev_ce、dev_accuracy）、metrics.jsonl 的 val 事件（macro_f1、balanced_accuracy）、config.json，
以及逐轮 ckpt_epNN.bin 的参数张量（用 tools/average_ckpt.read_ckpt 解析，比的是权重，不是文件字节：头里的契约段可能带构建信息）。
闸门：两边都有的最后一轮，val macro-F1 的差的绝对值 >= --max-dmf1（任务书给的 0.05）就不过。
退出码：0 过；1 不过（停下报告）；2 输入不全或口径对不上。
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import average_ckpt as ac  # noqa: E402

IGNORE_CONFIG_KEYS = ("source_git",)   # 从哪个目录起训就记哪个 git 的 HEAD；复训从非 git 目录起，空串是预期的


def read_results(run):
    """results.csv → {epoch: {train_ce, dev_ce, dev_accuracy}}。"""
    path = os.path.join(run, "results.csv")
    with open(path, encoding="utf-8") as f:
        return {int(r["epoch"]): {k: float(r[k]) for k in ("train_ce", "dev_ce", "dev_accuracy")} for r in csv.DictReader(f)}


def read_val_events(run):
    """metrics.jsonl 里 split=val 的 validation 事件，按出现顺序即轮序 → [{macro_f1, balanced_accuracy, loss, action_accuracy}]。"""
    out = []
    with open(os.path.join(run, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"event":"validation"' in line and '"split":"val"' in line:
                j = json.loads(line)
                out.append({k: j[k] for k in ("macro_f1", "balanced_accuracy", "loss", "action_accuracy")})
    return out


def epoch_ckpts(run):
    found = {}
    for n in os.listdir(run):
        m = re.fullmatch(r"ckpt_ep(\d+)\.bin", n)
        if m:
            found[int(m.group(1))] = os.path.join(run, n)
    return found


def compare_weights(ref, new, epochs):
    """逐轮比参数张量：{epoch: {"identical": bool, "max_abs_diff": float}}。"""
    a, b = epoch_ckpts(ref), epoch_ckpts(new)
    out = {}
    for e in epochs:
        if e not in a or e not in b:
            continue
        pa, pb = ac.read_ckpt(a[e])["params"], ac.read_ckpt(b[e])["params"]
        same = len(pa) == len(pb) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(pa, pb))
        diff = max(float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) for x, y in zip(pa, pb)) if len(pa) == len(pb) else float("nan")
        out[e] = {"identical": bool(same), "max_abs_diff": diff}
    return out


def config_diff(ref, new):
    with open(os.path.join(ref, "config.json"), encoding="utf-8") as f:
        a = json.load(f)
    with open(os.path.join(new, "config.json"), encoding="utf-8") as f:
        b = json.load(f)
    return {k: {"ref": a.get(k), "new": b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dmf1", type=float, default=0.05, help="末轮 val macro-F1 之差的闸门（任务书：0.05）")
    args = ap.parse_args(argv)
    try:
        rr, nr = read_results(args.ref), read_results(args.new)
        rv, nv = read_val_events(args.ref), read_val_events(args.new)
        cdiff = config_diff(args.ref, args.new)
    except (OSError, KeyError, ValueError) as e:
        print(f"gate_check: 输入不全或格式不符：{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    epochs = sorted(set(rr) & set(nr))
    if not epochs or len(rv) < len(epochs) or len(nv) < len(epochs):
        print("gate_check: 两边没有共同的轮，或 val 事件数不足", file=sys.stderr)
        return 2
    last = epochs[-1]
    per_epoch = {}
    for e in epochs:
        per_epoch[e] = {"d_dev_ce": nr[e]["dev_ce"] - rr[e]["dev_ce"], "d_dev_accuracy": nr[e]["dev_accuracy"] - rr[e]["dev_accuracy"],
                        "d_train_ce": nr[e]["train_ce"] - rr[e]["train_ce"],
                        "d_macro_f1": nv[e - 1]["macro_f1"] - rv[e - 1]["macro_f1"],
                        "d_balanced_accuracy": nv[e - 1]["balanced_accuracy"] - rv[e - 1]["balanced_accuracy"]}
    weights = compare_weights(args.ref, args.new, epochs)
    identical = [e for e, v in weights.items() if v["identical"]]
    d_last = per_epoch[last]["d_macro_f1"]
    ok = abs(d_last) < args.max_dmf1
    res = {"ref": os.path.abspath(args.ref), "new": os.path.abspath(args.new), "epochs_compared": len(epochs), "last_epoch": last,
           "last": {"ref": {"macro_f1": rv[last - 1]["macro_f1"], "balanced_accuracy": rv[last - 1]["balanced_accuracy"], **rr[last]},
                    "new": {"macro_f1": nv[last - 1]["macro_f1"], "balanced_accuracy": nv[last - 1]["balanced_accuracy"], **nr[last]},
                    **per_epoch[last]},
           "max_abs_d_macro_f1_all_epochs": max(abs(v["d_macro_f1"]) for v in per_epoch.values()),
           "max_abs_d_dev_ce_all_epochs": max(abs(v["d_dev_ce"]) for v in per_epoch.values()),
           "weights_identical_epochs": identical, "weights_not_identical_epochs": [e for e in epochs if e in weights and e not in identical],
           "weights_max_abs_diff_all_epochs": max((v["max_abs_diff"] for v in weights.values()), default=None),
           "config_diff": cdiff, "config_diff_ignored_keys": list(IGNORE_CONFIG_KEYS),
           "gate": {"max_dmf1": args.max_dmf1, "d_macro_f1_last": d_last, "pass": bool(ok)}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    bad_cfg = {k: v for k, v in cdiff.items() if k not in IGNORE_CONFIG_KEYS}
    print(f"gate_check: 对比 {len(epochs)} 轮；末轮（ep{last}）val macro-F1 参照 {res['last']['ref']['macro_f1']:.6f} 新 {res['last']['new']['macro_f1']:.6f} "
          f"差 {d_last:+.6f}（闸门 {args.max_dmf1}）→ {'过' if ok else '不过'}；权重逐位相同的轮 {len(identical)}/{len(epochs)}；"
          f"全程 val macro-F1 最大差 {res['max_abs_d_macro_f1_all_epochs']:.3g}；配置差别（不含 {','.join(IGNORE_CONFIG_KEYS)}）{bad_cfg or '无'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
