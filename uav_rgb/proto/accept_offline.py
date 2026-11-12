#!/usr/bin/env python3
"""卡 E2E-P0 验收 (a)：PyTorch 原型训练的离线指标与 C++ 三个参照种子对拍。

任务卡判据：trunc 数据集训 3 个种子；val 与 test 的宏 F1、每类召回，都落在 C++ 三个参照种子的范围放宽 ±0.03 之内。
读法（起跑前定下，不看结果改）：
  交付口径  每个 run 的交付权重 = ckpt_best（val 宏 F1 最高的一轮）。val 指标取该轮的 val 事件；test 指标取训练末尾对 ckpt_best 的那次评估。
            指标全部由事件里的混淆矩阵重算（行 = 真值，列 = 预测），两条线同一个口径；与事件里记的宏 F1 的差另列。
  范围      C++ 参照种子各自的值取 min 与 max，两端各放宽 TOL。
  主判据    字面读法：原型的每个种子 × val/test × {宏 F1，类 0..3 召回} 都在范围内才算 (a) 通过。
  补充读法  只报告，不改判：(1) 去掉真值样本数 <= 1 的类的召回（val 没有类 0，test 只有 1 帧类 0，这两个数不携带信息）；
            (2) 三个原型种子的均值在范围内；(3) 有额外 C++ 种子时，同一条判据套在 C++ 自己的新种子上（字面读法对「等价配方」的误判率）。

用法（任一 Python 3 + numpy 即可）：
  python proto/accept_offline.py --ref s0=<run> --ref s1=<run> --ref s2=<run> --proto p0=<run> --proto p1=<run> --proto p2=<run> \\
      [--control c3=<run> ...] --out-json <json> --out-md <md>
退出码：0 字面读法通过；1 未通过；2 输入不全。
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics as M  # noqa: E402

TOL = 0.03                                            # 任务卡：C++ 三个参照种子的范围放宽 ±0.03
METRICS = ("macro_f1", "recall0", "recall1", "recall2", "recall3")   # 任务卡：宏 F1 与每类召回
SPLITS = ("val", "test")
EPS = 1e-9                                            # 比较边界时的浮点余量
TINY = 1                                              # 真值样本数 <= TINY 的类，其召回不携带信息（补充读法 1）
SAMPLE_ID_TOL = 1e-6                                  # 重算的宏 F1 与事件里记的宏 F1（float32 打印）的容许差


def read_events(run):
    out = []
    with open(os.path.join(run, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def from_confusion(conf16, classes=4):
    conf = np.asarray(conf16, dtype=np.int64).reshape(classes, classes)
    s = M.summarize(conf)
    out = {"macro_f1": s["macro_f1"], "balanced_accuracy": s["balanced_accuracy"], "accuracy": s["accuracy"],
           "n_true": conf.sum(axis=1).tolist(), "confusion": [int(x) for x in conf.reshape(-1)]}
    for k, r in enumerate(s["recall"]):
        out[f"recall{k}"] = float(r)
    return out


def delivered(run):
    """一个 run 的交付口径指标：{best_epoch, val{...}, test{...}, logged_f1_diff}。"""
    ev = read_events(run)
    val = [e for e in ev if e.get("event") == "validation" and e.get("split") == "val"]
    test = [e for e in ev if e.get("event") == "validation" and e.get("split") == "test"]
    best_epoch = None
    for e in ev:
        msg = e.get("message", "") if e.get("event") == "status" else ""
        if msg.startswith("best checkpoint: epoch "):
            best_epoch = int(msg.split("epoch ")[1].split()[0])
    if best_epoch is None or not 1 <= best_epoch <= len(val) or not test:
        raise ValueError(f"{run}: 事件不全（best_epoch={best_epoch}，val 事件 {len(val)} 条，test 事件 {len(test)} 条）")
    out = {"run": run, "best_epoch": best_epoch, "epochs": len(val)}
    diffs = []
    for split, e in (("val", val[best_epoch - 1]), ("test", test[-1])):
        out[split] = from_confusion(e["confusion"])
        diffs.append(abs(out[split]["macro_f1"] - e["macro_f1"]))
    out["logged_f1_diff"] = max(diffs)
    return out


def ref_ranges(refs):
    """{(split, metric): (min, max)}"""
    return {(sp, m): (min(r[sp][m] for r in refs.values()), max(r[sp][m] for r in refs.values())) for sp in SPLITS for m in METRICS}


def is_tiny(sp, metric, refs):
    """召回指标对应的类，在该 split 里真值样本数 <= TINY（数据集定的，各 run 相同）。"""
    if not metric.startswith("recall"):
        return False
    k = int(metric[-1])
    return min(r[sp]["n_true"][k] for r in refs.values()) <= TINY


def inside(v, lo, hi):
    return lo - EPS <= v <= hi + EPS


def check(cand, rng, refs):
    """一个候选 run 对范围的逐项结果：{(split, metric): bool}"""
    return {(sp, m): inside(cand[sp][m], rng[(sp, m)][0] - TOL, rng[(sp, m)][1] + TOL) for sp in SPLITS for m in METRICS}


def verdict(per_cand_checks, refs, skip_tiny):
    fails = []
    for name, chk in per_cand_checks.items():
        for (sp, m), ok in chk.items():
            if skip_tiny and is_tiny(sp, m, refs):
                continue
            if not ok:
                fails.append(f"{name} {sp} {m}")
    return {"pass": not fails, "failures": fails}


def baseline(all_runs, skip_tiny):
    """C++ 自己的 run 之间套同一条判据：任取 3 个做参照，其余逐个（或逐三个）当候选。"""
    names = sorted(all_runs)
    per_seed, per_triple = [0, 0], [0, 0]
    for ref3 in itertools.combinations(names, 3):
        refs = {n: all_runs[n] for n in ref3}
        rng = ref_ranges(refs)
        rest = [n for n in names if n not in ref3]
        ok = {}
        for n in rest:
            chk = check(all_runs[n], rng, refs)
            ok[n] = verdict({n: chk}, refs, skip_tiny)["pass"]
            per_seed[0] += ok[n]
            per_seed[1] += 1
        for tri in itertools.combinations(rest, 3):
            per_triple[0] += all(ok[n] for n in tri)
            per_triple[1] += 1
    return {"per_seed_pass": per_seed[0], "per_seed_trials": per_seed[1],
            "per_triple_pass": per_triple[0], "per_triple_trials": per_triple[1]}


def evaluate(refs, protos, controls):
    rng = ref_ranges(refs)
    chk = {n: check(r, rng, refs) for n, r in protos.items()}
    rows = []
    for sp in SPLITS:
        for m in METRICS:
            lo, hi = rng[(sp, m)]
            vals = {n: r[sp][m] for n, r in protos.items()}
            rows.append({"split": sp, "metric": m, "ref": {n: r[sp][m] for n, r in refs.items()}, "ref_min": lo, "ref_max": hi,
                         "lo": lo - TOL, "hi": hi + TOL, "proto": vals, "ok": {n: chk[n][(sp, m)] for n in protos},
                         "tiny": is_tiny(sp, m, refs),
                         "proto_mean": float(np.mean(list(vals.values()))),
                         "mean_ok": inside(float(np.mean(list(vals.values()))), lo - TOL, hi + TOL)})
    res = {"tolerance": TOL, "refs": {n: {k: r[k] for k in ("run", "best_epoch", "val", "test")} for n, r in refs.items()},
           "protos": {n: {k: r[k] for k in ("run", "best_epoch", "val", "test")} for n, r in protos.items()},
           "rows": rows,
           "literal": verdict(chk, refs, skip_tiny=False),
           "excluding_tiny_classes": verdict(chk, refs, skip_tiny=True),
           "mean_in_range": {"pass": all(r["mean_ok"] for r in rows if not r["tiny"]),
                             "failures": [f"{r['split']} {r['metric']}" for r in rows if not r["tiny"] and not r["mean_ok"]]},
           "logged_f1_diff_max": max(r["logged_f1_diff"] for r in list(refs.values()) + list(protos.values()) + list(controls.values()))}
    if controls:
        cchk = {n: check(r, rng, refs) for n, r in controls.items()}
        res["controls"] = {"runs": {n: {k: r[k] for k in ("run", "best_epoch", "val", "test")} for n, r in controls.items()},
                           "literal": {n: verdict({n: cchk[n]}, refs, False) for n in controls},
                           "excluding_tiny_classes": {n: verdict({n: cchk[n]}, refs, True) for n in controls}}
        allr = dict(refs)
        allr.update(controls)
        if len(allr) >= 6:
            res["controls"]["baseline_literal"] = baseline(allr, False)
            res["controls"]["baseline_excluding_tiny"] = baseline(allr, True)
    return res


def f3(x):
    return f"{x:.3f}"


def render_md(res):
    refs, protos = list(res["refs"]), list(res["protos"])
    L = ["# 验收 (a)：离线指标对拍（PyTorch 原型 vs C++ 参照种子）", "",
         f"- 参照种子：{', '.join(refs)}；原型种子：{', '.join(protos)}；范围放宽 ±{TOL}。",
         f"- 交付口径：ckpt_best；val 取最佳轮的 val 事件，test 取训练末尾对 ckpt_best 的评估。最佳轮：" +
         "；".join(f"{n} 第 {r['best_epoch']} 轮" for n, r in list(res['refs'].items()) + list(res['protos'].items())) + "。",
         f"- 指标由混淆矩阵重算；与事件里记的宏 F1 的最大差 {res['logged_f1_diff_max']:.2e}（事件按 float32 打印）。", "",
         "## 1. 逐项（字面读法）", "",
         "| split | 指标 | " + " | ".join(f"C++ {n}" for n in refs) + " | 范围（含 ±0.03） | " +
         " | ".join(f"原型 {n}" for n in protos) + " | 全部在范围内 | 备注 |",
         "|" + "---|" * (4 + len(refs) + len(protos) + 1)]
    for r in res["rows"]:
        ok_all = all(r["ok"].values())
        note = "该类真值样本数 <= 1" if r["tiny"] else ""
        L.append(f"| {r['split']} | {r['metric']} | " + " | ".join(f3(r['ref'][n]) for n in refs) +
                 f" | [{f3(r['lo'])}, {f3(r['hi'])}] | " + " | ".join(f"{f3(r['proto'][n])}{'' if r['ok'][n] else ' ✗'}" for n in protos) +
                 f" | {'是' if ok_all else '否'} | {note} |")
    L += ["", "## 2. 判定", "",
          f"- **字面读法（主判据）：{'通过' if res['literal']['pass'] else '未通过'}**"
          + ("" if res["literal"]["pass"] else "；不在范围内：" + "；".join(res["literal"]["failures"])),
          f"- 去掉真值样本数 <= 1 的类的召回：{'通过' if res['excluding_tiny_classes']['pass'] else '未通过'}"
          + ("" if res["excluding_tiny_classes"]["pass"] else "；" + "；".join(res["excluding_tiny_classes"]["failures"])),
          f"- 三个原型种子的均值在范围内（同样不含真值样本数 <= 1 的类）：{'通过' if res['mean_in_range']['pass'] else '未通过'}"
          + ("" if res["mean_in_range"]["pass"] else "；" + "；".join(res["mean_in_range"]["failures"])), ""]
    c = res.get("controls")
    if c:
        L += ["## 3. C++ 对照（同一条判据套在 C++ 自己的新种子上）", "",
              "| C++ 新种子 | 最佳轮 | 字面读法 | 去掉样本数 <= 1 的类 |", "|---|---|---|---|"]
        for n, r in c["runs"].items():
            lit, ex = c["literal"][n], c["excluding_tiny_classes"][n]
            L.append(f"| {n} | {r['best_epoch']} | {'通过' if lit['pass'] else '未通过：' + '；'.join(lit['failures'])} | "
                     f"{'通过' if ex['pass'] else '未通过：' + '；'.join(ex['failures'])} |")
        for key, label in (("baseline_literal", "字面读法"), ("baseline_excluding_tiny", "去掉样本数 <= 1 的类")):
            b = c.get(key)
            if b:
                L += ["", f"- 全部 C++ 种子里任取 3 个做参照、其余当候选（{label}）：逐个候选通过 {b['per_seed_pass']}/{b['per_seed_trials']}"
                      f"（{b['per_seed_pass'] / b['per_seed_trials']:.0%}）；任取 3 个候选同时通过 {b['per_triple_pass']}/{b['per_triple_trials']}"
                      f"（{b['per_triple_pass'] / b['per_triple_trials']:.0%}）。这两个比例是 C++ 线自己对自己的通过率（参照集与候选集有重叠抽样，非独立样本）。"]
        L.append("")
    return "\n".join(L) + "\n"


def parse_named(items):
    out = {}
    for it in items or []:
        name, _, path = it.partition("=")
        if not path:
            raise SystemExit(f"参数 {it!r} 应写成 名字=目录")
        out[name] = path
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", action="append", help="名字=C++ 参照 run 目录（至少 2 个，卡规定 3 个）")
    ap.add_argument("--proto", action="append", help="名字=原型 run 目录")
    ap.add_argument("--control", action="append", help="名字=额外的 C++ run 目录（可选）")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    refs_p, protos_p, ctrl_p = parse_named(a.ref), parse_named(a.proto), parse_named(a.control)
    if len(refs_p) < 2 or not protos_p:
        print("至少要 2 个 --ref 和 1 个 --proto", file=sys.stderr)
        return 2
    try:
        refs = {n: delivered(p) for n, p in refs_p.items()}
        protos = {n: delivered(p) for n, p in protos_p.items()}
        controls = {n: delivered(p) for n, p in ctrl_p.items()}
    except (OSError, ValueError, KeyError) as exc:
        print(f"读 run 失败：{exc}", file=sys.stderr)
        return 2
    res = evaluate(refs, protos, controls)
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
            f.write("\n")
    md = render_md(res)
    if a.out_md:
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 0 if res["literal"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
