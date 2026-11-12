#!/usr/bin/env python3
"""卡 E2E-P0 验收 (a) 的事后补充读法：把原型的三个种子放进 C++ 全部种子的分布里看。只描述，不改判，不是卡的判据。

背景：卡的判据用 3 个 C++ 参照种子的范围放宽 ±0.03（accept_offline.py 的字面读法）。守护进程把同一条判据套在额外的 C++ 种子上，
C++ 自己对自己的通过率很低（val 类 3 只有 33 个样本，一个样本 = 0.030，三个种子的范围只有两个样本宽）。
这里换一个不依赖「选哪 3 个种子当参照」的读法：
  对每个 split × {宏 F1，类 1..3 召回}（真值样本数 <= TINY 的类不报，与 accept_offline 的补充读法一致）：
    C++ 全部种子（参照种子 + 额外种子）的均值、样本标准差、最小、最大；
    原型每个种子的值，以及它是否落在 C++ 全部种子的 [最小, 最大] 内；
    z = (原型均值 - C++ 均值) / (C++ 标准差 / sqrt(原型种子数))：原型均值相对「C++ 取同样个数的种子时的均值」的偏离，以标准误为单位。
  这是事后选的读法（看过字面读法的结果之后写的），没有预注册的阈值；结论只能写成「落在 C++ 种子分布里」或「偏出」，由主会话判断。

输入：accept_offline.py 带 --control 时写的 JSON（含 refs、controls.runs、protos 三段，各 run 的 val/test 指标与 n_true）。
用法（只用标准库）：
  python proto/distribution_check.py --accept <accept_with_control.json> --out-json <json> --out-md <md>
退出码：0 正常；2 输入不全（C++ 种子少于 3 个、没有原型种子、缺字段）。
"""
import argparse
import json
import math
import statistics
import sys

SPLITS = ("val", "test")
METRICS = ("macro_f1", "recall1", "recall2", "recall3")   # 类 0 的召回在 val 无样本、test 只有 1 帧，不携带信息，不报
TINY = 1                                                  # 真值样本数 <= TINY 的类不报（与 accept_offline.TINY 同值）
MIN_CPP = 3                                               # 至少 3 个 C++ 种子才有标准差可报（参照 3 个，加 1 个以上额外种子更好）
EPS = 1e-9                                                # 比较 [最小, 最大] 边界时的浮点余量


def collect(acc):
    """从 accept JSON 取 (C++ 全部种子, 原型种子)；两者都是 {名字: {val: {...}, test: {...}}}。"""
    try:
        cpp = dict(acc["refs"])
        cpp.update(acc.get("controls", {}).get("runs", {}))
        protos = dict(acc["protos"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"accept JSON 缺 refs 或 protos：{exc!r}")
    if len(cpp) < MIN_CPP:
        raise ValueError(f"C++ 种子只有 {len(cpp)} 个，至少 {MIN_CPP} 个")
    if not protos:
        raise ValueError("没有原型种子")
    return cpp, protos


def skipped(metric, split, cpp):
    """召回指标对应的类在该 split 里真值样本数 <= TINY：不报。"""
    if not metric.startswith("recall"):
        return False
    k = int(metric[-1])
    return min(r[split]["n_true"][k] for r in cpp.values()) <= TINY


def analyse(cpp, protos):
    rows = []
    for sp in SPLITS:
        for m in METRICS:
            if skipped(m, sp, cpp):
                continue
            c = [r[sp][m] for r in cpp.values()]
            p = {n: r[sp][m] for n, r in protos.items()}
            lo, hi = min(c), max(c)
            mean_c, sd_c = statistics.fmean(c), statistics.stdev(c)
            mean_p = statistics.fmean(p.values())
            se = sd_c / math.sqrt(len(p))
            rows.append({"split": sp, "metric": m, "cpp_n": len(c), "cpp_mean": mean_c, "cpp_sd": sd_c, "cpp_min": lo, "cpp_max": hi,
                         "proto": p, "proto_mean": mean_p,
                         "in_range": {n: lo - EPS <= v <= hi + EPS for n, v in p.items()},
                         "z": (mean_p - mean_c) / se if se > 0 else None})
    cells = [ok for r in rows for ok in r["in_range"].values()]
    zs = [abs(r["z"]) for r in rows if r["z"] is not None]
    return {"cpp_seeds": list(cpp), "proto_seeds": list(protos), "rows": rows,
            "cells_in_range": sum(cells), "cells_total": len(cells),
            "max_abs_z": max(zs) if zs else None}


def f3(x):
    return f"{x:.3f}"


def render_md(res):
    p_names = res["proto_seeds"]
    L = ["# 验收 (a) 补充读法：原型种子在 C++ 全部种子分布里的位置（事后分析，不是卡的判据）", "",
         f"- C++ 种子 {len(res['cpp_seeds'])} 个：{', '.join(res['cpp_seeds'])}；原型种子：{', '.join(p_names)}。",
         "- 指标取交付口径（ckpt_best，val 取最佳轮，test 取训练末尾的评估），与 accept.md 同一份数。",
         "- z = (原型均值 - C++ 均值) / (C++ 标准差 / sqrt(原型种子数))。没有预注册的阈值，只描述。", "",
         "| split | 指标 | C++ 均值 ± 标准差 | C++ [最小, 最大] | " + " | ".join(f"原型 {n}" for n in p_names) + " | 原型均值 | z | 原型种子都在 [最小, 最大] 内 |",
         "|" + "---|" * (7 + len(p_names))]
    for r in res["rows"]:
        z = "无（C++ 标准差为 0）" if r["z"] is None else f"{r['z']:+.2f}"
        L.append(f"| {r['split']} | {r['metric']} | {f3(r['cpp_mean'])} ± {f3(r['cpp_sd'])} | [{f3(r['cpp_min'])}, {f3(r['cpp_max'])}] | "
                 + " | ".join(f"{f3(r['proto'][n])}{'' if r['in_range'][n] else ' ✗'}" for n in p_names)
                 + f" | {f3(r['proto_mean'])} | {z} | {'是' if all(r['in_range'].values()) else '否'} |")
    zmax = "无" if res["max_abs_z"] is None else f"{res['max_abs_z']:.2f}"
    L += ["", f"- 原型种子 × 指标共 {res['cells_total']} 格，落在 C++ 全部种子的 [最小, 最大] 内的有 {res['cells_in_range']} 格。",
          f"- |z| 的最大值 {zmax}。", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--accept", required=True, help="accept_offline.py 带 --control 写的 JSON")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    try:
        with open(a.accept, encoding="utf-8") as f:
            acc = json.load(f)
        cpp, protos = collect(acc)
        res = analyse(cpp, protos)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        print(f"输入不全：{exc}", file=sys.stderr)
        return 2
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
            f.write("\n")
    md = render_md(res)
    if a.out_md:
        with open(a.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
