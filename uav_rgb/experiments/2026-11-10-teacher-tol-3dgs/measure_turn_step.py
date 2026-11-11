#!/usr/bin/env python3
"""hold 1.0 下教师一个偏航转角步有多大（任务卡 W1-11 的「先量再定」）：只读旧实验每集的 actions.jsonl，不加载场景、不占 GPU。

量什么：每个执行了 YAW_LEFT / YAW_RIGHT 的步，保持窗口内机头沿转向方向实际转过的角度 = 转向号 × dyaw_deg（左 +1、右 −1）。
按「上一步执行的动作」分四类：after_forward（直飞后首转）、after_hover（悬停后首转，含每集第 1 步）、same_dir（同向连转）、
reversal（反向：上一步转了另一边）。同类不同上下文的转角步不一样（执行器有惯性），而极限环要看的是反向那一类：
误差落在 (tol, 步长 − tol) 时转一步过冲到对侧容差之外，再转回来；无环条件是 2 × tol >= 步长。
符号为负的步（转向与命令相反）不丢弃，单列计数。

用法：python measure_turn_step.py [--hold 1.0] [--out DIR] [标签=实验臂目录 ...]
      不给目录时量三份旧数据（室外教师 12°、室内教师 12°、室内教师 18° 诊断）。--out 给了就写 turn_step.json 与 turn_step.md。
容差公式（yaw_tol_deg）、名义转角步取自 sim/3dgs/planner.py（要 numpy；没有就只打印分布）。
"""
import glob
import importlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
D = os.environ.get("DATA_ROOT", "/root/rivermind-data") + "/uav_rgb"
DEFAULT_SOURCES = [
    ("室外教师 12°", D + "/exp-bearing-guard-3dgs-20261110/teacher"),
    ("室内教师 12°", D + "/exp-sage3d-pilot-20261110/teacher"),
    ("室内教师 18°（诊断）", D + "/exp-sage3d-pilot-20261110-diag-teachertol18/teacher"),
]
S, F, L, R = 0, 1, 2, 3                                   # SAFE_NEUTRAL / FORWARD_SLOW / YAW_LEFT / YAW_RIGHT
CONTEXTS = ("after_forward", "after_hover", "same_dir", "reversal")
CONTEXT_ZH = {"after_forward": "直飞后首转", "after_hover": "悬停后首转", "same_dir": "同向连转", "reversal": "反向"}


def percentile(sorted_vals, q):
    """线性插值分位（与 numpy.percentile 默认一致），q 取 0–100；空列表返回 None。"""
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def describe(values):
    v = sorted(values)
    if not v:
        return {"n": 0, "median": None, "p10": None, "p90": None, "mean": None, "min": None, "max": None}
    return {"n": len(v), "median": percentile(v, 50), "p10": percentile(v, 10), "p90": percentile(v, 90),
            "mean": sum(v) / len(v), "min": v[0], "max": v[-1]}


def turn_steps(rows):
    """rows = 一集的 actions.jsonl（按 step 排序）。返回 [(上下文, 转角步度数)]，每个执行了转向的步一项。"""
    out = []
    for i, r in enumerate(rows):
        ex = r["action_executed"]
        if ex not in (L, R):
            continue
        prev = rows[i - 1]["action_executed"] if i > 0 else S          # 每集第 1 步当作悬停后
        ctx = "after_forward" if prev == F else "after_hover" if prev == S else "same_dir" if prev == ex else "reversal"
        out.append((ctx, (1.0 if ex == L else -1.0) * r["dyaw_deg"]))
    return out


def collect(root):
    """root = <臂目录>：其下 <场景>/r<N>/ep_*/actions.jsonl。返回 ({上下文: [度数]}, 集数, {sim_dt})。"""
    by_ctx = {c: [] for c in CONTEXTS}
    dts, n_ep = set(), 0
    for p in sorted(glob.glob(os.path.join(root, "*", "r*", "ep_*", "actions.jsonl"))):
        with open(p, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        n_ep += 1
        dts.update(r.get("sim_dt") for r in rows)
        for ctx, v in turn_steps(rows):
            by_ctx[ctx].append(v)
    return by_ctx, n_ep, dts


def planner_constants(hold):
    """名义转角步、容差下限与系数、hold 对应的容差；planner 导不进来（没有 numpy）时返回 None。"""
    sys.path.insert(0, REPO)
    try:
        pl = importlib.import_module("sim.3dgs.planner")
    except ImportError:
        return None
    return {"yaw_rate_rps": pl.YAW_RATE_RPS, "nominal_deg": math.degrees(pl.YAW_RATE_RPS * hold),
            "tol_floor_deg": pl.YAW_TOL_DEG, "tol_step_frac": pl.TOL_STEP_FRAC, "tol_hold_deg": pl.yaw_tol_deg(hold)}


def fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def main(argv):
    hold, out_dir, sources, i = 1.0, None, [], 0
    while i < len(argv):
        if argv[i] == "--hold":
            hold, i = float(argv[i + 1]), i + 2
        elif argv[i] == "--out":
            out_dir, i = argv[i + 1], i + 2
        else:
            label, _, path = argv[i].partition("=")
            sources.append((label, path))
            i += 1
    sources = sources or DEFAULT_SOURCES
    res, raw = {}, {}
    for label, root in sources:
        by_ctx, n_ep, dts = collect(root)
        raw[label] = by_ctx
        allv = [v for c in CONTEXTS for v in by_ctx[c]]
        res[label] = {"root": root, "episodes": n_ep, "sim_dt": sorted(x for x in dts if x is not None),
                      "all": describe(allv), "non_positive": sum(v <= 0 for v in allv),
                      "by_context": {c: describe(by_ctx[c]) for c in CONTEXTS}}
    const = planner_constants(hold)
    lines = [f"# hold {hold:g} s 教师转角步实测（任务卡 W1-11）", "",
             "- 口径：每个执行了 YAW_LEFT / YAW_RIGHT 的步，转向号 × `dyaw_deg`（窗口内机头沿转向方向转过的度数）；"
             "按上一步执行的动作分四类；分位为线性插值；负值不丢弃（单列）。", ""]
    lines += ["## 1. 分布（度）", "", "| 数据 | 上下文 | n | 中位数 | p10 | p90 | 均值 | 最小 | 最大 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, r in res.items():
        rows = [("全部", r["all"])] + [(CONTEXT_ZH[c], r["by_context"][c]) for c in CONTEXTS]
        for name, d in rows:
            lines.append(f"| {label}（{r['episodes']} 集） | {name} | {d['n']} | {fmt(d['median'])} | {fmt(d['p10'])} | {fmt(d['p90'])} "
                         f"| {fmt(d['mean'])} | {fmt(d['min'])} | {fmt(d['max'])} |")
    lines += ["", "转向与命令相反（转角步 <= 0）的步数：" + "；".join(f"{k} {r['non_positive']}" for k, r in res.items()),
              "记录里的 sim_dt（保持窗口秒）：" + "；".join(f"{k} {r['sim_dt']}" for k, r in res.items()), ""]
    check = None
    if const:
        lines += ["## 2. 与容差的关系（按数据源分开算，不合并）", "",
                  f"- 名义转角步 = YAW_RATE_RPS × hold = {const['yaw_rate_rps']} rad/s × {hold:g} s = {const['nominal_deg']:.2f}°。",
                  f"- 无环条件：2 × tol >= 转角步。容差 = max({const['tol_floor_deg']:g}°, {const['tol_step_frac']:g} × 名义转角步)"
                  f" = {const['tol_hold_deg']:.2f}°（hold {hold:g} s）。",
                  "- 表里「> 2 × 容差」的步 = 误差落进 (tol, 步长 − tol) 时能维持极限环的转角步；占比 0 表示该容差下没有任何实测步能维持。", "",
                  "| 数据 | 容差 | 2 × 容差 | 全部步 > 2 × 容差 | 反向步 > 2 × 容差 | 最大转角步 / 2 | 按最大步算的最小系数 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        check = {"nominal_deg": const["nominal_deg"], "tol_hold_deg": const["tol_hold_deg"], "sources": {}}
        for label, by_ctx in raw.items():
            allv = sorted(v for c in CONTEXTS for v in by_ctx[c])
            rev = sorted(by_ctx["reversal"])
            if not allv:
                continue
            need = allv[-1] / 2
            check["sources"][label] = {"max_step_deg": allv[-1], "frac_needed": need / const["nominal_deg"], "rows": []}
            for tol in (const["tol_floor_deg"], const["tol_hold_deg"]):
                share = sum(v > 2 * tol for v in allv) / len(allv)
                share_rev = sum(v > 2 * tol for v in rev) / len(rev) if rev else None
                check["sources"][label]["rows"].append({"tol_deg": tol, "share_gt_2tol": share, "share_reversal_gt_2tol": share_rev})
                lines.append(f"| {label} | {tol:.2f}° | {2 * tol:.2f}° | {100 * share:.1f}% "
                             f"| {fmt(None if share_rev is None else 100 * share_rev)}% | {need:.2f}° | {need / const['nominal_deg']:.3f} |")
        same = [res[k]["by_context"]["same_dir"]["median"] for k in res if res[k]["by_context"]["same_dir"]["n"]]
        if same:
            lines += ["", "- 同向连转的实测中位数（各数据源）：" + "、".join(f"{v:.2f}°" for v in same)
                      + f"；名义转角步 {const['nominal_deg']:.2f}°。"]
    text = "\n".join(lines) + "\n"
    print(text)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "turn_step.md"), "w", encoding="utf-8") as f:
            f.write(text)
        with open(os.path.join(out_dir, "turn_step.json"), "w", encoding="utf-8") as f:
            json.dump({"hold": hold, "constants": const, "check": check, "sources": res}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
