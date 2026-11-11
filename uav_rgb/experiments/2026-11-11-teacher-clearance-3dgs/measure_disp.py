#!/usr/bin/env python3
"""任务卡 W1-12 第一部分第 1 步「量」：每步实际位移（机体系）的两个预测器 P1 / P2，在检验那一半路线上比预测误差。

只读 W1-11 验收输出里新教师臂的 240 集（`exp-teacher-tol-3dgs-20261110/{outdoor,indoor}/teacher-tol`），不起仿真、不占 GPU，
只要 numpy（`.venv-tools` 或 `3dgs-venv` 的 python 都行）。

口径：
  - 位移 = 每步 `pose_before` -> `pose_gt` 的 (Δx, Δy)，按 `yaw_before` 换到动作前的机体系（x 向前，y 向左）。
  - 上一步执行的动作类：S 悬停（含第 0 步）、F 前进、L 左转、R 右转（`action_executed`）。
  - P1「上下文表」：按（上一步类，本步类）分组取位移向量的中位数（分量各取中位数）；转向后的前进步按上一步的转向方向分 L / R 两组，
    位移偏向哪一侧由数据量出来。本步是转向的步也按转向方向分开（左右转的横向分量互为镜像，并在一起中位数没有意义）。
  - P2「速度外推」：动作前的水平速度取 `trace.jsonl` 里上一步的最后一个样本（世界系 vel，换到动作前机体系；第 0 步没有上一步，按 0 处理），
    对每个本步类做最小二乘 `位移 = A·v + b`（A 2×2，b 2，共 6 个系数）。
  - 拟合与检验：按路线下标 r<N> 的奇偶分两半，偶数 = 拟合，奇数 = 检验；预测误差 = |预测位移 − 实际位移|，只报检验那一半。
  - 选择：取检验半上「前进步」预测误差 p90 小的（只有前进步会被否决，预测只用在前进步上）；两者相差不到 0.01 m 取 P1。
    全部步的误差一并报，两个口径选出的预测器不同时在输出里标出来。
  - reserve = 前进后转弯步（上一步 F、本步 L 或 R）位移模 p90 + 所选预测器在检验半前进步上的预测误差 p90。
用法：
  python measure_disp.py --out <目录>          # 写 disp_table.json / disp_table.md，并在标准输出打印同一份 markdown
  python measure_disp.py --out <目录> --planner-snippet   # 另打印可贴进 planner.py 的常量段（数字取自本次 JSON，不手写）
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

SRC_DEFAULT = "/root/rivermind-data/uav_rgb/exp-teacher-tol-3dgs-20261110"
ARM_DEFAULT = "teacher-tol"
SUITES = ("outdoor", "indoor")
CLS = {0: "S", 1: "F", 2: "L", 3: "R"}
NAMES = ("S", "F", "L", "R")
ERR_PAIR_MIN_GAP_M = 0.01      # 两个预测器的 p90 差不到它取 P1（任务卡）
FIT_PARITY = 0                 # 路线下标 % 2 == 0 的集拟合，其余检验


# ------------------------------------------------------------------ 纯函数（有单测）
def to_body(dx, dy, yaw_deg):
    """世界系水平向量 -> 动作前机体系（x 向前，y 向左）。"""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (c * dx + s * dy, -s * dx + c * dy)


def quantile(vals, q):
    """线性插值分位（与 W1-11 的 step_types.py 的 pct 同式）；空返回 None。"""
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def episode_samples(rows, trace):
    """一集的逐步样本。rows = actions.jsonl 各行，trace = trace.jsonl 各行（None 则没有速度）。
    返回 [{i, prev, cur, expert, x, y, z, yaw, d（机体系位移）, norm, v（动作前机体系速度或 None）, collision}]。
    cur = 执行的动作类，expert = 教师给的动作类（教师单独跑时两者相同，反事实脚本会核对）；x y z yaw = 动作前的位姿（yaw 为度）。"""
    last_vel = {}
    for t in trace or []:
        last_vel[t["step"]] = t["vel"]
    out = []
    for i, r in enumerate(rows):
        p0, p1 = r["pose_before"], r["pose_gt"]
        yaw = float(r["yaw_before"])
        d = to_body(p1["x"] - p0["x"], p1["y"] - p0["y"], yaw)
        prev = CLS[rows[i - 1]["action_executed"]] if i else "S"
        if i == 0:
            vel = (0.0, 0.0)                       # 第 0 步：READY 悬停，没有上一步的轨迹样本
        else:
            lv = last_vel.get(i - 1)
            vel = None if lv is None else (lv[0], lv[1])
        out.append({"i": i, "prev": prev, "cur": CLS[r["action_executed"]], "expert": CLS[r["action_expert"]],
                    "x": float(p0["x"]), "y": float(p0["y"]), "z": float(p0["z"]), "yaw": yaw,
                    "d": d, "norm": math.hypot(*d),
                    "v": None if vel is None else to_body(vel[0], vel[1], yaw), "collision": bool(r.get("collision"))})
    return out


def fit_p1(samples):
    """{(prev, cur): {"dx", "dy", "n"}}（分量中位数）；另给每个本步类的整体中位数 {("*", cur): ...}，组缺样本时回退用。"""
    groups = {}
    for s in samples:
        groups.setdefault((s["prev"], s["cur"]), []).append(s["d"])
        groups.setdefault(("*", s["cur"]), []).append(s["d"])
    return {k: {"dx": float(np.median([d[0] for d in v])), "dy": float(np.median([d[1] for d in v])), "n": len(v)}
            for k, v in groups.items()}


def predict_p1(table, s):
    g = table.get((s["prev"], s["cur"])) or table.get(("*", s["cur"]))
    return (g["dx"], g["dy"]) if g else (0.0, 0.0)


def fit_p2(samples, min_n=10):
    """每个本步类一个 {"w": 3×2 系数（行 = [vx, vy, 1]，列 = [dx, dy]）, "n"}；没有速度的样本不参与；样本不足的类不拟合。"""
    out = {}
    for cur in NAMES:
        ss = [s for s in samples if s["cur"] == cur and s["v"] is not None]
        if len(ss) < min_n:
            continue
        x = np.array([[s["v"][0], s["v"][1], 1.0] for s in ss])
        y = np.array([s["d"] for s in ss])
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        out[cur] = {"w": w.tolist(), "n": len(ss)}
    return out


def predict_p2(model, s, p1_table):
    m = model.get(s["cur"])
    if m is None or s["v"] is None:
        return predict_p1(p1_table, s)             # 没拟合的类 / 没有速度：回退 P1（输出里记回退数）
    w = np.array(m["w"])
    p = np.array([s["v"][0], s["v"][1], 1.0]) @ w
    return (float(p[0]), float(p[1]))


def error_stats(errs):
    return {"n": len(errs), "median": quantile(errs, 0.5), "p90": quantile(errs, 0.9), "max": max(errs) if errs else None}


def choose(p1_fwd_p90, p2_fwd_p90):
    """前进步检验半 p90：P2 比 P1 小 ERR_PAIR_MIN_GAP_M 以上才选 P2。"""
    if p1_fwd_p90 is None or p2_fwd_p90 is None:
        return "P1"
    return "P2" if p2_fwd_p90 < p1_fwd_p90 - ERR_PAIR_MIN_GAP_M else "P1"


def split_fit_test(eps, parity=FIT_PARITY):
    fit = [e for e in eps if e["route"] % 2 == parity]
    test = [e for e in eps if e["route"] % 2 != parity]
    return fit, test


# ------------------------------------------------------------------ 读盘
def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def load_episodes(root, arm=ARM_DEFAULT, suites=SUITES):
    """每集一个 dict：suite / scene / route / episode / dir / metrics / samples。只读已写 done.json 的路线。"""
    eps = []
    for suite in suites:
        for mp in sorted(glob.glob(os.path.join(root, suite, arm, "*", "r*", "ep_*", "metrics.json"))):
            d = os.path.dirname(mp)
            if not os.path.exists(os.path.join(os.path.dirname(d), "done.json")):
                continue
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            tp = os.path.join(d, "trace.jsonl")
            eps.append({"suite": suite, "scene": m["scene"], "route": m["route"], "episode": m["episode"], "dir": d, "metrics": m,
                        "samples": episode_samples(jsonl(os.path.join(d, "actions.jsonl")), jsonl(tp) if os.path.exists(tp) else None)})
    return eps


# ------------------------------------------------------------------ 主流程
def measure(eps):
    fit_eps, test_eps = split_fit_test(eps)
    fit_s = [s for e in fit_eps for s in e["samples"]]
    test_s = [s for e in test_eps for s in e["samples"]]
    all_s = [s for e in eps for s in e["samples"]]
    t1 = fit_p1(fit_s)
    t1_all = fit_p1(all_s)
    m2 = fit_p2(fit_s)
    res = {"n_episodes": len(eps), "n_steps": len(all_s), "n_fit_episodes": len(fit_eps), "n_test_episodes": len(test_eps),
           "n_fit_steps": len(fit_s), "n_test_steps": len(test_s)}

    def errs(pred, ss):
        return [math.hypot(pred(s)[0] - s["d"][0], pred(s)[1] - s["d"][1]) for s in ss]

    p1 = lambda s: predict_p1(t1, s)                      # noqa: E731
    p2 = lambda s: predict_p2(m2, s, t1)                  # noqa: E731
    fwd_test = [s for s in test_s if s["cur"] == "F"]
    res["errors"] = {}
    for name, f in (("P1", p1), ("P2", p2)):
        res["errors"][name] = {"all": error_stats(errs(f, test_s)), "forward": error_stats(errs(f, fwd_test)),
                               "forward_by_prev": {pv: error_stats(errs(f, [s for s in fwd_test if s["prev"] == pv])) for pv in NAMES}}
    res["p2_fallback_test_forward_steps"] = sum(1 for s in fwd_test if "F" not in m2 or s["v"] is None)
    res["p1_fallback_test_steps"] = sum(1 for s in test_s if (s["prev"], s["cur"]) not in t1)
    res["selected_by_forward"] = choose(res["errors"]["P1"]["forward"]["p90"], res["errors"]["P2"]["forward"]["p90"])
    res["selected_by_all"] = choose(res["errors"]["P1"]["all"]["p90"], res["errors"]["P2"]["all"]["p90"])
    res["selected"] = res["selected_by_forward"]
    ft = [s["norm"] for s in all_s if s["prev"] == "F" and s["cur"] in ("L", "R")]
    ft_indoor = [s["norm"] for e in eps if e["suite"] == "indoor" for s in e["samples"]
                 if s["prev"] == "F" and s["cur"] in ("L", "R") and not s["collision"]]
    res["slip"] = {"F_to_T_all": {"n": len(ft), "median": quantile(ft, 0.5), "p90": quantile(ft, 0.9)},
                   "F_to_T_indoor_noncollision_w111_definition": {"n": len(ft_indoor), "median": quantile(ft_indoor, 0.5),
                                                                  "p90": quantile(ft_indoor, 0.9)}}
    err_p90 = res["errors"][res["selected"]]["forward"]["p90"]
    res["reserve_m"] = res["slip"]["F_to_T_all"]["p90"] + err_p90
    res["p1_table_fit"] = {f"{k[0]}>{k[1]}": v for k, v in sorted(t1.items())}
    res["p1_table_all"] = {f"{k[0]}>{k[1]}": v for k, v in sorted(t1_all.items())}
    res["p2_model_fit"] = m2
    res["p2_model_all"] = fit_p2(all_s)       # 上线用：验证（选预测器、算 reserve）用拟合半 / 检验半，系数用全部 240 集重拟合一次
    # 同一张拟合半的表在两个场景类上的差别：只作旁证，看动力学表是否随场景类变
    by_suite = {}
    for suite in SUITES:
        ss = [s for e in eps if e["suite"] == suite for s in e["samples"]]
        by_suite[suite] = {f"{k[0]}>{k[1]}": v for k, v in sorted(fit_p1(ss).items()) if k[1] == "F"}
    res["p1_forward_by_suite_all_data"] = by_suite
    return res


def render(res, src):
    f3 = lambda v: "—" if v is None else f"{v:.3f}"                       # noqa: E731
    L = ["# 任务卡 W1-12 第 1 步：每步位移的预测器（量）", "",
         f"- 来源 `{src}`（新教师臂），{res['n_episodes']} 集 {res['n_steps']} 步；拟合 = 路线下标偶数 {res['n_fit_episodes']} 集 {res['n_fit_steps']} 步，"
         f"检验 = 奇数 {res['n_test_episodes']} 集 {res['n_test_steps']} 步。",
         "- 位移 = `pose_before` -> `pose_gt`，换到动作前机体系（x 向前，y 向左，单位 m）；上一步类 S 悬停（含第 0 步）、F 前进、L 左转、R 右转。", "",
         "## 1. P1 上下文表（拟合半，本步 = 前进 F）", "", "| 上一步 | n | dx 中位数 | dy 中位数 | 位移模（|(dx,dy)|）| 偏向 |", "| --- | --- | --- | --- | --- | --- |"]
    for pv in NAMES:
        g = res["p1_table_fit"].get(f"{pv}>F")
        if g:
            side = "左（+y）" if g["dy"] > 0.005 else "右（−y）" if g["dy"] < -0.005 else "居中"
            L.append(f"| {pv} | {g['n']} | {g['dx']:.4f} | {g['dy']:.4f} | {math.hypot(g['dx'], g['dy']):.4f} | {side} |")
    L += ["", "## 2. 检验半上的预测误差（预测位移与实际位移之差的模，m）", "",
          "| 预测器 | 范围 | n | 中位数 | p90 | 最大 |", "| --- | --- | --- | --- | --- | --- |"]
    for name in ("P1", "P2"):
        for scope, key in (("前进步", "forward"), ("全部步", "all")):
            e = res["errors"][name][key]
            L.append(f"| {name} | {scope} | {e['n']} | {f3(e['median'])} | {f3(e['p90'])} | {f3(e['max'])} |")
    L += ["", "前进步按上一步类分（检验半）：", "", "| 预测器 | " + " | ".join(f"上一步 {pv}（n / 中位 / p90）" for pv in NAMES) + " |",
          "| --- | " + " | ".join("---" for _ in NAMES) + " |"]
    for name in ("P1", "P2"):
        cells = []
        for pv in NAMES:
            e = res["errors"][name]["forward_by_prev"][pv]
            cells.append(f"{e['n']} / {f3(e['median'])} / {f3(e['p90'])}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", f"P2 在检验半前进步上回退 P1 的步数：{res['p2_fallback_test_forward_steps']}；P1 在检验半回退到本步类整体中位数的步数：{res['p1_fallback_test_steps']}。", "",
          "## 3. 选择", "",
          f"- 前进步检验半 p90：P1 = {f3(res['errors']['P1']['forward']['p90'])} m，P2 = {f3(res['errors']['P2']['forward']['p90'])} m，"
          f"相差不到 {ERR_PAIR_MIN_GAP_M} m 取 P1。**选 {res['selected']}**（按前进步）。",
          f"- 按全部步 p90 选出的是 {res['selected_by_all']}" + ("（与按前进步选的一致）" if res["selected_by_all"] == res["selected_by_forward"] else "（**与按前进步选的不同**）") + "。", "",
          "## 4. 前进后转弯步的前滑与 reserve", "",
          f"- 前进后的转弯步位移模（全部 {res['slip']['F_to_T_all']['n']} 步）：中位 {f3(res['slip']['F_to_T_all']['median'])} m，p90 {f3(res['slip']['F_to_T_all']['p90'])} m。",
          f"- 同口径取 W1-11 的定义（只算室内、未碰撞的步，{res['slip']['F_to_T_indoor_noncollision_w111_definition']['n']} 步）：中位 "
          f"{f3(res['slip']['F_to_T_indoor_noncollision_w111_definition']['median'])} m，p90 {f3(res['slip']['F_to_T_indoor_noncollision_w111_definition']['p90'])} m（W1-11 报告 0.066 / 0.134）。",
          f"- reserve = {f3(res['slip']['F_to_T_all']['p90'])} + {f3(res['errors'][res['selected']]['forward']['p90'])}（{res['selected']} 前进步检验半 p90）= **{res['reserve_m']:.4f} m**。", "",
          "## 5. 前进步位移表按场景类拆开（全部数据，只作旁证）", "", "| 场景类 | 上一步 | n | dx | dy |", "| --- | --- | --- | --- | --- |"]
    for suite in SUITES:
        for pv in NAMES:
            g = res["p1_forward_by_suite_all_data"][suite].get(f"{pv}>F")
            if g:
                L.append(f"| {suite} | {pv} | {g['n']} | {g['dx']:.4f} | {g['dy']:.4f} |")
    return L


def planner_snippet(res):
    """贴进 planner.py 的常量段。数字取自本次 JSON：按所选预测器给；P2 = 前进类的 d = A·v + b（系数取全部 240 集的重拟合）。"""
    lines = []
    if res["selected"] == "P2":
        m = res["p2_model_all"]["F"]
        w = m["w"]          # 3×2：行 [vx, vy, 1]，列 [dx, dy]
        lines += [f"# 前进步位移预测（机体系；动作前机体系速度 v = (vx, vy) -> 位移 d）：d = A·v + b；全部 240 集重拟合（n={m['n']}）",
                  f"SWEPT_PRED_A = (({w[0][0]:.4f}, {w[1][0]:.4f}), ({w[0][1]:.4f}, {w[1][1]:.4f}))   # 行 = d 的 (x, y)，列 = v 的 (x, y)",
                  f"SWEPT_PRED_B = ({w[2][0]:.4f}, {w[2][1]:.4f})"]
    else:
        code = {"S": "SAFE_NEUTRAL", "F": "FORWARD_SLOW", "L": "YAW_LEFT", "R": "YAW_RIGHT"}
        lines.append("SWEPT_DISP_TABLE = {   # 上一步执行的动作 -> 本步前进的预测位移（机体系 dx 向前、dy 向左，米）")
        for pv in NAMES:
            g = res["p1_table_fit"][f"{pv}>F"]
            lines.append(f"    {code[pv]}: ({g['dx']:.4f}, {g['dy']:.4f}),   # n={g['n']}")
        lines.append("}")
    lines.append(f"SWEPT_RESERVE_M = {res['reserve_m']:.4f}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=SRC_DEFAULT, help="W1-11 验收输出目录（只读）")
    ap.add_argument("--arm", default=ARM_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--planner-snippet", action="store_true")
    args = ap.parse_args(argv)
    eps = load_episodes(args.src, args.arm)
    if not eps:
        print(f"没读到集：{args.src}/<suite>/{args.arm}", file=sys.stderr)
        return 2
    res = measure(eps)
    res["source"] = {"dir": args.src, "arm": args.arm}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "disp_table.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    lines = render(res, args.src)
    with open(os.path.join(args.out, "disp_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.planner_snippet:
        print("\n".join(["", "```python"] + planner_snippet(res) + ["```"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
