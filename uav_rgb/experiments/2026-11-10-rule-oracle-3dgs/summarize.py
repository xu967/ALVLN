#!/usr/bin/env python3
"""rule-oracle-3dgs 汇总（任务卡 K1）：臂 × 场景 × 路线长度桶的闭环指标、预注册判决、三个主比较的配对检验、失败集分类、
规则条目触发占比、depth 与 occ 列距离一致性、与旧实验同名臂的逐集对照 -> <OUT>/summary.md、summary.json。

只读 <OUT>/{indoor,outdoor}/<臂>/<场景>/r<N>/{done.json, ep_*/{metrics.json, actions.jsonl, rule.jsonl}}（driver 每集落的）与
各类目录下的 config.json，不加载场景，随时可跑；没跑完时只统计已写 done.json 的路线（半条路线不进表）。
口径（METRIC_VERSION）：
  SR(stop)  首次进入 waypoint 半径早于首次碰撞；分母 = 已写 done.json 的路线数 × K（没出 metrics 的失败集算失败，单列）。
  碰撞率    整集任一步碰撞（含到点之后，与 11-09 / 两个 11-10 旧实验同口径；预注册的碰撞率用它）；撞在前 = 没到点就撞了。
            规则臂 / 盲前进到点后不停，会在 waypoint 附近继续动，所以碰撞率比撞在前高，两个都列。
  超时      既没到点也没撞。左右交替 = 整集 action_executed 里有 >= MIN_ALT_RUN 步严格 YAW_LEFT / YAW_RIGHT 交替的连续段
            （与 11-10 教师超时分析 teacher_osc.py 同定义）。
  失败集分类 = 没到点早于撞的集：撞在前 / 超时·含交替 / 超时·无交替 / 其他（既不是撞在前也不是超时，正常应为 0）。
  配对检验  同（场景, 路线, seed）两臂都有 metrics.json 的集，精确二项 McNemar（任何样本量都用精确，不切卡方）。
  depth 与 occ  规则臂 rule.jsonl 每步同时存了两种来源的 20 列距离（同一位姿），按列取 |d_depth - d_occ|。
用法：python summarize.py <OUT> [--old-indoor <旧室内实验目录>] [--old-outdoor <旧室外实验目录>]
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "..", "tools"))
from pipeline_common import wilson_ci  # noqa: E402  只用标准库的共用 Wilson 公式，3dgs venv 里也能 import

METRIC_VERSION = "3dgs.pilot.v2+stop.v1"
CAP_M = 4.0                 # rule_oracle 列距离上限（任务卡：上限 4.0 m）；「有障碍的列」= 两种来源里至少一个低于它
MISS_M = 0.3                # 任务卡：depth 比 occ 远 0.3 m 以上的列算漏检
MIN_ALT_RUN = 6             # 左右交替的最短连续步数，与 teacher_osc.py 一致
YAW_LEFT, YAW_RIGHT = 2, 3  # action_executed 的编码
MAIN_ARM, REF_ARM = "rule-occ+route", "forward+route"
# 预注册的数字（任务卡 K1 原文：主臂室内到点成功不低于 0.80，10 m 以上的路线不低于 0.60，碰撞率不高于 0.15；
# 室外不低于 0.84，碰撞率不高于 0.10；证伪：主臂室内低于 0.70）
PREREG = {"indoor": {"sr_min": 0.80, "sr_long_min": 0.60, "coll_max": 0.15, "falsify_lt": 0.70},
          "outdoor": {"sr_min": 0.84, "coll_max": 0.10}}
LONG_BUCKET = ">=10 m"      # routes_selected.json 里 ≥10 m 桶的标签
MAIN_PAIRS = [("M1", "rule-occ+route", "forward+route"), ("M2", "rule-depth+route", "rule-occ+route"),
              ("M3", "rule-occ+goal", "rule-occ+route")]
EXTRA_PAIRS = [("x1", "rule-depth+route", "forward+route"), ("x2", "rule-occ+goal", "forward+route"),
               ("x3", "rule-occ+route@0.5", "rule-occ+route"), ("x4", "forward+route@0.5", "forward+route"),
               ("x5", "rule-occ+route@0.5", "forward+route@0.5")]
CLASS_TITLE = {"indoor": "室内（SAGE-3D，试点同 40 条路线）", "outdoor": "室外（3DGS 四场景）"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def percentile(sorted_xs, q):
    """线性插值分位数（与 numpy 默认一致）；空列表返回 None。"""
    if not sorted_xs:
        return None
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def routes_of(cfg, scene):
    r = cfg["routes"]
    return r.get(scene, []) if isinstance(r, dict) else r


def load_class(cdir):
    """读一个类（indoor / outdoor）的目录。返回 dict：cfg、arms（config 顺序）、rows（metrics.json 字典，加 _dir）、
    routes_done[(臂, 场景)]、done_keys{(臂, 场景, 路线)}、failed（done.json 里的失败集）、route_info{(场景, 路线): routes_selected 条目}。"""
    cfg = read_json(os.path.join(cdir, "config.json"))
    if cfg is None:
        return None
    rows, routes_done, done_keys, failed = [], {}, set(), []
    for arm in cfg["arms"]:
        g = arm["name"]
        for s in cfg["scenes"]:
            for r in routes_of(cfg, s):
                d = os.path.join(cdir, g, s, f"r{r}")
                done = read_json(os.path.join(d, "done.json"))
                if done is None:
                    continue
                routes_done[(g, s)] = routes_done.get((g, s), 0) + 1
                done_keys.add((g, s, r))
                failed += [{"arm": g, "scene": s, "route": r, **f} for f in done.get("failures", [])]
                for p in sorted(glob.glob(os.path.join(d, "ep_*", "metrics.json"))):
                    m = read_json(p)
                    if m is not None:
                        m["_dir"] = os.path.dirname(p)
                        rows.append(m)
    rs_path = cfg.get("routes_selected")
    rs = read_json(rs_path) if rs_path else None
    if rs is None:
        rs = read_json(os.path.join(cdir, "routes_selected.json"))
    route_info = {(s, x["route"]): x for s, xs in rs["scenes"].items() for x in xs} if rs else {}
    labels = (rs or {}).get("params", {}).get("bucket_labels") or sorted({x["bucket"] for x in route_info.values()})
    return {"cfg": cfg, "arms": [a["name"] for a in cfg["arms"]], "arm_cfg": {a["name"]: a for a in cfg["arms"]}, "rows": rows,
            "routes_done": routes_done, "done_keys": done_keys, "failed": failed, "route_info": route_info,
            "bucket_labels": labels, "dir": cdir}


def agg(sel, n_req):
    """sel = 某臂（某子集）的 metrics 字典；n_req = 该子集的请求集数。"""
    n = len(sel)
    ks = sum(bool(x["success_stop"]) for x in sel)
    spl = [x["spl_stop"] for x in sel if x.get("spl_stop") is not None]
    return {"n_req": n_req, "n": n, "failed": n_req - n, "k_stop": ks,
            "sr_stop": ks / n_req if n_req else None, "ci_stop": wilson_ci(ks, n_req),
            "k_v2": sum(bool(x["success_v2"]) for x in sel),
            "k_coll": sum(bool(x["collision"]) for x in sel), "collision": sum(bool(x["collision"]) for x in sel) / n_req if n_req else None,
            "collide_first": sum(bool(x["collide_first"]) for x in sel) / n_req if n_req else None,
            "timeout": sum(bool(x["timeout"]) for x in sel) / n_req if n_req else None,
            "spl_stop": sum(spl) / n_req if n_req else None,
            "steps_mean": sum(x["steps"] for x in sel) / n if n else None}


def alt_runs(seq, min_run=MIN_ALT_RUN):
    """seq = action_executed 序列；返回严格 YAW_LEFT / YAW_RIGHT 交替、长度 >= min_run 的 (起, 止) 下标段（止含）。"""
    runs, i = [], 0
    while i < len(seq):
        if seq[i] not in (YAW_LEFT, YAW_RIGHT):
            i += 1
            continue
        j = i
        while j + 1 < len(seq) and seq[j + 1] in (YAW_LEFT, YAW_RIGHT) and seq[j + 1] != seq[j]:
            j += 1
        if j - i + 1 >= min_run:
            runs.append((i, j))
        i = j + 1
    return runs


def has_alternation(ep_dir):
    try:
        with open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") as f:
            seq = [json.loads(line)["action_executed"] for line in f if line.strip()]
    except (OSError, ValueError, KeyError):
        return None                         # 读不到动作序列：不猜，分类时单列
    return bool(alt_runs(seq))


def classify(m):
    """没到点早于撞的集（success_stop 假）归类；成功集返回 None。"""
    if m["success_stop"]:
        return None
    if m["collide_first"]:
        return "撞在前"
    if m["timeout"]:
        alt = has_alternation(m["_dir"])
        return "超时·无动作序列" if alt is None else "超时·含交替" if alt else "超时·无交替"
    return "其他"


FAIL_CLASSES = ["撞在前", "超时·含交替", "超时·无交替", "超时·无动作序列", "其他"]


def fail_counts(sel):
    out = {c: 0 for c in FAIL_CLASSES}
    for m in sel:
        c = classify(m)
        if c:
            out[c] += 1
    return out


def entry_shares(sel):
    """规则条目触发占比：sum(各集该条目步数) / sum(各集步数)，分「到点或撞之前」与「整集」。"""
    whole, pre, n_w, n_p = {}, {}, 0, 0
    for m in sel:
        r = m.get("rule")
        if not r:
            continue
        n_w, n_p = n_w + r["n_steps"], n_p + r["n_steps_pre"]
        for k, v in r["entries"].items():
            whole[k] = whole.get(k, 0) + v
        for k, v in r["entries_pre"].items():
            pre[k] = pre.get(k, 0) + v
    return {"n_steps": n_w, "n_steps_pre": n_p, "whole": whole, "pre": pre}


def column_stats(pairs):
    """pairs = [(d_occ, d_depth)]（单列的两个距离）；返回 n、|差| 中位数 / 90 分位、depth 比 occ 远 MISS_M 以上的占比（漏检）、反向占比。"""
    n = len(pairs)
    if not n:
        return {"n": 0, "median": None, "p90": None, "depth_farther": None, "occ_farther": None}
    diffs = sorted(abs(b - a) for a, b in pairs)
    return {"n": n, "median": percentile(diffs, 0.5), "p90": percentile(diffs, 0.9),
            "depth_farther": sum(b - a >= MISS_M for a, b in pairs) / n, "occ_farther": sum(a - b >= MISS_M for a, b in pairs) / n}


def depth_occ(sel):
    """sel = 规则臂的 metrics 字典；读各集 rule.jsonl，返回 {"all": 全部步的列对, "pre": 到点或撞之前的列对}。"""
    out = {"all": [], "pre": []}
    for m in sel:
        p = os.path.join(m["_dir"], "rule.jsonl")
        if not os.path.exists(p):
            continue
        end = min([x for x in (m["reach_step"], m["collision_step"]) if x is not None] or [m["steps"] - 1])
        with open(p, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if not t.get("d_occ") or not t.get("d_depth"):
                    continue                    # shadow 关掉时只有一种来源，没有可比的列
                cols = list(zip(t["d_occ"], t["d_depth"]))
                out["all"] += cols
                if t["step"] <= end:
                    out["pre"] += cols
    return out


def obstacle_cols(pairs):
    return [(a, b) for a, b in pairs if min(a, b) < CAP_M - 1e-9]


def depth_occ_summary(v):
    """summary.json 用：{范围: {列集: column_stats}}；不落原始列对（几十万个）。"""
    return {sc: {"all_cols": column_stats(v[sc]), "obstacle_cols": column_stats(obstacle_cols(v[sc]))} for sc in ("all", "pre")}


def paired(index, a, b):
    keys = [k[1:] for k in index if k[0] == a and (b, *k[1:]) in index]
    bb = sum(bool(index[(a, *k)]["success_stop"]) and not index[(b, *k)]["success_stop"] for k in keys)
    cc = sum(not index[(a, *k)]["success_stop"] and bool(index[(b, *k)]["success_stop"]) for k in keys)
    n = len(keys)
    delta = (sum(bool(index[(a, *k)]["success_stop"]) for k in keys) - sum(bool(index[(b, *k)]["success_stop"]) for k in keys)) / n if n else None
    return {"a": a, "b": b, "pairs": n, "b_count": bb, "c_count": cc, "p": exact_mcnemar(bb, cc), "delta_sr": delta}


def load_old(root):
    """旧实验目录：glob 所有 <臂>/<场景>/r*/ep_*/metrics.json，不依赖旧 config；目录不存在返回空。"""
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "r*", "ep_*", "metrics.json"))) if root else []:
        m = read_json(p)
        if m is not None:
            rows.append(m)
    return rows


def old_comparison(new_rows, old_rows):
    """新跑的 forward+route 与旧实验同名臂逐集对照（键 = 场景, 路线, seed），再给旧实验其余臂在同一批集上的成绩作参照。"""
    new = {(m["scene"], m["route"], m["seed"]): m for m in new_rows if m["arm"] == REF_ARM}
    old_arm = {}
    for m in old_rows:
        old_arm.setdefault(m["arm"], {})[(m["scene"], m["route"], m["seed"])] = m
    out = {"keys_new": len(new), "rerun": None, "reference": {}}
    same = old_arm.get(REF_ARM, {})
    ks = sorted(set(new) & set(same))
    if ks:
        def near(p, q):
            return all(abs(p[f] - q[f]) < 1e-6 for f in ("x", "y", "yaw_deg"))
        out["rerun"] = {"matched": len(ks),
                        "success_equal": sum(bool(new[k]["success_stop"]) == bool(same[k]["success_stop"]) for k in ks),
                        "collision_equal": sum(bool(new[k]["collision"]) == bool(same[k]["collision"]) for k in ks),
                        "trajectory_equal": sum((new[k]["steps"], new[k]["reach_step"], new[k]["collision_step"]) ==
                                                (same[k]["steps"], same[k]["reach_step"], same[k]["collision_step"]) for k in ks),
                        "start_pose_equal": sum(near(new[k]["start_pose"], same[k]["start_pose"]) for k in ks),
                        "k_new": sum(bool(new[k]["success_stop"]) for k in ks), "k_old": sum(bool(same[k]["success_stop"]) for k in ks)}
    for g, d in sorted(old_arm.items()):
        sel = [d[k] for k in sorted(set(new) & set(d))]
        out["reference"][g] = agg(sel, len(sel))
    return out


def summarize_class(cls, data, old_rows):
    cfg, arms, rows = data["cfg"], data["arms"], data["rows"]
    K = cfg["k"]
    by_arm = {g: [x for x in rows if x["arm"] == g] for g in arms}

    def n_req(g, scenes=None, routes=None):
        keys = [(s, r) for s in (scenes or cfg["scenes"]) for r in routes_of(cfg, s) if (g, s, r) in data["done_keys"]]
        return K * len([k for k in keys if routes is None or k in routes])

    res = {"class": cls, "config": cfg, "overall": {}, "per_scene": {}, "buckets": {}, "fail": {}, "entries": {}, "depth_occ": {},
           "pairs": [], "route_cells": {}}
    for g in arms:
        res["overall"][g] = agg(by_arm[g], n_req(g))
        for s in cfg["scenes"]:
            res["per_scene"][f"{g}|{s}"] = agg([x for x in by_arm[g] if x["scene"] == s], n_req(g, [s]))
        res["fail"][g] = {"all": fail_counts(by_arm[g])}
        if data["arm_cfg"][g]["base"].startswith("rule-"):
            res["entries"][g] = entry_shares(by_arm[g])
            res["depth_occ"][g] = depth_occ(by_arm[g])
        for s in cfg["scenes"]:
            for r in routes_of(cfg, s):
                if (g, s, r) in data["done_keys"]:
                    res["route_cells"][f"{g}|{s}|{r}"] = sum(bool(x["success_stop"]) for x in by_arm[g] if x["scene"] == s and x["route"] == r)
        for b in data["bucket_labels"] if data["route_info"] else []:
            keys = {k for k, x in data["route_info"].items() if x["bucket"] == b and (g, *k) in data["done_keys"]}
            sel = [x for x in by_arm[g] if (x["scene"], x["route"]) in keys]
            res["buckets"][f"{b}|{g}"] = agg(sel, K * len(keys))
            res["fail"][g][b] = fail_counts(sel)
    index = {(x["arm"], x["scene"], x["route"], x["seed"]): x for x in rows}
    res["pairs"] = [{"id": i, "main": i.startswith("M"), **paired(index, a, b)} for i, a, b in MAIN_PAIRS + EXTRA_PAIRS
                    if a in arms and b in arms]
    worst = 0.0
    for k in {k[1:] for k in index}:
        poses = [index[(g, *k)]["start_pose"] for g in arms if (g, *k) in index]
        for q in poses[1:]:
            worst = max(worst, abs(q["x"] - poses[0]["x"]), abs(q["y"] - poses[0]["y"]), abs(q["yaw_deg"] - poses[0]["yaw_deg"]) / 100.0)
    res["spawn_pair_max_diff"] = worst
    res["old"] = old_comparison(rows, old_rows) if REF_ARM in arms else None
    res["failed"] = data["failed"]
    res["prereg"] = prereg(cls, res, data)
    return res


def prereg(cls, res, data):
    """预注册判决（机械套用数字；点估计过线与否 + Wilson 区间是否跨线都给出）。主臂不在这个类里或没数据时返回空。"""
    t, a = PREREG[cls], res["overall"].get(MAIN_ARM)
    if not a or not a["n_req"]:
        return []
    out = []

    def row(name, k, n, thr, kind):
        ci = wilson_ci(k, n)
        v = k / n if n else None
        if kind == "ge":
            ok = v is not None and v >= thr
        elif kind == "le":
            ok = v is not None and v <= thr
        else:                                   # "lt"：证伪条件，成立 = 低于阈值
            ok = v is not None and v < thr
        straddle = ci is not None and ci[0] < thr < ci[1]
        out.append({"item": name, "k": k, "n": n, "value": v, "ci": ci, "threshold": thr, "kind": kind, "met": ok, "ci_straddles": straddle})

    row("到点成功 SR(stop)", a["k_stop"], a["n_req"], t["sr_min"], "ge")
    if cls == "indoor":
        b = res["buckets"].get(f"{LONG_BUCKET}|{MAIN_ARM}")
        if b is not None:
            row(f"{LONG_BUCKET} 路线的到点成功", b["k_stop"], b["n_req"], t["sr_long_min"], "ge")
    row("碰撞率（整集任一步碰撞）", a["k_coll"], a["n_req"], t["coll_max"], "le")
    if cls == "indoor":
        row("证伪条件：SR(stop) 低于 0.70", a["k_stop"], a["n_req"], t["falsify_lt"], "lt")
    return out


# ---------- 渲染 ----------
def f2(v):
    return "—" if v is None else f"{v:.2f}"


def ci_cell(a):
    if not a["n_req"]:
        return "—"
    ci = a["ci_stop"]
    return f"{a['k_stop']}/{a['n_req']} = {a['sr_stop']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def render_class(cls, res, data):
    cfg, arms, K = res["config"], data["arms"], res["config"]["k"]
    L = [f"## {CLASS_TITLE[cls]}", "",
         f"- hold {cfg['hold']} s（`@0.5` 臂另计）、每路线 {K} 集、出生扰动 ±{cfg['jitter']} m / ±10°、seed = {cfg['seed0']} + 100·路线 + k"
         f"（同 seed 各臂出生位姿相同）；护栏阈值 {cfg['guard_threshold_deg']}°，路线前视 {cfg['route_lookahead_m']} m；"
         f"规则参数 {json.dumps(cfg.get('rule_params'), ensure_ascii=False)}（冻结值，开发路线上选定）。", ""]
    if res["prereg"]:
        L += ["### 预注册判决（主臂 `rule-occ+route`；数字来自任务卡，起跑前写进 run.sh 头）", "",
              "| 项 | 实测 [Wilson 95%] | 阈值 | 判决 | 区间是否跨阈值 |", "| --- | --- | --- | --- | --- |"]
        for p in res["prereg"]:
            ci = p["ci"]
            cell = f"{p['k']}/{p['n']} = {p['value']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if p["n"] else "—"
            sym = {"ge": "≥", "le": "≤", "lt": "<"}[p["kind"]]
            verdict = ("证伪成立" if p["met"] else "未证伪") if p["kind"] == "lt" else ("达成" if p["met"] else "未达成")
            L.append(f"| {p['item']} | {cell} | {sym} {p['threshold']:.2f} | {verdict} | {'是' if p['ci_straddles'] else '否'} |")
        L.append("")
    L += ["### 合计", "",
          "| 臂 | SR(stop) [Wilson 95%] | SR(v2) | 碰撞率 | 撞在前 | 超时率 | SPL(stop) | 平均步数 | n（失败集） |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in arms:
        a = res["overall"][g]
        steps = "—" if a["steps_mean"] is None else f"{a['steps_mean']:.0f}"
        L.append(f"| `{g}` | {ci_cell(a)} | {a['k_v2']}/{a['n_req']} | {f2(a['collision'])} | {f2(a['collide_first'])} | {f2(a['timeout'])} "
                 f"| {f2(a['spl_stop'])} | {steps} | {a['n_req']}（{a['failed']}） |")
    L += ["", f"### 分场景 SR(stop)（括号内是碰撞率；每格 n = 已完成路线 × {K}）", "",
          "| 臂 | " + " | ".join(cfg["scenes"]) + " |", "| --- |" + " --- |" * len(cfg["scenes"])]
    for g in arms:
        cells = []
        for s in cfg["scenes"]:
            a = res["per_scene"][f"{g}|{s}"]
            cells.append(f"{a['k_stop']}/{a['n_req']}（{a['collision']:.2f}）" if a["n_req"] else "—")
        L.append(f"| `{g}` | " + " | ".join(cells) + " |")
    if data["route_info"]:
        L += ["", "### 按路线长度分桶（桶 = A* 长度，来自 routes_selected.json；每格 = 到点 / 请求，括号内是碰撞率）", "",
              "| 桶（计划路线数） | " + " | ".join(f"`{g}`" for g in arms) + " |", "| --- |" + " --- |" * len(arms)]
        for b in data["bucket_labels"]:
            n_plan = sum(1 for x in data["route_info"].values() if x["bucket"] == b)
            cells = []
            for g in arms:
                a = res["buckets"][f"{b}|{g}"]
                cells.append(f"{a['k_stop']}/{a['n_req']} = {a['sr_stop']:.2f}（{a['collision']:.2f}）" if a["n_req"] else "—")
            L.append(f"| {b}（{n_plan}） | " + " | ".join(cells) + " |")
    L += ["", "### 失败集分类（没到点早于撞的集；整集 action_executed 里有 ≥6 步严格左右交替记「含交替」）", "",
          "| 臂 | 桶 | 失败集 | " + " | ".join(FAIL_CLASSES) + " |", "| --- | --- | --- |" + " --- |" * len(FAIL_CLASSES)]
    for g in arms:
        for b in ["全部"] + (data["bucket_labels"] if data["route_info"] else []):
            c = res["fail"][g]["all" if b == "全部" else b]
            L.append(f"| `{g}` | {b} | {sum(c.values())} | " + " | ".join(str(c[x]) for x in FAIL_CLASSES) + " |")
    L += ["", "### 逐集配对（SR(stop)，同场景同路线同 seed，精确 McNemar；b = A 成 B 败，c = A 败 B 成；Δ = SR(A) − SR(B)）", "",
          "| 编号 | A vs B | 配对数 | b | c | Δ | p | 预注册 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in res["pairs"]:
        d = "—" if t["delta_sr"] is None else f"{t['delta_sr']:+.3f}"
        reg = "主比较" if t["main"] and cls == "indoor" else "补充（未预注册）"
        L.append(f"| {t['id']} | `{t['a']}` vs `{t['b']}` | {t['pairs']} | {t['b_count']} | {t['c_count']} | {d} | {t['p']:.4f} | {reg} |")
    if res["entries"]:
        names = sorted({k for e in res["entries"].values() for k in list(e["whole"]) + list(e["pre"])},
                       key=lambda k: (-sum(e["pre"].get(k, 0) for e in res["entries"].values()), k))
        L += ["", "### 规则条目触发占比（条目步数 / 总步数；每格「到点或撞之前 / 整集」）", "",
              "| 臂 | 步数（到点或撞前 / 整集） | " + " | ".join(names) + " |", "| --- | --- |" + " --- |" * len(names)]
        for g, e in res["entries"].items():
            cells = [f"{e['pre'].get(k, 0) / e['n_steps_pre']:.2f} / {e['whole'].get(k, 0) / e['n_steps']:.2f}"
                     if e["n_steps"] and e["n_steps_pre"] else "—" for k in names]
            L.append(f"| `{g}` | {e['n_steps_pre']} / {e['n_steps']} | " + " | ".join(cells) + " |")
    if res["depth_occ"]:
        L += ["", f"### depth 与 occ 列距离一致性（同一位姿；|d_depth − d_occ| 的中位数 / 90 分位；漏检 = depth 比 occ 远 ≥ {MISS_M} m 的列占比；"
                  "反向 = occ 比 depth 远 ≥ 同值）", "",
              "| 范围 | 列集 | 臂 | 列数 | 中位数 m | 90 分位 m | 漏检 | 反向 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for scope, label in (("all", "全部步"), ("pre", "到点或撞之前")):
            for colset, cname in (("all", "全部列"), ("obs", "有障碍的列（至少一侧 < 4 m）")):
                groups = [(g, v[scope]) for g, v in res["depth_occ"].items()] + [("合计", [p for v in res["depth_occ"].values() for p in v[scope]])]
                for g, pairs in groups:
                    st = column_stats(pairs if colset == "all" else obstacle_cols(pairs))
                    gname = g if g == "合计" else f"`{g}`"
                    L.append(f"| {label} | {cname} | {gname} | {st['n']} | {f2(st['median'])} | {f2(st['p90'])} "
                             f"| {f2(st['depth_farther'])} | {f2(st['occ_farther'])} |")
    L += ["", f"同 seed 各臂起飞后首步位姿的最大差（位置 m，航向 °/100 取大者）：{res['spawn_pair_max_diff']:.2e}（≈0 说明逐集配对的出生位姿一致）。"]
    return L


def render_old(cls, res):
    o = res["old"]
    L = [f"### {CLASS_TITLE[cls]}：与旧实验同名臂 `{REF_ARM}` 对照", ""]
    if o is None:
        return L + ["没有 `forward+route` 臂。"]
    if o["rerun"] is None:
        return L + [f"旧实验里没有与本次 `{REF_ARM}` 同键（场景, 路线, seed）的集（未提供旧目录，或键对不上），不能逐集对照。"]
    r = o["rerun"]
    n = r["matched"]
    L += [f"同键集 {n}（本次 `{REF_ARM}` 共 {o['keys_new']} 集）：到点判定一致 {r['success_equal']}/{n}，碰撞判定一致 {r['collision_equal']}/{n}，"
          f"（步数, 到点步, 碰撞步）完全一致 {r['trajectory_equal']}/{n}，出生位姿一致 {r['start_pose_equal']}/{n}；到点数 本次 {r['k_new']}、旧 {r['k_old']}。", "",
          "旧实验各臂在同一批（场景, 路线, seed）上的成绩（参照；只取与本次 `forward+route` 同键的集）：", "",
          "| 旧臂 | 集数 | SR(stop) [Wilson 95%] | 碰撞率 | 超时率 |", "| --- | --- | --- | --- | --- |"]
    for g, a in o["reference"].items():
        L.append(f"| `{g}` | {a['n_req']} | {ci_cell(a)} | {f2(a['collision'])} | {f2(a['timeout'])} |")
    return L


def render_routes(cls, res, data):
    cfg, arms, K = res["config"], data["arms"], res["config"]["k"]
    L = [f"### {CLASS_TITLE[cls]}：分路线 SR(stop)（每格 = 到点集数 / {K}）", "",
         "| 场景 r | " + " | ".join(f"`{g}`" for g in arms) + " |", "| --- |" + " --- |" * len(arms)]
    for s in cfg["scenes"]:
        for r in routes_of(cfg, s):
            x = data["route_info"].get((s, r))
            name = f"{s} r{r}" if x is None else f"{s} r{r}（{x['name']}，{x['astar_path_m']:.1f} m）"
            cells = [f"{res['route_cells'][f'{g}|{s}|{r}']}/{K}" if f"{g}|{s}|{r}" in res["route_cells"] else "—" for g in arms]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--old-indoor", default=None)
    ap.add_argument("--old-outdoor", default=None)
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    old_dirs = {"indoor": a.old_indoor, "outdoor": a.old_outdoor}
    lines = ["# 3DGS 上的规则层上限（rule-oracle-3dgs）", "",
             f"- metric_version `{METRIC_VERSION}`；输出目录 `{out}`。",
             "- SR(stop) = 首次进入 waypoint 半径早于首次碰撞；碰撞率含到点之后的碰撞（与旧实验同口径，预注册的碰撞率用它），"
             "撞在前 = 没到点就撞；超时 = 既没到点也没撞；SPL(stop) 把成功换成 SR(stop)。",
             "- 规则臂的列距离来自占据真值（occ）或渲染深度（depth），航向参照来自 oracle 路线或 waypoint 方位，都是特权信息：这是规则层的上限，不是学生能力。", ""]
    summary, any_class = {"metric_version": METRIC_VERSION}, False
    classes = {}
    for cls in ("indoor", "outdoor"):
        data = load_class(os.path.join(out, cls))
        if data is None:
            continue
        any_class = True
        res = summarize_class(cls, data, load_old(old_dirs[cls]))
        classes[cls] = (res, data)
        lines += render_class(cls, res, data) + [""]
        summary[cls] = {k: v for k, v in res.items() if k not in ("config", "depth_occ")}
        summary[cls]["depth_occ"] = {g: depth_occ_summary(v) for g, v in res["depth_occ"].items()}
    if not any_class:
        print(f"{out} 下没有 indoor / outdoor 的 config.json", file=sys.stderr)
        return 2
    lines += ["## 与旧实验的对照", ""]
    for cls, (res, _) in classes.items():
        lines += render_old(cls, res) + [""]
    lines += ["## 分路线明细", ""]
    for cls, (res, data) in classes.items():
        lines += render_routes(cls, res, data) + [""]
    failed = [x for _, (res, _) in classes.items() for x in res["failed"]]
    lines += [f"失败集（驱动没出 metrics，按失败计、单列）：{len(failed)}" + ("" if not failed else "，见 summary.json `failed`"),
              "", "来源：本目录 `{indoor,outdoor}/<臂>/<场景>/r<N>/ep_*/{metrics.json,rule.jsonl,actions.jsonl}`；输入 sha 见 `inputs.txt`。"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
