#!/usr/bin/env python3
"""3dgs-rebaseline 汇总（任务卡 S1b）：新旧碰撞口径（collision_model v1 vs v2）逐集配对、v1 对原实验复现对拍、续跑半成品记录。

只读 <OUT> 下 6 个套件 {indoor40,holdout,outdoor}_{v1,v2}/<臂>/<场景>/r<N>/{done.json, ep_*/metrics.json, ep_*/trace.jsonl}，
以及原实验目录（W1-11、W1-12、K1、K3）的已跑完结果。
读路线目录一律走 sim.3dgs.route_dirs（审核 E-12：只收 r<数字>，排除 r<N>.incomplete-<时刻>）；
同时用 glob 单独扫描 *.incomplete-*，按公共规则 §6（审核 E-1）专列一节「续跑与半成品记录」，列出每个重评过的 (套件, 臂, 场景, 路线) 及半成品里已完成的集数与到点/碰撞/超时计数。

口径与版本串：
  v1 评测口径版本串：`3dgs.pilot.v2+stop.v1`
  v2 评测口径版本串：`3dgs.pilot.v2+stop.v1+coll.v2`（sim.3dgs.collision.metric_version）
  分母一律用请求集数 n_req = sum_{预期路线} K（完成集少于请求集时差额记入未完成/失败，不缩小分母）。
  逐集配对（v2 对 v1）：同（路线集, 臂, 场景, 路线, seed）；由于 v2 出生检查是层带 + 欧氏膨胀核，可能比 v1 多拒绝候选出生点导致重抽，
  汇总同时列出「全部配对集」「同出生位姿集（位置差 <= 1e-6 m、航向差 <= 1e-6°）」「出生位姿不同集」三组计数。

预注册对照（任务卡 S1 原文，写在 run.sh 头部）：
  1. v1 对原实验同键逐集对拍（success_stop、collision、collide_first、timeout、steps、reach_step、collision_step、start_pose）：应当 0 不一致。
  2. 规则臂（rule-occ+route、rule2-occ+route、rule2-depth+route）在室内正式 40 条（120 集）上 v2 对 v1 的到点成功数变化 |k_v2 - k_v1| 是否 <= 3（留出 96 集与室外 120 集一并报出）。
  3. 规则 v2（rule2-occ+route、rule2-depth+route）在室内正式 40 条上到点成功数下降 (k_v1 - k_v2) 是否 > 6：若是，标出「窄通道（半宽 0.15 m）依赖了旧口径的宽松，E4 的教师要改用半宽 0.25 m 的变体」。
  4. 教师臂（teacher-tol、teacher-swept）与 forward+route：不预测方向，照实报 v1 与 v2 的到点、撞在前、超时、碰撞率与逐集翻转数（b = v2 成 v1 败，c = v1 成 v2 败），并对照 W1-12 的 swept 相对 ray 在 v2 口径下还剩多少增益。

用法：
  python summarize.py <OUT> [--w111 <dir>] [--w112 <dir>] [--k1 <dir>] [--k3 <dir>]
退出码：
  0 = 汇总生成完毕且（若未加 --allow-v1-mismatch）已跑完的 v1 集与原实验 0 不一致；
  6 = v1 与原实验存在逐集不一致（除非加 --allow-v1-mismatch）；
  2 = 目录缺失或无可汇总套件。
"""
import argparse
import glob
import importlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

rd = importlib.import_module("sim.3dgs.route_dirs")
coll_mod = importlib.import_module("sim.3dgs.collision")

METRIC_BASE = "3dgs.pilot.v2+stop.v1"
METRIC_V1 = coll_mod.metric_version(METRIC_BASE, "v1")
METRIC_V2 = coll_mod.metric_version(METRIC_BASE, "v2")
BASE_CLASSES = ("indoor40", "holdout", "outdoor")
MODELS = ("v1", "v2")
POSE_TOL = 1e-6
MATCH_FIELDS = ("success_stop", "collision", "collide_first", "timeout", "steps", "reach_step", "collision_step")
RULE_ARMS = ("rule-occ+route", "rule2-occ+route", "rule2-depth+route")
RULE_V2_ARMS = ("rule2-occ+route", "rule2-depth+route")
INCOMPLETE_RE = re.compile(r"^(r\d+)\.incomplete-(.+)$")


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def routes_of(cfg, scene):
    r = cfg["routes"]
    return list(r.get(scene, [])) if isinstance(r, dict) else list(r)


def wilson_ci(k, n, z=1.959963984540054):
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    p_tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2.0 * p_tail)


def pose_same(a, b, tol=POSE_TOL):
    if not a or not b:
        return False
    return all(abs(float(a[k]) - float(b[k])) <= tol for k in ("x", "y", "yaw_deg"))


def scan_incompletes(suite_dir):
    """公共规则 §6（审核 E-1）：扫描 suite_dir/<臂>/<场景>/r<N>.incomplete-<时刻>，统计半成品里已落盘的集数与结局。"""
    records = []
    if not os.path.isdir(suite_dir):
        return records
    for p in sorted(glob.glob(os.path.join(suite_dir, "*", "*", "r*.incomplete-*"))):
        if not os.path.isdir(p):
            continue
        base = os.path.basename(p)
        m = INCOMPLETE_RE.match(base)
        if not m:
            continue
        r_tag, ts = m.group(1), m.group(2)
        scene = os.path.basename(os.path.dirname(p))
        arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
        eps = []
        for mp in sorted(glob.glob(os.path.join(p, "ep_*", "metrics.json"))):
            mj = read_json(mp)
            if mj is not None:
                eps.append(mj)
        records.append({
            "arm": arm,
            "scene": scene,
            "route": int(r_tag[1:]),
            "timestamp": ts,
            "dir_name": base,
            "episodes_done": len(eps),
            "k_stop": sum(bool(e.get("success_stop")) for e in eps),
            "k_collide_first": sum(bool(e.get("collide_first")) for e in eps),
            "k_timeout": sum(bool(e.get("timeout")) for e in eps),
        })
    return records


def load_suite(suite_dir):
    """读取一个套件（如 indoor40_v1）。路线目录严格使用 sim.3dgs.route_dirs.route_dirs，排除 *.incomplete-*。"""
    cfg = read_json(os.path.join(suite_dir, "config.json"))
    if cfg is None:
        return None
    cm = cfg.get("collision_model", "v1")
    want_mv = coll_mod.metric_version(METRIC_BASE, cm)
    arms = [a["name"] for a in cfg["arms"]]
    k = int(cfg["k"])
    expected_routes = [(g, s, int(r)) for g in arms for s in cfg["scenes"] for r in routes_of(cfg, s)]
    done_routes = set()
    failures = []
    # 只走 route_dirs（depth=2：<suite_dir>/<臂>/<场景>/r<N>）
    valid_rdirs = {os.path.abspath(d) for d in rd.route_dirs(suite_dir, depth=2)}
    episodes = {}
    missing_trace = 0
    mv_mismatch = 0
    for g, s, r in expected_routes:
        d = os.path.abspath(os.path.join(suite_dir, g, s, f"r{r}"))
        if d not in valid_rdirs:
            continue
        dj = read_json(os.path.join(d, "done.json"))
        if dj is None:
            continue
        if dj.get("failures"):
            for fl in dj["failures"]:
                failures.append({"arm": g, "scene": s, "route": r, **fl})
        if not dj.get("failures") and len(dj.get("episodes", [])) >= k:
            done_routes.add((g, s, r))
        for mp in sorted(glob.glob(os.path.join(d, "ep_*", "metrics.json"))):
            mj = read_json(mp)
            if mj is None:
                continue
            ep_dir = os.path.dirname(mp)
            has_trace = os.path.isfile(os.path.join(ep_dir, "trace.jsonl"))
            if not has_trace:
                missing_trace += 1
            if mj.get("metric_version") != want_mv or mj.get("collision_model") != cm:
                mv_mismatch += 1
            mj["_has_trace"] = has_trace
            episodes[(g, s, int(mj["route"]), int(mj["seed"]))] = mj

    missing_routes = [list(x) for x in expected_routes if x not in done_routes]
    incompletes = scan_incompletes(suite_dir)
    n_req_per_arm = {g: k * sum(1 for s in cfg["scenes"] for _ in routes_of(cfg, s)) for g in arms}
    complete = (
        len(missing_routes) == 0
        and len(failures) == 0
        and all(sum(1 for key in episodes if key[0] == g) == n_req_per_arm[g] for g in arms)
    )
    return {
        "cfg": cfg,
        "collision_model": cm,
        "metric_version": want_mv,
        "arms": arms,
        "k": k,
        "n_req_per_arm": n_req_per_arm,
        "episodes": episodes,
        "done_routes": done_routes,
        "missing_routes": missing_routes,
        "failures": failures,
        "incompletes": incompletes,
        "missing_trace": missing_trace,
        "mv_mismatch": mv_mismatch,
        "complete": complete,
    }


def arm_stats(suite_data, arm):
    """单臂汇总：分母固定为请求集数 n_req（公共规则 §2、§6）。"""
    n_req = suite_data["n_req_per_arm"][arm]
    rows = [m for k, m in suite_data["episodes"].items() if k[0] == arm]
    n_done = len(rows)
    n_unfinished = max(0, n_req - n_done)
    k_stop = sum(bool(m["success_stop"]) for m in rows)
    k_coll = sum(bool(m["collision"]) for m in rows)
    k_cf = sum(bool(m["collide_first"]) for m in rows)
    k_to = sum(bool(m["timeout"]) for m in rows)
    k_spawn_coll = sum(m.get("collision_step") == 0 for m in rows)
    spls = [float(m["spl_stop"]) for m in rows if m.get("spl_stop") is not None]
    # 未完成集对 SPL 按 0 计入请求分母
    spl_mean = (sum(spls) / n_req) if (n_req > 0 and len(spls) > 0) else None
    steps_mean = (sum(int(m["steps"]) for m in rows) / n_done) if n_done > 0 else None
    ci = wilson_ci(k_stop, n_req)
    return {
        "n_req": n_req,
        "n_done": n_done,
        "n_unfinished": n_unfinished,
        "k_stop": k_stop,
        "sr_stop": (k_stop / n_req) if n_req else 0.0,
        "sr_ci": [round(ci[0], 4), round(ci[1], 4)],
        "k_coll": k_coll,
        "coll_rate": (k_coll / n_req) if n_req else 0.0,
        "k_collide_first": k_cf,
        "cf_rate": (k_cf / n_req) if n_req else 0.0,
        "k_timeout": k_to,
        "timeout_rate": (k_to / n_req) if n_req else 0.0,
        "k_spawn_coll": k_spawn_coll,
        "spl_stop": round(spl_mean, 4) if spl_mean is not None else None,
        "steps_mean": round(steps_mean, 2) if steps_mean is not None else None,
    }


def pair_v2_vs_v1(s_v1, s_v2, arm):
    """同一路线集上同一臂的 v2 对 v1 逐集配对，同时拆分「同出生位姿」与「出生位姿不同（重抽）」两组。"""
    e1, e2 = s_v1["episodes"], s_v2["episodes"]
    keys = sorted(k[1:] for k in e1 if k[0] == arm and (arm, *k[1:]) in e2)

    def calc(sub_keys):
        n = len(sub_keys)
        k1 = sum(bool(e1[(arm, *k)]["success_stop"]) for k in sub_keys)
        k2 = sum(bool(e2[(arm, *k)]["success_stop"]) for k in sub_keys)
        b = sum(bool(e2[(arm, *k)]["success_stop"]) and not bool(e1[(arm, *k)]["success_stop"]) for k in sub_keys)  # v2 成 v1 败
        c = sum(bool(e1[(arm, *k)]["success_stop"]) and not bool(e2[(arm, *k)]["success_stop"]) for k in sub_keys)  # v1 成 v2 败
        cf1 = sum(bool(e1[(arm, *k)]["collide_first"]) for k in sub_keys)
        cf2 = sum(bool(e2[(arm, *k)]["collide_first"]) for k in sub_keys)
        to1 = sum(bool(e1[(arm, *k)]["timeout"]) for k in sub_keys)
        to2 = sum(bool(e2[(arm, *k)]["timeout"]) for k in sub_keys)
        return {
            "pairs": n,
            "k_v1": k1,
            "k_v2": k2,
            "diff_v2_minus_v1": k2 - k1,
            "b_v2_only": b,
            "c_v1_only": c,
            "p_mcnemar": round(exact_mcnemar(b, c), 6),
            "cf_v1": cf1,
            "cf_v2": cf2,
            "to_v1": to1,
            "to_v2": to2,
        }

    same_keys = [k for k in keys if pose_same(e1[(arm, *k)].get("start_pose"), e2[(arm, *k)].get("start_pose"))]
    diff_keys = [k for k in keys if not pose_same(e1[(arm, *k)].get("start_pose"), e2[(arm, *k)].get("start_pose"))]
    return {
        "arm": arm,
        "all": calc(keys),
        "same_spawn": calc(same_keys),
        "diff_spawn": calc(diff_keys),
    }


def pair_between_arms(suite_data, arm_a, arm_b):
    """同一口径下两臂配对（例如 v2 口径下 teacher-swept 对 teacher-tol，检验 W1-12 的扫掠检查在 v2 下还剩多少作用）。"""
    ep = suite_data["episodes"]
    keys = sorted(k[1:] for k in ep if k[0] == arm_a and (arm_b, *k[1:]) in ep)
    ka = sum(bool(ep[(arm_a, *k)]["success_stop"]) for k in keys)
    kb = sum(bool(ep[(arm_b, *k)]["success_stop"]) for k in keys)
    b = sum(bool(ep[(arm_a, *k)]["success_stop"]) and not bool(ep[(arm_b, *k)]["success_stop"]) for k in keys)
    c = sum(not bool(ep[(arm_a, *k)]["success_stop"]) and bool(ep[(arm_b, *k)]["success_stop"]) for k in keys)
    return {"arm_a": arm_a, "arm_b": arm_b, "pairs": len(keys), "ka": ka, "kb": kb,
            "b_a_only": b, "c_b_only": c, "diff": ka - kb, "p_mcnemar": round(exact_mcnemar(b, c), 6)}


def load_ref_arm(ref_suite_dir, ref_arm):
    """用 route_dirs 读原实验目录下单个臂的全部 metrics.json：返回 {(scene, route, seed): metrics}。"""
    if not ref_suite_dir or not os.path.isdir(os.path.join(ref_suite_dir, ref_arm)):
        return None
    out = {}
    for mp in rd.episode_files(os.path.join(ref_suite_dir, ref_arm), "metrics.json", depth=1):
        mj = read_json(mp)
        if mj is not None:
            out[(mj["scene"], int(mj["route"]), int(mj["seed"]))] = mj
    return out


def check_v1_against_originals(suites_v1, refs):
    """核对本次 v1 口径下每个臂与原实验（W1-11 / W1-12 / K1 / K3）有交集的套件是否逐集一致。"""
    # (base_cls, 本次臂名) -> [(原实验标签, 原实验套件目录, 原实验臂名)]
    mapping = {
        ("indoor40", "teacher-tol"): [("W1-11 indoor", refs.get("w111_in"), "teacher-tol")],
        ("outdoor", "teacher-tol"): [("W1-11 outdoor", refs.get("w111_out"), "teacher-tol")],
        ("indoor40", "teacher-swept"): [("W1-12 indoor", refs.get("w112_in"), "swept")],
        ("outdoor", "teacher-swept"): [("W1-12 outdoor", refs.get("w112_out"), "swept")],
        ("indoor40", "forward+route"): [("K1 indoor", refs.get("k1_in"), "forward+route")],
        ("holdout", "forward+route"): [("K3 holdout", refs.get("k3_hold"), "forward+route")],
        ("outdoor", "forward+route"): [("K1 outdoor", refs.get("k1_out"), "forward+route")],
        ("indoor40", "rule-occ+route"): [("K1 indoor", refs.get("k1_in"), "rule-occ+route"),
                                         ("K3 formal40", refs.get("k3_f40"), "rule-occ+route")],
        ("holdout", "rule-occ+route"): [("K3 holdout", refs.get("k3_hold"), "rule-occ+route")],
        ("outdoor", "rule-occ+route"): [("K1 outdoor", refs.get("k1_out"), "rule-occ+route")],
        ("indoor40", "rule2-occ+route"): [("K3 formal40", refs.get("k3_f40"), "rule2-occ+route")],
        ("holdout", "rule2-occ+route"): [("K3 holdout", refs.get("k3_hold"), "rule2-occ+route")],
        ("outdoor", "rule2-occ+route"): [("K3 outdoor", refs.get("k3_out"), "rule2-occ+route")],
        ("indoor40", "rule2-depth+route"): [("K3 formal40", refs.get("k3_f40"), "rule2-depth+route")],
        ("holdout", "rule2-depth+route"): [("K3 holdout", refs.get("k3_hold"), "rule2-depth+route")],
        ("outdoor", "rule2-depth+route"): [("K3 outdoor", refs.get("k3_out"), "rule2-depth+route")],
    }
    checks = []
    for (bcls, arm), targets in sorted(mapping.items()):
        s1 = suites_v1.get(bcls)
        if s1 is None:
            continue
        new_rows = {(k[1], k[2], k[3]): m for k, m in s1["episodes"].items() if k[0] == arm}
        if not new_rows:
            continue
        for label, rdir, rarm in targets:
            old_rows = load_ref_arm(rdir, rarm)
            if old_rows is None:
                checks.append({"class": bcls, "arm": arm, "ref": label, "status": "ref_missing",
                               "matched": 0, "equal": 0, "diffs": []})
                continue
            common = sorted(set(new_rows) & set(old_rows))
            diffs = []
            for k in common:
                bad = [f for f in MATCH_FIELDS if new_rows[k].get(f) != old_rows[k].get(f)]
                if not pose_same(new_rows[k].get("start_pose"), old_rows[k].get("start_pose")):
                    bad.append("start_pose")
                if bad:
                    diffs.append({"key": list(k), "fields": bad,
                                  "new": {f: new_rows[k].get(f) for f in MATCH_FIELDS},
                                  "old": {f: old_rows[k].get(f) for f in MATCH_FIELDS}})
            checks.append({
                "class": bcls,
                "arm": arm,
                "ref": label,
                "status": "ok" if (len(diffs) == 0 and len(common) == len(new_rows)) else ("mismatch" if diffs else "partial_overlap"),
                "n_new": len(new_rows),
                "n_ref": len(old_rows),
                "matched": len(common),
                "equal": len(common) - len(diffs),
                "diffs": diffs[:20],
                "n_diffs": len(diffs),
            })
    return checks


def evaluate_prereg(pairs_by_class, complete_all):
    """按任务卡「起跑前写下的预期」生成对照结论（预注册写在 run.sh 头部）。"""
    out = {"complete": complete_all, "rule_change_le_3_indoor40": {}, "rule_v2_drop_gt_6_indoor40": {}, "e4_teacher_note": None}
    ind = pairs_by_class.get("indoor40", {})
    drops_gt_6 = []
    for arm in RULE_ARMS:
        if arm in ind:
            p = ind[arm]["all"]
            diff = p["diff_v2_minus_v1"]
            out["rule_change_le_3_indoor40"][arm] = {
                "pairs": p["pairs"], "k_v1": p["k_v1"], "k_v2": p["k_v2"],
                "diff_v2_minus_v1": diff, "abs_le_3": abs(diff) <= 3,
            }
    for arm in RULE_V2_ARMS:
        if arm in ind:
            p = ind[arm]["all"]
            drop = p["k_v1"] - p["k_v2"]
            trig = drop > 6
            out["rule_v2_drop_gt_6_indoor40"][arm] = {
                "pairs": p["pairs"], "k_v1": p["k_v1"], "k_v2": p["k_v2"],
                "drop": drop, "drop_gt_6": trig,
            }
            if trig:
                drops_gt_6.append(f"{arm} 下降 {drop}/{p['pairs']}")
    if not complete_all:
        out["e4_teacher_note"] = "数据未齐，暂不下正式结论"
    elif drops_gt_6:
        out["e4_teacher_note"] = (
            f"触发证伪条件（{', '.join(drops_gt_6)} > 6/120）：窄通道（半宽 0.15 m）依赖了旧口径的宽松，"
            "E4 的教师应改用半宽 0.25 m 的变体"
        )
    else:
        out["e4_teacher_note"] = "未触发 > 6/120 下降条件：规则 v2 窄通道（半宽 0.15 m）未表现出对旧宽松口径的依赖"
    return out


def fmt_rate(k, n, ci):
    if n <= 0:
        return "—"
    return f"{k}/{n} = {k / n:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return lines


def render_markdown(out_dir, suites, arm_tables, pairs_by_class, cross_arm, v1_checks, prereg):
    complete_all = prereg["complete"]
    L = [
        "# 3DGS 碰撞口径重定基线汇总（任务卡 S1b）",
        "",
        f"- 输出目录：`{out_dir}`",
        f"- 口径版本串：v1 = `{METRIC_V1}`，v2 = `{METRIC_V2}`（B1 欧氏距离膨胀核 + B2/B4/B7 层带 `z ± 0.053 m` 统一高度口径 + B3 SITL 10 Hz `trace.jsonl` 折线逐段扫）。",
        f"- 数据完整度：{'**全部完成**' if complete_all else '**部分完成 / 冒烟子集（数据不全，不作正式判决）**'}；分母一律为请求集数。",
        "",
        "## 1. 预注册预期对照",
        "",
        f"- E4 教师变体判读：{prereg['e4_teacher_note']}。",
    ]
    if prereg["rule_change_le_3_indoor40"]:
        rows = []
        for arm, d in prereg["rule_change_le_3_indoor40"].items():
            v2_trig = prereg["rule_v2_drop_gt_6_indoor40"].get(arm, {}).get("drop_gt_6")
            rows.append([
                f"`{arm}`", d["pairs"], d["k_v1"], d["k_v2"], f"{d['diff_v2_minus_v1']:+d}",
                "是" if d["abs_le_3"] else "否",
                "—" if v2_trig is None else ("**是（>6）**" if v2_trig else "否"),
            ])
        L += [""] + md_table(["室内正式 40 条规则臂", "配对集", "v1 到点", "v2 到点", "v2 − v1", "|变化| ≤ 3", "下降 > 6"], rows)

    L += ["", "## 2. 各路线集 × 各臂：v1 与 v2 结果及逐集配对", ""]
    titles = {
        "indoor40": "室内正式 40 条（K1 / W1-11 / W1-12 同批路线与 seed）",
        "holdout": "室内留出 32 条（K3 留出路线与 seed）",
        "outdoor": "室外 4 场景 × 6 条路线（K1 / K3 / W1-11 / W1-12 同批路线与 seed）",
    }
    for bcls in BASE_CLASSES:
        s1, s2 = suites.get(f"{bcls}_v1"), suites.get(f"{bcls}_v2")
        if s1 is None and s2 is None:
            continue
        L += [f"### 2.{BASE_CLASSES.index(bcls) + 1} {titles[bcls]}", ""]
        arms = (s1 or s2)["arms"]
        rows = []
        for arm in arms:
            a1 = arm_tables.get(f"{bcls}_v1", {}).get(arm)
            a2 = arm_tables.get(f"{bcls}_v2", {}).get(arm)
            p = pairs_by_class.get(bcls, {}).get(arm)
            c1 = fmt_rate(a1["k_stop"], a1["n_req"], a1["sr_ci"]) if a1 else "—"
            c2 = fmt_rate(a2["k_stop"], a2["n_req"], a2["sr_ci"]) if a2 else "—"
            cf_str = f"{a1['k_collide_first']} → {a2['k_collide_first']}" if (a1 and a2) else "—"
            to_str = f"{a1['k_timeout']} → {a2['k_timeout']}" if (a1 and a2) else "—"
            sp_str = f"{a1['k_spawn_coll']} → {a2['k_spawn_coll']}" if (a1 and a2) else "—"
            if p:
                pa, ps, pd = p["all"], p["same_spawn"], p["diff_spawn"]
                pair_str = f"{pa['diff_v2_minus_v1']:+d} (b={pa['b_v2_only']}, c={pa['c_v1_only']}, p={pa['p_mcnemar']:.4f})"
                spawn_split = f"同起点 {ps['pairs']} 对({ps['diff_v2_minus_v1']:+d}) / 异起点 {pd['pairs']} 对({pd['diff_v2_minus_v1']:+d})"
            else:
                pair_str, spawn_split = "—", "—"
            rows.append([f"`{arm}`", c1, c2, cf_str, to_str, sp_str, pair_str, spawn_split])
        L += md_table(
            ["臂", f"v1 到点 (`{METRIC_V1}`)", f"v2 到点 (`{METRIC_V2}`)", "撞在前 v1→v2", "超时 v1→v2", "出生即撞 v1→v2", "配对 v2−v1 (b=v2成v1败, c=v1成v2败)", "同出生位姿 / 异出生位姿拆分"],
            rows,
        )
        if bcls in cross_arm:
            ca = cross_arm[bcls]
            L += ["", "同口径下 `teacher-swept` 对 `teacher-tol`（核对 W1-12 的扫掠检查在 v2 下还剩多少作用）：", ""]
            crows = []
            for cm in MODELS:
                if cm in ca:
                    x = ca[cm]
                    crows.append([cm, x["pairs"], x["ka"], x["kb"], f"{x['diff']:+d}", x["b_a_only"], x["c_b_only"], f"{x['p_mcnemar']:.4f}"])
            L += md_table(["口径", "配对数", "swept 到点", "tol(ray) 到点", "swept − tol", "b(swept成)", "c(tol成)", "McNemar p"], crows)
        L.append("")

    L += ["## 3. v1 口径与原实验逐集对拍", ""]
    if not v1_checks:
        L.append("- 未提供原实验目录或尚未产生有交集的 v1 集。")
    else:
        vrows = []
        for c in v1_checks:
            vrows.append([c["class"], f"`{c['arm']}`", c["ref"], c.get("n_new", 0), c["matched"], c["equal"], c.get("n_diffs", 0), c["status"]])
        L += md_table(["路线集", "臂", "原实验来源", "本次集数", "同键集数", "逐项一致", "不一致", "状态"], vrows)
        for c in v1_checks:
            for d in c.get("diffs", []):
                L.append(f"- 不一致：`{c['class']} {c['arm']}` vs `{c['ref']}` 键 `{d['key']}` 差异字段 `{d['fields']}`：本次 `{d['new']}` vs 原实验 `{d['old']}`")

    L += ["", "## 4. 续跑与半成品记录（审核 E-1）", ""]
    all_inc = []
    for sname, sdata in sorted(suites.items()):
        for rec in sdata["incompletes"]:
            all_inc.append((sname, rec))
    if not all_inc:
        L.append("- 本次输出目录下无 `*.incomplete-*` 半成品目录（没有路线发生过中断重评）。")
    else:
        irows = [
            [sname, f"`{r['arm']}`", r["scene"], f"r{r['route']}", f"`{r['dir_name']}`",
             r["episodes_done"], r["k_stop"], r["k_collide_first"], r["k_timeout"]]
            for sname, r in all_inc
        ]
        L += md_table(["套件", "臂", "场景", "路线", "半成品目录", "已完成集数", "到点", "撞在前", "超时"], irows)

    L += ["", "## 5. 产物完整性检查（trace.jsonl 与口径后缀）", ""]
    trows = []
    for sname, sdata in sorted(suites.items()):
        trows.append([
            sname, f"`{sdata['metric_version']}`", len(sdata["episodes"]),
            len(sdata["missing_routes"]), len(sdata["failures"]),
            sdata["missing_trace"], sdata["mv_mismatch"],
        ])
    L += md_table(["套件", "期望 metric_version", "已落盘集数", "缺 done 路线数", "失败记录数", "缺 trace.jsonl 集数", "版本串不符集数"], trows)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--w111", default=None, help="exp-teacher-tol-3dgs-20261110 目录")
    ap.add_argument("--w112", default=None, help="exp-teacher-clearance-3dgs-20261111 目录")
    ap.add_argument("--k1", default=None, help="exp-rule-oracle-3dgs-20261110 目录")
    ap.add_argument("--k3", default=None, help="exp-rule-v2-3dgs-20261111 目录")
    ap.add_argument("--allow-v1-mismatch", action="store_true", help="v1 与原实验不一致时仍返回 0（仅诊断用）")
    args = ap.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    suites = {}
    for bcls in BASE_CLASSES:
        for cm in MODELS:
            sname = f"{bcls}_{cm}"
            sd = load_suite(os.path.join(out_dir, sname))
            if sd is not None:
                suites[sname] = sd
    if not suites:
        print(f"{out_dir} 下没有找到任何 *_v1 / *_v2 套件的 config.json", file=sys.stderr)
        return 2

    arm_tables = {}
    for sname, sdata in suites.items():
        arm_tables[sname] = {arm: arm_stats(sdata, arm) for arm in sdata["arms"]}

    pairs_by_class = {}
    cross_arm = {}
    for bcls in BASE_CLASSES:
        s1, s2 = suites.get(f"{bcls}_v1"), suites.get(f"{bcls}_v2")
        if s1 and s2:
            common_arms = [a for a in s1["arms"] if a in s2["arms"]]
            pairs_by_class[bcls] = {arm: pair_v2_vs_v1(s1, s2, arm) for arm in common_arms}
        for cm, sd in (("v1", s1), ("v2", s2)):
            if sd and "teacher-swept" in sd["arms"] and "teacher-tol" in sd["arms"]:
                cross_arm.setdefault(bcls, {})[cm] = pair_between_arms(sd, "teacher-swept", "teacher-tol")

    refs = {
        "w111_in": os.path.join(args.w111, "indoor") if args.w111 else None,
        "w111_out": os.path.join(args.w111, "outdoor") if args.w111 else None,
        "w112_in": os.path.join(args.w112, "indoor") if args.w112 else None,
        "w112_out": os.path.join(args.w112, "outdoor") if args.w112 else None,
        "k1_in": os.path.join(args.k1, "formal", "indoor") if args.k1 else None,
        "k1_out": os.path.join(args.k1, "formal", "outdoor") if args.k1 else None,
        "k3_f40": os.path.join(args.k3, "formal40") if args.k3 else None,
        "k3_hold": os.path.join(args.k3, "holdout") if args.k3 else None,
        "k3_out": os.path.join(args.k3, "outdoor") if args.k3 else None,
    }
    suites_v1 = {bcls: suites[f"{bcls}_v1"] for bcls in BASE_CLASSES if f"{bcls}_v1" in suites}
    v1_checks = check_v1_against_originals(suites_v1, refs)

    expected_suite_names = {f"{b}_{m}" for b in BASE_CLASSES for m in MODELS}
    complete_all = (set(suites.keys()) == expected_suite_names) and all(sd["complete"] for sd in suites.values())
    prereg = evaluate_prereg(pairs_by_class, complete_all)

    any_v1_mismatch = any(c.get("n_diffs", 0) > 0 for c in v1_checks)
    summary = {
        "metric_version_v1": METRIC_V1,
        "metric_version_v2": METRIC_V2,
        "complete": complete_all,
        "any_v1_mismatch": any_v1_mismatch,
        "prereg": prereg,
        "suites": {
            sname: {
                "collision_model": sd["collision_model"],
                "metric_version": sd["metric_version"],
                "complete": sd["complete"],
                "n_episodes": len(sd["episodes"]),
                "missing_routes_count": len(sd["missing_routes"]),
                "failures_count": len(sd["failures"]),
                "missing_trace": sd["missing_trace"],
                "mv_mismatch": sd["mv_mismatch"],
                "incompletes": sd["incompletes"],
                "arms": arm_tables[sname],
            }
            for sname, sd in suites.items()
        },
        "pairs_v2_vs_v1": pairs_by_class,
        "teacher_swept_vs_tol": cross_arm,
        "v1_vs_originals": v1_checks,
    }
    md = render_markdown(out_dir, suites, arm_tables, pairs_by_class, cross_arm, v1_checks, prereg)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    verdict_line = (
        f"complete={complete_all} v1_mismatch={any_v1_mismatch}；{prereg['e4_teacher_note']}"
    )
    with open(os.path.join(out_dir, "verdict.txt"), "w", encoding="utf-8") as f:
        f.write(verdict_line + "\n")

    if any_v1_mismatch and not args.allow_v1_mismatch:
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
