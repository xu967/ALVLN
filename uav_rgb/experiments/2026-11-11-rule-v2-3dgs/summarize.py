#!/usr/bin/env python3
"""rule-v2-3dgs 汇总（任务卡 K3）：留出路线上的主终点、正式 40 条上的次终点、室外的不劣判决（机械套用预注册的数字），加「另报」的失败形态。
只读 <OUT> 下 {holdout,formal40,outdoor}/<臂>/<场景>/r<N>/{done.json, ep_*/{metrics.json, actions.jsonl, rule.jsonl}} 与各类的 config.json、
dev_selection.json / selected.json / k1_match.json，不加载场景，随时可跑；没跑完时只统计已写 done.json 的路线，并在结论里写「数据不全，不判决」。
输出 <OUT>/summary.md、summary.json、verdict.txt（一行，run.sh 的 DONE 引用它）。

口径沿用 K1（实验入口 2026-11-10-rule-oracle-3dgs/summarize.py，本文件复用它的 load_class / agg / fail_counts / entry_shares / exact_mcnemar / wilson_ci）：
  SR(stop) = 首次进入 waypoint 半径早于首次碰撞；碰撞率 = 整集任一步碰撞（含到点之后）；撞在前 = 没到点就撞；超时 = 既没到点也没撞；
  逐集配对 = 同（场景, 路线, seed）两臂都有 metrics.json 的集，精确 McNemar 双侧；点估计对阈值，并列 Wilson 95% 区间。
预注册数字（任务卡 K3 原文，run.sh 头里逐字抄了；这里用 Fraction 做整数边界比较，避免浮点边界）：
  主终点（留出路线，rule2-occ+route 对 rule-occ+route）：SR(stop) 之差 ≥ 0.10 且 p < 0.05，主臂碰撞率 ≤ 0.10、撞在前 ≤ 0.08 → 「达到预测」；
    之差 < 0.05 或主臂碰撞率 > 0.15 → 「证伪」；其余 → 「介于预测与证伪之间」（证伪条件先判）。
  次终点（正式 40 条，只报告）：同样的比较，另报主臂 SR(stop) 是否 ≥ 0.80。
  室外（不劣）：主臂 SR(stop) ≥ 0.93，（v1 成 v2 败）−（v2 成 v1 败）≤ 5，碰撞率 ≤ 0.10；v1 取 K1 的结果按（场景, 路线, seed）配对。
  depth 臂只报告。
另报：失败形态（K1 分类 + V2-stuck 步占比 + 各级 V2 条目占比 + 每集转向步占比与平均步数）、路线长度分桶（≥10 m 单列）、出生即碰撞的集、
  导引目标点不前进的集数、以 V2-stuck 收尾的集逐集列出位置与到占据的最小间隙。
用法：python summarize.py <OUT> [--k1-root <K1 输出目录>]
"""
import argparse
import importlib.util
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name, *rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, *rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


k1s = _load("k1_summarize", "..", "2026-11-10-rule-oracle-3dgs", "summarize.py")
k1f = _load("k1_dev_failure", "..", "2026-11-10-rule-oracle-3dgs", "dev_failure_analysis.py")
import check_k1_match as ckm  # noqa: E402

METRIC_VERSION = k1s.METRIC_VERSION
V1, MAIN, DEPTH, FWD = "rule-occ+route", "rule2-occ+route", "rule2-depth+route", "forward+route"
YAW_LEFT, YAW_RIGHT = 2, 3
LAST_N = 20                              # 与 K1 失败分析同：末 20 步
# 预注册的数字（任务卡 K3 原文）；Fraction 做边界比较
PRE = {"delta_pred": Fraction(1, 10), "p_alpha": 0.05, "coll_pred": Fraction(1, 10), "cf_pred": Fraction(2, 25),
       "delta_fals": Fraction(1, 20), "coll_fals": Fraction(3, 20), "sec_sr": Fraction(4, 5),
       "out_sr": Fraction(93, 100), "out_bc": 5, "out_coll": Fraction(1, 10)}


# ---------------------------------------------------------------- 配对与判决（纯函数，单测直接调）
def paired2(index, a, b):
    """index = {(臂, 场景, 路线, seed): metrics}；返回整数计数：pairs、ka / kb（成功数）、b_count（a 成 b 败）、c_count（a 败 b 成）、p（精确 McNemar）。"""
    keys = sorted(k[1:] for k in index if k[0] == a and (b, *k[1:]) in index)
    ka = sum(bool(index[(a, *k)]["success_stop"]) for k in keys)
    kb = sum(bool(index[(b, *k)]["success_stop"]) for k in keys)
    bc = sum(bool(index[(a, *k)]["success_stop"]) and not index[(b, *k)]["success_stop"] for k in keys)
    cc = sum(not index[(a, *k)]["success_stop"] and bool(index[(b, *k)]["success_stop"]) for k in keys)
    return {"a": a, "b": b, "pairs": len(keys), "ka": ka, "kb": kb, "b_count": bc, "c_count": cc, "p": k1s.exact_mcnemar(bc, cc)}


def route_sign(index, a, b):
    """路线层面的补充（未预注册）：同路线 K 集里 a 的到点数多于 b 的路线数、少于的、相等的；精确符号检验 p。同一路线的几集不独立，这个数不受它夸大。"""
    per = {}
    for k in index:
        if k[0] == a and (b, *k[1:]) in index:
            r = per.setdefault((k[1], k[2]), [0, 0])
            r[0] += bool(index[k]["success_stop"])
            r[1] += bool(index[(b, *k[1:])]["success_stop"])
    win = sum(x > y for x, y in per.values())
    loss = sum(x < y for x, y in per.values())
    return {"routes": len(per), "win": win, "loss": loss, "tie": len(per) - win - loss, "p": k1s.exact_mcnemar(win, loss)}


def judge_primary(pc, main):
    """主终点判决。pc = paired2(主臂对 v1)；main = 主臂的 {n_req, k_coll, k_cf}（整集有撞与撞在前的集数，分母 n_req）。"""
    n = pc["pairs"]
    if n == 0 or main["n_req"] == 0:
        return {"verdict": "无数据", "pairs": n}
    d = pc["ka"] - pc["kb"]
    delta = Fraction(d, n)
    coll, cf = Fraction(main["k_coll"], main["n_req"]), Fraction(main["k_cf"], main["n_req"])
    out = {"pairs": n, "delta": float(delta), "delta_num": d, "p": pc["p"], "coll": float(coll), "cf": float(cf),
           "delta_ge_pred": delta >= PRE["delta_pred"], "p_lt_alpha": pc["p"] < PRE["p_alpha"],
           "coll_le_pred": coll <= PRE["coll_pred"], "cf_le_pred": cf <= PRE["cf_pred"],
           "delta_lt_fals": delta < PRE["delta_fals"], "coll_gt_fals": coll > PRE["coll_fals"]}
    out["predicted"] = out["delta_ge_pred"] and out["p_lt_alpha"] and out["coll_le_pred"] and out["cf_le_pred"]
    out["falsified"] = out["delta_lt_fals"] or out["coll_gt_fals"]
    out["verdict"] = "证伪" if out["falsified"] else "达到预测" if out["predicted"] else "介于预测与证伪之间"
    return out


def judge_secondary(pc, main):
    """次终点（正式 40 条，只报告）：同样的比较，另报主臂 SR(stop) 是否到 0.80。main = 主臂 {n_req, k_stop}。"""
    n = pc["pairs"]
    if n == 0 or main["n_req"] == 0:
        return {"verdict": "无数据", "pairs": n}
    d = pc["ka"] - pc["kb"]
    delta = Fraction(d, n)
    sr = Fraction(main["k_stop"], main["n_req"])
    return {"pairs": n, "delta": float(delta), "delta_num": d, "p": pc["p"], "sr": float(sr), "sr_ge_080": sr >= PRE["sec_sr"],
            "delta_ge_pred": delta >= PRE["delta_pred"], "p_lt_alpha": pc["p"] < PRE["p_alpha"], "verdict": "报告"}


def judge_outdoor(main, pc_v1, v1_missing):
    """室外不劣。main = 主臂 {n_req, k_stop, k_coll}；pc_v1 = paired2(v1（K1）, 主臂)：b_count = v1 成 v2 败，c_count = v2 成 v1 败；v1_missing = K1 缺的键数。"""
    if main["n_req"] == 0:
        return {"verdict": "无数据"}
    if v1_missing:
        return {"verdict": "K1 缺键，不判决", "v1_missing": v1_missing}
    sr, coll = Fraction(main["k_stop"], main["n_req"]), Fraction(main["k_coll"], main["n_req"])
    bc = pc_v1["b_count"] - pc_v1["c_count"]
    out = {"sr": float(sr), "sr_ge": sr >= PRE["out_sr"], "b_v1_only": pc_v1["b_count"], "c_v2_only": pc_v1["c_count"], "b_minus_c": bc,
           "bc_le": bc <= PRE["out_bc"], "coll": float(coll), "coll_le": coll <= PRE["out_coll"], "pairs": pc_v1["pairs"], "p": pc_v1["p"]}
    out["verdict"] = "不劣" if out["sr_ge"] and out["bc_le"] and out["coll_le"] else "不满足不劣"
    return out


# ---------------------------------------------------------------- 读数据
def episode_facts(m):
    """读一集的 actions.jsonl / rule.jsonl，只留汇总事实（不把整集留在内存里）。"""
    act = k1f.read_jsonl(os.path.join(m["_dir"], "actions.jsonl"))
    rule = k1f.read_jsonl(os.path.join(m["_dir"], "rule.jsonl"))
    f = {"n_act": None, "turn": None, "last_entry": None, "last_pose": None, "final_clear": None, "min_clear_last": None, "stalled": False}
    if act:
        f["n_act"] = len(act)
        f["turn"] = sum(a["action_executed"] in (YAW_LEFT, YAW_RIGHT) for a in act)
        p = act[-1]["pose_before"]
        f["last_pose"] = [round(p["x"], 2), round(p["y"], 2), round(act[-1]["yaw_before"], 1)]
        cl = [a["clearance_m"] for a in act[-LAST_N:] if a.get("clearance_m") is not None]
        f["final_clear"] = act[-1].get("clearance_m")
        f["min_clear_last"] = min(cl) if cl else None
    if rule:
        f["last_entry"] = rule[-1]["entry"]
        if act:
            f["stalled"] = bool(k1f.target_stalled({"m": m, "rule": rule, "act": act}, last=LAST_N))
    return f


def load_all_classes(out):
    res = {}
    for cls in ("holdout", "formal40", "outdoor"):
        d = k1s.load_class(os.path.join(out, cls))
        if d is not None:
            res[cls] = d
    return res


def read_json(path):
    return k1s.read_json(path)


def fmt_rate(k, n):
    if not n:
        return "—"
    ci = k1s.wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def md_table(headers, rows):
    L = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return L + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


# ---------------------------------------------------------------- 一个类的汇总
def summarize_class(cls, data):
    cfg, arms, rows = data["cfg"], data["arms"], data["rows"]
    K = cfg["k"]
    by_arm = {g: [x for x in rows if x["arm"] == g] for g in arms}
    expected = {(g, s, r) for g in arms for s in cfg["scenes"] for r in k1s.routes_of(cfg, s)}
    missing = sorted(expected - data["done_keys"])

    def n_req(g, scenes=None):
        return K * len([1 for s in (scenes or cfg["scenes"]) for r in k1s.routes_of(cfg, s) if (g, s, r) in data["done_keys"]])

    res = {"class": cls, "k": K, "arms": arms, "missing_routes": [list(x) for x in missing], "failed": data["failed"],
           "complete": not missing and not data["failed"] and all(len(by_arm[g]) == n_req(g) for g in arms), "overall": {}, "per_scene": {}, "buckets": {},
           "fail": {}, "entries": {}, "facts": {}, "route_cells": {}}
    facts = {}
    for g in arms:
        sel = by_arm[g]
        a = k1s.agg(sel, n_req(g))
        a["k_cf"] = sum(bool(x["collide_first"]) for x in sel)
        res["overall"][g] = a
        res["fail"][g] = {"all": k1s.fail_counts(sel)}
        for s in cfg["scenes"]:
            res["per_scene"][f"{g}|{s}"] = k1s.agg([x for x in sel if x["scene"] == s], n_req(g, [s]))
            for r in k1s.routes_of(cfg, s):
                if (g, s, r) in data["done_keys"]:
                    res["route_cells"][f"{g}|{s}|{r}"] = sum(bool(x["success_stop"]) for x in sel if x["scene"] == s and x["route"] == r)
        for b in data["bucket_labels"] if data["route_info"] else []:
            keys = {k for k, x in data["route_info"].items() if x["bucket"] == b and (g, *k) in data["done_keys"]}
            bsel = [x for x in sel if (x["scene"], x["route"]) in keys]
            res["buckets"][f"{b}|{g}"] = k1s.agg(bsel, K * len(keys))
            res["fail"][g][b] = k1s.fail_counts(bsel)
        if data["arm_cfg"][g]["base"].startswith("rule-"):
            res["entries"][g] = k1s.entry_shares(sel)
        facts[g] = {(x["scene"], x["route"], x["seed"]): episode_facts(x) for x in sel}
        fl = [f for f in facts[g].values() if f["n_act"]]
        res["facts"][g] = {"turn_share_pooled": (sum(f["turn"] for f in fl) / sum(f["n_act"] for f in fl)) if fl else None,
                           "turn_share_mean": (sum(f["turn"] / f["n_act"] for f in fl) / len(fl)) if fl else None,
                           "steps_mean": (sum(x["steps"] for x in sel) / len(sel)) if sel else None, "n_ep": len(sel)}
        fail_keys = {(x["scene"], x["route"], x["seed"]): x for x in sel if not x["success_stop"]}
        res["facts"][g]["spawn_collision"] = sorted([x["scene"], x["route"], x["seed"]] for x in sel if x["collision_step"] == 0)
        res["facts"][g]["collision_first4"] = sum(1 for x in sel if x["collision_step"] is not None and x["collision_step"] <= 3)
        res["facts"][g]["stalled_failed"] = sorted([k[0], k[1], k[2]] for k in fail_keys if facts[g].get(k, {}).get("stalled"))
        res["facts"][g]["stuck_ending"] = []
        for k, x in sorted(fail_keys.items()):
            f = facts[g].get(k)
            if f and f["last_entry"] == "V2-stuck":
                res["facts"][g]["stuck_ending"].append({"scene": k[0], "route": k[1], "seed": k[2], "outcome": "撞在前" if x["collide_first"] else "超时",
                                                        "steps": x["steps"], "last_pose": f["last_pose"], "final_clear": f["final_clear"],
                                                        "min_clear_last20": f["min_clear_last"]})
        res["facts"][g]["success_ending_stuck"] = sum(1 for k, f in facts[g].items() if f["last_entry"] == "V2-stuck" and k not in fail_keys)
    index = {(x["arm"], x["scene"], x["route"], x["seed"]): x for x in rows}
    worst = 0.0
    for k in {k[1:] for k in index}:
        poses = [index[(g, *k)]["start_pose"] for g in arms if (g, *k) in index]
        for q in poses[1:]:
            worst = max(worst, abs(q["x"] - poses[0]["x"]), abs(q["y"] - poses[0]["y"]), abs(q["yaw_deg"] - poses[0]["yaw_deg"]) / 100.0)
    res["spawn_pair_max_diff"] = worst
    return res, index


# ---------------------------------------------------------------- 渲染
def render_class(cls, title, res, data, extra_pairs):
    cfg, arms, K = data["cfg"], res["arms"], res["k"]
    L = [f"### {title}", "",
         f"- 每路线 {K} 集，seed = {cfg['seed0']} + 100·路线 + k，hold {cfg['hold']} s，出生扰动 ±{cfg['jitter']} m / ±10°；本类数据"
         f"{'完整' if res['complete'] else '**不完整**（缺 ' + str(len(res['missing_routes'])) + ' 条路线、失败集 ' + str(len(res['failed'])) + '）'}。", ""]
    rows = []
    for g in arms:
        a, f = res["overall"][g], res["facts"][g]
        rows.append([f"`{g}`", fmt_rate(a["k_stop"], a["n_req"]), fmt_rate(a["k_coll"], a["n_req"]), fmt_rate(a["k_cf"], a["n_req"]), k1s.f2(a["timeout"]),
                     k1s.f2(a["spl_stop"]), "—" if f["steps_mean"] is None else f"{f['steps_mean']:.1f}",
                     "—" if f["turn_share_mean"] is None else f"{f['turn_share_mean']:.3f} / {f['turn_share_pooled']:.3f}", a["n_req"]])
    L += md_table(["臂", "SR(stop) [Wilson 95%]", "碰撞率（整集）", "撞在前", "超时率", "SPL(stop)", "平均步数", "转向步占比（集均 / 合并）", "n"], rows)
    L += ["", f"分场景 SR(stop)（括号内是碰撞率）：", ""]
    L += md_table(["臂"] + cfg["scenes"], [[f"`{g}`"] + [(f"{res['per_scene'][f'{g}|{s}']['k_stop']}/{res['per_scene'][f'{g}|{s}']['n_req']}（{res['per_scene'][f'{g}|{s}']['collision']:.2f}）"
                                                         if res["per_scene"][f"{g}|{s}"]["n_req"] else "—") for s in cfg["scenes"]] for g in arms])
    if data["route_info"]:
        L += ["", "按路线长度分桶（A* 长度，来自路线文件；每格 = 到点 / 请求 = SR，括号内碰撞率；`>=10 m` 单列在最后）：", ""]
        labels = [b for b in data["bucket_labels"] if b != k1s.LONG_BUCKET] + ([k1s.LONG_BUCKET] if k1s.LONG_BUCKET in data["bucket_labels"] else [])
        brows = []
        for b in labels:
            n_plan = sum(1 for k, x in data["route_info"].items() if x["bucket"] == b and (arms[0], *k) in data["done_keys"])
            cells = []
            for g in arms:
                a = res["buckets"][f"{b}|{g}"]
                cells.append(f"{a['k_stop']}/{a['n_req']} = {a['sr_stop']:.2f}（{a['collision']:.2f}）" if a["n_req"] else "—")
            brows.append([f"{b}（{n_plan} 条）"] + cells)
        L += md_table(["桶"] + [f"`{g}`" for g in arms], brows)
    L += ["", "失败集分类（没到点早于撞的集；K1 的分类，整集 action_executed 里有 ≥6 步严格左右交替记「含交替」）：", ""]
    frows = []
    for g in arms:
        for b in ["全部"] + ([x for x in data["bucket_labels"]] if data["route_info"] else []):
            c = res["fail"][g]["all" if b == "全部" else b]
            frows.append([f"`{g}`", b, sum(c.values())] + [c[x] for x in k1s.FAIL_CLASSES])
    L += md_table(["臂", "桶", "失败集"] + k1s.FAIL_CLASSES, frows)
    if res["entries"]:
        names = sorted({k for e in res["entries"].values() for k in list(e["whole"]) + list(e["pre"])}, key=lambda k: (not k.startswith("V2-"), k))
        L += ["", "规则条目触发占比（条目步数 / 总步数；每格「到点或撞之前 / 整集」；V2-stuck = 规则顺序之外的缺口，V2-fwd<ℓ> 的 ℓ = 命中的通道级）：", ""]
        erows = []
        for g, e in res["entries"].items():
            cells = [f"{e['pre'].get(k, 0) / e['n_steps_pre']:.3f} / {e['whole'].get(k, 0) / e['n_steps']:.3f}" if e["n_steps"] and e["n_steps_pre"] else "—" for k in names]
            erows.append([f"`{g}`", f"{e['n_steps_pre']} / {e['n_steps']}"] + cells)
        L += md_table(["臂", "步数（到点或撞前 / 整集）"] + names, erows)
    L += ["", "出生即碰撞（`collision_step == 0`）、前 4 步内碰撞（K1 的口径，步号 ≤ 3）、导引目标点不前进的失败集数（末 20 步目标点不变且末位姿离它 < 0.5 m，K1 的判据）："]
    L += [""] + md_table(["臂", "出生即碰撞", "前 4 步内碰撞", "目标点不前进（失败集中）"],
                         [[f"`{g}`", f"{len(res['facts'][g]['spawn_collision'])}（" + "; ".join(f"{s} r{r} seed{sd}" for s, r, sd in res["facts"][g]["spawn_collision"][:6]) + "）"
                           if res["facts"][g]["spawn_collision"] else "0", res["facts"][g]["collision_first4"], len(res["facts"][g]["stalled_failed"])] for g in arms])
    stuck_arms = [g for g in arms if res["facts"][g]["stuck_ending"] or g.startswith("rule2")]
    if stuck_arms:
        L += ["", "以 V2-stuck 收尾的失败集（规则顺序之外的缺口）：位置 = 最后一步起飞前位姿（x, y, 航向°）；到占据的间隙 = actions.jsonl 的 `clearance_m`（飞机位姿到原始占据栅格的距离），列末步与末 20 步最小值；"
                  "没到点的集才列，到点后以 V2-stuck 结束的集只记个数："]
        for g in stuck_arms:
            lst = res["facts"][g]["stuck_ending"]
            L += ["", f"`{g}`：失败集中以 V2-stuck 收尾 {len(lst)} 集；成功集中末步为 V2-stuck {res['facts'][g]['success_ending_stuck']} 集。"]
            if lst:
                L += [""] + md_table(["场景", "路线", "seed", "结局", "总步数", "位置 (x, y, 航向°)", "末步间隙 m", "末 20 步最小间隙 m"],
                                     [[x["scene"], x["route"], x["seed"], x["outcome"], x["steps"], x["last_pose"],
                                       "—" if x["final_clear"] is None else f"{x['final_clear']:.2f}",
                                       "—" if x["min_clear_last20"] is None else f"{x['min_clear_last20']:.2f}"] for x in lst[:60]])
                if len(lst) > 60:
                    L.append(f"（共 {len(lst)} 集，只列前 60）")
    L += ["", "逐集配对（SR(stop)，同场景同路线同 seed，精确 McNemar；b = A 成 B 败，c = A 败 B 成；Δ = SR(A) − SR(B)；末两列是路线层面的补充，未预注册）：", ""]
    prow = []
    for pc, rs in extra_pairs:
        d = "—" if not pc["pairs"] else f"{(pc['ka'] - pc['kb']) / pc['pairs']:+.3f}"
        prow.append([f"`{pc['a']}` vs `{pc['b']}`", pc["pairs"], pc["ka"], pc["kb"], pc["b_count"], pc["c_count"], d, f"{pc['p']:.4f}",
                     f"{rs['win']}/{rs['loss']}/{rs['tie']}（{rs['routes']} 条）", f"{rs['p']:.4f}"])
    L += md_table(["A vs B", "配对数", "A 到点", "B 到点", "b", "c", "Δ", "p", "路线 A 多 / A 少 / 平", "路线符号检验 p"], prow)
    L += ["", f"同 seed 各臂首步位姿的最大差（位置 m，航向 °/100 取大者）：{res['spawn_pair_max_diff']:.2e}（≈0 说明逐集配对的出生位姿一致）。"]
    return L


def render_routes(cls, res, data):
    cfg, arms, K = data["cfg"], res["arms"], res["k"]
    rows = []
    for s in cfg["scenes"]:
        for r in k1s.routes_of(cfg, s):
            x = data["route_info"].get((s, r))
            name = f"{s} r{r}" if x is None else f"{s} r{r}（{x['name']}，{x['astar_path_m']:.1f} m）"
            rows.append([name] + [f"{res['route_cells'][f'{g}|{s}|{r}']}/{K}" if f"{g}|{s}|{r}" in res["route_cells"] else "—" for g in arms])
    return [f"#### {cls}：分路线 SR(stop)（每格 = 到点集数 / {K}）", ""] + md_table(["场景 r"] + [f"`{g}`" for g in arms], rows)


def mark(ok):
    return "是" if ok else "否"


def k1_dev_check(out, k1_root):
    """开发路线里 K1 的 20 条：本次 dev 的 v1 臂（同参数、同 hold、同 seed）与 K1 dev 的 rocc-r0.25-d1.2 臂逐集对拍（只读；报告，不判决）。"""
    cfg = read_json(os.path.join(out, "dev", "config.json"))
    if not cfg or not k1_root:
        return None
    k1_dev = os.path.join(k1_root, "dev", "indoor")
    if not os.path.isdir(k1_dev):
        return None
    pairs = [(s, r) for s in cfg["scenes"] for r in k1s.routes_of(cfg, s)]
    new, missing = ckm.load_arm(os.path.join(out, "dev"), "v1", pairs)
    old = ckm.load_all(k1_dev, "rocc-r0.25-d1.2")
    c = ckm.compare_rows(new, old)
    return {"new_episodes": len(new), "missing_routes": len(missing), "k1_episodes": len(old), "matched": c["matched"], "equal": c["equal"],
            "diffs": [{"key": d["key"], "fields": d["fields"]} for d in c["diffs"]]}


def pair_block(index, arms, spec):
    """spec = [(a, b)]；两臂都在 arms（或是 K1 的键名）里才算；返回 [(paired2, route_sign)]。"""
    out = []
    for a, b in spec:
        if any(k[0] == a for k in index) and any(k[0] == b for k in index):
            out.append((paired2(index, a, b), route_sign(index, a, b)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--k1-root", default=None, help="K1 的输出目录（室外的 v1 与开发路线的 v1 复核用）")
    a = ap.parse_args(argv)
    out = os.path.abspath(a.out)
    classes = load_all_classes(out)
    if not classes:
        print(f"{out} 下没有 holdout / formal40 / outdoor 的 config.json", file=sys.stderr)
        return 2
    sel = read_json(os.path.join(out, "selected.json")) or {}
    res, idx = {}, {}
    for cls, data in classes.items():
        res[cls], idx[cls] = summarize_class(cls, data)
    complete_all = all(r["complete"] for r in res.values())
    summary = {"metric_version": METRIC_VERSION, "complete": complete_all, "selected": {k: sel.get(k) for k in ("variant", "depth_variant", "gain_over_v1", "gate_bypassed")},
               "classes": {}, "verdict": {}, "pairs": {}}
    lines = ["# 3DGS 上的规则 v2（rule-v2-3dgs，任务卡 K3）", "",
             f"- metric_version `{METRIC_VERSION}`；输出目录 `{out}`；选中变体 `{sel.get('variant')}`（depth 臂用 `{sel.get('depth_variant')}`），开发路线上比 v1 多到点 {sel.get('gain_over_v1')} 集"
             f"{'（闸门被冒烟绕过）' if sel.get('gate_bypassed') else ''}。",
             "- SR(stop) = 首次进入 waypoint 半径早于首次碰撞；碰撞率含到点之后的碰撞，撞在前 = 没到点就撞；超时 = 既没到点也没撞；逐集配对用精确 McNemar（双侧）；区间是 Wilson 95%。",
             "- 规则臂的列距离来自占据真值（occ）或渲染深度（depth），航向参照来自 oracle 路线，都是特权信息：这是规则层的上限，不是学生能力。", ""]
    verdict_parts = []
    lines += ["## 结论（机械套用任务卡的预注册数字，起跑前写在 run.sh 头里）", ""]
    if not complete_all:
        lines += ["**数据不全**：" + "；".join(f"{c} 缺 {len(r['missing_routes'])} 条路线 / 失败集 {len(r['failed'])}" for c, r in res.items() if not r["complete"])
                  + "。下面的判决只对已写 done.json 的路线成立，不作最终判决。", ""]
    # ---------------- 主终点：留出路线
    if "holdout" in res and MAIN in res["holdout"]["arms"] and V1 in res["holdout"]["arms"]:
        r, ix = res["holdout"], idx["holdout"]
        pc, a_ = paired2(ix, MAIN, V1), r["overall"][MAIN]
        j = judge_primary(pc, {"n_req": a_["n_req"], "k_coll": a_["k_coll"], "k_cf": a_["k_cf"]})
        summary["verdict"]["primary"] = j
        lines += ["### 主终点：留出路线，`rule2-occ+route` 对 `rule-occ+route`（v1）", ""]
        if j["verdict"] == "无数据":
            lines += ["没有配对的集。", ""]
        else:
            lines += md_table(["项", "实测", "阈值", "满足"],
                              [["SR(stop) 之差 Δ", f"{j['delta']:+.3f}（{pc['ka']}/{pc['pairs']} − {pc['kb']}/{pc['pairs']}；b = {pc['b_count']}，c = {pc['c_count']}）", "≥ +0.10", mark(j["delta_ge_pred"])],
                               ["精确 McNemar 双侧 p", f"{j['p']:.4f}", "< 0.05", mark(j["p_lt_alpha"])],
                               ["主臂碰撞率（整集任一步）", fmt_rate(a_["k_coll"], a_["n_req"]), "≤ 0.10", mark(j["coll_le_pred"])],
                               ["主臂撞在前", fmt_rate(a_["k_cf"], a_["n_req"]), "≤ 0.08", mark(j["cf_le_pred"])],
                               ["证伪条件一：Δ 低于 0.05", f"{j['delta']:+.3f}", "< +0.05", mark(j["delta_lt_fals"])],
                               ["证伪条件二：主臂碰撞率高于 0.15", f"{j['coll']:.3f}", "> 0.15", mark(j["coll_gt_fals"])]])
            lines += ["", f"**判决：{j['verdict']}**" + ("" if complete_all else "（数据不全，仅供参考）"), "",
                      f"两臂 SR(stop)：v1 {fmt_rate(r['overall'][V1]['k_stop'], r['overall'][V1]['n_req'])}；主臂 {fmt_rate(a_['k_stop'], a_['n_req'])}。", ""]
            verdict_parts.append(f"主终点{j['verdict']}（Δ={j['delta']:+.3f}，p={j['p']:.4f}，碰撞{j['coll']:.3f}，撞在前{j['cf']:.3f}）")
    # ---------------- 次终点：正式 40 条
    if "formal40" in res and MAIN in res["formal40"]["arms"] and V1 in res["formal40"]["arms"]:
        r, ix = res["formal40"], idx["formal40"]
        pc, a_ = paired2(ix, MAIN, V1), r["overall"][MAIN]
        j = judge_secondary(pc, {"n_req": a_["n_req"], "k_stop": a_["k_stop"]})
        summary["verdict"]["secondary"] = j
        lines += ["### 次终点：正式 40 条（K1 的同一批），同样的比较，只报告", ""]
        if j["verdict"] == "无数据":
            lines += ["没有配对的集。", ""]
        else:
            lines += [f"Δ = {j['delta']:+.3f}（{pc['ka']}/{pc['pairs']} − {pc['kb']}/{pc['pairs']}；b = {pc['b_count']}，c = {pc['c_count']}），p = {j['p']:.4f}；"
                      f"主臂 SR(stop) {fmt_rate(a_['k_stop'], a_['n_req'])}，**{'到' if j['sr_ge_080'] else '没到'} 0.80**（K1 没达到的那条预测）。", ""]
            verdict_parts.append(f"次终点Δ={j['delta']:+.3f}（p={j['p']:.4f}），主臂SR {j['sr']:.3f}{'到' if j['sr_ge_080'] else '没到'}0.80")
    # ---------------- 室外：不劣；v1 取 K1 的结果按键配对
    k1_name = "K1:" + V1
    if "outdoor" in res and MAIN in res["outdoor"]["arms"]:
        r, ix = res["outdoor"], idx["outdoor"]
        a_ = r["overall"][MAIN]
        k1_rows = ckm.load_all(os.path.join(a.k1_root, "formal", "outdoor"), V1) if a.k1_root else {}
        for (s, rt, seed), m in k1_rows.items():
            ix[(k1_name, s, rt, seed)] = m
        main_keys = [(x["scene"], x["route"], x["seed"]) for x in classes["outdoor"]["rows"] if x["arm"] == MAIN]
        v1_missing = sum(1 for k in main_keys if k not in k1_rows)
        pc_v1 = paired2(ix, k1_name, MAIN)
        j = judge_outdoor({"n_req": a_["n_req"], "k_stop": a_["k_stop"], "k_coll": a_["k_coll"]}, pc_v1, v1_missing)
        summary["verdict"]["outdoor"] = j
        lines += ["### 室外（不劣）：`rule2-occ+route`，v1 取 K1 的 `rule-occ+route` 结果按（场景, 路线, seed）配对", ""]
        if j["verdict"] in ("无数据", "K1 缺键，不判决"):
            lines += [f"{j['verdict']}（K1 缺 {v1_missing} 个键；`--k1-root` {'未给' if not a.k1_root else a.k1_root}）。", ""]
        else:
            lines += md_table(["项", "实测", "阈值", "满足"],
                              [["主臂 SR(stop)", fmt_rate(a_["k_stop"], a_["n_req"]), "≥ 0.93", mark(j["sr_ge"])],
                               ["（v1 成而 v2 败）− （v2 成而 v1 败）", f"{j['b_v1_only']} − {j['c_v2_only']} = {j['b_minus_c']}（{j['pairs']} 对；p = {j['p']:.4f}）", "≤ 5", mark(j["bc_le"])],
                               ["主臂碰撞率（整集任一步）", fmt_rate(a_["k_coll"], a_["n_req"]), "≤ 0.10", mark(j["coll_le"])]])
            lines += ["", f"**判决：{j['verdict']}**" + ("" if complete_all else "（数据不全，仅供参考）"), ""]
            verdict_parts.append(f"室外{j['verdict']}（SR {j['sr']:.3f}，b−c={j['b_minus_c']}，碰撞{j['coll']:.3f}）")
    # ---------------- depth 臂：只报告
    dep = []
    for cls in ("holdout", "formal40", "outdoor"):
        if cls in res and DEPTH in res[cls]["arms"]:
            d_ = res[cls]["overall"][DEPTH]
            dep.append(f"{cls}：SR(stop) {fmt_rate(d_['k_stop'], d_['n_req'])}，碰撞 {fmt_rate(d_['k_coll'], d_['n_req'])}，撞在前 {fmt_rate(d_['k_cf'], d_['n_req'])}")
    if dep:
        lines += ["### depth 臂（只报告，无阈值）", ""] + [f"- {x}" for x in dep] + [""]
    chk = read_json(os.path.join(out, "k1_match.json"))
    if chk:
        lines += [f"F0（正式 40 条上 v1 重跑与 K1 逐集对拍）：本次 {chk['n_new']} 集，同键 {chk['matched']}，一致 {chk['equal']}，不一致 {len(chk['diffs'])}。", ""]
        summary["f0"] = {k: chk[k] for k in ("n_new", "matched", "equal")} | {"diffs": len(chk["diffs"])}
    dcheck = k1_dev_check(out, a.k1_root)
    if dcheck:
        summary["dev_k1_check"] = dcheck
        lines += [f"开发路线上 K1 的 20 条：本次 v1 臂与 K1 的 `rocc-r0.25-d1.2` 臂同键 {dcheck['matched']} 集，一致 {dcheck['equal']}，不一致 {len(dcheck['diffs'])}"
                  f"（本次 v1 共 {dcheck['new_episodes']} 集含新取路线；只对拍同键的）。", ""]
    sel_text = ""
    if os.path.exists(os.path.join(out, "dev_selection.md")):
        with open(os.path.join(out, "dev_selection.md"), encoding="utf-8") as f:
            sel_text = f.read()
    # ---------------- 各类的明细
    titles = {"holdout": "留出路线（主终点）", "formal40": "正式 40 条（次终点）", "outdoor": "室外 24 条（不劣）"}
    specs = {"holdout": [(MAIN, V1), (DEPTH, V1), (MAIN, FWD), (V1, FWD), (DEPTH, MAIN)], "formal40": [(MAIN, V1), (DEPTH, V1), (DEPTH, MAIN)],
             "outdoor": [(MAIN, k1_name), (DEPTH, k1_name), (DEPTH, MAIN)]}
    lines += ["## 各类明细", ""]
    for cls in ("holdout", "formal40", "outdoor"):
        if cls not in res:
            continue
        ps = pair_block(idx[cls], res[cls]["arms"], specs[cls])
        summary["pairs"][cls] = [{**p_, "route": rs} for p_, rs in ps]
        lines += render_class(cls, titles[cls], res[cls], classes[cls], ps) + [""]
        summary["classes"][cls] = {k: v for k, v in res[cls].items()}
    lines += ["## 开发路线与变体选取（run.sh 自动执行；下表是 dev_selection.md 原文）", "", sel_text or "（没有 dev_selection.md）", "", "## 分路线明细", ""]
    for cls in ("holdout", "formal40", "outdoor"):
        if cls in res:
            lines += render_routes(cls, res[cls], classes[cls]) + [""]
    lines += ["来源：本目录 `{holdout,formal40,outdoor}/<臂>/<场景>/r<N>/ep_*/{metrics.json,rule.jsonl,actions.jsonl}`；输入 sha 见 `inputs.txt`。"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "verdict.txt"), "w", encoding="utf-8") as f:
        f.write(("" if complete_all else "数据不全；") + "；".join(verdict_parts) + "\n")
    print("\n".join(lines[:60]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
