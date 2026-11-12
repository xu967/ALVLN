#!/usr/bin/env python3
"""卡 E2E-1 评测汇总：四个臂（A0 img/demo、A1 code/demo、A2 code/dagger、A3 img/dagger）× 三个训练种子，每个评测臂 20 个冻结 seed，
柱阵 K=1，不带 --heading-guard，--action-latency 取 L*（第二轮 B 组过门的 0.20）。口径：metric_version sim09.v2.coast1s.corridor3m，成功 = 无碰撞、过成功线、|y| ≤ 3 m。

用法：
  summarize_e2e.py OUT [--note 文字]      读 OUT 下 12 个评测臂目录（<臂名>-s<训练种子>，含重跑的 -retry1），写 OUT/summary.md、verdicts.json、episodes.csv
  summarize_e2e.py --late-seeds ARM_DIR   打印臂目录里含迟到步的 seed（逗号分隔）；run.sh 的重跑（条款 R6）用
汇总随时可跑；缺的臂标「未完成」，三个比较在 12 个臂都评完（含重跑结果）之前只给进度数，不判。

判定（逐字对应卡 E2E-1 的预注册，阈值写在下面的常量里，起跑后不改）。每个比较 60 个配对集 = 三个训练种子 × 20 个冻结 seed，
配对键 = （训练种子，评测 seed）：同一个训练种子下，两个臂在同一个评测 seed 的成败配对；精确 McNemar（双侧）。成功数 = 配对集里成功的集数。
  M1  A2 对 A3：预测 成功数差 A2−A3 ≥ 15 且 p < 0.01；证伪 差 < 6；其余 = 介于两者之间，不判。
  M2  A1 对 A0：预测 A1 的越界集数 ≤ A0 的一半（2·越界(A1) ≤ 越界(A0)）且 成功(A1) ≥ 成功(A0)；证伪 越界(A0) − 越界(A1) < 3。
      两条同时成立（A0 的越界集数 ≤ 5 时会出现）= 冲突，如实写出，不判；都不成立 = 介于两者之间，不判。
  M3  A2 对 A0：预测 成功数差 A2−A0 ≥ 6；证伪 A2 不高于 A0（差 ≤ 0）；其余不判。
  只报告：A1、A2 里成功数高的那个是否达到卡 W2-2 的判据（成功 ≥ 48/60、越界 ≤ 3/60）；带护栏的参照 = 第二轮 K=1 三个臂（exp-action-latency-r2-20261111/acc-L0.20-k1-s{0,1,2}）。
计数口径：越界集数 = 失败分类为「越界」的集数（lc.block_outcomes 的第一个坏事件，同一步撞柱与越界按撞柱记；与第二轮、长课程汇总同口径）；
  eval.json 的 corridor_violation 标记（过程中任一步 |y| > 3 m，含后来撞柱的集）另列，M2 另给一行按该标记计的判定，只作稳健性对照。
含迟到步的集（基础设施失败）按条款 R6 重跑一次（<臂>-retry1），汇总以重跑结果为准；重跑仍迟到、或没有重跑结果，按没成功算，首次记录另列。
"""
import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "2026-11-10-action-latency"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim", "tools"))
import e2e_arms as ea  # noqa: E402
import summarize_r2 as r2  # noqa: E402  第二轮汇总：逐集读取、重跑合并、时序统计、cgroup 记录（它再引入第一轮汇总与长课程汇总）
import nav_cond  # noqa: E402

s1 = r2.s1
lc = s1.lc
D = s1.D
OUT = HERE

EPISODES = 20
FROZEN_SEEDS = list(s1.FROZEN_SEEDS)
N_PAIRS = len(ea.SEEDS_TRAIN) * EPISODES
HALF_M = 3.0
GOAL_X_DEFAULT = 6.0
SAFE, FWD, YAW_L, YAW_R = 0, 1, 2, 3
RETRY = r2.RETRY_SUFFIX

# ---------------------------------------------------------------- 预注册阈值（卡 E2E-1，起跑后不改）
M1_PRED_DIFF, M1_PRED_P, M1_FALS_BELOW = 15, 0.01, 6
M2_PRED_FACTOR, M2_FALS_DROP_BELOW = 2, 3
M3_PRED_DIFF = 6
W22_SUCCESS_MIN, W22_CORRIDOR_MAX = 48, 3
REF_GUARD = [os.path.join(D, "exp-action-latency-r2-20261111", f"acc-L0.20-k1-s{s}") for s in ea.SEEDS_TRAIN]

PRED, FALS, MID, CONFLICT, UNDONE = "预测成立", "证伪成立", "介于预测与证伪之间，不判", "预测与证伪条款同时成立（冲突），不判", "未完成，不判"
CLASS_NAME = {"success": "成功", "collision": "撞柱", "corridor": "越界", "not_reached": "未到线", "aborted": "采集中止",
              "late": "迟到（基础设施失败）", "other": "未归类", "missing": "无记录"}
FAIL_CLASSES = ("collision", "corridor", "not_reached", "aborted", "late", "other")


# ---------------------------------------------------------------- 纯函数：配对、判定、分类、航向码统计
def pair_counts(x, y):
    """x、y = {键: 是否成功}（只取共同键）→ n、b（x 成 y 败）、c（x 败 y 成）、both、neither。"""
    keys = sorted(set(x) & set(y))
    b = sum(1 for k in keys if x[k] and not y[k])
    c = sum(1 for k in keys if y[k] and not x[k])
    both = sum(1 for k in keys if x[k] and y[k])
    return {"n": len(keys), "b": b, "c": c, "both": both, "neither": len(keys) - b - c - both,
            "succ_x": sum(1 for k in keys if x[k]), "succ_y": sum(1 for k in keys if y[k]), "p": lc.mcnemar_exact(b, c)}


def verdict_m1(diff, p):
    """M1：diff = 成功数(A2) − 成功数(A3)。"""
    if diff >= M1_PRED_DIFF and p < M1_PRED_P:
        return PRED
    if diff < M1_FALS_BELOW:
        return FALS
    return MID


def verdict_m2(corr_a0, corr_a1, succ_a0, succ_a1):
    """M2：返回 (判定, 预测条款是否成立, 证伪条款是否成立)。"""
    pred = M2_PRED_FACTOR * corr_a1 <= corr_a0 and succ_a1 >= succ_a0
    fals = corr_a0 - corr_a1 < M2_FALS_DROP_BELOW
    if pred and fals:
        return CONFLICT, pred, fals
    if pred:
        return PRED, pred, fals
    if fals:
        return FALS, pred, fals
    return MID, pred, fals


def verdict_m3(diff):
    """M3：diff = 成功数(A2) − 成功数(A0)。"""
    if diff >= M3_PRED_DIFF:
        return PRED
    if diff <= 0:
        return FALS
    return MID


def w22_status(success, corridor):
    """只报告：是否达到卡 W2-2 的判据（不带护栏：成功 ≥ 48/60、越界 ≤ 3/60）。"""
    return success >= W22_SUCCESS_MIN and corridor <= W22_CORRIDOR_MAX


def classify_row(success, aborted, actions, oracle, goal, half=HALF_M):
    """一集的结局类别 → (类别, 与轨迹判据是否一致)。success 取 eval.json 的成功标记，失败的集按 lc.block_outcomes 的第一个坏事件分类。"""
    fail = lc.block_outcomes(actions, oracle, [goal], half, aborted=aborted)["fail"]
    if success:
        return "success", fail is None
    if fail is None:
        return "other", False
    return fail[1], True


def code_stats(executed_lists):
    """若干集的已执行动作序列 → 航向码相关统计。码由已执行动作重算（nav_cond.heading_codes，H = 0）。
    corrective = 码 ≠ 0 的步里取了回转动作的次数（码 2 = 净左转超出 → YAW_RIGHT；码 3 = 净右转超出 → YAW_LEFT）。"""
    out = {"steps": 0, "action_counts": [0, 0, 0, 0], "code_steps": 0, "corrective": 0, "code0_steps": 0, "turn_in_code0": 0,
           "episodes": 0, "episodes_with_code": 0, "max_abs_n": 0}
    for ex in executed_lists:
        codes = nav_cond.heading_codes(ex)
        out["episodes"] += 1
        had, n = False, 0
        for a, c in zip(ex, codes):
            out["steps"] += 1
            if a in (SAFE, FWD, YAW_L, YAW_R):
                out["action_counts"][a] += 1
            if c == 0:
                out["code0_steps"] += 1
                out["turn_in_code0"] += a in (YAW_L, YAW_R)
            else:
                out["code_steps"] += 1
                had = True
                out["corrective"] += (c == 2 and a == YAW_R) or (c == 3 and a == YAW_L)
            n += (a == YAW_L) - (a == YAW_R)
            out["max_abs_n"] = max(out["max_abs_n"], abs(n))
        out["episodes_with_code"] += had
    return out


def merge_code_stats(parts):
    out = {"steps": 0, "action_counts": [0, 0, 0, 0], "code_steps": 0, "corrective": 0, "code0_steps": 0, "turn_in_code0": 0,
           "episodes": 0, "episodes_with_code": 0, "max_abs_n": 0}
    for p in parts:
        for k in ("steps", "code_steps", "corrective", "code0_steps", "turn_in_code0", "episodes", "episodes_with_code"):
            out[k] += p[k]
        out["action_counts"] = [x + y for x, y in zip(out["action_counts"], p["action_counts"])]
        out["max_abs_n"] = max(out["max_abs_n"], p["max_abs_n"])
    return out


def cond_consistency(rows):
    """采集行里记的 cond_code 与按已执行动作重算的码逐步比对 → (有 cond_code 的行数, 不一致的行数, 缺 cond_code 的行数)。没有任何 cond_code 键返回 (0, 0, len(rows))。"""
    want = nav_cond.heading_codes([r.get("action_executed") for r in rows])
    have = [r.get("cond_code") for r in rows]
    with_code = sum(1 for h in have if h is not None)
    bad = sum(1 for h, w in zip(have, want) if h is not None and h != w)
    return with_code, bad, len(rows) - with_code


def fmt_p(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def frac_text(k, n):
    if not n:
        return "-"
    lo, hi = lc.wilson(k, n)
    return f"{k}/{n}（{k / n:.0%}，Wilson 95% [{lo:.2f}, {hi:.2f}]）"


# ---------------------------------------------------------------- 读取
def eval_entries(edir, cache):
    """eval.json 的 per_episode → {seed: 条目}；没有文件 = {}。"""
    if edir not in cache:
        path = os.path.join(edir, "eval.json")
        cache[edir] = {int(x["seed"]): x for x in lc.read_json(path).get("per_episode", [])} if os.path.exists(path) else {}
    return cache[edir]


def load_arm_eps(edir, requested=EPISODES):
    """一个评测臂（含重跑合并）→ {"edir", "counts", "merged", "eps": {seed: {...}}}。eps 里的字段：
    final（first|retry|none）、first_late、retry、success、klass、consistent、steps、executed（已执行动作序列）、flag_corridor、flag_collision、cond（(有码行数, 不一致, 缺码)）。"""
    counts = r2.arm_counts(edir, requested)
    merged = r2.merged_rows(edir, requested)
    eps, cache = {}, {}
    if merged:
        succ = r2.success_map(merged, 1)
        first_ok = r2.first_run_map(merged, 1)
        for seed, item in merged.items():
            final = item["final"]
            rec = {"final": final, "first_late": item["first_late"], "retry": item["retry"], "success": bool(succ[seed]),
                   "klass": "late" if final == "none" else None, "consistent": True, "steps": None, "executed": [],
                   "flag_corridor": False, "flag_collision": False, "cond": (0, 0, 0), "first_success": bool(first_ok[seed])}
            if final != "none":
                used = edir if final == "first" else edir + RETRY
                ep = os.path.join(used, "episodes", f"seed_{seed}")
                actions = lc.read_jsonl(os.path.join(ep, "actions.jsonl"))
                oracle = lc.read_jsonl(os.path.join(ep, "oracle.jsonl"))
                man_path = os.path.join(ep, "episode_manifest.json")
                goal = (lc.read_json(man_path).get("goal_line_x") if os.path.exists(man_path) else None) or GOAL_X_DEFAULT
                row = item["row"]
                rec["klass"], rec["consistent"] = classify_row(bool(row.get("success")), bool(row.get("aborted")), actions, oracle, float(goal))
                rec["steps"] = len(actions)
                rec["executed"] = [a.get("action_executed") for a in actions]
                rec["cond"] = cond_consistency(actions) if actions else (0, 0, 0)
                entry = eval_entries(used, cache).get(seed, {})
                rec["flag_corridor"] = bool(entry.get("corridor_violation"))
                rec["flag_collision"] = bool(entry.get("collision"))
            eps[seed] = rec
    return {"edir": edir, "counts": counts, "merged": merged, "eps": eps}


def arm_done(arm):
    """评测臂是否完整：首次运行完整、20 个冻结 seed 都有记录、含迟到步的集都已有重跑结果（干净、仍迟到或重跑中止都算有结果）。"""
    if not arm["counts"]["complete"] or set(arm["eps"]) != set(FROZEN_SEEDS):
        return False
    return all(rec["retry"] != "missing" for rec in arm["eps"].values() if rec["first_late"])


def load_all(out):
    return {(a, s): load_arm_eps(ea.evdir(out, a, s)) for a in ea.ARMS for s in ea.SEEDS_TRAIN}


def success_keys(data, a):
    """{(训练种子, 评测 seed): 是否成功}，只含已有记录的集。"""
    return {(s, seed): rec["success"] for s in ea.SEEDS_TRAIN for seed, rec in data[(a, s)]["eps"].items()}


def class_counts(data, a, seeds=ea.SEEDS_TRAIN):
    cnt = {k: 0 for k in CLASS_NAME}
    for s in seeds:
        for rec in data[(a, s)]["eps"].values():
            cnt[rec["klass"] or "missing"] += 1
    cnt["n"] = sum(cnt.values())
    return cnt


def flag_counts(data, a):
    recs = [rec for s in ea.SEEDS_TRAIN for rec in data[(a, s)]["eps"].values() if rec["klass"] not in (None, "late")]
    return {"corridor_flag": sum(rec["flag_corridor"] for rec in recs), "collision_flag": sum(rec["flag_collision"] for rec in recs), "n": len(recs)}


def guard_reference():
    """带护栏的参照：第二轮 K=1 三个臂的成功数（读 exp-action-latency-r2-20261111，只读）；缺文件 = None。"""
    total, n = 0, 0
    per = []
    for d in REF_GUARD:
        merged = r2.merged_rows(d, EPISODES)
        if merged is None:
            return None
        succ = r2.success_map(merged, 1)
        per.append(sum(succ.values()))
        total += per[-1]
        n += len(succ)
    return {"success": total, "n": n, "per_seed": per}


# ---------------------------------------------------------------- 三个比较
def compute_verdicts(data, done):
    """-> dict：m1、m2、m3（各自的计数、p、判定）、w22；done = 12 个评测臂是否都完整。未完成时判定写 UNDONE，计数照给（只含已有的配对）。"""
    keys = {a: success_keys(data, a) for a in ea.ARMS}
    cls = {a: class_counts(data, a) for a in ea.ARMS}
    flg = {a: flag_counts(data, a) for a in ea.ARMS}
    out = {"complete": done, "pairs": {}}

    def pair(label, x, y):
        pc = pair_counts(keys[x], keys[y])
        pc["full"] = pc["n"] == N_PAIRS
        out["pairs"][label] = pc
        return pc

    p1 = pair("M1", "a2", "a3")
    diff1 = p1["succ_x"] - p1["succ_y"]
    out["m1"] = {"a2": p1["succ_x"], "a3": p1["succ_y"], "diff": diff1, "p": p1["p"], "b": p1["b"], "c": p1["c"], "n": p1["n"],
                 "verdict": verdict_m1(diff1, p1["p"]) if done and p1["full"] else UNDONE}
    p2 = pair("M2", "a1", "a0")
    v2, pred2, fals2 = verdict_m2(cls["a0"]["corridor"], cls["a1"]["corridor"], p2["succ_y"], p2["succ_x"])
    v2f = verdict_m2(flg["a0"]["corridor_flag"], flg["a1"]["corridor_flag"], p2["succ_y"], p2["succ_x"])[0]
    out["m2"] = {"a1_success": p2["succ_x"], "a0_success": p2["succ_y"], "a1_corridor": cls["a1"]["corridor"], "a0_corridor": cls["a0"]["corridor"],
                 "pred_clause": pred2, "fals_clause": fals2, "p": p2["p"], "b": p2["b"], "c": p2["c"], "n": p2["n"],
                 "verdict": v2 if done and p2["full"] else UNDONE,
                 "flag_corridor": {"a0": flg["a0"]["corridor_flag"], "a1": flg["a1"]["corridor_flag"], "verdict": v2f if done and p2["full"] else UNDONE}}
    p3 = pair("M3", "a2", "a0")
    diff3 = p3["succ_x"] - p3["succ_y"]
    out["m3"] = {"a2": p3["succ_x"], "a0": p3["succ_y"], "diff": diff3, "p": p3["p"], "b": p3["b"], "c": p3["c"], "n": p3["n"],
                 "verdict": verdict_m3(diff3) if done and p3["full"] else UNDONE}
    # 只报告：成功数高的航向码臂是否达到 W2-2 的判据
    succ = {a: sum(keys[a].values()) for a in ea.ARMS}
    best = [a for a in ("a1", "a2") if succ[a] == max(succ["a1"], succ["a2"])]
    out["w22"] = [{"arm": a, "success": succ[a], "corridor": cls[a]["corridor"], "corridor_flag": flg[a]["corridor_flag"], "n": cls[a]["n"],
                   "reached": w22_status(succ[a], cls[a]["corridor"]), "reached_by_flag": w22_status(succ[a], flg[a]["corridor_flag"])} for a in best]
    out["guard_reference"] = guard_reference()
    out["success"] = succ
    out["classes"] = cls
    out["flags"] = flg
    return out


# ---------------------------------------------------------------- 文字
def verdict_lines(v, done):
    m1, m2, m3 = v["m1"], v["m2"], v["m3"]
    state = "" if done else "（12 个评测臂尚未全部评完，下面只是进度数，不判）"
    lines = [f"- **M1 A2 对 A3**（同一份聚合数据，有无航向码）{state}：成功 A2 {m1['a2']}/{m1['n']}，A3 {m1['a3']}/{m1['n']}，差 {m1['diff']:+d}；b={m1['b']}（A2 成 A3 败），c={m1['c']}，精确 McNemar p={fmt_p(m1['p'])}。"
             f"预测 差 ≥ {M1_PRED_DIFF} 且 p < {M1_PRED_P}；证伪 差 < {M1_FALS_BELOW}。→ **{m1['verdict']}**",
             f"- **M2 A1 对 A0**（同一份示范数据，有无航向码）：成功 A1 {m2['a1_success']}/{m2['n']}，A0 {m2['a0_success']}/{m2['n']}；越界 A1 {m2['a1_corridor']}，A0 {m2['a0_corridor']}；"
             f"b={m2['b']}（A1 成 A0 败），c={m2['c']}，p={fmt_p(m2['p'])}。预测 2·越界(A1) ≤ 越界(A0) 且 成功(A1) ≥ 成功(A0)：{'成立' if m2['pred_clause'] else '不成立'}；"
             f"证伪 越界(A0) − 越界(A1) < {M2_FALS_DROP_BELOW}：{'成立' if m2['fals_clause'] else '不成立'}。→ **{m2['verdict']}**",
             f"  - 稳健性对照（按 eval.json 的 corridor_violation 标记计越界）：A0 {m2['flag_corridor']['a0']}，A1 {m2['flag_corridor']['a1']}，判定 {m2['flag_corridor']['verdict']}。",
             f"- **M3 A2 对 A0**：成功 A2 {m3['a2']}/{m3['n']}，A0 {m3['a0']}/{m3['n']}，差 {m3['diff']:+d}；b={m3['b']}（A2 成 A0 败），c={m3['c']}，p={fmt_p(m3['p'])}。"
             f"预测 差 ≥ {M3_PRED_DIFF}；证伪 A2 不高于 A0（差 ≤ 0）。→ **{m3['verdict']}**"]
    ref = v["guard_reference"]
    ref_text = "未核实（读不到第二轮的三个 K=1 臂）" if ref is None else f"{ref['success']}/{ref['n']}（三个训练种子 {'/'.join(map(str, ref['per_seed']))}，每个 /{EPISODES}）"
    for w in v["w22"]:
        lines.append(f"- 只报告（不设判决）：航向码臂里成功数高的 {ea.LABEL[w['arm']]}：成功 {w['success']}/{w['n']}，越界 {w['corridor']}/{w['n']}（按标记计 {w['corridor_flag']}）；"
                     f"卡 W2-2 的判据（不带护栏成功 ≥ {W22_SUCCESS_MIN}/60、越界 ≤ {W22_CORRIDOR_MAX}/60）：{'达到' if w['reached'] else '未达到'}"
                     f"（按标记计越界：{'达到' if w['reached_by_flag'] else '未达到'}）。带护栏的参照（第二轮 K=1 三个臂）：{ref_text}。")
    return lines


def arm_table(data, v):
    cls, flg = v["classes"], v["flags"]
    head = ["| 臂 | 成功 | 撞柱 | 越界 | 未到线 | 采集中止 | 迟到（基础设施失败） | 未归类 | 越界标记（过程中任一步横向偏离超过 3 m） | 碰撞标记 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    for a in ea.ARMS:
        c = cls[a]
        rows.append(f"| {ea.LABEL[a]} | {frac_text(c['success'], c['n'])} | {c['collision']} | {c['corridor']} | {c['not_reached']} | {c['aborted']} | {c['late']} | {c['other']} | "
                    f"{flg[a]['corridor_flag']} | {flg[a]['collision_flag']} |")
    return head + rows


def two_by_two(v):
    s, cls = v["success"], v["classes"]
    n = lambda a: cls[a]["n"]  # noqa: E731
    return ["| 数据 \\ 输入 | img（图像） | code（图像 + 航向码） |", "| --- | --- | --- |",
            f"| demo | A0：{s['a0']}/{n('a0')} | A1：{s['a1']}/{n('a1')} |",
            f"| dagger | A3：{s['a3']}/{n('a3')} | A2：{s['a2']}/{n('a2')} |"]


def per_seed_table(data):
    head = ["每格 = 成功 / 撞柱 / 越界 / 未到线 + （采集中止 + 迟到 + 未归类），分母 20。", "",
            "| 臂 | 训练种子 0 | 训练种子 1 | 训练种子 2 |", "| --- | --- | --- | --- |"]
    rows = []
    for a in ea.ARMS:
        cells = []
        for s in ea.SEEDS_TRAIN:
            c = class_counts(data, a, (s,))
            if not c["n"]:
                cells.append("-")
                continue
            other = c["aborted"] + c["late"] + c["other"]
            part = "" if c["n"] == EPISODES else f"（只有 {c['n']} 集）"
            cells.append(f"{c['success']} / {c['collision']} / {c['corridor']} / {c['not_reached']} + {other}{part}")
        rows.append(f"| {ea.LABEL[a]} | " + " | ".join(cells) + " |")
    return head + rows


def action_table(data):
    head = ["| 臂 | 步数 | SAFE_NEUTRAL | FORWARD_SLOW | YAW_LEFT | YAW_RIGHT | 重算码 ≠ 0 的步 | 其中取回转动作 | 码 = 0 的步里转向占比 | 出现过码 ≠ 0 的集 | 集内 \\|净转向数\\| 最大值 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = []
    stats = {}
    for a in ea.ARMS:
        parts = [code_stats([rec["executed"] for rec in data[(a, s)]["eps"].values() if rec["executed"]]) for s in ea.SEEDS_TRAIN]
        st = stats[a] = merge_code_stats(parts)
        n = st["steps"] or 1
        fr = [100.0 * x / n for x in st["action_counts"]]
        corr = "-" if not st["code_steps"] else f"{st['corrective']}/{st['code_steps']}={100.0 * st['corrective'] / st['code_steps']:.1f}%"
        t0 = "-" if not st["code0_steps"] else f"{100.0 * st['turn_in_code0'] / st['code0_steps']:.1f}%"
        rows.append(f"| {ea.LABEL[a]} | {st['steps']} | {fr[0]:.1f}% | {fr[1]:.1f}% | {fr[2]:.1f}% | {fr[3]:.1f}% | {st['code_steps']} | {corr} | {t0} | "
                    f"{st['episodes_with_code']}/{st['episodes']} | {st['max_abs_n']} |")
    return head + rows, stats


def cond_table(data):
    """A1、A2：采集行里的 cond_code 与按已执行动作重算的码是否逐步相同；A0、A3 不该有 cond_code。"""
    head = ["| 臂 | 带 cond_code 的行 | 与重算不一致 | 缺 cond_code 的行 |", "| --- | --- | --- | --- |"]
    rows, bad_total = [], 0
    for a in ea.ARMS:
        w = b = m = 0
        for s in ea.SEEDS_TRAIN:
            for rec in data[(a, s)]["eps"].values():
                w, b, m = w + rec["cond"][0], b + rec["cond"][1], m + rec["cond"][2]
        expect_code = ea.IS_CODE[a]
        flag = ""
        if expect_code and (b or m):
            flag = " ⚠"
        if not expect_code and w:
            flag = " ⚠（图像臂不该有 cond_code）"
        bad_total += bool(flag)
        rows.append(f"| {ea.LABEL[a]} | {w} | {b}{flag} | {m if expect_code else '-'} |")
    return head + rows, bad_total


def cg_row(label, edir):
    """一个评测臂的 cgroup 区间一行：墙钟、平均核数、限流次数增量、被限流时长及其占墙钟的比例、迟到步 / 总步（公共规则 §4，2026-11-12 01:40 起）。r2.OUT 须已指向输出目录。"""
    cg = r2.read_cg(os.path.basename(edir))
    t = s1.timing_stats([edir])
    late = "-" if t is None or t["late_steps"] is None else f"{t['late_steps']}/{t['steps']}"
    if cg is None:
        return f"| {label} | - | - | - | - | - | {late} |"
    share = cg["throttled_s"] / cg["wall_s"] if cg["wall_s"] > 0 else float("nan")
    flag = " ⚠" if cg["throttled"] > 0 else ""
    return f"| {label} | {cg['wall_s']:.0f} | {cg['avg_cores']:.1f} | {cg['throttled']}{flag} | {cg['throttled_s']:.2f} | {100 * share:.2f}% | {late} |"


def timing_tables(data, out):
    r2.OUT = out
    lines = list(r2.TIMING_HEAD)
    cg_lines = ["| 评测臂 | 墙钟 s | 平均核数 | 限流次数增量 | 被限流时长 s | 被限流时长占墙钟 | 迟到步 / 总步 |", "|---|---|---|---|---|---|---|"]
    for a in ea.ARMS:
        for s in ea.SEEDS_TRAIN:
            edir = data[(a, s)]["edir"]
            name = os.path.basename(edir)
            lines.append(r2.timing_row(f"{ea.LABEL[a]} s{s}", edir, EPISODES))
            cg_lines.append(cg_row(name, edir))
            if os.path.exists(os.path.join(edir + RETRY, "eval.json")):
                lines.append(r2.timing_row(f"{name}{RETRY}", edir + RETRY))
                cg_lines.append(cg_row(f"{name}{RETRY}", edir + RETRY))
    return lines + ["", "每个评测臂的 cgroup 区间（臂开始到结束，整个容器的 CPU；两条车道并行时区间重叠；限流时长 = cpu.stat 的 throttled_usec 增量；⚠ = 限流次数增量大于 0，不自动作废）：", ""] + cg_lines


def retry_table(data):
    head = ["| 评测臂 | 首次含迟到步的 seed | 重跑结果（seed：结果） | 首次记录里的成功数 | 采用后的成功数 |", "| --- | --- | --- | --- | --- |"]
    rows = []
    for a in ea.ARMS:
        for s in ea.SEEDS_TRAIN:
            arm = data[(a, s)]
            late = sorted(seed for seed, rec in arm["eps"].items() if rec["first_late"])
            first_ok = sum(1 for rec in arm["eps"].values() if rec["first_success"])
            final_ok = sum(1 for rec in arm["eps"].values() if rec["success"])
            res = "；".join(f"{seed}：{ {'ok': '干净', 'late': '仍迟到', 'failed': '重跑中止', 'missing': '没有重跑结果'}[arm['eps'][seed]['retry']] }" for seed in late)
            rows.append(f"| {os.path.basename(arm['edir'])} | {','.join(map(str, late)) or '-'} | {res or '-'} | {first_ok} | {final_ok} |")
    return head + rows


def offline_lines(out):
    path = os.path.join(out, "offline", "offline.json")
    if not os.path.exists(path):
        return ["（没有 offline/offline.json，离线部分未跑完）"]
    with open(path, encoding="utf-8") as f:
        off = json.load(f)
    lines = ["| 臂 | 种子 | 转向预测帧里码非 0 的帧：对调 2↔3 后方向翻转 | 码非 0 帧预测有变化 | 码恒置 0：val macro-F1 变化 |", "| --- | --- | --- | --- | --- |"]
    for a in ("a1", "a2"):
        for s in ea.SEEDS_TRAIN:
            p = off.get("probes", {}).get(f"{a}-s{s}-val")
            if not p:
                lines.append(f"| {ea.LABEL[a]} | {s} | 未完成 | | |")
                continue
            sw, z = p["swap"], p["zero"]

            def pct(d):
                return "-" if d["frac"] is None else f"{d['flip']}/{d['n']}={100 * d['frac']:.1f}%"
            lines.append(f"| {ea.LABEL[a]} | {s} | {pct(sw['turn_pred_code_nonzero'])} | {pct(sw['any_change_code_nonzero'])} | {z['delta_macro_f1']:+.4f}（{z['real']['macro_f1']:.4f} → {z['zeroed']['macro_f1']:.4f}） |")
    return lines


def start_lines(out):
    st = r2.read_start(out)
    cfg = r2.read_config(out)
    if not st:
        return ["（没有 start.txt：run.sh 还没过起跑条件）"]
    keys = ("started_at", "lock_wait_s", "idle_wait_s", "idle_cores", "idle_max_cores", "idle_limit_s", "idle_forced", "gz_count", "cpu_quota_cores", "lanes", "L_STAR")
    return ["- " + "，".join(f"{k}={st.get(k, cfg.get(k, '-'))}" for k in keys)]


def episode_rows(data):
    for a in ea.ARMS:
        for s in ea.SEEDS_TRAIN:
            for seed in sorted(data[(a, s)]["eps"]):
                rec = data[(a, s)]["eps"][seed]
                yield [a, s, seed, int(rec["success"]), rec["klass"], rec["final"], rec["first_late"], rec["retry"] or "", rec["steps"] if rec["steps"] is not None else "",
                       int(rec["flag_corridor"]), int(rec["flag_collision"])]


def build(out, note=None):
    """汇总全部：返回 (markdown 文本, verdicts dict)。"""
    data = load_all(out)
    states = {k: ("完整" if arm_done(v) else v["counts"]["state"]) for k, v in data.items()}
    done = all(arm_done(v) for v in data.values())
    v = compute_verdicts(data, done)
    n_done = sum(1 for x in data.values() if arm_done(x))
    lines = ["# exp-e2e-heading-code-20261112 汇总（卡 E2E-1：航向码端到端学生 2×2）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [f"评测臂完整 {n_done}/12（含重跑结果）。口径与判定规则见 summarize_e2e.py 文件头；阈值是卡 E2E-1 的预注册，起跑后不改。成功 = 无碰撞、过成功线、|y| ≤ 3 m（metric_version sim09.v2.coast1s.corridor3m）；"
              "K=1 柱阵，不带 --heading-guard；A1、A2 带 --student-cond heading-code。", "",
              "## 1. 预注册的三个比较", ""] + verdict_lines(v, done)
    lines += ["", "## 2. 2×2 成功数（60 集 = 3 个训练种子 × 20 个冻结 seed）", ""] + two_by_two(v)
    lines += ["", "## 3. 每臂结局（撞柱、越界、未到线）", ""] + arm_table(data, v)
    lines += ["", "## 4. 三个训练种子各自的数", ""] + per_seed_table(data)
    act, _stats = action_table(data)
    lines += ["", "## 5. 动作占比与回转（已执行动作；码由已执行动作重算）", ""] + act
    cond, cond_bad = cond_table(data)
    lines += ["", "## 6. 采集行 cond_code 与重算的码是否逐步相同", ""] + cond
    lines += ["", "## 7. 每步生效时长、迟到步、cgroup 平均核数与限流", ""] + timing_tables(data, out)
    lines += ["", "## 8. 含迟到步的集与重跑（条款 R6）", ""] + retry_table(data)
    lines += ["", "## 9. 离线探针（val；训练曲线与 val 指标见 offline/offline.md）", ""] + offline_lines(out)
    lines += ["", "## 10. 起跑记录", ""] + start_lines(out)
    inconsistent = sum(1 for arm in data.values() for rec in arm["eps"].values() if not rec["consistent"])
    lines += ["", "## 11. 自检", "",
              f"- eval.json 的成败与轨迹判据（lc.block_outcomes）不一致的集：{inconsistent}（应为 0）。",
              f"- cond_code 与重算不一致或图像臂带 cond_code 的臂：{cond_bad}（应为 0）。",
              "- 各评测臂状态：" + "；".join(f"{ea.NAME[a]}-s{s} {states[(a, s)]}" for a in ea.ARMS for s in ea.SEEDS_TRAIN)]
    return "\n".join(lines) + "\n", v, data


def write_outputs(out, note=None):
    text, v, data = build(out, note)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "verdicts.json"), "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(out, "episodes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "train_seed", "eval_seed", "success", "class", "final", "first_late", "retry", "steps", "corridor_flag", "collision_flag"])
        w.writerows(episode_rows(data))
    return text, v


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="输出目录 $OUT；--late-seeds 时是评测臂目录")
    ap.add_argument("--late-seeds", action="store_true", help="打印臂目录里含迟到步的 seed（逗号分隔）")
    ap.add_argument("--note", help="写在 summary.md 标题下的一行说明")
    args = ap.parse_args(argv)
    if args.late_seeds:
        print(",".join(str(s) for s in r2.late_seeds(args.paths[0])))
        return 0
    out = os.path.abspath(args.paths[0] if args.paths else OUT)
    text, _v = write_outputs(out, args.note)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
