#!/usr/bin/env python3
"""teacher-clearance-3dgs 验收分析（任务卡 W1-12）：臂 A（swept）对 W1-11 新教师臂逐集配对，预注册判决，臂 B 报告，步分支占比，
教师标签分布，仍然撞在前的集逐集归类，确定性回归比对。

子命令：
  acceptance <EXP_OUT> [--suites outdoor,indoor] [--replay] [--occ-cache DIR]
      读 <EXP_OUT>/<suite>/{config.json, <臂 A>/, <臂 B>/（可选）, <W1-11 新教师臂>/（W1-11 输出目录的只读符号链接）}，
      写 <EXP_OUT>/analysis/acceptance.md 与 acceptance.json；只统计已写 done.json 的路线、episode < k 的集。
      --replay：逐集离线回放教师（只加载场景占据，不渲染、不占 GPU）：臂 A / B 用 clearance="swept"，状态取记录的 pose_before / yaw_before
      与上一步 trace.jsonl 的末样本速度（第 0 步速度 0），W1-11 臂用 clearance="ray"。回放得到每步的规则分支（OccAstarTeacher.last_branch），
      用来出分支占比与撞在前的归类；复算动作与记录的 action_expert 不一致的步数一并报（应为 0）。要 3dgs-venv 的 python。
      --occ-cache：占据栅格的 pickle 缓存目录（反复跑分析时省时间；缺省不缓存）。
  regress <新目录> <旧目录> [--json 路径]
      逐集逐步精确比对两个臂目录里的 actions.jsonl（浮点按 == 比，不设容差）；全部一致返回 0，有差异返回 1。W1-11 analyze.py 的同名子命令。

指标口径沿用 W1-11 analyze.py（metric_version 3dgs.pilot.v2+stop.v1）：到点成功 = 首次进入 waypoint 半径早于首次碰撞；撞在前 = 没到点就撞；
超时 = 既没到点也没撞；SPL(stop) 分母是请求集数。配对 = 同场景同路线同集（同 seed）。

预注册判决（run.sh 头里的原文；这里用 Fraction 比边界，不用浮点）：
  预测：臂 A 室内到点 >= 0.90、撞在前 <= 0.06、超时 <= 0.03；室外到点 >= 0.97；室内「新成旧败」减「旧成新败」>= 8 集。
  证伪：室内到点 < 0.87，或超时 > 0.05。之间记「介于预测与证伪之间」。路线没跑完记「未判（数据不全）」。
config.json 里本脚本读的键：scenes、routes、k、seed0、jitter、hold、arm_a、arm_b（可为 None）、w111_arm、teacher_tol_deg、scenes_dir、repo、
routes_selected（可选，给了就出室内按路线长度分桶的表）。
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import pickle
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _load_w111():
    """W1-11 的 analyze.py（wilson / exact_mcnemar / summarize / pair / 读盘 / regress 等）；同一个检出里的文件，不复制。"""
    spec = importlib.util.spec_from_file_location("w111_analyze", os.path.join(HERE, "..", "2026-11-10-teacher-tol-3dgs", "analyze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load_w111()
SAFE, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAME = {SAFE: "悬停", FORWARD: "前进", LEFT: "左转", RIGHT: "右转"}
BRANCHES = ("forward", "turn_err", "turn_blocked", "turn_swept", "wait_swept", "stop", "plan_fail")

# ------------------------------------------------------------------ 预注册（逐字对应 run.sh 头里的话）
PRE_INDOOR_REACH_MIN = Fraction(90, 100)            # 臂 A 室内到点成功率不低于 0.90
PRE_INDOOR_COLLIDE_FIRST_MAX = Fraction(6, 100)     # 撞在前不高于 0.06
PRE_INDOOR_TIMEOUT_MAX = Fraction(3, 100)           # 超时不高于 0.03
PRE_OUTDOOR_REACH_MIN = Fraction(97, 100)           # 室外到点不低于 0.97
PRE_PAIR_NET_MIN = 8                                # 室内「新成旧败」减「旧成新败」不少于 8 集
FALS_INDOOR_REACH_BELOW = Fraction(87, 100)         # 证伪：室内到点低于 0.87
FALS_INDOOR_TIMEOUT_ABOVE = Fraction(5, 100)        # 或超时高于 0.05

CLS_SPAWN = "出生即撞（第 0 步）"
CLS_TURN = "撞在转向步"
CLS_CAP = "撞在前进步：连续等待满 2 步后放行"
CLS_PASS = "撞在前进步：swept 检查通过"
CLS_HOVER = "撞在悬停步"
CLS_NOREPLAY = "未归类：无回放或回放与记录不一致"
CLASSES = (CLS_SPAWN, CLS_TURN, CLS_CAP, CLS_PASS, CLS_HOVER, CLS_NOREPLAY)


# ------------------------------------------------------------------ 纯函数（有单测）
def _frac(k, n):
    return Fraction(k, n) if n else None


def _fmt(fr):
    return "—" if fr is None else f"{float(fr):.3f}"


def verdict_a(indoor, outdoor, pair_in, complete=True):
    """臂 A 的预注册判决。indoor / outdoor = summarize() 的结果（用 n_req、k_stop、collide_first、timeouts），pair_in = 室内 pair(臂 A, W1-11 新教师臂)
    （b = 新成旧败，c = 旧成新败），complete = 两个 suite 的路线都跑完。返回 {"checks", "falsify", "verdict", "text"}。"""
    ni = indoor["n_req"] if indoor else 0
    no = outdoor["n_req"] if outdoor else 0
    net = (pair_in["b"] - pair_in["c"]) if pair_in else None
    checks = [
        {"id": "indoor_reach", "text": f"室内到点成功率 >= {_fmt(PRE_INDOOR_REACH_MIN)}",
         "value": f"{indoor['k_stop']}/{ni} = {_fmt(_frac(indoor['k_stop'], ni))}" if ni else "无数据",
         "ok": None if not ni else _frac(indoor["k_stop"], ni) >= PRE_INDOOR_REACH_MIN},
        {"id": "indoor_collide_first", "text": f"室内撞在前 <= {_fmt(PRE_INDOOR_COLLIDE_FIRST_MAX)}",
         "value": f"{indoor['collide_first']}/{ni} = {_fmt(_frac(indoor['collide_first'], ni))}" if ni else "无数据",
         "ok": None if not ni else _frac(indoor["collide_first"], ni) <= PRE_INDOOR_COLLIDE_FIRST_MAX},
        {"id": "indoor_timeout", "text": f"室内超时 <= {_fmt(PRE_INDOOR_TIMEOUT_MAX)}",
         "value": f"{indoor['timeouts']}/{ni} = {_fmt(_frac(indoor['timeouts'], ni))}" if ni else "无数据",
         "ok": None if not ni else _frac(indoor["timeouts"], ni) <= PRE_INDOOR_TIMEOUT_MAX},
        {"id": "outdoor_reach", "text": f"室外到点成功率 >= {_fmt(PRE_OUTDOOR_REACH_MIN)}",
         "value": f"{outdoor['k_stop']}/{no} = {_fmt(_frac(outdoor['k_stop'], no))}" if no else "无数据",
         "ok": None if not no else _frac(outdoor["k_stop"], no) >= PRE_OUTDOOR_REACH_MIN},
        {"id": "indoor_pair_net", "text": f"室内配对「新成旧败」减「旧成新败」>= {PRE_PAIR_NET_MIN} 集",
         "value": f"{pair_in['b']} − {pair_in['c']} = {net:+d}" if pair_in else "无数据",
         "ok": None if net is None else net >= PRE_PAIR_NET_MIN},
    ]
    falsify = [
        {"id": "indoor_reach_below", "text": f"室内到点成功率 < {_fmt(FALS_INDOOR_REACH_BELOW)}",
         "hit": None if not ni else _frac(indoor["k_stop"], ni) < FALS_INDOOR_REACH_BELOW},
        {"id": "indoor_timeout_above", "text": f"室内超时 > {_fmt(FALS_INDOOR_TIMEOUT_ABOVE)}",
         "hit": None if not ni else _frac(indoor["timeouts"], ni) > FALS_INDOOR_TIMEOUT_ABOVE},
    ]
    if not complete or any(c["ok"] is None for c in checks) or any(f["hit"] is None for f in falsify):
        v = "未判（数据不全）"
    elif any(f["hit"] for f in falsify):
        v = "证伪成立"
    elif all(c["ok"] for c in checks):
        v = "预测成立"
    else:
        v = "介于预测与证伪之间"
    parts = [f"{c['text']}：{c['value']}（{'满足' if c['ok'] else '未满足' if c['ok'] is not None else '未判'}）" for c in checks]
    hits = [f["text"] for f in falsify if f["hit"]]
    text = f"{v}。" + "；".join(parts) + ("。证伪条件命中：" + "、".join(hits) if hits else "。证伪条件均未命中" if v != "未判（数据不全）" else "")
    return {"checks": checks, "falsify": falsify, "verdict": v, "text": text}


def nav_end(row):
    """导航阶段的末步下标（含）：首次到点或首次碰撞，没有则整集；与 driver.nav_end 同。"""
    ends = [x for x in (row.get("reach_step"), row.get("collision_step")) if x is not None]
    return min(ends) if ends else row["steps"] - 1


def label_counts(rows, nav=False):
    """action_expert 标签的类别计数 {0..3: 步数} 与总步数；nav = 只数导航阶段（到点或撞之前）。"""
    cnt, total = {SAFE: 0, FORWARD: 0, LEFT: 0, RIGHT: 0}, 0
    for r in rows.values():
        labels = r["labels"][:nav_end(r) + 1] if nav else r["labels"]
        for a in labels:
            cnt[a] += 1
        total += len(labels)
    return cnt, total


def _label_cells(cnt):
    return {"n_safe": cnt[SAFE], "n_forward": cnt[FORWARD], "n_left": cnt[LEFT], "n_right": cnt[RIGHT]}


def branch_counts(rows, nav=False):
    """回放得到的规则分支计数 {分支: 步数} 与总步数；没有回放结果的集不计。"""
    cnt, total = {b: 0 for b in BRANCHES}, 0
    for r in rows.values():
        rp = r.get("replay")
        if not rp:
            continue
        br = rp["branch"][:nav_end(r) + 1] if nav else rp["branch"]
        for b in br:
            cnt[b] = cnt.get(b, 0) + 1
        total += len(br)
    return cnt, total


def classify_collision(c, actions_exec, rep):
    """撞在前的一集里撞击步 c（actions 的下标）归哪一类。actions_exec = 每步 action_executed；rep = 回放结果（None = 没回放）。
    返回 (类别, 细节)。细节含 act（撞击步动作）、prev（上一步动作）、branch（撞击步回放分支）、prev_branches（前两步回放分支）、pred_err_m。"""
    act = actions_exec[c]
    prev = actions_exec[c - 1] if c else None
    ok = rep is not None and len(rep["branch"]) > c and rep["match"][c]
    det = {"step": c, "act": act, "prev": prev, "branch": rep["branch"][c] if ok else None,
           "prev_branches": rep["branch"][max(0, c - 2):c] if rep is not None else None,
           "pred_err_m": rep["pred_err"][c] if ok else None}
    if c == 0:
        return CLS_SPAWN, det
    if act in (LEFT, RIGHT):
        return CLS_TURN, det
    if act == SAFE:
        return CLS_HOVER, det
    if not ok:
        return CLS_NOREPLAY, det
    if rep["branch"][c] == "forward" and rep["blocked"][c]:
        return CLS_CAP, det
    return CLS_PASS, det


def predicted_body_disp(pred_a, pred_b, vbx, vby):
    """P2 前进步位移预测（机体系）：d = A·v + b；与 planner._swept_blocked 同式。"""
    return (pred_a[0][0] * vbx + pred_a[0][1] * vby + pred_b[0], pred_a[1][0] * vbx + pred_a[1][1] * vby + pred_b[1])


class _StateV:
    """回放用的状态：位姿取记录，世界系水平速度取上一步 trace 的末样本。"""

    def __init__(self, x, y, z, yaw, vx, vy):
        self.x, self.y, self.z, self.yaw, self.vx, self.vy = x, y, z, yaw, vx, vy


def replay_episode(teacher, actions, trace, pred_a=None, pred_b=None):
    """逐步复算教师（状态取记录，不渲染）。返回 None（swept 教师缺 trace 速度，无法复算）或
    {"n_steps", "mismatch", "branch": [...], "match": [...], "blocked": [...], "pred_err": [...]}：
    blocked = swept 检查在这一步是否会否决前进（ray 教师为 None）；pred_err = 预测的前进位移与实际位移之差的模（缺速度或缺预测系数为 None）。
    每集用新的教师对象（连续等待计数从 0 开始）。"""
    swept = getattr(teacher, "clearance", "ray") == "swept"
    last = {}
    for t in trace or []:
        last[t["step"]] = t["vel"]
    out = {"branch": [], "match": [], "blocked": [], "pred_err": []}
    for i, r in enumerate(actions):
        p = r["pose_before"]
        yaw = math.radians(float(r["yaw_before"]))
        vel = (0.0, 0.0) if i == 0 else (last.get(i - 1) or None)
        if vel is None and swept:
            return None
        vx, vy = (vel[0], vel[1]) if vel is not None else (0.0, 0.0)
        st = _StateV(float(p["x"]), float(p["y"]), float(p.get("z", 2.0)), yaw, vx, vy)
        blocked = bool(teacher._swept_blocked(st, st.x, st.y, st.z)) if swept else None
        act, _ = teacher(None, None, st)
        g = r.get("pose_gt")
        err = None
        if pred_a is not None and vel is not None and g is not None:
            c, s = math.cos(yaw), math.sin(yaw)
            vb = (c * vx + s * vy, -s * vx + c * vy)
            dx, dy = g["x"] - p["x"], g["y"] - p["y"]
            actual = (c * dx + s * dy, -s * dx + c * dy)
            pd = predicted_body_disp(pred_a, pred_b, vb[0], vb[1])
            err = math.hypot(pd[0] - actual[0], pd[1] - actual[1])
        out["branch"].append(teacher.last_branch)
        out["match"].append(act == r["action_expert"])
        out["blocked"].append(blocked)
        out["pred_err"].append(err)
    out["n_steps"] = len(actions)
    out["mismatch"] = sum(not m for m in out["match"])
    return out


def spawn_stats(rows_a, rows_b):
    """臂 B 对臂 A 的出生位姿：同 (场景, 路线, 集) 里起点（x, y, yaw）不同的集数，与各自第 0 步就撞的集数。"""
    keys = [k for k in rows_a if k in rows_b]
    diff = sum(1 for k in keys if _pose(rows_a[k]) != _pose(rows_b[k]))
    return {"pairs": len(keys), "start_pose_differs": diff,
            "collision_step0_a": sum(1 for k in keys if rows_a[k].get("collision_step") == 0),
            "collision_step0_b": sum(1 for k in keys if rows_b[k].get("collision_step") == 0)}


def _pose(row):
    p = row.get("start_pose") or {}
    return (p.get("x"), p.get("y"), p.get("yaw_deg"))


# ------------------------------------------------------------------ 读盘
def load_extra(row):
    """给 load_episodes 的一行补：撞 / 到点步下标、起点位姿、逐步标签与执行动作。"""
    m = W.load_json(os.path.join(row["dir"], "metrics.json")) or {}
    acts = W.jsonl(os.path.join(row["dir"], "actions.jsonl"))
    row.update(collision_step=m.get("collision_step"), reach_step=m.get("reach_step"), start_pose=m.get("start_pose"),
               labels=[a["action_expert"] for a in acts], executed=[a["action_executed"] for a in acts])


def load_occ(scene_mod, coll, cache_dir, scene, path):
    cp = os.path.join(cache_dir, f"{scene}.pkl") if cache_dir else None
    if cp and os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)
    occ = coll.from_scene(scene_mod.load_scene(path))
    if cp:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cp + ".tmp", "wb") as f:
            pickle.dump(occ, f, protocol=4)
        os.replace(cp + ".tmp", cp)
    return occ


def make_teacher(planner_mod, occ_planner, waypoint, tol, clearance, hold):
    """回放用的教师：与实验里该臂的教师同构造（容差取记录值；swept 才带 hold_s 与 clearance，ray 用缺省，与 W1-11 的教师构造相同）。"""
    kw = {"hold_s": hold, "clearance": "swept"} if clearance == "swept" else {}
    return planner_mod.OccAstarTeacher(occ_planner, waypoint, tol_deg=tol, **kw)


def replay_rows(cfg, rows_by_arm, clearance_of, occ_cache=None, repo=REPO):
    """给 rows_by_arm 里每一行写 row["replay"]（replay_episode 的结果，或 None）。每个场景只加载一次占据。"""
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "sim", "tools"))
    scene_mod = importlib.import_module("sim.3dgs.scene")
    coll = importlib.import_module("sim.3dgs.collision")
    planner_mod = importlib.import_module("sim.3dgs.planner")
    by_scene = {}
    for rows in rows_by_arm.values():
        for r in rows.values():
            by_scene.setdefault(r["scene"], []).append(r)
    for scene, rs in sorted(by_scene.items()):
        path = os.path.join(cfg["scenes_dir"], scene, "scene.json")
        occ = load_occ(scene_mod, coll, occ_cache, scene, path)
        occ_planner = planner_mod.OccPlanner(occ)
        for r in rs:
            sc = scene_mod.load_scene(path)
            scene_mod.apply_route(sc, r["route"])
            man = W.load_json(os.path.join(r["dir"], "episode_manifest.json")) or {}
            tol = W.resolve_tol(man, cfg.get("teacher_tol_deg"))
            swept = clearance_of[r["arm"]] == "swept"
            teacher = make_teacher(planner_mod, occ_planner, sc["waypoint"], tol, clearance_of[r["arm"]], cfg["hold"])
            tp = os.path.join(r["dir"], "trace.jsonl")
            r["replay"] = replay_episode(teacher, W.jsonl(os.path.join(r["dir"], "actions.jsonl")),
                                         W.jsonl(tp) if os.path.exists(tp) else None,
                                         planner_mod.SWEPT_PRED_A if swept else None, planner_mod.SWEPT_PRED_B if swept else None)
        del occ, occ_planner


# ------------------------------------------------------------------ 分析
def episode_label(r):
    return W.outcome_of(r)


def collide_first_table(rows, actions_by_dir=None):
    """臂内每个撞在前的集：归类与细节。"""
    out = []
    for key in sorted(rows):
        r = rows[key]
        if not r["collide_first"]:
            continue
        cls, det = classify_collision(r["collision_step"], r["executed"], r.get("replay"))
        out.append({"scene": r["scene"], "route": r["route"], "episode": r["episode"], "seed": r["seed"], "steps": r["steps"],
                    "class": cls, **det})
    return out


def timeout_table(rows):
    out = []
    for key in sorted(rows):
        r = rows[key]
        if not r["timeout"]:
            continue
        br = (r.get("replay") or {}).get("branch") or []
        out.append({"scene": r["scene"], "route": r["route"], "episode": r["episode"], "seed": r["seed"], "steps": r["steps"],
                    "final_wp_dist_m": r.get("final_wp_dist_m"), "n_wait_swept": br.count("wait_swept"),
                    "n_turn_swept": br.count("turn_swept"), "alt_longest": r["alt_longest"], "replayed": bool(br)})
    return out


def first_divergence(rows_a, rows_w, keys):
    """臂 A 与 W1-11 新教师臂同一集的第一个分歧步（W.compare_actions；两臂同 seed 同出生位姿时才有意义）。"""
    out = {}
    for k in keys:
        if k in rows_a and k in rows_w:
            res = W.compare_actions(W.jsonl(os.path.join(rows_w[k]["dir"], "actions.jsonl")),
                                    W.jsonl(os.path.join(rows_a[k]["dir"], "actions.jsonl")))
            out[k] = res["first_diff"]
    return out


def discordant(rows_a, rows_w):
    """配对里两臂结局不同的集：b = A 成 W 败，c = A 败 W 成。"""
    b, c = [], []
    for k in sorted(set(rows_a) & set(rows_w)):
        a, w = rows_a[k]["success_stop"], rows_w[k]["success_stop"]
        if a and not w:
            b.append(k)
        elif w and not a:
            c.append(k)
    return b, c


def analyze_suite(out, name, replay, occ_cache=None):
    suite = os.path.join(out, name)
    cfg = W.load_json(os.path.join(suite, "config.json"))
    arm_a, arm_b, arm_w = cfg["arm_a"], cfg.get("arm_b"), cfg["w111_arm"]
    arms = [arm_a] + ([arm_b] if arm_b else []) + [arm_w]
    arms = [a for a in arms if os.path.exists(os.path.join(suite, a))]
    k = cfg["k"]
    rows, done, summ = {}, {}, {}
    for arm in arms:
        done[arm] = W.done_routes(suite, cfg, arm)
        rows[arm] = W.load_episodes(suite, cfg, arm)
        for r in rows[arm].values():
            load_extra(r)
        summ[arm] = W.summarize(rows[arm], k * len(done[arm]))
    planned = sum(len(W.routes_of(cfg, s)) for s in cfg["scenes"])
    clearance_of = {arm_a: "swept", arm_w: "ray", **({arm_b: "swept"} if arm_b else {})}
    if replay:
        replay_rows(cfg, {a: rows[a] for a in arms}, clearance_of, occ_cache, cfg.get("repo") or REPO)
    res = {"name": name, "arms_order": arms, "arm_a": arm_a, "arm_b": arm_b, "arm_w": arm_w,
           "config": {x: cfg.get(x) for x in ("scenes", "k", "seed0", "jitter", "hold", "step_frac", "step_pad", "step_cap", "teacher_tol_deg")},
           "routes_planned": planned, "routes_done": {a: len(done[a]) for a in arms}, "arms": summ,
           "complete": len(done[arm_a]) == planned and len(done.get(arm_w, ())) >= planned}
    res["pairs"] = {}
    for lab, (x, y) in {"A_vs_W": (arm_a, arm_w), "B_vs_W": (arm_b, arm_w), "B_vs_A": (arm_b, arm_a)}.items():
        if x in rows and y in rows and rows[x] and rows[y]:
            res["pairs"][lab] = W.pair(rows[x], rows[y])
    b_keys, c_keys = discordant(rows[arm_a], rows[arm_w]) if arm_w in rows else ([], [])
    div = first_divergence(rows[arm_a], rows[arm_w], c_keys) if arm_w in rows else {}
    res["discordant"] = {"A_success_W_fail": [list(x) for x in b_keys],
                         "A_fail_W_success": [{"key": list(x), "A_outcome": episode_label(rows[arm_a][x]),
                                               "W_outcome": episode_label(rows[arm_w][x]), "first_diff": div.get(x)} for x in c_keys]}
    if arm_b and arm_b in rows:
        res["spawn"] = spawn_stats(rows[arm_a], rows[arm_b])
    res["per_scene"] = {}
    for s in cfg["scenes"]:
        res["per_scene"][s] = {}
        for arm in arms:
            routes = {x for x in done[arm] if x[0] == s}
            sel = {kk: v for kk, v in rows[arm].items() if kk[0] == s}
            res["per_scene"][s][arm] = W.summarize(sel, k * len(routes))
    res["buckets"] = None
    if cfg.get("routes_selected") and os.path.exists(cfg["routes_selected"]):
        rs = W.load_json(cfg["routes_selected"])
        labels = rs["params"].get("bucket_labels") or sorted(set(W.route_buckets(rs).values()))
        rm = W.route_buckets(rs)
        res["buckets"] = {"labels": labels, "tables": {arm: W.bucket_table(rows[arm], rm, labels, k, done[arm]) for arm in arms}}
    res["labels"] = {}
    for arm in arms:
        (call, tall), (cnav, tnav) = label_counts(rows[arm]), label_counts(rows[arm], True)
        res["labels"][arm] = {"all": _label_cells(call), "total": tall, "nav": _label_cells(cnav), "nav_total": tnav}
    res["branches"], res["replay_check"] = {}, {}
    if replay:
        for arm in arms:
            cnt, total = branch_counts(rows[arm])
            ncnt, ntotal = branch_counts(rows[arm], True)
            reps = [r["replay"] for r in rows[arm].values() if r.get("replay")]
            res["branches"][arm] = {"all": cnt, "total": total, "nav": ncnt, "nav_total": ntotal}
            res["replay_check"][arm] = {"episodes": len(rows[arm]), "replayed": len(reps), "steps": sum(x["n_steps"] for x in reps),
                                        "mismatch_steps": sum(x["mismatch"] for x in reps),
                                        "mismatch_episodes": [list(kk) for kk, r in sorted(rows[arm].items()) if r.get("replay") and r["replay"]["mismatch"]]}
    res["collide_first"] = {arm: collide_first_table(rows[arm]) for arm in arms if arm != arm_w}
    res["timeouts"] = {arm: timeout_table(rows[arm]) for arm in arms}
    return res


# ------------------------------------------------------------------ 渲染
def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def f3(v):
    return "—" if v is None else f"{v:.3f}"


def share(n, total):
    return "—" if not total else f"{100 * n / total:.1f}%（{n}/{total}）"


def render_suite(res):
    a, b, w = res["arm_a"], res["arm_b"], res["arm_w"]
    arms = res["arms_order"]
    cfgc = res["config"]
    role = {a: "臂 A（swept）", w: "W1-11 新教师臂（ray）", **({b: "臂 B（swept + 出生两层，只报告）"} if b else {})}
    L = [f"## {res['name']}", "",
         f"- 路线 {res['routes_done'][a]}/{res['routes_planned']}（臂 A 已写 done.json / 计划），每路线 {cfgc['k']} 集，seed = {cfgc['seed0']} + 100·路线 + 集，"
         f"出生扰动 ±{cfgc['jitter']} m，hold {cfgc['hold']} s，教师容差 {f2(cfgc['teacher_tol_deg'])}°"
         + ("" if res["complete"] else "；**路线没跑完**，下表是部分结果"), "",
         "### 1. 汇总", "",
         "| 臂 | 到点成功 k/n [Wilson 95%] | 撞在前 | 超时 | 其他（失败集） | 碰撞（任一步） | SPL(stop) | 每集平均步数 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in sorted(arms, key=lambda x: (x != a, x != b)):
        s = res["arms"][arm]
        ci = s["ci_stop"]
        L.append(f"| {role[arm]} `{arm}` | {s['k_stop']}/{s['n_req']} = {f3(s['sr_stop'])} [{f2(ci[0])}, {f2(ci[1])}] | {s['collide_first']} | {s['timeouts']} "
                 f"| {s['failed']} | {s['collisions']} | {f3(s['spl_stop'])} | {'—' if s['steps_mean'] is None else '%.1f' % s['steps_mean']} |")
    L += ["", "### 2. 逐集配对（到点成功，同场景同路线同 seed，精确 McNemar）", "",
          "| 比较 | 配对数 | b（前者成后者败） | c（前者败后者成） | b − c | Δ = SR(前者) − SR(后者) | p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    names = {"A_vs_W": f"臂 A 对 W1-11（主）", "B_vs_W": "臂 B 对 W1-11（只报告）", "B_vs_A": "臂 B 对臂 A（只报告）"}
    for lab, p in res["pairs"].items():
        L.append(f"| {names[lab]} | {p['pairs']} | {p['b']} | {p['c']} | {p['b'] - p['c']:+d} | {p['delta']:+.3f} | {p['p']:.3g} |")
    if "spawn" in res:
        sp = res["spawn"]
        L += ["", f"臂 B 与臂 A 同 seed，但出生开关会让被拒的候选点重抽，起点不同的集：{sp['start_pose_differs']}/{sp['pairs']}（这些集臂 B 对臂 A 不是同起点配对）；"
                  f"第 0 步就撞的集：臂 A {sp['collision_step0_a']}，臂 B {sp['collision_step0_b']}。"]
    d = res["discordant"]
    L += ["", f"臂 A 成 W1-11 败（b）{len(d['A_success_W_fail'])} 集；臂 A 败 W1-11 成（c）{len(d['A_fail_W_success'])} 集"
          + ("。" if not d["A_fail_W_success"] else "，逐集："), ""]
    if d["A_fail_W_success"]:
        L += ["| 场景 r 集 | 臂 A 结局 | W1-11 结局 | 与 W1-11 首个分歧（步 / 字段） |", "| --- | --- | --- | --- |"]
        for x in d["A_fail_W_success"]:
            fd = x["first_diff"]
            L.append(f"| {x['key'][0]} r{x['key'][1]} ep{x['key'][2]} | {x['A_outcome']} | {x['W_outcome']} | "
                     + ("—" if not fd else f"{fd['step']} / {fd['key']}") + " |")
    L += ["", "### 3. 规则分支的步占比（回放得到）", ""]
    if res["branches"]:
        L += ["| 臂 | 范围 | 总步数 | " + " | ".join(BRANCHES) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in BRANCHES) + " |"]
        for arm in sorted(arms, key=lambda x: (x != a, x != b)):
            br = res["branches"][arm]
            for scope, key, tk in (("全部步", "all", "total"), ("导航阶段（到点或撞之前）", "nav", "nav_total")):
                L.append(f"| `{arm}` | {scope} | {br[tk]} | " + " | ".join(share(br[key].get(x, 0), br[tk]) for x in BRANCHES) + " |")
        L += ["", "回放核对（复算动作与记录的 action_expert 不一致的步数，应为 0）：" + "；".join(
            f"`{arm}` {c['mismatch_steps']}/{c['steps']} 步，回放 {c['replayed']}/{c['episodes']} 集"
            + ("" if not c["mismatch_episodes"] else f"（不一致的集 {c['mismatch_episodes'][:8]}）") for arm, c in res["replay_check"].items()) + "。"]
    else:
        L.append("未回放（没给 --replay）。")
    L += ["", "### 4. 教师标签（action_expert）的类别分布", "",
          "| 臂 | 范围 | 总步数 | 悬停（SAFE_NEUTRAL） | 前进 | 左转 | 右转 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm in sorted(arms, key=lambda x: (x != a, x != b)):
        lb = res["labels"][arm]
        for scope, key, tk in (("全部步", "all", "total"), ("导航阶段", "nav", "nav_total")):
            c = lb[key]
            L.append(f"| `{arm}` | {scope} | {lb[tk]} | {share(c['n_safe'], lb[tk])} | {share(c['n_forward'], lb[tk])} | "
                     f"{share(c['n_left'], lb[tk])} | {share(c['n_right'], lb[tk])} |")
    for arm in [x for x in (a, b) if x]:
        L += ["", f"### 5. 仍然撞在前的集逐集归类：`{arm}`（{role[arm]}）", ""]
        tab = res["collide_first"].get(arm, [])
        if not tab:
            L.append("没有撞在前的集。")
            continue
        cnt = {}
        for t in tab:
            cnt[t["class"]] = cnt.get(t["class"], 0) + 1
        L += ["共 %d 集：" % len(tab) + "；".join(f"{c} {cnt[c]}" for c in CLASSES if c in cnt) + "。", "",
              "| 场景 r 集 | seed | 撞击步 | 该步动作 | 回放分支 | 上一步动作 | 前两步分支 | 预测位移误差 m | 归类 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for t in tab:
            L.append(f"| {t['scene']} r{t['route']} ep{t['episode']} | {t['seed']} | {t['step']} | {ACTION_NAME.get(t['act'], t['act'])} | {t['branch'] or '—'} | "
                     f"{ACTION_NAME.get(t['prev'], '—') if t['prev'] is not None else '—'} | {'、'.join(t['prev_branches']) if t['prev_branches'] else '—'} | "
                     f"{f3(t['pred_err_m'])} | {t['class']} |")
    for arm in [x for x in (a, b) if x]:
        tab = res["timeouts"].get(arm, [])
        L += ["", f"### 6. 超时的集：`{arm}`", ""]
        if not tab:
            L.append("没有超时的集。")
            continue
        L += ["| 场景 r 集 | seed | 步数 | 终点距离 m | wait_swept 步 | turn_swept 步 | 最长交替段 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for t in tab:
            L.append(f"| {t['scene']} r{t['route']} ep{t['episode']} | {t['seed']} | {t['steps']} | {f3(t['final_wp_dist_m'])} | "
                     f"{t['n_wait_swept'] if t['replayed'] else '—'} | {t['n_turn_swept'] if t['replayed'] else '—'} | {t['alt_longest']} |")
    L += ["", "### 7. 分场景（每格 = 到点 / 请求集数，括号内：撞在前 / 超时）", "", "| 场景 | " + " | ".join(f"`{x}`" for x in arms) + " |", "| --- | " + " | ".join("---" for _ in arms) + " |"]
    for s, dd in res["per_scene"].items():
        cells = []
        for arm in arms:
            x = dd[arm]
            cells.append("—" if not x["n_req"] else f"{x['k_stop']}/{x['n_req']}（{x['collide_first']} / {x['timeouts']}）")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    if res["buckets"]:
        L += ["", "### 8. 按路线长度分桶（桶 = A* 长度；每格 = 到点 / 请求集数，括号内：超时 / 碰撞）", "", "| 桶 | " + " | ".join(f"`{x}`" for x in arms) + " |",
              "| --- | " + " | ".join("---" for _ in arms) + " |"]
        for lb in res["buckets"]["labels"]:
            cells = []
            for arm in arms:
                t = res["buckets"]["tables"][arm][lb]
                cells.append("—" if not t["n"] else f"{t['k_stop']}/{t['n']} = {t['k_stop'] / t['n']:.2f}（{t['timeouts']} / {t['collisions']}）")
            L.append(f"| {lb} | " + " | ".join(cells) + " |")
    return L


def render_verdict(v):
    L = ["## 臂 A 的预注册判决（机械给出，结论以主会话为准）", "", f"**{v['verdict']}**", "",
         "| 预注册条件（预测） | 实测 | 满足 |", "| --- | --- | --- |"]
    for c in v["checks"]:
        L.append(f"| {c['text']} | {c['value']} | {'是' if c['ok'] else '否' if c['ok'] is not None else '未判'} |")
    L += ["", "| 证伪条件 | 命中 |", "| --- | --- |"]
    for f in v["falsify"]:
        L.append(f"| {f['text']} | {'是' if f['hit'] else '否' if f['hit'] is not None else '未判'} |")
    return L


def acceptance(args):
    suites = args.suites.split(",") if args.suites else [s for s in ("outdoor", "indoor") if os.path.exists(os.path.join(args.out, s, "config.json"))]
    results = {name: analyze_suite(args.out, name, args.replay, args.occ_cache) for name in suites}
    ind, outd = results.get("indoor"), results.get("outdoor")
    complete = bool(ind and outd and ind["complete"] and outd["complete"])
    v = verdict_a(ind["arms"][ind["arm_a"]] if ind else None, outd["arms"][outd["arm_a"]] if outd else None,
                  ind["pairs"].get("A_vs_W") if ind else None, complete)
    lines = ["# 3DGS 教师前向净空 swept：验收分析（任务卡 W1-12）", "",
             f"- metric_version `{W.METRIC_VERSION}`；到点成功 = 首次进入 waypoint 半径早于首次碰撞；撞在前 = 没到点就撞；超时 = 既没到点也没撞；"
             "SPL(stop) 分母是请求集数；平均步数按已写 done.json 的集算。",
             "- 来源：`<suite>/<臂>/<场景>/r<N>/ep_*/{metrics.json, actions.jsonl, episode_manifest.json, trace.jsonl}`；W1-11 新教师臂是旧实验输出目录的只读符号链接。",
             "- 回放：教师逐步复算，状态取记录的位姿与上一步 trace 末样本的速度（第 0 步速度 0）；未回放的数字标「未回放」。", ""]
    lines += render_verdict(v) + [""]
    for res in results.values():
        lines += render_suite(res) + [""]
    adir = os.path.join(args.out, "analysis")
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "acceptance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(adir, "acceptance.json"), "w", encoding="utf-8") as f:
        json.dump({"metric_version": W.METRIC_VERSION, "verdict_a": v, "suites": results}, f, ensure_ascii=False, indent=1)
    print("\n".join(lines))
    return 0


def regress(args):
    return W.regress(args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acceptance")
    a.add_argument("out")
    a.add_argument("--suites", default=None, help="逗号分隔；缺省 = outdoor、indoor 里有 config.json 的")
    a.add_argument("--replay", action="store_true", help="逐集离线回放教师（3dgs-venv，不占 GPU）")
    a.add_argument("--occ-cache", default=None, help="占据栅格 pickle 缓存目录")
    r = sub.add_parser("regress")
    r.add_argument("new")
    r.add_argument("old")
    r.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return acceptance(args) if args.cmd == "acceptance" else regress(args)


if __name__ == "__main__":
    sys.exit(main())
