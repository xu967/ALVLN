#!/usr/bin/env python3
"""teacher-tol-3dgs 验收分析（任务卡 W1-11）：新旧教师逐集配对、左右交替段统计、残留交替段的回放归因、确定性回归比对。

子命令：
  acceptance <EXP_OUT> [--suites outdoor,indoor] [--replay]
      读 <EXP_OUT>/<suite>/{config.json, <新臂>/, <旧臂>/}（臂目录结构同 bearing-guard 的 driver.py 输出），
      写 <EXP_OUT>/analysis/acceptance.md 与 acceptance.json。只统计已写 done.json 的路线。
      --replay：对含交替段的集离线回放教师（只加载场景占据，不渲染、不占 GPU），逐步复算动作与规则分支（OccAstarTeacher.last_branch），
      回答「残留的交替段是不是前向被堵时按误差符号转造成的」。要 3dgs-venv 的 python。
  regress <新目录> <旧目录>
      逐集逐步精确比对两个臂目录里的 actions.jsonl（浮点按 == 比，不设容差）；全部一致返回 0，有差异返回 1。

指标口径与 bearing-guard 的 summarize.py 一致（metric_version 3dgs.pilot.v2+stop.v1）：
  到点成功 = 首次进入 waypoint 半径早于首次碰撞（metrics.json success_stop）；碰撞率 = 整集任一步碰撞（含到点之后），撞在前 = 没到点就撞；
  超时 = 既没到点也没撞；SPL(stop) 把成功换成到点成功，分母是请求集数。
交替段（与 sage3d 报告 §4.5 的 teacher_osc.py 同判据）：action_executed 在 {2 (YAW_LEFT), 3 (YAW_RIGHT)} 里严格左右交替、连续 >= 6 步。
config.json 里本脚本读的键：scenes、routes（列表或字典）、k、new_arm、old_arm、teacher_tol_deg（新臂容差）、accept_sr（验收阈值）、
accept_timeouts_max（超时集数上限）、scenes_dir / repo（回放用）、routes_selected（可选，给了就出按路线长度分桶的表）。
"""
import argparse
import glob
import importlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
METRIC_VERSION = "3dgs.pilot.v2+stop.v1"
MIN_RUN = 6                      # 交替段最短步数
OLD_TOL_DEG = 12.0               # 旧教师容差 = planner.YAW_TOL_DEG；旧 manifest 没有 teacher_yaw_tol_deg 键时按它算
P_ALPHA = 0.05
SAFE, FORWARD, LEFT, RIGHT = 0, 1, 2, 3
COMPARE_KEYS = ("action_expert", "action_executed", "pose_before", "yaw_before", "dyaw_deg", "displacement_m", "collision")


# ------------------------------------------------------------------ 纯函数（有单测）
def alt_runs(seq, min_run=MIN_RUN):
    """seq = action_executed 序列；返回严格 LEFT / RIGHT 交替、长度 >= min_run 的极大段 [(起, 止)]（下标，含端点）。"""
    runs, i = [], 0
    while i < len(seq):
        if seq[i] not in (LEFT, RIGHT):
            i += 1
            continue
        j = i
        while j + 1 < len(seq) and seq[j + 1] in (LEFT, RIGHT) and seq[j + 1] != seq[j]:
            j += 1
        if j - i + 1 >= min_run:
            runs.append((i, j))
        i = j + 1
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def exact_mcnemar(b, c):
    """双侧精确检验（二项分布 p = 0.5），与 summarize.py 同式。"""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def verdict(sr, threshold):
    """预注册阈值判决：点估计 >= 阈值记达标（1e-12 只吸收浮点除法的误差）。"""
    return "达标" if sr is not None and sr >= threshold - 1e-12 else "未达标"


def verdict_max(value, limit):
    """上界判决（超时集数 <= 阈值记达标）。"""
    return "达标" if value <= limit else "未达标"


def route_buckets(rs):
    """routes_selected.json -> {(场景, 路线下标): 桶名}。"""
    return {(s, x["route"]): x["bucket"] for s, xs in rs["scenes"].items() for x in xs}


def routes_of(cfg, scene):
    return cfg["routes"].get(scene, []) if isinstance(cfg["routes"], dict) else cfg["routes"]


def summarize(rows, n_req):
    """rows = {(场景, 路线, 集): 行}；分母 n_req = 已写 done.json 的路线数 × 每路线集数（失败集算失败）。"""
    vals = list(rows.values())
    k = sum(bool(r["success_stop"]) for r in vals)
    to = [r for r in vals if r["timeout"]]
    spl = [r["spl_stop"] for r in vals if r["spl_stop"] is not None]
    return {"n_req": n_req, "n": len(vals), "failed": n_req - len(vals),
            "k_stop": k, "sr_stop": k / n_req if n_req else None, "ci_stop": wilson(k, n_req),
            "timeouts": len(to), "timeouts_alt": sum(bool(r["alt_runs"]) for r in to),
            "episodes_alt": sum(bool(r["alt_runs"]) for r in vals),
            "collisions": sum(bool(r["collision"]) for r in vals), "collide_first": sum(bool(r["collide_first"]) for r in vals),
            "spl_stop": sum(spl) / n_req if n_req else None,
            "steps_mean": sum(r["steps"] for r in vals) / len(vals) if vals else None}


def pair(rows_a, rows_b):
    """A 对 B 逐集配对（同场景同路线同集 = 同 seed）。b = A 成 B 败，c = A 败 B 成，Δ = SR(A) − SR(B)。"""
    keys = [k for k in rows_a if k in rows_b]
    b = sum(rows_a[k]["success_stop"] and not rows_b[k]["success_stop"] for k in keys)
    c = sum(not rows_a[k]["success_stop"] and rows_b[k]["success_stop"] for k in keys)
    n = len(keys)
    delta = (sum(bool(rows_a[k]["success_stop"]) for k in keys) - sum(bool(rows_b[k]["success_stop"]) for k in keys)) / n if n else None
    return {"pairs": n, "b": b, "c": c, "p": exact_mcnemar(b, c), "delta": delta}


def bucket_table(rows, rs_map, labels, k, done):
    """按路线长度分桶；n = 该桶内已写 done.json 的路线数 × k。"""
    out = {}
    for lb in labels:
        routes = {key for key, v in rs_map.items() if v == lb and key in done}
        sel = [r for r in rows.values() if (r["scene"], r["route"]) in routes]
        to = [r for r in sel if r["timeout"]]
        out[lb] = {"n": k * len(routes), "k_stop": sum(bool(r["success_stop"]) for r in sel), "timeouts": len(to),
                   "timeouts_alt": sum(bool(r["alt_runs"]) for r in to), "collisions": sum(bool(r["collision"]) for r in sel)}
    return out


def resolve_tol(manifest, arm_tol):
    """回放用的教师容差：episode_manifest.json 的 teacher_yaw_tol_deg 是记录，优先；没有这个键（旧代码产出的集）才用该臂在 config 里的值，
    再没有就是旧教师的 12°（OLD_TOL_DEG）。"""
    if manifest and manifest.get("teacher_yaw_tol_deg") is not None:
        return float(manifest["teacher_yaw_tol_deg"])
    return float(arm_tol) if arm_tol is not None else OLD_TOL_DEG


def replay_branches(teacher, actions):
    """逐步复算教师：状态取记录的 pose_before / yaw_before，返回每步 {action, branch, match（与记录的 action_expert 一致）}。
    teacher 只要 __call__(rgb, depth, state) -> (action, _) 与 last_branch；不需要图像。"""
    out = []
    for r in actions:
        p = r["pose_before"]
        st = _State(float(p["x"]), float(p["y"]), float(p.get("z", 2.0)), math.radians(float(r["yaw_before"])))
        act, _ = teacher(None, None, st)
        out.append({"action": act, "branch": teacher.last_branch, "match": act == r["action_expert"]})
    return out


class _State:
    def __init__(self, x, y, z, yaw):
        self.x, self.y, self.z, self.yaw = x, y, z, yaw


def classify_run(branches):
    """一段交替转向里每步的规则分支 -> 类别。turn_err = 航向误差超容差；turn_blocked = 前向被堵时按误差符号转。"""
    s = set(branches)
    if s == {"turn_err"}:
        return "只有航向误差超容差"
    if s == {"turn_blocked"}:
        return "只有前向被堵转向"
    if s == {"turn_err", "turn_blocked"}:
        return "两种都有"
    return "其他"


def compare_actions(a, b):
    """逐步逐字段精确比对（浮点按 == 比）。返回 {"identical", "first_diff": None | {step, key, old, new}}。"""
    if len(a) != len(b):
        return {"identical": False, "first_diff": {"step": min(len(a), len(b)), "key": "length", "old": len(a), "new": len(b)}}
    for i, (x, y) in enumerate(zip(a, b)):
        for key in COMPARE_KEYS:
            if x.get(key) != y.get(key):
                return {"identical": False, "first_diff": {"step": i, "key": key, "old": x.get(key), "new": y.get(key)}}
    return {"identical": True, "first_diff": None}


# ------------------------------------------------------------------ 读盘
def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def done_routes(suite, cfg, arm):
    """已写 done.json 的 (场景, 路线)。"""
    return {(s, r) for s in cfg["scenes"] for r in routes_of(cfg, s)
            if load_json(os.path.join(suite, arm, s, f"r{r}", "done.json")) is not None}


def load_episodes(suite, cfg, arm):
    """每集一行：metrics.json 的指标 + 交替段（来自 actions.jsonl）。只读已写 done.json 的路线、episode < k 的集
    （冒烟用 K_LIMIT 压低 k 时，旧臂符号链接里的集数比 k 多，不截掉分母会小于分子）。"""
    rows = {}
    for s, r in sorted(done_routes(suite, cfg, arm)):
        for mp in sorted(glob.glob(os.path.join(suite, arm, s, f"r{r}", "ep_*", "metrics.json"))):
            m = load_json(mp)
            if m is None or m["episode"] >= cfg["k"]:
                continue
            d = os.path.dirname(mp)
            seq = [a["action_executed"] for a in jsonl(os.path.join(d, "actions.jsonl"))]
            runs = alt_runs(seq)
            rows[(s, r, m["episode"])] = {
                "arm": arm, "scene": s, "route": r, "episode": m["episode"], "seed": m.get("seed"), "dir": d,
                "success_stop": bool(m["success_stop"]), "collision": bool(m["collision"]),
                "collide_first": bool(m["collide_first"]), "timeout": bool(m["timeout"]), "spl_stop": m.get("spl_stop"),
                "steps": m["steps"], "final_wp_dist_m": m.get("final_wp_dist_m"),
                "alt_runs": runs, "alt_steps": sum(j - i + 1 for i, j in runs),
                "alt_longest": max([j - i + 1 for i, j in runs], default=0)}
    return rows


def outcome_of(r):
    return "到点" if r["success_stop"] else "撞在前" if r["collide_first"] else "超时" if r["timeout"] else "其他"


# ------------------------------------------------------------------ 回放（要 3dgs-venv）
def replay_suite(cfg, arm_eps, tol_by_arm, repo=REPO):
    """arm_eps = {臂: [待回放的行]}，tol_by_arm = {臂: config 里的容差}。每个场景只加载一次占据；容差见 resolve_tol。
    给每行写 row["replay"] = {"mismatch": 复算与记录不一致的步数, "n_steps", "runs": [{"range", "class", "n_turn_err", "n_turn_blocked"}]}。"""
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "sim", "tools"))
    scene_mod = importlib.import_module("sim.3dgs.scene")
    coll = importlib.import_module("sim.3dgs.collision")
    planner_mod = importlib.import_module("sim.3dgs.planner")
    by_scene = {}
    for arm, rows in arm_eps.items():
        for r in rows:
            by_scene.setdefault(r["scene"], []).append(r)
    for scene, rows in by_scene.items():
        path = os.path.join(cfg["scenes_dir"], scene, "scene.json")
        occ = coll.from_scene(scene_mod.load_scene(path))
        occ_planner = planner_mod.OccPlanner(occ)
        for r in rows:
            sc = scene_mod.load_scene(path)
            scene_mod.apply_route(sc, r["route"])
            man = load_json(os.path.join(r["dir"], "episode_manifest.json")) or {}
            tol = resolve_tol(man, tol_by_arm.get(r["arm"]))
            teacher = planner_mod.OccAstarTeacher(occ_planner, sc["waypoint"], tol_deg=tol)
            actions = jsonl(os.path.join(r["dir"], "actions.jsonl"))
            res = replay_branches(teacher, actions)
            runs = []
            for i, j in r["alt_runs"]:
                br = [x["branch"] for x in res[i:j + 1]]
                runs.append({"range": [i, j], "class": classify_run(br),
                             "n_turn_err": br.count("turn_err"), "n_turn_blocked": br.count("turn_blocked")})
            r["replay"] = {"tol_deg": tol, "n_steps": len(res), "mismatch": sum(not x["match"] for x in res), "runs": runs}
        del occ, occ_planner


# ------------------------------------------------------------------ acceptance
def analyze_suite(out, name, replay):
    suite = os.path.join(out, name)
    cfg = load_json(os.path.join(suite, "config.json"))
    new_arm, old_arm = cfg["new_arm"], cfg["old_arm"]
    k = cfg["k"]
    rows, done, summ = {}, {}, {}
    for arm in (new_arm, old_arm):
        done[arm] = done_routes(suite, cfg, arm)
        rows[arm] = load_episodes(suite, cfg, arm)
        summ[arm] = summarize(rows[arm], k * len(done[arm]))
    planned = sum(len(routes_of(cfg, s)) for s in cfg["scenes"])
    res = {"name": name, "config": {x: cfg.get(x) for x in ("scenes", "k", "seed0", "jitter", "hold", "step_frac", "step_pad", "step_cap",
                                                            "teacher_tol_deg", "accept_sr", "accept_timeouts_max", "new_arm", "old_arm")},
           "routes_planned": planned, "routes_done": {a: len(done[a]) for a in done}, "arms": summ,
           "tol_deg": {new_arm: cfg.get("teacher_tol_deg"), old_arm: OLD_TOL_DEG}}
    res["accept_sr"] = cfg.get("accept_sr")
    res["verdict"] = verdict(summ[new_arm]["sr_stop"], cfg["accept_sr"]) if cfg.get("accept_sr") is not None else None
    res["accept_timeouts_max"] = cfg.get("accept_timeouts_max")
    res["verdict_timeouts"] = (verdict_max(summ[new_arm]["timeouts"], cfg["accept_timeouts_max"])
                               if cfg.get("accept_timeouts_max") is not None else None)
    res["complete"] = len(done[new_arm]) == planned
    res["pair"] = pair(rows[new_arm], rows[old_arm]) if rows[new_arm] and rows[old_arm] else None
    res["per_scene"] = {}
    for s in cfg["scenes"]:
        res["per_scene"][s] = {}
        for arm in (new_arm, old_arm):
            routes = {x for x in done[arm] if x[0] == s}
            sel = {kk: v for kk, v in rows[arm].items() if kk[0] == s}
            res["per_scene"][s][arm] = summarize(sel, k * len(routes))
    res["buckets"] = None
    if cfg.get("routes_selected") and os.path.exists(cfg["routes_selected"]):
        rs = load_json(cfg["routes_selected"])
        labels = rs["params"].get("bucket_labels") or sorted(set(route_buckets(rs).values()))
        rm = route_buckets(rs)
        res["buckets"] = {"labels": labels, "tables": {arm: bucket_table(rows[arm], rm, labels, k, done[arm]) for arm in (new_arm, old_arm)},
                          "route_info": {f"{s}|{r}": {"bucket": b} for (s, r), b in rm.items()}}
    alt = {arm: [r for _, r in sorted(rows[arm].items()) if r["alt_runs"]] for arm in (new_arm, old_arm)}
    if replay:
        replay_suite(cfg, alt, res["tol_deg"])
    res["alt_episodes"] = {arm: [{**{kk: v for kk, v in r.items() if kk != "dir"}, "outcome": outcome_of(r)} for r in alt[arm]] for arm in alt}
    return res


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def render_suite(res):
    new, old = res["config"]["new_arm"], res["config"]["old_arm"]
    a_new, a_old = res["arms"][new], res["arms"][old]
    cfgc = res["config"]
    L = [f"## {res['name']}", "",
         f"- 路线 {res['routes_done'][new]}/{res['routes_planned']}（新臂已写 done.json / 计划），每路线 {cfgc['k']} 集，seed = {cfgc['seed0']} + 100·路线 + 集，"
         f"出生扰动 ±{cfgc['jitter']} m，hold {cfgc['hold']} s；新臂 `{new}` 教师容差 {f2(res['tol_deg'][new])}°，旧臂 `{old}` 12°"
         + ("" if res["complete"] else "；**新臂路线没跑完**，下表是部分结果"), "",
         "### 1. 汇总", "",
         "| 教师 | 到点成功 k/n [Wilson 95%] | 超时 | 超时里含交替段 | 含交替段的集（全部） | 碰撞率（任一步 / 撞在前） | SPL(stop) | 平均步数 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, a in ((new, a_new), (old, a_old)):
        ci = a["ci_stop"]
        coll = f2(a["collisions"] / a["n_req"] if a["n_req"] else None)
        first = f2(a["collide_first"] / a["n_req"] if a["n_req"] else None)
        steps = "—" if a["steps_mean"] is None else "%.0f" % a["steps_mean"]
        L.append(f"| `{arm}` | {a['k_stop']}/{a['n_req']} = {f2(a['sr_stop'])} [{f2(ci[0])}, {f2(ci[1])}] | {a['timeouts']} | {a['timeouts_alt']} "
                 f"| {a['episodes_alt']} | {coll} / {first} | {f2(a['spl_stop'])} | {steps} |")
    L += ["", "### 2. 验收（预注册阈值，点估计）", ""]
    if res["accept_sr"] is not None:
        L.append(f"- 新教师到点成功 {a_new['k_stop']}/{a_new['n_req']} = {f2(a_new['sr_stop'])}，阈值 >= {res['accept_sr']:g}：**{res['verdict']}**"
                 f"（旧教师 {a_old['k_stop']}/{a_old['n_req']} = {f2(a_old['sr_stop'])}）。")
    if res["accept_timeouts_max"] is not None:
        L.append(f"- 新教师超时集数 {a_new['timeouts']}，上限 <= {res['accept_timeouts_max']}（旧教师 {a_old['timeouts']} 的 1/4 取整）：**{res['verdict_timeouts']}**；"
                 f"其中含交替段 {a_new['timeouts_alt']} 集。")
    p = res["pair"]
    L += ["", "### 3. 与旧教师逐集配对（到点成功，同场景同路线同 seed，精确 McNemar）", ""]
    if p:
        L += ["| 配对数 | b（新成旧败） | c（新败旧成） | Δ = SR(新) − SR(旧) | p |", "| --- | --- | --- | --- | --- |",
              f"| {p['pairs']} | {p['b']} | {p['c']} | {p['delta']:+.3f} | {p['p']:.3g} |"]
    else:
        L.append("新旧两臂至少有一个没有数据，无配对。")
    if res["buckets"]:
        L += ["", "### 4. 按路线长度分桶（桶 = A* 长度；每格 = 到点 / 请求集数，括号内：超时 / 其中含交替段 / 碰撞）", "",
              f"| 桶 | `{new}` | `{old}` |", "| --- | --- | --- |"]
        for lb in res["buckets"]["labels"]:
            cells = []
            for arm in (new, old):
                t = res["buckets"]["tables"][arm][lb]
                cells.append("—" if not t["n"] else f"{t['k_stop']}/{t['n']} = {t['k_stop'] / t['n']:.2f}（{t['timeouts']} / {t['timeouts_alt']} / {t['collisions']}）")
            L.append(f"| {lb} | " + " | ".join(cells) + " |")
    L += ["", "### 5. 分场景（每格 = 到点 / 请求集数，括号内：超时 / 其中含交替段 / 碰撞）", "", f"| 场景 | `{new}` | `{old}` |", "| --- | --- | --- |"]
    for s, d in res["per_scene"].items():
        cells = []
        for arm in (new, old):
            a = d[arm]
            cells.append("—" if not a["n_req"] else f"{a['k_stop']}/{a['n_req']}（{a['timeouts']} / {a['timeouts_alt']} / {a['collisions']}）")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    for arm, title in ((new, "6. 新教师仍含交替段的集"), (old, "7. 旧教师含交替段的集（对照，只列归因分类）")):
        eps = res["alt_episodes"][arm]
        L += ["", f"### {title}", "", f"共 {len(eps)} 集。"]
        replayed = [e for e in eps if "replay" in e]
        if replayed:
            cls = {}
            for e in replayed:
                for r in e["replay"]["runs"]:
                    cls[r["class"]] = cls.get(r["class"], 0) + 1
            mism = sum(e["replay"]["mismatch"] for e in replayed)
            steps = sum(e["replay"]["n_steps"] for e in replayed)
            L.append(f"回放：复算动作与记录的 action_expert 不一致 {mism}/{steps} 步；交替段共 {sum(cls.values())} 段，按规则分支分类："
                     + "、".join(f"{k} {v}" for k, v in sorted(cls.items())) + "。")
        if arm == new and eps:
            L += ["", "| 场景 r 集 | seed | 结局 | 交替段（起步-止步:长度） | 回放分类（每段） |", "| --- | --- | --- | --- | --- |"]
            for e in eps:
                runs = "、".join(f"{i}-{j}:{j - i + 1}" for i, j in e["alt_runs"])
                cl = "、".join(r["class"] for r in e["replay"]["runs"]) if "replay" in e else "未回放"
                L.append(f"| {e['scene']} r{e['route']} ep{e['episode']} | {e['seed']} | {e['outcome']} | {runs} | {cl} |")
    return L


def acceptance(args):
    suites = args.suites.split(",") if args.suites else [s for s in ("outdoor", "indoor") if os.path.exists(os.path.join(args.out, s, "config.json"))]
    results = {name: analyze_suite(args.out, name, args.replay) for name in suites}
    lines = ["# 3DGS 教师航向容差随转角步推导：验收分析（任务卡 W1-11）", "",
             f"- metric_version `{METRIC_VERSION}`；到点成功 = 首次进入 waypoint 半径早于首次碰撞；碰撞率含到点之后的碰撞，撞在前 = 没到点就撞；"
             "超时 = 既没到点也没撞；SPL(stop) 把成功换成到点成功，分母是请求集数。",
             f"- 交替段 = action_executed 在 {{YAW_LEFT, YAW_RIGHT}} 里严格左右交替、连续 >= {MIN_RUN} 步（sage3d 报告 §4.5 的判据）。",
             "- 来源：`<suite>/<臂>/<场景>/r<N>/ep_*/{metrics.json, actions.jsonl, episode_manifest.json}`；旧臂是旧实验输出目录的只读符号链接。", ""]
    for res in results.values():
        lines += render_suite(res) + [""]
    adir = os.path.join(args.out, "analysis")
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "acceptance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(adir, "acceptance.json"), "w", encoding="utf-8") as f:
        json.dump({"metric_version": METRIC_VERSION, "suites": results}, f, ensure_ascii=False, indent=1)
    print("\n".join(lines))
    return 0


def regress(args):
    new_root, old_root = os.path.abspath(args.new), os.path.abspath(args.old)
    n, same, diffs = 0, 0, []
    for p in sorted(glob.glob(os.path.join(new_root, "*", "r*", "ep_*", "actions.jsonl"))):
        rel = os.path.relpath(p, new_root)
        q = os.path.join(old_root, rel)
        if not os.path.exists(q):
            diffs.append({"episode": rel, "first_diff": {"step": None, "key": "旧目录没有这一集", "old": None, "new": None}})
            n += 1
            continue
        r = compare_actions(jsonl(q), jsonl(p))
        n += 1
        if r["identical"]:
            same += 1
        else:
            diffs.append({"episode": rel, "first_diff": r["first_diff"]})
    print(f"确定性回归：{n} 集，逐步逐字段完全一致 {same} 集，有差异 {len(diffs)} 集（新 {new_root}，旧 {old_root}）")
    for d in diffs[:20]:
        print("  差异", d["episode"], d["first_diff"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"n": n, "identical": same, "diffs": diffs, "new": new_root, "old": old_root}, f, ensure_ascii=False, indent=1)
    return 0 if n and not diffs else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acceptance")
    a.add_argument("out")
    a.add_argument("--suites", default=None, help="逗号分隔；缺省 = outdoor、indoor 里有 config.json 的")
    a.add_argument("--replay", action="store_true", help="对含交替段的集离线回放教师（3dgs-venv，不占 GPU）")
    r = sub.add_parser("regress")
    r.add_argument("new")
    r.add_argument("old")
    r.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return acceptance(args) if args.cmd == "acceptance" else regress(args)


if __name__ == "__main__":
    sys.exit(main())
