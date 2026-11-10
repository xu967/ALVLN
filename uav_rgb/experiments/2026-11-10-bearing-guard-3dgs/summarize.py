#!/usr/bin/env python3
"""bearing-guard-3dgs 汇总：臂 × 场景 × 路线的闭环指标 + 预注册的三个逐集配对检验 -> summary.md / summary.json。

只读 <OUT>/<臂>/<场景>/r<N>/{done.json, ep_*/metrics.json, ep_*/guard.jsonl}（driver 每集落的），不加载场景，随时可跑；
实验没跑完时只统计已写 done.json 的路线（半条路线不进表），表里的 n 标出来。
口径（metric_version 见 METRIC_VERSION）：
  SR(stop)  首次进入 waypoint 半径早于首次碰撞（与 11-09 主列同义；学生没有到点即停，教师有）。主列、三个预注册比较都用它。
  SR(v2)    eval_3dgs 的 3dgs.pilot.v2：过点后再撞也算失败。
  碰撞率    整集任一步碰撞（含到点之后；与 11-09 同）；撞在前 = 没到点就撞了；超时 = 既没到点也没撞。
  SPL(stop) 成功换成 SR(stop) 的 SPL；改写步占比 = 护栏把内层动作改成别的动作的步数 / 总步数，分「到点（或撞）之前」与「整集」
            两个：学生 / 盲前进到点后不停，会在 waypoint 附近绕圈，整集占比被绕圈的步数抬高。
分母 = 已写 done.json 的路线 × 每路线请求集数（失败集算失败，单列）；配对检验只用两臂都有 metrics.json 的 (场景, 路线, seed)。
用法：python summarize.py <OUT>
      python summarize.py <OUT> --gate 臂:阈值     闸门模式：按场景算该臂 SR(stop)，stdout 只打印通过的场景名（空格分隔），
                                                   结果写 <OUT>/gate.json，不写 summary（run.sh 用它决定哪些场景进下一阶段）
可选的 config.json 键（缺省时输出与原版逐字节一致，2026-11-10-sage3d-pilot 共用本脚本）：
  routes           列表 = 所有场景同一批路线；字典 {场景: [路线下标 ...]} = 每个场景各自的路线
  title / overall_label   标题与「四场景合计」的小节名
  main_pairs / extra_pairs  [[编号, A, B], ...]：主比较与补充比较；给了 main_pairs 就不再套用 P1–P3 的内置判决，
                           补充比较缺省为空；判决文字取 verdict_texts[编号] = {"ns", "up", "down"}：
                           p > 0.05 取 ns，p <= 0.05 且 b > c 取 up，p <= 0.05 且 b < c 取 down
  arm_checks       [{"id", "arm", "support_lt", "falsify_ge", "texts": {"support", "falsify", "between"}}]：
                           单臂 SR(stop) 对两个固定阈值的预注册判决
  gates            [{"id"（可选）, "arm", "min_sr", "note"}]：汇总里列出每个场景该臂的 SR(stop) 与是否过闸门；
                           SR 不够的场景写 note，路线没跑完或没有路线的场景写「未放行」与原因
  routes_selected  routes_selected.json 的路径：给了就多输出「按路线长度分桶」表，分路线表带路线名与 A* 长度
"""
import glob
import json
import math
import os
import sys

OUT = os.path.abspath(sys.argv[1])
CFG = json.load(open(os.path.join(OUT, "config.json"), encoding="utf-8"))
METRIC_VERSION = "3dgs.pilot.v2+stop.v1"
ARMS = {a["name"]: a for a in CFG["arms"]}
P_ALPHA = 0.05
# 预注册的三个主比较：(编号, A, B)；其后是不进判决的补充比较
MAIN_PAIRS = [("P1", "forward+route", "forward"), ("P2", "calswa-s1+route", "calswa-s1"),
              ("P3", "calswa-s1+route", "forward+route")]
EXTRA_PAIRS = [("x1", "calswa-s1", "forward"), ("x2", "calswa-s1+goal", "calswa-s1"),
               ("x3", "calswa-s1+goal", "calswa-s1+route"), ("x4", "calswa-s0+route", "forward+route"),
               ("x5", "calswa-s2+route", "forward+route"), ("x6", "teacher", "calswa-s1+route"),
               ("x7", "teacher", "forward+route")]
CUSTOM_PAIRS = "main_pairs" in CFG
if CUSTOM_PAIRS:
    MAIN_PAIRS, EXTRA_PAIRS = [tuple(p) for p in CFG["main_pairs"]], [tuple(p) for p in CFG.get("extra_pairs", [])]
GATE_ARG = sys.argv[sys.argv.index("--gate") + 1] if "--gate" in sys.argv else None


def routes_of(scene):
    return CFG["routes"].get(scene, []) if isinstance(CFG["routes"], dict) else CFG["routes"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


rows, routes_done, failed = [], {}, []        # routes_done[(臂, 场景)] = 已写 done.json 的路线数
done_keys = set()                             # 已写 done.json 的 (臂, 场景, 路线)，分桶表用
for g in ARMS:
    for s in CFG["scenes"]:
        for r in routes_of(s):
            d = os.path.join(OUT, g, s, f"r{r}")
            done = load_json(os.path.join(d, "done.json"))
            if done is None:
                continue
            routes_done[(g, s)] = routes_done.get((g, s), 0) + 1
            done_keys.add((g, s, r))
            failed += [{"arm": g, "scene": s, "route": r, **f} for f in done["failures"]]
            for p in sorted(glob.glob(os.path.join(d, "ep_*", "metrics.json"))):
                m = load_json(p)
                if m is not None:
                    rows.append(m)


def agg(sel, arm, scenes=None):
    """sel = 该臂的 rows（已按条件过滤）；分母 = 该臂在 scenes 里已写 done.json 的路线数 × K。"""
    n_req = CFG["k"] * sum(routes_done.get((arm, s), 0) for s in (scenes or CFG["scenes"]))
    n = len(sel)
    ks, kv = sum(x["success_stop"] for x in sel), sum(x["success_v2"] for x in sel)
    spl = [x["spl_stop"] for x in sel if x["spl_stop"] is not None]
    gs = [x["guard"] for x in sel if x["guard"]]
    steps_g, steps_p = sum(x["n_steps"] for x in gs), sum(x["n_steps_pre"] for x in gs)
    return {"n_req": n_req, "n": n, "failed": n_req - n,
            "k_stop": ks, "sr_stop": ks / n_req if n_req else None, "ci_stop": wilson(ks, n_req),
            "k_v2": kv, "sr_v2": kv / n_req if n_req else None,
            "collision": sum(x["collision"] for x in sel) / n_req if n_req else None,
            "collide_first": sum(x["collide_first"] for x in sel) / n_req if n_req else None,
            "timeout": sum(x["timeout"] for x in sel) / n_req if n_req else None,
            "spl_stop": sum(spl) / n_req if n_req and spl else None,
            "rewrite_frac": sum(x["n_rewritten"] for x in gs) / steps_g if steps_g else None,
            "trigger_frac": sum(x["n_triggered"] for x in gs) / steps_g if steps_g else None,
            "rewrite_pre": sum(x["n_rewritten_pre"] for x in gs) / steps_p if steps_p else None,
            "trigger_pre": sum(x["n_triggered_pre"] for x in gs) / steps_p if steps_p else None,
            "plan_failed": sum(bool(x["plan_failed"]) for x in gs),
            "steps_mean": sum(x["steps"] for x in sel) / n if n else None}


def by_arm(g, **kw):
    return [x for x in rows if x["arm"] == g and all(x[k] == v for k, v in kw.items())]


def gate_eval(arm, min_sr):
    """闸门：按场景算 arm 的 SR(stop)。分母 = config 里该场景的路线数 × K（没跑的 / 失败的集都算失败，与 agg 同口径）；
    路线没跑完（done.json 不全）或没有路线的场景不放行，原因写进 reason。SR >= min_sr 才通过。"""
    if arm not in ARMS:
        sys.exit(f"--gate / gates: 臂 {arm!r} 不在 config.json 的 arms 里")
    scenes = {}
    for s in CFG["scenes"]:
        planned, done = len(routes_of(s)), routes_done.get((arm, s), 0)
        n_req, k = CFG["k"] * planned, sum(x["success_stop"] for x in by_arm(arm, scene=s))
        sr = k / n_req if n_req else None
        kind = "no_routes" if not planned else "incomplete" if done < planned else "low_sr" if sr < min_sr else None
        reason = (None if kind is None else "没有可跑的路线（全部需满电档，或场景没有路线）" if kind == "no_routes" else
                  f"路线没跑完：done.json {done}/{planned} 条" if kind == "incomplete" else f"SR(stop) {sr:.2f} < {min_sr:g}")
        scenes[s] = {"k_stop": k, "n_req": n_req, "sr_stop": sr, "routes_done": done, "routes_planned": planned,
                     "passed": kind is None, "kind": kind, "reason": reason}
    return {"arm": arm, "min_sr": min_sr, "scenes": scenes,
            "passed": [s for s, v in scenes.items() if v["passed"]],
            "failed": [{"scene": s, **v} for s, v in scenes.items() if not v["passed"]]}


if GATE_ARG:
    gate = gate_eval(GATE_ARG.rsplit(":", 1)[0], float(GATE_ARG.rsplit(":", 1)[1]))
    with open(os.path.join(OUT, "gate.json"), "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=1)
    print(" ".join(gate["passed"]))
    sys.exit(0)


def f2(v):
    return "—" if v is None else f"{v:.2f}"


def fmt_row(a):
    if not a["n_req"]:
        return "| 未开始 | | | | | | | | |"
    ci = a["ci_stop"]
    steps = "—" if a["steps_mean"] is None else f"{a['steps_mean']:.0f}"    # 一集都没成功的臂（全是失败集）steps_mean 为 None，不让汇总崩在这里
    return (f"| {a['k_stop']}/{a['n_req']} = {a['sr_stop']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {a['k_v2']}/{a['n_req']} "
            f"| {f2(a['collision'])} | {f2(a['collide_first'])} | {f2(a['timeout'])} | {f2(a['spl_stop'])} "
            f"| {f2(a['rewrite_pre'])} / {f2(a['rewrite_frac'])} | {steps} | {a['n_req']}（失败 {a['failed']}） |")


def label(g):
    a = ARMS[g]
    guard = {None: "无", "route": "oracle 路线方位", "goal": "oracle waypoint 方位"}[a["guard"]]
    return f"`{g}`（{a['base']}，护栏：{guard}）"


RS = None
if CFG.get("routes_selected"):
    RS = load_json(CFG["routes_selected"])
    if RS is None:
        sys.exit(f"config.json 的 routes_selected 读不到：{CFG['routes_selected']}")
RS_ROUTE = {(s, x["route"]): x for s, xs in RS["scenes"].items() for x in xs} if RS else {}


def rlabel(s, r):
    x = RS_ROUTE.get((s, r))
    return f"{s} r{r}" if x is None else f"{s} r{r}（{x['name']}，{x['astar_path_m']:.1f} m）"


overall = {g: agg(by_arm(g), g) for g in ARMS}
lines = ["# " + CFG.get("title", "3DGS 路线方位护栏 × {盲前进, 学生} 判别实验（bearing-guard-3dgs）"), "",
         f"- metric_version `{METRIC_VERSION}`；hold {CFG['hold']} s、cf_sitl、每路线 {CFG['k']} 集、出生扰动 ±{CFG['jitter']} m / ±10°、"
         f"seed = {CFG['seed0']} + 100·路线 + k（同 seed 各臂出生位姿相同，逐集配对）；护栏阈值 {CFG['guard_threshold_deg']}°，路线前视 {CFG['route_lookahead_m']} m。",
         "- SR(stop) = 首次进入 waypoint 半径早于首次碰撞（主列）；SR(v2) = 过点后再撞也算失败；碰撞率含到点之后的碰撞；"
         "撞在前 = 没到点就撞；超时 = 既没到点也没撞；SPL(stop) 把成功换成 SR(stop)；改写 = 护栏把内层动作改成了别的动作"
         "（到点后学生 / 盲前进不停，会在 waypoint 附近绕圈，所以占比分到点或撞之前、整集两个）。",
         "- 护栏用真值位姿和占据 A* 路径（oracle，特权信息）：带 `+route` / `+goal` 的臂不是学生自己的能力。",
         f"- 失败集（算失败、单列）：{len(failed)}" + ("" if not failed else "，见 summary.json `failed`"), ""]
GATES = CFG.get("gates", [])
gate_res = [gate_eval(g["arm"], g["min_sr"]) for g in GATES]
for g, res in zip(GATES, gate_res):
    lines += ["## 0. 闸门" + (f"（预注册 {g['id']}）" if g.get("id") else ""), "",
              f"`{g['arm']}` 的 SR(stop) ≥ {g['min_sr']:g}，按场景判（分母 = 该场景计划路线 × {CFG['k']} 集）；没过的场景不跑其余臂。", "",
              "| 场景 | SR(stop) | 路线 done / 计划 | 判定 |", "| --- | --- | --- | --- |"]
    for s, v in res["scenes"].items():
        sr = "—" if v["sr_stop"] is None else f"{v['k_stop']}/{v['n_req']} = {v['sr_stop']:.2f}"
        verdict_cell = ("通过" if v["passed"] else f"{g.get('note', '闸门未过')}（{v['reason']}）" if v["kind"] == "low_sr"
                        else f"未放行：{v['reason']}")     # note 只对「SR 不够」有意义；没跑完 / 没路线是另一回事
        lines.append(f"| {s} | {sr} | {v['routes_done']}/{v['routes_planned']} | {verdict_cell} |")
    lines.append("")
lines += [f"## 1. {CFG.get('overall_label', '四场景合计')}", "",
          "| 臂 | SR(stop) [Wilson 95%] | SR(v2) | 碰撞率 | 撞在前 | 超时率 | SPL(stop) | 改写步占比（到点或撞前 / 整集） | 平均步数 | n（请求） |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
lines += [f"| {label(g)} " + fmt_row(overall[g]) for g in ARMS]
lines += ["", f"## 2. 分场景 SR(stop)（每格 n = 已完成路线数 × {CFG['k']}，括号内是碰撞率）", "",
          "| 臂 | " + " | ".join(CFG["scenes"]) + " |", "| --- |" + " --- |" * len(CFG["scenes"])]
per_scene = {}
for g in ARMS:
    cells = []
    for s in CFG["scenes"]:
        a = agg(by_arm(g, scene=s), g, [s])
        per_scene[f"{g}|{s}"] = a
        cells.append(f"{a['k_stop']}/{a['n_req']}（{a['collision']:.2f}）" if a["n_req"] else "—")
    lines.append(f"| `{g}` | " + " | ".join(cells) + " |")
lines += ["", f"## 3. 分路线 SR(stop)（每格 = 到点集数 / {CFG['k']}）", "",
          "| 场景 r | " + " | ".join(f"`{g}`" for g in ARMS) + " |", "| --- |" + " --- |" * len(ARMS)]
for s in CFG["scenes"]:
    for r in routes_of(s):
        cells = []
        for g in ARMS:
            sel = by_arm(g, scene=s, route=r)
            cells.append(f"{sum(x['success_stop'] for x in sel)}/{CFG['k']}" if (g, s) in routes_done and
                         load_json(os.path.join(OUT, g, s, f"r{r}", "done.json")) else "—")
        lines.append(f"| {rlabel(s, r)} | " + " | ".join(cells) + " |")

index = {(x["arm"], x["scene"], x["route"], x["seed"]): x for x in rows}


def paired(a, b):
    keys = [k[1:] for k in index if k[0] == a and (b, *k[1:]) in index]
    bb = sum(index[(a, *k)]["success_stop"] and not index[(b, *k)]["success_stop"] for k in keys)
    cc = sum(not index[(a, *k)]["success_stop"] and index[(b, *k)]["success_stop"] for k in keys)
    n = len(keys)
    delta = (sum(index[(a, *k)]["success_stop"] for k in keys) - sum(index[(b, *k)]["success_stop"] for k in keys)) / n if n else None
    return {"a": a, "b": b, "pairs": n, "b_count": bb, "c_count": cc, "p": exact_mcnemar(bb, cc), "delta_sr": delta}


def verdict(pid, t):
    if t["pairs"] == 0:
        return "无配对数据"
    if CUSTOM_PAIRS:    # 文字来自 config 的 verdict_texts；没写的编号（补充比较）不出判决
        tx = CFG.get("verdict_texts", {}).get(pid)
        return "" if not tx else tx["ns"] if t["p"] > P_ALPHA else tx["up"] if t["b_count"] > t["c_count"] else tx["down"]
    up, p = t["b_count"] > t["c_count"], t["p"]
    if pid == "P1":
        return ("支持预测（b>c 且 SR 合计高 ≥ 0.15）" if up and t["delta_sr"] >= 0.15 else
                "方向符合预测但幅度不足 0.15" if up else "证伪（A 没有比 B 高）：路线方向不是盲飞失败的主因")
    if pid == "P2":
        return "支持预测（b>c）" if up else "证伪（b≤c）：路线方向对学生没有帮助"
    if pid == "P3":
        return ("支持预测（p>0.05，无可检出差别）" if p > P_ALPHA else
                "证伪（学生+护栏显著好于盲飞+护栏）：柱阵感知有一部分迁移" if up else
                "p<0.05 但方向相反（学生+护栏显著差于盲飞+护栏；预注册没写这个方向）")
    return ""


pairs = [{"id": i, **paired(a, b)} for i, a, b in MAIN_PAIRS + EXTRA_PAIRS]
lines += ["", "## 4. 逐集配对（SR(stop)，同场景同路线同 seed，精确 McNemar；b = A 成 B 败，c = A 败 B 成；Δ = SR(A) − SR(B)）", "",
          "| 编号 | A vs B | 配对数 | b | c | Δ | p | 预注册判决 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for t in pairs:
    d = "—" if t["delta_sr"] is None else f"{t['delta_sr']:+.3f}"
    lines.append(f"| {t['id']} | `{t['a']}` vs `{t['b']}` | {t['pairs']} | {t['b_count']} | {t['c_count']} | {d} | {t['p']:.3f} "
                 f"| {verdict(t['id'], t)} |")
ARM_CHECKS = CFG.get("arm_checks", [])
arm_res = []
if ARM_CHECKS:
    lines += ["", "单臂预注册判决（SR(stop) 对固定阈值；分母 = 该臂已完成路线 × 每路线集数）：", "",
              "| 编号 | 臂 | SR(stop) [Wilson 95%] | 支持：SR < | 证伪：SR >= | 判决 |", "| --- | --- | --- | --- | --- | --- |"]
    for c in ARM_CHECKS:
        a = overall[c["arm"]]
        tx = {"support": "支持预测", "falsify": "证伪", "between": "介于预测与证伪之间", **c.get("texts", {})}
        v = ("无数据" if not a["n_req"] else tx["support"] if a["sr_stop"] < c["support_lt"] else
             tx["falsify"] if a["sr_stop"] >= c["falsify_ge"] else tx["between"])
        arm_res.append({**c, "k_stop": a["k_stop"], "n_req": a["n_req"], "sr_stop": a["sr_stop"], "verdict": v})
        cell = "—" if not a["n_req"] else f"{a['k_stop']}/{a['n_req']} = {a['sr_stop']:.2f} [{a['ci_stop'][0]:.2f}, {a['ci_stop'][1]:.2f}]"
        lines.append(f"| {c['id']} | `{c['arm']}` | {cell} | {c['support_lt']:g} | {c['falsify_ge']:g} | {v} |")


def id_range(ps):
    return ps[0][0] if len(ps) == 1 else f"{ps[0][0]}–{ps[-1][0]}"


lines += ["", f"{id_range(MAIN_PAIRS)} 是预注册的主比较（见 run.sh 文件头）；" + (f"{id_range(EXTRA_PAIRS)} 是补充比较，不进判决。" if EXTRA_PAIRS else "")
          + "判决由脚本按预注册规则机械给出，结论以主会话为准。"]

lines += ["", "## 5. 护栏统计与出生位姿配对检查", "",
          "| 臂 | 触发步占比（|误差| ≥ 阈值；到点或撞前 / 整集） | 改写步占比（到点或撞前 / 整集） | 路线规划失败的集数 |", "| --- | --- | --- | --- |"]
for g in ARMS:
    if ARMS[g]["guard"]:
        a = overall[g]
        lines.append(f"| `{g}` | {f2(a['trigger_pre'])} / {f2(a['trigger_frac'])} | {f2(a['rewrite_pre'])} / {f2(a['rewrite_frac'])} "
                     f"| {a['plan_failed']} |")
worst = 0.0
for k in {k[1:] for k in index}:
    poses = [index[(g, *k)]["start_pose"] for g in ARMS if (g, *k) in index]
    for q in poses[1:]:
        worst = max(worst, abs(q["x"] - poses[0]["x"]), abs(q["y"] - poses[0]["y"]), abs(q["yaw_deg"] - poses[0]["yaw_deg"]) / 100.0)
lines += ["", f"同 seed 各臂起飞后首步位姿的最大差（位置 m，航向 °/100 取大者）：{worst:.2e}（≈0 说明逐集配对的出生位姿一致）。"]
buckets = {}
if RS:
    labels = RS["params"].get("bucket_labels") or sorted({x["bucket"] for x in RS_ROUTE.values()})
    lines += ["", "## 6. 按路线长度分桶（桶 = A* 长度 sage.astar_path_m，来自 routes_selected.json；每格 = 到点集数 / 请求集数，只统计已写 done.json 的路线）", "",
              "| 桶（计划路线数） | " + " | ".join(f"`{g}`" for g in ARMS) + " |", "| --- |" + " --- |" * len(ARMS)]
    for b in labels:
        keys = [k for k, x in RS_ROUTE.items() if x["bucket"] == b]
        cells = []
        for g in ARMS:
            done_r = {k for k in keys if (g, *k) in done_keys}
            n_b, k_b = CFG["k"] * len(done_r), sum(x["success_stop"] for x in rows if x["arm"] == g and (x["scene"], x["route"]) in done_r)
            buckets[f"{b}|{g}"] = {"k_stop": k_b, "n_req": n_b}
            cells.append(f"{k_b}/{n_b} = {k_b / n_b:.2f}" if n_b else "—")
        lines.append(f"| {b}（{len(keys)}） | " + " | ".join(cells) + " |")
lines += ["", "来源：本目录 `<臂>/<场景>/r<N>/ep_*/{metrics.json,guard.jsonl,actions.jsonl,oracle.jsonl}`；输入 sha 见 `inputs.txt`。"]
with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
summary = {"metric_version": METRIC_VERSION, "config": CFG, "overall": overall, "per_scene": per_scene,
           "pairs": pairs, "failed": failed, "routes_done": {f"{g}|{s}": n for (g, s), n in routes_done.items()},
           "spawn_pair_max_diff": worst}
for key, val in (("gates", gate_res), ("arm_checks", arm_res), ("buckets", buckets)):
    if val:     # 可选小节：没配置时不加键，bearing-guard 的 summary.json 与原版一致
        summary[key] = val
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("\n".join(lines))
