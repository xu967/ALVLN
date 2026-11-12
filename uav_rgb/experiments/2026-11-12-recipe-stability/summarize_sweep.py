#!/usr/bin/env python3
"""工作点扫描（任务卡 E2E-R1 第二部分）的汇总：读 run_closed_loop.sh 的输出目录，写 summary.md 与 summary.json。零依赖（只用标准库）。

口径（预注册见 run_closed_loop.sh 文件头，这里不改）：
  · 三档 = 现有校准值上再加 −0.5 / 0 / +0.5 nat 的 FORWARD 偏置；每档 3 个训练种子 × 20 个几何 = 60 集；分母永远是请求数（20 / 臂，60 / 档），
    含迟到步的集（基础设施失败）与采集中止的集都不算成功，占分母。
  · 含迟到步的集按 run_r2.sh 条款 R6 重跑一次（<臂>-retry1，只含迟到的 seed）；汇总以重跑结果为准，重跑仍迟到 / 中止则按没成功，首次记录另列。
    采集中止（不是迟到）是测量结果，不重跑。有任何重跑的臂在表里标「重评」。
  · 单集归类：成功 / 撞柱 / 越界（无碰撞、出过 |y|≤3 m 走廊，没成功）/ 未到线（走廊内但步数用完）；另有「迟到」「中止」两类基础设施失败。
  · 预注册判定一：三档成功总数最高与最低相差 ≥ 8 集 → 工作点是闭环上要显式调的量；< 4 集 → 在 ±0.5 nat 内闭环对它不敏感；4–7 集 → 两条线之间，不下结论。
  · 预注册判定二（只在有「配方臂」时）：三个训练种子的成功数极差 ≤ 3，并且总数 ≥ 同批几何上 0 档的总数 − 3。
  · 检验（公共规则 §2，交叉审核 D2）：主检验按几何聚类：同一个几何上三个训练种子的成败差先加总，再对 20 个几何做符号翻转精确检验（零差的几何不进检验），
    区间 = 按几何重抽样的 bootstrap（百分位法，B=10000，固定随机种子）；逐集配对精确 McNemar 与 Wilson 并列，只作参考；两者结论不一致时照实写。
    不做多重比较校正：三个两两比较各报原始 p。

用法：
  summarize_sweep.py <OUT> [--note 文字]            写 <OUT>/summary.md 与 summary.json
  summarize_sweep.py --late-seeds <臂目录>          打印该臂 eval.json 里含迟到步的 seed（逗号分隔；没有则空）
"""
import argparse
import json
import math
import os
import random
import sys
from math import comb

# ───────────────────────────── 口径常量（预注册，不改） ─────────────────────────────
LEVELS = (("m0.5", -0.5), ("0", 0.0), ("p0.5", 0.5))   # 档位 key 与偏置平移（nat）；key 也是臂名的一部分
TRAIN_SEEDS = (0, 1, 2)
REQUESTED = 20                       # 每臂请求集数 = 冻结几何个数
SPREAD_SENSITIVE = 8                 # 三档最高与最低相差 ≥ 8 集：要显式调
SPREAD_INSENSITIVE = 4               # 相差 < 4 集：不敏感
EXTRA_RANGE_MAX = 3                  # 配方臂：三个训练种子成功数极差上限
SUBST_TAGS = ('R2e', 'R3e')          # 替代臂：卡里的 R2、R3 被训练器拒绝零权重，SAFE 权重改 1e-6（run_offline.sh 文件头）
EXTRA_TOTAL_MARGIN = 3               # 配方臂：总数不低于 0 档总数减这个数
BOOT_B = 10000
BOOT_SEED = 20261112
RETRY_SUFFIX = "-retry1"
ACTION_NAMES = {"0": "S", "1": "F", "2": "L", "3": "R"}   # actions 字典的键：0 SAFE、1 FORWARD_SLOW、2 YAW_LEFT、3 YAW_RIGHT
OUTCOMES = ("success", "collision", "corridor", "not_reached")
OUTCOME_ZH = {"success": "成功", "collision": "撞柱", "corridor": "越界", "not_reached": "未到线", "late": "迟到", "aborted": "中止"}
# ──────────────────────────────────────────────────────────────────────────────────


def level_label(delta):
    return f"{delta:+g} nat" if delta else "0（原校准）"


def lvl_arm(key, seed):
    return f"lvl-{key}-s{seed}"


def extra_arm(tag, seed):
    return f"x-{tag}-s{seed}"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 统计
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """配对精确 McNemar（二项双侧）；b = A 成功 B 失败，c = A 失败 B 成功。"""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def signflip_exact_p(diffs):
    """符号翻转精确检验，双侧：T = Σ ε_g d_g，ε_g 独立等概率 ±1；p = P(|T| ≥ |T_obs|)。零差的几何不影响分布，直接略去。
    用整数动态规划（和的取值范围 ≤ Σ|d|），不枚举 2^m。全是零差返回 1.0。"""
    d = [abs(int(x)) for x in diffs if int(x) != 0]
    if not d:
        return 1.0
    total = sum(d)
    obs = abs(sum(int(x) for x in diffs))
    dist = {0: 1}
    for v in d:
        nxt = {}
        for s, c in dist.items():
            nxt[s + v] = nxt.get(s + v, 0) + c
            nxt[s - v] = nxt.get(s - v, 0) + c
        dist = nxt
    hit = sum(c for s, c in dist.items() if abs(s) >= obs)
    return hit / 2 ** len(d)


def bootstrap_ci(diffs, b=BOOT_B, seed=BOOT_SEED, alpha=0.05):
    """按几何重抽样：每次从 G 个几何里有放回抽 G 个，对它们的 d_g 求和（= 这批几何上的成功数差），取百分位区间。返回 (低, 高)。"""
    g = len(diffs)
    if g == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    sums = sorted(sum(diffs[rng.randrange(g)] for _ in range(g)) for _ in range(b))
    lo = sums[int(math.floor(alpha / 2 * b))]
    hi = sums[min(b - 1, int(math.ceil((1 - alpha / 2) * b)) - 1)]
    return (lo, hi)


# ---------------------------------------------------------------- 读盘
def outcome(row):
    if row.get("success"):
        return "success"
    if row.get("collision"):
        return "collision"
    if row.get("corridor_violation"):
        return "corridor"
    return "not_reached"


def load_eval(edir):
    """<臂>/eval.json → {partial, done, failed_count, requested, rows{seed: 行}, failed{seed: 项}, late{seed: 项}}；没有文件返回 None。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
    try:
        e = read_json(path)
    except (OSError, ValueError):
        return None
    late = {int(x["seed"]): x for x in e.get("late_dispatch_episodes") or []}
    return {"partial": bool(e.get("partial")), "done": int(e.get("episodes_done") or 0), "failed_count": int(e.get("failed_count") or 0),
            "requested": int(e.get("episodes_requested") or 0), "ckpt": e.get("ckpt"), "ckpt_sha256": e.get("ckpt_sha256"),
            "action_latency_s": e.get("action_latency_s"), "metric_version": e.get("metric_version"), "split": e.get("split"),
            "rows": {int(x["seed"]): x for x in e.get("per_episode") or []},
            "failed": {int(x["seed"]): x for x in e.get("failed") or [] if "seed" in x}, "late": late, "wall_s": e.get("wall_s")}


def late_seeds(edir):
    """eval.json 里含迟到步的 seed（升序）；没有文件返回 []。"""
    ev = load_eval(edir)
    return sorted(ev["late"]) if ev else []


def read_cg(out, name):
    """<臂>.cg：两行「start|end epoch usage_usec nr_throttled throttled_usec」→ {wall_s, avg_cores, throttled, throttled_s}；缺 end 或没有文件返回 None。"""
    path = os.path.join(out, name + ".cg")
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 5 and p[0] in ("start", "end"):
                rows[p[0]] = [float(x) for x in p[1:]]
    if "start" not in rows or "end" not in rows:
        return None
    s, e = rows["start"], rows["end"]
    wall = e[0] - s[0]
    return {"wall_s": wall, "avg_cores": (e[1] - s[1]) / 1e6 / wall if wall > 0 else float("nan"),
            "throttled": int(e[2] - s[2]), "throttled_s": (e[3] - s[3]) / 1e6}


def merged_arm(out, name, seeds, requested=REQUESTED):
    """一个臂含重跑的逐集结果。返回 None（首次没有 eval.json）或：
      status   "none"（还没有 eval.json）| "partial"（被中断）| "incomplete"（done+failed < 请求数）| "complete"
      eps      {seed: {state, row, first}}，state ∈ first（首次干净）| retry（首次迟到，重跑干净）| late（迟到且重跑仍迟到 / 重跑中止 / 没重跑）| aborted（采集中止，不是迟到）| missing（没有记录）
      reran    是否有 <臂>-retry1 目录
    """
    first = load_eval(os.path.join(out, name))
    if first is None:
        return {"status": "none", "eps": {s: {"state": "missing", "row": None, "first": "missing"} for s in seeds}, "reran": False, "ev": None, "retry_ev": None}
    retry = load_eval(os.path.join(out, name + RETRY_SUFFIX))
    eps = {}
    for s in seeds:
        if s in first["rows"]:
            eps[s] = {"state": "first", "row": first["rows"][s], "first": "clean"}
        elif s in first["late"]:
            r = {"state": "late", "row": None, "first": "late"}
            if retry is not None:
                if s in retry["rows"]:
                    r.update(state="retry", row=retry["rows"][s])
            eps[s] = r
        elif s in first["failed"]:
            eps[s] = {"state": "aborted", "row": None, "first": "aborted"}
        else:
            eps[s] = {"state": "missing", "row": None, "first": "missing"}
    if first["partial"]:
        status = "partial"
    elif first["done"] + first["failed_count"] < requested:
        status = "incomplete"
    else:
        status = "complete"
    return {"status": status, "eps": eps, "reran": retry is not None, "ev": first, "retry_ev": retry}


def success_of(item):
    """这一集是否成功：只有干净（首次或重跑）并且 success 才算；其余按没成功（占分母）。"""
    return bool(item["row"] and item["row"].get("success"))


def outcome_of(item):
    if item["row"] is not None:
        return outcome(item["row"])
    return {"late": "late", "aborted": "aborted"}.get(item["state"], "aborted")


# ---------------------------------------------------------------- 档位与臂的聚合
def load_group(out, names_by_seed, seeds):
    """一个「档 / 臂组」（三个训练种子的臂）→ {arms{训练种子: merged}, complete, matrix{训练种子: {eval_seed: bool}}}"""
    arms = {s: merged_arm(out, names_by_seed[s], seeds) for s in TRAIN_SEEDS}
    complete = all(a["status"] == "complete" for a in arms.values())
    matrix = {s: {g: success_of(it) for g, it in a["eps"].items()} for s, a in arms.items()}
    return {"names": names_by_seed, "arms": arms, "complete": complete, "matrix": matrix}


def group_counts(grp, seeds):
    """success 总数、各训练种子数、各类结局数、动作占比（只用干净的集）。"""
    per_seed = {s: sum(grp["matrix"][s].values()) for s in TRAIN_SEEDS}
    oc = {k: 0 for k in list(OUTCOMES) + ["late", "aborted"]}
    act = {k: 0 for k in ACTION_NAMES}
    steps, clean = 0, 0
    for s in TRAIN_SEEDS:
        for it in grp["arms"][s]["eps"].values():
            oc[outcome_of(it)] += 1
            if it["row"] is not None:
                clean += 1
                steps += int(it["row"].get("steps") or 0)
                for k, v in (it["row"].get("actions") or {}).items():
                    act[str(k)] = act.get(str(k), 0) + int(v)
    tot_act = sum(act.values())
    shares = {ACTION_NAMES.get(k, k): (v / tot_act if tot_act else float("nan")) for k, v in act.items()}
    rng = max(per_seed.values()) - min(per_seed.values())
    return {"total": sum(per_seed.values()), "per_seed": per_seed, "range": rng, "outcomes": oc, "action_shares": shares,
            "action_counts": act, "steps": steps, "clean_episodes": clean, "n": len(TRAIN_SEEDS) * len(seeds)}


def geom_sums(grp, seeds):
    """{几何 seed: 三个训练种子上的成功数（0–3）}"""
    return {g: sum(int(grp["matrix"][s].get(g, False)) for s in TRAIN_SEEDS) for g in seeds}


def compare(grp_a, grp_b, seeds):
    """A 对 B 的配对比较：成功数差、几何聚类符号翻转 p、bootstrap 区间、逐集 McNemar。差 = A − B。"""
    ga, gb = geom_sums(grp_a, seeds), geom_sums(grp_b, seeds)
    d = [ga[g] - gb[g] for g in seeds]
    b = sum(1 for s in TRAIN_SEEDS for g in seeds if grp_a["matrix"][s].get(g) and not grp_b["matrix"][s].get(g))
    c = sum(1 for s in TRAIN_SEEDS for g in seeds if not grp_a["matrix"][s].get(g) and grp_b["matrix"][s].get(g))
    p_cl, p_mc = signflip_exact_p(d), mcnemar_exact(b, c)
    lo, hi = bootstrap_ci(d)
    return {"diff": sum(d), "geoms_nonzero": sum(1 for x in d if x), "geoms_pos": sum(1 for x in d if x > 0), "geoms_neg": sum(1 for x in d if x < 0),
            "p_cluster": p_cl, "ci": [lo, hi], "mcnemar_b": b, "mcnemar_c": c, "p_mcnemar": p_mc,
            "agree": (p_cl < 0.05) == (p_mc < 0.05)}


# ---------------------------------------------------------------- 预注册判定
def verdict_spread(totals):
    """totals：{档 key: 成功总数}；三个都有才判。返回 (code, 文字)。"""
    if len(totals) < len(LEVELS) or any(v is None for v in totals.values()):
        return "incomplete", "三档没有全部跑完，未判定"
    hi, lo = max(totals.values()), min(totals.values())
    spread = hi - lo
    best = [k for k, v in totals.items() if v == hi]
    s = "、".join(f"{level_label(dict(LEVELS)[k])} {v}/60" for k, v in totals.items())
    if spread >= SPREAD_SENSITIVE:
        return "sensitive", f"三档成功数 {s}；最高与最低相差 {spread} 集 ≥ {SPREAD_SENSITIVE}：工作点在闭环上是一个要显式调的量（最高档：{'、'.join(level_label(dict(LEVELS)[k]) for k in best)}）"
    if spread < SPREAD_INSENSITIVE:
        return "insensitive", f"三档成功数 {s}；最高与最低相差 {spread} 集 < {SPREAD_INSENSITIVE}：在正负 0.5 nat 之内闭环对工作点不敏感"
    return "between", f"三档成功数 {s}；最高与最低相差 {spread} 集，落在 {SPREAD_INSENSITIVE}–{SPREAD_SENSITIVE - 1} 之间：两条预注册线都没覆盖，如实记录，不下「敏感」或「不敏感」的结论"


def verdict_extra(cnt_extra, total0):
    """配方臂：三个训练种子成功数极差 ≤ 3 并且总数 ≥ 0 档总数 − 3。返回 (通过?, 文字)。"""
    ok_range = cnt_extra["range"] <= EXTRA_RANGE_MAX
    ok_total = cnt_extra["total"] >= total0 - EXTRA_TOTAL_MARGIN
    ps = "/".join(str(cnt_extra["per_seed"][s]) for s in TRAIN_SEEDS)
    txt = (f"三个训练种子成功数 {ps}，极差 {cnt_extra['range']}（{'≤' if ok_range else '>'} {EXTRA_RANGE_MAX}，{'过' if ok_range else '不过'}）；"
           f"总数 {cnt_extra['total']}，同批几何上 0 档 {total0}，下限 {total0 - EXTRA_TOTAL_MARGIN}（{'过' if ok_total else '不过'}）")
    return ok_range and ok_total, txt


# ---------------------------------------------------------------- 报告
def read_kv(path):
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    cfg[k] = v
    return cfg


def fmt_ci(ci):
    return f"[{ci[0]:+g}, {ci[1]:+g}]"


def extra_desc(tag):
    """概览里「配方臂」一行的说明；替代臂要写明来历（卡里没有规定替代臂的加臂）。"""
    if not tag:
        return "无（第一部分没有过线的臂，只做工作点扫描）"
    if tag in SUBST_TAGS:
        return (f"{tag}（替代臂：卡里的 R2、R3 被训练器拒绝零权重，SAFE 权重改 1e-6。第一部分里过线的只有替代臂；卡里没有规定替代臂的加臂，"
                "这个臂是执行者看到第一部分结果后按同一规则加的，主会话可以不采用。末 10 轮平均加校准）")
    return f"{tag}（第一部分过线者的末 10 轮平均加校准）"


def pct(x):
    return "—" if x != x else f"{100 * x:.1f}%"


def build(out, note=None):
    cfg = read_kv(os.path.join(out, "config.env"))
    start = read_kv(os.path.join(out, "start.txt"))
    seeds = [int(x) for x in cfg.get("SEEDS", "").split(",") if x.strip()]
    extra_tag = cfg.get("EXTRA_ARM") or ""
    extra_name = f"配方臂 {extra_tag}" + ("（替代）" if extra_tag in SUBST_TAGS else "")
    res = {"schema": "e2e-r1.sweep_summary.v1", "seeds": seeds, "extra_tag": extra_tag, "levels": {}, "tests": {}, "verdicts": {}}
    lines = ["# E2E-R1 第二部分：工作点扫描（闭环，val 分区 20 个几何）", ""]
    if note:
        lines += [f"> {note}", ""]
    if not seeds:
        lines += ["config.env 里没有 SEEDS，没有可汇总的内容。", ""]
        return "\n".join(lines), res
    groups = {k: load_group(out, {s: lvl_arm(k, s) for s in TRAIN_SEEDS}, seeds) for k, _ in LEVELS}
    extra = load_group(out, {s: extra_arm(extra_tag, s) for s in TRAIN_SEEDS}, seeds) if extra_tag else None
    cnt = {k: group_counts(g, seeds) for k, g in groups.items()}
    cnt_x = group_counts(extra, seeds) if extra else None
    gid_of = {}
    for g in list(groups.values()) + ([extra] if extra else []):
        for a in g["arms"].values():
            for sd, it in a["eps"].items():
                if it["row"] is not None and it["row"].get("geometry_gid"):
                    gid_of[sd] = it["row"]["geometry_gid"]
    done_arms = sum(1 for g in list(groups.values()) + ([extra] if extra else []) for a in g["arms"].values() if a["status"] == "complete")
    n_arms = len(groups) * len(TRAIN_SEEDS) + (len(TRAIN_SEEDS) if extra else 0)

    # 1 概览
    lines += ["## 1. 概览", "",
              f"- 臂完成 {done_arms}/{n_arms}（每臂请求 {REQUESTED} 集 = 冻结的 {len(seeds)} 个 val 几何；含重跑后统计）。",
              f"- 协议：`--action-latency {cfg.get('L_STAR', '?')}`，`--task base --steps 80 --split val`，不带 `--heading-guard`，cf_sitl / e1_mini / observation 窗口；口径 `sim09.v2.coast1s.corridor3m`。",
              f"- 评测 seed（冻结，升序）：{','.join(str(s) for s in seeds)}",
              f"- 代码版本：{cfg.get('CODE_HEAD', '未记录')}；起跑：{start.get('started_at', '未记录')}（等机器锁 {start.get('lock_wait_s', '?')} s）。",
              f"- 配方臂：{extra_desc(extra_tag)}。", ""]

    # 2 判定
    code, text = verdict_spread({k: (cnt[k]["total"] if groups[k]["complete"] else None) for k, _ in LEVELS})
    res["verdicts"]["spread"] = {"code": code, "text": text}
    lines += ["## 2. 预注册判定", "", f"- **判定一（工作点敏感度）**：{text}。"]
    if extra:
        if extra["complete"] and groups["0"]["complete"]:
            ok, txt = verdict_extra(cnt_x, cnt["0"]["total"])
            res["verdicts"]["extra"] = {"pass": ok, "text": txt}
            lines.append(f"- **判定二（{extra_name}）**：{'过线' if ok else '不过线'}。{txt}。")
        else:
            res["verdicts"]["extra"] = {"pass": None, "text": "配方臂或 0 档没有跑完，未判定"}
            lines.append(f"- **判定二（{extra_name}）**：配方臂或 0 档没有跑完，未判定。")
    lines.append("")

    # 3 三档对比
    lines += ["## 3. 三档对比（成功数 / 60；分母 = 请求集数，迟到与中止的集占分母、不算成功）", "",
              "| 档（FORWARD 偏置） | 成功 /60 | Wilson 95%（参考） | s0 / s1 / s2 | 极差 | 撞柱 | 越界 | 未到线 | 迟到 | 中止 | 动作占比 S/F/L/R | 平均步数 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def level_row(label, grp, c):
        if not grp["complete"]:
            n_have = sum(1 for a in grp["arms"].values() if a["status"] == "complete")
            return f"| {label} | 未完成（{n_have}/3 个臂） | | | | | | | | | | |"
        lo, hi = wilson(c["total"], c["n"])
        o = c["outcomes"]
        sh = c["action_shares"]
        shares = "/".join(pct(sh.get(k, float('nan'))) for k in ("S", "F", "L", "R"))
        mean_steps = c["steps"] / c["clean_episodes"] if c["clean_episodes"] else float("nan")
        return (f"| {label} | **{c['total']}** | {100 * lo:.0f}–{100 * hi:.0f}% | {' / '.join(str(c['per_seed'][s]) for s in TRAIN_SEEDS)} | {c['range']} "
                f"| {o['collision']} | {o['corridor']} | {o['not_reached']} | {o['late']} | {o['aborted']} | {shares} | {mean_steps:.1f} |")

    for k, d in LEVELS:
        lines.append(level_row(level_label(d), groups[k], cnt[k]))
    if extra:
        lines.append(level_row(f"{extra_name}（末 10 轮平均 + 校准，无平移）", extra, cnt_x))
    lines.append("")
    for k, d in LEVELS:
        res["levels"][k] = {"delta": d, "complete": groups[k]["complete"], **({kk: vv for kk, vv in cnt[k].items()} if groups[k]["complete"] else {})}
    if extra:
        res["extra"] = {"tag": extra_tag, "complete": extra["complete"], **(cnt_x if extra["complete"] else {})}

    # 4 检验
    pairs = [(("m0.5", "0"), "−0.5 对 0"), (("p0.5", "0"), "+0.5 对 0"), (("m0.5", "p0.5"), "−0.5 对 +0.5")]
    lines += ["## 4. 两两比较（差 = 前者 − 后者，单位：60 集里的成功数）", "",
              "主检验：按几何聚类的符号翻转精确检验（20 个几何各自把三个训练种子的成败差加总），区间 = 按几何重抽样 bootstrap（B=10000）；"
              "逐集配对精确 McNemar 并列，只作参考（把 60 集当独立样本，会高估证据）。不做多重比较校正。", "",
              "| 比较 | 差 | 差≠0 的几何（正 / 负） | p（几何聚类） | 95% 区间（bootstrap） | McNemar b / c | p（逐集 McNemar） | 两种检验结论 |", "|---|---|---|---|---|---|---|---|"]
    for pair, name in pairs:
        a, b = pair
        if groups[a]["complete"] and groups[b]["complete"]:
            r = compare(groups[a], groups[b], seeds)
            res["tests"][f"{a}_vs_{b}"] = r
            lines.append(f"| {name} | {r['diff']:+d} | {r['geoms_nonzero']}（{r['geoms_pos']} / {r['geoms_neg']}） | {r['p_cluster']:.4f} | {fmt_ci(r['ci'])} | {r['mcnemar_b']} / {r['mcnemar_c']} | {r['p_mcnemar']:.4f} | {'一致' if r['agree'] else '不一致（按 0.05 线）'} |")
        else:
            lines.append(f"| {name} | 未完成 | | | | | | |")
    if extra:
        if extra["complete"] and groups["0"]["complete"]:
            r = compare(extra, groups["0"], seeds)
            res["tests"]["extra_vs_0"] = r
            lines.append(f"| {extra_name} 对 0 档 | {r['diff']:+d} | {r['geoms_nonzero']}（{r['geoms_pos']} / {r['geoms_neg']}） | {r['p_cluster']:.4f} | {fmt_ci(r['ci'])} | {r['mcnemar_b']} / {r['mcnemar_c']} | {r['p_mcnemar']:.4f} | {'一致' if r['agree'] else '不一致（按 0.05 线）'} |")
        else:
            lines.append(f"| {extra_name} 对 0 档 | 未完成 | | | | | | |")
    lines.append("")

    # 5 逐几何
    head = ["| 几何 seed | gid |"] + [f" {level_label(d)} |" for _, d in LEVELS] + ([f" {extra_name} |"] if extra else [])
    lines += ["## 5. 逐几何成功数（0–3：三个训练种子里成功的个数）", "", "".join(head), "|" + "---|" * (2 + len(LEVELS) + (1 if extra else 0))]
    sums = {k: geom_sums(groups[k], seeds) for k, _ in LEVELS}
    sums_x = geom_sums(extra, seeds) if extra else None
    for sd in seeds:
        cells = [f"{sums[k][sd]}" if groups[k]["complete"] else "—" for k, _ in LEVELS]
        if extra:
            cells.append(f"{sums_x[sd]}" if extra["complete"] else "—")
        lines.append(f"| {sd} | {gid_of.get(sd, '')[:8]} | " + " | ".join(cells) + " |")
    lines.append("")

    # 6 逐臂明细
    lines += ["## 6. 逐臂明细", "",
              "| 臂 | 状态 | 成功 | 迟到集 | 重跑 | 中止 | 墙钟 s | 平均核数 | 限流次数增量 |", "|---|---|---|---|---|---|---|---|---|"]
    all_groups = [(k, groups[k]) for k, _ in LEVELS] + ([("x", extra)] if extra else [])
    for _k, g in all_groups:
        for s in TRAIN_SEEDS:
            name, a = g["names"][s], g["arms"][s]
            if a["status"] == "none":
                lines.append(f"| {name} | 未开始 | | | | | | | |")
                continue
            ev = a["ev"]
            first_late = len(ev["late"])
            reran = "重评" if a["reran"] else "—"
            aborted = sum(1 for it in a["eps"].values() if it["state"] == "aborted")
            ok = sum(1 for it in a["eps"].values() if success_of(it))
            cg = read_cg(out, name)
            cgtxt = (f"{cg['avg_cores']:.1f} | {cg['throttled']}{' ⚠' if cg['throttled'] > 0 else ''}") if cg else "— | —"
            wall = f"{ev['wall_s']:.0f}" if ev.get("wall_s") is not None else "—"
            lines.append(f"| {name} | {a['status']} | {ok}/{len(seeds)} | {first_late} | {reran} | {aborted} | {wall} | {cgtxt} |")
    lines += ["", "⚠ = 该臂运行期间容器 cgroup 的 nr_throttled 增加过（CPU 限流）。不自动作废；是否采信由主会话定。", ""]

    # 7 迟到与重跑
    rows = []
    for _k, g in all_groups:
        for s in TRAIN_SEEDS:
            a = g["arms"][s]
            if a["ev"] is None:
                continue
            for sd, info in sorted(a["ev"]["late"].items()):
                it = a["eps"][sd]
                final = {"retry": "重跑后干净，采用重跑结果", "late": "没有干净的重跑结果（重跑仍迟到 / 中止 / 还没重跑），按没成功算"}.get(it["state"], it["state"])
                rows.append(f"| {g['names'][s]} | {sd} | {info.get('late_step_ids')} | {info.get('dispatch_lag_s_max')} | {final} |")
    lines += ["## 7. 含迟到步的集（基础设施失败）与重跑（run_r2.sh 条款 R6：重跑一次）", ""]
    lines += (["| 臂 | seed | 首次迟到步 | 首次 lag 最大 s | 最终 |", "|---|---|---|---|---|"] + rows) if rows else ["没有含迟到步的集。"]
    lines.append("")
    return "\n".join(lines), res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--note")
    ap.add_argument("--late-seeds", metavar="ARM_DIR")
    args = ap.parse_args(argv)
    if args.late_seeds:
        print(",".join(str(s) for s in late_seeds(args.late_seeds)))
        return 0
    if not args.out:
        ap.error("需要 OUT 目录")
    text, res = build(args.out, args.note)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
