#!/usr/bin/env python3
"""exp-action-latency-r2-20261111 汇总（任务卡 W1-0「动作派发时刻固定到仿真时间」第二轮），写 summary.md。随时可跑，缺的臂标「未完成」。

第二轮是看到第一轮失败之后设计的（第一轮 L=0.16 与 0.20 的标定门都没过，没有进入验收 4）；第一轮的预注册与判定原样保留（run.sh、summarize.py 不动）。
本脚本的口径与 run_r2.sh 文件头的预注册逐条对应；第一轮的辅助函数（逐集读取、时序统计、精确 McNemar）直接用 summarize.py 的。

运行：.venv-tools/bin/python summarize_r2.py [输出目录] [--note 说明] [--disp-clause cruise|steady]
      ... summarize_r2.py --gate A|B|C [--disp-clause ...] [输出目录]   只判一组标定：stdout 一行「状态 说明」，退出码见下
      ... summarize_r2.py --late-seeds 臂目录                          打印该臂 eval.json 里含迟到步的 seed（逗号分隔；没有则空行），run_r2.sh 重跑迟到集用
      状态与退出码：PASS 0；FAIL_LATE 1（未过，原因里含迟到步）；INCOMPLETE 2（数据不全）；FAIL_NOLATE 3（未过，原因里没有迟到步，只有位移 / 转角 / 生效时长）

臂目录（run_r2.sh 的命名）：
  cal-L0.20-k1、cal-L0.20-k3             A 组：L=0.20，默认 RTF，教师 K=1 与 K=3 各 10 个 train 分区 seed
  cal-L0.20rtf0.5-k3                     B 组：L=0.20，K=3 带 --rtf 0.5；K=1 复用 A 组的 cal-L0.20-k1
  cal-L0.24-k1、cal-L0.24-k3             C 组：L=0.24，默认 RTF
  acc-L<L*>-k<1|3>-s<0|1|2>              学生 calswa 三个训练种子 + 护栏 2，20 个冻结 seed；<臂>-retry1 = 含迟到步的集重跑一次（只重跑迟到的 seed）
  forest-L<L*>-<sparse|medium|dense>     森林三档，教师，每档 3 集（只报迟到步与延迟分布，不设门）
  <臂>.cg                                 臂开始 / 结束时的 cgroup 记录（epoch、usage_usec、nr_throttled、throttled_usec），算平均核数与限流增量

标定门（逐条，条款与 run_r2.sh 文件头一致）：
  1. 迟到步：K=1 与 K=3 都为 0（含每集第 0 步）。
  2. 逐步生效时长：K=3 对 K=1 的相对差（各集均值的集间中位）≤ 2%。
  3. 转角：K=3 对 K=1 的相对差（YAW 步 |dyaw_deg| 池化均值）≤ 2%。
  4. 稳态前飞位移（只比第 3–30 步）：K=3 对 K=1 的相对差 ≤ 2%；口径由 DISP_CLAUSE 定：
       steady = 本步与上一步都是 FORWARD_SLOW、本步无碰撞（第一轮的稳态口径，限第 3–30 步）；
       cruise = 本步及前 3 步都是 FORWARD_SLOW、本步无碰撞（连续前进第 4 步起；第一轮归因：稳态口径里 r=2 / r=3 的步位移大 0.02–0.07 m，
                K=3 的直线更长、这两类步占比更低，差的主要来自这个构成，不是步段或电量）。
  5. 不进门，只报告：全部步稳态前飞位移的相对差、另一种位移口径的相对差。
  迟到步数用首次记录，标定臂不重跑；生效时长 = oracle.action_end_ns − action_start_ns − actions.dispatch_lag_s。

验收 2：学生 calswa 种子 1 + 护栏 2、K=1、20 个冻结 seed、新协议，对 11-10 下午（17/20）与晚上对照（17/20）逐集配对；预测：逐集一致率不低于 70%。
         三个训练种子的 K=1 合起来（60 集）是新协议下的 K=1 参照。
验收 4：K=3 第 1 块通过率对 K=1 成功率（同权重同 seed 配对，精确 McNemar；两者差 = K=1 − K=3）。预注册（任务卡 W1-0 第 4 条原文）：
         新协议下差不超过 0.08 → 11-10 的下降（旧协议 39/60 对 50/60）来自第一步的时序；K=3 仍低 0.12 以上 → 来自后面几块的柱子进了画面；
         在 0.08 与 0.12 之间：两条线都没覆盖，如实记录，不归因。含迟到步的集重跑一次，以重跑结果为准（首次记录另列）。
"""
import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize as s1  # noqa: E402  第一轮汇总：逐集读取、时序统计、精确 McNemar；它也加载 11-10 长课程汇总（lc）

lc = s1.lc
D = s1.D
OUT = HERE                                   # 默认输出目录 = 本脚本所在目录；命令行第 1 个位置参数可覆盖
FROZEN_SEEDS = s1.FROZEN_SEEDS
SEEDS3 = s1.SEEDS3
CAL_EPISODES = 10
HOLD_S = 1.0
GATE_REL = 0.02
COMMON_LO, COMMON_HI = 3, 30                 # 位移只比这一步段
CRUISE_MIN_RUN = 4                           # 巡航 = 连续前进第 4 步起
DISP_CLAUSES = ("cruise", "steady")
DEFAULT_DISP_CLAUSE = "cruise"
AGREE_MIN = s1.AGREE_MIN
GAP_SMALL, GAP_LARGE = s1.GAP_SMALL, s1.GAP_LARGE
OLD_K1 = (50, 60)                            # 11-10 下午 guard2-calswa-s0..2，K=1 成功 50/60
OLD_K3_BLOCKS = (39, 27, 19)                 # 11-10 晚长课程第二轮，K=3 逐块通过 39/27/19（共 60 集）；下面用目录复算核对
OLD_EFF_S = 0.871
HALF_BATTERY_STEPS = 100                     # 任务卡：K=3 约 100 步到半电池
ELASTICITY = 2.38                            # 位移对生效时长的弹性：analysis-round1.md 第 3 节，第一轮教师 K=1 L=0.16 与 L=0.20 的稳态位移与逐步生效时长之比
RETRY_SUFFIX = "-retry1"
GROUPS = (
    {"id": "A", "label": "L=0.20，默认 RTF", "L": 0.20, "rtf": None, "k1": "cal-L0.20-k1", "k3": "cal-L0.20-k3"},
    {"id": "B", "label": "L=0.20，K=3 带 --rtf 0.5（K=1 复用 A 组）", "L": 0.20, "rtf": 0.5, "k1": "cal-L0.20-k1", "k3": "cal-L0.20rtf0.5-k3",
     "own": ("k3",)},   # K=1 是 A 组的臂，不算 B 组自己的
    {"id": "C", "label": "L=0.24，默认 RTF", "L": 0.24, "rtf": None, "k1": "cal-L0.24-k1", "k3": "cal-L0.24-k3"},
)
GROUP_BY_ID = {g["id"]: g for g in GROUPS}
mean, median, rel, pct, fnum = s1.mean, s1.median, s1.rel, s1.pct, s1.fnum
mcnemar_exact, wilson = s1.mcnemar_exact, s1.wilson


def arm(name):
    return os.path.join(OUT, name)


def read_config(out=None):
    """$OUT/config.env（run_r2.sh 起跑时写，KEY=VALUE 每行一条）→ dict；没有文件返回 {}。"""
    path = os.path.join(out or OUT, "config.env")
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    cfg[k.strip()] = v.strip()
    return cfg


def disp_clause_of(explicit=None):
    """位移口径：命令行 > config.env > 缺省。"""
    v = explicit or read_config().get("DISP_CLAUSE") or DEFAULT_DISP_CLAUSE
    if v not in DISP_CLAUSES:
        raise ValueError(f"DISP_CLAUSE 必须是 {DISP_CLAUSES}，得到 {v!r}")
    return v


# ---------------------------------------------------------------- 逐步样本与位移口径
def fwd_samples(actions):
    """[(步号, 上一步动作, displacement_m, 连续前进位置 r)]：本步 FORWARD、无碰撞、有位移的样本；第 0 步没有上一步，不作样本但计入连续前进的计数。
    r = 本步是连续 FORWARD 的第几个（含本步）；r≥2 的上一步必是前进。与 analyze_round1.fwd_samples 同口径（测试里交叉核对）。"""
    out, run = [], 0
    for i, a in enumerate(actions):
        run = run + 1 if a.get("action_executed") == 1 else 0
        if i == 0:
            continue
        if a.get("action_executed") == 1 and not a.get("collision") and a.get("displacement_m") is not None:
            out.append((i, actions[i - 1].get("action_executed"), a["displacement_m"], run))
    return out


def disp_stats(edir):
    """一个臂（目录）的前飞位移四个口径：{名字: (池化均值, 样本数)}。
    steady_common = 上一步前进、步 3–30；cruise_common = 连续前进第 4 步起、步 3–30；steady_all / cruise_all = 同口径不限步段（不进门，只报告）。"""
    pools = {"steady_common": [], "cruise_common": [], "steady_all": [], "cruise_all": []}
    for _seed, actions, _oracle, _man in s1.episode_records(edir):
        for i, _p, d, r in fwd_samples(actions):
            if r >= 2:
                pools["steady_all"].append(d)
                if COMMON_LO <= i <= COMMON_HI:
                    pools["steady_common"].append(d)
            if r >= CRUISE_MIN_RUN:
                pools["cruise_all"].append(d)
                if COMMON_LO <= i <= COMMON_HI:
                    pools["cruise_common"].append(d)
    return {k: (mean(v), len(v)) for k, v in pools.items()}


def disp_key(clause):
    return "cruise_common" if clause == "cruise" else "steady_common"


# ---------------------------------------------------------------- cgroup 记录与起跑条件
def read_cg(name):
    """<臂>.cg：两行「start|end epoch usage_usec nr_throttled throttled_usec」→ {avg_cores, throttled, throttled_s, wall_s}；缺 end（臂还在跑）或没有文件返回 None。"""
    path = arm(name) + ".cg"
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5 and parts[0] in ("start", "end"):
                rows[parts[0]] = [float(x) for x in parts[1:]]
    if "start" not in rows or "end" not in rows:
        return None
    s, e = rows["start"], rows["end"]
    wall = e[0] - s[0]
    return {"wall_s": wall, "avg_cores": (e[1] - s[1]) / 1e6 / wall if wall > 0 else float("nan"),
            "throttled": int(e[2] - s[2]), "throttled_s": (e[3] - s[3]) / 1e6}


def read_start(out=None):
    """$OUT/start.txt（run_r2.sh 写的起跑条件记录，key=value 每行一条）→ dict。"""
    path = os.path.join(out or OUT, "start.txt")
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, _, v = line.strip().partition("=")
                    cfg[k] = v
    return cfg


# ---------------------------------------------------------------- 逐臂读数
def arm_counts(edir, requested):
    """一个臂的集数：{requested, done, failed, late（含迟到步的 seed 列表）, other_failed, complete, state} 或 None（没有 eval.json）。"""
    e, state, complete = s1.eval_state(edir, requested)
    if e is None:
        return {"requested": requested, "done": 0, "failed": 0, "late": [], "other_failed": 0, "complete": False, "state": state}
    late = sorted(int(x["seed"]) for x in e.get("late_dispatch_episodes") or [])
    failed = e.get("failed_count") or 0
    return {"requested": requested, "done": e.get("episodes_done") or 0, "failed": failed, "late": late,
            "other_failed": failed - len(late), "complete": complete, "state": state}


def late_seeds(edir):
    """eval.json 里含迟到步的 seed（升序）；没有文件返回 []。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return []
    return sorted(int(x["seed"]) for x in s1.read_json(path).get("late_dispatch_episodes") or [])


def merged_rows(edir, requested):
    """学生臂含重跑的逐集结果。返回 None（没有 eval.json）或 {seed: {...}}：
      row     用的那条记录（lc.load_arm 的行：success、pass、sim_s、steps、z_tail_min…）
      first   首次记录；first_late 首次是否含迟到步（基础设施失败）
      final   "first" | "retry" | "none"：最终采用哪次；含迟到步且重跑也没拿到干净结果 = "none"（按没成功算）
      retry   None | "ok" | "late" | "failed" | "missing"：首次迟到的集，重跑的结果（干净 / 仍迟到 / 重跑中止但不是迟到 / 没重跑过或目录里没有这个 seed）
    这个函数不看 requested 是否跑满，完整性由 arm_counts 报。"""
    first_arm = lc.load_arm(edir)
    if first_arm is None:
        return None
    late = set(late_seeds(edir))
    retry_dir = edir + RETRY_SUFFIX
    retry_arm = lc.load_arm(retry_dir) if os.path.exists(os.path.join(retry_dir, "eval.json")) else None
    retry_late = set(late_seeds(retry_dir))
    out = {}
    for seed, row in first_arm["eps"].items():
        item = {"first": row, "first_late": seed in late, "retry": None, "final": "first", "row": row}
        if seed in late:
            if retry_arm is None or seed not in retry_arm["eps"]:
                item["retry"], item["final"] = "missing", "none"
            elif seed in retry_late:
                item["retry"], item["final"] = "late", "none"
            elif retry_arm["eps"][seed].get("aborted"):
                item["retry"], item["final"] = "failed", "none"
            else:
                item["retry"], item["final"], item["row"] = "ok", "retry", retry_arm["eps"][seed]
        out[seed] = item
    return out


def success_map(merged, k):
    """{seed: 是否成功}：K=1 看 success，K=3 看第 1 块是否通过；迟到（最终没拿到干净结果）按没成功算。"""
    out = {}
    for seed, it in merged.items():
        clean = it["final"] in ("first", "retry") and not (it["final"] == "first" and it["first_late"])
        row = it["row"]
        ok = bool(row.get("success")) if k == 1 else lc.passed_k(row, 0)
        out[seed] = bool(ok and clean)
    return out


def first_run_map(merged, k):
    """只看首次记录（不用重跑）：迟到集按没成功算。"""
    out = {}
    for seed, it in merged.items():
        row = it["first"]
        ok = bool(row.get("success")) if k == 1 else lc.passed_k(row, 0)
        out[seed] = bool(ok and not it["first_late"])
    return out


# ---------------------------------------------------------------- 标定门
def gate_group(g, clause):
    """一组标定的门。返回 dict：status（PASS|FAIL_LATE|FAIL_NOLATE|INCOMPLETE）、text（一行）、reasons（代码列表 late|eff|yaw|disp）、
    rows（逐条款：name, k1, k3, value, limit, ok）、t1 / t3（时序统计）、d1 / d3（位移口径）。"""
    st, dd, notes, ok_data = {}, {}, [], True
    for k, name in ((1, g["k1"]), (3, g["k3"])):
        _e, text, complete = s1.eval_state(arm(name), CAL_EPISODES)
        st[k] = s1.timing_stats([arm(name)])
        dd[k] = disp_stats(arm(name)) if st[k] else None
        if not complete or st[k] is None:
            ok_data = False
            notes.append(f"K={k} 标定臂 {name}：{text}")
    if not ok_data:
        return {"status": "INCOMPLETE", "text": f"{g['id']} 组（{g['label']}）：" + "；".join(notes), "reasons": [], "rows": [],
                "t1": st[1], "t3": st[3], "d1": dd[1], "d3": dd[3]}
    t1, t3, d1, d3 = st[1], st[3], dd[1], dd[3]
    rows, reasons = [], []
    no_flag = t1["late_steps"] is None or t3["late_steps"] is None   # 行里没有 late_dispatch：没带 --action-latency，是数据错误
    late_ok = (not no_flag) and t1["late_steps"] == 0 and t3["late_steps"] == 0
    if no_flag:
        reasons.append("nolate_flag")
    elif not late_ok:
        reasons.append("late")
    rows.append({"name": "迟到步（含第 0 步）", "k1": f"{t1['late_steps']}/{t1['steps']}", "k3": f"{t3['late_steps']}/{t3['steps']}",
                 "value": None, "limit": "两个臂都为 0", "ok": late_ok})
    key = disp_key(clause)
    clauses = [("逐步生效时长（各集均值的集间中位 s）", t1["eff_med"], t3["eff_med"], 4, "eff", True),
               ("转角（YAW 步 |dyaw_deg| 池化均值 °）", t1["yaw"], t3["yaw"], 2, "yaw", True),
               (f"前飞位移（{'连续前进第 4 步起' if clause == 'cruise' else '上一步也是前进'}，步 {COMMON_LO}–{COMMON_HI}；m/步）",
                d1[key][0], d3[key][0], 4, "disp", True)]
    for name, a, b, nd, code, gated in clauses:
        d = rel(b, a)
        ok = d == d and abs(d) <= GATE_REL
        if not ok:
            reasons.append(code)
        rows.append({"name": name, "k1": fnum(a, nd), "k3": fnum(b, nd), "value": d, "limit": f"|相对差| ≤ {GATE_REL:.0%}", "ok": ok})
    other = "steady_common" if clause == "cruise" else "cruise_common"
    other_name = {"steady_common": f"（不进门）上一步也是前进，步 {COMMON_LO}–{COMMON_HI}", "cruise_common": f"（不进门）连续前进第 4 步起，步 {COMMON_LO}–{COMMON_HI}"}[other]
    for name, kk in ((other_name, other), ("（不进门）上一步也是前进，全部步（第一轮的门口径）", "steady_all"), ("（不进门）连续前进第 4 步起，全部步", "cruise_all")):
        if kk == key:
            continue
        d = rel(d3[kk][0], d1[kk][0])
        rows.append({"name": name, "k1": f"{fnum(d1[kk][0])}（n={d1[kk][1]}）", "k3": f"{fnum(d3[kk][0])}（n={d3[kk][1]}）", "value": d, "limit": "只报告", "ok": None})
    gated_codes = [c for c in reasons if c in ("late", "eff", "yaw", "disp", "nolate_flag")]
    if not gated_codes:
        status = "PASS"
    elif "late" in gated_codes and "nolate_flag" not in gated_codes:
        status = "FAIL_LATE"
    else:
        status = "FAIL_NOLATE"   # 含「行里没有 late_dispatch」：数据错误，阶梯不往下试
    bad = [r["name"] for r in rows if r["ok"] is False]
    head = (f"{g['id']} 组（{g['label']}）：K=3 对 K=1 生效时长 {pct(rows[1]['value'])}，转角 {pct(rows[2]['value'])}，前飞位移（{clause}）{pct(rows[3]['value'])}；"
            f"迟到步 K=1 {t1['late_steps']}/{t1['steps']}、K=3 {t3['late_steps']}/{t3['steps']}")
    return {"status": status, "text": head + ("；未过：" + "；".join(bad) if bad else ""), "reasons": gated_codes, "rows": rows,
            "t1": t1, "t3": t3, "d1": d1, "d3": d3}


def ladder_state(clause):
    """按阶梯规则逐组判：返回 (每组结果 dict {id: gate}, 选定的组 id 或 None, 阶梯状态文字)。
    规则（预注册）：A 过 → 选 A；A 未过且原因含迟到步 → 看 B；B 未过且原因含迟到步 → 看 C；C 过 → 选 C。
    某组未过且原因里没有迟到步（只有位移 / 转角 / 生效时长）→ 停；三组都没过 → 停；数据不全 → 进行中。"""
    results, selected, state = {}, None, "未开始"
    for g in GROUPS:
        res = gate_group(g, clause)
        results[g["id"]] = res
        if res["status"] == "INCOMPLETE":
            started = any(os.path.isdir(arm(g[k])) for k in g.get("own", ("k1", "k3")))
            state = f"{g['id']} 组{'进行中' if started else '还没跑'}"
            break
        if res["status"] == "PASS":
            selected, state = g["id"], f"{g['id']} 组通过"
            break
        if res["status"] == "FAIL_NOLATE":
            state = f"{g['id']} 组未过，原因里没有迟到步（{'、'.join(res['reasons'])}），按规则停"
            break
        state = f"{g['id']} 组未过（含迟到步）" + ("，往下试" if g["id"] != GROUPS[-1]["id"] else "，三组都没过，停")
    return results, selected, state


# ---------------------------------------------------------------- 各节
def timing_row(label, edir, requested=None):
    t = s1.timing_stats([edir])
    cg = read_cg(os.path.basename(edir))
    cores = "-" if cg is None else f"{cg['avg_cores']:.1f}"
    thr = "-" if cg is None else (f"{cg['throttled']}" + (" ⚠" if cg["throttled"] > 0 else ""))
    cnt = arm_counts(edir, requested) if requested else None
    infra = "-" if cnt is None else f"{cnt['failed']}（迟到 {len(cnt['late'])}，其它 {cnt['other_failed']}）"
    if t is None:
        return f"| {label} | 未完成 | | | | | | | | | | {infra} | {cores} | {thr} |"
    d = disp_stats(edir)
    late = "-" if t["late_steps"] is None else f"{t['late_steps']}/{t['steps']}"
    s0 = [x for x in t["step0"] if x[2] is not None]
    s0_late = "-" if not t["flagged"] else f"{sum(1 for x in s0 if x[3])}/{len(s0)}"
    return (f"| {label} | {t['n_eps']} | {t['steps']} | {fnum(t['eff_med'])} | {fnum(d['steady_common'][0])} | {fnum(d['cruise_common'][0])} | "
            f"{fnum(d['steady_all'][0])} | {fnum(t['yaw'], 2)} | {fnum(t['lag_med'], 3)} / {fnum(t['lag_max'], 3)} | {late} | {s0_late} | {infra} | {cores} | {thr} |")


TIMING_HEAD = ["| 臂 | 集数 | 步数 | 逐步生效时长 集间中位 s | 稳态位移 步 3–30（上一步前进）m | 巡航位移 步 3–30（r≥4）m | 稳态位移 全部步 m | 转角 °/步 | "
               "dispatch_lag_s 中位 / 最大 s | 迟到步 | 第 0 步迟到的集 | 基础设施失败集数（请求内） | 平均核数 | 限流增量 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]


def start_section():
    st = read_start()
    cfg = read_config()
    lines = ["## 1. 起跑条件与资源", ""]
    if not st:
        return lines + ["（没有 start.txt：守护进程还没起，或这个目录不是 run_r2.sh 的输出）", ""]
    lines += [f"- 起跑条件：配置的 GPU 在 `nvidia-smi -L` 里 = {st.get('gpu_ok', '-')}；cgroup 配额 {st.get('cpu_quota_cores', '-')} 核（要求 ≥ {st.get('min_cores', '12')}）；"
              f"等机器锁 {st.get('lock_wait_s', '-')} s；拿到锁后 gz sim 进程数 {st.get('gz_count', '-')}；{st.get('idle_window_s', '30')} s 窗口内容器平均 {st.get('idle_cores', '-')} 核"
              f"（要求 < {st.get('idle_max_cores', '1.5')}；等了 {st.get('idle_wait_s', '-')} s）。",
              f"- 起跑时刻 {st.get('started_at', '-')}；GPU {st.get('gpu', '-')}；LANES = {st.get('lanes', '-')}；位移口径 DISP_CLAUSE = {cfg.get('DISP_CLAUSE', DEFAULT_DISP_CLAUSE)}。", ""]
    return lines


def ladder_section(clause):
    results, selected, state = ladder_state(clause)   # 返回 (行, 每组结果, 选定组 id, 阶梯状态文字)
    lines = ["## 2. 标定阶梯（验收 1）：教师 K=1 与 K=3 各 10 个 train 分区 seed", "",
             f"阶梯：A（L=0.20，默认 RTF）→ 原因含迟到步则 B（L=0.20，K=3 带 `--rtf 0.5`，K=1 复用 A）→ 仍含迟到步则 C（L=0.24，默认 RTF）；第一个过门的组定下新协议参数，后面的不跑。"
             f"某组未过但原因里没有迟到步（只有位移 / 转角 / 生效时长）→ 停；三组都没过 → 停。位移条款口径 = **{clause}**（"
             + ("连续前进第 4 步起、步 3–30" if clause == "cruise" else "上一步也是前进、步 3–30") + "），全部条款见文件头。", "",
             f"阶梯状态：**{state}**" + (f"；选定 {selected} 组（L={GROUP_BY_ID[selected]['L']:.2f}" + (f"，教师 K=3 带 --rtf {GROUP_BY_ID[selected]['rtf']}" if GROUP_BY_ID[selected]["rtf"] else "") + "）" if selected else ""), ""]
    for g in GROUPS:
        res = results.get(g["id"])
        if res is None:
            lines += [f"### {g['id']} 组：{g['label']}：未判（阶梯在前面的组停了）", ""]
            continue
        lines += [f"### {g['id']} 组：{g['label']}：{res['status']}", "", res["text"], ""]
        if res["rows"]:
            lines += ["| 条款 | K=1 | K=3 | K=3 对 K=1 | 门限 | 判定 |", "|---|---|---|---|---|---|"]
            for r in res["rows"]:
                v = "" if r["value"] is None else pct(r["value"])
                verdict = "-" if r["ok"] is None else ("过" if r["ok"] else "**未过**")
                lines.append(f"| {r['name']} | {r['k1']} | {r['k3']} | {v} | {r['limit']} | {verdict} |")
            lines.append("")
        t1, t3 = res["t1"], res["t3"]
        for k, t in ((1, t1), (3, t3)):
            if t:
                s0 = "；".join(f"seed {s} {fnum(lag, 3)} s{'（迟到）' if late else ''}" for _, s, lag, late in t["step0"])
                lines.append(f"- K={k} 每集第 0 步 dispatch_lag_s：{s0}")
                if t["late_eps"]:
                    lines.append(f"- K={k} 迟到步明细：" + "；".join(f"seed {s} 步 {i}" for _, s, i in t["late_eps"]))
        lines.append("")
    return lines, results, selected, state


def pairs_line(label, a, b):
    """a、b = {seed: bool}：返回表格行（逐集一致、b、c、p）与一致率。"""
    same, n, bb, cc = s1.agreement(a, b)
    if not n:
        return f"| {label} | 没有共同的冻结 seed | | | | |", None
    return f"| {label} | {same}/{n} | {same / n:.2f} | {bb} | {cc} | {mcnemar_exact(bb, cc):.3f} |", same / n


def acceptance2_section(L):
    lines = ["## 3. 验收 2：学生 calswa 种子 1 + 护栏 2、K=1、20 个冻结 seed，新协议对 11-10 两次同配置评测", ""]
    if L is None:
        return lines + ["没有选定的 L（标定阶梯没有通过或还没走完），本节没有数据。", ""], None
    edir = arm(f"acc-L{L:.2f}-k1-s1")
    merged = merged_rows(edir, 20)
    if merged is None:
        return lines + [f"{os.path.basename(edir)}：{arm_counts(edir, 20)['state']}。", ""], None
    cnt = arm_counts(edir, 20)
    new = success_map(merged, 1)
    first = first_run_map(merged, 1)
    aft, eve = s1.success_by_seed(s1.REF_K1_AFTERNOON_S1), s1.success_by_seed(s1.REF_K1_EVENING_S1)
    lines += [f"- 新协议（L={L:.2f}，含迟到集重跑后）走廊内成功 {lc.fmt_frac(sum(new.values()), len(new))}；下午 "
              f"{'-' if aft is None else lc.fmt_frac(sum(aft.values()), len(aft))}；晚上对照 {'-' if eve is None else lc.fmt_frac(sum(eve.values()), len(eve))}。"
              f"（首次记录 {sum(first.values())}/{len(first)}；含迟到步的集 {cnt['late'] or '无'}；臂状态：{cnt['state']}）", ""]
    lines += ["| 配对（20 对） | 逐集一致 | 一致率 | 只有前者成功 b | 只有后者成功 c | 精确 McNemar p（只作参考） |", "|---|---|---|---|---|---|"]
    rates = {}
    variants = [("新协议 对 下午", new, aft), ("新协议 对 晚上对照", new, eve)]
    if first != new:
        variants += [("新协议（只看首次记录）对 下午", first, aft), ("新协议（只看首次记录）对 晚上对照", first, eve)]
    variants += [("下午 对 晚上对照（已知 16/20）", aft, eve)]
    for label, a, b in variants:
        if a is None or b is None:
            lines.append(f"| {label} | 参照缺失 | | | | |")
            continue
        row, rate = pairs_line(label, a, b)
        lines.append(row)
        rates[label] = rate
    ok = [rates.get("新协议 对 下午"), rates.get("新协议 对 晚上对照")]
    verdict = None
    if all(r is not None for r in ok):
        holds = all(r >= AGREE_MIN for r in ok)
        verdict = (holds, ok)
        lines += ["", f"预测（差别在同权重同 seed 重评的波动之内，逐集一致率 ≥ {AGREE_MIN:.0%}）：**{'成立' if holds else '未成立'}**（新对下午 {ok[0]:.2f}，新对晚上 {ok[1]:.2f}）。"
                  + ("" if cnt["complete"] else "（这一组还没评完，只作进度）")]
    lines.append("")
    # 三个训练种子合起来 = 新协议下的 K=1 参照
    ks = {s: merged_rows(arm(f"acc-L{L:.2f}-k1-s{s}"), 20) for s in SEEDS3}
    if all(v is not None for v in ks.values()):
        n_ok = sum(sum(success_map(v, 1).values()) for v in ks.values())
        n_all = sum(len(v) for v in ks.values())
        lines += [f"三个训练种子合起来是新协议下的 K=1 参照：{lc.fmt_frac(n_ok, n_all)}（旧协议 11-10 下午 {OLD_K1[0]}/{OLD_K1[1]}）。逐种子："
                  + "；".join(f"s{s} {sum(success_map(v, 1).values())}/{len(v)}" for s, v in ks.items()), ""]
    return lines, verdict


def k3_block_stats(merged):
    """K=3 一个臂（merged_rows 的结果）的逐块通过数、掉高集数、sim_s 与步数。S_k = 通过前 k 块的集数；迟到（最终没拿到干净结果）按没通过。"""
    n = len(merged)
    S = [0, 0, 0]
    drops, sim_s, steps = 0, [], []
    for it in merged.values():
        row = it["row"]
        clean = it["final"] in ("first", "retry") and not (it["final"] == "first" and it["first_late"])
        for k in range(3):
            if clean and lc.passed_k(row, k):
                S[k] += 1
        if row.get("z_tail_min") is not None and row["z_tail_min"] < lc.TAIL_LOW_Z_M:
            drops += 1
        if row.get("sim_s") is not None:
            sim_s.append(row["sim_s"])
        if row.get("steps") is not None:
            steps.append(row["steps"])
    return {"n": n, "S": S, "drops": drops, "sim_s": sim_s, "steps": steps}


def acceptance4_section(L):
    lines = ["## 4. 验收 4：K=3 第 1 块通过率对 K=1 成功率（学生 calswa 三个训练种子 + 护栏 2，同权重同 seed 配对）", ""]
    if L is None:
        return lines + ["没有选定的 L（标定阶梯没有通过或还没走完），本节没有数据。", ""], None, {}
    k1 = {s: merged_rows(arm(f"acc-L{L:.2f}-k1-s{s}"), 20) for s in SEEDS3}
    k3 = {s: merged_rows(arm(f"acc-L{L:.2f}-k3-s{s}"), 20) for s in SEEDS3}
    miss = [f"K=1 s{s}" for s in SEEDS3 if k1[s] is None] + [f"K=3 s{s}" for s in SEEDS3 if k3[s] is None]
    if miss:
        lines += ["没有 eval.json 的臂：" + "、".join(miss) + "。已有的配对如下。", ""]
    n = n1 = n3 = b = c = 0
    per = []
    for variant, mapper in (("final", success_map), ("first", first_run_map)):
        nn = nn1 = nn3 = bb = cc = 0
        pp = []
        for s in SEEDS3:
            if k1[s] is None or k3[s] is None:
                continue
            m1, m3 = mapper(k1[s], 1), mapper(k3[s], 3)
            seeds = [x for x in FROZEN_SEEDS if x in m1 and x in m3]
            a1, a3 = sum(m1[x] for x in seeds), sum(m3[x] for x in seeds)
            bs = sum(1 for x in seeds if m1[x] and not m3[x])
            cs = sum(1 for x in seeds if (not m1[x]) and m3[x])
            nn, nn1, nn3, bb, cc = nn + len(seeds), nn1 + a1, nn3 + a3, bb + bs, cc + cs
            pp.append(f"s{s}：K=1 {a1}/{len(seeds)}，K=3 第 1 块 {a3}/{len(seeds)}，只 K=1 过 {bs}、只 K=3 过 {cs}")
        if variant == "final":
            n, n1, n3, b, c, per = nn, nn1, nn3, bb, cc, pp
            first_res = None
        else:
            first_res = (nn, nn1, nn3, bb, cc)
    verdict = None
    if n:
        gap = (n1 - n3) / n
        lines += [f"- K=1 成功 {lc.fmt_frac(n1, n)}；K=3 第 1 块通过 {lc.fmt_frac(n3, n)}；差（K=1 − K=3）= **{gap:+.3f}**；"
                  f"配对 b（只 K=1 过）={b}，c（只 K=3 过）={c}，精确 McNemar p = {mcnemar_exact(b, c):.3f}（{n} 对不独立，p 只作参考）", ""]
        lines += [f"  - {p}" for p in per] + [""]
        counts = (n1, n3, n)
        if abs(gap) <= GAP_SMALL:
            verdict = ("small", gap, f"两者相差 |{gap:.3f}| ≤ {GAP_SMALL}：11-10 那次下降（旧协议 39/60 对 50/60）**来自第一步的时序**。", counts)
        elif gap >= GAP_LARGE:
            verdict = ("large", gap, f"K=3 第 1 块比 K=1 低 {gap:.3f} ≥ {GAP_LARGE}：**来自后面几块的柱子进了画面**（学生没见过远处的柱子）。", counts)
        else:
            verdict = ("between", gap, f"差（K=1 − K=3）= {gap:+.3f}，不在「相差不超过 {GAP_SMALL}」也不在「K=3 低 {GAP_LARGE} 以上」：预注册的两条线都没覆盖，如实记录，不归因。", counts)
        text = verdict[2] + ("" if len(per) == len(SEEDS3) else "（三个训练种子没有全部完成，只作进度）")
        lines += ["预注册判定：" + text, ""]
        if first_res and first_res[0]:
            fn, f1, f3, fb, fc = first_res
            lines += [f"只看首次记录（迟到集按没成功）：K=1 {f1}/{fn}，K=3 第 1 块 {f3}/{fn}，差 {(f1 - f3) / fn:+.3f}，b={fb}，c={fc}。", ""]
    else:
        lines += ["还没有可配对的完整臂。", ""]
    return lines, verdict, {"k1": k1, "k3": k3}


def k3_section(L, merged_k3):
    """K=3 逐块存活 S_1/S_2/S_3、掉高集数、sim_s 与步数分布；并排旧协议参照。"""
    lines = ["## 5. K=3 逐块存活与旧协议并排", ""]
    if L is None or not merged_k3:
        return lines + ["没有 K=3 学生臂的数据。", ""]
    rows, tot = [], {"n": 0, "S": [0, 0, 0], "drops": 0, "sim_s": [], "steps": []}
    for s in SEEDS3:
        m = merged_k3.get(s)
        if m is None:
            continue
        st = k3_block_stats(m)
        rows.append((f"新协议 K=3 s{s}", st))
        tot["n"] += st["n"]
        tot["drops"] += st["drops"]
        tot["sim_s"] += st["sim_s"]
        tot["steps"] += st["steps"]
        tot["S"] = [a + b for a, b in zip(tot["S"], st["S"])]
    ref_arms = [lc.load_arm(p) for p in s1.REF_K3]
    ref = None
    if all(a is not None for a in ref_arms):
        ref = {"n": sum(len(a["eps"]) for a in ref_arms), "S": [sum(1 for a in ref_arms for r in a["eps"].values() if lc.passed_k(r, k)) for k in range(3)],
               "drops": sum(1 for a in ref_arms for r in a["eps"].values() if r.get("z_tail_min") is not None and r["z_tail_min"] < lc.TAIL_LOW_Z_M),
               "sim_s": [r["sim_s"] for a in ref_arms for r in a["eps"].values() if r.get("sim_s") is not None],
               "steps": [r["steps"] for a in ref_arms for r in a["eps"].values() if r.get("steps") is not None]}
    lines += ["S_k = 通过前 k 块的集数 / 请求集数（累计）。掉高 = 末 10 步最低高度 < 1.5 m（电池掉压的症状，只作诊断）；sim_s = 末个动作终点的仿真秒（含起飞前约 13.5 s，电池按它消耗）。"
              f"新协议每步生效 hold − L（L={L:.2f} → {HOLD_S - L:.2f} s，比旧协议的 {OLD_EFF_S} s 短 {100 * (1 - (HOLD_S - L) / OLD_EFF_S):.0f}%）；每步前飞位移的缩幅比生效时长的缩幅大"
              f"（第一轮教师 K=1 在 L=0.16 与 0.20 之间量到的弹性约 {ELASTICITY}，来源 analysis-round1.md 第 3 节，只是估计），同一段航程要多飞多少步看下表的步数列；"
              f"多飞的步数让 K=3 离半电池的约 {HALF_BATTERY_STEPS} 步近一些；验收 4 只看第 1 块。", "",
              "| 臂 | 集数 | S_1 | S_2 | S_3 | 掉高集数 | 步数 均值 / 最大 | ≥ 100 步的集 | sim_s 中位 / p90 / 最大 |", "|---|---|---|---|---|---|---|---|---|"]

    def row(label, st):
        n = st["n"]
        sim = sorted(st["sim_s"])
        p90 = sim[min(len(sim) - 1, int(0.9 * len(sim)))] if sim else float("nan")
        long_n = sum(1 for x in st["steps"] if x >= HALF_BATTERY_STEPS)
        return (f"| {label} | {n} | " + " | ".join(f"{x}/{n}" for x in st["S"]) + f" | {st['drops']} | {fnum(mean(st['steps']), 1)} / {max(st['steps']) if st['steps'] else '-'} | {long_n} | "
                f"{fnum(median(sim), 1)} / {fnum(p90, 1)} / {fnum(max(sim) if sim else float('nan'), 1)} |")
    for label, st in rows:
        lines.append(row(label, st))
    if rows:
        lines.append(row("**新协议 K=3 合计**", tot))
    if ref:
        lines.append(row("旧协议 K=3（11-10 晚长课程第二轮，s0..2）", ref))
        lines += ["", f"旧协议参照复算：K=3 逐块 {ref['S'][0]}/{ref['n']}、{ref['S'][1]}/{ref['n']}、{ref['S'][2]}/{ref['n']}"
                  f"（记录值 {OLD_K3_BLOCKS[0]}/{OLD_K3_BLOCKS[1]}/{OLD_K3_BLOCKS[2]} 的 60 集）；K=1 11-10 下午 {OLD_K1[0]}/{OLD_K1[1]}。"]
    lines.append("")
    return lines


def retry_section(L):
    lines = ["## 6. 含迟到步的集的重跑记录（学生臂；标定臂不重跑，迟到步就是测量对象）", ""]
    if L is None:
        return lines + ["没有选定的 L。", ""]
    rows = []
    for k in (1, 3):
        for s in SEEDS3:
            name = f"acc-L{L:.2f}-k{k}-s{s}"
            merged = merged_rows(arm(name), 20)
            if merged is None:
                continue
            for seed, it in sorted(merged.items()):
                if it["first_late"]:
                    e = s1.read_json(os.path.join(arm(name), "eval.json"))
                    info = next((x for x in e.get("late_dispatch_episodes") or [] if int(x["seed"]) == seed), {})
                    res = {"ok": "重跑后干净，采用重跑结果", "late": "重跑后仍含迟到步，按没成功算", "failed": "重跑中止（不是迟到），按没成功算", "missing": "没有重跑记录（还没轮到，或没跑），暂按没成功算"}[it["retry"]]
                    rows.append(f"| {name} | {seed} | {info.get('late_step_ids')} | {info.get('dispatch_lag_s_max')} | {res} |")
    if not rows:
        return lines + ["学生臂没有含迟到步的集（或学生臂还没出结果）。", ""]
    return lines + ["| 臂 | seed | 首次迟到步 | 首次 lag 最大 s | 重跑结果 |", "|---|---|---|---|---|"] + rows + [""]


def forest_section(L):
    lines = ["## 7. 森林三档（教师，带 --action-latency，每档 3 集；只报迟到步与延迟分布，不设门）", ""]
    if L is None:
        return lines + ["没有选定的 L。", ""]
    lines += ["| 密度 | 集数 | 步数 | 柱子数 | dispatch_lag_s 中位 / p90 / 最大 s | 迟到步 | 第 0 步迟到的集 | 逐步生效时长均值 s（最小–最大） | 请求内基础设施失败集数 |", "|---|---|---|---|---|---|---|---|---|"]
    for dens in ("sparse", "medium", "dense"):
        edir = arm(f"forest-L{L:.2f}-{dens}")
        t = s1.timing_stats([edir])
        if t is None:
            lines.append(f"| {dens} | 未完成 | | | | | | | |")
            continue
        e = s1.read_json(os.path.join(edir, "eval.json")) if os.path.exists(os.path.join(edir, "eval.json")) else {}
        pillars = [s1.pillar_count(x.get("pillars")) for x in e.get("per_episode", [])]
        s0 = [x for x in t["step0"] if x[2] is not None]
        late = "-" if t["late_steps"] is None else f"{t['late_steps']}/{t['steps']}"
        lines.append(f"| {dens} | {t['n_eps']} | {t['steps']} | {fnum(mean(pillars), 0)} | {fnum(t['lag_med'], 3)} / {fnum(s1.percentile(t['lags'], 90), 3)} / {fnum(t['lag_max'], 3)} | "
                     f"{late} | {sum(1 for x in s0 if x[3])}/{len(s0)} | {fnum(t['eff_mean'])}（{fnum(t['eff_min'], 3)}–{fnum(t['eff_max'], 3)}） | {e.get('failed_count', '-')} |")
    lines.append("")
    return lines


def overview_section(L):
    lines = ["## 8. 全部臂的时序健康（每步生效时长、位移、转角、延迟、迟到步、基础设施失败、平均核数、限流）", "",
             "平均核数 = 臂开始到结束之间容器 cgroup 的 CPU 使用（usage_usec 增量 ÷ 墙钟）；限流增量 = nr_throttled 增量（⚠ = 大于 0）。两条车道并行时两个臂的区间重叠，数是容器整体，不是单臂独占。"
             "有 ⚠ 的臂不自动作废，是否采信由主会话定。", "", *TIMING_HEAD]
    names = []
    for g in GROUPS:
        for k in ("k1", "k3"):
            if g[k] not in [n for n, _r in names]:
                names.append((g[k], CAL_EPISODES))
    if L is not None:
        for k in (1, 3):
            for s in SEEDS3:
                names += [(f"acc-L{L:.2f}-k{k}-s{s}", 20), (f"acc-L{L:.2f}-k{k}-s{s}{RETRY_SUFFIX}", None)]
        names += [(f"forest-L{L:.2f}-{d}", 3) for d in ("sparse", "medium", "dense")]
    for name, req in names:
        if os.path.isdir(os.path.join(arm(name), "episodes")):
            lines.append(timing_row(name, arm(name), req))
    lines.append("")
    return lines


def conclusion_section(clause, results, selected, state, acc2, acc4):
    lines = ["## 0. 结论一览（自动生成，数字都来自下面各节）", ""]
    done = os.path.join(OUT, "DONE")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as f:
            lines += [f"- DONE：`{f.read().strip()}`"]
    lines += [f"- 标定阶梯：{state}。" + "".join(f"{g['id']} 组 {results[g['id']]['status']}；" for g in GROUPS if g["id"] in results)]
    if selected:
        g = GROUP_BY_ID[selected]
        lines += [f"- 新协议参数：L = {g['L']:.2f} s（每步生效 {HOLD_S - g['L']:.2f} s）" + (f"；教师 K=3 要带 `--rtf {g['rtf']}` 才过门，学生臂不带 `--rtf`" if g["rtf"] else "；教师不需要 `--rtf`") + "。"]
    else:
        lines += ["- 新协议参数：没有选定（标定阶梯没有通过，或还没走完）。"]
    if acc2:
        holds, ok = acc2
        lines += [f"- 验收 2：逐集一致率 新对下午 {ok[0]:.2f}、新对晚上 {ok[1]:.2f}，预测（≥ {AGREE_MIN:.0%}）{'成立' if holds else '未成立'}。"]
    else:
        lines += ["- 验收 2：没有结果。"]
    if acc4:
        n1, n3, n = acc4[3]
        lines += [f"- 验收 4：K=1 成功 {n1}/{n}，K=3 第 1 块通过 {n3}/{n}，差（K=1 − K=3）{acc4[1]:+.3f}；预注册判定：{acc4[2]}"]
    else:
        lines += ["- 验收 4：没有结果。"]
    lines += [f"- 位移条款口径：{clause}。"]
    lines.append("")
    return lines


def main(out=None, note=None, clause=None):
    global OUT
    if out:
        OUT = os.path.abspath(out)
    clause = disp_clause_of(clause)
    lad_lines, results, selected, state = ladder_section(clause)
    L_file = os.path.join(OUT, "L_STAR")
    L = GROUP_BY_ID[selected]["L"] if selected else None
    if os.path.exists(L_file):
        try:
            with open(L_file, encoding="utf-8") as f:
                L = float(f.read().strip())
        except ValueError:
            pass
    acc2_lines, acc2 = acceptance2_section(L)
    acc4_lines, acc4, merged = acceptance4_section(L)
    lines = ["# exp-action-latency-r2-20261111 汇总（任务卡 W1-0 第二轮）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["第二轮是看到第一轮失败之后设计的；第一轮的预注册与判定不变（`run.sh`、`summarize.py` 未动）。口径：metric_version `sim09.v2.coast1s.corridor3m`；"
              "带 `--action-latency` 的数据与旧数据不可混报；含迟到步的集算基础设施失败，不计入成功（学生臂重跑一次，见第 6 节）。", ""]
    lines += conclusion_section(clause, results, selected, state, acc2, acc4)
    lines += start_section() + lad_lines + acc2_lines + acc4_lines
    lines += k3_section(L, (merged or {}).get("k3") or {})
    lines += retry_section(L) + forest_section(L) + overview_section(L)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


EXIT = {"PASS": 0, "FAIL_LATE": 1, "INCOMPLETE": 2, "FAIL_NOLATE": 3}


def cli(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="输出目录（默认本脚本所在目录）；--late-seeds 时是臂目录")
    ap.add_argument("--gate", choices=[g["id"] for g in GROUPS], help="只判一组标定；退出码见文件头")
    ap.add_argument("--late-seeds", action="store_true", help="打印臂目录里含迟到步的 seed（逗号分隔）")
    ap.add_argument("--note", help="写在 summary.md 标题下的一行说明")
    ap.add_argument("--disp-clause", choices=DISP_CLAUSES, help="位移条款口径；缺省读 $OUT/config.env 的 DISP_CLAUSE，再缺省 " + DEFAULT_DISP_CLAUSE)
    args = ap.parse_args(argv)
    if args.late_seeds:
        print(",".join(str(s) for s in late_seeds(args.paths[0])))
        return 0
    if args.paths:
        OUT = os.path.abspath(args.paths[0])
    if args.gate:
        res = gate_group(GROUP_BY_ID[args.gate], disp_clause_of(args.disp_clause))
        print(f"{res['status']} {res['text']}")
        return EXIT[res["status"]]
    return main(note=args.note, clause=args.disp_clause)


if __name__ == "__main__":
    sys.exit(cli())
