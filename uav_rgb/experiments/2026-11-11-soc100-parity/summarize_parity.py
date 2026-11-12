#!/usr/bin/env python3
"""满电档（SOC=100%）对 50% 档的配对汇总与预注册判决（任务卡 W1-10，实验 exp-soc100-parity-20261111），写 summary.md 与 VERDICT。

随时可跑，缺的臂标「未完成」，没跑完不给判决。口径直接复用第二轮的 summarize_r2.py（每步前飞位移、转角、重跑合并、K=3 逐块与掉高），
所以「口径同 summarize_r2.py」是同一份代码，不是照抄。

运行：<.venv-tools 解释器> summarize_parity.py [输出目录] [--control 第二轮目录] [--note 说明]
      输出目录缺省 = 本脚本所在目录；对照目录缺省读 <输出目录>/config.env 的 CONTROL，再缺省 $D/exp-action-latency-r2-20261111。
退出码：0 PASS；1 FAIL（有预注册项没过，没触发证伪）；2 INCOMPLETE（数据不全）；3 FALSIFIED（K=1 位移或转角相差 > 10%，两档数据与结论不能互用）；
        4 INVALID（指纹不符或配对协议不一致，数据不可用）。

臂目录：
  对照（50% 电量，第二轮）  <对照目录>/acc-L<L>-k<1|3>-s<0|1|2>，含重跑 <臂>-retry1
  满电                      <输出目录>/soc100-L<L>-k<1|3>-s<0|1|2>，含重跑 <臂>-retry1
  L = <对照目录>/L_STAR（第二轮定的 L，0.20）。每臂 20 个冻结 seed，三个训练种子 calswa，--heading-guard 2。

预注册（任务卡 W1-10「11-11 更新」，逐字；run.sh 头部也有同一份，测试核对两处一致）：
  K=1 的每步前飞位移与转角（口径同 `summarize_r2.py`）与 50% 档相差不超过 5%，成功数配对无可检出的差别（精确 McNemar p > 0.05）；
  K=3 满电档里飞到电量掉高的集数为 0，第 1 块通过率与 50% 档相差不超过 0.08，三块全过的集数不少于 50% 档。证伪：K=1 的位移或转角相差超过 10%。
  差超过 10% 则两档的数据与结论不能互用，报告里写明并停下。

逐项落成（judge 函数，纯函数，有单测）：
  P1 K=1 每步前飞位移：满电对 50% 的相对差 |f/c − 1| ≤ 5%。口径 = 第二轮 config.env 的 DISP_CLAUSE（cruise：连续前进第 4 步起、步 3–30），三个训练种子的 60 集池化均值。
  P2 K=1 转角：YAW 步 |dyaw_deg| 的池化均值，相对差 ≤ 5%。
  P3 K=1 走廊内成功数（success，横向边界 3 m）：同 seed 同训练种子逐集配对，三个种子合并 60 对，精确 McNemar p > 0.05。
  P4 K=3 满电档里 z_tail_min < 1.5 m（末 10 步最低高度，电池压降的症状）的集数 = 0。
  P5 K=3 第 1 块通过率（通过数 / 60）：|满电 − 50%| ≤ 0.08。
  P6 K=3 三块全过的集数：满电 ≥ 50%。
  证伪 F：P1 或 P2 的相对差 > 10%。
  总判决：数据完整且指纹与配对协议核对通过时，F 触发 = FALSIFIED；否则 P1..P6 全过 = PASS，有一项没过 = FAIL。
同报告里另列（不进判决）：位移与转角的中位数、起飞到首步观测的仿真时刻（代理）、K=3 的 S_1/S_2/S_3 与 sim_s、探针的 READY 用时与悬停油门。
"""
import argparse
import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2026-11-10-action-latency"))
import summarize_r2 as s2  # noqa: E402  第二轮汇总：口径函数的唯一来源
s1, lc = s2.s1, s2.lc

D = s2.D
SEEDS3 = s2.SEEDS3
FROZEN_SEEDS = s2.FROZEN_SEEDS
mean, rel, pct, fnum = s2.mean, s2.rel, s2.pct, s2.fnum
mcnemar_exact, wilson = s2.mcnemar_exact, s2.wilson

# ---------------------------------------------------------------- 配置（预注册阈值写在 run.sh 头；改这里要换新文件名与输出目录）
EPISODES = 20                  # 每臂请求集数 = 冻结 seed 个数
REL_PASS = 0.05                # P1/P2：K=1 位移与转角的相对差上限
REL_FALSIFY = 0.10             # 证伪线
P_MIN = 0.05                   # P3：精确 McNemar p 必须大于它
S1_GAP_MAX = 0.08              # P5：第 1 块通过率差上限（绝对值，小数）
DROPS_MAX = 0                  # P4：K=3 满电档的掉高集数上限
EPS = 1e-9                     # 比较的浮点容差：恰好等于阈值算「不超过」（1.05/1.0 − 1 = 0.05000000000000004）
DEFAULT_CONTROL = os.path.join(D, "exp-action-latency-r2-20261111")   # 对照 = 第二轮输出（50% 电量）
EXPECT_CONTROL_PROFILE = "cf_sitl_v1:1bc8fedf:lib31"   # 对照臂各集 manifest 的 dynamics_profile（默认参数，电量 50%）
EXPECT_FULL_PROFILE = "cf_sitl_v1:06e5f017:lib31"      # 满电臂各集 manifest 的 dynamics_profile（起跑时 config.env 可覆盖）
PAIR_KEYS = ("infer_bin_sha256", "ckpt_sha256", "heading_guard", "hold_s", "window_from", "corridor_half_width", "action_latency_s",
             "dynamics", "vehicle", "template_sha256", "split", "requested_seeds", "task", "metric_version", "steps", "blocks", "policy",
             "camera_rate_hz", "aux_cameras")        # 两档对应臂的 eval.json 这些键必须相同，配对才成立
# 不用 seeds：它是评完的集，含迟到步的集进 failed[] 不在里面（真实臂 soc100-L0.20-k1-s1 少 seed 113），用它会让每个重跑过的臂判 INVALID；请求的全集是 requested_seeds
RETRY = s2.RETRY_SUFFIX
EXIT = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "FALSIFIED": 3, "INVALID": 4}


# ---------------------------------------------------------------- 纯函数
def median(xs):
    return statistics.median(xs) if xs else float("nan")


def judge(m):
    """预注册判决（纯函数）。m = {
         "k1": {"disp": (c, f), "turn": (c, f), "b": 只 50% 过的对数, "c": 只满电过的对数, "n": 对数},
         "k3": {"drops": (c, f), "s1": (c, f), "s3": (c, f), "n": 每档集数}}
    c = 50% 档（对照），f = 满电档。返回 {"items": [{id, name, rule, c, f, diff, ok, falsified}], "falsified": bool, "verdict": "PASS"|"FAIL"|"FALSIFIED"}。"""
    items = []

    def rel_item(pid, name, pair):
        c, f = pair
        d = rel(f, c)
        valid = d == d
        items.append({"id": pid, "name": name, "rule": f"相对差的绝对值 ≤ {REL_PASS:.0%}；> {REL_FALSIFY:.0%} 证伪", "c": c, "f": f, "diff": d,
                      "ok": bool(valid and abs(d) <= REL_PASS + EPS), "falsified": bool(valid and abs(d) > REL_FALSIFY + EPS)})

    k1, k3 = m["k1"], m["k3"]
    rel_item("P1", "K=1 每步前飞位移", k1["disp"])
    rel_item("P2", "K=1 每步转角", k1["turn"])
    p = mcnemar_exact(k1["b"], k1["c"])
    items.append({"id": "P3", "name": "K=1 走廊内成功数配对", "rule": f"精确 McNemar p > {P_MIN}", "c": k1["b"], "f": k1["c"], "diff": p,
                  "ok": p > P_MIN, "falsified": False})
    dc, df = k3["drops"]
    items.append({"id": "P4", "name": "K=3 满电档掉高集数", "rule": f"= {DROPS_MAX}", "c": dc, "f": df, "diff": df, "n": k3["n"], "ok": df <= DROPS_MAX, "falsified": False})
    n = k3["n"]
    sc, sf = k3["s1"]
    gap = (sf - sc) / n if n else float("nan")
    items.append({"id": "P5", "name": "K=3 第 1 块通过率", "rule": f"差的绝对值 ≤ {S1_GAP_MAX}", "c": sc, "f": sf, "diff": gap, "n": n,
                  "ok": bool(gap == gap and abs(gap) <= S1_GAP_MAX + EPS), "falsified": False})
    tc, tf = k3["s3"]
    items.append({"id": "P6", "name": "K=3 三块全过集数", "rule": "满电 ≥ 50%", "c": tc, "f": tf, "diff": tf - tc, "n": n, "ok": tf >= tc, "falsified": False})
    falsified = any(i["falsified"] for i in items)
    verdict = "FALSIFIED" if falsified else ("PASS" if all(i["ok"] for i in items) else "FAIL")
    return {"items": items, "falsified": falsified, "verdict": verdict}


def pooled_samples(dirs, clause):
    """若干臂（首次记录目录）合并的逐步样本：{"disp": [位移], "yaw": [|dyaw|], "step0_s": [首步观测仿真秒]}。
    位移 = summarize_r2.disp_stats 同一条件（clause 对应的 *_common：步 3–30，cruise r≥4 / steady r≥2）；转角 = s1.timing_stats 同一条件。"""
    disp, yaw, step0 = [], [], []
    for edir in dirs:
        for _seed, actions, _oracle, _man in s1.episode_records(edir):
            if not actions:
                continue
            for i, _prev, d, r in s2.fwd_samples(actions):
                need = s2.CRUISE_MIN_RUN if clause == "cruise" else 2
                if r >= need and s2.COMMON_LO <= i <= s2.COMMON_HI:
                    disp.append(d)
            yaw += [abs(a["dyaw_deg"]) for a in actions if a.get("action_executed") in (2, 3) and a.get("dyaw_deg") is not None]
            if actions[0].get("timestamp_ns") is not None:
                step0.append(actions[0]["timestamp_ns"] / 1e9)
    return {"disp": disp, "yaw": yaw, "step0_s": step0}


def step0_by_seed(edir):
    """{seed: 首步观测的仿真秒}（含起飞前的等待，只作起飞到就绪用时的代理）。"""
    out = {}
    for seed, actions, _o, _m in s1.episode_records(edir):
        if actions and actions[0].get("timestamp_ns") is not None:
            out[seed] = actions[0]["timestamp_ns"] / 1e9
    return out


def pair_counts(ctrl_merged, full_merged, k):
    """同 seed 配对：{n, c_ok, f_ok, b（只 50% 过）, c（只满电过）}。迟到（最终没拿到干净结果）按没成功，口径同 s2.success_map。"""
    mc, mf = s2.success_map(ctrl_merged, k), s2.success_map(full_merged, k)
    seeds = [x for x in FROZEN_SEEDS if x in mc and x in mf]
    return {"n": len(seeds), "c_ok": sum(mc[x] for x in seeds), "f_ok": sum(mf[x] for x in seeds),
            "b": sum(1 for x in seeds if mc[x] and not mf[x]), "c": sum(1 for x in seeds if (not mc[x]) and mf[x])}


def profiles_of(edir):
    """臂目录各集 manifest 的 dynamics_profile 计数；没有该键记「(无)」。"""
    c = collections.Counter()
    for _s, _a, _o, man in s1.episode_records(edir):
        c[man.get("dynamics_profile") or "(无)"] += 1
    return dict(c)


def protocol_diff(ctrl_dir, full_dir):
    """两个臂 eval.json 里 PAIR_KEYS 不同的键 → [(键, 对照值, 满电值)]；任一没有 eval.json 返回 None。"""
    pc, pf = os.path.join(ctrl_dir, "eval.json"), os.path.join(full_dir, "eval.json")
    if not (os.path.exists(pc) and os.path.exists(pf)):
        return None
    ec, ef = s1.read_json(pc), s1.read_json(pf)
    return [(k, ec.get(k), ef.get(k)) for k in PAIR_KEYS if ec.get(k) != ef.get(k)]


def read_kv(path):
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    cfg[k.strip()] = v.strip()
    return cfg


def read_cg(root, name):
    old, s2.OUT = s2.OUT, root
    try:
        return s2.read_cg(name)
    finally:
        s2.OUT = old


# ---------------------------------------------------------------- 读数
def names(L):
    return (lambda k, s: f"acc-L{L:.2f}-k{k}-s{s}"), (lambda k, s: f"soc100-L{L:.2f}-k{k}-s{s}")


def load_variant(root, namer):
    out = {}
    for k in (1, 3):
        for s in SEEDS3:
            edir = os.path.join(root, namer(k, s))
            out[(k, s)] = {"dir": edir, "merged": s2.merged_rows(edir, EPISODES), "counts": s2.arm_counts(edir, EPISODES)}
    return out


def completeness(variant, data):
    """没完成的原因列表（空 = 12 个臂中该档 6 个都完整）。"""
    why = []
    for (k, s), d in sorted(data.items()):
        c = d["counts"]
        if d["merged"] is None or not c["complete"] or len(d["merged"]) < EPISODES:
            why.append(f"{variant} K={k} s{s}：{c['state']}")
        elif any(it["retry"] == "missing" for it in d["merged"].values()):
            why.append(f"{variant} K={k} s{s}：含迟到步的集还没重跑")
    return why


def analyze(out, control, L, clause, expect_full=EXPECT_FULL_PROFILE):
    ctrl_name, full_name = names(L)
    A = {"out": out, "control": control, "L": L, "clause": clause, "expect_full": expect_full,
         "ctrl": load_variant(control, ctrl_name), "full": load_variant(out, full_name)}
    A["ctrl_name"], A["full_name"] = ctrl_name, full_name
    A["missing"] = completeness("对照", A["ctrl"]) + completeness("满电", A["full"])
    # 指纹与协议核对
    A["profiles"] = {}
    A["bad_profiles"] = []
    for variant, data, want in (("对照", A["ctrl"], EXPECT_CONTROL_PROFILE), ("满电", A["full"], expect_full)):
        for (k, s), d in sorted(data.items()):
            for suffix in ("", RETRY):
                edir = d["dir"] + suffix
                if not os.path.isdir(os.path.join(edir, "episodes")):
                    continue
                prof = profiles_of(edir)
                A["profiles"][(variant, k, s, suffix)] = prof
                bad = {p: n for p, n in prof.items() if p != want}
                if bad:
                    A["bad_profiles"].append(f"{variant} K={k} s{s}{suffix}：{bad}（应为 {want}）")
    A["proto_diff"] = {}
    for k in (1, 3):
        for s in SEEDS3:
            diff = protocol_diff(A["ctrl"][(k, s)]["dir"], A["full"][(k, s)]["dir"])
            if diff:
                A["proto_diff"][(k, s)] = diff
    A["metric_versions"] = sorted({s1.read_json(p).get("metric_version") or "(无)" for p in (
        os.path.join(d["dir"], "eval.json") for v in ("ctrl", "full") for d in A[v].values()) if os.path.exists(p)})
    # K=1
    A["k1"] = {}
    for variant in ("ctrl", "full"):
        dirs = [A[variant][(1, s)]["dir"] for s in SEEDS3 if A[variant][(1, s)]["merged"] is not None]
        A["k1"][variant] = {"pooled": pooled_samples(dirs, clause),
                            "per": {s: pooled_samples([A[variant][(1, s)]["dir"]], clause) for s in SEEDS3
                                    if A[variant][(1, s)]["merged"] is not None}}
    A["pairs1"], A["pairs3"] = {}, {}
    for s in SEEDS3:
        c1, f1, c3, f3 = A["ctrl"][(1, s)]["merged"], A["full"][(1, s)]["merged"], A["ctrl"][(3, s)]["merged"], A["full"][(3, s)]["merged"]
        if c1 is not None and f1 is not None:
            A["pairs1"][s] = pair_counts(c1, f1, 1)
        if c3 is not None and f3 is not None:
            A["pairs3"][s] = pair_counts(c3, f3, 3)
    # K=3
    A["k3"] = {}
    for variant in ("ctrl", "full"):
        per = {s: s2.k3_block_stats(A[variant][(3, s)]["merged"]) for s in SEEDS3 if A[variant][(3, s)]["merged"] is not None}
        aborted = {s: sum(1 for it in A[variant][(3, s)]["merged"].values() if it["row"].get("aborted")) for s in per}
        pooled = {"n": sum(v["n"] for v in per.values()), "S": [sum(v["S"][j] for v in per.values()) for j in range(3)],
                  "drops": sum(v["drops"] for v in per.values()), "sim_s": [x for v in per.values() for x in v["sim_s"]],
                  "steps": [x for v in per.values() for x in v["steps"]], "aborted": sum(aborted.values())}
        A["k3"][variant] = {"per": per, "aborted": aborted, "pooled": pooled}
    # 起飞到首步观测（代理）：逐 (K, 训练种子, seed) 配对
    A["step0"] = {}
    for k in (1, 3):
        diffs, cs, fs = [], [], []
        for s in SEEDS3:
            sc, sf = step0_by_seed(A["ctrl"][(k, s)]["dir"]), step0_by_seed(A["full"][(k, s)]["dir"])
            cs += list(sc.values())
            fs += list(sf.values())
            diffs += [sf[x] - sc[x] for x in sc if x in sf]
        A["step0"][k] = {"ctrl": cs, "full": fs, "diff": diffs}
    # 判决输入
    A["status"] = "INCOMPLETE" if A["missing"] else ("INVALID" if (A["bad_profiles"] or A["proto_diff"]) else None)
    if A["status"] is None:
        d1c, d1f = A["k1"]["ctrl"]["pooled"], A["k1"]["full"]["pooled"]
        b = sum(v["b"] for v in A["pairs1"].values())
        c = sum(v["c"] for v in A["pairs1"].values())
        pc, pf = A["k3"]["ctrl"]["pooled"], A["k3"]["full"]["pooled"]
        m = {"k1": {"disp": (mean(d1c["disp"]), mean(d1f["disp"])), "turn": (mean(d1c["yaw"]), mean(d1f["yaw"])),
                    "b": b, "c": c, "n": sum(v["n"] for v in A["pairs1"].values())},
             "k3": {"drops": (pc["drops"], pf["drops"]), "s1": (pc["S"][0], pf["S"][0]), "s3": (pc["S"][2], pf["S"][2]), "n": pf["n"]}}
        A["judge_input"] = m
        A["judge"] = judge(m)
        A["status"] = A["judge"]["verdict"]
    return A


# ---------------------------------------------------------------- 输出
def f4(x):
    return fnum(x, 4)


def probe_section(out):
    path = os.path.join(out, "probe", "probe.json")
    lines = ["## 6. 插件读参数的 Gazebo 探针（起跑前，同一个守护进程）", ""]
    if not os.path.exists(path):
        return lines + ["没有 probe/probe.json（探针没跑，或这个目录不是 run.sh 的输出）。", ""]
    p = s1.read_json(path)
    if p.get("infra_error"):
        return lines + [f"探针基础设施失败：{p['infra_error']}。", ""]
    if not isinstance(p.get("verdict"), dict) or "per_variant" not in p["verdict"]:
        return lines + ["probe/probe.json 里没有 verdict（格式不符，未核实）。", ""]
    v, pv = p["verdict"], p["verdict"]["per_variant"]
    lines += [f"- 判决：**{'通过' if v['ok'] else '未通过'}**：静止电压 copy − default = {fnum(v['rest_v_gap'])} V（门 ≥ {v['min_rest_v_gap']} V）；"
              f"插件日志的 params= 行读到 {v['log_lines_checked']}/{v['log_lines_total']} 个。"] + [f"- 未过：{x}" for x in v["reasons"]]
    lines += ["", "| 世界 | 到 READY 次数 | 静止 V_bus 中位 V | READY V_bus 中位 V | READY 悬停油门 中位 | READY 油门杆量 中位 | READY 用时（SITL s）中位 | 逐次 READY 用时 s |", "|---|---|---|---|---|---|---|---|"]
    for name in ("default", "copy"):
        x = pv[name]
        lines.append(f"| {name} | {x['n_ok']}/{x['n']} | {fnum(x['rest_v_median'])} | {fnum(x['ready_v_median'])} | {fnum(x['ready_hover_median'])} | "
                     f"{fnum(x['ready_throttle_median'])} | {fnum(x['ready_sitl_s_median'], 2)} | " + ", ".join(f"{t:.2f}" for t in x["ready_sitl_s_all"]) + " |")
    lines += ["", "完整逐次记录见 `probe/probe.md`、`probe/probe.json`、各 gz 日志。", ""]
    return lines


def render(A, note=None, done_text=None):
    L, clause = A["L"], A["clause"]
    lines = ["# exp-soc100-parity-20261111 汇总（任务卡 W1-10：满电档对 50% 档）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["对照 = 第二轮 `" + A["control"] + f"` 里 50% 电量的六个学生臂 `acc-L{L:.2f}-k{{1,3}}-s{{0,1,2}}`；满电 = 本目录 `soc100-L{L:.2f}-k{{1,3}}-s{{0,1,2}}`。"
              f"位移口径 = {clause}（取自第二轮 config.env，函数是 summarize_r2.py 里的同一份）。含迟到步的集按条款 R6 重跑一次，以重跑结果为准。"
              f"评测口径 metric_version `{', '.join(A['metric_versions']) or '未核实'}`（取自各臂 eval.json）；成功数的分母 = 请求集数（每臂 {EPISODES} 个冻结 seed）。", ""]
    lines += ["## 0. 结论一览（自动生成，数字都来自下面各节）", ""]
    if done_text:
        lines.append(f"- DONE：`{done_text}`")
    status = A["status"]
    lines.append(f"- **判决：{status}**")
    if status == "INCOMPLETE":
        lines += ["- 没完成：" + "；".join(A["missing"][:8]) + ("…" if len(A["missing"]) > 8 else "")]
    elif status == "INVALID":
        lines += [f"- 指纹不符：{x}" for x in A["bad_profiles"]] + [f"- 配对协议不一致 K={k} s{s}：{d}" for (k, s), d in sorted(A["proto_diff"].items())]
        lines.append("- 指纹或配对协议不一致时，两档数据不可配对，不给 P1..P6 的判决。")
    else:
        J = A["judge"]
        lines += ["", "| 条款 | 内容 | 预注册规则 | 50% 档 | 满电档 | 差 | 判定 |", "|---|---|---|---|---|---|---|"]
        for it in J["items"]:
            lines.append(item_row(it))
        lines.append("")
        if J["falsified"]:
            lines += ["- **证伪触发**：K=1 的位移或转角相差超过 10%。两档的数据与结论不能互用，报告里写明并停下。"]
        elif J["verdict"] == "PASS":
            lines += ["- 预注册 P1..P6 全部满足。"]
        else:
            lines += ["- 没过的条款：" + "、".join(f"{i['id']} {i['name']}" for i in J["items"] if not i["ok"]) + "（未到证伪线）。"]
    lines.append("")

    lines += ["## 1. 臂与完整性", "", "| 档 | 臂 | 请求 | 完成 | 采集失败（迟到 / 其它） | 重跑 | 状态 |", "|---|---|---|---|---|---|---|"]
    for variant, key in (("对照", "ctrl"), ("满电", "full")):
        for (k, s), d in sorted(A[key].items()):
            c = d["counts"]
            retry = "-"
            if d["merged"]:
                rs = collections.Counter(it["retry"] for it in d["merged"].values() if it["first_late"])
                retry = "，".join(f"{n} 集 {r}" for r, n in sorted(rs.items())) or "-"
            lines.append(f"| {variant} | {os.path.basename(d['dir'])} | {c['requested']} | {c['done']} | {c['failed']}（{len(c['late'])} / {c['other_failed']}） | {retry} | {c['state']} |")
    lines.append("")

    lines += ["## 2. 指纹与配对协议核对", "",
              f"预期：对照臂各集 `dynamics_profile` = `{EXPECT_CONTROL_PROFILE}`，满电臂 = `{A['expect_full']}`；不符则整档数据不可用（判决 INVALID）。", "",
              "| 档 | 臂 | manifest.dynamics_profile（集数） |", "|---|---|---|"]
    for (variant, k, s, suffix), prof in sorted(A["profiles"].items()):
        lines.append(f"| {variant} | K={k} s{s}{suffix} | " + "，".join(f"`{p}` × {n}" for p, n in sorted(prof.items())) + " |")
    if A["proto_diff"]:
        lines += ["", "对应臂的 eval.json 里下列键不同：", ""] + [f"- K={k} s{s}：{d}" for (k, s), d in sorted(A["proto_diff"].items())]
    else:
        lines += ["", f"对应臂的 eval.json 在 {len(PAIR_KEYS)} 个协议键（权重与推理二进制 sha256、护栏、hold、window_from、L、走廊、模板、seed 等）上逐臂相同。" if all(
            os.path.exists(os.path.join(A['ctrl'][(k, s)]['dir'], 'eval.json')) and os.path.exists(os.path.join(A['full'][(k, s)]['dir'], 'eval.json'))
            for k in (1, 3) for s in SEEDS3) else "（有臂还没有 eval.json，协议核对只对已有的臂）"]
    lines.append("")

    # K=1
    lines += ["## 3. K=1（柱阵，每步 1.0 s，L 固定）：位移、转角、成功数", "",
              f"位移 = 位移口径 `{clause}`、步 3–30 的逐步样本；转角 = YAW 步 |dyaw_deg|。样本取首次记录目录（与 summarize_r2.py 的 timing_row 同口径，含迟到集）。", "",
              "| 档 | 训练种子 | 位移均值 m（n） | 位移中位数 m | 转角均值 °（n） | 转角中位数 ° |", "|---|---|---|---|---|---|"]
    for variant, key in (("对照", "ctrl"), ("满电", "full")):
        for s, x in sorted(A["k1"][key]["per"].items()):
            lines.append(f"| {variant} | s{s} | {f4(mean(x['disp']))}（{len(x['disp'])}） | {f4(median(x['disp']))} | {fnum(mean(x['yaw']), 2)}（{len(x['yaw'])}） | {fnum(median(x['yaw']), 2)} |")
        x = A["k1"][key]["pooled"]
        lines.append(f"| **{variant}** | 合并 | **{f4(mean(x['disp']))}**（{len(x['disp'])}） | {f4(median(x['disp']))} | **{fnum(mean(x['yaw']), 2)}**（{len(x['yaw'])}） | {fnum(median(x['yaw']), 2)} |")
    pc, pf = A["k1"]["ctrl"]["pooled"], A["k1"]["full"]["pooled"]
    lines += ["", f"相对差（满电 / 50% − 1）：位移均值 {pct(rel(mean(pf['disp']), mean(pc['disp'])))}，位移中位数 {pct(rel(median(pf['disp']), median(pc['disp'])))}，"
              f"转角均值 {pct(rel(mean(pf['yaw']), mean(pc['yaw'])))}，转角中位数 {pct(rel(median(pf['yaw']), median(pc['yaw'])))}。预注册判的是均值（口径同 summarize_r2.py）。", ""]
    if A["pairs1"]:
        lines += ["走廊内成功数（`success`，横向边界 3 m；迟到集重跑后以重跑为准）：", "",
                  "| 训练种子 | 对数 | 50% 成功 | 满电成功 | 只 50% 过 b | 只满电过 c | 精确 McNemar p |", "|---|---|---|---|---|---|---|"]
        for s, v in sorted(A["pairs1"].items()):
            lines.append(f"| s{s} | {v['n']} | {v['c_ok']} | {v['f_ok']} | {v['b']} | {v['c']} | {mcnemar_exact(v['b'], v['c']):.3f} |")
        n = sum(v["n"] for v in A["pairs1"].values())
        co, fo = sum(v["c_ok"] for v in A["pairs1"].values()), sum(v["f_ok"] for v in A["pairs1"].values())
        b, c = sum(v["b"] for v in A["pairs1"].values()), sum(v["c"] for v in A["pairs1"].values())
        lines += [f"| **合并** | {n} | {lc.fmt_frac(co, n)} | {lc.fmt_frac(fo, n)} | {b} | {c} | **{mcnemar_exact(b, c):.3f}** |", "",
                  f"不一致对数 b + c = {b + c}；p > {P_MIN} 只说明没有检出差别，60 对且不一致对少时检出力有限，不等于两档等价。同训练种子的 20 集不独立，p 只作参考（预注册按它判）。", ""]

    # K=3
    lines += ["## 4. K=3（长课程 3 块，240 步预算）：逐块存活、掉高、sim_s", "",
              "S_k = 通过前 k 块的集数；掉高 = 末 10 步最低高度 < 1.5 m（电池压降的症状）；采集中止 = 固件 FAILSAFE 等采集端失败（计入没通过）。", "",
              "| 档 | 训练种子 | 集数 | S_1 | S_2 | S_3 | 掉高集数 | 采集中止 | sim_s 中位 s | 步数 中位 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for variant, key in (("对照", "ctrl"), ("满电", "full")):
        for s, v in sorted(A["k3"][key]["per"].items()):
            lines.append(f"| {variant} | s{s} | {v['n']} | {v['S'][0]} | {v['S'][1]} | {v['S'][2]} | {v['drops']} | {A['k3'][key]['aborted'][s]} | {fnum(median(v['sim_s']), 1)} | {fnum(median(v['steps']), 0)} |")
        v = A["k3"][key]["pooled"]
        lines.append(f"| **{variant}** | 合并 | {v['n']} | **{v['S'][0]}** | {v['S'][1]} | **{v['S'][2]}** | **{v['drops']}** | {v['aborted']} | {fnum(median(v['sim_s']), 1)} | {fnum(median(v['steps']), 0)} |")
    if A["pairs3"]:
        lines += ["", "第 1 块通过的逐集配对（同 seed 同训练种子）：", "", "| 训练种子 | 对数 | 50% 过 | 满电过 | 只 50% 过 b | 只满电过 c | 精确 McNemar p |", "|---|---|---|---|---|---|---|"]
        for s, v in sorted(A["pairs3"].items()):
            lines.append(f"| s{s} | {v['n']} | {v['c_ok']} | {v['f_ok']} | {v['b']} | {v['c']} | {mcnemar_exact(v['b'], v['c']):.3f} |")
        b, c = sum(v["b"] for v in A["pairs3"].values()), sum(v["c"] for v in A["pairs3"].values())
        lines.append(f"| **合并** | {sum(v['n'] for v in A['pairs3'].values())} | {sum(v['c_ok'] for v in A['pairs3'].values())} | {sum(v['f_ok'] for v in A['pairs3'].values())} | {b} | {c} | **{mcnemar_exact(b, c):.3f}** |")
    lines.append("")

    # 起飞代理
    lines += ["## 5. 起飞到就绪的时间（代理）", "",
              "每集第 0 步的观测时刻（`actions.jsonl` 第 0 行 `timestamp_ns`，Gazebo 仿真秒）= gz 起来后的等待 + reset + 解锁起飞到 READY + 稳定。"
              "解锁到 READY 的用时本身没有写进逐集输出（`CfSitlExecutor` 只写日志行），所以这里是上界型的代理；干净的 READY 用时见第 6 节探针。", "",
              "| K | 档 | 集数 | 首步观测 仿真秒 中位 | 最小 | 最大 | 配对差（满电 − 50%）中位 | 配对差 均值 | 配对数 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in (1, 3):
        x = A["step0"][k]
        for variant, key in (("对照", "ctrl"), ("满电", "full")):
            v = x[key]
            lines.append(f"| {k} | {variant} | {len(v)} | {fnum(median(v), 2)} | {fnum(min(v) if v else float('nan'), 2)} | {fnum(max(v) if v else float('nan'), 2)} | | | |")
        lines.append(f"| {k} | 配对 | | | | | **{fnum(median(x['diff']), 2)}** | {fnum(mean(x['diff']), 2)} | {len(x['diff'])} |")
    lines.append("")
    lines += probe_section(A["out"])

    # 起跑与资源
    st = s2.read_start(A["out"])
    lines += ["## 7. 起跑条件、输入核对与资源", ""]
    if st:
        lines += [f"- 起跑条件：GPU 在 `nvidia-smi -L` 里 = {st.get('gpu_ok', '-')}；cgroup 配额 {st.get('cpu_quota_cores', '-')} 核（要求 ≥ {st.get('min_cores', '-')}）；"
                  f"等机器锁 {st.get('lock_wait_s', '-')} s；拿到锁后 gz sim 进程数 {st.get('gz_count', '-')}；{st.get('idle_window_s', '-')} s 窗口内容器平均 {st.get('idle_cores', '-')} 核；"
                  f"起跑 {st.get('started_at', '-')}，车道 {st.get('lanes', '-')}。"]
    else:
        lines += ["- 没有 start.txt（守护进程还没起，或这个目录不是 run.sh 的输出）。"]
    diff_path = os.path.join(A["out"], "code_diff_vs_control.txt")
    if os.path.exists(diff_path):
        with open(diff_path, encoding="utf-8") as f:
            lines += ["- 本实验冻结代码与第二轮冻结代码在 `sim/` 下的差异文件（run.sh 起跑时记录，允许清单之外的差异会拒绝起跑）：", "", "```", f.read().rstrip(), "```"]
    lines += ["", "| 臂 | 墙钟 s | 平均核数 | 限流增量 |", "|---|---|---|---|"]
    for key, root in (("full", A["out"]),):
        for (k, s), d in sorted(A[key].items()):
            cg = read_cg(root, os.path.basename(d["dir"]))
            lines.append(f"| {os.path.basename(d['dir'])} | " + ("- | - | - |" if cg is None else f"{cg['wall_s']:.0f} | {cg['avg_cores']:.1f} | {cg['throttled']}" + (" ⚠" if cg["throttled"] > 0 else "") + " |"))
    lines.append("")
    return lines


def verdict_text(A):
    if A["status"] in ("PASS", "FAIL", "FALSIFIED"):
        J = A["judge"]
        body = "；".join(f"{i['id']} {'过' if i['ok'] else ('证伪' if i['falsified'] else '未过')}" for i in J["items"])
        return f"{A['status']} {body}"
    if A["status"] == "INCOMPLETE":
        return "INCOMPLETE " + "；".join(A["missing"][:4])
    return "INVALID " + "；".join(A["bad_profiles"][:2] + [f"K={k} s{s} 协议键不同" for (k, s) in sorted(A["proto_diff"])[:2]])


def item_row(it):
    pid = it["id"]
    if pid in ("P1", "P2"):
        c, f, d = f4(it["c"]) if pid == "P1" else fnum(it["c"], 2), f4(it["f"]) if pid == "P1" else fnum(it["f"], 2), pct(it["diff"])
    elif pid == "P3":
        c, f, d = f"b={it['c']}", f"c={it['f']}", f"p={it['diff']:.3f}"
    elif pid == "P5":
        c, f, d = f"{it['c']}/{it['n']}", f"{it['f']}/{it['n']}", f"{it['diff']:+.3f}"
    else:
        c, f, d = f"{it['c']}/{it['n']}", f"{it['f']}/{it['n']}", f"{it['diff']:+d}"
    res = "证伪" if it["falsified"] else ("过" if it["ok"] else "未过")
    return f"| {pid} | {it['name']} | {it['rule']} | {c} | {f} | {d} | **{res}** |"


def main(out=None, control=None, note=None):
    out = os.path.abspath(out or HERE)
    cfg = read_kv(os.path.join(out, "config.env"))
    control = os.path.abspath(control or cfg.get("CONTROL") or DEFAULT_CONTROL)
    L_file = os.path.join(control, "L_STAR")
    if not os.path.exists(L_file):
        print(f"没有 {L_file}：第二轮没有定下 L，本实验不能配对。", file=sys.stderr)
        return EXIT["INCOMPLETE"]
    with open(L_file, encoding="utf-8") as f:
        L = float(f.read().strip())
    clause = cfg.get("DISP_CLAUSE") or s2.read_config(control).get("DISP_CLAUSE") or s2.DEFAULT_DISP_CLAUSE
    A = analyze(out, control, L, clause, cfg.get("PARAMS_PROFILE") or EXPECT_FULL_PROFILE)
    done_text = None
    done = os.path.join(out, "DONE")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as f:
            done_text = f.read().strip()
    text = "\n".join(render(A, note, done_text)) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out, "VERDICT"), "w", encoding="utf-8") as f:
        f.write(verdict_text(A) + "\n")
    print(text)
    return EXIT[A["status"]]


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", help="输出目录（缺省本脚本所在目录）")
    ap.add_argument("--control", help="第二轮目录（缺省读 config.env 的 CONTROL）")
    ap.add_argument("--note", help="写在标题下的一行说明")
    args = ap.parse_args(argv)
    return main(args.out, args.control, args.note)


if __name__ == "__main__":
    sys.exit(cli())
