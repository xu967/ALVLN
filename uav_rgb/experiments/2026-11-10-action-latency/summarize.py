#!/usr/bin/env python3
"""exp-action-latency-20261110 汇总（任务卡 W1-0「动作派发时刻固定到仿真时间」的标定与验收），写 summary.md。随时可跑，缺的臂标「未完成」。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py [输出目录，默认本脚本所在目录]
      ... summarize.py --gate L [输出目录]     只判标定门（卡的验收 1）：stdout 一行 PASS|FAIL|INCOMPLETE + 原因；退出码 0 / 1 / 2
      ... summarize.py --timing [标签=]目录[,目录...] ...   只打印时序表（读 episodes/seed_N，任意臂目录都行，含旧口径的臂）
      ... summarize.py --note "说明" [输出目录]   在 summary.md 标题下加一行（事后重跑时写原因与提交哈希）

臂目录（run.sh 的命名）：cal-L<L>-k<K>（教师标定）、acc-L<L>-k<K>-s<种子>（学生验收；k1-s1 在 L=0.16 时同时是卡的验收 2）、forest-<密度>（森林延迟分布）。
逐步量（读 episodes/seed_N 下的 actions.jsonl、oracle.jsonl、episode_manifest.json；含迟到集：迟到集不在 eval.json 的 per_episode，但轨迹照样留着）：
  生效时长 eff = oracle.action_end_ns − oracle.action_start_ns − actions.dispatch_lag_s（派发位姿到窗口末位姿的仿真秒）；
    集间中位 = 各集 eff 均值的中位数（与 11-10 长课程汇总同口径：0.871 s 是那个口径下的 K=1 旧值）；逐步均值 / 最小 / 最大另列。
  稳态前飞位移 = 本步与上一步都执行 FORWARD_SLOW 且本步无碰撞的步的 displacement_m 池化均值；转角 = YAW_LEFT / YAW_RIGHT 步 |dyaw_deg| 池化均值。
  dispatch_lag_s 中位 / 最大 = 全部步池化；迟到步 = actions.late_dispatch 为 true 的步（就绪时仿真戳已超过 观测戳 + L + 0.02 s）。
验收 1（标定门）：K=3 对 K=1 的生效时长、稳态前飞位移、转角三项相对差都不超过 2%，且两个臂的迟到步都为 0（含每集第 0 步）。
验收 2：学生 calswa s1 + 护栏 2、K=1、20 个冻结 seed、L=0.16，与 11-10 下午、晚上两次同配置评测逐集配对（预测：逐集一致率不低于 70% = 14/20）。
验收 3：森林三档不带 --action-latency 的延迟分布（教师）；给出 L=0.16 / 0.20 下各档需要的 --rtf（见 forest_section 的公式）。
验收 4：K=1 对 K=3 第 1 块通过率（学生 calswa 三个训练种子 × 20 个冻结 seed，同权重同 seed 配对，精确 McNemar）。
"""
import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from math import comb

D = os.environ.get("UAVRGB_DATA", "/root/rivermind-data/uav_rgb")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE                                              # 默认输出目录 = 本脚本所在目录（run.sh 把它拷进 $OUT）；命令行第 1 个位置参数可覆盖
FROZEN_SEEDS = [1, 113, 122, 153, 172, 176, 180, 230, 250, 536, 555, 628, 683, 798, 929, 990, 992, 997, 1113, 1654]   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
LADDER = (0.16, 0.20)         # 预注册：先 0.16，不过再试 0.20
CAL_EPISODES = 10             # 每个标定臂的 train 分区集数
HOLD_S = 1.0
FRAME_AGE_S = 0.114           # 相机 10 Hz 的帧龄（任务卡 W1-0 背景：固定约 0.114 s）；只用于森林的 --rtf 推算
LATE_TOL_S = 0.02             # 迟到阈值 = L + 0.02 s（collect.LATE_DISPATCH_TOL_NS）
GATE_REL = 0.02               # 预注册：K=3 对 K=1 三项的相对差上限
AGREE_MIN = 0.70              # 预注册：验收 2 的逐集一致率下限
GAP_SMALL, GAP_LARGE = 0.08, 0.12   # 预注册：验收 4 的两条线
SEEDS3 = (0, 1, 2)            # 学生训练种子
# 旧口径参照（无 --action-latency）：11-10 的评测
REF_K1 = [os.path.join(D, "exp-heading-guard-20261110", f"guard2-calswa-s{s}") for s in SEEDS3]       # 下午，50/60
REF_K1_AFTERNOON_S1 = REF_K1[1]                                                                        # 下午 17/20
REF_K1_EVENING_S1 = os.path.join(D, "exp-k1-control-20261110", "guard2-calswa-s1")                    # 晚上对照 17/20，与下午逐集一致 16/20
REF_K3 = [os.path.join(D, "exp-long-course-v2-20261110", f"guard2-calswa-s{s}") for s in SEEDS3]     # 晚上长课程第二轮，第 1 块 39/60


def load_lc():
    """11-10 长课程汇总脚本里的函数（逐块判据 block_outcomes、load_arm、wilson、mcnemar_exact）：用冻结代码里的那份，没有就用仓库里的。"""
    for p in (os.path.join(OUT, "code", "experiments", "2026-11-10-long-course", "summarize.py"),
              os.path.join(HERE, "..", "2026-11-10-long-course", "summarize.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("lc_summarize", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    sys.exit("找不到 experiments/2026-11-10-long-course/summarize.py")


lc = load_lc()
read_jsonl, read_json, wilson, mcnemar_exact = lc.read_jsonl, lc.read_json, lc.wilson, lc.mcnemar_exact


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def rel(a, b):
    """a 相对 b 的差（小数）；缺值 nan。"""
    if a is None or b is None or a != a or b != b or b == 0:
        return float("nan")
    return a / b - 1.0


def pct(x):
    return "-" if x is None or x != x else f"{100.0 * x:+.1f}%"


def fnum(x, nd=4):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def cal_dir(L, k):
    return os.path.join(OUT, f"cal-L{L:.2f}-k{k}")


def acc_dir(L, k, s):
    return os.path.join(OUT, f"acc-L{L:.2f}-k{k}-s{s}")


def eval_state(edir, requested):
    """(eval.json 或 None, 状态文字, 是否完整)。完整 = 不是 partial，且 出结果 + 采集失败 = 请求数。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None, ("进行中（有轨迹目录）" if os.path.isdir(os.path.join(edir, "episodes")) else "未跑"), False
    e = read_json(path)
    done, failed = e.get("episodes_done") or 0, e.get("failed_count") or 0
    complete = bool(not e.get("partial") and done + failed == requested)
    return e, ("完成" if complete else f"未完成（done {done} + failed {failed} / 请求 {requested}{'，partial' if e.get('partial') else ''}）"), complete


def episode_records(edir):
    """[(seed, actions, oracle, manifest)]，读 episodes/seed_N；没有目录返回 []。"""
    root = os.path.join(edir, "episodes")
    out = []
    if os.path.isdir(root):
        seeds = sorted(int(n[5:]) for n in os.listdir(root) if re.fullmatch(r"seed_\d+", n))
        for s in seeds:
            ep = os.path.join(root, f"seed_{s}")
            man = os.path.join(ep, "episode_manifest.json")
            out.append((s, read_jsonl(os.path.join(ep, "actions.jsonl")), read_jsonl(os.path.join(ep, "oracle.jsonl")),
                        read_json(man) if os.path.exists(man) else {}))
    return out


def timing_stats(dirs):
    """若干臂目录合并的时序量；没有可用的集返回 None。旧口径的集（没有 dispatch_lag_s）只给集间 eff（manifest.performance）。"""
    eff_ep, eff_all, lag_all, fwd, yaw, step0 = [], [], [], [], [], []
    late_steps = steps = n_eps = n_with_lag = 0
    late_eps, flagged = [], False
    for edir in dirs:
        for seed, actions, oracle, man in episode_records(edir):
            if not actions:
                continue
            n_eps += 1
            steps += len(actions)
            lags = [a.get("dispatch_lag_s") for a in actions]
            if all(x is not None for x in lags) and len(oracle) >= len(actions):
                n_with_lag += 1
                eff = [(o["action_end_ns"] - o["action_start_ns"]) / 1e9 - a["dispatch_lag_s"] for a, o in zip(actions, oracle)]
                eff_ep.append(mean(eff))
                eff_all += eff
                lag_all += lags
            else:
                perf = man.get("performance") or {}
                if perf.get("steps") and perf.get("action_sim_s") is not None:
                    eff_ep.append(perf["action_sim_s"] / perf["steps"])
            if any("late_dispatch" in a for a in actions):
                flagged = True
                late = [i for i, a in enumerate(actions) if a.get("late_dispatch")]
                late_steps += len(late)
                if late:
                    late_eps.append((os.path.basename(edir), seed, late))
            step0.append((os.path.basename(edir), seed, actions[0].get("dispatch_lag_s"), actions[0].get("late_dispatch")))
            fwd += [a["displacement_m"] for prev, a in zip(actions, actions[1:])
                    if a.get("action_executed") == 1 and prev.get("action_executed") == 1 and not a.get("collision")
                    and a.get("displacement_m") is not None]
            yaw += [abs(a["dyaw_deg"]) for a in actions if a.get("action_executed") in (2, 3) and a.get("dyaw_deg") is not None]
    if not n_eps:
        return None
    return {"n_eps": n_eps, "n_with_lag": n_with_lag, "steps": steps,
            "eff_med": median(eff_ep), "eff_mean": mean(eff_all), "eff_min": min(eff_all) if eff_all else float("nan"),
            "eff_max": max(eff_all) if eff_all else float("nan"),
            "lag_med": median(lag_all), "lag_max": max(lag_all) if lag_all else float("nan"), "lags": lag_all,
            "late_steps": late_steps if flagged else None, "late_eps": late_eps, "flagged": flagged, "step0": step0,
            "fwd": mean(fwd), "n_fwd": len(fwd), "yaw": mean(yaw), "n_yaw": len(yaw)}


# ---------------------------------------------------------------- 验收 1：标定门
def gate(L):
    """(PASS|FAIL|INCOMPLETE, 一行原因, k1 统计, k3 统计)。"""
    parts, ok_data = [], True
    st = {}
    for k in (1, 3):
        e, text, complete = eval_state(cal_dir(L, k), CAL_EPISODES)
        st[k] = timing_stats([cal_dir(L, k)])
        if not complete or st[k] is None:
            ok_data = False
            parts.append(f"K={k} 标定臂{text}")
    if not ok_data:
        return "INCOMPLETE", f"L={L:.2f}：" + "；".join(parts), st[1], st[3]
    k1, k3 = st[1], st[3]
    bad = []
    for k, t in ((1, k1), (3, k3)):
        if t["late_steps"] is None:
            bad.append(f"K={k} 的行里没有 late_dispatch（没带 --action-latency？）")
        elif t["late_steps"] > 0:
            shown = "；".join(f"seed {s} 步 {i}" for _, s, i in t["late_eps"][:3])
            more = f"，共 {len(t['late_eps'])} 集有迟到步" if len(t["late_eps"]) > 3 else ""
            bad.append(f"K={k} 迟到步 {t['late_steps']}/{t['steps']}（{shown}{more}）")
    diffs = {"生效时长": rel(k3["eff_med"], k1["eff_med"]), "稳态前飞位移": rel(k3["fwd"], k1["fwd"]), "转角": rel(k3["yaw"], k1["yaw"])}
    for name, d in diffs.items():
        if d != d or abs(d) > GATE_REL:
            bad.append(f"{name}相对差 {pct(d)} 超过 {GATE_REL:.0%}")
    head = f"L={L:.2f}：K=3 对 K=1 生效时长 {pct(diffs['生效时长'])}，前飞位移 {pct(diffs['稳态前飞位移'])}，转角 {pct(diffs['转角'])}；" \
           f"迟到步 K=1 {k1['late_steps']}/{k1['steps']}、K=3 {k3['late_steps']}/{k3['steps']}"
    return ("FAIL" if bad else "PASS"), head + ("；未过：" + "；".join(bad) if bad else ""), k1, k3


def timing_row(label, t):
    if t is None:
        return f"| {label} | 未完成 | | | | | | | | | |"
    late = "-" if t["late_steps"] is None else f"{t['late_steps']}/{t['steps']}（{t['late_steps'] / t['steps']:.1%}）"
    s0 = [x for x in t["step0"] if x[2] is not None]
    s0_late = "-" if not t["flagged"] else f"{sum(1 for x in s0 if x[3])}/{len(s0)}"
    return (f"| {label} | {t['n_eps']} | {t['steps']} | {fnum(t['eff_med'])} | {fnum(t['eff_mean'])} | "
            f"{fnum(t['eff_min'], 3)}–{fnum(t['eff_max'], 3)} | {fnum(t['fwd'])}（n={t['n_fwd']}） | {fnum(t['yaw'], 2)}（n={t['n_yaw']}） | "
            f"{fnum(t['lag_med'], 3)} / {fnum(t['lag_max'], 3)} | {late} | {s0_late} |")


TIMING_HEAD = ["| 臂 | 集数 | 步数 | 生效时长 集间中位 s | 逐步均值 s | 逐步最小–最大 s | 稳态前飞位移 m/步 | 转角 °/步 | dispatch_lag_s 中位 / 最大 s | 迟到步 | 第 0 步迟到的集 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]


def calib_section():
    lines = ["## 1. 标定（验收 1）：教师 K=1 与 K=3 各 10 个 train 分区 seed", "",
             f"通过条件：K=3 对 K=1 的生效时长（集间中位）、稳态前飞位移、转角三项相对差都 ≤ {GATE_REL:.0%}，且迟到步为 0（含每集第 0 步）。"
             "未过时的预注册顺序：先查预热，再试 L=0.20，再考虑 `--rtf`；最后用哪个 L 由主会话定。", ""]
    verdicts = {}
    for L in LADDER:
        v, reason, k1, k3 = gate(L)
        verdicts[L] = v
        lines += [f"### L = {L:.2f} s：{v}", "", reason, ""]
        if k1 is None and k3 is None:
            lines += ["（没有这个 L 的标定数据）", ""]
            continue
        lines += TIMING_HEAD + [timing_row(f"K=1 教师 L={L:.2f}", k1), timing_row(f"K=3 教师 L={L:.2f}", k3)]
        if k1 and k3:
            lines.append(f"| K=3 对 K=1 | | | {pct(rel(k3['eff_med'], k1['eff_med']))} | {pct(rel(k3['eff_mean'], k1['eff_mean']))} | | "
                         f"{pct(rel(k3['fwd'], k1['fwd']))} | {pct(rel(k3['yaw'], k1['yaw']))} | | | |")
        lines.append("")
        for k, t in ((1, k1), (3, k3)):
            if not t:
                continue
            s0 = "；".join(f"seed {s} {fnum(lag, 3)} s{'（迟到）' if late else ''}" for _, s, lag, late in t["step0"])
            lines.append(f"- K={k} 每集第 0 步 dispatch_lag_s：{s0}")
            if t["late_eps"]:
                lines.append(f"- K={k} 迟到步明细：" + "；".join(f"seed {s} 步 {i}" for _, s, i in t["late_eps"]))
        lines.append("")
    ref1, ref3 = timing_stats(REF_K1), timing_stats(REF_K3)
    lines += ["### 旧口径参照（没有 --action-latency，生效时长只有每集均值）", "",
              "| 臂 | 集数 | 步数 | 生效时长 集间中位 s | 稳态前飞位移 m/步 | 转角 °/步 |", "|---|---|---|---|---|---|"]
    for label, t in (("K=1 11-10 下午 guard2-calswa-s0..2", ref1), ("K=3 11-10 晚长课程第二轮 guard2-calswa-s0..2", ref3)):
        lines.append("| " + label + " | " + (" | ".join(["未找到"] + [""] * 4) if t is None else
                     f"{t['n_eps']} | {t['steps']} | {fnum(t['eff_med'])} | {fnum(t['fwd'])} | {fnum(t['yaw'], 2)}") + " |")
    lines.append("")
    return lines, verdicts


# ---------------------------------------------------------------- 验收 2：学生 K=1 新口径对 11-10 两次评测
def success_by_seed(edir):
    """{seed: 成功?}，只含冻结 seed；含迟到集：迟到集不在 per_episode，按没成功记。没有完整 eval.json 返回 None。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
    e = read_json(path)
    if e.get("partial"):
        return None
    out = {int(x["seed"]): bool(x.get("success")) for x in e.get("per_episode", [])}
    for f in e.get("failed", []):
        out[int(f["seed"])] = False
    return out


def agreement(a, b):
    seeds = [s for s in FROZEN_SEEDS if s in a and s in b]
    same = sum(1 for s in seeds if a[s] == b[s])
    bb = sum(1 for s in seeds if a[s] and not b[s])
    cc = sum(1 for s in seeds if (not a[s]) and b[s])
    return same, len(seeds), bb, cc


def item2_section(L=0.16):
    new_dir = acc_dir(L, 1, 1)
    lines = ["## 2. 验收 2：学生 calswa 种子 1 + 护栏 2、柱阵 K=1、20 个冻结 seed，新口径（L=0.16）对 11-10 两次同配置评测", ""]
    new = success_by_seed(new_dir)
    aft, eve = success_by_seed(REF_K1_AFTERNOON_S1), success_by_seed(REF_K1_EVENING_S1)
    if new is None:
        e, text, _ = eval_state(new_dir, 20)
        return lines + [f"新口径这一组{text}。", ""]
    n_new = sum(new.values())
    lines += [f"- 新口径（L={L:.2f}）走廊内成功 {lc.fmt_frac(n_new, len(new))}；下午 "
              f"{'-' if aft is None else lc.fmt_frac(sum(aft.values()), len(aft))}；晚上对照 "
              f"{'-' if eve is None else lc.fmt_frac(sum(eve.values()), len(eve))}。", ""]
    e = read_json(os.path.join(new_dir, "eval.json"))
    if e.get("late_dispatch_episodes"):
        lines += [f"- 新口径有迟到集（不计成功，按基础设施失败）：{json.dumps(e['late_dispatch_episodes'], ensure_ascii=False)}", ""]
    lines += ["| 配对 | 逐集一致 | 一致率 | 只有前者成功 | 只有后者成功 | 精确 McNemar p（只作参考，20 对） |", "|---|---|---|---|---|---|"]
    rates = {}
    for label, a, b in (("新口径 对 下午", new, aft), ("新口径 对 晚上对照", new, eve), ("下午 对 晚上对照（已知 16/20）", aft, eve)):
        if a is None or b is None:
            lines.append(f"| {label} | 参照缺失 | | | | |")
            continue
        same, n, bb, cc = agreement(a, b)
        if not n:
            lines.append(f"| {label} | 没有共同的冻结 seed | | | | |")
            continue
        rates[label] = same / n
        lines.append(f"| {label} | {same}/{n} | {same / n:.2f} | {bb} | {cc} | {mcnemar_exact(bb, cc):.3f} |")
    ok = [rates.get("新口径 对 下午"), rates.get("新口径 对 晚上对照")]
    if all(r is not None and r == r for r in ok):
        holds = all(r >= AGREE_MIN for r in ok)
        lines += ["", f"预测（差别在同权重同 seed 重评的波动之内，逐集一致率 ≥ {AGREE_MIN:.0%}）："
                  f"**{'成立' if holds else '未成立'}**（新对下午 {ok[0]:.2f}，新对晚上 {ok[1]:.2f}）。"]
    lines.append("")
    return lines


# ---------------------------------------------------------------- 验收 3：森林延迟分布
def percentile(xs, q):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def pillar_count(v):
    """eval.json per_episode[].pillars → 柱子个数。真实数据是整数；演练夹具与更早的写法是柱子列表；缺省 0。"""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return len(v or [])


def forest_section():
    lines = ["## 3. 验收 3：森林 forest_e1 三档的延迟分布（教师，不带 --action-latency）", "",
             "延迟 = 观测戳到派发的仿真秒（`dispatch_lag_s`）= 帧龄（约 0.114 s，与场景无关）+ 策略段墙钟 × RTF。第 0 步缓存是冷的（这条路径没有预热），单列；"
             "稳态 = 第 1 步起。对每个 L：迟到阈值 = L + 0.02 s；需要的 `--rtf`（世界目标实时因子，只能压低）= RTF0 × (L + 0.02 − 0.114) / (最大延迟 − 0.114)，"
             "RTF0 = 该档实测的 action_rtf 均值；最大延迟不超过阈值就不需要 `--rtf`。最大值受集数限制（每档 3 集），只当下限用。", "",
             "| 密度 | 集数 | 步数 | 柱子数 | 稳态延迟 中位 / p90 / p99 / 最大 s | 第 0 步延迟 s | RTF0 | L=0.16：稳态迟到占比 → 需要 --rtf | L=0.20：稳态迟到占比 → 需要 --rtf |",
             "|---|---|---|---|---|---|---|---|---|"]
    for dens in ("sparse", "medium", "dense"):
        edir = os.path.join(OUT, f"forest-{dens}")
        recs = [r for r in episode_records(edir) if r[1]]
        if not recs:
            lines.append(f"| {dens} | 未完成 | | | | | | | |")
            continue
        steady, first, rtfs, pillars = [], [], [], []
        for seed, actions, oracle, man in recs:
            lag = [a.get("dispatch_lag_s") for a in actions]
            if any(x is None for x in lag):
                continue
            first.append(lag[0])
            steady += lag[1:]
            rtf = (man.get("performance") or {}).get("action_rtf")
            if rtf:
                rtfs.append(rtf)
        scene = os.path.join(edir, "eval.json")
        if os.path.exists(scene):
            pillars = [pillar_count(x.get("pillars")) for x in read_json(scene).get("per_episode", [])]
        rtf0 = mean(rtfs)
        cells = []
        for L in LADDER:
            thr = L + LATE_TOL_S
            frac = sum(1 for x in steady if x > thr) / len(steady) if steady else float("nan")
            mx = max(steady) if steady else float("nan")
            if mx != mx or mx <= thr:
                cells.append(f"{frac:.1%} → 不需要")
            else:
                need = rtf0 * (thr - FRAME_AGE_S) / (mx - FRAME_AGE_S)
                cells.append(f"{frac:.1%} → {need:.2f}")
        lines.append(f"| {dens} | {len(recs)} | {len(steady) + len(first)} | {mean(pillars):.0f} | "
                     f"{fnum(median(steady), 3)} / {fnum(percentile(steady, 90), 3)} / {fnum(percentile(steady, 99), 3)} / {fnum(max(steady) if steady else float('nan'), 3)} | "
                     f"{', '.join(fnum(x, 3) for x in first)} | {fnum(rtf0, 2)} | {cells[0]} | {cells[1]} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------- 验收 4：K=1 对 K=3 第 1 块
def item4_section():
    lines = ["## 4. 验收 4：柱阵 K=1 成功率对长课程 K=3 第 1 块通过率（学生 calswa 三个训练种子 + 护栏 2，20 个冻结 seed）", ""]
    Ls = [L for L in LADDER if any(os.path.isdir(acc_dir(L, 3, s)) for s in SEEDS3)]
    if not Ls:
        return lines + ["验收阶段没有进入（标定门未过，或还没轮到）。原因见 DONE。", ""]
    for L in Ls:
        k1 = {s: success_by_seed(acc_dir(L, 1, s)) for s in SEEDS3}
        k3 = {}
        for s in SEEDS3:
            arm = lc.load_arm(acc_dir(L, 3, s))
            if arm is None or not arm["complete"]:
                k3[s] = None
                continue
            late = {int(x["seed"]) for x in read_json(os.path.join(acc_dir(L, 3, s), "eval.json")).get("late_dispatch_episodes") or []}
            # 迟到集是基础设施失败，不计成功：即使轨迹过了第 1 块也按没通过算（K=1 的 success_by_seed 同口径）
            k3[s] = {sd: (lc.passed_k(r, 0) and sd not in late) for sd, r in arm["eps"].items()}
        lines += [f"### L = {L:.2f} s", ""]
        miss = [f"K=1 s{s}" for s in SEEDS3 if k1[s] is None] + [f"K=3 s{s}" for s in SEEDS3 if k3[s] is None]
        if miss:
            lines += ["未完成：" + "、".join(miss) + "。已完成的配对如下。", ""]
        n = n1 = n3 = b = c = 0
        per = []
        for s in SEEDS3:
            if k1[s] is None or k3[s] is None:
                continue
            seeds = [x for x in FROZEN_SEEDS if x in k1[s] and x in k3[s]]
            a1, a3 = sum(k1[s][x] for x in seeds), sum(k3[s][x] for x in seeds)
            bs = sum(1 for x in seeds if k1[s][x] and not k3[s][x])
            cs = sum(1 for x in seeds if (not k1[s][x]) and k3[s][x])
            n, n1, n3, b, c = n + len(seeds), n1 + a1, n3 + a3, b + bs, c + cs
            per.append(f"s{s}：K=1 {a1}/{len(seeds)}，K=3 第 1 块 {a3}/{len(seeds)}，只 K=1 过 {bs}、只 K=3 过 {cs}")
        if n:
            gap = (n1 - n3) / n
            lines += [f"- K=1 成功 {lc.fmt_frac(n1, n)}；K=3 第 1 块通过 {lc.fmt_frac(n3, n)}；差（K=1 − K=3）= **{gap:+.3f}**；"
                      f"配对 b（只 K=1 过）={b}，c（只 K=3 过）={c}，精确 McNemar p = {mcnemar_exact(b, c):.3f}（60 对不独立，p 只作参考）", ""]
            lines += [f"  - {p}" for p in per] + [""]
            if abs(gap) <= GAP_SMALL:
                verdict = f"两者相差 |{gap:.3f}| ≤ {GAP_SMALL}：11-10 晚上那次下降（旧协议 39/60 对 50/60）**来自第一步的时序**。"
            elif gap >= GAP_LARGE:
                verdict = f"K=3 第 1 块比 K=1 低 {gap:.3f} ≥ {GAP_LARGE}：**来自后面几块的柱子进了画面**（学生没见过远处的柱子）。"
            else:
                verdict = (f"差（K=1 − K=3）= {gap:+.3f}，不在「相差不超过 {GAP_SMALL}」也不在「K=3 低 {GAP_LARGE} 以上」："
                           "预注册的两条线都没覆盖，如实记录，不归因。")
            if len(per) < len(SEEDS3):
                verdict += "（三个训练种子没有全部完成，只作进度）"
            lines += ["预注册判定：" + verdict, ""]
        late = [f"K={k} s{s}" for k, dd in ((1, {s: acc_dir(L, 1, s) for s in SEEDS3}), (3, {s: acc_dir(L, 3, s) for s in SEEDS3}))
                for s, p in dd.items() if os.path.exists(os.path.join(p, "eval.json"))
                and read_json(os.path.join(p, "eval.json")).get("late_dispatch_episodes")]
        lines += ["有迟到集的臂（迟到集不计成功，上面的配对已按没成功算）：" + ("、".join(late) if late else "无"), ""]
    ref_k1 = [lc.load_ref(p) for p in REF_K1]
    ref_k3 = []
    for p in REF_K3:
        arm = lc.load_arm(p)
        ref_k3.append(None if arm is None else {sd: lc.passed_k(r, 0) for sd, r in arm["eps"].items()})
    if all(ref_k1) and all(ref_k3):
        n = sum(len(x) for x in ref_k1)
        lines += [f"旧协议参照（11-10）：K=1 {lc.fmt_frac(sum(sum(x.values()) for x in ref_k1), n)}，"
                  f"K=3 第 1 块 {lc.fmt_frac(sum(sum(x.values()) for x in ref_k3), sum(len(x) for x in ref_k3))}。", ""]
    return lines


# ---------------------------------------------------------------- 总览
def overview_section():
    lines = ["## 5. 全部臂的派发延迟与迟到步", "", *TIMING_HEAD]
    names = []
    for L in LADDER:
        names += [f"cal-L{L:.2f}-k1", f"cal-L{L:.2f}-k3"]
        names += [f"acc-L{L:.2f}-k{k}-s{s}" for k in (1, 3) for s in SEEDS3]
    names += [f"forest-{d}" for d in ("sparse", "medium", "dense")]
    for name in names:
        edir = os.path.join(OUT, name)
        if os.path.isdir(os.path.join(edir, "episodes")):
            lines.append(timing_row(name, timing_stats([edir])))
    lines.append("")
    return lines


def main(out=None, note=None):
    global OUT
    if out:
        OUT = os.path.abspath(out)
    lines = ["# exp-action-latency-20261110 汇总（任务卡 W1-0）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += ["口径：metric_version `sim09.v2.coast1s.corridor3m`；带 `--action-latency` 的数据与旧数据不可混报；含迟到步的集算基础设施失败，不计入成功。", ""]
    done = os.path.join(OUT, "DONE")
    if os.path.exists(done):
        with open(done, encoding="utf-8") as f:
            lines += [f"DONE：`{f.read().strip()}`", ""]
    cal, verdicts = calib_section()
    lines += cal + item2_section() + forest_section() + item4_section() + overview_section()
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


def timing_only(specs):
    print("\n".join(TIMING_HEAD))
    for spec in specs:
        label, _, dirs = spec.rpartition("=") if "=" in spec else ("", "", spec)
        dl = dirs.split(",")
        print(timing_row(label or os.path.basename(dl[0]), timing_stats(dl)))
    return 0


def cli(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="输出目录（默认本脚本所在目录）；--timing 时是 [标签=]目录[,目录...] 列表")
    ap.add_argument("--gate", type=float, metavar="L", help="只判标定门；退出码 0 PASS / 1 FAIL / 2 INCOMPLETE")
    ap.add_argument("--timing", action="store_true", help="只打印时序表")
    ap.add_argument("--note", help="写在 summary.md 标题下的一行说明（事后重跑时写明原因与提交哈希）")
    args = ap.parse_args(argv)
    if args.timing:
        return timing_only(args.paths)
    if args.paths:
        OUT = os.path.abspath(args.paths[0])
    if args.gate is not None:
        v, reason, _, _ = gate(args.gate)
        print(f"{v} {reason}")
        return {"PASS": 0, "FAIL": 1}.get(v, 2)
    return main(note=args.note)


if __name__ == "__main__":
    sys.exit(cli())
