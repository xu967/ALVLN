#!/usr/bin/env python3
"""exp-long-course-20261110 / -v2-20261110 汇总：K 个 base 柱阵块首尾相接的长课程，逐块存活率，写 summary.md（随时可跑，缺的臂标「未完成」）。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py [输出目录，默认本脚本所在目录] [--title 文字] [--present-only]
      /root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py --timing [标签=]目录[,目录...] ...   只打印时序健康表（标定用）
口径：metric_version sim09.v2.coast1s.corridor3m；分母 = 请求集数（20 个冻结 seed，采集失败的集算没通过）。
--present-only：分母改成目录里实际有轨迹的集数（被中断的那一集没有轨迹不计）；没有 eval.json 的臂按 episodes/seed_N 读。
  用在跑到一半被停掉的实验（第一轮 20:46 被中止）和跑的过程中随时看进度。--title 追加到一级标题后面。
第二轮新增（第 8–10 节）：时序健康（动作生效时长 / 稳态前飞位移 / 转角 对 K=1 参照）、护栏臂「数步数」航向与真实航向的差、块末 |y| 分布。

逐块存活（每集读 actions.jsonl + oracle.jsonl，第 i 行互相对应）：
  第 k 块通过 = 首次 pose_gt.x >= block_goal_x[k]（该块最远柱 x + 1.0），且到这一步（含）为止没有接触碰撞、|y| 全程 <= 走廊半宽。
  S_k = 通过前 k 块的集数 / 请求集数（累计，S_1 >= S_2 >= ... >= S_K）；条件存活 S_k / S_{k-1}。
  失败归属：第一个坏事件（接触碰撞 = 撞柱，|y| > 半宽 = 越界）发生时正在穿越的那一块；没有坏事件但集结束时还没过线 = 未到线。
  S_K 与 eval.json 的 success（整条课程过线 + 无碰撞 + 不越界）逐集应当一致，汇总末尾核对。
K=1 参照：同权重、同 seed 的今天的 K=1 评测（exp-swa-calib / exp-heading-guard），用来算第 1 块与它的逐集一致率与独立块模型预测。
"""
import argparse
import json
import math
import os
import re
import sys
from math import comb

# ---------------------------------------------------------------- 配置
D = os.environ.get("UAVRGB_DATA", "/root/rivermind-data/uav_rgb")
OUT = os.path.dirname(os.path.abspath(__file__))        # 默认输出目录 = 本脚本所在目录（run.sh 把它拷进 $OUT）；命令行第 1 个参数可覆盖
SWA = os.path.join(D, "exp-swa-calib-20261110")        # K=1 参照：calswa-s*（无护栏）
GUARD = os.path.join(D, "exp-heading-guard-20261110")  # K=1 参照：guard2-calswa-s*、guard2-s*（护栏 2）
BLOCK_PITCH_M = 5.0        # 同 scene_course.BLOCK_PITCH_M；采集失败集没有 block_goal_x，按 6.0 + 5.0·k 补（base 柱阵最远柱恒在 x=5.0，自检里断言过）
FIRST_GOAL_X = 6.0
DEFAULT_HALF_WIDTH_M = 3.0
TAIL_STEPS, TAIL_LOW_Z_M = 10, 1.5   # 末 10 步最低高度 < 1.5 m 视为掉高（电池压降的症状），只作诊断
P1_WINDOW = 0.15           # 预注册：S_K 落在独立块预测 p1^K 的 ±0.15 以内算「符合」
# 时序健康（第二轮新增）。--window-from observation 下动作窗口从观测帧起算：观测到派发的延迟越长，动作实际生效的时间越短，
# 每步前飞位移和转角随之缩水，学生在 K=1 学到的「一步多远、转多少」就不再成立。第一轮 K=3 世界生效时长 0.77–0.79 s 就是这样混杂进来的。
REF_EFF_S = 0.871          # K=1 参照：下午护栏臂 guard2-calswa-s0..2 共 60 集，每集 action_sim_s / steps 的中位数（现场复算 0.871）
REF_YAW_DEG = 26.1         # 同一批 K=1 护栏臂：YAW 步 |dyaw_deg| 的池化均值（现场复算 26.12）
TIMING_FLAG = 0.03         # 生效时长或转角对参照偏离超过 3% 的臂标 ⚠
COUNT_ANGLE_DEG = 26.1     # 护栏「数步数」估计航向时，每个转向步按多少度算（= K=1 转角）
DRIFT_SEG_STEPS = 20       # 航向漂移按 20 步一段统计
LONG_STEPS = 40            # 时序表里「长集」的步数门槛：已经过了第 1 块的集（学生 K=1 过线中位约 31 步）。短集（早期碰撞）的每步 A* 更贵，把均值拉低
# 变体显示名 -> (本实验的臂目录列表, K=1 参照目录列表)；列表第 i 项 = 训练种子 i，逐集配对的键 = (i, 评测 seed)
MAIN, NOGUARD, BEST = "calswa + 护栏2（主臂）", "calswa（无护栏）", "ckpt_best + 护栏2（次要）"
PAIRS = [(MAIN, NOGUARD), (MAIN, BEST)]    # 配对 McNemar：A 对 B


def variants(out):
    return {
        "教师": ([os.path.join(out, "teacher")], None),
        MAIN: ([os.path.join(out, f"guard2-calswa-s{s}") for s in (0, 1, 2)],
               [os.path.join(GUARD, f"guard2-calswa-s{s}") for s in (0, 1, 2)]),
        NOGUARD: ([os.path.join(out, f"calswa-s{s}") for s in (0, 1, 2)],
                  [os.path.join(SWA, f"calswa-s{s}") for s in (0, 1, 2)]),
        BEST: ([os.path.join(out, f"guard2-s{s}") for s in (0, 1, 2)],
               [os.path.join(GUARD, f"guard2-s{s}") for s in (0, 1, 2)]),
    }
# ----------------------------------------------------------------

FAIL_NAMES = {"collision": "撞柱", "corridor": "越界", "not_reached": "未到线", "aborted": "采集中止"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))   # 夹到 [0, 1]：0/n 时 c - h 是 -1e-17，会打印成 -0.00


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# 逐块判据 block_outcomes 在 sim/tools/eval_closed_loop.py：W1-4 起 eval.json 的 blocks_passed 与本脚本共用这一份。
# 脚本被拷进实验目录时用目录里冻结的 code/sim/tools（W1-4 之前冻结的快照没有这个函数，会 ImportError），在仓库里跑时用仓库的。
for _tools in (os.path.join(OUT, "code", "sim", "tools"),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim", "tools")):
    if os.path.isfile(os.path.join(_tools, "eval_closed_loop.py")):
        sys.path.insert(0, _tools)
        break
from eval_closed_loop import block_outcomes  # noqa: E402,F401  - 下面的 load_arm / 汇总用；test_course_summary 也经 sm.block_outcomes 取


def episode_stats(actions, oracle):
    """步数与仿真秒：flight_s = 首个动作起点到末个动作终点；sim_s = 末个动作终点（含起飞前的 ~13.5 s，电池按它算）。"""
    t0 = oracle[0].get("action_start_ns") if oracle else None
    t1 = oracle[-1].get("action_end_ns") if oracle else None
    zs = [float(o["pose_gt"]["z"]) for o in oracle[-TAIL_STEPS:] if isinstance(o.get("pose_gt"), dict) and "z" in o["pose_gt"]]
    return {"steps": len(actions),
            "sim_s": None if t1 is None else t1 / 1e9,
            "flight_s": None if t0 is None or t1 is None else (t1 - t0) / 1e9,
            "z_tail_min": min(zs) if zs else None,
            "guard_steps": sum(1 for a in actions if a.get("executed_by") == "heading_guard")}


def scan_arm(edir):
    """没有 eval.json 的臂（还在跑，或进程被硬杀）：按 episodes/seed_N 读。K 与每块过线 x 由 manifest 的 goal_line_x 反推（K = (x − 6)/5 + 1）。

    每集 success = 最后一块通过；manifest 的 error 非空是采集中止（多为固件 FAILSAFE），计入没通过。没有任何带轨迹的集返回 None。
    """
    root = os.path.join(edir, "episodes")
    if not os.path.isdir(root):
        return None
    eps, k_blocks = {}, None
    for name in sorted(os.listdir(root)):
        m = re.fullmatch(r"seed_(\d+)", name)
        if not m:
            continue
        ep = os.path.join(root, name)
        actions, oracle = read_jsonl(os.path.join(ep, "actions.jsonl")), read_jsonl(os.path.join(ep, "oracle.jsonl"))
        man_path = os.path.join(ep, "episode_manifest.json")
        man = read_json(man_path) if os.path.exists(man_path) else {}
        goal = man.get("goal_line_x")
        if goal is None or not oracle:
            continue
        k = int(round((float(goal) - FIRST_GOAL_X) / BLOCK_PITCH_M)) + 1
        goals = [float(goal) - BLOCK_PITCH_M * (k - 1 - j) for j in range(k)]
        err = man.get("error")
        row = {**episode_stats(actions, oracle), **block_outcomes(actions, oracle, goals, DEFAULT_HALF_WIDTH_M, aborted=bool(err))}
        row["success"] = bool(row["pass"][-1])
        if err:
            row.update(aborted=True, stderr=str(err))
        eps[int(m.group(1))] = row
        k_blocks = k
    if not eps:
        return None
    return {"complete": False, "requested": len(eps), "failed": sum(1 for r in eps.values() if r.get("aborted")),
            "skipped": 0, "K": k_blocks, "eps": eps, "partial": True, "interrupted": False}


def load_arm(edir, present_only=False):
    """-> None（没有 eval.json）或 {"complete", "requested", "failed", "K", "eps": {seed: 行}}。

    present_only=True：分母 = 目录里实际有轨迹的集数；没有轨迹的 failed 项（被 SIGTERM 中断的那一集，没有 episodes/seed_N）不计；
    没有 eval.json 的臂按目录读（scan_arm）。complete 仍按 eval.json 判。
    """
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return scan_arm(edir) if present_only else None
    e = read_json(path)
    half = e.get("corridor_half_width")
    half = DEFAULT_HALF_WIDTH_M if half is None else float(half)
    k_blocks = e.get("blocks")
    eps = {}
    for x in e.get("per_episode", []):
        ep = os.path.join(edir, x.get("episode_dir") or os.path.join("episodes", f"seed_{x['seed']}"))
        actions, oracle = read_jsonl(os.path.join(ep, "actions.jsonl")), read_jsonl(os.path.join(ep, "oracle.jsonl"))
        goals = x.get("block_goal_x")
        if not goals and k_blocks and x.get("goal_x") is not None:   # 旧 eval.json：由整条课程的成功线按块距反推
            goals = [float(x["goal_x"]) - BLOCK_PITCH_M * (k_blocks - 1 - k) for k in range(k_blocks)]
        row = {"success": bool(x.get("success")), **episode_stats(actions, oracle)}
        if goals:
            row.update(block_outcomes(actions, oracle, goals, half))
        eps[int(x["seed"])] = row
    # 采集失败的集不在 per_episode 里，但每集的 actions / oracle 照样留在 episodes/seed_N：按中止前的轨迹补算逐块结果
    for f in e.get("failed", []):
        ep = os.path.join(edir, "episodes", f"seed_{f['seed']}")
        actions, oracle = read_jsonl(os.path.join(ep, "actions.jsonl")), read_jsonl(os.path.join(ep, "oracle.jsonl"))
        if present_only and not oracle:
            continue
        row = {"success": False, "aborted": True, **episode_stats(actions, oracle),
               "stderr": " ".join(f.get("stderr_tail") or [f.get("error", "")])}
        if k_blocks and oracle:
            row.update(block_outcomes(actions, oracle, [FIRST_GOAL_X + BLOCK_PITCH_M * k for k in range(k_blocks)], half, aborted=True))
        eps[int(f["seed"])] = row
    requested_eval = e.get("episodes_requested") or len(eps)
    # 完整 = 没中断，且请求的集都跑过（done + failed）；电池掉压造成的采集中止是测量结果，不重跑
    complete = bool(not e.get("partial") and (e.get("episodes_done") or 0) + (e.get("failed_count") or 0) == requested_eval)
    return {"complete": complete, "requested": len(eps) if present_only else requested_eval,
            "failed": sum(1 for r in eps.values() if r.get("aborted")) if present_only else (e.get("failed_count") or 0),
            "skipped": e.get("skipped_count") or 0, "K": k_blocks, "eps": eps, "partial": bool(e.get("partial")),
            "interrupted": bool(e.get("interrupted"))}   # SIGTERM 中断时 eval.json 写 interrupted = {seed, reason}


def load_ref(edir):
    """K=1 参照：{seed: success}；没有完整 eval.json 返回 None。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
    e = read_json(path)
    if e.get("partial"):
        return None
    return {int(x["seed"]): bool(x.get("success")) for x in e.get("per_episode", [])}


def passed_k(r, k):
    """这一集是否通过第 k 块（0 起）；没有逐块结果（无轨迹）算没通过。"""
    return bool(r.get("pass") and r["pass"][k])


def fmt_frac(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n}={k / n:.2f} [{lo:.2f}, {hi:.2f}]" if n else "-"


def pooled(arms):
    """多个训练种子的臂合并成 {(训练种子序号, 评测 seed): 行}；同时返回请求总数与是否全部完整。"""
    rows, requested, ok = {}, 0, True
    for i, arm in enumerate(arms):
        if arm is None:
            ok = False
            continue
        requested += arm["requested"]
        ok = ok and arm["complete"]
        for s, r in arm["eps"].items():
            rows[(i, s)] = r
    return rows, requested, ok


# ---------------------------------------------------------------- 时序健康 / 航向漂移 / 块末 |y|（第二轮新增；纯函数 + 只读落盘文件）

def percentile(xs, q):
    """线性插值分位数（与 numpy 默认一致），q 取 0..100；空列表返回 nan。"""
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    """总体标准差（除以 n）；空列表 nan。"""
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def episode_dirs(dirs):
    """若干臂目录下的全部 episodes/seed_N 目录（按名字排序）。"""
    out = []
    for d in dirs:
        root = os.path.join(d, "episodes")
        if os.path.isdir(root):
            out += [os.path.join(root, n) for n in sorted(os.listdir(root)) if re.fullmatch(r"seed_\d+", n)]
    return out


def episode_timing(ep):
    """一集的时序量（读 episode_manifest.performance 与 actions.jsonl）；没有 performance 的半截集返回 None。

    eff = action_sim_s / steps：每步动作实际生效的仿真秒（派发位姿到窗口末位姿，已经扣掉观测到派发的延迟）；
    fwd = 稳态前飞的每步位移（本步与上一步都执行 FORWARD_SLOW，本步无碰撞）；yaw = 转向步（YAW_LEFT / YAW_RIGHT）的 |dyaw_deg|。
    """
    man = os.path.join(ep, "episode_manifest.json")
    if not os.path.exists(man):
        return None
    perf = read_json(man).get("performance") or {}
    steps, sim_s = perf.get("steps"), perf.get("action_sim_s")
    if not steps or sim_s is None:
        return None
    actions = read_jsonl(os.path.join(ep, "actions.jsonl"))
    fwd = [a["displacement_m"] for prev, a in zip(actions, actions[1:])
           if a.get("action_executed") == 1 and prev.get("action_executed") == 1 and not a.get("collision")
           and a.get("displacement_m") is not None]
    yaw = [abs(a["dyaw_deg"]) for a in actions if a.get("action_executed") in (2, 3) and a.get("dyaw_deg") is not None]
    return {"eff": sim_s / steps, "steps": steps, "fwd": fwd, "yaw": yaw}


def arm_timing(dirs):
    """若干臂目录合并（同一变体的几个训练种子）：集数、eff 的集间中位数与最小值、前飞位移与转角的池化均值；没有可用的集返回 None。"""
    effs, long_effs, fwd, yaw = [], [], [], []
    for ep in episode_dirs(dirs):
        t = episode_timing(ep)
        if t:
            effs.append(t["eff"])
            if t["steps"] >= LONG_STEPS:
                long_effs.append(t["eff"])
            fwd += t["fwd"]
            yaw += t["yaw"]
    if not effs:
        return None
    return {"n": len(effs), "eff_med": percentile(effs, 50), "eff_min": min(effs),
            "n_long": len(long_effs), "eff_long_med": percentile(long_effs, 50) if long_effs else None,
            "fwd": mean(fwd), "n_fwd": len(fwd), "yaw": mean(yaw), "n_yaw": len(yaw)}


def pct(x, ref):
    """相对参照的偏离（带符号的百分数文字）；缺值 '-'。"""
    if x is None or ref is None or ref != ref or x != x:
        return "-"
    return f"{100.0 * (x / ref - 1.0):+.1f}%"


def timing_flags(t, tol=TIMING_FLAG):
    """对 K=1 参照偏离超过 tol 的项，如 ['生效时长 -9.2%', '转角 -10.1%']；没有偏离返回 []。前飞位移不参与标记（它是生效时长的直接后果）。"""
    out = []
    if abs(t["eff_med"] / REF_EFF_S - 1.0) > tol:
        out.append(f"生效时长 {pct(t['eff_med'], REF_EFF_S)}")
    if t["yaw"] == t["yaw"] and abs(t["yaw"] / REF_YAW_DEG - 1.0) > tol:
        out.append(f"转角 {pct(t['yaw'], REF_YAW_DEG)}")
    return out


def timing_lines(rows, ref_fwd=None, tol=TIMING_FLAG):
    """时序健康表。rows = [(标签, arm_timing 的结果或 None, 是否参照行)]；返回 (表格行, 被标记的标签列表)。"""
    lines = [f"| 臂 | 集数 | 生效时长中位 s/步 | 集间最小 s/步 | 对 0.871 | ≥{LONG_STEPS} 步的集数 | 这些集的生效时长中位（对 0.871） | "
             "稳态前飞位移 m/步 | 对 K=1 前飞 | 转角 °/步 | 对 26.1° | 标记 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    flagged = []
    for label, t, is_ref in rows:
        if not t:
            lines.append(f"| {label} | 0 | | | | | | | | | | 没有可用的集 |")
            continue
        fl = timing_flags(t, tol)
        if fl and not is_ref:
            flagged.append(label)
        mark = "参照" if is_ref else (("⚠ " + "；".join(fl)) if fl else f"{tol:.0%} 内")
        long_eff = t.get("eff_long_med")
        long_txt = "-" if long_eff is None else f"{long_eff:.3f}（{pct(long_eff, REF_EFF_S)}）"
        lines.append(f"| {label} | {t['n']} | {t['eff_med']:.3f} | {t['eff_min']:.3f} | {pct(t['eff_med'], REF_EFF_S)} | "
                     f"{t.get('n_long', 0)} | {long_txt} | "
                     f"{t['fwd']:.3f} | {pct(t['fwd'], ref_fwd)} | {t['yaw']:.1f} | {pct(t['yaw'], REF_YAW_DEG)} | {mark} |")
    return lines, flagged


def heading_drift(ep_dirs, seg=DRIFT_SEG_STEPS, angle=COUNT_ANGLE_DEG):
    """-> {段号: [Δ（度）...]}，Δ = 真实航向 − 数步数估计的航向，逐步。

    真实航向 = 这一步窗口末的 yaw（actions 的 yaw_before + dyaw_deg）相对第 0 步 yaw_before，折到 (−180, 180]；
    估计航向 = angle × 净转向数，净转向数 = 到这一步（含）为止执行的 YAW_LEFT 数 − YAW_RIGHT 数——航向护栏就靠这个数判断「偏了多少」。
    """
    out = {}
    for ep in ep_dirs:
        actions = read_jsonl(os.path.join(ep, "actions.jsonl"))
        if not actions or actions[0].get("yaw_before") is None:
            continue
        yaw0, n = float(actions[0]["yaw_before"]), 0
        for i, a in enumerate(actions):
            n += (a.get("action_executed") == 2) - (a.get("action_executed") == 3)
            if a.get("yaw_before") is None or a.get("dyaw_deg") is None:
                continue
            true = wrap_deg(float(a["yaw_before"]) + float(a["dyaw_deg"]) - yaw0)
            out.setdefault(i // seg, []).append(wrap_deg(true - angle * n))
    return out


def drift_lines(label, drift, seg=DRIFT_SEG_STEPS):
    """某个变体的航向漂移表行：步段、样本数、Δ 均值（带符号）、Δ 标准差、|Δ| 均值、|Δ| 的 90 分位与最大。"""
    out = []
    for g in sorted(drift):
        d = drift[g]
        ad = [abs(x) for x in d]
        out.append(f"| {label} | {g * seg}–{g * seg + seg - 1} | {len(d)} | {mean(d):+.1f} | {std(d):.1f} | {mean(ad):.1f} | "
                   f"{percentile(ad, 90):.1f} | {max(ad):.1f} |")
    return out


def block_end_y_lines(label, rows, k_blocks):
    """某个变体每块块末（首次过线那一步）|y| 的分布行：只含通过该块的集（通过要求 |y| 全程不超走廊半宽）。"""
    out = []
    for k in range(k_blocks):
        ys = [r["cross_y"][k] for r in rows.values() if r.get("cross_y") and r["cross_y"][k] is not None]
        if ys:
            out.append(f"| {label} | {k + 1} | {len(ys)} | {mean(ys):.2f} | {percentile(ys, 50):.2f} | {percentile(ys, 90):.2f} | {max(ys):.2f} |")
        else:
            out.append(f"| {label} | {k + 1} | 0 | | | | |")
    return out


def state_text(ok, arm_list, n_rows):
    """完整 / 已中断（eval.json 里有 interrupted，即被 SIGTERM 停掉）/ 进行中。"""
    if ok:
        return "完整"
    if any(arm and arm.get("interrupted") for arm in arm_list):
        return f"已中断（有轨迹 {n_rows} 集）"
    return f"进行中（已出 {n_rows} 集）"


def timing_rows(out):
    """时序健康表的行：K=1 参照两行（现场复算）+ 本实验每个有数据的臂目录一行；返回 (rows, K=1 护栏参照的前飞位移)。"""
    v = variants(out)
    ref_g, ref_s = arm_timing(v[MAIN][1]), arm_timing(v[NOGUARD][1])
    rows = [("K=1 参照：护栏 2（下午 guard2-calswa-s0..2）", ref_g, True), ("K=1 参照：calswa 无护栏（exp-swa-calib）", ref_s, True)]
    for _name, (dirs, _refs) in v.items():
        rows += [(os.path.basename(d), t, False) for d in dirs if (t := arm_timing([d]))]
    return rows, (ref_g["fwd"] if ref_g else None)


def main(out=None, title=None, present_only=False):
    out = out or OUT
    arms = {name: [load_arm(d, present_only) for d in dirs] for name, (dirs, _refs) in variants(out).items()}
    data = {name: pooled(a) for name, a in arms.items()}
    ref = {name: None if refs is None else {(i, s): v for i, d in enumerate(refs) if (r := load_ref(d)) for s, v in r.items()}
           for name, (_dirs, refs) in variants(out).items()}
    k_blocks = next((arm["K"] for a in arms.values() for arm in a if arm and arm["K"]), None)
    if not k_blocks:
        print("还没有任何带 blocks 的 eval.json（实验未开始或第一集未结束）")
        return 1
    ks = range(k_blocks)
    t_rows, ref_fwd = timing_rows(out)
    t_table, t_flagged = timing_lines(t_rows, ref_fwd)
    lines = [f"# {os.path.basename(os.path.normpath(out))}：{k_blocks} 块长课程的逐块存活率" + (f"（{title}）" if title else ""), "",
             f"口径 `sim09.v2.coast1s.corridor3m`；成功线 = {6.0 + BLOCK_PITCH_M * (k_blocks - 1):g} m（6.0 + 5.0·(K−1)，K={k_blocks}）；"
             "20 个冻结评测 seed；hold 1.0，observation，cf_sitl lib31（起始 SoC 0.5 不变），E1 MINI。"
             "S_k = 通过前 k 块的集数 / " + ("目录里实际有轨迹的集数（累计；--present-only：被中断的那一集没有轨迹不计，各臂分母不同）" if present_only else "请求集数（累计）")
             + "；Wilson 95%；预注册见 `run.sh`（第一轮）/ `run_v2.sh`（第二轮）文件头。", ""]
    if t_flagged:
        lines += [f"> ⚠ 时序健康：{len(t_flagged)} 个臂的动作生效时长或转角对 K=1 参照偏离超过 {TIMING_FLAG:.0%}（{'、'.join(t_flagged)}），"
                  "这些臂的结果与 K=1 参照不在同一动力学时序下，见第 8 节。", ""]
    lines += ["## 1. 逐块存活率 S_k", "",
             f"| 变体 | {'有轨迹集数' if present_only else '请求集数'} | " + " | ".join(f"S_{k + 1}" for k in ks) + " | 状态 |",
             "| --- | --- | " + " | ".join("---" for _ in ks) + " | --- |"]
    surv = {}   # 变体 -> [S_1..S_K 的通过数]，只收有数据的
    vdirs = {name: dirs for name, (dirs, _refs) in variants(out).items()}
    for name, (rows, requested, ok) in data.items():
        if not rows:
            lines.append(f"| {name} | | " + " | ".join("" for _ in ks) + f" | {'未完成' if any(os.path.isdir(d) for d in vdirs[name]) else '未运行'} |")
            continue
        cnt = [sum(1 for r in rows.values() if passed_k(r, k)) for k in ks]
        surv[name] = (cnt, requested)
        state = state_text(ok, arms[name], len(rows))
        lines.append(f"| {name} | {requested} | " + " | ".join(fmt_frac(c, requested) for c in cnt) + f" | {state} |")
    lines += ["", "条件存活 S_k / S_{k-1}（通过前一块的集里，这一块也通过的比例）：", "",
              "| 变体 | " + " | ".join(f"块 {k + 1}" for k in ks) + " |", "| --- | " + " | ".join("---" for _ in ks) + " |"]
    for name, (cnt, requested) in surv.items():
        prev = [requested] + cnt[:-1]
        lines.append(f"| {name} | " + " | ".join(f"{c}/{p}={c / p:.2f}" if p else "-" for c, p in zip(cnt, prev)) + " |")
    lines += ["", "按臂目录（各训练种子各自的 S_k）：", "",
              "| 臂 | 集数 | " + " | ".join(f"S_{k + 1}" for k in ks) + " | 状态 |", "| --- | --- | " + " | ".join("---" for _ in ks) + " | --- |"]
    for name, dirs in vdirs.items():
        for d, arm in zip(dirs, arms[name]):
            if not arm or not arm["eps"]:
                continue
            n = arm["requested"]
            cnt = [sum(1 for r in arm["eps"].values() if passed_k(r, k)) for k in ks]
            state = state_text(arm["complete"], [arm], len(arm["eps"]))
            lines.append(f"| {os.path.basename(d)} | {n} | " + " | ".join(fmt_frac(c, n) for c in cnt) + f" | {state} |")

    lines += ["", "## 2. 与独立块模型对照（预注册 P1 / P2）", "",
              f"p1 = 同权重同 seed 的今天 K=1 评测成功率；独立块预测 S_k = p1^k；窗口 ±{P1_WINDOW}。", "",
              "| 变体 | K=1 参照 p1 | " + " | ".join(f"S_{k + 1} 实测 / 预测" for k in ks) + " | S_K 对窗口 |", "| --- | --- | " + " | ".join("---" for _ in ks) + " | --- |"]
    for name, (cnt, requested) in surv.items():
        r = ref.get(name)
        if not r:
            continue
        p1 = sum(r.values()) / len(r)
        pred = [p1 ** (k + 1) for k in ks]
        obs = [c / requested for c in cnt]
        verdict = "在窗内" if abs(obs[-1] - pred[-1]) <= P1_WINDOW else ("低于窗" if obs[-1] < pred[-1] else "高于窗")
        lines.append(f"| {name} | {sum(r.values())}/{len(r)}={p1:.2f} | " + " | ".join(f"{o:.2f} / {p:.2f}" for o, p in zip(obs, pred)) + f" | {verdict} |")

    lines += ["", "## 3. 失败类型按块", "",
              "进入 = 通过前一块（第 1 块 = 全部请求集）；撞柱 / 越界 = 第一个坏事件发生时正在穿越这一块；未到线 = 没有坏事件但集结束时还没过这一块的线；"
              "采集中止 = 采集进程中途失败（多为固件 FAILSAFE：电池掉压后推力不够），按中止前的轨迹归块，计入没通过。", "",
              "| 变体 | 块 | 进入 | 通过 | 撞柱 | 越界 | 未到线 | 采集中止 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, requested, _ok) in data.items():
        if not rows:
            continue
        for k in ks:
            entered = requested if k == 0 else sum(1 for r in rows.values() if passed_k(r, k - 1))
            passed = sum(1 for r in rows.values() if passed_k(r, k))
            fails = {t: sum(1 for r in rows.values() if r.get("fail") and r["fail"] == (k, t)) for t in FAIL_NAMES}
            lines.append(f"| {name} | {k + 1} | {entered} | {passed} | {fails['collision']} | {fails['corridor']} | "
                         f"{fails['not_reached']} | {fails['aborted']} |")
        if requested > len(rows):
            lines.append(f"| {name} | 未出结果（进行中或没有轨迹） | | | | | | {requested - len(rows)} 集，计入没通过 |")

    lines += ["", "## 4. 配对检验（精确 McNemar；b = 前者通过后者没通过，c = 前者没通过后者通过）", "",
              "配对键 = (训练种子序号, 评测 seed)；三个训练种子共用 20 个评测几何，60 个配对不独立，p 值只作参考。", "",
              "| A 对 B | 比较量 | 共同配对 | b | c | p |", "| --- | --- | --- | --- | --- | --- |"]
    for a, b_ in PAIRS:
        ra, rb = data[a][0], data[b_][0]
        keys = sorted(set(ra) & set(rb))
        for k, label in ((0, "S_1"), (k_blocks - 1, f"S_{k_blocks}")):
            if not keys:
                lines.append(f"| {a} 对 {b_} | {label} | 0 | | | |")
                continue
            ok_a = {key: passed_k(ra[key], k) for key in keys}
            ok_b = {key: passed_k(rb[key], k) for key in keys}
            b = sum(1 for key in keys if ok_a[key] and not ok_b[key])
            c = sum(1 for key in keys if not ok_a[key] and ok_b[key])
            lines.append(f"| {a} 对 {b_} | {label} | {len(keys)} | {b} | {c} | {mcnemar_exact(b, c):.3f} |")

    lines += ["", "## 5. 每集步数与仿真秒", "",
              "flight_s = 首个动作起点到末个动作终点；sim_s = 末个动作终点（含起飞前约 13.5 s，电池按它消耗）；掉高 = 末 "
              f"{TAIL_STEPS} 步最低高度 < {TAIL_LOW_Z_M:g} m 的集数（电池压降的症状，只作诊断）。", "",
              "| 变体 | 集数 | 步数均值 | 步数最大 | flight_s 均值 | sim_s 最大 | 掉高集数 | 未到线集的 sim_s 中位数 | 护栏改写的步占比 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, _requested, _ok) in data.items():
        if not rows:
            continue
        rs = list(rows.values())
        sim_s = [r["sim_s"] for r in rs if r["sim_s"] is not None]
        fl = [r["flight_s"] for r in rs if r["flight_s"] is not None]
        low = sum(1 for r in rs if r["z_tail_min"] is not None and r["z_tail_min"] < TAIL_LOW_Z_M)
        nr = sorted(r["sim_s"] for r in rs if r.get("fail") and r["fail"][1] == "not_reached" and r["sim_s"] is not None)
        total = max(sum(r["steps"] for r in rs), 1)
        lines.append(f"| {name} | {len(rs)} | {sum(r['steps'] for r in rs) / len(rs):.1f} | {max(r['steps'] for r in rs)} | "
                     f"{(sum(fl) / len(fl)) if fl else float('nan'):.1f} | {max(sim_s) if sim_s else float('nan'):.1f} | {low} | "
                     f"{f'{nr[len(nr) // 2]:.1f}' if nr else '-'} | {sum(r['guard_steps'] for r in rs) / total:.3f} |")

    lines += ["", "## 6. 第 1 块对今天 K=1 的逐集一致率（预注册 P3：>= 70%；同权重同 seed 自己重跑约 76%）", "",
              "第 1 块几何 = 同 seed 的 base 柱阵；两边都通过或都没通过算一致。", "",
              "| 变体 | 配对数 | 一致 | 一致率 | 都通过 | 课程过 / K=1 没过 | 课程没过 / K=1 过 | 都没过 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, _requested, _ok) in data.items():
        r = ref.get(name)
        if not rows or not r:
            continue
        keys = sorted(set(rows) & set(r))
        if not keys:
            continue
        both = sum(1 for k in keys if passed_k(rows[k], 0) and r[k])
        c_only = sum(1 for k in keys if passed_k(rows[k], 0) and not r[k])
        k1_only = sum(1 for k in keys if not passed_k(rows[k], 0) and r[k])
        neither = len(keys) - both - c_only - k1_only
        agree = both + neither
        lines.append(f"| {name} | {len(keys)} | {agree} | {agree / len(keys):.2f} | {both} | {c_only} | {k1_only} | {neither} |")

    mism = [(name, key) for name, (rows, _r, _o) in data.items() for key, r in rows.items()
            if r.get("pass") and r["pass"][-1] != r["success"]]
    lines += ["", "## 7. 核对", "",
              f"- S_K 与 eval.json 的 success 逐集不一致：{len(mism)} 集" + (f"（{mism[:5]}）" if mism else "") + "。",
              f"- 采集失败集（计入没通过）：{sum(arm['failed'] for a in arms.values() for arm in a if arm)} 集；"
              f"其中 stderr 含 FAILSAFE 的 {sum(1 for rows, _r, _o in data.values() for r in rows.values() if r.get('aborted') and 'FAILSAFE' in r.get('stderr', ''))} 集。",
              f"- 被跳过的集（split 不符或重复几何，必须为 0）：{sum(arm['skipped'] for a in arms.values() for arm in a if arm)} 集。"]

    lines += ["", "## 8. 时序健康（动作生效时长 / 前飞位移 / 转角 对 K=1 参照）", "",
              "生效时长 = performance.action_sim_s / steps（派发位姿到窗口末位姿的仿真秒；`--window-from observation` 下动作窗口从观测帧起算，"
              "观测到派发的延迟越长生效越短）。稳态前飞位移 = 本步与上一步都执行 FORWARD_SLOW 且无碰撞的步，位移池化均值。"
              "转角 = 转向步 |dyaw_deg| 的池化均值。集间最小 = 各集生效时长的最小值。"
              f"短集（早期碰撞，< {LONG_STEPS} 步）里学生飞在柱子附近，A* 每步更贵，且首步 A* 缓存是冷的，生效时长偏短；"
              f"「≥{LONG_STEPS} 步的集」单列，用来把这种短集效应与整体时序偏差分开（标记仍按全部集的中位数，口径同任务书）。"
              f"参照 = 下午 K=1 护栏臂（生效时长 {REF_EFF_S} s，转角 {REF_YAW_DEG}°；下表参照行是现场复算）；"
              f"生效时长或转角对参照偏离超过 {TIMING_FLAG:.0%} 标 ⚠（前飞位移是生效时长的直接后果，只显示不标记）。", ""] + t_table

    d_head = ["| 变体 | 步段 | 样本数 | Δ 均值 °（带符号） | Δ 标准差 ° | Δ 绝对值均值 ° | Δ 绝对值 90 分位 ° | Δ 绝对值最大 ° |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    d_rows = []
    for name in (MAIN, BEST):
        eps_ = episode_dirs(vdirs[name])
        if eps_:
            d_rows += drift_lines(name, heading_drift(eps_))
    g_refs = variants(out)[MAIN][1]
    if g_refs and episode_dirs(g_refs):
        d_rows += drift_lines("K=1 参照：护栏 2（下午）", heading_drift(episode_dirs(g_refs)))
    lines += ["", "## 9. 护栏臂「数步数」航向与真实航向的差（按步段）", "",
              f"估计航向 = {COUNT_ANGLE_DEG}° × 净转向数（已执行的 YAW_LEFT 数 − YAW_RIGHT 数，含当步）；真实航向 = 当步窗口末的 yaw 相对第 0 步。"
              f"Δ = 真实 − 估计，折到 ±180°，按 {DRIFT_SEG_STEPS} 步一段；样本 = 该段内所有集的所有步（同一集的相邻步不独立）。"
              "Δ 标准差为总体标准差。K=1 参照只有前两段（课程第 1 块约 30 步）。", ""] + d_head + (d_rows or ["| 没有护栏臂的轨迹 | | | | | | | |"])

    lines += ["", "## 10. 块末 |y| 分布（首次过线那一步的横向偏移）", "",
              "只含通过该块的集（通过要求全程 |y| 不超走廊半宽），所以分布截在半宽以内；越界的集见第 3 节。教师行是「飞得直」的基线。", "",
              "| 变体 | 块 | 集数 | y 绝对值均值 m | 中位 m | 90 分位 m | 最大 m |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, _requested, _ok) in data.items():
        if rows:
            lines += block_end_y_lines(name, rows, k_blocks)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


def timing_only(specs, tol=TIMING_FLAG):
    """--timing：只打印时序健康表（标定用）。每项 '标签=目录1,目录2' 或单个目录（标签取目录名）；按最后一个 = 切，标签里可以带 =（如 K=3）。"""
    rows = []
    for spec in specs:
        if "=" in spec:
            label, dirs = spec.rsplit("=", 1)
        else:
            label, dirs = os.path.basename(os.path.normpath(spec.split(",")[0])), spec
        rows.append((label, arm_timing(dirs.split(",")), False))
    lines, _flagged = timing_lines(rows, tol=tol)
    print("\n".join(lines))
    return 0


def cli(argv=None):
    ap = argparse.ArgumentParser(description="长课程汇总（summary.md）与时序健康表")
    ap.add_argument("out", nargs="?", help="输出目录，默认本脚本所在目录")
    ap.add_argument("--title", help="追加到 summary.md 一级标题后面的括号文字")
    ap.add_argument("--present-only", action="store_true", help="分母 = 目录里实际有轨迹的集数；没有 eval.json 的臂按 episodes/seed_N 读")
    ap.add_argument("--timing", nargs="+", metavar="[标签=]目录[,目录...]", help="只打印这些目录的时序健康表")
    ap.add_argument("--tol", type=float, default=TIMING_FLAG, help="--timing 的标记阈值（默认 0.03）")
    a = ap.parse_args(argv)
    if a.timing:
        return timing_only(a.timing, a.tol)
    return main(os.path.abspath(a.out) if a.out else None, title=a.title, present_only=a.present_only)


if __name__ == "__main__":
    sys.exit(cli())
