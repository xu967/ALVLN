#!/usr/bin/env python3
"""exp-long-course-20261110 汇总：K 个 base 柱阵块首尾相接的长课程，逐块存活率，写 summary.md（随时可跑，缺的臂标「未完成」）。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py [输出目录，默认本脚本所在目录]
口径：metric_version sim09.v2.coast1s.corridor3m；分母 = 请求集数（20 个冻结 seed，采集失败的集算没通过）。

逐块存活（每集读 actions.jsonl + oracle.jsonl，第 i 行互相对应）：
  第 k 块通过 = 首次 pose_gt.x >= block_goal_x[k]（该块最远柱 x + 1.0），且到这一步（含）为止没有接触碰撞、|y| 全程 <= 走廊半宽。
  S_k = 通过前 k 块的集数 / 请求集数（累计，S_1 >= S_2 >= ... >= S_K）；条件存活 S_k / S_{k-1}。
  失败归属：第一个坏事件（接触碰撞 = 撞柱，|y| > 半宽 = 越界）发生时正在穿越的那一块；没有坏事件但集结束时还没过线 = 未到线。
  S_K 与 eval.json 的 success（整条课程过线 + 无碰撞 + 不越界）逐集应当一致，汇总末尾核对。
K=1 参照：同权重、同 seed 的今天的 K=1 评测（exp-swa-calib / exp-heading-guard），用来算第 1 块与它的逐集一致率与独立块模型预测。
"""
import json
import math
import os
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
    return (c - h, c + h)


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


def block_outcomes(actions, oracle, goals, half_width=DEFAULT_HALF_WIDTH_M, aborted=False):
    """一集的逐块结果（纯函数）。goals = 每块的过线 x（block_goal_x）。

    返回 {"pass": [bool]*K, "fail": None 或 (块序号从 0 起, "collision"|"corridor"|"not_reached"), "cross_step": [首次过线的步号或 None]*K}。
    同一步既碰撞又越界按撞柱记；过线那一步本身有碰撞 / 越界，这一块不算过（与 K=1 判据：停线前任何碰撞都算失败 同口径）。
    aborted=True 是采集中途失败的集（多为固件 FAILSAFE：电池掉压后推力不够）：没有坏事件时记「采集中止」而不是「未到线」。
    """
    k_total = len(goals)
    passed, cross = [False] * k_total, [None] * k_total
    cur, fail = 0, None   # cur = 已通过的块数 = 正在穿越的块
    for i in range(max(len(actions), len(oracle))):
        a = actions[i] if i < len(actions) else {}
        o = oracle[i] if i < len(oracle) else {}
        pg = o.get("pose_gt") if isinstance(o, dict) else None
        has_pose = isinstance(pg, dict) and "x" in pg and "y" in pg
        collided = bool(a.get("collision")) and a.get("src") != "stuck"   # stuck（前飞位移不足）不是碰撞
        outside = has_pose and abs(float(pg["y"])) > half_width
        if fail is None and (collided or outside):
            fail = (cur, "collision" if collided else "corridor")
        if fail is None and has_pose:
            while cur < k_total and float(pg["x"]) >= goals[cur]:
                passed[cur], cross[cur] = True, i
                cur += 1
    if fail is None and cur < k_total:
        fail = (cur, "aborted" if aborted else "not_reached")
    return {"pass": passed, "fail": fail, "cross_step": cross}


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


def load_arm(edir):
    """-> None（没有 eval.json）或 {"complete", "requested", "failed", "K", "eps": {seed: 行}}。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
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
        row = {"success": False, "aborted": True, **episode_stats(actions, oracle),
               "stderr": " ".join(f.get("stderr_tail") or [f.get("error", "")])}
        if k_blocks and oracle:
            row.update(block_outcomes(actions, oracle, [FIRST_GOAL_X + BLOCK_PITCH_M * k for k in range(k_blocks)], half, aborted=True))
        eps[int(f["seed"])] = row
    requested = e.get("episodes_requested") or len(eps)
    # 完整 = 没中断，且请求的集都跑过（done + failed）；电池掉压造成的采集中止是测量结果，不重跑
    complete = bool(not e.get("partial") and (e.get("episodes_done") or 0) + (e.get("failed_count") or 0) == requested)
    return {"complete": complete, "requested": requested, "failed": e.get("failed_count") or 0,
            "skipped": e.get("skipped_count") or 0, "K": k_blocks, "eps": eps, "partial": bool(e.get("partial"))}


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


def main(out=None):
    out = out or OUT
    arms = {name: [load_arm(d) for d in dirs] for name, (dirs, _refs) in variants(out).items()}
    data = {name: pooled(a) for name, a in arms.items()}
    ref = {name: None if refs is None else {(i, s): v for i, d in enumerate(refs) if (r := load_ref(d)) for s, v in r.items()}
           for name, (_dirs, refs) in variants(out).items()}
    k_blocks = next((arm["K"] for a in arms.values() for arm in a if arm and arm["K"]), None)
    if not k_blocks:
        print("还没有任何带 blocks 的 eval.json（实验未开始或第一集未结束）")
        return 1
    ks = range(k_blocks)
    lines = [f"# exp-long-course-20261110：{k_blocks} 块长课程的逐块存活率", "",
             f"口径 `sim09.v2.coast1s.corridor3m`；成功线 = {6.0 + BLOCK_PITCH_M * (k_blocks - 1):g} m（6.0 + 5.0·(K−1)，K={k_blocks}）；"
             "20 个冻结评测 seed；hold 1.0，observation，cf_sitl lib31（起始 SoC 0.5 不变），E1 MINI。"
             "S_k = 通过前 k 块的集数 / 请求集数（累计）；Wilson 95%；预注册见 `run.sh` 文件头。", "",
             "## 1. 逐块存活率 S_k", "",
             "| 变体 | 请求集数 | " + " | ".join(f"S_{k + 1}" for k in ks) + " | 状态 |", "| --- | --- | " + " | ".join("---" for _ in ks) + " | --- |"]
    surv = {}   # 变体 -> [S_1..S_K 的通过数]，只收有数据的
    for name, (rows, requested, ok) in data.items():
        if not rows:
            lines.append(f"| {name} | | " + " | ".join("" for _ in ks) + " | 未完成 |")
            continue
        cnt = [sum(1 for r in rows.values() if passed_k(r, k)) for k in ks]
        surv[name] = (cnt, requested)
        state = "完整" if ok else f"进行中（已出 {len(rows)} 集）"
        lines.append(f"| {name} | {requested} | " + " | ".join(fmt_frac(c, requested) for c in cnt) + f" | {state} |")
    lines += ["", "条件存活 S_k / S_{k-1}（通过前一块的集里，这一块也通过的比例）：", "",
              "| 变体 | " + " | ".join(f"块 {k + 1}" for k in ks) + " |", "| --- | " + " | ".join("---" for _ in ks) + " |"]
    for name, (cnt, requested) in surv.items():
        prev = [requested] + cnt[:-1]
        lines.append(f"| {name} | " + " | ".join(f"{c}/{p}={c / p:.2f}" if p else "-" for c, p in zip(cnt, prev)) + " |")

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
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None))
