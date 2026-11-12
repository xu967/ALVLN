#!/usr/bin/env python3
"""W1-13 冒烟与插件确认时刻（plugin_ack_sim_t）测量的汇总，写 summary.md。只用标准库，随时可跑，缺的臂标「未完成」。

臂目录（run_smoke_ack.sh 的命名，每个臂是一次 eval_closed_loop.py 的 --out）：
  smoke   冒烟：教师 K=1，2 集，--action-latency L
  idle    测量，空闲条件：教师 K=1，5 集，不加占核进程
  load    测量，满载条件：教师 K=1，同样 5 集，另起 8 个占核进程
  <臂>.cg 臂开始 / 结束时的 cgroup 记录（epoch、usage_usec、nr_throttled、throttled_usec），算平均核数与限流增量
  <臂>.hog 满载臂：占核进程组的 PGID、启动时刻、命令行、停止记录

量（逐步，只用 episodes/seed_*/actions.jsonl）：
  观测戳        = 行里的 pose_ts（就是 collect 的 pose_before_ts，迟到判据的 target_ns = 观测戳 + L 用的就是它）
  Python 侧派发 = 观测戳 + dispatch_lag_s（collect 看到的派发位姿戳）
  插件侧确认    = plugin_ack_sim_t（state 缓冲里第一帧截止等于本步截止的 gz 仿真秒，分辨率 5 ms）
  报三个差（毫秒）：确认 − (观测戳 + L)（任务卡要的分布）、Python 侧派发 − (观测戳 + L)（现在迟到判据能看到的）、确认 − Python 侧派发（传输加插件排队）。

预注册的判定（run_smoke_ack.sh 文件头原文）：每个条件分别看「确认 − (观测戳 + L)」的 p99：两个条件都小于 10 ms 只记录；任一条件不小于 10 ms，
迟到判据改用 plugin_ack_sim_t，阈值按量到的分布定并写明依据（那是后续提交）。

冒烟判定（--smoke-check）：eval 正常结束；每一行都有 safe_gap_s、obs_gap_s、plugin_ack_sim_t 三个键；每集第 0 步前两个为 null、其余非 null；
plugin_ack_sim_t 非 null 的行不少于 95%；非 null 的确认时刻不早于 Python 侧派发戳减 6 ms（动作发布之后才可能被插件确认，6 ms = 一个 state 周期加余量）。

用法：summarize_ack.py [输出目录] [--note 文字] [--latency L]
      summarize_ack.py --smoke-check 臂目录 [--latency L]    stdout 一行「PASS|FAIL 说明」，退出码 0 / 1
"""
import argparse
import glob
import json
import os
import sys

P99_LIMIT_S = 0.010          # 任务卡：p99 小于 10 ms 就只记录
SMOKE_MIN_ACK_FRACTION = 0.95
SMOKE_ACK_EARLY_TOL_S = 0.006
ARMS = ("smoke", "idle", "load")
ARM_TEXT = {"smoke": "冒烟（教师 K=1，2 集）", "idle": "空闲条件（教师 K=1，5 集）", "load": "满载条件（教师 K=1，5 集，另起 8 个占核进程）"}


def percentile(xs, q):
    """线性插值分位数（与 numpy 默认一致）；q 在 0..100。"""
    s = sorted(xs)
    if not s:
        return None
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def describe(xs):
    """秒 → {n, min, median, p90, p99, max}（仍是秒）；空列表只有 n。"""
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": min(xs), "median": percentile(xs, 50), "p90": percentile(xs, 90),
            "p99": percentile(xs, 99), "max": max(xs)}


def read_episodes(arm_dir):
    """{seed: [行...]}（按 step 排序）；没有 episodes 目录返回 {}。"""
    out = {}
    for path in sorted(glob.glob(os.path.join(arm_dir, "episodes", "seed_*", "actions.jsonl"))):
        seed = int(os.path.basename(os.path.dirname(path)).split("_", 1)[1])
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        out[seed] = sorted(rows, key=lambda r: r.get("step", 0))
    return out


def read_eval(arm_dir):
    path = os.path.join(arm_dir, "eval.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def step_deltas(row, latency_s):
    """一行 → (确认 − (观测戳 + L), Python 侧派发 − (观测戳 + L), 确认 − Python 侧派发)，秒；缺字段的项为 None。"""
    obs = row["pose_ts"] / 1e9 if row.get("pose_ts") is not None else None
    lag = row.get("dispatch_lag_s")
    ack = row.get("plugin_ack_sim_t")
    d_py = None if lag is None else lag - latency_s
    d_ack = None if (ack is None or obs is None) else ack - (obs + latency_s)
    d_tr = None if (ack is None or obs is None or lag is None) else ack - (obs + lag)
    return d_ack, d_py, d_tr


def collect_deltas(episodes, latency_s):
    ack, py, tr = [], [], []
    steps = n_key = 0
    for rows in episodes.values():
        for r in rows:
            steps += 1
            n_key += "plugin_ack_sim_t" in r
            a, p, t = step_deltas(r, latency_s)
            if a is not None:
                ack.append(a)
            if p is not None:
                py.append(p)
            if t is not None:
                tr.append(t)
    return {"steps": steps, "with_key": n_key, "ack": ack, "py": py, "tr": tr}


def read_cg(path):
    """<臂>.cg → (平均核数, 限流次数增量, 限流微秒增量, 墙钟秒)；缺记录返回 None。"""
    if not os.path.isfile(path):
        return None
    marks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 5 and p[0] in ("start", "end"):
                marks[p[0]] = [float(x) for x in p[1:]]
    if set(marks) != {"start", "end"}:
        return None
    (t0, u0, n0, h0), (t1, u1, n1, h1) = marks["start"], marks["end"]
    wall = t1 - t0
    return ((u1 - u0) / 1e6 / wall if wall > 0 else None), int(n1 - n0), int(h1 - h0), wall


def smoke_check(arm_dir, latency_s):
    """冒烟判定：返回 (通过?, 说明)。"""
    ev = read_eval(arm_dir)
    if ev is None:
        return False, f"{arm_dir} 没有 eval.json（评测没有正常结束）"
    if ev.get("partial"):
        return False, "eval.json 是 partial（评测被中断）"
    eps = read_episodes(arm_dir)
    if not eps:
        return False, "episodes/ 下没有逐步日志"
    problems, steps, with_ack, early = [], 0, 0, 0
    for seed, rows in eps.items():
        if not rows:
            problems.append(f"seed {seed} 没有行")
            continue
        for i, r in enumerate(rows):
            steps += 1
            miss = [k for k in ("safe_gap_s", "obs_gap_s", "plugin_ack_sim_t") if k not in r]
            if miss:
                problems.append(f"seed {seed} 步 {r.get('step', i)} 缺键 {miss}")
                continue
            for k in ("safe_gap_s", "obs_gap_s"):
                if (i == 0) != (r[k] is None):
                    problems.append(f"seed {seed} 步 {r.get('step', i)} {k}={r[k]!r}（首步应为 null、其余应非 null）")
            if r["plugin_ack_sim_t"] is not None:
                with_ack += 1
                lag = r.get("dispatch_lag_s")
                if lag is not None and r.get("pose_ts") is not None and \
                        r["plugin_ack_sim_t"] < r["pose_ts"] / 1e9 + lag - SMOKE_ACK_EARLY_TOL_S:
                    early += 1
    frac = with_ack / steps if steps else 0.0
    if frac < SMOKE_MIN_ACK_FRACTION:
        problems.append(f"plugin_ack_sim_t 非 null 的行只有 {with_ack}/{steps}（{frac:.1%}，要求 ≥ {SMOKE_MIN_ACK_FRACTION:.0%}）")
    if early:
        problems.append(f"{early} 行的确认时刻早于 Python 侧派发戳 {SMOKE_ACK_EARLY_TOL_S * 1e3:g} ms 以上（不可能：说明匹配到了别的帧）")
    if problems:
        return False, "；".join(problems[:6]) + (f"；另有 {len(problems) - 6} 条" if len(problems) > 6 else "")
    return True, f"{len(eps)} 集 {steps} 步，三个键齐全，确认非 null {with_ack}/{steps}（{frac:.1%}）"


def measurement_check(out_dir, latency_s, min_episodes=5):
    """核对 idle 与 load 两个测量臂各有至少 min_episodes 集非空日志且确认率达标（防止采集全崩时 DONE 仍报 rc=0）。"""
    problems = []
    for arm in ("idle", "load"):
        adir = os.path.join(out_dir, arm)
        ev = read_eval(adir)
        if ev is None or ev.get("partial"):
            problems.append(f"{arm} 没有非 partial 的 eval.json")
            continue
        eps = {s: r for s, r in read_episodes(adir).items() if r}
        if len(eps) < min_episodes:
            problems.append(f"{arm} 非空轨迹集数 {len(eps)} < {min_episodes}")
            continue
        data = collect_deltas(eps, latency_s)
        frac = len(data["ack"]) / data["steps"] if data["steps"] else 0.0
        if frac < SMOKE_MIN_ACK_FRACTION:
            problems.append(f"{arm} 确认非 null 只有 {len(data['ack'])}/{data['steps']}（{frac:.1%}）")
    if problems:
        return False, "；".join(problems)
    return True, f"idle 与 load 各 ≥ {min_episodes} 集非空轨迹且确认率 ≥ {SMOKE_MIN_ACK_FRACTION:.0%}"


def scan_partials(out_dir):
    """E-1：扫描 <臂>.partial-* 目录，返回 [(目录名, eval 摘要字符串, 已落盘集数, 步数)]。"""
    found = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.partial-*"))):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        ev = read_eval(path)
        eps = read_episodes(path)
        steps = sum(len(r) for r in eps.values())
        ev_str = ("无 eval.json" if ev is None else
                  f"episodes_done={ev.get('episodes_done')} failed_count={ev.get('failed_count')} partial={bool(ev.get('partial'))}")
        found.append((name, ev_str, len(eps), steps))
    return found


def ms(x, nd=1):
    return "—" if x is None else f"{x * 1e3:.{nd}f}"


def stats_row(label, xs):
    d = describe(xs)
    if not d["n"]:
        return f"| {label} | 0 | — | — | — | — | — |"
    return (f"| {label} | {d['n']} | {ms(d['min'])} | {ms(d['median'])} | {ms(d['p90'])} | {ms(d['p99'])} | {ms(d['max'])} |")


def summarize(out, latency_s, note=""):
    lines = ["# W1-13 冒烟与插件确认时刻测量（plugin_ack_sim_t）", ""]
    if note:
        lines += [f"_{note}_", ""]
    lines += [f"L = {latency_s:g} s；观测戳 = 行里的 pose_ts；判定阈值 p99 < {P99_LIMIT_S * 1e3:g} ms（任务卡）。", ""]
    start = os.path.join(out, "start.txt")
    if os.path.isfile(start):
        lines += ["## 起跑记录", "", "```", open(start, encoding="utf-8").read().rstrip(), "```", ""]
    partials = scan_partials(out)
    if partials:
        lines += ["## 重评与半成品记录", ""]
        for pname, pev, peps, psteps in partials:
            lines.append(f"- `{pname}`：{pev}，已落盘 {peps} 集 / {psteps} 步")
        lines.append("")
    verdict, p99s, maxs = {}, {}, {}
    for arm in ARMS:
        adir = os.path.join(out, arm)
        lines += [f"## {arm}：{ARM_TEXT[arm]}", ""]
        ev = read_eval(adir)
        if ev is None:
            lines += ["未完成（没有 eval.json）。", ""]
            continue
        eps = read_episodes(adir)
        data = collect_deltas(eps, latency_s)
        lines += [f"- 评测：episodes_done={ev.get('episodes_done')} failed_count={ev.get('failed_count')} "
                  f"partial={bool(ev.get('partial'))} 迟到集 {len(ev.get('late_dispatch_episodes') or [])} "
                  f"停顿集 {len(ev.get('stall_episodes') or [])}",
                  f"- 步数 {data['steps']}，行里有 plugin_ack_sim_t 键 {data['with_key']}，值非 null {len(data['ack'])}"]
        cg = read_cg(os.path.join(out, arm + ".cg"))
        if cg:
            cores, dn, dh, wall = cg
            lines.append(f"- 容器 CPU：平均 {cores:.2f} 核（整个 cgroup，含别的 agent），限流 {dn} 次 / {dh / 1e6:.2f} s，墙钟 {wall:.0f} s")
        hog = os.path.join(out, arm + ".hog")
        if os.path.isfile(hog):
            lines += ["- 占核进程：", "```", open(hog, encoding="utf-8").read().rstrip(), "```"]
        lines += ["", "| 量（毫秒） | n | 最小 | p50（中位） | p90 | p99 | 最大 |", "|---|---|---|---|---|---|---|",
                  stats_row("确认 − (观测戳 + L)", data["ack"]),
                  stats_row("Python 侧派发 − (观测戳 + L)", data["py"]),
                  stats_row("确认 − Python 侧派发", data["tr"]), ""]
        if arm != "smoke" and data["ack"]:
            p99s[arm] = percentile(data["ack"], 99)
            maxs[arm] = max(data["ack"])
        if arm == "smoke":
            ok, text = smoke_check(adir, latency_s)
            lines += [f"冒烟判定：**{'PASS' if ok else 'FAIL'}**，{text}", ""]
            verdict["smoke"] = ok
    lines += ["## 判定（预注册：两个条件的 p99 都 < 10 ms 只记录；否则给建议判据与阈值交主会话定，本轮不改迟到判据）", ""]
    if not p99s:
        lines.append("没有测量数据。")
    else:
        for arm, p in p99s.items():
            lines.append(f"- {arm}：p99 = {p * 1e3:.1f} ms（最大 {maxs[arm] * 1e3:.1f} ms），{'< 10 ms' if p < P99_LIMIT_S else '≥ 10 ms'}")
        if len(p99s) == 2:
            if all(p < P99_LIMIT_S for p in p99s.values()):
                lines.append("- 结论：两个条件都 < 10 ms，插件确认时刻只记录，不改迟到判据。")
            else:
                worst_p99_ms = max(p99s.values()) * 1e3
                suggest_ms = max(25.0, round(worst_p99_ms + 10.0, 1))
                lines.append(f"- 结论：至少一个条件 ≥ 10 ms，本轮不改迟到判据；建议迟到判据要改用 plugin_ack_sim_t "
                             f"（`plugin_ack_sim_t is None or plugin_ack_sim_t - (pose_ts/1e9 + L) > 阈值`），"
                             f"建议阈值 {suggest_ms:.1f} ms（基于观测到的最高 p99 = {worst_p99_ms:.1f} ms 加 10 ms 余量），由主会话定。")
        else:
            lines.append("- 结论：两个条件的数据不全，不下结论。")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--note", default="")
    ap.add_argument("--latency", type=float, default=0.20)
    ap.add_argument("--smoke-check", metavar="ARMDIR", default=None)
    ap.add_argument("--measurement-check", metavar="OUTDIR", default=None)
    ap.add_argument("--min-episodes", type=int, default=5)
    a = ap.parse_args()
    if a.smoke_check:
        ok, text = smoke_check(a.smoke_check, a.latency)
        print(("PASS " if ok else "FAIL ") + text)
        return 0 if ok else 1
    if a.measurement_check:
        ok, text = measurement_check(a.measurement_check, a.latency, a.min_episodes)
        print(("PASS " if ok else "FAIL ") + text)
        return 0 if ok else 1
    print(summarize(a.out, a.latency, a.note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
