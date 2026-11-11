#!/usr/bin/env python3
"""迟到步的离线计时与 A* 展开量（第一轮复盘，任务卡 W1-0 的迟到步归因）。

问题：第一轮 L=0.16 的 K=3 臂有 36 个迟到步，它们慢在哪？主会话的推断是「左右平局推演（lr_tie_margin_steps，约 200 ms）」。
先说两个事实：lr_tie_margin_steps 只在 tiebreak="visible" 时才跑，第一轮所有臂的 manifest 都是 teacher_tiebreak=legacy，
legacy 下 plan_action_global_tiebreak 只调一次 plan_action_global（一次 A* + 纯追踪），返回的 tie_margin 恒为 None；
lr_tie_margin_steps 里滚出的是 YAW_LEFT / YAW_RIGHT 两次 A*（4 个候选是 plan_costs 的事）。所以平局推演不可能是第一轮的慢源，
这里用离线计时验证慢源到底是什么，并给出「迟到步是不是集中在会触发平局推演的位姿」的直接读数。

做法（只读 $OUT 下的 episodes/，不起 gz；教师代码用 $OUT/code 里的冻结副本，与第一轮同一份）：
  1. 位姿 = 每步 actions.jsonl 的 pose_before / yaw_before；迟到步 = late_dispatch 为 true 的步。
  2. 场景用 scene_gen（同 seed、同 scene-id、同参数）在临时目录重生成，world_sha256 与 scene_meta_sha256 对 episode_manifest 逐集核对；
     对不上的场景跳过并写明。柱子取 scene.json 的 pillars；教师按 collect 的口径 set_vehicle("e1_mini", task_of_scene(meta)) 并 warm_free_cells
     （与带 --action-latency 的每集预热一致）。
  3. 计数遍（教师四个臂的全部步）：teacher.heapq 换成计数代理，记 A* 出堆数（节点展开量）、入堆数；包装 astar_path / plan_action 记调用次数、
     A* 是否无路而退回局部规划。按计数把每步分三类（类别在看过迟到清单之后才定，属事后分类，不是预注册）：
       F = A* 无路，退回局部 plan_action；H = A* 有路且出堆数 ≥ POPS_HEAVY（起点离终点远的长路径）；N = 其余。
     输出每类的迟到率。
  4. 计时遍（抽样）：对每个位姿调 plan_action_global_tiebreak(x, y, yaw, pillars, "legacy")，重复 5 次，取墙钟中位数与 CPU 时间中位数
     （CPU 时间不受调度等待影响，机器被别的进程占满时读它）；每个位姿前 gc.collect()；位姿打乱顺序交错测。
     样本 = 全部迟到步 + 对照 A（K=3 同批集里随机 20 个不迟到步）+ 对照 B（步号 ≤ 3 的不迟到步，控制起飞段位置）+ 对照 C（不迟到的 YAW 步，随机 20）
     + 分类抽样 D（每类 15 个不迟到步，K=3 两个 L 合并）。
  5. 对每个计时位姿在 visible 口径下补算 lr_tie_margin_steps（3 次中位数）与 margin：margin < TIE_MARGIN_STEPS 且 legacy 动作是 YAW 才会触发改向；
     只回答「迟到步是不是集中在会触发平局推演的位姿」，第一轮没有跑它，不能算第一轮的耗时来源。
运行：.venv-tools/bin/python profile_late_steps.py [输出目录，默认 $UAVRGB_DATA/exp-action-latency-20261110] [--note 说明]
结果写 <输出目录>/profile-late-steps.md 与 profile-late-steps.json。运行期间 cgroup 的 nr_throttled 增量一并记下。
"""
import argparse
import gc
import hashlib
import heapq
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize as sm  # noqa: E402

REPEATS = 5                # 每个位姿的计时重复数（取中位数）
TIE_REPEATS = 3            # lr_tie_margin_steps 的重复数
N_CONTROL = 20             # 对照 A / C 的步数
N_PER_CLASS = 15           # 分类抽样 D 每类的步数
CONTROL_SEED = 20261111    # 抽样的随机种子
EARLY_MAX_STEP = 3         # 对照 B：步号 ≤ 3
POPS_HEAVY = 1500          # H 类的出堆数下限：起飞段长路径的出堆数在 1900–3000，沿航线中段多在 1000 以下
TEMPLATE_REL = os.path.join("sim", "worlds", "uav_rgb_e1.sdf")
ACTION_NAMES = {0: "SAFE_NEUTRAL", 1: "FORWARD_SLOW", 2: "YAW_LEFT", 3: "YAW_RIGHT"}
TURNS = (2, 3)
CLASS_NAMES = {"F": "F：A* 无路，退回局部规划", "H": f"H：A* 有路，出堆数 ≥ {POPS_HEAVY}", "N": f"N：A* 有路，出堆数 < {POPS_HEAVY}"}


# ---------------------------------------------------------------- 统计小工具（纯 Python，没有 scipy）
def median_iqr(xs):
    """(中位数, p25, p75)；空返回 (nan, nan, nan)。"""
    if not xs:
        return float("nan"), float("nan"), float("nan")
    s = sorted(xs)

    def pct(p):
        k = (len(s) - 1) * p
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return pct(0.5), pct(0.25), pct(0.75)


def ranks(xs):
    """平均秩（并列取平均）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman 秩相关；样本少于 3 或某一列没有变化时返回 nan。"""
    if len(xs) != len(ys) or len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def mann_whitney(a, b):
    """Mann–Whitney U（a 对 b）：返回 (U_a, z, 双侧 p)。正态近似带并列校正；任一边为空返回 nan。"""
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return float("nan"), float("nan"), float("nan")
    allv = list(a) + list(b)
    rk = ranks(allv)
    u1 = sum(rk[:n1]) - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    counts = {}
    for v in allv:
        counts[v] = counts.get(v, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return u1, float("nan"), float("nan")
    z = (u1 - n1 * n2 / 2.0) / math.sqrt(var)
    return u1, z, math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------- 计数代理
class CountingHeapq:
    """替换 teacher.heapq：转发 heappush / heappop 并计数；其余属性透传给真正的 heapq。"""

    def __init__(self):
        self.pops = 0
        self.pushes = 0

    def heappop(self, heap):
        self.pops += 1
        return heapq.heappop(heap)

    def heappush(self, heap, item):
        self.pushes += 1
        heapq.heappush(heap, item)

    def __getattr__(self, name):
        return getattr(heapq, name)


def count_call(teacher, fn, *args, **kwargs):
    """在计数代理下调一次 fn，返回 (结果, {pops, pushes, astar_calls, fallback_calls, path_cells})。
    astar_path 与 plan_action 在 teacher 模块里被包装一次后立刻还原，不改库文件。"""
    proxy = CountingHeapq()
    stats = {"astar_calls": 0, "fallback_calls": 0, "path_cells": None}
    orig_heapq, orig_astar, orig_plan = teacher.heapq, teacher.astar_path, teacher.plan_action

    def astar(*a, **k):
        stats["astar_calls"] += 1
        r = orig_astar(*a, **k)
        stats["path_cells"] = None if r is None else len(r)
        return r

    def plan(*a, **k):
        stats["fallback_calls"] += 1
        return orig_plan(*a, **k)
    teacher.heapq, teacher.astar_path, teacher.plan_action = proxy, astar, plan
    try:
        result = fn(*args, **kwargs)
    finally:
        teacher.heapq, teacher.astar_path, teacher.plan_action = orig_heapq, orig_astar, orig_plan
    stats["pops"], stats["pushes"] = proxy.pops, proxy.pushes
    return result, stats


def classify(counts, pops_heavy=POPS_HEAVY):
    """F = A* 无路退回局部规划；H = A* 有路且出堆数 ≥ pops_heavy；N = 其余。"""
    if counts["fallback_calls"] > 0:
        return "F"
    return "H" if counts["pops"] >= pops_heavy else "N"


def time_call(fn, *args, repeats=REPEATS, **kwargs):
    """fn 重复 repeats 次的 (墙钟 ms 列表, CPU 时间 ms 列表)；每次计时前 gc.collect()。"""
    wall, cpu = [], []
    for _ in range(repeats):
        gc.collect()
        c0, t0 = time.process_time(), time.perf_counter()
        fn(*args, **kwargs)
        wall.append((time.perf_counter() - t0) * 1000.0)
        cpu.append((time.process_time() - c0) * 1000.0)
    return wall, cpu


def measure_pose(teacher, pillars, x, y, yaw, repeats=REPEATS, tie_repeats=TIE_REPEATS):
    """一个位姿的计时读数：legacy 的墙钟 / CPU 中位数、visible 口径的平局 margin 与耗时。"""
    (action, tie_margin), _ = count_call(teacher, teacher.plan_action_global_tiebreak, x, y, yaw, pillars, "legacy")
    wall, cpu = time_call(teacher.plan_action_global_tiebreak, x, y, yaw, pillars, "legacy", repeats=repeats)
    tie_wall, tie_cpu = time_call(teacher.lr_tie_margin_steps, x, y, yaw, pillars, repeats=tie_repeats)
    margin = teacher.lr_tie_margin_steps(x, y, yaw, pillars)
    triggers = (action in TURNS) and margin < teacher.TIE_MARGIN_STEPS
    return {"action": action, "legacy_tie_margin": tie_margin, "wall_ms": statistics.median(wall), "cpu_ms": statistics.median(cpu),
            "wall_ms_all": wall, "cpu_ms_all": cpu, "tie_margin": margin, "tie_wall_ms": statistics.median(tie_wall),
            "tie_cpu_ms": statistics.median(tie_cpu), "tie_triggers": bool(triggers), "tie_below": bool(margin < teacher.TIE_MARGIN_STEPS)}


# ---------------------------------------------------------------- 步记录与抽样
def collect_steps(arms):
    """arms = [(标签, 目录, 种类 k1|k3)] → 步记录列表。每步：arm、kind、seed、step、action、x、y、yaw、lag、late、rtf、L、manifest 哈希。"""
    rows = []
    for label, edir, kind in arms:
        for seed, actions, oracle, man in sm.episode_records(edir):
            rtf = (man.get("performance") or {}).get("action_rtf")
            for i, a in enumerate(actions):
                pose = a.get("pose_before")
                if not pose or a.get("yaw_before") is None or pose.get("x") is None:
                    continue
                rows.append({"arm": label, "kind": kind, "seed": seed, "step": i, "action": a.get("action_executed"),
                             "x": pose["x"], "y": pose["y"], "yaw": a["yaw_before"], "lag": a.get("dispatch_lag_s"),
                             "late": bool(a.get("late_dispatch")), "rtf": rtf, "L": man.get("action_latency_s"),
                             "scene_meta_sha256": man.get("scene_meta_sha256"), "world_sha256": man.get("world_sha256")})
    return rows


def pick_controls(rows, late_arm, seed=CONTROL_SEED, n=N_CONTROL, early_max=EARLY_MAX_STEP):
    """对照三组，都取自 late_arm 里出现过迟到步的那批集的不迟到步。返回 {"A": [...], "B": [...], "C": [...]}。
    A 随机 n 步；B 步号 ≤ early_max 的全部；C 动作是 YAW 的随机 n 步（迟到集里没有 YAW 迟到步时为空）。"""
    arm_rows = [r for r in rows if r["arm"] == late_arm]
    late_seeds = {r["seed"] for r in arm_rows if r["late"]}
    pool = [r for r in arm_rows if r["seed"] in late_seeds and not r["late"]]
    rng = random.Random(seed)
    a = rng.sample(pool, min(n, len(pool)))
    b = [r for r in pool if r["step"] <= early_max]
    late_turn = any(r["late"] and r["action"] in TURNS for r in arm_rows)
    c_pool = [r for r in pool if r["action"] in TURNS] if late_turn else []
    c = rng.sample(c_pool, min(n, len(c_pool)))
    return {"A": a, "B": b, "C": c}


def pick_by_class(rows, seed=CONTROL_SEED, n=N_PER_CLASS):
    """rows 里已带 klass 的不迟到步，每类随机 n 个；返回 {类: [...]}。"""
    rng = random.Random(seed + 1)
    out = {}
    for k in CLASS_NAMES:
        pool = [r for r in rows if r.get("klass") == k and not r["late"]]
        out[k] = rng.sample(pool, min(n, len(pool)))
    return out


def late_rate_table(rows, arm_labels):
    """{臂: {类: [迟到数, 总数]}}；rows 里每步要带 klass。"""
    out = {}
    for arm in arm_labels:
        cells = {k: [0, 0] for k in CLASS_NAMES}
        for r in rows:
            if r["arm"] == arm and r.get("klass") in cells:
                cells[r["klass"]][0] += 1 if r["late"] else 0
                cells[r["klass"]][1] += 1
        out[arm] = cells
    return out


# ---------------------------------------------------------------- 场景重生成与核对
def regenerate_scene(code_dir, seed, kind, workdir):
    """用冻结副本里的 scene_gen 同参数重生成 seed 的场景；返回 (scene.json 路径, 错误文字或 None)。"""
    world = os.path.join(workdir, f"scene_{kind}_{seed}.sdf")
    cmd = [sys.executable, os.path.join(code_dir, "sim", "tools", "scene_gen.py"), "--seed", str(seed), "--scene-id", str(seed),
           "--out", world, "--template", os.path.join(code_dir, TEMPLATE_REL), "--appearance", "v1"]
    cmd += ["--task", "course_base", "--blocks", "3"] if kind == "k3" else ["--task", "base"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=code_dir)
    if r.returncode != 0:
        return None, f"scene_gen 失败：{r.stderr.strip()[:200]}"
    return os.path.join(workdir, f"scene_{kind}_{seed}.json"), None


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_scene(meta_path, world_sha, meta_sha):
    """核对重生成的场景与 episode_manifest 里记的哈希；返回 (world 一致, meta 一致)。manifest 没记的那项算不一致。"""
    meta = sm.read_json(meta_path)
    return (world_sha is not None and meta.get("world_sha256") == world_sha), (meta_sha is not None and sha256_file(meta_path) == meta_sha)


# ---------------------------------------------------------------- 报告
def fmt_stats(xs, nd=1):
    xs = [x for x in xs if x is not None]
    m, lo, hi = median_iqr(xs)
    if m != m:
        return "-"
    return f"{m:.{nd}f}（{lo:.{nd}f}–{hi:.{nd}f}；最大 {max(xs):.{nd}f}）"


def read_cpu_stat(path="/sys/fs/cgroup/cpu.stat"):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, v = line.split()
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


def group_row(label, ms):
    """ms = 计时结果（带 row、counts）。"""
    n = len(ms)
    if not n:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    yaw = sum(1 for m in ms if m["row"]["action"] in TURNS)
    below = sum(1 for m in ms if m["tie_below"])
    trig = sum(1 for m in ms if m["tie_triggers"])
    fb = sum(1 for m in ms if m["counts"]["fallback_calls"])
    return (f"| {label} | {n} | {fmt_stats([m['cpu_ms'] for m in ms])} | {fmt_stats([m['wall_ms'] for m in ms])} | {fmt_stats([m['counts']['pops'] for m in ms], 0)} | "
            f"{fb}/{n} | {yaw}/{n} | {below}/{n}（触发改向 {trig}） |")


def compare_line(label, late, ctrl, key="cpu_ms"):
    def med(ms, f):
        xs = [f(m) for m in ms]
        return statistics.median(xs) if xs else float("nan")
    ml, mc = med(late, lambda m: m[key]), med(ctrl, lambda m: m[key])
    pl, pc = med(late, lambda m: m["counts"]["pops"]), med(ctrl, lambda m: m["counts"]["pops"])
    _u, _z, p_t = mann_whitney([m[key] for m in late], [m[key] for m in ctrl])
    _u, _z, p_p = mann_whitney([m["counts"]["pops"] for m in late], [m["counts"]["pops"] for m in ctrl])
    ratio = ml / mc if mc == mc and mc > 0 else float("nan")
    pratio = pl / pc if pc == pc and pc > 0 else float("nan")
    return (f"- {label}（{len(late)} 对 {len(ctrl)}）：CPU 耗时中位 {ml:.1f} ms 对 {mc:.1f} ms，比值 {ratio:.2f}（Mann–Whitney p = {p_t:.3g}）；"
            f"A* 出堆数中位 {pl:.0f} 对 {pc:.0f}（p = {p_p:.3g}）。")


def class_section(rows, arms_labels):
    table = late_rate_table(rows, arms_labels)
    lines = ["## 2. 全部步按 A* 计数分类的迟到率（计数遍：教师四个臂的全部步）", "",
             "类别定义见文件头；类别在看过迟到清单之后才定，属事后分类。迟到率 = 迟到步数 / 该类步数。", "",
             "| 臂 | " + " | ".join(CLASS_NAMES.values()) + " |", "|---|" + "---|" * len(CLASS_NAMES)]
    for arm in arms_labels:
        cells = []
        for k in CLASS_NAMES:
            late, n = table[arm][k]
            cells.append(f"{late}/{n}（{100.0 * late / n:.1f}%）" if n else "0/0")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main(out=None, note=None, repeats=REPEATS):
    if out:
        sm.OUT = os.path.abspath(out)
    OUT = sm.OUT
    code_dir = os.path.join(OUT, "code")
    sys.path.insert(0, os.path.join(code_dir, "sim", "tools"))
    import teacher  # noqa: E402  冻结副本里的教师，与第一轮同一份
    teacher_sha = sha256_file(os.path.join(code_dir, "sim", "tools", "teacher.py"))
    teacher_arms = [("cal-L0.16-k3", sm.cal_dir(0.16, 3), "k3"), ("cal-L0.20-k3", sm.cal_dir(0.20, 3), "k3"),
                    ("cal-L0.16-k1", sm.cal_dir(0.16, 1), "k1"), ("cal-L0.20-k1", sm.cal_dir(0.20, 1), "k1")]
    student_arm = ("acc-L0.16-k1-s1（学生）", sm.acc_dir(0.16, 1, 1), "k1")
    rows = collect_steps(teacher_arms + [student_arm])
    cpu0, t_start = read_cpu_stat(), time.time()

    # 1) 重生成并核对全部用到的场景（同 (种类, seed, 哈希) 只生成一次）
    scenes, scene_rows = {}, []
    with tempfile.TemporaryDirectory(prefix="profile-late-", dir=os.environ.get("TMPDIR")) as tmp:
        for kind, seed, meta_sha, world_sha in sorted({(r["kind"], r["seed"], r["scene_meta_sha256"], r["world_sha256"]) for r in rows},
                                                       key=lambda t: (t[0], t[1], str(t[2]))):
            meta_path, err = regenerate_scene(code_dir, seed, kind, tmp)
            if err:
                scene_rows.append((kind, seed, None, None, err))
                continue
            w_ok, m_ok = verify_scene(meta_path, world_sha, meta_sha)
            scene_rows.append((kind, seed, w_ok, m_ok, None))
            if w_ok and m_ok:
                scenes[(kind, seed, meta_sha)] = sm.read_json(meta_path)
    rows = [r for r in rows if (r["kind"], r["seed"], r["scene_meta_sha256"]) in scenes]

    def with_scene(group_rows, fn):
        """按场景分组调 fn(meta, pillars, 本场景的行们)；组序固定。"""
        by = {}
        for r in group_rows:
            by.setdefault((r["kind"], r["seed"], r["scene_meta_sha256"]), []).append(r)
        for key in sorted(by, key=lambda k: (k[0], k[1])):
            meta = scenes[key]
            teacher.set_vehicle("e1_mini", teacher.task_of_scene(meta))
            teacher.warm_free_cells(meta["pillars"])
            fn(meta, meta["pillars"], by[key])

    # 2) 计数遍：教师四个臂的全部步
    teacher_labels = [a[0] for a in teacher_arms]

    def count_all(meta, pillars, rs):
        for r in rs:
            (action, tie), counts = count_call(teacher, teacher.plan_action_global_tiebreak, r["x"], r["y"], r["yaw"], pillars, "legacy")
            r["counts"], r["klass"], r["legacy_tie_margin"], r["replay_action"] = counts, classify(counts), tie, action
    with_scene([r for r in rows if r["arm"] in teacher_labels], count_all)
    counted = [r for r in rows if "counts" in r]
    n_agree = sum(1 for r in counted if r["replay_action"] == r["action"])

    # 3) 抽样并计时（学生臂的迟到步只计数 + 计时，不进分类表）
    late = [r for r in rows if r["late"]]
    controls = pick_controls(counted, "cal-L0.16-k3")
    by_class = pick_by_class([r for r in counted if r["kind"] == "k3"])
    wanted = [(r, "late") for r in late] + [(r, f"ctrl{g}") for g, rs in controls.items() for r in rs] + [(r, f"cls{k}") for k, rs in by_class.items() for r in rs]
    tags = {}
    for r, tag in wanted:
        tags.setdefault(id(r), (r, set()))[1].add(tag)
    to_time = [r for r, _t in tags.values()]
    rng = random.Random(CONTROL_SEED)
    rng.shuffle(to_time)
    results = {}

    def time_all(meta, pillars, rs):
        for r in rs:
            m = measure_pose(teacher, pillars, r["x"], r["y"], r["yaw"], repeats=repeats)
            if "counts" not in r:   # 学生臂的步：补一次计数
                _res, r["counts"] = count_call(teacher, teacher.plan_action_global_tiebreak, r["x"], r["y"], r["yaw"], pillars, "legacy")
                r["klass"] = classify(r["counts"])
            m.update({"row": r, "counts": r["counts"], "klass": r["klass"], "tags": sorted(tags[id(r)][1])})
            results[id(r)] = m
    with_scene(to_time, time_all)   # 组内顺序已被打乱的 to_time 决定；场景分组保证每个场景只 warm 一次
    cpu1 = read_cpu_stat()
    throttled = cpu1.get("nr_throttled", 0) - cpu0.get("nr_throttled", 0)

    def sel(tag, arm=None):
        return [m for m in results.values() if tag in m["tags"] and (arm is None or m["row"]["arm"] == arm)]
    late_016, late_020 = sel("late", "cal-L0.16-k3"), sel("late", "cal-L0.20-k3")
    late_k1 = sel("late", "cal-L0.16-k1") + sel("late", student_arm[0])
    ctrl_a, ctrl_b, ctrl_c = sel("ctrlA"), sel("ctrlB"), sel("ctrlC")

    lines = ["# 迟到步的离线计时与 A* 展开量（任务卡 W1-0 第一轮复盘）", ""]
    if note:
        lines += [f"> {note}", ""]
    lines += [f"方法与定义见 profile_late_steps.py 文件头。教师代码 = 冻结副本 sim/tools/teacher.py（sha256 {teacher_sha[:16]}…，与第一轮 inputs.txt 的是同一个文件）；"
              f"计时调用 = plan_action_global_tiebreak(legacy)，每个位姿 {repeats} 次取中位数，CPU 时间 = time.process_time。"
              f"测量期间 cgroup nr_throttled 增量 {throttled}，用时 {time.time() - t_start:.0f} s（容器被别的进程占满时墙钟被拉长，所以以 CPU 时间为主，读比值）。", "",
              "## 1. 场景重生成核对", "", "| 种类 | seed | world_sha256 与 manifest 一致 | scene_meta_sha256 与 manifest 一致 | 备注 |", "|---|---|---|---|---|"]
    for kind, seed, w_ok, m_ok, err in sorted(scene_rows, key=lambda t: (t[0], t[1])):
        lines.append(f"| {kind} | {seed} | {'-' if w_ok is None else ('是' if w_ok else '否')} | {'-' if m_ok is None else ('是' if m_ok else '否')} | {err or ''} |")
    n_bad = sum(1 for _k, _s, w, m, e in scene_rows if e or not (w and m))
    lines += ["", f"核对未过的场景 {n_bad} 个（其上的位姿没有测）。回放的教师动作与记录的 action_executed 逐步一致 {n_agree}/{len(counted)}。", ""]
    lines += class_section(rows, teacher_labels)
    lines += ["## 3. 分组读数（CPU 与墙钟单位 ms；括号内 = 四分位区间与最大值）", "",
              "| 组 | 位姿数 | CPU 耗时中位（5 次中位） | 墙钟中位 | A* 出堆数 | 退回局部规划 | YAW 动作 | visible 口径 margin < 0.5（其中会触发改向） |", "|---|---|---|---|---|---|---|---|",
              group_row("迟到步 · K=3 L=0.16", late_016), group_row("迟到步 · K=3 L=0.20", late_020), group_row("迟到步 · K=1（教师 + 学生影子标签）", late_k1),
              group_row("对照 A · K=3 随机不迟到步", ctrl_a), group_row("对照 B · K=3 步号 ≤ 3 的不迟到步", ctrl_b), group_row("对照 C · K=3 不迟到的 YAW 步", ctrl_c)]
    for k in CLASS_NAMES:
        lines.append(group_row(f"抽样 D · K=3 不迟到步 · {CLASS_NAMES[k][0]} 类", sel(f"cls{k}")))
    lines += ["", "### 迟到步对对照（CPU 耗时与出堆数的比值）", "",
              compare_line("K=3 L=0.16 全部迟到步 对 对照 A", late_016, ctrl_a),
              compare_line("K=3 L=0.16 步号 ≤ 3 的迟到步 对 对照 B", [m for m in late_016 if m["row"]["step"] <= EARLY_MAX_STEP], ctrl_b),
              compare_line("K=3 L=0.16 YAW 迟到步 对 对照 C", [m for m in late_016 if m["row"]["action"] in TURNS], ctrl_c),
              compare_line("K=3 L=0.16 F 类迟到步 对 N 类不迟到步（抽样 D）", [m for m in late_016 if m["klass"] == "F"], sel("clsN")), ""]
    allm = list(results.values())
    k3_late = late_016 + late_020
    rho = spearman([m["cpu_ms"] for m in allm], [m["counts"]["pops"] for m in allm])
    rho_lag = spearman([m["cpu_ms"] for m in k3_late], [m["row"]["lag"] for m in k3_late])
    tie_ratio = [m["tie_cpu_ms"] / m["cpu_ms"] for m in allm if m["cpu_ms"] > 0]
    tm = median_iqr(tie_ratio)
    lines += [f"CPU 耗时与 A* 出堆数的 Spearman 相关（全部 {len(allm)} 个计时位姿）：{rho:.2f}。迟到步里离线 CPU 耗时与实测 lag 的 Spearman 相关（K=3，{len(k3_late)} 步）：{rho_lag:.2f}。",
              f"legacy 路径下 plan_action_global_tiebreak 返回的 tie_margin 全为 None：{all(r['legacy_tie_margin'] is None for r in counted)}（{len(counted)} 个位姿）；"
              f"astar_path 每步调用次数的集合 {sorted({r['counts']['astar_calls'] for r in counted})}（legacy 下应只有 1）。",
              f"visible 口径补算的 lr_tie_margin_steps 是 legacy 一步 CPU 耗时的 {tm[0]:.1f} 倍（中位数；四分位 {tm[1]:.1f}–{tm[2]:.1f}）；第一轮没有跑它。", "",
              "## 4. 迟到步逐条（实测策略段墙钟推算 = (lag − 帧龄 0.114 s) ÷ 该集 action_rtf；离线 CPU / 墙钟 = 5 次中位数）", "",
              "| 臂 | seed | 步 | 动作 | x m | y m | 实测 lag s | 实测策略段墙钟推算 ms | 离线 CPU ms | 离线墙钟 ms | A* 出堆数 | 类 | visible margin | 会触发改向 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted([m for m in allm if "late" in m["tags"]], key=lambda m: (m["row"]["arm"], m["row"]["seed"], m["row"]["step"])):
        r = m["row"]
        real = (r["lag"] - sm.FRAME_AGE_S) / r["rtf"] * 1000.0 if r["rtf"] and r["lag"] is not None else float("nan")
        lines.append(f"| {r['arm']} | {r['seed']} | {r['step']} | {ACTION_NAMES.get(r['action'], r['action'])} | {r['x']:.2f} | {r['y']:.2f} | {r['lag']:.3f} | {real:.0f} | "
                     f"{m['cpu_ms']:.1f} | {m['wall_ms']:.1f} | {m['counts']['pops']} | {m['klass']} | {m['tie_margin']:.2f} | {'是' if m['tie_triggers'] else '否'} |")
    lines.append("")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "profile-late-steps.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(OUT, "profile-late-steps.json"), "w", encoding="utf-8") as f:
        json.dump({"teacher_sha256": teacher_sha, "repeats": repeats, "nr_throttled_delta": throttled, "timed": allm,
                   "counted": [{k: v for k, v in r.items()} for r in counted]}, f, ensure_ascii=False, indent=1)
    print(text)
    return 0


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", default=os.path.join(sm.D, "exp-action-latency-20261110"))
    ap.add_argument("--note", help="写在标题下的一行说明（生成时间、提交哈希）")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args(argv)
    return main(args.out, args.note, args.repeats)


if __name__ == "__main__":
    sys.exit(cli())
