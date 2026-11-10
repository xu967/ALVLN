#!/usr/bin/env python3
"""exp-rule-oracle-v1-20261110 汇总：真值列距离 + 规则 v1（短期障碍记忆）的闭环上限，写 summary.md（随时可跑，缺的臂标「进行中」）。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py   （在实验目录里；run.sh 会拷一份过去）
口径：success 取 eval.json 的 per_episode.success（sim09.v2.coast1s.corridor3m，已含 |y|<=3 m）；分母 = 请求集数（20 个冻结 seed，缺的集算没通过）。
撞柱分析：被撞柱 = 撞击步结束位姿离哪根柱子最近；方位 = 撞击步开始前（pose_before、yaw_before）看被撞柱中心的角度，正为左；
  柱子由 scene_gen.sample_pillars(seed) 重建，并用 split_manifest.geometry_gid 与每集 manifest 的 geometry_gid 对账，对不上的集不做撞柱分析。
n（净转向步数）= 到该步为止已执行 YAW_LEFT 数减 YAW_RIGHT 数，从 actions.jsonl 的 action_executed 逐步累计（与 collect.py 的 net_yaw 同义）。
长课程逐块存活：算法同 experiments/2026-11-10-long-course/summarize.py 的 block_outcomes（逐块过线 = 首次 pose_gt.x >= block_goal_x[k]，
  且到这一步为止无接触碰撞、|y| 全程 <= 走廊半宽；失败归属 = 第一个坏事件发生时正在穿越的那一块），这里复制了函数体，不 import 别的实验目录。
对照：v0（exp-rule-oracle-20261110 的 rule60 三次重复）与学生最好配置（calswa + 航向护栏 2，exp-heading-guard-20261110 的 3 个训练种子）。
  配对键 = (重复序号 或 训练种子序号, 评测 seed)；三次重复之间的序号并无对应关系，只是同 seed 的三次独立运行，所以配对 p 值只作参考。
"""
import json
import math
import os
import sys
from collections import Counter
from math import comb

OUT = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(OUT)

# ---------------------------------------------------------------- 配置
TOOLS = os.environ.get("RULE_TOOLS", "/root/claude-code-backup-20261028/code/wt/p12-rulev1/sim/tools")   # 冻结代码，重建柱子用
REPS = 3                                    # 每个视场的重复次数，与 run.sh 一致
FOV = {"rule60v1": 60.0, "rule81v1": 81.0}  # 视锥全角（度）：判被撞柱中心是否在视锥内
SUCCESS_PRED, SUCCESS_FALSIFY, COLLISION_PRED = 54, 48, 3   # 预注册线（run.sh 文件头）：rule60v1 的 60 集
COURSE_PRED, COURSE_INDEP = 16, 0.80        # 长课程 K=3 整条课程成功 >= 16/20；每块 0.93 的独立模型给 0.93^3 = 0.80
EPISODES = 20                               # 每臂请求集数
V0_ARMS = [f"{D}/exp-rule-oracle-20261110/rule60-rep{i}" for i in range(3)]
STUDENT = [f"{D}/exp-heading-guard-20261110/guard2-calswa-s{s}" for s in (0, 1, 2)]
TEACHER = f"{D}/exp-teacher-cost-20261109/cl1/eval-teacher-legacy"
ENTRIES = ("V1-back", "V1-fwd", "V1-more", "V1-side", "V1-last", "V1-left", "V1-cap")
V0_ENTRIES = ("R1", "R1-safe", "R1-fwd", "R2-sum", "R2-last", "R2-n", "R2-left", "R3")
ACT = "SFLR"                                # 动作编号 0..3 对应 SAFE_NEUTRAL / FORWARD_SLOW / YAW_LEFT / YAW_RIGHT
TAIL = 10                                   # 失败集逐步列出的末尾步数
BLOCK_PITCH_M, FIRST_GOAL_X, DEFAULT_HALF_WIDTH_M = 5.0, 6.0, 3.0   # 同 scene_course
TAIL_STEPS, TAIL_LOW_Z_M = 10, 1.5          # 末 10 步最低高度 < 1.5 m 视为掉高（电池压降的症状），只作诊断
# ----------------------------------------------------------------

sys.path.insert(0, TOOLS)
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402

FAIL_NAMES = {"collision": "撞柱", "corridor": "越界", "not_reached": "未到线", "aborted": "采集中止"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def half_extents(p):
    """柱子水平截面的半宽 (hx, hy)：圆柱取半径，方柱的 size 是 [sx, sy, sz]（与 rule_policy._hit 同一读法）。"""
    if "radius" in p:
        return float(p["radius"]), float(p["radius"])
    hx, hy = (v / 2 for v in p.get("size", (0.4, 0.4))[:2])
    return hx, hy


def dist_to_pillar(p, x, y):
    """点到柱子（方柱的方块、圆柱的圆）表面的距离，在柱内为 0。"""
    if "radius" in p:
        return max(math.hypot(x - p["x"], y - p["y"]) - p["radius"], 0.0)
    hx, hy = half_extents(p)
    return math.hypot(max(abs(x - p["x"]) - hx, 0.0), max(abs(y - p["y"]) - hy, 0.0))


def cone_class(bearing, dist, fov_deg, half_extent):
    """被撞柱相对视锥：中心在内 / 只有边缘擦进视锥（按柱子半宽张角）/ 整根在外。"""
    half = fov_deg / 2
    if abs(bearing) <= half:
        return "心在内"
    return "仅边缘" if abs(bearing) <= half + math.degrees(math.atan2(half_extent, dist)) else "全在外"


def net_yaw_series(seq):
    """每步「执行后」的 n：累计 YAW_LEFT(2) − YAW_RIGHT(3)。"""
    n, out = 0, []
    for a in seq:
        n += (a == 2) - (a == 3)
        out.append(n)
    return out


def step_text(a, n_before):
    """一步的文本：条目:动作 n=决策时的 n，v1 另带三个 clear（左转一步 / 保持 / 右转一步，1 = 畅通）与记忆点数。"""
    clr = a.get("rule_clear")
    extra = "" if clr is None else f" clear={''.join('1' if c else '0' for c in clr)} mem={a.get('rule_mem_n')}"
    return f"{a.get('rule_entry', '-')}:{ACT[a['action_executed']]} n={n_before}{extra}"


def tail_steps(acts, k=TAIL):
    ns = net_yaw_series([a.get("action_executed") for a in acts])
    before = [0] + ns[:-1]
    return [step_text(a, before[i]) for i, a in enumerate(acts)][-k:]


def stall_class(acts):
    """没到线也没撞的集的停滞形态：看末 10 步。cap 摆动 = V1-more / V1-cap 占 >= 6 步；左右拉扯 = 相邻 L/R 换向 >= 4 次；悬停 = SAFE >= 5 步。"""
    last = acts[-TAIL:]
    ent = [a.get("rule_entry") for a in last]
    seq = [a.get("action_executed") for a in last]
    flips = sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {2, 3})
    if sum(1 for e in ent if e in ("V1-more", "V1-cap")) >= 6:
        return "cap 摆动"
    if flips >= 4:
        return "左右拉扯"
    if sum(1 for s in seq if s == 0) >= 5:
        return "悬停"
    return "其它"


def collision_info(rows, oracle, pillars, fov_deg):
    """撞击步（第一个 collision=True 的步，没有则最后一步）的被撞柱、方位、撞前 3 步（含撞击步）的最小列距离。"""
    ci = next((i for i, r in enumerate(rows) if r.get("collision")), len(rows) - 1)
    after = oracle[ci].get("pose_gt") if ci < len(oracle) else None
    after = after if isinstance(after, dict) else rows[ci]["pose_before"]
    hit = min(pillars, key=lambda p: dist_to_pillar(p, after["x"], after["y"]))
    pb, yaw = rows[ci]["pose_before"], rows[ci]["yaw_before"]
    dx, dy = hit["x"] - pb["x"], hit["y"] - pb["y"]
    dist = math.hypot(dx, dy)
    bearing = (math.degrees(math.atan2(dy, dx)) - yaw + 180.0) % 360.0 - 180.0
    half_extent = max(half_extents(hit))
    pre = rows[max(0, ci - 2): ci + 1]
    mins = [min(r["rule_cols"]) for r in pre if r.get("rule_cols")]
    return {"step": ci, "pillar": (hit["x"], hit["y"]), "gap_m": dist_to_pillar(hit, after["x"], after["y"]),
            "dist": dist, "bearing": bearing,
            "own": cone_class(bearing, dist, fov_deg, half_extent), "c60": cone_class(bearing, dist, 60.0, half_extent),
            "dmin3": min(mins) if mins else None,
            "last3": " ".join(f"{r.get('rule_entry', '-')}:{ACT[r['action_executed']]}" for r in pre)}


def block_outcomes(actions, oracle, goals, half_width=DEFAULT_HALF_WIDTH_M, aborted=False):
    """一集的逐块结果（复制自 long-course/summarize.py）。goals = 每块的过线 x（block_goal_x）。
    返回 {"pass": [bool]*K, "fail": None 或 (块序号从 0 起, "collision"|"corridor"|"not_reached"|"aborted"), "cross_step": [...]}。"""
    k_total = len(goals)
    passed, cross = [False] * k_total, [None] * k_total
    cur, fail = 0, None
    for i in range(max(len(actions), len(oracle))):
        a = actions[i] if i < len(actions) else {}
        o = oracle[i] if i < len(oracle) else {}
        pg = o.get("pose_gt") if isinstance(o, dict) else None
        has_pose = isinstance(pg, dict) and "x" in pg and "y" in pg
        collided = bool(a.get("collision")) and a.get("src") != "stuck"
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
    """步数与仿真秒：sim_s = 末个动作终点（含起飞前约 13.5 s，电池按它算）。"""
    t1 = oracle[-1].get("action_end_ns") if oracle else None
    zs = [float(o["pose_gt"]["z"]) for o in oracle[-TAIL_STEPS:] if isinstance(o.get("pose_gt"), dict) and "z" in o["pose_gt"]]
    return {"sim_s": None if t1 is None else t1 / 1e9, "z_tail_min": min(zs) if zs else None}


def load_arm(edir, fov_deg=None):
    """-> {"rows": {seed: 行}, "complete": bool, "requested": int, "failed": int, "blocks": K 或 None}；没有 eval.json 返回 None。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
    e = json.load(open(path, encoding="utf-8"))
    half = e.get("corridor_half_width")
    half = DEFAULT_HALF_WIDTH_M if half is None else float(half)
    k_blocks = e.get("blocks")
    rows = {}
    for x in e.get("per_episode", []):
        ep = os.path.join(edir, x.get("episode_dir") or os.path.join("episodes", f"seed_{x['seed']}"))
        acts = read_jsonl(os.path.join(ep, "actions.jsonl"))
        oracle = read_jsonl(os.path.join(ep, "oracle.jsonl"))
        seq = [a.get("action_executed") for a in acts]
        ns = net_yaw_series(seq)
        turns = [a for a in seq if a in (2, 3)]
        runs, run = [], 0
        rev_runs, rev = [], 0
        for a in seq:
            run = run + 1 if a == 0 else 0
            runs.append(run)
        for a, b in zip(seq, seq[1:]):
            rev = rev + 1 if {a, b} == {2, 3} else 0
            rev_runs.append(rev)
        success, violation = bool(x.get("success")), bool(x.get("corridor_violation"))
        outcome = ("success" if success else "collision" if x.get("collision") else "corridor" if violation else "not_reached")
        mems = [a["rule_mem_n"] for a in acts if a.get("rule_mem_n") is not None]
        row = {
            "success": success, "outcome": outcome, "steps": x.get("steps") or len(seq), "spl": float(x.get("spl") or 0.0),
            "acts": Counter({int(k): v for k, v in (x.get("actions") or {}).items()}),
            "flips_adj": sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {2, 3}),
            "flips_turn": sum(1 for a, b in zip(turns, turns[1:]) if a != b),
            "rev_run": max(rev_runs, default=0), "safe_run": max(runs, default=0), "has_seq": bool(seq),
            "max_abs_n": max((abs(v) for v in ns), default=0),
            "entries": Counter(a.get("rule_entry") for a in acts if a.get("rule_entry")),
            "mem_sum": sum(mems), "mem_cnt": len(mems),
            "tail": tail_steps(acts), "stall": stall_class(acts) if acts and outcome == "not_reached" else "",
            "max_x": x.get("max_x"), "y_max": x.get("y_max_m"), "info": None, "note": "",
        }
        goals = x.get("block_goal_x")
        if not goals and k_blocks and x.get("goal_x") is not None:
            goals = [float(x["goal_x"]) - BLOCK_PITCH_M * (k_blocks - 1 - k) for k in range(k_blocks)]
        if goals and acts:
            row["blocks"] = block_outcomes(acts, oracle, goals, half)
            row.update(episode_stats(acts, oracle))
        if fov_deg and not k_blocks and outcome == "collision" and acts:
            seed = int(x["seed"])
            pillars = sg.sample_pillars(seed)[0]
            man = {}
            mpath = os.path.join(ep, "episode_manifest.json")
            if os.path.exists(mpath):
                man = json.load(open(mpath, encoding="utf-8"))
            if sm.geometry_gid(pillars) != (man.get("geometry_gid") or x.get("geometry_gid")):
                row["note"] = "几何对不上，未做撞柱分析"
            else:
                row["info"] = collision_info(acts, oracle, pillars, fov_deg)
        rows[int(x["seed"])] = row
    # 采集失败的集（多为长课程里电池掉压后固件 FAILSAFE）不在 per_episode 里：按中止前的轨迹补一行，算没通过
    for f in e.get("failed", []):
        ep = os.path.join(edir, "episodes", f"seed_{f['seed']}")
        acts, oracle = read_jsonl(os.path.join(ep, "actions.jsonl")), read_jsonl(os.path.join(ep, "oracle.jsonl"))
        row = {"success": False, "outcome": "aborted", "steps": len(acts), "spl": 0.0, "acts": Counter(), "flips_adj": 0, "flips_turn": 0,
               "rev_run": 0, "safe_run": 0, "has_seq": bool(acts), "max_abs_n": 0, "entries": Counter(), "mem_sum": 0, "mem_cnt": 0,
               "tail": tail_steps(acts) if acts else [], "stall": "", "max_x": None, "y_max": None, "info": None,
               "note": " ".join(f.get("stderr_tail") or [f.get("error", "")])[:160]}
        if k_blocks and oracle:
            row["blocks"] = block_outcomes(acts, oracle, [FIRST_GOAL_X + BLOCK_PITCH_M * k for k in range(k_blocks)], half, aborted=True)
            row.update(episode_stats(acts, oracle))
        rows[int(f["seed"])] = row
    requested = e.get("episodes_requested") or EPISODES
    complete = bool(not e.get("partial") and (e.get("episodes_done") or 0) + (e.get("failed_count") or 0) == requested)
    return {"rows": rows, "complete": complete, "requested": requested, "failed": e.get("failed_count") or 0, "blocks": k_blocks}


def merge(arms):
    """各重复合并成 {(重复序号, seed): 行}；各重复已出结果的集数。"""
    rows, done = {}, []
    for i, arm in enumerate(arms):
        done.append(None if arm is None else len(arm["rows"]))
        for seed, r in (arm or {"rows": {}})["rows"].items():
            rows[(i, seed)] = r
    return rows, done


def pct(k, n):
    return f"{k / n:.3f}" if n else "-"


def count(rows, key):
    return sum(1 for r in rows.values() if r["outcome"] == key)


def main():
    variants = {}
    for name in ("rule60v1", "rule81v1"):
        variants[name] = merge([load_arm(os.path.join(OUT, f"{name}-rep{i}"), FOV[name]) for i in range(REPS)])
    course_arm = load_arm(os.path.join(OUT, "rule60v1-course3"), 60.0)
    v0 = merge([load_arm(d) for d in V0_ARMS])
    student = merge([load_arm(d) for d in STUDENT])
    teacher = merge([load_arm(TEACHER)])
    table = {"规则 v1 60°（rule60v1）": variants["rule60v1"], "规则 v1 81°（rule81v1）": variants["rule81v1"],
             "参照：规则 v0 60°（rule60，同口径）": v0, "参照：学生 calswa + 护栏": student, "参照：教师 A*（v1 口径）": teacher}
    r60 = variants["rule60v1"][0]
    r81 = variants["rule81v1"][0]

    L = ["# exp-rule-oracle-v1-20261110：真值列距离 + 规则 v1（短期障碍记忆）的闭环上限", "",
         "口径 `sim09.v2.coast1s.corridor3m`（教师参照行是 11-09 的 `sim09.v1.coast1s`，无走廊判据，不可与其它行混报）；"
         "20 个冻结评测 seed；hold 1.0，observation，cf_sitl lib31，E1 MINI，80 步上限（长课程 240 步），到成功线即停。"
         "规则参数 `RULE_V1` 见 `eval.json` 顶层 `rule_params`，预注册见 `run.sh` 文件头。", "",
         "## 1 结果", "",
         "| 变体 | 集数 | 走廊内成功 | Wilson 95% | 撞柱 | 越界 | 未到线 | 各重复（或各训练种子）成功数 | SPL | 平均步数 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, done) in table.items():
        if not rows:
            L.append(f"| {name} | 进行中或缺失 | | | | | | | | |")
            continue
        n = len(rows)
        cnt = {k: count(rows, k) for k in ("success", "collision", "corridor", "not_reached")}
        lo, hi = wilson(cnt["success"], n)
        per = [("-" if d is None else sum(1 for (i, _), r in rows.items() if i == j and r["success"])) for j, d in enumerate(done)]
        L.append(f"| {name} | {n} | {cnt['success']} | [{lo:.2f}, {hi:.2f}] | {cnt['collision']} | {cnt['corridor']} | "
                 f"{cnt['not_reached']} | {' / '.join(str(p) for p in per)} | "
                 f"{sum(r['spl'] for r in rows.values()) / n:.3f} | {sum(r['steps'] for r in rows.values()) / n:.1f} |")
    L += ["", "SPL：失败集按 0，分母 = 已完成集数。教师参照行来自旧口径：过线后继续飞满 80 步，所以平均步数是 80，SPL 也按旧定义。", "",
          "## 2 预注册线对照", ""]
    n60 = len(r60)
    k60 = sum(1 for r in r60.values() if r["success"])
    c60 = count(r60, "collision")
    if n60 < REPS * EPISODES:
        L.append(f"- rule60v1：进行中，已完成 {n60}/{REPS * EPISODES} 集（成功 {k60}，撞柱 {c60}），60 集齐了再对照预注册线")
    else:
        verdict = ("达到预测线" if k60 >= SUCCESS_PRED else "低于证伪线" if k60 < SUCCESS_FALSIFY else "介于证伪线与预测线之间")
        lo, hi = wilson(k60, n60)
        L += [f"- rule60v1 走廊内成功 {k60}/{n60}（Wilson 95% [{lo:.2f}, {hi:.2f}]）：预测 >= {SUCCESS_PRED}/60，证伪 < {SUCCESS_FALSIFY}/60，结果{verdict}",
              f"- rule60v1 撞柱 {c60}/{n60}：预测 <= {COLLISION_PRED}/60，结果{'达到' if c60 <= COLLISION_PRED else '超出'}预测线"]
    n81, c81 = len(r81), count(r81, "collision")
    if n81:
        L.append(f"- 次要：rule81v1 撞柱 {c81}/{n81}，rule60v1 撞柱 {c60}/{n60}（预测 rule81v1 <= rule60v1"
                 + ("" if n81 >= REPS * EPISODES and n60 >= REPS * EPISODES else "；至少一边未跑满，暂不对照") + "）")
    else:
        L.append("- 次要：rule81v1 尚无结果")
    if course_arm and course_arm["rows"]:
        crow = course_arm["rows"]
        kc = sum(1 for r in crow.values() if r["success"])
        lo, hi = wilson(kc, len(crow))
        L.append(f"- 长课程 K={course_arm['blocks']}：整条课程成功 {kc}/{len(crow)}（Wilson [{lo:.2f}, {hi:.2f}]；{'已跑完' if course_arm['complete'] else '进行中'}），"
                 f"预测 >= {COURSE_PRED}/20（每块 0.93 的独立模型给 {COURSE_INDEP:.2f}）")
    else:
        L.append("- 长课程 K=3：尚无结果")
    L += ["", "## 3 配对比较", "",
          "| 比较 | 配对集数 | A 成 B 败 | A 败 B 成 | 精确 McNemar p |", "| --- | --- | --- | --- | --- |"]

    def paired(label, a, b, key="success"):
        keys = sorted(set(a) & set(b))
        if not keys:
            L.append(f"| {label} | 0 | - | - | - |")
            return
        if key == "success":
            bb = sum(1 for k in keys if a[k]["success"] and not b[k]["success"])
            cc = sum(1 for k in keys if b[k]["success"] and not a[k]["success"])
        else:
            bb = sum(1 for k in keys if a[k]["outcome"] == "collision" and b[k]["outcome"] != "collision")
            cc = sum(1 for k in keys if b[k]["outcome"] == "collision" and a[k]["outcome"] != "collision")
        L.append(f"| {label} | {len(keys)} | {bb} | {cc} | {mcnemar_exact(bb, cc):.3f} |")

    paired("走廊内成功：rule60v1 对 v0 rule60（配对键 重复序号, seed）", r60, v0[0])
    paired("走廊内成功：rule60v1 对学生 calswa + 护栏（配对键 训练种子序号, seed）", r60, student[0])
    paired("走廊内成功：rule81v1 对 rule60v1", r81, r60)
    paired("撞柱（A 撞 B 不撞 / B 撞 A 不撞）：rule81v1 对 rule60v1", r81, r60, key="collision")
    L += ["", "三次重复的序号之间并无对应关系，60 对也不独立（20 个几何 × 3 次），p 值只作参考；单次 n=20 的差异不作结论。", "",
          "## 4 行为", "",
          "| 变体 | SAFE | FORWARD | LEFT | RIGHT | 每集 L↔R 相邻换向 | 每集转向方向变化 | 含 >=3 连续 SAFE 的集数 | 最长连续换向 | 每集最大 \\|n\\| 均值 | 最大 \\|n\\| 分布 0/1/2/3/>=4 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, _) in table.items():
        if not rows:
            continue
        tot = Counter()
        for r in rows.values():
            tot.update(r["acts"])
        s = max(sum(tot.values()), 1)
        has_seq = all(r["has_seq"] for r in rows.values())
        if has_seq:
            dist = Counter(min(r["max_abs_n"], 4) for r in rows.values())
            tail = (f"{sum(r['flips_adj'] for r in rows.values()) / len(rows):.2f} | {sum(r['flips_turn'] for r in rows.values()) / len(rows):.2f} | "
                    f"{sum(1 for r in rows.values() if r['safe_run'] >= 3)} | {max(r['rev_run'] for r in rows.values())} | "
                    f"{sum(r['max_abs_n'] for r in rows.values()) / len(rows):.2f} | {'/'.join(str(dist[i]) for i in range(5))} |")
        else:
            tail = "- | - | - | - | - | - |"
        L.append(f"| {name} | {pct(tot[0], s)} | {pct(tot[1], s)} | {pct(tot[2], s)} | {pct(tot[3], s)} | " + tail)
    L += ["", "动作占比取 eval.json 的 per_episode.actions；换向、连续 SAFE、n 取 actions.jsonl（教师参照行没有逐集文件，记 -）。"
          "「相邻换向」= 相邻两步分别是 L、R；「转向方向变化」= 只看转向步，相邻两次转向方向不同；「最长连续换向」= 一集内连续相邻换向的最大次数"
          "（预注册的离线检查用「少于 4」）。", "",
          "## 5 规则条目触发占比（占各变体全部步）与记忆点数", "",
          "| 变体 | " + " | ".join(ENTRIES) + " | 总步数 | 每步记忆点数均值 |", "| --- | " + " | ".join("---" for _ in ENTRIES) + " | --- | --- |"]
    for name in ("规则 v1 60°（rule60v1）", "规则 v1 81°（rule81v1）"):
        rows = table[name][0]
        if not rows:
            continue
        tot = Counter()
        for r in rows.values():
            tot.update(r["entries"])
        s = max(sum(tot.values()), 1)
        ms, mc = sum(r["mem_sum"] for r in rows.values()), sum(r["mem_cnt"] for r in rows.values())
        L.append(f"| {name} | " + " | ".join(pct(tot[e], s) for e in ENTRIES) + f" | {sum(tot.values())} | {ms / mc:.1f} |" if mc else
                 f"| {name} | " + " | ".join(pct(tot[e], s) for e in ENTRIES) + f" | {sum(tot.values())} | - |")
    L += ["", "V1-back = 转回一步的通道畅通就转回，V1-fwd = 当前航向通道畅通就前进，V1-more = 两个通道都被挡且 |n| < n_cap，沿 n 的符号再转一步，"
          "V1-side / V1-last / V1-left = n=0 且受阻时的转向依据（两半之和 / 平局沿用上次 / 固定朝左），V1-cap = |n| >= n_cap 且两个通道都被挡，转回。", "",
          "## 6 长课程逐块存活（rule60v1-course3）", ""]
    if not course_arm or not course_arm["rows"]:
        L.append("尚无结果。")
    else:
        rows, K = course_arm["rows"], course_arm["blocks"] or 3
        n = course_arm["requested"]
        L += [f"请求 {n} 集，已出结果 {len(rows)} 集（其中采集中止 {course_arm['failed']} 集，算没通过）；分母 = 请求集数。", "",
              "| k | 通过前 k 块的集数 S_k | Wilson 95% | 条件存活 S_k / S_(k-1) |", "| --- | --- | --- | --- |"]
        prev = n
        for k in range(K):
            ok = sum(1 for r in rows.values() if r.get("blocks") and r["blocks"]["pass"][k])
            lo, hi = wilson(ok, n)
            L.append(f"| {k + 1} | {ok}/{n} = {ok / n:.2f} | [{lo:.2f}, {hi:.2f}] | {ok / prev:.2f} |" if prev else f"| {k + 1} | {ok}/{n} | | - |")
            prev = ok
        fails = Counter()
        for r in rows.values():
            if r.get("blocks") and r["blocks"]["fail"]:
                fails[(r["blocks"]["fail"][0] + 1, r["blocks"]["fail"][1])] += 1
        L += ["", "失败归属（块序号, 类型）：" + ("；".join(f"第 {b} 块{FAIL_NAMES[t]} {c} 集" for (b, t), c in sorted(fails.items())) or "无"), "",
              "| seed | 结果 | 通过块数 | 步数 | sim_s（含起飞前约 13.5 s） | 末 10 步最低高度 m | 末 10 步 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for seed, r in sorted(rows.items()):
            if r["success"]:
                continue
            b = r.get("blocks")
            passed = sum(b["pass"]) if b else "-"
            z = "-" if r.get("z_tail_min") is None else f"{r['z_tail_min']:.2f}"
            ss = "-" if r.get("sim_s") is None else f"{r['sim_s']:.0f}"
            L.append(f"| {seed} | {FAIL_NAMES.get(r['outcome'], r['outcome'])} | {passed} | {r['steps']} | {ss} | {z}{' 掉高' if r.get('z_tail_min') is not None and r['z_tail_min'] < TAIL_LOW_Z_M else ''} | "
                     + "<br>".join(r["tail"]) + " |")
        L += ["", "逐块定义见本文件头。sim_s 超过约 115 s 或末 10 步掉高，说明电池压降介入，此时失败不能全算导航。"]
    L += ["", "## 7 撞柱集逐集（base 任务）", ""]
    for name in ("rule60v1", "rule81v1"):
        L += [f"### {name}", "", "| 重复 | seed | 撞击步 | 被撞柱 (x, y) | 距离 m | 方位 ° | 自身视锥 | 60° 视锥 | 撞前 3 步最小列距离 m | 撞前 3 步（条目:动作） |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        n_coll = 0
        for (i, seed), r in sorted(variants[name][0].items()):
            if r["outcome"] != "collision":
                continue
            n_coll += 1
            f = r["info"]
            if f is None:
                L.append(f"| {i} | {seed} | | | | | | | | {r['note'] or '无逐集文件'} |")
                continue
            dm = "-" if f["dmin3"] is None else f"{f['dmin3']:.2f}"
            L.append(f"| {i} | {seed} | {f['step']} | ({f['pillar'][0]:.2f}, {f['pillar'][1]:.2f}) | {f['dist']:.2f} | {f['bearing']:+.1f} | "
                     f"{f['own']} | {f['c60']} | {dm} | {f['last3']} |")
        if not n_coll:
            L.append("| 无撞柱集（或该变体尚无完成的集） | | | | | | | | | |")
        L.append("")
    L += ["被撞柱 = 撞击步结束位姿离哪根柱子最近；方位与距离以撞击步开始前的位姿和航向计，正为左；"
          "视锥判定：中心在视锥半角内为「心在内」，中心在外但按柱子半宽张角仍擦进视锥为「仅边缘」，其余「全在外」。", "",
          f"## 8 失败集（撞柱 / 越界 / 未到线）的末 {TAIL} 步", "",
          "每步 = 条目:动作 n=决策时的净转向步数 clear=左转一步/保持/右转一步三个通道（1 畅通 0 被挡） mem=记忆点数。未到线集另给停滞形态"
          "（cap 摆动 = 末 10 步里 V1-more / V1-cap >= 6 步；左右拉扯 = 相邻换向 >= 4 次；悬停 = SAFE >= 5 步）。", ""]
    n_fail = 0
    for name in ("rule60v1", "rule81v1"):
        for (i, seed), r in sorted(variants[name][0].items()):
            if r["outcome"] in ("collision", "corridor", "not_reached"):
                n_fail += 1
                mx = "-" if r["max_x"] is None else f"{r['max_x']:.2f}"
                ym = "-" if r["y_max"] is None else f"{r['y_max']:.2f}"
                L += [f"**{name} 重复 {i} seed {seed}**：{FAIL_NAMES[r['outcome']]}，{r['steps']} 步，max_x {mx}，y 向最大偏离 {ym}，"
                      f"最大 |n| {r['max_abs_n']}" + (f"，停滞形态：{r['stall']}" if r["stall"] else ""), "", "```"] + r["tail"] + ["```", ""]
    if not n_fail:
        L.append("无失败集（或尚无完成的集）。")
    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
