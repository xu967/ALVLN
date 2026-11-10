#!/usr/bin/env python3
"""exp-rule-oracle-20261110 汇总：真值列距离 + 规则 v0 的闭环上限，写 summary.md（随时可跑，缺的臂标「进行中」）。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py   （在实验目录里；run.sh 会让它拷一份过去）
口径：success 取 eval.json 的 per_episode.success（sim09.v2.coast1s.corridor3m，已含 |y|<=3 m）；分母 = 已完成集数，完成后即请求集数 20。
撞柱分析：被撞柱 = 撞击步结束位姿离哪根柱子最近；方位 = 撞击步开始前（pose_before、yaw_before）看被撞柱中心的角度，正为左；
  柱子由 scene_gen.sample_pillars(seed) 重建，并用 split_manifest.geometry_gid 与每集 manifest 的 geometry_gid 对账，对不上的集不做撞柱分析。
参照行：教师 A*（11-09 的 v1 口径，没有走廊判据，也没有逐集文件）与学生最好配置（calswa + 航向护栏，同口径，3 个训练种子各 20 集）。
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
TOOLS = os.environ.get("RULE_TOOLS", f"{D}/tmp/phase12-20261110/ruleoracle/code-18f2ec0/sim/tools")  # 冻结代码，重建柱子用
REPS = 3                                    # 每个视场的重复次数，与 run.sh 一致
FOV = {"rule60": 60.0, "rule81": 81.0}      # 视锥全角（度）：判被撞柱中心是否在视锥内
SUCCESS_PRED, SUCCESS_FALSIFY, COLLISION_PRED = 54, 48, 3   # 预注册线（run.sh 文件头）
EPISODES = 20                               # 每臂请求集数
STUDENT = [f"{D}/exp-heading-guard-20261110/guard2-calswa-s{s}" for s in (0, 1, 2)]
TEACHER = f"{D}/exp-teacher-cost-20261109/cl1/eval-teacher-legacy"
ENTRIES = ("R1", "R1-safe", "R1-fwd", "R2-sum", "R2-last", "R2-n", "R2-left", "R3")
ACT = "SFLR"                                # 动作编号 0..3 对应 SAFE_NEUTRAL / FORWARD_SLOW / YAW_LEFT / YAW_RIGHT
# ----------------------------------------------------------------

sys.path.insert(0, TOOLS)
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402


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
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


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
    return {
        "step": ci, "pillar": (hit["x"], hit["y"]), "gap_m": dist_to_pillar(hit, after["x"], after["y"]),
        "dist": dist, "bearing": bearing,
        "own": cone_class(bearing, dist, fov_deg, half_extent), "c60": cone_class(bearing, dist, 60.0, half_extent),
        "dmin3": min(mins) if mins else None,
        "last3": " ".join(f"{r.get('rule_entry', '-')}:{ACT[r['action_executed']]}" for r in pre),
    }


def load_arm(edir, fov_deg=None):
    """-> {"rows": {seed: 行}, "partial": bool}；没有 eval.json 返回 None。fov_deg 给定时对撞柱集做撞柱分析。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
    e = json.load(open(path, encoding="utf-8"))
    rows = {}
    for x in e.get("per_episode", []):
        ep = os.path.join(edir, x.get("episode_dir") or os.path.join("episodes", f"seed_{x['seed']}"))
        acts = read_jsonl(os.path.join(ep, "actions.jsonl"))
        seq = [a.get("action_executed") for a in acts]
        turns = [a for a in seq if a in (2, 3)]
        runs, run = [], 0
        for a in seq:
            run = run + 1 if a == 0 else 0
            runs.append(run)
        success, violation = bool(x.get("success")), bool(x.get("corridor_violation"))
        outcome = ("success" if success else "collision" if x.get("collision")
                   else "corridor" if violation else "not_reached")
        row = {
            "success": success, "outcome": outcome, "steps": x.get("steps") or len(seq),
            "spl": float(x.get("spl") or 0.0),
            "acts": Counter({int(k): v for k, v in (x.get("actions") or {}).items()}),
            "flips_adj": sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {2, 3}),
            "flips_turn": sum(1 for a, b in zip(turns, turns[1:]) if a != b),
            "safe_run": max(runs, default=0), "has_seq": bool(seq),
            "entries": Counter(a.get("rule_entry") for a in acts if a.get("rule_entry")),
            "tail": " ".join(f"{a.get('rule_entry', '-')}:{ACT[a['action_executed']]}" for a in acts[-8:]),
            "max_x": x.get("max_x"), "y_max": x.get("y_max_m"), "info": None, "note": "",
        }
        if fov_deg and outcome == "collision" and acts:
            seed = int(x["seed"])
            pillars = sg.sample_pillars(seed)[0]
            man = {}
            mpath = os.path.join(ep, "episode_manifest.json")
            if os.path.exists(mpath):
                man = json.load(open(mpath, encoding="utf-8"))
            if sm.geometry_gid(pillars) != (man.get("geometry_gid") or x.get("geometry_gid")):
                row["note"] = "几何对不上，未做撞柱分析"
            else:
                row["info"] = collision_info(acts, read_jsonl(os.path.join(ep, "oracle.jsonl")), pillars, fov_deg)
        rows[int(x["seed"])] = row
    return {"rows": rows, "partial": bool(e.get("partial")) or len(rows) < e.get("episodes_requested", EPISODES)}


def merge(arms):
    """各重复合并成 {(重复序号, seed): 行}；各重复的完成情况。"""
    rows, done = {}, []
    for i, arm in enumerate(arms):
        done.append(None if arm is None else len(arm["rows"]))
        for seed, r in (arm or {"rows": {}})["rows"].items():
            rows[(i, seed)] = r
    return rows, done


def pct(k, n):
    return f"{k / n:.3f}" if n else "-"


def main():
    variants = {}
    for name in ("rule60", "rule81"):
        arms = [load_arm(os.path.join(OUT, f"{name}-rep{i}"), FOV[name]) for i in range(REPS)]
        variants[name] = merge(arms)
    student = merge([load_arm(d) for d in STUDENT])
    teacher_arm = load_arm(TEACHER)
    teacher = merge([teacher_arm])
    table = {"规则 60°（rule60）": variants["rule60"], "规则 81°（rule81）": variants["rule81"],
             "参照：学生 calswa + 护栏": student, "参照：教师 A*（v1 口径）": teacher}

    L = ["# exp-rule-oracle-20261110：真值列距离 + 规则 v0 的闭环上限", "",
         "口径 `sim09.v2.coast1s.corridor3m`（教师参照行是 11-09 的 `sim09.v1.coast1s`，无走廊判据，不可与其它行混报）；"
         "20 个冻结评测 seed；hold 1.0，observation，cf_sitl lib31，E1 MINI，80 步上限，到成功线即停。"
         "规则参数 `RULE_V0` 见 `eval.json` 顶层 `rule_params`，预注册见 `run.sh` 文件头。", "",
         "## 1 结果", "",
         "| 变体 | 集数 | 走廊内成功 | Wilson 95% | 撞柱 | 越界 | 未到线 | 各重复（或各训练种子）成功数 | SPL | 平均步数 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, done) in table.items():
        if not rows:
            L.append(f"| {name} | 进行中或缺失 | | | | | | | | |")
            continue
        n = len(rows)
        cnt = {k: sum(1 for r in rows.values() if r["outcome"] == k) for k in ("success", "collision", "corridor", "not_reached")}
        lo, hi = wilson(cnt["success"], n)
        per = [("-" if d is None else sum(1 for (i, _), r in rows.items() if i == j and r["success"])) for j, d in enumerate(done)]
        L.append(f"| {name} | {n} | {cnt['success']} | [{lo:.2f}, {hi:.2f}] | {cnt['collision']} | {cnt['corridor']} | "
                 f"{cnt['not_reached']} | {' / '.join(str(p) for p in per)} | "
                 f"{sum(r['spl'] for r in rows.values()) / n:.3f} | {sum(r['steps'] for r in rows.values()) / n:.1f} |")
    L += ["", "SPL：失败集按 0，分母 = 已完成集数。教师参照行来自旧口径：过线后继续飞满 80 步，所以平均步数是 80，SPL 也按旧定义。", "",
          "## 2 预注册线对照", ""]
    for name, key in (("rule60", "规则 60°（rule60）"),):
        rows = table[key][0]
        n = len(rows)
        k = sum(1 for r in rows.values() if r["success"])
        c = sum(1 for r in rows.values() if r["outcome"] == "collision")
        if n < REPS * EPISODES:
            L.append(f"- {name}：进行中，已完成 {n}/{REPS * EPISODES} 集（成功 {k}，撞柱 {c}），60 集齐了再对照预注册线")
        else:
            verdict = ("达到预测线" if k >= SUCCESS_PRED else "低于证伪线" if k < SUCCESS_FALSIFY else "介于证伪线与预测线之间")
            lo, hi = wilson(k, n)
            L += [f"- {name} 走廊内成功 {k}/{n}（Wilson 95% [{lo:.2f}, {hi:.2f}]）：预测 >= {SUCCESS_PRED}/60，证伪 < {SUCCESS_FALSIFY}/60，结果{verdict}",
                  f"- {name} 撞柱 {c}/{n}：预测 <= {COLLISION_PRED}/60，结果{'达到' if c <= COLLISION_PRED else '超出'}预测线"]
    r60, r81 = variants["rule60"][0], variants["rule81"][0]
    keys = sorted(set(r60) & set(r81))
    if keys:
        b = sum(1 for k in keys if r81[k]["outcome"] == "collision" and r60[k]["outcome"] != "collision")
        c = sum(1 for k in keys if r60[k]["outcome"] == "collision" and r81[k]["outcome"] != "collision")
        n81 = sum(1 for k in keys if r81[k]["outcome"] == "collision")
        n60 = sum(1 for k in keys if r60[k]["outcome"] == "collision")
        L.append(f"- 次要：rule81 撞柱 {n81}，rule60 撞柱 {n60}（{len(keys)} 对配对集；b = 81° 撞 60° 不撞 = {b}，c = 60° 撞 81° 不撞 = {c}，"
                 f"精确 McNemar p = {mcnemar_exact(b, c):.3f}；预测 rule81 <= rule60）")
        bs = sum(1 for k in keys if r81[k]["success"] and not r60[k]["success"])
        cs = sum(1 for k in keys if r60[k]["success"] and not r81[k]["success"])
        L.append(f"- 走廊内成功：81° 成 60° 败 {bs}，60° 成 81° 败 {cs}，精确 McNemar p = {mcnemar_exact(bs, cs):.3f}")
    L += ["", "60 个配对不独立（20 个几何 × 3 次重复），p 值只作参考；单次 n=20 的差异不作结论。", "",
          "## 3 行为", "",
          "| 变体 | SAFE | FORWARD | LEFT | RIGHT | 每集 L↔R 相邻换向 | 每集转向方向变化 | 含 >=3 连续 SAFE 的集数 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, _) in table.items():
        if not rows:
            continue
        tot = Counter()
        for r in rows.values():
            tot.update(r["acts"])
        s = max(sum(tot.values()), 1)
        has_seq = all(r["has_seq"] for r in rows.values())
        L.append(f"| {name} | {pct(tot[0], s)} | {pct(tot[1], s)} | {pct(tot[2], s)} | {pct(tot[3], s)} | "
                 + (f"{sum(r['flips_adj'] for r in rows.values()) / len(rows):.2f} | {sum(r['flips_turn'] for r in rows.values()) / len(rows):.2f} | "
                    f"{sum(1 for r in rows.values() if r['safe_run'] >= 3)} |" if has_seq else "- | - | - |"))
    L += ["", "动作占比取 eval.json 的 per_episode.actions；换向与连续 SAFE 取 actions.jsonl（教师参照行没有逐集文件，记 -）。"
          "「相邻换向」= 相邻两步分别是 L、R；「转向方向变化」= 只看转向步，相邻两次转向方向不同。", "",
          "## 4 规则条目触发占比（占各变体全部步）", "",
          "| 变体 | " + " | ".join(ENTRIES) + " | 总步数 |", "| --- | " + " | ".join("---" for _ in ENTRIES) + " | --- |"]
    for name in ("规则 60°（rule60）", "规则 81°（rule81）"):
        rows = table[name][0]
        if not rows:
            continue
        tot = Counter()
        for r in rows.values():
            tot.update(r["entries"])
        s = max(sum(tot.values()), 1)
        L.append(f"| {name} | " + " | ".join(pct(tot[e], s) for e in ENTRIES) + f" | {sum(tot.values())} |")
    L += ["", "R1 = 回转，R1-safe = 回转侧贴障时不转且前方受阻（SAFE），R1-fwd = 回转侧贴障时不转且前方不受阻（FORWARD），"
          "R2-sum / R2-last / R2-n / R2-left = 受阻转向的依据（两半之和 / 平局沿用上次 / 使 n 归零 / 固定朝左），R3 = 前进。", "",
          "## 5 撞柱集逐集", ""]
    for name in ("rule60", "rule81"):
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
          "## 6 其余失败集（越界、未到线），末 8 步（条目:动作）", "",
          "| 变体 | 重复 | seed | 结果 | 步数 | max_x | y 向最大偏离 m | 末 8 步 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    n_other = 0
    for name in ("rule60", "rule81"):
        for (i, seed), r in sorted(variants[name][0].items()):
            if r["outcome"] in ("corridor", "not_reached"):
                n_other += 1
                mx = "-" if r["max_x"] is None else f"{r['max_x']:.2f}"
                ym = "-" if r["y_max"] is None else f"{r['y_max']:.2f}"
                L.append(f"| {name} | {i} | {seed} | {r['outcome']} | {r['steps']} | {mx} | {ym} | {r['tail']} |")
    if not n_other:
        L.append("| 无 | | | | | | | |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
