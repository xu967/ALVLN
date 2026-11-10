#!/usr/bin/env python3
"""exp-heading-guard-20261110（R1）汇总：护栏臂对同时段的无护栏对照臂，写 summary.md（随时可跑，缺的臂标「未完成」）。

运行：/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py
口径：success 取 eval.json 的 per_episode.success（sim09.v2.coast1s.corridor3m，已含 |y|<=3 m）。
对照臂在 ../exp-swa-calib-20261110（best-s*、calswa-s*：同 20 个 seed、同口径、护栏关）。
more_demos 无护栏的对照取 DAgger 流水线的 eval20-more_demos（旧代码、无走廊参数），这里按 oracle 轨迹补算 |y|<=3。
"""
import json
import math
import os
from math import comb

OUT = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(OUT)
SWA = os.path.join(D, "exp-swa-calib-20261110")
MD_EVAL = os.path.join(D, "exp-dagger-gz-20261109", "full-1109", "eval20-more_demos")
# 变体 -> 各训练种子的 eval 目录（None 占位表示该变体只有一个权重）
VARIANTS = {
    "best（无护栏）": [os.path.join(SWA, f"best-s{s}") for s in (0, 1, 2)],
    "best + 护栏": [os.path.join(OUT, f"guard2-s{s}") for s in (0, 1, 2)],
    "权重平均校准（无护栏）": [os.path.join(SWA, f"calswa-s{s}") for s in (0, 1, 2)],
    "权重平均校准 + 护栏": [os.path.join(OUT, f"guard2-calswa-s{s}") for s in (0, 1, 2)],
    "more_demos（无护栏，补算走廊）": [MD_EVAL],
    "more_demos + 护栏": [os.path.join(OUT, "guard2-moredemos")],
}
PAIRS = [("best + 护栏", "best（无护栏）"), ("权重平均校准 + 护栏", "权重平均校准（无护栏）"),
         ("more_demos + 护栏", "more_demos（无护栏，补算走廊）"), ("权重平均校准 + 护栏", "best + 护栏")]


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


def load_eval(edir):
    """-> {评测 seed: 行}；行里补 outcome / flips / guard_steps / steps。没有完整 eval.json 返回 None。"""
    path = os.path.join(edir, "eval.json")
    if not os.path.exists(path):
        return None
    e = json.load(open(path, encoding="utf-8"))
    if e.get("partial"):
        return None
    has_corridor = e.get("corridor_half_width") is not None
    rows = {}
    for x in e.get("per_episode", []):
        # episode_dir 是相对 eval 目录的路径（episodes/seed_N）；绝对路径时 join 原样返回
        ep = os.path.join(edir, x.get("episode_dir") or os.path.join("episodes", f"seed_{x['seed']}"))
        acts = read_jsonl(os.path.join(ep, "actions.jsonl"))
        seq = [a.get("action_executed") for a in acts]
        success, violation = bool(x.get("success")), bool(x.get("corridor_violation"))
        if not has_corridor:   # 旧代码的 eval：按 oracle 轨迹补算走廊
            ys = [abs(o["pose_gt"]["y"]) for o in read_jsonl(os.path.join(ep, "oracle.jsonl"))
                  if isinstance(o.get("pose_gt"), dict)]
            violation = bool(ys) and max(ys) > 3.0
            success = success and not violation
        outcome = ("success" if success else "collision" if x.get("collision")
                   else "corridor" if violation else "not_reached")
        rows[int(x["seed"])] = {
            "success": success, "outcome": outcome, "steps": len(seq),
            "flips": sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {2, 3}),
            "guard_steps": sum(1 for a in acts if a.get("executed_by") == "heading_guard"),
        }
    return rows


def main():
    data = {}
    for name, dirs in VARIANTS.items():
        rows, per_seed = {}, []
        for i, d in enumerate(dirs):
            got = load_eval(d)
            per_seed.append(None if got is None else sum(1 for r in got.values() if r["success"]))
            for es, r in (got or {}).items():
                rows[(i, es)] = r
        data[name] = (rows, per_seed)
    lines = ["# exp-heading-guard-20261110（R1）：评测端航向护栏对无护栏", "",
             "口径 `sim09.v2.coast1s.corridor3m`；20 个冻结评测 seed；hold 1.0，observation，cf_sitl lib31，E1 MINI，80 步上限。"
             "护栏：已执行左转数减右转数的绝对值 ≥ 2 时强制朝回转一步。预注册见 `run.sh` 文件头。", "",
             "| 变体 | 集数 | 走廊内成功 | Wilson 95% | 碰撞 | 越界 | 未到线 | 各训练种子成功数 | 每集 L↔R 换向 | 护栏改写的步占比 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, (rows, per_seed) in data.items():
        if not rows:
            lines.append(f"| {name} | 未完成 | | | | | | | | |")
            continue
        cnt = {k: sum(1 for r in rows.values() if r["outcome"] == k) for k in ("success", "collision", "corridor", "not_reached")}
        lo, hi = wilson(cnt["success"], len(rows))
        steps = max(sum(r["steps"] for r in rows.values()), 1)
        lines.append(f"| {name} | {len(rows)} | {cnt['success']} | [{lo:.2f}, {hi:.2f}] | {cnt['collision']} | {cnt['corridor']} | "
                     f"{cnt['not_reached']} | {' / '.join('-' if k is None else str(k) for k in per_seed)} | "
                     f"{sum(r['flips'] for r in rows.values()) / len(rows):.1f} | "
                     f"{sum(r['guard_steps'] for r in rows.values()) / steps:.3f} |")
    lines += ["", "## 配对检验（精确 McNemar；b = 前者成后者败，c = 前者败后者成）", "",
              "| A 对 B | 共同配对 | b | c | p |", "| --- | --- | --- | --- | --- |"]
    for a, b_ in PAIRS:
        ra, rb = data[a][0], data[b_][0]
        keys = sorted(set(ra) & set(rb))
        if not keys:
            lines.append(f"| {a} 对 {b_} | 0 | | | |")
            continue
        b = sum(1 for k in keys if ra[k]["success"] and not rb[k]["success"])
        c = sum(1 for k in keys if not ra[k]["success"] and rb[k]["success"])
        lines.append(f"| {a} 对 {b_} | {len(keys)} | {b} | {c} | {mcnemar_exact(b, c):.3f} |")
    lines += ["", "说明：三个训练种子共用 20 个评测几何，60 个配对不独立，p 值只作参考；结论看方向与失败类型的变化。"
              "同权重同 seed 重跑的逐集不一致率约 24%（data-algo 报告 B.2），单集翻转不作结论。"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
