#!/usr/bin/env python3
"""失败集形态的描述性统计（只读各集日志；不改任何文件；分类阈值不属于预注册）。

用法：python3 failure_analysis.py [OUT]   结果打印成 markdown。

每个没到点的集（success_stop = false）取：
  结局        撞在前（collision）或超时；撞在前再区分出生即碰撞（collision_step == 0）
  末段净位移    最后 30 步第一步与最后一步的位置距离（m）
  目标点不变的末段  rule.jsonl 的 target 在最后一段里不变的步数，以及这一段里飞机离目标点最近的距离（m）
  目标点跳变    相邻两步 err_deg 的绕圈差 > 100° 的次数（飞机自己一步最多转约 32°，这样的变化来自飞机与目标点的相对位置）
  全堵占比     rule.jsonl 里 clear == [0,0,0] 的步数占比（规则的三条候选通道在 d_look 内都有障碍）
  转向步占比    action_executed 为左转或右转的步数占比
超时集按下面的口径分三类（先判前一类）：
  目标点停滞   目标点不变的末段 >= 10 步，且这一段里飞机到过离目标点 <= 0.25 m 的位置（飞机到了目标点，目标点没有前进）
  末段不动     末 30 步净位移 < 1.0 m
  仍在位移     其余
"""
import collections
import glob
import json
import math
import os
import sys

D = os.environ.get("D", "/root/rivermind-data/uav_rgb")
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{D}/exp-rule-v2-3dgs-20261111"
ARMS = ("rule-occ+route", "rule2-occ+route", "rule2-depth+route")
MAIN = "rule2-occ+route"
CLASSES = ("holdout", "formal40")
LAST, JUMP_DEG, STILL_M = 30, 100.0, 1.0
STALL_RUN, STALL_NEAR_M = 10, 0.25
LEFT, RIGHT = 2, 3
KINDS = ("出生即碰撞", "撞在前（出生后）", "超时·目标点停滞", "超时·末段不动", "超时·仍在位移")


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def features(d, m):
    acts, rule = jsonl(f"{d}/actions.jsonl"), jsonl(f"{d}/rule.jsonl")
    err = [r["err_deg"] for r in rule]
    pos = [(a["pose_before"]["x"], a["pose_before"]["y"]) for a in acts]
    n = len(rule)
    tg = [tuple(r["target"][:2]) for r in rule]
    k = n - 1
    while k > 0 and tg[k - 1] == tg[-1]:
        k -= 1
    tail = pos[-LAST:]
    return {
        "net": math.dist(tail[0], tail[-1]),
        "path": sum(math.dist(tail[i], tail[i + 1]) for i in range(len(tail) - 1)),
        "run": n - k,
        "tgt_min": min(math.dist(pos[i], tg[-1]) for i in range(k, n)),
        "jumps": sum(1 for i in range(n - 1) if abs(wrap(err[i + 1] - err[i])) > JUMP_DEG),
        "allblock": sum(1 for r in rule if r.get("clear") == [0, 0, 0]) / n,
        "turn": sum(1 for a in acts if a["action_executed"] in (LEFT, RIGHT)) / len(acts),
    }


def classify(m, f):
    if m["collision"]:
        return "出生即碰撞" if m["collision_step"] == 0 else "撞在前（出生后）"
    if f["run"] >= STALL_RUN and f["tgt_min"] <= STALL_NEAR_M:
        return "超时·目标点停滞"
    return "超时·末段不动" if f["net"] < STILL_M else "超时·仍在位移"


def collect(cls, arm):
    fails, succ = [], []
    for p in sorted(glob.glob(f"{OUT}/{cls}/{arm}/*/r*/ep_*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        f_ = features(os.path.dirname(p), m)
        (succ if m["success_stop"] else fails).append((m, f_, classify(m, f_) if not m["success_stop"] else None))
    return fails, succ


def main():
    mean = lambda xs: f"{sum(xs) / len(xs):.3f}" if xs else "-"
    print(f"## 失败集分类（描述性；目标点停滞：末段目标点不变 >= {STALL_RUN} 步且飞机到过离它 <= {STALL_NEAR_M} m；末段不动：末 {LAST} 步净位移 < {STILL_M} m）\n")
    print("| 类别 | 臂 | 失败集 | " + " | ".join(KINDS) + " | 涉及路线数 | 超时集转向步占比均值 | 成功集转向步占比均值 |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in KINDS) + " | --- | --- | --- |")
    data = {}
    for cls in CLASSES:
        for arm in ARMS:
            fails, succ = collect(cls, arm)
            data[cls, arm] = (fails, succ)
            cnt = collections.Counter(k for _, _, k in fails)
            routes = {(m["scene"], m["route"]) for m, _, _ in fails}
            to_turn = [f["turn"] for m, f, _ in fails if m["timeout"]]
            print(f"| {cls} | `{arm}` | {len(fails)} | " + " | ".join(str(cnt[k]) for k in KINDS)
                  + f" | {len(routes)} | {mean(to_turn)} | {mean([f['turn'] for _, f, _ in succ])} |")

    print(f"\n## 主臂 `{MAIN}` 的条目步数占比：成功集（到点那步为止）与超时集（整集）\n")
    print("| 类别 | 集合 | 集数 | 步数 | V2-fwd | V2-back | V2-seek | V2-stuck |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for cls in CLASSES:
        fails, succ = data[cls, MAIN]
        for name, items in (("成功集", succ), ("超时集", [x for x in fails if x[0]["timeout"]])):
            tot, steps = collections.Counter(), 0
            for m, _f, _k in items:
                d = f"{OUT}/{cls}/{MAIN}/{m['scene']}/r{m['route']}/ep_{m['episode']:05d}"
                rows = jsonl(f"{d}/rule.jsonl")
                if m["success_stop"]:
                    rows = rows[: m["reach_step"] + 1]
                steps += len(rows)
                for r in rows:
                    tot[r["entry"][3:].rstrip("0123456789")] += 1
            print(f"| {cls} | {name} | {len(items)} | {steps} | " + " | ".join(f"{tot[k] / steps:.3f}" for k in ("fwd", "back", "seek", "stuck")) + " |")

    for cls in CLASSES:
        print(f"\n## {cls} 主臂逐集\n")
        print("| 场景 | 路线 | seed | A* m | 步数 | 结局 | 类别 | 最近到终点 m | 末刻到终点 m | 末 30 步净位移 m | 末 30 步路径长度 m | 目标点不变的末段步数 | 该段内离目标点最近 m | 目标点跳变 | 全堵占比 | 转向步占比 |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        routes = json.load(open(f"{OUT}/routes/routes_{cls}.json"))
        astar = {(s, r["route"]): r.get("astar_path_m") for s, rs in routes["scenes"].items() for r in rs}
        for m, f, kind in data[cls, MAIN][0]:
            print(f"| {m['scene']} | {m['route']} | {m['seed']} | {astar[m['scene'], m['route']]:.1f} | {m['steps']} | {'撞' if m['collision'] else '超时'} | {kind} "
                  f"| {m['min_wp_dist_m']:.2f} | {m['final_wp_dist_m']:.2f} | {f['net']:.2f} | {f['path']:.2f} | {f['run']} | {f['tgt_min']:.2f} | {f['jumps']} | {f['allblock']:.2f} | {f['turn']:.2f} |")
    for cls in CLASSES:
        by_scene = collections.Counter(m["scene"] for m, _, _ in data[cls, MAIN][0])
        print(f"\n{cls} 主臂失败集按场景：" + "，".join(f"{s} {n}" for s, n in sorted(by_scene.items())))


if __name__ == "__main__":
    main()
