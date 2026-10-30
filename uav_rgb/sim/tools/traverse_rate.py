#!/usr/bin/env python3
"""采集集穿越率：max_x >= 最右柱 x + margin 且无 collision 算成功。

T1/T5 的验收口径（教师是否真的穿过柱区），只读采集产物，不起 Gazebo。
用法：
  python3 sim/tools/traverse_rate.py --dataset <collection root> [--out traverse.json]
"""
import argparse
import glob
import json
import os
import sys


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_stats(ep_dir, margin):
    with open(os.path.join(ep_dir, "scene.json"), encoding="utf-8") as f:
        scene = json.load(f)
    # safe 出生集整集悬停，按设计就不会穿越；算进分母会把混合数据的上限压到
    # 1 - safe 比例，T1/T5 的 90% 线必然误 FAIL。
    is_safe = (scene.get("spawn") or {}).get("mode") == "safe"
    pillars = scene["pillars"]
    goal_x = max(float(p["x"]) for p in pillars) + margin
    actions = load_jsonl(os.path.join(ep_dir, "actions.jsonl"))
    oracle = load_jsonl(os.path.join(ep_dir, "oracle.jsonl"))
    xs = [float(o["pose_gt"]["x"]) for o in oracle
          if isinstance(o.get("pose_gt"), dict) and "x" in o["pose_gt"]]
    max_x = max(xs) if xs else None
    collision = any(bool(a.get("collision")) for a in actions)
    return {
        "episode": os.path.basename(ep_dir),
        "scene_id": ep_dir,
        "steps": len(actions),
        "collision": collision,
        "max_x": max_x,
        "goal_x": goal_x,
        "is_safe_spawn": is_safe,
        "success": bool(not collision and max_x is not None and max_x >= goal_x),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--goal-margin", type=float, default=1.0)
    p.add_argument("--min-rate", type=float, default=None,
                   help="低于该成功率时非零退出（必填；T1/T5 验收用 0.9，只看数字用 0）")
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.min_rate is None:
        p.error("--min-rate 必填：验收用 0.9；只看数字用 0")
    root = os.path.abspath(args.dataset)
    episodes = []
    for split in ("train", "val", "test"):
        for d in sorted(glob.glob(os.path.join(root, split, "*"))):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "episode_manifest.json")):
                try:
                    st = episode_stats(d, args.goal_margin)
                except (OSError, ValueError, KeyError) as e:
                    print(f"跳过 {d}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                st["split"] = split
                episodes.append(st)
    if not episodes:
        print("没有可读 episode", file=sys.stderr)
        return 1
    safe = [st for st in episodes if st["is_safe_spawn"]]
    episodes = [st for st in episodes if not st["is_safe_spawn"]]
    if not episodes:
        print("没有可评的穿越集（全是 safe 出生集）", file=sys.stderr)
        return 1
    by_split = {}
    for st in episodes:
        d = by_split.setdefault(st["split"], {"n": 0, "success": 0, "collision": 0})
        d["n"] += 1
        d["success"] += int(st["success"])
        d["collision"] += int(st["collision"])
    ok = sum(d["success"] for d in by_split.values())
    print(f"episodes={len(episodes)} success={ok} "
          f"rate={ok / len(episodes):.1%} goal_margin={args.goal_margin}"
          + (f"（safe 出生 {len(safe)} 集不计入）" if safe else ""))
    for split in ("train", "val", "test"):
        if split in by_split:
            d = by_split[split]
            print(f"  {split:5} n={d['n']:3d} success={d['success']:3d} "
                  f"rate={d['success'] / d['n']:.1%} collision={d['collision']}")
    for st in episodes:
        if not st["success"]:
            print(f"  FAIL {st['split']}/{st['episode']}: collision={st['collision']} "
                  f"max_x={st['max_x']} goal={st['goal_x']}")
    report = {"dataset": root, "goal_margin": args.goal_margin,
              "episodes": episodes, "safe_episodes": safe, "summary": by_split,
              "success_rate": ok / len(episodes)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}")
    return 0 if ok / len(episodes) >= args.min_rate else 1


if __name__ == "__main__":
    sys.exit(main())
