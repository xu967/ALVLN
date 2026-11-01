#!/usr/bin/env python3
"""只保留“决定性帧”：同一状态把左右关系反过来后 teacher 动作会变的步。

目的：L1 训练集里大部分样本的动作由避障决定，跟指令无关，文本信号被稀释，
FiLM 也学不会。这里给每个决定性状态生成两条样本（原关系标签 + 反向关系标签），
训练集里这两条**唯一**的区别就是关系词，模型想同时拟合只能读文本。
图像用硬链接，不重采。

用法：
  python3 sim/tools/make_decisive.py --src <采集根> --out <新根> --splits train
"""
import argparse
import hashlib
import json
import os
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import instruction as ins  # noqa: E402
import teacher  # noqa: E402
from make_counterfactual import alt_target, dump_json, load_json, relabel  # noqa: E402


def write_episode(out_dir, src_dir, man, scene, rows, oracle, keep, labels, instruction,
                  relation, waypoint, out_sid):
    os.makedirs(out_dir, exist_ok=True)
    for i in keep:
        name = rows[i].get("frame") or f"rgb_{i:05d}.png"
        dst = os.path.join(out_dir, name)
        if not os.path.exists(dst):
            os.link(os.path.join(src_dir, name), dst)
    with open(os.path.join(out_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
        for i, act in zip(keep, labels):
            row = dict(rows[i])
            row["action_expert"] = act
            row["action_executed"] = act
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "oracle.jsonl"), "w", encoding="utf-8") as f:
        for i in keep:
            if i < len(oracle):
                f.write(json.dumps(oracle[i], ensure_ascii=False) + "\n")
    scene2 = dict(scene)
    scene2.update({"relation": relation, "instruction": instruction,
                   "waypoint": waypoint})
    scene_path = os.path.join(out_dir, "scene.json")
    dump_json(scene_path, scene2)
    man2 = dict(man)
    man2.update({
        "episode_id": out_sid, "scene_id": out_sid, "seed": out_sid,
        "instruction": instruction, "relation": relation,
        "target_color": scene.get("target_color"),
        "waypoint": waypoint, "num_steps": len(keep),
        "requested_steps": len(keep), "stopped_on_goal": False, "error": None,
        "decisive_of": man.get("episode_id"),
        "scene_meta_sha256": hashlib.sha256(open(scene_path, "rb").read()).hexdigest(),
    })
    dump_json(os.path.join(out_dir, "episode_manifest.json"), man2)


def make_one(ep_dir, out_root, split):
    man = load_json(os.path.join(ep_dir, "episode_manifest.json"))
    scene = load_json(os.path.join(ep_dir, "scene.json"))
    if man.get("planner") != "astar" or man.get("stop_radius") is None:
        return None
    alt = alt_target(scene)
    if alt is None:
        return None
    alt_relation, alt_wp = alt
    original_color = scene.get("target_color")
    original_relation = scene["relation"]
    stop = float(man["stop_radius"])
    rows = [json.loads(line) for line in open(os.path.join(ep_dir, "actions.jsonl"),
                                              encoding="utf-8") if line.strip()]
    oracle = [json.loads(line) for line in open(os.path.join(ep_dir, "oracle.jsonl"),
                                                encoding="utf-8") if line.strip()]
    if not rows:
        return None
    if any(r.get("pose_before") is None or r.get("yaw_before") is None for r in rows):
        return None
    alt_actions, _, _ = relabel(rows, scene["pillars"], alt_wp, stop)
    keep = [i for i, (a, b) in enumerate(zip([r["action_expert"] for r in rows], alt_actions))
            if a != b]
    if not keep:
        return None
    sid = int(man["episode_id"])
    # 固定高位段，避免 scene_id 与未来 base-seed 撞号。乘 2 是必须的：相邻集的
    # sid 只差 1，用 base+1/base+2 会让第 i 集的 +2 撞上第 i+1 集的 +1，
    # 后写的再把前一对的**反事实那一半**覆盖掉（v3 实测 478 个目录里 324 个撞号）。
    base = 9_000_000 + 2 * sid
    write_episode(os.path.join(out_root, split, str(base + 1)), ep_dir, man, scene, rows,
                  oracle, keep, [rows[i]["action_expert"] for i in keep],
                  scene.get("instruction")
                  or ins.make_instruction(original_color, original_relation),
                  original_relation, scene.get("waypoint"), base + 1)
    write_episode(os.path.join(out_root, split, str(base + 2)), ep_dir, man, scene, rows,
                  oracle, keep, [alt_actions[i] for i in keep],
                  ins.make_instruction(original_color, alt_relation), alt_relation,
                  alt_wp, base + 2)
    return {"source": ep_dir, "steps": len(keep), "episodes": 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    total_eps = total_steps = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        root = os.path.join(args.src, split)
        if not os.path.isdir(root):
            continue
        n = 0
        for name in sorted(os.listdir(root)):
            ep_dir = os.path.join(root, name)
            if not os.path.isdir(ep_dir):
                continue
            rec = make_one(ep_dir, args.out, split)
            if rec is None:
                continue
            n += 1
            total_eps += rec["episodes"]
            total_steps += rec["steps"] * rec["episodes"]
            if args.limit and n >= args.limit:
                break
        print(f"{split}: 决定性源集 {n}")
    dump_json(os.path.join(args.out, "decisive_manifest.json"), {
        "schema": "uavrgb.decisive.v1", "source": os.path.abspath(args.src),
        "splits": args.splits, "episodes": total_eps, "steps": total_steps,
    })
    print(f"episodes={total_eps} steps={total_steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
