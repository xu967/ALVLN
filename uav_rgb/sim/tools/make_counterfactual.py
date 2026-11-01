#!/usr/bin/env python3
"""用 teacher 重算“左右关系反过来”的标签，复制出反事实 episode（不重采）。

背景：同一张图只配一个指令时，优化器可以完全忽略文本——v2（异色柱）实测
蓝/红换词只改变 2e-4 的概率、argmax 动作 0 次变化，离线 L≈I≈S。v3 起两根彩色柱
**同色**，图像里根本没有能区分二者的线索，唯一变量是关系词 left/right；
把关系反过来后 teacher 的动作会变，这些变化就是文本的监督信号。

做法：对每个 astar 集，取另一根同色柱的 waypoint，用
`teacher.plan_action_global(..., stop_at_waypoint=True)` 逐帧重算标签；只输出
至少有一帧标签变化的集。图像用硬链接，不复制像素。test split 默认不动，
保证评测分布不变。

用法：
  python3 sim/tools/make_counterfactual.py --src <采集根> --out <新根> --splits train
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import instruction as ins  # noqa: E402
import landmarks as lm  # noqa: E402
import teacher  # noqa: E402


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def alt_target(scene):
    """返回 (反过来后的关系词, 另一根同色柱的 waypoint)；不可用时返回 None。

    两根彩色柱同色，只能靠左右位置区分；目标柱 = relation 那一侧的柱。
    """
    rel = scene.get("relation")
    spawn = scene.get("spawn")
    colored = [p for p in scene.get("pillars") or [] if p.get("color")]
    if rel not in ins.RELATIONS or len(colored) != 2 or not isinstance(spawn, dict):
        return None
    # 目标柱与 assign_landmarks 的映射一致（v4：relation=left → 目标在右边）。
    others = [p for p in colored
              if (lm.lateral(p, spawn) > 0) == (rel == "left")]
    alt = "right" if rel == "left" else "left"
    return alt, lm.waypoint_for(others[0], scene["pillars"], spawn)


def relabel(rows, pillars, waypoint, stop_radius):
    """逐帧用 teacher 重算动作。缺位姿不能保留原标签（那会造出换文本不换标签的
    矛盾样本），调用方必须先丢掉这种集；这里直接报错。"""
    out, changed = [], 0
    for row in rows:
        a = row.get("action_expert")
        pb, yb = row.get("pose_before"), row.get("yaw_before")
        if pb is None or yb is None:
            raise ValueError("缺 pose_before/yaw_before，不能生成反事实标签")
        alt = teacher.plan_action_global(
            pb["x"], pb["y"], yb, pillars, waypoint=waypoint,
            stop_at_waypoint=True, stop_radius=stop_radius)
        if alt != a:
            changed += 1
        out.append(alt)
    return out, changed, 0


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def make_one(ep_dir, out_dir, new_sid):
    man = load_json(os.path.join(ep_dir, "episode_manifest.json"))
    scene = load_json(os.path.join(ep_dir, "scene.json"))
    if man.get("planner") != "astar" or man.get("stop_radius") is None:
        return None
    alt = alt_target(scene)
    if alt is None:
        return None
    alt_relation, alt_wp = alt
    stop_radius = float(man["stop_radius"])
    with open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return None
    if any(r.get("pose_before") is None or r.get("yaw_before") is None for r in rows):
        return None
    alt_actions, changed, skipped = relabel(
        rows, scene["pillars"], alt_wp, stop_radius)
    if changed == 0:
        return None

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(ep_dir):
        if name.endswith(".png"):
            link_or_copy(os.path.join(ep_dir, name), os.path.join(out_dir, name))
    shutil.copy2(os.path.join(ep_dir, "oracle.jsonl"),
                 os.path.join(out_dir, "oracle.jsonl"))

    with open(os.path.join(out_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
        for row, act in zip(rows, alt_actions):
            row = dict(row)
            row["action_expert"] = act
            row["action_executed"] = act
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    instruction = ins.make_instruction(scene["target_color"], alt_relation)
    scene2 = dict(scene)
    scene2.update({"relation": alt_relation, "instruction": instruction,
                   "waypoint": alt_wp})
    scene_path = os.path.join(out_dir, "scene.json")
    dump_json(scene_path, scene2)
    sha = hashlib.sha256(open(scene_path, "rb").read()).hexdigest()

    man2 = dict(man)
    man2.update({
        "episode_id": new_sid, "scene_id": new_sid, "seed": new_sid,
        "instruction": instruction, "relation": alt_relation, "waypoint": alt_wp,
        "stopped_on_goal": False, "error": None,
        "counterfactual_of": man.get("episode_id"),
        "counterfactual_changed_steps": changed,
        "scene_meta_sha256": sha,
    })
    dump_json(os.path.join(out_dir, "episode_manifest.json"), man2)
    return {"source": ep_dir, "out": out_dir, "changed": changed,
            "steps": len(rows), "skipped_no_pose": skipped}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="采集根目录（含 split 子目录）")
    p.add_argument("--out", required=True, help="反事实集输出根目录")
    p.add_argument("--splits", default="train", help="逗号分隔，默认 train（test 不动）")
    p.add_argument("--limit", type=int, default=0, help="每个 split 最多处理多少集（0=全部）")
    args = p.parse_args()

    total = changed = steps = 0
    records = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        root = os.path.join(args.src, split)
        if not os.path.isdir(root):
            continue
        n = 0
        for name in sorted(os.listdir(root)):
            ep_dir = os.path.join(root, name)
            if not os.path.isdir(ep_dir):
                continue
            sid = int(name)
            new_sid = sid * 10 + 9
            rec = make_one(ep_dir, os.path.join(args.out, split, str(new_sid)), new_sid)
            if rec is None:
                continue
            rec["split"] = split
            records.append(rec)
            total += 1
            changed += rec["changed"]
            steps += rec["steps"]
            n += 1
            if args.limit and n >= args.limit:
                break
        print(f"{split}: 生成 {n} 个反事实集")

    dump_json(os.path.join(args.out, "counterfactual_manifest.json"), {
        "schema": "uavrgb.counterfactual.v1",
        "source": os.path.abspath(args.src),
        "splits": args.splits,
        "stop_at_waypoint": True,
        "episodes": total, "steps": steps, "changed_steps": changed,
        "changed_share": round(changed / steps, 4) if steps else 0.0,
        "records": records,
    })
    print(f"episodes={total} steps={steps} changed={changed} "
          f"({changed / steps:.1%})" if steps else "没有可生成的反事实集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
