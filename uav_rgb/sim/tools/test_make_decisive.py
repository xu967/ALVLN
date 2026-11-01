#!/usr/bin/env python3
"""make_decisive 的最小回归：过滤、两条标签、指令互换、步数一致。"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_decisive as md  # noqa: E402


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def make_episode(root, sid=700001):
    d = os.path.join(root, "train", str(sid))
    os.makedirs(d, exist_ok=True)
    # 同色柱对：两根柱颜色一样，只有关系词能区分目标。
    scene = {
        "pillars": [{"x": 2.0, "y": 0.8, "color": "red"},
                    {"x": 2.0, "y": -0.8, "color": "red"}],
        "spawn": {"x": -0.5, "y": 0.0, "z": 0.15, "yaw_deg": 0.0},
        "target_color": "red", "relation": "left",
        "instruction": "pass the red pillar on your left and stop",
        "waypoint": {"x": 2.0, "y": 0.8}, "stop_radius": 0.5,
    }
    man = {"episode_id": sid, "scene_id": sid, "split": "train",
           "planner": "astar", "stop_radius": 0.5, "teacher": "plan_action",
           "num_steps": 4, "requested_steps": 4}
    write_json(os.path.join(d, "scene.json"), scene)
    write_json(os.path.join(d, "episode_manifest.json"), man)
    rows = []
    for i in range(4):
        rows.append({"step": i, "action_expert": [1, 0, 1, 0][i],
                     "action_executed": [1, 0, 1, 0][i], "frame": f"rgb_{i:05d}.png",
                     "pose_before": {"x": 0.5 + i * 0.1, "y": 0.0},
                     "yaw_before": 0.0})
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, "oracle.jsonl"), "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"step": i}, ensure_ascii=False) + "\n")
    for i in range(4):
        open(os.path.join(d, f"rgb_{i:05d}.png"), "wb").close()
    return d


def main():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        ep = make_episode(src)
        # 手工指定“换色后”的动作：第 1、3 步不同 → 只保留这两步。
        md.relabel = lambda rows, pillars, wp, stop: ([1, 3, 1, 2], 2, 0)
        rec = md.make_one(ep, os.path.join(tmp, "out"), "train")
        assert rec == {"source": ep, "steps": 2, "episodes": 2}, rec
        base = 9_000_000 + 2 * 700001
        orig = os.path.join(tmp, "out", "train", str(base + 1))
        alt = os.path.join(tmp, "out", "train", str(base + 2))
        for d, want in ((orig, [0, 0]), (alt, [3, 2])):
            man = json.load(open(os.path.join(d, "episode_manifest.json"), encoding="utf-8"))
            acts = [json.loads(l)["action_expert"]
                    for l in open(os.path.join(d, "actions.jsonl"), encoding="utf-8")]
            assert man["num_steps"] == 2, man
            assert acts == want, (d, acts, want)
            assert os.path.isfile(os.path.join(d, "rgb_00001.png"))
            assert not os.path.exists(os.path.join(d, "rgb_00000.png"))
        mo = json.load(open(os.path.join(orig, "episode_manifest.json"), encoding="utf-8"))
        ma = json.load(open(os.path.join(alt, "episode_manifest.json"), encoding="utf-8"))
        assert mo["target_color"] == ma["target_color"] == "red"
        assert mo["relation"] == "left" and ma["relation"] == "right", (mo, ma)
        assert "on your left" in mo["instruction"], mo["instruction"]
        assert "on your right" in ma["instruction"], ma["instruction"]
        # 反事实的 waypoint 必须落到另一根柱旁，不能还是原来那个。
        assert ma["waypoint"] != mo["waypoint"], (mo["waypoint"], ma["waypoint"])

        # 相邻 episode_id 的输出目录不能撞号：撞号时后写的会把前一对的
        # 反事实那一半覆盖掉，而那一半正是文本分支唯一的监督信号。
        ep2 = make_episode(src, 700002)
        md.make_one(ep2, os.path.join(tmp, "out"), "train")
        made = sorted(os.listdir(os.path.join(tmp, "out", "train")))
        assert len(made) == 4, f"预期 4 个输出目录，实际 {len(made)}: {made}"
        for d in made:
            n_rows = sum(1 for _ in open(os.path.join(tmp, "out", "train", d,
                                                      "actions.jsonl"), encoding="utf-8"))
            n_png = len([x for x in os.listdir(os.path.join(tmp, "out", "train", d))
                         if x.endswith(".png")])
            assert n_rows == 2 and n_png == 2, (d, n_rows, n_png)  # 没有孤儿帧
    print("test_make_decisive: PASS")


if __name__ == "__main__":
    main()
