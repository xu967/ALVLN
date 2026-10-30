#!/usr/bin/env python3
"""traverse_rate 自检：safe 出生集不计入穿越率，否则 90% 验收线会随 safe 比例误 FAIL。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traverse_rate.py")


def write_ep(root, split, eid, x_end, spawn_mode):
    d = os.path.join(root, split, str(eid))
    os.makedirs(d)
    pillars = [{"name": "p", "x": 2.0, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
        json.dump({"pillars": pillars, "spawn": {"mode": spawn_mode}}, f)
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, "collision": False}) + "\n")
    with open(os.path.join(d, "oracle.jsonl"), "w", encoding="utf-8") as f:
        for x in (0.0, x_end):
            f.write(json.dumps({"pose_gt": {"x": x, "y": 0.0, "z": 2.0}}) + "\n")
    with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"split": split}, f)


def main():
    root = tempfile.mkdtemp(prefix="traverse_test_")
    try:
        write_ep(root, "test", 1, 3.5, "default")   # 过线（goal=3.0）
        write_ep(root, "test", 2, -0.5, "safe")     # 悬停，不应计入
        r = subprocess.run([sys.executable, TOOL, "--dataset", root, "--min-rate", "0.9"],
                           capture_output=True, text=True)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "episodes=1 success=1 rate=100.0%" in r.stdout, r.stdout
        assert "safe 出生 1 集不计入" in r.stdout, r.stdout
        # 不传 --min-rate 不能静默 exit 0：那是验收闸门，0% 也必须失败。
        r2 = subprocess.run([sys.executable, TOOL, "--dataset", root],
                            capture_output=True, text=True)
        assert r2.returncode != 0, (r2.returncode, r2.stdout)
        assert "--min-rate" in r2.stderr, r2.stderr
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("test_traverse_rate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
