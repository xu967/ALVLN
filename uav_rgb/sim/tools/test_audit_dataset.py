#!/usr/bin/env python3
"""audit_dataset.py 自检：干净数据全过，各类缺陷能被抓出来。"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_dataset as ad

PILLARS = [{"name": "pillar_0", "x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
OTHER = [{"name": "pillar_0", "x": 9.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]


def write_ep(root, split, eid, actions, dts=None, pillars=PILLARS):
    d = os.path.join(root, split, str(eid))
    os.makedirs(d)
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
        json.dump({"pillars": pillars}, f)
    man = {"split": split, "actions_ref": "actions.jsonl", "oracle_ref": "oracle.jsonl",
           "validity": {"camera_verified": True, "physics_verified": True,
                        "episode_executed": True}}
    with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)

    rows, oracle, t = [], [], 0
    for i, a in enumerate(actions):
        dt = dts[i] if dts else 1.0
        start, t = t, t + int(dt * 1e9)
        rows.append({"action_executed": a, "collision": False, "depth_rgb_gap_ns": 0,
                     "pose_rgb_gap_ns": 0,
                     "displacement_m": 0.25 if a == 1 else 0.0,
                     "dyaw_deg": {2: 10.0, 3: -10.0}.get(a, 0.0)})
        oracle.append({"action_start_ns": start, "action_end_ns": t,
                       "pose_gt": {"x": i * 0.25, "y": 0.0, "z": 2.0}})
    for name, items in (("actions.jsonl", rows), ("oracle.jsonl", oracle)):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in items)


def write_plan_ep(root, split, eid, n=12):
    """标签确实是 plan_action 算出来的集：复算应当逐条对上。

    这里只验审计脚本读对了哪些字段、比对口径对不对；plan_action 自身另有
    teacher.py 的 22 项自测。
    """
    d = os.path.join(root, split, str(eid))
    os.makedirs(d)
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
        json.dump({"pillars": PILLARS}, f)
    man = {"split": split, "teacher": "plan_action", "actions_ref": "actions.jsonl",
           "oracle_ref": "oracle.jsonl",
           "validity": {"camera_verified": True, "physics_verified": True,
                        "episode_executed": True}}
    with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    rows, oracle, t = [], [], 0
    for i in range(n):
        pb, yb = {"x": 0.0, "y": 0.05 * i, "z": 2.0}, 0.0
        a = ad.plan_action(pb["x"], pb["y"], yb, PILLARS)
        start, t = t, t + 1_000_000_000
        rows.append({"action_executed": a, "action_expert": a, "collision": False,
                     "depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0,
                     "displacement_m": 0.25 if a == 1 else 0.0,
                     "dyaw_deg": 0.0, "pose_before": pb, "yaw_before": yb})
        oracle.append({"action_start_ns": start, "action_end_ns": t,
                       "pose_gt": {"x": 0.0, "y": 0.05 * i, "z": 2.0}})
    for name, items in (("actions.jsonl", rows), ("oracle.jsonl", oracle)):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in items)
    return os.path.join(d, "actions.jsonl")


def statuses(root):
    eps = sorted(os.path.join(dp) for dp, _d, fs in os.walk(root)
                 if "episode_manifest.json" in fs)
    return {c["name"]: c["status"] for c in ad.audit([ad.episode_facts(e) for e in eps])}


def details(root):
    eps = sorted(os.path.join(dp) for dp, _d, fs in os.walk(root)
                 if "episode_manifest.json" in fs)
    return {c["name"]: c for c in ad.audit([ad.episode_facts(e) for e in eps])}


CLEAN = [0, 2, 1, 1, 3, 1] * 4


def main():
    root = tempfile.mkdtemp(prefix="audit_test_")
    try:
        # 1) 干净数据：除 split（只有单个 split）与需新字段的两项外都 PASS
        write_ep(root, "train", 1, CLEAN)
        st = statuses(root)
        for name in ("manifest_validity", "time_pairing", "action_window",
                     "forward_motion", "transition_lag", "yaw_direction", "class_share"):
            assert st[name] == "PASS", (name, st)
        assert st["split_geometry"] == "SKIP", st
        assert st["label_recompute"] == "SKIP", st

        # 2) 动作窗口漂移 -> action_window FAIL
        root2 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root2, "train", 1, CLEAN,
                 dts=[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] * 2)
        assert statuses(root2)["action_window"] == "FAIL"
        shutil.rmtree(root2)

        # 3) 只有一类 -> class_share FAIL
        root3 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root3, "train", 1, [1] * 12)
        assert statuses(root3)["class_share"] == "FAIL"
        shutil.rmtree(root3)

        # 4) train/test 几何相同 -> split_geometry FAIL
        write_ep(root, "test", 2, CLEAN, pillars=PILLARS)
        assert statuses(root)["split_geometry"] == "FAIL"

        # 5) plan_action 标签复算：自洽的过，改坏一条就 FAIL
        root5 = tempfile.mkdtemp(prefix="audit_test_")
        actions_path = write_plan_ep(root5, "train", 1)
        assert statuses(root5)["label_recompute"] == "PASS", statuses(root5)
        with open(actions_path, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        rows[0]["action_expert"] = (rows[0]["action_expert"] + 1) % 4
        with open(actions_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        assert statuses(root5)["label_recompute"] == "FAIL", statuses(root5)

        shutil.rmtree(root5)

        # 6) 一条都复算不了 -> SKIP，不能报成「100% 不一致」
        root6 = tempfile.mkdtemp(prefix="audit_test_")
        p6 = write_plan_ep(root6, "train", 1)
        with open(p6, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        for r in rows:
            r["yaw_before"] = None
        with open(p6, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        st6 = details(root6)["label_recompute"]
        assert st6["status"] == "SKIP", st6
        assert "缺动作前位姿" in st6["detail"], st6
        shutil.rmtree(root6)

        # 7) 新旧格式混在一起：旧集不能让新集一起降级成 SKIP
        root7 = tempfile.mkdtemp(prefix="audit_test_")
        write_plan_ep(root7, "train", 1)             # 12 步可复算
        write_ep(root7, "train", 2, CLEAN)           # 旧格式，无 pose_before
        st7 = details(root7)["label_recompute"]
        assert st7["status"] == "PASS", st7
        assert st7["covered_steps"] == 12, st7
        shutil.rmtree(root7)

        # 8) scene.json 里 pillars 写成 null：不能把整个 audit 弄崩
        root8 = tempfile.mkdtemp(prefix="audit_test_")
        write_plan_ep(root8, "train", 1)
        with open(os.path.join(root8, "train", "1", "scene.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"pillars": None}, f)
        st8 = details(root8)["label_recompute"]      # 不抛异常就算过
        assert st8["status"] == "SKIP", st8
        shutil.rmtree(root8)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("test_audit_dataset: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
