#!/usr/bin/env python3
"""audit_dataset.py 自检：干净数据全过，各类缺陷能被抓出来。"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_dataset as ad
import split_manifest as sm
from teacher import TEACHER_VERSION

PILLARS = [{"name": "pillar_0", "x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
OTHER = [{"name": "pillar_0", "x": 9.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
# L1 同色柱对：relation=right → 目标 = 左侧柱（y=+1），waypoint 在它外侧。
L1_PAIR = [
    {"name": "p_left", "x": 1.5, "y": 1.0, "z": 1.5, "size": [0.4, 0.4, 3.0],
     "color": "red"},
    {"name": "p_right", "x": 1.5, "y": -1.0, "z": 1.5, "size": [0.4, 0.4, 3.0],
     "color": "red"},
]
L1_SCENE = {
    "target_color": "red", "relation": "right",
    "waypoint": {"x": 1.5, "y": 1.8},
    "spawn": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
    "stop_radius": 0.5, "task_version": "l1.v4",
}
L1_INSTR = "pass the red pillar on your right and stop"


def write_ep(root, split, eid, actions, dts=None, pillars=PILLARS, instruction=None,
             window_from=None, scene_extra=None, man_extra=None):
    d = os.path.join(root, split, str(eid))
    os.makedirs(d)
    scene = {"pillars": pillars}
    scene.update(scene_extra or {})
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
        json.dump(scene, f)
    man = {"split": split, "teacher": "plan_action", "planner": "plan_action",
           "geometry_version": sm.SPLIT_SALT, "teacher_version": TEACHER_VERSION,
           "instruction": instruction,
           "actions_ref": "actions.jsonl", "oracle_ref": "oracle.jsonl",
           "validity": {"camera_verified": True, "physics_verified": True,
                        "episode_executed": True}}
    if window_from is not None:   # 缺省不写：模拟字段引入前的旧数据
        man["window_from"] = window_from
    man.update(man_extra or {})
    with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)

    rows, oracle, t = [], [], 0
    for i, a in enumerate(actions):
        dt = dts[i] if dts else 1.0
        start, t = t, t + int(dt * 1e9)
        rows.append({"step": i, "action_executed": a, "collision": False,
                     "depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0,
                     "displacement_m": 0.25 if a == 1 else 0.0,
                     "dyaw_deg": {2: 10.0, 3: -10.0}.get(a, 0.0)})
        oracle.append({"step": i, "action_start_ns": start, "action_end_ns": t,
                       "pose_gt": {"x": i * 0.25, "y": 0.0, "z": 2.0}})
    for name, items in (("actions.jsonl", rows), ("oracle.jsonl", oracle)):
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in items)


def write_plan_ep(root, split, eid, n=12, pillars=PILLARS, instruction=None,
                  scene_extra=None, man_extra=None):
    """标签确实是 plan_action 算出来的集：复算应当逐条对上。

    这里只验审计脚本读对了哪些字段、比对口径对不对；plan_action 自身另有
    teacher.py 的 22 项自测。
    """
    d = os.path.join(root, split, str(eid))
    os.makedirs(d)
    scene = {"pillars": pillars}
    scene.update(scene_extra or {})
    with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
        json.dump(scene, f)
    man = {"split": split, "teacher": "plan_action", "planner": "plan_action",
           "geometry_version": sm.SPLIT_SALT, "teacher_version": TEACHER_VERSION,
           "instruction": instruction,
           "actions_ref": "actions.jsonl", "oracle_ref": "oracle.jsonl",
           "validity": {"camera_verified": True, "physics_verified": True,
                        "episode_executed": True}}
    man.update(man_extra or {})
    with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    rows, oracle, t = [], [], 0
    for i in range(n):
        pb, yb = {"x": 0.0, "y": 0.05 * i, "z": 2.0}, 0.0
        a = ad.plan_action(pb["x"], pb["y"], yb, pillars)
        start, t = t, t + 1_000_000_000
        rows.append({"step": i, "action_executed": a, "action_expert": a,
                     "collision": False, "depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0,
                     "displacement_m": 0.25 if a == 1 else 0.0,
                     "dyaw_deg": 0.0, "pose_before": pb, "yaw_before": yb})
        oracle.append({"step": i, "action_start_ns": start, "action_end_ns": t,
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
        # 旧数据（无 window_from）新检查必须 SKIP。
        assert st["window_semantics"] == "SKIP", st

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

        # 9) --exclude-collision-episodes 排除前统计到的 SAFE+碰撞必须让检查 FAIL，
        #    不能因为排除后看不到 collision 步就恒 PASS。
        root9 = tempfile.mkdtemp(prefix="audit_test_")
        write_plan_ep(root9, "train", 1)
        facts9 = [ad.episode_facts(os.path.join(root9, "train", "1"))]
        c9 = {c["name"]: c for c in ad.audit(facts9, pre_safe_collisions=1)}
        assert c9["safe_collision"]["status"] == "FAIL", c9["safe_collision"]
        shutil.rmtree(root9)

        # 10b) 反向偏航接续步：上一步反向 YAW 时 net dyaw 可能≈0（惯性），
        #      不能判 FAIL；但非反转步方向错仍然要抓。
        #      序列 [1,2,3,1,2,3,1]：idx2、idx5 是反转步（2->3），idx4 是普通步。
        root10b = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root10b, "train", 1, [1, 2, 3, 1, 2, 3, 1])
        p10b = os.path.join(root10b, "train", "1", "actions.jsonl")
        with open(p10b, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        rows[2]["dyaw_deg"] = 0.1      # 反转步几乎没转：跳过，不判错
        rows[4]["dyaw_deg"] = -10.0    # 普通 YAW_LEFT 方向反了 -> FAIL
        with open(p10b, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        assert details(root10b)["yaw_direction"]["status"] == "FAIL"
        rows[4]["dyaw_deg"] = 10.0     # 修好普通步：只剩反转步被跳过 -> PASS
        with open(p10b, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        c10b = details(root10b)["yaw_direction"]
        assert c10b["status"] == "PASS" and "跳过 2 个反向接续步" in c10b["detail"], c10b
        shutil.rmtree(root10b)

        # 10c) 全部偏航步都是反转接续步：不能谎报「没有 dyaw」，要给出准确原因。
        root10c = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root10c, "train", 1, [1, 2, 3, 2])
        p10c = os.path.join(root10c, "train", "1", "actions.jsonl")
        with open(p10c, encoding="utf-8") as f:
            rows = [json.loads(x) for x in f if x.strip()]
        rows[1]["dyaw_deg"] = None     # 首个偏航步缺 dyaw：不进 yaw 列表
        with open(p10c, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        c10c = details(root10c)["yaw_direction"]
        assert c10c["status"] == "SKIP" and "反向接续步" in c10c["detail"], c10c
        shutil.rmtree(root10c)

        # 10) 全局每类 ≥5%，但某个 split 整类缺席：必须 FAIL，否则该 split 的
        #     recall 结构性为 0，T3 的「四类 recall 全 >0」会被静默绕过。
        root10 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root10, "train", 1, CLEAN)
        write_ep(root10, "val", 2, [1] * 12)       # val 里没有 SAFE/LEFT/RIGHT
        st10 = details(root10)["class_share"]
        assert st10["status"] == "FAIL", st10
        assert "val:SAFE_NEUTRAL" in st10["detail"], st10
        shutil.rmtree(root10)

        # 11) 被 --exclude 排除的集仍参与版本/validity/几何隔离判定：
        #     排除碰撞集不能把旧版本集一起洗成 PASS。
        root11 = tempfile.mkdtemp(prefix="audit_test_")
        write_plan_ep(root11, "train", 1)
        write_plan_ep(root11, "test", 2)
        old = os.path.join(root11, "test", "2", "episode_manifest.json")
        with open(old, encoding="utf-8") as f:
            man11 = json.load(f)
        man11["geometry_version"] = "uavrgb.geom.v2"
        man11["validity"]["physics_verified"] = False
        with open(old, "w", encoding="utf-8") as f:
            json.dump(man11, f)
        good = ad.episode_facts(os.path.join(root11, "train", "1"))
        bad = ad.episode_facts(os.path.join(root11, "test", "2"))
        c11 = {c["name"]: c for c in ad.audit([good], pre_excluded=[bad])}
        assert c11["geometry_version"]["status"] == "FAIL", c11["geometry_version"]
        assert c11["manifest_validity"]["status"] == "FAIL", c11["manifest_validity"]
        shutil.rmtree(root11)

        # 12a) oracle 比 actions 短：zip 会静默截短尾步，必须按「读不出来」判 FAIL。
        root12 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root12, "train", 1, CLEAN)
        p12 = os.path.join(root12, "train", "1", "oracle.jsonl")
        with open(p12, encoding="utf-8") as f:
            lines = [x for x in f if x.strip()]
        with open(p12, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])
        st12 = statuses(root12)
        assert st12.get("read_episodes") == "FAIL", st12
        assert st12.get("manifest_validity") != "PASS", st12
        shutil.rmtree(root12)

        # 12b) manifest 的 num_steps 与实际 actions 行数不一致：同样必须拒绝。
        root12b = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root12b, "train", 1, CLEAN)
        pm = os.path.join(root12b, "train", "1", "episode_manifest.json")
        with open(pm, encoding="utf-8") as f:
            man12 = json.load(f)
        man12["num_steps"] = 999
        with open(pm, "w", encoding="utf-8") as f:
            json.dump(man12, f)
        st12b = statuses(root12b)
        assert st12b.get("read_episodes") == "FAIL", st12b
        shutil.rmtree(root12b)

        # 12c) 行数相同但 step 错位：不能按行号硬配 oracle。
        root12c = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root12c, "train", 1, CLEAN)
        p12c = os.path.join(root12c, "train", "1", "oracle.jsonl")
        with open(p12c, encoding="utf-8") as f:
            recs12c = [json.loads(x) for x in f if x.strip()]
        recs12c[2]["step"], recs12c[3]["step"] = recs12c[3]["step"], recs12c[2]["step"]
        with open(p12c, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs12c)
        st12c = statuses(root12c)
        assert st12c.get("read_episodes") == "FAIL", st12c
        shutil.rmtree(root12c)

        # 13) gap 字段缺失：部分缺 -> FAIL；全部缺 -> SKIP，不能按 0 放过。
        root13 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root13, "train", 1, CLEAN)
        p13 = os.path.join(root13, "train", "1", "actions.jsonl")
        with open(p13, encoding="utf-8") as f:
            rows13 = [json.loads(x) for x in f if x.strip()]
        del rows13[3]["depth_rgb_gap_ns"]
        with open(p13, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows13)
        c13 = details(root13)["time_pairing"]
        assert c13["status"] == "FAIL" and "缺 gap" in c13["detail"], c13
        for r in rows13:
            r.pop("depth_rgb_gap_ns", None)
            r.pop("pose_rgb_gap_ns", None)
        with open(p13, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows13)
        c13b = details(root13)["time_pairing"]
        assert c13b["status"] == "SKIP", c13b
        shutil.rmtree(root13)

        # 14) scene.json 带 clearance 但与当前 scene_gen 常量不一致 -> FAIL。
        root14 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root14, "train", 1, CLEAN)
        p14 = os.path.join(root14, "train", "1", "scene.json")
        with open(p14, encoding="utf-8") as f:
            scene14 = json.load(f)
        scene14["clearance"] = dict(ad.GEOM_CONST)
        scene14["clearance"]["lane_sep_m"] = 99.0
        with open(p14, "w", encoding="utf-8") as f:
            json.dump(scene14, f)
        c14 = details(root14)["geometry_version"]
        assert c14["status"] == "FAIL" and "lane_sep_m" in c14["detail"], c14
        shutil.rmtree(root14)

        # 15) 指令覆盖：全有 PASS、部分缺 FAIL、含坐标 FAIL、全无 SKIP。
        root15 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root15, "train", 1, CLEAN, instruction="pass the red pillar")
        write_ep(root15, "train", 2, CLEAN, instruction="pass the blue pillar")
        c15 = details(root15)["instruction_coverage"]
        assert c15["status"] == "PASS", c15
        write_ep(root15, "train", 3, CLEAN)  # 缺 instruction
        c15b = details(root15)["instruction_coverage"]
        assert c15b["status"] == "FAIL", c15b
        shutil.rmtree(root15)

        root15b = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root15b, "train", 1, CLEAN, instruction="go to x=1.5")
        c15c = details(root15b)["instruction_coverage"]
        assert c15c["status"] == "FAIL" and "坐标" in c15c["detail"], c15c
        shutil.rmtree(root15b)

        # 16) 扫描范围：有 collection_manifest 时只审计登记目录，不能把 eval 的
        #     .work/ 当采集数据；没有 manifest 时 walk 也要剪掉 .work/。
        root16 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root16, "train", 1, CLEAN)
        write_ep(root16, ".work", 999, CLEAN)
        with open(os.path.join(root16, "collection_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"splits": {"train": [1]}}, f)
        dirs16, src16 = ad.collection_dirs(root16)
        assert src16 == "collection_manifest", (dirs16, src16)
        assert dirs16 == [os.path.join(root16, "train", "1")], dirs16
        shutil.rmtree(root16)
        root16b = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root16b, "train", 1, CLEAN)
        write_ep(root16b, ".work", 999, CLEAN)
        dirs16b, src16b = ad.collection_dirs(root16b)
        assert src16b == "walk", (dirs16b, src16b)
        assert dirs16b == [os.path.join(root16b, "train", "1")], dirs16b
        shutil.rmtree(root16b)

        # 17) 动作窗口语义：全一致 PASS、混用 FAIL、全缺失 SKIP（旧数据不能判死）。
        root17 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root17, "train", 1, CLEAN, instruction="pass the red pillar",
                 window_from="observation")
        write_ep(root17, "train", 2, CLEAN, instruction="pass the red pillar",
                 window_from="observation")
        all17 = details(root17)
        # 带 instruction 的常规数据：18 条 + 新增 waypoint_side = 19 条。
        assert len(all17) == 19, all17
        c17 = all17["window_semantics"]
        assert c17["status"] == "PASS" and "observation" in c17["detail"], c17
        write_ep(root17, "train", 3, CLEAN, window_from="dispatch")
        c17b = details(root17)["window_semantics"]
        assert c17b["status"] == "FAIL" and "dispatch" in c17b["detail"], c17b
        shutil.rmtree(root17)

        root17b = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root17b, "train", 1, CLEAN)
        write_ep(root17b, "train", 2, CLEAN)
        c17c = details(root17b)["window_semantics"]
        assert c17c["status"] == "SKIP", c17c
        assert "旧数据无此字段，无法判定" in c17c["detail"], c17c
        # 部分缺：新旧混包无法证明口径一致，不能当 SKIP 放过。
        write_ep(root17b, "train", 3, CLEAN, window_from="dispatch")
        assert details(root17b)["window_semantics"]["status"] == "FAIL"
        shutil.rmtree(root17b)

        # 18) L1 定义回归：彩色柱必须恰好 2 根同色；waypoint 必须在目标柱外侧。
        root18 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root18, "train", 1, CLEAN, pillars=L1_PAIR,
                 instruction=L1_INSTR, scene_extra=L1_SCENE)
        # relation_balance 要求 left/right 两类都有：补一集镜像的。
        write_ep(root18, "train", 2, CLEAN, pillars=L1_PAIR,
                 instruction="pass the red pillar on your left and stop",
                 scene_extra=dict(L1_SCENE, relation="left",
                                  waypoint={"x": 1.5, "y": -1.8}))
        d18 = details(root18)
        assert d18["target_color"]["status"] == "PASS", d18["target_color"]
        assert d18["relation_balance"]["status"] == "PASS", d18["relation_balance"]
        assert d18["waypoint_side"]["status"] == "PASS", d18["waypoint_side"]
        p18 = os.path.join(root18, "train", "1", "scene.json")
        with open(p18, encoding="utf-8") as f:
            s18 = json.load(f)
        # 两根彩色柱不同色 -> FAIL
        s18["pillars"][1]["color"] = "blue"
        with open(p18, "w", encoding="utf-8") as f:
            json.dump(s18, f)
        assert details(root18)["target_color"]["status"] == "FAIL"
        # 多于两根彩色柱 -> FAIL
        s18["pillars"][1]["color"] = "red"
        s18["pillars"].append({"name": "p_extra", "x": 2.0, "y": 1.6, "z": 1.5,
                               "size": [0.4, 0.4, 3.0], "color": "red"})
        with open(p18, "w", encoding="utf-8") as f:
            json.dump(s18, f)
        assert details(root18)["target_color"]["status"] == "FAIL"
        # 不同柱 / 异侧 / 贴轴线 三种 waypoint 都 -> FAIL
        s18["pillars"].pop()
        for wp in ({"x": 2.0, "y": 1.8}, {"x": 1.5, "y": -1.8},
                   {"x": 1.5, "y": 0.2}):
            s18["waypoint"] = wp
            with open(p18, "w", encoding="utf-8") as f:
                json.dump(s18, f)
            assert details(root18)["waypoint_side"]["status"] == "FAIL", wp
        shutil.rmtree(root18)

        # 19) 反事实集：轨迹类检查 SKIP（不假 FAIL），label_recompute 必须保留。
        root19 = tempfile.mkdtemp(prefix="audit_test_")
        write_plan_ep(root19, "train", 1, pillars=L1_PAIR, instruction=L1_INSTR,
                      scene_extra=L1_SCENE,
                      man_extra={"counterfactual_of": 900001})
        d19 = details(root19)
        for name in ("teacher_waypoint", "teacher_stop", "forward_motion",
                     "transition_lag", "yaw_direction", "class_share"):
            assert d19[name]["status"] == "SKIP", (name, d19[name])
            assert "反事实集" in d19[name]["detail"], (name, d19[name])
        assert d19["waypoint_side"]["status"] == "PASS", d19["waypoint_side"]
        assert d19["label_recompute"]["status"] == "PASS", d19["label_recompute"]
        shutil.rmtree(root19)

        # 20) provenance：teacher_version 有值必须统一；task_version 全缺视为同代。
        root20 = tempfile.mkdtemp(prefix="audit_test_")
        write_ep(root20, "train", 1, CLEAN)
        write_ep(root20, "train", 2, CLEAN)
        c20 = details(root20)["provenance_uniform"]
        assert c20["status"] == "PASS", c20
        assert "task_version 全部缺失" in c20["detail"], c20
        pm20 = os.path.join(root20, "train", "2", "episode_manifest.json")
        with open(pm20, encoding="utf-8") as f:
            m20 = json.load(f)
        m20["teacher_version"] = "uavrgb.teacher.v0"
        with open(pm20, "w", encoding="utf-8") as f:
            json.dump(m20, f)
        c20b = details(root20)["provenance_uniform"]
        assert c20b["status"] == "FAIL", c20b
        assert "teacher_version 混用" in c20b["detail"], c20b
        shutil.rmtree(root20)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("test_audit_dataset: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
