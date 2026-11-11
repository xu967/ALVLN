#!/usr/bin/env python3
"""audit_dataset 的标签复算按 manifest.teacher_version 分派（第十二阶段 W1-7：TEACHER_VERSION v5 → v7）。

升版之前，label_recompute 只复算「版本等于当前版本」的集，旧版本的集整集跳过（SKIP）。升版以后跳过会让 v5 的
0417 等历史集失去标签复算，所以改成分派：manifest 写 v5 的集按 v5 规则（B2、B3 关）复算，写 v7 的按 v7 规则（B2、B3 开）复算，
costmap 集按 cost v1 / cost v2 的规则复算；版本不认识的集仍然跳过。

覆盖：A* 集（B2 出口区、B3 无路重试两类状态）新旧版本各自复算通过、拿错规则会 FAIL；未知版本 SKIP；
同一次审计里 v5 与 v7 两集各按各的规则复算；costmap 集 v1 / v2；campus 身份校验认 v5 与 v7、不认其它版本。
"""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_dataset as ad  # noqa: E402
import split_manifest as sm  # noqa: E402
import teacher as T  # noqa: E402
import test_audit_dataset as tad  # noqa: E402  复用 write_ep / details 夹具

V5, V7 = "uavrgb.teacher.v5", "uavrgb.teacher.v7"
COST_V1, COST_V2 = "uavrgb.teacher_cost.v1", "uavrgb.teacher_cost.v2"
GATE = tad.PILLARS                                                     # 最远柱 x=1.5
WALL = [{"name": "wall", "x": 1.55, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
EXIT_STATES = [(2.0, 1.2, 0.0), (2.0, -1.2, 0.0), (2.4, 0.8, 0.0), (0.0, 0.0, 0.0)]   # B2：前三个在出口区
RETRY_STATES = [(1.1, 0.0, 0.0), (1.2, 0.0, 0.0), (1.3, 0.0, 0.0), (-0.5, 0.0, 0.0)]  # B3：前两个标准余量无路、重试有路
COST_STATES = [(2.0, 1.5, 0.0), (2.0, -1.5, 0.0), (0.0, 0.0, 0.0)]


def labels_for(states, pillars, version, planner="astar"):
    """按指定版本的规则算标签（e1_mini 档）；算完把模块状态恢复成 x3 + 当前版本。"""
    T.set_vehicle("e1_mini", "base", version=version)
    try:
        if planner == "costmap":
            return [T.plan_action_costmap(x, y, yaw, pillars)[0] for x, y, yaw in states]
        return [T.plan_action_global_tiebreak(x, y, yaw, pillars, tiebreak="legacy")[0] for x, y, yaw in states]
    finally:
        T.set_vehicle("x3")


def write_label_ep(root, eid, states, labels, pillars, version, planner="astar"):
    man = {"vehicle": "e1_mini", "dynamics": "cf_sitl", "planner": planner, "teacher_version": version}
    tad.write_ep(root, "train", eid, labels, pillars=pillars, man_extra=man)
    path = os.path.join(root, "train", str(eid), "actions.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for row, a, (x, y, yaw) in zip(rows, labels, states):
        row.update({"action_expert": a, "pose_before": {"x": x, "y": y, "z": 2.0}, "yaw_before": yaw})
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def recompute(root):
    try:
        return tad.details(root)
    finally:
        T.set_vehicle("x3")      # audit 按 manifest.vehicle / 版本切过教师状态；不能漏到下一条用例


def run_cases(name, states, pillars, planner, v_old, v_new):
    """4 个组合：(标签规则版本, manifest 版本) = (旧,旧) PASS、(新,新) PASS、(旧,新) FAIL、(新,旧) FAIL。"""
    lab_old = labels_for(states, pillars, v_old, planner)
    lab_new = labels_for(states, pillars, v_new, planner)
    assert lab_old != lab_new, f"{name}: 夹具里新旧规则的标签必须不同（{lab_old}）"
    n_diff = sum(a != b for a, b in zip(lab_old, lab_new))
    for labels, version, want in ((lab_old, v_old, "PASS"), (lab_new, v_new, "PASS"),
                                  (lab_old, v_new, "FAIL"), (lab_new, v_old, "FAIL")):
        root = tempfile.mkdtemp(prefix="audit_dispatch_")
        try:
            write_label_ep(root, 1, states, labels, pillars, version, planner)
            c = recompute(root)["label_recompute"]
            assert c["status"] == want, (name, version, labels, c)
            assert c["covered_steps"] == len(states), (name, version, c)
            if want == "FAIL":
                assert c["mismatch"] == round(n_diff / len(states), 4), (name, version, c)
        finally:
            shutil.rmtree(root, ignore_errors=True)


def campus_facts(seed=0):
    import campus
    import scene_gen
    with open(scene_gen.DEFAULT_TEMPLATE, encoding="utf-8") as fp:
        template = fp.read()
    root = tempfile.mkdtemp(prefix="audit_dispatch_campus_")
    _, meta = campus.build(template, seed, seed)
    split = campus.split(seed)
    extras = dict(meta, geometry_gid=sm.geometry_gid(meta["pillars"]), planner="astar")
    tad.write_ep(root, split, seed, [0, 0], pillars=meta["pillars"], instruction=meta["instruction"],
                 scene_extra=meta, man_extra=extras)
    ep = os.path.join(root, split, str(seed))
    man_path = os.path.join(ep, "episode_manifest.json")
    man = ad.load_json(man_path)
    with open(os.path.join(ep, "scene.json"), "rb") as sf:
        man["scene_meta_sha256"] = hashlib.sha256(sf.read()).hexdigest()
    with open(man_path, "w", encoding="utf-8") as fp:
        json.dump(man, fp)
    wp = dict(meta["waypoint"], z=2.1)          # 停在 waypoint 上：标签 SAFE，与教师版本无关
    for name in ("actions.jsonl", "oracle.jsonl"):
        path = os.path.join(ep, name)
        rows = ad.load_jsonl(path)
        for row in rows:
            if name == "actions.jsonl":
                row.update(pose_before=wp, yaw_before=0., action_expert=0)
            else:
                row["pose_gt"] = wp
        with open(path, "w", encoding="utf-8") as fp:
            fp.writelines(json.dumps(row) + "\n" for row in rows)
    return root, ep


def main():
    # 1) A* 集：B2 出口区
    run_cases("B2 出口区", EXIT_STATES, GATE, "astar", V5, V7)
    # 2) A* 集：B3 无路重试
    run_cases("B3 无路重试", RETRY_STATES, WALL, "astar", V5, V7)
    # 3) costmap 集：B2 目标取线，cost v1 → v2
    run_cases("costmap B2", COST_STATES, GATE, "costmap", COST_V1, COST_V2)

    # 4) 版本不认识：SKIP，计入「旧几何或教师版本」，不抛异常
    root = tempfile.mkdtemp(prefix="audit_dispatch_")
    try:
        labs = labels_for(EXIT_STATES, GATE, V7)
        write_label_ep(root, 1, EXIT_STATES, labs, GATE, "uavrgb.teacher.v9")
        c = recompute(root)["label_recompute"]
        assert c["status"] == "SKIP" and "旧几何或教师版本 4 步" in c["detail"], c
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 5) 同一次审计里 v5 与 v7 两集各按各的规则复算（混版本本身由 provenance_uniform 判 FAIL）
    root = tempfile.mkdtemp(prefix="audit_dispatch_")
    try:
        write_label_ep(root, 1, EXIT_STATES, labels_for(EXIT_STATES, GATE, V5), GATE, V5)
        write_label_ep(root, 2, EXIT_STATES, labels_for(EXIT_STATES, GATE, V7), GATE, V7)
        d = recompute(root)
        assert d["label_recompute"]["status"] == "PASS" and d["label_recompute"]["covered_steps"] == 8, d["label_recompute"]
        assert d["provenance_uniform"]["status"] == "FAIL" and "teacher_version 混用" in d["provenance_uniform"]["detail"], \
            d["provenance_uniform"]
        # 顺序反过来（先 v7 后 v5）同样各按各的：规则状态不能从上一集漏到下一集
        shutil.rmtree(os.path.join(root, "train", "1"))
        write_label_ep(root, 3, EXIT_STATES, labels_for(EXIT_STATES, GATE, V5), GATE, V5)
        d = recompute(root)
        assert d["label_recompute"]["status"] == "PASS" and d["label_recompute"]["covered_steps"] == 8, d["label_recompute"]
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 6) forest_e1 档：旧森林集（写 base 档版本）仍跳过；forest 档版本 v6 的集按 v6 复算（test_audit_dataset 的 21 号用例覆盖），
    #    这里只锁分派表不把 v5/v7 放进 forest 档
    assert T.recomputable_versions("forest_e1") == (T.teacher_version("forest_e1"),)

    # 7) campus 身份校验：v5 与 v7 都认（旧集仍能按旧规则复算），其它版本不认
    root, ep = campus_facts()
    try:
        man_path = os.path.join(ep, "episode_manifest.json")
        base_man = ad.load_json(man_path)
        for version, want in ((V5, "PASS"), (V7, "PASS"), ("uavrgb.teacher.v9", "FAIL"), (None, "FAIL")):
            man = copy.deepcopy(base_man)
            if version is None:
                man.pop("teacher_version", None)
            else:
                man["teacher_version"] = version
            with open(man_path, "w", encoding="utf-8") as fp:
                json.dump(man, fp)
            checks = {c["name"]: c for c in ad.audit([ad.episode_facts(ep)])}
            assert checks["campus_identity"]["status"] == want, (version, checks["campus_identity"])
            if want == "PASS":
                assert checks["label_recompute"]["status"] == "PASS", (version, checks["label_recompute"])
                assert checks["label_recompute"]["covered_steps"] == 2, (version, checks["label_recompute"])
            T.set_vehicle("x3")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        T.set_vehicle("x3")

    print("test_audit_teacher_dispatch: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
