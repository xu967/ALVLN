#!/usr/bin/env python3
"""校园任务契约：可见目标、矩形碰撞、拓扑隔离、半径与历史兼容。"""
import math
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import campus
import instruction
import scene_gen as sg
import split_manifest as sm
import teacher
import eval_closed_loop as ec


def main():
    template = sg._read_template(sg.DEFAULT_TEMPLATE)
    groups = {}
    for seed in range(48):
        world, meta = sg.build_scene(template, seed, seed, task="campus")
        assert (world, meta) == sg.build_scene(template, seed, seed, task="campus")
        ET.fromstring(world)
        split = sm.split_for_scene(meta)
        groups.setdefault(meta["map_id"], set()).add(split)
        assert len(instruction.tokenize(meta["instruction"])) <= 9
        assert instruction.UNK not in instruction.tokenize(meta["instruction"])
        assert "stop" not in meta["instruction"]
        wp = meta["waypoint"]
        target = next(p for p in meta["pillars"] if p.get("color") and p["y"] == wp["y"])
        assert abs(math.degrees(math.atan2(target["y"], target["x"] + .3))) < 30
        assert not teacher._point_blocked(wp["x"], wp["y"], meta["pillars"], .856)
        assert math.hypot(wp["x"] + .5, wp["y"]) > campus.RADIUS
        # 所有碰撞模型（地面/飞行器除外）必须进入教师几何。
        models = ET.fromstring(world).find("world").findall("model")
        collisions = {m.attrib["name"] for m in models if m.find(".//collision") is not None}
        assert collisions - {"ground_plane", "x3"} == {p["name"] for p in meta["pillars"]}
        shuffled = ec.override_instruction(meta, "shuffle")
        assert shuffled["waypoint"] == wp
        assert shuffled["instruction"] != meta["instruction"]
    assert len(groups) == 3 and all(len(v) == 1 for v in groups.values())
    assert {next(iter(v)) for v in groups.values()} == {"train", "val", "test"}
    forged = dict(meta, map_id="campus.v1/building_lane")
    try:
        sm.split_for_scene(forged)
    except ValueError:
        pass
    else:
        raise AssertionError("伪造地图组未被拒绝")
    # 矩形建筑不能仍按0.4m柱子处理，路径不能穿墙。
    wall = [dict(x=2., y=0., size=[1., 4., 4.])]
    assert teacher._point_blocked(2., 1.5, wall, .5)
    assert not teacher.ray_clear(0., 1.5, 0., 4., wall, .5)
    rows = [dict(collision=False, action_executed=1)]
    for distance, want in ((.49, True), (.5, True), (.51, False), (1.4, False)):
        result = ec.summarize_episode(rows, [dict(pose_gt=dict(x=distance, y=0.))], 5.,
                                      dict(x=0., y=0.), stop_radius=.5, goal_radius=.5)
        assert result["success"] is want
    # 使用真实打包器验证外观未标记/已标记、不同外观不能静默混包。
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        subprocess.run([sys.executable, str(root / "sim/tools/scene_gen.py"),
                        "--task", "campus", "--seed", "16", "--out", str(temp / "world.sdf")],
                       check=True, capture_output=True)
        subprocess.run([sys.executable, str(root / "tools/make_synthetic_episode.py"),
                        "--out", str(temp / "episodes")], check=True, capture_output=True)
        eps = [temp / "episodes" / name for name in ("ep1", "ep2", "ep3")]
        for i, ep in enumerate(eps):
            p = ep / "episode_manifest.json"
            man = json.loads(p.read_text())
            man["appearance_version"] = "v1" if i else "legacy"
            p.write_text(json.dumps(man))
        result = subprocess.run([sys.executable, str(root / "tools/preprocess.py"),
                                 "--out", str(temp / "packed"), *map(str, eps)],
                                capture_output=True, text=True)
        assert result.returncode != 0 and "appearance_version" in result.stderr, result.stderr
    print("test_campus: PASS（48 scenes / 3 topology holdouts / goal / collision geometry）")


if __name__ == "__main__":
    main()
