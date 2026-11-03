#!/usr/bin/env python3
"""出生位姿回归：default/random 不撞柱、SDF 的 x3 pose 真的换成 spawn；safe 模式已删除。"""
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402

MODES = ("default", "random")
# legacy 外观的 golden：改动前（2026-11-03）实测的 world sha256。
# 有人动了模板或外观分支却让默认输出变了，这里先红——旧数据/旧 ckpt 都按这个字节序。
GOLDEN_LEGACY = {
    0: "18ea28cc21d14f2d7b926c86f0b2cf2f4379b83a54d3e60b536156ed5241c345",
    1: "88fdabbe09fb832aec720abda55f5f6bcf40f8b963968444c265affc78d0c014",
    2: "7db4d5d325c40808a059c8dbc3bb047f2088694cf05b76431c5e81a3e0573b41",
}


def main():
    template = sg._read_template(sg.DEFAULT_TEMPLATE)
    for seed in range(20):
        pillars, _, _ = sg.sample_pillars(seed)
        # safe 出生模式已删除：plan_action 判 SAFE 的位姿必然让相机离柱面 0.10 m。
        # 有人把它加回来时这条会红。
        try:
            sg.make_spawn(pillars, seed, "safe")
        except ValueError:
            pass
        else:
            raise AssertionError(f"seed={seed}: safe 出生模式应已删除")
        for mode in MODES:
            spawn = sg.make_spawn(pillars, seed, mode)
            assert spawn["z"] == sg.SPAWN_Z, (seed, mode, spawn)
            assert not sg._spawn_collides(spawn["x"], spawn["y"], pillars), \
                (seed, mode, spawn)
        # SDF 里的 x3 pose 必须真的换成 spawn；生成两次必须字节一致。
        w1, m1 = sg.build_scene(template, seed, seed, "random")
        w2, _ = sg.build_scene(template, seed, seed, "random")
        assert w1 == w2, seed
        match = re.search(r'<model name="x3">\s*<pose>([^<]*)</pose>', w1)
        assert match, seed
        parts = match.group(1).split()
        assert abs(float(parts[0]) - m1["spawn"]["x"]) < 1e-6, (seed, parts)
        assert abs(float(parts[1]) - m1["spawn"]["y"]) < 1e-6, (seed, parts)
        assert m1["geometry_version"] == sg.GEOMETRY_VERSION, seed

    # 外观 v1：legacy 输出必须逐字节不变；v1 只准改外观（几何/出生位姿一致）。
    for seed, want in GOLDEN_LEGACY.items():
        legacy, lm = sg.build_scene(template, seed, seed)
        got = hashlib.sha256(legacy.encode("utf-8")).hexdigest()
        assert got == want, f"seed={seed}: legacy 外观输出变了 {got}"
        v1, vm = sg.build_scene(template, seed, seed, appearance="v1")
        assert v1 == sg.build_scene(template, seed, seed, appearance="v1")[0], seed
        assert 'name="landmark_' not in v1, f"seed={seed}: v1 仍有固定地标"
        assert "<scene>" in v1, f"seed={seed}: v1 缺 <scene>"
        assert vm["pillars"] == lm["pillars"], f"seed={seed}: v1 改了柱子几何"
        assert vm["spawn"] == lm["spawn"], f"seed={seed}: v1 改了出生位姿"
        assert vm["appearance_version"] == "v1", seed
    print(f"test_spawn: PASS（20 seeds × {len(MODES)} modes；safe 模式已删除；"
          f"legacy 字节不变 / v1 外观）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
