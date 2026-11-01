#!/usr/bin/env python3
"""出生位姿回归：default/random 不撞柱、SDF 的 x3 pose 真的换成 spawn；safe 模式已删除。"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402

MODES = ("default", "random")


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
    print(f"test_spawn: PASS（20 seeds × {len(MODES)} modes；safe 模式已删除）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
