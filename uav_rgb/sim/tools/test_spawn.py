#!/usr/bin/env python3
"""出生位姿回归（T2）：safe spawn 必须真的是 plan_action 的 SAFE，且不撞柱。"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402
import teacher  # noqa: E402


def main():
    template = sg._read_template(sg.DEFAULT_TEMPLATE)
    n = 0
    for seed in range(20):
        pillars, _, _ = sg.sample_pillars(seed)
        for mode in ("default", "random", "safe"):
            spawn = sg.make_spawn(pillars, seed, mode)
            assert spawn["z"] == sg.SPAWN_Z, (seed, mode, spawn)
            assert not sg._spawn_collides(spawn["x"], spawn["y"], pillars), \
                (seed, mode, spawn)
            if mode == "safe":
                want = teacher.plan_action(spawn["x"], spawn["y"],
                                           spawn["yaw_deg"], pillars)
                assert want == teacher.SAFE_NEUTRAL, (seed, spawn, teacher.ACTIONS[want])
                n += 1
        # SDF 里的 x3 pose 必须真的换成 spawn；生成两次必须字节一致。
        w1, m1 = sg.build_scene(template, seed, seed, "safe")
        w2, _ = sg.build_scene(template, seed, seed, "safe")
        assert w1 == w2, seed
        match = re.search(r'<model name="x3">\s*<pose>([^<]*)</pose>', w1)
        assert match, seed
        parts = match.group(1).split()
        assert abs(float(parts[0]) - m1["spawn"]["x"]) < 1e-6, (seed, parts)
        assert abs(float(parts[1]) - m1["spawn"]["y"]) < 1e-6, (seed, parts)
        assert m1["geometry_version"] == sg.GEOMETRY_VERSION, seed
    print(f"test_spawn: PASS（20 seeds × 3 modes；safe {n} 个全部 plan_action=SAFE 且无碰撞）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
