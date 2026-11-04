#!/usr/bin/env python3
"""出生位姿回归：default/random 不撞柱、SDF 的 x3 pose 真的换成 spawn；safe 模式已删除。"""
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402

MODES = ("default", "random")
# legacy 外观的 golden：2026-11-04 加 chase/top-view sensors 后更新。
# 本次（同日）再把顶视相机高度 18 m → 6 m：18 m 下柱阵只占画面 4%，几乎全空。
# 两次变化都只动无质量/无碰撞的虚拟相机，不改变柱阵、出生位姿或学生 RGB 话题。
# 后续修改模板/外观分支仍需显式更新这些字节级基线。
GOLDEN_LEGACY = {
    0: "670685c3c3a40bd73c492e54051ce44cf455e2152caf2aa66c357533062b0212",
    1: "ee2e4af1f604bfd0ae68edc6c7ae36768832f696a72f6fb60023436523060f00",
    2: "6f0506e60b38d74c5babf43904236157f39d6cffd3aa1a906450bd5fda9411d1",
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
