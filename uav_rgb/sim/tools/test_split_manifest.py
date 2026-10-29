#!/usr/bin/env python3
"""几何组 / 冻结 split 的回归用例（不启动 Gazebo，纯 CPU）。"""
import unittest

import split_manifest as sm


def pillars_of(seed):
    return sm.sg.sample_pillars(seed)[0]


class GeometryGroupTests(unittest.TestCase):
    def test_same_geometry_different_seed_shares_group(self):
        # 已知重复对：200 条数据集里真实出现的跨 split 泄漏。
        self.assertEqual(sm.geometry_gid(pillars_of(1023)),
                         sm.geometry_gid(pillars_of(1069)))

    def test_name_and_order_do_not_matter(self):
        ps = pillars_of(1042)
        shuffled = [dict(p, name="renamed") for p in reversed(ps)]
        self.assertEqual(sm.geometry_gid(ps), sm.geometry_gid(shuffled))

    def test_different_geometry_has_different_gid(self):
        a, b = sm.geometry_gid(pillars_of(1000)), sm.geometry_gid(pillars_of(1001))
        self.assertNotEqual(a, b)

    def test_generator_is_deterministic(self):
        self.assertEqual(sm.canonical_geometry(pillars_of(1007)),
                         sm.canonical_geometry(pillars_of(1007)))

    def test_negative_zero_and_zero_are_the_same_geometry(self):
        # seed 337 的 pillar y 是 -0.0，3651 是 0.0；物理布局相同，必须同 gid 同 split。
        self.assertEqual(sm._fmt(-0.0), sm._fmt(0.0))
        gid337 = sm.geometry_gid(pillars_of(337))
        gid3651 = sm.geometry_gid(pillars_of(3651))
        self.assertEqual(gid337, gid3651)
        self.assertEqual(sm.split_for_gid(gid337), sm.split_for_gid(gid3651))


class SplitAssignmentTests(unittest.TestCase):
    def test_split_only_depends_on_gid(self):
        gid = sm.geometry_gid(pillars_of(1013))
        self.assertEqual({sm.split_for_gid(gid) for _ in range(5)},
                         {sm.split_for_gid(gid)})

    def test_manifest_has_no_cross_split_group(self):
        seeds = list(range(1000, 1200))
        man = sm.build_manifest(seeds)
        pairs = [(s, g["split"]) for g in man["groups"] for s in g["seeds"]]
        self.assertEqual(len(pairs), len(seeds))
        self.assertEqual(sm.cross_split_groups(pairs), {})
        # val/test 不能被挤成个位数，否则宏指标没有意义。
        stats = sm.summarize(man)
        self.assertGreaterEqual(stats["per_split"]["val"]["groups"], 5)
        self.assertGreaterEqual(stats["per_split"]["test"]["groups"], 5)

    def test_current_generator_has_one_family(self):
        # 只有一族就不能宣称族留出；这个断言防止 family_holdout 被误读成 true。
        man = sm.build_manifest(list(range(1000, 1100)))
        self.assertFalse(sm.summarize(man)["family_holdout"])

    def test_legacy_policy_leakage_matches_handoff(self):
        # HANDOFF 记录 seed 0..1999 有 241 组跨 split、1124 种几何。
        # 改 scene_gen 几何语义会让这两个数字变，届时必须同步 bump SPLIT_SALT。
        pairs = [(s, "train" if s % 10 < 8 else ("val" if s % 10 == 8 else "test"))
                 for s in range(2000)]
        self.assertEqual(len(sm.cross_split_groups(pairs)), 241)
        self.assertEqual(len({sm.geometry_gid(pillars_of(s)) for s in range(2000)}), 1124)


if __name__ == "__main__":
    unittest.main()
