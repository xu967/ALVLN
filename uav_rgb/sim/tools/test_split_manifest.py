#!/usr/bin/env python3
"""几何组 / 冻结 split 的回归用例（不启动 Gazebo，纯 CPU）。"""
import json
import os
import unittest
from pathlib import Path

import split_manifest as sm

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/root/rivermind-data")) / "uav_rgb"


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
        # HANDOFF 记录旧几何（v2，缝 0.902 m）seed 0..1999 有 241 组跨 split、
        # 1124 种几何；T1 放宽到 v3（缝 1.2 m）后重算为 291 / 759。
        # 改 scene_gen 几何语义会让这两个数字变，届时必须同步 bump SPLIT_SALT。
        pairs = [(s, "train" if s % 10 < 8 else ("val" if s % 10 == 8 else "test"))
                 for s in range(2000)]
        self.assertEqual(len(sm.cross_split_groups(pairs)), 291)
        self.assertEqual(len({sm.geometry_gid(pillars_of(s)) for s in range(2000)}), 759)


class RecordedSplitTests(unittest.TestCase):
    """重打包必须按记录的 salt/策略/版本复算，不能回落当前 SPLIT_SALT。"""

    def test_frozen_v3_uses_recorded_version(self):
        # t5 test/200561 的真实 gid：v3→test，当前 v5→train（35% 不符的来源之一）。
        meta = {"geometry_version": "uavrgb.geom.v3",
                "geometry_gid": "0226c6cbd9d1a7e5"}
        self.assertEqual(sm.split_for_scene(meta), "test")
        self.assertEqual(sm.split_for_gid(meta["geometry_gid"]), "train")

    def test_recorded_split_salt_wins(self):
        meta = {"geometry_version": sm.SPLIT_SALT, "geometry_gid": "x",
                "split_salt": "uavrgb.geom.v3"}
        self.assertEqual(sm.split_for_scene(meta),
                         sm.split_for_gid("x", "uavrgb.geom.v3"))

    def test_campus_v1_frozen_map_dispatch(self):
        import campus
        meta = {"seed": 822, "geometry_version": "campus.v1",
                "map_id": "campus.v1/building_lane",
                "split_policy": "campus_map_v1"}
        self.assertEqual(sm.split_for_scene(meta), "train")
        # 版本与 map_id 不一致必须 fail-closed（当前 v2 meta + v1 map_id）。
        with self.assertRaises(ValueError):
            sm.split_for_scene(dict(meta, geometry_version=campus.VERSION))

    def test_unknown_campus_version_fails_closed(self):
        meta = {"seed": 0, "geometry_version": "campus.v99",
                "map_id": "campus.v99/building_lane",
                "split_policy": "campus_map_v1"}
        with self.assertRaises(ValueError):
            sm.split_for_scene(meta)

    def test_asset_campus_version_dispatch(self):
        import asset_campus
        key = asset_campus.map_key(9)
        meta = {"seed": 9, "geometry_version": asset_campus.VERSION,
                "map_id": key, "split_policy": "campus_assets_map_v1"}
        self.assertEqual(sm.split_for_scene(meta), asset_campus.split_key(key))


class FrozenDataRegressionTests(unittest.TestCase):
    """只读 $DATA_ROOT 下真实冻结 manifest；机器上没有就跳过（不写数据）。"""

    def _check(self, base, pattern):
        root = DATA_ROOT / base
        paths = sorted(root.glob(pattern))
        if not paths:
            self.skipTest("缺少冻结数据 %s" % root)
        bad = []
        for path in paths:
            man = json.loads(path.read_text(encoding="utf-8"))
            try:
                got = sm.split_for_scene(man)
            except ValueError as exc:
                bad.append((str(path), str(exc)))
                continue
            if got != man["split"]:
                bad.append((str(path), got, man["split"]))
        self.assertEqual(bad, [])

    def test_t5_frozen_manifests(self):
        # t5 = uavrgb.geom.v3；当前 v5 salt 会改掉其中约 35% 的 split。
        self._check("t5-full-1155", "*/*/episode_manifest.json")

    def test_campus_v1_frozen_manifests(self):
        self._check("campus-paired-20261103", "*/*/*/episode_manifest.json")


if __name__ == "__main__":
    unittest.main()
