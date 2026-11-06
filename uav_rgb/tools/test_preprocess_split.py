#!/usr/bin/env python3
"""preprocess 的 split 事实/诊断门禁/冻结复算/碰撞过滤回归（纯 CPU，临时目录）。

用 make_synthetic_episode 生成 3 条小 episode，再改写 manifest；不读真实数据。
"""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "tools"))
import asset_campus  # noqa: E402
import scene_forest as sf  # noqa: E402
import split_manifest as sm  # noqa: E402

PY = sys.executable
# t5 test/200561 的真实 gid：v3→test，当前 v5→train。
FROZEN_V3_GID = "0226c6cbd9d1a7e5"


def read_meta(path):
    with open(path, "rb") as f:
        head = f.read(48)
        meta_off = struct.unpack_from("<Q", head, 36)[0]
        meta_len = struct.unpack_from("<I", head, 44)[0]
        f.seek(meta_off)
        return json.loads(f.read(meta_len))


def gid_for(split, tag="g", salt=None):
    for i in range(10000):
        gid = "%s%d" % (tag, i)
        if sm.split_for_gid(gid, salt) == split:
            return gid
    raise AssertionError("找不到 %s split 的 gid" % split)


class PreprocessSplitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"),
                        "--out", str(self.tmp / "ep")], check=True,
                       capture_output=True)
        self.eps = [self.tmp / "ep" / n for n in ("ep1", "ep2", "ep3")]

    def tearDown(self):
        self._tmp.cleanup()

    def man(self, ep):
        return json.loads((ep / "episode_manifest.json").read_text(encoding="utf-8"))

    def save(self, ep, man):
        (ep / "episode_manifest.json").write_text(json.dumps(man), encoding="utf-8")

    def run_pre(self, out, eps, *flags):
        return subprocess.run(
            [PY, str(REPO / "tools/preprocess.py"), "--out", str(out),
             *flags, *map(str, eps)], capture_output=True, text=True)

    def test_records_real_split_facts(self):
        expect = {}
        for ep in self.eps:
            man = self.man(ep)
            man["geometry_gid"] = gid_for(man["split"], ep.name)
            expect[man["geometry_gid"]] = man["split"]
            self.save(ep, man)
        r = self.run_pre(self.tmp / "ds", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds" / "dataset.uavr")
        self.assertEqual(meta["split_policy"], "geometry_gid")
        self.assertEqual(meta["split_scheme"], "geometry_gid")
        self.assertEqual(meta["split_salt"], sm.SPLIT_SALT)
        self.assertEqual(meta["splits"], {"train": 10, "val": 5, "test": 0})
        groups = {g["geometry_gid"]: g["split"] for g in meta["split_groups"]}
        self.assertEqual(groups, expect)

    def test_same_scene_cross_split_rejected(self):
        ep1, ep2 = self.eps[0], self.eps[1]  # 两条都是 scene_id=42
        for ep, split in ((ep1, "train"), (ep2, "val")):
            man = self.man(ep)
            man["split"] = split
            man["geometry_gid"] = gid_for(split, ep.name)
            self.save(ep, man)
        r = self.run_pre(self.tmp / "ds", self.eps[:2])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("scene_id", r.stderr)

    def test_split_must_match_recorded_policy(self):
        man = self.man(self.eps[2])
        man["geometry_gid"] = gid_for("test", "mismatch")
        man["split"] = "train"  # gid 应属 test
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds", [self.eps[2]])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("split", r.stderr)

    def test_diagnostic_split_requires_flag(self):
        for ep in self.eps:
            man = self.man(ep)
            man["geometry_gid"] = gid_for(man["split"], ep.name)
            self.save(ep, man)
        r = self.run_pre(self.tmp / "ds_diag", self.eps, "--split-by-map", "diag1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--allow-diagnostic-split", r.stderr)
        r = self.run_pre(self.tmp / "ds_diag", self.eps, "--split-by-map", "diag1",
                         "--allow-diagnostic-split")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_diag" / "dataset.uavr")
        self.assertEqual(meta["split_scheme"], "map-hash:diag1")
        self.assertEqual(meta["split_policy"], "map-hash:diag1")
        self.assertEqual(meta["split_salt"], "diag1")

    def test_diagnostic_manifest_policy_rejected(self):
        man = self.man(self.eps[0])
        man["split_policy"] = "diag:probe"
        self.save(self.eps[0], man)
        r = self.run_pre(self.tmp / "ds", [self.eps[0]])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--allow-diagnostic-split", r.stderr)

    def test_frozen_v3_split_is_recomputed_from_recorded_version(self):
        self.assertNotEqual(sm.split_for_gid(FROZEN_V3_GID), "test")
        man = self.man(self.eps[0])
        man.update(split="test", geometry_version="uavrgb.geom.v3",
                   geometry_gid=FROZEN_V3_GID)
        self.save(self.eps[0], man)
        # 冻结版本必须显式放行；放行后按 v3 salt 复算，不能按当前 v5。
        r = self.run_pre(self.tmp / "ds_frozen", [self.eps[0]])
        self.assertNotEqual(r.returncode, 0)
        r = self.run_pre(self.tmp / "ds_frozen", [self.eps[0]],
                         "--allow-frozen-geometry")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_frozen" / "dataset.uavr")
        self.assertEqual(meta["split_scheme"], "geometry_gid")
        self.assertEqual(meta["split_salt"], "uavrgb.geom.v3")
        self.assertEqual(meta["splits"]["test"], 6)

    def test_campus_v1_frozen_split(self):
        man = self.man(self.eps[0])
        man.update(split="train", geometry_version="campus.v1",
                   geometry_gid="e4b44a42693b2b51",
                   split_policy="campus_map_v1",
                   map_id="campus.v1/building_lane")
        self.save(self.eps[0], man)
        r = self.run_pre(self.tmp / "ds_campus", [self.eps[0]],
                         "--allow-frozen-geometry")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_campus" / "dataset.uavr")
        self.assertEqual(meta["split_scheme"], "campus_map_v1")
        self.assertIsNone(meta["split_salt"])
        self.assertEqual(meta["splits"]["train"], 6)

    def test_asset_campus_version_is_accepted_without_frozen_flag(self):
        key = asset_campus.map_key(7)
        split = asset_campus.split_key(key)
        man = self.man(self.eps[0])
        man.update(split=split, geometry_version=asset_campus.VERSION,
                   geometry_gid=gid_for(split, "asset"),
                   split_policy="campus_assets_map_v1", map_id=key)
        self.save(self.eps[0], man)
        r = self.run_pre(self.tmp / "ds_asset", [self.eps[0]])
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_asset" / "dataset.uavr")
        self.assertEqual(meta["split_scheme"], "campus_assets_map_v1")
        self.assertEqual(meta["split_salt"], "asset-campus-split-v1")

    def test_collision_filter_is_recorded_and_changes_test_set(self):
        ep = self.eps[2]
        man = self.man(ep)
        man["split"] = "test"
        man["geometry_gid"] = gid_for("test", "collision")
        self.save(ep, man)
        actions = ep / "actions.jsonl"
        rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
        for row in rows:
            row["collision"] = row["step"] == 0
        actions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

        r = self.run_pre(self.tmp / "ds_keep", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_keep" / "dataset.uavr")
        self.assertEqual(meta["splits"]["test"], 5)
        self.assertEqual(meta["collision_filter"], {
            "drop_collision_steps": False, "exclude_collision_episodes": False,
            "excluded_episodes": 0})

        r = self.run_pre(self.tmp / "ds_excl", self.eps, "--exclude-collision-episodes")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_excl" / "dataset.uavr")
        # test 集随 flag 变化：跨 run 的 test 指标不可直接比较，meta 必须留痕。
        self.assertEqual(meta["splits"]["test"], 0)
        self.assertTrue(meta["collision_filter"]["exclude_collision_episodes"])
        self.assertEqual(meta["collision_filter"]["excluded_episodes"], 1)

        r = self.run_pre(self.tmp / "ds_drop", self.eps, "--drop-collision")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_drop" / "dataset.uavr")
        self.assertEqual(meta["splits"]["test"], 4)
        self.assertTrue(meta["collision_filter"]["drop_collision_steps"])
        self.assertEqual(meta["collision_dropped"], 1)

    def test_forest_version_accepted_without_frozen_flag(self):
        # forest_e1 与柱阵同约定「版本串 = salt」；episode manifest 不带 split_salt，按 geometry_version 复算。
        man = self.man(self.eps[0])
        man.update(split="train", geometry_version=sf.GEOMETRY_VERSION_FOREST,
                   geometry_gid=gid_for("train", "forest", sf.GEOMETRY_VERSION_FOREST))
        self.save(self.eps[0], man)
        r = self.run_pre(self.tmp / "ds_forest", [self.eps[0]])
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_forest" / "dataset.uavr")
        self.assertEqual(meta["geometry_version"], sf.GEOMETRY_VERSION_FOREST)
        self.assertEqual(meta["split_scheme"], "geometry_gid")
        self.assertEqual(meta["split_salt"], sf.GEOMETRY_VERSION_FOREST)
        self.assertFalse(meta["mixed_geometry"])
        self.assertEqual(list(meta["geometry_versions"]), [sf.GEOMETRY_VERSION_FOREST])
        self.assertEqual(meta["geometry_versions"][sf.GEOMETRY_VERSION_FOREST]["n_episodes"], 1)

    def _make_three_families(self):
        """ep1 柱阵、ep2 森林、ep3 3DGS，各按本族 salt 取 gid；hold/window 一致（混包只放开族身份键）。"""
        base, forest, gs = self.eps
        for ep, gver, policy, recorded_salt, split in (
                (base, sm.SPLIT_SALT, None, None, "train"),
                (forest, sf.GEOMETRY_VERSION_FOREST, None, None, "train"),
                (gs, "3dgs.v1", "geometry_gid", "3dgs.v1", "val")):
            man = self.man(ep)
            man.update(split=split, geometry_version=gver, hold_s=1.0, window_from="observation",
                       geometry_gid=gid_for(split, ep.name, recorded_salt or gver))
            if policy:
                man.update(split_policy=policy, split_salt=recorded_salt, task_version="3dgs.pilot.v1",
                           stop_radius=0.5)
            self.save(ep, man)

    def test_mixed_geometry_requires_flag_and_records_families(self):
        self._make_three_families()
        r = self.run_pre(self.tmp / "ds_mixed", self.eps)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("geometry_version", r.stderr)
        self.assertIn("--allow-mixed-geometry", r.stderr)
        r = self.run_pre(self.tmp / "ds_mixed", self.eps, "--allow-mixed-geometry")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_mixed" / "dataset.uavr")
        self.assertTrue(meta["mixed_geometry"])
        self.assertTrue(meta["geometry_version"].startswith("mixed:"), meta["geometry_version"])
        # C++ 读端只认 split_policy == split_scheme ∈ 四种方案；三族都按 geometry_gid 划分才能混。
        self.assertEqual(meta["split_policy"], "geometry_gid")
        self.assertEqual(meta["split_scheme"], "geometry_gid")
        self.assertIsNone(meta["split_salt"])   # 三族 salt 不同，顶层留空、逐族在 geometry_versions
        fams = meta["geometry_versions"]
        self.assertEqual(sorted(fams), sorted([sm.SPLIT_SALT, sf.GEOMETRY_VERSION_FOREST, "3dgs.v1"]))
        self.assertEqual({v: f["n_episodes"] for v, f in fams.items()},
                         {sm.SPLIT_SALT: 1, sf.GEOMETRY_VERSION_FOREST: 1, "3dgs.v1": 1})
        self.assertEqual({v: f["family_split_salt"] for v, f in fams.items()},
                         {sm.SPLIT_SALT: sm.SPLIT_SALT, sf.GEOMETRY_VERSION_FOREST: sf.GEOMETRY_VERSION_FOREST,
                          "3dgs.v1": "3dgs.v1"})
        self.assertEqual(sum(f["n_samples"] for f in fams.values()), meta["num_samples"])
        # C++ 读端按子串找 "episodes" 数组、取 split_* 的第一次出现：逐族表不能带 "episodes" 键，
        # 顶层 split_* 必须排在 episodes 数组之前（episode 项里本来就带 split_policy）。
        raw = (self.tmp / "ds_mixed" / "dataset.uavr").read_bytes()
        self.assertEqual(raw.count(b'"episodes":'), 1)
        for key in ("split_policy", "split_scheme", "split_salt"):
            self.assertLess(raw.find(('"%s":' % key).encode()), raw.find(b'"episodes":'), key)
        self.assertEqual(meta["splits"], {"train": 10, "val": 5, "test": 0})
        self.assertIsNone(meta["task_version"])   # 族间不一致的口径字段顶层为空

    def test_mixed_geometry_keeps_hold_and_camera_strict(self):
        self._make_three_families()
        man = self.man(self.eps[2])
        man["hold_s"] = 0.5   # 3DGS 历史口径：每步位移不同，混包也不放行
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds_hold", self.eps, "--allow-mixed-geometry")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("hold_s", r.stderr)
        self._make_three_families()
        for ep, prof in ((self.eps[0], "rgb320x240_fov60_v1"), (self.eps[1], "rgb320x240_fov60_v1")):
            man = self.man(ep)
            man["camera_profile"] = prof
            self.save(ep, man)
        # 一族缺 camera_profile（3DGS pilot 没写）：混包放行但打警告。
        r = self.run_pre(self.tmp / "ds_cam", self.eps, "--allow-mixed-geometry")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("camera_profile", r.stderr)
        # 两种不同的相机口径仍不能混。
        man = self.man(self.eps[2])
        man["camera_profile"] = "rgb640x480_fov90_v1"
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds_cam2", self.eps, "--allow-mixed-geometry")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("camera_profile", r.stderr)

    def test_legacy_episodes_record_scene_id_scheme(self):
        r = self.run_pre(self.tmp / "ds_legacy", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_legacy" / "dataset.uavr")
        self.assertEqual(meta["split_policy"], "scene_id")
        self.assertEqual(meta["split_scheme"], "scene_id")
        self.assertIsNone(meta["split_salt"])
        self.assertEqual(meta["split_groups"], [])


if __name__ == "__main__":
    unittest.main()
