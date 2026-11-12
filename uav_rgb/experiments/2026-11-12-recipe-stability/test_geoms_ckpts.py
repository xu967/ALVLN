"""select_val_geoms.py 与 make_shifted_ckpts.py 的测试。含用真实 scene_gen 与真实 cal_swa 权重的集成测试（文件不在就跳过，不静默通过）。

运行：cd experiments/2026-11-12-recipe-stability && <解释器> -m unittest test_geoms_ckpts
"""
import json
import os
import tempfile
import unittest

import numpy as np

import make_shifted_ckpts as mk
import select_val_geoms as sv

D = "/root/rivermind-data/uav_rgb"
SWA = os.path.join(D, "exp-swa-calib-20261110", "ckpt")
FROZEN = [30012, 30043, 30084, 30094, 30115, 30119, 30130, 30131, 30168, 30189, 30263, 30384, 30400, 30409, 30420, 30475, 30627, 30632, 30776, 30944]
FROZEN_SHA = "5058a33b8b0387e812e693c438d4edaec70de6ef357c80aaf19d67a9134a24ab"


class TestPick(unittest.TestCase):
    def test_pick_basic(self):
        seeds, gids = sv.pick(set(), 30000, 31000, 5)
        self.assertEqual(len(seeds), 5)
        self.assertEqual(seeds, sorted(seeds))
        self.assertEqual(len(set(gids)), 5)
        self.assertTrue(all(sv.sm.split_for_seed(s, "base", None) == "val" for s in seeds))

    def test_pick_excludes_gids(self):
        seeds, gids = sv.pick(set(), 30000, 31000, 5)
        seeds2, gids2 = sv.pick({gids[0], gids[1]}, 30000, 31000, 5)
        self.assertNotIn(gids[0], gids2)
        self.assertNotIn(gids[1], gids2)
        self.assertEqual(seeds2[0], seeds[2])                  # 前两个被排除，从第三个开始

    def test_pick_same_gid_twice_counts_once(self):
        # 高位 seed 会重复低位几何：同一个 gid 只取第一次出现
        seeds, gids = sv.pick(set(), 0, 3000, 40)
        self.assertEqual(len(set(gids)), 40)

    def test_pick_too_few_exits(self):
        with self.assertRaises(SystemExit):
            sv.pick(set(), 30000, 30010, 20)


class TestManifestRules(unittest.TestCase):
    def test_counts_table(self):
        v5 = sv.sm.SPLIT_SALT
        self.assertTrue(sv.manifest_counts({"task": "base", "geometry_version": v5}))
        self.assertTrue(sv.manifest_counts({}))                                               # 没记任务与版本：保守地算
        self.assertFalse(sv.manifest_counts({"task": "l1", "geometry_version": v5}))
        self.assertFalse(sv.manifest_counts({"task": "forest_e1", "geometry_version": "uavrgb.forest_e1.v1"}))
        self.assertFalse(sv.manifest_counts({"task": "base", "geometry_version": "uavrgb.geom.v3"}))
        self.assertFalse(sv.manifest_counts({"split_scheme": "geometry group: sha256(uavrgb.geom.v3:<gid>) % 100"}))
        self.assertFalse(sv.manifest_counts({"split_scheme": "scene_id % 10 < 8 -> train"}))
        self.assertTrue(sv.manifest_counts({"task": "base", "geometry_version": v5, "split_scheme": "geometry group: sha256(<SPLIT_SALT>:<gid>) % 100"}))


class TestExclusions(unittest.TestCase):
    def test_build_exclusions_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            for sp, s in (("train", 1), ("val", 2), ("test", 3)):
                os.makedirs(os.path.join(td, sv.DATASET_COL, sp, str(s)))
            def put(rel, obj):
                p = os.path.join(td, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(obj, f)
            put("a/collection_manifest.json", {"task": "base", "geometry_version": sv.sm.SPLIT_SALT, "splits": {"train": [5], "val": [7]}})
            put("b/collection_manifest.json", {"task": "l1", "geometry_version": sv.sm.SPLIT_SALT, "splits": {"train": [6]}})
            put("c/collection_manifest.json", {"splits": {"train": [8]}, "split_scheme": "sha256(uavrgb.geom.v3:<gid>)"})
            put("d/eval.json", {"task": "base", "geometry_version": sv.sm.SPLIT_SALT, "per_episode": [{"seed": 11, "geometry_gid": "feedfacefeedface"}]})
            put("e/eval.json", {"task": "l1", "geometry_version": sv.sm.SPLIT_SALT, "per_episode": [{"seed": 12, "geometry_gid": "0123456789abcdef"}]})
            put("code/x/eval.json", {"task": "base", "geometry_version": sv.sm.SPLIT_SALT, "per_episode": [{"seed": 13, "geometry_gid": "cafecafecafecafe"}]})   # 冻结代码目录被跳过
            gids, seeds, src = sv.build_exclusions(td)
            for s in (1, 2, 3, 5, 7):
                self.assertIn(sv.gid_of(s), gids)
            self.assertIn("feedfacefeedface", gids)
            self.assertNotIn("0123456789abcdef", gids)         # l1 的 eval 不取 gid
            self.assertNotIn("cafecafecafecafe", gids)         # code/ 下的不扫
            self.assertEqual(src["dataset"]["episodes"], 3)
            self.assertEqual(src["collection_manifests"]["counted_as_base_v5"], 1)
            self.assertEqual(src["collection_manifests"]["files"], 3)
            for s_ in (6, 8, 11, 12):                                # seed 区间核对用的全集：l1 / 旧版本清单 / 别的任务 eval 的 seed 也在
                self.assertIn(s_, seeds)
            self.assertNotIn(13, seeds)                              # code/ 下的仍然不扫

    def test_find_files_prunes(self):
        with tempfile.TemporaryDirectory() as td:
            for rel in ("x/eval.json", "x/y/z/eval.json", "code/eval.json", "x/episodes/seed_1/eval.json", "a/b/c/d/e/f/g/eval.json"):
                p = os.path.join(td, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").close()
            got = [os.path.relpath(p, td) for p in sv.find_files(td, "eval.json", 4)]
            self.assertEqual(sorted(got), ["x/eval.json", "x/y/z/eval.json"])


class TestFrozenList(unittest.TestCase):
    def test_frozen_check_ok(self):
        ok, rec = sv.frozen_check(FROZEN)
        self.assertTrue(ok, rec)
        self.assertEqual(rec["sha256"], FROZEN_SHA)
        self.assertEqual(rec["distinct_gids"], 20)

    def test_frozen_check_rejects(self):
        ok, rec = sv.frozen_check(FROZEN[:3] + [1])                  # seed 1 是冻结 test seed 之一
        self.assertFalse(ok)
        self.assertIn(1, rec["not_val"])
        self.assertTrue(rec["seeds_in_eval_seed_lists"])
        ok, rec = sv.frozen_check(FROZEN[:3] + FROZEN[:1])           # 重复 seed
        self.assertFalse(ok)

    def test_scene_gen_agrees(self):
        chk = sv.verify_with_scene_gen(FROZEN[:3], [sv.gid_of(s) for s in FROZEN[:3]])
        self.assertTrue(all(c["ok"] for c in chk), chk)

    def test_scene_gen_detects_wrong_gid(self):
        chk = sv.verify_with_scene_gen(FROZEN[:1], ["0000000000000000"])
        self.assertFalse(chk[0]["ok"])


@unittest.skipUnless(os.path.exists(os.path.join(SWA, "cal_swa_s0_k10.bin")), "没有 cal_swa_s0_k10.bin")
class TestShifted(unittest.TestCase):
    def test_shift_tag(self):
        self.assertEqual(mk.shift_tag(-0.5), "fwdm0.5")
        self.assertEqual(mk.shift_tag(0.5), "fwdp0.5")
        self.assertEqual(mk.shift_tag(0), "fwd0")

    def test_make_all_real(self):
        with tempfile.TemporaryDirectory() as td:
            rec = mk.make_all(SWA, td)
            self.assertEqual(len(rec["files"]), 9)                    # 3 个种子 × (0 档原文件 + 两个平移档)
            for r in rec["files"]:
                src = mk.ac.read_ckpt(os.path.join(SWA, mk.SRC_NAME.format(seed=r["seed"])))
                if r["delta"] == 0:
                    self.assertEqual(r["sha256"], mk.ac.sha256_file(os.path.join(SWA, mk.SRC_NAME.format(seed=r["seed"]))))
                    continue
                new = mk.ac.read_ckpt(r["path"])
                self.assertEqual(new["raw"][:mk.ac.CK_HEAD.size], src["raw"][:mk.ac.CK_HEAD.size])   # 头原样
                for i, (a, b) in enumerate(zip(src["params"], new["params"])):
                    if i == mk.ac.FC_B:
                        self.assertAlmostEqual(float(b[mk.ac.FORWARD] - a[mk.ac.FORWARD]), r["delta"], delta=1e-6)
                        others = [j for j in range(a.size) if j != mk.ac.FORWARD]
                        self.assertTrue((a[others] == b[others]).all())          # 另外三个类的偏置一个比特都没动
                    else:
                        self.assertTrue(np.array_equal(a, b))
                self.assertEqual(mk.ac.sha256_file(r["path"]), r["sha256"])
            with open(os.path.join(td, "shifts.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["shifts"], [-0.5, 0.5])
            self.assertFalse(any(n.startswith(".zero-check") for n in os.listdir(td)))   # 自己建的零平移核对文件已删

    def test_check_shift_catches_tampering(self):
        src = mk.ac.read_ckpt(os.path.join(SWA, "cal_swa_s0_k10.bin"))
        with tempfile.TemporaryDirectory() as td:
            nb = mk.write_shifted(src, 0.5, os.path.join(td, "x.bin"))
            mk.check_shift(src, nb, 0.5)
            bad = bytearray(nb)
            bad[200] ^= 0x01                                           # 别处动一个比特
            with self.assertRaises(ValueError):
                mk.check_shift(src, bytes(bad), 0.5)
            with self.assertRaises(ValueError):
                mk.check_shift(src, nb, -0.5)                          # 平移量对不上
            with self.assertRaises(ValueError):
                mk.check_shift(src, nb, 0.0)                           # 声称平移 0 却有改动


if __name__ == "__main__":
    unittest.main()
