#!/usr/bin/env python3
"""preprocess 的 split 事实/诊断门禁/冻结复算/碰撞过滤回归（纯 CPU，临时目录）。

用 make_synthetic_episode 生成 3 条小 episode，再改写 manifest；不读真实数据。
"""
import hashlib
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

    def test_goal_stopped_short_episode_packs_and_tiebreak_passthrough(self):
        """collect.py --stop-at-goal 的短集（num_steps < requested_steps）无需改 preprocess 即可打包；
        manifest.teacher_tiebreak 透传进 episodes 表，旧 manifest 没有该字段时为 null。"""
        ep1 = self.eps[0]
        rows = (ep1 / "actions.jsonl").read_text(encoding="utf-8").splitlines()[:3]
        (ep1 / "actions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        man = self.man(ep1)
        man.update({"num_steps": 3, "requested_steps": 6, "stopped_on_goal": True,
                    "goal_line_x": 5.97, "teacher_tiebreak": "visible"})
        self.save(ep1, man)
        for ep in self.eps:
            m = self.man(ep)
            m["geometry_gid"] = gid_for(m["split"], ep.name)
            self.save(ep, m)
        r = self.run_pre(self.tmp / "ds", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds" / "dataset.uavr")
        by_dir = {Path(e["source_dir"]).name: e for e in meta["episodes"]}
        # ep1 前 3 行没有碰撞行，3 步全进包；train = ep1 + ep2（默认丢碰撞行后的样本数）。
        self.assertEqual(by_dir["ep1"]["num_steps"], 3)
        self.assertEqual(meta["splits"]["train"], 3 + by_dir["ep2"]["num_steps"])
        self.assertEqual(by_dir["ep1"]["teacher_tiebreak"], "visible")
        self.assertIsNone(by_dir["ep2"]["teacher_tiebreak"])
        self.assertEqual(by_dir["ep2"]["executed_by"], "teacher")   # 老 manifest 的 DAgger 默认值不变

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

    # ---- UAVRAUX1 侧车（STUDENT-P2） ----
    def read_aux(self, ds_dir):
        b = (ds_dir / "dataset.aux").read_bytes()
        magic, n, flags, dh, dw, k, unk, _ = struct.unpack_from("<8sQIHHBBH", b, 0)
        self.assertEqual((magic, k, unk), (b"UAVRAUX1", 8, 255))
        self.assertEqual(hashlib.sha256((ds_dir / "dataset.uavr").read_bytes()).digest(), b[28:60])
        self.assertEqual(hashlib.sha256(b[128:]).digest(), b[60:92])
        prev_index = struct.unpack_from("<%dI" % n, b, 128)
        prev_action = b[128 + 4 * n:128 + 5 * n]
        depth = b[128 + 5 * n:] if flags & 1 else None
        self.assertEqual(len(b), 128 + 5 * n + (n * dh * dw if depth is not None else 0))
        return prev_index, prev_action, depth, (dh, dw)

    def test_aux_sidecar_prev_chain_without_depth(self):
        # 合成集没有深度：侧车只有 prev 链；首帧 0/6/10 指向自身且 prev=SAFE，其余 prev=上一步标签。
        r = self.run_pre(self.tmp / "ds", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds" / "dataset.uavr")
        self.assertEqual(meta["aux_sidecar"]["depth"], False)
        prev_index, prev_action, depth, grid = self.read_aux(self.tmp / "ds")
        self.assertIsNone(depth)
        self.assertEqual(grid, (0, 0))
        starts = {0: 0, 6: 6, 10: 10}
        for i in range(15):
            first = i in starts
            self.assertEqual(prev_index[i], i if first else i - 1)
            local_prev = 0 if first else (i - 1) - max(s for s in starts if s <= i - 1)
            self.assertEqual(prev_action[i], 0 if first else local_prev % 4, i)

    def test_aux_sidecar_drop_collision_breaks_chain(self):
        # ep1 的 step 2 是碰撞步并被 --drop-collision 丢掉：step 3 的上一帧指向自身（当首帧），
        # 但 prev-action 仍取 actions.jsonl 里 step 2 真正执行的动作（它确实发生过）。
        ep = self.eps[0]
        actions = ep / "actions.jsonl"
        rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
        for row in rows:
            row["collision"] = row["step"] == 2
            row["action_executed"] = (row["action_expert"] + 1) % 4  # 与标签不同，验证取的是 executed
        actions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        r = self.run_pre(self.tmp / "ds", self.eps, "--drop-collision")
        self.assertEqual(r.returncode, 0, r.stderr)
        prev_index, prev_action, _, _ = self.read_aux(self.tmp / "ds")
        # ep1 剩 5 样本：step 0,1,3,4,5 → 全局 0..4。
        self.assertEqual(prev_index[:5], (0, 0, 2, 2, 3))
        self.assertEqual(list(prev_action[:5]), [0, (0 + 1) % 4, (2 + 1) % 4, (3 + 1) % 4, (4 + 1) % 4])

    def write_depth(self, fmt, make_frame):
        """给三条 episode 写深度帧：fmt = png16_mm（uint16 毫米 PNG）或 npy_float32_m（float32 米 npy）；
        make_frame(step) 返回 (240, 320) 数组。manifest.depth_format 与 actions 的 depth_file 一起写。"""
        import numpy as np
        from PIL import Image
        ext = {"png16_mm": "png", "npy_float32_m": "npy"}[fmt]
        for ep in self.eps:
            man = self.man(ep)
            man["depth_format"] = fmt
            self.save(ep, man)
            (ep / "depth").mkdir(exist_ok=True)
            actions = ep / "actions.jsonl"
            rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
            for row in rows:
                name = "depth/depth_%05d.%s" % (row["step"], ext)
                frame = make_frame(row["step"])
                if fmt == "png16_mm":
                    Image.fromarray(np.asarray(frame, dtype=np.uint16)).save(ep / name)
                else:
                    np.save(ep / name, np.asarray(frame, dtype=np.float32))
                row["depth_file"] = name
            actions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def bands_of(self, ds_dir):
        import numpy as np
        _, _, depth, grid = self.read_aux(ds_dir)
        self.assertEqual(grid, (15, 20))
        return np.frombuffer(depth, dtype=np.uint8).reshape(-1, 15, 20)

    def test_aux_sidecar_depth_zero_semantics_by_source(self):
        # 深度 0 按来源区分（manifest.depth_format）：Gazebo png16 的 0 = 没打到东西（天空/超远）→ 最远带 7；
        # 3DGS npy 的 0 = alpha 空洞 → 未知 255。两种都整帧全 0，膨胀开着也不该改变结果（没有近处格可扩散）。
        import numpy as np
        self.write_depth("png16_mm", lambda step: np.zeros((240, 320)))
        r = self.run_pre(self.tmp / "ds_gz", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.bands_of(self.tmp / "ds_gz") == 7).all(), "Gazebo 全 0 → 全部第 8 带")
        self.write_depth("npy_float32_m", lambda step: np.zeros((240, 320)))
        r = self.run_pre(self.tmp / "ds_3dgs", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((self.bands_of(self.tmp / "ds_3dgs") == 255).all(), "3DGS 全 0 → 全部未知")
        # 3DGS 里 nan / 负值同样是未知；有效像素照常分带，且格内 nan 不拉低/抬高 min。
        def mixed(step):
            d = np.full((240, 320), 3.0, dtype=np.float32)
            d[:16, :] = np.nan
            d[16:32, :] = -1.0
            d[32:48, ::2] = np.nan  # 半格 nan：另一半 3 m 仍给出带 6
            return d
        self.write_depth("npy_float32_m", mixed)
        r = self.run_pre(self.tmp / "ds_mixed", self.eps, "--envelope-radius", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        b = self.bands_of(self.tmp / "ds_mixed")
        self.assertTrue((b[:, 0, :] == 255).all() and (b[:, 1, :] == 255).all(), "nan / 负值 → 未知")
        self.assertTrue((b[:, 2:, :] == 6).all(), "3 m（含半格 nan）→ 带 6")

    def test_aux_sidecar_depth_bands_and_all_or_nothing(self):
        # 三条 episode 写 16 位毫米深度：左半 0.4 m（带 1）、右半 3 m（带 6）、顶部一行 0 = Gazebo 没打到 → 带 7。
        # 包络膨胀关掉（--envelope-radius 0）以便逐格核对。
        import numpy as np

        def frame(step):
            d = np.full((240, 320), 3000, dtype=np.uint16)
            d[:, :160] = 400
            d[:16, :] = 0
            return d
        self.write_depth("png16_mm", frame)
        r = self.run_pre(self.tmp / "ds", self.eps, "--envelope-radius", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds" / "dataset.uavr")
        self.assertEqual(meta["aux_sidecar"]["depth"], True)
        self.assertEqual(meta["aux_sidecar"]["depth_grid"], [15, 20])
        bands = self.bands_of(self.tmp / "ds")
        self.assertTrue((bands[:, 0, :] == 7).all(), "Gazebo 顶行 0 = 天空 → 最远带 7")
        self.assertTrue((bands[:, 1:, :10] == 1).all(), "0.4 m → 带 1 (0.3..0.5)")
        self.assertTrue((bands[:, 1:, 10:] == 6).all(), "3 m → 带 6 (2.5..4)")
        # 膨胀：0.4 m 处 r=0.102 m 对应 f·r/z ≈ 277*0.102/0.4 ≈ 71 px ≈ 4.4 格，右侧 3 m 区会被近处带 1 侵蚀。
        r = self.run_pre(self.tmp / "ds_dil", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        bands_d = self.bands_of(self.tmp / "ds_dil")
        self.assertTrue((bands_d[:, 1:, 10:14] == 1).all(), "包络膨胀把带 1 推进右侧 ≥4 格")
        self.assertTrue((bands_d[:, 1:, 16:] == 6).all(), "远处格仍是带 6")
        self.assertTrue((bands_d[:, 0, :10] == 1).all() and (bands_d[:, 0, 16:] == 7).all(),
                        "顶行天空格被近处带 1 覆盖，远离近处的仍是带 7（inf 本身不向外膨胀）")
        # 部分 episode 缺深度 → 拒绝（一个包要么全有要么全无）。
        ep = self.eps[1]
        actions = ep / "actions.jsonl"
        rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
        for row in rows:
            row.pop("depth_file", None)
        actions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        r = self.run_pre(self.tmp / "ds_partial", self.eps)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("部分缺失", r.stderr)

    def test_aux_sidecar_dilation_uses_each_episodes_focal(self):
        # FIXLIST_R2 B1：包络膨胀的焦距按每条 episode 自己的 camera.fx 取，不能拿分块里第一条的
        # 焦距套给整块——否则多相机混包时一半样本的膨胀半径是错的，而写出来的侧车看不出来。
        # ep1 给极小 fx=50（半径 50·0.102/0.4/16 ≈ 0.8 格 < 1 → 不膨胀），ep2/ep3 保持默认 277 → ≈4.4 格。
        import numpy as np

        def frame(step):
            d = np.full((240, 320), 3000, dtype=np.uint16)
            d[:, :160] = 400
            return d
        self.write_depth("png16_mm", frame)
        man = self.man(self.eps[0])
        man["camera"]["fx"] = 50.0
        self.save(self.eps[0], man)
        r = self.run_pre(self.tmp / "ds_focal", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        bands = self.bands_of(self.tmp / "ds_focal")
        narrow = set()
        for ep in read_meta(self.tmp / "ds_focal" / "dataset.uavr")["episodes"]:
            if Path(ep["source_dir"]).resolve() == self.eps[0].resolve():
                narrow.update(range(ep["sample_start"], ep["sample_start"] + ep["sample_count"]))
        self.assertTrue(narrow and len(narrow) < bands.shape[0], "ep1 应只占一部分样本")
        wide = sorted(set(range(bands.shape[0])) - narrow)
        self.assertTrue((bands[sorted(narrow)][:, :, 10:] == 6).all(), "fx=50 的 episode 不该膨胀")
        self.assertTrue((bands[wide][:, :, 10:14] == 1).all(), "默认焦距的 episode 仍膨胀 ≥4 格")

    def test_dilate_by_envelope_bounded_by_grid(self):
        # 一个极近的格（3DGS 深度 1e-6 m：相机贴着/钻进 splat）会让膨胀半径算到 1.8e6 格；
        # 偏移循环若按该半径走是 (2r+1)^2 ≈ 1e13 次，preprocess 卡死不报错。
        # 超出网格的偏移本就不落任何格，循环上界只能是网格尺寸；结果与原算法逐格相同。
        import signal
        import numpy as np
        sys.path.insert(0, str(REPO / "tools"))
        import preprocess as pre
        cells = np.full((2, 15, 20), 5.0, dtype=np.float32)
        cells[0, 7, 10] = 1e-6
        cells[1, 0, 0] = 0.4  # 普通近处格：半径 ≈ 4.4 格，只影响附近

        def on_alarm(*_):
            raise TimeoutError("dilate_by_envelope 没在 10 s 内结束（偏移循环没按网格截断）")
        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(10)
        try:
            out = pre.dilate_by_envelope(cells, 277.128, 16.0, 0.102)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
        self.assertTrue((out[0] == np.float32(1e-6)).all(), "极近格覆盖整张网格")
        self.assertEqual(out[1, 0, 0], np.float32(0.4))
        self.assertEqual(out[1, 0, 4], np.float32(0.4), "4 格内被近处格覆盖")
        self.assertEqual(out[1, 0, 5], np.float32(5.0), "5 格外不受影响")
        self.assertEqual(out[1, 14, 19], np.float32(5.0))

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

    def test_mixed_dynamics_requires_flag_and_records_profiles(self):
        """DAgger 并集：round0（lib v1，cf_sitl_v1:bd0dcd43）+ 新轮次（v3.1，683e7400）。默认仍拒绝（STRICT_KEYS）；
        --allow-mixed-dynamics 才放行，meta 记 dynamics_profiles 全部值并在 stdout 告警。部分集缺 profile 仍拒绝。"""
        for ep, prof in zip(self.eps, ("cf_sitl_v1:bd0dcd43", "cf_sitl_v1:bd0dcd43", "cf_sitl_v1:683e7400")):
            man = self.man(ep)
            man.update(dynamics_profile=prof, geometry_gid=gid_for(man["split"], ep.name))
            self.save(ep, man)
        r = self.run_pre(self.tmp / "ds_dyn", self.eps)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dynamics_profile", r.stderr)
        r = self.run_pre(self.tmp / "ds_dyn", self.eps, "--allow-mixed-dynamics")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dynamics_profile", r.stdout)   # 告警行
        meta = read_meta(self.tmp / "ds_dyn" / "dataset.uavr")
        self.assertEqual(meta["dynamics_profiles"], ["cf_sitl_v1:683e7400", "cf_sitl_v1:bd0dcd43"])
        self.assertFalse(meta["mixed_geometry"])   # 只放开了动力学档，几何族仍是一族
        # 单一 profile 时也写（单元素），开关不影响其它口径键
        man = self.man(self.eps[2])
        man["dynamics_profile"] = "cf_sitl_v1:bd0dcd43"
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds_one", self.eps, "--allow-mixed-dynamics")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(read_meta(self.tmp / "ds_one" / "dataset.uavr")["dynamics_profiles"], ["cf_sitl_v1:bd0dcd43"])
        # 部分集缺 dynamics_profile：开关也不放行（缺字段的集动力学未知）
        man = self.man(self.eps[2])
        del man["dynamics_profile"]
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds_missing", self.eps, "--allow-mixed-dynamics")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("dynamics_profile", r.stderr)

    def test_action_latency_is_a_strict_protocol_key(self):
        """W1-0：--action-latency 采的集（manifest.action_latency_s）与旧集、与不同 L 的集都不能混包（STRICT_KEYS，
        与 hold_s / window_from 同级）；同一 L 才打包，meta 顶层记它；旧数据 meta 不多这个键。"""
        for ep in self.eps:
            man = self.man(ep)
            man.update(action_latency_s=0.16, geometry_gid=gid_for(man["split"], ep.name))
            self.save(ep, man)
        r = self.run_pre(self.tmp / "ds_same", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_same" / "dataset.uavr")
        self.assertEqual(meta["action_latency_s"], 0.16)
        self.assertEqual({e["action_latency_s"] for e in meta["episodes"]}, {0.16})
        man = self.man(self.eps[2])
        man["action_latency_s"] = 0.20
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds_two_values", self.eps)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("action_latency_s", r.stderr)
        man = self.man(self.eps[2])
        del man["action_latency_s"]
        self.save(self.eps[2], man)
        r = self.run_pre(self.tmp / "ds_partial", self.eps)       # 一集是旧口径：部分缺也拒绝
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("action_latency_s", r.stderr)
        r = self.run_pre(self.tmp / "ds_partial_mixed", self.eps, "--allow-mixed-geometry")   # 族开关不放开口径键
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("action_latency_s", r.stderr)
        for ep in self.eps:                                        # 全是旧口径：照常打包，meta 没有这个键
            man = self.man(ep)
            man.pop("action_latency_s", None)
            self.save(ep, man)
        r = self.run_pre(self.tmp / "ds_legacy_only", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_legacy_only" / "dataset.uavr")
        self.assertNotIn("action_latency_s", meta)
        self.assertTrue(all("action_latency_s" not in e for e in meta["episodes"]))

    def test_legacy_episodes_record_scene_id_scheme(self):
        r = self.run_pre(self.tmp / "ds_legacy", self.eps)
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds_legacy" / "dataset.uavr")
        self.assertEqual(meta["split_policy"], "scene_id")
        self.assertEqual(meta["split_scheme"], "scene_id")
        self.assertIsNone(meta["split_salt"])
        self.assertEqual(meta["split_groups"], [])

    def test_build_meta_directly_rejects_cross_split_geometry_gid(self):
        """I2-9 第 2 点：绕过 load_episode 与 scan（如 airsim_to_uavr 直接构造 episode 字典调 write_dataset -> build_meta）
        时，build_meta 自身的第三层几何隔离门必须拦住同 geometry_gid 跨 split，并报出独有文案「几何组 … 跨 split」。"""
        sys.path.insert(0, str(REPO / "tools"))
        import preprocess as pre

        def fake_ep(eid, split, gid, start):
            return {
                "dir": "/fake/ep%d" % eid,
                "episode_id": eid,
                "scene_id": eid,
                "seed": eid,
                "version": "synthetic",
                "split": split,
                "geometry_gid": gid,
                "geometry_version": sm.SPLIT_SALT,
                "src_w": 320,
                "src_h": 240,
                "num_samples": 1,
                "sample_start": start,
                "samples": [{"label": 1, "collision": False}],
            }

        ok_meta = pre.build_meta(
            [fake_ep(1, "train", "gid_train_only", 0), fake_ep(2, "val", "gid_val_only", 1)],
            total=2, size=(160, 120))
        self.assertEqual(ok_meta["split_groups"], [
            {"geometry_gid": "gid_train_only", "split": "train"},
            {"geometry_gid": "gid_val_only", "split": "val"},
        ])

        with self.assertRaises(SystemExit) as cm:
            pre.build_meta(
                [fake_ep(1, "train", "shared_gid_x", 0), fake_ep(2, "val", "shared_gid_x", 1)],
                total=2, size=(160, 120))
        msg = str(cm.exception)
        self.assertIn("几何组 shared_gid_x 跨 split（train 与 val）", msg)
        self.assertNotIn("同几何必须同 split", msg)
        self.assertNotIn("按记录策略应属的 split", msg)


if __name__ == "__main__":
    unittest.main()
