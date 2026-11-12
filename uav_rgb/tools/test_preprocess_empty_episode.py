#!/usr/bin/env python3
"""preprocess 行级过滤整集淘空回归测试（第十二阶段卡 PK1，审核 I2-6 第 1 点；纯 CPU，临时目录）。

覆盖：
  1. --drop-collision 把两个 episode 的全部步滤空且带 instruction 时（TEST 2d）：被淘空的集不进 meta.episodes、
     不进 meta.scenes、不计入 geometry_versions[*].n_episodes、不写入 instructions.jsonl（无 sample_count=0 重叠区间），
     单独计数并打印「跳过整集 N 个（行级过滤后剩余 0 步）」，collision_dropped 照实累加被丢步数。
  2. --drop-late-and-stall 把整集全部步滤空时同样跳过并打印，timing_dropped 照实累加。
  3. 所有输入集均被行级过滤淘空时（total == 0）报错「没有可写样本」，不写出空数据集。
"""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "tools"))
sys.path.insert(0, str(REPO / "tools"))
import preprocess as pre  # noqa: E402
import split_manifest as sm  # noqa: E402

PY = sys.executable
HEADER = struct.Struct("<4sHHIHHBBBBQQQI")


def gid_for(split, tag="g"):
    for i in range(10000):
        gid = "%s_%d" % (tag, i)
        if sm.split_for_gid(gid) == split:
            return gid
    raise AssertionError(split)


def read_meta(uavr_path: Path):
    raw = uavr_path.read_bytes()
    _, _, _, n, _, _, _, _, _, _, _, _, mo, ml = HEADER.unpack_from(raw, 0)
    return n, json.loads(raw[mo:mo + ml].decode("utf-8"))


class EmptyAfterStepFilterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")],
                       check=True, capture_output=True)
        # 构造 4 集：ep1(train,6步), ep2(train,4步), ep3(train,5步), ep4(val,5步)，每集独立 scene_id 与 gid，均带合法指令
        shutil.copytree(self.tmp / "ep/ep3", self.tmp / "ep/ep4")
        self.eps = [self.tmp / "ep" / n for n in ("ep1", "ep2", "ep3", "ep4")]
        specs = [("ep1", 1, 101, "train"), ("ep2", 2, 102, "train"),
                 ("ep3", 3, 103, "train"), ("ep4", 4, 104, "val")]
        for ep, (name, eid, sid, sp) in zip(self.eps, specs):
            mp = ep / "episode_manifest.json"
            man = json.loads(mp.read_text(encoding="utf-8"))
            man.update({
                "episode_id": eid,
                "scene_id": sid,
                "split": sp,
                "geometry_gid": gid_for(sp, name),
                "instruction": "fly to the red pillar and stop",
                "window_from": "observation",
                "hold_s": 1.0,
                "action_latency_s": 0.2,
            })
            mp.write_text(json.dumps(man), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def set_all_collision(self, ep: Path):
        ap = ep / "actions.jsonl"
        rows = [json.loads(ln) for ln in ap.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for r in rows:
            r["collision"] = True
        ap.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def set_all_late_or_stall(self, ep: Path):
        ap = ep / "actions.jsonl"
        rows = [json.loads(ln) for ln in ap.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for i, r in enumerate(rows):
            r["late_dispatch"] = (i % 2 == 0)
            r["obs_gap_s"] = None if i == 0 else (1.8 if i % 2 == 1 else 0.9)
        ap.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_drop_collision_emptied_episodes_excluded_from_meta_and_instructions(self):
        # TEST 2d：ep2(4步) 与 ep3(5步) 整集碰撞滤空，且全包带 instruction
        self.set_all_collision(self.eps[1])
        self.set_all_collision(self.eps[2])
        out = self.tmp / "ds_coll"
        r = subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--out", str(out), "--drop-collision",
                            *map(str, self.eps)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("跳过整集 2 个（行级过滤后剩余 0 步）", r.stdout)
        n, meta = read_meta(out / "dataset.uavr")
        self.assertEqual(n, 11)  # ep1(6) + ep4(5)
        self.assertEqual(meta["collision_dropped"], 9)  # ep2(4) + ep3(5)
        self.assertEqual([e["episode_id"] for e in meta["episodes"]], [1, 4])
        self.assertTrue(all(e["sample_count"] > 0 for e in meta["episodes"]))
        self.assertEqual([s["scene_id"] for s in meta["scenes"]], [101, 104])
        self.assertEqual(meta["geometry_versions"][sm.SPLIT_SALT]["n_episodes"], 2)

        ins = [json.loads(ln) for ln in (out / "instructions.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual([(x["episode_id"], x["sample_start"], x["sample_count"]) for x in ins],
                         [(1, 0, 6), (4, 6, 5)])

    def test_drop_late_and_stall_emptied_episodes_excluded_and_counted_in_skip_stats(self):
        for ep in self.eps:
            ap = ep / "actions.jsonl"
            rows = [json.loads(ln) for ln in ap.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for i, r in enumerate(rows):
                r["late_dispatch"] = False
                r["obs_gap_s"] = None if i == 0 else 0.9
            ap.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.set_all_late_or_stall(self.eps[1])
        self.set_all_late_or_stall(self.eps[2])

        stats = {}
        eps, total, dropped, skipped = pre.scan(
            [str(e) for e in self.eps], drop_late_and_stall=True, skip_stats=stats)
        self.assertEqual(total, 11)
        self.assertEqual(dropped, 0)
        self.assertEqual(skipped, 0)
        self.assertEqual(stats.get("emptied_by_step_filter"), 2)
        self.assertEqual(stats.get("timing_dropped"), 9)
        self.assertEqual([e["episode_id"] for e in eps], [1, 4])

    def test_all_episodes_emptied_by_step_filter_fails_with_no_samples(self):
        for ep in self.eps:
            self.set_all_collision(ep)
        out = self.tmp / "ds_all_empty"
        r = subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--out", str(out), "--drop-collision",
                            *map(str, self.eps)], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("跳过整集 4 个（行级过滤后剩余 0 步）", r.stdout + r.stderr)
        self.assertIn("没有可写样本", r.stdout + r.stderr)
        self.assertFalse((out / "dataset.uavr").exists())


if __name__ == "__main__":
    unittest.main()
