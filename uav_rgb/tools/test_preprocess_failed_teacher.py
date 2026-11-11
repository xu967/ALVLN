#!/usr/bin/env python3
"""preprocess --exclude-failed-teacher（第十二阶段 W1-7 的 B1）回归（纯 CPU，临时目录）。

背景（docs/reviews/2026-11-10/data-algo.md §A1）：0417 采集里有 26 集教师自己没过成功线（最远柱 x + 1.0）；
这些集的后半段标签是撞柱前后的挣扎与 SAFE，学生学到的是失败轨迹。开关默认关；打开后，
策略是教师的集，oracle.jsonl 里 pose_gt.x 的最大值小于「最远柱 x + 1.0」就整集跳过，并打印跳过几集。
带 waypoint 的任务、DAgger 学生自采集、缺 scene.json 柱子或 oracle 的集不适用，不跳。

用 make_synthetic_episode 生成 3 条小 episode，再写 scene.json 与 oracle；不读真实数据。
"""
import itertools
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim" / "tools"))
sys.path.insert(0, str(REPO / "tools"))
import split_manifest as sm  # noqa: E402

PY = sys.executable
# 一根柱 x=0.5 → 成功线 1.5；合成集的 ep1/ep2/ep3 步数 6/4/5。
PILLAR = {"name": "p0", "x": 0.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}
LINE_X = 1.5
STEPS = {"ep1": 6, "ep2": 4, "ep3": 5}


def read_meta(path):
    with open(path, "rb") as f:
        head = f.read(48)
        meta_off = struct.unpack_from("<Q", head, 36)[0]
        meta_len = struct.unpack_from("<I", head, 44)[0]
        f.seek(meta_off)
        return json.loads(f.read(meta_len))


def gid_for(split, tag="g"):
    for i in range(10000):
        gid = "%s%d" % (tag, i)
        if sm.split_for_gid(gid) == split:
            return gid
    raise AssertionError("找不到 %s split 的 gid" % split)


class FailedTeacherTests(unittest.TestCase):
    def setUp(self):
        self._out_n = itertools.count()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")],
                       check=True, capture_output=True)
        self.eps = [self.tmp / "ep" / n for n in ("ep1", "ep2", "ep3")]
        # ep1 过线（max x 2.0）、ep2 没过线（max x 1.0）、ep3 max x 正好等于成功线（不算失败）。
        self.set_max_x(self.eps[0], 2.0)
        self.set_max_x(self.eps[1], 1.0)
        self.set_max_x(self.eps[2], LINE_X)
        for ep in self.eps:
            self.write_scene(ep, [PILLAR])
            man = self.man(ep)
            man["geometry_gid"] = gid_for(man["split"], ep.name)
            self.save(ep, man)

    def tearDown(self):
        self._tmp.cleanup()

    # ---- 夹具 ----
    def man(self, ep):
        return json.loads((ep / "episode_manifest.json").read_text(encoding="utf-8"))

    def save(self, ep, man):
        (ep / "episode_manifest.json").write_text(json.dumps(man), encoding="utf-8")

    def write_scene(self, ep, pillars):
        (ep / "scene.json").write_text(json.dumps({"pillars": pillars}), encoding="utf-8")

    def set_max_x(self, ep, max_x):
        """oracle.jsonl 的 pose_gt.x 线性爬升，最后一步到 max_x。"""
        path = ep / "oracle.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        n = len(rows)
        for i, row in enumerate(rows):
            row["pose_gt"]["x"] = max_x * (i + 1) / n
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def run_pre(self, out, *flags, eps=None):
        return subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--out", str(out), *flags,
                               *map(str, eps or self.eps)], capture_output=True, text=True)

    def meta_of(self, out):
        return read_meta(Path(out) / "dataset.uavr")

    def names(self, meta):
        return sorted(Path(e["source_dir"]).name for e in meta["episodes"])

    # ---- 缺省关：行为不变 ----
    def test_default_off_packs_everything_and_adds_no_meta_keys(self):
        r = self.run_pre(self.tmp / "ds")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(self.tmp / "ds")
        self.assertEqual(self.names(meta), ["ep1", "ep2", "ep3"])
        self.assertEqual(meta["collision_filter"], {
            "drop_collision_steps": False, "exclude_collision_episodes": False, "excluded_episodes": 0})
        self.assertEqual(meta["splits"], {"train": 10, "val": 5, "test": 0})
        self.assertNotIn("failed_teacher", (self.tmp / "ds" / "dataset.uavr").read_bytes().decode("latin-1"))
        self.assertNotIn("--exclude-failed-teacher", r.stdout + r.stderr)

    # ---- 打开：跳过没过线的集 ----
    def test_flag_skips_only_the_episode_below_the_line(self):
        r = self.run_pre(self.tmp / "ds", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("跳过整集 1 个（--exclude-failed-teacher", r.stdout + r.stderr)
        meta = self.meta_of(self.tmp / "ds")
        # ep2（max x 1.0 < 1.5）被跳过；ep3（max x == 1.5）保留，边界不算失败
        self.assertEqual(self.names(meta), ["ep1", "ep3"])
        self.assertEqual(meta["splits"], {"train": 6, "val": 5, "test": 0})
        cf = meta["collision_filter"]
        self.assertTrue(cf["exclude_failed_teacher"])
        self.assertEqual(cf["failed_teacher_skipped"], 1)
        self.assertEqual(cf["failed_teacher_goal_margin_m"], 1.0)
        self.assertEqual(cf["excluded_episodes"], 0)               # 碰撞开关的计数只数碰撞整集
        self.assertFalse(cf["exclude_collision_episodes"])

    def test_success_line_is_the_farthest_pillar_plus_margin(self):
        far = dict(PILLAR, name="p1", x=1.0)                         # 两根柱：最远柱 x=1.0 → 成功线 2.0
        for ep in self.eps:
            self.write_scene(ep, [PILLAR, far])
        r = self.run_pre(self.tmp / "ds", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(self.tmp / "ds")
        self.assertEqual(self.names(meta), ["ep1"])                 # ep1 max x 2.0 到线；ep3 的 1.5 不到 2.0
        self.assertEqual(meta["collision_filter"]["failed_teacher_skipped"], 2)

    def test_message_is_printed_even_when_nothing_is_skipped(self):
        for ep in self.eps:
            self.set_max_x(ep, 3.0)
        r = self.run_pre(self.tmp / "ds", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("跳过整集 0 个（--exclude-failed-teacher", r.stdout + r.stderr)
        self.assertEqual(self.meta_of(self.tmp / "ds")["collision_filter"]["failed_teacher_skipped"], 0)

    # ---- 不适用：不跳 ----
    def assert_kept_as_not_applicable(self, expect_na=1):
        out = self.tmp / ("ds_na%d" % next(self._out_n))
        r = self.run_pre(out, "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(out)
        self.assertEqual(meta["collision_filter"]["failed_teacher_skipped"], 0, r.stdout)
        self.assertEqual(self.names(meta), ["ep1", "ep2", "ep3"])
        self.assertIn("%d 集不适用" % expect_na, r.stdout + r.stderr)

    def test_waypoint_task_is_not_applicable(self):
        # waypoint 任务的 stop_radius 是族身份键，全包必须一致：三集都是 waypoint 任务；ep2 的 oracle 照样没过线，也不跳
        for ep in self.eps:
            man = self.man(ep)
            man["waypoint"] = {"x": 1.0, "y": 0.0}
            man["stop_radius"] = 0.5
            self.save(ep, man)
        self.assert_kept_as_not_applicable(expect_na=3)

    def test_dagger_student_executed_episode_is_not_applicable(self):
        man = self.man(self.eps[1])
        man.update({"executed_by": "student", "dagger_round": 1, "beta": 0.0})
        self.save(self.eps[1], man)
        self.assert_kept_as_not_applicable()

    def test_missing_scene_or_pillars_or_oracle_is_not_applicable(self):
        for ep in self.eps:
            (ep / "scene.json").unlink()                              # 全部缺 scene.json（打包要求 scene_json 同口径）
        r = self.run_pre(self.tmp / "ds_noscene", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.names(self.meta_of(self.tmp / "ds_noscene")), ["ep1", "ep2", "ep3"])
        self.assertIn("3 集不适用", r.stdout + r.stderr)
        for ep in self.eps:
            self.write_scene(ep, [])                                  # 有 scene.json 但没有柱子
        self.assert_kept_as_not_applicable(expect_na=3)
        for ep in self.eps:
            self.write_scene(ep, [PILLAR])
        (self.eps[1] / "oracle.jsonl").write_text("", encoding="utf-8")   # oracle 空
        self.assert_kept_as_not_applicable()
        (self.eps[1] / "oracle.jsonl").unlink()                          # oracle 缺
        self.assert_kept_as_not_applicable()

    # ---- 与碰撞开关并用 ----
    def test_collision_episode_is_counted_by_the_collision_flag_first(self):
        # ep2 既没过线又有碰撞：两个开关都开时记在碰撞计数里，不重复记
        actions = self.eps[1] / "actions.jsonl"
        rows = [json.loads(line) for line in actions.read_text().splitlines() if line]
        for row in rows:
            row["collision"] = row["step"] == 0
        actions.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        r = self.run_pre(self.tmp / "ds", "--exclude-collision-episodes", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        cf = self.meta_of(self.tmp / "ds")["collision_filter"]
        self.assertEqual((cf["excluded_episodes"], cf["failed_teacher_skipped"]), (1, 0))
        self.assertIn("跳过整集 1 个（--exclude-collision-episodes）", r.stdout + r.stderr)
        # 再让 ep1 也没过线（无碰撞）：碰撞开关与失败开关各数各的
        self.set_max_x(self.eps[0], 0.5)
        r = self.run_pre(self.tmp / "ds2", "--exclude-collision-episodes", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(self.tmp / "ds2")
        cf = meta["collision_filter"]
        self.assertEqual((cf["excluded_episodes"], cf["failed_teacher_skipped"]), (1, 1))
        self.assertEqual(self.names(meta), ["ep3"])
        # 只开失败开关：ep2（有碰撞但不带 --exclude-collision-episodes）照样因没过线被跳
        r = self.run_pre(self.tmp / "ds3", "--exclude-failed-teacher")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(self.tmp / "ds3")
        self.assertEqual(self.names(meta), ["ep3"])
        self.assertEqual(meta["collision_filter"]["failed_teacher_skipped"], 2)

    def test_all_episodes_skipped_is_an_error_not_an_empty_dataset(self):
        for ep in self.eps:
            self.set_max_x(ep, 0.5)
        r = self.run_pre(self.tmp / "ds", "--exclude-failed-teacher")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("没有可写样本", r.stdout + r.stderr)

    # ---- 口径与 collect/eval 同一来源 ----
    def test_goal_margin_equals_collect_and_eval(self):
        import collect
        import preprocess as pre
        self.assertEqual(pre.FAILED_TEACHER_GOAL_MARGIN_M, collect.GOAL_MARGIN_M)


if __name__ == "__main__":
    unittest.main()
