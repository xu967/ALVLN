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
        # ep3（val 分区唯一一集）未到 2.0 被跳过 → 默认因 val 跳空报错；显式 --allow-empty-split 才放行
        r_err = self.run_pre(self.tmp / "ds_err", "--exclude-failed-teacher")
        self.assertNotEqual(r_err.returncode, 0)
        self.assertIn("split=val", r_err.stderr)
        self.assertIn("--allow-empty-split", r_err.stderr)
        r = self.run_pre(self.tmp / "ds", "--exclude-failed-teacher", "--allow-empty-split")
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
            (ep / "scene.json").write_text('{"other": 1}', encoding="utf-8")  # 非柱阵任务无 pillars 键：不适用
        self.assert_kept_as_not_applicable(expect_na=3)
        for ep in self.eps:
            self.write_scene(ep, [PILLAR])
        (self.eps[1] / "oracle.jsonl").write_text("", encoding="utf-8")   # oracle 空
        self.assert_kept_as_not_applicable()
        (self.eps[1] / "oracle.jsonl").unlink()                          # oracle 缺
        self.assert_kept_as_not_applicable()

    def test_corrupted_oracle_jsonl_is_an_error_with_file_and_line(self):
        # I2-4a / TEST 2a：未过线的 ep2 若 oracle.jsonl 末尾有半截损坏 JSON，不能静默当「不适用」打进训练集
        ora = self.eps[1] / "oracle.jsonl"
        orig = ora.read_text(encoding="utf-8")
        ora.write_text(orig + '{"step": 3, "pose_gt": {"x": 0.9', encoding="utf-8")
        r = self.run_pre(self.tmp / "ds_2a", "--exclude-failed-teacher")
        self.assertNotEqual(r.returncode, 0, "损坏的 oracle.jsonl 必须报错停下")
        self.assertIn(str(ora), r.stderr)
        self.assertIn("第 5 行", r.stderr)
        self.assertFalse((self.tmp / "ds_2a" / "dataset.uavr").exists())

        lines = orig.splitlines()
        for bad_line in ("[1, 2, 3]", '{"step": 1, "pose_gt": "bad"}',
                         '{"step": 1, "pose_gt": {"x": "bad"}}', '{"step": 1, "pose_gt": {"x": true}}'):
            ora.write_text(lines[0] + "\n" + bad_line + "\n" + "\n".join(lines[2:]) + "\n", encoding="utf-8")
            r_bad = self.run_pre(self.tmp / "ds_bad_ora", "--exclude-failed-teacher")
            self.assertNotEqual(r_bad.returncode, 0, bad_line)
            self.assertIn(str(ora), r_bad.stderr)
            self.assertIn("第 2 行", r_bad.stderr)

    def test_corrupted_scene_json_is_an_error_with_file_and_line(self):
        # I2-4a：scene.json 存在但语法损坏或 pillars 字段非法时，报错指明文件与行号
        sc = self.eps[1] / "scene.json"
        sc.write_text('{\n  "pillars": [\n', encoding="utf-8")
        r = self.run_pre(self.tmp / "ds_bad_sc", "--exclude-failed-teacher")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(str(sc), r.stderr)
        self.assertIn("第 3 行", r.stderr)

        for bad_doc in ('[]', '{"pillars": "bad"}', '{"pillars": [{"name": "p0"}]}',
                        '{"pillars": [{"x": "bad"}]}', '{"pillars": [{"x": true}]}'):
            sc.write_text(bad_doc, encoding="utf-8")
            r_bad = self.run_pre(self.tmp / "ds_bad_sc2", "--exclude-failed-teacher")
            self.assertNotEqual(r_bad.returncode, 0, bad_doc)
            self.assertIn(str(sc), r_bad.stderr)
            self.assertRegex(r_bad.stderr, r"第 \d+ 行")

    def test_nonfinite_oracle_or_pillar_x_is_an_error_with_file_and_line(self):
        # I2-4b / TEST 2b：过线集 ep1 首行 pose_gt.x=NaN 时不能污染 max_x 误删整集，必须报错指明文件与行号
        ora = self.eps[0] / "oracle.jsonl"
        orig_lines = [json.loads(ln) for ln in ora.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for bad_val, bad_idx in ((float("nan"), 0), (float("inf"), 2), (float("-inf"), 2)):
            rows = [json.loads(json.dumps(r)) for r in orig_lines]
            rows[bad_idx]["pose_gt"]["x"] = bad_val
            ora.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            r = self.run_pre(self.tmp / "ds_2b", "--exclude-failed-teacher")
            self.assertNotEqual(r.returncode, 0, f"oracle pose_gt.x={bad_val} 必须报错")
            self.assertIn(str(ora), r.stderr)
            self.assertIn("第 %d 行" % (bad_idx + 1), r.stderr)
        ora.write_text("".join(json.dumps(r) + "\n" for r in orig_lines), encoding="utf-8")

        sc = self.eps[0] / "scene.json"
        for bad_px in (float("nan"), float("inf")):
            bad_pillar = dict(PILLAR, x=bad_px)
            sc.write_text(json.dumps({"pillars": [bad_pillar]}), encoding="utf-8")
            r_sc = self.run_pre(self.tmp / "ds_2b_sc", "--exclude-failed-teacher")
            self.assertNotEqual(r_sc.returncode, 0, f"pillars[0].x={bad_px} 必须报错")
            self.assertIn(str(sc), r_sc.stderr)
            self.assertIn("第 1 行", r_sc.stderr)

    def test_truncated_actions_jsonl_is_caught_before_episode_filters(self):
        # I2-4c / TEST 2c：ep2 的 actions.jsonl 截断（2 行 != num_steps=4），开 --exclude-failed-teacher 或
        # --exclude-collision-episodes 也必须先报文件截断，不能静默按整集过滤跳过
        act = self.eps[1] / "actions.jsonl"
        rows = [json.loads(ln) for ln in act.read_text(encoding="utf-8").splitlines() if ln.strip()][:2]
        rows[0]["collision"] = True
        act.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        for flag in ("--exclude-failed-teacher", "--exclude-collision-episodes"):
            r = self.run_pre(self.tmp / "ds_2c", flag)
            self.assertNotEqual(r.returncode, 0, f"开启 {flag} 时截断 actions.jsonl 仍须报错")
            self.assertIn(str(act), r.stderr)
            self.assertIn("第 2 行", r.stderr)
            self.assertIn("actions.jsonl 2 行 != manifest num_steps 4", r.stderr)

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
        # 再让 ep1 也没过线（无碰撞）：train 分区被全部跳空，默认报错；加 --allow-empty-split 后两个开关各数各的
        self.set_max_x(self.eps[0], 0.5)
        r_err = self.run_pre(self.tmp / "ds2_err", "--exclude-collision-episodes", "--exclude-failed-teacher")
        self.assertNotEqual(r_err.returncode, 0)
        self.assertIn("split=train", r_err.stderr)
        r = self.run_pre(self.tmp / "ds2", "--exclude-collision-episodes", "--exclude-failed-teacher", "--allow-empty-split")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(self.tmp / "ds2")
        cf = meta["collision_filter"]
        self.assertEqual((cf["excluded_episodes"], cf["failed_teacher_skipped"]), (1, 1))
        self.assertEqual(self.names(meta), ["ep3"])
        # 只开失败开关：ep2（有碰撞但不带 --exclude-collision-episodes）照样因没过线被跳（train 空，需显式开关）
        r = self.run_pre(self.tmp / "ds3", "--exclude-failed-teacher", "--allow-empty-split")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = self.meta_of(self.tmp / "ds3")
        self.assertEqual(self.names(meta), ["ep3"])
        self.assertEqual(meta["collision_filter"]["failed_teacher_skipped"], 2)

    def test_empty_train_or_val_after_filtering_is_rejected_unless_flag_is_passed(self):
        # I2-4d：val 被 --exclude-failed-teacher 跳空，或 train 被 --drop-collision 清空时，必须报错并指明文件与行号；
        # test 分区被跳空且 train/val 非空时不拦。
        self.set_max_x(self.eps[2], 0.5)  # ep3 是唯一的 val 集
        r_val = self.run_pre(self.tmp / "ds_val0", "--exclude-failed-teacher")
        self.assertNotEqual(r_val.returncode, 0)
        self.assertIn(str(self.eps[2] / "oracle.jsonl"), r_val.stderr)
        self.assertIn("第 1 行", r_val.stderr)
        self.assertIn("split=val", r_val.stderr)
        self.assertIn("--allow-empty-split", r_val.stderr)
        self.assertFalse((self.tmp / "ds_val0" / "dataset.uavr").exists())
        r_val_ok = self.run_pre(self.tmp / "ds_val_ok", "--exclude-failed-teacher", "--allow-empty-split")
        self.assertEqual(r_val_ok.returncode, 0, r_val_ok.stderr)
        self.assertEqual(self.meta_of(self.tmp / "ds_val_ok")["splits"], {"train": 6, "val": 0, "test": 0})

        # 恢复 ep3 过线，把 train 的 ep1、ep2 全部标为 collision=True，验证行级过滤清空 train 同样被拦
        self.set_max_x(self.eps[2], 2.0)
        for ep in self.eps[:2]:
            act = ep / "actions.jsonl"
            rows = [json.loads(ln) for ln in act.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for row in rows:
                row["collision"] = True
            act.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        r_tr = self.run_pre(self.tmp / "ds_tr0", "--drop-collision")
        self.assertNotEqual(r_tr.returncode, 0)
        self.assertIn(str(self.eps[0] / "actions.jsonl"), r_tr.stderr)
        self.assertIn("第 1 行", r_tr.stderr)
        self.assertIn("split=train", r_tr.stderr)
        r_tr_ok = self.run_pre(self.tmp / "ds_tr_ok", "--drop-collision", "--allow-empty-split")
        self.assertEqual(r_tr_ok.returncode, 0, r_tr_ok.stderr)
        self.assertEqual(self.meta_of(self.tmp / "ds_tr_ok")["splits"]["train"], 0)

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
