#!/usr/bin/env python3
"""audit_features.py 的测试（卡 E2E-0）：合成两根柱的场景上核对列距离、n、hist、选步与对账函数。

不读真实采集、不起 Gazebo。坐标约定同 rule_policy / teacher：x 前、y 左，yaw 度（0 = +x，正 = 左转），方位正 = 左，列从左到右排。
运行（仓库根；numpy + Pillow 即可，teacher-venv 或 .venv-tools 的 python 都行）：
  TMPDIR=$D/tmp/phase12-cards/e2e-0/tmp PYTHONDONTWRITEBYTECODE=1 python experiments/2026-11-12-decidability-audit/test_audit_features.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audit_features as af  # noqa: E402

CAM = 0.045   # e1_mini.json camera_mount.xyz[0]：相机在机体前 0.045 m


def square(x, y, size=0.4):
    return {"name": "p", "x": x, "y": y, "z": 1.5, "size": [size, size, 3.0]}


class ColumnFeatureTests(unittest.TestCase):
    def test_shapes_and_narrow_is_center_slice_of_wide(self):
        # 两根柱：A 正前方 2.0 m，B 在左前 (1.0, 1.2)，从相机看方位约 41°–62°，落在 60° 视场（±30°）之外
        pillars = [square(2.0, 0.0), square(1.0, 1.2)]
        d, wide = af.column_features(0.0, 0.0, 0.0, pillars)
        self.assertEqual((len(d), len(wide)), (20, 60))
        self.assertEqual(list(d), list(wide[20:40]))
        # 与库函数直接算 60° 的结果逐位相同：切片不是另一套实现
        self.assertEqual(list(d), af.rp.column_distances(0.0, 0.0, 0.0, pillars, hfov_deg=60.0))

    def test_front_pillar_center_columns(self):
        d, _ = af.column_features(0.0, 0.0, 0.0, [square(2.0, 0.0)])
        # 柱前表面 x = 1.8，相机 x = 0.045：正前方射线 1.755 m；列 9、10 的 0° 边缘射线打在柱面中线
        self.assertAlmostEqual(d[9], 1.8 - CAM, places=9)
        self.assertAlmostEqual(d[10], 1.8 - CAM, places=9)
        self.assertTrue(all(v == af.MAX_RANGE for v in d[:4] + d[16:]), "柱半角约 6° 以内，|方位| ≥ 12° 的列看不到它")

    def test_wide_sees_the_pillar_the_narrow_camera_cannot(self):
        side = [square(1.0, 1.2)]
        d, wide = af.column_features(0.0, 0.0, 0.0, side)
        self.assertEqual(list(d), [af.MAX_RANGE] * 20, "60° 视场里没有它")
        seen = [j for j, v in enumerate(wide) if v < af.MAX_RANGE]
        self.assertTrue(seen, "180° 视场看得到")
        # 列 j 的方位 = (29.5 − j)·3°（正 = 左）：柱 B 占方位 41°–62°，即列 9…15 之间
        self.assertTrue(all(8 <= j <= 16 for j in seen), seen)
        self.assertEqual(sum(1 for j in seen if j >= 20), 0, "右半边一列都不该变短")

    def test_mirror_scene_reverses_columns(self):
        _, left = af.column_features(0.0, 0.0, 0.0, [square(1.0, 1.2)])
        _, right = af.column_features(0.0, 0.0, 0.0, [square(1.0, -1.2)])
        self.assertEqual(list(left), list(right)[::-1])

    def test_yaw_left_turns_a_front_pillar_to_the_right_side(self):
        _, wide0 = af.column_features(0.0, 0.0, 0.0, [square(2.0, 0.0)])
        _, wide1 = af.column_features(0.0, 0.0, 30.0, [square(2.0, 0.0)])   # 机头左转 30°：柱在机头右前
        near0 = min(range(60), key=lambda j: wide0[j])
        near1 = min(range(60), key=lambda j: wide1[j])
        self.assertTrue(near1 > near0, "柱的方位从 0° 变成约 −30°，列号变大（列从左到右排）")


class HistoryTests(unittest.TestCase):
    F, L, R, S = af.FWD, af.LEFT, af.RIGHT, af.SAFE

    def test_card_example_f_l_l_f_r(self):
        h = af.executed_history([self.F, self.L, self.L, self.F, self.R])
        self.assertEqual(list(h["n_raw"]), [0, 0, 1, 2, 2], "n = 该步之前已执行的左转数减右转数")
        self.assertEqual(list(h["prev"]), [0, 1, 2, 2, 1], "首步没有上一步，按 SAFE_NEUTRAL（与 preprocess 一致）")
        self.assertEqual(list(h["last_dir"]), [0, 0, 1, 1, 1])
        self.assertEqual(list(h["since"]), [5, 5, 1, 1, 2])
        self.assertEqual([af.heading_code(n) for n in h["n_raw"]], [0, 0, 0, 2, 2], "W1-9 卡的例子：码序列 0 0 0 2 2")

    def test_n_clip_and_code(self):
        h = af.executed_history([self.L] * 7)
        self.assertEqual(list(h["n_raw"]), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(list(af.clip_n(h["n_raw"])), [0, 1, 2, 3, 4, 4, 4], "截到 ±4")
        self.assertEqual(list(af.clip_n([-9, -4, 0, 4, 9])), [-4, -4, 0, 4, 4])
        self.assertEqual([af.heading_code(n) for n in (-3, -2, -1, 0, 1, 2, 3)], [3, 3, 0, 0, 0, 2, 2])

    def test_right_turns_count_negative_and_cancel(self):
        h = af.executed_history([self.R, self.R, self.L, self.F])
        self.assertEqual(list(h["n_raw"]), [0, -1, -2, -1])
        self.assertEqual(list(h["last_dir"]), [0, 2, 2, 1], "最近一次转向：右 → 右 → 左")

    def test_since_is_capped_at_5(self):
        h = af.executed_history([self.L] + [self.F] * 8)
        self.assertEqual(list(h["since"]), [5, 1, 2, 3, 4, 5, 5, 5, 5])
        self.assertEqual(list(h["last_dir"]), [0] + [1] * 8)

    def test_safe_is_not_a_turn(self):
        h = af.executed_history([self.S, self.S, self.F])
        self.assertEqual(list(h["n_raw"]), [0, 0, 0])
        self.assertEqual(list(h["last_dir"]), [0, 0, 0])
        self.assertEqual(list(h["since"]), [5, 5, 5])


def write_episode(root, name, rows, pillars, split="train", executed_by=None, extra=None):
    """合成一集：manifest 满足 preprocess.load_episode 的校验，PNG 只需存在（load_episode 不读像素）。"""
    ep = os.path.join(root, name)
    os.makedirs(ep)
    man = {"schema": "navlab.episode.v1", "episode_id": name, "scene_id": name, "seed": 1, "version": "1",
           "split": split, "geometry_version": af.pp.sm.SPLIT_SALT, "teacher": "plan_action",
           "actions_ref": "actions.jsonl", "num_steps": len(rows),
           "camera": {"width": 320, "height": 240},
           "validity": {"camera_verified": True, "physics_verified": True, "episode_executed": True}}
    if executed_by:
        man["executed_by"] = executed_by
    man.update(extra or {})
    with open(os.path.join(ep, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    with open(os.path.join(ep, "scene.json"), "w", encoding="utf-8") as f:
        json.dump({"pillars": pillars}, f)
    with open(os.path.join(ep, "actions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        open(os.path.join(ep, "rgb_%05d.png" % r["step"]), "wb").close()
    return ep


def row(step, label, executed=None, x=0.0, y=0.0, yaw=0.0, collision=False, **kw):
    r = {"step": step, "action_expert": label, "action_executed": label if executed is None else executed,
         "collision": collision, "pose_before": {"x": x, "y": y, "z": 2.0}, "yaw_before": yaw}
    r.update(kw)
    return r


class EpisodeRecordTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_collision_step_dropped_but_still_counts_in_history(self):
        pillars = [square(3.0, 0.0), square(2.0, 1.0)]
        # 步 3 撞了（标签 FORWARD、实际执行 YAW_RIGHT），被丢弃；它飞过，所以后面各步的历史量要把它算进去
        rows = [row(0, af.FWD, x=0.0), row(1, af.LEFT, executed=af.LEFT, x=0.25),
                row(2, af.LEFT, executed=af.LEFT, x=0.3, yaw=26.0),
                row(3, af.FWD, executed=af.RIGHT, x=0.35, yaw=52.0, collision=True),
                row(4, af.RIGHT, executed=af.RIGHT, x=0.5, yaw=26.0),
                row(5, af.FWD, executed=af.FWD, x=0.6, yaw=0.0)]
        ep = write_episode(self.root, "e1", rows, pillars)
        rec = af.episode_records(ep)
        self.assertEqual(list(rec["step"]), [0, 1, 2, 4, 5], "--drop-collision：碰撞步不进训练集（与 preprocess 同一个函数）")
        self.assertEqual(rec["n_rows"], 6)
        self.assertEqual(rec["n_dropped"], 1)
        # 只在保留的步上算会得到 n_raw [0,0,1,2,1]、prev [0,1,2,2,3]、last_dir [0,0,1,1,2]；按全部行算才是下面这组
        self.assertEqual(list(rec["n_raw"]), [0, 0, 1, 1, 0], "步 3 的右转算已执行：步 4 看到 n=1，步 5 看到 n=0")
        self.assertEqual(list(rec["prev"]), [0, 1, 2, 3, 3], "步 4 的上一步是被丢弃的步 3（右转）")
        self.assertEqual(list(rec["last_dir"]), [0, 0, 1, 2, 2])
        self.assertEqual(list(rec["since"]), [5, 5, 1, 1, 1])
        self.assertEqual(list(rec["label"]), [1, 2, 2, 3, 1])
        self.assertEqual(rec["d_wide"].shape, (5, 60))
        self.assertEqual(rec["d_wide"].dtype.name, "float32")
        self.assertEqual(float(rec["x_rem"][0]), 3.0 - 0.0, "到最远柱的纵向距离 = 最远柱 x − 机体 x（柱中心）")
        self.assertEqual(float(rec["y"][1]), 0.0)

    def test_empty_frame_flag_and_wide_values(self):
        far = [square(30.0, 0.0)]
        ep = write_episode(self.root, "e2", [row(0, af.FWD), row(1, af.FWD, x=0.25)], far)
        rec = af.episode_records(ep)
        self.assertTrue(bool((rec["d_wide"][:, 20:40] == af.MAX_RANGE).all()), "4 m 内没有障碍：20 列全是上限")

    def test_executed_by_codes_and_header(self):
        rows = [row(0, af.FWD, executed_by="student"), row(1, af.LEFT, executed_by="expert"), row(2, af.FWD)]
        ep = write_episode(self.root, "e3", rows, [square(2.0, 0.0)], split="val", executed_by="student",
                           extra={"dagger_round": 1, "beta": 0.5, "geometry_gid": None})
        rec = af.episode_records(ep)
        self.assertEqual(list(rec["exec_code"]), [af.EXEC_STUDENT, af.EXEC_EXPERT, af.EXEC_TEACHER])
        self.assertEqual((rec["split"], rec["episode_id"], rec["dagger_round"], rec["beta"]), ("val", "e3", 1, 0.5))

    def test_rejects_gap_in_steps(self):
        ep = write_episode(self.root, "e4", [row(0, af.FWD), row(2, af.FWD)], [square(2.0, 0.0)],
                           extra={"num_steps": 2})
        with self.assertRaises(ValueError):
            af.episode_records(ep)


PACK_LOG = """警告: episodes 的 scene.json 不一致，未复制到数据集根目录
episodes: 3  samples: 7  scenes: 3
splits: train=4 val=3 test=0
split: policy=geometry_gid scheme=geometry_gid salt=uavrgb.geom.v5
  train {'SAFE_NEUTRAL': 0, 'FORWARD_SLOW': 3, 'YAW_LEFT': 1, 'YAW_RIGHT': 0}
  val   {'SAFE_NEUTRAL': 1, 'FORWARD_SLOW': 1, 'YAW_LEFT': 0, 'YAW_RIGHT': 1}
  test  {'SAFE_NEUTRAL': 0, 'FORWARD_SLOW': 0, 'YAW_LEFT': 0, 'YAW_RIGHT': 0}
collision: dropped 2 of 9 steps (22.2%), kept 7
  episode_id=a scene_id=a split=train steps=4 sample_start=0
  episode_id=b scene_id=b split=val steps=2 sample_start=4
  episode_id=c scene_id=c split=val steps=1 sample_start=6
"""


class PackLogTests(unittest.TestCase):
    def test_parse_pack_log(self):
        log = af.parse_pack_log(PACK_LOG)
        self.assertEqual((log["episodes"], log["samples"], log["dropped"], log["seen"]), (3, 7, 2, 9))
        self.assertEqual(log["splits"], {"train": 4, "val": 3, "test": 0})
        self.assertEqual(log["class_counts"]["val"], [1, 1, 0, 1])
        self.assertEqual(log["per_episode"], {"a": ("train", 4), "b": ("val", 2), "c": ("val", 1)})

    def _summary(self, **over):
        s = {"a": {"split": "train", "retained": 4, "total": 5, "class_counts": [0, 3, 1, 0]},
             "b": {"split": "val", "retained": 2, "total": 3, "class_counts": [1, 0, 0, 1]},
             "c": {"split": "val", "retained": 1, "total": 1, "class_counts": [0, 1, 0, 0]}}
        for k, v in over.items():
            s[k] = {**s[k], **v}
        return s

    def test_check_against_log_passes_when_everything_matches(self):
        res = af.check_against_log(self._summary(), af.parse_pack_log(PACK_LOG))
        self.assertTrue(res["ok"], res)

    def test_check_against_log_catches_count_label_and_split_mismatch(self):
        log = af.parse_pack_log(PACK_LOG)
        for over in ({"a": {"retained": 3}}, {"b": {"class_counts": [0, 1, 0, 1]}}, {"c": {"split": "test"}}):
            res = af.check_against_log(self._summary(**over), log)
            self.assertFalse(res["ok"], over)
            self.assertTrue(res["problems"], over)

    def test_check_against_log_catches_missing_episode(self):
        s = self._summary()
        del s["c"]
        res = af.check_against_log(s, af.parse_pack_log(PACK_LOG))
        self.assertFalse(res["ok"])
        self.assertTrue(any("不在提取结果里" in p for p in res["problems"]), res["problems"])

    def test_check_against_log_catches_swapped_splits_even_when_totals_agree(self):
        # 两集步数与类别数相同、split 对调：各 split 的合计与类别数都对得上，只有逐集比对能发现
        text = ("episodes: 2  samples: 4\nsplits: train=2 val=2 test=0\n"
                "  train {'SAFE_NEUTRAL': 0, 'FORWARD_SLOW': 2, 'YAW_LEFT': 0, 'YAW_RIGHT': 0}\n"
                "  val   {'SAFE_NEUTRAL': 0, 'FORWARD_SLOW': 2, 'YAW_LEFT': 0, 'YAW_RIGHT': 0}\n"
                "  test  {'SAFE_NEUTRAL': 0, 'FORWARD_SLOW': 0, 'YAW_LEFT': 0, 'YAW_RIGHT': 0}\n"
                "collision: dropped 0 of 4 steps (0.0%), kept 4\n"
                "  episode_id=x scene_id=x split=train steps=2 sample_start=0\n"
                "  episode_id=y scene_id=y split=val steps=2 sample_start=2\n")
        mk = lambda sp: {"split": sp, "retained": 2, "total": 2, "class_counts": [0, 2, 0, 0]}  # noqa: E731
        parsed = af.parse_pack_log(text)
        self.assertTrue(af.check_against_log({"x": mk("train"), "y": mk("val")}, parsed)["ok"])
        res = af.check_against_log({"x": mk("val"), "y": mk("train")}, parsed)
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["problems"]), 2, res["problems"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
