#!/usr/bin/env python3
"""航向码 nav_cond 的纯 CPU 回归（任务卡 W1-9）：heading_code 的边界与首帧、动作序列 F L L F R 的码序列、
一致性闸门的判定表与「训练时条件类型」的读取（临时文件，不读真实数据，不起 Gazebo）。

运行（仓库根）：.venv-tools/bin/python sim/tools/test_nav_cond.py
"""
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_cond  # noqa: E402

FWD, LEFT, RIGHT = 1, 2, 3   # FORWARD_SLOW / YAW_LEFT / YAW_RIGHT


def write_dataset(path, meta):
    """最小 UAVR 文件：48 字节头（meta_off 在 36、meta_len 在 44）+ meta JSON；够 read_dataset_meta 读，不含图像。"""
    body = json.dumps(meta).encode("utf-8")
    head = struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, 0, 120, 160, 3, 1, 1, 0, 48, 48, 48, len(body))
    with open(path, "wb") as f:
        f.write(head + body)


class HeadingCodeTests(unittest.TestCase):
    def test_boundaries(self):
        for n, want in ((0, 0), (1, 0), (-1, 0), (2, 2), (3, 2), (10, 2), (-2, 3), (-3, 3), (-10, 3)):
            self.assertEqual(nav_cond.heading_code(n), want, f"n={n}")

    def test_code_one_is_never_produced(self):
        self.assertNotIn(1, {nav_cond.heading_code(n) for n in range(-60, 61)})

    def test_setpoint_is_subtracted_from_n(self):
        self.assertEqual(nav_cond.heading_code(4, setpoint=3), 0)    # n − H = 1
        self.assertEqual(nav_cond.heading_code(5, setpoint=3), 2)    # n − H = 2
        self.assertEqual(nav_cond.heading_code(0, setpoint=3), 3)    # n − H = −3
        self.assertEqual(nav_cond.heading_code(3, setpoint=3), 0)

    def test_card_example_codes_use_n_before_the_step(self):
        # 动作 F L L F R：执行前的 n 是 0 0 1 2 2，码是 0 0 0 2 2（任务卡 W1-9 的验收例）
        self.assertEqual(nav_cond.heading_codes([FWD, LEFT, LEFT, FWD, RIGHT]), [0, 0, 0, 2, 2])

    def test_first_frame_is_code_zero_and_empty_input(self):
        self.assertEqual(nav_cond.heading_codes([]), [])
        self.assertEqual(nav_cond.heading_codes([LEFT]), [0])
        self.assertEqual(nav_cond.heading_codes([RIGHT, RIGHT]), [0, 0])

    def test_turning_back_returns_to_code_zero(self):
        # n 在执行前依次是 0 1 2 3 2 1 0
        self.assertEqual(nav_cond.heading_codes([LEFT, LEFT, LEFT, RIGHT, RIGHT, RIGHT, FWD]), [0, 0, 2, 2, 2, 0, 0])

    def test_right_drift_gives_code_three(self):
        self.assertEqual(nav_cond.heading_codes([RIGHT, RIGHT, RIGHT, FWD]), [0, 0, 3, 3])

    def test_non_turn_actions_do_not_move_n(self):
        self.assertEqual(nav_cond.heading_codes([0, FWD, 0, FWD, LEFT, 0]), [0] * 6)

    def test_codes_stay_inside_the_prev_action_range(self):
        codes = nav_cond.heading_codes([LEFT] * 20 + [RIGHT] * 50)
        self.assertTrue(all(0 <= c <= 3 for c in codes))

    def test_self_check_passes(self):
        nav_cond.self_check()


class DatasetCondTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def test_reads_the_tag_from_the_meta_tail(self):
        write_dataset(self.path("a.uavr"), {"schema": "uavrgb.dataset.v1", "cond": nav_cond.DATASET_COND_TAG})
        write_dataset(self.path("b.uavr"), {"schema": "uavrgb.dataset.v1"})
        self.assertEqual(nav_cond.dataset_cond_tag(self.path("a.uavr")), "heading_code.v1")
        self.assertIsNone(nav_cond.dataset_cond_tag(self.path("b.uavr")))

    def test_bad_files_raise_value_error(self):
        with open(self.path("junk.uavr"), "wb") as f:
            f.write(b"x" * 100)
        with self.assertRaises(ValueError):
            nav_cond.dataset_cond_tag(self.path("junk.uavr"))
        with self.assertRaises(OSError):
            nav_cond.dataset_cond_tag(self.path("missing.uavr"))


class ConditionGateTests(unittest.TestCase):
    """起跑前闸门：模型条件类型（prev_dim 与训练数据 meta）对 --student-cond 的判定表。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)

    def make_run(self, name, meta, config_extra=None, sha_ok=True, with_config=True):
        """run 目录：ckpt.bin + config.json（指向数据集）+ 数据集；meta=None 时不建数据集文件。"""
        run = self.path(name)
        os.makedirs(run)
        ds = self.path(name + "_ds", "dataset.uavr")
        os.makedirs(os.path.dirname(ds))
        sha = ""
        if meta is not None:
            write_dataset(ds, meta)
            with open(ds, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
        if not sha_ok:
            sha = "0" * 64
        if with_config:
            with open(os.path.join(run, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"schema": "uavrgb.train_config.v1", "data": ds, "dataset_sha256": sha, "prev_action": True,
                           **(config_extra or {})}, f)
        ckpt = os.path.join(run, "ckpt_best.bin")
        open(ckpt, "wb").close()
        return ckpt, ds

    def test_model_without_condition_input(self):
        ckpt, _ = self.make_run("r", {"schema": "x"})
        ok = nav_cond.check_student_cond("prev-action", ckpt, 0)
        self.assertEqual((ok["status"], ok["trained"]), ("ok", "none"))
        bad = nav_cond.check_student_cond("heading-code", ckpt, 0)
        self.assertEqual((bad["status"], bad["trained"]), ("mismatch", "none"))
        self.assertIn("不会生效", bad["message"])

    def test_prev_action_model(self):
        ckpt, _ = self.make_run("r", {"schema": "x"})
        self.assertEqual(nav_cond.check_student_cond("prev-action", ckpt, 4)["status"], "ok")
        bad = nav_cond.check_student_cond("heading-code", ckpt, 4)
        self.assertEqual((bad["status"], bad["trained"]), ("mismatch", "prev-action"))

    def test_heading_code_model(self):
        ckpt, _ = self.make_run("r", {"schema": "x", "cond": "heading_code.v1"})
        ok = nav_cond.check_student_cond("heading-code", ckpt, 4)
        self.assertEqual((ok["status"], ok["trained"]), ("ok", "heading-code"))
        bad = nav_cond.check_student_cond("prev-action", ckpt, 4)   # 忘了给 --student-cond（默认 prev-action）
        self.assertEqual((bad["status"], bad["trained"]), ("mismatch", "heading-code"))
        self.assertIn("heading-code", bad["message"])

    def test_unknown_when_nothing_to_read_warns_instead_of_guessing(self):
        # 没有 config.json：拿不到训练条件类型
        ckpt, _ = self.make_run("r1", {"schema": "x", "cond": "heading_code.v1"}, with_config=False)
        for requested in ("prev-action", "heading-code"):
            got = nav_cond.check_student_cond(requested, ckpt, 4)
            self.assertEqual((got["status"], got["trained"]), ("unverified", None), requested)
            self.assertTrue(got["message"])
        # 数据集文件已不在
        ckpt, _ = self.make_run("r2", None)
        self.assertEqual(nav_cond.check_student_cond("heading-code", ckpt, 4)["status"], "unverified")
        # 数据集 sha256 与训练时记的不同：文件被换过，meta 不可信
        ckpt, _ = self.make_run("r3", {"schema": "x", "cond": "heading_code.v1"}, sha_ok=False)
        got = nav_cond.check_student_cond("heading-code", ckpt, 4)
        self.assertEqual((got["status"], got["trained"]), ("unverified", None))
        self.assertIn("sha256", got["message"])
        # 认不得的条件版本标签
        ckpt, _ = self.make_run("r4", {"schema": "x", "cond": "heading_code.v9"})
        self.assertEqual(nav_cond.check_student_cond("heading-code", ckpt, 4)["status"], "unverified")

    def test_calibrated_ckpt_finds_the_run_through_the_sidecar_json(self):
        # average_ckpt 的产物放在别的目录，旁边的 <out>.json 记 run 路径；闸门经它找到训练 config
        ckpt, _ = self.make_run("run", {"schema": "x", "cond": "heading_code.v1"})
        out = self.path("ckpt", "cal.bin")
        os.makedirs(os.path.dirname(out))
        open(out, "wb").close()
        with open(out + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "run": os.path.dirname(ckpt)}, f)
        got = nav_cond.check_student_cond("heading-code", out, 4)
        self.assertEqual((got["status"], got["trained"]), ("ok", "heading-code"))
        self.assertEqual(nav_cond.check_student_cond("prev-action", out, 4)["status"], "mismatch")

    def test_result_records_where_the_answer_came_from(self):
        ckpt, ds = self.make_run("r", {"schema": "x", "cond": "heading_code.v1"})
        got = nav_cond.check_student_cond("heading-code", ckpt, 4)
        self.assertEqual(set(got), {"status", "requested", "trained", "source", "message"})
        self.assertEqual(got["requested"], "heading-code")
        self.assertIn(ds, got["source"])


if __name__ == "__main__":
    unittest.main()
