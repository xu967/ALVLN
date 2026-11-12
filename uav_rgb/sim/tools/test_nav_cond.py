#!/usr/bin/env python3
"""航向码 nav_cond 的纯 CPU 回归（任务卡 W1-9）：heading_code 的边界与首帧、动作序列 F L L F R 的码序列、
一致性闸门的判定表与「训练时条件类型」的读取（临时文件，不读真实数据，不起 Gazebo）。

运行（仓库根）：.venv-tools/bin/python sim/tools/test_nav_cond.py
"""
import ast
import hashlib
import inspect
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

    def test_sidecar_json_dataset_cond_and_abs_data_prevent_unverified_degradation(self):
        """I2-5 第 1 点：<ckpt>.json 记有 dataset_cond 或绝对路径 data 时优先使用，
        即使训练 config.json 写相对路径、切换工作目录、甚至原 dataset.uavr 已被归档移走，也不退化为 unverified。"""
        # 1) 新版 <ckpt>.json 记了 dataset_cond：连 dataset.uavr 与 run/config.json 都删掉仍能准确拦截或放行
        cal_hc = self.path("exp1", "exported", "cal_hc.bin")
        os.makedirs(os.path.dirname(cal_hc), exist_ok=True)
        with open(cal_hc, "wb") as f:
            f.write(b"dummy")
        with open(cal_hc + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "run": self.path("exp1", "gone_run"),
                       "data": self.path("exp1", "gone_ds", "dataset.uavr"), "data_sha256": "a" * 64,
                       "dataset_cond": "heading_code.v1"}, f)
        self.assertEqual(nav_cond.check_student_cond("heading-code", cal_hc, 4)["status"], "ok")
        bad = nav_cond.check_student_cond("prev-action", cal_hc, 4)
        self.assertEqual((bad["status"], bad["trained"]), ("mismatch", "heading-code"))

        # dataset_cond 为 null（prev-action 数据集）：同样无需原数据集文件即可判定 prev-action
        cal_pa = self.path("exp1", "exported", "cal_pa.bin")
        with open(cal_pa, "wb") as f:
            f.write(b"dummy")
        with open(cal_pa + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "data": self.path("exp1", "gone_ds", "pa.uavr"),
                       "dataset_cond": None}, f)
        self.assertEqual((nav_cond.check_student_cond("prev-action", cal_pa, 4)["status"],
                          nav_cond.check_student_cond("prev-action", cal_pa, 4)["trained"]), ("ok", "prev-action"))
        self.assertEqual(nav_cond.check_student_cond("heading-code", cal_pa, 4)["status"], "mismatch")

        # 2) 旧 dc8e5ff 产出的 <ckpt>.json（只有绝对路径 data 与 data_sha256，无 dataset_cond），而 run/config.json 是相对路径：
        # 复现审核 I2 verify_i2_part2.py TEST 6，切到其它 cwd 时应通过 sidecar["data"] 拦下错配，不能退化为 unverified
        exp2 = self.path("exp2")
        ds2 = os.path.join(exp2, "ds", "dataset.uavr")
        os.makedirs(os.path.dirname(ds2), exist_ok=True)
        write_dataset(ds2, {"schema": "uavrgb.dataset.v1", "cond": "heading_code.v1"})
        sha2 = hashlib.sha256(open(ds2, "rb").read()).hexdigest()
        os.makedirs(os.path.join(exp2, "run"), exist_ok=True)
        with open(os.path.join(exp2, "run", "config.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "uavrgb.train_config.v1", "data": "ds/dataset.uavr", "dataset_sha256": sha2}, f)
        cal_dc = os.path.join(exp2, "exported", "cal.bin")
        os.makedirs(os.path.dirname(cal_dc), exist_ok=True)
        open(cal_dc, "wb").write(b"dummy")
        with open(cal_dc + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "run": os.path.join(exp2, "run"),
                       "data": ds2, "data_sha256": sha2,
                       "cond": {"prev_dim": 4, "column": "prev_action", "val_value_counts": [10, 0, 2, 2]}}, f)
        other_cwd = self.path("other_cwd")
        os.makedirs(other_cwd, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(other_cwd)
            got_dc = nav_cond.check_student_cond("prev-action", cal_dc, 4)
        finally:
            os.chdir(old_cwd)
        self.assertEqual((got_dc["status"], got_dc["trained"]), ("mismatch", "heading-code"))

        # 3) 若 <ckpt>.json 记了 out_sha256 而权重文件本身被覆盖替换，也必须判 mismatch 而不盲信旧 sidecar
        with open(cal_hc + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "dataset_cond": "heading_code.v1",
                       "out_sha256": hashlib.sha256(b"other-bytes").hexdigest()}, f)
        self.assertEqual(nav_cond.check_student_cond("heading-code", cal_hc, 4)["status"], "mismatch")

    @staticmethod
    def _write_v6_ckpt_header(path, dataset_sha_hex, nw=4, lr=1e-3, batch=128):
        """构造足够读出 v5 训练契约的 UAVRCK01 v6 头（对齐 src/checkpoint.cpp:save_checkpoint）。"""
        head = struct.pack("<8sIIiiiiIII", b"UAVRCK01", 6, 1, 120, 160, 3, 4, 10, 5, nw)
        weights = struct.pack(f"<{nw}f", *([1.0] * nw))
        v3_v4 = struct.pack("<fi", 0.5, 0)
        contract = (bytes.fromhex(dataset_sha_hex) + struct.pack("<B", 1) + b"\x00" * 32
                    + struct.pack("<ffIi", 0.0, lr, 42, batch) + b"\x00" * 32)
        v6 = struct.pack("<ii", 0, 4)
        with open(path, "wb") as f:
            f.write(head + weights + v3_v4 + contract + v6)

    def test_resume_config_takes_precedence_over_initial_config(self):
        """I1-6 第 1 点：同目录续训时先看 config.resume.json 再看 config.json（直接同目录与经 sidecar run 目录均如此）。"""
        ds_pa = self.path("pa.uavr")
        ds_hc = self.path("hc.uavr")
        write_dataset(ds_pa, {"schema": "uavrgb.dataset.v1"})
        write_dataset(ds_hc, {"schema": "uavrgb.dataset.v1", "cond": "heading_code.v1"})
        sha_pa = hashlib.sha256(open(ds_pa, "rb").read()).hexdigest()
        sha_hc = hashlib.sha256(open(ds_hc, "rb").read()).hexdigest()

        run = self.path("run_resumed")
        os.makedirs(run)
        with open(os.path.join(run, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "uavrgb.train_config.v1", "data": ds_pa, "dataset_sha256": sha_pa}, f)
        with open(os.path.join(run, "config.resume.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "uavrgb.train_config.v1", "data": ds_hc, "dataset_sha256": sha_hc}, f)
        ckpt = os.path.join(run, "ckpt_best.bin")
        open(ckpt, "wb").close()

        cfg_path, _ = nav_cond.find_train_config(ckpt)
        self.assertEqual(cfg_path, os.path.join(run, "config.resume.json"))
        ok = nav_cond.check_student_cond("heading-code", ckpt, 4)
        self.assertEqual((ok["status"], ok["trained"]), ("ok", "heading-code"))
        bad = nav_cond.check_student_cond("prev-action", ckpt, 4)
        self.assertEqual((bad["status"], bad["trained"]), ("mismatch", "heading-code"))

        # 经 <out>.json 的 run 字段查找时也优先读 config.resume.json
        cal = self.path("exported", "cal.bin")
        os.makedirs(os.path.dirname(cal))
        open(cal, "wb").close()
        with open(cal + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "run": run}, f)
        self.assertEqual(nav_cond.find_train_config(cal)[0], os.path.join(run, "config.resume.json"))
        self.assertEqual(nav_cond.check_student_cond("heading-code", cal, 4)["status"], "ok")

    def test_ckpt_header_dataset_sha256_mismatch_is_caught(self):
        """I1-6 第 2 点：ckpt 二进制头内嵌的训练契约 dataset_sha256 与配置或侧车不一致时判 mismatch。"""
        ckpt, ds_hc = self.make_run("hc_dir", {"schema": "uavrgb.dataset.v1", "cond": "heading_code.v1"})
        sha_hc = hashlib.sha256(open(ds_hc, "rb").read()).hexdigest()
        other_sha = "ab" * 32

        # 契约 sha256 与 config.json 一致 -> ok
        self._write_v6_ckpt_header(ckpt, sha_hc)
        self.assertEqual(nav_cond.check_student_cond("heading-code", ckpt, 4)["status"], "ok")

        # 把别的运行训出的 ckpt 拷进 hc_dir（契约记 other_sha != config.json 的 sha_hc）-> mismatch
        self._write_v6_ckpt_header(ckpt, other_sha)
        got = nav_cond.check_student_cond("heading-code", ckpt, 4)
        self.assertEqual((got["status"], got["trained"]), ("mismatch", None))
        self.assertIn("dataset_sha256", got["message"])

        # 配置未写 dataset_sha256，但磁盘数据集 sha256 与 ckpt 契约不符 -> mismatch
        with open(os.path.join(os.path.dirname(ckpt), "config.json"), "w", encoding="utf-8") as f:
            json.dump({"schema": "uavrgb.train_config.v1", "data": ds_hc}, f)
        self.assertEqual(nav_cond.check_student_cond("heading-code", ckpt, 4)["status"], "mismatch")

        # <ckpt>.json 侧车记的 data_sha256 与 ckpt 契约不符 -> mismatch
        cal = self.path("cal_mismatch.bin")
        self._write_v6_ckpt_header(cal, other_sha)
        with open(cal + ".json", "w", encoding="utf-8") as f:
            json.dump({"tool": "tools/average_ckpt.py", "dataset_cond": "heading_code.v1",
                       "data": ds_hc, "data_sha256": sha_hc}, f)
        self.assertEqual(nav_cond.check_student_cond("heading-code", cal, 4)["status"], "mismatch")

        # 全零占位契约（测试假权重）按无契约处理，不误拒
        ckpt_zero, _ = self.make_run("zero_dir", {"schema": "uavrgb.dataset.v1", "cond": "heading_code.v1"})
        self._write_v6_ckpt_header(ckpt_zero, "00" * 32, lr=0.0, batch=0)
        self.assertEqual(nav_cond.check_student_cond("heading-code", ckpt_zero, 4)["status"], "ok")

    def test_explicit_prev_action_on_pure_image_model_is_mismatch(self):
        """I1-6 第 3 点：显式要求 prev-action 而模型 prev_dim=0 时判 mismatch；默认 explicit=False 保持兼容放行。"""
        ckpt, _ = self.make_run("r_img", {"schema": "uavrgb.dataset.v1"})
        # 默认 explicit=False（兼容现有未改的调用方）：ok
        ok_default = nav_cond.check_student_cond("prev-action", ckpt, 0)
        self.assertEqual((ok_default["status"], ok_default["trained"]), ("ok", "none"))
        # explicit=True：显式传 --student-cond prev-action 给纯视觉模型必须拦截
        bad_exp = nav_cond.check_student_cond("prev-action", ckpt, 0, explicit=True)
        self.assertEqual((bad_exp["status"], bad_exp["trained"]), ("mismatch", "none"))
        self.assertIn("不会生效", bad_exp["message"])
        # prev_dim=4 的 prev-action 模型在 explicit=True 下正常放行
        ok_pa = nav_cond.check_student_cond("prev-action", ckpt, 4, explicit=True)
        self.assertEqual((ok_pa["status"], ok_pa["trained"]), ("ok", "prev-action"))

    def test_three_positional_args_call_like_collect_py_on_main(self):
        """卡 W1-13 第二轮（main 的 abad4cf）在 sim/tools/collect.py:1846 这样调：
        nav_cond.check_student_cond(args.student_cond, args.student_ckpt, getattr(student, "prev_dim", 0))
        三个位置参数、不带 explicit。本卡加的 explicit 必须有缺省值 False，这个调用的判定与结果同改前一致；
        仓库里所有对它的调用都能绑到当前签名，也没有谁用位置参数传第 4 个（explicit 只能用关键字）。"""
        sig = inspect.signature(nav_cond.check_student_cond)
        names = list(sig.parameters)
        self.assertEqual(names[:3], ["requested", "ckpt", "prev_dim"])
        self.assertTrue(all(sig.parameters[n].default is not inspect.Parameter.empty for n in names[3:]),
                        f"第 3 个之后的参数都要有缺省值：{names}")
        self.assertIs(sig.parameters["explicit"].default, False)

        class Student:   # 推理进程没报 prev_dim 时 collect.py 取缺省 0
            pass
        no_attr = getattr(Student(), "prev_dim", 0)
        ckpt_img, _ = self.make_run("r_img", {"schema": "x"})
        ckpt_pa, _ = self.make_run("r_pa", {"schema": "x"})
        ckpt_hc, _ = self.make_run("r_hc", {"schema": "x", "cond": "heading_code.v1"})
        ckpt_none, _ = self.make_run("r_none", {"schema": "x"}, with_config=False)
        cases = [   # (--student-cond, ckpt, prev_dim, 期望 (status, trained))
            ("prev-action", ckpt_img, no_attr, ("ok", "none")),
            ("heading-code", ckpt_img, no_attr, ("mismatch", "none")),
            ("prev-action", ckpt_pa, 4, ("ok", "prev-action")),
            ("heading-code", ckpt_pa, 4, ("mismatch", "prev-action")),
            ("heading-code", ckpt_hc, 4, ("ok", "heading-code")),
            ("prev-action", ckpt_hc, 4, ("mismatch", "heading-code")),
            ("prev-action", ckpt_none, 4, ("unverified", None)),
        ]
        for requested, ckpt, prev_dim, want in cases:
            got = nav_cond.check_student_cond(requested, ckpt, prev_dim)   # collect.py 的调用形态
            self.assertEqual((got["status"], got["trained"]), want, (requested, ckpt, prev_dim))
            self.assertEqual(got, nav_cond.check_student_cond(requested, ckpt, prev_dim, explicit=False))

        here = os.path.dirname(os.path.abspath(__file__))
        scanned = 0
        for folder in (here, os.path.join(here, "..", "..", "tools")):
            for fn in sorted(os.listdir(folder)):
                if not fn.endswith(".py") or fn.startswith("test_") or fn == "nav_cond.py":
                    continue
                path = os.path.join(folder, fn)
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if not (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "check_student_cond"):
                        continue
                    scanned += 1
                    self.assertLessEqual(len(node.args), 3, f"{fn}:{node.lineno} 不该用位置参数传 explicit")
                    if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                        continue   # *args / **kwargs 静态看不出形态
                    sig.bind(*[None] * len(node.args), **{k.arg: None for k in node.keywords})   # 绑不上就 TypeError
        self.assertGreaterEqual(scanned, 1, "sim/tools 里至少 eval_closed_loop.py 有一处调用")


if __name__ == "__main__":
    unittest.main()
