#!/usr/bin/env python3
"""真实 StudentPolicy + 假推理进程，检查 3DGS 学生/DAgger 共用的输入闸门。"""
import contextlib
import importlib
import io
import json
import os
import random
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
from student import StudentPolicy
import nav_cond

collect = importlib.import_module("sim.3dgs.collect_3dgs")
fixtures = importlib.import_module("sim.3dgs.test_3dgs_student_prev_action")


class StudentInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.counter = 0

    def student(self, text_dim=0, text_ids=None, prev_dim=0, tag=None):
        self.counter += 1
        path = os.path.join(self.tmp.name, f"infer{self.counter}.py")
        source = fixtures.fake_infer_source(prev_dim, 1).replace(
            '"text_dim": 0', f'"text_dim": {text_dim}')
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        os.chmod(path, 0o700)
        if tag is not None:
            with open(path + ".json", "w", encoding="utf-8") as f:
                json.dump({"dataset_cond": tag}, f)
        policy = StudentPolicy(path, path, text_ids=text_ids)
        self.addCleanup(policy.close)
        return policy

    def rejected(self, policy, pattern, dagger=False):
        proc = policy.proc
        with self.assertRaisesRegex(RuntimeError, pattern):
            if dagger:
                collect.DaggerPolicy(policy, fixtures.FakeExpert(3), 1.0, random.Random(0))
            else:
                collect.StudentAdapter(policy)
        self.assertIsNone(policy.proc)
        self.assertIsNotNone(proc.poll(), "拒绝后不得留下等待 stdin 的进程")

    def test_missing_or_pad_only_instruction_is_rejected(self):
        for ids in (None, [], [0, 0], [1, 1]):
            with self.subTest(ids=ids):
                self.rejected(self.student(text_dim=16, text_ids=ids), "文本模型缺少有效指令")

    def test_dagger_also_rejects_missing_instruction(self):
        self.rejected(self.student(text_dim=16), "文本模型缺少有效指令", dagger=True)

    def test_pure_vision_cannot_claim_to_use_supplied_text(self):
        self.rejected(self.student(text_ids=[2, 5, 6, 7, 11]), "纯视觉模型不接收文本")

    def test_valid_instruction_and_shuffled_tokens_reach_infer(self):
        for ids in ([2, 5, 6, 7, 11], [11, 7, 6, 5, 2]):
            with self.subTest(ids=ids):
                policy = self.student(text_dim=16, text_ids=ids)
                adapter = collect.StudentAdapter(policy)
                self.assertEqual(policy.proc.args[-2:], ["--text", ",".join(map(str, ids))])
                self.assertEqual(fixtures.drive(adapter), [1, 1, 1])

    def test_existing_blank_and_probe_calls_remain_valid(self):
        # Gazebo 显式 blank 消融与元数据探针没有走 3DGS adapter，不改变历史协议。
        for ids in (None, []):
            policy = self.student(text_dim=16, text_ids=ids)
            self.assertEqual(policy.text_dim, 16)
            self.assertIsNone(policy.student_cond_check)

    def test_heading_code_checkpoint_rejected_in_both_wrappers(self):
        for dagger in (False, True):
            with self.subTest(dagger=dagger):
                self.rejected(self.student(prev_dim=4, tag=nav_cond.DATASET_COND_TAG),
                              "条件类型 heading-code 不一致", dagger=dagger)

    def test_verified_previous_action_is_accepted_and_cached(self):
        policy = self.student(prev_dim=4)
        with open(policy.ckpt + ".json", "w", encoding="utf-8") as f:
            json.dump({"dataset_cond": None}, f)
        adapter = collect.StudentAdapter(policy)
        check = policy.student_cond_check
        self.assertEqual((check["status"], check["trained"]), ("ok", "prev-action"))
        self.assertEqual(fixtures.drive(adapter), [1, 1, 1])
        collect.StudentAdapter(policy)
        self.assertIs(policy.student_cond_check, check)

    def test_unknown_checkpoint_provenance_warns_and_remains_unverified(self):
        policy = self.student(prev_dim=4)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            collect.StudentAdapter(policy)
        self.assertEqual(policy.student_cond_check["status"], "unverified")
        self.assertIn("条件输入一致性未核实", output.getvalue())


if __name__ == "__main__":
    unittest.main()
