#!/usr/bin/env python3
"""compare_packs.py 与 check_cond_codes.py 的回归（卡 E2E-1 打包阶段的两道核对；纯 CPU，临时目录，合成 episode，不读真实数据）。

夹具与 tools/test_preprocess_cond.py 同一套：3 个合成 episode，其中 ep1 的第 2 步是转向且碰撞（--drop-collision 丢样本不能丢计数）。
正例：同参数重打的两个默认包逐项相同；默认包对 --cond heading-code 包（--cond-pack）通过；航向码独立重算与侧车逐样本一致。
反例：改一个标签字节、少一个 episode、侧车里放码 1、拿默认包当航向码包、篡改一个码、改动一集的 actions.jsonl（保留步数对不上）——都必须被拦下。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_pack_checks.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import compare_packs as cp  # noqa: E402

PY = sys.executable
F, L, R = 1, 2, 3
# 每集：(执行动作, 教师标签, 碰撞步集合)；标签故意与执行动作不同（DAgger 形态）
EPISODES = {
    "ep1": ([F, L, L, F, R, L], [3, 1, 0, 2, 2, 1], {2}),
    "ep2": ([L, L, L, R], [2, 2, 2, 3], set()),
    "ep3": ([R, R, R, F, L], [1, 1, 0, 0, 3], set()),
}


class PackCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")], check=True, capture_output=True)
        self.eps = {n: (self.tmp / "ep" / n).resolve() for n in EPISODES}
        for name, (executed, expert, collided) in EPISODES.items():
            rows = [{"step": i, "timestamp_ns": 1_000_000_000 + 100_000_000 * i, "action_expert": expert[i], "action_executed": executed[i],
                     "collision": i in collided} for i in range(len(executed))]
            (self.eps[name] / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.default = self.pack("p-default")
        self.default2 = self.pack("p-default2")
        self.hc = self.pack("p-hc", "--cond", "heading-code")

    def pack(self, name, *flags, eps=None):
        out = self.tmp / name
        r = subprocess.run([PY, str(REPO / "tools/preprocess.py"), "--out", str(out), "--drop-collision", *flags,
                            *map(str, eps if eps is not None else self.eps.values())], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        return out

    def compare(self, orig, new, *flags):
        r = subprocess.run([PY, str(HERE / "compare_packs.py"), str(orig / "dataset.uavr"), str(new / "dataset.uavr"), *flags], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def codes(self, pack, *flags):
        r = subprocess.run([PY, str(HERE / "check_cond_codes.py"), str(pack / "dataset.uavr"), *flags], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    def copy_pack(self, src, name):
        dst = self.tmp / name
        shutil.copytree(src, dst)
        return dst

    # ---------------------------------------------------------------- compare_packs
    def test_default_repack_is_identical(self):
        rc, out = self.compare(self.default, self.default2)
        self.assertEqual(rc, 0, out)
        self.assertIn("RESULT: 核对项全部相同", out)

    def test_cond_pack_against_default(self):
        rc, out = self.compare(self.default, self.hc, "--cond-pack")
        self.assertEqual(rc, 0, out)
        self.assertIn("新包 prev_action 列只含码 0/2/3", out)
        # 参照侧车显式给出：同样通过
        rc, out = self.compare(self.default, self.hc, "--cond-pack", "--aux-ref", str(self.default2 / "dataset.aux"))
        self.assertEqual(rc, 0, out)

    def test_cond_pack_without_flag_is_rejected(self):
        rc, out = self.compare(self.default, self.hc)
        self.assertEqual(rc, 1, out)
        self.assertIn("不同  侧车 prev_action", out)

    def test_default_pack_is_not_a_cond_pack(self):
        rc, out = self.compare(self.default, self.default2, "--cond-pack")
        self.assertEqual(rc, 1, out)
        self.assertIn("不同  新包 meta 的 cond 标签", out)

    def test_label_byte_flip(self):
        bad = self.copy_pack(self.default2, "p-bad-label")
        head = cp.read_head(str(bad / "dataset.uavr"))
        with open(bad / "dataset.uavr", "r+b") as f:
            f.seek(head["lo"] + 3)
            b = f.read(1)
            f.seek(head["lo"] + 3)
            f.write(bytes([(b[0] + 1) % 4]))
        rc, out = self.compare(self.default, bad)
        self.assertEqual(rc, 1, out)
        self.assertIn("不同  标签区逐字节", out)

    def test_missing_episode(self):
        fewer = self.pack("p-fewer", eps=[self.eps["ep1"], self.eps["ep2"]])
        rc, out = self.compare(self.default, fewer)
        self.assertEqual(rc, 1, out)
        self.assertIn("不同  样本数 N、H、W、C", out)

    def test_cond_column_with_code_one_is_rejected(self):
        bad = self.copy_pack(self.hc, "p-hc-one")
        head = cp.read_head(str(bad / "dataset.uavr"))
        with open(bad / "dataset.aux", "r+b") as f:
            f.seek(cp.AUX_HEAD_SIZE + 4 * head["n"])     # 第一个样本的 prev_action
            f.write(bytes([1]))
        rc, out = self.compare(self.default, bad, "--cond-pack")
        self.assertEqual(rc, 1, out)
        self.assertIn("不同  新包 prev_action 列只含码 0/2/3", out)

    # ---------------------------------------------------------------- check_cond_codes
    def test_codes_match_on_real_cond_pack(self):
        rc, out = self.codes(self.hc, "--json", str(self.tmp / "codes.json"))
        self.assertEqual(rc, 0, out)
        res = json.loads((self.tmp / "codes.json").read_text(encoding="utf-8"))
        self.assertEqual((res["episodes"], res["samples"], res["mismatch_episodes"], res["mapping_failures"]), (3, 14, [], []))
        self.assertTrue(res["drop_collision"])
        # 手算：ep1 全部步的码 [0,0,0,2,2,0]，第 2 步被丢；ep2 [0,0,2,2]；ep3 [0,0,3,3,3] → 保留 14 个样本里码 2 有 4 个、码 3 有 3 个
        self.assertEqual(res["hist"], [7, 0, 4, 3])
        self.assertEqual(res["nonzero"], 7)

    def test_codes_reject_default_pack(self):
        rc, out = self.codes(self.default)
        self.assertEqual(rc, 1, out)
        self.assertIn("RESULT FAIL", out)

    def test_codes_reject_tampered_column(self):
        bad = self.copy_pack(self.hc, "p-hc-tamper")
        head = cp.read_head(str(bad / "dataset.uavr"))
        with open(bad / "dataset.aux", "r+b") as f:
            f.seek(cp.AUX_HEAD_SIZE + 4 * head["n"])       # 第 1 个样本：码本来是 0，改成 3
            f.write(bytes([3]))
        rc, out = self.codes(bad)
        self.assertEqual(rc, 1, out)
        self.assertIn("首个不一致", out)

    def test_codes_reject_when_collision_step_not_counted(self):
        # 把 ep1 被丢的碰撞步（第 2 步，转向 L）改成直行：重算的码少一次累计 → 与包里的码对不上
        path = self.eps["ep1"] / "actions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[2]["action_executed"] = F
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        rc, out = self.codes(self.hc)
        self.assertEqual(rc, 1, out)

    def test_codes_mapping_failure_when_kept_steps_differ(self):
        path = self.eps["ep2"] / "actions.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")        # 少一行：保留步数 ≠ 样本数
        rc, out = self.codes(self.hc)
        self.assertEqual(rc, 1, out)
        self.assertIn("mapping_failures=1", out)

    def test_codes_error_on_missing_aux(self):
        bad = self.copy_pack(self.hc, "p-hc-noaux")
        (bad / "dataset.aux").unlink()
        rc, out = self.codes(bad)
        self.assertEqual(rc, 2, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
