#!/usr/bin/env python3
"""freeze.sh 与 lib.sh 的 verify_frozen 的演练（卡 E2E-1）：在临时 git 仓库里冻结，核对冻结副本、拒绝脏工作区、拒绝冻结后换代码、发现冻结副本被改。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_freeze.py      约 5 s
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENTRY_NAME = "2026-11-12-e2e-heading-code"
GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false"]


def sh(cmd, cwd, env=None, check=True):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=check)


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.src = tmp / "src"
        self.out = tmp / "out"
        entry = self.src / "experiments" / ENTRY_NAME
        entry.mkdir(parents=True)
        for f in ("lib.sh", "freeze.sh"):
            shutil.copy(HERE / f, entry / f)
        (self.src / "sim" / "tools").mkdir(parents=True)
        (self.src / "sim" / "tools" / "a.py").write_text("A = 1\n", encoding="utf-8")
        (self.src / "tools").mkdir()
        (self.src / "tools" / "b.py").write_text("B = 2\n", encoding="utf-8")
        (self.src / "docs").mkdir()
        (self.src / "docs" / "note.md").write_text("n\n", encoding="utf-8")
        sh(["git", "init", "-q"], self.src)
        sh(["git", "add", "-A"], self.src)
        sh(GIT + ["commit", "-q", "-m", "init"], self.src)
        self.entry = entry

    def env(self):
        return {**os.environ, "OUT": str(self.out), "SRC": str(self.src), "CODE": "", "PYTHONDONTWRITEBYTECODE": "1"}

    def head(self):
        return sh(["git", "rev-parse", "HEAD"], self.src).stdout.strip()

    def freeze(self):
        env = self.env()
        env.pop("CODE")                                                           # lib.sh 的 CODE 取脚本所在位置的上两级
        return sh(["bash", str(self.entry / "freeze.sh")], self.src, env, check=False)

    def verify(self):
        env = self.env()
        env["CODE"] = str(self.out / "code")
        r = sh(["bash", "-c", f'source "{self.entry}/lib.sh"; verify_frozen; echo "rc=$? bad=$FROZEN_BAD"'], self.src, env, check=False)
        return r.stdout.strip().splitlines()[-1]

    def test_freeze_clean_head(self):
        r = self.freeze()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.out / "code_head.txt").read_text().strip(), self.head())
        self.assertEqual((self.out / "code" / "sim" / "tools" / "a.py").read_text(), "A = 1\n")
        self.assertTrue((self.out / "code" / "experiments" / ENTRY_NAME / "lib.sh").exists())
        self.assertEqual(self.verify(), "rc=0 bad=")
        self.assertEqual(self.freeze().returncode, 0)                             # 同一个 HEAD 再冻结一次：允许（幂等）

    def test_dirty_tracked_file_is_refused(self):
        (self.src / "tools" / "b.py").write_text("B = 3\n", encoding="utf-8")
        r = self.freeze()
        self.assertEqual(r.returncode, 4)
        self.assertIn("未提交的改动", r.stderr)
        self.assertFalse((self.out / "code_head.txt").exists())

    def test_untracked_file_in_code_dirs_is_refused(self):
        (self.src / "sim" / "tools" / "new.py").write_text("C = 1\n", encoding="utf-8")
        self.assertEqual(self.freeze().returncode, 4)

    def test_dirty_outside_code_dirs_is_allowed(self):
        (self.src / "docs" / "note.md").write_text("changed\n", encoding="utf-8")   # docs/ 不在冻结范围内
        self.assertEqual(self.freeze().returncode, 0)

    def test_new_head_after_freeze_is_refused(self):
        self.assertEqual(self.freeze().returncode, 0)
        (self.src / "tools" / "b.py").write_text("B = 3\n", encoding="utf-8")
        sh(["git", "add", "tools/b.py"], self.src)
        sh(GIT + ["commit", "-q", "-m", "change"], self.src)
        r = self.freeze()
        self.assertEqual(r.returncode, 4)
        self.assertIn("冻结后不换代码", r.stderr)

    def test_verify_detects_modified_and_missing_frozen_files(self):
        self.assertEqual(self.freeze().returncode, 0)
        target = self.out / "code" / "sim" / "tools" / "a.py"
        target.write_text("A = 99\n", encoding="utf-8")
        self.assertEqual(self.verify(), "rc=1 bad=sim/tools/a.py")
        target.write_text("A = 1\n", encoding="utf-8")
        self.assertEqual(self.verify(), "rc=0 bad=")
        (self.out / "code" / "tools" / "b.py").unlink()
        self.assertEqual(self.verify(), "rc=1 bad=tools/b.py")

    def test_verify_without_code_head_or_with_wrong_commit(self):
        self.assertEqual(self.verify().split(" ", 1)[0], "rc=1")                  # 还没冻结
        self.assertEqual(self.freeze().returncode, 0)
        (self.out / "code_head.txt").write_text("0" * 40 + "\n", encoding="utf-8")
        self.assertTrue(self.verify().startswith("rc=1 bad="))


if __name__ == "__main__":
    unittest.main(verbosity=2)
