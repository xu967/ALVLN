#!/usr/bin/env python3
"""清理只限本次进程组；组长退出后也要收掉忽略 TERM 的子进程。"""
import ctypes
import os
import signal
import subprocess
import sys
import unittest
from unittest.mock import Mock, call, patch

import collect_parallel as cp


class CleanupTests(unittest.TestCase):
    def test_exited_leader_does_not_skip_kill(self):
        proc = Mock(pid=31337)
        proc.poll.return_value = 0
        with patch.object(cp.os, "killpg") as kill, \
             patch.object(cp, "_group_running", side_effect=[True, False]), \
             patch.object(cp.time, "monotonic", side_effect=[0, 9, 9]), \
             patch.object(cp, "_unregister") as unregister:
            self.assertTrue(cp.stop_group(proc, "mock"))
        self.assertEqual(kill.call_args_list, [call(31337, signal.SIGTERM),
                                             call(31337, signal.SIGKILL)])
        unregister.assert_called_once_with(proc)

    def test_failed_cleanup_stays_registered(self):
        proc = Mock(pid=31337)
        with patch.dict(cp._PROC_GROUPS, {proc.pid: proc}, clear=True), \
             patch.object(cp.os, "killpg"), \
             patch.object(cp, "_group_running", return_value=True), \
             patch.object(cp.time, "monotonic", side_effect=[0, 9, 9, 15]):
            self.assertFalse(cp.stop_group(proc, "mock"))
            self.assertIs(cp._PROC_GROUPS[proc.pid], proc)

    def test_cleanup_does_not_scan_or_kill_other_tasks(self):
        proc = Mock(pid=31337)
        with patch.dict(cp._PROC_GROUPS, {proc.pid: proc}, clear=True), \
             patch.object(cp, "stop_group") as stop, \
             patch.object(cp.subprocess, "run") as scan, \
             patch.object(cp.os, "kill") as kill:
            cp.cleanup_all()
        stop.assert_called_once_with(proc, "cleanup")
        scan.assert_not_called()
        kill.assert_not_called()

    def test_group_running_ignores_only_its_zombies(self):
        for output, want in (("31337 Z\n44444 S\n", False),
                             ("31337 Z\n31337 Sl\n", True),
                             ("44444 S\n", True), ("malformed\n", True)):
            with self.subTest(output=output), patch.object(cp.os, "killpg"), \
                 patch.object(cp.subprocess, "run", return_value=Mock(
                     returncode=0, stdout=output)):
                self.assertEqual(cp._group_running(31337), want)

    def test_group_probe_failure_keeps_registration(self):
        with patch.object(cp.os, "killpg"), patch.object(cp.subprocess, "run",
                side_effect=subprocess.TimeoutExpired("ps", 2)):
            self.assertTrue(cp._group_running(31337))
        with patch.object(cp.os, "killpg", side_effect=ProcessLookupError):
            self.assertFalse(cp._group_running(31337))

    @unittest.skipUnless(sys.platform == "linux", "Linux process groups / subreaper")
    def test_real_exited_leader_and_term_ignoring_child(self):
        # 本测试收养并 wait 自己的孤儿，避免容器 PID 1 不回收而留下 zombie。
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        proc = None
        child_pid = None
        code = """
import os, signal
read_fd, write_fd = os.pipe()
child = os.fork()
if child == 0:
    os.close(read_fd)
    os.close(1)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(write_fd, b'1')
    os.close(write_fd)
    while True:
        signal.pause()
os.close(write_fd)
os.read(read_fd, 1)
os.close(read_fd)
print(child, flush=True)
"""
        try:
            proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            cp._register(proc)
            stdout, _ = proc.communicate(timeout=5)
            child_pid = int(stdout.strip())
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(cp.stop_group(proc, "real regression", term_timeout=.05))
            self.assertNotIn(proc.pid, cp._PROC_GROUPS)
            reaped, status = os.waitpid(child_pid, os.WNOHANG)
            self.assertEqual(reaped, int(stdout.strip()))
            self.assertTrue(os.WIFSIGNALED(status))
            self.assertEqual(os.WTERMSIG(status), signal.SIGKILL)
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
                # 仅回收本测试创建的进程组成员。
                try:
                    while os.waitpid(-proc.pid, 0)[0]:
                        pass
                except ChildProcessError:
                    pass
                cp._unregister(proc)
                if proc.stdout is not None:
                    proc.stdout.close()
            libc.prctl(36, previous.value, 0, 0, 0)


if __name__ == "__main__":
    unittest.main()
