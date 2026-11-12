"""run_closed_loop.sh 的演练测试：用评测桩（stub_eval.py）与假的 nvidia-smi / cgroup 文件，把入口脚本从头到尾跑一遍，不起 Gazebo、不占机器锁。

覆盖：全流程 rc=0（含迟到集重跑、配方臂、汇总）、整臂失败 rc=11 与续跑补齐、拿不到机器锁时的阻塞等待、正式模式遇到覆盖变量 rc=5、第一部分结果缺失 rc=5。
冻结代码走 git archive，所以脚本相关文件要先提交，测试用的是当前 HEAD 里的版本。

运行：cd experiments/2026-11-12-recipe-stability && <解释器> -m unittest test_rehearsal
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPT = os.environ.get("RS_TEST_SCRIPT") or os.path.join(HERE, "run_closed_loop.sh")   # 变异检查时指向被改过的副本
STUB = os.path.join(HERE, "stub_eval.py")
D = "/root/rivermind-data/uav_rgb"
SWA = os.path.join(D, "exp-swa-calib-20261110", "ckpt")
GPU = "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740"


@unittest.skipUnless(os.path.exists(os.path.join(SWA, "cal_swa_s0_k10.bin")), "没有 cal_swa 权重")
class TestRehearsal(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        t = self.td.name
        self.out = os.path.join(t, "out")
        self.part1 = os.path.join(t, "part1")
        for s in (0, 1, 2):
            d = os.path.join(self.part1, "avg", f"R1-s{s}")
            os.makedirs(d)
            shutil.copy(os.path.join(SWA, f"cal_swa_s{s}_k10.bin"), os.path.join(d, "avg_cal.bin"))
            with open(os.path.join(d, "AVG_OK"), "w") as f:
                f.write("rc=0\n")
        self.stab = os.path.join(t, "stab.json")
        with open(self.stab, "w") as f:
            json.dump({"extra_arm": "R1"}, f)
        self.smi = os.path.join(t, "smi.sh")
        with open(self.smi, "w") as f:
            f.write(f"#!/bin/sh\necho 'GPU 0: Fake (UUID: {GPU})'\n")
        os.chmod(self.smi, 0o755)
        self.cpu_max, self.cpu_stat = os.path.join(t, "cpu.max"), os.path.join(t, "cpu.stat")
        with open(self.cpu_max, "w") as f:
            f.write("1200000 100000\n")
        with open(self.cpu_stat, "w") as f:
            f.write("usage_usec 1000000\nnr_throttled 0\nthrottled_usec 0\n")
        self.lock = os.path.join(t, "machine.lock")
        open(self.lock, "w").close()

    def tearDown(self):
        self.td.cleanup()

    def env(self, **kw):
        e = dict(os.environ, RS_REHEARSAL="1", RS_OUT=self.out, RS_PART1=self.part1, RS_STABILITY_JSON=self.stab, RS_SELFCHECK="0", RS_EVAL=STUB,
                 RS_MACHINE_LOCK=self.lock, RS_GZ_PROBE="echo 0", RS_IDLE_WINDOW_S="1", RS_LOCK_LOG_S="2", RS_GZ_RECHECK_S="1", RS_NVIDIA_SMI=self.smi,
                 RS_CPU_MAX_FILE=self.cpu_max, RS_CPU_STAT_FILE=self.cpu_stat, STUB_SLEEP="0.01", PYTHONDONTWRITEBYTECODE="1")
        e.update(kw)
        return e

    def run_script(self, env, timeout=180):
        os.makedirs(self.out, exist_ok=True)
        return subprocess.run(["bash", SCRIPT], env=env, cwd=self.out, capture_output=True, text=True, timeout=timeout)

    def done(self):
        with open(os.path.join(self.out, "DONE"), encoding="utf-8") as f:
            return f.read()

    def test_full_run(self):
        r = self.run_script(self.env())
        self.assertEqual(r.returncode, 0, r.stderr + self.done())
        self.assertTrue(self.done().startswith("rc=0"))
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            res = json.load(f)
        for k in ("m0.5", "0", "p0.5"):
            self.assertTrue(res["levels"][k]["complete"], k)
            self.assertEqual(res["levels"][k]["n"], 60)
        self.assertTrue(res["extra"]["complete"])
        self.assertIn(res["verdicts"]["spread"]["code"], ("sensitive", "insensitive", "between"))
        self.assertTrue(any(n.endswith("-retry1") for n in os.listdir(self.out)))          # 桩在首次运行里制造迟到集，重跑目录要出现
        with open(os.path.join(self.out, "launches.log"), encoding="utf-8") as f:
            self.assertIn("模式=演练", f.read())
        with open(os.path.join(self.out, "ckpt", "shifts.json"), encoding="utf-8") as f:
            shifts = json.load(f)
        self.assertEqual(len(shifts["files"]), 9)
        self.assertEqual(set(os.listdir(os.path.join(self.out, "ckpt"))) - {"shifts.json"}, {f"cal_swa_s{s}_k10_fwd{t}.bin" for s in (0, 1, 2) for t in ("m0.5", "p0.5")})

    def test_substitute_extra_arm_runs_and_is_labelled(self):
        for s in (0, 1, 2):
            d = os.path.join(self.part1, "avg", f"R2e-s{s}")
            os.makedirs(d)
            shutil.copy(os.path.join(SWA, f"cal_swa_s{s}_k10.bin"), os.path.join(d, "avg_cal.bin"))
            with open(os.path.join(d, "AVG_OK"), "w") as f:
                f.write("rc=0\n")
        with open(self.stab, "w") as f:
            json.dump({"extra_arm": "R2e", "extra_arm_card": None, "extra_arm_is_substitute": True}, f)
        r = self.run_script(self.env())
        self.assertEqual(r.returncode, 0, r.stderr + self.done())
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual(res["extra"]["tag"], "R2e")
        self.assertTrue(res["extra"]["complete"])
        with open(os.path.join(self.out, "summary.md"), encoding="utf-8") as f:
            md = f.read()
        self.assertIn("配方臂 R2e（替代）", md)
        self.assertTrue(all(os.path.isdir(os.path.join(self.out, f"x-R2e-s{s}")) for s in (0, 1, 2)))

    def test_no_extra_arm(self):
        with open(self.stab, "w") as f:
            json.dump({"extra_arm": None}, f)
        r = self.run_script(self.env())
        self.assertEqual(r.returncode, 0, r.stderr + self.done())
        self.assertFalse(any(n.startswith("x-") for n in os.listdir(self.out)))
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            self.assertNotIn("extra", json.load(f))

    def test_failed_arm_then_resume(self):
        r = self.run_script(self.env(STUB_FAIL_ARM="lvl-p0.5-s1"))
        self.assertEqual(r.returncode, 11, r.stderr)
        self.assertIn("lvl-p0.5-s1", self.done())
        r = self.run_script(self.env())
        self.assertEqual(r.returncode, 0, r.stderr + self.done())
        with open(os.path.join(self.out, "run.log"), encoding="utf-8") as f:
            log = f.read()
        self.assertIn("lvl-0-s0 已评完，跳过", log)

    def test_waits_for_machine_lock(self):
        holder = subprocess.Popen(["bash", "-c", f"exec 8>> {self.lock}; flock 8; sleep 6"])
        try:
            import time
            time.sleep(1)
            r = self.run_script(self.env())
            self.assertEqual(r.returncode, 0, r.stderr + self.done())
            with open(os.path.join(self.out, "run.log"), encoding="utf-8") as f:
                self.assertIn("机器锁", f.read())
            with open(os.path.join(self.out, "start.txt"), encoding="utf-8") as f:
                lock_wait = [l for l in f.read().splitlines() if l.startswith("lock_wait_s=")][0]
            self.assertGreaterEqual(int(lock_wait.split("=")[1]), 3)
        finally:
            holder.wait(timeout=30)

    def test_formal_mode_refuses_overrides(self):
        e = self.env()
        e.pop("RS_REHEARSAL")
        r = self.run_script(e)
        self.assertEqual(r.returncode, 5)
        self.assertIn("覆盖配置的环境变量", r.stderr)
        self.assertTrue(self.done().startswith("rc=5"))
        self.assertFalse(os.path.exists(os.path.join(self.out, "run.pid")))                # 没进入正式流程，不写运行状态

    def test_shift_generation_failure_is_rc11(self):
        swa = os.path.join(self.td.name, "swa")
        os.makedirs(swa)
        for s in (0, 1, 2):
            shutil.copy(os.path.join(SWA, f"cal_swa_s{s}_k10.bin"), swa)
        with open(os.path.join(swa, "cal_swa_s1_k10.bin"), "r+b") as f:                    # 截断：文件在，但读不出来
            f.truncate(5000)
        r = self.run_script(self.env(RS_SWA=swa))
        self.assertEqual(r.returncode, 11, r.stderr)
        self.assertIn("生成平移权重失败", self.done())
        self.assertFalse(any(n.startswith("lvl-") and os.path.isdir(os.path.join(self.out, n)) for n in os.listdir(self.out)))   # 没有任何臂开跑

    def test_missing_part1_result_refuses(self):
        r = self.run_script(self.env(RS_STABILITY_JSON=os.path.join(self.td.name, "nope.json")))
        self.assertEqual(r.returncode, 5)
        self.assertIn("第一部分的结果还没有提交", self.done())

    def test_missing_extra_arm_input_refuses(self):
        os.remove(os.path.join(self.part1, "avg", "R1-s1", "AVG_OK"))
        r = self.run_script(self.env())
        self.assertEqual(r.returncode, 5)
        self.assertIn("R1-s1", self.done())


if __name__ == "__main__":
    unittest.main()
