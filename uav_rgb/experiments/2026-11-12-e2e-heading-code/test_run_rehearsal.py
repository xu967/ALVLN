#!/usr/bin/env python3
"""run.sh 的演练（卡 E2E-1）：假评测桩（stub_eval_e2e.py）+ 假 nvidia-smi / cgroup 文件 / 临时机器锁，不起 Gazebo、不用 GPU。
覆盖：正常跑完（12 个臂、命令行逐项核对、种子优先的顺序、.cg 记录、起跑记录、DONE rc=0）；迟到集重跑与续跑；半成品臂挪开重评；
拒绝起跑的各个 rc（PREP_DONE 缺失 / 不是 rc=0、L_STAR 格式、权重与侧车缺失、GPU 不在、cgroup 配额不足、已有实例）；等机器锁。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_run_rehearsal.py      约 2 分钟（单进程，会起桩进程 2 条车道，不要和 Gazebo 评测同时跑）
"""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2e_arms as ea  # noqa: E402

STUB = HERE / "stub_eval_e2e.py"
PY = os.environ.get("PY_FOR_RUN") or sys.executable
GPU = "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740"
FROZEN_SEEDS = "1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654"


class World:
    """一个演练用的输出目录与全部假输入（A0 权重、两份数据集也用假文件，不碰真实数据）。"""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.out = self.tmp / "out"
        (self.out / "ckpt").mkdir(parents=True)
        self.swa = self.tmp / "swa"
        self.swa.mkdir()
        for s in ea.SEEDS_TRAIN:
            (self.swa / f"cal_swa_s{s}_k10.bin").write_bytes(b"fake-a0-%d" % s)             # A0 的现成权重没有 .json 侧车，这里照样不放
        for a in ("a1", "a2", "a3"):
            for s in ea.SEEDS_TRAIN:
                c = Path(self.ckpt(a, s))
                c.write_bytes(b"fake-weights-%s-%d" % (a.encode(), s))
                Path(str(c) + ".json").write_text(json.dumps({"run": str(self.out / "runs" / f"{ea.NAME[a]}-s{s}")}), encoding="utf-8")
        self.demo_ds = self.tmp / "demo.uavr"
        self.demo_ds.write_bytes(b"fake-demo")
        self.dagger_ds = self.tmp / "dagger.uavr"
        self.dagger_ds.write_bytes(b"fake-dagger")
        (self.out / "PREP_DONE").write_text("rc=0 2026-11-12 00:00:00 演练\n", encoding="utf-8")
        self.lstar = self.tmp / "L_STAR"
        self.lstar.write_text("0.20\n", encoding="utf-8")
        self.smi = self.tmp / "nvidia-smi"
        self.smi.write_text(f"#!/bin/bash\necho 'GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: {GPU})'\n", encoding="utf-8")
        self.smi.chmod(0o755)
        self.cpu_max = self.tmp / "cpu.max"
        self.cpu_max.write_text("1200000 100000\n", encoding="utf-8")
        self.cpu_stat = self.tmp / "cpu.stat"
        self.cpu_stat.write_text("usage_usec 0\nnr_throttled 0\nthrottled_usec 0\n", encoding="utf-8")
        self.infer = self.tmp / "uavrgb_infer"
        self.infer.write_bytes(b"fake-infer")
        self.plugin = self.tmp / "plugin"
        self.plugin.mkdir()
        (self.plugin / "libCfSitlSystem.so").write_bytes(b"fake-plugin")
        self.lock = self.tmp / "machine.lock"
        self.order_log = self.tmp / "order.log"

    def ckpt(self, arm, seed):
        return str(self.swa / f"cal_swa_s{seed}_k10.bin") if arm == "a0" else ea.ckpt_path(str(self.out), arm, seed)

    def env(self, **extra):
        env = {**os.environ, "OUT": str(self.out), "SELFCHECK": "0", "SKIP_SHA_CHECK": "1", "EVAL": str(STUB), "NVIDIA_SMI": str(self.smi),
               "CPU_MAX_FILE": str(self.cpu_max), "CPU_STAT_FILE": str(self.cpu_stat), "MACHINE_LOCK": str(self.lock), "L_STAR_FILE": str(self.lstar),
               "INFER": str(self.infer), "PLUGIN": str(self.plugin), "SWA": str(self.swa), "DEMO_DS": str(self.demo_ds), "DAGGER_DS": str(self.dagger_ds),
               "GZ_PROBE": "echo 0", "IDLE_WINDOW_S": "1", "IDLE_RECHECK_S": "1", "IDLE_LOG_S": "1", "LOCK_LOG_S": "1", "STUB_SLEEP": "0.2",
               "STUB_ORDER_LOG": str(self.order_log), "PY": PY, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
        env.update(extra)
        return env

    def run(self, **extra):
        return subprocess.run(["bash", str(HERE / "run.sh")], env=self.env(**extra), capture_output=True, text=True, timeout=300)

    def done(self):
        p = self.out / "DONE"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def log(self):
        p = self.out / "run.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""


class RunRehearsal(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.w = World(self._tmp.name)

    def assertRc(self, rc):
        done = self.w.done()
        self.assertIsNotNone(done, "没有 DONE\n" + self.w.log()[-1500:])
        self.assertTrue(done.startswith(f"rc={rc} "), f"期望 rc={rc}，DONE={done}\n{self.w.log()[-1500:]}")

    # ---------------------------------------------------------------- 正常跑完
    def test_full_run(self):
        r = self.w.run(STUB_THROTTLE="1")
        self.assertEqual(r.returncode, 0, r.stderr[-800:] + self.w.log()[-1500:])
        self.assertRc(0)
        out = self.w.out
        names = [f"{ea.NAME[a]}-s{s}" for s in ea.SEEDS_TRAIN for a in ea.ARMS]
        for n in names:
            args = (out / n / "stub_args.txt").read_text(encoding="utf-8").split()
            joined = " ".join(args)
            self.assertNotIn("--heading-guard", args, n)
            for frag in ("--policy student", "--task base --steps 80", "--split test", f"--seeds {FROZEN_SEEDS}", "--corridor-half-width 3.0",
                         "--action-latency 0.20", "--episodes 20", "--sim-timeout 120.0 --episode-timeout 1500",
                         "--dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation", f"--infer-bin {self.w.infer}"):
                self.assertIn(frag, joined, n)
            arm = next(a for a in ea.ARMS if n.startswith(ea.NAME[a]))
            seed = int(n.rsplit("-s", 1)[1])
            self.assertIn(f"--ckpt {self.w.ckpt(arm, seed)}", joined, n)
            self.assertEqual("--student-cond" in args, ea.IS_CODE[arm], n)                  # 只有 A1、A2 带 --student-cond
            if ea.IS_CODE[arm]:
                self.assertEqual(args[args.index("--student-cond") + 1], "heading-code", n)
            cg = (out / f"{n}.cg").read_text().strip().splitlines()
            self.assertEqual([ln.split()[0] for ln in cg], ["start", "end"], n)
        # 种子优先：训练种子 s 的四个臂先于训练种子 s+1 的任何臂开始；两条车道各取队列里最前面的臂
        starts = {}
        for line in self.w.order_log.read_text().splitlines():
            n, t = line.split()
            starts[n] = float(t)
        self.assertEqual(sorted(starts), sorted(names))
        order = sorted(starts, key=starts.get)
        for k in range(3):
            self.assertEqual(sorted(order[4 * k:4 * k + 4]), sorted(names[4 * k:4 * k + 4]), f"第 {k} 组")
        self.assertEqual(set(order[:2]), {names[0], names[1]})
        # 起跑记录与输出
        start = (out / "start.txt").read_text(encoding="utf-8")
        for key in ("gpu_ok=是", "cpu_quota_cores=12.00", "L_STAR=0.20", "lanes=2", "idle_cores=", "lock_wait_s=", f"gpu={GPU}",
                    "idle_max_cores=8", "idle_limit_s=600", "idle_forced=0"):                    # 公共规则 §4（01:40 起）的门槛与最长等待
            self.assertIn(key, start)
        cfg = (out / "config.env").read_text(encoding="utf-8")
        self.assertIn("L_STAR=0.20", cfg)
        self.assertIn(f"SEEDS_EVAL={FROZEN_SEEDS}", cfg)
        inputs = (out / "inputs.txt").read_text(encoding="utf-8")
        for a in ea.ARMS:
            for s in ea.SEEDS_TRAIN:
                self.assertIn(self.w.ckpt(a, s), inputs)
        for f in ("summary.md", "verdicts.json", "episodes.csv", "run.pid"):
            self.assertTrue((out / f).exists(), f)
        verdicts = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))
        self.assertTrue(verdicts["complete"])
        summary = (out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("限流增量", summary)
        self.assertIn("⚠", summary)            # STUB_THROTTLE=1：每个臂结束时 nr_throttled 增加，汇总标 ⚠

    # ---------------------------------------------------------------- 迟到集重跑与续跑
    def test_retry_and_resume(self):
        late = {"a1-code-demo-s1": [113, 172], "a3-img-dagger-s2": [1654]}
        r = self.w.run(STUB_LATE=json.dumps(late))
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertRc(0)
        out = self.w.out
        self.assertTrue((out / "a1-code-demo-s1-retry1" / "eval.json").exists())
        self.assertTrue((out / "a3-img-dagger-s2-retry1" / "eval.json").exists())
        self.assertFalse((out / "a0-img-demo-s0-retry1").exists())
        args = (out / "a1-code-demo-s1-retry1" / "stub_args.txt").read_text(encoding="utf-8")
        self.assertIn("--seeds 113,172", args)
        self.assertIn("--episodes 2", args)
        self.assertIn("--student-cond heading-code", args)                              # 重跑沿用原臂的全部参数
        self.assertIn("含迟到步的集 [113,172]，重跑一次 → a1-code-demo-s1-retry1", self.w.log())
        self.assertTrue(json.loads((out / "verdicts.json").read_text(encoding="utf-8"))["complete"])
        # 续跑：同一条命令重起，评完的臂（含重跑）全部跳过
        before = (out / "a0-img-demo-s0" / "eval.json").stat().st_mtime_ns
        retry_before = (out / "a1-code-demo-s1-retry1" / "eval.json").stat().st_mtime_ns
        r2 = self.w.run(STUB_LATE=json.dumps(late))
        self.assertEqual(r2.returncode, 0, self.w.log()[-1500:])
        self.assertEqual((out / "a0-img-demo-s0" / "eval.json").stat().st_mtime_ns, before)
        self.assertEqual((out / "a1-code-demo-s1-retry1" / "eval.json").stat().st_mtime_ns, retry_before)
        self.assertIn("a0-img-demo-s0 已评完，跳过", self.w.log())
        self.assertIn("a1-code-demo-s1-retry1 已评完，跳过", self.w.log())
        self.assertEqual(len(list(out.glob("*.partial-*"))), 0)

    def test_partial_arm_is_moved_aside_and_rerun(self):
        out = self.w.out
        (out / "a2-code-dagger-s0").mkdir()
        (out / "a2-code-dagger-s0" / "eval.json").write_text(json.dumps({"partial": True, "episodes_done": 3, "failed_count": 0}), encoding="utf-8")
        r = self.w.run()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertEqual(len(list(out.glob("a2-code-dagger-s0.partial-*"))), 1)
        self.assertTrue((out / "a2-code-dagger-s0" / "eval.json").exists())
        self.assertFalse(json.loads((out / "a2-code-dagger-s0" / "eval.json").read_text(encoding="utf-8")).get("partial"))

    # ---------------------------------------------------------------- 拒绝起跑
    def test_refuse_without_prep_done(self):
        (self.w.out / "PREP_DONE").unlink()
        self.assertEqual(self.w.run().returncode, 6)
        self.assertRc(6)
        self.assertFalse((self.w.out / "start.txt").exists())

    def test_refuse_when_prep_failed(self):
        (self.w.out / "PREP_DONE").write_text("rc=32 2026-11-12 00:00:00 对拍不一致\n", encoding="utf-8")
        self.assertEqual(self.w.run().returncode, 6)

    def test_refuse_bad_l_star(self):
        self.w.lstar.write_text("abc\n", encoding="utf-8")
        self.assertEqual(self.w.run().returncode, 5)
        self.w.lstar.unlink()
        self.assertEqual(self.w.run().returncode, 5)

    def test_sidecar_required_for_code_arms_only(self):
        Path(self.w.ckpt("a3", 0) + ".json").unlink()                           # 图像臂缺侧车：允许
        r = self.w.run(MIN_CORES="99")                                          # 用 rc=13 拦在评测之前，证明没有在 rc=7 处被拦
        self.assertEqual(r.returncode, 13, self.w.log()[-800:])
        Path(self.w.ckpt("a2", 1) + ".json").unlink()                           # 航向码臂缺侧车：拒绝
        self.assertEqual(self.w.run().returncode, 7)
        self.assertIn("cal_swa_a2-code-dagger_s1_k10.bin.json", self.w.log())

    def test_refuse_missing_weight(self):
        Path(self.w.ckpt("a1", 2)).unlink()
        self.assertEqual(self.w.run().returncode, 7)

    def test_refuse_missing_a0_weight(self):
        Path(self.w.ckpt("a0", 1)).unlink()
        self.assertEqual(self.w.run().returncode, 7)

    def test_refuse_gpu_missing(self):
        self.w.smi.write_text("#!/bin/bash\necho 'GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: GPU-other)'\n", encoding="utf-8")
        self.assertEqual(self.w.run().returncode, 12)

    def test_refuse_small_cgroup(self):
        self.w.cpu_max.write_text("800000 100000\n", encoding="utf-8")
        self.assertEqual(self.w.run().returncode, 13)
        self.assertRc(13)

    def test_refuse_when_instance_running(self):
        (self.w.out / ".run.lock").touch()
        with open(self.w.out / ".run.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            r = self.w.run()
        self.assertEqual(r.returncode, 3)
        self.assertFalse((self.w.out / "DONE").exists())                        # 已有实例时不写 DONE，不盖掉实例自己的结果

    # ---------------------------------------------------------------- 起跑条件（公共规则 §4，2026-11-12 01:40 起）
    def test_below_threshold_starts_without_waiting_for_quiet(self):
        # 容器占用 6 核（< 8）：不要求空闲到 < 1.5 核，立刻起跑
        stop = threading.Event()

        def burn():                                                              # 每 0.1 s 给 cpu.stat 的 usage_usec 加 0.6 s = 6 核
            usage, tmp = 0, str(self.w.cpu_stat) + ".burn"
            while not stop.is_set():
                usage += 600_000
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(f"usage_usec {usage}\nnr_throttled 0\nthrottled_usec 0\n")
                os.replace(tmp, self.w.cpu_stat)                                 # 原子替换，读的一方不会读到半个文件
                time.sleep(0.1)

        th = threading.Thread(target=burn)
        th.start()
        try:
            r = self.w.run(IDLE_WINDOW_S="1", IDLE_LIMIT_S="30")
        finally:
            stop.set()
            th.join()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        start = (self.w.out / "start.txt").read_text(encoding="utf-8")
        self.assertIn("idle_forced=0", start)
        cores = float([ln for ln in start.splitlines() if ln.startswith("idle_cores=")][0].split("=")[1])
        self.assertTrue(4.0 < cores < 8.0, cores)
        self.assertIn("条件 ④ 满足", self.w.log())

    def test_busy_container_starts_at_deadline_and_records_cores(self):
        # 门槛压到 0（永远不满足）+ 最长等待 3 s：到点照样起跑，start.txt 记下 idle_forced=1 与当时的核数
        r = self.w.run(IDLE_MAX_CORES="0", IDLE_LIMIT_S="3", GZ_PROBE="echo 2")
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertRc(0)
        start = (self.w.out / "start.txt").read_text(encoding="utf-8")
        for key in ("idle_forced=1", "idle_limit_s=3", "idle_max_cores=0", "gz_count=2", "idle_cores="):
            self.assertIn(key, start)
        self.assertIn("按公共规则到点起跑", self.w.log())
        self.assertTrue(json.loads((self.w.out / "verdicts.json").read_text(encoding="utf-8"))["complete"])

    # ---------------------------------------------------------------- 等机器锁
    def test_waits_for_machine_lock(self):
        holder = open(self.w.lock, "w")
        fcntl.flock(holder, fcntl.LOCK_EX)
        timer = threading.Timer(3.0, lambda: (fcntl.flock(holder, fcntl.LOCK_UN), holder.close()))
        timer.start()
        t0 = time.time()
        r = self.w.run()
        timer.join()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertGreaterEqual(time.time() - t0, 3.0)
        log = self.w.log()
        for frag in ("条件 ③ 机器锁", "被占用，阻塞等待", "拿到锁"):
            self.assertIn(frag, log)
        wait = [ln for ln in (self.w.out / "start.txt").read_text(encoding="utf-8").splitlines() if ln.startswith("lock_wait_s=")][0]
        self.assertGreaterEqual(int(wait.split("=")[1]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
