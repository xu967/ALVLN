#!/usr/bin/env python3
"""prepare.sh 的演练（卡 E2E-1）：把它调用的外部程序（打包、对账、训练、平均校准、对拍、离线报告）全部换成桩，只演练它自己的流程控制：
阶段顺序与完成标记、续跑、各个 rc（20/22/23/30/31/32/33/34/12/3/130）、训练池锁的等待与释放、训练命令行（--prev-action 只在 A1、A2 上、配方参数、种子、冻结提交）。
不起 GPU、不用真实数据；桩的行为用环境变量控制。check_recipe.py 用真实脚本，A0 的参照取真实的 A0 config.json。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_prepare_rehearsal.py      约 1 分钟
"""
import fcntl
import json
import os
import shutil
import signal
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

PY = os.environ.get("PY_FOR_RUN") or sys.executable
A0_REAL = Path(ea.A0_RUN_DIR)
GPU = "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740"
ENTRY_NAME = "2026-11-12-e2e-heading-code"

STUBS = {
    "compare_packs.py": '''
import os, sys
fail = os.environ.get("STUB_COMPARE_FAIL", "")
cond = "--cond-pack" in sys.argv
print("compare_packs stub", " ".join(sys.argv[1:]))
if (fail == "default" and not cond) or (fail == "hc" and cond):
    print("RESULT FAIL stub 故意不一致"); sys.exit(1)
print("RESULT OK stub"); sys.exit(0)
''',
    "check_cond_codes.py": '''
import json, os, sys
if os.environ.get("STUB_CODES_FAIL") == "1":
    print("RESULT FAIL stub"); sys.exit(1)
if "--json" in sys.argv:
    open(sys.argv[sys.argv.index("--json") + 1], "w").write(json.dumps({"stub": True}))
print("RESULT OK stub"); sys.exit(0)
''',
    "make_dagger_dirs.py": '''
import os, sys
n = int(os.environ.get("STUB_DAGGER_LINES", "592"))
open(sys.argv[1], "w").write("".join("/fake/dagger/ep_%d/\\n" % i for i in range(n)))
''',
    "verify_infer_vs_evalonly.py": '''
import os, sys
print("verify stub", " ".join(sys.argv[1:]))
if os.environ.get("STUB_VERIFY_FAIL") == "1":
    print("RESULT FAIL stub"); sys.exit(1)
print("RESULT OK stub"); sys.exit(0)
''',
    "offline_report.py": '''
import os, sys
if os.environ.get("STUB_OFFLINE_FAIL") == "1":
    print("boom"); sys.exit(1)
root = sys.argv[sys.argv.index("--out-root") + 1]
os.makedirs(os.path.join(root, "offline"), exist_ok=True)
open(os.path.join(root, "offline", "offline.json"), "w").write("{}")
print("offline stub ok")
''',
}
TOOL_STUBS = {
    "preprocess.py": '''
import os, sys
a = sys.argv[1:]
out = a[a.index("--out") + 1]
flags = [x for x in a if x.startswith("--")]
dirs = [x for x in a if not x.startswith("--") and x != out and x != "heading-code"]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "dataset.uavr"), "wb").write(("pack %s dirs=%d" % (" ".join(flags), len(dirs))).encode())
open(os.path.join(out, "dataset.aux"), "wb").write(b"aux")
print("preprocess stub", " ".join(flags), "dirs", len(dirs))
''',
    "average_ckpt.py": '''
import json, os, sys
a = sys.argv[1:]
get = lambda k: a[a.index(k) + 1]
if os.environ.get("STUB_AVG_FAIL") == "1":
    print("boom"); sys.exit(1)
import time
time.sleep(float(os.environ.get("STUB_AVG_SLEEP", "0")))
assert os.path.exists(os.path.join(get("--run"), "ckpt_ep30.bin")), "run 里没有 ckpt_ep30.bin"
open(get("--out"), "wb").write(b"avg:" + get("--last-k").encode())
open(get("--out") + ".json", "w").write(json.dumps({"run": get("--run"), "data": get("--data"), "last_k": get("--last-k")}))
print("average stub ok")
''',
}
TRAINER = '''#!/usr/bin/env python3
import hashlib, json, os, sys, time
a = sys.argv[1:]
get = lambda k: a[a.index(k) + 1]
data, out, seed = get("--data"), get("--out"), int(get("--seed"))
print("ARGS", " ".join(a), "SOURCE_GIT=" + os.environ.get("UAVRGB_SOURCE_GIT", ""), flush=True)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "metrics.jsonl"), "w").write("{}\\n")
time.sleep(float(os.environ.get("STUB_TRAIN_SLEEP", "0.5")))
if os.environ.get("STUB_TRAIN_FAIL_SEED") not in (None, "") and seed == int(os.environ["STUB_TRAIN_FAIL_SEED"]):
    print("boom", flush=True); sys.exit(7)
for e in range(1, 31):
    open(os.path.join(out, "ckpt_ep%02d.bin" % e), "wb").write(b"w")
cfg = json.load(open(os.environ["STUB_REF_CONFIG"]))
cfg.update({"data": os.path.abspath(data), "dataset_sha256": hashlib.sha256(open(data, "rb").read()).hexdigest(), "seed": seed,
            "prev_action": "--prev-action" in a, "source_git": os.environ.get("UAVRGB_SOURCE_GIT", "")})
if "dagger" in data:
    cfg["class_weights"] = "[1.0, 2.0, 3.0, 4.0]"
bad = os.environ.get("STUB_BAD_LR_SUBSTR")
if bad and bad in data:
    cfg["lr"] = 0.002
json.dump(cfg, open(os.path.join(out, "config.json"), "w"))
print("epoch 30 done", flush=True)
'''


class World:
    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.out = self.tmp / "out"
        self.out.mkdir()
        (self.out / "code_head.txt").write_text("0123456789abcdef0123456789abcdef01234567\n", encoding="utf-8")
        self.code = self.tmp / "code"
        entry = self.code / "experiments" / ENTRY_NAME
        entry.mkdir(parents=True)
        for f in ("prepare.sh", "lib.sh", "check_recipe.py"):
            shutil.copy(HERE / f, entry / f)
        for name, text in STUBS.items():
            (entry / name).write_text(text, encoding="utf-8")
        (self.code / "tools").mkdir()
        for name, text in TOOL_STUBS.items():
            (self.code / "tools" / name).write_text(text, encoding="utf-8")
        (self.code / "sim" / "tools").mkdir(parents=True)
        (self.code / "sim" / "tools" / "nav_cond.py").write_text("", encoding="utf-8")
        self.entry = entry
        self.a0_run = self.tmp / "a0runs"
        for s in ea.SEEDS_TRAIN:
            d = self.a0_run / f"seed{s}" / "run"
            d.mkdir(parents=True)
            shutil.copy(A0_REAL / f"seed{s}" / "run" / "config.json", d / "config.json")
        self.ref_config = str(self.a0_run / "seed0" / "run" / "config.json")
        self.swa = self.tmp / "swa"
        self.swa.mkdir()
        for s in ea.SEEDS_TRAIN:
            (self.swa / f"cal_swa_s{s}_k10.bin").write_bytes(b"a0-%d" % s)
        self.col = self.tmp / "col"
        for split, name in (("train", "ep1"), ("train", "ep2"), ("val", "ep3"), ("test", "ep4")):
            (self.col / split / name).mkdir(parents=True)
        self.demo_ds = self.tmp / "demo.uavr"
        self.demo_ds.write_bytes(b"demo")
        self.dagger_ds = self.tmp / "dagger.uavr"
        self.dagger_ds.write_bytes(b"dagger")
        self.infer = self.tmp / "uavrgb_infer"
        self.infer.write_bytes(b"infer")
        self.train_src = self.tmp / "train_stub"
        self.train_src.write_text(TRAINER, encoding="utf-8")
        self.train_src.chmod(0o755)
        self.smi = self.tmp / "nvidia-smi"
        self.smi.write_text(f"#!/bin/bash\necho 'GPU 0: NVIDIA GeForce RTX 2080 Ti (UUID: {GPU})'\n", encoding="utf-8")
        self.smi.chmod(0o755)
        self.cpu_stat = self.tmp / "cpu.stat"
        self.cpu_stat.write_text("usage_usec 0\nnr_throttled 0\nthrottled_usec 0\n", encoding="utf-8")
        self.lock = self.tmp / "train.lock"

    def env(self, **extra):
        env = {**os.environ, "OUT": str(self.out), "SELFCHECK": "0", "SKIP_SHA_CHECK": "1", "MIN_FREE_GB": "0", "PY": PY, "NVIDIA_SMI": str(self.smi),
               "TRAIN_SRC": str(self.train_src), "INFER": str(self.infer), "DEMO_COL": str(self.col), "DEMO_DS": str(self.demo_ds), "DAGGER_DS": str(self.dagger_ds),
               "SWA": str(self.swa), "A0_RUN_DIR": str(self.a0_run), "TRAIN_LOCK": str(self.lock), "CPU_STAT_FILE": str(self.cpu_stat), "LOCK_LOG_S": "1",
               "STUB_REF_CONFIG": self.ref_config, "STUB_TRAIN_SLEEP": "0.5", "LAST_K": "10", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
        env.update(extra)
        return env

    def data_for(self, arm):
        return {"a0": str(self.demo_ds), "a1": str(self.out / "data" / "demo-hc" / "dataset.uavr"),
                "a2": str(self.out / "data" / "dagger-hc" / "dataset.uavr"), "a3": str(self.dagger_ds)}[arm]

    def cmd(self):
        return ["bash", str(self.entry / "prepare.sh")]

    def run(self, **extra):
        return subprocess.run(self.cmd(), env=self.env(**extra), capture_output=True, text=True, timeout=300)

    def prep_done(self):
        p = self.out / "PREP_DONE"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def log(self):
        p = self.out / "prepare.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def stages(self):
        d = self.out / "stage"
        return sorted(p.stem for p in d.glob("*.done")) if d.exists() else []


class PrepareRehearsal(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.w = World(self._tmp.name)

    def assertRc(self, rc):
        done = self.w.prep_done()
        self.assertIsNotNone(done, "没有 PREP_DONE\n" + self.w.log()[-1500:])
        self.assertTrue(done.startswith(f"rc={rc} "), f"期望 rc={rc}，PREP_DONE={done}\n{self.w.log()[-1500:]}")

    # ---------------------------------------------------------------- 全流程
    def test_full_flow(self):
        r = self.w.run()
        self.assertEqual(r.returncode, 0, r.stderr[-800:] + self.w.log()[-2000:])
        self.assertRc(0)
        out = self.w.out
        self.assertEqual(self.w.stages(), sorted(["preflight", "pack", "train", "recipe", "calibrate", "verify", "offline"]))
        # 打包：默认包的 dataset.uavr 删了、哈希与侧车留着；航向码包两份都在；dagger 清单 592 项
        sums = (out / "recon" / "SHA256SUMS").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(sums), 2)
        for name in ("demo", "dagger"):
            self.assertFalse((out / "recon" / f"{name}-default" / "dataset.uavr").exists(), name)
            self.assertTrue((out / "recon" / f"{name}-default" / "dataset.aux").exists(), name)
            hc_pack = (out / "data" / f"{name}-hc" / "dataset.uavr").read_text(encoding="utf-8")
            self.assertIn("--cond", hc_pack)                                          # 桩把收到的开关写进包内容：航向码包带 --cond heading-code
            self.assertIn("--drop-collision", hc_pack)
            self.assertNotIn("--cond", (out / "recon" / f"{name}-default.log").read_text(encoding="utf-8"))   # 默认重打包不带新开关
        self.assertIn("--allow-mixed-dynamics", (out / "data" / "dagger-hc" / "dataset.uavr").read_text(encoding="utf-8"))
        self.assertNotIn("--allow-mixed-dynamics", (out / "data" / "demo-hc" / "dataset.uavr").read_text(encoding="utf-8"))
        self.assertEqual(len((out / "dagger_dirs.txt").read_text(encoding="utf-8").splitlines()), 592)
        self.assertIn("dirs=592", (out / "data" / "dagger-hc" / "dataset.uavr").read_text(encoding="utf-8"))
        self.assertIn("dirs=4", (out / "data" / "demo-hc" / "dataset.uavr").read_text(encoding="utf-8"))               # demo：train 2 + val 1 + test 1
        # 对账顺序：先默认后航向码；航向码对账带 --cond-pack 与 --aux-ref
        for name in ("demo", "dagger"):
            self.assertNotIn("--cond-pack", (out / "recon" / f"compare-{name}-default.log").read_text(encoding="utf-8"))
            hc = (out / "recon" / f"compare-{name}-hc.log").read_text(encoding="utf-8")
            self.assertIn("--cond-pack", hc)
            self.assertIn(f"--aux-ref {out}/recon/{name}-default/dataset.aux", hc)
            self.assertTrue((out / "recon" / f"codes-{name}-hc.json").exists())
        # 训练：9 个 run，命令行逐项核对
        expect_flags = "--epochs 30 --batch 128 --lr 0.001 --class-weights auto --drop-collision --save-every-epoch"
        for a in ("a1", "a2", "a3"):
            for s in ea.SEEDS_TRAIN:
                run = Path(ea.run_dir(str(out), a, s))
                first = (Path(str(run) + ".log")).read_text(encoding="utf-8").splitlines()[0]
                self.assertIn(f"--data {self.w.data_for(a)} --out {run} {expect_flags} --seed {s}", first, a)
                self.assertEqual("--prev-action" in first, ea.IS_CODE[a], first)
                self.assertTrue(first.endswith("SOURCE_GIT=0123456789abcdef0123456789abcdef01234567"), first)
                self.assertTrue((run / "ckpt_ep30.bin").exists())
                ck = Path(ea.ckpt_path(str(out), a, s))
                self.assertTrue(ck.exists() and Path(str(ck) + ".json").exists(), ck)
                self.assertEqual(ck.read_bytes(), b"avg:10")
        self.assertTrue((out / "train.cg").exists())
        self.assertIn("释放训练池锁", self.w.log())
        # 12 个对拍、离线报告、配方核对日志
        self.assertEqual(len(list((out / "offline" / "verify").glob("*.log"))), 12)
        self.assertTrue((out / "offline" / "offline.json").exists())
        self.assertIn("RESULT OK 核对 12 个 run", (out / "recon" / "recipe.log").read_text(encoding="utf-8"))
        self.assertTrue((out / "inputs-prepare.txt").exists())

    # ---------------------------------------------------------------- 续跑
    def test_resume_only_reruns_unfinished_stages(self):
        self.assertEqual(self.w.run().returncode, 0, self.w.log()[-1500:])
        marker = self.w.out / "ckpt" / f"cal_swa_a1-code-demo_s0_k10.bin"
        mtime = marker.stat().st_mtime_ns
        runlog = Path(ea.run_dir(str(self.w.out), "a2", 1) + ".log")
        runlog_mtime = runlog.stat().st_mtime_ns
        (self.w.out / "stage" / "verify.done").unlink()
        (self.w.out / "stage" / "offline.done").unlink()
        r = self.w.run()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertRc(0)
        self.assertEqual(marker.stat().st_mtime_ns, mtime)                        # 校准没重做
        self.assertEqual(runlog.stat().st_mtime_ns, runlog_mtime)                 # 训练没重做
        self.assertEqual(len(list(self.w.out.glob("**/*.partial-*"))), 0)

    def test_resume_inside_calibrate_skips_finished_weights(self):
        self.assertEqual(self.w.run().returncode, 0)
        for f in ("calibrate", "verify", "offline"):
            (self.w.out / "stage" / f"{f}.done").unlink()
        victim = Path(ea.ckpt_path(str(self.w.out), "a3", 2))
        victim.unlink()
        Path(str(victim) + ".json").unlink()
        r = self.w.run()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertTrue(victim.exists())
        self.assertEqual(self.w.log().count("已有交付权重，跳过"), 8)

    # ---------------------------------------------------------------- 各个 rc
    def test_pack_default_mismatch_stops_before_hc_and_training(self):
        r = self.w.run(STUB_COMPARE_FAIL="default")
        self.assertEqual(r.returncode, 20)
        self.assertRc(20)
        self.assertEqual(self.w.stages(), ["preflight"])
        self.assertFalse((self.w.out / "data" / "demo-hc").exists())
        self.assertEqual(list((self.w.out / "runs").iterdir()), [])

    def test_pack_hc_mismatch_stops_before_training(self):
        self.assertEqual(self.w.run(STUB_COMPARE_FAIL="hc").returncode, 20)
        self.assertEqual(self.w.stages(), ["preflight"])
        self.assertEqual(list((self.w.out / "runs").iterdir()), [])

    def test_codes_mismatch_stops(self):
        self.assertEqual(self.w.run(STUB_CODES_FAIL="1").returncode, 20)
        self.assertIn("独立重算", self.w.prep_done())

    def test_dagger_list_wrong_length_stops(self):
        self.assertEqual(self.w.run(STUB_DAGGER_LINES="591").returncode, 20)
        self.assertIn("592", self.w.prep_done())
        self.assertEqual(self.w.stages(), ["preflight"])

    def test_train_failure(self):
        r = self.w.run(STUB_TRAIN_FAIL_SEED="1")
        self.assertEqual(r.returncode, 30)
        self.assertRc(30)
        self.assertNotIn("train", self.w.stages())
        self.assertIn("a1-code-demo s1 训练失败", self.w.prep_done())

    def test_recipe_mismatch_stops_before_calibrate(self):
        r = self.w.run(STUB_BAD_LR_SUBSTR="dagger.uavr")                         # 只有 A3 的 lr 被改
        self.assertEqual(r.returncode, 34)
        self.assertRc(34)
        self.assertEqual(self.w.stages(), ["pack", "preflight", "train"])
        text = (self.w.out / "recon" / "recipe.log").read_text(encoding="utf-8")
        self.assertIn("RESULT FAIL", text)
        self.assertIn("键 lr", text)
        self.assertEqual(len(list((self.w.out / "ckpt").iterdir())), 0)

    def test_calibrate_verify_offline_failures(self):
        self.assertEqual(self.w.run(STUB_AVG_FAIL="1").returncode, 31)
        self.assertEqual(self.w.run(STUB_VERIFY_FAIL="1").returncode, 32)          # 续跑：校准已完成，卡在对拍
        self.assertEqual(self.w.stages(), ["calibrate", "pack", "preflight", "recipe", "train"])
        self.assertEqual(self.w.run(STUB_OFFLINE_FAIL="1").returncode, 33)
        self.assertEqual(self.w.stages(), ["calibrate", "pack", "preflight", "recipe", "train", "verify"])
        self.assertEqual(self.w.run().returncode, 0)
        self.assertEqual(len(self.w.stages()), 7)

    def test_preflight_refusals(self):
        self.w.smi.write_text("#!/bin/bash\necho 'GPU 0: x (UUID: GPU-other)'\n", encoding="utf-8")
        self.assertEqual(self.w.run().returncode, 12)
        self.w.smi.write_text(f"#!/bin/bash\necho 'GPU 0: x (UUID: {GPU})'\n", encoding="utf-8")
        self.w.demo_ds.unlink()
        self.assertEqual(self.w.run().returncode, 22)
        self.w.demo_ds.write_bytes(b"demo")
        self.assertEqual(self.w.run(MIN_FREE_GB="999999999").returncode, 23)

    def test_sha_check_catches_changed_a0_weight(self):
        r = self.w.run(SKIP_SHA_CHECK="0")
        self.assertEqual(r.returncode, 21)
        self.assertIn("sha256", self.w.prep_done())

    def test_refuse_when_instance_running(self):
        with open(self.w.out / ".prepare.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            r = self.w.run()
        self.assertEqual(r.returncode, 3)
        self.assertIsNone(self.w.prep_done())

    # ---------------------------------------------------------------- 训练池锁与信号
    def test_waits_for_train_lock_and_packs_first(self):
        holder = open(self.w.lock, "w")
        fcntl.flock(holder, fcntl.LOCK_EX)
        timer = threading.Timer(4.0, lambda: (fcntl.flock(holder, fcntl.LOCK_UN), holder.close()))
        timer.start()
        r = self.w.run()
        timer.join()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        log = self.w.log()
        self.assertIn("被占用，阻塞等待", log)
        self.assertLess(log.index("阶段 pack 完成"), log.index("被占用，阻塞等待"))      # 打包不占训练池锁，先做完
        self.assertLess(log.index("拿到锁"), log.index("a1-code-demo s0 开始训练"))

    def test_train_lock_is_released_when_training_ends(self):
        seen = []

        def watcher():
            deadline = time.time() + 90
            while time.time() < deadline and "释放训练池锁" not in self.w.log():
                time.sleep(0.05)
            for _ in range(40):                                                  # 2 s 内应当能拿到锁，且 prepare 还没结束（校准桩每个权重睡 0.5 s）
                with open(self.w.lock, "a") as lk:
                    try:
                        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        seen.append(self.w.prep_done() is None)
                        return
                    except BlockingIOError:
                        pass
                time.sleep(0.05)
            seen.append(False)

        th = threading.Thread(target=watcher)
        th.start()
        r = self.w.run(STUB_AVG_SLEEP="0.5")
        th.join()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        self.assertEqual(seen, [True], "训练结束后训练池锁应已释放，且此时 prepare 仍在跑后续阶段")

    def test_sigterm_then_resume(self):
        proc = subprocess.Popen(self.w.cmd(), env=self.w.env(STUB_TRAIN_SLEEP="30"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.time() + 60
        while time.time() < deadline and "a1-code-demo s0 开始训练" not in self.w.log():
            time.sleep(0.3)
        self.assertIn("a1-code-demo s0 开始训练", self.w.log())
        os.killpg(proc.pid, signal.SIGTERM)                                         # 与正式停法相同：只按进程组
        proc.wait(timeout=30)
        time.sleep(1.0)
        self.assertRc(130)
        self.assertNotIn("train", self.w.stages())
        # 同一条命令重起：半成品挪走，重新训练到底
        r = self.w.run()
        self.assertEqual(r.returncode, 0, self.w.log()[-1500:])
        partial = sorted(p.name for p in (self.w.out / "runs").glob("*.partial-*") if p.is_dir())
        self.assertEqual(len(partial), 3, partial)                                 # A1 的三个种子被打断
        self.assertTrue(all(n.startswith("a1-code-demo-s") for n in partial), partial)


if __name__ == "__main__":
    unittest.main(verbosity=2)
