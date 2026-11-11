#!/usr/bin/env python3
"""run.sh 整条链的集成测试（干跑：桩驱动 dry_run_driver.py，不碰 GPU、不渲染；FREEZE=tree 冻结当前工作树，所以未提交的改动也被测到）。
用真实的场景 scene.json、试点路线文件与 K1 开发路线文件（只读）选路线，K1 的结果换成桩造的假 K1（dry_run_driver.py --make-k1）。

覆盖：
  整条链走通（选路线 → 开发 → 选变体 → F0 → 正式 → DONE rc=0）并检查各阶段产物的数量与臂表；续跑跳过评完的路线、半成品改名重跑；config.json 与重算不同时拒绝续跑；
  闸门不过（rc=20，DONE 写「开发路线上 v2 没有比 v1 好」，不进正式阶段）；F0 对拍不一致（rc=6，不进正式阶段）；驱动任务失败（rc=7）；
  formal 带覆盖变量起跑被拒（rc=2）；无卡退出（rc=4）；第二个实例被拒（rc=3）；产出超体积上限停整个进程组（rc=5）。
  cgroup 的 usage_usec 与 nr_throttled 起止各记一次（公共规则 §4，2026-11-11 23:55 起）：每个实例 cgroup.txt 里一对「起」「止」，续跑追加，早退（没到记「起」那一步）不留半对。
只用标准库。用法：PYTHONDONTWRITEBYTECODE=1 nice -n 10 python test_run_chain.py   （约 1 分钟；单进程跑，进程里起的是桩驱动与 bash）
"""
import fcntl
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_SH = os.path.join(HERE, "run.sh")
STUB = os.path.join(HERE, "dry_run_driver.py")
DATA = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
PILOT = os.path.join(DATA, "exp-sage3d-pilot-20261110", "routes_selected.json")
K1_DEV = os.path.join(DATA, "exp-rule-oracle-3dgs-20261110", "dev", "indoor", "routes_dev.json")
INDOOR = ["sage3d_841150", "sage3d_840865", "sage3d_841012", "sage3d_840139", "sage3d_839992"]
OUTDOOR = ["garden", "wanderland_1MlUucug", "bicycle", "callinstrasse"]
TMP_ROOT = os.path.join(DATA, "tmp")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def count_eps(root, arm):
    return len(glob.glob(os.path.join(root, arm, "*", "r*", "ep_*", "metrics.json")))


def _cgroup_readable():
    try:
        return "usage_usec" in read_text("/sys/fs/cgroup/cpu.stat") and "nr_throttled" in read_text("/sys/fs/cgroup/cpu.stat")
    except OSError:
        return False


CG_STRICT = _cgroup_readable()          # 这台机器的 cgroup v2 读得到时，usage_usec 与 nr_throttled 必须是数字，不许记 NA
CG_LINE = re.compile(r"^(起|止) \d{4}-\d\d-\d\d \d\d:\d\d:\d\d usage_usec=(\d+|NA) nr_throttled=(\d+|NA)$")


def cgroup_kinds(out):
    """cgroup.txt 的每一行都符合格式；返回行首标记的序列（「起」「止」）。usage_usec 与 nr_throttled 是累计量，数值时不许倒退。"""
    p = os.path.join(out, "cgroup.txt")
    if not os.path.exists(p):
        return []
    kinds, last = [], {"usage_usec": 0, "nr_throttled": 0}
    for ln in read_text(p).splitlines():
        m = CG_LINE.match(ln)
        assert m, f"cgroup.txt 的行格式不对：{ln!r}"
        kinds.append(m.group(1))
        for key, val in (("usage_usec", m.group(2)), ("nr_throttled", m.group(3))):
            assert val != "NA" or not CG_STRICT, f"{key} 读不到（记成 NA）：{ln!r}"
            if val != "NA":
                assert int(val) >= last[key], f"{key} 倒退：{ln!r}"
                last[key] = int(val)
    return kinds


class TestChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in (PILOT, K1_DEV, os.path.join(DATA, "3dgs-scenes", INDOOR[0], "scene.json")):
            if not os.path.exists(p):
                raise unittest.SkipTest(f"缺数据 {p}")
        cls._tmp = tempfile.mkdtemp(prefix="k3-chain-", dir=TMP_ROOT if os.path.isdir(TMP_ROOT) else None)
        cls.k1 = os.path.join(cls._tmp, "k1")
        subprocess.run([sys.executable, STUB, "--make-k1", cls.k1, PILOT, *INDOOR, "--", *OUTDOOR], check=True, capture_output=True)
        os.makedirs(os.path.join(cls.k1, "dev", "indoor"))
        shutil.copy(K1_DEV, os.path.join(cls.k1, "dev", "indoor", "routes_dev.json"))
        cls._n = 0
        cls._happy = None

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def out_dir(self):
        type(self)._n += 1
        return os.path.join(self._tmp, f"out{self._n}")

    def run_chain(self, out, extra=None, dry=True, timeout=180):
        env = {**os.environ, "FREEZE": "tree", "DRIVER": STUB, "K1_ROOT": self.k1, "OUT_ROOT": out, "WATCH_S": "0.2", "LANES": "2", "STUB_SLEEP": "0.1",
               "PY": sys.executable, "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": self._tmp}
        for k in ("STUB_NO_GAIN", "STUB_PERTURB_V1", "STUB_FAIL_SCENE", "STUB_FAIL_CLASS", "SMOKE", "DRY_RUN", "NO_GATE"):
            env.pop(k, None)
        if dry:
            env["DRY_RUN"] = "1"
        env.update(extra or {})
        proc = subprocess.Popen(["setsid", "-w", "bash", RUN_SH], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            so, se = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            pid_file = os.path.join(out, "run.pid")       # 超时：按 PGID 停掉脚本（它在自己的会话里，subprocess 的 kill 够不着），再报错
            if os.path.exists(pid_file):
                try:
                    os.killpg(int(read_text(pid_file)), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            proc.kill()
            proc.communicate()
            raise
        r = types.SimpleNamespace(returncode=proc.returncode, stdout=so, stderr=se)
        done = read_text(os.path.join(out, "DONE")).strip() if os.path.exists(os.path.join(out, "DONE")) else None
        return r, done

    # ---------------------------------------------------------------- 走通、续跑、改 config 拒绝
    def test_happy_path_then_resume_then_changed_config_is_refused(self):
        out = self.out_dir()
        r, done = self.run_chain(out)
        self.assertEqual(r.returncode, 0, (r.stderr, done))
        self.assertTrue(done.startswith("rc=0 "), done)
        self.assertEqual(read_text(os.path.join(out, "phase.txt")).split()[0], "done")
        self.assertEqual(cgroup_kinds(out), ["起", "止"], "一次完整运行：起止各记一次")
        self.assertEqual(read_text(os.path.join(out, "run.log")).count("usage_usec="), 2, "同样的两行也进了 run.log")
        n_dev_lines = read_text(os.path.join(out, "run.log")).count("==== [dev] ")
        self.assertGreater(n_dev_lines, 0)
        # 路线与份数：开发 38、留出 32、正式 40（真实 scene.json 与试点文件选出来的数）
        n_routes = {k: sum(len(v) for v in read_json(os.path.join(out, "routes", f"routes_{k}.json"))["scenes"].values()) for k in ("dev", "holdout", "formal40")}
        self.assertEqual(n_routes, {"dev": 38, "holdout": 32, "formal40": 40})
        # 各阶段臂表与集数
        self.assertEqual([a["name"] for a in read_json(os.path.join(out, "dev", "config.json"))["arms"]], ["v1", "v2a", "v2b", "v2c", "v2d", "v2e", "v2f"])
        self.assertEqual([a["name"] for a in read_json(os.path.join(out, "holdout", "config.json"))["arms"]],
                         ["forward+route", "rule-occ+route", "rule2-occ+route", "rule2-depth+route"])
        self.assertEqual([a["name"] for a in read_json(os.path.join(out, "formal40", "config.json"))["arms"]], ["rule-occ+route", "rule2-occ+route", "rule2-depth+route"])
        self.assertEqual([a["name"] for a in read_json(os.path.join(out, "formal40", "config.f0.json"))["arms"]], ["rule-occ+route"])
        self.assertEqual([a["name"] for a in read_json(os.path.join(out, "outdoor", "config.json"))["arms"]], ["rule2-occ+route", "rule2-depth+route"])
        self.assertEqual(count_eps(os.path.join(out, "dev"), "v1"), 38)
        self.assertEqual({a: count_eps(os.path.join(out, "holdout"), a) for a in ("forward+route", "rule-occ+route", "rule2-occ+route", "rule2-depth+route")}, {a: 96 for a in (
            "forward+route", "rule-occ+route", "rule2-occ+route", "rule2-depth+route")})
        self.assertEqual({a: count_eps(os.path.join(out, "formal40"), a) for a in ("rule-occ+route", "rule2-occ+route", "rule2-depth+route")}, {a: 120 for a in (
            "rule-occ+route", "rule2-occ+route", "rule2-depth+route")})
        self.assertEqual({a: count_eps(os.path.join(out, "outdoor"), a) for a in ("rule2-occ+route", "rule2-depth+route")}, {"rule2-occ+route": 120, "rule2-depth+route": 120})
        # 选变体、F0、记录文件
        sel = read_json(os.path.join(out, "selected.json"))
        self.assertGreaterEqual(sel["gain_over_v1"], 3)
        self.assertFalse(sel["gate_bypassed"])
        self.assertIn("判定：一致", read_text(os.path.join(out, "k1_match.md")))
        inputs = read_text(os.path.join(out, "inputs.txt"))
        self.assertIn("git HEAD", inputs)
        self.assertIn("DRY_RUN=1", inputs)
        self.assertTrue(read_text(os.path.join(out, "code_head.txt")).strip().endswith("+tree"))
        self.assertEqual(read_text(os.path.join(out, "run.pid")).strip().isdigit(), True)
        # 续跑：删 DONE，删一条留出路线的 done.json（它变成半成品，应改名 .incomplete 重跑），其余路线必须被跳过
        victim = sorted(glob.glob(os.path.join(out, "holdout", "rule2-occ+route", "*", "r*", "done.json")))[0]
        victim_dir = os.path.dirname(victim)
        os.remove(victim)
        os.remove(os.path.join(out, "DONE"))
        os.remove(os.path.join(out, "formal40", ".f0_ok"))
        r2, done2 = self.run_chain(out)
        self.assertEqual(r2.returncode, 0, (r2.stderr, done2))
        self.assertEqual(cgroup_kinds(out), ["起", "止", "起", "止"], "续跑追加一对")
        self.assertEqual(read_text(os.path.join(out, "run.log")).count("==== [dev] "), n_dev_lines, "续跑不重跑已完成的 dev 阶段（.tasks_ok 在）")
        self.assertTrue(os.path.exists(victim), "半成品路线应重跑并重新写 done.json")
        self.assertTrue(glob.glob(victim_dir + ".incomplete-*"), "半成品应改名保留")
        logs = "".join(read_text(p) for p in glob.glob(os.path.join(out, "holdout", "driver-*.log")))
        self.assertIn("跳过（done.json 已完成）", logs)
        self.assertEqual(logs.count("跳过（done.json 已完成）") >= 30 * 4, True, "留出路线其余路线（31 条 × 4 臂 + …）都应被跳过")
        # 改 dev/config.json 的内容：与重算的不同 -> 拒绝续跑（rc=2），不动已有结果
        cfg_path = os.path.join(out, "dev", "config.json")
        cfg = read_json(cfg_path)
        cfg["k"] = 2
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        os.remove(os.path.join(out, "DONE"))
        r3, done3 = self.run_chain(out)
        self.assertEqual(r3.returncode, 2, (r3.stderr, done3))
        self.assertIn("与重算的不同", done3)
        self.assertEqual(cgroup_kinds(out), ["起", "止", "起", "止"], "拒绝发生在记「起」之前：不留半对")

    # ---------------------------------------------------------------- 闸门、F0、驱动失败
    def test_gate_failure_stops_before_formal_with_the_card_message(self):
        out = self.out_dir()
        r, done = self.run_chain(out, {"STUB_NO_GAIN": "1"})
        self.assertEqual(r.returncode, 20, (r.stderr, done))
        self.assertTrue(done.startswith("rc=20 "), done)
        self.assertIn("开发路线上 v2 没有比 v1 好", done)
        self.assertEqual(cgroup_kinds(out), ["起", "止"], "失败退出（fail_done）也记「止」")
        self.assertFalse(os.path.exists(os.path.join(out, "selected.json")))
        self.assertEqual(glob.glob(os.path.join(out, "holdout", "*", "*", "r*", "done.json")), [], "不进正式阶段")
        self.assertEqual(glob.glob(os.path.join(out, "outdoor", "*", "*", "r*", "done.json")), [])
        self.assertEqual(glob.glob(os.path.join(out, "formal40", "*", "*", "r*", "done.json")), [], "F0 也不跑")
        self.assertTrue(os.path.exists(os.path.join(out, "dev_selection.md")))

    def test_f0_mismatch_stops_before_formal(self):
        out = self.out_dir()
        r, done = self.run_chain(out, {"STUB_PERTURB_V1": "1"})
        self.assertEqual(r.returncode, 6, (r.stderr, done))
        self.assertIn("F0 对拍不一致", done)
        self.assertIn("不一致", read_text(os.path.join(out, "k1_match.md")))
        self.assertEqual(glob.glob(os.path.join(out, "holdout", "*", "*", "r*", "done.json")), [], "对拍不过就不进正式阶段")
        self.assertEqual(glob.glob(os.path.join(out, "outdoor", "*", "*", "r*", "done.json")), [])
        self.assertTrue(os.path.exists(os.path.join(out, "selected.json")), "选变体已经做完")

    def test_driver_failure_is_reported_as_rc7(self):
        out = self.out_dir()
        r, done = self.run_chain(out, {"STUB_FAIL_SCENE": "sage3d_840865"})
        self.assertEqual(r.returncode, 7, (r.stderr, done))
        self.assertIn("dev 有驱动任务失败", done)
        self.assertFalse(os.path.exists(os.path.join(out, "dev", ".tasks_ok")))

    def test_formal_stage_driver_failure_is_reported_as_rc7(self):
        """dev 与 F0 都过，正式阶段（holdout / formal40 / outdoor 一起跑）里有驱动进程非零退出：xargs 的 123 记成 7，DONE 里写明。"""
        out = self.out_dir()
        r, done = self.run_chain(out, {"STUB_FAIL_SCENE": "sage3d_840865", "STUB_FAIL_CLASS": "holdout"})
        self.assertEqual(r.returncode, 7, (r.stderr, done))
        self.assertTrue(done.startswith("rc=7 "), done)
        self.assertIn("有驱动任务失败", done)
        self.assertTrue(os.path.exists(os.path.join(out, "dev", ".tasks_ok")), "失败发生在正式阶段，dev 与 F0 已完成")
        self.assertTrue(os.path.exists(os.path.join(out, "formal40", ".f0_ok")))
        self.assertEqual(cgroup_kinds(out), ["起", "止"])

    # ---------------------------------------------------------------- 起跑前的拒绝
    def test_formal_refuses_environment_overrides(self):
        if subprocess.run("nvidia-smi -L 2>/dev/null | grep -q GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740", shell=True).returncode != 0:
            self.skipTest("这台机器没有钉死的 GPU：起跑会先因无卡退出（rc=4），覆盖检查测不到")
        out = self.out_dir()
        r, done = self.run_chain(out, dry=False, timeout=60)
        self.assertEqual(r.returncode, 2, (r.stderr, done))
        self.assertIn("formal 不接受环境变量覆盖", done)
        self.assertFalse(os.path.exists(os.path.join(out, "routes")), "拒绝发生在任何实质工作之前")
        self.assertEqual(cgroup_kinds(out), [], "没开始就拒绝：不记「起」也不记「止」")

    def test_no_gpu_exits_rc4_without_work(self):
        out = self.out_dir()
        fake = os.path.join(self._tmp, "fakebin")
        os.makedirs(fake, exist_ok=True)
        with open(os.path.join(fake, "nvidia-smi"), "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\necho 'GPU 0: Fake Card (UUID: GPU-00000000-0000-0000-0000-000000000000)'\n")
        os.chmod(os.path.join(fake, "nvidia-smi"), 0o755)
        r, done = self.run_chain(out, {"PATH": fake + os.pathsep + os.environ["PATH"]}, dry=False, timeout=60)
        self.assertEqual(r.returncode, 4, (r.stderr, done))
        self.assertIn("无卡模式，未起跑", done)
        self.assertFalse(os.path.exists(os.path.join(out, "routes")))

    def test_second_instance_is_refused(self):
        out = self.out_dir()
        os.makedirs(out)
        with open(os.path.join(out, ".lock"), "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            r, _ = self.run_chain(out, timeout=60)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("已有实例在跑", r.stderr)

    # ---------------------------------------------------------------- 体积守护
    def test_size_watchdog_stops_the_whole_group_with_rc5(self):
        out = self.out_dir()
        r, done = self.run_chain(out, {"SIZE_LIMIT_MB": "1", "STUB_SLEEP": "4"}, timeout=120)       # 冻结副本（FREEZE=tree）本身就超过 1 MB：不是只在限额 0 时才触发
        self.assertTrue(done and done.startswith("rc=5 "), (r.returncode, done))
        self.assertIn("超过 1 MB", done)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(cgroup_kinds(out), ["起", "止"], "被看门狗终止（on_term）也记「止」")
        self.assertEqual(glob.glob(os.path.join(out, "holdout", "*", "*", "r*", "done.json")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
