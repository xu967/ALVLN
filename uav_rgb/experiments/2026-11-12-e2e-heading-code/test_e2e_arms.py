#!/usr/bin/env python3
"""卡 E2E-1 入口里两份「臂表」（lib.sh 的 bash 函数、e2e_arms.py 的 Python 常量）逐项一致；训练配方与 A0 的 config.json 一致；check_recipe.py 的判定。纯 CPU。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_e2e_arms.py
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
import check_recipe as cr  # noqa: E402
import e2e_arms as ea  # noqa: E402

LIB = os.path.join(HERE, "lib.sh")
A0_CONFIG = os.path.join(ea.A0_RUN_DIR, "seed0", "run", "config.json")


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def sh(out, expr, extra_env=None):
    env = {**os.environ, **(extra_env or {})}
    r = subprocess.run(["bash", "-c", f"export OUT={shlex.quote(out)}; source {shlex.quote(LIB)}; {expr}"], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout.rstrip("\n")


class ArmTableAgreement(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = self._tmp.name

    def test_names_labels_flags(self):
        for a in ea.ARMS:
            self.assertEqual(sh(self.out, f"arm_name {a}")[1], ea.NAME[a])
            self.assertEqual(sh(self.out, f"arm_label {a}")[1], ea.LABEL[a])
            self.assertEqual(sh(self.out, f"arm_is_code {a}")[0] == 0, ea.IS_CODE[a], a)
            self.assertEqual(sh(self.out, f"arm_data {a}")[1], ea.data_path(self.out, a))

    def test_paths_per_seed(self):
        for a in ea.ARMS:
            for s in ea.SEEDS_TRAIN:
                self.assertEqual(sh(self.out, f"arm_run {a} {s}")[1], ea.run_dir(self.out, a, s), (a, s))
                self.assertEqual(sh(self.out, f"arm_ckpt {a} {s}")[1], ea.ckpt_path(self.out, a, s), (a, s))
                self.assertEqual(sh(self.out, f"arm_evdir {a} {s}")[1], ea.evdir(self.out, a, s), (a, s))

    def test_seed_lists(self):
        import eval_seeds
        self.assertEqual(sh(self.out, 'echo "$SEEDS_EVAL"')[1], ",".join(str(x) for x in eval_seeds.EVAL_SEEDS_BASE_20))
        self.assertEqual(sh(self.out, 'echo "$SEEDS_TRAIN"')[1], " ".join(str(s) for s in ea.SEEDS_TRAIN))
        self.assertEqual(sh(self.out, 'echo "$ARMS"')[1], " ".join(ea.ARMS))

    def test_a0_points_at_existing_weights(self):
        # A0 不重训：权重与训练 run 是 W1-5 / 11-09 的现成文件
        for s in ea.SEEDS_TRAIN:
            self.assertTrue(os.path.isfile(ea.ckpt_path(self.out, "a0", s)), ea.ckpt_path(self.out, "a0", s))
            self.assertTrue(os.path.isfile(os.path.join(ea.run_dir(self.out, "a0", s), "config.json")))
            expect = sh(self.out, 'echo "${A0_SHA_EXPECT[%d]}"' % s)[1]
            import hashlib
            with open(ea.ckpt_path(self.out, "a0", s), "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), expect)

    @unittest.skipUnless(os.path.isfile(A0_CONFIG), "A0 的 config.json 不在本机")
    def test_recipe_flags_match_a0_config(self):
        flags = sh(self.out, 'echo "$TRAIN_FLAGS"')[1].split()
        opts = {}
        i = 0
        while i < len(flags):
            if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
                opts[flags[i]] = flags[i + 1]
                i += 2
            else:
                opts[flags[i]] = True
                i += 1
        with open(A0_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(int(opts["--epochs"]), cfg["epochs"])
        self.assertEqual(int(opts["--batch"]), cfg["batch"])
        self.assertAlmostEqual(float(opts["--lr"]), cfg["lr"], places=8)
        self.assertEqual(opts["--class-weights"], cfg["class_weights_source"])
        self.assertTrue(opts["--drop-collision"] and cfg["drop_collision"])
        self.assertTrue(opts["--save-every-epoch"])
        self.assertEqual(set(opts) - {"--epochs", "--batch", "--lr", "--class-weights", "--drop-collision", "--save-every-epoch"}, set())


class RecipeCheckTests(unittest.TestCase):
    REF = {"schema": "uavrgb.train_config.v1", "data": "/d/demo.uavr", "epochs": 30, "batch": 128, "lr": 0.00100000005, "seed": 0, "drop_collision": True,
           "class_weights": "[36.6294632, 0.29793027, 3.21009398, 3.28200006]", "class_weights_source": "auto", "dataset_sha256": "aa", "source_git": "g0",
           "prev_action": False, "beta1": 0.899999976, "augment": ""}

    def cfg(self, **kw):
        c = dict(self.REF)
        c.update(kw)
        return c

    def test_img_demo_same_recipe_passes(self):
        self.assertEqual(cr.diff_config(self.REF, self.cfg(seed=1, source_git="g1", data="/x/demo.uavr", dataset_sha256="bb"), "img", 1, "demo"), [])

    def test_code_arm_needs_prev_action_true(self):
        self.assertEqual(cr.diff_config(self.REF, self.cfg(prev_action=True), "code", 0, "demo"), [])
        self.assertTrue(cr.diff_config(self.REF, self.cfg(prev_action=False), "code", 0, "demo"))
        self.assertTrue(cr.diff_config(self.REF, self.cfg(prev_action=True), "img", 0, "demo"))

    def test_changed_hyperparameter_is_caught(self):
        for key, val in (("epochs", 20), ("batch", 64), ("lr", 0.002), ("drop_collision", False), ("class_weights_source", "none"), ("augment", "x")):
            self.assertTrue(cr.diff_config(self.REF, self.cfg(**{key: val}), "img", 0, "demo"), key)

    def test_extra_or_missing_key_is_caught(self):
        c = self.cfg()
        c["new_key"] = 1
        self.assertTrue(cr.diff_config(self.REF, c, "img", 0, "demo"))
        c = self.cfg()
        del c["beta1"]
        self.assertTrue(cr.diff_config(self.REF, c, "img", 0, "demo"))

    def test_seed_must_match_declared(self):
        self.assertTrue(cr.diff_config(self.REF, self.cfg(seed=2), "img", 1, "demo"))

    def test_data_must_be_absolute(self):
        self.assertTrue(cr.diff_config(self.REF, self.cfg(data="rel/demo.uavr"), "img", 0, "demo"))

    def test_class_weights_demo_must_equal_dagger_free(self):
        other = "[1.0, 2.0, 3.0, 4.0]"
        self.assertTrue(cr.diff_config(self.REF, self.cfg(class_weights=other), "img", 0, "demo"))
        self.assertEqual(cr.diff_config(self.REF, self.cfg(class_weights=other), "img", 0, "dagger"), [])

    def test_cli_on_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = os.path.join(tmp, "ref.json")
            dump_json(ref, self.REF)
            ok = os.path.join(tmp, "ok")
            os.makedirs(ok)
            dump_json(os.path.join(ok, "config.json"), self.cfg(seed=1, prev_action=True))
            bad = os.path.join(tmp, "bad")
            os.makedirs(bad)
            dump_json(os.path.join(bad, "config.json"), self.cfg(seed=1, prev_action=True, epochs=10))
            r = subprocess.run([sys.executable, os.path.join(HERE, "check_recipe.py"), "--ref", ref, f"a1-s1={ok}:code:1:demo"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout)
            r = subprocess.run([sys.executable, os.path.join(HERE, "check_recipe.py"), "--ref", ref, f"a1-s1={ok}:code:1:demo", f"a1-s2={bad}:code:1:demo"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("不一致 1 个", r.stdout)
            r = subprocess.run([sys.executable, os.path.join(HERE, "check_recipe.py"), "--ref", ref, f"x={tmp}/nope:code:1:demo"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stdout)

    @unittest.skipUnless(os.path.isfile(A0_CONFIG), "A0 的 config.json 不在本机")
    def test_a0_own_runs_pass_against_seed0(self):
        # A0 的三个训练种子对 seed0：除 seed 以外全同（自检：核对工具对真实配置不会误报）
        for s in ea.SEEDS_TRAIN:
            run = ea.run_dir("/nonexistent", "a0", s)
            with open(os.path.join(run, "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            with open(A0_CONFIG, encoding="utf-8") as f:
                ref = json.load(f)
            self.assertEqual(cr.diff_config(ref, cfg, "img", s, "demo"), [], s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
