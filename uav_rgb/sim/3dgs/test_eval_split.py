#!/usr/bin/env python3
"""eval_3dgs 的评测集划分取自场景身份（审核 D1，卡 S1a）。

以前 eval_3dgs 把评测集的 split 写死成 "test"、split_policy 写死成 diag-eval-fixed-test，不看场景按几何组哈希应属的 split。
9 个 3DGS 评测场景按哈希是 8 个 train、1 个 val、0 个 test（卡 S1a 报告里有逐个场景的表），
在它们上评测得到的是训练 / 验证场景上的数字，产物里却标成 test。

现在：哈希 split 是 test，产物与以前逐位相同；不是 test 默认拒绝（退出码非 0，什么都不写），--allow-train-scene
才放行并写真实 split 与 split_policy = diag-train-scene，产物里不再出现 split = "test"。

合成走廊的哈希 split 取决于 PLY 内容：各用例按需要的 split 在 seed 里现找，不写死 seed。
跑法（要 numpy；不需要 torch / GPU）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_eval_split.py
UAV_RGB_ROOT_UNDER_TEST=<另一份代码树> 可以拿同一份用例去跑那份代码（改前红 / 改后绿的对照就是这么出的）。
"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.environ.get("UAV_RGB_ROOT_UNDER_TEST") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
scene_mod = importlib.import_module("sim.3dgs.scene")
collect = importlib.import_module("sim.3dgs.collect_3dgs")
eval_mod = importlib.import_module("sim.3dgs.eval_3dgs")
import split_manifest as sm  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP   # 子进程也不要写 /tmp

_SEEDS = {}     # 哈希 split -> 第一个落在这个 split 的走廊 seed


def corridor_seed_for(want):
    if want not in _SEEDS:
        for seed in range(200):
            with tempfile.TemporaryDirectory() as tmp:
                scene_mod.make_corridor(os.path.join(tmp, "s"), seed=seed, length=12.0)
                got = collect.scene_identity(scene_mod.load_scene(os.path.join(tmp, "s", "scene.json")))["split"]
            _SEEDS.setdefault(got, seed)
            if want in _SEEDS:
                break
    if want not in _SEEDS:
        raise unittest.SkipTest(f"200 个 seed 里没有哈希 split = {want} 的合成走廊")
    return _SEEDS[want]


def make_scene(tmp, want):
    """合成走廊，按几何哈希落在 want 这个 split；返回 (scene.json 路径, 场景身份)。"""
    scene_dir = os.path.join(tmp, "scene")
    scene_mod.make_corridor(scene_dir, seed=corridor_seed_for(want), length=12.0)
    path = os.path.join(scene_dir, "scene.json")
    identity = collect.scene_identity(scene_mod.load_scene(path))
    assert identity["split"] == want, identity
    return path, identity


def run_eval(scene_json, out, *extra):
    return subprocess.run(
        [sys.executable, "-m", "sim.3dgs.eval_3dgs", "--scene", scene_json, "--out", out,
         "--policy", "forward", "--mock-renderer", "--dynamics", "cf_blackbox", "--episodes", "1",
         "--steps", "2", "--hold", "0.5", *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=180)


def _load(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def json_files(root):
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith(".json"):
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as f:
                    yield os.path.relpath(path, root), json.load(f)


def split_values(obj):
    """JSON 里所有叫 split 的键的值（含嵌套）。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "split":
                yield v
            yield from split_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from split_values(v)


class TestEvalSplitFunction(unittest.TestCase):
    def test_test_scene_keeps_the_old_pair_either_way(self):
        ident = {"split": "test", "geometry_gid": "g" * 16}
        for allow in (False, True):
            self.assertEqual(eval_mod.eval_split(ident, allow), ("test", "diag-eval-fixed-test"))

    def test_train_and_val_scenes_are_refused_by_default(self):
        for split in ("train", "val"):
            with self.assertRaises(ValueError) as cm:
                eval_mod.eval_split({"split": split, "geometry_gid": "abcd1234abcd1234"}, False)
            msg = str(cm.exception)
            self.assertIn("--allow-train-scene", msg)
            self.assertIn(split, msg)
            self.assertIn("abcd1234abcd1234", msg)

    def test_allowed_train_and_val_scenes_get_their_real_split(self):
        for split in ("train", "val"):
            self.assertEqual(eval_mod.eval_split({"split": split, "geometry_gid": "g" * 16}, True), (split, "diag-train-scene"))

    def test_both_policies_are_diagnostic_so_preprocess_refuses_them_by_default(self):
        self.assertTrue(sm.is_diagnostic_scheme("diag-eval-fixed-test"))
        self.assertTrue(sm.is_diagnostic_scheme("diag-train-scene"))


class TestEvalCli(unittest.TestCase):
    def test_hash_test_scene_default_output_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, ident = make_scene(tmp, "test")
            out = os.path.join(tmp, "eval")
            r = run_eval(path, out)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            e = _load(out, "eval.json")
            m = _load(out, "ep_00000", "episode_manifest.json")
            self.assertEqual((e["scene"]["split"], e["scene"]["split_policy"]), ("test", "diag-eval-fixed-test"))
            self.assertEqual((m["split"], m["split_policy"]), ("test", "diag-eval-fixed-test"))
            self.assertEqual((e["scene"]["geometry_gid"], m["geometry_gid"]), (ident["geometry_gid"],) * 2)

    def test_hash_train_scene_is_refused_by_default_and_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, ident = make_scene(tmp, "train")
            out = os.path.join(tmp, "eval")
            r = run_eval(path, out)
            self.assertNotEqual(r.returncode, 0, "哈希 split 是 train 的场景默认必须拒绝")
            self.assertIn("--allow-train-scene", r.stderr)
            self.assertIn(ident["geometry_gid"], r.stderr)
            self.assertEqual(os.listdir(out) if os.path.isdir(out) else [], [])

    def test_hash_val_scene_is_refused_by_default_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = make_scene(tmp, "val")
            r = run_eval(path, os.path.join(tmp, "eval"))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--allow-train-scene", r.stderr)

    def test_allow_train_scene_writes_the_real_split_and_no_test(self):
        for want in ("train", "val"):
            with tempfile.TemporaryDirectory() as tmp:
                path, ident = make_scene(tmp, want)
                out = os.path.join(tmp, "eval")
                r = run_eval(path, out, "--allow-train-scene")
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                e = _load(out, "eval.json")
                m = _load(out, "ep_00000", "episode_manifest.json")
                self.assertEqual((e["scene"]["split"], e["scene"]["split_policy"]), (want, "diag-train-scene"))
                self.assertEqual((m["split"], m["split_policy"]), (want, "diag-train-scene"))
                self.assertTrue(sm.is_diagnostic_scheme(m["split_policy"]))
                seen = [(rel, v) for rel, doc in json_files(out) for v in split_values(doc)]
                self.assertTrue(seen, "产物里应当有 split 字段")
                self.assertNotIn("test", [v for _, v in seen], seen)

    def test_allow_train_scene_on_a_test_scene_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = make_scene(tmp, "test")
            out = os.path.join(tmp, "eval")
            r = run_eval(path, out, "--allow-train-scene")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            e = _load(out, "eval.json")
            self.assertEqual((e["scene"]["split"], e["scene"]["split_policy"]), ("test", "diag-eval-fixed-test"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
