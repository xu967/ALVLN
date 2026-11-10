#!/usr/bin/env python3
"""course_base（长课程：K 个 base 柱阵块首尾相接）回归。

覆盖：第 0 块与 base 同 seed 几何逐项相等、所有块同一 split、跨块缝宽契约、确定性、split_for_scene、
成功线 6.0+5.0·(K−1)、教师 A* 在 K=4 有解、scene_gen / collect_parallel / eval_closed_loop / teacher 的接线，
以及「不给 --blocks 时旧输出逐字节不变」的 golden（哈希在改代码之前、于 main 155e377 上算出）。
"""
import hashlib
import shutil
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect  # noqa: E402
import collect_parallel as cp  # noqa: E402
import eval_closed_loop as ec  # noqa: E402
import scene_gen as sg  # noqa: E402
import split_manifest as sm  # noqa: E402
import teacher  # noqa: E402

try:
    import scene_course as sc
except ImportError:   # 红灯阶段：模块还没写，每条用例各自报错而不是整个文件 ImportError
    sc = None

# tools/eval_seeds.py 的 EVAL_SEEDS_BASE_20（20 个冻结评测 seed，base 几何全是 test）
SEEDS20 = [1, 113, 122, 153, 172, 176, 180, 230, 250, 536, 555, 628, 683, 798, 929, 990, 992, 997, 1113, 1654]
TEMPLATE = sg._read_template(sg.VEHICLES["e1_mini"]["template"])
TMP = tempfile.mkdtemp(prefix="test_scene_course_", dir=os.environ.get("TMPDIR"))

# 改代码前在 main 155e377 上算的 golden：(world sha256 前 16 位, json.dumps(meta, sort_keys=True) sha256 前 16 位)
GOLDEN = {
    "base_e1_v1_1": ("3328b0a73af64c5d", "82c598ac4db5e744"),
    "base_e1_v1_113": ("629fa31b03b47608", "b646d3e455d60bd2"),
    "base_e1_v1_1654": ("ff55ca091ea4be94", "b22cb16b53a07f47"),
    "base_x3_legacy_0": ("670685c3c3a40bd7", "b5ab01b3ce527815"),
    "forest_medium_1": ("055d05ba5b308d73", "0d2317dc3266e154"),
    "l1_random_5": ("6004a5eb22d1460e", "80aab63fd9f052e0"),
}
GOLDEN_CLI = ("3328b0a73af64c5d", "4a65be6b5ebba798")   # scene_gen.py --seed 1 ... --task base 写出的 sdf / json 文件字节


def h16(data):
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()[:16]


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def need_course():
    assert sc is not None, "sim/tools/scene_course.py 还不存在"
    return sc


def course(seed, blocks, appearance=sg.APPEARANCE_DEFAULT):
    return need_course().build_scene(TEMPLATE, seed, seed, blocks, appearance)


def geom(p):
    return (p["x"], p["y"], p["z"], tuple(p["size"]))


def run_scene_gen(*argv):
    return subprocess.run([sys.executable, os.path.join(HERE, "scene_gen.py"), *argv], capture_output=True, text=True)


class CourseGeometry(unittest.TestCase):
    def test_block0_equals_base(self):
        for seed in SEEDS20:
            _w, meta = course(seed, 3)
            base = sg.sample_pillars(seed)[0]
            self.assertEqual([geom(p) for p in meta["pillars"][:len(base)]], [geom(p) for p in base], seed)
            self.assertTrue(all(p["name"].startswith("b0_") for p in meta["pillars"][:len(base)]))
            self.assertEqual(meta["spawn"], sg.make_spawn(base, seed, "default"))
            c = meta["course"]
            self.assertEqual((c["block_seeds"][0], c["block_gids"][0]), (seed, sm.geometry_gid(base)))

    def test_all_blocks_same_base_split_and_shifted_copies(self):
        for seed in SEEDS20:
            for k_blocks in (2, 3, 6):
                _w, meta = course(seed, k_blocks)
                c, pillars = meta["course"], meta["pillars"]
                self.assertEqual(len(c["block_seeds"]), k_blocks)
                splits = set()
                cursor = 0
                for k, bs in enumerate(c["block_seeds"]):
                    self.assertTrue(k == 0 or seed * 1000 + k * 100 <= bs < seed * 1000 + (k + 1) * 100, (seed, k, bs))
                    base = sg.sample_pillars(bs)[0]
                    self.assertEqual(c["block_gids"][k], sm.geometry_gid(base))
                    splits.add(sm.split_for_gid(c["block_gids"][k], sg.GEOMETRY_VERSION))
                    mine = pillars[cursor:cursor + len(base)]
                    cursor += len(base)
                    # 第 k 块 = 该子 seed 的 base 几何整体沿 x 平移 k·5.0（y/z/尺寸不变）
                    self.assertEqual([(round(p["x"] + k * 5.0, 3), p["y"], p["z"], tuple(p["size"])) for p in base],
                                     [geom(p) for p in mine], (seed, k))
                self.assertEqual(cursor, len(pillars))
                self.assertEqual(splits, {"test"}, (seed, k_blocks))   # test 课程的每一块都是 test 几何
                self.assertEqual(len({p["name"] for p in pillars}), len(pillars), "柱名必须唯一")
                self.assertEqual(c["base_geometry_version"], sg.GEOMETRY_VERSION)

    def test_cross_block_gap_contract(self):
        sc_ = need_course()
        for seed in SEEDS20:
            _w, meta = course(seed, 6)
            ps = meta["pillars"]
            worst = min(sg._pillar_gap(a, b) for i, a in enumerate(ps) for b in ps[:i])
            self.assertGreaterEqual(worst, sg.MIN_GAP, seed)
            self.assertGreaterEqual(meta["clearance"]["min_pair_gap_m"], sg.MIN_GAP)
        # 契约判得出坏的：把第 1 块只挪 3.5 m，它的门柱落到上一块最后一根柱（x=5.0）的位置上，必须判不过；挪 5.0 m 才过
        last = [{"name": "p", "x": 5.0, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
        gate = [{"name": "g", "x": 1.5, "y": 0.05, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
        self.assertFalse(sc_.cross_gap_ok(last, sc_.shift_block(gate, 1, pitch=3.5)))
        self.assertTrue(sc_.cross_gap_ok(last, sc_.shift_block(gate, 1)))
        self.assertEqual(sc_.shift_block(gate, 1)[0]["name"], "b1_g")

    def test_deterministic(self):
        for seed in (1, 536, 1654):
            w1, m1 = course(seed, 4)
            w2, m2 = course(seed, 4)
            self.assertEqual(h16(w1), h16(w2))
            self.assertEqual(h16(json.dumps(m1, sort_keys=True)), h16(json.dumps(m2, sort_keys=True)))
            # 子 seed 序列与 K 无关：K=3 课程的前两块就是 K=2 课程
            _w, m3 = course(seed, 3)
            _w, mk2 = course(seed, 2)
            self.assertEqual(m3["pillars"][:mk2["pillar_count"]], mk2["pillars"])
            # CLI 两次写出的 sdf/json 逐字节相同
            outs = []
            for n in (0, 1):
                out = os.path.join(TMP, f"det_{seed}_{n}.sdf")
                r = run_scene_gen("--seed", str(seed), "--out", out, "--template", sg.VEHICLES["e1_mini"]["template"],
                                  "--task", "course_base", "--blocks", "4")
                self.assertEqual(r.returncode, 0, r.stderr)
                outs.append((h16(read_bytes(out)), h16(read_bytes(out[:-4] + ".json"))))
            self.assertEqual(outs[0], outs[1])

    def test_goal_line(self):
        for blocks in range(2, 7):
            _w, meta = course(1, blocks)
            c = meta["course"]
            want = [6.0 + 5.0 * k for k in range(blocks)]
            self.assertEqual(len(c["block_goal_x"]), blocks)
            for got, w in zip(c["block_goal_x"], want):
                self.assertAlmostEqual(got, w, places=9)
            # 评测与采集的成功线都是「最远柱 x + 1.0」：整条课程的成功线 = 6.0 + 5.0·(K−1)
            self.assertAlmostEqual(max(p["x"] for p in meta["pillars"]) + ec.GOAL_MARGIN_M, want[-1], places=9)
            self.assertAlmostEqual(collect.goal_line_x(meta["pillars"]), want[-1], places=9)
            self.assertEqual(c["blocks"], blocks)
            self.assertEqual(c["block_pitch_m"], 5.0)
        self.assertEqual(need_course().GOAL_MARGIN_M, collect.GOAL_MARGIN_M)

    def test_teacher_astar_solvable_k4(self):
        clearance = teacher.CLEARANCE_BY_VEHICLE[("e1_mini", "base")][0]   # E1 教师 A* 膨胀 0.40
        for seed in SEEDS20[:6]:
            _w, meta = course(seed, 4)
            start = (meta["spawn"]["x"], meta["spawn"]["y"])
            goal = (max(p["x"] for p in meta["pillars"]) + 2.0, 0.0)       # teacher._global_path 的 base 目标
            path = teacher.astar_path(meta["pillars"], start, goal, clearance=clearance)
            self.assertIsNotNone(path, f"seed {seed} K=4 教师 A* 无解")
            self.assertAlmostEqual(path[-1][0], goal[0], places=6)
        # SPL 的 l_i（机体半径膨胀）在 K=3 也算得出来
        _w, meta = course(SEEDS20[0], 3)
        li = ec.teacher_shortest_path_m(meta["pillars"], (meta["spawn"]["x"], 0.0), goal_x=16.0,
                                        clearance=ec.spl_clearance("e1_mini", "course_base"))
        self.assertIsNotNone(li)
        self.assertGreater(li, 16.0 - meta["spawn"]["x"] - 1e-6)


class CourseWiring(unittest.TestCase):
    def test_split_for_scene_returns_block0_split(self):
        self.assertIn("course_block0", sm.SPLIT_POLICIES)
        for seed in SEEDS20:
            for blocks in (2, 3, 4, 6):
                _w, meta = course(seed, blocks)
                self.assertEqual(meta["split_policy"], "course_block0")
                self.assertEqual(sm.split_for_scene(meta), "test", (seed, blocks))
        # base 的 split_for_scene 不变
        for seed in (1, 5, 77, 1654):
            _w, bm = sg.build_scene(TEMPLATE, seed, seed, appearance=sg.APPEARANCE_DEFAULT)
            self.assertEqual(sm.split_for_scene(bm), sm.split_for_seed(seed))

    def test_split_for_scene_fails_closed(self):
        _w, meta = course(1, 2)
        no_course = {k: v for k, v in meta.items() if k != "course"}
        with self.assertRaises(ValueError):     # episode manifest 不带 course：复算不了，不能猜
            sm.split_for_scene(no_course)
        bad_ver = json.loads(json.dumps(meta))
        bad_ver["course"]["base_geometry_version"] = "uavrgb.geom.v4"
        with self.assertRaises(ValueError):     # 课程版本与它声称的 base 版本不符
            sm.split_for_scene(bad_ver)
        unknown = dict(meta, geometry_version="uavrgb.course_base.v0")
        with self.assertRaises(ValueError):
            sm.split_for_scene(unknown)
        for policy in (None, "geometry_gid"):   # 课程版本不配 course_block0，不能退回整张柱阵的 gid 算出另一个 split
            with self.assertRaises(ValueError):
                sm.split_for_scene(dict(meta, split_policy=policy))

    def test_split_salt_for_knows_course_version(self):
        need_course()
        self.assertEqual(sm.split_salt_for({"geometry_version": sc.GEOMETRY_VERSION}), sg.GEOMETRY_VERSION)
        self.assertEqual(sc.BASE_GEOMETRY_VERSION, "uavrgb.geom.v5")   # base 版本一变，这条红，提醒课程版本同步 bump
        with self.assertRaises(ValueError):
            sm.split_salt_for({"geometry_version": "uavrgb.course_base.v0"})

    def test_scene_gen_wiring(self):
        need_course()
        for blocks in (2, 5):
            wg, mg = sg.build_scene(TEMPLATE, 7, 7, task="course_base", blocks=blocks, appearance=sg.APPEARANCE_DEFAULT)
            wc, mc = sc.build_scene(TEMPLATE, 7, 7, blocks, sg.APPEARANCE_DEFAULT)
            self.assertEqual((wg, mg), (wc, mc))
        for bad in (dict(task="course_base"), dict(task="base", blocks=3), dict(task="course_base", blocks=1),
                    dict(task="course_base", blocks=7), dict(task="course_base", blocks=3, density="dense"),
                    dict(task="course_base", blocks=3, spawn_mode="random"),
                    dict(task="forest_e1", density="dense", blocks=3)):
            with self.assertRaises(ValueError, msg=str(bad)):
                sg.build_scene(TEMPLATE, 1, 1, **bad)
        tpl = sg.VEHICLES["e1_mini"]["template"]
        out = os.path.join(TMP, "cli.sdf")
        self.assertEqual(run_scene_gen("--seed", "1", "--out", out, "--template", tpl, "--task", "course_base").returncode, 2)
        self.assertEqual(run_scene_gen("--seed", "1", "--out", out, "--template", tpl, "--blocks", "3").returncode, 2)
        self.assertEqual(run_scene_gen("--seed", "1", "--out", out, "--template", tpl, "--task", "course_base",
                                       "--blocks", "9").returncode, 2)
        self.assertEqual(run_scene_gen("--seed", "1", "--out", out, "--template", tpl, "--task", "course_base",
                                       "--blocks", "3", "--spawn-mode", "random").returncode, 2)
        r = run_scene_gen("--seed", "1", "--scene-id", "1", "--out", out, "--template", tpl, "--task", "course_base",
                          "--blocks", "3", "--appearance", "v1")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = json.loads(read_bytes(out[:-4] + ".json"))
        self.assertEqual(meta["course"]["blocks"], 3)
        self.assertEqual(meta["pillar_count"], len(meta["pillars"]))
        self.assertEqual(read_bytes(out).decode("utf-8").count('<model name="b'), meta["pillar_count"])

    def test_collect_parallel_and_eval_wiring(self):
        need_course()
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(blocks=3)), ["--blocks", "3"])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(blocks=None, density=None)), [])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace()), [])
        self.assertEqual(cp.task_geometry_version("course_base"), sc.GEOMETRY_VERSION)
        self.assertEqual(cp.task_geometry_version("base"), sm.SPLIT_SALT)
        _w, meta = course(113, 3)
        self.assertEqual(ec.dedup_key(meta, "course_base"), sm.geometry_gid(meta["pillars"]))   # 与 base 同一条路径
        self.assertIn("course_base", ec.SPL_TASKS)
        # eval 的 --blocks 与 --task course_base 必须同时给（与 forest_e1 的 --density 同一种互斥校验）
        py, script = sys.executable, os.path.join(HERE, "eval_closed_loop.py")
        for argv, msg in ((["--task", "course_base"], "必须同时给"), (["--task", "base", "--blocks", "3"], "必须同时给"),
                          (["--task", "course_base", "--blocks", "7"], "--blocks 必须在")):
            r = subprocess.run([py, script, "--policy", "forward", "--out", os.path.join(TMP, "evalargs"), *argv],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, (argv, r.stderr[-300:]))
            self.assertIn(msg, r.stderr)

    def test_teacher_uses_base_tier(self):
        _w, meta = course(1, 3)
        self.assertEqual(teacher.task_of_scene(meta), "base")
        try:
            teacher.set_vehicle("e1_mini", teacher.task_of_scene(meta))
            self.assertEqual((teacher.ASTAR_CLEARANCE_M, teacher.LOCAL_CLEARANCE_M), (0.40, 0.30))
            self.assertEqual(teacher.teacher_version(), teacher.TEACHER_VERSION)
        finally:
            teacher.set_vehicle("x3")


class TrainingPackingNotWired(unittest.TestCase):
    """训练打包没接课程（scene_course 文件头的 shortcut）：课程数据必须仍被 preprocess 拒收，不能悄悄混进训练集。"""

    def test_preprocess_rejects_course_episodes(self):
        root = os.path.dirname(os.path.dirname(HERE))
        py = sys.executable
        syn = os.path.join(TMP, "syn")
        subprocess.run([py, os.path.join(root, "tools", "make_synthetic_episode.py"), "--out", syn],
                       check=True, capture_output=True)
        ep = os.path.join(syn, "ep1")
        pre = [py, os.path.join(root, "tools", "preprocess.py")]
        ok = subprocess.run(pre + ["--out", os.path.join(TMP, "ds_ok"), ep], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr[-300:])   # 对照：没改的合成集能打包，下面的拒收不是别的原因
        path = os.path.join(ep, "episode_manifest.json")
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
        man.update(geometry_version="uavrgb.course_base.v1", split_policy="course_block0", geometry_gid="check.course")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(man, f)
        r = subprocess.run(pre + ["--out", os.path.join(TMP, "ds_course"), ep], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("geometry_version", r.stderr)
        r2 = subprocess.run(pre + ["--allow-frozen-geometry", "--out", os.path.join(TMP, "ds_course2"), ep],
                            capture_output=True, text=True)
        self.assertNotEqual(r2.returncode, 0)   # 版本门放开后，episode manifest 不带 course，split 复算抛错，仍拒收
        self.assertIn("course", r2.stderr)


class DefaultUnchanged(unittest.TestCase):
    """不给 --blocks 时旧输出逐字节不变（golden 在改代码前算出）。"""

    def test_scene_gen_outputs(self):
        tplx = sg._read_template(sg.DEFAULT_TEMPLATE)

        def pair(world_meta):
            w, m = world_meta
            return h16(w), h16(json.dumps(m, sort_keys=True))
        got = {f"base_e1_v1_{s}": pair(sg.build_scene(TEMPLATE, s, s, appearance=sg.APPEARANCE_DEFAULT)) for s in (1, 113, 1654)}
        got["base_x3_legacy_0"] = pair(sg.build_scene(tplx, 0, 0, appearance=sg.APPEARANCE_LEGACY))
        got["forest_medium_1"] = pair(sg.build_scene(TEMPLATE, 1, 1, task="forest_e1", density="medium",
                                                     appearance=sg.APPEARANCE_DEFAULT))
        got["l1_random_5"] = pair(sg.build_scene(TEMPLATE, 5, 5, spawn_mode="random", task="l1",
                                                 appearance=sg.APPEARANCE_DEFAULT))
        self.assertEqual(got, GOLDEN)
        out = os.path.join(TMP, "base_cli.sdf")
        r = run_scene_gen("--seed", "1", "--scene-id", "1", "--out", out, "--template", sg.VEHICLES["e1_mini"]["template"],
                          "--task", "base", "--appearance", "v1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((h16(read_bytes(out)), h16(read_bytes(out[:-4] + ".json"))), GOLDEN_CLI)

    def test_no_blocks_no_extra_args(self):
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(camera_rate=None, no_aux_cameras=False, density=None, blocks=None)), [])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(density="dense")), ["--density", "dense"])
        self.assertEqual(cp.task_geometry_version("base"), sm.SPLIT_SALT)
        self.assertEqual(ec.SPL_TASKS[:3], ("base", "l1", "forest_e1"))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=1)
