#!/usr/bin/env python3
"""3dgs-rebaseline 入口（build_config.py / summarize.py / run.sh，任务卡 S1b & 审核 E）的单测与桩演练。

覆盖：
  1. 静态核对：run.sh 头部包含任务卡「起跑前写下的预期」预注册原文、LANES=2、DROP_TRACE=false、排 gs3d.lock、S1a 的 SPLIT_FROM_IDENTITY 预留说明、`bash -n` 语法无误；
  2. build_config：6 个套件 × 6 个臂参数、drop_trace 恒为 false（传 true 报错）、collision_model=v1/v2、S1a 的 split_from_identity 缺省不写而传 true 时写入；
  3. summarize：
     - 用 sim.3dgs.route_dirs 排除 `r<N>.incomplete-<时刻>` 半成品目录（E-12），同时在 summary.json / summary.md 第 4 节列出半成品计数（E-1）；
     - 分母固定为请求集数（未跑完时不缩小分母）；
     - v2 对 v1 配对同时拆分「同出生位姿」与「异出生位姿」；
     - 预注册判读：规则臂 |变化| <= 3 与规则 v2 下降 > 6/120 的触发分支；
     - v1 与原实验逐集对拍一致返回 0，出现差异返回 rc=6；
  4. run.sh 全链路桩演练（DRY_RUN=1）：
     - 正常跑完落 DONE rc=0，生成 code 冻结副本、带启动时刻的 run.pid（E-4）、追加式 inputs.<时刻>.txt（E-2）；
     - 续跑跳过已完成路线，保留首次 inputs.first.sha256 且 inputs.history.log 追加第二条记录，人为放入的半成品路线改名为 `*.incomplete-*` 并进入汇总第 4 节（E-1 / E-2 / E-9）；
     - 正式模式拒绝覆盖配置的环境变量，且不覆盖已有 DONE（E-5 / E-10）；
     - 重复实例被实例锁拦截退出码 3，不改写正在运行的 DONE / run.pid（E-5）；
     - 驱动任务失败时 DONE 记录 rc=7（不因 summarize 成功而掩盖为 rc=0，E-6）。
"""
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_DATA = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
if "TMPDIR" not in os.environ and os.path.isdir(os.path.join(_DATA, "tmp")):
    _TMP = os.path.join(_DATA, "tmp", "phase12-cards", "s1b", "tmp")
    os.makedirs(_TMP, exist_ok=True)
    tempfile.tempdir = _TMP
    os.environ["TMPDIR"] = _TMP


def _load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load_mod("rebaseline_build_config", "build_config.py")
sm = _load_mod("rebaseline_summarize", "summarize.py")
RUN_SH = os.path.join(HERE, "run.sh")
STUB_DRV = os.path.join(HERE, "dry_run_driver.py")


def _make_fake_inputs(tmp):
    """构造最小独立测试夹具（路线 JSON、selected.json、scene.json），不依赖数据盘真实大文件也能测。"""
    scenes_dir = os.path.join(tmp, "scenes")
    for s in ("s_in", "s_out"):
        os.makedirs(os.path.join(scenes_dir, s), exist_ok=True)
        with open(os.path.join(scenes_dir, s, "scene.json"), "w", encoding="utf-8") as f:
            json.dump({"name": s, "routes": [{"name": "r0"}, {"name": "r1"}]}, f)
    f40 = os.path.join(tmp, "routes_formal40.json")
    hold = os.path.join(tmp, "routes_holdout.json")
    with open(f40, "w", encoding="utf-8") as f:
        json.dump({"scenes": {"s_in": [{"route": 0}, {"route": 1}]}}, f)
    with open(hold, "w", encoding="utf-8") as f:
        json.dump({"scenes": {"s_in": [{"route": 0}]}}, f)
    sel = os.path.join(tmp, "selected.json")
    with open(sel, "w", encoding="utf-8") as f:
        json.dump({
            "variant": "v2f",
            "levels": [[0.15, 1.2, False], [0.15, 0.8, False], [0.12, 0.55, True]],
            "order": "near",
            "depth_variant": "v2d",
            "depth_levels": [[0.25, 1.2, False], [0.25, 0.8, False], [0.2, 0.55, True]],
            "depth_order": "near",
        }, f)
    return {
        "REPO": REPO,
        "SCENES_DIR": scenes_dir,
        "CF_SITL_PROC": "/bin/true",
        "HOLD": "1.0",
        "STEP_FRAC": "0.5",
        "STEP_PAD": "15",
        "GUARD_DEG": "45.0",
        "TEACHER_TOL_DEG": "18.907607239317166",
        "YAW_STEP": "31.7",
        "V1_R_SAFE": "0.25",
        "V1_D_LOOK": "1.2",
        "K3_SELECTED": sel,
        "ROUTES_FORMAL40": f40,
        "ROUTES_HOLDOUT": hold,
        "INDOOR_SCENES": "s_in",
        "OUTDOOR_SCENES": "s_out",
        "ROUTES_OUTDOOR": "0",
        "K_INDOOR40": "2",
        "K_HOLDOUT": "2",
        "K_OUTDOOR": "2",
        "SEED0_INDOOR40": "8000",
        "SEED0_HOLDOUT": "5000",
        "SEED0_OUTDOOR": "7000",
        "JITTER_INDOOR": "0.2",
        "JITTER_OUTDOOR": "0.5",
        "LOOKAHEAD_INDOOR": "1.5",
        "LOOKAHEAD_OUTDOOR": "3.0",
        "STEP_CAP_INDOOR": "100",
    }


class TestRunShStatic(unittest.TestCase):
    def setUp(self):
        with open(RUN_SH, encoding="utf-8") as f:
            self.src = f.read()

    def test_bash_syntax_ok(self):
        r = subprocess.run(["bash", "-n", RUN_SH], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_prereg_verbatim_and_lanes_and_s1a_hook(self):
        # 任务卡「起跑前写下的预期」原文关键句逐字核对
        self.assertIn("起跑前写下的预期：规则臂在新口径下的到点成功数变化不超过 3/120（主会话已按精确欧氏距离重判过 K3 的已录轨迹，0 集翻转）；教师的变化方向不预测，照实报。", self.src)
        self.assertIn("规则 v2 的到点成功下降超过 6/120，就说明它的窄通道（半宽 0.15 m）依赖了旧口径的宽松，E4 的教师要改用半宽 0.25 m 的变体。", self.src)
        self.assertIn("LANES=${LANES:-2}", self.src)
        self.assertIn("DROP_TRACE=${DROP_TRACE:-false}", self.src)
        self.assertIn("gs3d.lock", self.src)
        self.assertIn("SPLIT_FROM_IDENTITY=${SPLIT_FROM_IDENTITY:-true}", self.src)


class TestBuildConfig(unittest.TestCase):
    def test_all_six_suites_and_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_fake_inputs(tmp)
            for sname in bc.SUITES:
                cm = sname.rsplit("_", 1)[1]
                cfg = bc.build(sname, env)
                self.assertIs(cfg["drop_trace"], False)
                self.assertEqual(cfg["collision_model"], cm)
                base_arms = ["teacher-tol", "teacher-swept", "forward+route", "rule-occ+route", "rule2-occ+route", "rule2-depth+route"]
                want_arms = base_arms + (["rule2-depth-op030+route"] if cm == "v2" else [])
                self.assertEqual([a["name"] for a in cfg["arms"]], want_arms)
                self.assertIs(cfg["split_from_identity"], True)
                swept = [a for a in cfg["arms"] if a["name"] == "teacher-swept"][0]
                self.assertEqual(swept["teacher_clearance"], "swept")
                self.assertNotIn("spawn_two_layer", swept)
                if cm == "v2":
                    diag = [a for a in cfg["arms"] if a["name"] == "rule2-depth-op030+route"][0]
                    self.assertEqual((diag["base"], diag["rule_version"], diag["depth_opacity_min"]), ("rule-depth", "v2", 0.30))

    def test_drop_trace_true_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_fake_inputs(tmp)
            env["DROP_TRACE"] = "true"
            with self.assertRaises(ValueError):
                bc.build("indoor40_v1", env)

    def test_s1a_split_from_identity_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = _make_fake_inputs(tmp)
            env["SPLIT_FROM_IDENTITY"] = "false"
            cfg = bc.build("indoor40_v2", env)
            self.assertIs(cfg["split_from_identity"], False)
            env["SPLIT_FROM_IDENTITY"] = "none"
            self.assertNotIn("split_from_identity", bc.build("indoor40_v2", env))
            env["SPLIT_FROM_IDENTITY"] = "invalid"
            with self.assertRaises(ValueError):
                bc.build("indoor40_v2", env)


class TestSummarize(unittest.TestCase):
    def _write_ep(self, suite_dir, arm, scene, r, k, seed, cm, succ, cf=False, pose_x=1.0,
                  with_trace=True, r_suffix="", step_zs=None, spawn_z=1.2, z_slice=(1.0, 1.4)):
        rdir = os.path.join(suite_dir, arm, scene, f"r{r}{r_suffix}")
        ep_dir = os.path.join(rdir, f"ep_{k:05d}")
        os.makedirs(ep_dir, exist_ok=True)
        mv = "3dgs.pilot.v2+stop.v1" + ("+coll.v2" if cm == "v2" else "")
        zs = step_zs or [spawn_z]
        m = {
            "success_stop": succ, "success_v2": succ, "collision": cf, "collide_first": cf,
            "timeout": (not succ and not cf), "steps": len(zs),
            "reach_step": 0 if succ else None, "collision_step": 0 if cf else None,
            "spl": 0.7 if succ else 0.0, "spl_stop": 0.7 if succ else 0.0,
            "start_pose": {"x": pose_x, "y": 2.0, "yaw_deg": 0.0},
            "arm": arm, "scene": scene, "route": r, "episode": k, "seed": seed,
            "collision_model": cm, "metric_version": mv,
        }
        with open(os.path.join(ep_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f)
        with open(os.path.join(ep_dir, "scene.json"), "w", encoding="utf-8") as f:
            json.dump({"spawn": {"x": 0.0, "y": 0.0, "z": spawn_z},
                       "collision": {"z_slice": list(z_slice), "voxel_m": 0.1, "envelope": "e1_mini"}}, f)
        with open(os.path.join(ep_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
            for idx, z in enumerate(zs):
                f.write(json.dumps({"step": idx, "pose_before": {"x": pose_x, "y": 2.0, "z": z},
                                    "pose_gt": {"x": pose_x + 0.1, "y": 2.0, "z": z}}) + "\n")
        if with_trace:
            with open(os.path.join(ep_dir, "trace.jsonl"), "w", encoding="utf-8") as f:
                for idx, z in enumerate(zs):
                    f.write(json.dumps({"step": idx, "pos": [pose_x, 2.0, z]}) + "\n")
        return rdir

    def _write_done(self, rdir, arm, scene, r, n_eps, cm):
        mv = "3dgs.pilot.v2+stop.v1" + ("+coll.v2" if cm == "v2" else "")
        with open(os.path.join(rdir, "done.json"), "w", encoding="utf-8") as f:
            json.dump({
                "arm": arm, "scene": scene, "route": r, "episodes": [{"episode": i} for i in range(n_eps)],
                "failures": [], "collision_model": cm, "metric_version": mv,
            }, f)

    def test_excludes_incomplete_dirs_but_reports_them_and_splits_spawn_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cm in ("v1", "v2"):
                sdir = os.path.join(tmp, f"indoor40_{cm}")
                os.makedirs(sdir, exist_ok=True)
                with open(os.path.join(sdir, "config.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "collision_model": cm, "k": 2, "scenes": ["s1"], "routes": {"s1": [0]},
                        "arms": [{"name": "rule2-occ+route", "base": "rule-occ", "guard": "route"}],
                    }, f)
                # 正式完整路线 r0：ep0 同起点，ep1 在 v2 下异起点
                self._write_ep(sdir, "rule2-occ+route", "s1", 0, 0, 8000, cm, succ=True, pose_x=1.0)
                rdir = self._write_ep(
                    sdir, "rule2-occ+route", "s1", 0, 1, 8001, cm,
                    succ=(cm == "v1"), cf=(cm == "v2"), pose_x=(1.0 if cm == "v1" else 1.2),
                )
                self._write_done(rdir, "rule2-occ+route", "s1", 0, 2, cm)
                # 放入半成品 r0.incomplete-010203（含 1 集）：不能混进正式计数，但必须进 incompletes
                if cm == "v2":
                    self._write_ep(sdir, "rule2-occ+route", "s1", 0, 0, 8000, cm, succ=False, cf=True, r_suffix=".incomplete-010203")

            rc = sm.main([tmp])
            self.assertEqual(rc, 0)
            with open(os.path.join(tmp, "summary.json"), encoding="utf-8") as f:
                res = json.load(f)
            # 正式集数严格为 2（半成品未被算入）
            self.assertEqual(res["suites"]["indoor40_v2"]["n_episodes"], 2)
            self.assertEqual(res["suites"]["indoor40_v2"]["arms"]["rule2-occ+route"]["k_stop"], 1)
            # 半成品被单独记录
            inc = res["suites"]["indoor40_v2"]["incompletes"]
            self.assertEqual(len(inc), 1)
            self.assertEqual((inc[0]["dir_name"], inc[0]["episodes_done"], inc[0]["k_collide_first"]), ("r0.incomplete-010203", 1, 1))
            # 同起点 / 异起点配对拆分与第 8 条新增的两列（v2 判碰而 v1 未碰集数；规划与判定层不一致步数）
            p = res["pairs_v2_vs_v1"]["indoor40"]["rule2-occ+route"]
            self.assertEqual((p["all"]["pairs"], p["same_spawn"]["pairs"], p["diff_spawn"]["pairs"]), (2, 1, 1))
            self.assertEqual(p["diff_spawn"]["c_v1_only"], 1)
            self.assertEqual(p["all"]["coll_v2_not_v1"], 1, "ep1 在 v2 下碰撞而 v1 下未碰")
            self.assertEqual(res["suites"]["indoor40_v2"]["arms"]["rule2-occ+route"]["layer_outside_env_steps"], 0)

    def test_layer_outside_envelope_steps_detected_when_step_exceeds_envelope(self):
        """第 8 条：正常包络内飞行（如起飞过冲 +0.070 m）规划与判定层不一致步数为 0；人为造出超出包络的高度（如 z=1.35 m 进入包络外的层）时检出非零步数。"""
        with tempfile.TemporaryDirectory() as tmp:
            sdir = os.path.join(tmp, "indoor40_v2")
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"collision_model": "v2", "k": 2, "scenes": ["s1"], "routes": {"s1": [0]},
                           "arms": [{"name": "teacher-swept", "base": "teacher", "guard": None}]}, f)
            # ep0：z=1.270（起飞过冲 +0.070 m）仍在包络 [1.137, 1.303] 即层 1..3 内 -> 0 步越界
            self._write_ep(sdir, "teacher-swept", "s1", 0, 0, 8000, "v2", succ=True,
                           step_zs=[1.270, 1.190], spawn_z=1.2, z_slice=(1.0, 1.5))
            # ep1：第 1 步飙到 z=1.380（圆柱 [1.352, 1.408] 入层 4，超出 1.2 m 的包络层 1..3） -> 1 步越界
            rdir = self._write_ep(sdir, "teacher-swept", "s1", 0, 1, 8001, "v2", succ=True,
                                  step_zs=[1.200, 1.380], spawn_z=1.2, z_slice=(1.0, 1.5))
            self._write_done(rdir, "teacher-swept", "s1", 0, 2, "v2")
            rc = sm.main([tmp])
            self.assertEqual(rc, 0)
            with open(os.path.join(tmp, "summary.json"), encoding="utf-8") as f:
                res = json.load(f)
            self.assertEqual(res["suites"]["indoor40_v2"]["arms"]["teacher-swept"]["layer_outside_env_steps"], 1)

    def test_rescore_offline_episode_distinguishes_square_and_exact_and_trace(self):
        ro_off = _load_mod("rebaseline_rescore_offline", "rescore_offline.py")
        col = importlib.import_module("sim.3dgs.collision")
        # 放一个体素方块 [0.5, 0.6] × [0.3, 0.4] × [1.2, 1.3]：
        # 终点 (0.69, 0.49, 1.25) 在斜向邻格 (1,1) 内，距方块边角 hypot(0.09, 0.09)=0.127 m > 0.102 m：
        #   - v1（十字核）：不碰
        #   - r1_v2（3×3 方核）：误判碰
        #   - exact_band053 与 final_v2（精确距离）：不碰
        pts = np.array([[0.55, 0.35, 1.25]])
        kw = dict(z_range=(1.0, 1.4), voxel_m=0.1, body_radius_m=0.102)
        pack = {
            "v1": col.Occupancy(pts, body_half_height_m=0.028, collision_model="v1", **kw),
            "b053": col.Occupancy(pts, body_half_height_m=0.053, collision_model="v2", **kw),
            "v2": col.Occupancy(pts, body_half_height_m=0.028, collision_model="v2", **kw),
        }
        i0, j0 = pack["v2"]._cell(pts)[0][:2]
        corner_x, corner_y = pack["v2"].origin[0] + (i0 + 1) * 0.1, pack["v2"].origin[1] + (j0 + 1) * 0.1
        qx, qy = corner_x + 0.085, corner_y + 0.085   # 落在斜向邻格 (1,1) 内，距方块边角 hypot(0.085, 0.085)=0.1202 m > 0.102 m
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"success_stop": True, "collision": False, "reach_step": 0}, f)
            with open(os.path.join(tmp, "scene.json"), "w", encoding="utf-8") as f:
                json.dump({"spawn": {"x": 0.0, "y": 0.0, "z": 1.2}}, f)
            with open(os.path.join(tmp, "actions.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"step": 0, "pose_before": {"x": qx + 0.5, "y": qy, "z": 1.25},
                                    "pose_gt": {"x": qx, "y": qy, "z": 1.25}}) + "\n")
            res = ro_off.rescore_episode(tmp, pack, use_trace=False)
            self.assertEqual(res["v1"], (True, False))
            self.assertEqual(res["r1_v2"], (False, True), "第一轮 3×3 方核斜向误判碰")
            self.assertEqual(res["exact_band053"], (True, False), "精确距离不误判")
            self.assertEqual(res["final_v2"], (True, False), "定稿 v2 不误判")
            self.assertEqual(res["outside_env_steps"], 0)

    def test_prereg_drop_over_six_triggers_e4_warning_and_v1_mismatch_returns_6(self):
        pairs = {
            "indoor40": {
                "rule2-occ+route": {
                    "all": {"pairs": 120, "k_v1": 105, "k_v2": 97, "diff_v2_minus_v1": -8}
                }
            }
        }
        pr = sm.evaluate_prereg(pairs, complete_all=True)
        self.assertTrue(pr["rule_v2_drop_gt_6_indoor40"]["rule2-occ+route"]["drop_gt_6"])
        self.assertIn("0.25 m", pr["e4_teacher_note"])

        with tempfile.TemporaryDirectory() as tmp:
            sdir = os.path.join(tmp, "out", "indoor40_v1")
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"collision_model": "v1", "k": 1, "scenes": ["s1"], "routes": {"s1": [0]},
                           "arms": [{"name": "teacher-tol", "base": "teacher", "guard": None}]}, f)
            rdir = self._write_ep(sdir, "teacher-tol", "s1", 0, 0, 8000, "v1", succ=True)
            self._write_done(rdir, "teacher-tol", "s1", 0, 1, "v1")
            # 造一个结果相反的假 W1-11 原实验目录
            w111 = os.path.join(tmp, "w111")
            self._write_ep(os.path.join(w111, "indoor"), "teacher-tol", "s1", 0, 0, 8000, "v1", succ=False, cf=True)
            rc = sm.main([os.path.join(tmp, "out"), "--w111", w111])
            self.assertEqual(rc, 6)


class TestRunShDryRunChain(unittest.TestCase):
    @staticmethod
    def _read_text(path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def _run_dry(self, out_root, extra_env):
        env = os.environ.copy()
        env.update(extra_env)
        return subprocess.run(
            ["setsid", "-w", "bash", RUN_SH],
            env=env, capture_output=True, text=True, timeout=120,
        )

    def test_full_dry_run_resume_incomplete_refusal_and_failure_rc(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_env = _make_fake_inputs(tmp)
            out = os.path.join(tmp, "exp_out")
            dry_env = {
                **base_env,
                "DRY_RUN": "1",
                "FREEZE": "tree",
                "DRIVER": STUB_DRV,
                "OUT_ROOT": out,
                "PY": sys.executable,
                "SUITES": "indoor40_v1 indoor40_v2",
                "K_INDOOR40": "1",
                "ROUTES_LIMIT": "1",
                "WATCH_S": "0.1",
                "STUB_SLEEP_S": "0.01",
            }
            # 1) 首次桩演练全流程通过
            r1 = self._run_dry(out, dry_env)
            self.assertEqual(r1.returncode, 0, self._read_text(os.path.join(out, "run.log")) if os.path.exists(os.path.join(out, "run.log")) else r1.stderr)
            done_txt = self._read_text(os.path.join(out, "DONE"))
            self.assertTrue(done_txt.startswith("rc=0 "), done_txt)
            pid_txt = self._read_text(os.path.join(out, "run.pid"))
            self.assertIn("proc_start_ticks=", pid_txt)
            self.assertIn("pid1_start_ticks=", pid_txt)
            self.assertTrue(os.path.isdir(os.path.join(out, "code")))

            # 2) 模拟中断半成品（删掉一条路线的 done.json，留下 ep_00000），原参数续跑：半成品改名 *.incomplete-* 并被汇总记录，inputs.history.log 追加第 2 段
            broken_r0 = os.path.join(out, "indoor40_v2", "teacher-tol", "s_in", "r0")
            os.remove(os.path.join(broken_r0, "done.json"))
            r2 = self._run_dry(out, dry_env)
            self.assertEqual(r2.returncode, 0)
            hist = self._read_text(os.path.join(out, "inputs.history.log"))
            self.assertEqual(hist.count("started_at="), 2)
            with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
                summ = json.load(f)
            self.assertEqual(len(summ["suites"]["indoor40_v2"]["incompletes"]), 1)

            # 3) E-5 / E-10：不带 DRY_RUN / SMOKE 的正式运行带着覆盖变量启动时被拒（rc=2），且不覆盖已有的 rc=0 DONE
            formal_bad = dict(base_env)
            formal_bad["OUT_ROOT"] = out
            formal_bad["PY"] = sys.executable
            r3 = subprocess.run(["setsid", "-w", "bash", RUN_SH], env={**os.environ, **formal_bad}, capture_output=True, text=True, timeout=30)
            self.assertIn(r3.returncode, (2, 4))  # 有卡机器在覆盖检查报 2，无卡报 4，均不覆盖已有 DONE
            self.assertTrue(self._read_text(os.path.join(out, "DONE")).startswith("rc=0 "))
            self.assertTrue(os.path.isfile(os.path.join(out, "refused.txt")))

            # 4) E-5：实例锁被占时重复启动立即退出 rc=3，不碰 DONE
            r4 = subprocess.run(
                ["flock", os.path.join(out, ".lock"), "bash", "-c", f"setsid -w bash {RUN_SH}"],
                env={**os.environ, **dry_env}, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(r4.returncode, 3)
            self.assertTrue(self._read_text(os.path.join(out, "DONE")).startswith("rc=0 "))

            # 5) E-6：驱动任务失败时 DONE 记 rc=7（不能被 summarize.py 的 rc=0 掩盖）
            out_fail = os.path.join(tmp, "exp_fail")
            fail_env = {**dry_env, "OUT_ROOT": out_fail, "STUB_FAIL_SUITE": "indoor40_v2", "STUB_FAIL_SCENE": "s_in"}
            r5 = self._run_dry(out_fail, fail_env)
            self.assertEqual(r5.returncode, 7)
            self.assertTrue(self._read_text(os.path.join(out_fail, "DONE")).startswith("rc=7 "))


if __name__ == "__main__":
    unittest.main()
