#!/usr/bin/env python3
"""scene_import_mipnerf 测试（不要 GPU、不碰真实场景目录）：RANSAC 地面、PLY 缩放回读、尺度三种取法与低置信标记、
nav_to_world 刚体与原点、假 Mip-NeRF 下载目录上的 align -> ply -> scene -> probe（mock 渲染）端到端；
数据盘上有 garden / bicycle 原始件时，对拍 SCENE-B 写下的尺度与 nav_to_world（收拢一次性脚本后数值不变）。

运行：$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m unittest tools.test_scene_import_mipnerf
"""
import importlib
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
imp = importlib.import_module("scene_import_mipnerf")
scene_mod = importlib.import_module("sim.3dgs.scene")

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

UP = np.array([0.1, -0.7, -0.7]) / np.linalg.norm([0.1, -0.7, -0.7])   # 世界系 up 斜着（Mip-NeRF 360 的 COLMAP 世界系是任意的）
GROUND_D = 2.0            # 地面：UP·p + 2 = 0
TABLE_H = 1.35            # 中心桌面离地（COLMAP 单位）；配 ref_h_m 0.75 -> 0.5556 m/单位，与 garden 同形
CAM_H = 2.5               # 相机离地（单位）
R_ORBIT = 3.4


def _axes():
    ex = np.array([1.0, 0.0, 0.0]) - UP[0] * UP
    ex /= np.linalg.norm(ex)
    return ex, np.cross(UP, ex)


def _write_input_ply(path, xyz):
    """Open3D 导出的 COLMAP 稀疏点格式：double x/y/z/nx/ny/nz + uchar rgb（Inria 包里的 input.ply 就是这个）。"""
    dt = np.dtype([(k, "<f8") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")])
    arr = np.zeros(len(xyz), dt)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr["red"] = 100
    head = ["ply", "format binary_little_endian 1.0", "comment Created by Open3D", f"element vertex {len(xyz)}"]
    head += [f"property double {k}" for k in ("x", "y", "z", "nx", "ny", "nz")] + [f"property uchar {k}" for k in ("red", "green", "blue")]
    with open(path, "wb") as f:
        f.write(("\n".join(head + ["end_header"]) + "\n").encode("ascii"))
        arr.tofile(f)


def make_fake_mipnerf(src_dir, seed=0):
    """假的 Inria 预训练下载目录：cameras.json（环绕、看中心）、input.ply（地面 + 中心桌面 + 外围杂点）、
    point_cloud_30000.ply（nav (2.5,0) 一棵树挡住飞行层 + 半径 7 m 一圈树篱，米制按 0.75/1.35 推回单位）。"""
    rng = np.random.default_rng(seed)
    os.makedirs(src_dir, exist_ok=True)
    ex, ey = _axes()
    g0 = -GROUND_D * UP                                     # 轨道中心正下方的地面点
    s = 0.75 / TABLE_H                                      # 期望尺度 m/单位

    def world(x, y, h):                                     # 水平 (x,y) 单位 + 离地 h 单位 -> 世界
        return g0 + np.outer(x, ex) + np.outer(y, ey) + np.outer(h, UP)

    r, a = 10 * np.sqrt(rng.random(4000)), rng.uniform(0, 2 * np.pi, 4000)
    ground = world(r * np.cos(a), r * np.sin(a), rng.normal(0, 0.003, 4000))
    r, a = 0.6 * np.sqrt(rng.random(600)), rng.uniform(0, 2 * np.pi, 600)
    table = world(r * np.cos(a), r * np.sin(a), TABLE_H + rng.normal(0, 0.003, 600))
    r, a = rng.uniform(6, 9, 300), rng.uniform(0, 2 * np.pi, 300)
    clutter = world(r * np.cos(a), r * np.sin(a), rng.uniform(0, 3, 300))
    _write_input_ply(os.path.join(src_dir, "input.ply"), np.concatenate([ground, table, clutter]))
    cams = []
    for k in range(40):
        az = 2 * np.pi * k / 40
        c = world([R_ORBIT * np.cos(az)], [R_ORBIT * np.sin(az)], [CAM_H + 0.05 * np.sin(3 * az)])[0]
        fwd = world([0.0], [0.0], [1.0])[0] - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, UP)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        cams.append({"id": k, "img_name": f"IMG{k:03d}", "width": 400, "height": 300, "fx": 360.0, "fy": 360.0,
                     "position": c.tolist(), "rotation": np.stack([right, down, fwd], axis=1).tolist()})
    with open(os.path.join(src_dir, "cameras.json"), "w") as f:
        json.dump(cams, f)
    # 高斯（世界系、COLMAP 单位）：树干 + 树冠一根柱子在 nav (2.5,0)，半径 7 m 树篱；高度 0–3.3 m
    pts = []
    for z in np.arange(0.0, 3.3, 0.08) / s:
        for t in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            pts.append(world([2.5 / s + 0.35 / s * np.cos(t)], [0.35 / s * np.sin(t)], [z])[0])
        for t in np.linspace(0, 2 * np.pi, 360, endpoint=False):
            pts.append(world([7.0 / s * np.cos(t)], [7.0 / s * np.sin(t)], [z])[0])
    pts = np.array(pts)
    n = len(pts)
    scene_mod.write_gaussian_ply(os.path.join(src_dir, "point_cloud_30000.ply"), pts, np.full((n, 3), 0.4),
                                 scales=np.full((n, 3), 0.1 / s), opacities=np.full(n, 0.95))
    return s


class TestRansac(unittest.TestCase):
    def test_tilted_plane_with_clutter(self):
        rng = np.random.default_rng(3)
        ex, ey = _axes()
        g = (-GROUND_D * UP + np.outer(rng.uniform(-5, 5, 3000), ex) + np.outer(rng.uniform(-5, 5, 3000), ey)
             + np.outer(rng.normal(0, 0.005, 3000), UP))
        junk = rng.uniform(-5, 5, (1500, 3))
        n, d, n_inl = imp.fit_plane_ransac(np.concatenate([g, junk]), thresh=0.02, normal_hint=UP + [0.1, 0.0, 0.0])
        self.assertLess(math.degrees(math.acos(min(1.0, float(n @ UP)))), 0.5)   # 朝 hint 一侧，偏差 < 0.5°
        self.assertAlmostEqual(d, GROUND_D, delta=0.01)
        self.assertGreaterEqual(n_inl, 2900)


class TestScalePly(unittest.TestCase):
    def test_roundtrip_and_header_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "a.ply"), os.path.join(tmp, "b.ply")
            rng = np.random.default_rng(0)
            means = rng.normal(0, 1, (50, 3)).astype(np.float32)
            scene_mod.write_gaussian_ply(src, means, rng.random((50, 3)), scales=rng.uniform(0.01, 0.2, (50, 3)),
                                         opacities=rng.uniform(0.1, 0.9, 50))
            imp.scale_ply(src, dst, 0.5)
            a, b = scene_mod.read_gaussian_ply(src), scene_mod.read_gaussian_ply(dst)
            self.assertTrue(np.allclose(b["means"], 0.5 * a["means"], atol=1e-6))
            self.assertTrue(np.allclose(b["scales"], 0.5 * a["scales"], rtol=1e-5))
            self.assertTrue(np.array_equal(b["sh"], a["sh"]) and np.allclose(b["opacities"], a["opacities"]))
            self.assertEqual(imp.ply_scale_of(dst), 0.5)            # 缩放倍数写进头注释，尺度改了 ply 步能认出来重做
            self.assertIsNone(imp.ply_scale_of(src))

    def test_rejects_bad_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "a.ply")
            scene_mod.write_gaussian_ply(src, np.zeros((2, 3), np.float32), np.zeros((2, 3)))
            for bad in (0.0, -1.0, float("nan")):
                with self.assertRaises(ValueError):
                    imp.scale_ply(src, os.path.join(tmp, "b.ply"), bad)


class TestAlign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.s_true = make_fake_mipnerf(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_object_height(self):
        al = imp.align_scene(self.src, dict(method="object_height", ref="桌面", ref_h_m=0.75, ref_h_guess=1.35,
                                            center_r=1.5, band=0.08, confidence="medium", source="test"))
        self.assertLess(math.degrees(math.acos(min(1.0, float(np.dot(al["up_world"], UP))))), 0.5)
        self.assertAlmostEqual(al["scale_m_per_unit"], self.s_true, delta=0.003)
        m = np.asarray(al["nav_to_world"])
        r = m[:3, :3]
        self.assertTrue(np.allclose(r @ r.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(r) - 1) < 1e-9)
        self.assertTrue(np.allclose(r[:, 2], al["up_world"]))               # nav z = 地面法线
        # nav 原点 = 轨道中心投到（缩放后）地面；相机 nav z = 离地高度（米）
        cz = np.array([c["pos_nav"][2] for c in al["cams_nav"]])
        self.assertAlmostEqual(float(np.median(cz)), CAM_H * self.s_true, delta=0.02)
        cxy = np.array([c["pos_nav"][:2] for c in al["cams_nav"]])
        self.assertLess(float(np.linalg.norm(cxy.mean(axis=0))), 0.02)
        self.assertEqual((al["scale_confidence"], al["scale_unverified"]), ("medium", False))
        self.assertLess(max(al["up_angles_deg"].values()), 1.0)

    def test_camera_height_is_low_confidence(self):
        al = imp.align_scene(self.src, dict(method="camera_height", cam_h_m=1.5, confidence="low", source="手持假设"))
        self.assertAlmostEqual(al["scale_m_per_unit"], 1.5 / CAM_H, delta=0.003)
        self.assertEqual((al["scale_confidence"], al["scale_unverified"]), ("low", True))
        self.assertIn("1.5", al["scale_basis"])

    def test_fixed_and_bad_method(self):
        al = imp.align_scene(self.src, dict(method="fixed", m_per_unit=0.6, confidence="medium", source="用户给的树桩直径"))
        self.assertEqual(al["scale_m_per_unit"], 0.6)
        with self.assertRaises(ValueError):
            imp.align_scene(self.src, dict(method="guess"))
        with self.assertRaises(ValueError):                  # 猜测高附近没有稀疏点 = 参照物没找到，不能静默给个尺度
            imp.align_scene(self.src, dict(method="object_height", ref="x", ref_h_m=0.75, ref_h_guess=5.0,
                                           center_r=1.5, band=0.08, confidence="medium", source="t"))


class TestEndToEndMock(unittest.TestCase):
    """假下载目录 -> align / ply / scene / probe（mock 渲染器），产物齐、路线绕树、低置信标记进 scene.json。"""

    def test_pipeline(self):
        renderer_mod = importlib.import_module("sim.3dgs.renderer")
        pq = importlib.import_module("scene_probe_quality")
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            make_fake_mipnerf(src)
            cfg = dict(scale=dict(method="camera_height", cam_h_m=1.39, confidence="low", source="手持假设（测试）"),
                       routes=[{"name": "r0_around_tree", "spawn": {"x": -1.5, "y": 0.0, "yaw_deg": 0.0},
                                "waypoint": {"x": 5.0, "y": 0.0}, "note": "直线穿树，A* 须绕"}])
            saved = {k: getattr(imp, k) for k in ("SCENE", "CFG", "SRC_DIR", "SCENE_DIR", "SCENE_JSON", "OUT")}
            imp.SCENE, imp.CFG, imp.SRC_DIR = "fake", cfg, src
            imp.SCENE_DIR = os.path.join(tmp, "scenes", "fake")
            imp.SCENE_JSON = os.path.join(imp.SCENE_DIR, "scene.json")
            imp.OUT = os.path.join(tmp, "out")
            real = renderer_mod.make_renderer
            renderer_mod.make_renderer = lambda scene, mock=False, *a, **k: real(scene, True, *a, **k)
            try:
                imp.step_align()
                imp.step_ply()
                imp.step_scene()
                with self.assertRaises(SystemExit):        # mock 渐变帧 lapvar≈0.2 全判低质：过不了门槛要停，但 JSON 先落盘
                    imp.step_probe()
            finally:
                renderer_mod.make_renderer = real
                for k, v in saved.items():
                    setattr(imp, k, v)
            sdir = os.path.join(tmp, "scenes", "fake")
            sc = scene_mod.load_scene(os.path.join(sdir, "scene.json"))   # ply 在、nav_to_world 有限 4x4
            al = imp.read_json(os.path.join(sdir, "align.json"))
            s = al["scale_m_per_unit"]
            self.assertAlmostEqual(s, 1.39 / CAM_H, delta=0.003)
            self.assertEqual(imp.ply_scale_of(sc["ply_path"]), round(s, 6))
            self.assertEqual((sc["scale_confidence"], sc["scale_unverified"]), ("low", True))
            self.assertIn("手持假设", sc["scale_source"])
            self.assertTrue(any("米制结论" in n for n in sc["notes"]))         # 低置信尺度：文档口径写进 notes
            r0 = sc["routes"][0]
            self.assertEqual((sc["spawn"], sc["waypoint"], sc["max_steps"]), (r0["spawn"], r0["waypoint"], r0["max_steps"]))
            self.assertEqual(r0["spawn"]["z"], imp.FLY_Z)
            chk = imp.read_json(os.path.join(tmp, "out", "routes_check.json"))[0]
            self.assertTrue(chk["feasible"] and chk["clear_ok"])
            self.assertGreater(chk["turn_deg"], 20.0)                         # 树挡在直线上：A* 绕行
            self.assertEqual(r0["max_steps"], chk["steps_estimate"])
            self.assertTrue(os.path.isfile(os.path.join(sdir, "probe_quality", "map_routes.png")))
            rep = imp.read_json(os.path.join(sdir, "probe_quality", "routes_quality.json"))
            self.assertEqual(rep["lapvar_min_threshold"], pq.FRAME_LAPVAR_MIN)
            q0 = rep["routes"][0]
            self.assertEqual((q0["pass"], q0["n_low"]), (False, q0["n_frames"]))
            hc = rep["height_check"]                                          # 俯视深度复核：mock 深度常量 5 m
            self.assertEqual({round(h["z_nav"] - h["nadir_depth_m"], 3) for h in hc}, {round(z - 5.0, 3) for z in imp.HEIGHT_CHECK_Z})


@unittest.skipUnless(all(os.path.isfile(os.path.join(imp.DOWNLOADS, n, f)) for n in ("garden", "bicycle")
                         for f in ("cameras.json", "input.ply"))
                     and all(os.path.isfile(os.path.join(imp.SCENES_ROOT, n, "align.json")) for n in ("garden", "bicycle")),
                     "数据盘没有 garden/bicycle 原始件或 SCENE-B 的 align.json")
class TestSceneBParity(unittest.TestCase):
    """收拢 SCENE-B 一次性脚本（probe_align / build_scene）后，garden/bicycle 的尺度与 nav_to_world 必须和已发布的一致。"""

    def test_garden_bicycle(self):
        for name in ("garden", "bicycle"):
            old = imp.read_json(os.path.join(imp.SCENES_ROOT, name, "align.json"))
            sc = imp.read_json(os.path.join(imp.SCENES_ROOT, name, "scene.json"))
            al = imp.align_scene(os.path.join(imp.DOWNLOADS, name), imp.SCENES[name]["scale"])
            self.assertAlmostEqual(al["scale_m_per_unit"], old["scale_m_per_unit"], places=9, msg=name)
            self.assertTrue(np.allclose(np.round(al["nav_to_world"], 6), sc["nav_to_world"], atol=1e-9), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
