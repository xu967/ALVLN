#!/usr/bin/env python3
"""preprocess.py --depth-source analytic 回归（任务卡 W1-8；纯 CPU，临时目录，不读真实数据）。

覆盖：
  - 卡里指定的三条几何断言：正前方一根 0.4 m 方柱的中间列与最左最右列落在哪个分带；相机高 2.0 m 时最下一行的分带 = 地面深度对应的分带；
  - 与一份独立实现的三维求交（射线对轴对齐方盒 / 竖直圆柱 / 地面，写在本文件里，不调 preprocess 与 rule_policy）逐格相等；
  - 坐标约定（方位角左正、列从左到右、yaw 度逆时针）；柱顶以上的射线不算打到；
  - 命令行：meta 记 depth_source、侧车分带 = 函数输出（有 / 无包络膨胀）、忽略深度文件、不支持的集（waypoint、3DGS、非 e1_mini、缺 scene.json / 位姿）报错；
  - `--depth-source file`（默认）的输出与改动前逐字节相同：标签区、侧车 payload、规范化后的 meta 的 sha256 取自 3e03e7e 的 preprocess.py 在同一夹具上的输出；
    meta 里没有 depth_source 键。
"""
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "sim" / "tools"))
import preprocess as pre  # noqa: E402

PY = sys.executable
# 命令行用例跑的 preprocess.py；PREPROCESS_PY 可指到旧版副本，核对 GOLDEN 取自改动前的输出。
PRE_PY = Path(os.environ.get("PREPROCESS_PY") or REPO / "tools" / "preprocess.py")
SRC_W, SRC_H = 320, 240
GRID = (15, 20)
RAYS = 3                                            # 卡：每格 3×3 条射线
F = (SRC_W / 2.0) / math.tan(math.radians(30.0))   # 60° 水平视场的焦距，277.128 px（manifest 没写 fx 时的缺省）
MOUNT = (0.045, 0.015)
EDGES = np.asarray(pre.DEPTH_BAND_EDGES_M, dtype=np.float32)


def square(x, y, h=3.0, side=0.4):
    return {"name": "p", "x": x, "y": y, "z": h / 2.0, "size": [side, side, h]}


def cylinder(x, y, r=0.15, h=3.0):
    return {"name": "t", "x": x, "y": y, "z": h / 2.0, "radius": r, "shape": "cylinder", "size": [2 * r, 2 * r, h]}


def band_of(depth_m):
    return int(np.searchsorted(EDGES, np.float32(depth_m), side="right"))


def cells_of(pose, pillars, rays=RAYS):
    return pre.analytic_cell_depth(pose, pillars, F, (SRC_W, SRC_H), GRID, MOUNT, rays)


def bands_of(pose, pillars):
    return pre.depth_bands(cells_of(pose, pillars))


# ---------------------------------------------------------------- 独立的三维求交（测试里的真值）
def oracle_cell_depth(pose, pillars, rays=3):
    """每格 rays×rays 条射线沿光轴的最小深度。射线方向取前向分量为 1 的向量，参数 t 就是沿光轴的深度。
    方柱 = 轴对齐方盒（三维 slab），圆柱 = 闭竖直圆柱（侧面 + 顶底圆盘面），地面 z=0。"""
    x, y, z, yaw = pose
    cy_, sy_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    o = np.array([x + MOUNT[0] * cy_, y + MOUNT[0] * sy_, z + MOUNT[1]])
    fwd, left = np.array([cy_, sy_, 0.0]), np.array([-sy_, cy_, 0.0])
    dh, dw = GRID
    out = np.full(GRID, np.inf)
    for r in range(dh):
        for c in range(dw):
            best = math.inf
            for ky in range(rays):
                for kx in range(rays):
                    u = (c + (kx + 0.5) / rays) * SRC_W / dw
                    v = (r + (ky + 0.5) / rays) * SRC_H / dh
                    xr, yd = (u - SRC_W / 2.0) / F, (v - SRC_H / 2.0) / F
                    d = fwd - xr * left + np.array([0.0, 0.0, -yd])
                    t = math.inf
                    if d[2] < 0:
                        t = -o[2] / d[2]
                    for p in pillars:
                        h = p["size"][2]
                        zlo, zhi = p["z"] - h / 2, p["z"] + h / 2
                        tp = _hit3(o, d, p, zlo, zhi)
                        if tp is not None and tp < t:
                            t = tp
                    best = min(best, t)
            out[r, c] = best
    return out


def _hit3(o, d, p, zlo, zhi):
    if p.get("radius") is not None:
        R = float(p["radius"])
        ox, oy = o[0] - p["x"], o[1] - p["y"]
        if ox * ox + oy * oy <= R * R and zlo <= o[2] <= zhi:
            return 0.0
        cands = []
        a = d[0] ** 2 + d[1] ** 2
        if a > 1e-15:
            b = 2 * (ox * d[0] + oy * d[1])
            cc = ox * ox + oy * oy - R * R
            disc = b * b - 4 * a * cc
            if disc >= 0:
                for t in ((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)):
                    if t >= 0 and zlo <= o[2] + d[2] * t <= zhi:
                        cands.append(t)
        if abs(d[2]) > 1e-15:
            for zcap in (zlo, zhi):
                tc = (zcap - o[2]) / d[2]
                if tc >= 0 and (ox + tc * d[0]) ** 2 + (oy + tc * d[1]) ** 2 <= R * R:
                    cands.append(tc)
        return min(cands) if cands else None
    hx, hy = p["size"][0] / 2, p["size"][1] / 2
    lo, hi = -math.inf, math.inf
    for k, (c, hh) in enumerate(((p["x"], hx), (p["y"], hy), (p["z"], (zhi - zlo) / 2))):
        if abs(d[k]) < 1e-12:
            if abs(o[k] - c) > hh:
                return None
            continue
        a, b = (c - hh - o[k]) / d[k], (c + hh - o[k]) / d[k]
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return None if lo > hi or hi < 0 else max(lo, 0.0)


# ---------------------------------------------------------------- 夹具：合成 episode + e1_mini + 位姿 + scene.json
def read_meta(path):
    with open(path, "rb") as f:
        head = f.read(48)
        meta_off = struct.unpack_from("<Q", head, 36)[0]
        meta_len = struct.unpack_from("<I", head, 44)[0]
        f.seek(meta_off)
        return json.loads(f.read(meta_len))


def read_aux(ds_dir):
    b = (Path(ds_dir) / "dataset.aux").read_bytes()
    magic, n, flags, dh, dw, k, unk, _ = struct.unpack_from("<8sQIHHBBH", b, 0)
    assert magic == b"UAVRAUX1" and k == 8 and unk == 255
    depth = b[128 + 5 * n:] if flags & 1 else None
    return b, n, depth, (dh, dw)


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        subprocess.run([PY, str(REPO / "tools/make_synthetic_episode.py"), "--out", str(self.tmp / "ep")],
                       check=True, capture_output=True)
        self.eps = [self.tmp / "ep" / n for n in ("ep1", "ep2", "ep3")]

    def tearDown(self):
        self._tmp.cleanup()

    def man(self, ep):
        return json.loads((ep / "episode_manifest.json").read_text(encoding="utf-8"))

    def save(self, ep, man):
        (ep / "episode_manifest.json").write_text(json.dumps(man), encoding="utf-8")

    def rows(self, ep):
        return [json.loads(line) for line in (ep / "actions.jsonl").read_text().splitlines() if line]

    def save_rows(self, ep, rows):
        (ep / "actions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def poses(self, ep):
        """每个 episode 每步一个位姿：x 随步数前进、y 与 yaw 随 episode 和步数变化，高度 2.0 m 上下。"""
        k = int(ep.name[-1])
        return {s: (0.0 + 0.1 * s, 0.05 * k * (-1) ** s, 1.97 + 0.01 * s, 4.0 * s * (-1) ** k) for s in range(6)}

    def make_analytic(self, ep, pillars, vehicle="e1_mini", scene=True, pose=True):
        man = self.man(ep)
        if vehicle is not None:
            man["vehicle"] = vehicle
        self.save(ep, man)
        if scene:
            (ep / "scene.json").write_text(json.dumps({"pillars": pillars}), encoding="utf-8")
        rows = self.rows(ep)
        if pose:
            poses = self.poses(ep)
            for row in rows:
                x, y, z, yaw = poses[row["step"]]
                row["pose_before"] = {"x": x, "y": y, "z": z}
                row["yaw_before"] = yaw
            self.save_rows(ep, rows)

    def run_pre(self, out, eps, *flags):
        return subprocess.run([PY, str(PRE_PY), "--out", str(out), *flags, *map(str, eps)],
                              capture_output=True, text=True)


class AnalyticGeometryTests(unittest.TestCase):
    def test_card_pillar_one_meter_ahead(self):
        # 柱面距相机 1.0 m（柱心 1.2 m，宽 0.4 m），相机高 2.0 m（机体 z = 1.985 + 安装高 0.015）。
        # 地平线行 = 第 7 行（v = 112..128 含 cy = 120）。柱面深度恒为 1.0 m（沿光轴），落在 [0.8, 1.2) 的第 3 带。
        pillar = square(MOUNT[0] + 1.2, 0.0)
        b = bands_of((0.0, 0.0, 1.985, 0.0), [pillar])
        self.assertTrue((b[7, 9:11] == 3).all(), "中间列的地平线行落在 0.8–1.2 m 的分带")
        self.assertEqual(b[7, 0], 7, "最左列落在最远分带")
        self.assertEqual(b[7, 19], 7, "最右列落在最远分带")
        # 柱面横向范围 ±0.4/2 / 1.0 = ±0.2（tan），像素 ±55 px：第 6–13 列至少有一条射线打到柱，第 0–5、14–19 列没有。
        self.assertTrue((b[7, 6:14] == 3).all())
        self.assertTrue((b[7, :6] == 7).all() and (b[7, 14:] == 7).all())
        # 柱高 3.0 m、相机 2.0 m、面在 1.0 m：整列从上到下都是柱面（柱脚在画面下沿之外）
        self.assertTrue((b[:, 6:14] == 3).all())

    def test_ground_band_at_camera_height_2m(self):
        # 没有柱：最下一行的射线 v = 237.33 px → 地面沿光轴深度 = h·f/(v − cy) = 2.0 × 277.13 / 117.33 = 4.72 m，第 7 带。
        b = bands_of((0.0, 0.0, 1.985, 0.0), [])
        self.assertTrue((b[14] == band_of(2.0 * F / ((14 + 2.5 / 3) * 16 - 120.0))).all())
        self.assertTrue((b[14] == 7).all())
        self.assertTrue((b[:8] == 7).all(), "地平线及以上没有地面也没有柱：最远带")

    def test_ground_bands_at_low_camera(self):
        # 相机高 1.0 m 时地面分带随行变化：逐行与公式一致（地面深度与列无关），并且各行离分带边界 > 1e-6 m（防夹具恰好踩边界）。
        h = 1.0
        b = bands_of((0.0, 0.0, h - MOUNT[1], 0.0), [])
        seen = set()
        for r in range(8, 15):
            depth = h * F / ((r + 2.5 / 3) * 16 - 120.0)
            self.assertTrue(all(abs(depth - e) > 1e-6 for e in pre.DEPTH_BAND_EDGES_M), (r, depth))
            self.assertTrue((b[r] == band_of(depth)).all(), (r, depth, b[r]))
            seen.add(band_of(depth))
        self.assertGreaterEqual(len(seen), 2, "夹具要能区分不同行的分带")

    def test_rays_above_pillar_top_do_not_hit(self):
        # 相机高 2.9 m、柱高 3.0 m、面在 1.0 m：|yd| > 0.1 的射线在面上的高度超过柱顶，打不到（记最远带）。
        # 第 5 行（v = 80..96）的第三条子射线 yd = −0.0962 仍打到柱面；第 0–4 行全部越过柱顶。
        b = bands_of((0.0, 0.0, 2.885, 0.0), [square(MOUNT[0] + 1.2, 0.0)])
        self.assertTrue((b[:5, 6:14] == 7).all(), b[:6, 6:14])
        self.assertTrue((b[5, 6:14] == 3).all())
        self.assertTrue((b[6:, 6:14] == 3).all())

    def test_forest_cylinder_is_supported(self):
        trunk = cylinder(MOUNT[0] + 2.15, 0.0, r=0.15)
        b = bands_of((0.0, 0.0, 1.985, 0.0), [trunk])
        self.assertTrue((b[7, 9:11] == band_of(2.0)).all(), "正前方树干前沿 2.0 m：[1.8, 2.5) 的第 5 带")
        self.assertEqual(b[7, 0], 7)
        np.testing.assert_allclose(cells_of((0.0, 0.0, 1.985, 0.0), [trunk]),     # float32 的相对误差 ≤ 6e-8
                                   oracle_cell_depth((0.0, 0.0, 1.985, 0.0), [trunk]), rtol=1e-6)

    def test_cylinder_top_and_bottom_caps_match_box_and_oracle(self):
        # I2-8 第 2 点 / TEST 2f：相机高于柱顶（oz=2.115 > 柱顶 2.0）向下看 x=1.0 m 处高 2.0 m 的柱顶中心，
        # 圆柱不能穿透顶盖直落地面（15.700 m），必须命中顶盖圆盘，与同外接方柱及独立 3D oracle 一致；
        # 悬空圆柱从下方抬头看底盖（zc - zh）同样命中底盖。
        box_p = [{"name": "b", "x": 1.0, "y": 0.0, "z": 1.0, "size": [0.4, 0.4, 2.0]}]
        cyl_p = [{"name": "c", "x": 1.0, "y": 0.0, "z": 1.0, "radius": 0.2, "size": [0.4, 0.4, 2.0]}]
        pose_high = (0.0, 0.0, 2.1, 0.0)  # 相机 oz = 2.115 > 柱顶 2.0
        d_box = cells_of(pose_high, box_p)
        d_cyl = cells_of(pose_high, cyl_p)
        self.assertLess(float(d_cyl[9, 10]), 1.2, "圆柱顶盖必须被命中，不能穿透到地面 15.7 m")
        self.assertAlmostEqual(float(d_cyl[9, 10]), float(d_box[9, 10]), places=3)
        np.testing.assert_allclose(d_cyl, oracle_cell_depth(pose_high, cyl_p), rtol=1e-6)

        # 底盖（zc - zh = 1.5 > 相机 oz = 1.0）：向上射线打到底面圆盘
        float_cyl = [{"name": "fc", "x": 1.0, "y": 0.0, "z": 2.0, "radius": 0.25, "size": [0.5, 0.5, 1.0]}]
        pose_low = (0.0, 0.0, 1.0 - MOUNT[1], 0.0)  # oz = 1.0 < 底面 1.5
        d_bot = cells_of(pose_low, float_cyl)
        want_bot = oracle_cell_depth(pose_low, float_cyl)
        self.assertTrue(np.isfinite(want_bot[:5, 9:11]).any(), "夹具上方射线应打到悬空圆柱底盖")
        fin = np.isfinite(want_bot)
        self.assertEqual(np.isinf(d_bot).tolist(), np.isinf(want_bot).tolist())
        np.testing.assert_allclose(d_bot[fin], want_bot[fin], rtol=1e-6)

    def test_azimuth_is_left_positive_and_yaw_is_counter_clockwise(self):
        # 柱在机头左前方（x 前 1.0、y 左 0.4，方位 +21.8°）：出现在画面左侧（小列号），右侧没有。
        pillar = square(MOUNT[0] + 1.0, 0.4)
        b = bands_of((0.0, 0.0, 1.985, 0.0), [pillar])
        cols = np.nonzero((b[7] < 7))[0]
        self.assertTrue(len(cols) > 0 and cols.max() < 10, cols)
        # 机头左转 21.8°（yaw 增大 = YAW_LEFT）后，柱在正前方：中间两列（9、10）有柱，范围大致左右对称。
        yaw = math.degrees(math.atan2(0.4, 1.0))
        b2 = bands_of((0.0, 0.0, 1.985, yaw), [pillar])
        cols2 = np.nonzero((b2[7] < 7))[0]
        self.assertTrue(9 in cols2 and 10 in cols2, cols2)
        self.assertLessEqual(abs((cols2.min() + cols2.max()) / 2.0 - 9.5), 1.5, cols2)
        # 机头右转同样角度（yaw 减小 = YAW_RIGHT）：柱相对光轴偏到左 36°–59°，出了 ±30° 视场，地平线行没有柱
        b3 = bands_of((0.0, 0.0, 1.985, -yaw), [pillar])
        self.assertTrue((b3[7] == 7).all(), b3[7])

    def test_matches_independent_3d_oracle(self):
        rng = np.random.default_rng(20261112)
        for case in range(12):
            pose = (rng.uniform(-1.0, 3.0), rng.uniform(-1.5, 1.5), rng.uniform(1.0, 2.2), rng.uniform(-70.0, 70.0))
            cam = (pose[0] + MOUNT[0] * math.cos(math.radians(pose[3])), pose[1] + MOUNT[0] * math.sin(math.radians(pose[3])))
            pillars = []
            while len(pillars) < rng.integers(3, 8):
                x, y = rng.uniform(0.6, 5.0), rng.uniform(-2.5, 2.5)
                if math.hypot(x - cam[0], y - cam[1]) < 0.7:      # 相机不落在柱内 / 贴着柱：避开起点在柱内的退化情形
                    continue
                k = len(pillars)
                pillars.append(square(x, y) if k % 2 == 0 else cylinder(x, y, r=rng.uniform(0.08, 0.22), h=float(rng.choice([2.5, 3.0]))))
            got = cells_of(pose, pillars)
            want = oracle_cell_depth(pose, pillars)
            self.assertEqual(np.isinf(got).tolist(), np.isinf(want).tolist(), (case, pose))
            fin = np.isfinite(want)
            np.testing.assert_allclose(got[fin], want[fin], rtol=1e-6, err_msg="case %d pose %r" % (case, pose))
            self.assertTrue((fin & (want < 5.0)).any(), "夹具要有近处的命中，否则这条断言没有区分力：case %d" % case)

    def test_horizon_row_matches_rule_policy_hit(self):
        # 与既有的水平截面求交（rule_policy._hit，教师 / 规则策略共用）对拍：地平线行（第 7 行的中间子射线 yd = 0）只由柱面决定，
        # 沿光轴深度 = 水平射线距离 r · cosφ。不在柱子方向上的列落在地面（> 50 m）。
        import rule_policy as rp
        pose = (0.3, -0.2, 1.985, 17.0)
        pillars = [square(1.8, 0.1), cylinder(2.6, -0.9, 0.18), square(3.4, 0.6)]
        cells = cells_of(pose, pillars)
        x, y, _, yaw = pose
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        ox, oy = x + MOUNT[0] * c, y + MOUNT[0] * s
        hits = 0
        for col in range(20):
            best = math.inf
            for kx in range(RAYS):
                xr = ((col + (kx + 0.5) / RAYS) * 16 - 160) / F
                n = math.sqrt(1 + xr * xr)
                for p in pillars:
                    r = rp._hit(ox, oy, (c + xr * s) / n, (s - xr * c) / n, p)
                    if r is not None:
                        best = min(best, r / n)
            if math.isfinite(best):
                hits += 1
                self.assertAlmostEqual(float(cells[7, col]), best, places=4, msg="col %d" % col)
            else:
                self.assertGreater(float(cells[7, col]), 50.0, "col %d" % col)
        self.assertGreaterEqual(hits, 5)

    def test_dense_render_pixel_ids(self):
        # 稠密渲染（每格 16×16 条射线 = 像素中心）给出每个像素属于哪根柱（≥0）、地面（−1）、天空 / 远处（−2），compare.py 用它归类。
        pillars = [square(MOUNT[0] + 1.2, 0.0), square(MOUNT[0] + 2.5, 1.0)]
        depth, pid = pre.analytic_ray_depth((0.0, 0.0, 1.985, 0.0), pillars, F, (SRC_W, SRC_H), GRID, MOUNT, 16, want_pid=True)
        self.assertEqual(depth.shape, (SRC_H, SRC_W))
        self.assertEqual(pid.shape, (SRC_H, SRC_W))
        self.assertEqual(int(pid[119, 160]), 0, "画面中心打到第 0 根柱")
        self.assertEqual(int(pid[239, 5]), -1, "左下角是地面")
        self.assertEqual(int(pid[10, 5]), -2, "左上角是天空 / 远处")
        self.assertGreater(int((pid == 1).sum()), 0, "第 1 根柱（右前方、更远）也在视场里")
        # 与格内取 min 的版本一致（逐位）
        cells = pre.analytic_cell_depth((0.0, 0.0, 1.985, 0.0), pillars, F, (SRC_W, SRC_H), GRID, MOUNT, 16)
        np.testing.assert_array_equal(cells, depth.reshape(15, 16, 20, 16).min(axis=(1, 3)).astype(np.float32))


class AnalyticCliTests(Fixture):
    PILLARS = [square(MOUNT[0] + 1.2, 0.0), square(2.6, 0.9), cylinder(3.1, -0.7, r=0.15)]

    def setUp(self):
        super().setUp()
        for ep in self.eps:
            self.make_analytic(ep, self.PILLARS)

    def expected_bands(self, radius):
        out = []
        for ep in self.eps:
            poses = self.poses(ep)
            n = len(self.rows(ep))
            cells = np.stack([cells_of(poses[s], self.PILLARS) for s in range(n)])
            if radius > 0:
                cells = pre.dilate_by_envelope(cells, F, SRC_W / GRID[1], radius)
            out.append(pre.depth_bands(cells))
        return np.concatenate(out)

    def test_cli_writes_analytic_bands_without_envelope(self):
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "analytic", "--envelope-radius", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds" / "dataset.uavr")
        self.assertEqual(meta["aux_sidecar"]["depth"], True)
        self.assertEqual(meta["aux_sidecar"]["depth_grid"], [15, 20])
        self.assertEqual(meta["aux_sidecar"]["depth_source"], "analytic")
        _, n, depth, grid = read_aux(self.tmp / "ds")
        self.assertEqual(grid, (15, 20))
        got = np.frombuffer(depth, dtype=np.uint8).reshape(n, 15, 20)
        self.assertTrue((got == self.expected_bands(0.0)).all())
        self.assertTrue((got < 255).all(), "解析深度没有未知格")
        self.assertTrue((got == 7).any() and (got < 7).any(), "夹具里既有柱面格也有最远带")

    def test_cli_applies_envelope_dilation_like_the_file_path(self):
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "analytic")
        self.assertEqual(r.returncode, 0, r.stderr)
        _, n, depth, _ = read_aux(self.tmp / "ds")
        got = np.frombuffer(depth, dtype=np.uint8).reshape(n, 15, 20)
        self.assertTrue((got == self.expected_bands(pre.DEFAULT_ENVELOPE_RADIUS_M)).all())
        self.assertFalse((got == self.expected_bands(0.0)).all(), "夹具里的包络膨胀要改变一些格，否则这条断言没有区分力")
        self.assertEqual(read_meta(self.tmp / "ds" / "dataset.uavr")["aux_sidecar"]["envelope_radius_m"],
                         pre.DEFAULT_ENVELOPE_RADIUS_M)

    def test_depth_files_are_ignored_by_analytic(self):
        r0 = self.run_pre(self.tmp / "ds_nofile", self.eps, "--depth-source", "analytic")
        self.assertEqual(r0.returncode, 0, r0.stderr)
        from PIL import Image
        for ep in self.eps:
            man = self.man(ep)
            man["depth_format"] = "png16_mm"
            self.save(ep, man)
            (ep / "depth").mkdir(exist_ok=True)
            rows = self.rows(ep)
            for row in rows:
                name = "depth/depth_%05d.png" % row["step"]
                Image.fromarray(np.full((SRC_H, SRC_W), 700, dtype=np.uint16)).save(ep / name)
                row["depth_file"] = name
            self.save_rows(ep, rows)
        r1 = self.run_pre(self.tmp / "ds_withfile", self.eps, "--depth-source", "analytic")
        self.assertEqual(r1.returncode, 0, r1.stderr)
        a, b = read_aux(self.tmp / "ds_nofile")[2], read_aux(self.tmp / "ds_withfile")[2]
        self.assertEqual(a, b, "analytic 不读深度文件：有无文件结果相同")
        r2 = self.run_pre(self.tmp / "ds_file", self.eps, "--depth-source", "file")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertNotEqual(read_aux(self.tmp / "ds_file")[2], a, "file 读的是深度文件（常量 0.7 m），与解析结果不同")

    def test_waypoint_episode_is_an_error_before_any_output(self):
        for ep in self.eps:
            m = self.man(ep)
            m["waypoint"] = {"x": 3.0, "y": 0.0}
            m["stop_radius"] = 0.5
            self.save(ep, m)
        r = self.run_pre(self.tmp / "ds_wp", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("waypoint", r.stderr)
        self.assertFalse((self.tmp / "ds_wp" / "dataset.uavr").exists(), "报错发生在写任何输出之前")
        self.assertFalse((self.tmp / "ds_wp" / "dataset.aux").exists())

    def test_3dgs_episode_is_an_error(self):
        for ep in self.eps:
            m = self.man(ep)
            m["geometry_version"] = "3dgs.test"
            m["renderer"] = {"backend": "gsplat"}
            self.save(ep, m)
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("3DGS", r.stderr)
        self.assertFalse((self.tmp / "ds" / "dataset.uavr").exists())

    def test_non_e1_vehicle_missing_scene_or_pose_are_errors(self):
        for ep in self.eps:
            m = self.man(ep)
            m["vehicle"] = "x3"
            self.save(ep, m)
        r = self.run_pre(self.tmp / "ds_x3", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("e1_mini", r.stderr)
        for ep in self.eps:
            m = self.man(ep)
            m.pop("vehicle", None)
            self.save(ep, m)
        r = self.run_pre(self.tmp / "ds_novehicle", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("vehicle", r.stderr)

    def test_missing_scene_json_or_empty_pillars_is_an_error(self):
        (self.eps[1] / "scene.json").unlink()
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("scene.json", r.stderr)
        (self.eps[1] / "scene.json").write_text(json.dumps({"pillars": []}), encoding="utf-8")
        r = self.run_pre(self.tmp / "ds2", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("pillars", r.stderr)

    def test_missing_pose_is_an_error(self):
        rows = self.rows(self.eps[2])
        rows[3].pop("yaw_before")
        self.save_rows(self.eps[2], rows)
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "analytic")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("pose_before", r.stderr)

    def test_unknown_depth_source_is_rejected(self):
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "magic")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--depth-source", r.stderr)
        self.assertIn("invalid choice", r.stderr)
        self.assertTrue("'file'" in r.stderr and "'analytic'" in r.stderr, r.stderr)
        self.assertFalse((self.tmp / "ds" / "dataset.uavr").exists())

    def test_constants_follow_the_card(self):
        self.assertEqual(pre.DEPTH_SOURCES, ("file", "analytic"))
        self.assertEqual(pre.DEFAULT_DEPTH_SOURCE, "file")
        self.assertEqual(pre.ANALYTIC_RAYS, 3, "卡：每格 3×3 条射线")
        self.assertEqual(pre.DEPTH_GRID_DOWN, 8)

    def test_camera_mount_matches_e1_mini_json(self):
        mount = json.loads((REPO / "sim/dynamics/e1_mini.json").read_text(encoding="utf-8"))["camera_mount"]["value"]
        self.assertEqual(mount["pitch_deg"], 0.0)
        self.assertEqual(pre.CAMERA_MOUNT_BY_VEHICLE["e1_mini"], (mount["xyz"][0], mount["xyz"][2]))
        self.assertEqual(mount["xyz"][1], 0.0)


class FileSourceUnchangedTests(Fixture):
    """--depth-source file 与改动前逐字节相同（卡：用现有夹具）。"""

    # 取自 3e03e7e（改动前）的 preprocess.py 在本夹具（合成 episode + png16 深度，write_depth）上、不带 --depth-source 的输出（canonical()）。
    # 取法：git show 3e03e7e:tools/preprocess.py 的副本打包同一夹具；两次运行（不同临时目录）四个值逐位相同。
    GOLDEN = {
        "uavr_head": "fd7c03bc3712fa3b645b15dc7277d7cef7c0da3c046ffef8ab5bccbe21683326",
        "labels": "67be6528565979a2430be95358a31857dff97abc0522aeb6923974f86672fa4d",
        "aux": "354d2f99e1464abee3c4987ffa38edf138929e5f04c5a56ddb0896bd6b6a828c",
        "meta": "0be03c09c11d61f9fa5a0560cf9589337460007a2fd3fc7dec6a4efb0bd22f2c",
    }

    def write_depth(self):
        from PIL import Image
        for ep in self.eps:
            man = self.man(ep)
            man["depth_format"] = "png16_mm"
            self.save(ep, man)
            (ep / "depth").mkdir(exist_ok=True)
            rows = self.rows(ep)
            for row in rows:
                d = np.full((SRC_H, SRC_W), 3000, dtype=np.uint16)
                d[:, :160] = 400
                d[:16, :] = 0
                d[100:140, 200:260] = 900 + 10 * row["step"]
                name = "depth/depth_%05d.png" % row["step"]
                Image.fromarray(d).save(ep / name)
                row["depth_file"] = name
            self.save_rows(ep, rows)

    @staticmethod
    def canonical(ds_dir):
        """各部分的 sha256。随时间 / 临时目录变化的字段不参与：meta 去掉 created_utc 与 episodes[].source_dir，
        头里的 meta_len（44..48）与侧车头里绑定 dataset.uavr 的 SHA256（28..60）不取。像素区不取：它依赖 Pillow 的缩放实现，
        跨机器可能差 1 个灰度；改动前后的像素区逐字节相同由 w1-8 的一次性对拍（草稿目录 evidence/）核对。"""
        raw = (Path(ds_dir) / "dataset.uavr").read_bytes()
        n = struct.unpack_from("<I", raw, 8)[0]
        labels_off, meta_off = struct.unpack_from("<QQ", raw, 28)
        meta = read_meta(Path(ds_dir) / "dataset.uavr")
        meta.pop("created_utc")
        for ep in meta["episodes"]:
            ep.pop("source_dir")
        aux = (Path(ds_dir) / "dataset.aux").read_bytes()
        sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
        assert meta_off - labels_off == n
        return {
            "uavr_head": sha(raw[:44]),
            "labels": sha(raw[labels_off:meta_off]),
            "aux": sha(aux[:28] + aux[60:]),
            "meta": sha(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8")),
        }

    def test_file_output_is_unchanged_and_default(self):
        self.write_depth()
        r0 = self.run_pre(self.tmp / "ds_default", self.eps)
        r1 = self.run_pre(self.tmp / "ds_file", self.eps, "--depth-source", "file")
        self.assertEqual((r0.returncode, r1.returncode), (0, 0), r0.stderr + r1.stderr)
        c0, c1 = self.canonical(self.tmp / "ds_default"), self.canonical(self.tmp / "ds_file")
        self.assertEqual(c0, c1, "默认等于显式 file")
        meta = read_meta(self.tmp / "ds_file" / "dataset.uavr")
        self.assertNotIn("depth_source", meta["aux_sidecar"])
        self.assertNotIn("depth_source", json.dumps(meta))
        self.assertEqual(sorted(meta["aux_sidecar"]), ["band_edges_m", "depth", "depth_grid", "envelope_radius_m", "file", "format"])
        self.assertEqual(c0, self.GOLDEN, "与改动前（3e03e7e）的输出不同：%r" % c0)
        # 像素区：默认与显式 file 逐字节相同（同一台机器同一份 Pillow，不依赖跨机器的缩放实现）
        n = 6 + 4 + 5
        img = 48 + n * 3 * 160 * 120
        self.assertEqual((self.tmp / "ds_default" / "dataset.uavr").read_bytes()[48:img],
                         (self.tmp / "ds_file" / "dataset.uavr").read_bytes()[48:img])

    def test_file_without_depth_files_is_unchanged(self):
        r = self.run_pre(self.tmp / "ds", self.eps, "--depth-source", "file")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_meta(self.tmp / "ds" / "dataset.uavr")
        self.assertEqual(meta["aux_sidecar"]["depth"], False)
        self.assertNotIn("depth_source", meta["aux_sidecar"])


if __name__ == "__main__":
    unittest.main()
