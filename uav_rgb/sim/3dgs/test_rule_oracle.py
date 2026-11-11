#!/usr/bin/env python3
"""rule_oracle.py（3DGS 上的规则层上限策略）单测：纯 Python + numpy，不需要 torch / GPU。

用 3dgs-venv 或 .venv-tools 跑：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_rule_oracle.py
覆盖：
  1. occ 列距离：桩占据（一堵墙 + 一个门洞 + 实心背墙）上与闭式解析值一致；带膨胀时与 collides 判定逐射线一致；
  2. depth 列距离：解析光线投射造的桩深度图（z 深度）上与构造值一致，并与 Gazebo 的 rule_policy.column_distances 对拍；
     高度带（包络 ± 0.10 m）、roll/pitch、无效像素（0）三个口径；
  3. 策略：n 的符号（目标在左 -> n < 0 -> 规则朝左转回）、last_turn、记忆按集清零、永不 SAFE_NEUTRAL、
     两个来源都落日志而决策只用主来源、规则调用与 rule_policy.rule_v1_step 直接调用逐位相同、闭环里能转向目标。
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import importlib
import math
import os
import random
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
ro = importlib.import_module("sim.3dgs.rule_oracle")
scene_mod = importlib.import_module("sim.3dgs.scene")
collision = importlib.import_module("sim.3dgs.collision")
guard = importlib.import_module("sim.3dgs.guard")
import rule_policy as rp  # noqa: E402  sim/tools：Gazebo 侧同一份规则

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
MOUNT = {"xyz": [0.045, 0.0, 0.015], "pitch_deg": 0.0}          # e1_mini.json camera_mount
Z0 = 1.25                                                      # 桩场景的飞行高度：z 层 [1.2, 1.3) 的中心，避开体素边界
HALF_H = 0.028


# ---------------------------------------------------------------- 桩：占据
def stub_occ(door=True, dilate=False):
    """点云占据（点都落在体素中心，体素 0.1、原点在 0.1 的整数倍上，格边界没有歧义）：
    x ∈ [3.0, 3.1) 一堵墙，|y| < 0.4 为门洞（door=False 实心）；x ∈ [3.6, 3.7) 一堵实心背墙；两堵墙 y 向都到 ±3.0。
    dilate=False：机体半径 0.04 < 半个体素 -> 膨胀 0 格，栅格边界就是墙面，解析值是闭式；True：0.102 -> 膨胀 1 格（十字形）。"""
    pts = []
    for z in (1.05, 1.15, 1.25, 1.35):
        for i in range(-30, 30):
            y = (i + 0.5) * 0.1
            if not (door and abs(y) < 0.4):
                pts.append((3.05, y, z))
            pts.append((3.65, y, z))
    return collision.Occupancy(np.array(pts), None, None, None, z_range=(1.0, 1.4), voxel_m=0.1,
                               body_radius_m=0.102 if dilate else 0.04, body_half_height_m=HALF_H,
                               pad_m=1.05, min_points=1, max_scale_m=None, source="points")


def empty_occ():
    return collision.Occupancy(np.zeros((0, 3)), None, None, None, z_range=(1.0, 1.4), voxel_m=0.1, min_points=1,
                               max_scale_m=None, source="points")


def ray_boxes(ox, oy, theta_deg, boxes):
    """水平射线 (ox, oy) + t·(cosθ, sinθ) 与轴对齐盒 (x0, x1, y0, y1) 的最近交点距离（slab 法）；没有交点 inf。"""
    ux, uy = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    best = math.inf
    for x0, x1, y0, y1 in boxes:
        lo, hi = 0.0, math.inf
        for o, u, a, b in ((ox, ux, x0, x1), (oy, uy, y0, y1)):
            if abs(u) < 1e-12:
                if not (a <= o <= b):
                    lo, hi = 1.0, 0.0
                continue
            t0, t1 = (a - o) / u, (b - o) / u
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        if lo <= hi:
            best = min(best, lo)
    return best


STUB_BOXES = [(3.0, 3.1, 0.4, 3.0), (3.0, 3.1, -3.0, -0.4), (3.6, 3.7, -3.0, 3.0)]


def expected_occ_columns(x, y, yaw_deg, boxes=STUB_BOXES, n_cols=20, col_deg=3.0, n_rays=5, max_range=4.0, cam=0.045):
    """闭式期望：每列 5 条射线（含两条边缘）对盒子求交取最小，上限 max_range；写法独立于 rule_oracle。"""
    yaw = math.radians(yaw_deg)
    ox, oy = x + cam * math.cos(yaw), y + cam * math.sin(yaw)
    out = []
    for j in range(n_cols):
        phi = (n_cols / 2 - 0.5 - j) * col_deg
        out.append(min(max_range, min(ray_boxes(ox, oy, yaw_deg + phi + (k / (n_rays - 1) - 0.5) * col_deg, boxes)
                                      for k in range(n_rays))))
    return np.array(out)


# ---------------------------------------------------------------- 桩：状态与深度
def quat_zyx(roll_deg, pitch_deg, yaw_deg):
    """机体 FLU -> nav 的 (w, x, y, z)：R = Rz(yaw) Ry(pitch) Rx(roll)。"""
    r, p, y = (math.radians(v) / 2 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class State:
    """policy 接口里的 state（= 动力学对象）：x / y / z / yaw（弧度）、quat、params.camera_mount、camera_viewmat()。"""

    def __init__(self, x, y, yaw_deg, z=Z0, roll_deg=0.0, pitch_deg=0.0):
        self.x, self.y, self.z, self.yaw = x, y, z, math.radians(yaw_deg)
        self.quat = quat_zyx(roll_deg, pitch_deg, yaw_deg)
        self.params = types.SimpleNamespace(camera_mount=MOUNT)

    def camera_viewmat(self, nav_to_world=None):
        return scene_mod.viewmat_from_quat((self.x, self.y, self.z), self.quat, MOUNT, nav_to_world)


def wall_x(pos, z_lo=-9.0, z_hi=9.0, y_lo=-9.0, y_hi=9.0):
    return ("x", pos, (y_lo, y_hi), (z_lo, z_hi))


def synth_depth(view, walls=(), poles=(), w=320, h=240, hfov=60.0):
    """解析光线投射造 z 深度图（米，0 = 无效）。view = nav->camera 4x4；像素中心 (u+0.5, v+0.5)；
    射线方向取相机系 (a, b, 1)，所以射线参数 t 就是 z 深度（与 gsplat 的 expected depth 同一口径，GPU 对拍见 depth_geometry_check.py）。
    walls：轴 'x' 的平面片 ('x', 位置, (y 范围), (z 范围))；poles：竖直圆柱 (px, py, 半径, z0, z1)。"""
    fx = 0.5 * w / math.tan(math.radians(hfov) / 2)
    r_cn, t_c = view[:3, :3], view[:3, 3]
    cam = -r_cn.T @ t_c
    v, u = np.mgrid[0:h, 0:w]
    dc = np.stack([(u + 0.5 - w / 2) / fx, (v + 0.5 - h / 2) / fx, np.ones((h, w))], -1).reshape(-1, 3)
    dn = dc @ r_cn                                                # nav 系方向（行向量右乘 R = R^T 左乘列向量）
    best = np.full(len(dn), np.inf)
    for _axis, pos, (y_lo, y_hi), (z_lo, z_hi) in walls:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (pos - cam[0]) / dn[:, 0]
        p = cam[None, :] + t[:, None] * dn
        ok = (t > 0) & (p[:, 1] >= y_lo) & (p[:, 1] <= y_hi) & (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi)
        best = np.where(ok & (t < best), t, best)
    for px, py, rad, z0, z1 in poles:
        ox, oy = cam[0] - px, cam[1] - py
        a = dn[:, 0] ** 2 + dn[:, 1] ** 2
        b = 2 * (ox * dn[:, 0] + oy * dn[:, 1])
        disc = b * b - 4 * a * (ox * ox + oy * oy - rad * rad)
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2 * a)
        z = cam[2] + t * dn[:, 2]
        ok = (disc >= 0) & (t > 0) & (z >= z0) & (z <= z1)
        best = np.where(ok & (t < best), t, best)
    return np.where(np.isfinite(best), best, 0.0).reshape(h, w).astype(np.float32)


def depth_cols(depth, st, **kw):
    return ro.depth_columns(depth, st.camera_viewmat(None), st.z, math.degrees(st.yaw), HALF_H, **kw)


# ---------------------------------------------------------------- 1. occ 列距离
class TestOccColumns(unittest.TestCase):
    def test_wall_with_doorway_matches_closed_form(self):
        occ = stub_occ(door=True, dilate=False)
        for x, y, yaw in ((0.0, 0.0, 0.0), (0.5, 1.1, 0.0), (0.0, -0.8, 12.0), (1.0, 0.3, -20.0), (1.0, 0.6, -15.0)):
            with self.subTest(pose=(x, y, yaw)):
                got = ro.occ_columns(occ, x, y, Z0, yaw)
                want = expected_occ_columns(x, y, yaw)
                self.assertEqual(got.shape, (20,))
                self.assertLess(np.abs(got - want).max(), 0.01, (got.round(3), want.round(3)))

    def test_ray_that_only_clips_a_cell_corner_is_missed_by_design(self):
        # 步进只在采样点上查栅格：射线擦过门框格子的角、格内路程短于采样间距（0.05 m）时漏掉（这里 5 mm）。
        # 墙厚 >= 1 个体素，穿墙的射线格内路程 >= 0.1 m，不会漏；只有擦角的会，且漏掉的是角上几毫米，不影响规则（门框的点下一步 / 邻列会补上）。
        d = ro.occ_columns(stub_occ(door=True), 0.5, 0.9, Z0, 0.0)
        want = expected_occ_columns(0.5, 0.9, 0.0)
        self.assertAlmostEqual(float(want[18]), 2.863, delta=0.002)         # 闭式：第 18 列最右一条射线在 x=3.095 擦到门框内侧面
        self.assertGreater(float(d[18]) - float(want[18]), 0.3)             # 步进读到的是射线穿过门洞后的下一个命中
        self.assertLess(float(np.abs(np.delete(d, 18) - np.delete(want, 18)).max()), 0.01)    # 其余 19 列都对

    def test_doorway_columns_see_the_back_wall_and_the_rest_see_the_front_wall(self):
        d = ro.occ_columns(stub_occ(door=True), 0.0, 0.0, Z0, 0.0)
        # 正前方两列穿过门洞，看到背墙（x=3.6，相机在 x=0.045：3.555 m）；两侧看到前墙（x=3.0：2.955 m 再除 cos 方位）
        self.assertAlmostEqual(float(d[9]), 3.555, delta=0.01)
        self.assertAlmostEqual(float(d[10]), 3.555, delta=0.01)
        self.assertAlmostEqual(float(d[0]), 2.955 / math.cos(math.radians(27.0)), delta=0.01)      # 第 0 列扇区 [27°, 30°]，最近的射线在 27°
        self.assertAlmostEqual(float(d[19]), 2.955 / math.cos(math.radians(27.0)), delta=0.01)

    def test_solid_wall_gives_front_wall_everywhere(self):
        d = ro.occ_columns(stub_occ(door=False), 0.0, 0.0, Z0, 0.0)
        want = expected_occ_columns(0.0, 0.0, 0.0, boxes=[(3.0, 3.1, -3.0, 3.0)])
        self.assertLess(np.abs(d - want).max(), 0.01)
        self.assertTrue(((d > 2.95) & (d < 3.4)).all())

    def test_far_columns_are_capped_at_max_range(self):
        self.assertTrue((ro.occ_columns(empty_occ(), 0.0, 0.0, Z0, 0.0) == ro.MAX_RANGE_M).all())
        d = ro.occ_columns(stub_occ(door=False), -2.0, 0.0, Z0, 0.0)           # 墙在 5 m 外
        self.assertTrue((d == 4.0).all())

    def test_ray_origin_is_the_camera_not_the_body_centre(self):
        # 机体中心在 x=2.95 时相机在 2.995（前墙面 x=3.0）：正前方列距离 = 0.005；中心在 2.90、相机 2.945 -> 0.055
        d = ro.occ_columns(stub_occ(door=False), 2.90, 0.0, Z0, 0.0)
        self.assertAlmostEqual(float(d[9]), 0.055, delta=0.005)
        d2 = ro.occ_columns(stub_occ(door=False), 0.0, 0.0, Z0, 90.0)           # 朝 +y：相机在 y=0.045，前方没有墙
        self.assertTrue((d2 == 4.0).all())

    def test_origin_inside_obstacle_reads_zero(self):
        d = ro.occ_columns(stub_occ(door=False), 3.05, 1.0, Z0, 0.0)
        self.assertTrue((d[8:12] == 0.0).all(), d)

    def test_dilated_grid_matches_collides_ray_by_ray(self):
        occ = stub_occ(door=True, dilate=True)
        step = 0.001
        for x, y, yaw in ((0.0, 0.0, 0.0), (0.8, 0.7, -10.0), (1.2, -0.5, 25.0)):
            with self.subTest(pose=(x, y, yaw)):
                got = ro.occ_columns(occ, x, y, Z0, yaw)
                # 独立写法：每条射线 1 mm 细步进，逐点 collides（向量化的 _grid_hit 与 collides 同一判定）
                n_cols, n_rays = 20, 5
                ts = np.arange(0.0, 4.0 + step, step)
                want = []
                for j in range(n_cols):
                    best = 4.0
                    for k in range(n_rays):
                        th = math.radians(yaw + (n_cols / 2 - 0.5 - j) * 3.0 + (k / (n_rays - 1) - 0.5) * 3.0)
                        px = x + 0.045 * math.cos(math.radians(yaw)) + ts * math.cos(th)
                        py = y + 0.045 * math.sin(math.radians(yaw)) + ts * math.sin(th)
                        hit = occ._grid_hit(np.stack([px, py, np.full_like(px, Z0)], 1))
                        if hit.any():
                            best = min(best, float(ts[int(np.argmax(hit))]))
                    want.append(best)
                self.assertLess(np.abs(got - np.array(want)).max(), 0.004, (got.round(3), np.round(want, 3)))
        # 抽一条真值：第 9 列的距离处 collides 刚好由假变真
        d9 = float(ro.occ_columns(occ, 0.0, 0.0, Z0, 0.0)[9])
        self.assertTrue(d9 < 4.0)
        self.assertFalse(occ.collides(0.045 + d9 - 0.003, 0.0, Z0))
        self.assertTrue(occ.collides(0.045 + d9 + 0.003, 0.0, Z0))

    def test_unrefined_distance_is_within_half_voxel_step_of_refined(self):
        occ = stub_occ(door=True, dilate=True)
        a = ro.occ_columns(occ, 0.0, 0.0, Z0, 7.0, refine=False)
        b = ro.occ_columns(occ, 0.0, 0.0, Z0, 7.0, refine=True)
        self.assertTrue((a >= b - 1e-9).all())
        self.assertLessEqual(float((a - b).max()), 0.5 * occ.voxel + 1e-9)          # 采样间距 <= 半个体素


# ---------------------------------------------------------------- 2. depth 列距离
class TestDepthColumns(unittest.TestCase):
    def test_pole_matches_gazebo_analytic_column_distances(self):
        pole = {"x": 2.0, "y": 0.3, "radius": 0.15}
        for yaw in (0.0, 35.0, -70.0):
            with self.subTest(yaw=yaw):
                # 机体放在让杆子落在视场里的位置：杆在机体系 (2.0, 0.3)，换算回 nav
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                st = State(-(pole["x"] * c - pole["y"] * s), -(pole["x"] * s + pole["y"] * c), yaw)
                world_pole = {"x": 0.0, "y": 0.0, "radius": 0.15}
                depth = synth_depth(st.camera_viewmat(None), poles=[(0.0, 0.0, 0.15, 0.0, 3.0)])
                got = depth_cols(depth, st)
                want = np.array(rp.column_distances(st.x, st.y, yaw, [world_pole]))
                self.assertLess(np.abs(got - want).max(), 0.02, (got.round(3), want.round(3)))
                self.assertTrue((want < 4.0).any(), "杆子应当落在视场里，否则这个用例什么也没比")

    def test_wall_is_z_depth_not_ray_distance(self):
        st = State(0.0, 0.0, 0.0)
        depth = synth_depth(st.camera_viewmat(None), walls=[wall_x(3.0)])
        got = depth_cols(depth, st)
        want = np.array(rp.column_distances(0.0, 0.0, 0.0, [{"x": 3.5, "y": 0.0, "size": (1.0, 12.0)}]))
        self.assertLess(np.abs(got - want).max(), 0.02, (got.round(3), want.round(3)))
        self.assertGreater(got[0] - got[9], 0.3)                     # 平面墙的水平距离随方位拉长（2.955 -> 3.32 m）
        ray = depth_cols(depth, st, depth_is="ray")                 # 把同一张图当「沿射线的距离」读：边缘列不再变长，差 > 0.3 m
        self.assertGreater(float(np.abs(ray - got).max()), 0.3)

    def test_height_band_is_envelope_plus_ten_cm(self):
        st = State(0.0, 0.0, 0.0)
        view = st.camera_viewmat(None)
        band = HALF_H + ro.BAND_MARGIN_M
        cases = [("在包络内", (Z0 - 0.05, Z0 + 0.05), True), ("带内上沿", (Z0 + 0.10, Z0 + band - 0.01), True),
                 ("带外上沿", (Z0 + band + 0.01, Z0 + 0.8), False), ("带外下沿", (Z0 - 0.8, Z0 - band - 0.01), False),
                 ("整堵墙", (-9.0, 9.0), True)]
        for name, (lo, hi), seen in cases:
            with self.subTest(name):
                got = depth_cols(synth_depth(view, walls=[wall_x(3.0, lo, hi)]), st)
                if seen:
                    self.assertTrue(((got > 2.9) & (got < 3.4)).all(), got.round(2))
                else:
                    self.assertTrue((got == 4.0).all(), got.round(2))

    def test_roll_and_pitch_use_the_full_attitude(self):
        # 墙只有机体高度的一条带（±0.05 m）：机体俯仰 5° / 横滚 3° 时画面里的带是斜的，按完整四元数换到水平系才对得上
        level = State(0.0, 0.0, 0.0)
        ref = depth_cols(synth_depth(level.camera_viewmat(None), walls=[wall_x(3.0, Z0 - 0.05, Z0 + 0.05)]), level)
        for roll, pitch in ((0.0, 5.0), (3.0, 0.0), (3.0, -5.0)):
            with self.subTest(roll=roll, pitch=pitch):
                st = State(0.0, 0.0, 0.0, roll_deg=roll, pitch_deg=pitch)
                got = depth_cols(synth_depth(st.camera_viewmat(None), walls=[wall_x(3.0, Z0 - 0.05, Z0 + 0.05)]), st)
                self.assertLess(np.abs(got[2:-2] - ref[2:-2]).max(), 0.03, (got.round(3), ref.round(3)))   # 视场边缘两列在倾斜下覆盖不全，不比

    def test_invalid_pixels_are_ignored_and_empty_image_is_max_range(self):
        st = State(0.0, 0.0, 0.0)
        depth = synth_depth(st.camera_viewmat(None), walls=[wall_x(3.0)])
        depth[:, 140:180] = 0.0                                      # 中间约 7° 全部无效（渲染的 alpha 不足 / 天空）
        got = depth_cols(depth, st)
        self.assertEqual(float(got[9]), 4.0)
        self.assertEqual(float(got[10]), 4.0)
        self.assertLess(float(got[3]), 3.4)
        self.assertTrue((depth_cols(np.zeros((240, 320), np.float32), st) == 4.0).all())
        nan = depth.copy()
        nan[100, 100] = np.nan
        self.assertTrue(np.isfinite(depth_cols(nan, st)).all())

    def test_heading_is_respected_pole_to_the_left_lands_in_the_left_columns(self):
        st = State(0.0, 0.0, 0.0)
        left = depth_cols(synth_depth(st.camera_viewmat(None), poles=[(2.0, 0.9, 0.1, 0.0, 3.0)]), st)    # 偏左约 24°
        self.assertTrue((left[:4] < 4.0).any() and (left[10:] == 4.0).all(), left.round(2))
        right = depth_cols(synth_depth(st.camera_viewmat(None), poles=[(2.0, -0.9, 0.1, 0.0, 3.0)]), st)
        self.assertTrue((right[16:] < 4.0).any() and (right[:10] == 4.0).all(), right.round(2))


# ---------------------------------------------------------------- 3. 策略
def make_policy(occ=None, target=(5.0, 0.0), source="occ", params=None, **kw):
    return ro.RuleOraclePolicy(occ or empty_occ(), lambda x, y: target, source=source, params=params, **kw)


class TestSignAndRule(unittest.TestCase):
    def test_target_left_gives_negative_n_and_yaw_left(self):
        p = make_policy(target=(0.0, 5.0))                           # 目标正左，err = +90°
        action, _ = p(None, None, State(0.0, 0.0, 0.0))
        row = p.rows[0]
        self.assertGreater(row["err_deg"], 80.0)
        self.assertLess(row["n"], 0)
        self.assertEqual((action, row["entry"]), (YAW_LEFT, "V1-back"))        # n < 0 时「转回」是 YAW_LEFT

    def test_target_right_gives_positive_n_and_yaw_right(self):
        p = make_policy(target=(0.0, -5.0))
        action, _ = p(None, None, State(0.0, 0.0, 0.0))
        self.assertGreater(p.rows[0]["n"], 0)
        self.assertEqual(action, YAW_RIGHT)

    def test_n_is_round_of_minus_err_over_yaw_step(self):
        for err, step, want in ((66.0, 33.0, -2), (-66.0, 33.0, 2), (15.0, 33.0, 0), (20.0, 33.0, -1), (-100.0, 26.0, 4)):
            with self.subTest(err=err, step=step):
                tgt = (5.0 * math.cos(math.radians(err)), 5.0 * math.sin(math.radians(err)))
                p = make_policy(target=tgt, params={"yaw_step_deg": step})
                p(None, None, State(0.0, 0.0, 0.0))
                self.assertEqual(p.rows[0]["n"], want)
                self.assertAlmostEqual(p.rows[0]["err_deg"], err, places=1)

    def test_target_ahead_goes_forward(self):
        p = make_policy(target=(5.0, 0.0))
        action, _ = p(None, None, State(0.0, 0.0, 0.0))
        self.assertEqual((action, p.rows[0]["n"], p.rows[0]["entry"]), (FORWARD_SLOW, 0, "V1-fwd"))

    def test_heading_enters_the_error(self):
        p = make_policy(target=(5.0, 0.0))
        action, _ = p(None, None, State(0.0, 0.0, 90.0))               # 机头朝 +y，目标在右手边
        self.assertEqual(action, YAW_RIGHT)

    def test_rule_call_is_rule_v1_step_bit_for_bit(self):
        occ = stub_occ(door=True)
        pol = make_policy(occ, target=(6.0, 0.4))
        mem, last = [], None
        rng = random.Random(3)
        for i in range(40):
            st = State(rng.uniform(0.0, 2.4), rng.uniform(-1.5, 1.5), rng.uniform(-60, 60))
            action, _ = pol(None, None, st)
            d = ro.occ_columns(occ, st.x, st.y, st.z, math.degrees(st.yaw))
            err = guard.bearing_error_deg(st.x, st.y, math.degrees(st.yaw), (6.0, 0.4))
            n = int(round(-err / pol.params["yaw_step_deg"]))
            # 记忆是跨步的：用同一串位姿喂 rule_v1_step，步号 = 调用序号
            want, entry, mem, _ = rp.rule_v1_step(mem, i, st.x, st.y, math.degrees(st.yaw), [float(v) for v in d], n, last,
                                                  60.0, pol.params)
            self.assertEqual((action, pol.rows[-1]["entry"]), (want, entry), i)
            if want in (YAW_LEFT, YAW_RIGHT):
                last = want

    def test_never_safe_neutral(self):
        occ = stub_occ(door=True, dilate=True)
        rng = random.Random(11)
        pol = make_policy(occ, target=(6.0, 0.0))
        seen = set()
        for _ in range(300):
            pol.target_fn = lambda x, y, t=(rng.uniform(-6, 6), rng.uniform(-6, 6)): t
            st = State(rng.uniform(-2.0, 2.9), rng.uniform(-2.5, 2.5), rng.uniform(-180, 180))
            a, _ = pol(None, None, st)
            seen.add(a)
        self.assertTrue(seen <= {FORWARD_SLOW, YAW_LEFT, YAW_RIGHT}, seen)
        self.assertGreaterEqual(len(seen), 2)

    def test_last_turn_is_the_last_returned_turn(self):
        p = make_policy(target=(0.0, 5.0))
        p(None, None, State(0.0, 0.0, 0.0))                          # YAW_LEFT
        self.assertEqual(p.last_turn, YAW_LEFT)
        p.target_fn = lambda x, y: (5.0, 0.0)
        a, _ = p(None, None, State(0.0, 0.0, 0.0))                   # 目标在正前，FORWARD；last_turn 不变
        self.assertEqual((a, p.last_turn), (FORWARD_SLOW, YAW_LEFT))
        p.target_fn = lambda x, y: (0.0, -5.0)
        p(None, None, State(0.0, 0.0, 0.0))
        self.assertEqual(p.last_turn, YAW_RIGHT)

    def test_reset_clears_the_episode(self):
        occ = stub_occ(door=False)
        p = make_policy(occ, target=(6.0, 0.0))
        for i in range(5):
            p(None, None, State(2.2 + 0.1 * i, 0.0, 0.0))
        self.assertGreater(len(p.memory), 0)
        self.assertEqual(len(p.rows), 5)
        p.reset()
        self.assertEqual((p.memory, p.rows, p.step_idx, p.last_turn), ([], [], 0, None))
        self.assertEqual(sum(p.entry_counts.values()), 0)
        p(None, None, State(-2.0, 0.0, 0.0))                         # 墙在 5 m 外：本步视场里没有点
        self.assertEqual(p.rows[0]["step"], 0)
        self.assertEqual(p.rows[0]["mem_n"], 0)                       # 记忆里也没有上一集的点

    def test_memory_keeps_an_obstacle_after_it_leaves_the_view(self):
        occ = stub_occ(door=False)
        p = make_policy(occ, target=(6.0, 0.0))
        p(None, None, State(2.3, 0.0, 0.0))                          # 墙在 0.7 m 前，进记忆
        n0 = p.rows[0]["mem_n"]
        p(None, None, State(2.3, 0.0, 120.0))                        # 机头转开 120°，视场里没有墙了
        self.assertGreater(n0, 0)
        self.assertGreater(p.rows[1]["mem_n"], 0)                    # 记忆里仍有上一步的点（mem_steps 8 内）
        for i in range(2, 11):
            p(None, None, State(2.3, 0.0, 120.0))
        self.assertEqual(p.rows[-1]["mem_n"], 0)                     # 超过 mem_steps 全部忘掉

    def test_params_override_and_unknown_key_rejected(self):
        p = make_policy(params={"r_safe": 0.2, "d_look": 0.8, "yaw_step_deg": 33.3})
        self.assertEqual((p.params["r_safe"], p.params["d_look"], p.params["yaw_step_deg"]), (0.2, 0.8, 33.3))
        self.assertEqual(p.params["mem_steps"], rp.RULE_V1["mem_steps"])
        self.assertEqual(rp.RULE_V1["r_safe"], 0.35)                  # 没有改到规则模块的全局字典
        with self.assertRaises(ValueError):
            make_policy(params={"r_saf": 0.2})
        with self.assertRaises(ValueError):
            make_policy(source="lidar")

    def test_interface_has_no_goal_stop_and_no_guard_trace(self):
        p = make_policy()
        self.assertEqual(p.name, "rule_oracle")
        for attr in ("stop_radius", "trace", "prev"):                 # run_episode 靠这些属性判 goal-stop 教师；driver 靠 trace 判护栏
            self.assertFalse(hasattr(p, attr), attr)


class TestSources(unittest.TestCase):
    def free_depth(self, st):
        return np.zeros((240, 320), np.float32)                      # 全无效 = 深度说前方没东西

    def test_main_source_decides_and_both_are_logged(self):
        occ = stub_occ(door=False)
        st = State(2.3, 0.0, 0.0)                                    # 占据：0.7 m 外有墙；深度图：空
        depth = self.free_depth(st)
        po = make_policy(occ, target=(6.0, 0.0), source="occ")
        pd = make_policy(occ, target=(6.0, 0.0), source="depth")
        ao, _ = po(None, depth, st)
        ad, _ = pd(None, depth, st)
        self.assertIn(ao, (YAW_LEFT, YAW_RIGHT))                     # 占据看到墙：转开
        self.assertEqual(ad, FORWARD_SLOW)                           # 深度没看到：直行
        for pol in (po, pd):
            row = pol.rows[0]
            self.assertEqual(len(row["d_occ"]), 20)
            self.assertEqual(len(row["d_depth"]), 20)
            self.assertLess(min(row["d_occ"]), 1.0)
            self.assertEqual(min(row["d_depth"]), 4.0)
        self.assertEqual((po.rows[0]["src"], pd.rows[0]["src"]), ("occ", "depth"))

    def test_depth_source_without_a_depth_image_is_an_error(self):
        with self.assertRaises(ValueError):
            make_policy(source="depth")(None, None, State(0.0, 0.0, 0.0))

    def test_occ_source_without_depth_still_works(self):
        p = make_policy(stub_occ(door=False), target=(6.0, 0.0), source="occ")
        p(None, None, State(0.0, 0.0, 0.0))
        self.assertIsNone(p.rows[0]["d_depth"])

    def test_depth_and_occ_agree_on_a_consistent_scene(self):
        # 同一面墙既进占据（体素墙面 x=3.0）也进深度图（x=3.0 平面）：未膨胀时两个来源的列距离相差 < 1 cm 量级
        occ = stub_occ(door=False, dilate=False)
        st = State(0.4, 0.2, 0.0)
        depth = synth_depth(st.camera_viewmat(None), walls=[wall_x(3.0)])
        d_occ = ro.occ_columns(occ, st.x, st.y, st.z, 0.0)
        d_dep = depth_cols(depth, st)
        self.assertLess(float(np.abs(d_occ - d_dep).max()), 0.02)


class TestClosedLoop(unittest.TestCase):
    """最小运动学：FORWARD 前进 0.3 m，YAW_LEFT/RIGHT 转 ±33°（3DGS hold 1.0 的实测量级）。符号错了会越转越远。"""

    def run_loop(self, start_yaw, target, steps=40, occ=None):
        p = make_policy(occ, target=target, params={"yaw_step_deg": 33.0})
        x, y, yaw = 0.0, 0.0, start_yaw
        for _ in range(steps):
            a, _ = p(None, None, State(x, y, yaw))
            if a == YAW_LEFT:
                yaw += 33.0
            elif a == YAW_RIGHT:
                yaw -= 33.0
            else:
                x += 0.3 * math.cos(math.radians(yaw))
                y += 0.3 * math.sin(math.radians(yaw))
            if math.hypot(x - target[0], y - target[1]) < 0.8:
                return True, p
        return False, p

    def test_turns_around_and_reaches_a_target_behind(self):
        for yaw0 in (180.0, 135.0, -135.0, 90.0, -90.0):
            with self.subTest(yaw0=yaw0):
                ok, p = self.run_loop(yaw0, (6.0, 1.0))
                self.assertTrue(ok, [r["action"] for r in p.rows])


if __name__ == "__main__":
    unittest.main(verbosity=2)
