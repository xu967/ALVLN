#!/usr/bin/env python3
"""rule_policy.py 的纯函数测试（计划 D3）：特权列距离 column_distances 与固定规则 rule_v0。

不起 Gazebo。坐标与角度约定与 teacher.py 一致（x 前、y 左；yaw 度，0 = +x，正 = 左转；方位正 = 左），
这里用 teacher.visible_pillar_side / ray_clear 做交叉核对，约定漂移会在这里红。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_rule_policy.py
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher
from rule_policy import (COL_DEG, RULE_V0, SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT,
                         column_distances, rule_v0, rule_v0_entry)

MAXR = 4.0
CAM = 0.045          # e1_mini.json camera_mount.xyz[0]


def square(x, y, size=0.4):
    return {"x": x, "y": y, "z": 1.5, "size": [size, size, 3.0]}


def circle(x, y, r=0.2):
    return {"x": x, "y": y, "z": 1.5, "radius": r, "size": [2 * r, 2 * r, 3.0]}   # scene_forest 同时落外接方框


def cols(pillars, x=0.0, y=0.0, yaw=0.0, **kw):
    return column_distances(x, y, yaw, pillars, **kw)


def open_field(k=20):
    return [MAXR] * k


class ColumnDistanceTests(unittest.TestCase):
    def test_empty_scene_all_max_range_and_rule_goes_forward(self):
        d = cols([])
        self.assertEqual(d, open_field())
        self.assertEqual(rule_v0(d, 0, None), FORWARD_SLOW)

    def test_pillar_straight_ahead_1m(self):
        # 柱面离机体中心 1.0 m（中心 1.2），相机在机体前 0.045 m：正前方射线 0.955 m
        d = cols([square(1.2, 0.0)])
        self.assertEqual(len(d), 20)
        self.assertTrue(0.8 <= d[9] <= 1.0 and 0.8 <= d[10] <= 1.0, d[9:11])
        self.assertAlmostEqual(d[9], 1.0 - CAM, places=9)      # 列 9 的 0° 边缘射线正好打在柱面中线
        self.assertAlmostEqual(d[10], 1.0 - CAM, places=9)
        self.assertTrue(all(v == MAXR for v in d[:5] + d[15:]), "柱面半角约 11.8°，|方位| ≥ 15° 的列看不到它")

    def test_pillar_on_the_left_shortens_only_left_columns(self):
        d = cols([square(1.5, 0.8)])        # y > 0 = 左；正面与朝向相机的侧面对相机方位约 +20°…+38°
        self.assertTrue(d[0] < MAXR and all(v < MAXR for v in d[:4]), d[:5])
        self.assertTrue(all(v == MAXR for v in d[10:]), "右半边一列都不该变短")
        mirrored = cols([square(1.5, -0.8)])
        self.assertEqual(mirrored, d[::-1], "镜像场景的列距离左右对调")

    def test_yaw_left_moves_world_forward_pillar_to_the_right_side(self):
        # 机头左转 20° 后，世界 +x 方向的柱子在机体右侧：锁定 yaw 正 = 左转（teacher._rollout：YAW_LEFT → yaw + 27°）
        d = cols([square(1.5, 0.0)], yaw=20.0)
        self.assertTrue(all(v == MAXR for v in d[:10]), d)
        self.assertTrue(d[16] < 2.0 and d[17] < 2.0, d[13:])
        d_right = cols([square(1.5, 0.0)], yaw=-20.0)
        self.assertEqual(d_right, d[::-1])

    def test_camera_is_cam_forward_ahead_along_heading(self):
        d_off = cols([square(1.2, 0.0)], cam_forward=0.0)
        self.assertAlmostEqual(d_off[9], 1.0, places=9)
        # yaw = 90°：机头朝 +y，柱子放在 +y 方向 1.2 m，相机前移沿着机头，也是 0.955
        d90 = cols([square(0.0, 1.2)], yaw=90.0)
        self.assertAlmostEqual(d90[9], 1.0 - CAM, places=9)

    def test_square_pillar_uses_its_box(self):
        d = cols([square(1.2, 0.0)], col_deg=1.5)       # 40 列：列 12 的扇区 [10.5°, 12°] 只被方柱的角部区域碰到
        self.assertTrue(0.95 < d[12] < 1.0, d[12])

    def test_round_pillar_is_a_circle_not_its_bounding_box(self):
        c = cols([circle(1.2, 0.0)])
        self.assertAlmostEqual(c[9], 1.2 - 0.2 - CAM, places=9)
        self.assertAlmostEqual(c[10], 1.2 - 0.2 - CAM, places=9)
        # 同一位置同一扇区：圆柱在 ±9.97° 以外没有，方框的角部还在
        c_fine, s_fine = cols([circle(1.2, 0.0)], col_deg=1.5), cols([square(1.2, 0.0)], col_deg=1.5)
        self.assertEqual(c_fine[12], MAXR)
        self.assertLess(s_fine[12], MAXR)
        # 圆在斜向比方柱远：列 6（[9°, 12°]）里方柱先碰到前面，圆要到更远处才碰到
        self.assertGreater(c[6], cols([square(1.2, 0.0)])[6] + 0.05)

    def test_nearest_of_several_pillars_and_beyond_max_range(self):
        d = cols([square(3.0, 0.0), square(1.8, 0.0)])
        self.assertAlmostEqual(d[9], 1.6 - CAM, places=9)
        far = cols([square(6.0, 0.0)])
        self.assertEqual(far, open_field(), "超过 max_range 记 max_range")
        self.assertAlmostEqual(cols([square(3.0, 0.0)], max_range=2.5)[9], 2.5)

    def test_origin_inside_pillar_is_zero_and_pillar_behind_is_ignored(self):
        self.assertEqual(cols([square(0.0, 0.0)])[9], 0.0)
        self.assertEqual(cols([square(-1.5, 0.0)]), open_field())

    def test_corridor_boundary_is_not_a_solid(self):
        self.assertEqual(cols([], y=2.9, yaw=80.0), open_field(), "|y| = 3 m 只是判据，不是实体")

    def test_wider_view_has_27_columns_and_sees_more(self):
        p = square(1.2, 1.2)                # 对相机方位 +36.4°…+55.7°：60° 视场（±30°）看不到，81° 视场（±40.5°）最左两列看得到
        wide = cols([p], hfov_deg=81.0)
        self.assertEqual(len(wide), 27)
        self.assertEqual(len(cols([], hfov_deg=60.0)), 20)
        self.assertEqual(len(cols([], hfov_deg=60.0, col_deg=1.5)), 40)
        self.assertEqual(cols([p]), open_field())
        self.assertTrue(wide[0] < MAXR and wide[1] < MAXR and all(v == MAXR for v in wide[2:]), wide)
        self.assertAlmostEqual(cols([square(1.2, 0.0)], hfov_deg=81.0)[13], 1.0 - CAM, places=9)   # 中间列方位 0°

    def test_at_least_five_rays_per_column(self):
        with self.assertRaises(ValueError):
            cols([], n_rays=4)

    def test_default_column_width_is_three_degrees(self):
        self.assertEqual(COL_DEG, 3.0)


class ConventionMatchesTeacherTests(unittest.TestCase):
    """与 teacher.py 同一约定：visible_pillar_side 说柱在左（+1），最近那一列就在左半边。"""

    def test_side_agrees_with_visible_pillar_side(self):
        for yaw in (-40.0, 0.0, 25.0, 100.0, -170.0):
            for bearing in (-25.0, -10.0, 8.0, 22.0):
                th = math.radians(yaw + bearing)
                p = square(1.6 * math.cos(th), 1.6 * math.sin(th))      # 机体系方位 = bearing、距离 1.6 m
                side = teacher.visible_pillar_side(0.0, 0.0, yaw, [p])
                d = cols([p], yaw=yaw)
                nearest = d.index(min(d))
                self.assertEqual(side, 1 if bearing > 0 else -1, (yaw, bearing))
                self.assertEqual(nearest < 10, side == 1, (yaw, bearing, nearest))

    def test_ray_through_pillar_centre_is_blocked_in_teacher_ray_clear(self):
        for yaw in (0.0, 30.0, -75.0):
            for bearing in (-24.0, 0.0, 13.0):
                th = math.radians(yaw + bearing)
                ox, oy = CAM * math.cos(math.radians(yaw)), CAM * math.sin(math.radians(yaw))
                p = square(ox + 1.5 * math.cos(th), oy + 1.5 * math.sin(th))
                # clearance 取 1 cm：_pillar_blocked 用严格小于，0 余量永远不堵
                self.assertFalse(teacher.ray_clear(ox, oy, yaw + bearing, 3.0, [p], 0.01), (yaw, bearing))
                d = cols([p], yaw=yaw)
                j = round(len(d) / 2 - 0.5 - bearing / COL_DEG)            # 列中心方位 = bearing 的那一列
                self.assertLess(d[j], 1.5, (yaw, bearing, j))


class RuleV0Tests(unittest.TestCase):
    def test_params_are_the_preregistered_values(self):
        self.assertEqual(RULE_V0, {"r_safe": 0.35, "d_look": 1.2, "n_max": 2, "side_near": 0.5,
                                   "tie_frac": 0.10, "sum_clip": 4.0})

    def test_blocked_ahead_turns_toward_the_open_side(self):
        d = cols([square(1.2, 0.0), square(1.6, -0.7)])           # 正前方被挡，右侧也有柱 → 向左
        self.assertEqual(rule_v0(d, 0, None), YAW_LEFT)
        d = cols([square(1.2, 0.0), square(1.6, 0.7)])
        self.assertEqual(rule_v0(d, 0, None), YAW_RIGHT)

    def test_blocked_by_decision_uses_the_channel_not_just_the_centre_column(self):
        self.assertEqual(rule_v0(cols([square(1.2, 0.0)]), 0, None), YAW_LEFT)      # 0.955 m < d_look，且 n=0 平局 → 固定左
        self.assertEqual(rule_v0(cols([square(1.8, 0.0)]), 0, None), FORWARD_SLOW)  # 柱面 1.6 m ≥ d_look
        # 柱面 1.0 m，近侧边缘离机体轴 0.3 m < r_safe：通道里有障碍
        self.assertEqual(rule_v0(cols([square(1.2, 0.5)]), 0, None), YAW_RIGHT)
        # 近侧边缘离机体轴 0.4 m > r_safe：从旁边过得去
        self.assertEqual(rule_v0(cols([square(1.2, 0.6)]), 0, None), FORWARD_SLOW)
        self.assertEqual(rule_v0(cols([square(1.2, -0.6)]), 0, None), FORWARD_SLOW)

    def test_net_turn_limit_turns_back(self):
        d = open_field()
        self.assertEqual(rule_v0(d, 2, None), YAW_RIGHT)           # n > 0：已左偏 → 右转回来
        self.assertEqual(rule_v0(d, -2, None), YAW_LEFT)
        self.assertEqual(rule_v0(d, 3, YAW_LEFT), YAW_RIGHT)
        self.assertEqual(rule_v0(d, 1, None), FORWARD_SLOW)
        self.assertEqual(rule_v0(d, -1, None), FORWARD_SLOW)

    def test_turn_back_is_withheld_when_the_turn_back_side_edge_is_close(self):
        near_right = open_field()
        near_right[-2] = 0.45                                       # 回转一侧 = 右（n > 0）：最外 2 列里有 < 0.5
        self.assertEqual(rule_v0(near_right, 2, None), SAFE_NEUTRAL, "那一列同时在通道里 → blocked → SAFE")
        self.assertEqual(rule_v0(near_right, -2, None), YAW_LEFT, "n < 0 回转一侧是左，右侧近不影响例外")
        near_left = open_field()
        near_left[1] = 0.45
        self.assertEqual(rule_v0(near_left, -2, None), SAFE_NEUTRAL)
        self.assertEqual(rule_v0(near_left, 2, None), YAW_RIGHT, "n > 0 回转一侧是右，左侧近不影响")
        outer_ok = open_field()
        outer_ok[-3] = 0.2                                          # 第 3 列不算最外 2 列
        self.assertEqual(rule_v0(outer_ok, 2, None), YAW_RIGHT)

    def test_r1_exception_forward_branch_needs_a_smaller_r_safe(self):
        # 记录规则 v0 的一个性质：回转侧最外 2 列（方位 ≥ 25.5°）里 d < 0.5 时横向偏移 d·sin ≤ 0.22 < r_safe = 0.35，
        # 同时一定 blocked，所以默认参数下例外只会出 SAFE_NEUTRAL，「不 blocked → 前进」那一支到不了（见报告）。
        # 把 r_safe 调小才能走到那一支，这里用它锁住分支本身的行为。
        near_right = open_field()
        near_right[-2] = 0.45
        small = dict(RULE_V0, r_safe=0.1)
        self.assertEqual(rule_v0_entry(near_right, 2, None, small), (FORWARD_SLOW, "R1-fwd"))
        self.assertEqual(rule_v0_entry(near_right, 2, None), (SAFE_NEUTRAL, "R1-safe"))
        blocked = cols([square(1.2, 0.0)])
        blocked[-1] = 0.3
        self.assertEqual(rule_v0_entry(blocked, 2, None, small), (SAFE_NEUTRAL, "R1-safe"))

    def test_net_turn_limit_comes_before_blocked_turn(self):
        d = cols([square(1.2, 0.0), square(1.6, 0.7)])              # 单看 blocked 会向右；n = −2 要求向左
        self.assertEqual(rule_v0(d, 0, None), YAW_RIGHT)
        self.assertEqual(rule_v0(d, -2, None), YAW_LEFT)

    def test_tie_follows_last_turn_then_zeroes_n_then_left(self):
        d = cols([square(1.2, 0.0)])                                # 左右对称：平局
        self.assertEqual(rule_v0(d, 0, YAW_RIGHT), YAW_RIGHT)
        self.assertEqual(rule_v0(d, 0, YAW_LEFT), YAW_LEFT)
        self.assertEqual(rule_v0(d, 1, YAW_LEFT), YAW_LEFT, "last_turn 优先于「让 n 回零」")
        self.assertEqual(rule_v0(d, 1, None), YAW_RIGHT)
        self.assertEqual(rule_v0(d, -1, None), YAW_LEFT)
        self.assertEqual(rule_v0(d, 0, None), YAW_LEFT)

    def test_near_tie_is_ten_percent_of_the_larger_side(self):
        d = [1.0] * 20
        d[9] = 0.5                                                  # 正前方 blocked
        right_95 = list(d)
        right_95[10:] = [1.0 - 0.05 / 10] * 10                      # 右半边总和比左半边少 5%（含 d[9] 的 0.5 在内再算）
        left, right = sum(d[:10]), sum(right_95[10:])
        self.assertLess(abs(left - right), 0.10 * max(left, right))
        self.assertEqual(rule_v0(right_95, 0, YAW_RIGHT), YAW_RIGHT, "差不到 10%：沿用 last_turn")
        wide = list(d)
        wide[10:] = [0.5] * 10                                      # 右半边总和只有左半边的 5/9.5：差远超 10%
        self.assertEqual(rule_v0(wide, 0, YAW_RIGHT), YAW_LEFT, "差够大：向开阔一侧，不看 last_turn")

    def test_sum_is_clipped_at_sum_clip(self):
        d = [MAXR] * 20
        d[9] = 0.6                                                  # blocked
        d[:9] = [9.0] * 9                                           # 左边看起来远得离谱，裁到 4.0 后与右边相同 → 平局
        self.assertEqual(rule_v0(d, 0, YAW_RIGHT), YAW_RIGHT)

    def test_entry_names(self):
        free, ahead = open_field(), cols([square(1.2, 0.0)])
        near_right = open_field()
        near_right[-1] = 0.3
        self.assertEqual(rule_v0_entry(free, 0, None), (FORWARD_SLOW, "R3"))
        self.assertEqual(rule_v0_entry(free, 2, None), (YAW_RIGHT, "R1"))
        self.assertEqual(rule_v0_entry(near_right, 2, None, dict(RULE_V0, r_safe=0.1)), (FORWARD_SLOW, "R1-fwd"))
        self.assertEqual(rule_v0_entry(near_right, 2, None), (SAFE_NEUTRAL, "R1-safe"))
        self.assertEqual(rule_v0_entry(cols([square(1.2, 0.0), square(1.6, 0.7)]), 0, None), (YAW_RIGHT, "R2-sum"))
        self.assertEqual(rule_v0_entry(ahead, 0, YAW_RIGHT), (YAW_RIGHT, "R2-last"))
        self.assertEqual(rule_v0_entry(ahead, 1, None), (YAW_RIGHT, "R2-n"))
        self.assertEqual(rule_v0_entry(ahead, 0, None), (YAW_LEFT, "R2-left"))

    def test_rule_works_for_the_27_column_view(self):
        d = cols([square(1.2, 0.0), square(1.6, 0.7)], hfov_deg=81.0)
        self.assertEqual(len(d), 27)
        self.assertEqual(rule_v0(d, 0, None), YAW_RIGHT)
        self.assertEqual(rule_v0(open_field(27), 0, None), FORWARD_SLOW)
        mirror = cols([square(1.2, 0.0), square(1.6, -0.7)], hfov_deg=81.0)
        self.assertEqual(mirror, d[::-1])
        self.assertEqual(rule_v0(mirror, 0, None), YAW_LEFT, "奇数列：正中间那一列不属于任何一侧，镜像仍对称")


if __name__ == "__main__":
    unittest.main()
