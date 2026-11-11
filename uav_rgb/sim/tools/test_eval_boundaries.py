#!/usr/bin/env python3
"""eval_closed_loop.summarize_episode 的判据边界（交叉审核 F8，2026-11-12）。

位姿是连续量，恰好落在边界上的概率低，但判据的方向由等号决定，改错一个比较符号时
test_eval_metrics / summary / requested_set / sim09v2 / provenance 全部不会红。
这里各锁一个用例，值都取二进制精确可表示的数，边界两侧用相邻浮点数：
  过线：max_x >= goal_x，max_x 恰好等于 goal_x 算过线；差一个浮点步长不算。
  走廊：y_max > half_width，|y| 恰好等于半宽不算越界；多一个浮点步长算。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec

GOAL_X = 4.0
HALF_WIDTH = 3.0
ROWS = [{"collision": False, "action_executed": 1}] * 3


def episode(xs_ys, corridor_half_width=None):
    """无碰撞、无 waypoint 的验收线任务（route_choice 口径）；xs_ys 是 oracle 的 (x, y) 序列。"""
    oracle = [{"pose_gt": {"x": x, "y": y}} for x, y in xs_ys]
    return ec.summarize_episode(ROWS[:len(oracle)], oracle, GOAL_X, corridor_half_width=corridor_half_width)


def test_goal_line_equality():
    on_line = episode([(0.0, 0.0), (2.0, 0.0), (GOAL_X, 0.0)])
    assert on_line["max_x"] == GOAL_X and on_line["success"] is True, on_line
    just_short = math.nextafter(GOAL_X, 0.0)
    short = episode([(0.0, 0.0), (2.0, 0.0), (just_short, 0.0)])
    assert short["max_x"] == just_short and short["success"] is False, short
    just_past = math.nextafter(GOAL_X, 8.0)
    past = episode([(0.0, 0.0), (2.0, 0.0), (just_past, 0.0)])
    assert past["success"] is True, past


def test_corridor_equality():
    # |y| 恰好等于半宽：不越界，过线则成功；左右两侧都一样。
    for y in (HALF_WIDTH, -HALF_WIDTH):
        on_edge = episode([(0.0, 0.0), (2.0, y), (4.6, 0.0)], corridor_half_width=HALF_WIDTH)
        assert on_edge["y_max_m"] == HALF_WIDTH, on_edge
        assert on_edge["corridor_violation"] is False and on_edge["success"] is True, on_edge
    # 多一个浮点步长：越界，成功被否决（即使过了线）。
    over = math.nextafter(HALF_WIDTH, 8.0)
    for y in (over, -over):
        out = episode([(0.0, 0.0), (2.0, y), (4.6, 0.0)], corridor_half_width=HALF_WIDTH)
        assert out["y_max_m"] == over, out
        assert out["corridor_violation"] is True and out["success"] is False, out


def test_both_equalities_at_once():
    both = episode([(0.0, 0.0), (2.0, HALF_WIDTH), (GOAL_X, 0.0)], corridor_half_width=HALF_WIDTH)
    assert both["success"] is True and both["corridor_violation"] is False, both


def main():
    test_goal_line_equality()
    test_corridor_equality()
    test_both_equalities_at_once()
    print("test_eval_boundaries: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
