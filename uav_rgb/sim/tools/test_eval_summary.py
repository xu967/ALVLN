#!/usr/bin/env python3
"""eval_closed_loop.summarize_episode 自检：contact/stuck 口径（EVAL_STANDARD §5.1）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_closed_loop as ec


def run(rows, x=6.0, goal_x=5.0):
    return ec.summarize_episode(rows, [{"pose_gt": {"x": x}}], goal_x)


def main():
    # contact：算碰撞、算 false_forward，即使过了线也不算 success
    s = run([{"collision": True, "src": "contact", "action_executed": 1, "infer_ms": 1.0}])
    assert s["collision"] and not s["stuck"] and s["false_forward"] and not s["success"], s

    # stuck-only：不是碰撞，也不进 false_forward（过线仍算 success）
    s = run([{"collision": True, "src": "stuck", "action_executed": 1}])
    assert not s["collision"] and s["stuck"] and not s["false_forward"] and s["success"], s

    # contact+stuck 同一步：按 contact 算
    s = run([{"collision": True, "src": "contact+stuck", "action_executed": 1}])
    assert s["collision"] and s["stuck"] and s["false_forward"] and not s["success"], s

    # 旧 episode 没有 src 字段：保持旧口径（collision 就是碰撞）
    s = run([{"collision": True, "action_executed": 1}])
    assert s["collision"] and not s["stuck"] and s["false_forward"] and not s["success"], s

    # 无碰撞且过线 = success
    s = run([{"collision": False, "action_executed": 1}])
    assert s["success"] and not s["collision"] and not s["stuck"], s

    # L1：经过 waypoint 才算 success；没经过即使过线也不算
    rows = [{"collision": False, "action_executed": 1}]
    oracle = [{"pose_gt": {"x": 6.0, "y": 0.0}}]
    s = ec.summarize_episode(rows, oracle, 5.0, {"x": 6.0, "y": 0.0})
    assert s["success"] and s["waypoint_min_dist"] == 0.0, s
    s = ec.summarize_episode(rows, oracle, 5.0, {"x": 6.0, "y": 2.0})
    assert not s["success"] and s["waypoint_min_dist"] == 2.0, s

    # 空集：字段齐全，不计成功
    s = ec.summarize_episode([], [], 5.0)
    assert s["steps"] == 0 and "stuck" in s and not s["success"], s

    print("test_eval_summary: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
