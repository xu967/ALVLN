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

    # L1 goal-reach：只要求无碰撞 + 末态在 REACH_RADIUS(1.5) 内，末动作不参与判定。
    wp = {"x": 2.0, "y": 1.0}
    oracle_stop = [{"pose_gt": {"x": 2.1, "y": 1.1}}]
    for last in (0, 1):  # 末动作 SAFE 或 FORWARD，只要到了就算成功
        s = ec.summarize_episode([{"collision": False, "action_executed": last}],
                                 oracle_stop, 6.0, wp, stop_radius=0.5)
        assert s["success"] and s["final_waypoint_dist"] < 0.2, (last, s)
    s = ec.summarize_episode([{"collision": False, "action_executed": 0}],
                             [{"pose_gt": {"x": 5.0, "y": 1.0}}], 6.0, wp,
                             stop_radius=0.5)
    assert not s["success"] and s["final_waypoint_dist"] > 1.5, s
    # 半径边界：1.4 m 算到，1.6 m 不算（REACH_RADIUS=1.5）
    for dy, want in ((1.4, True), (1.6, False)):
        s = ec.summarize_episode([{"collision": False, "action_executed": 0}],
                                 [{"pose_gt": {"x": 2.0, "y": 1.0 + dy}}], 6.0, wp,
                                 stop_radius=0.5)
        assert s["success"] is want, (dy, s)
    # 到位但撞了柱：不算成功
    s = ec.summarize_episode([{"collision": True, "src": "contact", "action_executed": 0}],
                             oracle_stop, 6.0, wp, stop_radius=0.5)
    assert not s["success"] and not s["stop_success"], s

    # 2026-11-02 口径：success 看「轨迹上曾进入 1.5 m」，stop_success 看「末位姿」。
    # 到过 0.1 m 又飞走到 5 m：算 success，不算 stop_success。
    flyby = [{"collision": False, "action_executed": 0},
             {"collision": False, "action_executed": 1}]
    oracle_flyby = [{"pose_gt": {"x": 2.1, "y": 1.1}}, {"pose_gt": {"x": 7.0, "y": 1.0}}]
    s = ec.summarize_episode(flyby, oracle_flyby, 6.0, wp, stop_radius=0.5)
    assert s["success"] and not s["stop_success"], s
    assert s["waypoint_min_dist"] < 0.2 and s["final_waypoint_dist"] >= 5.0, s

    # 全程在 1.5 m 外：两个口径都不算成功
    s = ec.summarize_episode(flyby, [{"pose_gt": {"x": 5.0, "y": 1.0}},
                                     {"pose_gt": {"x": 5.5, "y": 1.0}}],
                             6.0, wp, stop_radius=0.5)
    assert not s["success"] and not s["stop_success"], s

    # 经过时撞柱：即使进入过 1.5 m 也不算成功（两个口径都要无碰撞）
    s = ec.summarize_episode(
        [{"collision": True, "src": "contact", "action_executed": 1}],
        [{"pose_gt": {"x": 2.1, "y": 1.1}}], 6.0, wp, stop_radius=0.5)
    assert not s["success"] and not s["stop_success"], s

    print("test_eval_summary: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
